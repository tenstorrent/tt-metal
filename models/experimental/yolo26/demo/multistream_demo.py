# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO26 multi-stream people counting on one chip.

N video streams (default 20) are paced at a target rate (default 30 FPS each). Every tick, the latest frame of each
stream is letterboxed and grouped into batches of --batch-size images; each batch runs through the traced model with
2 command queues (upload of the next batch on CQ1 overlaps the current trace on CQ0, outputs are read back on CQ1).
Persons (COCO class 0) are counted per stream and drawn on a mosaic with a side panel of per-stream counts.

The mosaic is served live as MJPEG on http://<host>:--port/ and optionally recorded to --record.

    python models/experimental/yolo26/demo/multistream_demo.py --videos <dir> --variant yolo26s --batch-size 4
"""

import argparse
import glob
import http.server
import os
import queue
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
import ttnn
from loguru import logger

from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE
from models.experimental.yolo26.runner.performant_runner import YOLO26PerformantRunner
from models.experimental.yolo26.tt.ttnn_yolo26 import UINT8_INPUT_CHANNELS

STRIDES = [8, 16, 32]
PERSON = 0
PANEL_W = 420


def load_clips(video_dir, num_streams, max_frames):
    """Decode clips into memory (BGR). Streams beyond the number of clips reuse clips at a different start frame."""
    paths = sorted(glob.glob(os.path.join(video_dir, "*.mpg")) + glob.glob(os.path.join(video_dir, "*.mp4")))
    if not paths:
        raise FileNotFoundError(f"no .mpg / .mp4 videos in {video_dir}")
    clips = []
    for path in paths[:num_streams]:
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if frames:
            # Model input copy: RGB + a zero 4th byte, so a pixel is one uint32 (see Letterbox.fill)
            rgba = [cv2.cvtColor(f, cv2.COLOR_BGR2RGBA) for f in frames]
            for f in rgba:
                f[..., 3] = 0
            clips.append((os.path.splitext(os.path.basename(path))[0], frames, rgba))
            logger.info(f"loaded {path}: {len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]}")
    streams = []
    for i in range(num_streams):
        name, frames, rgba = clips[i % len(clips)]
        streams.append({"name": name, "frames": frames, "rgba": rgba, "offset": (i // len(clips)) * len(frames) // 2})
    return streams


class Letterbox:
    """Ultralytics LetterBox (pad 114) of a fixed source size into the model's uint8 input row layout."""

    def __init__(self, src_h, src_w, size):
        self.scale = min(size / src_h, size / src_w)
        self.new_h, self.new_w = round(src_h * self.scale), round(src_w * self.scale)
        self.pad_h, self.pad_w = (size - self.new_h) // 2, (size - self.new_w) // 2
        self.size = size

    def fill(self, dst, frame_rgba):
        """
        dst: [H, W, UINT8_INPUT_CHANNELS] uint8 view, pre-filled with padding. Writes the resized RGB0 pixels into
        channels 0..3 as one uint32 per pixel (4x fewer strided writes than per-channel assignment).
        """
        resized = cv2.resize(frame_rgba, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        dst32 = dst.view(np.uint32)
        dst32[self.pad_h : self.pad_h + self.new_h, self.pad_w : self.pad_w + self.new_w, 0] = resized.view(np.uint32)[
            ..., 0
        ]

    def to_source(self, xyxy):
        """Input-image pixels -> source pixels."""
        xyxy = xyxy.copy()
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - self.pad_w) / self.scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - self.pad_h) / self.scale
        return xyxy


def anchor_grid(size):
    anchors, strides = [], []
    for stride in STRIDES:
        n = size // stride
        sy, sx = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
        anchors.append(np.stack((sx, sy), -1).reshape(-1, 2))
        strides.append(np.full((n * n, 1), float(stride)))
    return np.concatenate(anchors).astype(np.float32), np.concatenate(strides).astype(np.float32)


def decode_people(host_outputs, shapes, batch, anchors, strides, conf):
    """
    host_outputs: [box_p3, box_p4, box_p5, cls_p3, cls_p4, cls_p5] host tile tensors.
    Returns per image an array [k, 5] of (x1, y1, x2, y2, conf) in input-image pixels for class 0 (person).
    NMS-free one2one head: boxes are ltrb distances in units of the stride, scores are logits.
    """
    logit = np.log(conf / (1 - conf))
    boxes, scores = [], []
    for t, (h, w) in zip(host_outputs[:3], shapes):
        boxes.append(ttnn.to_torch(t).reshape(batch, h * w, -1)[..., :4].float().numpy())
    for t, (h, w) in zip(host_outputs[3:], shapes):
        scores.append(ttnn.to_torch(t).reshape(batch, h * w, -1)[..., PERSON].float().numpy())
    boxes, scores = np.concatenate(boxes, 1), np.concatenate(scores, 1)
    people = []
    for b in range(batch):
        keep = np.nonzero(scores[b] > logit)[0]
        ltrb, a, s = boxes[b, keep], anchors[keep], strides[keep]
        xyxy = np.concatenate((a - ltrb[:, :2], a + ltrb[:, 2:]), 1) * s
        p = 1 / (1 + np.exp(-scores[b, keep]))
        people.append(np.concatenate((xyxy, p[:, None]), 1))
    return people


class Mjpeg:
    """Latest JPEG frame served as multipart MJPEG at / (HTML) and /stream."""

    def __init__(self, port):
        self.jpeg = None
        self.cond = threading.Condition()
        mjpeg = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/":
                    body = b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body></html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                try:
                    while True:
                        with mjpeg.cond:
                            mjpeg.cond.wait_for(lambda: mjpeg.jpeg is not last, timeout=5)
                            last = mjpeg.jpeg
                        if last is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(last)}\r\n\r\n".encode() + last + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, jpeg):
        with self.cond:
            self.jpeg = jpeg
            self.cond.notify_all()


class Recorder:
    """H.264 recording in a separate ffmpeg process fed by a thread, so encoding stays off the Python lock."""

    def __init__(self, path, width, height, fps):
        # Low priority and 4 encoder threads: leave the other cores to the inference pipeline
        cmd = ["nice", "-n", "10", "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24"]
        cmd += ["-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast"]
        cmd += ["-crf", "23", "-threads", "4"]
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.q = queue.Queue(maxsize=60)
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while (frame := self.q.get()) is not None:
            self.proc.stdin.write(frame)

    def write(self, frame):
        try:
            self.q.put_nowait(frame.tobytes())
        except queue.Full:
            self.dropped += 1

    def close(self):
        self.q.put(None)
        self.thread.join()
        self.proc.stdin.close()
        self.proc.wait()
        if self.dropped:
            logger.warning(f"recorder dropped {self.dropped} frames")


class Display:
    """Per-stream annotated tiles, a side panel of people counts and throughput, composed into one mosaic."""

    def __init__(self, streams, cols, tile_w, tile_h, args):
        self.streams, self.cols, self.tile_w, self.tile_h, self.args = streams, cols, tile_w, tile_h, args
        rows = (len(streams) + cols - 1) // cols
        self.canvas = np.zeros((rows * tile_h, cols * tile_w + PANEL_W, 3), np.uint8)

    def tile(self, i, frame, people_src, fps):
        r, c = divmod(i, self.cols)
        sx, sy = self.tile_w / frame.shape[1], self.tile_h / frame.shape[0]
        view = self.canvas[r * self.tile_h : (r + 1) * self.tile_h, c * self.tile_w : (c + 1) * self.tile_w]
        cv2.resize(frame, (self.tile_w, self.tile_h), dst=view, interpolation=cv2.INTER_AREA)
        for x1, y1, x2, y2, p in people_src:
            cv2.rectangle(view, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (60, 220, 60), 2)
        label = f"CAM {i + 1:02d}  people: {len(people_src)}"
        cv2.rectangle(view, (0, 0), (self.tile_w, 22), (0, 0, 0), -1)
        cv2.putText(view, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        color = (90, 230, 90)
        text = f"{fps:4.1f} FPS"
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(view, text, (self.tile_w - w - 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.rectangle(view, (0, 0), (self.tile_w - 1, self.tile_h - 1), (40, 40, 40), 1)

    def panel(self, stats, counts, averages):
        x0 = self.canvas.shape[1] - PANEL_W
        p = self.canvas[:, x0:]
        p[:] = (24, 24, 24)
        put = lambda text, y, scale=0.55, color=(230, 230, 230), thick=1: cv2.putText(
            p, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
        )
        put(f"YOLO26{self.args.variant[-1]} people counting", 32, 0.7, (255, 255, 255), 2)
        put(f"1 Tenstorrent Blackhole chip, batch {self.args.batch_size}", 58)
        put(f"{len(self.streams)} streams x {stats['stream_fps']:.1f} FPS", 96, 0.75, (90, 230, 90), 2)
        put(f"= {stats['total_fps']:.0f} images/s", 124)
        put(f"latency {stats['latency_ms']:.0f} ms (median)   display {stats['display_fps']:.0f} FPS", 148)
        total = sum(counts)
        put(f"People now: {total}", 186, 0.75, (255, 255, 255), 2)
        top = 214
        row_h = max(18, (p.shape[0] - top - 10) // len(self.streams))
        put("camera          people now      avg 10 s", top - 6, 0.42, (150, 150, 150))
        bar_x0, bar_x1 = 90, PANEL_W - 130
        scale = max(4, max(counts))
        text = lambda s, x, y, color=(255, 255, 255): cv2.putText(
            p, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
        for i, (n, avg) in enumerate(zip(counts, averages)):
            y = top + 4 + i * row_h
            text(f"CAM {i + 1:02d}", 16, y + 13, (210, 210, 210))
            cv2.rectangle(p, (bar_x0, y + 2), (bar_x1, y + row_h - 4), (45, 45, 45), -1)
            cv2.rectangle(
                p, (bar_x0, y + 2), (bar_x0 + int((bar_x1 - bar_x0) * n / scale), y + row_h - 4), (60, 170, 60), -1
            )
            text(f"{n:2d}", PANEL_W - 116, y + 13)
            text(f"{avg:4.1f}", PANEL_W - 60, y + 13, (170, 170, 170))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="directory of .mpg / .mp4 clips")
    ap.add_argument("--variant", default="yolo26s")
    ap.add_argument("--streams", type=int, default=20)
    ap.add_argument("--fps", type=int, default=30, help="target FPS per stream")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--duration", type=float, default=60, help="seconds to run")
    ap.add_argument("--conf", type=float, default=0.35, help="person confidence threshold")
    ap.add_argument("--max-frames", type=int, default=1500, help="frames decoded per clip (loops)")
    ap.add_argument("--port", type=int, default=8090, help="MJPEG port, 0 = off")
    ap.add_argument("--record", default="", help="record the mosaic to this .mp4")
    ap.add_argument("--snapshot", default="", help="save the last mosaic frame to this .jpg")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tile", default="384x288")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()
    size = 640
    B = args.batch_size
    if args.streams % B:
        raise ValueError("--streams must be a multiple of --batch-size")
    nb = args.streams // B

    streams = load_clips(args.videos, args.streams, args.max_frames)
    src_h, src_w = streams[0]["frames"][0].shape[:2]
    letterbox = Letterbox(src_h, src_w, size)
    anchors, strides = anchor_grid(size)
    tile_w, tile_h = map(int, args.tile.split("x"))

    device = ttnn.open_device(
        device_id=args.device_id, l1_small_size=YOLO26_L1_SMALL_SIZE, trace_region_size=8000000, num_command_queues=2
    )
    runner = YOLO26PerformantRunner(device, B, size, args.variant)
    runner.capture_trace_2cq()
    outputs = runner.runner_infra.output_tensor
    out_list = [t for k in ("boxes", "scores") for t, _, _ in outputs[k]]
    shapes = [(h, w) for _, h, w in outputs["boxes"]]

    # Host input buffers: per batch slot, [B, H, W, 8] uint8 pre-filled with the letterbox padding (114 in RGB)
    pad = np.zeros((B, size, size, UINT8_INPUT_CHANNELS), np.uint8)
    pad[..., :3] = 114
    prep_q = queue.Queue(maxsize=2 * nb)
    post_q = queue.Queue()  # decode keeps up easily (~2 ms per batch); unbounded so the device loop never waits
    stop = threading.Event()
    t_start = [0.0]

    def prep():
        """Pace ticks at --fps; for each tick build nb host batches of the current frame of each stream."""
        period = 1.0 / args.fps
        tick = 0
        t0 = time.perf_counter()
        t_start[0] = t0
        while not stop.is_set():
            due = t0 + tick * period
            now = time.perf_counter()
            if due > now:
                time.sleep(due - now)
            captured = time.perf_counter()
            for b in range(nb):
                buf = pad.copy()
                ids = list(range(b * B, (b + 1) * B))
                frame_idx = []
                for j, s in enumerate(ids):
                    st = streams[s]
                    f = (tick + st["offset"]) % len(st["frames"])
                    letterbox.fill(buf[j], st["rgba"][f])
                    frame_idx.append(f)
                host = ttnn.from_torch(
                    torch.from_numpy(buf).reshape(B, 1, size, size * UINT8_INPUT_CHANNELS),
                    dtype=ttnn.uint8,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                prep_q.put((tick, b, ids, frame_idx, captured, host))
            tick += 1

    class Results:
        """Latest detections per stream, written for every batch; the renderer reads whatever is newest."""

        def __init__(self):
            self.lock = threading.Condition()
            self.latest = [None] * args.streams  # (frame_idx, dets) per stream
            self.history = [deque(maxlen=args.fps * 10) for _ in range(args.streams)]
            self.updates = [deque() for _ in range(args.streams)]  # result timestamps, last 1 s
            self.done_ticks, self.images = deque(), deque()
            self.latencies = deque(maxlen=200)
            self.tick = -1  # last tick with all streams done
            self.tick_captured = 0.0
            self.stats = {}

    results = Results()

    def post():
        """Decode every batch (cheap); never blocks the device loop on drawing."""
        pending = {}
        while (item := post_q.get()) is not None:
            tick, b, ids, frame_idx, captured, host_outputs = item
            people = decode_people(host_outputs, shapes, B, anchors, strides, args.conf)
            now = time.perf_counter()
            with results.lock:
                for j, s in enumerate(ids):
                    src = letterbox.to_source(people[j][:, :4]) if len(people[j]) else np.zeros((0, 4))
                    dets = np.concatenate((src, people[j][:, 4:]), 1) if len(people[j]) else []
                    results.latest[s] = (frame_idx[j], dets)
                    results.history[s].append(len(dets))
                    u = results.updates[s]
                    u.append(now)
                    while now - u[0] > 1.0:
                        u.popleft()
                results.images.append((now, B))
                pending[tick] = pending.get(tick, 0) + 1
                if pending[tick] < nb:
                    continue
                del pending[tick]
                results.done_ticks.append(now)
                for d in (results.done_ticks, results.images):
                    while d and now - (d[0] if d is results.done_ticks else d[0][0]) > 2.0:
                        d.popleft()
                window = min(2.0, now - t_start[0]) or 1e-3
                results.stats = {
                    "stream_fps": len(results.done_ticks) / window,
                    "total_fps": sum(n for _, n in results.images) / window,
                }
                results.tick, results.tick_captured = tick, captured
                results.lock.notify_all()
        with results.lock:
            results.tick = None
            results.lock.notify_all()

    def render():
        """Draw the newest completed tick; if drawing is slower than inference, display frames are skipped."""
        display = Display(streams, args.cols, tile_w, tile_h, args)
        mjpeg = Mjpeg(args.port) if args.port else None
        writer = (
            Recorder(args.record, display.canvas.shape[1], display.canvas.shape[0], args.fps) if args.record else None
        )
        shown, last_tick, last_frame = deque(), -1, None
        while True:
            with results.lock:
                results.lock.wait_for(lambda: results.tick is None or results.tick != last_tick)
                if results.tick is None:
                    break
                last_tick, captured = results.tick, results.tick_captured
                latest = list(results.latest)
                counts = [len(r[1]) if r else 0 for r in latest]
                averages = [float(np.mean(h)) if h else 0.0 for h in results.history]
                fps = [(len(u) - 1) / (u[-1] - u[0]) if len(u) > 1 and u[-1] > u[0] else 0.0 for u in results.updates]
                stats = dict(results.stats)
            for s, r in enumerate(latest):
                if r is not None:
                    display.tile(s, streams[s]["frames"][r[0]], r[1], fps[s])
            now = time.perf_counter()
            results.latencies.append((now - captured) * 1000)
            shown.append(now)
            while now - shown[0] > 2.0:
                shown.popleft()
            stats["latency_ms"] = float(np.median(results.latencies))
            stats["display_fps"] = len(shown) / (min(2.0, now - t_start[0]) or 1e-3)
            display.panel(stats, counts, averages)
            if mjpeg:
                mjpeg.publish(cv2.imencode(".jpg", display.canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            if writer:
                writer.write(display.canvas)
            last_frame = display.canvas.copy() if args.snapshot else None
            post.stats = stats
        if writer:
            writer.close()
        if last_frame is not None:
            cv2.imwrite(args.snapshot, last_frame)

    post.stats = {}
    threads = [threading.Thread(target=f, daemon=True) for f in (prep, post, render)]
    for t in threads:
        t.start()
    if args.port:
        logger.info(f"live view: http://localhost:{args.port}/")

    # Device loop: CQ1 uploads the next batch while CQ0 runs the trace; outputs go DRAM -> host on CQ1.
    staging, l1_input = runner.tt_image_res, runner.input_tensor
    read_event = ttnn.record_event(device, 1)
    item = prep_q.get()
    ttnn.wait_for_event(1, runner.op_event)
    ttnn.copy_host_to_device_tensor(item[-1], staging, 1)
    write_event = ttnn.record_event(device, 1)
    in_flight = None
    processed = 0
    t_end = time.perf_counter() + args.duration
    try:
        while time.perf_counter() < t_end:
            ttnn.wait_for_event(0, write_event)
            ttnn.copy(staging, l1_input)
            first_op_event = ttnn.record_event(device, 0)
            ttnn.execute_trace(device, runner.tid, cq_id=0, blocking=False)
            ttnn.wait_for_event(0, read_event)
            dram = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in out_list]
            last_op_event = ttnn.record_event(device, 0)

            current = item
            item = prep_q.get()
            ttnn.wait_for_event(1, first_op_event)
            ttnn.copy_host_to_device_tensor(item[-1], staging, 1)
            write_event = ttnn.record_event(device, 1)

            ttnn.wait_for_event(1, last_op_event)
            host = [ttnn.from_device(t, blocking=False, cq_id=1) for t in dram]
            read_event = ttnn.record_event(device, 1)

            # Hand the previous batch's outputs over once their read has finished (the current trace runs meanwhile)
            if in_flight is not None:
                ttnn.event_synchronize(in_flight[0])
                post_q.put(in_flight[1])
            tick, b, ids, frame_idx, captured, _ = current
            in_flight = (read_event, (tick, b, ids, frame_idx, captured, host))
            processed += B
        ttnn.event_synchronize(in_flight[0])
        post_q.put(in_flight[1])
    finally:
        stop.set()
        post_q.put(None)
        for t in threads[1:]:
            t.join(timeout=30)
        elapsed = time.perf_counter() - t_start[0]
        logger.info(
            f"{args.streams} streams, target {args.fps} FPS each: processed {processed} images in {elapsed:.1f} s "
            f"= {processed / elapsed:.1f} images/s = {processed / elapsed / args.streams:.1f} FPS per stream; "
            f"last window {post.stats}"
        )
        runner.release()
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
