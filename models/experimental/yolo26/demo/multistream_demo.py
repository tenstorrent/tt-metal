# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO26 multi-stream people counting on one chip.

N video streams (default 20) are paced at a target rate (default 30 FPS each). Every tick, the latest frame of each
stream is letterboxed and grouped into batches of --batch-size images; each batch runs through the traced model with
2 command queues (upload of the next batch on CQ1 overlaps the current trace on CQ0, outputs are read back on CQ1).
Persons (COCO class 0) are counted per stream and drawn on a mosaic with a side panel of per-stream counts.

Processes (run_pipeline, shared with multichip_demo.py), so that preparing, decoding and drawing do not delay the
device loop on the Python lock:
- --prep-workers processes letterbox frames into a shared-memory ring of input buffers (PrepPool);
- this process uploads, runs the trace and decodes; the trace also packs the outputs on device into one small
  row-major tensor (pack_outputs), so read-back and decode stay cheap;
- a render process draws the newest completed tick, serves the mosaic live as MJPEG on http://<host>:--port/ and
  optionally records it to --record.

    python models/experimental/yolo26/demo/multistream_demo.py --videos <dir> --variant yolo26s --batch-size 4
"""

import argparse
import glob
import http.server
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import time
from collections import deque
from multiprocessing import shared_memory

import cv2
import numpy as np
import torch
import ttnn
from loguru import logger

from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE
from models.experimental.yolo26.runner.performant_runner import YOLO26PerformantRunner
from models.experimental.yolo26.tt.ttnn_yolo26 import UINT8_INPUT_CHANNELS

SIZE = 640
STRIDES = [8, 16, 32]
PERSON = 0
PANEL_W = 420


def load_clips(video_dir, num_streams, max_frames, rgba=True):
    """
    Decode clips into memory (BGR, and the model-input RGBA copy unless rgba=False). Streams beyond the number of clips
    reuse clips at a different start frame.
    """
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
            inputs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGBA) for f in frames] if rgba else None
            for f in inputs or []:
                f[..., 3] = 0
            clips.append((os.path.splitext(os.path.basename(path))[0], frames, inputs))
            logger.info(f"loaded {path}: {len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]}")
    streams = []
    for i in range(num_streams):
        name, frames, inputs = clips[i % len(clips)]
        streams.append({"name": name, "frames": frames, "rgba": inputs, "offset": (i // len(clips)) * len(frames) // 2})
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


def pack_outputs(output):
    """
    On device, inside the trace: [boxes p3..p5 (ltrb x4), scores p3..p5 (80 logits)] -> one row-major DRAM tensor
    [1, 1, sum(B * h * w), 5] of (l, t, r, b, person logit), levels in order, images in order within a level. Read-back
    and host conversion of this are ~100x smaller than of the six tile outputs.
    """
    dram = ttnn.DRAM_MEMORY_CONFIG
    parts = []
    for (box, _, _), (score, _, _) in zip(output["boxes"], output["scores"]):
        # Copy out of L1 first: the slice / concat buffers do not fit next to the model's L1 outputs
        box, score = ttnn.to_memory_config(box, dram), ttnn.to_memory_config(score, dram)
        n = box.shape[2]
        person = ttnn.slice(score, [0, 0, 0, PERSON], [1, 1, n, PERSON + 1], memory_config=dram)
        parts.append(ttnn.concat([box, person], dim=3, memory_config=dram))
    return ttnn.to_layout(ttnn.concat(parts, dim=2, memory_config=dram), ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)


def decode_packed(packed, shapes, per_chip, anchors, strides, conf):
    """
    packed: [chips, 1, rows, 5] float (see pack_outputs). Returns per image (chip-major) an array [k, 5] of
    (x1, y1, x2, y2, conf) in input-image pixels. NMS-free one2one head: boxes are ltrb distances in units of the
    stride, scores are logits.
    """
    logit = np.log(conf / (1 - conf))
    people = []
    for chip in packed[:, 0]:
        levels, off = [], 0
        for h, w in shapes:
            levels.append(chip[off : off + per_chip * h * w].reshape(per_chip, h * w, 5))
            off += per_chip * h * w
        for img in np.concatenate(levels, 1):
            keep = np.nonzero(img[:, 4] > logit)[0]
            ltrb, a, s = img[keep, :4], anchors[keep], strides[keep]
            xyxy = np.concatenate((a - ltrb[:, :2], a + ltrb[:, 2:]), 1) * s
            p = 1 / (1 + np.exp(-img[keep, 4]))
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

    def __init__(self, args, num_streams, chips):
        self.clips = load_clips(args.videos, num_streams, args.max_frames, rgba=False)
        self.table = stream_table(self.clips, num_streams, chips)
        self.n, self.cols, self.args = num_streams, args.cols, args
        self.tile_w, self.tile_h = map(int, args.tile.split("x"))
        rows = (num_streams + self.cols - 1) // self.cols
        self.canvas = np.zeros((rows * self.tile_h, self.cols * self.tile_w + PANEL_W, 3), np.uint8)

    def tile(self, i, frame_idx, people_src, fps):
        frame = self.clips[self.table[i][0]]["frames"][frame_idx]
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

    def panel(self, stats, counts):
        x0 = self.canvas.shape[1] - PANEL_W
        p = self.canvas[:, x0:]
        p[:] = (24, 24, 24)
        put = lambda text, y, scale=0.55, color=(230, 230, 230), thick=1: cv2.putText(
            p, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
        )
        put(f"YOLO26{self.args.variant[-1]} people counting", 32, 0.7, (255, 255, 255), 2)
        put(f"1 Tenstorrent Blackhole chip, batch {self.args.batch_size}", 58)
        put(f"{self.n} streams x {stats['stream_fps']:.1f} FPS", 96, 0.75, (90, 230, 90), 2)
        put(f"= {stats['total_fps']:.0f} images/s", 124)
        put(f"latency {stats['latency_ms']:.0f} ms (median)   display {stats['display_fps']:.0f} FPS", 148)
        total = sum(counts)
        put(f"People now: {total}", 186, 0.75, (255, 255, 255), 2)
        top = 214
        row_h = max(18, (p.shape[0] - top - 10) // self.n)
        put("camera                                   people now", top - 6, 0.42, (150, 150, 150))
        bar_x0, bar_x1 = 90, PANEL_W - 70
        scale = max(4, max(counts))
        text = lambda s, x, y, color=(255, 255, 255): cv2.putText(
            p, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
        for i, n in enumerate(counts):
            y = top + 4 + i * row_h
            text(f"CAM {i + 1:02d}", 16, y + 13, (210, 210, 210))
            cv2.rectangle(p, (bar_x0, y + 2), (bar_x1, y + row_h - 4), (45, 45, 45), -1)
            cv2.rectangle(
                p, (bar_x0, y + 2), (bar_x0 + int((bar_x1 - bar_x0) * n / scale), y + row_h - 4), (60, 170, 60), -1
            )
            text(f"{n:2d}", PANEL_W - 56, y + 13)


def stream_table(clips, num_streams, chips):
    """
    Stream s -> (clip entry, start frame). clips: load_clips(..., per_chip streams); the copies of an entry on
    different chips start 1 / chips of a clip apart.
    """
    n = len(clips)
    return [
        (s % n, clips[s % n]["offset"] + (s // n) * len(clips[s % n]["frames"]) // chips) for s in range(num_streams)
    ]


def ring_views(shm, slots, batch):
    return np.ndarray((slots, batch, SIZE, SIZE, UINT8_INPUT_CHANNELS), np.uint8, buffer=shm.buf)


def prep_worker(rank, workers, args, layout, shm_name, slots, free_q, work_q, ready_q, start, t0, stop):
    """
    Letterbox the current frame of each stream of this worker's passes into free ring slots, paced at --fps. Pass b
    of a tick is captured b / passes of a tick after the tick starts (like free-running cameras), unless
    --synchronized-capture, where all streams are captured at the start of the tick.
    """
    per_chip, B, chips = layout
    clips = load_clips(args.videos, per_chip, args.max_frames)
    table = stream_table(clips, chips * per_chip, chips)
    rgba = [c["rgba"] for c in clips]
    letterbox = Letterbox(*rgba[0][0].shape[:2], SIZE)
    shm = shared_memory.SharedMemory(name=shm_name)
    ring = ring_views(shm, slots, chips * B)
    ready_q.put(rgba[0][0].shape[:2])
    start.wait()
    period = 1.0 / args.fps
    passes = per_chip // B
    tick = 0
    mine = [b for b in range(passes) if b % workers == rank]
    while not stop.is_set():
        for b in mine:
            due = t0.value + tick * period + (0 if args.synchronized_capture else b * period / passes)
            now = time.perf_counter()
            if due > now:
                time.sleep(due - now)
            captured = time.perf_counter()
            while True:
                try:
                    slot = free_q.get(timeout=0.5)
                    break
                except queue.Empty:
                    if stop.is_set():
                        return
            ids = [g * per_chip + b * B + i for g in range(chips) for i in range(B)]
            frame_idx = []
            for j, s in enumerate(ids):
                clip, offset = table[s]
                f = (tick + offset) % len(rgba[clip])
                letterbox.fill(ring[slot, j], rgba[clip][f])
                frame_idx.append(f)
            work_q.put((tick, b, ids, frame_idx, captured, slot))
        tick += 1


class PrepPool:
    """
    Host input preparation in worker processes. Each mesh pass b of a tick holds image j = chip g * B + i of stream
    g * per_chip + b * B + i; passes are split round-robin over the workers. Buffers live in a shared-memory ring
    whose letterbox padding (114 in RGB) is written once; a slot is free again once its pass has been read back.
    """

    def __init__(self, args, per_chip, chips, batch_size, workers):
        self.batch = chips * batch_size
        self.slots = 2 * (per_chip // batch_size) + 6
        self.shm = shared_memory.SharedMemory(
            create=True, size=self.slots * self.batch * SIZE * SIZE * UINT8_INPUT_CHANNELS
        )
        self.ring = ring_views(self.shm, self.slots, self.batch)
        self.ring[..., :3] = 114
        self.ring[..., 3:] = 0
        ctx = mp.get_context("spawn")
        self.free_q, self.work_q, self.ready_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
        for k in range(self.slots):
            self.free_q.put(k)
        self.start_event, self.stop, self.t0 = ctx.Event(), ctx.Event(), ctx.Value("d", 0.0)
        self.halt = threading.Event()
        layout = (per_chip, batch_size, chips)
        self.procs = [
            ctx.Process(
                target=prep_worker,
                args=(rank, workers, args, layout, self.shm.name, self.slots, self.free_q, self.work_q)
                + (self.ready_q, self.start_event, self.t0, self.stop),
                daemon=True,
            )
            for rank in range(workers)
        ]
        for p in self.procs:
            p.start()

    def start(self, prep_q, mesh_mapper=None):
        """
        Wait for the workers, start the ticks and feed ring slots as host tensors (no copy) to prep_q. Returns the
        time of the first tick; src_hw is the source frame size.
        """
        self.src_hw = [self.ready_q.get() for _ in self.procs][0]
        self.t0.value = time.perf_counter() + 0.1
        self.start_event.set()

        def feed():
            while not self.halt.is_set():
                try:
                    tick, b, ids, frame_idx, captured, slot = self.work_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                host = ttnn.from_torch(
                    torch.from_numpy(self.ring[slot]).reshape(self.batch, 1, SIZE, SIZE * UINT8_INPUT_CHANNELS),
                    dtype=ttnn.uint8,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper,
                )
                prep_q.put((tick, b, ids, frame_idx, captured, slot, host))

        threading.Thread(target=feed, daemon=True).start()
        return self.t0.value

    def release(self, slot):
        self.free_q.put(slot)

    def close(self):
        self.stop.set()
        self.halt.set()
        for p in self.procs:
            p.join(timeout=30)
        self.shm.unlink()  # the mapping itself goes with the process


def render_process(make_display, args, num_streams, chips, render_q, ready_q):
    """
    Separate process: as soon as results arrive (merging any that queued meanwhile), their tiles are drawn and the
    mosaic is composed and published as MJPEG; the latest mosaic is recorded on a fixed --fps clock (real-time video).
    On-screen latency is per result: capture of a tile's frame to the publication of the first mosaic showing it.
    make_display(args, num_streams, chips) gives an object with canvas, tile(s, frame_idx, dets, fps) and
    panel(stats, counts).
    """
    display = make_display(args, num_streams, chips)
    mjpeg = Mjpeg(args.port) if args.port else None
    writer = Recorder(args.record, display.canvas.shape[1], display.canvas.shape[0], args.fps) if args.record else None
    ready_q.put("ready")
    period = 1.0 / args.fps
    counts, captured = [0] * num_streams, [None] * num_streams
    stats, recent, all_latencies, composed = {}, deque(maxlen=20 * num_streams), [], deque()
    recorded, next_record, last = 0, None, None
    shown = {"latency_ms": 0.0, "display_fps": 0.0}  # renderer-side stats, carried across updates
    finished = False
    while not finished:
        timeout = None if next_record is None else max(0.0, next_record - time.perf_counter())
        msgs = []
        try:
            msgs.append(render_q.get(timeout=timeout))
            while True:
                msgs.append(render_q.get_nowait())
        except queue.Empty:
            pass
        fresh = set()
        for msg in msgs:
            if msg is None:
                finished = True
                break
            updates, stats = msg
            for s, frame_idx, dets, fps, cap in updates:
                display.tile(s, frame_idx, dets, fps)
                counts[s], captured[s] = len(dets), cap
                fresh.add(s)
        if fresh and "stream_fps" in stats:
            display.panel({**stats, **shown}, counts)
            if mjpeg:
                mjpeg.publish(cv2.imencode(".jpg", display.canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            now = time.perf_counter()
            ages = [(now - captured[s]) * 1000 for s in fresh]
            recent.extend(ages)
            all_latencies.extend(ages)
            shown["latency_ms"] = float(np.median(recent))
            composed.append(now)
            while now - composed[0] > 2.0:
                composed.popleft()
            shown["display_fps"] = len(composed) / min(2.0, max(now - composed[0], period))
            last = display.canvas
            next_record = next_record or now
        now = time.perf_counter()
        if next_record is not None and now >= next_record and last is not None:
            if writer:
                writer.write(last)
            recorded += 1
            next_record += period
            if next_record < now:  # fell behind: skip instead of bursting
                next_record = now + period
    if writer:
        writer.close()
    if args.snapshot and last is not None:
        cv2.imwrite(args.snapshot, last)
    ready_q.put((float(np.median(all_latencies)) if all_latencies else 0.0, recorded))


def run_device_loop(device, runner, out_list, prep_q, post_q, t_end, images_per_batch, wait_next=0.0):
    """
    Run queued host batches until t_end. prep_q items end with the host input tensor; each item is handed to post_q
    with the host outputs in place of the input once their read has finished.

    CQ1 uploads the next batch while CQ0 runs the trace; outputs go DRAM -> host on CQ1, queued after the next
    upload so that the upload overlaps the current trace. A batch is handed over as soon as the trace after it is
    queued (not after the next input arrives). If no next batch arrives within wait_next seconds (end of a tick), the
    device would idle anyway: the current batch is read and handed over before waiting. With staggered capture, wait_next
    of one capture interval keeps the upload-first order (a read queued first would hold the upload until the trace
    ends).
    """
    staging, l1_input = runner.tt_image_res, runner.input_tensor
    read_event = ttnn.record_event(device, 1)
    item = prep_q.get()
    ttnn.wait_for_event(1, runner.op_event)
    ttnn.copy_host_to_device_tensor(item[-1], staging, 1)
    write_event = ttnn.record_event(device, 1)
    in_flight = None
    processed = 0
    while time.perf_counter() < t_end:
        ttnn.wait_for_event(0, write_event)
        ttnn.copy(staging, l1_input)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, runner.tid, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)
        dram = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in out_list]
        last_op_event = ttnn.record_event(device, 0)

        # Hand the previous batch over now, before waiting for the next input: its read was queued last iteration
        # and this trace is already queued behind it, so the device stays busy while the host waits
        if in_flight is not None:
            ttnn.event_synchronize(in_flight[0])
            post_q.put(in_flight[1])
            in_flight = None

        current = item
        try:
            item = prep_q.get(timeout=wait_next) if wait_next > 0 else prep_q.get_nowait()
        except queue.Empty:
            item = None
        if item is None:
            ttnn.wait_for_event(1, last_op_event)
            host = [ttnn.from_device(t, blocking=False, cq_id=1) for t in dram]
            read_event = ttnn.record_event(device, 1)
            ttnn.event_synchronize(read_event)
            post_q.put(current[:-1] + (host,))
            item = prep_q.get()
            ttnn.wait_for_event(1, first_op_event)
            ttnn.copy_host_to_device_tensor(item[-1], staging, 1)
            write_event = ttnn.record_event(device, 1)
        else:
            ttnn.wait_for_event(1, first_op_event)
            ttnn.copy_host_to_device_tensor(item[-1], staging, 1)
            write_event = ttnn.record_event(device, 1)

            ttnn.wait_for_event(1, last_op_event)
            host = [ttnn.from_device(t, blocking=False, cq_id=1) for t in dram]
            read_event = ttnn.record_event(device, 1)
            in_flight = (read_event, current[:-1] + (host,))
        processed += images_per_batch
    if in_flight is not None:
        ttnn.event_synchronize(in_flight[0])
        post_q.put(in_flight[1])
    return processed


def run_pipeline(device, args, chips, per_chip, make_display, mesh_mapper=None, mesh_composer=None):
    """
    The whole demo on an open device or mesh: per_chip streams on each of chips chips, batch args.batch_size per chip.
    Prep workers -> device loop (trace + on-device output packing, 2 CQs) -> decode (this process) -> render process.
    """
    B = args.batch_size
    if per_chip % B:
        raise ValueError("streams per chip must be a multiple of --batch-size")
    num_streams, nb, mesh_batch = chips * per_chip, per_chip // B, chips * B

    pool = PrepPool(args, per_chip, chips, B, args.prep_workers)
    ctx = mp.get_context("spawn")
    render_q, ready_q = ctx.Queue(), ctx.Queue()  # every pass's update is applied, so none may be dropped
    renderer = ctx.Process(
        target=render_process, args=(make_display, args, num_streams, chips, render_q, ready_q), daemon=True
    )
    renderer.start()

    runner = YOLO26PerformantRunner(device, B, SIZE, args.variant)
    infra, model_run = runner.runner_infra, runner.runner_infra.run

    def run_and_pack():
        out = model_run()
        infra.packed = pack_outputs(out)
        return out

    infra.run = run_and_pack  # the packing is captured in the same trace
    runner.capture_trace_2cq()
    shapes = [(h, w) for _, h, w in infra.output_tensor["boxes"]]
    anchors, strides = anchor_grid(SIZE)
    while True:  # the renderer loads and resizes its clips meanwhile; stop if it died
        try:
            ready_q.get(timeout=5)
            break
        except queue.Empty:
            if not renderer.is_alive():
                raise RuntimeError("render process exited during start-up (see its traceback above)")

    prep_q = queue.Queue(maxsize=2 * nb)
    post_q = queue.Queue()  # decode keeps up easily; unbounded so the device loop never waits
    stats, pass_latency = {}, [[] for _ in range(nb)]
    t_start = pool.start(prep_q, mesh_mapper)
    letterbox = Letterbox(*pool.src_hw, SIZE)

    def post():
        """Decode every pass, free its ring slot and send its streams' results to the renderer."""
        updates = [deque() for _ in range(num_streams)]
        pending, done_ticks, images = {}, deque(), deque()
        while (item := post_q.get()) is not None:
            tick, b, ids, frame_idx, captured, slot, host_outputs = item
            pool.release(slot)
            packed = ttnn.to_torch(host_outputs[0], mesh_composer=mesh_composer).float().numpy()
            people = decode_packed(packed.reshape(chips, 1, -1, 5), shapes, B, anchors, strides, args.conf)
            now = time.perf_counter()
            if captured - t_start > 2.0:
                pass_latency[b].append((now - captured) * 1000)
            sent = []
            for j, s in enumerate(ids):
                src = letterbox.to_source(people[j][:, :4]) if len(people[j]) else np.zeros((0, 4))
                dets = np.concatenate((src, people[j][:, 4:]), 1) if len(people[j]) else []
                u = updates[s]
                u.append(now)
                while now - u[0] > 1.0:
                    u.popleft()
                fps = (len(u) - 1) / (u[-1] - u[0]) if len(u) > 1 and u[-1] > u[0] else 0.0
                sent.append((s, frame_idx[j], dets, fps, captured))
            render_q.put((sent, dict(stats)))
            images.append((now, mesh_batch))
            pending[tick] = pending.get(tick, 0) + 1
            if pending[tick] < nb:
                continue
            del pending[tick]
            done_ticks.append(now)
            while now - done_ticks[0] > 2.0:
                done_ticks.popleft()
            while now - images[0][0] > 2.0:
                images.popleft()
            window = min(2.0, now - t_start) or 1e-3
            stats.update(stream_fps=len(done_ticks) / window, total_fps=sum(n for _, n in images) / window)

    poster = threading.Thread(target=post, daemon=True)
    poster.start()
    if args.port:
        logger.info(f"live view: http://localhost:{args.port}/")

    processed = 0
    try:
        # staggered capture: the next batch is due one capture interval after the current one
        wait_next = 0.0 if args.synchronized_capture else 1.0 / (args.fps * nb)
        processed = run_device_loop(
            device, runner, [infra.packed], prep_q, post_q, t_start + args.duration, mesh_batch, wait_next
        )
    finally:
        elapsed = time.perf_counter() - t_start
        post_q.put(None)
        poster.join(timeout=30)
        pool.close()
        render_q.put(None)
        try:
            shown = ready_q.get(timeout=120)
        except queue.Empty:
            shown = None
        renderer.join(timeout=30)
        groups = "; ".join(
            f"CAM {b * B + 1:02d}-{(b + 1) * B:02d} {np.median(v):.1f} ms" for b, v in enumerate(pass_latency) if v
        )
        logger.info(
            f"{num_streams} streams on {chips} chip(s), target {args.fps} FPS each: processed {processed} images in "
            f"{elapsed:.1f} s = {processed / elapsed:.1f} images/s = {processed / elapsed / num_streams:.1f} FPS per "
            f"stream; last window {stats}; capture-to-results median per pass (chip 1 cameras): {groups}; "
            f"on screen per tile (median ms, frames recorded): {shown}"
        )
        runner.release()


def add_common_args(ap):
    ap.add_argument("--videos", required=True, help="directory of .mpg / .mp4 clips")
    ap.add_argument("--variant", default="yolo26s")
    ap.add_argument("--fps", type=int, default=30, help="target FPS per stream")
    ap.add_argument("--batch-size", type=int, default=4, help="images per chip per pass")
    ap.add_argument("--prep-workers", type=int, default=2, help="processes letterboxing frames")
    ap.add_argument("--duration", type=float, default=60, help="seconds to run")
    ap.add_argument("--conf", type=float, default=0.35, help="person confidence threshold")
    ap.add_argument("--max-frames", type=int, default=1500, help="frames decoded per clip (loops)")
    ap.add_argument("--port", type=int, default=8090, help="MJPEG port, 0 = off")
    ap.add_argument("--record", default="", help="record the mosaic to this .mp4")
    ap.add_argument("--snapshot", default="", help="save the last mosaic frame to this .jpg")
    ap.add_argument(
        "--synchronized-capture",
        action="store_true",
        help="capture all streams at the start of each tick (default: each pass's cameras at their own phase)",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--streams", type=int, default=20)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tile", default="384x288")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()
    device = ttnn.open_device(
        device_id=args.device_id, l1_small_size=YOLO26_L1_SMALL_SIZE, trace_region_size=16000000, num_command_queues=2
    )
    try:
        run_pipeline(device, args, 1, args.streams, Display)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
