# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO26 multi-stream people counting on all chips of a QB2 (data parallel, one mesh).

Same pipeline as multistream_demo.py, on a 1 x --chips mesh: the traced model runs --batch-size images per chip, so
one mesh pass carries batch_size x chips images. Each chip serves its own --streams-per-chip streams (default 20, i.e.
80 streams on 4 chips); the same clips are reused on every chip, each chip starting at a different frame.

At 80 streams the host work no longer fits next to the device loop in one Python process, so it is split:
- --prep-workers processes letterbox frames into a shared-memory ring of mesh input buffers (padding written once);
- this process uploads, runs the trace and decodes. The trace also packs the outputs on device into one small
  row-major tensor (4 box values + the person logit per anchor), so read-back and host conversion stay small;
- a render process draws the mosaic, serves MJPEG and records.

    python models/experimental/yolo26/demo/multichip_demo.py --videos <dir> --record out.mp4
"""

import argparse
import multiprocessing as mp
import queue
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
from models.experimental.yolo26.demo.multistream_demo import (
    PERSON,
    Letterbox,
    Mjpeg,
    Recorder,
    anchor_grid,
    load_clips,
    run_device_loop,
)
from models.experimental.yolo26.runner.performant_runner import YOLO26PerformantRunner
from models.experimental.yolo26.tt.ttnn_yolo26 import UINT8_INPUT_CHANNELS

SIZE = 640
PANEL_W = 420
TILE_COLS = 10  # 2 rows of 10 tiles per chip


def stream_table(clip_lengths, num_streams, chips):
    """Stream s -> (clip, start frame); the copies of a clip on different chips start 1 / chips of a clip apart."""
    n = len(clip_lengths)
    return [(s % n, (s // n) * clip_lengths[s % n] // chips) for s in range(num_streams)]


def pack_outputs(output):
    """
    On device, inside the trace: [boxes p3..p5 (ltrb x4), scores p3..p5 (80 logits)] -> one row-major DRAM tensor
    [1, 1, sum(B * h * w), 5] of (l, t, r, b, person logit), levels in order, images in order within a level.
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
    """packed: [chips, 1, rows, 5] float (see pack_outputs). Returns per image (chip-major) [k, 5] xyxy + conf."""
    logit = np.log(conf / (1 - conf))
    people = []
    for chip in packed[:, 0]:
        levels, off = [], 0
        for h, w in shapes:
            levels.append(chip[off : off + per_chip * h * w].reshape(per_chip, h * w, 5))
            off += per_chip * h * w
        chip = np.concatenate(levels, 1)
        for img in chip:
            keep = np.nonzero(img[:, 4] > logit)[0]
            ltrb, a, s = img[keep, :4], anchors[keep], strides[keep]
            xyxy = np.concatenate((a - ltrb[:, :2], a + ltrb[:, 2:]), 1) * s
            p = 1 / (1 + np.exp(-img[keep, 4]))
            people.append(np.concatenate((xyxy, p[:, None]), 1))
    return people


def ring_views(shm, slots, mesh_batch):
    shape = (slots, mesh_batch, SIZE, SIZE, UINT8_INPUT_CHANNELS)
    return np.ndarray(shape, np.uint8, buffer=shm.buf)


def prep_process(rank, workers, args, plan, shm_name, slots, free_q, work_q, ready_q, start, t0, stop):
    """Letterbox the current frame of each stream of this worker's passes into free ring slots, paced at --fps."""
    spc, B, chips, nb = plan
    clips = load_clips(args.videos, spc, args.max_frames)
    rgba = [c["rgba"] for c in clips]
    del clips
    num_streams = chips * spc
    table = stream_table([len(r) for r in rgba], num_streams, chips)
    letterbox = Letterbox(*rgba[0][0].shape[:2], SIZE)
    shm = shared_memory.SharedMemory(name=shm_name)
    ring = ring_views(shm, slots, chips * B)
    ready_q.put(("prep", rank, rgba[0][0].shape[:2]))
    start.wait()
    t0 = t0.value
    period = 1.0 / args.fps
    tick = 0
    mine = [b for b in range(nb) if b % workers == rank]
    while not stop.is_set():
        due = t0 + tick * period
        now = time.perf_counter()
        if due > now:
            time.sleep(due - now)
        captured = time.perf_counter()
        for b in mine:
            while True:
                try:
                    slot = free_q.get(timeout=0.5)
                    break
                except queue.Empty:
                    if stop.is_set():
                        return
            ids = [g * spc + b * B + i for g in range(chips) for i in range(B)]
            frame_idx = []
            for j, s in enumerate(ids):
                clip, offset = table[s]
                f = (tick + offset) % len(rgba[clip])
                letterbox.fill(ring[slot, j], rgba[clip][f])
                frame_idx.append(f)
            work_q.put((tick, b, ids, frame_idx, captured, slot))
        tick += 1
    del ring
    shm.close()


class Display:
    """80 small tiles (one band of 2 rows per chip) and a side panel of per-chip camera counts."""

    def __init__(self, num_streams, chips, tile_w, tile_h, args):
        self.n, self.chips, self.tile_w, self.tile_h, self.args = num_streams, chips, tile_w, tile_h, args
        self.per_chip = num_streams // chips
        rows = (num_streams + TILE_COLS - 1) // TILE_COLS
        self.canvas = np.zeros((rows * tile_h, TILE_COLS * tile_w + PANEL_W, 3), np.uint8)

    def tile(self, s, small, people_src, sx, sy, fps):
        r, c = divmod(s, TILE_COLS)
        view = self.canvas[r * self.tile_h : (r + 1) * self.tile_h, c * self.tile_w : (c + 1) * self.tile_w]
        view[:] = small
        for x1, y1, x2, y2, _ in people_src:
            cv2.rectangle(view, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (60, 220, 60), 1)
        cv2.rectangle(view, (0, 0), (self.tile_w, 15), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(view, f"CAM {s + 1:02d}: {len(people_src)}", (4, 11), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        text = f"{fps:4.1f} FPS"
        (w, _), _ = cv2.getTextSize(text, font, 0.38, 1)
        cv2.putText(view, text, (self.tile_w - w - 4, 11), font, 0.38, (90, 230, 90), 1, cv2.LINE_AA)
        cv2.rectangle(view, (0, 0), (self.tile_w - 1, self.tile_h - 1), (40, 40, 40), 1)

    def panel(self, stats, counts):
        x0 = self.canvas.shape[1] - PANEL_W
        p = self.canvas[:, x0:]
        p[:] = (24, 24, 24)
        font = cv2.FONT_HERSHEY_SIMPLEX
        put = lambda text, x, y, scale=0.55, color=(230, 230, 230), thick=1: cv2.putText(
            p, text, (x, y), font, scale, color, thick, cv2.LINE_AA
        )
        put(f"YOLO26{self.args.variant[-1]} people counting", 16, 32, 0.7, (255, 255, 255), 2)
        put(f"{self.chips} Tenstorrent Blackhole chips, batch {self.args.batch_size} each", 16, 58)
        put(f"{self.n} streams x {stats['stream_fps']:.1f} FPS", 16, 96, 0.75, (90, 230, 90), 2)
        put(f"= {stats['total_fps']:.0f} images/s", 16, 124)
        put(f"latency {stats['latency_ms']:.0f} ms (median)   display {stats['display_fps']:.0f} FPS", 16, 148)
        put(f"People now: {sum(counts)}", 16, 186, 0.75, (255, 255, 255), 2)

        top, cols = 204, 5
        block_h = (p.shape[0] - top - 6) // self.chips
        cell_w = (PANEL_W - 32) // cols
        rows = (self.per_chip + cols - 1) // cols
        cell_h = (block_h - 30) // rows
        scale = max(4, max(counts))
        for g in range(self.chips):
            y0 = top + g * block_h
            first, last = g * self.per_chip, (g + 1) * self.per_chip
            chip_counts = counts[first:last]
            put(f"Chip {g + 1}   CAM {first + 1:02d}-{last:02d}", 16, y0 + 18, 0.5, (210, 210, 210))
            text = f"people {sum(chip_counts)}"
            (w, _), _ = cv2.getTextSize(text, font, 0.5, 1)
            put(text, PANEL_W - 16 - w, y0 + 18, 0.5, (255, 255, 255))
            for k, n in enumerate(chip_counts):
                r, c = divmod(k, cols)
                x, y = 16 + c * cell_w, y0 + 26 + r * cell_h
                cv2.rectangle(p, (x, y), (x + cell_w - 4, y + cell_h - 4), (45, 45, 45), -1)
                bar = int((cell_h - 4) * n / scale)
                cv2.rectangle(p, (x, y + cell_h - 4 - bar), (x + cell_w - 4, y + cell_h - 4), (60, 140, 60), -1)
                put(f"{first + k + 1:02d}", x + 4, y + 13, 0.36, (170, 170, 170))
                put(f"{n}", x + cell_w - 22, y + cell_h - 12, 0.6, (255, 255, 255), 1)


def render_process(args, num_streams, chips, render_q, ready_q):
    """Draw the newest completed tick, serve MJPEG, record."""
    tile_w, tile_h = map(int, args.tile.split("x"))
    clips = load_clips(args.videos, num_streams // chips, args.max_frames, rgba=False)
    table = stream_table([len(c["frames"]) for c in clips], num_streams, chips)
    src_h, src_w = clips[0]["frames"][0].shape[:2]
    sx, sy = tile_w / src_w, tile_h / src_h
    # Tiles are drawn from frames resized once at load (tile size, ~2 GB for the CAVIAR set)
    small = [[cv2.resize(f, (tile_w, tile_h), interpolation=cv2.INTER_AREA) for f in c["frames"]] for c in clips]
    del clips
    display = Display(num_streams, chips, tile_w, tile_h, args)
    mjpeg = Mjpeg(args.port) if args.port else None
    writer = Recorder(args.record, display.canvas.shape[1], display.canvas.shape[0], args.fps) if args.record else None
    ready_q.put(("render",))
    latencies, shown, t_start, last = deque(maxlen=200), deque(), None, None
    all_latencies = []
    finished = False
    while not finished:
        msg = render_q.get()
        try:  # skip to the newest tick if drawing fell behind
            while msg is not None:
                newer = render_q.get_nowait()
                if newer is None:
                    finished = True
                    break
                msg = newer
        except queue.Empty:
            pass
        if msg is None:
            break
        tick, captured, latest, fps, stats = msg
        for s, r in enumerate(latest):
            if r is not None:
                display.tile(s, small[table[s][0]][r[0]], r[1], sx, sy, fps[s])
        now = time.perf_counter()
        t_start = t_start or now
        latencies.append((now - captured) * 1000)
        all_latencies.append(latencies[-1])
        shown.append(now)
        while now - shown[0] > 2.0:
            shown.popleft()
        stats["latency_ms"] = float(np.median(latencies))
        stats["display_fps"] = len(shown) / (min(2.0, now - t_start) or 1e-3)
        display.panel(stats, [len(r[1]) if r else 0 for r in latest])
        if mjpeg:
            mjpeg.publish(cv2.imencode(".jpg", display.canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        if writer:
            writer.write(display.canvas)
        last = display.canvas
    if writer:
        writer.close()
    if args.snapshot and last is not None:
        cv2.imwrite(args.snapshot, last)
    ready_q.put(("render_done", float(np.median(all_latencies)) if all_latencies else 0.0, len(all_latencies)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="directory of .mpg / .mp4 clips")
    ap.add_argument("--variant", default="yolo26s")
    ap.add_argument("--chips", type=int, default=4)
    ap.add_argument("--streams-per-chip", type=int, default=20)
    ap.add_argument("--fps", type=int, default=30, help="target FPS per stream")
    ap.add_argument("--batch-size", type=int, default=4, help="images per chip per pass")
    ap.add_argument("--prep-workers", type=int, default=2)
    ap.add_argument("--duration", type=float, default=60, help="seconds to run")
    ap.add_argument("--conf", type=float, default=0.35, help="person confidence threshold")
    ap.add_argument("--max-frames", type=int, default=1500, help="frames decoded per clip (loops)")
    ap.add_argument("--port", type=int, default=8090, help="MJPEG port, 0 = off")
    ap.add_argument("--record", default="", help="record the mosaic to this .mp4")
    ap.add_argument("--snapshot", default="", help="save the last mosaic frame to this .jpg")
    ap.add_argument("--tile", default="192x144")
    args = ap.parse_args()
    B, chips, spc = args.batch_size, args.chips, args.streams_per_chip
    if spc % B:
        raise ValueError("--streams-per-chip must be a multiple of --batch-size")
    num_streams, nb, mesh_batch = chips * spc, spc // B, chips * B

    # Ring of mesh input buffers; the letterbox padding (114 in RGB) is written once and never overwritten
    slots = 2 * nb + 6
    shm = shared_memory.SharedMemory(create=True, size=slots * mesh_batch * SIZE * SIZE * UINT8_INPUT_CHANNELS)
    ring = ring_views(shm, slots, mesh_batch)
    ring[..., :3] = 114
    ring[..., 3:] = 0

    ctx = mp.get_context("spawn")
    free_q, work_q, ready_q, render_q = ctx.Queue(), ctx.Queue(), ctx.Queue(), ctx.Queue(maxsize=8)
    for k in range(slots):
        free_q.put(k)
    start, stop, t0 = ctx.Event(), ctx.Event(), ctx.Value("d", 0.0)
    procs = [ctx.Process(target=render_process, args=(args, num_streams, chips, render_q, ready_q), daemon=True)]
    plan = (spc, B, chips, nb)
    for rank in range(args.prep_workers):
        procs.append(
            ctx.Process(
                target=prep_process,
                args=(rank, args.prep_workers, args, plan, shm.name, slots, free_q, work_q, ready_q, start, t0, stop),
                daemon=True,
            )
        )
    for p in procs:
        p.start()

    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, chips),
        l1_small_size=YOLO26_L1_SMALL_SIZE,
        trace_region_size=8000000,
        num_command_queues=2,
    )
    mapper, composer = ttnn.ShardTensorToMesh(device, dim=0), ttnn.ConcatMeshToTensor(device, dim=0)
    runner = YOLO26PerformantRunner(device, B, SIZE, args.variant)
    infra, model_run = runner.runner_infra, runner.runner_infra.run

    def run_and_pack():
        out = model_run()
        infra.packed = pack_outputs(out)
        return out

    infra.run = run_and_pack  # the packing is captured in the same trace
    runner.capture_trace_2cq()
    out_list = [infra.packed]
    shapes = [(h, w) for _, h, w in infra.output_tensor["boxes"]]
    anchors, strides = anchor_grid(SIZE)

    src_hw = None
    for _ in procs:
        msg = ready_q.get()
        if msg[0] == "prep":
            src_hw = msg[2]
    letterbox = Letterbox(*src_hw, SIZE)
    t_start = time.perf_counter() + 0.1
    t0.value = t_start
    start.set()

    prep_q = queue.Queue(maxsize=2 * nb)
    post_q = queue.Queue()
    halt = threading.Event()

    def feed():
        """Ring slot -> host mesh tensor (no copy) for the device loop."""
        while not halt.is_set():
            try:
                tick, b, ids, frame_idx, captured, slot = work_q.get(timeout=0.5)
            except queue.Empty:
                continue
            host = ttnn.from_torch(
                torch.from_numpy(ring[slot]).reshape(mesh_batch, 1, SIZE, SIZE * UINT8_INPUT_CHANNELS),
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            prep_q.put((tick, b, ids, frame_idx, captured, slot, host))

    stats = {}

    def post():
        """Decode every pass and free its ring slot (its upload finished before its read-back); send ticks on."""
        latest = [None] * num_streams
        updates = [deque() for _ in range(num_streams)]
        pending, done_ticks, images = {}, deque(), deque()
        while (item := post_q.get()) is not None:
            tick, b, ids, frame_idx, captured, slot, host_outputs = item
            free_q.put(slot)
            packed = ttnn.to_torch(host_outputs[0], mesh_composer=composer).float().numpy()
            people = decode_packed(packed, shapes, B, anchors, strides, args.conf)
            now = time.perf_counter()
            for j, s in enumerate(ids):
                src = letterbox.to_source(people[j][:, :4]) if len(people[j]) else np.zeros((0, 4))
                latest[s] = (frame_idx[j], np.concatenate((src, people[j][:, 4:]), 1) if len(people[j]) else [])
                u = updates[s]
                u.append(now)
                while now - u[0] > 1.0:
                    u.popleft()
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
            fps = [(len(u) - 1) / (u[-1] - u[0]) if len(u) > 1 and u[-1] > u[0] else 0.0 for u in updates]
            try:
                render_q.put_nowait((tick, captured, list(latest), fps, dict(stats)))
            except queue.Full:
                pass

    threads = [threading.Thread(target=f, daemon=True) for f in (feed, post)]
    for t in threads:
        t.start()
    if args.port:
        logger.info(f"live view: http://localhost:{args.port}/")

    processed = 0
    try:
        processed = run_device_loop(device, runner, out_list, prep_q, post_q, t_start + args.duration, mesh_batch)
    finally:
        stop.set()
        halt.set()
        post_q.put(None)
        threads[1].join(timeout=30)
        elapsed = time.perf_counter() - t_start
        render_q.put(None)
        render = {}
        try:
            while True:
                msg = ready_q.get(timeout=120)
                if msg[0] == "render_done":
                    render = {"latency_ms_median": msg[1], "frames": msg[2]}
                    break
        except queue.Empty:
            pass
        for p in procs:
            p.join(timeout=30)
        logger.info(
            f"{num_streams} streams on {chips} chips, target {args.fps} FPS each: processed {processed} images in "
            f"{elapsed:.1f} s = {processed / elapsed:.1f} images/s = {processed / elapsed / num_streams:.1f} FPS per "
            f"stream; last window {stats}; display {render}"
        )
        runner.release()
        ttnn.close_mesh_device(device)
        shm.unlink()  # the mapping itself goes with the process


if __name__ == "__main__":
    main()
