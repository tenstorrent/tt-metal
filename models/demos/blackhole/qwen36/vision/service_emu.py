# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""An emulation of the tt-d-gen encoder service (multimodal/tt_encoder, design doc "Multimedia encoder service";
d-gen PRs #2366 / #2409 not landed with an encode path yet) driving the Qwen3.5 media module (media.py), to measure a
split PUP prefill. What it reproduces, and how:

- The service owns the devices and is not part of the prefill runner: its own processes, its own chips
  (``--devices``), disjoint from the runner's (checked by the caller).
- Data-parallel encode: the devices are cut into groups of K chips; each group is one worker process
  (TT_VISIBLE_DEVICES = its K chips, as tt-d-gen's multimodal WorkerManager starts one worker per DEVICE_IDS group) and
  runs ONE image per encode() call (no batch > 1, Sumant 11-10). A request's images go round-robin to the workers; a
  worker with several images encodes them one after the other. ``batch=True`` (one call per worker with all its
  images) is kept only as a reference point.
- Per request (the prefill adapter's call): request limits (10 images, 20 MiB), media key = SHA-256(image bytes +
  media_key_salt(item) + optional nonce), store cache check (``is_exist``: a hit is not re-encoded), decode, prepare
  (resize by the REQUEST's image count: n_refs = len(images)), value_size, encode straight into a buffer of the
  writer's registered memory, ``put_from`` into the requesting prefill host's segment (preferred_segment, soft pin),
  and the reply only after every image is stored: per image {placeholder, runs, media_key}.
- Store: Mooncake (``MooncakeWriter``, the same calls as d-gen #2409 / tt-metal #64095's tests) when a master is given,
  else an in-process dict (host tests).

Decode + prepare run in the workers (in parallel over images); the doc's service decodes in its front, which would add
the serial decode / resize of all images to the request's critical path.

Process layout: ``python -m models.demos.blackhole.qwen36.vision.service_emu serve --devices 0,1,2,3 --k 2 ...`` = the
front (listens on --address for the adapter's calls) + one worker process per K-chip group.
"""

import argparse
import hashlib
import io
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Listener

import numpy as np

AUTHKEY = b"qwen35-media-svc"


@dataclass
class Limits:
    max_images: int = 10
    max_bytes: int = 20 << 20


# ---- store -------------------------------------------------------------------------------------------------------


class DictStore:
    """In-process stand-in for the embedding store (host tests): put copies the value, as a put does."""

    def __init__(self):
        self.values = {}

    def is_exist(self, key: str) -> bool:
        return key in self.values

    def put_from(self, key: str, value: np.ndarray, preferred_segment: str) -> None:
        self.values[key] = (bytes(value), preferred_segment)

    def fetch_batch(self, requests):  # EmbeddingReader, for the runner side of the tests
        return [memoryview(self.values[r.media_key.hex()][0]).toreadonly() for r in requests]


class MooncakeWriter:
    """The encoder's Mooncake client (d-gen #2409 store.py, with put_from from registered memory as the design doc
    says): no segment of its own, a registered pool the encode writes into, one soft-pinned copy per value on the
    requesting prefill host's segment."""

    def __init__(self, master: str, metadata: str, hostname: str, pool_bytes: int, protocol: str = "tcp"):
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig, SoftPinAction

        self._ReplicateConfig, self._SoftPinAction = ReplicateConfig, SoftPinAction
        self.client = MooncakeDistributedStore()
        rc = self.client.setup(hostname, metadata, 0, 16 << 20, protocol, "", master)
        if rc != 0:
            raise RuntimeError(f"Mooncake writer setup failed ({rc}) at {master}")
        self.pool = np.empty(pool_bytes, dtype=np.uint8)
        rc = self.client.register_buffer(self.pool.ctypes.data, pool_bytes)
        if rc != 0:
            raise RuntimeError(f"registering the {pool_bytes} B encode pool failed ({rc})")

    def is_exist(self, key: str) -> bool:
        return self.client.is_exist(key) == 1

    def put_from(self, key: str, value: np.ndarray, preferred_segment: str) -> None:
        cfg = self._ReplicateConfig()
        cfg.preferred_segment = preferred_segment
        cfg.soft_pin_action = self._SoftPinAction.ENABLE
        cfg.soft_pin_ttl_ms = int(os.environ.get("MEDIA_STORE_SOFT_PIN_MS", "300000"))
        rc = self.client.put_from(key, value.ctypes.data, value.nbytes, cfg)
        if rc != 0:
            raise RuntimeError(f"put_from {key[:16]} ({value.nbytes} B) to {preferred_segment} failed ({rc})")

    def close(self):
        self.client.close()


# ---- worker: one K-chip group --------------------------------------------------------------------------------------


def decode_image(data: bytes) -> np.ndarray:
    """The service's decode: EXIF orientation, alpha over white, RGB uint8 [H, W, 3] (flux3/pup.py's own handling)."""
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


class EncodeWorker:
    """One group of K chips: prepare + encode + store for the images the front gives it.

    Pipelined like a serving worker: every image's decode + prepare (CPU, PIL releases the GIL) is submitted to a thread
    pool at once, the device encodes image i as soon as it is prepared while the CPU prepares i+1.., and each value's
    put_from runs on a store thread while the next image encodes. One encode() call per image (unless batch)."""

    def __init__(self, media, store, pool: np.ndarray | None = None, prep_threads: int = 4):
        from concurrent.futures import ThreadPoolExecutor

        self.media = media
        self.store = store
        self.pool = pool if pool is not None else np.empty(64 << 20, dtype=np.uint8)
        self.prep = ThreadPoolExecutor(max_workers=prep_threads, thread_name_prefix="prep") if prep_threads else None
        self.putter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="put")

    def _prepare(self, job):
        idx, data, n_refs, nonce = job
        a = time.perf_counter()
        item = self.media.prepare("image", decode_image(data), n_refs=n_refs)
        key = hashlib.sha256(data + self.media.media_key_salt(item) + nonce).hexdigest()
        return idx, item, key, (time.perf_counter() - a) * 1e3

    def _put(self, key, buf, segment):
        a = time.perf_counter()
        self.store.put_from(key, buf, segment)
        return (time.perf_counter() - a) * 1e3

    def run(self, jobs, preferred_segment: str, batch: bool = False):
        """jobs: [(index, image bytes, n_refs, salt nonce)] -> ([(index, key hex, runs, grid, prepare ms)], timing)."""
        t0 = time.perf_counter()
        if self.prep is not None:
            futs = [self.prep.submit(self._prepare, j) for j in jobs]
            get = lambda i: futs[i].result()
        else:
            done = [self._prepare(j) for j in jobs]
            get = lambda i: done[i]
        off, prepared, enc_ms, puts, waits = 0, [], [], [], []
        hits = 0
        todo_batch = []
        for i in range(len(jobs)):
            a = time.perf_counter()
            p = get(i)
            waits.append((time.perf_counter() - a) * 1e3)
            prepared.append(p)
            _, item, key, _ = p
            if self.store.is_exist(key):
                hits += 1
                continue
            n = self.media.value_size(item)
            off = -(-off // 64) * 64
            if off + n > self.pool.nbytes:
                raise RuntimeError(f"encode pool of {self.pool.nbytes} B too small for this request")
            buf = self.pool[off : off + n]
            off += n
            if batch:
                todo_batch.append((item, key, buf))
                continue
            a = time.perf_counter()
            self.media.encode([item], [memoryview(buf)])
            enc_ms.append((time.perf_counter() - a) * 1e3)
            puts.append(self.putter.submit(self._put, key, buf, preferred_segment))
        if todo_batch:
            a = time.perf_counter()
            self.media.encode([t[0] for t in todo_batch], [memoryview(t[2]) for t in todo_batch])
            enc_ms.append((time.perf_counter() - a) * 1e3)
            puts += [self.putter.submit(self._put, k_, b_, preferred_segment) for _, k_, b_ in todo_batch]
        t_enc = time.perf_counter()
        put_ms = [f.result() for f in puts]
        t1 = time.perf_counter()
        out = [(idx, key, self.media.expansion(item), item.grid_thw, pm) for idx, item, key, pm in prepared]
        timing = {
            "wall_ms": (t1 - t0) * 1e3,
            "prepare_ms": sum(p[3] for p in prepared),  # CPU time of all decodes + prepares (overlapped)
            "prepare_wait_ms": sum(waits),  # time the device loop waited for a prepared image
            "encode_ms": sum(enc_ms),
            "encode_calls_ms": enc_ms,
            "put_ms": sum(put_ms),
            "put_tail_ms": (t1 - t_enc) * 1e3,  # store writes still running after the last encode
            "cache_hits": hits,
            "last": dict(getattr(getattr(self.media, "_encoder", None), "last", {}) or {}),
        }
        return out, timing


def worker_main(args):
    """Worker process: open this group's (1, K) mesh (the service owns it), load the media module, serve the front."""
    import json

    import ttnn
    from loguru import logger

    from models.demos.blackhole.qwen36.vision.media import Qwen35Media

    weights = args.weights
    cfg = json.load(open(os.path.join(weights, "config.json")))
    media = Qwen35Media(cfg)
    k = len(os.environ["TT_VISIBLE_DEVICES"].split(","))
    fabric = {1: None, 2: ttnn.FabricConfig.FABRIC_1D, 4: ttnn.FabricConfig.FABRIC_1D_RING}.get(k)
    if fabric is not None:
        ttnn.set_fabric_config(fabric, ttnn.FabricReliabilityMode.STRICT_INIT)
    p = media.device_params
    t0 = time.perf_counter()
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, k),
        l1_small_size=p["l1_small_size"],
        num_command_queues=p["num_command_queues"],
        trace_region_size=p["trace_region_size"],
    )
    store = None
    try:
        warm = [[tuple(int(v) for v in g.split("x"))] for g in args.warm.split(",") if g]
        media.load(mesh, weights, "image", {"trace": True, "warm_grids": warm, "stub_embeddings": args.stub})
        if args.master:
            store = MooncakeWriter(args.master, args.metadata, f"127.0.0.1:{args.port}", args.pool_mb << 20)
            pool = store.pool
        else:
            store, pool = DictStore(), None
        worker = EncodeWorker(media, store, pool)
        logger.info(f"MEDIA worker {args.index} on {os.environ['TT_VISIBLE_DEVICES']} (1,{k}) ready in "
                    f"{time.perf_counter() - t0:.1f} s")
        conn = Client(args.address, authkey=AUTHKEY)
        conn.send(("ready", args.index))
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                break
            _, jobs, segment, batch = msg
            try:
                conn.send(("ok",) + worker.run(jobs, segment, batch))
            except Exception as e:  # the front fails the request
                logger.exception("encode failed")
                conn.send(("error", repr(e)))
    finally:
        media.close()
        if isinstance(store, MooncakeWriter):
            store.close()
        ttnn.close_mesh_device(mesh)
        if fabric is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---- front -------------------------------------------------------------------------------------------------------


@dataclass
class Front:
    """The service's request path. workers: objects with submit(jobs, segment, batch) / result() (processes or, in
    tests, EncodeWorkers wrapped by LocalWorker)."""

    workers: list
    placeholder: int
    limits: Limits = field(default_factory=Limits)
    batch: bool = False

    def encode(self, images: list[bytes], preferred_segment: str, nonce: bytes = b"") -> dict:
        t0 = time.perf_counter()
        if not images:
            return {"error": "no images"}
        if len(images) > self.limits.max_images:
            return {"error": f"{len(images)} images exceed the limit of {self.limits.max_images}"}
        if sum(len(d) for d in images) > self.limits.max_bytes:
            return {"error": f"images total over the limit of {self.limits.max_bytes} B"}
        n = len(images)
        shares = [[] for _ in self.workers]
        for i, data in enumerate(images):
            shares[i % len(self.workers)].append((i, data, n, nonce))  # n_refs = this request's image count
        busy = [(w, s) for w, s in zip(self.workers, shares) if s]
        for w, s in busy:
            w.submit(s, preferred_segment, self.batch)
        items, timings, error = [None] * n, [], None
        for w, _ in busy:
            r = w.result()
            if r[0] != "ok":
                error = r[1]
                continue
            for idx, key, runs, grid, _ in r[1]:
                items[idx] = {"placeholder": self.placeholder, "runs": runs, "media_key": key, "grid": grid}
            timings.append(r[2])
        if error:
            return {"error": error}
        wall = (time.perf_counter() - t0) * 1e3
        return {"items": items, "timing": {"wall_ms": wall, "workers": timings}}


class LocalWorker:
    def __init__(self, worker: EncodeWorker):
        self.worker, self._r = worker, None

    def submit(self, jobs, segment, batch):
        self._r = ("ok",) + self.worker.run(jobs, segment, batch)

    def result(self):
        return self._r


class ProcWorker:
    def __init__(self, conn):
        self.conn = conn

    def submit(self, jobs, segment, batch):
        self.conn.send(("encode", jobs, segment, batch))

    def result(self):
        return self.conn.recv()


def _accept(listener, procs, conns, n):
    """Accept worker connections until n are in, failing if a worker process exits first."""
    import threading

    while len(conns) < n:
        box = []
        t = threading.Thread(target=lambda: box.append(listener.accept()), daemon=True)
        t.start()
        while t.is_alive():
            t.join(1.0)
            dead = [p for p in procs if p.poll() is not None]
            if dead and t.is_alive():
                raise RuntimeError(f"encoder worker exited with {dead[0].returncode} before it was ready")
        c = box[0]
        conns[c.recv()[1]] = c


def serve_main(args):
    """Front process: start one worker per K-chip group, then answer the adapters' calls on --address."""
    import json

    from loguru import logger

    devices = [d for d in args.devices.split(",") if d]
    if len(devices) % args.k:
        raise SystemExit(f"{len(devices)} devices do not split into groups of {args.k}")
    groups = [devices[i : i + args.k] for i in range(0, len(devices), args.k)]
    cfg = json.load(open(os.path.join(args.weights, "config.json")))
    worker_addr = args.address + ".w"
    if os.path.exists(worker_addr):
        os.unlink(worker_addr)
    wl = Listener(worker_addr, authkey=AUTHKEY)
    procs, conns = [], {}
    for i, g in enumerate(groups):
        env = dict(os.environ, TT_VISIBLE_DEVICES=",".join(g))
        if len(g) == 1:  # one chip: the tower's sequence-parallel path needs a fabric; a whole image per chip does not
            env["Q_DPV_SP"] = "0"
        if len(g) == 4 and args.ring_mgd:
            env["TT_MESH_GRAPH_DESC_PATH"] = args.ring_mgd
        else:
            env.pop("TT_MESH_GRAPH_DESC_PATH", None)
        cmd = [sys.executable, "-u", "-m", "models.demos.blackhole.qwen36.vision.service_emu", "worker",
               "--index", str(i), "--address", worker_addr, "--weights", args.weights, "--warm", args.warm,
               "--port", str(args.port_base + i), "--pool-mb", str(args.pool_mb)]
        if args.master:
            cmd += ["--master", args.master, "--metadata", args.metadata]
        if args.stub:
            cmd += ["--stub"]
        log = open(f"{args.log_prefix}w{i}.log", "w")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        if i == 0 and len(groups) > 1:  # the first worker compiles alone (shared JIT cache), then the rest start
            _accept(wl, procs, conns, 1)
    _accept(wl, procs, conns, len(groups))
    front = Front([ProcWorker(conns[i]) for i in range(len(groups))], cfg["image_token_id"], batch=args.batch)
    logger.info(f"MEDIA service: {len(groups)} worker(s) of {args.k} chip(s) on {groups}, batch={args.batch}")
    if os.path.exists(args.address):
        os.unlink(args.address)
    listener = Listener(args.address, authkey=AUTHKEY)
    open(args.address + ".ready", "w").close()
    try:
        while True:
            conn = listener.accept()
            try:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    if msg[0] == "stop":
                        return
                    _, images, segment, nonce = msg
                    conn.send(front.encode(images, segment, nonce))
            finally:
                conn.close()
    finally:
        for c in conns.values():
            try:
                c.send(("stop",))
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()


class ServiceClient:
    """The prefill adapter's handle on the service (stands in for the Dynamo client)."""

    def __init__(self, address: str):
        self.conn = Client(address, authkey=AUTHKEY)

    def encode(self, images: list[bytes], preferred_segment: str, nonce: bytes = b"") -> dict:
        self.conn.send(("encode", images, preferred_segment, nonce))
        return self.conn.recv()

    def stop(self):
        self.conn.send(("stop",))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=("serve", "worker"))
    ap.add_argument("--address", required=True)
    ap.add_argument("--weights", default=os.environ.get("HF_MODEL", ""))
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--warm", default="1x32x32,1x64x64,1x44x44")
    ap.add_argument("--master", default="")
    ap.add_argument("--metadata", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--port-base", type=int, default=17810)
    ap.add_argument("--pool-mb", type=int, default=96)
    ap.add_argument("--ring-mgd", default=os.environ.get("Q_RING4_MGD", ""))
    ap.add_argument("--log-prefix", default="/tmp/media_svc_")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args(argv)
    (serve_main if args.role == "serve" else worker_main)(args)


if __name__ == "__main__":
    main()
