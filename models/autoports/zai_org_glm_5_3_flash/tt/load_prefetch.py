# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14b: parallel weight prefetch for the 45-layer TP load (coordinator 2026-10-20 15:16Z: 10 min for ~174 GB).

The tile cache lives on the NFS-over-RDMA share: one reader gets ~0.8 GB/s, 16-32 concurrent readers 5.7-7.4 GB/s
(measured 15:20Z, cold layer-43 expert files). The model constructor reads its files one by one, so the load is
read-latency bound. `LayerPrefetcher` runs a thread pool AHEAD of the constructor, a bounded number of layers at a time:
every manifest file of the layer is read once (-> page cache, so the constructor's own reads hit memory) and every
routed expert's fused_experts gate_up permutation (tt/ep_experts.py) is written ahead into the scratch area the loader
takes it from (python file reads / numpy release the GIL). `wait(layer)` blocks until that layer's files are in, and
schedules `ahead` layers further. Nothing else changes: the constructor still reads, checks and uploads every tensor.
GLM53_LOAD_THREADS (32), GLM53_LOAD_AHEAD (2 layers; ~4 GB of page cache each)."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

READ_CHUNK = 8 << 20


def _read(path: str) -> int:
    n = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(READ_CHUNK)
            if not b:
                return n
            n += len(b)


def transient_gate_up(layer: int, e: int) -> str:
    """the scratch path of expert e's permuted gate_up (shared with tt/ep_experts.load_ep_experts)"""
    tmp = os.environ.get("GLM53_EP_SCRATCH", "/home/alex/glm53/scratch-big/14b")
    return os.path.join(tmp, f"ep_gate_up.{os.getpid()}.L{layer:02d}.{e:03d}.tensorbin")


class LayerPrefetcher:
    def __init__(self, source, layers, *, expert_layers=(), num_experts: int = 288, fused_cache_dir: str = "",
                 threads: int | None = None, ahead: int | None = None):  # fmt: skip
        self.source, self.layers = source, list(layers)
        self.expert_layers = set(expert_layers)
        self.num_experts, self.fused_cache_dir = num_experts, fused_cache_dir
        self.ahead = int(ahead or os.environ.get("GLM53_LOAD_AHEAD", "2"))
        self.pool = ThreadPoolExecutor(int(threads or os.environ.get("GLM53_LOAD_THREADS", "32")))
        self.futs: dict[int, list] = {}
        self.bytes = 0
        self.lock = threading.Lock()
        self.errors: list = []
        for L in self.layers[: self.ahead]:
            self._schedule(L)

    def _task_read(self, path: str):
        try:
            n = _read(path)
            with self.lock:
                self.bytes += n
        except OSError as e:  # the constructor reads it again and fails closed on its own
            self.errors.append(repr(e)[:200])

    def _task_permute(self, layer: int, e: int):
        from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import _permuted_gate_up_file

        try:
            ge = self.source._entry(layer, f"mlp.experts.{e}.gate_up_proj")
            shared = os.path.join(self.fused_cache_dir, f"L{layer:02d}",
                                  f"{e:03d}.gate_up_proj.{ge['sha256'][:16]}.tensorbin")  # fmt: skip
            if os.path.exists(shared):
                self._task_read(shared)
                return
            dst = transient_gate_up(layer, e)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                _permuted_gate_up_file(os.path.join(self.source.cache_dir, ge["path"]), dst, ge["shape"], 576)
        except Exception as ex:  # noqa: BLE001 -- the loader permutes it itself then
            self.errors.append(repr(ex)[:200])

    def _schedule(self, L: int):
        if L in self.futs:
            return
        pre = f"layers.{L}."
        paths, fs = [], []
        for k, ent in self.source.tensors.items():
            if not k.startswith(pre):
                continue
            if L in self.expert_layers and ".mlp.experts." in k and k.endswith("gate_up_proj"):
                continue  # read by the permutation task
            paths.append(os.path.join(self.source.cache_dir, ent["path"]))
        fs += [self.pool.submit(self._task_read, p) for p in paths]
        if L in self.expert_layers:
            fs += [self.pool.submit(self._task_permute, L, e) for e in range(self.num_experts)]
        self.futs[L] = fs

    def wait(self, L: int):
        """block until layer L's files are in (scheduling it if needed) and schedule `ahead` layers past it"""
        self._schedule(L)
        for f in self.futs[L]:
            f.result()
        i = self.layers.index(L) if L in self.layers else -1
        for M in self.layers[i + 1 : i + 1 + self.ahead]:
            self._schedule(M)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        # permuted files the loader did not consume (an aborted build): remove this process's leftovers
        tmp = os.path.dirname(transient_gate_up(0, 0))
        if os.path.isdir(tmp):
            for f in os.listdir(tmp):
                if f.startswith(f"ep_gate_up.{os.getpid()}."):
                    try:
                        os.remove(os.path.join(tmp, f))
                    except OSError:
                        pass
