# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 12 for the TP production stack (Stage 14b's StackBackend, PRODUCTION_PATH.md M5 / M6): a per-chip DRAM image of
the TP / EP shard layout, the device-resident restart, and the replay of the loader's allocations.

Mesh tensors of a TP mesh are allocated lock-step: one address per tensor on every chip, per-chip contents. So an
image = one list of DRAM segments (the per-bank extents of every device tensor the backend holds) + one bank-major file
per chip (`chip<device id>.img`, the chip's bytes of those segments).

Key (a code change must not invalidate it): weights manifest sha + TP_LAYOUT (the shard
rule's version) + mesh shape + the ALLOCATION MAP itself (every recorded upload: kind, spec, address; its sha names the
image directory). A code change that keeps every weight allocation and the layout keeps the image; one that changes an
allocation fails the replay at that upload (ImageMismatch -> teardown -> the tensor loader).

record (TPWeightIO("record")): the normal load runs; every host -> device upload (FdTileCacheSource._load(device),
  ttnn.from_torch(device=), ttnn.to_device) is logged in call order with its spec and address; the host-side weight
  reads (FdTileCacheSource.host, ttnn.load_tensor, the sliced-expert readers) are logged with their specs, small host
  tensors kept (host_small.pt).
replay (TPWeightIO("replay", rows)): the SAME constructors run, but no weight file is read and nothing is uploaded:
  big host weights are torch meta tensors (shape-only host math), host weight files are HostPH placeholders, and an
  upload whose payload is a placeholder becomes ttnn.allocate_tensor_on_device with the recorded spec; every upload is
  checked against the recorded address. Device ops of the constructors run as usual (derived tensors: recomputed from
  resident inputs or overwritten by the pull). An L1 tensor or an unimaged DRAM tensor made from a placeholder fails the
  replay (its bytes would never be restored).
restore (restore_backend): pre-replay device block hashes on every chip (DRAM survives a process exit without a reset)
  -> per-chip partial pulls of the stale blocks (ImagePuller at the chip's mesh coordinate of the PARENT mesh: never 1x1
  submeshes of a mesh that holds allocations) -> replay -> post-replay hashes -> pull what the constructors clobbered ->
  final hashes == image. A chip that was reset or used by another layout degenerates to the full pull.
pull sets: both pulls repeat from one restart to the next -> recorded next to the image (pullset_<k>k.json; before the
  first restart: predicted from the map) and prefetched into the page cache by a child process while the restore
  hashes / replays (the images live on NFS: ~1.3-1.5 GB/s; the page cache ~10 GB/s); after a verified restore the chips
  are hashed once more at GLM53_IMAGE_FINE_BLOCK (256 KiB; blocks_<k>k.npz) and later restores pull those finer blocks.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import torch

import ttnn

FORMAT = "glm53-tp-dram-image-v1"
DEFAULT_ROOT = "/mnt/fab/mpi-shfs/tenstorrent/glm53-flash-bringup-data/dram_images/tp"
BLOCK = 4 << 20


class ImageMismatch(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[tp-image] {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def _di():
    return ttnn._ttnn.dram_image


# ================================================================================================ placeholders
class HostPH:
    """A host weight tensor that was not read (replay): its spec only. `n` > 1: a multi-shard host tensor
    (ttnn.from_host_shards / a mesh-mapped ttnn.from_torch) whose shards share this spec."""

    def __init__(self, shape, dtype, layout, n: int = 1, key: str = "", data: Optional[torch.Tensor] = None):
        self.shape, self.dtype, self.layout, self.n, self.key = list(shape), dtype, layout, int(n), key
        self.data = data  # a small host copy kept in the image: ttnn.to_torch returns it

    def storage_type(self):
        return ttnn.StorageType.HOST

    def __repr__(self):
        return f"HostPH({self.shape}, {self.dtype}, {self.layout}, n={self.n}, {self.key})"


def _spec(t) -> str:
    return f"{list(t.shape)}|{t.dtype}|{t.layout}|{t.memory_config()}"


def _hspec(t) -> dict:
    return dict(shape=[int(v) for v in t.shape], dtype=str(t.dtype), layout=str(t.layout))


def _dtype(s):
    """'DataType.BFLOAT8_B' / 'bfloat8_b' -> ttnn.DataType"""
    n = str(s).split(".")[-1].upper()
    return getattr(ttnn.DataType, n)


def _layout(s):
    """'Layout.TILE' / 'TILE_LAYOUT' / 'TILE' -> ttnn.Layout"""
    n = str(s).split(".")[-1].upper()
    n = n[: -len("_LAYOUT")] if n.endswith("_LAYOUT") else n
    return getattr(ttnn.Layout, n)


def _meta_of(ph: "HostPH") -> torch.Tensor:
    """ttnn.to_torch of an unread host tensor: its kept small copy, else a meta tensor of its logical shape (bfloat8_b / bfloat4_b / float32 ->
    float32, bfloat16 -> bfloat16, int / uint32 -> int32, uint8 -> uint8)"""
    if ph.data is not None:
        return ph.data.clone()
    n = str(ph.dtype).split(".")[-1].upper()
    dt = {"BFLOAT16": torch.bfloat16, "INT32": torch.int32, "UINT32": torch.int32, "UINT16": torch.int32,
          "UINT8": torch.uint8}.get(n, torch.float32)  # fmt: skip
    return torch.empty(ph.shape, dtype=dt, device="meta")


_TORCH = {str(v): v for v in (torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.uint8,
                              torch.int8, torch.int16, torch.bool, torch.float64)}  # fmt: skip


# ================================================================================================ record / replay
class TPWeightIO:
    """Context manager: record / replay the TP loader's host reads and uploads (see the module docstring)."""

    def __init__(self, mode: str, rows: Optional[dict] = None, host_small: Optional[dict] = None,
                 real_tile16: Optional[bool] = None, ops: Optional[list] = None):  # fmt: skip
        if mode not in ("record", "replay"):
            raise ValueError(mode)
        self.mode = mode
        # replay bf16 / fp32 TILE uploads of unread weights through the real call on zeros (the image's setting)
        self.real_tile16 = (
            os.environ.get("GLM53_IMAGE_REAL_TILE16") == "1" if real_tile16 is None else bool(real_tile16)
        )
        exp = rows or {}
        self.exp_up = list(exp.get("uploads", []))  # recorded uploads, call order
        self.exp_host = list(exp.get("host", []))  # recorded host tensor specs (load_tensor / xsl / from_torch host)
        self.exp_meta = dict(exp.get("meta", {}))  # big FdTileCacheSource.host results: key -> (shape, torch dtype)
        self.host_small: dict = dict(host_small or {})
        self.uploads: list = []
        self.host: list = []
        self.meta: dict = {}
        self.placeholder_allocs: list = []  # (index, address, spec, buffer_type) of uploads made from placeholders
        self.host_misses = 0
        self.alloc_s = 0.0
        # op records (OP_SKIP): every top-level ttnn op of the constructing thread, in call order (record: kept; replay:
        # an op with an input made from unread weights is not run - its recorded outputs are allocated instead)
        self.op_rows: list = []
        self.exp_ops = list(ops or [])
        self.op_skip = self.mode == "replay" and bool(self.exp_ops)
        self.ph_addr: set = set()  # device addresses whose content is not the weights' (placeholders, skipped outputs)
        self.ph_src: dict = {}  # replay: address of a placeholder loaded straight from the cache -> (source, entry)
        self.op_k = 0
        self.op_stats = dict(calls=0, skipped=0, skip_bytes=0, fail=0, ph_input_run=0, name_mismatch=0)
        self.op_why: dict = {}  # (op name, why it ran) -> count, for ops fed by unread weights
        # replay: DRAM uploads of REAL (kept) host tensors are allocations only (their bytes come from the image)
        self.emulate_real = self.mode == "replay" and os.environ.get("GLM53_IMAGE_EMULATE_REAL", "0") == "1"
        self._opd = threading.local()
        self._patched: list = []
        self._depth = threading.local()

    # ---- bookkeeping
    def _up(self, kind: str, key: str, t, placeholder: bool) -> None:
        i = len(self.uploads)
        row = dict(kind=kind, key=key, address=int(t.buffer_address()), spec=_spec(t),
                   extent=int(_di().buffer_info(t)["aligned_size_per_bank"]))  # fmt: skip
        if os.environ.get("GLM53_IMAGE_TRACE_STACK") == "1":  # debugging a replay divergence: the upload's call site
            import traceback

            row["where"] = [f"{os.path.basename(f.filename)}:{f.lineno}" for f in traceback.extract_stack(limit=9)[:-3]]
        if self.mode == "replay":
            if i >= len(self.exp_up):
                raise ImageMismatch(f"upload #{i} {kind} {key} has no recorded counterpart")
            e = self.exp_up[i]
            if (e["kind"], e["address"], e["spec"], e.get("extent", row["extent"])) != (
                row["kind"], row["address"], row["spec"], row["extent"]):  # fmt: skip
                raise ImageMismatch(f"upload #{i} {kind} {key} @ {row['address']:#x} {row['spec']} != recorded "
                                    f"{e['kind']} {e['key']} @ {e['address']:#x} {e['spec']}")  # fmt: skip
            if placeholder:
                self.placeholder_allocs.append((i, row["address"], row["spec"]))
            (self.ph_addr.add if placeholder else self.ph_addr.discard)(row["address"])
        self.uploads.append(row)

    def _host(self, kind: str, key: str, spec: dict):
        """record: log a host weight tensor's spec; replay: the recorded spec of this call (checked)."""
        i = len(self.host)
        if self.mode == "replay":
            if i >= len(self.exp_host):
                raise ImageMismatch(f"host tensor #{i} {kind} {key} has no recorded counterpart")
            e = self.exp_host[i]
            if e["kind"] != kind or (spec and spec != e["spec"]):
                raise ImageMismatch(f"host tensor #{i} {kind} {key} {spec} != recorded {e}")
            spec = e["spec"]
        self.host.append(dict(kind=kind, key=key, spec=spec))
        return spec

    def _emulate_upload(self, device, memory_config):
        """replay, a DRAM upload of a REAL (kept) host tensor: its bytes are in the image, so only its allocations are
        reproduced - a bf16 / fp32 TILE upload as its ROW_MAJOR device temporary + the TILE output (temporary freed),
        others as the output alone; kept only on the recorded address / spec (else undone -> None: the real call).
        The result counts as placeholder content (consumers' ops are skipped; restored / checked like the rest)."""
        i = len(self.uploads)
        e = self.exp_up[i] if i < len(self.exp_up) else None
        if e is None or e["kind"] != "from_torch" or "BufferType::DRAM" not in e["spec"]:
            return None
        shape, dt, lay = e["spec"].split("|")[:3]
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        tiled16 = dt.split(".")[-1] in ("BFLOAT16", "FLOAT32") and lay.split(".")[-1] == "TILE"
        t0 = time.perf_counter()
        for with_tmp in (True, False) if tiled16 else (False,):
            tmp = (ttnn.allocate_tensor_on_device(ttnn.Shape(json.loads(shape)), _dtype(dt), ttnn.ROW_MAJOR_LAYOUT,
                                                  device, mc) if with_tmp else None)  # fmt: skip
            out = ttnn.allocate_tensor_on_device(ttnn.Shape(json.loads(shape)), _dtype(dt), _layout(lay), device, mc)
            if tmp is not None:
                ttnn.deallocate(tmp)
            if int(out.buffer_address()) == e["address"] and _spec(out) == e["spec"]:
                self.alloc_s += time.perf_counter() - t0
                self.op_stats["real_emulated"] = self.op_stats.get("real_emulated", 0) + 1
                self._up("from_torch", "", out, True)
                return out
            ttnn.deallocate(out)
        self.alloc_s += time.perf_counter() - t0
        self.op_stats["real_uploaded"] = self.op_stats.get("real_uploaded", 0) + 1
        return None

    def _alloc(self, mesh, spec: dict, memory_config):
        if memory_config is None:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
        if memory_config.buffer_type != ttnn.BufferType.DRAM:
            raise ImageMismatch(f"an L1 tensor {spec} from an unread weight: DRAM images cannot restore it")
        t0 = time.perf_counter()
        t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(spec["shape"])), _dtype(spec["dtype"]),
                                           _layout(spec["layout"]), mesh, memory_config)  # fmt: skip
        self.alloc_s += time.perf_counter() - t0
        return t

    # ---- patches
    def __enter__(self):
        from models.autoports.zai_org_glm_5_3_flash.tt import fused_decoder as fd
        from models.autoports.zai_org_glm_5_3_flash.tt import optimized_decoder as od

        io, rep = self, self.mode == "replay"
        o_load, o_host = od.FdTileCacheSource._load, od.FdTileCacheSource.host
        o_load_tensor, o_from_torch, o_to_device = ttnn.load_tensor, ttnn.from_torch, ttnn.to_device
        o_shards, o_copy = ttnn.from_host_shards, ttnn.copy_host_to_device_tensor
        o_to_torch = ttnn.to_torch
        o_xsl_bytes, o_xsl_host = getattr(fd, "_xsl_slice_bytes", None), getattr(fd, "_xsl_host_tensor", None)
        o_xsl_multi = getattr(fd, "_xsl_slices_bytes", None)  # sliced experts: one read per expert file
        o_perm = getattr(fd, "_permuted_gate_up_file", None)
        inner = lambda: getattr(io._depth, "n", 0) > 0  # noqa: E731 (inside a wrapped call: pass through)

        @contextlib.contextmanager
        def nested():
            io._depth.n = getattr(io._depth, "n", 0) + 1
            try:
                yield
            finally:
                io._depth.n -= 1

        def host(src, layer_idx, name):
            key = f"layers.{layer_idx}.{name}"
            if rep:
                if key in io.host_small:
                    return io.host_small[key].clone()
                if key in io.exp_meta:
                    shape, dt = io.exp_meta[key]
                    return torch.empty(shape, dtype=_TORCH[dt], device="meta")
                io.host_misses += 1
            with nested():
                t = o_host(src, layer_idx, name)
            if not rep:
                if t.numel() * t.element_size() <= K.HOST_KEEP:
                    io.host_small[key] = t.clone()
                else:
                    io.meta[key] = ([int(v) for v in t.shape], str(t.dtype))
            return t

        def _load(src, e, device=None):
            if device is None and inner():  # host()'s own read (host() is logged itself)
                return o_load(src, e)
            if (
                device is None
            ):  # a host read outside host() (lm_head / embedding / norm host copies): not read in a replay
                if rep:
                    io._host("source_load", e["path"], None)
                    return HostPH(e["shape"], _dtype(e["dtype"]), _layout(e["layout"]), key=e["path"],
                                  data=io.host_small.get("load:" + e["path"]))  # fmt: skip
                with nested():
                    t = o_load(src, e)
                io._host("source_load", e["path"], _hspec(t))
                tt = o_to_torch(t)
                if tt.numel() * tt.element_size() <= K.HOST_KEEP:  # kept: a replay hands back the real values
                    io.host_small["load:" + e["path"]] = tt.clone()
                return t
            if rep:
                t = io._alloc(device, dict(shape=e["shape"], dtype=e["dtype"], layout=f"{e['layout']}_LAYOUT"),
                              ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
                io.ph_src[int(t.buffer_address())] = (src, e)  # host0 below: the weight file, read on the host
            else:
                with nested():
                    t = o_load(src, e, device)
            io._up("load", e["path"], t, rep)
            return t

        def load_tensor(path, *a, device=None, **kw):
            if inner():
                return o_load_tensor(path, *a, device=device, **kw)
            key = os.path.basename(str(path))
            if device is not None:  # a direct device load: an upload (DRAM interleaved only in a replay)
                if rep:
                    e = io.exp_up[len(io.uploads)] if len(io.uploads) < len(io.exp_up) else None
                    if (
                        e is None
                        or e["kind"] != "load_tensor"
                        or "INTERLEAVED" not in e["spec"]
                        or "DRAM" not in e["spec"]
                    ):
                        raise ImageMismatch(f"upload #{len(io.uploads)} load_tensor({key}) vs recorded {e}")
                    shape, dt, lay = e["spec"].split("|")[:3]
                    t = io._alloc(device, dict(shape=json.loads(shape), dtype=dt, layout=lay), ttnn.DRAM_MEMORY_CONFIG)
                else:
                    with nested():
                        t = o_load_tensor(path, *a, device=device, **kw)
                io._up("load_tensor", key, t, rep)
                return t
            if rep:
                spec = io._host("load_tensor", key, None)
                return HostPH(spec["shape"], _dtype(spec["dtype"]), _layout(spec["layout"]), key=key)
            t = o_load_tensor(path, *a, **kw)
            io._host("load_tensor", key, _hspec(t))
            return t

        def xsl_bytes(*a, **kw):  # replay: the expert slice is not read
            return b""

        def xsl_multi(src, shape, tp, slices, kind):  # replay: no expert file is read
            return [b""] * len(list(slices))

        def xsl_host(data, shape, tag):
            # a sliced expert's host tensor: bfloat4_b TILE of `shape` by construction (fused_decoder._xsl_host_tensor);
            # NOT logged: the sliced loader builds these in a thread pool (order-free), the uploads stay in order
            if rep:
                return HostPH([int(v) for v in shape], ttnn.bfloat4_b, ttnn.TILE_LAYOUT, key=str(tag))
            with nested():
                return o_xsl_host(data, shape, tag)

        def from_host_shards(tensors, *a, **kw):
            ts = list(tensors)
            if ts and all(isinstance(t, HostPH) for t in ts):
                s0 = ts[0]
                if any((t.shape, t.dtype, t.layout) != (s0.shape, s0.dtype, s0.layout) for t in ts):
                    raise ImageMismatch("from_host_shards of placeholders with different specs")
                return HostPH(s0.shape, s0.dtype, s0.layout, n=len(ts), key=s0.key)
            if any(isinstance(t, HostPH) for t in ts):
                raise ImageMismatch("from_host_shards mixes read and unread host tensors")
            return o_shards(tensors, *a, **kw)

        def to_device(t, device, *a, **kw):
            if inner() or device is None:
                return o_to_device(t, device, *a, **kw)
            if isinstance(t, HostPH):
                spec = dict(shape=t.shape, dtype=str(t.dtype), layout=str(t.layout))
                mc = kw.get("memory_config", a[0] if a else None)
                out = io._alloc(device, spec, mc)
                io._up("to_device", t.key, out, True)
                return out
            with nested():
                out = o_to_device(t, device, *a, **kw)
            if hasattr(out, "buffer_address") and out.storage_type() == ttnn.StorageType.DEVICE:
                io._up("to_device", "", out, False)
            return out

        def from_torch(tensor, *a, device=None, memory_config=None, **kw):
            if inner():
                return o_from_torch(tensor, *a, device=device, memory_config=memory_config, **kw)
            meta = isinstance(tensor, torch.Tensor) and tensor.is_meta
            if device is None:
                if rep and meta:
                    spec = io._host("from_torch", "", None)
                    return HostPH(spec["shape"], _dtype(spec["dtype"]), _layout(spec["layout"]))
                with nested():
                    out = o_from_torch(tensor, *a, memory_config=memory_config, **kw)
                io._host("from_torch", "", _hspec(out))
                return out
            if rep and meta:
                e = io.exp_up[len(io.uploads)] if len(io.uploads) < len(io.exp_up) else None
                if e is None or e["kind"] != "from_torch":
                    raise ImageMismatch(f"upload #{len(io.uploads)} from_torch has no recorded counterpart ({e})")
                shape, dt, lay = e["spec"].split("|")[:3]
                if (io.real_tile16 and dt.split(".")[-1] in ("BFLOAT16", "FLOAT32")
                        and lay.split(".")[-1] == "TILE"):  # fmt: skip
                    # opt-in: the REAL call on zeros - a bf16 / fp32 TILE upload allocates device temporaries (tilize)
                    # that a direct allocate does not, i.e. the same allocator history (lets GLM53_IMAGE_HOST_KEEP drop)
                    if memory_config is not None and memory_config.buffer_type != ttnn.BufferType.DRAM:
                        raise ImageMismatch(f"an L1 tensor {e['spec'][:80]} from an unread weight: DRAM images cannot "
                                            f"restore it")  # fmt: skip
                    if os.environ.get("GLM53_IMAGE_RT16_EMULATE", "1") == "1":
                        # the same allocator history without the host conversion or any write: a ROW_MAJOR device
                        # temporary of the rows, the TILE output, the temporary freed - kept only when the output lands
                        # on the recorded address / spec (else undone: the real call below)
                        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
                        t0 = time.perf_counter()
                        tmp = ttnn.allocate_tensor_on_device(ttnn.Shape(json.loads(shape)), _dtype(dt),
                                                             ttnn.ROW_MAJOR_LAYOUT, device, mc)  # fmt: skip
                        out = ttnn.allocate_tensor_on_device(ttnn.Shape(json.loads(shape)), _dtype(dt), _layout(lay),
                                                             device, mc)  # fmt: skip
                        ttnn.deallocate(tmp)
                        io.alloc_s += time.perf_counter() - t0
                        if int(out.buffer_address()) == e["address"] and _spec(out) == e["spec"]:
                            io.op_stats["rt16_emulated"] = io.op_stats.get("rt16_emulated", 0) + 1
                            io._up("from_torch", "", out, True)
                            return out
                        ttnn.deallocate(out)
                        io.op_stats["rt16_real"] = io.op_stats.get("rt16_real", 0) + 1
                    z = torch.zeros(tuple(tensor.shape), dtype=tensor.dtype)
                    with nested():
                        out = o_from_torch(z, *a, device=device, memory_config=memory_config, **kw)
                    io._up("from_torch", "", out, True)
                    return out
                out = io._alloc(device, dict(shape=json.loads(shape), dtype=dt, layout=lay), memory_config)
                io._up("from_torch", "", out, True)
                return out
            if meta:
                raise ImageMismatch("from_torch of a meta tensor while recording")
            if rep and io.emulate_real and (memory_config is None or memory_config.buffer_type == ttnn.BufferType.DRAM):
                out = io._emulate_upload(device, memory_config)  # the image holds these bytes: allocate, don't write
                if out is not None:
                    return out
            with nested():
                out = o_from_torch(tensor, *a, device=device, memory_config=memory_config, **kw)
            io._up("from_torch", "", out, False)
            return out

        def to_torch(t, *a, **kw):
            if isinstance(t, HostPH):
                return _meta_of(t)
            return o_to_torch(t, *a, **kw)

        def copy_host_to_device_tensor(src, dst, *a, **kw):
            if isinstance(src, HostPH):
                if rep:
                    if dst.memory_config().buffer_type != ttnn.BufferType.DRAM:
                        raise ImageMismatch(f"a copy of an unread weight into an L1 tensor {_spec(dst)[:80]}")
                    io.placeholder_allocs.append((-1, int(dst.buffer_address()), _spec(dst)))
                    io.ph_addr.add(int(dst.buffer_address()))
                    return None
                raise ImageMismatch("copy of a placeholder while recording")
            return o_copy(src, dst, *a, **kw)

        # ---- top-level ttnn ops of the constructing thread (FastOperation: ttnn's fast runtime mode)
        dec = ttnn.decorators
        o_call = dec.FastOperation.__call__
        tid = threading.get_ident()
        deny = ("from_torch", "to_torch", "to_device", "from_device", "deallocate", "allocate_tensor_on_device",
                "load_tensor", "dump_tensor", "synchronize", "copy_host_to_device", "get_device_tensors",
                "from_host_shards", "reallocate", "begin_trace", "end_trace", "execute_trace", "record_event",
                "wait_for_event")  # fmt: skip

        def _dev(t):
            return isinstance(t, ttnn.Tensor) and t.storage_type() == ttnn.StorageType.DEVICE and t.is_allocated()

        def _flat(res):
            if isinstance(res, ttnn.Tensor):
                return [res], "t"
            if isinstance(res, (list, tuple)) and res and all(isinstance(x, ttnn.Tensor) for x in res):
                return list(res), ("l" if isinstance(res, list) else "u") + str(len(res))
            return [], "x"

        def _mcs(args, kw, ins):
            out = [ttnn.DRAM_MEMORY_CONFIG]
            for v in list(args) + list(kw.values()):
                for x in v if isinstance(v, (list, tuple)) else [v]:
                    if isinstance(x, ttnn.MemoryConfig):
                        out.append(x)
            return out + [t.memory_config() for t in ins]

        def op_call(op, *args, **kw):
            name = getattr(op, "python_fully_qualified_name", "?")
            top = (threading.get_ident() == tid and not inner() and getattr(io._opd, "n", 0) == 0
                   and not any(d in name for d in deny))  # fmt: skip
            if not top:
                io._opd.n = getattr(io._opd, "n", 0) + 1
                try:
                    return o_call(op, *args, **kw)
                finally:
                    io._opd.n -= 1
            ins = [t for t in dec.get_all_tensors((args, kw)) if _dev(t)]
            in_addr = [int(t.buffer_address()) for t in ins]
            k = io.op_k
            io.op_k += 1
            io.op_stats["calls"] += 1
            ph_in = rep and any(a in io.ph_addr for a in in_addr)
            if ph_in and io.op_skip:
                e = io.exp_ops[k] if k < len(io.exp_ops) else None
                if e is None or e["name"] != name:
                    io.op_stats["name_mismatch"] += 1
                    io.op_skip = False  # out of step with the record: run everything from here on
                elif e["struct"] != "x" and e["outs"] and all(o["address"] not in in_addr and "BufferType::DRAM" in
                                                               o["spec"] for o in e["outs"]):  # fmt: skip
                    cands, made, ok = _mcs(args, kw, ins), [], True
                    mesh = ins[0].device()
                    why = None
                    for o in e["outs"]:
                        shape, dt, lay, mcs = o["spec"].split("|", 3)
                        # the op's own memory config (an argument / an input's) may print differently from the
                        # normalized one its output carries: try each, keep the first whose tensor matches the record
                        order = sorted(cands, key=lambda c: str(c) != mcs)
                        t, why = None, "memory config"
                        for mc in order:
                            try:
                                c = ttnn.allocate_tensor_on_device(ttnn.Shape(json.loads(shape)), _dtype(dt),
                                                                   _layout(lay), mesh, mc)  # fmt: skip
                            except Exception:  # noqa: BLE001 (a config this shape cannot take)
                                continue
                            if _spec(c) == o["spec"] and int(c.buffer_address()) == o["address"]:
                                t = c
                                break
                            why = "address" if _spec(c) == o["spec"] else "spec"
                            ttnn.deallocate(c)
                        if t is None:
                            ok = False
                            break
                        made.append(t)
                    if ok:
                        io.op_stats["skipped"] += 1
                        io.op_stats["skip_bytes"] += sum(o.get("extent", 0) for o in e["outs"])
                        for o in e["outs"]:
                            io.ph_addr.add(o["address"])
                        st = e["struct"]
                        return made[0] if st == "t" else (made if st[0] == "l" else tuple(made))
                    for t in made:  # undo: the real op allocates from the same free list
                        ttnn.deallocate(t)
                    io.op_stats["fail"] += 1
                    io.op_why[(name, why)] = io.op_why.get((name, why), 0) + 1
                else:
                    io.op_why[(name, "not skippable: " + e["struct"])] = (
                        io.op_why.get((name, "not skippable: " + e["struct"]), 0) + 1
                    )
            io._opd.n = 1
            try:
                res = o_call(op, *args, **kw)
            finally:
                io._opd.n = 0
            outs, st = _flat(res)
            outs = [t for t in outs if _dev(t)]
            if io.mode == "record":
                io.op_rows.append(dict(name=name, struct=st, outs=[dict(
                    address=int(t.buffer_address()), spec=_spec(t), extent=int(_di().buffer_info(t)["aligned_size_per_bank"]))
                    for t in outs]))  # fmt: skip
            elif ph_in:
                io.op_stats["ph_input_run"] += 1
            for t in outs:
                a = int(t.buffer_address())
                if a not in in_addr:
                    (io.ph_addr.add if ph_in else io.ph_addr.discard)(a)
            return res

        if (not rep and K.OP_RECORD) or (rep and self.op_skip):
            dec.FastOperation.__call__ = op_call
            self._patched.append((dec.FastOperation, "__call__", o_call))
        if rep and self.op_skip and os.environ.get("GLM53_IMAGE_HOST0_SKIP", "1") == "1":
            # mk_mla._host0 (TP MLA: the replicated q_b / o_proj / W_uk / W_uv read back to be re-sharded on the host)
            # of a tensor whose content is not the weights' in a replay: no device read, zeros of its shape (what it
            # holds is not the weights either; everything made from it lands in the image and is restored / checked).
            # Not meta tensors: the q_b norm fold multiplies them by a real norm weight.
            from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as _mk

            o_host0 = _mk._host0

            def host0(t):
                if _dev(t) and int(t.buffer_address()) in io.ph_addr:
                    # a weight loaded straight from the cache (e.g. W_uk / W_uv): its FILE read on the host (real bytes,
                    # no device read). Host copies made here outlive the replay (MeshHeadShard.host_weights): the TP
                    # prefill attached after the restore uploads them (zeros = W_uk = W_uv = 0 in every prefill MLA
                    # layer of a restored process, coordinator 09:42Z). Derived tensors (skipped ops' outputs): zeros.
                    ps = io.ph_src.get(int(t.buffer_address()))
                    if ps is not None and K.HOST0_SOURCE:
                        with nested():
                            h = o_to_torch(o_load(*ps)).float()
                        if tuple(h.shape) == tuple(int(v) for v in t.shape):
                            io.op_stats["host0_source"] = io.op_stats.get("host0_source", 0) + 1
                            return h
                    io.op_stats["host0_skipped"] = io.op_stats.get("host0_skipped", 0) + 1
                    return torch.zeros(tuple(int(v) for v in t.shape), dtype=torch.float32)
                return o_host0(t)

            _mk._host0 = host0
            self._patched.append((_mk, "_host0", o_host0))
        od.FdTileCacheSource._load = _load
        od.FdTileCacheSource.host = host
        ttnn.load_tensor, ttnn.from_torch, ttnn.to_device = load_tensor, from_torch, to_device
        ttnn.from_host_shards, ttnn.copy_host_to_device_tensor = from_host_shards, copy_host_to_device_tensor
        ttnn.to_torch = to_torch
        self._patched += [(od.FdTileCacheSource, "_load", o_load), (od.FdTileCacheSource, "host", o_host),
                         (ttnn, "load_tensor", o_load_tensor), (ttnn, "from_torch", o_from_torch),
                         (ttnn, "to_device", o_to_device), (ttnn, "from_host_shards", o_shards),
                         (ttnn, "copy_host_to_device_tensor", o_copy), (ttnn, "to_torch", o_to_torch)]  # fmt: skip
        if o_xsl_host is not None:
            fd._xsl_host_tensor = xsl_host
            self._patched.append((fd, "_xsl_host_tensor", o_xsl_host))
        if rep:
            if o_xsl_bytes is not None:
                fd._xsl_slice_bytes = xsl_bytes
                self._patched.append((fd, "_xsl_slice_bytes", o_xsl_bytes))
            if o_xsl_multi is not None:
                fd._xsl_slices_bytes = xsl_multi
                self._patched.append((fd, "_xsl_slices_bytes", o_xsl_multi))
            if o_perm is not None:  # a missing derived permutation file is not needed (nothing is read)
                fd._permuted_gate_up_file = lambda *a, **kw: None
                self._patched.append((fd, "_permuted_gate_up_file", o_perm))
            self._env = os.environ.get("GLM53_LOAD_PREFETCH")
            os.environ["GLM53_LOAD_PREFETCH"] = "0"  # the prefetcher would read every weight file
        return self

    def __exit__(self, *exc):
        for obj, name, orig in reversed(self._patched):
            setattr(obj, name, orig)
        self._patched = []
        if self.mode == "replay":
            if self._env is None:
                os.environ.pop("GLM53_LOAD_PREFETCH", None)
            else:
                os.environ["GLM53_LOAD_PREFETCH"] = self._env
        return False

    def rows(self) -> dict:
        return dict(uploads=self.uploads, host=self.host, meta=self.meta)

    def check_complete(self) -> None:
        if self.mode == "replay" and len(self.uploads) != len(self.exp_up):
            raise ImageMismatch(f"replay made {len(self.uploads)} uploads, the image recorded {len(self.exp_up)}")


def map_sha(rows: dict) -> str:
    blob = json.dumps([(r["kind"], r["address"], r["spec"]) for r in rows["uploads"]], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ================================================================================================ tensor map
_SKIP = (str, bytes, int, float, bool, type(None), type)


def device_tensors(root, prefix: str = "be") -> list:
    """(path, tensor) of every allocated device tensor reachable from `root` (model objects: classes from models.*,
    SimpleNamespace / dataclasses, lists, tuples, dicts); deterministic order; each object once."""
    import types

    out, seen = [], set()

    def visit(obj, path):
        if isinstance(obj, _SKIP) or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, ttnn.Tensor):
            if obj.storage_type() == ttnn.StorageType.DEVICE and obj.is_allocated():
                out.append((path, obj))
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                visit(v, f"{path}[{k!r}]")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]")
        elif isinstance(obj, types.SimpleNamespace) or type(obj).__module__.startswith("models."):
            d = getattr(obj, "__dict__", None)
            if d is not None:
                for k, v in list(d.items()):
                    visit(v, f"{path}.{k}")

    visit(root, prefix)
    return out


def tensor_map(root) -> dict:
    """{"tensors": [DRAM rows], "l1": [rows]} of the backend's device tensors (lock-step mesh tensors: one address)."""
    di = _di()
    m = {"tensors": [], "l1": []}
    for path, t in device_tensors(root):
        info = di.buffer_info(t)
        row = dict(path=path, address=int(info["address"]), extent=int(info["aligned_size_per_bank"]),
                   shape=[int(v) for v in t.shape], dtype=str(t.dtype), devices=int(info["num_devices"]))  # fmt: skip
        (m["tensors"] if info["dram"] else m["l1"]).append(row)
    return m


def merge_segments(rows: list) -> list:
    ext = sorted({(r["address"], r["address"] + r["extent"]) for r in rows})
    segs: list = []
    for lo, hi in ext:
        if segs and lo <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], hi)
        else:
            segs.append([lo, hi])
    return segs


def comparable(rows: list) -> list:
    return [(r["path"], r["address"], r["extent"], r["shape"], r["dtype"]) for r in rows]


def chip_coords(mesh) -> dict:
    """device id -> (row, col) of the mesh (ttnn.ShardTensorToMesh rank order = row-major coordinates)."""
    R, C = int(mesh.shape[0]), int(mesh.shape[1])
    out = {}
    for r in range(R):
        for c in range(C):
            out[int(mesh.get_device_id(ttnn.MeshCoordinate(r, c)))] = (r, c)
    return out


def covered(segs: list, address: int) -> bool:
    return any(lo <= address < hi for lo, hi in segs)


def live_dram(mesh) -> list:
    """[(lo, hi)] of the DRAM blocks allocated now (absolute per-bank addresses, like buffer_address(); lock-step: chip
    0's view = every chip's). The allocator's block table is relative to its base (dram_unreserved_base:
    free_list_opt.cpp keeps block addresses minus offset_bytes_), so the base is added here."""
    base = int(ttnn._ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.DRAM))
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return [(base + int(b["address"]), base + int(b["address"]) + int(b["size"])) for b in mv.block_table
            if str(b.get("allocated", "")).lower() in ("yes", "true", "1")]  # fmt: skip


def image_segments(tm: dict, rows: dict, mesh) -> tuple:
    """(segments, live uploads outside the object walk): the image = every DRAM tensor reachable from the backend + every
    upload still allocated now (a weight a closure / C++ object holds that the object walk cannot see; freed temporaries
    are not imaged)"""
    live = live_dram(mesh)
    alive = [r for r in rows["uploads"] if "BufferType::DRAM" in r["spec"]
             and any(lo <= r["address"] < hi for lo, hi in live)]  # fmt: skip
    walk = merge_segments(tm["tensors"])
    unreached = [r for r in alive if not covered(walk, r["address"])]
    return merge_segments(tm["tensors"] + [dict(address=r["address"], extent=r["extent"]) for r in alive]), unreached


def unimaged_live(mesh, segs: list, addresses) -> list:
    """the addresses (uploads of unread weights) that still sit in an allocated DRAM block the image does not cover: their
    bytes would never be restored (a freed temporary upload, e.g. an interleaved copy converted to DS, is fine)"""
    live = live_dram(mesh)
    return [a for a in addresses if not covered(segs, a) and any(lo <= a < hi for lo, hi in live)]


# ================================================================================================ identity
def weights_identity() -> dict:
    from models.autoports.zai_org_glm_5_3_flash.tests.common import MANIFEST

    with open(MANIFEST, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"manifest": str(MANIFEST), "sha256": sha, "id": sha[:12]}


def image_key(weights: dict, mesh, rows: dict) -> str:
    return f"{weights['id']}-{K.TP_LAYOUT}-{int(mesh.shape[0])}x{int(mesh.shape[1])}-{map_sha(rows)[:12]}"


# GLM53_DECODE_TEE: the G-snapshot tee is built in allocate_kv_cache, after the replayed construction (its ctl tensor and
# program come after the restore), so it never changes an image
_ENV_SKIP = ("GLM53_IMAGE_", "GLM53_TP_IMAGE_ROOT", "GLM53_WEIGHT_SOURCE", "GLM53_MOCK", "GLM53_CHIP", "GLM53_MESH",
             "GLM53_LOCK", "GLM53_WINDOW", "GLM53_LOAD_PREFETCH", "GLM53_CCACHE", "GLM53_DECODE_TEE")  # fmt: skip


def env_signature() -> dict:
    """the GLM53_* knobs of this process (minus image / lock / launcher controls): recorded with an image so restores
    try the images built under the same knobs first (ordering only - the allocation replay decides)"""
    skip = _ENV_SKIP
    if os.environ.get("GLM53_STACK_DSA_DEFER") == "1":  # Stage 14: the DSA stack attaches after the restore, so its
        # knobs leave the replayed construction (the dense sparse_dsa layers) unchanged; GLM53_PF_SPARSE_T: prefill only
        skip = skip + ("GLM53_STACK_DSA", "GLM53_MLA_DSA_", "GLM53_PF_SPARSE_T")
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("GLM53_") and not k.startswith(skip)}


def _signature_of(d: str) -> Optional[dict]:
    try:
        with open(os.path.join(d, "signature.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def candidates(root: str, weights: dict, mesh, signature: Optional[dict] = None) -> list:
    """image dirs of this weights id / layout / mesh shape (`mesh`: a mesh device or (rows, cols)): those whose recorded
    signature (build arguments + GLM53_* knobs) equals this process's first, then newest first (the allocation map
    decides in the replay)."""
    rc = (int(mesh.shape[0]), int(mesh.shape[1])) if hasattr(mesh, "shape") else tuple(int(v) for v in mesh)
    pre = f"{weights['id']}-{K.TP_LAYOUT}-{rc[0]}x{rc[1]}-"
    if not os.path.isdir(root):
        return []
    ds = [os.path.join(root, d) for d in os.listdir(root) if d.startswith(pre)]
    ds = [d for d in ds if os.path.exists(os.path.join(d, "manifest.json"))]
    want = dict(signature or {}, env=env_signature())
    return sorted(ds, key=lambda d: (_signature_of(d) != want, -os.path.getmtime(os.path.join(d, "manifest.json"))))


# ================================================================================================ build
class CacheGuard:
    """Keeps an image dump page-cache-neutral (8 chips x 26 GB of dirty / unstable NFS pages are charged to the
    writing process's cgroup and can exceed its MemoryMax): while read_image's threads write the chip
    files, one thread per file fdatasyncs it back to back through its own descriptor (on NFS: WRITE + COMMIT, so the
    unstable pages the writer's single final fsync would leave piling up are committed as they come; dirty pages are
    throttled by the kernel's per-cgroup dirty limit) and drops the clean pages (posix_fadvise DONTNEED). Host-side only:
    the C++ writer is unchanged."""

    PERIOD = float(os.environ.get("GLM53_IMAGE_DUMP_FLUSH_S", "0.05"))

    def __init__(self, paths: list):
        self.paths, self.stop_ = list(paths), threading.Event()
        self.rounds, self.errors, self.peak = 0, 0, dict(Dirty=0, Writeback=0, NFS_Unstable=0)
        self.lock = threading.Lock()
        self.th = [threading.Thread(target=self._run, args=(p,), daemon=True) for p in self.paths]
        self.mon = threading.Thread(target=self._monitor, daemon=True)

    def start(self) -> "CacheGuard":
        for t in self.th + [self.mon]:
            t.start()
        return self

    @staticmethod
    def meminfo_kb() -> dict:
        with open("/proc/meminfo") as f:
            return {ln.split(":")[0]: int(ln.split()[1]) for ln in f
                    if ln.split(":")[0] in ("Dirty", "Writeback", "NFS_Unstable")}  # fmt: skip

    @staticmethod
    def _dirty_kb() -> int:
        return sum(CacheGuard.meminfo_kb().values())

    def _monitor(self) -> None:
        while not self.stop_.wait(0.1):
            m = self.meminfo_kb()
            for k, v in m.items():
                self.peak[k] = max(self.peak[k], v)

    def _flush(self, fd) -> None:
        try:
            os.fdatasync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            with self.lock:
                self.errors += 1
        with self.lock:
            self.rounds += 1

    def _run(self, p: str) -> None:
        fd = None
        while fd is None and not self.stop_.is_set():
            try:
                fd = os.open(p, os.O_RDONLY)
            except FileNotFoundError:  # not created yet
                self.stop_.wait(self.PERIOD)
        while fd is not None and not self.stop_.is_set():
            self._flush(fd)
            self.stop_.wait(self.PERIOD)
        if fd is not None:
            self._flush(fd)  # the writer's fsync ran: drop what is left
            os.close(fd)

    def stop(self) -> dict:
        self.stop_.set()
        for t in self.th + [self.mon]:
            t.join()
        return dict(flushes=self.rounds, errors=self.errors, period_s=self.PERIOD,
                    peak_gb={k: round(v / 1e6, 2) for k, v in self.peak.items()})  # fmt: skip


def build_image(be, io: TPWeightIO, mesh, root: Optional[str] = None, *, signature: Optional[dict] = None,
                write: bool = True) -> tuple:  # fmt: skip
    """Dump every chip's bytes of the backend's DRAM tensors (chips in parallel threads; each read is one chip's shard
    on the parent mesh's queue) -> (image dir, manifest). The manifest is written last (atomically)."""
    di = _di()
    t0 = time.perf_counter()
    root = root or os.environ.get("GLM53_TP_IMAGE_ROOT", DEFAULT_ROOT)
    weights, rows = weights_identity(), io.rows()
    tm = tensor_map(be)
    segs, unreached = image_segments(tm, rows, mesh)
    d = os.path.join(root, image_key(weights, mesh, rows))
    os.makedirs(d, exist_ok=True)
    coords = chip_coords(mesh)
    results, errors = {}, []

    par = threading.Semaphore(max(1, int(os.environ.get("GLM53_IMAGE_DUMP_PAR", "4"))))  # chips writing at once

    def dump(dev, rc):
        path = os.path.join(d, f"chip{dev}.img")
        try:
            with par:
                r = di.read_image(mesh, [tuple(s) for s in segs], (path + ".tmp") if write else "", coord=rc)
            if write:
                os.replace(path + ".tmp", path)
            results[dev] = r
        except BaseException as e:  # noqa: BLE001
            errors.append((dev, repr(e)))

    th = [threading.Thread(target=dump, args=(dev, rc)) for dev, rc in sorted(coords.items())]
    tmps = [os.path.join(d, f"chip{dev}.img.tmp") for dev in sorted(coords)]
    guard = CacheGuard(tmps).start() if write else None
    for t in th:
        t.start()
    for t in th:
        t.join()
    flushed = guard.stop() if guard else None
    if errors:
        raise RuntimeError(f"image dump failed: {errors}")
    blob = json.dumps(dict(rows=rows, tensors=tm["tensors"], l1=tm["l1"]), separators=(",", ":")).encode()
    with open(os.path.join(d, "map.json"), "wb") as f:
        f.write(blob)
    if io.op_rows:  # the construction's top-level ops (replays skip the ones fed by unread weights: OP_SKIP)
        save_ops(d, rows, io.op_rows)
    with open(os.path.join(d, "signature.json"), "w") as f:  # candidates(): same knobs first
        json.dump(dict(signature or {}, env=env_signature()), f, indent=1)
    torch.save(io.host_small, os.path.join(d, "host_small.pt.tmp"))
    os.replace(os.path.join(d, "host_small.pt.tmp"), os.path.join(d, "host_small.pt"))
    with open(os.path.join(d, "host_small.pt"), "rb") as f:
        hs_sha = hashlib.sha256(f.read()).hexdigest()
    man = dict(format=FORMAT, layout=K.TP_LAYOUT, weights=weights, mesh=[int(mesh.shape[0]), int(mesh.shape[1])],
               map_sha=map_sha(rows), map_file_sha=hashlib.sha256(blob).hexdigest(), host_small_sha=hs_sha,
               signature=signature or {}, built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               segments=segs, block_bytes=BLOCK, host_keep=K.HOST_KEEP, replay_real_tile16=io.real_tile16, uploads=len(rows["uploads"]), tensors=len(tm["tensors"]),
               l1_tensors=len(tm["l1"]),
               live_uploads_not_in_tensor_map=[(r["kind"], r["key"], r["spec"][:100]) for r in unreached][:20],
               chips={str(dev): dict(coord=list(coords[dev]), image=f"chip{dev}.img", bytes=int(r["bytes"]),
                                     checksums=[int(x) for x in r["checksums"]], block_bytes=int(r["block_bytes"]),
                                     block_hashes=[int(x) for x in r["block_hashes"]],
                                     dump_s=round(float(r["total_s"]), 3), dram_read_s=round(float(r["dram_read_s"]), 3))
                      for dev, r in sorted(results.items())})  # fmt: skip
    man["build_s"] = round(time.perf_counter() - t0, 3)
    man["dump_page_cache_guard"] = flushed
    with open(os.path.join(d, "manifest.json.tmp"), "w") as f:
        json.dump(man, f, indent=1)
    os.replace(os.path.join(d, "manifest.json.tmp"), os.path.join(d, "manifest.json"))
    log(
        f"image {d}: {len(coords)} chips x {sum(h - l for l, h in segs) * 8 / 1e9:.2f} GB, {len(segs)} segments, "
        f"{len(tm['tensors'])} tensors, {len(rows['uploads'])} uploads ({len(unreached)} live outside the object walk, "
        f"imaged too), {man['build_s']} s"
    )
    return d, man


# ================================================================================================ restore
def device_hashes(mesh, man: dict, block_bytes: Optional[int] = None) -> dict:
    """device id -> on-device block hashes of the image segments (every chip in one program)."""
    r = _di().device_block_hashes(mesh, [tuple(s) for s in man["segments"]],
                                  block_bytes=block_bytes or man["block_bytes"])  # fmt: skip
    return {int(k): [int(x) for x in v] for k, v in r["per_device"].items()}


def stale_blocks(man: dict, hashes: dict, ref: Optional[dict] = None) -> dict:
    """device id -> indices of the blocks whose hash differs from the image's (ref: device id -> hashes at a finer block
    size, fine_hashes(); default the manifest's)"""
    out = {}
    for dev, h in hashes.items():
        want = ref[dev] if ref is not None else np.asarray(man["chips"][str(dev)]["block_hashes"], dtype=np.uint64)
        out[dev] = np.nonzero(np.asarray(h, dtype=np.uint64) != want)[0].tolist()
    return out


# ================================================================================================ pull sets
# A resident restart pulls the blocks that differ from the image twice: before the replay (stale: what the last process
# changed - state / argument tables inside the image) and after it (clobbered: what the constructors' device ops wrote -
# derived tensors computed from unread uploads, device temporaries). Both sets repeat from one restart to the next, so
# (1) the restore records them next to the image (pullset_<block>k.json, a hint) and prefetches them into the page cache
# in background threads while it hashes and replays (the files live on NFS: ~1.3 GB/s, the page cache ~10 GB/s), and
# (2) after a verified restore it hashes the chips once more at a finer block size (blocks_<block>k.npz, bound to this
# image's manifest), so later restores move ~0.6x the bytes (TP8 2x4: 4 MiB -> 256 KiB blocks).
class _Knobs:
    """The image knobs, read from the environment WHEN USED (not at import: a long-lived server takes env changes
    from its jobs)."""

    @property
    def FINE(self) -> int:  # block bytes after a verified restore; 0: manifest blocks only
        return int(os.environ.get("GLM53_IMAGE_FINE_BLOCK", str(256 << 10)))

    @property
    def OP_RECORD(self) -> bool:  # builds log their top-level ops (ops.json)
        return os.environ.get("GLM53_IMAGE_OP_RECORD", "1") == "1"

    @property
    def OP_SKIP(self) -> bool:  # replays skip the ops fed by unread weights (needs ops.json)
        return os.environ.get("GLM53_IMAGE_OP_SKIP", "1") == "1"

    @property
    def PREFETCH_THREADS(self) -> int:  # 0: no prefetch
        return int(os.environ.get("GLM53_IMAGE_PREFETCH_THREADS", "32"))

    @property
    def PREFETCH_GB(self) -> float:  # page cache a restore may fill ahead (it is charged to the process's cgroup)
        return float(os.environ.get("GLM53_IMAGE_PREFETCH_GB", "16"))

    @property
    def PREFETCH_NICE(self) -> int:
        return int(os.environ.get("GLM53_IMAGE_PREFETCH_NICE", "10"))

    @property
    def TP_LAYOUT(self) -> str:  # the shard rule's version (part of the image key): bump when bytes change in place
        return os.environ.get("GLM53_TP_LAYOUT", "tp-ds-v1")

    @property
    def HOST0_SOURCE(self) -> bool:  # replay host0 of a cache-loaded placeholder: the file's real bytes (else zeros)
        return os.environ.get("GLM53_IMAGE_HOST0_SOURCE", "1") == "1"

    @property
    def HOST_KEEP(self) -> int:  # builds keep host weights <= this (bytes) real in host_small.pt, bigger: meta
        return int(os.environ.get("GLM53_IMAGE_HOST_KEEP", str(32 << 20)))

    @property
    def MIRROR(self) -> Optional[str]:  # a faster local tier: <MIRROR>/<image key>/chip<id>.img
        return os.environ.get("GLM53_TP_IMAGE_MIRROR") or None


K = _Knobs()


def _image_digest(man: dict) -> str:
    h = hashlib.sha256(man["map_file_sha"].encode())
    for dev in sorted(man["chips"], key=int):
        h.update(json.dumps(man["chips"][dev]["block_hashes"]).encode())
    return h.hexdigest()[:24]


def _banks(man: dict) -> int:
    return next(iter(man["chips"].values()))["bytes"] // sum(h - l for l, h in man["segments"])


def block_file_runs(man: dict, block_bytes: int) -> list:
    """[(file offset, length)] of every block, in dram_image.cpp blocks_of order (segment, bank, block)"""
    banks, out, off = _banks(man), [], 0
    for lo, hi in man["segments"]:
        n = hi - lo
        for _ in range(banks):
            out.extend((off + k, min(block_bytes, n - k)) for k in range(0, n, block_bytes))
            off += n
    return out


def fine_hashes(d: str, man: dict) -> Optional[dict]:
    """device id -> uint64 hashes at FINE bytes per block, recorded after a verified restore of THIS image; else None"""
    if not K.FINE or K.FINE >= man["block_bytes"]:
        return None
    try:
        with np.load(os.path.join(d, f"blocks_{K.FINE >> 10}k.npz")) as z:
            if str(z["digest"]) != _image_digest(man) or int(z["block_bytes"]) != K.FINE:
                return None
            out = {int(k[4:]): z[k].astype(np.uint64) for k in z.files if k.startswith("chip")}
    except (OSError, KeyError, ValueError):
        return None
    return out if sorted(out) == sorted(int(k) for k in man["chips"]) else None


def save_fine_hashes(mesh, d: str, man: dict, img_key: Optional[dict] = None) -> None:
    """hash every chip at FINE bytes per block (DRAM == image just verified) -> blocks_<FINE>k.npz, keyed by the image's
    chips (img_key: this mesh's device id -> image chip key; best effort)"""
    try:
        h = device_hashes(mesh, man, K.FINE)
        if img_key is not None:
            h = {int(img_key[dev]): v for dev, v in h.items()}
        p = os.path.join(d, f"blocks_{K.FINE >> 10}k.npz")
        with open(p + ".tmp", "wb") as f:
            np.savez(f, digest=np.array(_image_digest(man)), block_bytes=np.array(K.FINE),
                     **{f"chip{dev}": np.asarray(v, dtype=np.uint64) for dev, v in h.items()})  # fmt: skip
        os.replace(p + ".tmp", p)
        log(f"fine block hashes ({K.FINE >> 10} KiB, {len(next(iter(h.values())))} per chip) -> {p}")
    except Exception as e:  # noqa: BLE001
        log(f"fine block hashes not saved: {type(e).__name__}: {str(e)[:200]}")


def _read_pullset(d: str, man: dict, block_bytes: int) -> Optional[dict]:
    try:
        with open(os.path.join(d, f"pullset_{block_bytes >> 10}k.json")) as f:
            ps = json.load(f)
    except (OSError, ValueError):
        return None
    return ps if ps.get("digest") == _image_digest(man) and ps.get("block_bytes") == block_bytes else None


def _convert_blocks(man: dict, blocks: list, src: int, dst: int) -> list:
    """block indices at src bytes per block -> the indices at dst bytes covering the same file bytes"""
    want = sorted(block_file_runs(man, src)[i] for i in blocks)
    runs = block_file_runs(man, dst)
    starts = [r_[0] for r_ in runs]
    import bisect

    out, j = set(), 0
    for off, n in want:
        j = max(0, bisect.bisect_right(starts, off) - 1)
        while j < len(runs) and runs[j][0] < off + n:
            if runs[j][0] + runs[j][1] > off:
                out.add(j)
            j += 1
    return sorted(out)


def load_pullset(d: str, man: dict, block_bytes: int) -> Optional[dict]:
    """the pull set recorded at block_bytes; else one recorded at another block size, converted (the first restore
    on the finer blocks still prefetches what the last one pulled)"""
    ps = _read_pullset(d, man, block_bytes)
    if ps is not None:
        return ps
    for other in (4 << 20, man["block_bytes"], 256 << 10, 1 << 20):
        if other == block_bytes:
            continue
        o = _read_pullset(d, man, other)
        if o is None:
            continue
        conv = lambda m: {k: (_convert_blocks(man, v, other, block_bytes) if isinstance(v, list) else v)  # noqa: E731
                          for k, v in m.items()}  # fmt: skip
        return dict(o, block_bytes=block_bytes, stale=conv(o["stale"]), clobbered=conv(o["clobbered"]),
                    converted_from=other)  # fmt: skip
    return None


def save_pullset(d: str, man: dict, block_bytes: int, stale: dict, clobbered: dict) -> None:
    """stale / clobbered: image chip key -> block indices (None = all)"""
    try:
        p = os.path.join(d, f"pullset_{block_bytes >> 10}k.json")
        with open(p + ".tmp", "w") as f:
            json.dump(dict(digest=_image_digest(man), block_bytes=block_bytes,
                           utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           stale={str(k): (v if v is not None else "all") for k, v in stale.items()},
                           clobbered={str(k): v for k, v in clobbered.items()}), f)  # fmt: skip
        os.replace(p + ".tmp", p)
    except Exception as e:  # noqa: BLE001
        log(f"pull set not saved: {type(e).__name__}: {str(e)[:200]}")


def predicted_clobber(man: dict, mp: dict, block_bytes: int) -> list:
    """block indices (every chip: lock-step) a replay's constructors write - no pull set recorded yet: the windows of
    the derived tensors (tensor-map entries no upload made), of the uploads freed before the build ended (temporaries:
    the derived tensors' inputs) and of the bf16 / fp32 TILE uploads the REAL_TILE16 rule rewrites"""
    ups, tm = mp["rows"]["uploads"], mp["tensors"]
    made = {u["address"] for u in ups}
    live = {t["address"] for t in tm}
    ext = [(t["address"], t["extent"]) for t in tm if t["address"] not in made]
    ext += [(u["address"], u["extent"]) for u in ups if u["address"] not in live and "BufferType::DRAM" in u["spec"]]
    if man.get("replay_real_tile16"):
        ext += [(u["address"], u["extent"]) for u in ups if u["kind"] == "from_torch"
                and u["spec"].split("|")[1].split(".")[-1] in ("BFLOAT16", "FLOAT32")
                and u["spec"].split("|")[2].split(".")[-1] == "TILE"]  # fmt: skip
    banks, out, base = _banks(man), set(), 0
    for lo, hi in man["segments"]:
        nblk = -(-(hi - lo) // block_bytes)
        for a, e in ext:
            if a + e <= lo or a >= hi:
                continue
            k0, k1 = (max(a, lo) - lo) // block_bytes, (min(a + e, hi) - lo - 1) // block_bytes
            for b in range(banks):
                out.update(base + b * nblk + k for k in range(k0, k1 + 1))
        base += banks * nblk
    return sorted(out)


_PREFETCH_CHILD = r"""
import json, os, sys, threading, time
os.nice(int(sys.argv[2]))  # below the replay's Python
items = json.load(sys.stdin)
fds = {p: os.open(p, os.O_RDONLY) for p in {it[0] for it in items}}
nxt, lock, done = [0], threading.Lock(), [0, 0]
t0 = time.perf_counter()
def run():
    buf = memoryview(bytearray(8 << 20))
    while True:
        with lock:
            if nxt[0] >= len(items):
                return
            path, off, n = items[nxt[0]]
            nxt[0] += 1
        got = 0
        try:
            while got < n:
                r = os.preadv(fds[path], [buf[: n - got]], off + got)
                if r <= 0:
                    break
                got += r
        except OSError:
            with lock:
                done[1] += 1
        with lock:
            done[0] += got
th = [threading.Thread(target=run, daemon=True) for _ in range(int(sys.argv[1]))]
[t.start() for t in th]
while any(t.is_alive() for t in th):
    time.sleep(0.5)
    print("P", done[0], round(time.perf_counter() - t0, 3), done[1], flush=True)
print("D", done[0], round(time.perf_counter() - t0, 3), done[1], flush=True)
"""


class Prefetcher:
    """Background page-cache warm-up of image file ranges a restore is about to pull: `groups` = [[(path, [block
    index])]] in priority order, read round-robin over the files by a CHILD PROCESS with PREFETCH_THREADS threads
    (os.preadv into scratch buffers). A child, not threads, so the reads never compete with the replay's Python for the
    GIL. A hint only: the pull reads the files itself."""

    CHUNK = 8 << 20

    def __init__(self, man: dict, block_bytes: int, groups: list):
        runs = block_file_runs(man, block_bytes)
        self.items = []
        # the page cache is charged to the restoring process's cgroup: prefetch at most PREFETCH_GB (priority order);
        # partial pulls that read O_DIRECT bypass the page cache - nothing to prefetch for them
        cap = 0 if os.environ.get("GLM53_IMAGE_DIRECT_PARTIAL") == "1" else K.PREFETCH_GB * 1e9
        for jobs in groups:  # group order = priority
            per = []
            for path, blocks in jobs:
                merged = []
                for off, n in sorted(runs[i] for i in blocks):  # coalesce adjacent blocks into <= CHUNK reads
                    if merged and merged[-1][1] + merged[-1][2] == off and merged[-1][2] + n <= self.CHUNK:
                        merged[-1][2] += n
                    else:
                        merged.append([path, off, n])
                per.append(merged)
            for k in range(max((len(m) for m in per), default=0)):  # round-robin over the chips' files
                self.items.extend(m[k] for m in per if k < len(m))
        acc, keep = 0, []
        for it in self.items:
            if acc + it[2] > cap:
                break
            keep.append(it)
            acc += it[2]
        self.items = keep
        self.proc, self.reader = None, None
        self.bytes, self.t_done, self.errors, self.t0 = 0, None, 0, 0.0

    def start(self, detach: bool = False) -> "Prefetcher":
        """detach: the child outlives this process (a launcher's pre-open warm-up), no progress is read"""
        if not self.items or K.PREFETCH_THREADS <= 0:
            return self
        import subprocess
        import sys

        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen([sys.executable, "-I", "-S", "-c", _PREFETCH_CHILD, str(K.PREFETCH_THREADS),
                                      str(K.PREFETCH_NICE)], stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL if detach else subprocess.PIPE, text=True,
                                     start_new_session=detach)  # fmt: skip
        self.proc.stdin.write(json.dumps(self.items))
        self.proc.stdin.close()
        if detach:
            self.proc = None
            return self

        def read():
            for line in self.proc.stdout:
                f = line.split()
                if len(f) == 4:
                    self.bytes, self.errors = int(f[1]), int(f[3])
                    if f[0] == "D":
                        self.t_done = float(f[2])

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        return self

    def stop(self) -> dict:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
            self.proc.wait()
            self.reader.join(timeout=5)
            self.proc = None
        return dict(bytes=self.bytes, items=len(self.items), done_s=self.t_done, errors=self.errors,
                    stopped_s=round(time.perf_counter() - self.t0, 3) if self.t0 else None)  # fmt: skip


def chip_file(d: str, chip: dict) -> str:
    """the chip file a restore reads: the local mirror's copy (`tp_image mirror`) when it is complete (same size), else
    the image dir's. A wrong mirror cannot pass: whole pulls check the per-GiB checksums, every restore ends with the
    device block hashes."""
    if K.MIRROR:
        p = os.path.join(K.MIRROR, os.path.basename(os.path.normpath(d)), chip["image"])
        try:
            if os.path.getsize(p) == chip["bytes"]:
                return p
        except OSError:
            pass
    return os.path.join(d, chip["image"])


def free_hugepages() -> int:
    with open("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages") as f:
        return int(f.read())


def pull(
    mesh,
    d: str,
    man: dict,
    only: dict,
    *,
    slots: Optional[int] = None,
    cores: int = 32,
    threads: int = 16,
    block_bytes: Optional[int] = None,
    direct_from: Optional[int] = None,
) -> dict:
    """Pull the given blocks (device id -> block indices at block_bytes, default the manifest's; [] = nothing, None =
    the whole image) of every chip, chips in parallel threads (each puller: the chip's mesh coordinate on the parent
    mesh; enqueues serialized in C++). direct_from (whole-image pulls): file bytes from there on are read with O_DIRECT
    (no page-cache copy, so the cached pull sets of the next resident restart are not evicted)."""
    di = _di()
    want = {dev: v for dev, v in only.items() if v is None or v}
    if not want:
        return {}
    n = len(want)
    slots = slots or max(1, min(3, (free_hugepages() - 2) // n))
    if free_hugepages() < slots * n:
        raise ImageMismatch(f"need {slots * n} free 1 GiB hugepages, {free_hugepages()} free")
    pullers = {}
    for dev, blocks in want.items():
        c = man["chips"][str(dev)]
        kw = dict(direct_from=int(direct_from)) if direct_from is not None and blocks is None else {}
        if blocks is not None and os.environ.get("GLM53_IMAGE_DIRECT_PARTIAL") == "1":
            kw["direct_partial"] = True  # partial pulls read O_DIRECT (needs a _ttnn.so with that parameter)
        try:
            p = di.ImagePuller(mesh, [tuple(s) for s in man["segments"]], chip_file(d, c), slots=slots,
                               num_cores=cores, read_threads=threads, checksum=blocks is None,
                               only_blocks=[int(b) for b in (blocks or [])], block_bytes=block_bytes or man["block_bytes"],
                               coord=tuple(c["coord"]), **kw)  # fmt: skip
        except TypeError:
            if "direct_partial" not in kw:
                raise
            kw.pop("direct_partial")  # an older _ttnn.so: buffered partial reads
            log("ImagePuller has no direct_partial (older _ttnn.so): buffered partial reads")
            p = di.ImagePuller(mesh, [tuple(s) for s in man["segments"]], chip_file(d, c), slots=slots,
                               num_cores=cores, read_threads=threads, checksum=blocks is None,
                               only_blocks=[int(b) for b in (blocks or [])], block_bytes=block_bytes or man["block_bytes"],
                               coord=tuple(c["coord"]), **kw)  # fmt: skip
        p.start()
        pullers[dev] = (p, blocks is not None)
    out, errors = {}, []

    def fin(dev, p):
        try:
            p.finish()
        except BaseException as e:  # noqa: BLE001
            errors.append((dev, repr(e)))

    th = [threading.Thread(target=fin, args=(dev, p)) for dev, (p, _) in pullers.items()]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        raise ImageMismatch(f"pull failed: {errors}")
    for dev, (p, partial) in pullers.items():
        r = p.result()
        if not partial and [int(x) for x in r["checksums"]] != man["chips"][str(dev)]["checksums"]:
            raise ImageMismatch(f"chip {dev}: image file checksum mismatch")
        out[dev] = {k: (float(v) if not isinstance(v, list) else None) for k, v in r.items() if k != "checksums"}
        st = out[dev]
        log(f"pull chip {dev}: {st.get('bytes', 0) / 1e9:.2f} GB in {st.get('finish_s', 0):.2f} s "
            f"(read {st.get('read_s', 0):.2f} s / wall {st.get('read_wall_s', 0):.2f} s, wait read {st.get('wait_read_s', 0):.2f} s, "
            f"wait device {st.get('wait_device_s', 0):.2f} s, {st.get('slots', 0):.0f} slots, {st.get('cores', 0):.0f} cores, "
            f"cached {st.get('bytes_cached', 0) / 1e9:.2f} GB / direct {st.get('bytes_direct', 0) / 1e9:.2f} GB)")  # fmt: skip
    return out


def load_image_meta(d: str) -> tuple:
    """(manifest, map): the small sidecars a restore needs before anything else (manifest.json ~1 MB, map.json ~27 MB,
    hash-checked). host_small.pt (the kept small host weights: 3.4 GB on a 32 MiB host-keep image) is NOT read here:
    HostSmallLoader reads it in the background (the replay is the first consumer)."""
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    if man.get("format") != FORMAT:
        raise ImageMismatch(f"{d}: format {man.get('format')} != {FORMAT}")
    with open(os.path.join(d, "map.json"), "rb") as f:
        blob = f.read()
    if hashlib.sha256(blob).hexdigest() != man["map_file_sha"]:
        raise ImageMismatch(f"{d}: map.json hash mismatch")
    return man, json.loads(blob)


def _verified_stat_path(sha: str) -> Optional[str]:
    """the stat sidecar of a file whose sha256 was verified by an earlier restore (GLM53_IMAGE_VERIFY_CACHE, default
    ~/.cache/glm53/image_verify; empty: no sidecars)"""
    root = os.environ.get("GLM53_IMAGE_VERIFY_CACHE", os.path.expanduser("~/.cache/glm53/image_verify"))
    return os.path.join(root, f"{sha}.json") if root else None


def _file_stat_key(path: str) -> dict:
    st = os.stat(path)
    return dict(size=int(st.st_size), mtime_ns=int(st.st_mtime_ns), path=os.path.realpath(path))


def stat_verified(path: str, sha: str) -> bool:
    """an earlier restore hashed this very file (size, mtime, path recorded under its sha256)"""
    side = _verified_stat_path(sha)
    if side is None:
        return False
    try:
        with open(side) as f:
            rec = json.load(f)
        return all(rec.get(k) == v for k, v in _file_stat_key(path).items())
    except (OSError, ValueError):
        return False


def page_in(path: str, chunk: int = 8 << 20) -> int:
    """read the file through the page cache into a scratch buffer (no copy kept): what a following mmap faults from"""
    n = 0
    buf = memoryview(bytearray(chunk))
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        with contextlib.suppress(OSError, AttributeError):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
        while True:
            r = os.preadv(fd, [buf], n)
            if r <= 0:
                return n
            n += r
    finally:
        os.close(fd)


def read_verified(path: str, sha: str, *, label: str) -> bytes:
    """the file's bytes, verified against `sha`. GLM53_IMAGE_HOST_SMALL_VERIFY: "stat" (default: the sha256 is computed
    once per file; a stat sidecar (size, mtime, path) then stands for it on later restores - 3.4 GB of SHA-256 at
    ~0.5 GB/s was 7 s of every restart), "sha" (every restore hashes), "none" (size + mtime only; the manifest's sha is
    never checked). The read happens once: the bytes are returned for torch.load (the old path read the file twice)."""
    mode = os.environ.get("GLM53_IMAGE_HOST_SMALL_VERIFY", "stat")
    if mode not in ("stat", "sha", "none"):
        raise ValueError(f"GLM53_IMAGE_HOST_SMALL_VERIFY={mode!r}: stat | sha | none")
    key = _file_stat_key(path)
    side = _verified_stat_path(sha) if mode == "stat" else None
    verified = stat_verified(path, sha) if mode == "stat" else False
    with open(path, "rb") as f:
        buf = f.read()
    if mode == "sha" or (mode == "stat" and not verified):
        if hashlib.sha256(buf).hexdigest() != sha:  # (hashlib releases the GIL over the whole buffer)
            raise ImageMismatch(f"{label}: hash mismatch")
        if side is not None:
            with contextlib.suppress(OSError):
                os.makedirs(os.path.dirname(side), exist_ok=True)
                tmp = f"{side}.{os.getpid()}.tmp"
                with open(tmp, "w") as f:
                    json.dump(dict(key, sha256=sha, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), f)
                os.replace(tmp, side)
    return buf


class HostSmallLoader:
    """host_small.pt read + verified + torch.loaded in a background thread (GLM53_IMAGE_HOST_SMALL_ASYNC=1, default):
    started right after the manifest, joined by the restore just before the replay (its first consumer), so the device
    pre-hash and the stale pull hide it. `get()` raises what the thread raised (ImageMismatch on a hash mismatch: the
    restore tears down and falls back as before). stats: read_s / wall_s / wait_s / verify."""

    def __init__(self, d: str, man: dict):
        self.path = os.path.join(d, "host_small.pt")
        self.sha = man["host_small_sha"]
        self.hs, self.err, self.stats = None, None, {}
        self.thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        t0 = time.perf_counter()
        try:
            mode = os.environ.get("GLM53_IMAGE_HOST_SMALL_VERIFY", "stat")
            if mode == "sha" or (mode == "stat" and not stat_verified(self.path, self.sha)):
                buf = read_verified(self.path, self.sha, label=self.path)  # one read + SHA-256 (records the stat)
                self.stats["bytes"], self.stats["hashed"] = len(buf), True
                del buf
            else:
                self.stats["bytes"], self.stats["hashed"] = os.path.getsize(self.path), False
                page_in(self.path)  # (the page cache; the mapped tensors below then fault from memory)
            self.stats["read_s"] = round(time.perf_counter() - t0, 3)
            # mmap: no 3.4 GB copy and no unpickle copy (torch.load(path) of a 32 MiB-keep image cost 2.6-2.8 s on top
            # of the read); the mapping is private (copy-on-write), the replay may write its tensors
            try:
                self.hs = torch.load(self.path, mmap=True)
            except TypeError:  # an older torch
                self.hs = torch.load(self.path)
        except BaseException as e:  # noqa: BLE001
            self.err = e
        self.stats["wall_s"] = round(time.perf_counter() - t0, 3)
        self.stats["verify"] = os.environ.get("GLM53_IMAGE_HOST_SMALL_VERIFY", "stat")

    def start(self) -> "HostSmallLoader":
        if os.environ.get("GLM53_IMAGE_HOST_SMALL_ASYNC", "1") == "1":
            self.thread = threading.Thread(target=self._run, name="host_small", daemon=True)
            self.thread.start()
        else:
            self._run()
        return self

    def get(self) -> dict:
        t = time.perf_counter()
        if self.thread is not None:
            self.thread.join()
        self.stats["wait_s"] = round(time.perf_counter() - t, 3)
        if self.err is not None:
            raise self.err
        return self.hs


def load_image(d: str) -> tuple:
    """(manifest, map, host_small): the synchronous form (the CLI: rerule / record checks)"""
    man, mp = load_image_meta(d)
    return man, mp, HostSmallLoader(d, man).start().get()


def prefetch_files(paths: list, *, label: str = "files") -> Optional["Prefetcher"]:
    """Page-cache warm-up of whole files by the restore's prefetch child (nice, no GIL): e.g. the weight-cache files
    the deferred DSA halves read at attach_dsa (StackBackend: GLM53_DSA_PREFETCH=1), started while the replay runs.
    Returns the running Prefetcher (stop() for its stats) or None (no threads / nothing to read)."""
    items = []
    for p in dict.fromkeys(paths):
        try:
            n = os.path.getsize(p)
        except OSError:
            continue
        chunk = getattr(Prefetcher, "CHUNK", 8 << 20)
        for off in range(0, n, chunk):
            items.append([p, off, min(chunk, n - off)])
    if not items or K.PREFETCH_THREADS <= 0:
        return None
    pf = Prefetcher.__new__(Prefetcher)
    pf.items, pf.proc, pf.reader = items, None, None
    pf.bytes, pf.t_done, pf.errors, pf.t0 = 0, None, 0, 0.0
    pf.label = label
    return pf.start()


def save_ops(d: str, rows: dict, ops: list) -> None:
    """ops.json: the construction's top-level ttnn ops (name, outputs: address / spec / extent), bound to the allocation
    map (written by a build, or by a MOCK record of the same construction: its allocations equal the image's)"""
    with open(os.path.join(d, "ops.json.tmp"), "w") as f:
        json.dump(dict(map_sha=map_sha(rows), ops=ops), f, separators=(",", ":"))
    os.replace(os.path.join(d, "ops.json.tmp"), os.path.join(d, "ops.json"))


def load_ops(d: str, man: dict) -> Optional[list]:
    try:
        with open(os.path.join(d, "ops.json")) as f:
            o = json.load(f)
    except (OSError, ValueError):
        return None
    return o["ops"] if o.get("map_sha") == man["map_sha"] else None


def chip_view(man: dict, mesh) -> tuple:
    """(the manifest with its chips keyed by THIS mesh's device ids, {device id: image chip key}). Image chips are
    matched by mesh coordinate - a TP shard belongs to a coordinate, not to a physical chip - so an image built on one
    card restores on another (its resident check finds every block stale there: the whole image moves)."""
    coords = chip_coords(mesh)
    by_coord = {tuple(v["coord"]): k for k, v in man["chips"].items()}
    if len(by_coord) != len(coords) or any(tuple(rc) not in by_coord for rc in coords.values()):
        raise ImageMismatch(f"image chips at {sorted(by_coord)} != mesh coordinates "
                            f"{sorted(tuple(rc) for rc in coords.values())}")  # fmt: skip
    key = {dev: by_coord[tuple(rc)] for dev, rc in coords.items()}
    return dict(man, chips={str(dev): man["chips"][k] for dev, k in key.items()}), key


def restore_backend(build: Callable[[], Any], mesh, d: str, *, teardown: Callable[[Any], None] = lambda b: None,
                    timeline: Optional[dict] = None, resident: Optional[bool] = None):  # fmt: skip
    """Replay `build()` (the normal StackBackend construction) against image `d` and bring every chip's DRAM to the image
    (resident: only stale + clobbered blocks move). Raises ImageMismatch (after teardown) on any mismatch."""
    tl = timeline if timeline is not None else {}
    t0 = time.perf_counter()
    man, mp = load_image_meta(d)
    if (
        man["mesh"] != [int(mesh.shape[0]), int(mesh.shape[1])]
        or man["weights"]["sha256"] != weights_identity()["sha256"]
    ):
        raise ImageMismatch(f"{d}: image of mesh {man['mesh']} / weights {man['weights']['id']}")
    hs_loader = HostSmallLoader(d, man).start()  # host_small.pt in the background: joined just before the replay
    coords = chip_coords(mesh)
    img = man  # the image's own manifest (sidecars: keyed by its chips); `man` below = keyed by this mesh's devices
    man, img_key = chip_view(img, mesh)
    if any(str(dev) != k for dev, k in img_key.items()):
        tl["chip_map"] = {str(dev): k for dev, k in img_key.items()}
    tl["load_manifest_s"] = time.perf_counter() - t0
    resident = os.environ.get("GLM53_IMAGE_RESIDENT", "1") == "1" if resident is None else resident
    fine = fine_hashes(d, img)
    if fine is not None:
        fine = {dev: fine[int(img_key[dev])] for dev in coords}
    bb = K.FINE if fine is not None else man["block_bytes"]
    tl["block_bytes"] = bb
    files = {dev: chip_file(d, man["chips"][str(dev)]) for dev in coords}
    if any(not f_.startswith(d) for f_ in files.values()):
        tl["mirror"] = K.MIRROR
    ps = load_pullset(d, img, bb)
    pred_clob = ({dev: ps["clobbered"].get(img_key[dev], []) for dev in coords} if ps
                 else dict.fromkeys(coords, predicted_clobber(man, mp, bb)))  # fmt: skip
    tl["pull_set_hint"] = "recorded" if ps else "predicted from the map"
    # Order. pull-first (default): DRAM is brought to the image BEFORE any construction op runs - pre-hash, pull the
    # stale blocks (cold: the whole image), replay, pull what the replay still writes, final hashes. replay-first
    # (GLM53_IMAGE_ORDER / GLM53_IMAGE_COLD_ORDER=replay-first): the replay runs on whatever DRAM holds, then ONE hash +
    # ONE pull - it rejects a stale image before any pull, but construction ops then run on DRAM the image has not
    # restored (a 2x4 mesh hung inside such a replay on 2026-10-21, cause not established: opt-in only). Both orders
    # check that every live DRAM block is inside the image and every L1 tensor is a real upload (else ImageMismatch).
    order = os.environ.get("GLM53_IMAGE_ORDER" if resident else "GLM53_IMAGE_COLD_ORDER", "pull-first")
    tl["order"] = order
    pf = None
    obj = None
    try:
        if order == "replay-first" and not resident:
            # page cache is charged to the server's cgroup (a MemoryMax scope): a modest head
            gb = float(os.environ.get("GLM53_IMAGE_COLD_PREFETCH_GB", "16"))
            if os.environ.get("GLM53_IMAGE_DIRECT_PARTIAL") == "1":
                gb = 0.0  # O_DIRECT reads everywhere: the whole pull below reads O_DIRECT from byte 0
            budget = min(gb, K.PREFETCH_GB) * 1e9 / max(1, len(coords))
            runs, head, acc = block_file_runs(man, bb), [], 0
            for i in sorted(range(len(runs)), key=lambda k: runs[k][0]):  # file order
                if acc >= budget:
                    break
                head.append(i)
                acc += runs[i][1]
            pf = Prefetcher(man, bb, [[(files[dev], head) for dev in coords]]).start()
            head_bytes = (max(runs[i][0] + runs[i][1] for i in head) // (1 << 20) * (1 << 20)) if head else 0
        elif resident:  # stale blocks first (the pull right after the pre-hash), then the replay's clobber set
            pred_stale = {dev: (ps["stale"].get(img_key[dev], []) if ps else []) for dev in coords}
            pred_stale = {dev: (v if isinstance(v, list) else []) for dev, v in pred_stale.items()}
            pf = Prefetcher(man, bb, [[(files[dev], pred_stale[dev]) for dev in coords],
                                      [(files[dev], pred_clob[dev]) for dev in coords]]).start()  # fmt: skip
        t = time.perf_counter()
        if order == "replay-first":
            stale = {dev: [] for dev in coords}  # nothing moves before the replay
        else:
            stale = stale_blocks(man, device_hashes(mesh, man, bb), fine) if resident else {dev: None for dev in coords}
        tl["pre_hash_s"] = time.perf_counter() - t
        tl["stale_blocks"] = ("deferred (replay first: the post-replay hashes decide)" if order == "replay-first" else
                              {str(k): (len(v) if v is not None else "all") for k, v in stale.items()})  # fmt: skip
        tl["blocks"] = (
            len(block_file_runs(man, bb))
            if fine is not None
            else len(next(iter(man["chips"].values()))["block_hashes"])
        )
        log(f"resident check {d}: stale blocks {tl['stale_blocks']} of {tl['blocks']} per chip ({bb >> 10} KiB blocks, "
            f"pull-set hint {tl['pull_set_hint']})")  # fmt: skip
        # a chip whose DRAM mostly moved (other jobs between opens, a reset) is pulled whole with O_DIRECT reads: a
        # buffered partial pull of most of the image is several times slower than one sequential O_DIRECT read
        whole = float(os.environ.get("GLM53_IMAGE_WHOLE_FRAC", "0.3"))
        tl["whole_pull_chips"] = sorted(str(k) for k, v in stale.items()
                                        if resident and v is not None and len(v) > whole * tl["blocks"])  # fmt: skip
        for k in list(stale):
            if str(k) in tl["whole_pull_chips"]:
                stale[k] = None
        if tl["whole_pull_chips"]:
            log(f"whole-image O_DIRECT pull for chips {tl['whole_pull_chips']} (> {whole:.0%} of their blocks stale)")
        t = time.perf_counter()
        whole_direct = (not resident or tl["whole_pull_chips"]) and os.environ.get(
            "GLM53_IMAGE_COLD_DIRECT", "1"
        ) == "1"
        tl["pull_stale"] = pull(
            mesh, d, man, stale, block_bytes=bb, direct_from=0 if whole_direct else None
        )  # whole chips: O_DIRECT (no page-cache copy)
        tl["pull_stale_s"] = time.perf_counter() - t
        if pf is None and order == "pull-first":  # whole image pulled: now warm the clobber set during the replay
            pf = Prefetcher(man, bb, [[(files[dev], pred_clob[dev]) for dev in coords]]).start()
        t = time.perf_counter()
        prof = None
        if os.environ.get("GLM53_IMAGE_PROFILE") == "1":  # where the replay's time goes (adds ~10-20 % to it)
            import cProfile

            prof = cProfile.Profile()
        ops = load_ops(d, img) if K.OP_SKIP else None
        tl["op_records"] = len(ops) if ops else 0
        hs = hs_loader.get()  # (raises ImageMismatch on a hash mismatch: teardown + fallback as before)
        tl["host_small"] = dict(hs_loader.stats)
        with TPWeightIO("replay", mp["rows"], hs, real_tile16=man.get("replay_real_tile16", False), ops=ops) as io:
            if prof is not None:
                prof.enable()
            try:
                obj = build()
            finally:
                if prof is not None:
                    prof.disable()
        io.check_complete()
        tl["replay_s"] = time.perf_counter() - t
        if prof is not None:
            import io as _io
            import pstats

            for key in ("tottime", "cumulative"):
                buf = _io.StringIO()
                pstats.Stats(prof, stream=buf).sort_stats(key).print_stats(30)
                print(f"[tp-image] replay profile ({key}, profiled replay {tl['replay_s']:.2f} s)\n{buf.getvalue()}",
                      flush=True)  # fmt: skip
        tl["replay_alloc_s"] = io.alloc_s
        tl["replay_uploads"] = len(io.uploads)
        tl["replay_host_misses"] = io.host_misses
        if ops:
            tl["op_skip"] = dict(io.op_stats)
            log(f"replay op skip: {io.op_stats}")
        got = tensor_map(obj)
        if comparable(got["tensors"]) != comparable(mp["tensors"]):
            a, b = comparable(got["tensors"]), comparable(mp["tensors"])
            i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            raise ImageMismatch(f"tensor map differs from the image at #{i}: {a[i] if i < len(a) else None} != "
                                f"{b[i] if i < len(b) else None}")  # fmt: skip
        segs = man["segments"]
        # every live tensor whose content is not the weights' (placeholder uploads, skipped ops' outputs) must lie inside
        # the image - else its bytes are never restored
        bad = unimaged_live(mesh, segs, sorted({p[1] for p in io.placeholder_allocs} | set(io.ph_addr)))
        if bad:
            raise ImageMismatch(
                f"{len(bad)} live tensors made from unread weights outside the image (first {bad[0]:#x})"
            )
        l1_ph = [t_["path"] for t_ in got["l1"] if t_["address"] in io.ph_addr]  # L1 is never imaged
        if l1_ph:
            raise ImageMismatch(f"L1 tensors made from unread weights: {l1_ph[:3]}")
        if order == "replay-first":
            # data tensors allocate bottom-up from the DRAM base; program binaries (kernel_bin_buf, program.cpp) and
            # other top-down buffers sit above the image's top: only blocks below it must be inside the image
            top = max(hi for _, hi in segs)
            live = live_dram(mesh)
            outside = [(lo, hi) for lo, hi in live if lo < top
                       and sum(max(0, min(hi, b) - max(lo, a)) for a, b in segs) < hi - lo]  # fmt: skip
            tl["live_above_image_top_bytes_per_bank"] = sum(hi - max(lo, top) for lo, hi in live if hi > top)
            phs = {p_[1] for p_ in io.placeholder_allocs} | set(io.ph_addr)
            l1_up = {u["address"] for u in io.uploads if "BufferType::L1" in u["spec"] and u["address"] not in phs}
            l1_other = [t_["path"] for t_ in got["l1"] if t_["address"] not in l1_up]
            if outside or l1_other:
                raise ImageMismatch(f"replay-first: {len(outside)} live DRAM blocks outside the image "
                                    f"({outside[:3]}), L1 tensors not uploaded {l1_other[:3]} - use "
                                    f"GLM53_IMAGE_ORDER / GLM53_IMAGE_COLD_ORDER=pull-first")  # fmt: skip
            tl["prefetch_at_post_pull"] = dict(bytes=pf.bytes, done_s=pf.t_done) if pf else None
            if resident:  # one hash: every block that differs (the last process's changes + the replay's writes)
                t = time.perf_counter()
                diff = stale_blocks(man, device_hashes(mesh, man, bb), fine)
                tl["post_hash_s"] = time.perf_counter() - t
                tl["clobbered_blocks"] = {str(k): len(v) for k, v in diff.items()}
                tl["clobbered_predicted"] = {str(k): len(set(v) & set(pred_clob[k])) for k, v in diff.items()}
                t = time.perf_counter()
                tl["pull_clobbered"] = pull(mesh, d, man, diff, block_bytes=bb)
                tl["pull_clobbered_s"] = time.perf_counter() - t
                tl["prefetch"] = pf.stop() if pf else None
                pf = None
            else:
                tl["prefetch"] = pf.stop() if pf else None
                pf = None
                t = time.perf_counter()
                # the prefetched head through the page cache, the rest O_DIRECT (keeps the resident pull sets cached)
                tl["pull_stale"] = pull(mesh, d, man, {dev: None for dev in coords}, block_bytes=bb,
                                        direct_from=head_bytes if os.environ.get("GLM53_IMAGE_COLD_DIRECT", "1") == "1"
                                        else None)  # fmt: skip
                tl["pull_stale_s"] = time.perf_counter() - t
            if not resident or any(diff.values()):  # (resident, nothing differed: the hash above verified it)
                t = time.perf_counter()
                left = {k: v for k, v in stale_blocks(man, device_hashes(mesh, man, bb), fine).items() if v}
                tl["final_hash_s"] = time.perf_counter() - t
                if left:
                    raise ImageMismatch(f"blocks still differ after the pull: { {k: len(v) for k, v in left.items()} }")
            tl["restore_s"] = time.perf_counter() - t0
            log(f"restored {d} (replay first, then " + (f"{tl['clobbered_blocks']} differing blocks pulled ({bb >> 10} KiB, "
                f"predicted {tl['clobbered_predicted']})" if resident else "the whole image pulled") +
                f"; prefetch {tl['prefetch']}): DRAM == image on every chip; {tl['restore_s']:.2f} s")  # fmt: skip
            t = time.perf_counter()
            if resident:
                save_pullset(d, img, bb, {img_key[k]: [] for k in diff}, {img_key[k]: v for k, v in diff.items()})
            if fine is None and K.FINE and K.FINE < man["block_bytes"]:
                save_fine_hashes(mesh, d, img, img_key)
            tl["after_restore_s"] = time.perf_counter() - t
            return obj
        t = time.perf_counter()
        clob = stale_blocks(man, device_hashes(mesh, man, bb), fine)
        tl["post_hash_s"] = time.perf_counter() - t
        tl["clobbered_blocks"] = {str(k): len(v) for k, v in clob.items()}
        tl["clobbered_predicted"] = {str(k): len(set(v) & set(pred_clob[k])) for k, v in clob.items()}
        tl["prefetch_at_post_pull"] = dict(bytes=pf.bytes, done_s=pf.t_done) if pf else None
        t = time.perf_counter()
        tl["pull_clobbered"] = pull(mesh, d, man, clob, block_bytes=bb)
        tl["pull_clobbered_s"] = time.perf_counter() - t
        if any(clob.values()):
            t = time.perf_counter()
            left = {k: v for k, v in stale_blocks(man, device_hashes(mesh, man, bb), fine).items() if v}
            tl["final_hash_s"] = time.perf_counter() - t
            if left:
                raise ImageMismatch(f"blocks still differ after the pull: { {k: len(v) for k, v in left.items()} }")
        tl["prefetch"] = pf.stop() if pf else None
        pf = None
        tl["restore_s"] = time.perf_counter() - t0
        log(f"restored {d}: clobbered {tl['clobbered_blocks']} ({bb >> 10} KiB blocks, predicted "
            f"{tl['clobbered_predicted']}); prefetch {tl['prefetch']}; DRAM == image on every chip; "
            f"{tl['restore_s']:.2f} s")  # fmt: skip
        t = time.perf_counter()
        save_pullset(d, img, bb, {img_key[k]: v for k, v in stale.items()}, {img_key[k]: v for k, v in clob.items()})
        if fine is None and K.FINE and K.FINE < man["block_bytes"]:
            save_fine_hashes(mesh, d, img, img_key)
        tl["after_restore_s"] = time.perf_counter() - t
        return obj
    except BaseException:
        if pf is not None:
            with contextlib.suppress(Exception):
                pf.stop()
        with contextlib.suppress(BaseException):
            hs_loader.get()  # (never a stray 3.4 GB read beside the fallback loader)
        gc.collect()
        if obj is not None:
            with contextlib.suppress(Exception):
                teardown(obj)
        raise


# ================================================================================================ entry point
def build_or_restore(build: Callable[[], Any], mesh, *, root: Optional[str] = None, signature: Optional[dict] = None,
                     teardown: Callable[[Any], None] = lambda b: None, make_image: Optional[bool] = None,
                     timeline: Optional[dict] = None, on_fail: Optional[Callable[[], None]] = None):  # fmt: skip
    """The fast-start wrapper around a StackBackend construction: try every image of this weights / layout / mesh
    (newest first: the allocation map decides), else build normally under the recorder and (GLM53_IMAGE_BUILD=1 or
    make_image) write the image. Returns (backend, info). on_fail: called after a failed restore, before the next try /
    the tensor loader (e.g. clear a shared weight cache the failed replay filled with unrestored tensors). A restore
    that fails with a device-side exception (validation, e.g. a CB clash) falls back too; the image stays."""
    tl = timeline if timeline is not None else {}
    root = root or os.environ.get("GLM53_TP_IMAGE_ROOT", DEFAULT_ROOT)
    weights = weights_identity()
    tried = []
    if os.environ.get("GLM53_WEIGHT_SOURCE", "image") == "image":
        pinned = os.environ.get("GLM53_TP_IMAGE_DIR")  # one image only (measurements; else every candidate)
        want = dict(signature or {}, env=env_signature())
        for d in [pinned] if pinned else candidates(root, weights, mesh, signature):
            sig = _signature_of(d)
            if (not pinned and sig is not None and sig != want
                    and os.environ.get("GLM53_IMAGE_ANY_SIGNATURE") != "1"):  # fmt: skip
                # built under other build arguments / knobs: skipped before any pull (pull-first would move its stale
                # blocks before the replay could reject it)
                tried.append((os.path.basename(d), "signature differs"))
                continue
            sub = {}
            try:
                be = restore_backend(build, mesh, d, teardown=teardown, timeline=sub)
                tl.update(source="image", image=d, **sub)
                return be, tl
            except (ImageMismatch, RuntimeError) as e:
                tried.append((os.path.basename(d), f"{type(e).__name__}: {str(e)[:300]}"))
                log(f"image {os.path.basename(d)} unusable: {type(e).__name__}: {str(e)[:300]}")
                if on_fail is not None:
                    on_fail()
                gc.collect()
    t = time.perf_counter()
    io = TPWeightIO("record")
    with io:
        be = build()
    tl.update(source="tensors", load_s=time.perf_counter() - t, tried=tried)
    if make_image if make_image is not None else os.environ.get("GLM53_IMAGE_BUILD", "0") == "1":
        d, man = build_image(be, io, mesh, root, signature=signature)
        tl.update(image_built=d, image_build_s=man["build_s"])
    return be, tl


# ================================================================================================ ahead of the open
def prefetch_pull_sets(shape, *, root: Optional[str] = None, image: Optional[str] = None, wait: bool = True) -> dict:
    """Host only (no device; a launcher, BEFORE the server opens the mesh): warm the page cache with the pull set the
    next resident restart of `shape` (rows, cols) will need - the recorded stale + clobbered blocks (else the map's
    prediction) of the image it tries first (GLM53_TP_IMAGE_DIR / `image`, else the newest candidate)."""
    root = root or os.environ.get("GLM53_TP_IMAGE_ROOT", DEFAULT_ROOT)
    d = image or os.environ.get("GLM53_TP_IMAGE_DIR") or next(iter(candidates(root, weights_identity(), shape)), None)
    if d is None:
        return dict(image=None)
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    fine = fine_hashes(d, man)
    bb = K.FINE if fine is not None else man["block_bytes"]
    ps = load_pullset(d, man, bb)
    if ps is not None:
        want = {k: sorted(set(ps["clobbered"].get(k, [])) | set(v if isinstance(v, list) else []))
                for k, v in ps["stale"].items()}  # fmt: skip
    else:
        with open(os.path.join(d, "map.json")) as f:
            pred = predicted_clobber(man, json.load(f), bb)
        want = dict.fromkeys(man["chips"], pred)
    jobs = [(chip_file(d, man["chips"][k]), v) for k, v in sorted(want.items())]
    t = time.perf_counter()
    pf = Prefetcher(man, bb, [jobs]).start(detach=not wait)
    if wait:
        while pf.proc is not None and pf.proc.poll() is None:
            time.sleep(0.2)
        st = pf.stop()
    else:
        st = dict(detached=True, items=len(pf.items))
    return dict(
        image=d,
        block_bytes=bb,
        pull_set="recorded" if ps else "predicted",
        planned_bytes=sum(n for _, _, n in pf.items),
        seconds=round(time.perf_counter() - t, 3),
        **st,
    )


def mirror(d: str, root: str, *, chips: Optional[list] = None, verify: bool = True) -> dict:
    """Host only: copy an image's chip files into a local tier `root`/<image key>/ (GLM53_TP_IMAGE_MIRROR=root makes
    restores read them), page-cache-neutral (64-MiB preads / pwrites, fdatasync + DONTNEED every 1 GiB on both sides),
    then verify each copy's block hashes against the manifest (dram_image.file_block_hashes). The metadata stays in d.
    """
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    out = os.path.join(root, os.path.basename(os.path.normpath(d)))
    os.makedirs(out, exist_ok=True)
    res, step, flush = {}, 64 << 20, 1 << 30
    for k, chip in sorted(man["chips"].items(), key=lambda x: int(x[0])):
        if chips is not None and int(k) not in chips:
            continue
        src, dst = os.path.join(d, chip["image"]), os.path.join(out, chip["image"])
        t = time.perf_counter()
        if not (os.path.exists(dst) and os.path.getsize(dst) == chip["bytes"]):
            fi, fo = os.open(src, os.O_RDONLY), os.open(dst + ".tmp", os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
            buf, off, last = bytearray(step), 0, 0
            mv = memoryview(buf)
            try:
                while off < chip["bytes"]:
                    n = os.preadv(fi, [mv[: min(step, chip["bytes"] - off)]], off)
                    if n <= 0:
                        raise OSError(f"short read of {src} at {off}")
                    os.pwrite(fo, mv[:n], off)
                    off += n
                    if off - last >= flush or off >= chip["bytes"]:
                        os.fdatasync(fo)
                        os.posix_fadvise(fo, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.posix_fadvise(fi, last, off - last, os.POSIX_FADV_DONTNEED)
                        last = off
            finally:
                os.close(fi)
                os.close(fo)
            os.replace(dst + ".tmp", dst)
        copy_s = time.perf_counter() - t
        ok = None
        if verify:
            t = time.perf_counter()
            h = _di().file_block_hashes(dst, [tuple(s_) for s_ in man["segments"]], banks=_banks(man),
                                        block_bytes=man["block_bytes"])  # fmt: skip
            ok = [int(x) for x in h["block_hashes"]] == chip["block_hashes"]
            if not ok:
                os.replace(dst, dst + ".bad")  # never read by a restore (size check fails on the missing name)
        res[k] = dict(file=dst, gb=round(chip["bytes"] / 1e9, 3), copy_s=round(copy_s, 2), verified=ok)
    return dict(image=d, mirror=out, chips=res)


def rerule(d: str, record: str, *, host_keep: int, real_tile16: bool) -> dict:
    """Switch an image to other replay rules WITHOUT re-dumping it: the chip files, segments, block hashes and the
    allocation map do not depend on the host-keep threshold / REAL_TILE16 - only map.json's host rows, host_small.pt and
    three manifest fields do. `record`: a record dir of the SAME construction (e.g. a mock record: tp_image_mock.py
    --mode record) made under the new rules; its uploads must give the image's map sha and its tensor map must equal
    the image's. The old files are kept as *.prev; the manifest is replaced last (a reader in between fails its sha
    check -> ImageMismatch -> fallback, never a mixed image)."""
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    with open(os.path.join(record, "map.json")) as f:
        rec = json.load(f)
    with open(os.path.join(d, "map.json")) as f:
        old = json.load(f)
    try:  # the record's identity (tp_image_mock.py --mode record writes identity.json)
        with open(os.path.join(record, "identity.json")) as f:
            ident = json.load(f)
    except OSError:
        raise ImageMismatch(f"record {record}: no identity.json (weights sha / layout / mesh of the record)")
    want = dict(weights=man["weights"]["sha256"], layout=man["layout"], mesh=man["mesh"])
    got_ = dict(weights=ident.get("weights"), layout=ident.get("layout"), mesh=ident.get("mesh"))
    if got_ != want:
        raise ImageMismatch(f"record {record}: identity {got_} != image {want}")
    if map_sha(rec["rows"]) != man["map_sha"]:
        raise ImageMismatch(f"record {record}: map sha {map_sha(rec['rows'])[:12]} != image {man['map_sha'][:12]}")
    if comparable(rec["tensors"]) != comparable(old["tensors"]) or comparable(rec["l1"]) != comparable(old["l1"]):
        raise ImageMismatch(f"record {record}: tensor map differs from the image's")
    blob = json.dumps(dict(rows=rec["rows"], tensors=rec["tensors"], l1=rec["l1"]), separators=(",", ":")).encode()
    with open(os.path.join(record, "host_small.pt"), "rb") as f:
        hs = f.read()
    prev = dict(host_keep=man.get("host_keep"), replay_real_tile16=man.get("replay_real_tile16"),
                map_file_sha=man["map_file_sha"], host_small_sha=man["host_small_sha"])  # fmt: skip
    for name in ("map.json", "host_small.pt", "manifest.json"):  # keep the old rules' files
        src = os.path.join(d, name)
        if os.path.exists(src) and not os.path.exists(src + ".prev"):
            os.link(src, src + ".prev") if not os.path.islink(src) else os.symlink(os.readlink(src), src + ".prev")
    for name, data in (("map.json", blob), ("host_small.pt", hs)):
        with open(os.path.join(d, name + ".tmp"), "wb") as f:
            f.write(data)
        os.replace(os.path.join(d, name + ".tmp"), os.path.join(d, name))
    man.update(map_file_sha=hashlib.sha256(blob).hexdigest(), host_small_sha=hashlib.sha256(hs).hexdigest(),
               host_keep=int(host_keep), replay_real_tile16=bool(real_tile16),
               rerule=dict(record=record, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), previous=prev))  # fmt: skip
    with open(os.path.join(d, "manifest.json.tmp"), "w") as f:
        json.dump(man, f, indent=1)
    os.replace(os.path.join(d, "manifest.json.tmp"), os.path.join(d, "manifest.json"))
    return dict(image=d, host_keep=host_keep, replay_real_tile16=real_tile16, map_bytes=len(blob),
                host_small_bytes=len(hs), previous=prev)  # fmt: skip


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="tp_image host tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prefetch", help="warm the page cache with the next restore's pull set (no device)")
    pp.add_argument("--shape", default="2x4", help="mesh shape RxC")
    pp.add_argument("--image", default=None)
    pp.add_argument("--root", default=None)
    pp.add_argument("--detach", action="store_true", help="return at once; the reader child finishes on its own")
    mm = sub.add_parser("mirror", help="copy an image's chip files into a local tier (GLM53_TP_IMAGE_MIRROR)")
    mm.add_argument("image")
    mm.add_argument("root")
    mm.add_argument("--chips", default=None, help="device ids, e.g. 0,1 (default all)")
    mm.add_argument("--no-verify", action="store_true")
    rr = sub.add_parser("rerule", help="switch an image's replay rules from a record of the same construction")
    rr.add_argument("image")
    rr.add_argument("record")
    rr.add_argument("--host-keep", type=int, required=True)
    rr.add_argument("--real-tile16", type=int, choices=[0, 1], required=True)
    a = ap.parse_args()
    if a.cmd == "mirror":
        cs = [int(v) for v in a.chips.split(",")] if a.chips else None
        print(json.dumps(mirror(a.image, a.root, chips=cs, verify=not a.no_verify)), flush=True)
    elif a.cmd == "prefetch":
        r_, c_ = (int(v) for v in a.shape.split("x"))
        print(json.dumps(prefetch_pull_sets((r_, c_), root=a.root, image=a.image, wait=not a.detach)), flush=True)
    else:
        print(json.dumps(rerule(a.image, a.record, host_keep=a.host_keep, real_tile16=bool(a.real_tile16))), flush=True)
