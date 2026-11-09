# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight sources of the persistent-stack port on this box, shared by the stack tests (test_mk_e2e / test_mk_backend_*)
and the vLLM adapter (models/demos/glm53_flash/tt/generator_vllm_stack.py). StackBackend / TPPrefill / SpecMTP construct
``optimized_decoder.FdTileCacheSource(CACHE_DIR, MANIFEST)`` -- the reference's tile cache, which is not here
(tests/common.py: ``CACHE_DIR = ""``) -- so a run installs one of these in its place:

* ``HFSource`` -- the reference's ``WeightSource`` over our HF checkpoint (per-layer raw dicts dequantised on the host,
  the per-expert tensors, the head / norm / embedding entries).
* ``MkTileCacheSource`` -- GLM53_MK_TILE_CACHE_V3=<dir>: the reference-format tile cache (tools/tile_cache_build.py),
  READ when COMPLETE, else self-extending from an inner ``HFSource`` (every tensor the stack asks for is dumped as it
  loads: ~1 h on a cold runner) and marked COMPLETE by the caller after the build.
* ``TileCache`` -- GLM53_MK_TILE_CACHE=<dir>: the v2 cache of the HF source's dequantised per-layer torch tensors.
* ``open_weight_source`` -- the environment's choice of the three; ``install_weight_source`` -- the same, installed
  as ``optimized_decoder.FdTileCacheSource`` for the process (the tests monkeypatch it instead, restored after).
"""

from __future__ import annotations

import json
import os
import time

import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import optimized_decoder as OD
from models.autoports.zai_org_glm_5_3_flash.tt.weights import TileCacheSource, WeightSource, cache_dtype
from models.demos.glm53_flash.reference.checkpoint import (
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    load_embedding,
    load_layer_raw,
    load_tensors,
    shard_headers,
)

DRAM = ttnn.DRAM_MEMORY_CONFIG


class HFSource(WeightSource):
    """The reference's weight source over our HF checkpoint: ``host`` / ``device`` per layer in the tile-cache layout
    (linear weights [in, out], hyper-connection ``fn`` [16384, 24], conv taps [4, C], ``kv_b_proj`` split into
    ``W_uk`` [h, nope, r] / ``W_uv`` [h, r, v]), the per-expert ``mlp.experts.{e}.gate_up_proj`` [H, 2I] /
    ``down_proj`` [I, H], and the head's ``tensors`` / ``_load`` entries (lm_head [4096, V] bfp8, norm, embed_tokens).
    """

    def __init__(self, snapshot, cfg):
        self.snapshot, self.cfg = snapshot, cfg
        self.headers = shard_headers(snapshot)
        self.raw: dict[int, dict] = {}
        self.cache: dict[tuple[int, str], torch.Tensor] = {}
        V, H = int(cfg.vocab_size), int(cfg.hidden_size)
        self.tensors = {
            "lm_head.weight": {"shape": [H, V], "dtype": "bfloat8_b"},
            "norm.weight": {"shape": [H], "dtype": "bfloat16"},
            "embed_tokens.weight": {"shape": [V, H], "dtype": "bfloat16"},
        }

    def _raw(self, layer_idx: int) -> dict:
        if layer_idx not in self.raw:
            # one layer resident at a time: a sparse layer's raw experts are ~14 GB in bf16 and the backend builds the
            # layers in order (45 layers resident = the host OOM of mk_backend_45)
            for old in [k for k in self.raw if k != layer_idx]:
                del self.raw[old]
            for key in [k for k in self.cache if k[0] != layer_idx]:
                del self.cache[key]
            # return the previous layer's ~14 GB of expert tensors to the OS now (the host RSS otherwise climbs
            # ~4 GB per built layer through the allocator's retained blocks)
            import ctypes
            import gc

            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except OSError:
                pass
            self.raw[layer_idx] = load_layer_raw(self.snapshot, layer_idx, self.headers)
        return self.raw[layer_idx]

    def host(self, layer_idx: int, name: str) -> torch.Tensor:
        key = (layer_idx, name)
        if key in self.cache:
            return self.cache[key]
        raw = self._raw(layer_idx)
        cfg = self.cfg
        if name.startswith("mlp.experts.") and name.endswith(("gate_up_proj", "down_proj")):
            e = name.split(".")[2]
            if name.endswith("gate_up_proj"):
                v = torch.cat(
                    [raw[f"mlp.experts.{e}.gate_proj.weight"].t(), raw[f"mlp.experts.{e}.up_proj.weight"].t()], -1
                )
            else:
                v = raw[f"mlp.experts.{e}.down_proj.weight"].t()
        elif name in ("self_attn.kv_b_proj.W_uk", "self_attn.kv_b_proj.W_uv"):
            h, nope, vd, r = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
            w = raw["self_attn.kv_b_proj.weight"].view(h, nope + vd, r)
            v = w[:, :nope, :] if name.endswith("W_uk") else w[:, nope:, :].mT
        else:
            if name not in raw:
                raise KeyError(f"layer {layer_idx}: {name} not in the raw HF dict ({sorted(raw)[:12]}...)")
            v = raw[name]
            if name in ("hc_attn_fn", "hc_ffn_fn"):
                v = v.t()
            elif "conv1d" in name:
                v = v.reshape(v.shape[0], -1).t()
            elif v.dim() == 2 and (name.endswith(".weight") or name.endswith("index_kpool_compress_gate")):
                v = v.t()  # HF [out, in] -> [in, out] (the indexer's compress gate [128, 4096] is a linear weight too)
        self.cache[key] = v.contiguous()
        return self.cache[key]

    def device(self, layer_idx: int, name: str, mesh_device) -> ttnn.Tensor:
        t = self.host(layer_idx, name)
        dtype = cache_dtype(name)
        return ttnn.from_torch(
            t.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # the head's entries (mk_head.lm_head_host / norm_host / embed_host read `tensors[name]` and `_load(entry)`)
    def _load(self, e: dict, device=None) -> ttnn.Tensor:
        shape = e["shape"]
        if shape == self.tensors["lm_head.weight"]["shape"]:
            w = load_tensors(self.snapshot, [LM_HEAD_KEY], self.headers)[LM_HEAD_KEY].t().contiguous()  # [H, V]
            return ttnn.from_torch(w.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        if shape == self.tensors["norm.weight"]["shape"]:
            w = load_tensors(self.snapshot, [FINAL_NORM_KEY], self.headers)[FINAL_NORM_KEY].reshape(-1)
            return ttnn.from_torch(
                w.to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        w = load_embedding(self.snapshot, self.headers)  # [V, H]
        return ttnn.from_torch(w.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


class MkTileCacheSource(TileCacheSource):
    """GLM53_MK_TILE_CACHE_V3=<dir>: the reference-format tile cache (tools/tile_cache_build.py: one .tensorbin per
    tensor in its cache dtype + manifest.json): the stack's own loaders read it directly (TileCacheSource.device/host,
    fused_decoder.load_fused_experts / load_xsl_experts cut the expert files byte-exactly), no host packing at load
    time. The head's entries (mk_head: source.tensors[name] -> source._load(e)) are the manifest's "lm_head.weight" /
    "norm.weight" / "embed_tokens.weight". SELF-EXTENDING: a tensor the manifest lacks (e.g. the MTP layer 45, which
    only the speculative step asks for) is built on demand from the HF source (`inner`), dumped and added to the
    manifest, so the cache grows with what the stack asks."""

    def __init__(self, cache_dir: str, manifest_path: str, inner=None):
        super().__init__(cache_dir, manifest_path)
        self.inner, self.manifest_path = inner, manifest_path
        self._added = 0
        self.ready = os.path.isfile(os.path.join(cache_dir, "COMPLETE"))

    @classmethod
    def open(cls, cache_dir: str, inner) -> "MkTileCacheSource":
        """the cache at cache_dir, complete or not: a missing manifest starts empty (the run builds every tensor the
        stack asks for, then complete() marks it)"""
        os.makedirs(cache_dir, exist_ok=True)
        mp = os.path.join(cache_dir, "manifest.json")
        if not os.path.isfile(mp):
            with open(mp, "w") as f:
                json.dump({"tensors": {}, "tt_metal_sha": ""}, f)
        src = cls(cache_dir, mp, inner=inner)
        if inner is not None:
            src._seed_head()
        return src

    HEAD = ("lm_head.weight", "norm.weight", "embed_tokens.weight")

    def _seed_head(self) -> None:
        """the head's entries (mk_head reads ``source.tensors[name]`` directly, never through ``_entry``): the ones the
        manifest lacks are built from the HF source into the cache (what tools/tile_cache_build.py does from v2's
        head.pt), so a first run on an empty cache does not die at the head load after building every layer"""
        import hashlib

        added = 0
        for name in self.HEAD:
            if name in self.tensors:
                continue
            e = self.inner.tensors[name]
            h = self.inner._load(e)  # the host tensor in its cache dtype / TILE layout
            rel = f"{name}.tensorbin"
            full = os.path.join(self.cache_dir, rel)
            ttnn.dump_tensor(full, h, mode=ttnn.DumpTensorMode.LOCAL)
            self.tensors[name] = {"path": rel, "shape": [int(v) for v in h.shape],
                                  "dtype": str(h.dtype).split(".")[-1].lower(), "layout": "TILE",
                                  "sha256": hashlib.sha256(name.encode()).hexdigest(), "bytes": os.path.getsize(full)}  # fmt: skip
            added += 1
        if added:
            self._added += added
            self.save_manifest()
            logger.info(f"MkTileCacheSource: {added} head tensors built from the HF source")

    def complete(self) -> None:
        """the manifest saved and COMPLETE written (a later run reads every tensor from the cache)"""
        self.save_manifest()
        with open(os.path.join(self.cache_dir, "COMPLETE"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        self.ready = True
        logger.info(f"MkTileCacheSource {self.cache_dir}: COMPLETE ({len(self.tensors)} tensors, {self._added} built)")

    def _entry(self, layer_idx: int, name: str) -> dict:
        key = f"layers.{layer_idx}.{name}"
        if key not in self.tensors:
            if self.inner is None:
                raise KeyError(f"{key} not in the weight-cache manifest")
            import hashlib

            t = self.inner.host(layer_idx, name)
            dt = cache_dtype(name)
            h = ttnn.from_torch(t.to(torch.float32 if dt == ttnn.float32 else torch.bfloat16), dtype=dt,
                                layout=ttnn.TILE_LAYOUT)  # fmt: skip
            rel = os.path.join(f"L{layer_idx:02d}", f"{name}.tensorbin")
            full = os.path.join(self.cache_dir, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            ttnn.dump_tensor(full, h, mode=ttnn.DumpTensorMode.LOCAL)
            # ("sha256" = the KEY's digest, as tools/tile_cache_build.py writes it: the reference format's identity field,
            # not a checksum of the bytes)
            self.tensors[key] = {"path": rel, "shape": [int(v) for v in t.shape], "dtype": str(dt).split(".")[-1].lower(),
                                 "layout": "TILE", "sha256": hashlib.sha256(key.encode()).hexdigest(),
                                 "bytes": os.path.getsize(full)}  # fmt: skip
            self._added += 1
            # BUILD mode: every 64 (and at the last expert of a layer), complete() saves the rest; READ mode (a COMPLETE
            # cache extended by e.g. the MTP layer): every addition, since nothing else saves it
            if self.ready or self._added % 64 == 1 or (name.endswith("down_proj") and ".experts.287." in name):
                self.save_manifest()
            if self._added == 1:
                logger.info(f"MkTileCacheSource: building layer {layer_idx} on demand from the HF source ({name} ...)")
        return self.tensors[key]

    def save_manifest(self) -> None:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"tensors": self.tensors, "tt_metal_sha": self.tt_metal_sha}, f)
        os.replace(tmp, self.manifest_path)

    def _load(self, e: dict, device=None) -> ttnn.Tensor:
        t = ttnn.load_tensor(os.path.join(self.cache_dir, e["path"]))
        return ttnn.to_device(t, device) if device is not None else t


class TileCache:
    """GLM53_MK_TILE_CACHE=<dir>: the HF source's DEQUANTISED per-layer tensors on disk (torch bf16 / fp32 in the cache
    layout: one L??.pt per layer, head.pt, manifest.json), so the ~27-min host dequantisation of the FP8 checkpoint
    (HFSource.load_layer_raw) is paid once; a run then torch.loads each layer (mmap) and converts to the device exactly
    as HFSource.device does (the same numerics as the plain HF path). Without a COMPLETE marker: BUILD mode -- every
    tensor the stack asks for comes from the HF source, is buffered per layer and torch.saved when the stack moves to
    the next layer (layers already on disk are reused: resumable); the marker is written when the e2e passes. With it:
    READ mode -- the HF source is never opened. (v1 dumped device tiles per tensor: the stack reads the routed experts
    through host() and slices them itself, so that path packed bfp4 on the host at 12 MB/s.)"""

    def __init__(self, inner, cache_dir: str):
        self.inner, self.dir = inner, cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.manifest_path = os.path.join(cache_dir, "manifest.json")
        self.ready = os.path.isfile(os.path.join(cache_dir, "COMPLETE"))
        self.meta = {"layers": {}, "head": {}}
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path) as f:
                self.meta = json.load(f)
        if self.ready and not self.meta["layers"]:
            raise FileNotFoundError(f"{self.manifest_path}: COMPLETE but empty")
        # the head's entries (mk_head: source.tensors[name] -> source._load(e)): the inner's shape / dtype + our key
        self.tensors = {k: dict(v, key=k) for k, v in inner.tensors.items()}
        self._buf: dict = {}  # BUILD: layer -> {name: torch} until the stack moves on (then torch.saved)
        self._res: dict = {}  # the resident cached layer(s) (one at a time: a sparse layer is ~14.5 GB)
        self._head: dict = {}
        logger.info(f"TileCache {cache_dir}: {'READ' if self.ready else 'BUILD'} mode, "
                    f"{len(self.meta['layers'])} layers + {len(self.meta['head'])} head tensors on disk")  # fmt: skip

    def _lpath(self, L: int) -> str:
        return os.path.join(self.dir, f"L{int(L):02d}.pt")

    def _save_meta(self) -> None:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.meta, f)
        os.replace(tmp, self.manifest_path)

    def _flush(self, L: int) -> None:
        d = self._buf.pop(L)
        tmp = self._lpath(L) + ".tmp"
        torch.save({k: v.contiguous() for k, v in d.items()}, tmp)
        os.replace(tmp, self._lpath(L))
        self.meta["layers"][str(L)] = sorted(d)
        self._save_meta()
        logger.info(f"TileCache: layer {L} saved ({len(d)} tensors)")

    def _on_disk(self, L: int) -> bool:
        return str(L) in self.meta["layers"] and os.path.isfile(self._lpath(L))

    def _layer(self, L: int) -> dict:
        if L not in self._res:
            self._res.clear()
            self._res[L] = torch.load(self._lpath(L), mmap=True, weights_only=True)
        return self._res[L]

    def complete(self) -> None:
        for L in list(self._buf):
            self._flush(L)
        self._save_meta()
        with open(os.path.join(self.dir, "COMPLETE"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        self.ready = True
        logger.info(f"TileCache {self.dir}: COMPLETE ({len(self.meta['layers'])} layers)")

    # ------------------------------------------------------------------ the WeightSource surface
    def host(self, L: int, name: str) -> torch.Tensor:
        if self._on_disk(L):
            d = self._layer(L)
            if name in d:
                return d[name]
            if self.ready:
                raise KeyError(f"layer {L}: {name} not in the tile cache {self.dir} (remove COMPLETE to rebuild)")
        elif self.ready:
            raise KeyError(f"layer {L} not in the tile cache {self.dir} (remove COMPLETE to rebuild)")
        t = self.inner.host(L, name)
        for old in [k for k in self._buf if k != L]:  # the stack builds the layers in order
            self._flush(old)
        self._buf.setdefault(L, {})[name] = t
        return t

    def device(self, L: int, name: str, mesh) -> ttnn.Tensor:
        t = self.host(L, name)
        dtype = cache_dtype(name)
        return ttnn.from_torch(
            t.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def experts(self, *args, **kwargs):
        return self.inner.experts(*args, **kwargs)

    def _load(self, e: dict, device=None) -> ttnn.Tensor:
        key = e["key"]
        hp = os.path.join(self.dir, "head.pt")
        m = self.meta["head"].get(key)
        if m is not None and os.path.isfile(hp):
            if not self._head:
                self._head = torch.load(hp, mmap=True, weights_only=True)
            t = ttnn.from_torch(self._head[key], dtype=getattr(ttnn, m["dtype"]), layout=getattr(ttnn, m["layout"]))
        else:
            if self.ready:
                raise KeyError(f"{key} not in the tile cache {self.dir} (remove COMPLETE to rebuild)")
            t = self.inner._load(e)
            w = ttnn.to_torch(t)
            d = torch.load(hp, weights_only=True) if os.path.isfile(hp) else {}
            d[key] = w.contiguous()
            torch.save(d, hp + ".tmp")
            os.replace(hp + ".tmp", hp)
            self.meta["head"][key] = {"dtype": str(t.dtype).split(".")[-1].lower(),
                                      "layout": str(t.layout).split(".")[-1] + "_LAYOUT", "shape": list(e["shape"])}  # fmt: skip
            self._save_meta()
        return ttnn.to_device(t, device) if device is not None else t


def open_weight_source(snapshot, cfg):
    """The run's weight source: GLM53_MK_TILE_CACHE_V3=<dir> -> ``MkTileCacheSource`` over an ``HFSource`` (READ when
    COMPLETE, else BUILT by this run), else the ``HFSource`` itself, under GLM53_MK_TILE_CACHE=<dir> wrapped in the v2
    ``TileCache``. ``snapshot``: the HF checkpoint directory; ``cfg``: its text config (``load_hf_text_config``)."""
    v3 = os.environ.get("GLM53_MK_TILE_CACHE_V3")
    if v3:
        source = MkTileCacheSource.open(v3, inner=HFSource(snapshot, cfg))
        logger.info(
            f"MkTileCacheSource {v3}: {len(source.tensors)} entries, {'READ' if source.ready else 'BUILD'} mode"
        )
        return source
    source = HFSource(snapshot, cfg)
    if os.environ.get("GLM53_MK_TILE_CACHE"):  # the dequantised per-layer torch files (see TileCache)
        source = TileCache(source, os.environ["GLM53_MK_TILE_CACHE"])
    return source


def install_weight_source(snapshot, cfg):
    """``open_weight_source`` + the process-wide patch the stack's construction needs (what test_mk_e2e.py
    monkeypatches): every ``optimized_decoder.FdTileCacheSource(cache_dir, manifest)`` the backend / TP prefill /
    spec step constructs returns this source. Returns the source; a ``MkTileCacheSource`` built by the run is the
    caller's to ``complete()`` once the backend stands (the e2e test does it right after the build)."""
    source = open_weight_source(snapshot, cfg)
    OD.FdTileCacheSource = lambda cache_dir, manifest: source
    return source
