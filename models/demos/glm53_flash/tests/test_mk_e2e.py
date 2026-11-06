# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end run of the persistent-stack port through the reference's own generator (Glm53Generator over
StackBackend): B users with the same chat prompt of ISL tokens, prefill + OSL greedy tokens; reports TTFT (the batch's
prefill wall time up to the first token), the decode step and tok/s per user, the per-user text agreement and user 0's
text. Prefill runs through the backend's decode-program chunks (GLM53_STACK_PREFILL=decode, the default).

    GLM53_MESH_SHAPE=8,4 GLM53_FABRIC=1d GLM53_MK_EP=1 GLM53_STACK_DSA=1 GLM53_MK_B=4 GLM53_MK_ISL=128 GLM53_MK_OSL=128 \
        pytest .../test_mk_e2e.py -s
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import optimized_decoder as OD
from models.autoports.zai_org_glm_5_3_flash.tt.config import load_hf_text_config
from models.autoports.zai_org_glm_5_3_flash.tt.weights import TileCacheSource, cache_dtype
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.test_mk_backend_probe import HFSource
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
B = int(os.environ.get("GLM53_MK_B", "4"))
ISL = int(os.environ.get("GLM53_MK_ISL", "128"))
OSL = int(os.environ.get("GLM53_MK_OSL", "128"))
SEQ = int(os.environ.get("GLM53_MK_SEQ", str(max(2048, 1 << (ISL + OSL + 64).bit_length()))))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
# the reference's SERVED worker L1 (pmk_layer WORKER_L1_SERVED): 16 KB more allocatable L1 than the mk_dense note's
# 1,264,512 -- the 45-layer stack sits at that floor (the sharded pool / the attached TP prefill clashed by 1 KB,
# 2026-11-05: "static circular buffers clash with L1 buffers ... 1269248 ... 1270272")
WORKER_L1 = int(os.environ.get("GLM53_MK_WORKER_L1", "1280896"))
FILLER = (
    "The Blackhole Galaxy is a 32-chip system; its chips talk over an Ethernet fabric and each one holds its own DRAM. "
    "A decoder layer of a mixture-of-experts model routes every token to a few experts, and the shared expert sees "
    "every token. Hyper-connections keep four residual streams per token and mix them before each sublayer. "
)
QUESTION = "Summarize the text above in two sentences."


def _layers() -> list[int]:
    spec = os.environ.get("GLM53_MK_LAYERS", "0-44")
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def chat_prompt(tok, isl: int, user: int = 0) -> list[int]:
    """a chat-rendered prompt of (about) isl tokens: filler text cut to the token budget + a question; or the text of
    GLM53_MK_PROMPT (=@file) as the whole user message (the HF greedy reference's prompt: ISL = its length).
    GLM53_MK_DISTINCT=1: user u's filler starts 7 u tokens into the text (the users route to different experts:
    every user of a batch with the same prompt picks the same experts, which flatters the EP expert phase)."""
    fixed = os.environ.get("GLM53_MK_PROMPT", "")
    if fixed.startswith("@"):
        with open(fixed[1:]) as f:
            fixed = f.read().strip()
    if fixed:
        return tok.encode_chat([{"role": "user", "content": fixed}])
    overhead = len(tok.encode_chat([{"role": "user", "content": QUESTION}]))
    budget = max(8, isl - overhead - 2)
    skip = 7 * user if os.environ.get("GLM53_MK_DISTINCT", "0") == "1" else 0
    filler_ids = tok.encode(FILLER * ((budget + skip) // 16 + 2))[skip : skip + budget]
    content = tok.decode(filler_ids) + "\n\n" + QUESTION
    ids = tok.encode_chat([{"role": "user", "content": content}])
    return ids


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
            self.tensors[key] = {"path": rel, "shape": [int(v) for v in t.shape], "dtype": str(dt).split(".")[-1].lower(),
                                 "layout": "TILE", "sha256": hashlib.sha256(key.encode()).hexdigest(),
                                 "bytes": os.path.getsize(full)}  # fmt: skip
            self._added += 1
            if self._added % 64 == 1 or name.endswith("down_proj") and ".experts.287." in name:
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


@pytest.mark.timeout(7000)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20, worker_l1_size=WORKER_L1)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_e2e(mesh_device, monkeypatch):
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    tok = load_tokenizer(snapshot)
    v3 = os.environ.get("GLM53_MK_TILE_CACHE_V3")
    if v3 and os.path.isfile(os.path.join(v3, "COMPLETE")):  # the reference-format tile cache (MkTileCacheSource)
        source = MkTileCacheSource(v3, os.path.join(v3, "manifest.json"), inner=HFSource(snapshot, cfg))
        logger.info(f"MkTileCacheSource {v3}: {len(source.tensors)} entries")
    else:
        source = HFSource(snapshot, cfg)
        if os.environ.get("GLM53_MK_TILE_CACHE"):  # the dequantised per-layer torch files (see TileCache)
            source = TileCache(source, os.environ["GLM53_MK_TILE_CACHE"])
    monkeypatch.setattr(OD, "FdTileCacheSource", lambda cache_dir, manifest: source)
    layers = _layers()
    ep = os.environ.get("GLM53_MK_EP", "1") == "1"
    native = os.environ.get("GLM53_MK_NATIVE_T", "1" if B > 4 else "0") == "1"
    t0 = time.perf_counter()
    backend = StackBackend(mesh_device, max_batch=B, layers=layers, hf_config=cfg, topk=1, sample=False, ep_weights=ep,
                           native_t=native)  # fmt: skip
    logger.info(f"StackBackend built in {time.perf_counter() - t0:.0f} s: mesh {MESH}, {len(layers)} layers, B {B}")
    gen = Glm53Generator(backend, tokenizer=tok, max_batch=B, max_seq_len=SEQ)
    t0 = time.perf_counter()
    gen.warmup(capture=True)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"generator warm-up (prefill buckets, decode, traces) {time.perf_counter() - t0:.1f} s")
    ids = chat_prompt(tok, ISL)
    # GLM53_MK_LIVE=n (default B): the live users of the run on a backend built for B rows (the speculative step,
    # GLM53_SPEC_K, serves ONE live user in slot 0 of a backend whose max_batch >= K + 1)
    live = int(os.environ.get("GLM53_MK_LIVE", str(B)))
    prompts = [list(chat_prompt(tok, ISL, user=u)) for u in range(live)]
    logger.info(f"prompt {len(ids)} tokens (ISL target {ISL}; per user {[len(p) for p in prompts]}); OSL {OSL}; "
                f"cache {B} x {SEQ}; distinct {os.environ.get('GLM53_MK_DISTINCT', '0')}")  # fmt: skip
    # TTFT: prefill of every user + the first token (the generator's prefill chunks + one sampler step)
    t0 = time.perf_counter()
    first = gen.generate_batch(prompts, 1, enable_trace=True)
    ttnn.synchronize_device(mesh_device)
    ttft_s = time.perf_counter() - t0
    gen.reset()
    t0 = time.perf_counter()
    outs = gen.generate_batch(prompts, OSL, enable_trace=True)
    ttnn.synchronize_device(mesh_device)
    total_s = time.perf_counter() - t0
    n_gen = max(len(o) for o in outs)
    step_ms = (total_s - ttft_s) / max(1, n_gen - 1) * 1e3
    same = all(o == outs[0] for o in outs)
    text = tok.decode([t for t in outs[0] if 0 <= t < tok.vocab_size])
    logger.info(
        f"MK_E2E: mesh {MESH} {len(layers)} layers B {B} live {live} spec_k {os.environ.get('GLM53_SPEC_K', '0')} "
        f"ISL {len(ids)} OSL {OSL}: TTFT {ttft_s * 1e3:.0f} ms "
        f"(all {live} users), decode {step_ms:.1f} ms/step = {1e3 / step_ms:.1f} tok/s/u, generated {n_gen}, "
        f"users agree {same}, first {first[0][:1]} ids {outs[0][:24]}"
    )
    logger.info(f"MK_E2E_TEXT: {text!r}")
    logger.info(f"MK_E2E counters: {gen.counters_report()}")
    assert n_gen >= 1
    if isinstance(source, TileCache) and not source.ready:
        source.complete()  # the e2e passed on the tiles it dumped: READ mode from the next run
    if isinstance(source, MkTileCacheSource) and source._added:
        source.save_manifest()
        logger.info(f"MkTileCacheSource: {source._added} tensors added to the manifest")
