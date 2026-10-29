# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's MoE layers on the MoE block (models/demos/moe_block), K3_MOE_IMPL=block: router, shared expert, latent
projections and routed experts in one call, on the post-ffn_norm residual as it is.

    x [1, 1, S/SP, H/TP] bf16 TILE (ffn_norm's output: rows over the SP axis, columns over TP)
    -> MoEBlock (KIMI_K3: 896 experts top-16, latent 3584 + its norm, SiTU shared expert 6144)
    -> routed + shared [1, 1, S/SP, H/TP] bf16 TILE (what TtMoe returns, unsqueezed)

GLM-5.3 (256 experts top-8, H 6144) too, through TtPrefillBlock (MOE_IMPL=block): weka only - the gate and shared expert
from $GLM53_HF_MODEL (fp8 dequantized), the routed experts from the model's own weight cache (cache_path).

K3_BLOCK_WEIGHTS:
  random (default) - seeded random weights: four distinct experts repeated over the 896 (the block tests' pool), one
      set for every layer (the routed experts laid out once). The integration does not depend on their values.
  ci - random's weights with the router replaced by the one TtMoe's CI perf gate (test_kimi_moe_perf) runs:
      $K3_BLOCK_CI_GATE (torch.save of run_model's gate_weights, MOE_X_DUMP) or create_gate_weights(seed 9012), its
      rows permuted to the block's expert order as weka's are - so the block routes CI's x exactly as TtMoe does.
  weka - the model's own weights, per layer: the gate (bf16, rows permuted to the block's expert order), the shared
      expert and the latent pair from the checkpoint ($KIMI_K3_CKPT); the routed experts read back from TtMoe's ttnn
      cache ($K3_BLOCK_WEKA_CACHE, default CI's on Weka) - bfp4 tensorbins, one per (local expert, projection), each
      chip's expert of that slot - dequantized on the host and laid out again the block's way. Nothing is written;
      every run reads and converts again (~15 GB of bfp4 a layer: one layer first).

Expert order: TtMoe holds checkpoint expert gtab[c, r, le] in slot le of chip (r, c) (ExpertMapping, column-major);
the block holds expert b = (r C + c) EPC + le there. The experts stay where TtMoe put them: block expert b is the
checkpoint's gtab[c, r, le], and the gate's rows (and bias) are permuted to match.

One block per (mesh, chunk) serves every layer, built at the first call (after the model's own ops placed their L1:
the exchange's arena address is baked into its kernels); with weka weights each layer holds its own MoELayer.
K3_BLOCK_CHECK=N: the first N calls of each layer checked against a host model of the block on the device's own
routing (PCC logged; > 0.98 asserted).
"""

from __future__ import annotations

import dataclasses
import os
import resource
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn

LINKS = 2  # (the block's exchange plans its ports for 2)
POOL = 4  # (distinct random experts, repeated over the 896)
_BLOCKS: dict = {}  # (id(mesh), chunk) -> (mesh, MoEBlock, the weights it was built with)
_POOL_EXPERTS: dict = {}  # (id(mesh), chunk) -> (mesh, RoutedExperts): real_gate's random experts, shared by every layer
_DONE: dict = {}  # id(mesh) -> (mesh, the done words every layer's flat op shares - L1: one a layer moved the arena)
_FACE_MAPS: dict = {}  # (H, I, E, the layout's packing fields) -> flat_gather.flat_face_map: the same every layer


def _cache_dir(mesh):
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import _checkpoint_identity

    R, C = tuple(mesh.shape)
    ckpt = Path(os.environ.get("KIMI_K3_CKPT", ""))
    root = Path(os.environ.get("K3_BLOCK_WEKA_CACHE", _WEKA_CACHE))
    return root / f"kimi_k3_bh_{R * C}dev" / _checkpoint_identity(ckpt) / f"mesh{R}x{C}.tpaxis1"


def weka_device_sources(mesh, layer_idx: int, epc: int, cache=None):
    """TtMoe's cached routed experts of a layer straight to DRAM (bfp4 as stored, interleaved): flat_gather's sources,
    s = le * 3 + (0 gate, 1 up, 2 down) - each [1, 1, in, out] a chip (W^T: the flat op's orientation)."""
    cache = Path(cache) if cache is not None else _cache_dir(mesh)
    srcs = []
    for le in range(epc):
        for proj in ("gate", "up", "down"):
            hits = sorted(cache.glob(f"layer_{layer_idx}.routed_expert.local_{le}_{proj}_dtype_*.tensorbin"))
            assert hits, f"K3_BLOCK_WEIGHTS=weka: no layer_{layer_idx}.routed_expert.local_{le}_{proj} in {cache}"
            assert "BFLOAT4_B" in hits[0].name, f"flat_gather copies bfp4 faces: {hits[0].name}"
            host = ttnn.load_tensor(str(hits[0]))
            srcs.append(ttnn.to_device(host, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            del host
    return srcs
_WEKA_CACHE = "/mnt/weka/model-cache/scratch/moonshotai/Kimi-K3-Cache/Kimi-K3-Cache-prefill"


def block_model(model_cfg):
    """The block's model (KIMI_K3 / GLM_53), checked against the model's config."""
    from models.demos.moe_block.moe_block import GLM_53, KIMI_K3

    if model_cfg.NUM_ROUTED_EXPERTS == GLM_53.num_experts and model_cfg.EMB_SIZE == GLM_53.emb_dim:
        m = GLM_53
        got = (model_cfg.NUM_ROUTED_EXPERTS, model_cfg.NUM_EXPERTS_PER_TOKEN, model_cfg.EMB_SIZE, model_cfg.MOE_INTERMEDIATE_SIZE)
        want = (m.num_experts, m.top_k, m.emb_dim, m.routed_expert_hidden_dim)
        assert got == want, f"MOE_IMPL=block: config (E, top-k, H, I) {got} != GLM_53 {want}"
        return m
    m = KIMI_K3
    got = (
        model_cfg.NUM_ROUTED_EXPERTS,
        model_cfg.NUM_EXPERTS_PER_TOKEN,
        model_cfg.EMB_SIZE,
        model_cfg.MOE_INTERMEDIATE_SIZE,
        model_cfg.ROUTED_EXPERT_HIDDEN_SIZE,
    )
    want = (m.num_experts, m.top_k, m.emb_dim, m.routed_expert_hidden_dim, m.latent_dim)
    assert got == want, f"K3_MOE_IMPL=block: config (E, top-k, H, I, latent) {got} != KIMI_K3 {want}"
    return m


def random_weights(m, seed=0):
    """Seeded random MoEWeights at m's shapes (the block tests' generator: POOL distinct experts over all E)."""
    from models.demos.moe_block.moe_block import MoEWeights, fuse_gate_up

    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.02: torch.randn(*s, generator=g) * sc  # noqa: E731
    H, Hr, I, E = m.emb_dim, m.routed_width, m.routed_expert_hidden_dim, m.num_experts
    pool = [{"gate_up": fuse_gate_up(r(I, Hr), r(I, Hr)), "down": r(Hr, I)} for _ in range(POOL)]
    Is = m.shared_expert_hidden_dim
    return MoEWeights(
        router={"weight": r(E, H), "bias": r(E, sc=0.01)},
        routed=[pool[e % POOL] for e in range(E)],
        shared={"gate_up": fuse_gate_up(r(Is, H), r(Is, H)), "down": r(H, Is)},
        latent={"down": r(Hr, H), "up": r(H, Hr), "norm": 1.0 + r(Hr, sc=0.1)},
    )


def ci_router(mesh, m):
    """TtMoe CI perf gate's router in the block's expert order (module docstring: ci)."""
    path = os.environ.get("K3_BLOCK_CI_GATE")
    if path:
        gate = torch.load(path)["gate"]
    else:
        from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_gate_weights

        gate = create_gate_weights(m.num_experts, m.emb_dim, seed=9012)
    R, C = tuple(mesh.shape)
    epc = m.num_experts // (R * C)
    gtab = _gtab(mesh, epc)
    perm = torch.tensor([int(gtab[c, r, le]) for r in range(R) for c in range(C) for le in range(epc)], dtype=torch.long)
    return {"weight": gate["weight"].float()[perm].contiguous(), "bias": gate["e_score_correction_bias"].float()[perm]}


def _gtab(mesh, epc):
    """gtab[c, r, le]: the checkpoint expert TtMoe holds in slot le of chip (r, c)."""
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping

    R, C = tuple(mesh.shape)
    return ExpertMapping.create_global_expert_idx_table(experts_per_chip=epc, dispatch_group_size=R, num_dispatch_groups=C)


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def glm_nonrouted(layer_idx: int, hf_config):
    """GLM-5.3 layer's gate (weight, e_score_correction_bias) and shared expert from $GLM53_HF_MODEL, fp8 dequantized."""
    import json

    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_tensors
    from models.demos.deepseek_v3_d_p.utils.test_utils import dequantize_state_dict

    ckpt = Path(os.environ.get("GLM53_HF_MODEL", "/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4"))
    with (ckpt / "model.safetensors.index.json").open(encoding="utf-8") as f:
        wm = json.load(f)["weight_map"]
    p = f"model.layers.{layer_idx}.mlp."
    names = [k for k in wm if k.startswith(p + "gate.") or k.startswith(p + "shared_experts.")]
    assert names, f"MOE_IMPL=block: no {p}gate / shared_experts keys in {ckpt}"
    t = load_tensors(ckpt, {k: k for k in names})
    if hf_config is None or getattr(hf_config, "quantization_config", None) is None:
        import types

        with (ckpt / "config.json").open(encoding="utf-8") as f:  # (only its quantization_config is read)
            hf_config = types.SimpleNamespace(quantization_config=json.load(f).get("quantization_config"))
    t = dequantize_state_dict(t, hf_config, dtype=torch.float32)
    return t, p


def weka_weights(mesh, m, layer_idx: int, routed: bool = True, cache=None, hf_config=None):
    """The layer's own weights in the block's expert order (module docstring) -> (MoEWeights with routed None: gate /
    shared / latent from the checkpoint, flat): flat = per chip (row-major), per local expert (Wg [Hr, I], Wu [Hr, I],
    Wd [I, Hr]) bf16 - TtMoe's cached W^T is the flat op's orientation as it is; block expert k EPC + le = flat[k][le]
    (routed False: flat None - real_gate's). bf16, not the contract's fused fp32 (RoutedExperts' conversion of 896
    distinct experts held ~120 GB of fp32 at once: the host OOM-killed the run)."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import (
        GATE_KEYS,
        LATENT_KEYS,
        SHARED_EXPERT_KEYS,
        _checkpoint_identity,
        layer_prefix,
        load_tensors,
    )
    from models.demos.moe_block.moe_block import MoEWeights, fuse_gate_up

    t0 = time.time()
    R, C = tuple(mesh.shape)
    E, epc = m.num_experts, m.num_experts // (R * C)
    gtab = _gtab(mesh, epc)
    perm = torch.tensor([int(gtab[c, r, le]) for r in range(R) for c in range(C) for le in range(epc)], dtype=torch.long)

    if m.latent_dim is None:  # (GLM-5.3)
        t, p = glm_nonrouted(layer_idx, hf_config)
        router = {"weight": t[p + "gate.weight"].float()[perm].contiguous(), "bias": t[p + "gate.e_score_correction_bias"].float()[perm]}
        sp = p + "shared_experts."
        shared = {
            "gate_up": fuse_gate_up(t[sp + "gate_proj.weight"].float(), t[sp + "up_proj.weight"].float()),
            "down": t[sp + "down_proj.weight"].float(),
        }
        latent = None
    else:
        ckpt = Path(os.environ.get("KIMI_K3_CKPT", ""))
        assert (ckpt / "model.safetensors.index.json").is_file(), f"K3_BLOCK_WEIGHTS=weka: no checkpoint at {ckpt}"
        p = layer_prefix(ckpt, layer_idx)
        keys = {f"g.{k}": p + v for k, v in GATE_KEYS.items()}
        keys.update({f"s.{k}": p + v for k, v in SHARED_EXPERT_KEYS.items()})
        keys.update({f"l.{k}": p + v for k, v in LATENT_KEYS.items()})
        t = load_tensors(ckpt, keys)
        router = {"weight": t["g.weight"].float()[perm].contiguous(), "bias": t["g.e_score_correction_bias"].float()[perm]}
        shared = {"gate_up": fuse_gate_up(t["s.gate_proj"].float(), t["s.up_proj"].float()), "down": t["s.down_proj"].float()}
        latent = {"down": t["l.down_proj"].float(), "up": t["l.up_proj"].float(), "norm": t["l.norm"].float()}
    t1 = time.time()

    w = MoEWeights(router=router, routed=None, shared=shared, latent=latent)
    if not routed:
        logger.info(f"K3 real_gate layer {layer_idx}: gate / shared / latent from the checkpoint in {t1 - t0:.0f} s")
        return w, None
    cache = Path(cache) if cache is not None else _cache_dir(mesh)
    stem = f"layer_{layer_idx}.routed_expert"
    flat = [[None] * epc for _ in range(R * C)]
    for le in range(epc):
        mats = {}
        for proj in ("gate", "up", "down"):
            hits = sorted(cache.glob(f"{stem}.local_{le}_{proj}_dtype_*.tensorbin"))
            assert hits, f"K3_BLOCK_WEIGHTS=weka: no {stem}.local_{le}_{proj} tensorbin in {cache}"
            host = ttnn.load_tensor(str(hits[0]))  # (host, every chip's [1, 1, in, out] - TtMoe stores W^T)
            mats[proj] = [ttnn.to_torch(d).reshape(d.shape[-2], d.shape[-1]) for d in ttnn.get_device_tensors(host)]
            del host
        for k in range(R * C):  # (row-major chips: the mapper's dims (0, 1))
            flat[k][le] = tuple(mats[n][k].to(torch.bfloat16).contiguous() for n in ("gate", "up", "down"))
        del mats
        if le == 0:
            wg, wu, wd = flat[0][0]
            logger.info(
                f"K3 weka layer {layer_idx}: routed expert slot 0 Wg {tuple(wg.shape)} Wu {tuple(wu.shape)} Wd "
                f"{tuple(wd.shape)} (expect [{m.routed_width}, {m.routed_expert_hidden_dim}] x 2 / "
                f"[{m.routed_expert_hidden_dim}, {m.routed_width}]) from {cache}"
            )
    logger.info(
        f"K3 weka layer {layer_idx}: checkpoint {t1 - t0:.0f} s, {epc} x 3 routed tensorbins {time.time() - t1:.0f} s; "
        f"max RSS {_rss_gb():.0f} GB"
    )
    return w, flat


def host_moe(x, w, m, ids, wts, flat=None):
    """The block's math in fp32 on the host for any routed experts (the matrix test's reference assumes a pool):
    expert e from w.routed (the contract), or from flat (weka: per chip, per local expert (Wg, Wu, Wd))."""
    from models.demos.moe_block.moe_block import act_host, split_gate_up

    epc = None if flat is None else len(flat[0])

    def expert(e):  # -> gate [I, Hr], up [I, Hr], down [Hr, I]
        if flat is not None:
            wg, wu, wd = flat[e // epc][e % epc]
            return wg.float().T, wu.float().T, wd.float().T
        gate, up = split_gate_up(w.routed[e]["gate_up"].float())
        return gate, up, w.routed[e]["down"].float()

    x = x.float()
    h = x if m.latent_dim is None else x @ w.latent["down"].float().T
    acc = torch.zeros(x.shape[0], m.routed_width)
    for e in torch.unique(ids).tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        gate, up, down = expert(e)
        a = act_host(m.routed_expert_activation, h[tok] @ gate.T, h[tok] @ up.T)
        acc.index_add_(0, tok, (a @ down.T) * wts[tok, kk].float().unsqueeze(1))
    y = acc
    if m.latent_dim is not None:
        if m.latent_norm:
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + m.norm_eps) * w.latent["norm"].float()
        y = y @ w.latent["up"].float().T
    if w.shared is not None:
        gs, us = split_gate_up(w.shared["gate_up"].float())
        y = y + act_host(m.shared_expert_activation, x @ gs.T, x @ us.T) @ w.shared["down"].float().T
    return y


def device_routing(rows, S, K):
    """The joint routing rows (per chip [S / 32, 32 x 2K] uint16, row-major chip order) -> ids, weights [N S, K]."""
    ids, wts = [], []
    for t in rows:
        u = t.to(torch.int64).reshape(S, 2 * K) & 0xFFFF
        ids.append(u[:, :K])
        wb = u[:, K:]
        wts.append(torch.where(wb >= 32768, wb - 65536, wb).to(torch.int16).view(torch.bfloat16).float())
    return torch.cat(ids), torch.cat(wts)


class _L1Trace:
    """Logs every L1 tensor allocated / freed through ttnn's Python entry points while open (K3_BLOCK_L1_TRACE=1): its
    address, bytes a core and the caller - where the block's persistent L1 lands and which transient ones left holes
    above it."""

    def __init__(self):
        import traceback

        self.n = 0
        self.saved = {}

        def where():
            for fr in reversed(traceback.extract_stack()[:-2]):
                if ("models/demos" in fr.filename or "ttnn/bringup" in fr.filename) and "moe_block_ffn" not in fr.filename:
                    return f"{Path(fr.filename).name}:{fr.lineno} {fr.name}"
            return "?"

        def per_core(t):
            ss = t.memory_config().shard_spec
            return ss.shape[0] * ss.shape[1] * t.element_size() if ss is not None else -1

        def wrap_alloc(name):
            fn = getattr(ttnn, name)
            self.saved[name] = fn

            def call(*a, **k):
                t = fn(*a, **k)
                try:
                    if (
                        isinstance(t, ttnn.Tensor)
                        and t.storage_type() == ttnn.StorageType.DEVICE
                        and t.memory_config().buffer_type == ttnn.BufferType.L1
                    ):
                        self.n += 1
                        logger.info(f"K3_L1_TRACE +{self.n} @{int(t.buffer_address())} {per_core(t)} B {name} {where()}")
                except Exception:  # noqa: BLE001 (a diagnostic)
                    pass
                return t

            setattr(ttnn, name, call)

        for name in ("allocate_tensor_on_device", "from_torch"):
            wrap_alloc(name)
        dealloc = ttnn.deallocate
        self.saved["deallocate"] = dealloc

        def free(t, *a, **k):
            try:
                if t.is_allocated() and t.memory_config().buffer_type == ttnn.BufferType.L1:
                    logger.info(f"K3_L1_TRACE - @{int(t.buffer_address())} {per_core(t)} B {where()}")
            except Exception:  # noqa: BLE001
                pass
            return dealloc(t, *a, **k)

        ttnn.deallocate = free
        gs = ttnn.create_global_semaphore
        self.saved["create_global_semaphore"] = gs

        def sem(*a, **k):
            s_ = gs(*a, **k)
            try:
                logger.info(f"K3_L1_TRACE +sem @{int(ttnn.get_global_semaphore_address(s_))} {where()}")
            except Exception:  # noqa: BLE001
                pass
            return s_

        ttnn.create_global_semaphore = sem

    def close(self):
        for name, fn in self.saved.items():
            setattr(ttnn, name, fn)


class TtK3MoEBlockFfn:
    """One K3 MoE layer on the shared MoE block: __call__(ffn_norm's x [1, 1, S/SP, H/TP]) -> [1, 1, S/SP, H/TP]."""

    def __init__(self, mesh_device, model_cfg, layer_idx: int, seed: int = 0, hf_config=None, cache_path=None):
        self.hf_config, self.cache_path = hf_config, cache_path  # (GLM: the dequant config, the routed experts' cache)
        self.kind = os.environ.get("K3_BLOCK_WEIGHTS", "random" if cache_path is None else "weka")
        assert self.kind in ("random", "ci", "weka", "real_gate"), f"K3_BLOCK_WEIGHTS={self.kind}: random, ci, weka or real_gate"
        self.mesh, self.layer_idx, self.seed = mesh_device, int(layer_idx), int(seed)
        self.model = block_model(model_cfg)
        self.calls = 0
        self.w = None  # (weka / real_gate: this layer's host weights; weka's routed experts in self.flat)
        self.flat = None  # (weka: per chip, per local expert (Wg, Wu, Wd) bf16 - dropped once laid out and checked)
        # (weka: the flat op's layout made on the device from TtMoe's bfp4 cache - flat_gather - unless K3_BLOCK_LAYOUT
        # =host: dequantized, packed and requantized on the host, ~4 min a layer)
        self.device_layout = self.kind == "weka" and os.environ.get("K3_BLOCK_LAYOUT", "device") == "device"
        self.gather_check = self.device_layout and os.environ.get("K3_BLOCK_GATHER_CHECK") == "1"
        self.layer = None  # (weka / real_gate: this layer's MoELayer, the block switches between them)

    def _l1_list(self, block, layer):
        """The L1 tensors the block keeps between calls: where, how much a core (the top of L1 is what the model's
        other programs' circular buffers need - a persistent L1 buffer there clashes with them)."""
        from models.demos.moe_block.moe_block import l1_tensors

        roots = [(block, "block"), (layer, "layer"), (_DONE, "_DONE"), (_POOL_EXPERTS, "_POOL_EXPERTS")]
        rows = []
        for path, t in l1_tensors(roots):
            mc = t.memory_config()
            ss = mc.shard_spec
            if ss is not None:
                cores = ss.grid.num_cores()
                per_core = ss.shape[0] * ss.shape[1] * t.element_size()
                where = f"{cores} cores {ss.grid}"
            else:
                cores = 0
                per_core = -1
                where = str(mc.memory_layout)
            rows.append((int(t.buffer_address()), per_core, path, tuple(t.shape), str(t.dtype), where))
        rows.sort(key=lambda r: r[0])
        tot = sum(r[1] for r in rows if r[1] > 0)
        logger.info(f"K3_BLOCK_L1 {len(rows)} persistent L1 tensors, {tot >> 10} KB a core summed (sharded ones):")
        for a, b, path, shape, dt, where in rows:
            logger.info(f"K3_BLOCK_L1   @{a} {b} B a core  {path}  {shape} {dt}  {where[:160]}")
        if rows:
            lo, hi = rows[0][0], max(a + max(b, 0) for a, b, *_ in rows)
            logger.info(f"K3_BLOCK_L1 span: lowest @{lo}, highest end {hi} -> {(hi - lo) >> 10} KB from the lowest")

    def _l1_report(self, when):
        """The L1 the model holds when the block is built: free a core (to the lowest allocation), the allocator's view."""
        from models.demos.moe_block.tt.se_engine3 import ARENA_MAX, l1_free_bytes

        mv = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1)
        logger.info(
            f"K3 MoE block L1 {when}: free below the lowest allocation {l1_free_bytes(self.mesh) >> 10} KB a core "
            f"(ARENA_MAX {ARENA_MAX >> 10} KB); allocator: allocated {int(mv.total_bytes_allocated_per_bank) >> 10} KB, "
            f"free {int(mv.total_bytes_free_per_bank) >> 10} KB, largest free block "
            f"{int(mv.largest_contiguous_bytes_free_per_bank) >> 10} KB a bank"
        )
        ms = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1_SMALL)
        logger.info(
            f"K3 MoE block L1_SMALL {when}: allocated {int(ms.total_bytes_allocated_per_bank)} B, free "
            f"{int(ms.total_bytes_free_per_bank)} B, largest free block {int(ms.largest_contiguous_bytes_free_per_bank)} B a "
            f"bank"
        )

    def _weights(self):
        if self.w is None:
            if self.kind == "random":
                self.w = random_weights(self.model, self.seed)
            elif self.kind == "ci":
                self.w = dataclasses.replace(random_weights(self.model, self.seed), router=ci_router(self.mesh, self.model))
            else:
                need_host = self.kind == "weka" and (
                    not self.device_layout or self.gather_check or int(os.environ.get("K3_BLOCK_CHECK", "0")) > 0
                )
                self.w, self.flat = weka_weights(
                    self.mesh, self.model, self.layer_idx, routed=need_host, cache=self.cache_path, hf_config=self.hf_config
                )
                if self.kind == "real_gate":  # (the random pool's experts, for the host check)
                    self.w = dataclasses.replace(self.w, routed=random_weights(self.model, self.seed).routed)
        return self.w

    def _moe_layer(self, chunk):
        """This layer's MoELayer (weka / real_gate): its routed experts laid out (real_gate: the random pool's, one set
        for every layer), host weights kept without them (59 GB of bf16 a weka layer otherwise)."""
        from models.demos.moe_block.moe_block import Geometry, MoEDtypes, MoELayer, RoutedExperts

        if self.layer is None:
            g = Geometry.of(self.mesh, self.model, chunk, LINKS, zigzag=False)
            w, dt = self._weights(), MoEDtypes()
            t0 = time.time()
            if self.kind == "real_gate":
                key = (id(self.mesh), chunk)
                hit = _POOL_EXPERTS.get(key)
                if hit is None or hit[0] is not self.mesh:
                    hit = _POOL_EXPERTS[key] = (self.mesh, RoutedExperts(self.mesh, g, self.model, w.routed, dt))
                experts = hit[1]
            else:  # (weka: the flat op laid out on the device from TtMoe's cache, or from the bf16 host experts)
                from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

                from models.demos.moe_block.moe_block import _FLAT_ACT

                m = self.model
                op = FlatRoutedExpert(
                    self.mesh,
                    "fake" if self.device_layout else self.flat,
                    m=RoutedExperts.cap_of(g),
                    H=m.routed_width,
                    I=m.routed_expert_hidden_dim,
                    gids=RoutedExperts.gids_of(g),
                    n_global=m.num_experts,
                    wdtype="bf4",
                    act=_FLAT_ACT[m.routed_expert_activation.kind],
                    pin=1,
                    layout=RoutedExperts.layout_of(g, m, True),
                )
                if self.device_layout:
                    self._gather(op, m, g)
                # (every layer's flat op shares one done-words buffer, as the GLM / DSv4.1 adapters do: the layers run one
                # at a time and each launch leaves it zeroed; one L1 buffer a layer piled up and moved the head's arena
                # at layer 10 - "the arena landed at 207296, not 209344: L1 allocated since")
                shared = _DONE.get(id(self.mesh))
                if shared is None or shared[0] is not self.mesh:
                    _DONE[id(self.mesh)] = (self.mesh, op.done)
                else:
                    ttnn.deallocate(op.done)
                    op.done = shared[1]
                experts = RoutedExperts(self.mesh, g, m, None, dt, op=op)
                if int(os.environ.get("K3_BLOCK_CHECK", "0")) == 0:
                    self.flat = None  # (laid out, nothing to check: the host experts go now)
            light = dataclasses.replace(w, routed=None)
            self.layer = MoELayer(self.mesh, g, self.model, lambda: light, experts, dt)
            logger.info(
                f"K3 {self.kind} layer {self.layer_idx}: MoELayer in {time.time() - t0:.0f} s; max RSS {_rss_gb():.0f} GB"
            )
        return self.layer

    def _gather(self, op, m, g):
        """Fill the flat op's (uninitialised) weight tensors on the device from TtMoe's cached bfp4 experts."""
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import WEIGHT_LAYOUT_KEYS
        from ttnn.bringup.flat_routed_expert_ttnn.flat_gather import flat_face_map, gather

        t0 = time.time()
        lay = dict(op.plan)
        E = g.experts_per_chip
        key = (m.routed_width, m.routed_expert_hidden_dim, E, repr([lay.get(k) for k in WEIGHT_LAYOUT_KEYS] + [lay.get("nu"), lay.get("hg")]))
        maps = _FACE_MAPS.get(key)
        if maps is None:
            maps = _FACE_MAPS[key] = flat_face_map(m.routed_width, m.routed_expert_hidden_dim, E, lay)
            logger.info(
                f"K3 weka gather: face maps in {time.time() - t0:.0f} s - "
                + ", ".join(f"{n} {t.shape[0]} pages ({int((t < 0).sum())} zero faces)" for n, t in maps.items())
            )
        t1 = time.time()
        srcs = weka_device_sources(self.mesh, self.layer_idx, E, cache=self.cache_path)
        t2 = time.time()
        for name, fm in maps.items():
            dst = getattr(op, name)
            pages = (int(dst.shape[-2]) // 32) * (int(dst.shape[-1]) // 32)
            assert pages == fm.shape[0], f"{name}: {pages} pages, the face map {fm.shape[0]}"
            gather(self.mesh, srcs, fm, dst)
        t3 = time.time()
        for t in srcs:
            ttnn.deallocate(t)
        logger.info(
            f"K3 weka layer {self.layer_idx}: device layout - {len(srcs)} tensorbins to DRAM {t2 - t1:.0f} s, gather "
            f"{t3 - t2:.1f} s; max RSS {_rss_gb():.0f} GB"
        )
        if self.gather_check:  # (the host path's tensors, value for value)
            from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

            from models.demos.moe_block.moe_block import _FLAT_ACT, RoutedExperts

            ref = FlatRoutedExpert(
                self.mesh,
                self.flat,
                m=RoutedExperts.cap_of(g),
                H=m.routed_width,
                I=m.routed_expert_hidden_dim,
                gids=RoutedExperts.gids_of(g),
                n_global=m.num_experts,
                wdtype="bf4",
                act=_FLAT_ACT[m.routed_expert_activation.kind],
                pin=1,
                layout=RoutedExperts.layout_of(g, m, True),
            )
            for name in maps:  # (chip by chip: the whole mesh in fp32 would be ~80 GB a tensor)
                bad, peak, shape = 0, 0.0, None
                for da, db in zip(ttnn.get_device_tensors(getattr(op, name)), ttnn.get_device_tensors(getattr(ref, name))):
                    a, b = ttnn.to_torch(da).float(), ttnn.to_torch(db).float()
                    shape = tuple(a.shape)
                    bad += int((a != b).sum())
                    peak = max(peak, float(b.abs().max()))
                    del a, b
                logger.info(
                    f"K3_GATHER_CHECK layer {self.layer_idx} {name}: {shape} a chip, {bad} values differ from the host "
                    f"layout over {self.mesh.get_num_devices()} chips (|host| max {peak:.3g})"
                )
                assert bad == 0, f"K3 gather: {name} differs from the host layout in {bad} values"
            for name in maps:
                ttnn.deallocate(getattr(ref, name))
            ttnn.deallocate(ref.done)  # (L1: left allocated it would move the head's arena - see _DONE)
            ttnn.deallocate(ref.gidx)

    def _block(self, chunk):
        """(the block, this call's MoELayer - None: random, the block's own weights - , the host weights to check)."""
        from models.demos.moe_block.moe_block import MoEBlock, WeightCache

        layer = None if self.kind in ("random", "ci") else self._moe_layer(chunk)
        key = (id(self.mesh), chunk)
        hit = _BLOCKS.get(key)
        if hit is None or hit[0] is not self.mesh:
            built = self._weights() if layer is None else layer
            logger.info(f"K3 MoE block: building for chunk {chunk} (first MoE layer {self.layer_idx}, {self.kind} weights)")
            self._l1_report("before the block")
            trace = _L1Trace() if os.environ.get("K3_BLOCK_L1_TRACE") == "1" else None
            try:
                from models.demos.moe_block.moe_block import MoEOptions

                opts = MoEOptions(head=os.environ.get("MOE_BLOCK_HEAD", "auto"))  # (MOE_BLOCK_HEAD=ops: the ops head)
                block = MoEBlock(
                    self.mesh, self.model, chunk, built, WeightCache(path=None, layer=0), options=opts, num_links=LINKS
                )
            finally:
                if trace is not None:
                    trace.close()
            # (the exchange's own flat op - never launched when every layer brings its MoELayer - shares the layers'
            # done words: its own was a second 2 KB of persistent L1 on that core, reserved on every core)
            shared = _DONE.get(id(self.mesh))
            xop = getattr(getattr(getattr(block, "exchange", None), "block", None), "op", None)
            if layer is not None and shared is not None and shared[0] is self.mesh and xop is not None:
                if getattr(xop, "done", None) is not None and xop.done is not shared[1]:
                    ttnn.deallocate(xop.done)
                    xop.done = shared[1]
            logger.info(f"K3 MoE block: {block.summary}")
            self._l1_report("after the block")
            if os.environ.get("K3_BLOCK_L1_LIST") == "1":
                self._l1_list(block, layer)
            hit = _BLOCKS[key] = (self.mesh, block, built)
        return hit[1], layer, self._weights()

    def __call__(self, x):
        R, C = tuple(self.mesh.shape)
        chunk = int(x.shape[-2]) * R
        block, layer, w = self._block(chunk)
        tmp = None
        if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
            x = tmp = ttnn.to_layout(ttnn.typecast(x, ttnn.bfloat16), ttnn.TILE_LAYOUT)
        check = self.calls < int(os.environ.get("K3_BLOCK_CHECK", "0"))
        if check:
            block.keep_taps = "device"
        y = block(x) if layer is None else block(x, layer=layer)
        if check:
            self._check(block, w, x, y)
            block.keep_taps = False
        if tmp is not None:
            ttnn.deallocate(tmp)
        self.calls += 1
        if self.kind == "weka" and self.calls >= int(os.environ.get("K3_BLOCK_CHECK", "0")) and self.flat is not None:
            self.flat = None  # (laid out and checked: the host experts go - ~59 GB of bf16 a layer)
        return y

    def _check(self, block, w, x, y):
        """y against the host model on the device's own routing."""
        g, m, shape = block.geometry, self.model, tuple(self.mesh.shape)
        composer = ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=shape, dims=(2, 3))
        T, H = g.chunk_tokens, m.emb_dim
        xs = ttnn.to_torch(x, mesh_composer=composer).reshape(T, H).float()
        ys = ttnn.to_torch(y, mesh_composer=composer).reshape(T, H).float()
        ids, wts = device_routing(block.taps()["rows"], g.tokens_per_chip, m.top_k)
        y_ref = host_moe(xs, w, m, ids, wts, flat=self.flat)
        p = torch.corrcoef(torch.stack([ys.flatten(), y_ref.flatten()]))[0, 1].item()
        finite = bool(torch.isfinite(ys).all())
        load = torch.bincount(ids.flatten(), minlength=m.num_experts).float()
        chip = load.reshape(-1, m.num_experts // (shape[0] * shape[1])).sum(1)
        logger.info(
            f"K3_BLOCK_CHECK layer {self.layer_idx} ({self.kind}): PCC {p:.5f} vs the host model (device routing); "
            f"x |max| {xs.abs().max():.3g}, y |max| {ys.abs().max():.3g}, finite {finite}; routing: chip max/mean "
            f"{chip.max() / chip.mean():.2f}, expert max/mean {load.max() / load.mean():.2f}, {int((load == 0).sum())} "
            f"experts empty"
        )
        assert finite and p > 0.98, f"K3 MoE block layer {self.layer_idx}: PCC {p:.5f}, finite {finite}"
