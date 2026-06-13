# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Qwen3.5 TP full-attention prefill and decode paths vs the HF golden.

The pytest form of rob.py: build ONE HF Qwen3_5Attention layer with random weights, hand
its state_dict to the TT Qwen35Attention module, then PCC-check the TT prefill output AND
the post-prefill KV-cache contents against HF. Random weights are fine because the SAME
state_dict drives both sides — the test compares the attention math, not any checkpoint.
Each test runs on both a (1, 1) single device and the (1, 4) tensor-parallel mesh the model
targets, so the sharded q/k/v/o, the reduce-scatter output and the per-device KV cache are
all exercised.

The decode test (test_attention_decode) deliberately does NOT run prefill: it injects the
SAME random post-RoPE K/V into both the HF DynamicCache and the TT internal cache, then
decodes N_DECODE tokens back-to-back off those caches. Each step's output is PCC-checked, and
the N_DECODE freshly written cache slots are compared against HF afterwards — so the per-step
ttnn cache update (paged_update_cache at cur_pos) is verified, not just the decode math
(decode RoPE table, SDPA-decode, GQA). All of it stays isolated from the prefill path.

Run:
    pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_attention.py -v
"""
import os

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tt.attention import Qwen35Attention
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_decode, rot_mats_prefill
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

# HF_MODEL is the single source of truth Qwen35ModelArgs parses for every dim. Set it before any
# args is built (setdefault, so an outer `export HF_MODEL=...` still wins). As the task asks this
# targets the 9B hub id — a bare id triggers a one-time snapshot_download of the config/weights.
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

# Prefill sequence buckets to sweep; MAX_SEQ sizes the KV cache so every bucket fits one args.
SEQ_LENS = [32, 512, 2048]
MAX_SEQ = 2048

# Number of tokens test_attention_decode generates back-to-back. >1 so the per-step KV-cache
# writes (one slot per step) get exercised and checked, not just a single isolated update.
N_DECODE = 5

# Cached-context lengths to sweep for decode: step i writes index ctx_len + i, so the last write
# lands at ctx_len + N_DECODE - 1, which must stay < MAX_SEQ (the cache's last writable slot).
# MAX_SEQ - N_DECODE fills the cache right up to its final slot.
DECODE_CTX_LENS = [32, 512, MAX_SEQ - N_DECODE]

# Decorator both tests share: run on a (1, 1) single device and the (1, 4) TP mesh, the latter
# with the 1D fabric the attention CCLs (tt_all_reduce / reduce-scatter) require. Indirect so the
# tuple feeds the framework `mesh_device` fixture and the dict feeds `device_params`.
parameterized_mesh_and_fabric = pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        ((1, 1), {}),
        ((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=True,
)


# ── HF reference helpers (inlined: tests/unit/reference.py was removed on this branch) ──
def hf_rope(cfg):
    """HF rotary table. For text-only position_ids the interleaved M-RoPE collapses to the
    standard partial RoPE rope_tp builds, so HF cos/sin and the TT tables are directly comparable."""
    return Qwen3_5TextRotaryEmbedding(cfg)


def causal_mask(seq_len):
    """[1, 1, S, S] additive causal mask (upper triangle = -inf) the eager HF attention honours."""
    return torch.full((1, 1, seq_len, seq_len), float("-inf")).triu(1)


# ── Fixtures ───────────────────────────────────────────────────────────────────────
@pytest.fixture
def args(mesh_device):
    """Qwen35ModelArgs for the active mesh, shared across this module's tests. max_seq_len is
    pinned to MAX_SEQ so one args (and its cache size) serves every SEQ_LENS bucket."""
    return Qwen35ModelArgs(mesh_device, max_batch_size=1, max_seq_len=MAX_SEQ)


@pytest.fixture
def reference_attention(args):
    """One HF Qwen3_5Attention layer (random weights, float32, eager) built from the SAME parsed
    config the TT side uses so q/k/v/o dims line up. Returns (hf_attn, state_dict, cfg); the RAW
    state_dict is what the TT loader consumes (HF adds the q/k-norm +1 internally, the loader bakes
    it in)."""
    cfg = args.hf_config.get_text_config()
    cfg._attn_implementation = "eager"
    hf_attn = Qwen3_5Attention(config=cfg, layer_idx=0).to(torch.float32).eval()
    return hf_attn, hf_attn.state_dict(), cfg


# ── Helpers to bring (possibly sharded) TT tensors back to torch ─────────────────────
def _gather_output(tt_out, mesh_device):
    """forward_prefill reduce-scatters its [1, 1, S, dim] output along the hidden dim, so concat
    dim=3 on a multi-device mesh reassembles it (dim=0 is the harmless single-device fallback)."""
    nd = mesh_device.get_num_devices()
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0)
    return ttnn.to_torch(tt_out, mesh_composer=composer)[0, 0].float()  # [S, dim]


def _gather_kv_cache(kv_cache, mesh_device):
    """Read the internal [k_cache, v_cache] back as torch [B, n_kv_heads, max_seq, HD]. Each TP
    device holds its local KV heads, so concat dim=1 stitches the head axis back together (dim=0
    is the single-device no-op). Assumes no KV replication (tp <= n_kv_heads), true for (1, 4)."""
    k_cache, v_cache = kv_cache
    nd = mesh_device.get_num_devices()
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=1 if nd > 1 else 0)
    k = ttnn.to_torch(k_cache, mesh_composer=composer).float()
    v = ttnn.to_torch(v_cache, mesh_composer=composer).float()
    return k, v


def _inject_kv_cache(attn, k_full, v_full, mesh_device):
    """Overwrite the TT module's KV cache with given [B, n_kv_heads, max_seq, HD] torch tensors.

    The exact inverse of _gather_kv_cache: shard the head axis (dim=1) across the TP devices so
    device d holds the SAME local KV heads its column-parallel wk/wv produce at decode time (no KV
    replication, tp <= n_kv_heads — true for (1, 4)). Reassigning attn.kv_cache mirrors how the
    vLLM path swaps in an external cache via set_paged_kv_cache, so forward_decode reads it as-is."""
    nd = mesh_device.get_num_devices()

    def _to_cache(t):
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1) if nd > 1 else ttnn.ReplicateTensorToMesh(mesh_device)
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )

    attn.kv_cache = [_to_cache(k_full), _to_cache(v_full)]


def _replicate(t, mesh_device):
    """Host tensor -> bf16 DRAM tensor replicated to every device (the prefill input layout)."""
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


# ── Test ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@parameterized_mesh_and_fabric
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"seq{s}")
def test_attention_prefill(mesh_device, seq_len, args, reference_attention, reset_seeds, ensure_gc):
    """Causal prefill PCC vs the HF golden across seq buckets, plus the post-prefill KV-cache
    content check. The cache check matters because decode continues from exactly those tensors:
    a wrong cache still passes the output PCC but corrupts generation one step later."""
    hf_attn, state_dict, cfg = reference_attention

    # One shared dummy activation [1, S, dim], float32 — the single input both sides consume.
    x = torch.randn(1, seq_len, args.dim, dtype=torch.float32)

    # HF golden. DynamicCache captures the post-RoPE / post-q-k-norm K/V (positions 0..S-1) that
    # the TT fill_cache must reproduce; eager attention consumes the explicit causal mask.
    rope = hf_rope(cfg)
    cache = DynamicCache()
    cos, sin = rope(x, torch.arange(seq_len).unsqueeze(0))
    ref, _ = hf_attn(x, position_embeddings=(cos, sin), attention_mask=causal_mask(seq_len), past_key_values=cache)

    # TT side: create_kv_cache=True so forward_prefill writes the prompt's K/V into the internal
    # caches compared below; the reference weights load through the state_dict= keyword.
    attn = Qwen35Attention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        args=args,
        tt_ccl=TT_CCL(mesh_device),
        create_kv_cache=True,
    )

    # Prefill input is x replicated as [1, 1, S, dim]; cos/sin tables cover positions 0..S-1.
    x_tt = _replicate(x.reshape(1, 1, seq_len, args.dim), mesh_device)
    cos_tt, sin_tt = rot_mats_prefill(mesh_device, args.rope_head_dim, seq_len, args.rope_theta)
    out = attn.forward_prefill(x_tt, cos_tt, sin_tt)
    out_torch = _gather_output(out, mesh_device)  # [S, dim]

    passing, pcc = comp_pcc(ref[0], out_torch, 0.99)
    logger.info(f"ATTENTION PREFILL output PCC (mesh={tuple(mesh_device.shape)}, S={seq_len}) = {pcc}")

    # KV-cache content check: gather the TT caches and compare the filled [0, :, :S] window to HF's
    # captured K/V. A tighter 0.999 threshold here — the cache is a direct copy, not a matmul chain.
    k_tt, v_tt = _gather_kv_cache(attn.kv_cache, mesh_device)
    ref_k = cache.layers[0].keys[0].float()  # [n_kv, S, HD], post-RoPE
    ref_v = cache.layers[0].values[0].float()
    pass_k, pcc_k = comp_pcc(ref_k, k_tt[0, :, :seq_len], 0.999)
    pass_v, pcc_v = comp_pcc(ref_v, v_tt[0, :, :seq_len], 0.999)
    logger.info(f"ATTENTION PREFILL KV-cache PCC (S={seq_len}): K={pcc_k} V={pcc_v}")

    assert passing, f"prefill output PCC too low (S={seq_len}): {pcc}"
    assert pass_k and pass_v, f"prefill KV-cache PCC too low (S={seq_len}): K={pcc_k} V={pcc_v}"


@torch.no_grad()
@parameterized_mesh_and_fabric
@pytest.mark.parametrize("ctx_len", DECODE_CTX_LENS, ids=lambda s: f"ctx{s}")
def test_attention_decode(mesh_device, ctx_len, args, reference_attention, reset_seeds, ensure_gc):
    """Multi-token decode PCC vs the HF golden, reading a cache pre-filled with random K/V.

    No prefill here (unlike rob.py): the SAME random post-RoPE K/V is injected into both the HF
    DynamicCache and the TT internal cache, then N_DECODE tokens are decoded back-to-back starting
    at position == ctx_len. Each step's output is PCC-checked, and after the loop the N_DECODE slots
    the decode wrote are compared against HF's appended K/V. Because step i reads back the slot step
    i-1 wrote, a wrong per-step cache update (the paged_update_cache at cur_pos) shows up as drift in
    the later steps' output AND as a direct cache mismatch — independent of the prefill path."""
    hf_attn, state_dict, cfg = reference_attention
    B, n_kv, HD = args.max_batch_size, args.n_kv_heads, args.head_dim

    # Random post-RoPE K/V for the ctx_len cached positions, rounded to bf16 so HF (float32) and TT
    # (bf16) hold byte-identical cache contents — this isolates the decode math from cache rounding.
    # The full [B, n_kv, MAX_SEQ, HD] tensor is zero past ctx_len: only 0..ctx_len-1 start filled,
    # the decode steps write indices ctx_len..ctx_len+N_DECODE-1, and each step attends 0..its pos.
    k_full = torch.zeros(B, n_kv, MAX_SEQ, HD)
    v_full = torch.zeros(B, n_kv, MAX_SEQ, HD)
    k_full[:, :, :ctx_len] = torch.randn(B, n_kv, ctx_len, HD).to(torch.bfloat16).float()
    v_full[:, :, :ctx_len] = torch.randn(B, n_kv, ctx_len, HD).to(torch.bfloat16).float()

    # TT side: allocate the caches, then overwrite them with the injected random K/V (sharded by head).
    attn = Qwen35Attention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        args=args,
        tt_ccl=TT_CCL(mesh_device),
        create_kv_cache=True,
    )
    _inject_kv_cache(attn, k_full, v_full, mesh_device)

    # HF side: pre-fill the DynamicCache with the same ctx_len window. Each decode step appends the
    # new token's K/V (DynamicLayer.update concats on the seq axis), so after step i HF holds
    # ctx_len + i + 1 positions and the query at ctx_len + i attends over all of them.
    cache = DynamicCache()
    cache.update(k_full[:, :, :ctx_len], v_full[:, :, :ctx_len], 0)
    rope = hf_rope(cfg)

    # Decode N_DECODE tokens back-to-back, advancing the position each step. The TT cache persists
    # across forward_decode calls, so step i reads back the slot step i-1 wrote — exactly the
    # production decode loop. No causal mask: each step's lone query validly attends to every cached
    # key plus itself (cf. rob.py decode).
    for i in range(N_DECODE):
        pos = ctx_len + i
        x = torch.randn(B, 1, args.dim, dtype=torch.float32)  # fresh activation per step

        # HF golden for this step: appends to the cache and returns the new token's output.
        cos, sin = rope(x, torch.full((B, 1), pos, dtype=torch.long))
        ref, _ = hf_attn(x, position_embeddings=(cos, sin), attention_mask=None, past_key_values=cache)
        ref = ref[:, 0]  # [B, dim]

        # TT decode layout: token replicated as [1, 1, B, dim]; per-user position; decode RoPE tables
        # rebuilt for this step's position. forward_decode writes the new K/V into the cache at
        # cur_pos, then SDPA-decode reads straight off it.
        x_tt = _replicate(x.reshape(1, 1, B, args.dim), mesh_device)
        positions = torch.full((B,), pos, dtype=torch.int32)
        pos_tt = ttnn.from_torch(
            positions, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
        )
        cos_tt, sin_tt = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, positions)
        out = attn.forward_decode(x_tt, pos_tt, cos_tt, sin_tt)
        out_torch = _gather_output(out, mesh_device)  # forward_decode output is [1, 1, B, dim] -> [B, dim]

        passing, pcc = comp_pcc(ref, out_torch, 0.99)
        logger.info(
            f"ATTENTION DECODE output PCC (mesh={tuple(mesh_device.shape)}, ctx={ctx_len}, step={i}, pos={pos}) = {pcc}"
        )
        assert passing, f"decode output PCC too low (ctx={ctx_len}, step={i}, pos={pos}): {pcc}"

    # KV-cache content check: the N_DECODE slots the decode just wrote must match HF's appended K/V.
    # Compare ONLY the [ctx_len : ctx_len+N_DECODE] window — the injected prefix is byte-identical and
    # would swamp the PCC, masking a bad decode write. The tight 0.999 threshold mirrors the prefill
    # cache check: each slot is an independent wk/wv matmul + norm + RoPE of its own activation (no
    # cross-step accumulation in the stored value), so it should be as clean as prefill's fill_cache.
    k_tt, v_tt = _gather_kv_cache(attn.kv_cache, mesh_device)
    ref_k = cache.layers[0].keys[0].float()  # [n_kv, ctx_len+N_DECODE, HD], post-RoPE
    ref_v = cache.layers[0].values[0].float()
    end = ctx_len + N_DECODE
    pass_k, pcc_k = comp_pcc(ref_k[:, ctx_len:end], k_tt[0, :, ctx_len:end], 0.999)
    pass_v, pcc_v = comp_pcc(ref_v[:, ctx_len:end], v_tt[0, :, ctx_len:end], 0.999)
    logger.info(f"ATTENTION DECODE KV-cache PCC (ctx={ctx_len}, wrote slots {ctx_len}..{end - 1}): K={pcc_k} V={pcc_v}")
    assert pass_k and pass_v, f"decode KV-cache PCC too low (ctx={ctx_len}): K={pcc_k} V={pcc_v}"
