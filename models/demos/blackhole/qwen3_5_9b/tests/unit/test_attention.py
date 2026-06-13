# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Qwen3.5 TP full-attention prefill path vs the HF golden.

The pytest form of rob.py: build ONE HF Qwen3_5Attention layer with random weights, hand
its state_dict to the TT Qwen35Attention module, then PCC-check the TT prefill output AND
the post-prefill KV-cache contents against HF. Random weights are fine because the SAME
state_dict drives both sides — the test compares the attention math, not any checkpoint.
Each test runs on both a (1, 1) single device and the (1, 4) tensor-parallel mesh the model
targets, so the sharded q/k/v/o, the reduce-scatter output and the per-device KV cache are
all exercised.

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
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_prefill
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

# HF_MODEL is the single source of truth Qwen35ModelArgs parses for every dim. Set it before any
# args is built (setdefault, so an outer `export HF_MODEL=...` still wins). As the task asks this
# targets the 9B hub id — a bare id triggers a one-time snapshot_download of the config/weights.
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

# Prefill sequence buckets to sweep; MAX_SEQ sizes the KV cache so every bucket fits one args.
SEQ_LENS = [32, 512, 2048]
MAX_SEQ = 2048

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
