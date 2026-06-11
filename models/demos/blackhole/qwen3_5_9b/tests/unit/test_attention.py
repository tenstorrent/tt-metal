# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TP full-attention path vs the HF Qwen3_5Attention golden.

Ported from models/demos/gemma4/tests/unit/test_attention.py:
* prefill PCC across seq buckets, plus a post-prefill KV-cache content check
* decode PCC against a pre-filled deep KV cache (uniform and ragged per-user positions)
* partial-RoPE PCC at high decode positions and over a full prefill
The pre-existing TP tests only check decode at position 0, where RoPE is the
identity (cos=1, sin=0) and attention over one key reduces to V — so deep
positions, RoPE broadcasting and cache indexing were previously unexercised.

Run:
    HF_MODEL=/home/ttuser/models/Qwen3.5-27B-FP8 \
      pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_attention.py -v
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tests.test_attention_tp import _load_attn_layer
from models.demos.blackhole.qwen3_5_9b.tests.unit.reference import (
    causal_mask,
    hf_attention,
    hf_rope,
    model_path,
    text_config,
)
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import (
    apply_partial_rope_decode,
    apply_partial_rope_prefill,
    rot_mats_decode,
    rot_mats_prefill,
)
from models.demos.blackhole.qwen3_5_9b.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs

# Same mesh resolution as the sibling TP tests: explicit MESH_DEVICE, else all
# local devices (capped at the (1,4) TP the model targets).
MAX_SEQ = 2048
PREFILL_BUCKETS = [128, 512, 2048]
DECODE_CACHE_LENS = [32, 512, 1500]
ROPE_POSITIONS = [0, 32, 511, 1024, 2047]

# Reusable parametrization stack (mesh shape + 1D fabric for the TP CCLs).
parameterized_mesh_and_fabric = pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        ((1, 1), {}),
        ((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=True,
)


def _setup(mesh_device, max_batch_size, max_seq_len=MAX_SEQ):
    """Build (args, layer state_dict, TPAttention, hf text config) for the first
    full-attention layer. Loads exactly one layer's dequantized weights — the
    full-model dequant is a known host-OOM hazard."""
    mp = model_path()
    os.environ.setdefault("HF_MODEL", mp)  # Qwen35ModelArgs resolves the checkpoint from HF_MODEL
    args = Qwen35ModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
    layer_idx = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    sd = _load_attn_layer(mp, layer_idx)

    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None
    tw = load_attention_weights_tp(mesh_device, sd, args)
    attn = TPAttention(mesh_device, args, tw, tt_ccl)
    return args, sd, attn, text_config(mp)


def _replicate(t, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _gather_dim3(t, mesh_device):
    """Bring a (possibly hidden-dim-fractured) module output back to torch."""
    nd = mesh_device.get_num_devices()
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0)
    return ttnn.to_torch(t, mesh_composer=composer)[0, 0].float()


def _inject_kv(attn, args, mesh_device, k_all, v_all):
    """Overwrite TPAttention's per-local-head KV caches with host data.

    k_all/v_all: [B, NKV_global, max_seq, HD], already bf16-rounded and in the
    cache's post-RoPE convention. Device d's local head h holds global head
    d*NKV_local + h, so k_all[:, h::NKV_local] enumerates exactly the heads
    ShardTensorToMesh(dim=1) must scatter for cache-list slot h. Assumes
    tp <= n_kv_heads (no KV replication), which holds for the (1,4) target.
    """
    nkv_local = getattr(args, "n_local_kv_heads", args.n_kv_heads)

    def mk(t):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    attn.k_caches = [mk(k_all[:, h::nkv_local]) for h in range(nkv_local)]
    attn.v_caches = [mk(v_all[:, h::nkv_local]) for h in range(nkv_local)]


def _gather_kv(attn, args, mesh_device):
    """Read the per-local-head caches back as one [B, NKV_global, max_seq, HD]
    torch tensor (inverse of _inject_kv's device-major head layout)."""
    nkv_local = getattr(args, "n_local_kv_heads", args.n_kv_heads)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=1)
    ks = [ttnn.to_torch(c, mesh_composer=composer).float() for c in attn.k_caches]
    vs = [ttnn.to_torch(c, mesh_composer=composer).float() for c in attn.v_caches]
    # stack local-head slots then flatten (device, local_head) -> global head
    k = torch.stack(ks, dim=2)
    v = torch.stack(vs, dim=2)
    B, nd, _, S, HD = k.shape
    return k.reshape(B, nd * nkv_local, S, HD), v.reshape(B, nd * nkv_local, S, HD)


# ── Prefill PCC (+ KV-cache fill) ─────────────────────────────────────────


@torch.no_grad()
@parameterized_mesh_and_fabric
@pytest.mark.parametrize("seq_len", PREFILL_BUCKETS, ids=lambda s: f"seq{s}")
def test_attention_prefill_pcc(mesh_device, seq_len, reset_seeds, ensure_gc):
    """Causal prefill vs HF golden at several bucket lengths, including the
    S>=2048 SDPA chunk-size branch. Also checks the post-prefill KV-cache
    contents: decode continues from those tensors, so a wrong cache passes the
    output PCC but breaks generation later."""
    from transformers.cache_utils import DynamicCache

    args, sd, attn, cfg = _setup(mesh_device, max_batch_size=1)
    attn.reset_state()  # allocate caches so the prefill cache-fill path (stateful production path) runs

    x = torch.randn(1, seq_len, args.dim, dtype=torch.float32)

    # HF golden: eager attention needs the explicit causal mask; DynamicCache
    # captures the post-RoPE K/V that ttnn.fill_cache must reproduce.
    hf = hf_attention(cfg, sd)
    cache = DynamicCache()
    cos, sin = hf_rope(cfg)(x, torch.arange(seq_len).unsqueeze(0))
    ref, _ = hf(x, position_embeddings=(cos, sin), attention_mask=causal_mask(seq_len), past_key_values=cache)

    x_tt = _replicate(x.to(torch.bfloat16).reshape(1, 1, seq_len, args.dim), mesh_device)
    cos_tt, sin_tt = rot_mats_prefill(mesh_device, args.rope_head_dim, seq_len, args.rope_theta)
    out = attn.forward_prefill(x_tt, cos_tt, sin_tt)
    out_torch = _gather_dim3(out, mesh_device)  # [S, dim]

    passing, pcc = comp_pcc(ref[0], out_torch, 0.99)
    logger.info(f"ATTENTION PREFILL PCC (S={seq_len}) = {pcc}")

    k_tt, v_tt = _gather_kv(attn, args, mesh_device)
    ref_k = cache.layers[0].keys.float()[0]  # [NKV, S, HD], post-RoPE
    ref_v = cache.layers[0].values.float()[0]
    pass_k, pcc_k = comp_pcc(ref_k, k_tt[0, :, :seq_len], 0.999)
    pass_v, pcc_v = comp_pcc(ref_v, v_tt[0, :, :seq_len], 0.999)
    logger.info(f"ATTENTION PREFILL CACHE PCC (S={seq_len}): K={pcc_k} V={pcc_v}")

    assert passing, f"prefill output PCC too low (S={seq_len}): {pcc}"
    assert pass_k and pass_v, f"prefill KV-cache PCC too low (S={seq_len}): K={pcc_k} V={pcc_v}"


# ── Decode PCC with deep pre-filled KV cache ──────────────────────────────


@torch.no_grad()
@parameterized_mesh_and_fabric
@pytest.mark.parametrize("cache_len", DECODE_CACHE_LENS, ids=lambda c: f"cache{c}")
def test_attention_decode_deep_cache(mesh_device, cache_len, reset_seeds, ensure_gc):
    """One decode step for B=32 users against a pre-filled cache of cache_len
    entries — real multi-key attention at deep positions (the sibling
    test_attention_tp only covers position 0)."""
    from transformers.cache_utils import DynamicCache

    B = 32
    args, sd, attn, cfg = _setup(mesh_device, max_batch_size=B)
    NKV, HD = args.n_kv_heads, args.head_dim

    # Cache contents are bf16-rounded ONCE on host and fed to both TT and HF,
    # so the comparison isolates decode math from cache quantization noise.
    k_cache = torch.randn(B, NKV, cache_len, HD).to(torch.bfloat16)
    v_cache = torch.randn(B, NKV, cache_len, HD).to(torch.bfloat16)
    x = torch.randn(B, 1, args.dim, dtype=torch.float32)
    positions = torch.full((B,), cache_len, dtype=torch.int32)

    # HF golden: pre-load the cache, decode one token per user at cache_len.
    # No mask needed: a single query token validly attends to every cache slot.
    hf = hf_attention(cfg, sd)
    cache = DynamicCache()
    cache.update(k_cache.float(), v_cache.float(), 0)
    cos, sin = hf_rope(cfg)(x, positions.unsqueeze(1).long())
    ref, _ = hf(x, position_embeddings=(cos, sin), attention_mask=None, past_key_values=cache)
    ref = ref[:, 0]  # [B, dim]

    # TT: same cache zero-padded to the allocation length, then one decode step
    # writes the new K/V at `positions` and attends over [0, cache_len].
    k_all = torch.zeros(B, NKV, MAX_SEQ, HD, dtype=torch.bfloat16)
    v_all = torch.zeros(B, NKV, MAX_SEQ, HD, dtype=torch.bfloat16)
    k_all[:, :, :cache_len] = k_cache
    v_all[:, :, :cache_len] = v_cache
    _inject_kv(attn, args, mesh_device, k_all, v_all)

    x_tt = _replicate(x.to(torch.bfloat16).reshape(1, 1, B, args.dim), mesh_device)
    pos_tt = ttnn.from_torch(
        positions, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    cos_tt, sin_tt = rot_mats_decode(mesh_device, args.rope_head_dim, MAX_SEQ, args.rope_theta, positions)
    out = attn.forward_decode(x_tt, pos_tt, cos_tt, sin_tt)
    out_torch = _gather_dim3(out, mesh_device)  # [B, dim]

    passing, pcc = comp_pcc(ref, out_torch, 0.99)
    logger.info(f"ATTENTION DECODE DEEP-CACHE PCC (cache_len={cache_len}) = {pcc}")
    assert passing, f"decode PCC too low at cache_len={cache_len}: {pcc}"


@torch.no_grad()
@parameterized_mesh_and_fabric
def test_attention_decode_ragged_positions(mesh_device, reset_seeds, ensure_gc):
    """Every user at a different cache depth (vLLM continuous batching):
    exercises paged_update_cache's per-user update_idxs and SDPA decode's
    cur_pos_tensor, which the uniform-position tests cannot distinguish from
    a single shared position."""
    from transformers.cache_utils import DynamicCache

    B = 32
    args, sd, attn, cfg = _setup(mesh_device, max_batch_size=B)
    NKV, HD = args.n_kv_heads, args.head_dim

    gen = torch.Generator().manual_seed(1234)
    positions = torch.randint(1, MAX_SEQ - 1, (B,), generator=gen, dtype=torch.int32)
    positions[0] = 0  # boundary: empty cache (pure self-attention)
    positions[1] = MAX_SEQ - 1  # boundary: last cache slot

    k_all = torch.zeros(B, NKV, MAX_SEQ, HD, dtype=torch.bfloat16)
    v_all = torch.zeros(B, NKV, MAX_SEQ, HD, dtype=torch.bfloat16)
    for u in range(B):
        k_all[u, :, : positions[u]] = torch.randn(NKV, int(positions[u]), HD, generator=gen).to(torch.bfloat16)
        v_all[u, :, : positions[u]] = torch.randn(NKV, int(positions[u]), HD, generator=gen).to(torch.bfloat16)
    x = torch.randn(B, 1, args.dim, dtype=torch.float32, generator=gen)

    # HF golden, per user: DynamicCache requires a uniform length per batch,
    # so each user gets its own B=1 forward at its own position.
    hf = hf_attention(cfg, sd)
    rope = hf_rope(cfg)
    refs = []
    for u in range(B):
        cache = DynamicCache()
        if positions[u] > 0:
            cache.update(k_all[u : u + 1, :, : positions[u]].float(), v_all[u : u + 1, :, : positions[u]].float(), 0)
        cos, sin = rope(x[u : u + 1], positions[u : u + 1].unsqueeze(1).long())
        ref_u, _ = hf(x[u : u + 1], position_embeddings=(cos, sin), attention_mask=None, past_key_values=cache)
        refs.append(ref_u[:, 0])
    ref = torch.cat(refs)  # [B, dim]

    _inject_kv(attn, args, mesh_device, k_all, v_all)
    x_tt = _replicate(x.to(torch.bfloat16).reshape(1, 1, B, args.dim), mesh_device)
    pos_tt = ttnn.from_torch(
        positions, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    cos_tt, sin_tt = rot_mats_decode(mesh_device, args.rope_head_dim, MAX_SEQ, args.rope_theta, positions)
    out = attn.forward_decode(x_tt, pos_tt, cos_tt, sin_tt)
    out_torch = _gather_dim3(out, mesh_device)  # [B, dim]

    passing, pcc = comp_pcc(ref, out_torch, 0.99)
    logger.info(f"ATTENTION DECODE RAGGED-POSITIONS PCC = {pcc}")
    assert passing, f"ragged-position decode PCC too low: {pcc}"


# ── Partial-RoPE PCC at decode positions / over prefill ──────────────────


def _rope_dims(cfg):
    """(rope_dim, theta) from the HF config — partial_rotary_factor lives in
    rope_parameters for this model family."""
    rope_dim = int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"])
    return rope_dim, cfg.rope_parameters["rope_theta"]


@torch.no_grad()
@parameterized_mesh_and_fabric
@pytest.mark.parametrize("position", ROPE_POSITIONS, ids=lambda p: f"pos{p}")
def test_rope_decode_position(mesh_device, position, reset_seeds, ensure_gc):
    """rot_mats_decode tables + apply_partial_rope_decode vs HF apply_rotary_pos_emb
    at deep positions. Catches table indexing errors and the [1,B,1,rope_dim] x
    [1,B,NH,rope_dim] broadcast — both invisible at position 0 where RoPE is the
    identity. No checkpoint weights needed (config only)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    cfg = text_config()
    rope_dim, theta = _rope_dims(cfg)
    NH, HD = cfg.num_attention_heads, cfg.head_dim

    q = torch.randn(1, NH, 1, HD).to(torch.bfloat16).float()  # bf16-rounded once for both sides
    cos, sin = hf_rope(cfg)(q, torch.tensor([[position]]))
    ref, _ = apply_rotary_pos_emb(q, q, cos, sin)  # [1, NH, 1, HD]

    # TT decode layout is [1, B, NH, HD] (users on dim 1) — transpose the HF layout.
    q_tt = _replicate(q.transpose(1, 2).to(torch.bfloat16), mesh_device)
    cos_tt, sin_tt = rot_mats_decode(mesh_device, rope_dim, MAX_SEQ, theta, torch.tensor([position]))
    out = apply_partial_rope_decode(q_tt, cos_tt, sin_tt, NH, 1, rope_dim)
    out_torch = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()  # replicated: device 0 suffices

    # Table values feeding the kernel must match HF cos/sin to bf16 precision;
    # PCC alone can hide a small constant frequency offset.
    cos_tab = ttnn.to_torch(ttnn.get_device_tensors(cos_tt)[0]).float()[0, 0, 0]
    sin_tab = ttnn.to_torch(ttnn.get_device_tensors(sin_tt)[0]).float()[0, 0, 0]
    cos_err = (cos_tab - cos[0, 0].float()).abs().max().item()
    sin_err = (sin_tab - sin[0, 0].float()).abs().max().item()

    passing, pcc = comp_pcc(ref.transpose(1, 2), out_torch, 0.999)
    logger.info(f"ROPE DECODE PCC (pos={position}) = {pcc} (cos_err={cos_err:.5f}, sin_err={sin_err:.5f})")
    assert cos_err < 0.02 and sin_err < 0.02, f"rope table mismatch at pos={position}: cos={cos_err} sin={sin_err}"
    assert passing, f"rope decode PCC too low at pos={position}: {pcc}"


@torch.no_grad()
@parameterized_mesh_and_fabric
def test_rope_prefill_full_range(mesh_device, reset_seeds, ensure_gc):
    """apply_partial_rope_prefill over all positions 0..MAX_SEQ-1 at once vs HF.
    Uses the per-device production head count (n_heads / 4) to keep it light."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    cfg = text_config()
    rope_dim, theta = _rope_dims(cfg)
    NH, HD = cfg.num_attention_heads // 4, cfg.head_dim
    S = MAX_SEQ

    q = torch.randn(1, NH, S, HD).to(torch.bfloat16).float()
    cos, sin = hf_rope(cfg)(q, torch.arange(S).unsqueeze(0))
    ref, _ = apply_rotary_pos_emb(q, q, cos, sin)  # [1, NH, S, HD]

    q_tt = _replicate(q.to(torch.bfloat16), mesh_device)  # prefill layout matches HF: [1, NH, S, HD]
    cos_tt, sin_tt = rot_mats_prefill(mesh_device, rope_dim, S, theta)
    out = apply_partial_rope_prefill(q_tt, cos_tt, sin_tt, NH, rope_dim)
    out_torch = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()

    passing, pcc = comp_pcc(ref, out_torch, 0.999)
    logger.info(f"ROPE PREFILL PCC (S={S}) = {pcc}")
    assert passing, f"rope prefill PCC too low: {pcc}"
