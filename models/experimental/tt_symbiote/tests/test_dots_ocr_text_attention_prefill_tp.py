# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-mesh TP4 (Wormhole, 4x n150) test for the dots.ocr TEXT-decoder prefill attention.

The text decoder is GQA (12 Q / 2 KV heads, head_dim 128). Its TP4 prefill op sequence,
matching ``TTNNDotsOCRAttention._forward_prefill_tp`` in dots_ocr_attention.py, at S=2816:

    QKV matmul   : COLUMN(N)-split by head. Each device owns 3 Q heads + the 1 KV head those
                   Q heads attend to (GQA group = 12/2 = 6; devices 0,1 -> KV0, devices 2,3 ->
                   KV1). Per-device fused slab is [Q_d(384) | K_d(128) | V_d(128)] = 640.
                   2816 x 1536 x 2048  ->  2816 x 1536 x 640 per device.
    create_heads : (num_heads=3, num_kv_heads=1) -> Q[1,3,S,128], K/V[1,1,S,128] per device.
    rotary(Q) / rotary(K) : 2D-RoPE-style half_half rotary (dtype-preserving).
    SDPA         : causal GQA (3 Q : 1 KV) per device.
    concat_heads : -> [1,1,S,384] per device (head-sharded).
    all_gather   : regather the 4 head shards -> full [1,1,S,1536] (o_proj input).
    o_proj matmul: COLUMN(N)-split (replicated input, col-sharded weight) -> [M, 384] per device.
                   2816 x 1536 x 1536  ->  2816 x 1536 x 384 per device. Output stays N-sharded
                   (the K shard for the next layer) -- no all-reduce after o_proj.

This is an op-level test: it reproduces the sharding inline and reads q/k/v/o weights off the
HF module, so it does NOT import the model package. The 2D-RoPE cos/sin are built in torch and
fed to both the device rotary and the float reference (self-consistent). The paged KV-cache
fill (``PagedFillCacheDeviceOperation`` in the perf table) is a cache side-write that does not
affect the attention output and needs a paged-cache object, so it is omitted here. PCC is
checked against a float 12-head GQA causal reference. Matmuls use the tuned 2D-mcast config.

Run (correctness):  pytest .../test_dots_ocr_text_attention_prefill_tp.py -s
Run (Tracy):        python -m tracy -v -r -p -m pytest .../test_dots_ocr_text_attention_prefill_tp.py
"""
from __future__ import annotations

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc
from models.experimental.tt_symbiote.tests._vision_tp_matmul import vision_matmul_program_config

HIDDEN_SIZE = 1536
NUM_Q = 12
NUM_KV = 2
HEAD_DIM = 128
ROPE_THETA = 1000000.0
DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"


def _resolve_model_path():
    import os

    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


def _text_attn_weights():
    """dots.ocr text-decoder layer-0 attention: q/k/v (with bias) + o_proj (no bias)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(_resolve_model_path(), trust_remote_code=True)
    cfg.num_hidden_layers = 1
    hf = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    a = hf.model.layers[0].self_attn

    def wb(lin):
        return lin.weight.data.clone(), (lin.bias.data.clone() if lin.bias is not None else None)

    qkv = (wb(a.q_proj), wb(a.k_proj), wb(a.v_proj))
    o = wb(a.o_proj)
    del hf
    return qkv, o


def _rope_cos_sin(s, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """1D half_half RoPE cos/sin ``[1,1,S,head_dim]`` (dtype-preserving rotary convention)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))  # [hd/2]
    freqs = torch.arange(s, dtype=torch.float32).unsqueeze(1) * inv_freq.unsqueeze(0)  # [S, hd/2]
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos.to(torch.bfloat16), sin.to(torch.bfloat16)


def _torch_reference(x, qkv_w, o_w, cos, sin):
    """Float 12-head GQA causal attention + o_proj on the same weights/RoPE."""
    (wq, bq), (wk, bk), (wv, bv) = qkv_w
    wo, bo = o_w
    s = int(x.shape[0])
    xf = x.float()
    group = NUM_Q // NUM_KV

    def proj(w, b):
        y = xf @ w.t().float()
        return y + b.float() if b is not None else y

    q = proj(wq, bq).reshape(s, NUM_Q, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)  # [1,12,S,hd]
    k = proj(wk, bk).reshape(s, NUM_KV, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)  # [1,2,S,hd]
    v = proj(wv, bv).reshape(s, NUM_KV, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)

    cos_f, sin_f = cos.float(), sin.float()

    def rope(t):
        x1, x2 = t[..., : HEAD_DIM // 2], t[..., HEAD_DIM // 2 :]
        return t * cos_f + torch.cat((-x2, x1), dim=-1) * sin_f

    q, k = rope(q), rope(k)
    k = k.repeat_interleave(group, dim=1)  # GQA expand -> [1,12,S,hd]
    v = v.repeat_interleave(group, dim=1)

    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)  # [1,12,S,S]
    causal = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    ctx = torch.softmax(scores + causal, dim=-1) @ v  # [1,12,S,hd]
    ctx = ctx.squeeze(0).permute(1, 0, 2).reshape(s, HIDDEN_SIZE)  # concat heads
    return ctx @ wo.t().float() + (bo.float() if bo is not None else 0.0)  # [S, 1536]


def _paged_fill_kv(mesh_device, k, v, s, replicate):
    """Fill K/V into a paged cache (the model's TP path: gather to full 2 KV heads, then fill).

    Per device holds 1 KV head; devices 0 and 2 own the distinct heads (KV0, KV1). Gather
    those, rebuild full [1, 2, S, 128] BF16 replicated, and write into a paged cache with
    ttnn.experimental.paged_fill_cache (one op per K and V).
    """
    tp = int(mesh_device.get_num_devices())
    src_devs = [0, 2]  # TP4 GQA: devices owning the distinct KV heads (kv_head = (d*3)//6)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    k_cat = ttnn.to_torch(k, mesh_composer=composer)  # [tp, 1, S, 128]
    v_cat = ttnn.to_torch(v, mesh_composer=composer)
    k_full = torch.cat([k_cat[d] for d in src_devs], dim=0).unsqueeze(0)  # [1, 2, S, 128]
    v_full = torch.cat([v_cat[d] for d in src_devs], dim=0).unsqueeze(0)
    k_fill = ttnn.from_torch(
        k_full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    v_fill = ttnn.from_torch(
        v_full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )

    block_size = 64
    max_num_blocks = (s + block_size - 1) // block_size  # ceil(S / block_size) = 44 at S=2816
    cache_shape = (max_num_blocks, NUM_KV, block_size, HEAD_DIM)
    k_cache = ttnn.zeros(
        cache_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    v_cache = ttnn.zeros(
        cache_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    page_table = ttnn.from_torch(
        torch.arange(max_num_blocks, dtype=torch.int32).reshape(1, max_num_blocks),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    ttnn.experimental.paged_fill_cache(k_cache, k_fill, page_table, batch_idx=0)
    ttnn.experimental.paged_fill_cache(v_cache, v_fill, page_table, batch_idx=0)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
def test_dots_ocr_text_attention_prefill_tp4(mesh_device):
    """TP4 text-decoder prefill attention: N-split QKV (3Q+1KV/dev), causal GQA SDPA, N-split o_proj."""
    tp = int(mesh_device.get_num_devices())
    if tp < 4:
        pytest.skip("text-attention prefill TP4 requires 4 devices")
    tp = 4
    if tuple(mesh_device.shape) != (1, 4):
        mesh_device.reshape(ttnn.MeshShape(1, 4))

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    mem = ttnn.DRAM_MEMORY_CONFIG
    l1 = ttnn.L1_MEMORY_CONFIG
    q_pd = NUM_Q // tp  # 3
    group = NUM_Q // NUM_KV  # 6
    s = 2816

    qkv_w, o_w = _text_attn_weights()
    (wq, bq), (wk, bk), (wv, bv) = qkv_w
    wo, bo = o_w
    has_bias = bq is not None

    x_torch = torch.randn(s, HIDDEN_SIZE, dtype=torch.bfloat16)
    cos_full, sin_full = _rope_cos_sin(s)
    ref = _torch_reference(x_torch, qkv_w, o_w, cos_full, sin_full)

    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True
    )
    grid = mesh_device.compute_with_storage_grid_size()
    sdpa_pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=256,
        k_chunk_size=256,
        exp_approx_mode=False,
    )
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    shard0 = ttnn.ShardTensorToMesh(mesh_device, dim=0)

    def up(t, dtype, mapper):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=mem, mesh_mapper=mapper
        )

    x_tt = up(x_torch.reshape(1, 1, s, HIDDEN_SIZE), ttnn.bfloat8_b, replicate)

    # Precomputed RoPE cache [1,1,MAX_SEQ,hd]; sliced to S below (mirrors
    # BailingRotarySetup.get_cos_sin_for_prefill -> the two ~11us SliceDeviceOperations).
    MAX_SEQ = 4096
    cos_big, sin_big = _rope_cos_sin(MAX_SEQ)
    cos_cache = up(cos_big, ttnn.bfloat8_b, replicate)
    sin_cache = up(sin_big, ttnn.bfloat8_b, replicate)

    # --- QKV: column(N)-split by head. Per-device slab [Q_d(384) | K_d(128) | V_d(128)]. ---
    slabs_w, slabs_b = [], ([] if has_bias else None)
    for d in range(tp):
        kv_head = (d * q_pd) // group  # GQA: which KV head this device's Q heads attend to
        q_sl = wq[d * q_pd * HEAD_DIM : (d * q_pd + q_pd) * HEAD_DIM]
        k_sl = wk[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
        v_sl = wv[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
        slabs_w.append(torch.cat([q_sl, k_sl, v_sl], dim=0))  # [640, 1536]
        if has_bias:
            qb = bq[d * q_pd * HEAD_DIM : (d * q_pd + q_pd) * HEAD_DIM]
            kb = bk[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
            vb = bv[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
            slabs_b.append(torch.cat([qb, kb, vb], dim=0))
    qkv_w_tt = up(torch.stack([w.t().contiguous() for w in slabs_w], dim=0), ttnn.bfloat8_b, shard0)  # [tp,1536,640]
    qkv_b_tt = (
        up(torch.stack([b.reshape(1, -1).contiguous() for b in slabs_b], dim=0), ttnn.bfloat8_b, shard0)
        if has_bias
        else None
    )

    qkv_pc = vision_matmul_program_config(mesh_device, s, HIDDEN_SIZE, (q_pd + 2) * HEAD_DIM)  # M x 1536 x 640
    qkv = ttnn.linear(
        x_tt,
        qkv_w_tt,
        bias=qkv_b_tt,
        dtype=ttnn.bfloat8_b,
        memory_config=mem,
        compute_kernel_config=ckc,
        program_config=qkv_pc,
    )

    # --- per-device heads -> RoPE -> causal GQA SDPA -> concat ---
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=q_pd, num_kv_heads=1, transpose_k_heads=False, memory_config=l1
    )
    ttnn.deallocate(qkv)

    # Slice the RoPE cache down to S (2 SliceDeviceOperations), then rotary.
    cos_tt = ttnn.slice(cos_cache, [0, 0, 0, 0], [1, 1, s, HEAD_DIM])
    sin_tt = ttnn.slice(sin_cache, [0, 0, 0, 0], [1, 1, s, HEAD_DIM])
    q = ttnn.experimental.rotary_embedding(q, cos_tt, sin_tt, memory_config=l1)
    k = ttnn.experimental.rotary_embedding(k, cos_tt, sin_tt, memory_config=l1)

    # Guarded post-RoPE trim to seq_len (mirrors the model). At tile-aligned S=2816 the
    # create_heads/rotary output is already S, so these are no-ops (the blank-time slices).
    if int(q.shape[2]) != s:
        q = q[:, :, :s, :]
    if int(k.shape[2]) != s:
        k = k[:, :, :s, :]

    # Paged KV-cache fill (mirrors _tp_gather_kv_for_paged_cache + paged_fill_on_device):
    # gather the per-device single KV head into the full 2 KV heads, replicate BF16, then
    # write K and V into a paged cache -> 2 PagedFillCacheDeviceOperations. Side-write only;
    # does not affect the SDPA output below.
    _paged_fill_kv(mesh_device, k, v, s, replicate)

    ctx = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        scale=HEAD_DIM**-0.5,
        program_config=sdpa_pc,
        compute_kernel_config=ckc,
        memory_config=l1,
    )
    ctx = ttnn.experimental.nlp_concat_heads(ctx, memory_config=l1)  # [1,1,S,384] per device

    # --- all_gather the head shards -> full [1,1,S,1536] (o_proj input) ---
    ctx_full = ttnn.all_gather(
        ctx, dim=3, num_links=1, cluster_axis=1, memory_config=mem, topology=ttnn.Topology.Linear
    )
    ttnn.deallocate(ctx)

    # --- o_proj: column(N)-split (replicated in, col-sharded weight) -> [M, 384]/dev, N-sharded ---
    # W_o = [out=1536, in=1536]; device d owns out rows [d*384:(d+1)*384]; W.T=[in, out_shard].
    o_slabs = [
        wo[d * (HIDDEN_SIZE // tp) : (d + 1) * (HIDDEN_SIZE // tp)].t().contiguous() for d in range(tp)
    ]  # [1536, 384]/dev
    o_w_tt = up(torch.stack(o_slabs, dim=0), ttnn.bfloat8_b, shard0)
    o_pc = vision_matmul_program_config(mesh_device, s, HIDDEN_SIZE, HIDDEN_SIZE // tp)  # M x 1536 x 384
    out_tt = ttnn.linear(
        ctx_full, o_w_tt, dtype=ttnn.bfloat8_b, memory_config=l1, compute_kernel_config=ckc, program_config=o_pc
    )

    # o_proj output is N-sharded (384/dev); gather on N to reconstruct the full [M, 1536].
    out_full = (
        ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
        .float()
        .reshape(s, HIDDEN_SIZE)
    )

    passing, value = comp_pcc(ref, out_full, 0.97)
    print(f"\n[text attn prefill TP4] S={s}  N-split QKV(3Q+1KV) + causal GQA SDPA + N-split o_proj  PCC={value}")
    assert passing, f"TP4 text attention prefill PCC below 0.97: {value}"
