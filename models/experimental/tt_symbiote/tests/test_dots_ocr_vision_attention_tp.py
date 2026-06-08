# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-mesh TP4 (Wormhole, 4x n150) test for the dots.ocr vision attention.

Megatron-style head/tensor parallel, matching the model's op sequence:

    QKV matmul   : COLUMN-parallel -> split the N (output=4608) dim by head.
                   11264 x 1536 x 4608  ->  11264 x 1536 x 1152 per device
                   (each device owns 3 Q + 3 K + 3 V heads, slab [Q_d|K_d|V_d]).
    create_heads / rotary(Q) / rotary(K) / typecast(V) / SDPA / concat_heads
                 : per device on its 3 heads -> concat is [11264, 384] per device
                   (the head shard stays un-gathered -- it is the K shard for o_proj).
    o_proj matmul: ROW-parallel -> split the K (input=1536) dim.
                   11264 x 1536 x 1536  ->  11264 x 384 x 1536 per device
                   (each device matmuls its 384-wide head shard against W[kshard]).
    all-reduce   : ReduceScatter (sum partials, scatter on N) + AllGather (regather N)
                   -> full [11264, 1536] replicated. Bias added while N-sharded.

So the QKV matmul shards N and the o_proj matmul shards K, with one all-reduce between
SDPA's concat and the replicated residual stream -- the column-then-row pattern that needs
a single collective. CCL runs on real fabric (FABRIC_1D); QKV / proj weights are read off
the HF module and 2D-RoPE cos/sin are computed in torch, so this does NOT import the
``dots_ocr_vision`` module.

PCC reference (float 12-head attention on the same weights) is checked only at small S; the
S=11264 shape is for Tracy profiling (the float SDPA reference would OOM), where it only
validates output shape + replication. Auto-skips with fewer than 4 devices.

Run (correctness):  pytest .../test_dots_ocr_vision_attention_tp.py -k s256 -s
Run (Tracy):        python -m tracy -v -r -p -m pytest .../test_dots_ocr_vision_attention_tp.py -k s11264
"""
from __future__ import annotations

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc
from models.experimental.tt_symbiote.tests._vision_tp_matmul import vision_matmul_program_config

HIDDEN_SIZE = 1536
NUM_HEADS = 12
HEAD_DIM = 128
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


def _vision_attention_weights():
    """dots.ocr vision attention weights: fused qkv ``[4608,1536]`` + proj ``[1536,1536]``."""
    from transformers import AutoConfig, AutoModelForCausalLM

    model_config = AutoConfig.from_pretrained(_resolve_model_path(), trust_remote_code=True)
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None:
        for attr in ("num_hidden_layers", "num_layers", "depth"):
            if hasattr(vision_config, attr):
                setattr(vision_config, attr, 1)
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    blocks = getattr(hf_model.vision_tower, "blocks", getattr(hf_model.vision_tower, "layers", None))
    assert blocks is not None, "dots.ocr vision tower should expose blocks/layers"
    attn = blocks[0].attn

    def wb(lin):
        return lin.weight.data.clone(), (lin.bias.data.clone() if lin.bias is not None else None)

    w_qkv, b_qkv = wb(attn.qkv)  # [3*1536, 1536], order [Q_all | K_all | V_all]
    w_o, b_o = wb(attn.proj)  # [1536, 1536] = [out, in]
    del hf_model
    return (w_qkv, b_qkv), (w_o, b_o)


def _vision_rope_cos_sin(h, w, head_dim=HEAD_DIM, theta=10000.0, sms=2):
    """2D factored RoPE cos/sin ``[1,1,S,head_dim]`` (matches TTNNDotsVision2DRoPE)."""
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    h_grid = torch.arange(h, dtype=torch.float32).unsqueeze(1).expand(h, w)
    w_grid = torch.arange(w, dtype=torch.float32).unsqueeze(0).expand(h, w)
    h_grid = h_grid.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).reshape(-1)
    w_grid = w_grid.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).reshape(-1)
    freqs_h = h_grid.unsqueeze(1) * inv_freq.unsqueeze(0)
    freqs_w = w_grid.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos_half = torch.cat([torch.cos(freqs_h), torch.cos(freqs_w)], dim=-1)
    sin_half = torch.cat([torch.sin(freqs_h), torch.sin(freqs_w)], dim=-1)
    cos = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos.to(torch.bfloat16), sin.to(torch.bfloat16)


def _head_shard_qkv(w_qkv, b_qkv, tp):
    """Per-device fused QKV slabs ``[Q_d | K_d | V_d]`` stacked on dim 0 (column/N split)."""
    q, k, v = torch.split(w_qkv, [HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE], dim=0)
    rows = (NUM_HEADS // tp) * HEAD_DIM  # 3 heads -> 384 rows
    slabs_w, slabs_b = [], ([] if b_qkv is not None else None)
    bq = bk = bv = (None, None, None)
    if b_qkv is not None:
        bq, bk, bv = torch.split(b_qkv, [HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE], dim=0)
    for d in range(tp):
        sl = slice(d * rows, (d + 1) * rows)
        slabs_w.append(torch.cat([q[sl], k[sl], v[sl]], dim=0))  # [1152, 1536]
        if b_qkv is not None:
            slabs_b.append(torch.cat([bq[sl], bk[sl], bv[sl]], dim=0))
    stacked_w = torch.stack([s.t().contiguous() for s in slabs_w], dim=0)  # [tp, 1536, 1152]
    stacked_b = torch.stack([s.reshape(1, -1).contiguous() for s in slabs_b], dim=0) if b_qkv is not None else None
    return stacked_w, stacked_b


def _torch_vision_attention_reference(hidden, w_qkv, b_qkv, w_o, b_o, cos, sin):
    """Full 12-head vision attention in float, matching the device op sequence exactly."""
    s = int(hidden.shape[2])
    x = hidden.reshape(s, HIDDEN_SIZE).float()
    qkv = x @ w_qkv.t().float()
    if b_qkv is not None:
        qkv = qkv + b_qkv.float()
    q, k, v = torch.split(qkv, [HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE], dim=-1)

    def to_heads(t):  # [S, dim] -> [1, H, S, hd]
        return t.reshape(s, NUM_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)

    q, k, v = to_heads(q), to_heads(k), to_heads(v)
    cos_f, sin_f = cos.float(), sin.float()  # [1,1,S,hd], broadcast over heads

    def rope(t):  # rotate_half: cat((-x2, x1))
        x1, x2 = t[..., : HEAD_DIM // 2], t[..., HEAD_DIM // 2 :]
        return t * cos_f + torch.cat((-x2, x1), dim=-1) * sin_f

    q, k = rope(q), rope(k)
    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    ctx = torch.softmax(scores.float(), dim=-1) @ v  # [1, H, S, hd]
    ctx = ctx.squeeze(0).permute(1, 0, 2).reshape(s, HIDDEN_SIZE)
    out = ctx @ w_o.t().float()
    if b_o is not None:
        out = out + b_o.float()
    return out  # [S, dim]


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
@pytest.mark.parametrize("grid_hw", [(16, 16), (88, 128)], ids=["s256", "s11264"])
def test_dots_ocr_vision_attention_tp4(mesh_device, grid_hw):
    """TP4 vision attention: QKV column(N)-split + o_proj row(K)-split with one all-reduce."""
    tp = int(mesh_device.get_num_devices())
    if tp < 4:
        pytest.skip("vision-attention TP4 requires 4 devices")
    tp = 4
    if tuple(mesh_device.shape) != (1, 4):
        mesh_device.reshape(ttnn.MeshShape(1, 4))
    assert NUM_HEADS % tp == 0, "12 vision heads must split evenly across TP4"

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    mem = ttnn.DRAM_MEMORY_CONFIG

    grid_h, grid_w = grid_hw
    seq_len = grid_h * grid_w
    check_pcc = seq_len <= 1024  # float SDPA reference OOMs at S=11264; large S is profiling-only

    (w_qkv, b_qkv), (w_o, b_o) = _vision_attention_weights()
    cos_full, sin_full = _vision_rope_cos_sin(grid_h, grid_w)

    hidden_torch = torch.randn(1, 1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    ref = (
        _torch_vision_attention_reference(hidden_torch, w_qkv, b_qkv, w_o, b_o, cos_full, sin_full)
        if check_pcc
        else None
    )

    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True
    )
    grid = mesh_device.compute_with_storage_grid_size()
    sdpa_pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=256,
        k_chunk_size=256,
        exp_approx_mode=True,
    )

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    shard0 = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    shardN = ttnn.ShardTensorToMesh(mesh_device, dim=-1)

    def up(t, dtype, mapper):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=mem, mesh_mapper=mapper
        )

    # Replicated activation + RoPE tables.
    x_tt = up(hidden_torch, ttnn.bfloat8_b, replicate)
    cos_tt = up(cos_full, ttnn.bfloat8_b, replicate)
    sin_tt = up(sin_full, ttnn.bfloat8_b, replicate)

    # --- QKV: column(N)-split by head -> [M, 1152] per device ---
    stacked_w, stacked_b = _head_shard_qkv(w_qkv, b_qkv, tp)
    qkv_w_tt = up(stacked_w, ttnn.bfloat8_b, shard0)
    qkv_b_tt = up(stacked_b, ttnn.bfloat8_b, shard0) if stacked_b is not None else None
    qkv_pc = vision_matmul_program_config(
        mesh_device, seq_len, HIDDEN_SIZE, (NUM_HEADS // tp) * 3 * HEAD_DIM
    )  # M x 1536 x 1152
    qkv = ttnn.linear(
        x_tt,
        qkv_w_tt,
        bias=qkv_b_tt,
        dtype=ttnn.bfloat8_b,
        memory_config=mem,
        compute_kernel_config=ckc,
        program_config=qkv_pc,
    )

    # --- per-device heads -> RoPE -> SDPA -> concat (stays head-sharded = K shard) ---
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=NUM_HEADS // tp, num_kv_heads=NUM_HEADS // tp, transpose_k_heads=False, memory_config=mem
    )
    ttnn.deallocate(qkv)
    q = ttnn.experimental.rotary_embedding(q, cos_tt, sin_tt, memory_config=mem)
    k = ttnn.experimental.rotary_embedding(k, cos_tt, sin_tt, memory_config=mem)
    v = ttnn.typecast(v, ttnn.bfloat4_b, memory_config=mem)  # mirror the model: BFP4 V into SDPA
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        scale=HEAD_DIM**-0.5,
        program_config=sdpa_pc,
        compute_kernel_config=ckc,
        memory_config=mem,
    )
    l1 = ttnn.L1_MEMORY_CONFIG
    ctx = ttnn.experimental.nlp_concat_heads(ctx, memory_config=l1)  # [1,1,M,384] per device, L1 (o_proj in)

    # --- o_proj: row(K)-split. W_o is [out=1536, in=1536]; shard the in (K) dim. ---
    # W_o.T = [in, out]; ShardTensorToMesh(dim=0) hands device d rows [d*384:(d+1)*384].
    # Out dtype BFP8, in/out kept in L1; reduce_scatter reads the L1 partial.
    o_w_tt = up(w_o.t().contiguous(), ttnn.bfloat8_b, shard0)
    o_pc = vision_matmul_program_config(
        mesh_device, seq_len, (NUM_HEADS // tp) * HEAD_DIM, HIDDEN_SIZE
    )  # M x 384 x 1536
    partial = ttnn.linear(
        ctx, o_w_tt, dtype=ttnn.bfloat8_b, memory_config=l1, compute_kernel_config=ckc, program_config=o_pc
    )  # [1,1,M,1536] partial
    ttnn.deallocate(ctx)

    # --- all-reduce = reduce_scatter (sum, scatter on N) + all_gather (regather N) ---
    out_rs = ttnn.reduce_scatter(
        partial, dim=3, num_links=1, cluster_axis=1, memory_config=mem, topology=ttnn.Topology.Linear
    )
    ttnn.deallocate(partial)
    if b_o is not None:
        # Bias added while N-sharded: device d holds output cols [d*384:(d+1)*384].
        o_b_tt = up(b_o.reshape(1, -1), ttnn.bfloat16, shardN)
        out_rs = ttnn.add(out_rs, o_b_tt)
    out_tt = ttnn.all_gather(
        out_rs, dim=3, num_links=1, cluster_axis=1, memory_config=mem, topology=ttnn.Topology.Linear
    )
    ttnn.deallocate(out_rs)

    out_gathered = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))  # [4,1,M,1536]
    assert list(out_gathered.shape) == [tp, 1, seq_len, HIDDEN_SIZE]
    for d in range(1, tp):
        assert comp_pcc(out_gathered[0], out_gathered[d], 0.999)[0], f"all-reduce output not replicated on device {d}"

    if not check_pcc:
        ttnn.synchronize_device(mesh_device)
        print(f"\n[vision attn TP4] S={seq_len}  profiling shape ran (PCC skipped: float ref OOMs)")
        return

    out_dev = out_gathered[0].reshape(seq_len, HIDDEN_SIZE)
    # BFP8 Q/K + BFP4 V + BFP8 o_proj output + LoFi SDPA is the precision floor; ~0.97
    # tight-but-survivable (measured ~0.98 at seed 1234).
    passing, value = comp_pcc(ref, out_dev, 0.97)
    print(f"\n[vision attn TP4] S={seq_len}  N-split QKV + K-split o_proj (BFP8 out, L1)  PCC={value}")
    assert passing, f"TP4 vision attention PCC below 0.97: {value}"
