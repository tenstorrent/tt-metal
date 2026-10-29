# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-VL prefill coverage for the Quasar SDPA fork (ttnn.experimental.quasar.transformer.scaled_dot_product_attention).

Shapes and configs follow the model's two prefill call sites (models/demos/qwen3_vl/tt/{vision_attention,attention}.py):
  * vision: 16 heads, head_dim 64, non-causal, scale 1/8
  * text:   32 query heads / 8 kv heads (GQA 4:1), head_dim 128, causal, scale 1/sqrt(128)
Both use 256x256 Q/K chunks and HiFi4 with fp32_dest_acc_en=True. The model runs these at 4096-12288 tokens;
here the sequence is cut to 256/512 (1-2 chunks) to keep emulator runs short. K/V are bf16 (the model's
bfloat8_b is not supported on Quasar).
"""

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.test_sdpa_attention_sink import (
    _check,
    _height_sharded_q,
    _to_device,
)

CHUNK = 256


def _compute_kernel_config(fp32_dest_acc_en):
    # models/tt_transformers/tt/model_config.py compute_kernel_config_hifi4 (fp32_dest_acc_en=True).
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


def _run(device, nh, nkv, seq, head_dim, is_causal, fp32_dest_acc_en):
    torch.manual_seed(1234)
    scale = head_dim**-0.5
    q = torch.randn(1, nh, seq, head_dim).bfloat16()
    k = torch.randn(1, nkv, seq, head_dim).bfloat16()
    v = torch.randn(1, nkv, seq, head_dim).bfloat16()

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=CHUNK,
        k_chunk_size=CHUNK,
        exp_approx_mode=False,
    )
    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention(
        _to_device(q, device),
        _to_device(k, device),
        _to_device(v, device),
        is_causal=is_causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=_compute_kernel_config(fp32_dest_acc_en),
    )
    out = ttnn.to_torch(out)[:, :, :seq, :].float()

    rep = nh // nkv
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(rep, dim=1),
        v.float().repeat_interleave(rep, dim=1),
        is_causal=is_causal,
        scale=scale,
    )
    _check(ref, out)


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32dest", "bf16dest"])
# seq200: the vision patch count need not be tile-aligned; K padding then ends inside a tile, which the
# non-causal padded mask covers with a partial (vertical) tile.
@pytest.mark.parametrize("seq", [200, 256, 512], ids=["seq200_padded", "seq256", "seq512"])
def test_sdpa_qwen3_vl_vision(device, seq, fp32_dest_acc_en):
    _run(device, nh=16, nkv=16, seq=seq, head_dim=64, is_causal=False, fp32_dest_acc_en=fp32_dest_acc_en)


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("seq", [256, 512], ids=["seq256", "seq512"])
def test_sdpa_qwen3_vl_text(device, seq, fp32_dest_acc_en):
    _run(device, nh=32, nkv=8, seq=seq, head_dim=128, is_causal=True, fp32_dest_acc_en=fp32_dest_acc_en)


# ------------------------------------------------------------------------------------------------
# Decode (models/demos/qwen3_vl/tt/attention.py: [paged_]scaled_dot_product_attention_decode)
# ------------------------------------------------------------------------------------------------
# Text decoder: 32/8 heads, head_dim 128, paged KV cache with 32-token blocks, q/k chunk size 0
# (op picks k_chunk), HiFi4 with fp32 dest (compute_kernel_config_sdpa). The cache is cut to 512 tokens.

DEC_NH = 32
DEC_NKV = 8
DEC_HEAD_DIM = 128
DEC_BLOCK_SIZE = 32
DEC_CACHE_SEQ = 512
# cur_pos 63: a partial first block run; 300: many blocks, mid-block end; 511: the whole cache.
DEC_POSITIONS = [63, 300, 511]


def _decode_kernel_config(fp32_dest_acc_en):
    # models/tt_transformers/tt/model_config.py compute_kernel_config_sdpa
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )


def _decode_inputs():
    torch.manual_seed(4321)
    q = torch.randn(1, 1, DEC_NH, DEC_HEAD_DIM).bfloat16()
    k = torch.randn(1, DEC_NKV, DEC_CACHE_SEQ, DEC_HEAD_DIM).bfloat16()
    v = torch.randn(1, DEC_NKV, DEC_CACHE_SEQ, DEC_HEAD_DIM).bfloat16()
    return q, k, v


def _decode_ref(q, k, v, cur_pos):
    # q: [1, 1, nh, d]; k/v: [1, nkv, S, d] -> [1, 1, nh, d]
    rep = DEC_NH // DEC_NKV
    kk = k[0, :, : cur_pos + 1].float().repeat_interleave(rep, dim=0)
    vv = v[0, :, : cur_pos + 1].float().repeat_interleave(rep, dim=0)
    w = torch.softmax(torch.einsum("hd,hnd->hn", q[0, 0].float(), kk) * DEC_HEAD_DIM**-0.5, dim=-1)
    return torch.einsum("hn,hnd->hd", w, vv)[None, None]


def _decode_program_config(device):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=0,
        exp_approx_mode=False,
    )


def _cur_pos_tensor(cur_pos, device):
    return _to_device(
        torch.tensor([cur_pos], dtype=torch.int32), device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("cur_pos", DEC_POSITIONS, ids=[f"pos{p}" for p in DEC_POSITIONS])
def test_paged_sdpa_decode_qwen3_vl_text(device, cur_pos, fp32_dest_acc_en):
    q, k, v = _decode_inputs()
    blocks = DEC_CACHE_SEQ // DEC_BLOCK_SIZE

    def to_paged(cache):
        return cache.reshape(DEC_NKV, blocks, DEC_BLOCK_SIZE, DEC_HEAD_DIM).transpose(0, 1)

    permutation = torch.randperm(blocks)
    page_table = torch.argsort(permutation).reshape(1, blocks).to(torch.int32)

    out = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode(
        _height_sharded_q(q, device),
        _to_device(to_paged(k)[permutation].contiguous(), device),
        _to_device(to_paged(v)[permutation].contiguous(), device),
        _to_device(page_table, device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        cur_pos_tensor=_cur_pos_tensor(cur_pos, device),
        scale=DEC_HEAD_DIM**-0.5,
        program_config=_decode_program_config(device),
        compute_kernel_config=_decode_kernel_config(fp32_dest_acc_en),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(out)[:, :, :DEC_NH, :]
    _check(_decode_ref(q, k, v, cur_pos), out.float())


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("cur_pos", DEC_POSITIONS, ids=[f"pos{p}" for p in DEC_POSITIONS])
def test_sdpa_decode_qwen3_vl_text(device, cur_pos, fp32_dest_acc_en):
    q, k, v = _decode_inputs()
    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode(
        _to_device(q, device),
        _to_device(k, device),
        _to_device(v, device),
        cur_pos_tensor=_cur_pos_tensor(cur_pos, device),
        scale=DEC_HEAD_DIM**-0.5,
        program_config=_decode_program_config(device),
        compute_kernel_config=_decode_kernel_config(fp32_dest_acc_en),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(out)[:, :, :DEC_NH, :]
    _check(_decode_ref(q, k, v, cur_pos), out.float())
