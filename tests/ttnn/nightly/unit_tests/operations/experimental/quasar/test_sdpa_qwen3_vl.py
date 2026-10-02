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
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.test_sdpa_attention_sink import _check, _to_device

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
@pytest.mark.parametrize("seq", [256, 512], ids=["seq256", "seq512"])
def test_sdpa_qwen3_vl_vision(device, seq, fp32_dest_acc_en):
    _run(device, nh=16, nkv=16, seq=seq, head_dim=64, is_causal=False, fp32_dest_acc_en=fp32_dest_acc_en)


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("seq", [256, 512], ids=["seq256", "seq512"])
def test_sdpa_qwen3_vl_text(device, seq, fp32_dest_acc_en):
    _run(device, nh=32, nkv=8, seq=seq, head_dim=128, is_causal=True, fp32_dest_acc_en=fp32_dest_acc_en)
