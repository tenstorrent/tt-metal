# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import comp_pcc

# Every test in this module takes the function-scoped `device` fixture without
# parametrizing device_params, so the device is opened once per module rather than
# once per test.
pytestmark = pytest.mark.use_module_device


def _program_config(device):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
    )


def _masked_decode_ref(q, k, v, mask, scale):
    # q: [1, b, nh, d]; k/v: [b, nkv, S, d]; mask: [b, nh, S] additive; returns [1, b, nh, d]
    _, b, nh, _ = q.shape
    rep = nh // k.shape[1]
    out = []
    for i in range(b):
        kk = k[i].float().repeat_interleave(rep, dim=0)
        vv = v[i].float().repeat_interleave(rep, dim=0)
        scores = torch.einsum("hd,hnd->hn", q[0, i].float(), kk) * scale + mask[i]
        out.append(torch.einsum("hn,hnd->hd", torch.softmax(scores, dim=-1), vv))
    return torch.stack(out)[None]


@pytest.mark.parametrize("mask_batch_dim", [0, 1], ids=["b_1_nh_s", "1_b_nh_s"])
def test_sdpa_decode_mask_batch_layout(device, mask_batch_dim):
    # [1, b, nh, s] (the documented layout) and [b, 1, nh, s] hold each user's tiles in the same order.
    # The reader once took the mask batch from dim 0 only, so [1, b, nh, s] gave every user user 0's mask.
    # Each user keeps a different number of keys, so reading another user's mask fails that user's check.
    torch.manual_seed(0)
    batch, nh, nkv, cache_seq, head_dim = 8, 8, 1, 1024, 128
    scale = head_dim**-0.5
    q = torch.randn(1, batch, nh, head_dim).bfloat16()
    k = torch.randn(batch, nkv, cache_seq, head_dim).bfloat16()
    v = torch.randn(batch, nkv, cache_seq, head_dim).bfloat16()
    mask = torch.zeros(batch, nh, cache_seq)
    for i in range(batch):
        mask[i, :, (i + 1) * cache_seq // batch :] = -1.0e4  # user i keeps (i + 1) * 128 keys
    tt_mask = mask.unsqueeze(0) if mask_batch_dim == 1 else mask.unsqueeze(1)

    to_dev = lambda t: ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode(
        to_dev(q),
        to_dev(k),
        to_dev(v),
        is_causal=False,
        attn_mask=to_dev(tt_mask),
        scale=scale,
        program_config=_program_config(device),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(out)[:, :, :nh, :].float()
    expected = _masked_decode_ref(q, k, v, mask, scale)
    for i in range(batch):
        passing, pcc = comp_pcc(expected[:, i], out[:, i], 0.99)
        assert passing, f"User {i} with mask {list(tt_mask.shape)}: {pcc}. The user likely read another user's mask."


@pytest.mark.parametrize(
    "mask_batch_shape, message",
    [
        ((2, 1), "attn_mask batch must be 1"),
        ((1, 2), "attn_mask batch must be 1"),
        ((2, 2), "but got dims 0 and 1 of 2 and 2"),
    ],
    ids=["dim0_mismatch", "dim1_mismatch", "both_dims"],
)
def test_sdpa_decode_mask_batch_rejected(device, expect_error, mask_batch_shape, message):
    batch, nh, cache_seq, head_dim = 4, 8, 256, 64
    to_dev = lambda t: ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, message):
        ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode(
            to_dev(torch.randn(1, batch, nh, head_dim)),
            to_dev(torch.randn(batch, 1, cache_seq, head_dim)),
            to_dev(torch.randn(batch, 1, cache_seq, head_dim)),
            is_causal=False,
            attn_mask=to_dev(torch.zeros(*mask_batch_shape, nh, cache_seq)),
            scale=head_dim**-0.5,
            program_config=_program_config(device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
