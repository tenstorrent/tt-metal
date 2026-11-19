# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Every test in this module takes the function-scoped `device` fixture without
# parametrizing device_params, so the device is opened once per module rather than
# once per test.
pytestmark = pytest.mark.use_module_device


def _decode_ref(q, k, v, cur_pos, scale):
    # q: [1, b, nh, d]; k/v: [b, nkv, S, d]; returns [1, b, nh, d]
    _, b, nh, _ = q.shape
    rep = nh // k.shape[1]
    out = []
    for i in range(b):
        kk = k[i, :, : cur_pos + 1].float().repeat_interleave(rep, dim=0)
        vv = v[i, :, : cur_pos + 1].float().repeat_interleave(rep, dim=0)
        w = torch.softmax(torch.einsum("hd,hnd->hn", q[0, i].float(), kk) * scale, dim=-1)
        out.append(torch.einsum("hn,hnd->hd", w, vv))
    return torch.stack(out)[None]


@pytest.mark.parametrize("head_dim", [384], ids=["d384"])
def test_sdpa_decode_head_dim_not_divisible_by_dst_size(device, head_dim):
    # head_dim / 32 = 12 tiles is not a multiple of the 8-tile half-sync dest, exercising the output
    # subblock split and the DHT granularity used by the normalisation loops.
    torch.manual_seed(0)
    batch, nh, nkv, cache_seq, cur_pos = 1, 8, 1, 512, 300
    scale = head_dim**-0.5
    q = torch.randn(1, batch, nh, head_dim).bfloat16()
    k = torch.randn(batch, nkv, cache_seq, head_dim).bfloat16()
    v = torch.randn(batch, nkv, cache_seq, head_dim).bfloat16()

    to_dev = lambda t, **kw: ttnn.from_torch(t, device=device, **kw)
    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode(
        to_dev(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        to_dev(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        to_dev(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        cur_pos_tensor=to_dev(torch.tensor([cur_pos] * batch, dtype=torch.int32), dtype=ttnn.int32),
        scale=scale,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(out)[:, :, :nh, :]
    assert_with_pcc(_decode_ref(q, k, v, cur_pos, scale), out.float(), 0.99)
