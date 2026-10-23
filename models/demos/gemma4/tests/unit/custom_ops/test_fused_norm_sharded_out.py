# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The fused decode norms writing a DRAM-sharded decode matmul's L1 activation layout (GEMMA4_NORM_SHARDED_OUT):
FusedNormDecode (single form, the input norm -> wqkv) and FusedNormChainDecode's y (pre-FF norm -> gate_up),
bit-identical to the interleaved outputs, and the matmul fed from them bit-identical to the interleaved path;
single chip, the dense shard shapes (31B / 12B at TP=4)."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.fused_norm_chain import FusedNormChainDecode
from models.demos.gemma4.custom_ops.fused_norm_decode import FusedNormDecode
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear, can_dram_shard

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
EPS = 1e-6


def _dev(t, device, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=mem)


@pytest.mark.parametrize(
    "R, W, N",
    [(32, 5376, 4096), (1, 5376, 4096), (32, 3840, 2048), (1, 3840, 2048), (32, 5376, 10752), (32, 3840, 7680)],
    ids=["31b_wqkv_b32", "31b_wqkv_b1", "12b_wqkv_b32", "12b_wqkv_b1", "31b_gateup_b32", "12b_gateup_b32"],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_fused_norm_sharded_out(mesh_device, device_params, R, W, N, monkeypatch):
    if not can_dram_shard(W, N, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{W}, {N}] has no DRAM-sharded decode config on this grid")
    monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1")
    torch.manual_seed(0)
    x = (torch.randn(1, 1, R, W) * 4).to(torch.bfloat16)
    res = (torch.randn(1, 1, R, W) * 3).to(torch.bfloat16)
    g1 = (torch.rand(W) + 0.5).to(torch.bfloat16)
    g2 = (torch.rand(W) + 0.5).to(torch.bfloat16)
    w = (torch.randn(1, 1, W, N) * 0.02).to(torch.bfloat16)
    lin = DramShardedLinear(w, mesh_device, None, k=W, n=N, dtype=ttnn.bfloat8_b, cache_file_name=None)
    assert lin.accepts_sharded_rows()
    mem = lin.act_memory_config
    tg1 = FusedNormDecode.full_gamma(_dev(g1.reshape(1, 1, 1, W), mesh_device))
    tg2 = FusedNormDecode.full_gamma(_dev(g2.reshape(1, 1, 1, W), mesh_device))
    # single form: norm -> matmul
    n_i = FusedNormDecode.op(_dev(x, mesh_device), [tg1], EPS)[0]
    n_s = FusedNormDecode.op(_dev(x, mesh_device), [tg1], EPS, out_memory_configs=[mem])[0]
    assert n_s.memory_config() == mem and list(n_s.shape) == [1, 1, R, W]
    n_i_t, n_s_t = ttnn.to_torch(n_i).float(), ttnn.to_torch(n_s).float()
    assert torch.equal(n_s_t, n_i_t), "sharded-output norm differs from the interleaved output"
    y_i = ttnn.to_torch(lin(n_i)).float()
    y_s = ttnn.to_torch(lin(n_s)).float()
    assert list(y_s.shape) == [1, 1, R, N] and torch.equal(y_s, y_i), "matmul on the sharded norm output differs"
    # chain form: (post-attn + residual, pre-FF norm sharded) -> matmul
    h_i, c_i = FusedNormChainDecode.op(_dev(x, mesh_device), _dev(res, mesh_device), tg1, tg2, EPS)
    h_s, c_s = FusedNormChainDecode.op(_dev(x, mesh_device), _dev(res, mesh_device), tg1, tg2, EPS, y_memory_config=mem)
    assert c_s.memory_config() == mem and not h_s.is_sharded()
    assert torch.equal(ttnn.to_torch(h_s).float(), ttnn.to_torch(h_i).float())
    assert torch.equal(ttnn.to_torch(c_s).float(), ttnn.to_torch(c_i).float())
    z_i = ttnn.to_torch(lin(c_i)).float()
    z_s = ttnn.to_torch(lin(c_s)).float()
    assert torch.equal(z_s, z_i), "matmul on the chain's sharded y differs"
    print(f"[R{R} W{W} N{N}] sharded-output norms bit-identical; matmul outputs bit-identical")
