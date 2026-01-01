# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GegluShardedDecode (custom op: up * gelu(gate) from the gate_up decode matmul's L1 shards into the down matmul's
L1 activation layout) vs GegluDecode on the interleaved activation and the torch golden, and the DRAM-sharded
down matmul fed from it vs the interleaved path; single chip, the dense MLP shard shapes (31B / 12B at TP=4 / 8)."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.geglu_decode import GegluDecode
from models.demos.gemma4.custom_ops.geglu_sharded import GegluShardedDecode
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear, can_dram_shard

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize(
    "R, H, I",
    [(32, 5376, 5376), (1, 5376, 5376), (32, 3840, 3840), (1, 3840, 3840), (32, 5376, 2688), (32, 3840, 1920)],
    ids=["31b_tp4_b32", "31b_tp4_b1", "12b_tp4_b32", "12b_tp4_b1", "31b_tp8", "12b_tp8"],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_geglu_sharded_matches_interleaved(mesh_device, device_params, R, H, I, monkeypatch):
    if not (can_dram_shard(H, 2 * I, dtype=ttnn.bfloat8_b) and can_dram_shard(I, H, dtype=ttnn.bfloat8_b)):
        pytest.skip(f"no DRAM-sharded decode config for gate_up [{H}, {2 * I}] / down [{I}, {H}] on this grid")
    monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1")
    torch.manual_seed(0)
    x = (torch.randn(1, 1, R, H) * 2).to(torch.bfloat16)
    w_gu = (torch.randn(1, 1, H, 2 * I) * 0.02).to(torch.bfloat16)
    w_dn = (torch.randn(1, 1, I, H) * 0.02).to(torch.bfloat16)
    gate_up = DramShardedLinear(w_gu, mesh_device, None, k=H, n=2 * I, dtype=ttnn.bfloat8_b, cache_file_name=None)
    down = DramShardedLinear(w_dn, mesh_device, None, k=I, n=H, dtype=ttnn.bfloat8_b, cache_file_name=None)
    assert down.accepts_sharded_rows()
    dev = lambda t: ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    # the interleaved chain: gate_up -> DRAM -> GegluDecode -> down (reshard inside)
    gu_i = gate_up(dev(x))
    h_i = GegluDecode.op(gu_i)
    y_i = ttnn.to_torch(down(h_i)).float()
    h_i_t = ttnn.to_torch(h_i).float()
    # the sharded chain
    gu_s = gate_up(dev(x), keep_sharded=True)
    assert gu_s.is_sharded() and gu_s.memory_config().buffer_type == ttnn.BufferType.L1
    h_s = GegluShardedDecode.op(gu_s, down.act_memory_config)
    assert h_s.memory_config() == down.act_memory_config
    h_s_t = ttnn.to_torch(h_s).float()
    golden = GegluShardedDecode.golden(ttnn.to_torch(gu_s).float())
    print(
        f"[R{R} H{H} I{I}] geglu pcc vs golden {_pcc(golden, h_s_t):.6f}; max|diff| vs interleaved {(h_s_t - h_i_t).abs().max():.4f}"
    )
    assert torch.isfinite(h_s_t).all()
    assert _pcc(golden, h_s_t) > 0.9999
    assert torch.equal(h_s_t[:, :, :R], h_i_t[:, :, :R]), "sharded GeGLU differs from the interleaved GeGLU"
    y_s = ttnn.to_torch(down(h_s, rows=R)).float()
    assert list(y_s.shape) == [1, 1, R, H], y_s.shape
    assert torch.equal(y_s, y_i), f"down on the sharded GeGLU differs: max|diff| {(y_s - y_i).abs().max()}"
    # program-cache hit on a fresh activation
    gu_s2 = gate_up(dev(x), keep_sharded=True)
    h_s2 = ttnn.to_torch(GegluShardedDecode.op(gu_s2, down.act_memory_config)).float()
    assert torch.equal(h_s2, h_s_t)
