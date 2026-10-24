# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The native width-sharded decode RMSNorm run on a DRAM-sharded decode matmul's activation grid with its output
left in that layout (RMSNorm.forward(out_memory_config=...), GEMMA4_NORM_SHARDED_OUT=1 on the native norm path):
the norm equals the module's own-grid path up to the reduction order, the matmul fed from it equals the matmul fed
from the interleaved output, and a layout of wide shards (the 8-core wqkv grids) is declined unless the cap allows
it (the 8-core wqkv grids are within the default cap of 24); single chip, the dense shard shapes (31B / 12B at TP=4)."""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear, can_dram_shard
from models.demos.gemma4.tt.rms_norm import SHARDED_OUT_DEFAULT_MAX_W, RMSNorm, sharded_layout_program_config

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
EPS = 1e-6


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


CI = os.environ.get("CI") == "true"
CASES = {
    "31b_gateup_b32": (32, 5376, 10752, None),
    "31b_gateup_b1": (1, 5376, 10752, None),
    "12b_gateup_b32": (32, 3840, 7680, None),
    "12b_gateup_b1": (1, 3840, 7680, None),
    "31b_wqkv_declined_w16": (32, 5376, 4096, 16),
    "31b_wqkv": (32, 5376, 4096, None),
    "12b_wqkv": (32, 3840, 2048, None),
}
CI_CASES = ("31b_gateup_b1", "12b_wqkv")  # every case compiles its own LayerNorm grid (~20 s cold): two under CI


@pytest.mark.parametrize(
    "R, W, N, max_w",
    [
        pytest.param(*v, id=k, marks=[] if (k in CI_CASES or not CI) else [pytest.mark.skip(reason="CI subset")])
        for k, v in CASES.items()
    ],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_rms_norm_sharded_out(mesh_device, device_params, R, W, N, max_w, monkeypatch):
    if not can_dram_shard(W, N, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{W}, {N}] has no DRAM-sharded decode config on this grid")
    monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1")
    monkeypatch.delenv("GEMMA4_NORM_FUSED", raising=False)
    if max_w is not None:
        monkeypatch.setenv("GEMMA4_NORM_SHARDED_OUT_MAX_W", str(max_w))
    else:
        monkeypatch.delenv("GEMMA4_NORM_SHARDED_OUT_MAX_W", raising=False)
    torch.manual_seed(0)
    x = (torch.randn(1, 1, R, W) * 4).to(torch.bfloat16)
    g = (torch.rand(W) + 0.5).to(torch.bfloat16)
    w = (torch.randn(1, 1, W, N) * 0.02).to(torch.bfloat16)
    lin = DramShardedLinear(w, mesh_device, None, k=W, n=N, dtype=ttnn.bfloat8_b, cache_file_name=None)
    assert lin.accepts_sharded_rows()
    mem = lin.act_memory_config
    norm = RMSNorm(mesh_device, SimpleNamespace(hidden_size=W, rms_norm_eps=EPS), {"weight": g})
    tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ref = norm.forward(tx)
    out = norm.forward(tx, out_memory_config=mem)
    assert not ref.is_sharded()
    ref_t, out_t = ttnn.to_torch(ref).float(), ttnn.to_torch(out).float()
    assert list(out_t.shape) == [1, 1, R, W]
    block_w = int(mem.shard_spec.shape[1]) // ttnn.TILE_SIZE
    if sharded_layout_program_config(mem) is None:
        # the layout was declined (shards wider than the cap): the native own-grid path ran, interleaved output
        assert block_w > (SHARDED_OUT_DEFAULT_MAX_W if max_w is None else max_w) and not out.is_sharded()
        assert torch.equal(out_t, ref_t)
        return
    assert out.is_sharded() and out.memory_config() == mem
    # the same kernel on another grid accumulates the row statistics in another order and with other partial
    # roundings (own grid vs matmul grid: 0.08 max diff, pcc 0.99999 between them at these widths), so both forms
    # are measured against the fp32 golden and the sharded-out form must be as accurate as the own-grid one
    xf = x.float()
    golden = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * g.float()
    pcc_ref, pcc_out = _pcc(ref_t, golden), _pcc(out_t, golden)
    assert pcc_out > 0.9999 and (out_t - golden).abs().max().item() <= 0.25, f"sharded-out norm vs golden {pcc_out:.6f}"
    assert (
        pcc_out >= pcc_ref - 5e-5
    ), f"sharded-out norm less accurate than the own-grid norm: {pcc_out:.6f} vs {pcc_ref:.6f}"
    y_ref = ttnn.to_torch(lin(ref)).float()
    y_out = ttnn.to_torch(lin(out)).float()
    assert list(y_out.shape) == [1, 1, R, N]
    assert (
        _pcc(y_out, y_ref) > 0.9999
    ), "matmul on the sharded norm output differs from the matmul on the interleaved one"
