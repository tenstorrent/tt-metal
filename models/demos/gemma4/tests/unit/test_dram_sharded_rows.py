# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DramShardedLinear on decode rows below one tile (batch 1 / 8): the GEMMA4_DRAM_SHARD_ROW_VIEW path (metadata
reshapes of the row tile) must give the same result as the shipped Pad -> matmul -> Slice path and launch two
programs fewer per matmul (the dense 31B / 12B run four of these per layer). Single chip, the dense per-chip
shapes (31B tp4 wqkv 5376 x 4096, o_proj 2048 x 5376; 12B tp4 o_proj 1024 x 3840)."""

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear, can_dram_shard

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


def _device_ops(captured):
    """Device operations of a graph capture (ReshapeView launches no program)."""
    return [
        n["params"].get("name", "?")
        for n in captured
        if n["node_type"] == "function_start"
        and "Operation" in n["params"].get("name", "")
        and n["params"].get("name") != "ReshapeViewDeviceOperation"
    ]


@pytest.mark.parametrize("K, N", [(5376, 4096), (2048, 5376), (1024, 3840)], ids=["wqkv31b", "oproj31b", "oproj12b"])
@pytest.mark.parametrize("M", [1, 8], ids=["b1", "b8"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_row_view_matches_pad_slice(mesh_device, device_params, K, N, M, monkeypatch):
    if not can_dram_shard(K, N, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{K}, {N}] has no DRAM-sharded decode config on this grid")
    torch.manual_seed(0)
    w = (torch.randn(1, 1, K, N) * 0.02).to(torch.bfloat16)
    x = (torch.randn(1, 1, M, K) * 2).to(torch.bfloat16)
    lin = DramShardedLinear(w, mesh_device, None, k=K, n=N, dtype=ttnn.bfloat8_b, cache_file_name=None)

    def run(view: bool):
        monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1" if view else "0")
        tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        out = lin(tx)  # compile pass
        ttnn.synchronize_device(mesh_device)
        out_t = ttnn.to_torch(out).float()
        # second call under a graph capture: the launched device ops of the warm path
        tx2 = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out2 = lin(tx2)
        ttnn.synchronize_device(mesh_device)
        ops = _device_ops(ttnn.graph.end_graph_capture())
        out2_t = ttnn.to_torch(out2).float()
        assert list(out2.shape) == [1, 1, M, N], out2.shape
        assert torch.equal(out_t, out2_t), "warm call differs from the compile call"
        return out_t, ops

    ref, ops_pad = run(False)
    got, ops_view = run(True)
    print(f"[K{K} N{N} M{M}] pad/slice path ops: {ops_pad}")
    print(f"[K{K} N{N} M{M}] row-view path ops: {ops_view}")
    assert torch.isfinite(got).all()
    assert torch.equal(got, ref), f"row-view result differs: max|diff| {(got - ref).abs().max()}"
    golden = (x.float() @ w.float().reshape(K, N)).reshape(1, 1, M, N)
    pcc = torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()
    assert pcc > 0.999, f"pcc vs torch {pcc}"
    assert len(ops_view) == len(ops_pad) - 2, (ops_pad, ops_view)
    assert not any(("Pad" in o or "Slice" in o or "Untilize" in o or "Tilize" in o) for o in ops_view), ops_view


@pytest.mark.parametrize("K, N", [(5376, 4096), (2048, 5376), (5376, 10752)], ids=["wqkv31b", "oproj31b", "gateup31b"])
@pytest.mark.parametrize("M", [32, 1], ids=["b32", "b1"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_decode_1d_matches_dram_sharded(mesh_device, device_params, K, N, M, monkeypatch):
    """GEMMA4_DRAM_SHARD_DECODE_1D: the 1D-multicast decode on the interleaved copy of the .ws weight gives the
    DRAM-sharded kernel's result (same bfp8 weight, HiFi2; PCC > 0.9999) in one program instead of three."""
    if not can_dram_shard(K, N, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{K}, {N}] has no DRAM-sharded decode config on this grid")
    torch.manual_seed(0)
    w = (torch.randn(1, 1, K, N) * 0.02).to(torch.bfloat16)
    x = (torch.randn(1, 1, M, K) * 2).to(torch.bfloat16)
    monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1")
    outs, opss = {}, {}
    for mode in ("sharded", "1d"):
        monkeypatch.setenv("GEMMA4_DRAM_SHARD_DECODE_1D", "1" if mode == "1d" else "0")
        lin = DramShardedLinear(w, mesh_device, None, k=K, n=N, dtype=ttnn.bfloat8_b, cache_file_name=None)
        tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        lin(tx)  # compile
        tx2 = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = lin(tx2)
        ttnn.synchronize_device(mesh_device)
        opss[mode] = _device_ops(ttnn.graph.end_graph_capture())
        outs[mode] = ttnn.to_torch(out).float()
        assert list(out.shape) == [1, 1, M, N], out.shape
        lin.weight.deallocate(True)
    golden = (x.float() @ w.float().reshape(K, N)).reshape(1, 1, M, N)
    for mode, o in outs.items():
        pcc = torch.corrcoef(torch.stack([golden.flatten(), o.flatten()]))[0, 1].item()
        print(f"[K{K} N{N} M{M} {mode}] pcc vs torch {pcc:.6f} ops {opss[mode]}")
        assert torch.isfinite(o).all()
        assert pcc > 0.999, (mode, pcc)
    pcc = torch.corrcoef(torch.stack([outs["sharded"].flatten(), outs["1d"].flatten()]))[0, 1].item()
    assert pcc > 0.9999, f"1D decode differs from the DRAM-sharded kernel: pcc {pcc}"
    assert len(opss["1d"]) == 1 and "Matmul" in opss["1d"][0], opss["1d"]
