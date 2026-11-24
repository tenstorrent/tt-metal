# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 RMSNorm — uses HF Gemma4RMSNorm as reference.

    pytest -k "1x1"   # single card
    pytest -k "1x8"   # T3K (RMSNorm is replicated, no TP sharding)
"""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear, can_dram_shard
from models.demos.gemma4.tt.rms_norm import SHARDED_OUT_DEFAULT_MAX_W, RMSNorm, sharded_layout_program_config

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    get_pcc_threshold,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
)


@parametrize_mesh_with_fabric()
@parametrize_batch_seq()
def test_rms_norm_with_scale(batch_size, seq_len, mesh_device, reset_seeds, request):
    """Test RMSNorm (with_scale=True) against HF Gemma4RMSNorm."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    hf_config = TestFactory.create_hf_config()
    hidden_size = hf_config.hidden_size

    hf_norm = Gemma4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps, with_scale=True)
    hf_norm.eval()
    state_dict = {"weight": hf_norm.weight.data.clone()}

    tt_norm = RMSNorm(mesh_device=mesh_device, hf_config=hf_config, state_dict=state_dict, with_scale=True)

    x_torch = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_output = hf_norm(x_torch.float()).to(torch.bfloat16)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_norm.forward(x_tt)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"RMSNorm with_scale PCC too low: {pcc_msg}"


@parametrize_mesh_with_fabric()
@parametrize_batch_seq()
def test_rms_norm_without_scale(batch_size, seq_len, mesh_device, reset_seeds, request):
    """Test RMSNorm (with_scale=False) against HF Gemma4RMSNorm."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    hf_config = TestFactory.create_hf_config()
    hidden_size = hf_config.hidden_size

    hf_norm = Gemma4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps, with_scale=False)
    hf_norm.eval()

    tt_norm = RMSNorm(mesh_device=mesh_device, hf_config=hf_config, state_dict={}, with_scale=False)

    x_torch = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_output = hf_norm(x_torch.float()).to(torch.bfloat16)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_norm.forward(x_tt)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"RMSNorm without_scale PCC too low: {pcc_msg}"


# ===== merged from tests/unit/test_rms_norm_sharded_out.py: The native width-sharded decode RMSNorm run on a DRAM-sharded decode matmul's activation grid with its output
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
