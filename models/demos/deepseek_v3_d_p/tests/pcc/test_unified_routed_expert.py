# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for the unified (single-op) routed expert FFN.

Mirrors test_single_routed_expert.py but calls the new
ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn directly
instead of going through TtRoutedExpert.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from tests.ttnn.utils_for_testing import comp_pcc

COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [
        (2048, 7168, 2048),  # DeepSeek V3 dims, 2K tokens (single chunk)
        (4096, 7168, 2048),  # 2 chunks
        (8192, 7168, 2048),  # 4 chunks
    ],
    ids=["ds-v3-2k", "ds-v3-4k", "ds-v3-8k"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            1,
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            id="single-chip",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_unified_routed_expert(
    mesh_device,
    device_params,
    num_tokens: int,
    emb_dim: int,
    hidden_dim: int,
):
    """Minimum-viable PCC test for the unified single-op routed expert."""

    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)
    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)

    with torch.no_grad():
        torch_output = torch_expert(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )

    # TTNN weights — match the existing routed-expert layout:
    #   gate/up: (emb, hidden), down: (hidden, emb)
    def _w(t):
        return ttnn.from_torch(
            t.T.contiguous(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat4_b,
        )

    tt_gate = _w(weights["gate_proj"])
    tt_up = _w(weights["up_proj"])
    tt_down = _w(weights["down_proj"])

    def _idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    counts = _idx_tensor([num_tokens])
    idx_table = _idx_tensor([0])
    # Single-expert PCC test: only one region starting at row 0, so the
    # device-side region_offsets read returns 0 and start_tile_row = 0.
    region_offsets = _idx_tensor([0])

    tt_output = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
        tt_input,
        tt_gate,
        tt_up,
        tt_down,
        counts,
        idx_table,
        region_offsets,
        local_expert_id=0,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
    )

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )

    print(
        f"[unified] tt out: shape={tt_output_torch.shape}, "
        f"min={tt_output_torch.min().item():.4f}, max={tt_output_torch.max().item():.4f}, "
        f"abs_mean={tt_output_torch.float().abs().mean().item():.4f}, "
        f"nz_frac={(tt_output_torch.float().abs() > 1e-6).float().mean().item():.4f}",
        flush=True,
    )
    print(
        f"[unified] torch out: min={torch_output.min().item():.4f}, max={torch_output.max().item():.4f}, "
        f"abs_mean={torch_output.float().abs().mean().item():.4f}",
        flush=True,
    )
    _, pcc = comp_pcc(torch_output, tt_output_torch)
    logger.info(f"unified routed expert PCC: {pcc:.6f}")
    assert pcc >= 0.97, f"PCC {pcc:.6f} below threshold 0.97"
    assert not torch.isnan(tt_output_torch).any(), "Output has NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output has Inf"


def _make_weights(emb_dim: int, hidden_dim: int, seed: int = 42):
    torch.manual_seed(seed)
    return {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }


def _replicated_tensor(t: torch.Tensor, mesh_device, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=layout,
        device=mesh_device,
        dtype=dtype,
    )


def _tt_weights(weights: dict, mesh_device):
    def _w(t):
        return _replicated_tensor(t.T.contiguous(), mesh_device, ttnn.bfloat4_b)

    return _w(weights["gate_proj"]), _w(weights["up_proj"]), _w(weights["down_proj"])


def _idx_tensor(values, mesh_device):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )


# ---------------------------------------------------------------------------
# Coverage tests for the new kernel features (reviewer #13).
#
# All cases here use a single expert + DS-V3 dims, varying the kernel-level
# inputs the parent test does not exercise:
#   - count < num_tokens (chunk-bounded loop runtime path)
#   - count == 0 (zero-token expert path)
#   - region_offsets > 0 (use_region_offsets=true with mid-buffer placement)
#   - non-power-of-2 chunk picker candidates (M_tiles not divisible by 64)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "buf_tokens, count_tokens",
    [
        (2048, 1024),  # half-full single chunk
        (4096, 2048),  # full first chunk, empty second
        (4096, 1024),  # quarter-full
    ],
    ids=["buf2k-cnt1k", "buf4k-cnt2k", "buf4k-cnt1k"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")],
    indirect=["mesh_device", "device_params"],
)
def test_unified_routed_expert_partial_count(mesh_device, device_params, buf_tokens: int, count_tokens: int):
    """Verify the kernel's count-bounded chunk loop: only the first
    ceil(count/TILE) tile-rows are computed; the rest of the buffer is ignored."""
    emb_dim, hidden_dim = 7168, 2048
    weights = _make_weights(emb_dim, hidden_dim)
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    torch_input = torch.randn(buf_tokens, emb_dim, dtype=torch.float32)
    with torch.no_grad():
        torch_output_full = torch_expert(torch_input)
    # Reference: only the first `count_tokens` rows are valid.
    torch_output = torch_output_full[:count_tokens]

    tt_input = _replicated_tensor(torch_input, mesh_device, ttnn.bfloat8_b)
    tt_gate, tt_up, tt_down = _tt_weights(weights, mesh_device)

    counts = _idx_tensor([count_tokens], mesh_device)
    idx_table = _idx_tensor([0], mesh_device)
    region_offsets = _idx_tensor([0], mesh_device)

    tt_output = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
        tt_input,
        tt_gate,
        tt_up,
        tt_down,
        counts,
        idx_table,
        region_offsets,
        local_expert_id=0,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
    )
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Compare only the valid prefix.
    tt_valid = tt_output_torch[:count_tokens]
    _, pcc = comp_pcc(torch_output, tt_valid)
    logger.info(f"[partial-count buf={buf_tokens} cnt={count_tokens}] PCC: {pcc:.6f}")
    assert pcc >= 0.97, f"PCC {pcc:.6f} below threshold 0.97"
    assert not torch.isnan(tt_valid).any()
    assert not torch.isinf(tt_valid).any()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")],
    indirect=["mesh_device", "device_params"],
)
def test_unified_routed_expert_zero_count(mesh_device, device_params):
    """count == 0: the kernel must complete without crashing and produce
    no NaN/Inf in the output buffer (output content beyond count is don't-care)."""
    buf_tokens, emb_dim, hidden_dim = 2048, 7168, 2048
    weights = _make_weights(emb_dim, hidden_dim)

    torch_input = torch.randn(buf_tokens, emb_dim, dtype=torch.float32)
    tt_input = _replicated_tensor(torch_input, mesh_device, ttnn.bfloat8_b)
    tt_gate, tt_up, tt_down = _tt_weights(weights, mesh_device)

    counts = _idx_tensor([0], mesh_device)
    idx_table = _idx_tensor([0], mesh_device)
    region_offsets = _idx_tensor([0], mesh_device)

    tt_output = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
        tt_input,
        tt_gate,
        tt_up,
        tt_down,
        counts,
        idx_table,
        region_offsets,
        local_expert_id=0,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
    )
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # No correctness gate on the content (count==0 means no rows are
    # computed); just confirm the kernel finished cleanly.
    assert not torch.isnan(tt_output_torch).any(), "Output has NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output has Inf"


@pytest.mark.parametrize(
    "buf_tokens, count_tokens, offset_tokens",
    [
        (2048, 1024, 1024),  # second half of a 2k buffer
        (4096, 1024, 2048),  # third quarter of a 4k buffer
    ],
    ids=["buf2k-off1k", "buf4k-off2k"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")],
    indirect=["mesh_device", "device_params"],
)
def test_unified_routed_expert_region_offset(
    mesh_device, device_params, buf_tokens: int, count_tokens: int, offset_tokens: int
):
    """use_region_offsets=true with a non-zero start: kernels must read from
    and write to a tile-aligned mid-buffer slice."""
    assert offset_tokens % 32 == 0, "region_offsets must be tile-aligned"
    emb_dim, hidden_dim = 7168, 2048
    weights = _make_weights(emb_dim, hidden_dim)
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    torch_input = torch.randn(buf_tokens, emb_dim, dtype=torch.float32)
    expected = torch_input[offset_tokens : offset_tokens + count_tokens]
    with torch.no_grad():
        torch_output = torch_expert(expected)

    tt_input = _replicated_tensor(torch_input, mesh_device, ttnn.bfloat8_b)
    tt_gate, tt_up, tt_down = _tt_weights(weights, mesh_device)

    counts = _idx_tensor([count_tokens], mesh_device)
    idx_table = _idx_tensor([0], mesh_device)
    region_offsets = _idx_tensor([offset_tokens], mesh_device)

    tt_output = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
        tt_input,
        tt_gate,
        tt_up,
        tt_down,
        counts,
        idx_table,
        region_offsets,
        local_expert_id=0,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
    )
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    tt_valid = tt_output_torch[offset_tokens : offset_tokens + count_tokens]
    _, pcc = comp_pcc(torch_output, tt_valid)
    logger.info(f"[region-offset buf={buf_tokens} off={offset_tokens} cnt={count_tokens}] PCC: {pcc:.6f}")
    assert pcc >= 0.97, f"PCC {pcc:.6f} below threshold 0.97"


@pytest.mark.parametrize(
    "num_tokens",
    # Sizes that force the picker off pure powers-of-2:
    #   3072 = 96 tiles  (picker picks 48 -> 2 chunks, or 32 -> 3 chunks)
    #   5120 = 160 tiles (picker picks 40 -> 4 chunks, or 32/64 -> 5/3 chunks)
    [3072, 5120],
    ids=["ds-v3-3k-odd", "ds-v3-5k-odd"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")],
    indirect=["mesh_device", "device_params"],
)
def test_unified_routed_expert_odd_chunks(mesh_device, device_params, num_tokens: int):
    """Token counts that don't divide cleanly by 64 — exercise the picker's
    non-power-of-2 chunk_M_tiles candidates."""
    emb_dim, hidden_dim = 7168, 2048
    weights = _make_weights(emb_dim, hidden_dim)
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)
    with torch.no_grad():
        torch_output = torch_expert(torch_input)

    tt_input = _replicated_tensor(torch_input, mesh_device, ttnn.bfloat8_b)
    tt_gate, tt_up, tt_down = _tt_weights(weights, mesh_device)

    counts = _idx_tensor([num_tokens], mesh_device)
    idx_table = _idx_tensor([0], mesh_device)
    region_offsets = _idx_tensor([0], mesh_device)

    tt_output = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
        tt_input,
        tt_gate,
        tt_up,
        tt_down,
        counts,
        idx_table,
        region_offsets,
        local_expert_id=0,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
    )
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    _, pcc = comp_pcc(torch_output, tt_output_torch)
    logger.info(f"[odd-chunks {num_tokens}] PCC: {pcc:.6f}")
    assert pcc >= 0.97, f"PCC {pcc:.6f} below threshold 0.97"
