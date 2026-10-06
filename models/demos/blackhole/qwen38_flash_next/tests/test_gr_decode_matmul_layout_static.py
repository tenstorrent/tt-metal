# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contracts for the GR and final-mixer decode matmul layouts.

The branch-major residual [1,4,1,640] reads as one flat row indexed
(branch, local hidden).  The weight loaders stack each device's blocks along
the sharded axis in (device, branch, local hidden) order so that a plain
contiguous mesh shard hands every device its own hidden slice of all four
branches, and one DRAM-sharded matmul replaces each batched projection plus
branch sum.  These tests prove the stacked host layouts against the checkpoint
matrices and the TP4 CPU reference, and pin the decode program configs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import ttnn
from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidual, Qwen38GatedResidualWeights
from models.demos.blackhole.qwen38_flash_next.tt.model import Qwen38FinalMixerWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as final_mixer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (
    dram_sharded_matmul_configs,
    dram_sharded_weight_memory_config,
)

TP_SIZE = 4
BRANCHES = 4
HIDDEN = 2560
LOCAL_HIDDEN = HIDDEN // TP_SIZE
RANK = 320
RESIDUAL_WIDTH = BRANCHES * HIDDEN
FLAT_WIDTH = BRANCHES * LOCAL_HIDDEN
# Ten rank tiles, the injection tile, one zero tile: the up weight's K is
# zero-padded to the same row so it splits over two storage cores.
PARTIAL_WIDTH = RANK + 2 * 32


class _Placement:
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        residual_branches=BRANCHES,
        residual_rank=RANK,
        residual_width=RESIDUAL_WIDTH,
        rms_norm_eps=1e-6,
    )

    @property
    def hidden_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple((index * LOCAL_HIDDEN, (index + 1) * LOCAL_HIDDEN) for index in range(TP_SIZE))


def _gr_source(seed: int) -> Qwen38GatedResidualWeights:
    generator = torch.Generator().manual_seed(seed)

    def weight(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * 0.05).to(torch.bfloat16)

    return Qwen38GatedResidualWeights(
        placement=_Placement(),
        layer_index=0,
        block="attn",
        norm=weight(RESIDUAL_WIDTH),
        down=weight(RANK, RESIDUAL_WIDTH),
        up=weight(RESIDUAL_WIDTH, RANK),
        inject=weight(BRANCHES, RESIDUAL_WIDTH),
    )


def _by_device_branch_hidden(matrix: torch.Tensor, axis: int) -> torch.Tensor:
    """Split one residual-width axis (branch, hidden) into (device, branch, local hidden)."""

    split = matrix.unflatten(axis, (BRANCHES, TP_SIZE, LOCAL_HIDDEN))
    return split.movedim(axis + 1, axis).flatten(axis, axis + 2)


def test_gr_stacked_weights_hold_each_device_hidden_slice_in_flat_row_order() -> None:
    source = _gr_source(seed=1)
    prepared = gr_module._prepare_host_weights(source)

    assert tuple(prepared["down_inject"].shape) == (1, 1, TP_SIZE * FLAT_WIDTH, PARTIAL_WIDTH)
    assert tuple(prepared["up"].shape) == (1, 1, PARTIAL_WIDTH, TP_SIZE * FLAT_WIDTH)
    assert tuple(prepared["norm_scale"].shape) == (1, BRANCHES, 1, HIDDEN)
    assert prepared["down_inject"].dtype == prepared["up"].dtype == torch.bfloat16
    # gamma = 1 + w in FP32 with the 1/4 branch mean folded in (exact shift).
    assert torch.equal(prepared["norm_scale"] * BRANCHES, (1.0 + source.norm.float()).reshape(1, BRANCHES, 1, HIDDEN))

    # Checkpoint down is [rank, (branch, hidden)]; the device-stacked K axis is
    # (device, branch, local hidden) so contiguous dim-2 shards are exact.
    down_inject = prepared["down_inject"][0, 0]
    assert torch.equal(down_inject[:, :RANK], _by_device_branch_hidden(source.down, 1).transpose(0, 1))
    assert torch.equal(
        down_inject[:, RANK : RANK + BRANCHES], _by_device_branch_hidden(source.inject, 1).transpose(0, 1)
    )
    assert torch.equal(down_inject[:, RANK + BRANCHES :], torch.zeros_like(down_inject[:, RANK + BRANCHES :]))
    # Checkpoint up is [(branch, hidden), rank]; the stacked N axis matches and
    # the K rows past the rank are the zero pad the fused row's tail meets.
    up = prepared["up"][0, 0]
    assert torch.equal(up[:RANK], _by_device_branch_hidden(source.up, 0).transpose(0, 1))
    assert torch.equal(up[RANK:], torch.zeros_like(up[RANK:]))

    # Device d's local block is rows/columns [2560 d, 2560 (d + 1)): its own
    # hidden slice of every branch, in the flat (branch, local hidden) order.
    for device in range(TP_SIZE):
        shard = source.device_shard(device)
        rows = slice(device * FLAT_WIDTH, (device + 1) * FLAT_WIDTH)
        assert torch.equal(down_inject[rows, :RANK], shard.down.permute(1, 2, 0).reshape(FLAT_WIDTH, RANK))
        assert torch.equal(
            down_inject[rows, RANK : RANK + BRANCHES], shard.inject.permute(1, 2, 0).reshape(FLAT_WIDTH, BRANCHES)
        )
        assert torch.equal(up[:RANK, rows], shard.up.permute(2, 0, 1).reshape(RANK, FLAT_WIDTH))


def test_gr_flat_row_matmuls_reproduce_the_tp4_reference_read() -> None:
    source = _gr_source(seed=2)
    prepared = gr_module._prepare_host_weights(source)
    reference = Qwen38GatedResidual(source)
    generator = torch.Generator().manual_seed(3)
    residual = torch.randn(1, RESIDUAL_WIDTH, generator=generator).to(torch.bfloat16)
    shards = reference.shard_residual(residual)
    reference_blocks, reference_state = reference.read_tp4(shards)

    # Same normalization as the reference; each device's shard is already the
    # flat (branch, local hidden) row that the branch-major tiles view as.
    locals_ = [shard.unflatten(-1, (BRANCHES, LOCAL_HIDDEN)) for shard in shards]
    variance = sum(local.float().square().sum(dim=-1, keepdim=True) for local in locals_) / HIDDEN
    inverse_rms = torch.rsqrt(variance + 1e-6)
    flat_rows = [
        (local.float() * inverse_rms * (1.0 + source.device_shard(device).norm.float())).to(torch.bfloat16).flatten(-2)
        for device, local in enumerate(locals_)
    ]

    # Exact (fp64) check of the stacked layouts against the checkpoint matrices
    # on the same normalized rows: one K=2560 matmul per device, summed over
    # TP4, equals the reference's flattened linear over all 10,240 inputs.
    normalized_full = torch.cat([row.unflatten(-1, (BRANCHES, LOCAL_HIDDEN)) for row in flat_rows], dim=-1).flatten(-2)
    stacked_partial = sum(
        flat_rows[device].double()
        @ prepared["down_inject"][0, 0, device * FLAT_WIDTH : (device + 1) * FLAT_WIDTH].double()
        for device in range(TP_SIZE)
    )
    assert torch.allclose(stacked_partial[:, :RANK], F.linear(normalized_full.double(), source.down.double()))
    assert torch.allclose(
        stacked_partial[:, RANK : RANK + BRANCHES], F.linear(normalized_full.double(), source.inject.double())
    )
    assert torch.equal(stacked_partial[:, RANK + BRANCHES :], torch.zeros(1, PARTIAL_WIDTH - RANK - BRANCHES).double())
    # The whole row (SiLU of the injection tile and the zero tail included)
    # feeds the K-padded up weight; the pad rows are zero, so the result is
    # the rank-320 projection.
    low_rank = F.silu(stacked_partial / BRANCHES)
    stacked_up = torch.cat(
        [
            low_rank @ prepared["up"][0, 0, :, device * FLAT_WIDTH : (device + 1) * FLAT_WIDTH].double()
            for device in range(TP_SIZE)
        ],
        dim=-1,
    )
    expected_up = F.linear(low_rank[:, :RANK], source.up.double()).unflatten(-1, (BRANCHES, HIDDEN))
    assert torch.allclose(
        stacked_up.unflatten(-1, (TP_SIZE, BRANCHES, LOCAL_HIDDEN)).movedim(-3, -2).flatten(-2), expected_up
    )

    # The device pipeline's rounding points (FP32 partials of the quarter-scaled
    # normalized row, one BF16 cast, no scale) land within BF16 tolerance of
    # the reference read.
    row_bf16 = (stacked_partial / BRANCHES).float().to(torch.bfloat16)
    low_rank_bf16 = F.silu(row_bf16)
    injection = 2.0 * torch.sigmoid(row_bf16[:, RANK : RANK + BRANCHES])
    torch.testing.assert_close(injection.float(), reference_state.injection.float(), atol=2e-2, rtol=2e-2)
    for device in range(TP_SIZE):
        up_flat = (
            low_rank_bf16.float() @ prepared["up"][0, 0, :, device * FLAT_WIDTH : (device + 1) * FLAT_WIDTH].float()
        ).to(torch.bfloat16)
        gate = torch.sigmoid(up_flat).unflatten(-1, (BRANCHES, LOCAL_HIDDEN))
        block = (gate * (flat_rows[device] / BRANCHES).unflatten(-1, (BRANCHES, LOCAL_HIDDEN))).sum(dim=-2)
        torch.testing.assert_close(block.float(), reference_blocks[device].float(), atol=2e-2, rtol=2e-2)


def test_final_mixer_stacked_weights_match_the_gr_layout() -> None:
    generator = torch.Generator().manual_seed(4)
    source = Qwen38FinalMixerWeights(
        placement=_Placement(),
        norm=(torch.randn(RESIDUAL_WIDTH, generator=generator) * 0.05).to(torch.bfloat16),
        down=(torch.randn(RANK, RESIDUAL_WIDTH, generator=generator) * 0.05).to(torch.bfloat16),
        up=(torch.randn(RESIDUAL_WIDTH, RANK, generator=generator) * 0.05).to(torch.bfloat16),
    )
    prepared = final_mixer_module._prepare(source)

    assert tuple(prepared["down"].shape) == (1, 1, TP_SIZE * FLAT_WIDTH, RANK)
    assert tuple(prepared["up"].shape) == (1, 1, RANK, TP_SIZE * FLAT_WIDTH)
    assert torch.equal(prepared["down"][0, 0], _by_device_branch_hidden(source.down, 1).transpose(0, 1))
    assert torch.equal(prepared["up"][0, 0], _by_device_branch_hidden(source.up, 0).transpose(0, 1))
    for device in range(TP_SIZE):
        shard = source.device_shard(device)
        rows = slice(device * FLAT_WIDTH, (device + 1) * FLAT_WIDTH)
        assert torch.equal(prepared["down"][0, 0, rows], shard.down.permute(1, 2, 0).reshape(FLAT_WIDTH, RANK))
        assert torch.equal(prepared["up"][0, 0, :, rows], shard.up.permute(2, 0, 1).reshape(RANK, FLAT_WIDTH))


def test_decode_program_configs_minimize_k_blocks_on_the_eight_bank_grid() -> None:
    mesh_device = SimpleNamespace(dram_grid_size=lambda: ttnn.CoreCoord(8, 1))

    # Fused down+injection: K=2560 over five storage cores gives sixteen tiles
    # per core and the eight-tile K block, ten K blocks in all; the twelve
    # output tiles spread three per core.
    activation, program = dram_sharded_matmul_configs(mesh_device, FLAT_WIDTH, PARTIAL_WIDTH, num_cores=5)
    assert program.in0_block_w == 8
    assert program.per_core_M == 1
    assert program.per_core_N == 3
    assert program.fused_activation is None
    assert tuple(activation.shard_spec.shape) == (32, FLAT_WIDTH // 5)
    assert activation.shard_spec.grid.num_cores() == 5
    weight = dram_sharded_weight_memory_config(mesh_device, FLAT_WIDTH, PARTIAL_WIDTH)
    assert weight.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert weight.buffer_type == ttnn.BufferType.DRAM
    assert tuple(weight.shard_spec.shape) == (FLAT_WIDTH, 64)

    # Up: the K-padded row (K=384) over two storage cores gives six tiles per
    # core and the six-tile K block, one K block in all; N=2560 spreads forty
    # tiles per storage core.
    activation, program = dram_sharded_matmul_configs(mesh_device, PARTIAL_WIDTH, FLAT_WIDTH, num_cores=2)
    assert program.in0_block_w == 6
    assert program.per_core_M == 1
    assert program.per_core_N == 40
    assert tuple(activation.shard_spec.shape) == (32, PARTIAL_WIDTH // 2)
    assert activation.shard_spec.grid.num_cores() == 2
    weight = dram_sharded_weight_memory_config(mesh_device, PARTIAL_WIDTH, FLAT_WIDTH)
    assert tuple(weight.shard_spec.shape) == (PARTIAL_WIDTH, FLAT_WIDTH // 8)

    # The final mixer's down has no injection tile: ten output tiles, two per
    # storage core.
    _, program = dram_sharded_matmul_configs(mesh_device, FLAT_WIDTH, RANK, num_cores=5)
    assert (program.in0_block_w, program.per_core_N) == (8, 2)


def test_gr_module_binds_exactly_these_decode_configs() -> None:
    text = Path(gr_module.__file__).read_text(encoding="utf-8")
    assert (
        "dram_sharded_matmul_configs(\n            mesh_device, FLAT_LOCAL_WIDTH, PARTIAL_WIDTH, num_cores=5\n        )"
        in text
    )
    assert (
        "dram_sharded_matmul_configs(\n            mesh_device, PARTIAL_WIDTH, FLAT_LOCAL_WIDTH, num_cores=2\n        )"
        in text
    )
    assert "dram_sharded_weight_memory_config(mesh_device, FLAT_LOCAL_WIDTH, PARTIAL_WIDTH)" in text
    assert "dram_sharded_weight_memory_config(mesh_device, PARTIAL_WIDTH, FLAT_LOCAL_WIDTH)" in text
    assert '"down_inject_w384_dram_sharded"' in text and '"up_k384_dram_sharded"' in text
    assert '"norm_scale_q_bm"' in text
    assert gr_module.FLAT_LOCAL_WIDTH == FLAT_WIDTH and gr_module.PARTIAL_WIDTH == PARTIAL_WIDTH

    mixer_text = Path(final_mixer_module.__file__).read_text(encoding="utf-8")
    assert (
        "dram_sharded_matmul_configs(\n            mesh_device, FLAT_LOCAL_WIDTH, RESIDUAL_RANK, num_cores=5\n        )"
        in mixer_text
    )
    assert (
        "dram_sharded_matmul_configs(\n            mesh_device, RESIDUAL_RANK, FLAT_LOCAL_WIDTH, num_cores=2\n        )"
        in mixer_text
    )
    assert '"down-dram-sharded.bf16"' in mixer_text and '"up-dram-sharded.bf16"' in mixer_text
    assert '"norm-scale-q-bm.fp32"' in mixer_text
