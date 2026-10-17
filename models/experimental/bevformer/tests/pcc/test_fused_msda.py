# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""BEVFormer's MSDA module against the PyTorch reference.

`TTMSDeformableAttention` computes its core attention with a single
`ttnn.experimental.fused_msda_from_offsets`. The sampling position is formed on
the SFPU from the reference point and the raw offset.

Same weights, same inputs, compared against the reference module rather than
against another TT implementation.

Complements `test_ms_deformable_attention.py`, which checks the same module with
the repo's usual tolerance bundle; here the gate is a tighter PCC across several
seeds, because a location-generation bug shows up as a seed-dependent drift
rather than a uniform offset.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.bevformer.config import DeformableAttentionConfig
from models.experimental.bevformer.config.encoder_config import get_preset_config
from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention
from models.experimental.bevformer.tests.test_utils import check_with_pcc
from models.experimental.bevformer.tt.model_preprocessing import create_ms_deformable_attention_parameters
from models.experimental.bevformer.tt.tt_ms_deformable_attention import (
    TTMSDeformableAttention,
    multi_scale_deformable_attn_fused,
)

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device({"l1_small_size": 10 * 1024})


def _build(device, config_name, batch_size, num_queries, seed):
    """Shared setup: reference module, preprocessed TT params, and the inputs."""
    torch.manual_seed(seed)

    preset_config = get_preset_config(config_name)
    if preset_config is None:
        pytest.fail(f"Configuration '{config_name}' not found")

    model_config = preset_config.model_config
    num_levels = model_config.num_levels
    spatial_shapes = torch.tensor(preset_config.dataset_config.spatial_shapes[:num_levels], dtype=torch.long)
    num_keys = int(spatial_shapes.prod(dim=1).sum().item())

    config = DeformableAttentionConfig(
        embed_dims=model_config.embed_dims,
        num_heads=model_config.num_heads,
        num_levels=num_levels,
        num_points=model_config.num_points,
    )

    query = torch.randn(batch_size, num_queries, config.embed_dims, dtype=torch.float32)
    value = torch.randn(batch_size, num_keys, config.embed_dims, dtype=torch.float32)
    reference_points = torch.rand(batch_size, num_queries, num_levels, 2, dtype=torch.float32)

    ref_model = MSDeformableAttention(config)
    ref_model.eval()

    tt_parameters = create_ms_deformable_attention_parameters(
        torch_model=ref_model, device=device, config=config, dtype=ttnn.float32
    )

    torch_inputs = (query, value, reference_points, spatial_shapes)
    return config, ref_model, tt_parameters, torch_inputs


def _record_value_placement(monkeypatch):
    """Record the buffer type of each value the fused op actually receives."""
    placements = []
    fused_op = ttnn.experimental.fused_msda_from_offsets

    def spy(value, *args, **kwargs):
        placements.append(value.memory_config().buffer_type)
        return fused_op(value, *args, **kwargs)

    monkeypatch.setattr(ttnn.experimental, "fused_msda_from_offsets", spy)
    return placements


def _run_tt(device, config, tt_parameters, torch_inputs):
    query, value, reference_points, spatial_shapes = torch_inputs

    tt_query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_value = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_reference_points = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_model = TTMSDeformableAttention(
        config=config,
        device=device,
        params=tt_parameters,
        spatial_shapes=spatial_shapes,
    )
    out = tt_model(query=tt_query, value=tt_value, reference_points=tt_reference_points)
    return ttnn.to_torch(out).to(torch.float32)


@pytest.mark.parametrize(
    "config_name, batch_size, num_queries",
    [
        ("nuscenes_tiny", 1, 900),
        ("nuscenes_base", 1, 2500),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_msda_matches_reference(device, config_name, batch_size, num_queries, seed):
    config, ref_model, tt_parameters, torch_inputs = _build(device, config_name, batch_size, num_queries, seed)
    query, value, reference_points, spatial_shapes = torch_inputs

    expected = ref_model(query, value, reference_points=reference_points, spatial_shapes=spatial_shapes)
    got = _run_tt(device, config, tt_parameters, torch_inputs)

    assert got.shape == expected.shape, f"{tuple(got.shape)} != {tuple(expected.shape)}"
    logger.info(f"{config_name} q={num_queries} seed={seed}: output {tuple(got.shape)}")

    passed, message = check_with_pcc(expected, got, pcc=0.999)
    assert passed, f"msda_output[{config_name}, seed={seed}]: {message}"


def test_msda_output_is_deterministic(device):
    """Two identical calls must agree exactly.

    The op distributes (batch, head, query-block) work units across cores and
    accumulates each unit's `levels * points` reduction locally, so the summation
    order is fixed and the result should not vary run to run. A mismatch here
    would point at uninitialized L1 being read -- the reader explicitly zeroes
    the input rows whose bilinear corner it did not gather, so a nonzero
    difference here points straight at that zeroing.
    """
    config, _, tt_parameters, torch_inputs = _build(device, "nuscenes_tiny", 1, 900, 0)

    first = _run_tt(device, config, tt_parameters, torch_inputs)
    second = _run_tt(device, config, tt_parameters, torch_inputs)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("value_memory, buffer_type", [("l1", ttnn.BufferType.L1), ("dram", ttnn.BufferType.DRAM)])
def test_msda_value_memory_modes_match_reference(device, monkeypatch, value_memory, buffer_type):
    config, ref_model, tt_parameters, torch_inputs = _build(device, "nuscenes_tiny", 1, 64, 0)
    config.value_memory = value_memory
    query, value, reference_points, spatial_shapes = torch_inputs
    placements = _record_value_placement(monkeypatch)

    expected = ref_model(query, value, reference_points=reference_points, spatial_shapes=spatial_shapes)
    got = _run_tt(device, config, tt_parameters, torch_inputs)

    assert placements == [buffer_type]
    passed, message = check_with_pcc(expected, got, pcc=0.999)
    assert passed, f"msda_output[value_memory={value_memory}]: {message}"


@pytest.mark.parametrize("batch_size", [1, 6], ids=["tsa_like", "sca_six_cams"])
def test_msda_auto_keeps_value_in_l1_at_nuscenes_base(device, monkeypatch, batch_size):
    """The production default must stay on the L1 path at nuscenes_base.

    DRAM is about 3x slower for this gather, and ``auto`` falls back silently
    apart from a log line, so an over-conservative fit check would only show
    up as lost performance. Six batch rows match SCA, which folds the cameras
    into the batch.
    """
    config, _, tt_parameters, torch_inputs = _build(device, "nuscenes_base", batch_size, 2500, 0)
    placements = _record_value_placement(monkeypatch)

    _run_tt(device, config, tt_parameters, torch_inputs)

    assert placements == [ttnn.BufferType.L1]


def test_msda_auto_falls_back_to_dram_under_l1_pressure(device, expect_error):
    spatial_shapes = torch.tensor([(29, 50), (15, 25), (8, 13), (4, 7)], dtype=torch.long)
    batch_size, num_queries, num_heads, num_points, embed_dims = 6, 128, 8, 4, 256
    num_keys = int(spatial_shapes.prod(dim=1).sum().item())

    torch.manual_seed(0)
    value = ttnn.from_torch(
        torch.randn(batch_size, num_keys, embed_dims, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    reference_points = ttnn.from_torch(
        torch.rand(batch_size, num_queries, 4, 2, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    sampling_offsets = ttnn.from_torch(
        torch.randn(
            batch_size,
            num_queries,
            num_heads,
            len(spatial_shapes) * num_points * 2,
            dtype=torch.bfloat16,
        ),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    attention_weights = ttnn.from_torch(
        torch.softmax(
            torch.randn(
                batch_size,
                num_queries,
                num_heads,
                len(spatial_shapes) * num_points,
                dtype=torch.float32,
            ),
            dim=-1,
        ).to(torch.bfloat16),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    expected = multi_scale_deformable_attn_fused(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes,
        value_memory="dram",
    )
    expected_torch = ttnn.to_torch(expected)

    assert ttnn.experimental.can_use_fused_msda_from_offsets_l1_value(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes.tolist(),
        reference_mode="pillar",
    )

    l1_view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    tile_bytes = 2048
    # The fused program needs about 120 KiB/core for CBs. This leaves those CBs
    # room to run from DRAM while making its additional L1 value and output unsafe.
    target_free_per_bank = 176 * 1024
    blocker_tiles_per_bank = (l1_view.largest_contiguous_bytes_free_per_bank - target_free_per_bank) // tile_bytes
    assert blocker_tiles_per_bank > 0
    blocker = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32 * blocker_tiles_per_bank, 32 * l1_view.num_banks]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )

    def allocated_per_bank():
        return tuple(
            ttnn.get_memory_view(device, buffer_type).total_bytes_allocated_per_bank
            for buffer_type in (ttnn.BufferType.L1, ttnn.BufferType.DRAM)
        )

    try:
        allocated_before_fallback = allocated_per_bank()
        assert not ttnn.experimental.can_use_fused_msda_from_offsets_l1_value(
            value,
            reference_points,
            sampling_offsets,
            attention_weights,
            spatial_shapes.tolist(),
            reference_mode="pillar",
        )
        with expect_error(RuntimeError, "value_memory='l1' is unsafe"):
            multi_scale_deformable_attn_fused(
                value,
                reference_points,
                sampling_offsets,
                attention_weights,
                spatial_shapes,
                value_memory="l1",
            )
        actual = multi_scale_deformable_attn_fused(
            value,
            reference_points,
            sampling_offsets,
            attention_weights,
            spatial_shapes,
            value_memory="auto",
        )
        assert actual.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        torch.testing.assert_close(ttnn.to_torch(actual), expected_torch, rtol=0, atol=0)

        repeated = multi_scale_deformable_attn_fused(
            value,
            reference_points,
            sampling_offsets,
            attention_weights,
            spatial_shapes,
            value_memory="auto",
        )
        assert repeated.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        torch.testing.assert_close(ttnn.to_torch(repeated), expected_torch, rtol=0, atol=0)
        # Only the outputs may outlive the calls; the staged DRAM value must not.
        ttnn.deallocate(actual)
        ttnn.deallocate(repeated)
        assert allocated_per_bank() == allocated_before_fallback
    finally:
        ttnn.deallocate(blocker)
