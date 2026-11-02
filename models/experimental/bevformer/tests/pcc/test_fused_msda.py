# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""BEVFormer's MSDA module against the PyTorch reference.

`TTMSDeformableAttention` computes its core attention with a single
`ttnn.experimental.fused_msda_from_offsets`. The sampling position is formed on
the SFPU from the reference point and the raw offset.

The numerics tests compare against the reference module, same weights and same
inputs. The placement tests check where `multi_scale_deformable_attn_fused`
stages value, and that the DRAM fallback is bit-identical to forced DRAM.

Complements `test_ms_deformable_attention.py`, which checks the same module with
the repo's usual tolerance bundle; here the gate is a tighter PCC across several
seeds, because a location-generation bug shows up as a seed-dependent drift
rather than a uniform offset.
"""

from functools import partial

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.bevformer.model_config import DeformableAttentionConfig
from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention
from models.experimental.bevformer.tests.common import PRESETS, assert_pcc
from models.experimental.bevformer.tt.model_preprocessing import create_ms_deformable_attention_parameters
from models.experimental.bevformer.tt import tt_ms_deformable_attention
from models.experimental.bevformer.tt.tt_ms_deformable_attention import (
    TTMSDeformableAttention,
    multi_scale_deformable_attn_fused,
)

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device({"l1_small_size": 10 * 1024})


def _build(device, config_name, batch_size, num_queries, seed):
    """Shared setup: reference module, preprocessed TT params, and the inputs."""
    torch.manual_seed(seed)

    model_config = PRESETS[config_name]
    num_levels = model_config.num_levels
    spatial_shapes = torch.tensor(model_config.spatial_shapes, dtype=torch.long)
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

    tt_parameters = create_ms_deformable_attention_parameters(ref_model, device=device, dtype=ttnn.float32)

    torch_inputs = (query, value, reference_points, spatial_shapes)
    return config, ref_model, tt_parameters, torch_inputs


def _record_value_placement(monkeypatch):
    """Record the buffer type of each value the fused op receives.

    Placement is decided and staged inside `multi_scale_deformable_attn_fused`,
    so the op's input is the only place to observe it.
    """
    placements = []
    fused_op = ttnn.experimental.fused_msda_from_offsets

    def spy(value, *args, **kwargs):
        placements.append(value.memory_config().buffer_type)
        return fused_op(value, *args, **kwargs)

    monkeypatch.setattr(ttnn.experimental, "fused_msda_from_offsets", spy)
    return placements


def _force_value_memory(monkeypatch, value_memory):
    """Pin the module's placement decision, which it otherwise makes from live L1 state.

    `forward` looks the function up in the module's globals at call time, so the
    module attribute is what has to be patched.
    """
    monkeypatch.setattr(
        tt_ms_deformable_attention,
        "multi_scale_deformable_attn_fused",
        partial(multi_scale_deformable_attn_fused, value_memory=value_memory),
    )


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

    passed, message = assert_pcc(expected, got, pcc=0.999)
    assert passed, f"msda_output[{config_name}, seed={seed}]: {message}"


def test_msda_output_is_deterministic(device):
    """Two identical calls must agree exactly.

    The op distributes (batch, head, query-block) work units across cores and
    accumulates each unit's `levels * points` reduction locally, so the summation
    order is fixed and the result should not vary run to run. A mismatch points
    at uninitialized L1 being read: a corner the gathers skip keeps whatever its
    input slot held, so the one-time zeroing of the input CB and the zero weight
    compute gives that corner are what keep stale bytes out of the result.
    """
    config, _, tt_parameters, torch_inputs = _build(device, "nuscenes_tiny", 1, 900, 0)

    first = _run_tt(device, config, tt_parameters, torch_inputs)
    second = _run_tt(device, config, tt_parameters, torch_inputs)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("value_memory, buffer_type", [("l1", ttnn.BufferType.L1), ("dram", ttnn.BufferType.DRAM)])
def test_msda_value_memory_modes_match_reference(device, monkeypatch, value_memory, buffer_type):
    config, ref_model, tt_parameters, torch_inputs = _build(device, "nuscenes_tiny", 1, 64, 0)
    query, value, reference_points, spatial_shapes = torch_inputs
    placements = _record_value_placement(monkeypatch)

    expected = ref_model(query, value, reference_points=reference_points, spatial_shapes=spatial_shapes)
    _force_value_memory(monkeypatch, value_memory)
    got = _run_tt(device, config, tt_parameters, torch_inputs)

    assert placements == [buffer_type]
    passed, message = assert_pcc(expected, got, pcc=0.999)
    assert passed, f"msda_output[value_memory={value_memory}]: {message}"


@pytest.mark.parametrize("batch_size", [1, 6], ids=["tsa_like", "sca_six_cams"])
def test_msda_auto_keeps_value_in_l1_on_large_blackhole(device, monkeypatch, batch_size):
    """The production default must stay on L1 for large-grid Blackhole.

    DRAM is about 3x slower for this gather (re-measure with
    ``test_fused_msda_perf.py::test_fused_msda_module_perf``), and ``auto``
    falls back with only a one-time warning, so an over-conservative fit check
    would only show up as lost performance. Six batch rows match SCA, which folds the cameras
    into the batch. The capability gate stays independent of the fit query under test.
    """
    if not ttnn.device.is_blackhole(device):
        pytest.skip("strict fused-MSDA L1 placement requires Blackhole")

    l1_num_banks = ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks
    if l1_num_banks < 90:
        pytest.skip(f"strict fused-MSDA L1 placement requires 90 L1 banks; found {l1_num_banks}")

    config, _, tt_parameters, torch_inputs = _build(device, "nuscenes_base", batch_size, 2500, 0)
    placements = _record_value_placement(monkeypatch)

    _run_tt(device, config, tt_parameters, torch_inputs)

    assert placements == [ttnn.BufferType.L1]


# SCA-shaped fallback case: six cameras folded into the batch, value_proj's tiled packed value.
FALLBACK_NUM_HEADS = 8


def _fallback_case(device):
    """(value, reference_points, sampling_offsets, attention_weights, spatial_shapes) on device."""
    spatial_shapes = torch.tensor([(29, 50), (15, 25), (8, 13), (4, 7)], dtype=torch.long)
    batch_size, num_queries, num_heads, num_points, embed_dims = 6, 128, FALLBACK_NUM_HEADS, 4, 256
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
    return value, reference_points, sampling_offsets, attention_weights, spatial_shapes


def _blocked_l1(device, free_per_bank):
    """An L1 tensor that leaves about ``free_per_bank`` bytes of the largest free block per bank."""
    l1_view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    tile_bytes = 2048
    blocker_tiles_per_bank = (l1_view.largest_contiguous_bytes_free_per_bank - free_per_bank) // tile_bytes
    assert blocker_tiles_per_bank > 0
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32 * blocker_tiles_per_bank, 32 * l1_view.num_banks]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )


def test_msda_auto_falls_back_to_dram(device, monkeypatch, expect_error):
    """Cover both a conservative capacity fallback and a post-check L1 OOM."""
    value, reference_points, sampling_offsets, attention_weights, spatial_shapes = _fallback_case(device)

    expected = multi_scale_deformable_attn_fused(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes,
        value_memory="dram",
    )
    expected_torch = ttnn.to_torch(expected)
    ttnn.deallocate(expected)

    # Without the blocker this case fits L1, so the blocker alone flips the decision.
    assert ttnn.experimental.can_use_fused_msda_from_offsets_l1_value(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes.tolist(),
        reference_mode="pillar",
    )

    fused_op = ttnn.experimental.fused_msda_from_offsets
    to_layout = ttnn.to_layout
    attempted_memory_configs = []
    staged_values = []
    l1_error_message = ["Out of Memory: injected L1 output allocation failure"]

    def record_staged_value(tensor, *args, **kwargs):
        staged = to_layout(tensor, *args, **kwargs)
        memory_config = kwargs.get("memory_config")
        if tensor is value and memory_config is not None:
            staged_values.append((memory_config, staged))
        return staged

    def fail_l1_output_allocation(value, *args, **kwargs):
        memory_config = kwargs["memory_config"]
        attempted_memory_configs.append(memory_config)
        if memory_config == ttnn.L1_MEMORY_CONFIG:
            raise RuntimeError(l1_error_message[0])
        return fused_op(value, *args, **kwargs)

    with monkeypatch.context() as oom_patch:
        oom_patch.setattr(ttnn.experimental, "can_use_fused_msda_from_offsets_l1_value", lambda *args, **kwargs: True)
        oom_patch.setattr(ttnn, "to_layout", record_staged_value)
        oom_patch.setattr(ttnn.experimental, "fused_msda_from_offsets", fail_l1_output_allocation)

        actual = multi_scale_deformable_attn_fused(
            value,
            reference_points,
            sampling_offsets,
            attention_weights,
            spatial_shapes,
            value_memory="auto",
        )
        assert attempted_memory_configs == [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG]
        assert [memory_config for memory_config, _ in staged_values] == [
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ]
        assert all(not staged.is_allocated() for _, staged in staged_values)
        assert actual.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        torch.testing.assert_close(ttnn.to_torch(actual), expected_torch, rtol=0, atol=0)
        ttnn.deallocate(actual)

        attempted_memory_configs.clear()
        staged_values.clear()
        with expect_error(RuntimeError, "injected L1 output allocation failure"):
            multi_scale_deformable_attn_fused(
                value,
                reference_points,
                sampling_offsets,
                attention_weights,
                spatial_shapes,
                value_memory="l1",
            )
        assert attempted_memory_configs == [ttnn.L1_MEMORY_CONFIG]
        assert len(staged_values) == 1
        assert not staged_values[0][1].is_allocated()

        l1_error_message[0] = "injected kernel failure"
        attempted_memory_configs.clear()
        staged_values.clear()
        with expect_error(RuntimeError, "injected kernel failure"):
            multi_scale_deformable_attn_fused(
                value,
                reference_points,
                sampling_offsets,
                attention_weights,
                spatial_shapes,
                value_memory="auto",
            )
        assert attempted_memory_configs == [ttnn.L1_MEMORY_CONFIG]
        assert len(staged_values) == 1
        assert not staged_values[0][1].is_allocated()

    # The fused program's CBs need ~121 KiB/core here. With 176 KiB free the 95%
    # program-L1 margin leaves ~167 KiB: enough for the CBs alone (value in DRAM),
    # not once the 6.4 MB of L1 value and output are spread over the banks
    # (~59 KiB/bank at 110 banks, more with fewer), so the check must say unsafe.
    blocker = _blocked_l1(device, 176 * 1024)

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

        # Only the output may outlive the call; the staged DRAM value must not.
        ttnn.deallocate(actual)
        assert allocated_per_bank() == allocated_before_fallback
    finally:
        ttnn.deallocate(blocker)


def _round_up(n, multiple):
    return -(-n // multiple) * multiple


def _l1_bytes_per_bank(device, num_pages, page_bytes):
    """What the fit check charges a ROW_MAJOR interleaved L1 tensor per bank: whole pages per
    bank at L1 alignment, rounded up to the allocator's DRAM-alignment granule."""
    num_banks = ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks
    granule = ttnn.get_dram_alignment()
    per_bank = -(-num_pages // num_banks) * _round_up(page_bytes, ttnn.get_l1_alignment())
    return _round_up(max(per_bank, granule), granule)


def test_msda_l1_check_charges_head_major_staging(device):
    """With value_layout="head_major" the fit check charges what the producer allocates.

    The producer's output pads each head's keys to a multiple of 32 in 32*D*2-byte pages, so
    per bank it differs from the packed (B, S, H*D) value the "auto" check charges. Free L1 is
    set in 64-byte steps with a ROW_MAJOR blocker, and for each layout the smallest free size
    the check admits is found by bisection. Both checks charge the same output and circular
    buffers against 95% of the free span, so the two thresholds must differ by the value's
    charge difference / 0.95. At the head-major threshold the real call must then run in L1.
    """
    value, reference_points, sampling_offsets, attention_weights, spatial_shapes = _fallback_case(device)
    batch, num_keys, channels = value.shape
    head_dim = channels // FALLBACK_NUM_HEADS
    packed_bytes = _l1_bytes_per_bank(device, batch * num_keys, channels * 2)
    head_major_bytes = _l1_bytes_per_bank(device, batch * FALLBACK_NUM_HEADS * -(-num_keys // 32), 32 * head_dim * 2)
    charge_gap = head_major_bytes - packed_bytes
    assert charge_gap > 0, (head_major_bytes, packed_bytes)

    l1_view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    num_banks = l1_view.num_banks
    unblocked_free = l1_view.largest_contiguous_bytes_free_per_bank
    step = 64

    def fits_with_free(free_per_bank, value_layout):
        pages_per_bank = (unblocked_free - free_per_bank) // step
        # One 64-byte page per row, pages_per_bank of them in every bank.
        blocker = ttnn.allocate_tensor_on_device(
            ttnn.Shape([pages_per_bank * num_banks, step // 2]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG,
        )
        try:
            return ttnn.experimental.can_use_fused_msda_from_offsets_l1_value(
                value,
                reference_points,
                sampling_offsets,
                attention_weights,
                spatial_shapes.tolist(),
                reference_mode="pillar",
                value_layout=value_layout,
            )
        finally:
            ttnn.deallocate(blocker)

    def admitted_from(value_layout):
        lo, hi = 64 * 1024 // step, 384 * 1024 // step
        assert not fits_with_free(lo * step, value_layout) and fits_with_free(hi * step, value_layout)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fits_with_free(mid * step, value_layout):
                hi = mid
            else:
                lo = mid
        return hi * step

    auto_threshold = admitted_from("auto")
    head_major_threshold = admitted_from("head_major")
    measured_gap = head_major_threshold - auto_threshold
    expected_gap = charge_gap * 100 / 95
    assert abs(measured_gap - expected_gap) <= 2 * step, (measured_gap, expected_gap, charge_gap)

    pages_per_bank = (unblocked_free - head_major_threshold) // step
    blocker = ttnn.allocate_tensor_on_device(
        ttnn.Shape([pages_per_bank * num_banks, step // 2]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    try:
        output = multi_scale_deformable_attn_fused(
            value,
            reference_points,
            sampling_offsets,
            attention_weights,
            spatial_shapes,
            value_layout="head_major",
            value_memory="l1",
        )
        assert output.memory_config() == ttnn.L1_MEMORY_CONFIG
        ttnn.deallocate(output)
    finally:
        ttnn.deallocate(blocker)


@pytest.mark.parametrize(
    "l1_failure",
    [
        "Out of Memory: injected L1 output allocation failure",
        "Statically allocated circular buffers in program 1 clash with L1 buffers on core range [injected]",
    ],
    ids=["oom", "cb_clash"],
)
def test_msda_auto_falls_back_to_dram_through_head_major_producer(device, monkeypatch, l1_failure):
    """The head-major path retries in DRAM after an L1 failure, frees what it staged, and matches packed."""
    value, reference_points, sampling_offsets, attention_weights, spatial_shapes = _fallback_case(device)
    call = partial(
        multi_scale_deformable_attn_fused, value, reference_points, sampling_offsets, attention_weights, spatial_shapes
    )

    expected = call(value_memory="dram")
    expected_torch = ttnn.to_torch(expected)
    ttnn.deallocate(expected)

    fused_op = ttnn.experimental.fused_msda_from_offsets
    producer = ttnn.experimental.fused_msda_head_major_value
    attempted = []
    staged_values = []

    def record_producer(tensor, num_heads, **kwargs):
        staged = producer(tensor, num_heads, **kwargs)
        staged_values.append((kwargs["memory_config"], staged))
        return staged

    def fail_in_l1(staged, *args, **kwargs):
        attempted.append((kwargs["memory_config"], kwargs["value_layout"]))
        if kwargs["memory_config"] == ttnn.L1_MEMORY_CONFIG:
            raise RuntimeError(l1_failure)
        return fused_op(staged, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ttnn.experimental, "can_use_fused_msda_from_offsets_l1_value", lambda *args, **kwargs: True)
        patch.setattr(ttnn.experimental, "fused_msda_head_major_value", record_producer)
        patch.setattr(ttnn.experimental, "fused_msda_from_offsets", fail_in_l1)
        actual = call(value_layout="head_major", value_memory="auto")

    assert attempted == [(ttnn.L1_MEMORY_CONFIG, "head_major"), (ttnn.DRAM_MEMORY_CONFIG, "head_major")]
    assert [memory_config for memory_config, _ in staged_values] == [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG]
    assert all(not staged.is_allocated() for _, staged in staged_values)
    assert actual.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    torch.testing.assert_close(ttnn.to_torch(actual), expected_torch, rtol=0, atol=0)
    ttnn.deallocate(actual)


def test_msda_head_major_value_follows_value_memory(device, monkeypatch):
    """A value already head-major is moved to the memory value_memory selects, like any other value."""
    value, reference_points, sampling_offsets, attention_weights, spatial_shapes = _fallback_case(device)
    head_major = ttnn.experimental.fused_msda_head_major_value(
        value, FALLBACK_NUM_HEADS, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    placements = []
    fused_op = ttnn.experimental.fused_msda_from_offsets

    def spy(staged, *args, **kwargs):
        placements.append(staged.memory_config().buffer_type)
        return fused_op(staged, *args, **kwargs)

    monkeypatch.setattr(ttnn.experimental, "fused_msda_from_offsets", spy)
    output = multi_scale_deformable_attn_fused(
        head_major,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes,
        value_layout="head_major",
        value_memory="dram",
    )
    assert placements == [ttnn.BufferType.DRAM]
    assert head_major.is_allocated() and head_major.memory_config() == ttnn.L1_MEMORY_CONFIG
    assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    ttnn.deallocate(output)
    ttnn.deallocate(head_major)
