# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for typecast with layout and memory config transforms.

Validates that ttnn.typecast transparently handles arbitrary combinations of:
  - Input/output layout (TILE <-> ROW_MAJOR)
  - Input/output memory config (interleaved <-> sharded, L1 <-> DRAM)
  - Input/output dtype (all supported typecast pairs)
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

TILE_HEIGHT = 32
TILE_WIDTH = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_torch_input(shape, pt_dtype, low=0, high=100):
    if pt_dtype in (torch.int, torch.int32):
        return torch.randint(low, high, shape, dtype=pt_dtype)
    elif pt_dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=pt_dtype)
    return (torch.rand(shape) * (high - low) + low).to(pt_dtype)


def _make_sharded_mem_config(shard_layout, tensor_shape):
    h, w = tensor_shape[-2], tensor_shape[-1]
    if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        num_cores = min(max(1, h // TILE_HEIGHT), 8)
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
        shard_h = ((h + num_cores - 1) // num_cores + TILE_HEIGHT - 1) // TILE_HEIGHT * TILE_HEIGHT
        shard_shape = [shard_h, w]
    elif shard_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        num_cores = min(max(1, w // TILE_WIDTH), 8)
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
        shard_w = ((w + num_cores - 1) // num_cores + TILE_WIDTH - 1) // TILE_WIDTH * TILE_WIDTH
        shard_shape = [h, shard_w]
    else:
        num_cores_h = max(1, min(h // TILE_HEIGHT, 2))
        num_cores_w = max(1, min(w // TILE_WIDTH, 2))
        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1))}
        )
        shard_h = ((h + num_cores_h - 1) // num_cores_h + TILE_HEIGHT - 1) // TILE_HEIGHT * TILE_HEIGHT
        shard_w = ((w + num_cores_w - 1) // num_cores_w + TILE_WIDTH - 1) // TILE_WIDTH * TILE_WIDTH
        shard_shape = [shard_h, shard_w]
    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)


def _run_typecast_and_verify(
    device,
    torch_input,
    tt_input_dtype,
    tt_output_dtype,
    input_layout,
    output_layout,
    input_mem_config,
    output_mem_config,
    pcc,
):
    """Common test body: typecast on device, compare with CPU golden."""
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=tt_input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_mem_config,
    )
    kwargs = {"dtype": tt_output_dtype, "memory_config": output_mem_config}
    if output_layout != input_layout:
        kwargs["output_layout"] = output_layout
    tt_output = ttnn.typecast(tt_input, **kwargs)

    assert tt_output.dtype == tt_output_dtype
    if output_layout != input_layout:
        assert tt_output.layout == output_layout

    cpu_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
    cpu_cast = ttnn.typecast(cpu_input, dtype=tt_output_dtype)
    assert_with_pcc(ttnn.to_torch(cpu_cast), ttnn.to_torch(tt_output), pcc=pcc)


# ---------------------------------------------------------------------------
# Dtype pair definitions
# ---------------------------------------------------------------------------
DTYPE_PAIRS_CROSS_LAYOUT = [
    (torch.bfloat16, ttnn.bfloat16, ttnn.float32, 0.99),
    (torch.float32, ttnn.float32, ttnn.bfloat16, 0.99),
    (torch.float32, ttnn.float32, ttnn.int32, 0.99),
    (torch.int, ttnn.int32, ttnn.float32, 0.99),
    (torch.bfloat16, ttnn.bfloat16, ttnn.uint16, 0.99),
    (torch.int, ttnn.uint16, ttnn.bfloat16, 0.99),
]

DTYPE_PAIRS_SAME = [
    (torch.bfloat16, ttnn.bfloat16, ttnn.bfloat16, 0.9999),
    (torch.float32, ttnn.float32, ttnn.float32, 0.9999),
    (torch.int, ttnn.int32, ttnn.int32, 0.9999),
]

BFP_TILE_TO_RM_PAIRS = [
    (torch.bfloat16, ttnn.bfloat8_b, ttnn.float32, 0.99),
    (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, 0.99),
    (torch.bfloat16, ttnn.bfloat4_b, ttnn.float32, 0.98),
]

BFP_RM_TO_TILE_PAIRS = [
    (torch.float32, ttnn.float32, ttnn.bfloat8_b, 0.99),
    (torch.bfloat16, ttnn.bfloat16, ttnn.bfloat8_b, 0.99),
]


# ---------------------------------------------------------------------------
# 1. Cross-layout transforms (TILE <-> RM) with dtype change
# ---------------------------------------------------------------------------
class TestTypecastLayoutTransforms:
    @pytest.mark.parametrize("pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc", DTYPE_PAIRS_CROSS_LAYOUT)
    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    @pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
    @pytest.mark.parametrize(
        "input_layout, output_layout",
        [(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT)],
    )
    def test_cross_layout_typecast(
        self,
        device,
        pt_input_dtype,
        tt_input_dtype,
        tt_output_dtype,
        pcc,
        input_shape,
        mem_config,
        input_layout,
        output_layout,
    ):
        torch.manual_seed(0)
        _run_typecast_and_verify(
            device,
            _make_torch_input(input_shape, pt_input_dtype),
            tt_input_dtype,
            tt_output_dtype,
            input_layout,
            output_layout,
            mem_config,
            mem_config,
            pcc,
        )


# ---------------------------------------------------------------------------
# 2. Same-dtype layout transforms
# ---------------------------------------------------------------------------
class TestTypecastSameDtypeLayoutTransforms:
    @pytest.mark.parametrize("pt_dtype, tt_dtype, _, pcc", DTYPE_PAIRS_SAME)
    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    @pytest.mark.parametrize(
        "input_layout, output_layout",
        [(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT)],
    )
    def test_same_dtype_layout_change(
        self, device, pt_dtype, tt_dtype, _, pcc, input_shape, input_layout, output_layout
    ):
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_dtype)
        tt_input = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=input_layout, device=device)
        tt_output = ttnn.typecast(tt_input, dtype=tt_dtype, output_layout=output_layout)
        assert tt_output.layout == output_layout and tt_output.dtype == tt_dtype
        assert_with_pcc(torch_input, ttnn.to_torch(tt_output), pcc=pcc)


# ---------------------------------------------------------------------------
# 3. Memory config transforms (interleaved <-> sharded, same layout)
# ---------------------------------------------------------------------------
class TestTypecastMemoryConfigTransforms:
    @pytest.mark.parametrize(
        "pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc",
        [(torch.bfloat16, ttnn.bfloat16, ttnn.float32, 0.99), (torch.float32, ttnn.float32, ttnn.int32, 0.99)],
    )
    @pytest.mark.parametrize(
        "shard_layout", [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED]
    )
    @pytest.mark.parametrize("direction", ["interleaved_to_sharded", "sharded_to_interleaved"])
    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_memory_config_change(
        self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, shard_layout, direction, input_shape
    ):
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)
        sharded_mc = _make_sharded_mem_config(shard_layout, input_shape)
        if direction == "interleaved_to_sharded":
            input_mc, output_mc = ttnn.DRAM_MEMORY_CONFIG, sharded_mc
        else:
            input_mc, output_mc = sharded_mc, ttnn.DRAM_MEMORY_CONFIG
        _run_typecast_and_verify(
            device,
            torch_input,
            tt_input_dtype,
            tt_output_dtype,
            ttnn.TILE_LAYOUT,
            ttnn.TILE_LAYOUT,
            input_mc,
            output_mc,
            pcc,
        )

    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_memory_config_change_same_dtype(self, device, input_shape):
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        sharded_mc = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # interleaved -> sharded
        tt_out = ttnn.typecast(tt_input, dtype=ttnn.bfloat16, memory_config=sharded_mc)
        assert tt_out.is_sharded()
        assert_with_pcc(torch_input, ttnn.to_torch(tt_out), pcc=0.9999)
        # sharded -> interleaved
        tt_out2 = ttnn.typecast(tt_out, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert not tt_out2.is_sharded()
        assert_with_pcc(torch_input, ttnn.to_torch(tt_out2), pcc=0.9999)


# ---------------------------------------------------------------------------
# 4. Combined: layout + memory + dtype (all 4 direction combos)
# ---------------------------------------------------------------------------
COMBINED_DTYPE_PAIRS = [
    (torch.bfloat16, ttnn.bfloat16, ttnn.float32, 0.99),
    (torch.float32, ttnn.float32, ttnn.bfloat16, 0.99),
    (torch.float32, ttnn.float32, ttnn.int32, 0.99),
    (torch.int, ttnn.int32, ttnn.float32, 0.99),
]


class TestTypecastCombinedTransforms:
    @pytest.mark.parametrize("pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc", COMBINED_DTYPE_PAIRS)
    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    @pytest.mark.parametrize(
        "input_layout, output_layout, input_sharded, output_sharded",
        [
            (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, False, True),
            (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, True, False),
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, False, True),
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, True, False),
        ],
    )
    def test_combined_transform(
        self,
        device,
        pt_input_dtype,
        tt_input_dtype,
        tt_output_dtype,
        pcc,
        input_shape,
        input_layout,
        output_layout,
        input_sharded,
        output_sharded,
    ):
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)
        sharded_mc = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        input_mc = sharded_mc if input_sharded else ttnn.DRAM_MEMORY_CONFIG
        output_mc = sharded_mc if output_sharded else ttnn.DRAM_MEMORY_CONFIG
        _run_typecast_and_verify(
            device,
            torch_input,
            tt_input_dtype,
            tt_output_dtype,
            input_layout,
            output_layout,
            input_mc,
            output_mc,
            pcc,
        )

    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    @pytest.mark.parametrize(
        "input_layout, output_layout, input_sharded, output_sharded",
        [
            (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, True, False),
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, False, True),
        ],
    )
    def test_combined_same_dtype(self, device, input_shape, input_layout, output_layout, input_sharded, output_sharded):
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        sharded_mc = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        input_mc = sharded_mc if input_sharded else ttnn.DRAM_MEMORY_CONFIG
        output_mc = sharded_mc if output_sharded else ttnn.DRAM_MEMORY_CONFIG
        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=input_mc
        )
        tt_output = ttnn.typecast(tt_input, dtype=ttnn.bfloat16, memory_config=output_mc, output_layout=output_layout)
        assert tt_output.layout == output_layout
        assert_with_pcc(torch_input, ttnn.to_torch(tt_output), pcc=0.9999)


# ---------------------------------------------------------------------------
# 5. Regression: existing same-layout behavior preserved
# ---------------------------------------------------------------------------
class TestTypecastRegressionSameLayout:
    @pytest.mark.parametrize(
        "pt_input_dtype, tt_input_dtype, tt_output_dtype",
        [(torch.bfloat16, ttnn.bfloat16, ttnn.float32), (torch.float32, ttnn.float32, ttnn.int32)],
    )
    @pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
    @pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
    def test_no_layout_change(self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, layout, mem_config):
        torch.manual_seed(0)
        torch_input = _make_torch_input([1, 1, 32, 32], pt_input_dtype)
        _run_typecast_and_verify(
            device,
            torch_input,
            tt_input_dtype,
            tt_output_dtype,
            layout,
            layout,
            mem_config,
            mem_config,
            0.99,
        )


# ---------------------------------------------------------------------------
# 6. Host tensor layout change
# ---------------------------------------------------------------------------
class TestTypecastHostTensorLayoutTransform:
    @pytest.mark.parametrize(
        "input_layout, output_layout",
        [(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT)],
    )
    def test_host_layout_change(self, input_layout, output_layout):
        torch.manual_seed(0)
        cpu_input = ttnn.from_torch(
            torch.randn([1, 1, 32, 32], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=input_layout
        )
        cpu_output = ttnn.typecast(cpu_input, dtype=ttnn.float32, output_layout=output_layout)
        assert cpu_output.layout == output_layout and cpu_output.dtype == ttnn.float32


# ---------------------------------------------------------------------------
# 7. BFP cross-layout (tile-only input/output types)
# ---------------------------------------------------------------------------
class TestTypecastBfpLayoutTransform:
    @pytest.mark.parametrize("pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc", BFP_TILE_TO_RM_PAIRS)
    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    def test_bfp_tile_to_rm(self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, input_shape):
        torch.manual_seed(0)
        _run_typecast_and_verify(
            device,
            _make_torch_input(input_shape, pt_input_dtype),
            tt_input_dtype,
            tt_output_dtype,
            ttnn.TILE_LAYOUT,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            pcc,
        )

    @pytest.mark.parametrize("pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc", BFP_RM_TO_TILE_PAIRS)
    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    def test_rm_to_bfp_tile(self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, input_shape):
        torch.manual_seed(0)
        _run_typecast_and_verify(
            device,
            _make_torch_input(input_shape, pt_input_dtype),
            tt_input_dtype,
            tt_output_dtype,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.TILE_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            pcc,
        )


# ---------------------------------------------------------------------------
# 8. Reshard: different sharding strategies
# ---------------------------------------------------------------------------
class TestTypecastReshardTransform:
    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_height_sharded_to_width_sharded(self, device, input_shape):
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        input_mc = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        output_mc = _make_sharded_mem_config(ttnn.TensorMemoryLayout.WIDTH_SHARDED, input_shape)
        _run_typecast_and_verify(
            device,
            torch_input,
            ttnn.bfloat16,
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            ttnn.TILE_LAYOUT,
            input_mc,
            output_mc,
            0.99,
        )


# ---------------------------------------------------------------------------
# 9. Negative: BFP output in ROW_MAJOR must be rejected
# ---------------------------------------------------------------------------
class TestTypecastNegativeCases:
    @pytest.mark.parametrize("bfp_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
    @pytest.mark.parametrize(
        "pt_input_dtype, tt_input_dtype, input_layout",
        [
            (torch.bfloat16, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            (torch.float32, ttnn.float32, ttnn.TILE_LAYOUT),
            (torch.int, ttnn.int32, ttnn.TILE_LAYOUT),
            (torch.bfloat16, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
            (torch.bfloat16, ttnn.bfloat4_b, ttnn.TILE_LAYOUT),
            (torch.float32, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        ],
    )
    def test_bfp_rm_output_rejected(self, device, bfp_dtype, pt_input_dtype, tt_input_dtype, input_layout):
        """Any typecast producing BFP output in RM layout must be rejected."""
        tt_input = ttnn.from_torch(
            _make_torch_input([1, 1, 32, 32], pt_input_dtype),
            dtype=tt_input_dtype,
            layout=input_layout,
            device=device,
        )
        output_layout = ttnn.ROW_MAJOR_LAYOUT if input_layout == ttnn.TILE_LAYOUT else None
        with pytest.raises(RuntimeError):
            if output_layout is not None:
                ttnn.typecast(tt_input, dtype=bfp_dtype, output_layout=output_layout)
            else:
                ttnn.typecast(tt_input, dtype=bfp_dtype)
