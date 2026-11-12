# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The GR rows forms (1..32 lanes, the 32- and 128-row chunks) against the 1-row read/write, on a torch-backed ttnn.

The GR fake is the rows fake plus the GR ops the 1-row path uses. Every op is per row or per element, so
``read_rows``/``write_rows`` must equal the 1-row ``read``/``write`` calls row for row (at 128 rows: the four 32-row
calls tile for tile) bitwise.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    BF16,
    FP32,
    TILE,
    TP,
    FakeContract,
    FakeTensor,
    assert_rows_bitwise,
    bf16_randn,
    function_calls,
    make_gr_ttnn,
    use_tile_padding,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as decode_matmul_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS, LONG_CHUNK_ROWS
from models.demos.blackhole.qwen38_flash_next.ttnn.gr import (
    BLOCK_ROWS_LOCAL_SHAPE,
    RESIDUAL_ROWS_LOCAL_SHAPE,
    Qwen38TTNNGatedResidual,
)


@pytest.fixture
def gr_fake(monkeypatch):
    use_tile_padding(monkeypatch)
    fake_ttnn = make_gr_ttnn()
    monkeypatch.setattr(gr_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(decode_matmul_module, "ttnn", fake_ttnn)
    return fake_ttnn


def test_gr_rows_shapes_follow_the_row_count_and_keep_the_chunk_constants(expect_error) -> None:
    assert RESIDUAL_ROWS_LOCAL_SHAPE == (1, 4, 32, 640) and BLOCK_ROWS_LOCAL_SHAPE == (1, 1, 32, 640)
    chunk = gr_module.gr_rows_shapes(32)
    assert chunk["residual"] == RESIDUAL_ROWS_LOCAL_SHAPE and chunk["flat"] == gr_module.FLAT_ROWS_LOCAL_SHAPE
    assert chunk["block"] == BLOCK_ROWS_LOCAL_SHAPE and chunk["injection"] == gr_module.INJECTION_ROWS_SHAPE
    assert chunk["partial"] == gr_module.PARTIAL_REDUCTION_ROWS_SHAPE and chunk["flat_padded"] == chunk["flat"]
    eight = gr_module.gr_rows_shapes((1, 4, 8, 640))
    assert eight["rows"] == 8 and eight["residual"] == (1, 4, 8, 640) and eight["token_major"] == (1, 8, 4, 640)
    assert eight["flat"] == (1, 1, 8, gr_module.FLAT_LOCAL_WIDTH)
    assert eight["flat_padded"] == (1, 1, 32, gr_module.FLAT_LOCAL_WIDTH)
    assert eight["partial"] == (1, 1, 8, gr_module.PARTIAL_WIDTH)
    assert eight["partial_padded"] == (1, 1, 32, gr_module.PARTIAL_WIDTH)
    with expect_error(ValueError, match=r"GR block rows must be in \[1,32\], got 0"):  # allow-pytest.raises: contract
        gr_module.gr_rows_shapes((1, 1, 0, 640), label="GR block rows")
    with expect_error(ValueError, match=r"must be rank 4 .*, got \[1, 4, 640\]"):  # allow-pytest.raises: contract
        gr_module.gr_rows_shapes((1, 4, 640))


@pytest.mark.parametrize(
    "rows,flat_views",
    [(1, False), (1, True), (2, True), (8, False), (8, True), (32, False), (32, True), (LONG_CHUNK_ROWS, False)],
)
def test_read_rows_and_write_rows_equal_the_one_row_calls_row_for_row(
    expect_error, gr_fake, rows: int, flat_views: bool
) -> None:
    torch.manual_seed(7)
    width, partial = gr_module.FLAT_LOCAL_WIDTH, gr_module.PARTIAL_WIDTH
    weights = gr_module.Qwen38TTNNGatedResidualWeights(
        norm_scale=FakeTensor([(torch.randn(1, 4, 1, 640) * 0.1 + 0.25) for _ in range(TP)], FP32, TILE, 3),
        down_inject=FakeTensor(
            [bf16_randn(1, 1, width, partial, scale=width**-0.5) for _ in range(TP)], BF16, TILE, 2
        ),
        up=FakeTensor([bf16_randn(1, 1, partial, width, scale=partial**-0.5) for _ in range(TP)], BF16, TILE, 3),
        replicated_anchor=FakeTensor([torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16) for _ in range(TP)], BF16, TILE),
        epsilon=1e-6,
        layer_index=0,
        block="attn",
        namespace="backbone",
    )
    module = object.__new__(gr_module.Qwen38TTNNGatedResidual)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.weights = weights
    module.tt_ccl = SimpleNamespace(
        get_num_links=lambda axis: 1,
        get_and_cycle_ag_semaphore_handles=lambda axis: "ag",
        get_and_cycle_barrier_semaphore_handle=lambda axis: "barrier",
    )
    module.collective_topology = "linear"
    module.compute_config = "compute"
    module.down_inject_act_memory_config = gr_fake.DRAM_MEMORY_CONFIG
    module.down_inject_program_config = "down"
    module.up_act_memory_config = gr_fake.DRAM_MEMORY_CONFIG
    module.up_program_config = "up"
    module.norm_scale_flat = gr_fake.experimental.view(weights.norm_scale, (1, 1, 1, width))
    torch.manual_seed(11 + rows)
    residual = FakeTensor([bf16_randn(1, 4, rows, 640) for _ in range(TP)], BF16, TILE, 3)
    block_rows = FakeTensor([bf16_randn(1, 1, rows, 640) for _ in range(TP)], BF16, TILE, 3)
    block_input, state = module.read_rows(residual, flat_views=flat_views)
    written = module.write_rows(block_rows, state)
    assert block_input.shape == (1, 1, rows, 640) and state.injection.shape == (1, 1, rows, 4)
    assert written.shape == (1, 4, rows, 640) and state.residual is residual

    unit = CHUNK_ROWS if rows == LONG_CHUNK_ROWS else 1
    read, write = (module.read_rows, module.write_rows) if unit > 1 else (module.read, module.write)
    residual_parts, block_parts = (
        [
            FakeTensor([x.narrow(2, start, unit).clone() for x in whole.torch_shards()], BF16, TILE, 3)
            for start in range(0, rows, unit)
        ]
        for whole in (residual, block_rows)
    )
    reads = [read(part) for part in residual_parts]
    label = f"GR {rows} rows (flat views {flat_views})"
    assert_rows_bitwise(block_input, [block for block, _ in reads], dim=2, label=f"{label} block input")
    assert_rows_bitwise(state.injection, [part.injection for _, part in reads], dim=2, label=f"{label} injection")
    writes = [write(part, part_state) for part, (_, part_state) in zip(block_parts, reads)]
    assert_rows_bitwise(written, writes, dim=2, label=f"{label} write")
    if rows > 1:
        # The injection depends on the row: the per-row test is not vacuous.
        injection = state.injection.torch_shards()[0]
        assert not torch.equal(injection[..., 0, :], injection[..., 1, :])
    if flat_views:
        # Every branch is one 32-row tile row whatever the row count, so the branch-major tile sequence is the flat
        # rows' tile sequence: the view over the same pages reads what the permute + reshape pair materializes (the
        # fake models the tile pages, the rows below 32 as their tile padding).
        permuted, permuted_state = module.read_rows(residual)
        assert_rows_bitwise(block_input, [permuted], dim=2, label=f"{label} vs the permutes, block input")
        assert_rows_bitwise(state.injection, [permuted_state.injection], dim=2, label=f"{label} vs the permutes")
        flat = (1, 1, rows, gr_module.FLAT_LOCAL_WIDTH)
        by_view = gr_fake.experimental.view(residual, flat)
        by_permute = gr_fake.reshape(gr_fake.permute(residual, (0, 2, 1, 3)), flat)
        assert_rows_bitwise(by_view, [by_permute], dim=2, label=f"flat rows view vs permute + reshape at {rows} rows")
        assert by_view.padded_shape == (1, 1, CHUNK_ROWS, gr_module.FLAT_LOCAL_WIDTH)
        if rows > 1:
            # A plain torch reshape of the branch-major rows is not the device view: the fake's tile-page model matters.
            assert not torch.equal(
                by_view.torch_shards()[0].view(torch.int16),
                gr_fake.experimental.torch_view(residual, flat).torch_shards()[0].view(torch.int16),
            )
    if rows == 8:
        # Four block rows against the eight-row state: the row count comes from the residual rows.
        with expect_error(ValueError, match=r"GR block rows must be TILE BF16 \(1, 1, 8, 640\)"):
            module.write_rows(FakeTensor([bf16_randn(1, 1, 4, 640) for _ in range(TP)], BF16, TILE, 3), state)
    if rows == LONG_CHUNK_ROWS:
        with expect_error(ValueError):  # allow-pytest.raises: 64 rows are not a chunk form
            module.read_rows(FakeTensor([bf16_randn(1, 4, 64, 640) for _ in range(TP)], BF16, TILE, 3))


def test_gr_bodies_keep_their_op_walks_and_touch_no_host() -> None:
    read_calls = function_calls(Qwen38TTNNGatedResidual.read_rows)
    assert read_calls.count("ttnn.permute") == 2 and read_calls.count("ttnn.experimental.view") == 2
    assert read_calls.count("ttnn.reshape") == 5  # stats, flat rows, token-major rows, the two row restores
    assert read_calls.count("ttnn.linear") == 2 and read_calls.count("ttnn.experimental.fast_reduce_nc") == 2
    assert read_calls.count("ttnn.experimental.all_gather_async") == 1 and "ttnn.all_reduce" not in read_calls
    assert inspect.signature(Qwen38TTNNGatedResidual.read_rows).parameters["flat_views"].default is False
    write_calls = function_calls(Qwen38TTNNGatedResidual.write_rows)
    assert write_calls.count("ttnn.permute") == 1 and "ttnn.reshape" not in write_calls
    for calls in (read_calls, write_calls):
        for forbidden in ("ttnn.from_torch", "ttnn.to_torch", "ttnn.copy_host_to_device_tensor"):
            assert forbidden not in calls
    # The rows path derives its shapes from the validated row count of its input (1..32 lanes; 32 or 128 = the
    # chunk forms; the slab counts), never from a runtime int of its own: every shape comes from the module's
    # row-shape helpers, and below 32 rows the padded shapes are one tile (tile_rows).
    source = inspect.getsource(Qwen38TTNNGatedResidual.read_rows) + inspect.getsource(
        Qwen38TTNNGatedResidual.write_rows
    )
    assert source.count("residual_rows_shape(rows)") >= 3 and ".shape[" not in source
    assert "tile_rows = max(rows, CHUNK_ROWS)" in source and "block_rows_shape(rows)" in source
    # The 1-row read and write keep their views and op walk.
    for name in ("read", "_normalize"):
        calls = function_calls(getattr(Qwen38TTNNGatedResidual, name))
        assert calls.count("ttnn.experimental.view") == 1 and "ttnn.permute" not in calls, name
    write_one = [call for call in function_calls(Qwen38TTNNGatedResidual.write) if call.startswith("ttnn.")]
    assert write_one == ["ttnn.reshape", "ttnn.multiply", "ttnn.add"]
    partial = inspect.getsource(Qwen38TTNNGatedResidual._all_reduce_partial)
    assert "expected_shape != PARTIAL_REDUCTION_SHAPE" in partial and "CHUNK_ROWS" not in partial
