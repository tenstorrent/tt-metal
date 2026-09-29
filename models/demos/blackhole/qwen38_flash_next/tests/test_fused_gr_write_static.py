# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused GR write without a device: its registry entry, the kernel argument contracts against the Python side,
the exactness pins in the compute kernel (the chain's two binary_ng SFPU calls in the chain's operand order), the
reader's column patch and the writer's page map against a torch model of the tile layout, the composed chain's op
order against the model's ``write`` / ``write_rows``, and the model hook."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_write as gw
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

SOURCES = {name: (fp.REPO_ROOT / path).read_text() for name, path in gw.KERNELS.items()}
GR_SOURCE = Path(__file__).resolve().parents[1] / "ttnn" / "gr.py"


def test_registered_bitwise_with_a_gate():
    entry = fused.kernel("gr_write")
    assert entry.tolerance == fused.BITWISE
    assert entry.fused is gw.gr_write and entry.composed is gw.gr_write_composed
    assert entry.gate is not None and entry.gate.layers == tuple(range(1, 48)) and entry.gate.reference is None
    default = gw.gr_write if "gr_write" in fused.DEFAULT_ON else gw.gr_write_composed  # the registry list decides
    assert fused.resolve("gr_write", {}) is default
    assert fused.resolve("gr_write", {fused.OFF_ENV: "gr_write"}) is gw.gr_write_composed
    assert fused.resolve("gr_write", {fused.ENV: "gr_write"}) is gw.gr_write
    assert inspect.signature(gw.gr_write).parameters.keys() == inspect.signature(gw.gr_write_composed).parameters.keys()


def test_geometry_and_cb_table():
    assert (gw.BRANCHES, gw.LOCAL_HIDDEN, gw.HIDDEN_TILES, gw.UNITS) == (4, 640, 20, 80)
    indices = [index for _name, index in gw.CBS]
    assert len(set(indices)) == len(indices) and max(indices) < fp.CB_COUNT
    assert set(gw.NAMED) == set(gw.CB_INDEX) | {"hidden_tiles", "branches"}


@pytest.mark.parametrize("kernel", ["reader", "compute", "writer"])
def test_named_compile_time_args_exist_on_the_python_side(kernel):
    names = set(re.findall(r'get_named_compile_time_arg_val\("([a-z_0-9]+)"\)', SOURCES[kernel]))
    assert names, kernel
    assert names <= set(gw.NAMED), names - set(gw.NAMED)


def test_runtime_arg_layout_matches_the_python_side():
    for kernel, args in (("reader", gw.READER_ARGS), ("writer", gw.WRITER_ARGS), ("compute", gw.COMPUTE_ARGS)):
        assert [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d)\)", SOURCES[kernel])] == list(range(len(args)))
    reader, writer = SOURCES["reader"], SOURCES["writer"]
    assert "TensorAccessorArgs<0>()" in reader and reader.count("next_compile_time_args_offset()") == 2
    assert "TensorAccessorArgs<0>()" in writer and "next_compile_time_args_offset()" not in writer
    source = re.sub(r"\s+", "", inspect.getsource(gw.gr_write_program))
    assert "fp.accessor_args(block_output)+fp.accessor_args(residual)+fp.accessor_args(injection)" in source
    assert (
        "[block_output.buffer_address(),residual.buffer_address(),injection.buffer_address(),w.start,w.count,]"
        in source
    )
    assert "[output.buffer_address(),w.start,w.count]" in source and "[(w.core,[w.count])forwinwork]" in source
    assert "fp32_dest=False" in source and "fp.core_rectangle(work," in source  # 16-bit dest, one kernel group


def test_compute_kernel_pins_the_chains_two_programs_in_operand_order():
    compute = SOURCES["compute"]
    assert "constexpr uint32_t dst_upd = 0, dst_coef = 1;" in compute
    assert "compute_kernel_hw_startup(cb_block, cb_coef, cb_out);" in compute
    # ttnn.multiply (fast_and_approximate_mode defaults to False): binary_ng's SFPU kernel = copy lhs, broadcast rhs,
    # mul_binary_tile (fp32 product, software RNE, 0 * x = 0), pack bf16
    assert "copy_tile(cb_block, 0, dst_upd);" in compute
    assert (
        "unary_bcast_init<BroadcastType::COL>(cb_coef);" in compute
        and "unary_bcast<BroadcastType::COL>(cb_coef, 0, dst_coef);" in compute
    )
    assert "mul_binary_tile(dst_upd, dst_coef, dst_upd);" in compute and "pack_tile(dst_upd, cb_upd);" in compute
    # ttnn.add (fast_and_approximate_mode defaults to True): binary_ng's FPU kernel = add_tiles(lhs residual, rhs update)
    assert "binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_res, cb_upd);" in compute
    assert "add_tiles(cb_res, cb_upd, 0, 0, 0);" in compute and "pack_tile(0, cb_out);" in compute
    assert compute.index("mul_binary_tile(dst") < compute.index("pack_tile(dst_upd") < compute.index("add_tiles(cb_res")
    for wrong in ("mul_tiles", "add_binary_tile", "fp32_dest=True"):
        assert wrong not in compute, wrong


def _tile_words(matrix: torch.Tensor) -> torch.Tensor:
    """A 32x32 matrix in bf16 tile memory order (16-bit words): faces 0..3, row-major inside each 16x16 face."""

    faces = [matrix[r : r + 16, c : c + 16].reshape(-1) for r in (0, 16) for c in (0, 16)]
    return torch.cat(faces)


def test_reader_column_patch_matches_the_tile_layout():
    reader = SOURCES["reader"]
    assert (
        "const uint32_t row0 = (r >> 4) * 512 + (r & 15) * 16;" in reader and "tile[row0] = tile[row0 + b];" in reader
    )
    assert "for (uint32_t r = 0; r < 32; ++r)" in reader
    tile = torch.arange(1, 32 * 32 + 1).reshape(32, 32)
    words = _tile_words(tile)
    for b in range(gw.BRANCHES):
        for r in range(32):
            row0 = (r >> 4) * 512 + (r & 15) * 16
            assert words[row0] == tile[r, 0] and words[row0 + b] == tile[r, b]


def test_page_maps_are_the_residuals_branch_major_tiles():
    reader, writer = SOURCES["reader"], SOURCES["writer"]
    assert "const uint32_t b = u % branches;" in reader and "const uint32_t j = u / branches;" in reader
    assert "{.page_id = j}" in reader and "{.page_id = hidden_tiles * b + j}" in reader and "{.page_id = 0}" in reader
    assert "{.page_id = hidden_tiles * (u % branches) + u / branches}" in writer
    # residual [1, 4, 32, 640] TILE: tile (branch b, column j) is page 20 b + j; column-major unit order groups a column's
    # four branch tiles on one core when the 80 units split over 20 cores
    pages = [gw.HIDDEN_TILES * (u % gw.BRANCHES) + u // gw.BRANCHES for u in range(gw.UNITS)]
    assert sorted(pages) == list(range(gw.UNITS)) and pages[:4] == [0, 20, 40, 60] and pages[4:8] == [1, 21, 41, 61]


def _grid(x: int, y: int):
    return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=x, y=y))


def test_split_work_cores_knob(expect_error):
    default = fp.split_work(gw.UNITS, _grid(13, 10))
    assert len(default) == 80 and {w.count for w in default} == {1}
    twenty = fp.split_work(gw.UNITS, _grid(13, 10), cores=20)
    assert len(twenty) == 20 and {w.count for w in twenty} == {4} and [w.start for w in twenty] == list(range(0, 80, 4))
    assert len(fp.split_work(gw.UNITS, _grid(8, 8))) == 64
    with expect_error(ValueError):
        fp.split_work(gw.UNITS, _grid(13, 10), cores=81)
    with expect_error(ValueError):
        fp.split_work(gw.UNITS, _grid(13, 10), cores=0)
    assert gw.cores_of(_grid(13, 10), {}) == 80 and gw.cores_of(_grid(13, 10), {gw.CORES_ENV: "20"}) == 20
    rectangle = fp.core_rectangle(fp.split_work(gw.UNITS, _grid(13, 10)), _grid(13, 10)).ranges()
    assert len(rectangle) == 1 and (rectangle[0].end.x, rectangle[0].end.y) == (7, 9)  # 80 cores = 8 columns x 10
    assert len(fp.core_rectangle(fp.split_work(gw.UNITS, _grid(13, 10), cores=20), _grid(13, 10)).ranges()) == 1
    partial = fp.core_rectangle(fp.split_work(25, _grid(13, 10)), _grid(13, 10)).ranges()  # two full columns + 5
    assert len(partial) == 2 and (partial[1].start.x, partial[1].start.y, partial[1].end.x, partial[1].end.y) == (
        2,
        0,
        2,
        4,
    )
    four = fp.core_rectangle(fp.split_work(4, _grid(13, 10)), _grid(13, 10)).ranges()  # gr_read's 4-core programs
    assert len(four) == 1 and (four[0].end.x, four[0].end.y) == (0, 3)


def _calls(function: ast.FunctionDef) -> list[str]:
    """The ``ttnn.<op>`` calls of a function in source order (deallocate excluded)."""

    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ttnn"
        and node.func.attr != "deallocate"
    ]
    return [f"ttnn.{node.func.attr}" for node in sorted(calls, key=lambda node: (node.lineno, node.col_offset))]


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGatedResidual":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(name)


def test_composed_chain_is_the_models_write_and_write_rows_op_for_op():
    tree = ast.parse(GR_SOURCE.read_text(encoding="utf-8"))
    assert _calls(_method(tree, "write")) == ["ttnn.reshape", "ttnn.multiply", "ttnn.add"]
    assert _calls(_method(tree, "write_rows")) == ["ttnn.permute", "ttnn.multiply", "ttnn.add"]
    composed = ast.parse(inspect.getsource(gw.gr_write_composed))
    assert _calls(composed.body[0]) == ["ttnn.reshape", "ttnn.permute", "ttnn.multiply", "ttnn.add"]
    source = inspect.getsource(gw.gr_write_composed)
    assert "ttnn.reshape(injection, (1, BRANCHES, 1, 1))" in source and "ttnn.permute(injection, (0, 3, 2, 1)" in source
    assert "ttnn.multiply(block_output, coefficient" in source and "ttnn.add(residual, update" in source


def test_model_hook_binds_write_and_write_rows_when_switched_on():
    source = GR_SOURCE.read_text(encoding="utf-8")
    hook = source[source.index('fused_kernels.enabled("gr_write")') :]
    assert "self.write = functools.partial(fused_gr_write.write_fused, self)" in hook
    assert "self.write_rows = functools.partial(fused_gr_write.write_rows_fused, self)" in hook
    assert "type(module).write_rows(module, block_rows, state)" in inspect.getsource(
        gw.write_rows_fused
    )  # 128 rows keep the chain
    assert "if int(state.residual.shape[2]) > fp.TILE:" in inspect.getsource(gw.write_rows_fused)


def test_rows_of_rejects_other_shapes(expect_error):
    def tensor(shape, padded_rows=32, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        padded = list(shape)
        padded[2] = padded_rows
        return SimpleNamespace(shape=shape, padded_shape=padded, dtype=dtype, layout=layout)

    assert gw.rows_of(tensor((1, 1, 5, 640)), tensor((1, 4, 5, 640)), tensor((1, 1, 5, 4))) == 5
    with expect_error(ValueError):
        gw.rows_of(tensor((1, 1, 5, 640)), tensor((1, 4, 64, 640), padded_rows=64), tensor((1, 1, 64, 4)))
    with expect_error(ValueError):
        gw.rows_of(tensor((1, 1, 4, 640)), tensor((1, 4, 5, 640)), tensor((1, 1, 5, 4)))
    with expect_error(ValueError):
        gw.rows_of(tensor((1, 1, 5, 640), dtype=ttnn.float32), tensor((1, 4, 5, 640)), tensor((1, 1, 5, 4)))
