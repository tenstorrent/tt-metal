# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of ``gdn_post_rows``: the CB tables shared by each program's three kernels and the Python
side, the kernel argument layouts against the runners' source, the LLK text pins of both compute kernels, the two
constant tiles, the page maps against ``gdn_rows_reference``'s ``fold_heads`` and ``history_next`` on an index
tensor, the unit split, and the row-mask semantics of the reference.

Two of the pins deliberately follow the RUNTIME source rather than the lane design's prose.  (1) The reduce's
epilogue: ``numeric.h``'s ``accumulate_compute_loop`` ends with ``reduce_uninit`` and a reconfig, and
``row_wise_accumulate_with_epilogue`` calls the 1/W ``scale_dest`` after that, still inside the same
``tile_regs_acquire`` -- so the order pinned here is reduce_tile, reduce_uninit, scale_dest, not the design's
"scale_dest then reduce_uninit".  (2) The two APPROX = true calls are the op's own macro calls with ``true``
written in place of ``APPROX``: on Blackhole there is no ``llk_math_eltwise_unary_sfpu_rsqrt`` free function (that
name exists only under the Quasar LLK tree), the API's ``rsqrt_tile`` / ``mul_unary_tile`` reach the ckernel
functions through ``SFPU_UNARY_CALL`` / ``SFPU_UNARY_INIT_FN``, and those macros substitute the kernel's own
APPROX -- which is false here, because this kernel also runs binary_ng's Precise multiply.
"""

from __future__ import annotations

import inspect
import re
import struct
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_post_rows as module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_reference as gr
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

KERNELS = Path(module.__file__).parent / "kernels"
CAST_SOURCES = {role: (KERNELS / f"{role}_cast.cpp").read_text() for role in ("reader", "compute", "writer")}
NORM_SOURCES = {role: (KERNELS / f"{role}_norm.cpp").read_text() for role in ("reader", "compute", "writer")}
TILE = gr.TILE


def _cb_constants(source: str) -> dict[str, int]:
    return {name: int(value) for name, value in re.findall(r"\b(CB_[A-Z0-9_]+)\s*=\s*(\d+)", source)}


def _code(source: str) -> str:
    """The kernel with its ``//`` comments removed: a pin must match a call, never a sentence about one."""

    return "\n".join(line.split("//")[0] for line in source.splitlines())


def _squeezed(function) -> str:
    return re.sub(r"\s+", "", inspect.getsource(function))


# ------------------------------------------------------------------------------------------------- CB tables


def test_cast_cb_indices_agree_between_kernels_and_python():
    declared = {index: dtype for index, dtype, _pages in module.CAST_CBS}
    for role, source in CAST_SOURCES.items():
        names = _cb_constants(source)
        assert names, role
        assert names.get("CB_IN", module.CB_CAST_IN) == module.CB_CAST_IN
        assert names.get("CB_OUT", module.CB_CAST_OUT) == module.CB_CAST_OUT
        for name, index in names.items():
            assert index in declared, f"{role}: {name} = {index} is not in CAST_CBS"
    assert _cb_constants(CAST_SOURCES["compute"]) == {"CB_IN": module.CB_CAST_IN, "CB_OUT": module.CB_CAST_OUT}
    assert declared[module.CB_CAST_IN] is module.FP32 and declared[module.CB_CAST_OUT] is module.BF16
    # the typecast op unpacks its fp32 input straight to the destination; nothing else may
    assert module.CAST_FP32_COPY_CBS == (module.CB_CAST_IN,)


def test_norm_cb_indices_agree_between_kernels_and_python():
    declared = {index: dtype for index, dtype, _pages in module.NORM_CBS}
    expected = {
        "CB_X": module.CB_X,
        "CB_SIG": module.CB_SIG,
        "CB_SCALER": module.CB_SCALER,
        "CB_EPS": module.CB_EPS,
        "CB_GAMMA": module.CB_GAMMA,
        "CB_PROJ": module.CB_PROJ,
        "CB_XMM2": module.CB_XMM2,
        "CB_EX2": module.CB_EX2,
        "CB_EX2PE": module.CB_EX2PE,
        "CB_FUSION": module.CB_FUSION,
        "CB_NRM": module.CB_NRM,
        "CB_OUT": module.CB_OUT,
        "CB_HIST": module.CB_HIST,
    }
    assert set(expected.values()) == set(declared)
    assert len(set(expected.values())) == len(expected)
    for role, source in NORM_SOURCES.items():
        for name, index in _cb_constants(source).items():
            assert name in expected, f"{role}: unknown {name}"
            assert expected[name] == index, f"{role}: {name} = {index}, Python has {expected[name]}"
    # every CB of this program is bf16: one destination width, one format
    assert all(dtype is module.BF16 for dtype in declared.values())
    # the compute kernel never sees the history path
    compute = _cb_constants(NORM_SOURCES["compute"])
    assert "CB_PROJ" not in compute and "CB_HIST" not in compute
    # and the writer never sees the norm's intermediates
    writer = set(_cb_constants(NORM_SOURCES["writer"]))
    assert writer == {"CB_PROJ", "CB_OUT", "CB_HIST"}


def test_cb_indices_are_addressable_and_the_l1_footprint_is_reported():
    for table in (module.CAST_CBS, module.NORM_CBS):
        for index, dtype, pages in table:
            assert 0 <= index < fp.CB_COUNT and pages >= 1 and dtype in fp.TILE_BYTES
    cast_bytes = sum(fp.TILE_BYTES[dtype] * pages for _index, dtype, pages in module.CAST_CBS)
    norm_bytes = sum(fp.TILE_BYTES[dtype] * pages for _index, dtype, pages in module.NORM_CBS)
    assert cast_bytes == 8 * 4096 + 8 * 2048  # 48 KiB
    assert norm_bytes == 68 * 2048  # 136 KiB
    assert cast_bytes < 1 << 20 and norm_bytes < 1 << 20


# --------------------------------------------------------------------------------------- argument layouts


def test_cast_argument_layout_matches_the_runner():
    reader, writer, compute = (_code(CAST_SOURCES[r]) for r in ("reader", "writer", "compute"))
    assert reader.count("TensorAccessorArgs<") == 1 and "TensorAccessorArgs<0>()" in reader
    assert writer.count("TensorAccessorArgs<") == 1 and "TensorAccessorArgs<0>()" in writer
    assert reader.count("get_arg_val<uint32_t>(arg++)") == 3  # address, groups, first group
    assert writer.count("get_arg_val<uint32_t>(arg++)") == 3
    assert "get_arg_val<uint32_t>(0)" in compute  # units on this core
    source = _squeezed(module.post_cast)
    assert "fp.accessor_args(o)" in source and "fp.accessor_args(o16)" in source
    assert "[o.buffer_address(),w.count,w.start]" in source
    assert "[o16.buffer_address(),w.count,w.start]" in source
    assert "[(w.core,[w.count])forwinwork]" in source
    assert "fp32_dest=True" in source and "approx=False" in source
    assert "unpack_to_dest_fp32=CAST_FP32_COPY_CBS" in source
    assert "fp.core_rectangle(work,mesh)" in source


def test_norm_argument_layout_matches_the_runner():
    reader, writer, compute = (_code(NORM_SOURCES[r]) for r in ("reader", "writer", "compute"))
    assert reader.count("TensorAccessorArgs<") == 5  # o16, sig, norm, scalars, projected
    assert reader.count("next_compile_time_args_offset()") == 4
    assert reader.count("get_arg_val<uint32_t>(arg++)") == 9  # 5 addresses, chunks, history, units, first
    assert writer.count("TensorAccessorArgs<") == 2  # gated, history_next
    assert writer.count("get_arg_val<uint32_t>(arg++)") == 7  # 2 addresses, chunks, rows, history, units, first
    assert "get_arg_val<uint32_t>(0)" in compute
    source = _squeezed(module.post_norm)
    assert "fortensorin(o16,sig,norm,scalars,projected):" in source
    assert "reader_cta.extend(fp.accessor_args(tensor))" in source
    assert "[*fp.accessor_args(gated),*fp.accessor_args(hist_out)]" in source
    assert "[*reader_addrs,chunks,flag,w.count,w.start]" in source
    assert "[*writer_addrs,chunks,rows,flag,w.count,w.start]" in source
    assert "fp32_dest=False" in source and "approx=False" in source
    assert "unpack_to_dest_fp32" not in source  # every CB is bf16
    assert "fp.core_rectangle(work,mesh)" in source


# ------------------------------------------------------------------------------------------------- LLK pins


def test_compute_cast_is_the_typecast_ops_call_sequence():
    compute = _code(CAST_SOURCES["compute"])
    assert "compute_kernel_hw_startup(CB_IN, CB_OUT);" in compute
    assert "copy_init(CB_IN);" in compute
    assert "copy_tile(CB_IN, 0, 0);" in compute
    # The data-format constants are named DF_*: a bare BF16 collides with a macro the JIT injects on Blackhole, and
    # the kernel then fails to compile ("parse error in template argument list" on the typecast; measured on one die,
    # the whole device suite failed until the rename).  No kernel of this lane may use a bare FP32 / BF16 identifier.
    assert "typecast_tile_init<DF_FP32, DF_BF16>();" in compute
    assert "typecast_tile<DF_FP32, DF_BF16>(0);" in compute
    assert "pack_tile(0, CB_OUT);" in compute
    assert compute.index("copy_tile(CB_IN") < compute.index("typecast_tile_init<") < compute.index("typecast_tile<DF_")
    assert compute.index("typecast_tile<DF_") < compute.index("pack_tile(0, CB_OUT)")
    assert "constexpr uint32_t DF_FP32 = static_cast<uint32_t>(DataFormat::Float32);" in compute
    assert "constexpr uint32_t DF_BF16 = static_cast<uint32_t>(DataFormat::Float16_b);" in compute
    for source in CAST_SOURCES.values():
        assert not re.search(
            r"\b(?<!DF_)(FP32|BF16)\b(?!_)", _code(source)
        ), "a bare FP32 / BF16 collides with a JIT macro"
    for wrong in ("mul_tiles", "reduce_tile", "rsqrt", "mul_binary_tile"):
        assert wrong not in compute, wrong


def test_compute_norm_is_the_layernorm_and_binary_ng_call_sequences_in_order():
    compute = _code(NORM_SOURCES["compute"])
    ordered = [
        "compute_kernel_hw_startup(CB_X, CB_X, CB_XMM2);",
        "mul_init(CB_X, CB_X);",
        "mul_tiles(CB_X, CB_X, i, i, i);",
        "pack_tile(i, CB_XMM2);",
        "reconfig_data_format(CB_SCALER, CB_XMM2);",
        "reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XMM2, CB_SCALER, CB_EX2);",
        "reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XMM2, CB_SCALER, j, 0, dst0);",
        "reduce_uninit();",
        "binop_with_scalar_tile_init();",
        "calculate_binop_with_scalar,",
        "pack_tile(dst0, CB_EX2);",
        "add_init(CB_EX2, CB_EPS);",
        "add_tiles(CB_EX2, CB_EPS, 0, 0, dst0);",
        "sfpu::rsqrt_init,",
        "calculate_rsqrt,",
        "pack_tile(dst0, CB_EX2PE);",
        "mul_bcast_cols_init(CB_X, CB_EX2PE);",
        "mul_tiles_bcast_cols(CB_X, CB_EX2PE, i, 0, i);",
        "pack_tile(i, CB_FUSION);",
        "mul_bcast_rows_init(CB_FUSION, CB_GAMMA);",
        "mul_tiles_bcast_rows(CB_FUSION, CB_GAMMA, i, i, i);",
        "pack_tile(i, CB_NRM);",
        "copy_tile(CB_NRM, i, dst0);",
        "copy_tile(CB_SIG, i, dst1);",
        "mul_binary_tile_init();",
        "mul_binary_tile(0, 1, 0);",
        "pack_tile(dst0, CB_OUT);",
    ]
    positions = []
    for text in ordered:
        assert text in compute, text
        positions.append(compute.index(text))
    assert positions == sorted(positions), "the call sequence is out of the op's order"
    # numeric.h's real order, not the design prose's: reduce_uninit comes BEFORE the 1/W scale_dest
    assert compute.index("reduce_uninit();") < compute.index("binop_with_scalar_tile_init();")
    # the 1/W constant is bit_cast<uint32_t>(1.0f / 128)
    assert "constexpr uint32_t RECIP_W = 0x3C000000u;" in compute
    assert module.RECIP_HEAD_DIM_BITS == struct.unpack("<I", struct.pack("<f", 1.0 / gr.HEAD_DIM))[0]
    # mul_binary_tile's init sits immediately before the call, as binary_ng's kernel has it
    gate = compute.index("mul_binary_tile_init();")
    assert compute[gate : gate + 200].index("mul_binary_tile(0, 1, 0);") < 80
    # the destination width is the op's: no fp32 dest, and no typecast in this kernel
    for wrong in ("typecast_tile", "fp32_dest=True", "unpack_to_dest"):
        assert wrong not in compute, wrong


def test_compute_norm_spells_the_two_approx_true_calls_out():
    compute = _code(NORM_SOURCES["compute"])
    squeezed = re.sub(r"\s+", " ", compute)
    # the 1/W scale: ckernel::sfpu::calculate_binop_with_scalar<true, MUL_UNARY, 8, false>
    assert (
        "MATH(SFPU_UNARY_CALL( DST_SYNC_MODE, false /* is_fp32_dest_acc_en */, calculate_binop_with_scalar, "
        "(true /* APPROX */, MUL_UNARY, 8 /* ITERATIONS */, false /* is_fp32_dest_acc_en */), dst0, "
        "VectorMode::RC, RECIP_W));" in squeezed
    )
    # rsqrt: ckernel::sfpu::rsqrt_init<true, false> and calculate_rsqrt<true, 8, false, false, false>
    assert (
        "MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::rsqrt_init, (true /* APPROX */, false /* legacy_compat */)));" in squeezed
    )
    assert (
        "MATH(SFPU_UNARY_CALL( DST_SYNC_MODE, false /* is_fp32_dest_acc_en */, calculate_rsqrt, "
        "(true /* APPROX */, 8 /* ITERATIONS */, false /* is_fp32_dest_acc_en */, false /* FAST_APPROX */, "
        "false /* legacy_compat */), dst0, VectorMode::RC));" in squeezed
    )
    # the API wrappers would pass the kernel's own APPROX (false here): they must not appear
    for wrong in ("rsqrt_tile<", "rsqrt_tile_init<", "mul_unary_tile("):
        assert wrong not in compute, wrong


def test_writer_norm_pins_the_row_mask_and_the_history_face_arithmetic():
    writer = _code(NORM_SOURCES["writer"])
    assert "constexpr uint32_t row_half(uint32_t row, uint32_t segment) {" in writer
    assert "return ((row >> 4) * 2 + segment) * FACE_BYTES + (row & 15) * HALF_ROW_BYTES;" in writer
    assert "const uint32_t source = TILE_ROWS - HISTORY_ROWS + r;" in writer
    assert "chunk * VALUE_TILES + head * HEAD_TILES + d" in writer
    assert "head * HISTORY_COLUMNS + j" in writer
    assert "if (valid < TILE_ROWS)" in writer
    reader = _code(NORM_SOURCES["reader"])
    assert "chunk * VALUE_TILES + head * HEAD_TILES + d" in reader
    assert "chunk * PROJECTION_TILES + head * HISTORY_COLUMNS + j" in reader
    assert "unit * HEAD_TILES + d" in reader


def test_kernel_geometry_constants_agree_with_python():
    for source in list(CAST_SOURCES.values()) + list(NORM_SOURCES.values()):
        for name, value in re.findall(r"constexpr uint32_t ([A-Z_]+) = (\d+);", source):
            expected = {
                "HEAD_TILES": module.HEAD_TILES,
                "GROUP": module.HEAD_TILES,
                "VALUE_TILES": module.VALUE_TILES,
                "PROJECTION_TILES": module.PROJECTION_TILES,
                "HISTORY_HEADS": module.HISTORY_HEADS,
                "HISTORY_COLUMNS": module.HISTORY_COLUMNS,
                "HISTORY_ROWS": gr.HISTORY_ROWS,
                "TILE_ROWS": TILE,
                "BF16_TILE": module.TILE_BF16,
                "FP32_TILE": module.TILE_FP32,
            }.get(name)
            if expected is not None:
                assert int(value) == expected, f"{name} = {value}, Python has {expected}"


# ------------------------------------------------------------------------------------------ the constant tiles


def _bits16(tensor: torch.Tensor) -> int:
    """The raw 16 bits of a one-element bf16 tensor."""

    return int(tensor.reshape(-1).contiguous().view(torch.int16)[0]) & 0xFFFF


def test_scalar_tiles_are_the_layernorm_readers_two_tiles():
    tiles = module.scalar_tiles()
    assert tuple(tiles.shape) == (1, 1, TILE, 2 * TILE) and tiles.dtype is torch.bfloat16
    scaler, epsilon = tiles[0, 0, :, :TILE], tiles[0, 0, :, TILE:]
    # calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>: an exact 1.0 in row 0 of each of the four faces,
    # which is whole tile rows 0 and 16; every other element zero
    rows_with_ones = {int(r) for r in (scaler != 0).any(dim=1).nonzero().flatten()}
    assert rows_with_ones == {0, fp.FACE}
    assert torch.equal(scaler[0], torch.ones(TILE, dtype=torch.bfloat16))
    assert torch.equal(scaler[fp.FACE], torch.ones(TILE, dtype=torch.bfloat16))
    assert float(scaler.float().sum()) == 2 * TILE
    # generate_bcast_col_scalar: bits(eps) >> 16 (a truncation) in column 0 of every row
    expected_bits = struct.unpack("<I", struct.pack("<f", gr.RMS_NORM_EPS))[0] >> 16
    column = epsilon[:, 0]
    assert _bits16(column[:1]) == expected_bits
    assert int((column != column[0]).sum()) == 0
    assert float(epsilon[:, 1:].float().abs().sum()) == 0.0
    # the truncation is the point: an RNE rounding would carry for an eps whose low 16 bits are above half
    raw = struct.unpack("<I", struct.pack("<f", 1.7e-7))[0]
    assert raw & 0xFFFF > 0x8000 and (raw >> 16) + 1 != raw >> 16
    awkward = module.scalar_tiles(1.7e-7)[0, 0, :, TILE:]
    assert _bits16(awkward[0, :1]) == raw >> 16


# ---------------------------------------------------------------------------------------------- the page maps


def _head_major_pages(x: torch.Tensor) -> torch.Tensor:
    """``[12, T, 128]`` -> its TILE pages in buffer order."""

    heads, rows, width = x.shape
    return (
        x.reshape(heads, rows // TILE, TILE, width // TILE, TILE)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, TILE, TILE)
        .contiguous()
    )


def _token_major_pages(x: torch.Tensor) -> torch.Tensor:
    """``[T, W]`` -> its TILE pages in buffer order."""

    rows, width = x.shape
    return x.reshape(rows // TILE, TILE, width // TILE, TILE).permute(0, 2, 1, 3).reshape(-1, TILE, TILE).contiguous()


@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_the_writers_page_map_is_the_chains_head_fold(chunks):
    rows = chunks * TILE
    index = torch.arange(gr.HEADS * rows * gr.HEAD_DIM, dtype=torch.float32).reshape(gr.HEADS, rows, gr.HEAD_DIM)
    source = _head_major_pages(index)
    folded = _token_major_pages(gr.fold_heads(index))
    for head in range(gr.HEADS):
        for chunk in range(chunks):
            for column in range(module.HEAD_TILES):
                read = module.head_tile_page(head, chunk, column, chunks)
                write = module.token_tile_page(chunk, head, column)
                assert torch.equal(source[read], folded[write]), (head, chunk, column)
    # the head-major page of a unit's first column tile is 4 * unit: the reader's contiguous run
    for unit, (head, chunk) in enumerate(module.units(chunks)):
        assert module.head_tile_page(head, chunk, 0, chunks) == unit * module.HEAD_TILES


@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_the_history_row_and_face_arithmetic_reproduces_history_next(chunks):
    rows = chunks * TILE
    index = torch.arange(rows * gr.PROJECTION_WIDTH, dtype=torch.float32).reshape(rows, gr.PROJECTION_WIDTH)
    projection = _token_major_pages(index)
    expected = _token_major_pages(gr.history_next(index[:, : gr.QKV_WIDTH]))
    assert expected.shape[0] == module.QKV_TILES
    seen = set()
    for (head, chunk), columns in module.history_units(chunks).items():
        assert chunk == chunks - 1 and head < module.HISTORY_HEADS
        for column in columns:
            seen.add(column)
            page = projection[module.projection_tile_page(chunk, column)]
            built = torch.zeros(TILE, TILE)
            built[: gr.HISTORY_ROWS] = page[TILE - gr.HISTORY_ROWS :]
            assert torch.equal(built, expected[column]), (head, column)
    assert seen == set(range(module.QKV_TILES))  # every q|k|v column tile is written exactly once
    assert sum(len(c) for c in module.history_units(chunks).values()) == module.QKV_TILES


@pytest.mark.parametrize("chunks", [1, 2, 64])
def test_the_unit_split_covers_every_head_and_tile_row_once(chunks):
    work = module.units(chunks)
    assert len(work) == gr.HEADS * chunks
    assert sorted(work) == sorted((head, chunk) for head in range(gr.HEADS) for chunk in range(chunks))
    assert len(set(work)) == len(work)
    # a core's contiguous run of units walks one head's tile rows: the head-major pages stay sequential
    for unit in range(1, len(work)):
        head, chunk = work[unit]
        previous_head, previous_chunk = work[unit - 1]
        assert (head, chunk) == (previous_head, previous_chunk + 1) or (head, chunk) == (previous_head + 1, 0)
    # the history units are NC apart, so ten different cores carry them under any even split
    history = sorted(work.index(unit) for unit in module.history_units(chunks))
    assert history == [head * chunks + chunks - 1 for head in range(module.HISTORY_HEADS)]


def test_geometry_constants():
    assert (module.HEAD_TILES, module.VALUE_TILES, module.QKV_TILES, module.PROJECTION_TILES) == (4, 48, 80, 130)
    assert module.HISTORY_HEADS * module.HISTORY_COLUMNS == module.QKV_TILES
    assert module.HISTORY_HEADS == 10 and module.HISTORY_COLUMNS == 8


# ------------------------------------------------------------------------------------------------- the reference


def _case(chunks: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    rows = chunks * TILE
    o = torch.randn(gr.HEADS, rows, gr.HEAD_DIM, generator=g) * 0.7
    z = (torch.randn(rows, gr.VALUE_WIDTH, generator=g) * 3.0).to(torch.bfloat16)
    norm = (1.0 + torch.randn(gr.HEAD_DIM, generator=g) * 0.1).to(torch.bfloat16)
    projected = (torch.randn(rows, gr.PROJECTION_WIDTH, generator=g) * 0.6).to(torch.bfloat16)
    return o, z, norm, projected


def test_reference_is_post_reference_plus_the_history_and_the_row_mask():
    o, z, norm, projected = _case(2, seed=11)
    out = module.reference(o, z, norm, projected)
    assert torch.equal(out["gated"].view(torch.int16), gr.post_reference(o, z, norm).view(torch.int16))
    assert torch.equal(
        out["history_next"].view(torch.int16), gr.history_next(projected[:, : gr.QKV_WIDTH]).view(torch.int16)
    )
    assert tuple(out["gated"].shape) == (2 * TILE, gr.VALUE_WIDTH) and out["gated"].dtype is torch.bfloat16
    assert tuple(out["history_next"].shape) == (TILE, gr.QKV_WIDTH)
    assert module.reference(o, z, norm, projected, history=False)["history_next"] is None


def test_reference_takes_the_pre_programs_sigmoid_as_well_as_z():
    o, z, norm, projected = _case(1, seed=12)
    sig = gr.typecast_to_bf16(gr.sigmoid_fp32(gr.typecast_to_fp32(z)))
    from_z = module.reference(o, z, norm, projected)["gated"]
    from_sig = module.reference(o, sig, norm, projected, sigmoid_applied=True)["gated"]
    assert torch.equal(from_z.view(torch.int16), from_sig.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 5, 31, 32])
def test_the_row_mask_zeroes_the_partial_tile_rows_tail(rows):
    o, z, norm, projected = _case(1, seed=13)
    full = module.reference(o, z, norm, projected)["gated"]
    masked = module.reference(o, z, norm, projected, rows=rows)["gated"]
    assert torch.equal(masked[:rows].view(torch.int16), full[:rows].view(torch.int16))
    tail = masked[rows:]
    assert torch.equal(tail, torch.zeros_like(tail))
    # exact +0, as the chain's x * 0.0 leaves under the bf16 multiply's zero clamp
    assert int(tail.view(torch.int16).abs().sum()) == 0


def test_the_row_mask_spans_whole_tile_rows_too():
    o, z, norm, projected = _case(3, seed=14)
    masked = module.reference(o, z, norm, projected, rows=40)["gated"]
    assert int(masked[40:].view(torch.int16).abs().sum()) == 0
    assert int(masked[:40].view(torch.int16).abs().sum()) != 0


def test_the_module_registers_nothing_yet():
    # the lane lead adds the registry entry and the manifest rows when the pair is integrated
    source = Path(module.__file__).read_text()
    assert "register(" not in source and "FusedKernel(" not in source
    from models.demos.blackhole.qwen38_flash_next.ttnn import fused

    assert "gdn_post_rows" not in fused.kernels()
