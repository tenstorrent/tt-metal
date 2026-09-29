# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused position derive without a device: its registry entry, the kernel's argument contract against the Python
side (21 tensors, chained accessors, named constants), the staging layout, the one-hot lane formula against a torch
model of the tile layout, the geometry formulas against ``qsa_selection_geometry``, and the composed chain against the
model body's four lines."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import position_derive as pd
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp


def _qsa():
    """The QSA module, or a skip where only the ttnn import stub is present (no-runtime hosts)."""

    try:
        from models.demos.blackhole.qwen38_flash_next.ttnn import qsa
    except (AttributeError, RecursionError) as error:  # the import stub has no device-only attributes
        pytest.skip(f"needs the real ttnn: {error}")
    return qsa


SOURCE = (fp.REPO_ROOT / pd.KERNEL).read_text()
LANES_SOURCE = (fp.REPO_ROOT / pd.LANES_KERNEL).read_text()
MODEL_SOURCE = Path(__file__).resolve().parents[1] / "ttnn" / "model.py"


def test_registered_bitwise():
    entry = fused.kernel("position_derive")
    assert entry.tolerance == fused.BITWISE and entry.gate is None
    assert entry.fused is pd.derive_fused and entry.composed is pd.derive_composed
    default = pd.derive_fused if "position_derive" in fused.DEFAULT_ON else pd.derive_composed  # the list decides
    assert fused.resolve("position_derive", {}) is default
    assert fused.resolve("position_derive", {fused.OFF_ENV: "position_derive"}) is pd.derive_composed
    assert fused.resolve("position_derive", {fused.ENV: "position_derive"}) is pd.derive_fused
    assert (
        inspect.signature(pd.position_derive).parameters.keys()
        == inspect.signature(pd.position_derive_composed).parameters.keys()
    )


def test_runtime_and_compile_time_arg_contract():
    # 21 tensors of the chain plus the rotary shift appended as the last input (arg 21): the output addresses keep
    # their argument numbers and the shift is read beside P
    assert len(pd.RUNTIME_ARGS) == 22 and pd.QSA_OUTPUTS == pd.RUNTIME_ARGS[6:15]
    assert pd.OUTPUT_ARGS == pd.RUNTIME_ARGS[6:21] and pd.RUNTIME_ARGS[21] == "rope_shift"
    used = sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", SOURCE)})
    assert used == list(range(22))
    assert "TensorAccessorArgs<0>()" in SOURCE and SOURCE.count("next_compile_time_args_offset()") == 21
    assert "const uint32_t rope_p = p - shift;" in SOURCE and "rope_block_start = block_start - shift;" in SOURCE
    assert SOURCE.count("ROPE_ROW_BYTES, rope_p, 0,") == 2 and SOURCE.count("ROPE_ROW_BYTES, rope_block_start, 0,") == 2
    named = set(re.findall(r'get_named_compile_time_arg_val\("([a-z_0-9]+)"\)', SOURCE))
    python_named = set(
        re.findall(r'"([a-z_0-9]+)": ', inspect.getsource(pd.position_derive).split("named = {")[1].split("}")[0])
    )
    assert named == python_named, named ^ python_named
    source = re.sub(r"\s+", "", inspect.getsource(pd.position_derive))
    assert (
        "tensors=([position,bf16_tpl,u32_tpl,tile_tpl,cos_table,sin_table]+[outs[name]fornameinOUTPUT_ARGS]+[rope_shift])"
        in source
    )
    assert "[(core,[t.buffer_address()fortintensors])]" in source


def test_staging_layout_fits_the_cb():
    blocks, slots = 8192, 2080
    bf16_row, u32_row = blocks * 2, slots * 4
    stage_u32 = bf16_row
    stage_tiles = stage_u32 + ((u32_row + 63) & ~63)
    stage_rope = stage_tiles + 2 * 2048
    stage_index = stage_rope + 4 * 128
    stage_bytes = stage_index + 2 * 128 + 64
    assert stage_bytes <= pd.STAGE_PAGES * pd.STAGE_PAGE_BYTES
    for name, value in (
        ("STAGE_U32 = STAGE_BF16 + BF16_ROW_BYTES", None),
        ("STAGE_TILES = STAGE_U32 + ((U32_ROW_BYTES + 63) & ~63u)", None),
    ):
        assert name in SOURCE
    assert all(offset % 16 == 0 for offset in (stage_u32, stage_tiles, stage_rope, stage_index))  # L1 NoC alignment


def _tile_words(matrix: torch.Tensor) -> torch.Tensor:
    faces = [matrix[r : r + 16, c : c + 16].reshape(-1) for r in (0, 16) for c in (0, 16)]
    return torch.cat(faces)


def test_one_hot_lane_formula_matches_the_tile_layout():
    assert (
        "(kv_row >> 4) * 512 + (kv_row & 15) * 16" in SOURCE
        and "(ring_row >> 4) * 512 + (ring_row & 15) * 16" in SOURCE
    )
    tile = torch.arange(1, 32 * 32 + 1).reshape(32, 32)
    words = _tile_words(tile)
    for row in range(32):
        assert words[(row >> 4) * 512 + (row & 15) * 16] == tile[row, 0]


def test_geometry_formulas_match_the_selection_geometry():
    qsa = _qsa()

    assert "const uint32_t complete_blocks = context >> 2;" in SOURCE
    assert (
        "const uint32_t lo = selected << 2;" in SOURCE and "const uint32_t hi = lo + (context & RING_MASK);" in SOURCE
    )
    assert "const uint32_t tail_shift = (complete_blocks - selected) << 2;" in SOURCE
    for position in (0, 1, 3, 4, 5, 31, 32, 2047, 2048, 2049, 2051, 2052, 8191, 8192, 32767):
        geometry = qsa.qsa_selection_geometry(position + 1)
        context = position + 1
        complete_blocks = context >> 2
        selected = min(complete_blocks, qsa.BLOCK_TOPK)
        lo, hi = selected << 2, (selected << 2) + (context & 3)
        assert (lo, hi - lo, (complete_blocks - selected) << 2) == (
            geometry.complete_token_count,
            geometry.tail_count,
            geometry.tail_start - geometry.complete_token_count,
        )


def test_row_fill_regions_match_the_emulator():
    qsa = _qsa()

    slots = torch.arange(qsa.SPARSE_INDEX_CAPACITY, dtype=torch.int64)
    for position in (0, 3, 4, 33, 2047, 2048, 2050, 8191, 32767):
        ref = qsa.emulate_qsa_position_inputs(position, allocated_compressed_blocks=8192)
        context = position + 1
        cb = context >> 2
        sel = min(cb, qsa.BLOCK_TOPK)
        lo, hi, shift = sel << 2, (sel << 2) + (context & 3), (cb - sel) << 2
        fill = torch.where(
            slots >= hi,
            torch.full_like(slots, qsa.ALL_ONES_U32),
            torch.where(slots >= lo, slots + shift, torch.zeros_like(slots)),
        )
        keep = torch.where(slots < lo, torch.full_like(slots, qsa.ALL_ONES_U32), torch.zeros_like(slots))
        assert torch.equal(fill, ref["row_fill"].reshape(-1)) and torch.equal(keep, ref["row_keep_bits"].reshape(-1))
        assert int(ref["kv_block_start"].reshape(-1)[0]) == position & qsa.KV_BLOCK_START_MASK
        assert int(ref["block_index_i32"].reshape(-1)[0]) == position >> 2


def _calls(function: ast.FunctionDef) -> list[str]:
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr not in ("deallocate",)
    ]
    return [ast.unparse(node.func) for node in sorted(calls, key=lambda n: (n.lineno, n.col_offset))]


ADVANCE_SOURCE = (fp.REPO_ROOT / pd.ADVANCE_KERNEL).read_text()
CONTRACTS_SOURCE = Path(__file__).resolve().parents[1] / "ttnn" / "contracts.py"


def test_advance_is_registered_and_is_the_chains_in_place_add():
    entry = fused.kernel("position_advance")
    assert entry.tolerance == fused.BITWISE and entry.gate is None and entry.default_on is False
    assert entry.fused is pd.advance and entry.composed is pd.advance_composed
    assert fused.resolve("position_advance", {}) is pd.advance_composed
    assert fused.resolve("position_advance", {fused.ENV: "position_advance"}) is pd.advance
    assert inspect.signature(pd.advance).parameters.keys() == inspect.signature(pd.advance_composed).parameters.keys()
    used = sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", ADVANCE_SOURCE)})
    assert used == [0] and ADVANCE_SOURCE.count("TensorAccessorArgs<") == 1
    named = set(re.findall(r'get_named_compile_time_arg_val\("([a-z_0-9]+)"\)', ADVANCE_SOURCE))
    assert named == {"cb_stage", "count"}
    source = inspect.getsource(pd.advance)
    assert set(re.findall(r'"([a-z_0-9]+)": ', source.split("named={")[1].split("}")[0])) == named
    assert (
        "words[0] = words[0] + ADVANCE_COUNT;" in ADVANCE_SOURCE
    )  # the uint32 add of the chain's ttnn.add(scalar, count)
    assert "noc.async_write(stage, position, SCALAR_BYTES," in ADVANCE_SOURCE  # in place, the 4-byte page
    assert "\n        [position, position]," in source  # the resident scalar is the input and the output (in place)
    composed = inspect.getsource(pd.advance_composed)
    assert "ttnn.add(position, count, memory_config=ttnn.DRAM_MEMORY_CONFIG)" in composed
    assert "ttnn.copy(advanced, position)" in composed


def test_composed_is_the_model_bodys_derive_lines():
    source = MODEL_SOURCE.read_text(encoding="utf-8")
    body = source[source.index("index_row = state.position.index_row()") :]
    body = body[: body.index("residual = head.residual")]
    for line in (
        "rope_index_row = state.position.rope_index_row(index_row)",
        "rope_block_start_row = state.position.rope_block_start_index_row(index_row)",
        "rope = self.rope_table.rows(rope_index_row, rope_block_start_row)",
        "_deallocate_unique(index_row, rope_index_row, rope_block_start_row)",
        "derive_qsa_position_inputs(state.position.scalar, self.qsa_position_constants)",
    ):
        assert line in body, line
    composed = inspect.getsource(pd.derive_composed)
    for call in (
        "state.position.index_row()",
        "state.position.rope_index_row(index_row)",
        "state.position.rope_block_start_index_row(index_row)",
        "model.rope_table.rows(rope_index_row, rope_block_start_row)",
        "_deallocate_unique(index_row, rope_index_row, rope_block_start_row)",
        "derive_qsa_position_inputs(state.position.scalar, model.qsa_position_constants)",
    ):
        assert call in composed, call


def test_templates_are_the_chains_constants():
    qsa = _qsa()

    source = inspect.getsource(pd.prepare)
    assert (
        "torch.full((blocks,), qsa.INDEXER_MASK_VALUE)" in source
        and "qsa.ALL_ONES_U32" in source
        and "tiles[:, TILE] = 1.0" in source
    )
    assert (
        pd.ONE_BF16 == 0x3F80
        and torch.tensor(qsa.INDEXER_MASK_VALUE).to(torch.bfloat16).view(torch.int16).item() == -129
    )  # 0xFF7F


def test_lane_form_runtime_and_compile_time_arg_contract():
    """derive_lanes.cpp: 19 tensors (the lanes' rotary shift row last) then the lane and the lane count; the same named constants as derive.cpp; one
    core per lane row; the lane offsets ride the fill's tail ids; the model's lane body binds it only with the six QSA
    programs (its outputs are the fused QSA lane body's inputs)."""

    assert len(pd.LANES_RUNTIME_ARGS) == 19 and pd.LANES_QSA_OUTPUTS == pd.LANES_RUNTIME_ARGS[7:12]
    assert pd.LANES_OUTPUT_ARGS == pd.LANES_RUNTIME_ARGS[7:18] and pd.LANES_RUNTIME_ARGS[18] == "rope_shift_row"
    used = sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", LANES_SOURCE)})
    assert used == list(range(21))
    assert "TensorAccessorArgs<0>()" in LANES_SOURCE and LANES_SOURCE.count("next_compile_time_args_offset()") == 18
    assert "const uint32_t shift = words[ROW_SH / 4 + lane];" in LANES_SOURCE
    assert (
        LANES_SOURCE.count("ROPE_ROW_BYTES, rope_p, 0,") == 2
        and LANES_SOURCE.count("ROPE_ROW_BYTES, rope_block_start, 0,") == 2
    )
    named = set(re.findall(r'get_named_compile_time_arg_val\("([a-z_0-9]+)"\)', LANES_SOURCE))
    assert named == set(re.findall(r'get_named_compile_time_arg_val\("([a-z_0-9]+)"\)', SOURCE))
    python_named = set(
        re.findall(r'"([a-z_0-9]+)": ', inspect.getsource(pd.position_derive_lanes).split("named = {")[1].split("}")[0])
    )
    assert named == python_named, named ^ python_named
    assert "words[STAGE_U32 / 4 + k] = k + offset + tail_shift;" in LANES_SOURCE  # lane u's KV region offset
    assert "if (lane < lane_count) {" in LANES_SOURCE and "if (lane == 0) {" in LANES_SOURCE
    source = re.sub(r"\s+", "", inspect.getsource(pd.position_derive_lanes))
    assert "cores=[ttnn.CoreCoord(u%LANE_COLUMNS,u//LANE_COLUMNS)foruinrange(TILE)]" in source
    assert "[(core,[t.buffer_address()fortintensors]+[u,lanes])foru,coreinenumerate(cores)]" in source
    fused_lanes = inspect.getsource(pd.derive_lanes_fused)
    assert "state.qsa_lane_constants.kv_offsets_row" in fused_lanes and "Qwen38TTNNQSAFusedLaneInputs(" in fused_lanes
    composed = inspect.getsource(pd.derive_lanes_composed)
    for call in (
        "state.position.index_row()",
        "state.position.rope_index_row(index_row)",
        "state.position.rope_block_start_index_row(index_row)",
        "model.rope_table.rows_chunk(rope_index_row, rope_block_start_row)",
        "derive_qsa_lane_inputs(",
    ):
        assert call in composed, call
    model = MODEL_SOURCE.read_text(encoding="utf-8")
    init = model[model.index('if fused_kernels.enabled("position_derive"):') :]
    init = init[: init.index("self._state_owner = object()")]
    assert "if all(fused_kernels.enabled(name) for name in QSA_LANE_FUSED_KERNELS):" in init
    assert "self._position_derive_lanes = fused_kernels.position_derive.derive_lanes_fused" in init
    body = model[model.index("    def forward_decode_lanes(") : model.index("    def resolve_lane_tokens(")]
    assert "rope, qsa_lanes = self._position_derive_lanes(self, state)" in body
    assert "qsa_lanes = qsa_module.derive_qsa_lane_inputs(" in body  # the chain stays the fallback
    stage_u32 = 8192 * 2
    stage_tile = stage_u32 + ((2080 * 4 + 63) & ~63)
    stage_rows = stage_tile + 2048 + 4 * 128
    assert stage_rows + 5 * 128 <= pd.LANES_STAGE_PAGES * pd.STAGE_PAGE_BYTES
