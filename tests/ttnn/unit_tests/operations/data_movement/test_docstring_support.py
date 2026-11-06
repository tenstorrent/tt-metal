# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the docstring parsing in docstring_support.py, fed with fake docstrings."""

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    UnreadableDocstring,
    check_readable,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
)

HEADER = """\
               * - Dtypes
                 - Layouts
"""
ROWS = """\
               * - BFLOAT16, FLOAT32
                 - TILE, ROW_MAJOR

               * - BFLOAT8_B
                 - TILE
"""
DIRECTIVE = """
            .. list-table::
               :header-rows: {}

"""
TABLE = DIRECTIVE.format(1) + HEADER + ROWS
ALL_PAIRS = ["BFLOAT16-TILE", "BFLOAT16-ROW_MAJOR", "FLOAT32-TILE", "FLOAT32-ROW_MAJOR", "BFLOAT8_B-TILE"]


def fake_op(doc):
    return type("Op", (), {"__doc__": doc})


def note(*sections):
    return "Does something.\n\n        Note:\n" + "".join(sections)


def caption(text="Supported dtypes and layouts"):
    return f"            {text}:\n"


def memory_support(*items):
    return "Does something.\n\n        Memory Support:\n" + "".join(f"            {item}\n" for item in items)


def ids(params):
    return [param.id for param in params]


def assert_unreadable(params, what, reason, expect_error):
    assert len(params) == 1, ids(params)
    assert all(isinstance(value, UnreadableDocstring) for value in params[0].values), params[0].values
    assert f"could not read the {what}" in params[0].values[0]
    assert reason in params[0].values[0]
    with expect_error(pytest.fail.Exception, f"could not read the {what}"):
        check_readable(*params[0].values)


def test_dtype_layout_table_gives_every_pair():
    params = docstring_dtype_layout_params(fake_op(note(caption(), TABLE)))
    assert ids(params) == ALL_PAIRS
    assert params[0].values == (ttnn.bfloat16, ttnn.TILE_LAYOUT)
    assert params[-1].values == (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)


def test_dtype_layout_table_joins_a_wrapped_cell():
    table = TABLE.replace("* - BFLOAT16, FLOAT32", "* - BFLOAT16,\n                   FLOAT32")
    assert ids(docstring_dtype_layout_params(fake_op(note(caption(), table)))) == ALL_PAIRS


def test_dtype_layout_table_ends_before_the_text_after_it():
    after = "\n            A list after the table:\n\n            * - UINT32\n              - TILE\n"
    assert ids(docstring_dtype_layout_params(fake_op(note(caption(), TABLE, after)))) == ALL_PAIRS


def test_dtype_layout_table_follows_the_requested_caption():
    index_table = TABLE.replace("BFLOAT8_B", "UINT32")
    doc = note(caption("Input dtypes"), TABLE, "\n", caption("Index dtypes"), index_table)
    assert ids(docstring_dtype_layout_params(fake_op(doc), "Input dtypes"))[-1] == "BFLOAT8_B-TILE"
    assert ids(docstring_dtype_layout_params(fake_op(doc), "Index dtypes"))[-1] == "UINT32-TILE"


@pytest.mark.parametrize(
    "doc, reason",
    [
        (note(caption(), DIRECTIVE.format(0) + ROWS), "no Dtypes/Layouts header row"),
        (note(caption(), DIRECTIVE.format(1), caption("Index dtypes"), TABLE), "no Dtypes/Layouts header row"),
        (note(caption(), "\n", caption("Index dtypes"), TABLE), "no list-table directly after"),
        (note(caption(), TABLE.replace("BFLOAT8_B", "BFLOAT9_B")), "BFLOAT9_B"),
        (note(caption(), TABLE.replace("TILE, ROW_MAJOR", "TILE, BLOCK")), "BLOCK"),
        (note(caption(), TABLE + "                 - DRAM\n"), "expected two cells per row"),
        (note(TABLE), ""),
    ],
    ids=[
        "no-header-row",
        "empty-table-before-another",
        "caption-without-its-table",
        "unknown-dtype",
        "unknown-layout",
        "three-cells",
        "no-caption",
    ],
)
def test_dtype_layout_table_that_cannot_be_read_fails(doc, reason, expect_error):
    assert_unreadable(docstring_dtype_layout_params(fake_op(doc)), "dtype/layout table", reason, expect_error)


@pytest.mark.parametrize(
    "items, label, expected",
    [
        (["- Interleaved: DRAM and L1"], None, ["DRAM", "L1"]),
        (["- Interleaved: DRAM, L1 or L1_SMALL"], None, ["DRAM", "L1", "L1_SMALL"]),
        (["- Interleaved: L1 and DRAM; DRAM is the default"], None, ["L1", "DRAM"]),
        (["- Interleaved: DRAM and", "  L1"], None, ["DRAM", "L1"]),
        (["- Input: interleaved DRAM and L1; HEIGHT_SHARDED in L1", "- Output: interleaved DRAM"], "Output", ["DRAM"]),
        (["- Input: interleaved DRAM", "- Output: interleaved DRAM", "  or L1"], "Output", ["DRAM", "L1"]),
    ],
    ids=["dram-and-l1", "l1-small", "clause-ends-at-semicolon", "wrapped-item", "label", "wrapped-labelled-item"],
)
def test_interleaved_buffer_types(items, label, expected):
    params = docstring_interleaved_buffer_type_params(fake_op(memory_support(*items)), label=label)
    assert ids(params) == expected
    assert [param.values[0] for param in params] == [getattr(ttnn.BufferType, name) for name in expected]


SHARDED_ONLY = memory_support("- Sharded: WIDTH_SHARDED in DRAM")


@pytest.mark.parametrize(
    "doc, reason",
    [
        (memory_support("- Interleaved: DRAM and SRAM"), "not a ttnn.BufferType: ['SRAM']"),
        (memory_support("- Interleaved: DRAM and", "  SRAM"), "not a ttnn.BufferType: ['SRAM']"),
        (memory_support("- Interleaved: DRAM and L1 (TILE only)"), "not a ttnn.BufferType"),
        (SHARDED_ONLY, ""),
        (SHARDED_ONLY + "\n        Example:\n            Interleaved: DRAM and L1\n", ""),
        ("Does something.\n", ""),
    ],
    ids=["unknown-name", "wrapped-unknown-name", "extra-words", "no-interleaved-item", "text-after-list", "no-section"],
)
def test_interleaved_buffer_types_that_cannot_be_read_fail(doc, reason, expect_error):
    params = docstring_interleaved_buffer_type_params(fake_op(doc))
    assert_unreadable(params, "interleaved Memory Support line", reason, expect_error)


def test_check_readable_passes_real_values():
    check_readable(ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM)
