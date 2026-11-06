# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Read the inputs that a ttnn op's docstring says it supports, so a test can run the op on exactly those.

A test parametrized with these helpers follows the docstring: a dtype, layout or buffer type added to the
docstring must work, and one the op stops supporting fails until the docstring drops it.

The helpers run while pytest collects. If a docstring can't be read, they return a single case that fails with the
reason, so the rest of the test module still collects and runs.
"""

import re

import pytest
import torch
import ttnn

_LAYOUTS = {"TILE": ttnn.TILE_LAYOUT, "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT}


def _list_table_rows(doc, start):
    """Cells of each row of the first ``.. list-table::`` at or after ``start``, header row included.

    Blank lines between rows are skipped. The table ends at the first non-blank line that is neither one of its
    rows nor indented under one, so a list that follows the table is never read as part of it.
    """
    lines = doc[doc.index(".. list-table::", start) :].splitlines()[1:]
    rows = []
    row_indent = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("* -") and row_indent in (None, indent):
            row_indent = indent
            rows.append([stripped[len("* -") :].strip()])
        elif rows and indent > row_indent and stripped.startswith("- "):
            rows[-1].append(stripped[len("- ") :].strip())
        elif rows and indent > row_indent:
            rows[-1][-1] += " " + stripped
        elif rows:
            break
    return rows


class UnreadableDocstring(str):
    """Stands in for a parameter when the docstring can't be read; using it fails the test with the reason."""


def _unreadable_params(op, what, error, arity):
    reason = UnreadableDocstring(f"could not read the {what} from the {op} docstring: {error}")
    return [pytest.param(*([reason] * arity), id="docstring-unreadable")]


def check_readable(*values):
    """Fail the test if any value stands in for an unreadable docstring."""
    for value in values:
        if isinstance(value, UnreadableDocstring):
            pytest.fail(value)


def docstring_dtype_layout_params(op, caption="Supported dtypes and layouts"):
    """One ``pytest.param(dtype, layout)`` per combination in the docstring table that follows ``caption``."""
    try:
        doc = op.__doc__
        params = []
        rows = _list_table_rows(doc, doc.index(caption))
        assert all(len(row) == 2 for row in rows), f"expected two cells per row after '{caption}', got {rows}"
        for dtypes, layouts in rows[1:]:
            for dtype_name in dtypes.split(","):
                for layout_name in layouts.split(","):
                    dtype = getattr(ttnn.DataType, dtype_name.strip())
                    layout = _LAYOUTS[layout_name.strip()]
                    params.append(pytest.param(dtype, layout, id=f"{dtype_name.strip()}-{layout_name.strip()}"))
        assert params, f"no dtype/layout table after '{caption}'"
        return params
    except Exception as error:
        return _unreadable_params(op, "dtype/layout table", error, arity=2)


def docstring_interleaved_buffer_type_params(op, label=None):
    """One ``pytest.param(buffer_type)`` per buffer type on the docstring's interleaved Memory Support line.

    ``label`` picks the line that starts with it, such as ``"Output"`` when the docstring lists input and output
    placement on separate lines. By default the first line that mentions interleaved placement is used.
    """
    try:
        memory_support = op.__doc__.split("Memory Support:", 1)[1]
        lines = [line for line in memory_support.splitlines() if re.search(r"\binterleaved\b", line, re.IGNORECASE)]
        if label is not None:
            lines = [line for line in lines if line.strip().lstrip("- ").startswith(label)]
        line = lines[0]
        clause = re.split(r";|\.\s", line.split("nterleaved", 1)[1])[0]
        names = re.findall(r"\b(DRAM|L1)\b", clause)
        assert names, f"no buffer type on the interleaved line: {line.strip()!r}"
        return [pytest.param(getattr(ttnn.BufferType, name), id=name) for name in dict.fromkeys(names)]
    except Exception as error:
        return _unreadable_params(op, "interleaved Memory Support line", error, arity=1)


def interleaved_memory_config(buffer_type):
    check_readable(buffer_type)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)


def torch_input(shape, dtype, seed=0):
    """Random values that ``dtype`` can hold. Block floats get float32 values, which they quantize."""
    check_readable(dtype)
    torch.manual_seed(seed)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)
    if dtype in (ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b):
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31 - 1, shape, dtype=torch.int32)
    if dtype == ttnn.uint16:
        return torch.randint(0, 2**15, shape, dtype=torch.int16)
    if dtype == ttnn.uint8:
        return torch.randint(0, 2**8, shape, dtype=torch.uint8)
    raise ValueError(f"no torch input for {dtype}")


def assert_exact(expected, actual):
    """Same shape and the same values, compared as float64 so every dtype above compares exactly."""
    assert list(expected.shape) == list(actual.shape), f"shape {list(actual.shape)} != {list(expected.shape)}"
    assert torch.equal(expected.to(torch.float64), actual.to(torch.float64)), "values differ"
