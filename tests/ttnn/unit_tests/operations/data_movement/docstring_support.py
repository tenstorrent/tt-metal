# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Read a ttnn op's supported dtypes, layouts and interleaved buffer types from its docstring, so a test can run the
op on exactly those.

The helpers read the dtype/layout table and the interleaved Memory Support item. Sharded placement is not read.
A test parametrized with them follows the docstring: a dtype, layout or buffer type added to the docstring must work,
and one the op stops supporting fails until the docstring drops it.

The helpers run while pytest collects. If a docstring can't be read, they return a single case that fails with the
reason, so the rest of the test module still collects and runs.
"""

import re

import pytest
import ttnn

_LAYOUTS = {"TILE": ttnn.TILE_LAYOUT, "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT}


def _list_table_rows(doc, caption):
    """Cells of each row of the ``.. list-table::`` that directly follows ``caption``, header row included.

    The first non-blank line after the caption must be the directive, so a caption without its own table is never
    matched to a later one. Blank lines between rows are skipped. The table ends at the first non-blank line that is
    not indented under the directive, or that is neither one of its rows nor indented under one, so text after the
    table is never read as part of it, even when the table has no rows.
    """
    lines = [line for line in doc[doc.index(caption) :].splitlines()[1:] if line.strip()]
    assert lines and lines[0].strip().startswith(".. list-table::"), f"no list-table directly after '{caption}'"
    directive_indent = len(lines[0]) - len(lines[0].lstrip())
    rows = []
    row_indent = None
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= directive_indent:
            break
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


def _memory_support_items(doc):
    """The items of the docstring's Memory Support list, each with its wrapped continuation lines joined into it.

    The list ends at the first non-blank line that is neither one of its items nor indented under one.
    """
    items = []
    item_indent = None
    for line in doc.split("Memory Support:", 1)[1].splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("- ") and item_indent in (None, indent):
            item_indent = indent
            items.append(stripped[len("- ") :].strip())
        elif items and indent > item_indent:
            items[-1] += " " + stripped
        else:
            break
    return items


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
    """One ``pytest.param(dtype, layout)`` per combination in the docstring table that follows ``caption``.

    The table must start with a ``Dtypes`` / ``Layouts`` header row, so a table without that header fails the test
    instead of losing its first row.
    """
    try:
        doc = op.__doc__
        params = []
        rows = _list_table_rows(doc, caption)
        assert all(len(row) == 2 for row in rows), f"expected two cells per row after '{caption}', got {rows}"
        assert rows[:1] == [["Dtypes", "Layouts"]], f"no Dtypes/Layouts header row after '{caption}', got {rows[:1]}"
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
    """One ``pytest.param(buffer_type)`` per buffer type on the docstring's interleaved Memory Support item.

    ``label`` picks the item that starts with it, such as ``"Output"`` when the docstring lists input and output
    placement in separate items. By default the first item that mentions interleaved placement is used. An item
    wrapped over several lines is read as one line.

    Every word in the interleaved clause other than ``and`` and ``or`` must name a ``ttnn.BufferType``. Any other
    word fails the test, so no buffer type the docstring lists is left untested.
    """
    try:
        items = [item for item in _memory_support_items(op.__doc__) if re.search(r"\binterleaved\b", item, re.I)]
        if label is not None:
            items = [item for item in items if item.startswith(label)]
        line = items[0]
        clause = re.split(r";|\.(?:\s|$)", line.split("nterleaved", 1)[1])[0]
        names = [word for word in re.findall(r"[^\s,:]+", clause) if word not in ("and", "or")]
        assert names, f"no buffer type on the interleaved line: {line!r}"
        unknown = [name for name in names if not isinstance(getattr(ttnn.BufferType, name, None), ttnn.BufferType)]
        assert not unknown, f"not a ttnn.BufferType: {unknown} on the interleaved line: {line!r}"
        return [pytest.param(getattr(ttnn.BufferType, name), id=name) for name in dict.fromkeys(names)]
    except Exception as error:
        return _unreadable_params(op, "interleaved Memory Support line", error, arity=1)


def interleaved_memory_config(buffer_type):
    check_readable(buffer_type)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)
