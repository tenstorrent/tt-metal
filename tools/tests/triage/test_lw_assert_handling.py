# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for callstack_provider and dump_lightweight_asserts helper functions.
These tests use mocks and temp files—no device or hardware required.
"""

import os
import sys
import tempfile
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

metal_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
triage_home = os.path.join(metal_home, "tools", "triage")
sys.path.insert(0, triage_home)

from dump_lightweight_asserts import extract_assert_code
from callstack_provider import get_function_die, extract_template_params
from ttexalens.hardware.risc_debug import CallstackEntry, CallstackEntryVariable


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_die(tag, name=None, parent=None, children=None, attributes=None, value=None, resolved_type=None):
    """Build a lightweight mock that quacks like an ElfDie."""
    die = MagicMock()
    die.tag = tag
    die.name = name
    die.parent = parent
    die.value = value
    die.attributes = attributes or {}
    die.resolved_type = resolved_type if resolved_type is not None else die

    child_list = children or []
    die.iter_children = MagicMock(return_value=iter(child_list))
    die.get_DIE_from_attribute = MagicMock(return_value=None)
    return die


def _make_variable(die):
    """Build a CallstackEntryVariable with a mock die and None value."""
    return CallstackEntryVariable(die=die, value=None)


def _make_entry(arguments=None, locals_=None):
    """Build a CallstackEntry with the given arguments and locals."""
    return CallstackEntry(
        pc=0x1000,
        function_name="test_func",
        file="test.cpp",
        line=10,
        arguments=arguments or [],
        locals=locals_ or [],
    )


# ===========================================================================
# extract_assert_code
# ===========================================================================


class TestExtractAssertCode:
    """Tests for the assert-code extraction logic in dump_lightweight_asserts."""

    def test_single_line_assert(self, tmp_path):
        src = tmp_path / "test.cpp"
        src.write_text("    ASSERT(x > 0);\n")
        assert extract_assert_code(str(src), 1, None) == "ASSERT(x > 0)"

    def test_prefixed_assert_single_line(self, tmp_path):
        src = tmp_path / "test.cpp"
        src.write_text("    LLK_ASSERT(x > 0);\n")
        assert extract_assert_code(str(src), 1, None) == "LLK_ASSERT(x > 0)"

    def test_prefixed_assert_with_column(self, tmp_path):
        """LLK_ASSERT at column 5 (1-indexed) must be found despite 'ASSERT(' appearing later."""
        src = tmp_path / "test.cpp"
        #                 12345
        src.write_text("    LLK_ASSERT(x > 0);\n")
        result = extract_assert_code(str(src), 1, 5)
        assert result == "LLK_ASSERT(x > 0)"

    def test_multi_line_assert(self, tmp_path):
        src = tmp_path / "test.cpp"
        src.write_text("    ASSERT(\n" "        x > 0\n" "    );\n")
        result = extract_assert_code(str(src), 1, None)
        assert "ASSERT(" in result
        assert "x > 0" in result
        assert result.endswith(")")

    def test_multi_line_prefixed_assert(self, tmp_path):
        src = tmp_path / "test.cpp"
        src.write_text("    LLK_ASSERT(\n" "        (is_configured(a, b)),\n" '        "error");\n')
        result = extract_assert_code(str(src), 1, 5)
        assert result.startswith("LLK_ASSERT(")
        assert "is_configured(a, b)" in result
        assert result.endswith(")")

    def test_nested_parens_multi_line(self, tmp_path):
        src = tmp_path / "test.cpp"
        src.write_text("    ASSERT(foo(bar(x),\n" "               baz(y)));\n")
        result = extract_assert_code(str(src), 1, None)
        assert result.startswith("ASSERT(")
        assert "foo(" in result
        assert "baz(y)" in result
        assert result.endswith(")")

    def test_none_file(self):
        assert extract_assert_code(None, 1, None) == "?"

    def test_none_line(self, tmp_path):
        src = tmp_path / "test.cpp"
        src.write_text("ASSERT(1);\n")
        assert extract_assert_code(str(src), None, None) == "?"

    def test_file_not_found(self):
        assert extract_assert_code("/nonexistent/path.cpp", 1, None) == "?file not found?"

    def test_wrong_line_number(self, tmp_path):
        src = tmp_path / "test.cpp"
        src.write_text("ASSERT(1);\n")
        result = extract_assert_code(str(src), 999, None)
        assert "wrong line number" in result

    def test_no_assert_on_line(self, tmp_path):
        src = tmp_path / "test.cpp"
        src.write_text("    int x = 42;\n")
        result = extract_assert_code(str(src), 1, None)
        assert "not found" in result

    def test_column_disambiguates_multiple_asserts(self, tmp_path):
        src = tmp_path / "test.cpp"
        # Two asserts on one line
        src.write_text("ASSERT(a); ASSERT(b);\n")
        # Column 1 (1-indexed) points to the first ASSERT -> selects it
        assert extract_assert_code(str(src), 1, 1) == "ASSERT(a)"
        # Column 12 (1-indexed) points to the second ASSERT -> selects it
        assert extract_assert_code(str(src), 1, 12) == "ASSERT(b)"

    def test_unclosed_assert_no_more_lines(self, tmp_path):
        src = tmp_path / "test.cpp"
        src.write_text("    ASSERT(x > 0\n")
        result = extract_assert_code(str(src), 1, None)
        assert "closing paren not found" in result


# ===========================================================================
# get_function_die
# ===========================================================================


class TestGetFunctionDie:
    """Tests for navigating from a CallstackEntry to the parent function die."""

    def test_finds_subprogram_parent(self):
        func_die = _make_die("DW_TAG_subprogram", name="my_func")
        param_die = _make_die("DW_TAG_formal_parameter", name="x", parent=func_die)
        entry = _make_entry(arguments=[_make_variable(param_die)])

        result = get_function_die(entry)
        assert result is func_die

    def test_walks_through_lexical_block(self):
        func_die = _make_die("DW_TAG_subprogram", name="my_func")
        block_die = _make_die("DW_TAG_lexical_block", parent=func_die)
        var_die = _make_die("DW_TAG_variable", name="tmp", parent=block_die)
        entry = _make_entry(locals_=[_make_variable(var_die)])

        result = get_function_die(entry)
        assert result is func_die

    def test_inlined_subroutine_returns_abstract_origin(self):
        origin_die = _make_die("DW_TAG_subprogram", name="inlined_func")
        inlined_die = _make_die("DW_TAG_inlined_subroutine")
        inlined_die.get_DIE_from_attribute = MagicMock(return_value=origin_die)
        param_die = _make_die("DW_TAG_formal_parameter", name="x", parent=inlined_die)
        entry = _make_entry(arguments=[_make_variable(param_die)])

        result = get_function_die(entry)
        assert result is origin_die

    def test_inlined_subroutine_no_abstract_origin_returns_self(self):
        inlined_die = _make_die("DW_TAG_inlined_subroutine")
        inlined_die.get_DIE_from_attribute = MagicMock(return_value=None)
        param_die = _make_die("DW_TAG_formal_parameter", name="x", parent=inlined_die)
        entry = _make_entry(arguments=[_make_variable(param_die)])

        result = get_function_die(entry)
        assert result is inlined_die

    def test_empty_entry_returns_none(self):
        entry = _make_entry()
        assert get_function_die(entry) is None

    def test_variable_with_no_die_is_skipped(self):
        var_no_die = MagicMock(spec=CallstackEntryVariable)
        del var_no_die.die  # getattr(var, "die", None) -> None
        entry = _make_entry(arguments=[var_no_die])
        assert get_function_die(entry) is None

    def test_uses_locals_when_no_arguments(self):
        func_die = _make_die("DW_TAG_subprogram", name="my_func")
        local_die = _make_die("DW_TAG_variable", name="local_var", parent=func_die)
        entry = _make_entry(locals_=[_make_variable(local_die)])

        result = get_function_die(entry)
        assert result is func_die


# ===========================================================================
# extract_template_params
# ===========================================================================


class TestExtractTemplateParams:
    """Tests for extracting template parameters from a function die."""

    def test_type_param(self):
        type_die = _make_die("DW_TAG_base_type", name="int")
        type_die.resolved_type = type_die
        child = _make_die("DW_TAG_template_type_param", name="T", resolved_type=type_die)
        func_die = _make_die("DW_TAG_subprogram", children=[child])

        result = extract_template_params(func_die)
        assert result == [("T", "int")]

    def test_value_param(self):
        child = _make_die("DW_TAG_template_value_param", name="N", value=42)
        func_die = _make_die("DW_TAG_subprogram", children=[child])

        result = extract_template_params(func_die)
        assert result == [("N", "42")]

    def test_value_param_bool_false(self):
        child = _make_die("DW_TAG_template_value_param", name="flag", value=0)
        func_die = _make_die("DW_TAG_subprogram", children=[child])

        result = extract_template_params(func_die)
        assert result == [("flag", "0")]

    def test_mixed_params(self):
        type_die = _make_die("DW_TAG_base_type", name="float")
        type_die.resolved_type = type_die
        type_child = _make_die("DW_TAG_template_type_param", name="T", resolved_type=type_die)
        value_child = _make_die("DW_TAG_template_value_param", name="N", value=8)
        func_die = _make_die("DW_TAG_subprogram", children=[type_child, value_child])

        result = extract_template_params(func_die)
        assert result == [("T", "float"), ("N", "8")]

    def test_no_template_params(self):
        param_child = _make_die("DW_TAG_formal_parameter", name="x")
        func_die = _make_die("DW_TAG_subprogram", children=[param_child])

        result = extract_template_params(func_die)
        assert result == []

    def test_gnu_template_parameter_pack(self):
        type_die = _make_die("DW_TAG_base_type", name="int")
        type_die.resolved_type = type_die
        pack_type = _make_die("DW_TAG_template_type_param", name="T", resolved_type=type_die)
        pack_value = _make_die("DW_TAG_template_value_param", name="V", value=99)
        pack = _make_die("DW_TAG_GNU_template_parameter_pack", children=[pack_type, pack_value])

        func_die = _make_die("DW_TAG_subprogram", children=[pack])

        result = extract_template_params(func_die)
        assert result == [("T", "int"), ("V", "99")]

    def test_specification_redirect(self):
        type_die = _make_die("DW_TAG_base_type", name="double")
        type_die.resolved_type = type_die
        child = _make_die("DW_TAG_template_type_param", name="T", resolved_type=type_die)
        spec_die = _make_die("DW_TAG_subprogram", children=[child])

        func_die = _make_die("DW_TAG_subprogram", attributes={"DW_AT_specification": True})
        func_die.get_DIE_from_attribute = MagicMock(return_value=spec_die)

        result = extract_template_params(func_die)
        assert result == [("T", "double")]

    def test_type_param_none_name_resolved(self):
        """type_die.name is None -> should fall back to '?'."""
        type_die = _make_die("DW_TAG_base_type", name=None)
        type_die.resolved_type = type_die
        child = _make_die("DW_TAG_template_type_param", name="T", resolved_type=type_die)
        func_die = _make_die("DW_TAG_subprogram", children=[child])

        result = extract_template_params(func_die)
        assert result == [("T", "?")]

    def test_value_param_none_value(self):
        child = _make_die("DW_TAG_template_value_param", name="N", value=None)
        func_die = _make_die("DW_TAG_subprogram", children=[child])

        result = extract_template_params(func_die)
        assert result == [("N", "?")]

    def test_type_param_resolved_to_self(self):
        """When resolved_type is the child itself, treat as unresolvable."""
        child = _make_die("DW_TAG_template_type_param", name="T")
        child.resolved_type = child  # self-referencing
        func_die = _make_die("DW_TAG_subprogram", children=[child])

        result = extract_template_params(func_die)
        assert result == [("T", "?")]

    def test_param_name_is_none(self):
        child = _make_die("DW_TAG_template_value_param", name=None, value=7)
        func_die = _make_die("DW_TAG_subprogram", children=[child])

        result = extract_template_params(func_die)
        assert result == [(None, "7")]
