# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for check_bool_from_arithmetic.py."""

import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_bool_from_arithmetic as chk  # noqa: E402


def run(tmp_path, body, capsys, baseline=None, name="f.cpp"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    argv = [str(p)]
    if baseline is not None:
        argv += ["--baseline", str(baseline)]
    code = chk.main(argv)
    return code, capsys.readouterr().out


@pytest.mark.parametrize(
    "decl",
    [
        "bool x = a / b;",
        "const bool x = a * b;",
        "constexpr bool x = N % 4;",
        "static bool x = a / b;",
        "bool x{a / b};",
        "bool x = (a + b - 1) / b;",
        "bool x = ((a + b - 1) / b);",
        "bool x = -a / b;",
        "bool x = static_cast<int>(a) / b;",
        "bool x = (int)a / b;",
        "bool x = s.size() / n;",
        "bool x = v[i] % n;",
        "bool x = a * *p;",
        "void f(bool x = a / 2);",
        "bool const x = a / b;",
        "static bool const x = a * b;",
        "bool volatile x{a % n};",
    ],
)
def test_flagged(tmp_path, capsys, decl):
    code, out = run(tmp_path, f"void g() {{\n    {decl}\n}}\n", capsys)
    assert code == 1
    assert "multiplicative expression" in out


@pytest.mark.parametrize(
    "decl",
    [
        "bool x = f(a / b);",
        "bool x = a / b > 0;",
        "bool x = (a / b) != 0;",
        "bool x = a % 2 == 0;",
        "bool x = a / b && c;",
        "bool x = a / b || c;",
        "bool x = c ? a / b : 0;",
        "bool x = flags & MASK;",
        "bool x = a / b + c;",
        "bool x = a - b;",
        "bool x = a << b;",
        "bool x = *p;",
        "bool x = !p;",
        "bool x = std::any_of(v.begin(), v.end(), [](int i) { return i % 2; });",
        "bool x = a < b * c;",
        "bool x = a < b > c * d;",
        "bool x = (a) < b > (c) * d;",
        "mybool x = a / b;",
        "// bool x = a / b;",
        "/* bool x = a / b; */",
        'const char* s = "bool x = a / b;";',
        "std::vector<bool> x = a / b;",
        "bool x = true;",
        "bool operator==(const T& o) const { return a / b == o.a; }",
        "bool x, y = a;",
        "bool x(a / b);",  # direct-initialization: documented gap, reads as a function declaration
        "bool f(int* p, int n);",
        "bool const& x = a / b;",
    ],
)
def test_clean(tmp_path, capsys, decl):
    code, out = run(tmp_path, f"void g() {{\n    {decl}\n}}\n", capsys)
    assert code == 0, out
    assert out == ""


def test_preprocessor_and_raw_string_are_ignored(tmp_path, capsys):
    body = """
    #define BAD bool x = a / b;
    const char* r = R"(bool y = a / b;)";
    auto w = LR"xy(bool ") z = a / b;)xy";
    int big = 0x8'0000; const char* s = "bool q = a / b;";
    """
    assert run(tmp_path, body, capsys) == (0, "")


def test_code_after_char_and_digit_separator_is_still_checked(tmp_path, capsys):
    code, out = run(tmp_path, "char c = u8'a'; int n = 1'000; bool z = n / 2;\n", capsys)
    assert code == 1
    assert "bool z" in out


def test_reports_the_line(tmp_path, capsys):
    code, out = run(tmp_path, "int a;\n\nbool tiles = n / 32;\n", capsys)
    assert code == 1
    assert ":3: " in out and "bool tiles = n / 32;" in out


def test_baseline_suppresses_by_source_line_not_number(tmp_path, capsys):
    p = tmp_path / "f.cpp"
    rel = os.path.relpath(p, chk.REPO_ROOT)
    base = tmp_path / "baseline.txt"
    base.write_text(f"# comment\n{rel}\tbool x = a / b;\n")
    # Moved down two lines: still matched.
    code, out = run(tmp_path, "\n\nbool x = a / b;\n", capsys, baseline=base)
    assert code == 0, out
    # A second, different site in the same file is not covered.
    code, out = run(tmp_path, "bool x = a / b;\nbool y = a % b;\n", capsys, baseline=base)
    assert code == 1
    assert "bool y" in out and "bool x" not in out


def test_stale_baseline_entry_is_reported_but_does_not_fail(tmp_path, capsys):
    p = tmp_path / "f.cpp"
    rel = os.path.relpath(p, chk.REPO_ROOT)
    base = tmp_path / "baseline.txt"
    base.write_text(f"{rel}\tbool x = a / b;\n")
    code, out = run(tmp_path, "std::uint32_t x = a / b;\n", capsys, baseline=base)
    assert code == 0
    assert "no longer found" in out


def test_missing_file_is_skipped(tmp_path, capsys):
    assert chk.main([str(tmp_path / "nope.cpp")]) == 0
