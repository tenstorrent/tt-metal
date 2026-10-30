# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""reference_<name> comparisons: no false identical, no spurious difference."""

import pytest
from conftest import needs_toolchain
from hal_explorer import pipeline

pytestmark = needs_toolchain

NOP, STALL = "TTI_NOP;", "TTI_STALLWAIT(0x80, 0x1);"


def verdicts(tables, code):
    result = pipeline.run("code", code, tables)
    assert result.ok, result.diagnostics
    return {c.name: c for c in result.comparisons}


def verdict(tables, code):
    (comparison,) = verdicts(tables, code).values()
    return comparison


DIFFERENT = {
    "else-if versus two ifs": (
        f"void reference_f(bool a, bool b) {{ if (a) {{ {NOP} }} else if (b) {{ {STALL} }} }}",
        f"void f(bool a, bool b) {{ if (a) {{ {NOP} }} if (b) {{ {STALL} }} }}",
    ),
    "one switch case": (
        "void reference_f(unsigned x) { switch (x) { "
        + " ".join(f"case {k}: TTI_SETDMAREG(0, {k}, 0, 0); break;" for k in range(8))
        + " } }",
        "void f(unsigned x) { switch (x) { "
        + " ".join(
            f"case {k}: TTI_SETDMAREG(0, {k if k != 7 else 9}, 0, 0); break;"
            for k in range(8)
        )
        + " } }",
    ),
    "extra MMIO write": (
        f"void reference_f(unsigned v) {{ {NOP} }}",
        f"void f(unsigned v) {{ {NOP} ckernel::cfg_write(184, v); }}",
    ),
    "loop-carried value": (
        "void reference_f(unsigned n) { unsigned v = 1; for (unsigned i = 0; i < n; i++) { TT_SETDMAREG(0, v & 0x3fff, 0, 0); v = v * 3; } }",
        "void f(unsigned n) { unsigned v = 1; for (unsigned i = 0; i < n; i++) { TT_SETDMAREG(0, v & 0x3fff, 0, 0); v = v * 5; } }",
    ),
    "a difference on one of 2048 paths": (
        "void reference_f("
        + ", ".join(f"bool b{k}" for k in range(11))
        + ") { "
        + " ".join(f"if (b{k}) {{ {NOP} }}" for k in range(11))
        + " }",
        "void f("
        + ", ".join(f"bool b{k}" for k in range(11))
        + ") { "
        + " ".join(f"if (b{k}) {{ {NOP} }}" for k in range(11))
        + f" if (!b0) {{ {STALL} }} }}",
    ),
    "swapped parameters": (
        f"void reference_f(bool a, bool b) {{ if (a) {{ {NOP} }} }}",
        f"void f(bool b, bool a) {{ if (a) {{ {NOP} }} }}",
    ),
    "MMIO access width": (
        "void reference_f(unsigned v) { *(volatile unsigned *)0xFFB80000 = v; }",
        "void f(unsigned v) { *(volatile unsigned char *)0xFFB80000 = v; }",
    ),
    "builtin operand": (
        '#include "sfpi.h"\nvoid reference_f() { __builtin_rvtt_ttincrwc(0, 2, 0, 0); }',
        "void f() { __builtin_rvtt_ttincrwc(0, 4, 0, 0); }",
    ),
}


@pytest.mark.parametrize("name", DIFFERENT)
def test_differences_are_found(tables, name):
    reference, rewrite = DIFFERENT[name]
    comparison = verdict(tables, reference + "\n" + rewrite)
    assert comparison.verdict == "differs", comparison
    assert comparison.counterexample["diff"]


IDENTICAL = {
    "renamed parameters": (
        f"void reference_f(bool a) {{ if (a) {{ {NOP} }} }}",
        f"void f(bool partial_face) {{ if (partial_face) {{ {NOP} }} }}",
    ),
    "or versus else-if": (
        f"void reference_f(bool a, bool b) {{ if (a || b) {{ {NOP} }} }}",
        f"void f(bool a, bool b) {{ if (a) {{ {NOP} }} else if (b) {{ {NOP} }} }}",
    ),
    "immediate versus stored word": (
        "void reference_f() { TTI_SETDMAREG(0, 5, 0, 0); }",
        "void f() { TT_SETDMAREG(0, 5, 0, 0); }",
    ),
    "loop versus unrolled": (
        f"void reference_f() {{ for (unsigned i = 0; i < 40; i++) {{ {NOP} }} }}",
        "void f() { " + NOP * 40 + " }",
    ),
    "min written two ways": (
        "void reference_f(unsigned a, unsigned b) { TT_SETDMAREG(0, (a < b ? a : b) & 0x3fff, 0, 0); }",
        "void f(unsigned a, unsigned b) { TT_SETDMAREG(0, (b < a ? b : a) & 0x3fff, 0, 0); }",
    ),
}


@pytest.mark.parametrize("name", IDENTICAL)
def test_equivalent_code_is_identical(tables, name):
    reference, rewrite = IDENTICAL[name]
    comparison = verdict(tables, reference + "\n" + rewrite)
    assert comparison.verdict == "identical", comparison


def test_overloads_are_paired_by_signature(tables):
    code = (
        f"void reference_g(int a) {{ {NOP} }}\nvoid g(int a) {{ {NOP} {NOP} }}\n"
        f"void reference_g(bool a) {{ {NOP} }}\nvoid g(bool a) {{ {NOP} }}"
    )
    found = verdicts(tables, code)
    assert {name: c.verdict for name, c in found.items()} == {
        "g(int)": "differs",
        "g(bool)": "identical",
    }


def test_namespaced_functions_are_paired(tables):
    code = f"namespace k {{ void reference_h() {{ {NOP} }} void h() {{ {NOP} }} }}"
    assert verdict(tables, code).verdict == "identical"


@pytest.mark.parametrize(
    "code",
    [
        f"void reference_pack_rows_init() {{ {NOP} }}\nvoid pack_rows_inti() {{ {NOP} }}",
        f"void reference_f(int a) {{ {NOP} }}\nvoid f(unsigned a) {{ {NOP} }}",
    ],
)
def test_unpaired_references_are_reported(tables, code):
    found = verdicts(tables, code)
    assert [c.verdict for c in found.values()] == ["unpaired"]


def test_existing_kernel_rewrite_reports_the_missing_nop(tables):
    with open(
        pipeline.build.LLK_ROOT
        / "tools/python/hal_explorer/examples/pack_reduce_mask_clear.cpp",
        encoding="utf-8",
    ) as f:
        comparison = verdict(tables, f.read())
    assert comparison.verdict == "differs"
    assert comparison.counterexample["diff"][-1] == "-NOP 0x02000000"
