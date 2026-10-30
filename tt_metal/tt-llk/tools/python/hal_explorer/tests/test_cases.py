# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hand-checked expectations for control flow, loops, MMIO and warnings."""

import pytest
from conftest import needs_toolchain
from hal_explorer import pipeline
from hal_explorer.model import Access, Instruction

pytestmark = needs_toolchain

NOP = "TTI_NOP;"


def set_reg(value):
    return f"TTI_SETDMAREG(0, {value}, 0, 0);"


def analyse(tables, code, mode="code", variables=()):
    result = pipeline.run(mode, code, tables, list(variables))
    assert result.ok, result.diagnostics
    return result


def path_counts(function):
    return sorted(
        (" && ".join(c) or "always", g.count) for g in function.groups for c in g.when
    )


def test_if_else_both_arms_fall_through(tables):
    code = f"void f(bool a) {{ if (a) {{ {set_reg(1)} }} else {{ {set_reg(2)} {NOP} }} {NOP} }}"
    (function,) = analyse(tables, code).functions
    assert path_counts(function) == [("!a", 3), ("a", 2)]


def test_dense_switch_lists_every_case(tables):
    cases = " ".join(f"case {k}: {' '.join([NOP] * (k + 1))} break;" for k in range(8))
    code = f"void f(unsigned x) {{ switch (x) {{ {cases} default: break; }} }}"
    (function,) = analyse(tables, code).functions
    counts = dict(path_counts(function))
    assert {counts[f"x == {k}"] for k in range(8)} == set(range(1, 9))
    assert counts["x not in {0, 1, 2, 3, 4, 5, 6, 7}"] == 0


def test_trap_ends_the_path(tables):
    code = f"void f(bool c) {{ {NOP} if (c) __builtin_trap(); {NOP} {NOP} }}"
    (function,) = analyse(tables, code).functions
    assert path_counts(function) == [("!c", 3), ("c", 1)]


def test_runtime_loop_count_is_fitted(tables):
    code = f"void f(unsigned n) {{ for (unsigned i = 0; i < n; i++) {{ {NOP} }} {set_reg(3)} }}"
    (function,) = analyse(tables, code).functions
    looping = next(g for g in function.groups if g.count is None)
    assert looping.count_text.startswith("1 + n  (fitted")


def test_loop_carried_word_changes_every_iteration(tables):
    code = (
        "void f() { for (unsigned i = 0; i < 40; i++) { TT_SETDMAREG(0, i, 0, 5); } }"
    )
    (function,) = analyse(tables, code).functions
    (group,) = function.groups
    payloads = [
        i.fields[1]["value"] for i in group.sequence if isinstance(i, Instruction)
    ]
    assert payloads == list(range(40))


def test_down_counting_and_narrow_loops(tables):
    code = (
        f"void down(unsigned n) {{ for (unsigned i = n; i > 0; i--) {{ {NOP} }} }}\n"
        f"void narrow(unsigned char n) {{ for (unsigned char i = 0; i < n; i++) {{ {NOP} }} }}"
    )
    result = analyse(tables, code)
    for function in result.functions:
        complete = [s for s in function.samples if s.complete]
        assert len(complete) > 5
        for sample in complete:
            assert sample.count == sample.inputs["n"], (function.name, sample)


def test_nested_loops_keep_every_path(tables):
    code = f"void f(unsigned n) {{ for (unsigned i = 0; i < n; i++) {{ for (unsigned j = 0; j < i; j++) {{ {NOP} }} {set_reg(1)} }} }}"
    (function,) = analyse(tables, code).functions
    assert not function.warnings
    for sample in function.samples:
        n = sample.inputs["n"]
        if sample.complete:
            assert sample.count == n * (n - 1) // 2 + n


def test_runtime_word_keeps_its_constant_part(tables):
    code = "void f(int d, unsigned a) { TT_SETDMAREG(0, d, 0, 0); TT_SETDMAREG(1, a ^ 3, 0, a & 1); }"
    (function,) = analyse(tables, code).functions
    mnemonics = [
        i.mnemonic for i in function.groups[0].sequence if isinstance(i, Instruction)
    ]
    assert mnemonics == ["SETDMAREG", "SETDMAREG"]


def test_mmio_through_a_pointer_is_listed(tables):
    code = (
        "void f(unsigned v) { ckernel::regfile[0] = v; ckernel::pc_buf_base[2] = v; }"
    )
    (function,) = analyse(tables, code).functions
    accesses = [i for i in function.groups[0].sequence if isinstance(i, Access)]
    assert [a.target for a in accesses] == ["regfile[0]", "pc_buf_base[2]"]


def test_config_register_names(tables):
    code = (
        '#include "hal/cfg.h"\n'
        "void f(unsigned v) { hal::cfg::write<hal::cfg::Access::MMIO, hal::cfg::Thcon[hal::cfg::Reg3].Base_address, hal::cfg::Sec::S0>(v); }"
    )
    (function,) = analyse(tables, code).functions
    targets = {
        i.target for g in function.groups for i in g.sequence if isinstance(i, Access)
    }
    assert targets == {
        "cfg state 0: THCON_SEC0_REG3_Base_address",
        "cfg state 1: THCON_SEC0_REG3_Base_address",
    }


@pytest.mark.parametrize(
    "code, message",
    [
        ("void helper(); void f() { TTI_NOP; helper(); TTI_NOP; }", "call to helper"),
        (
            '#include "sfpi.h"\nvoid f() { sfpi::vFloat x = sfpi::dst_reg[0]; sfpi::dst_reg[0] = x * x; }',
            "SFPI",
        ),
    ],
)
def test_hidden_instructions_are_reported(tables, code, message):
    result = analyse(tables, code)
    assert any(message in w["message"] for w in result.functions[0].warnings)
    assert result.incomplete


def test_replay_instructions_are_counted(tables):
    code = (
        "void f() { hal::replay::record<hal::replay::BufferRange {4, 3}>([] { TTI_NOP; TTI_NOP; TTI_NOP; });"
        " hal::replay::run<hal::replay::BufferRange {4, 3}>(); }"
    )
    (function,) = analyse(tables, code).functions
    sequence = function.groups[0].sequence
    assert [i.mnemonic for i in sequence] == ["REPLAY", "NOP", "NOP", "NOP", "REPLAY"]
    assert [i.word for i in sequence][0::4] == [0x04010031, 0x04010030]


def test_body_mode_with_variables(tables):
    body = f"if (a) {{ {NOP} }}\nscalar::set(hal::gpr<14>(), v);"
    result = analyse(
        tables,
        body,
        "body",
        [{"type": "bool", "name": "a"}, {"type": "std::uint32_t", "name": "v"}],
    )
    (function,) = result.functions
    assert path_counts(function) == [("!a", 2), ("a", 3)]
    runtime = [
        i
        for i in function.groups[0].sequence
        if isinstance(i, Instruction) and i.runtime_mask
    ]
    assert [i.issue for i in runtime] == ["store", "store"]
    assert runtime[0].runtime_expr == "(((v << 8) & 0xffff00) + 0x4500001c)"


def test_body_mode_rejects_unbalanced_braces(tables):
    result = pipeline.run("body", "}(); } void g() { TTI_NOP;", tables)
    assert not result.ok and "unbalanced" in result.diagnostics


def test_no_function_emitted(tables):
    result = pipeline.run("code", "inline void f() { TTI_NOP; }", tables)
    assert not result.ok and "No function was emitted" in result.diagnostics


def test_compile_errors_point_at_the_input(tables):
    result = pipeline.run("code", "void f() {\n  nope();\n}", tables)
    assert not result.ok
    assert result.errors[0]["file"] == "input" and result.errors[0]["line"] == 2
