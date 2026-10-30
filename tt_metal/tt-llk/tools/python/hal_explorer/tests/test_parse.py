# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Parser, expression and walker tests on hand-written dumps; no compiler needed."""

import pytest
from hal_explorer.analysis.paths import Walker
from hal_explorer.gimple import loops, reader
from hal_explorer.gimple.expr import Scope, definitions_of, parse

NOP = 33554432


def dump(body, signature="void f (unsigned int x, bool a)", declarations=""):
    header = [
        ";; Function f (_Z1fjb, funcdef_no=1, decl_uid=1, cgraph_uid=1, symbol_order=1)",
        "",
        signature,
        "{",
    ]
    return "\n".join(header + declarations.splitlines()) + "\n" + body + "}\n"


def block(index, statements, successors):
    lines = [f";;   basic block {index}, loop depth 0", ";;    pred:       ENTRY"]
    lines += [f"  {s}" for s in statements]
    lines.append(f";;    succ:       {successors[0]}")
    lines += [f";;                {s}" for s in successors[1:]]
    return "\n".join(lines) + "\n"


def parse_one(text):
    (function,) = reader.parse_functions(text)
    return function


class Sources:
    def origin(self, location):
        return None


def walk(function):
    found = loops.find_loops(function)
    walker = Walker(
        function,
        found,
        tables=None,
        sources=Sources(),
        definitions=definitions_of(function),
    )
    return walker, walker.paths()


def counts(function):
    _, paths = walk(function)
    return sorted((" && ".join(p.conditions) or "always", len(p.items)) for p in paths)


def test_location_less_goto_ends_the_then_arm(monkeypatch):
    monkeypatch.setattr(
        "hal_explorer.isa.events.describe", lambda stmt, scope, tables, sources: stmt
    )
    text = dump(
        block(
            2,
            [
                "[in:1:1] if (a_2(D) != 0)",
                "  goto <bb 3>; [50.00%]",
                "else",
                "  goto <bb 4>; [50.00%]",
            ],
            [3, 4],
        )
        + block(
            3,
            [
                f'[in:2:1] __asm__ __volatile__(".ttinsn %0" :  : "n" {NOP});',
                "[0:0] goto <bb 5>; [100.00%]",
            ],
            [5],
        )
        + block(
            4, [f'[in:3:1] __asm__ __volatile__(".ttinsn %0" :  : "n" {NOP});'] * 2, [5]
        )
        + block(5, ["[in:4:1] return;"], ["EXIT"])
    )
    assert counts(parse_one(text)) == [("!a", 2), ("a", 1)]


def test_switch_cases_and_ranges_become_edges(monkeypatch):
    monkeypatch.setattr(
        "hal_explorer.isa.events.describe", lambda stmt, scope, tables, sources: stmt
    )
    insn = f'__asm__ __volatile__(".ttinsn %0" :  : "n" {NOP});'
    text = dump(
        block(
            2,
            [
                "[in:1:1] switch (x_1(D)) <default: <L9> [25.00%], case 0: <L0> [25.00%], case 3 ... 5: <L1> [50.00%]>"
            ],
            [3, 4, 5],
        )
        + block(3, ["<L0>:", insn], ["6"])
        + block(4, ["<L1>:", insn, insn], ["6"])
        + block(5, ["<L9>:"], ["6"])
        + block(6, ["return;"], ["EXIT"])
    )
    assert counts(parse_one(text)) == [
        ("3 <= x <= 5", 2),
        ("x == 0", 1),
        ("x not in {0, 3..5}", 0),
    ]


def test_block_without_successors_ends_the_path(monkeypatch):
    monkeypatch.setattr(
        "hal_explorer.isa.events.describe", lambda stmt, scope, tables, sources: stmt
    )
    insn = f'__asm__ __volatile__(".ttinsn %0" :  : "n" {NOP});'
    text = dump(
        block(
            2,
            [
                "if (a_2(D) != 0)",
                "  goto <bb 3>; [0.00%]",
                "else",
                "  goto <bb 4>; [100.00%]",
            ],
            [3, 4],
        )
        + ";;   basic block 3, loop depth 0\n;;    pred:       2\n  __builtin_trap ();\n;;    succ:      \n"
        + block(4, [insn, "return;"], ["EXIT"])
    )
    function = parse_one(text)
    assert function.blocks[3].terminator == ("noreturn",)
    assert counts(function) == [("!a", 1), ("a", 0)]


def test_annotations_and_volatile_forms_are_classified():
    text = dump(
        block(
            2,
            [
                "[in:1:1] __builtin_rvtt_ttreplay (&__instrn_buffer, 4, 0, 0, 16, 0, 0); [tail call]",
                "[in:2:1] *_3 ={v} 5;",
                "[in:3:1] MEM[(volatile uint32_t *)4290248704B] ={v} 4;",
                "[in:4:1] _4 ={v} MEM[(volatile uint16_t *)pc_buf_base.1_2 + 36B];",
                "[in:5:1] MEM <volatile uint32_t[<unknown>]> [(volatile uint32_t[<unknown>] &)&__instrn_buffer][0] ={v} _4;",
                "[in:6:1] opaque_helper ();",
                "[in:7:1] _5 = __builtin_rvtt_sfpload (0B, 0, 0, 0, 0, 7);",
                "[in:8:1] return;",
            ],
            ["EXIT"],
        ),
        declarations="  volatile uint8_t * _3;",
    )
    kinds = [
        (s.kind, s.data.get("address", {}).get("width"))
        for s in parse_one(text).blocks[2].statements
    ]
    assert kinds == [
        ("ttreplay", None),
        ("mmio_write", 8),
        ("mmio_write", 32),
        ("mmio_read", 16),
        ("store", None),
        ("call", None),
        ("unsupported", None),
    ]


def test_multi_line_asm_is_one_statement():
    text = dump(
        block(
            2,
            [
                '[in:1:1] __asm__ __volatile__("sw %0, (%1)',
                '\tlw %0, (%1); and x0, x0, %0" : "=r" raw_7 : "r" _6, "0" 0 : "memory");',
                "return;",
            ],
            ["EXIT"],
        )
    )
    statements = parse_one(text).blocks[2].statements
    assert [s.kind for s in statements] == ["def", "asm"]
    assert statements[1].data["template"] == "sw %0, (%1); lw %0, (%1); and x0, x0, %0"


def test_unrecognised_issuing_statement_warns():
    text = dump(
        block(2, ['[in:1:1] __asm__ goto ("x" :::: <L1>);', "return;"], ["EXIT"])
    )
    (statement,) = parse_one(text).blocks[2].statements
    assert statement.kind == "unsupported"


def test_parameters_and_declarations():
    text = dump(
        block(2, ["return;"], ["EXIT"]),
        signature="void f (const uint32_t n, volatile uint8_t * p)",
        declarations="  long unsigned int _4;",
    )
    function = parse_one(text)
    assert function.parameters == [("const uint32_t", "n"), ("volatile uint8_t *", "p")]
    assert function.declarations == {"_4": "long unsigned int"}


@pytest.mark.parametrize(
    "rhs, declared, values, expected",
    [
        ("_1 + 4294967295", "unsigned int", {"_1": 0}, 0xFFFFFFFF),
        ("(unsigned char) _1", "unsigned char", {"_1": 0x1FF}, 0xFF),
        ("(signed char) _1", "signed char", {"_1": 0xFF}, -1),
        ("_1 / -2", "int", {"_1": -7}, 3),
        ("_1 % -2", "int", {"_1": -7}, -1),
        ("_1 r>> 4", "unsigned int", {"_1": 0x12345678}, 0x81234567),
        ("MIN_EXPR <_1, _2>", "unsigned int", {"_1": 3, "_2": 9}, 3),
        ("_1 ? 7 : 9", "unsigned int", {"_1": 0}, 9),
        ("~_1", "unsigned int", {"_1": 0}, 0xFFFFFFFF),
    ],
)
def test_expression_values(rhs, declared, values, expected):
    declarations = f"  {declared} _9;\n  {declared} _1;\n  {declared} _2;"
    function = parse_one(
        dump(block(2, [f"_9 = {rhs};", "return;"], ["EXIT"]), declarations=declarations)
    )
    scope = Scope(function, values=values)
    assert scope.value("_9") == expected


def test_runtime_bits_and_constant_part():
    function = parse_one(
        dump(
            block(
                2,
                [
                    "_3 = x_1(D) << 8;",
                    "_4 = _3 & 16776960;",
                    "_5 = _4 + 1157627932;",
                    "return;",
                ],
                ["EXIT"],
            ),
            declarations="  unsigned int _3;\n  unsigned int _4;\n  unsigned int _5;",
        )
    )
    scope = Scope(function)
    assert scope.runtime_mask(parse("_5")) == 0x00FFFF00
    assert scope.constant_part(parse("_5")) == 1157627932
    assert scope.text(parse("_5")) == "(((x << 8) & 0xffff00) + 0x4500001c)"
