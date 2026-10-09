# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Which SFPU ops did the PR change? Compile both sides and compare machine code.

A header can be included by many kernels (``ckernel_sfpu_exp.h`` reaches sigmoid,
gelu, softplus, ...), so the changed file list does not say which ops changed.
The compiled code does: every perf variant is compiled on both sides, and an op is
affected when the ``.text`` of any of its math-thread ELFs differs.

Variants are matched across the sides by their generated ``build.h``, which holds
every compile-time parameter of the variant and no paths, so it is identical on
both sides for the same variant. Compilation needs no device.
"""

import hashlib
import re
import subprocess

import runner

#: The perf modules whose variants the detector compiles. Each covers one family.
MODULES = {
    "unary": "perf_eltwise_unary_sfpu.py",
    "typecast": "perf_eltwise_unary_typecast.py",
    "binary": "perf_eltwise_binary_sfpu.py",
}
#: The kernel source each module compiles; its artefacts live under sources/<name>/.
SOURCES = {
    "unary": "eltwise_unary_sfpu_perf.cpp",
    "typecast": "eltwise_unary_typecast_perf.cpp",
    # Shared with the functional tests; only the perf module is compiled here.
    "binary": "sfpu_binary_test.cpp",
}

_BINARY_OP = re.compile(r"SFPU_BINARY_OPERATION\s*=\s*ckernel::BinaryOp::(\w+)")
_SFPU_OP = re.compile(r"SFPU_UNARY_OPERATION\s*=\s*SfpuType::(\w+)")
_TYPECAST = re.compile(r"TYPECAST_(?:IN|OUT)\w*\s*=\s*[\w:]*?(\w+)\s*;")


def _objcopy(side):
    return side.llk / "tests" / "sfpi" / "compiler" / "bin" / "riscv-tt-elf-objcopy"


def text_hash(elf, objcopy):
    """sha256 of the ELF's ``.text``: the instructions, without debug info or paths."""
    blob = subprocess.run(
        [str(objcopy), "-O", "binary", "--only-section=.text", str(elf), "/dev/stdout"],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def op_of(build_h_text, family):
    if family == "typecast":
        found = _TYPECAST.findall(build_h_text)
        return "typecast" + ("(" + "->".join(found) + ")" if found else "")
    m = (_BINARY_OP if family == "binary" else _SFPU_OP).search(build_h_text)
    return m.group(1) if m else None


def compiled_variants(side, family):
    """{build.h sha: (op, math .text sha)} for every variant the side compiled."""
    out = {}
    objcopy = _objcopy(side)
    for build_h in side.artefacts.glob(f"sources/{SOURCES[family]}/*/build.h"):
        math = build_h.parent / "elf" / "math.elf"
        if not math.exists():
            continue
        text = build_h.read_text()
        op = op_of(text, family)
        if op is None:
            continue
        key = hashlib.sha256(text.encode()).hexdigest()
        out[key] = (op, text_hash(math, objcopy))
    return out


def compile_all(side, arch, family, log, jobs=8, only_ops=()):
    """Compile (never run) every variant of the family's perf module, math only."""
    op_args = [a for op in only_ops for a in ("--op", op)]
    runner.pytest(
        side,
        arch,
        [
            "--compile-producer",
            "-n",
            str(jobs),
            "-m",
            "perf",
            "--speed-of-light",
            *op_args,
            MODULES[family],
        ],
        env={"LLK_PERF_RUN_TYPES": "MATH_ISOLATE"},
        log=log,
        # A variant that does not build on one side is left out of the comparison
        # (and reported as unmatched), rather than failing the whole detection.
        check=False,
    )


def changed_ops(base, head, arch, log, families=("unary", "typecast", "binary"), jobs=8, only_ops=()):
    """Returns ``{family: {"changed": [...], "unchanged": [...], "unmatched": [...]}}``.

    ``unmatched`` are ops whose variants exist on one side only (the PR added or
    removed a variant); they are reported, and measured when they exist on both.
    """
    result = {}
    slot = base.build.parent / "slot"
    for family in families:
        found = {}
        for side in (base, head):
            at = runner.at_slot(side, slot)
            compile_all(at, arch, family, log, jobs, only_ops)
            found[side.name] = compiled_variants(at, family)
        b, h = found["base"], found["head"]
        changed, seen, unmatched = set(), set(), set()
        for key, (op, text) in h.items():
            seen.add(op)
            if key not in b:
                unmatched.add(op)
            elif b[key][1] != text:
                changed.add(op)
        for key, (op, _) in b.items():
            seen.add(op)
            if key not in h:
                unmatched.add(op)
        result[family] = {
            "changed": sorted(changed),
            "unchanged": sorted(seen - changed - unmatched),
            "unmatched": sorted(unmatched - changed),
        }
    return result
