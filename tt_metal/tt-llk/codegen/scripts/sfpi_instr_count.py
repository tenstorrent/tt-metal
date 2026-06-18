#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Count generated instructions in a compiled LLK kernel ELF, and compare two builds.

This backs the optimizer's **SFPI-vs-TTI** decision (see
``codegen/agents/quasar/llk-optimizer.md``). When a kernel is requested
explicitly as an *SFPI version*, the optimizer reimplements the working raw
``TTI_`` kernel in the ``sfpi::`` C++ DSL and must prove the SFPI form is **no
worse** than the hand-written intrinsics before it is allowed to replace them.
"No worse" is measured here: the number of machine instructions the compiler
emits for the kernel.

Two metrics are available:

* whole-ELF (default, no ``--symbol``): total instruction count across every
  executable section. This is **inlining-immune** — Quasar SFPU kernels are
  ``always_inline`` and unrolled, so ``_calculate_{op}_`` often has no standalone
  symbol. When the TTI build and the SFPI build are the *same test variant*,
  everything except the kernel body is byte-identical, so the difference in the
  total count is exactly the difference in the kernel. This is the recommended
  comparison metric.
* ``--symbol RE``: count only the instructions inside the first function symbol
  whose demangled name matches ``RE`` (e.g. ``_calculate_zero_comp_``). Use this
  when the kernel function survives as its own symbol and you want the kernel's
  own instruction count rather than a whole-section delta. Errors if no symbol
  matches (so the caller knows to fall back to ``--text``).

``--sfp-only`` further restricts either metric to SFPU instructions (mnemonics
beginning ``sfp``).

ELFs land at
``/tmp/tt-llk-build/sources/<arch>/<test_cpp>/<variant_id>/elf/{unpack,math,pack,sfpu}.elf``
after a compile (see ``tests/python_tests/helpers/test_config.py``). For a Quasar
SFPU kernel the body is in ``math.elf`` — SFPU ops are dispatched from the MATH
thread (``_llk_math_eltwise_*_sfpu_*``), so ``sfpu.elf`` (the ``ISOLATE_SFPU``
build) is empty of the kernel. Pass ``math.elf``, or pass the ``elf/`` directory
and the tool picks ``math.elf`` then ``sfpu.elf``.

On Quasar the SFP ops disassemble with real mnemonics in math.elf
(``sfpload``/``sfpabs``/``sfpstore``/``sfpconfig``/…), so ``--sfp-only`` is a
meaningful kernel-focused metric there. Note that the *static* instruction count
depends on loop unrolling: keep the unroll pragma identical between the TTI and
SFPI versions so the counts are comparable (a rolled loop has far fewer static
instructions than an 8×-unrolled one).

Usage:
    # one ELF, total .text instructions
    python -m scripts.sfpi_instr_count count /tmp/tt-llk-build/.../elf/sfpu.elf

    # compare the TTI baseline against the SFPI reimplementation (same variant)
    python -m scripts.sfpi_instr_count compare tti/sfpu.elf sfpi/sfpu.elf
    #   exit 0  -> SFPI <= TTI  (keep SFPI)
    #   exit 1  -> SFPI  > TTI  (keep TTI)

    # kernel-symbol count only, SFP ops only
    python -m scripts.sfpi_instr_count count sfpu.elf --symbol _calculate_zero_comp_ --sfp-only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Default toolchain objdump, resolved relative to the tt-llk repo root (this
# file lives at <tt-llk>/codegen/scripts/sfpi_instr_count.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBJDUMP = (
    _REPO_ROOT / "tests" / "sfpi" / "compiler" / "bin" / "riscv-tt-elf-objdump"
)

# An objdump `-d` instruction line: leading ws, hex address, ':', then a tab.
# Matches real instructions and data-in-text (.word) lines alike — both are
# "generated" entries and both are byte-identical between two builds outside the
# kernel body, so counting them is correct for the delta metric.
_INSN_RE = re.compile(r"^\s+[0-9a-fA-F]+:\t")
# A symbol header line: `0001abcd <demangled name>:`
_SYM_RE = re.compile(r"^[0-9a-fA-F]+ <(.+)>:\s*$")


def _mnemonic(line: str) -> str:
    """Extract the mnemonic from an objdump instruction line, or '' if none."""
    # Format: "   1abce:\t<hex bytes>\t<mnemonic> <operands>"
    parts = line.split("\t")
    if len(parts) < 3:
        return ""
    return parts[2].split()[0] if parts[2].split() else ""


def _disassemble(elf: Path, objdump: Path) -> list[str]:
    # `-d` disassembles every executable section (.text, .text.*, …); counting
    # all instruction lines across them is the inlining-immune total. No `-j`
    # restriction — section names vary (e.g. .text.<fn>) and would be missed.
    cmd = [str(objdump), "-d", "--demangle", str(elf)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"objdump failed on {elf}:\n{proc.stderr.strip()}")
    return proc.stdout.splitlines()


def _resolve_elf(p: Path) -> Path:
    """Allow passing an elf/ directory: prefer math.elf, then sfpu.elf.

    Quasar SFPU ops (unary/binary/ternary) are dispatched from the MATH thread
    (`_llk_math_eltwise_*_sfpu_*`), so the kernel's SFP instructions land in
    math.elf — sfpu.elf (the ISOLATE_SFPU build) is empty of the kernel for that
    dispatch path. Validated on the abs kernel: math.elf had 8×sfpload/sfpabs/
    sfpstore, sfpu.elf had none.
    """
    if p.is_dir():
        for name in ("math.elf", "sfpu.elf"):
            cand = p / name
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No math.elf or sfpu.elf under {p}")
    if not p.exists():
        raise FileNotFoundError(f"ELF not found: {p}")
    return p


def count_instructions(
    elf: Path,
    objdump: Path,
    symbol: str | None = None,
    sfp_only: bool = False,
) -> int:
    """Count instructions in `elf`.

    With `symbol`, restrict to the first function whose demangled name contains
    the `symbol` regex. Without it, count the whole `.text` section.
    With `sfp_only`, count only mnemonics beginning with 'sfp'.
    """
    elf = _resolve_elf(elf)
    lines = _disassemble(elf, objdump)

    sym_re = re.compile(symbol) if symbol else None
    in_target = symbol is None  # whole-ELF mode counts every instruction
    found = False
    count = 0

    for line in lines:
        sym_match = _SYM_RE.match(line)
        if sym_match:
            if sym_re is not None:
                in_target = bool(sym_re.search(sym_match.group(1)))
                found = found or in_target
            continue
        if not in_target:
            continue
        if not _INSN_RE.match(line):
            continue
        if sfp_only and not _mnemonic(line).lower().startswith("sfp"):
            continue
        count += 1

    if sym_re is not None and not found:
        raise LookupError(
            f"No symbol matching /{symbol}/ in {elf.name} "
            f"(kernel may be fully inlined — fall back to --text)"
        )
    return count


def count_mnemonic(elf: Path, objdump: Path, mnemonic: str) -> int:
    """Count instructions whose mnemonic exactly equals `mnemonic`.

    Used as a loop-structure probe: an N×-unrolled SFPU face emits N `sfpstore`
    (one per row-store), a rolled loop emits 1. Comparing `sfpstore` counts
    between two builds detects unroll divergence — the failure mode where a
    rolled SFPI loop shows a misleadingly low *static* instruction count.
    """
    elf = _resolve_elf(elf)
    lines = _disassemble(elf, objdump)
    return sum(
        1 for line in lines if _INSN_RE.match(line) and _mnemonic(line) == mnemonic
    )


def _is_coproc(mn: str) -> bool:
    """SFPU / Tensix coprocessor op (the instructions the kernel actually emits)."""
    return mn.lower().startswith(("sfp", "tt"))


def dump_ops(
    elf: Path,
    objdump: Path,
    symbol: str | None = None,
    include_all: bool = False,
) -> list[str]:
    """Return the generated instruction sequence, address-stripped for diffing.

    Default: only SFPU/Tensix coprocessor ops (`sfp*`/`tt*`) — the program the
    kernel actually pushes to the vector unit, which is what determines whether
    one lowering is better than another. `include_all=True` keeps the RISC-V
    scaffolding too. With `symbol`, restrict to that function's range.

    Addresses and raw encodings are dropped so a plain `diff` of two dumps
    (TTI vs SFPI) surfaces exactly the instruction-level differences.
    """
    elf = _resolve_elf(elf)
    lines = _disassemble(elf, objdump)
    sym_re = re.compile(symbol) if symbol else None
    in_target = symbol is None
    out: list[str] = []
    for line in lines:
        sym_match = _SYM_RE.match(line)
        if sym_match:
            if sym_re is not None:
                in_target = bool(sym_re.search(sym_match.group(1)))
            continue
        if not in_target or not _INSN_RE.match(line):
            continue
        mn = _mnemonic(line)
        if not include_all and not _is_coproc(mn):
            continue
        # parts[2] is "<mnemonic> <operands>"; strip the trailing "# <addr> <sym>" note.
        body = line.split("\t", 2)[2].split("#", 1)[0].strip()
        out.append(body)
    return out


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--symbol",
        metavar="RE",
        help="Count only instructions inside the first symbol matching this "
        "regex (e.g. _calculate_zero_comp_). Default: whole .text section.",
    )
    p.add_argument(
        "--sfp-only",
        action="store_true",
        help="Count only SFPU instructions (mnemonics beginning 'sfp').",
    )
    p.add_argument(
        "--objdump",
        type=Path,
        default=DEFAULT_OBJDUMP,
        help=f"Path to riscv-tt-elf-objdump (default: {DEFAULT_OBJDUMP}).",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("count", help="Count instructions in one ELF.")
    pc.add_argument("elf", type=Path, help="ELF file or elf/ directory.")
    _add_common_flags(pc)

    pd = sub.add_parser(
        "dump",
        help="Print the generated coprocessor op sequence (address-stripped, "
        "diff-friendly) so you can read what a build actually emits.",
    )
    pd.add_argument("elf", type=Path, help="ELF file or elf/ directory.")
    pd.add_argument(
        "--symbol",
        metavar="RE",
        help="Restrict to the first symbol matching this regex (e.g. run_kernel).",
    )
    pd.add_argument(
        "--all",
        action="store_true",
        dest="include_all",
        help="Include RISC-V scaffolding too (default: only sfp*/tt* ops).",
    )
    pd.add_argument(
        "--objdump",
        type=Path,
        default=DEFAULT_OBJDUMP,
        help=f"Path to riscv-tt-elf-objdump (default: {DEFAULT_OBJDUMP}).",
    )

    pk = sub.add_parser("compare", help="Compare TTI vs SFPI builds.")
    pk.add_argument("tti_elf", type=Path, help="TTI baseline ELF or elf/ dir.")
    pk.add_argument(
        "sfpi_elf", type=Path, help="SFPI reimplementation ELF or elf/ dir."
    )
    pk.add_argument(
        "--store-mnemonic",
        default="sfpstore",
        help="Mnemonic used as the unroll-structure probe (default: sfpstore — "
        "one per emitted row-store). Equal counts => same unroll.",
    )
    pk.add_argument(
        "--no-structure-check",
        action="store_true",
        help="Disable the unroll-structure gate (compare raw counts only).",
    )
    _add_common_flags(pk)

    args = parser.parse_args(argv)
    objdump = args.objdump
    if not Path(objdump).exists():
        print(f"ERROR: objdump not found at {objdump}", file=sys.stderr)
        return 3

    try:
        if args.cmd == "count":
            n = count_instructions(args.elf, objdump, args.symbol, args.sfp_only)
            if args.json:
                print(json.dumps({"elf": str(args.elf), "instructions": n}))
            else:
                print(n)
            return 0

        if args.cmd == "dump":
            ops = dump_ops(args.elf, objdump, args.symbol, args.include_all)
            print("\n".join(ops))
            return 0

        # compare
        tti = count_instructions(args.tti_elf, objdump, args.symbol, args.sfp_only)
        sfpi = count_instructions(args.sfpi_elf, objdump, args.symbol, args.sfp_only)
        keep_sfpi = sfpi <= tti
        delta = sfpi - tti

        # Loop-structure gate: a lower static count is only a real win if both
        # builds emit the SAME number of row-stores. If the TTI face was
        # 8×-unrolled (8 sfpstore) but the SFPI loop rolled (1 sfpstore), SFPI's
        # static count collapses by ~the unroll factor — misleadingly "fewer"
        # instructions that actually execute MORE times. Detect that here so a
        # lower count can't be trusted blindly.
        structure_ok = True
        store_mn = args.store_mnemonic
        tti_stores = sfpi_stores = None
        if not args.no_structure_check:
            tti_stores = count_mnemonic(args.tti_elf, objdump, store_mn)
            sfpi_stores = count_mnemonic(args.sfpi_elf, objdump, store_mn)
            # Only meaningful if the baseline actually has stores to compare.
            if tti_stores > 0 and tti_stores != sfpi_stores:
                structure_ok = False

        # A lower count is only a trustworthy win when structure matches.
        inconclusive = (not structure_ok) and keep_sfpi

        if args.json:
            print(
                json.dumps(
                    {
                        "tti": tti,
                        "sfpi": sfpi,
                        "delta": delta,
                        "keep_sfpi": keep_sfpi,
                        "tti_stores": tti_stores,
                        "sfpi_stores": sfpi_stores,
                        "structure_ok": structure_ok,
                        "inconclusive": inconclusive,
                    }
                )
            )
        else:
            verdict = "KEEP SFPI" if keep_sfpi else "KEEP TTI"
            sign = f"+{delta}" if delta > 0 else str(delta)
            print(f"TTI : {tti}")
            print(f"SFPI: {sfpi}  ({sign} vs TTI)")
            if not structure_ok:
                print(
                    f"STRUCTURE MISMATCH: TTI emits {tti_stores} {store_mn}, "
                    f"SFPI emits {sfpi_stores} — the loops unroll differently, so "
                    f"the static counts are not comparable."
                )
            if inconclusive:
                print(
                    "INCONCLUSIVE — a lower SFPI count here is an unroll artifact, "
                    "not a real win. Match the #pragma GCC unroll (and ensure the "
                    "SFPI body is unroll-eligible), recompile, and re-compare."
                )
            else:
                print(
                    f"{verdict}  (rule: keep SFPI iff SFPI <= TTI and structure matches)"
                )

        if inconclusive:
            return 2
        return 0 if keep_sfpi else 1
    except (RuntimeError, FileNotFoundError, LookupError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
