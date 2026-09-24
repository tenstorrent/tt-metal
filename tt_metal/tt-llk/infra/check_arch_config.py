#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Keep ckernel_arch_config.h the only reader of the generated per-project macros.

Quasar parts differ by a regenerated ckernel_proj_params.h. That file is allowed exactly one reader,
the translation layer, which turns its macros into named constants. Everything downstream reads the
constants. Without that rule the macros spread, each new part adds conditionals at every site that
tested one, and the tree drifts back towards a copy of the kernels per part.

Two things are checked:

  1. Only ckernel_arch_config.h includes ckernel_proj_params.h.
  2. Only ckernel_arch_config.h names a macro defined by ckernel_proj_params.h.

Run with no arguments to check the whole Quasar tree, or pass paths (as pre-commit does).
"""

import argparse
import re
import sys
from pathlib import Path

LLK_ROOT = Path(__file__).resolve().parent.parent
QUASAR_ROOT = LLK_ROOT / "tt_llk_quasar"
PROJ_DIR = QUASAR_ROOT / "proj"
PROJ_PARAMS = sorted(PROJ_DIR.glob("*/ckernel_proj_params.h"))
ARCH_CONFIG = QUASAR_ROOT / "common" / "inc" / "ckernel_arch_config.h"

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]ckernel_proj_params\.h[">]', re.M)
DEFINE_RE = re.compile(r"^\s*#\s*define\s+(\w+)", re.M)
# Skip the macro name where it appears in prose rather than as code.
COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def generated_macros() -> set:
    """Macro names the configurations define, checked to be the same set in each.

    The configurations under proj/ are hand-maintained copies of one upstream-generated file, so
    they can drift. They are allowed to disagree on values -- that is what distinguishes them -- but
    a name present in one and missing from another means a re-sync landed in only some of them.
    """
    if not PROJ_PARAMS:
        sys.exit(
            f"no ckernel_proj_params.h under {PROJ_DIR}; the Quasar tree layout has moved"
        )

    per_file = {path: set(DEFINE_RE.findall(path.read_text())) for path in PROJ_PARAMS}
    reference, expected = next(iter(per_file.items()))
    for path, macros in per_file.items():
        if macros != expected:
            missing = sorted(expected - macros)
            extra = sorted(macros - expected)
            sys.exit(
                f"{path} has drifted from {reference}: "
                f"missing {missing or 'nothing'}, unexpected {extra or 'nothing'}. "
                f"Re-sync every configuration under proj/ together."
            )
    return expected


def sources(paths):
    if paths:
        return [
            Path(p) for p in paths if Path(p).suffix in (".h", ".hpp", ".cpp", ".cc")
        ]
    return sorted(QUASAR_ROOT.rglob("*.h")) + sorted(QUASAR_ROOT.rglob("*.cpp"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", help="files to check; defaults to the whole Quasar tree"
    )
    args = parser.parse_args()

    macros = generated_macros()
    macro_re = re.compile(
        r"\b(" + "|".join(sorted(macros, key=len, reverse=True)) + r")\b"
    )

    failures = []
    for path in sources(args.paths):
        resolved = path.resolve()
        if (
            resolved == ARCH_CONFIG
            or resolved.parent.parent == PROJ_DIR
            or not resolved.is_relative_to(QUASAR_ROOT)
        ):
            continue

        text = path.read_text()

        if INCLUDE_RE.search(text):
            failures.append(
                f"{path}: includes ckernel_proj_params.h directly. "
                f"Include ckernel_arch_config.h and read the named constant instead."
            )

        code = COMMENT_RE.sub("", text)
        for name in sorted(set(macro_re.findall(code))):
            failures.append(
                f"{path}: names the generated macro {name}. "
                f"Add a named constant for it to ckernel_arch_config.h and read that instead."
            )

    for failure in failures:
        print(failure, file=sys.stderr)

    if failures:
        print(
            f"\n{len(failures)} violation(s). ckernel_arch_config.h is the single translation layer "
            f"from generated macros to typed constants; see its header comment.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
