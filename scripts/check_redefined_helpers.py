#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit guard against local copies of helpers that already exist.

Host and device are different compilers and do not share headers. A helper that
exists on one side is not a reason to paste it onto the other, and a helper
that exists on the same side should be called, not redefined.

The check looks only at lines added in the commit (or the whole file with
``--all-lines``). Existing copies stay until someone edits them. A deliberate
exception needs a same-line comment: ``// allow-redefined-helper: <why>``.

Run ``python3 scripts/check_redefined_helpers.py --list`` to print the catalog.
"""

import argparse
import os
import re
import subprocess
import sys

from precommit_common import REPO_ROOT, blank_noncode, safe_git_rev, safe_repo_path

# Files that are the source of truth. Definitions inside them are the API.
ALLOWLIST = {
    "tt_metal/api/tt-metalium/math.hpp",
    "tt_metal/api/tt-metalium/constants.hpp",
    "tt_metal/api/tt-metalium/tt_align.hpp",
    "tt_metal/api/tt-metalium/tilize_utils.hpp",
    "tt_metal/impl/data_format/tilize_utils.cpp",
    "tt_metal/hw/inc/api/tile_constants.h",
    "tt_metal/hw/inc/api/alignment.h",
    "ttnn/cpp/ttnn/operations/math.hpp",
    "ttnn/cpp/ttnn/operations/core/program_cache_l1.hpp",
    "ttnn/cpp/ttnn/operations/data_movement/common/common.hpp",
    "ttnn/cpp/ttnn/operations/data_movement/common/common.cpp",
}

# Low-level or vendored trees with their own copies. Not this check's job.
EXCLUDED_PREFIXES = (
    "tt_metal/third_party/",
    "tt_metal/tt-llk/",
    "tt_metal/hw/ckernels/",
)

# Device L1/DRAM/PCIE alignment macros live in the arch noc headers.
EXCLUDED_ALIGNMENT_HEADERS = re.compile(r"^tt_metal/hw/inc/internal/.*/noc/noc_parameters\.h$")

CATALOG = """\
Host (g++ / the ttnn and tt_metal host libraries)
  Ceiling division, round to a multiple, round down, power-of-two align
    tt::div_up, tt::round_up, tt::round_down, tt::align
    #include <tt-metalium/math.hpp>
    tt::align requires a positive power-of-two alignment. tt::round_up does not.
  Default 32x32 tile and 16x16 face
    tt::constants::TILE_HEIGHT, TILE_WIDTH, TILE_HW, FACE_HEIGHT, FACE_WIDTH, FACE_HW
    #include <tt-metalium/constants.hpp>
    Host only. Device code must not include this header.
  Tile-multiple wrappers around the same math
    round_up_to_mul16, round_up_to_mul32, round_up_to_tile
    #include <tt-metalium/tilize_utils.hpp>
    Prefer tt::round_up(value, tt::constants::TILE_HEIGHT) in new code.
  Power-of-two test, ttnn host only
    tt::tt_metal::is_power_of_two
    #include "ttnn/operations/math.hpp"
    There is no metalium equivalent. tt_metal host code should not grow a third copy.
  L1 budget for a program
    ttnn::operations::core::available_program_l1_capacity(device)
    ttnn::operations::core::usable_program_l1_capacity(device)   // 95% margin
    #include "ttnn/operations/core/program_cache_l1.hpp"
    Live free L1 (moves as tensors allocate):
      ttnn::operations::data_movement::get_max_l1_space(tensor)
    Static per-core budget, independent of current occupancy:
      ttnn::operations::data_movement::get_static_l1_space(tensor)
    Declared in ttnn/operations/data_movement/common/common.hpp.
  Alignment in bytes, from the architecture rather than a literal 16 or 32
    tt::tt_metal::hal::get_l1_alignment, get_dram_alignment, get_pcie_alignment
    #include <tt-metalium/hal.hpp>

Device (RISC-V kernels, firmware, tt_metal/hw)
  Default 32x32 tile and 16x16 face
    tt::constants::TILE_HEIGHT, TILE_WIDTH, TILE_HW, FACE_HEIGHT, FACE_WIDTH, FACE_HW
    #include "api/tile_constants.h"
    Kernels with a non-default tile take the dimensions as compile-time or runtime
    arguments. A local TILE_HEIGHT = get_arg(...) is fine. TILE_HEIGHT = 32 is not.
  Round up to a multiple, power-of-two align, power-of-two test
    align(addr, multiple), align_power_of_2(addr, alignment), is_power_of_2(n)
    #include "api/alignment.h"
    align() is the device round_up. It accepts any positive multiple.
  L1 alignment
    L1_ALIGNMENT from the architecture noc parameters header already included by
    kernel builds. Do not #define a second one.
  No shared device div_up or round_down
    The data-movement kernel header defines div_up and round_up and tells callers
    not to include it (code size). Do not add another shared device header by
    pasting that one. A kernel-local div_up is still a copy, but this check does
    not fail it, because there is no header that every kernel can include.
    Host .cpp files must call tt::div_up / tt::round_up instead of growing another copy.
"""

FUNC_NAME = re.compile(
    r"\b(?P<name>(?:div_up|round_up|round_down|align_up|align_down|align|is_power_of_two|is_power_of_2)"
    r"(?:_[A-Za-z0-9]+)*)\s*(?:<[^;>]{0,160}>)?\s*\("
)
MACRO_NAME = re.compile(
    r"^\s*#\s*define\s+(?P<name>(?:div_up|round_up|round_down|align_up|align_down|"
    r"TILE_HEIGHT|TILE_WIDTH|TILE_HW|FACE_HEIGHT|FACE_WIDTH|FACE_HW|L1_ALIGNMENT))\b"
)
CONST_NAME = re.compile(
    r"\b(?P<name>TILE_HEIGHT|TILE_WIDTH|TILE_HW|FACE_HEIGHT|FACE_WIDTH|FACE_HW|L1_ALIGNMENT)\s*"
    r"(?:=\s*(?P<eq>[^;]*)|\{\s*(?P<br>[^}]*)\})"
)
L1_FUNC = re.compile(
    r"\b(?P<name>get_max_l1_space|get_static_l1_space|available_program_l1_capacity|"
    r"usable_program_l1_capacity)\s*\("
)
L1_VAR = re.compile(
    r"\b(?:max_l1(?:_[A-Za-z0-9]+)?|usable_l1(?:_[A-Za-z0-9]+)?|available_l1(?:_[A-Za-z0-9]+)?|"
    r"free_l1(?:_[A-Za-z0-9]+)?|static_l1(?:_[A-Za-z0-9]+)?|l1_budget|l1_space)\b"
)
L1_RAW = re.compile(r"\b(?:l1_size_per_core|lowest_occupied_compute_l1_address|get_base_allocator_addr)\s*\(")
L1_CANON_CALL = re.compile(
    r"\b(?:get_max_l1_space|get_static_l1_space|available_program_l1_capacity|usable_program_l1_capacity)\s*\("
)
INT_LITERAL = re.compile(r"^\(?\s*[+-]?(?:0[xX][0-9A-Fa-f]+|\d+)(?:[uUlL])*\s*\)?$")
STATEMENT_KEYWORDS = {
    "return",
    "co_return",
    "if",
    "for",
    "while",
    "switch",
    "case",
    "throw",
    "new",
    "delete",
    "sizeof",
    "alignof",
    "else",
    "do",
    "catch",
    "goto",
    "break",
    "continue",
    "and",
    "or",
    "not",
}
DECL_MARKERS = (
    "constexpr",
    "consteval",
    "static",
    "inline",
    "template",
    "FORCE_INLINE",
    "virtual",
    "explicit",
    "friend",
)
PLAIN_TYPES = {
    "auto",
    "void",
    "bool",
    "int",
    "char",
    "unsigned",
    "signed",
    "long",
    "short",
    "size_t",
    "T",
    "U",
    "Integer",
}


def relpath(path):
    """Slash-normalized path. Absolute paths are made repo-relative when they stay inside it."""
    path = path.replace("\\", "/")
    if os.path.isabs(path):
        confined = safe_repo_path(path)
        if confined is None:
            return path
        return os.path.relpath(confined, REPO_ROOT).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def side_of(path):
    """Which compiler this file is built with: host, device, or both."""
    if "hostdev" in path:
        return "both"
    if path.startswith("tt_metal/hw/") or "/hw/inc/" in path or "/firmware/" in path:
        return "device"
    if re.search(r"(^|/)(test_)?kernels?(/|$)", path):
        return "device"
    return "host"


def excluded(path):
    if path in ALLOWLIST or EXCLUDED_ALIGNMENT_HEADERS.match(path):
        return True
    return any(path.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def comment_opts_out(original, masked):
    """True when this line's opt-out marker sits in a comment, not in a literal.

    ``masked`` is ``original`` after blank_noncode, so a comment or literal is spaces
    while the surrounding code is unchanged. The marker counts only when the blanked
    run that holds it opens with ``//`` or ``/*``.
    """
    start = 0
    while True:
        idx = original.find("allow-redefined-helper", start)
        if idx < 0:
            return False
        start = idx + 1
        if idx >= len(masked) or masked[idx] != " ":
            continue
        k = idx
        while k > 0 and masked[k - 1] == " ":
            k -= 1
        while k < idx and original[k] == " ":
            k += 1
        if k < len(masked) and masked[k] == " " and original.startswith(("//", "/*"), k):
            return True


def base_of(name):
    for base in (
        "is_power_of_two",
        "is_power_of_2",
        "div_up",
        "round_up",
        "round_down",
        "align_up",
        "align_down",
        "align",
    ):
        if name == base or name.startswith(base + "_"):
            return base
    return name


def _type_token(prefix):
    """Return the type immediately before a declarator, if the prefix looks like one."""
    prefix = re.sub(r"(?:[\w]+::)+\s*$", "", prefix)
    match = re.search(
        r"(?:^|[\s;{}])(?:[\w:]+)\s+(?:(?:const|volatile)\s+)*(?:[*&]\s*)*$",
        prefix,
    )
    if not match:
        return None
    token = re.findall(r"[\w:]+", match.group(0))
    if not token:
        return None
    return token[-1]


def _type_is_declarator(prefix, type_token):
    if type_token in STATEMENT_KEYWORDS:
        return False
    if any(re.search(rf"\b{marker}\b", prefix) for marker in DECL_MARKERS):
        return True
    if type_token in PLAIN_TYPES or "::" in type_token:
        return True
    return type_token.endswith("_t") or type_token.endswith("_type")


def is_function_declarator(line, match):
    prefix = line[: match.start()]
    if re.search(r"(?:\.|->)\s*$", prefix):
        return False
    type_token = _type_token(prefix)
    if type_token is None:
        return False
    return _type_is_declarator(prefix, type_token)


def advice_for(kind, name, side):
    host_math = "include <tt-metalium/math.hpp> and call tt::div_up, tt::round_up, tt::round_down, or tt::align"
    device_align = 'include "api/alignment.h" and call align (any multiple) or align_power_of_2'
    if kind == "const":
        if name == "L1_ALIGNMENT":
            if side == "host":
                return "include <tt-metalium/hal.hpp> and call tt::tt_metal::hal::get_l1_alignment()"
            return "use the L1_ALIGNMENT macro from the architecture noc parameters header"
        if side == "device":
            return 'include "api/tile_constants.h" and use tt::constants::' + name
        if side == "both":
            return (
                "host code uses tt::constants::"
                + name
                + " from <tt-metalium/constants.hpp>; device code uses the same name from "
                + '"api/tile_constants.h". A shared header should take the value as an argument'
            )
        return "include <tt-metalium/constants.hpp> and use tt::constants::" + name
    if kind == "l1":
        return (
            "call ttnn::operations::data_movement::get_max_l1_space (live free L1) or "
            "get_static_l1_space (occupancy-independent), or "
            "ttnn::operations::core::available_program_l1_capacity / usable_program_l1_capacity"
        )
    if kind == "power":
        if side == "device":
            return 'include "api/alignment.h" and call is_power_of_2'
        return 'include "ttnn/operations/math.hpp" and call tt::tt_metal::is_power_of_two'
    # Arithmetic helpers.
    if side == "device":
        return device_align + ". There is no shared device div_up; this check only flags round/align copies"
    if side == "both":
        return (
            f"from host code, {host_math}. From device code, {device_align}. "
            "A third implementation in a hostdev header is the copy this check blocks"
        )
    if name.startswith("round_up_to_tile") or name in {"round_up_to_mul16", "round_up_to_mul32"}:
        return (
            "call tt::round_up(value, tt::constants::TILE_HEIGHT) from <tt-metalium/math.hpp> "
            "and <tt-metalium/constants.hpp>, or the existing wrappers in <tt-metalium/tilize_utils.hpp>"
        )
    return host_math


def applies(kind, name, side, path):
    """False when this side has no canonical function to point at."""
    base = base_of(name)
    if kind == "power":
        if side == "device":
            return True
        return side == "host" and (path.startswith("ttnn/") or path.startswith("tt-train/"))
    if kind == "l1":
        return side != "device"
    if kind == "const":
        return True
    if side == "device" and base in {"div_up", "round_down"}:
        return False
    if side == "both" and base in {"div_up", "round_down"}:
        return False
    return True


def _emit(found, path, line_no, kind, name, side):
    if not applies(kind, name, side, path):
        return
    found.append((line_no, kind, name, advice_for(kind, name, side)))


def find_violations(path, text, only_lines=None):
    """Return (line, kind, name, advice) for redefinitions in ``text``.

    ``only_lines`` limits the report to those 1-based line numbers. The whole
    file is still parsed so a definition split across lines is visible.
    """
    path = relpath(path)
    if excluded(path):
        return []
    side = side_of(path)
    raw_lines = text.splitlines()
    # Keep #define visible. Comments, strings and character literals are blanked, newlines stay.
    masked_lines = blank_noncode(text, blank_directives=False).splitlines()
    if len(masked_lines) < len(raw_lines):
        masked_lines.extend([""] * (len(raw_lines) - len(masked_lines)))
    code_lines = ["" if comment_opts_out(raw, masked) else masked for raw, masked in zip(raw_lines, masked_lines)]
    found = []

    def selected(line_no):
        return only_lines is None or line_no in only_lines

    for line_no, code in enumerate(code_lines, 1):
        if not code or not selected(line_no):
            continue
        macro = MACRO_NAME.match(code)
        if macro:
            name = macro.group("name")
            kind = "const" if name.isupper() else "func"
            _emit(found, path, line_no, kind, name, side)
            continue
        const = CONST_NAME.search(code)
        if const and re.search(r"\b(?:constexpr|const)\b", code[: const.start()]):
            rhs = (const.group("eq") if const.group("eq") is not None else const.group("br")).strip()
            if INT_LITERAL.match(rhs):
                _emit(found, path, line_no, "const", const.group("name"), side)
        for match in FUNC_NAME.finditer(code):
            if not is_function_declarator(code, match):
                continue
            name = match.group("name")
            kind = "power" if base_of(name) in {"is_power_of_two", "is_power_of_2"} else "func"
            _emit(found, path, line_no, kind, name, side)
        for match in L1_FUNC.finditer(code):
            if is_function_declarator(code, match):
                _emit(found, path, line_no, "l1", match.group("name"), side)

    # A reinvented L1 budget is often split over the initializer's next line.
    statement = []
    start = None
    for line_no, code in enumerate(code_lines, 1):
        if code.strip():
            if start is None:
                start = line_no
            statement.append(code)
        if start is not None and (";" in code or line_no == len(code_lines)):
            joined = " ".join(statement)
            if selected(start) and L1_VAR.search(joined) and L1_RAW.search(joined) and not L1_CANON_CALL.search(joined):
                _emit(found, path, start, "l1", "max L1 budget", side)
            statement = []
            start = None
        elif not code.strip() and not statement:
            start = None

    found.sort(key=lambda item: item[0])
    # One report per line: a signature can match the function rule and little else.
    deduped = []
    seen = set()
    for item in found:
        key = (item[0], item[2])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def parse_added_lines(diff_text):
    """1-based new-file line numbers of '+' lines in a unified diff."""
    added = set()
    new_line = None
    for line in diff_text.splitlines():
        header = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if header:
            new_line = int(header.group(1))
            continue
        if new_line is None:
            continue
        if line.startswith("+"):
            added.add(new_line)
            new_line += 1
        elif line.startswith("-") or line.startswith("\\"):
            continue
        else:
            new_line += 1
    return added


def _b_path(header):
    """Repo-relative path from a ``diff --git a/... b/...`` header, or None."""
    quoted = re.fullmatch(r'diff --git "a/(.*)" "b/(.*)"', header)
    if quoted:
        return quoted.group(2).replace('\\"', '"').replace("\\\\", "\\")
    plain = re.fullmatch(r"diff --git a/(.*) b/(.*)", header)
    if plain is None:
        return None
    if plain.group(1) == plain.group(2):
        return plain.group(2)
    marker = " b/"
    index = header.rfind(marker)
    if index < 0:
        return None
    return header[index + len(marker) :]


def added_lines_by_path(diff_text):
    """Map a repo-relative path to the 1-based lines added in that file's diff."""
    found = {}
    for chunk in re.split(r"(?=^diff --git )", diff_text, flags=re.M):
        if not chunk.startswith("diff --git "):
            continue
        header = chunk.split("\n", 1)[0]
        path = _b_path(header)
        if path:
            found[path] = parse_added_lines(chunk)
    return found


def _git_diff(args):
    try:
        return subprocess.check_output(
            ["git", "diff", "-U0", "--no-ext-diff", *args, "--"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def commit_added_lines():
    """Added lines for every file in this commit. None when git cannot answer.

    The diff is of the whole commit. Caller-supplied paths are not command arguments.
    ``PRE_COMMIT_FROM_REF`` and ``PRE_COMMIT_TO_REF`` are passed only after ``safe_git_rev``.
    """
    from_ref = os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF")
    if from_ref or to_ref:
        safe_from = safe_git_rev(from_ref)
        safe_to = safe_git_rev(to_ref)
        if safe_from is None or safe_to is None:
            return None
        diff = _git_diff([safe_from, safe_to])
    else:
        diff = _git_diff(["--cached"])
        if diff is not None and not diff.strip():
            diff = _git_diff([])
    if diff is None:
        return None
    return added_lines_by_path(diff)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Source files to check")
    parser.add_argument(
        "--all-lines",
        action="store_true",
        help="Scan every line. The pre-commit hook scans added lines only.",
    )
    parser.add_argument("--list", action="store_true", help="Print the catalog of existing helpers and exit")
    args = parser.parse_args(argv)

    if args.list:
        print(CATALOG)
        return 0
    if not args.files:
        parser.error("pass files to check, or --list")

    added = None
    if not args.all_lines:
        added = commit_added_lines()
        if added is None:
            print(
                "check_redefined_helpers: git diff failed or the revision was rejected; skipping (fail open)",
                file=sys.stderr,
            )
            return 0

    violations = []
    for filepath in args.files:
        resolved = safe_repo_path(filepath)
        if resolved is None:
            print(f"check_redefined_helpers: refusing path outside the repo: {filepath}", file=sys.stderr)
            return 1
        rel = os.path.relpath(resolved, REPO_ROOT).replace("\\", "/")
        try:
            with open(resolved, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError as exc:
            print(f"check_redefined_helpers: cannot read {rel}: {exc}", file=sys.stderr)
            return 1
        only = None if added is None else added.get(rel, set())
        for line_no, _kind, name, advice in find_violations(rel, text, only):
            violations.append((rel, line_no, name, advice))

    if not violations:
        return 0

    print("Redefined a helper or constant that already exists.", file=sys.stderr)
    print("Use the shared one. Host and device headers are not interchangeable.", file=sys.stderr)
    print("", file=sys.stderr)
    for path, line_no, name, advice in violations:
        print(f"  {path}:{line_no}: {name}", file=sys.stderr)
        print(f"    {advice}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Catalog: python3 scripts/check_redefined_helpers.py --list", file=sys.stderr)
    print("Deliberate exception, on the same line: // allow-redefined-helper: <why>", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
