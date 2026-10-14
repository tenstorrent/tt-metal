#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Forbid `get_read_ptr(cb)` when the most recent CB-state op on that cb id was
`cb_reserve_back` (writer scratchpad) rather than `cb_wait_front` (read).

See issue #39432. `get_read_ptr` returns the CB's read pointer, which does NOT
advance with `cb_reserve_back` / `cb_push_back`. Using it as a write scratchpad
silently aliases successive reservations to the same L1 address. Multi-iter
loops that intend distinct scratch slots end up sharing one, mangling data.

Heuristic, scoped per function via brace depth. Tracks the last cb_reserve_back
or cb_wait_front per CB identifier; flags get_read_ptr(X) when the last op on X
was cb_reserve_back. Only CB identifiers that look like a single C identifier
are tracked (skips array indexing, expressions, etc.).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RESERVE_RE = re.compile(r"\bcb_reserve_back\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*[,)]")
PUSH_RE = re.compile(r"\bcb_push_back\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*[,)]")
WAIT_RE = re.compile(r"\bcb_wait_front\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*[,)]")
POP_RE = re.compile(r"\bcb_pop_front\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*[,)]")
READ_PTR_RE = re.compile(r"\bget_read_ptr\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")

LINE_COMMENT_RE = re.compile(r"//.*$")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# Pre-existing violations not addressed in this PR. Each was audited to be safe
# by construction (single-tile CB, sharded input CB, or single-shot reserve
# where read_ptr == write_ptr at the call site). They're tracked separately so
# the lint can still catch NEW violations. Remove an entry as each is fixed.
KNOWN_VIOLATIONS: frozenset[str] = frozenset(
    {
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/kernels/dataflow/writer_llama_reduce_scatter.cpp",
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_boltz/device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_boltz_sharded.cpp",
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp",
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_height_width_sharded.cpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp",
        "tt-train/sources/ttml/metal/ops/select_target_logit/device/kernels/dataflow/select_target_logit_reader.cpp",
        "tt-train/sources/ttml/metal/ops/subtract_at_target/device/kernels/dataflow/subtract_at_target_reader.cpp",
        "tt-train/sources/ttml/metal/ops/cross_entropy_fw/device/kernels/dataflow/reader_cross_entropy_fw_interleaved_start_id.cpp",
        # Reviewed-safe, NOT a to-fix violation: this writer reserves N packet-header slots at once,
        # takes the pinned base via get_read_ptr a single time (nothing pops this scratch CB, so the
        # read ptr never moves), then manually offsets each header by sizeof(PACKET_HEADER_TYPE) — the
        # headers are distinct and there is no per-iteration get_read_ptr aliasing. Do not "fix" it.
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/writer_all_to_all_combine.cpp",
    }
)


def strip_comments(text: str) -> str:
    # Preserve newlines inside block comments so line numbers stay accurate.
    text = BLOCK_COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    return "\n".join(LINE_COMMENT_RE.sub("", line) for line in text.splitlines())


def check_file(path: Path) -> list[tuple[Path, int, str, int]]:
    try:
        src = path.read_text()
    except (UnicodeDecodeError, OSError):
        return []
    src = strip_comments(src)

    # Per-CB state. "writing" iff cb_reserve_back was the most recent state-setting
    # op AND no matching cb_push_back has happened yet. get_read_ptr while writing
    # is the bug. cb_push_back exits writing; cb_wait_front enters reading;
    # cb_pop_front exits reading.
    state: dict[str, tuple[str, int]] = {}
    depth = 0
    errors: list[tuple[Path, int, str, int]] = []

    for lineno, line in enumerate(src.splitlines(), start=1):
        for cb in RESERVE_RE.findall(line):
            state[cb] = ("writing", lineno)
        for cb in PUSH_RE.findall(line):
            if state.get(cb, ("", 0))[0] == "writing":
                state[cb] = ("after_push", lineno)
        for cb in WAIT_RE.findall(line):
            state[cb] = ("reading", lineno)
        for cb in POP_RE.findall(line):
            if state.get(cb, ("", 0))[0] == "reading":
                state[cb] = ("after_pop", lineno)
        for cb in READ_PTR_RE.findall(line):
            prev = state.get(cb)
            if prev and prev[0] == "writing":
                errors.append((path, lineno, cb, prev[1]))

        depth += line.count("{") - line.count("}")
        if depth <= 0:
            depth = 0
            state.clear()

    return errors


KERNEL_PATH_RE = re.compile(r"(^|/)kernels?/")
KERNEL_SUFFIXES = {".cpp", ".cc", ".h", ".hpp"}


def is_kernel_source(p: Path) -> bool:
    return p.suffix in KERNEL_SUFFIXES and KERNEL_PATH_RE.search(str(p)) is not None


def iter_default_paths() -> list[Path]:
    roots = [Path("tt_metal"), Path("ttnn"), Path("tt-train")]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and is_kernel_source(p):
                out.append(p)
    return out


def main(argv: list[str]) -> int:
    args = [Path(a) for a in argv[1:]]
    if args:
        paths = [p for p in args if p.is_file() and is_kernel_source(p)]
    else:
        paths = iter_default_paths()

    all_errors: list[tuple[Path, int, str, int]] = []
    for p in paths:
        if str(p) in KNOWN_VIOLATIONS:
            continue
        all_errors.extend(check_file(p))

    if not all_errors:
        return 0

    for path, lineno, cb, reserve_lineno in all_errors:
        print(
            f"{path}:{lineno}: error: get_read_ptr({cb}) follows "
            f"cb_reserve_back({cb}) at line {reserve_lineno} — "
            f"use get_write_ptr({cb}). See issue #39432.",
            file=sys.stderr,
        )
    print(
        "\nget_read_ptr() returns the CB's read pointer, which does not advance "
        "with cb_reserve_back/cb_push_back.\nUsing it as a write scratchpad aliases "
        "successive reservations to the same L1 address.\nUse get_write_ptr() when "
        "the CB is reserved (about to be written).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
