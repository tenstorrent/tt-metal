#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every replay-buffer record goes through ckernel::load_replay_buf.

Blackhole instruction gathering is incompatible with loading the replay buffer. load_replay_buf
turns gathering off for the record and back on after it (under ENABLE_GATHERING); a record issued
any other way runs with gathering on. On Wormhole the helper only records, so the rule costs
nothing there and keeps shared kernels, which build for both, correct on Blackhole.

Flagged, in Blackhole, Wormhole and architecture-shared kernel code:
  * lltt::record outside load_replay_buf's own body;
  * TTI_REPLAY / TT_REPLAY / TT_OP_REPLAY / TT_REPLAY_VALID whose load_mode (4th argument) is
    not the literal 0 or false: a record, or a mode the checker cannot prove is a playback;
  * __builtin_rvtt_ttreplay whose record flag (4th argument of the lltt form) is not 0/false.

Not seen: records the compiler makes on its own (sfpi -mtt-tensix-optimize-replay). Gathering
builds pass -mno-tt-tensix-optimize-replay; the ELF test in test_gathering_config.py checks the
built kernels.

Quasar is out of scope: its records do not need the bracket and it keeps its own helpers.

Zero findings, no baseline: the hook fails on any finding.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

SOURCE_SUFFIXES = (".h", ".hpp", ".cpp", ".cc", ".inl")

# Skipped: Quasar, vendored code, build output and virtualenvs.
EXCLUDED = re.compile(
    r"(^|/)([^/]*quasar[^/]*|third_party|\.venv[^/]*|build[^/]*|\.git|\.cpmcache)(/|$)",
    re.IGNORECASE,
)

# The helpers themselves: the one lltt::record they each wrap is the point of the rule.
HELPER_FILES = {
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h",
    "tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel.h",
}
HELPER_NAME = "load_replay_buf"

REPLAY_MACROS = ("TTI_REPLAY", "TT_REPLAY", "TT_OP_REPLAY", "TT_REPLAY_VALID")
TOKEN = re.compile(
    r"\b(lltt\s*::\s*record|__builtin_rvtt_ttreplay|" + "|".join(REPLAY_MACROS) + r")\b"
)
PLAYBACK_MODE = re.compile(r"^\(?\s*(0|0u|0U|false)\s*\)?$")


def strip_comments_and_strings(text: str) -> str:
    """Blank out comments and string/char literals, keeping offsets and newlines."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(
                c
                + re.sub(r"[^\n]", " ", text[i + 1 : j - 1])
                + (c if j - i > 1 else "")
            )
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def blank_define_lines(text: str) -> str:
    """Macro definitions (#define, with continuations) define the instruction, not a use of it."""
    lines = text.split("\n")
    in_define = False
    for k, line in enumerate(lines):
        if in_define or re.match(r"\s*#\s*define\b", line):
            in_define = line.rstrip().endswith("\\")
            lines[k] = " " * len(line)
    return "\n".join(lines)


def call_args(text: str, open_paren: int):
    """Top-level comma-separated arguments of the call whose '(' is at open_paren, or None."""
    depth, start, args = 0, open_paren + 1, []
    for k in range(open_paren, len(text)):
        ch = text[k]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                args.append(text[start:k].strip())
                return args
        elif ch == "," and depth == 1:
            args.append(text[start:k].strip())
            start = k + 1
    return None


def helper_body_spans(text: str):
    """(start, end) offsets of every load_replay_buf definition body in text."""
    spans = []
    for m in re.finditer(r"\b" + HELPER_NAME + r"\s*\([^;{]*\)\s*\{", text):
        depth = 0
        for k in range(m.end() - 1, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((m.end(), k))
                    break
    return spans


def check_text(rel: str, raw: str):
    text = blank_define_lines(strip_comments_and_strings(raw))
    allowed = helper_body_spans(text) if rel in HELPER_FILES else []
    findings = []
    for m in TOKEN.finditer(text):
        name = re.sub(r"\s+", "", m.group(1))
        line = text.count("\n", 0, m.start()) + 1
        if any(a <= m.start() < b for a, b in allowed):
            continue
        rest = text[m.end() :]
        lead = len(rest) - len(rest.lstrip())
        if name == "lltt::record":
            findings.append((rel, line, "lltt::record outside load_replay_buf"))
            continue
        if not rest[lead:].startswith("("):
            continue  # a name, not a call (e.g. a using-declaration)
        args = call_args(text, m.end() + lead)
        if args is None or len(args) < 4:
            findings.append((rel, line, f"{name}: cannot read the load_mode argument"))
            continue
        mode = args[-1] if name == "__builtin_rvtt_ttreplay" else args[3]
        if not PLAYBACK_MODE.match(mode):
            findings.append(
                (
                    rel,
                    line,
                    f"{name} with load_mode {mode!r}: a record, or not provably a playback",
                )
            )
    return findings


def candidate_files(paths):
    if paths:
        files = [Path(p) for p in paths]
    else:
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout
        files = [REPO / p for p in out.splitlines()]
    for f in files:
        f = f if f.is_absolute() else (REPO / f)
        try:
            rel = f.resolve().relative_to(REPO).as_posix()
        except ValueError:
            rel = f.as_posix()
        if not rel.endswith(SOURCE_SUFFIXES) or EXCLUDED.search(rel) or not f.is_file():
            continue
        yield rel, f


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "files", nargs="*", help="files to check (default: every tracked source file)"
    )
    args = parser.parse_args(argv)
    # A change to the checker rescans every file.
    files = (
        []
        if any(Path(p).name == Path(__file__).name for p in args.files)
        else args.files
    )
    findings = []
    for rel, f in candidate_files(files):
        raw = f.read_text(errors="replace")
        if not TOKEN.search(raw):
            continue
        findings.extend(check_text(rel, raw))
    for rel, line, msg in findings:
        print(f"{rel}:{line}: {msg}; record through ckernel::load_replay_buf")
    if findings:
        print(
            f"\n{len(findings)} replay record(s) outside load_replay_buf.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
