#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build the M3-tokenized code prompt used as the "code" input of the op-profile campaign.

Layout (same as results_torus/skew_check/code, but longer): the chat-template header of the M3-tokenized
longbook_56320 golden (system / developer / user turn opening, taken verbatim from its token ids), then the
repository .py files of --roots concatenated in sorted order, each introduced by "# file: <path>", then the
golden's closing tokens (end of user turn + assistant turn opening). The repeated SPDX license header
lines are stripped from every file. Files are appended whole until --max-tokens would be exceeded.

  python_env/bin/python m3_budget_study/results_ops/tools/make_code_corpus.py \
      --out m3_budget_study/results_ops/inputs/code_m3/metadata.json

Writes metadata.json ({"token_ids", "n_tokens", + provenance}) and files.txt (the files used, in order).
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[3]
TOKENIZER = "/mnt/weka/model-weights/llm/minimax/MiniMax-M3/tokenizer.json"
PROSE = "/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden/longbook_56320/metadata.json"
ROOTS = ["models/demos/minimax_m3", "models/demos/common", "models/demos/deepseek_v3_d_p"]
USER_OPEN = "]~b]user\n"
TURN_CLOSE = "[e~["


def strip_spdx(text):
    """Drop the SPDX license lines (identical boilerplate in every file) and the blank lines they leave
    at the top of the file."""
    kept = "".join(l for l in text.splitlines(keepends=True) if not l.startswith("# SPDX-"))
    return kept.lstrip("\n")


def chat_frame(tok, ids):
    """(header, tail) token lists of the golden: everything up to the user turn's content, and from the
    end of the user turn onward."""
    head = next(k for k in range(1, 2000) if tok.decode(ids[:k], skip_special_tokens=False).endswith(USER_OPEN))
    tail = next(k for k in range(1, 200) if tok.decode(ids[-k:], skip_special_tokens=False).startswith(TURN_CLOSE))
    return ids[:head], ids[-tail:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+", default=ROOTS)
    ap.add_argument("--max-tokens", type=int, default=600_000)
    ap.add_argument("--min-tokens", type=int, default=65_536)
    args = ap.parse_args()

    tok = Tokenizer.from_file(TOKENIZER)
    prose = json.load(open(PROSE))["token_ids"]
    header, tail = chat_frame(tok, prose)

    files = []
    for root in args.roots:
        files += sorted(p for p in (REPO / root).rglob("*.py") if "__pycache__" not in p.parts)

    body, used = [], []
    budget = args.max_tokens - len(header) - len(tail)
    for p in files:
        rel = p.relative_to(REPO).as_posix()
        text = strip_spdx(p.read_text(errors="replace"))
        if not text.strip():
            continue
        ids = tok.encode(f"# file: {rel}\n{text}\n\n", add_special_tokens=False).ids
        if len(body) + len(ids) > budget:
            break
        body += ids
        used.append(rel)

    token_ids = header + body + tail
    assert len(token_ids) >= args.min_tokens, f"only {len(token_ids)} tokens"
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", *args.roots], capture_output=True, text=True
    ).stdout.split("\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "token_ids": token_ids,
        "n_tokens": len(token_ids),
        "prompt_source": (
            "longbook_56320 chat-template header + repository .py files (SPDX header lines stripped, "
            "'# file: <path>' before each) + longbook_56320 closing tokens; M3 tokenizer.json "
            "(tokenizers lib, add_special_tokens=False for the body)"
        ),
        "roots": args.roots,
        "n_files": len(used),
        "n_files_available": len(files),
        "header_tokens": len(header),
        "tail_tokens": len(tail),
        "tokenizer": TOKENIZER,
        "git_sha": sha,
        "worktree_changes_in_roots": [l[3:] for l in dirty if l.strip()],
        "builder": "m3_budget_study/results_ops/tools/make_code_corpus.py",
    }
    json.dump(meta, open(out, "w"))
    (out.parent / "files.txt").write_text("\n".join(used) + "\n")
    print(f"{len(token_ids)} tokens from {len(used)}/{len(files)} files -> {out}")


if __name__ == "__main__":
    main()
