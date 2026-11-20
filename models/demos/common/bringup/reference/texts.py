# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F276: several real texts for position runs (owner 2026-11-19: no tuning to one trace).

A position run (testing/profile.PositionGolden: BRINGUP_POS, test_skip_positions, the census) needs only realistic
token ids: nothing is compared. An MoE's work depends on them (routing), so every perf choice is picked on some texts
and validated on others, and walls are reported as a spread over texts.

    BRINGUP_POS_TOKENS = prose | agentic | code | math | multilingual | random     (default: prose)

Every text is built from files already on the machine (no downloads), tokenized with the model's tokenizer, and cached
in $ART/input/texts/<name>.json with the sha256 of its ids and a description of its sources. A text shorter than the
run is repeated (the record says how often). Every metric of a run with BRINGUP_POS_TOKENS set carries ``_t<name>``
(core/metrics.suffixed), so one gate can hold several texts at one position.

    python -m models.demos.common.bringup.reference.texts --spec S [--name agentic]   # build / show the texts
"""

from __future__ import annotations

import argparse
import ast
import glob
import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch

from models.demos.common.bringup.core.spec import CODE_ROOT, Spec

ENV = "BRINGUP_POS_TOKENS"
DEFAULT = "prose"
TARGET_TOKENS = 1_100_000  # enough for a 1M position without repeating, where the sources allow
AGENT_DIRS_ENV = "BRINGUP_TEXT_AGENT_DIRS"  # ':'-separated dirs of agent stream-json transcripts


def name() -> str:
    return os.environ.get(ENV) or DEFAULT


def explicit() -> str | None:
    """The text a run named (its metrics then carry _t<name>), None for the default."""
    v = os.environ.get(ENV)
    return v if v and v != "text" else None


def _agent_dirs(spec: Spec) -> list[Path]:
    env = os.environ.get(AGENT_DIRS_ENV)
    if env:
        return [Path(p) for p in env.split(":") if p]
    return sorted(Path("/scratch").glob(f"{os.environ.get('USER', '*')}/bringup/*/runs/*/*/agents"))


def _agentic(spec: Spec) -> tuple[str, str]:
    """Agentic tool use (AgentX-like): the bring-up agents' own Claude Code sessions (stream-json), as a transcript of
    assistant text, tool calls (name + JSON input) and tool results (code, logs, diffs)."""
    parts, files = [], []
    for d in _agent_dirs(spec):
        for f in sorted(d.glob("*.jsonl")):
            files.append(str(f))
            for line in f.read_text(errors="replace").splitlines():
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                msg = ev.get("message") if isinstance(ev, dict) else None
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, list):
                    continue
                for c in content:
                    t = c.get("type")
                    if t == "text" and c.get("text"):
                        parts.append(f"<assistant>\n{c['text']}\n</assistant>")
                    elif t == "tool_use":
                        parts.append(
                            f"<tool_call name=\"{c.get('name')}\">\n{json.dumps(c.get('input'))}\n</tool_call>"
                        )
                    elif t == "tool_result":
                        r = c.get("content")
                        r = "\n".join(x.get("text", "") for x in r if isinstance(x, dict)) if isinstance(r, list) else r
                        if r:
                            parts.append(f"<tool_result>\n{str(r)[:20000]}\n</tool_result>")
    return "\n".join(parts), f"{len(files)} agent transcripts under {', '.join(map(str, _agent_dirs(spec)))}"


def _code(spec: Spec) -> tuple[str, str]:
    """Source code: the repo's C++ op sources and their Python, sorted (deterministic)."""
    pats = ["ttnn/cpp/ttnn/operations/**/*.cpp", "ttnn/cpp/ttnn/operations/**/*.hpp", "ttnn/ttnn/operations/**/*.py"]
    files = sorted({f for p in pats for f in glob.glob(str(CODE_ROOT / p), recursive=True)})
    text = "\n".join(f"// file: {Path(f).relative_to(CODE_ROOT)}\n{Path(f).read_text(errors='replace')}" for f in files)
    return text, f"{len(files)} files: {', '.join(pats)}"


def _math(spec: Spec) -> tuple[str, str]:
    """Mathematics: the docstrings of sympy and mpmath (definitions, formulas, worked examples)."""
    import mpmath
    import sympy

    parts, n = [], 0
    for pkg in (sympy, mpmath):
        for f in sorted(Path(pkg.__file__).parent.rglob("*.py")):
            if "/tests/" in str(f):
                continue
            try:
                tree = ast.parse(f.read_text(errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    doc = ast.get_docstring(node)
                    if doc and len(doc) > 200:
                        parts.append(doc)
            n += 1
    return "\n\n".join(parts), f"docstrings of {n} files of sympy {sympy.__version__}, mpmath {mpmath.__version__}"


_TROFF = re.compile(r"^\.[A-Za-z]+\s?|\\f[BIRP]|\\-|\\\(..|\\&|\\e|\\\*\(..|\\s[+-]?\d")


def _multilingual(spec: Spec) -> tuple[str, str]:
    """Other languages: the system's localized man pages (Chinese, Japanese, Russian, German), troff markup stripped."""
    parts, n = [], 0
    for lang in ("zh_CN", "ja", "ru", "de"):
        for f in sorted(Path("/usr/share/man", lang).rglob("*.gz")):
            try:
                raw = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            lines = [_TROFF.sub("", ln) for ln in raw.splitlines() if not ln.startswith(('.\\"', "'\\\""))]
            parts.append("\n".join(ln for ln in lines if ln.strip()))
            n += 1
    return "\n\n".join(parts), f"{n} man pages from /usr/share/man/{{zh_CN,ja,ru,de}}"


def _prose(spec: Spec) -> tuple[str, str]:
    """Prose: the canonical prompt's source book, whole (the canonical prompt holds only its first ~56k tokens)."""
    from models.demos.common.bringup.reference import prompt

    path, text = prompt._source(spec)
    return text, f"the canonical prompt's source {path.name}"


# F287 (owner 2026-11-19): public coding-agent traces (nvidia/Open-SWE-Traces, CC-BY-4.0, already on /data), rendered as
# the model sees them: the checkpoint's own chat template with the system prompt, tool schemas, tool calls, tool
# results and reasoning. Sessions in a fixed shuffled order; a text holds whole sessions up to TARGET_TOKENS, its record
# lists each session (repo, language, start, n) so a pick can be split by repo. "<name>_s<k>" is the k-th such slice.
TRACES_DIR = os.environ.get("BRINGUP_TRACES_DIR", "/data/dnijemcevic/bringup/agent_texts_20261119")
TRACES = {
    "agentic_cc": "claudecode_sample.parquet",
    "agentic_ds": "openhands_deepseek_v4_flash_0731_scale-swe_train-00000-of-00011.parquet",
    "agentic_ds_ml": "openhands_dsv4flash_swerebench_sample.parquet",
}


def _tool_schema(t):
    t = json.loads(t) if isinstance(t, str) else t
    if "function" in t:
        return t
    params = t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {"name": t["name"], "description": t.get("description", ""), "parameters": params},
    }


def _render_session(tok, row) -> str:
    msgs = []
    for m in row["messages"]:
        m = dict(m)
        calls = []
        for c in m.get("tool_calls") or []:
            fn = dict(c.get("function") or {})
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except ValueError:
                    pass
            calls.append({**c, "function": fn})
        if calls:
            m["tool_calls"] = calls
        else:
            m.pop("tool_calls", None)
        if not m.get("reasoning_content"):
            m.pop("reasoning_content", None)
        msgs.append({k: v for k, v in m.items() if v is not None})
    tools = [_tool_schema(t) for t in (row.get("tools") or [])]
    return tok.apply_chat_template(msgs, tools=tools or None, tokenize=False)


def _traces(spec, which: str, tok):
    import random

    import pyarrow.parquet as pq

    base, _, k = which.partition("_s")
    k = int(k or 0)
    table = pq.read_table(
        Path(TRACES_DIR) / TRACES[base], columns=["instance_id", "repo", "language", "messages", "tools"]
    )
    order = list(range(table.num_rows))
    random.Random(f"{base}-F287").shuffle(order)
    ids, sessions, used = [], [], 0
    for i in order:
        row = table.slice(i, 1).to_pylist()[0]
        n_ids = tok(_render_session(tok, row), add_special_tokens=False)["input_ids"]
        if used < k * TARGET_TOKENS:  # earlier slices
            used += len(n_ids)
            continue
        if len(ids) + len(n_ids) > TARGET_TOKENS:
            if sessions:
                break
            n_ids = n_ids[:TARGET_TOKENS]
        sessions.append(
            {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "language": row["language"],
                "start": len(ids),
                "n": len(n_ids),
            }
        )
        ids += n_ids
    return (
        ids,
        sessions,
        f"{len(sessions)} sessions of {TRACES[base]} (nvidia/Open-SWE-Traces, CC-BY-4.0), slice {k}, chat template",
    )


BUILDERS = {"prose": _prose, "agentic": _agentic, "code": _code, "math": _math, "multilingual": _multilingual}


def path(spec: Spec, which: str) -> Path:
    return spec.art / "input" / "texts" / f"{which}.json"


def build(spec: Spec, which: str, tok=None) -> dict:
    """Tokenize the text once and cache it (the ids' sha256 pins it for every later run)."""
    p = path(spec, which)
    if p.exists():
        rec = json.loads(p.read_text())
        if hashlib.sha256(json.dumps(rec["token_ids"]).encode()).hexdigest() != rec["sha256"]:
            raise ValueError(f"{p}: token ids do not match their recorded hash")
        return rec
    from models.demos.common.bringup.reference.golden import tokenizer

    tok = tok or tokenizer(spec)
    sessions = None
    if which.partition("_s")[0] in TRACES:  # F287
        ids, sessions, source = _traces(spec, which, tok)
    elif which in BUILDERS:
        text, source = BUILDERS[which](spec)
        ids = tok(text[: TARGET_TOKENS * 8], add_special_tokens=False)["input_ids"][:TARGET_TOKENS]
    else:
        raise ValueError(f"{ENV}={which!r}: use one of {sorted(BUILDERS) + sorted(TRACES)} (+ _s<k>) or random")
    rec = {
        "name": which,
        "source": source,
        "n": len(ids),
        "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "token_ids": ids,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **({"sessions": sessions} if sessions is not None else {}),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, p)
    return rec


def tokens(spec: Spec, n: int, which: str | None = None) -> tuple[torch.Tensor, dict]:
    """n token ids of the text (repeated if it is shorter) and what they are."""
    which = which or name()
    if which == "text":
        which = DEFAULT
    rec = build(spec, which)
    ids = torch.tensor(rec["token_ids"], dtype=torch.int64)
    info = {k: rec[k] for k in ("name", "source", "n", "sha256")}
    info["repeats"] = -(-n // len(ids))
    return ids.repeat(info["repeats"])[:n], info


def main(argv=None):
    from models.demos.common.bringup.reference.golden import load_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--name", action="append")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    for which in a.name or sorted(BUILDERS):
        rec = build(spec, which)
        print(f"{which}: {rec['n']:,} tokens, sha {rec['sha256'][:12]}, {rec['source']} -> {path(spec, which)}")


if __name__ == "__main__":
    main()
