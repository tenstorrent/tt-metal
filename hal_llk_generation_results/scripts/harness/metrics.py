#!/usr/bin/env python3
"""
Metrics of a single run from a Claude Code transcript (stream-json):
  - tokenomics: input / output / cached tokens, cost, duration, number of steps
  - iterations: every pytest invocation, classified (pass / fail / compile
    error / hang / compile only / other), card resets,
    file edits, denied tool calls

  - process metrics (benchmark specification, 4.1-4.2): first full pass per test
    (the whole test selection as in the original, pass count >= original, 0 failed), time and tokens up to it,
    first full pass of the final code, Success@k, time and tokens per iteration outcome kind,
    reading of reference material and re-reading

Iteration = a run of the test with which the agent checks its code, including a compile failure;
compile only (--compile-producer), listing, denied calls and runs in which
no test was executed ("no tests ran") are not counted.

Version 2 (2026-11-18), corrections relative to version 1:
  1. A single Bash command with several pytest calls (e.g. producer + consumer, or STEP1 + STEP2)
     yields several runs: each pytest call is paired with its own final line
     ("N passed, M failed ... in X s"); v1 took only the first pass/fail count.
  2. The first full pass requires the SAME test selection as the original: -k equal to the run's filter
     (run_meta.txt: tests=test:k / op_filter) or the whole test file, or the same number of selected and
     deselected variants as the original; selection by IDs (::, @file), --lf, --deselect does not count.
     v1 accepted e.g. -k 'not add' (2748 passed > 1660).
  3. Without the original's pass count (no logs/baseline_<test>.txt and no --baseline) the first full
     pass is not counted (v1 then also accepted a pass of a single variant).
  4. Denied calls: also by the system/permission_denied event (tool_use_id), not only by text;
     v1 counted denied commands with a different message as error/hang iterations.
  5. When pytest reports a pass with no failures, the outcome is pass even if the word
     "timeout"/"error: " appears in the output (v1: hang / compile_error).
  6. "pytest @file": the test is determined from the content of the file the agent wrote with the Write tool.
  7. New: first full pass of the FINAL code (the first full pass after the last change of
     the agent's code: Edit/Write on tests/hal_llk/*.h|cpp or *_hal_test.cpp, or a Bash command
     that modifies such a file; heuristic).

Usage:
  metrics.py <run_dir | transcript.jsonl | archive.tar.gz> [--json out.json] [--md out.md]
             [--baseline test=N] [--k-filter test=expression]
  metrics.py run1.tar.gz run2.tar.gz ...        # comparison of several runs in one table
"""
import json
import re
import sys
import tarfile
from pathlib import Path

VERSION = 2


# ----------------------------------------------------------- loading ---
def _logs_files(src: str, want):
    """Contents of the files in logs/ (run dir, logs/transcript.jsonl or archive) for which want(name) holds."""
    out = {}
    p = Path(src)
    if p.is_file() and p.name.endswith(".jsonl"):
        p = p.parent.parent if p.parent.name == "logs" else p.parent
    if p.is_dir():
        for f in (p / "logs").glob("*"):
            if f.is_file() and want(f.name):
                out[f.name] = f.read_text(errors="replace")
    elif str(p).endswith((".tar.gz", ".tgz")):
        with tarfile.open(p) as t:
            for m in t.getmembers():
                n = Path(m.name).name
                if m.isfile() and Path(m.name).parent.name == "logs" and want(n):
                    out[n] = t.extractfile(m).read().decode(errors="replace")
    return out


def load_lines(src: str):
    p = Path(src)
    if p.is_dir():
        p = p / "logs" / "transcript.jsonl"
    if str(p).endswith((".tar.gz", ".tgz")):
        with tarfile.open(p) as t:
            m = next((m for m in t.getmembers() if m.name.endswith("logs/transcript.jsonl")), None)
            if m is None:
                raise SystemExit(f"{src}: no logs/transcript.jsonl in the archive")
            data = t.extractfile(m).read().decode(errors="replace")
        return data.splitlines()
    return p.read_text(errors="replace").splitlines()


def load_baselines(src: str) -> dict:
    """Final line of the original test by test name, from logs/baseline_<test>.txt:
    {test: {"passed": N, "skipped": N, "deselected": N, ...}}."""
    out = {}
    for name, text in _logs_files(src, lambda n: n.startswith("baseline_test_") and n.endswith(".txt")).items():
        sums = summaries(text)
        if sums:
            out[name[len("baseline_"):-4]] = sums[-1]
        else:
            m = re.findall(r"(\d+) passed", text)
            if m:
                out[name[len("baseline_"):-4]] = {"passed": int(m[-1])}
    return out


def load_k_filters(src: str) -> dict:
    """-k filter per test from logs/run_meta.txt (tests=test[:k] ...; older runs: op_filter for eltwise)."""
    meta = {}
    for text in _logs_files(src, lambda n: n == "run_meta.txt").values():
        for ln in text.splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                meta[k.strip()] = v.strip()
    out = {}
    if meta.get("tests"):
        for item in meta["tests"].split():
            name, _, k = item.partition(":")
            out[name] = k or None
    elif meta.get("op_filter"):
        out["test_eltwise_binary"] = meta["op_filter"]
    return out


ITER_KINDS = ("pass", "fail", "hang", "error", "compile_error", "other")
REF_CATEGORIES = {   # reading of material, by path in the tool input
    "hal_docs": re.compile(r"docs/hal/"),
    "raw_ref": re.compile(r"raw_ref/"),
    "spec": re.compile(r"llk_spec/"),
    "tests": re.compile(r"tests/(sources|python_tests|helpers)/"),
    "own_code": re.compile(r"tests/hal_llk/"),
}
READ_TOOLS = ("Read", "Grep", "Glob")
EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
BASH_READ = re.compile(r"^\s*(cat|head|tail|sed|less|more|ls|find|wc)\b")
SUCCESS_K = (5, 10, 15, 20, 25, 50)
CODE_FILE = re.compile(r"(tests/hal_llk/[^\s'\"]*\.(?:h|hh|hpp|c|cc|cpp|cxx|inl|inc)|_hal_test\.cpp)$")
BASH_MODIFY = re.compile(r"\bsed\s+(?:-\w*\s+)*-i|\bperl\s+-\w*i|\btee\b|\brm\s|\bmv\s|\bcp\s|\bpatch\b|\btruncate\b|\btouch\b|"
                         r">\s*\S*\.(?:h|hh|hpp|c|cc|cpp|cxx|inl|inc)\b")
BASH_CODE_NAME = re.compile(r"[\w./-]+\.(?:h|hh|hpp|c|cc|cpp|cxx|inl|inc)\b")


def parse_ts(s):
    from datetime import datetime
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


# -------------------------------------------------- pytest classification ---
PASSED = re.compile(r"(\d+) passed")
FAILED = re.compile(r"(\d+) failed")
ERRORS = re.compile(r"(\d+) errors?\b")
SUMMARY = re.compile(r"^[=\s]*((?:\d+ (?:passed|failed|skipped|deselected|errors?|xfailed|xpassed|warnings?|rerun)"
                     r"(?:, )?)+|no tests ran)[^\n]*? in [\d.]+s", re.M)
PYTEST_WORD = re.compile(r"\bpytest\b(?!\.)")
HANG_TEXT = re.compile(r"timed? ?out|timeout|hang|did not finish|tensix.*stuck|kernel.*not.*complete")
COMPILE_TEXT = re.compile(r"error: |fatal error|undefined reference|static assertion failed|compilation failed|compile error")
DENIED_TEXT = re.compile(r"permission|not allowed|denied", re.I)


def summaries(out: str):
    """All pytest final lines in the output, in order: [{"passed": N, "failed": N, ...}]."""
    res = []
    for m in SUMMARY.finditer(out):
        d = {}
        for n, w in re.findall(r"(\d+) (passed|failed|skipped|deselected|errors?|xfailed|xpassed|warnings?|rerun)", m.group(1)):
            w = {"errors": "error", "warnings": "warning"}.get(w, w)
            d[w] = d.get(w, 0) + int(n)
        res.append(d)
    return res


def classify_pytest(cmd: str, out: str, is_error: bool) -> str:
    """Classification of the whole command by text (v1; used when the output has no pytest final line)."""
    low = out.lower()
    if "--collect-only" in cmd or "--co " in cmd + " ":
        return "collect"
    if re.search(r"\bpytest\s+(?:[^|;&]*\s)?(?:--help|-h|--version|--fixtures|--markers)\b", cmd):
        return "help"
    if "permission" in low and ("denied" in low or "not allowed" in low) and len(out) < 600:
        return "denied"
    if HANG_TEXT.search(low):
        return "hang"
    if COMPILE_TEXT.search(low):
        return "compile_error"
    f = FAILED.search(out); p = PASSED.search(out)
    if f and int(f.group(1)) > 0:
        return "fail"
    if "--compile-producer" in cmd:
        return "compile_only"
    if p and not f:
        return "pass"
    if ERRORS.search(out):
        return "error"
    return "other" if not is_error else "error"


def segment_info(seg: str, written: dict) -> dict:
    """A single pytest call from a command: tests, -k, selection mode, producer/consumer."""
    tests = sorted(set(re.findall(r"(test_\w+)\.py", seg)))
    k = re.search(r"(?:^|\s)-k\s*(\"[^\"]*\"|'[^']*'|[^\s;&|)]+)", seg)
    at = re.search(r"(?:^|\s)@([^\s;&|]+)", seg)
    if at and not tests:
        tests = sorted(written.get(Path(at.group(1)).name, set()))
    return {
        "tests": tests,
        "k": k.group(1).strip("'\"").strip() if k else None,
        "nodeids": "::" in seg,
        "atfile": at.group(1) if at else None,
        "lf": bool(re.search(r"\s(--lf|--last-failed|--sw|--stepwise)\b", seg)),
        "deselect": "--deselect" in seg,
        "producer": "--compile-producer" in seg,
        "consumer": "--compile-consumer" in seg,
        "collect": bool(re.search(r"--collect-only|\s--co\b", seg)),
        "help": bool(re.search(r"\s(--help|-h|--version|--fixtures|--markers)\b", seg)),
    }


def kind_from_counts(seg: dict, c: dict, low: str) -> str:
    failed, passed, errors = c.get("failed", 0), c.get("passed", 0), c.get("error", 0)
    if seg["producer"]:
        return "compile_error" if (failed or errors) else "compile_only"
    if failed or errors:
        if HANG_TEXT.search(low):
            return "hang"
        if COMPILE_TEXT.search(low):
            return "compile_error"
        return "fail" if failed else "error"
    if passed:
        return "pass"
    return "empty"          # "no tests ran" or only skipped/deselected variants


def selection(seg: dict, c: dict, k_filter, base: dict) -> str:
    """'full' if the selection is the same as for the original (or the whole test file), otherwise a description of the subset.
    The same number of selected and deselected variants as for the original always means 'full' (e.g. --lf without earlier failures)."""
    if base and c and all(k in base for k in ("passed", "deselected")):
        sel = c.get("passed", 0) + c.get("failed", 0) + c.get("skipped", 0) + c.get("error", 0)
        if sel == base["passed"] + base.get("skipped", 0) and c.get("deselected", 0) == base["deselected"]:
            return "full"
    if seg["nodeids"]:
        return "subset:nodeids"
    if seg["atfile"]:
        return "subset:@file"
    if seg["lf"]:
        return "subset:--lf"
    if seg["deselect"]:
        return "subset:--deselect"
    if seg["k"] is None:
        return "full"
    if k_filter is not None and seg["k"] == k_filter:
        return "full"
    return "subset:-k"


# ------------------------------------------------------------- analysis ---
def analyze(lines, baselines=None, k_filters=None):
    baselines = baselines or {}
    k_filters = k_filters or {}
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    denied_ids = {e.get("tool_use_id") for e in events
                  if e.get("type") == "system" and e.get("subtype") == "permission_denied"}

    t0 = None
    usage_by_msg = {}           # message id -> usage (last copy; parts of the same message are not summed)
    reads = {k: {"calls": 0, "targets": set()} for k in REF_CATEGORIES}
    tool_uses = {}              # id -> (name, input)
    written = {}                # file name -> tests mentioned in the content (Write), for "pytest @file"
    runs = []                   # one record per pytest call
    code_changes = []           # number of records in runs at the moment of a successful change of the agent's code
    n_cmd = 0
    ambiguous = 0
    counts = {"tool_calls": 0, "denied": 0, "edits": 0, "writes": 0, "reads": 0,
              "bash": 0, "tt_smi_reset": 0, "subagents": 0, "mcp_calls": 0}
    usage_sum = {"input_tokens": 0, "output_tokens": 0,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    result = None
    model = None

    for e in events:
        t = e.get("type")
        ts = parse_ts(e["timestamp"]) if isinstance(e.get("timestamp"), str) else None
        if ts is not None and t0 is None:
            t0 = ts
        if t == "system" and e.get("subtype") == "init":
            model = e.get("model", model)
        elif t == "assistant":
            msg = e.get("message", {})
            for k in usage_sum:
                usage_sum[k] += int((msg.get("usage") or {}).get(k) or 0)
            if msg.get("usage"):
                usage_by_msg[msg.get("id") or len(usage_by_msg)] = msg["usage"]
            for c in msg.get("content", []):
                if c.get("type") != "tool_use":
                    continue
                counts["tool_calls"] += 1
                name, inp = c.get("name"), c.get("input", {})
                tool_uses[c.get("id")] = (name, inp)
                if name in READ_TOOLS or (name == "Bash" and BASH_READ.search(inp.get("command", ""))):
                    txt = json.dumps(inp)
                    target = inp.get("file_path") or inp.get("path") or inp.get("pattern") or inp.get("command", "")
                    for cat, rx in REF_CATEGORIES.items():
                        if rx.search(txt):
                            reads[cat]["calls"] += 1
                            reads[cat]["targets"].add(str(target)[:300])
                if name == "Write" and isinstance(inp.get("content"), str):
                    written[Path(inp.get("file_path", "")).name] = set(re.findall(r"(test_\w+)\.py", inp["content"]))
                if name == "Edit":
                    counts["edits"] += 1
                elif name == "Write":
                    counts["writes"] += 1
                elif name == "Read":
                    counts["reads"] += 1
                elif name and name.startswith("mcp__"):
                    counts["mcp_calls"] += 1
                elif name in ("Agent", "Task"):
                    counts["subagents"] += 1
                elif name == "Bash":
                    counts["bash"] += 1
                    if "tt-smi -r" in inp.get("command", ""):
                        counts["tt_smi_reset"] += 1
        elif t == "user":
            for c in (e.get("message", {}) or {}).get("content", []) or []:
                if not isinstance(c, dict) or c.get("type") != "tool_result":
                    continue
                out = text_of(c.get("content"))
                is_err = bool(c.get("is_error"))
                was_denied = c.get("tool_use_id") in denied_ids or (
                    is_err and DENIED_TEXT.search(out) and len(out) < 800)
                if was_denied:
                    counts["denied"] += 1
                name, inp = tool_uses.get(c.get("tool_use_id"), (None, {}))
                # change of the agent's code (for the first full pass of the final code)
                if not was_denied and not is_err:
                    if name in EDIT_TOOLS and CODE_FILE.search(inp.get("file_path", "") or inp.get("notebook_path", "")):
                        code_changes.append(len(runs))
                    elif name == "Bash":
                        bc = inp.get("command", "")
                        if (BASH_MODIFY.search(bc) and BASH_CODE_NAME.search(bc)
                                and ("hal_llk" in bc or "_hal_test.cpp" in bc)):
                            code_changes.append(len(runs))
                if name == "Bash" and PYTEST_WORD.search(inp.get("command", "")):
                    cmd = inp["command"]
                    n_cmd += 1
                    cum = {k: sum(int(u.get(k) or 0) for u in usage_by_msg.values()) for k in usage_sum}
                    base_rec = {"n": n_cmd, "elapsed_min": round((ts - t0) / 60, 2) if ts is not None and t0 is not None else None,
                                "cum_tokens": cum, "cmd": cmd[:160]}
                    pos = [m.start() for m in PYTEST_WORD.finditer(cmd)]
                    segs = [segment_info(cmd[a:b], written) for a, b in zip(pos, pos[1:] + [len(cmd)])]
                    real = [s for s in segs if not s["collect"] and not s["help"]]
                    sums = summaries(out)
                    low = out.lower()
                    pairs = None
                    if was_denied:
                        kind = "denied"
                    elif not real:
                        kind = "collect" if any(s["collect"] for s in segs) else "help"
                    elif sums and len(sums) == len(real):
                        pairs = list(zip(real, sums))
                    elif sums and len(real) == 1:
                        pairs = [(real[0], s) for s in sums]
                    else:
                        if sums:
                            ambiguous += 1
                        kind = classify_pytest(cmd, out, is_err)
                    if pairs is None:
                        u = {"tests": sorted({t for s in (real or segs) for t in s["tests"]}),
                             "k": next((s["k"] for s in (real or segs) if s["k"]), None),
                             "nodeids": any(s["nodeids"] for s in segs), "atfile": next((s["atfile"] for s in segs if s["atfile"]), None),
                             "lf": any(s["lf"] for s in segs), "deselect": any(s["deselect"] for s in segs),
                             "producer": any(s["producer"] for s in segs), "consumer": any(s["consumer"] for s in segs)}
                        p = PASSED.search(out); f = FAILED.search(out)
                        cnt = sums[0] if sums else None
                        pairs_out = [(u, cnt, kind, int(p.group(1)) if p else None, int(f.group(1)) if f else None,
                                      bool(sums) and len(real) > 1)]
                    else:
                        pairs_out = [(s, cnt, kind_from_counts(s, cnt, low), cnt.get("passed", 0), cnt.get("failed", 0), False)
                                     for s, cnt in pairs]
                    for j, (s, cnt, kind, passed, failed, amb) in enumerate(pairs_out, 1):
                        test = s["tests"][0] if len(s["tests"]) == 1 else None
                        base = baselines.get(test[:-4] if test and test.endswith("_hal") else test) if test else None
                        kf = k_filters.get(test[:-4] if test and test.endswith("_hal") else test) if test else None
                        runs.append(dict(base_rec, seg=j, kind=kind, tests=s["tests"], passed=passed, failed=failed,
                                         k=s["k"], selection=selection(s, cnt, kf, base), consumer=s["consumer"],
                                         ambiguous=amb))
        elif t == "result":
            result = e

    # tokenomics: the official numbers from 'result' take precedence (they include subagents)
    tok = dict(usage_sum)
    cost = None
    per_model = {}
    if result:
        cost = result.get("total_cost_usd")
        mu = result.get("modelUsage") or {}
        if mu:
            tok = {"input_tokens": 0, "output_tokens": 0,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
            for m, u in mu.items():
                per_model[m] = {
                    "input_tokens": u.get("inputTokens", 0),
                    "output_tokens": u.get("outputTokens", 0),
                    "cache_read_input_tokens": u.get("cacheReadInputTokens", 0),
                    "cache_creation_input_tokens": u.get("cacheCreationInputTokens", 0),
                    "cost_usd": u.get("costUSD"),
                }
                for k in tok:
                    tok[k] += per_model[m][k]
        elif result.get("usage"):
            tok = {k: int(result["usage"].get(k) or 0) for k in tok}

    total_in = tok["input_tokens"] + tok["cache_creation_input_tokens"] + tok["cache_read_input_tokens"]
    kinds = {}
    for r in runs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    device_runs = [r for r in runs if r["kind"] in ("pass", "fail", "hang", "error", "other")]
    first_pass = next((r["n"] for r in runs if r["kind"] == "pass"), None)
    proc = process_metrics(runs, baselines, usage_sum, code_changes)
    proc["ambiguous_commands"] = ambiguous
    ref_reads = {k: {"calls": v["calls"], "distinct": len(v["targets"]), "rereads": v["calls"] - len(v["targets"]),
                     "share_of_tool_calls": round(v["calls"] / counts["tool_calls"], 3) if counts["tool_calls"] else None}
                 for k, v in reads.items()}
    by_test = {}
    for r in runs:
        for t in r["tests"] or ["(unknown)"]:
            by_test.setdefault(t, {})
            by_test[t][r["kind"]] = by_test[t].get(r["kind"], 0) + 1

    return {
        "metrics_version": VERSION,
        "model": model,
        "duration_min": round(result.get("duration_ms", 0) / 60000, 1) if result else None,
        "duration_api_min": round(result.get("duration_api_ms", 0) / 60000, 1) if result else None,
        "num_turns": result.get("num_turns") if result else None,
        "finished": bool(result) and result.get("subtype") == "success",
        "cost_usd": round(cost, 2) if isinstance(cost, (int, float)) else None,
        "tokens": tok,
        "tokens_total_input": total_in,
        "cache_hit_ratio": round(tok["cache_read_input_tokens"] / total_in, 3) if total_in else None,
        "tokens_per_model": per_model,
        "pytest_runs_total": n_cmd,
        "pytest_invocations_total": len(runs),
        "pytest_runs_by_kind": kinds,
        "pytest_runs_by_test": by_test,
        "process": proc,
        "reference_reads": ref_reads,
        "device_test_runs": len(device_runs),
        "first_full_pass_at_run": first_pass,
        "counts": counts,
        "pytest_timeline": runs,
    }


def process_metrics(runs, baselines, usage_sum, code_changes=()):
    """Iterations, first full pass per test (and of the final code), Success@k, cost by outcome kind."""
    # Iteration = the agent checks ITS OWN code: runs of *_hal tests (the original run for comparison is not counted)
    def own(r):
        return [t for t in r["tests"] if t.endswith("_hal")]
    idx = {id(r): i for i, r in enumerate(runs)}
    iters = [dict(r, tests=own(r), _pos=idx[id(r)]) for r in runs if r["kind"] in ITER_KINDS and own(r)]
    per_test_iter = {}
    for i, r in enumerate(iters, 1):
        r["iteration"] = i
        for t in r["tests"]:
            per_test_iter[t] = per_test_iter.get(t, 0) + 1
        r["test_iteration"] = per_test_iter.get(r["tests"][0]) if len(r["tests"]) == 1 else None
    last_change = max(code_changes) if code_changes else 0

    def entry(r, orig):
        tok = r["cum_tokens"]
        return {"iteration": r["iteration"], "test_iteration": r["test_iteration"], "pytest_command": r["n"],
                "elapsed_min": r["elapsed_min"], "baseline_passed": orig, "passed": r["passed"],
                "tokens": tok, "tokens_input_total": tok["input_tokens"] + tok["cache_creation_input_tokens"]
                + tok["cache_read_input_tokens"]}
    first, final = {}, {}
    for r in iters:
        if len(r["tests"]) != 1 or r["kind"] != "pass" or (r["failed"] or 0) > 0 or r["selection"] != "full":
            continue
        t = r["tests"][0]
        b = baselines.get(t[:-4] if t.endswith("_hal") else t)
        orig = b.get("passed") if isinstance(b, dict) else b
        if orig is None or (r["passed"] or 0) < orig:
            continue
        if t not in first:
            first[t] = entry(r, orig)
        if t not in final and r["_pos"] >= last_change:
            final[t] = entry(r, orig)
    tests = sorted({t for r in iters for t in r["tests"]})
    success_at_k = {t: {k: bool(t in first and first[t]["test_iteration"] is not None and first[t]["test_iteration"] <= k)
                        for k in SUCCESS_K} for t in tests}
    # cost by outcome kind: the segment from the previous iteration to this one is attributed to this iteration's outcome
    by_kind = {}
    prev_t, prev_tok = 0.0, {k: 0 for k in usage_sum}
    for r in iters:
        b = by_kind.setdefault(r["kind"], {"iterations": 0, "minutes": 0.0, "tokens_input_total": 0})
        b["iterations"] += 1
        if r["elapsed_min"] is not None:
            b["minutes"] = round(b["minutes"] + r["elapsed_min"] - prev_t, 2); prev_t = r["elapsed_min"]
        tok = r["cum_tokens"]
        b["tokens_input_total"] += sum(tok[k] - prev_tok[k] for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        prev_tok = tok
    all_done = max((v["iteration"] for v in first.values()), default=None) if tests and all(t in first for t in tests) else None
    return {"iterations": len(iters), "first_full_pass": first, "final_code_first_full_pass": final,
            "code_changes": len(code_changes), "success_at_k": success_at_k,
            "all_tests_full_pass_at_iteration": all_done, "cost_by_outcome": by_kind,
            "baselines": {t: (b.get("passed") if isinstance(b, dict) else b) for t, b in baselines.items()}}


# --------------------------------------------------------------- output ---
def fmt(n):
    return f"{n:,}".replace(",", ".") if isinstance(n, int) else ("—" if n is None else str(n))

ROWS = [
    ("Duration (min)", lambda m: m["duration_min"]),
    ("  of which model time (min)", lambda m: m["duration_api_min"]),
    ("Number of steps (turns)", lambda m: m["num_turns"]),
    ("Input tokens (uncached)", lambda m: m["tokens"]["input_tokens"]),
    ("Input tokens written to cache", lambda m: m["tokens"]["cache_creation_input_tokens"]),
    ("Input tokens read from cache", lambda m: m["tokens"]["cache_read_input_tokens"]),
    ("Output tokens", lambda m: m["tokens"]["output_tokens"]),
    ("Cache share of input", lambda m: m["cache_hit_ratio"]),
    ("Cost (USD)", lambda m: m["cost_usd"]),
    ("Bash commands with pytest", lambda m: m["pytest_runs_total"]),
    ("Pytest calls in them", lambda m: m["pytest_invocations_total"]),
    ("  on the card", lambda m: m["device_test_runs"]),
    ("  compile only", lambda m: m["pytest_runs_by_kind"].get("compile_only", 0)),
    ("  compile error", lambda m: m["pytest_runs_by_kind"].get("compile_error", 0)),
    ("  wrong results", lambda m: m["pytest_runs_by_kind"].get("fail", 0)),
    ("  hang / timeout", lambda m: m["pytest_runs_by_kind"].get("hang", 0)),
    ("  pass", lambda m: m["pytest_runs_by_kind"].get("pass", 0)),
    ("  denied", lambda m: m["pytest_runs_by_kind"].get("denied", 0)),
    ("Iterations (agent test)", lambda m: m["process"]["iterations"]),
    ("First pass of any selection (command no.)", lambda m: m["first_full_pass_at_run"]),
    ("Card resets (tt-smi -r)", lambda m: m["counts"]["tt_smi_reset"]),
    ("File edits (Edit / Write)", lambda m: f'{m["counts"]["edits"]} / {m["counts"]["writes"]}'),
    ("File reads (Read)", lambda m: m["counts"]["reads"]),
    ("Helper agents", lambda m: m["counts"]["subagents"]),
    ("Confluence / MCP calls", lambda m: m["counts"].get("mcp_calls", 0)),
    ("Denied tool calls", lambda m: m["counts"]["denied"]),
]

def to_md(names, metrics):
    out = [f"_metrics.py version {VERSION}_", "", "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, f in ROWS:
        out.append(f"| {label} | " + " | ".join(fmt(f(m)) for m in metrics) + " |")
    if len(metrics) == 1:
        m = metrics[0]
        out += ["", "### Flow of pytest invocations", "",
                "| # | Call | Outcome | Passed | Failed | Selection | Command |", "|---|---|---|---|---|---|---|"]
        for r in m["pytest_timeline"]:
            cmd = r["cmd"].replace("|", "\\|")
            out.append(f'| {r["n"]} | {r["seg"]} | {r["kind"]} | {fmt(r["passed"])} | {fmt(r["failed"])} | '
                       f'{r["selection"]} | `{cmd}` |')
        out += ["", "_The outcome classification is heuristic (based on the pytest output) and is meant as a guide._"]
        pr = m.get("process") or {}
        out += ["", "### First full pass per test", "",
                "| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens | Original passed |",
                "|---|---|---|---|---|---|"]
        for t, v in sorted((pr.get("first_full_pass") or {}).items()):
            out.append(f'| {t} | {v["iteration"]} / {fmt(v["test_iteration"])} | {v["pytest_command"]} | {fmt(v["elapsed_min"])} | '
                       f'{fmt(v["tokens_input_total"])} | {fmt(v["baseline_passed"])} |')
        out += ["", "### First full pass of the final code (after the last change of the agent's code)", "",
                "| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens |", "|---|---|---|---|---|"]
        for t, v in sorted((pr.get("final_code_first_full_pass") or {}).items()):
            out.append(f'| {t} | {v["iteration"]} / {fmt(v["test_iteration"])} | {v["pytest_command"]} | {fmt(v["elapsed_min"])} | '
                       f'{fmt(v["tokens_input_total"])} |')
        out += ["", f'Iterations total: {fmt(pr.get("iterations"))}; all tests fully passed at iteration: '
                f'{fmt(pr.get("all_tests_full_pass_at_iteration"))}; changes of the agent\'s code: {fmt(pr.get("code_changes"))}; '
                f'ambiguous commands (the number of pytest calls and final lines does not match): {fmt(pr.get("ambiguous_commands"))}', "",
                "| Test | " + " | ".join(f"Success@{k}" for k in SUCCESS_K) + " |", "|---|" + "---|" * len(SUCCESS_K)]
        for t, d in sorted((pr.get("success_at_k") or {}).items()):
            out.append(f"| {t} | " + " | ".join("yes" if d[k] else "no" for k in SUCCESS_K) + " |")
        out += ["", "### Cost by iteration outcome kind", "", "| Outcome | Iterations | Minutes | Input tokens |", "|---|---|---|---|"]
        for k, v in sorted((pr.get("cost_by_outcome") or {}).items()):
            out.append(f'| {k} | {v["iterations"]} | {v["minutes"]} | {fmt(v["tokens_input_total"])} |')
        out += ["", "### Reading of material", "", "| Category | Calls | Distinct | Re-reads | Share of all tool calls |", "|---|---|---|---|---|"]
        for k, v in (m.get("reference_reads") or {}).items():
            out.append(f'| {k} | {v["calls"]} | {v["distinct"]} | {v["rereads"]} | {fmt(v["share_of_tool_calls"])} |')
        out += ["", "_Iteration = a run of the agent's (`*_hal`) test, including a compile failure. Full pass = "
                "the same selection of variants as the original (same -k, the whole test file, or the same number of selected and deselected variants), "
                "pass count >= original, 0 failed. Input tokens up to the first pass are the sum of `usage` from the transcript messages up to "
                "that moment; output tokens cannot be counted reliably per moment from the transcript, so they are given only "
                "in total (from the `result` event)._"]
    return "\n".join(out)


def kv_args(args, flag):
    out = {}
    while flag in args:
        i = args.index(flag)
        k, _, v = args[i + 1].partition("=")
        out[k] = v
        del args[i:i + 2]
    return out


def main():
    args = sys.argv[1:]
    out_json = out_md = None
    if "--json" in args:
        i = args.index("--json"); out_json = args[i + 1]; del args[i:i + 2]
    if "--md" in args:
        i = args.index("--md"); out_md = args[i + 1]; del args[i:i + 2]
    cli_base = {k: {"passed": int(v)} for k, v in kv_args(args, "--baseline").items()}
    cli_k = kv_args(args, "--k-filter")
    if not args:
        print(__doc__); sys.exit(1)
    names, metrics = [], []
    for a in args:
        names.append(Path(a).name.replace(".tar.gz", ""))
        base = load_baselines(a); base.update(cli_base)
        kf = load_k_filters(a); kf.update(cli_k)
        metrics.append(analyze(load_lines(a), base, kf))
    md = to_md(names, metrics)
    print(md)
    if out_json:
        Path(out_json).write_text(json.dumps(metrics[0] if len(metrics) == 1 else dict(zip(names, metrics)), indent=2))
    if out_md:
        Path(out_md).write_text(md + "\n")


if __name__ == "__main__":
    main()
