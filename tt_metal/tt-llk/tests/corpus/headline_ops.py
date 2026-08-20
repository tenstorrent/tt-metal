#!/usr/bin/env python3
"""headline_ops.py — derive the headline sweep's default --ops list.

Default ops = HEADLINE_ROWS (passed by the wrapper from sweep_2x2.conf)
UNION every sweep row whose fresh body / golden / mapped test changed since
the LAST PIN, derived from git:

  * "last pin" = the second-newest commit that changed
    _REVIEWED_CC1PLUS_SHA256 in sweep_2x2.conf (the newest is the current
    pin cut, so <prev-pin>..HEAD is exactly the content that entered the
    current pin, plus anything landed since);
  * changed rows come from `git diff --name-only <prev-pin>..HEAD` over
    tt-llk, resolved to ops-TSV rows by
      - test/perf file basenames appearing in a row's node-id columns,
      - function-level diff of tests/helpers/include/fresh_cpp_operations.h
        (calculate_<tok>_fresh_cpp spans) and
        tests/python_tests/helpers/golden_generators.py (top-level defs),
        matched into rows by separator-bounded token containment,
      - ckernel_sfpu_*.h basename stems matched against corpus_id.

Prints the comma list on stdout; every derivation decision on stderr.
Exit 0 with at least the headline rows even when git derivation fails
(SPEED-FIRST: a headline run must never be blocked by ancestry archaeology) —
the failure is loud on stderr.

Owned by the wrapper surface (lane DD).  sweep_2x2.py is NOT touched: this
is a pure producer of its --ops argument.
"""

import argparse
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
CONF_RELPATH = "tt_metal/tt-llk/tests/corpus/sweep_2x2.conf"
FRESH_RELPATH = "tt_metal/tt-llk/tests/helpers/include/fresh_cpp_operations.h"
GOLDEN_RELPATH = "tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py"

NODE_COLS = ("sem_corr", "sem_perf", "hand_corr", "hand_perf")


def log(msg):
    print(f"headline_ops: {msg}", file=sys.stderr)


def run_git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def load_rows(ops_tsv):
    rows = []
    header = None
    for line in ops_tsv.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        d = dict(zip(header, parts))
        if d.get("op"):
            rows.append(d)
    return rows


def norm(s):
    return s.lower().replace("-", "_")


def token_in(token, hay):
    """Separator-bounded containment: 'abs' matches '_abs[' but not
    '_absint32_' (tokens may themselves contain '_', e.g. mul_int)."""
    return (
        re.search(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9_])", hay)
        is not None
        or re.search(r"(?<![a-z0-9])" + re.escape(token) + r"(?=[_\[.(])", hay)
        is not None
    )


def squash(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def rows_matching_token(rows, token):
    token = norm(token).strip("_")
    out = []
    for r in rows:
        hay = " ".join(
            [norm(r["op"]), norm(r.get("corpus_id", ""))]
            + [norm(r.get(c, "")) for c in NODE_COLS]
        )
        cid_tail = r.get("corpus_id", "").split("__", 1)[-1]
        if (
            norm(r["op"]) == token
            or token_in(token, hay)
            # squashed-name fallback: '_cast_fp32_to_fp16a' == 'castfp32tofp16a'
            or squash(token) == squash(r["op"])
            or squash(token) == squash(cid_tail)
        ):
            out.append(r["op"])
    return out


def changed_function_tokens(repo, prev, relpath, def_re, token_of):
    """Function-level diff: which def_re-matched spans in HEAD's version of
    relpath overlap a changed hunk of prev..HEAD?  Returns tokens."""
    try:
        diff = run_git(repo, "diff", "-U0", f"{prev}..HEAD", "--", relpath)
    except subprocess.CalledProcessError as e:
        log(f"WARN: git diff failed for {relpath}: {e.stderr.strip()}")
        return set()
    if not diff:
        return set()
    changed_lines = set()
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.M):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        if count == 0:
            # pure deletion: attribute to the surrounding position
            changed_lines.update((max(start, 1), start + 1))
        else:
            changed_lines.update(range(start, start + count))
    try:
        text = run_git(repo, "show", f"HEAD:{relpath}")
    except subprocess.CalledProcessError:
        log(f"WARN: {relpath} missing at HEAD; skipping function attribution")
        return set()
    lines = text.splitlines()
    spans = []  # (start_line, token) 1-based
    for i, line in enumerate(lines, 1):
        m = def_re.search(line)
        if m:
            spans.append((i, token_of(m)))
    tokens = set()
    for idx, (start, tok) in enumerate(spans):
        end = spans[idx + 1][0] - 1 if idx + 1 < len(spans) else len(lines)
        if any(start <= ln <= end for ln in changed_lines):
            tokens.add(tok)
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline", default="", help="conf HEADLINE_ROWS (comma list)")
    ap.add_argument("--ops-tsv", type=pathlib.Path, default=HERE / "sweep_2x2_ops.tsv")
    ap.add_argument(
        "--repo",
        type=pathlib.Path,
        default=None,
        help="tt-metal checkout (default: this file's repo)",
    )
    ap.add_argument(
        "--since", default=None, help="override the derived previous-pin rev"
    )
    args = ap.parse_args()

    rows = load_rows(args.ops_tsv)
    known = {r["op"] for r in rows}
    ops = []  # ordered, deduped

    def add(op, why):
        if op not in known:
            log(f"DROP {op}: not an ops-TSV row ({why})")
            return
        if op not in ops:
            ops.append(op)
            log(f"ADD  {op}: {why}")

    for h in [x for x in args.headline.split(",") if x]:
        add(h, "HEADLINE_ROWS")

    repo = args.repo or HERE
    try:
        repo_root = pathlib.Path(run_git(repo, "rev-parse", "--show-toplevel").strip())
        prev = args.since
        if not prev:
            pins = run_git(
                repo_root,
                "log",
                "--format=%H",
                "-G",
                "^_REVIEWED_CC1PLUS_SHA256=",
                "--",
                CONF_RELPATH,
            ).split()
            if len(pins) < 2:
                raise RuntimeError(f"found {len(pins)} pin-change commits; need 2")
            prev = pins[1]
            log(f"previous pin commit: {prev[:12]} (current: {pins[0][:12]})")

        changed = run_git(
            repo_root,
            "diff",
            "--name-only",
            f"{prev}..HEAD",
            "--",
            "tt_metal/tt-llk",
        ).split()
        log(f"{len(changed)} tt-llk paths changed since {prev[:12]}")

        # (a) mapped test/perf files, FUNCTION level: the shared suites
        # (perf_eltwise_unary_sfpu.py, test_sfpu_binary.py, ...) host a
        # hundred-plus rows each, so file-level attribution would degenerate
        # into the full surface.  Attribute only rows whose node ids
        # reference a test function whose span actually changed.
        for path in changed:
            base = pathlib.Path(path).name
            if not (base.startswith(("test_", "perf_")) and base.endswith(".py")):
                continue
            toks = changed_function_tokens(
                repo_root,
                prev,
                path,
                re.compile(r"^def (test_[a-z0-9_]+)\("),
                lambda m: m.group(1),
            )
            if not toks:
                continue
            try:
                diff_txt = run_git(
                    repo_root, "diff", "-U0", f"{prev}..HEAD", "--", path
                )
            except subprocess.CalledProcessError:
                diff_txt = ""
            changed_txt = "\n".join(
                ln
                for ln in diff_txt.splitlines()
                if ln[:1] in "+-" and ln[:2] not in ("++", "--")
            ).lower()
            for tok in sorted(toks):
                hosted = [
                    r
                    for r in rows
                    if any(
                        f"::{tok}[" in r.get(c, "") or r.get(c, "").endswith(f"::{tok}")
                        for c in NODE_COLS
                    )
                ]
                # Mega-parametrized shared functions (the causal-lift and
                # fitted families host dozens of rows): a one-op parametrize
                # addition must not fan out to every hosted row.  Tier 2:
                # keep only rows whose mathop token appears in the changed
                # lines; fall back to the full hosted set when the change is
                # row-agnostic (helper/loop refactor touching them all).
                if len(hosted) > 8:
                    specific = []
                    for r in hosted:
                        mtoks = set()
                        for c in NODE_COLS:
                            mtoks.update(
                                m.lower()
                                for m in re.findall(
                                    r"mathop:([A-Za-z0-9_]+)", r.get(c, "")
                                )
                            )
                        if any(
                            re.search(
                                r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])",
                                changed_txt,
                            )
                            for t in mtoks
                        ):
                            specific.append(r)
                    if specific:
                        for r in specific:
                            add(r["op"], f"row-specific change in {base}::{tok}")
                        continue
                    log(
                        f"NOTE {base}::{tok}: row-agnostic change — "
                        f"fanning out to all {len(hosted)} hosted rows"
                    )
                for r in hosted:
                    add(r["op"], f"test function changed: {base}::{tok}")

        # (b) fresh bodies: calculate_<tok>_fresh_cpp function-level diff
        if FRESH_RELPATH in changed:
            toks = changed_function_tokens(
                repo_root,
                prev,
                FRESH_RELPATH,
                re.compile(r"\bcalculate_([a-z0-9_]+?)_fresh_cpp\b"),
                lambda m: m.group(1),
            )
            for tok in sorted(toks):
                hits = rows_matching_token(rows, tok)
                if not hits:
                    log(f"NOTE fresh body '{tok}' changed but matched no row")
                for op in hits:
                    add(op, f"fresh body changed: calculate_{tok}_fresh_cpp")

        # (c) goldens: top-level defs of golden_generators.py
        if GOLDEN_RELPATH in changed:
            toks = changed_function_tokens(
                repo_root,
                prev,
                GOLDEN_RELPATH,
                re.compile(r"^(?:    )?def ([a-z0-9_]+)\("),
                lambda m: m.group(1),
            )
            for tok in sorted(toks):
                hits = rows_matching_token(rows, tok)
                if not hits:
                    log(f"NOTE golden '{tok}' changed but matched no row")
                for op in hits:
                    add(op, f"golden changed: {tok}")

        # (d) hand LLK kernels: ckernel_sfpu_* stems vs corpus_id
        for path in changed:
            base = pathlib.Path(path).name
            m = re.match(r"(ckernel_sfpu_[a-z0-9_]+)\.(h|hpp)$", base)
            if not m:
                continue
            stem = m.group(1)
            for r in rows:
                cid = r.get("corpus_id", "")
                if cid.split("__", 1)[-1] == stem:
                    add(r["op"], f"hand kernel changed: {base}")
    except Exception as e:  # noqa: BLE001 — headline must not be blocked
        log(f"WARN: git derivation FAILED ({e}); falling back to HEADLINE_ROWS only")

    if not ops:
        log("FATAL: empty ops list (no HEADLINE_ROWS and derivation empty)")
        return 1
    print(",".join(ops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
