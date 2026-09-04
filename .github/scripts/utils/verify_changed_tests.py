#!/usr/bin/env python3
"""
All logic for .github/workflows/verify-changed-tests.yaml, in a single pass.

The gate proves that the test entries a PR edits in tests/pipeline_reorg/ still
run green, and nothing more. One invocation does three things:

  1. Diffs the changed yamls at test-entry level, working out which entries were
     added or behaviourally changed, which legs (entry x SKU) must run, which are
     blocked behind an owner review, and which build flavours those legs need.

  2. Runs prepare_test_matrix.py over the changed yamls and narrows its output to
     exactly those legs, so the dispatched matrix is the touched work and nothing
     else.

  3. Requires an approving review on the current head for any blocked leg, and
     exits non-zero without one. Who may approve is already decided by
     CODEOWNERS, which the main ruleset enforces via require_code_owner_review.

With --event merge_group only step 1 runs: those legs already ran on the PR head,
and the queue only needs the scope to resolve.

Design notes
------------
Entries are keyed on (name, arch, gtest_shard_index). `name` alone is not
unique: llk_merge_gate_tests.yaml has four "LLK FD wormhole" entries differing
only by shard, and vllm_model_tests.yaml has entries differing only by arch.

Edits that cannot change how a test executes do not need hardware:

  * owner_id / team    Ownership metadata.
  * timeout            The ceiling is already enforced statically by
                       verify_time_budget.py inside every pipeline's own
                       load-test-matrix step, so a hardware run here proves
                       nothing a static check has not already proven.

Anything else -- cmd, adding or removing a SKU, arch, dispatch_mode, tier, shard
layout, model fields -- is treated as behaviour-affecting.

Time budgets are deliberately not checked here. The budget key for a yaml is not
derivable from it (the filename convention breaks on nine files, and tiered keys
come from per-SKU `tier` fields), and this gate is not the place to house that
mapping. Every pipeline already verifies its own budgets when it loads its matrix.

prepare_test_matrix.py is untouched: it is called normally with ALL_SKUS_IN_TESTS
and its output is narrowed here. Because each entry expands to one row per SKU,
filtering rows to the touched legs *is* the SKU narrowing -- there is no separate
sku-allowlist step.

Fails closed throughout. A leg that cannot be resolved to a matrix row, or a
matrix shorter than expected, fails the gate rather than passing with reduced
coverage -- a short matrix would otherwise look exactly like a pass.
"""

import argparse
import copy
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

TESTS_DIR = "tests/pipeline_reorg"

# Top-level entry fields that cannot change how a test executes.
METADATA_FIELDS = {"owner_id", "team"}

# Per-SKU sub-keys that cannot change how a test executes.
METADATA_SKU_FIELDS = {"timeout"}

# A pytest per-test timeout, which is stripped from simulator legs: ttsim is
# 10-50x slower than hardware, so a limit sized for silicon kills slow-but-correct
# tests. The job's own timeout-minutes stays as the backstop. ttnn-sanity-tests-impl
# does the same with `sed -E 's/[[:space:]]+--timeout[[:space:]]+[0-9]+//g'`.
PYTEST_TIMEOUT_FLAG = re.compile(r"\s+--timeout\s+\d+")

# Paths only the installed tt-metalium debs provide; a cmd touching one needs the
# packages artifact rather than a build tree.
PACKAGE_INSTALL_PREFIXES = ("/usr/share/tt-metalium", "/usr/libexec/tt-metalium")

DEFAULT_CODEOWNERS = ".github/CODEOWNERS"
DEFAULT_TTSIM_SKIP_LIST = "tests/pipeline_reorg/ttsim-skip-list.yaml"
DEFAULT_SKU_CONFIG = ".github/sku_config.yaml"
DEFAULT_PREPARE_SCRIPT = ".github/scripts/utils/prepare_test_matrix.py"


class GateError(Exception):
    """Anything the gate cannot resolve. Always fails closed."""


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def split_list(raw):
    return [token.strip() for token in re.split(r"[,\s]+", raw or "") if token.strip()]


def write_github_output(pairs):
    """Append key/value pairs to $GITHUB_OUTPUT, heredoc-quoting multi-line values."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        for key, value in pairs:
            if isinstance(value, str) and "\n" in value:
                f.write(f"{key}<<EOF\n{value}\nEOF\n")
            else:
                f.write(f"{key}={value}\n")


def git_show(ref, path):
    """Return file contents at ref, or None when the file does not exist there."""
    result = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    return None if result.returncode != 0 else result.stdout


def merge_base(base):
    """
    Resolve base to its merge base with HEAD.

    Diffing against the merge base rather than the base tip keeps commits that
    landed on main after the branch point out of scope, and makes the working
    tree visible -- which is what makes this runnable locally before pushing.
    """
    result = subprocess.run(["git", "merge-base", base, "HEAD"], capture_output=True, text=True)
    if result.returncode != 0:
        return base
    return result.stdout.strip() or base


def changed_files(base):
    """
    pipeline_reorg yamls that differ between base and the working tree.

    Untracked files count too: a brand new pipeline yaml is exactly the case that
    must not slip past the gate, and it is still untracked when this is run
    locally before committing.
    """
    diff = subprocess.run(["git", "diff", "--name-only", base, "--", TESTS_DIR], capture_output=True, text=True)
    if diff.returncode != 0:
        raise GateError(f"git diff against {base} failed: {diff.stderr.strip()}")

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", TESTS_DIR],
        capture_output=True,
        text=True,
    )
    if untracked.returncode != 0:
        raise GateError(f"git ls-files failed: {untracked.stderr.strip()}")

    names = set(diff.stdout.splitlines()) | set(untracked.stdout.splitlines())
    return sorted(name for name in names if name.strip().endswith((".yaml", ".yml")))


def load_review_only_skus(raw, sku_config_path):
    """
    Parse the review-only SKU list.

    The list lives in verify-changed-tests.yaml so the gate's hardware policy sits
    next to the gate itself. Names are validated against sku_config.yaml so a typo
    fails the gate instead of silently letting a galaxy leg auto-run.
    """
    names = set(split_list(raw))
    if not names:
        return names

    with open(sku_config_path, "r") as f:
        known = set((yaml.safe_load(f) or {}).get("skus") or {})
    unknown = sorted(names - known)
    if unknown:
        raise GateError(
            f"review-only SKUs not present in {sku_config_path}: {', '.join(unknown)}. "
            "Fix REVIEW_ONLY_SKUS in .github/workflows/verify-changed-tests.yaml."
        )
    return names


def load_ttsim_skips(path):
    """
    Load tests/pipeline_reorg/ttsim-skip-list.yaml as {arch: [test ids]}.

    These are the tests that do not yet work under the simulator. The owning
    pipeline passes them as pytest --deselect args on every sim_* leg (see
    ttnn-sanity-tests-impl.yaml's "Load TTSim skip list" step) so the simulator
    runs the same cmd as hardware minus the known-broken tests. Without them a
    sim leg fails on a test its own pipeline knowingly skips.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        skips = yaml.safe_load(f) or {}
    if not isinstance(skips, dict):
        raise GateError(f"{path} is not a mapping of arch -> test ids")
    return {arch: list(ids or []) for arch, ids in skips.items()}


def ttsim_skip_arch(sku):
    """Which skip-list arch a sim SKU belongs to, or None when it has no list."""
    if sku.startswith("sim_wh"):
        return "wormhole_b0"
    if sku.startswith("sim_bh"):
        return "blackhole"
    return None


def deselect_args(sku, skips):
    """The pytest --deselect args a sim leg should carry, as one shell-ready string."""
    arch = ttsim_skip_arch(str(sku))
    if arch is None:
        return ""
    return " ".join("--deselect=" + test_id for test_id in skips.get(arch) or [])


def load_codeowners(path):
    """
    Parse CODEOWNERS into an ordered list of (pattern, owners).

    Order is preserved because GitHub gives precedence to the *last* matching
    pattern -- which is what makes `tests/pipeline_reorg/` a fallback beneath the
    more specific per-file lines that follow it.
    """
    rules = []
    if not os.path.exists(path):
        return rules
    with open(path, "r") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            rules.append((parts[0], parts[1:]))
    return rules


def codeowners_match(pattern, path):
    """Match one CODEOWNERS pattern against a repo-relative path."""
    pattern = pattern.lstrip("/")
    if pattern.endswith("/"):
        return path.startswith(pattern)
    if pattern.startswith("**/"):
        tail = pattern[3:]
        return fnmatch.fnmatch(path, tail) or fnmatch.fnmatch(path, f"*/{tail}")
    if "/" in pattern:
        return fnmatch.fnmatch(path, pattern)
    return fnmatch.fnmatch(os.path.basename(path), pattern)


def owners_for(path, rules):
    """
    Owners of a path, honouring CODEOWNERS last-match-wins precedence.

    Returns (individual logins, team slugs). Teams cannot be expanded with the
    workflow token, so a path owned only by teams falls back to accepting any
    approving review and letting the ruleset's own code-owner requirement decide.
    """
    owners = []
    for pattern, entries in rules:
        if codeowners_match(pattern, path):
            owners = entries
    individuals = {o.lstrip("@") for o in owners if o.startswith("@") and "/" not in o}
    teams = {o.lstrip("@") for o in owners if o.startswith("@") and "/" in o}
    return individuals, teams


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------


def parse_entries(text):
    """
    Return the test entries in a tests yaml, or None when it is not a list at all.

    A missing, empty or comment-only file is an empty matrix, matching
    prepare_test_matrix.py's placeholder handling. Any other non-list shape is
    returned as None for the caller to judge: it is only legitimate for a file
    declared as holding no test matrix, and is an error anywhere else.
    """
    if text is None:
        return []
    entries = yaml.safe_load(text)
    if entries is None:
        return []
    if not isinstance(entries, list):
        return None
    if any(not isinstance(entry, dict) for entry in entries):
        raise GateError("test matrix entries must be mappings")
    return entries


def entry_key(entry):
    return (
        str(entry.get("name", "")),
        str(entry.get("arch", "")),
        str(entry.get("gtest_shard_index", "")),
    )


def key_str(key):
    name, arch, shard = key
    parts = [name]
    if arch:
        parts.append(f"arch={arch}")
    if shard:
        parts.append(f"shard={shard}")
    return " | ".join(parts)


def index_entries(entries, path):
    """Map composite key -> entry, rejecting collisions rather than guessing."""
    index = {}
    for entry in entries:
        key = entry_key(entry)
        if key in index:
            raise GateError(
                f"{path}: two entries share the key '{key_str(key)}'. "
                "The gate cannot tell which one an edit touched -- give them "
                "distinguishing name/arch/gtest_shard_index values."
            )
        index[key] = entry
    return index


def behavioural_view(entry):
    """The parts of an entry that can change how a test executes."""
    view = {k: copy.deepcopy(v) for k, v in entry.items() if k not in METADATA_FIELDS}
    skus = view.get("skus")
    if isinstance(skus, dict):
        view["skus"] = {
            name: (
                {k: v for k, v in config.items() if k not in METADATA_SKU_FIELDS}
                if isinstance(config, dict)
                else config
            )
            for name, config in skus.items()
        }
    return view


def resolve_profile(path, tracy_files):
    """
    Which build flavour a leg needs.

    Every build carries the wheel. Whether a test needs it cannot be read off the
    command: ops_integration_tests.yaml runs `python3 scripts/...` and its pipeline
    passes a wheel, runtime_unit_tests.yaml runs `python3 tests/scripts/...` and its
    pipeline does not, and shell wrappers like run_ttnn_examples.sh hide python
    altogether. Since a wheel that goes unused costs only build time while a missing
    one fails the leg outright, the gate always builds one -- a strict superset of
    what every owning pipeline provides.

    That leaves tracy as the only real distinction, and it has to be told: nothing in
    a test entry says "this needs a profiler build", and matching cmds on "profiler"
    also hits runtime_unit_tests.yaml and fabric_perf_tests.yaml.
    """
    if os.path.basename(path) in tracy_files:
        return "profiler"
    return "default"


def needs_packages(entry):
    """
    Whether a leg runs against the installed tt-metalium debs rather than a build tree.

    A handful of entries invoke things the packages put on the system -- the examples
    under /usr/share/tt-metalium, for instance -- which no build artifact provides.
    ttsim-sanity-tests-impl.yaml serves those by installing the packages artifact
    instead of calling setup-job, and the gate mirrors that for exactly those legs.
    """
    cmd = entry.get("cmd") or ""
    return any(prefix in cmd for prefix in PACKAGE_INSTALL_PREFIXES)


def scope_file(path, base, review_only, tracy_files, non_matrix_files, unsupported_files):
    """Diff one tests yaml and return its scoping result."""
    if os.path.basename(path) in non_matrix_files:
        # Declared as holding no test matrix (e.g. ttsim-skip-list.yaml, a per-arch
        # mapping of test ids). Nothing here for the gate to prove.
        return {"no_entries": True, "run_legs": [], "review_legs": [], "metadata_only": []}

    unsupported = os.path.basename(path) in unsupported_files

    old_entries = parse_entries(git_show(base, path))
    new_entries = parse_entries(open(path).read() if os.path.exists(path) else None)

    # Reaching here with a non-list means a real test matrix was reshaped into
    # something else. prepare_test_matrix.py errors on that shape, and so does the
    # gate -- silently marking it "no tests" would pass the PR with zero coverage.
    for entries, label in ((new_entries, path), (old_entries, f"{path}@{base}")):
        if entries is None:
            raise GateError(
                f"{label} is not a list of test entries. If it legitimately holds no test "
                "matrix, add its filename to NON_MATRIX_YAMLS in "
                ".github/workflows/verify-changed-tests.yaml; otherwise fix the file."
            )

    old_index = index_entries(old_entries, f"{path}@{base}")
    new_index = index_entries(new_entries, path)

    touched, metadata_only = [], []
    for key, entry in new_index.items():
        previous = old_index.get(key)
        if previous is None:
            touched.append((key, entry, "added"))
        elif canonical(behavioural_view(previous)) != canonical(behavioural_view(entry)):
            touched.append((key, entry, "changed"))
        elif canonical(previous) != canonical(entry):
            metadata_only.append(key_str(key))
    # Removed entries are intentionally ignored: there is no test left to prove.

    run_legs, review_legs = [], []
    for key, entry, reason in touched:
        skus = entry.get("skus")
        if not isinstance(skus, dict) or not skus:
            raise GateError(f"{path}: entry '{key_str(key)}' has no skus mapping; cannot resolve any leg to run")
        team = entry.get("team")
        for sku in skus:
            leg = {
                "file": path,
                "name": entry.get("name"),
                "arch": entry.get("arch"),
                "gtest_shard_index": entry.get("gtest_shard_index"),
                "sku": sku,
                "team": team,
                "reason": reason,
                "profile": resolve_profile(path, tracy_files),
                "packages": needs_packages(entry),
            }
            if unsupported:
                # The gate cannot build a runnable matrix for this pipeline, so the
                # edit goes to its code owners instead of passing unverified.
                leg["blocked_by"] = "unsupported_yaml"
                review_legs.append(leg)
            elif sku in review_only:
                leg["blocked_by"] = "review_only_sku"
                review_legs.append(leg)
            else:
                run_legs.append(leg)

    return {
        "no_entries": False,
        "run_legs": run_legs,
        "review_legs": review_legs,
        "metadata_only": metadata_only,
    }


def build_scope(base, files, review_only, tracy_files, non_matrix_files, unsupported_files):
    run_legs, review_legs, metadata_only, skipped = [], [], [], []

    for path in sorted(files):
        scoped = scope_file(path, base, review_only, tracy_files, non_matrix_files, unsupported_files)
        if scoped["no_entries"]:
            skipped.append(path)
            continue
        run_legs.extend(scoped["run_legs"])
        review_legs.extend(scoped["review_legs"])
        metadata_only.extend(f"{path}: {name}" for name in scoped["metadata_only"])

    if review_legs:
        status = "blocked"
    elif run_legs:
        status = "run"
    else:
        status = "no_op"

    # Identifies exactly this set of legs, so the merge_group run can confirm it is
    # looking at the same work the pull_request run proved.
    digest = hashlib.sha256(canonical(sorted(canonical(leg) for leg in run_legs + review_legs)).encode()).hexdigest()

    # Only the build flavours the surviving legs need, so a PR touching one cpp
    # pipeline does not pay for a wheel or a tracy build.
    profiles = sorted({leg["profile"] for leg in run_legs})

    return {
        "status": status,
        "changed_files": sorted(files),
        "skipped_files": skipped,
        "run_legs": run_legs,
        "review_legs": review_legs,
        "expected_leg_count": len(run_legs),
        "metadata_only": metadata_only,
        "profiles": profiles,
        "leg_digest": digest,
    }


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def row_key(row):
    """
    Key a matrix row the same way an entry in the source yaml is keyed.

    build_test_matrix() always appends " [<concrete sku>]" to the name, and sets
    logical_sku when the concrete SKU differs from the one the yaml names. Both
    are undone here so rows line up with the legs scope produced.
    """
    sku = str(row.get("sku", ""))
    name = str(row.get("name", ""))
    suffix = f" [{sku}]"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return (
        name,
        str(row.get("arch", "")),
        str(row.get("gtest_shard_index", "")),
        str(row.get("logical_sku") or sku),
    )


def leg_row_key(leg):
    shard = leg.get("gtest_shard_index")
    return (
        str(leg.get("name") or ""),
        str(leg.get("arch") or ""),
        "" if shard is None else str(shard),
        str(leg.get("sku") or ""),
    )


def describe_row(key):
    name, arch, shard, sku = key
    parts = [name, f"sku={sku}"]
    if arch:
        parts.append(f"arch={arch}")
    if shard:
        parts.append(f"shard={shard}")
    return " | ".join(parts)


def parse_matrix_output(text, path):
    """Pull the heredoc-quoted `matrix` value back out of a GITHUB_OUTPUT file."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line != "matrix<<EOF":
            continue
        body = []
        for value in lines[index + 1 :]:
            if value == "EOF":
                return json.loads("\n".join(body))
            body.append(value)
        raise GateError(f"unterminated matrix output for {path}")
    raise GateError(f"prepare_test_matrix.py emitted no matrix for {path}")


def build_matrices(scope, prepare_script, sku_config, work_dir):
    """
    Run prepare_test_matrix.py over each changed yaml and collect its matrix.

    Invoked exactly as the pipelines invoke it -- ALL_SKUS_IN_TESTS, no
    sku-allowlist -- so the rows carry the same runs_on, cmd, timeout and
    weights-cache fields the real pipeline would use. GITHUB_OUTPUT is redirected
    to a scratch file so the matrix can be read back without writing to the
    caller's step outputs.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    skipped = set(scope.get("skipped_files") or [])

    matrices = {}
    for path in scope["changed_files"]:
        if path in skipped:
            continue
        stem = Path(path).stem
        capture = work_dir / f"{stem}.github-output"
        capture.write_text("")

        result = subprocess.run(
            [sys.executable, prepare_script, path, "ALL_SKUS_IN_TESTS", sku_config],
            capture_output=True,
            text=True,
            env=dict(os.environ, GITHUB_OUTPUT=str(capture)),
        )
        if result.returncode != 0:
            raise GateError(f"prepare_test_matrix.py failed for {path}:\n{result.stdout}\n{result.stderr}")

        matrices[stem] = parse_matrix_output(capture.read_text(), path)

    if not matrices:
        raise GateError("no matrices were produced for the changed files")
    return matrices


def load_matrices(matrix_dir):
    """Map source yaml stem -> matrix rows. Used by the tests and for local debugging."""
    matrices = {}
    for path in sorted(Path(matrix_dir).glob("*.json")):
        with open(path, "r") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise GateError(f"{path} does not contain a matrix array")
        matrices[path.stem] = rows
    if not matrices:
        raise GateError(f"no matrices found in {matrix_dir}")
    return matrices


def filter_matrix(scope, matrices, skips):
    """
    Narrow each changed yaml's matrix to the legs scoped from that same yaml.

    The index is keyed by source yaml as well as by leg, because entries are
    mirrored between pipelines: "t3k fabric tests" appears in both
    fabric_sanity_tests.yaml and fabric_tests.yaml, and "fabric fast unit tests"
    in both blackhole_e2e_tests.yaml and fabric_sanity_tests.yaml. A single global
    index would read those as a collision and fail a legitimate edit, and could
    match a leg against the other pipeline's row. Two mirrored definitions are two
    entries, so touching both runs both.
    """
    index = {}
    for source, rows in matrices.items():
        for row in rows:
            key = (source, row_key(row))
            if key in index:
                raise GateError(
                    f"{source}: two matrix rows share the key '{describe_row(row_key(row))}'. "
                    "Mirrored definitions in different yamls are fine, but one yaml cannot "
                    "hold the same entry twice."
                )
            enriched = dict(row)
            enriched["source_yaml"] = source
            index[key] = enriched

    selected, missing = [], []
    for leg in scope["run_legs"]:
        source = Path(leg["file"]).stem
        row = index.get((source, leg_row_key(leg)))
        if row is None:
            missing.append(f"{describe_row(leg_row_key(leg))} [{source}]")
        else:
            row = dict(row)
            # Carry the build flavour onto the row so the runner can pick the matching
            # build artifact without re-deriving it.
            row["gate_profile"] = leg["profile"]
            row["gate_packages"] = leg["packages"]
            if str(row.get("sku", "")).startswith("sim_"):
                stripped = PYTEST_TIMEOUT_FLAG.sub("", row.get("cmd") or "")
                if stripped != row.get("cmd"):
                    row["gate_stripped_timeout"] = True
                row["cmd"] = stripped
                row["gate_pytest_deselect"] = deselect_args(row.get("sku", ""), skips)
            selected.append(row)

    if missing:
        raise GateError(
            "these legs did not resolve to a matrix row: "
            + "; ".join(missing)
            + ". The gate cannot prove a test it cannot dispatch."
        )

    expected = scope["expected_leg_count"]
    if len(selected) != expected:
        raise GateError(f"expected {expected} legs, resolved {len(selected)}")

    multihost = [row for row in selected if row.get("multihost")]
    if multihost:
        raise GateError(
            "these legs need multi-host runners, which the gate must not dispatch: "
            + "; ".join(describe_row(row_key(row)) for row in multihost)
            + ". Add their SKUs to REVIEW_ONLY_SKUS in verify-changed-tests.yaml."
        )

    # One matrix, not two: the workflow runs every leg from a single job and picks
    # the simulator or hardware path per leg off the sim_ prefix. Splitting here
    # would leave one half with no job to run in.
    sim = [row for row in selected if str(row.get("sku", "")).startswith("sim_")]

    unfetchable = [row for row in sim if not row.get("ttsim_lib")]
    if unfetchable:
        raise GateError(
            "these simulator legs name no ttsim_lib, so their binary cannot be fetched: "
            + "; ".join(describe_row(row_key(row)) for row in unfetchable)
            + f". Give the SKU a ttsim_lib in {DEFAULT_SKU_CONFIG}."
        )

    sim_libs = sorted({row["ttsim_lib"] for row in sim})
    return selected, sim_libs


# ---------------------------------------------------------------------------
# reviews
# ---------------------------------------------------------------------------


def fetch_reviews(repo, pr, token):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr}/reviews?per_page=100"
    request = urllib.request.Request(url)
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        raise GateError(f"GitHub API returned {err.code} fetching reviews for {repo}#{pr}")
    except urllib.error.URLError as err:
        raise GateError(f"could not reach the GitHub API: {err.reason}")


def check_reviews(review_legs, repo, pr, head_sha, codeowners_path):
    """
    Require an approving review from the edited yaml's own code owners.

    Checked per file, because two blocked yamls can have different owners --
    an approval from the models owners says nothing about a fabric edit.

    dismiss_stale_reviews_on_push is false on main, so an approval survives later
    pushes. Requiring it to sit on the current head stops an approval of one
    galaxy edit from covering a different one pushed afterwards.

    Where a path resolves to no individual owners -- unowned, or owned only by a
    team, which the workflow token cannot expand -- this falls back to accepting
    any approving review and leaves the question to the normal CODEOWNERS system,
    which the ruleset enforces via require_code_owner_review.
    """
    if not review_legs:
        print("No legs need an owner review.")
        return 0

    rules = load_codeowners(codeowners_path)
    reviews = fetch_reviews(repo, pr, os.environ.get("GH_TOKEN"))
    approvers = {
        (review.get("user") or {}).get("login")
        for review in reviews
        if review.get("state") == "APPROVED" and review.get("commit_id") == head_sha
    } - {None}

    by_file = {}
    for leg in review_legs:
        by_file.setdefault(leg["file"], []).append(leg)

    unmet = []
    for path, legs in sorted(by_file.items()):
        reasons = sorted({leg.get("blocked_by") or "review_only_sku" for leg in legs})
        individuals, teams = owners_for(path, rules)
        matched = sorted(approvers & individuals)

        print(f"\n{path}  ({len(legs)} leg(s); {', '.join(reasons)})")
        for leg in legs:
            print(f"    - {leg['name']} | sku={leg['sku']}")

        if individuals:
            print(f"    code owners: {', '.join('@' + o for o in sorted(individuals))}")
            if matched:
                print(f"    approved by: {', '.join('@' + o for o in matched)}")
            else:
                unmet.append((path, sorted(individuals)))
        else:
            # No individual owner to match against; defer to the ruleset.
            where = f"team-owned ({', '.join(sorted(teams))})" if teams else "no code owner"
            print(f"    {where}; falling back to the normal CODEOWNERS review requirement")
            if approvers:
                print(f"    approved by: {', '.join('@' + o for o in sorted(approvers))}")
            else:
                unmet.append((path, []))

    if unmet:
        print("", file=sys.stderr)
        for path, owners in unmet:
            who = ", ".join("@" + o for o in owners) if owners else "a code owner"
            print(
                f"::error::{path} has legs this gate does not run. It needs an approving "
                f"review on {head_sha} from {who}.",
                file=sys.stderr,
            )
        return 1

    print(f"\nAll {len(review_legs)} leg(s) needing review are approved on {head_sha}.")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(args):
    """
    One pass: scope the diff, narrow the matrix, and enforce the owner review.

    Outputs are written before the review check so the workflow still has the
    matrix and counts even when the gate fails on a missing approval.
    """
    review_only = load_review_only_skus(args.review_skus, args.sku_config)
    tracy_files = set(split_list(args.tracy_files))
    non_matrix_files = set(split_list(args.non_matrix_files))
    unsupported_files = set(split_list(args.unsupported_files))
    base = merge_base(args.base)
    files = args.files if args.files is not None else changed_files(base)
    files = [f for f in files if f.startswith(TESTS_DIR + "/")]

    scope = build_scope(base, files, review_only, tracy_files, non_matrix_files, unsupported_files)

    # In a merge group the legs already ran on the PR head, so there is no matrix
    # to build and no review to re-check -- only the scope needs to resolve.
    dispatching = args.event == "pull_request"

    legs, sim_libs = [], []
    if dispatching and scope["run_legs"]:
        matrices = (
            load_matrices(args.matrix_dir)
            if args.matrix_dir
            else build_matrices(scope, args.prepare_script, args.sku_config, args.work_dir)
        )
        legs, sim_libs = filter_matrix(scope, matrices, load_ttsim_skips(args.ttsim_skip_list))

    scope["legs"] = legs
    scope["sim_libs"] = sim_libs

    text = json.dumps(scope, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")

    profiles = scope["profiles"]
    write_github_output(
        [
            ("status", scope["status"]),
            ("expected-leg-count", scope["expected_leg_count"]),
            ("review-leg-count", len(scope["review_legs"])),
            ("leg-digest", scope["leg_digest"]),
            ("changed-file-count", len(scope["changed_files"])),
            ("needs-default", str("default" in profiles).lower()),
            ("needs-profiler", str("profiler" in profiles).lower()),
            ("matrix", json.dumps(legs)),
            ("sim-libs", json.dumps(sim_libs)),
            ("leg-count", len(legs)),
        ]
    )

    if legs:
        print(f"\nWill run {len(legs)} leg(s):")
        for row in legs:
            path = "sim" if str(row.get("sku", "")).startswith("sim_") else "hw"
            print(f"  - [{path}] {describe_row(row_key(row))}  [{row['source_yaml']}]")

    if dispatching and scope["review_legs"]:
        return check_reviews(scope["review_legs"], args.repo, args.pr, args.head_sha, args.codeowners)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="origin/main", help="Base ref or sha to diff against")
    parser.add_argument(
        "--event",
        default="pull_request",
        choices=["pull_request", "merge_group"],
        help="merge_group scopes only: no matrix is built and no review is re-checked",
    )
    parser.add_argument("--files", nargs="*", default=None, help="Changed yamls; omit to derive from the diff")
    parser.add_argument("--sku-config", default=DEFAULT_SKU_CONFIG)
    parser.add_argument(
        "--review-skus",
        default=os.environ.get("REVIEW_ONLY_SKUS", ""),
        help="SKUs the gate must not dispatch (default: $REVIEW_ONLY_SKUS)",
    )
    parser.add_argument(
        "--tracy-files",
        default=os.environ.get("TRACY_BUILD_YAMLS", ""),
        help="Yaml basenames whose tests need a tracy build (default: $TRACY_BUILD_YAMLS)",
    )
    parser.add_argument(
        "--non-matrix-files",
        default=os.environ.get("NON_MATRIX_YAMLS", ""),
        help=(
            "Yaml basenames that legitimately hold no test matrix, e.g. ttsim-skip-list.yaml "
            "(default: $NON_MATRIX_YAMLS). Any other non-list yaml fails the gate."
        ),
    )
    parser.add_argument(
        "--unsupported-files",
        default=os.environ.get("UNSUPPORTED_YAMLS", ""),
        help=(
            "Yaml basenames the gate cannot dispatch, e.g. vllm_model_tests.yaml whose "
            "entries carry no cmd (default: $UNSUPPORTED_YAMLS). Edits to these are routed "
            "to their code owners for review instead of being run."
        ),
    )
    parser.add_argument("--codeowners", default=DEFAULT_CODEOWNERS)
    parser.add_argument("--ttsim-skip-list", default=DEFAULT_TTSIM_SKIP_LIST)
    parser.add_argument("--prepare-script", default=DEFAULT_PREPARE_SCRIPT)
    parser.add_argument("--work-dir", default="gate-matrices")
    parser.add_argument("--matrix-dir", default=None, help="Pre-built matrices; omit to run prepare_test_matrix.py")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""), help="owner/repo")
    parser.add_argument("--pr", default="")
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--output", default=None, help="Also write the JSON result here")

    args = parser.parse_args(argv)
    try:
        return run(args)
    except GateError as err:
        print(f"::error::{err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
