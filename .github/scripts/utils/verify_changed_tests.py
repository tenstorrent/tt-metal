#!/usr/bin/env python3
"""
All logic for .github/workflows/verify-changed-tests.yaml, in a single pass.

The gate proves that the test entries a PR edits in tests/pipeline_reorg/ still
run green, and nothing more. One invocation does two things:

  1. Diffs the changed yamls at test-entry level, working out which entries were
     added or behaviourally changed, which legs (entry x SKU) must run, and which
     build flavours those legs need. Also scopes entries whose cmd names an edited
     file (see "Source references").

  2. Runs prepare_test_matrix.py over those yamls and narrows its output to exactly
     those legs, so the dispatched matrix is the touched work and nothing else.

With --event merge_group only step 1 runs: those legs already ran on the PR head,
and the queue only needs the scope to resolve.

It decides what to run and says nothing about who may merge it. Anything it cannot
dispatch is dropped, not reported: UNSUPPORTED_YAMLS before the file is read, and
UNSUPPORTED_SKUS per entry (an entry left with no runnable SKU is dropped whole).
No approval is looked up -- CODEOWNERS already gates edits to these files.

Design notes
------------
Entries are keyed on (name, gtest_shard_index) within the yaml they came from, and
matrix rows on that pair plus the SKU. The source yaml is part of the identity because
the same test is deliberately mirrored between pipelines at different gate tiers.
prepare_test_matrix.py and verify_time_budget.py both reject a yaml that reuses a pair.

Edits that cannot change how a test executes do not need hardware:

  * owner_id / team    Ownership metadata.

Anything else -- cmd, timeout, adding or removing a SKU, arch, dispatch_mode,
tier, shard layout, model fields -- is treated as behaviour-affecting.

`timeout` counts: verify_time_budget.py only proves it fits the team's budget, not
that the test still finishes inside it. Lowering one is how a leg starts timing out.

Source references
-----------------
An entry is also scoped when its cmd names a file the PR changed. A token counts
only if it resolves to a tracked path -- a file, or a directory pytest is pointed
at; filters and flags are ignored.

Not transitive, and `./build/test/...` is untracked build output, so gtest legs are
reached only through their yaml.

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
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

TESTS_DIR = "tests/pipeline_reorg"

# Top-level entry fields that cannot change how a test executes.
METADATA_FIELDS = {"owner_id", "team"}

# A pytest per-test timeout, which is stripped from simulator legs: ttsim is
# 10-50x slower than hardware, so a limit sized for silicon kills slow-but-correct
# tests. The job's own timeout-minutes stays as the backstop. ttnn-sanity-tests-impl
# does the same with `sed -E 's/[[:space:]]+--timeout[[:space:]]+[0-9]+//g'`.
PYTEST_TIMEOUT_FLAG = re.compile(r"\s+--timeout\s+\d+")

# Paths only the installed tt-metalium debs provide; a cmd touching one needs the
# packages artifact rather than a build tree.
PACKAGE_INSTALL_PREFIXES = ("/usr/share/tt-metalium", "/usr/libexec/tt-metalium")

# Where the LLK pipelines run their cmds from (their LLK_PATH). Their entries open
# with `cd tests/python_tests`, which from the gate's /work would otherwise land in
# tt-metal's own tests/.

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


def git_show_many(ref, paths):
    """{path: contents at ref}, with a path missing from ref mapped to None.

    One `git cat-file --batch` rather than a `git show` per file. Each object comes
    back as a header -- `<sha> blob <size>`, or `<request> missing` -- then exactly
    <size> bytes and a newline. Reading by that size rather than splitting on
    newlines is what stops a yaml containing a `missing` line from being read as a
    header.
    """
    paths = list(dict.fromkeys(paths))
    if not paths:
        return {}

    ref = checked_revision(ref)
    request = "".join(f"{ref}:{path}\n" for path in paths)
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        input=request.encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise GateError(f"git cat-file at {ref} failed: {result.stderr.decode(errors='replace').strip()}")

    out, pos, contents = result.stdout, 0, {}
    for path in paths:
        newline = out.find(b"\n", pos)
        if newline == -1:
            raise GateError(f"git cat-file output ended before {ref}:{path}")
        header = out[pos:newline].decode(errors="replace")
        pos = newline + 1
        if header.endswith(" missing"):
            contents[path] = None
            continue
        # Anything but a blob names a tree or tag, whose bytes are not a tests yaml.
        fields = header.rsplit(" ", 2)
        if len(fields) != 3 or fields[1] != "blob" or not fields[2].isdigit():
            raise GateError(f"unexpected git cat-file header for {ref}:{path}: {header!r}")
        size = int(fields[2])
        if pos + size > len(out):
            raise GateError(f"git cat-file returned a short blob for {ref}:{path}")
        contents[path] = out[pos : pos + size].decode(errors="replace")
        pos += size + 1  # trailing newline cat-file adds after each blob
    return contents


# What a git ref or object id may contain, and it must open with an alphanumeric.
SAFE_REVISION = re.compile(r"\A[0-9A-Za-z][0-9A-Za-z._/@^~-]*\Z")


def checked_revision(revision):
    """A git revision that cannot be read as an option, or GateError."""
    if not isinstance(revision, str) or not SAFE_REVISION.match(revision):
        raise GateError(f"refusing to use {revision!r} as a git revision: not a well-formed ref or object id")
    return revision


def merge_base(base):
    """
    Resolve base to its merge base with HEAD.

    Diffing against the merge base rather than the base tip keeps commits that
    landed on main after the branch point out of scope, and makes the working
    tree visible -- which is what makes this runnable locally before pushing.

    Checks the revision here so every later call inherits a validated one.
    """
    result = subprocess.run(["git", "merge-base", checked_revision(base), "HEAD"], capture_output=True, text=True)
    if result.returncode != 0:
        return base
    return checked_revision(result.stdout.strip()) if result.stdout.strip() else base


def changed_files(base):
    """
    pipeline_reorg yamls that differ between base and the working tree.

    Untracked files count too: a brand new pipeline yaml is exactly the case that
    must not slip past the gate, and it is still untracked when this is run
    locally before committing.
    """
    diff = subprocess.run(
        ["git", "diff", "--name-only", checked_revision(base), "--", TESTS_DIR], capture_output=True, text=True
    )
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


def all_test_yamls():
    """Every tests yaml at HEAD; the source pass needs all of them, not just edited ones."""
    if not os.path.isdir(TESTS_DIR):
        return []
    root = Path(TESTS_DIR)
    return sorted(str(p) for suffix in ("*.yaml", "*.yml") for p in root.rglob(suffix))


def tracked_paths(base):
    """Files git tracks at `base` or at HEAD.

    Both revisions: a cmd can name a file the PR deletes or renames away, and
    resolving against HEAD alone would reject the token and drop the leg. Build
    output is tracked in neither, so generated paths still never match.
    """
    paths = set()
    tree = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", checked_revision(base), "--"], capture_output=True, text=True
    )
    if tree.returncode != 0:
        raise GateError(f"git ls-tree at {base} failed: {tree.stderr.strip()}")
    paths |= set(tree.stdout.splitlines())

    # The index plus untracked files rather than ls-tree HEAD, so a test file added
    # but not yet committed still resolves when this is run locally. --exclude-standard
    # honours .gitignore, which is what keeps build output out.
    index = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True, text=True
    )
    if index.returncode != 0:
        raise GateError(f"git ls-files failed: {index.stderr.strip()}")
    paths |= set(index.stdout.splitlines())
    return paths


def changed_sources(base):
    """Changed files outside the tests yamls, i.e. the code a cmd might name.

    The yamls are excluded because scope_file already diffs them at entry level.

    --no-renames because detection (on by default) reports only a rename's new path,
    so a cmd naming the old one would match nothing. Unlike rename_sources(), which
    wants the pairing, both sides matter here: the cmd is broken either way.
    """
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", checked_revision(base), "--"], capture_output=True, text=True
    )
    if diff.returncode != 0:
        raise GateError(f"git diff against {base} failed: {diff.stderr.strip()}")

    # Untracked too, so a brand new source file counts when this is run locally
    # before committing -- the same reason changed_files() collects them.
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True)
    if untracked.returncode != 0:
        raise GateError(f"git ls-files --others failed: {untracked.stderr.strip()}")

    names = set(diff.stdout.splitlines()) | set(untracked.stdout.splitlines())
    return {name for name in names if name and not name.startswith(TESTS_DIR + "/")}


def tracked_dirs(tracked):
    """Every directory prefix of a tracked file.

    Precomputed so the directory fallback below is a set lookup rather than a scan
    of all ~24k tracked paths per token, per entry, per yaml.
    """
    dirs = set()
    for path in tracked:
        head = path.rsplit("/", 1)[0] if "/" in path else ""
        while head and head not in dirs:
            dirs.add(head)
            head = head.rsplit("/", 1)[0] if "/" in head else ""
    return dirs


# Shell punctuation and pytest/CLI decorations wrapped around a path. `::` splits a
# pytest node id off its file, `=` an --deselect=<path> style flag from its value.
CMD_TOKEN_SPLIT = re.compile(r"[\s;|&()<>{}\"'`=]+")


def submodule_paths():
    """Tracked gitlinks, by mode 160000.

    A tracked ancestor is not enough: a cmd can name a bogus path under a regular
    file, and charging that to the file would be wrong.
    """
    result = subprocess.run(["git", "ls-files", "--stage"], capture_output=True, text=True)
    if result.returncode != 0:
        raise GateError(f"git ls-files --stage failed: {result.stderr.strip()}")
    links = set()
    for line in result.stdout.splitlines():
        meta, _, path = line.partition("\t")
        if meta.startswith("160000") and path:
            links.add(path)
    return links


def gitlink_ancestor(path, gitlinks):
    """The submodule root path sits inside, or None."""
    head = path
    while "/" in head:
        head = head.rsplit("/", 1)[0]
        if head in gitlinks:
            return head
    return None


def cmd_referenced_paths(cmd, tracked, dirs, workdir="", gitlinks=frozenset()):
    """Repo paths a cmd names outright.

    A token counts only if it is a tracked file or a directory tracked files sit
    under, so `-k some_filter` and untracked `./build/test/...` never match.

    workdir is the leg's working directory (see entry_workdir): the LLK yamls'
    cmds are relative to it. Tokens resolve workdir-first then from the root, since
    a cmd can legitimately name both.
    """
    prefixes = ([workdir] if workdir else []) + [""]
    paths = set()
    for raw in CMD_TOKEN_SPLIT.split(cmd or ""):
        stripped = raw.split("::", 1)[0].strip().removeprefix("./")
        token = stripped.rstrip("/")
        if not token or ("/" not in token and not stripped.endswith("/")):
            continue
        for prefix in prefixes:
            candidate = f"{prefix}/{token}" if prefix else token
            if candidate in tracked:
                paths.add(candidate)
                break
            if candidate in dirs:
                paths.add(candidate + "/")
                break
            # A submodule's contents are one gitlink, so a path inside it resolves
            # to nothing. Charge the gitlink: bumping it is what changes those tests.
            gitlink = gitlink_ancestor(candidate, gitlinks)
            if gitlink:
                paths.add(gitlink)
                break
    return paths


def sources_touching(cmd, tracked, dirs, changed, workdir="", gitlinks=frozenset()):
    """The changed files a cmd names, directly or via a directory it points at."""
    hits = set()
    for path in cmd_referenced_paths(cmd, tracked, dirs, workdir, gitlinks):
        if path.endswith("/"):
            hits |= {name for name in changed if name.startswith(path)}
        elif path in changed:
            hits.add(path)
    return sorted(hits)


def rename_sources(base):
    """
    {new path: path it was renamed from} for yamls moved between base and HEAD.

    Without this a rename reads as a brand new file: `git show base:<new path>`
    finds nothing, the base matrix comes back empty, and every entry in the file
    looks added -- so renaming a pipeline yaml demanded a hardware run and an
    owner review for its whole matrix. Pairing the paths means a pure rename
    scopes to nothing, and a rename that also edits entries scopes to just those.

    -M is passed explicitly rather than relying on diff.renames, which a local
    config can switch off; without rename detection git reports the pair as an
    unrelated delete plus add and the pairing is lost again.

    A yaml moved in from outside TESTS_DIR is deliberately not paired: the gate
    never saw it before, so its entries are genuinely new to the gate's scope.
    """
    diff = subprocess.run(
        ["git", "diff", "--name-status", "-M", checked_revision(base), "--", TESTS_DIR],
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0:
        raise GateError(f"git diff --name-status against {base} failed: {diff.stderr.strip()}")

    sources = {}
    for line in diff.stdout.splitlines():
        fields = line.split("\t")
        # "R<similarity>\told\tnew"; plain edits are "M\tpath" and are not pairs.
        if len(fields) == 3 and fields[0].startswith("R"):
            _, old_path, new_path = fields
            sources[new_path] = old_path
    return sources


def load_unsupported_skus(raw, sku_config_path):
    """
    Parse the list of SKUs this gate must never dispatch.

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
            f"unsupported SKUs not present in {sku_config_path}: {', '.join(unknown)}. "
            "Fix UNSUPPORTED_SKUS in .github/workflows/verify-changed-tests.yaml."
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
    """Identity of a test entry within its own yaml: name plus gtest shard.

    Callers supply the yaml half -- entries are indexed per file. Shard counts because
    llk_merge_gate_tests.yaml runs four "LLK FD wormhole" entries differing only by it.
    """
    return (
        str(entry.get("name", "")),
        str(entry.get("gtest_shard_index", "")),
    )


def key_str(key):
    name, shard = key
    return f"{name} | shard={shard}" if shard else name


def index_entries(entries, path, strict=True):
    """Map composite key -> entry, rejecting collisions rather than guessing.

    The base revision is indexed with strict=False. It predates this guard, so it can
    still hold the duplicates a PR is removing -- erroring there would fail the very PR
    that fixes them. A duplicated key is dropped from the base index instead, which
    reads the head entry as added and runs its legs: the same answer the gate already
    gives when it cannot match an entry to a previous one.
    """
    index = {}
    ambiguous = set()
    for entry in entries:
        key = entry_key(entry)
        if key in index or key in ambiguous:
            if strict:
                raise GateError(
                    f"{path}: two entries share the key '{key_str(key)}'. "
                    "The gate cannot tell which one an edit touched -- names must be unique "
                    "within one yaml, with gtest_shard_index counting as part of the name."
                )
            index.pop(key, None)
            ambiguous.add(key)
            continue
        index[key] = entry
    return index


def behavioural_view(entry):
    """The parts of an entry that can change how a test executes.

    Everything under `skus` counts, `timeout` included.
    """
    return {k: copy.deepcopy(v) for k, v in entry.items() if k not in METADATA_FIELDS}


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


def entry_env(entry, path):
    """The entry's declared `env`, flattened to KEY=VALUE lines for the run step.

    Some cmds read ambient variables -- the LLK ones run `pytest -m "$PYTEST_MARKERS"`
    under `set -u` -- which the owning pipeline sets from a workflow input. Declaring
    them on the entry keeps the value next to the cmd that needs it, so this gate and
    any other consumer read the same source instead of each carrying its own copy.
    """
    env = entry.get("env")
    if env is None:
        return ""
    if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
        raise GateError(f"{path}: entry '{entry.get('name')}' has an invalid 'env'; expected a mapping of names")
    for key, value in env.items():
        if "\n" in key or "\n" in str(value):
            raise GateError(f"{path}: env {key!r} spans lines; the run step reads one KEY=VALUE per line")
    return "\n".join(f"{key}={value}" for key, value in env.items())


def entry_workdir(entry, path):
    """The directory a leg's cmd runs from, relative to the checkout root.

    Declared on the entry, like `env`, because it is a property of the test rather
    than of this gate: the LLK cmds open with `cd tests/python_tests` or call
    `./tests/run_llk_perf_*.sh`, both relative to the tree their pipeline sets as
    its working-directory. Empty means the checkout root, which is everything else.
    """
    workdir = entry.get("workdir")
    if workdir is None:
        return ""
    if not isinstance(workdir, str) or not workdir or "\n" in workdir:
        raise GateError(f"{path}: entry '{entry.get('name')}' has an invalid 'workdir'; expected a relative path")
    if workdir.startswith("/") or ".." in workdir.split("/"):
        raise GateError(f"{path}: entry '{entry.get('name')}' workdir {workdir!r} must stay inside the checkout")
    return workdir.rstrip("/")


def scope_file(
    path,
    base,
    unsupported_skus,
    tracy_files,
    non_matrix_files,
    unsupported_files,
    base_path=None,
    tracked=None,
    dirs=None,
    gitlinks=None,
    changed_src=None,
    edited=True,
    base_blobs=None,
):
    """Diff one tests yaml and return its scoping result.

    base_path is where the file lived at `base` when it has since been renamed;
    the base matrix is read from there so a rename is not mistaken for a file
    full of new entries.

    tracked/dirs/changed_src drive the source pass: an entry is scoped when its cmd
    names an edited file, even if its own yaml is untouched.

    edited says whether the PR touched this yaml. The source pass reads every yaml,
    so an unedited broken one is skipped rather than failing the gate -- for every
    way it can be broken, not just a reshaped matrix: bad YAML syntax, entries that
    are not mappings and duplicate keys all raise too.
    """
    skip = {"no_entries": True, "run_legs": [], "dropped_legs": []}

    # Excluded before anything is read: not parsed, diffed or scanned. non_matrix
    # holds no test matrix at all; unsupported holds one the gate cannot dispatch.
    if os.path.basename(path) in non_matrix_files or os.path.basename(path) in unsupported_files:
        return skip

    old_path = base_path or path

    try:
        new_entries = parse_entries(open(path).read() if os.path.exists(path) else None)
        # An unedited yaml cannot differ from its base, so it is never requested.
        old_entries = new_entries if not edited else parse_entries((base_blobs or {}).get(old_path))

        if not edited and (new_entries is None or old_entries is None):
            return skip

        # Reaching here with a non-list means a real test matrix was reshaped into
        # something else. prepare_test_matrix.py errors on that shape, and so does the
        # gate -- silently marking it "no tests" would pass the PR with zero coverage.
        for entries, label in ((new_entries, path), (old_entries, f"{old_path}@{base}")):
            if entries is None:
                raise GateError(
                    f"{label} is not a list of test entries. If it legitimately holds no test "
                    "matrix, add its filename to NON_MATRIX_YAMLS in "
                    ".github/workflows/verify-changed-tests.yaml; otherwise fix the file."
                )

        old_index = index_entries(old_entries, f"{old_path}@{base}", strict=False)
        new_index = index_entries(new_entries, path)
    except (GateError, yaml.YAMLError):
        if edited:
            raise
        return skip

    touched = []
    for key, entry in new_index.items():
        previous = old_index.get(key)
        if previous is None:
            touched.append((key, entry, "added"))
        elif canonical(behavioural_view(previous)) != canonical(behavioural_view(entry)):
            touched.append((key, entry, "changed"))
    # Removed entries are ignored: no test left to prove. Metadata-only edits fall
    # out here too, since behavioural_view() excludes them.

    # Entries already touched above keep their reason and are not added twice.
    if changed_src:
        already = {key for key, _, _ in touched}
        for key, entry in new_index.items():
            if key in already:
                continue
            hits = sources_touching(
                entry.get("cmd") or "",
                tracked or set(),
                dirs or set(),
                changed_src,
                entry_workdir(entry, path),
                gitlinks or frozenset(),
            )
            if hits:
                touched.append((key, entry, "source:" + ",".join(hits)))

    run_legs, dropped = [], []
    for key, entry, reason in touched:
        skus = entry.get("skus")
        if not isinstance(skus, dict) or not skus:
            raise GateError(f"{path}: entry '{key_str(key)}' has no skus mapping; cannot resolve any leg to run")

        # Counted, never gated on, so a touched galaxy entry does not read as
        # "nothing to prove". Who may merge it stays CODEOWNERS' business.
        runnable = [sku for sku in skus if sku not in unsupported_skus]
        dropped.extend({"file": path, "name": entry.get("name"), "sku": sku} for sku in skus if sku in unsupported_skus)
        if not runnable:
            continue

        team = entry.get("team")
        workdir = entry_workdir(entry, path)
        for sku in runnable:
            run_legs.append(
                {
                    "file": path,
                    "name": entry.get("name"),
                    "gtest_shard_index": entry.get("gtest_shard_index"),
                    "sku": sku,
                    "team": team,
                    "reason": reason,
                    "profile": resolve_profile(path, tracy_files),
                    "packages": needs_packages(entry),
                    "workdir": workdir,
                    "env": entry_env(entry, path),
                }
            )

    return {"no_entries": False, "run_legs": run_legs, "dropped_legs": dropped}


def build_scope(
    base,
    files,
    unsupported_skus,
    tracy_files,
    non_matrix_files,
    unsupported_files,
    renames=None,
    changed_src=None,
):
    run_legs, dropped_legs, skipped = [], [], []
    renames = renames or {}

    # The source pass must read every yaml; without it the scan is just the diff.
    tracked = tracked_paths(base) if changed_src else set()
    dirs = tracked_dirs(tracked) if changed_src else set()
    gitlinks = submodule_paths() if changed_src else frozenset()
    scanned = sorted(set(files) | (set(all_test_yamls()) if changed_src else set()))

    # Every base revision the diff needs, in one call. Unedited and excluded yamls
    # are never parsed, so neither is requested.
    excluded = non_matrix_files | unsupported_files
    base_blobs = git_show_many(
        base, [renames.get(path, path) for path in files if os.path.basename(path) not in excluded]
    )

    for path in scanned:
        scoped = scope_file(
            path,
            base,
            unsupported_skus,
            tracy_files,
            non_matrix_files,
            unsupported_files,
            base_path=renames.get(path),
            tracked=tracked,
            dirs=dirs,
            gitlinks=gitlinks,
            changed_src=changed_src,
            edited=path in files,
            base_blobs=base_blobs,
        )
        if scoped["no_entries"]:
            # Only a yaml the PR edited is worth reporting as skipped.
            if path in files:
                skipped.append(path)
            continue
        run_legs.extend(scoped["run_legs"])
        dropped_legs.extend(scoped["dropped_legs"])

    status = "run" if run_legs else "no_op"

    # Identifies exactly this set of legs, so the merge_group run can confirm it is
    # looking at the same work the pull_request run proved.
    digest = hashlib.sha256(canonical(sorted(canonical(leg) for leg in run_legs)).encode()).hexdigest()

    # Only the build flavours the surviving legs need, so a PR touching one cpp
    # pipeline does not pay for a wheel or a tracy build.
    profiles = sorted({leg["profile"] for leg in run_legs})

    return {
        "status": status,
        "changed_files": sorted(files),
        "changed_sources": sorted(changed_src or []),
        "skipped_files": skipped,
        "run_legs": run_legs,
        "dropped_legs": dropped_legs,
        "expected_leg_count": len(run_legs),
        "profiles": profiles,
        "leg_digest": digest,
    }


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def row_key(row):
    """
    Key a matrix row the same way an entry in the source yaml is keyed.

    entry_key()'s pair plus the SKU, since one entry expands to one row per SKU.
    build_test_matrix() appends " [<concrete sku>]" to the name and sets logical_sku
    when the concrete SKU differs; both are undone here so rows line up with the legs.
    """
    sku = str(row.get("sku", ""))
    name = str(row.get("name", ""))
    suffix = f" [{sku}]"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    logical_sku = str(row.get("logical_sku") or sku)
    return (
        name,
        str(row.get("gtest_shard_index", "")),
        logical_sku,
    )


def leg_row_key(leg):
    shard = leg.get("gtest_shard_index")
    return (
        str(leg.get("name") or ""),
        "" if shard is None else str(shard),
        str(leg.get("sku") or ""),
    )


def describe_row(key):
    name, shard, sku = key
    parts = [name, f"sku={sku}"]
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

    # Off the legs, not the changed files: a source-scoped leg can come from a yaml
    # the PR never touched.
    needed = sorted({leg["file"] for leg in scope["run_legs"]})

    matrices = {}
    for path in needed:
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
            row["gate_workdir"] = leg["workdir"]
            row["gate_env"] = leg["env"]
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
            + ". Add their SKUs to UNSUPPORTED_SKUS in verify-changed-tests.yaml."
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
# entry point
# ---------------------------------------------------------------------------


def run(args):
    """One pass: scope the diff, then narrow the matrix to what it found."""
    unsupported_skus = load_unsupported_skus(args.unsupported_skus, args.sku_config)
    tracy_files = set(split_list(args.tracy_files))
    non_matrix_files = set(split_list(args.non_matrix_files))
    unsupported_files = set(split_list(args.unsupported_files))
    base = merge_base(args.base)
    files = args.files if args.files is not None else changed_files(base)
    files = [f for f in files if f.startswith(TESTS_DIR + "/")]

    # A renamed yaml's base matrix lives at its old path, so a rename is scoped as
    # the no-op it is rather than as a file full of newly added entries.
    renames = rename_sources(base)
    if renames:
        for new_path, old_path in sorted(renames.items()):
            print(f"Renamed: {old_path} -> {new_path} (diffed against its old path)")

    # Opt-out exists because the scan reads every tests yaml, not just --files.
    changed_src = set() if args.no_source_scope else changed_sources(base)
    if changed_src:
        print(f"{len(changed_src)} changed source file(s) outside {TESTS_DIR}/")

    scope = build_scope(
        base,
        files,
        unsupported_skus,
        tracy_files,
        non_matrix_files,
        unsupported_files,
        renames=renames,
        changed_src=changed_src,
    )

    # In a merge group the legs already ran on the PR head, so there is no matrix
    # to build -- only the scope needs to resolve.
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
            ("dropped-leg-count", len(scope["dropped_legs"])),
            ("leg-digest", scope["leg_digest"]),
            ("changed-file-count", len(scope["changed_files"])),
            ("needs-default", str("default" in profiles).lower()),
            ("needs-profiler", str("profiler" in profiles).lower()),
            ("matrix", json.dumps(legs)),
            ("sim-libs", json.dumps(sim_libs)),
            ("leg-count", len(legs)),
        ]
    )

    if scope["dropped_legs"]:
        print(f"\nDropped {len(scope['dropped_legs'])} leg(s) on SKUs this gate does not dispatch:")
        for leg in scope["dropped_legs"]:
            print(f"  - {leg['name']} | sku={leg['sku']}  [{leg['file']}]")

    if legs:
        print(f"\nWill run {len(legs)} leg(s):")
        for row in legs:
            path = "sim" if str(row.get("sku", "")).startswith("sim_") else "hw"
            print(f"  - [{path}] {describe_row(row_key(row))}  [{row['source_yaml']}]")

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
        "--unsupported-skus",
        default=os.environ.get("UNSUPPORTED_SKUS", ""),
        help="SKUs the gate must not dispatch (default: $UNSUPPORTED_SKUS)",
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
    parser.add_argument(
        "--no-source-scope",
        action="store_true",
        help=(
            "Skip the source pass, scoping only from edits to the tests yamls themselves. "
            "Without it an entry is also scoped when its cmd names a file the PR changed."
        ),
    )
    parser.add_argument("--ttsim-skip-list", default=DEFAULT_TTSIM_SKIP_LIST)
    parser.add_argument("--prepare-script", default=DEFAULT_PREPARE_SCRIPT)
    parser.add_argument("--work-dir", default="gate-matrices")
    parser.add_argument("--matrix-dir", default=None, help="Pre-built matrices; omit to run prepare_test_matrix.py")
    parser.add_argument("--output", default=None, help="Also write the JSON result here")

    args = parser.parse_args(argv)
    try:
        return run(args)
    except GateError as err:
        print(f"::error::{err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
