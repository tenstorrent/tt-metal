# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Overlays are captured against wherever a model's demo lived at the time. When
routing later moves the demo -- registering a family backend relocated one model
out of `models/tt_transformers/demo/<slug>/` and into `models/demos/<slug>/` --
`apply_for` kept replaying the patches at the recorded paths, because it runs
BEFORE scaffold and cannot know where this run will scaffold.

The failure was silent in the worst way: the run reported a healthy
`applied 16 model overlay(s)` while the live demo had no `.last_good_*` snapshot
at all, so every previously graduated component looked ungraduated and 17 hours
of bring-up was redone from scratch.

These pin the post-scaffold re-hang: files land under the demo this run actually
scaffolded, the index is re-keyed so it cannot recur, the stale copies are removed
so end-of-run capture stops re-recording them, and nothing is assumed about either
directory layout."""
import json

import pytest

from scripts.tt_hw_planner import overlay_manager as om


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A throwaway repo + overlay home, so nothing touches the real tree."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(om, "_REPO_OVERRIDE", root.resolve(), raising=False)
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays", raising=False)
    monkeypatch.setattr(om, "_overlays_dir_cache", None, raising=False)
    return root, tmp_path / "overlays"


MODEL = "vendor/Some-Model-30B"
SLUG = "some_model_30b"
OLD = f"models/tt_transformers/demo/{SLUG}"
NEW = f"models/demos/{SLUG}"


def _seed(root, overlays, entries, *, write_files=True):
    """Register `entries` (rel -> content) in the index, optionally materializing
    them at their recorded path the way a pre-scaffold `apply_for` would have."""
    idx = {}
    md = overlays / om._slug(MODEL)
    md.mkdir(parents=True, exist_ok=True)
    for rel, content in entries.items():
        patch_name = rel.replace("/", "__") + ".patch"
        (md / patch_name).write_text(f"diff --git a/{rel} b/{rel}\n(body)\n")
        idx[rel] = {"patch_file": patch_name, "classification": "shared", "sha256": "x"}
        if write_files:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
    (md / "index.json").write_text(json.dumps(idx, indent=2, sort_keys=True))
    return idx


def _demo(root):
    d = root / NEW
    (d / "_stubs").mkdir(parents=True, exist_ok=True)
    return d


# --- the suffix split ---------------------------------------------------------


def test_suffix_split_is_layout_agnostic():
    """The tail is found by the demo's OWN directory name, so neither the old nor
    the new prefix is assumed anywhere."""
    f = om._demo_relative_suffix
    assert f(f"{OLD}/_stubs/x.py", SLUG) == "_stubs/x.py"
    assert f(f"{NEW}/_stubs/x.py", SLUG) == "_stubs/x.py"
    assert f(f"models/anything/deeper/{SLUG}/tests/pcc/test_x.py", SLUG) == "tests/pcc/test_x.py"
    assert f(f"{OLD}/bringup_status.json", SLUG) == "bringup_status.json"


def test_suffix_split_declines_unrelated_paths():
    """A shared/tool path names no demo dir, so it must be left exactly alone."""
    f = om._demo_relative_suffix
    assert f("models/tt_transformers/tt/attention.py", SLUG) is None
    assert f("scripts/tt_hw_planner/cli.py", SLUG) is None
    assert f(f"models/demos/{SLUG}", SLUG) is None, "the demo dir itself has no in-demo tail"


# --- the re-hang --------------------------------------------------------------


def test_rehangs_snapshots_onto_the_scaffolded_demo(repo):
    """The reported bug: graduation snapshots sat in the dead path while the live
    demo had none, so the gate counted zero graduated."""
    root, overlays = repo
    _seed(
        root,
        overlays,
        {
            f"{OLD}/_stubs/comp_a.py": "native body",
            f"{OLD}/_stubs/comp_a.py.last_good_native": "native body",
            f"{OLD}/_stubs/comp_a.py.last_good_sharded": "sharded body",
        },
    )
    demo = _demo(root)
    n, files = om.reconcile_for(MODEL, demo)
    assert n == 3
    assert sorted(files) == sorted(
        [
            f"{NEW}/_stubs/comp_a.py",
            f"{NEW}/_stubs/comp_a.py.last_good_native",
            f"{NEW}/_stubs/comp_a.py.last_good_sharded",
        ]
    )
    assert (demo / "_stubs" / "comp_a.py.last_good_native").read_text() == "native body"
    assert (demo / "_stubs" / "comp_a.py.last_good_sharded").read_text() == "sharded body"


def test_index_is_rekeyed_so_it_cannot_recur(repo):
    root, overlays = repo
    _seed(root, overlays, {f"{OLD}/_stubs/comp_a.py.last_good_native": "native"})
    om.reconcile_for(MODEL, _demo(root))
    idx = om._load_index(MODEL)
    assert list(idx) == [f"{NEW}/_stubs/comp_a.py.last_good_native"]
    assert not any(k.startswith(OLD) for k in idx)


def test_stale_copy_is_removed_so_capture_stops_rerecording_it(repo):
    """End-of-run capture records changed files in the worktree. Leaving the stale
    copy behind would re-register the dead prefix and the next run would inherit it."""
    root, overlays = repo
    _seed(root, overlays, {f"{OLD}/_stubs/comp_a.py.last_good_native": "native"})
    om.reconcile_for(MODEL, _demo(root))
    assert not (root / OLD / "_stubs" / "comp_a.py.last_good_native").exists()


def test_restored_stub_wins_over_a_freshly_scaffolded_wrapper(repo):
    """Reconcile runs AFTER scaffold precisely so a graduated stub is not left
    sitting next to the torch wrapper the scaffolder just emitted -- the gate
    refuses to count that as graduated."""
    root, overlays = repo
    _seed(root, overlays, {f"{OLD}/_stubs/comp_a.py": "native ttnn body"})
    demo = _demo(root)
    (demo / "_stubs" / "comp_a.py").write_text("torch wrapper from scaffold")
    om.reconcile_for(MODEL, demo)
    assert (demo / "_stubs" / "comp_a.py").read_text() == "native ttnn body"


def test_already_correct_paths_are_untouched(repo):
    """A model whose demo never moved must be a complete no-op: no copies, no
    index rewrite, nothing reported."""
    root, overlays = repo
    before = _seed(root, overlays, {f"{NEW}/_stubs/comp_a.py.last_good_native": "native"})
    n, files = om.reconcile_for(MODEL, _demo(root))
    assert (n, files) == (0, [])
    assert om._load_index(MODEL) == before
    assert (root / NEW / "_stubs" / "comp_a.py.last_good_native").is_file()


def test_shared_and_tool_paths_are_left_alone(repo):
    """Shared model code and tool code are not demo-local; re-hanging them under
    the demo would corrupt the scope."""
    root, overlays = repo
    shared = "models/tt_transformers/tt/attention.py"
    tool = "scripts/tt_hw_planner/cli.py"
    _seed(root, overlays, {shared: "shared", tool: "tool"})
    n, _ = om.reconcile_for(MODEL, _demo(root))
    assert n == 0
    assert sorted(om._load_index(MODEL)) == sorted([shared, tool])
    assert (root / shared).read_text() == "shared"


def test_stale_duplicate_is_retired_not_allowed_to_clobber_the_good_entry(repo):
    """If a later capture already recorded the correct path, both keys map to the
    same target. Re-keying the stale one would silently overwrite the good entry's
    metadata, so the stale duplicate must be retired instead."""
    root, overlays = repo
    good = f"{NEW}/_stubs/comp_a.py.last_good_native"
    stale = f"{OLD}/_stubs/comp_a.py.last_good_native"
    _seed(root, overlays, {stale: "old body", good: "current body"})
    n, _ = om.reconcile_for(MODEL, _demo(root))
    idx = om._load_index(MODEL)
    assert list(idx) == [good], "the stale key must be dropped, the good one kept"
    assert n == 0, "nothing was re-hung — the good file was already in place"
    assert (root / good).read_text() == "current body", "the good file must not be overwritten"
    assert not (root / stale).exists(), "the stale duplicate must be removed"


def test_missing_stale_file_falls_back_to_the_patch(repo, monkeypatch):
    """If the pre-scaffold apply never materialized the file, the index patch is
    still the source of truth -- re-hang from it rather than losing the entry."""
    root, overlays = repo
    rel = f"{OLD}/_stubs/comp_a.py.last_good_native"
    _seed(root, overlays, {rel: "native"}, write_files=False)
    applied = {}

    def fake_apply(patch_text, **kw):
        applied["text"] = patch_text
        return 0, ""

    monkeypatch.setattr(om, "_git_apply", fake_apply)
    n, files = om.reconcile_for(MODEL, _demo(root))
    assert n == 1 and files == [f"{NEW}/_stubs/comp_a.py.last_good_native"]
    assert f"b/{NEW}/_stubs/comp_a.py.last_good_native" in applied["text"]
    assert OLD not in applied["text"], "the patch body must be re-pathed, not just the index key"


def test_stored_patch_body_is_repathed_so_a_second_run_cannot_strand_it(repo):
    """The trap this closes: re-keying the index alone leaves the patch BODY on the
    old path. `apply_for` targets the body, so the next run writes back to the dead
    location, while `reconcile_for` sees a correct key and no-ops -- stranding the
    snapshots again with nothing reported."""
    root, overlays = repo
    rel = f"{OLD}/_stubs/comp_a.py.last_good_native"
    _seed(root, overlays, {rel: "native"})
    om.reconcile_for(MODEL, _demo(root))

    idx = om._load_index(MODEL)
    new_rel = f"{NEW}/_stubs/comp_a.py.last_good_native"
    assert list(idx) == [new_rel]
    stored = (overlays / om._slug(MODEL) / idx[new_rel]["patch_file"]).read_text()
    assert OLD not in stored, "a stale body would send the next run back to the dead path"
    assert f"b/{new_rel}" in stored


def test_repath_keeps_filename_key_and_body_consistent(repo):
    """Index key, patch filename and patch body must agree, using the same filename
    convention `store_patch` writes."""
    root, overlays = repo
    rel = f"{OLD}/_stubs/comp_a.py"
    _seed(root, overlays, {rel: "native"})
    om.reconcile_for(MODEL, _demo(root))
    new_rel = f"{NEW}/_stubs/comp_a.py"
    meta = om._load_index(MODEL)[new_rel]
    assert meta["patch_file"] == om._patch_filename(new_rel)
    patch = overlays / om._slug(MODEL) / meta["patch_file"]
    assert patch.is_file()
    import hashlib

    assert meta["sha256"] == hashlib.sha256(patch.read_text().encode("utf-8")).hexdigest()
    assert not (overlays / om._slug(MODEL) / om._patch_filename(rel)).exists(), "stale patch file must go"


def test_reconcile_is_idempotent(repo):
    """Running it twice must be a no-op the second time — with the body repathed,
    the second pass has nothing left to correct."""
    root, overlays = repo
    _seed(root, overlays, {f"{OLD}/_stubs/comp_a.py.last_good_native": "native"})
    demo = _demo(root)
    first = om.reconcile_for(MODEL, demo)
    idx_after_first = om._load_index(MODEL)
    second = om.reconcile_for(MODEL, demo)
    assert first[0] == 1
    assert second == (0, [])
    assert om._load_index(MODEL) == idx_after_first


def test_repath_patch_rewrites_both_diff_sides():
    old, new = f"{OLD}/_stubs/x.py", f"{NEW}/_stubs/x.py"
    text = f"diff --git a/{old} b/{old}\n--- a/{old}\n+++ b/{old}\n@@ -1 +1 @@\n-x\n+y\n"
    out = om._repath_patch(text, old, new)
    assert f"a/{new}" in out and f"b/{new}" in out
    assert old not in out


def test_repath_patch_leaves_the_stubs_own_imports_alone():
    """An ADAPT stub legitimately imports shared library modules that live under the
    prefix being moved away from. Rewriting a content line would turn a working
    import into a module that does not exist, so only path headers are touched."""
    old, new = f"{OLD}/_stubs/x.py", f"{NEW}/_stubs/x.py"
    body_line = "+from models.tt_transformers.tt.mixtral_moe import TtMoeLayer\n"
    text = f"diff --git a/{old} b/{old}\n--- /dev/null\n+++ b/{old}\n@@ -0,0 +1 @@\n{body_line}"
    out = om._repath_patch(text, old, new)
    assert body_line in out, "the stub's own import must survive untouched"
    assert f"+++ b/{new}" in out
    assert f"b/{old}" not in out


def test_empty_or_absent_index_is_a_noop(repo):
    root, _ = repo
    assert om.reconcile_for(MODEL, _demo(root)) == (0, [])


def test_demo_outside_the_repo_is_declined(repo, tmp_path):
    """Nothing to reconcile if the demo isn't in the active repo; must not raise."""
    root, overlays = repo
    _seed(root, overlays, {f"{OLD}/_stubs/comp_a.py": "native"})
    assert om.reconcile_for(MODEL, tmp_path / "elsewhere" / SLUG) == (0, [])


def _code_without_docstrings(*fns) -> str:
    """Source of `fns` with docstrings stripped. The prose is allowed to name the
    two layouts that motivated this fix; the executable code is not."""
    import ast
    import inspect
    import textwrap

    out = []
    for fn in fns:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    if isinstance(body[0].value.value, str):
                        node.body = body[1:] or [ast.Pass()]
        out.append(ast.unparse(tree))
    return "\n".join(out)


def test_no_model_or_stage_vocabulary_in_the_rehang():
    """Constraint 3: the re-hang is driven by the demo's own directory name and the
    recorded paths, never by a component, stage, or layout name typed in here."""
    src = _code_without_docstrings(om._demo_relative_suffix, om.reconcile_for)
    for forbidden in (
        "tt_transformers",
        "models/demos",
        "nemotron",
        "lightning",
        "prefill",
        "decode",
        "attention",
        "mixer",
    ):
        assert forbidden not in src, f"re-hang must not reference {forbidden!r}"
