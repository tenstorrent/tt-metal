"""Create/verify a local immutable research snapshot without touching its index."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "experiments/sdpa-l2/"
BRANCH = "cglagovich/sdpa-recipes-frozen-20260921"
TAG = "sdpa-recipes-20260921-v1"
ROOTS = ("compute-sprint-v1", "compute-sprint-v2", "compute-sprint-v3",
         "flux2-frontier-v1", "wan-frontier-v1", "production-usage-audit-v1",
         "pareto-ring-v1", "recipe-freeze-v1")
TEXT_SUFFIXES = {".py", ".cpp", ".hpp", ".h", ".sh", ".md", ".json", ".csv", ".patch", ".svg"}


def git(*args, env=None):
    return subprocess.check_output(["git", *args], cwd=ROOT, env=env)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(ref):
    manifest = json.loads(git("show", f"{ref}:{PREFIX}recipe-freeze-v1/manifest.json"))
    for path, record in manifest["files"].items():
        blob = git("show", f"{ref}:{path}")
        assert len(blob) == record["bytes"] and sha(blob) == record["sha256"], path
    assert git("rev-parse", f"{ref}^").decode().strip() == manifest["research_parent"]
    print(f"Verified {len(manifest['files'])} file hashes at {ref}")


def create():
    manifest_path = HERE / "manifest.json"
    assert not manifest_path.exists(), "Freeze already exists"
    for ref in (f"refs/heads/{BRANCH}", f"refs/tags/{TAG}"):
        assert subprocess.run(["git", "show-ref", "--verify", "--quiet", ref], cwd=ROOT).returncode == 1, ref
    before_index = git("diff", "--cached", "--binary")
    assert not before_index, "Unexpected staged changes; preserve and resolve explicitly"
    before_diff = git("diff", "--binary")
    original_branch = git("symbolic-ref", "HEAD")
    parent = git("rev-parse", "HEAD").decode().strip()
    selected = set()
    for name in ROOTS:
        for p in (ROOT / PREFIX / name).rglob("*"):
            if not p.is_file() or p.is_symlink() or p.name.startswith("._") or "__pycache__" in p.parts:
                continue
            relative = str(p.relative_to(ROOT))
            media = p.suffix == ".png" and name in ("compute-sprint-v3", "pareto-ring-v1")
            media |= p.suffix in (".png", ".mp4") and any(s in relative for s in (
                "flux2-frontier-v1/replacement-suite-01/", "wan-frontier-v1/suite-01/"))
            if p.suffix in TEXT_SUFFIXES or media:
                selected.add(relative)
    selected.update(git("diff", "--name-only").decode().splitlines())
    # Include and verify the current decisive suite's exact selected-source pins.
    evidence = ["compute-sprint-v3/pareto/matched-v1.json"]
    evidence += [f"compute-sprint-v3/{folder}/{mode}-v1.json"
                 for folder in ("kv-precision-v1", "no-rounding-v1") for mode in ("smoke", "accuracy", "perf")]
    for relative in evidence:
        result = json.loads((ROOT / PREFIX / relative).read_text())
        assert result["complete"] and result["selected_sources_immutable"], relative
        for path, digest in result["source_sha256"].items():
            assert sha((ROOT / path).read_bytes()) == digest, (relative, path)
            selected.add(path)
    secret_pattern = re.compile(rb"hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----")
    for path in selected:
        if Path(path).suffix in TEXT_SUFFIXES:
            assert not secret_pattern.search((ROOT/path).read_bytes()), f"Potential credential in {path}; do not snapshot"
    files = {p: {"sha256": sha((ROOT / p).read_bytes()), "bytes": (ROOT / p).stat().st_size}
             for p in sorted(selected)}
    manifest = dict(schema_version=1, date="2026-09-21", research_parent=parent,
                    fresh_main=git("rev-parse", "origin/main").decode().strip(),
                    snapshot_branch=BRANCH, snapshot_tag=TAG,
                    selected_recipes=["D", "C", "B", "A", "E_bf16", "E_bfp8", "E_bfp4"],
                    diagnostic_only=["E_bf16_plain", "E_bfp8_plain", "E_bfp4_plain"],
                    excluded_recipe="F", research_tracked_diff_sha256=sha(before_diff),
                    submodules=git("submodule", "status").decode().splitlines(),
                    decisive_integrity_reports=evidence, files=files,
                    selected_file_count=len(files), selected_bytes=sum(x["bytes"] for x in files.values()),
                    exclusions="Untracked logs, PT/TRACY dumps, caches, compiler outputs, non-final media; existing parent tree retained")
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    selected.add(str(manifest_path.relative_to(ROOT)))
    with tempfile.TemporaryDirectory(prefix="sdpa-freeze-") as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp)/"index"))
        git("read-tree", parent, env=env)
        pathspec = Path(tmp)/"paths"
        pathspec.write_bytes(b"\0".join(p.encode() for p in sorted(selected))+b"\0")
        git("add", "-f", f"--pathspec-from-file={pathspec}", "--pathspec-file-nul", env=env)
        tree = git("write-tree", env=env).decode().strip()
        commit = git("commit-tree", tree, "-p", parent, "-m",
                     "Freeze selected SDPA recipes and measured evidence for production extraction", env=env).decode().strip()
    git("update-ref", f"refs/heads/{BRANCH}", commit, "0"*40)
    git("tag", "-a", TAG, commit, "-m", "Frozen SDPA D/C/B/A/E storage recipes and evidence; research only")
    assert git("symbolic-ref", "HEAD") == original_branch
    assert git("diff", "--cached", "--binary") == before_index
    assert git("diff", "--binary") == before_diff
    verify(TAG)
    print(json.dumps(dict(commit=commit, tag=TAG, branch=BRANCH, files=len(files),
                          bytes=manifest["selected_bytes"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--ref", default=TAG)
    args = parser.parse_args()
    create() if args.command == "create" else verify(args.ref)
