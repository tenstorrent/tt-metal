#!/usr/bin/env python3
"""Reproduce frozen stratified cohorts from pinned, metadata-only checkouts."""
import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def pinned_clean(root, revision):
    assert git(root, "rev-parse", "HEAD").decode().strip() == revision
    assert not git(root, "status", "--porcelain")


def ordering(entries, seed):
    def key(name):
        return hashlib.sha256((seed + "\0" + name).encode("utf-8")).hexdigest()

    groups = collections.defaultdict(list)
    for name, metadata in entries.items():
        groups[metadata["stratum"]].append(name)
    strata = sorted(groups, key=key)
    queues = {name: collections.deque(sorted(groups[name], key=key)) for name in strata}
    ordered = []
    while any(queues.values()):
        for name in strata:
            if queues[name]:
                ordered.append(queues[name].popleft())
    assert len(ordered) == len(set(ordered)) == len(entries)
    return ordered, strata


def task_metadata(path, stratum, root):
    data = path.read_bytes()
    parsed = tomllib.loads(data.decode())
    environment = parsed["environment"]
    assert "cpus" in environment
    assert "memory" in environment or "memory_mb" in environment
    resources = {
        key: environment[key]
        for key in (
            "cpus",
            "memory",
            "memory_mb",
            "storage",
            "storage_mb",
            "gpus",
            "build_timeout_sec",
        )
        if key in environment
    }
    return {
        "stratum": stratum,
        "definition": str(path.relative_to(root)),
        "definition_sha256": hashlib.sha256(data).hexdigest(),
        "environment_resources": resources,
        "agent_timeout_sec": parsed.get("agent", {}).get("timeout_sec"),
        "verifier_timeout_sec": parsed.get("verifier", {}).get("timeout_sec"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    cohorts_path = args.execution / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/cohorts.json"
    before = cohorts_path.read_bytes()
    cohorts = json.loads(before)
    harbor = args.execution / "bringup/artifacts/parser-protocol-harbor"
    harbor_sha = "e76c762fca5267dfb36e1f900f0a7724ed72e735"
    swe_root = args.metadata_root / "swebench-task-metadata"
    swe_sha = "86723674f04e4209ac479d0fb75d9d9f44b4377e"
    tb_root = args.metadata_root / "terminal-bench-2-1-metadata"
    tb_sha = "12b7565679ce5006230b820666353673a133db93"
    pinned_clean(swe_root, swe_sha)
    pinned_clean(tb_root, tb_sha)
    registry_raw = git(harbor, "show", harbor_sha + ":registry.json")
    registry = json.loads(registry_raw)
    swe = next(x for x in registry if x["name"] == "swebench-verified" and x["version"] == "1.0")
    assert swe == json.loads(
        (args.execution / "bringup/artifacts/stage12-dataset-metadata/swebench-verified-registry.json").read_text()
    )
    entries = {"swe_bench_verified": {}, "terminal_bench_2_1": {}}
    for task in swe["tasks"]:
        assert task["git_commit_id"] == swe_sha
        assert task["git_url"] == "https://github.com/laude-institute/harbor-datasets.git"
        name = task["name"]
        assert name not in entries["swe_bench_verified"]
        entries["swe_bench_verified"][name] = task_metadata(
            swe_root / task["path"] / "task.toml", name.split("__")[0], swe_root
        )
    # Original SWE stratification used the owner prefix before '__', recorded as
    # repository in task_resources.json, not the owner/repo pair.
    for path in sorted((tb_root / "tasks").glob("*/task.toml")):
        metadata = tomllib.loads(path.read_text())["metadata"]
        name = "terminal-bench/" + path.parent.name
        assert name not in entries["terminal_bench_2_1"]
        entries["terminal_bench_2_1"][name] = task_metadata(path, metadata["category"], tb_root)
    sources = {
        "swe_bench_verified": {
            "registry": "Harbor registry.json@" + harbor_sha,
            "registry_file_sha256": hashlib.sha256(registry_raw).hexdigest(),
            "catalog_version": "1.0",
            "task_repository": "https://github.com/laude-institute/harbor-datasets.git",
            "task_revision": swe_sha,
            "catalog_count": 500,
            "stratum_definition": "task name prefix before '__' (original frozen owner grouping)",
        },
        "terminal_bench_2_1": {
            "task_repository": git(tb_root, "remote", "get-url", "origin").decode().strip(),
            "task_revision": tb_sha,
            "catalog_count": 89,
            "stratum_definition": "task.toml metadata.category; names prefixed terminal-bench/",
        },
    }
    sizes = {"swe_bench_verified": [10, 20, 40, 80, 160, 320, 500], "terminal_bench_2_1": [20, 40, 80, 89]}
    candidates = {}
    for suite in entries:
        seed = cohorts[suite]["seed"]
        ordered, strata = ordering(entries[suite], seed)
        assert len(ordered) == sources[suite]["catalog_count"]
        frozen = cohorts[suite]["rungs"]["nightly"]
        assert ordered[: len(frozen)] == frozen, (suite, "frozen prefix mismatch")
        lists = {str(n): ordered[:n] for n in sizes[suite]}
        for previous, current in zip(sizes[suite], sizes[suite][1:]):
            assert lists[str(current)][:previous] == lists[str(previous)]
        assert set(lists[str(sizes[suite][-1])]) == set(entries[suite])
        candidates[suite] = {
            "candidate_only": True,
            "seed": seed,
            "rule": "Sort strata and task names by SHA256(UTF8(seed + NUL + name)); round-robin one task from each nonempty stratum in that fixed order until exhausted.",
            "source": sources[suite],
            "strata_order": strata,
            "frozen_nightly_prefix": frozen,
            "frozen_prefix_matches": True,
            "candidates": lists,
            "full_catalog_size": sizes[suite][-1],
            "full_catalog_exactly_covered": True,
            "resources_verified_for_all_tasks": len(entries[suite]),
        }
    assert cohorts_path.read_bytes() == before
    (out / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (out / "task_resources.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(
        json.dumps(
            {suite: {"count": len(entries[suite]), "prefix_matches": True, "sizes": sizes[suite]} for suite in entries}
        )
    )


if __name__ == "__main__":
    main()
