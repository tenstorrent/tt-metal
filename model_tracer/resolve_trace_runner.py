#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve the runner pool and weights cache for a model trace, and check weights coverage.

The trace runs on the same runner pools and weights mounts as the model e2e pipeline
(``.github/sku_config.yaml``), so a model finds its weights wherever CI already runs it.
Shared caches are read-only and offline, so a model only has weights there if some CI test
uses it on a SKU with the same mount; per-runner caches download from HF on first use.

The coverage check is advisory: it reads CI test YAML, which can miss how a model reaches
its weights, so it only warns. The trace step's check against the mounted cache decides.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

# Old runner-label choices that still name the same pool as their SKU.
LABEL_ALIASES = {
    "topology-6u": "wh_galaxy",
    "BH-LoudBox": "bh_loudbox",
}

# Old runner-label choices that used to pick the CIv2 pool, while the same-named SKU is CIv1.
# Mapping them either way would silently change pools, so the caller picks the SKU.
RETIRED_LABELS = {
    "N150": ("wh_n150", "wh_n150_civ2"),
    "N300": ("wh_n300", "wh_n300_civ2"),
    "P100a": ("bh_p100", "bh_p100a_civ2_viommu"),
    "P150b": ("bh_p150", "bh_p150b_civ2"),
    "n300-llmbox": ("wh_llmbox", "wh_llmbox_civ2_viommu"),
}

ONLINE_MODES = ("local-disk", "lfc")

_HF_MODEL_RE = re.compile(r"""\bHF_MODEL[ \t]*[=:][ \t]*["']?([A-Za-z0-9_.\-/]+)""")
_HUB_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def is_hub_id(model, repo_root="."):
    """True for an HF hub id (org/name); False for a local path, absolute or repo-relative."""
    return bool(_HUB_ID_RE.match(model)) and not (Path(repo_root) / model).is_dir()


def resolve_sku(label, skus):
    if label in RETIRED_LABELS:
        civ1, civ2 = RETIRED_LABELS[label]
        raise ValueError(
            f"runner-label {label!r} is no longer accepted; pick a SKU: {civ1} (CIv1, shared /mnt/MLPerf "
            f"weights) or {civ2} (CIv2)"
        )
    sku = LABEL_ALIASES.get(label, label)
    if sku not in skus:
        known = ", ".join(sorted(set(LABEL_ALIASES) | set(skus)))
        raise ValueError(f"unknown runner-label {label!r}; use a SKU from .github/sku_config.yaml or one of: {known}")
    entry = skus[sku]
    runs_on = entry.get("runs_on")
    if not runs_on:
        raise ValueError(f"SKU {sku} has no runs_on in .github/sku_config.yaml")
    # Same rule as .github/scripts/utils/prepare_test_matrix.py. Multihost pools need the worker
    # setup from models-e2e-tests-multihost-impl.yaml, which the trace job does not do.
    if "allocation" in entry or any(
        isinstance(runner_tag, str) and "exabox-multihost" in runner_tag for runner_tag in runs_on
    ):
        raise ValueError(f"SKU {sku} is a multihost pool; model tracing supports single-host SKUs only")
    return sku, entry


def arch_for(sku, runs_on):
    if sku.startswith("bh_") or "arch-blackhole" in runs_on:
        return "blackhole"
    if sku.startswith("wh_") or "arch-wormhole_b0" in runs_on:
        return "wormhole_b0"
    raise ValueError(f"cannot tell the arch of SKU {sku}")


def weights_mount(mode):
    """Host -> container mount and HF offline flag, mirroring models-e2e-tests-impl.yaml.

    Weights always appear at /mnt/MLPerf/huggingface inside the container. Shared mounts stay
    offline because HF writes metadata on every lookup, which fails on a read-only mount.
    """
    if mode == "cloud-mlperf":
        return "/mnt/MLPerf/huggingface:/mnt/MLPerf/huggingface:ro", "1"
    if mode == "yyz4-mnt-models":
        return "/mnt/models/huggingface:/mnt/MLPerf/huggingface:ro", "1"
    if mode in ONLINE_MODES:
        return "/localdev/blackhole_demos/huggingface_data:/mnt/MLPerf/huggingface:rw", "0"
    if mode:
        raise ValueError(f"unknown weights-cache-mode {mode!r}")
    return "/mnt/MLPerf:/mnt/MLPerf:ro", "1"


def _hf_models_in(node):
    """HF_MODEL values set anywhere in a test entry: shell assignments in strings, env keys."""
    found = set()
    if isinstance(node, str):
        found.update(_HF_MODEL_RE.findall(node))
    elif isinstance(node, dict):
        for key, value in node.items():
            if key in ("HF_MODEL", "hf_model") and isinstance(value, str):
                found.add(value.strip())
            else:
                found |= _hf_models_in(value)
    elif isinstance(node, list):
        for item in node:
            found |= _hf_models_in(item)
    # Template placeholders and shell expansions are not model names; real ones are an
    # org/name hub id or a path, so both contain a slash.
    return {m for m in found if "/" in m and m.strip("/") and re.fullmatch(r"[A-Za-z0-9_.\-/]+", m)}


def hf_model_coverage(test_yaml_dir):
    """Map each HF_MODEL named by a CI test to the SKUs that run it."""
    coverage = {}
    for path in sorted(Path(test_yaml_dir).glob("*.yaml")):
        try:
            entries = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("skus"), dict):
                continue
            shared = {k: v for k, v in entry.items() if k != "skus"}
            for model in _hf_models_in(shared):
                coverage.setdefault(model, set()).update(entry["skus"])
            # A per-SKU block can pick its own model, which only that SKU runs.
            for sku, sku_entry in entry["skus"].items():
                for model in _hf_models_in(sku_entry):
                    coverage.setdefault(model, set()).add(sku)
    return coverage


def check_models(hf_models, sku, skus, coverage, repo_root="."):
    """Return warnings for hub models whose weights the SKU's cache likely lacks."""
    mode = skus[sku].get("weights-cache-mode", "")
    if mode in ONLINE_MODES:
        return []
    mount = weights_mount(mode)[0]
    online = sorted(s for s, e in skus.items() if e.get("weights-cache-mode") in ONLINE_MODES)
    warnings = []
    for model in hf_models:
        if not model or not is_hub_id(model, repo_root):
            continue
        covered = coverage.get(model, set())
        if sku in covered:
            continue
        same_mount = sorted(
            s for s in covered if s in skus and weights_mount(skus[s].get("weights-cache-mode", ""))[0] == mount
        )
        if same_mount:
            warnings.append(
                f"{model} has no CI test on {sku}, but {', '.join(same_mount)} share its weights mount; "
                "assuming the weights are there"
            )
            continue
        where = f"CI runs it on: {', '.join(sorted(covered))}" if covered else "no CI test uses it"
        warnings.append(
            f"{model} is likely not in the {sku} weights cache (read-only, offline; {where}). "
            f"Use one of those SKUs, or a SKU that downloads from HF: {', '.join(online)}"
        )
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runner-label", required=True)
    parser.add_argument("--matrix", default='{"include": []}', help="trace matrix JSON (entries with hf_model)")
    parser.add_argument("--sku-config", default=".github/sku_config.yaml")
    parser.add_argument("--test-yaml-dir", default="tests/pipeline_reorg")
    args = parser.parse_args()

    with open(args.sku_config) as f:
        skus = yaml.safe_load(f)["skus"]
    try:
        sku, entry = resolve_sku(args.runner_label, skus)
        mode = entry.get("weights-cache-mode", "")
        hf_mount, hf_offline = weights_mount(mode)
        arch = arch_for(sku, entry["runs_on"])
    except ValueError as e:
        print(f"::error::{e}")
        return 1

    hf_models = [item.get("hf_model", "") for item in json.loads(args.matrix).get("include", [])]
    for w in check_models(hf_models, sku, skus, hf_model_coverage(args.test_yaml_dir)):
        print(f"::warning::{w}")

    outputs = {
        "arch": arch,
        "platform": "Ubuntu 22.04",
        "sku": sku,
        "runs_on": json.dumps(entry["runs_on"]),
        "weights_cache_mode": mode or "mlperf",
        "hf_mount": hf_mount,
        "hf_offline": hf_offline,
    }
    print(json.dumps(outputs, indent=2))
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            for k, v in outputs.items():
                f.write(f"{k}={v}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
