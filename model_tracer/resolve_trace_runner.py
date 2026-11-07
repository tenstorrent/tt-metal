#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve the runner pool and weights cache for a model trace.

The trace runs on the same runner pools and weights mounts as the model e2e pipeline
(``.github/sku_config.yaml``), so a model finds its weights wherever CI already runs it.
Shared caches are read-only and offline; per-runner caches download from HF on first use.
"""

import argparse
import json
import os
import sys

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runner-label", required=True)
    parser.add_argument("--sku-config", default=".github/sku_config.yaml")
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
