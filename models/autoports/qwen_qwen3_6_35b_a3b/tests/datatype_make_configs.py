# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Construct explicit, validated coarse precision candidates (no devices)."""

import argparse
import copy
import json
import re
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tt.precision import baseline_policy, resolve_precision

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc/datatype_sweep/configs"


def build_configs(baseline, suffix=""):
    """Keep baseline boundaries/exceptions; vary only the existing candidate axes."""
    baseline = resolve_precision(baseline)
    if any(dtype != "bfp8" for dtype in baseline["weight_groups"].values()):
        raise ValueError("Canonical candidate IDs require BFP8 baseline weight groups; layer exceptions are allowed")
    configs = {}

    def add(cid, groups=(), dtype=None, fidelity=None, inner=False, **switches):
        config = copy.deepcopy(baseline)
        config["config_id"] = cid
        config.update(switches)
        for group in groups:
            if dtype:
                config["weight_groups"][group] = dtype
            if fidelity:
                config["compute_fidelities"][group]["math_fidelity"] = fidelity
        if inner:
            endpoints = dict(
                layers=[0, 39],
                weight_groups={g: baseline["weight_groups"][g] for g in groups},
                compute_fidelities={g: baseline["compute_fidelities"][g] for g in groups},
            )
            if not baseline["layer_exceptions"]:
                config["layer_exceptions"] = [endpoints]
            else:
                # Split existing multi-layer exceptions so changing an endpoint
                # never changes an interior layer or discards unrelated groups.
                by_layer = {}
                for exception in baseline["layer_exceptions"]:
                    for layer in exception["layers"]:
                        by_layer[layer] = dict(
                            layers=[layer],
                            weight_groups=copy.deepcopy(exception["weight_groups"]),
                            compute_fidelities=copy.deepcopy(exception["compute_fidelities"]),
                        )
                for layer in (0, 39):
                    exception = by_layer.setdefault(
                        layer, dict(layers=[layer], weight_groups={}, compute_fidelities={})
                    )
                    for group in groups:
                        exception["weight_groups"].setdefault(group, baseline["weight_groups"][group])
                        exception["compute_fidelities"].setdefault(
                            group, copy.deepcopy(baseline["compute_fidelities"][group])
                        )
                config["layer_exceptions"] = [by_layer[layer] for layer in sorted(by_layer)]
        config = resolve_precision(config)
        configs[cid] = config

    add("baseline_bfp8")
    add("experts_bfp8_hifi2", ("expert_gu", "expert_down", "expert_ep"), fidelity="HiFi2")
    add(
        "projections_bfp8_lofi",
        ("attention_input", "attention_output", "shared_packed", "shared_down"),
        fidelity="LoFi",
    )
    add("head_bfp8_lofi", ("lm_head",), fidelity="LoFi")
    add("attention_bfp8_lofi", ("attention_input", "attention_output"), fidelity="LoFi")
    add("shared_bfp8_lofi", ("shared_packed", "shared_down"), fidelity="LoFi")
    add(
        "accuracy_bf16_dense",
        ("attention_input", "attention_output", "shared_packed", "shared_down", "lm_head"),
        dtype="bf16",
        fidelity="HiFi4",
    )
    for group in ("expert_gu", "expert_down", "expert_ep"):
        configs["accuracy_bf16_dense"]["compute_fidelities"][group]["math_fidelity"] = "HiFi4"
    add("kv_bfp8", kv_cache_dtype="bfp8")
    add("ccl_bfp8", ccl_dtype="bfp8")
    for fidelity in ("LoFi", "HiFi2"):
        tag = fidelity.lower()
        add(
            "inner_experts_bfp4_" + tag,
            ("expert_gu", "expert_down", "expert_ep"),
            dtype="bfp4",
            fidelity=fidelity,
            inner=True,
        )
        add("inner_expert_gu_bfp4_" + tag, ("expert_gu",), dtype="bfp4", fidelity=fidelity, inner=True)
        add("inner_shared_bfp4_" + tag, ("shared_packed", "shared_down"), dtype="bfp4", fidelity=fidelity, inner=True)
    result = {}
    for cid, config in configs.items():
        config["config_id"] = cid + suffix
        result[cid + suffix] = resolve_precision(config)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=OUT, help="Output directory; differing existing files are never replaced"
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="Complete baseline policy accepted by the installed resolver; unknown schemas fail closed",
    )
    parser.add_argument(
        "--version-suffix",
        help="Append _SUFFIX to every candidate ID and filename, e.g. v2; original IDs remain unchanged by default",
    )
    args = parser.parse_args(argv)
    if args.version_suffix is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.version_suffix):
        parser.error(
            "--version-suffix must start with an alphanumeric character and contain only letters, digits, _ or -"
        )
    baseline = json.loads(args.baseline_json.read_text()) if args.baseline_json else baseline_policy()
    configs = build_configs(baseline, "_" + args.version_suffix if args.version_suffix else "")
    output_dir = args.output_dir.resolve()
    pending = {}
    for cid, config in configs.items():
        path = output_dir / (cid + ".json")
        try:
            build_path = path.relative_to(Path.cwd())
        except ValueError:
            build_path = path
        pending[path] = (json.dumps(config, indent=2) + "\n").encode()
        pending[output_dir / (cid + ".build.json")] = (
            json.dumps(dict(precision_config=str(build_path), max_seq_len=2048), indent=2) + "\n"
        ).encode()
    # Preflight the entire batch before creating anything. Preserve even the
    # timestamps of identical files; a changed policy needs a new ID/directory.
    conflicts = [str(path) for path, data in pending.items() if path.exists() and path.read_bytes() != data]
    if conflicts:
        parser.error(
            "Refusing to replace differing files; use --version-suffix or a new --output-dir:\n" + "\n".join(conflicts)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for path, data in pending.items():
        if path.exists():
            if path.read_bytes() != data:
                raise FileExistsError(f"Output changed after preflight: {path}")
            continue
        with path.open("xb") as stream:
            stream.write(data)
        created += 1
    print("Validated", len(configs), "candidate policies; created", created, "files; preserved", len(pending) - created)


if __name__ == "__main__":
    main()
