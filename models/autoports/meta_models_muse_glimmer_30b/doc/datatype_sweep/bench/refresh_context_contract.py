# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild ``doc/context_contract.json`` for the datatype-sweep stage.

The full-model stage's builder, loaded by path, pointed at this stage's evidence
and stamped with this stage's name -- so the contract's byte budget, capacity and
performance rows are read off *this* stage's selected-config runs rather than
inherited.

The stage-specific part is the KV-cache dtype.  Cache dtype is the one precision
choice that moves memory capacity, so the contract gains a
``kv_cache_dtype_capacity`` block built from the sweep's own per-candidate
capability reports: for every KV-cache dtype evaluated, the measured per-device
cache bytes, the resulting long-lived footprint, and how many full-context
sequences fit.  That is what makes "the selected cache dtype does not reduce the
advertised context" a measurement instead of an assertion.

Usage::

    python doc/datatype_sweep/bench/refresh_context_contract.py
    python doc/datatype_sweep/bench/refresh_context_contract.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

STAGE = "datatype_sweep"
#: The stage this one demotes.  Named rather than guessed, so a re-run restores
#: the right block instead of the parent's hardcoded one.
PREVIOUS_STAGE = "optimized_full_model"
EVIDENCE = ROOT / "doc/datatype_sweep"
CONTRACT = ROOT / "doc/context_contract.json"


def _parent():
    path = ROOT / "doc/full_model/bench/refresh_context_contract.py"
    spec = importlib.util.spec_from_file_location("muse_glimmer_full_model_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EVIDENCE = EVIDENCE
    module.STAGE = STAGE
    return module


def kv_cache_dtype_capacity() -> dict:
    """Per-KV-dtype memory capacity, from the sweep's own capability reports.

    Cache dtype is the one precision choice that moves memory capacity, so it is
    priced here per dtype rather than only for the selection.  Two things this
    block is careful about:

    * only configs that actually ran contribute -- a candidate that died on an op
      contract has no capability report and must not appear as a measurement;
    * the per-device *cache* bytes depend only on the cache dtype, but the
      long-lived *total* also contains the weights, which several candidates
      change. So the totals are reported as the measured range across the configs
      at that dtype, with the selected config's own figures given separately from
      this stage's evidence run.
    """
    results_path = EVIDENCE / "sweep_results.json"
    if not results_path.is_file():
        return {}
    results = json.loads(results_path.read_text())
    selected = results["selected"]["config_id"]
    by_dtype: dict[str, dict] = {}
    for row in results["configs"]:
        memory = row.get("memory") or {}
        if row.get("status") != "ok" or memory.get("per_device_kv_cache_bytes") is None:
            continue
        # Key on the *measured* cache bytes rather than on the policy's
        # ``kv_cache_dtype`` field.  A layer-scoped cache (BFP4 on the 39 sliding
        # layers, BFP8 on the 13 full-attention ones) has one dtype in the policy
        # field and a footprint that is neither dtype's, so keying on the field
        # would file it under a heading whose byte figure is wrong for it.
        # Grouping by the number the model reports cannot mis-group.
        # Label by the dtype that is *scoped*, not by the policy default: `c20`
        # is BFP4 on the sliding layers with a BFP8 exception and `c22` is the
        # mirror, so keying on the default would give the two opposite policies
        # labels that each name the other's minority dtype.
        exceptions = json.loads(row["policy"]["layer_exceptions"] or "[]")
        scoped_dtypes = {entry["kv_cache_dtype"] for entry in exceptions if "kv_cache_dtype" in entry}
        default_dtype = row["policy"]["kv_cache_dtype"]
        if scoped_dtypes:
            minority = sorted(scoped_dtypes)[0]
            label = f"{default_dtype}-with-{minority}-exceptions"
        else:
            label = default_dtype
        entry = by_dtype.setdefault(
            label,
            {
                "per_device_kv_cache_bytes": memory["per_device_kv_cache_bytes"],
                "per_device_dram_capacity_bytes": memory["per_device_dram_capacity_bytes"],
                "supported_context": memory["supported_context"],
                "per_device_total_long_lived_bytes_range": [],
                "measured_on_configs": [],
                "top1_range": [],
                "selected": False,
            },
        )
        if entry["per_device_kv_cache_bytes"] != memory["per_device_kv_cache_bytes"]:
            raise SystemExit(
                f"two configs share the KV label {label!r} but report different cache bytes: "
                f"{entry['per_device_kv_cache_bytes']} and {memory['per_device_kv_cache_bytes']}"
            )
        entry["measured_on_configs"].append(row["config_id"])
        entry["per_device_total_long_lived_bytes_range"].append(memory["per_device_total_bytes"])
        entry["top1_range"].append(row["accuracy"]["top1"])
        if row["config_id"] == selected:
            entry["selected"] = True
    for entry in by_dtype.values():
        totals = entry.pop("per_device_total_long_lived_bytes_range")
        entry["per_device_total_long_lived_bytes_min"] = min(totals)
        entry["per_device_total_long_lived_bytes_max"] = max(totals)
        tops = [t for t in entry.pop("top1_range") if t is not None]
        entry["full_model_top1_min"] = min(tops) if tops else None
        entry["full_model_top1_max"] = max(tops) if tops else None
        # Headroom at the advertised context, computed against the *largest*
        # long-lived total measured at this dtype, which is the conservative end.
        cache = entry["per_device_kv_cache_bytes"]
        other = entry["per_device_total_long_lived_bytes_max"] - cache
        entry["full_context_sequences_that_fit"] = int(
            (entry["per_device_dram_capacity_bytes"] - other) // max(cache, 1)
        )

    accuracy = json.loads((EVIDENCE / "evidence_accuracy.json").read_text())
    capacity = accuracy["capacity"]
    return {
        "note": (
            "Every KV-cache dtype the sweep evaluated and that built, with the per-device bytes read "
            "off each built model's capability_report rather than a formula. The advertised "
            "131072-token context is supported at every one of them: a lower cache dtype buys "
            "headroom (more concurrent full-context sequences), it does not buy context this model "
            "was not already advertising, and no candidate reduced it. BFP4 was measured and "
            "rejected on full-model top-1, not on capacity -- and the loss is concentrated in the 13 "
            "full-attention layers rather than spread evenly: 0.990 with BFP8 everywhere, 0.980 with "
            "BFP4 on the 39 sliding layers only (c20), 0.970 with BFP4 on the 13 full-attention layers "
            "only (c22), and 0.970 with BFP4 on all 52 (c05/c08/c09). The capacity saving runs the "
            "other way, being per layer: c20 holds 1.200 GB/device against c22's 1.636 GB, so c20 is "
            "the efficient trade for a serving stage that wants cache headroom."
        ),
        "advertised_context": 131072,
        "selected_config_id": selected,
        "selected": {
            "kv_cache_dtype": "bfloat8_b",
            "per_device_kv_cache_bytes": capacity["per_device_kv_cache_bytes"],
            "per_device_kv_cache_bytes_per_block": capacity["per_device_kv_cache_bytes_per_block"],
            "per_device_total_long_lived_bytes": capacity["per_device_total_bytes"],
            "per_device_free_after_long_lived_bytes": capacity["per_device_free_after_long_lived_bytes"],
            "full_context_sequences_that_fit": capacity["full_context_sequences_that_fit"],
            "supported_context": capacity["supported_context"],
            "source": "doc/datatype_sweep/evidence_accuracy.json:capacity",
        },
        "by_dtype": by_dtype,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    module = _parent()
    previous = json.loads(CONTRACT.read_text())
    if previous.get("stage") == STAGE:
        # A re-run: rebuild from the block this stage demoted, not from itself.
        nested = {key: value for key, value in previous.items() if isinstance(value, dict) and "stage" in value}
        if PREVIOUS_STAGE not in nested:
            raise SystemExit(
                f"{CONTRACT} is already the {STAGE} top level but has no {PREVIOUS_STAGE!r} block to restore"
            )
        restored = dict(nested[PREVIOUS_STAGE])
        for key, value in nested.items():
            if key != PREVIOUS_STAGE:
                restored[key] = value
        previous = restored

    contract = module.build(previous)

    stage_dir = "doc/datatype_sweep"
    if "performance" in contract:
        contract["performance"]["source"] = f"{stage_dir}/evidence_perf.json"
        # The parent copies the layer-stack floor straight through, and
        # ``evidence_perf.json`` flags it as measured under a *different* precision
        # policy.  Dropping that flag here would leave the artifact downstream
        # stages read asserting a floor for a policy it was not measured under.
        floor = contract["performance"].get("layer_stack_lower_bound_ms_per_token")
        perf = json.loads((EVIDENCE / "evidence_perf.json").read_text())["performance"]
        flagged = perf.get("layer_stack_lower_bound_ms_per_token") or {}
        if floor is not None and not flagged.get("measured_under_the_selected_policy", True):
            contract["performance"]["layer_stack_lower_bound_ms_per_token"] = None
            contract["performance"]["layer_stack_lower_bound_note"] = (
                f"withheld: the inherited figure ({floor} ms/token) was measured under the previous "
                "stage's precision policy, which the selection changed. The selected policy's own "
                "per-layer floor is 39 x 0.4345 + 13 x 0.4032 = 22.187 ms/token at decode context 256 "
                f"({stage_dir}/logs/layer_ab.log, layer_ab.py --candidates baseline,selected), against "
                "the previous policy's 22.421 ms at the same context, and the measured decode trace of "
                "22.434 ms sits 1.11 % above it."
            )
        contract["performance"]["measurement_regime"] = (
            "warmed token-out, batch 1, prompt 128 / generate 128, end to end through the public "
            "generator with device-side token feedback and one 32-uint32 token readback per step -- "
            "the same benchmark the optimized full model reported, re-run on the selected precision "
            "config through the default construction path"
        )
    budget = contract.get("byte_budget_at_full_context")
    if isinstance(budget, dict):
        if "measured_from" in budget:
            budget["measured_from"] = f"{stage_dir}/evidence_accuracy.json (capability_report over the built model)"
        # The parent's ``note`` restates the long-lived total in prose, and this
        # stage moves it (BFP4 attention weights are 576 MB/device lighter). A
        # recomputed contract whose note contradicts its own numbers is worse than
        # no note, so the figure is rewritten from the measured value rather than
        # carried over. Round 1 of the stage review caught it.
        note = budget.get("note")
        total = budget.get("per_device_total_bytes")
        if isinstance(note, str) and isinstance(total, int):
            budget["note"] = re.sub(
                r"needs \*\*[0-9.]+ GB\*\*/device",
                f"needs **{total / 1e9:.2f} GB**/device",
                re.sub(r"needs [0-9.]+ GB/device", f"needs {total / 1e9:.2f} GB/device", note),
            )
            if f"{total / 1e9:.2f} GB" not in budget["note"]:
                raise SystemExit(
                    "byte_budget_at_full_context.note does not contain a '<n> GB/device' figure this "
                    f"script knows how to rewrite; it reads: {note!r}"
                )

    accuracy = json.loads((EVIDENCE / "evidence_accuracy.json").read_text())
    selected_id = accuracy.get("capacity", {}).get("precision_config_id")
    contract["precision_policy"] = {
        "selected_config_id": selected_id,
        "artifact": f"{stage_dir}/selected_precision_config.json",
        "artifact_is_required_by_the_build": (
            "tt/generator.py::build_generator reads it on every build; a missing or malformed file "
            "raises rather than falling back to a module constant"
        ),
        "realised_policy_source": f"{stage_dir}/evidence_accuracy.json:capacity.precision_policy",
        "sweep_results": f"{stage_dir}/sweep_results.json",
    }
    contract["kv_cache_dtype_capacity"] = kv_cache_dtype_capacity()

    tested = contract.get("tested")
    if isinstance(tested, dict):
        tested["commands"] = [
            f"python {stage_dir}/bench/candidates.py",
            f"python {stage_dir}/bench/smoketest.py",
            f"bash {stage_dir}/bench/run_sweep.sh",
            f"python {stage_dir}/bench/analyse.py",
            f"python {stage_dir}/bench/evidence.py --stages capacity,prefill,teacher,shapes,sampling,fallback "
            "--out evidence_accuracy.json",
            f"python {stage_dir}/bench/evidence.py --stages perf --out evidence_perf.json",
            "python doc/full_model/bench/qualitative.py --arm hf "
            "(run by the full-model stage; this stage reuses its output via --reuse-hf-control)",
            f"python {stage_dir}/bench/qualitative.py --arm tt --reuse-hf-control",
            f"python {stage_dir}/bench/qualitative.py --arm compare",
            f"python {stage_dir}/bench/qualitative.py --vs-optimized-full-model",
            "pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_precision_config.py",
            "pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_full_model.py",
        ]
        misses = tested.get("prefill_misses")
        if isinstance(misses, dict) and "note" in misses:
            misses["note"] = (
                f"not re-collected by this stage; the selected config's prefill top-1/top-5/top-100 are in "
                f"{stage_dir}/evidence_accuracy.json and its per-candidate values in {stage_dir}/sweep_results.json"
            )

    notes = contract.get("notes", "")
    notes = notes.replace(
        "The optimized-full-model stage takes no capability reduction",
        "The datatype-sweep stage takes no capability reduction",
    ).replace(
        "The full-model stage takes no capability reduction",
        "The datatype-sweep stage takes no capability reduction",
    )
    for wrong in ("optimized_multichip_decoder", "full_model", "optimized_full_model"):
        notes = notes.replace(
            f"the previous top level is now the '{wrong}' block.",
            f"the previous top level is now the '{PREVIOUS_STAGE}' block.",
        )
    notes += (
        " The selected precision policy is a required build input "
        "(doc/datatype_sweep/selected_precision_config.json), and the KV-cache dtype -- the one "
        "precision choice that moves memory capacity -- is priced per dtype in "
        "kv_cache_dtype_capacity from the sweep's own capability reports."
    )
    contract["notes"] = notes

    rendered = json.dumps(contract, indent=2) + "\n"
    if args.check:
        if CONTRACT.read_text() != rendered:
            print(f"{CONTRACT} is stale; run this script without --check", file=sys.stderr)
            return 1
        print(f"{CONTRACT} is up to date")
        return 0
    CONTRACT.write_text(rendered)
    print(f"wrote {CONTRACT}: supported_context={contract['current_supported_context']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
