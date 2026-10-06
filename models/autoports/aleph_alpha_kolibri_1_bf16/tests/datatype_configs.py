# SPDX-License-Identifier: Apache-2.0
"""Generate reproducible precision candidates without opening a device."""

import copy
import json
from pathlib import Path

from ..tt.precision import baseline_config


def main():
    root = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/configs"
    root.mkdir(parents=True, exist_ok=True)
    baseline = baseline_config()
    candidates = []

    def add(name, mesh=None, runtime=None, exceptions=None):
        value = copy.deepcopy(baseline)
        value["config_id"] = name
        value["mesh_policy"].update(mesh or {})
        value["runtime"].update(runtime or {})
        value["layer_exceptions"] = exceptions or {}
        (root / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
        candidates.append(name)

    add("baseline_bfp4_lofi")
    add(
        "bfp4_hifi2",
        {
            k: "HiFi2"
            for k in (
                "attention_fidelity",
                "shared_fidelity",
                "expert_fidelity",
                "prefill_expert_fidelity",
                "router_fidelity",
            )
        },
    )
    for fidelity in ("LoFi", "HiFi2"):
        add(
            "dense_bfp8_" + fidelity.lower(),
            dict(
                attention_dtype="bfloat8_b",
                shared_dtype="bfloat8_b",
                attention_fidelity=fidelity,
                shared_fidelity=fidelity,
            ),
        )
        add("expert_down_bfp8_" + fidelity.lower(), dict(expert_down_dtype="bfloat8_b", expert_fidelity=fidelity))
        add("expert_gate_up_bfp8_" + fidelity.lower(), dict(expert_gate_up_dtype="bfloat8_b", expert_fidelity=fidelity))
        add(
            "expert_gate_up_bfp8_" + fidelity.lower() + "_chunk2048",
            dict(expert_gate_up_dtype="bfloat8_b", expert_fidelity=fidelity, prefill_chunk_size=2048),
        )
        add("head_bfp8_" + fidelity.lower(), runtime=dict(head_weight_dtype="bfloat8_b", head_fidelity=fidelity))
        add("head_bfp4_" + fidelity.lower(), runtime=dict(head_weight_dtype="bfloat4_b", head_fidelity=fidelity))
    for dtype in ("bfloat8_b", "bfloat4_b"):
        for fidelity in ("LoFi", "HiFi2"):
            short = "bfp8" if dtype == "bfloat8_b" else "bfp4"
            add(
                "head_" + short + "_" + fidelity.lower() + "_logits_bf16",
                runtime=dict(
                    head_weight_dtype=dtype,
                    head_fidelity=fidelity,
                    head_output_dtype="bfloat16",
                    sampling_logits_dtype="bfloat16",
                ),
            )
    # Full-model head/cache interactions: both reduced weight formats, both
    # fidelities, both supported logits formats, and all three cache formats.
    # KV8 rows already exist above; KV4/KV16 rows complete the finite matrix.
    for dtype in ("bfloat8_b", "bfloat4_b"):
        for fidelity in ("LoFi", "HiFi2"):
            for logits in ("float32", "bfloat16"):
                for cache in ("bfloat4_b", "bfloat16"):
                    short = "bfp8" if dtype == "bfloat8_b" else "bfp4"
                    name = "head_" + short + "_" + fidelity.lower()
                    if logits == "bfloat16":
                        name += "_logits_bf16"
                    name += "_kv_" + ("bfp4" if cache == "bfloat4_b" else "bf16")
                    add(
                        name,
                        runtime=dict(
                            head_weight_dtype=dtype,
                            head_fidelity=fidelity,
                            head_output_dtype=logits,
                            sampling_logits_dtype=logits,
                            kv_cache_dtype=cache,
                        ),
                    )
    add("head_bf16_hifi4", runtime=dict(head_weight_dtype="bfloat16"))
    add("ccl_bfp8", dict(ccl_dtype="bfloat8_b"))
    add("projection_activation_bfp8", dict(projection_activation_dtype="bfloat8_b"))
    add("expert_activation_bfp8", dict(expert_activation_dtype="bfloat8_b"))
    add("residual_bfp8", runtime=dict(residual_dtype="bfloat8_b"))
    add("kv_bfp4", runtime=dict(kv_cache_dtype="bfloat4_b"))
    add("kv_bf16", runtime=dict(kv_cache_dtype="bfloat16"))
    add("router_bfp8_lofi", dict(router_dtype="bfloat8_b", router_fidelity="LoFi"))
    add("router_bfp8_hifi2", dict(router_dtype="bfloat8_b", router_fidelity="HiFi2"))
    # Canonical accuracy-family approximation: retain BFP4 expert storage
    # because complete BFP8 duplicate expert banks exceed physical DRAM.
    add(
        "accuracy_dense_bf16_hifi4",
        dict(
            attention_dtype="bfloat16",
            shared_dtype="bfloat16",
            attention_fidelity="HiFi4",
            shared_fidelity="HiFi4",
            router_dtype="bfloat16",
            router_fidelity="HiFi4",
            expert_fidelity="HiFi2",
            prefill_expert_fidelity="HiFi2",
        ),
        runtime=dict(kv_cache_dtype="bfloat16"),
    )
    priority = [
        "baseline_bfp4_lofi",
        "head_bfp8_lofi",
        "head_bfp8_hifi2",
        "head_bfp4_lofi",
        "head_bfp4_hifi2",
        "head_bf16_hifi4",
        "ccl_bfp8",
        "kv_bfp4",
        "kv_bf16",
        "residual_bfp8",
        "projection_activation_bfp8",
        "expert_activation_bfp8",
    ]
    candidates = priority + [name for name in candidates if name not in priority]
    (root / "matrix.json").write_text(json.dumps(candidates, indent=2) + "\n")


if __name__ == "__main__":
    main()
