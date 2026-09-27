# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write explicit precision policies for the full-model coarse search."""
import copy
import json
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tt.precision_policy import baseline_precision_config

ROOT = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/configs")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    baseline = baseline_precision_config()

    def emit(name, *, layers=None, model=None, inner=None):
        policy = copy.deepcopy(baseline)
        policy["config_id"] = name
        if layers:
            for kind in policy["layer_types"]:
                policy["layer_types"][kind].update(layers)
        if model:
            policy["model"].update(model)
        if inner:
            policy["layer_overrides"] = {str(i): dict(inner) for i in range(1, 29)}
        (ROOT / (name + ".json")).write_text(json.dumps(policy, indent=2) + "\n")

    emit("baseline_mixed_lofi")
    emit(
        "inner_bfp4_lofi",
        inner={"qkv_weight_dtype": "bfloat4_b", "output_weight_dtype": "bfloat4_b", "expert_gate_dtype": "bfloat4_b"},
    )
    emit("inner_expert_gate_bfp4_lofi", inner={"expert_gate_dtype": "bfloat4_b"})
    emit("inner_qkv_bfp4_lofi", inner={"qkv_weight_dtype": "bfloat4_b"})
    emit("inner_output_bfp4_lofi", inner={"output_weight_dtype": "bfloat4_b"})
    emit("shared_down_bfp4_lofi", layers={"shared_down_dtype": "bfloat4_b"})
    emit("shared_down_bfp4_hifi2", layers={"shared_down_dtype": "bfloat4_b", "shared_fidelity": "HiFi2"})
    emit(
        "mixed_hifi2",
        layers={k: "HiFi2" for k in ["qkv_fidelity", "output_fidelity", "expert_fidelity", "shared_fidelity"]},
    )
    emit("head_bfp8_hifi2", model={"head_weight_dtype": "bfloat8_b", "head_fidelity": "HiFi2"})
    emit("head_bfp8_lofi", model={"head_weight_dtype": "bfloat8_b", "head_fidelity": "LoFi"})
    emit("head_bfp4_lofi", model={"head_weight_dtype": "bfloat4_b", "head_fidelity": "LoFi"})
    emit("head_bfp4_hifi2", model={"head_weight_dtype": "bfloat4_b", "head_fidelity": "HiFi2"})
    emit(
        "activation_bfp8",
        layers={"qkv_input_dtype": "bfloat8_b", "output_input_dtype": "bfloat8_b", "shared_input_dtype": "bfloat8_b"},
    )
    emit("ccl_bfp8", layers={"attention_ccl_dtype": "bfloat8_b", "moe_ccl_dtype": "bfloat8_b"})
    emit("kv_bf16", layers={"kv_cache_dtype": "bfloat16"})
    emit(
        "decode_bfp8_hifi2",
        layers={
            "qkv_weight_dtype": "bfloat8_b",
            "output_weight_dtype": "bfloat8_b",
            "expert_gate_dtype": "bfloat8_b",
            "expert_down_dtype": "bfloat8_b",
            "shared_gate_dtype": "bfloat8_b",
            "shared_down_dtype": "bfloat8_b",
            **{k: "HiFi2" for k in ["qkv_fidelity", "output_fidelity", "expert_fidelity", "shared_fidelity"]},
        },
    )
    emit(
        "decode_bfp8_lofi",
        layers={
            "qkv_weight_dtype": "bfloat8_b",
            "output_weight_dtype": "bfloat8_b",
            "expert_gate_dtype": "bfloat8_b",
            "expert_down_dtype": "bfloat8_b",
            "shared_gate_dtype": "bfloat8_b",
            "shared_down_dtype": "bfloat8_b",
        },
    )

    emit(
        "inner_bfp4_hifi2",
        inner={
            "qkv_weight_dtype": "bfloat4_b",
            "output_weight_dtype": "bfloat4_b",
            "expert_gate_dtype": "bfloat4_b",
            "qkv_fidelity": "HiFi2",
            "output_fidelity": "HiFi2",
            "expert_fidelity": "HiFi2",
        },
    )
    emit(
        "decode_bf16_hifi4",
        layers={
            **{
                k: "bfloat16"
                for k in [
                    "qkv_weight_dtype",
                    "output_weight_dtype",
                    "expert_gate_dtype",
                    "expert_down_dtype",
                    "shared_gate_dtype",
                    "shared_down_dtype",
                ]
            },
            **{k: "HiFi4" for k in ["qkv_fidelity", "output_fidelity", "expert_fidelity", "shared_fidelity"]},
        },
    )


if __name__ == "__main__":
    main()
