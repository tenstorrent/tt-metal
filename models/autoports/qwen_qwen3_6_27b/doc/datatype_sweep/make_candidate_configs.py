"""Generate the checked-in coarse precision matrix from the safe baseline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tt.precision_config import load_precision_config, safe_baseline_config

ROOT = Path(__file__).resolve().parent / "candidates"


def candidate(config_id, *, weights=None, fidelities=None, kv=None, ccl=None, overrides=None, activation=None):
    value = copy.deepcopy(safe_baseline_config())
    value["config_id"] = config_id
    value["weight_groups"].update(weights or {})
    value["compute_fidelities"].update(fidelities or {})
    if kv:
        value["kv_cache_dtype"] = kv
    if ccl:
        value["ccl_dtype"].update(ccl)
    if activation:
        value["activation_residual_dtype"] = activation
    if overrides:
        value["layer_exceptions"] = [{"layers": list(range(64)), "layer_kind": "any", "overrides": overrides}]
    load_precision_config(value)
    return value


CANDIDATES = [
    candidate("baseline_optimized_default"),
    candidate("baseline_bf16_kv", kv="BF16"),
    candidate(
        "full_attention_bfp8_hifi2",
        weights={"full_attention_qkv": "BFP8_B", "full_attention_o": "BFP8_B"},
        fidelities={"full_attention_sdpa": "HiFi2", "full_attention_qkv": "HiFi2", "full_attention_o": "HiFi2"},
    ),
    candidate(
        "full_attention_bfp8_lofi",
        weights={"full_attention_qkv": "BFP8_B", "full_attention_o": "BFP8_B"},
        fidelities={"full_attention_sdpa": "LoFi", "full_attention_qkv": "LoFi", "full_attention_o": "LoFi"},
    ),
    candidate(
        "full_attention_bfp4_hifi2",
        weights={"full_attention_qkv": "BFP4_B", "full_attention_o": "BFP4_B"},
        fidelities={"full_attention_sdpa": "HiFi2", "full_attention_qkv": "HiFi2", "full_attention_o": "HiFi2"},
    ),
    candidate(
        "full_attention_bfp4_lofi",
        weights={"full_attention_qkv": "BFP4_B", "full_attention_o": "BFP4_B"},
        fidelities={"full_attention_sdpa": "LoFi", "full_attention_qkv": "LoFi", "full_attention_o": "LoFi"},
    ),
    candidate(
        "all_projection_bfp8_hifi2",
        weights={
            "full_attention_qkv": "BFP8_B",
            "full_attention_o": "BFP8_B",
            "linear_attention_internal": "BFP8_B",
            "linear_input_projection": "BFP8_B",
            "linear_output_projection": "BFP8_B",
            "mlp_gate_up": "BFP8_B",
            "mlp_down": "BFP8_B",
        },
        fidelities={
            "full_attention_sdpa": "HiFi2",
            "full_attention_qkv": "HiFi2",
            "full_attention_o": "HiFi2",
            "linear_attention_internal": "HiFi2",
            "linear_input_projection": "HiFi2",
            "linear_output_projection": "HiFi2",
            "mlp": "HiFi2",
        },
        overrides={"mlp_down_in0_block_w": 1},
    ),
    candidate(
        "all_projection_bfp8_lofi",
        weights={
            "full_attention_qkv": "BFP8_B",
            "full_attention_o": "BFP8_B",
            "linear_attention_internal": "BFP8_B",
            "linear_input_projection": "BFP8_B",
            "linear_output_projection": "BFP8_B",
            "mlp_gate_up": "BFP8_B",
            "mlp_down": "BFP8_B",
        },
        fidelities={
            "full_attention_sdpa": "LoFi",
            "full_attention_qkv": "LoFi",
            "full_attention_o": "LoFi",
            "linear_attention_internal": "LoFi",
            "linear_input_projection": "LoFi",
            "linear_output_projection": "LoFi",
            "mlp": "LoFi",
        },
        overrides={"mlp_down_in0_block_w": 1},
    ),
    candidate("baseline_bfp8_ccl", ccl={"token_mixer": "BFP8_B", "mlp": "BFP8_B"}),
    candidate(
        "selected_bfp4_mlp_hifi2",
        weights={"full_attention_qkv": "BFP4_B", "full_attention_o": "BFP4_B"},
        fidelities={
            "full_attention_sdpa": "LoFi",
            "full_attention_qkv": "LoFi",
            "full_attention_o": "LoFi",
            "mlp": "HiFi2",
        },
    ),
    candidate(
        "selected_bfp4_linear_hifi2",
        weights={"full_attention_qkv": "BFP4_B", "full_attention_o": "BFP4_B"},
        fidelities={
            "full_attention_sdpa": "LoFi",
            "full_attention_qkv": "LoFi",
            "full_attention_o": "LoFi",
            "linear_attention_internal": "HiFi2",
            "linear_input_projection": "HiFi2",
            "linear_output_projection": "HiFi2",
        },
    ),
    candidate(
        "selected_bfp8_activation_ccl",
        activation="BFP8_B",
        ccl={"token_mixer": "BFP8_B", "mlp": "BFP8_B"},
    ),
]


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    for value in CANDIDATES:
        (ROOT / f"{value['config_id']}.json").write_text(json.dumps(value, indent=2) + "\n")
    print(f"wrote {len(CANDIDATES)} validated candidate configs to {ROOT}")
