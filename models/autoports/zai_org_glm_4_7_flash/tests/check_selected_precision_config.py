# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Propagation check for doc/datatype_sweep/selected_precision_config.json.

Proves two independent things, without opening a device or building the
47-layer model again:

1. The *live source defaults* (`GLM47FlashModel.from_pretrained`'s keyword
   defaults and `SharedRopeDecoder`'s class attributes -- the values
   `build_generator()` actually uses when nothing overrides them) match the
   `construction` block of `selected_precision_config.json` field-for-field.
2. Those same source defaults match the `policy_snapshot` recorded in
   `doc/datatype_sweep/runs/C00_baseline.json`, which was introspected from a
   real, once-built 47-layer model's actual ttnn tensors and compute-kernel
   configs (not from requested kwargs). This is the "a JSON field ignored by
   hard-coded model code does not satisfy this requirement" check from
   `$datatype-sweep`.

    python -m models.autoports.zai_org_glm_4_7_flash.tests.check_selected_precision_config
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = MODEL_DIR / "doc" / "datatype_sweep" / "selected_precision_config.json"
BASELINE_RUN_PATH = MODEL_DIR / "doc" / "datatype_sweep" / "runs" / "C00_baseline.json"


def _dtype_short(name: str) -> str:
    # "bfloat4_b (default)" -> "bfloat4_b"; "DataType.BFLOAT4_B" -> "bfloat4_b"
    token = name.split(" ")[0]
    if token.startswith("DataType."):
        token = token.split(".")[-1].lower()
    return token


def _fidelity_key_from_prose(text: str) -> str:
    """'HiFi2', 'LoFi', 'HiFi2+fp32acc', 'HiFi4+fp32acc' (optionally with a
    trailing ' (decode) / ...' clause already stripped by the caller) -> the
    FIDELITY preset key used by tt/optimized_decoder.py."""
    first = text.split(" ")[0]
    table = {"lofi": "lofi", "hifi2": "hifi2", "hifi2+fp32acc": "hifi2_fp32", "hifi4+fp32acc": "hifi4_fp32"}
    return table[first.lower()]


def _fidelity_key_from_snapshot(fid: dict) -> str:
    mf = fid["math_fidelity"].split(".")[-1]
    fp32 = fid["fp32_dest_acc_en"]
    table = {
        ("LoFi", False): "lofi",
        ("HiFi2", False): "hifi2",
        ("HiFi2", True): "hifi2_fp32",
        ("HiFi4", True): "hifi4_fp32",
    }
    return table[(mf, fp32)]


def check_source_defaults(config: dict) -> list[str]:
    from models.autoports.zai_org_glm_4_7_flash.tt.model import GLM47FlashModel, SharedRopeDecoder

    errors = []
    sig = inspect.signature(GLM47FlashModel.from_pretrained)
    live_kwargs = {
        "expert_dtype": str(sig.parameters["expert_dtype"].default),
        "weight_dtype": str(sig.parameters["weight_dtype"].default),
        "cache_dtype": str(sig.parameters["cache_dtype"].default),
        "embed_dtype": str(sig.parameters["embed_dtype"].default),
        "lm_head_dtype": str(sig.parameters["lm_head_dtype"].default),
        "lm_head_fidelity": sig.parameters["lm_head_fidelity"].default,
        "decoder_cls": sig.parameters["decoder_cls"].default.__name__,
    }
    expected_kwargs = config["construction"]["from_pretrained_kwargs_matching_this_config"]
    for key, expected in expected_kwargs.items():
        live = live_kwargs[key]
        live_norm = _dtype_short(live) if key != "lm_head_fidelity" and key != "decoder_cls" else live
        expected_norm = (
            _dtype_short(expected) if key != "lm_head_fidelity" and key != "decoder_cls" else expected.split(" ")[0]
        )
        if str(live_norm) != str(expected_norm):
            errors.append(f"from_pretrained default {key}={live!r} != selected_precision_config {expected!r}")

    live_attrs = {
        "attn_fidelity": SharedRopeDecoder.attn_fidelity,
        "mlp_fidelity": SharedRopeDecoder.mlp_fidelity,
        "expert_fidelity": SharedRopeDecoder.expert_fidelity,
        "router_fidelity": SharedRopeDecoder.router_fidelity,
        "attn_weight_dtype": str(SharedRopeDecoder.attn_weight_dtype),
        "mlp_gateup_dtype": str(SharedRopeDecoder.mlp_gateup_dtype),
        "mlp_down_dtype": str(SharedRopeDecoder.mlp_down_dtype),
        "dense_mlp_dtype": "null"
        if SharedRopeDecoder.dense_mlp_dtype is None
        else str(SharedRopeDecoder.dense_mlp_dtype),
        "prefill_proj_fidelity": SharedRopeDecoder.prefill_proj_fidelity,
        "prefill_expert_fidelity": SharedRopeDecoder.prefill_expert_fidelity,
    }
    expected_attrs = config["construction"]["decoder_class_attrs_matching_this_config"]
    for key, expected in expected_attrs.items():
        live = live_attrs[key]
        live_norm = _dtype_short(live) if "dtype" in key else live
        expected_norm = _dtype_short(expected) if "dtype" in key else expected.split(" ")[0]
        if str(live_norm) != str(expected_norm):
            errors.append(f"decoder class attr default {key}={live!r} != selected_precision_config {expected!r}")
    return errors


def check_baseline_run_matches(config: dict) -> list[str]:
    if not BASELINE_RUN_PATH.exists():
        return [f"missing {BASELINE_RUN_PATH} -- rerun dev_datatype_sweep.py --config-id C00_baseline first"]
    run = json.loads(BASELINE_RUN_PATH.read_text())
    snap = run["policy_snapshot"]
    errors = []

    groups = config["weight_groups"]
    pairs = [
        (
            _dtype_short(groups["routed_experts_gate_up"]["dtype"]),
            _dtype_short(snap["moe_layer"]["expert_gate_up_dtype"]),
            "routed_experts_gate_up.dtype",
        ),
        (
            _dtype_short(groups["routed_experts_down"]["dtype"]),
            _dtype_short(snap["moe_layer"]["expert_down_dtype"]),
            "routed_experts_down.dtype",
        ),
        (
            _dtype_short(groups["shared_expert_gate_up"]["dtype"]),
            _dtype_short(snap["moe_layer"]["shared_gate_up_dtype"]),
            "shared_expert_gate_up.dtype",
        ),
        (
            _dtype_short(groups["shared_expert_down"]["dtype"]),
            _dtype_short(snap["moe_layer"]["shared_down_dtype"]),
            "shared_expert_down.dtype",
        ),
        (
            _dtype_short(groups["attention_decode_dram_sharded_copies_and_absorbed_kv_b"]["dtype"]),
            _dtype_short(snap["moe_layer"]["attn_weight_dtype"]),
            "attention_decode.dtype",
        ),
        (
            _dtype_short(groups["dense_mlp_gate_up_down"]["dtype"]),
            _dtype_short(snap["dense_layer"]["mlp_gate_dtype"]),
            "dense_mlp.dtype",
        ),
        (
            _dtype_short(groups["dense_mlp_gate_up_down"]["dtype"]),
            _dtype_short(snap["dense_layer"]["dense_down_dtype"]),
            "dense_mlp_down.dtype",
        ),
        (_dtype_short(groups["lm_head"]["dtype"]), _dtype_short(snap["lm_head_dtype"]), "lm_head.dtype"),
        (
            _dtype_short(config["kv_cache_dtype"]["selected"]),
            # snap["kv_cache_dtype"] is read from generator._kv_cache[0].dtype, the real
            # allocated cache tensor -- not model.cache_dtype (which only echoes the
            # requested constructor kwarg and would pass even if allocate_kv_cache()
            # silently ignored it).
            _dtype_short(snap["kv_cache_dtype"]),
            "kv_cache.dtype",
        ),
        (
            _fidelity_key_from_prose(
                groups["attention_decode_dram_sharded_copies_and_absorbed_kv_b"]["compute_fidelity"]
            ),
            _fidelity_key_from_snapshot(snap["moe_layer"]["ck_attn"]),
            "attention_decode.compute_fidelity",
        ),
        (
            _fidelity_key_from_prose(groups["routed_experts_gate_up"]["compute_fidelity"]),
            _fidelity_key_from_snapshot(snap["moe_layer"]["ck_expert"]),
            "routed_experts.compute_fidelity",
        ),
        (
            _fidelity_key_from_prose(groups["router_gate"]["compute_fidelity"]),
            _fidelity_key_from_snapshot(snap["moe_layer"]["ck_router"]),
            "router_gate.compute_fidelity",
        ),
        (
            _fidelity_key_from_prose(groups["lm_head"]["compute_fidelity"]),
            _fidelity_key_from_snapshot(snap["ck_lm_head"]),
            "lm_head.compute_fidelity",
        ),
        (
            # shared_expert's decode fidelity is the "LoFi (decode) / HiFi2+fp32acc
            # (prefill)" string's first clause; ck_mlp_shared is the decode kernel config.
            _fidelity_key_from_prose(groups["shared_expert_gate_up"]["compute_fidelity"]),
            _fidelity_key_from_snapshot(snap["moe_layer"]["ck_mlp_shared"]),
            "shared_expert.compute_fidelity(decode)",
        ),
        (
            _fidelity_key_from_prose(groups["dense_mlp_gate_up_down"]["compute_fidelity"]),
            _fidelity_key_from_snapshot(snap["dense_layer"]["ck_mlp_dense"]),
            "dense_mlp.compute_fidelity(decode)",
        ),
        (
            _dtype_short(groups["router_gate"]["dtype"]),
            _dtype_short(snap["moe_layer"]["router_dtype"]),
            "router_gate.dtype",
        ),
    ]
    for expected, live, label in pairs:
        if expected != live:
            errors.append(
                f"{label}: selected_precision_config says {expected!r}, baseline run policy_snapshot says {live!r}"
            )
    return errors


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text())
    errors = check_source_defaults(config) + check_baseline_run_matches(config)
    if errors:
        print("PRECISION_CONFIG_PROPAGATION_CHECK: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PRECISION_CONFIG_PROPAGATION_CHECK: OK")
    print("selected_precision_config.json matches both the live source defaults and the introspected C00_baseline run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
