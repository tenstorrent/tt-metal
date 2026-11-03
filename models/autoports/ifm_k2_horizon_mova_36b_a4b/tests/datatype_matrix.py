# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible coarse precision matrix, starting from the accepted policy."""

import copy
import json
from pathlib import Path

from ..tt.precision import BASELINE

DOC = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"


def main():
    configs = []

    def add(name, **updates):
        p = copy.deepcopy(BASELINE)
        p["config_id"] = name
        for k, v in updates.items():
            if isinstance(v, dict):
                p[k].update(v)
            else:
                p[k] = v
        configs.append(p)

    add("baseline_bfp4_lofi")
    for dtype in ("bfloat8_b", "bfloat4_b"):
        for fidelity in ("LoFi", "HiFi2"):
            add(f"head_{dtype}_{fidelity}", lm_head_dtype=dtype, lm_head_fidelity=fidelity)
    add("decoder_bfp4_hifi2", compute_fidelities={k: "HiFi2" for k in BASELINE["compute_fidelities"]})
    for fidelity in ("LoFi", "HiFi2"):
        # Projection-only BFP8 family fits full capacity; expert banks are separate.
        add(
            f"projections_bfp8_{fidelity}",
            weight_groups={
                k: "bfloat8_b"
                for k in (
                    "attention_dtype",
                    "attention_decode_dtype",
                    "output_weight_dtype",
                    "mlp_dtype",
                    "down_weight_dtype",
                )
            },
            compute_fidelities={"attention": fidelity, "mlp": fidelity},
        )
        add(
            f"experts_bfp8_{fidelity}",
            weight_groups={k: "bfloat8_b" for k in ("experts_dtype", "expert_down_dtype", "value_dtype")},
            compute_fidelities={"experts": fidelity, "value": fidelity},
        )
    for fidelity in ("LoFi", "HiFi2"):
        for group, keys, compute in (
            ("expert_down", ("expert_down_dtype",), "experts"),
            ("expert_gate_up", ("experts_dtype",), "experts"),
            ("value", ("value_dtype",), "value"),
        ):
            add(
                f"{group}_bfp8_{fidelity}",
                weight_groups={k: "bfloat8_b" for k in keys},
                compute_fidelities={compute: fidelity},
            )
    add("dense_prefill_bfp4_lofi", layer_exceptions=[])
    add("ccl_bf16", ccl_dtype="bfloat16")
    add("prefill_ccl_bfp8", prefill_ccl_dtype="bfloat8_b")
    add("ccl_bfp4", ccl_dtype="bfloat4_b")
    add("kv_bfp4", kv_cache_dtype="bfloat4_b")
    add("kv_bf16", kv_cache_dtype="bfloat16")
    add("residual_bfp8", residual_dtype="bfloat8_b")
    add("normed_activations_bfp4", activations={"attention_activation": "bfloat4_b", "mlp_activation": "bfloat4_b"})
    add(
        "expert_activations_bfp4",
        activations={
            k: "bfloat4_b"
            for k in (
                "expert_activation",
                "expanded_expert_activation",
                "prefill_value_activation",
                "ep_value_activation",
            )
        },
    )
    add("activations_bf16", activations={k: "bfloat16" for k in BASELINE["activations"]})
    add("head_bfp4_lofi_block5", lm_head_dtype="bfloat4_b", lm_head_fidelity="LoFi", lm_head_in0_block_w=5)
    add("head4_ccl4_lofi", lm_head_dtype="bfloat4_b", lm_head_fidelity="LoFi", ccl_dtype="bfloat4_b")
    add(
        "head4_ccl4_norm4_lofi",
        lm_head_dtype="bfloat4_b",
        lm_head_fidelity="LoFi",
        ccl_dtype="bfloat4_b",
        activations={"attention_activation": "bfloat4_b", "mlp_activation": "bfloat4_b"},
    )
    # BFP4 cache recovers enough DRAM to evaluate the closest all-BFP8 weight
    # policy without reducing the checkpoint's advertised context capacity.
    for fidelity in ("LoFi", "HiFi2"):
        for kv_bits in (4, 8):
            add(
                f"canonical_bfp8_kv{kv_bits}_{fidelity}",
                weight_groups={k: "bfloat8_b" for k in BASELINE["weight_groups"]},
                layer_exceptions=[],
                compute_fidelities={k: fidelity for k in BASELINE["compute_fidelities"]},
                lm_head_dtype="bfloat8_b",
                lm_head_fidelity=fidelity,
                kv_cache_dtype=f"bfloat{kv_bits}_b",
            )
    (DOC / "configs").mkdir(parents=True, exist_ok=True)
    for config in configs:
        (DOC / "configs" / f'{config["config_id"]}.json').write_text(json.dumps(config, indent=2) + "\n")
    (DOC / "candidate_matrix.json").write_text(json.dumps([p["config_id"] for p in configs], indent=2) + "\n")


if __name__ == "__main__":
    main()
