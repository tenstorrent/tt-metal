"""Deterministic coarse precision matrix from the reviewed stage7 baseline."""

import copy
import json
from pathlib import Path

DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/datatype_sweep")


def main():
    base = json.loads((DOC / "configs/baseline_mixed_hifi2_lofi.json").read_text())
    configs = [base]

    def candidate(name, changes=None, layers=None, ccl=None, head=None):
        c = copy.deepcopy(base)
        c["config_id"] = name
        if changes:
            if layers is None:
                c["layer_defaults"].update(changes)
                for override in c["layer_exceptions"].values():
                    for key in changes:
                        override.pop(key, None)
            else:
                for layer in layers:
                    c["layer_exceptions"].setdefault(str(layer), {}).update(changes)
        if ccl:
            c["dtypes"]["ccl"] = ccl
        if head:
            c["head"].update(head)
        configs.append(c)

    candidate("canonical_accuracy_bfp8_hifi2", dict(mlp="bfloat8_b", mlp_fidelity="HiFi2"))
    candidate("canonical_performance_mlp4_lofi", dict(mlp="bfloat4_b", mlp_fidelity="LoFi"))
    candidate("baseline_mlp4_hifi2", dict(mlp_fidelity="HiFi2"), [i for i in range(36) if i not in [0, *range(3, 11)]])
    candidate("promoted_mlp8_lofi", dict(mlp_fidelity="LoFi"), [0, *range(3, 11)])
    for role in ["qkv", "o", "down"]:
        candidate(f"{role}8_lofi", {role + "_fidelity": "LoFi"})
    candidate("head8_lofi", head=dict(compute_fidelity="LoFi"))
    candidate(
        "all_projection_lofi",
        dict(qkv_fidelity="LoFi", o_fidelity="LoFi", mlp_fidelity="LoFi", down_fidelity="LoFi"),
        head=dict(compute_fidelity="LoFi"),
    )
    candidate("kv16", dict(kv="bfloat16"))
    candidate("kv4", dict(kv="bfloat4_b"))
    candidate("ccl8", ccl="bfloat8_b")
    candidate("decode_activation8", dict(attention_activation="bfloat8_b", mlp_activation="bfloat8_b"))
    candidate(
        "ccl8_decode_activation8", dict(attention_activation="bfloat8_b", mlp_activation="bfloat8_b"), ccl="bfloat8_b"
    )
    candidate("all_prefill_activation8", dict(prefill_qkv_activation="bfloat8_b"))
    # Matched fidelity pairs for each BFP4 group; preserve first/last where
    # reducing a formerly BFP8 group. The original last MLP is already BFP4.
    for role, field in [("mlp", "mlp"), ("down", "down"), ("qkv", "qkv_dtype"), ("o", "o_dtype")]:
        for fidelity in ["LoFi", "HiFi2"]:
            candidate(
                f"inner_{role}4_{fidelity.lower()}", {field: "bfloat4_b", role + "_fidelity": fidelity}, range(1, 35)
            )
    for fidelity in ["LoFi", "HiFi2"]:
        candidate("head4_" + fidelity.lower(), head=dict(weight_dtype="bfloat4_b", compute_fidelity=fidelity))
    candidate(
        "aggressive_all_bfp4_lofi",
        dict(
            qkv_dtype="bfloat4_b",
            o_dtype="bfloat4_b",
            mlp="bfloat4_b",
            down="bfloat4_b",
            qkv_fidelity="LoFi",
            o_fidelity="LoFi",
            mlp_fidelity="LoFi",
            down_fidelity="LoFi",
        ),
    )
    for c in configs:
        (DOC / "configs" / f"{c['config_id']}.json").write_text(json.dumps(c, indent=2) + "\n")
    (DOC / "coarse_matrix.json").write_text(json.dumps([c["config_id"] for c in configs], indent=2) + "\n")


if __name__ == "__main__":
    main()
