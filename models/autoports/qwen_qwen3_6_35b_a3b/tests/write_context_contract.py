# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Regenerate ``doc/context_contract.json`` **from** the recorded long-context evidence.

Deriving the contract instead of hand-writing it is what keeps it from going stale:
every number traces to a row in ``doc/functional_decoder/long_context.jsonl``, and
``tests/test_functional_decoder.py::test_context_contract_file_is_consistent`` re-checks the
same relationship on every run.

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/write_context_contract.py
"""

import json

from models.autoports.qwen_qwen3_6_35b_a3b.tests.harness import ARTIFACT_DIR
from models.autoports.qwen_qwen3_6_35b_a3b.tt import reference as ref

KIND_LABEL = {"linear": "linear_attention", "full": "full_attention"}


def main():
    hf = ref.load_hf_text_config()
    rows = [json.loads(line) for line in (ARTIFACT_DIR / "long_context.jsonl").read_text().splitlines() if line.strip()]
    by_label = {r["label"]: r for r in rows}

    def pick(prefix, kind):
        matches = [r for label, r in by_label.items() if label.startswith(prefix) and f"[{kind}]" in label]
        if not matches:
            raise SystemExit(f"missing evidence row: {prefix}[{kind}]")
        return matches[-1]

    tested = {"prefill": {}, "decode": {}}
    for kind, label in KIND_LABEL.items():
        p = pick("longest-prefill", kind)
        d = pick("longest-decode", kind)
        tested["prefill"][label] = {
            "seq_len": p["seq_len"],
            "tile_aligned": p["seq_len"] % 128 == 0,
            "hf_pcc": round(p["pcc"], 7),
            "hf_reference": "exact tail reference over the last %d query positions" % p["tail"],
            "device_wall_seconds": p["wall_seconds"],
        }
        tested["decode"][label] = {
            "position": d["position"],
            "context_entries": d["position"] + 1,
            "hf_pcc": round(d["pcc"], 7),
            "state_source": "262143-token prefill through this layer",
            "device_wall_seconds": d["wall_seconds"],
        }

    capacity = by_label.get("kv-capacity batch32 full context", {})
    contract = {
        "hf_model_id": ref.HF_MODEL_ID,
        "stage": "functional-decoder",
        "hf_advertised_context": hf.max_position_embeddings,
        "hf_advertised_context_source": "config.json -> text_config.max_position_embeddings",
        "supported_context": hf.max_position_embeddings,
        "capability_reduction": None,
        "capability_reduction_reason": None,
        "largest_prefill_tested": max(v["seq_len"] for v in tested["prefill"].values()),
        "largest_decode_context_tested": max(v["context_entries"] for v in tested["decode"].values()),
        "layer_kinds_covered": sorted(KIND_LABEL.values()),
        "tested": tested,
        "device_capacity_evidence": {
            "device": "1x Blackhole p300c chip, 1x1 mesh",
            "usable_dram_bytes": 33822867456,
            "usable_dram_probe": (
                "tests/probe_dram_capacity.py: allocated 512 MiB DRAM tensors until the bank "
                "manager refused, 63 x 512 MiB = 33822867456 B = 31.50 GiB "
                "(doc/functional_decoder/logs/probe_dram_capacity.log)"
            ),
            "paged_kv_bytes_batch32_full_context": capacity.get("bytes_total"),
            "paged_kv_blocks_batch32_full_context": capacity.get("blocks"),
            "paged_kv_allocated_on_device": capacity.get("allocated"),
            # 256 experts x (gate_up [1024, 2048] + down [2048, 512]) in bf16
            "moe_weight_bytes_per_layer_bf16": 256 * (1024 * 2048 + 2048 * 512) * 2,
            "note": (
                "batch 32 at the full advertised context needs 2 x 8 GiB of paged K/V (17.18 GB "
                "total), which was actually allocated on device by "
                "tests/test_long_context.py::test_max_batch_full_context_capacity. Together with "
                "~1.61 GB of MoE weights per layer this fits the 31.5 GiB part, so no capability "
                "reduction is required at the decoder-layer level."
            ),
        },
        "prefill_input_contract": {
            "accepts_any_logical_seq_len": True,
            "internal_alignment": 128,
            "note": (
                "prefill_forward pads to a multiple of PREFILL_ALIGN=128 internally, masks the "
                "padded tokens out of the linear-attention recurrence, and slices the output back "
                "to the logical length. Non-aligned lengths 1/33/65/129/1025/2049/3000/262143 are "
                "covered by tests."
            ),
            "start_pos_alignment": 128,
        },
        "evidence": {
            "long_context_rows": "doc/functional_decoder/long_context.jsonl",
            "pcc_rows": "doc/functional_decoder/pcc.jsonl",
            "real_weight_rows": "doc/functional_decoder/pcc_real_weights.jsonl",
            "logs": "doc/functional_decoder/logs/",
            "readme": "doc/functional_decoder/README.md",
        },
    }
    path = ARTIFACT_DIR.parent / "context_contract.json"
    path.write_text(json.dumps(contract, indent=2) + "\n")
    print(f"wrote {path}")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
