# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite the top level of ``doc/context_contract.json`` for the full-model stage.

The contract file is shared by every bringup stage: its **top level** belongs to
the newest stage and each earlier stage keeps its own nested block.  This script
moves the previous top level down into its own block (keyed by its ``stage``
field), then writes the full-model top level from the *measured* fields in
``doc/full_model/evidence_*.json`` -- so the byte budget cannot drift from the
build that produced it.

Every number it writes is a transcription of a committed run.  Prose, the
capability contract itself and the coverage lists are authored here.

Usage::

    python doc/full_model/bench/refresh_context_contract.py            # rewrite
    python doc/full_model/bench/refresh_context_contract.py --check     # exit 1 if stale
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
CONTRACT = ROOT / "doc/context_contract.json"
EVIDENCE = ROOT / "doc/full_model"
HF_CONTEXT = 131072
STAGE = "full_model"


def load_json(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing evidence file {path}; run bench/evidence.py first")
    return json.loads(path.read_text())


def build(previous: dict) -> dict:
    accuracy = load_json(EVIDENCE / "evidence_accuracy.json")
    capacity = accuracy["capacity"]
    misses_path = EVIDENCE / "evidence_misses.json"
    if misses_path.is_file() and "prefill_misses" not in accuracy:
        accuracy["prefill_misses"] = json.loads(misses_path.read_text()).get("prefill_misses")
    perf_path = EVIDENCE / "evidence_perf.json"
    perf_all = load_json(perf_path) if perf_path.is_file() else {}
    perf = perf_all.get("performance", {})

    demoted = {key: value for key, value in previous.items() if not isinstance(value, dict) or "stage" not in value}
    nested = {key: value for key, value in previous.items() if isinstance(value, dict) and "stage" in value}
    previous_stage = previous.get("stage", "unknown")

    contract: dict = {
        "schema_version": previous.get("schema_version", 1),
        "hf_model": previous["hf_model"],
        "stage": STAGE,
        "hf_advertised_context": HF_CONTEXT,
        "hf_advertised_context_source": previous["hf_advertised_context_source"],
        "current_supported_context": capacity["supported_context"],
        "capability_reduction": "none",
        "limiting_reason": None,
        "device": previous["device"],
        "implementation": {
            "model": "models/autoports/meta_models_muse_glimmer_30b/tt/model.py",
            "generator": "models/autoports/meta_models_muse_glimmer_30b/tt/generator.py",
            "layer_stack": "tt/multichip_decoder.py, unchanged (one additive optional kwarg: rope_cache)",
            "num_layers": capacity["num_layers"],
            "layer_kind_counts": {
                "sliding": sum(1 for kind in capacity["layer_kinds"] if kind == "sliding"),
                "full": sum(1 for kind in capacity["layer_kinds"] if kind == "full"),
            },
            "embedding": "column-parallel on the hidden dim + one async all_gather; bf16 ROW_MAJOR table",
            "embedding_pad_row": (
                "one extra zero row at index vocab_size, so a non-tile-aligned prompt pads with an id whose "
                "embedding is exactly zero rather than leaving uninitialised tile rows"
            ),
            "final_norm": "MuseGlimmerRMSNorm(with_scale=True): multiplies by w, not by 1+w",
            "lm_head": "column-parallel on the vocab dim; no logits gather on the token-out path",
            "lm_head_dtype": capacity["lm_head_dtype"],
            "lm_head_matmul": capacity["lm_head_matmul"],
            "lm_head_geometry": {
                "cores": capacity["lm_head_cores"],
                "in0_block_w": capacity["lm_head_in0_block_w"],
            },
            "logit_softcap": "T * tanh(h @ W * m / T), m/T folded into the weight at setup; T = 20.0",
            "vocab_size": capacity["vocab_size"],
            "padded_vocab_size": capacity["padded_vocab_size"],
            "padded_vocab_reason": (
                "202048/4 = 50512 is not tile-aligned, and the DRAM-sharded matmul additionally needs one "
                "weight shard per DRAM bank (per-device width a multiple of 256); the sampler masks the "
                "704 padded ids, because it is given the real vocab size alongside the padded one"
            ),
            "sampling": capacity["sampling_implementation"],
            "force_argmax": capacity["force_argmax"],
            "decode_rows": capacity["decode_rows"],
            "page_block_size": capacity["page_block_size"],
            "prefill_chunk_size": capacity["prefill_chunk_size"],
            "rope_tables": "one shared set for all 39 sliding layers (uniform layer_rope_theta = 500000.0)",
            "kv_cache_dtype": "BFLOAT8_B (carried forward unchanged from the optimized multichip decoder)",
            "residual_layout": (
                "decode: WIDTH_SHARDED L1 on 16 cores, [32, 416] shards, replicated, across every layer "
                "boundary and into the terminal norm; prefill: DRAM-interleaved replicated"
            ),
        },
        "byte_budget_at_full_context": {
            "measured_from": "doc/full_model/evidence_accuracy.json (capability_report over the built model)",
            "per_device_layer_weight_bytes": capacity["per_device_layer_weight_bytes"],
            "per_device_terminal_weight_bytes": capacity["per_device_terminal_weight_bytes"],
            "per_device_terminal_weight_formula": (
                "embed_tokens (202049 x 1664 x 2 B BF16 ROW_MAJOR) = 672,419,072 + lm_head "
                "(6656 x 50688 x 0.5625 B BFLOAT4_B) = 189,760,512 + two terminal norms"
            ),
            "per_device_rope_table_bytes": capacity["per_device_rope_table_bytes"],
            "per_device_rope_table_formula": (
                "4 tables (cos/sin x ROW_MAJOR/TILE) x 131072 positions x 128 head_dim x 2 B BF16, shared "
                "by all 39 sliding layers"
            ),
            "per_device_kv_cache_bytes": capacity["per_device_kv_cache_bytes"],
            "per_device_kv_cache_formula": (
                "52 layers x 2 (K,V) x 1 local KV head x 128 head_dim x 131072 tokens x 1.0625 B "
                "(BFLOAT8_B: 1024 mantissa + 64 exponent bytes per 32x32 tile)"
            ),
            "per_device_kv_cache_bytes_per_block": capacity["per_device_kv_cache_bytes_per_block"],
            "per_device_total_bytes": capacity["per_device_total_bytes"],
            "per_device_dram_capacity_bytes": capacity["per_device_dram_capacity_bytes"],
            "per_device_dram_capacity_source": (
                "ttnn.get_memory_view(mesh, BufferType.DRAM): total_bytes_per_bank x num_banks, read from "
                "the allocator rather than a data sheet"
            ),
            "per_device_free_after_long_lived_bytes": capacity["per_device_free_after_long_lived_bytes"],
            "trace_region_bytes": 400_000_000,
            "note": (
                "The advertised 131072-token context needs 7.18 GB/device of long-lived DRAM out of 31.46 "
                "GiB, so no capability reduction is required and none is taken. The full-context KV cache "
                "is 1.854 GB/device, so the free space after weights holds ~14 more full-length sequences; "
                "the batch contract below is what that buys."
            ),
        },
        "batch_contract": {
            "primary": "batch 1 at the full 131072-token context (the single-user latency target)",
            "decode_rows": capacity["decode_rows"],
            "decode_rows_note": (
                "every decode tensor is 32 rows wide regardless of batch -- the activation is tile-padded "
                "and nlp_create_qkv_heads_decode caps num_users at 32 -- and inactive rows carry "
                "current_pos = -1, the sentinel the paged attention ops skip"
            ),
            "cache_slots_are_a_separate_knob": (
                "max_batch_size sizes the paged pool (max_num_blocks = max_batch_size x blocks_per_seq); "
                "context and batch trade against each other inside the same byte budget"
            ),
            "device_op_ceiling": 32,
            "device_op_ceiling_source": (
                "nlp_create_qkv_heads_decode_device_operation.cpp:45-51 hard-caps num_users at 32"
            ),
            "full_context_sequences_that_fit": capacity["full_context_sequences_that_fit"],
        },
        "notes": (
            "The full-model stage takes no capability reduction. The advertised 131072-token context is "
            "supported and exercised end to end (52-layer paged prefill fill plus paged traced decode) at "
            "batch 1, and prompt lengths that are not divisible by the tile, the 64-token page or the "
            "8192-token prefill chunk are ordinary inputs: the generator pads the token ids with a "
            "zero-embedding pad id, the layer stack sees an aligned prompt, and the logits are sliced back "
            "to the logical last position. Earlier stages keep their own blocks below; the previous top "
            f"level is now the '{previous_stage}' block."
        ),
    }
    if perf:
        contract["performance"] = {
            "workload": perf["workload"],
            "ttft_ms": perf["ttft_ms"]["min"],
            "token_out_decode_ms_per_token": perf["token_out_decode_ms_per_token"]["min"],
            "token_out_decode_tok_s_u": perf["token_out_decode_tok_s_u"],
            "traced_decode_logits_only_ms_per_token": perf["traced_decode_logits_only_ms_per_token"]["min"],
            "traced_decode_logits_only_tok_s_u": perf["traced_decode_logits_only_tok_s_u"],
            "sampling_trace_ms_per_token": perf["sampling_trace_ms_per_token"]["min"],
            "layer_stack_lower_bound_ms_per_token": perf["layer_stack_lower_bound_ms_per_token"]["total_ms"],
            "source": "doc/full_model/evidence_perf.json",
        }
    contract["tested"] = load_tested(accuracy, perf_all)
    # The previous stage wrote *both* a top level and a nested block of its own
    # (its top level carried the device/byte-budget/tested fields, the nested block
    # its stage-specific supplement).  Merge them into one block for that stage, or
    # the update below silently drops whichever came first.
    merged = dict(demoted)
    merged.update(nested.get(previous_stage, {}))
    contract[previous_stage] = merged
    for key, value in nested.items():
        if key != previous_stage:
            contract[key] = value
    return contract


def by_reference(source: dict, key: str) -> dict:
    """``{reference name: {top1, top5, top100, ...}}`` for a runner's rows."""
    rows = source.get(f"{key}_by_reference") or {}
    return {
        name: {metric: row["per_entry"][0][metric] for metric in ("top1", "top5", "top100", "total", "k")}
        for name, row in rows.items()
        if row.get("per_entry")
    }


def load_tested(accuracy: dict, perf: dict | None = None) -> dict:
    tested: dict = {
        "commands": [
            "python doc/full_model/bench/evidence.py --stages capacity,prefill,teacher,sampling,shapes,fallback",
            "python doc/full_model/bench/evidence.py --stages autoregress",
            "python doc/full_model/bench/evidence.py --stages perf",
            "python doc/full_model/bench/qualitative.py --arm hf",
            "python doc/full_model/bench/qualitative.py --arm tt",
            "pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_full_model.py",
        ],
    }
    shapes = accuracy.get("prompt_shapes") or []
    if shapes:
        tested["prompt_shapes"] = {
            "lengths": [row["prompt_len"] for row in shapes],
            "non_aligned_lengths": [row["prompt_len"] for row in shapes if not row["aligned"]["tile"]],
            "largest_tested": max(row["prompt_len"] for row in shapes),
            "note": (
                "each length runs the public generator end to end (embed -> 52 layers -> terminal path -> "
                "traced decode); 'aligned' records tile/page/chunk divisibility per length"
            ),
        }
    prefill = accuracy.get("prefill_check", {}).get("per_entry")
    if prefill:
        tested["prefill_check"] = {
            "top1": prefill[0]["top1"],
            "top5": prefill[0]["top5"],
            "top100": prefill[0]["top100"],
            "total": prefill[0]["total"],
            "k": prefill[0]["k"],
        }
    teacher = accuracy.get("teacher_forcing", {}).get("per_entry")
    if teacher:
        tested["teacher_forcing"] = {
            "top1": teacher[0]["top1"],
            "top5": teacher[0]["top5"],
            "top100": teacher[0]["top100"],
            "total": teacher[0]["total"],
            "k": teacher[0]["k"],
            "ttft_ms": teacher[0].get("ttft_ms"),
            "decode_tok_s_u": teacher[0].get("decode_t/s/u"),
        }
    misses = accuracy.get("prefill_misses")
    if misses:
        tested["prefill_misses"] = {
            "non_top1_positions": misses["non_top1_positions"],
            "outside_top_k_positions": misses["outside_top_k_positions"],
            "k": misses["k"],
            "gen_len": misses["gen_len"],
            "note": "per-position detail in doc/full_model/evidence_misses_*.json",
        }
    if perf:
        # The shipped configuration measured alongside perf, against both the bf16
        # reference and the fp32 control, in one build.
        prefill_refs = by_reference(perf, "prefill_check")
        teacher_refs = by_reference(perf, "teacher_forcing")
        if prefill_refs:
            tested["prefill_check_by_reference"] = prefill_refs
        if teacher_refs:
            tested["teacher_forcing_by_reference"] = teacher_refs
        if perf.get("prefill_misses"):
            tested["prefill_misses_shipped_run"] = {
                "non_top1_positions": perf["prefill_misses"]["non_top1_positions"],
                "outside_top_k_positions": perf["prefill_misses"]["outside_top_k_positions"],
            }
    sampling = accuracy.get("split_sampling")
    if sampling:
        tested["split_sampling"] = sampling
    fallback = accuracy.get("fallback_audit")
    if fallback:
        tested["fallback_audit"] = fallback
    return tested


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    previous = json.loads(CONTRACT.read_text())
    if previous.get("stage") == STAGE:
        # Already the full-model top level: rebuild from the block we demoted last time.
        nested = {key: value for key, value in previous.items() if isinstance(value, dict) and "stage" in value}
        restored = dict(nested["optimized_multichip_decoder"])
        for key, value in nested.items():
            if key != "optimized_multichip_decoder":
                restored[key] = value
        previous = restored
    contract = build(previous)
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
