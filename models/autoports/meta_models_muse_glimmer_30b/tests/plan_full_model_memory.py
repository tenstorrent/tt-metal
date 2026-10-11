# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recompute full-stack DRAM accounting without opening hardware."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
previous = json.loads((ROOT / "doc/optimized_multichip_decoder/memory_capacity_plan.json").read_text())
c = json.loads((ROOT / "tests/assets/config.json").read_text())["text_config"]
buckets = (32, 64, 128, 256, 512, 1024, 2048, 4096)
vocab_shard = math.ceil(c["vocab_size"] / 128) * 32
terms = {
    "decoder_matrices_both_selected_layouts": previous["per_device_terms"]["decoder_matrix_weights_both_layouts"],
    "decoder_norms_tiled_and_row_major": previous["per_device_terms"][
        "replicated_four_hidden_norms_tiled_and_row_major"
    ],
    "shared_absolute_positions": 131072 * 4,
    "shared_rope_tiled_and_embedding_tables": 2 * 2 * 131072 * 128 * 2,
    "normalized_embedding_hidden_tp4": c["vocab_size"] * 6656 * 2 // 4,
    "lm_head_dram_sharded_bf16": 6656 * math.ceil(vocab_shard / (8 * 32 * 2)) * (8 * 32 * 2) * 2,
    "final_norm_row_major": 6656 * 2,
    "prepared_prefill_hidden_buckets": sum(buckets) * 6656 * 2,
    "prefill_token_and_rope_ids": sum(buckets) * 4 * 2,
    "prefill_page_tables": len(buckets) * 4096 * 4 + sum(buckets) // 32 * 4,
    "b32_decode_page_table": 32 * 4096 * 4,
    "b32_persistent_logits_and_prefill_accumulator": 2 * 32 * vocab_shard * 2,
    "terminal_slice_control_backing": 2 * 2048,
    "short_prefill_tail_position": 4,
    "sampler_masks_offsets_params_and_terminal_buffers_reservation": 64 * 1024**2,
    "trace_reservation": 1024**3,
    "activation_ccl_program_cache_allocator_slack_reservation": 8 * 1024**3,
    "preserved_nontext_weight_reservation": previous["per_device_terms"][
        "reserved_replicated_bf16_vision_and_projectors"
    ],
}
kv_per_page = 52 * 2 * 32 * 128 // 1024 * 1088
dram = previous["per_device_dram_arch_descriptor_bytes"]
page_capacity = (dram - sum(terms.values())) // kv_per_page
result = dict(
    classification="modeled",
    status="construction plan; full-stack allocation probe pending",
    model_id=previous["model_id"],
    revision=previous["revision"],
    mesh=[1, 4],
    context=131072,
    max_batch=32,
    default_physical_pages=4096,
    cache_dtype="bfloat8_b",
    cache_geometry_per_layer_per_tensor=["physical_pages", 1, 32, 128],
    per_device_terms=terms,
    per_device_dram_bytes=dram,
    per_device_kv_bytes_per_page_all_layers=kv_per_page,
    per_device_kv_bytes_default_pool=kv_per_page * 4096,
    per_device_default_total_bytes=sum(terms.values()) + kv_per_page * 4096,
    per_device_default_margin_bytes=dram - sum(terms.values()) - kv_per_page * 4096,
    modeled_max_pages_with_reservations=page_capacity,
    modeled_full_context_concurrency=page_capacity // 4096,
    capability_reduction=None,
    admission="At most32 fixed logical slots; active requests own disjoint physical pages. Resident tokens share the cache pool. Per-request context is131072.",
    caveat="Measured allocator, trace-region and program-cache consumption still needs validation. No unsupported capability reduction follows from this plan.",
)
for name in ("full_model_ready.json", "full_model_final.json"):
    evidence = ROOT / "doc/full_model" / name
    if evidence.exists():
        measured = json.loads(evidence.read_text())
        if "allocator" in measured:
            result["measured_allocator"] = dict(
                classification="measured",
                evidence=f"full_model/{name}",
                views=measured["allocator"],
                note="TRACE view is zero for dynamic trace allocation; dynamic trace buffers are included in DRAM allocation.",
            )
            result["status"] = "full-stack construction and preparation fit; capacity arithmetic remains modeled"
            result[
                "caveat"
            ] = "Reservation terms are modeled, not measured allocations. Six full-context concurrent requests are a modeled upper admission budget, not a tested concurrency claim."
            break
context_evidence = ROOT / "doc/full_model/context_full.json"
if context_evidence.exists():
    measured = json.loads(context_evidence.read_text())
    assert measured["layers"] == 52 and measured["context"] == 131072
    result.update(
        status="full-stack context validated at131071/131072; no capability reduction",
        supported_context=131072,
        largest_tested_prefill=131072,
        largest_tested_decode_context=131072,
        last_tested_position=131071,
        context_evidence="full_model/context_full.json",
    )
batch_evidence = ROOT / "doc/full_model/batch32_full.json"
if batch_evidence.exists():
    measured = json.loads(batch_evidence.read_text())
    assert measured["layers"] == 52 and measured["batch"] == 32
    result.update(largest_tested_batch=32, batch_evidence="full_model/batch32_full.json")
(ROOT / "doc/full_model/memory_capacity_plan.json").write_text(json.dumps(result, indent=2) + "\n")
contract_path = ROOT / "doc/context_contract.json"
contract = json.loads(contract_path.read_text())
contract["full_model_stage_context_preservation"] = result
contract_path.write_text(json.dumps(contract, indent=2) + "\n")
print(
    json.dumps(
        {
            key: result[key]
            for key in (
                "per_device_default_total_bytes",
                "per_device_default_margin_bytes",
                "modeled_full_context_concurrency",
            )
        },
        indent=2,
    )
)
