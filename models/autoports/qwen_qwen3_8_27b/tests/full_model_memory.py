# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-stack TP4 DRAM plan using stored tile sizes and the selected decoder plan."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
prior = json.loads((root / "doc/optimized_multichip_decoder/memory_capacity_plan.json").read_text())
config = json.loads((root / "doc/functional_decoder/hf_config.json").read_text())["text_config"]
context = config["max_position_embeddings"]
vocab = config["vocab_size"]
hidden = config["hidden_size"]
linear = config["layer_types"].count("linear_attention")
full = config["layer_types"].count("full_attention")
resources = {
    "decoder_projections_bfp4_both_layouts": prior["all_decoder_projection_bytes_per_device"],
    "embedding_bf16_hidden_shard": vocab * hidden * 2 // 4,
    "lm_head_bfp8_vocab_shard": (hidden // 32) * (vocab // 4 // 32) * 1088,
    "lm_head_bfp8_dram_decode_copy": sum(
        (hidden // 32) * (((min(8192, vocab // 4 - start) + 511) // 512) * 512 // 32) * 1088
        for start in range(0, vocab // 4, 8192)
    ),
    "decoder_norms_constants_reserve": prior["norm_and_constant_reserve_bytes_per_device"],
    "final_norm_bf16": 32 * hidden * 2,
    "rope_bf16_tables": 2 * context * 64 * 2,
    "full_kv_bfp8": full * 2 * (context // 32) * (config["head_dim"] // 32) * 1088,
    "linear_recurrence_fp32": linear * 12 * 128 * 128 * 4,
    "linear_conv_bf16_row_major": linear * 3 * 2560 * 2,
    "page_table_int32": context // 32 * 4,
    "trace_region_per_device": 200000000,
    "persistent_ccl_and_sampling_reserve": 256 * 1024**2,
    "bounded_4096_prefill_activations_and_scratch_reserve": 2 * 1024**3,
}
total = sum(resources.values())
physical = prior["device_dram_bytes"]
assert total < physical
evidence = root / "doc/full_model/readiness_confirmed.json"
executed = json.loads(evidence.read_text())["final_context"]
assert (evidence.with_suffix(".exit_status")).read_text().strip() == "0"
assert [(r["length"], r["generated"], r["position"]) for r in executed] == [
    (context - 1, 2, context),
    (context, 1, context),
]
report = dict(
    supported_context=context,
    hf_advertised_context=context,
    capability_reduction=None,
    mesh=[1, 4],
    batch=1,
    page_size=32,
    kv_dtype="bfloat8_b",
    prefill_fill_dtype="bfloat8_b",
    decode_update_dtype="bfloat16",
    linear_recurrent_dtype="float32",
    resources_bytes_per_device=resources,
    total_bytes_per_device=total,
    device_dram_bytes=physical,
    headroom_bytes_per_device=physical - total,
    projection_source="optimized_multichip_decoder/memory_capacity_plan.json",
    public_sequence_alignment_requirement=None,
    prefill_chunk_size=4096,
    executed_capacity_evidence="full_model/readiness_confirmed.json: all64 layers with selected DRAM terminal and repaired native reader; S262143+decode and S262144 prefill pass, process exit0.",
    largest_tested_prefill=context,
    largest_tested_decode_context=context,
    batch_coverage=dict(
        largest_tested_batch=32,
        mixed_prompt_lengths=[31, 33],
        fixed_slots=[31, 0],
        full_context_batch=1,
        evidence="full_model/contract_full_b32.json",
        scope="Batch32 short prompts and batch1 maximum context are separate coverage points, not every joint allocation.",
    ),
    page_allocation_contract="ceil(logical capacity/32) pages per slot; SDPA K chunk is reduced until it divides mapped capacity",
    arithmetic_scope="Full 64-layer weights and state; bounded stack-level prefill activations. BF8 tile1088B, BF4 tile576B. Reserves conservative; capacity execution recorded separately.",
    source_sha256={
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "tt/model.py", root / "tt/generator.py")
    },
)
(root / "doc/full_model/memory_capacity_plan.json").write_text(json.dumps(report, indent=2) + "\n")
p = root / "doc/context_contract.json"
contract = json.loads(p.read_text())
contract["full_model"] = report
contract["note"] = (
    "No capability reduction. Long-context and batch32 tests are separate coverage points, not every context/batch combination. "
    "Earlier decoder-only batch coverage uses shared logical lengths. The full-model generator owns paged KV, "
    "absolute positions, RoPE, recurrent/conv state and padding, and supports mixed logical lengths, fixed slots and inactive rows."
)
p.write_text(json.dumps(contract, indent=2) + "\n")
print(json.dumps(report, indent=2))
