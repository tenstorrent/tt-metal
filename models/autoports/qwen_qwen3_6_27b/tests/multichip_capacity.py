# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only TP4 resident/traffic accounting; no device imports or allocations."""

import json
import math
from pathlib import Path


def plan():
    def matrix(k, n):
        # Eight banks, two readers per bank, eight workers.
        physical_n = math.ceil(n / 512) * 512
        return {
            "logical_shape": [k, n],
            "physical_shape": [k, physical_n],
            "dram_bank_shard": [k, physical_n // 8],
            "activation_core_shard": [32, k // 8],
            "dtype": "bfloat8_b",
            "bytes": k * physical_n // 1024 * 1088,
        }

    # Selected decode policy removes the separate gate/up weights before DRAM
    # sharding. Concatenating them avoids their independent N512 padding.
    common = {name: matrix(k, n) for name, k, n in [("gate_up", 5120, 8704), ("down", 4352, 5120)]}
    gdn = common | {
        name: matrix(k, n) for name, k, n in [("input", 5120, 4160), ("correction", 5120, 2624), ("output", 1536, 5120)]
    }
    full = common | {
        name: matrix(k, n)
        for name, k, n in [("qkv", 5120, 2048), ("gate_attention", 5120, 1536), ("output", 1536, 5120)]
    }
    gdn_bytes, full_bytes = (sum(x["bytes"] for x in group.values()) for group in (gdn, full))
    banks, alignment = 8, 64

    def bank_bytes(pages, page_bytes):
        # BankManager::allocate_buffer / calculate_bank_size_spread reserves the
        # same rounded page count on every bank, including unused final pages.
        return banks * math.ceil(pages / banks) * math.ceil(page_bytes / alignment) * alignment

    def tensor(shape, dtype, layout, instances):
        element_bytes = {"bfloat8_b": 1, "bfloat16": 2, "float32": 4, "uint32": 4, "int32": 4}[dtype]
        if layout == "TILE":
            pages = math.prod(shape[:-2]) * math.ceil(shape[-2] / 32) * math.ceil(shape[-1] / 32)
            page_bytes = 1088 if dtype == "bfloat8_b" else 1024 * element_bytes
        else:
            assert dtype != "bfloat8_b", "BFP8 accounting requires tiled exponent blocks"
            pages = math.prod(shape[:-1])
            page_bytes = shape[-1] * element_bytes
        allocation = bank_bytes(pages, page_bytes)
        return {
            "logical_shape": list(shape),
            "dtype": dtype,
            "layout": layout,
            "instances_per_device": instances,
            "pages_per_instance": pages,
            "page_bytes_before_alignment": page_bytes,
            "allocated_bytes_per_instance": allocation,
            "allocated_bytes_all_instances": allocation * instances,
        }

    # These vectors retain tile-row padding. Counting only parameter elements
    # misses, for example, 100MiB of hidden norm vectors across 64 layers.
    auxiliary = {
        "replicated_hidden_norms": tensor((1, 1, 5120), "float32", "TILE", 64 * 2),
        "sharded_hidden_norms": tensor((1, 1, 1280), "float32", "TILE", 64 * 2),
        "full_attention_qk_norms": tensor((1, 1, 256), "float32", "TILE", 16 * 2),
        "gdn_norm": tensor((1, 1, 128), "bfloat16", "TILE", 48),
        "gdn_dt_bias_a_log_and_neg_a": tensor((1, 1, 12), "float32", "TILE", 48 * 3),
        "gdn_tiled_convolution_taps": tensor((1, 1, 2560), "bfloat16", "TILE", 48 * 4),
        "gdn_row_major_tap_copies": tensor((1, 1, 2560), "bfloat16", "ROW_MAJOR", 48 * 4),
        "gdn_convolution_exchange": tensor((1, 12, 2560), "bfloat16", "ROW_MAJOR", 48),
        "gdn_eye_lower_ones": tensor((1, 1, 32, 32), "float32", "TILE", 48 * 3),
        "gdn_masks": tensor((1, 1, 32, 96), "float32", "TILE", 48),
        "gdn_convolution_start_and_512_ends": tensor((1,), "uint32", "ROW_MAJOR", 48 * 513),
        "attention_page_tables": tensor((1, 8192), "int32", "ROW_MAJOR", 16),
    }
    auxiliary_bytes = sum(value["allocated_bytes_all_instances"] for value in auxiliary.values())
    states = {
        "gdn_recurrent": tensor((1, 12, 128, 128), "bfloat16", "TILE", 48),
        "gdn_convolution_history": tensor((1, 3, 2560), "bfloat16", "ROW_MAJOR", 48),
    }
    recurrent = sum(value["allocated_bytes_all_instances"] for value in states.values())
    # Future full-model estimates, not an implementation in this decoder stage.
    # Keep independent BF16 embedding/head weights and allow the same N512
    # projection padding policy for the vocab-sharded head (62080 -> 62464).
    embedding = tensor((248320 // 4, 5120), "bfloat16", "TILE", 1)
    head = tensor((5120, math.ceil((248320 // 4) / 512) * 512), "bfloat16", "TILE", 1)
    final_norm = tensor((1, 1, 5120), "float32", "TILE", 1)
    embedding_head = embedding["allocated_bytes_all_instances"] + head["allocated_bytes_all_instances"]
    row_copy = common["down"]["bytes"] + gdn["output"]["bytes"]
    matrix_weights = 48 * gdn_bytes + 16 * full_bytes
    # Prefill minimal SwiGLU uses an independently allocated, tile-pair-
    # interleaved copy; it cannot alias the contiguous gate/up decode matrix.
    swiglu_copy = tensor((5120, 8704), "bfloat8_b", "TILE", 64)
    swiglu_copy_bytes = swiglu_copy["allocated_bytes_per_instance"]
    weights = matrix_weights + embedding_head + 64 * row_copy + swiglu_copy["allocated_bytes_all_instances"]
    kv_pages_per_tensor = 8192 * 32 * 256 // 1024
    kv = 16 * 2 * bank_bytes(kv_pages_per_tensor, 1088)
    resident = weights + kv + recurrent + auxiliary_bytes + final_norm["allocated_bytes_all_instances"]
    reserve = 4 * 1024**3
    hidden = 262144 * 1280 * 2
    rope = 2 * 262144 * 64 * 2
    trace = 32 * 1024**2
    return {
        "model": "Qwen/Qwen3.6-27B",
        "mesh": [1, 4],
        "chips": [0, 1, 2, 3],
        "context": 262144,
        "batch_at_full_context": 1,
        "capability_reduction": None,
        "selected_policy": {"packed_mlp": True, "minimal_swiglu": "prefill", "fused_mmrs": True},
        "dram_allocator": {
            "banks": banks,
            "alignment_bytes": alignment,
            "formula": "banks * ceil(pages/banks) * align64(page_bytes)",
        },
        "matrices_per_device": {"linear_attention": gdn, "full_attention": full},
        # Traffic accounting consumes this field: one copy of each matrix read
        # by one token, including correction. Do not add resident duplicate copies.
        "layer_weights_bytes": {"linear_attention": gdn_bytes, "full_attention": full_bytes},
        "layer_weights_bytes_scope": "per-token decode matrix reads; packed gate/up read once; excludes duplicate resident MMRS/SwiGLU copies and auxiliary tensors",
        "all_64_layers_matrix_weights_bytes": matrix_weights,
        "all_64_layers_plus_bf16_embedding_head_bytes": weights,
        "fused_mmrs_copies_bytes_per_layer": row_copy,
        "fused_mmrs_copies_bytes_all_layers": 64 * row_copy,
        "prefill_swiglu_copy_per_device": swiglu_copy,
        "prefill_swiglu_copies_bytes_per_layer": swiglu_copy_bytes,
        "prefill_swiglu_copies_bytes_all_layers": swiglu_copy["allocated_bytes_all_instances"],
        "resident_layer_weights_bytes": {
            "linear_attention": gdn_bytes + row_copy + swiglu_copy_bytes,
            "full_attention": full_bytes + row_copy + swiglu_copy_bytes,
        },
        "resident_layer_weights_bytes_scope": "matrix weights plus MMRS and prefill SwiGLU copies; auxiliary tensors accounted separately",
        "persistent_auxiliary_tensors_per_device": auxiliary,
        "persistent_auxiliary_bytes_per_device": auxiliary_bytes,
        "persistent_gdn_states_per_device": states,
        "future_full_model_persistent_estimates": {
            "bf16_vocab_sharded_embedding": embedding,
            "bf16_vocab_sharded_head_with_n512_padding": head,
            "replicated_fp32_final_norm": final_norm,
            "unimplemented": True,
        },
        "all_16_attention_layers_kv_bytes_per_device": kv,
        "kv_physical_pages_per_layer": 8192,
        "all_48_gdn_layers_state_bytes_per_device": recurrent,
        "estimated_resident_bytes_per_device": resident,
        "reserved_activation_trace_metadata_bytes": reserve,
        "trace_region_component_bytes": trace,
        "decoder_transient_breakdown": {
            "full_context_sharded_hidden_tensor_bytes": hidden,
            "input_plus_retained_chunk_outputs_plus_concat_output_bytes": 3 * hidden,
            "replicated_rope_pair_bytes": rope,
            "one_extra_fp32_gdn_state_bytes": 12 * 128 * 128 * 4,
            "remainder_for_chunk_workspaces_allocator_metadata_and_other_transients_bytes": reserve
            - 3 * hidden
            - rope
            - trace
            - 12 * 128 * 128 * 4,
        },
        "future_logits_contract": {
            "expectation": "final-token logits or bounded token chunks; never materialize all-token logits at maximum context",
            "bf16_local_vocabulary_logits_bytes_if_all_tokens": 262144 * (248320 // 4) * 2,
            "logical_local_vocabulary": 248320 // 4,
            "head_physical_vocabulary": head["logical_shape"][-1],
        },
        "twenty_gib_additional_reservation_bytes": 20 * 1024**3,
        "twenty_gib_reservation_minus_estimated_full_stack_resident_bytes": 20 * 1024**3 - resident,
        "estimated_total_bytes_per_device": resident + reserve,
        "nominal_dram_bytes_per_device": 32 * 1024**3,
        "validation_status": "planning; device allocation/context gates pending",
        "notes": [
            "BFP8 exponent bytes, DRAM N padding, TILE vector padding, and per-bank page rounding included.",
            "4GiB is a decoder transient reserve: full input, retained chunk outputs, concat output, rope, one widened recurrent state, chunk workspaces, metadata and trace. Device peak validation remains required.",
            "Future BF16 embedding/head and FP32 final norm are estimates; no full-model code is implemented. Head N512 padding is conservative.",
            "Packed gate/up replaces separate padded matrices for decode; minimal_swiglu=prefill retains a distinct tile-pair-interleaved BFP8 copy of shape[5120,8704] in every layer.",
            "Fused MMRS retains interleaved down/output copies in addition to DRAM-sharded decode weights; all64 copies are counted as residency, not extra per-token matrix reads.",
            "CCL and convolution global semaphores reside in L1, not DRAM; their runtime resources still require separate device validation.",
            "Sparse physical page IDs, extra padding pages, and simultaneous users require additional cache capacity. The context runner's 8193-page table rounds to8208, above this exact8192-page plan.",
            "A 20GiB synthetic reservation is additional to the runner's actual layer/cache/IO and exceeds the estimated full-stack resident bytes; it tests aggregate capacity, not identical fragmentation.",
        ],
    }


if __name__ == "__main__":
    root = Path(__file__).parents[1]
    result = plan()
    (root / "doc/multichip_decoder/capacity_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    contract = json.loads((root / "doc/context_contract.json").read_text())
    contract["multichip_decoder"] = {
        "status": "planned; validation pending",
        "target_mesh": [1, 4],
        "supported_context": 262144,
        "capability_reduction": None,
        "per_device_kv_bytes_all_layers_batch1": result["all_16_attention_layers_kv_bytes_per_device"],
        "per_device_weight_bytes_all_layers_and_embedding_head": result["all_64_layers_plus_bf16_embedding_head_bytes"],
        "reserved_trace_activation_metadata_bytes": result["reserved_activation_trace_metadata_bytes"],
        "per_device_persistent_auxiliary_bytes": result["persistent_auxiliary_bytes_per_device"],
        "per_device_estimated_resident_bytes": result["estimated_resident_bytes_per_device"],
        "capacity_accounting": "multichip_decoder/capacity_plan.json",
        "largest_tested_prefill": None,
        "largest_tested_decode_context": None,
    }
    (root / "doc/context_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print({k: v for k, v in result.items() if isinstance(v, int)})
