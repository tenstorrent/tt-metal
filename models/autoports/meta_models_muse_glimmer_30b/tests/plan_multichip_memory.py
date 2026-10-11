# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Modeled full text-stack budget; no full-model implementation or TT device use."""

import argparse
import json
import math
import os
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ROOT / "doc/multichip_decoder/memory_capacity_plan.json")
parser.add_argument("--optimized", action="store_true", help="Stage5 QKV/O three-reader bank padding")
args = parser.parse_args()
c = json.loads((ROOT / "tests/assets/config.json").read_text())["text_config"]
shapes = {"qkvg": (6656, 2304), "o": (1024, 6656), "gate_up": (6656, 9984), "down": (4992, 6656)}
bank_multiple = {name: 8 * 32 * (3 if args.optimized and name in ("qkvg", "o") else 2) for name in shapes}
weights = {
    name: dict(
        local_shape=list(shape),
        interleaved_bfp4_bytes=shape[0] * shape[1] // 1024 * 576,
        dram_sharded_bfp4_bytes=shape[0]
        * math.ceil(shape[1] / bank_multiple[name])
        * bank_multiple[name]
        // 1024
        * 576,
    )
    for name, shape in shapes.items()
}
layer_count = c["num_hidden_layers"]
sliding_count = c["layer_types"].count("sliding_attention")
snapshot = (
    Path(os.environ.get("HF_HOME", "/mnt/models/huggingface"))
    / "hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
)
index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
nontext = {}
for filename in sorted(set(index.values())):
    with (snapshot / filename).open("rb") as file:
        header = json.loads(file.read(struct.unpack("<Q", file.read(8))[0]))
    for name, metadata in header.items():
        if name == "__metadata__" or name.startswith("model.language_model.") or name == "lm_head.weight":
            continue
        group = ".".join(name.split(".")[:2])
        nontext[group] = nontext.get(group, 0) + 2 * math.prod(metadata["shape"])
per_layer = sum(w["interleaved_bfp4_bytes"] + w["dram_sharded_bfp4_bytes"] for w in weights.values())
terms = {
    "decoder_matrix_weights_both_layouts": layer_count * per_layer,
    "replicated_four_hidden_norms_tiled_and_row_major": layer_count * 4 * (32 * 6656 * 2 + 6656 * 2),
    "replicated_positions_per_layer": layer_count * 131072 * 4,
    "replicated_rope_per_sliding_layer": sliding_count * 2 * 131072 * 128 * 2,
    "reserved_replicated_bf16_text_embedding": c["vocab_size"] * 6656 * 2,
    "reserved_tp4_bf16_text_lm_head": math.ceil(c["vocab_size"] / (4 * 32)) * 32 * 6656 * 2,
    "reserved_text_final_norm_tiled_and_row_major": 32 * 6656 * 2 + 6656 * 2,
    "reserved_replicated_bf16_vision_and_projectors": sum(nontext.values()),
    "reserved_trace_bytes": 1024**3,
    # Four complete full-context BF16 hidden tensors cover caller input, layer
    # input/output and the chunk-join transient; the rest covers chunk-local
    # activations, CCL and allocator slack. Prefill requests are streamed.
    "reserved_activations_ccl_allocator_slack": 8 * 1024**3,
}
# One prepared B32 pair per layer; includes both persistent inputs and outputs.
# Do not assume future full-model sharing, and do not retain every batch signature.
if args.optimized:
    terms["prepared_short_prefill_b32_inputs_outputs"] = layer_count * 2 * 32 * (32 + 64) * 6656 * 2
    terms["prepared_short_prefill_b32_page_tables"] = layer_count * 32 * 4096 * 4
kv_per_layer = 2 * 4096 * 1 * 32 * 128 // 1024 * 1088
kv = layer_count * kv_per_layer
capacity = 8 * 4278190080
result = dict(
    status="modeled full-checkpoint budget; per-layer context validated separately",
    model_id="meta-models/Muse-Glimmer-30B",
    revision="a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
    target="QB2/P300x2",
    mesh=[1, 4],
    tp=4,
    text_layer_count=layer_count,
    sliding_layer_count=sliding_count,
    full_layer_count=layer_count - sliding_count,
    context=131072,
    capability_reduction=None,
    per_layer_context_evidence=[
        f"final_capacity_{'' if args.optimized else 'replicated_'}l{layer}_s{length}.json"
        for layer in (0, 3)
        for length in (131071, 131072)
    ],
    cache_shape_per_tensor_per_request=[4096, 1, 32, 128],
    nontext_checkpoint_header_bf16_reservations=nontext,
    full_context_hidden_tensor_bytes=131072 * 6656 * 2,
    simultaneous_full_context_hidden_tensor_allowance=4,
    cache_dtype="bfloat8_b",
    weights=weights,
    per_device_terms=terms,
    per_device_kv_bytes_per_layer_request=kv_per_layer,
    per_device_kv_bytes_per_stack_request=kv,
    per_device_one_full_context_request_total=sum(terms.values()) + kv,
    per_device_dram_arch_descriptor_bytes=capacity,
    per_device_margin_bytes=capacity - sum(terms.values()) - kv,
    modeled_full_context_requests_with_reservations=(capacity - sum(terms.values())) // kv,
    notes=[
        "All byte cells are modeled including BFP exponent overhead and per-bank padding.",
        "Independent prefill/decode BFP4 matrix copies; tile-paired prefill gate/up replaces unpacked interleaved copy.",
        "Conservative replicated embeddings and per-layer RoPE tables; a later full-model stage can share tables.",
        "52 layers with batch32 full-context nonrolling caches exceed device DRAM. Pooled physical pages must be admitted by total token budget; each request retains context131072.",
        "Nontext weight reservations are computed from checkpoint safetensors headers and conservatively replicated in BF16. No full model, LM head, embedding, vision tower or serving path is implemented in this stage.",
        "The decoder accepts a caller-owned num_pages allocation and replicated logical page tables. Physical-page ownership is independent of logical maximum context.",
    ],
)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            k: result[k]
            for k in (
                "per_device_one_full_context_request_total",
                "per_device_dram_arch_descriptor_bytes",
                "per_device_margin_bytes",
            )
        },
        indent=2,
    )
)
