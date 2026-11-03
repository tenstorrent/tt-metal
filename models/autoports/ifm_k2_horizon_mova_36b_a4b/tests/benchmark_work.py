# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only useful-work/unique-traffic model; never imports TTNN.

Inputs are completed *physical* model calls, not HTTP request counts:
prefill_rows = [{"tokens": 256, "start_pos": 0, "head_tokens": 0}, ...]
decode_steps = [{"positions": [4096, ...], "batch_size": 32}, ...]
Positions enumerate all valid device rows (including repeated slots if executed).
Rows with negative positions are inactive and must be omitted. Each prefill row
is one user's chunk; head_tokens is the actual number sent to model.terminal.
This reports useful FLOPs and a unique-traffic estimate, not physical
DRAM counters. Expert-union ranges never assume random/uniform routes.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "doc/datatype_sweep/selected_default_performance_proof.json"
H, Q, KV, D, V = 2560, 32, 8, 128, 250624
CHIPS, LAYERS, DENSE, SPARSE = 4, 48, 3, 45
TILE_BYTES = {"BFLOAT4_B": 576, "BFLOAT8_B": 1088, "BFLOAT16": 2048, "FLOAT32": 4096}


def _tile_bytes(shape, dtype):
    return (
        math.prod(shape[:-2]) * math.ceil(shape[-2] / 32) * math.ceil(shape[-1] / 32) * TILE_BYTES[dtype.split(".")[-1]]
    )


def _useful(tokens, pairs, head_tokens):
    # Q/K/O/gate, dense V, MoVA router/value4, shared FFN, router, MLP8.
    common = H * (2 * Q * D + KV * D) + Q * D * H
    dense = common + H * KV * D + 3 * H * 6144
    sparse = common + H * 64 + 4 * H * KV * D + 3 * H * 768 + H * 100 + 8 * 3 * H * 768
    components = {
        "dense_layer_projections": 2 * tokens * DENSE * dense,
        "sparse_layer_projections_active_experts": 2 * tokens * SPARSE * sparse,
        "causal_qk_av": 4 * LAYERS * Q * D * pairs,
        "lm_head_unpadded_vocabulary": 2 * H * V * head_tokens,
    }
    return components


def accounting(prefill_rows, decode_steps, slots):
    """Return aggregate mesh work with explicitly labeled lower-bound estimates.

    Neither request count nor slots is enough to infer executed batch/context.
    Missing required per-call metadata raises ValueError. Runtime proof is a
    source for constructed shapes, not a claim these were observed in this run;
    the caller must separately validate the current server against that proof.
    """
    if slots not in (1, 32):
        raise ValueError("Only required one-slot and 32-slot profiles are supported")
    if not isinstance(prefill_rows, (list, tuple)) or not isinstance(decode_steps, (list, tuple)):
        raise ValueError("Per-call records required; aggregate token/step counts are insufficient")
    proof = json.loads(PROOF.read_text())
    runtime = proof["runtime"]
    if runtime["config_id"] != "head4_ccl4_norm4_lofi" or len(runtime["layers"]) != 48:
        raise ValueError("Full48 selected-precision construction proof required")
    prefill = {k: 0 for k in _useful(0, 0, 0)}
    decode = {k: 0 for k in prefill}
    for row in prefill_rows:
        if not {"tokens", "start_pos", "head_tokens"} <= row.keys():
            raise ValueError("Prefill record needs tokens, start_pos, head_tokens")
        n, start, heads = (int(row[k]) for k in ("tokens", "start_pos", "head_tokens"))
        if n <= 0 or start < 0 or start + n > 524288 or not 0 <= heads <= n:
            raise ValueError("Invalid prefill extent")
        for k, value in _useful(n, n * start + n * (n + 1) // 2, heads).items():
            prefill[k] += value
    traffic = {
        k: 0
        for k in (
            "dense_and_shared_projection_weights",
            "expert_weights_minimum_union",
            "expert_weights_maximum_union",
            "kv_read_unique_rounded_1024",
            "kv_update_tile_read_modify_write",
            "norm_gamma_and_router_bias",
            "lm_head_weights",
            "embedding_and_rope_rows",
            "terminal_logits_materialization_and_read",
            "terminal_hidden_normalization",
            "long_capacity_query_spill",
            "page_table_entries_unique",
        )
    }
    n_decode_tokens = 0
    for step in decode_steps:
        if not {"positions", "batch_size"} <= step.keys():
            raise ValueError("Decode record needs positions and batch_size")
        positions, batch = step["positions"], int(step["batch_size"])
        if batch != len(positions) or not 1 <= batch <= slots or any(p < 0 or p >= 524288 for p in positions):
            raise ValueError("Positions must cover every valid row in the actual executed batch")
        n_decode_tokens += batch
        for k, value in _useful(batch, sum(p + 1 for p in positions), batch).items():
            decode[k] += value
        for layer in runtime["layers"]:
            i, manifest = layer["index"], layer["weight_manifest"]
            names = [
                "attention_decode.weight" if i < 3 and slots == 1 else "attention_packed.weight",
                "self_attn.o_proj.weight",
            ]
            names += (
                ["mlp.gate_up.weight", "mlp.down_proj.weight"]
                if i < 3
                else ["mlp.shared_router.weight", "mlp.shared_experts.down_proj.weight"]
            )
            for name in names:
                traffic["dense_and_shared_projection_weights"] += CHIPS * _tile_bytes(
                    layer["decode_shapes"][name], layer["decode_weights"][name]
                )
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                traffic["norm_gamma_and_router_bias"] += CHIPS * _tile_bytes(
                    manifest[name]["local_shape"], manifest[name]["dtype"]
                )
            if i >= 3:
                for name, selected, population in (
                    ("value_experts", 4, 64),
                    ("experts.gate_up", 8, 100),
                    ("experts.down_proj", 8, 100),
                ):
                    weight = manifest[name]
                    per_expert = CHIPS * _tile_bytes(weight["local_shape"][-2:], weight["dtype"])
                    traffic["expert_weights_minimum_union"] += selected * per_expert
                    traffic["expert_weights_maximum_union"] += min(population, selected * slots) * per_expert
                for name in ("self_attn.v_router.bias", "mlp.gate.bias"):
                    traffic["norm_gamma_and_router_bias"] += CHIPS * _tile_bytes(
                        manifest[name]["local_shape"], manifest[name]["dtype"]
                    )
        # BFP8 KV, two global banks (K,V), all eight KV heads. Full-capacity
        # attention uses k_chunk1024; update reads and writes one 32-row tile.
        traffic["kv_read_unique_rounded_1024"] += (
            LAYERS * 2 * KV * D * sum(math.ceil((p + 1) / 1024) * 1024 for p in positions) * 1088 // 1024
        )
        traffic["kv_update_tile_read_modify_write"] += LAYERS * batch * 2 * KV * (D // 32) * 1088 * 2
        traffic["lm_head_weights"] += _tile_bytes([H, 262144], runtime["head"])
        traffic["embedding_and_rope_rows"] += CHIPS * slots * (H + 2 * D) * 2
        # Model.terminal: two head output writes, concatenation read/write,
        # generator persistent-logits copy read/write and sampling read.
        # Full physical32 rows even for B1.
        traffic["terminal_logits_materialization_and_read"] += 6 * 32 * 262144 * 2
        # Terminal input store + two norms read/write + concat read/write;
        # replicated BF16 hidden. This counts five full hidden transfers.
        traffic["terminal_hidden_normalization"] += CHIPS * 5 * 32 * H * 2
        traffic["long_capacity_query_spill"] += 2 * LAYERS * slots * Q * D * 2
        traffic["page_table_entries_unique"] += CHIPS * LAYERS * sum(math.ceil((p + 1) / 64) * 4 for p in positions)
    base = sum(
        v for k, v in traffic.items() if k not in ("expert_weights_minimum_union", "expert_weights_maximum_union")
    )
    peak_lofi = 4 * 110 * 5.4e12
    prefill_total = sum(prefill.values())
    prefill_hifi2 = prefill["causal_qk_av"]
    prefill_ideal_seconds = (prefill_total - prefill_hifi2) / peak_lofi + prefill_hifi2 / (peak_lofi / 2)
    effective_peak = prefill_total / prefill_ideal_seconds if prefill_ideal_seconds else None
    return {
        "mesh_prefill_effective_peak_flops_per_second": effective_peak,
        "prefill_fidelity_weighted_theoretical_seconds": prefill_ideal_seconds,
        "prefill_fidelity_basis": "Selected policy compute_fidelities and lm_head_fidelity LoFi; sdpa_fidelities.prefill HiFi2; peak HiFi2=LoFi/2 per tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:63-69. Weighted useful-work ceiling excludes non-matmul work.",
        "schema_version": 1,
        "decode_collective_ring_link_payload_bytes_estimate": len(decode_steps)
        * 2
        * LAYERS
        * CHIPS
        * 2
        * (CHIPS - 1)
        // CHIPS
        * _tile_bytes([32, H], "BFLOAT4_B"),
        "collective_scope": "Two per-layer replicated all-reduces, BFP4 physical32 rows, ideal ring send payload; not DRAM bytes or a measured network counter",
        "policy": runtime["config_id"],
        "layers": LAYERS,
        "chips": CHIPS,
        "server_slots": slots,
        "prefill_useful_flops": sum(prefill.values()),
        "prefill_flop_components": prefill,
        "decode_useful_flops": sum(decode.values()),
        "decode_flop_components": decode,
        "decode_tokens_in_model_calls": n_decode_tokens,
        "decode_model_calls": len(decode_steps),
        "decode_dram_bytes_estimate": base + traffic["expert_weights_minimum_union"],
        "decode_dram_bytes_estimate_kind": "minimum unique-transfer model with minimum expert union; not measured physical bandwidth",
        "decode_dram_bytes_expert_union_high_scenario": base + traffic["expert_weights_maximum_union"],
        "decode_dram_components": traffic,
        "hardware_reference": {
            "mesh_lofi_peak_flops_per_second": peak_lofi,
            "mesh_prefill_effective_peak_flops_per_second": effective_peak,
            "mesh_dram_peak_bytes_per_second": 4 * 512e9,
            "clock_basis": "nominal 1.35GHz; not measured clock",
            "compute_source": "tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:52-69",
            "core_grid_source": "tt_metal/core_descriptors/blackhole_140_arch.yaml:60-65",
            "dram_source": "tech_reports/CCLs/CCL_bandwidth.md:87",
        },
        "construction_proof": str(PROOF.relative_to(ROOT)),
        "construction_proof_sha256": hashlib.sha256(PROOF.read_bytes()).hexdigest(),
        "assumptions": [
            "Useful FLOPs count multiply-add as two; exclude padding, redundant TP router projection, sparse tile expert unions, routing/nonlinear/reduction operations.",
            "Prefill effective ceiling weights LoFi projections/head and HiFi2 causal QK/AV by their theoretical times; nonlinear and other non-matmul work remains in the full host wall time.",
            "Decode bytes sum all four chips, once per actual model call, using padded constructed weight shapes and BFP tile overhead.",
            "Generator always executes configured max_batch_size: one-slot exactly value4/MLP8;32-slot uses value4..64,MLP8..100 even if only one request is active. No observed routes or uniform-routing assumption.",
            "Upper expert-union scenario is not an upper bound on all physical DRAM traffic; duplicate kernel reads and unmodeled transient transfers may increase it.",
            "Activation traffic includes explicit terminal DRAM materialization, embedding/RoPE rows, query spill and cache/page operations. Layer intermediates predominantly stay in L1; spills, repeated kernel reads and sampler scratch are not measured.",
            "Communication is not DRAM traffic: collective host/device execution and waiting belong in the full wall-time denominator; never add link bytes to DRAM numerator.",
            "No device profiler, synchronization, live device access or routing readback is performed by this host-only calculation.",
        ],
        "exact_physical_accounting_available": False,
    }
