# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only source-derived useful work; never measured device transactions.

Input is the phase recorder's immutable host metadata. No torch/TTNN imports,
route reads, synchronization, or profiler. See setup/accounting_design.md.
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H, I, E, K, V = 2048, 512, 256, 8, 248320
NL, NF, TP = 30, 10, 4
BFP8 = 17 / 16  # 1024 mantissa + 64 shared-exponent bytes per tile
SOURCE = "https://tenstorrent.com/hardware/cards; https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md; https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md"


def _contract():
    cfg = json.loads((ROOT / "doc/functional_decoder/hf/config.json").read_text())["text_config"]
    expected = dict(
        hidden_size=H,
        moe_intermediate_size=I,
        num_experts=E,
        num_experts_per_tok=K,
        vocab_size=V,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("Model shape changed; re-audit useful work")
    policy = json.loads((ROOT / "doc/datatype_sweep/selected_precision_config.json").read_text())
    if policy["config_id"] != "baseline_bfp8" or set(policy["weight_groups"].values()) != {"bfp8"}:
        raise ValueError("Precision policy changed; re-audit useful work")
    if policy["kv_cache_dtype"] != "bf16" or policy["recurrent_state_dtype"] != "fp32":
        raise ValueError("State precision changed")


def peak_rates(prefill=None, worker_cores=110):
    """Optimistic nominal hardware ceilings; no sustainable-rate claim.

    Mixed ceiling uses sum(F_i / P_i), i.e. sequential ideal resource service.
    SFPU's optimistic 64 FLOP/cycle assumes MAD (32 lanes, 1 IPC, 2 FLOPs).
    Separate add/multiply and correction/reduction overhead cannot reach it.
    Matrix LoFi=4096 FLOP/cycle/core; HiFi2/4 require 2/4 phases.
    """
    if not isinstance(worker_cores, int) or not 0 < worker_cores <= 140:
        raise ValueError("Invalid observed worker core count")
    matrix = TP * worker_cores * 1.35e9 * 4096
    rates = dict(LoFi=matrix, HiFi2=matrix / 2, HiFi4=matrix / 4, SFPU_FP32=TP * worker_cores * 1.35e9 * 64)
    result = dict(
        source=SOURCE,
        chips=TP,
        worker_cores_per_chip=worker_cores,
        nominal_clock_hz=1.35e9,
        per_fidelity_flops_per_second=rates,
        dram_bytes_per_second=TP * 512e9,
        prefill_flops_per_second=matrix,
        peak_method="Unweighted LoFi capability only until a prefill workload is supplied",
    )
    if prefill is not None:
        work = prefill["flops_by_fidelity"]
        ideal = sum(value / rates[key] for key, value in work.items())
        result.update(
            prefill_flops_per_second=prefill["flops"] / ideal,
            prefill_ideal_compute_seconds=ideal,
            ideal_seconds_by_fidelity={key: value / rates[key] for key, value in work.items()},
            peak_method="Work-weighted harmonic ceiling: total useful FLOPs / sum(FLOPs_by_fidelity / peak_by_fidelity); sequential optimistic service, not measured sustainable compute",
        )
    return result


def prefill_work(chunks):
    """Useful logical graph FLOPs across all four chips and 40 layers.

    Terminal head runs once per serving row/chunk, not once per input token.
    Chunk64 internal logical matrices use their actual selected algorithm;
    padding-only extra operations are reported separately and not rewarded.
    """
    _contract()
    chunks = list(chunks)
    if not chunks or any(not 0 <= c["start"] < c["end"] <= 262144 for c in chunks):
        raise ValueError("Invalid prefill chunks")
    tokens = sum(c["end"] - c["start"] for c in chunks)
    terms = {}

    def add(name, flops, fidelity, formula):
        terms[name] = dict(flops=flops, fidelity=fidelity, formula=formula)

    # Linear pack: Q2048,K2048,V4096,Z4096,A32,B32. No zero padding columns.
    add(
        "linear_input_ordered_projection",
        NL * tokens * 2 * H * 12352,
        "SFPU_FP32",
        "30*T*2*2048*12352; TP4 partitioned, no factor4",
    )
    add("linear_output_projection", NL * tokens * 2 * 4096 * H, "HiFi4", "30*T*2*4096*2048; prefill native BF16/HiFi4")
    add("linear_convolution", NL * tokens * 8192 * 7, "SFPU_FP32", "30*T*8192*(4 multiply+3 add)")
    logical_recurrence, padded_recurrence = 0, 0

    def graph(s):
        # 8 matrix products in Chunk64StateGraph + Chunk64OutputGraph:
        # kb*kT; A*vb; A*(kb*exp); tk*state; wkT*vnew;
        # q*kT; (q*exp)*state; attn*vnew.
        matrices = 10 * s * s * 128 + 6 * s * 128 * 128
        triangular = sum(2 * j * j for j in range(1, s))
        # Useful scalar/vector algebra, counting transcendental calls as one
        # operation-equivalent: gate cumsum, masks/decay, beta/scales, state add.
        elementwise = 10 * s * 128 + 6 * s * s + 2 * 128 * 128 + 12 * s
        return 32 * (matrices + triangular + elementwise)

    internal_chunks = []
    for c in chunks:
        length = c["end"] - c["start"]
        for offset in range(0, length, 64):
            n = min(64, length - offset)
            logical_recurrence += graph(n)
            padded_recurrence += graph(64)
            internal_chunks.append(n)
    add(
        "linear_chunk64_state_and_output",
        NL * logical_recurrence,
        "SFPU_FP32",
        "30*32heads*[10*s^2*128+6*s*128^2+2*sum(j^2,j=1..s-1)+elementwise]; eight corrected FP32 matrix products",
    )
    add(
        "linear_qk_l2_output_norm_and_gate",
        NL * tokens * (4 * 4096 + 6 * 4096 + 6 * 8192 + 6 * 32),
        "SFPU_FP32",
        "Q/K L2, gated output RMSNorm, convolution SiLU and beta/gate scalar operation-equivalents",
    )
    # Four ranks each own four Q heads and duplicate one of the two KV heads.
    add(
        "full_attention_input_projection",
        NF * tokens * 2 * H * (4 * 2560),
        "HiFi2",
        "10*T*2*2048*(4ranks*2560cols), includes paired-rank KV duplication",
    )
    add("full_attention_output_projection", NF * tokens * 2 * 4096 * H, "HiFi2", "10*T*2*4096*2048")
    pairs = sum((c["end"] * (c["end"] + 1) - c["start"] * (c["start"] + 1)) // 2 for c in chunks)
    add(
        "full_attention_qk_av",
        NF * 4 * 16 * 256 * pairs,
        "HiFi4",
        "10*4*16heads*256*sum causal attended positions; QK and AV",
    )
    add(
        "full_attention_softmax",
        NF * 16 * pairs * 5,
        "SFPU_FP32",
        "10*16heads*causal_pairs*5 operation-equivalents; subtract/exp/sum/divide/scale",
    )
    add(
        "full_attention_norm_rope_output_gate",
        NF * tokens * (5 * (4096 + 1024) + 6 * (16 + 4) * 64 + 5 * 4096),
        "SFPU_FP32",
        "Q/K norms including duplicate K, 64dim rotary, output sigmoid gate",
    )
    add(
        "routed_expert_gemm",
        40 * tokens * K * 6 * H * I,
        "LoFi",
        "40*T*8experts*3projections*2*2048*512; EP4 partitions experts",
    )
    add(
        "shared_expert_gemm_and_replicated_scalar",
        40 * tokens * (6 * H * I + TP * 2 * H),
        "HiFi2",
        "shared TP4 gate/up/down plus one scalar gate duplicated4",
    )
    add(
        "router_native_replicated",
        40 * tokens * TP * 2 * H * E,
        "HiFi4",
        "40*T*4ranks*2*2048*256 FP32 router native prefill",
    )
    add(
        "expert_nonlinearity_and_combine",
        40 * tokens * (K * (5 * I + 2 * H) + 5 * I + 5 * H + TP * 5 * K),
        "SFPU_FP32",
        "SiLU+multiply, score weight/reduce, shared sigmoid, route-score softmax; top-k comparisons excluded",
    )
    add(
        "layer_norms_and_residuals",
        40 * tokens * TP * (2 * (4 * H + 3) + 2 * H),
        "SFPU_FP32",
        "two replicated RMSNorms and two residual adds per layer",
    )
    add(
        "terminal_lm_head",
        len(chunks) * 2 * H * V,
        "HiFi2",
        "one final-row vocabulary-parallel head per serving row/chunk",
    )
    add(
        "terminal_norm",
        len(chunks) * TP * (4 * H + 3),
        "SFPU_FP32",
        "one replicated terminal RMSNorm per serving row/chunk",
    )
    by_fidelity = defaultdict(int)
    for term in terms.values():
        by_fidelity[term["fidelity"]] += term["flops"]
    return dict(
        flops=sum(by_fidelity.values()),
        flops_by_fidelity=dict(by_fidelity),
        terms=terms,
        input_tokens=tokens,
        terminal_rows=len(chunks),
        causal_attention_pairs=pairs,
        linear_internal_chunk_lengths=internal_chunks,
        excluded_chunk64_padding_flops=NL * (padded_recurrence - logical_recurrence),
        scope="Four-chip useful graph arithmetic, all40layers, actual chunk positions; logical shapes, not issued instruction count",
        limitations=[
            "Transcendentals count as one operation-equivalent; actual polynomial/correction instructions excluded",
            "Top-k comparisons, sampling, CCL, casts, padding and host work are timed but not credited as useful FLOPs",
            "Corrected SFPU matmul rounding corrections excluded from useful MAC counts",
            "Causal SDPA masked/padded tile overhead excluded",
            "Weights and router replicated only where source requires; no blanket factor4",
        ],
    )


def decode_work(members, trace_key):
    """Logical streaming-byte estimate with route union bounds per decode call."""
    _contract()
    members = list(members)
    if not members or any(not 0 <= m["position"] < 262144 for m in members):
        raise ValueError("Invalid decode members")
    if len({m["slot"] for m in members}) != len(members):
        raise ValueError("Duplicate active slot")
    key = trace_key["key"] if isinstance(trace_key, dict) else trace_key
    server_slots = trace_key.get("server_slots") if isinstance(trace_key, dict) else None
    if key == "model":
        physical = 1
    elif key == "model_solo0":
        physical = 2
    elif key == "model_fixed":
        physical = server_slots
    elif key.startswith("model_") and key[6:].isdigit():
        physical = int(key[6:])
    else:
        raise ValueError(f"Unknown model trace {key}")
    n = len(members)
    if physical is not None and max(m["slot"] for m in members) >= physical:
        raise ValueError("Active slot outside trace")
    # Scalar indexed linear decode orders the BF16 router. Full-attention
    # FusedDecoder.linear returns from its native branch before ordered routing.
    # Batched _moe does not set ordered_router, hence FP32 native router.
    indexed = physical == 1
    router_bytes = 2 if indexed else 4
    expert_one = 3 * H * I * BFP8
    unions = dict(lower=K, expected=E * (1 - (1 - K / E) ** n), upper=min(K * n, E))
    terms = {
        # Linear scan loops rows: there is no across-row batched weight reuse.
        "linear_projection_streams_bf16": NL * n * (H * 12352 + 4096 * H) * 2,
        "linear_conv_taps_fp32": NL * n * 4 * 8192 * 4,
        "linear_selected_decay_values_fp32": NL * n * 32 * 4,
        "linear_recurrent_state_read_write_fp32": NL * n * 2 * 32 * 128 * 128 * 4,
        "linear_convolution_history_read_write_bf16": NL * n * 2 * 3 * 8192 * 2,
        "full_projection_once_per_batch_bfp8": NF * (H * TP * 2560 + 4096 * H) * BFP8,
        "full_kv_read_bf16_duplicated_heads": NF * sum(m["position"] + 1 for m in members) * TP * 2 * 256 * 2,
        "full_kv_write_bf16_duplicated_heads": NF * n * TP * 2 * 256 * 2,
        "shared_expert_weights_once_per_batch_bfp8": 40 * (3 * H * I + TP * H) * BFP8,
        "router_weights_replicated": (NL * router_bytes + NF * 4) * TP * H * E,
        "layer_norm_weights_fp32_replicated": 40 * TP * 2 * H * 4,
        "full_qk_norm_weights_fp32_replicated": NF * TP * 2 * 256 * 4,
        "terminal_norm_weights_fp32_replicated": TP * H * 4,
        "lm_head_once_per_batch_bfp8": H * V * BFP8,
        "embedding_row_bf16_sharded": n * H * 2,
        "layer_residual_boundary_read_write_bf16": 40 * n * TP * 2 * H * 2,
        "terminal_input_read_bf16_replicated": n * TP * H * 2,
        "logits_write_and_sampler_read_bf16": n * V * 2 * 2,
        "sampled_token_write_read_uint32": n * TP * 4 * 2,
    }
    useful_terms = dict(terms)
    useful_fixed = sum(useful_terms.values())
    if physical is None:
        raise ValueError("model_fixed requires {'key': key, 'server_slots': observed_capacity}")
    attention_rows = 1 if key == "model_solo0" else physical
    # Fixed-slot masks suppress state writes and expert routes, but not linear
    # projection, convolution or recurrent reads/computation. Charge those rows.
    for name in ("linear_projection_streams_bf16", "linear_conv_taps_fp32", "linear_selected_decay_values_fp32"):
        terms[name] *= attention_rows / n
    terms["linear_projection_packed_zero_columns_bf16"] = NL * attention_rows * H * (12544 - 12352) * 2
    terms["linear_recurrent_state_read_write_fp32"] = NL * (attention_rows + n) * 32 * 128 * 128 * 4
    terms["linear_convolution_history_read_write_bf16"] = NL * (attention_rows + n) * 3 * 8192 * 2
    for name in (
        "embedding_row_bf16_sharded",
        "layer_residual_boundary_read_write_bf16",
        "terminal_input_read_bf16_replicated",
    ):
        terms[name] *= physical / n
    terms.pop("logits_write_and_sampler_read_bf16")
    # Physical terminal tile and sampler carrier have32rows, even on B1.
    # One terminal write/read plus sampler-buffer write/read; no intermediate
    # padded-concat copies or sampler sorting/intermediate traffic inferred.
    terms["terminal_logits_and_sampler_staging_fixed32_bf16"] = 4 * 32 * V * 2
    terms["sampled_token_write_read_uint32"] = 32 * TP * 4 * 2
    fixed = sum(terms.values())
    route_bytes = {bound: 40 * union * expert_one for bound, union in unions.items()}
    terms["routed_expert_expected_stream_bfp8"] = route_bytes["expected"]
    return dict(
        dram_bytes=sum(terms.values()),
        dram_bytes_bounds={k: fixed + v for k, v in route_bytes.items()},
        terms=terms,
        accounting_basis="Executed-row logical streaming scenario; includes dormant attention read/scan work, packed linear zero columns and fixed32 sampler staging. Still not physical transaction counters",
        useful_active_scenario=dict(
            dram_bytes=useful_fixed + route_bytes["expected"],
            dram_bytes_bounds={k: useful_fixed + v for k, v in route_bytes.items()},
            terms={**useful_terms, "routed_expert_expected_stream_bfp8": route_bytes["expected"]},
        ),
        active_requests=n,
        positions=[m["position"] for m in members],
        trace_key=key,
        physical_moe_rows=physical,
        physical_attention_rows=1 if key == "model_solo0" else physical,
        expert_path="indexed_TP4" if indexed else "grouped_EP4",
        expert_union_model=unions,
        expert_payload_bytes_per_layer=expert_one,
        assumptions=[
            "Expected expert union =256*(1-(31/32)^active_requests), hypothetical independent uniform8-distinct experts per row; no observed routes",
            "Bounds cover possible unique expert union under ideal one-stream-per-selected-expert reuse, not total physical traffic bounds",
            "Primary executed-row estimate charges linear ordered weights per physical attention row; useful_active_scenario retains active-only comparison; native full/shared/router/head weights once per call",
            "KV includes newly written token in its read and separately its write; paired-rank duplication included",
            "No cross-step weight retention assumed; indexed and EP resident copies never summed",
            "Primary includes inactive linear projection/conv/state reads, packed projection zero columns and fixed32 terminal/sampler boundaries; excludes additional tiling, intermediate rereads, CCL, sorting traffic and LUT cache-line amplification",
            "FP32 norm gamma assumed streamed unless selected linear output gamma is resident L1 (excluded)",
            "Weights may be retained in L1; logical streaming estimate is not guaranteed DRAM traffic or a measured transaction count",
        ],
    )


if __name__ == "__main__":
    p = prefill_work([dict(start=0, end=4096)])
    d = decode_work([dict(slot=i, position=4096) for i in range(8)], dict(key="model_fixed", server_slots=8))
    print(
        json.dumps(
            dict(
                prefill_flops=p["flops"],
                prefill_by_fidelity=p["flops_by_fidelity"],
                peaks=peak_rates(p),
                decode_bytes=d["dram_bytes"],
                decode_bounds=d["dram_bytes_bounds"],
            ),
            indent=2,
        )
    )
