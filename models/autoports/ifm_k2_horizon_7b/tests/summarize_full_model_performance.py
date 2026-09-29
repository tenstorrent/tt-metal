"""CPU-only accounting from selected full-stack timing and reduced device evidence."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--layers", type=Path, required=True)
    args = parser.parse_args()
    timing_path = args.doc / "performance_selected.json"
    policy_path = args.doc / "selected_policy.json"
    timing = json.loads(timing_path.read_text())
    policy = json.loads(policy_path.read_text())
    stack = json.loads(args.stack.read_text())
    spans = json.loads(args.spans.read_text())
    layers = json.loads(args.layers.read_text())

    # Tile storage includes the BFP exponent metadata (64 bytes per 1024 values).
    bytes_per_tile = {"bfloat8_b": 1088, "bfloat4_b": 576, "bfloat16": 2048}
    weights = 0
    for layer in policy["weight_allocations"]:
        for _role, mode, shape, dtype in layer:
            if mode == "decode":
                assert all(d % 32 == 0 for d in shape)
                weights += shape[0] * shape[1] // 1024 * bytes_per_tile[dtype]
    head = 4096 * 65536 // 1024 * bytes_per_tile[policy["head_dtype"]]
    # Position128..254 has a 256-token read window at K-chunk128.
    kv_window = 256
    kv = 36 * 2 * 2 * 128 * kv_window // 1024 * bytes_per_tile["bfloat8_b"]
    embedding = 1024 * 2
    final_norm = 32 * 1024 * 2  # Tiled BF16 final gamma; decoder gammas are folded into weights.
    bytes_per_rank = weights + head + kv + embedding + final_norm
    bandwidth_per_rank = 512e9  # tt-perf-report1.3.0 blackhole architecture table
    roofline_ms = bytes_per_rank / bandwidth_per_rank * 1000
    e2e = timing["selected"]["decode_ms_per_token"]
    layer_floor = layers["weighted_stack_minimum_ms"]
    terminal = layers["terminal"]["median_us"] / 1000
    embedding_ms = stack["maximum_rank_us"]["embedding_rope_us"] / 1000
    estimate = layer_floor + terminal + embedding_ms
    result = {
        "workload": {"profile": "single_user_decode", "prompt_len": 128, "gen_len": 128, "batch": 1},
        "ttft_ms": timing["selected"]["ttft_seconds"] * 1000,
        "decode_ms_per_token_e2e": e2e,
        "decode_ms_per_token_device": None,
        "device_time_scope": "No full36 device profile, per optimize skill. Actual reduced-path span and weighted full36 estimate are separate fields below.",
        "reduced_two_layer_token_out_device_ms": spans["token_out"]["device_us"] / 1000,
        "profiled_full36_fw_span_ms_estimate": stack["stack_plus_terminal_ms"],
        "profiled_layer_stack_ms_estimate": stack["selected_policy_stack_ms"],
        "full36_component_ms_estimate": estimate,
        "layer_stack_lower_estimate_ms": layer_floor,
        "layer_stack_median_estimate_ms": layers["weighted_stack_median_ms"],
        "terminal_norm_head_split_sampler_ms_measured": terminal,
        "embedding_rope_ms_profiled": embedding_ms,
        "e2e_minus_stack_ms": e2e - layer_floor,
        "e2e_over_stack_plus_terminal_fraction": e2e / estimate - 1,
        "unexplained_positive_gap_fraction": max(0, e2e / estimate - 1),
        "accounting_scope": "Layer-only lower estimate uses minimum of five unprofiled256-replay trials for each selected class, weighted9/27. Terminal is independently traced without profiler. Embedding/RoPE uses per-rank profiled span. Position advance, history and final output are excluded from component estimate. This representative-layer sum is an estimate, not a mathematical lower bound or a full36 device timeline; isolation and full-chain scheduling/allocation differ.",
        "component_evidence_reuse": "The isolated layer/terminal measurements precede prefill-only geometry overrides. Decode and terminal bodies, numerical policy and measured source hashes are unchanged in that evidence; the final default benchmark and reduced profiles are refreshed.",
        "roofline_ms_per_token_estimate": roofline_ms,
        "roofline_inputs": {
            "mesh_ranks": 4,
            "per_rank_decode_weight_bytes": weights,
            "per_rank_head_weight_bytes": head,
            "per_rank_kv_read_bytes": kv,
            "kv_read_window_tokens": kv_window,
            "per_rank_embedding_row_bytes": embedding,
            "per_rank_final_norm_weight_bytes": final_norm,
            "per_rank_total_bytes": bytes_per_rank,
            "aggregate_dram_bandwidth_bytes_per_second": 4 * bandwidth_per_rank,
            "source": "tt-perf-report1.3.0 ArchitectureSpec blackhole, 8 DRAM banks; selected_policy.json stored decode weights",
            "scope": "Ideal one-read weight/KV traffic floor, excluding repeats, intermediates, CCL and launch latency. Decoder gammas are folded into projection weights. Not a prediction of achievable latency.",
        },
        "named_limitations": [
            f"Instrumented FW estimate/full36 ratio={stack['stack_plus_terminal_ms'] / e2e:.5f}; isolated component/full36 ratio={estimate / e2e:.5f}. Do not interpret a negative difference as negative overhead or an optimization credit.",
            "Hundreds of small sharded ops, CCL and physical tile32 work lower achieved bandwidth versus the ideal traffic floor.",
            "The honored AG startup barrier is required to order ready-semaphore reuse across the buffered argmax/model transition.",
        ],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [timing_path, policy_path, args.stack, args.spans, args.layers]
        },
    }
    assert timing["pass"] and timing["greedy_strategies_same_tokens"]
    assert layers["pass"] and all(row["eager_and_replay_exact_all_ranks"] for row in layers["records"])
    assert layers["terminal"]["eager_and_replay_exact_all_ranks"]
    assert abs(e2e / estimate - 1) < 0.05, "Reconcile component measurement with full-chain timing"
    assert result["e2e_over_stack_plus_terminal_fraction"] <= 0.15, "Investigate avoidable full-path gap"
    output = args.doc / "perf_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
