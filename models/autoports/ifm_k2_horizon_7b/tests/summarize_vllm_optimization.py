"""Derive stage10 comparisons from preserved, unmodified runner JSON."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "doc/optimized_vllm"
    manifests = [json.loads((root / name).read_text()) for name in ("before_manifest.json", "final_manifest.json")]
    before_server, after_server = [m["servers"][0] for m in manifests]
    assert before_server["argv"] == after_server["argv"]
    assert before_server["environment"] == after_server["environment"]
    changed_runtime_sources = [
        path for path, sha in manifests[0]["code_sha256"].items() if manifests[1]["code_sha256"][path] != sha
    ]
    assert {Path(p).name for p in changed_runtime_sources} == {"generator.py", "generator_vllm.py"}
    result = dict(same_server_argv_and_environment=True, changed_runtime_sources=changed_runtime_sources, profiles={})
    for profile, name in (("primary", "vllm_benchmark.json"), ("ci_burst", "vllm_ci_serving_benchmark.json")):
        rows = {}
        for phase in ("before", "candidate", "after"):
            path = root / phase / name
            data = json.loads(path.read_text())
            assert data["missing_output_tokens"] == 0
            assert data["completed_requests"] == data["config"]["num_requests"]
            rows[phase] = {
                k: data[k]
                for k in (
                    "config",
                    "completed_requests",
                    "total_input_tokens",
                    "total_output_tokens",
                    "ttft_ms",
                    "tpot_ms",
                    "itl_ms",
                    "output_throughput_tok_per_s",
                    "vllm_mean_tpot_decode_tps",
                )
            }
            rows[phase].update(artifact=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        assert rows["before"]["config"] == rows["candidate"]["config"] == rows["after"]["config"]
        result["profiles"][profile] = rows
    selected = result["profiles"]["primary"]["after"]
    prior = result["profiles"]["primary"]["before"]
    result["primary_ttft_reduction_percent"] = 100 * (1 - selected["ttft_ms"]["p50"] / prior["ttft_ms"]["p50"])
    result["primary_decode_rate_change_percent"] = 100 * (
        selected["vllm_mean_tpot_decode_tps"] / prior["vllm_mean_tpot_decode_tps"] - 1
    )
    full_model = json.loads((root.parent / "datatype_sweep/post_selection_token_out.json").read_text())
    full_model_tps = full_model["selected"]["decode_tokens_per_second_per_user"]
    result["full_model_comparison"] = dict(
        source="../datatype_sweep/post_selection_token_out.json",
        workload="actual128/128/B1, cache256, selected datatype, traced split sampling, buffered output",
        decode_tokens_per_second_per_user=full_model_tps,
        zero_per_token_readback_tps=full_model["token_out_no_readback"]["tokens_per_second_per_user"],
        serving_slower_percent=100 * (1 - selected["vllm_mean_tpot_decode_tps"] / full_model_tps),
        limitation="Serving actual127/128/1 uses vLLM-owned maximum-context cache, one per-step output read and HTTP scheduling; comparable, not identical timing boundaries.",
    )
    # Stage7 one-read traffic floor, adjusted for the selected11 BFP4 down layers.
    prior_roofline = json.loads((root.parent / "optimized_full_model/perf_summary.json").read_text())["roofline_inputs"]
    down_saving = 11 * (3072 * 4096 // 1024) * (1088 - 576)
    per_rank_bytes = prior_roofline["per_rank_total_bytes"] - down_saving
    roofline_ms = per_rank_bytes * 4 / prior_roofline["aggregate_dram_bandwidth_bytes_per_second"] * 1000
    perf = dict(
        workload=dict(
            profile="single_user_decode",
            prompt_len=128,
            actual_prompt_len=127,
            gen_len=128,
            batch=1,
            concurrency=1,
            max_num_seqs=32,
            max_model_len=524288,
            mesh="P300x2 / 1x4",
        ),
        ttft_ms=selected["ttft_ms"]["p50"],
        decode_ms_per_token_e2e=selected["tpot_ms"]["mean"],
        decode_ms_per_token_device=None,
        device_profile=None,
        roofline_ms_per_token_estimate=roofline_ms,
        roofline_basis=dict(
            prior_source="../optimized_full_model/perf_summary.json",
            per_rank_bytes=per_rank_bytes,
            selected_down_weight_saving_bytes=down_saving,
            kv_window_tokens=256,
            scope="Ideal one-read traffic estimate with BFP8/BFP4 tile headers; excludes intermediates, CCL, repeats and dispatch. Not a new device measurement.",
        ),
        named_limitations=[
            "vllm_serving_profiler_disabled_to_protect_hardware",
            "Primary profile has one measured request; TTFT/TPOT P99 are not population tails.",
            "CI burst includes scheduler admission and shape transitions and is secondary capacity evidence.",
            "Only single-request start-zero lengths1..4096 have prefill traces; other valid requests preserve eager/chunked support.",
            "Full sampling suite uses explicit host compatibility; strict device performance mode has no host fallback.",
        ],
    )
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "perf_summary.json").write_text(json.dumps(perf, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "primary_ttft_reduction_percent",
                    "primary_decode_rate_change_percent",
                    "full_model_comparison",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
