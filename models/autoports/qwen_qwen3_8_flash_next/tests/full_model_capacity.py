"""Full-model persistent DRAM plan, retaining the optimized decoder ledger."""

import json
from pathlib import Path
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_memory_plan import plan, tiled


def full_plan(context=262144, batch=1):
    p = plan(context=context, batch=batch, requests=2 * batch)
    weights = p["weight_breakdown"]
    weights["terminal_hc_mixer"] = 2 * tiled((2560, 320), 2048) + tiled((4, 640), 2048)
    cache = p["cache_breakdown"]
    cache["qsa_inactive_scratch_pages_12"] = 12 * batch * 512 * (512 + 128 * 2)
    cache["sampled_token_history"] = batch * (context + 32) * 4
    buckets = [1 << i for i in range(10 if batch == 1 else 4)]
    cache["ple_compact_norm_gamma_buffers"] = 3 * sum(tiled((4 * batch * n, 640), 4096) for n in buckets)
    cache["sampler_ready_logits"] = tiled((batch, 62080), 2048)
    cache["terminal_ccl_and_sampler_buffers_reserve"] = 8 * 1024**2
    cache["fixed_prefill_token_inputs_and_controls_reserve"] = 256 * 1024
    p["weight_bytes_per_device"] = sum(weights.values())
    p["cache_bytes_per_device"] = sum(cache.values())
    p["total_bytes_per_device"] = (
        sum(weights.values()) + sum(cache.values()) + p["reserved_trace_activation_bytes_per_device"]
    )
    p["allocatable_dram_bytes_per_device"] = 34178731008
    p["remaining_bytes_per_device"] = p["allocatable_dram_bytes_per_device"] - p["total_bytes_per_device"]
    p["caveat"] = (
        "Full-wrapper arithmetic; execution evidence is recorded separately in context_contract.json. "
        "The 1GiB reserve covers 160MB trace plus transient activations/program buffers. No context reduction."
    )
    return p


if __name__ == "__main__":
    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    p = full_plan()
    (root / "doc/full_model/memory_capacity_plan.json").write_text(json.dumps(p, indent=2) + "\n")
    contract = json.loads((root / "doc/context_contract.json").read_text())
    validation = contract.setdefault("full_model_validation", {"status": "in_progress"})
    validation.update(
        {
            "supported_context": 262144,
            "hf_advertised_context": 262144,
            "full_context_batch": 1,
            "capability_reduction": None,
            "weight_bytes_per_device": p["weight_bytes_per_device"],
            "cache_bytes_per_device": p["cache_bytes_per_device"],
            "reserved_trace_activation_bytes_per_device": p["reserved_trace_activation_bytes_per_device"],
            "planned_total_bytes_per_device": p["total_bytes_per_device"],
            "allocatable_dram_bytes_per_device": p["allocatable_dram_bytes_per_device"],
            "memory_plan": "full_model/memory_capacity_plan.json",
        }
    )
    (root / "doc/context_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(p, indent=2))
