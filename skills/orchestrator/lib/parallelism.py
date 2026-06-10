"""Parallelism planner — derives per-component placement (TP vs replicate) from facts.

The decision the planner encodes, in priority order:

1. Feasibility: if the full replicated model does not fit per chip (with
   headroom for activations/KV cache), sharding is FORCED for the largest
   components regardless of cadence.
2. Cadence: per_token components (decoder stack, LM head) amortize CCL cost
   over thousands of invocations -> shard. per_input components (run-once
   encoders) cannot amortize -> replicate when they fit. Replicating a
   per_input encoder also deletes the boundary CCL into a TP'd decoder,
   since column-parallel inputs must be replicated anyway.
3. Divisibility: q_heads must divide by num_devices for clean head sharding;
   kv_heads < num_devices forces KV replication across device groups.
4. Anything the rules cannot settle is returned as a `judgment` item for the
   architecture worker to decide and record rationale.

Pure Python — no torch, no ttnn.
"""

from __future__ import annotations

DEFAULT_HEADROOM = 0.5  # fraction of per-chip DRAM reserved for activations/KV cache/CBs
# Run-once encoders at/above this sequence length amortize per-layer CCLs
# within the single pass (dots.ocr: 11k-token tower replicate 889ms vs TP4 486ms).
LARGE_ENCODER_TOKENS = 4096


def plan_parallelism(
    components: list[dict],
    num_devices: int,
    dram_bytes_per_device: int,
    headroom: float = DEFAULT_HEADROOM,
) -> dict:
    """components: [{name, cadence: 'per_token'|'per_input', param_bytes,
    q_heads?, kv_heads?}]. Returns {plan: [...], judgments: [...]} where each
    plan row is {name, placement: 'shard'|'replicate', kv_replication?, rationale}.
    """
    budget = dram_bytes_per_device * (1.0 - headroom)
    total = sum(c.get("param_bytes", 0) for c in components)
    fits_replicated = total <= budget

    plan, judgments = [], []
    for c in components:
        name, cadence = c["name"], c.get("cadence")
        if num_devices == 1:
            plan.append({"name": name, "placement": "replicate", "rationale": "single device"})
            continue
        if cadence == "per_token":
            row = {
                "name": name,
                "placement": "shard",
                "rationale": "per-token cadence amortizes CCLs over the decode loop",
            }
            q, kv = c.get("q_heads"), c.get("kv_heads")
            if q and q % num_devices:
                judgments.append(f"{name}: q_heads={q} not divisible by {num_devices} — pad heads or uneven shard")
            if kv and kv < num_devices:
                row["kv_replication"] = num_devices // kv
                row[
                    "rationale"
                ] += f"; kv_heads={kv} < {num_devices} devices -> replicate each KV head x{row['kv_replication']} (chip-local SDPA)"
            plan.append(row)
        elif cadence == "per_input":
            tokens = c.get("production_tokens")
            if not fits_replicated:
                plan.append(
                    {
                        "name": name,
                        "placement": "shard",
                        "rationale": "replicated model exceeds per-chip DRAM budget — sharding forced",
                    }
                )
            elif tokens is not None and tokens >= LARGE_ENCODER_TOKENS:
                # dots.ocr finding: a run-once encoder that is large at
                # production input sizes amortizes its CCLs within the single
                # pass (measured: vision tower 889ms replicate -> 486ms TP4 at
                # 11k tokens). Shard, and let perf A/B confirm.
                plan.append(
                    {
                        "name": name,
                        "placement": "shard",
                        "rationale": f"run-once but large ({tokens} tokens >= {LARGE_ENCODER_TOKENS}); CCLs amortize within one pass — shard, perf phase A/Bs vs replicate",
                    }
                )
            else:
                row = {
                    "name": name,
                    "placement": "replicate",
                    "rationale": "run-once encoder; replication costs idle memory, saves all CCLs incl. the boundary handoff",
                }
                if tokens is None:
                    judgments.append(
                        f"{name}: production_tokens unknown — replicate chosen; supply production input size to confirm (shard wins above {LARGE_ENCODER_TOKENS})"
                    )
                plan.append(row)
        else:
            judgments.append(f"{name}: unknown cadence — label per_token or per_input")
    return {
        "plan": plan,
        "judgments": judgments,
        "fits_replicated": fits_replicated,
        "per_chip_budget_bytes": int(budget),
        "total_param_bytes": total,
    }


# (Resolved dots.ocr blind spot: per_input now carries a size term via
# production_tokens + LARGE_ENCODER_TOKENS above.)
