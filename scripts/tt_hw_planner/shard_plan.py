"""Tensor-parallel shard GUIDANCE for the LLM bring-up agent (not a deterministic shard prescription).

The tool uses Claude Code precisely because the correct shard scheme for a layer is a reasoning problem
(which axis, which collective, expert-parallel vs weight-parallel, how a recurrent state parallelizes)
that cannot be read off tensor shapes or names. So this module does NOT dictate per-weight axes. It
hands the agent the general TP principles + reference implementations to study, and lets the agent
derive the scheme for the component in front of it.

What stays DETERMINISTIC is the graduation FLOW, not the scheme: the gate offers a shard rung for a
graduated compute component, `run_component(mode='shard')` measures gathered-PCC on the mesh, and only
a pass writes the `.py.last_good_sharded` snapshot (a wrong axis/collective fails PCC or times out and
is capped). Correctness is judged by the gate; the scheme is authored by the agent.

The one heuristic kept here is an efficiency filter, not a scheme decision: pure elementwise / lookup
roles (norms, embeddings, rotary tables, activations, biases) shard in NO scheme, so the gate skips
offering them a rung. Everything else is shard-eligible and the agent reasons out how.
"""

from __future__ import annotations

import re
from typing import Optional


_REPLICATE_ONLY = r"(norm|layernorm|rmsnorm|embed|embedding|rotary|\brope\b|activation|act_fn|dropout|scale|gate_norm)"


_TP_PRINCIPLES = (
    "Derive the tensor-parallel scheme for THIS component and let gathered-PCC validate it (a wrong "
    "axis or misplaced collective fails PCC or times out — the gate catches it; if it will not "
    "converge the gate caps the rung and the component stays replicated). General TP principles: "
    "(1) Split large matmul weights, NOT elementwise/lookup tensors — keep norms, embeddings, rotary "
    "tables, biases, and router logits REPLICATED. (2) A projection whose OUTPUT feeds a per-element "
    "op (nonlinearity/gate) is column-parallel: split its OUTPUT features across the TP chips. The "
    "projection that REDUCES back to model dim is row-parallel: split its INPUT features and all_reduce "
    "the partial sums afterward. (3) MoE: prefer EXPERT-parallel — a disjoint subset of experts per "
    "chip, router replicated, combined with an all-to-all / gather over the expert axis (do NOT "
    "column/row-split a single expert). (4) Mamba / SSD: shard the HEAD/CHANNEL dimension of "
    "in_proj / x / B / C / conv1d; keep the state SCAN sequential within each shard; never shard the "
    "time/sequence axis; norm replicated. (5) Anything needing the full hidden dim must all_gather / "
    "all_reduce first. The gathered output MUST equal the single-device golden — placement changes, "
    "math does not. Use ttnn.ShardTensorToMesh(mesh_device, dim=...) to shard and the ttnn collectives "
    "(all_gather / all_reduce / reduce_scatter) to reassemble."
)

_REFERENCE_HINTS = (
    "Reference implementations to STUDY and adapt (read them; don't blind-copy): standard attention / "
    "MLP / lm_head column-row-parallel + collectives in models/tt_transformers/tt/attention.py, "
    ".../mlp.py, .../lm_head.py; expert-parallel MoE in models/tt_transformers/tt/mixtral_moe.py; "
    "tensor-parallel model assembly + mesh mapping in models/tt_transformers/tt/model.py. If this "
    "component is a standard transformer attention/MLP/lm_head, its reference gives the exact scheme; "
    "if it is a novel layer (Mamba mixer, MoE, exotic attention), adapt the nearest reference by the "
    "principles above."
)

_RESIDENCY_PRINCIPLES = (
    "RESIDENCY-FIRST degree derivation (applies when the target is a host-free / trace-capturable model, "
    "i.e. ALL weights must stay RESIDENT on the mesh with no per-layer host streaming). Derive every "
    "parallel DEGREE from the ACTUAL mesh you were given and THIS model's config — never assume a fixed "
    "number. Let C = product of the mesh dims (total chips). "
    "(A) ATTENTION tensor-parallel is capped by the KV structure: with grouped-query attention you can "
    "cleanly head-shard K/V only up to num_key_value_heads, so TP_attn = the largest divisor of C (or of "
    "one mesh axis) that is <= num_key_value_heads; going past it forces KV replication + extra "
    "all_reduce and is usually not worth it. "
    "(B) MoE experts are the memory bulk: shard the ROUTED EXPERTS across as many chips as possible "
    "(expert-parallel). EP = the largest divisor of the mesh capacity left after TP that also divides "
    "n_routed_experts; each chip then holds n_routed_experts/EP WHOLE experts resident, router "
    "replicated, combined by all-to-all / gather over the expert axis. "
    "(C) DP (data-parallel) REPLICATES weights across its chips — it does NOT cut per-chip weight memory, "
    "so it PREVENTS residency. Therefore spend mesh capacity on TP and EP FIRST; give an axis to DP ONLY "
    "if weights already fit resident and you want extra batch throughput. If the model does not fit "
    "resident under TP alone (weights stream per layer), CONVERT the DP capacity into EP: a plan that was "
    "TP=t x DP=d becomes TP=t x EP=(d * <expert-axis already in TP>) so experts spread over all C chips. "
    "(D) A DENSE (non-MoE) model has no expert axis, so residency there comes from higher TP or "
    "pipeline-parallel (layers split across chip-groups), never from DP. "
    "(E) This is mesh- AND model-agnostic — you compute the numbers, they are not baked in. Worked "
    "examples, all from the SAME rule: 4 chips, 2 KV heads, 128 experts -> TP=2 x EP=4 (128%4==0); "
    "an 8-chip 2x4 mesh, 8 KV heads, 64 experts -> TP up to 8 x EP over the rest (or TP=2 x EP=8 if "
    "residency needs the experts spread wider); a dense 2 KV-head model on 4 chips -> TP=2 + "
    "pipeline-parallel for the other axis, no EP, no DP. Always verify mesh-axis divisibility and "
    "n_routed_experts % EP == 0; the gate then validates gathered-PCC."
)


def shard_guidance(component: str, cfg: Optional[dict] = None, goal: Optional[str] = None) -> Optional[dict]:
    """TP guidance for `component`, or None if it is a replicate-only role.

    None  = pure elementwise/lookup role (norm/embedding/rotary/activation/bias) — shards in no scheme,
            so the gate should not offer it a shard rung.
    dict  = shard-eligible: {principles, reference_hints}. The agent reasons out the actual scheme
            (axes, collectives, expert- vs weight-parallel) and the gate validates it by gathered-PCC.

    This intentionally does NOT return per-weight axes: choosing them is the agent's job, not a
    deterministic lookup."""
    name = (component or "").lower()
    if re.search(_REPLICATE_ONLY, name):
        return None
    principles = _TP_PRINCIPLES
    if (goal or "").lower().replace("-", "_") in ("host_free", "residency", "resident", "trace", "traceable"):
        principles = _TP_PRINCIPLES + " " + _RESIDENCY_PRINCIPLES
    return {"principles": principles, "reference_hints": _REFERENCE_HINTS}


def is_shard_eligible(component: str, cfg: Optional[dict] = None) -> bool:
    """True iff the gate should offer a shard rung (any weight-bearing compute layer); False for a
    replicate-only role. Whether sharding actually succeeds is decided later by gathered-PCC, not here."""
    return shard_guidance(component, cfg) is not None


__all__ = [
    "shard_guidance",
    "is_shard_eligible",
]
