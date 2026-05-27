# MiniMax M2.7 Prefill Bringup Design (TTNN)

## Goal and Scope

This note defines the initial execution strategy for bringing up **MiniMax M2.7 prefill** on Tenstorrent using TTNN ops.

In scope:
- Prefill-only path (decode remains out of scope for this phase).
- Correctness-first bringup order for module and model integration.
- Parallelization scheme and first-pass performance estimates.

Out of scope:
- Production decode path.
- Final kernel-level micro-optimizations.
- vLLM productionization details beyond prefill compatibility.

## Current Bringup Snapshot

Based on the current `minimax_m27` tree and branch state:
- Real weight loading and cache machinery are present (`utils/weight_config.py`, `utils/hf_model_utils.py`).
- Embedding and LM head modules have dedicated MiniMax tests and appear aligned with your reported PCC progress.
- The broader stack still contains significant `deepseek_v3` namespace coupling in runtime and test imports; this is a key prefill integration risk and must be cleaned before scaling bringup.

## Proposed Prefill Parallelization Scheme

### Mesh Mapping

Use a 2D mesh shape `(rows, cols)` (for example TG: `4 x 8`):
- **Axis 1 (cols)**: tensor parallel dimension for feature-projection work (Q/K/V/O, expert projections, lm_head shard).
- **Axis 0 (rows)**: sequence/user-distribution and MoE dispatch dimension (including all-to-all style expert routing steps where applicable).

This matches the structure in current `RowBatchedModel` + `Embedding2D` + `MLA2D` + decoder stack and is the shortest path to a full prefill baseline.

### Prefill Dataflow (High-Level)

1. **Embedding**
   - Token IDs replicated to mesh.
   - Embedding projection + collective normalization of shard outputs.

2. **Per-decoder-layer block**
   - Distributed RMSNorm.
   - MLA prefill path:
     - Sequence/activation collective prep.
     - TP-local projection work.
     - RoPE + cache update + flash MLA prefill.
     - Output collective to restore expected shard layout.
   - Residual add.
   - Post-attention norm.
   - MoE path:
     - MoE gate + dispatch + experts + combine.
   - Residual add.

3. **Final norm + LM head**
   - Distributed final RMSNorm.
   - Gather to lm_head shard layout and run vocab projection.

### Why Row-Batched First

Use `RowBatchedModel` as the first full-model bringup target:
- It is already wired into the current prefill flow and tests.
- It minimizes parallel scheduling complexity while correctness is still moving.
- It keeps optimization options open (including row-pipelined follow-up) after baseline confidence is established.

`RowPipelinedModel` should be treated as a phase-2 optimization branch, not a correctness dependency.

## Bringup Order

Use the detailed tracker in `prefill_bringup_sheet.csv`, but enforce this strict dependency chain:

1. Config + namespace hygiene (remove DeepSeek coupling from MiniMax path).
2. Core correctness blocks (norm, MLA, MoE gate/experts).
3. Decoder block integration.
4. Full row-batched model prefill pass.
5. Generator/demo wiring.
6. Perf instrumentation and optimization.

Do not start perf tuning until step 4 is stable with real weights.

## First-Order Performance Model

The numbers below are rough, compute-centric estimates for planning. They are intended for prioritization and target setting, not as final benchmark commitments.

Assumed model constants (from MiniMax M2.7 config / code paths):
- `num_hidden_layers = 62`
- `hidden_size = 3072`
- `num_attention_heads = 48`
- `head_dim = 128`
- `num_experts_per_tok = 8`
- `num_local_experts = 256`
- `intermediate_size = 1536`

Per-layer prefill FLOP approximation at sequence length `L`:

```text
F_layer(L) ~= L * (F_attn_proj + F_moe + F_gate) + F_attn_quadratic(L)

F_attn_proj      ~=  88.08M FLOPs/token
F_moe            ~= 226.49M FLOPs/token
F_gate           ~=   1.57M FLOPs/token
F_attn_quadratic ~= 4 * num_heads * L^2 * head_dim
```

Equivalent per-layer totals:
- `L=128`: 0.0409 TF
- `L=512`: 0.1683 TF
- `L=2048`: 0.7505 TF
- `L=4096`: 1.7072 TF
- `L=8192`: 4.2391 TF

Full-model totals (`62` layers):
- `L=128`: 2.53 TF
- `L=512`: 10.44 TF
- `L=2048`: 46.53 TF
- `L=4096`: 105.85 TF
- `L=8192`: 262.83 TF

At `L=4096`, estimated compute share is roughly:
- MoE experts: ~54%
- Attention quadratic term: ~24%
- Attention projections: ~21%
- MoE gate: <1%

This indicates that early optimization effort should focus on:
1) MoE expert path and all-to-all overhead, then
2) long-sequence attention behavior and collective overlap.

## Rough Latency Bands (Planning)

If sustained effective throughput lands in a **120-250 TF/s** envelope for full prefill execution, rough latency bands are:
- `L=128`: ~10-21 ms
- `L=512`: ~42-87 ms
- `L=2048`: ~186-388 ms
- `L=4096`: ~423-882 ms
- `L=8192`: ~1051-2190 ms

These are prefill-kernel dominated estimates and do not include one-time compile cost, host orchestration variance, or full end-to-end service overhead.

## Initial Perf Targets (Bringup Phase)

For the first pass (post-correctness, pre-decode):
- Stable prefill on real weights at `L=128/512/2048/4096`.
- No OOM/hangs across long-seq chunk boundaries.
- Stage-level latency breakdown available per run:
  - embedding
  - decoder MLA
  - decoder MoE
  - final norm
  - lm_head
  - collectives (aggregate)

## Key Risks and Mitigations

- **Namespace coupling risk**
  - Risk: `deepseek_v3` imports in MiniMax runtime path can cause hidden behavior drift.
  - Mitigation: complete namespace scrub before full-model validation.

- **Config field mismatch risk**
  - Risk: mixed use of `num_local_experts` vs `n_routed_experts`-style fields.
  - Mitigation: enforce one MiniMax config contract check in fixtures.

- **Collective ordering and semaphore risk**
  - Risk: subtle deadlocks/incorrectness under stress.
  - Mitigation: deterministic CCL ordering tests + long-seq regression sweep.

- **MoE dispatch/combine pressure**
  - Risk: dominant runtime and memory traffic hotspot.
  - Mitigation: instrument first, then tune all-to-all and sparse path systematically.

## Next Step

Execute Milestone M1 and M2 tasks in `prefill_bringup_sheet.csv` first, then lock a full-model `L=128` real-weight prefill baseline before any heavy performance tuning.
