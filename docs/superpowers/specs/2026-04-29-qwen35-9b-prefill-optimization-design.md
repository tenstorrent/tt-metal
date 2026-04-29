# Qwen3.5-9B Prefill Optimization Plan (Blackhole P150)

**Date**: 2026-04-29
**Author**: brainstorming session, atupe@tenstorrent.com
**Status**: design — pending implementation plan

## Context & baseline

A Tracy profile of `traced_4k` (4096-token prefill, 4 layers profiled = one hybrid block of 3 linear-attention + 1 full-attention) gives a per-block kernel time of **422 ms**, extrapolating to **~3.4 s** for the full 32-layer model.

Distribution (extrapolated to one full forward pass):

| Op family | ms / pass | % | Mean FPU% |
|-----------|----------:|--:|----------:|
| GDN custom kernel (`GenericOpDeviceOperation`)   | ~2 050 | 60.7 | not reported |
| Matmul (MLP + attention projections)             | ~ 590  | 17.5 | **11–15 %** (catastrophically low) |
| Reshape / Tilize / Untilize / Slice / Concat     | ~ 470  | 14.0 | mostly 0 |
| Binary / Unary (residual, silu, mul, scaling)    | ~ 175  | 5.2  | 0 |
| LayerNorm                                        | ~  27  | 0.8  | 0 |
| SDPA (full-attention only)                       | ~  43  | 1.3  | 7.1 |
| LM head + embedding (once-per-prefill)           | ~   3  | 0.1  | — |

**Every prefill tensor in the profile is in `DEV_*_DRAM_INTERLEAVED`.** That is the systemic signal driving the plan below.

## Goals

1. **Reduce TTFT** for prefill across all sequence lengths (not just `traced_4k`).
2. **Preserve PCC** against the existing demo (`models/demos/blackhole/qwen3_5_9b/demo/text_demo.py`).
3. **Make every change scale** to long contexts (target 128 k tokens) without re-engineering.

## Non-goals

- Decode tok/s improvements (separate workstream).
- Multi-device tensor parallelism (single Blackhole P150 only).
- Model-quality changes (no quantisation downgrades, no algorithm substitutions).

## Long-context extensibility — invariants every sub-item must hold

The model already chunks prefill (`qwen35_model.py:153 prefill_layer_chunked`). Linear-attention layers process at `chunk_size=2048`; full-attention at `attn_chunk_size = max(chunk_size, 4096)`. GDN sub-chunks at 64 (or 128 for long contexts) for Neumann-series stability. **The L1-sharding work happens at the chunk level, not the full-sequence level.** Concretely:

- **L1 sharding budget**: every memory-config decision is sized for a single chunk (2048 tokens for DeltaNet projections, 4096 for full-attention projections), not the full sequence. A 128 k-token prefill processes 64 DeltaNet chunks × 2048 + 32 full-attn chunks × 4096; each chunk's tensors fit the same L1 budget the 4 k profile produced.
- **Per-chunk per-core math (must hold for every Phase-1/2/3 change)**:
  - DeltaNet activation `2048 × 4096` bf16 = 16 MB → 125 KB/core on 128 cores ✓
  - DeltaNet activation `2048 × 12288` bf16 = 48 MB → 370 KB/core on 128 cores ✓
  - Full-attn activation `4096 × 4096` bf16 = 32 MB → 250 KB/core ✓
  - Full-attn activation `4096 × 12288` bf16 = 96 MB → 750 KB/core (tighter — see Phase 1 fallbacks)
  - Weights stay DRAM-sharded (footprint independent of seq).
- **No hidden `seq_len`-shaped allocations**: every new shard spec must use `chunk_size` (or `attn_chunk_size`) as its leading dim. A reviewer must be able to point to the chunk constant the spec is keyed off.
- **KV cache** (paged, DRAM-resident) is unchanged. At 128 k it scales as O(seq), but page allocation already exists; this plan does not touch it.
- **Trace compilation cost**: each unique chunk size produces a separate compiled kernel. Sticking with the existing `chunk_size=2048` and `attn_chunk_size=4096` keeps cache hits maximal. Any new chunk size proposed by a sub-item must justify its compile-cache cost.

## Phase 0 — Pre-flight checks (½–1 day)

Before touching any optimization, confirm the baseline and harness work:

- **0.1** Re-run `traced_4k` and confirm 3.4 s TTFT (within 5 %). Capture profile, save under `generated/profiler/reports/<phase0-baseline>/`.
- **0.2** Add a `traced_32k` and `traced_128k` test case (or identify if one exists) so every phase can be regression-tested against long contexts, not just 4 k. **Critical for the long-context extensibility invariant.**
- **0.3** Pin a PCC test set: a small fixed prompt set with known logits, run after every sub-item.

**Fallback**: if `traced_128k` is infeasible to run end-to-end (e.g. trace compile time too long), substitute a `traced_16k` smoke test that exercises ≥ 8 DeltaNet chunks and ≥ 4 full-attn chunks. Document the substitution in the phase notes.

## Phase 1 — Tactical config wins (1 week, ~450–550 ms saved)

**Goal**: lift matmul FPU utilization from 11–15 % to 40–60 %; eliminate the worst DRAM round-trips on the conv-state path.

### 1.1 — Raise MLP L1 threshold; shard MLP activations and weights *(2–3 days)*

- **File**: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py:48-60`
- **Change**:
  - Replace the `T <= 512` threshold with sharded L1 memory configs valid through `chunk_size = 2048`.
  - Activations (`2048×4096`, `2048×12288`) → `L1, BLOCK_SHARDED` on 8×16 grid.
  - Weights → `DRAM_SHARDED` along the output dim (parallel bank reads).
  - Outputs → `L1, BLOCK_SHARDED` on the same grid as the consumer.
  - Add `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` (or equivalent) on every `ttnn.linear` call.
- **Long-context note**: the threshold is keyed on the *DeltaNet chunk size* (2048), not on `T` (total seq). This sub-item must NOT use `T <= N` thresholds — instead it conditionally selects sharded L1 when `T == chunk_size` and sequentially calls the same kernel for every chunk in `prefill_layer_chunked`. At 128 k, this kernel is invoked 64 times; per-chunk footprint is identical to 4 k.
- **Expected**: matmul share drops from 18 % → 6 %; ~300–400 ms saved per pass.
- **Fallback ladder** (in order, stop at the first that works):
  1. **Step down grid**: 8×16 → 8×8 → 4×8 cores. Smaller grids increase per-core footprint but reduce CB overhead. Document the chosen grid as a constant in `model_config.py`.
  2. **Per-projection isolation**: if PCC fails, isolate which linear (gate / up / down) regresses. Keep the offender on `DRAM_INTERLEAVED`, ship the others sharded.
  3. **Activations only, weights stay DRAM-interleaved**: still recovers the activation round-trip win even if DRAM-sharded weights have a kernel-config bug.
  4. **Revert and gate behind a feature flag**: keep the `T <= 512` threshold, add a config flag `qwen35_mlp_l1_shard=False`, ship Phase 2 without 1.1.

### 1.2 — Shard QKV projections in attention paths *(1–2 days)*

- **Files**: `qwen35_gated_attention.py:130-165` (full-attn QKV), `qwen35_gated_deltanet.py:489` (`mega_fused_weight` linear).
- **Change**: same pattern as 1.1 — L1-shard activations, DRAM-shard weights.
- **Long-context note**: full-attn projections operate at `attn_chunk_size = 4096`, so the per-core activation footprint is 2× the DeltaNet case. Re-run the per-core math (`4096 × 4096` bf16 / 128 cores = 250 KB; `4096 × 12288` bf16 / 128 cores = 750 KB) — still fits, but is the tightest budget in the plan. If full-attn projections are also called with `attn_chunk_size = 8192` for some configurations, this sub-item MUST validate the budget at that size or fall back to `chunk_size = 4096`.
- **Expected**: ~50–80 ms saved per pass.
- **Fallback ladder**:
  1. **Reduce full-attn chunk size**: keep DeltaNet at 2048, force full-attn back to 4096 (it's already the floor) and validate.
  2. **Activations-only L1 sharding**: keep weights DRAM-interleaved if DRAM-sharded weights fail to compile for the wider full-attn projection shape.
  3. **DeltaNet-only**: ship 1.2 only for `mega_fused_weight`; defer full-attn projections to Phase 4.

### 1.3 — Move conv-state pad/concat path to L1 *(1–2 days)*

- **File**: `qwen35_gated_deltanet.py:505-540`.
- **Change**: the 18× `Tilize → … → UntilizeWithUnpadding → Slice → Concat → Tilize` rolling-buffer pattern currently goes DRAM→DRAM at every step. Switch the conv-state ring buffer to L1, height-sharded across the seq dim.
- **Long-context note**: conv state size is independent of total seq (it's a fixed 3-token pad rolling window), so L1 footprint is constant regardless of total prefill length. Free win at long contexts.
- **Expected**: ~50–80 ms saved per pass.
- **Fallback ladder**:
  1. **Skip the redundant Tilize**: the pattern ends with `... → Concat → Tilize`. If full L1 residency fails, just remove the trailing Tilize when the next op accepts ROW_MAJOR or the Concat is moved post-Tilize. Smaller win (~15–25 ms).
  2. **Stay in DRAM, eliminate one round-trip**: collapse the explicit `to_memory_config` calls if any inserted DRAM staging shows up in the profile. Smallest win (~5–10 ms).
  3. **Defer**: leave conv-state as is; its cost shrinks anyway once Phases 2/3 land.

**Phase 1 exit criterion**: re-profile shows MatMul `pct_of_kernel_time` < 10 % and matmul mean FPU% > 35 %. Cumulative TTFT drop ≥ 400 ms.

**Phase 1 abort criterion**: if total wall-clock TTFT regresses by > 50 ms after PCC passes (e.g. dispatch overhead from program-cache pressure outweighs the kernel savings), revert all 1.x changes and hand off the program-cache investigation to a separate workstream before continuing.

## Phase 2 — Structural data-path cleanup (1–2 weeks, ~100–150 ms saved)

**Goal**: keep activations L1-resident across consecutive ops; eliminate residual layout churn.

### 2.1 — Activation L1-residency through layer norm / residual / SiLU *(3–5 days)*

- **Files**: `qwen35_decoder.py`, `qwen35_gated_deltanet.py:606-615`, `qwen35_gated_attention.py`.
- **Change**: audit consumers of the now-L1-sharded matmul outputs (`LayerNormDeviceOperation`, `BinaryNgDeviceOperation`, `UnaryDeviceOperation`) and pass through the matching sharded memory config so no implicit `to_memory_config` to DRAM is inserted. Phase 1 made the producers shard; this sub-item ensures consumers don't re-stage to DRAM.
- **Long-context note**: every consumer kernel must accept the sharded layout for both DeltaNet (`chunk_size=2048`) and full-attn (`attn_chunk_size=4096`) shapes. Add a sharded LayerNorm program config selector keyed on the chunk constant.
- **Expected**: ~50–80 ms saved per pass.
- **Fallback ladder**:
  1. **Per-op opt-in**: switch ops to sharded L1 one at a time (LayerNorm first, then BinaryNg, then Unary). Each step PCC-tested.
  2. **Boundary `to_memory_config`**: if a consumer can't accept sharded input, force a single `to_memory_config(L1_BLOCK_SHARDED)` at that boundary instead of letting the framework round-trip through DRAM.
  3. **Drop the ops from L1**: leave specific ops on DRAM-interleaved if their sharded variants are buggy — still benefit from L1 producer outputs (one-way trip rather than round trip).

### 2.2 — Eliminate the 2080→2051→2048 padding cycle *(2–3 days)*

- **Pattern observed**: `Tilize 2048×8192 → … → UntilizeWithUnpadding 2080→2051 → Slice 2051→2048`. The 2051 dimension is conv-state pad (3) + seq (2048).
- **Change**: process the concat directly in TILE layout with explicit padding awareness so the Untilize+Slice+Tilize cycle disappears.
- **Long-context note**: the 3-token conv pad is fixed regardless of seq. The padded shape `chunk_size + 3` rounds to `chunk_size + 32` after tile alignment regardless of `chunk_size`, so this fix is sequence-length-invariant.
- **Expected**: ~30–50 ms saved per pass.
- **Fallback ladder**:
  1. **Eliminate one of the two Tilize/Untilize pairs only**: drop the trailing Tilize but keep the leading Untilize+Slice if PCC requires it. Half the win, half the risk.
  2. **Defer**: leave the cycle as is; its cost is ~30 ms which is < 1 % of TTFT after Phases 1/3.

### 2.3 — Reshape fold-throughs *(1–2 days)*

- **Targets**: head split/merge `2048×4096 ↔ 2048×32×128` (12 calls × 1.35 ms = 16 ms per block); the `4096×8192 → 4096×16×512` reshape on the full-attn path (5.7 ms per block).
- **Change**: with consistent block-sharding from Phase 1/2.1, these become logical views and can be replaced with `ttnn.reshape` (no-data-movement) instead of `ReshapeViewDeviceOperation` (which currently moves data because the source/dest shard schemes don't match).
- **Long-context note**: reshape semantics are seq-length-independent; once the sharding is consistent at chunk_size, the fold works at any context length.
- **Expected**: ~10–20 ms saved per pass.
- **Fallback ladder**:
  1. **Keep one DRAM-staging reshape**: if a specific reshape's source/dest sharding can't be unified, leave it as a `ReshapeViewDeviceOperation` but ensure both endpoints are L1-resident (single DRAM hop).
  2. **Defer**: the savings here are <1 % of TTFT.

**Phase 2 exit criterion**: glue ops (`ReshapeView`, `Tilize`, `Untilize`, `Slice`, `Concat`) drop below 5 % of prefill kernel time. Cumulative TTFT drop ≥ 550 ms (Phase 1 + 2).

## Phase 3 — GDN kernel internals (2–3 weeks, ~200–500+ ms saved)

**Goal**: cut GDN absolute kernel time by ≥ 200 ms per pass (3.1–3.3) or ≥ 800 ms (with 3.4 fusion). Note: by the time Phase 3 starts, GDN's *share* of TTFT will have grown (~75 % of the post-Phase-1/2 total) because Phases 1 + 2 reduce everything else. Phase 3 attacks the absolute number, not the ratio.

### 3.1 — GDN input I0 (`2048×8192` bf16) L1 block-sharded *(1 week)*

- **Files**: `models/demos/blackhole/qwen3_5_9b/tt/gdn_kernel/program_factory.py` (kernel I/O), `qwen35_gated_deltanet.py` (host wrapper).
- **Change**: change kernel circular-buffer wiring to expect L1-sharded inputs on the same 128-core grid the kernel already uses. Per-core 256 × 512 bf16 = 256 KB.
- **Long-context note**: GDN runs on `chunk_size=2048` per invocation regardless of total seq, so the per-core 256 KB budget is invariant. At 128 k, the kernel is invoked 64 × num_linear_layers times.
- **Expected**: 5–15 % cut on the (then ~1.5 s) GDN total → 100–300 ms saved per pass.
- **Fallback ladder**:
  1. **DRAM-sharded I0 instead of L1**: 32 MB I0 stays DRAM-resident but striped across banks for parallel reads. Smaller win (~50–100 ms) but lower kernel-rewrite scope.
  2. **L1-shard I1 (`2048×32`, 128 KB) only**: tiny tensor, trivial to shard, recovers a small win (~20 ms) without touching the I0 path.
  3. **Defer to Phase 4**: skip 3.1 entirely if kernel-internal CB budget is exhausted; Phase 1/2 still drop GDN's relative share by reducing the denominator.

### 3.2 — GDN output (`65536×32×128` bf16) DRAM-sharded *(2–3 days)*

- **Change**: output is 512 MB — too large for L1 at any context length. Switch from `DRAM_INTERLEAVED` to `DRAM_SHARDED` along the leading 65536 dim so the downstream reshape parallelises across DRAM banks.
- **Long-context note**: output footprint scales with chunk_size, not seq. At chunk_size=2048, 65536 = 2048 × 32 (seq × num_value_heads). Always too large for L1; DRAM-sharded is the only viable route at any seq.
- **Expected**: ~20–30 ms saved on the trailing reshape.
- **Fallback ladder**:
  1. **Keep DRAM_INTERLEAVED**: defer to 3.3 (which eliminates the reshape entirely, making 3.2 redundant).

### 3.3 — Fold trailing reshape into GDN kernel output ordering *(3–5 days)*

- **Change**: have the kernel write `[num_heads, seq, head_dim]` natively instead of `[seq×num_heads, head_dim]`. Eliminates the 6× post-GDN reshape entirely.
- **Long-context note**: kernel output ordering is independent of total seq; the fold is a one-time kernel change that benefits every chunk at every context length.
- **Expected**: ~50 ms saved per pass.
- **Fallback ladder**:
  1. **Cheaper reshape via 3.2**: skip 3.3, rely on the DRAM-sharded reshape from 3.2. Recovers ~20 ms instead of ~50 ms.
  2. **Defer entirely**: the reshape is < 2 % of TTFT after other phases.

### 3.4 — Investigate why 2 GDN calls per linear layer *(1–2 days investigation; 1+ week if fusable)*

- **Question**: 6 GDN calls span 3 linear-attention layers per block = 2 calls per layer. Identify the second call's purpose (Q/K-side vs V-side? Forward + state-update? Something else).
- **Change**: if the two are independent or can be fused into a single kernel invocation, this is the highest-variance bet in the plan.
- **Long-context note**: any merged kernel must still compile cleanly at `chunk_size=2048`. No new chunk size introduced.
- **Expected**: 0 ms (if not fusable) up to ~1000 ms (if fully mergeable, halves GDN cost across all 24 linear-attention layers).
- **Fallback ladder**:
  1. **Investigation only**: cap effort at 2 days. If no clear fusion exists, document why and stop.
  2. **Partial fusion**: fuse only the cheap shared work (e.g. shared input load) without merging the kernels themselves. Smaller but real win (~50–100 ms).
  3. **Defer**: skip entirely; not blocking for the rest of the plan.

**Phase 3 exit criterion**: GDN kernel time drops by ≥ 200 ms per pass (3.1 + 3.2 + 3.3 alone) or ≥ 800 ms (with 3.4 fusion).

## Phase 4 — Smaller wins / opportunistic (in parallel, ~30–60 ms saved)

> Note: the original "investigate full_attention seq=4096 vs linear seq=2048" item from the analysis is RESOLVED — `qwen35_model.py:194` sets `attn_chunk_size = max(chunk_size, 4096)` intentionally. Removed from this plan.

### 4.1 — SDPA Q-sharding for full attention *(1–2 days)*

- Currently 7.1 % FPU on a 5.4 ms call (×8 layers = 43 ms per pass).
- **Change**: L1-shard Q (4 MB, 32 KB/core), leave paged K/V in DRAM.
- **Long-context note**: at 128 k, full-attn runs `attn_chunk_size = 4096` chunks against a growing K/V cache. Q-shard scales linearly in chunk count; K/V cache stays in DRAM (paged). No new constraints.
- **Expected**: ~10–20 ms saved.
- **Fallback**: skip; SDPA is < 2 % of TTFT.

### 4.2 — Fused matmul+silu+mul (if it ships in ttnn) *(½ day investigation, 1–2 days integration)*

- Grep `ttnn/cpp/ttnn/operations/experimental/matmul/` for an existing fused gate-up matmul. If one ships for Blackhole, replace the `linear+linear+mul` triple in `qwen35_mlp.py:53-55`.
- **Long-context note**: chunk-size-invariant.
- **Expected**: ~10–20 ms saved if available.
- **Fallback**: if no fused op exists, do not author one in this plan — defer to a future ttnn-side workstream.

## Validation strategy

After **every sub-item** (not just every phase):

1. **PCC**: run the existing demo and the pinned PCC test set. PCC must match within tolerance (existing thresholds in the test files).
2. **Per-chunk profile**: re-profile `traced_4k`. Confirm the predicted ms-shifts materialise. If a sub-item is supposed to save 50 ms and saves 0, stop and root-cause before continuing.
3. **Long-context regression**: every sub-item that lands gets exercised by `traced_32k` or `traced_128k` (whichever was selected in Phase 0.2). PCC must hold; TTFT-per-token must not regress.
4. **Memory baseline**: update `project_qwen35_perf_baseline.md` in the user's `MEMORY.md` index with the new TTFT after each sub-item.

## Risks & cross-cutting fallbacks

| Risk | Likelihood | Mitigation | Cross-cutting fallback |
|------|:----------:|------------|------------------------|
| PCC regression from BF4/BF8 + sharded kernels | **HIGH** | Per-sub-item PCC test; isolate offenders | Each sub-item has a per-projection / per-op fallback (see ladders above) |
| L1 OOM (CB headroom + activations) at full-attn `chunk_size=4096` | MEDIUM | Per-core math validated for both 2048 and 4096 chunks | Step-down grid (8×16 → 8×8); split full-attn projections across more passes |
| Trace compilation explosion (each new chunk size = new compiled kernel) | MEDIUM | Re-use existing `chunk_size=2048` and `attn_chunk_size=4096`; never introduce new chunk sizes | Cache prewarm during model load; profile compile time after every phase |
| GDN kernel rewrite blocks Phase 3 entirely | MEDIUM | Phase 3 sub-items are independent (3.1 / 3.2 / 3.3 / 3.4 individually shippable) | Deferral path: skip 3.1, ship 3.2+3.3 alone; or skip all of Phase 3 — Phase 1/2 still recover ~550 ms |
| 128 k regression from a Phase 1 change | MEDIUM | Phase 0.2 long-context test runs after every sub-item | Per-sub-item revert via feature flags in `model_config.py` |
| Plan stalls on 3.4 investigation | LOW | 2-day cap on investigation; stop and document if no fusion found | Skip 3.4 entirely; Phase 3 still ships 3.1–3.3 |
| Dispatch / program-cache overhead grows with sharding | LOW–MEDIUM | Profile op-to-op latency, not just kernel duration, after each phase | Phase 1 abort criterion: revert all if wall-clock regresses |

## Total expected savings (sum across phases)

| Stage | Cumulative TTFT | Reduction |
|-------|----------------:|----------:|
| Baseline | 3.4 s | — |
| After Phase 1 | ~2.85 s | −16 % |
| After Phase 2 | ~2.7 s | −20 % |
| After Phase 3 (3.1 + 3.2 + 3.3 only) | ~2.4 s | −29 % |
| After Phase 3 (incl. 3.4 fusion) | ~1.5 s | **−56 %** |
| After Phase 4 | ~1.4–2.4 s | −30 to −59 % |

**Conservative target**: ~2.4 s TTFT (≈ 30 % reduction), achievable with Phases 1 + 2 + 3.1–3.3.
**Stretch target**: ~1.5 s TTFT (≈ 55 % reduction), conditional on the 3.4 GDN call-fusion investigation paying off.

## What is NOT in this plan

- **Decode optimization** — same DRAM-interleaved pattern likely applies; defer to a separate plan.
- **Multi-device tensor parallelism** — single Blackhole P150 only.
- **Quantisation downgrades** (e.g. moving down-proj to BF4) — leave to model-quality team.
- **LM head sharding** — runs once per prefill, ~3 ms total, not worth tuning.
- **Authoring new fused ttnn ops** — Phase 4.2 only adopts existing fused ops if they ship; does not create new ones.

## Open questions

1. Does `traced_128k` exist as a runnable test, or does Phase 0.2 need to author one?
2. Is the GDN kernel already touched by other in-flight work (the user's untracked `test_gdn_*` files are exploratory and not in `text_demo`, so the answer is currently "no" — re-confirm before Phase 3 starts).
3. Are there shape configurations of full-attn `attn_chunk_size > 4096` exercised by any production test? If yes, Phase 1.2's per-core budget needs re-validation.
