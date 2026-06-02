# opt_transfer — Size-Aware Memory-Placement Optimization — Design

**Date:** 2026-06-02
**Status:** Design approved; ready for implementation planning
**Author:** sdawle (with Claude Code)
**Builds on:** `docs/superpowers/specs/2026-06-01-optimization-transfer-bringup-design.md`
and the implemented `models/experimental/opt_transfer/` package (Plan-1, op-fusion transfer).

## Problem

`opt_transfer` today transfers **op fusions** but does **not** reason about **L1-vs-DRAM tensor
placement** at all. Verified against the code: `codegen.py` hardcodes `DRAM_MEMORY_CONFIG`; there is
no L1-budget model, no memory-config transfer, no residency/placement search anywhere in the package.

Memory placement is a large, separate source of TTNN latency. dots.ocr (`models/demos/rednote_hilab_dots.ocr/`)
is the motivating case: a correct, already-fusion-optimized model (e2e 2559 ms, OCR exact vs HF) where
remaining latency is dominated by buffer placement, not missing fusions.

**Critical nuance (from dots.ocr's own `PERF_NOTES.md`): "keep everything in L1" is wrong.** The
biggest LM-prefill win there was the *opposite* — the **DRAM-residual fix (−863 ms, 4.3×)**: an
unconditionally L1-pinned `[seq≈4891, 1536]` (~15 MB) residual overflowed L1 and forced a pathological
matmul path (34 ms/layer vs 6 ms with a DRAM input). The correct rule is **size/shape/budget-aware
placement**, and the donors already encode it (`L1 if seq<=1024 else DRAM`).

## Goal

Add a **size-aware memory-placement optimization** pass to `opt_transfer`:
1. The KB records, per op, donor-observed **(size regime → memory_config + program_config)** mappings,
   including size-conditional placement rules.
2. At bring-up, the framework picks L1-vs-DRAM (+ sharding/program config) per tensor by its **actual**
   size and the **device L1 budget**, dataflow-propagated, honoring "don't pin large activations."
3. Every placement move is **perf-gated** against the current DRAM baseline — kept only if it's
   actually faster; PCC-gated for safety; L1-budget-checked for feasibility.

First target: dots.ocr's wrongly-DRAM **small/hot** tensors, leaving large activations on DRAM.

## Non-goals

- Not a global ILP buffer allocator. Decisions are per-tensor along producer→consumer chains, KB-guided.
- Not "force L1." The perf gate + budget guardrail explicitly preserve correct DRAM placements.
- Does not replace op-fusion transfer; it is an additional, composable axis.
- Vision-attention's architectural O(seq²) cost is out of scope (algorithmic, not placement).

## Architecture (extends the three layers of the base spec)

### 1. KB enrichment — placement observations (Layer A)

Add to each `KBEntry` (or a sibling record) a list of placement observations mined from donors:

```
placement_observations: [
  {
    op,                       # ttnn op the observation is about
    tensor_role,              # activation | weight | qkv_out | residual | score | ...
    size_descriptor,          # {dims: "[seq, hidden]", dtype, bytes_expr: "seq*hidden*2"}
    memory_config: {
      buffer,                 # L1 | DRAM
      layout,                 # interleaved | width_sharded | height_sharded | block_sharded
      shard_spec_template,    # how grid/shard-shape scale with dims + core_grid (a formula, not a literal)
    },
    program_config,           # matmul/SDPA program config, parameterized by dims + grid
    condition,                # size-conditional rule when the donor encodes one (e.g. "seq <= 1024")
    source,                   # file:line provenance
  }, ...
]
```

**Mining (same tiers as base spec):**
- **Donor call sites** (`tt_transformers`/`tt_dit`/`demos`): the captured snippet includes the config
  *construction*; the LLM extractor pulls `memory_config` (buffer/layout/shard spec), `program_config`,
  and any **size-conditional placement** (the `L1 if seq<=N else DRAM` pattern). Shard specs captured as
  **templates** (scale with dims/grid), not literals.
- **Unit tests** (`tests/.../operations`): concrete shapes + configs → real (size → config) anchor points.
- **tech_reports**: L1-budget rules / "don't pin large activations" / sharding guidance → `applicability_notes`.

### 2. Placement decision + codegen (Layer B/C)

A **placement step** between `match` and `codegen`:
1. For each op/tensor in the traced graph, compute the **concrete** size from model dims + trace
   (`seq`, `hidden`, heads, dtype).
2. Retrieve the op's `placement_observations`; **evaluate the size-conditional rule against the actual
   size and the device L1 budget** (per-core L1 for the target arch; dots.ocr = Blackhole p150). Choose
   L1 (shard spec + program config *instantiated* to the real shape/grid) or DRAM-interleaved.
3. **L1-budget guardrail (hard backstop):** if the tensor + required co-resident buffers exceed the L1
   budget, force DRAM regardless of donor preference. The donor `condition` is a proxy; the budget check
   is authoritative.
4. **Dataflow-aware:** propagate placement along producer→consumer chains, because regressions come from
   *interactions* (the dots.ocr L1-tensor→matmul case), not isolated ops.
5. **Codegen emitters honor the chosen `memory_config` + `program_config`** (instantiating shard specs to
   the actual shape/grid) — replacing the hardcoded `DRAM_MEMORY_CONFIG`.

### 3. Verification (Layer C)

- **PCC gate:** placement is numerically neutral; PCC must stay > 0.99 — catches config *bugs*
  (invalid shard spec, wrong grid).
- **L1-budget feasibility check** before device run → over-budget L1 choice falls back to DRAM (no OOM/hang).
- **Perf gate = the decision-maker.** Measure **block/chain-level** latency on the traced path (`perf`
  skill) with the chosen placement **vs the current DRAM baseline at the real size**. Keep the move only
  if faster by the configured threshold; else revert. This is how "some DRAM ops are correctly there" is
  enforced. The **repair loop** reverts/adjusts placements that regress.

## Integration with existing package

| Piece | Change |
|-------|--------|
| `schema.py` | add `placement_observations` to `KBEntry`; carry chosen `memory_config`/`program_config` on the resolved proposal |
| `kb/miner.py` + `matcher.py` (`LLMClient`) | extraction also emits placement observations (buffer/layout/shard template/program config/condition) |
| new `placement.py` | size+budget-aware decision over the traced graph (chains), produces per-tensor placement |
| `codegen.py` | emitters honor `memory_config`/`program_config` (instantiate shard specs); stop hardcoding DRAM |
| `graph.py` | new `placement` node between `match`→`gate`/`codegen`; `verify` adds the block-level perf comparison; `repair` reverts regressing placements |
| `verify.py` | reuse `perf_gain_pct`/`perf_gate_pass`; add L1-budget feasibility check |
| config | per-arch L1 budget (Blackhole p150), placement perf-gain threshold |

## Scope decomposition

1. **KB placement-observation mining** (offline) — enrich the KB with size→config observations; validated
   by "contains real L1/sharded configs + at least one size-conditional rule from the donors."
2. **Placement decision + budget model + emitter config-honoring** — validated offline (correct L1/DRAM
   choice for a given size+budget) + on-device PCC-neutral.
3. **Perf-gated placement loop on dots.ocr** — profile to get DRAM-resident hotspots+sizes, propose
   KB-driven L1 moves for small/hot tensors, perf-gate vs the current baseline, keep the wins; confirm the
   large activations stay on DRAM.

Build a thin vertical slice first (one op family — e.g. the attention head-split/qkv whose L1-pinning is
already known-good in dots.ocr) through all three, then broaden.

## Open questions / risks

- **Symbolic shapes in donor code.** Sizes are usually symbolic (`seq`, `hidden`); concrete anchors come
  from unit tests + the new model's dims. KB stores the donor *rule/template*, evaluated at bring-up.
- **Shard-spec templates are the hard part.** Capturing how a shard grid/shape scales with dims+core-grid
  (not a literal) is non-trivial; start with the common interleaved-L1 and width/block-sharded matmul
  patterns the donors use, expand as needed.
- **Chain-level optimization is combinatorial.** Keep it greedy + perf-gated (KB-guided per-chain), not a
  global search; log any chain left unoptimized.
- **L1-budget model accuracy.** Per-core L1 + co-resident buffer estimate may mis-predict; the feasibility
  check + perf gate are the backstops (a wrong L1 choice fails feasibility or loses the perf gate).
- **Profiling fidelity (dots.ocr).** Per `PERF_NOTES.md`, tracy op-name enrichment on the decode pattern
  needs care; profile the traced path and confirm a clean hotspot table before proposing moves.
