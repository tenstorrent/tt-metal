# Planning: Matmul Beyond DeepSeek

## Objective

Improve matmul efficiency in shared paths for N150/T3K with regime-aware rules keyed on inspectable properties
(`M,K,N`, layout, memory/sharding), not model-specific logic.

## Current Status

- [x] Regime manifest and kernelbench infrastructure in place (`matmul_n150_regimes.json`, `matmul_n150_kernelbench.py`)
- [x] Alternating baseline-first A/B runner in place (`matmul_n150_alternating_ab.py`)
- [x] Exploration scoreboard in place for non-mergeable mixed outcomes
- [x] Regime-by-regime memory-policy screening workflow validated (decode/prefill/tiny independently)
- [ ] Promote a first mergeable regime-aware shared-path rule

## Idea Backlog (Next)

- [ ] Build discrete per-regime config search (not global overrides):
  - focus knobs: `in0_block_w`, `per_core_M`, `per_core_N`, `out_subblock_w`.
  - constrain by existing LLK/matmul validity rules.
  - screen with kernelbench first; require clear kernel gain before promotion.
- [ ] Add regime predicate schema for candidate rules:
  - predicates on `M,K,N`, batch volume, layout (`INTERLEAVED`/`SHARDED`), buffer types.
  - avoid model-name and input-hash conditions.
- [ ] Implement first shared C++ regime-aware rule in `create_simple_matmul_program_config(...)`:
  - target one regime with strongest repeated signal.
  - keep fallback path unchanged outside predicate.
- [ ] Validate each promoted rule in three stages:
  - target regime A/B,
  - cross-regime A/B (other classes),
  - full `matmul_n150` protocol strict gate.
- [ ] Add neutral-band reporting (exploration only) while keeping strict merge gate unchanged:
  - report when non-target regimes are statistically neutral vs regressive.
- [ ] Expand regime manifest with additional canonical non-DeepSeek shapes (Llama/Mistral/ViT/SDXL paths) if gaps remain.

## Notes

- Keep changes narrowly scoped; avoid model-specific hand-tuning.
- Prefer improvements that can benefit generic `ttnn.matmul` / `ttnn.linear` usage.
