# MiniMax-M2 Prefill — Skeleton & Work Split

Companion to `PREFILL_PROPOSAL.md`. Maps each scaffold seam to an owner + blocker so
work can be split. **Rule:** every item is either *validated* (PCC test) or *scaffold*
(`NotImplementedError` + owner). Nothing in between.

## Validated core (do not regress) — `tt/`
- attention / router / experts / full decoder layer — PCC vs HF @ TP=1 (see `tests/unit/test_*_vs_hf.py`).
- `model.py` carries an `on_layer_complete(layer_idx)` seam in the layer loop (no-op default).

## Scaffold seams (each is a grabbable task)

| Seam | File | Tier | Owner | Blocked on | Validatable here? |
|---|---|---|---|---|---|
| Chunked prefill in attention | `tt/attention/prefill.py` | 1 | model | — (op exists, §11.2) | Yes — chunked vs full PCC @ reduced scale |
| Standalone runner | `tt/runners/prefill_runner.py` | 1 | model/runner | full model fits only on Galaxy | Reduced-config only here |
| `MiniMaxPrefillPipeline.prefill` chunk loop | `tt/tt_minimax_prefill_pipeline.py` | 1 | model/runner | chunked-SDPA wiring above | Reduced-config |
| `_prepare_input_tensor` (SP shard after chunk) | pipeline | 1 | runner | — | Reduced-config |
| Per-layer KV migration | `tt/runners/migration_setup.py` | 2 | **migration team** | ttnn disaggregation API | No (needs fabric + multi-mesh) |
| SHM request loop | `tt/runners/prefill_runner.py` | 2 | serving team | SHM protocol + C++ server | No |
| P/D disaggregation glue | pipeline + migration | 2 | migration + runner | migration API | No |
| EP=8 experts | `tt/experts_throughput/` (uses ttnn fused moe_gpt op; rewrite for EP) | 3 | model | **EP design undefined** | No (multi-card) |
| Perf (trace, slice_write, reduce_scatter, grids) | various | 3 | perf | functional first | Partly |

## Suggested first cuts
1. **Chunked prefill** (Tier 1, model) — only piece that's both high-value and validatable here. Wire `chunked_scaled_dot_product_attention` (paged) + chunk loop + PCC test.
2. **Migration endpoint** (Tier 2, migration team) — fill `setup_prefill_migration`; pipeline already calls `migrate_layer`/`wait` via the `on_layer_complete` seam.
3. **Runner SHM loop** (Tier 2, serving) — independent of the model.
4. **EP=8** (Tier 3) — design first, then implement in `experts_throughput/`.
