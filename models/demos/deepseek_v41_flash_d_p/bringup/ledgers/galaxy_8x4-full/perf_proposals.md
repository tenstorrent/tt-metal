# Host-time proposals: DeepSeek-V4.1, Galaxy 8x4, one warm 5120-token chunk (40 layers)

Perf review, 2026-10-28, tree 8ad76bad6bb. Proposals only: no code changed, nothing run on the device.

**Configuration (owner decision, 2026-10-28): the only one shipped is the block MoE with its fused head and fused
tail on a torus** (MOE_IMPL=block, MOE_HEAD=fused, MOE_TAIL=fused, FABRIC_2D_TORUS_XY). Flat and FABRIC_2D are
dropped. The earlier flat analysis is kept in Appendix A; its numbers are flat's, not this configuration's.

Sources (warm, this box):
- T = `/scratch/dnijemcevic/posprof/hostprof_torus.pstats`: cProfile of 5 warm chunk runs at context 0, block
  fused on the torus. Per chunk = /5.
- D = `/scratch/dnijemcevic/posprof/results/TBLOCK0_ops.json.gz` and `TBLOCK0_profile.json`: per-layer device
  ops.
- Baseline:
  - Device 219.5 ms, 2,029 device ops a chunk.
  - Uninstrumented warm wall 624.4 ms.
  - Under cProfile, 649 ms of host time a chunk:
    - **375.7 ms ttnn dispatch** (`decorators.py:1217`, 3,055 calls).
    - **145.1 ms `begin_chunk`**: engram `host_rows` 107.6 + `from_torch` 28.4 + 9.0 in begin_chunk itself.
    - **36.0 ms `ccl.dealloc`** (1,986 calls, 18 us each).
    - **32.9 ms** in the fused head's `run_prepared` (`fused_head.py:675 _run`). It is not counted in the ttnn
      dispatch above: it calls the binding directly.
    - The rest is Python glue.
  - Per layer: 12.5 ms of host against 5.5 ms of device. The device waits on the host for most of the chunk.

Where the 40 layers' host time goes (T, cum, ms a chunk):

| step | host ms | of which |
|---|---|---|
| attention | 149.0 | out-projection reduce-scatter (`sum_cols`) 36.1, q path 34.9, `_reshard` 15.7 (2 x 40 all_to_all, 194 us each), rope 26.6 (all 136 calls) |
| experts (MoE block: `moe_block.py:600 forward`) | 141.8 | see the next table |
| q_a | 53.5 | `tp_sum` = reduce-scatter + all_gather, 56.7 incl. compressor |
| attn_hc + ffn_hc | 26.4 | fabric all_gather + mhc_pre_xing |
| attn_norm + ffn_norm | 26.0 | rms_norm (stats), all_gather, rms_norm |
| residuals | 12.4 | mhc_post |
| kv, compressor, indexer, engram_add | 63 | |

The MoE block, per chunk (T):

| piece | code | host ms | per call |
|---|---|---|---|
| fused tail | `tail_a2a.py:601 run` (alloc + **plain `ttnn.generic_op`**: no frozen hash, not prepared) | 45.4 | ~1.05 ms per generic_op |
| fused head | `fused_head.py:608 run` (`run_prepared` 32.9 + arena realloc `se_engine3.py:1514` 8.3 + free 1.2) | 43.2 | 822 us `run_prepared` |
| exchange | `moe_block.py:1591 prepped`: `arena_fits` 7.9 (alloc + dealloc of the arena, only to check its address), alloc_arena 5.6, dispatch 12.3 (manifest alloc, prepared sparse_all_gather_2d, dealloc, reshape), `_experts_combine` 20.2 (flat op, 473 us) | 48.4 | |
| head input copy | `moe_block.py:1221` (`ttnn.copy` of x into the head's fixed input) | 1.7 | 41 us |

Host cost per call, this configuration (T; ttnn calls credited to their caller):

| class | example | us a call |
|---|---|---|
| plain `ttnn.generic_op` over 32 per-chip programs | fused tail; fabric_reduce_scatter `_fast` (alloc + generic_op, 184 calls, 73.7 ms) | ~1,050 (tail), ~760 (RS) |
| prepared generic_op | `prepared_program.generic_op` (40 calls, 6.7 ms); head `run_prepared` (patches x, w, router slots) | 167; 822 |
| C++ op, one program per coordinate | `rope_stock` 192; `_reshard` (all_to_all_async_generic) 194 | ~190 |
| fabric all_gather | `ccl.all_gather` (262 gathers, 50.3 ms cum) | 192 a gather |
| single-program op | matmul 78, rms_norm 61, `attention.write` 51 | 50-80 |
| dealloc | `ccl.dealloc` | 18 |

---

## Ranked proposals (block fused head and tail, torus)

Each entry: the setting (env `DS41_<NAME>`, default on, the old path selectable); what goes or gets cheaper; the
estimate; accuracy; effort and risk. "Block code" means models/demos/moe_block or
deepseek_v3_d_p/tt/moe (Marko's / the owner's): it changes only with the owner's agreement, behind an option.

### 1. ENGRAM_ROWS_ASYNC: engram host rows off the critical path (Appendix A entry 2, re-estimated)

- **Today:** `model.py:736 begin_chunk` blocks for **145.1 ms** before layer 0. Engram layers L1 and L14.
  - `engram.py:51 host_rows` 107.6: `weights.py:179 rows` 62.0 (memmap gather 43.5 tottime, `.float()` 15.3),
    `torch.zeros` 5.4, `.to(bf16)` 5.2.
  - `engram.py:71 from_torch` 28.4: an upload replicated 4x over TP.
  - 9.0 ms in begin_chunk itself.
- **Change:** as Appendix A entry 2. A worker thread per engram layer (L1 first) does the hash, gather, bf16
  dequant (exact) and the host tilize; the main thread joins at `take_engram_rows` and calls `ttnn.to_device`.
  Rows are uploaded split over TP, with one fabric all_gather over TP on the device.
- **Estimate:** L14's ~65 ms hides behind layers 1-13 (~160 ms of dispatch). L1's half-cost dequant overlaps embed
  plus layer 0 (~12 ms). The TP-split upload cuts the H2D by ~3/4.
  - **~95 ms a chunk (80-110).**
  - 0 ttnn calls removed; +2 small device gathers (~0.2 ms each).
- **Accuracy:** bit-identical rows (e4m3 x 2^k is exact in bf16).
- **Effort and risk:** 4-6 h, model code (engram.py, model.py). GIL hand-off is the main risk (5 ms switch
  interval); measure it. The contract tests re-run.

### 2. GENERIC_OP_PREPARED: the 92 reduce-scatters through the prepared-program fork (Appendix A entry 1)

- **Today:** `fabric_reduce_scatter._fast` (`fabric_reduce_scatter.py:1164`), 92 a chunk:
  - attention out 40 (`ccl.sum_cols`; D: 640x5120 bf16, 0.151 ms device);
  - q_a 40 (`ccl.tp_sum`; 640x1792, 0.101 ms);
  - compressor / indexer 12.

  184 decorator calls, 73.7 ms (alloc + plain generic_op): ~760 us per generic_op.
- **Change:** fork option `prepared=True`. Keep the address-keyed descriptor LRU, dispatch each cached descriptor
  through `prepared_program.generic_op` (its own cache), so no slot is ever patched.
- **Estimate:** 92 x (~760 -> ~200-250 us) = **~50 ms a chunk (40-60)**. 0 calls removed, no device change.
- **Accuracy:** bit-identical (same programs and arguments).
- **Effort and risk:** 3-4 h, fork (Python, no build) plus tt/ccl.py. Low risk.

### 3. TAIL_A2A_PREPARED: the fused tail's generic_op through the prepared-program fork (new, block)

- **Today:** `models/demos/moe_block/tt/tail_a2a.py:601 run`, 40 a chunk:
  - allocates `out` [1, 1, 640, 1280] bf16 every call;
  - looks up a descriptor (16-entry dict keyed by the y / se / out addresses);
  - calls plain **`ttnn.generic_op`** (`:623`) on a 32-program descriptor of inline kernels that is never frozen
    (no `ttnn.freeze_program_hashes`).

  80 decorator calls, **45.0 ms**: ~1.05 ms per generic_op. That is the most expensive single dispatch in the
  chunk. Device: D L3.experts `generic_op 160x5120 · 640x1280` = 0.079 ms.
- **Change:**
  - `ttnn.freeze_program_hashes(md)` when a descriptor is built. Every other combine_rf builder already does this,
    e.g. `combine_plan.py:2226`.
  - Dispatch through `ttnn.bringup.prepared_program.generic_op`.
  - Optionally, a persistent `out` (the `out=` argument exists). The model must then not free it: under
    MOE_TAIL=fused the head already persists its outputs (`fused_head.py:_out`, "nothing outlives a forward").
  - Option on TailA2A (e.g. `prepared=True`); the DS41 adapter turns it on.
- **Estimate:** 40 x (~1,050 -> ~200 us) = **~34 ms**; with the persistent `out`, +40 allocs and +40 deallocs
  (~2 ms). **~36 ms a chunk.** No device change.
- **Accuracy:** bit-identical.
- **Effort and risk:** 2-3 h. Block code (owner's / Marko's), an option with the old path default. Low risk: the
  descriptor objects are already reused per key, which is what the prepared fork's contract needs.

### 4. FUSED_HEAD_PREPARED_PER_LAYER: one prepared head program per layer, so a call patches nothing (new, block)

- **Today:** `fused_head.py:675 _run`: one prepared program shared by the 40 layers (`layers=True`). Every call
  patches the runtime-arg slots of the layer's shared-expert weights `w` and router weights `rw` (see the comment
  at `:690` and the `wpos` key). `x` is fixed: `moe_block.py:1221` copies it into the head's input. One whole-grid
  program on 32 chips: **822 us a call, 32.9 ms a chunk.** The flat path's prepared sites with nothing to patch run
  at 117-252 us (blockfix).
- **Change:** keep up to 40 `PreparedProgram`s keyed by the full `skey` with the weight addresses in it (a dict in
  `self.state` in place of the single `pp`). With x, y, the arena and the outputs pinned, `run_prepared` then
  patches nothing.
  - Host memory: 40 copies of the head descriptor.
  - Same programs on the device; the program cache keys are unchanged (prepared hashes exclude patched slots).
- **Estimate:** 40 x (822 -> ~250 us) = **~20 ms a chunk (10-25).** Check first with `PreparedProgram.patch_count`
  (diagnostic) how many slots a call patches today. If few, the 822 us is the dispatch itself and this drops out.
- **Accuracy:** bit-identical.
- **Effort and risk:** 3-4 h, block code. Low risk. A first prepare per layer happens in warm-up.

### 5. CCL_PERSISTENT_OUT: persistent outputs for the per-layer fabric gathers and reduce-scatters (Appendix A entry 3)

- **Today:** 262 `ccl.all_gather` a chunk:
  - attn_hc, ffn_hc, attn_norm stats, ffn_norm stats, q_a, attention kv: 40 each;
  - compressor / indexer / engram: 22.

  Plus 92 reduce-scatters. Each allocates an output that is later freed (18 us).
- **Change:** as Appendix A entry 3. A pair of load-time buffers per in-step site (the existing `out=` /
  `output=` arguments), with no dealloc for them.
- **Estimate:** 354 allocs + 354 deallocs, ~47 us each: **~17 ms a chunk (12-22).** It also makes entry 2's
  descriptors always hit.
- **Accuracy:** bit-identical.
- **Effort and risk:** 4-6 h, tt/ccl.py plus call sites. Medium risk (lifetimes); add a debug assert.

### 6. ROPE_UNIFORM_PROGRAM: one RoPE program for the mesh, not 32 (Appendix A entry 4)

- **Today:** 136 `rotary_embedding_indexed` a chunk (attention q / o 80, kv 40, compressor 8, indexer 8): **192
  us** each. The op stamps a program per coordinate (`my_sp_coord` is a compile-time arg) and re-applies run args
  to all 32 on every hit.
- **Change:** fork it to `ttnn.bringup.rotary_embedding_indexed` with option `uniform=True` (one program, valid
  for the model's per-chip pre-sliced cos / sin).
- **Estimate:** 136 x (192 -> ~65 us) = **~17 ms a chunk.** No device change.
- **Accuracy:** bit-identical, provided each chip's cos / sin row offset is 0 with pre-sliced tables. Verify that
  first.
- **Effort and risk:** 5-7 h, C++ fork plus build and fork tests. Low risk.

### 7. TP_SUM_ALLREDUCE: q_a's reduce-scatter + all_gather as one fabric all-reduce program (Appendix A entry 6)

- **Today:** `ccl.tp_sum` (`ccl.py:228`), 46 a chunk, 56.7 ms cum. q_a's reduce-scatter (0.101 ms device) plus
  all_gather 640x448 -> 1792 (0.083 ms device) per layer, and compressor.
- **Change:** a fabric_reduce_scatter option `gather=True`: the final cores forward their reduced block along the
  line, so the gather happens inside the same program.
  - On this torus the TP axis is a ring. The option can use both directions (topology Ring), which halves the
    busiest link's bytes.
- **Estimate:** -44 gathers, -44 allocs, -44 deallocs: **~88 calls, ~9 ms a chunk.** Device about -2 to -3 ms.
- **Accuracy:** bit-identical (same adds in the same order; the gather is a copy).
- **Effort and risk:** 8-10 h, fork kernel work (Python op). Medium risk.

### 8. MOE_DISPATCH_IN_PROGRAM: the token dispatch inside the experts' program (new; an existing block option)

- **Today:** `tt_k3_token_split_moe.py:1046 dispatch`, 40 a chunk, 12.3 ms cum (307 us a layer):
  - allocate the manifest ring;
  - the prepared `sparse_all_gather_2d` program;
  - deallocate;
  - `ttnn.reshape`.

  Then the experts' program. `MoEOptions.experts_during_dispatch` (`moe_block.py:134`, default False; the DS41
  adapter `moe_block_adapter.py:82` leaves it off) runs the dispatch as the first phase of the combine's port
  kernels in the experts' program (`_routed_in_program`, `tt_k3_token_split_moe.py:1494`). The weights then stream
  from the program start, under the dispatch.
- **Change:** a setting forwarding `experts_during_dispatch=True` in `moe_block_adapter.block_options`. No new
  code.
- **Estimate:** -1 program, -1 alloc, -1 dealloc a layer: **~6-8 ms host a chunk.** Device: the dispatch overlaps
  the weight stream. The experts are 104.5 ms of the chunk's 219.5; Marko's claim, not measured on DS41. This is
  the one block change here that also targets device time.
- **Accuracy:** the same dispatch and combine kernels, phased in one program. The combine snap
  (MOE_HEAD=fused) keeps the sums order-independent, so results should be bit-identical; confirm with the ladder.
- **Effort and risk:** 1 h plus validation (L.s10240, L.smoke). Risk: the manifest's L1 placement on 8x4 torus
  (`alloc_arena_disp`) was never run on this model. Hangs are possible; use the 1-layer repro first.

### 9. ARENA_FITS_FOLD: drop the exchange's redundant arena probe (new, block)

- **Today:** `moe_block.py:1595 prepped` asserts `self.block.arena_fits()`. That is
  `tt_k3_token_split_moe.py:1077`: allocate the whole experts arena, compare its address, `ttnn.deallocate`. Right
  after, `routed_prepped -> alloc_arena` (`:1084`) allocates it again and asserts the same address with its own
  message. 80 calls a chunk, 7.9 ms.
- **Change:** prepped skips `arena_fits` and relies on `alloc_arena`'s assert (or `arena_fits` hands its tensor
  to `alloc_arena`). The same condition is still checked once.
- **Estimate:** **-80 calls, ~8 ms a chunk.** No device change.
- **Accuracy:** nothing computed changes.
- **Effort and risk:** 0.5 h, block code. No risk beyond the assert's message.

### 10. NORM_STATS_IN_MHC: mhc_pre_xing writes the norm's partial sum of squares (Appendix A entry 7)

- **Today:** attn_norm and ffn_norm each start with `rms_norm(stats_out=True)` (`norm.py:136`), 80 a chunk (D: 3
  calls a norm in TBLOCK). It re-reads the y mhc_pre_xing just wrote.
- **Change:** a `mhc_pre_xing` option `norm_stats_out`.
- **Estimate:** **-80 calls, ~6.3 ms a chunk.** Device -0.48 ms.
- **Accuracy:** the same fp32 sums in another order. Deterministic, not bit-identical.
- **Effort and risk:** 6-8 h, C++ fork (mhc_pre_ttnn). Low-medium risk.

### 11. ATTN_SCALE_IN_SDPA (Appendix A entry 10)

- **Change:** drop the q pre-scale multiply (`attention.py:543`; D: multiply 640x1280, 0.011 ms) and pass the scale
  to `sparse_sdpa`, with the sink stored as sink / scale.
- **Estimate:** **-80 calls (with deallocs), ~4.5 ms a chunk.** Device -0.4 ms.
- **Accuracy:** the scale is applied to the fp32 scores (as the reference does); one bf16 rounding of q fewer. The
  ladder re-runs.
- **Effort and risk:** 2 h, low.

### 12. SETTINGS_CACHED (Appendix A entry 9)

- **Today:** `model_settings.get` runs 1,174 times a chunk, 4.2 ms cum (mostly `ccl.fabric` / `bf16_sums` /
  `mm_configs.tuned`).
- **Change and estimate:** read once at load. **~4 ms a chunk.** 1 h, no risk.

### 13. HEAD_INPUT_IN_PLACE: ffn_norm writes the fused head's input buffer (new, block + fork)

- **Today:** `moe_block.py:1221`: `ttnn.copy(x, self.x)` every layer, because the head reads a fixed input (D:
  copy 640x1280, 0.009 ms device). 40 calls plus the dealloc of ffn_norm's output.
- **Change:** a `ttnn.bringup.rms_norm` option for a preallocated output (the fork has none today). ffn_norm
  writes into `MoEBlock.input_buffer()`.
- **Estimate:** **-80 calls, ~2.4 ms a chunk.**
- **Accuracy:** bit-identical.
- **Effort and risk:** 3-4 h, C++ fork option. Low.

### 14. COMPRESSOR_POOL_SIGMOID (Appendix A entry 11)

- **Estimate:** ~2-3 ms a chunk (`compressor.py:422 _pool`; T: 36 calls, 3.6 ms). Unchanged from the flat
  analysis.

### Flat-only entries: n/a on this configuration

- **ROUTING_PACKED** (Appendix A entry 5): the fused head routes inside its own program.
- **MOE_ADD_IN_RS** (Appendix A entry 8): the fused tail adds the shared expert's rows itself.
- Entry 1 of Appendix A's planner / zero / barrier sites: weighted_combine is not on this path.

### Totals (block fused head and tail, torus)

Sum of entries 1-14, each estimated alone: 95 + 50 + 36 + 20 + 17 + 17 + 9 + 7 + 8 + 6.3 + 4.5 + 4 + 2.4 + 2.5 =
**~280 ms a chunk** (range about 230-320). Entries 2 and 5 overlap a little on the reduce-scatter allocs.
- **Expected warm wall: 624 -> ~340-390 ms**, still host-bound against 219.5 ms of device.
- Device time: about -3 to -6 ms from entries 7, 10, 11 and 13, plus whatever entry 8 brings (unmeasured).
- Even with everything done, about 3,000 dispatches remain at the 50-200 us floor, ~1,700 deallocs and ~55 ms of
  glue. Getting to the device time needs trace (see "Host time no fusion removes").

## Torus-specific notes (device, not host)

- The TP-axis CCLs run Linear: `ccl.py` passes no topology to `fabric_reduce_scatter` / `fabric_all_gather`. The
  torus closes the TP ring, so Ring topology on the attention and q_a reduce-scatters and the hc / norm / q_a /
  kv gathers would halve their busiest link's bytes.
  - D at L3: the RS 0.151 + 0.101 ms; the gathers 0.05-0.11 ms each.
  - Device only, a few ms a chunk; host unchanged.
  - It fits as a mode of entry 7's fork round.
- all_to_all_async_generic (attention reshard): 2 x 40 a chunk, 22.4 ms of device, the largest attention op after
  sparse_sdpa. It already takes `cluster_axis=TP_AXIS` with 2 links; whether it uses the wrap link on a torus is
  worth a check.

## Looked at and rejected (this configuration)

- **Attention without the head <-> row reshard.** Rejected as in Appendix A: each chip would read KV for 4x the
  query rows, about +1 ms device a layer, for ~16 ms of host.
- **Keeping the fused head's arena allocated across layers** (`se_engine3.alloc_arena` 208 us + free 29 us a
  layer, 9.5 ms a chunk). The head gives its L1 back for the experts' arena between calls (`fused_head.py:613
  free_arena`). A permanent arena would collide with the experts' L1.
- **Persisting the exchange's experts arena.** Same L1-sharing reason (`alloc_arena` asserts the address the
  programs baked).
- **The ops-head reduce-scatter cache_descriptors fix** (supervision 07:00). Not on this path: the fused head
  does its own reduce-scatter in-program.
- Appendix A's rejections still apply: duplicated norm or router work, the 2D row-major gather, and q_a without
  the TP split.

## Host time no fusion removes (this configuration, after 1-14)

- **Remaining ttnn dispatch: ~2,700 calls, ~250-270 ms.** Single-program ops are at 50-80 us. The per-chip-program
  ops cost 150-200 us: fabric all_gather, a2a 194 us x 80, the flat experts op 473 us x 40.
- **Prepared block programs:** ~250 us x ~120 a chunk.
- **Deallocs:** ~1,700 x 18 us = ~30 ms.
- **Python glue:** ~50-55 ms.
- **Only trace removes the rest.** BREADCRUMBS P.21 lists the blockers (per-start scalars, the address-keyed
  flat-op descriptors, the per-chunk uploads). Entries 2-5 remove some.

## To verify (CPU reading cannot settle these)

- The fused head's patch count per call (`PreparedProgram.patch_count`). Entry 4 depends on it.
- The tail's prepared cost. Estimated from the 117-252 us of the other prepared sites on this box.
- Entry 8's L1 placement on 8x4 torus, with the 1-layer repro.
- Entry 6's cos / sin offset with pre-sliced tables.
- Entry 1's GIL hand-off.

---

# Appendix A: flat MoE on FABRIC_2D (superseded configuration, kept for reference)


Perf review, 2026-10-28, tree 8ad76bad6bb. Proposals only: no code changed, nothing run on the device.

Sources (all warm, this box, tonight):
- P = `/scratch/dnijemcevic/posprof/hostprof_flat.pstats` (flat MoE, shipped default, 5 runs; per chunk = /5).
  `hostprof_blockfix.pstats` = block MoE, ops head, with the overseer's cache_descriptors patch.
- D = `/scratch/dnijemcevic/posprof/results/FLAT0_ops.json.gz` (+ `TBLOCK0`, `BLOCK0`), per-layer device ops.
- Baseline (flat, ctx 0): device 308.1 ms, warm wall 639-672 ms (FLATwall0.log 639.4, supervision 05:10: 671.8).
  Under cProfile, one chunk is 676 ms of host time:
  - 443.1 ms ttnn dispatch (`decorators.py:1217`, 3,575 calls).
  - 44.1 ms `ccl.dealloc` (2,586 calls, **17.1 us each**).
  - 122.4 ms `begin_chunk`: engram `host_rows` 100.3 + `from_torch` 21.8.
  - ~67 ms Python glue.
  The device sits idle during that host time. Every host ms removed is a wall ms saved, until the host drops to
  about device time.

How a call's host cost depends on the op (P, ttnn calls credited to their caller; tottime / calls):

| class | examples (caller) | us / ttnn call |
|---|---|---|
| Python `ttnn.generic_op` over 32 per-chip programs | `fabric_reduce_scatter._fast` (alloc + generic_op, 264 calls, 97.1 ms); `run_planners` (40 calls, 33.3 ms) | ~700 (RS generic_op, if the alloc is ~40 us), 832 (planners) |
| same, through `prepared_program.generic_op` | blockfix: dispatch_prep / alloc_arena / sparse_all_gather_2d (120 calls, 20.9 ms) | 174 avg (117-252) |
| C++ op with one program per mesh coordinate | `rope_stock` (rotary_embedding_indexed, 136 calls, 26.8 ms); `_reshard` (all_to_all_async_generic, 80 calls, 15.9 ms) | 197, 198 |
| fabric all_gather (alloc + op) | `ccl.all_gather` (684 calls, 57.9 ms; 66.0 ms cum with glue for 342 gathers) | 85 per call, 193 per gather |
| single-program C++ op | matmul `mm_configs._run` 80; `norm._fork` (rms_norm) 62; `_sdpa` 57; `attention.write` 52 | 50-80 |
| `ttnn.deallocate` (binding, GLUE_FOLD) | `ccl.dealloc` | 17 |

Removing an op saves its dispatch plus the dealloc of its output: ~70-100 us for a single-program op, and
~200-900 us for a collective or generic_op. Changing a generic_op site to the prepared path saves ~500 us a call
with no fusion at all. That is why entry 1 comes first.

---

### Ranked proposals

#### 1. GENERIC_OP_PREPARED: run every Python generic_op on the flat path through the prepared-program fork

- **Today:** four Python `ttnn.generic_op` sites, 252 calls a chunk. Each call re-applies every runtime arg of 32
  per-chip programs (prepared_program/README.md "What it does").
  - `fabric_reduce_scatter._fast` (`ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/fabric_reduce_scatter.py:1164`):
    132 a chunk. That is 40 attention out (`attention.py:_out_proj` -> `ccl.sum_cols`), 40 moe_add
    (`model.py:487` -> `experts.reduce`), 40 q_a (`ccl.tp_sum`) and 12 compressor / indexer. Shapes 640x5120,
    640x1792, 640x1024 / 512 / 32 bf16 (D). `fabric_reduce_scatter` cum 112.8 ms (855 us a call, all in).
  - `WeightedCombine.run_planners` (`tt/weighted_combine.py:428`): 40 a chunk, 832 us each, 33.3 ms. The
    descriptor is built per chip from inline kernel sources and **never frozen**, so generic_op also hashes 32
    programs' sources on every call.
  - the zero program (`weighted_combine.py:472`; l1_zero freezes its hash) and the column barrier (`:474`). The
    barrier is a 32-program descriptor (`_column_barrier`, `:143`) with an inline `SOURCE_CODE` kernel and **no
    frozen hash**. 80 calls a chunk, inside `WeightedCombine.__call__`'s 53.6 ms for 320 calls.
- **Change:**
  - **(a)** `fabric_reduce_scatter` gets fork option `prepared=True`. It keeps today's address-keyed descriptor LRU
    (`dkey`) and dispatches the cached descriptor through `ttnn.bringup.prepared_program.generic_op`, with its own
    prepared cache so it does not share the 512-entry cap with the MoE block. Because each descriptor only ever
    sees the addresses it was built for, `run_prepared` never patches a slot. That makes the README's exact-match
    caveat (a constant that happens to equal an io address) moot.
  - **(b)** `ttnn.freeze_program_hashes` on the planner and barrier descriptors (the combine_rf modules already do
    this, e.g. `combine_plan.py:2226`), plus dispatch through `prepared_program.generic_op`.
  - `ccl.py:_rs_bf16` / `tp_sum` / `sum_rows` / `sum_cols` pass `prepared=` from the setting. Off = today's
    `program_hash` path.
- **Evidence and estimate:**
  - RS: ~700 -> ~200-250 us (the prepared sites on this box average 174 us; the RS program is larger). 132 x
    ~0.47 ms = **~62 ms**.
  - Planners: 832 -> ~250 us = **~23 ms**.
  - Barrier + zero: **~5-10 ms**. The split inside `__call__` is not measured.
  - **Total ~90 ms a chunk (70-110).** 0 ttnn calls removed, no device change: same programs, same runtime args.
- **Configurations:**
  - Flat: all of it.
  - Block, either head: the attention / q_a / compressor RS (92 a chunk), ~45 ms.
  - Block ops head: moe_block's own `_tp_rs_rows` RS. The overseer's cache_descriptors patch is a prerequisite;
    blockfix shows 491 us a call at 80 calls.
  - The block path's own generic_op sites (`tail_a2a.py:601` 570 us x 80, `combine_plan.run` 788 us x 40,
    `fabric_all_gather_py` 478 us x 40 in blockfix) are candidates for the same treatment. They are Marko's / the
    owner's code, so the owner decides.
- **Accuracy:** bit-identical. The device programs and arguments are unchanged; only host dispatch differs.
- **Effort and risk:** 3-5 agent hours. Fork change in fabric_reduce_scatter_ttnn (Python, no build) plus
  weighted_combine.py.
  - Tests: the fork's regression run with the option off, a prepared vs plain bit-identity case per site, and the
    model gate.
  - Risk is low. Warm-up prepares each descriptor once.
  - 2x2 seed box: same code; the saving is smaller (4 programs a call).

#### 2. ENGRAM_ROWS_ASYNC: engram host rows off the critical path, dequantized in bf16, uploaded 1/TP

- **Today:** `model.py:736 begin_chunk` runs before layer 0 and blocks for **122.4 ms**: 2 engram layers (L1, L14).
  - `engram.py:51 host_rows` 100.3 ms cum, of which `weights.py:179 EngramTable.rows` 63.4 ms: a memmap gather,
    `.float()` 13.2 ms, an fp32 multiply, `.to(bf16)` 5.4 ms. The rest is `torch.zeros` of [5120, 6144] plus the
    slice copy.
  - `engram.py:71 from_torch` 21.8 ms: host tilize plus an upload **replicated 4x over TP**, 7.9 MB per chip,
    32 chips.
  - L14's rows are not read until ~13 layers later; L1's are read after layer 0.
- **Change:**
  - **(a)** `begin_chunk` starts a worker thread per engram layer, L1 first. The worker does the hash, the table
    gather and the dequant, plus a host-only `ttnn.from_torch` (no `device=`: the host tilize). `take_engram_rows`
    joins it and does `ttnn.to_device` on the main thread, so the mesh command queue is still touched by one thread
    only.
  - **(b)** Dequant straight to bf16 into a reused output buffer: `w.view(e4m3).to(bf16) * s.view(e8m0).to(bf16)`.
    This drops the fp32 intermediate, the `zeros` and the copy.
  - **(c)** Upload rows split over TP as well ([S/SP/TP] rows per chip: 1/4 of the bytes), then one fabric
    all_gather over TP on the device (+1 alloc, +1 gather a layer).
  - Rule 5 still holds: the host does only the hash and lookup, and the step stays CPU-tagged in begin_chunk.
- **Evidence and estimate:**
  - L14's work (~50 ms plus its upload) hides behind layers 1-13 (~180 ms of dispatch).
  - L1's work, about half after (b), overlaps embed plus layer 0 (~14 ms of host).
  - (c) cuts the H2D part of `from_torch` by ~3/4.
  - **Saving ~80 ms a chunk (65-95)**, in every configuration (begin_chunk is shared).
  - Device: +2 small TP gathers (640x6144 bf16 per layer, ~0.2 ms each, estimated).
  - This is DS41-99's own "next lever: faster host dequant or a host thread".
- **Accuracy:** bit-identical rows. An e4m3 value times 2^k is exact in bf16 (3-bit mantissa, an exponent inside
  bf16's range), so the bf16 product equals today's fp32 product rounded to bf16. Rows and tail are identical.
- **Effort and risk:** 4-6 h, model code only (tt/engram.py, model.py begin_chunk / take_engram_rows).
  - GIL: torch kernels release it, but a waiting worker may stall up to `sys.getswitchinterval()` (5 ms) per
    Python step while the main thread dispatches. Keep the worker to a few torch calls, or lower the switch
    interval. Measure both.
  - The serving runtime calls begin_chunk the same way, so the contract tests (K.1 smoke) re-run.
  - 2x2: same code (DS41-99: 30.9 ms of begin_chunk there).

#### 3. CCL_PERSISTENT_OUT: persistent output buffers for the per-layer fabric gathers and reduce-scatters

- **Today:** every `ccl.all_gather` (`ccl.py:179`) calls `ttnn.allocate_tensor_on_device`. That is 342 a chunk:
  attn_hc, attn_norm stats, q_a, attention kv, ffn_hc, ffn_norm stats, ffn_norm y, experts xg (40 each), plus 22.
  Every `fabric_reduce_scatter` allocates its output too (132). Each of these 474 outputs is later freed with
  `ccl.dealloc` at 17.1 us.
- **Change:** each per-layer site that consumes its result within the step gets a load-time output (a pair per
  site), passed through the existing `out=` / `output=` arguments. EXPERTS_XG_PINNED is the precedent; this needs
  a generalised, cheaper pin check. The step code then skips their dealloc. Results kept across steps (memo, state,
  Keep) stay freshly allocated.
- **Evidence and estimate:** 474 allocs (not measured alone; ~20-40 us each, the same 32-buffer mesh path as the
  dealloc) plus 474 deallocs x 17.1 us. **~474 ttnn calls removed, ~22 ms a chunk (15-30).** Stable addresses also
  make the RS descriptor LRU (and entry 1's prepared cache) always hit. No device change.
- **Configurations:** all. In TBLOCK the ffn_norm-y and xg gathers sit inside the fused head (~280 gathers + 92 RS).
- **Accuracy:** bit-identical (same programs, other addresses). Device order is FIFO, so reusing a site's buffer
  at the next layer is safe once every reader of the previous value is enqueued. That holds for in-step
  temporaries.
- **Effort and risk:** 4-6 h, tt/ccl.py plus call sites. Medium risk: a missed lifetime is silent corruption.
  Mitigations: a debug assert that a pinned buffer is never deallocated, and the full ladder.
  - Memory: 2 x (640x5120 bf16 ... 640x32 fp32) per site, ~25 MB a chip on 8x4, fits the plan.
  - 2x2: per-site shapes come from the mesh, so it runs as-is.

#### 4. ROPE_UNIFORM_PROGRAM: one RoPE program for the whole mesh instead of 32

- **Today:** `tt/rope.py:83 rope_stock` -> `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed`, 136 a
  chunk: attention q 40 and inverse o 40, kv 40, compressor 8, indexer 8 (D: e.g. 16x640x64, 16x640x512,
  640x512 bf16).
  - The op's `create_mesh_workload` builds one program per mesh coordinate (`my_sp_coord` is a compile-time arg,
    `rotary_embedding_indexed_device_operation.cpp:403`, `:761`). Every hit re-applies run args to all 32
    (`:804`): **197 us a call**, against 50-80 us for single-program ops.
  - The model pre-slices cos / sin per chip and passes `kv_actual_global=0` (rope.py docstring: "every chip reads
    its cos / sin from row 0").
- **Change:** fork it to `ttnn.bringup.rotary_embedding_indexed` (fork_op.py; rule 6, never edit the stock op)
  with option `uniform=True`. It builds one program over the full mesh range with the coordinate term fixed for
  pre-sliced tables, so a hit updates one program. rope_stock passes the option from the setting.
- **Evidence and estimate:** 136 x (197 - ~65) us = **~18 ms a chunk**, in every configuration (attention / kv /
  compressor / indexer are shared). 0 calls removed, no device change: same kernel, and 8 distinct compiled
  binaries become 1.
- **Accuracy:** bit-identical, provided the reader's cos row offset is 0 on every chip for pre-sliced tables.
  Verify first with a model-shaped fork test, uniform vs today.
- **Effort and risk:** 5-7 h. A C++ fork plus build, the fork's source-test baseline (skill 2b) and a model case
  (O.1). Low risk. 2x2: the same calls, smaller saving (4 programs).

#### 5. ROUTING_PACKED: one routing gather instead of typecast + 2 stock gathers (flat only)

- **Today:** `tt/weighted_combine.py:400 routing`: 120 calls a chunk, 16.0 ms (133 us avg).
  - `ttnn.typecast` u16 -> u32 640x6 (40). The fused router emits uint16 ids (`router.py:135`).
  - **stock** `ttnn.all_gather` u32 640x6: 12.95 ms of device for 40 calls, **0.326 ms a call**.
  - stock `ttnn.all_gather` bf16 640x6: 0.026 ms a call.
  - The u32 gather alone costs 12x the bf16 one for the same rows (D, L3.experts).
- **Change:**
  - **(a)** `ttnn.bringup.moe_grouped_topk` gets an output option that packs `id << 16 | bf16(w)` into one uint32
    [s, K] tensor (an extension of the existing fork).
  - **(b)** One `ccl.all_gather` (the fabric fork; `_fits` holds: TILE, dim 2, 640 % 32 == 0).
  - **(c)** The combine_rf column planner gets an option that reads the packed word (planner kernel: ids = hi 16,
    wts = lo 16).
  - A cheaper first step, no kernel change: gather the uint16 ids before the typecast (or the fabric fork for
    both) and typecast the gathered [5120, 6]. That removes the slow u32 gather but keeps the call count.
- **Evidence and estimate:** -40 typecast, -40 gathers, -80 deallocs: **~80 ttnn calls, ~8 ms host a chunk**
  (60 + 100 + 2 x 17 us a layer). **Device about -12 ms** (12.95 -> ~0.6 ms for the remaining gather).
  - Flat only: block heads route inside their own programs.
- **Accuracy:** bit-identical (a pure data move; an id < 2^16 and the bf16 bits pack exactly).
- **Effort and risk:** (a)+(b)+(c) 6-8 h, two forks (moe_grouped_topk C++ plus a build; combine_rf Python), with
  fork tests and a planner table bit-identity test (`tests/unit/test_column_planner_mc.py` style). The cheap step
  is 1 h. 2x2: R = 2 gathers, same code.

#### 6. TP_SUM_ALLREDUCE: tp_sum's reduce-scatter + all_gather as one fabric all-reduce program

- **Today:** `ccl.py:228 tp_sum` (fabric_bf16) = `_rs_bf16` (RS) + `ccl.all_gather` of the columns. 44 a chunk:
  q_a 40 (RS 640x1792 0.111 ms + AG 640x448->1792 0.104 ms device a layer, D L3.q_a) and compressor 4.
- **Change:** a fabric_reduce_scatter option `gather=True`. After its final add, each final core also sends its
  block to the other G-1 chips of the line over the connections the op already opens: the all-gather half done
  inside the same program, with an output of the full width. tp_sum takes it when the option is on.
- **Evidence and estimate:** -44 gathers, -44 allocs, -44 deallocs: **~88 calls, ~9 ms a chunk** (193 + 17 us a
  gather). Device about -2 to -3 ms: one program start and one DRAM round trip fewer a layer (estimated, to
  measure).
- **Configurations:** all (q_a is shared).
- **Accuracy:** bit-identical. The RS adds are unchanged (the same ring order, TP_SUM_BLOCKS=cols) and the gather
  half is a copy.
- **Effort and risk:** 8-10 h, fork kernel work in fabric_reduce_scatter_ttnn (Python ProgramDescriptor, no
  build). Medium risk (fabric flow control and semaphores). The fork's full regression must run with the option
  off. 2x2: G = 2 is the trivial case.

#### 7. NORM_STATS_IN_MHC: mhc_pre_xing writes the norm's partial sum of squares

- **Today:** attn_norm and ffn_norm each start with `ttnn.bringup.rms_norm(x, stats_out=True)` (`norm.py:136`):
  80 a chunk, 640x1280 bf16, 0.006 ms device each. It re-reads the y that `mhc_pre_xing` (attn_hc / ffn_hc) just
  wrote.
- **Change:** a `ttnn.bringup.mhc_pre_xing` option `norm_stats_out` that writes, in the same pass, the
  [S/SP, 32 B] fp32 partial row the rms_norm fork's `partial_stats` reads. `TtDistributedRmsNorm._fork` takes it
  in place of its first call.
- **Evidence and estimate:** -80 rms_norm, -80 deallocs: **~80 calls, ~6.3 ms a chunk** (62 + 17 us). Device
  -0.46 ms and one DRAM read of y a norm fewer. All configurations; TBLOCK keeps attn_norm (ffn_norm already 3
  calls there).
- **Accuracy:** the same quantity, the fp32 sum of the bf16 y's squares, in another add order. Not bit-identical,
  deterministic, at the shipped precision.
- **Effort and risk:** 6-8 h, fork kernel change in mhc_pre_ttnn (C++ build), README / CHANGELOG / tests. Low-medium
  risk. 2x2: same code.

#### 8. MOE_ADD_IN_RS: the routed + shared partial add inside the MoE reduce-scatter (flat only)

- **Today:** `model.py:487 moe_add` = `ttnn.add(a, b)` 640x5120 bf16 (40 a chunk, 1.83 ms device, 116 us host
  each), then the RS of the sum.
- **Change:** a fabric_reduce_scatter option `addend=` (a second local partial of the same shape). It is added once
  to the chip's own blocks where the op reads them (its readers already fetch own pages and feed an add), packed to
  the input dtype before entering the line.
- **Evidence and estimate:** -40 add, -40 dealloc: **~80 calls, ~5.3 ms a chunk**. Device -1.8 ms plus one
  640x5120 DRAM round trip a layer.
- **Accuracy:** bit-identical if the own sum is packed to bf16 before the line, as `ttnn.add(dtype=bf16)` does
  today.
- **Effort and risk:** 5-6 h, fork kernel work in the same Python op. Pairs well with entry 6 (one fork round).
  2x2: same.

#### 9. SETTINGS_CACHED: read the per-call switches once at load

- **Today:** `model_settings.py:37 get` runs 1,614 times a chunk, 5.9 ms cum, each an `os.environ` lookup. Callers:
  - `ccl.fabric` 476, `ccl.bf16_sums` 436, `mm_configs.tuned` 227, `ccl.glue_fold` 132, `bf16_activations` 83;
  - per-call reads in `tp_sum`, `gather_x`, `WeightedCombine.__call__` and `q_a.__call__`.
- **Change:** `ccl.configure()` (already called at model load) and each module's `__init__` store the values;
  per-call code reads the attribute. Behaviour is identical: the A/B harness sets DS41_* before the process
  starts.
- **Evidence and estimate:** **~5 ms a chunk**, all configurations, no device change. 1 h, no risk.

#### 10. ATTN_SCALE_IN_SDPA: the softmax scale in sparse_sdpa, not as a q pre-multiply

- **Today:** `attention.py:543 _q_stock`: `ttnn.multiply(q_resid, 512^-0.5, dtype=bf16)` before wq_b (40 a chunk,
  640x1280, 0.41 ms device). `_sdpa` passes `scale=1.0` (`:584`) and stores the sink in scale-1 units (`:433`).
- **Change:** drop the multiply (q_resid is already bf16 under BF16_ACTIVATIONS=on) and pass `scale=512^-0.5` to
  `ttnn.transformer.sparse_sdpa`. Store `sink / scale`, since the op multiplies the sink by its scale.
- **Evidence and estimate:** -40 multiply, -40 dealloc: **~80 calls, ~4.5 ms a chunk** (the `_q_stock` calls
  average 104 us). Device -0.41 ms. All configurations.
- **Accuracy:** the scale is applied to the fp32 scores, which is where the reference applies it, instead of
  rounding q x s to bf16. One bf16 rounding of q fewer and one of the sink moved; this does not exceed the
  reference's precision. Determinism holds. The ladder must re-run (q's values change by one rounding).
- **Effort and risk:** 2 h, model code only. Low risk. 2x2: same.

#### 11. COMPRESSOR_POOL_SIGMOID: the pair softmax as sigmoid + addcmul

- **Today:** `compressor.py:422 _pool` (the ratio-2 compressors, 3 a chunk): reshape, 4 slices, maximum, 2
  subtract, 2 exp, add, 2 divide, 2 multiply and add, about 17 calls plus about 15 deallocs per call (D: compressor
  slice / subtract / exp / add / div / multiply / maximum rows).
- **Change:** softmax over 2 equals sigmoid of the difference: `acc = kv1 + (kv0 - kv1) * sigmoid(g0 - g1)` with
  `ttnn.sigmoid`, 2 `ttnn.subtract` and `ttnn.addcmul`. These are stock ops; rule 16 allows them, and no INDEX
  entry covers a 2-way gate.
- **Evidence and estimate:** about -8 calls and -8 deallocs per pool: **~48 calls, ~2-3 ms a chunk**. Device about
  -0.1 ms.
- **Accuracy:** an exact identity, with fp32 rounding in another order. Deterministic and within the reference's
  fp32 pooling. 2 h, low risk.

Sum of 1-11, each estimated alone: **~255 ms a chunk** (90 + 80 + 22 + 18 + 8 + 9 + 6.3 + 5.3 + 5 + 4.5 + 2.5;
range about 210-300). Entries 1 and 3 overlap slightly on the RS allocs.
- Flat: warm wall ~640-672 -> **~390-430 ms**, still host-bound against 308 ms of device. Device time also drops
  ~17 ms (entries 5-11), mostly from entry 5.
- TBLOCK (torus, fused head; wall 615 / device 219.5): entries 1 (attention / q_a RS part), 2, 3, 4, 6, 7, 9, 10
  and 11 apply, about 190-210 ms.

---

### Looked at and rejected

- **Attention without the head <-> row reshard** (sparse_sdpa on 16 heads x 640 rows, dropping both
  `all_to_all_async_generic`, `attention.py:436`).
  - Gain: 80 calls, ~17 ms host, and 0.67 ms device a layer of a2a.
  - Cost: each chip would then gather KV for 4x the query rows. sparse_sdpa's 0.348 ms for 64x160 matches a
    KV-read-bound kernel (~100 MB a chip), so about +1 ms device a layer, ~+28 ms a chunk.
  - Rejected: it trades device time for host time and duplicates KV reads.
- **ffn_norm as "gather x, then a full-row norm"** (4 calls -> 2). Every TP chip would normalize all 5120 columns:
  duplicated work (owner rule).
- **Router on the column-gathered x** (dropping both routing gathers). 8x the router's work on every chip:
  duplicated.
- **One 2D (TP x SP) row-major gather of x for the experts.** fabric_all_gather refuses a ROW_MAJOR gather of the
  innermost dim (`ccl._fits`), and the router needs ffn_norm's TILE TP-gathered rows anyway.
- **q_a without the TP split, or the norm summing gathered partials.** The first duplicates the [5120, 1792] matmul
  4x. The second gathers 4x the bytes (~+0.1 ms device a layer) and sums in fp32 where the shipped exchange sums in
  bf16, which is a precision change.
- **Persistent L1 `words` / `relay_words` of the weighted combine** (4 calls a layer). L1 placement order matters
  (`weighted_combine.py:452-454`: the arena must sit on top and the CBs must clear it, as seen on layer 14).
- **Skipping deallocs.** The Python destructor frees the same buffer, so the cost only moves.
- **Shared expert as an extra always-on expert inside the routed flat op** (removes the shared flat op, 322 us x
  40, and the add). Real but large: the flat op's plan, weight layout and cap all change. A candidate for the new
  flat expert op, not this round.
- **Norm stats from an mHC Gram matrix** (no stats gather at all). It needs mhc_post, mhc_pre_xing and rms_norm
  changes, widens the partial row from 32 to 64 columns, and carries cancellation risk. Deferred behind entry 7.
- **RS fused into mhc_post** (the attention and MoE sums feed it directly). High kernel effort for ~80 calls.
- **The block's own generic_op sites** (tail_a2a, combine_plan, fabric_all_gather_py). These fall under entry 1's
  idea but are Marko's / the owner's code (moe_block): reported, not proposed.

### Host time no fusion removes (flat, after 1-11)

- **Remaining ttnn dispatch: ~3,000 calls, ~300 ms.**
  - Single-program ops cost 50-80 us each (matmul 80, rms_norm 62, sparse_sdpa 57, cache writes 52) and are near
    the floor.
  - Collectives with per-chip programs cost ~150-200 us (fabric all_gather; all_to_all_async_generic 198 us x 80).
  - The flat expert ops: shared `flat_routed_expert` 322 us x 40, plus `flat_routed_expert_with_program`.
  - After entry 1, ~200 us per prepared generic_op.
- **Deallocs:** ~2,000 x 17 us = ~35 ms. MeshBuffer::deallocate walks 32 per-device buffers
  (`tt_metal/distributed/mesh_buffer.cpp:394`).
- **Python glue:** ~55-60 ms (run_block, `_free`, step wrappers, shape checks).
- **Only trace removes all of the above.** P.21's ceiling on 2x2: replay 87.0 vs 103.7 ms untraced. The blockers
  are listed in BREADCRUMBS P.21: per-start host scalars and slices, the address-keyed flat / planner descriptors,
  and the per-chunk engram / id uploads. Entries 1-3 remove part of the descriptor and allocation blockers.

### To verify before or with the A/B (CPU reading cannot settle these)

- `ttnn.allocate_tensor_on_device` cost per call. Entry 3's range and the RS generic_op figure (~700 us) depend
  on it.
- The prepared RS cost on 8x4. It is estimated from the block's prepared sites (174 us avg), not measured.
- Entry 4: that the reader's cos / sin row offset is 0 on every chip with pre-sliced tables.
- Entry 2: GIL hand-off under a busy dispatch thread; set the switch interval if the join waits.
