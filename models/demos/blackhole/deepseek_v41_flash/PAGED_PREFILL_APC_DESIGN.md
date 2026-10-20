# DeepSeek-V4.1-Flash: zero-cost prefix caching through vLLM with paged prefill (design and scoping note)

Written 2026-10-19 from a read-only analysis, and revised the same day after the user's direction (next section). No model change and no device run was made for this note.
The revision ran two host-only experiments:
- Q3: a hit-pattern simulation, `tools/apc_hit_sim.py`;
- Q8: a CPU test with the fake model, `tests/test_apc_tracker_hit_binding.py`.

Revised again on 2026-10-20: paged prefill is required (Direction 2). No device run was made: both allowed hosts were busy.

**Structure:**
- **Direction 2 + Part 0 / 0b** are the current plan: the required paged-prefill workstream PF0-PF5, re-sequenced with the host tier. Where they disagree with Part 1, they win.
- **Part 1** (the Direction section and R1-R9) is the host-tier design; its D1 / D3 rows and its R6 milestone table are superseded by Direction 2 / S.4.
- **Part 2** (§0-§7) is the original device-resident analysis. It is kept for the code inventory, the vLLM mechanics and the paged-prefill scoping. Where the two disagree, Part 1 wins.

Code read:

| Component | Branch / location | Commit |
|---|---|---|
| Model | `ssinghal/dsv4p1-final` | b71080881c6 |
| Async snapshots | `ssinghal/dsv4p1-apc-fast` | 76b3fafa239 |
| Plugin (deployed with APC) | `wt/plugin_apc2`, branch `ssinghal/dsv4p1-apc-r` | ff71d29 |
| vLLM 0.31.0 | `wt/v031/venv` | — |

Logs read: `serving_bisect`, `apcfast`, `apc`, `ctr2`.

**Path prefixes:**
- `M/` = `models/demos/blackhole/deepseek_v41_flash/tt/`
- `D/` = `models/demos/blackhole/deepseek_v41_flash/`
- `P/` = `/mnt/tt-data/ssinghal/wt/plugin_apc2/src/vllm_tt_plugin/`
- `V/` = `/mnt/tt-data/ssinghal/wt/v031/venv/lib/python3.10/site-packages/vllm/`
- `L/` = `/mnt/tt-data/ssinghal/dsv4-logs/`

**Evidence tags:**
- **[V]** I verified it in the code at the cited lines.
- **[R]** A research pass read it in the vLLM or plugin source for this note, at the cited lines. Where I re-read a line myself it is tagged [V].
- **[M]** Measured; the log or note is cited.
- **[I]** My inference or estimate; the arithmetic is shown.

## Direction 2 (user, 2026-10-20): prefill must be paged, regardless of APC

> "it should be paged nevertheless": paged attention is a primary requirement.

**What changes:**
- **Paged prefill is a required, first-class workstream (PF0-PF4).** It merges the old M3 (paged prefill attention) with the index-key pages and the paged prefill indexer of the old M5. It is no longer optional H5.
- **After it, a prefill chunk reads and writes only the paged pool and paged key caches.** The dense kv tables, the kv-table FIFO shift, the key FIFO and the prefill→pool latent hand-off are removed.
- **The host tier (Part 1) is re-based on pages.**
  - Write-through = copying finished pool pages + key pages + the small boundary state to the host.
  - A hit = uploading pages + a page-table update.
  - The sequence below runs both workstreams in parallel and keeps the current snapshot APC working until the page-based store replaces it.

**Measurement status:** no device experiment was run for this revision. On 2026-10-20 both allowed hosts (.45 and .46) were running GPQA eval sequences (`L/evals_gpqa_tau3/scripts/evseq.sh` h45_b16 / h46_b4), with `tt-smi -glx_reset` in progress. The ≤ 30 min microbenchmark that would sharpen the estimates (Q16 + Q13, §P.4) is specified, not run.

### Decision table (supersedes the D1 and D3 rows of Direction 1)

| Decision | Answer now | Basis |
|---|---|---|
| **D1** pool dtype | **Open; decided first, by PF0 (Q2a, Q2b, Q16).** Prefill now reads the pool, so the pool dtype is the prefill's KV precision. Default if the measurements are neutral: **P1, a bf16 pool** (bit-exact against today's prefill). Fallback: **P2b**, fp8 latents with a bf16 window (§P.4) | DRAM allows P1 once the tables are gone (§P.5). Its cost is decode attention reading 2x the bytes (Q16). GPU precedent: one KV cache per group, in the format the kernels read (fp8_ds_mla etc.) (`V/models/deepseek_v41/attention.py:1100-1125` [V]) |
| **D2** eviction | unchanged: the host LRU belongs to the connector, as on GPU | Part 1, R1 |
| **D3** 1M | the host tier serves 1M prefix caching. **Paged prefill is required anyway** and frees 1.36 GiB (P1) to 2.62 GiB (P2) per chip at 1M (§P.5) | §P.5 |
| **Q12** bit-exactness | unchanged: not the GPU norm; Q7 decides bounded replay vs halos. With paged prefill, a host hit is bit-exact against cold **in any pool dtype**: it re-uploads the very pages the cold prefill wrote | §S.1 |
| **D4** (new) ordering | **In parallel.** One engineer on the paged-prefill workstream (model + kernels), one on the plugin KV connector (H2a, fronting today's snapshot store). The page-based host store (H1') lands after PF4 | §S.2 |
| **D5** (new) decode index keys | **DECIDED (user, 2026-10-20): stay a dense per-user slab.** It is written in-trace from the chunk's keys and the export is removed. The keys are small: 640 B/token bf16 on the prefill side, 340 B/token bfp8 in the slab, about 20 % of a cached token. The paged decode indexer stays **deferred**: the matmul backend needs contiguous keys, and the fused backend has an unexplained accuracy loss at U=4 and is slower at 1M (75.3 vs 43.8 ms/token, `D/RELEASE_NOTES.md:54`). The prefill-side key pages and the paged prefill indexer (PF1 / PF3) stay as planned | §P.1 |

## Part 0: the paged-prefill workstream (required)

### P.1 What lives where after paged prefill

| State | Today (dense prefill) | After PF4 |
|---|---|---|
| Compressed latents, read by prefill | Per-slot bf16 kv-table FIFO [Up,1,L+128+C,512]: shifted every chunk, gated by the user mask (`M/prefill_sparse.py:72-93, 957-974` [V]) | **Pool pages only**, 320 rows per 128-token page (`M/paged_ops.py:15-16` [V]). Written before the attention of the chunk |
| Compressed latents, read by decode | Pool pages, written by the sink after the attention (`M/prefill_handoff.py:84-92, 204-208` [V]) | the same pages; the hand-off is gone (one write) |
| Index keys, read by prefill | Per-slot key FIFO, tile [Up,1,L,128]; column-sharded `keys_rm` + `keys` (2 copies) from 131,072 entries (`M/prefill_sparse.py:419-550` [V]) | **Paged key caches**, one per owner (2, 8, 14: 64 entries per page; 20: 128 per page), sharing the latent page ids. Tile layout, bf16. Page-strided over the 8 columns from 131,072 entries |
| Index keys, read by decode | Dense per-user bfp8 slab [U,1,n_alloc,128], exported from the FIFO when the prompt ends (`M/dsv41_model.py:550-561, 1210-1235` [V]) | Same slab, written **in-trace per chunk** from the chunk's keys (D5, decided); the export is removed |
| Window rows of the current chunk | Kv-table scratch rows [L, L+128+C) (`M/prefill_sparse.py:996-997` [V]) | **Scratch region in the pool tensor** per prefill slot (128 + C rows), after the ring region |
| Halo (128 window rows before the chunk) | 40 persistent halo tensors [Up,1,128,512] bf16, in-trace update + gate (`M/prefill_attention.py:382, 466-475` [V]) | PF1-PF4: unchanged (the only prefill-private carried state left). PF5 (optional): ring rows in the pool, written at every chunk end; then no prefill-private state at all |
| Decode ring rows | Pool tail, last RING prompt positions (`M/prefill_handoff.py:82-83` [V]) | unchanged (PF5: written for every chunk end) |
| `prev_cs` (ratio-2 compressor, last prompt token) | sink, prompt end (`M/prefill_handoff.py:209-235` [V]) | unchanged |
| Spec-decode taps (layers 37-39, last 128 positions) | stash rows (`M/prefill_taps.py:4-14` [V]) | unchanged; the 128-token replay margin keeps them fresh on a hit |
| Engram history | host (`M/dsv41_model.py:1495-1501` [V]) | unchanged |
| RoPE tables / masks / lim / off | `DynCtx`, global per replay (`M/prefill_dyn.py:131-160` [V]) | visibility in logical entry units (off = 0). PF5: per-row offsets |

**Consequences [I]:**
- **Chunked-prefill continuations need only pages + halos.** After PF5 they need only pages + ring rows, so prefill slots hold no state and the "evicted continuation recomputes from 0" case (`D/INTERLEAVE_NOTES.md:55`) disappears.
- **A window can start at any 128-token multiple:** pages are position-indexed, with no FIFO end alignment.

### P.2 Milestones (each behind a flag and A/B-able against the dense path until PF4)

| # | Scope | C++ / kernel work | Validation | Effort (engineer-weeks) | Main risk / decision point |
|---|---|---|---|---|---|
| **PF0** | Decide the pool dtype: experiments Q2a, Q2b, Q16 (+ Q13 in the same script) | none. Q2b needs an env-gated rounding of the latent writes through fp8 (`ttnn.typecast` bf16→fp8→bf16 in `attend_dyn`), ~10 lines | §P.4 | 0.5-1, plus ~3 device-hours | blocks PF2's kernel choice |
| **PF1** | **Paged writes.** Owner latents scattered into pages *before* the indexer / attention (move the sink's latent scatter). Index keys into the paged key caches (`paged_fill_cache` over the chunk's block ids; page-strided column ownership when sharded). The same keys written in-trace into the decode slab. **Dual-write:** the tables are kept and read, so behaviour is unchanged | none: `paged_scatter_rows` (`M/paged_ops.py:324`), `ttnn.experimental.paged_fill_cache` / `paged_update_cache` with `page_table` (`ttnn/cpp/ttnn/operations/experimental/paged_cache/paged_cache_nanobind.cpp:54-70, 150-163` [V]) | Page rows == kv-table rows bit-exactly (bf16 pool; after the fp8 cast for an fp8 pool). Key pages == FIFO keys. Decode slab == exported slab. Decode logits and state unchanged (extend `tests/test_interleave_device.py`) | 1-1.5 | low |
| **PF2** | **Paged `sparse_sdpa` reads.** Translate the indexer's logical entry ids to pool rows: page-table gather; one translation per (kv source, index source) pair = 8 per chunk (`M/kv_paged.py:86-92, 231-236` [V]). Window ids point to the pool scratch region. `sparse_sdpa` runs on the pool [1,1,R,512] instead of the table | **P1: no kernel change** (`sparse_sdpa` takes any [1,1,T,512] kv with arbitrary row ids; decode already uses it on the pool, `M/paged_attention.py:87-101` [V]). New: an id-translation op (a small dataflow kernel next to `M/paged_kernels/paged_kv_step.cpp`, or ~10 composite ttnn ops). **P2b:** `sparse_sdpa` reader (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_reader.cpp`, `sparse_sdpa_gather.hpp`) + op types / program factory (`sparse_sdpa_device_operation_types.hpp:16-40`, `sparse_sdpa_program_factory.cpp`) to read a second kv tensor (bf16 window) selected by an id range | **Chunk-level bit-exactness** of the attention output against the dense path (P1: same values, same id order), 4 / 40 layers, 2.6k / 16k / 128k tokens, `PIDX_SHARD` 0 / 1. Per-layer PCC for P2b. First-token logits bit-equal (P1) | 1.5-2.5 (+1.5-2.5 for P2b) | P2b kernel; the 8 translations per chunk must stay < 1 % of a chunk |
| **PF3** | **Paged prefill indexer.** `indexer_score_dsa` reads the paged key cache through a page table (logical key tile t → page t / tiles_per_page). Visibility by logical entry (off = 0). Column-sharded variant: each column scores the pages it owns; the existing local top-512 + merge is reused (`M/prefill_sparse.py:552-614` [V]) | **C++:** `ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/reader_indexer_score.cpp` (page translation of key tile reads; work split `indexer_score_work_split.hpp`, causal geometry `indexer_score_causal_geometry.hpp`); `device/indexer_score_device_operation_types.hpp` (optional page-table tensor + entries per page, hashed); `indexer_score_program_factory.cpp` (accessor, compile args); `indexer_score_nanobind.cpp`. The op already takes device-side `chunk_start_idx_tensor` / `valid_end_tensor` (`types.hpp:145-158` [V]): a trace-safe valid length could bound the scoring to the prefix [I] | Replicated: selected ids identical to the FIFO path (same scores, monotonic id map). Sharded: identical up to ties at the 512th score (`D/KV_SEQSHARD_DESIGN.md:53`), so check by quality (needle 2k-1M, GSM8K). Op unit test vs host top-512 (`tests/test_paged_ops_device.py` pattern) | **3-5 (high uncertainty)** | longest C++ item; ring / fused factory variants (`ring_indexer_score_dsa_program_factory.cpp`) must keep working |
| **PF4** | **Remove** `PrefillKV` tables, the latent FIFO shift and its gates, the key FIFOs, the latent hand-off and the key export. Port the APC snapshot code (`M/prefix_cache.py`) to pages (snapshot = latent pages + key pages + halos; restore = page upload + page table + halos) so that **today's APC keeps working** | none | 1M DRAM report (MEMREPORT at the end of build); the 1,000,028-token prompt; needle sweep 2k-1M; GSM8K 256; `serving_bisect` rows within noise; prefill chunk time (expect -1.5 to -4 %, §P.6); APC tests (`test_apc_m2_snapshot_restore.py`, `test_apc_m3_integration.py`) on pages | 1.5-2.5 | trace-region headroom at 1M (3-9 MiB/bank) |
| PF5 (optional) | Halo = pool ring rows: the sink writes the last 128 positions of every chunk into the ring; window ids address the ring + scratch; layers 0-1 move to `sparse_sdpa` with window ids. Prefill slots become stateless; per-row window offsets give 128-token resumes | none | layers 0-1 PCC (SDPA → `sparse_sdpa` is not bit-identical); bit-exact otherwise | 2-4 | couples the halo to pool precision (fp8 ring in P2) |

**Required total PF0-PF4:**
- **7.5-12.5 engineer-weeks** with P1;
- **9-15** with P2b.

Uncertainty is ±50 %, dominated by PF3.

### P.3 C++ / kernel work list (exact files)

1. **`indexer_score_dsa`, paged key reads (PF3):**
   - `ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/reader_indexer_score.cpp`, plus the helpers `indexer_score_work_split.hpp`, `indexer_score_causal_geometry.hpp`, `indexer_score_runtime_args.hpp`;
   - `device/indexer_score_device_operation_types.hpp`, `device/indexer_score_device_operation.cpp`, `device/indexer_score_program_factory.cpp`;
   - `indexer_score_nanobind.cpp`.
2. **Id translation (PF2):** a new dataflow kernel in `M/paged_kernels/` (a `generic_op` like `paged_kv_step`). Alternative: no kernel, composite ttnn ops.
3. **`sparse_sdpa`, second kv tensor (PF2, only if P2b):** `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_reader.cpp`, `sparse_sdpa_gather.hpp`, `sparse_sdpa_device_operation_types.hpp`, `sparse_sdpa_program_factory.cpp`, `sdpa_nanobind.cpp`.
   - Option P2c: the existing `SparseKVFormat::SCALED_FP8` packed rows (`ttnn/cpp/ttnn/operations/transformer/sdpa/sparse_sdpa.hpp:14-18` [V]) would need the pool write kernels (`M/paged_kernels/row_scatter.cpp`, `paged_kv_step.cpp`) to produce scaled rows.
4. **No change** to `paged_scatter_rows`, `paged_fill_cache` or `paged_update_cache` (these already exist with page tables).

### P.4 Precision: prefill now reads the pool

**What the code does today:**
- Prefill reads **bf16** latents and bf16 window rows.
- The shipped decode reads **fp8** latents **and fp8 ring rows** (the window) at every step (`DSV41_POOL_DTYPE=fp8`). Its GSM8K is 95-98 % (`D/APC_CHUNKED_PREFILL_PLAN.md:103-113` [M]).

**The one fp8-prefill measurement:** with the whole prefill table fp8 (`DSV41_PFA_SP_FP8=1`, window scratch included), the per-layer PCC of the sparse attention fell from 0.9995 to 0.997 (`D/PREFILL_BACKLOG.md:12` [M]).

**What that means for a *required* paged prefill [I]:**
- 1 - PCC grows about 6x per layer. It was judged as a 2 % speed optimisation against a strict per-layer gate.
- For a required paged prefill, the bar is end-to-end: GSM8K, needle, and first-token agreement with today.
- That measurement included fp8 *window* rows. A paged prefill can keep the window in bf16 (P2b) or put everything in bf16 (P1).

**Options:**

| Option | Prefill reads | Pool DRAM | Kernel work | Fidelity vs today's prefill |
|---|---|---|---|---|
| **P1** bf16 pool | bf16 latents + bf16 window | x2 (affordable after PF4, §P.5) | none | **bit-exact.** Decode becomes *more* precise than shipped (reads bf16) and gathers 2x the bytes (Q16) |
| **P2b** fp8 latents + bf16 window | fp8 latents (as decode today), bf16 window | as shipped | `sparse_sdpa` second-tensor reader | not bit-exact: Q2b measures it |
| P2a all fp8 | fp8 latents + fp8 window | as shipped | none | the measured 0.997 per layer: Q2a measures end to end |
| P2c scaled fp8 | packed per-row scaled fp8 | ~as fp8 + scales | pool write kernels | between P1 and P2a (not measured) |

**Experiments that settle it**, each ≤ 1 device-hour on .45 / .46 when they are free:

| # | What | How | Measures |
|---|---|---|---|
| Q2a | P2a numerics | existing flag `DSV41_PFA_SP_FP8=1` | first-token logits vs bf16 (max abs, PCC, argmax) on 3 real prompts (2.6k / 16k / 64k); GSM8K 8-shot, 128 questions; needle at 16k / 64k |
| Q2b | P2b numerics | ~10-line debug flag: round only the latent writes through fp8 inside the bf16 table; window rows stay bf16 | same as Q2a |
| Q16 (≤ 30 min microbenchmark, no weights) | `sparse_sdpa` cost bf16 vs fp8 | decode shape (16 rows x 640 ids) and prefill shape (512 x 640 ids) on a 2.67M-row pool, plus the dense-table baseline. Same script: Q13 host↔device rates | decode attention delta x 38 layers = TPOT delta of P1 |

**Rule:**
1. **P1** if Q16 shows ≤ +2 % TPOT.
2. Else **P2b**, if Q2b is within GSM8K noise, the needle is correct, and logits PCC ≥ 0.999 against bf16.
3. **P2a** only if Q2a passes as well; it needs no kernel change.

### P.5 DRAM freed, and what it enables

Per chip. Sizes are computed from the build formulas as in §2.5 of Part 2; free DRAM is MEMREPORT [M].

| | 16 users / 262,144 | 4 users / 1,048,576 |
|---|---|---|
| Free at end of build today | 3.5 GiB | 0.84 GiB (largest block 256 MiB) |
| Removed: kv tables | 0.63 GiB | 2.50 GiB + 0.04 GiB (16,384 context) |
| Removed: key FIFOs (2 copies, sharded) | ~0.04 GiB | ~0.16 GiB |
| Added: key pages, bf16, page-strided over the 8 columns (8,196 / 8,193 pages x 320 x 256 B / 8) | 0.08 GiB | 0.08 GiB |
| Added: pool scratch region (640 rows) | 0.6 MiB | 0.6 MiB |
| **P1 (bf16 pool: +1.25 GiB pages, + rings)** net / free after | -0.68 GiB → **2.8 GiB** | +1.36 GiB → **2.2 GiB** |
| **P2 (fp8 pool)** net / free after | +0.59 GiB → **4.1 GiB** | +2.62 GiB → **3.5 GiB** |

**Also removed:** the transient 16 MiB FIFO-shift pieces and the gate copies. These matter for the 1M build's fragmentation (largest free block 32 MiB/bank).

**What the headroom enables [I]:**
- **Larger prefill chunks at 1M:** C = 1024 needs about +0.3 GiB of activations at Up=1 (0.6 MiB per chunk token, the estimate in `M/kv_paged.py:148-150`), so half the replays per prompt.
- **More trace region** at 1M (today 3-9 MiB/bank free).
- **Room for the vision tower** at 1M.
- **Possibly dropping the 16,384-token second context at 1M,** if PF3's device-side valid length bounds the scoring to the prefix.
- B = 8 at 1M stays out of reach (+1.6 GiB per extra user per row, fp8).

### P.6 Device time per chunk [I]

- **Removed:**
  - the latent FIFO shift, measured 0.9 / 2.6 / 5.2 ms at 32 / 256 / 512 MiB (commit 84a77c9a24c): ≈ 7 ms at 262,144 and ≈ 26 ms at 1M per chunk;
  - the gate pieces;
  - the key-FIFO shifts (~1 ms);
  - the separate window-scratch write into the table, which moves into the pool.
- **Added:** 8 id translations (0.3-2.5 ms); page-table indirection in the indexer reader (same bytes).
- **Net: about -1.5 % (262,144) to -4 % (1M)** of a 0.428 / 0.661 s chunk replay. Q11 measures the shift + gate share.

## Part 0b: re-sequencing with the host tier

### S.1 What paged prefill simplifies in the host tier (Part 1)

- **Write-through = copying finished pages.** A 128-token page is exactly a vLLM block:
  - 320 latent rows (one device per row; replicated over the columns);
  - the key pages (each column its page-strided share).

  There are no kv-table / FIFO tail offsets, no FIFO column-slot arithmetic, and no `CarriedSpec` kinds (`M/prefix_cache.py:39-99`).
- **Boundary state = the 40 halos (5 MiB)** at retained boundaries. After PF5 it is the ring rows, which are also pool rows.
- **A hit = upload the pages into freshly allocated pages of the target row + one page-table update + the halos.**
  - Pages of a request that is not yet running are read by no trace. They can therefore be uploaded asynchronously while the request waits in WAITING_FOR_REMOTE_KVS, on a second command queue.
  - Only the 5 MiB halo restore sits between replays. After PF5 nothing does.
  - The bf16 kv-table critical-path upload of R3 disappears.
- **Bit-exact by construction:** the hit reads the same page bytes the cold prefill wrote, in any pool dtype.
- **Host bytes per token** = what the pool holds:
  - P1: 2,560 + 640 = 3,200 B;
  - P2: 1,280 + 640 = **1,920 B**, i.e. R5's capacities x 1.67.
- **A device "L1" (refcounted pages shared within a row, the old M1) becomes cheap to add later**, because the prefill already reads through page tables.

### S.2 Ordering, with the dependencies

```
H0 (tracker fix) ──┬──────────────────────────────────────────────► ships now
                   └─► H2a: plugin KV connector over today's snapshot store ─────────────┐
PF0 (dtype) ─► PF1 (paged writes, dual) ─┬─► PF2 (paged sparse_sdpa) ─┐                  │
                                         └─► PF3 (paged indexer, C++) ┴─► PF4 (remove     │
                                                                          tables; APC     ▼
                                                                          on pages) ─► H1' page store
                                                                                       + write-through
                                                                                       ─► H3 overlap
                                                                                       ─► H4 (Q7)
```

| Ordering | Pro | Con | Verdict |
|---|---|---|---|
| Paged prefill first, then the host store | the host store is written once, on pages | APC gains wait ~8-12 weeks | — |
| Host store first (Part 1's H1 on the dense tables), then paged prefill | earlier APC gain | the device-side read / restore code is written twice (tables, then pages); ~1-1.5 weeks thrown away | — |
| **Parallel (recommended)** | the plugin connector (H2a) is independent of the device layout. It ships correct vLLM accounting and trusted hits on today's snapshots first. H1' is written once, on pages | needs two engineers; H2a's worker-side restore is swapped from snapshots to pages at H1' | **chosen** |

### S.3 Shippable intermediate steps (current APC keeps working throughout)

1. **H0:** fix the PrefillTracker defect (R8). Ships alone.
2. **PF1 dual-write:** behaviour unchanged; snapshots still come from the tables.
3. **PF2 / PF3 behind flags,** with the tables still written. A/B against dense; snapshots unaffected.
4. **H2a: TT KV connector fronting the snapshot store.**
   - Sync load. vLLM's local prefix caching is off; block hashes still come with a connector (`V/v1/engine/core.py:226` [V]).
   - "Retry" while a donor's snapshot is pending; the generator trusts `start_pos` for connector rows.
   - This removes the "hit the model cannot resume" stall class and the hold-back mis-sizing (`P/scheduler.py:858, 894` [R]) while still on snapshots.
5. **PF4:** tables removed; snapshot APC ported to pages; 1M DRAM gain.
6. **H1':** the page-block store with write-through replaces the pow2 snapshots. The connector switches to page uploads (async).
7. **H3 / H4:** overlap and bandwidth; boundary policy (Q7).

### S.4 Combined milestone list, effort, critical path

| # | Milestone | Effort (engineer-weeks, ±50 %) | Depends on |
|---|---|---|---|
| H0 | PrefillTracker fix | 0.5-1 | — |
| PF0 | Pool-dtype decision (Q2a, Q2b, Q16 + Q13) | 0.5-1 + ~3 device-hours | hosts free |
| PF1 | Paged writes (dual) | 1-1.5 | PF0 |
| PF2 | Paged `sparse_sdpa` reads | 1.5-2.5 (+1.5-2.5 if P2b) | PF1 |
| PF3 | Paged prefill indexer (C++) | 3-5 | PF1 |
| PF4 | Remove tables / FIFOs / hand-off / export; APC on pages | 1.5-2.5 | PF2, PF3 |
| PF5 (optional) | Halo in pool ring; stateless slots; 128-token resumes | 2-4 | PF4 |
| H2a | Plugin KV connector (snapshots first) | plugin 3-5 + model 1 | H0 |
| H1' | Page-block host store + write-through | 2-3.5 (H1 was 3-5 on tables) | PF4, H2a |
| H3 | Overlap / bandwidth (second command queue, async page upload) | 1-2 | H1', Q13 |
| H4 | Boundary policy (Q7); bounded replay simpler after PF5 | 1-2 | Q7 (+ PF5 for the 128 grid) |
| H6 (optional) | Generated-token reuse, NVMe tier | 2-4 | H1' |

**Totals [I]:**
- **Required paged prefill (PF0-PF4):** 7.5-12.5 engineer-weeks (P1) / 9-15 (P2b).
- **Host tier (H0, H2a, H1', H3, H4):** 8.5-15.5 engineer-weeks, of which 3-5 are plugin work.
- **Together:** 16-30.5 engineer-weeks. With two engineers in parallel, about 10-17 calendar weeks.

**Critical path:** PF0 → PF1 → **PF3** → PF4 → H1' → H3 / H4. That is about 9.5-17 engineer-weeks in series; PF3 (C++ `indexer_score_dsa` paged reader) is the longest and least certain item. H2a runs alongside it.

### S.5 What to measure first

On .45 / .46 when free; ssh for device commands; never .44 / .35 / .33 / .34 / .48 / .42 / .43 / .40 / .29 / .30 / .31 / .41.

1. **Q16 + Q13:** one ≤ 30 min microbenchmark without weights. It gives the `sparse_sdpa` bf16 vs fp8 cost (decode + prefill shapes, pool vs dense table) and the host↔device rates per mesh row. It decides P1's cost and the hit-upload model.
2. **Q2a** (existing flag) and **Q2b** (10-line flag): fp8-prefill accuracy end to end. Decides D1.
3. **Q7:** bounded-replay accuracy. Decides H4 and the host bytes per token (R5).
4. **Q11:** shift + gate cost inside the trace. Sizes PF4's speed-up.
5. **Q14:** server host RSS. Decides the host store budget.

## Direction (user, 2026-10-19)

> "depends on how it is done in general for GPU; if it requires a lot of DRAM, then it should be host, especially if different users > 16 are supposed to have caching stored"

**What changes.** The plan below (Part 1) replaces the device-resident plan of Part 2.
- **The host store is the cache of record.** Its capacity scales with host RAM (566 GiB per device host), independently of device DRAM and of the 16 / 4 user slots. Prefixes of users beyond the live 16 (or 4) stay cached.
- **Device pages hold only active requests.**
- **Write-through instead of snapshots.** Finished 128-token blocks, plus the halos at boundaries, are copied to the host during prefill, overlapped with compute, and keyed by vLLM's block hashes.
- **A hit uploads the needed blocks** into the user's row, whichever row it is.

**What this resolves.** Row privacy, the bf16-pool DRAM question and the 1M DRAM budget all stop being blockers.

**Precedent.** This is how upstream GPU vLLM 0.31 does a CPU tier: its OffloadingConnector (§R1).

### Answers to the open decisions under this direction

| Decision | Answer | GPU precedent (vLLM 0.31) and reasoning |
|---|---|---|
| **D1** bf16 pool at 262,144? | **Superseded by Direction 2** (prefill now reads the pool; PF0 decides the dtype). The original answer: no, the pool stays fp8. | Write-through copies the **bf16** latents from the prefill kv table to the host. A hit restores the bf16 table exactly and writes the fp8 pages through the converting `paged_scatter_rows` (`M/paged_ops.py:324-341` [V]), the same conversion a cold prompt uses. So the hit is bit-exact against cold with an fp8 pool. Upstream offload copies whatever format the device cache holds, including the packed DSv4 layout as one region (`V/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:133-162` [R]), and never keeps a second precision. The pool dtype only comes back as a question if paged prefill (R6 H5) is adopted. |
| **D2** must vLLM own eviction? | **The connector framework owns it, as on GPU.** No vLLM-owned device blocks (M2 dropped). | The upstream CPU tier is evicted by the connector's own `OffloadingManager` (LRU by default, ARC available, pluggable), and the device KV manager is never told. Host events reach only the external KV-event publisher (`V/v1/kv_offload/cpu/spec.py:135-138`, `cpu/manager.py:304-336`, `V/v1/core/sched/scheduler.py:2312-2325` [R]). On TT the device retains nothing after a request, so vLLM's local prefix cache is switched off. The TT connector answers every lookup from its host index; it gets the block hashes even with local caching off (`V/v1/engine/core.py:226` [V]: hashes are built if prefix caching **or** a KV connector is on). |
| **D3** device-resident APC at 1M? | **No.** 1M prefix caching is the host tier. **Paged prefill is now required** (Direction 2), not optional. | One 1M prefix is 3.4 GB on the host without halos and 14.1 GB with a halo every 512 tokens (R5). In the Q3 simulation a per-row device cache at 4 users / 1M recovers only 7-15 % of prompt tokens in a multi-tenant mix, against 65-96 % for a 256 GB host store (R7). |
| **Q12** must a hit be bit-exact against cold? | **Not the GPU norm. Decide by Q7.** | Upstream DeepSeek-V4.1's default is *bounded replay*: the sliding-window KV is not cached, the last 128 tokens of a hit are recomputed with the window clamped, so the hit is not bit-exact (`V/config/cache.py:242-245`, `V/v1/attention/backends/mla/sparse_swa.py:366-371` [V]). GPU vLLM gives no batch-invariance guarantee either. The practical bar is GSM8K + needle equivalence. If Q7 passes, drop the halos (4.2x less host RAM than exact halos every 512 tokens, hits at 128-token granularity). Otherwise store exact halos every 512 (or, as a compromise, every 2,048). |

## Part 1: revised plan, host tier first

### R1. The GPU precedent in vLLM 0.31 [R; read in the source, file:line cited]

The paths are relative to `V/`.

**Native CPU offload:** `OffloadingConnector` + `v1/kv_offload/`.
- **Enabling it:** `--kv-offloading-size` (GiB, default off) and `--kv-offloading-backend native|lmcache` (default native) (`config/cache.py:256-262`; `config/vllm.py:1130-1159`).
- **Keys:** block hash + group index (`v1/kv_offload/base.py:22-30`). A chunk is `blocks_per_chunk` device blocks, default 1 (`.../offloading/config.py:98-135`).
- **Write-through:** every step, for the chunks this step's forward completes. The store is submitted at the start of the next step, "AFTER transfers related to token sampling" (`.../offloading/scheduler.py:1507-1525`; `.../offloading/worker.py:246-271`). Only prompt blocks are offloaded by default (`offload_prompt_only=True`, `v1/kv_offload/base.py:603-609`).
- **No phantom hits:** a store is inserted with ref_cnt -1 ("not ready"). A lookup then returns HIT_PENDING and the scheduler retries; the chunk becomes hittable only when every worker has reported the store complete (`cpu/policies/base.py:25-45`; `cpu/manager.py:187-194`; `.../offloading/scheduler.py:694-703, 1883-1900`).
- **Loads are async:** `(hits, True)`. The request waits in WAITING_FOR_REMOTE_KVS; `prepare_load` pins the chunks so they cannot be evicted (`.../offloading/scheduler.py:1094, 1187-1209`; `V/v1/core/sched/scheduler.py:1258-1288, 3099-3113`).
- **Eviction:** LRU or ARC; only ready, unpinned chunks can go. When no space is left, a store is silently dropped and becomes a future miss (`cpu/manager.py:304-319`).
- **Transfers:** a stream pool, ordered behind compute; completion is polled via events (`cpu/gpu_worker.py:286-291, 690-759`).
- **Hybrid support:** both offload connectors implement `SupportsHMA`, and DeepSeek-V4's packed layout is offloaded as one region per block (`.../offloading/worker.py:133-162`).

**Other connectors:**
- `SimpleCPUOffloadConnector`: a mirrored CPU BlockPool with eager or lazy store (`v1/simple_kv_offload/manager.py:279-310, 593-654`).
- `LMCacheConnectorV1` / MP: no HMA support, and `lmcache` is not installed here.
- The factory lists more (`.../factory.py:153-249`).

**Upstream GPU DeepSeek-V4.1** offloads only the prefix-cacheable groups: compressed latents and indexer keys (MLA specs). It stores **no halo**: bounded replay recomputes the last window of every hit (`models/deepseek_v41/attention.py:519-546, 1100-1195` [V]).

**TT [I].** Same structure:
- a TT connector (`TTHostPrefixConnector`) in the plugin;
- the block store in host RAM of the engine-core process (uni executor, `P/platform.py:591` [V]);
- device-to-host reads queued behind the prefill replays;
- uploads before the request's first window.

The stock CPU spec cannot be reused: it raises on anything but CUDA/XPU (`v1/kv_offload/cpu/spec.py:200-206` [R]). An out-of-tree spec can be loaded via `spec_module_path` (`v1/kv_offload/factory.py:30-51` [R]), or the plugin can implement its own connector.

### R2. Write-through: what is copied, when, and keyed by what

**Per finished 512-token window and user (Up = 1: 4 users per replay, one per row) [I]:**

| Item | Source on device | Bytes per window | Reads per window |
|---|---|---|---|
| Latents of the 4 kv sources (bf16) | kv-table rows of the window's new entries, **one** device of the row (replicated over columns) | 512 x 2.5 KiB = 1.25 MiB | 4 slices |
| Index keys of the 4 owners (bf16) | key FIFO, column-sharded: 8 shards per owner | 512 x 640 B = 320 KiB | 32 |
| Halo of the 40 layers, only at retained boundaries (R4) | persistent halo tensors, one device | 5 MiB | 40 (1 if batched into one buffer) |

**Host block format.**
- One 128-token block = 320 latent rows x 512 bf16 + 320 key rows x 128 bf16 = **400 KiB**. Shared prefixes are stored once (deduplicated by hash).
- A halo is attached to the block that **ends** at its boundary, under key = (block hash, "halo"), like upstream's (hash, group) keys.

**Keys.** vLLM's chained block hashes (sha256 over parent hash + tokens + extra keys: multimodal identifiers / LoRA / cache salt; `V/v1/core/kv_cache_utils.py:614-714` [R]).
- The plugin passes `request.block_hashes` in the connector metadata.
- Image blocks become cacheable through the multimodal extra keys, so the vision guard (`M/prefix_cache.py:509-512`) can go.
- The hashes are stable across restarts with the default seed (`kv_cache_utils.py:118-172` [R]), which would allow a persistent NVMe tier later.

**Pending until complete.** A block becomes *ready* only when its read has completed and been materialised, and the worker has reported it in its connector output. Until then lookups see HIT_PENDING (upstream pattern). Same-step phantom hits are therefore impossible: a same-step dependent either waits (`None` = retry) or takes the shorter ready hit.

**Eviction.**
- Host LRU by bytes under a RAM budget (R5).
- Only ready, unpinned blocks are evictable. Blocks are pinned from lookup until the load completes. A halo is evicted with its block.
- A dropped store is a future miss. vLLM is told nothing; KV events (BlockRemoved, medium "CPU") are optional, for monitoring.

**Cold-path cost (estimates from measured figures) [I]:**
- **Engine-thread enqueue:** about 16 µs per queued read (168 queued reads took 2.7 ms) and 65 µs per slice (20 slices took 1.3 ms) (`tests/test_apc_snapshot_cost.py` docstring, `L/apcfast/ut_cost1.log` [M]).
  - Exact halos: ~1.5 ms per user-window → **~6 ms per 428 ms replay (1.4 %)**.
  - No halos: ~3.6 ms (0.8 %).
- **Device-side read time:** queued reads of 533 MiB completed about 72 ms after the start (40-46 ms enqueue + 26-32 ms synchronize), i.e. about 7-8 GB/s.
  - Per replay: 26 MiB (exact halos) ≈ 3.5 ms (0.8 %), or 6.3 MiB (no halos) ≈ 1 ms.
  - Today these reads run on CQ0 between replays. A second command queue would hide them completely (Q13).
- **Host-side conversion to torch** ("materialize", 116-153 ms per 533 MiB [M]) runs on a worker thread, off the engine thread.
- **Compared with today:** the async pow2 snapshots cost 8-9 ms of engine thread *per snapshot* and re-read the whole prefix each time. Write-through reads each token once, and every window boundary becomes hittable.

### R3. A hit: an upload into the user's row, with a cost model

**Steps:**
1. **Scheduler (TT connector).** Find the deepest s such that:
   - s is on the grid (512 with halos; 128 with bounded replay);
   - s ≤ N - 128;
   - every block in [0, s) is ready, and the halo at s exists if needed.

   Return (s, async = True). vLLM sets `num_computed_tokens = s`; the request waits in WAITING_FOR_REMOTE_KVS.
2. **Worker.** Upload the blocks [0, s) and the halo into the request's prefill slot, onto **whatever row the request was placed** (row privacy no longer matters):
   - the bf16 kv-table tail, then a device-local scatter into its pool pages (converted to fp8);
   - the key-FIFO tail;
   - the halos;
   - the Engram history, derived from the tokens on the host.

   Then report `finished_recving`.
3. **Model.** Start exactly at s; the generator trusts `start_pos` for requests the connector loaded. vLLM's hit then equals what the model skips; the ~31 s stall class of bug disappears.

**Host bytes per hit of P tokens:** P x 3,200 B + one 5 MiB halo.

Today's restore path manages about 0.2 GB/s. Its reasons: pieces of 2,048 rows, a host tensor zero-padded for all 4 rows and replicated over the 8 columns, and the latents written twice (`M/prefix_cache.py:116-148, 375-380` [V]). The efficient-path rates are assumptions until Q13 measures them.

| Prefix P | Bytes | 0.2 GB/s (today's path; measured 7.6-8.7 s at 524,288, `D/RELEASE_NOTES.md:19`) | 2 GB/s | 8 GB/s | For scale |
|---|---|---|---|---|---|
| 100k | 0.33 GB | 1.6 s | 0.16 s | 0.04 s | Cold: 107-124 s. Today's hit: 39-46 s, because the pow2 grid leaves 35k tokens to recompute (`D/APC_CHUNKED_PREFILL_PLAN.md:114` [M]) |
| 524,288 | 1.68 GB | 8.4 s | 0.84 s | 0.21 s | — |
| 1,048,576 | 3.36 GB | 16.8 s | 1.7 s | 0.42 s | Cold at 300k on the 1M build: 504 s [M] |

After the upload, the tail recompute is at most 1 window with exact halos every 512 (0.43 s at 262,144, 0.66 s at 1M [M]), and up to 4-5 windows with halos every 2,048.

**Overlap [I]:**
- The fp8 **pool pages** of the loading request are read by no running trace. They can be uploaded on a second command queue while other requests compute (WAITING_FOR_REMOTE_KVS keeps the request out of the batch).
- The **prefill slot state** (kv table, key FIFO, halos) cannot be written while a replay runs.
  - The masked trace writes back old values for masked slots: `gate_user_state` does `where` + `copy` over the whole tensor (`M/prefill_attention.py:55-80`, `M/prefill_sparse.py:957-974` [V]). A concurrent upload would race with it.
  - So the slot state is either uploaded into a staging buffer on the second queue and copied between replays (device-local, ~ms), or uploaded on the critical path.
  - At 1M the DRAM is too tight for a full staging copy: stream it in pieces.
- **Paged prefill (H5)** removes the bf16 kv table, so the only latent upload goes to pages and can run fully async. It needs a bf16 pool (§2.4).

### R4. Boundary state: exact halos or bounded replay (Q7)

| Option | Hit grid | Host bytes per token | Hit fidelity | Model work |
|---|---|---|---|---|
| **Bounded replay** (upstream default) | 128, any block | 3,200 B | approximate: replays [s-128, e) with the window clamped at s-128 (Q7 measures it) | Window mask that treats the halo as absent at s0 > 0 (`M/prefill_dyn.py:146-153`). Per-row window offsets, so that starts not on the 512 grid can share a replay (old M1.5) |
| Exact halo every 512 | 512 | 13,440 B (3,200 + 10,240) | bit-exact against cold (today's restore is) | as today's restore |
| Exact halo every 2,048 | 2,048 | 5,760 B | bit-exact; the tail costs up to 4-5 windows | as today |
| Halos at powers of two + each prompt's last full window | sparse | 3,200 B + ≤ 13 x 5 MiB per prompt | bit-exact, but the hit falls to the previous power of two whenever prompts diverge between two boundaries (e.g. a 100k document → 65,536) | as today |

### R5. Host memory sizing

**Per token, one copy (not per column or row, bf16) [I]:** latents 2,560 B + index keys 640 B = **3,200 B**, plus 5 MiB per retained halo.

Users below means full-length prefixes; GB = 10^9.

| Halo policy | Bytes per cached token | One 262,144-token prefix | One 1,048,576-token prefix | Users in 256 GB (262k / 1M) | Users in 400 GB (262k / 1M) |
|---|---|---|---|---|---|
| None (bounded replay) | 3,200 | 0.84 GB | 3.36 GB | 305 / 76 | 476 / 119 |
| Powers of two + last window | 3,200 + ≤ 68 MB/prefix | 0.90 GB | 3.42 GB | 285 / 74 | 446 / 117 |
| Every 2,048 | 5,760 | 1.51 GB | 6.04 GB | 169 / 42 | 265 / 66 |
| Every 512 (exact, 512 grid) | 13,440 | 3.52 GB | 14.09 GB | 72 / 18 | 113 / 28 |
| Every 128 (exact, 128 grid) | 44,160 | 11.58 GB | 46.31 GB | 22 / 5 | 34 / 8 |

**Shared prefixes count once.** A 2k system prompt shared by every user costs 6.5 MB (+ halos).

**Real budget.** Each device host has 566 GiB (`free -g` on .45 and .46 [M]).
- The server already holds the Engram tables in RAM with `DSV41_ENGRAM_RAM=1`, the default: **188.8 GiB** (fp8 + scales of `layers.{1,14}.engram.embed`, from the checkpoint headers [V]).
- The rest of the server process is unmeasured (Q14).
- **A realistic store budget is ~250 GiB.** 400 GB would need the Engram tables memory-mapped (`DSV41_ENGRAM_RAM=0`; speed impact unmeasured) or a second tier (NVMe, as upstream's tiering).

### R6. Revised milestones (superseded by S.4: H5 became the required PF0-PF4, H1 became H1' on pages, H2 became H2a)

| # | Milestone | Effort (engineer-weeks, ±50 %) | Validation | Decision point / risk |
|---|---|---|---|---|
| H0 | Keep the shipped async snapshots (76b3fafa239) as the fallback. **Fix the PrefillTracker defect** (R8): the plugin passes a stable request key or a new-vs-continuation flag | 0.5-1 | `tests/test_apc_tracker_hit_binding.py` must turn from xfail to pass | — |
| H1 | **Host block store + write-through, model-side.** Our 128-token chain hash, keyed like vLLM's. Per-window queued reads (R2); deduplicated host store with LRU by bytes; hit = restore from blocks into any row on the 512 grid, using an efficient per-row upload. Replaces `PrefixStore` snapshots | 3-5 | Bit-exact hit vs cold: the existing `test_apc_m2_snapshot_restore.py` / `test_apc_m3_integration.py` composition, sourced from blocks. Host bytes match R5. `serving_bisect` 16 x 1024 row within noise of APC off. Prefix sweep: 100k hit TTFT expected to fall from 39-46 s to ≈ upload + 1-2 windows. GSM8K | Stop if cold overhead > 2 % (engine thread + device) or the upload is < 1 GB/s after H3 |
| H2 | **vLLM integration as a KV connector** (`TTHostPrefixConnector`, OffloadingConnector-shaped). Keys = vLLM block hashes; pending / ready states; async load (WAITING_FOR_REMOTE_KVS); vLLM local prefix caching off; TT model runner calls the connector each step (it has no connector wiring today, §4.5); reset hook; vision guard removed | plugin 3-5, model 1-2 | Mock-scheduler tests: hit = exact resume point, pending → retry, pin / unpin, eviction under the budget, reset. Servers as in H1. Vision prompts that share text and images | Plugin owner. Lane mode hard-codes `connector=None` (not needed under this plan) |
| H3 | **Overlap and bandwidth.** Second command queue for reads and uploads; staging + device-local copy between replays; batched halo reads; column-sharded upload + all-gather instead of a replicated upload | 1.5-3 | Q13 numbers; TTFT of 100k / 524k / 1M hits | Decides whether H5 is needed |
| H4 | **Boundary decision (Q7).** Bounded replay (window clamp + per-row window offsets) or exact halos every 512 / 2,048 | 1.5-3 | Q7, then needle sweep 2k-1M + GSM8K, hit vs cold | User sign-off on approximate hits |
| H5 (optional; old M3 demoted) | Paged prefill attention: latents only in (bf16) pages; async upload; kv tables dropped (2.5 GiB per chip at 1M) | 4-6 | §6 M3 of Part 2 | Only if H3 leaves the critical-path slot restore too slow, or 1M DRAM is needed |
| H6 (optional) | Reuse of generated tokens (decode blocks + key pages; multi-turn chat) and a second tier (NVMe) beyond RAM | 2-4 | multi-turn accuracy | Product priority |

**Dropped or demoted from Part 2:**
- **M1** (device-resident page cache): demoted. At most a later "device L1" of recently freed pages.
- **M2** (vLLM-owned device blocks via lane-DP): dropped.
- **M3**: becomes H5, optional.
- **M4** (cross-row copy): dropped. The host is shared by all rows.
- **M5** (paged indexer): dropped.

**Total to a GPU-equivalent host tier: H0-H4 ≈ 10-19 engineer-weeks.**

### R7. Q3 result: host-only simulation of hit patterns (`tools/apc_hit_sim.py`, run 2026-10-19, no device)

**Method.**
- Client workloads rebuilt token-exact with the checkpoint tokenizer and the bundle's chat template:
  - `L/apc/apc_client.py` default run + mixed runs: 98 requests, 165,070 prompt tokens;
  - its `--gsm 128` run: 131 requests, 94,611 tokens;
  - `L/apc/apc_prefixsweep.py`: 18 requests, 666,540 tokens.
- Total: 247 requests, 926,221 prompt tokens.
- Scheduler model:
  - FCFS steps under a token budget;
  - 4 rows x U slots;
  - blocks written in step k become hittable from step k+1 (the "ready" rule of R2);
  - hit = deepest grid boundary ≤ N - 128.
- Device cache: per row, with row-private pages, LRU over the row's free pages.
- Host cache: one shared store, LRU by bytes, 3,200 B per token + 5 MiB halos every 512 tokens (or none with bounded replay).
- Placement: `bal` = today's least-loaded row; `aff` = row with the deepest cached prefix.

**Measured traces:**

| Layout, budget | Device per-row hit share of prompt tokens (bal / aff) | Any row | Hitting requests whose own row holds less (bal / aff) | Host store hit share at 1 GB / ≥ 4 GB |
|---|---|---|---|---|
| 16 users, 4,096, grid 512 + halos | 0.666 / 0.666 | 0.668 | 5 / 4 of 239 = 2.1 % / 1.7 % | 0.453 / 0.668 |
| same, budget 67,584 | 0.662 / 0.662 | 0.668 | 5.0 % / 5.0 % | 0.453 / 0.668 |
| 4 users (1M layout), 4,096 | 0.667 / 0.667 | 0.668 | 1.3 % / 1.3 % | 0.453 / 0.668 |
| 16 users, grid 128, no halos (bounded replay) | 0.669 / 0.670 | 0.672 | 2.1 % / 1.7 % | 0.672 already at 1 GB |

**Per workload (16 users):**
- client: 0.661 / 0.664 (5.2 % / 4.2 % cross-row);
- GSM8K: 0.676 / 0.682 (3.8 % / 3.1 %);
- sweep: 0.661 (0 %).

The measured benchmarks have tiny working sets: under 4 GB even with halos every 512. Their ceiling of about 0.67 is compulsory, i.e. first occurrences. **They cannot tell a host tier from a device cache.**

**Synthetic multi-tenant workload** (labelled SYNTHETIC; not a measured trace).
- Each request: a shared 2,048-token system prompt + the user's private document + a fresh 100-token question.
- Users drawn Zipf(1).
- 4,000 requests; 2,000 for the 128k documents; 400 for the 1M documents.

| Case (layout) | Device per-row hits (bal / aff) | Any row | Cross-row (bal / aff) | Host, exact halos every 512: 4 / 16 / 64 / 128 / 256 / 400 GB | Host, bounded replay: 16 / 64 / 128 / 256 / 400 GB |
|---|---|---|---|---|---|
| 16 users x 32k (16u) | 0.968 / 0.971 | 0.979 | 1.3 % / 0.9 % | 0.770 / 0.979 / 0.979 / … | — |
| 64 x 32k (16u) | 0.769 / 0.940 | 0.932 | 17.7 % / 2.9 % | 0.452 / 0.812 / 0.968 / 0.968 / 0.968 / 0.968 | — |
| 256 x 32k (16u) | 0.535 / 0.742 | 0.737 | 21.9 % / 4.1 % | 0.304 / 0.566 / 0.853 / 0.924 / 0.924 / 0.924 | 0.872 / 0.934 / 0.934 / 0.934 / 0.934 |
| 64 x 128k (16u) | 0.374 / 0.629 | 0.641 | 27.4 % / 6.0 % | 0.130 / 0.439 / 0.816 / 0.964 / 0.964 / 0.964 | 0.834 / 0.967 / 0.967 / 0.967 / 0.967 |
| 16 x 1M (4u) | 0.129 / 0.147 | 0.378 | 25.8 % / 23.8 % | 16 GB 0.117, 64 0.426, 128 0.723, 256 0.960, 400 0.960 | 0.426 / 0.960 / 0.960 / 0.960 / 0.960 |
| 64 x 1M (4u) | 0.067 / 0.082 | 0.241 | 18.3 % / 16.8 % | 16 GB 0.097, 64 0.254, 128 0.476, 256 0.648, 400 0.750 | 0.254 / 0.658 / 0.820 / 0.860 / 0.860 |

**Reading:**
- With more than 16 users, or with long documents, a device-resident cache is capped by the row pools: one 1M prefix per row at 4 users, about 1M tokens per row at 16 users. The host tier is not capped this way.
- Prefix-affine placement matters a lot for a device cache (17-27 % cross-row → 3-6 % at 16 users) and does not matter for the host tier.
- Halos every 512 cost about 4x the host capacity compared with bounded replay. For example 256 x 32k at 16 GB: 0.566 with halos vs 0.872 without.
- Saturation at 0.92-0.97 is the compulsory-miss ceiling of the synthetic stream.

### R8. Q8 result: the PrefillTracker defect is reproduced (`tests/test_apc_tracker_hit_binding.py`, CPU, fake model)

Both scenarios of §4.7 reproduce on the current code. The tests assert the correct behaviour and are marked `xfail(strict=True)`; with `--runxfail` they fail as shown.

**Scenario 1.** A is a 1,024-token prompt, prefilled but not yet decoded. B is a new request with A's 1,024 tokens + 200 more, and vLLM hit = 1,024.
- B's call runs `[(user 0, 1024, 1224)]`, i.e. on **A's user**.
- The slot map becomes `[1, 0, 2, 3]`.
- A's next decode raises `RuntimeError: decode of logical slot 0 (user 1) that was never prefilled`.
- That is a request failure, and possibly the engine.

**Scenario 2.** A is mid-prefill at 512. B is new and shares A's first 512 tokens, hit = 512.
- B runs `[(user 3, 512, 812)]` on A's user.
- A's next piece is then recomputed from 0 on another user: `[(1, 0, 1024)]`. The result is correct but wasted.

**Fix (H0).** The model must tell a new request from a continuation, either with a stable request key from the plugin or with a new-vs-continuation flag. Matching on the token prefix alone is not enough.

**Existing suite.** `tests/test_vllm_interface.py` passes apart from the 6 known spec tests that need `vllm_tt_plugin`, which is not in `python_env`.

### R9. Still undecided

Decided since this list was written: **D5** (user, 2026-10-20). The decode index keys stay a dense per-user slab and the paged decode indexer is deferred; the prefill key pages (PF1 / PF3) stay. See the Direction 2 decision table.

1. **Q7:** bounded replay (approximate, no halos) vs exact halos, and if exact, whether every 512 or every 2,048 tokens. Needs the measurement plus user sign-off.
2. **Host RAM budget for the store.** Engram in RAM (188.8 GiB) vs memory-mapped; the server's RSS is unmeasured (Q14). Also whether a second tier (NVMe) is wanted.
3. **Host-to-device and device-to-host bandwidth** of the efficient per-row path (Q13). It sets the hit TTFT at 524k / 1M and whether H5 (paged prefill) is needed.
4. **Plugin ownership of H2** (TT KV connector; runner wiring) vs keeping the store model-internal behind the plugin's hit-rounding hook for longer.
5. **Priority of multi-turn reuse of generated tokens (H6).** Upstream offloads prompt blocks only by default.

**New open questions:**

| # | Question | Experiment | Effort |
|---|---|---|---|
| Q13 | Host↔device bandwidth per mesh row | Upload [N,512] bf16 into one row: (a) replicated, (b) column-sharded + all-gather over the 8 columns. Non-blocking reads from one device per row. CQ0 vs CQ1 overlap with a running replay | ≤ 45 min on .45 / .46 |
| Q14 | Host RSS of a running server, and therefore the store budget | Read `/proc/<pid>/status` VmRSS of the shipped 16-user / 262,144 and 4-user / 1M servers after warm-up; no load | 0 device-hours beyond an existing server run |
| Q15 | TT runner connector wiring | CPU mock: `TTScheduler` + a stub connector returning async hits; check WAITING_FOR_REMOTE_KVS → `finished_recving` → `num_computed_tokens` reaches the TT model runner as `start_pos` | 0 device-hours |


## Part 2: original device-resident analysis (2026-10-19, kept for reference)

Superseded by Part 1:
- the summary's plan (item 8);
- §3's recommendation of (b) + (c) (the host tier is shared by all rows);
- §5's zero-cost path;
- §6's milestone list (see R6);
- Q3 / Q8 in §7 (answered in R7 / R8).

Still valid:
- the state inventory (§1);
- the paged-prefill scoping (§2), now the required PF workstream (Part 0);
- the vLLM mechanics (§4).

### 0. Summary of the device-resident analysis

1. **Today's prefix cache is host-side.** APC takes host snapshots of one user's dense prefill state at power-of-two chunk boundaries (`M/prefix_cache.py:4-24`) and restores them into any user.
   - vLLM block ids do not address device memory (`M/generator_vllm.py:12-16, 645-661` [V]).
   - A vLLM hit is never trusted: a new request whose `start_pos` has no carried state is recomputed from 0 (`M/generator_vllm.py:831-846` [V]).
   - Cost: blocking, a snapshot cost the engine thread 84-88 ms (V1 runs, `L/serving_bisect/README.md` [M]). Queued, it costs 8-9 ms (76b3fafa239 [M]).
2. **Most of what a resume needs is a pure function of the prefix tokens, block-aligned at 128 tokens, and either already paged or cheap to page:**
   - the compressed latents of the 4 kv-source layers, which every prefill chunk already writes into the paged pool (`M/prefill_handoff.py:84-92, 204-208` [V]);
   - the index keys, which are not paged today;
   - the Engram history, which lives on the host and is derived from the tokens.
3. **The only true per-boundary state is the window halo of each layer.** That is 40 x 128 rows x 512 bf16 = 5 MiB per user (640 KiB per chip if column-sharded), and it is most of the 7.9 MiB snapshot at 512 tokens.
   - Compressor state needs nothing: no ratio-2 group is open at an even boundary.
   - Ring rows, `prev_cs` and drafter taps are rewritten by the replayed tail of at least 128 tokens.
4. **Hits are row-local.** Pages belong to one mesh row: there are 4 allocators, and user b lives on row b // U (`M/paged_ops.py:77, 82-84` [V]).
   - Block-table reuse (no copy) works only within a row.
   - A hit from another row needs a device-to-device copy per hit (fabric), or it is a miss.
   - Replicating every page on all rows does not fit in DRAM. A pool that is not row-private costs about 4-8 % of decode and is prohibitive in prefill.
5. **Precision couples to DRAM.** The shipped pool is fp8 (`DSV41_POOL_DTYPE=fp8` in the `*_apc.json` specs [V]) while prefill reads a bf16 kv table.
   - A bit-exact device-resident hit needs a bf16 pool: +1.25 GiB per chip.
   - That fits at 16 users / 262,144 tokens (3.5 GiB free [M]).
   - It does not fit at 4 users / 1M (0.84 GiB free [M]) until paged prefill drops the 2.5 GiB of kv tables.
6. **vLLM 0.31 constrains the design:**
   - it hashes blocks at schedule time, so same-step "phantom" hits remain;
   - it never tells the worker about evictions;
   - it has no channel for "the model computed more than scheduled".
   So the model must resume exactly at `num_computed_tokens`, and the hit must be rounded in `TTScheduler` against an index of boundaries the worker has reported.
7. **Upstream GPU vLLM ships DeepSeek-V4.1.** It caches the compressed latents and indexer keys as MLA blocks and does not cache the sliding-window KV: on a hit it replays the last 128 tokens with the window clamped, which is not bit-exact (`V/models/deepseek_v41/`, §4.4).
8. **Plan:**
   - **M0** (done): async snapshots, plus a boundary-policy tweak.
   - **M1**: a device-resident, model-owned page cache per row: refcounted pages, key pages, boundary slots, copy-free decode. Needs no host traffic and no new kernels.
   - **M1.5**: 128-token hit granularity.
   - **M2**: vLLM-owned blocks through the plugin's lane-DP, one KV manager per mesh row.
   - **M3**: paged prefill attention. Drops the kv tables; enables 1M.
   - **M4**: cross-row copy-on-hit.
   - **M5**: paged indexer (optional).
   - **M6**: reuse of generated tokens and images.
9. **Cost after M1 (estimates [I]):**
   - cold: in-trace writes of key pages and boundary slots, under 1 ms on a 428 ms chunk;
   - hit: a page-table upload plus device-local gathers, about 2-3 ms per 100k prefix tokens plus about 2 ms fixed. Today a restore takes 0.06 s at short prefixes and 7.6-8.7 s at 524,288 tokens [M].
10. **Effort:** M1 5-7, M1.5 1.5-3, M2 6-10, M3 4-6, M4 3-5, M5 3-5 engineer-weeks, each ±50 %.

## 1. State a prefill chunk carries and reads

### 1.1 Layer types (correction to the brief)

V4.1-Flash has no ratio-4 or ratio-128 layers.
- **Checkpoint** (`/mnt/tt-data/ssinghal/deepseek-v41-flash/inference/config.json` [V]):
  - `compress_ratios` = 0 for layers 0-1 and the MTP layers 40-42; 2 for layers 2-19; 1 for layers 20-39;
  - `kv_source_layers` = 2, 8, 14, 20;
  - `index_source_layers` = 2, 8, 14, 20, 24, 28, 32, 36;
  - `engram_layer_ids` = 1, 14;
  - `dspark_target_layer_ids` = 37, 38, 39.
- **Readers and key owners** (`M/kv_paged.py:10-15, 74-84` [V]):
  - Layers 3-7 read layer 2's latents, 9-13 read 8, 15-19 read 14, 21-39 read 20.
  - Layers 24/28/32/36 score against layer 20's keys (`M/dsv41_model.py:562-570` [V]).
- Upstream's V4.1 compressor accepts only ratios 1 and 2; "ratio-4/128" is V4.0 (`V/models/deepseek_v41/compressor.py:192-197` [R]).
- The "ratio-4" and "ratio-128" labels in `D/PREFILL_ATTN_NOTES.md:7` are stale names.

### 1.2 Inventory

Sizes are per user, per chip. Every pool and table is replicated over the 8 columns of a mesh row unless stated.

| # | State | Layers | Where it lives today | Format, size | Pure function of prefix? | Block-aligned at 128? | Role at a resume point s |
|---|---|---|---|---|---|---|---|
| 1a | Compressed latents (K = V, RoPE'd, 512 wide), paged | sources 2, 8, 14 (1 entry / 2 tokens), 20 (1 / token) | Paged pool, one per mesh row. A page is 128 tokens = 320 rows: src2 64, src8 64, src14 64, src20 128 (`M/paged_ops.py:15-16, 27-31`, `M/kv_paged.py:213-237` [V]). The sink writes it on every chunk; positions ≥ S are SKIPped (`M/prefill_handoff.py:84-92, 204-208` [V]) | fp8 (shipped) 1.25 KiB/token; bf16 2.5 KiB/token | yes | yes, one page | **per-token**: shareable by block table |
| 1b | Same latents, dense prefill copy | per kv-source group, shared by its readers | Per-slot kv table `PrefillKV` [Up,1,L+128+C,512]. A FIFO: each chunk shifts it by Cc rows and appends at the end (`M/prefill_sparse.py:72-93, 957-974` [V]) | bf16 2.5 KiB/token; 0.63 GiB at 262,144, 2.5 GiB at 1M (§2.5) | yes | yes; FIFO slot = entry + off | per-token: can be **rebuilt from pages** (copy) or removed (M3) |
| 2 | Window KV, prefill halo | all 40 layers | Persistent per-layer halo [Up,1,128,512] bf16 tile (`M/prefill_attention.py:382`), updated in-trace under the user mask (`:466-475` [V]) | 128 KiB/layer = **5 MiB** | yes (KV of positions s-128 .. s-1) | exactly one 128-token block per layer. Keeping it for *every* block costs 40 KiB/token, 16x the bf16 latents | **BOUNDARY STATE**: the only one |
| 3 | Window KV, decode ring | all 40 layers | Ring region in the pool tail, slot = pos % ring_rows. Rows: 128, or 288 in spec builds (`M/spec_policy.py:19`). The sink writes the last RING prompt positions (`M/prefill_handoff.py:82-83` [V]) | pool dtype; 288 rows: 5.6 MiB fp8, 11.25 MiB bf16 | yes | — | recomputed by the replayed tail (≥ 128 tokens) |
| 4 | Window scratch rows of the kv table [L, L+128+C) | per group | Rewritten by every layer for every chunk (`M/prefill_sparse.py:996-997` [V]) | 640 rows | — | — | transient, nothing to keep |
| 5a | Index keys, prefill FIFO (128 wide, RoPE'd, fp4-simulated) | owners 2, 8, 14, 20 | Per-slot FIFO, tile [Up,1,L,128] (`M/prefill_sparse.py:419-443`). Column-sharded from 131,072 entries: `keys_rm` + tile `keys`, slot p on column p % 8 (`:137-168, 450-550` [V]) | bf16 640 B/token; 2 copies when sharded, 1/8 per column | yes | yes: 320 entries per page | per-token, **not paged today** → needs key pages |
| 5b | Index keys, decode slab | owners; 24-36 alias 20 | Per-user dense [U,1,n_alloc,128] bfp8 (`M/dsv41_model.py:550-561` [V]). Exported from the FIFO when the prompt ends (`:1210-1235` [V]) | 340 B/token | yes | yes | rebuilt by the export at prompt end (unchanged) |
| 6 | Compressor partial group / `prev_cs` | ratio-2 owners 2, 8, 14 | Prefill pools pairs inside the chunk, nothing carried (`M/prefill_attention.py:247-260` [V]). Decode `prev_cs` [1,1,U,1024] fp32 = [kv \| score] of the last prompt token (`M/prefill_handoff.py:209-235` [V]) | 4 KiB/owner | yes | at an even boundary no group is open | none at s. `prev_cs` is recomputed by the tail (it always contains e-1) |
| 7 | Dense-path latent FIFO `lat_buf` | sources, only when the sparse path is off (L < 544) | `M/prefill_attention.py:386-391` [V] | — | — | — | not used at shipped sizes |
| 8 | Engram n-gram history | layers 1, 14 | Host `hasher.st.cache[b]`. Rows are gathered per chunk from host tables (`M/dsv41_model.py:1495-1501` [V]) | host | yes | yes | re-derived from the tokens on the host (`M/prefix_cache.py:384-387` [V]). Free |
| 9 | mHC streams | all | Per token. `M/mhc.py` only caches norm weights [V] | — | — | — | none |
| 10 | Drafter taps (stream means at the inputs of 37 / 38 / 39) | spec builds | Per-user stash, last 128 positions (`M/prefill_taps.py:4-14, 42-61` [V]) | 30 KiB/token = 3.75 MiB | yes | yes | recomputed by the tail. Boundary state only if the 128-token margin is dropped (§5.4) |
| 11 | Drafter ring rows | DSpark drafter | Seeded from the taps after prefill (`M/generator_vllm.py:1242-1272` [V]) | — | derived | — | derived |
| 12 | RoPE tables, masks, lim / off | per chunk | `DynCtx`. **Positions are global per replay** (replicated) (`M/prefill_dyn.py:104-121, 131-160` [V]) | — | positions only | — | none. But users at different offsets cannot share a replay (§6, M1.5) |
| 13 | Host metadata: page tables, `pf_resume`, slot maps, token book | — | host + one device page table (`M/paged_ops.py:121-132` [V]) | — | — | — | rebuilt |

### 1.3 What a resume at a boundary s needs

**What the snapshot holds.** It is exactly items 1b, 2 and 5a, plus the Engram history derived on the host (`M/prefix_cache.py:63-99, 266-308` [V]). Rings and `prev_cs` are left out because the lookup enforces s ≤ e - 128 (`M/prefix_cache.py:16-17, 473-477` [V]).

**Snapshot size at 512 tokens: 7.9 MiB** (40 layers, real server: `D/APC_CHUNKED_PREFILL_PLAN.md:67` [M]). My breakdown [I]:

| Part | Size |
|---|---|
| Halos, 40 x 128 KiB | 5.0 MiB |
| Latents, 512 x 2.5 KiB | 1.25 MiB |
| Keys, 512 x 640 B | 0.31 MiB (0.63 MiB when column-sharded, 2 copies) |
| **Total** | **6.6-6.9 MiB** |

That leaves about 1 MiB not attributed (Q6).

The synthetic composition of `tests/test_apc_snapshot_cost.py` (8 key FIFOs, an upper bound) gives 9.1 / 13.2 / 71 / 533 MiB at 512 / 1k / 8k / 64k tokens (`L/apcfast/ut_cost1.log` [M]).

**Conclusion:** at short prefixes the bulk is the halo. It is the one piece that is real boundary state, and its size is fixed per boundary. The per-token part duplicates what pages can hold.

## 2. How prefill reads KV today, how decode does, and what paged prefill needs

### 2.1 Prefill today (traced chunk, `DSV41_PREFILL_UP=1`)

**Shape.** One trace of C = 512 tokens per prefill slot is replayed per window. One replay is the window [s0, s0+C) of every participating slot, at global positions (`D/INTERLEAVE_NOTES.md:9-12`).

**Per compressed layer** (`M/prefill_sparse.py:942-1027` [V]):
1. The owner shifts the kv table by Cc rows, in 16,384-row pieces gated by the user mask, and appends the chunk's latents (`:957-974`).
2. The owner appends the chunk's keys to the key FIFO (`:992-995`).
3. Halo + chunk KV are written into the scratch rows (`:996-997`).
4. Index sources score the C queries against **all L FIFO slots** with `indexer_score_dsa`, mask them with a [C, L] visibility mask built from lim / off, and take the top-512 FIFO slots (`:552-641`).
5. `sparse_sdpa` runs per user (`cache_batch_idx = u`) on the kv table, with rows [128 window ids | 512 selected ids] (`:1005-1027`).
6. After the attention:
   - the sink writes latents into the pool pages, ring rows and `prev_cs` (`M/prefill_attention.py:466-468` → `M/prefill_handoff.py:183-238` [V]);
   - the halo is updated (`:469-475`).

**Every latent is written twice per chunk:** once into the FIFO (read by prefill) and once into a page (read by decode).

### 2.2 Decode today

- `paged_kv_step` does three things on the device (`M/paged_attention.py:66-105`, `M/paged_ops.py:202-240` [V]):
  - writes the ring row;
  - writes the latent through the page table;
  - builds the index rows [ring rows | page-translated top-512].
- `sparse_sdpa` reads the flat pool [1,1,R,512] (fp8 or bf16) with the attention sink.
- The indexer runs a matmul over the dense per-user key slab (`M/indexer.py:311-337, 371-405` [V]).

Decode is therefore already "paged attention". Prefill only *writes* the pages.

### 2.3 What a paged prefill attention needs

1. **`sparse_sdpa` over the pool with translated ids.**
   - Translation: physical row = pt[u][j // (128/r)] x 320 + SRC_OFF[src] + j % (128/r) (`M/kv_paged.py:231-236`, `M/paged_ops.py:28-31` [V]).
   - One translation is needed per (kv source, index source) pair per chunk, i.e. 8 per chunk (`M/kv_paged.py:86-92` [V]).
   - Implementation: a small kernel derived from the index build of `M/paged_kernels/paged_kv_step.cpp`, or about 10 composite ttnn ops (typecast, floor, embedding over the page-table row).
2. **Window scratch region inside the pool tensor.** 128 + C rows per prefill slot, appended after the ring region (as `M/paged_ops.py:58-60` does for rings), so that one `sparse_sdpa` call addresses both the window and the latents. The pool dtype question follows from this (§2.4).
3. **Write before read.** The owner's paged latent write moves from the sink, after the attention, to before the indexer and attention of the same chunk: queries see groups completed inside the chunk.
4. **No latent FIFO.** The kv table is deleted. Entry j sits at its page row, so a window can start at any 128-token multiple (with item 12 in §1.2 also made per row, see M1.5).
5. **Indexer unchanged in M3.** The prefill key FIFO stays.
   - `indexer_score_dsa` has no page table, only `cache_batch_idx` and `kv_len` (`ttnn/cpp/ttnn/operations/experimental/indexer_score/device/indexer_score_device_operation_types.hpp:103-116` [V]).
   - A paged indexer means a change to the reader kernel (M5).
6. **Hand-off.**
   - The latent hand-off disappears: the page is the only copy.
   - Ring rows, `prev_cs` and taps of the tail are still written by the sink. They are small.
   - The key export (FIFO → decode slab) stays until M5. It costs 0.01-0.12 s per finishing window (`D/INTERLEAVE_NOTES.md:63` [M]).
7. **Trace safety.** The page-table rows of the participants and the scratch bases become persistent tensors, refreshed by the pre-replay hook like the sink's (`M/prefill_handoff.py:163-180` [V]).

### 2.4 Precision coupling (the decisive constraint)

**Facts:**
- The shipped pool is fp8; the prefill kv table is bf16 (`M/prefill_sparse.py:82-93` [V]).
- An fp8 table for prefill (`DSV41_PFA_SP_FP8`) was rejected. Layer PCC fell from 0.9995 to 0.997 for a 2 % TTFT gain (`D/PREFILL_BACKLOG.md:12` [M]). That test made the **whole** table fp8, window scratch included.

**Options for paged prefill:**

| Option | What it is | Assessment |
|---|---|---|
| **P1** | bf16 pool | Bit-exact against today's prefill [I]: same bf16 values, same id order. Decode reads bf16, which is more precise than shipped, and gathers 2x the bytes. TPOT is not measured (Q1). |
| **P2** | fp8 latents + bf16 window | Needs `sparse_sdpa` to read two KV tensors, or (max, sum) stats for a softmax merge (`D/KV_SEQSHARD_DESIGN.md:76`). That is a kernel change, plus accuracy validation; not bit-exact. |
| **P3** | everything fp8 | Rejected (measured above). |

The same coupling applies to M1. Rebuilding the bf16 kv table from **fp8** pages is not bit-exact (Q2).

### 2.5 Cost: device time and DRAM

**Device time [I]:**
- **Removed:** the FIFO shift of the kv tables. Measured at 0.9 / 2.6 / 5.2 ms for 32 / 256 / 512 MiB (commit 84a77c9a24c [M]).
  - 262,144: 3 x 128 MiB + 256 MiB ≈ 7 ms per chunk (1.7 % of a 0.428 s replay).
  - 1M: 3 x 512 MiB + 1 GiB ≈ 26 ms per chunk (3.9 % of 0.661 s).
  - Replay times: `D/RELEASE_NOTES.md:25` [M].
  - The user-mask gates of these pieces come on top; they are not measured separately (Q11).
- **Added:** 8 id translations per chunk, about 0.3-2.5 ms (kernel vs composite ops).
- **`sparse_sdpa`:** gathers the same rows and bytes from a bf16 pool as from the bf16 table. It is DRAM-gather bound at ~280 GB/s (`D/PREFILL_BACKLOG.md:25`).
- **Net:** break-even to about -3 % chunk time.

**DRAM per chip.** Sizes are computed from the build formulas (`M/dsv41_model.py:110-128`, `M/paged_ops.py:56-60`, `M/prefill_sparse.py:77-93`). Free DRAM comes from MEMREPORT [M] (8 banks per chip).

| Item | 16 users / 262,144 (U=4) | 4 users / 1,048,576 (U=1) |
|---|---|---|
| Free at end of build [M] | 447.6 MiB/bank = **3.5 GiB** (`L/serving_bisect/runs/V5_h46/server.log`) | 107.7 MiB/bank = **0.84 GiB**, largest block 32 MiB/bank (`L/ctr2/s1_base_1m/server.log`) |
| Pool pages, fp8 (U x ceil((ctx+128)/128) pages x 320 rows x 512 B) | 8,196 pages: 1.25 GiB | 8,193 pages: 1.25 GiB |
| Pool pages, bf16 | 2.50 GiB (+1.25) | 2.50 GiB (+1.25) |
| Prefill kv tables (Up=1, bf16, (L+640) x 1 KiB per group) | 3 x 128.6 + 256.6 MiB = **0.63 GiB** | 3 x 512.6 + 1024.6 MiB = **2.50 GiB**, + 42 MiB (16,384 context) = 2.54 GiB |
| Prefill key FIFOs (sharded, 2 copies) | ~40 MiB | ~160 MiB |
| Decode key slabs (bfp8) | 356 MiB | 340 MiB |
| Free after M3, P1 (bf16 pool, no tables) | 3.5 + 0.63 - 1.25 = **2.9 GiB** | 0.84 + 2.54 - 1.25 = **2.1 GiB** |
| Free after M3, P2 (fp8 pool, no tables) | 4.1 GiB | 3.4 GiB |

**The 1M build has nothing to spare today.**
- The 524,288-token snapshot sliced one 512 MiB kv-table buffer while the largest free block was about 41 MiB. That was the out-of-memory error (`D/RELEASE_NOTES.md:19` [M]).
- 3.0 MiB per bank of trace region is left at the end of the build (`:25` [M]).

Paged prefill is the main source of DRAM for any device-resident APC at 1M.

## 3. Row privacy: four ways to share blocks across users

**Facts:**
- The pool tensor has the same shape on all 32 chips.
- Each mesh row has its own `PageAllocator` (free-list stack, no refcounts, no sharing) (`M/paged_ops.py:77`, `M/kv_paged.py:239-304` [V]).
- The page table is sharded over rows (`M/paged_ops.py:121-132` [V]).
- Rings are per (layer, in-row user).
- The **model** chooses a new request's row with `claim_balanced` / `claim_packed`; the plugin does not see this (`M/generator_vllm.py:823-853`, `M/vllm_state.py:129-172` [V]).
- Decode attention on row r reads only row r's DRAM.

| Option | Hit cost | DRAM | Decode / spec | 16 users (U=4/row) | 4 users (U=1/row) | Verdict |
|---|---|---|---|---|---|---|
| **(a)** Replicate every page on all 4 rows (all-gather each new latent over the rows) | 0 | Pool x4: fp8 1.25 → 5 GiB at 262,144 (> 3.5 GiB free). Or the same DRAM at a quarter of the capacity: 1M tokens shared by 16 users, i.e. one 262,144 prompt per 4 users | +1 small all-gather per source layer per step [I ~0.1-0.2 ms]. Spec rollback must touch 4 copies | breaks the 262,144-per-user spec | impossible (1 user at 1M fills a row) | **not feasible** as the default |
| (a') Pin a few hot headers (system prompts) on all rows | 0 | x4 per pinned prefix only | none | useful | useful | optional policy on top of (b)/(c) |
| **(b)** Row-affine placement + block-table sharing within a row | page-table update only | none extra: cached prefixes live in free pages | none: decode writes only private tail pages; a hit never covers the last 128 positions | **bursts lose nothing**: the first wave of every row runs in the *same* replay, so computing a header on 4 rows costs the same replays as on 1 (today's 8 → 5 → 4 replays, `D/APC_CHUNKED_PREFILL_PLAN.md:68`) | 1 slot per row: a hit needs its home row to be free, else wait or miss | **recommended base** (M1/M2) |
| **(c)** Copy-on-hit across rows | proportional to the prefix, per **hit** not per prompt: latents 1.25 / 2.5 KiB per token + keys 80 B per token (sharded) + boundary 640 KiB, per chip. At 100k ≈ 125-250 MiB per chip | staging in pieces | none | rare | the common case at U=1 | **recommended fallback** (M4) |
| **(d)** Pool not row-private (pages striped over rows, remote rows gathered) | 0 | /4 per chip | +62 µs per layer plain decode, +234 µs per layer per spec block ≈ +4 % / +8 % (`D/KV_SEQSHARD_DESIGN.md:59-64` [M]) | — | — | Prefill would gather C x 640 rows x 1 KiB = 320 MiB per layer per chunk (`D/KV_SEQSHARD_DESIGN.md:66-68` [I]). **Not feasible for prefill**; only a side effect if long-context row-sharding ever lands |

**Fabric for (c) [I].** Two routes:
- an all-gather of a staging buffer over mesh axis 0 (an existing collective, moving 4x the bytes);
- `ttnn.experimental.send_async` / `recv_async` over a `MeshSocket`, point to point. These exist under `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/` [V] but are untested here.

The row-axis bandwidth is not measured (Q4). With a guessed 10-20 GB/s per chip:

| Prefix | Copy time | Cold prefill [M] (`D/APC_CHUNKED_PREFILL_PLAN.md:114`) |
|---|---|---|
| 100k tokens | ~10-25 ms | 107-124 s |
| 1M tokens | ~0.1-0.25 s | 504 s (at 300k on the 1M build) |

**vLLM side of (b).** vLLM has one BlockPool per KV cache manager and no per-device-group pools (`P/model_runner.py:757-768` [R]). Standard DP runs one process per rank, which does not fit a single 4x8 mesh, and the platform refuses MoE models under standard DP (`P/platform.py:2322-2330` [R]).

The plugin's **lane-DP** matches the mesh rows if per_lane = U:
- `TTLaneCoordinator` runs one `TTScheduler` + KV cache manager per lane, with lane-local block ids;
- all lanes share one prefill-or-decode mode per step;
- rows are lane x per_lane + s (`P/lane_scheduler.py:1-27, 242-322`, `P/input_batch.py:705-727` [R]).

Gaps [R]:
- lane mode + spec decode is refused (`P/platform.py:2281-2289`);
- the model must accept `tt_data_parallel=4`; today it raises unless it is 1 (`M/generator_vllm.py:185-186` [V]);
- the reset flag is lost in the lane merge (`P/lane_scheduler.py:217-238`);
- the connector is hard-coded to None;
- `_pick_lane` is not prefix-affine (`P/lane_scheduler.py:344-370`). `Request.block_hashes` already exists at intake (`V/v1/request.py:225-230`), so the hits of each lane's manager can be queried there.

## 4. vLLM 0.31 APC and what the plugin must do

### 4.1 Hashes, full blocks, the last-token rule [R]

**Block hashes:**
- A block hash is sha256 over (parent hash, token ids, extra keys) (`V/v1/core/kv_cache_utils.py:684-714`).
- Extra keys are LoRA, multimodal (identifier, offset), and the cache salt (on block 0 only) (`:614-652`).
- Only full blocks are hashed (`:836-895`).

**The last-token rule:**
- The hit is capped at `num_tokens - 1` and then rounded down to the block (`V/v1/core/kv_cache_manager.py:289-299`; `single_type_kv_cache_manager.py:799-805, 839`). So 1 to 128 tokens are always recomputed.
- Our rule is stricter: at least 128 tokens are replayed.
  - The plugin rounds `min(hit, N - 128)` to the model's grid (`P/config.py:281-291`, `P/scheduler.py:764-783` [R]).
  - The model's store applies the same margin (`M/prefix_cache.py:473-477` [V]).

### 4.2 Same-step phantom hits [R]

**Still present in 0.31:**
- `allocate_slots` caches blocks at schedule time for tokens that are only scheduled (`kv_cache_manager.py:603-612`).
- A later waiting request in the same `schedule()` hits them; nothing checks whether a block has been computed (`scheduler.py:872-937`).
- `delay_cache_blocks` is set only for async connector loads (`:1216`).
- Only `MambaManager` defers a same-step hit: it returns `num_gpu_blocks + 1` (`single_type_kv_cache_manager.py:1726-1734`).

**Why GPU vLLM is safe:** every layer writes the KV of all tokens in the batch before attention reads it (`V/model_executor/layers/attention/attention.py:540-578`). Upstream V4.1 also writes its compressed and indexer caches before reading them (`V/models/deepseek_v41/compressor.py:246-322`).

**TT [I].**
- Today a hit is never trusted, so phantom hits are harmless. A host snapshot only exists after the producer's window, so it can never serve a same-step hit.
- In a paged design a hit is safe only if both of these hold:
  1. The producer's window covering the hit ran earlier in the same call. The window loop is ordered by position (`M/dsv41_model.py:1464` [V]), but groups run by **descending** start (`:1444-1446` [V]). That order is required for evictable continuations, but it would run a new request with a hit before its donor. It must change for those rows.
  2. The boundary state exists.
- **Robust rule:** a boundary is hittable only once the worker has *reported* it, i.e. after the producer's step.
  - TT runs with async scheduling off (`supports_async_decode: False`, `M/generator_vllm.py:162` [V]; the platform then disables async scheduling, `P/platform.py:2219-2236` [R]), so step N's report is processed before step N+1 is scheduled.
  - Phantom hits then become impossible by construction. Any extra pages vLLM hit beyond the last reported boundary are simply rounded off.
- Upstream also has an inferred hazard under the PRIORITY policy: a preempted same-step request leaves hashed-but-unwritten blocks. Keep FCFS.

### 4.3 Eviction and how the model can learn about it [R]

- Eviction is lazy: a block is evicted when `get_new_blocks` reuses it, which drops its hashes (`V/v1/core/block_pool.py:684-692, 732-753`).
- Freed hashed blocks go to the back of the LRU; a request's tail blocks are evicted first (`:777-808`; `single_type_kv_cache_manager.py:581-589`).
- **There is no worker callback.** KV events (BlockStored / BlockRemoved) exist only inside the engine core, and only when they are enabled (`block_pool.py:190-191`; `scheduler.py:2311-2326`).

Two ways for the model to learn about evictions [I]:
- **Implicitly.** Block b's old content is dead when b appears in `NewRequestData.block_ids` or `CachedRequestData.new_block_ids` beyond `num_computed_tokens` (`V/v1/core/sched/output.py:58-83, 137-152`). This is the invalidation rule for anything attached to a block, such as a boundary slot.
- **In-process.** The executor is "uni", in the engine-core process (`P/platform.py:591, 2106-2122` [V]), so `TTScheduler` could read `take_events()` directly.

### 4.4 Hybrid groups, recurrent state, upstream DeepSeek-V4.1, bounded replay [R]

**Spec classes** (`V/v1/kv_cache_interface.py`):
- `MLAAttentionSpec(tokens_per_state)`;
- `SlidingWindowMLASpec(bounded_replay)`, where `prefix_cacheable = not bounded_replay` and `prefix_replay_tokens = window` (`:930-955` [V]);
- `CircularBufferSpec`, never cached (`:889-926`);
- `MambaSpec` with modes align / all / none (`:1041-1119`).

**Upstream DeepSeek-V4.1 on GPU** (`V/models/deepseek_v41/`):

| Cache | Spec | Block size | Prefix-cached |
|---|---|---|---|
| Compressed KV, kv sources only | `MLAAttentionSpec(tokens_per_state = ratio)` (`attention.py:1100-1125` [V]) | cache block | yes |
| Indexer K | `MLAAttentionSpec` (`:1164-1195` [V]) | cache block | yes |
| SWA KV, every layer | `SlidingWindowMLASpec(window 128)` (`V/v1/attention/backends/mla/sparse_swa.py:115-139` [V]) | 32 | only with bounded replay off |
| Compressor ring, ratio 2 | `CircularBufferSpec` (`compressor.py:132-163` [V]) | — | no |

**Bounded replay** is the default (`swa_bounded_replay=True`, `V/config/cache.py:242-245`, V2 runner):
- the scheduler rewinds every hit by 128 tokens and sends `replay_start` (`V/v1/core/sched/scheduler.py:279-290, 3038-3050` [V]);
- hits ≤ 128 are dropped (`:1016-1026`);
- the cached compressed and indexer pages are not rewritten (`PAD_SLOT_ID`, `V/models/deepseek_v41/nvidia/model_state.py:52-81, 212-226`);
- the replayed tokens' window is clamped at `replay_start` (`sparse_swa.py:366-371`). Their rebuilt SWA KV therefore differs from a cold prefill above layer 0. **Not bit-exact.**

**With bounded replay off** (V1 runner, ROCm, prefill context parallelism > 1: `attention.py:519-546`), the SWA KV is prefix-cached:
- a hit needs the cdiv(127, 32) = 4 SWA blocks before the boundary (`single_type_kv_cache_manager.py:951-1069`);
- with the default `prefix_cache_retention_interval = 0` (`V/config/cache.py:160-166`), SWA and Mamba blocks are hashed only at "semantic" boundaries: the prompt end and the shared-prefix junction (`single_type_kv_cache_manager.py:1071-1151`).

**Mapping onto TT [I].** Our boundary slot is exactly upstream's non-bounded SWA block at a semantic boundary: one 128-token block, since window = block = 128. But vLLM's hybrid allocator cannot carry it on TT:
- **Shared pool, unified page size.** All groups share one BlockPool with a unified page size: packed groups stride a block by the sum of every group's page (`kv_cache_utils.py:2001-2135`), and unification scales or pads pages (`:1383-1445`).
  - A 5 MiB SWA page next to a ~400 KiB latent + key page means every block id reserves the SWA bytes.
  - The Mamba-style alignment instead grows the attention block until its page covers the state page (`V/platforms/interface.py:835-1004`): about 1.6k tokens per block, which breaks 128-token granularity.
- **Plugin limits.** The plugin accepts only attention groups and one buffer per layer (`P/model_runner.py:645-663, 753-850`).
- **Mamba align mode** needs block = `cache_config.block_size` (`scheduler.py:440`) and saves one state per chunk end.

**Conclusion:**
- Declare only the per-token group (latents + keys) to vLLM.
- Keep the boundary slots model-owned, behind a scheduler-visible index.
- Or adopt bounded replay: no boundary state at all, but approximate (Q7).

### 4.5 KV connector as a vehicle [R]

**The API** (`V/distributed/kv_transfer/kv_connector/v1/base.py`):
- Scheduler side:
  - `get_num_new_matched_tokens` (`:486`) returns the tokens available beyond the local hit;
  - `update_state_after_alloc` (`:521`) receives the new blocks;
  - `build_connector_meta` (`:547`) and `request_finished` (`:579`).
- Worker side: `start_load_kv` / `wait_for_save` (`:313-406`).
- External tokens are added into `num_computed_tokens` (`scheduler.py:940-1241`). Sync loads are hashed at once and then serve ordinary local hits.

**Constraints:**
- With hybrid groups the connector must implement `SupportsHMA` (`factory.py:54-60`).
- A multi-group model cannot report `invalid_block_ids` (`scheduler.py:3288-3297`).
- The TT worker and runner have no connector wiring, and lane mode hard-codes `connector = None`.
- The per-layer hooks never fire on TT. The TT worker would call `start_load_kv` once per step.

**Fit [I].** It is the right shape for (c), the cross-row copy, and for a host second tier (today's snapshots as an offload store). Same-row hits do not need it.

### 4.6 Keeping vLLM's count equal to what the model really skips

**Facts [R]:**
- `num_computed_tokens` is advanced at schedule time.
- `ModelRunnerOutput` has no field for "computed a different number" (`V/v1/outputs.py:261-305`).
- `PrefixCacheStats` records the hit after the TT rounding (`scheduler.py:1252-1256` with `P/scheduler.py:764-783`).
- The plugin's hold-back sizing ignores hits (`P/scheduler.py:858, 894`).
- The plugin docs have no APC section (`D/SCHEDULING.md` of the plugin).

**Requirements [I]:**
1. The model starts exactly at `num_computed_tokens`: no private recompute and no private deeper restore. Today it may recompute more, which caused the 31 s stall class of bug (`D/APC_CHUNKED_PREFILL_PLAN.md:89`).
2. All rounding happens in `TTScheduler._get_local_prefix_cache_hit`:
   - the grid (512 until M1.5, then 128);
   - the margin N - 128;
   - the deepest boundary on the request's row with a *reported* slot.
3. No `replay_start` unless the model implements it. The stats would count the replayed tokens.
4. After M2, the hold-back and budget logic should use the hit.

### 4.7 A defect found while reading (reproduced on CPU after the revision, see R8)

**Where.** `PrefillTracker.find` matches only by token prefix and carries no request identity (`M/vllm_state.py:237-246` [V]).

**Trigger.** A *new* request whose grid-rounded hit equals the end of another prompt that is still in `inprog`. That covers two cases:
- a prompt still being prefilled;
- a prompt whose prefill is finished but which has not decoded yet. `inprog` drops a user only at decode (`M/generator_vllm.py:1023-1024` [V]), and `pf_resume` is set when e % 512 == 0 (`M/dsv41_model.py:1519-1522` [V]).

**What happens.**
1. The new request is bound to that user: `SlotTable.bind` swaps the two logical slots (`M/generator_vllm.py:833-840`, `M/vllm_state.py:119-127` [V]).
2. The new request continues on the other prompt's state and overwrites its decode state.
3. The other prompt's next decode then raises "decode of logical slot … that was never prefilled" (`M/generator_vllm.py:1044-1047` [V]).

`D/APC_CHUNKED_PREFILL_PLAN.md:90` lists only a milder form of this.

**Fix.** The plugin passes a stable request key, or a new-vs-continuation flag, with each prefill row. See Q8.

## 5. Zero cost: definition and how to get there

### 5.1 Definition

- **Cold prompt.** No per-prompt snapshot and no host traffic. The only extra work is the in-trace writes of key pages and boundary slots. Target: ≤ 0.5 % of a chunk replay.
- **Hit.** A block-table (page-table) pointer update, plus a restore of the boundary state, plus (until M3 and M5) device-local gathers into the slot's dense tables. Target: ≤ 5 ms + O(prefix / 100 GB/s), and no host transfer.
- **Eviction.** No host LRU. Pages are evicted by the page allocator (M1) or by vLLM (M2).
- **Today, for comparison:**
  - cold: 8-9 ms of engine thread per snapshot queued (84-88 ms blocking) plus host memory. That gave +14 % TTFT / +21 % TPOT at 16 x 1024 (V1 vs V0, `L/serving_bisect/README.md`) and +17 % TTFT at 16 x 8192 with APC as the only change (V4 vs V0) [M];
  - hit: a restore of 0.06 s at short prefixes (4 users, 40 layers) and 7.6-8.7 s at 524,288 tokens [M].

### 5.2 Boundary state: bytes per slot [I]

| Content | Replicated over 8 columns | Column-sharded (rows p % 8) |
|---|---|---|
| Halos, 40 layers x 128 rows x 512 bf16 (required) | 5 MiB | **640 KiB**. Restore = one all-gather over the columns (5 MiB per row) + a masked write |
| + drafter taps (only if the margin is dropped) | +3.75 MiB | +480 KiB |
| Compressor partial group | 0: no open group at an even boundary | 0 |

**Why the halo is the bulk, and why it cannot simply be paged per block:**
- The halo is one block per layer, i.e. 40 KiB per token per chip if it were kept for every block.
- That is 16x the bf16 latents (2.5 KiB per token) and 32x the fp8 latents.

**It can live in the paged pool as a block attachment:**
- a slot belongs to the page that *ends* at the boundary;
- it is freed when that page is evicted.
- Budget example: 1 GiB per chip = 1,638 sharded slots per row. At one slot per 512 tokens that covers 838k tokens of boundaries.

**Retention.** Write a slot at every window end (cheap, in-trace). Retain with LRU, keeping each prompt's last aligned boundary longest. This is vLLM's "semantic boundary" retention.

### 5.3 Cost per step of the path

Device-local copies are estimated at the ~200 GB/s of the FIFO-shift microbenchmark [I].

| | Today (M0, host, async) | M1 (device store, same row) | M3 (+ paged prefill) | M5 (+ paged indexer) |
|---|---|---|---|---|
| Cold, per boundary | 8-9 ms engine thread + 7.9 MiB host (+2.5-3.75 KiB per token) | in-trace: 4 key-page scatters + boundary write, ≤ 1 ms per chunk (Q5) | same | same |
| Hit, 100k-token prefix | ~0.1 s (4 users at 1-1.5k) to 8.7 s (512k); 39-46 s hit TTFT vs 107-124 s cold at 100k | page table + kv-table gather 244 MiB + keys 15 MiB + halos 5 MiB ≈ 2.6 ms + ~2-5 ms op overhead | page table + keys + halos ≈ 0.2 ms + ops | page table + halos |
| Hit granularity | 512 x powers of two | every 512 (M1), every 128 (M1.5) | same | same |
| Eviction | host LRU, 6 GiB | per-row page LRU (model). M2: vLLM | same | same |
| Device DRAM | 0 | key pages 80 MiB (sharded) + slots 640 KiB each (+1.25 GiB for a bf16 pool, 262,144 only) | frees the kv tables (0.63 GiB at 262,144, 2.54 GiB at 1M). Against today's fp8 pool, P1 is net -0.62 GiB at 262,144 and +1.29 GiB at 1M | frees the key FIFOs (160 MiB at 1M) |

### 5.4 The 128-token margin costs nothing once hits are at 128-token granularity [I]

- **On the 512 grid**, s = floor512(e - 128), so e - s falls in [128, 640). The tail needs 2 windows when e mod 512 is in [1, 128): about 25 % of uniformly spread prompt ends.
- **On a 128 grid** (M1.5), e - s falls in [128, 256): always 1 window. That is the same as vLLM's own rule (replay at least 1 token).

So keep the margin. The boundary state then needs neither taps, nor ring rows, nor `prev_cs`.

## 6. Staged plan (device-resident; superseded by S.4. M1 demoted, M2 / M4 dropped, M3 + the index-key pages and paged indexer of M5 = required PF0-PF4)

**Global rule.** Each milestone ships behind a flag and is measured on the `serving_bisect` harness:
- 16 x 1024 / 16 x 8192 / 1 x 8192 rows, against the V0 / V3 / V6 rows of `L/serving_bisect/README.md`;
- the APC guide benchmark (16 x GSM8K 8-shot sharing a ~640-token header);
- the shared-prefix needle sweep, 2k-100k (`L/apc/apc_prefixsweep.py`);
- GSM8K, 256 questions, sampled.

Greedy text identity is **not** a valid APC check under k=5 (near-tie differences). Use bit-exact model-level tests plus GSM8K accuracy.

### M0: current APC, async snapshot (done) + boundary policy (0.5-1 engineer-week)

- **Done:** 76b3fafa239 (branch `ssinghal/dsv4p1-apc-fast`).
  - Engine time per snapshot is 8-9 ms instead of 84-88.
  - 16 x 1024: 3592 ms TTFT / 166.4 ms TPOT (.46), against V0's 3588 / 164.6 [M].
- **To add:**
  - **Policy:**
    - "all boundaries ≥ K", or "powers of two from K" (`DSV41_APC_MIN`);
    - "last aligned boundary of each prompt" (multi-turn, repeated documents).
    - Today the default skips 2560 for a 3000-token header: the hit falls back to 2048, so 3 windows instead of 1 [I].
  - **The `PrefillTracker` identity fix** (§4.7).
- **Tests:** `tests/test_prefix_store.py` (policy), `tests/test_vllm_interface.py` (identity), server guide benchmark.
- **Decision:** ship on the 16-user / 262,144 spec as the stop-gap.

### M1: device-resident, model-owned page cache per mesh row (5-7 engineer-weeks)

**Scope:**
1. **Refcounted allocator.** `PageAllocator` becomes a refcounted per-row page cache (`M/kv_paged.py:239-304`):
   - pages carry (chain hash, refcount);
   - an LRU of cached free pages;
   - allocation takes never-cached free pages first, then evicts the LRU.
   - The chain hash is a 128-token sha256 chain plus the model / config signature (as `M/prefix_cache.py:405-414`, at 128 granularity). Image blocks are excluded until M6.
2. **Key pages.** A new row-major key pool: 4 keys x 128 wide per 512-wide row, because `paged_scatter_rows` takes bf16 [N, 512] rows (`M/paged_ops.py:324-330`, `M/prefill_taps.py:26` [V]).
   - Column-sharded consistently with the FIFO (column = (entry + L) mod 8).
   - Written in-trace by the 4 owners every chunk, with persistent id tensors refreshed by the sink's pre-replay hook.
3. **Boundary slots.** A per-row pool of column-sharded halo slots.
   - Written in-trace at the end of each window whose end the policy wants (SKIP ids otherwise).
   - Attached to the page that ends at the boundary.
4. **Hit path.** Replaces `_apc_restore_wave` (`M/dsv41_model.py:1285-1303`). For each new request:
   - take the deepest s on the 512 grid such that s ≤ e - 128, pages [0, s) are cached on the row, and a slot exists at s;
   - **adopt** those pages (refcount +1, page-table upload; no latent write; replaces the copy of `M/prefix_cache.py:375-380`);
   - fill the slot's carried state on the device:
     - kv-table tail ← `ttnn.embedding` gather of the pool rows + `paged_scatter_rows` with SKIP on other rows;
     - key-FIFO tail ← key pages;
     - halos ← all-gather of the slot + the masked `where`/`copy` (`M/prefix_cache.py:189-226`).
   - Engram history comes from the tokens; `pf_resume[b] = s`.
5. **Affine placement.** A new `SlotTable.claim_affine` picks, among the free slots, the row with the deepest hit; ties use the `claim_balanced` load rule.
6. **Scheduler sync.** A model capability `prefix_cache_hit_fn`, called in-process by `TTScheduler._get_local_prefix_cache_hit` (uni executor), so that vLLM's `num_computed_tokens` equals the model's resume point. Fallback: the current grid rounding with policy `all`.
7. **Same-step ordering.** A boundary becomes hittable only after the window that wrote it. Rows of the same call that have a hit run after their donors; alternatively, defer them one step, Mamba-style.
8. **Pool dtype.** bf16 at 262,144 (decision D1 after Q1). With an fp8 pool the latent restore is not bit-exact (Q2).
9. **Host store.** Kept as an optional second tier, or deleted.

**Kernels:** none required. M1 uses `paged_scatter_rows` (`M/paged_ops.py:324`, `M/paged_kernels/row_scatter.cpp`), `ttnn.embedding`, all-gather, `where`/`copy` and `slice_write`. Optional: a batched multi-tensor halo restore.

**Tests:**
- CPU: allocator and hash-chain unit tests.
- Device, hit vs cold, **bit-exact** in first-token logits and the full decode state: 4 / 21 / 40 layers, 2,600 and 16,000 tokens (`PIDX_SHARD=1`). Extend `tests/test_apc_m2_snapshot_restore.py` / `test_apc_m3_integration.py`.
- Eviction under a full row.
- Donor finished vs donor still in progress.
- Decode after a hit: rings, `prev_cs`, decode key slab.
- Servers as in the global rule.

**Metrics:**
- `apc_snapshot_s` goes to 0.
- Hit restore ≤ 5 ms + page table.
- Cold chunk-replay delta ≤ 0.5 %.
- 16 x 1024 within the noise of APC off (V3 / V6).
- Burst TTFT gain at least today's 2.9x (6.2 → 2.1 s, `D/APC_CHUNKED_PREFILL_PLAN.md:107-108`).

**Risks:**
- The in-trace writes enlarge the trace. Trace region left at the end of the build: 39 MiB/bank on the 16-user / 262,144 build, 3-9 MiB/bank on the 1M build (MEMREPORT; `D/RELEASE_NOTES.md:25`) [M].
- bf16-pool TPOT.
- Gathers between replays (Q9).

**Stop / decide:**
- If the cold overhead exceeds 1 % or a hit costs more than 20 ms: stop and redesign the writes.
- If the bf16 pool costs more than 3 % TPOT: take the fp8-restore variant (needs Q2 passed) or wait for M3.

### M1.5: 128-token hit granularity (1.5-3 engineer-weeks)

- Make the per-chunk position tensors of `DynCtx` row-sharded: RoPE tables, masks, lim / off, `sp_win` / `sp_perm`, visibility (`M/prefill_dyn.py:84-160`, `M/prefill_sparse.py:840-941`). Then the 4 rows can be at different s0 in one replay.
- Hits move to a 128 grid. With the 512 grid, any start that is not a multiple of C breaks the shared window loop (`M/dsv41_model.py:1464-1465`).
- **Kernels:** none.
- **Tests:** bit-exact against the C-grid run; mixed offsets in one replay.
- **Gain:** deeper hits, and the margin no longer costs an extra window (§5.4).

### M2: vLLM-owned blocks, eviction by vLLM (6-10 engineer-weeks, high uncertainty)

**Plugin:**
- Lane-DP with 4 lanes = 4 mesh rows, per_lane = U, 8,196 (+1 null) blocks per lane (§3).
- Lift the lane + spec refusal; per-lane cap and reset; prefix-affine `_pick_lane`.
- A TT `ModelRunnerOutput` field reporting boundary slots (created / freed per block hash), consumed in `TTScheduler.update_from_output` (`P/scheduler.py:1184`). Hit rounding uses those reported boundaries.

**Model:**
- vLLM block id → row-local page (id + 1 because block 0 is vLLM's null block, `V/v1/core/block_pool.py:840-841` [R]).
- Page tables come from the `page_table` argument instead of the model allocator. Today the model raises on per-layer tables and ignores the argument (`M/generator_vllm.py:682-683` [V]).
- Slot attachments are invalidated when a block id is re-allocated (§4.3).

**Kernels:** none.

**Tests:**
- Plugin mock-scheduler tests: lane affinity, rounding to reported boundaries, same-step deferral, eviction → attachment invalidation, spec lookahead (`P/scheduler.py:108-125`).
- Model: block-table-driven prefill and decode bit-exact against the M1 allocator.
- Servers.

**Decision:** only if "eviction by vLLM" and vLLM-native cache metrics are required. M1 already gives the zero-cost behaviour.

### M3: paged prefill attention (4-6 engineer-weeks)

**Scope:** §2.3 items 1-4 and 7, with P1 (bf16 pool). This enables device-resident APC at 1M:
- net free DRAM 0.84 → 2.1 GiB per chip;
- removes the 512 MiB kv-table slices.

**Kernels:**
- New: an index-translation op (`M/paged_kernels/`) or composite ops.
- Unchanged for P1: `sparse_sdpa` (`ttnn/cpp/ttnn/operations/transformer/sdpa/`).
- For P2: a two-tensor `sparse_sdpa` reader, or `return_stats` with a merge (`D/KV_SEQSHARD_DESIGN.md:76`).

**Tests:**
- Bit-exact against the dense-table prefill (P1): 4 / 40 layers, 2.6k / 16k / 128k tokens (`tests/test_prefill_sparse_device.py` pattern, DYN=1).
- 1M probe: MEMREPORT and the 1,000,028-token prompt.
- Needle at 100k; GSM8K.
- Chunk time at 262,144 / 1M; expect the FIFO shift to drop out (Q11).

**Decision:** after Q1 and Q11; mandatory only if 1M APC is required (D3).

### M4: cross-row copy-on-hit (3-5 engineer-weeks)

- **Scheduler:** place by affinity. If the deepest hit is on a busy or other row, place the request elsewhere and mark an "external hit" (KV-connector-shaped: matched tokens on the scheduler side, a copy list in the metadata).
- **Worker, before the prefill:**
  1. gather pages, key pages and the slot on the source row;
  2. transfer them over axis 0 in staging pieces (all-gather, or a `send_async`/`recv_async` socket);
  3. scatter them into the destination's new pages;
  4. register the hashes there.
- **Kernels:** none if all-gather is used. A `MeshSocket` pair otherwise.
- **Tests:** bit-exact against a local hit; bandwidth (Q4); 4 users / 1M with follow-up questions on a long document while its home row is busy.
- **Decision:** after Q3. If fewer than 10 % of hits are cross-row at the 16-user spec, ship only the 1M part.

### M5 (optional): paged indexer (3-5 engineer-weeks)

- **Kernel work:** page-table translation of the key tile rows in `indexer_score_dsa`. Touches `ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/reader_indexer_score.cpp`, the device operation types and program factory (hash), and nanobind, for both the replicated and the column-sharded variants.
- **Result:**
  - prefill reads the key pages directly;
  - the key FIFO, its shift and the key gather on a hit are removed;
  - the decode slab is exported from the pages.
- **Stays dense:** the decode indexer. The matmul backend needs a dense slab; the fused backend has an unexplained accuracy loss at U=4 and is slower (75.3 vs 43.8 ms/token at 1M, `D/RELEASE_NOTES.md:54` [M]).

### M6: reuse of generated tokens and images (2-4 engineer-weeks)

- **Generated tokens:**
  - decode also writes key pages (one more paged write per owner per step in `M/indexer.py:311-337`);
  - at request finish, the decode ring (288 rows contain the halo of floor128(end)) is copied into a boundary slot;
  - vLLM already hashes decoded blocks.
- **Images:** enable image blocks. vLLM's extra keys carry the multimodal identifier and offset (`V/v1/core/kv_cache_utils.py:499-564` [R]). Remove the vision guard (`M/prefix_cache.py:509-512`).
- **Validation:** multi-turn accuracy. This is not bit-exact against cold by nature: KV computed by decode differs from KV computed by prefill (the same holds on GPU). Image hit tests.

### Not feasible, and why

- **(a) Replicate every page on all rows:** pool x4 (§3).
- **(d) Remote reads in prefill:** 320 MiB per layer per chunk.
- **Exact halo rebuild by replay:** the receptive field is about 40 layers x 128 = 5,120 tokens, so a hit would cost about 10 extra windows.
- **vLLM hybrid SWA / Mamba groups for the boundary state on TT** (§4.4):
  - page unification over a shared BlockPool;
  - the plugin accepts only attention groups;
  - Mamba align mode needs block = cache block and keeps one state per chunk end.
- **Zero-copy decode index keys without a paged decode indexer** (M5 note).
- **Bit-exact hits with an fp8 pool against a bf16 prefill** (§2.4).
- **Hits closer than 128 tokens to the prompt end:** they need taps in the slot (3.75 MiB) and gain nothing (§5.4).
- **Zero-cost host snapshots at 1M:** they are GBs, and reading them alone exceeds the zero-cost budget.
- **Bounded replay (upstream) as a bit-exact path:** it is approximate by design. Usable only if Q7 shows the accuracy holds and D4 accepts it.

## 7. Open questions, each with an experiment (≤ 1 device-hour, on an allowed host: .45 / .46 / .47)

| # | Question | Experiment | Owner / decision |
|---|---|---|---|
| Q1 | Decode TPOT and DRAM with a bf16 pool vs fp8 | `tests/test_paged_decode_steps.py`, 40 layers, `DSV41_SYN_CTX=65536`. B=16 with `DSV41_POOL_DTYPE` bf16 vs fp8, plus one spec verify round. Record ms/token and MEMREPORT (method of `D/PAGED_NOTES.md:22-30`). ~45 min | D1: bf16 pool at 262,144 |
| Q2 | Accuracy of a hit restored from **fp8** pages into the bf16 kv table | Model-level, 40 layers, a 16k needle prompt. Restore the prefix (s = 8192) from fp8 pool pages vs from the bf16 snapshot. Compare first-token logits (max abs diff, PCC), 64 greedy tokens, the needle answer. ~45 min | Decides the fp8-restore variant |
| Q3 | **ANSWERED (R7).** Share of hits that would be cross-row | Host-only simulation: replay the client traces of `L/apc` and `L/serving_bisect` through `claim_balanced` vs `claim_affine` on 4 rows x U slots. 0 device-hours | M4 priority |
| Q4 | Row-axis copy bandwidth | Microbenchmark: all-gather over mesh axis 0 of 16 / 64 / 256 MiB per chip (row-major [N,512] bf16) vs `send_async`/`recv_async` between (0,c) and (3,c); GB/s per chip (pattern `tests/test_rowgather_microbench.py`). ~30 min | M4 design |
| Q5 | Cost of the in-trace key-page + boundary writes | Add 4 key-page scatters + 40 halo scatters (SKIP ids) to the chunk trace (4 and 40 layers, C = 512, S_pad 262,144); measure the replay delta and trace-region use. ~45 min | M1 go / no-go |
| Q6 | The unattributed ~1 MiB of the 7.9 MiB snapshot at 512 tokens | Print the per-tensor `nbytes` of one snapshot in the real 40-layer build (env-gated print in `snapshot_user`). ~20 min, can run alongside Q2 | slot sizing |
| Q7 | Accuracy of upstream-style bounded replay (no boundary state) | Model-level: a hit at s with the halo zeroed and the window clamped, replaying [s-128, e), vs an exact restore. Logits PCC, greedy, needle at 16k-64k. ~1 h | D4: is approximate acceptable? |
| Q8 | **ANSWERED (R8): reproduced.** `PrefillTracker` defect (§4.7) | CPU adapter test with the fake model (`tests/test_vllm_interface.py` style): A = 1024-token prompt finished but not yet decoded; B = A + 200 tokens arriving at start 1024. Assert A's slot map and live set are intact. 0 device-hours; a 1-hour server repro only if the CPU test fails | fix in M0 |
| Q9 | Are eager gathers and scatters between two replays of one wave safe (mid-call copy-on-hit)? | 4-layer interleave test: insert a restore (masked writes + `paged_scatter_rows`, pre-allocated buffers only) between two replays; check bit-exactness and that nothing hangs. ~30 min | M1 item 7 |
| Q10 | Lane-DP for DSV4.1 vs a row-partitioned BlockPool | CPU mock of `TTLaneCoordinator`: 4 lanes x U=4, spec k=5 gate lifted; check scheduling, lookahead and resets. 0 device-hours | plugin owner, M2 design |
| Q11 | FIFO shift + gate cost inside the trace (size of M3's speed-up) | `tests/test_prefill_prof.py` with `PROF_NCH` (d0af4327870), one layer per group at 262,144 and 1M; per-op time of the shift pieces and the gate. ~45 min | M3 priority |
| Q12 | Is hit-vs-cold bit-exactness a requirement, or is GSM8K / needle equivalence enough? | none (decision) | user. Bit-exact means a bf16 pool and no bounded replay |
| D2 | Is vLLM-owned eviction (M2) required, or is model-owned (M1) acceptable? | none | user / plugin owner |
| D3 | Is device-resident APC needed at 1M (M3), or is the host APC enough there? | none | user |

## Appendix A: figures used

| Figure | Value | Source |
|---|---|---|
| Snapshot engine time, blocking / queued | 84-88 ms / 8-9 ms | `L/serving_bisect/README.md`, commit 76b3fafa239 [M] |
| Snapshots per 16 x 1024 run | 128 snapshots, 11.3 s (`apc_snapshot_s` 182 ms of a 1,724 ms prefill call) | `L/serving_bisect/README.md` [M] |
| 16 x 1024, APC on vs V0 | TTFT +13.6 %, TPOT +20.6 % | `L/serving_bisect/README.md` [M] |
| 16 x 8192, APC only | TTFT +17.3 % | `L/serving_bisect/README.md`, V4 vs V0 [M] |
| Snapshot size at 512 tokens | 7.9 MiB | `D/APC_CHUNKED_PREFILL_PLAN.md:67` [M] |
| Synthetic snapshot sizes, 512 / 1k / 8k / 64k | 9.1 / 13.2 / 71 / 533 MiB | `L/apcfast/ut_cost1.log` [M] |
| Restore time | 0.06 s for 4 users (40 layers, ~1.5k); 7.6-8.7 s at 524,288 | `D/APC_CHUNKED_PREFILL_PLAN.md:67`, `D/RELEASE_NOTES.md:19` [M] |
| Chunk replay | 0.428 s at 262,144; 0.661 s at 1M (4 users) | `D/RELEASE_NOTES.md:25` [M] |
| Hit TTFT | 39-46 s hit vs 107-124 s cold at 100k; 69 s vs 504 s at 300k on 1M | `D/APC_CHUNKED_PREFILL_PLAN.md:114` [M] |
| FIFO shift | 0.9 / 2.6 / 5.2 ms at 32 / 256 / 512 MiB | commit 84a77c9a24c [M] |
| Free DRAM, end of build | 447.6 MiB/bank (16 users / 262,144); 107.7 MiB/bank, largest 32 MiB/bank (4 users / 1M) | MEMREPORT [M] |
| Page / row geometry | 320 rows per 128-token page; fp8 row 512 B, bf16 1 KiB; ring 288 rows in spec builds | `M/paged_ops.py:27-33, 53`, `M/spec_policy.py:19` [V] |
| Row-gather decode overhead | 62 / 234 µs per layer (plain / spec block) | `D/KV_SEQSHARD_DESIGN.md:59-64` [M] |

---

# Part 3: paged-prefill implementation log (branch `ssinghal/dsv4p1-paged-prefill`, off `ssinghal/dsv4p1-final` 5cc1dcf209a)

Evidence tags as above. **[T]** = measured by a device test in this part.

## 3.1 (a) Inventory: every place prefill reads or writes dense per-user state, and the paged equivalent

All references are to `tt/` on this branch (traced-chunk path, the production path; the eager `ingest` / `attend` path of `prefill_sparse.py` is only used by small unit tests).

| # | State | Written at | Read at | Paged equivalent | Status |
|---|---|---|---|---|---|
| 1 | Latent kv table `PrefillKV` `[Up,1,L+128+C,512]` bf16 per kv-source group (4 groups) | `attend_dyn`: FIFO shift in `GATE_ROWS` pieces gated by `umask` + append of the chunk's latents; window scratch rows `[L, L+128+C)` | `sparse_sdpa(kt, idx, cache_batch_idx=u)` in `attend_dyn`; APC snapshot / restore (`prefix_cache.py`, `CarriedSpec L*.kvt`) | pool pages `[1,1,R,512]` through the page table, 320 rows per 128-token page, entry j of a source at `pt[u][j*r/128]*320 + SRC_OFF + (j*r%128)/r`; window rows in a scratch region of the pool (`Up*(128+C)` rows after the rings) | PF1 (write, before attention) and PF2 (read, no table): done, 4k bit-exact [T] |
| 2 | Selected ids: FIFO slots from the indexer | `select_dyn` / `select_dyn_sharded` | used directly as kv-table rows | slot -> entry (`slot - off`, `off = L - (s0+C)/r`) -> pool row: new dataflow kernel `paged_kernels/prefill_id_translate.cpp`, one translation per (kv source, index source) pair (8 per chunk), done in the index-source layer and shared by its readers | PF2 done [T] |
| 3 | Window ids `ctx.sp_win[r]` | `_upd_hook` per replay | `attend_dyn` | `PagedCtx.win` `[Up,1,C,128]` (scratch base per slot added) | done |
| 4 | Index-key FIFO `ix.keys` (tile `[Up,1,L,128]`, or sharded `keys_rm` + `keys`) | `add_keys_dyn(_sharded)`: shift + append, gated by `umask` | `select_dyn(_sharded)` (`indexer_score_dsa`), `keys_full` | paged key caches + paged `indexer_score_dsa` reader (PF3) | not done (needs the C++ reader) |
| 5 | Key export to the decode slab | `_export_keys_users` / `export_keys` at prompt end | decode indexer `k_cache` | in-trace write of the chunk's keys into the slab (D5) | not done |
| 6 | Halo `[Up,1,128,512]` x 40 layers | `_finish_dyn` (gated) | `attend_dyn`, window layers' SDPA | unchanged (PF5 would move it to ring rows) | unchanged |
| 7 | Dense latent FIFO `lat_buf` | `forward_dyn` when the sparse path is off (`L < 544`) | dense SDPA | not allocated when the sparse path is on; below 544 entries the dense path stays (the sink still writes the latents after the attention: `_lat_written` flag) | unchanged |
| 8 | Sink: ring rows, latents, `prev_cs`, taps | `PagedStateSink.write` after the attention | decode | latents moved before the attention by `write_lat` (PF1); ring rows, `prev_cs`, taps unchanged | PF1 done [T] |
| 9 | APC snapshots of the carried state | `prefix_cache.py` (kvt tail, lat_buf, halos, key FIFOs) | restore into any user | snapshot = pages + key pages + halos (PF4) | incompatible with `read` today: APC reads `sp.kvt.t` (None in read mode) |

## 3.2 What was implemented (PF1 + PF2)

- `DSV41_PF_PAGED=0|write|read` (default `0`: the release code path), module `tt/prefill_paged.py`; `prefill_paged.set_mode` switches at run time, the next chunk-trace capture follows it.
- `PagedKVPool(scratch_rows=...)` (`tt/paged_ops.py`): the window scratch region (`Up*(128+DSV41_PF_CMAX)` rows, 2 MiB at Up=1, C max 2048) after the rings; allocated whenever the mode at model build is not `0`.
- `write`: `PagedStateSink.write_lat` scatters the chunk's latents (`paged_scatter_rows`) at the start of `attend_dyn`, before the indexer and the attention; `write` skips them (flag `_lat_written`). The dense tables stay and are read.
- `read`: no `PrefillKV` (a `PagedView` carries the dtype); `PagedCtx` holds persistent per-replay tensors (page-table rows of the users in the prefill slots, `off` per ratio, window ids), refreshed by the DynCtx update hook; the window rows `concat([halo, kv])` are written into the pool scratch; `sparse_sdpa` runs on the pool tensor without `cache_batch_idx`. Needs a bf16 pool (P1). Entries whose page is not allocated (padded tail beyond the prompt) translate to the sentinel.
- Test: `tests/test_paged_prefill_device.py` (dense, write, read on one model object, re-capturing the chunk trace between modes).

## 3.3 Results

### 4k, 25 layers (0-24), B=16, Up=1, chunk 512, .46, bf16 pool, `DSV41_MO_OVERLAP=0` [T]

Users 0/5/10/15 with 4096 / 3899 / 3069 / 3584 tokens (8 chunks of 512).

| | dense | write (PF1) | read (PF2) |
|---|---|---|---|
| first tokens | 121254 / 68709 / 68038 / 105393 | same | same |
| last-token logits vs dense | | bit-equal | bit-equal |
| whole pool (pages + rings) vs dense | | bit-identical | bit-identical |
| replay of 8 chunks | 1.640 s | 1.640 s | 1.641 s |
| DRAM allocated per bank | 1975.1 MiB | 1975.1 MiB | 1974.2 MiB |

`DSV41_MO_OVERLAP` is automatically ON at small `max_ctx`; its sub-device load is illegal inside a trace capture (`TT_FATAL Cannot reset worker state during trace capture`, `moe_overlap.py:194`). This is independent of paged prefill (dense mode fails identically) and is off at the production sizes.

### 32k, all 40 layers, B=16, Up=1, chunk 512, .46, bf16 pool, `DSV41_MO_OVERLAP=0` [T]

Users 0/5/10/15 with 32768 / 32571 / 31741 / 32256 tokens (64 chunks).

| | dense | write (PF1) | read (PF2) |
|---|---|---|---|
| first tokens | 223 / 223 / 7835 / 4097 | same | same |
| last-token logits vs dense | | bit-equal (max abs 0) | bit-equal (max abs 0) |
| whole pool (pages + rings) vs dense | | bit-identical | bit-identical |
| replay of 64 chunks | 21.261 s (332.2 ms) | 21.258 s (332.2 ms) | 21.122 s (330.0 ms, -0.7 %) |
| DRAM allocated per bank | 3102.4 MiB | 3102.4 MiB | 3093.2 MiB (-9.2 MiB: the 32k tables, 3 x 16384 + 32768 rows x 1 KiB / 8 banks ~ 10 MiB) |

Log: `/mnt/tt-data/ssinghal/dsv4-logs/paged_prefill/t32k_L40.log`.

### 262,144, all 40 layers, B=16, Up=1, chunk 512, .46, bf16 pool of 2300 pages per mesh row (smaller than production: the bf16 pool next to the dense tables must fit), `DSV41_MO_OVERLAP=0` (its production default at this size) [T]

Users 0/5/10/15 with 262144 / 261947 / 261117 / 261632 tokens (512 chunks per user), column-sharded prefill indexer (`DSV41_PIDX_SHARD=auto`, active at this size).

| | dense | write (PF1) | read (PF2) |
|---|---|---|---|
| first tokens | 20538 / 201 / 115 / 201 | same | same |
| last-token logits vs dense | | bit-equal (max abs 0) | bit-equal (max abs 0) |
| whole pool (pages + rings) vs dense | | bit-identical | bit-identical |
| replay of 512 chunks | 219.698 s (429.1 ms) | 219.841 s (429.4 ms) | 207.939 s (**406.1 ms, -5.4 %**) |
| DRAM allocated per bank | 3300.0 MiB | 3300.0 MiB | 3220.7 MiB (**-79.3 MiB per bank = 0.62 GiB per chip: the kv tables of the note's section P.5, 0.63 GiB**) |
| free / largest block per bank | 583.7 / 515.4 MiB | 583.7 / 519.9 | 662.9 / 535.9 |

The speed-up is larger than the note's estimate (P.6: -1.5 % at 262,144, from the FIFO shift alone): the removed latent FIFO shifts + user-mask gates + the table copy of the window rows are worth 23 ms per chunk, 5.4 %. Log: `/mnt/tt-data/ssinghal/dsv4-logs/paged_prefill/t256k_L40.log`.

### 1M build (max_ctx 1,000,064, S_pad 1,000,448), all 40 layers, B=4 (1 user per mesh row, Up=1), chunk 512, trace region 650 MB, .46, `DSV41_MO_OVERLAP=0` [T]

Users 0 / 1 with 1,000,000 / 999,803 tokens (1954 chunks). A bf16 pool next to the dense tables does not fit at 1M (note P.5), so the comparison is across two boots: dense with the production **fp8** pool (the dense prefill reads the bf16 tables, its logits do not depend on the pool dtype), read with a **bf16** pool; logits saved by the first, compared by the second. No pool comparison and no `write` mode at this size.

| | dense (fp8 pool, production) | read (PF2, bf16 pool) |
|---|---|---|
| first tokens | 26704 / 1831 | same |
| last-token logits | reference | **bit-equal (max abs 0)** |
| replay of 1954 chunks | 1463.955 s (**749.2 ms/chunk**) | 1060.576 s (**542.8 ms/chunk, -27.5 %**) |
| DRAM allocated per bank (after capture + run) | 3659.9 MiB | 3508.3 MiB (-151.6 MiB) |
| free / largest block per bank | 337.0 / 235.1 MiB | 488.6 / 311.2 MiB |

The -151.6 MiB per bank is the net of the removed tables (2.50 + 0.04 GiB = ~325 MiB per bank) and the bf16 pool of the two admitted prompts (the pool here holds 1 user per row: 7,813 pages bf16 vs fp8 = +~125 MiB per bank over the fp8 pool); the production pool (4 users per row sharing the row's pages) is analysed in P.5. The -27.5 % chunk time is far above the note's estimate (-4 %, P.6, shift only): the in-place FIFO shift in 16,384-row pieces, the `umask` gates and the table copies are about 207 ms per chunk at 1M. Logs: `t1m_dense.log`, `t1m_read.log`, `t1m_dense_logits.pt`.

### Prefill slots

All tables above used `DSV41_PREFILL_UP=1` (production). A `DSV41_PREFILL_UP=4` 262,144 B=16 run hung (pytest alive 2 h 50 min, no progress after the JIT of the first chunk; killed, `.46` reset); it was not chased (UP=4 also hung the production server under load, and UP=1 with `DSV41_KEY_EXPORT_FILL=1` is the correct production config). The paged code indexes by prefill slot (scratch region, page-table rows, window ids, the translation kernel's per-slot user), nothing assumes Up = 1.

### 262,144 B=16 rerun on the corrected production config (`DSV41_PREFILL_UP=1`, `DSV41_KEY_EXPORT_FILL=1`, branch with b16fix 121e75b70a9 cherry-picked as 5d39e9ec3dd) [T]

Same shape as the first 262,144 table (bf16 pool of 2300 pages per mesh row, 40 layers, chunk 512). Dense == write == read: first tokens 20538 / 201 / 115 / 201, last-token logits bit-equal (max abs 0), pool (pages + rings) bit-identical. Replay: dense 429.1 ms/chunk, write 429.3, read 406.2 (-5.3 %). DRAM allocated per bank: 3277.6 / 3277.6 / 3198.3 MiB (read -79.3 MiB). Log `t256k_fill1.log`. The earlier table (without the key-export fix) is identical in logits and pool: the export only touches the decode key slab, which this test does not compare, and the prefill logits do not read it.

## 3.4 Not done and why

- **PF3 (paged prefill indexer)** needs a change to the C++ reader of `indexer_score_dsa` (`ttnn/cpp/ttnn/operations/experimental/indexer_score/`) and a rebuild of `_ttnn.so`. Every worktree (this one, the server's `wt/vllm_test`, the other agents' runs) shares the single build of `/mnt/tt-data/ssinghal/tests/tt-metal/build`: rebuilding it while the isolation and eval agents run on it is not safe, and the server worktree must be moved to the same commit afterwards (memory note host-server-uses-vllm-test-worktree). It needs its own build tree or an agreed window; estimate in P.2 unchanged (3-5 weeks).
- **PF4** (remove tables / key FIFOs / hand-off / export, port APC to pages) depends on PF3 for the key FIFOs. The latent-table part is already optional by flag (`read`): what remains for it is the APC port (`prefix_cache.py` reads `sp.kvt.t`, absent in `read`) and deleting the dual path.
- Pool dtype (PF0): all measurements use a bf16 pool (P1). At production sizes the pool is fp8; with `read` the window rows and latents of the prefill would be read as fp8 (P2a: 0.9995 -> 0.997 per-layer PCC measured earlier). The Q2a / Q2b / Q16 experiments were not run.
- Decode after a paged prefill and spec decode were not exercised in this test (prefill logits and the pool were compared).
- Design-note assumptions checked: PF1 / PF2 "no kernel change for P1" is wrong in one detail: an id-translation kernel IS needed (the indexer returns FIFO slots, not entries; the selected ids need `slot - off` and a page-table lookup per user), written as `paged_kernels/prefill_id_translate.cpp` (a dataflow kernel, JIT-compiled, no C++ rebuild). The P.6 time estimate (-1.5 to -4 %) is too low: measured -5.3 % at 262,144 and -27.5 % at 1M.
