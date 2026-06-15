# Work log — `sparse_sdpa` implementation

Goal: **Stage 1 (H=32, full `{32,32}` tiles) working end-to-end, PCC ≥ 0.99 vs `sparse_mla` golden.**
Branch: `pjosipovic/sparse_mla_prefill_ref`. Plan: `PLAN_sparse_sdpa.md`. Golden: `models/demos/deepseek_v32/reference_cpu/sparse_sdpa_prefill.py::sparse_mla`.

## Strategy decision (v0)
Implement the **`compute_common` materialized path first** (not the streaming `kt_inplace_v` path) to reach a passing result fastest, then optimize:
- tilize K → `k_tiles[Skt,DHt]`; **transpose** → `kt_tiles[DHt,Skt]` (Kᵀ, for QK); **copy compact V** → `v_tiles[Skt,vDHt]` (rope cols dropped, for PV).
- QK = `matmul_blocks(q_tiles, kt_tiles, transpose=false)` → `qk` (kt is already Kᵀ, standard NN).
- mask add = plain `add_block_inplace(qk, mask)` — **valid here** because `qk` is a normal reserved/pushed CB (the hold-wr-ptr problem is specific to the streaming `cb_qkt_im`; we're not on that path).
- softmax = `reduce_c<MAX,REDUCE_ROW>` → `sub_exp_block_bcast_cols_inplace` → `matmul_reduce(col_identity,sum)` → `recip_block_inplace` → `mul_block_bcast_cols`.
- PV = `matmul_blocks(probs, v_tiles, transpose=false)` → `out`.
- untilize `out` → row-major; writer writes H rows.

Rationale: separate Kᵀ/V buffers ⇒ standard `matmul_blocks` (pops its in1 fine), no custom MOPs, no hold-wr-ptr/mask-stamp subtleties. The streaming overlap path (`PLAN §6.3`) is the perf follow-up. (Heavier L1: `k_tiles` + `kt_tiles` + `v_tiles` — may force small `k_chunk_size`.)

## Phases (per PLAN §8)
1. Scaffold + identity (op callable, compiles, registers)
2. Reader gather (debug dump K)
3. Tilize/untilize round-trip
4. QK + masked softmax (intermediate check)
5. PV + normalize → full op, PCC
6. Sweep k_chunk + edge cases

## Log
- (start) Plan committed `788593b564f`. Reference impl committed `d48f5da128e`.
- **Phase 1 DONE ✅** — scaffold builds + registers + runs e2e on BH, `test_sparse_sdpa_phase1_runs` PASS.
  - Files: `device/sparse_sdpa_device_operation_types.hpp`, `.../sparse_sdpa_device_operation.{hpp,cpp}`, `.../sparse_sdpa_program_factory.cpp`, `sparse_sdpa.{hpp,cpp}`, kernels `dataflow/sparse_sdpa_{reader,writer}.cpp` + `compute/sparse_sdpa_compute.cpp`; wired `sources.cmake`, `CMakeLists.txt` (FILE_SET api), `sdpa_nanobind.cpp`.
  - Validated: ProgramDescriptor new-infra (all-static, `prim::sparse_sdpa`→`launch<>`); validate (BH gate, H==32, ROW_MAJOR/DRAM, padded==logical, TOPK≤2048, TOPK%k_chunk==0, fp32_dest_acc_en==false via `get_compute_kernel_config_args`); ROW_MAJOR output spec; per-core token split; RM stick writer (`TensorAccessor(args,addr).get_noc_addr(h*S+t)`, page=row).
  - Gotchas hit: BH config accessor is `get_compute_kernel_config_args` (no `BlackholeComputeKernelConfig`); BH-gate decorator is `models.common.utility_functions.run_for_blackhole` (NOT `models.utility_functions`). Two unused-var warnings in program_factory (`q_row_bytes`/`idx_row_bytes`) — consumed in phase 2.
- **Phases 2–5 (single-chunk) DONE ✅** — reader gather+mask, compute tilize/Kᵀ-reposition/QK/masked-softmax/PV/normalize/untilize, writer-from-compute all wired; `test_sparse_sdpa_pcc_single_chunk` **9/9 PASS** (PCC ≥ 0.99) vs `sparse_mla`, across (S,T,TOPK)∈{(64,256,32),(64,512,64),(32,1024,128)} × {all_valid,few_valid,boundary}.
  - Compile fix: compute kernel needs `compute_common.hpp` defines — added `STATS/SUB_EXP/MUL_BCAST/DHT/REDUCE_GRANULARITY=1` + `EXP_APPROX_MODE` to `compute_desc.defines` (granularity=1 is always-correct: it divides any dim).
  - **Root-cause bug (Skt≥2 only): `pack_tile` out-of-order index ignored without `<true>`.** The Kᵀ grid-reposition packed with bare `pack_tile(0, CB_KT, d*Skt+sk)`; the explicit `output_tile_index` is honored ONLY with `pack_tile<true>(...)` (template `out_of_order_output`, default false → sequential pack). Bare pack wrote sequentially in `sk`-outer/`d`-inner order = natural K `[Skt,DHt]`, but the matmul reads in1 as `[DHt,Skt]` (`in1_index += N`). Coincides at Skt=1 (KT==K), scrambles at Skt≥2. Symptom: Skt=1 perfect, Skt≥2 PCC 0.39–0.85, output norm correct (denominator fine) but values scrambled, and even tile-0-only corrupted. Fix: `pack_tile<true>` for both CB_KT and CB_V packs. (CB_V happened to be correct anyway — its intended index `sk*vDHt+d` matched the sequential order.) Diagnosis via numeric probes (`/tmp/dbg*.py`): norm match + tile-isolation + the matmul `in1_index += N` convention. **Lesson: any out-of-order `pack_tile(dst, cb, idx)` MUST be `pack_tile<true>`.**
  - Known latent (not yet hit): CB_QK_IM/CB_MAX/CB_SUM/CB_OUT_IM are never popped, so >1 token/core would overflow — fine for S≤num_cores (≈1 tok/core) but must be addressed for production S. And the chunk loop does NOT flash-accumulate yet (overwrites max/sum/out per chunk) → multi-chunk (`test_sparse_sdpa_pcc`) still TODO.
- **Multi-chunk flash + multi-token DONE ✅ — Stage 1 COMPLETE.** Full suite **22/22 PASS** (phase1×1, single_chunk×9, multi-chunk×9, multitoken×3), all PCC ≥ 0.99 vs `sparse_mla`.
  - **Flash (online softmax)** rewrote the compute chunk loop into the canonical running-(max,sum,output) form. Per chunk: QK→mask→`reduce_c<MAX>`(do_eltwise_max=c>0, running max)→`sub_exp` in-place→`reduce_c<SUM>`(chunk sum)→PV; then for c>0 combine via `correction = sub_exp_block(prev_max,cur_max)` (=exp((prev−cur)·scale)), `prev_sum*=corr` (`mul_tiles_bcast_cols_inplace`)+`cur_sum+=prev_sum`, and `cur_out += prev_out*corr` via `mul_block_bcast_cols<...,pack_accumulate=true>` (L1 acc). Ping-pong (prev/cur) CB handles as runtime uint32 + `std::swap`; max needs 2 buffers (reduce_c reads prev_cb≠out_cb), sum/out also ping-pong. SUM done with explicit `reduce_c<SUM>` (not SDPA's sub_exp-partial + `matmul_reduce`+col_identity) → no col_identity CB needed. n_chunks==1 degenerates to plain single-chunk (same end state), so it subsumes the earlier path.
  - **untilize takes a compile-time CB id** but the running out lives in a swapped (runtime) handle → `copy_block(out_prev, CB_OUT_IM)` into a fixed CB, then `untilize<vDHt,CB_OUT_IM,CB_OUT_RM>`.
  - **Per-token CB hygiene** (matters for >1 token/core, i.e. S>num_cores; num_cores=110 here): everything must be empty at token end. `matmul_blocks` only pops in1, so `CB_QK_IM` popped after PV and **`CB_Q_IN` popped after the chunk loop** (Q is reused across chunks). The final running max is never consumed by a combine → explicit `cb_pop_front(max_prev)` at token end. `CB_SCALE` is intentionally persistent (reduce scaler, never popped). Verified by `test_sparse_sdpa_multitoken` (S=256>110 ⇒ 2–3 tok/core) at 1/2/4 chunks.
  - CB set (20): +ping-pong `CB_{MAX,SUM,OUT}_{A,B}`, `CB_CORR`; `CB_OUT_{A,B}` single-buffered (=vDHt tiles) for L1 acc. Writer `CB_OUT_RM` 13→18, reader `CB_IDX` 14→19.
- **Stage 1 acceptance met.** Remaining (future): Stage 2 = H=16 / `{16,32}` tiles (LLK uplift per PLAN); perf path (fuse sub_exp partial-sum, streaming overlap); larger TOPK sweep up to 2048.

## Perf (Stage 1 baseline + opts)
Method: `python -m tracy -p -r -v -m pytest <file>::test_sparse_sdpa_perf -k <id>`. Read `DEVICE KERNEL DURATION [ns]` (SparseSDPAOperation) from `generated/profiler/reports/*/ops_perf_results_*.csv`. Per-zone: `DeviceZoneScopedN` in reader (between reserve_back/push_back → productive time only) + compute; parse `generated/profiler/.logs/profile_log_device.csv` (cols: zone=10, phase=11 ZONE_START/END, cycles=5; 1350 MHz). 125-scope/core limit → profile a 1-token/core shape (S=110).
- **Production shape** Q[32,640,576], KV[56320,576], idx[640,2048], TOPK=2048: **baseline 11.12 ms → 8.45 ms after mask opt (−24%)**. Independent of T (cache size) and k_chunk; scales with S·TOPK. S=640 on 110 cores = ceil 6 tokens/core (90 cores×6 + 20×5 → busiest bounds it).
- **Bottleneck = reader (NCRISC), ~100% busy** (coupled RISCs all show equal KERNEL duration; the per-zone split is what localizes it). NoC check: reader already uses the **preferred NOC_0 for DRAM reads** (`ReaderConfigDescriptor{}`→`ReaderDataMovementConfig`, program.cpp:341); don't split DRAM reads to NOC_1 (non-preferred for reads).
- **Per-token zones (1 tok/core, TOPK=2048, 8 chunks), µs/chunk:**
  - rdr_Kchunk (indexed K gather) 141 — **now 94% of reader, the dominant cost**.
  - rdr_mask 62 → **2.8 after opt** (build 1 row + `add_tiles_bcast_rows` row-broadcast instead of replicating 32 rows; reader writes 1 row, compute bcasts mask row 0 across query rows). cmp_tilizeMask stall 51 → 0.3.
  - real compute (repose 8, QK-math 54, softmax 3, PV 3) ≈ 67 µs/chunk ≪ reader 145 → reader-bound by ~2×.
- **K-gather optimizations (DONE).** Two landed, both exploiting the contiguous-sentinel-tail contract:
  1. **Chunk-skipping.** Reader binary-searches `nv` (first sentinel = valid-key count), processes only `n_active = ceil(nv/k_chunk)` chunks; all-sentinel chunks are skipped entirely (no DRAM read, no fill, no compute). TRISC compute can't issue NoC reads, so it can't derive `nv` itself — the reader passes `n_active` via a tiny `CB_CTRL` (compute reads it with `get_tile_address(CB_CTRL,0)` + L1 deref, NOT `get_read_ptr` which is dataflow-only) and both loop the same count. Within a chunk, reads only the `valid` prefix.
  2. **Prebuilt-zero NoC fill.** Sentinel suffix of the boundary chunk filled by NoC-copying a once-built zero K-row (`CB_ZERO`, local L1->L1 via `get_noc_addr(my_x[noc_index], my_y[noc_index], zero_l1)`) — independent copies, all in flight under the existing gather barrier — instead of a per-row scalar store loop. Offloads the fill from the RISC to the NoC.
- **Perf after K-gather opts** (prod S=640/TOPK=2048, by sparsity `nv`):

  | nv | baseline→mask→chunk-skip→+NoC-fill |
  |---|---|
  | mixed `1+(s*7)%K` (~½ sentinel) | 11.1 → 8.45 → 2.83 → **2.13 ms** (5.2× total) |
  | causal `min(s+1,K)` (realistic prefill, the user's shape) | — → — → 1.64 → **0.81 ms** |
  | dense `nv=K` (no sentinels, worst case) | **4.41 ms** (only mask-opt applies; ~340 GB/s DRAM, near the data-movement floor) |

  Realistic prefill (S<TOPK ⇒ nv=position+1) lands at **0.81 ms**. Dense is the floor (pure K-gather BW). Sentinel handling is now ~free.
- **Mask-skip (done, wall-clock-neutral).** Mask is built/tilized/added only on the last active chunk (`c==n_active-1`); earlier active chunks are fully valid (all-zero mask = no-op). Correct (22/22) but ~0 wall-clock change — the row-broadcast opt already made the mask cheap (~2.8 µs/chunk vs K-gather ~140), and the bigger savings (compute mask tilize/add) aren't on the critical path. Kept anyway: removes redundant work, no downside, helps if compute ever bottlenecks.
- **Where time goes now:** reader is essentially pure K-gather (scattered DRAM reads of real keys; interleaved tensors → every read bank-scattered). Dense (4.40 ms) is DRAM-BW + cross-core-contention bound.

## Dense-case investigation (what does NOT help, and what does)
- **Diagnosis tools:** per-zone `DeviceZoneScopedN` (reader rdr_Q/rdr_Kchunk; compute cmp_*) + per-occurrence parse of `profile_log_device.csv`. Key: profile a 1-token/core shape (S=110) to fit the 125-scope/core limit; an 8-core shape (S=8) isolates contention from per-core latency.
- **Contention is the story.** Per-chunk K read: **8 cores → 8 µs flat; 110 cores → chunk0 228 µs decaying to 10** as cores desync. So at scale all 110 cores hit DRAM in lockstep and saturate/collide.
- **Tried & REJECTED (no wall-clock change — proven, not guessed):**
  - *Double-buffer CB_K_RM:* can't — L1 ~full at kc=256 (K_RM 295KB + K_IN+KT+V ~850KB). *Double-buffer CB_Q_RM:* no-op (reader BW-saturated, no idle slack to prefetch into).
  - *Swap Q/K read order (K first):* no-op. rdr_Q dropped 94→18.8 µs on warm DRAM but chunk0 rose 132→228 — the first-access/contention cost is FIXED and just moves to whichever read goes first. Conserved.
- **Trid ring — WORKS (the win).** Tag each K read with a NoC transaction id cycling `N_TRIDS` slots; before reusing a slot, `noc_async_read_barrier_with_trid` its previous read. Bounds outstanding reads/core to `N_TRIDS` → caps aggregate NoC/DRAM congestion. (Free fns `noc_async_read_set_trid`/`noc_async_read_barrier_with_trid`; plain `noc_async_read` preserves the set trid since `ncrisc_noc_fast_read` never writes `NOC_PACKET_TAG`. Pattern from conv3d `reader_vol2col`.)
  - **Workload-dependent**: helps congested (dense), hurts uncongested (sparse — ring overhead with no congestion to recover). Resolved with a per-token gate: ring only when `n_active >= n_chunks/2` (dense-ish; the kernel knows n_active, and uniform-dense ⇒ synced ⇒ congested).
  - **Sweep (prod, ms):** depth: 1=5.38(worse), 2=4.14, **4=4.01**, 6/8≈4.03 on dense. depth-4 always-on hurt causal/mixed +8%; depth-8 is gentler (less throttle) → near-neutral on sparse. **Winner = depth 8 + gate `n_active>=n_chunks/2`:** dense 4.42→**4.03 (−9%)**, causal 0.80→0.81, mixed 2.15→2.16 (both neutral). Now the **default** (env `SPARSE_SDPA_K_TRIDS`/`SPARSE_SDPA_RING_MIN_ACTIVE` override for tuning).
- **Remaining dense lever:** fewer K bytes (row-major fp8 `Lf8`/`Fp8_e4m3` — page-gatherable, tilize-converts to bf16; NOT BFP8 which is tiled). ttnn has no fp8 tensor dtype → uint8-backed + caller quantizes. ~2× but an accuracy/API decision; not pursued. Stateful NoC read APIs don't help: they fix the bank coordinate (same-bank sequential), but our reads scatter across all banks (interleaved + random keys).

## Object-based kernel APIs (modernization, all 22/22 PCC, perf neutral)
Rewrote all three kernels to the object NoC + CB APIs (pattern from conv3d reader_vol2col):
- **Reader/Writer:** `Noc` (`noc.h`) + `CircularBuffer`/`experimental::CB` (`circular_buffer.h`); reads/writes via `noc.async_read/async_write(accessor_or_cb, ..., {.page_id,.offset_bytes}, {.offset_bytes})` (CB-as-write-src uses read_ptr, CB-as-read-dst uses write_ptr). Trid ring via `experimental::{set_read_trid,async_read_barrier_with_trid}`. **Sentinel suffix now `noc.async_write_zeros(cb, bytes, {.offset_bytes})` + `write_zeros_l1_barrier()` → dropped `CB_ZERO`** (prebuilt-zero-row hack removed).
- **Compute:** `CircularBuffer` objects for the bare reserve/push/wait/pop (incl. runtime-swapped ping-pong CBs via `CircularBuffer(runtime_id).pop_front(...)`). LLK helpers (tilize/matmul_blocks/reduce_c/copy_tile/pack_tile/add_tiles_bcast_rows) and `get_tile_address(CB_CTRL,0)` keep constexpr/explicit ids — object CBs are dataflow-idiomatic; LLKs are inherently id-based (conv3d's own compute also uses free-fn cb + ids). `circular_buffer.h` is TRISC-safe (`COMPILE_FOR_TRISC`→PACK/UNPACK llk; no noc.h).
- **Profiling note:** `DeviceZoneScopedN` zones currently in both kernels (reader rdr_Q/rdr_Kchunk/rdr_mask, compute cmp_*) for the analysis above — strip before the final perf commit (small overhead + 125-scope/core limit).
