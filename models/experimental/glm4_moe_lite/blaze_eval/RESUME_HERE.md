# Resume point — GLM blaze integration

Written at a context checkpoint. Everything below is measured on the 32-chip BH Galaxy unless
marked otherwise. Full context: [`../BLAZE_EVALUATION.md`](../BLAZE_EVALUATION.md).

## READ FIRST — 2026-08-06: three claims below are refuted by benchmarks A and B

This document is append-only and has already misled this project once by carrying stale numbers in
its early sections while its later sections said the opposite. Nothing below is deleted; three
specific claims are now **superseded by direct measurement** and are marked in place. If you read
only one thing, read this table.

| where | the claim as written | what was measured | status |
|---|---|---|---|
| §"Why blaze loses, mechanically", point 2 (L806) | "Even per byte the DSM is 2.4x off ttnn's mcast matmul" | DSM core **285.0 GB/s** against ttnn's **278.3** — ratio **1.024**, parity | **FALSE** |
| §"CORRECTION + gated results" (L497) | "the output gather is now the whole problem: ~53 µs" | **6.2 µs** after patch 09's gather chunking — an 8.6x reduction | **STALE** |
| §"CORRECTION + gated results" W table (L486-492) | W-scaling table compared against "ttnn = 47.4 µs" | 47.4 µs is the **retracted DRAM-sharded** reference the model does not run; the real one is **10.5 µs** | **WRONG REFERENCE** |

Each is struck through in place with a `SUPERSEDED 2026-08-06` block underneath it; the history is
intact. The replacement measurements are appended at the end of this document. (Line numbers drift
as this file is appended to — search for `SUPERSEDED 2026-08-06` if they no longer match.)

**What replaces them** (`bench_a_concat_boundary.py`, `bench_b_worker_sweep.py`, one BH device,
all PCC gates ≥ 0.9998, additivity 0.6%, drift 0.2%):

- The concatenated `q_kv_a` program at K=2048 N=1344 W=4 decomposes as **core 15.6 µs, input
  boundary (`TileRowReplicate`) 17.5, output boundary (`GatherRowToDRAM`) 6.2, total 39.1** —
  boundaries are **60.1%** of the program. "Blaze arithmetic is slow" is false; blaze *boundaries*
  are expensive, which is what stage fusion removes.
- Blaze's remaining disadvantage at W=4 was **padding, not kernel efficiency**: N=1344 pads to 2048,
  so blaze reads 1.52x the bytes and the core alone still loses to ttnn's whole op in wall time.
- **Benchmark B settles that too.** Sweeping W ∈ {1..7} core-only, **W=6 pads only 1.14x and runs
  at 9.4 µs — 0.60x of W=4's 15.6, and faster than ttnn's entire 10.5 µs op.** Padding really was
  the whole residual. See the appended section at the end of this document.

## NEXT TASK (decided): widen DRAMStreamingMatmul past 8 cores

User's call, after seeing the numbers below: **stay on blaze.** Since both fused ops are
regressions *because* of the 8-worker limit, the only route to a blaze-based e2e win is removing
that limit. Everything else in this document is the evidence for why this is the task.

Goal: more than one worker per DRAM bank, so the matmul reaches something closer to ttnn's
152 GB/s instead of 56 GB/s. **This is a kernel redesign, not a config change.** Change points,
all in `blaze/ops/dram_streaming_matmul/common.py`, found by reading the source:

| line | what it does now | what it needs |
|---|---|---|
| 34 `dram_bank_worker_cores` | returns exactly 1 core per bank | return W cores per bank, still bank-adjacent on NOC_0 |
| 396 `num_banks = len(all_worker_cores)` | conflates core count with bank count | must become the true bank count, independent of cores |
| 401 `bank_id = idx % num_banks` | no-op today, since num_banks == len(cores) | **already the right shape** — becomes meaningful once 396 is decoupled |
| 274 `per_core_N = weights_shard[1] // tile_w` | per-core N derived from the DRAM weight shard | must subdivide one bank's shard across its W workers |

**The per-core plumbing already exists.** Line 414 passes `bank_id` as a per-core *map*
(`bank_id_map`), so per-core CT args are an established pattern here — an `n_offset` map can be
added the same way. That is a meaningful simplification over "redesign the page walk".

Concrete recipe:

1. `dram_bank_worker_cores(device, workers_per_bank=W)` -> W cores per bank, ordered
   `[bank0..bank7, bank0..bank7, ...]` so line 401's `idx % num_banks` maps correctly as-is.
   The extra cores will not be NOC-optimal (`get_pinned_optimal_...` only yields 8); any free
   core works, just with a longer hop.
2. `num_banks` = `device.dram_grid_size().x`, not `len(all_worker_cores)`.
3. `per_core_N = (weights_shard[1] // tile_w) // W`.
4. New per-core map `n_offset = (idx // num_banks) * per_core_N`, passed like `bank_id_map`.
5. **The one genuinely new piece:** consume `n_offset` in the reader's weight page index inside
   the kernel, so the two workers on a bank read disjoint column halves of the same shard.

Step 5 is the only unknown; steps 1-4 are mechanical. `weights_tensor_addr` (line 435) stays a
single scalar — the offset belongs in the page index, not the base address.

The real work is the last row. Today each core reads its bank's **entire** shard, so there is no
per-core column offset anywhere in the reader's page walk; W workers per bank means each reads a
disjoint column sub-range of the same shard. That offset has to be threaded through the reader CT
args and the page walk in `emit_dram_stream`.

That line 401 modulo is the encouraging sign: the original author anticipated W > 1, so the
bank/VC assignment logic should already tolerate it.

**Validation is already built.** Re-run both A/Bs — they gate PCC against the ttnn path they would
replace, so a wrong widening cannot pass as a fast one:

```bash
cd /home/ttuser/sdawle/tt-blaze && source env.sh && unset TT_MESH_GRAPH_DESC_PATH
python <tt-metal>/models/experimental/glm4_moe_lite/blaze_eval/native_boundary_ab.py   # 0.52x today
python <tt-metal>/models/experimental/glm4_moe_lite/blaze_eval/oproj_residual_ab.py    # 0.37x today
```

Targets to beat: q_kv_a 47.5 µs, o_proj+residual 69.1 µs. Either clearing ~1.1x is worth
integrating; both clearing it is worth ~2-6 ms/token. **Coordinate first — the other agent owns
this tree and `dram_streaming_matmul` is shared by every blaze op, so a regression here breaks all
of them, not just GLM's.**

## VERDICT: both GLM fused ops lose at the model boundary. Do not integrate either.

Two independent clusters, both correctness-gated against the ttnn path they would replace, both
measured with builds hoisted and the profiler correctly enabled:

| cluster | ttnn | blaze | | |
|---|---:|---:|---:|---|
| `GLMQKVAProjection` (q_a+kv_a, K=2048) | **47.5** | 94.0 | **0.52x** | PCC 0.999954 / 0.999938 |
| `GLMOProjResidual` (o_proj+residual, K=5120, replaces 5 dispatches) | **69.1** | 187.8 | **0.37x** | PCC 0.999932 |

The second was the strongest remaining candidate — the layer's largest matmul, five ttnn
dispatches collapsed into one program, one boundary pair amortising over far more work — and it
loses by *more*. It was measured with the chunked `TileRowReplicate` fix already in place.

### Why, quantified — and it is structural, not a tuning gap

`dram_bank_worker_cores` (dram_streaming_matmul/common.py:34) pins **exactly one worker per DRAM
bank** — 8 cores on Blackhole, no parameter, each owning a disjoint N slice. ttnn spreads the same
matmul over up to 80. Verified by reading the source, not inferred.

The achieved bandwidth shows what that costs. `w_o` is 5120x2048 bf8 ~= 10.5 MB:

| | time | achieved | of 512 GB/s peak |
|---|---:|---:|---:|
| blaze (8 bank workers) | 187.8 µs | **56 GB/s** (~7 GB/s/core) | 11% |
| ttnn (up to 80 cores) | 69.1 µs | **152 GB/s** | 30% |

**A single Tensix core cannot saturate a DRAM bank.** One worker per bank therefore strands most
of the device's bandwidth, and no amount of kernel tuning inside that layout recovers it — the
fix is more cores per bank, which is a `DRAMStreamingMatmul` redesign.

It also explains why the *bigger* cluster scored worse: larger K needs the activation replicated
to every bank worker first (K=5120 -> 160 pages vs K=2048 -> 64), so the work meant to amortise
the boundary grows the boundary instead.

Consistent with the earlier unexplained null: `GLM4_MOE_LITE_DS_CORE_CAP=8`, which hands ttnn
blaze's 8-core layout, measured 33.2 -> 33.4 ms. Constraining ttnn the same way did not help
either — what a structural parallelism limit looks like from the other side.

## HEADLINE: the boundary is closed, priced, and the q_kv_a cluster LOSES

The blocker described further down is **CLOSED** — a second agent added `TileRowReplicate` (input)
and `GatherRowToDRAM` (output) and wired them into `GLMQKVAProjection`, so it now consumes the
model's native 32x32 TILE DRAM activation and writes DRAM outputs. Gated on hardware:
**q_a PCC 0.999916, kv_a PCC 0.999910**. The section below is kept for its findings; its
conclusion is superseded.

With the boundary closed it could finally be *priced* instead of estimated, and the answer is
negative. `blaze_eval/boundary_decompose.py`, single BH device, profiler on, builds hoisted:

| variant | µs | |
|---|---:|---|
| A core only (blaze-native in, L1 out) | **36.5** | |
| B A + `TileRowReplicate` | 76.2 | input boundary **39.7** |
| C A + 2x `GatherRowToDRAM` | 77.0 | output boundary **40.6** |
| D both — what integration pays | **117.1** | additivity holds: B+C−A = 116.8 |
| ttnn fused `q_kv_a` (shipping path) | **47.4** | **blaze is 0.40x** |

**The verdict does not depend on the core.** The two boundary halves alone are 39.7 + 40.6 =
**80.3 µs, already more than the entire ttnn path at 47.4 µs**. Even with a hypothetically free
matmul this cluster loses as a drop-in. No core tuning rescues it.

The one escape hatch — keep outputs in L1 and adapt the downstream consumers — reaches B =
76.2 µs, still **0.62x**. Only removing *both* boundaries, i.e. a fully blaze-native neighbourhood
where the activation arrives already replicated, reaches A = 36.5 µs for **1.30x**: about
**0.5 ms of a 33.2 ms token (1.6%)**, and only if the whole surrounding chain is converted.

**Do not integrate this cluster as a drop-in.** That is a settled negative result, not a pending
task.

### ROOT CAUSE of the 80 µs boundary: a barrier inside the loop, in both micro-ops

Neither boundary op is bandwidth-limited — both are latency-serialised:

- `tile_row_replicate/kernels/op.hpp` reads a **whole 2 KB tile to extract 64 bytes** (one row),
  and calls `noc_async_read_barrier()` **inside the per-tile loop**. At GLM's K=2048 that is
  `num_tile_cols = 64` full DRAM round-trips, ~0.6 µs each → **~40 µs**, matching the measured
  39.7 µs.
- `gather_row_to_dram/kernels/op.hpp` has the same shape: `noc_async_write_page` followed by
  `noc_async_write_barrier()` per page.

**Batching is confirmed to work, on hardware.** Hoisting the barrier and reading only the two
face lines took the fused op **117.3 → 81.3 µs (−36 µs)**, i.e. it removed essentially the whole
input boundary.

**But that version is WRONG — PCC 0.71, not 0.999.** Cause: each face line is 32 bytes, so
successive destination addresses land at 32-mod-64, and Blackhole DRAM reads require 64-byte
alignment. Roughly half the reads land misaligned. The gate caught it; the timing win was real
and the correctness was not. Reverted — `blaze/ops/tile_row_replicate/` is byte-identical to the
other agent's version, verified with `cmp` and `git diff`.

The fix that should work is **chunked** rather than fully batched: keep the aligned full-page
reads, give `scratch` depth ~8 instead of `num_pages=1`, issue 8 page reads, one barrier, then 8
copies. That is 8 barriers instead of 64 at 16 KB of L1, and preserves page alignment throughout.

### FIXED, correctly, and landed in the blaze tree

Chunked the reads: `scratch` depth 1 -> `_READ_CHUNK = 8`, issue 8 page reads, one barrier per
chunk. Reads stay **whole-page** deliberately — that is what avoids the alignment trap above.

    q_kv_a fused op   117.3 -> 94.0 us     input boundary 39.7 -> ~16 us
    PCC vs ttnn       0.999954 / 0.999938  (identical to before the change)

chunk=32 gives only 2 us more for 4x the L1 (64 KB vs 16 KB), so 8 is the trade taken — this op's
own docstring notes L1 headroom matters for the SDPA that follows.

Saved as `blaze_eval/upstream/07-tile-row-replicate-chunked-reads.patch`, because
`blaze/ops/tile_row_replicate/` is **untracked** in the tt-blaze tree — the other agent has not
committed it, so `git diff` there reports nothing and the change would be lost on a reset. Use
`cmp` against a backup, not `git diff`, to verify state of those files.

### It still does not rescue q_kv_a — but it is worth ~3.5x more to the next op

q_kv_a is now 47.5 us ttnn vs 94.0 us blaze = **0.52x**. Still a loss, still not to be integrated;
the remaining gap is the output gather (~40 us, same barrier-per-page defect, not yet fixed) plus
a 36.5 us core against a 47.5 us ttnn op that is simply not slow enough to beat.

The fix matters because of where it lands next. `GLMOProjResidual` — the other agent's third GLM
op, replacing **five** ttnn dispatches around the layer's largest matmul (K=5120, N=2048) — calls
`TileRowReplicate` with **160 + 64 = 224 pages** against q_kv_a's 64. Same defect, ~3.5x the
exposure: it would have paid ~134 us of input boundary and should now pay ~55 us. That op is the
best remaining candidate for an e2e win, because one boundary pair amortises over far more work.

**It has no test yet and is mid-development.** Do not A/B it until the other agent lands one.

### Two measurement corrections that invalidate earlier numbers

1. **0.53x is retracted.** The first boundary A/B rebuilt the `FusedProgram` and re-ran `emit()`
   inside the timed callable, while ttnn paid nothing equivalent because it caches programs. The
   corrected bench hoists the build and **verifies repeated `run()` is bit-identical across 3
   runs** — necessary, since `run()` mutates `_prepare_for_build`/`_compaction_applied` state.
2. **`ab_harness.set_profiler_env()` must precede the ttnn import.** It did not. `_measure()`
   returns `None` on zero profiler samples instead of raising, so the misordering degraded
   silently into a missing result rather than an error.

### Unresolved: 36.5 µs vs the 9.5 µs this evaluation was built on

Variant A is the *same configuration* that was recorded at 9.5 µs and produced the headline
**4.76x**. It now measures 36.5 µs, which would make the core only ~1.3x. Candidates: the
profiler-env ordering bug above (undercounts silently), or the op's new default `subblock_k=2`
(its comment says 1 deadlocks on a *mesh*; this is a single device). A sweep is wired into
`boundary_decompose.py` behind `GLM_SWEEP_SUBBLOCK_K=1`, left **opt-in** because four fresh JIT
compiles ran past 15 min of Galaxy time without finishing — and because the verdict above is
insensitive to the answer. **Treat every per-op speedup in this document as suspect until
re-measured under the corrected harness.**

## State of the two GLM fused ops

Both are **correctness-gated and cluster-measured**. Neither is in the model yet.

| op | correctness | cluster timing | vs |
|---|---|---|---|
| `GLMQKVAProjection` (q_a + kv_a, shared act) | PCC **0.9999** both outputs | 45.1 → **9.5 µs**, **4.76x** | ttnn's *fused* `q_kv_a` (the real shipping path) |
| `GLMQANormQBProjection` (q_a RMSNorm → q_b, chained) | `pcc_vs_device` **0.99989**, `pcc_vs_model` **0.99988** | 25.9 → **21.1 µs**, **1.22x** | ttnn `rms_norm` + `q_b` matmul |

Sources live in the tt-blaze tree; copies archived here under `glm_qkv_a_projection/` and
`glm_qa_norm_qb_projection/`, with `GLM_FUSED_OP_HANDOFF.md` for the second.

**Upper bound if both landed and fully translated: ~1.9 ms/token of 33.2 ms (~5.7%).** Treat as a
ceiling, not a forecast — see the two null results below.

## START HERE ON RESUME — in this order, no shortcuts

The last run **hung** (EXIT=124, and **0 JIT compiles** in its log, so it was genuinely hung, not
compilation latency). **Treat it as a real new blocker.** Per F12 a hang degrades the device while
`open/close` still succeeds, so do not infer health from a device opening.

1. **Reset first, unconditionally** — do not skip this on the assumption the device is fine:

   ```bash
   /home/ttuser/sdawle/tt-metal/python_env/bin/tt-smi -r
   ```
   On Galaxy it warns CPLD FW v1.16+ is wanted for `-r` and suggests `-glx_reset`; `-r` has
   worked here. Block on completion — never start device work while a reset is in flight.

2. **Then the known-good control**, before believing any new result:

   ```bash
   cd /home/ttuser/sdawle/tt-blaze && source env.sh && unset TT_MESH_GRAPH_DESC_PATH
   timeout 300 python -m pytest glm47_all_shapes_check.py -q     # expect 6 passed, ~11 s
   ```

3. **Then instrument the q_kv_a row-major CB path before retrying it.** Do not just re-run
   `gate5_shard_shape_attempt.py` and hope. Use the documented triage path:
   `TT_METAL_INSPECTOR=1`, then
   `python tt-metal/tools/triage/triage.py --run=dump_callstacks --inspector-log-path <cwd>/generated/inspector`
   (the logs land in `<cwd>/generated/inspector`, **not** the `/tmp/tt-metal/inspector` the error
   message claims). The hanging process must still be alive for triage — kill it only afterwards,
   then reset again before the next attempt. `BLAZE_DEBUG_KERNELS=1` names the stalling phase.

   The suspects are the reshape/reshard chain in `build_act()` and the `(64,32)` shard CB view;
   triage distinguishes them, guessing does not.

## The one open blocker: feeding the model's activation to `GLMQKVAProjection`

`DRAMStreamingMatmul` wants the activation replicated per DRAM-bank worker as **1×32 tiles**.
The model holds `[1,1,32,2048]` in 32×32 tiles. Building the former from the latter costs
**4.8 µs/layer** against ~36 µs of headroom, so the reshard is affordable — that is settled.

What is not settled is declaring the CB. Findings, in order:

1. Passing the raw row-major tensor fails: blaze calls `tensor.get_tile()` →
   `'NoneType' object has no attribute 'height'`.
2. Pass a **CBHandle** instead, built with `f.cb_from_tensor(t, tile=Tile([1,32]), page_size=64)`.
   `tile=` alone is ignored — `page_size=` is what selects the branch that honours it
   (`fused_program.py:1380-1387`). This clears the tile error.
3. Next wall: `K mismatch: act gives 32, weights gives 2048`.
   `K_from_act = num_pages * tile_w` (`dram_streaming_matmul/common.py:278`), and for row-major
   `n_pages = shard.shape[0]` (`fused_program.py:1408`) — the shard **row count**, which ignores
   `page_size`. With shard `(1, 2048)` that is 1 page, so K reads as 32.
4. `total_size=` would set page count directly, but it reaches `_resolve_tensor_geometry` and is
   then **rejected by `BlazeProgram.cb_from_tensor`** — not plumbed through.
5. **`total_size` IS available** — not on `cb_from_tensor`, but `cb_from_tensor_overlapped(t,
   address_offset, total_size, page_size, tile=...)` takes all three as first-class arguments
   (`fused_program.py:1601`). No blaze change needed. With `total_size=K*2, page_size=64,
   tile=Tile([1,32])` the K-mismatch error is gone.

6. **THE REAL BLOCKER — the op hangs on a DEVICE-BUILT activation.** Tried twice:

   | attempt | shard | CB | result |
   |---|---|---|---|
   | `gate5_shard_shape_attempt.py` | `(64,32)` + reshape | `cb_from_tensor` | **hang** |
   | `gate6_overlapped_cb_attempt.py` | `(1,2048)`, no reshape | `cb_from_tensor_overlapped` | **hang** |

   Both hang, so it is **not** the reshape and **not** the shard shape — those differed. What is
   constant is that the activation was built on device and is **ROW_MAJOR**, where blaze's own
   `_make_act_tensor` produces **TILE** layout with a 1x32 tile. The same fused op with that
   from_torch activation gates at **PCC 0.9999**, and `build_act()` measured alone completes fine
   at 4.8 us — so neither half is broken in isolation; the hang is the op consuming a row-major
   backing tensor.

   **Leading hypothesis:** for a 1x32 tile the bytes ought to be identical to row-major, but ttnn
   may pad or align row-major shard rows differently than tiled pages, so the reader walks wrong
   addresses and waits forever. Test by comparing the two tensors' `buffer_address()`,
   page/stride metadata and shard-spec padding *before* running the op — that is a host-side
   comparison and costs no device time.

   Next after that: option (b) below (retilize inside the fused op), which sidesteps row-major
   entirely rather than trying to make blaze accept it.

7. Superseded by 5/6 — declaring the per-core shard as `(64, 32)` instead of `(1, 2048)`
   — identical bytes, but makes `n_pages = 64` and `page_size = 64` fall out naturally. This
   **hung the device**. Script preserved at
   `/tmp/.../scratchpad/gate5_script.py` (regenerate from this doc if gone).
   Unknown whether the hang is the reshape/reshard chain or the CB view; use `triage-hang`
   (`TT_METAL_INSPECTOR=1`, logs land in `<cwd>/generated/inspector`, **not** `/tmp/tt-metal`).

Two cleaner options than fighting this from the caller:

- **(a)** Plumb `total_size` through `BlazeProgram.cb_from_tensor` (one-line-ish, blaze-side).
- **(b)** Do the retilize **inside** the fused op as its first phase. blaze ships `Retilize` for
  the 1×32 → N×32 direction; the input direction needs the reverse. This is the architecturally
  right answer: it removes the 4 ttnn conversion ops entirely instead of paying them.

`GLMQANormQBProjection` has the **same** activation requirement, so whichever fix lands unblocks
both.

## Remaining steps to the end-to-end swap

1. Control the device (above).
2. Close the CB gap via (a) or (b); re-gate PCC ≥ 0.99 for both ops.
3. Wire into `tt/blaze_ops.py` (the seam already exists, import-guarded and inert on our tree),
   then swap the call sites behind `GLM4_MOE_LITE_BLAZE_QKV_A=1`:
   - `attention_decode.py:144` — the fused `w_q_kv_a` matmul + its two `_safe_slice`s
   - `attention_decode.py:327-328` — `q_a_layernorm` then `attn_linear(w_q_b)`
4. Per-layer PCC gate against the ttnn path, then traced end-to-end.

### The end-to-end command, and what to compare against

The model runs traced **in blaze's tree** (that work is done). Three optimizations must be off
there because blaze's older ttnn lacks the parameters they use:

```bash
cd /home/ttuser/sdawle/tt-blaze && source env.sh && unset TT_MESH_GRAPH_DESC_PATH
HF_HOME=/dev/shm/hf \
GLM4_MOE_LITE_FUSE_DOWN_ROUTING_SCALE=0 \
GLM4_MOE_LITE_FUSED_COLLECTIVE_EPILOGUE=0 \
GLM4_MOE_LITE_FUSED_ROUTER=0 \
timeout 560 python tt-metal/models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "What is the capital city of Australia? Answer with just the city name." \
  --simulate-context-len 128 --min-cache-tokens 256 --max-new-tokens 32 --batch-size 1 \
  --mesh-rows 4 --mesh-cols 8 --kv-cache-dtype bf16 --phase both \
  --enable-trace --trace-mode sampling --cache-dir /dev/shm/ttnn_cache_blaze
```

**Compare against 34.8 ms/token** — same tree, same flags, no blaze op. NOT against 33.2 ms,
which is our tree with all optimizations on. Keeping those straight is the whole point:

| config | our tt-metal | blaze's tt-metal |
|---|---:|---:|
| full default flags | 33.2 | cannot run |
| the three off | 34.1 | **34.8** ← the baseline to beat |

## Two null results that should temper expectations

Both measured in the shipping traced regime, and both argue the cluster wins may not translate:

- **`GLM4_MOE_LITE_DS_CORE_CAP=8`** — giving ttnn blaze's 8-core bank layout changed the step
  33.2 → **33.4 ms**. No gain.
- **The bandwidth ceiling** — doubling every dense weight's bytes costs only **+3.9 ms (11.7%)**,
  yet the six per-op wins sum to 31.5% of the step. Most of that cannot be on the critical path.

Also relevant: fusing two projections that *share* an input contributed only ~4% (the rest was
`DRAMStreamingMatmul` beating ttnn), and the chained fusion that genuinely removes a DRAM
round-trip is worth 1.22x. Mechanism 2 works; at these cluster sizes it is small.

## Environment

- `/dev/shm` holds the 62.5 GB checkpoint (`HF_HOME=/dev/shm/hf`) and two converted-weight caches
  (`ttnn_cache`, `ttnn_cache_blaze`), ~93 GB total. Needed for any end-to-end run; `rm -rf` to
  reclaim.
- tt-blaze working tree holds uncommitted F3 (`mla_q_grid`, `scattered_q_heads`) and F11
  (`num_total_experts`, `zero_tail`) work — **validated on hardware this session**: F3 33/33;
  F11's hang fixed (17.8 s vs >700 s), GLM-5 dims pass, GLM-4.7's 64 experts run but PCC 0.0235.
  Do not clobber: syncs must only replace `tt-metal/models/experimental/glm4_moe_lite`.

## SIZING UPDATE: the widening is ~5 lines of kernel, not a page-walk redesign

Read the reader to size the "one unknown" step properly. `kernels/op.hpp:140` derives the entire
weight stream from a single base:

    weights_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, weights_tensor_addr)

The DRAM shard is **column-major tile-shuffled**, so a column sub-range is a *contiguous byte
range*. Two workers on one bank therefore differ by a constant byte offset into the same shard —
there is no page-index arithmetic to restructure. Add a per-core `weights_byte_offset` CtRtArg
(same mechanism `bank_id` already uses at op.hpp:65/135) and add it to the base:

    get_noc_addr_from_bank_id<true>(dram_bank_id, weights_tensor_addr + w_off)
    w_off = (idx // num_banks) * (shard_bytes // W)

**Make it opt-in: `workers_per_bank=1` by default.** At W=1 the core list, `num_banks`,
`per_core_N` and `w_off` are all unchanged, so the change is a provable no-op until W>1 is asked
for. That is what makes it safe to land incrementally in a kernel every blaze op shares — it can
absorb that, but not a breaking edit. This removes the main reason the widening kept being
deferred.

Steps 1-4 of the recipe above are mechanical; step 5 is ~5 lines plus a per-core map. Roughly an
hour of work with the hardware gate already built, not the redesign it was previously scoped as.

## BREAKTHROUGH: the widening works — 3.02x on the matmul core

Implemented and measured. `BLAZE_DSM_WORKERS_PER_BANK` (default 1) puts W workers on each DRAM
bank; per-core N is divided by W and each worker takes a contiguous byte offset into its bank's
shard. Patch: `upstream/08-dram-streaming-matmul-workers-per-bank.patch`.

**W=1 is a verified no-op** — 93.7 µs and PCC 0.999954/0.999938, identical to before the change.
That is what makes it safe in a kernel every blaze op shares.

Matched work (N padded to 1024 for every W, so the only variable is core count):

| W | cores | core only | + input boundary |
|---|---:|---:|---:|
| 1 | 8 | 48.4 µs | 64.7 |
| 2 | 16 | 29.9 µs | 46.0 |
| 4 | 32 | **16.0 µs** | **33.6** |

**3.02x on the core, and at W=4 the op with its input boundary is 33.6 µs against ttnn's 47.4 —
1.41x, a win** — provided the outputs stay in L1. The 8-worker deficit was the entire problem and
it is removable.

### Two constraints found

1. **`N % (32 * banks * W) == 0`.** A core's shard must be a whole number of 32-wide tiles.
   GLM's `q_a`=768 / `kv_a`=576 need padding to 1024 at W=2/W=4 (33%/78% waste), but **o_proj's
   N=2048 divides exactly at W=2 and W=4** — the big matmul is the clean fit. W=8 needs
   N % 2048 == 0 and is blocked at these shapes.
2. **`GatherRowToDRAM` still assumes 8 senders** — "sender pages do not cover destination width".
   This is now the blocker for the *whole* fused op, and it is the same op that still has the
   barrier-per-page defect. Fixing it is the remaining work.

### Next, in order

1. Teach `GatherRowToDRAM` the sender count (and give it the chunked-barrier treatment).
2. Re-run `native_boundary_ab.py` and `oproj_residual_ab.py` at W=2/W=4 — o_proj is the better
   target on both shape and size grounds.
3. If they clear ~1.1x, enable `GLM4_MOE_LITE_BLAZE_QKV_A=1` and measure traced e2e against
   34.8 ms in the blaze tree.

### CORRECTION + gated results

The 3.02x above was measured **before** a correctness gate and with a core-ordering bug. Corrected
and now trustworthy: the full op gates at **PCC 0.999947 / 0.999957 at W=2 and W=4**, so the core
numbers below are computing the right answer.

The bug: cores were ordered `[b0..b7, b0..b7]`, so sender *i* was not output column block *i*,
which is what `GatherRowToDRAM` assumes. Per-core maths was right, columns were assembled in the
wrong order — **PCC 0.13**. Fixed by ordering bank-major (`[b0w0, b0w1, b1w0, ...]`), with
`bank_id = idx // W` and `offset = (idx % W) * per_core_N * Kt * tile_size`.

Full decomposition, N padded to 1024 at every W so only core count varies:

| W | cores | A core | B +input | C +gather | D both |
|---|---:|---:|---:|---:|---:|
| 1 | 8 | 48.4 | 64.7 | 102.3 | 118.9 |
| 2 | 16 | 30.1 | 46.5 | 84.1 | 100.2 |
| 4 | 32 | **16.1** | **33.3** | 69.4 | 86.5 |

~~ttnn = **47.4 µs**.~~

- **The core scales: 3.01x at W=4.** Confirmed and gated.
- ~~**`B` at W=4 = 33.3 µs beats ttnn's 47.4 — 1.43x.** That is a real win, and it is available
  *only if outputs stay in L1*.~~
- ~~**The output gather is now the whole problem: ~53 µs, and flat in W** (53.9 at W=1, 53.4 at
  W=4). It does not benefit from more workers and is 3.3x the cost of the core it drains.~~

> **SUPERSEDED 2026-08-06 — the reference is retracted and the gather number is stale. Do not quote
> this table against ttnn, and do not quote the ~53 µs at all.**
>
> **(a) `ttnn = 47.4 µs` is the wrong reference.** It is the **DRAM-sharded** path.
> `GLM4_MOE_LITE_DRAM_SHARDED_ATTN` is off by default (`layer_weights.py:118`, "resharding overhead
> + trace issues"), so decode never runs it. The matmul the model actually calls is `ttnn.linear`
> with a 1D-mcast program config over an interleaved DRAM weight, L1 in and out: **10.50 µs** —
> 4.5x cheaper. Every ratio computed against 47.4 in this section is inflated by that factor, so the
> "1.43x win" for variant B is really **0.32x**. This is the same retraction recorded at the end of
> this document for the 1.29x; it applies to this table too.
>
> **(b) The ~53 µs gather no longer exists.** Patch 09 chunked `GatherRowToDRAM`'s
> barrier-per-page loop. Re-measured at the concatenated shape by `bench_a_concat_boundary.py`
> (K=2048, N=1344, N_pad=2048, W=4), the output boundary is **6.2 µs (C − A)** — an **8.6x
> reduction**, and the first direct measurement of it. The remaining boundary cost is now the
> *input* side: `TileRowReplicate` at **17.5 µs**, 2.8x the gather. "The output gather is the whole
> problem" was true when written and is false now; the sentence below about it being "the last
> blocker" should be read the same way.
>
> **(c) This table is also not the shipping program.** It is `GLMQKVAProjection` — two separate DSM
> programs at N_pad=1024 each and two gathers, pre-patch-09. The shipped program is one DSM at
> N_pad=2048 with one gather. Same weight bytes, different program; the two cannot be spliced. The
> replacement single clean profile at the concatenated shape is in the appended section at the end
> of this document.

So `D`, the drop-in integration, is still 86.5 µs = 0.55x. The widening moved the bottleneck from
the matmul to `GatherRowToDRAM`; it did not remove it.

### The remaining work is now one op

`GatherRowToDRAM` carries the same barrier-per-page defect I fixed in `TileRowReplicate`
(`noc_async_write_page` + `noc_async_write_barrier()` per page, kernels/op.hpp:94-95). Chunking it
bought 23.4 µs on the replicate; the same treatment here is the last blocker. If it lands anywhere
near the replicate's improvement, `D` at W=4 lands close to or under ttnn's 47.4 µs and the flag
can be turned on.

Note also W=2 is *worse* than W=1 on the full op (100.2 vs 118.9 is better, but on the real
unpadded shapes 100.6 vs 94.0 is worse) because q_a/kv_a pad 768/576 -> 1024. o_proj's N=2048
divides exactly and pays no such penalty — it remains the better integration target.

### The gather's ~53 µs, broken down (read from the kernel)

`gather_row_to_dram/kernels/op.hpp:80-96`, the receiver's DM0 loop, per output tile:

```
cb_reserve_back(tile, 1)
for (i < dst_page_size/4) out[i] = 0;      // zeros a FULL 2 KB tile, scalar loop
for (i < 8) { out[i] = row[i]; ... }       // fills 16 values
noc_async_write_page(page, writer, ...)
noc_async_write_barrier();                 // <-- barrier per page
```

Two costs, both fixable, and `tile: Internal = Internal(num_pages=1)` (op.py:17) is what forces
the serialisation — exactly the role `scratch` played in `TileRowReplicate`:

1. **Barrier per page.** 32 output tiles at N=1024 → 32 serialised DRAM round-trips, ~19 µs.
   Fix: raise the `tile` CB depth to CHUNK, write CHUNK pages, one barrier. `num_pages` is 32 at
   N=1024 and 24 at N=768, both divisible by 8; guard with a fallback to 1 as
   `TileRowReplicate._READ_CHUNK` does.
2. **Zeroing 2 KB per tile with a scalar word loop** to write 16 useful values — ~16K word-writes
   for 32 tiles, ~12 µs. Only row 0 of each output tile is ever read downstream. Either hoist the
   zeroing (the destination is reused every token) or zero only the two face lines that are not
   overwritten.

Together these plausibly account for most of the 53 µs. `D` at W=4 is 86.5 µs against ttnn's
47.4; the core is 16.1 and the input boundary 17.2, so the gather has to come down to roughly
14 µs for the drop-in to break even — which is in range if both fixes land, but is not a
certainty. **Measure, do not assume.**

### Gather chunking landed — and the residual is write amplification, not barriers

Applied the same chunked-barrier fix to `GatherRowToDRAM` (`_WRITE_CHUNK = 8`, tile CB depth 8,
one barrier per chunk). Patch: `upstream/09-gather-row-to-dram-chunked-writes.patch`.
PCC unchanged: 0.999954/0.999938 at W=1, 0.999947/0.999957 at W=4.

| config | full op | vs ttnn 47.4 |
|---|---:|---:|
| W=1, original | 117.3 µs | 0.40x |
| W=1, + replicate fix | 94.0 | 0.50x |
| W=4, + widening | 86.4 | 0.55x |
| W=4, + gather chunking | **78.6** | **0.60x** |
| W=4, write_chunk=32 | 77.1 | 0.62x |

**1.52x faster than where this started, all gated — and still a 0.62x loss.**

chunk=32 buys only 1.5 µs over chunk=8 for 4x the L1, so the barriers are essentially gone. The
residual gather cost (~44 µs) is **write amplification**: it writes a full 2 KB DRAM page per
output tile when only **64 bytes** (row 0, two face lines) are meaningful — 32x more traffic than
needed, plus a scalar loop zeroing all 512 words of each page.

### The one remaining idea, and its hazard

Write only row 0: two 32-byte `noc_async_write`s per tile at page offsets 0 and 512, skipping the
zeroing entirely (the destination is pre-allocated and only row 0 is ever read downstream).
That removes ~44 µs and would put the op near 34 µs — **comfortably under ttnn's 47.4**.

**The hazard is the one that already bit me once.** 32-byte writes put successive addresses at
32-mod-64, and Blackhole DRAM wants 64-byte alignment; the analogous read optimisation in
`TileRowReplicate` returned PCC 0.71 for exactly this reason. Either pair the two face lines into
one 64-byte aligned write, or verify alignment before trusting any speedup. **The PCC gate in
`native_boundary_ab.py` catches it in one run — do not skip it.**

## RESULT: q_kv_a is now FASTER than ttnn — 1.29x, gated

Writing only row 0 of each destination tile was the last piece. The gather had staged a full tile,
zeroed all 512 words with a scalar loop, and written the whole 2 KB page to deliver **64 bytes** —
32x the needed traffic. Replaced with two 32-byte writes per tile at page offsets 0 and 512.

**The alignment hazard did not bite, and why is worth keeping:** both offsets are multiples of 64,
so the DRAM side is 64-byte aligned as Blackhole requires. The `TileRowReplicate` attempt that gave
PCC 0.71 failed because its *L1 destination* advanced by 32 and landed at 32-mod-64. Same idea,
other side of the transfer, opposite outcome — check which side carries the alignment requirement
before assuming a partial transfer is unsafe.

Full arc, every step PCC-gated against the ttnn op it replaces:

| step | full op | vs ttnn 47.4 |
|---|---:|---:|
| original | 117.3 µs | 0.40x |
| + `TileRowReplicate` chunking | 94.0 | 0.50x |
| + `DRAMStreamingMatmul` widening (W=4) | 86.4 | 0.55x |
| + gather barrier chunking | 78.6 | 0.60x |
| + **gather row-0-only writes** | **36.8** | **1.29x** |

**3.19x faster than where this started, and beating the shipping path.** By W: W=1 0.85x,
W=2 0.95x, **W=4 1.29x** (PCC 0.999947/0.999957 throughout). W=4 is required — there is no win at
the default worker count.

Upper bound if it translates: **+0.51 ms/token of 33.2 ms (~1.5%)**. An upper bound, and the traced
regime has repeatedly failed to honour cluster wins here — **measure, do not assume.**

### Must be resolved before shipping

The row-0-only write is **batch=1 only**: it writes one row per destination tile because that is
all decode at bs=1 needs. A batched destination needs the per-row loop restored, guarded on batch.
Correct for the measured configuration, not general.

### o_proj: two small blockers, neither from the widening

1. **W=1:** `ResidualAdd needs 8 addressable DST tiles ... fp32_dest_acc_en=True ... gives 4.`
   Bench fix: pass `dst_full_sync_en=True` on the `FusedProgram`, as `BlazeCompiler` does.
2. **W=4:** `out width (2048) does not match o_weights N (8192)`. `GLMOProjResidual.emit` computes
   `hidden = weights_shard[1] * len(worker_cores)` — the same core-count/bank-count conflation
   already fixed in `common.py`. Needs the true bank count.

o_proj remains the better target: N=2048 divides exactly at W=2/W=4, so it pays none of the
768->1024 padding waste q_a/kv_a do.

### Next
1. Fix the two o_proj blockers; re-run `oproj_residual_ab.py` at W=4.
2. Enable `GLM4_MOE_LITE_BLAZE_QKV_A=1` **with `BLAZE_DSM_WORKERS_PER_BANK=4`** and measure traced
   e2e in the blaze tree against **34.8 ms** (not 33.2).

### CORRECTION: the integration seam is NOT a flag flip away

`blaze_ops.qkv_a()` calls `prepare_qkv_a_weights(device, w.w_q_a_torch, w.w_kv_a_torch)`, and
**neither attribute exists** anywhere in the model — I introduced them without checking. The seam
is inert today only because `blaze_available()` is False on this tree; the moment blaze imports
and the flag is set, it raises `AttributeError`.

So the seam is correctly wired at the *call site* but not at the *weight source*. What is missing:

- the torch-side `w_q_a` / `w_kv_a` (or their ttnn tensors converted back) must be reachable from
  the weight object at decode time, or
- better, `prepare_qkv_a_weights` should run once at **weight-load** time and cache the
  DRAM-sharded, tile-shuffled result on the weight object, rather than needing torch tensors on
  the hot path at all. The prep is a load-time transform — that is how it was described in
  `blaze_ops.py` and it should be built that way.

Also, the model-side change lives in **this** tt-metal checkout. The e2e run happens in the blaze
tree, so `tt-metal/models/experimental/glm4_moe_lite` must be synced across first (and *only* that
directory — the tt-blaze working tree holds uncommitted F3/F11 work).

**Corrected next steps:**
1. Plumb the weights at load time (above). Without this, enabling the flag raises.
2. Sync `models/experimental/glm4_moe_lite` into the blaze tree.
3. `GLM4_MOE_LITE_BLAZE_QKV_A=1 BLAZE_DSM_WORKERS_PER_BANK=4`, traced, vs **34.8 ms**.
4. Restore the per-row loop in the gather for batch > 1 before this is anything but a bs=1 demo.

### The weight layouts do not match — and the simpler op may be the better one

The model's **default** decode path holds `w_q_kv_a`, a single **concatenated** 2048x1344 weight
(`decoder_layer_tt.py:913`, `attention_decode.py:143`). `w.w_q_a` / `w.w_kv_a` exist only on the
non-default unfused branch. `GLMQKVAProjection` is built around **two separate** weights, so it
does not match what the model actually carries.

Two ways out, and the second is probably better:

1. Split `w_q_kv_a` into its two column ranges at load time and keep the two-matmul fused op.
2. **Drop `GLMQKVAProjection` for this call site and emit a single `DRAMStreamingMatmul` over the
   concatenated 2048x1344 weight**, which is exactly what the ttnn path does — and exactly what
   the 1.29x was measured against.

Option 2 is simpler and likely faster: `GLMQKVAProjection`'s only structural advantage is sharing
one activation between two matmuls, and a concatenated weight already shares it by construction,
in one matmul instead of two serialised on the same cores. The op's own docstring concedes the
two matmuls "serialise on it rather than overlapping", and that fusing shared-input projections
contributed only ~4% — the rest was `DRAMStreamingMatmul` beating ttnn. All of the measured win
comes from the streaming matmul plus the three boundary fixes, none of which need the two-weight
structure.

That also removes the padding penalty: concatenated N=1344 pads to 2048 at W=4 (N % 1024 == 0),
against 768->1024 plus 576->1024 separately, i.e. one 2048-wide matmul instead of two 1024s.

**Recommended next session:** implement option 2 behind the existing flag, prepare the
concatenated weight once at load time, then measure e2e. It is less code than the weight plumbing
option 1 needs, and it is closer to what the model already does.

---

## 2026-08-06 — Option 2 landed, gated, and measured e2e. It is a **3.7x regression**, and the reason is that every prior measurement used the wrong reference.

**Verdict: do not enable `GLM4_MOE_LITE_BLAZE_QKV_A`. The op is numerically correct and is faster
than the ttnn matmul it was benchmarked against, but that matmul is not the one the model runs.**

### Headline

Option 2 is implemented, PCC-gated in-model, and measured traced e2e. It works, it is correct, and
it makes the model **8.2% slower**. The 1.29x was real but measured against a `ttnn` configuration
(`DRAM_SHARDED_ATTN`) that the model **deliberately does not use**. Against the matmul decode
actually calls, blaze is **0.27x**.

### Same-session traced A/B, Blackhole Galaxy 4x8, 31 decode steps each

All four runs back-to-back in one shell, same tree, same flags, `FUSE_DOWN_ROUTING_SCALE=0
FUSED_COLLECTIVE_EPILOGUE=0 FUSED_ROUTER=0`, profiler env explicitly `unset` and verified.

| run | device/step | min | p50 | max | wall/step | vs baseline |
|---|---|---|---|---|---|---|
| **A** baseline, no blaze | **33.242 ms** | 33.118 | 33.241 | 33.337 | 34.666 ms | — |
| **B** blaze, W=4 | **35.965 ms** | 35.833 | 35.970 | 36.065 | 37.498 ms | **+2.723 ms (+8.19%)** |
| **C** arena-only (allocate, don't run) | 33.244 ms | 33.135 | 33.246 | 33.346 | 34.683 ms | +0.002 ms (+0.01%) |
| **D** blaze, W=1 | 36.503 ms | 36.383 | 36.509 | 36.588 | 38.048 ms | +3.261 ms (+9.81%) |

Spread is tight — max-min is ~0.2 ms in every run, so a 2.7 ms delta is ~13 sigma. This is not noise.

Baseline A reproduces the expected 34.8 ms/token wall to within 0.13 ms.

**+2.723 ms / 47 layers = +58 µs per layer.** Run C isolates the memory cost: allocating the
47 KB/core shared L1 arena and the prepared DRAM weight, then running ttnn anyway, costs **2 µs
per step, i.e. nothing**. The whole regression is the op itself. Run D confirms the widening
tuning transfers (W=4 beats W=1 in-model, as in isolation) — it is just not enough to matter.

### Flag and execution confirmed

Both were faked once in this project's history, so both are proven from inside the run:

```
BLAZE_QKV_A prep: GLM4_MOE_LITE_BLAZE_QKV_A='1' BLAZE_DSM_WORKERS_PER_BANK='4'
                  K=2048 N=1344 N_pad=2048 banks=8 W=4 dtype=DataType.BFLOAT8_B
BLAZE_QKV_A first call site hit: x shape=(1,1,1,2048) padded=(1,1,32,2048)
                  dtype=BFLOAT16 mem=BufferType.L1
BLAZE_QKV_A shared scratch arena: 46848 B/core over 120 cores, 6 CBs
```

The prep line appears **47 times** (once per layer, load time) and the call-site line **once** in
run B; both appear **zero times** in baseline run A. The run is also 2.7 ms slower — a run where
the op did not execute could not be slower.

### Per-layer PCC gate — PASS

`GLM4_MOE_LITE_BLAZE_QKV_A_GATE=1` runs the real ttnn `q_kv_a` alongside blaze on the *same*
activation, inside the real model, every layer, and PCCs them. Not a torch golden — the actual op
being displaced. **94 gated layer-instances:**

```
q_a  min=0.999932  mean=0.999961
kv   min=0.999951  mean=0.999975
```

Correctness is not the problem. (The gate mode aborts at trace capture with "Reads are not
supported during trace capture" — expected; it is a correctness mode, not a timing mode.)

### Root cause: the reference was wrong for 4 sessions

`GLM4_MOE_LITE_DRAM_SHARDED_ATTN` is **off by default** — `layer_weights.py:118` says why:
*"resharding overhead + trace issues"*. So decode's `q_kv_a` is **not** a DRAM-sharded matmul. It
is `linear_helpers.n` → `ttnn.linear` with a `MatmulMultiCoreReuseMultiCast1DProgramConfig` over an
**interleaved** DRAM bf8 weight, L1 activation in, L1 activation out, **no reshard at either end**.

Benched all three on one device at bf8 (`concat_qkv_a_ab.py`, extended this session):

| path | time | note |
|---|---|---|
| ttnn DRAM-sharded (**the 1.29x reference**) | **47.49 µs** | reproduces the historical 47.4 µs exactly |
| ttnn **as the model actually runs it** | **10.50 µs** | 1D mcast, grid 12x3, `in0_block_w=8`, interleaved weight |
| blaze concatenated DSM, W=4 | **38.92 µs** | PCC 0.999956 against the as-model path |

**0.27x. +28.4 µs/layer, +1.34 ms/token predicted** — the same sign and order as the +2.72 ms
measured, with the mesh accounting for the rest (32 chips, each step waits on the slowest; plus
blaze writes DRAM while every downstream consumer of `q_a`/`kv` is L1-resident).

The old reference was 4.5x too expensive because it pays two `to_memory_config` reshards the model
never pays. That reshard overhead is precisely why `DRAM_SHARDED_ATTN` is off.

### Why blaze loses, mechanically — two compounding factors

| | bytes read | time | effective BW |
|---|---|---|---|
| ttnn 1D mcast | 2048x1344 bf8 = **2.93 MB** | 10.50 µs | **279 GB/s** over 36 cores |
| blaze DSM W=4 | 2048x**2048** bf8 = **4.46 MB** | 38.92 µs | **115 GB/s** over 32 workers |

1. **Padding tax, +52% bytes.** At W=4 the DSM requires `N % (banks x W x 32) == 0`, i.e. N%1024.
   1344 pads to 2048. Note this **corrects an earlier claim in this document**: the previous
   section argued the concatenated form "removes the padding penalty" because 1344→2048 beats
   768→1024 plus 576→1024. Those are both 2048. There was never a padding advantage to option 2.
2. ~~**Bandwidth gap, 2.4x.** Even per byte the DSM is 2.4x off ttnn's mcast matmul on this shape.
   Both stream the whole weight from DRAM once, so this is DSM efficiency at K=2048/M=1, not an
   algorithmic difference.~~

~~1.52 x 2.4 = 3.7x. That is the whole regression.~~

> **SUPERSEDED 2026-08-06 — point 2 is FALSE, and so is the 3.7x decomposition that rests on it.**
> Measured by `bench_a_concat_boundary.py` at this exact shape (K=2048, N=1344, N_pad=2048, W=4,
> bf8_b), all five PCC gates passing and additivity closing to 0.6%:
>
> | | bytes read | time | effective BW |
> |---|---|---:|---:|
> | ttnn 1D mcast | 2.925 MB | 10.5 µs | **278.3 GB/s** |
> | blaze **DSM core alone** (L1 in, L1 out) | 4.456 MB | 15.6 µs | **285.0 GB/s** |
> | blaze whole program (D) | 4.456 MB | 39.1 µs | 113.8 GB/s |
>
> **Core-vs-ttnn bandwidth ratio is 1.024 — parity, not 2.4x.** The 115 GB/s figure in the table
> above is the *whole program* divided by weight bytes, and the program is
> `TileRowReplicate → DRAMStreamingMatmul → GatherRowToDRAM`. Dividing a three-stage program by one
> stage's bytes measures boundary cost, not matmul efficiency. The correct decomposition of the
> 38.92 µs is **core 15.6 + input boundary 17.5 + output boundary 6.2 = 39.1**, i.e. **60.1%
> boundary**. Point 1 (the padding tax) stands and is now known to be the *entire* residual: see the
> W sweep at the end of this document, where W=6 pads 1.14x instead of 1.52x and the core drops to
> **9.4 µs, below ttnn's 10.5**.

### What this means for the ceiling

With the true 10.50 µs, `q_kv_a` is **1.5% of a 707 µs layer**, not 7.3%. Restating the ceiling:

- Deleting `q_kv_a` **entirely**: 0.49 ms/token (1.4%).
- Making it 1.29x faster: **0.11 ms/token (0.3%)** — below the ±0.2 ms per-step spread.

**This is why four previous attempts returned exactly 0.0 ms.** They were not defeated by
integration overhead absorbing a win. There was no measurable win available at this call site, and
there never was. The remaining ~697 µs/layer is elsewhere: MoE experts, collectives, SDPA, norms.

### Known limitation (unfixed, by instruction)

**The gather's row-0-only write is batch=1 only.** `GatherRowToDRAM` writes one row per destination
tile because that is all bs=1 decode needs. A batched destination requires the per-row loop
restored, guarded on batch. Every number above is bs=1. This would be a silent wrong answer at
batch > 1, not a crash. It is moot while the flag stays off, but it must be fixed before the flag
is ever turned on.

### State of the code

Implemented in `tt/blaze_ops.py`, `tt/layer_weights.py`, `tt/attention_decode.py`; synced into
tt-blaze (that directory only; F3/F11 uncommitted work untouched, `tile_row_replicate/` verified by
`cmp`). Default off. Worth keeping as a working, gated, correct integration — the plumbing
(load-time weight prep, shared L1 scratch arena, C++-safe named-arg prefixes, program cache keyed
on weight id + activation address, trace-safe) is the reusable part and it all works.

**If blaze is picked up again on this model, benchmark against `ttnn.linear` with
`compute_1d_prog_cfg` and an interleaved DRAM weight — not against `get_dram_sharded_matmul_config`.**

### Trust

- **Trust:** the four e2e runs (same session, back-to-back, tight spread, arena control run
  isolating allocation from execution); the PCC gate (94 layer-instances, in-model, against the
  real op); the 47.49 µs reproduction of the historical reference, which validates the harness that
  also produced the 10.50 µs.
- **Do not trust:** the +1.34 ms single-device prediction as a precise account of the +2.72 ms
  measured — half is attributed to mesh and output-placement effects that were reasoned about, not
  measured. The direction and magnitude of the regression do not depend on that attribution.
- **Not measured:** whether an L1 output or an unpadded-N DSM variant would close the gap. It
  would have to be 3.7x to break even, and the ceiling if it succeeded is 0.11 ms (0.3%).

> **2026-08-06 update to this last bullet.** It is now measured, and it closes. The "3.7x to break
> even" figure was `1.52 (padding) × 2.4 (bandwidth)`; the 2.4x does not exist (see the SUPERSEDED
> block above), so the real gap was only the 1.52x padding. Benchmark B removes that too: at W=6 the
> pad is 1.14x and the core runs at **9.4 µs against ttnn's 10.5**. What remains is the input
> boundary, not the arithmetic. The 0.11 ms / 0.3% ceiling on *this call site* is unchanged — it is
> set by the call site being 1.5% of a layer, and no kernel result moves it.

---

# 2026-08-06 — benchmarks A and B: the concatenated-shape profile, and the W sweep

Both run on one BH device with `bench_guard.preflight()` (profiler env before the ttnn import,
`TT_METAL_CACHE` redirected to `/dev/shm`). Every PCC gate passed at ≥ 0.9998 against a torch
golden, so no timing here is a fast-wrong-kernel. Logs: `/dev/shm/bench_a.log`, `/dev/shm/bench_b.log`.

## Benchmark A — the shipped program, decomposed at the shape it actually runs

`bench_a_concat_boundary.py`, K=2048 N=1344 N_pad=2048 banks=8 **W=4** bf8_b. Four variants of one
program, each with one boundary independently removable, plus the ttnn path the model runs.

| variant | activation | output | µs |
|---|---|---|---:|
| A | native (1,32) L1 replicated | L1 width-sharded | **15.6** |
| B | model (32,32) TILE DRAM | L1 width-sharded | 33.2 |
| C | native | DRAM TILE | 21.8 |
| D | model | DRAM TILE — **the shipped program** | **39.1** |
| — | ttnn 1D mcast, L1 in/out — **what the model runs** | | **10.5** |

    input  boundary (TileRowReplicate)  17.5 µs   (B−A)
    output boundary (GatherRowToDRAM)    6.2 µs   (C−A)
    core                                15.6 µs   (A)
    total                               39.1 µs   (D)
    boundary share                      60.1%
    additivity |B+C−A−D|/D               0.6%

- **D = 39.1 µs reproduces the recorded 38.92 µs**, so this is the same configuration the retracted
  2.4x was computed from and it resolves that question rather than a different one.
- **Core bandwidth 285.0 GB/s against ttnn's 278.3 — ratio 1.024.** Verdict as the bench defines it:
  **KILLS "blaze arithmetic is slow"**. Boundaries are the majority of the program and stage fusion
  is attacking the real cost.
- **Parity is per byte, not per unit time.** Blaze reads 1.52x the bytes (N_pad 2048 vs N 1344), so
  the core alone at 15.6 µs is still **1.49x ttnn's entire op**. On useful bytes only, blaze is
  187.1 GB/s — 0.67x. At W=4, padding is the entire remaining residual.

## Benchmark B — sweep W, core only, both shapes

`bench_b_worker_sweep.py`, `subblock_k=2` fixed so W is the only variable, W set per case by the
script. All 14 points gated and passing. Drift control (`q_kv_a` W=4 re-measured last): **15.6 then
15.7 µs, 0.2%** — well inside the 5% contamination threshold.

### `q_kv_a` — K=2048, N=1344 (N_t=42). Padding is non-monotonic in W.

| W | cores | N_pad | pad | per_core_N | MB read | core µs | GB/s | eff (time) | eff (per byte) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1536 | 1.14x | 6 | 3.34 | 36.0 | 93 | 1.00 | 1.00 |
| 2 | 16 | 1536 | 1.14x | 3 | 3.34 | 22.1 | 151 | 0.81 | 0.81 |
| 3 | 24 | 1536 | 1.14x | 2 | 3.34 | 15.4 | 217 | 0.78 | 0.78 |
| 4 | 32 | 2048 | 1.52x | 2 | 4.46 | 15.6 | 285 | 0.58 | 0.77 |
| 5 | 40 | 2560 | 1.90x | 2 | 5.57 | 16.2 | 344 | 0.44 | 0.74 |
| **6** | **48** | **1536** | **1.14x** | **1** | **3.34** | **9.4** | **356** | **0.64** | **0.64** |
| 7 | 56 | 1792 | 1.33x | 1 | 3.90 | 10.4 | 373 | 0.49 | 0.57 |

**`t(W=6) / t(W=4) = 0.602`. Verdict: CONFIRMS "pick W to fit N".** The rule fires at ≤ 0.85 and
this clears it with room. W=6 is a call-site parameter change with zero source changes:
`_workers_per_bank()` is `max(1, int(env))` (`common.py:34-37`), the `vc = bank_id & 0x3` masking is
over 8 banks and independent of W (`common.py:429-430`), and the call site already derives its pad
multiple from `workers_per_bank()` (`tt/blaze_ops.py:237-241`).

**The headline is not the ratio, it is the absolute number: 9.4 µs against ttnn's 10.5 µs.** The
blaze DSM core is now **0.90x ttnn's entire op** in wall time on this shape — and on useful bytes
312 GB/s against 278.3, **1.12x**. Blaze wins on both denominators at W=6. Benchmark A left padding
as the one remaining residual and this removes it.

**Read the scaling-efficiency column with the byte counts attached.** The script's
`t(W=1)/(W·t(W))` mixes two effects on this shape, because N_pad is not constant across W. The
per-byte column (`bw(W)/(W·bw(1))`) separates them, and it declines smoothly — 1.00, 0.81, 0.78,
0.77, 0.74, 0.64, 0.57. The W=4 point's apparently poor 0.58 is mostly its 1.52x pad, not poor
worker scaling. Sub-linear scaling is real and is the binding constraint above W≈4, but it costs
~36% by W=6 and the padding saving more than pays for it here.

Also worth noting: **W=3 gives 15.4 µs on 24 cores**, matching W=4's 15.6 on 32 while reading 25%
fewer bytes. If core budget ever matters — a stage-fusion program co-resident with other work — W=3
buys W=4's time for three quarters of the cores.

### `o_proj` — K=5120, N=2048 (N_t=64). The control: divides exactly at W=1/2/4/8.

| W | cores | N_pad | pad | MB read | core µs | GB/s | eff (per byte) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 2048 | 1.00x | 11.14 | 118.9 | 94 | 1.00 |
| 2 | 16 | 2048 | 1.00x | 11.14 | 72.2 | 154 | 0.82 |
| 3 | 24 | 2304 | 1.12x | 12.53 | 54.8 | 229 | 0.81 |
| **4** | **32** | **2048** | **1.00x** | **11.14** | **37.2** | **300** | **0.80** |
| 5 | 40 | 2560 | 1.25x | 13.93 | 38.4 | 362 | 0.77 |
| 6 | 48 | 3072 | 1.50x | 16.71 | 44.9 | 372 | 0.66 |
| 7 | 56 | 3584 | 1.75x | 19.50 | 49.4 | 395 | 0.60 |

**Non-power-of-two W costs nothing intrinsic.** This is the point of the control: N=2048 divides
exactly at W=1/2/4, so the odd-W points differ only in how much they pad. Per byte the odd W values
sit exactly on the smooth declining curve their neighbours define — W=3's 0.81 between W=2's 0.82
and W=4's 0.80, W=5's 0.77 between W=4's 0.80 and W=6's 0.66. There is no parity penalty. The
non-pinned workers and the one-shot VC dedup loop (`common.py:431-434`) cost a smooth ~5-6
percentage points of efficiency per added worker above W=2, and they cost it at even W too.

W=4 is fastest here purely because it is the largest W that pads 1.00x. **The rule that comes out of
both shapes is the same one: pick the W that minimises `n_pad_tiles = pad_up(N_t, banks·W)`, subject
to the efficiency decay above W≈6.** That is `q_kv_a` → W=6 and `o_proj` → W=4, which is exactly
what the sweep found independently.

## What to do with this

1. **Adopt W=6 at the `q_kv_a` call site** if the op is ever enabled. It is free — no source change.
   Note that `BLAZE_DSM_WORKERS_PER_BANK` is read from `os.environ` at emit time and is
   process-global (`UPSTREAM_BUGS.md` §J), so a per-call-site W means the two call sites cannot both
   have their optimum in one process until that is fixed.
2. **Land the §J4 asserts in `upstream/08-dram-streaming-matmul-workers-per-bank.patch` itself**, in
   the same change. `common.py:301` is
   `per_core_N = (weights_shard[1] // weights_tile_shape[1]) // _workers_per_bank()` and *neither*
   division is checked; a W that does not divide silently gives workers overlapping or out-of-shard
   ranges — a finite wrong answer, not a fault. This sweep only exercises W values where both hold,
   so it cannot demonstrate the defect, but it is what turns W into a tuning knob with a silent
   failure mode one value away.
3. **None of this changes the e2e verdict for `q_kv_a`.** The call site is 1.5% of a layer; deleting
   it entirely is 0.49 ms/token. The flag stays off. What changed is *why* blaze lost, and that
   matters for what gets built next: not "the arithmetic is slow" but "the boundaries are expensive
   and the shape constraint was mis-tuned", both of which are addressable.

## Trust

- **Trust:** the core-only numbers and the W ordering. Fourteen gated points, one process, a 0.2%
  drift control, PCC 0.9999+ throughout, and two shapes whose padding patterns differ in a way that
  predicts the result independently.
- **Trust with the denominator stated:** the 1.024 bandwidth parity and the 9.4-vs-10.5 comparison.
  Both are like-for-like on boundary conditions — variant A and ttnn are both L1-in, L1-out — but
  blaze's L1 input must already be in the native (1,32) replicated layout, which is what the 17.5 µs
  `TileRowReplicate` produces. **9.4 µs is the arithmetic, not a drop-in replacement for ttnn's
  10.5.** It is the right number for bounding what stage fusion can deliver and the wrong number for
  claiming blaze beats ttnn today.
- **Do not trust:** any projection of the *full program* at W=6. The 17.5 µs input and 6.2 µs output
  boundaries were measured at W=4 only; both are plausibly W-dependent and neither was re-measured.
  The shipped program at W=6 is not known.
- **Do not trust the historical 48.4 / 30.1 / 16.1 trend** for anything. The in-run controls are
  36.0 / 22.1 / 15.6 at the concatenated shape. W=4 agrees closely (15.6 vs 16.1) but W=1 and W=2
  are 26% and 27% apart, because the historical points were two N_pad=1024 programs on a different
  op before patch 09.
- **Not measured:** W=8 and above (`o_proj` would divide exactly at W=8; both shapes' efficiency was
  still falling at W=7), and whether the efficiency decay is NOC hop distance, VC collision, or
  dispatch. The sweep says the decay is real and smooth; it does not say what causes it.

---

## 2026-08-07 — real two-projection Q stage integrated; hardware gate blocked by device state

The model now has a stage-level Blaze path behind
`GLM4_MOE_LITE_BLAZE_Q_STAGE=1`:

```
attention_decode.q_projection
  -> TileRowReplicate(x)
  -> DRAMStreamingMatmul(w_q_a)
  -> GatherRowToDRAM(write_to_dram=False)
  -> Mcast
  -> GLMQANormQBProjection(deferred RMSNorm + w_q_b)
  -> GatherRowToDRAM(model-visible q)
```

The q_a intermediate never leaves L1. When the flag is unset, the original ttnn
q_a matmul -> RMSNorm -> q_b matmul path is unchanged. KV switches from the
concatenated q_kv_a projection to its existing kv_a-only fallback while this
stage owns q_a, so no duplicate q_a projection is executed.

Files: `tt/blaze_ops.py`, `tt/layer_weights.py`, `tt/attention_decode.py`,
synced to the same paths in tt-blaze's embedded tt-metal tree.

Host validation:

- all six edited Python modules compile (both trees);
- importing `blaze_ops`, `attention_decode`, and `layer_weights` in the
  tt-blaze environment prints `IMPORT_OK`;
- `test_glm_qa_norm_qb_derivation.py`: **23 passed**.

Hardware attempts:

- Galaxy health control before the first attempt:
  `python -m pytest glm47_all_shapes_check.py -q` -> **6 passed in 10.96 s**.
- The full 47-layer in-model PCC run reached
  `BLAZE_Q_STAGE first call` and allocated the shared **65,728 B/core** scratch
  arena, then timed out during first-time JIT compilation before a PCC sample.
- A retry timed out during model construction.
- A one-layer retry then failed before model creation while opening the mesh:
  `Device 0: Timeout ... waiting for physical cores to finish` /
  `Device 0 init: failed to initialize FW! Try resetting the board.`

No numerical failure has been observed, but no PCC or E2E timing can be claimed.
The user explicitly prohibited another reset, so baseline/Blaze E2E was not run
against the degraded device.

### 2026-08-07 startup diagnosis and bounded fused-call probe

The 5,406-second retry that exited 124 was not sufficient evidence of JIT
compilation: its last output was model-shell creation, before layer weights or
the Blaze call. Bounded baseline and instrumented Blaze construction now rule
out a constructor deadlock:

- baseline, one layer, `--max-new-tokens 1`: layer preload **0.3 s**, prefill
  **0.191 s**, normal exit;
- Blaze, same run: layer preload **2.1 s**, prefill **0.190 s**, normal exit;
- the four Blaze allocations took `w_q_a=0.025 s`, `w_q_b=0.064 s`,
  `q_a_staging_shape=0.013 s`, and `out=0.015 s`;
- the remaining approximately **1.7 s** is the one-time `tt-blaze` import,
  not a weight-construction hang.

A separate strict 60-second run with `--max-new-tokens 2` reached
`BLAZE_Q_STAGE first call` and the 65,728 B/core scratch allocation. A thread
snapshot after 21 seconds showed one runnable host worker at 63.5% CPU and
dozens of active workers at 2-12%, while device/service threads were sleeping;
the run exited 124 without PCC. This localizes that timeout after construction,
inside first-call host kernel compilation.

Targeted diagnostics added:

- `GLM4_MOE_LITE_BLAZE_STARTUP_PROFILE=1` prints per-allocation constructor
  timing in both tt-metal trees.

A temporary no-LTO/compute-opt override experiment was built, but it was
reverted after the Galaxy failed before model creation; it produced no usable
PCC evidence and is not part of the retained change.

The Galaxy subsequently failed firmware initialization before model creation,
including after `tt-smi -r`. `tt-smi -glx_reset` reported a sudo reset error
but re-enumerated 32 boards; the next bounded run still failed at device 0
firmware initialization. Therefore the O0 diagnostic did not reach model
construction, the in-model PCC gate is still unmeasured, and identical
baseline/Blaze E2E runs remain blocked. The earlier timeout is a startup/fused
first-call diagnosis; the **current** blocker is the post-timeout device state.

## MESH: it runs on all 32 Galaxy chips, and the win holds — 1.39x

Every prior number in this document was taken on `ttnn.open_device(device_id=0)` — **one chip**,
while the model runs on a 4x8 mesh of 32. That gap was never tested until now.
`blaze_eval/mesh_qkv_a_ab.py` closes it.

| | per chip |
|---|---:|
| blaze q_kv_a (W=4) | **33.7 µs** |
| ttnn q_kv_a | 47.0 µs |
| | **1.39x** |

**PCC 0.999881 on all 32 shards**, checked per device rather than on device 0 — a program that
only landed on one chip would otherwise pass a device-0 check while leaving 31 wrong.

Two things that cost time getting here, both worth knowing:

1. **`ab_harness._measure` sums kernel durations across every device in the mesh** (its loop is
   over device ids). The raw figure on 32 chips is 32x a per-chip cost — 1078 µs, not 1078 µs of
   latency. Divide by `get_num_devices()` or the number is meaningless.
2. **Dots in a Blaze prefix break on mesh dispatch.** `prefix="qkv_a.mm"` produces the per-core
   named arg `mm.bank_id`, and `program.cpp:427` requires valid C++ identifiers:
   `TT_FATAL: Named arg field: 'mm.bank_id' is not a valid C++ identifier`. Use underscores.
   This does not surface on the single-device path.

Mesh setup follows blaze's own GLM-5 harness: open the 4x8 parent, then
`parent.create_submesh(ttnn.MeshShape(r, c))`, and close submeshes before the parent.

### On "each stage on a specific number of chips"

`create_submesh` is exactly that mechanism, and blaze uses it. But note what our model actually
does: GLM-4.7-Flash here is **replicated across all 32 chips** — every chip runs every layer on
its own batch/data shard — not pipelined with different layers on different chips. So for us
"specific number of chips" is currently *all 32 for every stage*, and the part of the pattern that
applies is the other half: **one fused kernel per stage, feeding the next**.

blaze's multi-host `PipelineBlock`/`model_pipeline` machinery (D2D sockets, `PipelineLayout`,
`submesh_partition`) is for splitting a model across *processes/galaxies*, which is a different
axis from what this model needs. Do not reach for it to get a fused stage chain on one Galaxy.

## E2E MEASURED: integration is correct, and it makes the model SLOWER

First actual end-to-end run of a Blaze op inside the model. Same script, same flags, back to back
in the blaze tree, 32 chips, traced, bs=1:

| | ms/token |
|---|---:|
| baseline (no blaze) | **34.6** (min 34.4, max 34.9) |
| `GLM4_MOE_LITE_BLAZE_QKV_A=1`, `BLAZE_DSM_WORKERS_PER_BANK=4` | **36.8** (min 36.4, max 37.1) |

**+2.2 ms/token, a 6.4% regression** — from an op measured at **1.39x faster** in isolation on the
same 32-chip mesh. The op definitely executed (`BLAZE_QKV_A first call site hit` in the log).

**The integration itself is correct:** `GENERATED_IDS` are **bit-identical** to baseline. So this is
not a correctness failure or a plumbing failure — the fused op is a faithful drop-in that is
simply slower in the traced model than the ttnn op it replaces, despite winning on kernel time.

Trace capture went 7.0 s -> 43.3 s (building 47 Blaze programs), but that is one-time and not part
of the per-token number.

### This is the third time a cluster win has failed to translate

- removing 23% of all ops under trace: **0.0 ms** (`18479ed4ad0`)
- `DS_CORE_CAP=8` giving ttnn blaze's core layout: 33.2 -> **33.4 ms**
- a genuine 1.39x kernel-time win on the real mesh: **+6.4% e2e**

The mechanism recorded in `18479ed4ad0` explains the direction: under trace replay the command
stream is pre-recorded and per-op firmware setup is *pipelined*, so the profiler serialises what
trace overlaps. Kernel-time wins measured by the profiler therefore do not map to step time, and a
`ttnn.generic_op` dispatch that trace cannot overlap the way it overlaps native ttnn ops can cost
more than the kernel saves.

**Conclusion: do not enable this flag.** The op boundary is not the right unit of optimisation for
this model under trace. Any future Blaze work here must be gated on a traced e2e measurement, not
an op-boundary A/B — the two disagree in sign, not just magnitude.

### What would be worth measuring instead

The step is weight-bandwidth-bound and dense weights are **replicated** across all 32 chips
(`ReplicateTensorToMesh` in layer_weights.py), so at bs=1 every chip reads the full dense weight
set to compute the same result. `GLM4_MOE_LITE_TP` — which would shard those weights and cut
per-chip bytes on the serial path — is in `perf_defaults.PINNED_OFF` for a *correctness*
regression, not a performance one. That is a far larger lever than op fusion and is a bug to fix
rather than a kernel to write.

## STAGE FUSION: the thesis is right, the building blocks are not there yet

Measured the two-matmul Q stage (`bench_e_fused_stage.py`) standalone, across worker counts:

| W | cores | fused stage | intra-stage L1 boundary | ttnn same 2 matmuls |
|---|---:|---:|---:|---:|
| 1 | 8 | 63.4 µs | — | 18.8 |
| 2 | 16 | 39.6 µs | **0.5 µs** | 18.8 |
| 4 | 32 | **31.9 µs** | **1.0 µs** | 18.8 |

Two things are now established:

1. **Bigger stages are the right direction.** Chaining ops inside one program costs **0.5-1.0 µs**.
   Crossing back to the model's tensor boundary costs **~40 µs**. Fusing the two matmuls took the
   unfused pair from 92.3 -> 63.4 µs at W=1 (1.46x), purely by deleting one boundary crossing.
2. **The widening compounds with it:** 63.4 -> 31.9 µs (2.0x) from W=1 to W=4.

But the stage is still **1.7x slower than ttnn's 18.8 µs** for the same two matmuls, and W=4 makes
it worse in one respect: N pads 1536 -> 2048, 33% work ttnn never does. Note this bench's ttnn
reference is much faster relative to blaze than the q_kv_a one was (where blaze won 1.39x), so the
two ttnn references are not configured alike — do not average them.

### The Q stage HANGS in the model, but not standalone

`GLM4_MOE_LITE_BLAZE_Q_STAGE=1` deadlocks the device during the first decode token, at **both W=1
and W=4** — so the widening is not the cause. Evidence:

- all 47 layers prepare (`BLAZE_Q_STAGE prep` x47) and the stage is invoked once
- JIT compile of `glm_q_stage_*.cpp` completes, including the `GatherRowToDRAM` instantiation
- then host CPU time stays at **0 s** for 19+ minutes with the log frozen -> host blocked on the
  device, i.e. a device-side deadlock, not a slow kernel and not compile latency
- the same stage runs clean standalone in 13 s

So it is **model-context specific**: trace capture, 47 sequential layer programs, the shared
`_PROGRAM_SEMAPHORES` dict, or interaction with SDPA's static L1 region. `blaze_ops.py` already
documents two related fragilities (shared semaphores across layers; the shared-scratch arena
wedging the *second* `_build_q_stage_program`).

**Next diagnostic:** `BLAZE_DEBUG_KERNELS=1` names the stalling phase. Do NOT kill the process
before triaging — `TT_METAL_INSPECTOR=1` writes `generated/inspector/*.yaml`, but
`triage.py --run=dump_callstacks` wants the **live RPC** at localhost:50051 and reports
"Inspector unavailable" from serialized logs alone.

### Recovery needs TWO resets

After this hang a single `tt-smi -r` left the control plane broken:
`Physical chip id 0 not found in control plane chip mapping`, and the control failed 6/6. A second
`tt-smi -r` restored it (6 passed, 11 s). Do not conclude the machine is broken after one reset.

### Q-stage deadlock: TRIAGED, exact stall site

Caught the hang live (poll for `log frozen` + `pcpu==0`, do **not** kill) and ran
`triage.py --run=dump_callstacks`. It works against the **live process**; from serialized yaml
alone it reports "Inspector unavailable", which is why the first attempt failed.

Stall site — core **(0,0)**, trisc0, program `glm_q_stage_*`:

```
llk_wait_tiles  <-  ckernel::cb_wait_front  (cb_api.h:44)
  <- blaze::DRAMStreamingMatmul::Op<ct_args::glm_q_stage__norm_q_b__q_b_proj>::operator()
     at blaze/ops/dram_streaming_matmul/kernels/op.hpp:329
  <- kernel_main at generated/kernels/glm_q_stage_*_debug.cpp:111
```

Other workers sit in `dram_stream::accumulate_output_tile` (op.hpp:390, 413) — i.e. downstream of
the same starved CB.

**The second matmul of the stage (`q_b_proj`) waits forever for its activation tiles.** Those
tiles arrive by Mcast from the norm stage; the bench reports `gather_receiver=mcast_sender=(11,9)`.
So the mcast never delivers in the model context, while the identical chain completes standalone
in 13 s.

**Prime suspect: `_PROGRAM_SEMAPHORES` in `tt/blaze_ops.py`, shared across all 47 layer programs
on purpose.** Mcast signals completion through a named semaphore. One semaphore object shared by
47 programs, captured into a trace, is exactly the shape that leaves a stale or mis-signalled
count so the receiver waits forever. The module comment says a fresh dict per program "allocates
one mesh-global L1 semaphore per layer and eventually collides with SDPA's static CB region" —
so both options have a failure mode and the fix has to reset the semaphore per dispatch rather
than simply un-sharing it.

Next: confirm by giving one layer its own semaphore dict (single-layer run), or by asserting the
semaphore's value at stage entry. `BLAZE_DEBUG_KERNELS=1` alone produced nothing useful — the
device-side prints need DPRINT cores configured, so triage is the tool, not that flag.

### Semaphore hypothesis: TESTED AND WRONG. The suspect is Mcast-under-trace.

Gave the Q stage its own semaphore dict (`GLM4_MOE_LITE_BLAZE_Q_STAGE_OWN_SEM=1`, wired in
`_build_q_stage_program`). **It still hangs**, in the same place. So sharing
`_PROGRAM_SEMAPHORES` across 47 layers is not the cause, and that line of investigation is closed.

Second triage, and the scale is the new information:

| stalled | where |
|---:|---|
| **256 cores** | `DRAMStreamingMatmul::Op<glm_q_stage__norm_q_b__q_b_proj>`, `cb_wait_front`/`llk_wait_tiles` at op.hpp:329, plus 390/413/246 |
| 64 cores | `GatherRowToDRAM::Op<glm_q_stage__output>` |

256 = 8 bank workers x 32 chips — i.e. **every worker on every device**, uniformly. So this is
structural in the stage, not a per-device or fabric problem.

**The discriminating fact: the mcast-free `qkv_a` path runs fine under trace (it produced the
36.8 ms/token number); the Q stage differs from it by having an `Mcast`.** And the Q stage runs
clean *standalone without trace* in 13 s. That points at **Mcast inside a traced program**: the
mcast rendezvous is a semaphore handshake, and trace capture/replay does not reproduce it, so
`q_b_proj` waits on tiles the sender never pushes.

**Next test, cheap and decisive:** run the Q stage with `--enable-trace` **omitted**. If it
completes, Mcast-under-trace is confirmed and the fix is either to keep mcast out of traced
programs or to make the handshake trace-safe. If it still hangs, the stage is broken in the
multi-layer model context for some other reason and the next lever is a single-layer run.

Recovery reminder: after every one of these hangs, TWO `tt-smi -r` cycles were needed; control
then passes 6/6 in ~11 s.

### Mcast-under-trace: ALSO WRONG. The untested variable is Mcast on a MESH.

Ran the Q stage with `--enable-trace` omitted. **It still hangs**, same place, ~150 s in. So trace
is not the cause either. Two hypotheses tested and both disproved:

| hypothesis | test | result |
|---|---|---|
| shared `_PROGRAM_SEMAPHORES` across 47 layers | `..._OWN_SEM=1` | still hangs |
| Mcast rendezvous broken by trace capture | no `--enable-trace` | still hangs |

What is actually different between "works" and "hangs" is narrower than I had been assuming:

- `bench_e_fused_stage.py` — the standalone run that completes in 13 s — opens
  **`ttnn.open_device(device_id=0)`: ONE chip.**
- the model runs on a **4x8 mesh of 32**.
- `mesh_qkv_a_ab.py` proved the **mcast-free** chain (replicate -> DSM -> gather) on 32 chips.
  **The Q stage's `Mcast` has never been run on a mesh at all.**

So the leading hypothesis is now **Mcast on a 32-chip mesh**, not trace, not semaphore sharing.
It is consistent with every observation: single-chip fine, mesh hangs, mcast-free mesh path fine,
and all 256 workers (8 x 32) starved uniformly at the consumer of the mcast.

**Next test — do NOT run the 10-minute model again for this.** Take
`bench_e_fused_stage.py` and open a mesh instead of `open_device(0)`, the same way
`mesh_qkv_a_ab.py` does (`open_mesh_device(MeshShape(4,8))` + `create_submesh`, `ReplicateTensorToMesh`
on every tensor, underscore-only prefixes). That reproduces or clears the hang in ~1 minute rather
than ~10, and if it hangs it is a small self-contained repro to hand to the blaze owners.
