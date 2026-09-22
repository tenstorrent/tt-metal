# Gemma4 prefill: why chunk 2k/4k lose to 8k, and what we fixed

**Measured on** `mmanzoor/svuckovic/gemma4-L1-activations` (Asif's branch, PR #56862),
latest commit **`9dc8e32a2`**, with your multi-hop sliding-halo C++ rebased on top so the
narrow chunk widths are legal at all. Mesh 8x4 (CP8/TP4), bh-glx-120-b03u02, ctx 262,144,
`GEMMA4_PREFILL_L1_ACT=1`. Runs, logs and all per-op tables: `/data/kmabee/gemma4_runs_sept20/`

## TL;DR

- **Two separate problems, not one.** (a) A per-chunk cost that does not shrink when the
  chunk shrinks, so you pay it more often. (b) At chunk 2048 *only*, the global SDPA leaves
  71% of the core grid idle.
- On Asif's branch with the flag on, **~100 ms of every chunk is chunk-invariant** — 79% of a
  2048 chunk but only 49% of an 8192 chunk. That is most of the gap on its own.
- We landed two fixes on top: **−12.0% / −7.1% / −6.2%** on first-chunk time at 2k/4k/8k and
  **−7.0% / −4.5% / −3.1%** on a full 256k prefill.
- Nearly all of it is one change: **block-sharding the RMSNorm**. The MLP-matmul root cause we
  found independently turned out to be the same thing Asif's `core_grid` commit already fixes,
  so we dropped ours.

## The model

Per-chunk time is `t_i = a + slope·i`, so a whole prefill is `T = N·a + slope·N(N−1)/2`
with `N = ISL/chunk`.

- **`a`** = first-chunk time = the floor every chunk pays regardless of depth
- **`slope`** = extra cost per additional chunk of history

Fits are R² ≥ 0.9999 at every width, so this is a description of the behaviour, not a
curve-fit. A cheap ctx-32k run predicts the 256k total to ≤1.2%.

## Baseline on Asif's branch, flag on, direct ctx-256k

| chunk | `a` (ms) | `slope` | 256k total |
|---|---|---|---|
| 2048 | 126.2 | 1.453 | 27.96 s |
| 4096 | 163.7 | 2.978 | 16.48 s |
| 8192 | 205.4 | 12.168 | 12.61 s |

### Problem (a) — the floor

Solving `a(C) = F + k·C` gives **F ≈ 100 ms that does not change with chunk size.** Halve the
chunk and only the `k·C` part halves. So `a(2048)` is 126 ms, not `205/4 = 51` — and you pay
it **128 times instead of 32**.

### Problem (b) — the prefix term, at chunk 2048 only

Total attention-over-history work is chunk-size-independent, so `slope` *should* scale as C².

| chunk | `slope` ratio to 8192 | C² would give | off by |
|---|---|---|---|
| 4096 | 0.2447 | 0.2500 | **0.98x — fine** |
| 2048 | 0.1194 | 0.0625 | **1.91x** |

Single-layer captures say why. Work in the global SDPA is split as
`units = 8·ceil((chunk/CP)/q_chunk)` with `q_chunk=64` on a ~110-core grid, and **idle cores
are not skipped** — they run padded handshake iterations, so time tracks *grid passes*:

| chunk | work units | grid passes | useful |
|---|---|---|---|
| 2048 | 32 | 1 | **29%** |
| 4096 | 64 | 1 | 58% |
| 8192 | 128 | 2 | 58% |

That predicts, at matched prior context, growth ratios of **0.500 / 0.500 / 1.000**
(occupancy) rather than 0.250 / 0.500 / 0.500 (work-proportional). Measured:
**0.490 / 0.501 / 0.978.** So chunk 2048 and 4096 cost the *same* for prefix work — halving
4096 → 2048 halves the work and buys nothing. The sliding layer was flat across all of it,
which is the control that makes the global number credible.

⚠️ **The `Cores` column cannot show this.** The global SDPA reports 114 cores at every chunk
width. It has to come from the work-unit math in `ring_joint_sdpa_program_factory.cpp`.

## What we changed

| | what | result | status |
|---|---|---|---|
| ~~Exp 1~~ | explicit matmul program config on the MLP projections | **same root cause Asif's `core_grid` commit fixes** — his is cleaner (one kwarg vs a hand-built config with an M-dependent L1 gate) | **dropped** |
| Exp 2 | `core_grid` on the 4 attention projections, which Asif's branch does not touch | **+1.5 / +2.0 / −4.1 ms** at 2k/4k/8k — helps only at 8192, small regression at 2k/4k | committed, **gated off** |
| **Exp 4** | **block-shard the prefill RMSNorm** | **−16.7 / −13.6 / −8.7 ms** | committed |

**Exp 4 detail.** `ttnn.rms_norm`'s default path parallelises over **rows only** — one core per
32 rows — so the 5376-wide prefill norm ran on **8 of 120 cores** at chunk 2048 and is
width-bound. `LayerNormShardedMultiCoreProgramConfig` splits both axes; we get **4.36x on the
op**, at the cost of an interleaved→sharded in and sharded→interleaved out per call.

This is **independent of** Asif's "Keep Gemma4 post-sublayer norm outputs in L1" — that moves
the norm's output *buffer*, ours changes how the op *computes*. Measured: his commit improved
LayerNorm only at chunk 8192 (0.611 → 0.523 ms/layer) and left chunk 2048 at **0.462 ms with a
scaling ratio of 0.884 against an ideal 0.250**. That is what was left on the table.

Both fixes are per-chunk-floor changes: **`slope` moved ≤0.3% at every width**, confirming the
prefix term was untouched. `F` drops **99.8 → 83.8 ms**.

| chunk | `a` before → after | 256k before → after |
|---|---|---|
| 2048 | 126.2 → **111.0** (−12.0%) | 27.96 → **26.00 s** (−7.0%) |
| 4096 | 163.7 → **152.1** (−7.1%) | 16.48 → **15.74 s** (−4.5%) |
| 8192 | 205.4 → **192.6** (−6.2%) | 12.61 → **12.22 s** (−3.1%) |

Also worth noting: Asif's branch reproduces its own claim. Against a pre-branch baseline,
chunk 8192's `a` goes 243.5 → 205.4 = **−38.1 ms**, against the ~38 ms/chunk the PR states.
The `GEMMA4_PREFILL_L1_ACT` flag itself is worth −4.3 / −8.2 / −18.5 ms — note it gates *only*
memory placement; the `core_grid`, the fused GELU and the fused `layer_scalar` are
unconditional, so "flag off" is not a pre-branch baseline.

## Still open

1. **The 2048 prefix penalty is untouched** — that is the SDPA occupancy above, and `q_chunk`
   is **not** the lever: measured, `q=128` is a 1.0% *regression* at chunk 2048 and `q=32` is
   illegal on the sliding path (TT_FATAL). Remaining suspect is the constant 1024-token halo.
2. **Exp 2's sign flip is unexplained.** Possibly Asif's own observation — an L1-interleaved
   `in0` costing 1205 µs vs 301 on the simple program config — playing out at small M. One
   per-op capture pair with the flag on/off would settle it.
3. **The norm shard does not apply at chunk ≥ 16384** — the per-core block exceeds the ~84-tile
   ceiling `dataflow_buffer.cpp` enforces. It falls back to the default path, measured at exact
   parity, so nothing regresses.
4. **~84 ms of chunk-invariant cost remains.** Largest known next item is the two reshards the
   norm fix pays: ~22% of its gross win, removable by feeding the next op a sharded tensor
   instead of round-tripping through DRAM.

## The per-layer views you asked for

**All 12 are already rendered.** Cells are {global, sliding} × {2048, 4096, 8192} ×
{first chunk, at depth}, at **matched prior context 57,344 tokens**:

| chunk | layer | first chunk (idx 0) | at depth (prior ctx 57,344) |
|---|---|---|---|
| 2048 | global | `per_op/c2048_floor_global.txt` | `per_op/c2048_deep_global.txt` (idx 28) |
| 2048 | sliding | `per_op/c2048_floor_local.txt` | `per_op/c2048_deep_local.txt` |
| 4096 | global | `per_op/c4096_floor_global.txt` | `per_op/c4096_deep_global.txt` (idx 14) |
| 4096 | sliding | `per_op/c4096_floor_local.txt` | `per_op/c4096_deep_local.txt` |
| 8192 | global | `per_op/c8192_floor_global.txt` | `per_op/c8192_deep_global.txt` (idx 7) |
| 8192 | sliding | `per_op/c8192_floor_local.txt` | `per_op/c8192_deep_local.txt` |

All 12 are committed here under [`per_op/`](per_op/), and the side-by-side that compares
them across widths is [`PER_OP_COMPARISON.md`](PER_OP_COMPARISON.md).

On the box that produced them: raw captures (ops CSV kept, multi-GB device logs pruned) in
`/data/kmabee/gemma4_runs_sept20/captures/`, matching `.csv` in `.../reports/`, and the
script that rendered them in `.../render.sh`.

### Five things moved since your commands

| yours | now |
|---|---|
| `models/demos/gemma4/` | **`models/demos/gemma4_d_p/`** |
| mesh `4x8` | **`8x4`**, and the node id carries `sz{chunk}` and `ctx_{N}k` |
| `sliding` | **`local`** — both the `layer_type` param and the signpost name |
| `GEMMA4_PERF_CHUNK_IDX=7` | chunk index is a **test parameter**: `chunk7` in the node id |
| one run per layer type | **`layer_type=both`** does both in one session (~4 min saved) |

```bash
# one run gives BOTH layer types
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
  python -m tracy -r -p -v -o cap/profiler -m pytest \
  "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk7-both-sz8192-ctx_256k-8x4]" -sv

CSV=$(ls cap/profiler/reports/*/ops_perf_results_*.csv)
tt-perf-report --start-signpost gemma4-layer-global-chunk7-start \
               --end-signpost   gemma4-layer-global-chunk7-stop  "$CSV"
tt-perf-report --start-signpost gemma4-layer-local-chunk7-start  \
               --end-signpost   gemma4-layer-local-chunk7-stop   "$CSV"
```

`TT_METAL_DEVICE_PROFILER=1` is redundant — `-r -p` sets it.

**Two traps when comparing widths:**

1. **Hold prior context fixed, not chunk index.** Index 7 is 57,344 tokens of history at chunk
   8192 but only 14,336 at 2048. Use index **7 / 14 / 28** for 8k/4k/2k. This is what turns the
   comparison into a test with a predicted number.
2. **A capture is not done when pytest prints `passed`** — Tracy post-processes for ~4 more
   minutes. Poll for `ops_perf_results_*.csv` at `-size +1M`. Each capture writes ~12 GB, of
   which only the ~37 MB ops CSV is needed afterwards.

**On core counts:** they are in the tables and meaningful for LayerNorm (8 / 16 / 32 cores at
chunk 2048 / 4096 / 8192 = `(chunk/CP)/32` exactly — the defect, visible directly) and for the
matmuls (84 vs 96). They are **not** meaningful for the SDPA, which always reads 114.

## Why the `Cores` column is not occupancy — with the evidence

This matters because the `Cores` column is the first thing anyone reads off these tables, and
for the op that dominates the prefix term it is **actively misleading**.

**The SDPA's cost varies 50x while `Cores` never moves.** Same op, all six cells in
[`per_op/`](per_op/):

| cell | Device Time | `Cores` |
|---|---|---|
| `c2048_floor_global` | 0.189 ms | **114** |
| `c4096_floor_global` | 0.356 ms | **114** |
| `c8192_floor_global` | 1.314 ms | **114** |
| `c2048_deep_global` | 4.249 ms | **114** |
| `c4096_deep_global` | 4.410 ms | **114** |
| `c8192_deep_global` | 9.426 ms | **114** |

A column that reads an identical 114 while the op ranges over **0.189 → 9.426 ms** is reporting
the *grid it was given*, not the work it did. Meanwhile the real occupancy nearly doubles across
those widths and the column cannot show it:

| chunk | work units | grid passes | useful |
|---|---|---|---|
| 2048 | 32 | 1 | **29.1%** |
| 4096 | 64 | 1 | 58.2% |
| 8192 | 128 | 2 | 58.2% |

**The source says why explicitly.**
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:834`:

> *"A core with no Q chunks (`global_q_start == global_q_end`) is **NOT dead**: in the GQA /
> shared-K row-wide multicast path it runs padded handshake iterations
> (`loop_q_count = *_max_q_per_core`) so the injector's mcast rectangle never targets a silent
> worker."*

So every core in the rectangle is genuinely dispatched to and genuinely reports in — they just
have no Q chunk to work on. The profiler counts them because they ran. Occupancy has to come
from the factory's own work-unit math (`all_heads_num_q_chunks = B * NH * num_q_chunks`,
`max_q_per_core = div_up(all_heads_num_q_chunks, num_cores)`, same file, line 1306).

**By contrast `Cores` IS informative for LayerNorm** — and this is the useful half of the story:

| cell | Device Time | `Cores` |
|---|---|---|
| `c2048_floor_local` | 0.462 ms | **64,32,8** |
| `c4096_floor_local` | 0.468 ms | **120,64,16** |
| `c8192_floor_local` | 0.523 ms | **120,32** |

The small entry is the hidden-width norm, and it reads **8 / 16 / 32** — exactly
`(chunk/CP)/32`, one core per row-tile. Here the column moves, it matches the row-parallel
decomposition, and the op's time barely moves because so few cores are doing it. That is a
defect you can read straight off the table, and it is the one this branch's norm-sharding
commit fixes (8 → 16 cores at chunk 2048, 4.36x on the op).

**Rule of thumb:** trust `Cores` when it *varies* with the shape in a way the op's
decomposition predicts. Distrust it when it is pinned to the grid size — that means idle cores
are being counted, and you need the op's work-unit math instead.
