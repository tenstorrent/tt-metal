# Resume point — GLM blaze integration

Written at a context checkpoint. Everything below is measured on the 32-chip BH Galaxy unless
marked otherwise. Full context: [`../BLAZE_EVALUATION.md`](../BLAZE_EVALUATION.md).

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
