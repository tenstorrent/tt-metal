# Work log — stage 03, multichip decoder

Chronological. Every number here is either a cell of a file in this directory or
the output of a script in `probes/`, and the command that produced it is given.
Where something did not work, the measurement that says so is given too.

Baseline: `tt/optimized_decoder.py` at `0ba65b321c1`. Design: `mesh_plan.md`
(written before any implementation, as the contract requires).

---

## 0. What was already decided, and what was not

`mesh_plan.md` chose scheme C — attention TP=4, experts EP=4, router/norms/
residual replicated, ring fabric with two links, two all-reduces per layer — and
backed it with probe measurements. Implementation did not revisit that choice.
What implementation *did* have to settle, because the plan flagged each as an
assumption rather than a measurement:

| plan risk | how it turned out |
|---|---|
| §7.2 top-k determinism across dies | **Holds.** Asserted directly, 8 seeds plus an all-zero input. §2 below. |
| §7.1 `nnz` mismatch is a device hang | **Confirmed the hard way** — a test of mine hung the mesh and needed `tt-smi -r`. §5. |
| §7.1 `E_local = 0` must contribute exact zero | **Holds**, exactly 0.0, not merely finite. §5. |
| §7.3 `_dram_sharded_ok` at N=1280 / K=1024 | Holds, and is now asserted at upload time and by a test. §6. |
| §7.4 the 40 MB L1 budget is inherited from a different problem | Re-derived by sweep: 40 MB -> **128 MB**. §9. |
| §1.1 `paged_update_cache` at 1 KV head unexercised | Works. Covered by every decode test. |
| §7.5 prefill attention at 1 KV head | Works. |
| §7.6 trace + CCL interaction | Works; semaphores are allocated in `MeshContext.__init__`, before capture. |
| §7.7 non-aligned sequence lengths | Preserved. S = 32, 128, 512, 33, 100, 257 all pass. |
| §7.8 watcher cost on 4 dies | **Worse than expected — the watcher does not fit on active ethernet cores at all.** §11. |

Three things the plan did not anticipate at all turned up in implementation and
are recorded in §3, §4 and §7. The plan's decode collective choice also turned
out to be measured on the wrong shape (§7), and its expert-matmul speedup did not
survive contact with the layer (§8).

---

## 1. Attention TP=4 first, validated before anything else

Sequenced first because the wqkv column split is the one weight transform that
fails silently. The checkpoint stores `[Wq(4096) | Wk(512) | Wv(512)]`, so a
plain `ShardTensorToMesh(dim=-1)` gives die 0 nothing but Q heads and die 3
nothing but K and V — and `nlp_create_qkv_heads_decode(num_heads=8,
num_kv_heads=1)` accepts 1280 columns whatever is in them, so there is no shape
error to catch it. `head_interleaved_wqkv` rebuilds the tensor so that die *d*'s
contiguous slice is `[Q heads 8d..8d+7 | K head d | V head d]`.

Measured, before any expert work (`scratchpad/smoke1.log`, prefill S=128, real
weights, against the *replicated* single-chip optimized attention on the same
mesh):

```
baseline replicated spread across 4 dies      0.0
TP=4 partial shape                            [1, 1, 128, 2048]
all-reduce output spread across dies          0.0
PREFILL ATTENTION TP=4 vs single-chip    PCC  0.9999377
```

`wo`'s row split needs no permutation: once the Q head assignment is contiguous
per die, die *d* owns rows `1024d … 1024d+1023`.

`test_wqkv_column_split_is_head_interleaved` checks the transform on the host,
per die, head by head, and additionally checks it is a permutation — as sorted
multisets, not as row sums, because reordering changes float addition order and
a row-sum comparison fails for the wrong reason.

## 2. Top-k determinism, pinned before building on it

The scheme assumes four dies independently reach the same top-8 from
bit-identical replicated logits, so that the four 32-expert windows partition it.
Nothing enforces that; if `ttnn.topk` ever broke a tie differently the layer
would be silently wrong.

Measured (`smoke1.log`, then `test_topk_is_identical_across_dies`): identical
across all four dies for 8 cases -- 7 random inputs of 128 tokens (1024
selections each) **and one all-zero input**, where every logit is the projection of zero and
the top-8 is decided entirely by tie-breaking. That degenerate case is the one an
ordinary random test never reaches.

`test_router_windows_partition_global_routing` then asserts the stronger
end-to-end property: concatenating the four dies' `[1,1,S,32]` windows reproduces
the single-chip router's `[1,1,S,128]` dense output **exactly**, `max |diff| =
0.0`, at S = 1, 33 and 128.

## 3. The window matmul had to be forced to HiFi4 (not in the plan)

A mesh op is SPMD: one program, four dies, so `ttnn.slice` cannot take a
different start offset per die and there is no way to ask for "columns
32d..32d+31". The device-varying constant is built the only way a mesh tensor can
vary by device — a `[4, 1, 128, 32]` one-hot tensor sharded on dim 0 — and
applied as a matmul.

First spelling used the matmul default fidelity and measured
`max |stitched − global| = 9.77e-4`, one bf16 ulp: **LoFi keeps ~5 mantissa bits,
so multiplying by 1.0 is not a copy.** The operand is 0/1 and the intent is
selection, not arithmetic, so the matmul is given a HiFi4 compute config and the
difference goes to exactly 0.0. K = 128 is 4 tiles and N = 32 is 1, so exactness
is free.

The matmul is not pure overhead either: it *replaces* width. The divide that
follows now runs over 32 columns instead of 128.

## 4. SDPA-decode refuses 1 KV head on the contiguous cache (not in the plan)

`test_multichip_decode_vs_single_chip[contiguous]` failed with

```
sdpa_decode_program_factory.cpp:245:
Tree reduction max 6 rounds (64 cores/head), got 110 cores/head
```

This is a failure *created by the head split*. With no program config the op sets
`max_cores_per_head = num_cores_available`, so
`num_cores_per_head = 110 / batch / num_kv_heads`; at the single-chip 4 KV heads
and batch 1 that is 27, and at TP=4's 1 KV head it is 110, past the tree
reduction's 64. It bites only the contiguous path — the paged one runs at the
default — and only at small batch, because the expression divides by batch.

Fix: pass an `SDPAProgramConfig` with `max_cores_per_head_batch=64`, the op's own
limit. `attention_decode_optimized` gained an optional `sdpa_program_config`
parameter that defaults to `None`, so **no single-chip number in stage 02 is
affected**; the multichip layer passes the capped config only when the cache is
not paged, so the paged path — the production one, and the one every decode
number here was measured on — also runs at the stage-02 default.

Recorded as a named TTNN limit in `README.md`.

## 5. Expert parallelism, `nnz=None`, and the hang

### 5.1 The hang, reproduced by accident and then on purpose

`test_expert_window_can_be_empty` builds a router that forces the top-8 into the
low expert range. Its first version scaled *synthetic* rows by 10x. Against the
raw 0.02-scaled hidden that is harmless; inside the layer the router sees the
**rms-normed** activation, which is O(1) per element, so the logits came out with
a standard deviation near 450 and a top-8 spread past 1000. `exp(-1000)` is
exactly zero in bf16, two of the eight routing weights underflowed, and
`count_nonzero(sparsity)` fell below the `nnz = top_k * batch` that the
*single-chip* baseline leg of the test passes.

The board deadlocked, exactly as `sparse_matmul_device_operation.cpp:205-211`
says it will. The process sat at 410% CPU with no output for four minutes, had to
be `kill -9`'d, and the mesh needed `tt-smi -r`. **The multichip leg, on
`nnz=None`, was unaffected.**

That is the plan's §7.1 risk arriving in practice, on the *baseline* path rather
than the new one, and it is the strongest possible argument for the `nnz=None`
choice: the hazard is not hypothetical and it does not announce itself.

`probes/nnz_hazard_probe.py` reproduces it deliberately and in isolation — one
die, no fabric, no CCL, so that a hang costs one board — and under the watcher,
which turns the deadlock into a loud on-device assert. Output in §5.3.

The test itself now uses 4x the *real* router rows, where the top-8 spread is a
few units and every routing weight stays normal.

### 5.2 What the empty window actually does

Measured (`test_expert_window_can_be_empty`): under the forced routing the live
expert counts per die are **[6, 2, 0, 0]**. That is a better case than the
[8,0,0,0] originally aimed for, because it exercises both hazards in one program:
`E_local = 0` on two dies and `0 < E_local < top_k` on another. The four counts
sum to 8, so the windows are still a partition; the two empty dies contribute
`max |value| = 0.0` — exactly zero, not merely finite — and the full layer still
matches the single-chip baseline under this routing.

The assertions are on those properties rather than on the exact split, so the
test does not become brittle against a different checkpoint.

## 6. Runtime fallback audit

Three helpers imported from stage 02 choose a slower path *silently* and all
three see different inputs under TP/EP than they were tuned against, so a PCC
test cannot notice. `fallback_audit()` returns them as data and
`test_no_runtime_fallbacks` asserts on it at batch 1 and batch 32:

| helper | risk under TP/EP | measured |
|---|---|---|
| `_dram_sharded_usable` | per-die wqkv N is 1280, not 5120; `_dram_sharded_ok` needs a multiple of `8 banks x 32 = 256`, and 1280 = 5x256 is one factor of two from failing | taken, `qkv (2048, 1280)`, `wo (1024, 2048)` |
| `_tuned_sparse_matmul_config` | lowers `in0_block_w` to the largest divisor of K in tiles; if EP had touched K this would be scheme A's regression by the back door | 16 and 12, i.e. stage 02's tuned values, unchanged |
| `_decode_expert_memory_config` | EP shrank the intermediates 4x, so the inherited byte budget would change which batches use L1 | L1 at batch 1, DRAM at batch 32 |

`upload_multichip_weights` also asserts `_dram_sharded_ok` on the per-die dims at
upload time, so a future change to the head split fails loudly instead of
quietly returning `None` and dropping the DRAM-sharded projections.

## 7. The plan's decode collective was the wrong choice, and the profile said so

`mesh_plan.md` §5 chose AG-of-partials-plus-local-sum for decode from
`probes/ar_probe.py`: 19.96 µs against RS+AG's 23.69 at `[1,1,32,2048]`. The
first decode profile says that probe measured the wrong shape. The shipped
decode tensor is `[1,1,1,2048]` — **one** logical row padded to a tile — and
`ttnn.sum` over a tensor whose last two dims are not both tile-aligned drags a
`FillPad` behind it (`fill_pad.cpp:17-24`). That is the same hazard stage 02
removed from the router, arriving in a different place.

From `ops_perf_multichip_decode_agsum.csv`, device 0 — a profile of this layer
with the plan's spelling, kept as an artifact precisely because the shipped path
no longer produces those rows (`probes/profile_layer.py decode-agsum`). Same
convention as every other window here: the second and last decode iteration,
rows 138–205 sorted by `HOST START TS`, 68 ops (four more than the shipped
layer's 64 — the two extra ops per all-reduce are the point):

| | AllGather | FillPad | FastReduceNC | Slice | total |
|---|---|---|---|---|---|
| attention all-reduce, rows 160–163 | 22.31 µs | 5.89 | 2.44 | 1.32 | **31.96** |
| expert all-reduce, rows 201–204 | 18.65 | 5.64 | 2.43 | 1.31 | **28.04** |

against 19.96 µs each as promised. `ttnn.sum` is not one op here, it is three.

Measured on the whole traced layer at ctx 128, median of 100
(`probes/allreduce_ab.py`):

| spelling | traced decode | agreement |
|---|---|---|
| AG(dim 0) + `ttnn.sum` — the plan's choice | 0.4801 ms | reference |
| **reduce-scatter + all-gather — adopted** | **0.4760 ms** | max abs diff 4.9e-4 |
| AG + reshape-to-padded + sum, to dodge the `FillPad` | **did not run** | `reshape_common.cpp:50`, `new_volume == old_volume` |

0.9%. Small, and honestly reported as small — but it is also three fewer ops per
all-reduce, one code path instead of two, and it is the direction the standalone
probe got backwards. Both modes now call the same `all_reduce`.

## 8. What dynamic `nnz` actually costs — the plan's biggest miss

The plan predicted the expert matmul pair would go 2.13× faster under EP, from a
sweep whose E=128 baseline read 264.65 µs where the profiled single-chip layer
reads 92.06. It used only ratios from that sweep, which was the right caution,
and the ratio still did not survive: the multichip decode profile reads
**82.65 µs** for the pair (SparseMatmul 42.48 + 40.16) against the single chip's
92.06, i.e. **1.11×, not 2.13×**.

`probes/nnz_cost_probe.py` prices the difference directly, at the shipped shapes
— E=32, M=1, bfloat4_b, LoFi, L1 output, stage 02's tuned block widths:

| | gate_up | down | pair |
|---|---|---|---|
| `nnz=None` — shipped, and the only legal choice under EP | 90.39 µs | 67.62 | **158.01** |
| `nnz=8` exact — illegal here, the count is data-dependent | 68.98 | 38.75 | 107.73 |
| `nnz=2` exact — the mean local load | 24.42 | 18.72 | 43.15 |

**Dynamic mode is 1.47× the exact-`nnz` pair** (158.01 / 107.73 = 1.467) — 50.3 µs over 32 slots and two
matmuls, i.e. 0.79 µs per slot per matmul, which is exactly the rate the design
sweep measured. What the design sweep got wrong was not the rate, it was how much
of the layer that rate would eat once the fracture had made the rest cheap. (The absolute values do not match
the profile's 82.65 µs — this probe holds its input in DRAM and traces the two
matmuls in isolation — so only the ratio is used, exactly as the plan's own
sweeps were used.) Applied to the profiled pair, ~26 µs of the multichip decode
layer is dynamic-`nnz` overhead that a single-chip implementation does not pay:
about 5% of the layer, and it lands directly on top of the 4× the fracture buys.

That is the honest accounting for why decode scales at 1.18× rather than the
plan's 1.61×: EP does fracture the expert matmuls, and then hands part of the
win back at the door.

**Ways out that were considered and rejected:**

* *Capacity padding to `nnz = top_k`.* Needs a sparsity tensor with exactly 8
  non-zeros on every die regardless of how many experts are really live. The
  only way to build one on device is a second `topk`, this time over the local
  32 — and `topk` over a single row is a **26.32 µs** one-core op in this very
  profile, which is more than the ~26 µs it would save. Rejected on its own
  measurement.
* *Dense all-expert decode* (`nnz = 32`). Forbidden by the contract unless
  proven faster, and it is not: the `nnz=8` row above is 107.73 µs for 8 live
  slots, and 32 live slots is strictly more work than that.
* *Computing `nnz` on the host.* This is the deadlock in §5. Not available.

## 9. Re-deriving the L1 budget, and a sweep that contradicted itself

Stage 02's `_DECODE_EXPERT_L1_BUDGET_BYTES = 40 MB` sat between its batch 1
(29.4 MB) and batch 2 (58.8 MB) and its own comment calls it "asserted, not
measured". Under EP the intermediates are a quarter of the size
(`batch × 7.34 MB`), so inheriting 40 MB would have started admitting batch 5 by
accident, on a mesh where L1 also holds the fabric's and the CCL's persistent
buffers.

`probes/l1_budget_probe.py`, eager, ms per decode step:

| batch | intermediates | L1 | DRAM |
|---|---|---|---|
| 1 | 7.34 MB | 1.7336 | **1.6924** |
| 2 | 14.68 | **1.7816** | 1.7930 |
| 4 | 29.36 | **1.8887** | 1.9753 |
| 8 | 58.72 | **2.4969** | 2.9082 |
| 16 | 117.44 | **3.3040** | 4.0386 |
| 32 | 234.88 | allocator refuses (`bank_manager.cpp:462`) | works |

So the threshold goes between 117.44 and 234.88 MB: **128 MB**.

The batch-1 row reads the other way, which is worth stating rather than
quietly dropping. It is an *eager* measurement — 1.7 ms where the traced layer is
0.48 — so most of what it times is host dispatch, and the ordering at a 41 µs
margin is not trustworthy. The warmed traced A/B that actually decides the
shipped configuration is unambiguous and goes the other way:

    expert intermediates in L1      0.4766 ms   (shipped)
    expert intermediates in DRAM    0.5128 ms   7.6% worse

(`probes/decode_levers.py`.) Batch 1 keeps L1, and batch 1 is the latency target.

## 10. The 48-layer footprint, allocated rather than computed

`mesh_plan.md` §9 lists this as the one capacity claim it did not measure.
`probes/footprint_probe.py` allocates it — 48 layers of per-die sharded weights,
embed and lm_head, and 48 paged KV caches at the full advertised context, all
live at once. Archived verbatim at `footprint_probe.log`:

```
DRAM per die: 34.18 GB total
after 48 layers of sharded weights:                  4.596 GB/die
+ embed (replicated) and lm_head (column-parallel):  5.374 GB/die
+ paged KV for 48 layers at ctx 262144:             11.829 GB/die
ALL 48 layers' weights + KV at ctx 262144, batch 1: 11.829 GB/die of 34.18, 22.350 free
```

The arithmetic in `mesh_plan.md` §8 predicted 11.80 GB. The allocator says
11.829. `doc/context_contract.json` now quotes the allocation, not the sum.

A first version of this probe called `upload_multichip_weights` 48 times and had
to be killed: that helper runs `.contiguous().float()` on a 402M-element expert
tensor every call, so the loop measured numpy rather than DRAM. It builds the
per-die host tensors once now.

## 11. Watcher

**The watcher does not fit on active ethernet cores under `FABRIC_1D_RING`.**
First attempt, `TT_METAL_WATCHER=10` on the 4-die mesh:

```
TT_FATAL: Program size (29072) too large for kernel config buffer (25600) on ACTIVE_ETH
```

This is the watcher's instrumentation being added to the fabric router's own
ethernet program, not anything about this model. `TT_METAL_WATCHER_DISABLE_ETH=1`
turns off ethernet-core instrumentation and leaves every worker core watched —
including `reader_bmm_tile_layout_in0_sender_padding.cpp`, which is exactly where
the `nnz` assert lives, so the coverage that matters most here is retained. §5's
`nnz_hazard.log` is the proof: the watcher trips that assert on demand.

Watcher-clean evidence is `watcher.log`.

## 12. Summary of where the plan was wrong

Recorded together because the plan was mostly right, and the exceptions are the
useful part.

| plan said | measured | why |
|---|---|---|
| decode all-reduce: AG-of-partials, 19.96 µs | RS+AG wins; 0.4760 vs 0.4801 ms on the layer | the probe used 32 logical rows, the layer has 1, and `ttnn.sum` pulls a `FillPad` |
| expert pair 2.13× under EP | **1.11×** | dynamic `nnz` costs 1.47×, and the probe's inflated baseline hid it |
| collectives 39.92 µs/layer in decode | **66.76 µs** | a trace of 16 back-to-back collectives pipelines; one sitting between dependent ops does not |
| decode ≈ 1.61× | **1.18×** | the two rows above, 66 µs between them |
| prefill ≈ 3.58× at S=512 | **3.53×**, and 3.80× at S=2048 | right |
| per-die footprint 11.80 GB | **11.829 GB** allocated | right |
| `_dram_sharded_ok` holds at N=1280 | holds | right |
| top-k determinism | holds | right |
| Ring beats Linear | holds, 1.5% on the layer (the standalone sweep said 21%) | collectives are a smaller share of a layer than of a collective-only trace |

## 13. Review pass 1 — the profile window was off by two ops

Review validated the scheme, the code, the PCC, the footprint and every contract
item, and returned one blocking finding plus five derivation errors. **No
re-measurement was needed: the correct data was already in the committed CSVs.**
All of it was re-derivation and text. Recorded here because the defect class —
a number in prose that no artifact produces — is the one that has now cost this
model a review in all three stages.

**The blocking one.** The published decode window was `ops_perf_multichip_decode.csv`
device 0 rows 132–197, "66 ops, 435.83 µs". `probes/profile_layer.py` runs a
prefill priming pass and then *two* decode iterations, and on device 0 sorted by
`HOST START TS` the rows partition as 0–67 prefill, 68–69 setup, 70–133
iteration 1 (64 ops, 414.521 µs), 134–197 iteration 2 (64 ops, 414.661 µs). Rows
132 and 133 are the *previous* layer's closing `AllGatherAsync` and residual
add, so the published window was one full layer plus a two-op tail.

The tell was in the document itself: its op-family table reported **3
`AllGatherAsync` against 2 `ReduceScatterMinimalAsync`**, which is impossible
for a layer with two RS+AG all-reduces. That invariant, and the layer's
start-at-`LayerNorm`/end-at-residual shape, are now written into
`probes/profile_layer.py` next to the row ranges, so the next person to
transcribe this CSV has a checkable boundary rather than an eyeballed one. The
README now quotes both iterations — 414.52 and 414.66, agreeing to 0.03% — for
the same reason.

Everything downstream of the window moved:

| | published | corrected |
|---|---|---|
| decode layer | 66 ops, 435.83 µs | **64 ops, 414.66 µs** |
| `AllGatherAsync` | 47.67 µs, 10.9% | **28.40 µs, 6.8%** |
| everything else | 101.9, 23.4% | **100.15, 24.2%** |
| collectives | 86.03 µs, 2.2× the budget | **66.76 µs, 1.67× the 39.92 budget** |
| gap to the plan's 319 | 116.8 µs | **95.7 µs** |
| of which collectives | +46.1 | **+26.9** |
| collectives as a share of the layer | 19.7% | **16.1%** |
| decode device-time speedup | — | **1.24×** (512.65 / 414.66) |

The term-by-term table in `README.md` was rebuilt at the same time so that it is
a *decomposition* rather than a selection: eleven contiguous row ranges whose
measured values sum to 414.661 exactly. That surfaced two more things the old
table hid — the expert eltwise tail is 70.02 µs, not the "~85" that was typed
(2.48× the single die, not ~2×), and the attention body is 43.42 µs against the
plan's ~30, which is a fourth contributor to the gap that the "three things"
sentence had absorbed into the others.

**The five smaller ones.**

* *Device mismatch.* "attention projections … 17.71" was device 3 in a table
  headed device 0. Per device: d0 17.600, d1 17.636, d2 17.550, d3 17.708. Now
  17.600 and 2.80×, matching the header.
* *Replicated-work figure not derivable.* "132.9 µs … 25.9% … ceiling 3.9×"
  does not follow from its own stated components. Norms 40.234 (rows 45 and 68
  of `../optimized_decoder/ops_perf_optimized_decode.csv`) + router block 88.858
  (rows 69–88) = **129.092 µs = 25.18%** of 512.655, ceiling **3.97×**. Fixed in
  `README.md`, `tt/multichip_decoder.py` and — as a marked arithmetic
  correction, since the document is otherwise frozen — in `mesh_plan.md` §4.2,
  where the error originated.
* *Test coverage overstated by one case.* "8 random inputs **and** an all-zero
  input" is nine; `test_topk_is_identical_across_dies` parametrises `seed ∈
  [0..7]` and spends seed 0 on the zero input, so it is **7 random + 1 all-zero,
  8 cases** — which is what the 8 lines in `pcc_log.txt` say. The test docstring
  was already right; the two docs were not.
* *Internal inconsistencies here.* §8 said "the nnz=8 row above is 107.94 µs"
  where its own table and `nnz_cost.log` say 107.73; §8 said 1.47× and §12 said
  1.46× where 158.01/107.73 = **1.467**; §8 said decode scales at 1.17× where
  `perf_decode.csv` and `perf_baseline_1x1_decode.csv` give 0.5634/0.4767 =
  **1.182**.
* *Context contract contradicted its own formula.*
  `multichip_largest_feasible.batch_at_full_context` read 3 against a stated
  basis of `(34.18 − 5.374 − ~2) / 6.442` = 4.16. The 3 was inherited from
  `mesh_plan.md` §9's draft of the file, which computed it against the *nominal*
  30 GB board figure — `(30 − 5.36 − ~2) / 6.44` = 3.52 — before
  `footprint_probe.log` showed the allocator reports 34.18. Corrected to **4**,
  with the arithmetic spelled out in the file.

One more number was caught by re-deriving rather than by the review: "collectives
are 1.85% of the prefill profile" is not produced by any window of
`ops_perf_multichip_prefill_s512.csv`. The prefill capture is two passes of 261
ops (rows 14–274 and 275–535, after 14 rows of upload); on device 0 the
published second pass is 9221.63 µs with 181.86 µs of collective, i.e. **1.97%**.

**Also done, from the reviewer's non-blocking list.**

* `matmul_reduce_scatter_async` on the `wo`→RS edge is **not** excluded by the
  "adjacent op is a norm or a residual" argument, because `wo` is a matmul. That
  argument covers three of the four collective edges; the fourth is now named as
  an untried lever rather than an excluded one, and bounded — the RS it would
  absorb is 22.04 µs, 5.3% of the layer.
* `_sdpa_program_config` also pins `q_chunk_size=32, k_chunk_size=32`, which the
  default path chose for itself. An unmeasured change riding along with the
  correctness fix; now documented as such in limitation 3. No published number
  goes through that path — every decode timing here is paged, at the op default.
* The workaround was only exercised at batch 1 (`[contiguous]` is batch 1, and
  `test_multichip_decode_batch` is paged), which is precisely the case where the
  cap is what makes the op run at all. `test_multichip_decode_contiguous_batch8`
  adds batch 8, where `num_cores_per_head` would have been legal without the cap
  and the cap therefore has to be proved harmless.
* `PCC_VS_SINGLE_CHIP` was 0.99 for a comparison where only sharding differs.
  Tightened to **0.999**; the worst actual is 0.99945 (two stacked layers) and
  everything else is 0.9996+.
* "Nothing in this table is typed by hand" was true of the Results table and not
  of the prose around it, which is how all of the above happened. The claim is
  now scoped to the table, and the prose commits instead to naming a row range
  next to every figure taken from a profile.

---

## Reconciliation pass — six non-blocking review items

Post-`clean-pass` polish. **No claim changes**; every edit is a published figure
being made to reconcile to the artifact it came from. No re-measurement — all
six were recomputed from the existing CSVs.

| # | where | before | after | artifact |
|---|---|---|---|---|
| 1 | `tt/multichip_decoder.py:747` capacity-padding `topk` | 26.33 µs | **26.32** | `ops_perf_multichip_decode.csv`, device 0, row 162 = 26.324 µs; matches README's op-family table |
| 2 | README gap to plan (§summary and §"Where decode goes") | 95.7 | **95.42** | plan column of the table sums to 319.24, not 319; 414.661 − 319.24 = 95.421 |
| 3 | single-die expert `sparse_matmul` pair, README prose + `multichip_decoder.py:738,741` + work_log §8 | 92.07 | **92.06** | `../optimized_decoder/ops_perf_optimized_decode.csv` rows 90 + 98 = 61,997 + 30,065 ns = 92.062 µs. The README *table* already read 92.06 and was right |
| 4 | README finding 2, collectives overrun | "1.67×" with two baselines in one paragraph | **1.67× against the plan's 39.92 budget, 1.41× against a standalone RS+AG prediction** | 39.92 = 2 × 19.96 AG-of-partials (`mesh_plan.md:491, 505`); 2 × 23.69 RS+AG = 47.38; 66.765 / 47.38 = 1.41 |
| 5 | README decode window | "device 0" | **"device 0, the slowest of the four dies"** + all four values | same CSV, rows 134–197 per device: d0 414.661, d1 394.210, d2 408.148, d3 390.298; mean 401.83 |
| 6 | `context_contract.json` `single_die_equivalent_gb` | 45.3 (19.50 weights) | **45.28 (19.52 weights)**, with the convention stated | recomputed from `footprint_probe.log`'s own per-die tensor list at the shipped dtypes |

On item 2, the four named contributors now reconcile exactly: 39.45 + 26.85 +
15.02 + 13.42 = 94.74, and the six residual blocks sum to +0.68 (+0.041 −0.670
−0.122 −0.063 +1.383 +0.113), for 95.42.

On item 6, the arithmetic in full, at the probe's own `GB = 1e9`: per-die
per-layer, sharded tensors are 94.962 MB and replicated ones 0.786 MB, so
48 × (4 × 94.962 + 0.786) MB = 18.270 GB, plus `embed_tokens` 0.622 GB counted
once and `lm_head` 4 × 0.156 GB = 19.515 → **19.52**. The old 19.50 is
`mesh_plan.md` §8's table, which omits the two tile-padded RMSNorm vectors
(0.26 MB/layer) and rounds each row. Counting the replicated router and norms 4×
instead reads 19.63. All three are far above the 34.18 GB that has to hold them,
so the conclusion is untouched; the field now says which convention it used.

`mesh_plan.md` is deliberately **not** edited. Its 92.07 (lines 386, 687) and
19.50 / 45.3 (lines 517, 610, 612) are the frozen design-phase inputs that the
plan column and the ≈319 total are derived from; correcting them there would
retroactively change a prediction rather than a measurement. Both are now named
and reconciled from the README side instead.
