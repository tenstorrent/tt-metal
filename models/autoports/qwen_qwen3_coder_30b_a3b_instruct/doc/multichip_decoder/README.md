# Multichip decoder — Qwen3-Coder-30B-A3B-Instruct on 4 Blackhole dies

Stage 03. The stage-02 optimized decoder layer, parallelised across the full
4-die `ClusterType.P300_X2` mesh: **attention tensor-parallel by 4, experts
expert-parallel by 4, router and norms and residual replicated**, ring fabric
with both ethernet links, two all-reduces per layer.

Semantics, paged KV cache, non-aligned sequence lengths, determinism, batch
coverage to 32 and the 262144-token context contract are all preserved. The
context contract is not merely preserved — this is the stage that **discharges**
the forward-looking caveat stage 01 left, and it does so by allocating the whole
model rather than by arithmetic.

> **Forward pointer, added by stage 04 — no figure in this document changes.**
> `tt/multichip_decoder.py` was optimized **in place** by stage 04, so the file
> this document describes no longer produces the numbers below: the decode layer
> is now 362.828 µs of device time against the 414.661 recorded here, and
> traced decode at ctx 128 is 0.4286 ms against this directory's 0.4767.
> Everything in this
> directory is the frozen *before* half of that comparison and is deliberately
> never regenerated — in particular `perf_prefill.csv`, `perf_decode.csv` and
> the three `ops_perf_multichip_*.csv.gz`. The perf tests that used to write
> them are now `test_optimized_multichip_*` and write into
> `../optimized_multichip_decoder/`, which is also where stage 04's README,
> work log and operation-topology audit live. The parallelisation, the `nnz`
> contract, the context contract and every correctness claim below are
> unchanged.

Design, with the probe evidence behind it, is `mesh_plan.md`, written before any
of `tt/multichip_decoder.py` existed. `work_log.md` records what happened when
that design met the hardware, including the four places it was wrong.

## Results

| | single chip (1x1) | 4 dies (1x4) | speedup | efficiency |
|---|---|---|---|---|
| prefill S=128 | 71.96 µs/tok | **24.57** | 2.93× | 73% |
| prefill S=512 | 68.93 | **19.51** | 3.53× | 88% |
| prefill S=1024 | 68.82 | **18.38** | 3.74× | 94% |
| prefill S=2048 | 69.34 | **18.26** | **3.80×** | **95%** |
| traced decode ctx128 | 0.5634 ms | **0.4767** | 1.18× | 30% |
| traced decode ctx1k | 0.6619 | **0.5722** | 1.16× | 29% |
| traced decode ctx4k | 0.9917 | **0.9124** | 1.09× | 27% |

Efficiency is speedup / 4. Every cell is a cell of a CSV in this directory —
`perf_baseline_1x1_prefill.csv`, `perf_prefill.csv`,
`perf_baseline_1x1_decode.csv`, `perf_decode.csv` — and the speedup and
efficiency columns are computed from those cells by
`probes/summarize_perf.py`, which writes `perf_summary.json`. No cell of *this
table* is typed by hand. The prose and the op-level tables below are typed by
hand from the CSVs, so every number in them is re-derivable from a named row
range — and where one was not, review caught it; see `work_log.md` §13.

**The single-chip column is measured here, not quoted from stage 02.** It is the
same `optimized_decoder.py`, the same harness, the same tree, run on a 1x1 mesh
by `test_multichip_baseline_1x1_prefill` / `_decode` into this directory. Stage
02's own CSVs are left untouched, because re-running them would move their third
significant figure and silently invalidate the document that quotes them cell by
cell. (The two agree: stage 02's `perf_decode.csv` reads 0.5634 ms at ctx128 and
so does this one.)

**Prefill is the headline: 3.80× on four dies, 95% parallel efficiency.**
Collectives are 1.97% of the prefill profile — 181.86 µs of 9221.63
(`ops_perf_multichip_prefill_s512.csv`, device 0, rows 275–535, the second of
the two prefill passes `probes/profile_layer.py` runs). Prefill's expert path is dense per
32-token tile, so expert parallelism fractures it with zero load imbalance and an
exact `nnz`, which is the best case this mesh has.

**Decode is 1.18×, against the plan's predicted 1.61×, and the gap is
accounted for term by term.** 25.18% of the single-die decode layer is
replicated work that no parallelisation on this mesh can remove — the two
RMSNorms and the router — which is why the ceiling is 3.97× even at infinite
dies. On top of that, the measured layer is 414.66 µs of device time against the
plan's estimated 319.24, and 94.74 µs of the 95.42 µs difference is four things:
the expert matmuls cost 39.45 µs more than predicted (dynamic `nnz`), collectives
26.85 µs more than the plan budgeted, the expert eltwise tail 15.02 µs more, and
the attention body 13.42 µs more; the other six blocks add +0.68 between them. See "Where decode goes" below, where every
row is a row range of the profile CSV and the eleven rows sum to the total.

## Correctness

Validated against the **single-chip TTNN optimized decoder**, not only against
HuggingFace. The baseline is run *on the same mesh with every tensor replicated*,
so each die independently computes the exact single-chip answer from the same
host tensors in the same process — which removes every source of numerical
difference except the sharding and the collectives.
`test_baseline_upload_is_actually_replicated` asserts the four dies agree to
0.0 before anything else divides by that reference.

| | vs single-chip TTNN | vs HF |
|---|---|---|
| prefill S = 32, 33, 100, 128, 257, 512 | 0.99962 – 0.99987 | 0.99896 (S=33), 0.99909 (S=128) |
| decode, contiguous cache, batch 1 | 0.99997 | — |
| decode, contiguous cache, batch 8, per user | 0.99996 – 0.99998 | — |
| decode, paged cache (block 32) | 0.99997 | — |
| decode, 4 consecutive steps | — | 0.99878 – 0.99931 |
| decode, batch 1 / 2 / 8 / 32, per user | — | 0.99335 – 0.99959 |
| two stacked layers | 0.99945 | — |
| layer under maximally unbalanced routing | 0.99994 | — |
| router windows stitched vs global dense routing | **exactly 0.0 difference** | — |
| per-die KV heads stitched vs single-chip cache | 1.0 | — |

All of these are lines of `pcc_log.txt`, which is the filtered log of the
38-test run of `tests/test_multichip_decoder.py`.

Also asserted, because each is a way this scheme can be silently wrong:

* **`ttnn.topk` is bit-identical across the four dies** — 8 cases: 7 random
  128-token inputs and one all-zero input, where the top-8 is decided entirely
  by tie-breaking. If it were not, the four 32-expert windows would stop being a
  partition of the global top-8 and the layer would drift with no error.
* **A die holding none of the global top-8 contributes exactly 0.0**, not merely
  something finite. Constructed, not hoped for.
* **Trace replay is bit-identical to eager** (PCC 1.0) and reads the live input
  buffer (0.158 delta on swap).
* **20 consecutive decode steps are bit-identical**, which is the determinism
  claim for two async collectives per layer with cycling semaphores.
* **Three prefill runs are bit-identical** on all four dies.
* **No runtime fallback fires** — see below.

## The parallelisation, per tensor

Per die, at the shipped dtypes. Every division is exact, so **the scheme needs
zero load-time padding** and the contract's allowance for it goes unused.

| tensor | full | mesh mapping | per die | per die bytes |
|---|---|---|---|---|
| `input_layernorm`, `post_attention_layernorm` | [2048] bf16 | replicate | [2048] | 4 KB each |
| `wqkv` (prefill, interleaved) | [2048, 5120] bfp8 | **shard N, head-interleaved** | [2048, 1280] | 2.785 MB |
| `wqkv_decode` (DRAM width-sharded, 8 banks) | [2048, 5120] bfp8 | same | [2048, 1280] | 2.785 MB |
| `q_norm`, `k_norm` | [128] bf16 | replicate | [128] | 512 B each |
| K cache / V cache | [B, 4, ctx, 128] bf16 | **shard KV head** | [B, **1**, ctx, 128] | 512 B/token |
| `wo` (prefill) | [4096, 2048] bfp8 | **shard K by Q head** | [1024, 2048] | 2.228 MB |
| `wo_decode` (DRAM width-sharded) | [4096, 2048] bfp8 | same | [1024, 2048] | 2.228 MB |
| `router` | [2048, 128] bf16 | **replicate** | [2048, 128] | 0.524 MB |
| `expert_window` (new) | — | **device-varying** | [128, 32] bf16 | 8 KB |
| `gate_up_proj` | [1, 128, 2048, 1536] bfp4 | **shard expert dim** | [1, **32**, 2048, 1536] | 56.623 MB |
| `down_proj` | [1, 128, 768, 2048] bfp4 | **shard expert dim** | [1, **32**, 768, 2048] | 28.312 MB |
| | | | **per layer** | **95.5 MB** |

Program configs and shard specs, per die:

| config | single chip | under this scheme |
|---|---|---|
| `_dram_sharded_program_config(qkv)` | `in0_block_w=8, per_core_M=1, per_core_N=20` | `in0_block_w=8, per_core_M=1,` **`per_core_N=5`** |
| `_dram_sharded_program_config(wo)` | `in0_block_w=16, per_core_M=1, per_core_N=8` | **`in0_block_w=4`**`, per_core_M=1, per_core_N=8` |
| DRAM weight shard (qkv) | [2048, 640] × 8 banks | [2048, **160**] × 8 banks |
| DRAM weight shard (wo) | [4096, 256] × 8 banks | [**1024**, 256] × 8 banks |
| `_width_sharded_l1` qkv out | [32, 640] | [32, **160**] |
| `_width_sharded_l1` wo in | [32, 512] | [32, **128**] |
| `_tuned_sparse_matmul_config` gate/up | M=1, N=1536, `in0_block_w=16` | **unchanged** |
| `_tuned_sparse_matmul_config` down | M=1, N=2048, `in0_block_w=12` | **unchanged** |
| decode `nnz` | `top_k * batch`, exact | **`None`** — see below |
| prefill `nnz` | `128 * group_size` | `32 * group_size`, still exact |
| `_DECODE_EXPERT_L1_BUDGET_BYTES` | 40 MB (asserted) | **128 MB (swept)** |
| SDPA-decode program config | `None` | `None` paged, `max_cores_per_head_batch=64` contiguous |

Both expert `sparse_matmul` configs survive untouched because EP fractures the
*batch* (expert) dimension and leaves M, N and K exactly as stage 02 tuned them.
That is the whole reason EP was chosen over splitting `moe_intermediate`, which
measured 1.51× against EP's 2.13× in the design sweep.

The **wqkv column split is a permutation, not a slice**. The checkpoint stores
`[Wq(4096) | Wk(512) | Wv(512)]`, so a contiguous 4-way split gives die 0 nothing
but Q heads — with no shape error, because
`nlp_create_qkv_heads_decode(num_heads=8, num_kv_heads=1)` accepts 1280 columns
whatever is in them. `head_interleaved_wqkv` rebuilds the tensor so die *d*'s
contiguous slice is `[Q heads 8d..8d+7 | K head d | V head d]`.

## Where decode goes, and why it is 1.18× rather than 1.61×

The window is **one decode layer: rows 134–197 of
`ops_perf_multichip_decode.csv` sorted by `HOST START TS`, device 0, 64 ops,
414.66 µs** of device time against the single chip's 512.65 µs. **Device 0 is
the slowest of the four dies**, and is published for that reason: the same
window reads 394.21 µs on device 1, 408.15 on device 2 and 390.30 on device 3
(mean 401.83). A synchronized mesh advances at its slowest die, so every decode
figure below is taken against the worst of the four rather than the mean or the
best. `probes/profile_layer.py` runs a prefill priming pass and then *two* decode
iterations, so the CSV holds two complete layers on each die: rows 70–133 are
iteration 1 at **414.52 µs** and rows 134–197 are iteration 2 at **414.66 µs**.
The published window is **iteration 2** — the second and last, the one with a
fully warm program cache. The two agree to 0.03%, which is the corroboration
that the window boundaries are right; the boundary itself is structural, since
a layer must contain exactly two `ReduceScatterMinimalAsync` and two
`AllGatherAsync` and begin at the `input_layernorm` that follows the previous
layer's residual add.

| op family | multichip | share |
|---|---|---|
| `SparseMatmul` (the experts) | 82.65 µs | 19.9% |
| `LayerNorm` (2 residual norms + 2 per-head) | 47.92 | 11.6% |
| `Matmul` (router, qkv, wo, window, ones) | 46.04 | 11.1% |
| `ReshapeView` (expert M-padding compaction) | 44.80 | 10.8% |
| `ReduceScatterMinimalAsync` | 38.36 | 9.3% |
| `AllGatherAsync` | 28.40 | 6.8% |
| `TopK` (one core, one 128-wide row) | 26.32 | 6.3% |
| everything else | 100.15 | 24.2% |

Term by term against the single chip, and against what the plan predicted. The
row ranges are the two profiles sorted by `HOST START TS`: multichip is
`ops_perf_multichip_decode.csv` device 0 rows 134–197, single die is
`../optimized_decoder/ops_perf_optimized_decode.csv` rows 45–103. **Every block
is a contiguous row range, and the eleven measured values sum to 414.661
exactly** — so this table is a decomposition, not a selection.

| block | 1 die | plan | measured | verdict |
|---|---|---|---|---|
| `input_layernorm` | 20.04 µs | 20.04 | 20.08 | replicated, as designed |
| attention projections (qkv + wo) | 49.24 | 18.27 | **17.60** | **2.80×**, as designed |
| attention body (heads, per-head norms, RoPE, cache, SDPA, layout) | 64.62 | ~30 | 43.42 | 1.49× — launch floors, not work |
| all-reduce after `wo` | 0 | 19.96 | **36.32** | **1.82× the budget** |
| residual add | 2.00 | 2.00 | 1.88 | replicated |
| `post_attention_layernorm` | 20.19 | 20.19 | 20.13 | replicated, as designed |
| router block | 88.86 | 88.86 | 90.24 | replicated, as designed |
| expert `sparse_matmul` pair | 92.06 | 43.2 | **82.65** | **1.11×, not 2.13×** |
| expert reshape/eltwise tail | 173.88 | ~55 | 70.02 | 2.48× |
| all-reduce after the experts | 0 | 19.96 | **30.45** | **1.53× the budget** |
| residual add | 1.76 | 1.76 | 1.87 | replicated |
| **total** | **512.65** | **≈319** | **414.66** | **1.24× on device time** |

The plan column sums to **319.24** — the "≈319" in the table is that figure
rounded — so the gap to the measured 414.66 is **95.42 µs**. Four blocks account
for 94.74 of it: the expert matmul pair (+39.45), the two collectives together
(+26.85), the expert eltwise tail (+15.02) and the attention body (+13.42). The
remaining six blocks contribute **+0.68 between them** and none is off by more
than 1.4 µs individually. The first two are real findings. The last two are the
plan's two "~4× on work, with a launch floor added by hand" estimates
(`mesh_plan.md` §4.2) being optimistic by a combined 28.4 µs — both blocks are
long strings of small ops, so what the fracture buys in work the per-op launch
cost takes back.

The two findings:

1. **Dynamic `nnz` costs ~26 µs.** Under EP the locally-live expert count is
   data-dependent, so decode *must* pass `nnz=None` (see below), and dynamic mode
   measures **1.47×** the exact-`nnz` pair at the shipped shapes — 158.01 µs
   against 107.73 (`probes/nnz_cost_probe.py`). The design phase priced this from
   a sweep whose E=128 baseline read 264.65 µs where the real layer reads 92.06,
   and used only ratios from it, which was the right caution and still was not
   enough.
2. **Collectives cost 66.8 µs, not 39.9.** In the layer the two all-reduces cost
   36.32 and 30.45 µs. That is **1.67× the plan's 39.92 µs budget**, which is
   2 × 19.96 — the *AG-of-partials* figure the plan chose for decode
   (`mesh_plan.md:491, 505`), not the shipped op pair. Against a like-for-like
   *standalone RS+AG prediction* — 2 × the 23.69 µs the same sweep measured for
   RS+AG at `[1,1,32,2048]` (`mesh_plan.md:491`), i.e. 47.38 µs — the overrun is
   **1.41×**. The two baselines answer different questions and neither should be
   read as the other: 1.67× is the miss against what the plan budgeted, 1.41× is
   the miss against what a standalone sweep of the shipped collectives would have
   predicted. Either way the cause is the same: a trace of 16 back-to-back
   collectives pipelines in a way that a collective sitting between two dependent
   ops cannot.

And the structural ceiling was known in advance: **129.1 µs of the 512.65 µs
single-die decode layer — 25.18% — is replicated work** (both residual RMSNorms
40.23 µs, rows 45 and 68; the router block 88.86 µs, rows 69–88). Both RMSNorms
are latency-bound (128 KB in 20 µs is 6.5 GB/s) and `topk` over a 128-wide row
occupies one core. Even at infinite dies decode cannot exceed **3.97×**.

## The `nnz` contract — a device hang, reproduced

`ttnn.sparse_matmul` bakes `nnz` in as a compile-time arg and requires
`count_nonzero(sparsity) == nnz` exactly; a mismatch **deadlocks the device**
(`sparse_matmul_device_operation.cpp:205-211`, tt-metal #45943). A mesh op is
SPMD, so one `nnz` is compiled for four dies, and under EP the local live count is
data-dependent in 0…8 and different on each. **Decode passes `nnz=None`.**

This stage hit the hazard for real: an early version of the empty-window test
built a router whose logits, against the rms-normed activation, spanned more than
1000, two of the eight routing weights underflowed `exp()` to exactly zero in
bf16, and the **single-chip** leg of the test — which passes `nnz = top_k * batch`
— deadlocked the mesh and needed `tt-smi -r`. The multichip leg, on `nnz=None`,
was unaffected. `work_log.md` §5 has the account.

`probes/nnz_hazard_probe.py` reproduces it deliberately, on one die, under the
watcher, which turns the hang into an abort. Archived at `nnz_hazard.log`:

```
P|exact nnz=8 with 8 live: ok
P|nnz=None with 8 live:    ok
P|nnz=None with 0 live:    ok, max|out| = 0.0
P|about to run nnz=8 against 2 live entries...
  Device 3 worker core(x=0,y=0): BRISC tripped an assert on line 431.
  Current kernel: reader_bmm_tile_layout_in0_sender_padding.cpp
```

## Runtime fallback audit — clean

Three helpers imported from stage 02 pick a slower path *silently*, and all three
see inputs they were not tuned against. `fallback_audit()` returns them as data
and `test_no_runtime_fallbacks` asserts on it:

```
batch 1:  dram_sharded_taken True, qkv (2048, 1280), wo (1024, 2048),
          gate_up_in0_block_w 16, down_in0_block_w 12,
          expert_intermediate_buffer L1, local_heads (8, 1), local_experts 32
batch 32: ... expert_intermediate_buffer DRAM
```

`_dram_sharded_ok` needs both dims divisible by `8 banks × 32 = 256` and per-die
wqkv N is 1280 = 5×256 — **one factor of two from failing**, at which point
stage 02's 1.11× DRAM-sharded decode attention would disappear with no error at
all. `upload_multichip_weights` asserts it at upload time as well.

## What was tried and did not help

Measured on the whole warmed traced layer at ctx128, median of 100
(`probes/decode_levers.py`), re-audited *after* the collective change:

| lever | traced decode | verdict |
|---|---|---|
| shipped (RS+AG, ring, 2 links, L1 intermediates) | **0.4766 ms** | — |
| expert intermediates in DRAM instead of L1 | 0.5128 | **7.6% worse**; L1 confirmed at batch 1 |
| `num_links=1` | 0.4738 | 0.6% better — noise-level, and prefill needs 2 links (1.84× at 2 MB), so a shared code path keeps 2 |
| bfloat8_b collective payload | 0.4854 | 1.8% worse **and** less accurate; decode is latency-bound, so halving the bytes buys nothing |
| `Topology.Linear` (the repo default for a 4-device mesh) | 0.4836 | 1.5% worse — in-situ confirmation of the plan's Ring choice, though much smaller than the 1.21× the standalone sweep suggested |

Earlier in the stage:

| lever | result |
|---|---|
| AG-of-partials + local sum for decode — **the plan's choice** | 0.4801 ms against RS+AG's 0.4760. The standalone probe measured 32 logical rows; the shipped tensor has 1, so `ttnn.sum` pulls a 5.7 µs `FillPad`. Reversed and replaced. |
| AG + reshape-to-padded + sum, to dodge that `FillPad` | **did not run** — `reshape_common.cpp:50`, `new_volume == old_volume` |
| default (LoFi) fidelity on the expert-window matmul | 9.77e-4 error against the single-chip routing — one bf16 ulp. LoFi keeps ~5 mantissa bits, so multiplying by 1.0 is not a copy. HiFi4 makes it exact and is free at 4 tiles × 1. |
| capacity-padding decode to `nnz = top_k` | needs a second `topk` over the local 32 to build a fixed-count sparsity, and `topk` on one row is **26.32 µs** in this very profile — more than the ~26 µs it would save |
| dense all-expert decode | forbidden unless faster, and it is not: `nnz=8` over 8 live slots is already 107.73 µs and 32 slots is strictly more |

## Named limitations

1. **Dynamic `nnz` is 1.47× exact `nnz`** and there is no legal exact value under
   EP. ~26 µs of every decode layer. Blocked on TTNN offering a device-side or
   upper-bounded `nnz`, not on this model.
2. **Collectives are 16.1% of the decode layer** (66.8 µs) against the 39.92 µs
   the plan budgeted (47.38 µs if predicted from the standalone RS+AG sweep
   instead — see finding 2 above). `all_gather_matmul_async` /
   `matmul_reduce_scatter_async` were not evaluated. For three of the four
   edges the reason is that the adjacent op is an RMSNorm or a residual add,
   not a matmul, so neither fusion applies without restructuring the residual
   contract, which §4.1 of the plan already rejected on traffic grounds. **The
   fourth edge is not covered by that argument and is a real untried lever:**
   `wo` *is* a matmul and it feeds the first reduce-scatter directly, so
   `matmul_reduce_scatter_async` is shape-eligible there. It was not tried, for
   two reasons that are a judgement and not a measurement — `wo` is the
   DRAM-width-sharded decode projection whose program config stage 02 tuned and
   the fused op takes its own, and the RS it would absorb is 22.04 µs of a
   414.66 µs layer, so the whole edge is worth at most 5.3%. Naming it as
   untried rather than excluded.
3. **SDPA-decode refuses 1 KV head on the contiguous cache at small batch.**
   `sdpa_decode_program_factory.cpp:245`, "Tree reduction max 6 rounds (64
   cores/head), got 110 cores/head" — a failure created by the head split, since
   4 KV heads gave 27 cores/head. Worked around with
   `max_cores_per_head_batch=64`. The paged path is unaffected and runs at the op
   default. **Two caveats on the workaround, both named rather than measured
   away.** First, supplying an `SDPAProgramConfig` at all means supplying
   `q_chunk_size` and `k_chunk_size`, and this one pins both to 32 where the
   default path chose them itself — a second change riding along with the
   correctness fix, unmeasured because no published number goes through this
   path (every decode timing here is paged, at the op default). 32 is the
   minimum legal chunk and the decode Q is one tile, so it is the conservative
   choice, not a tuned one. Second, the "only at small batch" claim is now
   exercised in both directions: `test_multichip_decode_vs_single_chip[contiguous]`
   covers batch 1, where the cap is what makes the op run at all, and
   `test_multichip_decode_contiguous_batch8` covers batch 8, where
   `num_cores_per_head` would have been legal without the cap and the cap must
   therefore be proved harmless.
4. **The watcher cannot be enabled on active ethernet cores** on this fabric
   config: `Program size (29072) too large for kernel config buffer (25600) on
   ACTIVE_ETH`. Watcher runs use `TT_METAL_WATCHER_DISABLE_ETH=1`, which still
   instruments every worker core — including the `sparse_matmul` reader where the
   `nnz` assert lives, as `nnz_hazard.log` shows.
5. **Batch is capped at 32**, unchanged by TP:
   `nlp_create_qkv_heads_decode_device_operation.cpp:51` asserts
   `num_users <= 32`.
6. Everything stage 02 named — prefill's dense-per-tile expert path, the
   `Tile([1,32])` output-tile blocker, the interleaved prefill attention copy —
   carries over unchanged, because EP does not touch M, N or K.

## Context contract

`doc/context_contract.json` is updated, and **capability goes up, not down**.
`probes/footprint_probe.py` allocates the real thing rather than computing it —
48 layers of per-die sharded weights, embed and lm_head, and 48 paged KV caches
at the full 262144 context, all live at once (`footprint_probe.log`):

```
after 48 layers of sharded weights:                        4.596 GB/die
+ embed (replicated) and lm_head (column-parallel):        5.374 GB/die
+ paged KV for 48 layers at ctx 262144:                   11.829 GB/die
                          of 34.18 GB, 22.350 GB/die free
```

One die would need **19.52 GB** of weights plus 25.77 GB of KV — **45.28 GB**,
which does not fit. That weight figure is recomputed from the probe's own per-die
tensor list on a stated convention: sharded tensors (both expert tensors, both
`wqkv` copies, both `wo` copies, `lm_head`) counted 4×, replicated tensors (the
router, the two RMSNorm vectors, `embed_tokens`) counted once, since one die
would hold exactly one of each. Counting the replicated router and norms 4× as
well reads 19.63; `mesh_plan.md` §8's 19.50 is the same arithmetic with the two
per-layer RMSNorm vectors (0.26 MB/layer) left out and each row rounded. All
three land in the same place against 34.18 GB of DRAM. **The mesh is a capability requirement here, not only a speed one**, and
the stage-01 `forward_looking_note` ("stage 05 will have to weigh KV dtype,
paging across dice, or a served-context cap") is discharged: none of those
trade-offs is needed. KV dtype, paging and the served context are all unchanged.

The arithmetic in `mesh_plan.md` §8 predicted 11.80 GB; the allocator says
11.829.

## Verification

```bash
source python_env/bin/activate

# correctness on the 4-die mesh (38 tests)
pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/test_multichip_decoder.py -q

# whole suite, no watcher, perf deselected
pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/ -q \
  -m "not models_performance_bare_metal"

# watcher-clean. DISABLE_ETH is required: the watcher's active-eth program does
# not fit alongside the fabric router. Never combine with the perf tests --
# they rewrite the published CSVs and the watcher inflates device timings ~8x.
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/ -q \
  -m "not models_performance_bare_metal"

# performance (rewrites the four CSVs in this directory)
pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/test_perf.py -q -k multichip
python models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/multichip_decoder/probes/summarize_perf.py

# op-level profile
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/multichip_decoder
python -m tracy -v -r -p --sync-host-device -o /tmp/prof_mc_dec $D/probes/profile_layer.py decode
python -m tracy -v -r -p --sync-host-device -o /tmp/prof_mc_pf  $D/probes/profile_layer.py prefill
tt-perf-report /tmp/prof_mc_dec/reports/*/ops_perf_results_*.csv

# the probes behind the numbers above
python $D/probes/decode_levers.py        # the lever table
python $D/probes/allreduce_ab.py         # the collective A/B
python $D/probes/nnz_cost_probe.py       # dynamic vs exact nnz
python $D/probes/l1_budget_probe.py      # the L1 threshold sweep
python $D/probes/footprint_probe.py      # the 48-layer allocation
TT_METAL_WATCHER=10 python $D/probes/nnz_hazard_probe.py   # aborts on purpose

# context contract
python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/qwen_qwen3_coder_30b_a3b_instruct \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct --require-contract
```

## Artifacts in this directory

| file | what it is |
|---|---|
| `mesh_plan.md` | the design, written before the code, with its own probe evidence |
| `work_log.md` | what happened when the design met the hardware |
| `perf_baseline_1x1_prefill.csv`, `perf_baseline_1x1_decode.csv` | the single-chip baseline, re-measured here |
| `perf_prefill.csv`, `perf_decode.csv` | the multichip measurements |
| `perf_summary.json` | speedup and efficiency, computed from those four by `probes/summarize_perf.py` |
| `pcc_log.txt` | every PCC quoted above, as logged by the 38-test run |
| `ops_perf_multichip_decode.csv`, `tt_perf_report_multichip_decode.txt` | the decode profile |
| `ops_perf_multichip_decode_agsum.csv` | the same layer with the plan's rejected all-reduce, kept so its `FillPad` rows stay on disk |
| `allreduce_ab.log`, `decode_levers.log`, `l1_budget.log`, `nnz_cost.log` | the four sweeps quoted above |
| `ops_perf_multichip_prefill_s512.csv`, `tt_perf_report_multichip_prefill_s512.txt` | the prefill profile |
| `footprint_probe.log` | the 48-layer allocation the context contract quotes |
| `nnz_hazard.log` | the deliberate `nnz` mismatch, aborted by the watcher |
| `watcher.log` | the watcher-clean run |
| `probes/` | every script above, runnable |

## Note on the raw op profiles

The three `ops_perf_multichip_*.csv` files are stored **gzipped**. A 4-die
profile carries four device columns, so they run 1.2–3.4 MB uncompressed and
exceed the repo's 500 KB per-file limit; gzip -9 brings them to 98–272 KB with
no loss. Every row-number citation in this README and in `work_log.md` refers
to the *uncompressed* file, so read them with:

```bash
zcat doc/multichip_decoder/ops_perf_multichip_decode.csv.gz | less
# or, to re-derive a window in python:
#   import gzip, csv; rows = list(csv.DictReader(gzip.open(path, "rt")))
```

Row indices are 0-based over the rows of one device after sorting by
`HOST START TS`, as described beside each table.
