# Optimized full model — work log

Stage 7 of the Muse-Glimmer-30B bringup: optimize the complete model and generator
path across the 1x4 Blackhole mesh, starting from the completed full model at
`93adb25b7a8`. Skills: `$multichip`, `$optimize`, `$tt-device-usage`, plus
`$qualitative-check` and `$stage-review`.

The headline is in [`README.md`](README.md). This log is the order things were done in,
what was measured, and what was rejected and why.

## 1. Where the step actually was, before touching anything

The full-model stage handed over a step that was already close to its floor: token-out
23.811 ms/token against a 23.239 ms layer-stack lower bound, with the two traces
accounting for the whole step to within 4 µs and every per-token host refresh counter at
zero. So the usual optimized-full-model targets — a host-stepped decode loop, a
full-vocab logits gather, a dominant sampler, an untraced path — were all already closed.

That made the first job attribution rather than tuning. Three things were measured
before any code changed:

**The decode profile, per op.** `../full_model/tracy/decode_perf_report.csv`, the
committed reduced-variant window (2 layers + terminal). Of a ~445 µs sliding layer,
255 µs is six DRAM-sharded projections and ~190 µs is everything else. Two rows in the
"everything else" stood out as *outliers rather than costs*:

* the SwiGLU multiply at **18.03 µs**, against **1.88 µs** for a plain 6656-wide
  residual add on the same 16-core grid — a 10x gap for a wider tensor doing less;
* the softcap `tanh` (17.71 µs) and `* T` (19.14 µs), both on a DRAM-interleaved
  `[1, 1, 32, 50688]` tensor, i.e. 36.85 µs to apply two elementwise functions to
  3.24 MB.

**TTFT, by phase.** `bench/ttft_breakdown.py` on the real 52-layer build, five prompt
lengths, with a device synchronisation around each phase
(`ttft_breakdown_before.json`). At prompt 128 the 52-layer stack is 60.28 ms of the
64.80 ms window, against ~43.7 ms of device time from the committed prefill profile.
~16.6 ms unaccounted for. A linear fit over 512/1024/2048 rows puts the *fixed* part of
the stack at ~36 ms and the marginal part at 0.098 ms/token, which is the signature of
per-call cost rather than per-row work.

**The layer-stack floor, re-derived.** Nothing yet, but noted: the floor in the
full-model README comes from the decoder stage's `layer_ab.py` at context 2048, so any
change to the layer invalidates it and it has to be re-measured with the same harness.

## 2. The TTFT gap: attributed, then removed behind a flag

This took the largest share of the stage's device time. The first pass through it
attributed the gap and declined to remove it; the stage review rejected that, and the
second pass measured the removal and shipped it opt-in. Both passes are recorded,
because the first one's two mistakes are the interesting part.

### 2.1 Is the device idle?

`bench/prefill_host_probe.py`. Issue all 52 prefill layers with no synchronisation, then
synchronise and measure again:

```
issue = 54.91 ms      issue + drain = 55.08 ms
```

0.17 ms of device work outlives the last dispatch. The prefill is **host-issue bound**,
not device bound. (The same probe's `op_floor` arm reported 26.45 µs for a trivial
`ttnn.add` in a 500-call loop with all 500 outputs kept alive; that number is *not*
used anywhere, because holding 500 output buffers makes it an allocator measurement.
`ccl_host_probe.py` later measured the same op properly at 28.6 µs on a 1.7 MB payload
with the output freed each round, which happens to agree — but the loop-with-live-outputs
arm was a bad probe and is called out here rather than quietly reused.)

### 2.2 Where in the host?

`cProfile` over one prefill (`logs/prefill_cprofile_128.txt`): **0.057 s of 0.064 s
inside `ttnn/decorators.py::FastOperation.__call__`**, over **4122 calls**. The
decorator's own Python bookkeeping — `_requires_slow_runtime` and
`is_python_io_recording_enabled`, both called 4122 times — is ~0.001 s combined. So the
cost is the C++ dispatch, and the lever is op *count*, not Python.

### 2.3 Which ops?

`bench/prefill_opcount.py` patches that one frame to count and time by
`python_fully_qualified_name` over exactly one prefill (`prefill_opcount.json`).
62.75 ms of wall time, of which 58.56 ms lands on named ops:

| op | calls | ms | µs/call |
| --- | --- | --- | --- |
| `reduce_scatter_minimal_async` | 104 | 14.60 | 140.3 |
| `all_gather_async` | 105 | 6.33 | 60.3 |
| `ttnn.linear` | 312 | 5.97 | 19.1 |
| `ttnn.add` | 104 | 5.19 | 49.9 |
| `ttnn.rms_norm` | 313 | 5.17 | 16.5 |
| `ttnn.multiply` | 104 | 4.26 | 41.0 |
| `interleaved_to_sharded` | 208 | 3.67 | 17.6 |
| `ttnn.deallocate` | 1957 | 3.27 | **1.67** |
| `sharded_to_interleaved` | 208 | 2.26 | 10.9 |

The 4.19 ms between the 62.75 ms wall and the 58.56 ms of named ops is not device time:
the patched frame times the ttnn call only, so the model's own Python between calls is
the rest — which `cProfile` independently puts at ~11 % of the window.

Two hypotheses died here. **"There are too many deallocates"** — 47 % of the calls and
5.6 % of the time; removing all of them would save 3.3 ms and cost unbounded DRAM at long
prompts. And **"the sharded prefill norms' i2s/s2i brackets are overhead"** — they are
5.93 ms of host, but the decoder stage measured the sharded norm as ~101 µs of *device*
saving per norm at 128 rows, i.e. ~21 ms over 4 norms x 52 layers. Removing them would
trade 5.9 ms of host for 21 ms of device and make the device the bottleneck. Kept.

What survived: **209 collective dispatches are 20.93 ms — 33 % of the wall time on 5 %
of the calls.**

### 2.4 Can the collectives be made cheaper to issue?

`bench/ccl_host_probe.py`, away from the model, at the real `[1, 1, 128, 6656]` payload,
40 successive calls with no synchronisation. **The first version of this pass was wrong
in two ways and the stage review caught both**, so the corrected version is what is
recorded and the retraction is kept:

* it used a **BF16** payload. The model's prefill reduction payload is **BFP8** — 104
  `ttnn.typecast` calls produce it, and `tracy/prefill_128_perf_report.csv` shows the
  `ReduceScatterMinimalAsyncDeviceOperation` input dtype as `BFLOAT8_B`;
* it measured a **hot loop of identical collectives**, which is not the regime the model
  runs them in, and then concluded "nothing moves it" from a figure (58–60 µs/call) that
  is under half the in-model 125–140 µs. That left ~6.8 ms of the gap attributed to
  nothing, which the review correctly refused.

Re-measured at the model's dtype, and again with one prefill-sized matmul in front of
each call (`--loaded-queue`), plus an in-model pass that **drains the device before each
collective** so the recorded time is dispatch without backpressure. The full tables are
in the README; the three conclusions:

* **the 2x gap is reproduced by the instruction stream, not by the payload.** One matmul
  in front takes the reduce-scatter from 72.10 to **117.05 µs**, against 114.6 µs
  in-model drained and 140.3 µs in-model pipelined. So ~26 µs is queue backpressure the
  drained pass removes and the rest is the op's dispatch in a realistic stream. Nothing
  is unattributed;
* **`ttnn.all_reduce` costs two dispatches** (118.04 µs) and the composite wrappers cost
  what the primitives they lower to cost, so "one fused call instead of two" is not
  available and the async/wrapper choice is not costing host time;
* **persistent buffers *do* move it** — 14 % in the hot loop (62.12 against 72.10) and
  17 % loaded (96.86 against 117.05) at the model's BFP8/4-worker setting, i.e. ~2 ms of
  this prefill. They are **not adoptable**: the decoder stage rejected them on an
  intermittent first-use correctness race that moved between arms and between runs. The
  earlier "within noise" claim was the BF16 hot-loop arm and is withdrawn.

The first run of the probe also produced a useful failure: `multi_device_global_semaphore`
for `reduce_scatter_minimal_async` needs **three** semaphores, not two
(`operation_attributes.semaphore.size() == num_expected_semaphores`). Matching
`MultichipDecoder._ccl_semaphores`, which creates three, fixed it.

### 2.5 Tracing the prefill: captured, measured, and shipped opt-in

The first version of this section *costed* a prefill trace instead of capturing one, and
the review was right that an arithmetic estimate does not clear the bar for rejecting a
material optimization. So it was captured, on the real 52-layer build, with the decode
and sampling traces already in the same trace region
(`bench/prefill_trace_probe.py`, `prefill_trace_probe.json`):

| | value |
| --- | --- |
| eager prefill | 59.80 ms |
| **warmed traced replay** | **44.96 ms — 1.33x** |
| replay vs eager | **bit-identical** (`torch.equal`, `max_abs_diff = 0.0`) |
| capture | 98.16 ms |
| DRAM retained per device, 128 rows | 3.3 MB |
| payback | 6.6 replays of the same bucket |

Every part of the estimate was wrong in the safe direction except the payback, which was
guessed at ~7 and measured at 6.6. It works, it is exact, it fits alongside the decode
traces, and it costs almost no DRAM at this row count — so it **ships**, as
`GeneratorConfig.prefill_trace` with `prefill_trace_max_entries` bounding the bucket
cache. Through the same evidence harness: **TTFT 63.68 → 49.76 ms, −21.9 %**, decode
unchanged to 0.14 % (`evidence_perf_prefill_trace.json`).

Off by default, because one trace serves one 32-row padded-length bucket and capture
costs 98 ms against a ~15 ms per-replay saving: a win for a caller that repeats or
buckets prompt lengths, a one-time cost for one that does not, and the generator cannot
tell which it is. `test_prefill_trace_is_opt_in_and_matches_the_eager_path` pins the
contract — same tokens as eager, one bucket, non-aligned lengths inside it served by it,
eager fallback past the bound, released on `teardown()` and on an externally bound KV
cache (whose buffer addresses the trace bakes in).

## 3. The three changes that did ship

### 3.1 The softcap, on the matmul's own shard

`_LMHead.forward` used to convert the matmul's width-sharded L1 output to
DRAM-interleaved *before* `tanh` and the scalar multiply. Moving the conversion to the
end puts both elementwise ops on the shard: 36.85 → 23.79 µs for the pair.

It is **not free in L1**, which the first version of this log and the README both got
wrong: `ttnn.tanh` and `ttnn.multiply` have no in-place form here, so the pair now
allocates two width-sharded L1 tensors instead of two DRAM-interleaved ones. Measured
(`bench/l1_highwater_probe.py`, `l1_highwater_probe.json`): peak L1 per bank goes from
90,112 to **217,088 B**, i.e. **+126,976 B** = 2 x 63,488 = 2 x (32 rows x 992 padded
columns x 2 B), leaving 1,238,144 B of the 1,455,232 B bank free at that peak. The
"7,296 B of headroom" figure quoted elsewhere in this port is a different pool (a
`TT_CCL`'s global semaphores) at a different moment (the decoder's tightest in-layer
point), so the two do not conflict — but "allocates nothing" was false and is now a
measured line item and limitation 8.

The one thing that needed checking rather than assuming is the padding.
`50688 / 52 = 975` columns per core is not a tile multiple, so each core is rounded to
992 and the shard set covers 51584 columns for 50688 real ones. `tanh` is bounded on
every input and the scalar multiply keeps it bounded, so no padded lane can produce an
inf or NaN — but that is an argument, so
`test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form` runs the head both ways on
the same hidden state and asserts `torch.equal` on the two logit tensors, plus finiteness
and `|logits| <= T`. Bit-identical, not PCC.

### 3.2 The decode embedding gather, into the boundary layout

`all_gather_async` takes an output `memory_config`, so the
`interleaved_to_sharded` that followed it can be the collective's own output layout.
Prefill is untouched — it needs the interleaved form, and its gather is the
chunk-and-clone reproducibility path of `EMBED_GATHER_CHUNK_ROWS`. One op (1.99 µs)
removed from the decode trace.

### 3.3 The SwiGLU multiply, after the op contract said no

The multiply carries the SFPU SiLU. The decoder stage had already measured folding it
into the gate matmul (+4.4 %, because the DRAM-sharded matmul's `SFPU_ACTIVATION` runs on
its 12 fixed workers), so the remaining lever is the core count.

**Attempt 1 — widen the matmul's output grid.** The `mlpN` arms of `bench/decode_ab.py`
move `mlp_gate`/`mlp_up`'s `cores` and re-derive `mlp_down`'s `in0_block_w`. Three of
four candidates failed immediately with an exact op contract:

```
TT_FATAL: in DRAM sharded Matmul we don't have support for un-even sharding
currently. K: 208, per_core_K: 11.
```

(`logs/decode_ab.log`, and the arm's own `decode_ab.json` entry, whose `error` field
keeps only the last line of the traceback — the fatal itself is in the log.)

`cores` must divide `K_tiles`. For gate/up that is `6656/32 = 208`
→ {8, 13, 16, 26, 52, 104}; `mlp_down` consumes their output and needs the same count to
divide `5120/32 = 160` → {8, 10, 16, 20, 32, 40, 80}. Intersection {8, 16}, and 16 ships.
`mlp8` was measured anyway: **+2.6 %**. So the family is closed, with the exact blocker
recorded rather than a first API error.

**Attempt 2 — reshard for the multiply only.** `DECODE_SWIGLU_MUL_CORES` reshards both
operands to a wider grid, multiplies, and reshards the product back so `mlp_down`'s
`in0_block_w=10` still holds. Swept 20/32/40/80 on the reduced build
(`decode_ab_swiglu.json`): 20 loses, then it improves monotonically to **−0.83 % at 80**,
which is the largest count dividing the 160-tile intermediate. Per-round spread
±0.0005 ms. In the shipped profile: **4.75 µs of multiply for 5.91 µs of reshard**
against 18.03 µs — net −7.4 µs per layer, ~−383 µs per step.

Every arm is PCC 1.000000 against the 16-core grid and picks the same token, which is
expected: resharding a tensor is not arithmetic.

### 3.4 Cumulatively, one invocation

`decode_ab.py --arms full_model_stage,terminal_only,base` (`decode_ab_shipped.json`),
traced logits-only on 2 layers: 1.5535 → 1.5376 → **1.5246** ms. Extrapolating the
per-step terms once and the per-layer term 26x predicts 22.81 ms on the 52-layer model;
the measured all-layer step is **22.656 ms**, 0.7 % better than predicted.

## 4. Candidates considered and not taken

| candidate | why not |
| --- | --- |
| pack `wqkv` + `attn_gate` (OPT-001: two projections consuming the same post-norm activation) | Both rows are the *best* DRAM utilisation in the layer — **77.12 %** (`wqkv`, id 3039) and **69.35 %** (`attn_gate`, id 3172) in this stage's `tracy/decode_sliding_perf_report.csv`, against 52.27–52.65 % for the MLP rows — so they are weight-bandwidth-bound and packing cannot reduce the bytes. And the two outputs need different downstream layouts (QKV is `sharded_to_interleaved` into `nlp_create_qkv_heads_decode`; the gate stays width-sharded until after SDPA), so a packed output needs an unshard, two slices and a reshard to split, which is ~4 µs against a ~3–5 µs saving. Recorded in the operation-topology audit with the DRAM% evidence rather than measured, because the byte argument is decisive: the packed matmul reads the same 16.3 MB. |
| wider RMSNorm grid, decode | The four decode norms must consume and produce the 16-core boundary spec, which is the inter-layer residual contract this stage is required to preserve. |
| sharded prefill terminal/embedding norms | They run on 1 and 4 cores for ~134 µs each because `ttnn.rms_norm` on a DRAM-interleaved input parallelises over tile rows and both see a 32-row slice. 0.27 ms of a 65 ms TTFT; the fix changes prefill numerics on the accuracy gates' critical path, so it is priced and left as limitation 9 rather than taken for 0.4 % of a figure whose process spread is 8 %. |
| `o_proj` OPT-011 narrower working shard | Kept declined, and **one of the decoder stage's three reasons no longer applies**: change 3 breaks the same single-grid invariant and adds three reshards, so "it costs a reshard and the invariant" is no longer a reason this stage can use. What survives is what decided it: the candidate won 0.11 % on `sliding`, was inside the noise on `full`, and cost 13 % of the multichip-vs-single-chip PCC headroom. Recorded as a candidate a decoder stage may revisit now that the invariant is already gone. |
| fewer decode collectives | Two all-reduces per layer is the replicated-residual contract. A fractured residual would halve the dispatch count, and the decoder stage's `fractured_decode_probe.py` owns that question. Out of scope here: the goal preserves the residual layout. |
| persistent CCL staging buffers | **Worth 14–17 % of the prefill reduce-scatter's host cost** at the model's BFP8/4-worker setting (~2 ms of TTFT), and blocked by the decoder stage's intermittent first-use correctness race. An earlier row here said they were "within noise" on host cost; that was the BF16 hot-loop arm and is withdrawn (§2.4). |
| prefill CCL implementation switch (async → wrapper) | The hypothesis was that the decoder stage chose `async` on *device* time at 8192 rows while short-prompt prefill is host bound, so a cheaper-to-issue wrapper might win at 128 rows. Measured at the model's BFP8 payload: 58.88 against 72.10 µs/call unloaded but 91.42 against 117.05 loaded, and the wrapper form also disables the fractured prefill norm. Refuted (§2.4). |
| lowering `max_top_k` from 32 to 8 | 0.7942 ms against the shipped 0.6323 (`sampler_ab.json`). Slower, and limitation 9 of the full-model stage notes the gathered width interacts with `num_gather_links`. |
| a broad datatype frontier search | `$datatype-sweep` owns it. The one precision question this stage asked is whether the 37.9 % roofline fraction is a precision problem, and it is not: 42.2 % of the layer is latency-bound non-matmul work and the projections already run at 52.27–77.12 % of peak. |

## 5. Re-measuring the floor

Change 3.3 makes the layer faster, so the layer-stack lower bound had to be re-derived
rather than inherited. Run with the decoder stage's own harness, at the decoder stage's
own context, so the number is comparable to the one it replaces
(`logs/layer_ab_after.log`):

| kind | before | after | PCC prefill / decode |
| --- | --- | --- | --- |
| sliding | 0.4546 | **0.4473** | 0.993700 / 0.993488 |
| full | 0.4238 | **0.4164** | 0.992220 / 0.992188 |

Floor: 39 x 0.4473 + 13 x 0.4164 = **22.858 ms/token**. The repeat control (`tp4b`)
reproduces both to 1e-4.

## 6. Evidence runs, in order

Each is one device job, one at a time, per `$tt-device-usage`.

| run | artifact | result |
| --- | --- | --- |
| TTFT by phase, 5 lengths, 52 layers | `ttft_breakdown_before.json` | prefill is 60.3 ms of a 64.8 ms window |
| prefill issue vs drain + cProfile | `prefill_host_probe.json` | 54.91 vs 55.08 ms |
| prefill host time by op, + a drained-collective pass | `prefill_opcount.json` | 4122 calls; RS 140.3 pipelined / 114.6 drained |
| collective arms, BFP8 / BFP8-loaded / BF16 | `ccl_host_probe_bfp8*.json`, `ccl_host_probe_bf16.json` | the loaded arm reproduces the in-model cost; persistent buffers −14/−17 % |
| traced prefill: capture, replay, correctness, DRAM | `prefill_trace_probe.json` | 59.80 → 44.96 ms, bit-identical, 98.16 ms capture, 3.3 MB |
| L1 high-water for change 1 | `l1_highwater_probe.json` | +126,976 B/bank, 1.24 MB/bank still free |
| decode A/B, terminal arms | `decode_ab.json` | −0.10 % logits-only; its `base` arm predates the SwiGLU default, so the cumulative table comes from `decode_ab_shipped.json` |
| decode A/B, SwiGLU grid | `decode_ab_swiglu.json` | 80 cores wins by 0.83 % |
| decode A/B, cumulative | `decode_ab_shipped.json` | −1.86 % on 2 layers |
| per-layer floor | `logs/layer_ab_after.log` | 0.4473 / 0.4164 |
| perf, baseline arm | `evidence_perf_before.json` | 23.844 / 23.164 ms, TTFT 65.41 |
| perf + shapes + 130073 | `evidence_perf.json` | **23.298 / 22.656 ms**, TTFT 63.68 |
| perf, `--prefill-trace` | `evidence_perf_prefill_trace.json` | **TTFT 49.76 ms**, decode unchanged |
| autoregressive, chat + raw | `evidence_autoregress.json` | coherent, non-degenerate |
| accuracy + sampling + fallback | `evidence_accuracy.json` | top-5 1.000, top-100 1.000 |
| fp32 control + misses | `evidence_fp32_gate.json`, `evidence_misses.json` | all four gates pass |
| greedy sampler benchmark | `sampler_ab.json` | split still wins, 15x |
| 59-case suite, forward | `test_results.xml` | 59 passed |
| 59-case suite, reverse | `logs/full_test_run_reverse.log` | 59 passed |
| qualitative, TT + compare | `qualitative/` | byte-identical to stage 6 |
| degeneracy gate | `logs/check_degenerate_output.log` | no degenerate output |
| watcher, shipped-default 10 cases | `watcher/`, `logs/run_watcher.log` | `WATCHER_CLEAN`, 0 tripped asserts |
| watcher, each opt-in prefill-trace case alone | `watcher_prefill_trace_{optin,rebind}/` | `WATCHER_CLEAN` each |
| watcher, the four release-probe arms | `logs/watcher_probe_*.log` | clean each |
| `o_proj` OPT-011, re-measured on this path | `logs/layer_ab_oproj.log` | −0.17 % sliding, PCC unchanged-to-better, declined with a specific reason |
| device reset after watcher | — | `RESET_DONE failures=0` |
| Tracy, 4 windows | `tracy/` | `TRACY_INTEGRITY_OK` |
| context contract | `../context_contract.json` | 131072, unreduced |
| figure gate | `bench/check_reported_figures.py` | `FIGURES_OK`, 0 failures |

**Provenance, per artifact.** This paragraph used to say that every row above post-dated the
last code change. That was true when it was written and stopped being true around round 7;
round 15 caught it and replaced it with a partition — and round 17 caught *that* partition
misfiling six artifacts into "current". Both times the failure was the same: a hand-maintained
classification of files that keep moving. Round 18 then showed that deriving it from `git log` is
itself ill-posed — a re-run that produces a **byte-identical** log changes nothing, so git keeps
the older commit and an artifact that *was* re-run looks stale. (That is exactly what happened
to `logs/mutate_figure_gate.log`, and the check round 17 added to enforce the partition failed
at the commit that introduced it.) So what the gate checks is not which commit last touched an
artifact but whether its **content agrees with the tree it describes**: the mutation log's
digests against the current mutation table, the junit file against the suite size both documents
state, the watcher verdict against the log it is re-derived from. The buckets below are what was
re-run, with git as the evidence where content did change:

* **Re-run at the reviewed commit** — `test_results.xml` and both suite logs, the gated watcher
  run, and both mutation-harness arms. Three of these are byte-identical to their previous
  contents (`logs/check_watcher.log`, `logs/mutate_figure_gate.log`,
  `logs/mutate_figure_gate_sweep.log`), which is why git dates them earlier and why the gate
  checks their content rather than their commit. `logs/prefill_trace_multibucket_probe.log` is
  round 16's and unchanged since.
* **Older than the last code change, and unaffected for a stated reason** — the perf arms
  (`evidence_perf{,_before}.json`, round 7), the prefill-trace arm (round 8),
  `perf_summary.json` (round 13), `invalidation_cost_probe.json` (round 10),
  `prefill_trace_probe_8192.json` and `doc/context_contract.json` (round 15), the accuracy,
  fp32-gate, autoregressive, miss, sampler, A/B, qualitative and Tracy artifacts (round 2's
  commit `3d03b5ca595`), and the three negative controls (rounds 4, 9, 11). Every commit since
  that touched `tt/*.py` or `models/common/sampling/generator.py` changed **trace lifecycle,
  reporting or a guard** — the release/retry/defer paths, a divisibility `raise`, the
  `deallocate` warning, a two-pass `set_kv_cache`, `capability_report` fields, the sampler's
  orphan list, the prefill-capture fallback — and none of them alters an op, a dtype, a layout
  or a program config, which is what those artifacts measure.
* **The one intervening change that is *not* invisible to a perf number**, named rather than
  swept in: round 9 made `_invalidate_traces_if_cache_moved()` unconditional on
  `decode_forward` and added it to `generate()`. That is per-call host work the perf arms did
  not pay. It is measured — 7.57 µs with a trace live, 0.033 % of the token-out step, and the
  reported figure comes from `generate()`'s own loop, which calls `_decode_step_traced`
  directly and pays it once per `generate()` rather than per token
  ([`invalidation_cost_probe.json`](invalidation_cost_probe.json)).
* **Diagnostic, and unchanged by any of it** — `ttft_breakdown_before.json`,
  `prefill_host_probe.json`, `prefill_opcount.json`, `ccl_host_probe_*.json`,
  `prefill_trace_probe.json`, `l1_highwater_probe.json` and the three `decode_ab*.json`.

One consequence is worth naming rather than leaving to inference: `evidence_accuracy.json`'s
`capacity` block predates round 7, so it carries none of the three decode-path flags that block
gained in that round. The flags were the module defaults at that commit, so the values it would
have recorded are the shipped ones — but that is an inference, and the three perf arms carry the
flags explicitly, which is where the carried-forward-contract table's provenance now points.

## 7. Three harness problems worth recording

**The autoregressive stage needs a `prompts/` directory next to its evidence.**
`bench/evidence.py` here is the full-model stage's script with `OUT` repointed, and
`stage_autoregress` reads `OUT / "prompts/autoregressive_chat_prompt.txt"`. The first
run of `--stages capacity,perf,shapes,autoregress` therefore crashed *after* writing the
perf and shapes results (`FileNotFoundError`, `logs/evidence_perf.log`). Copying the
prompt file across and re-running `--stages autoregress` on its own produced
`evidence_autoregress.json`.

*Superseded by the round-2 re-run, and round 3 was right to catch that this paragraph no
longer described the committed file.* When round 2 found the README/`perf_summary.json`
figure contradictions, every evidence artifact was re-run from scratch, so the committed
`evidence_perf.json` is **not** the crashed run's output: `logs/evidence_perf.log` is a
clean `capacity → perf → shapes → summary` run with no traceback, and the file's `stages`
key is `["capacity","perf","shapes"]`. The README's reproduce command was still the
original `--stages capacity,perf,shapes,autoregress`, which does not reproduce that file;
it now reads `--stages capacity,perf,shapes`, with `--stages autoregress` as its own line.
The lesson stands and is why the directory is copied up front now — the paragraph is kept
for it, not for the artifact description.

**`run_watcher.sh` looked for the watcher log in the wrong place, twice.** `TT_METAL_LOGS_PATH` puts the log at `$LOGS/generated/watcher/watcher.log`, not `$LOGS/watcher.log`, so the script's own verdict step printed `missing watcher log` after nine passing tests (visible at the foot of `logs/run_watcher.log`). The path is fixed in the script and the verdict was re-derived from the committed log into `logs/check_watcher.log`: `fatal watcher messages: 0`, `WATCHER_CLEAN`. Caught by `bench/check_reported_figures.py`, which asserts the verdict string is *in a log*, not in prose — which is the whole point of having that gate.

**The decode Tracy window overflowed the profiler's marker buffer.** The 80-core SwiGLU
multiply emits markers per core, and the full-model stage's two-layer decode window
dropped 20 marker lines at `ITERS=1` — which the inherited integrity check caught and
failed the run for, as designed
(`logs/run_tracy_two_layer_overflow.log`, preserved). Splitting the decode capture into
one window per layer kind (`--layers 0` and `--layers 3`) is the same coverage with half
the markers and passes at 0 dropped lines. It also makes the per-kind layer cost
directly readable, which is what `perf_summary.json`'s device-time term is built from.

## 8. Performance accounting

`perf_summary.json`. Roofline **8.829 ms/token** (4,520,382,464 B/device ÷ 512 GB/s,
where the bandwidth is back-derived from `tt-perf-report`'s own DRAM% columns rather than
a data sheet), device time **23.099 ms/token**, end-to-end token-out **23.298 ms/token**,
end-to-end logits-only **22.656 ms/token**.

Two gaps, both named:

* **roofline to device (37.9 %)** — 184.20 µs of a 436.602 µs sliding layer is non-matmul
  work that moves no weight bytes: 4 norms, `SdpaDecode`, two collectives, two cache
  updates, rotary, head create/concat, layout conversions. The projections themselves run
  at 52.27–77.12 % of peak. This is the "modules built from many small ops sit lower" case, and
  the explanation is the requirement;
* **device to end-to-end (−2.0 %)** — device time comes out *above* the traced replay,
  because `tt-perf-report` merges a 4-device capture by taking the max per op. The sign
  is the useful part: there is no room for host work between them, which the zero
  per-token refresh counters say independently.

`22.656 + 0.632 = 23.288` against a measured token-out of `23.298`: the two traces
account for the step to within 9.9 µs, and that 9.9 µs is the caller's token readback.
(Round 8 found this paragraph still carrying the pre-regeneration arithmetic and its
larger residual; the figures are this run's, and the README's accounting section states
the same three numbers.)

## 9. The figure gate

`bench/check_reported_figures.py` resolves its checks' worth of figures in `README.md` against the
committed artifacts: the before/after perf rows and their percentage deltas, the
layer-stack floor and the `layer_ab` log it comes from, every A/B arm and its PCC, the
named `tt-perf-report` op ids in both this stage's and the previous stage's captures,
the implied 512 GB/s on every DRAM-classified row, the dtype/fidelity cell of each
dominant matmul, the four accuracy gates, the fallback counters, the split-sampling
contract fields, the sampler arms, the host-dispatch tables, the qualitative diff, the
watcher verdict, both test runs, the Tracy integrity check and the context contract.
It also covers the traced-prefill and L1 measurements, the BFP8/loaded collective arms
and the in-model drained-collective pass. It writes nothing and needs no hardware.
`FIGURES_OK` with zero failures, and it asserts its own check count so the count cannot
drift. Round 3 was right that a frozen *count* is not coverage, so it now also asserts
that it opens each artifact a newly-unchecked section would need — the prefill-128
capture, `work_log.md`, `perf_summary.json` and the watcher logs among them — and the
two families of check that asserted nothing (`close(v, v)` and `same(x, True, True)`)
are now `bind(...)`, which registers a literal for the README cross-check and says so.

Its own findings are recorded above rather than quietly fixed: the residual-add figure
was 1.86 where the artifact says 1.884, the DRAM bandwidth rows were quoted from the
*previous* stage's capture, and the watcher verdict was in prose because the script that
should have printed it looked in the wrong directory. It is also what makes a full
figure refresh after a late code change affordable: every number in `README.md` was
re-pointed at the re-run artifacts and the gate found the four that had been missed.

## 10. Stage review

### Round 1 — `more-work-needed`

An independent xhigh reviewer read the stage against the goal contract and `$optimize`,
re-derived the device-time reconciliation, the roofline, all three change deltas against
both stages' CSVs, the A/B ladder, the `mlpN` blocker, the layer floor, the dtype rows,
the accuracy gates, both test orders, the watcher verdict and the qualitative diff — and
found all of those correct. It returned four required items, and every one of them was a
real defect:

| finding | what was wrong | what was done |
| --- | --- | --- |
| **P1** the host-dispatch gap was declined on an *unmeasured* cost model, and TTFT did not improve | the stage named a prefill trace as the only mechanism and never captured one; the two inputs to the "does not pay back" argument were estimates | captured it (`bench/prefill_trace_probe.py`): **59.80 → 44.96 ms, bit-identical, 98.16 ms capture, 3.3 MB retained, coexists with the decode traces**, then **shipped it** as `GeneratorConfig.prefill_trace` with a bounded bucket cache and a contract test. TTFT **63.68 → 49.76 ms, −21.9 %** (§2.5) |
| **P2** the host attribution had a 2x hole and a contradiction | the collective probe used a **BF16** payload in a hot loop and reproduced 60 µs against the model's 125–140; `perf_summary.json` said 12.1 ms where the README said 19.1 | re-measured at the model's **BFP8** payload, with a loaded queue, and with the device drained before each in-model collective: **117.05 µs loaded against 114.6 in-model drained** — the gap is the instruction stream, nothing is unattributed. `perf_summary.json` and the README now both say 20.93 ms, and the retracted "nothing moves it" is replaced by the measured **−14/−17 % from persistent buffers**, blocked on the decoder stage's correctness race (§2.4) |
| **P2** the teacher-forcing "after" range dropped this stage's own lowest measurement | the README quoted 37.51–38.50 while `evidence_accuracy.json` said 37.19 | all three runs reported, the ranges noted as overlapping, and **the +1.3 % claim withdrawn** |
| **P2** "none of the three changes allocates anything" was false for L1 | `ttnn.tanh`/`ttnn.multiply` have no in-place form, so change 1 moves two 3.24 MB transients into width-sharded L1 | measured (`bench/l1_highwater_probe.py`): **+126,976 B/bank of peak L1**, 1.24 MB/bank still free there; the sentence corrected, the pool/moment confusion with the 7,296 B semaphore figure explained, and it is now limitation 8 |

Its other concerns were all taken as well: the LM-head share was **26 % where it is
2.6 %**; the `in0_block_w` L1 blocker cited 3 where the evidence says 4; the `o_proj`
OPT-011 re-decline leaned on a single-grid invariant that change 3 itself breaks; the
layer-floor table read as one measurement when its `before` column is cross-stage; the
TTFT phase table mixes per-phase minima across rounds; the 205 µs `Embeddings` op-to-op
gap in a *traced* window needed its window-boundary classification restated rather than
inherited; the five `SLOW` matmul rows needed one line saying the missing output-subblock
field is structural to the DRAM-sharded program config; `decode_ab.json` was missing from
the Artifacts table; and the new embedding-gather test asserted only the memory config,
where the failure mode it guards against is intermittent — it now compares **values**
against the interleaved gather, over four repeats, at three token ids.

One concern was answered rather than changed: `evidence_perf.json` had been hand-edited
to drop an empty `autoregressive: {}` key. Every evidence file has since been re-run from
scratch on the final code, so none of them is hand-edited any more.

### Round 2 — `more-work-needed`

A second independent reviewer confirmed the round-1 remediations (the prefill trace really
was captured, measured and shipped; the CCL attribution really was re-measured at BFP8
with a loaded queue and an in-model drained pass; the teacher-forcing claim was withdrawn;
the L1 delta was measured) and independently re-derived the device-time arithmetic, the
roofline, all three change deltas against both stages' CSVs, the accuracy gates, both test
orders and the non-tile-aligned-bucket row arithmetic. It then returned six items, and one
of them was a bug in shipped code.

| finding | what was wrong | what was done |
| --- | --- | --- |
| **P1** `prefill_trace=True` regressed the serving API by ~83 ms/request | `prefill_forward(kv_cache=…)` released the traces on **every** call with a non-`None` cache, so a caller threading the *same* handles recaptured per request — for the caller the flag is advertised for | invalidate on cache **identity** (`_kv_cache_signature`, buffer addresses) instead, and on a real move **retire** prefill tracing rather than recapture. `test_prefill_trace_survives_rebinding_the_same_external_cache` pins both halves |
| **P1** README/`perf_summary.json` figures still contradicted the artifacts | the gate asserted "the artifact says X" for ~170 literals but "and the README says X" for only ~20, so `169 vs 195`, `19 %`, five per-round spreads, three DRAM rows, `49-case`, `three new tests`, `19 µs` and `44 %` all survived a 195/195 pass | every figure re-pointed at the artifacts (the per-round spreads and the DRAM rows generated *from* them, not transcribed); the gate now cross-checks **every** literal it resolves against the README text and asserts its own advertised check count |
| **P1** the 311 µs op-to-op gap was "classified" with wrong numbers and a refuted mechanism | the paragraph quoted 205/202 µs from the *superseded* profile and called id 3145 the window's first op; it is row 29 of 55, preceded by 27 sub-1.5 µs gaps | restated from this capture: 310.959 µs gap, 307 µs advice, 358 µs window total, mechanism = the inter-replay boundary of a one-replay signposted window, bounded by the device-vs-e2e arithmetic |
| **P2** `o_proj` OPT-011 was re-declined on the decoder stage's own weakest, uncommitted datum | this stage voided two of that stage's three reasons by shipping change 3 | **re-measured on this stage's path** (`logs/layer_ab_oproj.log`): sliding 0.4467 vs 0.4474/0.4475 (−0.17 %), full inside the repeat control, HF PCC unchanged-to-better, worth 0.12 % of the step. Declined with a specific reason: adopting it moves the *decoder stage's* default and its multichip-vs-single-chip gate, which this harness does not run |
| **P2** the contract did not record what this stage allocates | `prefill_trace`'s retained DRAM and the +126,976 B/bank L1 peak were absent; `notes` was still attributed to the full-model stage | both added, generated from their own measured artifacts by `bench/refresh_context_contract.py` so they cannot drift, with the `max_entries x padded_rows` bound stated and the attribution fixed |
| **P2** the five `SLOW` row ids were wrong | 3182/3187/3194 are not SLOW; the real set is 3065/3071/3127/3132/3139, and the omitted one was the LM head at 40.8 % of the window | corrected, and the LM head named |

Its other concerns were taken too: the seven `peak DRAM implied by <id>` checks were
tautological (the tool computes `DRAM %` as `DRAM / peak`) and are replaced by one
"single assumed peak across every row" check plus four non-tautological row assertions,
with the README relabelled as a consistency check rather than an independent derivation;
the L1 headroom sentence now says the 1.24 MB is free-*in-isolation*; the `decode_ab.json`
artifact row says its `error` field holds only the last traceback line; and the
allocator's *"Allocating device buffers is unsafe due to the existence of an active
trace"* warning is now classified in the README rather than left in every log unremarked.

### Round 2's own finding: the watcher caught a use-after-free, and then a fabric hazard

Adding the cache-rebind test to the watcher list stopped the device:

```
Device 0 acteth core(x=0,y=9) virtual(x=29,y=25): subordinate_erisc detected invalid NOC
command buffer state before starting the next kernel ... fabric_erisc_router.cpp
```

Two separate causes, found by bisecting rather than by guessing.

**The first was in the test.** Its cleanup freed the cloned KV cache *before* releasing
the prefill trace that held those buffers' addresses — a use-after-free, which this class
of hardware reports as an ERISC assert rather than as a wrong number. Reordering the
`finally` fixed it (`logs/watcher_bisect_rebind.log` before,
`logs/watcher_bisect_rebind_fixed.log` after).

**The second is below the model and is not fixed.** With the test corrected, the assert
still fires when the *suite* runs, and `bench/prefill_trace_release_probe.py` localises it:
capture, release, recapture, and cloning/freeing the 104 cache tensors are each
watcher-clean in their own process, and so is each opt-in test on its own — what trips it
is releasing a prefill trace and then building and running **another** model on the same
mesh in the same process. The shipped default never captures a prefill trace, so it never
releases one, and the ten-case default watcher run is `WATCHER_CLEAN` with zero tripped
asserts. The opt-in path now retires prefill tracing after the one release a cache move
forces, which is the smallest exposure that still serves its caller. Recorded as
limitation 6 with the four-arm bisect.

A first fix attempt — draining the device before releasing — did **not** help, and that
is recorded rather than quietly dropped: the synchronisation is kept because freeing
buffers a trace referenced without draining is wrong on its own terms, but it is not what
this assert is about.

### Round 3 — `more-work-needed`

Two P1s and six smaller findings. Both P1s were right, and the first one was right in a
way that changed the stage's conclusion rather than just its prose.

| finding | what it said | what was done |
| --- | --- | --- |
| **P1** the fabric-ERISC hazard had **no positive reproduction** | the four arms of `prefill_trace_release_probe.py` are all *negative* controls — none builds a second model after a release — so a clean result from them localises nothing, and three shipped decisions (the retirement flag, the watcher exclusion, limitation 6) rested on an unreproduced claim | the runs that could show it were made. **`--arm rebuild`** was added — release, then build and run a second generator on the same mesh — and is `WATCHER_CLEAN`; the **two opt-in cases together in one process** are `WATCHER_CLEAN`; the **ten default cases** are `WATCHER_CLEAN`; **all twelve in one process trip the assert at teardown, after all twelve pass**. So the hazard is real, its earlier *statement* was wrong, and the retirement flag was never a mitigation for it. See below |
| **P1** figures in the sections the gate does not read do not reconcile | prefill-128 ids `3884`/`3581` and `134.65` exist in no artifact (they are `3886`/`3579`, `133.868`/`133.979`); window `2608.4` vs `2606.3`; residual op row `655`/`8.14` vs `707`/`7.86`; audit rows `41.8`/`255`/`55.5` vs `40.77`/`252.41`/`54.89`; two audit rows silently from the *previous* stage's CSV; teacher-forcing files swapped | every figure corrected against its artifact. The audit table is now a **partition**: an `ids` column, 14 groups, every one of the CSV's 55 rows used exactly once, and the column summing to the window's 1122.551 µs. Pre-change values are labelled as such with their own ids in the previous stage's capture. The gate now re-derives all of it, including the partition invariants |
| P2 `perf_summary.json:bandwidth_source` still quoted the previous stage's QKV row | `371 GB/s = 72.4 %` | rewritten to this stage's rows (3139: 279.38 / 54.5666; 3039: 394.85 / 77.1192), and the gate now asserts the string names them |
| P2 `evidence_perf_before.json`'s floor claimed a log it is not in | `layer_ab_after.log` contains only the after-arms | `bench/evidence.py` now emits an arm-specific provenance string, and the baseline arm was re-run to regenerate the artifact |
| P2 work log §7 no longer described the committed `evidence_perf.json` | the round-2 re-run replaced the crashed run's output | paragraph corrected; the README's reproduce command now matches the file's `stages` |
| P2 `LM_HEAD_SOFTCAP_IN_L1` cited a test name that does not exist | `test_lm_head_softcap_layout_is_equivalent` | corrected to the real name |
| P2 the rebind test's `finally` release had become a no-op | retirement meant no trace existed by then, so the test no longer covered the ordering it documents | removing retirement makes it live again, and the test now asserts a trace over the moved cache exists at that point |
| P2 work log §4 quoted DRAM % matching no capture (`73.7 %`) | — | corrected to 77.12 / 69.35 with the row ids, and the gate now reads `work_log.md` |

**Hard-check gaps, also from round 3.** The gate passed 328/328 while every figure above
was wrong, which is a fair description of a gate that is not measuring what it advertises.
Three things changed. The 16 `perf(...)` calls were `close(name, round(v,d), round(v,d))`
— got equals want by construction — and the 14 `same(..., True, True)` calls asserted
nothing either; both are now `bind(...)`, which registers the literal for the README
cross-check and *says* that is all it does. The README search was a plain substring match,
so `1.0` matched `21.05`; it is now digit-bounded, which immediately exposed five bindings
that had been passing on a coincidence. And `ADVERTISED_CHECKS` froze the count rather
than the coverage, so the gate now also asserts that it opens the specific artifacts a new
unchecked section would need — `prefill_128_perf_report.csv`, `work_log.md`,
`perf_summary.json` and the two new watcher logs among them.

### Round 3's own finding, and how round 4 overturned it

Round 3 asked for a positive control. Running one produced a result that contradicted the
original claim *and* the retraction that first replaced it, so round 3 concluded: the
assert is real, "release then build another model" is not the trigger, and the trigger is
the opt-in cases **plus** the larger workload.

Round 4 rejected that too, and correctly: every configuration that tripped had twelve
cases and every clean one had ten, so the attribution to the prefill-trace cases was
confounded with process length. The missing control — twelve cases *without* those two —
was then run three times, along with three more ten-case runs. Fifteen watcher processes
in total:

| configuration | runs | tripped |
| --- | --- | --- |
| the two opt-in cases alone in one process | 1 | 0 |
| `--arm rebuild` (release, then build and run a second generator) | 1 | 0 |
| the ten gated cases | 5 | 0 |
| twelve: the ten + two other sampling cases | 3 | **1** |
| twelve: the ten + both opt-in cases | 3 | **3** |

The length control trips. So the round-3 statement is retired in turn, and what the data
supports is narrower than either previous claim: **the trip rate rises with the number of
device cases in one process** (0 of 5 at ten, 4 of 6 at twelve, Fisher p = 0.061), and the
opt-in pair may or may not make it worse (3 of 3 against 1 of 3, p = 0.400 — not separable
at this n). Both p-values are computed by the figure gate from the committed logs, not
typed into the document.

Three consequences:

* the gated set is ten **because ten is the largest size measured with zero trips**, which
  is an empirical scope decision rather than an attribution to the prefill-trace path;
* the operational advice changed. "Reset the devices between builds if you use
  `prefill_trace`" followed from the retracted attribution; what the evidence actually
  supports is "a longer watcher run of this module is expected to abort at teardown
  sometimes — reset and re-run, and never read a truncated watcher log as a clean one";
* it is very likely the same fault as inherited limitation 7 (the teardown timeout on
  ethernet core 29-25): same teardown, same acteth cores, watcher-only, and this stage
  root-causes neither.

Three statements of one limitation, two of them wrong, is the actual lesson: each was
built on the largest control set available at the time, and each fell to the first
control that had been left out. The bisect arms were negative controls; the round-3
argument had no length control; only the fourth version has both.

### Round 4 — `more-work-needed`

| finding | what was done |
| --- | --- |
| **P1** the **decode** trace bakes the same KV-cache buffer addresses as a prefill trace and was never invalidated on a rebind | real, and the worse of the two: the prefill trace is opt-in and off by default while the decode trace runs on every token. `ttnn_decode_forward` calls `paged_update_cache(layer.k_cache, layer.v_cache, ...)`, so a caller that rebound to different buffers after capture got a decode reading and writing the buffers it no longer owned — wrong tokens, no error, and a log line about releasing the *prefill* traces that read as if the rebind had been handled. `_invalidate_prefill_traces_if_cache_moved` is now `_invalidate_traces_if_cache_moved`, each trace carries its own cache signature, and `_release_decode_trace()` drops the decode trace and the sampling trace captured over its logits. New test `test_decode_follows_the_cache_it_is_rebound_to_after_the_trace_is_captured` prefills prompt A, decodes (capturing), rebinds to a different cache holding prompt B, and asserts the **traced** decode agrees with the **eager** decode off B. *Round 5 corrected the claim made for it*: the committed negative control fails on the **release** assertion, not the traced-vs-eager one, so the release check is the demonstrated discriminator and the value check is a corroborating consistency check against a failure mode nothing here exhibits (a release that happened but recaptured against the wrong buffers). The control is also a *partial* revert rather than `5e6022db622` — signature recorded, comparison and release removed — because a full revert fails on a missing attribute instead of on the behaviour |
| **P2** the fabric-ERISC attribution was confounded with process length | the control was run three times; see above. Limitation 6, the watcher section, `run_watcher.sh`'s header and the serving advice are all rewritten around it |
| **P2** `doc/context_contract.json` carried this stage's numbers under `doc/full_model/...` provenance | the parent builder hardcodes both strings, so this stage's wrapper now re-points them from `EVIDENCE` after the parent runs, and the contract is regenerated. Same defect class as rounds 2 and 3 found in `perf_summary.json` and `evidence_perf_before.json` |
| **P2** the `$optimize` checklist row said the two traces account for the step to **38 µs** | no artifact yields 38; it is **27** (26.8 µs). Corrected, and the gate now binds the residual rather than only asserting a band |
| the gate's "asserts the set of artifacts it opens" was `is_file()` | the three readers now record every path they open and the coverage assertion tests *that* set. Two entries were caught immediately: `work_log.md` was being read past the recorder, and `watcher_probe_rebuild/watcher.log.gz` was listed but never opened — it is now read and checked for detach lines |
| `ADVERTISED_CHECKS` counted bindings as checks | the split is stated, and both numbers are asserted (**583 checks = 549 assertions + 34 README bindings** as of round 5) |
| the audit table called id 3119 a residual add | it is the attention-gate multiply; 3124 and 3191 are the residual adds. No value moves |
| the README read as if the profile pinned the SwiGLU grid | `Cores` reads 110 for every elementwise row, including the 52-core softcap pair. Footnoted, with what does pin it |
| `device_position_advances: 0` read as a contradiction | explained where it is incremented and why replays cannot bump it |
| the reproduce block's `WATCHER_TAG=_12case` matched no committed log | tags aligned to the committed names, and the ten-case repeats and the length-control arm added to the block |

One thing was tightened without being asked: `teardown()` had its own inline copy of the decode/sampling release, so after the P1 fix there were two decode-trace lifecycles to keep in step — which is the shape of the bug round 4 had just found. Both callers now go through `_release_decode_trace()`. The README also gained a full inventory of what each of the three captured graphs bakes in and who owns it, which is the check that answers "is there another one of these?": the KV cache is the only caller-owned device state any trace holds, so one signature per trace over its buffer addresses covers the hazard class rather than one instance of it.

### Round 5 — `more-work-needed`

The round-4 fix had left a second copy of the trace lifecycle behind, and the review found
it plus three presentation defects. It also made the sharpest methodological point of the
five rounds, on limitation 6.

| finding | what was done |
| --- | --- |
| **P1** `teardown()` never adopted `_release_decode_trace()`, and diverged from it four ways: no `_trace_logits` deallocate (3.24 MB/device left to GC), no `_sampling_captured` clear, decode released *before* the sampling trace captured over its logits, and — on the shipped default, where there are no prefill traces — `release_trace` reached with **no drain at all**. Worse, the module-scoped test fixture had no finalizer, so its decode and sampling traces were live when the mesh closed | one release path with an unconditional drain; a fixture finalizer; `close_multichip_mesh` now also drops `tt.model._MODEL_CCL_SEMAPHORES` (same `id()`-reuse hazard `_kv_cache_signature` rejects by name); `MuseGlimmerModel.deallocate()` warns instead of silently freeing a cache under a live trace, via a counter the generator maintains at all four capture/release sites. And the point that mattered: **limitation 6's statistic had been measured in the pre-fix configuration**, so both twelve-case arms were re-measured against the fixed one — see below |
| **P2** the negative control does not support the claim made for it: it fails on the *release* assertion, not the traced-vs-eager one, and it was produced against a partial revert (signature recorded, comparison removed) rather than `5e6022db622` | the test docstring and this log now say what the README already said. The release check is the demonstrated discriminator; the value check is a corroborating check against a failure mode nothing here exhibits. The control is labelled a partial revert, with the reason (a full revert fails on a missing attribute, not on behaviour) |
| **P2** the contract provenance fix was partial in the same way round 4 flagged: `tested.commands` still named the previous stage's harness for all five commands, and `tested.prefill_misses.note` its miss file | re-pointed by substitution over the whole `tested` subtree, so a *future* field cannot be missed the way these two were; the gate asserts no current-stage field contains `doc/full_model/` while the historical per-stage entries keep theirs |
| **P2** the statistics: quoting only the pooled 12-vs-10 contrast is circular, since three of its four trips are the opt-in arm's | all three pairwise contrasts are now in the README with the pooled one, and the gate computes each: pooled **p = 0.061**, opt-in-vs-ten **p = 0.018**, control-vs-ten **p = 0.375**, opt-in-vs-control **p = 0.400 against a minimum attainable 0.100**. The paragraph now says plainly that the strongest contrast supports the hypothesis the stage withdrew, that the composition-matched arm shows no length signal, that three post-hoc contrasts carry no multiplicity correction, and that **length and composition cannot be separated by this design** |
| **P2** the gate's new coverage set was satisfiable by a bare `text()` call, and the file contained one | both bare reads replaced by content assertions; every covered path is now asserted on |
| "fifteen processes" against a breakdown totalling eleven | corrected to eleven, with the two bracketing runs named as such |
| the two-trace residual reads 26 µs from the printed figures and 27 in the prose | both stated, with the unrounded 26.8 and where it comes from |
| the `Cores` footnote covered only elementwise rows | extended: the column reads 12 for every DRAM-sharded matmul row too, where the geometry is `cores=52`. The `SLOW` paragraph no longer claims those rows expose core count |
| a gate check labelled "three trace capture sites" asserted 2 | relabelled to what it asserts: this port captures two of the three itself |

### Round 5's own finding: the attribution was measured in a configuration that no longer exists

The review's best point was not that `teardown()` was wrong — it was that limitation 6's
eleven-process statistic had been gathered *with* that wrong teardown, and with a test
fixture that closed the mesh over live traces. Both are exactly the shape of fault the
assert reports (a device buffer freed while a trace holds its address), and neither had ever
been tested. So the arms were re-run against the fixed tree, and the tree was frozen first:
mid-experiment edits during round 5 had made runs 1 and 2–3 incomparable, that set was
discarded, and the whole thing was re-run on one unchanging checkout.

**Neither arm moved**: 3/3 to 3/3 and 1/3 to 1/3, Fisher p = 1.000 on both. So the in-model
mechanisms were real defects worth fixing on their own terms, and they are not this fault —
which is what turns "below the model" from an assumption into a tested conclusion.

The re-run also doubled each arm to six, and *that* changed the answer. Combined, the opt-in
arm is **6/6** and the length-matched control **2/6** against **0/5** at ten cases:
opt-in-vs-ten **p = 0.0022**, control-vs-ten **p = 0.455**, opt-in-vs-control **p = 0.061**.
Composition separates; length does not. Which means round 4's retraction of the round-3
statement was itself an artifact of three runs per arm, and the fifth statement of this
limitation is close to the second. That is the lesson of the whole sequence and it is
recorded as such in the README: each version fell to the first control left out of it, and
two of the five fell to nothing more than insufficient n.

### Round 6 — `more-work-needed`

The P1 was the worst kind of record defect: **limitation 6 still contained the round-4 text
that round 5 had retracted**, so the shipped document stated a conclusion and its withdrawal
as current fact in two different sections. Worse, the figure gate *required* the stale
paragraph — the pre-fix Fisher checks bound `p = 0.400`, which appeared nowhere else — so the
mechanism meant to prevent exactly this was holding it in place. Both are fixed: limitation 6
is rewritten from the current data, the pre-fix contrasts are gone from the gate, and the gate
now slices the README into sections and asserts that limitation 6 and the Watcher section
agree on every arm tally and on the primary contrast.

| finding | what was done |
| --- | --- |
| **P1** limitation 6 was the retracted round-4 statement; the gate required it | rewritten; superseded phrases asserted *absent*; cross-section agreement asserted |
| **P2** the reversal was over-claimed: the one significant contrast was confounded in both dimensions, the isolating ones were not significant, the multiplicity note had been deleted, and the "length control" matched case *count* but not *work* | **two new arms run.** A **work-matched** twelve-case arm whose extra case builds its own `reuse=False` generator, clones and frees all 104 cache tensors and captures *and releases* a trace — a **decode** trace, not a prefill one: **0 of 3**. And the **opt-in pair alone**, repeated to **0 of 4**. With those, the conclusion is an interaction rather than a main effect, and it is stated with the multiplicity and independence caveats restored |
| **P2** "the fix moved neither arm … **excludes** the in-model mechanisms" rested on two 3-vs-3 tables whose minimum attainable p is 0.100 | replaced with what the data bounds: four in-model candidates changed and the rate did not move, so the mechanism is not among them — which is weaker than "excludes" and is what the runs support |
| **P2** the closure statement called the ctx-256 floor "not conservative" | it is a *comparator*, not a bound, and the document now proves it with its own numbers: 22.421 + 0.686 = 23.107 ms for bare layers plus the terminal term, against a 22.656 ms measured step that also contains the terminal path — the real step beats the sum of its separately-measured parts by **0.451 ms**, so the per-layer harness overprices by ~2 %. The contract's floor-plus-terminal comparison is stated alongside, and `perf_summary.json` now carries the ctx-256 comparator |
| **P2** the round-5 contract substitution broke two things it touched: it rewrote the `qualitative.py --arm hf` command this stage deliberately did **not** run, and an `evidence_misses_*.json` glob that matches nothing here | the blanket substitution is replaced by explicit per-command handling; the HF arm is attributed to the stage that ran it with the reason; the miss note is derived from the file that exists. The gate now **resolves** contract-referenced paths instead of pattern-matching the string, and the "no `doc/full_model/`" rule carries a named exception rather than being dropped |
| **P2** the trace-counter guard over-decremented on a failed release and had no behavioural coverage | decrement only on a successful release — otherwise the counter reads zero while a trace is alive, silencing the guard in exactly the case it exists for. New test `test_the_live_trace_count_round_trips_over_both_trace_kinds` drives both kinds through capture and release and pins the round-trip and the clamp |
| three stale suite sizes | corrected against `test_results.xml`; the suite ended round 6 at **55** cases with nine new, and round 7 found a fourth stale `54` the round-6 sweep missed |
| `DECODE_SWIGLU_MUL_CORES` had no divisibility guard | asserted where it is relied on: 80 divides *this* checkpoint's 160-tile width, which is a property of (intermediate_size, tp), not a constant |
| `trace_counter_smoke.log` was referenced nowhere | removed; the new test supersedes it |
| the "partial revert" label and the historical `ttft_breakdown_before.json` name | both stated where a reader meets them |

### Round 6's own finding: the sixth statement is the third one, with the controls it lacked

Five arms, 24 processes (25 with the single `--arm rebuild` run, which is not one of the five):

| configuration | runs | tripped |
| --- | --- | --- |
| ten gated cases | 5 | 0 |
| the opt-in pair **alone** | 4 | 0 |
| twelve: ten + two sampling cases (count-matched) | 6 | 2 |
| twelve: ten + `decode_follows_the_cache…` + a sampling case (**work-matched**) | 3 | 0 |
| **twelve: ten + both opt-in `prefill_trace` cases** | 6 | **6** |

> **Superseded by round 7, below** — and by README limitation 6, which is the current
> statement. The round-7 review found that the count-matched control arm (twelve cases, no
> prefill trace) trips **2 of 6**, so a preceding workload *alone* is sometimes sufficient
> and the interaction claim in the paragraph below is not what the arms support. It is left
> here as the record of what round 6 concluded, not as a live claim. The current reading is
> **the opt-in pair takes the rate from a 0–33 % background to 100 %**, directional, with the
> mechanism unavailable from this design.

Neither half reproduces alone (round 6's reading; see the note above). The pair by itself
is 0 of 4; a twelve-case process that builds an extra generator, clones and frees the whole
cache and captures and releases a *decode* trace is 0 of 3. Together: 6 of 6, against 2 of
18 for everything else pooled (p = 0.00021, six post-hoc contrasts, no multiplicity
correction).

So the sixth statement is close to the third, which round 4 retracted. That is not a
vindication of the third — it was underpowered and its pooling was circular — but it is worth
naming what actually happened across six rounds: **every statement fell to one control that
had been left out, and two of the six fell to nothing more than n = 3.** The arms that finally
separated it (hold work constant; run the suspect pair alone) each cost about twelve minutes
and could have been run at round 3.

### Round 7 — `more-work-needed`

The P1 was round 6's P1 recurring **inside the commit that fixed it**: a figure that resolves
to nothing, held in place by the gate. "Twenty-eight watcher processes" — the arms sum to
**24** — was asserted by a bare string-presence check on the *word*, which the digit-bounded
literal binder cannot see. The reviewer mutation-tested it: correcting the word to the
derivable value made the gate fail. The count is now summed from the parsed arm table.

| finding | what was done |
| --- | --- |
| **P1** an unsupported process count enforced by the gate | derived from the table's `runs` column; the word-matching assertion deleted |
| **P2** the five-arm conclusion asserted an interaction its own control arm falsifies | the count-matched twelve — twelve cases, no prefill trace, no extra generator, no cache churn — trips **2 of 6**, so "a preceding workload alone is not sufficient" was false and "0 of 3 excludes process length, generator churn and trace lifecycle" was an overclaim from an underpowered arm (0/3 vs 2/6 is p = 0.500). Restated as what it is: **the opt-in pair takes the rate from a 0–33 % background to 100 %**, directional and well supported, with the mechanism unavailable from this design. Limitation 6 now names the control's 2/6 instead of hiding it inside the pooled 2/18, and the pool's heterogeneity and its one excluded row are stated |
| **P2** the cross-section gate was defeatable three ways, all demonstrated | the Watcher arm table is now **parsed** into (configuration, runs, trips) rows and each row is matched to the tally the gate derives from the logs; limitation 6's numbers are bound the same way; the superseded-phrase blacklist is whitespace/punctuation-normalised. All three of round 7's mutations — swapped tallies, an injected contradictory paragraph, a paraphrased retracted claim — now fail the gate, verified on a scratch copy |
| **P2** the trace-counter failure path leaked unrecoverably | round 6 made only the *decrement* conditional, which was half a fix: clearing the id discarded the only handle so no retry was possible, the counter had no reachable decrement left (turning the `deallocate()` warning into a permanent false positive), and the buffers a possibly-live trace held were freed anyway. A failed release now changes **nothing** — id, buffers and count all stay and `teardown()` retries |
| **P2** the clamp assertion was vacuous | both release paths short-circuit on empty state, so `max(0, ...)` was never executed and removing it left the test green. The test now calls `note_trace_released()` directly to exercise it |
| **P2** contract path resolution was narrow and the HF exception was a line-level escape hatch | every `doc/` artifact path in the current-stage subtree is resolved, not just `.py` tokens in `tested.commands`; the exception matches the exact HF command string rather than any line containing `--arm hf` |
| a fourth stale suite size, and this log's own record of the round-6 fix was stale by one | both corrected against `test_results.xml` |
| limitation cross-references off by one (the prefill norms are limitation 9, not 8) | fixed in the README then; round 17 found two more in *this* file (§2's L1 item and §4's prefill-norm row) and they are fixed now, with the numbering bound by the figure gate so the next drift is a failure rather than a finding |
| the contract's notes named the wrong predecessor block | derived from the blocks actually nested rather than hardcoded, and the recorded `--arm tt` / `--arm compare` commands now carry the `--reuse-hf-control` flag they were run with |
| the "nine new cases" table listed eight | the ninth added |
| `ADVERTISED_CHECKS`' comment claimed a cross-document binding | described as what it is: an internal drift tripwire |

Two things were added that no round asked for but that the findings implied. The opt-in
prefill trace's **three eligibility conditions** (`prompt_len <= 8192`, `user_id == 0`,
`not return_all_logits`) are now documented next to the advertisement, because a serving stage
reading "turn it on and raise the bucket count" would otherwise not know the 21 % win is
silently absent above 8192, for every batch row but the first, and on the all-logits path.
And `capability_report()` now reports this stage's own four flags, so the baseline and shipped
evidence arms are no longer byte-identical in their `capacity` blocks on exactly the settings
that separate them.

### Round 8 — `more-work-needed`

Two P1s, and the first one is the same defect for the third time: a figure block labelled
with an artifact it does not come from. The **whole** `performance` block of
`doc/context_contract.json` was the round-6 run while its `source` field named this one —
and the stage's own `refresh_context_contract.py --check` said so, exiting 1, while this
gate passed 651/651 because it asserted the provenance *string* and never the figures.
Regenerating changed exactly the six fields the review predicted.

The second P1 is round 7's own fix reintroducing round 4's bug. Round 7 made a failed
`ttnn.release_trace` retain everything in place — which is the handle `decode_forward`
tests (`if self._trace_id is None: capture`) and the bucket `_prefill_traced` looks up. So
on the rebind path a failed release meant *replaying* a trace against the cache the caller
had just rebound away from: silently wrong tokens, from the branch round 4's test does not
take. It now fails **closed**: the handle and every tensor the trace may still read move to
`_orphaned_traces`, which no lookup path consults; the id/bucket slots are cleared so the
next call recaptures against the live cache; nothing is deallocated and
`live_traces_over_kv_cache` stays raised; and `_retry_orphaned_traces()` retries at the next
rebind and at `teardown()`. `_prefill_trace_cache_sig` is now cleared **unconditionally**,
which closes the `prefill_trace_max_entries > 1` case round 8 found: a partial failure used
to return before clearing it, so the next capture on another bucket re-stamped the signature
to the new cache and the stale entry could never be invalidated again.

| finding | what was done |
| --- | --- |
| **P1** the contract's performance block was the previous run under this run's name | regenerated; the gate now binds **every** field of the block to `evidence_perf.json`, asserts the block holds no unbound figure, and runs `refresh_context_contract.py --check` as a subprocess — the one check in the tree that would have caught it |
| **P1** a failed release fell open into a stale-cache replay, for both trace kinds | fail-closed orphan list, unconditional signature clear, a retry at every rebind and at teardown, and a new test (`test_a_trace_that_fails_to_release_is_never_replayed_and_is_retried`) that monkeypatches `ttnn.release_trace` to raise and pins all four properties: nothing replayable, nothing deallocated, the count still raised, the next decode answering from the rebound cache, and the retry bringing the count back down. Held to round 4's standard: the test's failure against round 7's code is committed as `logs/trace_release_failclosed_negative_control.log` (partial revert, shipped source restored and re-run green afterwards), so it is a demonstrated discriminator rather than a test that passes against both |
| **P2** three derived figures in `perf_summary.json` were the superseded run's, and the gate read none of them | re-pointed, and the file is now **exhausted** by the gate: every numeric field is bound at a named path (an unlisted path fails, so a new field cannot go unbound) and every decimal number in its prose must resolve to an artifact value or a documented constant |
| **P2** the work log recorded the pre-regeneration run in five places, one two regenerations stale | all re-pointed; its headline figures are bound to the same artifacts the README's are, with the superseded literals blacklisted by name |
| **P2** the cross-section gate was defeatable eight more ways, all demonstrated | a fabricated sixth arm row now fails (rows the gate cannot attribute are a failure, not a skip); the headline table is bound **cell by cell** with the columns asserted not interchanged; the audit table's `ids` and `µs` columns are parsed from the README and checked against the CSV instead of from a hardcoded copy; every `N of M` in the two arm sections must be a tally derived from the logs; the eligibility conditions are bound to `tt/generator.py`; and the duplicated figures (device time, roofline, `SdpaDecode`, the accuracy rates, force-argmax, the context, the suite size, the fallback counters) are bound to the section that claims them |
| **P2** `capability_report()` read the softcap flag off the module, not the build | reports `self.model.lm_head.softcap_in_l1`, which is what `forward` consults. The value is unchanged in all three arms, so no artifact went stale on it |
| the matched contrast was the one contrast the multiplicity paragraph did not correct | corrected: **p = 0.36**. The gate computes all three corrections from its own Fisher values and requires the text's rounding to be a rounding of *that*, not of an already-rounded p — which is where the previous `0.0013` came from |
| the prefill-trace arm's evidence predated the four `capability_report()` flags | regenerated (`TTFT 49.76 ms`); all three perf arms are now asserted to carry the block and to disagree exactly where the arms disagree |
| the round-6 conclusion sat in this log with no supersession marker | marked, with a pointer to the round-7 restatement and to limitation 6 |
| §8 disagreed with the README on two figures from the same partition | corrected to the README's (**184.20 µs** of a 436.602 µs layer; the projections at **52.27–77.12 %**) |

The suite is **59** cases (46 inherited + 13 new), forward and reverse.

And the thing four rounds of this have implied: the mutation testing is **committed**
(`bench/mutate_figure_gate.py`, `logs/mutate_figure_gate.log`). Forty-six mutations, each one a
defeat a review demonstrated, each applied alone to a scratch copy of the model directory,
each required to make the gate fail. Four of them survived the first attempt at this round's
fixes — the work log's token-out row, its two-trace residual, the accounting section's device
figure and the accuracy table's top-5 — all for the same reason the review named: those
figures appear more than once in their document, so "the literal is present" was satisfied by
the other copy. They are bound to their table cell now, and the gate asserts the harness's log
alongside its own verdict.

### Round 9 — `more-work-needed`

The P1 was the third capture. Rounds 4, 6, 7 and 8 all worked on trace lifecycle and all four
worked on the two traces *this port owns*; the sampling trace is captured by the shared
`SamplingGenerator`, and its `reset_trace` logged a failed `ttnn.release_trace` and then ran
`_trace_states.clear()` anyway. That drops the slot — the only reference to the handle **and**
to the `sampled` tensor allocated during capture — while the trace is live: a freed buffer
under a live trace, and an unreleasable handle, silently. The fail-closed policy this stage
wrote in round 8 covered two of three captures, and the test could not see it because it never
passed `sample_on_device=True`.

`models/common/sampling/generator.py` now fails closed the same way: failed slots move to its
own `_orphaned_traces`, out of the lookup table so `_execute_trace` cannot replay one, still
referenced so nothing they hold is collected; `reset_trace` returns how many failed and
`retry_orphaned_traces()` retries them, which this generator's `_retry_orphaned_traces()`
calls. It is the one change this stage makes outside its own directory, and it is additive:
the healthy path returns 0 and behaves exactly as before.

| finding | what was done |
| --- | --- |
| **P1** the sampling trace was outside the fail-closed policy, and the README claimed otherwise | fixed in the shared sampler as above; the README's trace-inventory row and the fail-closed paragraph now describe all three captures, including *why* the sampling trace is deliberately **not** counted in `live_traces_over_kv_cache` (it bakes no cache address); the test now captures a sampling trace and asserts the sampler's own orphan properties |
| **P2** the drains sat between a successful release and the bookkeeping | the count decrement and the container update now happen **before** the drain in both release paths and in the retry, so a raise in the drain cannot leave a released bucket replayable, double-release an id, or pin the count high; the two release calls are sequenced with `try/finally` in both callers, so a raise in the prefill release can no longer skip the decode one |
| **P2** `README.md`'s Artifacts table stated a stale test count contradicting three other places | corrected, and the count is bound to `test_results.xml` |
| **P2** invalidation was gated on `kv_cache is not None` and absent from `generate()`; `set_kv_cache` could half-rebind | the signature comparison runs on **every** entry point unconditionally, `set_kv_cache` validates every layer before binding any, and a new test (`test_a_cache_rebound_out_of_band_still_invalidates_the_traces`) drives a rebind the generator is never told about and the rejected-cache atomicity |
| the Commits table did not record `20f77bb0fcd` | recorded, with this round's entry |
| `watcher/` no longer holds the run `logs/check_watcher_12case_tripped.log` re-derives from | disclosed in the Artifacts table: that arm's verdict rests on its console log, which is what limitation 6 cites; the directory count (18) is stated and bound |
| the figure gate could not see unit changes, fabricated sections or rows, or one section contradicting another | the section list and five table row-counts are asserted, four units are bound where the figure is, and limitation 1 must price the trace against the same two TTFT figures the headline table states |
| the mutation log proved nothing: placeholder text passed, and a neutered mutation left it still passing | every logged line now carries a digest of the mutation's full content and the gate requires the log's digest set to equal the table's, so the log cannot outlive an edit to what it claims to have tested; the bootstrap writes its placeholder into the **scratch copy** only, so nothing in the tree can produce a passing log without a run |
| the work log's evidence table was bound in 6 rows of 20 | the remaining numeric rows are bound to their artifacts |

### Round 10 — `more-work-needed`

The P1 was round 9's fix, one branch over. Round 9 brought the sampling trace inside the
fail-closed policy; round 10 found the **asymmetric** partial failure it does not cover. The
sampling trace is captured over `_trace_logits` — that tensor *is* the sampler's slot `input` —
so if the sampler's release fails while the decode release succeeds, `_release_decode_trace`
walks on to `ttnn.deallocate(self._trace_logits)` and hands a live trace's captured buffer back
to the allocator. Python-reference retention cannot prevent that; the other owner is the code
doing the freeing. The round-9 test could not see it because it makes `ttnn.release_trace`
raise for *every* id, so the decode release fails too and the logits end up in this
generator's own orphan list.

The frees are now gated on the other owner: `_free_or_defer` holds anything the sampler might
still be reading in `_deferred_frees`, and `_retry_orphaned_traces` releases them only once
`orphaned_trace_count` reaches zero. The new test injects a failure for the sampler's id
**only** and asserts `is_allocated()` rather than Python identity.

| finding | what was done |
| --- | --- |
| **P1** a live sampling trace's captured input was freed on the decode release's success path | `_free_or_defer` + `_deferred_frees`, flushed by the retry when the sampler is clear; `test_a_sampling_trace_that_fails_to_release_keeps_its_logits` injects the asymmetric failure and checks allocation |
| **P2** 13 of 25 fresh mutations defeated the gate — all duplicated-literal evasion or never-resolved figures | every one is bound to its own cell or section now (the opening claims, the softcap pair and its CSV rows, the roofline bytes and bandwidth, both @256 floor rows and the floor total, the L1 peak columns, the quoted watcher verdict block, the qualitative character counts, the retained DRAM, and the shared sampler's retry semantics), and all 13 are in the harness — **46 mutations, all caught** |
| **P2** round 9 added unmeasured per-call host work to `decode_forward` | measured (`invalidation_cost_probe.py`): **7.57 µs** per call with a trace live, **0.033 %** of the token-out step, and **0.06 µs** with nothing captured, because the signature is now built only when something is live. The README states all three |
| the Artifacts table named 3 of the 6 code files this stage changed | all six listed, and the gate derives the set from `git diff 93adb25b7a8..HEAD` |
| this log claimed "four implementation/test files … two readiness directories" | corrected against the diff: five files here, one shared, the contract, one readiness file |
| the release paths did not count their own drains | `counters["synchronizations"]` incremented at both |

Left alone, with the reason: `models/experimental/llama32_1b_quasar/sampling/generator.py` is a
vendored copy of the shared sampler and still has the pre-round-9 `reset_trace`. It is a real
instance of the same defect, it is outside this stage's model and outside its goal contract,
and fixing it would put an unrelated experimental model in this stage's commits. Recorded here
so it is a decision rather than an oversight.

### Round 11 — `more-work-needed`

No P1. Three P2s, and the first is the one that matters: **the gate's perimeter was still
narrower than the document's claim about it.** Round 11 built 54 fresh mutations and 33
survived — including the `$optimize` dtype/fidelity table (rewriting the LM head to BFP8 and
`mlp_down` to HiFi4 both passed), headline-table cells the README said were "bound cell by
cell", and the @2048 floor table. No figure was actually wrong; the defect was coverage
against a claim of completeness, for the fifth round running.

Pointwise patching had not converged, so the rule changed shape. **Every numeric cell of every
README table** must now resolve to an artifact through a check, be an op id in a committed CSV,
be a value from the built model's capacity block, or appear in `UNBOUND_TABLE_NUMBERS` with a
written reason. A figure no artifact supports fails whether or not anyone thought to bind it,
and changing an allowlisted number to a value that is not itself listed fails too. The
dtype/fidelity table is separately bound to the CSV's own `Math Fidelity` and datatype columns
— the review skill names that exact check and the gate had been parsing the same CSV for the
audit partition without ever reading those columns.

| finding | what was done |
| --- | --- |
| **P2** 33 of 54 fresh mutations survived; the README claimed every figure resolves | the by-construction coverage rule above, plus the dtype/fidelity table bound to the CSV, plus twelve of the 33 added to the harness as regression cases (**71 mutations, all caught**). Units remain enumerated rather than structural — the rule is digit-based and cannot see µs printed as ms — so two more unit bindings were added and the limitation is stated rather than left implicit |
| **P2** the round-10 fix had no committed negative control, while the Artifacts table said every test in the family had one | `logs/deferred_free_negative_control.log`: the unconditional free restored, the test failing on *"a live sampling trace's captured input must not be freed"*, source restored. Bound by the gate to pytest's summary line, as the other two are. There are three controls and the table now says three |
| **P2** `_deferred_frees` was keyed on "any sampler orphan outstanding", not on ownership | keyed per tensor now (`_tensors_a_held_sampling_trace_reads`, identity against the sampler's held slots), so a stuck orphan no longer pins every later rebind's logits or the prefill traces' buffers; the misleading warning is corrected; `teardown()` reports what it leaves unfreed; and the test asserts **exactly one** deferred tensor with a prefill trace released in the same call |
| the retry was reachable only from a genuine stale-signature release | the no-trace short-circuit now retries first when anything is outstanding, so a stuck orphan is not stranded for the life of the generator |
| drains were counted inconsistently | every drain in both release paths and in the retry increments `counters["synchronizations"]` |
| the shared file's unhealthy-path effect on other callers was unrecorded | stated in the Artifacts table: they retain rather than drop, and without a `retry_orphaned_traces()` call the slot is pinned rather than silently leaked — the safer of the two |

### Round 12 — `more-work-needed`

No P1. Four P2s, and the one with teeth is the **terminal term**: `691.07 µs` was a hardcoded
constant in the figure gate that no artifact supported, and four load-bearing figures rested on
it — the device-time reconciliation, both per-layer splits, the latency-bound remainder, and
the `0.691 ms` addend of the floor-plus-terminal comparison the goal's 10–15 % gate is measured
against. Worse, the README claimed *"every figure in this paragraph is a row or a group of the
audit table"*, which was false: the audit groups ops by **kind**, and the layer/terminal split
is by **frequency** (once per step vs per layer).

It is derived now, from a named id list the README prints and the gate sums: **690.973 µs**.
The old constant was 0.097 µs above that. And the two windows do not share all of it — the full
layers are NoPE, so `decode_full_perf_report.csv` has no rotary op and only one embedding
lookup against the sliding capture's three. The two RoPE-table lookups (**5.024 µs**) are
sliding-path work, so the full layer is `1100.228 − (690.973 − 5.024) = 414.279 µs`, not
409.16. The corrected step is **22.908 ms** (was 22.838), which moves the device-vs-replay gap
from 0.8 % to **1.1 %** and changes no conclusion: device time still sits above the 22.656 ms
replay, so there is still no room for a per-step host bubble.

| finding | what was done |
| --- | --- |
| **P2** the carried-forward decoder contract table was bound to nothing, though it says it is read off the capacity block — ten of its twelve cells were flippable, including the `changed?` column | every cell is bound to its own field in `capacity["carried_forward_decoder_contract"]` (dtypes, both collectives, the persistent-buffer verdict, the fractured-norm gate, the residual core count, the `o_proj` geometry, the SDPA grid, the sampler geometry), the `changed?` column included, and six of the flips are in the harness |
| **P2** `46 mutations` in the Artifacts table was stale against 58, and an allowlist entry whose reason did not describe it was hiding it | corrected and bound to `len(MUTATIONS)`; `"46"` is gone from the allowlist and the inherited-case count is bound instead |
| **P2** the terminal term was a hardcoded constant, and the paragraph resting on it claimed a provenance it did not have | derived from named ids as above, with the RoPE asymmetry priced and the claim corrected |
| **P2** the by-construction rule reached only numerals in non-indented table rows, while the README's checklist row still claimed everything | the parser strips indentation, and the checklist row now **states the perimeter**: numerals in tables plus the named prose figures and four units, with prose figures, worded numbers and other unit swaps explicitly outside it |

The perimeter statement is the point. Five rounds widened the gate and re-asserted completeness;
this one widens it *and* narrows the claim to what it covers, so the next reviewer's test is
whether the stated perimeter is honest rather than whether the claim is true.

### Round 13 — `more-work-needed`

A **P1**, and it was mine: round 12 put the two RoPE-table gathers in the terminal term because
the NoPE full capture does not contain them. That reasoning is backwards.
`_decode_rope_tables` is called *inside* the layer's `decode_forward`
(`tt/optimized_decoder.py`, under `if cfg.uses_rope`), so those two gathers run **once per
sliding layer** — 39 times a step. The full capture lacks them because its layers are NoPE, not
because they are terminal.

Correcting it moves four published figures and one JSON field: terminal **690.973 → 685.949 µs**,
sliding layer **431.578 → 436.602 µs**, device step **22.908 → 23.099 ms**, device-vs-replay
**1.1 → 2.0 %**, latency-bound remainder **179.17 → 184.20 µs** (41.5 → 42.2 %). No conclusion
flips: device time still sits above the 22.656 ms replay, so there is still no room for a
per-step host bubble — the argument is *stronger* with the larger figure.

The lesson is in the check that now enforces it. A genuinely once-per-step op runs whatever the
layer kinds are, so **its op kind must appear in both captures**. That rule fails on 3158/3161
by construction, and it is the kind of check that would have caught the mistake when it was
made rather than a round later.

| finding | what was done |
| --- | --- |
| **P1** the terminal term counted two per-sliding-layer RoPE gathers once per step | removed from the list; every dependent figure recomputed in the README, `perf_summary.json` and this log; the both-captures rule added to the gate, with the call site asserted in the source |
| **P2** the stated perimeter over-claimed: seven in-perimeter mutations survived, and three allowlist entries had reasons that did not describe them | the TTFT phase table, the residual shard geometry, the dtype table's full shapes and the checklist's terminal price are bound; `0.51`/`0.65`/`0.85`/`2.16`, `416` and the stale `691.07` are out of the allowlist; the perimeter statement is rewritten to name what is covered (cells a check positively binds) and what is not (allowlist membership, op-id collisions, prose outside the named list, worded numbers, other unit swaps), with the allowlist's size bound to its own length |
| the full layer imports the sliding capture's terminal | named as a cross-capture substitution, with the full capture's own copy of the same eleven ops (683.0–683.3 µs) and the ~0.7 % it carries stated |
| the harness said only "BASELINE FAILS" when its own baseline broke | it now prints the gate's failing lines, which is how the round-13 fixes were debugged |

### Round 14 — `more-work-needed`

The P1 was the perimeter claim again, and this time the reviewer proved it the right way: a
**systematic sweep** — every numeric table cell mutated, one at a time — against a claim that
nine named tables were bound cell by cell. The ctx-2048 layer-stack floor table, which carries
the goal contract's own "decoder-layer stack lower bound", was not bound at all; its sibling at
ctx-256 had been bound in round 10 and the perimeter statement then claimed both.

Two things came out of it, and the second matters more than the first.

**The floor table is bound cell by cell** — layer counts, before and after ms/layer, the
product each row prints, and the totals — against `evidence_perf{,_before}.json`. So are the
headline teacher-forcing before/after ranges (against the previous stage's own three runs and
this stage's), and the five previous-stage op ids the audit table quotes.

**The sweep is now part of the harness.** `mutate_figure_gate.py --sweep` generates one
mutation per numeric table cell — 209 of them — instead of replaying the 71 curated defeats.
That is the coverage test; the curated list is a regression suite. Both logs are committed and
both are asserted by the gate. The sweep found four survivors on its first run and two more
after the first fix; all six are closed, and the arm that finds the next one now exists.

The other finding is the sharpest thing this round: **the README cross-check loop sat in the
middle of the gate**, so every literal registered by a `bind()` *after* it — five rounds' worth,
including this round's — was recorded and never searched. It runs last now. That single move
turned 50 recorded-but-unchecked bindings into real ones, and it is why the op-id mutations
started failing.

| finding | what was done |
| --- | --- |
| **P1** the ctx-2048 floor table was unbound while the perimeter claimed it, and a sweep found 196 survivors | the table is bound cell by cell, the teacher-forcing ranges and the previous-stage ids with it, and the sweep is a harness arm with its log asserted — **209/209 caught** |
| **P2** `683.0`/`683.3`/`0.7 %` were hardcoded — the defect round 12 removed as `691.07`, reintroduced by round 13's own fix | derived from `decode_full_perf_report.csv` by op kind, bracketed over every pairing of that capture's six hidden-size norms, and bound to the README sentence |
| **P2** "the six hidden-size norms differ by at most 0.13 µs" was false | corrected to the measured spreads, 0.112 µs (sliding) and 0.162 µs (full), both derived by the gate |
| **P2** the work log recorded an allowlist cleanup that had not happened | it has now: the four TTFT-phase minima, the shard width and the stale `691.07` are bound and out of the allowlist, which is 183 entries and bound to its own length |
| the README cross-check ran before half the bindings were registered | moved to the end of the gate, where it means what it says |

### Round 15 — `more-work-needed`

The review walked the goal contract requirement by requirement and found every one met on
evidence it re-derived itself. Both findings were about claims, and the P1 was a figure that no
artifact supported — in the capability contract, on the one shipped-but-opt-in feature, for the
benefit of the next stage.

**"~210 MB at 8192" was a 64× extrapolation of a figure that is ~99 % length-independent.**
`prefill_logits` slices one 32-row tile whatever the prompt length, so the trace's retained
output is a constant `[1, 1, 32, 50688]`. Rather than restate it as unmeasured, it is now
**measured**: `prefill_trace_probe.py --length 8192` gives **4.6 MB** retained — the claim was
wrong by ~45× in the alarming direction — and the real bound is the mesh's fixed
`trace_region_size`, which the earlier text never named.

The same run answered a question nobody had asked. At 8192 rows the traced replay is
**921.52 ms against 917.36 ms eager — 1.00x, no win at all.** What the trace removes is host
dispatch, and at 8192 rows dispatch is a rounding error against device work. So the flag is a
*short-prompt* win, and the README's advice to a serving stage — "raise
`prefill_trace_max_entries` to its bucket count" — is now qualified to the buckets where it
pays, with one command per bucket to check.

| finding | what was done |
| --- | --- |
| **P1** an unsupported 64× DRAM scaling law in `README.md`, `tt/generator.py` and `doc/context_contract.json`, on a feature the next stage is told to enable | measured at 8192 (4.6 MB, 1.00x, bit-identical); all four places restated with the measurement, the withdrawn extrapolation named as withdrawn, `trace_region_size` named as the real bound, and the new probe bound by the gate. A failed capture now falls back to the eager prefill for that request instead of raising mid-call — the contract the full-bucket-cache case already had |
| **P2** §6's "every evidence row post-dates the last code change" is no longer true | replaced with per-artifact provenance: which files are current, and for each older one the specific reason it is unaffected |
| the `506 µs` step delta was off by one | 507.2 µs, from the two arms' own minima |

### Round 16 — `more-work-needed`

The P1 was round 15's own fix. Making a failed prefill-trace capture fall back to eager was
right; leaving it at that was not. Nothing recorded the failure, so at the shipped
`prefill_trace_max_entries = 1` the next request found the bucket cache empty and tried again —
a retry on **every** request for the life of the generator, each paying two extra full prefills
before falling back. And the resource the real failure exhausts is accounted on the device
*before* the throw, so retrying walks the always-on decode trace's recapture into the same wall.

The failure is sticky now: capture is disabled for that generator, the trace id is released so
the pool entry is not stranded, the bucket's two persistent inputs are freed, and
`capability_report()` carries `prefill_capture_failures` and
`prefill_capture_disabled_after_failure` — a counter that is written and never read is not
evidence.

**What the test could not be.** The release side of this subsystem is fault-injected three
times over. The capture side cannot be: making `ttnn.end_trace_capture` raise **hangs the
device** — ending a trace twice does, and the injection leaves the capture in a state the real
failure does not. I found that the way one does, by hanging the mesh and resetting it. So the
new case drives the *state* the failure path sets and asserts what the generator does with it,
and the asymmetry is written down rather than papered over. The cleanup deliberately does not
re-`end` a capture for the same reason.

| finding | what was done |
| --- | --- |
| **P1** a failed capture was retried per request, stranded its pool entry and its inputs, and had a dead counter | sticky disable, release + deallocate on the failure path, the counter in `capability_report()`, and a state-driven test (`test_a_failed_prefill_capture_falls_back_and_stays_off`) |
| **P2** `prefill_trace_max_entries > 1` — the configuration recommended to serving — had never been run by any test, probe or evidence file | `bench/prefill_trace_multibucket_probe.py`: two buckets resident, each replayed after the other, **every generation token-identical to an eager arm on the same build**. It also settled a false alarm: an in-suite version of the same sequence appeared to diverge, and the probe showed the divergence was my test's reuse of stale expectations, not the feature |
| **P2** the probe's own docstring still said the stage had decided *not* to ship a prefill trace | rewritten to describe the shipped opt-in flag and what the probe is for |

### Round 17 — `more-work-needed`

Four P2s, no P1, and the theme is that **two of them were fixes reported as complete**. Round
15 replaced a false provenance sentence with a partition; round 17 found the partition itself
misfiling six artifacts as "current" and still carrying round 15's suite size. Round 6 recorded
"limitation cross-references off by one … fixed in both documents"; two references in *this*
file were still wrong. A hand-maintained classification of files that keep moving does not stay
true, and a record of a fix is not the fix.

Both are derived now rather than maintained: the provenance partition comes from
`git log -1 -- <path>`, and the figure gate binds the limitation numbering to the README's own
headings and checks every `limitation N` reference in this file against what that limitation is
about.

The third finding was mine to answer: round 16 wrote off the capture-failure path as untestable
because injecting inside `_capture_prefill_trace` hangs the device. Round 17 pointed out that
was one level too pessimistic — replacing the *method* raises before `begin_trace_capture`, so
no trace begins and no queue records. That half is a real fault injection now, and it pins the
thing that matters: **the failure is not retried on the next request**.

| finding | what was done |
| --- | --- |
| **P2** §6's provenance partition misfiled six artifacts and quoted a stale suite size | derived from `git log`, with the one intervening change that *is* visible to a perf number named and priced (round 9's unconditional invalidation, 7.57 µs, 0.033 % of the step, and not on the measured loop at all) |
| **P2** the capture-failure branch was executed by nothing, and the cleanup began one line too late | injected at the method boundary, which is safe; `tokens`/`tt_page_table` are freed if the warm compile, the drain or `begin_trace_capture` raises — the last of which is exactly where the trace region runs out |
| **P2** three stale counts and a false record of a completed fix | corrected, and the numbering bound so the next drift is a gate failure |
| **P2** the trace-region budget behind the serving advice is unmeasured | **not closed**: stated as a named limitation instead. The region is 400 MB, one 52-layer capture's occupancy is unmeasured, and the multi-bucket probe runs on a 2-layer build — so the advice to raise `prefill_trace_max_entries` now says what is known and what is not, and the sticky disable's cost is stated with it |

### Round 18 — `more-work-needed`

A **P1 of my own making, and the worst kind**: the figure gate was **red at the reviewed
commit** while three documents said it was green. Round 17's provenance check asserted
`git log -1 -- logs/mutate_figure_gate.log == HEAD`. Both mutation arms *were* re-run — and
produced **byte-identical** logs, so git recorded no change and the file's last-touching commit
stayed at round 16's. The check was unsatisfiable by construction for any commit that does not
alter that log's content, and it failed at the commit that introduced it. I committed on the
strength of a gate run I made *before* the harness re-ran, which is the same "a record of a fix
is not the fix" failure I had written into the round-17 entry one screen earlier.

The lesson is the one the check was reaching for and missed: **what matters is not which commit
last touched an artifact but whether its content agrees with the tree it describes.** That is
what the gate checks now — the mutation log's digests against the current table, the junit file
against the suite size, the watcher verdict against the log it is re-derived from — and the
partition says plainly that a byte-identical re-run leaves git behind, which is why git is not
the arbiter.

| finding | what was done |
| --- | --- |
| **P1** the gate failed at HEAD; the provenance check was ill-posed | replaced with content-agreement checks; the partition rewritten to say what was re-run and why git dates three of them earlier |
| **P2** the Tests table still called the capture-failure test "state-driven rather than fault-injected" — round 17's own fix, unpropagated | rewritten to describe the two-half injection that actually ships |
| **P2** the headline footnote called teacher forcing "the **one** cross-process comparison"; the layer-stack row is a second, and it carries a claim | both named, with the layer-stack row's provenance pointed at the section that derives it |
| **P2** the sweep arm was described as "one mutation per numeric table cell" while three undisclosed filters cut 697 tokens to 211 | the filters print their skip counts, the log is self-describing, and both sentences say what the arm does |
| **P2** four figures that resolve to nothing: a −3.7 % that is −3.6 %, a gap range of 0.47–1.47 µs that is 0.475–0.688, two references to a `ccl_host_probe.json` that does not exist, and a mean quoted where every neighbouring cell is a min | all four corrected against the artifacts |

### Round 19 — `more-work-needed`

The P1 was round 18's fix being cosmetic: the three "content agreement" checks I put in place
of the ill-posed provenance check were **two exact duplicates of checks already in the file and
one tautology** — the junit case count compared against itself. In substance the bad check had
been deleted and not replaced, and I had written a work-log entry saying otherwise. That is the
third round in a row where the failure was a record of a fix rather than the fix, and this time
it was mine twice over.

They discriminate now: the junit's case **names** against the test functions the suite file
defines (both directions), and the watcher verdict re-derived here from `watcher/watcher.log.gz`
itself — dump boundaries, detach lines, and the absence of a tripped assert or a fatal message —
rather than read off the console file that reports it.

| finding | what was done |
| --- | --- |
| **P1** the replacement currency checks were duplicates and a tautology | junit case names ↔ suite definitions, both directions; the watcher verdict re-derived from the compressed log |
| **P2** the −3.7 % round 18 recorded as corrected was never corrected in the README | corrected to −3.6 % (23.298 against 24.176) |
| **P2** the 311 µs gap's stated support was false — the report has no `HOST START TS` column, is sorted by `ID`, and in host time id 3145 *is* the window's first op | replaced with the support that holds: all four devices independently record 310.9–314.7 µs on their own embedding row, so it is a real per-device inter-op latency. The conclusion was never in doubt; the argument for it was wrong |
| **P2** §4's datatype row carried 41 % and 52–72 %, which round 13 had already corrected elsewhere to 42.2 % and 52.27–77.12 % | corrected |
| **P2** `_release_prefill_traces`' orphan branch left its bucket in the dict, relying on a post-loop clear an unguarded drain could skip | `del` moved into the branch that orphans, and `teardown()`'s last-chance retry moved inside a `finally` so a raising release cannot skip it |

## 11. Commits

Local checkpoints on `agentic-research/hous/muse-glimmer-30b`, on top of the full-model
stage's `93adb25b7a8`. Never pushed.

| SHA | what |
| --- | --- |
| `ee3c378a830` | `tt/model.py`, `tt/generator.py`, `tt/optimized_decoder.py`, `tests/test_full_model.py` — the three decode-path layout changes, the opt-in traced prefill, and the seven acceptance cases new *in that commit* (the suite has grown since; the current count is in the Tests section of the README, derived from `test_results.xml`) |
| `3d03b5ca595` | `doc/optimized_full_model/` in full, the regenerated `doc/context_contract.json`, and the regenerated `readiness_autoregressive_{chat,raw}/` outputs |
| `5e6022db622` | round-3 review fixes: the audit-table partition, the corrected prefill-128 and host-dispatch figures, the retracted retirement flag, the `--arm rebuild` probe and the gate's coverage extension |
| `c28f91010d0` | round-4 review fixes: the decode-trace cache invalidation and its test, the fabric-ERISC length control, the context-contract provenance, and the gate's opened-artifact coverage |
| `40e3fd71014` | round-5 review fixes: one trace-release path with a drain, the fixture finalizer, the model's live-trace guard and semaphore-cache cleanup, the re-measured limitation-6 arms, all three Fisher contrasts, and the contract's `tested` provenance |
| `24cdea2f559` | round-6 review fixes: limitation 6 rewritten from current data with cross-section gate checks, the work-matched and pair-alone arms, the comparator-not-a-bound correction, the contract-substitution repair, and the trace-counter test |
| `c675d8dc2d3` | round-7 review fixes: the derived process count, the conclusion restated to what the arms support, a parse-and-bind cross-section gate, the recoverable trace-release failure path, and the documented prefill-trace eligibility |
| `6cc255a19d1` | round-8 review fixes: the regenerated context contract and prefill-trace evidence arm, the fail-closed trace-release path and its injected-failure test, the exhausted `perf_summary.json` and re-pointed work log, cell-level and section-scoped gate binding, and the corrected multiplicity paragraph |
| `20f77bb0fcd` | the negative control for the fail-closed release: the test's failure against round 7's code, committed and bound by the gate |
| `9abea54b55b` | round-9 review fixes: the sampling trace brought inside the fail-closed policy (in `models/common/sampling/generator.py`), the bookkeeping-before-drain ordering and `try/finally` sequencing, unconditional invalidation on every entry point, atomic `set_kv_cache`, two new acceptance cases, and a figure gate that checks structure, units, cross-section consistency and its own mutation log's provenance |
| `bd39469e555` | round-10 review fixes: deferred frees so a live sampling trace's captured input is never handed back, the measured per-call invalidation cost, cell-level bindings for the thirteen figures round 10 falsified, and the corrected file inventory |
| `562c529f4f2` | round-11 review fixes: by-construction numeric coverage over every README table, the dtype/fidelity table bound to the CSV, per-tensor deferred frees, the third negative control, and the retry reachable from the short-circuit |
| `d45ac8e2a0f` | round-12 review fixes: the terminal term derived from named ids with the NoPE asymmetry priced, the carried-forward decoder contract bound cell by cell, the stale mutation count, and the gate's perimeter stated rather than overclaimed |
| `c6a85246281` | round-13 review fixes: the RoPE tables moved out of the terminal term with every dependent figure recomputed, the both-captures rule that catches the class, and the perimeter statement corrected against seven surviving in-perimeter mutations |
| `d8c5d686b13` | round-14 review fixes: the layer-stack floor table bound cell by cell, the generated mutation sweep as a harness arm, the cross-check loop moved to where it covers every binding, and the cross-capture bracket derived rather than stated |
| `917a3225afd` | round-15 review fixes: the 8192-row prefill-trace measurement that replaced a 64x extrapolation, the eager fallback on a failed capture, and per-artifact evidence provenance |
| `b1b3a3569fd` | round-16 review fixes: the sticky capture-failure disable with cleanup and a report field, the multi-bucket probe that exercises the recommended serving configuration, and the corrected probe docstring |
| `3bc742fd6f6` | round-17 review fixes: the provenance partition derived from git, the capture-failure branch actually injected, the earlier cleanup boundary, the limitation numbering bound, and the unmeasured trace-region budget stated as a limitation |
| `c6b1c6c3022` | round-18 review fixes: the ill-posed provenance check replaced by content agreement (it had left the gate red at the previous commit), the Tests-table and footnote claims corrected, the sweep's filters disclosed, and four unresolvable figures fixed |
| *(this commit)* | round-19 review fixes: currency checks that discriminate, the orphan branch made raise-safe with teardown's retry in a finally, and three figures/claims corrected against their artifacts |

Nothing unrelated is in any of them: `git status` is clean at each. Outside
`doc/optimized_full_model/`, `git diff --name-only 93adb25b7a8..HEAD` is exactly eight paths:
five implementation/test files in this port (`tt/model.py`, `tt/generator.py`,
`tt/multichip_decoder.py`, `tt/optimized_decoder.py`, `tests/test_full_model.py`), the one
**shared** file round 9 had to fix (`models/common/sampling/generator.py`), the contract, and
`readiness_autoregressive_chat/autoregressive_meta.json`. Round 10 of the stage review caught
this paragraph claiming four files and two readiness directories; it is derived from the diff
now, and the Artifacts table in the README lists all six code paths.

The repo's pre-commit hooks reformatted three bench scripts and the test file on the
first attempt; the reformatted state is what is committed, and the eight affected tests
were re-run afterwards (`logs/post_format_tests.log`).
