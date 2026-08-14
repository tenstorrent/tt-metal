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
cache. Through the same evidence harness: **TTFT 63.66 → 50.19 ms, −21.2 %**, decode
unchanged to 0.06 % (`evidence_perf_prefill_trace.json`).

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
measured line item and limitation 7.

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
the measured all-layer step is **22.657 ms**, 0.7 % better than predicted.

## 4. Candidates considered and not taken

| candidate | why not |
| --- | --- |
| pack `wqkv` + `attn_gate` (OPT-001: two projections consuming the same post-norm activation) | Both rows are the *best* DRAM utilisation in the layer — **77.12 %** (`wqkv`, id 3039) and **69.35 %** (`attn_gate`, id 3172) in this stage's `tracy/decode_sliding_perf_report.csv`, against 52.27–52.65 % for the MLP rows — so they are weight-bandwidth-bound and packing cannot reduce the bytes. And the two outputs need different downstream layouts (QKV is `sharded_to_interleaved` into `nlp_create_qkv_heads_decode`; the gate stays width-sharded until after SDPA), so a packed output needs an unshard, two slices and a reshard to split, which is ~4 µs against a ~3–5 µs saving. Recorded in the operation-topology audit with the DRAM% evidence rather than measured, because the byte argument is decisive: the packed matmul reads the same 16.3 MB. |
| wider RMSNorm grid, decode | The four decode norms must consume and produce the 16-core boundary spec, which is the inter-layer residual contract this stage is required to preserve. |
| sharded prefill terminal/embedding norms | They run on 1 and 4 cores for ~134 µs each because `ttnn.rms_norm` on a DRAM-interleaved input parallelises over tile rows and both see a 32-row slice. 0.27 ms of a 65 ms TTFT; the fix changes prefill numerics on the accuracy gates' critical path, so it is priced and left as limitation 8 rather than taken for 0.4 % of a figure whose process spread is 8 %. |
| `o_proj` OPT-011 narrower working shard | Kept declined, and **one of the decoder stage's three reasons no longer applies**: change 3 breaks the same single-grid invariant and adds three reshards, so "it costs a reshard and the invariant" is no longer a reason this stage can use. What survives is what decided it: the candidate won 0.11 % on `sliding`, was inside the noise on `full`, and cost 13 % of the multichip-vs-single-chip PCC headroom. Recorded as a candidate a decoder stage may revisit now that the invariant is already gone. |
| fewer decode collectives | Two all-reduces per layer is the replicated-residual contract. A fractured residual would halve the dispatch count, and the decoder stage's `fractured_decode_probe.py` owns that question. Out of scope here: the goal preserves the residual layout. |
| persistent CCL staging buffers | **Worth 14–17 % of the prefill reduce-scatter's host cost** at the model's BFP8/4-worker setting (~2 ms of TTFT), and blocked by the decoder stage's intermittent first-use correctness race. An earlier row here said they were "within noise" on host cost; that was the BF16 hot-loop arm and is withdrawn (§2.4). |
| prefill CCL implementation switch (async → wrapper) | The hypothesis was that the decoder stage chose `async` on *device* time at 8192 rows while short-prompt prefill is host bound, so a cheaper-to-issue wrapper might win at 128 rows. Measured at the model's BFP8 payload: 58.88 against 72.10 µs/call unloaded but 91.42 against 117.05 loaded, and the wrapper form also disables the fractured prefill norm. Refuted (§2.4). |
| lowering `max_top_k` from 32 to 8 | 0.7942 ms against the shipped 0.6323 (`sampler_ab.json`). Slower, and limitation 9 of the full-model stage notes the gathered width interacts with `num_gather_links`. |
| a broad datatype frontier search | `$datatype-sweep` owns it. The one precision question this stage asked is whether the 37.9 % roofline fraction is a precision problem, and it is not: 41 % of the layer is latency-bound non-matmul work and the projections already run at 52–72 % of peak. |

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
| perf, baseline arm | `evidence_perf_before.json` | 23.815 / 23.164 ms, TTFT 65.94 |
| perf + shapes + 130073 | `evidence_perf.json` | **23.315 / 22.657 ms**, TTFT 63.66 |
| perf, `--prefill-trace` | `evidence_perf_prefill_trace.json` | **TTFT 50.19 ms**, decode unchanged |
| autoregressive, chat + raw | `evidence_autoregress.json` | coherent, non-degenerate |
| accuracy + sampling + fallback | `evidence_accuracy.json` | top-5 1.000, top-100 1.000 |
| fp32 control + misses | `evidence_fp32_gate.json`, `evidence_misses.json` | all four gates pass |
| greedy sampler benchmark | `sampler_ab.json` | split still wins, 15x |
| 54-case suite, forward | `test_results.xml` | 54 passed |
| 54-case suite, reverse | `logs/full_test_run_reverse.log` | 54 passed |
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

Every perf, accuracy, autoregressive, qualitative, watcher, Tracy and contract row above
was re-run **after** the last code change — including the round-2 cache-identity and
retirement fix — so no reported figure predates the code it describes. The one exception
is named rather than hidden: `ttft_breakdown_before.json`, `prefill_host_probe.json`,
`prefill_opcount.json`, `ccl_host_probe_*.json`, `prefill_trace_probe.json`,
`l1_highwater_probe.json` and the three `decode_ab*.json` files are *diagnostic* runs
whose subject (host dispatch cost, the collectives, the traced-prefill mechanism, the L1
delta, the A/B arms) is unchanged by that fix, which only alters when an already-captured
prefill trace is released.

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
a data sheet), device time **22.838 ms/token**, end-to-end token-out **23.315 ms/token**,
end-to-end logits-only **22.657 ms/token**.

Two gaps, both named:

* **roofline to device (37.9 %)** — 176 µs of a 431 µs sliding layer is non-matmul work
  that moves no weight bytes: 4 norms, `SdpaDecode`, two collectives, two cache updates,
  rotary, head create/concat, layout conversions. The projections themselves run at
  52–72 % of peak. This is the "modules built from many small ops sit lower" case, and
  the explanation is the requirement;
* **device to end-to-end (−0.8 %)** — device time comes out *above* the traced replay,
  because `tt-perf-report` merges a 4-device capture by taking the max per op. The sign
  is the useful part: there is no room for host work between them, which the zero
  per-token refresh counters say independently.

`22.657 + 0.632 = 23.289` against a measured token-out of `23.315`: the two traces
account for the step to within 27 µs, and that 27 µs is the caller's token readback.

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
| **P1** the host-dispatch gap was declined on an *unmeasured* cost model, and TTFT did not improve | the stage named a prefill trace as the only mechanism and never captured one; the two inputs to the "does not pay back" argument were estimates | captured it (`bench/prefill_trace_probe.py`): **59.80 → 44.96 ms, bit-identical, 98.16 ms capture, 3.3 MB retained, coexists with the decode traces**, then **shipped it** as `GeneratorConfig.prefill_trace` with a bounded bucket cache and a contract test. TTFT **63.66 → 50.19 ms, −21.2 %** (§2.5) |
| **P2** the host attribution had a 2x hole and a contradiction | the collective probe used a **BF16** payload in a hot loop and reproduced 60 µs against the model's 125–140; `perf_summary.json` said 12.1 ms where the README said 19.1 | re-measured at the model's **BFP8** payload, with a loaded queue, and with the device drained before each in-model collective: **117.05 µs loaded against 114.6 in-model drained** — the gap is the instruction stream, nothing is unattributed. `perf_summary.json` and the README now both say 20.93 ms, and the retracted "nothing moves it" is replaced by the measured **−14/−17 % from persistent buffers**, blocked on the decoder stage's correctness race (§2.4) |
| **P2** the teacher-forcing "after" range dropped this stage's own lowest measurement | the README quoted 37.51–38.50 while `evidence_accuracy.json` said 37.19 | all three runs reported, the ranges noted as overlapping, and **the +1.3 % claim withdrawn** |
| **P2** "none of the three changes allocates anything" was false for L1 | `ttnn.tanh`/`ttnn.multiply` have no in-place form, so change 1 moves two 3.24 MB transients into width-sharded L1 | measured (`bench/l1_highwater_probe.py`): **+126,976 B/bank of peak L1**, 1.24 MB/bank still free there; the sentence corrected, the pool/moment confusion with the 7,296 B semaphore figure explained, and it is now limitation 7 |

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
| **P1** the **decode** trace bakes the same KV-cache buffer addresses as a prefill trace and was never invalidated on a rebind | real, and the worse of the two: the prefill trace is opt-in and off by default while the decode trace runs on every token. `ttnn_decode_forward` calls `paged_update_cache(layer.k_cache, layer.v_cache, ...)`, so a caller that rebound to different buffers after capture got a decode reading and writing the buffers it no longer owned — wrong tokens, no error, and a log line about releasing the *prefill* traces that read as if the rebind had been handled. `_invalidate_prefill_traces_if_cache_moved` is now `_invalidate_traces_if_cache_moved`, each trace carries its own cache signature, and `_release_decode_trace()` drops the decode trace and the sampling trace captured over its logits. New test `test_decode_follows_the_cache_it_is_rebound_to_after_the_trace_is_captured` prefills prompt A, decodes (capturing), rebinds to a different cache holding prompt B, and asserts the **traced** decode agrees with the **eager** decode off B — which is the assertion that fails without the fix |
| **P2** the fabric-ERISC attribution was confounded with process length | the control was run three times; see above. Limitation 6, the watcher section, `run_watcher.sh`'s header and the serving advice are all rewritten around it |
| **P2** `doc/context_contract.json` carried this stage's numbers under `doc/full_model/...` provenance | the parent builder hardcodes both strings, so this stage's wrapper now re-points them from `EVIDENCE` after the parent runs, and the contract is regenerated. Same defect class as rounds 2 and 3 found in `perf_summary.json` and `evidence_perf_before.json` |
| **P2** the `$optimize` checklist row said the two traces account for the step to **38 µs** | no artifact yields 38; it is **27** (26.8 µs). Corrected, and the gate now binds the residual rather than only asserting a band |
| the gate's "asserts the set of artifacts it opens" was `is_file()` | the three readers now record every path they open and the coverage assertion tests *that* set. Two entries were caught immediately: `work_log.md` was being read past the recorder, and `watcher_probe_rebuild/watcher.log.gz` was listed but never opened — it is now read and checked for detach lines |
| `ADVERTISED_CHECKS` counted bindings as checks | the split is stated: **526 checks (492 assertions, 34 README bindings)**, and both numbers are asserted |
| the audit table called id 3119 a residual add | it is the attention-gate multiply; 3124 and 3191 are the residual adds. No value moves |
| the README read as if the profile pinned the SwiGLU grid | `Cores` reads 110 for every elementwise row, including the 52-core softcap pair. Footnoted, with what does pin it |
| `device_position_advances: 0` read as a contradiction | explained where it is incremented and why replays cannot bump it |
| the reproduce block's `WATCHER_TAG=_12case` matched no committed log | tags aligned to the committed names, and the ten-case repeats and the length-control arm added to the block |

## 11. Commits

Local checkpoints on `agentic-research/hous/muse-glimmer-30b`, on top of the full-model
stage's `93adb25b7a8`. Never pushed.

| SHA | what |
| --- | --- |
| `ee3c378a830` | `tt/model.py`, `tt/generator.py`, `tt/optimized_decoder.py`, `tests/test_full_model.py` — the three decode-path layout changes, the opt-in traced prefill, and the seven new acceptance cases |
| `3d03b5ca595` | `doc/optimized_full_model/` in full, the regenerated `doc/context_contract.json`, and the regenerated `readiness_autoregressive_{chat,raw}/` outputs |
| `5e6022db622` | round-3 review fixes: the audit-table partition, the corrected prefill-128 and host-dispatch figures, the retracted retirement flag, the `--arm rebuild` probe and the gate's coverage extension |
| *(this commit)* | round-4 review fixes: the decode-trace cache invalidation and its test, the fabric-ERISC length control, the context-contract provenance, and the gate's opened-artifact coverage |

Nothing unrelated is in any of them: `git status` is clean at each, and the
only files touched outside `doc/optimized_full_model/` are the four implementation/test
files, the contract, and the two readiness output directories the autoregressive stage
rewrites.

The repo's pre-commit hooks reformatted three bench scripts and the test file on the
first attempt; the reformatted state is what is committed, and the eight affected tests
were re-run afterwards (`logs/post_format_tests.log`).
