# T3K TP=8 multichip evidence

Target: Wormhole B0 T3K (4x n300, 8 chips), `ClusterType.T3K`, `(1, 8)` mesh, TP=8.
Baseline: the measured Blackhole P300_X2 TP=4 implementation in this same tree, which is the
only prior art for these kernels; there is no single-chip TTNN baseline for this model because
the decoder's policy assumes a mesh.

## Strategy

1D tensor parallelism on the single mesh axis, which is what `tech_reports/LLMs/llms.md` §3.3
recommends for a 1D mesh of eight chips. Weights shard on the projection axis, the residual
stream stays replicated, and collectives run as Ring over `FABRIC_1D_RING`.

Ring is not a free choice. `all_reduce_async` at the model's shapes completes in 0.06 s on
`FABRIC_1D_RING` and never completes on `FABRIC_1D`, because a ring collective needs the hop
from the last device back to the first. The same op as a Linear collective runs fine on
`FABRIC_1D`. Only the mismatch deadlocks, so `validate_fabric_topology` rejects it at
construction rather than leaving it to a device timeout minutes into serving.

`num_links` is 1. Two links on Ring hang and wedge the devices; the second ethernet link of
each chip pair carries the dispatch datapath.

## Per-device geometry at TP=8

Every head count below is already per-device. Seven QB2 constants had to become
platform-derived; each was a real hardware difference, not a bug.

| quantity | QB2 TP=4 | T3K TP=8 | why it moves |
| --- | --- | --- | --- |
| fabric packet payload | 8192 B | 6144 B | Three whole 2048 B pages, not the 7616 B ceiling |
| head / embedding shard | `vocab/4`, `hidden/4` | `/TP` | slice ran past the per-device tensor |
| sharded layernorm grid | `(10, 4)` | `(8, 5)` | `CoreCoord(9, ...)` does not exist on 8x8 |
| DRAM readers per bank | 2-3 | 1 | `num_workers_per_dram_bank > 1` is Blackhole-only |
| `down_cores` | 8 | 34 | `down_proj` K is `intermediate/TP`, 68 tiles; 8 does not divide it |
| LM head chunk width | 16384 | multiple of `banks*64` | an over-allocated bank shard returns wrong values silently |
| `prefill_1d_down_k` | 8 (GDN) / 17 | 17 | same 68 tiles, different matmul path |
| KV heads | 1 per device | 1 per device, replication 2 | 4 heads cannot shard 8 ways |
| `recurrent` state | `(B, 12, 128, 128)` | `(B, 6, 128, 128)` | `linear_num_value_heads/TP` |
| `conv` state | `(B, 3, 2560)` | `(B, 3, 1280)` | `2*k*128 + v*128` per device |

`down_cores` and `prefill_1d_down_k` are the same underlying fact reached by two paths, so
`tests/unit/test_policy_blocking.py` now checks every K block size and DRAM core count against
every role's K on both widths rather than relying on inspection.

### Fabric packet payload

The packet size is chosen by how many whole CCL pages it carries, not by the link ceiling.
`ccl_common.cpp` takes `min(hw_max / page, 4) * page`, so a 2048 B bfloat16 page gives 8192 B
on Blackhole and 6144 B on Wormhole; the QB2 value of 8192 was that arch's ideal rather than a
number to clamp against, which is what an earlier revision of this tree did. A 1088 B bfloat8_b
page gives 4352 B on both, which is already ttnn's default, so it is the prefill collective's
bfloat16 page that the setting has to serve.

`fabric_payload_bytes()` derives it for the demo and test paths, which are the only paths where
it takes effect. **The serving path cannot set it from here.** The vLLM plugin builds its own
fabric argument and drops the key: the served run logs
`tt/worker.py:796] Setting fabric config: {'config': FabricConfig.FABRIC_1D_RING,
'reliability_mode': FabricReliabilityMode.STRICT_INIT}`, with no router config, and
the size can only be carried by the last argument of
`ttnn.set_fabric_config(config, reliability_mode, num_planes, fabric_tensix_config,
fabric_udm_mode, fabric_manager_mode, router_config)`, where `router_config` is a
`ttnn.FabricRouterConfig` whose single field `max_packet_payload_size_bytes` defaults to `None`.
Passing only the first two arguments leaves that field unset, which is the 4352 B default. There
is no environment variable for it.

Benchmark run 36682118819 proves the key is inert rather than merely undocumented. It ran with
`fabric_max_packet_payload_size_bytes: 6144` accepted into `additional_config` and echoed in the
engine arguments, and the runtime still reported the same thing it reported for the run that set
nothing at all: `Fabric packet size 4352 B is suboptimal for transporting 2048 B pages.
Configure 6144 B packet size to maximize throughput.` Decode throughput was unchanged across the
two, 48.0 against 49.6 tokens per second at eight concurrent requests, which is what an inert
setting predicts. Closing this needs a change in the plugin, not in this tree or its spec.

## Context contract

`doc/context_contract.json`. The full advertised 262144-token window is kept: a single
full-context request costs 2.14 GiB of the 7.192 GiB left per device after weights and the
trace reserve. Only 16 of the 64 layers hold a paged KV cache; the other 48 hold 18.4 MiB of
recurrent state that does not grow with context.

## Correctness

Against an fp32 HuggingFace reference, with the bfloat4_b weight-quantization floor as the
control, because the precision policy costs far more than any absolute PCC threshold worth
writing down:

| check | result |
| --- | --- |
| per-layer prefill, `linear_attention` | 0.999784 against a 0.999325 floor |
| per-layer prefill, `full_attention` | 0.978978 against a 0.980890 floor |
| end-to-end logits, 7 steps | matches or beats the same floor at 3 of 7 |
| greedy agreement, teacher-forced | 7/7 |
| DRAM head vs interleaved head | `logit_pcc` 0.9999963, identical greedy tokens |
| replica agreement across 8 devices | bit-identical (`dev_spread` 0) |

Full-vocabulary PCC reads 0.82-0.98 and top-k PCC is no better, so the divergence is not
confined to an irrelevant tail. It is the precision policy: an fp32 reference carrying nothing
but bfloat4_b weights scores the same band. Softmax KL is useless here, reading 0.00000 beside
a top-1000 PCC of 0.822, because a wide top-1 gap makes the distribution a near-delta on both
sides.

`tests/test_layer_pcc.py` gates per-layer PCC against the floor rather than a fixed number, so
it self-calibrates if the precision policy changes. Its `FLOOR_MARGIN` of 0.01 is a hand-set
slack, not a derived bound: `full_attention` measures 0.978978 against a 0.980890 floor, so it
passes on that margin rather than on merit.

The head is not where the logit error lives. The shipping `config/precision.json` is
`head_bfp4_lofi`, putting the LM head at `bfloat4_b` and LoFi. Running the same prompt and the
same teacher-forced token sequence against the same fp32 reference under a policy that differs
only in the head group, `bfloat8_b` and HiFi2, moves mean top-1000 PCC from 0.8963 to 0.9024:

| step | bfp4/LoFi head | bfp8/HiFi2 head | delta |
| --- | ---: | ---: | ---: |
| prefill | 0.8461 | 0.8544 | +0.0083 |
| decode1 | 0.8220 | 0.8263 | +0.0043 |
| decode2 | 0.8846 | 0.8894 | +0.0047 |
| decode3 | 0.9078 | 0.9121 | +0.0042 |
| decode4 | 0.9639 | 0.9691 | +0.0052 |
| decode5 | 0.8974 | 0.9054 | +0.0080 |
| decode6 | 0.9519 | 0.9602 | +0.0083 |

Top-10 overlap against the reference is identical at every step under both policies, and greedy
agreement is 7/7 under both. So the head accounts for roughly 0.6 of the ten-point gap from
1.0; the remainder is the `bfloat4_b` projections compounding over 64 layers. That also earns
the shipping head choice: the cheaper head costs 0.006 PCC and changes no ranking.

## Behaviour

Full 64 layers: correct Rayleigh-scattering answer, `17 * 24 = 408` with the right stop token,
identical output across replays. Paged KV cache and warmed traced decode replay both exercised;
400 decode steps with 16 scheduler-style slot remaps held a flat 6.4 s per 50 steps with no
drift, and 40 cycles of interleaved prefill and decode were equally stable.

### Serving batch is bounded at 16 by a correctness failure, not by capacity

Attention decode returns wrong results at batch 32 on this mesh: per-user PCC 0.019, against
roughly 0.9999 at batches 1, 2, 4, 8 and 16. Nothing between 17 and 31 has been measured. The
model, the generator and the vLLM adapter therefore all refuse a batch above 16 before weights
load, and the refusal is unconditional: it cannot depend on `QWEN_DECODE_BUCKETS`, because the
served configuration leaves that unset and so takes the `_decode_fixed` path that produces the
wrong values rather than the bucket path that already rejected the width.

This is a separate limit from the capacity frontier in `doc/context_contract.json`, which shows
batch 32 at 32768 tokens needing 9.07 GiB against a 7.192 GiB budget. Batch 32 is unusable on
both grounds, and the memory result should not be read as the reason.

The cause is not diagnosed. It is per-user PCC, so it is a slot-indexing or state-partitioning
suspicion rather than a numerics one, but nothing here narrows it further.

## Instrumented run

Waypoint-and-assert clean over a full prefill and traced decode: 21542 lines, zero trip
markers, clean detach, and `retraining events: 0` on all 40 ethernet rows.

This is **not** an unqualified watcher-clean run. A bare `TT_METAL_WATCHER=10` cannot link on
Wormhole: the instrumented fabric erisc router puts `.text` at `0xee08`, past the `0xEBE0` end
of `ERISC_APP_KERNEL_CODE`. The run above therefore has `NOC_SANITIZE`, `SANITIZE_NOC` and
`ETH` disabled, which is exactly the coverage that would catch an out-of-bounds transaction, so
that fault class remains unchecked. A failed watcher build also leaves the devices needing
`tt-smi -r`; a subsequent run reports an unexpected `run_mailbox` value until it is reset.

## End-to-end eval

`r1_gpqa_diamond` was run against a served instance of this tree on a T3K, tt-shield run
36587180186, on tt-metal `04bc281998a` and tt-inference-server `39906703`, which is the commit
that routes the T3K serving class at this tree rather than the Blackhole one. The workflow logs
name `qwen38_27b_t3k` 42 times and the QB2 tree not at all, so the run exercised this
implementation.

| metric | value |
| --- | ---: |
| score | 84.85 |
| published reference | 89.2 |
| ratio | 0.9512 |
| samples scored | 198 |
| samples failed | 0 |
| mean seconds per task | 163.9 |
| serving duration | 9 h 20 m |

The harness recorded the accuracy check as a pass and acceptance as a pass. **That pass does
not hold, and this run does not establish an accuracy result.** The threshold is 0.95 x 89.2 =
84.74 and the score is 84.8485, a margin of 0.11 points against an `exact_match_stderr` of 2.56
points, so it sits 0.04 standard errors above the line. Three of the 168 credited samples commit
no answer at all, and excluding any one of them fails the check: 0.9456 at one, 0.9399 at two,
0.9342 at three. Earlier preserved full runs of this model on this mesh scored 81.31 and 81.82.
A single stochastic run at temperature 1.0 cannot settle this gate in either direction.

Three things about the run were checked rather than assumed, because an earlier run in this
project reported a passing score of 35.0 while the server was dead for 33 of 40 prompts, its
error sentinels letter-matched into spurious credit:

- No sample carries `__INFERENCE_ERROR__` or `__PARTIAL_OUTPUT__`, so the sentinel guard that
  exists because of that earlier run reports zero failures. **The guard does not cover this
  run's actual defect.** Searching instead for the answer form `boxed{`, five of the 198
  responses commit no answer, and three of those five are scored correct: doc_ids 48 and 71,
  cap-truncated mid-sentence at 92,758 and 103,840 characters, and doc_id 127 at 11,218
  characters. The task extractor synthesises the choice text as an alternative to the letter,
  which is how a trace with no committed answer earns one. The generation cap is 32768 tokens,
  set in tt-inference-server `f81066cc`. Two of the seven responses lacking a closed `</think>`
  do carry a real `boxed{` answer, doc_ids 99 and 107, and both score-0 responses, 79 and 147,
  lack one. An earlier revision of this section reported these counts wrongly because it matched
  the bare substring `boxed`, which also occurs in the prompt instruction echoed inside the
  reasoning text.
- The `tt_triage` capture is routine, not a hang record. Its only matches for timeout, fatal
  error or engine death are an unknown-environment-variable warning for `VLLM_RPC_TIMEOUT` and
  `EngineCore loop active`; the log ends on `EngineCore waiting for work` one second before the
  samples file was written.
- The duration is eval work, not a stall: 163.9 s per task over 198 tasks is 9 h 01 m of the
  9 h 20 m.

The intermittent `binary_ng` device hang that ended three earlier serving runs did not occur in
nine hours of continuous serving. That is consistent with the fabric-and-topology mismatch
having been the cause of those hangs rather than a separate defect, though it does not prove it.

The run produced no benchmark, spec-test or agentic block. What it does establish is
stability: nine hours of continuous serving at TP=8 with the engine alive and idle at the end.
It does not establish accuracy. The mandatory text-LLM gates `meta_ifeval` and `meta_gpqa_cot`
were not run and no linked issue covers their absence.

The report metadata labels the implementation `qwen36` and the model id
`id_qwen36_Qwen3.8-27B_t3k`. That is a stale `impl:` key on the spec entry, not a wrong code
path, since `TT_MODEL_CLASS_OVERRIDES` does the routing and the logs confirm which tree loaded.
Correcting it would change `model_id`, which keys the eval artifact directory.

## Gaps

- No PCC against a single-chip TTNN baseline, which is what would separate sharding and
  collective error from HuggingFace-versus-TTNN numerics. The measurements above bundle both.
  The baseline itself is now known to run: `Qwen38Decoder` with `replicated_mesh=True` builds
  and decodes an unsharded layer of either kind on this mesh once the Blackhole core counts are
  replaced. Only four keys are illegal here, since `output_cores` at 48 and `down_cores` at 32
  already divide the unsharded K of 192 and 544 tiles; `attention`, `gate`, `up` and `residual`
  move from 80 to 40 cores, `rectangular_working` goes false, and the readers drop to one.
  The remaining work is the PCC comparison itself, not the baseline.
- No `tt-perf-report` for this path. The KDA blocker is real but narrow:
  `kda_performance_model.cpp` asserts Blackhole and `qkv_causal_conv1d_silu` and
  `sigmoid_gated_rms_norm` reach it from `create_op_performance_model`, so only the 48
  `linear_attention` layers cannot be profiled. The 16 `full_attention` layers, the LM head,
  the norms and every collective can be, and this tree already holds wormhole_b0 profiler
  results for the MLP and CCL sweeps. The missing report is a scoped exclusion, not an
  impossibility.
- No runtime fallback audit.
- Batched prefill has not been swept across the length branches; the sweep below is batch 1.
- 262144 tokens is a capacity result, not a latency or quality result; no run at that length
  has been executed on this mesh.
- Single-chip-versus-multichip speedup has no valid referent and is not reported. Unsharded
  weights are roughly 27 to 30 GiB against 12 GiB per chip, so a single-chip full model cannot
  exist; tensor parallelism here is what makes the model representable, not a throughput
  choice. A per-layer proxy was measured and rejected: calling `decode_forward` directly runs
  eagerly, and at 5.0 ms per layer against the 0.968 ms traced marginal cost it is 5.2 times
  dispatch-bound, with that overhead identical in both arms. It reports 1.05x for a GDN layer
  and 1.21x for a full-attention layer, which measures host dispatch rather than parallel
  efficiency.
- No top-1/top-5/top-100 agreement against a reference implementation and no AIME24 run. The
  qualitative suite in `readiness_qualitative/` covers prompt format and answer quality on the
  shared six prompts at 256 tokens against a native-bfloat16 control, and the GPQA Diamond
  result above is an external accuracy gate, but neither is a logit-level agreement measurement.
- The residual and fused-CCL families are still unmeasured here. `fused_grid` and
  `rs_core_offset` hold Blackhole values that no 8x8 worker grid can satisfy, so `mmrs`,
  `agmm` and the sharded-residual path cannot run without T3K-legal values first. A shape
  error is not a rejection, so this family remains open.

## Batch-1 performance

Warmed, prompt 128 and generate 128, the shape the serving profile uses, at the shipping
`ccl_dtype`, now bfloat8_b:

| metric | shipping | all bfloat16 |
| --- | ---: | ---: |
| TTFT | 244.9 ms | 247.0 ms |
| decode, token out | 60.338 ms, 16.57 t/s/u | 72.321 ms, 13.83 t/s/u |
| decode, no readback | 59.531 ms, 16.80 t/s/u | 71.522 ms, 13.98 t/s/u |

The collective dtype is split by phase: bfloat8_b for decode, bfloat16 for prefill.

Both decode boundaries are reported because they answer different questions: the no-readback
figure is the logits-side comparison, and token out adds the final norm, LM head, sampling and
the caller-visible readback. The boundary between them costs 0.799 ms, 1.10% of the step.

**That 0.799 ms is the readback, not the sampler, and an earlier revision wrongly read it as
evidence that sampling is cheap.** The sampling trace replays on both sides of that boundary,
so the difference never contained it. Measured directly on this mesh, `_sampling_step` is
**8.817 ms** eager at batch 1, with a p10 to p90 spread of 0.03 ms -- tight enough to indicate
device-bound work rather than host dispatch, unlike a decoder layer which spreads 4.28 to
7.65 ms eager against 0.72 to 0.97 ms traced. It is the largest single cost outside the layer
stack. See the fixed-cost attribution below.

Steady-state host work is one refresh each of token, position and RoPE per 148 trace replays,
not one per generated token, so decode state advances on device rather than from the host.

### Layer-stack lower bound

Decode latency is linear in layer count. Least squares over four depths, each measured the same
way, gives 0.9676 ms per layer with a 9.600 ms intercept and residuals within 0.010 ms:

| layers | no readback | token out |
| ---: | ---: | ---: |
| 4 | 13.468 ms | 14.383 ms |
| 16 | 25.078 ms | 25.966 ms |
| 32 | 40.573 ms | 41.350 ms |
| 64 | 71.522 ms | 72.321 ms |

The 64-layer stack is therefore 61.93 ms and the full-model-only cost is 10.40 ms, 14.4% of the
step. Only 0.799 ms of that is the token-out boundary, so the remainder is the embedding
all-gather, the final norm, the LM head and logits movement, all of which sit inside both decode
measurements. The LM head is the largest single candidate there.

This is a marginal per-layer cost measured inside the real traced loop, not a standalone
optimized per-layer latency, because no such per-layer measurement exists for this mesh.

### Against the Blackhole reference

QB2 publishes 39.4 t/s/u and 67.9 ms TTFT at the same shape, so this mesh reaches 35% of its
decode throughput and 3.6 times its TTFT. With the bfloat8_b collective measured below it would
be near 16.8 t/s/u, still 43%. That is consistent with the platform deltas recorded above:
64 worker cores against roughly 110, one usable ethernet link against two, and one DRAM reader
per bank because multiple readers are Blackhole-only. It is a hardware gap rather than a defect,
but it is larger than the phrase "slower on Wormhole" would suggest.

## Fixed per-step cost, attributed

The layer-stack fit reports a 9.600 ms intercept, which is a derived number rather than a
measurement, so it was attributed directly. Traced decode was measured at shallow depths where
the original fit had no data (its depths were 4, 16, 32 and 64):

| depth | traced decode, token out | the 4/16/32/64 line predicts |
| ---: | ---: | ---: |
| 1 | 13.259 ms | 10.568 ms |
| 2 | 13.029 ms | 11.535 ms |
| 4 | 14.984 ms | 13.470 ms |
| 8 | 17.967 ms | 17.341 ms |

A shallow fit gives 0.7249 ms per layer on a 12.091 ms intercept, so the fixed cost is real and
larger than the original fit implied; depth 2 is inside noise of depth 1, which is what a
dominant fixed term looks like. Layer composition does not explain the slope difference: depth
8 is 6 GDN and 2 full-attention, the same 25% full-attention fraction as the whole model.

The components, measured eager at batch 1 on a one-layer model, since none of them depends on
depth:

| component | eager p50 | p10 to p90 |
| --- | ---: | --- |
| `_sampling_step`, top-k then two all-gathers then sample | **8.817 ms** | 8.808 to 8.840 |
| `logits`, norm and head and moves and concat | 1.469 ms | 1.415 to 1.571 |
| `rope` | 0.960 ms | 0.940 to 0.981 |
| `embed`, including the ring all-gather | 0.889 ms | 0.865 to 0.967 |
| sum | 12.135 ms | against a 12.091 ms intercept |

Two traces and a token readback -- the whole scaffolding of a decode step -- cost 0.278 ms
together, measured with no model present: 0.060 ms for a device synchronise, 0.160 ms for the
readback, and about 0.05 ms per trace launch. So neither trace replay nor host synchronisation
is where the fixed cost goes.

The sampler runs in the served path. `compat` comes from the generator's `host_sampling`
constructor argument, not from `QWEN_VLLM_HOST_COMPATIBILITY`, which only permits host paths
rather than selecting them; the adapter passes `host_sampling=not device_sampling`, and benign
sampling parameters leave device sampling on.

Eager timings carry per-op host dispatch and so bound the traced cost from above. They are used
here to locate the dominant term, which they do unambiguously, not to state its traced value.

### The sampler's top-k ran on one core, and no longer does

`248320 / 8` is 31040, which is not a power of two, and the multi-core bitonic top-k network
requires one (`topk_multicore_structurally_eligible` in `topk_utils.cpp`). The large-indices
route that would otherwise carry this width returns false off Blackhole (`topk.cpp:362`), which
is why the identical configuration costs little on QB2. So the top-k fell to the single-core
factory across the whole row, at roughly 137 ns per element (`topk.cpp:248`), or about 4.25 ms
of the 8.8 ms.

`pad_logits_to_power_of_2=True` pads the row to 32768 for one `ttnn.pad` and restores
multi-core eligibility:

Profiled per-op device time, batch 1, one sampling step:

| | top-k device time | cores | step device FW |
| --- | ---: | ---: | ---: |
| unpadded | 7625.58 us | **1** | 7.936 ms |
| padded to 32768 | **276.07 us** | **17** | **0.664 ms** |

The top-k is 27.6 times faster and the step's device time falls 91.6%, a saving of 7.27 ms.
Against a 72.32 ms token-out step that is about 10%. The sampler's cost is fixed in batch --
tile padding makes one row and 32 rows the same 970 tiles -- so the same absolute saving is
spread across however many users are being served.

Host-timed eager figures were 8.809 ms and 3.968 ms, which is -55% and understates it. The
reason is that an unpadded step is one 7.6 ms blocking op, behind which host dispatch hides;
once the device work collapses, dispatch for 29 programs on an eight-device mesh is exposed and
dominates the eager number. A traced decode pays the device time and not the dispatch, so 7.27 ms
is the figure that applies to a real step.

This also settles the composition of the 12.091 ms fixed term: the unpadded sampler is 7.94 ms
of it, 66%, so the eager attribution that put it at 73% was close for the wrong reason.

Where the remaining 0.664 ms sits, per step: top-k 276.07 us on 17 cores, `ttnn.sampling`
76.17 us on 32, two all-gathers 74.54 us, fill-pad 74.20 us, twelve tie-break binaries 56.66 us,
the new pad 38.11 us, manual seed 26.90 us, and 41.7 us across the rest, over 29 programs.

The tie-break programs are 8.5% of the step. Making them redundant by flipping `_topk_stable`
in `models/common/sampling/tt_sampling.py` would therefore buy at most about 57 us while
changing shared code with 88 callers, so that lever is closed on evidence rather than untried.

This was a misconfiguration rather than a discovery.
`should_pad_sampling_logits_to_power_of_2` in `models/tt_transformers/tt/model_config.py:202`
already returns true exactly when per-device vocab is not a power of two, for issue 40399,
"models that regress to single-core TopK". The Blackhole sibling sets it; this tree did not.

Padding is safe because the pad fills with `-float_max`, which cannot win a top-k against any
finite logit, and the per-device index offset still uses `padded_vocab_size // 8`. Verified
rather than argued: greedy sampling returns token 103695 against a host argmax over all 248320
columns, both before and after.

Profiling this path needs a `full_attention` layer, every fourth index. A `linear_attention`
layer reaches the KDA performance model, which asserts Blackhole and aborts the run under the
profiler, so 48 of the 64 layers cannot be profiled on this mesh. That is a real constraint on
any future profiling here, confirmed by hitting it rather than inferred.

## Served performance, which is not the traced-decode performance

Benchmark run 36682118819 on tt-metal `5936475725f`, through vLLM in the release harness, over
twenty ISL/OSL/concurrency combinations:

| concurrency | ISL | TTFT | TPOT | output tput |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 324.0 ms | 168.6 ms | 5.9 t/s |
| 1 | 1024 | 618.9 ms | 170.5 ms | 5.7 t/s |
| 1 | 4096 | 1679.6 ms | 169.3 ms | 5.5 t/s |
| 1 | 16384 | 5946.0 ms | 172.7 ms | 4.6 t/s |
| 1 | 32768 | 12001.5 ms | 172.5 ms | 3.8 t/s |
| 8 | 128 | 3240.2 ms | 159.5 ms | 43.6 t/s |
| 8 | 4096 | 12486.0 ms | 161.4 ms | 31.0 t/s |
| 8 | 16384 | 46856.0 ms | 167.1 ms | 15.0 t/s |

**TPOT is 168.6 ms served against the 60.3 ms this tree measures locally at the same shape.**
The batch-1 figure recorded under Batch-1 performance is a traced-decode measurement through
`generate()`, and it does not survive the serving loop. Nothing here supersedes it; the two
measure different things, and the serving number is the one a user sees.

Most of that difference is outside this tree. `decode_forward`, the entry point the plugin
calls, was driven directly on this mesh in both the configuration the plugin uses and the one
the local loop uses, with nothing else changed:

| batch | plugin's reload path | steady-state path | reload cost | served TPOT | above `decode_forward` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 63.7 ms | 58.3 ms | 5.37 ms | 168.6 ms | 104.9 ms, 62% |
| 8 | 93.6 ms | 89.1 ms | 4.51 ms | 159.5 ms | 65.9 ms, 41% |

So the model path accounts for 38% of a served step at batch 1 and 59% at batch 8, and the rest
is work the plugin and vLLM do around the call. The steady-state figures also confirm the
traced-decode numbers reproduce through the adapter rather than only through `generate()`:
58.3 ms against 60.3 ms at batch 1, and 89.1 ms against 89.5 ms at batch 8.

The served configuration was also missing every optimization flag `tests/run_ci.sh` sets, which
is a second and larger reason the served numbers were poor. Measured on this mesh at batch 8
with all ten set against none:

| batch 8 | no flags | all flags | gain |
| --- | ---: | ---: | ---: |
| plugin reload path | 93.60 ms | 83.52 ms | 10.8% |
| steady state | 89.10 ms | 78.28 ms | 12.1% |

With eight slots active the decode bucket equals the batch, so that figure is what compact
decode attention and MLP plus batched RoPE are worth; the bucket is worth more and separately.
Without `QWEN_DECODE_BUCKETS` every step takes the full batch-8 shape however few requests are
active, so one user pays 89.1 ms for one token where a batch-1 step costs 58.3 ms. Without
`QWEN_BATCHED_PREFILL` prefill serves one request at a time, which is why a served TTFT of
344 ms at one request became 3482 ms at eight. Every local figure recorded above this section
was also measured without these flags, so they understate the model path too.

`reset_batch` is what separates the two arms. `decode_forward` treats it as
`refresh = reset_batch or not self._decode_bound or ...`, and a refresh rebinds sampling and
rewrites tokens and positions from host; the plugin passes it on every step while it reports
that this adapter does not advertise `decode_input_update_contract >= 1`. That costs 4.5 to
5.4 ms, about 5%, measured at both batches. Worth advertising the contract for, but it is not
the gap, and the earlier guess that it was is wrong.

The remaining cost is measured, not explained. It is **not** a fixed per-step host cost: a fixed
cost would be equal at both batches and it is 105 ms against 66 ms, so something in it
amortizes with batch. Attributing it needs a profile above `decode_forward`, which is why that
entry point now carries optional timing: `QWEN_DECODE_STEP_TIMING=N` logs its own p50, p10 and
p90 every N steps, so a serving run reports the split against the harness's TPOT directly. The
instrumentation agrees with external timing to 0.1 ms.

The shape of the gap identifies where it is not. TPOT is flat within 13 ms across every input
length from 128 to 32768, so it is not attention or KV work, which grow with context. It barely
improves from concurrency 1 to 8, 168.6 ms to 159.5 ms, where this tree's own local measurement
moves the other way, 60.3 ms to 89.5 ms at batch 8, because more batch is more compute. A
per-step cost that ignores both context and batch is a fixed cost per step, either host work that
does not overlap the device or a constant inside the collectives.

It is not async scheduling. The run logs `Asynchronous scheduling is enabled.` twice, the
plugin's own `Disabling async scheduling` warning never fires, and `TTScheduler` subclasses
`AsyncScheduler`. The `scheduler.py:192` warning about degraded performance is vLLM's standard
notice for any custom scheduler class and is conditional on subclassing `Scheduler` instead,
which is not the case here.

Async scheduling being enabled does not prove the host work overlaps the device step, so the
cost is still unattributed. Separating it needs a decode-step trace on the tt-metal side, host
time against device time for one step; until that exists, nothing here says whether the fixed
cost is host-side or in the collectives.

Acceptance reported `PASS` for this run on `0/26 passed, 6 waived, 20 NA`. No benchmark target
was met: the strictest tier wants 16.89 t/s/u and the functional tier 1.68, so only the
functional tier passes on throughput while every `complete` and `target` tier check fails.
TTFT passes all three tiers at every length.

## Served performance after the flags, the padding and decode-only tracing

Benchmark run 37001043718, tt-metal `8ca73539dd7` and tt-inference-server `ab5d20e9`, against
run 36719378594 on `a2675e8ea96` / `9074368a`:

| concurrency | ISL | TPOT before | TPOT after | t/s/u before | t/s/u after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 96.1 ms | **48.1 ms** | 10.4 | **20.8** |
| 1 | 4096 | 96.9 ms | 48.9 ms | 10.3 | 20.4 |
| 1 | 16384 | 98.3 ms | 49.4 ms | 10.2 | 20.2 |
| 1 | 32768 | 159.3 ms | **50.2 ms** | 6.3 | **19.9** |
| 1 | 131072 | 167.1 ms | 55.9 ms | 6.0 | 17.9 |
| 8 | 128 | 87.5 ms | **70.4 ms** | 11.4 | **14.2** |
| 8 | 32768 | 97.2 ms | 78.1 ms | 10.3 | 12.8 |

Acceptance reports 4 of 26 benchmark targets passed, the first run in which any passed. The
strictest `target` tier now passes at concurrency 1 for ISL 128, 1024, 4096 and 32768, exceeding
its 16.89 t/s/u bar by roughly 23% where the same ratio was 0.35 before. One check still fails:
`complete`-tier output throughput at ISL 32768, ratio 0.87.

**The step between 16384 and 32768 is gone.** It was 98.3 to 159.3 ms and is now 49.4 to 50.2 ms.
That step was not a context cost at all: without decode buckets every step ran the full batch-8
shape however few requests were active, so a single request at 32768 paid eight requests' worth
of attention and cache work. `QWEN_DECODE_BUCKETS=1` makes a lone request take a batch-1 step.
The sampler padding cannot explain it, since the sampler's cost is fixed in context.

**Three changes landed together, so the attribution below is a reconciliation and not a proof.**
The earlier run predates the optimization flags: its tt-inference-server commit `9074368a` is the
step-timing commit, and `6e0e16ea` added the flags afterwards. So this run carries the nine
`run_ci.sh` flags, the sampler top-k padding, and decode-only tracing at a 128 MiB region.

Against the local measurements the parts add up. At concurrency 8 the decode bucket contributes
nothing, because eight active slots select the batch-8 bucket, leaving the compact-decode flags
at about 12% of an 89 ms step, near 10.7 ms, plus 7.27 ms of sampler device time, near 18 ms
against the 17.1 ms observed. At concurrency 1 the bucket adds the difference between a batch-8
and a batch-1 step, 89.1 against 58.3 ms locally, which brings the expected total near the 48 ms
observed.

**Prefill is noisier and partly worse.** Concurrency-1 TTFT improves at short input, 344.4 to
141.6 ms at ISL 128, and is flat at long input. At concurrency 8 it is unreliable between runs
rather than simply better or worse: ISL 128 with OSL 128 rose 48% while ISL 128 with OSL 1024
fell 44%, and TTFT does not depend on output length, so those two disagree about the same
quantity. Scheduling, not a property of the configuration.

`trace_mode: decode_only` with a 128 MiB region replaced a 1 GiB region with prefill tracing,
which left too little DRAM for the batch-8 startup prefill warmup that `QWEN_PREFILL_STARTUP_WARMUP`
requests. Decode remains traced, which is why TPOT is the clean signal here and TTFT is not.

## Sequence-length branches

Every length-dependent gate in the shipping policy was exercised at its value and on both
sides: `dram_prefill_max` 32, `prefill_1d` 64 to 256, `minimal_role_min` 128 for output and
down, `minimal_prefill_min` 512 for every role, and the 4096-token chunk loop. Lengths 31, 32,
33, 63, 64, 65, 66, 67, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1024, 4095, 4096, 4097 and
5000 all prefill and generate coherently, with no failure at any of them. Thirteen of the
non-tile-aligned lengths are in that list, so the public path does not require a length
divisible by the tile, page or chunk size.

Two independent oracles back the run rather than just an exit code:

The same prefix was reached a second way for every length up to 513, by prefilling 32 tokens
and then teacher-forcing the remaining corpus tokens through decode. That traverses
`direct_allreduce` and the DRAM-sharded projections instead of `prefill_1d` and `_minimal`, and
the two routes agree on the predicted token at all fifteen cross-checked lengths.

The probe corpus is a sentence tiled to length, which has a period of 13 tokens, so lengths
sharing a residue must predict the same token. Across seven residues holding more than one
length there are no mismatches, which carries the short-length decode agreement up to the long
lengths that are too expensive to reach through decode: 4095 pairs with 65, 1024 with 127 and
257, 5000 with 255, and the chunk boundary itself, 4096 and 4097, with 66 and 67.

## Collective dtype

`num_links` is 1 here against QB2's 2, so the collective carries twice the hops on half the
links, and the dtype it moves was carried over from the two-link measurement. Measured warmed,
single variable through `precision_config`, 20 warm-up steps then 100 timed steps at ISL 128:

| batch | ccl bfloat16 | ccl bfloat8_b | saving |
| ---: | --- | --- | --- |
| 1 | 71.522 ms, 13.98 t/s/u | 59.543 ms, 16.79 t/s/u | 11.98 ms, 16.8% |
| 8 | 101.411 ms, 9.86 t/s/u | 89.462 ms, 11.18 t/s/u | 11.95 ms, 11.8% |

Every arm held a spread under 0.2% between p10 and p90. The saving is the same 12 ms at both
batches, not a larger fraction at higher concurrency: the all-reduce workspace is
`[1, 1, 32, 5120 * TP]`, tile-padded to 32 rows whatever the batch, so its payload is fixed per
decode step and halving the dtype halves a constant cost. That puts the bfloat16 collective at
roughly 24 ms of every step, about 0.37 ms per layer.

It costs nothing measurable. Against the same fp32 reference and the same teacher-forced
tokens, mean top-1000 logit PCC moves 0.8963 to 0.8989, two of seven steps negative and five
positive, with identical top-10 overlap at every step and 7/7 greedy agreement. Free-running
greedy output is byte-identical across all four arms above.

This is now the default for this tree, set in `config/precision.json` and in the `BASELINE`
fallback so that selecting `baseline` does not quietly give back the saving. The precision
policy is the only level that can own it, because `decoder_policy` supplies `ccl_dtype` after
the platform overlay is merged.

Applying it to both phases made prefill 5.3% slower, 247.03 ms to 260.22 ms, because a prefill
collective carries the whole sequence rather than the tile-padded 32-row decode workspace and
the typecast then costs more than the halved payload saves. Splitting the dtype by phase
recovers that without giving up the decode win, so `ccl_dtype_prefill` holds bfloat16 in the
platform overlay while the precision policy keeps bfloat8_b for decode. The two live prefill
collective sites, the general reduce-scatter in `_linear` and the distributed norm gather in
`_norm`, select on sequence length:

| config | TTFT | decode token out | 128 in, 1 out | 128 in, 128 out |
| --- | ---: | ---: | ---: | ---: |
| bfloat16 both | 247.03 ms | 72.321 ms | baseline | baseline |
| bfloat8_b both | 260.22 ms | 60.216 ms | +0.3% | -16.2% |
| split | 244.85 ms | 60.338 ms | -4.4% | -16.2% |

The split is the better configuration at every workload length: TTFT returns to baseline within
noise, decode holds to 0.12 ms of the all-bfloat8_b figure, which is inside the p10 to p90
spread of either, and the single-token request that the uniform policy slightly lost now gains.

`ccl_dtype_prefill` can live in the platform overlay because `decoder_policy` supplies only
`ccl_dtype`, so unlike the decode value it is not overwritten by the caller.
