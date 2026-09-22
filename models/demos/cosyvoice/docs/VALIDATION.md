# Validation evidence, requirement by requirement

Every requirement in the bring-up scope, what validates it, and where its number
lives. This document is a map, not a second copy of the numbers: measured figures
appear once, in [`../PERF.md`](../PERF.md), and are linked from here. Two documents
quoting the same measurement is how they drift apart.

Read the columns as:

* validated by — the test that decides it. A name in `tests/` is executable; run
  it and it passes or fails. Where a requirement is validated by inspection or by a
  measurement with no check, the cell says so rather than naming a test that does not
  enforce it.
* evidence — the section of `PERF.md` carrying the number, or the artefact.

## How the numeric thresholds are enforced

Before 2026-08-29 the perf suite printed its figures and asserted `total_s > 0` — a
timing harness, not a check. It now enforces every numeric threshold through
[`../tests/perf/gates.py`](../tests/perf/gates.py), which separates two things that
were previously conflated:

* `GATES` — the thresholds themselves, quoted from the scope. One declaration,
  used by every test that produces one of these numbers.
* `EXPECTATIONS` — the per-architecture verdict. A threshold recorded as met is
  asserted against the requirement, so a regression fails. One recorded as unmet
  is asserted against its recorded measurement, both directions, exactly as
  `models/perf/device_perf_utils.check_device_perf` does: slower fails because that is
  a regression, and *faster* fails because it means the published figure is stale.

Nothing is `xfail`-ed. An unmet target gets a measured number, a band it must stay
inside, and a named lever.

| threshold | scope stage | enforced in |
|---|---|---|
| `>= 30 tok/s` semantic generation | Stage 1 | `test_device_end_to_end_rtf`, `test_device_traced_throughput`, `test_device_inplace_throughput` |
| `RTF < 0.5` | Stage 1 | `test_device_end_to_end_rtf` |
| `>= 60 tok/s` | Stage 3 stretch | same three as the 30 tok/s threshold |
| `RTF < 0.2` | Stage 3 stretch | `test_device_end_to_end_rtf` |
| token agreement `> 95 %` | Stage 1 | `test_gate1_teacher_forced_argmax_match`, `test_gate1b_teacher_forced_argmax_through_the_kv_cache` |
| per-module PCC `>= 0.99` | Stage 1 | every `tests/pcc/test_device_*` |
| streaming content equivalence | Stage 3 stretch | `test_device_streamed_matches_non_streamed` |
| batching amortises the weight read | Stage 3 | `test_device_batched_decode_throughput` |
| streaming starts before generation ends | Stage 3 | `test_device_streaming_first_audio_latency` |

`WER < 3.0` and `speaker similarity > 60` are not enforced by a test in this tree,
and that is a deliberate split rather than an omission: scoring needs whisper
`large-v3` and `WavLMForXVector` in the reference venv (see
[`security.md`](security.md)), which tt-metal's `python_env` does not carry and this
demo does not install. They are produced by `scripts/eval_wer_sim.py` and reported in
`PERF.md`'s *Speech quality*.

## The device matrix

Three boards, one commit, one day — see `PERF.md`'s *The certification run* for the
commit, the date and the per-board environment. Every figure in `PERF.md` comes from
that run; where a board has no result, the cell is empty rather than filled from
the other board of the same architecture.

| board | architecture | in scope because |
|---|---|---|
| Wormhole n300 | Wormhole | the named Wormhole target |
| Blackhole `p150a` | Blackhole | actively cooled, the faster of the two |
| Blackhole `p150b` | Blackhole | passively cooled; ~5 % slower per token on identical work |

The n300 board reports two Wormhole B0 chips — `n300 L` and `n300 R` — and the model
runs on the local one, selected with `TT_VISIBLE_DEVICES=0`. Nothing in this port is
multi-chip: there are no collectives, no fabric traffic and no mesh device; the
tensor-parallel prototype that would have used the second chip was measured and not
shipped (see below).

---

## Stage 1 — bring-up

| requirement | status | validated by | evidence |
|---|---|---|---|
| CosyVoice-300M implemented with TTNN APIs | ✅ | `tests/pcc/` — every module against a captured PyTorch golden | `PERF.md` *Accuracy* |
| LLM backbone for semantic tokens | ✅ | `test_device_ar_prefill_and_decode`, `test_device_text_encoder` | *Accuracy* |
| Flow-based decoder | ✅ | `test_device_flow_tokens_to_mel`, `test_device_estimator_matches_golden`, `test_device_flow_encoder_matches_golden` | *Accuracy* |
| Vocoder | ✅ | `test_device_hift_decode_matches_golden`, `test_device_istft_matches_golden` | *Accuracy* |
| Runs on the named hardware with no errors | ✅ n300 | full `tests/pcc/` + `tests/e2e/` on all three boards | *The certification run* |
| SFT mode | ✅ | `test_modes_differ_only_in_prompt_construction`; `demo/demo.py --mode sft` | *Generation modes* |
| Zero-shot mode | ✅ | same | *Generation modes* |
| Cross-lingual mode | ✅ | same | *Generation modes* |
| Instruct mode | ✅ | same | *Generation modes* |
| Valid audio, 5 languages | ✅ 20/20 | `demo/sweep.py` — all four modes × zh/en/ja/ko/yue | *Speech quality* |
| Verifiable against the PyTorch reference | ✅ | `tests/pcc/` PCC checks; `test_device_tokens_to_waveform` end to end | *Accuracy* |
| `>= 30 tok/s` semantic generation | ✅ | checked — see the table above | *Semantic-token throughput* |
| `RTF < 0.5` | ✅ Blackhole · ❌ n300 | checked, with the n300 shortfall held to a recorded band | *End-to-end real-time factor* |
| Token accuracy `> 95 %` | ✅ | `test_gate1_teacher_forced_argmax_match`, `..._through_the_kv_cache`, `test_gate2_free_running_greedy` | *Accuracy* |
| WER `< 3.0`, speaker similarity `> 60` | ✅ | `scripts/eval_wer_sim.py`, reference venv | *Speech quality* |
| Setup and run instructions | ✅ | [`../README.md`](../README.md) | — |

## Stage 2 — basic optimizations

| requirement | status | validated by | evidence |
|---|---|---|---|
| Optimal sharded / interleaved memory configs | **measured; the default wins almost everywhere** | `scripts/probe_linear_grid.py`, `scripts/probe_ff2_shard.py` — no check, these are sweeps | `PERF.md` *Tuning flags*, *What limits the step* |
| Sharding: token embeddings | not sharded — the tensors are one row at decode | inspection | *What limits the step* |
| Sharding: transformer layers | see above; explicit grids lost in 10 of 12 combinations tried | `scripts/probe_linear_grid.py` | *Tuning flags* |
| Sharding: multi-head attention | superseded by the fused kernel — `sdpa_decode` owns its own parallelisation | `test_device_fused_attention_matches_explicit` | *Fused decode attention* |
| Sharding: flow decoder layers | superseded by fused `sdpa` in the estimator | `test_device_estimator_matches_golden` | *Flash attention* |
| Fuse simple ops | ✅ | `test_device_rel_pos_attention_matches_golden`, `test_device_ar_prefill_and_decode` | *Fused decode attention* |
| Store activations in L1 where beneficial | partial — `l1_small_size` tuned for conv weights; activations left interleaved | inspection | *Operational notes* |
| Recommended TTNN LLM flows | ✅ fixed-width KV cache, trace capture, program-cache-friendly shapes | `test_device_fixed_shape_cache_matches_the_growing_one`, `test_device_traced_matches_untraced` | *Fixed-width KV cache* |
| Efficient KV-cache management | ✅ | `test_device_inplace_matches_untraced`, `test_device_inplace_throughput` | *KV-cache layout* |
| Optimize the flow decoder | ✅ | `test_device_flow_tokens_to_mel`; timing in the RTF breakdown | *The flow decoder* |
| Optimize vocoder integration | ✅ | `test_hift_trace_is_bit_identical`, `test_hift_trace_is_faster` | *The vocoder* |

## Stage 3 — deeper optimization

| requirement | status | validated by | evidence |
|---|---|---|---|
| Maximize core counts | **measured; TTNN's default wins on most ops** — one exception shipped as a flag, and it is a *smaller* grid | `scripts/probe_linear_grid.py` | `PERF.md` *Tuning flags* |
| Efficient KV-cache for long sequences | ✅ | `test_device_fixed_shape_cache_matches_the_growing_one` | *Fixed-width KV cache* |
| Flash attention or equivalent | ✅ both stages | `test_device_fused_attention_matches_explicit` | *Fused decode attention*, *Flash attention* |
| Minimize token generation latency | ✅ | `test_device_traced_throughput`, `test_device_inplace_throughput` | *The LLM decode step* |
| **Batch processing for multiple utterances** | ✅ decode and end-to-end synthesis, both checked | `test_device_batched_decode_matches_single` (correctness, ragged prefixes), `test_device_batched_decode_throughput` (the sweep, checked), `test_device_batched_synthesis_agrees_with_one_at_a_time` (end to end, since 2026-09-23) | *Batched decode* |
| Efficient sampling strategies | ✅ top-k / top-p / RAS, host-side **by measurement** | `test_nucleus_filter_*`, `test_ras_*`, `scripts/profile_token_tail.py` | *The LLM decode step* |
| **Pipeline semantic generation with acoustic modeling** | ✅ | `test_device_streaming_first_audio_latency` (both schedules, all three stages real; Blackhole), `test_device_streaming_generates_the_same_tokens_as_batch` (the shipped API, all three boards) | *Streaming* |
| Optimize flow decoder computation | ✅ | `test_device_solve_euler_matches_golden`; trace-cache timing | *The flow decoder* |
| Minimize memory and TM overheads | ✅ `permute` removed from the decode step | `scripts/count_decode_ops.py` | *Removing token-independent recomputation* |
| Speculative decoding | ❌ **not explored** — see below | — | — |
| Multi-chip / tensor parallelism | **measured, not shipped** — see below | `scripts/probe_tp_decode.py` (a scratch probe, not in this tree) | *Known limitations* |
| Document tuning, limitations, trade-offs | ✅ | this document, `PERF.md` *Tuning flags* and *Known limitations* | — |
| `60+ tok/s` | ✅ | checked | *Semantic-token throughput* |
| `RTF < 0.2` | ❌ floored, not merely unmet — see below | checked against a recorded band | *End-to-end real-time factor* |
| Streaming inference | ✅ content, schedule and audio | `test_device_streamed_matches_non_streamed` (content), `test_device_streaming_first_audio_latency` (schedule), `test_device_streaming_generates_the_same_tokens_as_batch` (interleaved audio, since 2026-09-22) | *Streaming* |
| Efficient multi-lingual switching | ✅ 5 languages × 4 modes | `demo/sweep.py` | *Speech quality* |

---

## What is not met, what was fixed, and why

Three requirements are unmet (`RTF < 0.2`, `RTF < 0.5` on n300, speculative decoding)
and three device defects remain open (the Wormhole `test_streaming_perf` hang, the n300
amplitude difference, and the vocoder's per-geometry L1_SMALL growth). Two entries that
were in this list are now fixed -- the interleaved schedule's corrupt audio and
end-to-end batched synthesis -- and their accounts are kept, because in both cases the
remedies that failed are the instructive half.

### `RTF < 0.2` — reached the floor of this decomposition

Not a tuning shortfall. The flow decoder alone consumes a large fraction of the
`0.2` budget after a fused SDPA and a trace cache, and its cost is 64 transformer
blocks × 10 Euler steps — the Euler count is a model parameter, and lowering it
costs accuracy (`PERF.md` records what 5 steps buys and what it costs). The LLM's
share needs the decode step under 1.5 ms on its own, against a best measured step that
is bandwidth-limited on the AR decoder's weights. Both figures are in `PERF.md`
*End-to-end real-time factor*; the threshold is enforced against a recorded band so a
future improvement cannot pass unnoticed.

### `RTF < 0.5` on Wormhole n300

Met on both Blackhole boards, not met on n300, and n300 is a named target — so this
is reported on its own rather than folded into a Blackhole result. The gap is the
compute grid:
8 × 8 = 64 cores against Blackhole's 13 × 10 = 130, on a decode step whose cost is
dominated by weight traffic. `COSYVOICE_FF2_GRID=8x2` closes part of it. The lever and
the measured band are in `PERF.md` and in `tests/perf/gates.py`'s `WORMHOLE` table.

### Speculative decoding — not explored, and the reason is structural

Speculative decoding wins when a small draft model agrees with a large target model
often enough that verifying `k` drafted tokens in one target pass beats `k` sequential
passes. Two properties of this model make that a poor fit, and neither is a matter of
effort:

* There is no draft model. CosyVoice-300M ships one LLM; a draft would have to be
  trained or distilled, which is model work rather than a bring-up optimisation.
* Sampling is not greedy. The reference decodes with RAS — nucleus sampling plus a
  repetition-aware resample over the emitted history. Speculative decoding's
  acceptance test is defined for a fixed conditional distribution; RAS's rejection
  branch rewrites a score *based on tokens already emitted*, so the target
  distribution at step `i` depends on the accepted prefix in a way the draft cannot
  anticipate. Making the two agree would mean changing the sampler, which changes the
  audio.

The lever that *was* available at the same place in the pipeline — reducing the
per-token cost rather than the number of sequential steps — was taken instead: trace
capture, the fused decode attention, the fixed-width and in-place KV caches, and now
batching. `PERF.md` *The LLM decode step* carries what each was worth.

### End-to-end batched synthesis — fixed 2026-09-23

Closed. `test_device_batched_synthesis_agrees_with_one_at_a_time` ran for the life of
this PR as a skip, on the grounds that `synthesize_batch` wedged the board on the second
utterance and the cause was not established. It now passes in 18 s, two utterances at
100 % token agreement against the same two run alone.

`TtTransformerLM.generate_batch` was never the problem and was always verified: batched
rows match single-row decode at ragged prompt lengths, and the `B = 1..8` sweep fails if
batching amortises nothing. That is where the win is — the LLM runs once per *token* and
is the large majority of an utterance, while the flow decoder and the vocoder run once
per utterance each.

What blocked it was two traces alive at once. `COSYVOICE_CFM_TRACE_CACHE` keeps the flow
decoder's estimator trace across utterances, which is what makes a second utterance of
the same mel length cheap. `synthesize_batch` then captures a *decode* trace in
`generate_batch` and runs the flow decoder once per utterance, so a cached estimator
trace is live while another trace is captured — and TTNN is explicit: *"Allocating
device buffers is unsafe due to the existence of an active trace."* The last line before
each hang is that warning.

Four configurations on `p150a`, and the scope is the part worth recording:

| configuration | result |
|---|---|
| cache on (the shipped default) | hangs; 40-minute timeout, twice |
| cached trace released at entry to `synthesize_batch` | hangs; 40-minute timeout |
| cache disabled for the duration of `synthesize_batch` | hangs; 40-minute timeout |
| cache never captured in this process (`COSYVOICE_CFM_TRACE_CACHE=0`) | **passes, 18 s, 100 % agreement** |

So a *released* trace still makes a later capture unsafe. That is an upstream TTNN
property rather than something this port can fix from the outside, and it is why the
test sets the variable before the pipeline is constructed — `TtConditionalCFM` reads it
once, in its constructor — rather than toggling it around the call.

The cost is one estimator capture per utterance instead of one per distinct mel length,
paid only by `synthesize_batch`. Single-utterance `synthesize` keeps the cache and its
figures in `PERF.md` are unchanged.

### The L1_SMALL growth across vocoder geometries — measured, still open

Separate from the above, and the reason this section used to blame it. The vocoder parks
prepared `conv_transpose2d` weights in L1_SMALL per distinct mel geometry and never frees
them. Measured on `p150a` by synthesising one prompt at a sweep of token budgets on one
open device:

| geometries seen | L1_SMALL allocated | per-geometry cost |
|---|---:|---:|
| 1 (96 tokens) | `16 896 B` | — |
| 2 (128) | `31 680 B` | `+14 784` |
| 3 (160) | `48 640 B` | `+16 960` |
| 4 (192) | `65 280 B` | `+16 640` |
| 7 (288) | `123 520 B` | `+20 160` |

Revisiting a geometry already seen costs nothing — five calls alternating two geometries
stay flat at `32 064 B`. So the growth is per distinct geometry, not per call, and it
scales with mel length.

At the `l1_small_size = 131072` the e2e tests ask for, that admits about three geometries
before the allocator's top clashes with `conv_transpose2d`'s static circular-buffer
region:

```
RuntimeError: Statically allocated circular buffers in program 2455 clash with L1
buffers on core range [0-0 - 7-9]. L1 buffer allocated at 1384576 and static circular
buffer region ends at 1395648
```

That is a clean exception rather than a hang, which is how it is distinguishable from
the trace defect above. `test_device_batched_synthesis_agrees_with_one_at_a_time` asks
for `524288` for headroom. Freeing the per-geometry state is upstream work and is not
done here; `scripts/probe_l1_growth.py` reproduces the sweep.

### The interleaved schedule's corrupt audio — fixed 2026-09-22

Closed. `CosyVoiceTTNN.synthesize_streaming` used to return audio peaking around 72
against a batch path peaking at 0.001 on the same prompt — identical tokens, correct
chunk schedule, destroyed waveform. Measured on `p150a` one commit apart, with the
whole `tests/e2e/` suite and `tests/perf/test_streaming_perf.py` in the same two runs:

| | streamed peak | batch peak | `test_streaming_perf` |
|---|---:|---:|---|
| before | `72.5000` | `0.0003` | passes, first-audio gain `1.19×` |
| after | `0.0006` | `0.0005` | passes, first-audio gain `1.19×` |

The remedy is in `TtStreamingSynthesizer._carry_store`. The four tensors a stream
carries across a seam now live in persistent buffers allocated before any trace is
captured, and are written thereafter only by `ttnn.copy` — so neither an allocation nor
a readback crosses a live trace. The first chunk of the warm-up pass adopts each
buffer, which is what sizes and types them, and is why both callers run that warm-up
before capturing a trace.

`tests/e2e/test_pipeline_api.py` no longer pins a defect band; it asserts `peak < 1.5`
and that the streamed peak stays in proportion to the batch path's. Streaming content
equivalence is unchanged at mel-space PCC `0.901830`, the same figure `PERF.md` already
records. The full suite at that commit: 20 passed, 1 skipped — the skip is end-to-end
batched synthesis, which is a different defect and still open.

The account below is kept because it is what made the fix findable, and because the
remedy that did *not* work is the more useful half of it.

The cause was established first:

* `generate(use_trace=False)` makes it correct. Same conditioning, same schedule,
  same tokens; the only variable is whether a decode trace exists. So it is not the
  per-chunk conditioning, which an earlier revision of this document wrongly blamed.
* Per chunk against a no-trace reference: chunk 0, vocoded mid-generation with the
  trace live, matches at mel PCC `0.99999988` and waveform PCC `0.99999994`. Chunk 1,
  the finalize, has a bit-identical mel at PCC `1.000000000` and waveform PCC
  `0.011`.

Identical mel with destroyed audio rules out the flow decoder and the vocoder's
arithmetic and leaves what `StreamState` carries across a seam — `mel_overlap`,
`hift_mel`, `hift_source`, `hift_speech` — allocated during chunk 0 while the trace was
live and clobbered by a later `execute_trace`. TTNN warns about exactly this: *"These
buffers may be corrupted once a trace is executed."*

The first remedy tried was to park those four on the host between chunks (`to_torch`
out, `from_torch` back). It fixes the audio — verified on `p150a` and n300 — and it is
*not* what shipped, because it wedges `tests/perf/test_streaming_perf.py` on Blackhole,
where that test otherwise passes in 12.7 s. Draining the queue before the readback and
hoisting the synthesizer out of the traced region were also tried, and changed nothing.

Why it wedged is the part worth keeping, because it is what pointed at the fix that
worked. Parking trades one half of TTNN's warning for the other: instead of *carrying*
a device buffer across a live trace, it *allocates* one (`from_torch`) and *reads one
back* (`to_torch`) inside the window where a trace is live, at every seam. The readback
is the expensive half — it forces a device-to-host transfer at a point where the trace
owns the queue.

What ships avoids both. The carried tensors live in buffers allocated *before* any
trace is captured, and every later chunk writes into them with a device-to-device
`ttnn.copy`: no allocation, no readback, nothing new asked of the allocator at a moment
when the trace already owns addresses. That also explains why the ordering constraint
is real rather than superstition — a buffer first allocated *after* capture can sit on
an address the trace has baked in, which is the original defect wearing a different
hat. Both callers therefore build their synthesizer before capturing a trace, and
`tests/perf/test_streaming_perf.py` was changed to reuse one rather than construct a
second after capture.

### `test_streaming_perf` hangs on Wormhole — open

`tests/perf/test_streaming_perf.py::test_device_streaming_first_audio_latency` wedges
n300: log frozen, JIT cache flat, CPU pegged, board needing a reset. Both Blackhole
boards run it. It is skipped on Wormhole with that reason attached rather than left to
hang, because a wedged board costs every later test in the run.

The cause is not established, and one candidate has been eliminated.

An earlier revision of this document named an upstream TTNN defect: re-seeding a
trace's persistent buffers after that trace had executed. That is withdrawn. The probe
it rested on captured its trace before the first `prefill()` had ever run, so the
prefill compiled its kernels under a live trace — a property of the probe, not of the
path it was standing in for. Adding a warm-up before capture removes the hang on both
architectures:

| sequence, one variable apart | Wormhole n300 | Blackhole p150a |
|---|---|---|
| capture, then first prefill | hangs at the second seed | hangs at `close_device` |
| one prefill, then capture | clean, teardown included | clean, teardown included |

Four passes of seed plus 164 traced steps, warmed, complete in 14.7 s on n300 and
8.7 s on p150a. So the decode-only sequence is ruled out, along with the re-seed and
the trace's lifetime on their own.

What remains is the work this test runs *under* the live trace and
`synthesize_streaming` does not: the flow decoder and the vocoder, repeatedly, across
four passes. That is where to look next, and it is a narrowing rather than a diagnosis.

Ruled out along the way: the trace region size (384 MB → 64 MB changed nothing — it
captures one trace, not the in-place path's 65); and the `StreamState` fix above.

Two different warm-ups are in play here and they are worth keeping apart. The one this
test already performs warms the flow decoder and the vocoder before the AR trace is
captured. The one that mattered for the probe above warms the AR decoder's own prefill.
This test does not do that second one — its prefill still compiles under a live trace,
after capture — which makes it the cheapest thing to try next.

**The warm-before-capture constraint no longer holds on Blackhole, as of the carry-buffer
fix.** That constraint is what made this lead hard to test without a Wormhole: reversing
the order — capture the decode trace first, then drive the flow decoder and the vocoder
*through* it — used to hang Blackhole outright, log frozen and JIT cache flat. It is the
same mechanism the Wormhole hang is now narrowed to, and it is reproducible on hardware
that is available.

`scripts/probe_warm_order.py --order reversed` forces exactly that ordering. Measured on
`p150a` at the commit that fixed the streaming carry buffers:

| JIT cache | result |
|---|---|
| warm | survives; capture at `0.2 s`, warm-up through the traced path at `3.8 s`, interleaved pass complete at `6.2 s` |
| **cleared** | survives; capture at `10.7 s`, warm-up at `258.8 s`, interleaved pass complete at `273.8 s` |

The cleared-cache row is the one that matters, because the recorded hang was reproducible
"with a cleared cache on a freshly reset board" — so in that run `248 s` of kernels
genuinely compiled while the trace was live, which is the condition the constraint
existed for.

What that does and does not say. It says the Blackhole-reproducible instance of
"allocating under a live trace wedges the board" is fixed by holding the carried buffers
in allocations made before capture. It does **not** say the n300 hang is fixed: n300 was
unavailable throughout this work and the claim has not been tested there. What it changes
is the priority — the next person with an n300 should run
`tests/perf/test_streaming_perf.py` unskipped before investigating further, because the
mechanism it was blocked on now survives the equivalent test on the other architecture.

### An n300/Blackhole amplitude difference on a synthetic case — open

Surfaced by the probe above and not yet explained. On a greedy, 160-token-capped
synthesis of one prompt, Blackhole gives batch and streaming peaks that match
(`0.001` each) while n300 gives batch `0.001` and streaming `0.660`, identically with
and without a trace.

Which number is wrong is not established. `0.001` is near-silence and `0.660` is a
plausible speech peak, so the batch path may be the degenerate one on a capped greedy
run rather than streaming being broken. Ruled out: the live trace, and the known
Wormhole `ttnn.conv1d` prepared-weight defect (`COSYVOICE_CONV_PREPARE=0` gives the
same figure). The content-comparison test,
`test_device_streamed_matches_non_streamed`, passes on n300 at mel-space PCC
`0.902` — that uses the golden's own prompt and full token list rather than this case.

### Multi-chip tensor parallelism — measured, and it does not compound

A two-chip Megatron-sharded decoder was prototyped and measured on an n300 pair. It
works and it is not enough on its own; more importantly it collides with
`COSYVOICE_FF2_GRID` rather than compounding: tensor parallelism halves the FFN's
second linear to `K = 2048`, and the core-grid win that is large at `K = 4096` nearly
vanishes there. Same lever, different granularity, already mostly spent once TP has
sharded. Not shipped, and the measurement is why. `PERF.md` *Known limitations*.

---

## Reproducing the whole thing

```bash
# host tier, no device, ~90 s
pytest models/demos/cosyvoice/tests/ -k "not device"

# device tier: correctness
pytest models/demos/cosyvoice/tests/pcc/ models/demos/cosyvoice/tests/e2e/ -v

# device tier: the checked performance suite
pytest models/demos/cosyvoice/tests/perf/ -v -s

# the two tuning flags, each a full perf pass
COSYVOICE_FF2_GRID=8x2 pytest models/demos/cosyvoice/tests/perf/ -v -s
COSYVOICE_KV_INPLACE=1 pytest models/demos/cosyvoice/tests/perf/ -v -s
```

Weights and goldens have to exist first; [`../README.md`](../README.md) has the
export and capture steps. The perf suite skips itself with a stated reason rather than
failing when they do not.
