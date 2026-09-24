# Validation evidence, requirement by requirement

Every requirement in the bring-up scope, the test that decides it, and where its figure
lives. Measured performance and accuracy figures are in [`../PERF.md`](../PERF.md) and
are linked from here rather than repeated. This document is also the one place for what
is unmet, what is still open, and the workarounds the tree carries.

Read the columns as:

* validated by — the test that decides it. A name in `tests/` is executable: run it and
  it passes or fails. Where a requirement is validated by inspection or by an unchecked
  measurement, the cell says so.
* evidence — the section of `PERF.md` carrying the number, or the artefact.

## How the numeric thresholds are enforced

The perf suite enforces every numeric threshold through
[`../tests/perf/gates.py`](../tests/perf/gates.py):

* `GATES` — the thresholds, quoted from the scope and declared once for every test that
  produces one of these numbers.
* `EXPECTATIONS` — the per-architecture verdict. A threshold recorded as met is asserted
  against the requirement, so a regression fails. One recorded as unmet is asserted
  against its recorded measurement in both directions, as
  `models/perf/device_perf_utils.check_device_perf` does: slower fails as a regression,
  and faster fails because the published figure is stale.

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

`WER < 3.0` and `speaker similarity > 60` are not enforced by a test in this tree:
scoring needs whisper `large-v3` and `WavLMForXVector` from the reference venv (see
[`security.md`](security.md)), which tt-metal's `python_env` does not carry.
`scripts/eval_wer_sim.py` produces them and `PERF.md` *Speech quality* reports them.

## The device matrix

Three boards, one commit, one day; `PERF.md` *The certification run* has the commit, the
date and the per-board environment. A board with no result has an empty cell rather
than one filled from the other board of the same architecture.

| board | architecture | in scope because |
|---|---|---|
| Wormhole n300 | Wormhole | the named Wormhole target |
| Blackhole `p150a` | Blackhole | actively cooled, the faster of the two |
| Blackhole `p150b` | Blackhole | passively cooled; ~5 % slower per token on identical work |

The n300 board reports two Wormhole B0 chips, `n300 L` and `n300 R`; the model runs on
the local one, selected with `TT_VISIBLE_DEVICES=0`. Nothing in this port is
multi-chip: no collectives, no fabric traffic, no mesh device.

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
| Valid audio, 5 languages | ✅ zero-shot and cross-lingual 5/5 · SFT and instruct not re-run on this tree | `demo/sweep.py` — all four modes × zh/en/ja/ko/yue | *Speech quality* |
| Verifiable against the PyTorch reference | ✅ | `tests/pcc/` PCC checks; `test_device_tokens_to_waveform` end to end | *Accuracy* |
| `>= 30 tok/s` semantic generation | ✅ | checked — see the table above | *Semantic-token throughput* |
| `RTF < 0.5` | ✅ Blackhole · ❌ n300 | checked, with the n300 shortfall held to a recorded band | *End-to-end real-time factor* |
| Token accuracy `> 95 %` | ✅ | `test_gate1_teacher_forced_argmax_match`, `..._through_the_kv_cache`, `test_gate2_free_running_greedy` | *Accuracy* |
| WER `< 3.0`, speaker similarity `> 60` | ✅ English WER, both measured modes | `scripts/eval_wer_sim.py`, reference venv; `test_scoring.py` checks its English normaliser | *Speech quality* |
| Setup and run instructions | ✅ | [`../README.md`](../README.md) | — |

## Stage 2 — basic optimizations

| requirement | status | validated by | evidence |
|---|---|---|---|
| Optimal sharded / interleaved memory configs | **measured; the default wins almost everywhere** | `probe_linear_grid.py`, `probe_ff2_shard.py` (scratch probes, not in this tree) — no check, these are sweeps | `PERF.md` *Tuning flags*, *What limits the step* |
| Sharding: token embeddings | not sharded — the tensors are one row at decode | inspection | *What limits the step* |
| Sharding: transformer layers | see above; explicit grids lost in 10 of 12 combinations tried | `probe_linear_grid.py` (a scratch probe, not in this tree) | *Tuning flags* |
| Sharding: multi-head attention | superseded by the fused kernel — `sdpa_decode` owns its own parallelisation | `test_device_fused_attention_matches_explicit` | *Fused decode attention* |
| Sharding: flow decoder layers | superseded by fused `sdpa` in the estimator | `test_device_estimator_matches_golden` | *Flash attention* |
| Fuse simple ops | ✅ | `test_device_rel_pos_attention_matches_golden`, `test_device_ar_prefill_and_decode` | *Fused decode attention* |
| Store activations in L1 where beneficial | partial — `l1_small_size` tuned for conv weights; activations left interleaved | inspection | *L1_SMALL grows with each distinct vocoder geometry*, below |
| Recommended TTNN LLM flows | ✅ fixed-width KV cache, trace capture, program-cache-friendly shapes | `test_device_fixed_shape_cache_matches_the_growing_one`, `test_device_traced_matches_untraced` | *Fixed-width KV cache* |
| Efficient KV-cache management | ✅ | `test_device_inplace_matches_untraced`, `test_device_inplace_throughput` | *KV-cache layout* |
| Optimize the flow decoder | ✅ | `test_device_flow_tokens_to_mel`; timing in the RTF breakdown | *The flow decoder* |
| Optimize vocoder integration | ✅ | `test_hift_trace_is_bit_identical`, `test_hift_trace_is_faster` | *The vocoder* |

## Stage 3 — deeper optimization

| requirement | status | validated by | evidence |
|---|---|---|---|
| Maximize core counts | **measured; TTNN's default wins on most ops** — one exception shipped as a flag, and it is a *smaller* grid | `probe_linear_grid.py` (a scratch probe, not in this tree) | `PERF.md` *Tuning flags* |
| Efficient KV-cache for long sequences | ✅ | `test_device_fixed_shape_cache_matches_the_growing_one` | *Fixed-width KV cache* |
| Flash attention or equivalent | ✅ both stages | `test_device_fused_attention_matches_explicit` | *Fused decode attention*, *Flash attention* |
| Minimize token generation latency | ✅ | `test_device_traced_throughput`, `test_device_inplace_throughput` | *The LLM decode step* |
| **Batch processing for multiple utterances** | ✅ decode and end-to-end synthesis, both checked | `test_device_batched_decode_matches_single` (correctness, ragged prefixes), `test_device_batched_decode_throughput` (the sweep, checked), `test_device_batched_synthesis_agrees_with_one_at_a_time` (end to end) | *Batched decode* |
| Efficient sampling strategies | ✅ top-k / top-p / RAS, host-side **by measurement** | `test_nucleus_filter_*`, `test_ras_*`, `scripts/profile_token_tail.py` | *The LLM decode step* |
| **Pipeline semantic generation with acoustic modeling** | ✅ | `test_device_streaming_first_audio_latency` (both schedules, all three stages real; Blackhole), `test_device_streaming_generates_the_same_tokens_as_batch` (the shipped API, all three boards) | *Streaming* |
| Optimize flow decoder computation | ✅ | `test_device_solve_euler_matches_golden`; trace-cache timing | *The flow decoder* |
| Minimize memory and TM overheads | ✅ `permute` removed from the decode step | `scripts/count_decode_ops.py` | *Removing token-independent recomputation* |
| Speculative decoding | ❌ **not explored** — see below | — | — |
| Multi-chip / tensor parallelism | **measured, not shipped** — see below | `scripts/probe_tp_decode.py` (a scratch probe, not in this tree) | *Measured and not shipped* |
| Document tuning, limitations, trade-offs | ✅ | this document, `PERF.md` *Tuning flags* | — |
| `60+ tok/s` | ✅ | checked | *Semantic-token throughput* |
| `RTF < 0.2` | ❌ floored, not merely unmet — see below | checked against a recorded band | *End-to-end real-time factor* |
| Streaming inference | ✅ content and schedule on Blackhole · ❌ content on Wormhole at `e0de3009` · the interleaved stream's audio is wrong at its chunk geometry (*Streaming content on Wormhole* and *Streamed audio at the stream's chunk geometry*, under Open defects) | `test_device_streamed_matches_non_streamed` (content, 120-token chunks), `test_device_streaming_first_audio_latency` (schedule), `test_device_streaming_generates_the_same_tokens_as_batch` (interleaved tokens and peak) | *Streaming* |
| Efficient multi-lingual switching | ✅ zero-shot and cross-lingual, 5 languages per sweep · SFT and instruct not re-run on this tree | `demo/sweep.py` | *Speech quality* |

---

## Unmet requirements

### `RTF < 0.2` — the floor of this decomposition

Not a tuning shortfall. The flow decoder alone takes a large share of the `0.2` budget
with fused SDPA and a replayed trace, and its cost is 64 transformer blocks × 10 Euler
steps; the Euler count is a model parameter, and lowering it costs accuracy. The LLM's
share would need the decode step under 1.5 ms. `PERF.md` *End-to-end real-time factor*
has the figures. The threshold is asserted against a recorded band, so an improvement
fails the test until the published figure moves with it.

### `RTF < 0.5` on Wormhole n300

Met on both Blackhole boards and not on n300, a named target. The gap is the compute
grid: 8 × 8 = 64 cores against Blackhole's 13 × 10 = 130, on a decode step dominated by
weight traffic. `COSYVOICE_FF2_GRID=8x2` closes part of it. The lever and the band are
in `PERF.md` and in `tests/perf/gates.py`'s `WORMHOLE` table.

### Speculative decoding — not explored

Speculative decoding pays when a small draft model agrees with the target often enough
that verifying `k` drafted tokens in one target pass beats `k` sequential passes. Two
properties of this model rule it out:

* There is no draft model. CosyVoice-300M ships one LLM; a draft would have to be
  trained or distilled, which is model work rather than a bring-up optimisation.
* Sampling is not greedy. RAS — nucleus sampling plus a repetition-aware resample over
  the emitted history — rewrites a score based on tokens already emitted, so the target
  distribution at step `i` depends on the accepted prefix in a way a draft cannot
  anticipate. Making the two agree would change the sampler, and so the audio.

The per-token cost was reduced instead: trace capture, fused decode attention, the
fixed-width and in-place KV caches, and batching. `PERF.md` *The LLM decode step* has
what each is worth.

### Multi-chip tensor parallelism — measured, not shipped

A two-chip Megatron-sharded decoder on an n300 pair works, but does not compound with
`COSYVOICE_FF2_GRID`: tensor parallelism halves the FFN's second linear to `K = 2048`,
where the core-grid win that is large at `K = 4096` nearly vanishes. `PERF.md`
*Measured and not shipped* has the figures.

## Open defects

### `test_streaming_perf` hangs on Wormhole and can take a Blackhole CI host down

`tests/perf/test_streaming_perf.py::test_device_streaming_first_audio_latency` wedges
n300: log frozen, JIT cache flat, CPU pegged, board needing a reset. On Blackhole it
completes on a p150a and a p150b, but in the upstream CI on `bh_p150` two consecutive
runs of the perf suite lost the runner about 12 minutes in, and a run without this file
completed. So it runs only with `COSYVOICE_RUN_STREAMING_PERF=1`, on either architecture,
because a lost or wedged machine costs every later test in the run. `synthesize_streaming`
itself runs on n300 (what it produces there is under Streaming content on Wormhole, below);
this test's head-to-head timing has no Wormhole figure.

The cause is not established. Ruled out:

* The decode-only sequence. A trace captured before the first `prefill()` makes the
  prefill compile under the live trace, which hangs both architectures; one prefill
  before capture removes that hang:

  | sequence, one variable apart | Wormhole n300 | Blackhole p150a |
  |---|---|---|
  | capture, then first prefill | hangs at the second seed | hangs at `close_device` |
  | one prefill, then capture | clean, teardown included | clean, teardown included |

  Warmed that way, four passes of seed plus 164 traced steps complete in 14.7 s on n300
  and 8.7 s on p150a, so neither re-seeding a trace nor the trace's lifetime is the
  cause on its own.
* The trace region size: 384 MB → 64 MB changes nothing. The test captures one trace,
  not the in-place path's 65.
* The warm-before-capture ordering.
* Parking `StreamState` on the host between chunks. The persistent carry buffers that
  ship instead (`TtStreamingSynthesizer._carry_store`) are untested on n300.

What remains is the work this test runs under the live trace and `synthesize_streaming`
does not: the flow decoder and the vocoder, repeatedly, across four passes, with one
decode trace live throughout, where `synthesize_streaming` captures and releases per
call. The test also compiles its own prefill under the live trace, after capture;
warming that before capture is the cheapest thing to try next.

The reverse warm order — capture the decode trace first, then run the flow decoder and
the vocoder through it — completes on Blackhole:
`scripts/probe_warm_order.py --order reversed` finishes on `p150a` in 6.2 s with a warm
JIT cache and 273.8 s with a cleared one, 248 s of that compiling kernels under the live
trace. It is untested on n300, where the next step is to run `test_streaming_perf` with
`COSYVOICE_RUN_STREAMING_PERF=1`.

### Streaming content on Wormhole

At `e0de3009`, on tt-metal main `86d0bdb8ccbe` (2026-09-24), `tests/e2e/test_streaming.py::test_device_streamed_matches_non_streamed`
fails on n300: mel PCC `0.21` against the `0.85` gate, streamed RMS `0.633` against `0.049`
for the non-streamed run of the same tokens and seed
(https://github.com/tenstorrent/tt-metal/actions/runs/35976068834). At `a7c8416d3` the same
test passed on n300 with RMS `0.052`, and its log shows `_verify_prepared` rejecting the
prepared weight of `Conv1d(128->128, k=11)` at length 8321, inside the Wormhole `ttnn.conv1d`
defect range. `TtStreamingSynthesizer` now pauses that check for the length of a stream
(`TtHiFTGenerator.pause_weight_verification`), so on Wormhole the affected geometry runs its
prepared weight unchecked and the chunk comes out wrong; `0.21` is the figure §3.2 of PERF.md
records for exactly that case. Blackhole is unaffected at this utterance length. Not fixed:
the geometries a stream will use need their verification before the decode trace goes
live (the warm-up chunk is the place), or Wormhole streams need the op's own preparation
for the vocoder convs.

### A longer streamed utterance wedges the board

`test_device_streaming_first_audio_latency` measures one utterance length. Run at a
longer one — a wider trace region, and more and larger buffers live beside it — it
wedged `p150a` for 45 minutes at 100 % CPU with the JIT cache flat, twice, on two
boards, before the carry buffers existed; it has not been re-run since. So how first
audio scales with utterance length is not measured. A shared cause with the L1_SMALL
growth below is possible and unverified.

### Streamed audio at the stream's chunk geometry

At the interleaved stream's chunk geometry (conv input length 3457), two of the vocoder's
convolutions disagree with torch on Blackhole (`p150a`), measured inside a synthesis process:
`Conv1d(128 → 128, k=11)` with prepared weights scores PCC 0.14, and
`Conv1d(18 → 256, k=16, stride 8)` scores 0.003 whichever way its weights are prepared. In a
fresh process both are exact, and what makes them wrong in the model is not established. So the
audio `synthesize_streaming` emits is not right at that geometry. The interleaved streaming
tests check the tokens and the peak, which pass; `test_device_streamed_matches_non_streamed`
feeds 120-token chunks, a different geometry. A comment on
[tenstorrent/tt-metal#55545](https://github.com/tenstorrent/tt-metal/issues/55545) mentions both
convolutions as in-model observations; there is no standalone reproducer.

### L1_SMALL grows with each distinct vocoder geometry

The vocoder keeps prepared `conv_transpose2d` weights in L1_SMALL for each distinct mel
geometry and never frees them: 15-20 KB per geometry, growing with mel length.
Revisiting a geometry costs nothing. At `l1_small_size = 131072` about three geometries
fit before the allocator's top clashes with `conv_transpose2d`'s static circular-buffer
region, which raises rather than hangs:

```
RuntimeError: Statically allocated circular buffers in program 2455 clash with L1
buffers on core range [0-0 - 7-9]. L1 buffer allocated at 1384576 and static circular
buffer region ends at 1395648
```

`demo/demo.py` opens a fresh device per utterance for this reason, and
`test_device_batched_synthesis_agrees_with_one_at_a_time` asks for `524288`. Freeing the
per-geometry state is upstream work. `scripts/probe_l1_growth.py` measures the growth.

### An n300/Blackhole amplitude difference on a synthetic case

On a greedy, 160-token-capped synthesis of one prompt, Blackhole gives matching batch
and streaming peaks (`0.001` each), while n300 gives batch `0.001` and streaming
`0.660`, with and without a trace. Which figure is wrong is not established: `0.001` is
near-silence and `0.660` a plausible speech peak, so the batch path may be the
degenerate one on a capped greedy run. Ruled out: the live trace, and the Wormhole
`ttnn.conv1d` defect below (`COSYVOICE_CONV_PREPARE=0` gives the same figure).
`test_device_streamed_matches_non_streamed` passed on n300 while the stream still
verified prepared conv weights; at `e0de3009` it fails there (below). It uses the golden's
own prompt and full token list rather than this case.

### Dependency advisories — disposition requested

Four advisories against the reference venv's pins are open: three `torch` MEDIUM and one
`transformers` HIGH. None is reachable from the reference path, and the merge installs
none of that venv. [`security.md`](security.md) has the evidence and the disposition
being requested.

## Workarounds in the tree

### Buffers allocated while a trace is live

TTNN warns: *"Allocating device buffers is unsafe due to the existence of an active
trace. These buffers may be corrupted once a trace is executed."* The tree avoids it in
seven places:

* Streaming. The state `StreamState` carries across chunk seams lives in persistent
  buffers allocated before the AR decode trace is captured, written afterwards only
  with `ttnn.copy`. `synthesize_streaming` pushes one warm-up chunk before `generate`
  captures; any other caller must build and warm its `TtStreamingSynthesizer` before
  capture (`_carry_store`).
* The flow's trace between utterances. `tokens_to_mel` (so `synthesize`,
  `synthesize_batch` and `demo/demo.py --inputs`) releases the CFM estimator trace on
  both sides of its solve, `synthesize_streaming` once the stream ends, and
  `demo/sweep.py` after each utterance (`TtMaskedDiffWithXvec.release_trace`). Every
  utterance pays one estimator capture; a stream keeps its trace across its own chunks.
  Without the release, the next utterance of the same mel length replays the trace after
  the vocoder and the LLM have allocated device buffers while it was live (the LLM also
  captures and releases a trace of its own), and the device stalls at the first read of
  the waveform. Which buffer the replay corrupts is not established.
  `test_device_consecutive_utterances_with_one_flow_length` checks two utterances in a
  row, and `test_device_stream_leaves_no_flow_trace` a stream followed by `synthesize`.
* Batched synthesis. `test_device_batched_synthesis_agrees_with_one_at_a_time` sets
  `COSYVOICE_CFM_TRACE_CACHE=0` before the pipeline is built; `TtConditionalCFM` reads it
  once, in its constructor. With an estimator trace kept from an earlier utterance,
  `generate_batch`'s decode-trace capture hung the device after the warning above, and so
  did releasing that trace at entry to `synthesize_batch` or disabling the cache only for
  the call. With the releases above, no flow trace is live when `generate_batch`
  captures; whether that makes the variable unnecessary is untested.
* The vocoder's prepared-weight check. The vocoder checks each prepared conv weight once
  per `(input length, batch)` geometry, at that geometry's first call, against the op's own
  preparation, element by element; where they disagree it switches that geometry to the
  op's own preparation for good, and that path allocates on every call. So the check does
  not run with a trace live: a `StreamSession` pauses it from construction to close
  (`TtHiFTGenerator.pause_weight_verification`), because an interleaved stream calls the
  vocoder between replays of the LLM decode trace. Batch synthesis keeps it, since the LLM
  and CFM traces are released before the vocoder runs. A geometry already switched stays
  switched, inside a stream too.
* The CFM solver. Its traced body is a whole Euler step, so the replay loop allocates
  nothing between replays.
* Trace lifetime. `generate` releases its decode trace in a `finally`, and
  `TracedDecodeStep.capture` closes the capture on failure: a trace left open or live
  corrupts or hangs whatever runs next.
* `test_streaming_perf` captures one decode trace and re-seeds it for each pass, since
  four captures in one process hung the board, and asks for a 64 MB trace region, since
  384 MB hung n300.

### `ttnn.copy` into and out of the CFM trace

Two copies misbehave in the CFM solver's trace, with nothing raised. Refreshing a trace
input with `ttnn.copy` from the output of a dim-0 `ttnn.concat` writes wrong data (PCC
0.768, where the same copy from a plain device tensor is bit-exact;
`scripts/probe_cfm_trace.py`), so the input buffer holds one row and the CFG doubling
happens inside the traced body. Ending the traced body with a `ttnn.copy` into a buffer
allocated before capture has no effect on replay (the solver reads back zeros), so the
trace owns its output instead (`TtConditionalCFM._capture`).

### `ttnn.transformer.scaled_dot_product_attention` and tile padding

SDPA masks padded key columns itself, but its output is wrong (PCC ~0 against torch)
when the tile padding of both k and v holds large or non-finite values, at any sequence
length that is not a tile multiple. The flow estimator's convolutions leave such values
in their output padding, and when a UNet level's length is 1 mod 32 they reach k and v
as NaN: about half the mel comes out non-finite and the vocoder rails at its clamp. The
zero-shot Japanese case at 332 tokens (flow lengths 897 and 449) hits both levels.
`TtAttention` zero-fills the k/v tile padding with `ttnn.fill_implicit_tile_padding`
before SDPA whenever the length is not a tile multiple;
`test_device_attention_ignores_tile_padding` checks it at 449, 450 and 897, and PERF.md
Part II §2.1 has the cost. Reported upstream as
[tenstorrent/tt-metal#57608](https://github.com/tenstorrent/tt-metal/issues/57608).

### `ttnn.conv1d` with prepared weights

`ttnn.conv1d` with weights from `ttnn.prepare_conv_weights` disagrees with the op's own weight
preparation at some input lengths, on both architectures, and nothing is raised. On Wormhole,
lengths 8193–8704 are off by up to `1e37` for the vocoder's `Conv1d(128 → 128, k=11, pad=5)`. On
Blackhole (p150a), the vocoder's `Conv1d(256 → 256, k=7)` and `k=11` return `inf` at length 8264
with a 512 KB L1_SMALL reservation and are exact with 32 KB (other sizes are not measured); the
length regulator's `Conv1d(80 → 80, k=3, pad=1)` scores PCC 0.32–0.41 against torch at
T = 1717–1910 with either reservation. Cross-lingual flows run at those lengths, because the
flow length includes the 1289-frame prompt: four of five cross-lingual utterances came out as
the right voice saying nothing, and the fifth, at 1033 mel frames, as a railed waveform.

The vocoder therefore verifies each `(input_length, batch)` geometry once on both
architectures, running prepared and unprepared weights and keeping the prepared one only where
the two outputs agree element by element (`TtConv1d._verify_prepared`,
`prepare_weights_default`); `COSYVOICE_CONV_PREPARE=1` skips the check. The regulator never runs
inside a trace and takes the op's own preparation. The flow estimator's convolutions have to
stay prepared, since they run inside the CFM trace; its nine geometries matched torch at every
length from 97 to 2659 in steps of 61, at batch 2 with the 512 KB reservation, on p150a.
Reported upstream as
[tenstorrent/tt-metal#55545](https://github.com/tenstorrent/tt-metal/issues/55545): the Wormhole
case in the issue body, the Blackhole cases in a comment on it.
`scripts/repro_conv1d_wormhole.py` reproduces the Wormhole case without the model, and with
`IN_C = OUT_C` set to 80 or 256 the Blackhole ones (the comment has the arguments);
`scripts/probe_prepared_weights.py` is the check to run once it is fixed.

### `ttnn.cumsum` accuracy in fp32

On this branch's tt-metal, `ttnn.cumsum` accumulates fp32 sequentially, and over the
vocoder's 72 192-sample f0 scan the error is over half a cycle of phase. `phase_mod1`
(`tt/hifigan/source.py`) reduces each block mod 1 before accumulating and is used
instead; it is also faster than the plain scan, which runs on one core. Reported
upstream as [tenstorrent/tt-metal#55542](https://github.com/tenstorrent/tt-metal/issues/55542),
closed as completed on 2026-09-22, after this branch's tt-metal base; re-measuring
`ttnn.cumsum` on a newer base decides whether `phase_mod1` can go. `PERF.md` Part II
§3.1 has both paths' accuracy and speed.

### `ttnn::concat` segfaults on an empty tensor

A zero-length input crashes the process inside `ttnn::concat` instead of raising. `sft`
and `instruct` have no prompt audio, so `TtMaskedDiffWithXvec.regulate` and
`.conditions` skip the concat when the prompt is empty (`tt/flow/model.py`).

### `ttnn.group_norm` rejects the estimator's shapes

Native `ttnn.group_norm` rejects these shapes at `G = 8` on both architectures, so the
estimator computes GroupNorm as a matmul against a `[C, G]` indicator. That form
computes `var = E[x²] − E[x]²`, which bfloat16 rounding can push slightly negative;
`rsqrt` of it is an unraised `Inf` that reaches the vocoder as clipped noise, so the
variance is clamped with `ttnn.relu` before `eps`. `COSYVOICE_GN_PERMUTE=1` restores the
permute-based form.

### `sdpa_decode` with `k_chunk_size = 32`

`scaled_dot_product_attention_decode` accepts `k_chunk_size = 32` and returns wrong
attention at key widths under 512: PCC 0.29–0.70 against a torch golden at widths 256,
384 and 448, where 64 and up give 0.9999 and non-powers of two raise. Limiting
`max_cores_per_head_batch` to 1 or 2 makes 32 correct at width 384, so the fault is in
the multi-core split of the key axis when chunks are one tile deep.
`TtRelPosAttention._sdpa_program` uses the largest power of two dividing the key width,
capped at 128; `scripts/probe_sdpa_chunk_sweep.py` is the sweep.

### Ops composed from primitives

`ttnn.conv_transpose1d` does not exist: the vocoder's two upsamplers run as
`conv_transpose2d` at `H = 1` (`tt/hifigan/upsample.py`), and they dominate the
vocoder's time. There is no FFT; the iSTFT is a matmul, a `conv_transpose2d` and a
multiply (`README.md` has the derivation). Snake is composed from five elementwise ops in
`TtSnake`, although `ttnn.snake_beta(x, alpha, alpha)` computes the same activation
natively; the port does not use it yet, and `scripts/probe_snake_native.py` compares the
two.

### The simulator

`ttsim` supports neither `float32` nor `ttnn.cumsum`, and both abort the process rather
than raising, so `tests/pcc/test_source.py`'s SineGen tests skip when
`TT_METAL_SIMULATOR` is set.

---

## Reproducing the whole thing

```bash
# host tier, no device, ~90 s
pytest models/experimental/cosyvoice/tests/ -k "not device"

# device tier: correctness
pytest models/experimental/cosyvoice/tests/pcc/ models/experimental/cosyvoice/tests/e2e/ -v

# device tier: the checked performance suite
pytest models/experimental/cosyvoice/tests/perf/ -v -s

# the two tuning flags, each a full perf pass
COSYVOICE_FF2_GRID=8x2 pytest models/experimental/cosyvoice/tests/perf/ -v -s
COSYVOICE_KV_INPLACE=1 pytest models/experimental/cosyvoice/tests/perf/ -v -s
```

Weights and goldens have to exist first; [`../README.md`](../README.md) has the
export and capture steps. The perf suite skips itself with a stated reason rather than
failing when they do not.
