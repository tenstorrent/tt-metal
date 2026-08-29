# Validation evidence, requirement by requirement

Every requirement in the bring-up scope, what validates it, and where its number
lives. This document is a **map, not a second copy of the numbers**: measured figures
appear once, in [`../PERF.md`](../PERF.md), and are linked from here. Two documents
quoting the same measurement is how they drift apart.

Read the columns as:

* **validated by** — the test that decides it. A name in `tests/` is executable; run
  it and it passes or fails. Where a requirement is validated by inspection or by a
  measurement with no gate, the cell says so rather than naming a test that does not
  enforce it.
* **evidence** — the section of `PERF.md` carrying the number, or the artefact.

## How the numeric thresholds are enforced

Before 2026-08-29 the perf suite printed its figures and asserted `total_s > 0` — a
timing harness, not a gate. It now enforces every numeric threshold through
[`../tests/perf/gates.py`](../tests/perf/gates.py), which separates two things that
were previously conflated:

* **`GATES`** — the thresholds themselves, quoted from the scope. One declaration,
  used by every test that produces one of these numbers.
* **`EXPECTATIONS`** — the per-architecture verdict. A gate recorded as met is
  asserted against the threshold, so a regression fails. A gate recorded as **missed**
  is asserted against its recorded measurement, **both bounds**, exactly as
  `models/perf/device_perf_utils.check_device_perf` does: slower fails because that is
  a regression, and *faster* fails because it means the published figure is stale.

Nothing is `xfail`-ed. A missed target gets a measured number, a band it must stay
inside, and a named lever.

| threshold | scope stage | enforced in |
|---|---|---|
| `>= 30 tok/s` semantic generation | Stage 1 | `test_device_end_to_end_rtf`, `test_device_traced_throughput`, `test_device_inplace_throughput` |
| `RTF < 0.5` | Stage 1 | `test_device_end_to_end_rtf` |
| `>= 60 tok/s` | Stage 3 stretch | same three as the 30 tok/s gate |
| `RTF < 0.2` | Stage 3 stretch | `test_device_end_to_end_rtf` |
| token agreement `> 95 %` | Stage 1 | `test_gate1_teacher_forced_argmax_match`, `test_gate1b_teacher_forced_argmax_through_the_kv_cache` |
| per-module PCC `>= 0.99` | Stage 1 | every `tests/pcc/test_device_*` |
| streaming content equivalence | Stage 3 stretch | `test_device_streamed_matches_non_streamed` |
| batching amortises the weight read | Stage 3 | `test_device_batched_decode_throughput` |
| streaming starts before generation ends | Stage 3 | `test_device_streaming_first_audio_latency` |

`WER < 3.0` and `speaker similarity > 60` are **not** enforced by a test in this tree,
and that is a deliberate boundary rather than an omission: scoring needs whisper
`large-v3` and `WavLMForXVector` in the reference venv (see
[`security.md`](security.md)), which tt-metal's `python_env` does not carry and this
demo does not install. They are produced by `scripts/eval_wer_sim.py` and reported in
`PERF.md`'s *Speech quality*.

## The device matrix

Three boards, one commit, one day — see `PERF.md`'s *The certification run* for the
commit, the date and the per-board environment. Every figure in `PERF.md` comes from
that run; where a board is missing a cell, the cell is empty rather than filled from
the other board of the same architecture.

| board | architecture | in scope because |
|---|---|---|
| Wormhole n300 | Wormhole | the scope names N150 **or** N300; this is the named target |
| Blackhole `p150a` | Blackhole | actively cooled, the faster of the two |
| Blackhole `p150b` | Blackhole | passively cooled; ~5 % slower per token on identical work |

**N150 was not available, and what that does and does not leave open is worth being
precise about.** The n300 board reports two Wormhole B0 chips — `n300 L` and `n300 R` —
and the model runs on the local one, selected with `TT_VISIBLE_DEVICES=0`. **Nothing in
this port is multi-chip**: there are no collectives, no fabric traffic and no mesh
device; the tensor-parallel prototype that would have used the second chip was measured
and not shipped (see below). So the compute the model actually uses on the n300 is one
Wormhole B0 with an 8 × 8 grid — the same compute an N150 provides.

That makes an N150 result *predictable* rather than *measured*, and the distinction is
kept: nothing in `PERF.md` reports an N150 figure, and `tests/perf/gates.py`'s Wormhole
expectations are recorded from n300 and will trip their two-sided band rather than
silently absorb a different part. What would be expected to differ on an N150 is the
host interface and the board's power envelope, not the grid the kernels run on.

---

## Stage 1 — bring-up

| requirement | status | validated by | evidence |
|---|---|---|---|
| CosyVoice-300M implemented with TTNN APIs | ✅ | `tests/pcc/` — every module against a captured PyTorch golden | `PERF.md` *Accuracy* |
| LLM backbone for semantic tokens | ✅ | `test_device_ar_prefill_and_decode`, `test_device_text_encoder` | *Accuracy* |
| Flow-based decoder | ✅ | `test_device_flow_tokens_to_mel`, `test_device_estimator_matches_golden`, `test_device_flow_encoder_matches_golden` | *Accuracy* |
| Vocoder | ✅ | `test_device_hift_decode_matches_golden`, `test_device_istft_matches_golden` | *Accuracy* |
| Runs on N150 or N300 with no errors | ✅ n300 | full `tests/pcc/` + `tests/e2e/` on all three boards | *The certification run* |
| SFT mode | ✅ | `test_modes_differ_only_in_prompt_construction`; `demo/demo.py --mode sft` | *Generation modes* |
| Zero-shot mode | ✅ | same | *Generation modes* |
| Cross-lingual mode | ✅ | same | *Generation modes* |
| Instruct mode | ✅ | same | *Generation modes* |
| Valid audio, 5 languages | ✅ 20/20 | `demo/sweep.py` — all four modes × zh/en/ja/ko/yue | *Speech quality* |
| Verifiable against the PyTorch reference | ✅ | `tests/pcc/` PCC gates; `test_device_tokens_to_waveform` end to end | *Accuracy* |
| `>= 30 tok/s` semantic generation | ✅ | gated — see the table above | *Semantic-token throughput* |
| `RTF < 0.5` | ✅ Blackhole · ❌ n300 | gated, with the n300 miss held to a recorded band | *End-to-end real-time factor* |
| Token accuracy `> 95 %` | ✅ | `test_gate1_teacher_forced_argmax_match`, `..._through_the_kv_cache`, `test_gate2_free_running_greedy` | *Accuracy* |
| WER `< 3.0`, speaker similarity `> 60` | ✅ | `scripts/eval_wer_sim.py`, reference venv | *Speech quality* |
| Setup and run instructions | ✅ | [`../README.md`](../README.md) | — |

## Stage 2 — basic optimizations

| requirement | status | validated by | evidence |
|---|---|---|---|
| Optimal sharded / interleaved memory configs | **measured; the default wins almost everywhere** | `scripts/probe_linear_grid.py`, `scripts/probe_ff2_shard.py` — no gate, these are sweeps | `PERF.md` *Tuning flags*, *What the decode step is bound by* |
| Sharding: token embeddings | not sharded — the tensors are one row at decode | inspection | *What the decode step is bound by* |
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
| **Batch processing for multiple utterances** | ✅ | `test_device_batched_decode_matches_single` (correctness, ragged prefixes), `test_device_batched_decode_throughput` (the sweep, gated) | *Batched decode* |
| Efficient sampling strategies | ✅ top-k / top-p / RAS, host-side **by measurement** | `test_nucleus_filter_*`, `test_ras_*`, `scripts/profile_token_tail.py` | *The LLM decode step* |
| **Pipeline semantic generation with acoustic modeling** | ✅ | `test_device_streaming_first_audio_latency` (both schedules, all three stages real, two utterance lengths) | *Streaming* |
| Optimize flow decoder computation | ✅ | `test_device_solve_euler_matches_golden`; trace-cache timing | *The flow decoder* |
| Minimize memory and TM overheads | ✅ `permute` removed from the decode step | `scripts/count_decode_ops.py` | *Removing token-independent recomputation* |
| Speculative decoding | ❌ **not explored** — see below | — | — |
| Multi-chip / tensor parallelism | **measured, not shipped** — see below | `scripts/probe_tp_decode.py` (a scratch probe, not in this tree) | *Known limitations* |
| Document tuning, limitations, trade-offs | ✅ | this document, `PERF.md` *Tuning flags* and *Known limitations* | — |
| `60+ tok/s` | ✅ | gated | *Semantic-token throughput* |
| `RTF < 0.2` | ❌ **bounded below, not merely missed** — see below | gated against a recorded band | *End-to-end real-time factor* |
| Streaming inference | ✅ | `test_device_streamed_matches_non_streamed` (content), `test_device_streaming_first_audio_latency` (schedule) | *Streaming* |
| Efficient multi-lingual switching | ✅ 5 languages × 4 modes | `demo/sweep.py` | *Speech quality* |

---

## The three that are not met, and why

### `RTF < 0.2` — reached the floor of this decomposition

Not a tuning shortfall. The flow decoder alone consumes a large fraction of the
`0.2` budget after a fused SDPA and a trace cache, and its cost is 64 transformer
blocks × 10 Euler steps — the Euler count is a **model** parameter, and lowering it
costs accuracy (`PERF.md` records what 5 steps buys and what it costs). The LLM's
share needs the decode step under 1.5 ms on its own, against a best measured step that
is bandwidth-bound on the AR decoder's weights. Both figures are in `PERF.md`
*End-to-end real-time factor*; the gate is enforced against a recorded band so a
future improvement cannot pass unnoticed.

### `RTF < 0.5` on Wormhole n300

Met on both Blackhole boards, missed on n300, and n300 is a named target — so this is
reported as a miss rather than as a Blackhole result. The gap is the compute grid:
8 × 8 = 64 cores against Blackhole's 13 × 10 = 130, on a decode step whose cost is
dominated by weight traffic. `COSYVOICE_FF2_GRID=8x2` closes part of it. The lever and
the measured band are in `PERF.md` and in `tests/perf/gates.py`'s `WORMHOLE` table.

### Speculative decoding — not explored, and the reason is structural

Speculative decoding wins when a small draft model agrees with a large target model
often enough that verifying `k` drafted tokens in one target pass beats `k` sequential
passes. Two properties of this model make that a poor fit, and neither is a matter of
effort:

* **There is no draft model.** CosyVoice-300M ships one LLM; a draft would have to be
  trained or distilled, which is model work rather than a bring-up optimisation.
* **Sampling is not greedy.** The reference decodes with RAS — nucleus sampling plus a
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

### Multi-chip tensor parallelism — measured, and it does not compound

A two-chip Megatron-sharded decoder was prototyped and measured on an n300 pair. It
works and it is not enough on its own; more importantly it **collides** with
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

# device tier: the gated performance suite
pytest models/demos/cosyvoice/tests/perf/ -v -s

# the two tuning flags, each a full perf pass
COSYVOICE_FF2_GRID=8x2 pytest models/demos/cosyvoice/tests/perf/ -v -s
COSYVOICE_KV_INPLACE=1 pytest models/demos/cosyvoice/tests/perf/ -v -s
```

Weights and goldens have to exist first; [`../README.md`](../README.md) has the
export and capture steps. The perf suite skips itself with a stated reason rather than
failing when they do not.
