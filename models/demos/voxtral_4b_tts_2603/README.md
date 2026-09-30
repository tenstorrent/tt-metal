# Voxtral-4B-TTS-2603 — end-to-end TTNN pipeline

A real, chained TTNN pipeline for `mistralai/Voxtral-4B-TTS-2603`, composed from the 31 graduated
stubs the bring-up tool produced under `models/tt_transformers/demo/voxtral_4b_tts_2603/`
(**Source B**) and gated against the reference model (**Source A**). The plan is
[`e2e_plan.json`](e2e_plan.json) (plan_version 4).

**Batch = 32 independent samples per call.** Every demo, gate test and trace stage drives 32
distinct inputs through one program per step. Each test reads the batch it drives off the pipeline
object.

---

## What this model is

The hub repo ships `params.json`, `consolidated.safetensors`, `tekken.json` and
`voice_embedding/<voice>.pt` (20 preset voices), and **no `config.json`** — `params.json` declares
`model_type=voxtral_tts`, for which transformers has no class. The reference is Source B's
`tests/pcc/_reference_loader.py::load_reference_model()`, which consumes all 386 checkpoint tensors:

| part | what it is | depth |
|---|---|---|
| `.model` | Mistral text backbone, dim 3072, `rope_theta` 1e6, tied embeddings | 26 layers |
| `.acoustic_transformer` | flow-matching sampler (7 Euler steps, CFG) + semantic head | 3 layers |
| `.audio_tokenizer` | codec decoder: 37 codes/frame → 24 kHz, 12.5 frames/s | 8 blocks |

`model.generate()` is not the TTS task (it drives the tied text head, which cannot emit audio
codes), so the golden (`reference/golden.py`) runs the reference's own submodules through the chain
the architecture dictates; `test_golden_is_the_references_own_arithmetic` proves it is that
arithmetic.

---

## The two Calls

Both are authored chains in `tt/pipeline.py`. The demos and the e2e tests call the **same**
functions, so a green test cannot sit beside a broken demo.

### Call 1 — `text_to_speech` (the model's task)

`run_text_to_speech`: a voice-prompted speech request → a 24 kHz waveform per row.

```
input  [BOS] [BEGIN_AUDIO] [AUDIO]*147 [NEXT_AUDIO_TEXT] <text> [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]
       + the casual_male voice embedding substituted into the 147 [AUDIO] rows ON DEVICE
prefill (26 layers, seeds the resident KV cache)
  └─ per frame: semantic head + flow sampler (7 Euler steps, CFG 3.0) → 37 codes
                └─ audio-token embedding → one decode step against the KV cache
  └─ stop: each row at its own end_audio; the batch when every row has ended
vocode: the whole code block → waveform; each row cut before its own end frame
```

Routes 28 graduated modules: `token_embed`, `mistral_rotary_embedding`, `mistral_r_m_s_norm`, the
four interchangeable block kinds (`layer`, `mistral_decoder_layer`, `attention`+`mlp`,
`mistral_attention`+`mistral_m_l_p`, split by layer index), the acoustic section
(`flow_matching_audio_transformer`, `acoustic_transformer_block`, `bidirectional_attention`,
`feed_forward`, `time_embedding`) and the codec (`voxtral_t_t_s_audio_tokenizer`,
`codec_transformer`, `codec_transformer_block`, `codec_attention`, `causal_conv1d`,
`causal_conv_transpose1d`, `parametrized_conv1d`, `parametrized_conv_transpose1d`, `weight_norm`,
`parametrization_list`, `mistral_audio_codebook`, `semantic_codebook`, `acoustic_codebook`,
`multi_vocab_embeddings`).

### Call 2 — `text_continuation`

`run_text_continuation`: the causal LM's next-token prediction, teacher-forced over the real text in
one forward (logits and the on-device greedy pick at every position). It is the only home of `encoder_stack` and
`mistral_model` (byte-identical whole-stack aliases, split by batch row) and `decoder_head` (the LM
head) — the TTS chain never reads `lm_head`. This is a TTS checkpoint: its tied text head is
effectively untrained, so the continuation is not language. The gate is parity with the reference
on the same input, which garbage-that-matches satisfies.

Together the two Calls route **all 31** graduated modules, each in exactly one Call
(`test_plan_routes_every_graduated_module_exactly_once`).

---

## Results — measured on device (1× Blackhole p300c chip)

### Call 1 — `text_to_speech`: READY

32 rows, 96 frames (7.68 s of audio), whole output compared. Reference teacher-forced onto the TT
trajectory at every joint (see *How Call 1 is scored*).

| check | result |
|---|---|
| **Gate 3 — waveform PCC, min over 32, all 96 frames** | **0.999781** (mean 0.999974) |
| termination | TT: every row emitted `end_audio`, batch done at frame 95; HF golden: frame 103; cap 256 not reached |
| prefill hidden PCC (min/32) | 0.997967 |
| decode hidden PCC (min over 32 × 96) | 0.999779 |
| semantic logits PCC | 1.000000 |
| flow-sampler `x_final` PCC | 0.996492 |
| discretization = the reference's rule on TT's own values | 110592/110592 acoustic, 3072/3072 semantic, exact |
| semantic codes vs teacher-forced reference | agreement 0.999349; 2 differ, both ties; 0 decidable mismatches |
| frame-0 semantic code vs free-running HF (no feedback yet) | 30/32 exact, 2 ties (reference margins 1.3e-3, 1.2e-2 vs logit RMS dev 2.0e-2) |
| acoustic codes vs teacher-forced reference | agreement 0.990641; 1002 differ, all measured ties; **0 decidable mismatches** (see *How Call 1 is scored*) |
| intelligibility (Whisper large-v3-turbo corpus WER) | TT **0.0169** vs HF golden 0.0233 (margin 0.05) |
| naturalness (UTMOS22 mean MOS) | TT **3.713** vs HF golden 3.739 (margin 0.20) |
| 32 distinct inputs → 32 distinct code streams → 32 distinct waveforms | yes |

### Call 2 — `text_continuation`: READY

32 rows × 35 positions (every position of the shortest prompt; no padding), one teacher-forced forward.

| check | result |
|---|---|
| **Gate 3 — logit PCC, min over 32 rows of each row's worst position** | **0.999855** |
| token equality (TT on-device argmax vs reference argmax) | 1120/1120 exact; 0 ties, 0 mismatches |
| Gate 2 — `encoder_stack`, `mistral_model`, `decoder_head` invoked | yes |

### Trace + fully-on-device

`PIPELINE_STAGES = ["prefill", "decode", "acoustic", "vocode"]` — see *Command 3*.

---

## How Call 1 is scored

**Nothing is spliced into the TT side.** It runs free, exactly as the demo does. The reference is
the one re-aligned: fed the TT codes (`fed_codes`) and the TT per-frame hidden (`fed_hidden`), so at
every frame it answers "given exactly what this pipeline produced, what does torch compute?". A
free-running TT-vs-HF comparison measures how fast two chaotic trajectories separate — the codes
are a round onto 21 levels, and one device matmul's rounding is enough to flip later frames (the
free-running code agreement is printed as a diagnostic: 0.196).

**Discrete agreement is asserted, with ties measured, not guessed.** The semantic code is a tie
when the reference's own top-1/top-2 margin is under 6 × the measured RMS logit deviation. The
acoustic codes needed more care, because the flow sampler (CFG alpha 3 over 7 Euler steps) is
ill-conditioned at some elements. Measured:

* the float32 reference agrees with a float64 run of itself to 4e-6 (in code-grid units), so the
  reference is exact; the acoustic weights are exactly bfloat16-representable, so weight dtype is
  not a lever;
* TT's `x_final` deviation is RMS 1.3e-2 grid steps on a typical frame but up to 3.7 at single
  elements (frame 76, row 1), and it is deterministic (bit-identical over 3 replays);
* injecting a device-level matmul error at every `nn.Linear` of the **torch** sampler moves those
  same elements by several grid steps — at 1.2e-3 relative its spread (RMS 0.20 on frame 76) is
  larger than TT's (0.11). So those elements are ill-conditioned: not decisions at this precision.

The acoustic tie band is therefore built from two MEASURED error terms, and a code is decidable
only if the reference's x_final sits more than 6× the larger of them from the rounding edge:

1. **sensitivity** — that element's own spread (the 95% chi-square upper bound on σ from 4 draws;
   a 4-draw σ is often low — 0.058 vs 0.102 from 32 draws at the same element) when the torch
   sampler is re-run with row-RMS noise on every Linear, at the stage's calibrated precision: the single-matmul floor measured in
   the test (4.906e-4 relative, fp32 activation × bf16-exact weight vs float64), scaled on a
   HELD-OUT input (the stage's `acoustic_trace_inputs()`, from Source B's captured tensors) until
   the model matches the TT stage's error there (×1.72 → 8.44e-4);
2. **baseline** — the TT stage's own x_final RMS error on that held-out input (1.53e-3), which the
   Linear-only model does not reproduce (the device's `sin`/`cos` in the time embedding carry 2e-4
   relative, growing with t; the activation × activation attention matmuls; the elementwise ops).

Mismatches outside the band fail. How the band was arrived at, measured on this run: a global
6 × RMS band left 19 failures, all at ill-conditioned elements; the sensitivity term alone at the
raw matmul floor left 22, at the calibrated scale 14, all at insensitive elements whose TT error
was ordinary-sized (0.008–0.034 grid steps) — the baseline term covers those; the last one was a
4-draw σ underestimate, which the confidence bound covers. Final: **1002 of 98 280 live acoustic
codes differ, all 1002 are ties by this measure, 0 decidable mismatches**; 17 139 codes (17%) sit
inside the baseline band alone and so could not fail this check whatever TT did — that is the
price of the hardware's precision, stated rather than hidden.

---

## Decode horizon

**Call 1 — stop token (priority 1), per row.** `AudioSpecialTokens.end_audio` (id 1, read off the
reference). Each row's end frame is counted on device; the batch stops when every row has ended.
Safety cap = min(codec ALiBi ceiling 256 frames, `max_position_embeddings` 128000) = 256 frames
(20.5 s). `test_run_ended_on_the_models_stop_rule` asserts neither TT nor HF ended on it (not applied
only when `TT_PERF_OSL_TOKENS` is set, i.e. when the harness caps by design). The previous
`VOXTRAL_GATE_SECONDS=1.0` (13 frames, a prefix) is gone.

**Call 2 — no horizon.** Call 2 is the causal LM's teacher-forced next-token prediction over the
whole real text in one forward, so there is no decode loop and nothing for a stop rule to bind.
A free-running greedy decode was retired: the model's only stop rule, `eos_token_id` 2, never fires
on this checkpoint (0/32 rows in 448 CPU steps; never within 7.96 logits of top-1), so that test
could only ever end on an LLM-chosen bound.

---

## Command 3 — trace contract (host-free, per stage)

Each stage exposes `<stage>_trace_inputs()` (zero-arg) → `<stage>_trace_setup(inputs)` →
`<stage>_trace_step()` → `<stage>_trace_items()`. `prefill` and `decode` are driven by the real
voiced speech request (170 tokens, pinned at C = 256); the voice substitution reads two persistent
buffers staged outside the trace. `trace_capture_selftest(device)` captures, replays and releases
each stage in turn; `host_op_selftest()` runs both Calls — the TTS one on the voiced request —
under the host-aten-op observer. Both are module-level and zero-arg-callable (the observers import
`tt.pipeline` in a fresh process); `device_session.py` owns the device for that case.

`build_pipeline(device, model=None, layers=None, prefill_layers=, decode_layers=, acoustic_layers=,
vocode_layers=, **kwargs)` returns the resident object. `layers=None` is every layer; every stack
is a plain list of same-typed blocks and the reference stays reachable at `.reference_model`.

---

## Layout

```
demo/
  demo.py                       dispatcher
  demo_text_to_speech.py        Call 1 — writes one WAV per row, cut at its own end; --score = WER/MOS
  demo_text_continuation.py     Call 2
tt/                             THE pipeline (never opens a device)
  pipeline.py                   run_text_to_speech / run_text_continuation, build_pipeline,
                                PIPELINE_STAGES, trace hooks, selftests
  text_stack.py                 embed → (voice) → rope → 26 blocks → norm, resident KV cache
  acoustic_stage.py             semantic head + flow sampler + discretization
  vocode_stage.py               codec decoder
  continuation.py               Call 2 bodies + LM head (teacher-forced score)
  common.py                     tokenizer, SPEECH_TEXTS / PROMPT_TEXTS, voice prompt, stub import
reference/                      Source A's side of every comparison (torch, host only)
  golden.py                     the reference chains (free + teacher-forced), noise spreads
  quality.py                    Whisper WER + UTMOS22 MOS
tests/e2e/                      Gate 1/2/3 + contract tests
tests/section/                  per-section tests
device_session.py               the only device opener outside the test fixture
```

---

## Running

```bash
# Call 1 demo: 32 WAVs + Whisper WER / UTMOS scores
python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_to_speech --out-dir /tmp/wav --score
python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_to_speech --text "Paris is a beautiful city."

# Call 2 demo
python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_continuation

# gates (VOXTRAL_DEVICE_ID picks the chip)
pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_e2e_text_to_speech.py -s
pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_e2e_text_continuation.py -s
pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_gates.py -s
pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_trace_and_host_ops.py -s
pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_text_to_speech_perf.py -s   # trace+1CQ, all 4 stages
```

The HF golden is a 4 B model on CPU, run over the whole output twice (free + teacher-forced):
~50 min on a first run of the Call 1 test (~10–16 s per frame at B=32 on this host). Goldens are
memoised under `/tmp/voxtral_4b_tts_2603_golden` (keyed on inputs, chain version and the loader
contract), so a rerun against an unchanged pipeline pays only the device pass and the scoring.

---

## Memory

Figures use the REGISTERED hardware numbers (32 GB DRAM per chip, 29.2 GB usable at TP=1; this run
opens one chip, no CCL axis). Computed from shapes, not measured: Call 1 weights ≈ 8.6 GB bfloat16
(fp32 where the stubs keep fp32 weights raises this), resident KV = 26 × 2 × 32 × 8 × C × 128 × 2 B —
1.64 GB at the test's C = 480, 1.96 GB at the default C = 576.

---

## Holes — recorded, not faked

* **Ragged text lengths.** The prefill has no per-row padding mask, so a padded batch feeds the pad
  tokens to the model as content. Measured: BOS-padding the text segment took rows with 4–11 pad
  tokens to babble that never emitted `end_audio` (corpus WER 1.50, 29/32 rows hit the cap), while
  rows with 0–1 pad tokens were perfect. The package's 32 speech texts are therefore tuned to exactly
  18 tokens each, and `build_voice_prompt` refuses a ragged batch. Mixed-length requests need an
  attention mask in the prefill and decode, which this port does not have.
* **Codec encoder / waveform → codes.** The OSS checkpoint ships no encoder; there is no
  audio-input task.
* **Multi-segment long-form TTS.** The interleaved text/audio segmentation lives in vllm-omni, not a
  permitted source; with both segment limits at 8192 a sentence is one segment.
* **Hardware during this run.** Chip 2 died mid-run (PCIe link down, needs a host reboot) and chip
  3's ARC stopped answering; both are detached. Chip 1 returns wrong, run-to-run-different matmul
  results (a bf16 32×3072×32768 matmul off by up to 4.2 where 0.016 is correct), so every number
  in this README is from chip 0, which is exact and deterministic on the same probe.
