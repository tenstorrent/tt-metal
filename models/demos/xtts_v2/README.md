# coqui/XTTS-v2: end-to-end TTNN pipeline (text → speech)

## Platforms

| Device | Status | Notes |
|---|---|---|
| BH (Blackhole p300c), single chip | Supported, verified 2026-08-07 | one chip of a 4-chip QuietBox-2; requires a `TT_VISIBLE_DEVICES` pin + the single-chip mesh descriptor (see [Run](#run)). `l1_small_size=24576` |
| Multi-chip mesh (TP/DP) | Not supported | the pipeline opens one device; no `ShardTensor*Mesh`, no collectives, no mesh mapper |
| WH (Wormhole) | Not tested | nothing arch-specific in the modules, but no run exists; do not assume it works |

## Introduction

[`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) is a multilingual zero-shot
voice-cloning text-to-speech model: a GPT-2-style autoregressive text→mel-code decoder feeding a
HiFi-GAN vocoder, conditioned on a speaker embedding taken from a reference wav. This port runs
the full reference `Xtts.inference` chain (speaker encode → conditioning encode → AR mel-code
decode → GPT latents → vocode) on Tenstorrent hardware through 29 native TTNN modules, compared
against the HuggingFace/coqui reference implementation.

Parameter counts are measured from the real checkpoint, not estimated: **466.87 M total**:
`gpt` (GPT-2-style AR decoder, 30 blocks + conditioning-perceiver cross-attention) **441.02 M**,
`hifigan_decoder` (vocoder + nested ResNet speaker encoder) **25.86 M**.

The demo and the e2e test share one pipeline (`tt/pipeline.py::run_tts`), so a passing test means
a working demo. Only integer token bookkeeping and the repetition-penalty logit adjustment run on
host; next-token selection is on device (`ttnn.argmax`).

This branch adds only `models/demos/xtts_v2/` on top of `tenstorrent/tt-metal` main, so it
merges cleanly. Every gate below was re-run on main @ `32cdc03d6` (2026-08-07).

## Pipeline (all native TTNN)

```
speaker wav ─(16 kHz)─> res_net_speaker_encoder ─(l2norm)─> d-vector g [1,512,1]
            ─(mel 80) ─> conditioning_encoder → perceiver_resampler → dropout1d
                                                       └─> cond_latent [1,32,1024]
text ──(VoiceBpeTokenizer)──────────────────────────────> text tokens
cond_latent + text ─(prefix seed)─> g_p_t2_inference_model ── AR greedy ──> mel codes [1,N]
codes + cond_latent ──────────────> g_p_t (return_latent) ──> gpt_latents [1,N-4,1024]
gpt_latents + g ──────────────────> hifi_decoder ──> waveform [1,1,S] @ 24 kHz
```

Stage contract: `PIPELINE_STAGES = [speaker_encode, conditioning_encode, gpt_prefill, gpt_decode,
gpt_latents, vocode]`. The chain is fully self-fed: no reference tensor is injected at any joint.

## Key model parameters

| Parameter | Value |
|---|---|
| Reference class | `TTS.tts.models.xtts.Xtts` (coqui runtime, not HF `AutoModel`) |
| Total parameters | 466.87 M (`gpt` 441.02 M + `hifigan_decoder` 25.86 M) |
| AR decoder | GPT-2 style, 30 blocks, hidden 1024 |
| Conditioning latent | `[1, 32, 1024]` (conditioning encoder + perceiver resampler) |
| Speaker embedding | 512-d d-vector, L2-normalized (ResNet speaker encoder) |
| Tokenizer | `VoiceBpeTokenizer` (multilingual) |
| Speaker-wav input / mel bins | 16 kHz / 80 |
| Output audio | 24 kHz mono waveform |
| Decode | greedy, `do_sample=False`, `repetition_penalty=5.0` |
| KV-cache | none by default (repeat-prefill); experimental `decode_mode="kv"`/`"trace"` opt-ins are accuracy-capped (see [Performance](#performance)) |

## Modules (29, all invoked in the e2e path)

Ordered leaf → composite, as the pipeline builds them (`tt/pipeline.py::_MODULE_ORDER`):

| Subsystem | Modules |
|---|---|
| GPT AR decoder (7) | `conv1_d`, `learned_position_embeddings`, `dropout1d`, `g_p_t2_block`, `g_p_t2_model`, `g_p_t2_inference_model`, `g_p_t` |
| Conditioning (7) | `group_norm32`, `q_k_v_attention_legacy`, `attend`, `g_e_g_l_u`, `attention_block`, `conditioning_encoder`, `perceiver_resampler` |
| Speaker encoder (8) | `adaptive_avg_pool2d`, `s_e_layer`, `s_e_basic_block`, `instance_norm1d`, `mel_scale`, `mel_spectrogram`, `pre_emphasis`, `res_net_speaker_encoder` |
| HiFi-GAN vocoder (7) | `weight_norm`, `parametrization_list`, `parametrized_conv1d`, `parametrized_conv_transpose1d`, `res_block1`, `hifigan_generator`, `hifi_decoder` |

Invocation is proven by an execution tracker (`P.instrument_modules()` / `P.INVOKED`): each module
is recorded when it actually runs, not by the caller's optimism.

## Precision

Weights and activations are bfloat16. Float32 is used deliberately at a few points: the
repetition-penalty / presence bookkeeping around the logits, the host-side mel and conditioning
seeds, the speaker-encoder input waveform and its log-mel front end, and the LM-head projection,
which runs `MathFidelity.HiFi4` with `fp32_dest_acc_en=True` and `packer_l1_acc=True`. The
speaker-encoder input is uploaded in fp32 because bf16 quantization of the 16 kHz waveform,
not any kernel, was the d-vector accuracy floor: near-silent mel bins carry large relative
error, which `log(mel + 1e-6)` amplifies (see
[Accuracy decomposition](#accuracy-decomposition)).

## Run

Weights download from HuggingFace on first use; accept the Coqui Public Model License on
`coqui/XTTS-v2` first. Build the standard tt-metal Python environment per
[`INSTALLING.md`](../../../INSTALLING.md), then add the reference stack:

```bash
uv pip install -r models/demos/xtts_v2/requirements.txt
uv pip install torchaudio==2.11.0+cpu torchcodec==0.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

On a multi-chip Blackhole box, pin one chip and point at the single-chip mesh descriptor
(omitting `TT_VISIBLE_DEVICES` on a 4-chip box fails in `control_plane.cpp` at open):

```bash
export TT_METAL_HOME=$PWD
export TT_VISIBLE_DEVICES=1
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
```

```bash
# e2e gate (Gate 1 native / Gate 2 all-29-invoked / per-stage PCC / Gate 3 waveform >= 0.95)
XTTS_E2E_N=40 python -m pytest models/demos/xtts_v2/tests/e2e/test_e2e_tts.py -s

# demo -> writes a 24 kHz wav and prints the achieved PCC
python -m models.demos.xtts_v2.demo.demo_tts \
    --text "hello world." --language en --tokens 40 --out /tmp/xtts_tt.wav

# smoke: every module forwards on device
python -m pytest models/demos/xtts_v2/tests/e2e/test_00_forward_on_device.py -s

# trace + 2CQ substrate self-tests and timing
python -m pytest models/demos/xtts_v2/tests/e2e/test_trace_2cq.py models/demos/xtts_v2/tests/e2e/test_trace_2cq_timing.py -s
```

`XTTS_E2E_N` caps the AR horizon (both TT and HF sides) so the on-device gate stays fast.

## Correctness gates (text `"hello world."`, `en`, N = 40)

Measured 2026-08-07 on main @ `32cdc03d6`, single Blackhole p300c chip
(QuietBox-2), greedy decode
with repetition penalty. Every row is asserted by `tests/e2e/test_e2e_tts.py`; the demo wrote a
valid 24 kHz WAV (44,544 samples) in the same session.

| Gate | Result |
|---|---|
| Gate 1: routed modules composed as-is as native TTNN (no reference module substituted in the chain) | PASS (structural) |
| Gate 2: all 29 modules invoked in the real forward path | **PASS 29/29** (`missing=[]`) |
| Gate 3: e2e waveform PCC vs HF reference ≥ 0.95 | **0.9904 PASS** |

Per-stage PCC (each TT stage vs the HF reference run on the previous TT output; every stage gated
at ≥ 0.95):

| Stage | PCC |
|---|---|
| speaker embedding (`res_net_speaker_encoder`) | 0.9996 |
| conditioning latent (encoder + perceiver) | 0.9985 |
| AR token match (TT vs HF greedy, capped N) | **1.0** |
| AR per-step logits | 0.9993 |
| GPT latents (`g_p_t` on TT codes) | 0.9994 |
| waveform (the gated number) | **0.9904** |
| _supplementary: full independent TT chain vs full HF chain_ | _0.9279_ |
| _supplementary: full-chain log-mel spectral PCC / mean L1 (phase-insensitive)_ | _0.9966 / 0.287_ |

The supplementary full-chain numbers are printed, not gated: they compound every stage's error.
The raw-sample number penalizes the phase HiFi-GAN generates; the log-mel spectral number absorbs
phase and is the perceptually meaningful yardstick (see
[Accuracy decomposition](#accuracy-decomposition)).

## Accuracy decomposition

Re-derive every number in this section with one print-only command (nothing is gated):

```bash
python -m pytest models/demos/xtts_v2/tests/e2e/test_accuracy_decomposition.py -s
```

**2x2 vocoder ablation**: swap TT/HF GPT latents and d-vector independently through the reference
HiFi-GAN (PCC against the TT-chain waveform):

| Comparison | PCC | Isolates |
|---|---|---|
| vocode(lat_tt, g_tt) vs vocode(lat_tt, **g_hf**) | 0.9809 | the d-vector alone |
| vocode(**lat_tt**, g_tt) vs vocode(**lat_hf**, g_tt) | 0.9401 | the GPT latents alone |
| vocode(lat_tt, g_tt) vs vocode(lat_hf, g_hf) | 0.9279 | both, i.e. the full-chain number |

The **latents term is a metric artifact, not a defect**: `latents_pcc` is 0.9994 and that
6e-4-level error costs raw waveform PCC only because HiFi-GAN generates phase. The
phase-insensitive log-mel PCC is 0.9966. The **d-vector term was root-caused to the bf16 upload
of the speaker waveform** (fixed, 2026-08-06): the front end runs fp32, but the 16 kHz input was
uploaded as bf16; near-silent mel bins carry large relative error which `log(mel + 1e-6)`
amplifies. Sub-stage PCCs vs the fp32 reference, before and after:

| input dtype | mel | log-mel | InstanceNorm | embedding PCC | embedding cosine |
|---|---|---|---|---|---|
| bf16 (old) | 0.9999973 | 0.9924389 | 0.9673168 | 0.9713815 | 0.9714249 |
| fp32 (current) | 0.9999998 | 0.9999438 | 0.9996410 | **0.9995873** | **0.9995895** |

The fp32 upload moved the full-chain number from 0.7482 to 0.9279 (speaker embedding 0.9714 →
0.9996) with the gate's AR token match unchanged at 1.0.

## Performance

**Honest summary: this workload is host-dispatch-bound, not kernel-bound.** The wins that
survived end-to-end verification are host-side (build/upload hoisting, conv weight preparation);
every structural decode lever below was implemented, measured, and hit a correctness ceiling,
documented as such. The rebase itself is fresh evidence for the diagnosis: moving from the
2026-06 base to current main, with zero model-code changes, cut the warm wall ~3.5×. The drift
between the two bases is tt-metal host-dispatch work, which is exactly where this model's time goes.

Wall clock, resident pipeline (`build_pipeline`: weights upload once per process), N = 40 tokens
(44,544 samples = 1.856 s of audio), single p300c chip at AICLK 1350 MHz, measured 2026-08-07 on
main @ `32cdc03d6` (second run for stability: 741.2 ms cold / 201.9 ms warm):

| Metric | Value |
|---|---|
| Cold forward (includes one-time program compile) | 750.4 ms (RTF 0.40) |
| **Warm forward** | **199.5 ms (RTF 0.107)** |
| Warm speedup vs per-forward weight upload (pre-`build_pipeline`) | **3.39×** (measured at introduction, pre-rebase base) |

Warm wall split, measured on the pre-rebase base (warm wall there 695.9 ms; the split is not
re-measured on the rebased tip): AR decode (40 eager steps) 399 ms (57%), HiFi-GAN vocoder 166 ms
(24%), speaker encoder 30 ms (4%), latents / conditioning / perceiver 22 ms (3%), glue + host
feature extraction 79 ms (11%).

Stage walls measured with device syncs around each stage entry point; the wrapped and unwrapped
warm walls are identical, so stages fully serialize; there is no inter-stage overlap left to
reclaim. Two host-side levers are already in:

1. **Resident pipeline** (`build_pipeline`): weights upload once per process instead of once per
   forward (the 3.39× warm speedup).
2. **Conv weight pre-preparation** (`ttnn.prepare_conv_weights` per (weight, input shape)):
   `ttnn.conv2d` otherwise re-prepares raw OIHW weights on *every* call: a device→host pull-back,
   host prep, and H2D push per conv, ~74 ms of pure host overhead per forward in the speaker
   encoder alone (104 → 30 ms). Bit-identical (the same preparation function conv2d calls
   internally); the e2e gate did not move a digit.

**Measured ceilings (why the remaining warm wall is what it is).** Three structural levers were
implemented and gated; all three hit the same correctness wall, which is bf16 numerics, not
engineering effort:

- **KV-cached decode step** (available as the experimental opt-in `decode_mode="kv"`): removes the
  repeat-prefill recompute, but per-step logits drift at the 0.9996-PCC level from the eager bf16
  trajectory (different kernel scheduling at sequence length 1 vs repeat-prefill), and over 40
  greedy steps a thin margin (steps 18–21 on the gate text) flips an argmax. The gate requires
  `ar_token_match == 1.0` against the eager trajectory, so the lever is accuracy-capped. Even an
  fp32 attention/head variant diverges *more*: passing requires reproducing eager's exact bf16
  arithmetic, not approximating fp32 truth.
- **Fixed-capacity traced decode** (`decode_mode="trace"`): same ceiling, same cause: tracing
  pins the sequence length, which changes kernel scheduling, which drifts the logits.
- **Traced non-AR stages** (speaker encoder, vocoder): trace replay is not bit-identical on this
  stack: captured intermediates are freed after capture and their addresses reused, so replay
  corrupts live state (observed: garbage waveform with intact codes); the vocoder additionally
  performs a per-call host→device upload that trace capture forbids outright. The experiment was
  removed rather than shipped behind a flag.

On the pre-rebase base, the eager AR decode's ~10 ms/step of host dispatch was 57% of the warm
wall. Historical
planner evidence: an automated `tt_hw_planner optimize` sweep (25 attempts across 14 ops) moved
`device_ms` 98.52 → 98.49 (1.00×) with ~94.9 ms of ~98.5 ms being host dispatch overhead, so op-level
levers provably cannot reach this bottleneck. A whole-pipeline Tracy profile (reduced-depth
configuration `TT_PERF_LAYERS=2`, 4 tokens, pre-rebase base; absolute numbers not comparable to
the walls above):
21,752 op invocations, 247.9 ms device-kernel, 114.0 ms host-dispatch; `SliceDeviceOperation` alone
is 3.2% of device time but 34% of host dispatch, the repeat-prefill signature of re-slicing a
uniquely-shaped growing sequence tensor per token.

Reproduce:

```bash
# bounded warm-protocol perf test (resident pipeline, full depth)
python -m pytest models/demos/xtts_v2/tests/e2e/test_tts_perf.py -s

# whole-pipeline Tracy profile + per-op CSV (reduced-depth perf configuration)
TT_PERF_TRACE=1 TT_PERF_NUM_CQ=2 TT_PERF_MAX_NEW_TOKENS=4 TT_PERF_LAYERS=2 \
  python -m tracy -r -p -m pytest models/demos/xtts_v2/tests/e2e/test_tts_perf.py::test_tts_perf

# trace + 2CQ self-tests / timing harness
python -m pytest models/demos/xtts_v2/tests/e2e/test_trace_2cq.py -s
python -m pytest models/demos/xtts_v2/tests/e2e/test_trace_2cq_timing.py -s
```

## Trace + 2CQ

- **AR decode (default path): not supported, by construction.** The perf harness raises
  `TRACE_REPLAY_SKIPPED = AttributeError("pipeline exposes no decode_step(state); its decode is
  repeat-prefill …")`. Trace replay needs a fixed-shape, cacheable single step; the default decode
  grows its shape every step, so there is no stable program to capture. The fixed-shape
  alternatives exist as experimental opt-ins (`decode_mode="trace"` / `"kv"`) and hit the measured
  accuracy ceiling documented in [Performance](#performance).
- **Non-AR stages: measured, and not viable on this stack.** Trace capture/replay of the speaker
  encoder and vocoder was implemented and tested (2026-08-06): replay is not bit-identical
  (captured intermediates are freed after capture and their addresses reused, so replay corrupts
  live state), and the vocoder performs a per-call host→device upload that capture forbids
  ("Writes are not supported during trace capture"). The experiment was removed; the two harnesses
  above remain as self-tests of the ttnn trace/2CQ substrate itself.

## Determinism

Decode is greedy (`do_sample=False`) with a fixed repetition penalty and `torch.manual_seed(0)` in
the gate, and next-token argmax runs on device. Measured 2026-08-06: three forwards in
one process are bit-identical, and three fresh-process runs produce identical SHA-256 hashes for
both the waveform and the codes. Cross-process determinism also holds on the refreshed tip
(re-verified 2026-08-07 on main @ `32cdc03d6`): the e2e gate, the accuracy-decomposition test, and
the demo (three separate processes) print the identical waveform PCC, 0.9903768689534943. The AR
token match against HF greedy decode is exact (1.0) at the gated horizon.

## Known limitations

- **Default AR decode is repeat-prefill** (no KV-cache). KV-cached and fixed-capacity traced
  decode exist as experimental opt-ins (`decode_mode="kv"` / `"trace"`) and are accuracy-capped at
  bf16 (see [Performance](#performance)); they are not the default and the gate does not use them.
- **Single chip only.** No mesh sharding or collectives.

## Layout

```
models/demos/xtts_v2/
  tt/pipeline.py                     # the ONE shared chained forward (demo + tests import this)
  demo/demo_tts.py                   # runnable demo (argparse + __main__)
  tests/e2e/test_e2e_tts.py          # e2e gate: Gate 1/2/3 + per-stage PCC
  tests/e2e/test_accuracy_decomposition.py  # print-only: 2x2 ablation, sub-stage PCCs, log-mel
  tests/e2e/test_00_forward_on_device.py   # per-module device forward smoke
  tests/e2e/test_tts_perf.py         # Tracy / 2CQ / trace-replay perf harness
  tests/e2e/test_trace_2cq*.py       # trace + 2CQ self-tests and timing
  tt/modules/*.py                    # the 29 native TTNN modules
  reference.py                       # loads the native coqui reference model (demo + tests)
  requirements.txt                   # demo-specific pip deps (coqui-tts, transformers<5)
```
