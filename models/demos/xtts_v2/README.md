# coqui/XTTS-v2 — end-to-end TTNN pipeline (text → speech)

## Platforms

| Device | Status | Notes |
|---|---|---|
| BH (Blackhole p300c), single chip | Supported, verified 2026-08-05 | one chip of a 4-chip QuietBox-2; requires a `TT_VISIBLE_DEVICES` pin + the single-chip mesh descriptor (see [Run](#run)). `l1_small_size=24576` |
| Multi-chip mesh (TP/DP) | Not supported | the pipeline opens one device; no `ShardTensor*Mesh`, no collectives, no mesh mapper |
| WH (Wormhole) | Not tested | nothing arch-specific in the stubs, but no run exists — do not assume it works |

## Introduction

[`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) is a multilingual zero-shot
voice-cloning text-to-speech model: a GPT-2-style autoregressive text→mel-code decoder feeding a
HiFi-GAN vocoder, conditioned on a speaker embedding taken from a reference wav. This port runs
the full reference `Xtts.inference` chain (speaker encode → conditioning encode → AR mel-code
decode → GPT latents → vocode) on Tenstorrent hardware through 29 native TTNN stubs, compared
against the HuggingFace/coqui reference implementation.

Parameter counts are measured from the real checkpoint, not estimated: **466.87 M total** —
`gpt` (GPT-2-style AR decoder, 30 blocks + conditioning-perceiver cross-attention) **441.02 M**,
`hifigan_decoder` (vocoder + nested ResNet speaker encoder) **25.86 M**.

The demo and the e2e test share one pipeline (`tt/pipeline.py::run_tts`), so a passing test means
a working demo. Only integer token bookkeeping and the repetition-penalty logit adjustment run on
host; next-token selection is on device (`ttnn.argmax`).

## This branch

`xtts-v2-bringup` = the `tt_hw_planner`-generated bring-up plus three attention-fusion commits:

| Commit | Change |
|---|---|
| `4e17c472b9` | fuse GPT-2 attention head split/merge via `ttnn.transformer.split_query_key_value_and_split_heads` / `concatenate_heads` (removes per-head layout churn) |
| `37d9c5660a` | fuse GPT-2 attention into `ttnn.transformer.scaled_dot_product_attention` (FlashAttention-2) |
| `05a5f4aba0` | fuse the conditioning encoder's `QKVAttentionLegacy` into `ttnn` SDPA |

Base: `apande-TT/tt-metal` `feature/tt-hw-planner` @ `23e613b4`
([tt-metal PR #46283](https://github.com/tenstorrent/tt-metal/pull/46283), 2026-07-07); that
branch forks upstream `tenstorrent/tt-metal` @ `88873ad0` (2026-06-07). It therefore does **not**
diff cleanly against current tt-metal `main` — a rebase would need the full e2e gate re-run to
re-establish the numbers below.

**Three further commits were tried and dropped.** Custom tt-lang kernels (head-split,
residual-add, concat-heads) each passed their op-level check but end-to-end verification showed
the head-split kernel broke decode correctness (AR token match vs the HF reference fell from 1.0
to 0.0, e2e PCC −0.05); the other two depend on its rewrite of `g_p_t2_block.py` and could not be
separated from it. All three are out. Bisect (each point e2e-tested): base PASS 0.9909 → head
split/merge PASS 0.9909 → SDPA PASS 0.9939 → **tt-lang head-split FAIL** → conditioning SDPA on
top of SDPA PASS 0.9936. The generic lesson, since it applies to any op-level optimizer:
**a device-time win measured per-op can still be an end-to-end correctness regression — re-run
the e2e gate on the final tip, not just the op benchmark.**

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
gpt_latents, vocode]`. The chain is fully self-fed — no reference tensor is injected at any joint.

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
| KV-cache | none — decode is repeat-prefill (see [Performance](#performance)) |
| Component split (planner) | REUSE 3 / ADAPT 0 / NEW 29 |

## Graduated modules (29, all invoked in the e2e path)

Ordered leaf → composite, as the pipeline builds them (`tt/pipeline.py::_STUB_ORDER`):

| Subsystem | Modules |
|---|---|
| GPT AR decoder (7) | `conv1_d`, `learned_position_embeddings`, `dropout1d`, `g_p_t2_block`, `g_p_t2_model`, `g_p_t2_inference_model`, `g_p_t` |
| Conditioning (7) | `group_norm32`, `q_k_v_attention_legacy`, `attend`, `g_e_g_l_u`, `attention_block`, `conditioning_encoder`, `perceiver_resampler` |
| Speaker encoder (8) | `adaptive_avg_pool2d`, `s_e_layer`, `s_e_basic_block`, `instance_norm1d`, `mel_scale`, `mel_spectrogram`, `pre_emphasis`, `res_net_speaker_encoder` |
| HiFi-GAN vocoder (7) | `weight_norm`, `parametrization_list`, `parametrized_conv1d`, `parametrized_conv_transpose1d`, `res_block1`, `hifigan_generator`, `hifi_decoder` |

Invocation is proven by an execution tracker (`P.instrument_stubs()` / `P.INVOKED`) — each module
is recorded when it actually runs, not by the caller's optimism.

## Precision

Weights and activations are bfloat16. Float32 is used deliberately at a few points: the
repetition-penalty / presence bookkeeping around the logits, the host-side mel and conditioning
seeds, and the LM-head projection, which runs `MathFidelity.HiFi4` with `fp32_dest_acc_en=True`
and `packer_l1_acc=True`. The supplementary full-chain PCC below is dominated by the vocoder's
bf16 sensitivity to the d-vector — that is the port's known precision-limited path.

## Run

Weights download from HuggingFace on first use — accept the Coqui Public Model License on
`coqui/XTTS-v2` first. Build the standard tt-metal Python environment per
[`INSTALLING.md`](../../../INSTALLING.md), then add the reference stack:

```bash
pip install coqui-tts==0.27.5 "transformers<5"
pip install torchaudio==2.11.0+cpu torchcodec==0.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
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

# smoke: every stub forwards on device
python -m pytest models/demos/xtts_v2/tests/e2e/test_00_forward_on_device.py -s

# per-component PCC (needs golden tensors in _captured/, not committed — see Known limitations)
python -m pytest models/demos/xtts_v2/tests/pcc/ -v
```

`XTTS_E2E_N` caps the AR horizon (both TT and HF sides) so the on-device gate stays fast.

## Correctness gates (text `"hello world."`, `en`, N = 40)

Measured 2026-08-05 on this branch's tip, single Blackhole p300c chip (QuietBox-2), greedy decode
with repetition penalty. Every row is asserted by `tests/e2e/test_e2e_tts.py`; the demo wrote a
valid 24 kHz WAV (44,544 samples) in the same session.

| Gate | Result |
|---|---|
| Gate 1 — routed stubs composed as-is as native TTNN (no reference module substituted in the chain) | PASS (structural) |
| Gate 2 — all 29 graduated modules invoked in the real forward path | **PASS 29/29** (`missing=[]`) |
| Gate 3 — e2e waveform PCC vs HF reference ≥ 0.95 | **0.9936 PASS** |

Per-stage PCC (each TT stage vs the HF reference run on the previous TT output; every stage gated
at ≥ 0.95):

| Stage | PCC |
|---|---|
| speaker embedding (`res_net_speaker_encoder`) | 0.9710 |
| conditioning latent (encoder + perceiver) | 0.9987 |
| AR token match (TT vs HF greedy, capped N) | **1.0** |
| AR per-step logits | 0.9993 |
| GPT latents (`g_p_t` on TT codes) | 0.9994 |
| waveform (the gated number) | **0.9936** |
| _supplementary: full independent TT chain vs full HF chain_ | _0.7482_ |

The supplementary full-chain number is printed, not gated: it compounds every stage's error and is
dominated by the HiFi-GAN vocoder's bf16 sensitivity to d-vector conditioning. It is reported for
transparency.

## Performance

**Honest summary: op-level optimization produced no speedup, because this workload is
host-dispatch-bound, not kernel-bound.**

Automated `tt_hw_planner optimize` sweep, 25 attempts across 14 distinct ops (grid / dtype /
tt-lang / C++ / host levels), single Blackhole p300c chip:

| Metric | Value |
|---|---|
| `device_ms` before → after | **98.52 → 98.49 ms (1.00×, +0.03%)** |
| of which host dispatch overhead | **~94.9 ms of ~98.5 ms** |
| Committed micro-wins surviving e2e verification | 3 (the attention fusions above) |
| Trace + replay of the decode step | not applicable (see [Trace + 2CQ](#trace--2cq)) |

Whole-pipeline Tracy profile (a *separate, reduced-depth* measurement configuration —
`TT_PERF_LAYERS=2`, 4 new tokens — so its absolute numbers are not comparable to the `device_ms`
row above): 21,752 op invocations, 247.9 ms total device-kernel time, 114.0 ms total
host-dispatch time.

| Op | % device time | count | % host-dispatch time |
|---|---|---|---|
| `MatmulDeviceOperation` | **28.2%** | 3094 | 11.4% |
| `UntilizeWithUnpaddingDeviceOperation` | **16.8%** | 2810 | 8.9% |
| `BinaryNgDeviceOperation` (elementwise) | 11.4% | 4100 | 7.0% |
| `ReshapeViewDeviceOperation` | 7.5% | 1382 | 3.0% |
| `TilizeWithValPaddingDeviceOperation` | 7.2% | 1728 | 4.9% |
| `PermuteDeviceOperation` | 5.2% | 1074 | 4.2% |
| `LayerNormDeviceOperation` | 5.2% | 622 | — |
| `SliceDeviceOperation` | 3.2% | 3390 | **34.0%** |

**Reading this.** Matmul dominating device time is expected (GPT-2 attention/MLP plus HiFi-GAN
convolutions as matmul). Two findings are actionable:

1. **Untilize + Tilize together cost ~24% of device time.** The 29 stubs were graduated
   independently and do not share a consistent tile layout across component boundaries, so layout
   conversions are paid at every seam. A layout/fusion pass across seams is the remaining
   device-side lever.
2. **`SliceDeviceOperation` is only 3.2% of device time but 34% of all host dispatch.** This is
   the repeat-prefill signature: with no KV-cache, every generated token re-slices a new,
   uniquely-shaped growing causal-mask/sequence tensor, which the shape-keyed resident weight
   cache cannot reuse, forcing a fresh host round-trip per token. The mask-cache log confirms it —
   a new tensor upload for every sequence length from 43 to 87+, one per AR step.

Wall clock is far from real-time: `FORWARD_WALL_MS = 117,560` for a 4-token decode at reduced
depth, with AICLK thermally clamped to 800 MHz (nominal 1350 MHz) on that host.

Reproduce:

```bash
# whole-pipeline Tracy profile + per-op CSV (reduced-depth perf configuration)
TT_PERF_TRACE=1 TT_PERF_NUM_CQ=2 TT_PERF_MAX_NEW_TOKENS=4 TT_PERF_LAYERS=2 \
  python -m tracy -r -p -m pytest models/demos/xtts_v2/tests/e2e/test_tts_perf.py::test_tts_perf

# trace + 2CQ self-tests / timing harness
python -m pytest models/demos/xtts_v2/tests/e2e/test_trace_2cq.py -s
python -m pytest models/demos/xtts_v2/tests/e2e/test_trace_2cq_timing.py -s
```

**The structural lever, not yet taken:** add a real KV-cache plus a fixed-shape single-token
`decode_step`. That removes the per-token re-slice (finding 2), makes the decode step
trace-capturable, and is the prerequisite for any meaningful speedup here. Op-level levers
(grid / dtype / kernel choice) provably cannot reach this bottleneck.

## Trace + 2CQ

- **AR decode: not supported, by construction.** The perf harness raises
  `TRACE_REPLAY_SKIPPED = AttributeError("pipeline exposes no decode_step(state); its decode is
  repeat-prefill …")`. Trace replay needs a fixed-shape, cacheable single step; this decode grows
  its shape every step, so there is no stable program to capture. Fixing this is the structural
  lever above, not a knob.
- **Non-AR stages (speaker encoder, conditioning encoder, vocoder): plausible candidates, not
  verified.** They are architecturally fixed-shape single-shot forwards, and `tt/pipeline.py`
  carries a per-stage trace/2CQ contract (`PIPELINE_STAGES`, `trace_capture_selftest(device)`)
  plus the two harnesses above. Those harnesses were **not exercised on this tip** — treat
  non-AR trace/2CQ as unexplored, not as working.

## Determinism

Decode is greedy (`do_sample=False`) with a fixed repetition penalty and `torch.manual_seed(0)` in
the gate, and next-token argmax runs on device, so a run is deterministic by construction on a
given chip. The AR token match against HF greedy decode is exact (1.0) at the gated horizon. A
repeat-run determinism sweep was not performed — the claim here is structural, not measured.

## Reference (GPU) comparison

Published third-party numbers, cited as-is — no independent GPU measurement by us, and the
conditions do not match (this port is a single-chip bring-up at a capped AR horizon).

| Path | Hardware | Metric | Value | Source |
|---|---|---|---|---|
| This work | 1× Blackhole p300c | e2e waveform PCC vs reference | 0.9936 | this branch, `test_e2e_tts.py` |
| This work | 1× Blackhole p300c | real-time factor | not approached (117.6 s / 4 tokens at reduced depth) | this branch, `test_tts_perf.py` |
| Reference | GPU (RTX 5090 class) | RTF | ≈ 0.3× (≈3× faster than real-time) | [GIGAGPU TTS latency benchmarks](https://gigagpu.com/tts-latency-benchmarks/) |
| Reference | GPU | first-chunk latency | 150–400 ms (320 ms on RTX 5090) | [GIGAGPU XTTS-v2 VRAM](https://gigagpu.com/xtts-v2-vram-requirements/) |
| Reference | CPU | RTF | ≈ 1.41× (slower than real-time) | [GIGAGPU TTS latency benchmarks](https://gigagpu.com/tts-latency-benchmarks/) |

## Known limitations

- **No KV-cache in the AR decode** (repeat-prefill). Root cause of both the host-dispatch bound
  and the missing trace path; end-to-end wall clock is far from real-time.
- **Single chip only.** No mesh sharding or collectives.
- **`demo/demo.py` is a dead auto-generated CPU scaffold** and fails on this checkpoint (coqui
  runtime, not HF `AutoModel`). Use `demo/demo_tts.py`.
- **Per-component PCC tests need golden tensors in `_captured/`**, which are gitignored and not
  committed. The e2e gate is the self-contained verification.
- **`tests/pcc/test_mel_spectrogram.py` fails** on a framing mismatch vs the torch reference (open
  bug; the e2e path is unaffected — `mel_spectrogram` is invoked and gated there).
- **Non-AR trace/2CQ is documented but unverified** (see above).
- Branch base predates current tt-metal `main` (see [This branch](#this-branch)).

## Layout

```
models/demos/xtts_v2/
  tt/pipeline.py                     # the ONE shared chained forward (demo + tests import this)
  demo/demo_tts.py                   # runnable demo (argparse + __main__)
  demo/demo.py                       # dead auto-generated scaffold — do not use
  tests/e2e/test_e2e_tts.py          # e2e gate: Gate 1/2/3 + per-stage PCC
  tests/e2e/test_00_forward_on_device.py   # per-stub device forward smoke
  tests/e2e/test_tts_perf.py         # Tracy / 2CQ / trace-replay perf harness
  tests/e2e/test_trace_2cq*.py       # trace + 2CQ self-tests and timing (unverified on this tip)
  tests/pcc/                         # 29 per-component PCC tests (need _captured/ goldens)
  _stubs/*.py                        # the 29 graduated native TTNN stubs
  e2e_plan.json                      # planner output
```

`_stubs/*.best_native`, `*.last_good_native` and `*.preiter_native` are the optimizer's per-attempt
snapshots, kept for provenance. `_stubs/{cpp_bmm,cpp_reduce,ttl_bmm,reduce_ttl}.py` are custom-kernel
helpers left over from the dropped tt-lang commits; no module on the verified path imports them.
