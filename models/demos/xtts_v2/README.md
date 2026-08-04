# coqui/XTTS-v2 — end-to-end TTNN pipeline (text → speech)

On-device TTNN implementation of [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
multilingual text-to-speech (~467M params: a GPT-2-style autoregressive text→mel decoder at
441M, plus a HiFi-GAN vocoder with ResNet speaker encoder at 26M). The full reference
`Xtts.inference` chain runs on graduated native TTNN stubs under `_stubs/`. Demo and e2e test
share one pipeline (`tt/pipeline.py::run_tts`), so a passing test means a working demo.

## This branch

`xtts-v2-bringup` = the tt_hw_planner-generated bring-up plus three attention-fusion commits:

- fuse GPT-2 attention head split/merge via `ttnn.transformer` ops (removes per-head layout churn)
- fuse GPT-2 attention into `ttnn` SDPA (FlashAttention)
- fuse conditioning `QKVAttentionLegacy` into `ttnn` SDPA (FlashAttention)

Base: `apande-TT/tt-metal` `feature/tt-hw-planner` @ `23e613b4`
([tt-metal PR #46283](https://github.com/tenstorrent/tt-metal/pull/46283), 2026-07-07);
that branch forks upstream `tenstorrent/tt-metal` @ `88873ad0` (2026-06-07).

Three further custom tt-lang kernel commits (head-split, residual-add, concat-heads) were tried
and dropped: end-to-end verification showed the head-split kernel broke decode correctness
(AR token match vs the HF reference fell to 0.0), and the other two could not be separated
from it.

## Pipeline (the chain, all native TTNN)

```
speaker wav ─(16 kHz)─> res_net_speaker_encoder ─(l2norm)─> d-vector g [1,512,1]
            ─(mel 80) ─> conditioning_encoder → perceiver_resampler → dropout1d
                                                       └─> cond_latent [1,32,1024]
text ──(VoiceBpeTokenizer)──────────────────────────────> text tokens
cond_latent + text ─(prefix seed)─> g_p_t2_inference_model ── AR greedy ──> mel codes [1,N]
codes + cond_latent ──────────────> g_p_t (return_latent) ──> gpt_latents [1,N-4,1024]
gpt_latents + g ──────────────────> hifi_decoder ──> waveform [1,1,S] @ 24 kHz
```

Autoregressive decode is greedy + repetition-penalty (`repetition_penalty=5.0`,
`do_sample=False`), the deterministic form of the real XTTS decode. Next-token selection is
on-device (`ttnn.argmax`); only integer token bookkeeping and the repetition-penalty logit
adjustment run on host. The pipeline is fully self-fed: no reference tensor is injected at
any joint. All 29 graduated modules are invoked in the real forward path.

## Run

Weights download from HuggingFace on first use — accept the Coqui Public Model License on
`coqui/XTTS-v2` first. Build the standard tt-metal Python environment per `INSTALLING.md`,
then add the reference stack:

```bash
pip install coqui-tts==0.27.5 "transformers<5"
pip install torchaudio==2.11.0+cpu torchcodec==0.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

On a multi-chip Blackhole box (verified on qb2, P300), pin one chip and point at the
single-chip mesh descriptor:

```bash
export TT_METAL_HOME=$PWD
export TT_VISIBLE_DEVICES=1
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
```

```bash
# e2e gate test (device)
XTTS_E2E_N=40 python -m pytest models/demos/xtts_v2/tests/e2e/test_e2e_tts.py -s

# runnable demo -> writes a 24 kHz wav and prints the achieved PCC
python -m models.demos.xtts_v2.demo.demo_tts \
    --text "hello world." --language en --tokens 40 --out /tmp/xtts_tt.wav
```

`XTTS_E2E_N` caps the AR horizon (both TT and HF) so the on-device gate stays fast.

## Verified results

Measured 2026-08-05 on this branch's tip, single Blackhole P300 chip (qb2), N=40 AR horizon,
greedy decode + repetition penalty. All 29 graduated modules invoked; the demo wrote a valid
24 kHz WAV (44,544 samples).

| stage | PCC vs HF reference |
|---|---|
| speaker embedding (`res_net_speaker_encoder`) | 0.9710 |
| conditioning latent (encoder + perceiver) | 0.9987 |
| AR token match (TT vs HF greedy, capped N) | 1.0 |
| AR per-step logits | 0.9993 |
| GPT latents (`g_p_t` on TT codes) | 0.9994 |
| **waveform — e2e gate, target ≥ 0.95** | **0.9936 PASS** |
| _supplementary: full independent TT-chain vs HF-chain_ | _0.7482_ |

The gated waveform number compares the HF reference vocoder against the pipeline's own TT
latents + d-vector, the same TT→reference gating used at every stage. The supplementary
full-chain number compounds every stage's error and is dominated by the HiFi-GAN vocoder's
bf16 sensitivity to d-vector conditioning; it is reported for transparency, not gated.

## Performance status

No meaningful speedup from op-level optimization: device time measured 98.52 ms before vs
98.49 ms after a 25-attempt automated sweep across 14 ops (+0.03%, 1.00x). Host dispatch
overhead alone is ~94.9 ms of that total, so the workload is host-dispatch-bound, not
kernel-bound, and op-level levers (grid/dtype/kernel choice) cannot move it.

The structural lever is the autoregressive decode itself: it is repeat-prefill with no
KV-cache, so every generated token re-runs the full growing sequence, and there is no
fixed-shape decode step for trace capture/replay to apply to. Adding a real KV-cache plus a
single-token `decode_step` is the next real optimization. Published GPU reference points
(RTF ≈ 0.3x, 150–400 ms first-chunk latency) are not approached.

`tt/pipeline.py` documents a per-stage trace/2CQ contract (`PIPELINE_STAGES`,
`trace_capture_selftest(device)`) for the fixed-shape non-AR stages; those have not been
independently exercised here.

## Known limitations

- `demo/demo.py` is an auto-generated CPU scaffold and fails on this checkpoint (coqui-native,
  not HF `AutoModel`). Use `demo/demo_tts.py`.
- Per-component PCC tests under `tests/pcc/` need golden tensors in `_captured/`, which are
  gitignored and not committed. The e2e test is the self-contained verification.
- `tests/pcc/test_mel_spectrogram.py` fails on a framing mismatch vs the torch reference
  (open bug).
- The AR decode has no KV-cache (see Performance status); end-to-end wall clock is far from
  real-time.

## Layout

```
models/demos/xtts_v2/
  tt/pipeline.py          # the ONE shared chained forward (demo + test import this)
  demo/demo_tts.py        # runnable per-task demo (argparse + __main__)
  tests/e2e/test_e2e_tts.py  # e2e gate (invocation, per-stage PCC, waveform >= 0.95)
  _stubs/*.py             # the 29 graduated native TTNN stubs
  tests/pcc/              # per-component PCC tests (need _captured/ goldens, not committed)
  e2e_plan.json           # the planner output
```
