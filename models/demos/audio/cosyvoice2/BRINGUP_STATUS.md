# CosyVoice2 TTNN bring-up — status handoff

Written 2026-09-18, for a future Claude Code session picking this up (this one
may not survive). Everything below is grounded in real, verified repo/test
state as of this commit — verify it again yourself before trusting it, same
discipline this whole bring-up has used. Don't take this file's claims about
what "used to be here" on faith any more than you'd take a verbal summary —
`git log`, `git show`, and re-running the test suite are always the ground
truth; this file is a map, not the territory.

**Bounty**: tenstorrent/tt-metal issue #54104 ($2,000, Hard tier). CosyVoice2
(Alibaba FunAudioLLM) TTNN bring-up: LLM (Qwen2-0.5B) → flow-matching decoder
→ HiFT vocoder → waveform.

**Repo**: `~/tt-metal`, fork `github.com/Sedherthe/tt-metal`, branch
`bringup/cosyvoice2-istft`. Reference repo (read-only, architecture patterns
only, NOT directly transferable — CosyVoice2 differs from CosyVoice1 in real
ways): `~/reference-cosyvoice1` (`ayewo/tt-metal`,
`bringup/cosy-voice-01`).

## What's built and verified (all real, all tested, all committed+pushed)

Everything below has a real PCC test in `models/demos/audio/cosyvoice2/tests/pcc/`
and passes on real N150 hardware. Full suite as of this write-up:
**118 passed, 0 failed** (`pytest models/demos/audio/cosyvoice2/tests/pcc/ -q`
— re-run this yourself first, don't trust this number once time has passed).

- iSTFT-as-matmul, HiFT vocoder (upsample/resblock/3-source-branch stack),
  `ConvRNNF0Predictor` (rebuilt from scratch this session — despite the name,
  NO recurrent layer, confirmed against real upstream source: 5 plain
  `Conv1d(k=3)+ELU` layers + `Linear(512,1)` + `abs()`), SineGen2/NSF
  excitation, Qwen2-0.5B LLM backbone with real sequence assembly/prefill/
  decode/RAS sampling, causal Conformer flow encoder, CFM flow-matching
  estimator (with real classifier-free guidance, `inference_cfg_rate=0.7`),
  outer `CausalMaskedDiffWithXvec`/`TtHiFTGenerator` wiring. Full pipeline
  (speech tokens → flow decoder → mel → real F0 predictor → HiFT vocoder →
  waveform) wired end-to-end, zero synthetic stand-ins anywhere.
- **Real CosyVoice2-0.5B checkpoint weights load into all four modules**
  (`FunAudioLLM/CosyVoice2-0.5B` on HF — `llm.pt`/`flow.pt`/`hift.pt`), each
  verified against its own real-checkpoint PCC test:
  `tests/pcc/test_hift_checkpoint.py`, `test_f0_predictor_checkpoint.py`,
  `test_flow_checkpoint.py`, `test_qwen2lm_checkpoint.py`. See
  `tt/checkpoint.py` for the shared download/remap helpers (`load_checkpoint_file`,
  `sub_state_dict`, `build_local_qwen2_checkpoint_dir`).
- **HiFT needs fp32, not bf16, with real weights** — bf16 PCC collapses to
  ~0.49 (real weights push `conv_post`'s pre-`exp()` values into a range bf16
  can't track precisely; random init never did this). `TtHiFTDecoder`/
  `TtHiFTGenerator` should be built `dtype=ttnn.float32` when loading a real
  checkpoint. F0 predictor and flow decoder do NOT have this problem at bf16
  (see below) — this is HiFT-specific, don't assume it generalizes.
- **`TtQwen2LM.generate()`'s stop-token/min_tokens handling is now correct**,
  matching real upstream `Qwen2LM.inference_wrapper`/`sampling_ids` exactly
  (fetched fresh from `cosyvoice/llm/llm.py`, not assumed): `stop_token_ids`
  checks all three real stop IDs, the break is unconditional (no
  `i >= min_tokens` gate — a real bug that existed until commit `c44cb8131a`),
  and `min_tokens` only masks `eos_token`'s logit pre-sampling. Covered by
  `test_device_generate_stops_on_any_stop_token_id`,
  `test_device_generate_masks_only_eos_before_min_tokens`,
  `test_device_generate_breaks_immediately_even_before_min_tokens` in
  `tests/pcc/test_qwen2lm_generate.py`.

Real Stage 1 targets (from the bounty issue itself, fetched fresh via
`gh api repos/tenstorrent/tt-metal/issues/54104` — don't re-derive these from
memory, re-fetch if unsure), measured with real infra (real LibriSpeech
test-clean sample, real `campplus.onnx`/`speech_tokenizer_v2.onnx`, real
Whisper ASR) in a scratch script (not yet a committed repo test — see
"Reproducing the Stage 1 eval" below):

| Target | Goal | Measured | Result |
|---|---|---|---|
| Token-level accuracy | >95% | 100% | PASS |
| WER | <5.0% | 4.17% | PASS |
| Speaker similarity | >0.60 cosine | 0.857 | PASS |
| RTF (non-streaming) | <1.0 | 8–20× | FAIL (expected — Stage 2/3 concern, zero perf optimization done) |

## THE OPEN PROBLEM — real audio still sounds noisy/robotic, root cause not found

The user listened to real synthesized audio (zero-shot cloned from a real
LibriSpeech reference clip) and reported it still sounds noisy, a bit
robotic, and the cloned voice doesn't match well — despite WER/speaker-sim
both passing comfortably. **This is real and unresolved.** Two things were
fixed along the way (see commits below) but neither fixed the actual sound
quality:

1. `generate()`'s min_tokens bug (real, fixed, `c44cb8131a`) — didn't
   actually fire on the specific real run tested, so didn't change the audio.
2. Flow decoder bf16→fp32 (real, measurable improvement: PCC 0.9994→0.9997,
   max abs diff 0.571→0.204) — but far more modest than HiFT's fix, and the
   user reported v1 (bf16 flow decoder) and v2 (fp32 flow decoder) sound
   "almost the same." **Not applied to the repo yet** — this was only tested
   in a scratch script (see below), not committed. If you do want it in the
   repo permanently: `ttnn.embedding` hard-requires bf16 weights (`TT_FATAL`
   otherwise), so only `TtSmallEmbedding`'s dtype needs to stay bf16 while
   the rest of `TtCausalMaskedDiffWithXvec` goes fp32 — see the monkeypatch
   pattern in the reproduction recipe below for exactly how.

### The real finding: our torch reference itself diverges from the actual official model

This is the important one. All PCC validation this entire session compared
our TT port against **our own hand-written torch reference** (built by
carefully transcribing real upstream source). That has a structural blind
spot: a bug that exists in how we transcribed something — even if done
carefully — would show up as high PCC on BOTH sides (TT matches our
reference), while both are still wrong relative to what the actual trained
model produces.

Real, external cross-check (not our own reference): `FunAudioLLM/CosyVoice2-0.5B`
ships `flow.decoder.estimator.fp32.onnx` — a real, officially-exported ONNX
version of the flow decoder's CFM estimator (the most complex piece: 12
attention blocks + classifier-free guidance). Feeding it the SAME real
`(x, mask, mu, t, spks, cond)` our torch reference solves with, using the SAME
real `flow.pt` weights:

**PCC 0.944, max abs diff 5.99, mean relative diff 8.2%** — our torch
reference's `CausalConditionalDecoderRef` genuinely diverges from the real
official model. This is pure torch vs. ONNX, zero ttnn/TT involvement — proof
the bug is upstream of the vocoder, in the flow decoder's estimator itself
(or in how the eval script feeds it), not something introduced by our TTNN
port or (necessarily) the vocoder the user suspected.

Confirmed the checkpoint itself isn't the issue: pulled the ONNX graph's
baked-in weights directly (`onnx.load(...).graph.initializer`) and diffed
against `flow.pt` — resnet conv weights match exactly (`max_diff=0.0`).
(Note: the ONNX export fuses Q/K/V/attention-output weights into constant-
folded graph nodes with no individual names, so those specific tensors
couldn't be diffed this same direct way — worth knowing if you try this
again.)

### What's been ruled out (verified against real source, not assumed)

Fetched and read line-by-line, not from memory: `cosyvoice/cli/cosyvoice.py`,
`cosyvoice/cli/model.py`, `cosyvoice/cli/frontend.py` +
`cosyvoice/utils/frontend_utils.py`, `cosyvoice/flow/flow.py`,
`cosyvoice/flow/flow_matching.py`, `cosyvoice/flow/decoder.py`,
`cosyvoice/hifigan/generator.py`, `cosyvoice/llm/llm.py`,
`cosyvoice/utils/common.py`, `cosyvoice/utils/mask.py`, plus external deps
`matcha-tts`'s `transformer.py`/`decoder.py` and `diffusers`' real
`Attention`/`AttnProcessor2_0`/`GELU` classes via direct Python introspection
(`inspect.getsource`, `diffusers` was already installed, no need to
reinstall).

Confirmed matching real source, NOT the bug:
- CFG (classifier-free guidance) — present and structurally correct in both
  `CausalConditionalCFMRef.solve_euler` (torch) and the TT device path.
- iSTFT magnitude clamp (`torch.clip(magnitude, max=1e2)`) — present both
  sides.
- Excitation branch (SineGen2/SourceModuleHnNSF) — matches real source
  exactly, including the "noise" branch being real dead code upstream too.
- RAS/nucleus sampling — matches real `cosyvoice/utils/common.py` exactly,
  including the `softmax(log_softmax(x)) == softmax(x)` no-op quirk and the
  vestigial unused `sampling` parameter.
- `text_normalize`/`frontend_utils.py` (`spell_out_number`, `split_paragraph`)
  — confirmed a true no-op for plain English test sentences under 80 tokens
  with no digits (doesn't explain anything we measured).
- Flow decoder concat order (x, mu, spks, cond via `einops.pack`) — matches.
- Attention mask construction (`add_optional_chunk_mask` reduces to a plain
  padding mask for non-streaming/`static_chunk_size=0`) — matches.
- Skip-connection timing (saved BEFORE `down_conv`, concatenated as
  `[x, skip]` before `up_resnet`) — matches, including for the single-stage
  (`channels=[256]`) config CosyVoice2 actually uses, where the "downsample"/
  "upsample" placeholders degenerate to plain stride-1 `CausalConv1d` (traced
  through the real `masks.append`/`masks[:-1]` bookkeeping to confirm no
  actual temporal downsampling happens).
- Timestep embedding's `scale=1000` factor in `SinusoidalPosEmb` — present.
- Resnet block math (`h=block1(x,mask); h+=mlp(t); h=block2(h,mask);
  return h+res_conv(x*mask)`) — matches.
- GELU (exact, `approximate="none"`, not tanh-approximate/GEGLU) — matches.
- `diffusers.Attention`/`AttnProcessor2_0` internals — `residual_connection`
  defaults `False` (no double-residual), `rescale_output_factor` defaults
  `1.0` (no-op), `to_out` bias `True`, Q/K/V bias `False`, mask preparation
  is shape-only (repeat/reshape for multi-head, no value changes) — all
  match what our hand-rolled attention does.

**Not yet checked**: the actual numeric behavior of `diffusers.Attention`'s
Q/K/V projection + SDPA call, end to end, with real weights — everything
above confirms the *documented*/*structural* behavior matches, but the
literal library class was never substituted in for a real, empirical
side-by-side. That's the recommended next step (below).

### Recommended next step

Swap our hand-rolled attention math in `BasicTransformerBlockRef.forward`
(one instance, e.g. `down_tbs[0]`) for a REAL `diffusers.models.attention_processor.Attention`
object, loaded with the real `flow.pt` weights for that block
(`decoder.estimator.down_blocks.0.1.0.attn1.*`), and compare its output
against our own implementation's for the same real input. If they match:
attention math is confirmed fine and the bug is somewhere else in the
estimator (worth then bisecting resnet vs. transformer-block contributions
separately, block by block, mid-solve). If they diverge: found it.

```python
from diffusers.models.attention_processor import Attention
attn = Attention(query_dim=256, heads=8, dim_head=64, dropout=0.0, bias=False)
# load real weights: attn.to_q.weight.data = flow_sd["decoder.estimator.down_blocks.0.1.0.attn1.to_q.weight"]
# ...to_k, to_v similarly; attn.to_out[0].weight/.bias from ...to_out.0.weight/.bias
# then attn(hidden_states, attention_mask=attn_bias) and diff against BasicTransformerBlockRef's own attn1 call
```

## Reproducing the Stage 1 eval / ONNX cross-check (scripts were in `/tmp` scratch, likely gone)

All of this session's diagnostic scripts lived in
`/tmp/claude-*/scratchpad/` — NOT committed, NOT guaranteed to survive this
session ending. If they're gone, here's what to rebuild and how, in order:

1. **Download real checkpoints** (re-download is cheap, HF caches):
   ```python
   from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file
   llm_sd = load_checkpoint_file("llm.pt")
   flow_sd = load_checkpoint_file("flow.pt")
   hift_sd = load_checkpoint_file("hift.pt")
   ```
   Plus real speaker/token models and the ONNX estimator, all from the same
   HF repo:
   ```python
   from huggingface_hub import hf_hub_download
   campplus_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="campplus.onnx")
   st_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="speech_tokenizer_v2.onnx")
   onnx_estimator_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="flow.decoder.estimator.fp32.onnx")
   ```
2. **Real preprocessing** (mel-spectrogram, fbank for xvec, whisper log-mel
   for speech tokenizer) had to be reimplemented by hand this session because
   installing `matcha-tts` via pip broke `torchaudio`'s ABI (see gotcha
   below) — the reimplementation is small, was saved to
   `/tmp/.../scratchpad/cv2_frontend.py`, likely gone. It's a direct,
   short transcription of `cosyvoice/cli/frontend.py`'s
   `_extract_speech_token`/`_extract_spk_embedding`/`_extract_speech_feat`
   and Matcha-TTS's `mel_spectrogram` (fetch both fresh from GitHub if
   needed — `FunAudioLLM/CosyVoice` and `shivammehta25/Matcha-TTS`, both
   public).
3. **Real test data**: `datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")`
   — 73 real LibriSpeech test-clean examples, single speaker (1272). Index 0
   was used as the reference/prompt, index 3 as the target text.
4. **ONNX cross-check**: build the real `(x, mask, mu, t, spks, cond)` tuple
   exactly as `CausalMaskedDiffWithXvecRef.inference` does internally (up to
   but not including the CFM solve), feed it to BOTH
   `flow_ref.decoder.estimator` (our torch reference — note it wants
   channels-LAST `[B,T,80]`, transpose from the ONNX model's channel-FIRST
   `[B,80,T]` convention before calling it) and the ONNX session, compare.

## Environment gotchas hit this session (see also memory files, same content)

- **Fresh build needs `uv pip install -e .` after `build_metal.sh`** —
  the C++ build alone doesn't install the `ttnn` Python package; without
  this, `import ttnn` resolves as a broken namespace package.
- **Never `pip`/`uv pip install` anything torch-adjacent without
  `--extra-index-url https://download.pytorch.org/whl/cpu`** — this venv's
  `torch` is a CPU-only build from PyTorch's own index; installing something
  that pulls in `torchaudio` from default PyPI silently breaks its ABI
  (`TT_FATAL`-style `OSError: Could not load this library:
  .../torchaudio/lib/_torchaudio.abi3.so`) even though `import torchaudio`
  succeeds. Diagnose via `uv pip show torch torchaudio` — the `+cpu` suffix
  should match on both. Fix: `uv pip install --extra-index-url
  https://download.pytorch.org/whl/cpu --reinstall "torchaudio==<version>"`.
  This is exactly what happened trying to install `matcha-tts` for its real
  `mel_spectrogram` code — broke 17 previously-passing tests, caught by
  immediately re-running the full suite, fixed, full suite reconfirmed clean
  before continuing. `onnx`/`onnxruntime`/`openai-whisper`/`jiwer` are all
  safe (pure Python or no torch dependency) — installed without issue.
- GitHub push auth: was broken (no `gh auth login`, no SSH key, no stored
  creds) at the start of this session, got fixed by the user at some point
  mid-session (unclear exactly when/how) — pushes succeeded later without
  intervention. If push fails with "could not read Username", that's this
  same issue — ask the user to run `gh auth login` or set up SSH.
- The user explicitly does NOT want `Co-Authored-By: Claude` in commit
  messages — author only. If you rewrite history to fix this on already-
  pushed commits, it needs `--force-with-lease` (confirm with the user
  first — this repo/branch is the user's own fork, low risk, but still
  confirm).

## Memory files (survive within this same environment, not across a full VM reset)

`/home/user/.claude/projects/-home-user-tt-metal/memory/` —
`cosyvoice2_bringup_state.md` (detailed real/claimed-progress log,
chronological) and `tt_metal_build_env_gotcha.md` (the two environment
gotchas above, same content). Read these too if they're still there; this
file is the durable (committed) version of the same information.
