# Voxtral-TTS on TTNN — status and resumption notes

**Read this first when picking the work back up.** It is written to be self-contained: state,
measurements, the traps that cost time, and what to do next. Architecture detail and the
reference-side findings live in `reference/PROVENANCE.md`; this file is the *work* state.

Branch: `lserbedzija/voxtral-tts-ttnn` (pushed). 17 commits, all under
`models/experimental/voxtral_tts/`. Nothing else in the repo is touched.

---

## 1. Where things stand

| Piece | State |
|---|---|
| CPU reference, 3 blocks + tokenizer + end-to-end pipeline | **done**, 30/30 vs upstream |
| Block 3 — codec decoder on TTNN | **done**, PCC 0.9994–0.9998, 242x real-time |
| Block 1 — 3.4B AR backbone on TTNN | **not started** ← the next big piece |
| Block 2 — flow-matching transformer on TTNN | **not started** |
| Codec **encoder** | **impossible** — weights absent from the public release |

The model is 4 networks but only 3 are portable. 118 tests pass with no device needed for 96 of
them (the reference suite runs weight-free off a vendored tensor manifest).

---

## 2. How to actually run things (do this before anything else)

There are **two** virtualenvs and neither is obvious.

**Main env — has ttnn, torch, torchaudio, transformers:**
```bash
source /localdev/lserbedzija/repos/xtts_ref_venv/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD      # from the repo root
```
Note this is the venv created for the XTTS-v2 work; the repo's own `python_env` is a docs env and
does **not** have torch/ttnn. `python_env` *does* have `black` if you need it.

**Run the tests** (the repo `conftest.py` needs torchvision, which is absent, hence `--noconftest`):
```bash
python -m pytest models/experimental/voxtral_tts/tests/test_common_ref.py \
  models/experimental/voxtral_tts/tests/test_tokenizer_ref.py \
  models/experimental/voxtral_tts/tests/test_backbone_pcc.py \
  models/experimental/voxtral_tts/tests/test_flow_pcc.py \
  models/experimental/voxtral_tts/tests/test_codec_pcc.py \
  models/experimental/voxtral_tts/tests/test_codec_ttnn_pcc.py \
  --noconftest -p no:cacheprovider -q
# expect: 118 passed
```
Passing the directory instead of the file list collects nothing — pass the files.

**Checkpoint** (8.0 GB, gitignored, CC BY-NC 4.0):
`models/experimental/voxtral_tts/reference/weights/` — `consolidated.safetensors`, `params.json`,
`tekken.json`, `voice_embedding/*.pt` (20 presets). Re-fetch per `reference/PROVENANCE.md`.
Without it, 96 reference tests still pass; the 22 device/weight tests skip.

**Second env — upstream comparison only** (needs einops + mistral-inference, no GPU required).
Recipe and rationale in `scripts/upstream_compare/README.md`. It lived in a session scratchpad and
is **gone**; recreate from the README. Also run `scripts/upstream_compare/fetch_upstream.py` to
re-download the two pinned vLLM-Omni files (gitignored on purpose).

**End-to-end speech** (needs ~50 s of CPU for the reference backbone):
```bash
# tokenizer prompt ids come from OUR reimplementation, no mistral_common needed
python models/experimental/voxtral_tts/reference/voxtral_pipeline_ref.py \
    --text "..." --voice neutral_male --threads $(nproc)
```

---

## 3. What is validated, and how

**Reference vs upstream: 30/30.** Harness in `scripts/upstream_compare/`.
- Block 1 vs `mistral_inference` (Mistral's own): RoPE table + application **bit-exact**,
  RMSNorm/SwiGLU/repeat_kv bit-exact, full 26-layer stack PCC 0.99999988.
- Block 1 input side + Blocks 2/3 vs vLLM-Omni's own `nn.Module`s: 37-code frame
  **bit-identical integers**, waveform PCC 0.99999982, all 8 decoder stages ≥ 0.99999.
- Tokenizer vs `mistral_common`: **exact token ids**, 15 prompts, 8 languages. Ground truth
  vendored at `tests/prompt_fixture.json`.

**End-to-end reference:** 0.0% WER (Whisper) on 4 runs — 24 words × 2 voices, a **125-word
paragraph in a single pass** (469 frames, 37.5 s), and French via `fr_female`.

**Block 3 on device:** PCC 0.9994–0.9998 vs the reference across T=1…1536, plus a real-speech
fixture test (`tests/real_frames_fixture.pt`, 6.4 KB of genuine Block 1+2 output) so the audible
path cannot silently regress — see trap #6.

---

## 4. Block 3 performance (warm, N150, defaults)

| T | audio | warm | RTF | vs real-time |
|---|---|---|---|---|
| 64 | 5.1 s | 43.6 ms | 0.0085 | 117x |
| 469 | 37.5 s | 155 ms | 0.0041 | 242x |
| 1500 | 120 s | 538 ms | 0.0045 | 223x |

Context: upstream report RTF **0.103 for their whole pipeline** on an H200. So Block 3 is ~4–8% of
the total budget — it is **not** where the end-to-end answer gets decided. Do not over-invest here.

### Optimizations applied, in order, with what each was worth

1. **bf16 attention** (default). Best PCC of all four dtype configs *and* faster; halves the bias
   tensor. bf16 **weights** are opt-in and now strictly bad — see trap #4.
2. **Chunked windowed attention**, slab 512, `chunk_min` 512. Turns attention from O(S²) into
   O(S·slab). At S=12000: warm 892→497 ms, cold 10580→1178 ms, mask **2304 MB→4.2 MB**. Exact, not
   an approximation (verified vs full-S attention, max abs diff 1.2e-7).
3. **Uniform slab-sized chunks.** Every chunk padded to `slab`, so ONE cached bias per window.
   Bias cache 23 tensors/53 MB → 5/21 MB, and stable across utterance lengths.
4. **Conv length bucketing**, `bucket=128`. Each distinct T otherwise compiled 5 new conv programs
   at 1–5 s each. On a stream of 12 distinct lengths: **120.9 s → 1.66 s (73x)**. Costs 7–25% on
   repeated identical lengths, which is the case production never sees.
5. **Hoisted conv weight preparation** (`prepare_conv_weights`). `ttnn.conv1d` was transforming and
   re-uploading weights *inside the op* on every call. **2.4x at short lengths** (112.8→43.7 ms at
   T=128); host share of wall time 88%→24%.
6. **Content-deduplicated prepared weights.** Only **8 distinct layouts** exist across 5 convs ×
   12 buckets. **730 MB → 98 MB** (0.8% of DRAM). Pure dedup, bit-identical, no accuracy question.

### Rejected after measuring
- **Device trace capture.** Works (conv1d *is* capturable once weights are pre-prepared) but
  measured **1.01x** — after #5 we are device-bound. Removed rather than kept; it added a stable
  input buffer, per-bucket trace lifecycle, and a device-wedging failure mode for no gain.
- **sdpa with native `sliding_window_size`.** Blocked: no ALiBi support, and folding ALiBi into an
  extra `k` column needs magnitudes to 42438 where bf16's spacing is 256 against a signal spanning
  192 — unrepresentable. Hence hand-rolled attention.
- **LRU eviction / bf16 / coarser buckets** for the memory ceiling — all dominated by dedup (#6).

---

## 5. Traps that cost real time — read these

1. **A failed trace capture wedges the device.** If an exception escapes between
   `begin_trace_capture` and `end_trace_capture`, `close_device` hangs and **every later run on the
   card blocks**. Cost ~20 min and required killing by PID. Always `end_trace_capture` in a
   `finally`. (No trace code remains, but this applies to Blocks 1/2 which will want tracing.)
2. **Prepared conv weights are NOT length-independent.** Same shape at every length, *different
   values* — cross-length reuse computes PCC **0.19**. I concluded "reusable" from matching shapes
   plus no crash. Shape equality and absence of a crash are not correctness.
3. **`prepare_conv_*`'s `input_dtype` is the ACTIVATION dtype**, not the weight dtype. Getting it
   backwards silently produced PCC 0.008.
4. **A stale benchmark outlived what it measured.** bf16 weights looked ~20% faster; that gap was
   conv weight-prep cost. After #5 they buy ~1% and still cost accuracy. Re-measure defaults after
   any change to the hot path.
5. **"Time is in the TTNN wrapper" ≠ "time is dispatch."** A profile showed 98% inside
   `ttnn/decorators.py:__call__` and I concluded tracing was the fix. It was work inside *one op*.
   Ask *which ops*, not just how much.
6. **Synthetic gates let the audible path rot.** Four numerical changes landed without re-running
   real speech, because that needed a 50 s backbone pass. Now pinned by a 6.4 KB fixture. If you
   add optimizations to Blocks 1/2, add the equivalent fixture test *first*.
7. **Comments outlive code.** Two review passes each found a comment describing behaviour from two
   commits earlier (the conv-upload comment; the "fp32 attention" docstring after the bf16 default
   landed). Re-read the docstring of anything whose defaults you change.
8. **`models/experimental` is NOT covered by the repo's black config** (pyproject `include` lists
   `models/demos`). Black flags 13/16 files; the sibling xtts_v2 code fails it too. Do not
   reformat. Do keep lines ≤ 120.
9. **PCC hides outliers.** It is a correlation — it can sit at 0.9999 while individual samples are
   badly wrong, and for audio the outliers are what you hear. The real-speech test also bounds the
   worst single sample at 2% of peak.

---

## 6. Open items

### Block 3 — two decisions, both waiting on the pipeline (not defects)
- **`bucket=128` is wrong for streaming.** Measured: a 1-second chunk costs the same 43.7 ms as a
  10-second one, because everything below 128 frames pads to 128. RTF degrades 10x. Streaming
  wants 16 or 32 (floor: 12.8 ms at bucket 16). One-line config change once the real chunk size is
  known. A **geometric** ladder (64,128,192,256,384,512,768,1024,1536) would beat the uniform grid
  — 9 buckets, and worst-case relative padding drops from 98% to ~50% — but tuning it needs a real
  utterance-length distribution.
- **batch=1 only.** Upstream serve concurrency 32. Not needed yet.
- Minor: the suite tests to T=469; T=1500 is measured by hand but not pinned.

### Block 1 — the next big piece
3.4B params, 26 layers, dim 3072, GQA 32/8, head_dim 128, SwiGLU 9216, RMSNorm, RoPE θ=1e6, tied
embeddings, **`n_heads*head_dim` (4096) ≠ `dim` (3072)** so wq/wo are not square.
- `tt_transformers` already supports `MistralForCausalLM`, so this is plausibly a config-add plus a
  **`params.json` → HF-config shim** (the checkpoint is Mistral-native: `consolidated.safetensors`
  + `params.json`, and the HF API reports an empty `config` field), not a hand-written port.
- **Decide before starting:** bf16 vs bfp8 weights, and single-N150 vs multi-chip. 3.4B at bf16 is
  ~6.8 GB before KV-cache; tight on a 12 GB card alongside Block 3's ~160 MB.
- RoPE must be **Mistral-native interleaved pairs**, not HF half-split. This is bit-exact verified
  in the reference — do not let a `tt_transformers` default silently switch it.
- Input side is already validated: `embed_frame` (37-codebook summed lookup) is bit-exact vs
  upstream, offsets included.

### Block 2 — smaller but more novel
390M, 3 layers, **3-token sequence**, bidirectional (no RoPE, no mask), 7 Euler steps per frame
with CFG batched to 2B. Fixed shapes and a fixed step count make it an ideal trace target —
upstream's own CUDA-graph version of exactly this gave them 47% latency / 2.5x RTF. Read trap #1
before writing trace code.

### Standing constraints (not fixable by us)
- Weights are **CC BY-NC 4.0**, non-commercial, including the reference voices. Same class of
  blocker as XTTS-v2's CPML. Needs legal sign-off before any product use.
- **Voice cloning from arbitrary audio is impossible** — the codec encoder is not in the release
  (0 of 386 tensors). Only the 20 shipped presets. A test asserts this so a future release that
  adds them fails loudly.

---

## 7. Suggested order when resuming

1. Re-read this file and `reference/PROVENANCE.md`. Recreate the two venvs (§2).
2. Run the 118 tests to confirm the environment before changing anything.
3. Block 1 via `tt_transformers` — settle the dtype/topology question first, then the config shim.
4. Block 2, with the 7-step ODE in a single trace.
5. `pipeline.py` end-to-end on device, then revisit Block 3's bucket for the real chunk size.

The end-to-end performance question is decided by Block 1: 87% of the parameters and 12.5
sequential steps per second of audio. Block 3 is done and should be left alone.
