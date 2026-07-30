# Voxtral-TTS on TTNN — status and resumption notes

**Read this first when picking the work back up.** It is written to be self-contained: state,
measurements, the traps that cost time, and what to do next. Architecture detail and the
reference-side findings live in `reference/PROVENANCE.md`; this file is the *work* state.

Branch: `lserbedzija/voxtral-tts-ttnn` (pushed). All work is under
`models/experimental/voxtral_tts/`. Nothing else in the repo is touched.

---

## 1. Where things stand

| Piece | State |
|---|---|
| CPU reference, 3 blocks + tokenizer + end-to-end pipeline | **done**, 30/30 vs upstream |
| Block 3 — codec decoder on TTNN | **CLOSED**, 242x real-time, see §4 |
| Block 1 — 3.4B AR backbone on TTNN | **done** — prefill + decode on `tt_transformers` |
| Block 2 — flow-matching transformer on TTNN | **done** — velocity PCC 0.9999989 |
| **End-to-end on device** (text ids + voice → 24 kHz wav) | **works**, 0.0% WER on natural text |
| Codec **encoder** | **impossible** — weights absent from the public release |

The model is 4 networks but only 3 are portable. 118 tests pass, 96 of them with no device
(the reference suite runs weight-free off a vendored tensor manifest).

**All three blocks now run on device and the pipeline generates real speech**, free-running, with
natural `[END_AUDIO]` termination on all 15 fixture prompts and 0.0% WER on every natural-language
one — including two 125-word paragraphs of 36–39 s. See §3. What remains is **performance**
(RTF ≈ 2.2–3.5, so ~2–3x slower than real time) and the licensing wall in §6.

**Block 3 is closed, not merely paused.** Six independent optimization attempts were measured and
all rejected (§4). The single remaining known win is worth ~0.1% end-to-end. Do not reopen it
without a new idea; read §4 and §5 first, because the obvious ideas are already disproven.

---

## 2. How to actually run things (do this before anything else)

There are **two** virtualenvs and neither is obvious.

**Main env — has ttnn, torch, torchaudio, transformers:**
```bash
source /localdev/lserbedzija/repos/xtts_ref_venv/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD      # from the repo root
```
This is the venv created for the XTTS-v2 work; the repo's own `python_env` is a docs env and does
**not** have torch/ttnn. `python_env` *does* have `black` if you need it (but see trap #8).

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

**End-to-end on device, and the quality numbers in §3.1:**
```bash
python models/experimental/voxtral_tts/scripts/export_backbone_hf.py --out /tmp/hf_backbone
export HF_MODEL=/tmp/hf_backbone                      # tt_transformers refuses to load without it
python models/experimental/voxtral_tts/scripts/generate_quality_set.py --out /tmp/vq
python models/experimental/voxtral_tts/scripts/score_quality_set.py /tmp/vq/results.json
```
All 15 fixture prompts take ~20 min on one N150. `--cases 0,1` for a quick check. The WAVs land in
`--out` and are the only way to actually *hear* the model — no metric substitutes for that.

**Opening a device for anything with convs** needs `l1_small_size`, or you get
`Out of Memory: ... bank size is 0 B`:
```python
dev = ttnn.open_device(device_id=0, l1_small_size=65536)   # as tests/test_codec_ttnn_pcc.py:31
```
Attention-only harnesses do not need it, which makes this an easy trap in a fresh script.

**Checkpoint** (8.0 GB, gitignored, CC BY-NC 4.0):
`reference/weights/` — `consolidated.safetensors`, `params.json`, `tekken.json`,
`voice_embedding/*.pt` (20 presets). Re-fetch per `reference/PROVENANCE.md`. Without it, 96
reference tests still pass; the 22 device/weight tests skip.

**Second env — upstream comparison only** (needs einops + mistral-inference, no GPU). Recipe and
rationale in `scripts/upstream_compare/README.md`. It lived in a session scratchpad and is **gone**;
recreate from the README. Also run `scripts/upstream_compare/fetch_upstream.py` to re-download the
two pinned vLLM-Omni files (gitignored on purpose).

**End-to-end reference speech** (~50 s of CPU for the reference backbone):
```bash
python models/experimental/voxtral_tts/reference/voxtral_pipeline_ref.py \
    --text "..." --voice neutral_male --threads $(nproc)
```

---

## 3. What is validated, and how

**Reference vs upstream: 30/30.** Harness in `scripts/upstream_compare/`.
- Block 1 vs `mistral_inference` (Mistral's own): RoPE table + application **bit-exact**,
  RMSNorm/SwiGLU/repeat_kv bit-exact, full 26-layer stack PCC 0.99999988.
- Block 1 input side + Blocks 2/3 vs vLLM-Omni's `nn.Module`s: 37-code frame **bit-identical
  integers**, waveform PCC 0.99999982, all 8 decoder stages >= 0.99999.
- Tokenizer vs `mistral_common`: **exact token ids**, 15 prompts, 8 languages. Ground truth
  vendored at `tests/prompt_fixture.json`.

**End-to-end reference:** 0.0% WER (Whisper) on 4 runs — 24 words x 2 voices, a **125-word
paragraph in a single pass** (469 frames, 37.5 s), and French via `fr_female`.

**Block 3 on device:** PCC 0.9994–0.9998 vs the reference across T=1…1536, plus a real-speech
fixture (`tests/real_frames_fixture.pt`, 6.4 KB of genuine Block 1+2 output) that asserts **both**
PCC > 0.9999 **and** worst sample < 2% of peak. Default config measures PCC 0.999984 / worst 1.16%.
That worst-sample bound is the gate that matters — see trap #9.

**Blocks 1 and 2 on device, per block:**

| | vs fp32 reference |
|---|---|
| Block 1 prefill (last position, real prompts) | PCC 0.99956–0.99958 |
| Block 1 decode | PCC 0.981 |
| Block 2 velocity | PCC 0.9999989; semantic codes exact, 73/74 frame codes exact |

### 3.1 End-to-end on device — the numbers that actually matter

Per-block PCC cannot tell you whether the audio is good, and neither can a teacher-forced code
comparison. These runs are **free-running**: the model is fed its own codes, which is what serving
does. Harness: `scripts/generate_quality_set.py` then `scripts/score_quality_set.py`.

- **Intelligibility — 1.17% WER over 341 natural-text words, and every deviation is a scoring
  artefact.** 12 of 12 natural-text cases: English (4, incl. two paragraphs), French, German,
  Italian, Portuguese, Hindi all exactly **0.0%**. The two non-zero ones are not TTS errors —
  Spanish 50% is the model correctly reading "42" aloud as "cuarenta y dos" against a reference
  containing the digits (3 phantom errors on a 6-word clip), and Arabic 33% is one grapheme
  (`حالك`→`حاله`) on a 3-word clip. This matches the reference's own 0.0% baseline, so
  **Block 1's bf16 weight floor costs nothing measurable in intelligibility**.
- **Long-form is the strong result.** The 676-char (125-word) paragraph: **0.0% WER at 459 frames
  / 36.7 s** (`neutral_male`) and **0.0% at 489 frames / 39.1 s** (`casual_female`). 459 steps of
  autoregressive decode with no drift, no repetition, no collapse. The reference generates the same
  paragraph in 469 frames / 37.5 s — within 2% of the device's duration, independently.
- **Termination: 15/15 prompts stop on `[END_AUDIO]` naturally.** Nothing runs away.
- **Artifacts: clean.** Across all 15 — 0.000% clipped samples everywhere, |DC offset| < 3e-4,
  0 clicks on 13/15 (2 on one German clip), peak 0.10–0.71 so nothing is near full scale.
- **Voice identity carries through all three blocks.** Fixture cases 0 and 2 are the same
  `neutral_male` preset reading *different* text, which is a positive control: their long-term
  spectra match at **0.984** cosine, the highest pair by a clear margin (next is 0.882), and median
  F0 separates cleanly — males 110.6 / 106.7 Hz vs females 205.1 / 192.0 Hz, with the two male
  readings agreeing within 4 Hz on unrelated text.

**Not measured:** a listening pass. Nobody has heard these clips; the artifact metrics catch
gross defects (clipping, clicks, DC, silence) but not prosody, naturalness or accent quality.
WAVs are written to the `--out` directory for exactly that.

Three fixture texts are deliberately adversarial — emoji, `!@#$%^&*()`, literal `\t`/`\n`. The
model tries to **vocalise** them (it renders `1234567890` correctly as "1 2 3 4 5 6 7 8 9 0", then
speaks the symbol run), so there is no well-defined reference transcript and they are reported
separately rather than folded into the headline WER.

**These are a model limitation, not a port defect — checked, not assumed.** Running the fp32 CPU
reference on the same two prompts: on the emoji text it collapses into a repetition loop
("emoji test kalinsan keep on faith i can say i can say i can say…", 6257% WER) where the device
at least keeps "and caps and mixed" (71%); on the symbol text both produce comparable nonsense
(reference 333%, device 450%). Duration agrees to within 2 frames on the symbol text (reference
242, device 240) and diverges on the emoji text (121 vs 64) — expected, since free-running
generation is chaotic and one differing code sends the two down different trajectories. **Do not
read the device being "better" on case 11 as a quality claim**; both outputs are unusable, and the
lesson is that the model is brittle on non-speech input. Sanitising or spelling out such text
belongs upstream of the model.

---

## 4. Block 3: performance, and everything that was tried

Warm, N150, defaults:

| T | audio | warm | RTF | vs real-time |
|---|---|---|---|---|
| 64 | 5.1 s | 43.9 ms | 0.0086 | 117x |
| 469 | 37.5 s | 155.5 ms | 0.0042 | 241x |
| 1500 | 120 s | 539.4 ms | 0.0045 | 222x |

Upstream report RTF **0.103 for their whole pipeline** on an H200, so Block 3 is ~4–8% of the total
budget. **It is not where the end-to-end answer gets decided** — Block 1 is (87% of the parameters,
12.5 sequential steps per second of audio). Do not over-invest here.

### Applied, with what each was worth
1. **bf16 attention** — best accuracy of four dtype configs *and* faster. Re-verified on the
   worst-sample metric, not just PCC (see below).
2. **Chunked windowed attention**, slab 512 — O(S²) → O(S·slab). At S=12000: warm 892→497 ms,
   cold 10580→1178 ms, mask 2304 MB→4.2 MB. **Exact**, not an approximation: chunked and unchunked
   give byte-identical worst-case error at all four stage lengths.
3. **Uniform slab-sized chunks** — one cached bias per window. Bias cache 23 tensors/53 MB →
   6/18.1 MB, and it stops growing with utterance length.
4. **Conv length bucketing**, 128 — on a stream of 12 distinct lengths, 120.9 s → 1.66 s (73x).
5. **Hoisted conv weight preparation** — 2.6x at T=128 (112.8→43.7 ms); host share 88%→24%.
6. **Content-deduplicated prepared weights** — 730 MB → 98 MB, bit-identical.

### Measured and REJECTED — do not retry without new information
| idea | result |
|---|---|
| **sdpa** for the attention interior | 1.44–2.27x FASTER, 3.3x worse worst-case error, failed 11 tests. See §4.1 |
| **Smaller slab** toward the compute optimum 2·window | slab=32 computes **9x fewer** scores and runs **9x slower** (334 vs 36 ms/pass) |
| **Device trace capture** | **1.00x** at every slab size, including 3570 ops. The async command queue already hides host dispatch behind device execution |
| **Batching the chunks** into one matmul | batched attention is 2.4x faster, but the gather costs more than it saves (11.30 vs 9.67 ms). Chunks overlap by `window`, and ttnn has no strided view, so the copy is unavoidable |
| **Unchunked attention** with a full [S,S] mask | identical accuracy, 3x slower at S=4096, mask grows to 268 MB. Only S<=1024 prefers it — a `chunk_min` question |
| **Changing the dtype default** | the current default already wins on *both* PCC and worst-sample |

The recurring lesson: **on this hardware, doing less arithmetic does not mean finishing sooner.**
Three of these reduce FLOPs and all three are slower. Cost is dominated by per-kernel overhead and
memory traffic.

### 4.1 The sdpa investigation (the deepest hole; read before touching attention)

sdpa **is** faster and its API **does** support ALiBi (via `attn_mask` with `is_causal=False`). The
old note claiming "no ALiBi support" was wrong. It is rejected on **accuracy**: against an exact
fp64 answer, worst-case error 1.95e-02 vs hand-rolled 5.85e-03.

Ruled out as the cause, each by measurement:
- our ALiBi mask — an **all-zero** mask still gives 4.7x (1.135e-03 vs 2.395e-04)
- `MASK_NEG` magnitude — identical to 4 digits from -1e4 to -1e30
- ALiBi precision — per-head error is uniform, *anti*-correlated with slope
- online-softmax rescaling — 1 k-block vs 4 k-blocks: identical
- `exp_approx_mode`, `math_fidelity` HiFi3/HiFi4, `fp32_dest_acc_en`
- legacy v1 vs streaming v2 kernel (`fp32_dest_acc_en` selects between them!) — 2.066e-02 vs 2.163e-02
- the fp32→bf16 input conversion — costs 3.70e-03, identically for both paths
- **patching `sdpa_program_factory.cpp` so `im_df`/`stats_df` are Float32, and rebuilding** —
  marker-verified live (`im_df=Float32`), error unchanged. So closed issue #13364 is a red herring.

Established: a **mask-independent precision floor intrinsic to the fused kernel**, 3.7–10.7x worse
worst-case. Telling detail: hand-rolled error *falls* with S (4.57e-04→1.27e-04 for S=128→2048)
because softmax averages more values, while sdpa stays flat (1.7e-03→1.4e-03). sdpa has a
fixed-magnitude error term that does not benefit from averaging. Numerical attribution matches
"bf16 accumulation of P@V" (6.8e-04) rather than "bf16 scores+probs" (1.1e-04, which is what
hand-rolled's 2.4e-04 is consistent with).

**Where it stopped:** the one-k-block result rules out repeated cross-block accumulation, so it is a
single fixed-precision step inside the fused pipeline. Pinning it further needs instrumenting
`kernels/compute/sdpa.cpp` to dump intermediates. Not done.

**This matters for Block 1** — see §7.

---

## 5. Traps that cost real time — read these

1. **A failed trace capture wedges the device.** If an exception escapes between
   `begin_trace_capture` and `end_trace_capture`, `close_device` hangs and **every later run on the
   card blocks**. Cost ~20 min and required killing by PID. Always `end_trace_capture` in a
   `finally`. (No trace code remains in Block 3, but Blocks 1/2 will want tracing.)
2. **Prepared conv weights are NOT length-independent.** Same shape at every length, *different
   values* — cross-length reuse computes PCC **0.19**. Shape equality and absence of a crash are not
   correctness. The cause: ttnn lowers conv to an implicit GEMM whose weight operand is
   `[C_in·K, C_out]` (no L in it, hence the constant shape), but the sharding/blocking plan is
   chosen from the GEMM's row count `N·L_out`, and the weight is pre-arranged to match that plan.
3. **`prepare_conv_*`'s `input_dtype` is the ACTIVATION dtype**, not the weight dtype. Getting it
   backwards silently produced PCC 0.008.
4. **A stale benchmark outlived what it measured.** bf16 weights looked ~20% faster; that gap was
   conv weight-prep cost. Re-measure defaults after any change to the hot path.
5. **"Time is in the TTNN wrapper" != "time is dispatch."** A profile showed 98% inside
   `ttnn/decorators.py:__call__` and I concluded tracing was the fix. It was work inside *one op*.
   Tracing later measured 1.00x. Ask *which ops*, not just how much.
6. **Synthetic gates let the audible path rot.** Four numerical changes landed without re-running
   real speech. Now pinned by a 6.4 KB fixture. If you optimize Blocks 1/2, add the equivalent
   fixture test *first*.
7. **Comments outlive code.** Every review pass has found one: the conv-upload comment, an
   "fp32 attention" docstring after bf16 became default, a wrong sdpa rationale, and the same
   optimization quoted as both 2.4x and 2.6x. Re-read the docstring of anything you change.
8. **`models/experimental` is NOT covered by the repo's black config** (pyproject `include` lists
   `models/demos`). Black flags 13/16 files; the sibling xtts_v2 code fails it too. **Do not
   reformat.** Do keep lines <= 120.
9. **PCC HIDES OUTLIERS — the single most expensive lesson.** It is a correlation: it can sit at
   0.9998 while individual samples are badly wrong, and for audio the outliers are what you hear.
   sdpa passed at PCC 0.9998 per slab and still failed 11 tests, one stage measuring PCC 0.916.
   Always pair PCC with a worst-sample bound.
10. **Verifying a C++ rebuild is harder than it looks.** Three separate traps in one experiment:
    - `nohup cmd &` inside a backgrounded tool call gets reaped when the wrapper exits; the harness
      reports the *wrapper's* exit code, not the build's.
    - `ldd` resolved `_ttnncpp.so` from `build_Release/lib/`, a **stale separate copy**, not the
      freshly linked `build_Release/ttnn/` one. `LD_LIBRARY_PATH` fixes it (it is RUNPATH, so
      LD_LIBRARY_PATH wins).
    - **A relinked `.so` with a fresh timestamp is NOT evidence your source change is in it.** The
      first build reported exit 0, relinked both `.so`s, and had never recompiled the edited file
      (it only rebuilt `tt_metal/fabric`). Only a runtime `log_warning` marker proved it. Note
      `log_debug` is compiled out in Release, so it cannot be used for this.
11. **`fp32_dest_acc_en` selects the sdpa kernel variant**, it is not just a precision knob:
    `can_use_streaming_compute(fp32_dest_acc_en) { return !fp32_dest_acc_en; }`. Passing `True`
    forces the legacy v1 kernel. Holding it `True` in every test meant streaming v2 was never
    exercised for a long time.

12. **RANDOM ACTIVATIONS ARE A PESSIMISTIC PROXY FOR WEIGHT PRECISION — this one cost the most
    time on Block 1.** Feeding `randn * 0.02` embeddings through 26 layers reported PCC 0.892 for
    BFP8 MLP weights vs 0.969 for bf16, which looked like a crisis and drove a long precision hunt.
    On **real tokenized prompts** the same two configs score 0.99938 and 0.99970 — the entire
    spread is 3e-4, and every variant clears 0.999. Random embeddings are off-manifold, so
    activations do not sit in the range the weights were trained for, and block-float (one shared
    exponent per block) is punished hardest by an inflated dynamic range. Never quote a
    low-precision-vs-fp32 PCC from synthetic activations; use `tests/prompt_fixture.json`.

13. **A too-small frame budget is indistinguishable from a model that will not terminate.** Frames
    are 12.5 Hz and speech runs ~18 chars/s, so the 676-char fixture paragraphs need ~460 frames.
    Run them at `max_frames=200` and they stop mid-sentence and report "no natural stop" — which
    reads as a generation bug and sent me looking for one in the KV-cache padding. It also
    correlated with prefill length purely because the long texts are the long prompts.
    `generate_quality_set.py` derives the budget from text length for this reason.

14. **Three ASR-harness traps, all of which fake a bad TTS result.** Whisper's encoder is a fixed
    30 s window, so a plain `generate` call on a 37 s clip silently transcribes the first 30 s and
    charges the rest as deletions (~20% phantom WER); an `[^a-z0-9' ]` normaliser erases Hindi and
    Arabic and scores them 100% on perfect audio; and a voice-name prefix is not a language —
    fixture case 4 is English "Hello." spoken by `ar_male`, and forcing Arabic decoding made
    Whisper hallucinate a filler and report 100% WER on one word. All three are handled and
    documented in `scripts/score_quality_set.py`.

---

## 6. Open items

### Block 3 — closed, with two deferred decisions (not defects)
- **`chunk_min` should probably be 1024, not 512.** Measured crossover is S ~ 2000: at S=1024
  unchunked is 1.5x faster (1.40 vs 2.10 ms). Worth 1.06x on attention, ~1.7% of the block,
  **~0.1% end-to-end**, and costs a 16.8 MB bias. Recorded in the code; not worth a commit alone.
- **`bucket=128` is wrong for streaming.** A 1-second chunk costs the same 43.7 ms as a 10-second
  one, because everything below 128 frames pads to 128. Streaming wants 16 or 32 (floor: 12.8 ms at
  bucket 16). A **geometric** ladder (64,128,192,256,384,512,768,1024,1536) would beat the uniform
  grid — worst-case relative padding drops from 98% to ~50% — but tuning it needs a real
  utterance-length distribution.
- **Streaming needs boundary state throughout**, not just a bucket change. Three places fabricate
  or discard context that a continuing stream would supply: the transposed convs' trimmed tail, the
  forward convs' `_pad_causal` (which would replicate the chunk's own first frame instead of the
  previous chunk's real tail), and attention's first `window` rows. Ordinary work, but list it.
- **batch=1 only.** Upstream serve concurrency is 32. Not needed yet.
- **bf16 weights + bf16 attention has almost no margin**: 1.93% worst-sample against a 2.00% gate,
  for a ~1% speed gain. The knob exists for experiments; do not ship it.
- Minor: the suite tests to T=469; T=1500 is measured by hand but not pinned.

### Block 1 — DONE, on `tt_transformers`
3.4B params, 26 layers, dim 3072, GQA 32/8, head_dim 128, SwiGLU 9216, RMSNorm, RoPE θ=1e6, tied
embeddings, **`n_heads*head_dim` (4096) != `dim` (3072)** so wq/wo are not square.

Implemented in `tt/ttnn_voxtral_backbone.py` on top of `models/tt_transformers`, fed by an HF-format
export (`scripts/export_backbone_hf.py`). Prefill + decode share one KV cache. Read that module's
docstring before touching precision — it carries the measured table and the L1 constraint that
forces FF1_FF3 to BFP8.

**Still open on Block 1:**
- **It dominates runtime.** ~0.17 s per decode step is essentially all of the frame budget, so it
  is where the RTF fix has to come from. Nothing here is traced yet.
- Decode PCC 0.981 is well below prefill's 0.9996. It does not hurt WER, but it is unexplained and
  worth a look before trusting Block 1 in a different configuration.
- Prefill is capped near 512 by L1 (see the module docstring). Prompts beyond ~1024 tokens would
  need real chunked prefill, which needs paged attention — `MAX_PREFILL_CHUNK_SIZE` cannot help
  because its unit is 1024.

**Risks investigated and CLEARED before the port** (all in `models/tt_transformers/tt/`):
- `head_dim=128` is honoured: `model_config.py:2678` reads it from config, falling back to
  `dim // n_heads` (which would give a wrong 96).
- The non-square `wo` is handled: `model_config.py:1983` uses `k_dim = n_heads * head_dim`.
- **`forward()` takes embeddings, not token ids** (`model.py:852`, `x: ttnn.Tensor`), so our
  37-codebook summed input works — `embedding.py` is a separate module we simply bypass.
- `lm_head` is skippable, and we *should* skip it: Block 2 consumes the post-final-norm hidden
  state (`voxtral_backbone_ref.py:157`), and we never emit text tokens. That is 402M tied
  parameters (131072x3072) we do not pay for.
- RoPE convention: `use_hf_rope` defaults to `False` (Meta/interleaved), which matches our
  bit-exact reference. Do not let a default flip it to HF half-split.

**Constraint found and now handled by `scripts/export_backbone_hf.py`:** `tt_transformers`
**requires HF-format checkpoints** —
`model_config.py:588` raises unless `HF_MODEL` is set, and the Meta `consolidated.00.pth` path is
vestigial. Ours is Mistral-native `consolidated.safetensors` + `params.json`. So the shim is bigger
than a config-add: a `config.json` **plus a checkpoint transcription to HF parameter names**.
Mechanical and CPU-testable. `model_params/Mistral-7B-Instruct-v0.3/config.json` is the schema to
copy; the field mapping is dim→hidden_size, n_layers→num_hidden_layers, n_heads→num_attention_heads,
n_kv_heads→num_key_value_heads, hidden_dim→intermediate_size, norm_eps→rms_norm_eps,
tied_embeddings→tie_word_embeddings, plus an explicit `head_dim: 128`.

**Topology is settled:** this box has exactly one device (WORMHOLE_B0, 1 visible, 1 PCIe), so
single-N150. Memory: 3.03B params in the 26 layers + 402M embedding = 3.43B →
**bf16 6.86 GB, bfp8 3.64 GB**, plus ~218 MB KV cache at seq 2048 and ~260 MB for Block 3, against
~11 GB usable. bf16 fits with headroom; start there and only drop to bfp8 if speed demands it.

### Block 2 — DONE, and the accuracy is not the problem
390M, 3 layers, **3-token sequence**, bidirectional (no RoPE, no mask), 7 Euler steps per frame with
CFG batched to 2B. Implemented in `tt/ttnn_voxtral_flow.py` at fp32/HiFi4: velocity PCC 0.9999989,
semantic codes exact. Semantic argmax and FSQ quantise stay on host (see that module's docstring).

**Still open:** the 7-step ODE is **untraced**, and it is the clearest remaining perf win. Fixed
shapes and a fixed step count make it an ideal trace target — upstream's own CUDA-graph version of
exactly this gave them 47% latency / 2.5x RTF. Block 3's tracing null result does **not** transfer:
that was a long chain of ~30 us device-bound ops, whereas Block 2's ops are tiny, which is exactly
the regime where host dispatch becomes exposed. Read trap #1 before writing trace code.

### Standing constraints (not fixable by us)
- Weights are **CC BY-NC 4.0**, non-commercial, including the reference voices. Same class of
  blocker as XTTS-v2's CPML. Needs legal sign-off before any product use.
- **Voice cloning from arbitrary audio is impossible** — the codec encoder is not in the release
  (0 of 386 tensors). Only the 20 shipped presets. A test asserts this so a future release that
  adds them fails loudly.

---

## 7. Suggested order when resuming

**All three blocks work and the audio is good (§3.1). The open work is performance, and the order
is set by where the time actually goes: Block 1's decode step is ~0.17 s and Block 2's untraced
7-step ODE is next; Block 3 is 242x real-time and irrelevant to RTF.**

1. Re-read this file and `reference/PROVENANCE.md`. Recreate the two venvs (§2).
2. Run the 118 tests, then `generate_quality_set.py --cases 0,1` to confirm the device path still
   produces speech before changing anything.
3. **Trace Block 2's 7-step ODE.** Fixed shapes, fixed step count, tiny ops — the regime where
   dispatch is exposed. Upstream's CUDA-graph version of exactly this was worth 47% latency. Read
   trap #1 first: a failed trace capture wedges the device.
4. **Attack Block 1's decode step**, which is essentially the whole frame budget. Start by finding
   out where the 0.17 s goes (device vs host dispatch) before optimizing anything —
   `tt_transformers` has its own trace/prefetcher machinery that we are not using.
5. Re-run §3.1's harness after each change. Trap #6 is that synthetic gates let the audible path
   rot; WER is the gate that matters.

**Deferred and still worth doing:**
- A **listening pass**. Never done. Everything in §3.1 is objective metrics and ASR.
- Block 1 **decode PCC 0.981** vs prefill's 0.9996 — unexplained, harmless so far.
- **The sdpa compounding probe** — ~1 hour, needs no checkpoint. Block 1 currently does *not* use
  sdpa, so this is a latent perf lever rather than a blocker. Block 1 escapes sdpa's *minor*
  problem (the `attn_mask`/`is_causal` exclusivity: its attention is causal + RoPE with no additive
  bias, so the native fast path works and no mask tensor is needed) but would **inherit the major
  one** — the mask-independent 3.7–10.7x worst-case penalty in §4.1, proven with an all-zero mask.
  Chain 26 sdpa attentions at Block 1's real shapes (GQA 32/8, head_dim 128, dim 3072, causal,
  seq ~500) against an fp64 reference and report PCC **and worst-sample per layer**, to see whether
  the error compounds or stays bounded. Cover **both** kernels — `sdpa` for prefill and
  `sdpa_decode` for decode are separate code paths, and the XTTS-v2 work hit an odd-tile bug in the
  latter needing the KV-cache length rounded to a multiple of 64. If it comes back unbounded, that
  is worth escalating: it would affect every model in the repo.
- **Prefill beyond ~1024 tokens** needs paged-attention chunked prefill (§6, Block 1).
- Block 3's `bucket` for the real streaming chunk size (§6).

The performance question is decided by Block 1: 87% of the parameters and 12.5 sequential steps per
second of audio. Block 3 is done and should be left alone.
