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
| Block 1 — 3.4B AR backbone on TTNN | **done — OURS** (`tt/ttnn_voxtral_gpt.py`), the default |
| Block 2 — flow-matching transformer on TTNN | **done** — velocity PCC 0.9999989 |
| **End-to-end on device** (text ids + voice → 24 kHz wav) | **works**, 0.88% WER on natural text |
| Codec **encoder** | **impossible** — weights absent from the public release |

**Block 1 now runs on our own implementation, not `tt_transformers`.** The wrapper is DELETED, not
parked behind a flag -- `git show d3dcb0fb7c6^:models/experimental/voxtral_tts/tt/ttnn_voxtral_backbone.py`
if you ever need it back for bisection (it also needs `scripts/export_backbone_hf.py` from the same
commit, plus `HF_MODEL`). Measured against the fp32 CPU reference on real prompts, and end to end
on the 15-case fixture:

| | ours | `tt_transformers` |
|---|---|---|
| prefill, last position | 0.999881 | 0.999564 |
| decode step | 0.99991 | 0.981 |
| decode ms/frame | 34.9 | 48 |
| natural-text WER | 0.88% | 1.17% |

**Performance: 83.7 → 49.0-52.5 ms/frame (mean 50.4), RTF 0.62-0.71 on 14 of 15 cases.** Per frame:
Block 1 ~25.7 ms, Block 2 ~23.0 ms, host 0.2 ms. Block 1 is marginally the larger half.
goes" map at the top with the ceiling for each line item. Two sweeps got here — §6.6 (GQA row fold,
width-sharded RMSNorm in Block 2, qkv fusion) and §6.8 (BFP8 on wqkv/wo, semantic head on device) —
and §6.7 is the one to read first: it explains why the WER headline cannot gate any of this.

**Shipped configuration**, pinned by `tests/test_tt_defaults.py`:
Block 1 mixed precision (BFP8 on FF1_FF3 only) + decode-native heads; Block 2 BFP8 weights at
HiFi4 + fp32 accumulation; no device traces (measured, removed -- see §6); no program-cache
clearing. There are no runtime toggles: every rejected alternative was deleted.

### THE ONE THING THAT WILL WASTE YOUR TIME: fixture case 4

Case 4 is "Hello." — one word of text on 74 prompt tokens, almost all voice conditioning. **The
MODEL is chaotic on it**, and that is not a port defect. Free-running frame counts, 9 correct:

    fp32 CPU reference (no device, no precision loss)   81 /  8 / 57
    tt_transformers + bf16 flow                          9 /  8 / 72 / 88 / 55 / 118
    ours, four different weight configurations          8 to 183

Every implementation, including pure torch, lands anywhere from 8 to 183 frames on the noise draw
alone. The likely mechanism is that one word cannot compete with 74 tokens of voice conditioning
at the last prefill position; the `ssinghal/voxtral_tts` branch reports the same effect
independently ("147 voice frames dilute text signal"). At 1 reference word a rambling take scores
tens of thousands of percent and swamps 340 good words, so `score_quality_set.py` now reports it
in a separate **model-unstable** bucket. It cost real debugging time TWICE — do not read it as a
regression, and do not tune precision against it.

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
python models/experimental/voxtral_tts/scripts/generate_quality_set.py
# No HF export and no HF_MODEL: Block 1 is ours and loads the Mistral-native checkpoint directly.
# Those were only ever needed by the deleted tt_transformers wrapper (see §1).
python models/experimental/voxtral_tts/scripts/score_quality_set.py \
  models/experimental/voxtral_tts/generated/results.json
```
All 15 fixture prompts take ~20 min on one N150. `--cases 0,1` for a quick check. The WAVs land in
**`generated/`** (gitignored — audio is large and CC BY-NC derived, so it never leaves the box) and
are the only way to actually *hear* the model. No metric substitutes for that.

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
| Block 1 prefill (last position, real prompts) | PCC 0.999881 |
| Block 1 decode | PCC 0.99991 |
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

**Listening pass: done, informally — verdict "sounds ok".** The long-form clips in `generated/`
were listened to by the author (2026-07-30). That clears the bar of "no audible defect the metrics
missed", which is what it was for. Read it as exactly that and no further: **it is not evidence of
listener-facing naturalness.** WER measures intelligibility, and one developer saying "ok" is not a
substitute for a MOS-style eval with real raters and a side-by-side against the fp32 reference
(`generated/ref_current.wav` and the `case11_*_REF.wav` pair are there for that). If someone needs
to claim naturalness — for a customer, a demo, or a comparison against another vendor — that eval
has not been done.

### 3.2 bf16 in Block 2 — what was checked before flipping the default

Block 2's output is 36 integers rounded onto 21 FSQ levels, so PCC is the wrong gate: what matters
is how often a value crosses a rounding boundary, and whether that compounds over hundreds of
autoregressive steps. All 15 prompts were regenerated at bf16 and rescored.

| | fp32 | bf16 |
|---|---|---|
| acoustic codes differing from the reference | 0.81% | 1.16% (all off-by-one) |
| semantic codes wrong (12 seeds, batch 2) | 0/24 | 0/24 |
| **natural-text WER, 341 words** | **1.17%** | **1.17%** |
| natural `[END_AUDIO]` termination | 15/15 | 15/15 |
| 125-word paragraph, free-running | 459 / 489 frames | **458 / 490 frames** |
| voice identity (same-voice vs next-nearest) | 0.984 / 0.882 | 0.985 / 0.884 |

The WER total is unchanged and its *composition* moved: German and Hindi each gained one word-error,
Spanish lost three. All single-token, in both directions, on 3–7 word clips where one word is
14–33% WER. The Spanish case shows it is divergence and not degradation — bf16 read "42" as digits
where fp32 spelled out "cuarenta y dos", and bf16 only scores better because the reference text
contains digits.

Two things were checked and one was a false alarm. Across 12 matched natural-speech pairs the output
is **~2% quieter** (median, ~0.2 dB, inaudible; negative in 11/12) and **F0 is unchanged** (median
−0.25%, negative in only 6/12). An apparent downward F0 bias seen on the first 3 pairs did **not**
survive the larger sample — the two large positives came from 0.7–2.9 s clips where a median-F0
estimator is unreliable. Spectral similarity between the fp32 and bf16 renderings is 0.9875–0.9992
on 11 of 12, i.e. tighter than the 0.984 that two utterances of the *same voice* score against each
other.

**Not covered:** nobody has A/B listened to fp32 vs bf16. Matched pairs are in `generated/` as
`caseN_<voice>.wav` against `caseN_<voice>_bf16flow.wav`. Level-match first — the 2% median is
irrelevant but cases 0, 1 and 5 differ by 11–16%, enough to bias a casual comparison.

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

### Block 1 — DONE, ours (`tt/ttnn_voxtral_gpt.py`)

3.4B params, 26 layers, dim 3072, GQA 32/8, head_dim 128, SwiGLU 9216, RMSNorm, RoPE θ=1e6, tied
embeddings, **`n_heads*head_dim` (4096) != `dim` (3072)** so wq/wo are not square. Loads the
Mistral-native checkpoint directly — no HF export, no `HF_MODEL`, no RoPE-permute hazard.

Beats the `tt_transformers` wrapper it replaced on every metric: prefill last-position PCC
0.999881 vs 0.999564, decode 0.99991 vs 0.981, 34.9 ms/frame vs 48, natural-text WER 0.88% vs
1.17%. Read the module's "where the time goes" map before optimizing — it lists the ceiling for
each line item, not just the cost.

**What is left in Block 1, honestly:** not much. 23.6 of its 34.9 ms is weight streaming at
194 GB/s, which is the measured ceiling for a plain interleaved matmul here (hand-tuned matmul
program configs are SLOWER — 169 vs 193 GB/s on wq). 6 ms is RMSNorm whose fp32 accumulation is
load-bearing. So only fewer weight BYTES help, and that is capped by the hang below.

**THE HANG, and the precision boundary it imposes.** All-BFP8 weights are ~5 ms/frame faster and
trigger a silent, card-wedging hang in multi-utterance runs (recovery needs a `tt-smi` board
reset). It needs five things at once — all-BFP8 Block 1, Block 2 in the loop, Block 3 on device,
two distinct codec buckets, and a generation between two same-bucket decodes. Remove any one and
it completes. **BFP8 on FF1_FF3 is safe; adding FF2 brings it back.** `tt_transformers` never saw
it because it uses the same mixed precision. Ruled out by measurement, so do not retry: memory
(flat, 8 GB free at the hang), program-cache COUNT (576 entries fine; we died at 310–341 while
tt_transformers lived at 329), Block 3 length/content, a Block 1 leak (1400 steps clean), and
every distinctive Block 1 op. The underlying ttnn failure — a hang rather than an error — is
unreported upstream and still unexplained.

**Device trace** of the decode step is implemented and correct (PCC 0.99982 traced vs 0.99985)
and OFF: at equal window it is 0.7 ms SLOWER, because decode here is device-bound, not
host-dispatch bound. Tracing Block 1 AND Block 2 together also works now and is bit-identical
(0 differing codes of 2849) after fixing the capture ORDER — all warm-ups before any capture —
but costs ~6 ms/frame, so both stay off. The `ign/voxtral_p150_qb2` branch reports decode as
dispatch-bound; that is Blackhole, and it does not transfer to this N150 — see §6.5.

### 6.5 — `ign/voxtral_p150_qb2`, measured

Their code will NOT run on our tree (`TT_THROW: Only L1 buffers can have an associated circular
buffer!`; they sit ~1100 commits back). So their tt-metal was built from their own commit into
`/localdev/lserbedzija/ign_build`, with a separate python env at `/localdev/lserbedzija/ign_venv`
that layers our site-packages WITHOUT our ttnn editable-install finder — that finder is a meta-path
hook and wins over PYTHONPATH, which is why the obvious `PYTHONPATH=` approach silently kept using
our ttnn. Their build runs fine on this Wormhole N150 (8x8 grid).

    pytest models/experimental/voxtraltts/tests/perf/test_e2e_performant.py -q -s -k F128
    HF_MODEL=<our weights dir>   # their loader takes the same Mistral-native layout

**Result: 598.1 ms/frame, 1.67 frames/s over 128 decode steps** (build 24 s, warm-up 121 s), against
our 77.6. No spills, no host fallbacks, trace+1CQ, weights all-BFP8.

READ THAT WITH THE CAVEAT IT DESERVES. Their branch targets a Blackhole P150 — bigger grid, more
DRAM, and a config tuned for it. This measures THEIR CODE ON OUR CARD, which answers "could we just
adopt theirs on N150" (no) and NOT "is their P150 work slow" (unmeasured; we have no P150).

Two things it does settle. Their own comment calls the acoustic FM core "the 78%/step bottleneck",
which independently reproduces our Block 2 finding on a different implementation. And their Block 1
runs ALL-BFP8 — the exact config that hangs for us — without hanging, which is more evidence the
hang is a tt-metal-version property rather than something about our weights.

**Still open:** prefill beyond ~1024 tokens would need chunked prefill. batch=1 only.

### Block 2 — DONE, and the accuracy is not the problem
390M, 3 layers, **3-token sequence**, bidirectional (no RoPE, no mask), 7 Euler steps per frame with
CFG batched to 2B. Implemented in `tt/ttnn_voxtral_flow.py` at fp32/HiFi4: velocity PCC 0.9999989,
semantic codes exact. Semantic argmax and FSQ quantise stay on host (see that module's docstring).

**THE BIG ONE — a batched matmul re-reads the whole weight per batch element.** CFG makes the trunk
a batch-2 forward, and that was doubling every weight read for nothing. Measured on
`[*,3,3072] @ [3072,9216]`:

| activation | ms | effective |
|---|---|---|
| `[1,3,K]` | 0.294 | 192 GB/s |
| `[2,3,K]` (CFG batch 2) | 0.581 | 97 GB/s |
| `[4,3,K]` (CFG batch 4) | 1.154 | 49 GB/s |
| **`[1,6,K]` (batch 2 FOLDED into rows)** | **0.296** | **191 GB/s** |
| `[1,32,K]` (full tile row) | 0.294 | 193 GB/s |

Cost scales linearly with batch; folding the batch into the ROW dimension is **free**, because 6 rows
still fit one 32-row tile. A linear applies per row independently, so `_trunk` now runs the whole
trunk as `[1, B*3, 3072]` and only splits the batch out for attention. **Worth 2.23x on device time
(136.9 -> 61.3 ms) with BIT-IDENTICAL output** — verified against the previous build's WAVs on three
prompts including a 458-frame clip. This is what the "13% of DRAM peak" figure actually was: a single
matmul reaches 66% of peak, and CFG batching was throwing half of it away.

**Corollary — ~26 of 32 tile rows are still free, and that is throughput for nothing.** `[1,32,K]`
costs the same as `[1,6,K]` (0.294 vs 0.293 ms), and the fold uses only 6. Nothing in a single
utterance can use the rest: the Euler steps are sequential and frames are autoregressive. But
CONCURRENT REQUESTS fit exactly — 5 simultaneous utterances fold to 30 rows at essentially the same
per-frame cost, i.e. ~5x throughput free. Latency per stream would not improve; only throughput.

Running the two CFG branches as separate passes instead of folding was measured and is strictly
worse: 2 passes cost 0.586 ms against 0.293 folded, exactly 2x. Passing the SAME weight tensor to
both costs the same as two different weights (0.586 vs 0.587), which shows there is no cross-op
weight reuse -- every `ttnn.linear` streams its weights from DRAM independently. There is also no
parallelism to win on one device: a single matmul already spans the core grid and both branches want
the same bytes from the same DRAM. On two devices it would be a different question.

It also makes bfp8 weights unnecessary: bf16 **with** the fold (61.3 ms) beats bfp8 **without** it
(98.3 ms). `WEIGHT_DTYPE` stays `None`; bfp8 remains measured and available but costs accuracy for
less than the free fix delivers.

Also applied: **k and v share one weight and one matmul**, split by `nlp_create_qkv_heads` which
builds the head layout in the same op — 3 linears + 6 reshape/permutes become 2 linears + 1 op.
Worth 1.6%, numerically identical. The old comment claiming the fused op "needs a batch layout this
does not have" was simply wrong.

**DRAM-sharded weights — available, but the ceiling is ~1.5x and it is real plumbing.**
`create_dram_sharded_mem_config` + `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, as
`tt_transformers` uses for decode. Probed at Block 2's real shape and it needs THREE things to line
up, each a hard TT_FATAL, found one at a time:
1. the ACTIVATION must be L1 WIDTH_SHARDED (`input_tensor_a.is_sharded()`, matmul_device_operation
   .cpp:1226ff), not just the weights;
2. the shard grid must be **TENSIX** cores, whose grid is 8x8 — using the 12-wide DRAM bank grid
   fails with "No core coordinate found at location: (8, 0, TENSIX, LOGICAL)". So the activation
   shards over at most 8 cores here (3072/8 = 384, tile-aligned), not 12;
3. the OUTPUT memory config must be sharded as well.
4. **`M` must be exactly ONE tile** — `TT_FATAL(M == 1, "currently only support in0 tensor height of
   tile height")`, matmul_device_operation.cpp:1245. Our folded trunk qualifies (6 rows in one tile);
   anything taller does not, so `M=64` is categorically unsupported.

With all four satisfied at `M=32`, it still fails to build: **circular buffers grow to 1,678,528 B
against L1's 1,499,136 B**, ~175 KB over. With only 8 compute cores and N=9216, `per_core_N` is 36
tiles and the streaming buffers plus the output shard do not fit. Shrinking `per_core_N` means more
cores — `K/cores` must stay tile-aligned so cores must divide 96, and 48 would give `per_core_N=6` —
but the weight shard lives on the **DRAM** grid (12 banks, y=1) while the activation lives on a
**Tensix** grid, so the two grids cannot both be 48, and that is exactly the `per_core_N` vs DRAM
shard width mismatch model_config.py:716 warns gives **silently bad PCC**.

So this needs a deliberate design pass reconciling three grids (DRAM banks, compute cores,
`per_core_N`) under an L1 budget, with the integer-code gate wired in from the first run. Not a
drop-in.

**The ceiling is modest.** After the batch fold, a plain interleaved matmul at this shape already
runs at **192 GB/s, ~66% of peak**, so perfect sharding is worth at most ~1.5x on the matmuls, and
less end to end. Block 1 is the larger target now (~48 ms of an ~80 ms frame). Do this as deliberate
work with a numerical gate, not as a quick win.

**Measured and REJECTED — sdpa in the attention interior.** `ttnn.transformer.scaled_dot_product_
attention` fuses 7 ops into 1 there (2x `repeat_interleave`, transpose, matmul, scale, softmax,
matmul) and supports GQA natively, so k/v go in unexpanded and the 4x expansion disappears. It
works and it is faster, but not for free:

| | hand-rolled | sdpa |
|---|---|---|
| frame, batch 2 | 145.5 ms | 122.3 ms (**1.19x**) |
| velocity PCC (worst of 12 seeds) | 0.9999428 | 0.9990333 |
| acoustic codes differing from reference | 1.16% | **4.28%** |
| max deviation | 1 FSQ level | **2 FSQ levels** |
| frames exactly right | 6/12 | 3/12 |

**3.7x more differing codes, and that factor is not a coincidence** — §4.1 measured sdpa's
mask-independent worst-case penalty in Block 3 at 3.7–10.7x. The same characteristic reproduced here
at the bottom of that range, which says the penalty travels with the op rather than with the shapes.
Rejected on sequencing, not on principle: tracing attacks the same dispatch bottleneck for **zero**
accuracy cost, and bf16 has already spent some of the code-error margin that protects the 0.0% WER
(0.81% -> 1.16%). Revisit **after** tracing, and gate it on end-to-end WER and long-form frame
counts rather than on code-diff counts. Note that sdpa's 19% will shrink once tracing removes the
host-dispatch component the two share. `is_causal` **defaults to True** and must be passed `False`
here; causal masking would cut position 0 off from the time and LLM tokens, which does not raise --
it silently produces a velocity conditioned on nothing.

**Tracing: IMPLEMENTED, verified, and worth 0.1% today — but keep it, it is a prerequisite.**
`_solve()` is a pure device graph and `_trace()` captures it (one trace per batch/n_steps/cfg_alpha).
Correct: 0/74 codes differ from the reference, repeatable, input refresh verified. The measurement
that matters, at batch 2:

| | host enqueue | device wait |
|---|---|---|
| traced | **0.0 ms** | 139.1 ms |
| eager | **121.7 ms** | 139.3 ms |

That 121.7 ms "host enqueue" figure is **backpressure, not host cost** — enqueueing faster than the
device drains blocks the enqueue call. The real host overhead is the constant gap between the traced
device floor and the eager frame time, and it is ~6.5 ms at every configuration measured (bf16 W
139.1/145.5, bfp8 W 100.5/107.0, bfp4 W 90.1/96.6). **So tracing is worth ~6.5 ms — about 6% at the
current device floor.** An earlier version of this section claimed dispatch would become a binding
wall at 121.7 ms once device work dropped; that was wrong, built on the backpressure misreading.

Also wrong in that earlier version: predicting the Block 3 tracing null result would not transfer,
on the grounds that a 3-token sequence means tiny ops. A tile is 32x32, so every matmul does 32 rows
of work for 3 useful tokens against 3072x9216 weights. Upstream's 47%-from-CUDA-graphs does not
apply — their bottleneck was launch overhead.

**Tracing is GONE from both blocks -- deleted, not disabled.** Both captures were built, proven
bit-identical, and measured on this N150: Block 1's decode trace showed no win, and Block 2's was
~6 ms/frame SLOWER. Under the no-toggles rule a dormant second path is not worth its weight, so the
capture machinery went with them (`git show 5beef54ad3c` for Block 1's, `d0653e476a6` for the
removal). The earlier "~6% from tracing" figure came from a device-bound A/B that the decode-native
work has since invalidated. If you rebuild it: read trap #1, and remember that Block 1 and Block 2
must finish ALL warm-ups before EITHER captures, or the second capture hangs.
Upstream's 47%-from-CUDA-graphs figure does not apply to us: their bottleneck was launch overhead,
ours is arithmetic. Read trap #1 before touching capture code.

### 6.6 — the optimization sweep: what landed, what failed, what is left

Swept the repo, the TTNN API and the shared branches against our two hot blocks. The framing that
made the difference: **Block 2 was not weight-read bound, whatever its docstring said.** Its
velocity net is 349M params at BFP8, so 7 Euler steps stream ~2.6 GB, and at the 194 GB/s Block 1
demonstrably reaches that is a 13.4 ms floor against a measured 35 ms — 38% of ceiling. The gap is
OP COUNT (~88 ops per step, 18 of which carry weights). Every win below removes ops, not bytes.

**Landed — 77.6 → ~66 ms/frame, RTF ~0.97 → 0.81–0.90, WER 0.88% unchanged, 15/15 termination:**

| change | where | gain | accuracy |
|---|---|---|---|
| **GQA row fold** — reshape q `[B,32,3,d]`→`[B,8,12,d]` instead of `repeat_interleave` on k/v | Block 2 `_block` | **1.40x on the block** | same velocity PCC, same 3/74 codes |
| **width-sharded RMSNorm** (8x1 grid, fp32 acc intact) | Block 2, 49 calls/frame | 1.46x on the norm+linear pair | velocity PCC 0.9999845 → 0.9999852 |

The row fold is the CFG lesson the module already preached, never applied to GQA: query head *j*
reads kv head *j//4*, so those 4 heads' 3 tokens stack as 12 ROWS against one kv head. The two
attention matmuls go batch-32 → batch-8, and rows inside a 32-row tile are free. Proven equivalent
on host in fp32 (agrees to 6e-07, pure reduction order) — not an approximation.

**Reverted — width-sharded RMSNorm in BLOCK 1.** Same 1.46x, ~5 ms/frame, and every cheap signal
said yes: 0.9999973 per op, decode PCC unmoved at 0.99991. But over 24 REAL teacher-forced frames
the **worst sample went 1.06% → 1.95%** while PCC stayed flat. Teacher-forced matters: both builds
see identical inputs at every step, so that comparison is deterministic. This is the second
instance of the trap `_norm` documents — a ~1e-6 per-op difference amplified through 26 layers —
and per-op PCC hid it both times. Block 2 keeps the sharded norm because 3 layers give nothing to
amplify. (This entry originally also cited WER 0.88% → 2.06%; see below for why that half of the
evidence was worthless.)

**This list has since been worked through — see §6.8 for the results.** Kept only as the record of
what was outstanding at the end of the first sweep:
1. ~~wqkv + wo → BFP8~~ **DONE**, 3.32 ms, no hang (§6.8).
2. ~~sdpa for Block 2's attention~~ **REJECTED** on a deterministic gate, 7/288 → 21/288 (§6.8).
3. ~~semantic argmax on device~~ **DONE**, 1.49 ms, fp32 (§6.8).
4. ~~Re-measure the device trace~~ **DONE**, still no (below).
5. **Fewer Euler steps** (7→5 removes 28% of the solve) — STILL OPEN. A MODEL change, needs a
   listening pass, not a metric.
6. **Concurrent requests** — STILL OPEN. Throughput, not latency.

**w2 in BFP8: RETRIED AND IT STILL HANGS.** Worth doing once wqkv and wo turned out fine, because
that made "all-BFP8 hangs" look over-attributed in general. It is not: w2 is genuinely the trigger.
It is worth **2.5 ms/step** (31.4 → 28.9) at PCC 0.99977–0.99985, so the hang is the only obstacle.

Two things the retry changed. First, it now hangs **earlier and harder**: the documented repro died
on the third utterance inside Block 3, whereas this died during the FIRST case, right after the
first compute op, with no pipeline output at all — today's op mix reaches the trigger sooner, so
the five-condition sequence is no longer minimal. Second, recovery: `tt-smi` is **not on PATH** on
this machine. The Wormhole build is at `/home/software/syseng/wh/tt-smi` and this vintage takes
`-wr 0`, not `-r`. `open_device` hangs indefinitely until you run it.

**Device tracing, re-measured after the op-count wins — still not worth it, but the reason
changed.** The old verdict was that tracing COST time (Block 1 -0.7 ms, Block 2 -6 ms/frame). That
penalty is gone; what is left is a gain too small to bank:

| | untraced | traced | delta |
|---|---|---|---|
| Block 1, 26-layer decode step | 34.75 ms | 34.55 ms | +0.20 ms (1.006x) |
| Block 2, whole 7-step solve | 26.07 ms | 25.91 ms | +0.16 ms (1.006x) |

Both captures are CORRECT: Block 1's PCC and worst-sample are unchanged (0.999893 / 1.06%) and
Block 2's integer codes are bit-identical to the untraced path. Both were timed with a readback
every frame, because without one the host runs ahead and pipelines the next frame's dispatch under
the current frame's device work — which hides exactly the cost a trace removes.

**What this settles: host dispatch is not a bottleneck for either block.** Tracing removes host
command submission almost entirely, and removing it changes nothing. So Block 2's remaining ~2x gap
over its 13.4 ms weight-read floor is DEVICE-side per-kernel cost — kernel launch on the cores,
circular-buffer setup, intermediates round-tripping DRAM — not commands queuing up on the host.
That is worth knowing before optimizing here: work that only reduces command COUNT will not help;
work that makes the kernels themselves bigger and fewer will.

0.5% does not justify what tracing costs to keep: `trace_region_size` on every caller, buffers
addressed by pointer so a fresh upload is silently ignored, a capture ORDER constraint between the
two blocks (all warm-ups before either capture, or the second hangs), and a reserved scratch cache
row so the warm-up and capture writes do not corrupt the prompt. Not re-added. The probes are in
the session scratchpad if this needs re-running after a big change.

**Matmul fusion: one win, and it CLOSES the matmul side of Block 2.** Following the trace result
above (device-side per-kernel cost, so make kernels bigger rather than merely fewer), the two pairs
of Block 2 matmuls that read the same input were merged: `wq`+`wkv`, and `w1`+`w3`. Only the first
paid. Per-matmul efficiency against the 194 GB/s ceiling explains exactly why:

| matmul | MB/call | us/call | GB/s | % of ceiling |
|---|---|---|---|---|
| B1 wqkv (bf16, already fused) | 37.7 | 185 | 204 | 105% |
| B1 wo (bf16) | 25.2 | 129 | 195 | 100% |
| B2 wq | 13.4 | 74 | 182 | 94% |
| **B2 wkv** | **6.7** | **74** | **91** | **47%** |
| B2 wqkv, fused | 20.1 | 101 | 198 | 102% |
| B2 w1 | 30.1 | 157 | 192 | 99% |

`wkv` was the ONLY launch-bound weight matmul anywhere in the model — at 2048 wide it cost the same
74 us as the 4096-wide `wq`, so there is a fixed ~40-50 us per launch and width past ~4096 is
nearly free. Folding q in with it took the pair from an effective 137 GB/s to 198, worth **0.96
ms/frame**. It is mathematically the same arithmetic — a linear computes each output column
independently, and 4096 is tile-aligned so the BFP8 blocks are unchanged — but NOT bit-identical:
a `[3072,6144]` matmul picks a different internal block decomposition than a 4096 and a 2048
separately, which reorders the sum over the 3072 inner dim. Same class of difference as the row
fold. 11 of 15 fixture cases reproduce their frame count exactly and 4 shift, which is the usual
chaotic-trajectory response to a last-bit change, so it is gated on WER like everything else.
`w1`+`w3` are 9216 wide each and already at 99% — merging them measured **0.998x**, and
0.951x once the output split is charged, so they stay separate. Cleared by the gates that resolve
anything: velocity PCC 0.99998522 (identical to pre-fusion), same 3/74 codes, long-form WER 0.00%,
15/15 termination. Its single-seed headline WER read 1.76%, which means nothing — see §6.7.

**Every weight matmul in both blocks is now at the DRAM ceiling.** There is no fusion left to do,
in either block: Block 1's are all 25-38 MB per call and were never launch-bound. Block 2's
remaining ~13 ms above its weight-read floor is entirely in the NON-matmul ops, and the next real
attack on it is sdpa (item 2 above), which replaces the inherently batch-8 attention interior — the
one part the row fold could not flatten, because each batch element needs a different k.

### 6.7 — the WER gate is noisier than the changes it was gating

**Same code. Only the generation seed changed.**

| seed | natural-text WER | long-form (298 w) | short (42 w) |
|---|---|---|---|
| 0 | 1.76% | **0.00%** | 14.29% |
| 1 | 2.06% | **0.00%** | 16.67% |
| 2 | 0.88% | **0.00%** | 7.14% |

0.88–2.06% is the ENTIRE range seen across every implementation variant in §6.6, endpoints
included. So **every single-seed WER comparison made during that sweep was uninformative**, and two
conclusions drawn from it have been corrected above: the row fold "held WER at 0.88%" (luck), and
the Block 1 sharded norm "cost 0.88% → 2.06%" (also luck — that revert stands on its worst-sample
measurement, which is deterministic, not on this).

The cause is word counts, not audio. Per case, same code, seeds 0/1/2:

| case | lang | words | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|---|
| 7 | spanish | 6 | 50.0% | 0.0% | 0.0% |
| 9 | portuguese | 6 | 0.0% | 83.3% | 0.0% |
| 13 | arabic | 3 | 33.3% | 0.0% | 33.3% |
| 6 | german | 7 | 28.6% | 28.6% | 28.6% |
| 0,1,2,3 | english | 298 total | 0.0% | 0.0% | 0.0% |

On a 6-word clip one Whisper disagreement is 17% and three is 50%, so 42 words of short prompts
swing the 340-word aggregate by more than a point while 298 words of English sit at exactly zero.
Case 6's 28.6% is the one short-prompt error that is REAL — it reproduces on every seed.

**The gate is now the long-form number**, which `score_quality_set.py` prints separately: 0.00%
over 298 words, in every run ever measured. Plus 15/15 natural termination and the voice-identity
check, both of which have also held everywhere.

**For judging a numerical change, prefer the deterministic gates entirely** — teacher-forced PCC
and worst-sample against the fp32 reference in `tests/tt_gates.py`. They feed both builds identical
inputs, so no chaotic trajectory is involved. That is what actually caught the Block 1 norm, and
what correctly cleared the qkv fusion. Use end-to-end WER to catch gross breakage, not to resolve
a last-bit difference — and if you must, run at least three seeds and compare ranges.

### 6.8 — second sweep, gated deterministically

Run after §6.7 established that end-to-end WER cannot resolve changes this small. Every accuracy
number below is teacher-forced against the fp32 reference — identical inputs to both builds, no
generation loop, no chaotic trajectory.

**Rejected — sdpa for Block 2's attention interior.** 1.147x (28.73 → 25.05 ms), and the accuracy
cost is real: integer codes differing from the fp32 reference go **7/288 → 21/288** over 8
independent (h, x_0) draws. That is the same ~3x §7 measured before BFP8, so the old rejection
holds and is now confirmed on a gate that can actually see it. Note an earlier single-draw check
in this session read 2/74 against the row fold's 3/74 and looked *better* — one draw is worthless.

**Rejected — two tiny-op reductions in Block 2, both worth nothing.**
- CFG combine as multiply-by-constant + sum instead of 2 slices + 2 multiplies + add: **1.001x**.
- `inplace=True` on the sharded norm program config: **0.997x**.
Removing small ops does not help. That is consistent with the trace result (§6.6): the cost is
device-side per KERNEL, and these ops are already at the per-kernel floor, so deleting a few of
them changes nothing. Only work that makes kernels BIGGER pays — which is why the qkv fusion did
and these did not. The same reasoning predicts the `concat([x, x])` idea is also worthless; it was
not isolated separately.

**Shipped — semantic head on device, 1.49 ms/frame.** Was a host CPU matmul at 2.74 ms; on device
in fp32 it is 1.25 ms. **fp32 is mandatory**: the output is an INDEX, and over 64 hidden states
bf16 weights pick a different index on 4 of them while fp32 matches the host answer 64/64. bf16
would save a further 0.2 ms in exchange for a wrong primary code on ~6% of frames.

**Shipped — wqkv and wo in BFP8 (Block 1), 3.32 ms/frame.** The untested precision point from
§6.6. It does NOT trigger the hang: the documented minimal repro (short gen + 128-bucket decode,
long gen + 512-bucket, long gen + 512-bucket as a pure cache hit) runs clean, as does the full
15-case set. w2 stays bf16 — that one is still the pinned trigger and was not retried.

**A statistics correction that matters for every future decision here.** Worst-sample was being
read as a MAX over frames, and a max is an unstable order statistic. Over 44 teacher-forced frames
across two real prompts:

| Block 1 config | ms/step | min PCC | mean ws | p90 ws | max ws |
|---|---|---|---|---|---|
| shipped (bf16 + FF BFP8) | 34.67 | 0.999884 | 0.86% | 1.11% | 1.34% |
| sharded norm only | 29.78 | 0.999867 | 0.92% | 1.16% | **4.28%** |
| + wqkv BFP8 | 32.51 | 0.999872 | 0.86% | 1.14% | 2.10% |
| + wqkv and wo BFP8 | 31.35 | 0.999852 | 0.86% | 1.10% | **1.28%** |
| + sharded norm too | 26.51 | 0.999851 | 0.84% | 1.06% | 1.40% |

The max column is not monotone in how much error is introduced — adding wo on top of wqkv took it
from 2.10% down to 1.28%. **Use mean and p90; treat max as an anecdote.** On mean/p90 the BFP8
rows are indistinguishable from shipped, which is why they ship.

This also weakens, without overturning, the §6.6 sharded-norm revert: that call cited max going
1.06% → 1.95%, and max is the unreliable column. On the stable statistics the sharded norm is
still worse alone (mean 0.86% → 0.92%), just by less than was claimed. It stays out, but the
honest reason is "slightly worse on a stable metric for 5 ms", not the dramatic doubling reported.

**Checked and ruled out** (so nobody re-runs them):
- `rms_norm(residual_input_tensor=)` returns only the normed value and discards the sum; a pre-norm
  block needs that sum for the next residual. Post-norm models (BERT) can use it, we cannot.
- **Feeding sharded activations straight to a matmul is SLOWER** — 8.94 vs 5.32 ms per 26
  norm+linear pairs, because a DRAM-interleaved weight makes the matmul gather the shards itself.
  This is why `ign`'s keep-everything-in-L1 approach does not transfer to us as-is.
- Fused SiLU in Block 2's multiply (`input_tensor_a_activations`): no measurable gain. Block 1
  already fuses it via `activation="silu"`.
- `ttnn.deallocate` on every intermediate (ign's style): no gain here.
- Norm grids 8x2 / 8x4 / 8x8: all slower than 8x1 at our shapes. 8x8 will not even build.
- Row fold in Block 1 prefill (needs a 4x-tall mask, and prefill is ~1.3% of wall) and in the codec
  (it is MHA 8/8, no grouping to fold).


### 6.9 — the sharded norm, shipped on the second look

**Block 1's two RMSNorms are now width-sharded over 8 cores: 6.1 ms → 1.1 ms, worth 4.4 ms/frame
(31.4 → 26.6 ms/step, mean frame 59.9 → 55.5, RTF 0.74-0.82 → 0.68-0.78).**

THIS REVERSES §6.6, and the reason it was wrong is the useful part. That revert measured the
sharded norm while `wqkv` and `wo` were still bf16, where it read mean worst-sample 0.86% → 0.92%.
On the CURRENT weights it reads 0.86% → 0.84% mean and 1.10% → 1.06% p90 — no cost, reproduced in
two independent runs. **A precision change here is not separable from the others; re-measure against
the config you actually ship, never against a recorded number.**

Core count barely matters, and it is worth knowing why:

| norm | us/call | ms/step | mean ws | p90 ws |
|---|---|---|---|---|
| interleaved | 115.5 | 31.42 | 0.86% | 1.10% |
| 2-core | 42.4 | 27.50 | 0.83% | 1.08% |
| 4-core | 40.5 | 26.81 | 0.81% | 1.10% |
| 8-core ← shipped | 44.1 | 26.54 | 0.84% | 1.06% |

Flat, because the norm COMPUTE is ~16 us at any count and the two `to_memory_config` calls are the
other ~28. The reshard is the tax, not the reduction. (Feeding the sharded result straight to the
next matmul to dodge the second reshard is SLOWER — 8.94 vs 5.32 ms per 26 norm+linear pairs.)

**THE COST, stated exactly, because it is not zero.** Long-form errors over three seeds:

| config | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| interleaved | 0 wrong | 0 wrong | 0 wrong |
| 8-core sharded | 1 wrong | 1 wrong | 0 wrong |

Both errors are the SAME word in the SAME sentence: the model contracts "I am" to "I'm" in "…now
that I have it, I am not going to be silent." That is the only difference across 894 long-form words
(298 × 3 seeds); termination is 15/15 and voice identity passes everywhere. Shipped on the judgement
that a contraction is what a human reader does, not a defect — an explicit call, so if a future
regression traces here, this is the trade to question first.


### 6.10 — nlp_create_qkv_heads: the op has a floor, but its OUTPUT CONFIG was free money

Chased the biggest non-matmul line in Block 2 (2.7 ms/frame, 129 us x21 — more than the wqkv matmul
that feeds it, and 18x Block 1's decode variant at 7.0 us).

**The op's cost is a FIXED ~97 us, not data movement.** The same call on S=32 — 10.7x the real data
— also measures 97 us. So there is nothing to win by feeding it less or laying the input out
differently, and indeed every restructuring is worse: hand-rolled slice+reshape+permute is 158 us
against the shipped chain's 140, and riding the CFG pair on the sequence dim is 259 us. Both were
verified bit-identical first, so those are speed results, not broken reimplementations. The sibling
ttnn ops (`create_qkv_heads`, `transformer.split_query_key_value_and_split_heads`) reject GQA shapes.

**What paid was the output memory config, and the op-level number hides why.** Isolated, an L1
output saves ~7 us on the op. In the real block it is worth **2.5 ms/frame**, because q/k/v then
stay in L1 for the four ops that consume them:

| output | transpose_k | ms/frame | codes ≠ fp32 ref (8 draws) |
|---|---|---|---|
| DRAM | False ← was | 26.75 | 7/288 |
| DRAM | True | 26.72 | 7/288 |
| L1 | False | 24.28 | 9/288 |
| L1 | True ← shipped | **24.17** (1.106x) | 9/288 |

L1 carries all the speed AND all the cost — 2 extra differing codes in 288. `transpose_k_heads=True`
is free (it emits k already transposed, deleting our transpose op) but worth 1.001x alone; it is on
for tidiness. NOT bit-exact end to end: the three tensors are identical either way, but an
L1-resident operand makes the downstream matmul choose a different program config. Velocity PCC
0.99998522 → 0.99998164.

Gated on the full run: **long-form 0 wrong words in 298**, 15/15 natural termination, voice PASS.
For scale, the accepted BFP8-weights trade was 1.23x for one extra code in 222; the rejected sdpa
trade was 1.147x for fourteen (7→21). This is 1.106x for two.

**Generalisable lesson: for the small tensors in this block, WHERE a tensor lives matters as much as
how big the kernel is.** Every previous win here came from making kernels bigger; this one came from
keeping an operand in L1 across its consumers.

**That generalised, and was worth another 1.15 ms.** Applying the same idea to the rest of
`_block`'s intermediates, cumulatively, at IDENTICAL accuracy throughout (9/288 differing codes in
every row):

| | ms/frame | vs q/k/v-only |
|---|---|---|
| q/k/v L1 only | 24.18 | — |
| + attention interior (scores, scaled, av) | 23.85 | 1.014x |
| + MLP intermediates (g, w3_out, u) | 23.22 | 1.041x |
| + residual stream | **23.04** | **1.049x** ← shipped |

The one candidate that does NOT pay is the `_norm` output: **0.999x alone**, so it stays DRAM. So
this is not "L1 everywhere is better" — it is specifically values with a consumer close behind, and
the norm's output is immediately eaten by a big DRAM-weight matmul that dominates it. The LIMIT is
still §6.6: a width-SHARDED activation into a DRAM-weight matmul is slower. Interleaved-L1 is the
useful middle.

**IT TRANSFERS TO BLOCK 1 TOO, for another 0.9 ms at zero accuracy cost** — min PCC 0.999850, mean
worst-sample 0.85%, p90 1.09% and even max 1.40% all byte-identical before and after, over 44
teacher-forced frames:

| Block 1 decode | ms/step | vs shipped |
|---|---|---|
| shipped | 26.43 | — |
| + wo output and residual L1 | 26.19 | 1.009x |
| + MLP intermediates (g, u) L1 | **25.53** | **1.035x** ← shipped |

And a THIRD negative that sharpens the rule: **sdpa_decode's output stays forced to DRAM.** That
`to_memory_config(o, DRAM)` looks like exactly the round trip this whole finding is about, and
routing it to L1 measures 0.999x — nothing. So the pattern is not "L1 is faster", nor even "avoid
DRAM round trips". It is specifically **values with a consumer close behind**; the norm output and
sdpa output both feed a single large op that dominates whatever the transfer cost was.

`_mlp` is shared with prefill, so the memory config is passed in at the call site rather than baked
in — prefill's `g` is [1,S,9216] with S up to 384, i.e. 6.8 MB, so it passes DRAM. That mirrors how
`h` is already handled, and for the same reason: it is what the two paths do differently.

**BLOCK 3 IS NOT THE NEXT TARGET, and the "~9% of wall" figure quoted in §6.6 and §6.8 was wrong.**
That number came from `wall − prefill − generate` on the 15-case run, which lumps FIRST-CALL KERNEL
COMPILATION for each new codec bucket in with the codec's actual compute. Measured directly:

| | T=461 | T=512 |
|---|---|---|
| cold (a length not compiled yet) | **3399 ms** | 97 ms |
| warm | 97 ms | 97 ms |
| `_graph` at an UNBUCKETED length | **13245 ms** | 95 ms |

So the decoder's steady-state compute is **97 ms against a 25.8 s generation — 0.4%**, and the
seconds-scale figure is compile cost, paid once per distinct bucket (12 of them for the ~1500-frame
ceiling, so a long-lived server pays it 12 times ever). Same class of mistake as reading case 0's
RTF 1.89 as a slow case. **Optimizing codec compute is worth ~0.4% of wall; there is nothing there.**

Two ideas were measured against it before that was understood, and both LOSE, so they are closed:
- **Per-stage SLAB sized to the window.** The single global 512 computes a 512x512 score matrix where
  only a (window+1)-wide band survives the mask — 97% waste at the last stage, 99.4% at the first.
  Shrinking it is much worse, because the per-chunk machinery dominates the wasted arithmetic: at
  L=4096, slab 64/128/256/512/1024 measures 40.4/19.5/12.1/**9.7**/11.5 ms. 512 is genuinely optimal
  for three of four stages, as its comment claimed. The one win is tf1 (L=1024) at slab 1024, where
  it becomes a single chunk — 2.7 ms of 94, and not bit-exact (2.4e-04).
- **The score chain in L1**, the trick that paid in §6.10. It LOSES here, 0.82–0.94x. The score
  tensor is 4.2 MB at slab 512, far past the ~100 KB scale where L1 helped in the other two blocks.
  **That is the size boundary on the §6.10 finding**: L1 for intermediates works at
  tens-to-hundreds of KB, not megabytes.

Also a useful negative on `ign/voxtral_opt`'s lead: the transformer layers are **92%** of the codec
(86.4 of 94.35 ms at T=512) and all the convs together are 8.5%, so their conv-split optimization
would not have transferred either.

If anything in Block 3 ever deserves attention it is the COMPILE cost, not the compute — and
`BUCKET = 128` already exists to cap it.


### 6.11 — the norm's second reshard cannot be dodged, both ways measured

`_norm_dec` width-shards, norms, then converts back to DRAM. That second `to_memory_config` is 0.33
ms/frame and looks removable. It is not, and the two candidates fail for different reasons.

**Default matmul, sharded activation: 8.94 vs 5.32 ms per 26 norm+linear pairs.** The cause is the
AXIS, not L1 vs DRAM. Width-sharding splits the matmul's CONTRACTION dimension, so each core forms
only a partial sum and the cross-core reduce is full-output-sized ([32,6144] x 8 cores). Interleaved,
ttnn splits by OUTPUT COLUMNS: each core owns its columns, reads the whole 6 KB activation, nothing to
reduce. **The same axis that makes the norm fast makes the matmul slow** — the norm REDUCES over
width (cross-core step is 8 scalars), the matmul CONTRACTS over it.

**The matmul that wants a sharded activation: builds, still loses.**
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` requires exactly this layout. The DRAM-sharded-weights note above recorded
it overflowing L1 at Block 2's shape (N=9216, per_core_N=36 tiles, CBs 1,678,528 B against
1,499,136). Block 1's wqkv is N=6144, per_core_N=**24** tiles, and it does build — the capacity
projection was right. It is still slower:

| | us | vs shipped |
|---|---|---|
| shipped: unshard + default matmul | **100.9** | 1.000x |
| DRAM-sharded, sharded activation in | 125.4 | 0.805x |
| + unshard the output | 128.5 | 0.785x |

**0.72 ms/frame worse**, and not bit-exact (4.9e-04). The reason is simply that the default path
already runs this matmul at ~198 GB/s, i.e. at the DRAM ceiling, so there is no bandwidth for the
DRAM-sharded machinery to win back — it only adds its own cost. That config earns its keep when the
weight read is contended or spread across devices, not when a plain 1D config already saturates.

**And sharding only the WEIGHT is not available.** Width-sharding the weight splits the OUTPUT
columns — the axis needing no reduction — so it looks like the free version of this idea. ttnn
couples them: a width-sharded `in1` is accepted by ONLY the DRAM-sharded config, which also demands
a width-sharded `in0`; every other config asserts `in1` is INTERLEAVED
(`matmul_device_operation.cpp:1188ff`, with the pairing spelled out in the comment at :1199). So
"sharded weight, interleaved activation" cannot be expressed, and the paired form is the 0.805x above.

An **L1-resident weight** fails the same assertion. It was the interesting one on paper — a resident
weight is never re-read from DRAM — but the ceiling kills it regardless: Block 1 streams ~3.9 GB per
frame against **96 MB of total L1 (64 x 1.5 MB), 2.4% of the model**, and that L1 is also wanted by
the activations and circular buffers. **The model not fitting in L1 is exactly why 194 GB/s is the
wall**, and no sharding of a 3.9 GB working set changes that.

**Generalisable:** before reaching for a fancier matmul config, check what the current one achieves.
Every hand-tuned config tried in this port has lost, and in each case the plain one was already at
the ceiling (see also the 169-vs-193 GB/s program-config result in ttnn_voxtral_gpt).

### Standing constraints (not fixable by us)
- Weights are **CC BY-NC 4.0**, non-commercial, including the reference voices. Same class of
  blocker as XTTS-v2's CPML. Needs legal sign-off before any product use.
- **Voice cloning from arbitrary audio is impossible** — the codec encoder is not in the release
  (0 of 386 tensors). Only the 20 shipped presets. A test asserts this so a future release that
  adds them fails loudly.

---

## 7. Suggested order when resuming

**All three blocks work, quality is good, and the port is no longer the bottleneck — the model is.**

Per frame, steady state on one N150: **Block 1 ~35 ms, Block 2 ~42.5 ms, host 0.2 ms, ~77 ms
total, RTF ~1.0.** Both hot modules open with a "where the time goes" map giving the ceiling for
each line item; read those before optimizing anything.

1. Re-read this file. Recreate the venvs (§2). Run the tests, then
   `generate_quality_set.py --cases 0,1` to confirm the device path still speaks.
2. **BLOCK 3 IS THE LEAST EXPLORED THING LEFT — ~9% of wall time and never swept.** Both hot
   blocks have had two optimization passes; the codec has had none since its own bring-up.
   `ign/voxtral_opt` has a `keep_sharded_splits` commit claiming ~18 ms → ~6 ms on their conv
   stack by removing ~104 layout conversions, which is exactly the class of thing that pays here.
3. **Both hot blocks are near their floors, for different reasons.** Block 1 (~31.4 ms): every
   weight matmul is at the 194 GB/s ceiling and the only bf16 weight left is w2, the pinned hang
   trigger — worth ~3.6 ms if it can ever be moved. Block 2 (~28 ms): sits ~2x above its 13.4 ms
   weight-read floor on device-side per-kernel cost, and everything op-level has been tried. The
   two ideas left there are structural — fewer Euler steps, or concurrent requests.
4. **Gate on the DETERMINISTIC metrics, not on WER.** This is the most expensive lesson in the
   file and it was learned twice. End-to-end WER cannot resolve a numerical change: the same code
   at seeds 0/1/2 spans 0.88–2.06% (§6.7). Worst-sample read as a MAX is also unreliable — it
   moved 1.28–4.28% non-monotonically across configs (§6.8). What works: teacher-forced MEAN and
   P90 worst-sample against the fp32 reference (`tests/tt_gates.py`), long-form WER for gross
   breakage, and integer-code counts over several draws for Block 2.
5. `tests/test_tt_defaults.py` pins the shipped configuration with the reason for each choice. If
   you change a default, change that test in the same commit.

**Deferred and still worth doing:**
- A **listening pass** — WER cannot see timbre, and nobody has A/B'd the current build.
- **Report the ttnn hang upstream** (§6, Block 1). A silent hang needing a board reset is their
  bug regardless of what we feed it, and we have a ~90 s repro.
- **Prefill beyond ~1024 tokens** needs chunked prefill.
- A **like-for-like comparison against `ign/voxtral_p150_qb2` on THEIR hardware** — see below for
  what we could and could not settle.
