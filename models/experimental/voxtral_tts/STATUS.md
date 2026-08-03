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

**Block 1 now runs on our own implementation, not `tt_transformers`.** The wrapper stays runnable
for bisection (`VOXTRAL_BACKBONE=tt_transformers`, needs `HF_MODEL`). Measured against the fp32
CPU reference on real prompts, and end to end on the 15-case fixture:

| | ours | `tt_transformers` |
|---|---|---|
| prefill, last position | 0.999881 | 0.999564 |
| decode step | 0.99991 | 0.981 |
| decode ms/frame | 34.9 | 48 |
| natural-text WER | 0.88% | 1.17% |

**Performance: 83.7 → ~77 ms/frame, steady-state RTF ~1.0.** Per frame: Block 1 ~35 ms, Block 2
~42.5 ms, host 0.2 ms. Block 2 is now the LARGER half and is where the next work belongs. Both
modules carry a "where the time goes" map at the top with the ceiling for each line item.

**Shipped configuration**, pinned by `tests/test_tt_defaults.py`:
Block 1 mixed precision (BFP8 on FF1_FF3 only) + decode-native heads; Block 2 BFP8 weights at
HiFi4 + fp32 accumulation; device traces implemented but off; no program-cache clearing.

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
# No HF export and no HF_MODEL: Block 1 is ours now and loads the Mistral-native checkpoint.
# Only the bisection path needs them:
#   scripts/export_backbone_hf.py --out /tmp/hf_backbone
#   VOXTRAL_BACKBONE=tt_transformers HF_MODEL=/tmp/hf_backbone python .../generate_quality_set.py
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
dispatch-bound; that is Blackhole, and it does not transfer to this N150.

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

`USE_TRACE` is left **False**: ~6% is real but it forces `trace_region_size` on every caller. Flip it
when that 6% is worth the coupling. Read trap #1 before touching capture code.
Upstream's 47%-from-CUDA-graphs figure does not apply to us: their bottleneck was launch overhead,
ours is arithmetic. Read trap #1 before touching capture code.

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
2. **Block 2 is the larger half and the right target.** 35 of its 42.5 ms is 7 SEQUENTIAL Euler
   steps, each a 3-layer transformer over 3 tokens — every matmul does 32 tile rows of work for 6
   useful ones. Already tried and rejected: lower math fidelity (~4 ms for 10–20x the integer-code
   errors) and a device trace (bit-identical, ~6 ms/frame slower). BFP8 weights ARE on and were
   worth 1.23x. The untried idea is CONCURRENT REQUESTS filling the wasted 26 tile rows —
   throughput, not latency.
3. **Block 1 is close to its floor.** Only fewer weight bytes help, and BFP8 beyond FF1_FF3 hits
   the hang. Small change left: `paged_fused_update_cache` and `rotary_embedding_llama_fused_qk`
   are worth ~0.7 ms combined.
4. **Gate every change on WER, not PCC.** Two lessons from this port, both expensive: a cheap
   RMSNorm looked fine at per-op PCC 0.999993 and took model decode PCC to 0.992; and case 4
   cannot measure an implementation at all (see §1). Per-case WER on a short clip is trajectory
   noise — the same case swings 0%–28.6% on the SEED alone at fixed precision. Only the aggregate
   over 340+ words means anything.
5. `tests/test_tt_defaults.py` pins the shipped configuration with the reason for each choice. If
   you change a default, change that test in the same commit.

**Deferred and still worth doing:**
- A **listening pass** — WER cannot see timbre, and nobody has A/B'd the current build.
- **Report the ttnn hang upstream** (§6, Block 1). A silent hang needing a board reset is their
  bug regardless of what we feed it, and we have a ~90 s repro.
- **Prefill beyond ~1024 tokens** needs chunked prefill.
- A **real comparison against `ign/voxtral_p150_qb2`** — theirs is Blackhole P150 on a larger 4B
  variant, so none of the published numbers are like-for-like with ours.
