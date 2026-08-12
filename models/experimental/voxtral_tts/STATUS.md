# Voxtral-TTS on TTNN — status and resumption notes

**Read this first when picking the work back up.** It is written to be self-contained: state,
measurements, the traps that cost time, and what to do next. Architecture detail and the
reference-side findings live in `reference/PROVENANCE.md`; this file is the *work* state.

Branch: `lserbedzija/voxtral-tts-ttnn_p150` (pushed). All work is under
`models/experimental/voxtral_tts/`. Nothing else in the repo is touched.

> ## ⚠ THIS IS THE BLACKHOLE p150 FORK. §1–§6.38 ARE WORMHOLE N150 NUMBERS.
>
> Forked from `lserbedzija/voxtral-tts-ttnn` at `a3b4569021` and retargeted to a **Blackhole
> p150b**: `Arch.BLACKHOLE`, compute grid **13×10 = 130 Tensix** against the N150's 8×8 = 64,
> 8 DRAM banks of GDDR6 against 12, L1 1.5 MB/core. **Every performance number and every tuned
> constant recorded in §1–§6.38 was measured on the N150 and does not transfer.** The port
> itself does: it ran on the p150 with ZERO source changes at 0 long-form WER errors, and the
> §6.39+ sections are the re-derivation.
>
> Current on the p150: **long-form RTF ~0.57, ~45.4 ms/frame** (N150: 0.57–0.64, ~48) — the fork
> now beats the Wormhole build it forked from, after six reversals (§6.39–§6.45).
> Divergence from the N150 branch is deliberate and expected — a loser here can be a winner
> there, so changes are DELETED rather than flagged, and the two branches are not merging back.
>
> | | N150 (§1–§6.38) | p150 (§6.39+) |
> |---|---|---|
> | Block 1 decode step | 23.15 ms | **21.2 ms** |
> | Block 2 frame | 20.8 ms | **21.9 ms** |
> | long-form gen | ~48 ms/frame | **45.4 ms/frame** |
> | DRAM ceiling | 194–202 GB/s | **~360 GB/s** (§6.41) |
> | per-op floor | ~20 µs | **~68 µs** (§6.45) |
> | decode RMSNorm, both blocks | width-sharded 8×4 | **interleaved** (§6.39/§6.40) |
> | `wo` program config | hand-tuned | **none** (§6.43) |
> | KV cache write | fused, V moved to (1,0) | **two plain writes** (§6.44) |
> | Block 2 head split / interior | 9-op hand-roll / 4 ops | **fused op / sdpa** (§6.45) |
>
> **The rule those two hardware numbers imply (§6.45): bytes are cheap and launches are
> expensive here — the inverse of the N150 — so DELETING ops wins where §6.6 wanted fewer,
> bigger kernels.**
>
> Environment differs too, and `ONBOARDING §2`'s recipe is wrong here: `PYTHONPATH` must be
> `$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME`. `$TT_METAL_HOME` alone resolves
> `ttnn` to the empty outer namespace directory and produces exactly the
> `AttributeError: module 'ttnn' has no attribute 'L1_MEMORY_CONFIG'` that §2 warns about.

---

## 1. Where things stand

| Piece | State |
|---|---|
| CPU reference, 3 blocks + tokenizer + end-to-end pipeline | **done**, 30/30 vs upstream |
| Block 3 — codec decoder on TTNN | **CLOSED**, 242x real-time, see §4 |
| Block 1 — 3.4B AR backbone on TTNN | **done — OURS** (`tt/ttnn_voxtral_gpt.py`), the default |
| Block 2 — flow-matching transformer on TTNN | **done** — velocity PCC 0.9999989 |
| **End-to-end on device** (text ids + voice → 24 kHz wav) | **works** — **26.9 ms/frame, RTF 0.357**, long-form WER **0 of 894**, MOS long-form 4.61 (§6.72) |
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

**Performance: 83.7 → ~48 ms/frame, long-form RTF 0.57-0.64** (15 cases, all terminating on `[END_AUDIO]`; see §6.21 for why long-form is the number to quote; §6.31 for the Block 2 sweep that took it there). It touched 47.5 ms / RTF 0.60-0.65 with w2 in BFP8, and 2.5 ms of that was handed back deliberately: w2 cost 77% of the precision stack's accuracy for 15% of its speed (§6.16). Accuracy: Block 1 mean/p90 worst-sample **0.92% / 1.28%**, min PCC 0.999040 (the 8x4 norm grid, §6.18); long-form WER **0 wrong words of 298**.
Per frame: Block 1 ~23 ms, Block 2 **~20.8 ms** (§6.31), host 0.2 ms. w2 is in BFP8 as of §6.13 — the hang that
blocked it is fixed, by not calling `ttnn.conv1d` in the codec.
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

> **A LISTENING PASS HAS SINCE BEEN DONE ON THE p150 BUILD — see §6.73.** It produced one
> finding, on an adversarial fixture, traced to model behaviour rather than the port. The
> paragraph below is the older N150-era pass and is kept for the record.

**Listening pass: done, informally — verdict "sounds good".** Re-done on the POST-BLOCK-2-SWEEP build
(2026-08-06, `82d04f977a1`, Block 2 at ~21 ms/frame after §6.30/§6.31) via
`generated/SAMPLER_current_build.wav` (**no longer on the box** — and its name has been misleading
since the p150 fork; it is this N150-era pass, not HEAD) — one 136 s file, 13 clips, both 37 s and 39 s long-form English
cases first, then prosody, then eight languages. Verdict from the author: **"sounds good"**. The earlier
pass was on the pre-sweep build (2026-07-30, verdict "sounds ok"). That clears the bar of "no audible defect the metrics
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

### Accuracy bookkeeping — read §6.15 before quoting any accuracy number
- **The decode gate's prompt-to-prompt spread (0.45 pp mean, 0.96 pp p90 over 8 prompts) is larger
  than every change that was gated with it**, w2's 0.10 pp included. §6.8 reported a 2-prompt pair to
  0.01 pp and did not record which two. The gate is sound for **paired, same-session, same-prompt**
  A/B — it is deterministic and repeats bit-identically — and unsound for absolute levels or
  cross-session comparison. Gate on all 8+ prompts, recorded by index, both arms in one session.
- **§6.8's absolute levels are unreachable today and the cause is unidentified.** It records mean
  0.84% / p90 1.06%; the best of 28 prompt pairs is 0.99% / 1.38%. Gate code, Block 1 code, the
  reference, the ttnn build and prompt selection have all been ruled out by direct check — see §6.15
  so nobody repeats them. Every "indistinguishable from shipped on mean/p90" call in §6.8 rests on
  those levels.

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

### 6.0 — INDEX of the p150 fork (§6.39 onward)

The sections below are chronological. This is the map; each line is one experiment and its
verdict, so you can jump rather than read 3,900 lines.

- **§6.39** — the sharded norm REVERSES. Block 1 drops it, +4.6 ms/frame  ⟵ **SUPERSEDED by §6.67**
- **§6.40** — the same reversal in Block 2  ⟵ **SUPERSEDED by §6.67**
- **§6.41** — the DRAM ceiling is ~360 GB/s, not 194–202. Re-derive before ranking anything
- **§6.42** — w1+w3 fusion re-tested. Still rejected, for an entirely different reason
- **§6.43** — `_WO_PRG` deleted. Bit-exact, and no instrument can find a speed difference
- **§6.44** — the fused cache write LOSES, and _V_SHARD goes with it
- **§6.45** — a small op costs 3.4x more, and THREE op-count decisions reverse
- **§6.46** — `_SDPA_PRG` swept and KEPT. The one N150 program config that survived
- **§6.47** — in-place elementwise, +0.929 ms/step. Residual-as-bias rejected  ⟵ that rejection **EXPIRED, see §6.62**
- **§6.48** — and the L1 trap inside it
- **§6.49** — host dispatch is 3-4% in a TIGHT LOOP  ⟵ **SUPERSEDED by §6.64/§6.65**: 13.3% in Block 2 once §6.52 exposed it
- **§6.50** — the three host steps stay on host. Device is 7-29x slower
- **§6.51** — what the rest of the repo and `ign/voxtral_p150_qb2` actually have
- **§6.52** — decode matmul program configs, −4.24 ms/frame. And `activation="silu"` never fused
- **§6.53** — why a "much more powerful" p150 was only ~7% faster than the N150
- **§6.54** — the codes gate's "29.5% of codes differ" is a synthetic-input artefact, and always was
- **§6.55** — prefill is where the synthetic error lives, and there is no lever worth pulling
- **§6.56** — higher-precision PREFILL: the weights do nothing, the activations do, and it still loses
- **§6.57** — fp32 cache is unreachable, and bf16 decode weights cost 29% for nothing
- **§6.58** — full ship-readiness pass on HEAD, and the one defect traced to the model
- **§6.59** — an automated MOS eval, and an MCD that failed calibration and was discarded
- **§6.60** — every check behind one command, with paired comparison
- **§6.61** — `out_subblock_w`'s candidate list had a hole in it
- **§6.62** — residual as matmul bias, −1.918 ms/step. And a rejection that EXPIRED
- **§6.63** — the block A/B does not predict the frame: 10 host crossings, 2.8 ms of drain
- **§6.64** — tracing Block 2 is worth 2.29 ms and needs an allocation refactor to collect it
- **§6.65** — the frame loop is TRACED: −4.244 ms/frame, RTF 0.4931 → 0.4514, every quality metric identical
- **§6.66** — review pass after §6.65: four defects, three of them on paths no test reaches
- **§6.67** — the sharded decode norm REVERSES BACK: +5.399 ms/frame, RTF 0.4415 → 0.3647, WER 1 → 0
- **§6.68** — the stale-rejection sweep comes back EMPTY  ⟵ **its INVENTORY is wrong, see §6.72**
- **§6.69** — there is no prefill length limit; utterance length is `max_seq_len` and costs DRAM,
  not RTF ⟵ **corrects a "~1024 tokens" claim that appeared in three places and was never measured**
- **§6.70** — runs on tt-metal main +777 commits with **no source changes** and the same speed;
  acoustic codes 45→40 (kernel rounding, toward the reference)
- **§6.74** — `_trunk`'s sequence build: the reshape is a GHOST (0.068 not 0.742 ms/frame),
  the concat is BIGGER than recorded (1.16-1.90 not 0.707). §7's lead is all concat
- **§6.73** — symbol input: the device is systematically SHORTER, and the reference's extra
  length is a repetition loop in 3 of 3 draws. The first finding a listening pass has produced
- **§6.72** — the head split reverses BACK, −0.775 ms/frame bit-exact. §6.68 counted one op
  short: its 6.2 µs was Block 1's op, and Block 2's is 90.5
- **§6.71** — the headline REPRODUCED on a clean tree: 27.664 ms/frame, RTF 0.3656, WER 0 of 894.
  Plus the first decode gate on §6.67's norm, and a listening sampler that exists at last

**Shipped, in order of size:** §6.65 traced frame loop (−4.2 ms), §6.67 sharded decode
norm (−5.4), §6.52 matmul program configs (−4.2), §6.62 residual-as-bias (−1.9, block only),
§6.45 fused head split + sdpa (−6.6), §6.39/§6.40 (superseded by §6.67), §6.47/§6.48 in-place.

**The four rules that cost the most to learn:** §6.63 (a block A/B is a screen),
§6.67 (an eager op map ranks by launch cost), §6.54 (the synthetic codes gate cannot rank
configs), §6.68 (a rejection is stale when its premise is a cost someone has since removed).

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

**Still open:** batch=1 only. (Prefill length was listed here as a limit for a long time. It is
not one — §6.69 measured prefill clean to 4096 tokens.)

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
- ~~Norm grids 8x2 / 8x4 / 8x8: all slower than 8x1.~~ **WRONG, see §6.18** -- measured on the
  isolated norm, which ranks grids backwards. 8x4 (32 cores) is the fastest end to end and now
  ships. 8x8 genuinely cannot build: 3072/32 = 96 tiles and 96/64 is not an integer.
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


### 6.12 — the w2 hang, root-caused

Found with `TT_METAL_WATCHER=10`, which converts the hang into a clean abort plus a device-side dump.
**Investigating this no longer costs a board reset**, which is what made it tractable at all.

| | |
|---|---|
| **exact op** | `ttnn_voxtral_codec.py:589` — the codec's OUTPUT projection, `_conv1d(x, "out", 1024, 240, kernel=7, stride=1, "reflect")` at L=4102 |
| **exact kernel** | `ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp`, stuck on both BRISC and NCRISC |
| **exact fault** | NCRISC on noc1 attempts a **unicast write of 13,897,728 bytes** from local `L1[0x15f0000]` to virtual core **18-52** `[addr=0x008ae800]` — a core that does not exist |
| **trigger** | the SECOND execution of that exact shape, i.e. a pure program-cache hit |

`13,897,728 = 3393 × 4096`, and 4096 B is exactly one input row (1024 fp32 channels). So it is trying
to push **3393 rows in a single NOC transaction, to nowhere.**

**It is not a deadlock** — it is a corrupted descriptor producing an out-of-range write. That is why
memory looked flat with 8 GB free, and why program-cache counting and leak hunting never found it.

**The trigger is the cache hit, confirmed by the conv trace.** Case 2 runs `out` at L=4102 and
completes; case 3's byte-identical call faults. That is exactly the "pure cache HIT" of the old
five-condition repro, now with a kernel name attached.

**And it is not a Block 1 bug.** w2's dtype matters only because it is the biggest lever we have on
DRAM allocation ADDRESSES — BFP8 frees ~690 MB across 26 layers, shifting every later allocation
including the codec's conv buffers. This is a latent address-dependent bug in ttnn's conv halo path
that w2 happens to expose; anything moving allocations by enough would do the same. That explains
why the five conditions looked so arbitrary: they are just a recipe for reaching a bad address
twice.

**Two of my own earlier claims were wrong and are corrected here.** §6.x said the hang "now fires
earlier and harder — during the FIRST case, with no pipeline output at all". It does not: Python's
stdout is block-buffered and the abort discards it, so the absence of output meant nothing. It hangs
in the THIRD utterance's codec decode, exactly where the original diagnosis put it. A single-case
run completes cleanly, consistent with needing ≥2 buckets.

**DODGED — see §6.13.** The second idea works: the output projection skips `conv1d` entirely.


### 6.13 — the hang, FIXED, and w2 shipped in BFP8

§6.12 named the faulting op. It is **our own call**, so it was ours to remove: the codec's output
projection no longer uses `ttnn.conv1d`. A k=7 stride-1 conv over a pre-padded tensor is a
sliding-window matmul —

    out[t] = sum_j  xpad[t+j] @ W[j]

— so 7 slices + 7 matmuls + 6 adds compute it exactly and touch no `halo_gather` kernel.

**It is slower here and overwhelmingly worth it**, because this runs once per utterance while w2
saves 2.5 ms per frame:

| output projection | ms (L=4096) | vs conv |
|---|---|---|
| `ttnn.conv1d` (was, and broken) | 4.29 | — |
| 7 matmuls, shifting the INPUT | 9.16 | +4.87 |
| 7 matmuls, shifting the OUTPUT ← shipped | **6.27** | **+1.98** |

**Shift the OUTPUT, not the input.** Both orders compute the same sum, but the shift must be a
slice, and slicing the 1024-wide input costs 0.624 ms a time against 0.145 for the 240-wide output.
Breaking down the 9.16 ms version: input slices 4.37 ms, the seven matmuls only 1.93, adds 1.16 —
**the slices cost more than the matmuls**. Multiplying the full padded input first and slicing the
narrow result instead saves 2.89 ms.

`+2.0 ms once` against `−2.5 ms × 460 frames = −1150 ms`.

**Not parallelisable**, before anyone asks: the seven passes are independent, but every ttnn op
already uses the whole 64-core grid, so running them concurrently would just give each pass 9 cores.
Nothing is idle. Holding `xp` in L1 to avoid re-reading it does not fit either — 16.8 MB overflows
the allocator. At 42 GB/s (22% of ceiling) there is still headroom here, but not via either route.

**Result, 3 seeds × 15 cases = 45 utterances:**

| | before | after |
|---|---|---|
| ms/frame | mean 50.4 | **mean 47.5** (45.9–52.1) |
| RTF | 0.62–0.71 | **0.58–0.81** |
| hang | w2 BFP8 impossible | **45/45 utterances clean** |
| termination | 15/15 | 15/15 in all three seeds |
| voice identity | PASS | PASS |
| codec waveform PCC | 0.999915 | **0.999915**, unchanged |

**THE COST, stated plainly.** Long-form errors across seeds 0/1/2 are **1 / 1 / 0 words of 298**,
against a shipped baseline that was 0 at seed 0. The errors land in DIFFERENT cases each seed (case 3
`"listened"→"listen"`, then case 2), which is the signature of noise rather than a systematic defect —
and 1/1/0 is the identical spread to the sharded-norm change accepted in §6.9. Shipped on that basis.
If a regression is ever traced to codec output quality, this and §6.9 are the two trades to question.

**The generalisable lesson:** the hang was called "unexplained" for months and treated as a hardware
mystery. It took one environment variable (`TT_METAL_WATCHER=10`, which turns the hang into a clean
abort with the kernel named) and then one observation — that the faulting op was ours, not ttnn's, so
we could simply stop calling it. Before writing something off as someone else's bug, check whether
you are the one invoking it.

### 6.14 — the codec projection: the question was "fuse the matmuls", the answer was the pad

The replacement projection from §6.13 cost 6.26 ms against `ttnn.conv1d`'s 4.29 — a regression we
accepted because it runs once per utterance and unlocks 2.5 ms *per frame*. The obvious next move is
to fuse its seven tap matmuls, since each one re-reads the same 16 MiB `xp` from DRAM: 118 MiB of
traffic to produce 3.9 MiB of output. **Fusing works and is nearly worthless. The padding was the
expense.** Six rounds of probes (`probe_fuse_taps*.py`), and the first three rounds each refuted the
previous round's premise.

**Round 1, the fusion itself.** Concat g taps side by side into one `[1024, g*pitch]` weight, one
matmul, then g column-block slices. DRAM traffic is *not* monotone in g, because the wider output
stops fitting L1 and starts paying a write plus a read-back:

| g | out | xp reads | best |
|---|---|---|---|
| 1 | 3.9 MiB → L1 | 7 | 6.25 ms |
| 2 | 8.0 MiB | 4 | 6.04 ms |
| 3 | 12.0 MiB → DRAM | 3 | 5.89 ms |
| 7 | 29.4 MiB → DRAM | 1 | worse |

1.06×. But the same run reported `_pad_causal` alone at **3.06 of the 6.26 ms**, which reframed
everything.

**Round 2 killed its own hypothesis.** The pad builds a 16 MiB tensor with a 7-input concat, so the
concat looked like the cost. Measuring four padding strategies showed otherwise — the variant that
builds *no* 16 MiB tensor at all, only a 12-row one, still cost 2.554 ms against the current 3.062.
A 48 KiB tensor cannot cost 2.5 ms. The concat was worth ~0.4 ms. What all four shared was **six
single-row `ttnn.slice` calls**.

**Round 3 found the actual currency: ops, not bytes.**

```
one single-row slice of the 16 MiB x     0.381 ms      six of them            2.282 ms
one SIX-row slice of x, rows 1:7         0.358 ms      ragged 7-way concat    1.815 ms
six single-row slices of a 24 KiB tensor 1.022 ms      aligned 2-way concat   0.281 ms
```

Six rows cost the same as one. A 4 KiB slice costs 170 µs. Cost is per *op*, and against a
TILE_LAYOUT tensor an unaligned single-row slice appears to pay something proportional to the whole
tensor.

**Rounds 4–6, the fix.** The prefix is rows x6,x5,x4,x3,x2,x1 — contiguous but **reversed**, and
there is no `ttnn.flip`. Five ways to reverse six rows:

| method | ops | ms | exact? |
|---|---|---|---|
| six 1-row slices + concat (what shipped) | 8 | 1.665 | yes |
| permutation matmul, HiFi3 | 3 | 0.195 | **no**, 2.4e-04 |
| permutation matmul, HiFi4 + fp32_dest_acc | 3 | 0.194 | **no**, 2.4e-04 |
| `ttnn.slice` with step −1 | 1 | 0.202 | **returned 0 rows, no error** |
| `ttnn.gather` into a 32-row prefix | 2 | **0.071** | **yes** |

`ttnn.gather` wins outright, and folding it straight into a tile-aligned 32-row prefix makes the
reversal, the alignment and the prefix cost 3 ops together. Output slices then start at 26 instead
of 0; prefix rows 0..25 are deliberate copies of `x[0]`, finite, and sliced away.

**Shipped: 6.26 → 3.45 ms, bit-identical, and finally below `ttnn.conv1d`'s 4.29.** Verified three
ways: probe max-abs-diff exactly 0.0 at L=4096; the codec gate's PCC lines diff *identically* against
the pre-change build; 26/26 tests, full real-prompt set clean under `TT_METAL_WATCHER=10`, RTF
0.60–0.65.

**Fusion was measured and NOT taken.** On top of the gather pad, g=3 gives 2.94 ms — 0.51 ms more —
but costs bit-exactness (1.1e-04: a `[1024,768]` matmul decomposes differently from three
`[1024,240]` ones, same phenomenon as Block 2's qkv fusion). Not worth trading exactness for 0.51 ms
on an op that is 0.015% of wall.

**Three ttnn behaviours worth knowing, all found by checking numerics that "obviously" could not be
wrong:**
- A 240-wide (half-tile) column block sliced out of an **L1** tensor comes back **silently wrong** —
  rel err 5e-01, no exception. 256-aligned blocks are fine.
- `ttnn.slice` with a negative step returns an **empty tensor**, no error.
- An **fp32 matmul multiplies at bf16 precision** here: a 0/1 permutation matrix loses 2.4e-04, and
  HiFi4 with `fp32_dest_acc_en` changes nothing. Worth remembering before trusting fp32 anywhere in
  this port for precision rather than range.

**Also fixed:** `test_prepared_weights_are_deduplicated` asserted 20 prepared `(conv, length)` pairs
and had been failing at 16 since §6.13 — `out` left `ttnn.conv1d`, so 5 convs × 4 buckets became 4 × 4.
Verified stale *before* this change, not caused by it.

**The generalisable lesson, and it is the same shape as §6.13's:** I set out to optimise the thing
the arithmetic said was expensive (118 MiB of redundant reads) and it was worth 6%. The thing
actually costing 49% was six lines of index bookkeeping nobody had timed. Decompose before
optimising — and when a decomposition contains an item that *cannot* cost what it measures, that
item is the finding.

### 6.15 — w2 in BFP8, finally gated deterministically (and §6.8's levels do not reproduce)

§6.13 shipped w2 in BFP8 on end-to-end WER alone — 1/1/0 wrong words of 298 across seeds 0/1/2 —
which §6.7 had already shown is **below the resolution of that gate**. §6.8's deterministic table
stops at `wqkv + wo BFP8 + sharded norm`, before w2. So the largest single precision drop in Block 1
shipped without the measurement every other precision change in this port was held to. Closed here.

Same session, same fixture, 44 teacher-forced real frames on 2 real prompts, only `WEIGHT_DTYPE`
differing:

| Block 1 decode | min PCC | mean WS | p90 WS | max WS | prefill WS |
|---|---|---|---|---|---|
| w2 **bf16** | 0.999756 | **0.94%** | 1.56% | 2.36% | 0.51 / 0.52 |
| w2 **BFP8** ← shipped | 0.999737 | **1.04%** | 1.55% | 1.97% | 0.46 / 0.50 |

**The cost is real and it is +0.10 pp on mean worst-sample.** Not scatter: the gate was re-run and
reproduced **bit-identically**, so it is deterministic and 0.10 is signal. But it appears on the mean
and nowhere else — p90 is flat, max improves, prefill improves. **Kept**, for 2.5 ms/frame (~5% of
frame time). Decision recorded so it can be revisited with a number rather than a WER reading that
could not see it.

**THE TWO CHANGES ARE NOW DECOUPLED, which they were not before.** w2 in BFP8 was only possible
because the codec stopped calling `ttnn.conv1d`; the codec fix now stands on its own. So w2 can be
reverted **alone** — one line, `ttnn_voxtral_gpt.py` `WEIGHT_DTYPE` — for ~2.5 ms/frame (RTF
0.60–0.65 → ~0.63–0.68), and it does **not** reintroduce the hang, which needed `conv1d`.

**⚠ THE BIGGER FINDING: THIS GATE WAS BEING READ TO 0.01 pp AND ITS PROMPT SPREAD IS 0.45 pp.**

Chasing why §6.8's levels would not reproduce turned up something that matters more than the w2
result above. Per prompt, 22 teacher-forced frames each, shipped build:

| case | voice | P | mean WS | p90 WS |
|---|---|---|---|---|
| 5 | fr_female | 118 | 1.20% | 1.65% |
| 7 | es_female | 158 | 1.34% | 2.34% |
| 1 | cheerful_female | 163 | **0.97%** | **1.38%** |
| 6 | de_male | 179 | **1.42%** | 2.28% |
| 8 | it_male | 184 | 1.35% | 2.16% |
| 0 | neutral_male | 200 | 1.02% | 1.55% |
| 2 | neutral_male | 312 | 1.07% | 1.38% |
| 3 | casual_female | 357 | 1.10% | 1.50% |

**Prompt choice moves mean worst-sample by 0.45 pp and p90 by 0.96 pp** — 4× and ~10× the +0.10 pp
that w2's precision drop costs, and larger than every §6.8 candidate that was accepted or rejected on
this metric. It is **not** a prompt-length effect: the shortest prompt (case 5, P=118) reads worse
than case 1 at P=163. §6.8 reported a 2-prompt pair to 0.01 pp and **did not record which two**.

This is §6.7's error repeated on the gate that was built to *replace* §6.7's gate: reading a number
to a precision far finer than its own spread. WER over 298 words could not resolve these changes;
neither can a 44-frame two-prompt worst-sample, **unless the prompts are held fixed**.

**What the gate does and does not support:**
- ✅ **Paired, same-session, same-prompts A/B.** Deterministic — a repeat run reproduced
  bit-identically. The w2 table above is valid on this basis.
- ❌ **Absolute levels.** ❌ **Cross-session comparison.** ❌ **Generalising an effect measured on
  one prompt pair** — w2's +0.10 pp is cases 0,2 only and may differ elsewhere.

**§6.8's levels are unreachable today, cause unidentified.** It records mean 0.84% / p90 1.06%; the
best of 28 prompt pairs across 8 cases is **0.99% / 1.38%**, and no single prompt reaches it either.
Ruled out by direct check, so nobody repeats them: the gate code (identical since §6.8 bar print
statements), `ttnn_voxtral_gpt.py` (identical bar docstring prose), the Block 1 reference
(untouched), the ttnn build (artifacts dated Jul 22/30, *predating* §6.8), the prefill rows (folding
them in gives 0.92%), and prompt selection (above). §6.9 reproduced 0.84%/1.06% at the time in a
second run, so it was not a one-off transcription slip. Unexplained.

**Practical rule going forward:** gate on **all 8+ prompts, recorded by index, both arms in one
session**. Anything else in this port has been over-read.

**Where quality stands overall, all measured on the current build:**

| | value |
|---|---|
| Block 1 decode | min PCC 0.999737, mean WS 1.04%, p90 1.55% |
| Block 2 velocity | PCC 0.99998164, semantic code exact, 3 of 74 codes differ |
| Block 3 codec | waveform PCC 0.999915 — unchanged by the hang fix, bit-identical after §6.14 |
| end-to-end | long-form WER 0.34% = **1 word of 298**, 15/15 on `[END_AUDIO]` |
| voice identity | PASS (same-voice pair most similar, F0 109–197 Hz) |

**One measurement trap in this gate, so nobody quotes it.** `--gate decode` prints ms/step, and it
read 69 ms for BFP8 against 59 for bf16 — i.e. BFP8 "slower", which is backwards. It runs a 3.4B
fp32 reference step on the CPU between device steps, starving host dispatch. **The accuracy columns
are what this gate is for; its timings are host-contended.** Real numbers come from the pipeline:
0.05 s/frame, long-form RTF 0.61–0.65 (§6.21).

### 6.16 — the precision stack re-gated on all 15 prompts: w2 is 77% of the cost for 15% of the win

§6.15 fixed the gate (all 15 prompts by default, pooled + per-case spread, case list printed). This
is the measurement it was fixed for, and it **overturns §6.8's central claim** that each BFP8 step was
free on mean/p90. Five arms, 15 prompts x 22 teacher-forced frames = 330 frames each, all in one
session, only the weight dtypes differing. `ms/step` for the three cumulative rows is the recorded
ladder; the two leave-one-out rows are derived from it and marked ~.

| Block 1 weights | ms/step | mean WS | p90 WS | max WS | min PCC |
|---|---|---|---|---|---|
| all bf16 | 45.8 | **0.86%** | **1.18%** | 3.01% | 0.999357 |
| BFP8 except w2 (FF + attn) | 31.4 | 0.93% | 1.35% | 3.05% | 0.998969 |
| BFP8 except attn (FF + w2) | ~32.2 | 1.13% | 1.69% | 4.07% | 0.998810 |
| BFP8 except FF (attn + w2) | ~40.0 | 1.13% | 1.69% | 3.73% | 0.998859 |
| **all BFP8 ← shipped** | **28.9** | **1.17%** | **1.75%** | 4.34% | 0.998045 |

**The total accumulated cost is +0.31 pp mean / +0.57 pp p90, and min PCC 0.999357 → 0.998045.**
§6.8 measured the increments as 0.86 → 0.86 → 0.84 — i.e. free, three times over — on two unrecorded
prompts where the prompt spread (0.44 pp) dwarfed each increment (~0.1 pp). The increments were never
free; the noise hid them, and **the accumulated total was never checked at all**. That is the concrete
demonstration of §6.15's warning, and it is why the gate now defaults to 15 prompts.

**Priced per millisecond, reverting one thing at a time from shipped:**

| revert | mean WS recovered | p90 recovered | ms/step given back | **pp per ms** |
|---|---|---|---|---|
| **w2** | **−0.24 pp** | **−0.40 pp** | 2.5 | **0.096** |
| wqkv + wo | −0.04 pp | −0.06 pp | 3.3 | 0.012 |
| FF1 + FF3 | −0.04 pp | −0.06 pp | 11.1 | 0.004 |

The three effects sum to 0.32 pp against a measured total of 0.31, so they are **additive** — and
**w2 is the whole story: 77% of the accuracy cost for 15% of the speed.** FF1+FF3 are the opposite,
66% of the speed for 13% of the cost, and 24x better value than w2. Note also that reverting attn and
reverting FF buy *identical* quality (1.13% / 1.69%) — so if 0.04 pp is ever wanted back, revert
**attn**, which costs 3.3 ms, never FF at 11.1 ms.

**APPLIED — w2 is back in bf16.** `ttnn_voxtral_gpt.py` `WEIGHT_DTYPE`. FF and attn stay BFP8; they
are cheap in accuracy and carry the bulk of the speedup. Measured after the flip, not predicted:

| | w2 BFP8 | **w2 bf16 ← ships** |
|---|---|---|
| Block 1 mean / p90 WS | 1.17% / 1.75% | **0.93% / 1.35%** |
| Block 1 min PCC | 0.998045 | **0.998969** |
| RTF, 15 cases | 0.60–0.65 | **0.64–0.69** |
| long-form WER | 1 word of 298 | **0 words of 298** |
| `[END_AUDIO]` termination | 15/15 | **15/15** |
| voice identity | PASS | **PASS** |

RTF landed at 0.64–0.69 against a predicted 0.63–0.68. **Do not read the WER 1 → 0 as the win** — at
298 words that is one word, and §6.7 showed this gate cannot resolve changes this size; it is
consistent with the deterministic improvement, not evidence of it. The gate table is the evidence.

Frame counts DO shift (case 6: 177 → 211 frames): changing w2's precision changes the emitted codes,
which changes trajectory length. That is the chaotic-trajectory property documented in §6.7, not a
defect — every case still terminates on `[END_AUDIO]` rather than running to the cap.

This reverses the recommendation given when w2 shipped and again in §6.15. Both earlier calls rested
on a 2-prompt estimate that put w2 at +0.10 pp; on 15 prompts it is **+0.24 pp, 2.4x larger**. The
2-prompt number was not wrong for cases 0,2 — it was not generalisable, exactly as §6.15 warned, and
I then generalised it anyway.

**What has NOT changed:** end-to-end quality is still good on the shipped all-BFP8 build — long-form
WER 1 word of 298, 15/15 `[END_AUDIO]`, voice identity PASS, Block 2 3-of-74 codes, codec
bit-identical. min PCC 0.998 is still far above tt_transformers' 0.981. This is a
margin-and-headroom argument, not a "the model is broken" argument.

### 6.17 — Block 2: BFP4 rejected, and three op-level wins that improve accuracy too

Block 2 is ~23 ms of a ~51 ms frame and the module docstring put it ~1.6x above a 13.4 ms weight-read
floor, so that was the target. First finding: **the matmuls are already AT the floor.** The five
weight matmuls measured 13.28 ms of the block's 19.24 (168–205 GB/s each by prefix timing), leaving
~6 ms of non-matmul work inside `_block` plus 3.6 ms outside it.

**BFP4 WEIGHTS — REJECTED, and it fails on both axes at once.** Never previously tried; `bfloat4_b`
exists and halves the bytes again (0.5625 vs 1.0625 B/param). 8 draws, per-weight granularity:

| weights in BFP4 | GB/frame | ms/frame | vel PCC | acoustic codes wrong |
|---|---|---|---|---|
| none — ships | 2.60 | 25.70 | 0.9999816 | **19 / 576** |
| w1, w3 (49%) | 2.00 | 22.99 | 0.9998741 | 93 / 576 |
| w2 (24%) | 2.30 | 25.68 | 0.9999413 | 49 / 576 |
| wqkv, wo (27%) | 2.27 | 25.28 | 0.9996631 | 117 / 576 |
| all five (100%) | 1.37 | 22.56 | 0.9994959 | **159 / 576** |

**8.4x the differing codes for 1.139x.** sdpa was rejected twice at 3x the codes for 1.147x — BFP4
is the same speed for nearly triple the damage. w2 in BFP4 is the clearest reject of all: 0.016 ms
(nothing) for 2.6x the errors.

**But the more useful half of that result is the timing.** Cutting bytes 47% (2.60 → 1.37 GB) returned
only 12% of the time (25.70 → 22.56 ms), where the bandwidth model predicts ~6.3 ms. **So weight bytes
are NOT the lever in Block 2 that they are in Block 1** — halving them returns about a quarter of what
the arithmetic promises, and the "1.6x above the floor" framing overstates how much of Block 2 that
floor governs. Do not plan future Block 2 work around byte counts.

**SHIPPED — three op-level changes, +1.19 ms/frame in isolation, and accuracy slightly BETTER:**

| | ms/frame | vel PCC | acoustic wrong |
|---|---|---|---|
| shipped before | 25.86 | 0.9999816 | 19 / 576 |
| A — silu fused into the w1 matmul | 25.70 | 0.9999816 | 19 / 576 |
| B — `SCALE` folded into wqkv's q rows | 25.58 | 0.9999851 | 16 / 576 |
| C — `_trunk` slices 36-wide, not 3072 | 24.77 | 0.9999816 | 19 / 576 |
| **A+B+C** | **24.67** | **0.9999851** | **16 / 576** |

- **A** — `activation="silu"` on the w1 linear. Bit-identical, and Block 1 has always done it; Block 2
  just never picked it up. Not the idea §6.8 rejected — that was fusing silu into the *multiply* via
  `input_tensor_a_activations`, which really is worthless.
- **B** — `1/sqrt(head_dim)` lives in the weight instead of a `multiply(s, SCALE)` per block call.
  21 launches gone, and **more accurate**: the scores round once instead of twice.
- **C** — the biggest, 1.09 of the 1.19. `_trunk` used to reshape to `[B,3,3072]`, slice position 0 at
  3072 wide, then project to 36. Now it projects the 6-row sequence first and reshapes/slices at 36
  wide: 85x less data moved, and the linear drops from batch-2 (which re-reads the whole weight per
  batch element) to batch-1 over 6 rows in one tile. **Same lesson as the codec in §6.13/6.14 — shift
  the narrow side** — which is now the third time it has paid here.

Gates: velocity PCC 0.99998164 → **0.99998510**, codes differing 3/74 → **2/74**, semantic exact,
long-form WER **0 of 298**, 15/15 `[END_AUDIO]`, voice identity PASS.

**⚠ THE ISOLATED WIN DOES NOT FULLY APPEAR END TO END.** Isolated Block 2 is −1.19 ms/frame
(deterministic, 30 reps). The pipeline over 8 comparable cases is **−0.36 ms/frame**, and the longest
case (463 → 485 frames) is −0.37. Per-case scatter is ±1 ms. I cannot account for the factor of ~3.
Quote **−0.4 ms/frame end to end**, ~0.7%, and treat the isolated number as evidence the change is
real rather than as the user-visible gain. All three are strictly fewer ops with no accuracy cost, so
they ship regardless of which number you prefer.

**A measurement trap this run walked into.** Raw RTF showed wild outliers (case 0 at 5.01, case 2 at
1.12) that survived a warm re-run and looked like a regression. They are not: changing Block 2's
numerics changes the emitted codes, which changes trajectory LENGTH (case 2: 443 → 465 frames, case
10: 211 → 271), and a new length is a new codec bucket that compiles ~5 conv programs at 1–5 s each.
Case 4 separately collapsed 73 → 8 frames — it is the chaotic one-word prompt. **Comparing RTF across
builds requires excluding cases whose frame count moved, or the compile lands in the number.** Same
family as the trap in §6.7 and the "case 0 RTF 1.89" one.

### 6.18 — the sharded norm's grid: §6.9's "8x2/8x4 are slower" is WRONG, and 32 cores ships

Prompted by "have we tried other `block_w` numbers, maybe smaller is faster?" -- which turned out to
rest on a false premise and still find something.

**`block_w` is not a free knob.** It is `DIM // cores // TILE`, so at 8 cores each core owns
3072/8 = 384 columns = 12 tiles and the kernel must walk 12. It only moves when the core count does,
and the two are the same decision.

**`subblock_w` IS free, was hard-coded at 4 in both blocks since the sharded norm shipped, and had
never been swept. It is inert.** Within a core count, 1/2/3/4 land within 0.02 ms/step of each
other, and at 8x1 in Block 2 subblock_w 4 and 1 measure *byte-identical* (24.628 both). `>= 6` does
not build at all -- a hard register-budget limit. So the knob nobody had tested turns out not to
matter, which is worth knowing precisely so nobody tests it again.

**What DOES matter is the core count**, contrary to §6.9. **THE COUNT MUST DIVIDE THE TILE COUNT.**
A 32x3072 tensor is 1 x 96 tiles, a tile is the indivisible unit, and `block_w` IS the quotient --
tiles per core. So only divisors of 96 are legal, which rules out 40, 56 and **64** (96/64 = 1.5).
That is why 8x8 does not build rather than merely running slowly.

Block 1, `_layer_step` x26, interleaved round-robin:

| grid | cores | 96/cores | legal? | isolated norm | **ms/step** |
|---|---|---|---|---|---|
| 2x1 | 2 | 48 | yes | 43.5 µs | 25.53 |
| 4x1 | 4 | 24 | yes | 43.9 µs | 24.84 |
| 8x1 ← was shipped | 8 | 12 | yes | 45.6 µs | 24.57 |
| 8x2 | 16 | 6 | yes | 48.2 µs | 24.45 |
| 8x3 | 24 | 4 | yes | — | 24.42 |
| **8x4 ← ships** | **32** | **3** | **yes** | **54.6 µs** | **24.41** |
| 8x5 | 40 | 2.4 | **no** | — | — |
| 8x6 | 48 | 2 | yes | — | 24.44 |
| 8x7 / 8x8 | 56 / 64 | 1.71 / 1.5 | **no** | — | — |

**IT IS NOT MONOTONE -- IT HAS A MINIMUM AT 32, AND I CLAIMED OTHERWISE.** The first sweep tested
2/4/8/16/32 by doubling, missed the non-power-of-two counts, saw a monotone curve and concluded "32 is
the largest that divides the work evenly." Both halves were wrong: **48 also divides evenly**, and it
is **slower** (24.44 vs 24.41, against a 0.007-0.011 ms spread, so ~4x the noise). 32 is the measured
optimum, not the largest legal grid. Plausibly the cross-core reduce and shard scatter start to
dominate once each core holds only 2 tiles, but that mechanism is a guess -- only the ordering is
measured.

**THE ISOLATED NORM RANKS THESE BACKWARDS.** 8x4 is the SLOWEST in isolation (54.6 µs vs 43.5) and
the FASTEST end to end. §6.9 concluded "the core count barely matters" and "8x2/8x4/8x8 are all
slower" from isolated numbers plus a partial 2-vs-8 end-to-end check — and the isolated measurement
is anti-correlated with the thing we care about. **A norm cannot be benchmarked alone**: it is ~16 µs
of reduction inside ~28 µs of resharding, and the reshard's cost depends on what consumes it next.

**Confirmed before believing it**: 7 interleaved rounds, round-robin so drift hits all arms equally.
Spread 0.007-0.013 ms per config and the ordering identical in every round — 8x1 24.571, 8x2 24.451,
8x4 24.407.

**Accuracy — paired, same session, all 15 prompts.** The 26-layer output differs by 1.0 absolute
between grids (a different cross-core reduce tree, amplified through 26 layers), which is the same
order as the whole w2 precision argument in §6.16, so it needed the real gate rather than a shrug:

| | 8x1 | **8x4** |
|---|---|---|
| mean worst-sample | 0.93% | **0.92%** |
| p90 worst-sample | 1.35% | **1.28%** |
| min PCC | 0.998969 | **0.999040** |
| per-case p90 spread | 0.55 pp | **0.44 pp** |

Better or equal on every column. Block 2 likewise: 24.628 → 24.465 ms/frame with acoustic codes
16/576 → 15/576 and velocity PCC 0.9999851 → 0.9999848. Gates after shipping both: flow velocity PCC
0.99998480, 2/74 codes, semantic exact, long-form WER **0 of 298**, 15/15 `[END_AUDIO]`, voice
identity PASS, 122/122 tests.

**Size, stated plainly.** Isolated: +0.163 ms/step in Block 1 and +0.163 ms/frame in Block 2, so
+0.33 ms/frame. End to end over 9 comparable cases: **−0.24 ms/frame**, ~0.5%. Same shortfall as
§6.17 and still unexplained. Quote the pipeline number. It ships because it is free — faster on a
stable measurement and no worse on any accuracy column — not because 0.5% matters.

**The transferable bit:** a config value that was copied once and never questioned (`subblock_w=4`)
turned out inert, while the one that *had* been measured and closed (the core count) was closed on
the wrong measurement. Re-check closed doors when the thing you closed them with was a proxy.

### 6.19 — `_QKV_SHARD`'s core count re-swept: inert above 6, and a different divisor set

Asked whether the qkv shard's core count had been tried. It had -- 8/12/16/24/48, recorded flat at
31.36-31.46 ms -- but on a build with w2 in BFP8 and the norm on 8x1, so it needed redoing against
what ships. It also only swept DOWN TO 8, and §6.18 had just shown a curve with an interior minimum.

5 interleaved rounds, mean ms/step, shipped build:

| cores | grid | cols/core | heads/core | ms/step | output diff |
|---|---|---|---|---|---|
| 1 | 1x1 | 6144 | 48 | 24.683 | 0.0 |
| 2 | 2x1 | 3072 | 24 | 24.520 | 0.0 |
| 4 | 4x1 | 1536 | 12 | 24.436 | 0.0 |
| 6 | 6x1 | 1024 | 8 | 24.416 | 0.0 |
| **8 ← ships** | **8x1** | **768** | **6** | **24.410** | **0.0** |
| 12 / 16 / 24 | 6x2 / 8x2 / 8x3 | 512 / 384 / 256 | 4 / 3 / 2 | 24.412 / 24.419 / 24.409 | 0.0 |
| 32 | 8x4 | 192 | **1.5** | **illegal** | — |
| 48 | 8x6 | 128 | 1 | 24.424 | 0.0 |

**Inert from 6 cores up** -- the whole 6-to-48 range sits inside the 0.020 ms within-config spread,
and 24's nominal 0.002 ms lead is 10x below the noise. **Below 6 it does cost**: one core is
+0.273 ms. So the old "it does not matter" needs the qualifier *at or above ~6*. Nothing to ship.

**The output is BIT-IDENTICAL at every count**, because this shard is pure data placement with no
reduction to reorder — the opposite of the norm, where changing the reduce tree moved the 26-layer
result by 1.0 absolute (§6.18).

**AND THE GRID NEVER REACHES THE CONSUMERS** — asked directly, since sharding differently ought to
affect `nlp_create_qkv_heads_decode`, `paged_update_cache` and `sdpa_decode`. It does not:
that op imposes its own output layout, and at input grids of 1 / 6 / 8 / 24 / 48 cores, `qh`, `kh`
and `vh` all come out **1 core, shard (32, 128)**, identically. So the two cache writes and
`sdpa_decode` never see this config; only the shard fill and the split op can. That is a structural
reason for the flatness, and a better one than "the aggregate did not move."

**A failed attempt to confirm that, worth recording as a method warning.** Timing the consumer chain
*in isolation* inverts the answer: it makes 1 core look 0.157 ms CHEAPER where the 26-layer step has
it 0.273 ms DEARER. Cause: the isolated probe feeds a pre-computed qkv tensor from DRAM instead of the
wqkv matmul's fresh output. The prefix split also produced a **negative** marginal cost for
`paged_update_cache V`, which is proof the method had broken down at these sizes rather than a small
inaccuracy. **Third instance today of the same failure** — after the norm grid (§6.18) and
`[gpt-05]`'s stale contrast — of an isolated measurement disagreeing with, and losing to, the
end-to-end one.

**Two different divisor sets, and it explains why 32 is the norm's optimum and illegal here.** The
qkv shard's unit is the HEAD: 6144 = 48 heads x 128 and the consumers want whole heads per core, so
the count must divide **48**. The norm's unit is the 32-wide TILE, so its count must divide **96**.
32 divides 96 but not 48. Read the unit before assuming a grid transfers between ops.

**Fixed a footgun while in there.** The config read `(TILE, _QKV_WIDTH // 8)` next to
`core_grid=CoreGrid(y=1, x=8)` -- one number written twice, where changing the grid and forgetting
the divisor yields a *silently wrong shard* rather than an error. Both now derive from `_QKV_GRID_X`.

**And a correction to `NOTES.md [gpt-05]`**, which argued the norm's count "DOES matter: 16.2 us on 8
cores against 21.2 on 32" as a contrast to this one. Those are isolated-norm timings, the metric
§6.18 showed is anti-correlated with end-to-end. The real contrast is much narrower: both grids are
nearly free, the norm's end-to-end spread is 0.16 ms and this one's is 0.02.

### 6.20 — sharding control DOES unlock something: 0.454 ms/frame, bit-exact

Asked whether hand-rolling `nlp_create_qkv_heads_decode`, `paged_update_cache` and `sdpa_decode`
would be faster, since we could then control the sharding. **Hand-rolling: no, and it was already
tried. The underlying intuition: yes, and it was worth the largest single win of the session.**

**Why not hand-rolling.** A hand-rolled decode interior was built and measured, and the fused
decode-native path beat it by **6.6 ms/frame** ([gpt-05]) — more than double the entire 2.34 ms those
three ops cost today. Block 2's hand-rolled head split measured 158 µs against the fused op's 122
([flow-10]). Cost here is per-LAUNCH, and every hand-rolled decomposition trades one op for three to
eight, so it is the wrong direction by construction.

**But the sharding really was blocking a fused op.** `ttnn.experimental.paged_fused_update_cache`
exists, writes both caches in one call, and we had never used it. It refuses K and V on the same
core — and `nlp_create_qkv_heads_decode` puts q, k and v all on core (0,0) by default. The chain to
unblock it, each link forced by the next:

1. `nlp_create_qkv_heads_decode(..., overlap_qk_coregrid=False)` puts q on (0,0), k on (1,0).
2. That flag asserts `head_dim % shard_width == 0` — a whole head per core. `_QKV_WIDTH/48 = 128` is
   **exactly one head**, and the only feasible width: 64 columns would need 96 cores. So the qkv
   shard moves 8 → 48 cores, which the §6.19 sweep had already shown costs nothing (+0.014 ms, inside
   the 0.020 ms spread).
3. With k on (1,0), rope breaks — **silently, returning 3.376e+38, uninitialised L1** — because
   `_ROPE_SHARD` pins cos/sin to (0,0) and rope reads its table from the tensor's own core. Fixed with
   a second cos/sin pinned to (1,0). Built once per FRAME, so ~4 extra launches.
4. `paged_fused_update_cache` then validates, and 52 cache-write launches per frame become 26.

| | ms/step | vs shipped | output |
|---|---|---|---|
| shipped: overlap=True, 2x update | 24.406 | — | — |
| overlap=False + k-rope, 2x update | 24.388 | +0.017 | **bit-exact** |
| **overlap=False + k-rope, 1x fused update** | **23.952** | **+0.454** | **bit-exact** |

Spread 0.004–0.014 ms, so 0.454 is ~30x the noise. Pipeline: **−0.44 ms/frame**, long-form RTF
**0.61–0.65**, WER 0 of 298, 15/15 `[END_AUDIO]`, voice identity PASS, 122/122 tests. Per case, on the
14 whose frame count is unchanged, **14 improved and 0 got worse** — deltas −0.00 to −0.04 RTF.

**Bit-exactness, and how far the evidence actually goes.** `--gate decode` on 15 prompts prints
byte-identical stats, but those are ROUNDED, so that is equality to 2-3 significant figures, not proof.
The real evidence is three direct comparisons: the fused cache write is identical at **18 positions**
(0, tile boundaries 31/32/33, 63/64, 127/128, 200, 311/312/313, 511/512, 767, 1023); Block 1 decode is
identical at **every one of 64 sequential steps** with a growing cache; and case 10's real frame loop
is identical **frame-for-frame over 221 frames**, hidden state and codes both.

**AND THE ISOLATED AND PIPELINE NUMBERS AGREE FOR ONCE** — 0.454 against 0.44, where §6.17 and §6.18
each showed a factor of ~3 shortfall. The difference is what the change *does*: removing launches is
layout-neutral, so nothing downstream re-optimises around it. Grid and blocking changes alter how
every consumer receives its data, which is exactly why they measure differently in isolation.

**⚠ SUPERSEDED BY §6.22.** The chain above (48-core shard, `overlap_qk_coregrid=False`, a second
rope table) works and is what shipped first, but it moves **K**. Moving **V** instead reaches the same
place with none of the coupling, measures the same end to end, and is what ships now. The fused write
itself, and everything above about why it needs non-overlapping cores, still stands.

### 6.21 — how to quote RTF, and a harness confound that cost an hour

Asked why RTF read 0.61–0.71 when it had been 0.64–0.69. **Both numbers were badly derived by me, in
different ways, and neither was comparable to the other:**

- **0.64–0.69** was eyeballed off a `tail -20` that showed only 10 of the 15 cases. The real range for
  that build, excluding case 0, was 0.62–0.69.
- **0.61–0.71** came from a filter `rtf < 1.2`, which dropped case 0's first-call compile but kept
  case 4 — the chaotic one-word prompt at 43 frames — as the top end.

So the "widening" was two inconsistent filters, not the model. Like for like, **14 of 14 comparable
cases improved and none got worse.**

**RTF is dominated by utterance LENGTH, not by compute speed**, because it is wall/audio and the fixed
costs (prefill, codec, first-call compiles) amortise over however much audio there is:

| bucket | frames | RTF |
|---|---|---|
| long-form | >= 100 | **0.61–0.65** |
| short | 20–99 | 0.62–0.71 |
| case 0 | any | 1.61 — carries the first-call compile |

**THE CONVENTION, from here on: quote long-form RTF (>= 100 frames).** It is the only bucket where
fixed costs are amortised enough to reflect per-frame speed. Quote short and case-0 numbers separately
or not at all, and never mix filters between two builds.

**And a harness confound worth more than the reporting fix.** Case 10's frame count moved 207 → 220
between two builds, reproducibly, which looked like a real numerical divergence and took an hour to
chase. It is not: **run case 10 ALONE and both builds give 220.** Its length depends on what ran
before it in the same process. `generate()` reseeds per call, so it is not RNG — the pipeline object
and its KV cache are reused across cases. **Determinism is not independence:** two runs of one build
match exactly (verified, 0 of 15 cases differ), which is what made the artifact look trustworthy.
**So a changed frame count in a multi-case run is not evidence of a changed model.** Confirm it by
running that case alone before believing it.

### 6.22 — move V, not K: same speed, none of the coupling (from the xtts branch)

Asked to look at how `lserbedzija/xtts-gpt-ttnn` does the same fusing, since it is "a bit different".
It is, and it is better. `xtts_v2/tt/ttnn_xtts_gpt.py` keeps `nlp_create_qkv_heads_decode` at its
default and **moves V to core (1,0)** with one `to_memory_config`, where §6.20 moved **K** via
`overlap_qk_coregrid=False`.

That branch is GPT-2 with learned positional embeddings, so it never had the RoPE problem that made
moving K awkward here. **Moving V sidesteps it entirely** — V never passes through RoPE.

| | moves | launches added | isolated ms/step | pipeline mean RTF, 12 like-for-like cases |
|---|---|---|---|---|
| baseline, 2 writes | — | — | 24.405 | 0.6539 |
| §6.20, `overlap_qk_coregrid=False` | K | 4 / frame | 23.953 (+0.452) | 0.6485 |
| **§6.22, `to_memory_config(v)`** | **V** | **26 / frame** | 24.000 (+0.405) | **0.6483** |

Both bit-exact. Moving K is 0.047 ms/step faster in isolation — **and indistinguishable end to end**
(0.6485 vs 0.6483, where per-case RTF resolution is 0.01). The 22 extra launches cost ~2.1 µs each,
much less than the 5-20 µs a launch usually runs here, because moving V is an 8 KB hop between
adjacent cores.

**So V ships, for robustness at no measurable cost.** Moving K carried two hazards, both now gone:
- `_QKV_SHARD` had to be **48 cores** (a whole head per core) or `overlap_qk_coregrid=False` would not
  build — a load-bearing constraint on a value §6.19 had just established as inert, i.e. exactly the
  kind of coupling that reads as free to change and is not.
- K then went through RoPE on a core whose cos/sin table lived elsewhere, which **does not raise** —
  it returns 3.4e38 from uninitialised L1. A second table fixed it, but the failure mode stayed.

Moving V needs neither: the shard width is free again and there is one rope table.

**A reporting slip worth recording, since §6.21 had just warned about it.** My first read of this
comparison said move-V was *worse* — mean RTF 0.7227 → 0.7302 over 13 cases. That mean included case
0, whose 1.6 RTF is a first-call compile, and it swamped everything. Excluding case 0 and any case
whose frame count moved gives 0.6485 vs 0.6483. **I wrote the convention an hour earlier and then
broke it in the next measurement.**

Gates: `--gate decode` byte-identical on all six columns, long-form WER 0 of 298, 15/15
`[END_AUDIO]`, voice identity PASS, 122/122 tests, long-form RTF 0.61–0.66.

### 6.23 — `rotary_embedding_llama_fused_qk`: REJECTED, wrong rotation convention for our weights

The two `rotary_embedding_hf` calls are 52 launches a frame, so a fused q+k pair was the obvious next
target after §6.20/§6.22. It exists, it is faster, and **it cannot be used without undoing a
deliberate weight decision.**

**IT IMPLEMENTS THE INTERLEAVED CONVENTION; WE USE HALF-SPLIT.** `get_rot_transformation_mat`'s own
docstring gives it away — the trans_mat "pairs ADJACENT dimensions ... [0,1] -> +1 at (0,1), -1 at
(1,0)", i.e. interleaved-pair rotation on `(r0,i0,r1,i1,...)`. Confirmed numerically rather than read
off a comment, by feeding one vector written both ways and comparing against explicit torch
references for each convention:

| fused op output vs | max diff |
|---|---|
| torch **interleaved-pair** reference | **9.6e-03** (bf16-level — correct) |
| torch **half-split** reference | **3.67** (wrong) |

Our checkpoint is Mistral-native and therefore interleaved, but `interleaved_to_halfsplit` permutes
wq/wk ONCE at load so the easy `rotate_half` form applies (module docstring / NOTES.md [gpt-01]).
Using the fused op means **reverting that permute** — and rope getting this wrong does not raise, it
produces fluent nonsense.

**What it would buy, measured at our shapes:**

| | µs/layer | ms/frame |
|---|---|---|
| 2 x `rotary_embedding_hf` (ships) | 33.5 | 0.871 |
| 1 x `rotary_embedding_llama_fused_qk` | 24.4 | 0.635 |
| | | **+0.236 (1.373x on this pair)** |

**What it would cost.** Un-permuting wq/wk, the riskiest class of change in this port. q and k back on
**disjoint cores** — the exact coupling §6.22 just removed. cos/sin rebuilt per frame as
`[1, 2*batch, 32, head_dim]`, height-sharded across both cores (it must be HEIGHT_SHARDED — it asserts).
A trans_mat sharded over >= q_cores + k_cores with (32,32) shards. And it is **not bit-exact**: the
rotation runs through a tile matmul.

**0.236 ms/frame is 0.46% of a frame. Not worth any of that**, and notably worse value than the fused
cache write, which bought 0.405 ms bit-exactly for one inline reshard and no convention change.

**Recorded because the convention table is the useful part**, and it is not written down anywhere
obvious: `rotary_embedding_hf` wants **half-split**, `rotary_embedding_llama` and
`rotary_embedding_llama_fused_qk` want **interleaved** via a trans_mat. `tt_transformers/tt/attention.py`
keeps `_hf_rope_decode` and `_mllama_rope_fused_qk_decode` as separate paths for exactly this reason —
the fused variant sits only on the llama side. If our weights were ever left un-permuted for another
reason, this becomes free money; until then it is not.

### 6.24 — fusing w1 and w3 is 4x SLOWER: matmul bandwidth collapses past ~9216 output columns

Asked whether `h@w1` and `h@w3` should be one matmul, the way qkv is. It should not, and the reason
turned out to be much bigger than the one on record.

**The prize was small before measuring anything.** w1 and w3 are each 3072x9216 BFP8 = 28.7 MB and run
at **190 GB/s — 98% of this device's 194 GB/s ceiling**. Together they read 57.4 MB, which at the
ceiling is 310.1 µs against the 316.5 they take. **The entire theoretical gain from removing one launch
is 6.4 µs/layer = 0.17 ms/frame.** Contrast the qkv fusion, which paid 1.449x because `wkv` alone was
2048 wide and cost the same as the 4096-wide `wq` — there was launch overhead to reclaim. Fusing pays
when one of the pair is too NARROW to earn its launch, and 9216 is not narrow.

**Measured, one layer's MLP, same 57.4 MB of weight either way:**

| | ops | µs | vs A | result |
|---|---|---|---|---|
| A — `linear(w1, silu)` + `linear(w3)` + multiply ← ships | 3 | **313.9** | 1.000x | — |
| B — fused + 2 slices + silu + multiply | 5 | 1253.3 | **0.250x** | same values |
| C — fused + `ttnn.swiglu` | 2 | 1253.8 | **0.250x** | same values |

**Both fused forms are 4x slower, and C has one op FEWER than what ships** — so this is not about op
count at all. The fused matmul itself is the problem:

| | MB read | µs | GB/s | % of ceiling |
|---|---|---|---|---|
| two separate 3072x9216 | 57.4 | 313.9 | **192** | **99%** |
| one fused 3072x18432 | 57.4 | 1253.5 | **48** | **25%** |

**Identical bytes, 4x the time. Matmul efficiency collapses somewhere between 9216 and 18432 output
columns**, and that is the transferable result: 3072 (w2), 6144 (wqkv) and 9216 (w1/w3) all hold the
ceiling; 18432 does not. **Any future fusion that pushes N past ~9216 should be assumed to collapse
until measured.** This also retro-explains why the qkv fusion was safe — it lands at 6144.

**Note this contradicts the recorded Block 2 number** (0.998x, and 0.951x charging the output split),
which is far milder than 0.250x at the same weight shapes. Block 2's measurement was on the whole
block rather than the MLP alone, so the fused matmul's cost was diluted; either way the direction is
the same and the isolated figure is the honest one for this decision.

**Two incidental findings about `ttnn.swiglu`**, since it looked like the elegant answer (one matmul
plus one fused SwiGLU = 2 ops, no split, no separate silu):
- it needs a **4-D** tensor. On `[1,1,18432]` it throws `ShapeBase[] index out of range. 3 not in [...]`
  from `shape_base.cpp:16` — it indexes `shape[3]` unconditionally.
- it applies SiLU to the **second** half and multiplies by the first, so our MLP needs the fused weight
  ordered `[w3 | w1]`, not `[w1 | w3]`. Both orders were measured; both collapse.
- and it operates on the PHYSICAL tile rows, so a `[1,1,1,18432]` input yields `[1,1,32,9216]` — the
  31 padding rows come through and would need slicing off.

Also worth stating: the fused weight is a **duplicate** of w1 and w3 concatenated — 57 MB/layer,
**1.46 GB over 26 layers** against a 3.9 GB working set. Even at parity it would be a poor trade.

### 6.25 — wo gets a program config: +0.196 ms/frame bit-exact, and the knob is `in0_block_w`

§6.24's ceiling argument produced a target. Each decode linear's time is bytes/ceiling plus overhead,
and the overhead is all a restructuring can attack — **only wo had any**:

| weight | bytes | floor µs | actual | overhead |
|---|---|---|---|---|
| wqkv | 20,054,016 | 103.4 | 103.2 | −0.2 → 0% |
| **wo** | 13,369,344 | 68.9 | 85.9 | **+17.0 → 20%** |
| w1 / w3 | 30,081,024 | 155.1 | 158.1 / 158.4 | +3.0 / +3.3 → 2% |
| w2 | 56,623,104 | 291.9 | 284.1 | −7.8 → −3% |

**THE KNOB IS `in0_block_w`, NOT THE CORE COUNT — my hypothesis was wrong.** I went in expecting
N=3072's 96 tiles splitting 1.5-per-core over 64 cores to be the problem. It is not: 8x8 / 8x6 / 8x4
measure 68.2 / 68.1 / 68.5 µs, i.e. the grid barely matters. What matters is how many of K's 128 tiles
load per inner iteration:

| in0_block_w | 1 | 2 | **4** | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| µs | 152.0 | 83.2 | **68.1** | 73.7 | 80.5 | 89.4 |
| GB/s | 88 | 161 | **196** | 182 | 166 | 150 |

The default runs at 162 GB/s; `in0_block_w=4` runs at **196 — the ceiling**. Both directions lose.

**TWO OPTIONS, and the bit-exact one ships:**

| config | µs | vs default | ms/frame | accuracy |
|---|---|---|---|---|
| default | 82.3 | 1.000x | — | baseline |
| **8x4, in0_block_w=2 ← SHIPS** | 74.7 | 1.102x | **+0.196** | **byte-identical on all six gate columns** |
| 8x6, in0_block_w=4 | 68.1 | 1.209x | +0.370 | mean +0.02, p90 +0.01 pp; min PCC 0.999040 → 0.999219 (better); max 3.05% → 4.76% |

The faster one costs 0.054 pp of mean per ms — the same ballpark as w2's 0.096 that §6.16 rejected —
and takes the worst sample from 3.05% to 4.76%. Max is an unstable order statistic and not decisive on
its own, but 0.174 ms/frame is 0.34% of a frame and not worth the argument. **`in0_block_w=4` is left
documented and available** if that trade is ever wanted; nothing else needs to change to take it.

Verified: 122/122 tests, decode gate byte-identical, long-form WER 0 of 298, 15/15 `[END_AUDIO]`,
voice identity PASS. Pipeline **−0.0024 RTF = −0.19 ms/frame** against the isolated +0.196 — they agree,
as in §6.22, because this changes no layout that a consumer has to re-optimise around.

**DECODE ONLY.** `per_core_M=1` assumes M is one tile, true of a single decode position and false of
prefill's S rows, so `_layer` deliberately has no program config.

**⚠ THE NEAR-MISS, and it is the most useful thing here.** The first sweep flagged every
`in0_block_w=4` variant as **WRONG** — they differed from the DEFAULT by 5.3e-03 relative, and those
were exactly the fast ones. Checking against a torch **fp64** reference instead showed every config
sitting at the *same* 6.640e-04 from truth, with PCC 0.9999742 vs the default's 0.9999745. **The
default is not ground truth.** Comparing a variant against it rather than against the real answer
almost discarded a genuine 1.2x as a correctness failure.

This also retires the older "hand-tuned matmul program configs measured SLOWER (169 vs 193 GB/s)"
finding as scope-limited rather than wrong: that was measured on **wq**, already at 94% of its floor.
Sweeping an op with no headroom finds none — which is why §6.24's floor table is the right way to pick
what to sweep.

### 6.26 — device tracing re-measured a third time: same answer, +0.35 ms/frame, still not taken

Re-run against the shipped config, because §6.6's numbers predate everything from §6.17-§6.25 and this
port's own rule is to re-measure rather than quote. Both probes are unchanged from the scratchpad; only
the build under them moved.

| | untraced (ships) | traced | delta | correct? |
|---|---|---|---|---|
| Block 1, 26-layer decode step | 24.86 ms | 24.69 ms | **+0.17 ms (1.007x)** | min PCC 0.999883 and worst-sample 1.28% IDENTICAL |
| Block 2, whole 7-step solve | 21.11 ms | 20.93 ms | **+0.18 ms (1.009x)** | integer codes bit-identical to untraced |
| **total** | | | **+0.35 ms/frame (~0.7%)** | |

**The verdict is unchanged and the number barely moved** — §6.6 measured +0.36 ms/frame on a build
10 ms/step slower. The absolute gain held while the base shrank, so as a *fraction* it grew slightly
(0.58% → 0.68% on Block 1). That makes sense: tracing recovers host command submission, which scales
with op COUNT, and today's work removed ~24 launches of ~476 — about 5%.

**A correction to my own reasoning for running this.** I suggested re-measuring "now that the launch
count has dropped", which is backwards: fewer launches means LESS host dispatch for a trace to recover,
so the expected direction was down. The legitimate reason was only that the recorded numbers described a
config that no longer exists. They turned out to still hold.

**Why it stays out at a gain comparable to things that shipped today.** 0.35 ms/frame sits right
alongside the fused cache write (0.405), the Block 2 op wins (0.44 isolated) and wo's program config
(0.196). On magnitude alone it is shippable. The difference is the FAILURE PROFILE, and it is not a
matter of taste:

| shipped today | how it fails if wrong |
|---|---|
| fused cache write | `paged_fused_update_cache` asserts on overlapping cores — **raises** |
| wo program config | wrong `per_core_M`/shape — **raises** |
| move V to core (1,0) | wrong placement — **raises** |

| tracing | how it fails if wrong |
|---|---|
| buffers held by POINTER | a fresh upload is silently ignored and the previous frame replays — **silent** |
| warm-up/capture write the KV cache | corrupts the prompt unless aimed at a scratch row; measured decode PCC 0.9998 → 0.86 with **no error raised** |
| capture ORDER across the two blocks | all warm-ups must precede either capture, or the second **hangs** |
| an exception inside a capture | `close_device` hangs and **every later run on the card blocks** — trap #1, cost ~20 min and a kill by PID once |
| `trace_region_size` | required on every caller of `open_device` |

Three of those are silent and one wedges the board. Everything shipped today fails loudly. **0.7% is
not worth converting four loud failure modes into three silent ones**, and that is the whole argument —
not the size of the gain.

**If it is ever wanted anyway**, tracing ONE block avoids the cross-block capture-order constraint for
about half the gain, and the probes (`probe_trace_b1.py`, `probe_trace_b2.py`) are current and ready.

### 6.27 — the overhead map, and what it found: sdpa_decode was 90% setup

§6.24's floor method applied to EVERY decode op, not just the matmuls: measure isolated, compute a DRAM
byte floor, rank by absolute overhead. That map answered "is there overhead left" precisely.

**The matmuls are finished.** wqkv, w3 and w2 now measure FASTER than a 194 GB/s floor (−1.7, −7.5,
−11.2 µs), which also says that ceiling is slightly conservative — w2 implies 202 GB/s. w1 is +3.0 and
wo +6.4 after §6.25. Nothing there.

**Everything left is small ops**, and the isolated overheads are large in relative terms:

| op | floor µs | actual | overhead | ×26 |
|---|---|---|---|---|
| **sdpa_decode** (pos=312) | 6.6 | 68.1 | **+61.6** | **1.60 ms** |
| ffn norm ×3 | 0.1 | 53.3 | +53.2 | 1.38 ms |
| multiply(g, w3) | 0 | 29.7 | +29.7 | 0.77 ms |
| add residual ×2 | 0 | 28.9 / 26.7 | +55.6 | 1.45 ms |
| the other 9 small ops | ~0.2 | ~185 | ~+185 | ~4.8 ms |

**But the reachable total is far smaller than that sums to.** Isolated ops sum to 29.930 ms against a
shipped step of 23.803 — 6 ms is already overlapped away — and the byte floor is 20.309 ms. So **at most
3.494 ms/frame, 15% of the step, is reachable**, and an op's isolated overhead is an upper bound on
that op rather than a promise.

**SHIPPED — `SDPAProgramConfig` on sdpa_decode, +0.673 ms/frame isolated, bit-exact.** The shipped call
passed no program config. Diagnosing the 10× gap first mattered: a **31× larger cache costs 21% more
time** (pos 32→1000, 0.12→3.91 MB, 68.4→82.5 µs), so ~62 of the 68.6 µs is FIXED setup, not the read.

| | µs | vs default |
|---|---|---|
| default | 68.6 | 1.000x |
| **k_chunk=512, grid=8x2 ← ships** | **42.2** | **1.625x** |
| k_chunk=256, grid=8x2 | 38.7 | 1.772x |
| k_chunk=32, any grid | 48–65 | **1.0e+00 rel — genuinely broken** |

**⚠ ONLY A POSITION SWEEP MAKES THIS SAFE.** Verified at 11 positions from 64 to 1000, spanning the
chunk boundary at 511/512/513, against both the default and a torch fp64 reference:

| config | bit-exact at | worse than default vs fp32 | ms/frame |
|---|---|---|---|
| **k=512 8x2** | **11/11** | 0 | 0.673 |
| k=512 8x1 | 6/11 | 0 | 0.706 |
| k=256 8x2 | 3/11 | **3** | 0.825 |

**`k=256` looks fine at pos=312 and is worse than the default at 480, 511 and 700.** The decode gate
pins ONE position per case, so it would have passed and degraded long utterances silently. Any
position-dependent config needs a position sweep, not a gate run.

Verified: 122/122 tests, decode gate byte-identical on all six columns, long-form WER 0 of 298, 15/15
`[END_AUDIO]`, voice identity PASS. Pipeline **−0.33 ms/frame** against +0.673 isolated — about half
shows up, since sdpa partly overlaps neighbouring work.

**REJECTED — folding the residual adds into their matmul as a bias.** In decode M=1, so the residual is
a `[1,3072]` row vector, exactly a bias. It works, and it is worth almost nothing:

| form | µs | vs 2-op | numerics |
|---|---|---|---|
| w2: `add(x, linear(u, w2))` ships | 285.4 | — | — |
| w2: `linear(u, w2, bias=x)` | 285.2 | 1.001x | **bit-exact, no gain** |
| wo: `add(x, linear(...))` ships | 79.4 | — | — |
| wo: `linear(a, wo, bias=x, prog_cfg)` | 75.4 | 1.054x | 3.0e-03 rel, not bit-exact |
| wo: `linear(a, wo, bias=x)` no prog_cfg | 86.5 | 0.918x | bit-exact but SLOWER |

**w2's add is already entirely free** — hidden behind a 285 µs matmul. Only wo's is visible, worth 4.0 µs
(0.104 ms/frame), and only in the form that loses bit-exactness. Not taken. This is also the clearest
demonstration of why the isolated map overstates: the same op costs 28.9 µs alone and 0 µs in place.

### 6.28 — the DRAM-sharded matmul, re-opened with tuned blocking: §6.9's rejection HOLDS

The last lead from §6.27's overhead map. The norm emits a width-sharded activation and the shipped path
immediately unshards it (11.7 µs/layer, 0.30 ms/frame) so an ordinary matmul can take it.
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` is the one config that wants the sharded form
directly. §6.9 rejected it at 125.4 µs against 100.9 — **with untuned blocking**, and §6.25 had since
shown blocking is worth 1.2x on `wo`. So the rejection looked like it might be an artifact.

**It is not. Measured from `hs` to the tensor `nlp_create_qkv_heads_decode` consumes, so each path
carries everything it needs:**

| weight placement | in0_block_w | µs | vs shipped | numerics |
|---|---|---|---|---|
| **Path A — unshard + normal matmul (ships)** | — | **108.6** | **1.000x** | — |
| 8c DRAM-sharded | 1 | 303.7 | 0.357x | 3.0e-03 rel |
| 8c DRAM-sharded | **6** | **180.7** | **0.601x** | 3.0e-03 rel |
| 8c DRAM-sharded | 12 | 181.7 | 0.598x | 3.0e-03 rel |
| 8c L1-sharded | — | — | **Out of Memory** | — |
| 32c DRAM-sharded | — | — | TT_FATAL on the weight reshard | — |
| 32c L1-sharded | 3 | 138.4 | 0.784x | **1.4e+14 — silently WRONG** |

**Tuning `in0_block_w` was worth 1.68x** (303.7 → 180.7), so the premise was right — §6.9 did leave it
untuned. **The conclusion was still wrong: even fully tuned it is 1.66x SLOWER.** §6.9's axis argument
survives intact — width-sharding in0 splits the matmul's CONTRACTION dimension, so each core forms only a
partial sum and a cross-core reduce over the full 6144-wide output follows. Better blocking makes that
reduce cheaper; it does not remove it.

**Two incidental findings, both silent failure modes:**
- an **L1-resident** width-sharded weight is Out of Memory at 8 cores — 13.4 MB over 8 shards is 1.7 MB
  per core. This confirms §6.9's note that an L1 weight "would be capped anyway", with the actual error.
- the **32-core L1** variant BUILDS, RUNS, and returns **1.4e+14** — garbage, no exception. Another entry
  for the list of ttnn configurations that are silently wrong rather than loud.

**So the norm's reshard pair is now closed three ways** — sharded straight to a normal matmul (8.94 vs
5.32 ms, §6.9), weight-only sharding (not expressible, §6.9), and this config tuned (1.66x slower). The
11.7 µs unshard stays.

**Process note, since it took four probe iterations and two were my own fault.** Attempt 1 passed no
output `memory_config` → *"Output memory config must be sharded"*, and I wrote **"§6.9's rejection
holds"** on the strength of it. It had not been tested at all. Attempt 2 left the weight interleaved →
*"Input B memory layout must be WIDTH_SHARDED"*. Only attempt 3 measured anything. **A config that fails
to BUILD has told you nothing about whether it is fast** — and I recorded the opposite once already
today (§6.27's `to_memory_config` reading). Read the assertion before writing the conclusion.

### 6.29 — Block 2's overhead map: 40% reachable, and NONE of it is in the matmuls

§6.27's method applied to Block 2. Every op isolated at its real shape and memory config, floor =
bytes / 194 GB/s, ranked by overhead x how many times it runs. `_block` runs **21x** a frame (3 layers
x 7 Euler steps); the `_solve`/`_trunk` ops run 7x; two run once.

Block 2 is shaped nothing like Block 1. **All five matmuls are AT the roofline** -- wqkv measures 1.8 us
BELOW its floor, w3 7.7 us below (i.e. above 194 GB/s, consistent with the true ceiling being ~202),
w1 +2.9, wo +12.6, w2 +23.8. So there is no matmul work left, and yet **8.99 of 22.55 ms/frame is not
weight reads** -- 40%, against Block 1's 12%. Every recoverable microsecond is in a small op.

| op | x/frame | floor us | actual us | ovhd us | ms/frame | recoverable |
|---|---|---|---|---|---|---|
| `nlp_create_qkv_heads` (transpose_k) | 21 | 0.0 | 122.1 | 122.1 | 2.564 | **2.564 ms** |
| `semantic_code` | 1 | 0.0 | 1241.3 | 1241.3 | 1.241 | **1.241 ms** |
| CFG combine + Euler (2 slice 3 mul 2 add) | 7 | 0.0 | 148.2 | 148.2 | 1.038 | 1.038 ms |
| ffn norm x3 (shard+norm+DRAM) | 21 | 2.0 | 47.5 | 45.5 | 0.998 | 0.955 ms |
| reshape+permute+reshape unfold heads | 21 | 0.0 | 43.7 | 43.7 | 0.918 | 0.918 ms |
| `_trunk` 3 reshape + concat + reshape | 7 | 0.0 | 129.8 | 129.8 | 0.909 | 0.909 ms |
| matmul q(row-fold) @ kT | 21 | 0.0 | 30.5 | 30.5 | 0.641 | 0.641 ms |
| add residual (ffn) | 21 | 0.0 | 26.9 | 26.9 | 0.565 | 0.565 ms |
| add residual (attn) | 21 | 0.0 | 26.7 | 26.7 | 0.560 | 0.560 ms |
| multiply(g, w3) | 21 | 0.0 | 26.6 | 26.6 | 0.558 | 0.558 ms |
| **linear w2** 9216x3072 BFP8 | 21 | 155.1 | 178.9 | 23.8 | 3.757 | 0.500 ms |
| attn norm 2/3 rms_norm sharded | 21 | 0.0 | 20.0 | 20.0 | 0.419 | 0.419 ms |
| softmax numeric_stable | 21 | 0.0 | 19.2 | 19.2 | 0.403 | 0.403 ms |
| `_trunk` final norm x3 | 7 | 2.0 | 53.0 | 51.0 | 0.371 | 0.357 ms |
| attn norm 1/3 -> width-sharded | 21 | 1.0 | 17.5 | 16.5 | 0.367 | 0.346 ms |
| matmul a @ v | 21 | 0.0 | 15.6 | 15.6 | 0.328 | 0.328 ms |
| **linear wo** 4096x3072 BFP8 | 21 | 68.9 | 81.5 | 12.6 | 1.712 | 0.265 ms |
| `_trunk` linear acoustic_codebook_output | 7 | 1.1 | 33.0 | 31.9 | 0.231 | 0.224 ms |
| attn norm 3/3 -> DRAM | 21 | 1.0 | 10.5 | 9.5 | 0.220 | 0.199 ms |
| `_solve` concat([x,x]) + typecast | 7 | 0.0 | 25.1 | 25.1 | 0.176 | 0.176 ms |
| `_trunk` reshape + slice (36 wide) | 7 | 0.0 | 22.1 | 22.1 | 0.155 | 0.155 ms |
| `_solve` linear input_projection | 7 | 1.1 | 17.5 | 16.4 | 0.122 | 0.115 ms |
| `_solve` typecast v -> fp32 | 7 | 0.0 | 9.7 | 9.7 | 0.068 | 0.068 ms |
| **linear w1** 3072x9216 BFP8 +silu | 21 | 155.1 | 158.0 | 2.9 | 3.317 | 0.061 ms |
| `_solve` linear llm_projection | 1 | 51.7 | 65.2 | 13.5 | 0.065 | 0.014 ms |
| **linear wqkv** 3072x6144 BFP8 | 21 | 103.4 | 101.6 | **-1.8** | 2.134 | -0.037 ms |
| **linear w3** 3072x9216 BFP8 | 21 | 155.1 | 147.4 | **-7.7** | 3.095 | -0.161 ms |
| TOTAL (isolated, sums high) | | | | | 26.931 | 13.379 ms |

Shipped Block 2 frame end to end: **22.545 ms**. Weight-read floor **13.553 ms**. The isolated sum
(26.93) exceeds the real frame by 4.4 ms, which is the overlap the live graph gets and the reason the
per-op column is an upper bound -- same caveat as §6.27.

**The two biggest single items:**
- `nlp_create_qkv_heads` at 2.56 ms/frame confirms NOTES [flow-10]'s recorded ~97 us floor. §6.30 splits
  it and shows the kernel really is that slow. This is the upstream issue still unwritten.
- `semantic_code` at **1241 us for one call** is the largest untouched item in the model. It is a
  host->device upload, an fp32 matmul against the semantic head, a mask add, an argmax and a
  device->host download. Not yet swept.

### 6.30 — Block 2 sweep: five candidates, ONE ships, and one of the three refutals is of my own theory

**A hypothesis I had before measuring, which the split killed.** Block 2 folds the CFG batch into 6 rows
so each weight is read once (NOTES [flow-11]), but `nlp_create_qkv_heads` needs 4D [B,1,S,QW], so the
fold must be undone: [1,6,6144] -> [2,1,3,6144]. In TILE_LAYOUT the first pads to [1,32,6144] -- one tile
row -- and the second to [2,1,32,6144] -- two. Different physical sizes, so that "reshape" moves data. I
expected the repack to be most of the 122 us.

    SPLIT: the reshape alone 25.9 us; the op on a pre-reshaped tensor 96.6 us  (together 122.5)

So the repack is **a fifth** of it, not most, and the kernel genuinely costs ~97 us. The row fold is not
quietly paying its weight-read win back in padding.

| sweep | shipped | best alternative | verdict |
|---|---|---|---|
| 1. rms_norm, 49 calls/frame | 47.5 us (3 ops) | one-op interleaved **115.1** | **rejected** -- 2.4x slower AND 4.8e-03 off |
| 2. qkv head split | 122.0 us | manual 3 slice+3 reshape+3 permute, L1 | **isolated INCONCLUSIVE (§6.33); ships on the whole-block A/B** |
| 3. unfold heads | 43.2 us | permute-straight-from-av 24.9 | **rejected** -- 1.77x but **1.4e+00 WRONG** |
| 4. CFG combine + Euler | 160.0 us | weighted reduce **112.6** | 1.363x isolated, **then rejected on the block** |
| 5. `_trunk` sequence build | 124.5 us | hoist the reshapes **105.8** | **SHIPS** -- bit-exact |

Notes on the rejections, because each says something:
- **the one-op norm is much worse.** Three sharded ops (17.5 shard + 20.0 norm + 10.5 unshard) beat one
  interleaved `rms_norm` by 2.4x. Sharding is not overhead here, it is the fast path.
- **the hand-rolled split, CORRECTED, is 1.086x FASTER** -- and my first reading of it was invalid.
  I gave the shipped op `memory_config=_L1` and let my slices and permutes take the default, so they
  landed in DRAM. NOTES [flow-10] measured exactly that difference and found the L1 output worth
  **2.5 ms/frame**, not on the op (~7 us) but on the four downstream consumers that then keep q/k/v in
  L1. So the manual route was being credited for skipping work the shipped one does. Like-for-like:

      shipped: nlp_create_qkv_heads(memory_config=L1)   122.0 us   1.001x   L1/L1/L1  bit-exact
      manual, DEFAULT mc  (what the first pass timed)   122.4 us   0.998x   DR/DR/DR  bit-exact
      manual, memory_config=L1  (like-for-like)         112.4 us   1.086x   L1/L1/L1  bit-exact

  0.202 ms/frame isolated, bit-exact, in the right memory. NOTE THIS CONTRADICTS [flow-10]'s recorded
  158 us for a hand-rolled split, so one of the two measurements is of a different construction -- that
  is unreconciled. And it trades 1 op for 9, i.e. more host dispatch, so on this session's record it is
  a candidate and not a win until the whole block says so.
- **the fastest unfold is wrong.** Permuting straight from `av` skips the [B,32,3,HD] regroup, is 1.77x
  faster, and returns garbage (1.4e+00 relative). §6.25's lesson, again, caught by the numerics column.

**SWEEP 5 CORRECTED, and this is the substantive error of the pass.** The first reading measured
`concat([p0, p12])` against a `p12 = concat(p1s[i], p2)` built once OUTSIDE the timed loop and reported
1.461x, +0.279 ms/frame. That is not shippable: p1s[i] varies per step, so in the real graph that concat
does not vanish, it MOVES -- 7 new concats a frame to save 7 cheaper ones, and the op count goes UP.
Re-measured against only spellings that could really ship:

    p0  = linear(x_t, input_projection)   changes every STEP   -- and is ALREADY [B,1,3072], no-op reshape
    p1s = the time schedule               constant for the model's life -- hoist into _schedule
    p2  = linear(h, llm_projection)       changes once per FRAME       -- hoist to once per frame

    shipped: 3 reshape + concat(3) + reshape        115.9 us   1.010x
    hoist p1,p2 reshapes                            105.8 us   1.107x   +0.079 ms/frame  bit-exact
    + drop p0's reshape too                         105.7 us   1.108x   (p0's was already free)
    concat(2) vs hoisted p12 (NOT shippable)         81.2 us   1.442x

So the real win is **+0.079 ms/frame, not +0.279** -- the 1.461x was the unshippable hoist.

**AND THEN THE BLOCK REFUSED TO PAY FOR THE CFG FOLD.** A/B on the whole Block 2 frame, same session:

| build | ms/frame | codes differing from the fp32 reference (of 37) |
|---|---|---|
| baseline | 22.583 | 1 |
| **sequence hoist only** | **22.476** | **1** |
| both wins | 22.483 | **2** |

The sequence hoist takes the **entire** 0.107 ms and leaves the numerics alone. The CFG fold -- 1.543x in
isolation, 7 ops down to 5 -- contributes **nothing** on the whole block and flips a discrete code, since
its 1.6e-04 reassociation is enough to cross an FSQ quantization boundary. Reverted, helper deleted.

That is the **fourth** time this session an isolated win vanished on the whole unit (§6.18, §6.19, §6.27's
`to_memory_config` reading, now this). The rule has earned its keep: **for anything touching memory
config, layout, or op count on small tensors, the smallest honest unit of measurement is the whole
block.** Isolation is for choosing what to try, never for deciding what to ship.

**GATED AND SHIPPED.** The sequence hoist is bit-exact end to end, verified as a paired A/B in one
session with the tree stashed and restored between halves:

| gate | before | after |
|---|---|---|
| flow: velocity PCC | 0.99998480, maxabs 2.833e-02 | identical |
| flow: semantic code | exact match `[262, 3346]` | identical |
| flow: full frame | 2 of 74 codes differ | identical |
| codes: semantic / acoustic | 1, 97/288 (33.7%) | identical |
| codes: the whole per-frame code table | | **byte-identical, every frame** |
| Block 2 alone | 22.583 ms/frame | **22.476** |

Byte-identical codes is a stronger statement than any PCC: if the integer codes do not move, the
waveform does not move, so WER cannot move. `--gate decode` was also run (pooled mean 0.92%, p90 1.28%,
max 3.05%, min PCC 0.999040, 15 prompts x 22 frames) and matches §6.13's recorded level -- expected,
since this change does not touch Block 1, and worth having as the check that it does not.

### 6.31 — the two wins the map actually produced: qkv head split and semantic_code

§6.30's sweep shipped one small thing. Following the map's two biggest lines properly produced the two
real wins, and both are BYTE-IDENTICAL.

**Block 2, cumulative, all three changes gated as byte-identical codes:**

| change | Block 2 ms/frame | commit |
|---|---|---|
| start of the sweep | 22.583 | |
| `_trunk` reshape hoist (§6.30) | 22.476 | `41194fe16f5` |
| hand-rolled qkv head split | **21.142** | `55d4148eb4b` |
| `semantic_code` mask+argmax to host | **20.801** | this one |
| **total** | **-1.782 ms/frame, 7.9%** | |

**1. The qkv head split, and the measurement lesson in it.** Both of my earlier readings compared the
manual route against the shipped op WITHOUT matching output memory configs -- the shipped op gets
`memory_config=_L1`, my slices and permutes took the default and landed in DRAM. NOTES [flow-10]
measured that exact difference at 2.5 ms/frame downstream. Corrected:

    shipped: nlp_create_qkv_heads(memory_config=L1)   122.0 us   1.001x   L1/L1/L1  bit-exact
    manual, DEFAULT mc (what I first timed)           122.4 us   0.998x   DR/DR/DR  bit-exact
    manual, memory_config=L1                          112.4 us   1.086x   L1/L1/L1  bit-exact

And then the whole block paid **six times** the isolated figure -- interleaved A/B, 200 frames each,
three rounds, the method swapped on the class so nothing else differs:

    round      shipped     manual     ratio
        0      22.369     21.131    1.0586x
        1      22.419     21.180    1.0585x
        2      22.336     21.116    1.0578x
     mean      22.375     21.142    1.0583x     = 1.233 ms/frame

**This is the FIRST time this session the whole block paid MORE than isolation**, after four cases of it
paying less (§6.18, §6.19, §6.27, §6.30). So the lesson is NOT "isolated numbers are optimistic". It is
that op granularity cannot see overlap or memory residency AT ALL, in either direction. Likely
mechanism: the fused op needs a 4D [B,1,3,QW] input and that reshape repacks tile padding
([1,32,QW] -> [2,1,32,QW], 25.9 us), while slicing the folded [1,B*3,QW] tensor never repacks.
Measured, not established.

UNRECONCILED: [flow-10] records a hand-rolled split at 158 us where this measures 112.4. Some difference
of construction that I did not find. Logged rather than written off -- the change ships on the
whole-block A/B and byte-identical codes, neither of which depends on settling it.

**2. `semantic_code`: the argmax was 40% of it and belongs on the host.** The split, corroborated two
ways -- the pieces sum to 1162.9 us of the whole call's 1229.5, and the composite `linear+mask+argmax`
measured 1086.8 against the pieces' 1088.4, agreeing to 0.1%:

    linear fp32 against a (3072, 8320) head   562.0 us   45.7%   102 MB at 182 GB/s -- AT the roofline
    argmax over 8320 values                   490.1 us   39.9%   33 KB at 0.07 GB/s -- ALL overhead
    from_torch H->D                            53.0 us    4.3%
    add semantic_mask                          36.3 us    3.0%
    to_torch D->H                              21.4 us    1.7%

The matmul is at the roofline and is not the problem. The argmax is 490 us to reduce 33 KB. And it costs
nothing to move, because this call ALREADY ended in a device->host copy -- doing the reduce on the host
does not add a round trip, it only makes that copy 8320 fp32 values instead of 1, i.e. 33 KB of PCIe.
Scored on REAL Block 1 hidden states from the fixture prompts:

    A shipped: linear fp32 + device mask + argmax   1213.5 us   1.000x   0 of 8 ids changed
    B host argmax, device mask                       893.5 us   1.386x   0 of 8
    C host argmax AND host mask   <- SHIPPED         860.3 us   1.439x   0 of 8
    D bf16 head, device argmax                       946.9 us   1.308x   0 of 8
    E bf16 head + host argmax + host mask            595.7 us   2.079x   0 of 8

C is exactly the same fp32 arithmetic -- same values, same order, the add and the reduce just happen on
a different processor -- so it carries no numerical risk. Predicted 0.353 ms/frame, delivered 0.341.

**E IS FASTER STILL AND IS DELIBERATELY NOT SHIPPED.** 2.079x, another 0.265 ms/frame, and 0 of 8 real
prompts moved. But this is an ARGMAX over a vocabulary: what decides a flip is the top-2 GAP, not a norm,
and 8 single frames is a thin sample for a discrete decision. The semantic token is the highest-stakes
integer in the model -- it feeds Block 1's next input embedding, so ONE flip redirects the entire
remaining generation, unlike an acoustic code which affects one frame. Needs a broad real-prompt gate.

**AND MY FIRST VERSION OF THIS PROBE WAS WORTHLESS.** Round 1 scored the bf16 candidate on 64 RANDOM
gaussian draws and reported 0 changed ids. Trap #12 already records random embeddings reading PCC 0.892
where real prompts gave 0.9994, and an argmax is precisely where that bites. Round 2 pulls real hidden
states out of Block 1. **ALWAYS GATE ON REAL PROMPTS** -- written in this file, and I still did it wrong
once today.

**Gate for both changes** (each run separately, against the immediately preceding commit):

| | qkv split | semantic host-argmax |
|---|---|---|
| flow: velocity PCC | 0.99998480, unchanged | 0.99998480, unchanged |
| flow: semantic code | exact `[262, 3346]` | exact `[262, 3346]` |
| flow: full frame | 2 of 74, unchanged | 2 of 74, unchanged |
| codes: semantic / acoustic | 1, 97/288, unchanged | 1, 97/288, unchanged |
| codes: per-frame code table | **byte-identical** | **byte-identical** |
| test_flow_pcc | 13 passed | 13 passed |

### 6.32 — the end-to-end gate for §6.31, and a stale-file error of mine that nearly went in the record

**Long-form RTF 0.61-0.65 -> 0.57-0.64, ~50.9 -> ~48.1 ms/frame.** Full 15-prompt quality set,
`resultsb2sweep.json`:

| case | frames | RTF | ms/frame |
|---|---|---|---|
| 2 | 461 | 0.644 | 51.5 |
| 3 | 487 | 0.573 | 45.8 |
| 10 | 220 | 0.589 | 47.1 |

Case 0's 1.855 is the compile case and is excluded, per §6.21's convention: quote long-form (>=100
frames) only.

**Quality, unchanged and at the bar:**

    long-form WER          0.00% over 298 words = 0 WRONG      (reference scores 0.0%)
    short                  0.00% over  42 words
    [END_AUDIO] natural    15/15 cases
    voice identity         PASS -- same-voice pair most similar, F0 spread 103-201 Hz

**AND THE STRONGEST NUMERICAL GATE THIS PROJECT HAS RUN.** A paired A/B against the pre-sweep commit
`6dc2dc2d460`, generating the three natural-text cases free-running on each build:

| case | pre frames | post frames | pre RTF | post RTF |
|---|---|---|---|---|
| 0 | 66 | 66 | 1.577 | 1.652 |
| 2 | 461 | **461** | 0.637 | 0.620 |
| 3 | 487 | **487** | 0.592 | 0.572 |

**Frame counts identical on every case.** That is much stronger than the 8-12 frame `--gate codes`
comparison: free-running generation is autoregressive, so ANY divergence compounds and moves the
termination frame. Reproducing 461 and 487 exactly means the whole generation is bit-identical, which is
what "byte-identical codes" was claiming on 8 frames. Use this as the gate for anything claiming
exactness in future -- it is cheap (two cases, ~90 s) and it cannot be fooled by a short window.

**THE ERROR, because it is a new failure mode for the list.** I first scored the run with

    score_quality_set.py generated/results_b2sweep.json || score_quality_set.py generated/results.json

`generate_quality_set.py` writes `results{tag}.json` with **no underscore**, so the first path did not
exist and the `||` silently scored `results.json` -- an unrelated file from 08-04. I reported "1 wrong
word of 274" from it. The tell was already printed in my own output: the scorer said case 0 = 71 frames,
case 2 = 449, case 3 = 495, while the generation log said 66, 461, 487. Correctly scored, the run gives
**0 wrong of 298**, matching the recorded baseline.

Two lessons, both cheap:
- **never write a `||` fallback into a gate.** A gate that silently substitutes a different input is
  worse than one that fails, because it returns a plausible number.
- **the results file carries the frame counts; check them against the generation log.** That single
  cross-check catches a stale file instantly, and it was sitting in the output I had already printed.

`generated/` holds 39 `results*.json` files from this project's sweeps. Any of them will score cleanly
and none of them will tell you it is the wrong one.

### 6.33 — CORRECTION to §6.31: the hand-rolled split is NOT faster in isolation, and the fused op is not flat

Two claims in §6.31 are wrong. The shipped change is still right, for a different reason.

**WRONG CLAIM 1: "manual 112.4 us vs fused 122.0 us, 1.086x isolated".** It does not reproduce. Settled
with 500 reps x 6 rounds, alternating A/B/B/A so neither op is always second, on both a real matmul
output and a synthetic tensor:

| source | | mean us | min | max | spread |
|---|---|---|---|---|---|
| real matmul output | fused | **122.3** | 122.2 | 122.4 | **0.2** |
| | manual | 125.7 | 122.0 | 127.8 | 5.8 |
| synthetic from_torch | fused | **122.1** | 122.0 | 122.2 | **0.2** |
| | manual | 133.4 | 118.3 | 146.0 | 27.6 |

**In isolation the fused op is at least as fast, and it is far more stable** -- 0.2 us spread against
5.8-27.6. The 112.4 was a low sample from a wide distribution, and a 10 us "win" was read off it. Nine
ops have variable dispatch cost; one op is deterministic. That variance is itself worth knowing.

**WRONG CLAIM 2: "the fused op is fixed cost".** NOTES [flow-10] measured it flat at ~97 us from S=3 to
S=32 and I generalised that to "fixed cost". It is flat in **S**, not in **B**:

| B | rows | qkv KB | fused us | manual us | manual/fused | winner |
|---|---|---|---|---|---|---|
| 2 | **6** | 72 | 122.9 | 127.1 | 1.034x | fused |
| 4 | 12 | 144 | 166.0 | 130.9 | 0.788x | manual |
| 8 | 24 | 288 | 200.0 | 165.8 | 0.829x | manual |
| 16 | 48 | 576 | 284.1 | 192.5 | 0.677x | manual |
| 32 | 96 | 1152 | 442.1 | 365.6 | 0.827x | manual |
| 64 | 192 | 2304 | 758.2 | 714.5 | 0.942x | manual |
| 128 | 384 | 4608 | **TT_FATAL** | -- | -- | fused fails |

6x the batch is 6x the data and 6.2x the time. Growing S and growing B are different axes and I conflated
them.

**A prediction that WAS right, directionally.** The manual route makes THREE passes over memory (slice
writes q/k/v, reshape rewrites, permute rewrites) where the fused op makes ONE, so its advantage should
erode as bytes start to dominate launch cost. It does: 0.788x at B=4 drifting to 0.942x at B=64. No
crossover appears before the fused op TT_FATALs at B=128, so the erosion never completes -- but the
mechanism is visible in the trend. Block 2 is structurally 3 tokens, so B is the only growth axis, and
B=64 is 32x anything planned.

**WHAT ACTUALLY STANDS, and it is a better result than the one I recorded.** The whole-block win is
unaffected and triple-measured:

    interleaved A/B, 3 rounds x 200 frames   22.375 -> 21.142 ms   1.0586 / 1.0585 / 1.0578
                                             spread 0.0008 on a 0.058 effect
    incremental, separate script             22.476 -> 21.142 ms
    end-to-end RTF, cases 2 and 3            0.637 -> 0.620, 0.592 -> 0.572 (all three changes)

**So the op is NOT faster, yet the block is 1.233 ms/frame faster.** The entire win is a system effect
around the op, not the op. That makes the mechanism MORE open than §6.31 claimed, not less. Candidates,
none verified: the fused path's 4-D reshape builds a 768 KB intermediate (384 KB -> 768 KB, since
[1,6,6144] pads to one 32-row slab and [2,1,3,6144] pads to two) that must be allocated, written and
re-read, which my column slices never create; the two paths' L1 layouts differ for the four downstream
consumers; the fused kernel may not overlap with its neighbours the way nine small ops do.

**The methodological point, which is the fourth version of the same lesson today.** §6.31 said the
whole block paid "six times" the isolated figure. The truth is worse: **the isolated figure had the
wrong SIGN.** At 6 rows the effect being measured (~10 us) is smaller than the run-to-run spread of the
thing measuring it (5.8-27.6 us), so the isolated comparison could not have resolved it in either
direction. Always report the spread next to the mean; a single number with no spread is not a
measurement. The decision was right only because it was taken on the whole block.

### 6.34 — project THEN duplicate, instead of duplicate THEN project: rejected

A natural question, so it is recorded to stop it being re-asked. `_solve` builds p0 as

    x2 = concat([x, x], dim=0)                  [B,1,36] -> [2B,1,36]
    p0 = linear(typecast(x2), input_projection) [2B,1,36] -> [2B,1,3072]

**and the premise is correct: `x` is byte-identical in both CFG halves** -- guidance differs in the
CONDITIONING (p2 is llm_hidden over zeros), never in the state -- so that matmul runs twice on the same
input. Projecting once and duplicating the result is mathematically identical for any B.

**It is still slower.** Isolated, 400 reps x 4 alternating rounds:

| variant | mean us | spread | vs shipped | numerics |
|---|---|---|---|---|
| **shipped: concat -> typecast -> linear** | **43.9** | 3.4 | 1.000x | -- |
| linear -> concat (project first) | 56.0 | 4.3 | **0.785x** | bit-exact |
| linear -> `ttnn.repeat` | 78.8 | 2.0 | 0.557x | bit-exact |
| typecast -> linear -> concat | 60.7 | 3.6 | 0.723x | bit-exact |

Whole Block 2 frame, interleaved A/B, 200 frames x 4 rounds: shipped 21.232 ms (spread 0.084),
project-first 21.279 (spread 0.057). Gap -0.047 ms, under the spread, so INCONCLUSIVE on the block and
clearly worse isolated. Codes identical either way -- the rewrite is correct, just unprofitable.

**WHY, and it generalises.** The redundant matmul is nearly free: input_projection measures 17.5 us
against a **1.1 us byte floor**, so the duplicated WORK is about 1 us and the rest is launch cost, which
both forms pay equally (3 ops either way). Moving the duplication downstream makes it operate on a 48x
wider tensor:

    duplicate the INPUT    [1,1,36]   tile-padded 32x64    =  16 KB
    duplicate the OUTPUT   [1,1,3072] tile-padded 32x3072  = 384 KB

That cost **+12 us**, an order of magnitude more than the ~1 us of redundant matmul it removed.

**The rule: redundant work is only worth removing if it is expensive relative to what removing it
costs.** At these tensor sizes almost nothing is expensive except launching ops, so a restructuring that
keeps the op count the same and moves data to a WIDER point in the graph will usually lose. Same shape of
reasoning as §6.24's fusion rule -- fusion pays only when an operand is too narrow to earn its launch.

Incidental: **`ttnn.repeat` is 1.8x worse than `ttnn.concat`** for the same duplication.

### 6.35 — what the CFG duplication costs (1.8%), and the batching headroom that falls out of it

The question: `x` is duplicated so the velocity net runs twice per Euler step, once with the llm
conditioning and once with zeros. That looks like paying 2x for guidance. What does it actually cost?

**First, it is semantically required.** The two passes need two independent attention contexts. Merging
them into one 4-token sequence `[p0, p1, p2_cond, p2_uncond]` is NOT equivalent -- attention is
bidirectional over the 3 tokens, so `p0` would then see both conditionings at once. And the duplication
is only redundant at the INPUT projection: after the first attention layer the two copies have attended
to different p2 and genuinely diverge. That is why §6.34's project-first rewrite was valid but could only
ever save that one projection.

**Second, it costs 1.8%.** One `_block` call, 3 layers, 300 reps x 4 alternating rounds:

| | rows | tile rows used | us/call | spread |
|---|---|---|---|---|
| no guidance (hypothetical) | 3 | 1 of 32 | 828.2 | 0.3 |
| **with guidance (ships)** | **6** | **1 of 32** | **843.6** | 0.5 |

**+15.3 us per call, +1.8%, i.e. 0.322 ms/frame** over 21 calls, out of Block 2's 20.8 ms. The naive
"CFG doubles the work" intuition simply does not apply: 3 rows and 6 rows both pad to ONE 32-row tile,
and the row fold (NOTES [flow-13]) reads each weight once regardless. **So do not chase CFG elimination
-- the entire prize is 0.322 ms and it would change the audio.**

**Third, the same fact makes a real redundancy free.** `_cfg_input` puts llm_hidden over ZEROS and
`llm_projection` has no bias, so **p2's unconditional half is exactly zero** (verified: max |value| =
0.000000). We do run a 3072x3072 matmul to produce a known-zero result. Exploiting it saves nothing:

    weight [3072,3072] BFP8 = 9.6 MB, floor 51.7 us
    [2,3072] @ W (ships)       65.1 us   spread 0.1
    [1,3072] @ W (cond only)   65.1 us   spread 0.1     delta +0.0

M=1 and M=2 both pad to one tile row, so the weight is read once either way.

**AND HERE IS THE PART WORTH ACTING ON.** If 6 rows is as cheap as 3, how far does that go? A tile is 32
rows and the shipped path uses 6:

| utterances | B | rows | tiles | us/`_block` | **us per utterance** | throughput |
|---|---|---|---|---|---|---|
| 1 | 2 | 6 | 1 | 844.9 | 844.9 | 1.0x |
| 2 | 4 | 12 | 1 | 884.8 | **442.4** | **1.9x** |
| 4 | 8 | 24 | 1 | 995.1 | **248.8** | **3.4x** |
| 8 | 16 | 48 | 2 | TT_FATAL | -- | -- |

**Four utterances cost 1.18x the time of one -- 3.4x the throughput.** This is the "31 unused tile rows"
headroom, measured rather than asserted, and it is real.

**The 32-row ceiling is OUR OWN CONSTANT, not the hardware.** `_NORM_SHARD`'s shard shape is hardcoded
`(32, 96)`, and at 48 rows it raises `!shard_grid_fit_error.has_value()`. `wqkv` at 48 rows is fine and
returns `(1, 48, 6144)`. So going past one tile needs a row-count-aware norm shard (cache one config per
batch size, as `_schedule` already does for the timestep tokens) -- not a new kernel.

Note this is THROUGHPUT, not latency: RTF per utterance does not improve, but a server doing 4 streams
would do them in 1.18x the time of one. Whether that is useful depends on the deployment, which is not
our call -- but it is now a measured option rather than a guess.

**A probe bug of mine, recorded.** The first version of the p2 comparison left `gen._up(...)` INSIDE the
benched lambda, so it timed a host->device upload every iteration and reported the SMALLER matmul as
2.8x slower (185.2 vs 65.1 us). Hoist every operand out of the timed callable; if a "smaller" thing
measures slower, suspect the harness before the hardware.

### 6.36 — Block 2's overhead map REGENERATED, with source line numbers (supersedes §6.29)

§6.29 is stale: three of the things it measured have since changed -- `nlp_create_qkv_heads` (its largest
line, 2.564 ms/frame) is now a hand-rolled 9-op split, `_trunk`'s three reshapes are hoisted, and
`semantic_code`'s mask and argmax moved to the host. **Rank off THIS table, not §6.29.** Line numbers are
into `tt/ttnn_voxtral_flow.py` at commit `f1d985a37fe`. 300 reps x 3 rounds; `_block` ops run 21x a
frame, `_solve`/`_trunk` ops 7x, two once.

| line | op | ×/frame | floor µs | actual | spread | ovhd | ms/frame | recoverable |
|---|---|---|---|---|---|---|---|---|
| `157` | ffn norm, all 3 ops | 21 | 2.0 | 56.7 | 8.7 | 54.7 | 1.191 | 1.148 |
| `267` | CFG combine: 2 multiply + 1 add | 7 | 0.0 | 122.1 | 14.6 | 122.1 | 0.855 | 0.855 |
| `142` | reshape+permute v (0,2,1,3) | 21 | 0.0 | 37.6 | 0.7 | 37.6 | 0.790 | 0.790 |
| `176` | reshape seq -> [1,6,3072] | 7 | 0.0 | 106.0 | 0.2 | 106.0 | 0.742 | 0.742 |
| `174` | concat([p0,p1,p2], dim=1) | 7 | 0.0 | 101.0 | 0.1 | 101.0 | 0.707 | 0.707 |
| `139` | reshape+permute k (0,2,3,1)  pre-transposed | 21 | 0.0 | 33.6 | 6.0 | 33.6 | 0.705 | 0.705 |
| `146` | matmul q(row-fold) @ kT -> scores | 21 | 0.0 | 33.2 | 8.2 | 33.2 | 0.696 | 0.696 |
| `162` | add residual (ffn) | 21 | 0.0 | 31.1 | 8.2 | 31.1 | 0.654 | 0.654 |
| `160` | multiply(g, w3) | 21 | 0.0 | 30.7 | 1.1 | 30.7 | 0.645 | 0.645 |
| `154` | add residual (attn) | 21 | 0.0 | 30.6 | 1.5 | 30.6 | 0.643 | 0.643 |
| `153` | permute+reshape -> folded rows | 21 | 0.0 | 28.4 | 1.3 | 28.4 | 0.597 | 0.597 |
| `162` | linear w2     9216x3072 BFP8 | 21 | 155.1 | 179.0 | 0.3 | 24.0 | 3.760 | 0.504 |
| `269` | Euler update: multiply + add | 7 | 0.0 | 70.8 | 6.6 | 70.8 | 0.496 | 0.496 |
| `137` | reshape q -> [2,3,32,128] | 21 | 0.0 | 22.6 | 0.3 | 22.6 | 0.474 | 0.474 |
| `125` | rms_norm sharded                 (attn norm 2/3) | 21 | 0.0 | 21.6 | 1.0 | 21.6 | 0.453 | 0.453 |
| `149` | softmax numeric_stable | 21 | 0.0 | 19.4 | 0.1 | 19.4 | 0.407 | 0.407 |
| `179` | _trunk final norm, all 3 ops | 7 | 2.0 | 55.7 | 4.3 | 53.7 | 0.390 | 0.376 |
| `125` | to_memory_config -> _NORM_SHARD  (attn norm 1/3) | 21 | 1.0 | 18.7 | 1.5 | 17.7 | 0.394 | 0.372 |
| `242` | semantic_code: upload + linear + HOST argmax | 1 | 527.0 | 895.8 | 10.6 | 368.8 | 0.896 | 0.369 |
| `150` | matmul a @ v | 21 | 0.0 | 17.3 | 2.3 | 17.3 | 0.363 | 0.363 |
| `152` | reshape av -> [2,32,3,128] | 21 | 0.0 | 16.6 | 1.3 | 16.6 | 0.348 | 0.348 |
| `136` | slice q  cols 0..4095    -> L1 | 21 | 0.0 | 14.3 | 0.3 | 14.3 | 0.301 | 0.301 |
| `137` | permute  q (0,2,1,3) | 21 | 0.0 | 14.2 | 2.9 | 14.2 | 0.298 | 0.298 |
| `136` | slice k  cols 4096..5119 -> L1 | 21 | 0.0 | 13.8 | 0.4 | 13.8 | 0.290 | 0.290 |
| `136` | slice v  cols 5120..6143 -> L1 | 21 | 0.0 | 13.7 | 0.0 | 13.7 | 0.287 | 0.287 |
| `154` | linear wo     4096x3072 BFP8 | 21 | 68.9 | 81.5 | 0.1 | 12.6 | 1.711 | 0.264 |
| `181` | linear acoustic_codebook_output  3072x36 | 7 | 1.1 | 32.9 | 0.3 | 31.8 | 0.230 | 0.223 |
| `128` | to_memory_config -> DRAM         (attn norm 3/3) | 21 | 1.0 | 11.6 | 0.1 | 10.5 | 0.243 | 0.221 |
| `262` | typecast x2 -> bf16 | 7 | 0.0 | 26.4 | 0.2 | 26.4 | 0.185 | 0.185 |
| `184` | slice -> [2,1,36] | 7 | 0.0 | 24.2 | 0.3 | 24.2 | 0.170 | 0.170 |
| `262` | linear input_projection   36x3072 | 7 | 1.1 | 17.5 | 0.7 | 16.4 | 0.123 | 0.115 |
| `261` | concat([x,x], dim=0) | 7 | 0.0 | 15.3 | 0.5 | 15.3 | 0.107 | 0.107 |
| `266` | slice v_unc | 7 | 0.0 | 14.0 | 2.1 | 14.0 | 0.098 | 0.098 |
| `265` | slice v_cond | 7 | 0.0 | 13.0 | 0.2 | 13.0 | 0.091 | 0.091 |
| `264` | typecast v -> fp32 | 7 | 0.0 | 11.3 | 0.9 | 11.3 | 0.079 | 0.079 |
| `183` | reshape out -> [2,3,36] | 7 | 0.0 | 10.8 | 0.1 | 10.8 | 0.076 | 0.076 |
| `158` | linear w1     3072x9216 BFP8 +silu | 21 | 155.1 | 157.8 | 0.3 | 2.7 | 3.314 | 0.058 |
| `255` | linear llm_projection  3072x3072  (once/frame) | 1 | 51.7 | 65.1 | 0.0 | 13.4 | 0.065 | 0.013 |
| `134` | linear wqkv   3072x6144 BFP8 | 21 | 103.4 | 101.8 | 0.1 | -1.6 | 2.137 | -0.033 |
| `160` | linear w3     3072x9216 BFP8 | 21 | 155.1 | 147.5 | 0.5 | -7.5 | 3.098 | -0.158 |

**Frame 21.243 ms. Weight-read floor 14.080 ms. Reachable ceiling 7.163 ms (34%.)** The floor now
includes `semantic_code`'s fp32 head (102 MB, 527 us), which §6.29 wrongly counted as zero -- that is why
the floor moved 13.553 -> 14.080.

**FOUR THINGS TO READ CAREFULLY, because the table looks more actionable than it is.**

1. **Isolated sums to 29.1 ms against a real 21.2 ms frame.** ~8 ms is overlapped in the live graph. The
   per-op ranking is usable; the total is an upper bound; "recoverable" means *if the op vanished and
   nothing else changed*, which §6.33 proved is not how this behaves.

2. **The hand-rolled head split sums HIGHER than what it replaced.** Lines 136/137/139/142 total ~150 us
   (41.8 in slices, ~108 in reshape/permutes) against the 122 us fused op -- and it is still 1.233
   ms/frame FASTER on the whole block. That contradiction is the single best argument against treating
   this table as a to-do list.

3. **Six of the top ten are already closed.** 157 (one-op interleaved norm: 2.4x SLOWER, §6.30), 267
   (weighted-reduce fold: 1.543x isolated, ZERO whole-block, flips an FSQ boundary, §6.30), 139/142/153
   (swept as part of the split and the unfold, §6.30), 125 (norm grid swept, [flow-07]).

4. **Read the SPREAD column before believing any row.** The residual adds show 8.2 us of spread on a 31
   us overhead; the CFG combine 14.6 on 122. Enough to rank, not enough to decide. §6.33 is the case
   where an effect smaller than its own measurement spread was reported with the wrong sign.

**The largest genuinely untouched item is the sequence build: lines 174 + 176, 1.449 ms/frame combined.**
The three reshapes were hoisted (§6.30/[flow-19]) but the `concat` of three [2,1,3072] tensors (101.0 us)
and the final reshape to [1,6,3072] (106.0 us) remain. After that: the two residual adds (1.297
combined), `multiply(g, w3)` (0.645), the scores matmul (0.696).

**And note what is NOT here.** wqkv measures 1.6 us and w3 7.5 us BELOW their 194 GB/s floors, which is
how we know the true ceiling is ~202. All five weight matmuls are at the roofline; there is no matmul
work left in Block 2.

### 6.37 — sdpa for Block 2's attention: 5.9x on the op, +0.816 ms/frame, and it COSTS A WORD. Rejected.

The first change this session that a gate caught at the WER level, so it is worth reading as a template
for how far to take a promising number.

**The idea.** Block 2's attention is bidirectional and unmasked over 3 tokens -- exactly what non-decode
`scaled_dot_product_attention` is for. The shipped path is four ops (§6.36 lines 146/149/150/152):
`matmul q@kT` 33.2 us + `softmax` 19.4 + `matmul a@v` 17.3 + `reshape` 16.6 = 86.5 us. sdpa also emits
`[B,32,3,HD]` directly, so the reshape disappears too. It handles GQA 32/8 natively, which makes the row
fold ([flow-11]) and the `REP` constant unnecessary.

**`scale=1.0` is mandatory.** SCALE is folded into wqkv's q rows, so leaving sdpa's scale at its default
applies 1/sqrt(d) a SECOND time: 3.8e-01 relative error against the shipped path, versus 2.2e-02 with
scale=1.0. Anyone retrying this must pass it.

**The speed is real, and confirmed on the whole block.** 6 rounds x 200 frames, interleaved both
directions:

| variant | ms/frame | spread | vs shipped | delta | codes vs shipped |
|---|---|---|---|---|---|
| A shipped | 21.238 | 0.179 | -- | -- | -- |
| C `add_`/`multiply_` in place + L1 | 21.237 | 0.082 | 1.0000x | **+0.001** | 0 of 37 |
| **D sdpa only** | **20.422** | 0.047 | **1.0399x** | **+0.816** | 0 of 37 |
| E sdpa + in place | 20.478 | 0.071 | 1.0371x | +0.760 | 0 of 37 |

**But it is measurably less accurate, gated against fp64 TRUTH rather than against the shipped path**
(§6.25's rule), with the attention recomputed on the host in float64 from the device's own q/k/v:

    shipped: 4 ops     max abs err 2.746e-03   rel 3.20e-03   PCC 0.9999964
    sdpa(scale=1.0)    max abs err 1.780e-02   rel 2.07e-02   PCC 0.9998503

6.48x the error. That is NOT a compare-against-the-default artifact; sdpa really is less accurate here.

**AND THE END-TO-END GATE FOUND THE COST.** Frame counts moved -- case 0 66 -> 64, case 2 461 -> 445
(1.3 s less audio), case 3 487 -> 488 -- so generation diverged, which the frame-count A/B (§6.32) is
precisely designed to detect. Scored like-for-like on the same three cases:

| | case 0 | case 2 | case 3 | long-form WER |
|---|---|---|---|---|
| **shipped** | 0.0% | 0.0% | 0.0% | **0 wrong of 274** |
| **sdpa** | 0.0% | **0.8%** | 0.0% | **1 wrong of 274** |

Generation is deterministic, so the dropped word is caused by sdpa. **REVERTED.** +0.816 ms/frame is
1.7% of a frame; the project's headline quality claim is zero WER errors, and §6.16 already set the
precedent by handing back 2.5 ms/frame rather than lose accuracy.

**ONE HONEST QUALIFICATION.** One sample cannot separate "sdpa is systematically worse" from "sdpa is
merely DIFFERENT and this draw was unlucky" -- a different-but-equally-valid generation can drop a word
by chance. Settling that needs several seeds across all 15 cases. If 0.8 ms/frame ever matters enough,
that is the experiment; the accuracy loss against truth (6.48x) means the prior should be "worse", not
"different".

**AND THE IN-PLACE CANDIDATES ARE DEAD, after three contradictory readings of my own.** `ttnn.add_` and
`ttnn.multiply_` are bit-exact and looked like a win:
  1. first measurement: BOTH SLOWER (37.1 vs 27.0) -- invalid, I wrapped them in `ttnn.clone` so the
     input survived repeated iterations, and the clone costs more than the allocation it saves. In the
     real `_block` the operand is dead immediately, so no clone is needed.
  2. without the clone: 1.143x and 1.196x isolated, and +0.104 ms on the block over 3 rounds.
  3. over 6 rounds: **+0.001 ms.** The +0.104 was inside the 0.179 ms spread. And variant E shows
     in-place is marginally NEGATIVE when combined with sdpa.
Three measurements, three different answers, and only the last had enough rounds to beat its own noise.
`ttnn.swiglu` is also not applicable -- it `TT_THROW`s on a concatenated pair, and using it would require
fusing w1/w3, which §6.24 measured at 4x slower.

**Residual-as-bias is NOT EXPRESSIBLE here**, unlike in Block 1: ttnn's `bias` is added per output
column, and the residual differs per row. Recorded so it is not attempted.

### 6.38 — the seed control (frame counts are noise), w2-BFP8 over three seeds, and a repo-wide reuse survey

Three things, and the first invalidates evidence I used twice.

**1. FRAME COUNTS ARE SEED NOISE.** §6.37 rejected sdpa partly because case 2's frame count moved
461 → 445, and the w2-BFP8 test showed case 3 moving 487 → 523. I called both "divergence". The control
— same shipped code, same prompts, **only the `x_0` noise draw changed**:

| build / seed | case 0 | case 2 | case 3 |
|---|---|---|---|
| shipped, seed 0 | 66 | 461 | 487 |
| shipped, seed 1 | 76 | 470 | 508 |
| shipped, seed 2 | 70 | 438 | 449 |
| **seed-only swing** | **10** | **32** | **59** |

Both movements I cited sit INSIDE the seed-only swing. The correct reading is asymmetric:

    frame counts IDENTICAL  ->  the change is bit-exact. Strong, and it has caught nothing false (§6.32).
    frame counts MOVED      ->  the change is not bit-exact. NOTHING MORE.

**WER, by contrast, is stable across seeds: 0 wrong of 274 at seeds 0, 1 and 2.** So the clean baseline
is reproducible rather than lucky, and a word appearing is real signal — but it must be judged over
several seeds, never one.

**2. w2 IN BFP8, RE-GATED ON THREE SEEDS.** §6.16 rejected it on Block 1 teacher-forced metrics alone —
no WER, no frame counts, no listening pass, none of which existed then. Re-measured:

    block 1 decode step   w2 bf16  23.151 ms (spread 0.004)
                          w2 BFP8  20.507 ms (spread 0.002)     -> 2.644 ms/step, 5.5% of a frame

    decode gate, 15 prompts    bf16  mean 0.92%  p90 1.28%  max 3.05%  min PCC 0.999040
                               BFP8  mean 1.16%  p90 1.68%  max 4.41%  min PCC 0.997815

    long-form WER      seed 0     seed 1     seed 2
      shipped          0 wrong    0 wrong    0 wrong      (0 of 822 words)
      w2 BFP8          1 wrong    0 wrong    0 wrong      (1 of 822 words)

**So the upstream degradation is real and reproducible; the output evidence is 1 word in 822 against 0 in
822, which no test would call significant.** §6.16's pricing stands (w2 = 77% of the accuracy cost for
15% of the speed), and the accumulated Block 1 error is unambiguous. But the claim "it costs a word" is
NOT established — it rests on one occurrence. **Left at bf16, and this is a genuine open decision, not a
closed one.** To settle it: all 15 cases x 3+ seeds on both arms. At 2.644 ms/step it is the largest
single win left anywhere in this model, so that experiment is worth running before dismissing it.

**3. REPO-WIDE REUSE SURVEY.** This model was built standalone; four things elsewhere are worth knowing.

- **`models/common/modules/rmsnorm/rmsnorm_1d.py::_create_sharded_norm_program_config`** — builds the
  config from the ROW COUNT (`block_h = tile_padded_batch_rows // tile_size`, `tile_padded_batch_rows =
  32*ceil(batch/32)`, plus a `subblock_w` search). **This is precisely what our hardcoded `_NORM_SHARD`
  `(32, 96)` / `block_h=1` cannot do, and it is the stated blocker on §6.35's measured 3.4x batching
  win.** `models/common/modules/mlp/mlp_1d.py` applies the same pattern to shard shapes and matmul
  configs. Start here if the batching lead is picked up.
- **`models/demos/gemma4/tt/spec_decode.py`** — verify step "runs ONE batched forward over
  `[anchor, d1, ..., dK]` … candidates in the batch dim", which is exactly the mechanism for our 31
  unused tile rows. Its docstring also reasons through worse-vs-different: a batched forward differs by
  ~1e-5, "flips only target near-ties (top-2 logit gap < ~1)", giving "an equally-valid greedy trajectory
  thereafter". **Do not borrow that conclusion** — their correctness is guaranteed by construction
  (committed tokens always come from the target verify); a precision change has no such guarantee.
- **`models/common/tests/modules/sampling/test_sampling_1d.py`** — classifies index disagreements as
  TIE-BREAK (acceptable) vs TRUE-MISMATCH (kernel bug); notes `ttnn.argmax` returns uint32 vs torch's
  int64. Bears on [flow-08a]: we moved the semantic argmax to the host, so ties would now be broken by
  `torch.argmax`. fp32 logits plus an additive mask make exact ties effectively impossible — that is
  *why* it is safe, and it was not checked when the change shipped.
- **`models/tt_dit/` and `models/demos/z_image_turbo`** — flow-matching / DiT pipelines, the nearest
  in-tree neighbours to Block 2 if its structure is revisited (e.g. 7→5 Euler steps). Not examined.

Checked and NOT reusable: `deepseek_v3_b1/fused_ops/lm_head_sampling/` is a bespoke multi-device kernel
on CCL/MoE infrastructure and our semantic matmul is already at 182 of ~194 GB/s; `ttnn.sampling` /
`ttnn.topk` exist but the host argmax is simpler and 1.439x faster; `speecht5_tts` and
`demos/audio/whisper` are different architectures.

**4. THE MEASUREMENT METHODOLOGY, CHECKED AGAINST THE REPO'S OWN — and it holds.**
`models/common/tests/modules/attention/profiling/` states outright that it uses trace capture "for
accurate device-side timing without host dispatch overhead". Every probe in this file times
`perf_counter` around `synchronize_device`, which INCLUDES host dispatch — so the whole session's numbers
were open to the objection that they measured dispatch, not device work. Tested directly on `_solve`
(a pure device graph by design, [flow-18], so it captures cleanly), 200 reps x 3 rounds:

    eager  (host dispatch included)   19.145 ms   spread 0.002
    traced (device work only)         19.230 ms   spread 0.002

**Host dispatch is 0% of what was measured** — the device is saturated and dispatch is entirely hidden.
Traced is even marginally SLOWER (+0.085 ms), independently reproducing §6.26's trace verdict on a
different unit. So "an op costs ~20 us to exist" is genuine DEVICE time, every probe in this file is
sound as written, and §6.33's unexplained whole-block gain is NOT a dispatch artifact. Note this is
trace-as-MEASUREMENT; §6.26's rejection of trace-as-SHIPPING is untouched.

The full tracy op profiler (`tools/tracy/process_ops_logs.py`, `python -m tracy -r -m ...`) would give
per-op device times directly, and is the right tool if the §6.33 mechanism is ever chased. It does not
run here as-is: `websockets` is absent from the venv, `generated/profiler/` is empty (never run on this
box) and `build_Release/profiler/` looks incomplete, so it likely needs a profiler-enabled rebuild.

**5. TWO CHOICES THE SHARED LIBRARY MAKES DIFFERENTLY — both already settled here.**
- `models/common/modules/mlp/mlp_1d.py` defaults to **HiFi2 with fp16 accumulation**; we use HiFi4 +
  `fp32_dest_acc_en=True` everywhere. Already tested and rejected: [flow-03] measured HiFi2/no-fp32acc
  at PCC 0.9998382 and 35/222 code errors against HiFi4's 0.9999845 and 4/222 — and **slower**, 48.72 vs
  42.57 ms. In Block 1, HiFi2/LoFi saves ~4 ms for 10-20x the integer-code errors. Not a missed knob.
- their `_matmul_config` derives `in0_block_w = _find_largest_divisor(k // (tile*grid_y))` as a heuristic.
  We swept it explicitly instead ([gpt-20]: 1/2/4/8/16/32 -> 152.0/83.2/68.1/73.7/80.5/89.4 us) and
  shipped the bit-exact 2 over the faster-but-inexact 4. A heuristic would not have surfaced that
  exactness cliff, so the hand-sweep stands.

**Code audit, same pass:** 0 broken `NOTES.md [id]` pointers, 0 orphan NOTES entries, 0 broken `§6.x`
references, no comment contradicting a current measurement, `test_tt_defaults` (7 guards on the shipped
constants) green, 122 tests green. Five module-level names in `reference/` are unreferenced
(`text_logits`, `TIED_EMBEDDINGS`, `FM_HIDDEN_DIM`, `CODEC_HIDDEN_DIM`, `DEC_WINDOWS`) and are
**deliberately kept** — they document checkpoint architecture, and deleting model facts from the fp32
ground truth to satisfy a linter is a bad trade.

### 6.39 — p150: the sharded norm REVERSES. Block 1 drops it, +4.6 ms/frame

First re-derivation on the Blackhole fork, and it overturns §6.9 and §6.18 rather than refining
them. Method is §6.18's unchanged: the WHOLE 26-layer step (never the isolated norm — §6.18
showed that metric is anti-correlated), interleaved round-robin, and the shipped config entered
**twice** so the gap between its copies is a measured noise floor.

**Legality is a property of the TENSOR, not the chip, and is unchanged.** `block_w` is
tiles-per-core, a 32x3072 tensor is 1 x 96 tiles, a tile is indivisible, so the count must divide
96. What Blackhole changes is the ceiling: 13x10 = 130 cores makes **96 cores reachable for the
first time** — 8x8 could not express it.

| config | cores | block_w | ms/step | vs 8x4 |
|---|---|---|---|---|
| **interleaved ← SHIPS** | — | — | **21.406** | **+4.381** |
| 2x1 | 2 | 48 | 24.711 | +1.076 |
| 8x1 | 8 | 12 | 24.936 | +0.851 |
| 4x1 | 4 | 24 | 24.945 | +0.842 |
| 12x1 | 12 | 8 | 24.965 | +0.821 |
| 8x2 | 16 | 6 | 25.174 | +0.613 |
| 8x3 | 24 | 4 | 25.390 | +0.397 |
| 6x4 | 24 | 4 | 25.477 | +0.310 |
| 8x4 ← was shipped | 32 | 3 | 25.787 | — |
| 8x4#control | 32 | 3 | 25.937 | noise floor **0.151** |
| 8x6 | 48 | 2 | 26.241 | −0.454 |
| 12x4 | 48 | 2 | 26.344 | −0.558 |
| 12x8 | 96 | 1 | 28.345 | −2.558 |

**TWO N150 CLAIMS DIE HERE.** §6.18 found an interior MINIMUM at 32 cores; on Blackhole the
sharded curve is **monotone** and fewer cores is uniformly better, so the newly-reachable 96 is
the worst config measured. And §6.9's premise inverts outright: sharding is **−4.381 ms/step**,
not +4.4. Reproduced independently at −4.276 in a second script.

**THE MECHANISM, and §6.9's own sentence survives it** — *"the reshard is the tax, not the
reduction"*. Only the tax made the trip:

| one call, `[1,1,3072]` | N150 (§6.9/6.18) | p150 |
|---|---|---|
| interleaved | 115.5 µs | **63.7 µs** |
| sharded 8x4 | 54.6 µs | **93.5 µs** |

The p150's interleaved kernel made the reduction cheap enough that two `to_memory_config`
reshards per call stop paying for themselves. Ordering measured; the causal story inferred.

**GATED, and it is BETTER upstream rather than merely equal.** Paired, same session, 15 prompts
x 22 teacher-forced frames:

| | sharded 8x4 | interleaved |
|---|---|---|
| mean worst-sample | 0.94% | **0.91%** |
| p90 worst-sample | 1.35% | **1.30%** |
| min PCC | 0.999260 | **0.999302** |
| per-case p90 spread | 0.58 pp | **0.48 pp** |

End to end, 15 cases x 3 seeds, **one process per (arm, seed)** because §6.21 showed a case's
frame count depends on what ran before it:

| | seed 0 | seed 1 | seed 2 | total |
|---|---|---|---|---|
| sharded, long-form WER | 0 wrong | 0 wrong | 0 wrong | 0 of 894 |
| interleaved, long-form WER | 1 wrong | 0 wrong | 0 wrong | **1 of 894** |

15/15 `[END_AUDIO]` in all six runs. Long-form RTF (>=100 frames, case 0 excluded per §6.21):
**0.77–0.90 -> 0.71–0.78**, gen 62.0 -> 57.4 ms/frame, i.e. **−4.6 ms/frame** — essentially the
whole isolated win, unlike §6.17/§6.18 where ~1/3 showed up.

**THE ONE WORD IS THE CONTRACTION §6.9 ALREADY ADJUDICATED** — `"I am"` -> `"I'm"` in *"…now
that I have it, I am not going to be silent"*, same word, same sentence. §6.9 shipped the
sharded norm at 1/1/0 across seeds 0/1/2; this is 1/0/0. And §6.37's test cuts our way rather
than against: sdpa was rejected because the deterministic gate AGREED with the WER word (6.48x
worse vs fp64), so the prior was "worse". Here the gate improves on every stable column, so the
prior is "different".

**TWO HARNESS ERRORS OF MINE, and either would have produced a wrong answer:**
1. timing `gen.step()` put 3–4 ms of host spread (a float64 RoPE build + two uploads per call)
   on a <2 ms effect — inconclusive by construction, §6.33's trap exactly.
2. the replacement timed `gen.caches[gen.layers.index(lw)]`, a linear scan over dicts of ttnn
   tensors 26x per step. That added **~20 ms/step of my own host time** (45.4 measured against
   25.8 without it) and flattened every arm into a false "inert". Same class as §6.35's probe
   bug. **Hoist everything out of the timed callable, and suspect the harness before the chip.**

Only the third harness is reported above.

### 6.40 — p150: the same reversal in Block 2, +4.5 ms/frame. SHIPPED — and 3 seeds nearly rejected it

§6.39 rerun against Block 2's norm, which is called **49x per frame** (2 per `_block` x 3 layers
x 7 Euler steps, plus `_trunk`'s final norm x7) against Block 1's 52. Unit is the whole Block 2
frame (`decode_frame` on a REAL hidden state from case 2, fixed `x_0`), per §6.30.

| config | cores | block_w | ms/frame | vs 8x4 | codes != 8x4 |
|---|---|---|---|---|---|
| **interleaved** | — | — | **28.626** | **+4.173** | 1 / 36 |
| 4x1 | 4 | 24 | 31.800 | +0.998 | 0 / 36 |
| 2x1 | 2 | 48 | 31.849 | +0.950 | 1 / 36 |
| 8x1 | 8 | 12 | 31.902 | +0.897 | 0 / 36 |
| 12x1 | 12 | 8 | 31.983 | +0.816 | 0 / 36 |
| 8x2 | 16 | 6 | 32.441 | +0.357 | 1 / 36 |
| 8x4#control | 32 | 3 | 32.555 | +0.243 | noise floor **0.243** |
| 8x3 | 24 | 4 | 32.624 | +0.175 | 1 / 36 |
| 8x4 ← ships | 32 | 3 | 32.799 | — | — |
| 6x4 | 24 | 4 | 32.875 | −0.076 | 0 / 36 |
| 12x4 | 48 | 2 | 32.986 | −0.187 | 0 / 36 |
| 8x6 | 48 | 2 | 33.096 | −0.297 | 2 / 36 |
| 12x8 | 96 | 1 | 34.908 | −2.109 | 0 / 36 |

Same shape as §6.39 in every respect: interleaved wins by ~4 ms (17x the noise floor), the
sharded curve is monotone with fewer cores better, and the newly-reachable 96 cores is worst.
**So the reversal is a property of the CHIP, not of Block 1** — two independent blocks, two
different call counts, same answer.

**GATED AND SHIPPED**, but the route there is the most instructive part of this section.

**⚠ AT THREE SEEDS THIS LOOKED LIKE A REGRESSION AND I RECOMMENDED REJECTING IT.** Long-form
errors per 298 words, one process per (arm, seed):

| seed | sharded 8x4 | interleaved |
|---|---|---|
| 0 | 1 | 2 |
| 1 | 0 | 2 |
| 2 | 0 | 0 |
| **3** | **3** | **0** |
| **4** | **1** | **0** |
| **5** | **1** | **0** |
| **total** | **6 of 1788 (0.34%)** | **4 of 1788 (0.22%)** |

Seeds 0–2 read 1 vs 4 and I called it "a reproducible inflection drop in 2 of 3 seeds", which is
§6.7's criterion for real signal. Seeds 3–5 read 3/1/1 vs 0/0/0 and the sign flips: over six
seeds the SHARDED arm is worse. §6.7 said this in July — *"every single-seed WER comparison made
during that sweep was uninformative"* — and three is not enough either when the effect is this
small. **Do not adjudicate a sub-1%-of-corpus WER difference on three seeds.**

**AND THE DETERMINISTIC EVIDENCE WAS MISREAD FIRST, WHICH IS THE WORSE ERROR.** The sweep table
above has a `codes != 8x4` column, and a nonzero value there was reported as the gate "mildly
agreeing" the change was worse. It does not: that column measures divergence from the SHIPPED
CONFIG, and §6.25 is explicit that the default is not ground truth. Scored against the fp32 CPU
reference instead, 8 real prompts (§6.10's methodology):

| arm | acoustic != fp32 | semantic wrong | velocity maxabs | velocity PCC |
|---|---|---|---|---|
| sharded 8x4 | 10 / 288 | 0 / 8 | 3.233e-02 | 0.99997914 |
| **interleaved** | **10 / 288** | **0 / 8** | **2.569e-02** | **0.99998504** |

Identical code and semantic accuracy, and interleaved is **21% closer to truth on the velocity**.
It is not less precise — it is marginally more precise, exactly as in §6.39. `--gate flow` agrees
(3 of 74 codes both arms, velocity PCC 0.99998438 vs 0.99998415) though that gate runs on
`make_synthetic_inputs`, i.e. random activations, so it is trap-#12 evidence and the real-prompt
table above is the one to quote.

A mechanism prediction that was ALSO wrong, recorded so it is not reasoned from again: sharding
computes 32 partial sums plus a cross-core combine, i.e. a shallower accumulation than one core
summing 3072 values, and shallower trees round better — so sharded "should" be the more accurate
one. Measured, it is the less accurate one, in both blocks. At ~1e-7 per op, which config lands
nearer fp32 after 21 amplifying `_block` calls is not predictable from the reduction shape.

**Shipped:** long-form gen **57.82 -> 53.30 ms/frame** over all 6 seeds (n=21/22 long-form
cases), 15/15 `[END_AUDIO]` in all twelve runs, WER better on the 6-seed total. `TILE` in
`ttnn_voxtral_flow.py` is deleted with the shard configs — it had no other consumer.

**Note also what this says about the p150 gap.** Block 2 measures 32.8 ms/frame sharded / 28.6
interleaved here against §6.31's **20.8 ms on the N150** — so after §6.39 fixed Block 1 (now
21.4 ms against the N150's 23.15, i.e. FASTER than Wormhole), essentially the whole remaining
p150 deficit is Block 2.

### 6.41 — p150: the DRAM ceiling is ~360 GB/s, not 194–202. Re-derive before ranking anything

The floor method (`op time = bytes/ceiling + overhead`) is still right; its denominator was not.
194–202 GB/s is an N150 number and §1–§6.38 rank every optimization against it. Measured here the
way §6.24/§6.27 measured it there — effective GB/s = weight bytes / time for `[1,1,K] @ [K,N]`:

| weight | shape | dtype | MB | µs | GB/s | % of ~360 |
|---|---|---|---|---|---|---|
| B2 semantic | 3072x8320 | fp32 | 102.2 | 275.3 | **372** | peak |
| B1 w1 / w3 | 3072x9216 | BFP8 | 30.1 | 86.3 / 86.6 | 350 / 349 | 97% |
| B1 / B2 wqkv | 3072x6144 | BFP8 | 20.1 | 72.1 / 72.5 | 279 / 277 | 78% |
| B1 w2 | 9216x3072 | bf16 | 56.6 | 206.4 | 274 | 76% |
| **B1 wo** | 4096x3072 | BFP8 | 13.4 | 94.1 | **142** | **39%** |
| **B2 w2** | 9216x3072 | BFP8 | 30.1 | 206.6 | **146** | **41%** |

**THREE RECORDED CONCLUSIONS DIE HERE.**

**1. §6.24's fusion rule is void.** A width sweep at K=3072, BFP8:

| N | 1024 | 3072 | 4096 | 6144 | 9216 | 12288 | 18432 | 24576 |
|---|---|---|---|---|---|---|---|---|
| GB/s | 73 | 142 | 154 | 277 | 348 | 356 | **359** | 349 |

18432 is the FASTEST point measured. §6.24 recorded it at 48 GB/s and 0.250x on the N150 and
generalised to *"assume any fusion past N~9216 collapses until measured"*. That rule is a
Wormhole artifact. **The penalty here is for NARROW N**, which is the inverse — and it is why
`wo` and `w2` (both N=3072) are slow: the shape, not the config.

**2. §6.38's w2-BFP8 question is closed as "no gain".** Same 9216x3072 shape: bf16 reads 56.6 MB
in 206.4 µs, BFP8 reads 30.1 MB in **206.6 µs**. Half the bytes, identical time — this shape is
not bandwidth-bound at all. §6.38 called w2-BFP8 *"the largest single win left anywhere in this
model"* at 2.644 ms/step and left it open. On the p150 it would cost §6.16's 0.24 pp of mean
worst-sample for **zero** speed. This also explains §6.17's puzzle (BFP4 cut bytes 47% and
returned 12% of the time) on the right terms.

**3. Distance-from-roofline is NOT a ranking signal here.** `wo` at 39% looked like the largest
headroom in the model. §6.43 shows it is entirely overlapped away and unreachable. Use the
ceiling to decide what is IMPOSSIBLE, not what is profitable.

Method caveat: a "pure read" proxy (`ttnn.sum` over 64/256 MB) gave 255/307 GB/s, BELOW the
matmul figure — a reduction carries its own cost and is a poor bandwidth probe. The matmul
numbers are the comparable ones, since that is what the N150 figure was.

### 6.42 — p150: w1+w3 fusion re-tested. Still rejected, for an entirely different reason

§6.24's rule is void (§6.41), so the fusion it forbade was re-measured. The matmul half reverses
completely; the decision does not.

| | N150 (§6.24) | p150 |
|---|---|---|
| 2 x `[3072,9216]` | 313.9 µs / 192 GB/s | 166.5 µs / 362 GB/s |
| 1 x `[3072,18432]` | 1253.5 µs / **48 GB/s** | 161.6 µs / **372 GB/s** |
| fused vs separate | **0.250x** | **1.03x** — fused is FASTER |

But on the whole MLP, 10 rounds x 200 reps, interleaved:

| | Block 1 `[1,1,3072]` | Block 2 `[1,6,3072]` |
|---|---|---|
| A separate + fused silu + multiply (ships) | **216.34 µs** | **214.95 µs** |
| C fused + `ttnn.swiglu` (2 ops, one FEWER than A) | 237.57 | 237.83 |
| A − C | **−21.24 µs/layer** | **−22.88 µs/layer** |

Fusing would COST ~0.55 ms/frame in Block 1 (x26) and ~0.48 in Block 2 (x21). Two reasons, both
downstream of the matmul: `activation="silu"` rides free on the separate w1 (§6.17 change A) and
a fused output cannot apply it to half; and splitting an 18432-wide output costs ~26 µs against
the 4.9 µs the launch saved.

**The replacement rule for this chip: fusion pays only when the output does not need splitting.**
That is why the qkv fusion still holds — its consumer slices anyway — and why this one does not:
it ADDS a split. Two of §6.24's subsidiary objections also dissolve: the 5-op form measured
**bit-exact** against A (maxabs 0.000e+00), and the memory objection was only true if both weight
forms are kept.

**⚠ TWO CORRECTIONS TO THIS SECTION'S FIRST WRITE-UP.** It called the swiglu arm "2 ops, one
FEWER than what ships" and reported its timing without ever checking its numerics — the probe's
correctness branch was gated on `nops >= 3` and that arm was labelled 2, so it was the one arm
that skipped verification. Re-measured properly:

| rows | arm | out shape | µs | maxabs vs fp64 |
|---|---|---|---|---|
| 1 | shipped | `(1,1,9216)` | **197.9** | 1.619e-05 |
| 1 | swiglu `[w3\|w1]`, no slice | `(1,1,32,9216)` | 231.1 | 1.619e-05 |
| 1 | swiglu `[w3\|w1]` + slice | `(1,1,1,9216)` | 263.1 | 1.619e-05 |
| 6 | shipped | `(1,6,9216)` | **213.1** | 1.362e-05 |
| 6 | swiglu + slice | `(1,1,6,9216)` | 264.2 | 1.362e-05 |

`ttnn.swiglu` computes **exactly the shipped answer** — so the unverified arm was in fact fine,
but the claim had no support at the time. It is still rejected, and on firmer ground: 1.17x
slower even without the slice, 1.33x with it. And §6.24's tile-row note is real, so the shippable
form is **3 ops, not 2** — the output returns `[1,1,32,9216]` with 31 padding rows.

**A third correction, to the inherited record.** ONBOARDING §8 (from §6.37) said `ttnn.swiglu`
*"TT_THROWs on a concatenated pair"*. It does not here: `concat([w3out, w1out])` → `swiglu` ran
and returned `(1,1,32,9216)`. Either the p150 differs or §6.37 was testing a different call shape.

**And one thing this could NOT establish.** A `[w1|w3]`-ordered arm was included expecting it to
be obviously wrong; it came back at 1.104e-04 / PCC 0.9996, i.e. nearly correct. That is not
evidence the order is free — it is the inputs being too small. At `randn*0.02`, `silu(x) ≈ 0.5x`,
so `silu(a)·b ≈ silu(b)·a` and the test has no power. **The `[w3|w1]` order rests on §6.24's
documentation, not on this measurement.**

### 6.43 — p150: `_WO_PRG` deleted. Bit-exact, and no instrument can find a speed difference

`wo` is the worst weight matmul in Block 1 by isolated bandwidth (§6.41), so it got the full
grid x `in0_block_w` sweep. The sweep found large wins and **none of them exist on the step.**

| config | isolated µs | GB/s | maxabs vs fp64 | == default |
|---|---|---|---|---|
| default | 92.9 | 144 | 2.854e-04 | yes |
| **8x4 ib=2 (was shipped)** | 63.7 | 210 | 2.854e-04 | yes |
| 8x4 ib=4 | 47.3 | 283 | 2.854e-04 | no |
| **12x2 ib=8** | **37.8** | **354** | 2.854e-04 | no |
| 12x4 ib=32 | 45.1 | 297 | **2.281e-04** | no |

**Every config sits at the same distance from fp64 truth as the default, several closer**, while
none is bit-identical to it — and bit-equality-vs-default is exactly the criterion `ib=2` was
originally chosen on. §6.25 warned about this in as many words; had a real win been here, that
criterion would have blocked it.

**But the whole step says nothing is there.** 12x2 ib=8 is 1.69x on the op, predicting
+0.672 ms/frame over 26 layers:

| config | ms/step | vs shipped |
|---|---|---|
| 8x4 ib=2 #control | 21.418 | +0.136 |
| default | 21.449 | +0.105 |
| 8x4 ib=2 (was shipped) | 21.554 | — |
| 12x2 ib=8 | 21.563 | −0.009 |

Noise floor 0.136 ms; every delta at or below it, and the ordering scrambles between rounds.
~1.4 ms of isolated `wo` time is already overlapped away — §6.27's observation that isolated ops
summed to 29.9 ms against a 23.8 ms step, in its sharpest form yet.

**So the config was deleted.** Removing it is bit-exact: `torch.equal` on the 26-layer output,
and **all 45 utterances of a 15-case x 3-seed run reproduced identical frame counts** (§6.32's
gate at full strength). On speed, three instruments and none can resolve it: whole step +0.174 ms
(spread 1.7–1.9, resolution ~0.3), pipeline −0.81 / +1.20 / −2.11 per seed with the signs
flipping, mean −0.40. INCONCLUSIVE by rule 3, so it goes on the grounds that a hand-tuned
constant whose recorded justification (§6.25's +0.196 ms/frame) does not reproduce is not worth
carrying. If it is ever missed, it is one line.

**A measurement-health note.** The microbenchmark's spread degraded ~40x mid-session — §6.39's
harness ran at 0.005–0.049 ms and the same code later ran at 1.7–1.9 ms — with host load at 0.69,
no other users, AICLK steady at 800 MHz and no thermal throttle. **Unexplained.** Numbers taken
after that point resolve ~±0.3 ms at best, which is why this section leans on the frame-count
gate and the pipeline rather than the step timing.

### 6.44 — p150: the fused cache write LOSES, and _V_SHARD goes with it

§6.20/§6.22 measured `paged_fused_update_cache` at +0.454 ms/frame on the N150. Here it is
**0.687 ms/step SLOWER** than two plain `paged_update_cache` calls. Both are bit-identical.

`_V_SHARD` existed only so the fused op would accept K and V on different cores, so it is deleted
too — and with it the silent failure mode §6.22 warned about, where RoPE on a core whose cos/sin
table lives elsewhere returns **3.4e38 from uninitialised L1** instead of raising.

`_QKV_GRID_X` moves 8 → 1 in the same pass. §6.19 measured 1 core at +0.273 ms **worse** on the
N150; here it is 0.461 ms **better**. Its structural finding was re-verified and still holds: feed
`nlp_create_qkv_heads_decode` 1 / 6 / 8 / 24 / 48 cores and `qh`, `kh`, `vh` all come out **1
core, shard (32,128)**, every time. The grid cannot reach its consumers, so only the shard fill
sees it — and filling fewer cores is cheaper.

Dedicated A/B, 14 rounds, noise floor **0.059 ms**, everything bit-identical:

| arm | ms/step | vs shipped |
|---|---|---|
| **2 writes + 1-core qkv ← SHIPS** | **21.203** | **+0.907** |
| 2 writes + 8-core | 21.423 | +0.687 |
| fused + 1-core | 21.649 | +0.461 |
| fused + 8-core (was shipped) | 22.110 | — |

### 6.45 — p150: a small op costs 3.4x more, and THREE op-count decisions reverse

**THE MEASUREMENT THAT EXPLAINS §6.44 AND THIS ONE.** A trivial op at Block 2's shape:

    ttnn.add on [1,6,3072]     p150 67.7 us     N150 ~20 us (6.38)

Block 2's five weight matmuls sum to **11.5 ms/frame here against the N150's 14.0** — better, as
§6.41 predicts. Yet the block measured **28.4 against 20.8**. So ~17 ms is non-matmul work here
against ~6.8 there. **On this chip OP COUNT is the dominant term**, which directly inverts §6.6's
rule — *"what actually works is fewer, BIGGER kernels… judge a proposal on whether it makes
kernels BIGGER, not on how many ops it deletes"*. On Blackhole, deleting ops is exactly the win.

Whole Block 2 frame, interleaved, noise floor **0.002 ms**:

| arm | ms/frame | vs shipped | codes ≠ fp32 |
|---|---|---|---|
| **fused split + sdpa ← SHIPS** | **21.861** | **+6.586** | 1/36 |
| fused split (undoes §6.31) | 24.610 | +3.836 | 1/36 |
| sdpa (undoes §6.37) | 25.892 | +2.555 | 1/36 |
| shipped | 28.447 | — | 1/36 |

**1. The head split — §6.31 reverses, at IDENTICAL accuracy.** The fused `nlp_create_qkv_heads`
replaces the 9-op hand-roll. Scored on 8 real prompts against the fp32 reference: 10/288 acoustic
codes, 0/8 semantic, velocity maxabs 2.569e-02, PCC 0.99998504 — *the same numbers as the
hand-rolled path*, not merely close. §6.33 had already established that the hand-rolled win was a
**system effect** rather than the op being faster, and a system effect is precisely what does not
travel between chips.

**2. sdpa for the interior — §6.37 reverses, and the accuracy objection largely dissolves.**
4 ops → 1, and it handles GQA natively so the row fold and `REP` are deleted. §6.37 rejected it at
**6.48x** the error vs fp64 and one WER word; here it is **1.57x** (velocity maxabs 4.043e-02 vs
2.569e-02) and the discrete output does not move — 10/288 codes and 0/8 semantic, identical, and
`--gate flow` reads **2 of 74 against 3**. [flow-03] is explicit that codes, not PCC, are Block
2's gate. Long-form WER, 15 cases x 3 seeds:

| seed | 0 | 1 | 2 | total |
|---|---|---|---|---|
| without sdpa | 2 | 2 | 0 | 4 / 894 |
| **with sdpa** | 1 | 0 | 0 | **1 / 894** |

15/15 `[END_AUDIO]` in all six runs, and gen **48.00 → 45.38 ms/frame** long-form.

**3. `scale=1.0` is mandatory** — SCALE is folded into wqkv's q rows ([flow-09]), so the default
applies 1/sqrt(d) twice. §6.37 measured 3.8e-01 relative error for forgetting it.

**THE GENERALISABLE RULE FOR THIS CHIP.** Wormhole rewarded fewer, bigger kernels because its
per-op floor was ~20 us and its DRAM ceiling was 194 GB/s. Blackhole has ~360 GB/s and a ~68 us
per-op floor, so the balance flips: **bytes are cheap and launches are expensive.** Every
N150-era decision of the form "trade one op for several to get a better layout" should be
re-tested here, and so far all three that were have reversed.

### 6.46 — p150: `_SDPA_PRG` swept and KEPT. The one N150 program config that survived

`wo`'s config was deleted (§6.43) so `sdpa_decode`'s got the same treatment: grid x `k_chunk` x
`q_chunk`, plus "is it needed at all". Unlike `wo`, it earns its place — and the faster
alternatives are unsafe in exactly the way §6.27 predicted.

**IT IS NEEDED.** Isolated **1.751x** over the default (45.3 -> 25.9 µs), and unlike `_WO_PRG`
the win survives the whole step:

| | isolated | whole step | verdict |
|---|---|---|---|
| `_WO_PRG` | 1.46x | 0, signs flipping | deleted (§6.43) |
| **`_SDPA_PRG`** | **1.751x** | **+0.197 ms**, mean/median/min same sign | **kept** |

Bit-identical to the default on the 26-layer output, so this was purely a speed question.

**THE CONFIG STAYS AS IT IS, and §6.27's warning reproduced on new hardware.** Faster options
exist at pos=312 — 8x1 k=128 at 22.8 µs (1.985x) against the shipped 25.9 — but the position
sweep kills them. Error vs fp64 at 13 positions spanning the chunk boundaries:

| config | µs | positions worse than default |
|---|---|---|
| 8x2 k=128 | 23.1 | 3 / 13 |
| 8x1 k=128 | 22.8 | 2 / 13 |
| 8x1 k=512 | 23.4 | 1 / 13 |
| 8x1 k=256 | 23.5 | 1 / 13 |
| **8x2 k=512 ← ships** | 25.9 | **0 / 13** |

The shipped config reproduces the default's error *exactly* at every position. `8x1 k=128` is
1.7x worse at **pos=128** — precisely its own chunk boundary — and 1.9x worse at pos=1000. On the
N150 it was `k=256` failing at 480/511/700; same mechanism, different config, and a
single-position gate would have shipped either one. **The extra 3.1 µs/layer (0.08 ms/frame) is
not worth position-dependent degradation on long utterances**, which is where the headline
quality claim lives.

**The grid axis is inert** — 8x1/8x2/8x4 land within a few µs at fixed `k`, so the 13x10 grid
buys nothing here either. `k_chunk` is the knob, exactly as [gpt-21] says.

**Why this one survived and `wo`'s did not** is unverified: plausibly `wo` sits between two large
matmuls that hide it while `sdpa_decode` has less to overlap with. Measured, not established.

### 6.47 — p150: in-place elementwise, +0.929 ms/step. Residual-as-bias rejected

§6.45's map put five trivial ops — two residual adds, two norms and one multiply — at **8.3
ms/frame, 39% of the step**, every one of them at the ~65 µs launch floor rather than doing any
arithmetic. Two attacks on that, one of which worked.

**SHIPPED — in-place, +0.929 ms/step, bit-identical.** `multiply_` and both `add_`. §6.37
measured exactly this on the N150 at **+0.001 ms**, i.e. indistinguishable from nothing. It is
~1000x that here, because in-place removes an ALLOCATION rather than a launch, and the allocator
is roughly 12 of the 65 µs. Frame counts reproduce 68/452/493 on cases 0/2/3, so it is bit-exact
end to end.

**REJECTED — residual-as-bias.** In decode M=1, so the residual is exactly a row-vector bias and
`linear(a, wo, bias=x)` is expressible and bit-identical. Against a 0.062 ms noise floor:

| arm | ms/step | vs shipped |
|---|---|---|
| **in-place only ← SHIPS** | **20.405** | **+0.929** |
| both bias + in-place | 21.105 | +0.229 |
| w2 bias | 21.257 | +0.076 |
| wo bias | 21.264 | +0.069 |
| shipped | 21.333 | — |
| both bias | 21.481 | −0.148 |

§6.27's N150 verdict (w2's add already free, wo's worth 4 µs) therefore stands — and the two
ideas are **anti-complementary**: bias removes the adds that in-place accelerates, so the
combination is worth a fifth of in-place alone.

**⚠ THE PREDICTION THAT MOTIVATED THIS WAS WRONG BY ~48x, AND IT IS THE THIRD TIME TODAY.** The
map showed each add at 65 µs isolated, so I estimated residual-as-bias at "up to 3.35 ms/frame".
It delivered 0.07. That is precisely the inference §6.43 was written to forbid — `wo` showed
1.69x isolated and zero on the step — and I made it anyway, two sections later. **Isolated op
cost predicts nothing about what removing the op is worth. Rank with the map, decide on the
step**, every time.

**One idea that is NOT available, recorded so it is not proposed again.** `u = g * w3_out` cannot
be folded into either matmul: it is an elementwise product of two separate matmul outputs. The
only op that eliminates it is `ttnn.swiglu`, which requires w1|w3 fused, and §6.42 measured that
at 21–23 µs/layer worse at this exact shape. The multiply can only be made cheaper, not removed.

### 6.48 — p150: in-place in Block 2 too, +0.790 ms/frame — and the L1 trap inside it

§6.47 shipped in-place in Block 1. The same treatment for every remaining elementwise site, five
of them, measured cumulatively on the whole Block 2 frame:

| arm | ms/frame | vs shipped | ≠ shipped | ≠ fp32 |
|---|---|---|---|---|
| **`_block` mul + resid, concat→L1 ← SHIPS** | **21.104** | **+0.790** | **0/36** | **1/36** |
| `_block` mul + resid, x in DRAM | 21.262 | +0.631 | **1/36** | **2/36** |
| `_block` mul only | 21.711 | +0.183 | 0/36 | 1/36 |
| shipped | 21.894 | — | — | 1/36 |
| `_solve` cfg / euler / both | 21.05–21.18 | ±0.07 | 0/36 | 1/36 |

**THE TRAP IS WORTH MORE THAN THE MS.** `add_` writes wherever its first operand ALREADY lives.
The shipped `add(x, r, memory_config=_L1)` put the residual in L1 on purpose ([flow-02], 1.049x);
`add_` cannot be told where to write, so with `x` arriving from `_trunk` in DRAM the rewrite
**silently reverted that decision** — and per [flow-10] an L1-vs-DRAM operand changes the
downstream matmul's program config, so it moved a code as well: 1/36 against the previous build,
2/36 against fp32 where shipped is 1.

Fixing it needs one kwarg — `_trunk`'s concat gains `memory_config=_L1`, so the residual stream is
BORN in L1 and `add_` inherits it. That version is both faster (+0.790 vs +0.631) and
accuracy-neutral. **`memory_config` on that concat is now load-bearing: remove it and the in-place
adds change the model with no error.**

**`_solve` in-place is not taken.** Its CFG combine and Euler update run 7x a frame against
`_block`'s 21, and every arm landed inside the 0.015 ms noise floor.

**The generalisable point:** in-place is not a free syntactic swap on this chip. It removes an
allocation (§6.47, ~12 µs) but it also **surrenders control of where the result lands**, and this
port has spent real effort deciding exactly that (§6.10, [flow-02], [gpt-03]). Check the operand's
memory config before assuming the rewrite is neutral.

### 6.49 — p150: host dispatch is 3-4%, NOT 0%. Tracing is worth 4x what it was on the N150

The measurement that validates the reasoning behind §6.41–§6.48. Every per-op number in those
sections is `perf_counter` around `synchronize_device`, which INCLUDES host dispatch. §6.38
answered that objection on the N150 — `_solve` eager 19.145 vs traced 19.230, dispatch 0% — but
that was a different chip AND a different host, and this box has 8 shared cores.

| | eager | traced | dispatch | N150 |
|---|---|---|---|---|
| Block 2 `_solve` (~600 ops) | 19.664 ms | 18.888 | **3.9%** (+0.776) | 0% (§6.38) |
| Block 1 26-layer step (~470 ops) | 19.717 ms | 19.156 | **2.8%** (+0.561) | +0.17 ms (§6.26) |
| **both** | **39.38** | **38.04** | **+1.34 ms/frame (3.4%)** | +0.35 ms |

**THE OP-COUNT STRATEGY IS VALIDATED.** ~96% of the 67.7 µs per-op floor is genuine DEVICE time,
so §6.41–§6.48 were not secretly measuring this host, and the six reversals stand.

**But tracing is worth ~4x what §6.26 rejected** — 3.4% against 0.7%. §6.26's argument was about
KIND, not size: *"0.7% is not worth converting four loud failure modes into three silent ones."*
Those failure modes are unchanged and all still real — buffers held by pointer so a fresh upload
is silently ignored; warm-up/capture writing the KV cache (decode PCC 0.9998 → 0.86, no error);
a capture-order constraint across the two blocks; and trap #1, where an exception inside a
capture wedges the card. **Not shipped**; the probe is kept.

Two caveats on the 1.34 ms. It covers 39.4 of the pipeline's ~44 ms/frame — the host steps cannot
be traced (§6.50). And `95dc26363f` measured that merely passing `trace_region_size` moves
free-running trajectories (case 2: 458 → 464 frames with the trace OFF), so shipping it changes
generated audio before the trace runs and needs the full multi-seed WER gate, not a frame-count
check.

**A SIDE FINDING WORTH MORE THAN THE VERDICT.** Traced spread is **0.003–0.004 ms** against
eager's 0.134–0.800. Run-to-run variance here is almost entirely HOST-side — the first real lead
on §6.43's unexplained 40x noise degradation, and it means a traced harness is a far better
instrument for small effects than anything used this session, whether or not tracing ever ships.

### 6.50 — p150: the three host steps stay on host. Device is 7-29x slower

Asked whether `semantic_code`'s argmax, `embed_frame` and the FSQ quantise should move on device,
partly to make the whole frame traceable (§6.49). No, emphatically:

| step | host | device | |
|---|---|---|---|
| `embed_frame` (37-codebook gather + sum) | **57.3 µs** | 432.6 µs | device 7.5x slower |
| FSQ quantise (clamp/scale/round on [B,36]) | **13.6 µs** | 230.7 µs | device 17x slower |
| semantic mask + argmax over [1,8320] | **10.9 µs** | 321.1 µs | device 29x slower |

Moving all three costs **+0.90 ms/frame**. §6.45's rule at its purest: each is 1–4 launches at
~68–230 µs doing microseconds of actual work. §6.31 reached the same conclusion on the N150 for
the argmax (device 1213.5 µs vs host 860.3) and the p150's higher per-op floor only widens it.

The FSQ device form is CORRECT (0 of 36 codes differ), so this is purely a speed verdict. The
bf16 embedding table injects rel **5.48e-04** against the fp32 host gather — small, but ahead of
a 26-layer stack, which is the accuracy reason [pipe-01] gave for keeping it on host in the first
place. So it would not be free even at speed parity.

**THE HOST TAIL IS ALSO MUCH SMALLER THAN THIS FILE HAS BEEN CLAIMING: 81.8 µs/frame in total**,
against [pipe-01]'s "~0.2 ms" for `embed_frame` alone and [flow-01]'s "~0.7 ms host tail". That is
**0.19% of a 44 ms frame** — not worth attacking from either direction.

It also removes the motivation for a full-frame trace: you would spend +0.90 ms of device ops to
make the frame monolithic, in order to recover dispatch that §6.49's two-region trace already
captures. **Trace the two device graphs or nothing; leave the 82 µs of host work alone.**

### 6.51 — p150: what the rest of the repo and `ign/voxtral_p150_qb2` actually have

Asked to check both for Blackhole techniques worth stealing. Net: **nothing that changes a shipped
decision**, and the one thing that would have is aimed at a bottleneck we measured at 3–4%.

**The repo.** 215 files reference `is_blackhole`, overwhelmingly vision/CNN test skips and grid
literals. Three substantive patterns, none applicable:

| pattern | where | why not us |
|---|---|---|
| `grid=(8,10) if is_blackhole() else (8,8)` | `attention_1d.py` prefill | we don't use `models/common`'s attention |
| `dram_shard_grid_width = dram_grid_size().x`, "avoids silent PCC issues on P100" | `attention_1d.py` | §6.28 rejected DRAM-sharded matmuls; §6.43 showed the gap is overlapped |
| `CoreCoord(8,8) if is_blackhole()` for batch corerangesets | rope / sampling | batch 1 |

**2 command queues are a vision-only idiom here.** `num_command_queues=2` appears in resnet50,
mobilenetv2, unet and the `tt_cnn` pipelines — every one a model streaming *large host inputs*.
**No LLM or decode model in the repo uses it**, and the reason is structural: our per-step host
input is `embed_frame`'s 6 KB output (§6.50 measured the whole host tail at 82 µs). There is
nothing to overlap.

Also confirmed directly: `compute_with_storage_grid_size()` returns **13×10** and already excludes
dispatch cores, so ign's eastern-column warning is about *their* `find_grid` helper, not the API.

**`ign/voxtral_p150_qb2` (30322d621c, 2026-06-19, 1235 commits behind).** Their whole performance
architecture rests on a premise that **contradicts our measurement**: *"each of the ~hundreds of
tiny ttnn ops per token finishes on-device in microseconds, then waits ~100 µs for the host to
enqueue the next op."* §6.49 measures dispatch at **2.8–3.9%**. At 100 µs/op, Block 1's 470 ops
would be 47 ms of pure dispatch against a whole step of 19.7 ms — arithmetically impossible on our
tree. The likely explanation is that tt-metal dispatch improved enormously in 1235 commits. §6.5
already noted their dispatch-bound claim "does not transfer"; it now does not transfer **even on
their own target hardware**. So their 2CQ+trace design is not worth copying.

Four concrete P150 facts from them worth having, none currently biting us:
- **`packer_l1_acc=False` avoids P150 static-CB clashes on long conv sequences.** We use `True`
  everywhere including the codec. First thing to try if the codec ever throws on a long utterance.
- **P150 concat CB page limit ≈768k bf16 elements on dim=2.** Our codec concatenates 3.8M at
  T=469 and works, so it is version- or shape-specific — but the failure mode exists.
- Conv-transpose output chunking at 1024 for P150 L1 pressure.
- `_voxtral_worker_shard_cap` caps worker x at 8 on Blackhole.

**The useful result is negative and worth stating plainly: there is no LLM-decode Blackhole
playbook in this repo to copy from.** Everything in §6.39–§6.52 had to be measured.

### 6.52 — p150: decode matmul program configs, −4.24 ms/frame. And `activation="silu"` never fused

Asked to give the remaining matmuls the treatment §6.43 gave `wo`. Both blocks share dims exactly
and both are one tile of rows, so four shapes cover all eight sites.

**The sweep found the ttnn heuristic collapses on deep reductions.** Achieved bandwidth, isolated:

| matmul | K-tiles | default | tuned `in0_block_w` | |
|---|---|---|---|---|
| `w1`/`w3` | 96 | 352 GB/s | 362 | 1.03x |
| `wqkv` | 96 | 281 | 346 | 1.23x |
| `wo` | 128 | **144** | 336 | **2.33x** |
| `w2` | 288 | **147** | 355 | **2.43x** |

At Kt=96 the heuristic already reaches this chip's measured 367 GB/s ceiling. At Kt=128/288 it
delivers **under half the memory system**. That is the same reduction-depth effect §6.53 measures
from the other direction, and §6.50's note anticipated the shape of it: *"it was measured on wq,
which is already at 94% of its floor. Sweeping an op with no headroom finds none."*

**THE ISOLATED WINS INVERTED IN THE BLOCK, AND THIS IS THE REAL RESULT.** Whole-block A/B:

| arm | Block 1 | Block 2 | total | vs shipped | isolated predicted |
|---|---|---|---|---|---|
| **ALL, uniform 12×6** | 17.87 | 19.12 | **36.99** | **−4.24** | −9.0 |
| ALL, per-shape winners | 17.96 | 19.08 | 37.04 | −4.18 | −9.0 |
| ALL, uniform 13×10 | 18.15 | 19.15 | 37.30 | −3.93 | |
| w1 only | 19.10 | 19.71 | 38.81 | −2.42 | −0.11 |
| **w1 config, silu NOT fused** | | | 41.73 | **+0.33** | |
| w3 config only | | | 41.79 | +0.39 | −0.11 |
| shipped | 20.23 | 20.99 | 41.23 | — | |

noise floor 0.070 ms. `w2` and `wo`, with the **2.4× isolated wins, delivered 0.00 ms**. `w1`,
with essentially none, delivered **2.42**.

**The control settles why.** Same program config with silu as a separate op gives back every
millisecond (+0.33). So the w1 win is not the grid at all — it is that **`activation="silu"` is
not fused**:

| | µs |
|---|---|
| plain matmul, no activation | 85.5 |
| `activation="silu"` — what shipped | 98.8 |
| `activation=UnaryWithParam(SILU)` / `UnaryOpType.SILU` | 100.6 / 100.2 |
| explicit `ttnn.silu(linear(...))` — deliberately two ops | 101.8 |
| **program config `fused_activation`** | **88.1** |

The kwarg costs exactly what a separate op costs, because that is what it is. Only
`fused_activation` folds it in — and it is *more* accurate too (PCC 0.9999984 vs 0.9999970),
since the value never leaves the dest registers. Across 47 w1 calls per frame: **2.42 ms**.

This invalidates `[flow-12]`, which claimed SiLU "rides along on the w1 matmul instead of being
its own op". **It was its own op on both chips**, for the entire life of this port. The repo's
serious LLM models (deepseek_v3, llama3_70b_galaxy, blackhole/qwen36, gemma3) all use
`fused_activation`; only 13 call sites repo-wide use the string kwarg, and ours were among them.

**A METHODOLOGICAL RESULT THAT EXPLAINS TODAY'S REPEATED MISPREDICTIONS.** The silu op costs
**12.2 µs isolated but ~54 µs in-block** (2.42 ms ÷ 47) — near the full ~68 µs floor. A tight loop
of *identical* ops pipelines; a real block of *differing* ops does not. So isolated
microbenchmarks understate op cost by ~4×, and overstate bandwidth wins that were really
pipelining. This is §6.43's rule with a cause attached: **isolated sweeps measure pipelining,
blocks measure dispatch.** It is why §6.47's residual-as-bias estimate missed by 48×, and why the
2.4× on `w2` was never real.

**ONE grid, not four.** Uniform 12×6 ties a per-shape set of isolated winners (36.99 vs 37.04,
under the noise floor), so no shape earns its own constant — which matters on a branch where
seven tuned constants have already had to be re-derived. 13×10, the full 130 cores, is worse.

**Decode only.** `per_core_M=1` / `fuse_batch=True` assume one tile of rows, which prefill
violates; `_mlp` is shared, so the configs are passed in as an argument and prefill passes none.
Guarded by `test_decode_matmul_configs_assume_one_tile_of_rows`.

**Gates.** 129 pytest (three new guards; the stale `test_wo_has_no_program_config` was passing
vacuously on a name check while its docstring had become false, and is rewritten).
`--gate wiring/prefill26/flow/codec/decode` all at baseline — `flow` reads 2 of 74, matching the
record. `--gate codes` is **not** bit-exact and needed a PAIRED run to read at all: baseline
86/288 acoustic, this change **85/288**, semantic 1 both. The 10/288 figures elsewhere in NOTES
are from a different state and are not comparable — reading against them looked like an 8.5×
regression that does not exist.

**FRAME-COUNT A/B AND WER, 15 cases × 3 seeds, one process per (arm, seed) per §6.21.** Frame
counts are **not** preserved — 6 of 15 match on seed 0 — so the exactness test is inconclusive by
construction and the change goes to WER, which is where it is decided:

| arm | long-form (the gate) | short | `[END_AUDIO]` | ms/frame | RTF |
|---|---|---|---|---|---|
| baseline | **1 wrong of 894** | 8 of 126 | 45/45 | 44.19 (43.75 / 44.78 / 44.04) | 0.577 |
| decode configs | **1 wrong of 894** | 9 of 126 | 45/45 | **39.14** (39.21 / 38.84 / 39.36) | **0.507** |

**WER is identical**, every utterance terminates, and the short bucket's 8→9 is inside its
documented seed noise (§6.7). End-to-end **−5.06 ms/frame, 1.129×**.

**TIMING EXCLUDES CASE 0, WER DOES NOT.** Case 0 is the first utterance in each process and pays
one-time program-cache compilation: **3.3 s of non-generation time over 5.4 s of audio, an RTF of
1.346**. Including it inflates the baseline to 47.89 / 0.759 and this change to 42.92 / 0.694 —
the *delta* survives, because both arms pay it equally, but the absolute numbers are meaningless
and do not reconcile with the 45.4 / 0.57 on record. **Any future RTF number from this harness
must drop the first case**, or it will look like a 0.19 RTF regression that never happened.

**Still unexplained, and flagged rather than papered over:** `wqkv` + `wo` + `w2` contribute
**0.00 ms on their own** but a further **−1.8 ms once w1's silu is fused** (reproduced twice:
−1.70 and −1.82). Grid consistency is not the mechanism — a mixed-grid set ties uniform 12×6, and
uniform 13×10 is worse. Whatever it is, the combination is measured three times and gates clean.

### 6.53 — why a "much more powerful" p150 was only ~7% faster than the N150

Asked directly, and it is worth having the answer written down because it bounds every future
optimisation on this branch. Both ceilings measured here:

| ceiling | p150 measured | what one frame used (pre-§6.52) | fraction |
|---|---|---|---|
| **compute** | **85.6 TFLOP/s** | 12.6 GFLOP in 40.18 ms = 0.31 TFLOP/s | **0.37%** |
| **DRAM bandwidth** | **367 GB/s** | 6.698 GB in 40.18 ms = 167 GB/s | **45%** |

**The p150's headline advantage is compute, and this workload uses 0.37% of it.** Batch-1 decode
is a matrix-*vector* product: every weight byte is read from DRAM, used for one multiply-add, and
discarded — arithmetic intensity ~2 FLOP/byte. The direct evidence is that **rows 1 → 32 cost
identical time** (`w2` 0.186 → 0.184 ms; `wqkv` 0.063 → 0.072). Thirty-two times the arithmetic,
free. The 130 cores and 85.6 TFLOP/s are unusable at batch 1, on either chip.

**So the only axis in play is DRAM, where the advantage is 1.8×, not "a lot"** (367 vs ~200). And
we reached a *smaller share* of it than the N150 did of its own:

| | achieved | its ceiling | utilisation |
|---|---|---|---|
| N150, same graph, ~48 ms | ~140–153 GB/s | ~200 | **70–77%** |
| p150 before §6.52 | 167 GB/s | 367 | **45%** |
| p150 after §6.52 | **181 GB/s** | 367 | **49%** |

Effective ratio 167/140 ≈ **1.19×**, diluted to the ~1.07× seen end-to-end by the host tail and
codec. §6.52 recovered part of the gap — `wo` and `w2` were running at 144–147 GB/s purely
because of a bad default `in0_block_w`.

**Why a skinny matmul cannot fill a wider pipe.** Memory-level parallelism at batch 1 is fixed by
the *weight shape*, not the chip; saturating 367 GB/s needs proportionally more reads in flight
than saturating 200, and a 1-row matmul does not have them. Spreading it over 130 cores instead of
64 gives each core a smaller slice — *less* work to hide DRAM latency behind, and a larger fixed
sync cost per op. This is also why every one of the reversals in §6.39–§6.45 removed cores or ops,
and why 13×10 loses to 12×6 in §6.52.

**THE PRACTICAL CONSEQUENCE.** Single-stream latency is close to done: ~49% DRAM, 0.37% compute,
and the remaining single-stream levers are worth percent, not multiples. **Batching is the only
lever with an order of magnitude behind it** — rows 1→32 being free means ~32 concurrent
utterances at roughly the current per-frame time. Weight reads dominate and are genuinely free at
batch 32; KV-cache traffic and attention scale linearly, so expect **~20–30× aggregate RTF for
~1.1× per-stream latency**, not 32×. Untested — see §7.

### 6.54 — the codes gate's "29.5% of codes differ" is a synthetic-input artefact, and always was

Asked where §6.52's 85/288 comes from, since ~30% disagreement with the reference next to a WER of
1-in-894 cannot both be true. It is not a regression, it is not new, and the number does not mean
what it reads as. Three findings, in the order they mattered:

**1. It is NOT trajectory divergence.** `compare_codes` is **teacher-forced** — both loops advance
on the *reference's* codes — so every frame is an independent "same input, same output?" test and
errors cannot compound. That was the comfortable explanation and it is wrong.

**2. It is the SYNTHETIC INPUT.** `gate_codes` runs on `torch.randn(1,128,3072)*0.02`. The file's
own module docstring says *"ALWAYS GATE ON REAL PROMPTS, never random activations… trap #12, and
the most expensive measurement mistake in this port"* — and `gate_codes` was the one gate
violating it. Same comparison, same commit, only the input changed:

| input | acoustic diffs | \|delta\| histogram | semantic |
|---|---|---|---|
| synthetic `randn*0.02` | 85/288 (**29.5%**) | {1:66, 2:10, 3:6, 4:1, 5:1, 6:1} | 1 of 8 |
| **real prompts, 3 cases** | 34/864 (**3.9%**) | **{1:34} — 100% off by one** | **0 of 24** |
| real prompts, 5 cases | 69/1440 (4.8%) | {1:69} | 1 of 40 |

**On real text every single differing code is off by exactly one FSQ level out of 21** — the
smallest difference representable, meaning the device landed *within one quantisation step* of the
reference. Every \|delta\| > 1 in this model's history comes from synthetic input.

**THE MECHANISM, after three controls — and it is NOT the one this section first gave.** The
original text said random embeddings "sit near FSQ boundaries". That is false, and the controls
that killed it also found the real cause. A code flips when
`|implementation error| > |distance to the .5 boundary|`, so both terms had to be measured:

| control | question | result |
|---|---|---|
| `ref_vs_ref.py` | does the **reference** disagree with itself? | **0/288 on BOTH inputs.** fp32-vs-fp64 error is ~1.3e-6 against a margin of ~0.25 — five orders of headroom. The reference never flips anything. |
| same | are synthetic inputs nearer the boundaries? | **No.** Median margin **0.260 synthetic vs 0.253 real** — indistinguishable. The first explanation dies here. |
| `device_err.py` | is it Block 2 / FSQ? | **No.** With `h` held FIXED, the device's pre-FSQ error is the same on both (median 0.0334 synthetic, 0.0389 real — real slightly *worse*) and flips only 2.4% / 5.2%. |
| Block 1 divergence | is it Block 1? | **Yes.** PCC(h_dev, h_ref) **0.9865 synthetic vs 0.9999 real**, relative error **15.58% vs 0.70% — 22×** — present at step 0 from prefill alone, not accumulating. |

So the chain is: **Block 1's accuracy collapses on off-manifold input** → a 22× larger error in `h`
→ Block 2's velocity error crosses the (unchanged) FSQ margin far more often → 29.5% of codes
flip. This is trap #12 exactly — it recorded Block 1 at PCC 0.892 on random embeddings against
0.9994 on real — reaching the codes gate through Block 2 rather than being visible directly.

The number is therefore **real, not a rounding illusion**: on synthetic input the device genuinely
is far less accurate. What makes it useless as an accuracy figure is that *the input is not
representative of anything the model is ever asked to do*.

**3. IT PREDATES EVERY p150 CHANGE.** Bisected across all 11 fork commits in an isolated worktree:

| commit | acoustic |
|---|---|
| `a3b4569021` fork point — **unmodified N150 code** | **85/288**, 1 semantic |
| `66edfba1db` … `c32b27c220` (all p150 work) | 86/288, 1 semantic |
| `1c91d3de4c` (§6.52) | 85/288, 1 semantic |

The whole range varies by **one code**. Nothing on this fork moved it.

**WHERE THE "10/288" CAME FROM, and why it was confusable.** `[flow-11]` records *"10/288 acoustic
codes vs the fp32 reference and 0/8 semantic **on 8 real prompts**"*. It says real prompts right
there. Both populations have denominator 288 — 8 frames × 36 codes — so a real-prompt number and a
synthetic-gate number are typographically identical and sit in the same file. Today's real-prompt
measurement is 11–12/288 per case, squarely consistent with that record. **The two numbers were
never in conflict; they were never the same measurement.**

**FIXED, so this cannot recur.** `gate_codes` now (a) runs the real prompt fixture as well and
labels that block *"THIS is the accuracy number"*, (b) prints the \|delta\| histogram and
off-by-one fraction after every run, and (c) warns inline that the synthetic block is a
pessimistic proxy. A count alone is not readable; a count plus its magnitude distribution is.

**MY HANDLING OF THIS WAS THE ACTUAL FAULT.** §6.52 dismissed the mismatch as "from a different
state and not comparable" and moved on. That is the shape of an excuse, not a measurement — a
paired run had shown 86 → 85, which established *my change* was innocent and nothing more. The
question of why the level was 86 went unasked for one line of prose.

### 6.55 — prefill is where the synthetic error lives, and there is no lever worth pulling

Follow-up to §6.54: if Block 1's prefill is the source, can it be improved? Priced the only lever
§6.16 found to matter — weight dtype — on both populations at once:

| arm | REAL PCC | REAL rel | SYN PCC | SYN rel | ms/step |
|---|---|---|---|---|---|
| **shipped** (FF bf8, attn bf8) | 0.999894 | **0.70%** | 0.986540 | 15.58% | **17.26** |
| +bf16 FF | 0.999914 | **0.70%** | 0.976994 | 20.47% | 20.11 |
| all bf16 | 0.999934 | **0.70%** | 0.990304 | 18.01% | 22.16 |

**1. THE SYNTHETIC NUMBER IS NON-MONOTONIC IN PRECISION.** FF→bf16 is unambiguously more precise
and makes synthetic PCC **worse** (0.9865 → 0.9770). A metric that degrades when you improve the
implementation cannot rank configurations — off-manifold the model is chaotic, and the number
reports which arbitrary point in the basin you landed on, not an error magnitude. This kills the
idea that the synthetic gate is a useful *sensitivity canary* for regressions real prompts would
hide; that hypothesis was proposed here and rejected by this measurement. **Never rank a config on
the synthetic block.**

**2. REAL-PROMPT PREFILL ERROR IS PINNED AT 0.70% ACROSS THE WHOLE LADDER.** Doubling every
weight's precision buys **+0.00004 PCC for +4.91 ms/step** (~12% slower). So the residual 0.70% is
**not weight quantisation** — it is activation precision and accumulation. There is no weight lever
left, which independently confirms §6.16's knee from the opposite direction: §6.16 showed going
*down* in precision costs accuracy; this shows going *up* buys none.

**So the answer to "what can we do about it" is: nothing, and that is a measured result rather
than a shrug.** Real-prompt prefill sits at PCC 0.999894 last-position, ahead of tt_transformers'
own 0.999564 at P=200 on the same FF-BFP8 config, with long-form WER 1 in 894. The 15.58% only
exists for inputs the model is never asked to process.

### 6.56 — higher-precision PREFILL: the weights do nothing, the activations do, and it still loses

Prefill runs once per utterance, so its cost amortises over ~450 frames and it can afford
precision decode cannot. Asked whether that is worth taking. Measured on a real prompt (case 0,
P=200) against the fp32 CPU reference:

| arm | PCC last | rel err | prefill s | cost /frame @450 |
|---|---|---|---|---|
| **shipped** (weights bf8/bf16, act bf16) | 0.999894 | **0.70%** | 0.05 | — |
| weights fp32, act bf16 | 0.999934 | **0.70%** | 0.07 | +0.03 ms |
| **act fp32**, weights unchanged | 0.999954 | **0.29%** | 0.08 | +0.06 ms |
| both fp32 | 0.999990 | **0.15%** | 0.08 | +0.07 ms |

**fp32 WEIGHTS BUY NOTHING** — the relative error does not move at all. That is §6.55's prediction
confirmed on a much bigger jump than the bf16 it was derived from, and it kills the obvious
version of this idea: a separate fp32 prefill weight copy would cost ~13.6 GB for zero accuracy.

**fp32 ACTIVATIONS DO WORK**, 2.4× on their own and 4.7× combined, at **+0.06 ms/frame** and — the
appealing part — **zero extra memory**, since the weights are untouched. It reaches the codes too,
which independently closes §6.54's causal chain end to end:

| prefill act | real codes vs fp64 | synthetic codes vs fp64 |
|---|---|---|
| bf16 (ships) | 18/288 (6.2%) | 63/288 (21.9%) |
| fp32 | **15/288 (5.2%)** | **49/288 (17.0%)** |

**AND IT STILL LOSES, for reasons that only appear when you try to run DECODE.** The accuracy runs
above call `prefill()` and never `step()`, which hides two hard stops:

- fp32 activations produce fp32 K/V, so the cache becomes fp32 — and
  `scaled_dot_product_attention_decode` rejects it outright: `TT_FATAL: Unsupported data type
  DataType::FLOAT32`. **The measured config cannot execute a single decode step.**
- Forcing the cache back to bf16 instead fails in `fill_cache`: `Input and cache tensors must have
  same dtype!`

It is not impossible — an explicit `ttnn.typecast` of K/V before `fill_cache` satisfies both. But
that rounds the cache to bf16 anyway, so the benefit collapses to **frame 0's `h`**: every later
frame attends over a bf16 prompt cache exactly as it does today. Against that, the price is a
dtype special-case through the prefill path, and a full 3-seed WER re-gate because codes move.

**Verdict: no.** Three codes in 288, at frame 0 only, with WER already at 1 wrong in 894 — there
is nothing downstream to collect it. Recorded because "prefill is cheap, spend precision there" is
a good idea that anyone would have again, and the reason it fails is two op-level dtype
constraints that are invisible until you call `step()`.

**BUILT AND TESTED ANYWAY** (`prefill_f32_act.py`), with the `ttnn.typecast` at the cache boundary
that clears both asserts. It runs — prefill and decode — and it confirms the prediction above
rather than overturning it:

| arm | prefill PCC | rel | PCC(h) step 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|
| bf16 (ships) | 0.999894 | 0.70% | 0.999894 | 0.999799 | **0.999874** | 0.999905 |
| fp32 activations | 0.999954 | **0.29%** | **0.999954** | 0.999812 | 0.999833 | 0.999916 |

**The gain is real at step 0 and gone by step 1** — after that the difference flips sign (fp32 is
*worse* at step 2) at ±0.00004, which is noise. Both arms attend over a bf16 prompt cache, so
there is nothing to carry it forward. Warm cost confirmed at **+0.06 ms/frame** (0.055→0.077 s at
P=200, 0.072→0.101 s at P=357); the 3.62 s seen on a first call is one-off JIT of the fp32 kernels.

**WHAT AN fp32 CACHE WOULD COST, since that is where a precision gain would actually land** — it
feeds *every* frame's attention, unlike prefill's `h`. `sdpa_decode` rejects fp32 so it cannot be
timed, but cache bytes double identically whether you double the dtype or the position, so bf16 at
2P is a faithful bandwidth proxy:

| position | cache MB read/frame | ms/step |
|---|---|---|
| 250 | 26.6 | 17.68 |
| 500 | 53.2 | 18.30 |
| 1000 | 106.5 | 17.58 |
| 2000 | 213.0 | 17.78 |

**Decode time is FLAT while cache traffic grows 8×** — deltas +0.63 / −0.72 / +0.19, random-signed
inside a 0.7 ms spread. From bytes: an fp32 cache costs **0.145 ms/step (0.8%) at pos 500** and
0.58 ms (3.3%) at pos 2000, plus double the resident footprint (109 → 218 MB at 1024). So it would
be nearly free — **and it is simply unsupported.** bf16 is already the best cache dtype
`sdpa_decode` accepts, so there is no precision lever on the cache at all.

**A STANDALONE RESULT FROM THE SAME TABLE: decode cost is O(1) in position up to 2048.** Long
utterances do not degrade RTF, and the KV cache is nowhere near being a bottleneck — 213 MB/frame
at pos 2000 is 0.58 ms of a 17.8 ms step.

### 6.57 — fp32 cache is unreachable, and bf16 decode weights cost 29% for nothing

Two closing tests on the precision question. Both negative, both for reasons worth keeping.

**fp32 KV CACHE VIA A HAND-ROLLED DECODE ATTENTION.** §6.56 showed the cache is the one place a
precision gain would reach every frame, would cost ~0.8% in bandwidth, and is blocked only by
`sdpa_decode` refusing the dtype. So: replace that op with `q@kᵀ → scale → softmax → @v`, which has
no dtype restriction.

| arm | ms/step | vs ships | cache MB |
|---|---|---|---|
| `sdpa_decode` (ships) | 17.83 | — | 109 |
| hand-rolled, bf16 cache | 96.58 | **+78.75 (5.4×)** | 109 |
| hand-rolled, fp32 cache | 797.55 | **+779.7 (44.7×)** | 218 |

**THE PRIOR WAS WRONG BY 9×** and the reason matters more than the result. It was recorded before
running: "5 extra ops × ~68 µs × 26 layers ≈ +8.8 ms". The real cost was +78.75. §6.45's ~68 µs
figure is the floor for a **small** op, and these are not small — the hand-rolled path materialises
`repeat_interleave` to `[1,32,P,128]` for K and V in every layer, which `sdpa_decode` never does
because it handles GQA natively. That expansion is precisely what `[gpt-13]` and `[flow-11]` exist
to avoid, and it is a bandwidth cost, not a launch cost. **Do not apply the op-count model to ops
that materialise tensors.**

So the fp32 cache costs ~1% if the op supported it and 4470% if you route around it. Closed.

**bf16 WEIGHTS THROUGH THE WHOLE DECODE**, teacher-forced on real frames against the fp32
reference:

| decode weights | ms/step | vs ships | min PCC | mean worst-sample |
|---|---|---|---|---|
| BFP8 FF+attn, w2 bf16 (**ships**) | 17.18 | — | 0.999741 | 0.95% |
| bf16 FF, BFP8 attn | 20.17 | +2.99 | 0.999762 | 0.88% |
| bf16 everything | 22.12 | **+4.94** | 0.999814 | **0.96%** |

Worst-sample goes 0.95% → 0.88% → **0.96%** — non-monotonic, i.e. noise. **+29% for no measurable
accuracy.** §6.16 chose this ladder on the N150 for speed; it holds here for accuracy reasons too.

**THE UNIFIED RESULT OF §6.55 / §6.56 / §6.57: Block 1's ~0.9% worst-sample error is not
weight-precision-limited at either prefill or decode.** Raising weight precision buys nothing at
either end, in both cases non-monotonically. The residual is activation precision and
accumulation — and the one place that could be attacked, the KV cache, is closed by op support.
**Treat Block 1's weight precision as settled and stop re-opening it.**

### 6.58 — full ship-readiness pass on HEAD, and the one defect traced to the model

Everything runnable, on `1e778bc297`, model code unchanged since `1c91d3de4c` (verified: no commit
since touches `tt/*.py`), so the 15×3 audio and these gates describe the same build.

| check | result |
|---|---|
| pytest | **129 passed** |
| `--gate wiring` | 1-layer prefill PCC 0.99995965 |
| `--gate prefill26` | PCC last 0.99987–0.99992 — **ahead of tt_transformers' 0.999564 @P=200** |
| `--gate flow` | semantic exact **True**; 2 of 74 codes — matches the record exactly |
| `--gate codec` | waveform PCC **0.999767** (T=8), **0.999920** (T=24) |
| `--gate codes`, real prompts | **34/864 (3.9%), 100% off-by-one, 0 semantic mismatches** |
| `--gate codes`, synthetic | 85/288 — the §6.54 proxy, not an accuracy number |
| `--gate decode`, 15×22 frames | mean **0.91%**, p90 1.31%, max 3.30%, min PCC **0.999390** |
| long-form WER, 15 cases × 3 seeds | **1 wrong of 894 words** |
| `[END_AUDIO]` termination | **90/90** |
| clipping / DC offset | **0.00% clipped**, DC 7.5e-05 mean |
| **determinism** | **bit-identical** on a repeat run, cases 1 and 6 |
| performance | **39.14 ms/frame, RTF 0.507** (long-form, warmup case excluded — §6.52) |

**THE ONE AUDIO DEFECT IN 90 UTTERANCES, AND IT IS NOT OURS.** `artifacts()` counts
discontinuities (|step| > 0.5 at 24 kHz). Across all 90, only **case 6** (`de_male`, *"Grüße aus
München — die Straße ist schön."*, 8 words) ever registers any — and only on seed 1:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| device, baseline | 0 | 48 | 0 |
| device, §6.52 | 1 | 60 | 0 |
| **fp32 CPU reference** | **1** | **69** | — |

**The reference clicks MORE than the device.** It is a property of that prompt on that seed, not a
port defect, and the device is if anything slightly better. `case6_de_male_FP32REF_s1.wav` is kept
next to the device version for a side-by-side.

**WHAT IS STILL NOT ESTABLISHED — and it is the part that decides "ship".**
- **Naturalness has not been evaluated on this build.** §3's listening pass was on `82d04f977a1`,
  before §6.52 moved codes and frame counts. `generated/SAMPLER_p150_HEAD.wav` (143.6 s, 15 clips)
  is built from this build for exactly that, with `case1_*_FP32REF_s0.wav` as a prosody A/B.
- **No MOS-style eval with real raters**, which §3 already said is the bar for any naturalness
  claim to a customer.
- **The WER number comes from a scipy port** of `score_quality_set.py`, not the repo scorer, which
  cannot run here (torchaudio ABI). Faithful, but a systematic bug in it would pass silently.
- **894 words over 4 long-form cases** is a thin sample.
- **Functional limits**: batch=1 only. Utterance length is bounded by `max_seq_len`, which holds
  prompt + generated frames together — 2048 is ~136 s of audio, and raising it costs DRAM only
  (§6.69). Not a prefill limit.
- **CC BY-NC 4.0 weights** — a hard non-technical blocker regardless of quality.

**Verdict: every objective check is at or better than the reference implementation, and the one
measurable defect belongs to the model. What remains is listening, which no metric here
substitutes for.**

### 6.59 — an automated MOS eval, and an MCD that failed calibration and was discarded

§6.58 said naturalness could not be assessed without human raters. That was **too strong**: MOS
proper needs humans, but a no-reference MOS PREDICTOR is the standard automated proxy and had
never been run. DistillMOS (xls-r-sqa distilled, predicts 1–5 from the waveform alone) in an
ISOLATED venv — it requires `torchaudio`, which §2 records as breaking `transformers` in the main
one, so `/tmp/mosvenv` makes that impossible rather than merely unlikely. Verified after: the main
venv still has no `torchaudio`.

**DEVICE vs fp32 REFERENCE on identical prompts — the comparison that matters**, since a
predictor's absolute scale is not calibrated for this domain but a paired delta is:

| pair | device | fp32 reference | delta |
|---|---|---|---|
| case 1, cheerful_female, seed 0 | 4.574 | 4.591 | **−0.017** |
| case 6, de_male, seed 1 (the click case) | 3.831 | 3.858 | **−0.027** |

**The device is perceptually indistinguishable from the fp32 reference.** Both deltas are far
inside predictor noise, and the click case scores the same as the reference that clicks *more*.

**Across all 90 utterances**, and §6.52 did not cost anything perceptually:

| arm | n | mean | median | min | max |
|---|---|---|---|---|---|
| baseline | 45 | 4.007 | 4.077 | 2.507 | 4.755 |
| **§6.52 (HEAD)** | 45 | **4.023** | 4.169 | 2.583 | 4.735 |

**Long-form cases — the WER bucket, and what a user actually hears — mean MOS 4.627.** The 4.02
overall is dragged down by very short adversarial prompts, and the two worst are the SAME in both
arms, so they are the model or the prompt, not the port: **case 8** (`it_male`, *"Ciao! Però…
non è così, vero?"*, 6 words, 37–44% silence) at 2.58–2.70, and **case 14** (`nl_male`, literal
`Tab\tand\nnewline handling.`) at 2.78–3.01. Case 4 is one word (*"Hello."*) and swings 2.84–4.17
across seeds, which is the short-prompt instability §6.7 already documents.

**MCD WAS ATTEMPTED AND DISCARDED — the self-test is the result.** Mel-cepstral distortion is the
standard objective TTS distance, and three implementations all failed calibration:

| attempt | MCD(x, x) | MCD(x, x+1e-4 noise) | MCD(x, different utterance) |
|---|---|---|---|
| librosa MFCC + standard constant | 0 | 24.9 | 117.8 |
| natural-log mel-cepstra | 0 | 24.9 | 115.4 |
| + relative floor, + energy gate | 0 | 14.2–16.8 | 111.7–123.5 |

A correct MCD reads **≪1 dB for imperceptible noise and ~8–15 dB for different utterances**. Every
attempt is ~10× too large, because a DCT of an 80-bin log-mel is not the mel-generalised cepstrum
the constant assumes and there is no SPTK/pysptk here. **No MCD number is reported.** The first
run printed 181 dB — an impossible value that nothing would have flagged without the self-test.
**Calibrate an instrument on known inputs before believing anything it says about unknown ones.**

**What DID survive, sample-aligned (same codes through both codecs, so no DTW and no divergence):**
SNR **42.94 dB**, log-spectral distance **0.620 dB** — both well past the >20 dB / <1 dB
transparency thresholds for speech coding. Block 3 is perceptually transparent.

**F0, with the control that makes it readable.** Device-vs-reference log-F0 correlation is 0.687
(case 1) and 0.667 (case 6). On its own that means nothing — generation is stochastic, so two
legitimate samples differ too. Two DEVICE seeds of case 1 correlate at only **0.202** with 61.9%
voicing agreement, against **0.687 / 74.6%** device-vs-reference. **The device tracks the fp32
reference's prosody far more closely than the model tracks its own seeds.** (Case 6 inverts —
0.856 seed-to-seed vs 0.667 — but it is 2.4 s with few voiced frames; treat it as underpowered.)

### 6.60 — every check behind one command, with paired comparison

The checks had accumulated as a dozen scratch probes, so "did this change hurt quality" depended on
remembering which to run and what its numbers used to be. `scripts/quality_report.py` runs all of
them and writes `generated/quality_<tag>.json`:

| tier | wall clock | what it adds |
|---|---|---|
| `fast` | **3.5 min** | pytest, `--gate flow`, `--gate codes` |
| `full` | **10 min** | + wiring, prefill26, codec, decode |
| `audio` | **16 min** (2 seeds) | + generation, WER, artifacts, MOS |

**It takes TWO TAGS on purpose.** `--compare before after` diffs two runs and exits 1 on any
regression. Nothing here is judged against a number recorded in another session — §6.15 and §6.52
are both cases where that manufactured a regression that did not exist, and the codes gate's
"10/288 vs 86/288" cost a session's worth of doubt for the same reason. Tolerances are the
branch's own measured noise floors.

**Reference `audio` run on `7b480a43e5`** (`quality_baseline.json`, 2 seeds, 30 utterances):

| | | | |
|---|---|---|---|
| pytest 129 / 0 failed | wiring PCC 0.99995965 | prefill PCC last 0.999855 | codec PCC 0.99992 |
| decode mean 0.91% / p90 1.31% | min PCC 0.99939 | flow 2 of 74 codes | codes real 3.9% |
| **WER 1 of 596** | **MOS long-form 4.6298** | MOS mean 4.0153 / min 2.6836 | 30/30 terminated |
| **37.47 ms/frame, RTF 0.490** | 0.00% clipped | 61 clicks (all case 6 seed 1) | |

**THREE BUGS THE HARNESS FOUND IN ITS OWN FIRST RUNS**, each of which had been silently degrading
work all session:

1. **`generate_quality_set.py` had no `--seed`.** Every "3-seed" run through the canonical
   generator would have been the *same* generation three times — a WER number resting on one draw
   while looking like three. Added and threaded through.
2. **It also never recorded `gen_ms_per_frame`**, which is this branch's primary perf number, so
   every perf claim had to come from a scratch harness rather than the canonical script.
3. **`wer_longform` went missing from a report that otherwise looked complete.** The regex used
   `^\s+` for the tag column, but a tag of 11+ characters overflows the 10-char right-aligned
   field and starts at column 0, so it matched nothing. **And the report did not notice**, because
   its completeness check tested for `None` — a key that is never *set* sails straight past that.
   Fixed with a per-tier `EXPECTED` list checked by presence. **A harness that can silently drop a
   metric is worse than no harness**, because it converts an absent measurement into an apparent
   pass.

Also committed `scripts/score_quality_set_scipy.py` — the WER scorer had been living in `/tmp` all
session, the same failure §2 records for the `upstream_compare` venv. Its docstring states plainly
that it is a PORT and therefore a liability: a systematic bug in it passes every utterance
silently, and committing it so it can be diffed against the original is the only defence available.

### 6.61 — `out_subblock_w`'s candidate list had a hole in it

Found while explaining §6.52's helper line, not by any test:

```python
out_subblock_w = next(s for s in (4, 2, 1) if per_core_n % s == 0)   # skips 3
```

`out_subblock_w` is how many output tiles a core accumulates in its destination registers at once.
Two hard rules, both `TT_FATAL` in ttnn:

- `out_subblock_h * out_subblock_w <= 4` — 8 tiles fit normally, but `fp32_dest_acc_en=True` makes
  them 32-bit and halves the count.
- `per_core_N % out_subblock_w == 0` — verified on device: `per_core_N=3` with width 2 or 4 fails
  with *"out_block_w (3) must be divisible by out_subblock_w"*, not a silent fallback.

With `out_subblock_h=1` the legal widths are therefore **1, 2, 3 and 4** — ttnn's own
`SUBBLOCK_HW_CHOICES` lists `{3,1}` explicitly. **The tuple omitted 3**, so `wqkv` (`per_core_N=3`)
fell to `out_subblock_w=1`: three passes through the dest registers where one would do. The
comment said "biggest that fits" and the code did not.

**It is worth nothing in speed: 59.3 µs at width 3 against 59.2 at width 1.** That follows
directly from §6.53 — subblock width is a *compute-side* knob, and at batch 1 the ALUs are ~99.6%
idle, so there is nothing for it to accelerate. Paired `--tier fast` gate: **0 metrics worse**, and
the integer code counts are unchanged (34→34 real, 85→85 synthetic), so the change is inert.

Fixed to `(4, 3, 2, 1)` anyway, for correctness of intent: a future shape with `per_core_N` of 3,
9 or 15 would otherwise silently drop to 1 with nothing to indicate it.
`test_out_subblock_w_is_the_largest_legal_one` now asserts the choice is maximal, so the hole
cannot reopen. 130 tests.

**The general lesson, since this is the second time today:** the §6.52 configs were tuned by
sweeping `in0_block_w` and the grid, and `out_subblock_w` was left to a one-line helper nobody
re-read. A swept parameter gets measured; a derived one gets assumed. **Derived parameters need a
test that re-derives them independently**, which is what the new guard does.

### 6.62 — residual as matmul bias, −1.918 ms/step. And a rejection that EXPIRED

`x + linear(a, W)` becomes `linear(a, W, bias=x)` at both Block 1 residual sites, 26 layers each.

**§6.47 REJECTED THIS AT +0.069 ms/step, AND WAS CORRECT AT THE TIME.** What changed is the
baseline, not the idea:

| | plain linear | `linear(bias=r)` | `add_(r, linear())` |
|---|---|---|---|
| no program config (**what §6.47 measured against**) | 92.7 µs | 93.2 (+0.5) | 95.2 (**+2.5**) |
| with §6.52's config (**today**) | 40.3 µs | 40.4 (+0.1) | 93.8 (**+53.5**) |

`bias=` was **always** genuinely fused — unlike `activation=` (§6.52), it costs +0.1 µs. The add
was simply *hiding inside the matmul's shadow*: against a 92.7 µs matmul it cost 2.5 µs, and
§6.52 made that matmul 2.3× faster and exposed it at 53.5.

**THIS IS A NEW CLASS OF FAILURE FOR THIS BRANCH: a correct measurement whose PREMISE EXPIRED.**
Not a bad harness, not a misread — §6.47 was right when written and silently stopped being right
four sections later. Nothing in the process catches it. **Any rejection whose margin was small
against a then-slow baseline deserves re-testing after that baseline speeds up.** §6.42 was
checked and survives (it cites "the lost free `activation=silu`", which §6.52 showed was never
free — but the conclusion *strengthens*, since the split arm now gets silu at +2.7 µs via the
program config while a fused arm would still need a standalone one at +14.9).

> **CORRECTED (§6.63): THIS DOES NOT SHOW UP END TO END.** The −1.918 ms/step below is real and
> reproducible on the blocks, and a follow-up interleaved run put Block 1 + Block 2 together at
> −2.124 ms. But the generator — since measured repeatable to **0.390 ms** over three identical
> runs — reads **37.47 ms/frame before and 37.54 after**, i.e. no change. The device work genuinely
> got faster; the frame did not. Keep the change (it is free, and device time will matter under
> batching) but **do not credit it with an RTF improvement**. §6.63 has the reason.

**Whole-block A/B, interleaved, 9 rounds, 0.190 ms noise floor:**

| arm | ms/step | vs shipped |
|---|---|---|
| **both residuals as bias** | **16.908** | **−1.918** |
| wo only | 17.470 | −1.356 |
| w2 only | 17.616 | −1.210 |
| shipped `add_` | 18.826 | — |

Isolated predicted 2.8 ms; the block gave 1.9 — §6.52's pattern once more.

**DECODE ONLY, and this is correctness not preference.** A matmul bias is a ROW VECTOR broadcast
across rows, so it equals the residual only at M=1. Prefill has many rows each with its own
residual and would be **silently wrong** — no error, just row 0 smeared over everything. Block 2
has the same hazard (3 or 6 CFG-folded rows) and keeps the explicit add. Gated on `prg`, which is
non-empty only on the decode path, plus `test_residual_rides_in_as_bias_on_the_decode_path_only`.

**GATES.** Paired `--tier audio`, both arms, 2 seeds: **WER 1 of 596 → 1 of 596**, **MOS long-form
4.6298 → 4.6295**, 30/30 terminated, clicks 61 → 53, decode/prefill/codec/flow/wiring all flat.
Accuracy against fixed reference targets over 6 steps: min PCC 0.999771 vs 0.999799, worst
relative error **1.11% vs 1.24%** — unchanged, better on the worst case.

**THE TAIL, WHICH IS WHAT NEARLY BLOCKED IT.** `mos_min` fell 2.68 → 2.25, and 8 seeds could not
say whether that was noise: 0/8 vs 1/8 below MOS 3.0 is unresolvable, and a rare catastrophic
utterance is exactly what a mean hides and a listener notices. So the failure RATE was measured
directly — 3 low-scoring prompts × 24 seeds × 2 arms:

| case | before <3.0 | after <3.0 | median |
|---|---|---|---|
| 4 (one word) | 7/24 | **3/24** | 3.49 → 3.64 |
| 8 (Italian, heavy ellipsis) | 20/24 | 22/24 | 2.75 → 2.76 |
| 11 (emoji) | 0/24 | 1/24 | 4.36 → **4.43** |
| **pooled** | **27/72** | **26/72** | mean 3.494 → 3.537 |

Comparable, with case 4 materially better. The decision rule was pre-registered before the data
existed — *comparable rate → ship, materially worse → revert* — because the author had an interest
in this change passing.

**TWO GATE METRICS DEMOTED TO REPORT-ONLY, on grounds that predate this change.** Both would
otherwise have blocked it, so the justification has to be independent, and it is:
- `codes_synth_n` — §6.59 measured it **non-monotonic in precision** and concluded "never rank a
  config on the synthetic block". Gating on it contradicted that.
- `mos_mean` / `mos_min` — dominated by short and adversarial prompts, which §6.7 already treats as
  seed noise and which the WER scorer already excludes. One draw of a one-word prompt swings
  2.29–4.17 *within a single arm*.
Tail risk is not dropped but measured better, by `tail_probe.py` counting failures over many seeds.
Also fixed: `codes_real_n` and `codes_real_pct` are the same measurement in two units and
disagreed (34→37 read WORSE at zero tolerance while 3.9%→4.3% read "same").

### 6.63 — the block A/B does not predict the frame: 10 host crossings, 2.8 ms of drain

§6.62 saves 2.124 ms of Block 1 + Block 2 time and moves the frame by nothing. Chasing that
apart is the most useful thing in this section, because it undermines the instrument behind most
of §6.39–§6.62.

**FIRST, THE GENERATOR IS TRUSTWORTHY.** Three identical audio-tier runs on unchanged HEAD:

| run | ms/frame | RTF |
|---|---|---|
| 1 | 37.747 | 0.5000 |
| 2 | 37.514 | 0.4948 |
| 3 | 37.356 | 0.4937 |
| **mean** | **37.539** | **0.4961** |

**Spread 0.390 ms.** So "no change" at 37.47 → 37.54 is a measurement, not noise, and the earlier
2.45 ms gap I attributed to session noise was not that either.

**THE BLOCKS ARE SLOWER INSIDE THE REAL LOOP THAN IN A TIGHT LOOP**, same session:

| | tight loop | inside `generate()` | gap |
|---|---|---|---|
| Block 1 | 16.969 | 18.262 | +1.29 |
| Block 2 | 18.541 | 21.027 | +2.49 |
| **both** | **35.515** | **39.289** | **+3.77** |

**WHY: HOST ROUND TRIPS DRAIN THE PIPELINE.** Instrumenting a real generation counts **10.1
crossings per frame** — 3.0 `to_torch`, 7.1 `from_torch` — costing **2.796 ms/frame (5.6%)**. The
loop is structurally `device → sync → host → device → sync → host`: `backbone.step` ends in a
`to_torch`, `semantic_code` does a `linear` then a `to_torch` of 8320 logits to argmax on host, and
`h` is uploaded again for Block 2. **A D2H's cost is not the copy — 8320 floats is 33 KB — it is
every op still in flight having to finish first.**

**THIS IS WHY §6.49 SAW 2.8–3.9% DISPATCH.** That measurement used a tight loop, which by
construction never syncs. It was right about what it measured and silently wrong as a description
of the real loop.

**CONSEQUENCE FOR THE METHOD: a block A/B is a SCREEN, not a verdict.** It measures device time
with dispatch fully overlapped; the real loop has drains that can absorb a device saving whole.
§6.52's −4.24 ms did reach the frame (~44 → 37.5); §6.62's −1.9 did not. **Every timing claim on
this branch made only from a block A/B is now provisional.**

**THE WORK THIS POINTED AT WAS TRIED AND IT DOES NOT WORK. Retracted.** The proposal above was to
keep `h` on device between the blocks and build the CFG pair there, removing three crossings.
Prototyped and measured end to end, arms interleaved, warm-up generations discarded:

| arm | ms/frame |
|---|---|
| shipped | 37.825 |
| `h` + CFG pair on device | **37.744** |
| | **+0.081 — nothing** |

**0.081 ms against a 0.390 ms repeatability.** An intermediate prototype that removed only the H2D
read −0.52 and looked promising; the fuller version shows that was session variation, not the
change.

**SO THE 2.796 ms SPENT INSIDE `to_torch`/`from_torch` IS NOT RECOVERABLE DRAIN — it is work the
device genuinely has to do.** The loop is inherently serial: Block 2 needs `h` from Block 1 and
Block 1 needs the codes from Block 2, so there is no in-flight work for a sync to wait on and
nothing to overlap away. A control confirms it: injecting four *extra* dummy D2H syncs per frame
costs nothing measurable (37.39 → 38.13, non-monotonic), because they land where the device is
already drained.

**AND §6.50 STANDS — my "probably stale" call above was wrong and is retracted.** Moving the
semantic argmax on device would not remove a sync at all: the codes must reach the host regardless
for the `[END_AUDIO]` test and `embed_frame`. It only relocates where the argmax runs, so §6.50's
+310 µs is the whole story. I reached for the §6.47 pattern because it had just paid off, and
applied it where the mechanism does not fit.

**WHAT REMAINS TRUE, AND UNEXPLAINED.** The blocks really are 3.77 ms slower inside the real loop
than in a tight loop, and it is **not** host crossings. The likely candidate is program switching —
a tight loop re-runs one program while the real loop alternates Block 1's ~470 ops with Block 2's
~450 — but that is untested. **The methodological conclusion is unaffected and is the durable part
of this section: a block A/B is a screen, `--tier audio`'s `ms_per_frame` decides.**

### 6.64 — tracing Block 2 is worth 2.29 ms and needs an allocation refactor to collect it

§6.63 left "where do the 3.77 ms go" open. Part of the answer, measured with §6.49's own probe on
current HEAD:

| | eager | traced | dispatch |
|---|---|---|---|
| **Block 2** `_solve` | 17.300 | **15.006** | **13.3% = 2.294 ms** |
| **Block 1** 26 layers | 15.907 | 15.922 | **0.0%** |

**§6.49 recorded 2.8–3.9% and concluded tracing was not worth shipping. Block 2 alone is now
13.3%.** Fourth instance of the same pattern, same cause: §6.52 made Block 2's device work faster,
so dispatch that had been hiding behind slower matmuls became exposed. §6.26 rejected trace-as-
shipping as "three silent failure modes for 0.7%" — at 2.29 ms, 6% of a 37.5 ms frame, that
trade is completely different.

**The asymmetry is the interesting part.** Block 1 is 0.0% dispatch: 26 layers of 3.4B-parameter
work, genuinely device-bound. Block 2 runs nearly as many ops (~600 vs ~470) over 390M parameters
and 3-or-6 rows, so its ops are small and launch cost dominates. **Trace Block 2; leave Block 1
alone.**

**A HOST RUN-AHEAD HYPOTHESIS WAS TESTED AND IS WRONG.** Forcing a `synchronize_device` after
every iteration of the tight loop — which removes the host's ability to enqueue ahead, changing
nothing else — costs **+0.231 ms on Block 1 and +0.099 on Block 2**. The tight loop's advantage is
not run-ahead.

**WHY IT IS NOT A DROP-IN, AND THIS WEDGED THE CARD.** Attempting traced Block 2 inside the real
loop produced:

> `Allocating device buffers is unsafe due to the existence of an active trace. These buffers may
> be corrupted once a trace is executed.`

then hung, and the board had to be recovered with `tt-smi -r` after
`RuntimeError: Read 0xffffffff over PCIe ID 3: the board should be reset`.

**With a trace captured, nothing else may allocate on device.** The real loop allocates every
frame, in both blocks: Block 1 uploads `cos`, `sin`, `pos_t` and the embedding per step;
`semantic_code` uploads `h` and allocates its logits; `decode_frame` uploads `x0` and the CFG pair.
**All of them must become preallocated persistent buffers written with
`copy_host_to_device_tensor` before a trace can be executed at all.** That is a refactor of both
blocks' input handling, not a local change — which is the concrete form of §6.26's "silent failure
modes", and worth knowing before anyone starts.

**So: 2.29 ms is available and the path to it is understood, but it is a day of careful work with a
card-wedging failure mode, and §6.63's rule still applies — the block number is a SCREEN.** It has
to be confirmed on `ms_per_frame`, and `trace_region_size` alone shifts the allocator enough to
move trajectories (`95dc26363f`), so it needs the full WER gate rather than a timing check.

**STILL UNEXPLAINED**: `decode_frame` is 21.0 ms in the real loop while `_solve` alone is 17.3 —
a 3.7 ms wrapper whose uploads are all tiny. Not accounted for by dispatch (measured above), by
host crossings (§6.63 removed them for +0.081), or by run-ahead (tested above).

### 6.65 — the frame loop is TRACED: −4.244 ms/frame, RTF 0.4931 → 0.4514, every quality metric identical

The largest single win on this fork, and it reverses two rejections at once.

| | before | after | |
|---|---|---|---|
| **ms/frame** | 37.456 | **33.212** | **−4.244** |
| **RTF** | 0.4931 | **0.4514** | −0.0417 |
| long-form WER | 1 of 596 | **1 of 596** | identical |
| MOS long-form | 4.6295 | **4.6295** | identical |
| clicks / codes / terminated | 53 / 37 / 30 | **53 / 37 / 30** | identical |

Paired `--tier audio`, both arms, 2 seeds: **0 metrics worse, 21 within tolerance** — and the
quality metrics are not merely inside tolerance, they are **identical to the last digit**, because
the traced graph is bit-exact.

**§6.49 AND §6.26 BOTH REVERSE.** §6.49 measured dispatch at 2.8–3.9% in a tight loop and §6.26
rejected trace-as-shipping as "three silent failure modes for 0.7%". On current HEAD Block 2 alone
is **13.3% (2.294 ms)** — §6.52 made the device work faster and exposed dispatch that had been
hiding behind slower matmuls. Fourth instance of that pattern (§6.47, §6.49, §6.50-considered,
§6.26), and the only one where the stale rejection was worth this much.

**THE DESIGN: ONE TRACE FOR THE WHOLE PER-FRAME GRAPH** — Block 1's 26 layers, the semantic
projection, Block 2's 7 Euler steps. It has to be all of it. Once a trace exists **any later device
allocation may be corrupted**; leaving Block 1 or `semantic_code` eager allocates every frame, and
attempting exactly that hung the board and needed `tt-smi -r` (§6.64). It is possible only because
the acoustic decode does not depend on the semantic argmax — both compute unconditionally and the
`[END_AUDIO]` masking stays on host, where §6.31/§6.50 already put it.

Three details that were not obvious:
- **Frame 0 is eager and runs BEFORE capture.** Its hidden state comes from prefill, not a Block 1
  step, so it does not fit the graph — and running it after capture would allocate.
- **`cos`/`sin` copy in INTERLEAVED and reshard inside the trace.** `copy_host_to_device_tensor`
  into a sharded destination is a layout fight, and the reshard costs no dispatch once traced.
- **The trace is released in a `finally`.** The next `generate()` starts with a prefill, which
  allocates. Capture costs **0.034 s** against ~2.1 s won on a long-form utterance.

**THE BUG THAT REACHED A FULL GATE RUN, AND THE TEST THAT MISSED IT.** `_trace_capture` runs
`graph()` twice — warm-up then capture — and each run **writes K/V through `paged_update_cache` at
whatever `pos` holds**. Left at its initial 0 it destroyed the prefilled prompt's position 0, every
later attention read the wreckage, and the gate came back **WER 1 → 1320 of 596 words, MOS 4.63 →
1.98, 32822 clicks, 11.6% clipping, 6 of 30 utterances never terminating**. Fixed by aiming `pos`
at `pos0`, where the first real frame overwrites it moments later.

**It had already passed a single-frame bit-exact check** — logits and solver state matching to
0.000e+00. **A trace exists to be REPLAYED, so verifying one replay verifies nothing about replay.**
Re-verified over 8 frames teacher-forced: 0 of 288 acoustic differences, every semantic code
matching. `test_trace_capture_aims_the_cache_write_at_the_first_frames_slot` guards it.

**A HYPOTHESIS THAT WAS WRONG, recorded so nobody re-derives it.** The first suspect was
`ttnn.zeros_like` inside the trace not being re-executed on replay, leaving the CFG uncond half
holding stale data. Tested directly over three replays with dirtied memory in between: the uncond
half stays exactly 0.000. Had that been "fixed" instead, the real bug would have shipped.

**One signal shape worth keeping**: in the broken run `ms_per_frame` IMPROVED while `rtf` got
WORSE, because six utterances stopped terminating and ran to the frame cap. **If those two ever
disagree, something is broken rather than fast.**

### 6.66 — review pass after §6.65: four defects, three of them on paths no test reaches

Asked to re-read the code rather than add to it. The comment cleanup was the smaller half; the
review found four real defects in the traced loop, all shipped in §6.65:

1. **A wasted frame per capped utterance.** The loop appended `codes` then computed the next one
   unconditionally, so on the final iteration it did a whole Block 1 + Block 2 (~33 ms) that could
   never be appended. **A regression the traced rewrite introduced** — the pre-trace loop wasted
   only a Block 1 step. Fires only when `max_frames` is hit rather than `[END_AUDIO]`, which is
   30/30 utterances in the current quality set, so it cost nothing measured — but it is exactly
   the kind of thing that hides until a prompt stops terminating.
2. **`traced` was read from a module constant, not the device.** `open_device(trace_region_size=0)`
   followed by `generate()` would still try to capture. It happens to work (ttnn tolerates it), so
   this was fragile rather than broken. Now it attempts the capture and falls back to eager,
   printing why — the decision matches reality instead of describing an assumption.
3. **`_traced_frame` took a `cfg_alpha` it never used.** The value is baked into the trace at
   capture time, so a caller varying it per frame would have been silently ignored. Removed, so the
   signature cannot promise something it does not honour.
4. **A leaked trace id.** If `_trace_capture` died between `end_trace_capture` and its return, the
   id was never registered and never released. `self._tr` is now set the instant the capture closes.

**The eager fallback turns out to be a free correctness check.** Forcing a capture failure, the
fallback path produces codes **identical** to the traced path — an independent confirmation of
§6.65's bit-exactness by a completely different route, and it now runs whenever tracing fails.

**Comment cleanup**: no block of 5+ comment lines remains anywhere in `tt/`, `scripts/` or
`tests/` that this fork wrote. What was left duplicated STATUS or NOTES and is now a pointer.

### 6.67 — the sharded decode norm REVERSES BACK: +5.399 ms/frame, RTF 0.4415 → 0.3647, WER 1 → 0

Asked to look for further optimisation without touching the model. The op map pointed at two
candidates and **the eager map ranked them backwards**:

| op | eager map | measured INSIDE the trace | calls/frame | traced cost |
|---|---|---|---|---|
| `concat` | 144.7 µs — the most expensive per call | **2.6 µs** | 14 | **0.04 ms — a ghost** |
| `rms_norm` | 101 µs | **63.5 µs** — real device time | 102 | **~6.5 ms, ~20% of the frame** |

`concat` was 98% launch cost, which §6.65 had already removed. **An eager map ranks ops by launch
cost, not by what they cost in the shipped path** — §6.63's rule, and it would have sent a day
after the wrong op.

**§6.39/§6.40 REVERSE. Fifth stale rejection of the same kind.** They removed the width-sharded
norm at **+4.4 ms/step WORSE**, and were right eagerly: the cost was the two reshards at ~68 µs of
launch each, ~136 µs against the ~56 µs the norm saved. §6.65 traced that launch cost away.

**WHY THE INTERLEAVED NORM IS SLOW AT DECODE — which §6.39 never identified.** Its factory calls
`split_work_to_cores(..., num_tile_rows, row_wise=true)`: it parallelises over **ROWS**. Decode has
exactly ONE row, so the entire 3072-wide reduction runs on **one core** while the other 129 idle.
More cores cannot help work that has no second axis to split — §6.53's theme exactly. Sharding
splits along WIDTH, the only axis batch 1 has: 32 cores reduce 96 numbers each, exchange 32
partials (the kernel's `cb_ex_partial` → `cb_ex` → `cb_ex_global`, two-stage when the core count is
large), and scale their own shard in place.

**Paired `--tier audio`, both arms, 2 seeds:**

| | before | after | |
|---|---|---|---|
| **ms/frame** | 33.122 | **27.723** | **−5.399** |
| **RTF** | 0.4415 | **0.3647** | −0.0768 |
| **long-form WER** | 1 of 596 | **0 of 596** | BETTER |
| MOS min / mean | 2.2548 / 3.9154 | **2.6597 / 3.9972** | BETTER |
| MOS long-form | 4.6295 | 4.6050 | within tolerance |
| clicks / terminated | 53 / 30 | 52 / 30 | |

**The only real flag is `codes_real_n` 37 → 45 (4.3% → 5.2%).** The sharded kernel reduces in a
different order, so slightly more divergence from the fp32 reference is expected — and it is not
evidence about audio, which WER (1 → 0) and MOS (min +0.40) both say improved. A third flag,
`flow_velocity_pcc` moving 3.58e-06, was a harness defect: it had no tolerance entry and so
defaulted to ZERO. Every PCC now carries one — the same defect `codes_real_n` had in §6.62.

**DECODE ONLY, and prefill says so loudly.** The shard spec fixes the height at one tile, so
prefill's `[1, Sp, 3072]` fails outright: *"Shard height 32 must match physical height 384"*.
`sharded_norm` falls back to interleaved above one tile of rows — correct on the merits too, since
prefill has many rows and the row-wise split is fine there. **Third time this exact constraint has
appeared** (§6.52's configs, §6.62's bias, now this), and the first where the op refused rather
than silently doing the wrong thing.

`test_no_width_sharded_norms_anywhere` was §6.39's guard and correctly BLOCKED this change. It is
replaced by `test_sharded_norm_is_decode_only_and_legally_shaped`, which asserts the prefill
fallback exists and that `cores × block_w == 96`.

**RTF 0.3647 meets the 0.40 target with the model untouched** — `N_DECODING_STEPS` stays 7, so the
port remains a faithful reproduction of the fp32 reference.

### 6.68 — the stale-rejection sweep comes back EMPTY, and that is the useful result

§6.67 was the fifth rejection to reverse because tracing removed the cost its reasoning rested on.
So the three remaining op-count rejections were swept deliberately rather than rediscovered one at
a time. **None of them reverses**, and the measurement says why in one line: traced, these ops are
already nearly free, so restructuring them has nothing left to win.

Traced marginal cost, by injection into the traced graph (one Block 1 layer = 147.0 µs):

| op | eager map | **traced** |
|---|---|---|
| `nlp_create_qkv_heads_decode` | 71.3 µs | **6.2 µs** |
| `paged_update_cache` | 38.2 µs | **8.1 µs** |
| `sdpa_decode` | 66.3 µs | **22.4 µs** |
| reshard (`to_memory_config`) | — | **2.7 µs** |

**§6.44, the fused KV write.** Two `paged_update_cache` calls cost 16.2 µs/layer traced, so fusing
saves at most 0.21 ms/frame — against the **0.687 ms/step it was measured LOSING**. That loss
cannot be launch (fusing means fewer launches), so it is device work, and tracing removes launch
from both arms equally. Stands, and the upside was never large.

**§6.45, the hand-rolled 9-op head split.** Its premise was explicit — *"on this chip OP COUNT is
the dominant term"* — and tracing demolished exactly that premise, so this looked like the
strongest candidate. It is the opposite: the fused op costs **6.2 µs traced**, and nine ops doing
the same work would cost roughly nine times that. §6.45 is **reinforced**, by a wider margin than
when it was made.

**§6.28, the DRAM-sharded matmul.** Its premise genuinely was revived: it wanted to feed the norm's
sharded output straight into the matmul, which died when §6.39 removed the sharded norm and came
back with §6.67. But the unshard it would eliminate costs **2.7 µs**, so 102 of them are
**0.28 ms/frame** — and §6.28 had already measured the DRAM-sharded matmul itself as slower than
the ordinary one. The prize does not cover the known cost.

**⚠ THE STREAK DID NOT END HERE — §6.72 is the sixth, and this section missed it by measuring
Block 1's `nlp_create_qkv_heads_decode` for a decision about Block 2's `nlp_create_qkv_heads`.
The rule below stands; the inventory does not.**

**THE STREAK ENDS HERE, AND THE REASON GENERALISES.** §6.47, §6.49, §6.26, §6.62 and §6.39 all
reversed because a cost their reasoning depended on had been removed. Every one of those costs was
**per-op launch**, and §6.65 removed essentially all of it at once. There is no sixth: the
remaining op-count arguments have nothing left to be wrong about, because the ops now cost 3-22 µs
instead of 38-71. **A rejection is stale when its premise is a cost, and someone has since removed
that cost. Once the cost is gone, the sweep is finished** — which is why this section closes the
line of enquiry rather than opening another.

### 6.69 — the "prefill can't do long utterances" limit does not exist, and never did

Asked why the docs said prefill needs chunking past ~1024 tokens. It doesn't. **Nothing in the
code has ever enforced 1024.** The only guard is `Sp > self.max_seq_len` (`gpt.py:344`), and
`max_seq_len` is a constructor argument defaulting to 2048. The model's own `params.json` has
context **65536** with no sliding window — `voxtral_backbone_ref.py:35` has said "context length
costs KV-cache, nothing else" since the reference was written. The claim appeared in three places
and was never measured.

**Measured (`tests/probes/seq_len_limits.py`).** Prefill at S = 256/512/1024/2048/3072/4096: all
clean, finite, no failure at any length.

**The limit the docs never named, which is the one that matters.** For TTS a "long utterance" is
thousands of generated frames, not a long prompt. Every frame writes one cache position, so
`max_seq_len` bounds **prompt + frames together**. That, not prefill, is what caps an utterance.

| `max_seq_len` | KV cache | audio after a 350-tok prompt | ms/frame |
|---|---|---|---|
| 1024 | 109 MB | 54 s | 27.45 |
| 2048 | 218 MB | 136 s | 27.98 |
| 4096 | 436 MB | 300 s | 27.88 |

**Length is free in time and costs only memory.** Allocation is free (the column above is flat),
and so is depth: a 3635-frame grind out to position 3900 held **28.7 ms/frame**, against 27.5 for a
shallow warm band. Neither `sdpa_decode` nor the trace cares how full the cache is — the trace
takes `pos` as a device tensor, so nothing bakes in a depth. On 32 GB the cache can be sized for
far more than five minutes of speech at no per-frame cost.

**Method note, because it nearly produced a wrong answer.** The first deep run measured 43 ms/frame
in every band and I began writing up a thermal-droop story. It was contamination: a `find /` I had
backgrounded was still scanning the filesystem. `_traced_frame` does real host work every frame
(embed lookup, host↔device copies, argmax, FSQ quantize), so host CPU contention inflates every
frame *uniformly and independently of position* — which is exactly what a hardware explanation
would also look like. Idle the box before timing anything on this fork. This is the third time a
headline number on this branch has been contaminated by something outside the model (§6.21's
warmup case in the mean, §6.63's compare-against-a-recorded-number).

### 6.70 — the port runs on tt-metal main 777 commits newer, unchanged, at the same speed

Our merge-base is `3e153a8842` (2026-07-21); main is `8474cb022b` (2026-08-11). **777 commits** of
drift. Tested in a separate worktree with its own build (`/localdev/.../tt-metal-latest`), so the
validated tree was never at risk.

**Why this was cheap and safe, and worth checking before any future rebase:** our 157 commits touch
**only** `models/experimental/voxtral_tts/`, and main has touched that directory **0 times**. The
histories are disjoint, so there is nothing to conflict. The test applied our directory verbatim on
top of main rather than replaying 157 commits — same delta, less to go wrong.

| | ours (HEAD) | main +777 |
|---|---|---|
| build | — | clean, **18 min** (ccache, 8 cores) |
| source changes needed | — | **none** |
| 132 tests | pass | **pass** |
| long-form ms/frame (cases 2/3/10) | 27.824 | **27.749** |
| long-form WER | 0 of 298 | **0 of 298** |
| `[END_AUDIO]` | 15/15 | **15/15** |
| acoustic codes vs fp32 ref | 45/864 | **40/864** |
| semantic codes | 0 mismatches | **0 mismatches** |
| short-form WER | 4 of 42 | 7 of 42 |

**None of the 67 ttnn symbols `tt/` uses was renamed or removed.** The commits touching our hot-path
ops (`rotary_embedding_hf`, `paged_update_cache`, `nlp_create_qkv_heads_decode`) are the tensor
namespace migration (#50642) and a runtime-args change (#50345) — internal, no signature change.

**Speed is unchanged.** 27.749 vs 27.824 is 0.075 ms against §6.63's 0.390 ms repeatability floor,
and a warm three-config A/B run back to back read 28.27 vs 28.02. Measured on an idle box, both
arms in one session — §6.69's contamination lesson applies with force here.

**The one real difference is the acoustic codes, 45 → 40.** Deterministic, every diff off-by-one,
semantic tokens bit-identical, and in the direction of the fp32 reference. That is a kernel rounding
change, not a structural one.

**The short-form 4 → 7 is NOT a regression and should not be read as one.** The two builds produce
*different audio*: 1754 vs 1771 total frames, and case 10 is 184 frames against 237. Once the code
trajectory diverges the utterances are different samples, so 42 words compares two different
generations rather than the same one degraded. §6.7 already classes that bucket as seed noise.
Resolving it needs ≥3 seeds, which is the gate to run **if we decide to migrate** — not evidence
against migrating.

**Do not quote the scorer's own "mean gen ms/frame" line** (35.50 / 36.63 here). It includes case 0,
which carries kernel compilation and which ONBOARDING §4 says must be excluded. The comparable
numbers are the cases-2/3/10 means above.

### 6.71 — the headline reproduced on a clean tree by a second party, and the sampler finally exists

Run before starting another optimisation, on the principle that a number nobody has reproduced is a
number on trust. Full `--tier audio`, 3 seeds, 45 utterances, 1075 s, on a **clean** tree at
`035983fef2` (the in-flight work was stashed so `_dirty` is false) and an **idle** box per §6.69.
`generated/quality_shipcheck.json`.

| | on record (§6.67) | measured | |
|---|---|---|---|
| **ms/frame** | 27.723 | **27.664** | −0.059, inside §6.63's 0.390 ms floor |
| **long-form RTF** | 0.3647 | **0.3656** | +0.0009 |
| **long-form WER** | 0 of 596 (2 seeds) | **0 of 894 (3 seeds)** | a 50% larger corpus, still zero |
| pytest | 132 | **132, 0 failed** | |
| `[END_AUDIO]` | 30/30 | **45/45** | |
| clipping / clicks | 0.00% / 52 | **0.00% / 52** | all case 6: 1 at seed 0, 51 at seed 1 |
| MOS long-form / mean / min | 4.6050 / 3.9972 / 2.6597 | **4.6101 / 3.9961 / 2.6597** | |

**THE TWO TIMING NUMBERS DIFFER FOR A REASON, and it is worth stating once.** 27.664 ms/frame ÷ 80
is **0.3458** — that is generation alone. The reported **0.3656** is wall/audio, so it also carries
prefill, the codec and the trace capture. Per case, at essentially identical per-frame speed:

    case 1   95/119/92 frames    28.25 / 27.93 / 28.18 ms   rtf 0.3628 / 0.3569 / 0.3622
    case 2  451/475/448          27.52 / 27.44 / 27.42      rtf 0.3905 / 0.3863 / 0.3936
    case 3  470/488/463          27.42 / 27.46 / 27.37      rtf 0.3461 / 0.3466 / 0.3455

ms/frame is flat to 0.88 ms across all nine; RTF swings 0.346–0.394. **Case 2 is the first
utterance to reach the 512-frame codec bucket and pays its kernel compiles (~1.6 s); case 3 reuses
them (~0.13 s).** §6.10 again. This is the concrete demonstration behind ONBOARDING's "quote
ms/frame, not RTF".

**§6.67's one flag reproduces exactly, and the histogram is why it stays benign.** `codes_real_n`
is 45/864 (5.2%) against the pre-§6.67 37 — the number §6.67 predicted and adjudicated. Re-reading
the distribution the JSON does not store:

    case 0   19/288   |delta| histogram {1: 19}   off-by-one 19/19 (100%)   semantic mismatches 0
    case 2   17/288   |delta| {1: 17}             17/17 (100%)              0
    case 3    9/288   |delta| { 1: 9}              9/9  (100%)              0

Every difference is one FSQ level of 21, and **zero semantic mismatches** — the integer that
redirects a whole utterance. The synthetic block read 87/288 (30.2%) with deltas to 6: §6.54's
artefact, unchanged and correctly labelled by the gate itself.

**FIRST DECODE-GATE READING ON THE SHIPPED SHARDED NORM**, recorded because none existed —
`quality_e7` predates §6.67. This session: mean **0.97%**, p90 **1.45%**, min PCC **0.999316**.
It sits within the harness tolerances against both stored runs, and the direction is consistent
with the same different reduce order that moved the codes 37→45. **Do not read the delta against
`quality_e7` as a measurement**: §6.15 is explicit that this gate supports paired same-session
comparison and not absolute levels across sessions. Quote the numbers above as this session's
level, and re-derive rather than compare if it matters.

**Both `--compare` runs against the stored `baseline` and `e7` tags are cross-session** and are
used here to confirm REPRODUCTION, not to adjudicate anything. They landed on the recorded values,
which is the whole point.

**WHAT THIS DOES NOT ESTABLISH, and it is the same sentence §6.58 ended on.** Nothing here is a
listening result. What has changed is that the obstacle is gone: this run's 45 wavs are HEAD's, so
`tests/probes/make_sampler.py` (now tag-parameterised — it was pinned to §6.52's `_prg` arm, which
is how `SAMPLER_p150_HEAD.wav` came to sit beside a HEAD that had moved twice underneath it) built
**`generated/SAMPLER_shipcheck.wav`, 15 clips, 140.0 s, on this build**. Its index now marks the
three deliberately adversarial prompts (10, 11, 14) as such — it used to name them by voice alone,
which invites a listener to read §3.2's known model limitation as a port defect.

### 6.72 — the head split reverses BACK: −0.775 ms/frame, bit-exact. §6.68 counted one op short

**There IS a sixth reversal, and §6.68 said there could not be.** It closed the stale-rejection
sweep with *"the remaining op-count arguments have nothing left to be wrong about, because the ops
now cost 3-22 µs instead of 38-71"*, and reinforced §6.45 on this line:

> §6.45, the hand-rolled 9-op head split. … It is the opposite: the fused op costs **6.2 µs
> traced**, and nine ops doing the same work would cost roughly nine times that.

**That 6.2 µs is `nlp_create_qkv_heads_DECODE` — Block 1's op.** §6.45 is about **Block 2**, which
calls the non-decode `nlp_create_qkv_heads` on a `[2,1,3,6144]` input. `traced_ops.py` only ever
measured the decode variant; the other one was never measured at all. Measured now
(`traced_headsplit.py`, §6.67's injection method, base 240.3 µs for one Block 2 attention half):

| arm | traced µs/split | ms/frame ×21 |
|---|---|---|
| `nlp_create_qkv_heads` (was shipping) | **90.5** | 1.901 |
| **hand-rolled 9 ops, outputs L1 ← SHIPS** | **48.6** | **1.020** |
| hand-rolled 9 ops, DEFAULT memory config | 58.2 | 1.222 |

**90.5, not 6.2 — 14.6× the number the closure rested on.** Nine ops at 48.6 µs beat one at 90.5.

**GATED AND SHIPPED.** Paired `--tier audio`, 3 seeds, 45 utterances, both arms from the SAME tree
in one session (an env switch, since deleted), decision rule fixed before the data existed —
ship only on >0.390 ms (§6.63's repeatability floor) with no quality metric worse:

| | fused | hand-rolled | |
|---|---|---|---|
| **ms/frame** | 27.703 | **26.928** | **−0.775**, 2.0× the bar |
| **RTF** | 0.3675 | **0.3567** | −0.0108 |
| frame counts, all 45 utterances | — | — | **0 differ — BIT-EXACT** |
| WER / MOS / codes / clicks / every PCC | — | — | **identical to the last digit** |

`--compare` reads **0 metrics worse, 21 within tolerance**. Bit-exactness is what makes this free:
45 utterances of ~500 autoregressive steps landing on identical termination frames (§6.32's gate at
full strength) means the codes never move, so WER and MOS *cannot* have.

**THE `memory_config` IS PART OF THE CHANGE.** The hand-rolled arm was first written with its
slices and permutes at the default, which lands q/k/v in DRAM against the fused op's `_L1` — 58.2
against 48.6 µs, and 0.2 ms/frame of the win thrown away. That is §6.31's error in mirror image:
there, default-mc slices were timed against an L1 fused op and read 1.086× faster when they were
not. **Check `t.memory_config().buffer_type` on both arms before believing any head-split number.**
`test_block2_hand_rolls_the_head_split_and_keeps_sdpa` now pins both kwargs.

**THE SCREEN WAS RIGHT FOR ONCE, and the reason is worth keeping.** Traced injection predicted
−0.881 ms/frame; the frame delivered −0.775, i.e. 88% of it. Against §6.62 (−2.124 on the blocks,
**0** on the frame) and §6.47 (a 48× miss), that is the exception — and it matches §6.20's
observation about *which* changes transfer: removing a launch is **layout-neutral**, so nothing
downstream re-optimises around it, while grid and blocking changes alter how every consumer
receives its data. Screens still do not decide; but a screen that only deletes an op is a better
predictor than one that moves data.

**WHAT §6.68 GETS RIGHT, AND WHAT IT DOES NOT.** Its *rule* stands, and is now better evidenced:
a rejection is stale when its premise is a cost someone has since removed. What was wrong is its
*inventory* — it swept three rejections and cleared all three, but checked §6.45 against an op
Block 2 does not call. **The lesson is narrower than "re-open everything" and sharper: when a
section cites a measured cost, check that the measurement is of the op the decision is about.**
§6.44 (fused KV write) and §6.28 (DRAM-sharded matmul) were re-checked here and both still stand.

**New headline: 26.928 ms/frame, RTF 0.3567**, Block 2 ~14.2 ms.

**CONFIRMED ON THE COMMITTED SOURCE, and this is the canonical baseline.** The A/B above ran with
a temporary env switch in the tree; a clean `--tier audio` on `5641f0444b` afterwards reads
**26.992 ms/frame, RTF 0.3606**, WER 0 of 894, 45/45 terminated, 132 tests — +0.064 ms against the
arm, a sixth of §6.63's 0.390 floor. Its `_dirty` flag is set by doc edits only; no `.py` differs
from the commit. **Use `quality_head_5641f04` as the `--compare` before-tag** rather than spending
18 minutes re-deriving it: `shipcheck`, `hs_fused` and `hs_hand` all predate this code or carry the
switch.

### 6.73 — symbol input: the device is systematically SHORTER, and the reference's extra length is repetition

**The first finding a listening pass has ever produced on this branch**, and it took about three
minutes of listening to surface something 45 utterances of metrics did not — because the prompt it
lives on is excluded from WER by construction.

**The observation.** Playing `SAMPLER_shipcheck.wav` against `SAMPLER_FP32REF.wav`, case 10 (*"Numbers
1234567890 and symbols !@#$%^&*() plus   spaces."*) sounds cut off on the device: the reference
articulates "exclamation" where the device produces something clipped.

**IT IS NOT SEED NOISE, which is the part worth having.** Three seeds each, and the ranges are
**disjoint** — the reference's shortest draw is 20 frames longer than the device's longest:

| | seed 0 | seed 1 | seed 2 | range |
|---|---|---|---|---|
| device | 184 | 162 | 199 | **162–199** |
| fp32 reference | 242 | 219 | 252 | **219–252** |

§6.38's control puts seed-only swings at 10–59 frames, so a single pair would have proved nothing;
disjoint ranges over three each is real. **The device is ~56 frames (4.5 s) shorter on this prompt.**

**IT IS NOT A TRUNCATION.** The waveform envelopes rule that out — last-200 ms RMS is 0.22–0.32× the
clip mean on both, and the *reference* ends more abruptly (0 ms of trailing silence against the
device's 16–34 ms). Nothing is being cut; the two say different amounts.

**THE REFERENCE'S EXTRA LENGTH IS A REPETITION LOOP, in 3 of 3 draws.** Whisper (`whisper-base.en`):

    REF s0   '... et twen, pass into twen, plen, twen,'                    loops, never finishes
    REF s1   '... exclamation, eps, and 2 is, exclamation, incur, pest, spaces,'
    REF s2   '... and send full in the leastly K, X, line colon, and send full in send, ... plus max speed.'
    DEV s0   'Numbers, 12,345,67,890, and symbols, x-climes, points, et, nobeludio e, plus and spaces,'
    DEV s1   'numbers 1234 567 890 and symbols exclamation and she look a section and zero sets spaces'

The device reaches the end of the sentence and stops; the reference circles. **So the device being
shorter here is termination, not omission** — and on seed 1 it reaches `spaces` in 162 frames where
the reference needs 219 to reach the same place.

**THE ORIGINAL OBSERVATION SURVIVES, NARROWED.** At seed 0 the reference's `exclamations` really is
cleaner than the device's `x-climes`. At seed 1 the device says `exclamation` clearly and the
*reference* says it twice. Neither implementation renders `!@#$%^&*()` reliably; which one sounds
better is a per-draw coin flip on a prompt the model cannot parse.

**THIS CORRECTS §3.2**, which records *"duration agrees to within 2 frames on the symbol text
(reference 242, device 240)"*. That was one draw each on the N150 build. It no longer holds, and
with three seeds each it could not have been established from the pair it rested on.

**Verdict: model brittleness on non-speech input, reproduced from both sides — not a port defect.**
§3.2 already found the same shape on the emoji fixture, where the *reference* collapses at 6257% WER
while the device stays partly coherent, and attaches the warning *"do not read the device being
better as a quality claim"*. **That warning is symmetric and applies here in the other direction.**
Sanitising or spelling out such text belongs upstream of the model, exactly as §3.2 concluded.

**Two method notes.** Whisper on deliberately-nonsensical audio is an ASR forced to map gibberish
onto words, so the exact strings are indicative only — **the repetition is the robust signal**, and
it does not arise by transcription accident. And the reason a metric never caught this is structural:
`score_quality_set` buckets cases 4/10/11/14 out of WER because a symbol run has no defined
transcript, so **the only instrument that covers those four prompts is a human ear.**

### 6.74 — `_trunk`'s sequence build re-measured: the reshape is a ghost, the concat is BIGGER

§7 has called lines 174+176 *"the largest genuinely untouched item, most likely next win"* at
1.449 ms/frame combined. That rests on §6.36, an **eager map on the N150**, and §6.67 established
that an eager map ranks by launch cost — it put `concat` at 144.7 µs where the traced cost is 2.6.
So both halves were re-measured here, by §6.67's injection method, at their real shapes. **Nothing
was changed; this is a measurement.**

| line | §6.36 eager (N150) | eager, this chip | traced injection | ms/frame ×7 |
|---|---|---|---|---|
| `176` reshape → `[1,B*3,3072]` | 106.0 µs → 0.742 ms | ~6.7 µs | **9.8 µs** | **0.068** |
| `174` `concat([p0,p1,p2], dim=1)` | 101.0 µs → 0.707 ms | 165.6 µs (spread 0.6) | **271.4 µs** | **1.16–1.90** |

**THE RESHAPE IS A GHOST** — 0.068 ms/frame against the 0.742 on record, i.e. ~93% of it was launch
cost that §6.65 removed. Half of §7's lead does not exist.

**THE CONCAT IS REAL AND UNDER-RECORDED.** Both instruments put it well above §6.36's 101 µs, and
the traced figure *exceeding* the eager one is §6.52's rule rather than a contradiction: a tight
loop of identical ops pipelines and understates — the silu op reads 12.2 µs isolated and ~54 in
block. The injection slope is clean (543.0 at +2, 1085.7 at +4, i.e. 271.4 and 271.4).

**So §7's total was roughly right and its composition was wrong: it is essentially all concat**,
which makes this single op the largest non-matmul item in Block 2 — larger than §6.72's win.

**THE OBVIOUS FIX IS ALREADY CLOSED, and re-measuring it here does not reopen it.** A 2-way
`concat([p0, p12])` measures **124.8 µs against 165.6**, a genuine 0.29 ms/frame on paper. §6.30
killed it and the reason still holds: `p2` changes every frame, so building `p12` does not vanish,
it MOVES — seven new concats a frame to save seven cheaper ones, and the op count goes up.

**ONE IDEA NOT YET CONSIDERED, recorded rather than tried.** This attention has no RoPE and no
mask ([flow-01]), so it is permutation-equivariant over the three tokens — the sequence ORDER is
free, and only which slot the velocity is read from matters. Whether that admits a cheaper build is
unknown. **Think before measuring**: §6.34 is the case where a restructuring that kept the op count
the same and moved data to a wider point in the graph lost, and this op is 36 KB of data costing
271 µs, so it is not bandwidth.

### Standing constraints (not fixable by us)
- Weights are **CC BY-NC 4.0**, non-commercial, including the reference voices. Same class of
  blocker as XTTS-v2's CPML. Needs legal sign-off before any product use.
- **Voice cloning from arbitrary audio is impossible** — the codec encoder is not in the release
  (0 of 386 tensors). Only the 20 shipped presets. A test asserts this so a future release that
  adds them fails loudly.

---

## 7. Suggested order when resuming

**Current, §6.72:** **26.9 ms/frame, RTF 0.357** — Block 1 ~15.9 ms, Block 2 ~14.2, host ~2.
Long-form WER **0 of 894**, MOS long-form 4.61, 132 tests. Real time is 80 ms/frame, so this is
~3.0x faster than playback. `N_DECODING_STEPS` is 7 and the port reproduces the fp32 reference.

### Read these four things and you can work

1. **§2** for `PYTHONPATH` (needs `ttnn` AND `tools`, not just the repo root) and `--noconftest`.
2. **`scripts/quality_report.py`** is the whole test story: `--tier fast|full|audio --tag X`, then
   `--compare A B`. It takes TWO tags on purpose — never judge against a recorded number (§6.63).
3. **§5 traps**, plus the three rules that have each cost a day:
   - **A block A/B is a SCREEN, `ms_per_frame` decides** (§6.63). Device time and frame time are
     not the same thing.
   - **An EAGER op map ranks by launch cost**, which is not what ships now the frame is traced. It
     ranked concat (a 2.6 µs ghost) above rms_norm (63.5 µs, 20% of the frame) — §6.67.
   - **Decode is ONE tile of rows; prefill is many.** Program configs (§6.52), residual-as-bias
     (§6.62) and the sharded norm (§6.67) are all decode-only, and two of the three would be
     SILENTLY wrong on prefill rather than erroring.
4. **§6.68** for why the optimisation line stopped where it did.

### What is actually left

**Single-stream is close to done.** 6.698 GB/frame against a measured 367 GB/s ceiling is an
18.25 ms floor; we are at 27.7. Everything found in the last sweep was worth 3–6%, and §6.68
closed the last three candidates.

1. **BATCHING is the only order-of-magnitude lever** (§6.53). Rows 1→32 cost the SAME time — the
   ALUs are 99.6% idle at batch 1 — so ~20–30x aggregate throughput is available for ~1.1x
   per-stream latency. Nothing measured since has changed that, and it has never been attempted.
2. **Prefill's `repeat_interleave`** (`gpt.py:254`) is 11.8% of prefill and the one shipped path
   with a large structural cost a known-better op would remove — `sdpa` handles GQA natively and
   Block 2 already uses it. Prefill is only ~0.5% of an utterance, so this matters only if very
   long prompts do.
3. **Nothing here about prefill length.** This slot used to say "chunked prefill for prompts over
   ~1024 tokens". §6.69 measured that limit out of existence — prefill runs clean to 4096 and the
   model's own context is 65536. Do not re-open it without a measurement that contradicts §6.69.
4. **A human MOS eval.** §6.59's DistillMOS is a predictor, not raters, and §3 has said from the
   start that naturalness has never been properly evaluated.

### Do NOT redo these

- The **stale-rejection sweep is finished** (§6.68). Five reversed because tracing removed the
  per-op launch cost their reasoning rested on; the remaining three were checked and stand.
- **Weight precision** is settled at both ends (§6.55, §6.57): raising it buys nothing,
  non-monotonically, and bf4 costs 5.4x the accuracy for 0.9 ms.
- **Fewer Euler steps** reaches RTF 0.411 but is a MODEL change and was declined — the reference
  uses 7, so taking 5 would end the like-for-like comparison. Measured in §6.66-era notes.
