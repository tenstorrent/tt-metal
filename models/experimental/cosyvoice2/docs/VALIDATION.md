# Validation evidence

What has been measured, how, and what is still open. It grows as measurements arrive: nothing here is an estimate,
and every figure names its run. The bounty's numeric targets (tenstorrent/tt-metal#54104) are declared once in
[`../tests/perf/gates.py`](../tests/perf/gates.py), with the per-architecture verdicts recorded so far.

| target (#54104) | stage | status | enforced in |
|---|---|---|---|
| RTF < 1.0, non-streaming whole-utterance synthesis | Stage 1 | **measured, not met on distinct utterances** (below); verdict not recorded yet | `tests/perf/test_pipeline_perf.py` |
| token-level accuracy > 95 % against the PyTorch reference | Stage 1 | **met: 96.37 %**, teacher-forced over 1,349 positions, with the LLM's fp32-logit head (below); `Meets()` recorded | `tests/e2e/test_token_accuracy.py` |
| WER < 5.0 | Stage 1 | **measured: corpus WER 0.68 % for TT and for the PyTorch reference** (below) | not by a test: `scripts/eval_wer_sim.py` runs in the reference venv |
| speaker similarity > 0.60 | Stage 1 | **measured: 94.90 for TT, 95.21 for the reference** (WavLM-base-plus-sv cosine x 100; below) | same |
| time-to-first-packet < 500 ms; RTF < 0.4 streaming | Stage 3 | streaming not built | — |

## How the figures are produced

- **Board:** Wormhole N150 (one chip), host AMD EPYC 7352 (24 cores, 96 threads).
- **Corpus:** [`../scripts/corpus.py`](../scripts/corpus.py) version 1. LibriSpeech test-clean speakers 260 (M) and
  121 (F): one prompt utterance of about 7 s each, three target transcripts each of about 3, 8 and 15 s. The
  upstream frontend runs once in the reference venv ([`../scripts/prepare_inputs.py`](../scripts/prepare_inputs.py));
  the device side reads its `.npz` files.
- **Model configuration:** `CosyVoice2Config.reported()` ([`../tt/pipeline.py`](../tt/pipeline.py)). The LLM is
  bf16 with tt_transformers' default decoder precision (attention and KV cache bf16, MLP weights bfp8), and since
  2026-09-28 an fp32-logit output head (bf16 weights, fp32 accumulation). The decode trace is on, and sampling is RAS
  on the host, seed 1986. The flow is bf16, 10 Euler steps, eager (the CFM trace
  is off, see the module docstring). HiFT's decoder is fp32 and its F0 predictor and NSF source are fp32.
- **Timing:** stage times are device-synchronized. `wall s` spans the whole `synthesize` call, text normalization
  included. RTF = wall / audio duration, **per utterance, over distinct utterances**: each corpus sentence is
  synthesized once. No figure below is a repeated request unless its row says so.
- **Kernel cache:** tt-metal compiles device kernels on first use and keeps the binaries on disk
  (`~/.cache/tt-metal-cache`). Their compile-time arguments include tensor shapes, so a new sequence length means
  new kernels. Every table states whether the disk cache already held that run's kernels.

## Stage 1 RTF on distinct utterances: first measurement (2026-09-27)

`demo/demo.py --inputs <prepare_inputs dir> --out <dir>` (reported configuration, seed 1986), the first run of these
lengths on this machine. The disk kernel cache held nothing for them. One warm-up sentence ran first and is
reported as the process's cold call.

| utterance | audio s | tokens | LLM prefill s | LLM decode s | tok/s | flow encoder s | CFM s | HiFT s | wall s | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| (warm-up, first call in the process) | 1.20 | 30 | 12.407 | 5.637 | 5.3 | 27.105 | 21.071 | 158.119 | 224.352 | 186.960 |
| zero_shot_121-127105-0003 | 7.04 | 176 | 0.056 | 2.490 | 70.7 | 32.936 | 21.343 | 264.463 | 321.290 | 45.638 |
| zero_shot_121-127105-0015 | 3.36 | 84 | 0.029 | 1.615 | 52.0 | 24.725 | 21.557 | 203.778 | 251.705 | 74.912 |
| zero_shot_121-127105-0024 | 13.04 | 326 | 0.055 | 3.987 | 81.8 | 31.408 | 25.805 | 371.243 | 432.500 | 33.167 |
| zero_shot_260-123286-0014 | 3.04 | 76 | 0.058 | 2.244 | 33.9 | 18.950 | 14.458 | 193.015 | 228.726 | 75.239 |
| zero_shot_260-123440-0002 | 12.76 | 319 | 0.057 | 3.892 | 82.0 | 0.912 | 0.951 | 352.424 | 358.238 | 28.075 |
| zero_shot_260-123440-0010 | 8.08 | 202 | 0.041 | 2.739 | 73.7 | 27.358 | 15.561 | 289.218 | 334.918 | 41.450 |

Aggregate over the six: 47.32 s of audio in 1927.38 s, RTF 40.7; worst 75.2.

**What the table shows.**

- **The LLM is not the problem.** Prefill takes 0.03–0.06 s. Decode runs at 34–82 tokens/s; that includes the
  decode trace's capture, which happens once per `generate()` call, so short utterances show the lower rates.
- **First sight of a sequence length dominates everything else.** Non-streaming lengths are exact, so every distinct
  utterance brings new flow and HiFT geometries. Those cost JIT kernel compilation, the conv resolver's per-geometry
  verification (two conv variants plus a host comparison; a float64 host conv on disagreement), and weight
  preparation.
- **One row isolates the flow.** `zero_shot_260-123440-0002` has a flow length of 175 prompt + 319 generated =
  494 tokens. That is exactly `zero_shot_121-127105-0024`'s 168 + 326, so its flow reused a geometry already seen in
  the process: encoder 0.91 s and CFM 0.95 s, against 31.4 s and 25.8 s at first sight. Its HiFT length differed
  (638 vs 652 mel frames) and still cost 352 s.
- **DRAM grows by 10–19 MiB per bank with each new length** (106 → 182 MiB per bank over the six). That is prepared
  conv weights, cached per geometry and bounded by the free-DRAM eviction threshold. L1_SMALL stays at 0 B per bank
  throughout.

## Device memory and determinism across consecutive utterances

`tests/e2e/test_pipeline_api.py::test_device_consecutive_utterances_of_different_lengths` (passed, 2026-09-27, 18
min): the six corpus utterances (six different lengths) on one device, then the first three again with the same
seeds. No trace was alive after any call. The repeats reproduced their tokens and their audio bit for bit
(max |diff| 0).

| call | case | tokens | audio s | LLM s | flow encoder s | CFM s | HiFT s | wall s | RTF | DRAM MiB/bank | L1 KiB/bank | L1_SMALL B/bank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | zero_shot_121-127105-0003 | 176 | 7.04 | 2.72 | 1.36 | 2.12 | 8.07 | 14.28 | 2.028 | 98.12 | 1.1 | 0 |
| 1 | zero_shot_121-127105-0015 | 84 | 3.36 | 0.93 | 0.96 | 1.87 | 5.91 | 9.67 | 2.878 | 108.29 | 1.1 | 0 |
| 2 | zero_shot_121-127105-0024 | 326 | 13.04 | 3.24 | 1.61 | 2.51 | 268.76 | 276.12 | 21.174 | 130.06 | 1.1 | 0 |
| 3 | zero_shot_260-123286-0014 | 76 | 3.04 | 0.85 | 10.68 | 12.13 | 155.71 | 179.38 | 59.006 | 139.81 | 1.1 | 0 |
| 4 | zero_shot_260-123440-0002 | 319 | 12.76 | 3.21 | 0.75 | 0.92 | 313.00 | 317.89 | 24.913 | 158.64 | 1.1 | 0 |
| 5 | zero_shot_260-123440-0010 | 202 | 8.08 | 2.09 | 11.76 | 13.08 | 248.29 | 275.22 | 34.061 | 174.40 | 1.1 | 0 |
| 6 | zero_shot_121-127105-0003 (repeat) | 176 | 7.04 | 1.81 | 0.26 | 0.68 | 0.19 | 2.95 | 0.419 | 174.40 | 1.1 | 0 |
| 7 | zero_shot_121-127105-0015 (repeat) | 84 | 3.36 | 0.92 | 0.16 | 0.68 | 0.11 | 1.88 | 0.559 | 174.40 | 1.1 | 0 |
| 8 | zero_shot_121-127105-0024 (repeat) | 326 | 13.04 | 3.24 | 0.61 | 0.93 | 0.34 | 5.13 | 0.393 | 174.40 | 1.1 | 0 |

- **Memory does not grow the way CosyVoice1's did.**
  - L1_SMALL stays at 0 B per bank for all nine calls, because conv config tensors live in DRAM.
  - DRAM grows by 10–22 MiB per bank with each new length: prepared conv weights, cached per geometry. It stays flat
    once lengths repeat. The cache evicts when free DRAM per bank falls below its threshold, so a long-running
    process that meets many lengths is bounded, not leaking.
- **Three regimes, one pipeline.** Calls 0 and 1 found their kernels already on disk, compiled by an earlier
  pytest process that ran the same lengths: RTF 2.0–2.9. Calls 2–5 compiled kernels: RTF 21–59. Calls 6–8 repeated
  lengths already run in the process: RTF 0.39–0.56. The steady-state HiFT takes 0.11–0.34 s; its first-sight cost
  is all compilation and per-geometry verification.
- **The disk kernel cache is only partly shared across processes.** The demo had compiled these same lengths
  (same seeds, same tokens), yet the pytest processes recompiled HiFT for every one of them. The flow was reused in
  one case: call 2 (494 tokens) found the demo's flow kernels, while calls 3 and 5 recompiled theirs. Call 4 has the
  same flow length as call 2, so its warm flow is an in-process reuse. Two pytest processes running the same
  sequence of calls did reuse each other's kernels.
- **Why (identified 2026-09-27):** the conv reader kernels and the halo (`untilize_with_halo`) reader kernels take
  their config tensors' DRAM addresses as compile-time arguments when `config_tensors_in_dram=True`, which this
  port uses for every conv. So a binary on disk is reused only by a process whose config tensors land at the same
  DRAM addresses:
  - `conv2d_op_sharded_program_factory.cpp:871`, `conv2d_op_width_sharded_program_factory.cpp:562`;
  - `untilize_with_halo_program_factory.cpp:307-316`.

  Three fresh processes, each running the pipeline and then the same two-sentence warm-up (N150):

  | process | first call s | second call s | whole process s | kernel binaries compiled |
  |---|---|---|---|---|
  | 1: first run of these lengths | 270.1 | 274.1 | 566.5 | 2,373 |
  | 2: the identical sequence | 12.6 | 11.3 | 46.1 | **0** |
  | 3: identical, but 1 MiB allocated first | 220.4 | 222.6 | 464.6 | 1,132, all in `halo_gather` and the two conv reader kernels |

  A standalone `ttnn.conv1d` reproduces it. With the config in DRAM, a 1 MiB shift recompiles the conv reader and
  `halo_gather`. With the config in L1, the same shift compiles nothing.

## HiFT F0 predictor and NSF source: fp32 (default) vs bf16

The pipeline defaults the F0 predictor and NSF source to fp32 because bf16 roughly triples the F0 error (see
`CosyVoice2Config`). Does that cost time? Measured in one process on one pipeline (2026-09-27). A second
`TtHiFTGenerator` with a bf16 F0/source shares the same fp32 decoder. Each case's mel is generated once (LLM and
flow, seed 1986), then both generators run on that mel alternately, four times each. The warm figure is the median of
calls 2–4.

| case | audio s | HiFT first sight fp32 / bf16 s | HiFT warm fp32 / bf16 s | warm difference s | RTF difference |
|---|---|---|---|---|---|
| zero_shot_121-127105-0003 | 7.04 | 221.60 / 13.11 | 0.182 / 0.180 | +0.002 | +0.0003 |
| zero_shot_121-127105-0015 | 3.36 | 162.59 / 13.09 | 0.106 / 0.110 | -0.004 | -0.0011 |
| zero_shot_121-127105-0024 | 13.04 | 330.63 / 18.02 | 0.323 / 0.323 | -0.000 | -0.0000 |

- **Warm, the dtype makes no measurable difference.** The differences are at most 4 ms per utterance and change
  sign, so they are noise.
- **The first-sight figures are not comparable between the dtypes.** The fp32 call ran first and paid for compiling
  the shared decoder; the bf16 call then found it compiled. What bf16's 13–18 s does show is the first-sight cost of
  the F0 predictor and source path alone. This process also recompiled HiFT for lengths the demo had compiled (see
  above).

## Token accuracy, teacher-forced (2026-09-27, 2026-09-28)

`tests/e2e/test_token_accuracy.py`, CosyVoice1's method:
- For each corpus case (the six LibriSpeech targets plus the parity sentence), the PyTorch reference's own generated
  speech tokens are forced through TT's decode loop: prefill, then the traced decode, one step per token.
- At each of the N + 1 positions, TT's top-1 is compared with the reference's
  (`scripts/token_accuracy_reference.py` records the reference's top-5 along the same sequences).

**The lever is the output head's logits, not the decoder.** bf16 logits, about 8 in magnitude, resolve steps of
0.03–0.06. The reference's top two tokens are often closer than that: with bf16 logits, the 120 disagreements sat
at a median reference margin of 0.047 nats. With bf16 weights but fp32 accumulation and fp32 logits, the head gets
96.37 % (`CosyVoice2Config.llm_head_logits_dtype`, the pipeline's since 2026-09-28).

| LLM head (the decoder layers are tt_transformers' default: attention bf16 at HiFi4, MLP bfp8 at HiFi2) | top-1 agreement | disagreements: reference margin, median / max | seconds per token |
|---|---|---|---|
| **fp32 logits: bf16 weights, HiFi4, fp32 accumulation (the pipeline's)** | **96.37 %** | 49: 0.023 / 0.104 | 0.0110 |
| fp32 weights, HiFi4, fp32 accumulation, fp32 logits | 96.37 % | — | 0.0110 |
| the same at HiFi3 | 96.22 % | — | 0.0110 |
| bf16 logits (the pipeline's before 2026-09-28) | 90.66 % | — | 0.0107 |
| bf16 logits, with every decoder weight bf16 at HiFi4 (09-27, max_seq_len 2,304) | 90.07 % | — | not comparable |

- TT's top-1 is inside the reference's top-5 at every position, with either head (the test's own runs).
- Seconds per token: `teacher_forced_topk` over the seven cases, prefill included, warm kernel cache, 2026-09-28.
  The fp32 head costs about 0.3 ms per decode step.
- **Near-ties make the figure sensitive to anything that reorders rounding.** The same bf16-logit head measured
  91.10 % on 09-27 and 90.66 % after the LLM's `max_seq_len` went from 2,304 to 2,048 (a different KV-cache
  size, so a different attention chunking).

**The noise floor: the PyTorch reference against itself, bf16 vs fp32.** `token_accuracy_reference.py
--precision ... --against <the fp32 run>`, on the same forced sequences:

| the reference, run in | top-1 agreement with its fp32 run | disagreements: fp32 margin, median / max |
|---|---|---|
| bf16 throughout | 95.70 % | 58: 0.023 / 0.100 |
| bf16, fp32 output head | 98.37 % | 22: 0.008 / 0.033 |

- **So > 95 % is reachable in bf16, with almost no margin, and the head's precision is most of it.** An exact bf16
  implementation of this model sits at 95.7 %.
- TT with the fp32 head (96.37 %) disagrees about as often, and at the same margins, as a pure-bf16 PyTorch run.
  The 2 points between it and the bf16 + fp32-head reference (98.37 %) are the TT decoder's own rounding, not
  broken down further.

## Bucketing (2026-09-27)

Non-streaming geometries are bucketed and warmed at start-up (`tt/pipeline.py`, module docstring), because each new
exact length cost minutes of first-sight work.

**The bucket sets.** Upstream splits text into segments of at most 80 tokens and allows 20 speech tokens per text
token, so the flow sees at most 750 prompt + 1,600 generated tokens, and HiFT 3,200 mel frames.
- **HiFT can't take the top of that range in one pass.** At 3,200 frames, `ttnn.concat` inside HiFT needs a
  1,536,032-byte circular-buffer page against 1,393,440 bytes of L1 per core (`TT_FATAL`), which puts the
  single-pass limit near 2,900 frames (58 s).
- **Nor can it take 2,560 frames alongside the warmed set.** With every smaller bucket warmed and resident, the
  2,560-frame bucket failed to allocate: it needed 118 MB of contiguous DRAM per bank, and the largest free block was
  106 MB (2026-09-27, the first cold warm-up). 2,048 frames had run in the same process.
- **So a segment's speech is capped at 1,024 tokens** (2,048 frames, 41 s): `CosyVoice2Config.max_segment_speech_tokens`.
  It binds only when a segment of more than 51 text tokens has not ended after 41 s.
- **Past the cap, the pipeline raises `SegmentTooLong`, naming the segment's length**, rather than truncating the
  speech or failing inside a device op (`test_pipeline_api.py::test_segment_past_the_cap_raises`). The LLM runs one
  step past the cap, which tells a segment of exactly 1,024 tokens from a longer one. `tokens_to_mel` and
  `mel_to_wav` refuse more than the cap before any device work.
- **The resulting sets** step by one unit up to 8 units, then the step doubles every 8 steps (`tiered_buckets`):

  | stage | buckets |
  |---|---|
  | flow tokens | 15 buckets, 64 … 1792, strictly above prompt + generated |
  | HiFT mel frames | 12 buckets, 128 … 2048, at or above the mel length |
  | LLM prefill | 8 lengths, 128 … 1024; it already padded to multiples of 128 |

  The LLM's prefix embedding lookups pad to 128 as well.
- **Probe, first sight on a cold kernel cache:** flow at 1,024 and 2,350 tokens ran in 73 s and 101 s; HiFT at
  2,048 frames in 447 s.

**Flow: bucketed vs exact length, with the padding masked** (`test_flow_checkpoint.py::test_device_nonstreaming_bucketed_flow_matches_exact_real_checkpoint`):
- The encoder zeroes the padded rows after its embed, as upstream zero-pads before the look-ahead conv, and adds a
  key-padding bias in both attention stacks.
- The CFM runs its masked attention path with an all-zero chunk term, which is upstream's non-streaming mask.
- The naive control runs the same padded tokens with no masks.

| input | valid → bucket | bucketed vs exact: max \|diff\|, PCC | naive control: max \|diff\|, PCC |
|---|---|---|---|
| real corpus prompt (175 tokens) | 251 → 256 | 0.000, 1.000000 | 2.89, 0.9927 |
| real corpus prompt | 251 → 320 | 0.150, 0.999846 | 2.36, 0.9830 |
| real corpus prompt | 320 → 384 | 0.177, 0.999813 | 3.58, 0.9825 |
| the test's synthetic inputs | 156 → 192 | 0.254, 0.999696 | 3.78, 0.9547 |
| the test's synthetic inputs | 310 → 384 | 0.229, 0.999665 | 3.95, 0.9548 |

The test gates on max |diff| ≤ 0.4 and PCC ≥ 0.9995. The control fails it by a factor of 6 or more.

**HiFT: padding with silence, the tail it touches.** The mel is padded with `log(1e-5)`, the mel's own silence floor
(the quietest frames of real prompts sit exactly there), and the audio is trimmed back. The measurement injects
torch F0 (D16) and uses the same sine noise over the valid region. It separates two effects:
- **leakage:** silence padding vs zero padding at the same bucket, so the kernels are identical and any difference is
  the pad content reaching back into the valid audio;
- **geometry:** bucketed vs exact length. A different length runs different kernel blockings, so this adds rounding
  differences everywhere, which the NSF source's running phase accumulates.

The scale column is TT against the fp32 torch reference.

| mel (real speech) | valid → bucket | leakage: max \|diff\|, reach from the end | bucketed vs exact: PCC | TT vs torch: PCC | log-mel L1, bucketed vs exact: whole / last 160 ms | log-mel L1, TT vs torch: whole / last 160 ms |
|---|---|---|---|---|---|---|
| 121-127105-0003 prompt | 150 → 256 | 0.197, 294 ms | 0.9908 | 0.99943 | 0.027 / 0.218 | 0.030 / 0.034 |
| 121-127105-0003 prompt | 300 → 384 | 0.190, 379 ms | 0.9954 | 0.99929 | 0.023 / 0.269 | 0.038 / 0.038 |
| 260-123286-0014 prompt | 150 → 256 | 0.287, 219 ms | 0.9836 | 0.99960 | 0.031 / 0.198 | 0.036 / 0.036 |
| 260-123286-0014 prompt | 300 → 384 | 0.072, 238 ms | 0.9994 | 0.99960 | 0.023 / 0.247 | 0.039 / 0.047 |

- **The padding reaches back 0.22–0.38 s.**
- **Over the whole utterance**, bucketing's spectral error (0.023–0.031) is no larger than the port's own error
  against torch (0.030–0.039).
- **In the last 160 ms it is 0.20–0.27, about 6× the port's own.** These mels are cut mid-speech, the worst case.
  A generated utterance usually ends in trailing silence. Whether the tail matters is checked end to end by the
  Stage 1 WER/SIM re-score.

## Start-up: warming the bucket set (2026-09-28)

`warmup_buckets()` in two fresh processes, one after the other. `TT_METAL_CACHE` pointed at a new directory, so the
first started from an empty kernel cache. The second is identical: same code, configuration and device parameters.
The measurement script (notes branch, `scripts/2026-09-28/warmup_measure.py`) times each geometry, the conv safety
checks (`_verify_and_resolve`) and the weight preparation (`_prepared`), and counts kernel binaries on disk.

| process | kernel binaries compiled | warm-up s | LLM s | flow s | HiFT s | conv safety checks s | conv weight prep s |
|---|---|---|---|---|---|---|---|
| first, empty kernel cache | 19,068 | 4,561 | 146 | 945 | 3,469 | 1,628 | 25 |
| second, identical | **0** | **577** | 2.5 | 96 | 443 | 182 | 23 |

Pipeline construction took 17 s and 13 s before that. Per bucket, first process / second: LLM prefill 12–23 s /
0.2–0.3 s; flow 47–110 s / 2–20 s; HiFT 173–374 s / 7–151 s.

- **Kernel compilation is the cold cost.** The disk cache saves 3,984 of the 4,561 s (87 %). The first process's
  check time includes compiling the reference convs' kernels.
- **With every kernel on disk, start-up is 9.6 minutes, and nothing compiles.** The second process allocated
  exactly as the first: the same DRAM figures, to 0.1 MiB, after every geometry. So every conv kernel that carries
  a DRAM address in its compile-time arguments matched its binary on disk (see "Device memory and determinism"
  above).
- **The conv safety checks rerun in every process: 182 s, 32 % of the warm start-up.** They are not dead weight.
  Both processes found the same 43 disagreements, geometry for geometry:
  - **20 were real corruption of the fast path** (the prepared weight; tenstorrent/tt-metal#55545's class).
    Relative error against a float64 host conv:

    | conv | where | prepared weight | raw weight, used instead |
    |---|---|---|---|
    | `Conv1d(128->128, k=11)`, 6 resblock convs | the 640-frame bucket (length 5,120) | 1.0–2.6 | 0.003–0.005 |
    | `Conv1d(18->256, k=30, s=15)`, the first source downsampling | every bucket from 640 to 2,048 frames | 7.7 | 0.0018 |
    | `Conv1d(18->128, k=6, s=3)`, the second | every bucket from 896 to 2,048 frames | 0.14–0.19 | 0.0020 |

  - **23 were the reference's own error.** The raw-weight / safe-config reference was off by 0.05–0.10 while the
    fast path was at 0.004–0.005, and the float64 host conv kept the fast path. These arbitrations are most of the
    check time at long lengths.
- **HiFT is 77 % of the warm start-up.** Its two largest buckets (1,792 and 2,048 frames) take 254 s of it.

**Eviction cannot fire.** The warmed set holds 416 MiB of DRAM per bank (82 MiB after construction), and 601 MiB per
bank stays free. Across the warm-up's 3,060 conv-cache inserts, free DRAM never dropped below 559 MiB per bank.
The old eviction threshold was 150 MB free, so it would never have fired either. In bucketed mode the threshold is
0, so eviction is off (`test_geometry_cache.py::test_threshold_override_zero_never_evicts`).

## Speech quality: WER and speaker similarity (2026-09-27)

`scripts/eval_wer_sim.py`, run in the reference venv, scored the demo's TT run from the table above and the PyTorch
reference run (`scripts/run_reference.py`, after the fixes in `0d687d840e`). Both were scored by the same command.
- **ASR:** Whisper large-v3, CPU, greedy, English.
- **WER:** NFKC, lowercase, punctuation stripped, word-level edit distance against the LibriSpeech transcript.
  Corpus WER is total errors over total words.
- **Speaker similarity:** `microsoft/wavlm-base-plus-sv` x-vector cosine x 100, between the output and the case's
  prompt utterance (16 kHz). This is not the paper's SV model, so the comparison is TT against the reference,
  never against the paper's figure.

| case | words | WER % reference | WER % TT | SIM reference | SIM TT | audio s reference / TT |
|---|---|---|---|---|---|---|
| zero_shot_260-123286-0014 | 7 | 14.29 | 14.29 | 95.82 | 94.41 | 3.12 / 3.04 |
| zero_shot_260-123440-0010 | 20 | 0.00 | 0.00 | 96.85 | 97.26 | 8.32 / 8.08 |
| zero_shot_260-123440-0002 | 44 | 0.00 | 0.00 | 97.74 | 97.89 | 12.04 / 12.76 |
| zero_shot_121-127105-0015 | 10 | 0.00 | 0.00 | 92.61 | 91.42 | 3.68 / 3.36 |
| zero_shot_121-127105-0003 | 18 | 0.00 | 0.00 | 94.32 | 94.79 | 7.64 / 7.04 |
| zero_shot_121-127105-0024 | 48 | 0.00 | 0.00 | 93.94 | 93.64 | 12.84 / 13.04 |
| **corpus (6 utterances)** | 147 | **0.68** (1 error) | **0.68** (1 error) | **95.21** | **94.90** | |

- **TT and the reference are indistinguishable at this size.** The one error is the same substitution in both runs:
  "Truly this sea" transcribed as "Truly, the sea". Speaker similarity differs by 0.31 on average, and in both
  directions per utterance.
- **The corpus is small:** 6 utterances, 147 words. One error is 0.68 %.
- **The CAM++ cosine is recorded in `scores.json` as a diagnostic only.** It is self-referential, because the model
  conditions on it. Its means are 76.19 (reference, seven utterances) and 84.13 (TT, six).
- **The CosyVoice1-parity sentence** was run by the reference only: WER 0 %, SIM 94.86.
- **The reference venv's transformers doesn't change these numbers.** The venv runs transformers 5.12.1 with two
  exact shims, not upstream's 4.51.3 (`docs/security.md`). Under 4.51.3 with no shims, upstream reproduced this
  reference run's tokens and audio bit for bit on all seven cases. Re-scoring both runs under 4.51.3 gave identical
  transcripts and scores.

## The PyTorch reference, for scale

`scripts/run_reference.py` (upstream CosyVoice2 at 074ca6dc9e80, CPU fp32, torch 2.11.0+cpu, same host, seed
1986): RTF 5.9–7.3 per utterance over the same six sentences plus the parity sentence. It generates different
tokens from the TT port (its logits differ), so its audio lengths differ too: for example, 191 vs 176 tokens for
`121-127105-0003`. These figures are context, not a target.

## Open

- **Token accuracy** is not measured yet: teacher-forced, over full sequences with the speech prompt.
- **Stage 1 RTF on distinct utterances is far from 1.0** while every utterance brings new geometries. At steady state
  the same pipeline runs at 0.39–0.56. Its verdict in `gates.py` is not recorded until the measurement protocol
  (kernel cache state, geometry bucketing and pre-warming) is settled.
- **`tests/perf/test_pipeline_perf.py` has not been run.** No verdict is recorded yet, so it would stop at that
  assertion; `tests/perf/test_gates.py` covers the enforcement logic on the host.
