# Validation evidence

What has been measured, how, and what is still open. It grows as measurements arrive: nothing here is an estimate,
and every figure names its run. The bounty's numeric targets (tenstorrent/tt-metal#54104) are declared once in
[`../tests/perf/gates.py`](../tests/perf/gates.py), with the per-architecture verdicts recorded so far.

| target (#54104) | stage | status | enforced in |
|---|---|---|---|
| RTF < 1.0, non-streaming whole-utterance synthesis | Stage 1 | **measured, not met on distinct utterances** (below); verdict not recorded yet | `tests/perf/test_pipeline_perf.py` |
| token-level accuracy > 95 % against the PyTorch reference | Stage 1 | not measured yet (teacher-forced) | — |
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
  bf16 with tt_transformers' default decoder precision (attention and KV cache bf16, MLP weights bfp8), the decode
  trace is on, and sampling is RAS on the host, seed 1986. The flow is bf16, 10 Euler steps, eager (the CFM trace
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
  sequence of calls did reuse each other's kernels. The mechanism is not identified.

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
- **The transformers pin changed nothing measurable.** Pinning the reference venv to upstream's transformers 4.51.3
  removed its two behaviour shims. Upstream then reproduced this reference run's tokens and audio bit for bit on
  all seven cases. Re-scoring both runs with the pinned scorer gave identical transcripts and scores.

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
