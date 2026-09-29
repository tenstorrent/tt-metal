# Validation evidence

What has been measured, how, and what is still open. It grows as measurements arrive: nothing here is an estimate,
and every figure names its run. The bounty's numeric targets (tenstorrent/tt-metal#54104) are declared once in
[`../tests/perf/gates.py`](../tests/perf/gates.py), with the per-architecture verdicts recorded so far.

| target (#54104) | stage | status | enforced in |
|---|---|---|---|
| RTF < 1.0, non-streaming whole-utterance synthesis | Stage 1 | **met: worst 0.628, aggregate 0.479** over six distinct utterances after the bucket warm-up (3.2 min at a warm start; below); `Meets()` recorded | `tests/perf/test_pipeline_perf.py` |
| token-level accuracy > 95 % against the PyTorch reference | Stage 1 | **met: 95.94 %** teacher-forced over 5,003 positions (27 sequences, 4 speakers), with the LLM's fp32-logit head (below); `Meets()` recorded | `tests/e2e/test_token_accuracy.py` |
| WER < 5.0 | Stage 1 | **met: corpus WER 0.68 %** on the Stage 1 audio (chunked HiFT), the same as the PyTorch reference (below); `Meets()` recorded | not by a test: `scripts/eval_wer_sim.py` runs in the reference venv |
| speaker similarity > 0.60 | Stage 1 | **met: 95.87** on the Stage 1 audio (chunked HiFT), reference 95.21 (WavLM-base-plus-sv cosine x 100; below); `Meets()` recorded | same |
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
  is off, see the module docstring). HiFT's decoder is fp32 and its F0 predictor and NSF source are fp32; since
  2026-09-28 a mel of 512 frames or more runs through HiFT in 512-frame chunks ("Chunked HiFT" below).
- **Timing:** stage times are device-synchronized. `wall s` spans the whole `synthesize` call, text normalization
  included. RTF = wall / audio duration, **per utterance, over distinct utterances**: each corpus sentence is
  synthesized once. No figure below is a repeated request unless its row says so.
- **Kernel cache:** tt-metal compiles device kernels on first use and keeps the binaries on disk
  (`~/.cache/tt-metal-cache`). Their compile-time arguments include tensor shapes, so a new sequence length means
  new kernels. Every table states whether the disk cache already held that run's kernels.

## Stage 1 on chunked HiFT (2026-09-28)

The same protocol, re-run after chunked HiFT (below) and the cap lift, on a new, empty kernel cache.

**Start-up.** `warmup_buckets()` in a fresh process on the empty cache, then in an identical second one:

| | first process (empty kernel cache) | second, identical | before chunked HiFT: first / second |
|---|---|---|---|
| warm-up | **1,831 s (30.5 min)** | **194.6 s (3.2 min)** | 4,561 s / 577 s |
| kernel binaries compiled | 9,959 | 0 | 19,068 / 0 |
| LLM / flow / HiFT | 146 / 1,124 / 556 s | 2.4 / 157 / 20 s | 146 / 945 / 3,469 s; 2.5 / 96 / 443 s |
| conv safety checks | 435 s | 21.5 s | 1,628 s / 182 s |
| DRAM with every bucket warmed | 146.6 MiB/bank | the same | 416 MiB/bank |

- **Geometries:** 8 LLM prefill lengths and one decode, 17 flow buckets (64 … 2,560 tokens), and 2 HiFT buckets
  (256 and 512 frames), which chunking reuses. Before: 15 flow and 12 HiFT buckets.
- **The flow is now most of the warm start:** 157 of 195 s. The two buckets the cap lift added (2,048 and 2,560
  tokens) are its largest.
- **The checks caught one corrupted geometry**, in the flow's new 2,560-token bucket: the CFM's
  `Conv1d(320->256, k=3)` at length 5,120. The prepared weight was at relative error 2.13, the raw weight at 0.0026
  (#36487's bug again). HiFT's two geometries had none.
- Construction took 16 s and 13 s before the warm-up. Free DRAM never went below 860 MiB/bank at any conv-cache
  insert, and nothing was evicted.

**Requests** (`demo/demo.py`, the default `--warmup buckets`, on that cache; 0 binaries compiled, warm-up 178.7 s):

| utterance | audio s | tokens | HiFT path | LLM prefill s | LLM decode s | tok/s | flow encoder s | CFM s | HiFT s | wall s | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_shot_121-127105-0003 | 8.52 | 213 | one pass, 512 | 0.031 | 2.230 | 95.5 | 0.305 | 0.840 | 0.276 | 3.692 | 0.433 |
| zero_shot_121-127105-0015 | 3.80 | 95 | one pass, 256 | 0.031 | 1.048 | 90.7 | 0.237 | 0.784 | 0.145 | 2.247 | 0.591 |
| zero_shot_121-127105-0024 | 13.88 | 347 | 2 chunks | 0.032 | 3.569 | 97.2 | 1.117 | 1.384 | 0.709 | 6.813 | 0.491 |
| zero_shot_260-123286-0014 | 3.00 | 75 | one pass, 256 | 0.030 | 0.832 | 90.2 | 0.182 | 0.694 | 0.146 | 1.885 | 0.628 |
| zero_shot_260-123440-0002 | 12.68 | 317 | 2 chunks | 0.032 | 3.258 | 97.3 | 0.702 | 1.032 | 0.699 | 5.724 | 0.451 |
| zero_shot_260-123440-0010 | 8.08 | 202 | one pass, 512 | 0.031 | 2.104 | 96.0 | 0.316 | 0.846 | 0.277 | 3.574 | 0.442 |

- **Aggregate RTF 0.479, worst 0.628.** `tests/perf/test_pipeline_perf.py` passed in its own process: 0 binaries
  compiled, warm-up 176.2 s, worst 0.621, aggregate 0.490.
- **Chunking costs HiFT time on the long utterances:** two 512-frame calls take 0.70–0.71 s. Single pass at 640
  and 768 frames took 0.39–0.47 s. RTF stays under 0.5 for both.
- **WER and SIM on this audio** (`scripts/eval_wer_sim.py`): corpus WER 0.68 % (the same single error) and SIM
  95.87. Before chunking it was 0.68 % and 95.88; the reference scores 0.68 % and 95.21. The two chunked utterances
  moved by about 0.1 in SIM (93.63 → 93.52 and 98.26 → 98.30).
- **Token accuracy on this configuration:** 95.94 % over 5,003 positions ("Token accuracy" below).

### Re-verified on a second N150 (2026-09-29)

**The setup.** A different board, KMD 2.9.0 (the runs above used 2.3.0), firmware 19.11.0, and the same commit.
The reference side was rebuilt from the committed requirements: upstream at `074ca6dc9e80`, the checkpoint at
revision `eec1ae6c`.

**The regenerated PyTorch reference matches the one above:**
- the same audio length for every case;
- the same WER and SIM per case, to 0.01;
- teacher-forced sequences of the same lengths, 5,003 positions.

**The package's tests:**
- **The suite:** 204 passed and 3 skipped (the opt-in tracker and the two reference-venv tests), from an empty
  kernel cache in 73 minutes.
  - Token accuracy is 95.94 % again.
  - The chunked-HiFT seam gate reproduced "Chunked HiFT" below to the last printed digit.
- **The perf test**, in its own process, passed: worst 0.634, aggregate 0.479. It compiled 5,214 binaries: its
  process allocates differently from the suite's, so this start-up was neither cold nor warm.

**The demo** (Stage 1 protocol), on the kernel cache the perf test left:
- 0 binaries compiled; the warm-up took 182.3 s;
- the same tokens as the table above, and the same scores: WER 0.68 %, SIM 95.87;
- RTF 0.436–0.620, aggregate 0.476.

## Stage 1 under the protocol: distinct utterances, warmed buckets (2026-09-28, before chunked HiFT)

`demo/demo.py --inputs <prepare_inputs dir> --out <dir>`: the reported configuration, `--warmup buckets`, seed 1986.
It ran in a fresh process, on the kernel cache the start-up measurement below had filled. It allocates exactly
as those processes did, so every kernel came from disk: 11,702 of 11,702 JIT cache hits, 0 binaries compiled.

**Start-up:** construction 13.4 s, then `warmup_buckets()` 541.8 s (LLM 2.5 s, flow 95.0 s, HiFT 444.3 s). Then
each corpus utterance once; none repeats another.

| utterance | audio s | tokens | flow / HiFT bucket | LLM prefill s | LLM decode s | tok/s | flow encoder s | CFM s | HiFT s | wall s | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_shot_121-127105-0003 | 8.52 | 213 | 384 / 512 | 0.030 | 2.189 | 97.3 | 0.412 | 0.841 | 0.282 | 3.761 | 0.441 |
| zero_shot_121-127105-0015 | 3.80 | 95 | 320 / 256 | 0.031 | 1.026 | 92.6 | 0.255 | 0.794 | 0.147 | 2.254 | 0.593 |
| zero_shot_121-127105-0024 | 13.88 | 347 | 640 / 768 | 0.031 | 3.550 | 97.8 | 1.557 | 1.383 | 0.470 | 6.993 | 0.504 |
| zero_shot_260-123286-0014 | 3.00 | 75 | 256 / 256 | 0.031 | 0.836 | 89.8 | 0.186 | 0.699 | 0.146 | 1.900 | 0.633 |
| zero_shot_260-123440-0002 | 12.68 | 317 | 512 / 640 | 0.031 | 3.233 | 98.1 | 0.748 | 1.027 | 0.387 | 5.428 | 0.428 |
| zero_shot_260-123440-0010 | 8.08 | 202 | 384 / 512 | 0.032 | 2.117 | 95.4 | 0.422 | 0.848 | 0.281 | 3.702 | 0.458 |

Aggregate over the six: 49.96 s of audio in 24.04 s, **RTF 0.481; worst 0.633**. No conv cache evicted anything.
The flow bucket is in tokens (prompt + generated, strictly above), the HiFT bucket in mel frames.

- **`rtf_nonstreaming` is met: `Meets()` is recorded for Wormhole** and enforced by
  `tests/perf/test_pipeline_perf.py`. It passed in its own pytest process: 0 binaries compiled, the same tokens as
  the demo, warm-up 533.9 s, RTF 0.436–0.633 (worst 0.633, aggregate 0.484), no evictions.
- **LLM decode is now the largest stage: 44–60 % of each request, at 90–98 tokens/s.** The CFM is 0.70–1.38 s
  (10 Euler steps), and HiFT is 0.15–0.47 s.
- **The cold first request, for scale:** the same demo in a fresh process, with `--warmup none` and one utterance
  (`zero_shot_121-127105-0003`).
  - With no warm-up its allocations differ from the warm-up's, so the conv kernels that carry DRAM addresses
    recompiled: 706 binaries, 1,207 of 1,738 JIT cache hits.
  - That request took 277.2 s for 8.52 s of audio, RTF 32.5. HiFT alone was 251.5 s.
  - This is the cost `warmup_buckets()` moves to start-up. Warmed, the same utterance takes 3.76 s.

**WER and SIM on the bucketed audio** (`scripts/eval_wer_sim.py`, as below; baseline the PyTorch reference run):

| case | words | WER % reference | WER % TT | SIM reference | SIM TT |
|---|---|---|---|---|---|
| zero_shot_260-123286-0014 | 7 | 14.29 | 0.00 | 95.82 | 95.80 |
| zero_shot_260-123440-0010 | 20 | 0.00 | 0.00 | 96.85 | 98.24 |
| zero_shot_260-123440-0002 | 44 | 0.00 | 2.27 | 97.74 | 98.26 |
| zero_shot_121-127105-0015 | 10 | 0.00 | 0.00 | 92.61 | 94.68 |
| zero_shot_121-127105-0003 | 18 | 0.00 | 0.00 | 94.32 | 94.70 |
| zero_shot_121-127105-0024 | 48 | 0.00 | 0.00 | 93.94 | 93.63 |
| **corpus** | 147 | **0.68** | **0.68** | **95.21** | **95.88** |

The tokens differ from the 09-27 run: the fp32-logit head and the LLM's shorter context (2,048) change what RAS
samples. So these are new utterances, not the old ones re-vocoded. Bucketing's HiFT tail does not show at this
resolution.

**The HiFT tail, three pairs.** The same tokens, flow mel and sine noise, vocoded at the bucket and at the exact
length (levels RMS in dBFS; the diff is bucketed minus exact):

| case | mel frames → bucket | signal, last 0.4 s | diff, last 0.4 s (max) | diff before that (max) |
|---|---|---|---|---|
| zero_shot_260-123440-0010 | 404 → 512 | −52.6 | −55.2 (0.034) | −56.7 (0.016) |
| zero_shot_121-127105-0003 | 426 → 512 | −70.3 | −69.9 (0.004) | −57.6 (0.019) |
| zero_shot_260-123440-0002 | 634 → 640 | −37.1 | −64.2 (0.005) | bit-identical |

- Where the utterance ends in near-silence, the difference sits at the silence's own level.
- Where sound runs to the end, the difference is 27 dB below it.
- The low-level difference through the whole utterance, 34–36 dB below the speech, is the geometry effect described
  under "Bucketing": different kernel blockings, whose rounding the NSF source's running phase accumulates.
- The wavs are kept for listening outside the repository.

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

**The larger sample (2026-09-28).** The corpus's token-accuracy extension (scripts/corpus.py) adds 20
sequences: six more targets from each primary speaker, and two new speakers, 672 (M) and 237 (F), with four each.
Same method, same reference seed, and the pipeline's current configuration (LLM context 2,560):

| sequences | positions | TT (fp32-logit head) | PyTorch reference in bf16 vs fp32 | the same with an fp32 head |
|---|---|---|---|---|
| the first seven (six LibriSpeech + parity) | 1,349 | 96.37 % | 95.70 % | 98.37 % |
| the extension, 20 | 3,654 | 95.79 % | 96.72 % | 98.66 % |
| **all 27** | **5,003** | **95.94 %** | 96.45 % | 98.58 % |

- **It holds, with a thin margin:** 0.94 points over the target on 5,003 positions. The first seven gave the same
  96.37 % as before, case for case, so the longer context changed nothing there.
- **Per sequence it spans 92.4–100 %**, so a handful of sequences would not have been enough to tell.
- The 203 disagreements are all at small reference margins: median 0.024 nats, maximum 0.226.
- TT sits 0.5 points under the pure-bf16 PyTorch run of the same model.

**Why fp32 logits are the default.** The noise floor shows how little room a bf16 head leaves:
- even the reference itself, run in bf16 with no port error at all, clears 95 % by only 0.7 points (95.70 %);
- TT's bf16 head measured 90.66 %, 4.3 points under the target;
- an fp32 head lifts the bf16 reference by 2.7 points (to 98.37 %), and TT by 5.7 (to 96.37 %).

The head is one 896 × 6,564 matmul per decode step. Its fp32 accumulation and output cost about 0.3 ms per step,
about 3 % of the decode time. So `CosyVoice2Config.llm_head_logits_dtype` defaults to `"float32"`; `"bfloat16"`
restores the old head.

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
- **So a segment's speech was capped at 1,024 tokens** (2,048 frames, 41 s): `CosyVoice2Config.max_segment_speech_tokens`.
  Chunked HiFT (below) has since lifted it back to upstream's own 1,600 (2026-09-28).
- **Past the cap, the pipeline raises `SegmentTooLong`, naming the segment's length**, rather than truncating the
  speech or failing inside a device op (`test_pipeline_api.py::test_segment_past_the_cap_raises`). The LLM runs one
  step past the cap, which tells a segment of exactly the cap from a longer one, and `tokens_to_mel` refuses more
  than the cap before any device work. At upstream's own 1,600, no segment of at most 80 text tokens can reach it.
- **The resulting sets** step by one unit up to 8 units, then the step doubles every 8 steps (`tiered_buckets`).
  Since chunked HiFT (2026-09-28):

  | stage | buckets |
  |---|---|
  | flow tokens | 17 buckets, 64 … 2560, strictly above prompt + generated (15, up to 1792, under the 1,024 cap) |
  | HiFT mel frames | 2 buckets, 256 and 512, below one chunk; a longer mel runs in 512-frame chunks (12, up to 2048, before) |
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
  - **20 were real corruption of the fast path**, the prepared weight. Relative error against a float64 host
    conv:

    | conv | where | prepared weight | raw weight, used instead |
    |---|---|---|---|
    | `Conv1d(128->128, k=11)`, 6 resblock convs | the 128-frame bucket (length 5,120, 40 x 128) | 1.0–2.6 | 0.003–0.005 |
    | `Conv1d(18->256, k=30, s=15)`, the first source downsampling | every bucket from 640 to 2,048 frames | 7.7 | 0.0018 |
    | `Conv1d(18->128, k=6, s=3)`, the second | every bucket from 896 to 2,048 frames | 0.14–0.19 | 0.0020 |

    **What it is: tenstorrent/tt-metal#36487's bug.** `prepare_conv_weights` is wrong when the conv runs DRAM-sliced;
    all three convs here see DRAM inputs, which conv1d auto-slices. Checked on 2026-09-28:
    - **It is not our call.** Passing the conv's own compute config to `prepare_conv_weights` makes the k=11 case far
      worse (1e28–1e29), and passing a matching slice config changes nothing.
    - **It reproduces standalone** with random weights. With the same slice config given to prepare and to the conv,
      prepared weights are wrong under explicit DRAM slicing at every geometry tried, including ones that auto
      slicing gets right. The same conv with its input in L1, no slicing, is right. `act_block_h_override=1024`
      (#35852's workaround) doesn't help.
    - **#36487's own reproducer fails on this build too:** prepared PCC 0.00035, raw 0.999912.

    A comment for #36487 with these geometries is drafted (notes branch), not posted. Chunked HiFT runs only the
    256- and 512-frame geometries, where the prepared weights verified correct.

  - **23 were the reference's own error.** The raw-weight / safe-config reference was off by 0.05–0.10 while the
    fast path was at 0.004–0.005, and the float64 host conv kept the fast path. These arbitrations are most of the
    check time at long lengths.
- **HiFT is 77 % of the warm start-up.** Its two largest buckets (1,792 and 2,048 frames) take 254 s of it.

**Eviction cannot fire.** The warmed set holds 416 MiB of DRAM per bank (82 MiB after construction), and 601 MiB per
bank stays free. Across the warm-up's 3,060 conv-cache inserts, free DRAM never dropped below 559 MiB per bank.
The old eviction threshold was 150 MB free, so it would never have fired either. In bucketed mode the threshold is
0, so eviction is off (`test_geometry_cache.py::test_threshold_override_zero_never_evicts`).

**The L1 option doesn't fit with margin.** With config tensors in L1_SMALL, the kernels carry no DRAM address, so
the warm-up would not need to be deterministic. The same warm-up ran with `conv_config_tensors_in_dram=False` and
`l1_small_size` 192 KiB, set to stop once L1_SMALL passed 160 KiB per bank:

| geometries warmed | L1_SMALL per bank |
|---|---|
| all 15 flow buckets | 34.6 KiB (1.9–3.1 KiB each) |
| + HiFT 128 … 768 frames (6 buckets) | 172.6 KiB (18–33 KiB each, growing with length); stopped here |

- **The six HiFT buckets it didn't reach are the longest.** Even at the last bucket's 33 KiB each, the set would
  need at least 372 KiB per core; with growth proportional to length, about 540 KiB.
- **At 2,048 frames HiFT's `ttnn.concat` needs about 960 KiB** for one circular-buffer page. That is scaled from
  the 1,536,032 B measured at 3,200 frames, not measured itself. Of the 1,464 KiB of L1 per core (40 KiB
  reserved), that leaves at most 465 KiB for L1_SMALL.
- **So config tensors stay in DRAM,** and the deterministic warm-up is the design. In DRAM they add almost nothing:
  after the 768-frame bucket, DRAM stood at 199.6 MiB per bank with them in DRAM, 198.9 MiB with them in L1.

## Chunked HiFT (2026-09-28)

A mel of 512 frames or more now runs through HiFT in 512-frame calls with upstream's streaming cache
(`tt/hifigan/chunking.py`, `TtHiFTGenerator.inference_chunked`):
- each call after the first re-synthesizes the previous call's last 8 frames, with the previous call's NSF source
  carried over, and the two outputs are crossfaded with upstream's Hamming window (7,680 samples);
- the last call is anchored to the end of the mel, so nothing is padded, and it carries the source over its whole
  overlap;
- a mel shorter than 512 frames runs once at 256 or 512 frames, padded with silence as before.

So HiFT has two geometries instead of twelve, and no length limit: the segment cap is back to upstream's own 1,600
tokens (the flow's buckets and the LLM context grow to cover it: 17 flow buckets up to 2,560 tokens, context 2,560).

**The seam gate** (`tests/pcc/test_hift_chunked.py`). In the reference venv, upstream's own `HiFTGenerator`
(`scripts/hift_streaming_reference.py`) ran three real test-clean mels on the same schedule, through its own
`inference(cache_source=...)` and `fade_in_out`, with one fixed sine-noise draw. TT runs the same mels with the same
noise. Seam windows are the 160 ms crossfade ± 40 ms.

| mel | calls | seam | mechanism (upstream's F0 injected): PCC / max \|diff\| | no crossfade (control): PCC / max \|diff\| |
|---|---|---|---|---|
| 260-123288-0025, 600 frames | 2, the last anchored (424-frame overlap) | 1 | 0.99908 / 0.0135 | 0.99761 / 0.0199 |
| 4992-23283-0012, 1,016 frames | 2, lined up | 1 | 0.99862 / 0.0077 | **0.91408 / 0.1028** |
| 7021-79730-0003, 1,500 frames | 3, the last anchored (28-frame overlap) | 1 | 0.99855 / 0.0017 | 0.99795 / 0.0017 |
| | | 2 | 0.99948 / 0.0337 | 0.99810 / **0.0701** |

- **The gate:** at every seam, PCC ≥ 0.995 and max |diff| ≤ 0.05; over the whole signal, PCC ≥ 0.998 (measured
  0.99927–0.99940). The mechanism passes everywhere.
- **The control fails at two of the four seams.** At the other two, the two calls already agree over the overlap to
  within the gate (the 1,500-frame mel's first seam is near-silent: max |diff| 0.0017 either way), so a missing
  crossfade has nothing to show there.
- chunking.py's stitch reproduces upstream's `fade_in_out` stitch exactly (max |diff| 0).

**Own F0, spectral** (log-mel L1, TT chunked vs upstream chunked; seams ± 100 ms):

| mel | whole | around each seam | for scale: upstream chunked vs upstream single pass |
|---|---|---|---|
| 600 frames | 0.090 | 0.115 | 0.011 |
| 1,016 frames | 0.101 | 0.110 | 0.036 |
| 1,500 frames | 0.109 | 0.109, 0.107 | 0.039 |

The gate: whole ≤ 0.13, and no seam above 1.5x its utterance's whole-signal figure.

**Chunking adds nothing to the port's own spectral error.** The same mels through single-pass HiFT (log-mel L1):

| mel | TT vs upstream, single pass, own F0 | the same, torch F0 injected | TT vs upstream, chunked, own F0 | TT chunked vs TT single | upstream chunked vs upstream single |
|---|---|---|---|---|---|
| 600 frames | 0.090 | 0.050 | 0.090 | 0.024 | 0.011 |
| 1,016 frames | 0.100 | 0.062 | 0.101 | 0.033 | 0.036 |
| 1,500 frames | 0.121 | 0.087 | 0.109 | 0.069 | 0.039 |

- TT is as far from upstream chunked as single-pass: 0.09–0.12 is the port's own-F0 error on these mels.
- Chunking moves TT's spectrum about as much as it moves upstream's.
- In the waveform, chunked vs single pass has PCC only 0.49–0.82, even in upstream itself: past the carried overlap,
  each call's sine phase restarts. The spectrum barely moves.

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

- **Start-up is 3.2 minutes with the kernels on disk, 30.5 without** (chunked HiFT, above). The flow is now 81 %
  of the warm start (157 s). Persisting the conv safety checks' verdicts (21.5 s now) is deferred.
- **tenstorrent/tt-metal#36487** (prepared conv weights wrong under DRAM slicing) is worked around by the per-geometry
  checks. A comment with our geometries is drafted, not posted.
- **Streaming (Stages 2 and 3)** is not built. Chunked HiFT is the vocoder half of it.
