# Validation evidence

What has been measured, how, and what is still open. It grows as measurements arrive: nothing here is an estimate,
and every figure names its run. The bounty's numeric targets (tenstorrent/tt-metal#54104) are declared once in
[`../tests/perf/gates.py`](../tests/perf/gates.py), with the per-architecture verdicts recorded so far.

| target (#54104) | stage | status | enforced in |
|---|---|---|---|
| RTF < 1.0, non-streaming whole-utterance synthesis | Stage 1 | **met: worst 0.654, aggregate 0.483** over six distinct utterances after the bucket warm-up, on the masked HiFT (2026-09-30; the perf test: worst 0.675, aggregate 0.481; below); `Meets()` recorded | `tests/perf/test_pipeline_perf.py` |
| token-level accuracy > 95 % against the PyTorch reference | Stage 1 | **met: 95.94 %** teacher-forced over 5,003 positions (27 sequences, 4 speakers), with the LLM's fp32-logit head (below); `Meets()` recorded | `tests/e2e/test_token_accuracy.py` |
| WER < 5.0 | Stage 1 | **met: corpus WER 0.68 % in each of five noise draws**, the same as the PyTorch reference's (2026-09-30, masked HiFT; "WER and similarity over five noise draws" below); `Meets()` recorded | not by a test: `scripts/eval_wer_sim.py` runs in the reference venv |
| speaker similarity > 0.60 | Stage 1 | **met: 95.88 (95.84–95.92 over five noise draws)**, reference 95.22 (95.21–95.24) (WavLM-base-plus-sv cosine x 100; below); `Meets()` recorded | same |
| time-to-first-packet < 500 ms; RTF < 0.4 streaming | Stage 3 | **missed: first audio at 1.31–1.50 s, worst streaming RTF 1.10–1.12** (aggregate 0.84–0.85) over six distinct utterances, two runs after both warm-ups on the masked HiFT (2026-09-30; "Streaming, measured" below); `Misses()` recorded, with the lever | `tests/perf/test_pipeline_perf.py` (streaming): each figure held inside its recorded band |

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
  - The chunked-HiFT seam gate, still on its first three mels then, reproduced their 09-28 figures to the last
    printed digit.
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

**HiFT: padding with silence, the tail it touches.** (Superseded on 2026-09-30: the padding is masked and the tail
matches upstream's; "Masked end padding in HiFT".) The mel is padded with `log(1e-5)`, the mel's own silence floor
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
    - **It is not the configs we pass.** Passing the conv's own compute config to `prepare_conv_weights` makes the
      k=11 case far worse (1e28–1e29), and passing a matching slice config changes nothing. The input layout we
      *declare* does matter: "Prepared conv weights: the declared input layout" below.
    - **It reproduces standalone** with random weights. With the same slice config given to prepare and to the conv,
      prepared weights are wrong under explicit DRAM slicing at every geometry tried, including ones that auto
      slicing gets right. The same conv with its input in L1, no slicing, is right. `act_block_h_override=1024`
      (#35852's workaround) doesn't help.
    - **#36487's own reproducer fails on this build too:** prepared PCC 0.000768 (2026-09-29, a second board; 0.00035
      on 2026-09-28), raw 0.999912.

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

## Prepared conv weights: the declared input layout (#36487, 2026-09-29)

`TtConv1d` prepares each geometry's weight with `ttnn.prepare_conv_weights`, declaring the activation's layout
(TILE). Where `conv1d` slices its input through DRAM, that weight is wrong (#36487, above). Declaring ROW_MAJOR
instead gives a correct weight there, and a wrong one wherever the TILE declaration was right.

**Standalone**, relative error against a float64 torch conv:
- random weights and inputs;
- the pipeline's dtypes and configs, input TILE in DRAM;
- script: notes branch, `scripts/2026-09-29/r1_prepare_layout.py`.

| conv | length | TILE declared | ROW_MAJOR declared | raw weight |
|---|---|---|---|---|
| `Conv1d(128->128, k=11, d=1/3/5)`, HiFT resblocks | 4,320, 5,120, 8,320 (108, 128, 208 frames) | 1.36–3.78 | 0.0039–0.0055 | = ROW_MAJOR |
| the same, d=1 | 10,240 (256 frames) | 0.0040 | 1.37 | = TILE |
| `Conv1d(18->256, k=30, s=15)`, the first source downsampling | 640 … 2,048 frames (8 lengths) | 1.23 | 0.0040 | = ROW_MAJOR |
| the same | 108 … 512 frames (5 lengths) | 0.0019 | 1.13–1.17 | = TILE |
| `Conv1d(18->128, k=6, s=3)`, the second | 896 … 2,048 frames (6 lengths) | 1.15 | 0.0039 | = ROW_MAJOR |
| the same | 108 … 512 frames (5 lengths) | 0.0018 | 1.07 | = TILE |
| `Conv1d(320->256, k=3)`, the flow's CFM (bf16, batch 2) | 5,120 (the 2,560-token bucket) | 1.26 | 1.36 | 0.0036 |
| `Conv1d(256->256, k=3)`, the same place | 5,120 | 0.0032 | 0.0032 | 0.0032 |

- **Over these 36 geometries, one declaration is wrong wherever the other is right.** The right one gives exactly
  the raw weight's error.
- **The flow's `Conv1d(320->256, k=3)` is the exception:** wrong both ways.
- **#36487's own reproducer on this board:**
  - declaring TILE (as written): PCC 0.000768, with inf in the output;
  - declaring ROW_MAJOR: 0.999912;
  - the raw weight: 0.999912.
- **The code suggests why, for `conv1d`:**
  - `conv1d` routes DRAM inputs through DRAM width slicing (`conv1d.cpp:82-88`).
  - A sliced op, or a ROW_MAJOR input, gets a smaller input-channel alignment (`get_input_channels_alignment`,
    `conv2d_utils.cpp:92-99`).
  - But `prepare_conv_weights` never takes the DRAM path for a 1-D conv (`prepare_conv2d_weights.cpp:1313`). So
    given TILE, it pads the weight's channels for an unsliced TILE conv.

**The resolver's fourth candidate** (`TtConv1d._verify_and_resolve`):
- On a disagreement, a ROW_MAJOR-prepared weight joins the float64 arbitration, after the TILE-prepared weight and
  before the raw weights.
- On a tie a prepared, traceable weight wins.
- `test_conv1d_verification.py::test_conv1d_resolver_keeps_a_prepared_weight_where_the_tile_one_is_wrong` runs it at
  two real broken geometries. The ROW_MAJOR candidate wins at both, tied with the raw weight at 0.0039:
  - a resblock conv at 5,120 (TILE: inf);
  - the first source downsampling at 76,801 (TILE: 1.21).

**In the pipeline:** the Stage 1 demo after the change, on the kernel cache the baseline left.
- **898 binaries recompiled.** Where the checks fire, the extra candidate shifts the allocation sequence. The
  warm-up took 533.9 s.
- **Five geometries disagreed in the warm-up, and none changed hands.**
  - The flow CFM's `Conv1d(320->256, k=3)` at the 2,560-token bucket: TILE 2.126, ROW_MAJOR 2.267, raw 0.00265. It
    stays on the raw weight, as before.
  - Four HiFT resblock geometries (`128->128` at 10,240 and 20,480, `64->64` at 30,721 and 61,441): the safe
    reference's own error (0.055–0.099). The TILE-prepared weight is kept.
- **The six utterances:** the same tokens and bit-identical audio as the baseline before the change, at RTF
  0.432–0.630 (aggregate 0.474).

The candidate matters for streaming: HiFT at 108 and 208 frames puts every k=11 resblock conv where TILE is wrong.

## Chunked HiFT (2026-09-28)

A mel of 512 frames or more now runs through HiFT in 512-frame calls with upstream's streaming cache
(`tt/hifigan/chunking.py`, `TtHiFTGenerator.inference_chunked`):
- each call after the first re-synthesizes the previous call's last 8 frames, with the previous call's NSF source
  carried over, and the two outputs are crossfaded with upstream's Hamming window (7,680 samples);
- the last call is anchored to the end of the mel, so nothing is padded, and it carries the source over its whole
  overlap;
- a mel shorter than 512 frames runs once at 256 or 512 frames, padded with silence as before. (Since 2026-09-30
  that padding is masked, so it computes upstream's call exactly: "Masked end padding in HiFT" below.)

So HiFT has two geometries instead of twelve, and no length limit: the segment cap is back to upstream's own 1,600
tokens (the flow's buckets and the LLM context grow to cover it: 17 flow buckets up to 2,560 tokens, context 2,560).

**The seam gate** (`tests/pcc/test_hift_chunked.py`), strengthened on 2026-09-29.

The reference: in the reference venv, upstream's own `HiFTGenerator` (`scripts/hift_streaming_reference.py`) runs the
same schedule through its own `inference(cache_source=...)` and `fade_in_out`, with one fixed sine-noise draw. TT runs
the same mels with the same noise.

The mels:
- six real test-clean mels, one per speaker, three female and three male;
- each cut so that every crossfade lands in voiced speech: upstream's own F0 above 10 Hz over the crossfade ±4 frames;
- nine seams in all;
- the lengths cover an anchored last call with a long overlap (600, 1,100 frames) and with a short one (800,
  1,300), and calls that line up (1,016, 1,520).

The metric at each seam: the error relative to the signal over the 160 ms crossfade, ‖TT − upstream‖ / ‖upstream‖.

| mel (speaker) | frames, calls | seam | mechanism (upstream's F0 injected): rel. error / PCC ±40 ms | no crossfade (control): rel. error | upstream's own two calls over the crossfade |
|---|---|---|---|---|---|
| 121-123852-0000 (F) | 600, 2 (the last anchored, 424-frame overlap) | 1 | 0.075 / 0.99763 | 0.147 | 0.128 |
| 260-123288-0015 (M) | 800, 2 (the last anchored, 224) | 1 | 0.078 / 0.99919 | 0.145 | 0.597 |
| 1221-135766-0011 (F) | 1,016, 2 (lined up) | 1 | 0.044 / 0.99930 | 0.200 | 0.414 |
| 672-122797-0008 (M) | 1,100, 3 (the last anchored, 428) | 1 | 0.053 / 0.99884 | 0.180 | 0.318 |
| | | 2 | 0.041 / 0.99954 | 0.108 | 0.284 |
| 1995-1836-0004 (F) | 1,300, 3 (the last anchored, 228) | 1 | 0.051 / 0.99979 | 0.303 | 0.421 |
| | | 2 | 0.050 / 0.99958 | 0.116 | 0.275 |
| 908-157963-0007 (M) | 1,520, 3 (lined up) | 1 | 0.069 / 0.99820 | 0.473 | 0.625 |
| | | 2 | 0.066 / 0.99952 | 0.387 | 0.489 |

- **The gate:**
  - at every seam, relative error ≤ 0.10 and PCC ≥ 0.995;
  - over the whole signal, PCC ≥ 0.995 (measured 0.99641–0.99959);
  - the control must fail at every seam.
- **The mechanism passes with a modest margin:** 0.078 at worst, against 0.10.
- **The control fails at all nine seams** (0.108–0.473), narrowly at one of them (0.108).
- **max |diff| is printed, no longer gated.**
  - It follows loudness: at the loudest seam it is 0.11 in both arms, so it sits outside the crossfade itself.
  - The 09-28 gate (≤ 0.05) was set on mels whose seams were partly near-silent.
- **The crossfade's gain.** Upstream's two Hamming halves sum to 1.0798–1.0800, so where the two calls agree, the
  overlap is about 8 % louder than either. TT keeps that gain deliberately, for parity. Here, though, upstream's own
  two calls differ by 0.13–0.63 over every crossfade (the last column). So a missing crossfade shows as a real
  discontinuity at every seam, not only as missing gain.
- **The lowest whole-signal PCC** (0.99641, the high-F0 female voice 1221-135766-0011) is the port's own HiFT error
  with F0 injected, not the chunking: its seam measures 0.044.
- chunking.py's stitch reproduces upstream's `fade_in_out` stitch exactly (max |diff| 0).
- **The first gate** (2026-09-28): three mels, four seams, PCC and max |diff|.
  - Two of those seams were near-silent, and the control failed at only two of the four.
  - Measured by the relative error, the near-silent seam reads 0.154 for the mechanism, from a tiny signal. That is
    why every seam is voiced now.

**Own F0, spectral** (log-mel L1, TT chunked vs upstream chunked; seams ±100 ms):

| mel | whole | around each seam | for scale: upstream chunked vs upstream single pass |
|---|---|---|---|
| 121-123852-0000 | 0.097 | 0.120 | 0.013 |
| 260-123288-0015 | 0.091 | 0.110 | 0.019 |
| 1221-135766-0011 | 0.076 | 0.085 | 0.031 |
| 672-122797-0008 | 0.085 | 0.087, 0.090 | 0.035 |
| 1995-1836-0004 | 0.089 | 0.073, 0.084 | 0.038 |
| 908-157963-0007 | 0.090 | 0.088, 0.082 | 0.047 |

The gate: whole ≤ 0.13, and no seam above 1.5x its utterance's whole-signal figure (measured at most 1.24).

**Chunking adds nothing to the port's own spectral error** (2026-09-28, the first gate's three mels, through single-pass HiFT; log-mel L1):

| mel | TT vs upstream, single pass, own F0 | the same, torch F0 injected | TT vs upstream, chunked, own F0 | TT chunked vs TT single | upstream chunked vs upstream single |
|---|---|---|---|---|---|
| 600 frames | 0.090 | 0.050 | 0.090 | 0.024 | 0.011 |
| 1,016 frames | 0.100 | 0.062 | 0.101 | 0.033 | 0.036 |
| 1,500 frames | 0.121 | 0.087 | 0.109 | 0.069 | 0.039 |

- TT is as far from upstream chunked as single-pass: 0.09–0.12 is the port's own-F0 error on these mels.
- Chunking moves TT's spectrum about as much as it moves upstream's.
- In the waveform, chunked vs single pass has PCC only 0.49–0.82, even in upstream itself: past the carried overlap,
  each call's sine phase restarts. The spectrum barely moves.

## Masked end padding in HiFT (2026-09-30)

**Why.** Stage A's streaming WER was 1.36 % against 0.68 % on 09-28, because Whisper appended "you" to
260-123440-0010 (notes: B28). The cause was the padding of two HiFT calls:
- streaming's final call, padded at its end with silence mel to 128 or 256 frames;
- Stage 1's single call for a mel under 512 frames, padded the same way to 256 or 512.

HiFT's convs look ahead into that silence. So the last 500–620 samples (~25 ms) of every streamed utterance, and of
three Stage 1 utterances, fell to −104 to −139 dBFS, where upstream's audio runs on at −52 to −89 dBFS. On that clip
Whisper's first-token decision sits near a tie. The silenced ending tips it into a mode that decodes the last 20 ms
as "you" in 5 of 11 noise draws. It was not the scorer (12 identical runs) and not the lengths (identical to
upstream's, whole and per chunk).

**The fix** (`tt/hifigan/valid_length.py`). The call is still padded to its bucket, so no new geometry is compiled.
But it computes upstream's call at the real length, because every padding site is checked against upstream's code:
- the mel, every conv's output and each stage's sum are zeroed past the real length at their own rate. Snake,
  leaky ReLU and ELU keep zeros at zero;
- F0 is zero past the real frames. SineGen2's phase is a cumsum interpolated back up linearly, and upstream's
  interpolation clamps at the last real frame; a flat phase past it is the same thing;
- the 8 source samples after the real end become the reflection of the last real ones (`torch.stft`'s centering),
  and the STFT frames past the real ones are zeroed;
- the iSTFT's magnitude is zeroed past the real frames (`exp(0)` would be 1). The last 3 samples are rescaled to the
  real call's window normalization (`istft_end_gain`).

ReflectionPad1d((1, 0)) pads the start only, so it is untouched. The streaming first call, padded in front at the
utterance's start, is unchanged.

**The rules on the host** (`tests/pcc/test_hift_masked.py`, torch with the real checkpoint, 8 real/bucket pairs from
34/128 to 426/512):
- **With the same F0 on both sides**, the masked padded call equals the call at the real length: max |diff| ≤ 1.2e-6
  and PCC 1.00000000, over the whole waveform and over its last 20 ms.
- **The F0 predictor alone** equals it exactly in 5 of 8 cases, and within 7e-3 Hz in the others (its convs round
  differently at another length). SineGen2 integrates F0 into the phase over the whole call, so compared end to end
  that rounding grows to ~1e-3 in the waveform. Hence the two parts.
- **Each rule matters** (the "you" clip's shape, 62 of 128, last 20 ms max |diff|): without the source's
  reflection, 2.4e-5; without the end gain, 5.6e-4; with F0 held past the end instead of zeroed, 2.4e-3; with no
  masks at all, 4.7e-2.

**On the device, against the exact-length call**: the same HiFT at the real length, compiled for each length. That
was 5,429 kernels for ten lengths, which is why the exact length is only a reference, not the fix.

| call | real / bucket frames | last 20 ms: max \|diff\|, PCC | whole utterance: PCC |
|---|---|---|---|
| streaming final calls, six (same F0 / own F0) | 34–138 / 128–256 | ≤ 4.1e-4, 0.9988–0.9999 | 0.99993–1.000000 |
| Stage 1's padded calls, four (same F0 / own F0) | 150–426 / 256–512 | ≤ 4.9e-4, 0.9987–0.9998 | 0.99983–0.99994 |

- **The whole-utterance max |diff| reaches 3.6e-2.** The device builds SineGen2's phase with matrices sized to the
  call, so another length rounds the running phase differently. This is the geometry effect "Bucketing" describes.
  Silence padding scored PCC 0.9836–0.9994 on the same comparison.
- **The final chunk against upstream's streaming**, whole chunk, masked / exact: 0.99696 / 0.99749 on 121-127105-0015
  and 0.99931–0.99979 / 0.99953–0.99982 on the others. 0015's final chunk is 13 tokens ending near −71 dBFS, so its
  PCC measures the port's own noise floor at the exact length too.

**The end gate (notes: D41)** replaces D38's tail criterion. The last 20 ms must be within 3 dB of the reference's
RMS level, with no absolute floor:
- Stage 1: against torch's HiFT at the real length, same F0 and noise (`test_hift_masked.py`);
- streaming: against upstream's streaming, fed upstream's mel, F0 and noise (`test_streaming.py`).

D38's −50 dBFS floor is what let the silenced endings through: it passed 260-123440-0010's with the difference
2.5 dB below the signal. The new gate fails on the old padding and passes on the fix:

| gate | case | last 20 ms, dBFS: silence padding / masked / reference |
|---|---|---|
| Stage 1 | 121-127105-0003 | −115.2 / −64.5 / −65.0 |
| Stage 1 | 121-127105-0015 | −104.1 / −88.1 / −88.9 |
| Stage 1 | 260-123286-0014 | −138.2 / −62.7 / −63.1 |
| Stage 1 | 260-123440-0010 | −130.1 / −51.5 / −51.9 |
| streaming | 121-127105-0003 | −117.6 / −64.6 / −64.9 |
| streaming | 121-127105-0015 | −103.8 / −88.4 / −88.8 |
| streaming | 121-127105-0024 | −112.3 / −67.7 / −68.0 |
| streaming | 260-123286-0014 | −138.7 / −63.2 / −63.4 |
| streaming | 260-123440-0002 | −113.9 / −81.1 / −81.6 |
| streaming | 260-123440-0010 | −131.1 / −51.7 / −51.9 |

**The "you" clip, 11 noise draws** (`test_streaming.py::test_device_you_clip_noise_draws` renders them, and
`tests/reference/test_you_clip.py` transcribes them in the reference venv):
- Before: 5 of 11 ended in "you". The first token was within ±0.16 nats of a tie in every draw.
- Masked: 0 of 11, with 0 word errors in each. Every ending is within 0.2–0.6 dB of upstream's.

**Cost.**
- About 80 elementwise masks per padded call, and a host round trip of the source.
- No kernels per length: the masked programs are the bucket's, compiled in the warm-ups.
- The final call's HiFT took 0.140 s against 0.120 s in the interleaved test. That run shared the host with CPU jobs;
  the clean figure is in "Streaming, measured".

## WER and similarity over five noise draws (2026-09-30)

**Why.** One noise draw's trailing "you" moved the streaming corpus WER from 0.68 % to 1.36 % (B28). So WER and
similarity are now reported as the mean and range over five draws of the vocoder's noise per utterance (seeds 1–5).
The tokens are fixed by the LLM's seed (1986), so only the noise varies.
- TT: `scripts/noise_draws.py`.
- The reference: `run_reference.py --noise-seed` and `streaming_reference.py --noise-seed`.
- Scoring: `scripts/eval_draws.py`, which uses the corpus scorer's own functions, unchanged.

Every draw sampled the same tokens as the undrawn runs. For Stage 1, TT and the reference each use their own tokens,
which differ, as before. For streaming, both stream TT's Stage 1 tokens: TT live (`synthesize_stream`), upstream
offline over the same tokens. TT runs the masked HiFT (`ed1c3ad1c5`).

| case | words | WER %: TT Stage 1 | reference Stage 1 | TT streaming | reference streaming | SIM: TT Stage 1 | reference Stage 1 | TT streaming | reference streaming |
|---|---|---|---|---|---|---|---|---|---|
| 121-127105-0003 | 18 | 0.00 | 0.00 | 0.00 | 0.00 | 94.75 (94.68–94.87) | 94.36 (94.33–94.41) | 94.53 (94.48–94.61) | 95.11 (95.08–95.14) |
| 121-127105-0015 | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 94.68 (94.57–94.77) | 92.59 (92.56–92.61) | 93.40 (93.28–93.59) | 93.72 (93.67–93.76) |
| 121-127105-0024 | 48 | 0.00 | 0.00 | 0.00 | 0.00 | 93.51 (93.47–93.54) | 93.93 (93.90–93.96) | 93.84 (93.82–93.87) | 93.39 (93.38–93.40) |
| 260-123286-0014 | 7 | 0.00 | 14.29 | 0.00 | 0.00 | 95.91 (95.88–95.92) | 95.85 (95.77–95.89) | 96.42 (96.38–96.52) | 96.10 (96.07–96.13) |
| 260-123440-0002 | 44 | 2.27 | 0.00 | 2.27 | 2.27 | 98.29 (98.27–98.33) | 97.74 (97.72–97.74) | 98.47 (98.42–98.50) | 98.60 (98.59–98.61) |
| 260-123440-0010 | 20 | 0.00 | 0.00 | 0.00 | 0.00 | 98.19 (98.16–98.21) | 96.87 (96.85–96.94) | 98.32 (98.29–98.37) | 98.44 (98.39–98.47) |
| **corpus** | 147 | **0.68** | **0.68** | **0.68** | **0.68** | **95.88 (95.84–95.92)** | **95.22 (95.21–95.24)** | **95.83 (95.81–95.87)** | **95.89 (95.87–95.91)** |

- **No utterance's WER moves between draws**, in any of the four groups, so each WER cell is the same in all five.
- **TT's one error** (260-123440-0002, 1 word in 44) is the one upstream's streaming makes on the same tokens. The
  reference's Stage 1 error is on its own tokens of 260-123286-0014.
- **Similarity varies by at most 0.3 between draws.**
- **The "you" clip's regression test** (11 draws, "Masked end padding in HiFT") stays in the suite.

## Stage 1 and streaming re-run on the masked HiFT (2026-09-30)

Everything was re-run on `5317572d0c` on the second N150 (KMD 2.9.0), with nothing else on the host:
- **The device suite:** 228 passed and 4 skipped (the opt-in tracker and three reference-venv tests), in 27 min. It
  compiled 1,103 kernels.
- **The Stage 1 perf test**, in its own process: worst RTF 0.675, aggregate 0.481. The warm-up took 184.5 s and
  compiled nothing.
- **The Stage 1 demo**, with the same tokens as 09-28 and 09-29: RTF 0.441–0.654, aggregate 0.483. HiFT against
  09-29's run of the same tokens, with silence padding:

  | case | mel frames → HiFT call | HiFT s, 09-29 / masked | RTF, 09-29 / masked |
  |---|---|---|---|
  | 121-127105-0003 | 426 → 512, padded | 0.280 / 0.382 | 0.438 / 0.441 |
  | 121-127105-0015 | 190 → 256, padded | 0.145 / 0.215 | 0.588 / 0.606 |
  | 121-127105-0024 | 694, chunked | 0.709 / 0.726 | 0.487 / 0.481 |
  | 260-123286-0014 | 150 → 256, padded | 0.143 / 0.214 | 0.620 / 0.654 |
  | 260-123440-0002 | 634, chunked | 0.697 / 0.707 | 0.449 / 0.451 |
  | 260-123440-0010 | 404 → 512, padded | 0.277 / 0.378 | 0.436 / 0.457 |

  The masked call costs 0.07 s at the 256-frame bucket and 0.10 s at 512: about 80 masks, and a host round trip of
  the source. The chunked calls are unpadded and unchanged.
- **WER and similarity:** "WER and similarity over five noise draws" above. 0.68 % in every draw; similarity 95.88.
- **Streaming** (`demo.py --stream`, R5's protocol, two fresh processes, the Stage 1 tokens):

  | | run 1 | run 2 | 09-29, silence padding (R5) |
  |---|---|---|---|
  | time to first audio | 1.353–1.502 s | 1.313–1.432 s | 1.336–1.479 s |
  | streaming RTF, aggregate | 0.851 | 0.836 | 0.843–0.853 |
  | streaming RTF, worst (the 3.8 s utterance) | 1.121 | 1.103 | 1.057–1.122 |

  The warm-ups took 184.6 and 149.0 s, then 184.3 and 149.4 s, and compiled nothing. The final call's masks move
  neither figure outside the spread of 09-29's runs. The first chunk: 0.37–0.47 s until it starts, flow
  0.82–0.91 s (CFM 0.68–0.74 s), HiFT 0.12–0.13 s.
- **The streaming perf test** (`tests/perf/test_pipeline_perf.py`, R6, its own process, the same protocol) enforces
  both Stage 3 figures through `tests/perf/gates.py`. Its first run: worst first audio 1,469 ms (best 1,404) and
  worst RTF 1.110 (aggregate 0.867), both inside their bands. Both warm-ups took 335 s.

## Start-up and the cold first request, re-measured (2026-09-30)

On the masked HiFT (R6; notes B22, D37: the rebuild spec's figures for these were unverified). Two fresh processes
ran against one new `TT_METAL_CACHE` directory, the first starting empty and the second identical, each timing
construction and both warm-ups (notes: `scripts/2026-09-30/startup_measure.py`):

| | cold: empty kernel cache | warm: every kernel on disk |
|---|---|---|
| construction | 15.4 s | 12.0 s |
| `warmup_buckets()` | 1,885 s (31.4 min), 9,910 kernels | 188 s (3.1 min), 0 kernels |
| of it: LLM / flow / HiFT | 152 / 1,159 / 574 s | 2.4 / 165 / 21 s |
| of it: conv safety checks / weight preparation | 452 / 9.9 s | 23.5 / 9.2 s |
| `warmup_streaming()` | 779 s (13.0 min), 2,766 kernels | 150 s (2.5 min), 0 kernels |
| of it: streaming flow / streaming HiFT | 125 / 654 s | 124 / 26 s |

- **The buckets reproduce 09-28's figures** (1,831 s cold with 9,959 kernels, 195 s warm; "Stage 1 on chunked HiFT"
  above), the kernel count within 1 %. The masking changes none of the convs.
- **A cold streaming start is 44 minutes;** a warm one, 5.6.

**The cold first request** (`demo.py --warmup none`, one utterance, 121-127105-0003 at 8.52 s, fresh process):
- **On an empty kernel cache:** 563.4 s, RTF 66.1, 2,929 kernels compiled. HiFT took 386 s of it. The spec's 64.3 is
  close.
- **On a cache holding every binary it needs** (the day's earlier runs had compiled them): 17.4 s, RTF 2.04, nothing
  compiled. HiFT's first-sight conv checks take 10.7 s of it.
- **The spec's "RTF 32.5 on a cache filled by earlier warmed runs" doesn't reproduce as a figure.** It was 09-28's
  run above, on a cache that still lacked 706 of this request's binaries. What a filled cache gives depends on what
  compiled into it before.

## Streaming, stage A: offline, from fixed tokens (2026-09-29)

`tt/streaming.py` runs upstream's streaming schedule (`CosyVoice2Model.tts(stream=True)`, reproduced in its module
docstring) over a fixed token list, as if the LLM had finished.

**Per chunk:**
- The flow recomputes the prefix with chunk-causal masks and keeps the new frames. The final chunk runs the
  non-streaming flow, as upstream's does.
- Then HiFT runs with upstream's cache: 8 frames, source carry-over and the Hamming crossfade.
- The hop starts at 25, plus a pad bringing the prompt to a 25-token boundary, then 50, then 100. It restarts for
  every utterance; upstream's carries over between requests.

**The geometries stay finite:**
- **The flow** runs every chunk at a non-streaming flow bucket. The 3 look-ahead tokens sit right after the valid
  rows (`TtUpsampleConformerEncoder`'s `context_rows`), so the look-ahead layer runs at the bucket length, not at a
  length per chunk.
- **HiFT** runs the middle chunks at their exact 108 and 208 frames. The first chunk (50–98 frames) is padded to 128
  in front, and the final one to 128 or 256 at the end.
  - At 108, 128 and 208 frames, every k=11 resblock conv's TILE-prepared weight is wrong (1.0 up to 7.9e7, and inf).
  - The ROW_MAJOR-prepared candidate is right there (0.0035–0.0045), and the checks keep it ("Prepared conv weights"
    above).

**The gate** (`tests/e2e/test_streaming.py`) compares against upstream's own streaming run on the same tokens
(`scripts/streaming_reference.py`, reference venv):
- the tokens are TT's from the Stage 1 demo: six utterances, 23 chunks, 17 seams;
- the hop restarts at 25 on both sides.

| check | measured | gate |
|---|---|---|
| chunk plan (offsets, hops) | identical to upstream's, all six | equal |
| flow: each chunk's new mel vs upstream's streaming mel, relative L2 | 0.0085–0.0182 | ≤ 0.03 |
| control: upstream's non-streaming mel of the same frames vs its streaming mel | 0.022–0.130 | further away than ours, at every middle chunk |
| HiFT, mechanism (upstream's mel, F0 and noise per call): each chunk's emitted audio | PCC 0.99921–0.99985 | ≥ 0.999 |
| HiFT, mechanism: each seam (the crossfade ±40 ms) | PCC 0.99900–0.99989 | ≥ 0.998 |
| HiFT, the utterance's last 20 ms: level against upstream's (the masked final call, 2026-09-30) | within 0.2–0.5 dB | within 3 dB, no floor (D41) |
| HiFT, the final chunk's last 0.4 s: difference below the signal (masked) | 21–27 dB; 19.5–27.5 dB over six noise draws | at least 15 dB, no floor |
| HiFT, own F0: log-mel L1 vs upstream's streamed audio | 0.069–0.088 | ≤ 0.13 |

- **The final chunk's end.** Until 2026-09-30 the final call was padded with silence. Its last ~25 ms went silent
  in every case, and D38's tail criterion let that through its −50 dBFS floor (B28).
  - It is now masked ("Masked end padding in HiFT" above). The last 20 ms sit within 0.2–0.5 dB of upstream's, the
    last 0.4 s's difference 21–27 dB below the signal, and the final chunk's PCC before those 0.4 s is
    0.99921–0.99979.
  - Over the whole of 121-127105-0015's 13-token final chunk, PCC is 0.9970, and 0.9975 with the call at its exact
    length. The chunk ends near −71 dBFS, where PCC measures the port's own noise floor, so it is not gated.
- **The last 0.4 s's threshold is 15 dB** (2026-09-30; notes: `scripts/2026-09-30/tail_margin.py`). It was 20 dB,
  set from the six final chunks above. Over 36 final chunks (these six utterances under the suite's reference and
  D43's five noise draws), the masked call's margin is 19.5–27.5 dB, so at 20 dB one draw already failed.
  - **All five lowest are 121-127105-0015** (19.5–23.6 dB). Its last 0.4 s is near-silent (−71 dBFS), except for one
    20 ms burst at −58 dBFS, 0.38–0.36 s before the end. The burst holds 85–87 % of the window's signal energy.
  - **That burst sets the margin.** It also holds 81–94 % of the difference's energy, and in every draw the window's
    margin is the burst's own within 1 dB (19–24 dB).
    - Quieter frames, some with the difference only 13–19 dB below them, carry too little energy to move it.
    - The last 40 ms are at 22–24 dB.
  - **It is the port's own error on that burst, not the padding.** The same call unpadded, at its exact length
    (compiled for each length), scores 19.9–24.0 dB. The burst's difference is the same within about 1 dB (−78 dBFS
    both ways on the lowest draw).
  - **The headroom:** 15 dB is 4.5 dB below the lowest margin, and about 4.5 standard deviations below 0015's mean
    (21.6 ± 1.5 dB).
  - **What this criterion catches of the old silence padding:** 4 of the 6 utterances on its own (−1.9 to 10.9 dB).
    260-123286-0014 (17.8 dB) and 260-123440-0002 (23.4–24.8) pass it. The padding's effect reaches back 120–200 ms,
    and their last 0.4 s is dominated by louder speech before that. Their last 20 ms fail the level check by 75 and
    32 dB, so the end gate as a whole still fails the old padding on every utterance.
- **WER and SIM** (`scripts/eval_wer_sim.py`) of our offline-streamed audio (our flow, our HiFT, own F0), against
  upstream's streaming of the same tokens:
  - WER 1.36 % vs 0.68 %; SIM 95.83 vs 95.90.
  - The one extra word is Whisper appending "you" after the last word of 260-123440-0010. Otherwise the
    transcripts match upstream's word for word.
  - That was the silenced ending (B28). With the masked final call, none of 11 noise draws of that clip ends in
    "you" ("Masked end padding in HiFT" above).
  - For scale, the same tokens non-streamed (the Stage 1 demo) score 0.68 % and 95.87.

## Streaming, stage B: interleaved with the LLM (2026-09-29)

**How it runs** (`CosyVoice2TTNN.synthesize_stream`; `tt/streaming.py` `StreamSession`):
- The LLM's `generate(on_token=...)` hands each token to the session. When a chunk is due, its flow and HiFT run
  between two decode steps, while the decode trace is alive.
- So every streaming geometry is compiled and verified first, by `warmup_streaming()` after `warmup_buckets()`:
  - the streaming flow at all 17 buckets;
  - HiFT at 128 frames padded in front, 108 and 208, and 128 and 256 padded at the end, each with its conv checks.

  Nothing compiles or prepares weights under a live trace.
- `generate` releases the trace when it returns, before the final chunk. Nothing is alive after a call.
- HiFT's state between chunks stays on the host: 8 mel frames and 3,840 source and 3,840 output samples, about
  30 KB.
- The hop restarts at 25 for every segment.
- HiFT's noise comes from a generator of its own, so streaming leaves alone the global RNG that host-side RAS
  sampling draws from. A seeded streamed call samples the same tokens as the non-streamed one.

**The hang check came first** (this board runs KMD 2.9.0):
- the opt-in allocation tracker on the CFM traces: 6 passed;
- then the interleaved test itself under `TT_METAL_TRACE_ALLOC_TRACKING=1`, in its own process: passed, with no
  trace-allocation violation and no hang. The tracker slows every decode-trace replay about 40x, so that run's
  timings are not measurements.

**The test** (`tests/e2e/test_streaming.py::test_device_streaming_interleaved_with_llm`, greedy sampling), on
260-123286-0014:
- 180 tokens and 4 chunks, 3 of them ready while the LLM was still generating;
- the streamed tokens equal the batch tokens;
- no trace is alive afterwards;
- the streamed audio equals stage A's offline streaming of the same tokens with the same noise, bit for bit.

(Greedy decoding runs this sentence to 180 tokens, where RAS sampling gives 75.)

**The same test untracked, in the full suite:** first audio at 1.356 s, RTF 0.891. For the first chunk:
- 0.373 s of text normalization and LLM until its 28 tokens;
- flow 0.866 s, of which the CFM takes 0.685 s;
- HiFT 0.118 s.

The measurement proper follows ("Streaming, measured").

## Streaming, measured (2026-09-29)

**The run:** `demo/demo.py --stream` twice, each in a fresh process:
- first `warmup_buckets()` (186.1 s both times) and `warmup_streaming()` (149.2 s and 149.8 s);
- then the corpus's six distinct utterances, with RAS sampling, seed 1986.

The kernel cache held every kernel: 0 compiled in either run. The tokens equal the Stage 1 demo's. The board is the
second N150 (KMD 2.9.0, firmware 19.11.0.0). "First audio" runs from the call to the moment the first chunk's audio
is on the host; it is the time to first packet.

| utterance | audio s | tokens | chunks | first chunk tokens | first audio s, run 1 / 2 | RTF, run 1 / 2 |
|---|---|---|---|---|---|---|
| 121-127105-0003 | 8.52 | 213 | 4 | 32 | 1.455 / 1.418 | 0.806 / 0.812 |
| 121-127105-0015 | 3.80 | 95 | 3 | 32 | 1.365 / 1.479 | 1.057 / 1.122 |
| 121-127105-0024 | 13.88 | 347 | 5 | 32 | 1.439 / 1.475 | 0.835 / 0.810 |
| 260-123286-0014 | 3.00 | 75 | 2 | 25 | 1.398 / 1.399 | 0.973 / 0.945 |
| 260-123440-0002 | 12.68 | 317 | 5 | 25 | 1.413 / 1.336 | 0.819 / 0.787 |
| 260-123440-0010 | 8.08 | 202 | 4 | 25 | 1.403 / 1.441 | 0.847 / 0.850 |
| **corpus** (49.96 s) | | | | | **1.365–1.455 / 1.336–1.479** | aggregate **0.853 / 0.843**, worst **1.057 / 1.122** |

**The first chunk**, over both runs:
- 0.371–0.466 s until it starts: text normalization, the LLM's prefill, and its decode up to the chunk's tokens plus
  3 look-ahead tokens. The chunk is 25 tokens plus the prompt's padding to a multiple of 25: 32 for speaker 121.
- Its flow: 0.812–0.920 s, of which the CFM takes 0.674–0.731 s (10 Euler steps at 67–73 ms each, over the prompt
  plus the chunk).
- Its HiFT: 0.121–0.127 s.

**Against the targets** (recorded as `Misses()` in `tests/perf/gates.py`; since 2026-09-30 the streaming perf test
holds each inside its band):
- **Time to first packet < 500 ms: missed**, by about 3x. Without its flow, the first chunk would be ready at
  0.51–0.59 s (the time until it starts, plus HiFT), so even a free flow would miss. The flow is the lever.
- **Streaming RTF < 0.4: missed.** Every chunk reruns the flow over the whole prefix, as upstream does, and the final
  chunk runs it non-streaming over every token. The two short utterances (3.0 and 3.8 s) are the worst: they carry
  the ~1.4 s first chunk over the least audio.

**Speech quality** (`scripts/eval_wer_sim.py` on run 1's audio, against upstream's streaming of the same tokens,
stage A's reference):
- corpus WER 1.36 % (2 errors in 147 words) and similarity 95.85;
- upstream: 0.68 % and 95.90.

The extra error is Whisper appending "you" to 260-123440-0010, as it did on stage A's offline streaming of the same
tokens. It was the final call's silenced ending (B28). With the masked final call, over five noise draws: 0.68 % in
every draw and similarity 95.83 (95.81–95.87); upstream's streaming 0.68 % and 95.89 (95.87–95.91) ("WER and
similarity over five noise draws").

**Streaming without its warm-up is refused.** A third process ran one request (121-127105-0003) with `--warmup none`,
under `TT_METAL_TRACE_ALLOC_TRACKING=1`:
- It compiled 416 kernels, then failed at the first decode replay after the first chunk: `Found 1259 device
  buffer(s) still alive before trace replay. These will be corrupted on replay.`
- The first chunk's flow and HiFT allocated those buffers while the decode trace was alive, and they stay allocated.
  The tracker labels each with what allocated it:
  - 772 are host-to-device copies (`ttnn.to_device`): weights and constants moved to the device on first use;
  - 487 were created along with new programs, on program-cache misses (convs 110, halos 102, moves 42, matmuls 34,
    and others). The program cache keeps them.

  The tracker flags every buffer allocated under a live trace, because the replay writes to addresses that were
  free when the trace was captured. Untracked, the replay could have overwritten any of them silently.
- So `synthesize_stream` now raises unless `warmup_streaming()` has run, and `demo.py` refuses `--stream` without
  `--warmup buckets`.
- The interleaved test checks the refusal first. It passed again: 2 passed, 0 kernels compiled, first audio 1.373 s,
  RTF 0.890.
- Non-streaming requests are unaffected: `generate()` releases the trace before the flow and HiFT run.
- A cold streaming start is therefore the two warm-ups on an empty kernel cache. It is not measured yet.

## Streaming RTF above 1.0: which utterance, and why (Stage 3, 2026-09-30)

From the per-chunk records of the two masked-HiFT streaming runs ("Stage 1 and streaming re-run on the masked HiFT"
above; notes: `scripts/2026-09-30/rtf_breakdown.py`).

**A streamed utterance's wall time is the sum of its parts:** text and LLM, then each chunk's flow and HiFT, and
nothing else (at most 0.01 s). Every chunk runs between two decode steps, or after the LLM for the final chunk.

**Only 121-127105-0015 is above 1.0:** 1.121 and 1.103 (1.057 and 1.122 on 09-29). For its 3.80 s of audio:

| | run 1 | run 2 |
|---|---|---|
| text and LLM | 1.07 s | 1.05 s |
| three flows | 2.81 s (0.91 + 0.89 + 1.01), CFM 2.23 | 2.77 s (0.87 + 0.88 + 1.02), CFM 2.16 |
| three HiFT calls | 0.39 s | 0.37 s |
| wall | 4.26 s | 4.19 s |

- **A chunk's flow costs at least 0.82 s, however little audio it adds.**
  - Every chunk reruns the flow over the prompt and every token so far, as upstream does.
  - With these prompts (168 and 175 tokens), even the first chunk runs at the 256-token flow bucket (512 mel frames).
    There the CFM's 10 Euler steps take 68–74 ms each.
- **0015's tokens spill into a third chunk.**
  - Its 168-token prompt is padded to a multiple of 25, which makes the first hop 32.
  - After 32 + 50, 13 of its 95 tokens remain for the final chunk. That chunk is a full non-streaming flow over all
    263 tokens (bucket 320, 1.01 s) for 0.52 s of new audio.
- **The other short utterance stays under 1.0.** 260-123286-0014 (3.00 s, 75 tokens, a 175-token prompt needing no
  padding) splits into exactly 25 + 50. That is two flows, and RTF 0.957 and 0.905.
- **The long utterances sit at 0.79–0.84.** Their 100-token hops add 4 s of audio for a 1.13–1.70 s flow.

**The flow's cost per bucket**, both runs:

| flow bucket, tokens (mel frames) | chunks | flow s | of it, the CFM | CFM per Euler step | the flow outside the CFM |
|---|---|---|---|---|---|
| 256 (512) | 24 | 0.82–0.91 | 0.68–0.74 s | 68–74 ms | 0.14–0.19 s |
| 320 (640) | 2 | 1.01–1.02 | 0.78 s | 78 ms | 0.23–0.24 s |
| 384 (768) | 12 | 1.13–1.20 | 0.85–0.86 s | 85–86 ms | 0.27–0.34 s |
| 512 (1024) | 6 | 1.65–1.70 | 1.00–1.01 s | 100–101 ms | 0.64–0.70 s |
| 640 (1280) | 2 | 2.35–2.71 | 1.35–1.36 s | 135–136 ms | 1.00–1.36 s |

- **A CFM step's cost grows much more slowly than its length.** From 512 to 1,280 frames, 2.5 times the frames cost
  twice as much per step. Much of each step is fixed cost, and the next step (a profile of one Euler step) breaks
  it down.
- **The rest of the flow** grows faster: the Conformer encoder, the projections and the host transfers.

**Against the 0.4 target:** for every utterance, the flows alone take 0.47–0.74 of the audio's duration, before the
LLM and HiFT are counted. No schedule of the same chunks reaches 0.4. A chunk's flow has to get cheaper, through
fewer Euler steps or a cheaper step.

## One CFM Euler step, profiled (Stage 3, 2026-09-30)

**How** (notes: `scripts/2026-09-30/cfm_step_profile.py`, `cfm_profile_raw.py`):
- The flow is built as the pipeline builds it.
- The CFM's inputs are captured from real chunks:
  - 121-127105-0015's first chunk: bucket 256, 512 mel frames, 400 valid;
  - 121-127105-0024's 100-token hop and a later chunk: buckets 384 and 512;
  - 0015's final chunk: non-streaming, bucket 320.
- Per geometry: five eager solves timed; then each step split into its parts, with a device sync between them;
  then, for the streaming geometries, the same step traced and replayed.
- In a separate process under the device profiler (`python -m tracy -r`): one eager step at the first chunk's
  geometry, between signposts.

**Wall time per Euler step** (median of 50 steps; batch 2, for classifier-free guidance):

| geometry | mel frames | eager solve, 10 steps | eager step | of it, the host enqueueing the estimator | device busy after that | the step traced |
|---|---|---|---|---|---|---|
| first chunk, streaming, bucket 256 | 512 | 0.657–0.673 s | 64.7 ms | 62.6 ms | 0.07 ms | 49.2 ms |
| a 100-token hop, streaming, bucket 384 | 768 | 0.856–0.880 s | 85.4 ms | 66.6 ms | 16.0 ms | 80.5 ms |
| a later chunk, streaming, bucket 512 | 1,024 | 1.008–1.024 s | 102.2 ms | 72.9 ms | 25.7 ms | 94.9 ms |
| 0015's final chunk, non-streaming, bucket 320 | 640 | 0.774–0.782 s | 77.6 ms | 62.7 ms | 12.6 ms | not traceable (a padded non-streaming mask) |

- **The rest of a step takes 1.9–3.3 ms:** uploading x, the time embedding, downloading dphi, and the host's CFG blend
  and update.
- **At the first chunk's size, the step is host-bound.**
  - The host takes 62.6 ms to enqueue the estimator's ops, and the device finishes 0.07 ms after the last one.
  - Traced, with no host dispatch, the step takes 49.2 ms, and the solve 0.506 s against 0.66 s eager.
- **From 768 frames up, the device dominates:** traced 80.5 and 94.9 ms against eager 85.4 and 102.2 ms.
- **In the pipeline** the CFM call, which also uploads its conditioning, measured 68–74 ms per step at this size
  ("Streaming RTF above 1.0").

**On the device: one eager step at 512 frames, under the device profiler.**
- 1,158 ops.
- Device kernel time 47.8 ms. Firmware time is 59.0 ms, and the span 67.8 ms: the profiler slows dispatch, and the
  host took 71.3 ms between the signposts.

| op | calls | device kernel ms | what |
|---|---|---|---|
| reshape (`ReshapeViewDeviceOperation`) | 56 | 15.0 | merging the heads after attention (below) |
| matmul | 255 | 11.0 | QKV 3.8, the feed-forward's two 2.8 and 2.2, the output projection 1.5, the rest 0.7 |
| SDPA | 56 | 4.6 | 8 heads of 64, 512 × 512, with the streaming mask |
| elementwise binary | 275 | 4.1 | |
| transpose | 57 | 3.5 | 56 of them in the head merge |
| unary | 100 | 3.4 | 2.9 of it the GELU on the feed-forward's `[2, 512, 1024]` |
| layer norm | 141 | 2.5 | |
| create heads | 56 | 2.4 | the QKV split |
| conv, halo and their resharding | 155 | 1.2 | the resnet blocks' causal convs |

- **The head merge is 39 % of the step's device time: 18.5 of 47.8 ms.**
  - Each of the 56 transformer blocks transposes SDPA's `[2, 8, 512, 64]` to `[2, 512, 8, 64]`, then reshapes it to
    `[2, 512, 512]` (`tt/flow/decoder.py`).
  - In tile layout the 8 heads pad to a 32-row tile, so the reshape moves data: 0.27 ms per call.
  - `ttnn.experimental.nlp_concat_heads` does the same merge in one op, as the QKV side already does with the fused
    split. Not measured here.
- **The host takes about 54 µs to enqueue each op, against about 41 µs of device time per op.** At the first chunk's
  size, then, cutting device time alone does not shorten an eager step. The ops have to become fewer, or the step
  traced.
- **The CFM trace exists (`cfm_trace`) but is off in the pipeline.** A streaming chunk runs under the LLM's live
  decode trace, and the CFM keeps one trace at a time (`TtCausalConditionalCFM`'s docstring).

**For Stage 3:** at 10 Euler steps, the first chunk's CFM costs 0.66 s eager, or 0.51 s traced. The three levers are
the step count, the head merge and the trace. The step sweep measures the first.

## The Euler step sweep: 10, 8, 6 and 5 steps (Stage 3, 2026-09-30)

**How** (notes: `scripts/2026-09-30/steps_draws.py`, `ref_steps.py`, `phase_steps.sh`, `steps_summary.py`,
`steps_distance.py`):
- D43's protocol: five vocoder noise draws per utterance (seeds 1–5, the tokens fixed by LLM seed 1986), Stage 1 and
  streaming, TT and the reference, scored by `scripts/eval_draws.py`.
- The step count is the only change.
  - The port reads it from `tt/flow/flow.py`'s `N_TIMESTEPS`, and upstream from the `n_timesteps` its flow passes to
    `CausalConditionalCFM.forward`. The notes scripts set each one in their own process.
  - The PR's code and its 10 steps (upstream's) are unchanged.
  - The CFM's fixed starting noise and its cosine time schedule stay as they are.
- **TT:** one process, both warm-ups, then each step count in turn, with nothing else on the host. Its 10-step run
  reproduced D43's draws exactly: 60 of 60 wavs identical.
- **The reference:** upstream's Stage 1 on its own tokens, and upstream's streaming of TT's tokens, as in D43.
  - 10 steps is D43's draws.
  - Every run at 8, 6 and 5 steps sampled D43's tokens for its side and mode: 60 of 60, TT and reference.

**Quality:** corpus WER and SIM, mean (range) over the five draws:

| Euler steps | WER, every group | SIM, TT Stage 1 | SIM, reference Stage 1 | SIM, TT streaming | SIM, reference streaming |
|---|---|---|---|---|---|
| 10 | 0.68 % | 95.88 (95.84–95.92) | 95.22 (95.21–95.24) | 95.83 (95.81–95.87) | 95.89 (95.87–95.91) |
| 8 | 0.68 % | 95.95 (95.92–95.96) | 95.27 (95.26–95.28) | 95.91 (95.85–95.95) | 95.87 (95.85–95.89) |
| 6 | 0.68 % | 95.77 (95.76–95.80) | 95.09 (95.07–95.10) | 95.87 (95.84–95.91) | 95.86 (95.85–95.88) |
| 5 | 0.68 % | 95.91 (95.87–95.94) | 95.42 (95.41–95.43) | 95.98 (95.96–96.01) | 95.90 (95.88–95.92) |

- **WER is 0.68 % in every draw of every group.** No utterance's WER moves from its 10-step value, on either side,
  in either mode.
- **SIM stays within 0.2 of its 10-step value on both sides,** moving both ways across the groups.

**The audio does change.** Whole-utterance log-mel L1 (the stage A gate's measure), mean (max) over six utterances ×
five draws, between wavs of the same tokens and the same draw:

| Euler steps | TT Stage 1 vs its 10 steps | reference Stage 1 vs its 10 steps | TT streaming vs its 10 steps | reference streaming vs its 10 steps |
|---|---|---|---|---|
| 8 | 0.110 (0.143) | 0.101 (0.124) | 0.123 (0.169) | 0.114 (0.160) |
| 6 | 0.162 (0.220) | 0.140 (0.167) | 0.176 (0.250) | 0.169 (0.245) |
| 5 | 0.187 (0.240) | 0.167 (0.216) | 0.189 (0.222) | 0.181 (0.223) |

- **For scale:**
  - two noise draws of one side differ by 0.011–0.029;
  - TT's streaming differs from upstream's by 0.100–0.110 at every step count. The two draw their noise
    differently, so that includes a noise difference.
- **Upstream moves as much as the port does.** So this is the model's own sensitivity to the step count, not the
  port's.
- **At 5 steps each side moves 1.6–1.9 times the port's distance from upstream at 10.** WER and SIM do not
  register it, and neither measures naturalness. The wavs for listening are TT's and upstream's, seed 1, every step
  count (notes: `scripts/2026-09-30/steps_listening.sh`).

**Latency** (TT; 30 streamed utterances and 30 Stage 1 utterances per step count):

| Euler steps | first audio | first audio, worst per draw | streaming RTF, worst per draw | streaming RTF, aggregate | first chunk's flow | of it, the CFM | Stage 1 RTF, worst per draw | Stage 1 RTF, aggregate |
|---|---|---|---|---|---|---|---|---|
| 10 | 1.363–1.544 s | 1.450–1.544 s | 1.092–1.142 | 0.851–0.864 | 0.82–0.94 s | 0.68–0.75 s | 0.645–0.663 | 0.486–0.488 |
| 8 | 1.208–1.401 s | 1.321–1.401 s | 0.980–1.028 | 0.776–0.796 | 0.71–0.83 s | 0.54–0.61 s | 0.572–0.620 | 0.459–0.469 |
| 6 | 1.067–1.243 s | 1.186–1.243 s | 0.878–0.915 | 0.703–0.715 | 0.57–0.65 s | 0.41–0.45 s | 0.533–0.567 | 0.438–0.445 |
| 5 | 0.981–1.140 s | 1.107–1.140 s | 0.819–0.837 | 0.667–0.682 | 0.49–0.55 s | 0.34–0.36 s | 0.510–0.543 | 0.420–0.432 |

- **Each Euler step costs the first chunk about 70 ms.** The CFM falls in proportion (0.68–0.75 s at 10 steps,
  0.34–0.36 s at 5), and nothing else moves.
- **At 5 steps, first audio is still 0.98–1.14 s and the worst streaming RTF 0.82–0.84.** The step count alone reaches
  neither Stage 3 target (0.5 s, 0.4).
- **What is left of the first chunk at 5 steps:** the LLM's 0.37–0.47 s, the rest of the flow (0.15–0.20 s), the
  CFM's 0.35 s and HiFT's 0.12 s.

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

- **Start-up is 3.1 minutes with the kernels on disk and 31.4 without; the streaming set adds 2.5 and 13.0**
  ("Start-up and the cold first request, re-measured"). The flow is 88 % of the warm bucket start (165 of 188 s).
  Persisting the conv safety checks' verdicts (35 s of the warm start) won't be done (notes: D27).
- **tenstorrent/tt-metal#36487** (prepared conv weights wrong under DRAM slicing) is worked around by the per-geometry
  checks. Where the TILE-prepared weight is wrong, a ROW_MAJOR-prepared one usually isn't, and the checks keep it.
  A comment with our geometries is drafted, not posted.
- **Streaming misses both Stage 3 targets** ("Streaming, measured"):
  - first audio at 1.31–1.50 s against 0.5 s (four runs, 09-29 and 09-30);
  - a worst streaming RTF of 1.06–1.12 against 0.4.

  The first chunk's flow is the lever: its CFM alone takes 0.67–0.74 s. The streaming perf test holds both figures
  inside their recorded bands.
