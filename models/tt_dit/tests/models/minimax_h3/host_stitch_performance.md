# MiniMax-H3 host spatial tile stitching

`stitch_tiles` assembles cropped tiles directly into one preallocated output
canvas. The previous implementation concatenated each row, then concatenated
the row canvases. Blend arithmetic, vertical-before-horizontal order and
original neighbour selection remain unchanged.

This change serves the shared host helper used by the encoder and host decoder.
Device stitching bypasses it. The eliminated row canvases total about 331 MiB
for FP32 output `[1, 3, 28, 768, 1344]`; this is a structural allocation
reduction, not a measured whole-process peak-RSS saving.

## Reproduction

From the repository root in the installed TTNN environment:

```bash
python -m pytest models/tt_dit/tests/models/minimax_h3/test_vae_minimax_h3.py \
  -k stitch_tiles_bitwise

MINIMAX_H3_RUN_HOST_STITCH_BENCH=1 python -m pytest -s \
  models/tt_dit/tests/models/minimax_h3/test_vae_minimax_h3.py \
  -k stitch_tiles_performance
```

These selected tests request no device fixture and load no model weights. The
benchmark emits raw timing samples and metadata as JSON under pytest's temporary
directory. Speed is reported, not asserted; output-byte equality is asserted.

## Target-host module results: 30 September 2026

Source base: `7665eaca1937790e993036832622d15fec69d704`, with the proposed
stitching change. Normal pytest collection on `f02cs02` passed all 28 correctness
cases. Coverage includes pixel and 48-channel encoder-latent geometry,
single tiles/rows/columns, a 4x7 production grid, large and unequal overlaps,
FP32/BF16 and contiguous/noncontiguous tiles. Output bytes, shape, dtype,
contiguity and input immutability are checked against the previous algorithm.

Module timing conditions: AMD EPYC 9124 16-Core Processor, Python 3.10.20,
Torch 2.11.0+cpu, pytest 9.0.3, eight Torch threads, seed 73, contiguous FP32
tiles, 4x7 grid at 1344x768. Each independent invocation used two warmup pairs
and ten measured pairs, alternating execution order. Timing includes allocation,
blending and assembly, and excludes input generation and device decoding.
The installed TTNN native runtime is from `0dea474c`; it is imported for normal
collection but performs no device work in these tests.

| Frames/chunk | Invocation | Original median | Preallocated median | Speedup |
| --- | --- | ---: | ---: | ---: |
| 28 | First | 178.980 ms | 100.169 ms | 1.787x |
| 28 | Repeat | 225.622 ms | 158.421 ms | 1.424x |
| 1 | First | 11.014 ms | 10.745 ms | 1.025x |
| 1 | Repeat | 11.705 ms | 12.049 ms | 0.971x |

All four benchmark cases passed output-byte equality. Raw samples and provenance
are retained in [host_stitch_benchmark_results.json](host_stitch_benchmark_results.json).
The large-chunk improvement repeats on this host, but timing varies and the
one-frame repeat regresses. No universal speedup is claimed.

## Earlier observations

Local AMD EPYC 7313P, Torch 2.8.0+cu128 on CPU, eight threads, same paired
benchmark: 28-frame medians were 192.299 -> 122.152 ms and 207.887 -> 134.979 ms;
one-frame medians were 12.511 -> 14.022 ms and 11.078 -> 11.413 ms. All outputs
matched bytewise. An earlier six-pair 28-frame probe regressed from 287.590 to
294.950 ms. These observations reinforce the workload and host dependence.

On 29 September, the same preallocation algorithm was tested in the decoder
caller on tt-metal `0dea474c` plus four existing integration fixes: four
Blackhole p150b cards, 1x4 Ring, 11x10 workers/card, firmware 19.11.0, checkpoint
`42ed227ee7df40d41602854ae760620d6eb651fe`. A fixed `[1, 24, 37, 48, 84]` latent
produced 124 frames at 1344x768, using 196 tile/chunk units and two units/card/wave.
Input, weights and readback settings were unchanged.

| Timing boundary | Original warm runs | Preallocated warm runs |
| --- | ---: | ---: |
| Complete VAE decode | 8.159729 / 8.168667 s | 7.847995 / 7.870545 s |
| Accumulated host stitching | 1.173603 / 1.005851 s | 0.613070 / 0.619256 s |

Mean complete-decode saving was 0.304928 s (3.7%). Repeated decoded tensors
passed `torch.equal`. This is historical decoder evidence on an older runtime,
not current-main whole-generation performance or fresh hardware coverage of
the shared-helper port. Current-source hardware decode and encoder parity
remain to be checked before marking the PR ready for review.
