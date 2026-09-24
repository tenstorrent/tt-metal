# Prefill KV-cache accuracy for #57454

[#57454: [gemma4] Explicit matmul blocking and block-sharded RMSNorm for prefill](https://github.com/tenstorrent/tt-metal/pull/57454)
speeds up Gemma4-31B prefill by changing three call sites: the RMSNorm, the MLP projections and the attention projections.
This page summarises how those changes affect the prefill KV cache against the GPU reference, measured with
`test_prefill_migration[mock-256k]` on a BH Galaxy 8x4 (CP8 / TP4). The gate is a minimum per-head PCC of 0.91.

## TLDR
- As first posted, #57454 missed the 256k gate by about 0.001 (0.9083 at chunk 8192, 0.9094 at 2048). Base passes by
  about the same margin (0.9108 / 0.9106), so this gate is very tight.
- The cause was the MLP's explicit blocking accumulating in bf16. Layers 0-20 were already more accurate than base; the
  error only drifted in the deep layers.
- Math fidelity is not the cause. An explicit program config (or a core grid) makes `ttnn.linear` default to LoFi, but
  putting the MLP back to HiFi2 with bf16 accumulation made it worse. fp32 accumulation fixes it, even at LoFi.
- #57454 now accumulates the MLP in fp32 and uses the explicit MLP blocking only up to chunk 4096 (commit `993720c3d3c`).
  At chunk 8192 it keeps the default MLP config, which is as fast there.
- Result: it passes at all three chunk sizes with more margin than base, and it stays about as fast as first posted.

## Before and after (256k context, combined branch with #56862 and #57383)

| | chunk 2048 | chunk 4096 | chunk 8192 |
|---|---|---|---|
| Base: min PCC | 0.9106 | not run | 0.9108 |
| #57454 as first posted: min PCC | 0.9094 (fail) | not run | 0.9083 (fail) |
| #57454 now: min PCC | 0.9131 | 0.9121 | 0.9118 |
| Base: first chunk / 256k prefill | 126.1 ms / 28.03 s | 164.2 ms / 16.59 s | 207.2 ms / 12.84 s |
| #57454 now: first chunk / 256k prefill | 92.4 ms / 23.64 s | 121.6 ms / 14.08 s | 200.8 ms / 12.66 s |

The "#57454 now" values were measured with equivalent local switches just before the change was committed. A rerun on
the committed code is in progress, and this table will be updated with it.

## More detail
- [DETAILS.md](DETAILS.md): the test, how to read the metrics, each finding, and the perf cost of every variant tried.
- [RUNS.md](RUNS.md): all 42 runs (8k and 256k context), with flags and tree sha.
- [per_layer.csv](per_layer.csv): per-layer PCC and relative RMSE for every run.
- Raw logs: https://gist.github.com/kmabeeTT/edde8493f9ff0d889185be0742705821
