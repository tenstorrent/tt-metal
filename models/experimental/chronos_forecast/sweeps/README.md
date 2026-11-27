# Sweeps

Standalone scripts I used to find and check the optimizations in the model. They are not
pytest tests. Run them from the repo root with `PYTHONPATH=.` on a single Blackhole chip,
for example:

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/sweeps/sweep_matmul_l1.py qkv wo --base
```

Sweeps try many settings and pick a winner. Benches time one thing or compare two.

## Sweeps

| Script | What it does |
| --- | --- |
| `sweep_matmul_l1.py` | Tries matmul program configs for the encoder linears (QKV, Wo, Wv.Wo, FF-up, FF-down) at one L1 chunk (64 series x 160 tokens), with the performance dtypes and fidelities. It compares the model's current config against 2D mcast, transposed 2D, 1D in1-mcast and minimal_matmul, checks PCC, and times each under trace. Pass `--base` to time only what `program_configs.linear` picks. |
| `sweep_sdpa.py` | Tries SDPA chunk sizes and dtypes for time attention (B=1024, 12 heads, 133 tokens) and checks PCC against a reference. |
| `sweep_l1_chunk.py` | Runs the full paper-shape forward under trace replay for each precision and L1 chunk size, to see which chunk size is fastest. |

## Benches

| Script | What it does |
| --- | --- |
| `bench_encoder_ops.py` | Times the non-matmul encoder ops at one L1 chunk under trace: RoPE, add, RMSNorm, head split, fused QKV+RoPE and concat. It compares the model-local `ops/` versions with stock ttnn. Pass op names to run only some. |
| `bench_rope.py` | Compares `rotary_embedding` (HF rotate_half) with `rotary_embedding_llama` (interleaved). |
| `bench_sharded.py` | Compares L1-interleaved (11x10 grid) with block-sharded (8x10 grid) for RMSNorm, add and the QKV matmul on one chunk. |
| `bench_group_layout.py` | Times the layout choices on the grouped-series path. It has three sections, `permute`, `chunk` and `sdpa`, and runs all of them by default. `permute` compares tiled permute and transpose with a row-major round trip at the full batch. `chunk` repeats the permute at one L1 chunk shape, with bf16 vs bf8 input and DRAM vs L1 placement. `sdpa` times block SDPA against block size. |
| `time_grouped.py` | Times an eager forward with grouped series and checks PCC against the reference. Takes batch size, group spec (`4` or `mixed`), precision and `l1`. |

## Profiling helpers

| Script | What it does |
| --- | --- |
| `profile_small_batch.py` | Runs a signposted forward at a small batch so Tracy can capture it. The profiler buffers overflow at B=1024 when chunked. Takes batch, precision and chunk (or `none`). |
| `summarize.py` | Reads a profiler ops CSV, keeps the ops between two signposts, and totals time per op and shape. Use it to see where time goes. |
