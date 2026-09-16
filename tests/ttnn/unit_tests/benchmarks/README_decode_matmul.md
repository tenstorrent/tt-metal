# Decode-shaped matmul benchmarks (M = 32, single Blackhole chip)

Two manual-only, trace-timed `ttnn.matmul` benchmarks. Both skip unless `TTNN_RUN_GEMM_FLOPS_BENCHMARK=1`.

- `test_dram_sharded_decode_matmul.py` — weights DRAM width-sharded across the 8 banks (GEMM from DRAM).
- `test_l1_ring_matmul.py` — weights, activation and output L1 width-sharded on one core grid, ring gather. Not a GEMM-from-DRAM number.

## Run

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 pytest tests/ttnn/unit_tests/benchmarks/test_dram_sharded_decode_matmul.py --timeout 0
TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 pytest tests/ttnn/unit_tests/benchmarks/test_l1_ring_matmul.py --timeout 0
```

Results are logged per case and appended to `generated/dram_sharded_decode_matmul.csv` and `generated/l1_ring_matmul.csv`.

## Results

Blackhole p300c (one chip, 11x10 grid, 8 DRAM banks), tt-metal `main` at `df15dfd17d5`, 2026-09-16. Trace µs/op, best of 3 replays of 20 ops. DRAM BW util = bytes moved / time / 512 GB/s. PCC vs torch fp32.

| cell | DRAM-sharded decode matmul | grid | pcc | DRAM BW util | L1 width-sharded ring | grid | pcc | DRAM BW util |
|---|---|---|---|---|---|---|---|---|
| BF16 32x2048x2048 | 26.9 | 8x4 | 0.9999 | 63% | 7.4 | 8x4 | 0.9996 | n/a (228% equiv.) |
| BF16 32x4096x4096 | 70.8 | 8x4 | 0.9999 | 94% | 13.4 | 8x8 | 0.9990 | n/a (497% equiv.) |
| FP8 32x2048x2048 | 25.2 | 8x4 | 0.9998 | 36% | 7.4 | 8x4 | 0.9997 | n/a (121% equiv.) |
| FP8 32x4096x4096 | 40.9 | 8x4 | 0.9998 | 86% | 9.6 | 8x4 | 0.9994 | n/a (368% equiv.) |

The ring reads nothing from DRAM; "equiv." is the bandwidth DRAM would need to supply to match that time.
