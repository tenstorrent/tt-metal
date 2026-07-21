# tests/generalize

Generalized, reusable device sweeps decoupled from any specific model.

## matmul_sweep.py

A CLI tool that sweeps matmul program/memory/fidelity configurations for a given
GEMM shape and reports the fastest PCC-passing config per implementation.

You provide the GEMM dims `M, K, N` (`A[M,K] @ B[K,N] = C[M,N]`); the sweep owns
program-config generation, memory config, math fidelity, and dest/accumulation
modes. It derives valid per-core / block / subblock sizes from the shape + core
grid and skips combinations that violate the DST-register cap or divisibility, so
the run isn't just a wall of failures.

### Implementations (`--impl`, default: all)
| name        | op                                     | tunable program config |
|-------------|----------------------------------------|------------------------|
| `minmatmul` | `ttnn.experimental.minimal_matmul`     | `MinimalMatmulConfig` (M/K/N block tiles + subblock) |
| `matmul2d`  | `ttnn.matmul` (2D mcast)               | `MatmulMultiCoreReuseMultiCastProgramConfig` |
| `matmul1d`  | `ttnn.matmul` (1D mcast)               | `MatmulMultiCoreReuseMultiCast1DProgramConfig` |

### Swept axes
- program config: block sizes, per-core M/N, `in0_block_w`, subblock h/w, mcast/transpose
- `--dtypes` (bfloat16 / bfloat8_b / bfloat4_b)
- `--fidelities` (LoFi / HiFi2 / HiFi4)
- `--out-memcfgs` (dram / l1)
- `--fp32-dest` and `--packer-l1-acc` (dest/accumulation modes)

### Robustness
Every trial is isolated: input/output device tensors are always freed (even on
mid-trial failure) so a bad config never leaks L1/DRAM into the next one. Per-
config **OOM / TT_FATAL / bad-config** errors are caught, classified
(`OOM`/`SKIP`/`FATAL`), recorded, and the sweep continues. If an unexpected error
leaves the device unresponsive, a health probe aborts the run cleanly and still
writes partial results. `Ctrl-C` also writes partial results.

### Examples
```bash
# BGE-M3 MLP Wi shape, all impls, default sweep
python tests/generalize/matmul_sweep.py --M 98304 --K 1024 --N 4096

# One impl, custom minmatmul block grid, save CSV
python tests/generalize/matmul_sweep.py --M 98304 --K 4096 --N 1024 \
    --impl minmatmul --m-blocks 8 16 24 32 --k-blocks 8 16 32 \
    --csv tests/generalize/out/mlpwo.csv

# Pin dtype/fidelity, cap total configs tried
python tests/generalize/matmul_sweep.py --M 8192 --K 8192 --N 8192 \
    --dtypes bfloat8_b --fidelities LoFi --max-configs 40
```

Notes:
- `M, K, N` must be multiples of 32 (tile dim).
- Opens a single-chip `(1,1)` mesh (no fabric); set `TT_VISIBLE_DEVICES` to pick a board.
- Output: a `RESULT`/`OOM`/`SKIP`/`FATAL` line per config, a `BEST` line per impl,
  and (with `--csv`) a full results CSV.
