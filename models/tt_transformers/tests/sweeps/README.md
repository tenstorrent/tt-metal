# tt_transformers op program-config sweeps

Standalone CLI tools that sweep the program configuration (core grid, block
sizes, memory layout, math fidelity, dtype) of individual decode ops, time each
candidate on device, and report the fastest passing config. Useful for tuning a
new model shape or re-tuning an op on a new board.

All sweeps take the model/op shape as arguments (with `--preset` shortcuts for
common models) so they are not tied to a single model.

## Device-kernel timing

The op sweeps rank by **device kernel duration** captured from the profiler
(`ReadDeviceProfiler` + `get_latest_programs_perf_data`). This requires a
profiler build and the three env vars below; without them, rows fall back to
host wall-clock (dispatch-dominated) and are flagged `src=host` — do not use
those for ranking.

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) MESH_DEVICE=P150
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
```

## Files

| File | What it sweeps |
|------|----------------|
| `sweep_common.py` | Shared device-open / profiler-capture / CSV helpers (imported by the op sweeps). |
| `sweep_create_heads.py` | `nlp_create_qkv_heads_decode`: input shard grid, `overlap_qk_coregrid`, input/output memory layout. |
| `sweep_sdpa_decode.py` | `scaled_dot_product_attention_decode`: compute grid, q/k chunk sizes, `exp_approx_mode`, fidelity, KV/output buffer type. |
| `sweep_llama_mm_v2.py` | Decode matmuls (QKV/WO/FF1_FF3/FF2/LMHEAD): core grid, `in0_block_w`, `per_core_N`, factory (DRAM-sharded vs 1D-mcast), fidelity, dtype, output memcfg. Reports GB/s and DRAM-bandwidth utilisation. |
| `matmul_sweep.py` | General GEMM sweep for arbitrary `--M/--K/--N` across `minimal_matmul`, 2D-mcast and 1D-mcast, with PCC checked against a torch reference. |

## Examples

```bash
# Attention op sweeps (Llama-3.1-8B is the default shape)
python models/tt_transformers/tests/sweeps/sweep_create_heads.py --csv ch.csv
python models/tt_transformers/tests/sweeps/sweep_sdpa_decode.py --kv-len 1024 --sweep-chunks --sweep-fid --csv sdpa.csv

# Another model via preset (see PRESETS in each file: llama3-8b/70b/1b/3b, mistral-7b, qwen2-7b)
python models/tt_transformers/tests/sweeps/sweep_sdpa_decode.py --preset qwen2-7b --kv-len 2048

# Decode matmul sweep
python models/tt_transformers/tests/sweeps/sweep_llama_mm_v2.py --shapes FF2 QKV WO --factory both --csv mm.csv

# Arbitrary GEMM
python models/tt_transformers/tests/sweeps/matmul_sweep.py --M 32 --K 4096 --N 14336
```

Each sweep writes a CSV (one row per config, including failures with the reason
in a `note`/`status` column) and prints the BEST (fastest passing) config.
