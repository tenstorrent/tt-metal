# SDPA Single-Core Programming Example

Single-core Scaled Dot-Product Attention (SDPA) implementation using TT-Metalium. Processes Q, K, V matrices in chunks with online softmax normalization (FlashAttention-style).

## Setup

```bash
git checkout dnijemcevic/sdpa_benchmark

# Activate the Python venv
source python_env/bin/activate

# Verify env vars are set
echo $ARCH_NAME        # e.g. wormhole_b0, blackhole
echo $PYTHONPATH       # should include the tt-metal root
```

## Building

```bash
./build_metal.sh --build-programming-example
```

Binary output: `build/programming_examples/metal_example_sdpa_single_core`

## C++ Host Wrapper

### Benchmark mode (zero-filled data)

```bash
./build/programming_examples/metal_example_sdpa_single_core
```

Runs with hardcoded defaults: `Sq_chunk_t=7, Sk_chunk_t=16, head_dim_t=4, num_q_chunks=2, num_k_chunks=3, subblock_h=1, mm_throttle_level=0, exp_approx_mode=0`.

### Test mode (file-based I/O)

```bash
./build/programming_examples/metal_example_sdpa_single_core \
    --test <dir> \
    --Sq_chunk_t 7 --Sk_chunk_t 16 --head_dim_t 4 \
    --num_q_chunks 1 --num_k_chunks 5 \
    --subblock_h 1 --mm_throttle_level 0 --exp_approx_mode 0 \
    --padded_k_tiles 0 --kv_bf8b 0
```

Reads `<dir>/q.bin`, `k.bin`, `v.bin` (raw bfloat16), tilizes, runs the kernel, untilizes output, and writes `<dir>/device_output.bin`.

**File format:** Flat row-major arrays of `uint16` (bfloat16).

| File | Shape | Description |
|------|-------|-------------|
| `q.bin` | `[Sq, d]` | Query matrix |
| `k.bin` | `[Sk, d]` | Key matrix (reader transposes tile grid per chunk) |
| `v.bin` | `[Sk, d]` | Value matrix |
| `device_output.bin` | `[Sq, d]` | Device result (written by C++) |

Where `Sq = num_q_chunks * Sq_chunk_t * 32`, `Sk = num_k_chunks * Sk_chunk_t * 32`, `d = head_dim_t * 32`.

**CLI parameters:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test <dir>` | string | "" (benchmark) | Test directory with q.bin, k.bin, v.bin |
| `--Sq_chunk_t` | uint32 | 7 | Q sequence chunk tile height |
| `--Sk_chunk_t` | uint32 | 16 | K/V sequence chunk tile height |
| `--head_dim_t` | uint32 | 4 | Head dimension in tiles |
| `--num_q_chunks` | uint32 | 2 (bench) / 3 (test) | Number of Q chunks |
| `--num_k_chunks` | uint32 | 3 (bench) / 5 (test) | Number of K/V chunks |
| `--subblock_h` | uint32 | 1 | Subblock height (1 or 2) |
| `--mm_throttle_level` | uint32 | 0 | Matrix multiply throttle (0-5) |
| `--exp_approx_mode` | uint32 | 0 | Fast exponential approximation (0=off, 1=on) |
| `--padded_k_tiles` | uint32 | 0 | Padded K tiles in final chunk |
| `--kv_bf8b` | uint32 | 0 | Use bfloat8_b for K/V (0=bf16, 1=bfp8_b) |

**Padded K masking:** When `--padded_k_tiles N` is nonzero, the last N tiles of the last K chunk are treated as zero-padding. On the last K chunk, the compute kernel dispatches a reduced-path instantiation (`effective_Sk = Sk_chunk_t - N`) that skips all padded-tile computation — matmuls, softmax, reduce, and masking all operate on `effective_Sk` tiles only. K/V input files must still have the full padded shape.

**bfp8_b K/V:** When `--kv_bf8b 1`, K and V buffers use bfloat8_b data format (1 KB/tile vs 2 KB for bf16), halving K/V L1 footprint. Useful for large head dimensions (e.g., MLA with head_dim_t=16).

## Python Correctness Tests

Pytest-based tests that generate inputs, compute a PyTorch reference via `F.scaled_dot_product_attention`, run the C++ binary, and compare results (PCC, RMSE, max absolute error).

### Before running tests: reset the device

```bash
pkill -9 -f pytest || true
tt-smi -r $(tt-smi -ls 2>&1 | grep -oP '^\│ \K\d+' | head -1)
```

### Run all tests

```bash
pytest tt_metal/programming_examples/sdpa_single_core/generate_and_test_sdpa.py -v
```

### Main test cases

| ID | Q chunks | K/V chunks | Sq_chunk_t | Sk_chunk_t | head_dim_t | sbh | Padded | kv_bf8b | Data |
|----|----------|------------|------------|------------|------------|-----|--------|---------|------|
| `1q_1k-zeros-sk16` | 1 | 1 | 7 | 16 | 4 | 1 | 0 | 0 | All zeros |
| `1q_1k-ones-sk16` | 1 | 1 | 7 | 16 | 4 | 1 | 0 | 0 | All ones |
| `1q_1k-random-sk16` | 1 | 1 | 7 | 16 | 4 | 1 | 0 | 0 | `fa_rand` |
| `1q_5k-random-sk16` | 1 | 5 | 7 | 16 | 4 | 1 | 0 | 0 | `fa_rand` |
| `3q_5k-random-sk16` | 3 | 5 | 7 | 16 | 4 | 1 | 0 | 0 | `fa_rand` |
| `1q_1k-random-sk8` | 1 | 1 | 7 | 8 | 4 | 1 | 0 | 0 | `fa_rand` |
| `1q_5k-random-sk8` | 1 | 5 | 7 | 8 | 4 | 1 | 0 | 0 | `fa_rand` |
| `3q_5k-random-sk8` | 3 | 5 | 7 | 8 | 4 | 1 | 0 | 0 | `fa_rand` |
| `1q_5k-random-sk16-pad4` | 1 | 5 | 7 | 16 | 4 | 1 | 4 | 0 | `fa_rand` |
| `1q_5k-random-sk8-pad2` | 1 | 5 | 7 | 8 | 4 | 1 | 2 | 0 | `fa_rand` |
| `3q_5k-random-sk16-pad8` | 3 | 5 | 7 | 16 | 4 | 1 | 8 | 0 | `fa_rand` |
| `1q_1k-random-sk4-sbh2` | 1 | 1 | 4 | 4 | 4 | 2 | 0 | 0 | `fa_rand` |
| `1q_5k-random-sk4-sbh2` | 1 | 5 | 4 | 4 | 4 | 2 | 0 | 0 | `fa_rand` |
| `1q_5k-random-sk8-sbh2` | 1 | 5 | 4 | 8 | 4 | 2 | 0 | 0 | `fa_rand` |
| `3q_5k-random-sk4-sbh2-pad1` | 3 | 5 | 4 | 4 | 4 | 2 | 1 | 0 | `fa_rand` |
| `3q_19k-random-sk16-pad8-sq9` | 3 | 19 | 9 | 16 | 4 | 1 | 8 | 0 | `fa_rand` |
| `3q_5k-random-sk16-pad6` | 3 | 5 | 7 | 16 | 4 | 1 | 6 | 0 | `fa_rand` |
| `1q_5k-random-sk16-pad6` | 1 | 5 | 7 | 16 | 4 | 1 | 6 | 0 | `fa_rand` |
| `1q_1k-random-sk16-pad8` | 1 | 1 | 7 | 16 | 4 | 1 | 8 | 0 | `fa_rand` |
| `1q_2k-random-sk16-pad8` | 1 | 2 | 7 | 16 | 4 | 1 | 8 | 0 | `fa_rand` |

### Additional tests

| Test | Description |
|------|-------------|
| `test_padding_pcc_impact[pad0..pad15]` | Sweeps padding levels (0, 1, 2, 4, 8, 12, 15) on Sq=9, Sk=16, 1q, 3k |
| `test_padding_pcc_correlation` | Verifies PCC stays above threshold across all padding levels |
| `test_mla_vDHt16` | MLA config: head_dim_t=16, sbh=2, sk=8, kv_bf8b=1 |
| `test_mla_vDHt4` | MLA config: head_dim_t=4, sbh=2, sk=8 |
| `test_padded_s160_k512_sbh1` | Large padded config: Sq=160, Sk=512, sbh=1 |
| `test_padded_s160_k512_sbh2` | Large padded config: Sq=160, Sk=512, sbh=2 |

### Run a single test

```bash
pytest generate_and_test_sdpa.py -v -k "1q_1k-random-sk16"
```

### Generate permanent input folders for IDE debugging

```bash
pytest tt_metal/programming_examples/sdpa_single_core/generate_and_test_sdpa.py --save-inputs
```

Creates `test_inputs/<test_id>/` with:
- `q.bin`, `k.bin`, `v.bin` -- bfloat16 inputs
- `ref_output.bin` -- float32 PyTorch reference
- `cmd.txt` -- full C++ command line and `launch.json` args

Then point your IDE launch configuration at the generated directory:

```jsonc
// launch.json
{
    "program": "${workspaceFolder}/build/programming_examples/metal_example_sdpa_single_core",
    "args": [
        "--test", "${workspaceFolder}/tt_metal/programming_examples/sdpa_single_core/test_inputs/1q_5k-random-sk16",
        "--Sq_chunk_t", "7", "--Sk_chunk_t", "16", "--head_dim_t", "4",
        "--num_q_chunks", "1", "--num_k_chunks", "5",
        "--subblock_h", "1", "--mm_throttle_level", "0", "--exp_approx_mode", "0"
    ]
}
```

Switch test cases by changing the `--test` path and matching `--num_q_chunks`/`--num_k_chunks`. Each `cmd.txt` has the exact args to copy.

To regenerate inputs (e.g. after changing seed or data generation):

```bash
pytest tt_metal/programming_examples/sdpa_single_core/generate_and_test_sdpa.py --save-inputs
```

## Key Files

| File | Role |
|------|------|
| `kernels/compute/sdpa.cpp` | Compute kernel (all SDPA math) |
| `kernels/dataflow/reader.cpp` | Reader: Q/K/V from DRAM to L1 CBs |
| `kernels/dataflow/writer.cpp` | Writer: helper tiles + normalized output to DRAM |
| `sdpa_single_core.cpp` | Host code (CB setup, kernel dispatch) |
| `generate_and_test_sdpa.py` | Pytest harness (generates inputs, runs binary, compares) |
| `SDPA_kernel_analysis.md` | Full architecture and data flow documentation |
| `kernels/skip_padding.md` | Feasibility study for skip-padding optimization |
| `kernels/skip_padding_plan.md` | Implementation plan for Approach B (dual-path) |

## Important: After Editing Kernels

Kernels are JIT-compiled from **installed** copies, not the source tree. After editing kernel `.cpp` files, you must rebuild:

```bash
./build_metal.sh --build-programming-example
```

This re-installs to `build/share/tenstorrent/kernels/sdpa_single_core/`. If you see stale behavior, also clear the JIT cache:

```bash
rm -rf built/tt-metal-cache*
```
