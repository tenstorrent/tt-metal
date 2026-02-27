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
    --padded_k_tiles 0
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

**CLI parameter defaults** (when omitted): `Sq_chunk_t=7, Sk_chunk_t=16, head_dim_t=4, subblock_h=1, mm_throttle_level=0, exp_approx_mode=0, padded_k_tiles=0`. `num_q_chunks` and `num_k_chunks` default to 2/3 (benchmark mode) or 3/5 (test mode).

**Padded K masking:** When `--padded_k_tiles N` is nonzero, the last N tiles of the last K chunk are treated as zero-padding and masked with -inf before softmax. K/V input files must still have the full padded shape.

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

| ID | Q chunks | K/V chunks | Sq_chunk_t | Sk_chunk_t | sbh | Padded | Data |
|----|----------|------------|------------|------------|-----|--------|------|
| `1q_1k-zeros-sk16` | 1 | 1 | 7 | 16 | 1 | 0 | All zeros |
| `1q_1k-ones-sk16` | 1 | 1 | 7 | 16 | 1 | 0 | All ones |
| `1q_1k-random-sk16` | 1 | 1 | 7 | 16 | 1 | 0 | `fa_rand` |
| `1q_5k-random-sk16` | 1 | 5 | 7 | 16 | 1 | 0 | `fa_rand` |
| `3q_5k-random-sk16` | 3 | 5 | 7 | 16 | 1 | 0 | `fa_rand` |
| `1q_1k-random-sk8` | 1 | 1 | 7 | 8 | 1 | 0 | `fa_rand` |
| `1q_5k-random-sk8` | 1 | 5 | 7 | 8 | 1 | 0 | `fa_rand` |
| `3q_5k-random-sk8` | 3 | 5 | 7 | 8 | 1 | 0 | `fa_rand` |
| `1q_5k-random-sk16-pad4` | 1 | 5 | 7 | 16 | 1 | 4 | `fa_rand` |
| `1q_5k-random-sk8-pad2` | 1 | 5 | 7 | 8 | 1 | 2 | `fa_rand` |
| `3q_5k-random-sk16-pad8` | 3 | 5 | 7 | 16 | 1 | 8 | `fa_rand` |
| `1q_1k-random-sk4-sbh2` | 1 | 1 | 4 | 4 | 2 | 0 | `fa_rand` |
| `1q_5k-random-sk4-sbh2` | 1 | 5 | 4 | 4 | 2 | 0 | `fa_rand` |
| `1q_5k-random-sk8-sbh2` | 1 | 5 | 4 | 8 | 2 | 0 | `fa_rand` |
| `3q_5k-random-sk4-sbh2-pad1` | 3 | 5 | 4 | 4 | 2 | 1 | `fa_rand` |
| `3q_19k-random-sk16-pad8-sq9` | 3 | 19 | 9 | 16 | 1 | 8 | `fa_rand` |

### Padding PCC impact tests

Additional tests that sweep padding levels (0, 1, 2, 4, 8, 12, 15 tiles) on a fixed config (Sq_chunk_t=9, Sk_chunk_t=16, head_dim_t=4, 1 Q chunk, 3 K chunks) and verify PCC stays above threshold. `test_padding_pcc_correlation` also reports correlation between padding amount and PCC degradation.

```bash
# Run individual padding level tests
pytest generate_and_test_sdpa.py -v -k "test_padding_pcc_impact"

# Run the sweep/correlation test
pytest generate_and_test_sdpa.py -v -k "test_padding_pcc_correlation"
```

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

## Important: After Editing Kernels

Kernels are JIT-compiled from **installed** copies, not the source tree. After editing kernel `.cpp` files, you must rebuild:

```bash
./build_metal.sh --build-programming-example
```

This re-installs to `build/share/tenstorrent/kernels/sdpa_single_core/`. If you see stale behavior, also clear the JIT cache:

```bash
rm -rf built/tt-metal-cache*
```
