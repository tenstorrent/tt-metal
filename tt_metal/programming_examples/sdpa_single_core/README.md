# SDPA Single-Core Programming Example

Single-core Scaled Dot-Product Attention (SDPA) implementation using TT-Metalium. Processes Q, K, V matrices in chunks with online softmax normalization (FlashAttention-style).

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

Runs with hardcoded defaults: `Sq_chunk_t=7, Sk_chunk_t=16, head_dim_t=4, num_q_chunks=1, num_k_chunks=5, subblock_h=1, mm_throttle_level=0, exp_approx_mode=0`.

### Test mode (file-based I/O)

```bash
./build/programming_examples/metal_example_sdpa_single_core \
    --test <dir> \
    --Sq_chunk_t 7 --Sk_chunk_t 16 --head_dim_t 4 \
    --num_q_chunks 1 --num_k_chunks 5 \
    --subblock_h 1 --mm_throttle_level 0 --exp_approx_mode 0
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

**CLI parameter defaults** (when omitted): `Sq_chunk_t=7, Sk_chunk_t=16, head_dim_t=4, num_q_chunks=3, num_k_chunks=5, subblock_h=1, mm_throttle_level=0, exp_approx_mode=0`.

## Python Correctness Tests

Pytest-based tests that generate inputs, compute a PyTorch reference via `F.scaled_dot_product_attention`, run the C++ binary, and compare results (PCC, RMSE, max absolute error).

### Run all tests

```bash
pytest tt_metal/programming_examples/sdpa_single_core/generate_and_test_sdpa.py -v
```

### Test cases

| ID | Q chunks | K/V chunks | Data |
|----|----------|------------|------|
| `1q_1k-zeros` | 1 | 1 | All zeros |
| `1q_1k-ones` | 1 | 1 | All ones |
| `1q_1k-random` | 1 | 1 | `fa_rand` (FlashAttention-style) |
| `1q_5k-random` | 1 | 5 | `fa_rand` |
| `3q_5k-random` | 3 | 5 | `fa_rand` |

### Run a single test

```bash
pytest generate_and_test_sdpa.py -v -k "1q_1k-random"
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
        "--test", "${workspaceFolder}/tt_metal/programming_examples/sdpa_single_core/test_inputs/1q_5k-random",
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
