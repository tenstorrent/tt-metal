# wan2_2 tests: Fused AllGather+Matmul and pytest

## How the fused op replaces baseline and shows in the profiler

**Replacement is in Python, not in the backend.** There is no automatic “fusion” of two ops into one; the model chooses which path to run.

- **Baseline path:** `all_gather_async(act, ...)` then `minimal_matmul(gathered_act, weight, ...)`. Two separate device ops → two entries in the trace (e.g. `AllGatherAsync` and `MinimalMatmulDeviceOperation`).
- **Fused path:** When `USE_FUSED_AG_MM_PREFILL=1` and conditions are met (e.g. `seq_len >= 4096`, `batch_size == 1`), the model calls a **single** API: `ttnn.experimental.all_gather_minimal_matmul_async(act, weight, ...)`. That dispatches one device op.

**Profiler:** The C++ profiler records the **device operation type name** for each launched op (see `ttnn/api/tools/profiler/op_profiler.hpp`: `get_type_name<device_operation_t>()`). So:

- Baseline run: two ops → e.g. `AllGatherAsync` and `MinimalMatmulDeviceOperation` (or similar C++ type names).
- Fused run: one op → e.g. `ttnn::experimental::prim::AllGatherMinimalMatmulAsyncOp` (or the demangled name like `AllGatherMinimalMatmulAsyncOp`).

So the fused run shows as **one op** in the profiler with that single type name; there is no separate “AllGather” + “Matmul” in the trace when the fused path is used. To show a friendly name (e.g. “FusedAllGatherMatmul” or “AllGatherAsync_ffn_2”) in parsed reports or perf targets, add a mapping from that C++ op name in:

- `scripts/parse_profiler_report.py` (e.g. when building `OP_NAME`), and/or
- Perf test targets (e.g. `models/demos/llama3_70b_galaxy/tests/perf_targets/prefill_*.json` and any `mapping_op_code_to_name` in `test_prefill_device_perf.py`).

The model wiring is in `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` (branch on `use_fused_ag_mm_prefill` and `seq_len >= 4096`) and `llama_ccl.py` (semaphores, persistent buffers for fused path).

---

## Why pytest doesn’t work when run “individually”

Two common causes:

### 1. Missing fixtures (`M_block_size`, `K_block_size`, etc.)

Fixtures like `M_block_size`, `K_block_size`, `N_block_size`, `subblock_h`, `subblock_w` are defined in **this directory’s** `conftest.py`. Pytest loads `conftest.py` only from:

- The directory of the **collected test file**, and
- Its parent directories up to the root.

So:

- **Works:** Run from **repo root** and pass the **full path** to the test file so that this directory is part of the collection path:
  ```bash
  cd /path/to/tt-metal
  pytest models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py -k "test_linear and 4k4k4k and wh8x4links1 and fused and check" -v
  ```
- **Fails:** Running from another directory (e.g. `tests/`) or invoking the test by a path that doesn’t include `wan2_2` (e.g. `pytest test_all_gather_minimal_matmul_async.py`) so that `models/tt_dit/tests/models/wan2_2/conftest.py` is never loaded → fixtures not found.

### 2. Root conftest and indirect fixtures

`mesh_device` and `device_params` are **indirect** fixtures provided by the **root** `conftest.py`. Pytest must see the repo root so that:

- Root `conftest.py` is loaded (defines `mesh_device`, `device_params`, etc.).
- Parametrization (e.g. `(8, 4)`, `wh8x4links1`) is applied correctly.

So run pytest from the **tt-metal repo root** and use the **full path** to the test file (or a path that includes `wan2_2` and the root), e.g.:

```bash
cd /path/to/tt-metal
pytest models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py -v
```

If you run from a different clone (e.g. `metal/tt-metal`), that clone must have the same `conftest.py` under `models/tt_dit/tests/models/wan2_2/` (with `M_block_size`, etc.); otherwise you get “fixture 'M_block_size' not found”.

---

## 8×4 wormhole and L1

On wormhole with an 8×4 mesh, the fused fabric config can exceed L1 (`memory_map_end_address` > `l1_end_address`). In **teja/tt-metal** this is handled by:

- A **skip** at the start of `test_linear` for fused + 8×4 + `WORMHOLE_B0`.
- A **skip** in the root `mesh_device` fixture for `(8, 4)` + `wormhole_b0` so the device is never opened for that config.

So the corresponding tests are skipped instead of failing with a TT_FATAL.
