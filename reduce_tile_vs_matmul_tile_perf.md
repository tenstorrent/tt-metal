# reduce_tile vs matmul_tile Performance Analysis

## Setup

- **Machine:** WH (wormhole_b0)
- **Test kernel:** `rmsnorm_pre_allgather.cpp` via `test_distributed_layernorm_pre_allgather.py`
- **Input policy:** BulkWaitBulkPop (all tiles pre-loaded, no dataflow reader stalls)
- **Shape:** (1, 1, 32, 8192), n_devices=4, rmsnorm
- **Reduce:** REDUCE_ROW, PoolType::AVG
- **Measurement:** DeviceZoneScopedN per reduce_tile/matmul_tile call, TRISC_1 median (excl first call)
- **Matmul fidelity:** hardcoded HiFi4 via `REDUCE_MATMUL_FIDELITY` (ignores user-specified fidelity)

## Results

### BF16 input, fp32acc=OFF

| Path | Fidelity | TRISC_0 (unpack) | TRISC_1 (math) | Kernel total (ns) |
|------|----------|-----------------|---------------|-------------------|
| matmul | HiFi4 | 55 | 57 | 37,572 |
| matmul | HiFi2 | 55 | 56 | 37,684 |
| matmul | LoFi | 55 | 56 | 37,563 |
| reduce_tile | HiFi4 | 86 | 97 | 41,159 |
| reduce_tile | HiFi2 | 57 | 61 | 39,180 |
| reduce_tile | LoFi | 57 | 61 | 38,949 |

### BF16 input, fp32acc=ON

No change in per-tile cycles vs fp32acc=OFF (same 57/97 for matmul/reduce at HiFi4).

### FP32 input, fp32acc=OFF

| Path | Fidelity | TRISC_0 (unpack) | TRISC_1 (math) | Kernel total (ns) |
|------|----------|-----------------|---------------|-------------------|
| matmul | HiFi4 | 94 | 103 | 47,611 |
| matmul | HiFi2 | 93 | 103 | 47,515 |
| matmul | LoFi | 93 | 103 | 47,558 |
| reduce_tile | HiFi4 | 86 | 97 | 47,551 |
| reduce_tile | HiFi2 | 59 | 61 | 45,009 |
| reduce_tile | LoFi | 59 | 63 | 45,373 |

### FP32 input, fp32acc=ON, enforce_fp32_acc=true

| Path | Fidelity | TRISC_0 (unpack) | TRISC_1 (math) | Kernel total (ns) |
|------|----------|-----------------|---------------|-------------------|
| reduce_tile | HiFi4 | 121 | 127 | 49,541 |
| reduce_tile | HiFi2 | 71 | 80 | 46,252 |
| reduce_tile | LoFi | 55 | 62 | 44,977 |

## LLK Isolated Perf (no pipeline stalls)

From `tt_llk` perf tests, Float16_b, 16 tiles, TILE_LOOP:

| Op | MATH_ISOLATE per tile | UNPACK_ISOLATE total | L1-to-L1 per tile |
|----|----------------------|---------------------|-------------------|
| reduce_tile HiFi4 | 145 cyc | 780 cyc | 165 cyc |
| reduce_tile HiFi2 | 109 cyc | 780 cyc | 125 cyc |
| matmul_tile HiFi4 | 93 cyc | 1970 cyc | 137 cyc |
| matmul_tile LoFi | 88 cyc | 1970 cyc | 130 cyc |

Note: matmul unpack is 2.5x more expensive than reduce unpack in isolation, but matmul math is 1.56x faster. In the full L1-to-L1 pipeline, matmul's unpack cost partially cancels its math advantage.

## Key Conclusions

1. **Matmul is fidelity-independent** -- `REDUCE_MATMUL_FIDELITY` is hardcoded to HiFi4 in `reduce_helpers_compute.inl`. User-specified fidelity has no effect on the matmul path.

2. **With BF16 at HiFi4, matmul wins clearly** -- 1.7x faster per tile (57 vs 97 cycles), ~9.5% faster total kernel (37.6us vs 41.2us). This is the common case in production models.

3. **reduce_tile scales with fidelity, matmul doesn't** -- At HiFi2/LoFi, reduce_tile drops to 61 cycles/tile, closing the gap to ~9% per tile and ~4% total kernel vs matmul.

4. **With FP32 input, matmul gets significantly slower** -- matmul unpack nearly doubles (55->94 cycles) and math goes from 57 to 103. reduce_tile unpack is unaffected (86 stays 86). At HiFi4 they're equal (~97 vs 103). At HiFi2/LoFi, reduce_tile wins by ~5% total kernel.

5. **enforce_fp32_acc adds ~30% cost at HiFi4** (97->127 per tile) but is absorbed at lower fidelities (62 at LoFi vs 61 without enforce).

6. **fp32_dest_acc_en alone (without enforce_fp32_acc) has no effect** on per-tile reduce cycles.

7. **In production models, reduce always runs at HiFi4** -- no model code passes LoFi to layernorm/rmsnorm/mean ops. The fidelity advantage of reduce_tile is therefore not realized in practice.

8. **BulkWaitBulkPop vs WaitAndPopPerTile matters** -- With WaitAndPopPerTile (mean op), per-tile cycles are 5-10x higher (~450-520 cycles) because TRISC_1 stalls waiting for the unpack pipeline. With BulkWaitBulkPop (layernorm), tiles are pre-loaded and we see true compute performance (~57-97 cycles).

## How to Reproduce

### Prerequisites

- WH (wormhole_b0) machine
- tt-metal built from `sjovic/reduce-helpers` branch
- Python env activated: `source python_env/bin/activate`

### Files Modified for Profiling

1. **`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`** -- added `#include "tools/profiler/kernel_profiler.hpp"`
2. **`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl`** -- added `DeviceZoneScopedN("REDUCE-MATMUL-TILE")` / `DeviceZoneScopedN("REDUCE-TILE")` around reduce_matmul_tiles/reduce_tile calls in the BulkWaitBulkPop and WaitAndPopPerTile branches
3. **`ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp`** -- added `#include "tools/profiler/kernel_profiler.hpp"` (needed because this kernel doesn't transitively include it)
4. **`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp`** -- toggle `reduce_uses_matmul()` return value between `false` (force reduce_tile) and original (matmul for SUM/AVG + REDUCE_ROW)

### Switching Between matmul and reduce_tile Paths

In `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp`, change `reduce_uses_matmul()`:

```cpp
// Original (matmul path for SUM/AVG + REDUCE_ROW):
constexpr bool reduce_uses_matmul() {
    return (pool_type == ckernel::PoolType::SUM || pool_type == ckernel::PoolType::AVG) &&
           reduce_dim == ckernel::ReduceDim::REDUCE_ROW;
}

// Force reduce_tile path:
constexpr bool reduce_uses_matmul() {
    return false;
}
```

After changing, rebuild: `./build_metal.sh`

### Test Command (rmsnorm pre_allgather -- BulkWaitBulkPop)

```bash
# BF16 input
python -m tracy -r -m "pytest 'tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py::test_layernorm_part_1_with_program_cache[rmsnorm-n_devices=4-inp_shape=(1, 1, 32, 8192)-BFLOAT16-BFLOAT16]' --no-header -s --rootdir tests"

# FP32 input (requires adding ttnn.float32 to input_dtype parametrize list in the test)
python -m tracy -r -m "pytest 'tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py::test_layernorm_part_1_with_program_cache[rmsnorm-n_devices=4-inp_shape=(1, 1, 32, 8192)-BFLOAT16-FLOAT32]' --no-header -s --rootdir tests"
```

To change fidelity, edit line 117 in `test_distributed_layernorm_pre_allgather.py`:
```python
math_fidelity=ttnn.MathFidelity.HiFi4,  # or HiFi2, LoFi
```

To change fp32_dest_acc_en, edit line 119:
```python
fp32_dest_acc_en=False,  # or True
```

For enforce_fp32_acc=true with reduce_tile, also hardcode the template parameter in `reduce_helpers_compute.inl` at the `reduce_init` and `reduce_tile` calls.

### Test Command (mean reduce row -- WaitAndPopPerTile)

```bash
python -m tracy -r -m "pytest tests/ttnn/unit_tests/operations/reduce/test_reduction_mean.py::test_mean_reduce_row_perf[128x512-hifi4-fp32acc_off-input_bf16] --no-header -s --rootdir tests"
```

### LLK Perf Tests

```bash
cd tt_metal/third_party/tt_llk/tests
source .venv/bin/activate  # or source setup_external_testing_env.sh if first time
cd python_tests

# reduce perf (HiFi4 hardcoded in sources/reduce_perf.cpp:95, edit to change)
pytest "perf_reduce.py::test_perf_reduce[formats:Float16_b->Float16_b-dest_acc:No-reduce_dim:Row-pool_type:Average]" -s

# matmul perf (fidelity is a parameter)
pytest "perf_matmul.py::test_perf_matmul[combos:(InputOutputFormat[Float16_b,Float16_b], <DestAccumulation.No: False>, ([32, 32], [32, 32]))-math_fidelity:HiFi4]" -s
```

LLK results go to: `tt_metal/third_party/tt_llk/perf_data/perf_reduce/perf_reduce.csv` and `perf_matmul/perf_matmul.csv`

### Parsing Tracy Results

Tracy reports are generated under `generated/profiler/reports/<timestamp>/`. The per-tile zone data is in `profile_log_device.csv`.

Parse script (`parse_all_trisc.py`):
```python
import csv
import sys
from collections import defaultdict

zone_name = sys.argv[2] if len(sys.argv) > 2 else "REDUCE-MATMUL-TILE"
path = sys.argv[1]

starts = defaultdict(list)
ends = defaultdict(list)

with open(path) as f:
    reader = csv.reader(f)
    next(reader)
    next(reader)
    for row in reader:
        if len(row) >= 12 and zone_name in row[10]:
            risc = row[3]
            core = (row[1], row[2])
            cycles = int(row[5])
            if row[11] == "ZONE_START":
                starts[(risc, core)].append(cycles)
            elif row[11] == "ZONE_END":
                ends[(risc, core)].append(cycles)

for risc in ["TRISC_0", "TRISC_1", "TRISC_2"]:
    all_d = []
    cores = sorted(set(k[1] for k in starts.keys() if k[0] == risc))
    for core in cores:
        s = starts[(risc, core)]
        e = ends[(risc, core)]
        n = min(len(s), len(e))
        d = [e[i] - s[i] for i in range(n)]
        if len(d) > 1:
            all_d.extend(d[1:])
    if all_d:
        sorted_d = sorted(all_d)
        median = sorted_d[len(sorted_d) // 2]
        print(f"  {risc}: {len(all_d)} calls, median={median}, mean={sum(all_d)/len(all_d):.1f}, min={min(all_d)}, max={max(all_d)}")
```

Usage:
```bash
python3 parse_all_trisc.py generated/profiler/reports/<timestamp>/profile_log_device.csv REDUCE-MATMUL-TILE
python3 parse_all_trisc.py generated/profiler/reports/<timestamp>/profile_log_device.csv REDUCE-TILE
```

## Recommendation

For BF16 HiFi4 (the production case), matmul_tile is the better choice for REDUCE_ROW -- 1.7x faster per tile and ~9.5% faster total kernel. The reduce_tile path only wins in the FP32+lower-fidelity scenario, which doesn't occur in current models.

If we wanted reduce_tile to be competitive at HiFi4, the reduce LLK itself would need optimization -- both the unpack (86 vs 55 cycles) and math (97 vs 57 cycles) are significantly slower than the matmul equivalents.
