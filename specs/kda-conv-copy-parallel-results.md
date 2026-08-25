# KDA convolution copy parallelism prototype

## Verdict

The prototype is successful on Blackhole. Splitting aligned row-major copy work
within logical rows makes the KDA convolution adapter use 110 worker cores
instead of 3. At SP4xTP2, synchronized trace-wall convolution export improves
7.88x (94.331 to 11.974 us) and import improves 11.80x (133.946 to 11.349 us).
The complete adapter overhead falls from 1.136% to 0.283% for export and from
1.532% to 0.277% for import.

All three required layouts pass the real Kimi-K3 T=5120 workload. Real and
patterned cache round trips are bit-identical, and output/state PCC is unchanged.

## Prototype

The existing row-major copy program assigns complete logical rows to workers.
That limits convolution tensors shaped `[1, 3, W]` to three cores. The prototype
retains that program as a fallback and adds a guarded path when input and output
pages share an aligned common unit:

- `unit_elements = gcd(input_page_elements, output_page_elements)`;
- the unit must be smaller than a logical row, aligned for both buffers, and no
  larger than the existing 256 KiB subblock limit;
- `(logical row, unit)` work is distributed across the worker grid;
- reader and writer move each unit directly between its source/destination page
  offset through one double-buffered circular buffer.

For the KDA convolution cache the common unit is one 64-element BF16 contract
page (128 bytes). SP4xTP2 therefore exposes `3 * 288 = 864` independent units,
enough to use all 110 workers. The recurrent S cache remains on the existing
tiled copy path, which already uses 110 workers.

## Synchronized trace-wall results

Medians below use 20 samples of 100 trace replays. Times are milliseconds.

| Layout | Direction | Conv baseline | Conv prototype | Speedup | Combined baseline | Combined prototype | Layer overhead |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SP1xTP8 | export | 0.027374 | 0.009022 | 3.03x | 0.035402 | 0.014961 | 0.154% |
| SP1xTP8 | import | 0.036236 | 0.008885 | 4.08x | 0.044923 | 0.014835 | 0.153% |
| SP2xTP4 | export | 0.048952 | 0.009955 | 4.92x | 0.061719 | 0.024179 | 0.250% |
| SP2xTP4 | import | 0.069749 | 0.009657 | 7.22x | 0.082582 | 0.018192 | 0.188% |
| SP4xTP2 | export | 0.094331 | 0.011974 | 7.88x | 0.114825 | 0.028582 | 0.283% |
| SP4xTP2 | import | 0.133946 | 0.011349 | 11.80x | 0.154892 | 0.028045 | 0.277% |

Layer medians were 9.7115, 9.6619, and 10.1171 ms for SP1xTP8, SP2xTP4,
and SP4xTP2 respectively. The corresponding baseline medians were 9.7171,
9.6486, and 10.1103 ms, so the layer workload itself is stable.

## Tracy attribution

The targeted SP4xTP2 profile reports the intended parallel kernels and 110
cores for both convolution directions:

| Direction | Rows | Cores | Kernel min / median / max (us) | Baseline median | Kernel speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| interleaved to ND | 72 | 110 | 4.304 / 4.486 / 4.958 | 90.473 | 20.17x |
| ND to interleaved | 64 | 110 | 3.711 / 3.787 / 4.607 | 128.816 | 34.02x |

The unprofiled trace-wall numbers are the latency verdict; Tracy is used only
to attribute the improvement. The remaining 9-12 us trace-wall floor includes
operation/dispatch overhead and is larger than the profiled device kernel time.

## Correctness and validation

The full matrix reports `bit_identical_real_round_trip=true` and
`bit_identical_patterned_round_trip=true` for every layout. PCC ranges from
0.999799 to 0.999999 and matches the baseline.

Commands:

```bash
./build_metal.sh --build-type Release --enable-ccache

env TT_METAL_HOME=$PWD LD_LIBRARY_PATH=$PWD/build_Release/lib \
  scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/base_functionality/test_to_memory_config.py::test_to_memory_config_rm_dram_nd_intra_row_parallel -q -s

env TT_METAL_HOME=$PWD LD_LIBRARY_PATH=$PWD/build_Release/lib \
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/base_functionality/test_to_memory_config.py \
  -k 'test_to_memory_config_rm_preallocated_output or test_to_memory_config_rm_override_runtime_arguments' -q -s

env TT_METAL_HOME=$PWD LD_LIBRARY_PATH=$PWD/build_Release/lib \
  KIMI_K3_CKPT=/localdev/mvasilijevic/.cache/Kimi-K3/9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  KDA_ADAPTER_TIMING_SAMPLES=20 KDA_ADAPTER_TIMING_REPS=100 PERF_REPS=10 \
  scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_cache_adapter_perf.py -q -s
```

Results: build passed; focused test 1 passed; fallback/program-cache regression
4 passed; full matrix 3 passed; targeted Tracy 1 passed. Every device run ended
with `SAFE_PYTEST_RESULT: PASS`.

Raw timing records are in `specs/kda-conv-copy-parallel-results.jsonl`. Local
full logs are `specs/kda-conv-copy-parallel-safe.log` and
`specs/kda-conv-copy-parallel-tracy.log`. The Tracy CSV is
`generated/profiler/reports/2026_08_25_23_01_45/ops_perf_results_2026_08_25_23_01_45.csv`.

## Scope

This is a generic TTNN row-major copy prototype, not a KDA-only special case.
The guarded fallback preserves the old path for unaligned or uneven page widths;
the existing 20-element uneven-shard case exercises that fallback. Broader copy
regression coverage is still appropriate before proposing the mechanism for
production.
