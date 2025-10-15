# OP level performance insights - Compute vs DM time

## The idea
Use `cb_wait_front()` and `cb_reserve_back()` wait time from Compute and Data Movement kernel to get OP level performance realted insights:

1. Determine if OP is DM or Compute bound
2. Measure DM and Compute processing time

### DM vs Compute bound OP detection
In order to determine if OP is DM or Compute bound we should analyze waiting time spent in CB functions for Compute, DM Reader and DM Writer kernel. The decision should be made based on detecting where bottleneck is located e.g. which thread is waiting the most on CB calls:
1. `cb_wait_front()` from Compute kernel(executed on Unpacker thread) - bottlneck is in DM Reader kernel and it is limited by speed of reading from DRAM.
2. `cb_reserve_back()` from Compute kernel(executed on Packer thread) - bottleneck is in DM Writer kernel and it is limited by speed of writing to DRAM.
3. `cb_wait_front()` from DM Writer kernel - bottleneck is in Computer kernel on Packer thread. Determination of Compute kernel bottleneck root cause would require more detailed analysis since in most cases there are 3 threads running and double baffering of inputs/outputs is applied.
4. `cb_reserve_back()` from DM Reader kernel - bottleneck is in Compute kernel on Unpacker thread. Determination of Compute kernel bottleneck root cause would require more detailed analysis since in most cases there are 3 threads running and double baffering of inputs/outputs is applied.

### Measuring DM and Compute kernel processing time

Processing time of DM or Compute kernel in this context represent actual time kernel is doing meaningfull job, not waiting on CB to have new data or free space to store already processed data.

In terms of DM kernel, processing time considers number of cycles spent by NOC to read/write data from/to DRAM as well as cycles required for implementation of appropriate DM scheme(synchronization between cores, multicasting, etc.) depending on the OP.

In terms of Compute kernel, processing time is number of cycles required for Compute kernel(and Tensix engine) to read input data (assuming it is available in core's L1 memory), process it using Tensix engine and store results back in L1.

To estimate processing DM and Compute processing time we can use following simple formulas:
```
DM Reader time ~ DM Reader kernel time - cb_reserve_back() wait time
DM Writer time ~ DM Writer kernel time - cb_wait_front() wait time
Compute time ~ Compute(Unpacker) kernel time - cb_wait_front() wait time
```

Compute time estimation needs to be elaborated in more details since `cb_wait_front()` and `cb_reserve_back()` are executed on different threads(Unpacker and Packer) and they can happen in parallel.

## Testing and validation of methodology
In order to validate proposed methodology for detecting if OP is Compute or DM bound and estimating DM/Compute time two types of test scenarios are implemented:
1. Synthetic tests based on modified eltwise binary kernels. Idea is to have kernels with simulated workload using simple `wait for X cycles` approach to test and validate basic edge case scenarios: DM Reader bound OP, DM Writer bound OP, Compute bound OP.
2. Full grid matmul test that should cover realistic scenario of DM and Compute bound OPs.

### Test environment

Eltwise binary test case(`tests/tt_metal/tt_metal/test_eltwise_binary.cpp`) is used as baseline for implementing synthetic user controlled test benchamrk for validating proposed metodologhy. Reader, Write and Compute kernels are modified to enable following work types:
1. CB - kernel is executing only CB related functionality, `cb_wait_front()` / `cb_reserve_back()` paired with `cb_pop_front()` / `cb_push_back()` calls.
2. Wait X - kernel is executing controlable `wait for X cycles` between paired CB calls.
3. Compute/NOC - kerenel is executing real processing(data movement or compute) between paired CB calls.
4. Process - kernel is executing only processing functionality(data movement or compute) without CBs.

Every test is executed in 3 different test types:
1. Baseline - profiler is enabled measuring only kernel lenght. Kernel timings are used as baseline for overhead calculation since in this case no timing is measured inside CB calls.
2. Zone - profiler is measuring kernel length and `cb_wait_front()` / `cb_reserve_back()` wait times using profiler zones.
3. Counter - profiler is measuring kernel length using profiler zones and `cb_wait_front()` / `cb_reserve_back()` wait times using proposed counter approach that should reduce profiler overhead.


### Synthetic test scenarios
Synthetic tests are covering following scenarios:
1. Measuring CB synchronization overhead time
    - Reader: CB; Writer: CB; Compute: CB;
    - `TT_METAL_DEVICE_PROFILER=1 ./build_Release_tracy/test/tt_metal/test_eltwise_binary --reader 0 --writer 0 --compute 0`
2. DM Reader bound
    - Reader: wait 1000; Writer: CB; Compute: CB;
    - `TT_METAL_DEVICE_PROFILER=1 ./build_Release_tracy/test/tt_metal/test_eltwise_binary --reader 1000 --writer 0 --compute 0`
3. DM Writer bound
    - Reader: CB; Writer: wait 1000; Compute: CB;
    - `TT_METAL_DEVICE_PROFILER=1 ./build_Release_tracy/test/tt_metal/test_eltwise_binary --reader 0 --writer 1000 --compute 0`
4. Compute bound
    - Reader: CB; Writer: CB; Compute: wait 1000;
    - `TT_METAL_DEVICE_PROFILER=1 ./build_Release_tracy/test/tt_metal/test_eltwise_binary --reader 0 --writer 0 --compute 1000`
5. Eltwise test case
    - Reader: noc; Writer: noc; Compute: compute;
    - `TT_METAL_DEVICE_PROFILER=1 ./build_Release_tracy/test/tt_metal/test_eltwise_binary --reader 9999 --writer 9999 --compute 9999`

### Full grid matmul test scenarios
Full grid matmul tests are covering following scenarios:
1. DM bounded full grid matmul
    - `TT_METAL_DEVICE_PROFILER=1 pytest tests/didt/test_lm_head_matmul.py::test_lm_head_matmul -k "1chips" --didt-workload-iterations 1`
2. Compute bound full grid matmul
    - `TT_METAL_DEVICE_PROFILER=1 pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "with_gelu and 1chips" --didt-workload-iterations 1`

### Results
#### Synthetic test
In order to run and generate summaary of results for Synthetic test scenarios:
1. `git checkout skrsmanovic/cb-timing-testing branch`
2. `TT_METAL_DEVICE_PROFILER=1 pytest scripts/cb_analysis/sweep_cb_length.py`
3. Results are generated in `cb_length_logs/$timestamp/cb_length_all_stas.csv`

| variant  | reader           | writer           | compute          | trisc0_compute_block_duration | trisc1_compute_block_duration | trisc2_compute_block_duration | reader_block_duration | writer_block_duration | core_compute_cb_wait_front | core_compute_cb_reserve_back | core_writer_cb_wait_front | core_reader_cb_reserve_back | DM/Compute bound |
|:--------:|:----------------:|:----------------:|:----------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:---------------------:|:---------------------:|:--------------------------:|:----------------------------:|:-------------------------:|:---------------------------:|:----------------:|
| baseline | cb               | cb               | cb               | 194777.0                      | 18.0                          | 85322.0                       | 194873.0              | 85539.0               | 0.0                        | 0.0                          | 0.0                       | 0.0                         | N/A              |
| baseline | wait 1000 cycles | cb               | cb               | 4378695.0                     | 21.0                          | 97478.0                       | 4378793.0             | 97694.0               | 0.0                        | 0.0                          | 0.0                       | 0.0                         | N/A              |
| baseline | cb               | wait 1000 cycles | cb               | 213599.0                      | 22.0                          | 2175245.0                     | 213706.0              | 2177530.0             | 0.0                        | 0.0                          | 0.0                       | 0.0                         | N/A              |
| baseline | cb               | cb               | wait 1000 cycles | 2250901.0                     | 2133300.0                     | 2207752.0                     | 2248993.0             | 2207967.0             | 0.0                        | 0.0                          | 0.0                       | 0.0                         | N/A              |
| baseline | noc/compute      | noc/compute      | noc/compute      | 2484523.0                     | 2480977.0                     | 2483378.0                     | 2484582.0             | 2485171.0             | 0.0                        | 0.0                          | 0.0                       | 0.0                         | N/A              |
| zone     | cb               | cb               | cb               | 287252.0                      | 20.0                          | 155357.0                      | 287364.0              | 155615.0              | 153413.0                   | 88078.0                      | 78511.0                   | 147389.0                    | DM reader        |
| zone     | wait 1000 cycles | cb               | cb               | 4408428.0                     | 20.0                          | 155168.0                      | 4408522.0             | 155396.0              | 4286816.0                  | 81075.0                      | 70060.0                   | 124103.0                    | DM reader        |
| zone     | cb               | wait 1000 cycles | cb               | 313456.0                      | 21.0                          | 2226721.0                     | 313556.0              | 2228990.0             | 177273.0                   | 2164454.0                    | 59692.0                   | 160552.0                    | DM writer        |
| zone     | cb               | cb               | wait 1000 cycles | 2391157.0                     | 2130984.0                     | 2254036.0                     | 2389173.0             | 2254243.0             | 126067.0                   | 56299.0                      | 2167354.0                 | 2223522.0                   | Compute unpack   |
| zone     | noc/compute      | noc/compute      | noc/compute      | 2574946.0                     | 2571253.0                     | 2573743.0                     | 2574992.0             | 2575570.0             | 2358399.0                  | 1003773.0                    | 1692944.0                 | 133985.0                    | DM reader        |
| counter  | cb               | cb               | cb               | 206790.0                      | 22.0                          | 91310.0                       | 206889.0              | 91526.0               | 5300.0                     | 767.0                        | 41.0                      | 0.0                         | DM reader        |
| counter  | wait 1000 cycles | cb               | cb               | 4390007.0                     | 25.0                          | 100080.0                      | 4390088.0             | 100294.0              | 351057.0                   | 1051.0                       | 49.0                      | 0.0                         | DM reader        |
| counter  | cb               | wait 1000 cycles | cb               | 221940.0                      | 21.0                          | 2181700.0                     | 222059.0              | 2183966.0             | 5261.0                     | 159511.0                     | 20.0                      | 0.0                         | DM writer        |
| counter  | cb               | cb               | wait 1000 cycles | 2263588.0                     | 2125289.0                     | 2196124.0                     | 2261661.0             | 2196341.0             | 0.0                        | 0.0                          | 121414.0                  | 100630.0                    | Compute pack     |
| counter  | noc/compute      | noc/compute      | noc/compute      | 2485067.0                     | 2481496.0                     | 2483901.0                     | 2485132.0             | 2485687.0             | 185392.0                   | 69563.0                      | 133393.0                  | 0.0                         | DM reader        |

#### Full grid matmul test
In order to run and generate summaary of results for Full grid matmul test scenarios:
1. `git checkout skrsmanovic/cb-timing-testing-matmul`
2. `TT_METAL_DEVICE_PROFILER=1 pytest scripts/cb_analysis/sweep_cb_length.py`
3. Results are generated in `cb_matmul_length_logs/$timestamp/cb_matmul_length_all_stats.csv`

| test case                           | variant | core_compute_cb_wait_front | core_compute_cb_reserve_back | core_writer_cb_wait_front | core_reader_cb_reserve_back | DM/Compute bound |
|:-----------------------------------:|:-------:|:--------------------------:|:----------------------------:|:-------------------------:|:---------------------------:|:----------------:|
| 1D matmul - DM bound                | zone    | 1959086.0                  | 17623.0                      | 14952.0                   | 89292.0                     | DM reader        |
| 2D matmul with gelu - Compute bound | zone    | 91181.0                    | 53267.0                      | 181085.0                  | 1873649.0                   | Compute unpack   |
| 1D matmul - DM bound                | counter | 162504.0                   | 771.0                        | 1253.0                    | 5021.0                      | DM reader        |
| 2D matmul with gelu - Compute bound | counter | 7199.0                     | 512.0                        | 15152.0                   | 103827.0                    | Compute unpack   |
