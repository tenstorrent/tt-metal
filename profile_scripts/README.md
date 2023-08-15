# TT-Metal Profile Documentation

## Instructions for new users

### Set environment variables and build TT-Metal framework

Run any scripts in the `tt_metal` root directory, rather than this script directory. The scripts are in `profile_scripts/env.sh`. Running the script file directly may not work, if so, copy the scripts and execute them in the command line manually.

### Run examples

Build and test two simple examples - loopback and element-wise binary operation in `profile_scripts/examples.sh`.

## Tensix to Tensix read/write speed

Measuring read and write requires 2 separate tests. The following explanations are for read. The src file to measure read speed between 2 tensix cores is `tests/tt_metal/llrt/test_run_risc_read_speed.cpp`. Compile all tests src codes by `make tests`. The compiled binary is `./build/test/llrt/test_run_risc_read_speed`, which will be executed by host machine. But before executing this binary, we need to build the corresponding kernels (under `built_kernel` directory) by executing `./build/test/build_kernels_for_riscv/test_build_kernel_blank` and `./build/test/build_kernels_for_riscv/test_build_kernel_risc_read_speed`. Kernel binaries will be sent to and executed by Tensix baby-riscv cores.

```
bash profile_scripts/Tensix2Tensix.sh
```

## Tensix to Tensix Read/Write issue and barrier waiting latency

Measuring read and write requires 2 separate tests. The following explanations are for read. The profile markers `5, 6, 7` are inserted into the kernel `tt_metal/kernels/dataflow/risc_read_speed.cpp`. Add profile option to the kernel building src file `tests/tt_metal/build_kernel_for_riscv/test_build_kernel_risc_read_speed.cpp` and corresponding compilation flags. Add profile option to the test src file `tests/tt_metal/llrt/test_run_risc_read_speed.cpp` and dump profiling results to the host machine. Because of the on-chip memory limits to store the profile results, the experiments repeat 4 times rather than 10000 in the non-profile experiments. But repeat the entire running 250 times, totally 4 * 250 = 1000 times. THe profiles cycles could be visualized by executing `tt_metal/tools/profiler/process_device_log.py`. Download and execute the script `profile_website.sh` on the local machine and open http://localhost:8888 to visualize the profiling results. Cycles between time marker 5 and 6 are corresponding to noc_async_read issue, while that between 6 and 7 are corresponding to noc_async_read_barrier.

```
bash profile_scripts/Tensix2Tensix_issue_barrier.sh
```

## Tensix to Tensix Read/Write fine grain profile

Similar to previous issue and barrier profiling, but add more profiler markers `11~18` in `tt_metal/src/firmwareriscv/grayskull/noc_nonblocking_api.h`. Because of the on-chip memory limits to store the profile results, the experiments repeat 1 times rather than 10000 in the non-profile experiments. But repeat the entire running 1000 times, totally 1 * 1000 = 1000 times.

```
bash profile_scripts/Tensix2Tensix_fine_grain.sh
```

## DRAM to Tensix read/write speed

Measuring read and write requires 1 test. The src code to measure read speed between DRAM and tensix cores is `tests/tt_metal/llrt/test_run_risc_rw_read_speed_banked_dram.cpp`. Compile all tests src codes by `make tests`. The compiled binary is `./build/test/llrt/test_run_risc_rw_speed_banked_dram`, which will be executed by host machine. But before executing this binary, we need to build the corresponding kernels (under `built_kernel` directory) by executing `./build/test/build_kernels_for_riscv/test_build_kernel_risc_rw_speed_banked_dram`. Kernel binaries will be sent to and executed by Tensix baby-riscv cores. In the script `profile_scripts/DRAM2Tensix.sh`, adjust DRAM channels by specify `DRAM_channel_pow_2`.

```
bash profile_scripts/DRAM2Tensix.sh
```

## Fine-grain profile and visualization of noc_async_read components

Measuring noc_async_read issues and noc_asycn_read barrier latency requires manual modifications in read kernel. The data movement kernel utilized in this measurement is `tt_metal/kernels/dataflow/reader_bmm_single_core_tilize_untilize.cpp`. In this kernel, `dim_x` defines the transfer size and `dim_y` defines the number of transfers. The tile size to transfer is 256x256 in Bfloat16 format. Change the transfer size and execute the script `profile_scripts/bmm_tilize_untilize.sh`. THe profiled cycle numbers are stored in `tt_metal/tools/profiler/logs/profile_log_device.csv`. Visualize the profiling results by executing `tt_metal/tools/profiler/process_device_log.py`. Download and execute the script `profile_website.sh` on the local machine and open http://localhost:8888 to visualize the profiling results. Cycles between time marker 5 and 6 are corresponding to noc_async_read issue, while that between 6 and 7 are corresponding to noc_async_read_barrier.

## Profile overhead on 5 types of RISC cores

Use `Elewise_binary` programming example to profile the profiler marker overhead. Add 12 consecutive profiler marker in the kernels `tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp`, `tt_metal/kernels/dataflow/write_unary.cpp` and `tt_metal/kernels/compute/elewise_binary.cpp` with marker `5-16`. The post process script average the gap cycle number between neighboring profile markers. Note that the profiler marker has different overhead at different positions. BRISC and NCRISC have 35 and 27 cycle overhead at Tensix2Tensix read/write speed profile experiment, but has 45 and 38 cycle overhead in this experiment.

```
bash profile_scripts/profile_all_risc_overhead.sh
```

## Element-wise binary load from/store to Tensix

The original Element-wise binary example load from/store to data between the Tensix core and DRAM. The NOC latency is larger than compute latency. This programming exmple allocate CB on another neighboring Tensix core and instead load from/store to data between the Tensix core and CBs on the neighboring core. However, the modified kernel is still NOC bound because the element-wise computation is fast. The kernel is added to `tt-metal/programming_examples/loopback_eltwise_binary/loopback_eltwise_binary.cpp` and the relative Makefile is also modified to enable the compilation.

```
bash profile_scripts/loopback_eltwise_binary.sh
```

## Bmm_tilize_untilize load from/store to Tensix

Similar to the modification of the Element-wise example above, the bmm_tilize_untilize also load from/store to data between the Tensix core and CBs on the neighboring core. This example realizes a compute bound case. The new kernel locates in `libs/tt_dnn/op_library/bmm/single_core/bmm_op_single_core_loopback_tilize_untilize.cpp` and the new test is in `tests/python_api_testing/unit_testing/test_loopback_bmm_tilize_untilize.py`.

```
bash profile_scripts/loopback_bmm_tilize_untilize.sh
```


# Analytical Model Document

The analytical model is implemented in the `analytical_model.py`. The parameters in the analytical model refer to the `get_args()` function in the sourse code. The scripts in the `perf_model.sh` iterate different non NIU programming latency of both Tensix2Tensix read and write with transcation size < 8k. The scripts model the issue and barrier latency for both read and write under different buffer sizes and trasaction sizes, as well as the NOC utiilization in each case. `compute_perf_model.sh` shows an example of the single core bmm operation test. By configuring the read/write issue latency and round trip latency, the data movement latency could be calculated. By specifying the circular buffer synchronization latency as well as the compute latency, the entire bmm latency could be calculated.
