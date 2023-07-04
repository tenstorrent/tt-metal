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

Measuring read and write requires 2 separate tests. The following explanations are for read. The profile markers are inserted into the kernel `tt_metal/kernels/dataflow/risc_read_speed.cpp`. Add profile option to the kernel building src file `tests/tt_metal/build_kernel_for_riscv/test_build_kernel_risc_read_speed.cpp` and corresponding compilation flags. Add profile option to the test src file `tests/tt_metal/llrt/test_run_risc_read_speed.cpp` and dump profiling results to the host machine. Because of the on-chip memory limits to store the profile results, the experiments repeat 4 times rather than 10000 in the non-profile experiments. But repeat the entire running 250 times, totally 4 * 250 = 1000 times. THe profiles cycles could be visualized by executing `tt_metal/tools/profiler/process_device_log.py`. Download and execute the script `profile_website.sh` on the local machine and open http://localhost:8888 to visualize the profiling results. Cycles between time marker 5 and 6 are corresponding to noc_async_read issue, while that between 6 and 7 are corresponding to noc_async_read_barrier.

```
bash profile_scripts/Tensix2Tensix_issue_barrier.sh
```

## DRAM to Tensix read/write speed

Measuring read and write requires 1 test. The src code to measure read speed between DRAM and tensix cores is `tests/tt_metal/llrt/test_run_risc_rw_read_speed_banked_dram.cpp`. Compile all tests src codes by `make tests`. The compiled binary is `./build/test/llrt/test_run_risc_rw_speed_banked_dram`, which will be executed by host machine. But before executing this binary, we need to build the corresponding kernels (under `built_kernel` directory) by executing `./build/test/build_kernels_for_riscv/test_build_kernel_risc_rw_speed_banked_dram`. Kernel binaries will be sent to and executed by Tensix baby-riscv cores. In the script `profile_scripts/DRAM2Tensix.sh`, adjust DRAM channels by specify `DRAM_channel_pow_2`.

```
bash profile_scripts/DRAM2Tensix.sh
```

## Fine-grain profile of noc_async_read components

Measuring noc_async_read issues and noc_asycn_read barrier latency requires manual modifications in read kernel. The data movement kernel utilized in this measurement is `tt_metal/kernels/dataflow/reader_bmm_single_core_tilize_untilize.cpp`. In this kernel, `dim_x` defines the transfer size and `dim_y` defines the number of transfers. The tile size to transfer is 256x256 in Bfloat16 format. Change the transfer size and execute the script `profile_scripts/bmm_tilize_untilize.sh`. THe profiled cycle numbers are stored in `tt_metal/tools/profiler/logs/profile_log_device.csv`. Visualize the profiling results by executing `tt_metal/tools/profiler/process_device_log.py`. Download and execute the script `profile_website.sh` on the local machine and open http://localhost:8888 to visualize the profiling results. Cycles between time marker 5 and 6 are corresponding to noc_async_read issue, while that between 6 and 7 are corresponding to noc_async_read_barrier.
