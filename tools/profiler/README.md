# ll_buda profiler

This folder contains the library and postprocessing scripts for profiling device side and host side
kernels and interfaces.

## Profiling host side API

Profiling is provided through the `profiler.hpp` module. The idea is that portions of a function or
all of it can be wrapped between a start and end timer marks. After the execution of the function,
the delta between the two markers can be calculated as the period of the portion you wanted to
profile. For each ll_buda function such as `LaunchKernels` the entire function is wrapped in timers
and using `dumpProfilerResults` the result will be dumped into `profiler_log.json` in the current
directory.

With respect to how the host side api of `ll_buda` is designed, it is assumed that a subset of
methods from this module will execute __only once__ during each section of running an entire
program. `test_add_two_ints.cpp` is a good example for this. In the first section it configures the
device, launches the kernel, and reads from device L1. All of these tasks can be profiles. The
second section only launches the kernel with new args and reads from L1 as the device is already
configured and only new kernel args are being sent. In order to keep the distinction between the
first and second section `dumpProfilerResults(<section name>)`is used at the end of each section to
dump the host profiler results for each function of each section.  The code for this test is
demonstrative of what was described.

The following is a sample result from runn the `test_add_two_ints` ll_buda test. You can see that
the section name is a category in the csv. The start and stop counters are from the epoch.

```
Section Name, Function Name, Start timer count [ns], Stop timer count [ns], Delta timer count [ns]
first, LaunchKernels, 675598390620333, 675598390740682, 120349
first, ConfigureDeviceWithProgram, 675598152012369, 675598390619993, 238607624
first, CompileProgram, 675597384816840, 675598152009299, 767192459
second, LaunchKernels, 675598625865918, 675598625981107, 115189
second, ConfigureDeviceWithProgram, 675598392545035, 675598625864988, 233319953
```

## Profiling kernel side API

### Default Markers
On the host side minimal changes are necessary on the code.

1. The compile flag for kernel side profiling has to be set, this is done by setting the flag in `ll_buda::CompileProgram`.
2. Print server start flag must be set, this is done setting the flag in `ll_buda::ConfigureDeviceWithProgram` .
3. `ll_buda::stopPrintfServer` function has to run before another `ll_buda::ConfigureDeviceWithProgram` with print start server set to true can start.

e.g.
```
    constexpr bool profile_kernel = true;
    pass &= ll_buda::CompileProgram(device, program, skip_hlkc, profile_kernel);
    pass &= ll_buda::ConfigureDeviceWithProgram(device, program, profile_kernel);
    .
    .
    .
    .
    .
    ll_buda::WriteRuntimeArgsToDevice(device, add_two_ints_kernel, core, second_runtime_args);
    pass &= ll_buda::LaunchKernels(device, program);

    .
    .
    .
    .
    .
    ll_buda::stopPrintfServer();
```

After this setup, default markers will be generated and can be post-processed.

Default markers are:

1. Kernel start with timer_id 2
2. Kernel end with timer_id 3

The generated csv is `profile_log_kernel.csv` is saved under `tools/profiler/` byt default.

Sample generated csv for running a kernel on coer 0,0:

```
0, 0, 0, BRISC, 2, 46413751954532
0, 0, 0, BRISC, 3, 46413751954779
0, 0, 0, NCRISC, 2, 46413751955228
0, 0, 0, NCRISC, 3, 46413751955414
```

<!--`test_matmul_multi_core_multi_dram.cpp` is a good example that demonstrates how to grab kernel side-->
<!--profiler time. Both kernels `writer_matmul_tile_layout.cpp` and  `reader_matmul_tile_layout.cpp`-->
<!--used by these tests are modified to measure their entire execution period.-->

<!--Once the `ll_buda` test that runs the kernel under profile finishes, `profile_log_kernel.csv` is-->
<!--generated.-->

<!--The following is the sample result for the `test_add_two_ints.cpp`. You can see that inline with-->
<!--the host side example above, same markers are recorded twice as the same kernel runs on the same cores with different args.-->

<!--```-->
<!--0 ,1 ,1 ,BRISC ,0 ,1892410749640-->
<!--0 ,1 ,1 ,BRISC ,1 ,1892410749851-->
<!--0 ,1 ,1 ,BRISC ,0 ,1892694762480-->
<!--0 ,1 ,1 ,BRISC ,1 ,1892694762683-->
<!--```-->

### Postprocessing kernel profiler

<!--Plotting kernel profiler data requires setting up the `plot_steup.py`. Sample tests are added to this file. The setup is based on timer ID and which risc type they come from. In `test_add_two_ints` for examples shows the plotting of its `brisc`-->
<!--kernel profiling results. -->

1. Set up the environment for running the plotter:

```
cd tools/profiler/
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

2. Run plotter webapp:
```
cd tools/profiler/
./postproc_kerenel_log.py
```

3. Navigate to `<machine IP>:8050` to view output chart.

4. `kernel_perf.html` and `device_stats.txt` are generated that contain the plot and the chart for the stats.
