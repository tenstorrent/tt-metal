# tt_metal profiler

This folder contains the library and postprocessing scripts for profiling device side and host side
kernels and interfaces.

## Profiling host side API

Profiling is provided through the `profiler.hpp` module. The idea is that portions of a function or
all of it can be wrapped between a start and end timer marks. After the execution of the function,
the delta between the two markers can be calculated as the period of the portion you wanted to
profile. For each tt_metal function such as `LaunchKernels` the entire function is wrapped in timers
and using `dumpProfilerResults` the result will be dumped into `profiler_log.json` in the current
directory.

With respect to how the host side api of `tt_metal` is designed, it is assumed that a subset of
methods from this module will execute __only once__ during each section of running an entire
program. `test_add_two_ints.cpp` is a good example for this. In the first section it configures the
device, launches the kernel, and reads from device L1. All of these tasks can be profiles. The
second section only launches the kernel with new args and reads from L1 as the device is already
configured and only new kernel args are being sent. In order to keep the distinction between the
first and second section `dumpProfilerResults(<section name>)`is used at the end of each section to
dump the host profiler results for each function of each section.  The code for this test is
demonstrative of what was described.

The following is a sample result from runn the `test_add_two_ints` tt_metal test. You can see that
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

1. The compile flag for kernel side profiling has to be set, this is done by setting the flag in `tt_metal::CompileProgram`.
2. Print server start flag must be set, this is done setting the flag in `tt_metal::ConfigureDeviceWithProgram` .
3. `tt_metal::stopPrintfServer` function has to run before another `tt_metal::ConfigureDeviceWithProgram` with print start server set to true can start.

e.g.
```
    constexpr bool profile_kernel = true;
    pass &= tt_metal::CompileProgram(device, program, skip_hlkc, profile_kernel);
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program, profile_kernel);
    .
    .
    .
    .
    .
    tt_metal::WriteRuntimeArgsToDevice(device, add_two_ints_kernel, core, second_runtime_args);
    pass &= tt_metal::LaunchKernels(device, program);

    .
    .
    .
    .
    .
    tt_metal::stopPrintfServer();
```

After this setup, default markers will be generated and can be post-processed.

Default markers are:

1. Kernel start
2. Kernel end

The generated csv is `profile_log_kernel.csv` is saved under `tools/profiler/` by default.

Sample generated csv for running a kernel on core 0,0:

```
0, 0, 0, BRISC, 2, 46413751954532
0, 0, 0, BRISC, 3, 46413751954779
0, 0, 0, NCRISC, 2, 46413751955228
0, 0, 0, NCRISC, 3, 46413751955414
```


### Post-processing kernel profiler


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

4. The following artifact will also be generated under the `tools/profiler/` folder:
    - `device_perf.html` contains the interactive time series plot
    - `device_stats.txt` contains the extended stats for the run
    - `device_arranged_timestamps.csv` contains all timestamps arranged by each row dedicated to cores

5. For convenience all of these files are tarball into `device_perf_results.tar`
