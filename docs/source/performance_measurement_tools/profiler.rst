========================
Execution Time Profiler
========================

Host Side
=========

Host API is profiled by wrapping the portion of the code that needs profiling with start and end
markers with the same timer name. After the execution of the wrapped code, the start, end and the
delta in between them for all the timers is recorded in a CSV for further post processing.

Setup
-----

For profiling any module on the host side, an object of the of the ``Profiler`` class is needed
in order to record the marked times and dump the result to a CSV. The ``Profiler`` is defined under
the ``tools/profiler/profiler.hpp`` header which can be include as follows.

..  code-block:: C++

    #include "tools/profiler/profiler.hpp"

The module Make procedure should also include the profiler library. This can be done by adding the
the ``-lprofiler`` flag to the ``LDFLAG`` argument in the ``module.mk`` of that module. For example
for tests under ``tt_metal``, which uses the profiler, the following is the ``LDFLAG`` line in ``tt_metal/tests/module.mk``.

..  code-block:: MAKEFILE

    TT_METAL_TESTS_LDFLAGS = -ltt_metal_impl -ltt_metal -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp

With the instance of the ``Profiler`` class, ``markStart`` and ``markStop`` functions can be used to
profile the module. Taking ``tt_metal`` as an example, ``tt_metal_profiler`` is
instantiated as a static member of the module ``tt_metal/tt_metal.cpp`` as follows.

..  code-block:: C++

    static Profiler tt_metal_profiler = Profiler();

In functions such as ``LaunchKernels`` the entire code within the function is wrapped under the
``markStart`` and ``markStop`` calls with the timer name ``"LaunchKernels"``.

..  code-block:: C++

    bool LaunchKernels(Device *device, Program &program) {

        tt_metal_profiler.markStart("LaunchKernels");
        bool pass = true;

        auto cluster = device->cluster();

        <Internals of LaunchKernels>

        cluster->broadcast_remote_tensix_risc_reset(pcie_slot, TENSIX_ASSERT_SOFT_RESET);

        tt_metal_profiler.markStop("LaunchKernels");
        return pass;
    }

After the execution of all wrapped code. A call to  ``tt_metal::DumpHostProfileResults`` will process the deltas on all
timers and dump the results into a CSV called ``profile_log_host.csv``. The location of the CSV is
assigned by ``tt_metal::SetProfilerDir`` or it will be the default location ``tt_metal/tools/profiler/logs/``.

The ``tt_metal::DumpHostProfileResults`` also flushes all the timers data after the dump. This is so that the same
object can be used to perform multiple consecutive measurements on the same timer name. The ``name_append`` argument adds
a ``Section name`` column to the CSV that demonstrates which row in the CSV it
belongs to.

``tt_metal\tests\test_add_two_ints.cpp`` is a good example that demonstrates this scenario.
``LaunchKernels`` is called twice in this test, if we only dump results once at the end of the
execution, we will only get the results on the last call to that function. With the use of sections
names we can call ``DumpHostProfileResults`` twice and get and output such as the following in the
CSV.


..  code-block:: c++

    Section Name, Function Name, Start timer count [ns], Stop timer count [ns], Delta timer count [ns]
    first, LaunchKernels, 675598390620333, 675598390740682, 120349
    first, ConfigureDeviceWithProgram, 675598152012369, 675598390619993, 238607624
    first, CompileProgram, 675597384816840, 675598152009299, 767192459
    second, LaunchKernels, 675598625865918, 675598625981107, 115189
    second, ConfigureDeviceWithProgram, 675598392545035, 675598625864988, 233319953


Device Side
===========

Any point on the device side code can be marked with a time marker. The markers are stored in a statically assigned L1 location.
After LaunchKernel the markers are fetched from all the cores on the device. Default markers are present in device FW that mark kernel and FW start and end times.
Post processing scripts are provided to perform various statistical analysis on the markers data.

Setup
-----

On the host side minimal changes are necessary on the code.

1. The compile flag for device side profiling has to be set, this is done by setting the flag in ``tt_metal::CompileProgram``.
2. For each kernel launch through ``tt_metal::LaunchKernels(device, program);``  that you want device side profiler markers dumped,
   A call to ``tt_metal::DumpDeviceProfileResults(device, program);`` has to be made to append the markers to
   the current test device side output ``profile_log_device.csv``

e.g.

..  code-block:: c++

    constexpr bool profile_device = true;
    pass &= tt_metal::CompileProgram(device, program, profile_device);
    .
    .
    .
    .
    .
    tt_metal::WriteRuntimeArgsToDevice(device, add_two_ints_kernel, core, second_runtime_args);
    pass &= tt_metal::LaunchKernels(device, program);
    if (profile_device){
        tt_metal::DumpDeviceProfileResults(device, program);
    }

After this setup, default markers will be generated and can be post-processed.

Default markers are:

1. FW start
2. Kernel start
3. Kernel end
4. FW end

The generated csv is ``profile_log_device.csv`` is saved under ``tt_metal/tools/profiler/logs`` by default.

Sample generated csv for a run on core 0,0:

..  code-block:: c++

    0, 0, 0, NCRISC, 1, 1882735035004
    0, 0, 0, NCRISC, 2, 1882735036049
    0, 0, 0, NCRISC, 3, 1882735036091
    0, 0, 0, NCRISC, 4, 1882735036133
    0, 0, 0, BRISC, 1, 1882735032214
    0, 0, 0, BRISC, 2, 1882735035364
    0, 0, 0, BRISC, 3, 1882735035433
    0, 0, 0, BRISC, 4, 1882735035518


Post-processing device profiler
-------------------------------

1. Follow the tt-metal :ref:`Getting Started<Getting Started>` and
   :ref:`Getting Started for Devs<Getting started for devs>` guides and make sure ``PYTHONPATH``
   and other tt-metal environment variables are set. Activate the python environment as suggested by the guides.

2. Run plotter webapp with:

..  code-block:: sh

    cd $TT_METAL_HOME/tools/profiler/
    ./process_device_log.py

3. Navigate to ``<machine IP>:<PORT>`` to the Device Profiler Dashboard to view stats and timeline plots. ``<PORT>`` default is ``8050`` if not set by the ``-p/--port`` cli option.

4. The following are the notable artifacts that will be generated under the ``tt_metal/tools/profiler/output`` folder:
    - ``device_perf.html`` contains the interactive time series plot
    - ``device_stats.txt`` contains the extended stats for the run
    - ``device_rearranged_timestamps.csv`` contains all timestamps arranged by each row dedicated to cores

5. For convenience all of these artifacts are tarballed into ``device_perf_results.tar``. The file is under the same output folder as the artifacts and can be downloaded by clicking the ``DOWNLOAD ARTIFACTS`` button on the webapp.

6. Use  ``./process_device_log.py --help`` to get a list of available cli options to run the post processes differently. Some of the notable options are:
    - Path to device side profiler log csv
    - Path to artifacts output folder
    - Custom webapp port
    - Disabling printing stats, running webapp, generating plots and other portions of the default post-process flow


Limitations
-----------

* Each core has limited L1 buffer for recording device side markers. Flushing mechanism are in progress
  to push the data to DRAM and eventually the host to alleviate this limitation.

* The cycle counts give very good relative numbers with regards to various events that are marked
  on the kernel. Syncing this with the wall clock is not brought in yet. This will require
  collection on core reset times on the host side and syncing every cycle count accordingly

* It is relatively safe to assume that all RISCs on all cores are taken out of reset at the same
  time so processing the cycle counts read from various RISCs is reasonable.

* Debug print can not used in kernels that are being profiled.Correct usage of DPRINT and profiler is suggested in the `add_two_ints.cpp` tt_metal test. If `profile_device` is set, it profiles, if not it prints. The test will error out if DRPRINT and profiler are attempted to be used together.
