====================
Performance Profiler
====================

Quick Start Guide
=================

1. Build the tt_metal project with the profiler build flag set as follows:

..  code-block:: sh

    cd $TT_METAL_HOME
    make clean
    make build ENABLE_PROFILER=1

Or,

Run the following build script:

..  code-block:: sh

    cd $TT_METAL_HOME
    scripts/build_scripts/build_with_profiler_opt.sh


2. Determine the execution command for running your model of OP unit test. e.g.

``pytest tests/python_api_testing/models/stable_diffusion/test_residual_block.py``

3. In the same shell, run ``profile_this.py`` to profile the command.

..  code-block:: sh

    cd $TT_METAL_HOME
    ./tt_metal/tools/profiler/profile_this.py -c "pytest tests/python_api_testing/models/stable_diffusion/test_residual_block.py"

**NOTES**:

- Do not skip ``make clean`` as it is the only way to ensure all effected files are recompiled.
- Export any environment variables that your command requires and they will be picked up by the ``profile_this.py`` script.
- Make sure tt_metal python virtual env is setup as per :ref:`Getting Started<Getting Started>`
- ``-c`` is the only option for the script which is for providing the run the test execution script.
- If your command actually starts a collection of tests, please refer to the `Profiling OPs`_ section for more info on how to choose log locations for your tests.
- Once the script finished, it will provide log messages to tell you where the generated ops report csv is stored.

OPs Report
==========

The OPs profiler report demonstrates the execution flow of the OPs in the pipeline. Each row in the CSV represents an OP executed.

For each OP, multiple data points are provided in the columns of the CSV.

The headers of the columns with their descriptions is below:

- **OP CODE**: Operation name, for C++ level OPs this code is the name of the class for the OP

- **OP TYPE**: Operation type, where the op ran and which part of code it is coming from

    - *python_fallback*: OP fully implemented in python and running on CPU
    - *tt_dnn_cpu*: OP implemented in C++ and running on CPU
    - *tt_dnn_device*: OP implemented in C++ and running on DEVICE

- **GLOBAL CALL COUNT**: The index of the op in the execution pipeline

- **ATTRIBUTES**: Any additional attribute or meta-data that can be manually added during the execution of the op

    - ``op_profiler::append_meta_data`` can be used on the C++ side to add to this field
    - ``ttl.profiler.append_meta_data`` can be used on the Python side to add to this field

- **MATH FIDELITY**: Math fidelity of the fields

    - LoFi
    - HiFi2
    - HiFi3
    - HiFi4

- **CORE COUNT**: The number of cores used on the device for this operation

- **PARALLELIZATION STRATEGY**: How the device kernel parallelizes across device cores

- **HOST START TS**: System clock time stamp stored at the very beginning of the OP execution

- **HOST END TS**: System clock time stamp stored at the very end of the OP execution

- **HOST DURATION [ns]**: Duration of the OP in nanoseconds, calculated as end_ts - start_ts

- **DEVICE START CYCLE**: Tensix cycle count from the earliest RISC of the earliest core of the device that executed the OP kernel

- **DEVICE END CYCLE**: Tensix cycle count from the latest RISC of the latest core of the device that executed the OP kernel

- **DEVICE DURATION [ns]**: Duration on the device for the OP, calculated as (end_cycle - start_cycle)/core_frequency

- **Input & Output Tensor Headers**: Header template is {Input/Output}_{IO Number}_{Field}. e.g. INPUT_0_MEMORY

    - *W*: Tensor batch count
    - *Z*: Tensor channel count
    - *Y*: Tensor Height
    - *X*: Tensor Width
    - *LAYOUT*:
        - ROW_MAJOR
        - TILE
        - CHANNELS_LAST
    - *DATA TYPE*:
        - BFLOAT16
        - FLOAT32
        - UINT32
        - BFLOAT8_B
    - *MEMORY*
        - dev_0_dram
        - dec_0_l1
        - host

- **CALL DEPTH**: Level of the OP in the call stack. If OP call other OPs the child OP will have a CALL DEPTH one more than the CALL DEPTH of the caller

- **TT_METAL API calls**: Statistics on tt_metal calls, particularly how many times they were called during the OP and what was their average duration in nanoseconds

    - CompileProgram
    - ConfigureDeviceWithProgram
    - LaunchKernels
    - ReadFromDevice
    - WriteToDevice
    - DumpDeviceProfileResults

Deeper Dive
===========

Automated Script
----------------

The ``profile_this.py`` script is an automated script that cover most Models and OPs units test profiling scenarios.

This scripts performs the following items:

1. Checks if the project is correctly built with ``PROFLER="enabled"``
2. Executes the provided under test command to provide both host and device side profiling data
3. Post-processes all the collected log locations

Note on step two, because fetching the device profiling data adds high overhead to the actual execution time,
the under test command is executed twice, once with device profiling and once without.
The results of the two runs are then stitched together into on csv to present device data alongside host time data
that is not affected by device download overhead.

Setp 2 above can manually be replicated if we have a profiler enabled build:

1. Run your command without device profiling i.e. env variable ``TT_METAL_DEVICE_PROFILER=0``
2. Run your command with device profiling using the same logs folder location as step 1 i.e. env variable ``TT_METAL_DEVICE_PROFILER=0``
   te profiler will automatically append ``_device`` to the folder location
3. Run ``process_ops_logs.py`` with the input log location ``-i`` pointed to the logs location set by ``set_profiler_location``


Profiling OPs
-------------

Models and OPs unit tests are automatically profiled in PROFILER builds.

By default OPs logs are saved under ``$TT_METAL_HOME/tt_metal/tools/profiler/logs/ops/``.

This folder can be changed by using ``ttl.profiler.set_profiler_location`` function.

Refer to the ``ttl.profiler`` module of the python bindings' docs for info on more API functions available for profiling.

**NOTE**: ``ttl.profiler`` is a separate module from the ``utility_functions.profiler`` module. ``utility_functions.profiler`` will be deprecated once all of its features are
covered by ``ttl.profiler``.

Post-processing ops profiler
----------------------------

1. Follow the tt-metal :ref:`Getting Started<Getting Started>` and
   :ref:`Getting Started for Devs<Getting started for devs>` guides and make sure ``PYTHONPATH``
   and other tt-metal environment variables are set. Activate the python environment as suggested by the guides.

2. Run ops profiler script on default ops' logs folder ``$TT_METAL_HOME/tt_metal/tools/profiler/logs/ops/`` with:

..  code-block:: sh

    cd $TT_METAL_HOME/tt_metal/tools/profiler/
    ./process_ops_logs.py

3. Output csv will be generated under ``$TT_METAL_HOME/tt_metal/tools/profiler/output/ops/`` by default. CLI options can be used to change this directory and also prepend
   datetimestamp and append extra information to the name of the csv. A tarball of the ops logs folder is also generated with the same name as the csv under the same output folder.

4. Use  ``./process_ops_logs.py --help`` to get a list of available cli options to run the post processes differently. Some of the notable options are:
    - Path to ops' profiler logs folder
    - Path to  output folder
    - Run plots dashboard (Beta stage)
    - Custom webapp port


Profiling Device
================

Any point on the device side code can be marked with a time marker. The markers are stored in a statically assigned L1 location.
As part of tt_metal api ``LaunchKernel`` the markers are fetched from all the cores on the device.

Because downloading profiler results from device through has high overheads, ``TT_METAL_DEVICE_PROFILER=1`` environment variable has to be set for ``LaunchKernel`` to perform the download.

Default markers are present in device FW(i.e. ``.cc`` files) that mark kernel and FW start and end times.

Default markers are:

1. FW start
2. Kernel start
3. Kernel end
4. FW end

The generated csv is ``profile_log_device.csv`` and is saved under ``tt_metal/tools/profiler/logs`` by default.

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

    cd $TT_METAL_HOME/tt_metal/tools/profiler/
    ./process_device_log.py

3. Navigate to ``<machine IP>:<PORT>`` to the Device Profiler Dashboard to view
   stats and timeline plots. ``<PORT>`` default is ``8050`` if not set by the
   ``-p/--port`` cli option. Note that if you are using a Tenstorrent cloud
   machine and are viewing the dashboard through a localhost port forwarded via
   SSH, you will need to forward port ``<PORT>`` using the ``-L`` option when
   you connect via ``ssh``.  Otherwise, you will not be able to access the
   dashboard.

4. The following are the notable artifacts that will be generated under the ``tt_metal/tools/profiler/output/device`` folder:
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


TRACY
=====

Profiling
---------

Fresh build with the tracy flag is required for profiling with tracy profiler.

..  code-block:: sh

    make clean
    make build ENABLE_TRACY=1

With this build, all the marked zones in the code will be profiled.

GUI
---

On your mac you need install tracy GUI with brew. On your mac terminal run:

..  code-block:: sh

    brew install tracy

Once installed run tracy GUI with:

..  code-block:: sh

    TRACY_DPI_SCALE=1.0 tracy

In the GUI you should start listening to the machine that your are running your code on over port 8086 (e.g. 172.27.28.132:8086) but setting the client address and clicking
connect.


At this point once you run your program, tracy will automatically start profiling.
