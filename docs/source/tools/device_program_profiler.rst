Device Program Profiler
=======================

Any point on the device side code can be marked with a time marker. The markers are stored in a statically assigned L1 location.
As part of tt_metal api ``LaunchProgram`` the markers are fetched from all the cores on the device.

Because downloading profiler results from device has high overheads, ``TT_METAL_DEVICE_PROFILER=1`` environment variable has to be set for ``LaunchProgram`` to perform the download.

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

1. Follow the tt-metal :ref:`Getting Started<Getting Started>` guide and README make sure ``PYTHONPATH``
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
