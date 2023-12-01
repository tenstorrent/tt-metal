Device Program Profiler
=======================

Overview
--------

Device-side performance profiling is done by annotating device-side code with timestamp markers.

``kernel_profiler::mark_time(uint32_t timer_id)`` is the inline function for storing the execution timestamp of events and associating them with a ``timer_id``.


How to Run
~~~~~~~~~~

Device-side profiling is only allowed in profiler builds. ``scripts/build_scripts/build_with_profiler_opt.sh`` is the script for building profiler builds.

Because downloading profiler results from device adds high runtime overhead, ``TT_METAL_DEVICE_PROFILER=1`` environment variable has to be set to perform the download.

The commands to build and run the ``full_buffer`` example after following :ref:`Getting Started<Getting Started>`:

..  code-block:: sh

    cd $TT_METAL_HOME
    scripts/build_scripts/build_with_profiler_opt.sh
    make programming_examples
    TT_METAL_DEVICE_PROFILER=1 ./build/programming_examples/profiler/test_full_buffer

The generated csv is ``profile_log_device.csv`` and is saved under ``{$TT_METAL_HOME}/generated/profiler/.logs`` by default.


Example
-------

Description
~~~~~~~~~~~

``full_buffer`` is a profiler programming example. It show cases how we can use ``mark_time`` to annotate time on kernel code.

In this example device side profiling is used to measure the cycle count on the series of "nop" instructions that are looped.

The host code of the ``full_buffer`` example is in ``{$TT_METAL_HOME}/tt_metal/programming_examples/profiler/test_full_buffer/test_full_buffer.cpp``

On top of tt_metal's program dispatch API calls, two additional steps specific to this example are taken, which are:

1. Setting ``LOOP_COUNT`` and ``LOOP_SIZE`` defines for the kernels
2. Calling :ref:`DumpDeviceProfileResults<DumpDeviceProfileResults>` after the call to Finish to collect the device side profiling data

The kernel code for full buffer is in ``{$TT_METAL_HOME}/tt_metal/programming_examples/profiler/test_full_buffer/kernels/full_buffer.cpp`` and demonstrated below:

..  code-block:: c++
    :linenos:

    // SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
    //
    // SPDX-License-Identifier: Apache-2.0

    #include <cstdint>

    void kernel_main() {
        for (int i = 0; i < LOOP_COUNT; i ++)
        {
            kernel_profiler::mark_time(5);
    //Max unroll size
    #pragma GCC unroll 65534
            for (int j = 0 ; j < LOOP_SIZE; j++)
            {
                asm("nop");
            }
        }
    }

The inner for loop of "nop" instructions is executed multiple times. The count is determined by the define variable ``LOOP_COUNT`` defined by the host side code.

The beginning of each iteration of the outer loop is timestamped under id number 5.

Deivce-side profiler provides marker information for all riscs and cores used in the program. The following is the portion of the output csv of the ``full_buffer`` test for NCRISC on core 0,0 on device 0:

..  code-block:: c++

    ARCH: grayskull, CHIP_FREQ[MHz]: 1202
    PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]
    0, 0, 0, NCRISC, 1, 161095200021778
    0, 0, 0, NCRISC, 2, 161095200021933
    0, 0, 0, NCRISC, 5, 161095200021976
    0, 0, 0, NCRISC, 5, 161095200022211
    0, 0, 0, NCRISC, 5, 161095200022443
    0, 0, 0, NCRISC, 5, 161095200022675
    0, 0, 0, NCRISC, 5, 161095200022907
    0, 0, 0, NCRISC, 5, 161095200023139
    0, 0, 0, NCRISC, 5, 161095200023371
    0, 0, 0, NCRISC, 5, 161095200023603
    0, 0, 0, NCRISC, 5, 161095200023835
    0, 0, 0, NCRISC, 5, 161095200024067
    0, 0, 0, NCRISC, 5, 161095200024299
    0, 0, 0, NCRISC, 5, 161095200024531
    0, 0, 0, NCRISC, 3, 161095200026549
    0, 0, 0, NCRISC, 4, 161095200026598

ID numbers 1-4 mark default events that are always reported by the device profiler. You can see that additional to default markers, 12 more markers can be recorded on each RISC.

Default markers mark kernel and FW start and end events and are part of the tt_metal device infrastructure. In other words, kernels without any calls to ``kernel_profiler::mark_time(uint32_t timer_id)`` still report these markers.

.. list-table:: Default ID to Event table
   :widths: 15 15
   :header-rows: 1

   * - ID
     - Event
   * - 1
     - FW Start
   * - 2
     - Kernel Start
   * - 3
     - Kernel End
   * - 4
     - FW End

For example, In this run, FW start to Kernel start for the NCRISC took ``161095200021778 - 161095200021933 = 155`` cycles.
Due to non-deterministic HW behaviour, **Profiling overhead** fluctuates. On average, around 40 cycles is from profiling overhead when calculating durations. Kernels typically take 1000s of cycles and so this overhead is negligible.

Post-processing the data on ID number 5 can provide stats on how many cycles the inner loop of "nop" instructions took. The difference between each pair of adjacent ID number 5s denotes the duration of one iteration of the outer loop.

In this example, stats on inner loop durations are:

..  code-block:: c++

               Count  =          6
     Average [cycles] =        232
         Max [cycles] =        235
      Median [cycles] =        232
         Min [cycles] =        232


Limitations
-----------

* Each core has limited L1 buffer for recording device side markers. Space for only 16 total markers is reserved. 12 of the spots are for custom markers and 4 are for default markers. Flip-side of this limitation is that device-side profiling doesn't use L1 space available for kernels.

* The cycle count from RISCs on the same core are perfectly synced as they all read from the same clock counter.

* The cycle counts from RISCs on different cores are closely synced with minor skews, allowing for accurate comparisons on event timestamps across cores.
  **Note** on Grayskull ``tensix_reset`` and ``tt-smi`` soft resets will significantly worsen the skew between core clocks making core to core comparison inaccurate and wrong. Full host
  reboot is required for syncing core clocks if soft reset is used.

* Debug print can not used in kernels that are being profiled.
