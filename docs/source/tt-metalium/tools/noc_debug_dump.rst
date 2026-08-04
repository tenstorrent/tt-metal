NOC Debug Dump
==================

.. note::
   Tools are only fully supported on source builds.

.. caution::
    This is an experimental feature.

Overview
--------

The host can collect NOC traces from the device to identify potential kernel programming issues such as missing NOC barriers.

Each NOC transaction is instrumented to record metadata such as type, src/dst, NOC counters, and size. These packets are collected by the host and bucketed into events per core and RISC processor. As the host collects the trace, it compares it to previous traces as well as traces on other cores on the same device.

When the program finishes, the device closes, or ``tt::tt_metal::detail::ReadDeviceProfilerResults`` is manually called, the host will analyze the trace and print out any issues found grouped by core.


Enabling
--------

This feature is enabled by setting the environment variable ``TT_METAL_NOC_DEBUG_DUMP=1`` before running your application.

No kernel changes are needed to enable this feature. Trace collection is instrumented automatically.

.. note::
    Watcher, Profiler, or DPrint cannot be enabled at the same time as this feature due to kernel size constraints.

Example
-------

This unit test demonstrates the feature in action by running a kernel that issues a multicast write followed by a multicast semaphore increment with a missing write barrier afterward.

.. code-block:: bash

    TT_METAL_NOC_DEBUG_DUMP=1 build/test/tt_metal/unit_tests_noc_debugging --gtest_filter=NOCDebuggingFixture.McastOnlyWriteFlush

The output is printed to the console.

.. code-block::

    Running test on device 0
    ========== NOC Debug Summary ==========
    Unflushed async writes at kernel end (missing noc_async_write_barrier):
        Device 0 (18,18) Processor 0 [semaphore mcast]
    =======================================

    Finished running test on device 0.
    Running test on device 1.
    ========== NOC Debug Summary ==========
    Unflushed async writes at kernel end (missing noc_async_write_barrier):
    Device 0 (18,18) Processor 0 [semaphore mcast]
    Device 1 (18,18) Processor 0 [semaphore mcast]
    ========================================

    Finished running test on device 1.
    ========== NOC Debug Summary ==========
    Unflushed async writes at kernel end (missing noc_async_write_barrier):
    Device 0 (18,18) Processor 0 [semaphore mcast]
    Device 1 (18,18) Processor 0 [semaphore mcast]
    ========================================

    ========== NOC Debug Summary ===========
    Unflushed async writes at kernel end (missing noc_async_write_barrier):
    Device 0 (18,18) Processor 0 [semaphore mcast]
    Device 1 (18,18) Processor 0 [semaphore mcast]
    ========================================

Limitations
-----------

- Not all issues can be detected due to the non-deterministic nature of the NOC. Acknowledgement of reads/writes can be returned before the trace can detect a missing barrier.
- There is overhead on the H2D and D2H path due to host data transfers, and an additional 1-15% kernel cycle overhead.
