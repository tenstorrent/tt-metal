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

Lossless trace artifacts
------------------------

Use the non-dropping debug mode together with Tracy NoC trace collection to retain full transfer sizes and include
the resulting artifacts in the timestamped report:

.. code-block:: bash

    TT_METAL_NOC_DEBUG_DUMP=1 python -m tracy -p -r -v \
        --collect-noc-traces \
        --sync-host-device \
        -m pytest <test> -sv

The report contains:

``lossless_noc_events.jsonl``
    One JSON object per recorded transaction. Each object contains ``operation.runtime_id``, ``operation.trace_id``,
    ``operation.trace_replay_session_id``, ``operation.name``, ``device_id``, ``core.x``, ``core.y``, ``risc``,
    ``issue_timestamp``, ``type``, ``noc``, ``vc``, ``destinations``, the full 32-bit ``num_bytes``, and
    ``debug_metadata``. The debug metadata contains the available ``posted``, ``src_addr``, ``dst_addr``, and
    ``counter`` values. ``destinations`` is an array: it is empty for destination-less events, contains one
    ``{"x", "y"}`` object for unicast, or contains one object with ``start`` and ``end`` coordinates for multicast.
    Destination coordinates are normalized to the NOC0 coordinate system; ``noc`` identifies the issuing NoC.

``lossless_noc_manifest.json``
    Describes the capture and its timestamp semantics:

.. code-block:: json

    {
      "schema_version": 1,
      "capture_mode": "non_dropping",
      "complete": true,
      "timestamp_semantics": "issue_cycles",
      "device_frequency_mhz": 1000,
      "events": {
        "path": "lossless_noc_events.jsonl",
        "count": 42
      },
      "npe_modeled_semantics": {
        "input": "noc_trace*.json",
        "input_timestamps": "issue_cycles",
        "completion_timestamps": "modeled_by_tt_npe"
      }
    }

``issue_timestamp`` is the unadjusted device cycle at which the transfer was issued. Transfer completion is not
measured by this capture. The existing ``noc_trace*.json`` files remain the input to tt-npe, which models completion
timestamps. Lossless capture timing does not replace the standard profiler timing used by
``profile_log_device.csv`` or ``ops_perf_results*.csv``.

Limitations
-----------

- Not all issues can be detected due to the non-deterministic nature of the NOC. Acknowledgement of reads/writes can be returned before the trace can detect a missing barrier.
- There is overhead on the H2D and D2H path due to host data transfers, and an additional 1-15% kernel cycle overhead.
- Non-dropping capture reserves four profiler markers per local NoC event and continuously drains profiler buffers.
  The overhead and artifact volume can be substantial for long-running or highly iterative workloads.
- Watcher, the normal dropping profiler mode, and DPrint cannot be combined with non-dropping NoC debug capture due
  to kernel size and profiler-buffer constraints.
