Profiling TT-NN Operations
==========================

The following set of commands will generate perf reports for ``bert_tiny`` as an example.

..  code-block:: sh

    cd $TT_METAL_HOME
    build_metal.sh
    ./tools/tracy/profile_this.py -n bert_tiny -c "pytest models/demos/wormhole/bert_tiny/demo/demo.py::test_demo"

After the commands finish, the location of the generated csv will be printed on console similar to the image below:

.. image:: ../_static/ops_perf_location_example.png
    :alt: CSV path

The ``-n`` option is used to give a shorter version of the test name to be appended to the CSV file name and be used as the folder name.

The ``profile_this.py`` script and its CLI options are explained under `profile_this description`_.

The headers for the CSV are explained under `Perf Report Headers`_.

Instructions on using the performance report with `TT-NN Visualizer <https://github.com/tenstorrent/ttnn-visualizer>`_ can be found in their documentation under `Loading Data <https://docs.tenstorrent.com/ttnn-visualizer/src/installing.html#loading-data>`_.

**IMPORTANT NOTES**:

- If this is the first time you are running ``profile_this.py``, it requires `developer dependencies <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md#step-4-installing-developer-dependencies>`_ to be installed.
- If you have done a reset on your GS device with ``tt_smi`` or ``tensix_reset.sh``, profiling results are not valid due to tensix cores' skewed timer starts. You need to perform a full reboot with ``sudo reboot`` on your host machine to align the timer starts. WH does not have this issue and profiling can be performed after ``tt_smi`` resets.

- In order to populate program cache, tests should run their inference layer at least twice and should run it in the same process. If pytest is being used, that would be running in
  the same pytest run. Only the host times for the second run of the layer should be analyzed as the first run was populating the cache and will have much higher times for host side.

- The first 1000 ops for each device is automatically collected by pytest fixtures at the end of your test.
  If your test has more than 1000 ops, ``ttl.device.ReadDeviceProfiler(device)`` should be called at every n number of layers that total to less than 1000 ops in order to avoid dropping profiling data of new ops.
  For example for a model with around 120 ops for a single inference layer, if the test calls the layer more than 8 times, ``ttl.device.ReadDeviceProfiler(device)`` should be called at least every eighth layer run.
  If profiling data is dropped, you will receive warning messages in the execution log mentioning which RISC of what core of what device dropped profiling data. Note that dispatch
  cores fill up their profiling buffers faster and if only those cores are giving warnings your OP analysis is not affected.

Perf Report Headers
-------------------

The OPs profiler report demonstrates the execution flow of the OPs in the pipeline. Each row in the CSV represents an OP executed.

For each OP, multiple data points are provided in the columns of the CSV.

The headers of the columns with their descriptions is below:

- **OP CODE**: Operation name, for C++ level OPs this code is the name of the class for the OP

- **OP TYPE**: Operation type, where the op ran and which part of code it is coming from

    - *python_fallback*: OP fully implemented in python and running on CPU
    - *tt_dnn_cpu*: OP implemented in C++ and running on CPU
    - *tt_dnn_device*: OP implemented in C++ and running on DEVICE

- **GLOBAL CALL COUNT**: The index of the op in the execution pipeline

- **DEVICE ID**: ID of the device the operation ran on

- **ATTRIBUTES**: Operation attributes

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

- **DEVICE FW START CYCLE**: Tensix cycle count from the earliest RISC of the earliest core of the device that executed the OP kernel

- **DEVICE FW END CYCLE**: Tensix cycle count from the latest RISC of the latest core of the device that executed the OP kernel
- **DEVICE FW DURATION [ns]**: FW duration on the device for the OP, calculated as (last FW end cycle - first FW start cycle)/core_frequency with cycle markers chosen across all cores and all RISCs

- **DEVICE KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across all cores and all RISCs

- **DEVICE BRISC KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across BRISCs of all cores

- **DEVICE NCRISC KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across NCRISCs of all cores

- **DEVICE TRISC0 KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across TRISC0s of all cores

- **DEVICE TRISC1 KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across TRISC1s of all cores

- **DEVICE TRISC2 KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across TRISC2s of all cores

- **DEVICE COMPUTE CB WAIT FRONT [ns]**: Total time spent on ``cb_wait_front`` on TRISC0, averaged across all cores

- **DEVICE COMPUTE CB RESERVE BACK [ns]**: Total time spent on ``cb_reserve_back`` on TRISC2, averaged across all cores

- **COMPUTE KERNEL PATH**: Path of the compute kernels in the program

- **COMPUTE KERNEL HASH**: Kernel hash for compute kernel cache

- **DATAMOVEMENT KERNEL PATH**: Path of the datamovement kernels in the program

- **DATAMOVEMENT KERNEL HASH**: Kernel hash for datamovement kernel cache

- **Input & Output Tensor Headers**: Header template is {Input/Output}_{IO Number}_{Field}. e.g. INPUT_0_MEMORY

    - *SHAPE*
        - W: Tensor batch count
        - Z: Tensor channel count
        - Y: Tensor Height
        - X: Tensor Width
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


Hardware Performance Counters
-----------------------------

Tenstorrent devices contain hardware performance counters that measure cycle-level events inside each Tensix core. These counters provide visibility into compute utilization, memory traffic, instruction pipeline stalls, and NOC bandwidth that is not available from kernel-level timestamps alone.

**Quick Start**

To capture performance counters alongside profiling data, use the ``python -m tracy`` CLI with the ``--profiler-capture-perf-counters`` option:

..  code-block:: sh

    python -m tracy --profiler-capture-perf-counters=fpu,pack,l1_0 \
        -m "pytest your_test.py -x -v"

Available counter groups:

- ``fpu`` — compute utilization (FPU, SFPU, math counters)
- ``pack`` — packer activity (dest read, packer busy, scoreboard)
- ``unpack`` — unpacker activity, math pipeline stalls, source register writes
- ``l1_0`` — L1 memory ports 0-7 (unpacker, packer, TDMA, NOC Ring 0)
- ``l1_1`` — L1 memory ports 8-15 (extended unpacker, NOC Ring 1)
- ``instrn`` — per-thread instruction availability, stalls, and issue counts
- ``all`` — the architecture's full set (needs multiple passes, see below)

Blackhole-only groups: ``l1_2``, ``l1_3``, ``l1_4`` (extended L1 client ports); on Blackhole ``all`` includes them.

**Multi-pass capture**: two limits cap what one run can measure — the BRISC firmware image only fits the readout code for 3 counter groups, and the L1 banks share a hardware mux, so at most one L1 bank counts per run. ``python -m tracy`` schedules the passes automatically: a request that fits one pass runs once as before, and a larger request (such as ``all``) stops with the printed pass plan unless ``--perf-counter-multipass`` is given, in which case the workload is replayed once per pass and the per-pass results are merged.

..  code-block:: sh

    python -m tracy --perf-counter-multipass \
        --profiler-capture-perf-counters=all \
        -m "pytest your_test.py -x -v"

The ``--perf-counter-multipass`` option and the arch-wide expansion of ``all`` land with tt-metal PR #55166; until it merges, request at most three groups and one L1 bank per run.

**Output**

The profiler generates the standard ops performance CSV at ``generated/profiler/reports/ops_perf_results.csv`` with additional columns for perf counter metrics. Console output also includes raw counter values and derived efficiency metrics with Min/Median/Max/Avg statistics across cores per operation.

**Derived Metrics Reference**

Derived metrics are computed per operation and per core, then aggregated to Min, Median, Max, and Avg columns across the cores of each operation in the CSV and console output. The formulas live in one shared module, ``tools/tracy/perf_metrics_common.py`` (also used by the tt-llk test harness), and the complete catalogue of all 106 metrics, grouped by topic with formulas and notes, is in the `PerfCounters tech report <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/PerfCounters/perf-counters.md#derived-metrics-reference>`_.

Two metric families appear in the output:

- Bounded percentages, with a ``(%)`` unit: utilizations, efficiencies, and stall or wait rates in 0-100%.
- Unbounded ratios, with a ``(ratio)`` unit and a raw value: ``Math-to-Pack Handoff Efficiency`` (above 1 the packer is the handoff bottleneck), ``Compute-to-Unpack Ratio`` (above 1 = compute-bound), ``Unpacker/Packer L1 Efficiency`` (above 1 = ample L1 bandwidth), and ``Stall Overlap T0/T1/T2`` (above 1 = several waits overlap in the same cycle). These are never clamped; the excess over 1 is the signal.

A metric whose counters do not exist on the running architecture reports N/A (blank), never 0: the Wormhole-only per-engine packer metrics (Packer Engine 0/1/2 Util, Packer Load Imbalance) are N/A on Blackhole, and the Blackhole-only extended L1 metrics (L1 Ext Packer Util/Backpressure, L1 Tag Search Util/Backpressure) are N/A on Wormhole. Cross-bank metrics are also N/A when one of their counter groups was not captured in the run.

**Architecture Differences**

Wormhole and Blackhole expose different raw hardware signals:

- ``PACK_COUNT=1`` on Blackhole ties the per-engine packer busy and dest-read signals for engines 1-3 to constants, so per-engine packer metrics (Packer Engine 0/1/2 Util, Packer Load Imbalance) are WH-only.
- Blackhole has additional L1 mux positions (3 extra for Tensix) providing deeper memory visibility through ``l1_2``, ``l1_3``, ``l1_4`` counter groups.
- ``Math-to-Pack Handoff Efficiency`` falls back to the bank's reference cycles as denominator when ``PACKER_BUSY = 0`` on a given op (e.g. pure-SFPU ops); ``Packer Efficiency`` reports N/A there.

For the authoritative per-architecture metric list, raw counter set, register maps, and signal definitions, see ``tech_reports/PerfCounters/perf-counters.md``.


profile_this description
------------------------

CLI options of the  ``profile_this.py`` script are:

- ``-c``, ``--command``: This is the required CLI option for providing the test command that has to be profiled

- ``-o``, ``--output-folder``: This option is for providing the output folder for storing the performance report folder created. The default output folder is ``${TT_METAL_HOME}/generated/profiler/reports``

- ``-n``, ``--name-append``: Name to be appended to the the performance report folder and its files

- ``--collect-noc-traces``: Specifying this option will also create timeline files using `tt-npe <https://github.com/tenstorrent/tt-npe>`_ in a subdirectory named ``npe_viz`` under the the perf report folder. These are used in the NPE tab on TT-NN Visualizer to visualize NoC traffic and congestion.
  **Note**: This option requires that npe is properly installed (See `here <https://github.com/tenstorrent/tt-npe/blob/main/docs/src/getting_started.md#quick-start>`_ for instructions).

This scripts performs the following items:

1. Executes the provided under test command to generate both host and device side profiling logs
2. Post-processes all the collected logs and aggregate them into the perf csv with a timestamped name (e.g. ``ops_perf_results_2025_06_25_14_04_34.csv``)

Using the Performance Report with TT-NN Visualizer
--------------------------------------------------

The perf report should be created under a folder with a timestamped name (e.g. ``2025_06_25_14_04_34``) and look like the following (the ``npe_viz`` subdirectory only exists if ``--collect-noc-traces`` is specified):

.. image:: ../_static/tracy_perf_report.png
    :alt: Tracy performance report

This folder can be uploaded under the Reports tab in `TT-NN Visualizer <https://github.com/tenstorrent/ttnn-visualizer>`_:

.. image:: ../_static/ttnn_visualizer_perf_report_upload.png
    :alt: TT-NN Visualizer performance report upload

The uploaded data can then be viewed in the Performance tab:

.. image:: ../_static/ttnn_visualizer_performance.png
    :alt: TT-NN Visualizer Performance analysis

and NPE tab (if ``--collect-noc-traces`` was used):

.. image:: ../_static/ttnn_visualizer_npe.png
    :alt: TT-NN Visualizer NPE
