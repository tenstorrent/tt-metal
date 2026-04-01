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

To capture performance counters alongside profiling data:

..  code-block:: sh

    # Using environment variables directly
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILE_PERF_COUNTERS=47 \
        pytest your_test.py -x -v

    # Process the results
    python tools/tracy/process_ops_logs.py --device-only

The ``TT_METAL_PROFILE_PERF_COUNTERS`` value is a bitfield selecting which counter groups to capture:

- ``1`` = FPU (compute utilization)
- ``2`` = PACK (packer activity)
- ``4`` = UNPACK (unpacker activity, math pipeline)
- ``8`` = L1_0 (L1 memory ports 0-7: unpacker, packer, TDMA, NOC Ring 0)
- ``16`` = L1_1 (L1 memory ports 8-15: extended unpacker, NOC Ring 1)
- ``32`` = INSTRN (instruction availability, stalls, issue counts per thread)
- ``47`` = all of the above (recommended starting point)

**Note**: L1_0 and L1_1 share hardware mux state and cannot be captured simultaneously in a single run. Use separate runs if both are needed.

Blackhole-only groups:

- ``64`` = L1_2 (NOC Ring 2 ports)
- ``128`` = L1_3 (NOC Ring 3 ports)
- ``256`` = L1_4 (misc L1 ports)

**Output**

The profiler produces two types of output:

1. **Console**: Raw counter values per counter type, followed by derived efficiency metrics with Min/Median/Max/Avg statistics across cores.
2. **CSV** (``generated/profiler/reports/ops_perf_results.csv``): Derived metrics only, with Min/Median/Max/Avg columns per operation.

**Derived Metrics Reference**

The following metrics are automatically computed from raw counters. Each metric appears in the CSV and console output with Min, Median, Max, and Avg aggregations across all cores for each operation.

*Compute Utilization*

- **SFPU Util (%)**: Fraction of cycles the SFPU was executing a valid operation. Higher is better for SFPU-heavy ops (e.g. sqrt, gelu).
- **FPU Util (%)**: Fraction of cycles the FPU was executing. Higher is better for FPU-heavy ops (e.g. matmul).
- **MATH Util (%)**: Fraction of cycles either FPU or SFPU was active (combined). Measures total math unit utilization.

*Pipeline Efficiency*

- **Packer Efficiency (%)**: Fraction of packer busy cycles where the destination register read was available. 100% means the packer never waited for data from the math unit.
- **Math-to-Pack Handoff Efficiency (%)**: How quickly math results reach the packer. Values >100% indicate math completes faster than the packer consumes.
- **Unpacker-to-Math Data Flow (%)**: Ratio of source register write availability to unpacker busy time. Higher means data flows smoothly from unpack to math.

*Thread Analysis*

- **Thread 0/1/2 Stall Rate (%)**: Fraction of cycles each thread was stalled. Thread 0 = unpack, Thread 1 = math, Thread 2 = pack. High values indicate the thread is waiting for a resource.
- **Thread 0/1/2 IPC**: Instructions per cycle for each thread. Higher means more instructions issued per clock.

*Pipeline Wait Metrics*

- **SrcA/SrcB Valid Wait (%)**: Cycles waiting for source register data to become valid (data from unpacker not yet ready).
- **SrcA/SrcB Clear Wait (%)**: Cycles waiting for source register to be cleared (previous math operation still using it).
- **Math Idle Wait T1 (%)**: Cycles math thread waited for the math unit to become idle.
- **Pack Idle Wait T2 (%)**: Cycles pack thread waited for pack hardware.
- **Unpack Idle Wait T0 (%)**: Cycles unpack thread waited for unpack hardware.

*Semaphore Waits*

- **Semaphore Zero Wait T0/T1/T2 (%)**: Cycles each thread waited for a semaphore to become non-zero (waiting for producer).
- **Semaphore Full Wait T0/T1/T2 (%)**: Cycles each thread waited for a semaphore to become non-full (waiting for consumer).

*TDMA Stall Metrics*

- **Data Hazard Stall Rate (%)**: Cycles stalled by dest-to-src data hazards (MOVD2A/MOVD2B). High values in matmul are expected.
- **Fidelity Phase Overhead (%)**: Cycles spent on HiFi fidelity phases. Higher fidelity = higher accuracy but more overhead.
- **SrcA Write Port Blocked Rate (%)**: Cycles where DMA to srcA was blocked by overwrite protection.
- **Dest Read Backpressure (%)**: Cycles where destination register read was requested but not granted.
- **Math Dest Write Port Stall Rate (%)**: Cycles where math was stalled by destination register write port contention.
- **Math Scoreboard Stall Rate (%)**: Cycles where math was stalled by FPU data hazard scoreboard.

*Instruction Availability*

- **CFG/SYNC/THCON/MOVE Instrn Avail Rate T0 (%)**: Fraction of cycles each instruction type was available in thread 0's instruction buffer. Shows which instruction types occupy the most scheduling time.
- **MATH Instrn Avail Rate T1 (%)**: Math instruction availability on the math thread.
- **UNPACK/PACK Instrn Avail Rate T0/T2 (%)**: Unpack and pack instruction availability on their primary threads.

*Stall Breakdown*

- **THCON/MOVE Idle Stall Pct T0 (%)**: What percentage of thread 0's total stalls are caused by THCON or MOVE waits. Helps identify the dominant stall reason.
- **MMIO/SFPU Idle Stall Pct T1 (%)**: What percentage of thread 1's stalls are MMIO or SFPU waits.

*Write Port Analysis*

- **SrcB Write Port Blocked Rate (%)**: Cycles where srcB DMA write was blocked by write port unavailability.
- **SrcA Write Actual Efficiency (%)**: Fraction of srcA write attempts that succeeded. 100% = no blocking.

*L1 Memory Utilization*

- **L1 Unpacker/Packer Port Util (%)**: Fraction of cycles the unpacker or packer L1 port had a transaction.
- **L1 TDMA Bundle Util (%)**: Average utilization of the two TDMA/RISC L1 ports.
- **NOC Ring 0/1 Outgoing/Incoming Util (%)**: Average utilization of NOC channels on each ring.
- **RISC Core L1 Util (%)**: RISC core L1 access utilization (Blackhole only, requires L1_1 group).

*L1 Backpressure*

- **NOC Ring 0/1 Outgoing/Incoming Backpressure (%)**: Fraction of NOC transaction cycles where L1 was not ready. Higher = more contention.
- **L1 Unpacker/Packer Port Backpressure (%)**: L1 port contention for unpacker and packer.

*L1 Composite Metrics*

- **L1 Total Bandwidth Util (%)**: Sum of all 8 L1 port request cycles divided by theoretical maximum (8 ports x ref_cnt). Shows overall L1 saturation.
- **L1 Read vs Write Ratio (%)**: Read port traffic as a fraction of total traffic. 50% = balanced, >50% = read-heavy.
- **NOC Ring 0 Asymmetry (%)**: Outgoing traffic as a fraction of total NOC Ring 0 traffic. 50% = balanced send/receive.
- **L1 Contention Index (%)**: Average backpressure across all active L1 ports. Single number summarizing L1 memory stress.
- **Unpacker L1 Efficiency (%)**: When the unpacker is busy, how often does L1 actually serve it.
- **Packer L1 Efficiency (%)**: When the packer is busy, how often does L1 serve it.
- **NOC vs Compute Balance (%)**: NOC cycles as a fraction of NOC + FPU cycles. >50% = NOC-bound, <50% = compute-bound.
- **TDMA vs NOC L1 Share (%)**: RISC/TDMA traffic as a fraction of all L1 traffic. Shows how much bandwidth goes to firmware vs NOC.

*Wormhole-Only Metrics*

These metrics depend on hardware signals inactive on Blackhole and are automatically hidden from BH output.

- **Math Pipeline Utilization (%)**: Math instruction flow efficiency (MATH_INSTRN_STARTED / MATH_INSTRN_AVAILABLE).
- **Math Src Data Ready Rate (%)**: Cycles where math was not blocked by source data.
- **SrcB Write Actual Efficiency (%)**: Fraction of srcB write attempts that succeeded.
- **HiFi2/LoFi/HiFi4 Instrn Rate (%)**: Fraction of math instructions at each fidelity level.
- **Packer Engine 0/1/2 Util (%)**: Per-engine packer utilization (Wormhole has 4 packer engines).
- **Unpacker0/1 Write Efficiency (%)**: Source register write throughput per unpacker.
- **FPU Execution Efficiency (%)**: FPU active cycles as fraction of math instruction availability.

*Additional Idle Waits*

- **MMIO/SFPU/THCON/MOVE Idle Wait T0/T1 (%)**: Fraction of total cycles each thread spent waiting for specific hardware units.

**Architecture Differences**

Wormhole collects 172 raw counters and computes 86 derived metrics. Blackhole collects 190 raw counters (126 Tensix + 64 ERISC) and computes 74 derived metrics. The difference is due to TDMA signals that are inactive on Blackhole (``PACK_COUNT=1``, ``o_math_instrnbuf_rden`` tied off). Metrics that depend on dead signals are automatically hidden from the Blackhole output. Blackhole has additional L1 mux positions (3 extra for Tensix, 3 for Ethernet) providing deeper memory visibility.

For detailed hardware counter documentation including register maps and signal definitions, see ``tech_reports/PerfCounters/perf-counters.md``.


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
