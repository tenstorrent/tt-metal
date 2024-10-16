Profiling ttnn Operations
=========================

The following set of commands will generate perf reports for ``resnet`` as an example.

..  code-block:: sh

    cd $TT_METAL_HOME
    build_metal.sh --enable-profiler
    ./tt_metal/tools/profiler/profile_this.py -n resnet -c "pytest models/demos/resnet/tests/test_perf_resnet.py::test_perf_bare_metal[20-0.0185-25]"

After the commands finish, the location of the generated csv will be printed on console similar to the image below:

.. image:: ../_static/ops_perf_location_example.png
    :alt: CSV path

The ``-n`` option is used to give a shorter version of the test name to be appended to the CSV file name and be used as the folder name.

The ``profile_this.py`` script and its CLI options are explained under `profile_this description`_.

The headers for the CSV are explained under `Perf Report Headers`_.

**IMPORTANT NOTES**:

- If this is the first time you are running ``profile_this.py``, it requires `developer dependencies <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md#step-4-installing-developer-dependencies>`_ to be installed.
- If you have done a reset on your GS device with ``tt_smi`` or ``tensix_reset.sh``, profiling results are not valid due to tensix cores' skewed timer starts. You need to perform a full reboot with ``sudo reboot`` on your host machine to align the timer starts. WH does not have this issue and profiling can be performed after ``tt_smi`` resets.

- In order to populate program cache, tests should run their inference layer at least twice and should run it in the same process. If pytest is being used, that would be running in
  the same pytest run. Only the host times for the second run of the layer should be analyzed as the first run was populating the cache and will have much higher times for host side.

- The first 1000 ops for each device is automatically collected by pytest fixtures at the end of your test.
  If your test has more than 1000 ops, ``ttl.device.DumpDeviceProfiler(device)`` should be called at every n number of layers that total to less than 1000 ops in order to avoid dropping profiling data of new ops.
  For example for resnet with around 120 ops for a single inference layer, if the test calls the layer more than 8 times, ``ttl.device.DumpDeviceProfiler(device)`` should be called at least every eighth layer run.
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


profile_this description
------------------------

CLI options of the  ``profile_this.py`` script are:

- ``-c``, ``--command``: This is the required CLI option for providing the test command that has to be profiled

- ``-o``, ``--output-folder``: This option is for providing the output folder for storing the ``csv`` and ``tgz`` files generated by the script. The default output folder is ``{$TT_METAL_HOME}/generated/profiler/reports``

- ``-n``, ``--name-append``: Name to be appended to ``csv`` and ``tgz`` filenames and also be used as the folder name under the given or default output folder

This scripts performs the following items:

1. Executes the provided under test command to generate both host and device side profiling logs
2. Post-processes all the collected logs and aggregate them into the perf csv with a timestamped name.
