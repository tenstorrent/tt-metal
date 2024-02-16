Profiling ttnn Operations
=========================

The following set of commands will generate perf reports for ``resnet`` as an example.

..  code-block:: sh

    cd $TT_METAL_HOME
    scripts/build_scripts/build_with_profiler_opt.sh
    ./tt_metal/tools/profiler/profile_this.py -n resnet -c "pytest models/demos/resnet/tests/test_perf_resnet.py::test_perf_bare_metal[8-0.024-28]"

After the commands finish, the location of the generated csv will be printed on console similar to the image below:

.. image:: ../_static/ops_perf_location_example.png
    :alt: CSV path

The ``-n`` option is used to give a shorter version of the test name to be appended to the CSV file name and be used as the folder name.

The ``profile_this.py`` script and its CLI options are explained under `profile_this description`_.

The headers for the CSV are explained under `Perf Report Headers`_.

**IMPORTANT NOTES**:

- If you have done a reset on your GS device with ``tt_smi`` or ``tensix_reset.sh``, profiling results are not valid due to tensix cores' skewed timer starts. You need to perform a full reboot with ``sudo reboot`` on your host machine to align the timer starts. WH does not have this issue and profiling can be performed after ``tt_smi`` resets.

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

- **DEVICE FW START CYCLE**: Tensix cycle count from the earliest RISC of the earliest core of the device that executed the OP kernel

- **DEVICE FW END CYCLE**: Tensix cycle count from the latest RISC of the latest core of the device that executed the OP kernel

- **DEVICE FW DURATION [ns]**: FW duration on the device for the OP, calculated as (last FW end cycle - first FW start cycle)/core_frequency with cycle markers chosen across all cores and all RISCs

- **DEVICE KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across all cores and all RISCs

- **DEVICE BRISC KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across BRISCs of all cores

- **DEVICE NCRISC KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across NCRISCs of all cores

- **DEVICE TRISC0 KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across TRISC0s of all cores

- **DEVICE TRISC1 KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across TRISC1s of all cores

- **DEVICE TRISC2 KERNEL DURATION [ns]**: Kernel duration on the device for the OP, calculated as (last Kernel end cycle - first Kernel start cycle)/core_frequency with cycle markers chosen across TRISC2s of all cores

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
    - LaunchProgram
    - ReadFromDevice
    - WriteToDevice
    - DumpDeviceProfileResults


profile_this description
------------------------

CLI options of the  ``profile_this.py`` script are:

- ``-c``, ``--command``: This is the required CLI option for providing the test command that has to be profiled

- ``-o``, ``--output-folder``: This option is for providing the output folder for storing the ``csv`` and ``tgz`` files generated by the script. The default output folder is ``{$TT_METAL_HOME}/generated/profiler/reports``

- ``-n``, ``--name-append``: Name to be appended to ``csv`` and ``tgz`` filenames and also be used as the folder name under the given or default output folder

- ``-d``, ``--device-only``: Only profile device side, note in this mode host side readings will still be reported but should be ignored

- ``-m``, ``--host-only``: Only profile host side

This scripts performs the following items:

1. Checks if the project is correctly built with ``PROFILER="enabled"``
2. Executes the provided under test command to generate both host and device side profiling logs
3. Post-processes all the collected logs and aggregate them into the perf csv with a timestamped name.
4. Compress all the raw host and device side logs into a tarball for future reference.
