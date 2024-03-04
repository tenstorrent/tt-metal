Watcher
*******

Overview
--------

The Watcher consists of two components: one instruments the firmware and kernels to catch common programming errors and
the other launches a thread on the host which periodically monitors the status of the device. If a programming error is
encountered, the Watcher stops the program and reports the error.  If a hang occurs, the data in the log shows the state
of the device at the time of the hang. Watcher functionality includes:

- Logged "waypoints" indicate which piece of code last executed on each RISC V
- Sanitized NOC transactions prevent transactions with invalid coordinates or addresses from being submitted to the
  hardware and stops the offending RISC V via a soft hang
- Memory corruption detection at address 0 of L1
- Kernel paths and names of the currently executing kernel
- Flags which indicate which RISC Vs are executing in the current kernel

Enabling
--------

Enable the Watcher by setting the following environment variables:

.. code-block::

   export TT_METAL_WATCHER=120        # the number of seconds between Watcher updates (longer is less invasive)
   export TT_METAL_WATCHER_APPEND=1   # optional: append to the end of the existing log file (vs creating a new file)
   export TT_METAL_WATCHER_DUMP_ALL=1 # optional: dump all state including unsafe state

Note that ``TT_METAL_WATCHER_DUMP_ALL`` dumps state that can lead to a hang when a kernel is running.  Only set this if
needed and use a time interval large enough to ensure the kernel is stopped when the Watcher polls.

After starting the program, the log messages will indicate when the watcher attaches/detaches from a device and where
the log file is stored as well as a message each time the watcher checks the device.

Details
-------

When enabled the Watcher both dumps status updates to a log file and stops execution if a fatal error (e.g., bad NOC
address) is encountered.  The log file contains one line for each core with a cryptic status.  The top of the log file
includes a legend to help decipher the results.  One datum is the last "waypoint" each of the 5 RISCVs
encountered as a string of up to 4 characters.  These waypoints can be inserted into kernel/firmware code as shown
below:

.. code-block::

    #include "debug/status.h"

    void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
        DEBUG_STATUS('N', 'S', 'W');
        while ((*sem_addr) != val)
            ;
        DEBUG_STATUS('N', 'S', 'D');
    }

Waypoints have no overhead when the watcher is disabled and can be used inside user written kernels.  They indicate
the last code block executed before a hang condition (e.g., waiting for data that never arrives).  This mechanism is
separate from the fault detection mechanism.

The characters in a waypoint name are a mnemonic unique to each waypoint.  By convention, the last character is one of:

- ``W``: waiting at the top of a loop
- ``D``: done waiting after a loop

When dumping state for each RISC V, the Watcher always dumps in the order BRISC, NCRISC, TRISC0, TRISC1, TRISC2.

The path to the log file is printed to the screen during application initialization when the watcher is enabled.

gdb Integration
---------------

The Watcher state can be dumped using ``gdb`` regardless of whether or not the Watcher was enabled; however, if it is
disabled the dumped state won't include the debug only state such as waypoints.  In the example below, gdb's responses
to commands have been removed for brevity.  After attaching to the program and stopping it with ``ctl-c``:

.. code-block::

    thread 1   # make sure the main thread is present
    up         # repeat until in the "tt" namespace (not in, e.g., template library code)
    call tt::watcher::dump(stderr, true) # the "true" at the end enables dumping HW registers

Example
-------

The log file will contain lines such as the following:

.. code-block::

    Core (x=1,y=1):    CWFW,CRBW,R,R,R rmsg:D0G|BNT smsg:GGGG k_ids:4|3|5

- The hang above originated on core (1,1) in physical coords (i.e., the top left core)
- BRISC last hit waypoint ``CWFW`` (CB Wait Front Wait), NCRISC hit ``CRBW`` (NOC CB Reserve Back Wait) and each TRISC
  is in the Run ``R`` state (running a kernel). Look in the source (dataflow_api.h primarily) to decode the obscure names,
  search for ``DEBUG_STATUS``
- The run message ``rmsg`` sent from the host to the device, says the kernel was Device ``D`` dispatched, BRISC is
  using NOC ``0`` (NCRISC is using the other NOC, NOC 1), the host run state is Go ``G`` and each of BRISC, NCRISC and
  TRISC kernels are running (capital ``BNT``; lowercase would signify no kernel running)
- The slave message ``smsg`` sent from BRISC to the other RISC Vs are all Go ``G``; ``D`` would indicate Done
- The kernel IDs ``k_ids`` running are ``4`` on BRISC, ``3`` on NCRISC and ``5`` on TRISC; look further down the log file
  to see the names and paths to those kernels
