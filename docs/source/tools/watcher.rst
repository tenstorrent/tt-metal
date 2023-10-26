Watcher
=======

The Watcher is a thread that monitors the status of the TT device to help with
debug.  It:

- logs "waypoints" during execution on each core to help determine where a
  hang occured
- watches L1 address 0 to look for memory corruption
- sanitizes all NOC transactions and reports invalid X,Y and addresses.  Further,
  any core with an invalid transaction will soft hang at the point of that
  transaction

It is enabled with:

- ``export TT_METAL_WATCHER=<n>`` where <n> is the number of seconds between status updates; use 0 for the default
- optionally ``export TT_METAL_WATCHER_APPEND=1`` to append to the end of the file instead (useful for tests which construct/destruct devices in a loop)
- optionally ``export TT_METAL_WATCHER_DUMP_ALL=1`` to dump, eg, CB sync registers.  Note that reading these registesrs while a kernel is running can induce a hang so only set this if needed and use a time interval large enough to ensure the kernel is hung when the watcher polls

When enabled, the watcher both dumps status updates to a log file and stops
execution if a fatal error (e.g., bad NOC address) is encountered.  The log
file contains one line for each core with a cryptic status.  The top of the
log file includes a legend to help with deciphering the results.  One datum
is the last "waypoint" each of the 5 riscvs encountered as a string of up to 4
characters.  These way points can be inserted into kernel/firmware code with
the following, eg:

- ``DEBUG_STATUS('I');``
- ``DEBUG_STATUS('D', 'E', 'A', 'D');``

gdb Integration
---------------

Watcher information can be dumped using ``gdb``:

- Stop in gdb w/ ``ctl-c``
- Make sure the main thread is current, try ``thread 1``
- If needed, walk up the call stack until TT symbols are available
- ``call tt::llrt::dump(stderr, true)``.  The ``true`` at the end enables dumping registers

Examples:
---------

.. code-block::

    Core (x=1,y=1):    CWFW,CRBW,R,R,R rmsg:D0G|BNT smsg:GGGG k_ids:4|3|5

- The hang above originated on core (1,1) in physical coords (ie, the top left
  core)
- BRISC last hit way point CWFW (CB Wait Front Wait), NCRISC hit CRBW (NOC
  CB Reserve Back Wait) and each Trisc is in the Run state (running a kernel).
  Look in the source (dataflow_api.h primarily) to decode the obscure
  names, search for DEBUG_STATUS
- The run message (rmsg) says the kernel was Device dispatched (D), brisc is
  using NOC 0, the host run state is Go (G) and each of Brisc, NCrisc and
  Trisc kernels are running (capital BNT.  lowercase signfies not kernel
  running).
- The slave messages (smsg) are all Go
- The kernel ids that are running are 4 on Brisc, 3 on NCrisc and 5 on Trisc,
  look futher down the log file to see the names and paths to those kernels.
