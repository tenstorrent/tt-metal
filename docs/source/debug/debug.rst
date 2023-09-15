Debug Tools
#########################


tt-gdb
****************************************
.. raw:: html

    <video width="800" height="600" controls>
        <source src="../_static/DebuggerTutorial.mp4" type="video/mp4">
    <p>Your browser does not support the video tag.</p>
    </video>

Enabling the breakpoint server
**********************************
To enable the breakpoint server, in your test, you must include `tools/tt_gdb/tt_gdb.hpp`, and call it right before calling your test function.

For example, below is a snippet of enabling tt_gdb in `test_run_datacopy`.

.. code-block:: cpp

    const vector<CoreCoord> cores = {core};
    const vector<string> ops = {op};

    tt_gdb(cluster, chip_id, cores, ops);
    pass &= run_data_copy_multi_tile(cluster, chip_id, core, 2048);

where cluster was defined earlier in the file.

Installing python packages
**************************

Within the ``tools/tt_gdb`` directory,
``pip3 install -r requirements.txt``

Compiling under debug mode
**************************
To get the most out of `tt_gdb`, you must compile under debug mode. This will use different debug ld files for the riscs, and will enable debug symbols. To enable under debug mode, use
`DEBUG_MODE=1` when compiling a test.

For example, below is a snippet of compiling with `DEBUG_MODE=1` for compilation of the datacopy op.

``DEBUG_MODE=1 ./build/test/build_kernels_for_riscv/tests/test_compile_datacopy``

Limitations
***********
So far, debug mode only compiles BRISC with proper debug symbols, therefore the `p` functionality will only work for BRISC; however, the breakpoint function should work for any RISC if your goal is just to hang a program.

Additionally, unit testing needs to be added for the debugger. If you have any issues,
please create an issue under the debugger board.

## Using breakpoints in your code
`breakpoint` is a macro part of `riscv/common`, which means that you don't need to include any special header files in your kernels. It compiles differently for each of brisc, ncrisc, and triscs, which means you can use it generically. Just insert `breakpoint();` into your code wherever you need it.

For example, below is a dummy brisc kernel running on core 1-1:

.. code-block:: cpp

    #include "dataflow_api.h"

    void kernel_main() {
        volatile uint32_t dst_addr  = get_arg_val<uint32_t>(0);
        uint32_t dst_noc_x = get_arg_val<uint32_t>(1);
        uint32_t dst_noc_y = get_arg_val<uint32_t>(2);
        uint32_t num_tiles = get_arg_val<uint32_t>(3);

        constexpr uint32_t cb_id_out0 = 16;

        // single-tile ublocks
        uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
        uint32_t ublock_size_tiles = 1;




        breakpoint();
        ...


When I run my test, as soon as this breakpoint is reached, it opens up an interactive debugger. You will be greeted with this prompt:

.. image:: ../_static/grid-debugger.png
    :width: 400
    :alt: First login


The blue core represents the core you are currently hovering over, and the red cores represent cores in which a breakpoint was hit. If you are hovering over a core that has a breakpoint, it will appear as grey.

You can move around the grid with the arrow keys, and when you would like to debug a particular core, press enter on that core. For example, if I move to core 1-1 and press enter, I will see this prompt:

.. image:: ../_static/core-view.png
    :width: 400
    :alt: First login

Gray will represent your current cursor, so in this image, we are hovering over trisc0. Just like before, you can use the arrow keys to move around. Since we hit a breakpoint for brisc, you will see a message notifying you that a breakpoint has been hit, as well as which line the breakpoint was on.

TODO(agrebenisan): In the future, would like to make this a hyperlink that brings you directly to the file and line number.

From this screen, we can select a particular thread to debug. For example, if we move to the brisc thread and press enter, we will see this prompt:

.. code-block:: bash

    (tt_gdb)

Here, you can write `help` or `h` to display available commands:

.. code-block:: bash

    (tt_gdb) help
    Documented commands (type help <topic>):
    ========================================

    c       e       h    help    p       q


You can get more info by following the above instructions, however here is one example of printing a local variable:

.. code-block:: bash

    (tt_gdb) p dst_addr
    536870912

You may notice I specifically chose to print out a volatile variable, since volatile variables live in L1. So far, non-L1 variable printing is not supported, so you may need to modify your code to make local variables volatile for the time being.

|

|

Kernel printf()
********************************


The codebase supports debug prints from device kernels that get displayed on the host. On device the APIs are defined in src/firmware/riscv/common/debug_print.h.
To use debug printing capability, it is first required to start the debug print server on the host - use the env variables defined below.

*Basic use steps:*
------------------

- Include the device side header ``#include "debug_print.h"`` in your kernel.
- Use ``DPRINT << "string" << 1 << SETW(4) << F32(2.0f) << ENDL();`` std::cout-style syntax to print values.
- For the prints to show up on the host, it is first required to start the host-side server using:

  - ``export TT_METAL_DPRINT_CORES=<list of cores>``
  - ``<list of cores>`` is one of:

    - single core: ``x,y``
    - multiple cores: ``"(x,y),(x,y)"`` (etc)
    - range of cores: ``"(x,y)-(x,y)"`` (bounding box, inclusive)

- Optionally (for multi-chip parts) ``export TT_METAL_DPRINT_CHIPS=<list of chips>``

  - use a comma separated list
  - default is 0

- Optionally ``export TT_METAL_DPRINT_RISCV=<riscv>``

  - ``<riscv>`` is one of:

    - ``NC``, ``BR``, ``TR0``, ``TR1``, ``TR2``

  - default is all riscvs

- Optionally ``export TT_METAL_DPRINT_FILE=<filename>``
- Note that the core coordinates are currently NOC coordinates (not logical).
- Since on TRISCs the same code compiles 3 times and executes on 3 threads, you can use the pattern ``MATH(( DPRINT << x << ENDL() ));`` to make prints execute only on one of 3 threads.
- The two other available macros for pack/unpack threads are ``PACK(( ))`` and ``UNPACK(( ))``
- ``DPRINT << SETW(width, sticky);`` supports a sticky flag (defaults to different froim std::setw() behavior).

*Printing tiles:*
-----------------

Debug print supports printing tile contents with the help of TSLICE macros.
These allow to sample a tile with a given sample count, starting index and stride.
This can be used on TRISCs as follows:
``PACK(( { DPRINT  << TSLICE(CB::c_intermed1, 0, SliceRange::hw0_32_16()) << ENDL(); } ));``
This will extract a numpy slice ``[0:32:16, 0:32:16]`` from tile 0 from CB::c_intermed1.
The ``PACK(())`` wrapper will limit printing to only be from the pack trisc thread.
Note that due to asynchronous and triple-threaded nature of compute engine kernels, this print statement has restrictions on validity and cannot be inserted at an arbitrary point in the kernel code.
``UNPACK(( DPRINT << ... ))`` only works between between cb_wait_front and cb_pop_front.
``PACK(( DPRINT << ... ))`` only works between cb_reserve_back and cb_push_back.
This applies both to TRISCs and in the reader/writer kernels.


*Known issues:*
---------------

- ``DPRINT << 0.1245f`` is currently broken for float constants. A suggested workaround is to use ``DPRINT << F32(0.1235f)``
- ``DPRINT << "string"`` is currently broken on the writer RISC core (BRISC) and causes a hang during kernel load.
  Instead you can output single characters, ``DPRINT << 's' << 't' << 'r'`` etc.
- Writes to L1 buffer used by DPRINT are not visible to the host if the kernel hangs in an infinite loop.

*Advanced use:*
---------------

The print server currently launches a separate thread for each thread on the device.
It supports parsing tokens RAISE(index) and WAIT(index) so that it can synchronize host-side with device-side ordering of prints.
For instance a NCRISC kernel could perform a DPRINT << RAISE(123) and BRISC kernel could DPRINT << WAIT(123) and the print server on the host in the BRISC thread will wait until it sees that signal raised.
The signal is then cleared by the debug print thread that parsed the WAIT signal.
This could be used to implement debug print ordering between different cores as well as threads.
For instance, the signal id could be computed as ``core_idx*5+thread_id`` to create a (core+thread)-specific index.

*Extending with new types:*
---------------------------

Not all types are by default supported by ``DPRINT << variable;`` syntax. However the code framework was designed with ease of extensibility in mind.
To add a new type, on the device you'll need to add a new ID to debug_print_common.h, then add a template instantiation DebugPrintTypeToId in debug_print.h.
On the host you'll need to modify tt_debug_print_server.cpp, look for the switch statement that parses, for instance, DEBUG_PRINT_TYPEID_FLOAT32, and add a new switch branch.

Watcher
*******

The Watcher is a thread that monitors the status of the TT device to help with
debug.  It:

- logs "waypoints" during execution on each core to help determine where a
  hang occured
- watches L1 address 0 to look for memory corruption
- sanitizes all transactions and reports invalid X,Y and addresses.  Further,
  any core with an invalid transaction will soft hang at the point of that
  transaction

It is enabled with:

- ``export TT_METAL_WATCHER=<n>`` where <n> is the number of seconds between status updates; use 0 for the default
- optionally ``export TT_METAL_WATCHER_APPEND=1`` to append to the end of the file instead (useful for tests which construct/destruct devices in a loop)

The output file contains a legend to help with deciphering the results.  The contents contain the last waypoint of each of the 5 riscvs encountered as a string of up to 4 characters.  These way points can be inserted into kernel/firmware code with the following, eg:

- ``DEBUG_STATUS('I');``
- ``DEBUG_STATUS('D', 'E', 'A', 'D');``

Examples:
---------

.. code-block::

    Core (x=1,y=1):     CWFW,NARW,R,R,R noc1:ncrisc{(02,08) 0x0065de40, 64}  rmb:R cb[1](rcv 1!=ack 0)

- The hang above originated on core (1,1) in physical coords (ie, the top left
  core)
- BRISC last hit way point CWFW (CB Wait Front Wait), NCRISC hit NARW (Noc
  Async Read Wait) and each Trisc is in the Run state (running a kernel).
  Look in the source (dataflow_api.h primarily) to decode the obscure
  names,search for DEBUG_STATUS
- There was an error on noc1 on NCRISC. Based on the waypoint, it was *reading*
  64 bytes from core (2,8) from an illegal L1 address of 0x65de40
- The printed bad noc state includes an X,Y location and since this was a
  read, it is clear that the bad address was the read address.  The address
  being written to would be local and so wouldn't include an X,Y if it was at
  fault
- The run mailbox says brisc was running
- Circular buffer #1 receive and acknowledge counts mismatch which matches
  that BRISC is stopped in CB Wait Front
