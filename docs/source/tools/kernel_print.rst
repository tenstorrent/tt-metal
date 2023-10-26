Kernel printf
=============

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
