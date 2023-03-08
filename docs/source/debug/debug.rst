Debug Tools
#########################


tt-gdb
****************************************
.. raw:: html

    <video width="800" height="600" controls>
        <source src="../_static/DebuggerTutorial.mp4" type="video/mp4">
    <p>Your browser does not support the video tag.</p>
    </video>

|

|

Kernel printf()
********************************


The codebase supports debug prints from device kernels that get displayed on the host. On device the APIs are defined in src/firmware/riscv/common/debug_print.h.
To use debug printing capability, it is first required to start the debug print server on the host - use the API defined in llrt/tt_debug_print_server.h.

*Basic use steps:*
------------------

- Include the device side header ``#include "debug_print.h"`` in your kernel.
- Use ``DPRINT << "string" << 1 << SETW(4) << F32(2.0f) << ENDL();`` std::cout-style syntax to print values.
- For the prints to show up on the host, it is first required to start the host-side server using ``tt_start_debug_print_server(...)`` API.
  - If the server isn't started, then the prints will not show up and do nothing.
- Tests test_debug_print_br/nc.cpp and test_run_test_debug_print.cpp show an example of using debug prints in a complete running kernel example::

    auto device = tt_metal::CreateDevice(tt_metal::DeviceType::Grayskull, pci_express_slot);
    ...
    vector<tt_xy_pair> cores = {{1,1}};
    int hart_mask = DPRINT_HART_NC | DPRINT_HART_BR;
    tt_start_debug_print_server(device->cluster(), {chip_id}, cores, hart_mask);
    ...
- Note that the core coordinates used in tt_start_debug_print_server are currently NOC coordinates (not logical).


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
