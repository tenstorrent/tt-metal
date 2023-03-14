========================
Execution Time Profiler
========================

Host Side
========================

Host API is profiled by wrapping the portion of the code that needs profiling with start and end
markers with the same timer name. After the execution of the wrapped code, the start, end and the
delta in between them for all the timers is recorded in a CSV for further post processing.

Setup
------------------------

For profiling any module on the host side, an object of the of the ``Profiler`` class is needed
in order to record the marked times and dump the result to a CSV. The ``Profiler`` is defined under
the ``tools/profiler/profiler.hpp`` header which can be include as follows.

..  code-block:: C++

    #include "tools/profiler/profiler.hpp"

The module Make procedure should also include the profiler library. This can be done by adding the
the ``-lprofiler`` flag to the ``LDFLAG`` argument in the ``module.mk`` of that module. For example
for tests under ``tt_metal``, which uses the profiler, the following is the ``LDFLAG`` line in ``tt_metal/tests/module.mk``.

..  code-block:: MAKEFILE

    TT_METAL_TESTS_LDFLAGS = -ltt_metal_impl -ltt_metal -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -lhlkc_api -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp

With the instance of the ``Profiler`` class, ``markStart`` and ``markStop`` functions can be used to
profile the module. Again taking ``tt_metal`` as an example, ``tt_metal_profiler`` is
instantiated as a static member of the module ``tt_metal/tt_metal.cpp`` as follows.

..  code-block:: C++

    static Profiler tt_metal_profiler = Profiler();

In functions such as ``LaunchKernels`` the entire code within the function is wrapped under the
``markStart`` and ``markStop`` calls with the timer name ``"LaunchKernels"``.

..  code-block:: C++

    bool LaunchKernels(Device *device, Program *program) {

        tt_metal_profiler.markStart("LaunchKernels");
        bool pass = true;

        auto cluster = device->cluster();

        <Internals of LaunchKernels>

        cluster->broadcast_remote_tensix_risc_reset(pcie_slot, TENSIX_ASSERT_SOFT_RESET);

        tt_metal_profiler.markStop("LaunchKernels");
        return pass;
    }

After the execution of all wrapped code. A call to  ``dumpResults`` will process the deltas on all
timers and dump the results into a CSV in the current directory called ``profile_log_host.csv``. In
``tt_metal`` this function is wrapped under another function called ``DumpHostProfileResults`` to
simplify the decision of when to generate the CSV on tests that are using the ``tt_metal`` API.

The ``dumpResults`` also flushes all the timers data after the dump. This is so that the same
object can be used to perform multiple consecutive measurements on the same timer name. The ``name_append`` argument adds
a ``Section name`` column to the CSV that demonstrates which ``dumpResults`` a row in the CSV
belongs to.

``tt_metal\tests\test_add_two_ints.cpp`` is a good example that demonstrates this scenario.
``LaunchKernels`` is called twice in this test, if we only dump results once at the end of the
execution, we will only get the results on the last call to that function. With the use of sections
names we can call ``DumpHostProfileResults`` twice and get and output such as the following in the
CSV.


..  code-block:: c++

    Section Name, Function Name, Start timer count [ns], Stop timer count [ns], Delta timer count [ns]
    first, LaunchKernels, 675598390620333, 675598390740682, 120349
    first, ConfigureDeviceWithProgram, 675598152012369, 675598390619993, 238607624
    first, CompileProgram, 675597384816840, 675598152009299, 767192459
    second, LaunchKernels, 675598625865918, 675598625981107, 115189
    second, ConfigureDeviceWithProgram, 675598392545035, 675598625864988, 233319953


Limitations
------------------------
* Each core has limited L1 buffer for dprinting which the profiler is built on top of. Depending on
  how well the host side server can keep up with flushing the buffers, care has to be taken in
  adding ``mark_time`` calls in the kernel code. Roughly speaking, with current size of the profile
  messages, after around 15 consecutive ``mark_time`` calls the buffer can fill up if it is not
  flushed. The core will stall at this point.

* The cycle counts give very good relative numbers with regards to various events that are marked
  on the kernel. Syncing this with the wall clock is not brought in yet. This will require
  collection on core reset times on the host side and syncing every cycle count accordingly

* It is relatively safe to assume that all RISCs on all cores are taken out of reset at the same
  time so processing the cycle counts read from various RISCs is reasonable. But caution has to be
  taken when doing such measurements

* TRISC0,1,2 measurements are not supported. Further development on underlying APIs are required
  inorder to bring profiling to these cores.

* Debug print can not used in kernels that are being profiled.
