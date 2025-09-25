.. _tracy_profiler:

Tracy Profiler
==============

.. note::
   Tools are only fully supported on source builds.

Profiling is an essential part of software development that helps developers gain insight into how their code executes, where time is spent, and which parts of the system may be causing performance bottlenecks. By collecting and analyzing detailed runtime data, profiling tools enable developers to identify inefficient code paths, understand resource usage, and make informed decisions about where to focus optimization efforts. Effective profiling is key to improving performance.

Overview
--------

`Tracy <https://github.com/wolfpld/tracy>`_ is an opens-source C++ profiling tool with sampling and code instrumentation profiling capabilities. Metalium uses a fork for Tracy adapted to the the Tensix Processors as it's primary profiling tool.

Detais documentation about Tracy itself can be found here: https://github.com/wolfpld/tracy/releases/latest/download/tracy.pdf. Reading the ``Quick-start guide`` section can help with the rest of this documentation.

All host-side code, including python code in ``tt_metal`` can be profiled using Tracy.

Enabling Tracy
--------------

To use Tracy for profiling, you need to build Metalium with profiler support enabled. This is done by setting the ``ENABLE_TRACY=ON`` option when configuring CMake. Or with the ``--enable-profiler`` flag in the build script. Profiling features are only available in builds where this option is set.

You can enable the profiler by running:

..  code-block:: bash

    # Via build script
    ./build_metal.sh --enable-profiler

    # Via CMake flags
    cmake . -DENABLE_TRACY=ON
    ninja
    ninja install

Note that Tracy support is not included in prebuilt binaries or release builds by default. You must build from source with the profiler enabled to use these features.

GUI
---

Tracy provides a GUI application for viewing profiling results. You can open saved profiling sessions or connect to a remote machine to view real-time profiling data, as long as network access to the remote system is available.

Installing for Mac users
~~~~~~~~~~~~~~~~~~~~~~~~

Mac users can install Tracy using Homebrew. Open a terminal and run:

..  code-block:: bash

    brew uninstall tracy # Remove any old version of Tracy
    wget -P ~/ --no-check-certificate --no-cache --no-cookies https://raw.githubusercontent.com/tenstorrent-metal/tracy/master/tracy.rb
    brew install ~/tracy.rb
    rm ~/tracy.rb

After installation, start the Tracy GUI with:

..  code-block:: bash

    tracy

Building for Linux users
~~~~~~~~~~~~~~~~~~~~~~~~

For Linux users, you need to build the Tracy GUI from source. First, clone the Tracy repository.

..  code-block:: bash

    git clone https://github.com/tenstorrent/tracy.git
    cd tracy/profiler/build/unix
    make -j8

A ``Tracy-release`` binary will be generated in the current directory after the build completes. You can run it directly from there or copy it to a directory in your PATH for easier access

.. code-block:: bash

    ./Tracy-release

Starting the GUI
~~~~~~~~~~~~~~~~

The application will start showing a window similar to the image below.

.. image:: ../_static/tracy-get-started.png
    :alt: Tracy get started
    :width: 77%

Capturing Profiling Data
------------------------

Set the client address to the IP address of the remote machine and port 8086 (e.g. 172.27.28.132:8086), then click connect.

A "Waiting for connection ..." dialog will appear after clicking connect.

.. image:: ../_static/tracy-waiting-connection-dialog.webp
    :alt: Tracy waiting for connection

When the host machine starts running a tracy-enabled application, the GUI will automatically collect profiling data and display it in real time.

Counterintuitively, the Tracy GUI connects as a TCP server, while the profiled application runs as a TCP client, usually connecting to port 8086. If your application host is on a different network than the Tracy GUI, you may need to set up port forwarding or a VPN connection. SSH port forwarding is a common solution:

.. code-block:: bash

    ssh -NL 8086:127.0.0.1:8086 user@remote-machine

Capturing via Command Line
~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, use the ``capture-release`` CLI tool built under **tt-metal** when Tracy is enabled, under ``tt-metal/build/tools/profiler/bin`` when Tracy is enabled. This tool acts as a client that saves the profile to disk, which can then be copied and loaded into the GUI later. To use it, run the following command before starting the application:

.. code-block:: bash

    ./capture-release -o output_file_name.tracy

.. note::

    The output of ``capture-release`` is quite compressible. For large profile files, it is recommended to compress them before transferring over the network. You can use the ``-z`` option with rsync, ``-C`` with scp, or standalone tools like gzip or zstd.

Profiling Host Code
-------------------

C++
~~~

With Tracy enabled in the Metalium build, all C++ marked zones will be profiled. Zones in Tracy are marked sections of code that users are interested in profiling. Tracy provides macros such as  ``ZoneScoped;`` to accomplish this.

Please refer to section 3 of Tracy's documentation for further information on zones and available macros.

The following image is a snapshot of Tracy C++ profiling:

.. image:: ../_static/tracy-c++-run.png
    :alt: Tracy C++ run

For example, the ``Device`` constructor shown above is instrumented as follows:

..  code-block:: C++

    Device::Device(chip_id_t device_id, const std::vector<uint32_t>& l1_bank_remap) : id_(device_id)
    {
        ZoneScoped;
        this->initialize(l1_bank_remap);
    }

Python
~~~~~~

Python provides the standard ``sys.setprofile`` and ``sys.settrace`` functions for tracing and profiling Python code. These are used to integrate Python profiling with Tracy.

There are several ways to profile Python code with Tracy in Metalium projects.

Python Scripts
^^^^^^^^^^^^^^

To profile an entire Python script, run your program using the ``tracy`` module as follows:

..  code-block:: sh

    python -m tracy {test_script}.py

Pytest Sessions
^^^^^^^^^^^^^^^

For pytest-based tests, import pytest as a module and pass its arguments as needed. For example, to profile a BERT unit test, run:

..  code-block:: sh

    python -m tracy -m pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_split_and_transform_qkv_heads.py::test_split_query_key_value_and_split_heads_with_program_cache

.. image:: ../_static/tracy-python-run.png
    :alt: Tracy Python run


Instrumenting Python Code
^^^^^^^^^^^^^^^^^^^^^^^^^

Instrumentation can also be done without using the pytest fixture.

The following shows how to profile a function called ``function_under_test`` and all of its child python function calls by manually enabling tracy profiler.

..  code-block:: python

    def function_under_test():
        child_function_1()
        child_function_2()


    from tracy import Profiler
    profiler = Profiler()

    profiler.enable()
    function_under_test()
    profiler.disable()

Similar to the pytest setup, calling the parent script with ``-p`` option will profile the region where profiler is enabled.

**Note**, it is recommended to sandwich the function call between the enable and disable calls rather than having them as first and last calls in the function being profiled. As ``settrace`` and ``setprofile`` trigger on more relevant events when the setup is done previous to the function call.

Signposts in Python Code
^^^^^^^^^^^^^^^^^^^^^^^^

``signpost(header, message)`` from the ``tracy`` module can be placed anywhere in the code path for your test. This call will produce a row in the op report CSV and a message in the tracy run.

..  code-block:: python

    from tracy import signpost

    signpost(header="Run number 5", message="This is the run after 5 warmup runs")

    run_inference()

    signpost(header="Run result post proc")

    post_proc()

The above example will show up as follows.

Op report CSV

.. image:: ../_static/tracy-signpost-opreprot.png
    :alt:

Tracy run

.. image:: ../_static/tracy-signpost-run.png
    :alt: Tracy get started

Line-level Profiling
^^^^^^^^^^^^^^^^^^^^

In some cases, significant duration of a function, does not get broken down to smaller child calls with explainable durations. This is usually either due to inline work that is
not wrapped inside a function or a call to a function that is defined as part of a shared object. For example, ``pytorch`` function calls do not come in as native python calls and will not generate python call events.

Line-level profiling is only provided with partial profiling because it produces substantially more data.

Add  ``-l`` option to enable line-level profiling:

..  code-block:: sh

    python -m tracy -p -l -m pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_split_and_transform_qkv_heads.py::test_split_query_key_value_and_split_heads_with_program_cache

Profiling Device Code
---------------------

The version of Tracy used in Metalium supports profiling device-side code, including individual Baby RISC-V cores on each Tensix and other tiles on the NoC.

For more details on device-side profiling with Tracy, see :ref:`Device Program Profiler<device_program_profiler>`.
