Tracy Host Profiler
===================

Profiling
---------

Tracy is an alternative method for profiling host-side python and C++ code.

Build with the tracy flag set is required for profiling with tracy profiler.

..  code-block:: sh

    make clean
    make build ENABLE_TRACY=1

With this build, all the marked zones in the C++ code will be profiled.


Profiling python code with tracy requires running your python code with the `tracy` module similar to the `cProfile` standard python module.

Also similar to the `cProfile` module, `sys.setprofile` and `sys.settrace` functions are used to set profiling callbacks.

For profiling your entire python program in tracy run your program as follows:

..  code-block:: sh

    python -m tracy {test_script}.py

For **pytest** scripts you can import pytest as a module and pass its argument accordingly.

For example to profile a bert unit test you can run the following:

..  code-block:: sh

    python -m tracy -m pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_split_and_transform_qkv_heads.py::test_split_fused_qkv_and_split_heads_with_program_cache


Python programs can also be partially profiled by instrumenting parts of the code intended to be profiled and run them with the `-p` option of the `tracy` module set.

For instrumenting pytest tests, `tracy_profile` fixture can be used to profile the entire test function.

The following example shows how to use the fixture and the required CLI command to do partial profiling.

Adding fixture:

..  code-block:: python

    def test_split_fused_qkv_and_split_heads_with_program_cache(device, use_program_cache, tracy_profile):

Running the `tracy` module with the `-p` option to do partial profiling:

..  code-block:: sh

    python -m tracy -p -m pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_split_and_transform_qkv_heads.py::test_split_fused_qkv_and_split_heads_with_program_cache

Instead of profiling the entirety of the pytest run, python functions only called as part of the `test_split_fused_qkv_and_split_heads_with_program_cache` functions are profiled in
tracy.

Instrumentation can also be done without using the pytest fixture.

The following shows how to profile an example `function_under_test` function.


..  code-block:: python

    def function_under_test():
        child_function_1()
        child_function_2()


    from tracy import Profiler
    profiler = Profiler()

    profiler.enable()
    funtion_under_test()
    profiler.disable()

Similar to the pytest setup, calling the parent script with `-p` option of the `tracy` set will profile the region where profiler is enabled.

**Note**, it is recommended to sandwich the function call between the enable and disable calls rather than having them as first first and last calls in the function being profiled.
This is because `settrace` and `setprofile` trigger on more relevant events when the setup is done previous to the functions call.

In some cases, significant durations of a function, do not get broken down to smaller child calls with explainable durations. This is usually either due to inline work that is
not wrapped inside a function or a call to a function that is defined as parte of a shared object. For example, `pytorch` function calls do not come in as native python calls and will not generate python call events.

For these cases, the line profiling feature of the `settrace` functions is utilized to provide line by line profiling. Because, substantially more data is produced in line by line
profiling, this options is only provided with partial profiling.

The same pytest example above will be profiled line by line by adding the `-l` option to the list of `tracy` module options.

..  code-block:: sh

    python -m tracy -p -l -m pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_split_and_transform_qkv_heads.py::test_split_fused_qkv_and_split_heads_with_program_cache


GUI
---

On your mac you need install tracy GUI with brew. On your mac terminal run:

..  code-block:: sh

    brew install tracy

Once installed run tracy GUI with:

..  code-block:: sh

    TRACY_DPI_SCALE=1.0 tracy

In the GUI you should start listening to the machine that your are running your code on over port 8086 (e.g. 172.27.28.132:8086) but setting the client address and clicking
connect.


At this point once you run your program, tracy will automatically start profiling.
