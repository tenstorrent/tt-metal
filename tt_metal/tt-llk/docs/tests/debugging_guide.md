# Debugging guide and checklists

The point of this guide is to serve as a checklist for everything you may have forgotten to put in your test files. All checklists are sorted by bug frequency.

## Table of contents
| | |
|:----|:----|
| 1 | [Quality of life tips & tricks](#quality-of-life-tips--tricks) |
| 2 | [Compilation errors](#quality-of-life-tips--tricks) |
| 3 | [Runtime errors](#runtime-errors) |
| 4 | [Assertion errors](#assertion-errors) |
| 5 | [Test flakiness](#test-flakiness) |
---

# Quality of life tips & tricks

## Enhanced view of error matrices
If your terminal is too narrow or too short to display a complete dump of all tiles your test variant processed, the better approach is to redirect `pytest`'s `stderr` to a file like this:

`pytest --compile-consumer -x ./my_test_name.py 2>./my_file_path.txt`

In order to view an error matrix as you can in your terminal, you need to install the VSCode/Cursor extension [ANSI Colors](https://marketplace.visualstudio.com/items?itemName=iliazeus.vscode-ansi). Afterwards, open the file and select ANSI preview to see the colors as they are in the terminal.

You can use this approach when you have many errors in your variants, to speed up their execution by redirecting the `stderr` to a file. In most execution cases, `pytest` is actually bound by the terminal throughput of relatively large error messages produced.

# Compilation errors

- Did you include all default headers provided in the example for your test type?
    - `#include "params.h"` is mandatory because it's the source of your entire `cpp` test configuration;
- Does your `run_kernel` look like this:
    - ```void run_kernel(const volatile struct RuntimeParams *params)```;
    - Did you put all the keywords?
- How are my Python passed template and runtime parameters accessed in my C++ kernel code?
    - TODO
- I'm getting a compilation error when I compile with coverage enabled.
    - This can be a consequence of a bad LLK API call, that is written in such a way that the compiler fails to deal with it when coverage is enabled;
    - If errors are of type: `Can't fit 32-bit value in 16-bit TTI buffer`, it's probably an LLK API error that is only caught when compiling for coverage;

# Runtime errors

- TTException - can't find an object file:
    - TODO
- My kernel hangs the core when I add my new runtime parameter to the runtimes list of TestConfig.
    - TODO

# Assertion errors

- Do you know exactly which assert failed?
    - If no, **please** put a small comment after your asserts like this to enhance your visibility:
    ```
    assert len(res_from_L1) == len(golden_tensor), "Result tensor and golden tensor are not of the same length"
    ```
- Did you hardcode your stimuli addresses?
    - Firstly, you're not supposed to do this. Stimuli is accessed from the kernel code using `buffer_A`, `buffer_B`, and `buffer_Res` variables.
    - If you are 110% sure you must hardcode your addresses, **please** consult the `L1 memory layouts` section of `infra_architecture.md`, to be sure your stimuli is in L1 where your kernel expects it.
    - To make Python actually write stimuli to your specific address, you need to reassign `StimuliConfig.STIMULI_L1_ADDRESS` static field with your new address. Keep in mind that this will make other tests that use default addresses fail because you changed where their stimuli is loaded to L1.
- Did you access any hardcoded addresses of your choosing?
    - If you are 110% sure you must do this, **[please see L1 layout](infra_architecture.md#l1-memory-layouts)** to be sure you didn't accidentally overwrite some other important piece of data used by the kernel or read some garbage.
- Is your error matrix the same every time you run your failing variant?
    - To be extra sure run `tt-smi -r` between every `pytest` invocation;
    - If this is indeed the case, your kernel really does process the data you supplied, but it's configured in an invalid way. Please check all arguments to the `TestConfig` object to be sure everything is as you expect it. If you're sure, check the build.h of your variant to check if C++ gets parameterized correctly;
- Is your error matrix different every time you run your failing variant?
    - This means that your kernel is not processing any stimuli you supplied to it, thus your kernel is malconfigured.

# Test flakiness

This section will be written when dumping tensix state is incorporated in testing infra. TODO @ajankovicTT
