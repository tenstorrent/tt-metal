# Runtime & Loader

Central location for Versim and Silicon device backend code. Core components are runtime, loader, and cluster.

Architecture level documentation is done on Sharepoint. 
- [Overview](https://tenstorrent.sharepoint.com/:w:/s/Software/EdNy2TJnUiFNmWVQnFW88bEB0zv6vE_a_4MOW2potQkSDQ?e=eI7b6J)

## Directory structure
```shell
.
├── *.cpp/*.hpp           # flat directory for core components
│
├── tests
│   ├── <test>.cpp        # test flow files
│   └── net_<testsuite>   # test suite directories
│    └── *.yaml
│
└── scripts               # misc scritps for loader
```

## Bug tracking

We will use [Device Runtime board](https://yyz-gitlab.local.tenstorrent.com/tenstorrent/budabackend/-/boards/49?label_name[]=Runtime) for tracking our bugs:


All issues should be tagged with `Runtime` label for them to show up on the board. Other labels can be applied on top - `Versim`, `Silicon`, etc.

## Testing

### Build steps
`make`

`make loader/tests`

### Testing
Individual tests can be invoked via directly calling the executable
`./build/test/loader/tests/test_pybuda_api --netlist loader/tests/net_basic/netlist_unary_multicore.yaml`

All loader tests should support the following flags:
- `--silicon` run on silicon target, otherwise run on versim
- `--netlist` user specified netlist to run, note the test may only be compatible with a select few

Test writers are free to define test specific command line arguments.

Some tests may support the following flags
- `--O` optimization level for runtime & loader, similar to gcc -O where higher means more optimized
- `--arch` architecture type, such as grayskull, wormhole

### Test checking
Test writer also has a lot of flexibility in terms of the pass/fail criteria of loader tests.

The philosophy is to leave OP testing to OP owners, similarly for net2pipe/pipegen and other components, but only worry about how we integrate these together. For example the middle OP of a graph that does not read/write DRAM is a don't care for us, we care about those that interface with DRAM/Host since loader code initializes/pushes/pops queues, and manages the schedule of epochs that use those queues.

As a result, numerical accuracy of a kernel is not the goal of our tests. Some of our tests may not even pop outputs to check results. Eg. `test_queue_state.cpp` only checks final state of queues (rdptr, wrptr, etc) but does not pop the contents for checking.

We should also minimize the amount of custom test only code in loader components, if they are included in core components, there should be NO performance impact.
   
### CI and Regressions

Test suites location for Buda Backend: $ROOT/ci/test-lists

Current runtime test lists:
- runtime_grayskull_versim_push.yaml
- runtime_grayskull_silicon_push.yaml

Kibana dashboard is used for monitoring run results, custom views can be created and saved using 'Share->Permalinks'
- [Master Dashboard](http://yyz-elk/goto/80b5ea9c27ce1dfbb7b3911f8aa44709) - master branch recent runs
- [User Dashboard](yyz-elk/goto/77a747bf2434c01b261ccf1b90f3670b), user specific runs on all branches, tzhou as example here 

For more info see CI doc:
- https://tenstorrent.sharepoint.com/:p:/s/Software/EcVDiJXTT8pDiWI6_4Q0RDMBADqqZiV8xLIfG_kQeI8yLw?e=Fs5H3Z

## Coding Style

### General Rules
- No code formatters, they often do a worse job than humans. It's only useful for standardizing code base for a larger team of collaborators.
- Split hpp/cpp unless code is trivial or types only, this greatly reduces recompile cost and also keeps header files clean.
- When in doubt, follow the same style of the code being edited. Even if it does not conform to overall code style. This is better for consistency, and perserving git blame history by avoiding large reformatting changes.
- We write performance critical code, understand data structures and their performance. Eg. unordered version strictly dominates their ordered counterparts, typically [O(1) vs. O(logn)](http://supercomputingblog.com/windows/ordered-map-vs-unordered-map-a-performance-study/) 
- We write applications that run for a long time, DO NOT LEAK MEMORY! Run valgrind on the codebase from time to time
```shell
valgrind --leak-check=full <test command>
```

### Coding practices
- Only use TT_ASSERT/THROW or log_error/fatal for error logging. No explicit assert, throw allowed are our CI and PyBuda handling of them are undefined.
- Use `auto` only for iterators, and complex types. For example, iterating over a map is easier with `for (auto &it : some_map)`
- Always pass by reference or pointer. The exceptions are simple types - int, string (if they are short), simple structs
- lambdas should be avoided in core component code, ok to use in test code
- Nested namespaces are only allowed up to 1 additional level - `tt::io` is okay, `tt::runtime::io` is not
- Avoid complex template metaprogramming, ok to use templating for performance and code reduction
- Use obvious names for methods, start the method name with a verb whenever it makes sense
  - `set_*` setter methods
  - `get_*` getter methods, for those with non-void return types
  - `dump_*` for those that dump to screen or file
  - exceptions are callback functions, which do not need to start with a verb but should end with `*_callback`
  - other exceptions include legacy functions, eg. `channel_0_address` rather than `get_channel_0_address` to match driver

### Logging

Use log_info/debug/warning, with the relevant tt::LogType, see $ROOT/common/logger.hpp for list of available tt::Log types.

Verbosity and component dumping control are available via `LOGGER_LEVEL` and `LOGGER_TYPES` environment variables. Eg.
```shell
# Dump all messages at Debug verbosity
LOGGER_LEVEL=Debug ./build/test/loader/tests/test_pybuda_api --silicon --netlist loader/tests/net_basic/netlist_unary_multicore.yaml

# Dump only tt::LogLLRuntime, tt::LogIO, tt::LogTest messages at Debug verbosity
LOGGER_LEVEL=Trace LOGGER_TYPES=Loader,IO,Test ./build/test/loader/tests/test_pybuda_api --silicon --netlist loader/tests/net_basic/netlist_unary_multicore.yaml
```

- `log_info` is by default printed, it should only be used for program/graph level events
- `log_debug` for per queue/txn/core level of verbose dumping
- `log_trace` for the most verbose trace info

### Device-side logging / debugging with prints

The codebase supports debug prints from device kernels that get displayed on the host. On device the APIs are defined in src/firmware/riscv/common/debug_print.h.
To use debug printing capability, it is first required to start the debug print server on the host - use the API defined in llrt/tt_debug_print_server.h.

Basic use steps:

- Include the device side header #include "debug_print.h" in your kernel.
- Use DPRINT << "string" << 1 << SETW(4) << F32(2.0f) << ENDL(); cout style syntax to print values.
- On the host it is first required to start the server using tt_start_debug_print_server(...) API.
- Tests test_debug_print_br/nc.cpp and test_run_test_debug_print.cpp show an example of using debug prints in a complete running kernel example.


Known issues:

- DPRINT << 0.1245f is currently broken for float constants (also due to a compiler bug). A suggested workaround is to use DPRINT << F32(0.1235f)
- DPRINT << "string" is currently broken on BRISC and causes a hang during kernel load. This is a bug in tt::memory object that we need to fix.
  Instead you can output single characters, DPRINT << 's' << 't' << 'r' etc.
- Writes to L1 buffer used by DPRINT are not visible to the host if the kernel hangs in an infinite loop. This could be an issue with MMIO/paging and needs investigation.

Advanced use:

The print server currently launches a separate thread for each thread on the device.
It supports parsing tokens RAISE(index) and WAIT(index) so that it can synchronize host-side with device-side ordering of prints.
For instance a NCRISC kernel could perform a DPRINT << RAISE(123) and BRISC kernel could DPRINT << WAIT(123) and the server on the host in the BRISC thread will wait until it sees that signal raised.
The signal is then cleared by the debug print thread that parsed the WAIT signal.
This could be used to implement debug print ordering between different cores as well as threads.
For instance, the signal id could be computed as core_idx\*5+thread_id to create a (core+thread)-specific index.

Extending with new types:

On the device you'll need to add a new ID to debug_print_common.h, then add a template instantiation DebugPrintTypeToId in debug_print.h.
On the host you'll need to modify tt_debug_print_server.cpp, look for the switch statement that parses for instance DEBUG_PRINT_TYPEID_FLOAT32 and add a new clause.



