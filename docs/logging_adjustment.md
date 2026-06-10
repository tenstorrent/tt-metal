# Various logging settings for tt-metal

## Table of Contents

1. [logging in python](#logging-in-python)
   1. [verbosity setup](#verbosity-setup-python)
   2. [code sample](#code-sample-python)
   3. [printing of full tensors](#print-full-tensors-python)
2. [logging in CPP](#logging-in-cpp)
   1. [verbosity setup](#verbosity-setup-cpp)
   2. [code sample](#code-sample-cpp)
   3. [TT_LOGGER_FILE](#tt_logger_file)
   4. [TT_LOGGER_TYPES](#tt_logger_types)
3. [kernel code logging](#kernel-code-logging)
   1. [verbosity setup kernel](#verbosity-setup-kernel)
   2. [code sample kernel](#code-sample-kernel)
   3. [important notes kernel](#important-notes-kernel)

## logging in python

### verbosity setup python

Set the console verbosity using the commands in the table below.

**Note:** Status messages like PASSED / FAILED will still be visible.

| level | command | filtering |
| --- | --- | --- |
| TRACE | `export LOGURU_LEVEL=TRACE` | all messages will be visible |
| DEBUG | `export LOGURU_LEVEL=DEBUG` | all messages except TRACE will be visible |
| INFO | `export LOGURU_LEVEL=INFO` | shows INFO messages and higher |
| SUCCESS | `export LOGURU_LEVEL=SUCCESS` | shows SUCCESS messages and higher |
| WARNING | `export LOGURU_LEVEL=WARNING` | shows WARNING  messages and higher |
| ERROR | `export LOGURU_LEVEL=ERROR` | shows ERROR and CRITICAL messages |
| CRITICAL | `export LOGURU_LEVEL=CRITICAL` | shows CRITICAL only |
### code sample python
```python
import os
import sys
# include loguru
from loguru import logger

# set necessary level of LOGURU
# use LOGURU_LEVEL from environment if set - INFO by default in line below
level = os.getenv("LOGURU_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=level)

logger.trace("Trace message: very fine-grained diagnostic detail")  # logger.trace(...)
logger.debug("Debug message: developer-focused diagnostic detail")  # logger.debug(...)
logger.info("Info message: configured Loguru level is {}", level)  # logger.info(...)
logger.success("Success message: operation completed successfully")  # logger.success(...)
logger.warning("Warning message: non-fatal unexpected condition")  # logger.warning(...)
logger.error("Error message: failure that needs investigation")  # logger.error(...)
logger.critical("Critical message: unrecoverable error condition")  # logger.critical(...)

```

### print full tensors python
```python
  torch.set_printoptions(profile="full")
  ttnn.set_printoptions(profile="full")

  print(torch_output)
  print(actual_output)
```

## logging in CPP

### verbosity setup cpp
Set the C++ log verbosity with `TT_LOGGER_LEVEL` (case-insensitive; common values: trace, debug, info, warn/warning, error, critical/fatal, off):

| level | command | filtering |
| --- | --- | --- |
| TRACE | `export TT_LOGGER_LEVEL=TRACE` | shows all messages |
| DEBUG | `export TT_LOGGER_LEVEL=DEBUG` | shows DEBUG, INFO, WARNING, ERROR, CRITICAL |
| INFO | `export TT_LOGGER_LEVEL=INFO` | shows INFO, WARNING, ERROR, CRITICAL |
| WARNING | `export TT_LOGGER_LEVEL=WARNING` | shows WARNING, ERROR, CRITICAL |
| ERROR | `export TT_LOGGER_LEVEL=ERROR` | shows ERROR, CRITICAL |
| CRITICAL | `export TT_LOGGER_LEVEL=CRITICAL` | shows CRITICAL only |
| OFF | `export TT_LOGGER_LEVEL=OFF` | disables logging output |

### code sample cpp
```cpp
#include <tt-logger/tt-logger.hpp>

int main() {
    // set in shell, for example: export TT_LOGGER_LEVEL=DEBUG
    log_trace(tt::LogMetal, "Trace message");      // log_trace(...)
    log_debug(tt::LogMetal, "Debug message");      // log_debug(...)
    log_info(tt::LogMetal, "Info message");        // log_info(...)
    log_warning(tt::LogMetal, "Warning message");  // log_warning(...)
    log_error(tt::LogMetal, "Error message");      // log_error(...)
    log_fatal(tt::LogMetal, "Critical message");   // log_fatal(...)
    return 0;
}
```
### TT_LOGGER_FILE
`TT_LOGGER_FILE` is handled by the external tt-logger dependency
when set like, for example
```bash
export TT_LOGGER_FILE=/tmp/tt_metal.log
```
all C++ log output is redirected to that file instead of stdout/the console

### TT_LOGGER_TYPES

To filter log messages by type, set the `TT_LOGGER_TYPES` environment variable to a concatenated list of type names (no separators):
Always, Test, Timer, Device, Distributed, LLRuntime, Loader, BuildKernels, Verif, Op, Dispatch, Fabric, Metal, TTNN, MetalTrace, Inspector, UMD, EmulationDriver
```bash
export TT_LOGGER_TYPES=OpTTNN     # only Op + TTNN (+ Always) logs
```
in cpp code for `Verif` type:
```cpp
#include <tt-logger/tt-logger.hpp>

// Pick the severity you need:
log_trace(tt::LogVerif, "Verif trace message");
log_debug(tt::LogVerif, "Verif debug message");
log_info(tt::LogVerif, "Verif info message");
log_warning(tt::LogVerif, "Verif warning message");
log_error(tt::LogVerif, "Verif error message");
log_critical(tt::LogVerif, "Verif critical message");
```

## kernel code logging

### verbosity setup kernel

Use `DPRINT(...)` in kernel code, then configure host-side capture:

| option | command | effect |
| --- | --- | --- |
| required | `export TT_METAL_DPRINT_CORES=<core_spec>` | enables kernel print collection (examples: `all`, `0,0`, `"(0,0),(1,2),(3,4)"`, `"(0,0)-(2,2)"`) |
| optional | `export TT_METAL_DPRINT_RISCVS=BR` there are more possible values! (not covered here) | prints only selected RISCs |
| optional | `export TT_METAL_DPRINT_FILE=kernel_debug.log` | writes output to file |
| optional | `export TT_METAL_DPRINT_ONE_FILE_PER_RISC=1` | creates per-RISC log files |

### code sample kernel
```cpp
#include "api/debug/dprint.h"  // required for DPRINT

void kernel_main() {
    uint32_t tile_id = get_arg_val<uint32_t>(0);
    DPRINT("kernel start, tile_id={}\n", tile_id);   // DPRINT(...)
    DPRINT_DATA0("reader running on noc0\n");        // DPRINT_DATA0(...)
    DPRINT_DATA1("writer running on noc1\n");        // DPRINT_DATA1(...)
}
```

### important notes kernel

- Every `DPRINT` line must end with `\n`; otherwise prints may stay buffered.
- Do not set `TT_METAL_DPRINT_CORES`, `TT_METAL_WATCHER`, and `TT_METAL_DEVICE_PROFILER` together.

[Home](#various-logging-settings-for-tt-metal)
