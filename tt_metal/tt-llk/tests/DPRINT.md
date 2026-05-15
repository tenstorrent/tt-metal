# Device Print

`DEVICE_PRINT` is a formatted printing facility that lets LLK code efficiently emit logs that the Python test harness picks up and renders.

It is a direct port from Metal (see
[tt_metal/hw/inc/api/debug/device_print.h](../../hw/inc/api/debug/device_print.h)
and [tt_metal/impl/debug/dprint_parser.cpp](../../impl/debug/dprint_parser.cpp))
adapted to LLK infra. The device-side header is reused through a thin wrapper, and the host-side parser is rewritten in Python on top of `tt-exalens`.

## Quick start

Kernel side:

```cpp
#include "dprint.h"

void run_kernel(RUNTIME_PARAMETERS)
{
    DEVICE_PRINT("test: i32={} hex={:08x}\n", (int32_t)-1, (uint32_t)0xDEADBEEF);
    DEVICE_PRINT("string: {}\n", CTSTR("done"));
}
```

Python side:

```python
from helpers.device_print import run_with_device_print
from helpers.test_config import DevicePrintBuild, TestConfig

config = TestConfig(
    "sources/my_test.cpp",
    formats,
    device_print_build=DevicePrintBuild.Yes,  # opts the variant into dprint
)
outcome, lines = run_with_device_print(config)
assert "test: i32=-1 hex=deadbeef" in "".join(lines)
```

Each line is also written to the loguru session log at `INFO` level, so you see them in `test_run.log` without needing to inspect `lines` yourself.

You can take a look at the dprint test for a full example:
- [sources/device_print_test.cpp](sources/device_print_test.cpp)
- [python_tests/test_device_print.py](python_tests/test_device_print.py)

## How it works

```
              kernel                           host
   ┌─────────────────────────┐       ┌─────────────────────────┐
   │ DEVICE_PRINT(fmt, ...)  │       │  run_with_device_       │
   │   ├─ stamp format str   │       │  print(config)          │
   │   │  into .device_      │       │   ├─ poll ring buffer   │
   │   │  print_strings      │       │   ├─ resolve strings    │
   │   ├─ stamp info record  │       │   │  via ELF section    │
   │   │  into ...info       │  <-   │   ├─ unpack args        │
   │   └─ write header+args  │ poll  │   │  in size-desc order │
   │      to buffer in L1.   │  ->   │   └─ render + log       │
   └─────────────────────────┘       └─────────────────────────┘
```

There's a shared ring buffer on the device. Every call writes a 4-byte header followed by the packed argument bytes. The format string itself is never copied. It lives in a non-loaded ELF section that the host accesses.

When the buffer fills, the kernel sets a stall flag and waits for the host to drain it before continuing.

## Public API

Defined in [helpers/include/dprint.h](helpers/include/dprint.h):

- `DEVICE_PRINT(fmt, ...)`: Emit a record. Uses fmtlib-style placeholders.
- `CTSTR(literal)`: Save a string literal into the ELF and pass it as an argument. Use this for all string literals (except the format strings themselves).

One might also notice the macros `DEVICE_PRINT_INITIALIZE_LOCK()` and `DEVICE_PRINT_KERNEL_FINISHED()` in the Metal device print header. Don't call these from LLK; test infra does the job of both.

When `device_print_build=DevicePrintBuild.No` (default), or under `COVERAGE`, `DEVICE_PRINT` invocations expand to nothing. It is currently not supported to use device print in coverage builds.

Make sure to call `TestConfig.setup_arch()` before constructing a config with `device_print_build=Yes`.

## Format specifiers

`DEVICE_PRINT` uses libfmt syntax. Common forms in the test suite:

| Spec | Meaning | Example |
|------|---------|---------|
| `{}` | Default rendering | `i32=-1` |
| `{:08x}` | Hex, 8-wide, zero-pad | `hex=00000abc` |
| `{:>8}` | Right-align in 8 cols | `pad=    test` |
| `{:#}` | Enum: render fully qualified | `flag=Perm::R \| Perm::W` |

The Python renderer applies the spec via Python's `format()`, so most fmtlib
specifiers translate directly. If you write something fmtlib accepts but
Python doesn't, the renderer falls back to `str(value)`.

### Enums

Enum arguments are resolved against DWARF debug info at decode time, so you
get readable names without registering anything:

```cpp
enum class Color : uint8_t { Red = 0, Green = 1, Blue = 2 };
enum class Perm  : uint32_t { R = 1, W = 2, X = 4 };

DEVICE_PRINT("c={}\n",  Color::Green);          // c=Green
DEVICE_PRINT("p={}\n",  Perm::R | Perm::X);     // p=R | X       (flag enum)
DEVICE_PRINT("p={:#}\n", Perm::R | Perm::W);    // p=Perm::R | Perm::W
DEVICE_PRINT("p={}\n",  static_cast<Perm>(24)); // p=(Perm)24    (unknown bits)
```

## What was ported

| Component | Source of truth | LLK equivalent |
|-----------|-----------------|----------------|
| Kernel-side macros, packing, buffer writer | [api/debug/device_print.h](../../hw/inc/api/debug/device_print.h) | Reused as-is, save for some `#ifdef`s; wrapped by [helpers/include/dprint.h](helpers/include/dprint.h) |
| Host parser (C++) | [impl/debug/dprint_parser.cpp](../../impl/debug/dprint_parser.cpp) | Rewritten in Python: [helpers/device_print.py](python_tests/helpers/device_print.py) |
| Per-RISC `PROCESSOR_INDEX`, buffer base, buffer size | Metal constants | Passed via `-D` from [test_config.py](python_tests/helpers/test_config.py); single source of truth is `TestConfig.RISC_INFO` |
| `.device_print_strings{,_info}` sections | Metal linker scripts | [helpers/ld/sections.ld](helpers/ld/sections.ld) (named high VMAs at the top) |

Note that our port will be further simplified somewhat when the now-deprecated `DEBUG_PRINT` gets deleted from Metal.

## Files of interest

- [helpers/include/dprint.h](helpers/include/dprint.h): kernel-side wrapper
- [python_tests/helpers/device_print.py](python_tests/helpers/device_print.py): parser, `run_with_device_print`
- [python_tests/helpers/test_config.py](python_tests/helpers/test_config.py): `DevicePrintBuild`, `RISC_INFO`, `-D` injection
- [helpers/ld/sections.ld](helpers/ld/sections.ld): string tables
- [sources/device_print_test.cpp](sources/device_print_test.cpp) + [python_tests/test_device_print.py](python_tests/test_device_print.py): complete example
