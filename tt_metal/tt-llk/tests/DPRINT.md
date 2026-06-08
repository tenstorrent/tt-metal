# Device Print

`DEVICE_PRINT` is a formatted printing facility that lets LLK code efficiently emit logs that the Python test harness picks up and renders.

## Quick start

Add a print in your kernel:

```cpp
#include "dprint.h"

void run_kernel(RUNTIME_PARAMETERS)
{
    DEVICE_PRINT("test: i32={} hex={:08x}\n", (int32_t)-1, (uint32_t)0xDEADBEEF);
    DEVICE_PRINT("string: {}\n", CTSTR("done"));
}
```

Run pytest at `--logging-level=debug` (or `trace`):

```bash
pytest --logging-level=debug ./python_tests/test_foo.py
```

That's it. At debug/trace logging, every test compiles with device print enabled and lines stream into the pytest log at `DEBUG` level. They're visible in the terminal and are written to `test_run.log`.

At any coarser level (INFO and up, which is the default), every `DEVICE_PRINT(...)` call compiles to nothing. Coverage builds also force it off, because the coverage linker scripts currently grow TRISC sections past the device print buffer slot.

## Public API

Defined in [helpers/include/dprint.h](helpers/include/dprint.h):

- `DEVICE_PRINT(fmt, ...)`: Emit a record. Uses fmtlib-style placeholders.
- `CTSTR(literal)`: Save a string literal into the ELF and pass it as an argument. Use this for all string literals (except the format strings themselves).

## Format specifiers

`DEVICE_PRINT` uses fmtlib syntax. Common forms:

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

## Tests that assert on prints

`TestConfig.run()` returns a `TestOutcome` whose `device_print_lines` field holds every line emitted by `DEVICE_PRINT()` during that run, in order. When device print isn't on for the session the list is empty.

```python
def test_my_kernel():
    outcome = TestConfig("sources/my_kernel_test.cpp", formats).run()
    assert "expected output" in "".join(outcome.device_print_lines)
```

### Self-enabling a test in CI

CI runs the LLK suite at `LOGURU_LEVEL=INFO`, which leaves `TestConfig.DEVICE_PRINT_ENABLED = False`. A test that just asserts on `outcome.device_print_lines` would see an empty list by default.

`test_device_print.py` works around this with a module-scoped autouse fixture that forces `TestConfig.DEVICE_PRINT_ENABLED = True` for the duration of the module and restores it on teardown:

```python
@pytest.fixture(scope="module", autouse=True)
def _force_device_print_enabled():
    prev = TestConfig.DEVICE_PRINT_ENABLED
    TestConfig.DEVICE_PRINT_ENABLED = True
    yield
    TestConfig.DEVICE_PRINT_ENABLED = prev
```

Use this same pattern for any other test that asserts on device print output.
