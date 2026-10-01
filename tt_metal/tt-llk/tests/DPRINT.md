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

## Printing arrays, tiles, and DEST

Beyond scalars, `DEVICE_PRINT` can dump rows of data in a given `DataFormat`:

### Typed arrays (`dp_typed_array_t`)

Pulled in through [dprint.h](helpers/include/dprint.h). It wraps a `uint32_t` buffer plus a `DataFormat`.

```cpp
std::uint32_t arr[4] = {0x3F800000, 0x40000000, 0x40400000, 0x40800000};
DEVICE_PRINT("{}", dp_typed_array_t<4>(static_cast<std::uint16_t>(DataFormat::Float32), arr));
// ->  1.00   2.00   3.00   4.00
```

The template parameter is the buffer length in `uint32_t` words; pack narrower elements (Float16, Int8, ...) into the words yourself and pass the matching format code.

### Tile slices

Defined in [api/debug/dprint_tile.h](../../hw/inc/api/debug/dprint_tile.h). The `tile_slice_from_l1<MAX_BYTES>(l1_addr, fmt, sr, ...)` helper fills a tile straight from an L1 address, decoding it as `fmt` over the cells picked by `SliceRange sr`. Tile and face dimensions default to the standard 32x32 layout but can be passed explicitly as trailing arguments.

```cpp
#include "dprint.h"

const DataFormat fmt = static_cast<DataFormat>(formats.unpack_A_src);
DEVICE_PRINT("{}", tile_slice_from_l1<64>(params.buffer_A[0], fmt, SliceRange::hw0_32_8()));
```

- `SliceRange` (e.g. `hw0_32_8()`, `hw0_32_4()`) selects which cells of the tile to sample.
- `MAX_BYTES` (template param, default 64) caps the captured payload. If the slice produces more data than fits, the row is emitted truncated.

### DEST register dump

Defined in [dprint_tensix.h](helpers/include/dprint_tensix.h). Dumps the Tensix DEST register, one row per DEST row, decoded against the configured DEST data format. (Wormhole and Blackhole only, Quasar is currently unsupported.) Call it from the **MATH** thread, after filling DEST and before `dest_section_done`:

```cpp
#include "dprint.h"

// ... Operations on DEST ...
dprint_tensix_dest_reg(params.DST_INDEX);   // prints "Tile ID = N" + one row per DEST row
_llk_math_dest_section_done_<...>();
```

## Tests that assert on prints

`TestConfig.run()` returns a `TestOutcome` whose `device_print_lines` field holds every line emitted by `DEVICE_PRINT()` during that run, in order. When device print isn't on for the session, the list is empty.

```python
def test_my_kernel():
    outcome = TestConfig("sources/my_kernel_test.cpp", formats).run()
    assert "expected output" in "".join(outcome.device_print_lines)
```

### Self-enabling a test in CI

CI runs the LLK suite at `LOGURU_LEVEL=INFO`, which leaves the global `TestConfig.DEVICE_PRINT_ENABLED = False`. A test that just asserts on `outcome.device_print_lines` would see an empty list by default.

Pass `requires_device_print=True` to `TestConfig(...)` to opt that variant into device print regardless of logging level:

```python
outcome = TestConfig(
    "sources/my_kernel_test.cpp",
    formats,
    requires_device_print=True,
).run()
assert "expected output" in "".join(outcome.device_print_lines)
```
