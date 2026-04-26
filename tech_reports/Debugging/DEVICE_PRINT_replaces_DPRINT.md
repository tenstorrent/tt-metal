# Deprecating `DPRINT` in favor of `DEVICE_PRINT`

**TL;DR**

- Use `DEVICE_PRINT` for all new kernels and firmware.
- During the deprecation window, set `TT_METAL_DEVICE_PRINT=1` to opt into the new system; legacy `DPRINT` still works but is deprecated.
- After the window, `DPRINT` remains only as a thin alias for `DEVICE_PRINT`; old stream-style `DPRINT << ... << ENDL();` will not compile.

We are deprecating the legacy **`DPRINT`** device-side debug print API in favor of the new **`DEVICE_PRINT`** system. `DEVICE_PRINT` is now our recommended, production-ready mechanism for printing from firmware and kernels. During a transition period, `DPRINT` will remain available, after which it will be removed.

---

## 1. Why we are changing it

The legacy `DPRINT` implementation has several limitations:

- **Device memory footprint:**
  `DPRINT` embeds format strings in global sections that are copied to device memory, increasing risc local data memory (LDM) usage and scaling poorly as more prints are added.

- **C++ stream-style syntax:**
  `DPRINT` relies on a custom `<<` stream API and helper manipulators (`DEC()`, `HEX()`, `SETW()`, `ENDL()`, etc.), which is verbose and harder to read and maintain.

- **Limited and ad-hoc type support:**
  Adding new types or richer formatting is awkward and requires more custom helpers.

`DEVICE_PRINT` addresses these issues:

- **Host-only format strings:**
  `DEVICE_PRINT` stores strings in dedicated ELF sections that won't be loaded into device memory and will live only on host. This keeps device-side footprint small and bounded.

- **Fmt-style format strings:**
  It uses a `fmt`-like format syntax (`"value = {}"` with `{:.3f}`, `{:#x}`, alignment, width, etc.), with compile-time checks for format/argument mismatches.

- **Richer type and tooling support:**
  It adds first-class support for:
  - Enums (decoded to names via DWARF when debug info is present).
  - Tiles / tile slices via `TSLICE(...)`.
  - A broad set of built-in integer and floating-point types.
  - A dedicated host-side server/parser with tests and diagnostics.

Overall, `DEVICE_PRINT` gives more readable code, better diagnostics, and a smaller, more predictable device footprint.

---

## 2. Deprecation plan

We are following a staged deprecation plan to minimize disruption.

### Phase 1 – Opt‑in (already available)

- `DEVICE_PRINT` is shipped alongside `DPRINT`.
- To enable the new system, you:
  1. Enable device printing as usual.
  2. Export the feature toggle:
     ```bash
     export TT_METAL_DEVICE_PRINT=1
     ```
- With this flag set, the runtime uses the `DEVICE_PRINT` infrastructure instead of the legacy `DPRINT` backend. Only one printing mechanism is active at a time (they share the same L1 buffer).

### Phase 2 – Deprecation window (current / upcoming)

During this period:

- `DEVICE_PRINT` is treated as **production-ready**, and **all new code should use `DEVICE_PRINT`** directly.
- Existing `DPRINT` usage:
  - Continues to compile.
  - Will show **deprecation warnings** during execution.
- Internally we are:
  - Duplicating important `DPRINT` sites with equivalent `DEVICE_PRINT` calls.
  - Porting `DPRINT` tests to `DEVICE_PRINT` tests.
  - Updating documentation, labs, and programming examples to show only `DEVICE_PRINT`.

This deprecation window will last **at least one month** from when `DEVICE_PRINT` becomes the default implementation, to give downstream users time to migrate.

### Phase 3 – Alias and cleanup

After the deprecation window:

- We will:
  - Remove all remaining internal `DPRINT` usage from the codebase (firmware, kernels, tests, and docs).
  - Introduce a user-facing alias, for example:
    ```cpp
    #define DPRINT       DEVICE_PRINT
    #define DPRINT_MATH  DEVICE_PRINT_MATH
    // ...and so on for the other variants
    ```
- At this point:
  - `DEVICE_PRINT` is the only implementation used under the hood.
  - `DPRINT` exists only as a shorter spelling that forwards to `DEVICE_PRINT`.
  - `TT_METAL_DEVICE_PRINT` is no longer required and is ignored.

**What this means for you:**

- Start using `DEVICE_PRINT` for all new kernels and firmware.
- During the deprecation window, gradually migrate existing `DPRINT` call sites to `DEVICE_PRINT` (recommended), or rely on the post-window alias if you prefer the shorter spelling.
- After the deprecation window, `DPRINT` is only available as a function-style alias (for example, `DPRINT("value = {}\\n", v)`); legacy stream-style usage such as `DPRINT << "value = " << v << ENDL();` will fail kernel compilation and must be rewritten to the new `DEVICE_PRINT`/`DPRINT("...", ...)` style.

---

## 3. Usage examples – `DPRINT` vs `DEVICE_PRINT`

### 3.1 Basic value printing

**Before – `DPRINT` (stream-style):**
```cpp
// Print a character, an int, and a float.
DPRINT << "Test string "
       << 'a'
       << " "
       << 5
       << " "
       << 0.123456f
       << ENDL();
```

**After – `DEVICE_PRINT` (fmt-style):**
```cpp
// Same output, more compact and readable.
DEVICE_PRINT("Test string a {} {}\n", 5, 0.123456f);
```

### 3.2 Number formatting

**Before – `DPRINT` with helpers:**
```cpp
// Hex, decimal, octal, binary representations of the same value.
DPRINT << HEX() << "0x" << SETW(8) << value << " ";
DPRINT << DEC() << value << " ";
DPRINT << OCT() << value << " ";
DPRINT << BIN() << value << ENDL();
```

**After – `DEVICE_PRINT` with format specifiers:**
```cpp
DEVICE_PRINT("{0:#010x} {0} {0:o} {0:b}\n", value);
//        ^ hex with 0x and width    ^dec  ^oct   ^binary
```

### 3.3 Enum printing

**Before – `DPRINT`:**
```cpp
// Often required manual mapping to names:
DPRINT << "Mode: ";
if (mode == Mode::Idle) {
    DPRINT << "Idle";
} else if (mode == Mode::Running) {
    DPRINT << "Running";
} else {
    DPRINT << "Unknown(" << static_cast<uint32_t>(mode) << ")";
}
DPRINT << ENDL();
```

**After – `DEVICE_PRINT` (DWARF-aware enums):**
```cpp
// Automatically prints enum name when debug info is available.
DEVICE_PRINT("Mode: {}\n", mode);

// Or include fully qualified enum type name:
DEVICE_PRINT("Mode: {:#}\n", mode);
```

### 3.4 Tile / `TileSlice` printing

Both `DPRINT` and `DEVICE_PRINT` support `TileSlice` via the `TSLICE(...)` helper; the main difference is the formatting style (stream-style for `DPRINT`, fmt-style for `DEVICE_PRINT`).

**Before – `DPRINT` with `TSLICE(...)`:**
```cpp
// Stream-style TileSlice printing.
DPRINT << "Tile from Data0:\n";
DPRINT << TSLICE(cb_id,
                 /*tile_index=*/0,
                 SliceRange::hw0_32_8(),
                 TSLICE_INPUT_CB,
                 TSLICE_RD_PTR,
                 /*print_coords=*/true,
                 /*is_tilized=*/is_tilized)
       << ENDL();
```

**After – `DEVICE_PRINT` with `TSLICE(...)`:**
```cpp
// Let DEVICE_PRINT serialize and format the same TileSlice.
auto slice = TSLICE(cb_id,
                    /*tile_index=*/0,
                    SliceRange::hw0_32_8(),
                    TSLICE_INPUT_CB,
                    TSLICE_RD_PTR,
                    /*print_coords=*/true,
                    /*is_tilized=*/is_tilized);

DEVICE_PRINT("Tile from Data0:\n{}\n", slice);
// Or control numeric formatting with fmt-style specifiers:
DEVICE_PRINT("Tile from Data0:\n{:.4f}\n", slice);
```
### 3.5 Core-specific prints

The **intent** of the core-specific macros is unchanged; you primarily update the macro names.

**Before – `DPRINT` variants:**
```cpp
DPRINT_MATH(   DPRINT << "this is the math kernel"   << ENDL() );
DPRINT_PACK(   DPRINT << "this is the pack kernel"   << ENDL() );
DPRINT_UNPACK( DPRINT << "this is the unpack kernel" << ENDL() );
DPRINT_DATA0(  DPRINT << "this is the data movement kernel on noc 0" << ENDL() );
DPRINT_DATA1(  DPRINT << "this is the data movement kernel on noc 1" << ENDL() );
```

**After – `DEVICE_PRINT` variants:**
```cpp
DEVICE_PRINT_MATH("this is the math kernel\n");
DEVICE_PRINT_PACK("this is the pack kernel\n");
DEVICE_PRINT_UNPACK("this is the unpack kernel\n");
DEVICE_PRINT_DATA0("this is the data movement kernel on noc 0\n");
DEVICE_PRINT_DATA1("this is the data movement kernel on noc 1\n");
```

### 3.6 Enabling `DEVICE_PRINT` during the transition

To opt into `DEVICE_PRINT` on an existing setup that already enables device printing:

```bash
# Existing DPRINT enablement (unchanged).
export TT_METAL_ENABLE_DPRINT=1          # example; use your current setting

# New feature flag to switch to DEVICE_PRINT backend:
export TT_METAL_DEVICE_PRINT=1
```

With this configuration:

- Your new `DEVICE_PRINT(...)` calls will use the new system.
- Legacy `DPRINT` calls will be ignored by the compiler (only `DEVICE_PRINT`-style usage is honored when `TT_METAL_DEVICE_PRINT` is set).

If you **do not** set `TT_METAL_DEVICE_PRINT`:

- `DEVICE_PRINT` calls will be ignored.
- Legacy `DPRINT` will be used, but with a deprecation warning message reminding you to migrate to `DEVICE_PRINT` and to enable it via `TT_METAL_DEVICE_PRINT`.

---

In summary:

- **Why**: `DEVICE_PRINT` gives better readability, functionality, and device-footprint characteristics than `DPRINT`.
- **Plan**: we are transitioning through opt‑in → deprecation window → alias → removal, with at least one month of overlap before removal.
- **Usage**: you get a fmt-style API, richer types (enums, tiles), and mostly mechanical changes from `DPRINT` → `DEVICE_PRINT`.
