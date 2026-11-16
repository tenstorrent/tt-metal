# DPRINT implementation

This is today's implementation of the [DPRINT specification](DEVICE_PRINT_spec.md).
[Device Debug Print](../../docs/source/tt-metalium/tools/device_print.rst) documents how DPRINT is used. The streaming
pipeline will replace the transport (§3).

## 1. Code

| Part | Where |
|---|---|
| Message creation (§2): macros, compile-time checks, call-site records, message layout | `tt_metal/hw/inc/api/debug/device_print.h`; `dprint.h` (the `DPRINT` names and the legacy-syntax error); `tt_metal/hw/inc/hostdev/device_print_structures.h` (message header, string-info entries) |
| Never-loaded ELF sections | `tt_metal/hw/toolchain/main.ld`, `script_tng.ld` and `erisc-b0-kernel.ld`; the loader's `.device_print` exception in `tt_metal/llrt/tt_elffile.cpp` |
| Transport (§3): print memory | `tt_metal/hw/inc/hostdev/device_print_common.h` (buffer layout), the flag values in `tt_metal/hostdevcommon/api/hostdevcommon/dprint_common.h`, `tt_metal/hw/inc/internal/tt-*/*/device_print_mem.h` (sizes), `tt_metal/hw/inc/internal/debug/dprint_buffer.h` (`get_device_print_buffer()`), the `dprint_buf` mailbox field in `tt_metal/hw/inc/hostdev/dev_msgs.h`, and `HalL1MemAddrType::DPRINT_BUFFERS` in the per-core HAL files under `tt_metal/llrt/hal/` |
| Transport: firmware hooks | `DEVICE_PRINT_INITIALIZE_LOCK()` and `DEVICE_PRINT_KERNEL_FINISHED()` in `tt_metal/hw/firmware/src/tt-1xx/` and `tt-2xx/` |
| Transport: fast dispatch | `tt_metal/impl/dispatch/kernels/device_print_dispatch.h` (the aggregator), `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp` (dispatch_s, which runs it), `tt_metal/impl/dispatch/kernel_config/dispatch_s.cpp` (its configuration), `tt_metal/impl/dispatch/dispatch_mem_map.cpp` (its L1), the aggregator records in `device_print_structures.h`, and `HalDramMemAddrType::DEVICE_PRINT_DISPATCH` (its DRAM ring) |
| Transport: host | `tt_metal/impl/debug/dprint_server.cpp` (attaching, polling, `await()`) |
| Decoding (§4) | `tt_metal/impl/debug/dprint_server.cpp` (line assembly, output files), `dprint_parser.cpp` (ELF sections, format strings, enums and call stacks from DWARF), and the Inspector (`tt_metal/impl/debug/inspector/`), which maps kernel ids to ELF paths |
| Library (§5) | `dprint_tile.h`, `dprint_tensix.h`, `dprint_tensix_pack.h`, `dprint_tensix_unpack.h`, `dprint_pages.h` and `dump.h` in `tt_metal/hw/inc/api/debug/`; `SliceRange` and the `TileSlice` host view in `dprint_common.h`; typed arrays and call stacks in `device_print.h`; their host rendering in `dprint_parser.cpp` |
| Debug features that use DPRINT | checkpoints (`tt_metal/hw/inc/api/debug/checkpoint.h`), CB hash (`tt_metal/hw/inc/api/compute/debug/cb_hash.h`) and the LLK sanitizer (`tt_metal/tt-llk/tools/include/sanitizer/`), which reports through `CTSTR`, `dp_type_name_t`, `dp_top_callstack_t` and `\r` lines, and whose host tests pass its formats to `fmt::print` |
| Build flags | `tt_metal/jit_build/build.cpp` (`-DDEBUG_PRINT_ENABLED`), `tt_metal/impl/dispatch/kernel_config/fd_kernel.cpp` (`FORCE_DPRINT_OFF`), the HAL (`PROCESSOR_INDEX`) |
| Headers installed for kernel builds | `HW_JIT_API_HEADERS` in `tt_metal/hw/sources.cmake`, `HOSTDEVCOMMON_JIT_API_HEADERS` in `tt_metal/hostdevcommon/sources.cmake`, `JITAPI_FILES` in `tt_metal/sources.cmake` |
| Configuration | `tt_metal/llrt/rtoptions.cpp` (`TT_METAL_DPRINT_*`, `TT_METAL_DEVICE_PRINT_DISPATCH_*`); `Hal::parse_processor_set_spec` in `tt_metal/llrt/hal.cpp` (RISC selection) |
| Tests (§8) | `tests/tt_metal/tt_metal/debug_tools/device_print/`, fixtures in `tests/tt_metal/tt_metal/debug_tools/debug_tools_fixture.hpp`, kernels in `tests/tt_metal/tt_metal/test_kernels/device_print/` |
| Documentation | the manual; the migration guide `tech_reports/Debugging/DEVICE_PRINT_replaces_DPRINT.md` |
| tt-llk's own host side (§7) | `tt_metal/tt-llk/tests/python_tests/helpers/device_print.py` (decoder) and `test_config.py` (build flags, polling); `tt_metal/tt-llk/tests/helpers/include/dprint.h` (what its kernels include) |

## 2. Message creation

- `DEVICE_PRINT` expands to three calls that each take the arguments by value: one runs the compile-time checks and
  registers the call site, one builds the message header, and one stores the arguments.
- With printing off, `DEVICE_PRINT` expands to nothing, so a variable or parameter that only prints use is left
  unused. The JIT build doesn't warn about unused variables or parameters and only warns about set-but-unused ones
  (§6); tt-llk's stricter build fails on all three (§7).
- The LLK sanitizer (`tt_metal/tt-llk/tools/include/sanitizer/`) prints through its own `SAN_PRINT`. With printing
  off, `SAN_PRINT` puts its arguments in a `sizeof`, which counts them as used without producing code. C++17 bars
  lambdas from `sizeof`, and `CTSTR` expands to one, so the sanitizer includes `device_print.h` only when printing is
  on (`deps/device.h`). A `DEVICE_PRINT` that named its arguments in a `sizeof` when off would fail on `CTSTR`
  arguments the same way. The sanitizer's functions are built with `[[gnu::optimize("Os")]]` (`SAN_FUNC`) inside
  `-O3` compute kernels, and `DEVICE_PRINT`'s `always_inline` lambdas inline into them.
- Each call site defines a local tag type, so it gets its own `register_string_info<Tag, Args...>()`. That puts a
  `DevicePrintStringInfo` (format, file and line) in `.device_print_strings_info` and the strings in
  `.device_print_strings`. The call site's id is the entry's index from `__device_print_strings_info_start`.
- `CTSTR` puts its literal in `.device_print_strings` too, and a `ct_string` holds the literal's address. The device
  sends it like a `const char*`, and the host prints the text when the address falls inside that section.
- The stored format is rewritten at compile time. Each placeholder becomes `{slot,type[:spec]}` with a one-letter
  type (`b B h H i I q Q` integers, `f d` floats, `?` bool, `e E w` block floats, `s` strings, `p` pointers, `t`
  tiles, `A` typed arrays, `c` call stacks), or `/e_<type>_<Name>` for an enum and `/E_<type>_<Name>` for a flag enum.
  The arguments are stored largest first, and `slot` is an argument's position in that order.
- A message is a 4-byte header, `is_kernel:1 | risc_id:5 | message_payload:10 | info_id:16`
  (`device_print_structures.h`), followed by the arguments, padded to 4 bytes. Messages are only 4-byte aligned, and
  Quasar's 64-bit RISCs, which store 8 bytes in one instruction, fault on such a store to an address that isn't
  8-byte aligned. So an 8-byte value is stored as two 4-byte halves. That costs less memory than 8-byte headers with
  every message aligned to 8 bytes, because 8-byte values are rare.
- A print names its RISC with `PROCESSOR_INDEX`. The HAL defines it everywhere except on Quasar's DMs, TRISCs and
  dispatch engines, where `device_print.h` falls back to `internal_::get_hw_thread_idx()`, a thread-local index that
  firmware sets up during boot, so firmware prints only after that.
- A call site hands its message to the transport through two shared functions: `begin_message_write()` writes the
  header and returns where the arguments go, the call site stores them there, and `end_message_write()` finishes the
  message (§3.1).

## 3. Transport

Today's two transports share the device side. In slow dispatch the host reads each core's print memory, and in fast
dispatch dispatch_s copies it into DRAM, where the host reads it. The streaming pipeline will replace both.

### 3.1 Print memory

- Each core's print memory is a fixed region of its L1 mailbox, the `dprint_buf` field of `mailboxes_t`, sized per
  core type in `device_print_mem.h` (the table below gives the sizes) and bounded by the mailbox sizes in each
  `dev_mem_map.h`. The host finds it through `HalL1MemAddrType::DPRINT_BUFFERS`. The RISCs of a core share one
  region, except on Quasar, where the DMs and the TRISCs have one each. Growing it is a memory-map change. NOC
  transfer recording, which can't run with DPRINT, reuses 128 bytes of the region per RISC through
  `get_device_print_buffer()` and `DPRINT_BUFFERS`.
- Each print region is a `DevicePrintBuffer` (`device_print_common.h`): read and write positions, each RISC's state
  (kernel announced or not, printing disabled), a lock, and the data. Quasar's header takes 128 bytes, so that the
  lock gets a cache line of its own: the lock is swapped through the cached L1 alias while the rest is written
  uncached or over the NOC, and writing back a dirty line would overwrite its neighbours. The data is a ring in which
  equal positions mean empty, so the write position never catches up with the read position.
- A region that several RISCs share is locked. Wormhole has no atomic swap, so a RISC writes its index into the lock
  and reads it back several times to confirm. Blackhole and Quasar swap atomically, Quasar through the cached L1
  alias. After taking the lock, a RISC invalidates its L1 cache, because on Blackhole and Quasar it sees L1 that
  others wrote only after a fence or a cache invalidation.
- `begin_message_write()` takes the lock, waits for space, and writes the header. A message that doesn't fit before
  the end of the data goes at its start, after a header-only record with info id 0xFFFF where that fits.
  `end_message_write()` moves the write position past the message and releases the lock.
- A RISC's first print after a kernel starts is preceded by a header-only record with payload 1023 and the kernel id
  as its `info_id`, which tells the host which kernel's ELF decodes that RISC's messages.
- The positions double as flags (`hostdevcommon/dprint_common.h`): write position 0xf8f8f8f8 means printing is
  disabled, 0x98989898 that the host has just attached, read position 0xF0E1D2C3 a reset, and bit 31 of the write
  position that the writer waits for space.
- A RISC waits for space by spinning, but a disabled, just-attached or reset region doesn't make it wait. While it
  spins, an ERISC yields with `internal_::risc_context_switch()`, the rule for all ERISC code that waits. Because of
  that yield, the fabric router and EDM adapters turn off their stateful NOC paths when `DEBUG_PRINT_ENABLED` is
  defined.
- At boot one firmware per core calls `DEVICE_PRINT_INITIALIZE_LOCK()`: BRISC, the first ERISC, DRISC, and DM0 on
  Quasar. Every firmware calls `DEVICE_PRINT_KERNEL_FINISHED()` after each kernel, so that the RISC announces the next
  one. Quasar's `dm.cc` also resets the TRISC print state.

The print regions that the memory map gives DPRINT:

| Core | Region | Data after the header | RISCs |
|---|---|---|---|
| Wormhole and Blackhole Tensix | 1020 bytes | 1000 bytes | 5 |
| Wormhole ERISC | 204 bytes | 188 bytes | 1 |
| Blackhole ERISC | 408 bytes | 392 bytes | 2 |
| Blackhole DRISC | 204 bytes | 188 bytes | 1 |
| Quasar Tensix, DMs | 1632 bytes | 1504 bytes | 8 |
| Quasar Tensix, TRISCs | 3264 bytes | 3136 bytes | 16 |
| Quasar dispatch engine | 1632 bytes | 1504 bytes | 8 |

### 3.2 Slow dispatch

- The host reads each selected core's region directly and sleeps 1 ms between idle passes. It does the same in fast
  dispatch whenever dispatch_s isn't draining.

### 3.3 Fast dispatch

- dispatch_s runs an aggregator (`device_print_dispatch.h`) that copies regions through an L1 cache into a DRAM ring,
  which the host reads. Every 50 µs it drains the regions whose writer waits for space, and every 100 ms all of them.
  `TT_METAL_DEVICE_PRINT_DISPATCH_STALL_US` and `TT_METAL_DEVICE_PRINT_DISPATCH_FULL_US` set the two periods, and
  `TT_METAL_DEVICE_PRINT_DISPATCH_L1_CACHE_BYTES` the cache size.
- dispatch_s is then the only thing that drains the regions, so its waits keep draining: a print on dispatch_s that
  waits for space runs the aggregator itself (`device_print_dispatcher_execute_hook()`). A wait that doesn't drain
  deadlocks.
- The aggregator splits its NOC reads and writes into packets of at most `NOC_MAX_BURST_SIZE`. The NIU splits a
  bigger transfer itself and counts every packet, while the software counts the transfer once, so the read and write
  barriers would never be satisfied. Blackhole DRISC regions are read at the 64-byte DRAM alignment
  (`use_dram_alignment`), and other regions at 16 bytes.

### 3.4 Host

- `MetalContext` creates and destroys `DPrintServer` (`dprint_server.cpp`), and dispatch setup asks it whether
  dispatch cores print. The server attaches each device before firmware starts, so boot prints are kept. It writes the
  disabled flag into the regions of unselected cores and the just-attached flag into selected ones, marks RISCs
  excluded by `TT_METAL_DPRINT_RISCVS` as printing-disabled, and starts polling.
- A message reaches the host within one poll in slow dispatch, and within 100 ms in fast dispatch, or 50 µs when its
  writer waits for space. Hang handlers reset the device after 5 s (the LLK-assert flow's `tt-smi -r`).
- `await()`, which the debug-tools fixtures, several tests and `test_device_pcie_loopback` call, sleeps two
  full-dispatch periods and then waits until two polls find nothing new.
- Teardown detaches the devices without a final read.

## 4. Decoding

- The server knows the device, core and RISC of each message. A message with `is_kernel` set is decoded with the ELF
  of the kernel that its RISC announced (§3.1), whose path the Inspector gives for the kernel id, and any other
  message with the RISC's firmware ELF, whose path the build environment gives.
- `DevicePrintParser` (`dprint_parser.cpp`) reads the two sections from that ELF, parses the rewritten formats, reads
  enum names and call stacks from DWARF, and formats each value with fmt.
- Per RISC, the server assembles lines, holds `\r` blocks until the next `\n`, adds the prefix, and writes to stdout,
  a file or one file per RISC.

## 5. Library

- The library is in `tt_metal/hw/inc/api/debug/`: tiles in `dprint_tile.h` (`TileSlice`, `TSLICE`,
  `tile_slice_from_l1`; `SliceRange` is in `hostdevcommon/dprint_common.h`), typed arrays and call stacks in
  `device_print.h` (`dp_typed_array_t`, `dp_top_callstack_t`), DEST and configuration registers in `dprint_tensix.h`,
  `dprint_tensix_pack.h` and `dprint_tensix_unpack.h`, and circular buffers and L1 in `dprint_pages.h` and `dump.h`.
- Printing and the library aren't separate today. Tiles, typed arrays and call stacks are argument types that
  printing itself knows: `device_print.h` includes `dprint_tile.h` in kernel builds and serializes them with their own
  type letters (`t`, `A`, `c`), and `dprint_parser.cpp` renders them. It decodes tile and typed-array data by data
  format and resolves call stacks from DWARF through tt-exalens.
- A `TileSlice` samples its tile on the device when it's constructed. On a RISC that can't read circular buffers, or
  for a tile it can't read, it sends an error code instead, and the host prints a warning.
- The register and memory dumps are functions made of ordinary prints. `dprint_tensix.h` reads DEST rows and prints
  each as a typed array, the pack and unpack headers print configuration fields, and `dprint_pages.h` and `dump.h`
  print circular-buffer state, pages and L1 words.
- With printing off, `TileSlice` and `SliceRange` stay declared and `dump.h` has empty versions of its functions, but
  `dprint_pages.h`'s page and tile printers aren't declared, so calls to them need a guard, and the `dprint_tensix*.h`
  dumps still read their registers; only their prints disappear.

## 6. Build and configuration

- Kernels build as C++17 with the toolchain's `-ftt-*` extensions (including `-ftt-no-dyninit`, which bans dynamic
  initialization), without exceptions or RTTI, and with `-Wall -Werror`. `build.cpp` turns off the unused-variable
  and unused-function warnings and keeps a few others, set-but-unused variables among them, from failing the build.
- The JIT build adds `-DDEBUG_PRINT_ENABLED` when any core list is set, and only that enters a user kernel's build
  hash: one kernel binary runs on selected and unselected cores. Dispatch kernels get `FORCE_DPRINT_OFF` unless
  dispatch cores are selected. The HAL defines `PROCESSOR_INDEX` on most RISCs (§2).
- `rtoptions.cpp` reads the `TT_METAL_DPRINT_*` variables, and `Hal::parse_processor_set_spec` maps the names in
  `TT_METAL_DPRINT_RISCVS` to fixed Wormhole and Blackhole processor indices.
- The linker scripts (`main.ld`, `script_tng.ld`, `erisc-b0-kernel.ld`, and tt-llk's `tests/helpers/ld/sections.ld`)
  put `.device_print_strings` at 0x6400000 and `.device_print_strings_info` at 0x6500000 as never-loaded sections,
  below the profiler's sections at 0x6600000, and define `__device_print_strings_info_start`. A section that a script
  doesn't list can land in L1; `erisc-b0-app.ld`, for Wormhole active-ETH firmware, lists none. The loader accepts an
  allocated section outside every segment only if its name starts with `.device_print`.
- Code outside DPRINT tests its macros: kernels, helper headers, the fabric router and EDM adapters check
  `DEBUG_PRINT_ENABLED` (the router and the adapters because of the ERISC yield, §3.1), and other code tests
  `FORCE_DPRINT_OFF` and `PROCESSOR_INDEX`.
- New DPRINT headers go into `HW_JIT_API_HEADERS` (`tt_metal/hw/sources.cmake`), `HOSTDEVCOMMON_JIT_API_HEADERS`
  (`tt_metal/hostdevcommon/sources.cmake`) or `JITAPI_FILES` (`tt_metal/sources.cmake`). Installed builds compile
  kernels against those lists while source builds read the repo, so a missing header breaks only installed builds.

## 7. tt-llk

tt-llk's test harness (`tt_metal/tt-llk/tests/`) builds its kernels with DPRINT's device headers and has its own host
side, written in Python, which runs without the tt-metal runtime. tt-llk keeps that host side compatible with DPRINT.

What tt-llk has:

- `test_config.py` builds every kernel with `-DENV_LLK_INFRA`. Printing is on in tests that need it and under
  `pytest --logging-level=debug` or `trace`, and then each kernel also gets `-DDEBUG_PRINT_ENABLED`, its RISC's
  `PROCESSOR_INDEX`, the region's address and sizes (`LLK_DEVICE_PRINT_BUFFER_BASE`, `DEVICE_PRINT_BUFFER_SIZE`,
  `DEVICE_PRINT_BUFFER_SIZE2`) and `LLK_RUNTIME_ARGS_START`.
- It warns more than the JIT build: it adds `-Wunused-parameter -Wfloat-equal -Wpointer-arith -Wnull-dereference
  -Wredundant-decls -Wuninitialized -Wmaybe-uninitialized` to `-Wall -Werror`, and unused variables and functions stay
  errors.
- Kernels include tt-llk's `tests/helpers/include/dprint.h`. It defines `invalidate_l1_cache()`, which tt-metal
  kernels get through `risc_common.h`, includes `api/debug/device_print.h` and `api/debug/dprint_tile.h`, and checks
  that the region ends before the runtime arguments. With printing off it defines an empty `DEVICE_PRINT` instead.
  Its `dprint_tensix.h` defines `DPRINT` and wraps tt-metal's `api/debug/dprint_tensix.h`.
- Under `ENV_LLK_INFRA`, DPRINT's headers find the region at `LLK_DEVICE_PRINT_BUFFER_BASE` instead of the mailbox,
  take its sizes from the two size macros, and leave out what needs tt-metal's firmware and runtime (`risc_common.h`,
  `dev_msgs.h`, circular buffers, the compute API). Quasar's lock doesn't get a cache line of its own, and no RISC
  announces its kernel.
- Before each run the harness zeroes the region's header, so the first print finds the region empty and the lock
  free. It polls while the kernel runs and drains once more at the end.
- The decoder, `tests/python_tests/helpers/device_print.py`, is a port of `dprint_parser.cpp`. It reads the region and
  the ELF sections through tt-exalens and writes back only the read position, never the reset flag. It prints each
  message on its own line with a `[RISC|file:line] ` prefix and formats values with Python's `format()`, so a float
  `{}` prints `1.0`.
- `tests/python_tests/test_device_print.py` checks that output, with kernels in `tests/sources/`. The tensor-shape
  coverage tables (`tt_metal/tt-llk/common/tensor_shape_coverage_*.h`) are regenerated from what LLK prints in
  tt-llk's tests, by `tests/python_tests/helpers/tensor_shape_coverage_parser.py`.

When changing DPRINT:

- The decoder repeats today's binary format: the `DevicePrintBuffer` header, the message header with its 1023 and
  0xFFFF markers, the stall bit, the section names and `DevicePrintStringInfo`, the rewritten placeholders and their
  type letters, the typed-array and tile-slice encodings, and each architecture's data-format numbers. tt-llk's
  linker script repeats the section addresses (§6). Changing any of these breaks tt-llk's tests until tt-llk updates
  its side.
- The device headers build in tt-llk's environment too. Keep the `ENV_LLK_INFRA` branches and the macros the harness
  passes, and keep the headers warning-free under its flags. A new include has to resolve from tt-llk's include paths
  without tt-metal's firmware, or go in an `ENV_LLK_INFRA` branch.
- `device_print.h` doesn't define `DPRINT`, because tt-llk's `dprint_tensix.h` does.

## 8. Tests

DPRINT's tests are in `tests/tt_metal/tt_metal/debug_tools/device_print/` and their kernels in
`tests/tt_metal/tt_metal/test_kernels/device_print/`. `test_print_tensix_dest.cpp` and
`test_print_config_register.cpp` test the library, and `test_checkpoint.cpp` and `test_cb_hash.cpp` test debug
features that use DPRINT.

Writing a test:

- A test is `test_<name>.cpp`, listed in `sources.cmake`. The target is a unity build, so file-local helpers go in
  `CMAKE_UNIQUE_NAMESPACE`. A new case goes into an existing test's kernel where one fits. Each new test is broken
  once on purpose to see it fail.
- A compute kernel that reads runtime arguments includes `api/compute/common.h`, guarded by `COMPILE_FOR_TRISC` when
  data-movement RISCs build the same source.
- `DevicePrintFixture` (`debug_tools_fixture.hpp`) selects all non-dispatch Tensix, Ethernet and DRAM cores, turns the
  prefix off and writes to a fresh file. Its SetUp also resets the RISC selection and one-file-per-RISC, so
  `TT_METAL_DPRINT_*` variables can't change results. Options that the host reads when devices open are set in
  `ExtraSetUp()`. `DevicePrintSeparateFilesFixture` writes one file per RISC.
- Only fixtures call DPRINT's internal host APIs: `dprint_server()->await()`, `clear_log_file()` and the `rtoptions()`
  setters, which stand in for the documented variables.
- ERISC kernels launch only in slow dispatch, and DRISC kernels always launch through the slow-dispatch path. Ethernet
  kernels need `eth_test_common::set_arch_specific_eth_config()`, idle Ethernet tests call
  `RequestFreshDevicesAfterThisTest()`, DRISC kernels run on logical DRAM core {0,1} with the slow-dispatch launch in
  `DevicePrintDramFixture::RunDramProgram`, and Quasar uses `experimental::ProgramSpec`.
- A compile-error test is a kernel in `failures/` and a `TestCompileKernelFailure(kernel_path, message)` test in
  `test_compilation_failures.cpp`. The too-big print is bigger than all of DPRINT's device memory on the core, so it
  fails on any implementation.
- Expected text: tiles and typed arrays print what the golden generators in `test_print_tile.cpp` and
  `test_print_tensix_dest.cpp` produce, a `\r` test also checks that each block's lines stay adjacent, a call-stack
  test compares frame names and the `...` marker, and Quasar tests drop the known per-launch firmware lines while they
  exist.
- Remote-chip volume tests print from the active ERISCs of a multi-chip Wormhole system, whose base firmware carries
  the host's reads and writes to the remote chips.
- CI runs the DPRINT tests named in `tests/pipeline_reorg/*.yaml` and `tests/scripts/`, so new and renamed tests go
  there too. Build with `./build_metal.sh --build-tests`, then run
  `./build/test/tt_metal/unit_tests_debug_tools --gtest_filter='*DevicePrint*:CbHash*:Quasar*Print*'` once as is and
  once with `TT_METAL_SLOW_DISPATCH_MODE=1`. Stress tests repeat with `--gtest_repeat`.
- The JIT build turns some warnings off, so the warning checks build with the cross compiler and
  `-Wall -Wextra -Werror`.
- Most DPRINT calls are in `tt_metal/`, `tests/` and `ttnn/`. To check that every call still compiles with printing
  on, the kernels that test suites build are rebuilt by replaying their JIT commands with `-DDEBUG_PRINT_ENABLED`, and
  the rest are read.

What the fixtures lack for the spec's tests:

- `RunKernel(mesh_device, risc, kernel_path, args)`, which runs a kernel on any RISC and waits for the program and its
  output.
- `ReadAndClearOutput()`, which returns everything printed since the last call, byte for byte.
- `WaitForOutput(expected, deadline)`, which waits for text on a second thread while `RunKernel` blocks.
- Per-RISC file checks for any core and device. The files are removed first, so a missing file means no output.
- Builds with printing off.

With them, a test reads like this:

```cpp
// Kernel print_hex.cpp:
//   void kernel_main() { uint32_t x = get_arg_val<uint32_t>(0); DPRINT("x = {0}\nhex = {0:#x}\n", x); }
#include "debug_tools_fixture.hpp"
using namespace tt::tt_metal;

TEST_F(DevicePrintFixture, PrintHexNcrisc) {
    for (auto& mesh_device : this->devices_) {
        RunKernel(mesh_device, Risc::NCRISC, "tests/tt_metal/tt_metal/test_kernels/device_print/print_hex.cpp", {7});
        EXPECT_EQ(ReadAndClearOutput(), "x = 7\nhex = 0x7\n");
    }
}
```

What today's tests get wrong:

- Most output tests use `FileContainsAllStrings`, which hides missing and duplicated lines. `PrintBasicTypes` passes
  with a line it expects twice printed once, `DprintAliasMatchesDevicePrint` passes when `DPRINT` prints nothing, and
  `TwoWorkerProgramsBackToBack` can't tell whether its second run printed.
- `test_quasar_fw_boot_print.cpp` sleeps in a loop and accepts a lost TRISC line; it needs `WaitForOutput` and every
  line exactly once.
- `TensixTestPrintInvalidCore` sets its cores too late to fail; they belong in `ExtraSetUp()`.
- Some tests look inside DPRINT, which makes them helpers (spec §4.1), so what they check needs an output test too:
  `test_format_updates.cpp` reads ELF sections, `test_quasar_print_lock.cpp` reads DPRINT's device memory (a Quasar
  volume test covers it), and `test_dram_print_output.cpp` and `test_mesh_coords.cpp` check server and option state.
- Test bodies call internal host APIs: `await()` in `test_print_output.cpp` and `test_quasar_fw_boot_print.cpp`,
  `clear_log_file()` in `test_multi_kernel_print.cpp`, `test_mesh_coords.cpp` and `test_eth_cores.cpp`, and the
  prefix setter in `test_print_prepend_device_core_risc.cpp`.
- `tests/scripts/t3000/run_t3000_unit_tests.sh` filters on the removed `DPrintMeshFixture`, so T3K runs no DPRINT
  tests, and N300 runs only slow dispatch, so multi-chip fast dispatch is never tested.

## 9. Numbers

| Measure | Today |
|---|---|
| Call site on Blackhole | 4 instructions, one store per argument of up to 4 bytes (two for 8 bytes), and 1 to 3 setup instructions shared by a function's prints; a lone print costs 5 plus stores. Measured as the code-size difference between kernels with K and K+1 copies of a print, K at least 8 |
| Shared device code | 664 bytes on BRISC |
| Message | a 4-byte header and the arguments padded to 4 bytes, plus one 4-byte record per RISC for each kernel that prints |
| Largest print | 1020 bytes of arguments (10-bit payload field) |
| Call sites per ELF | 65,535 (16-bit id; 0xFFFF marks a wrap) |

§3 gives the transport's numbers: the print regions' sizes and the latency.

## 10. Causes of the known issues

In the manual's order:

- **Lone `}`, fill outside ASCII.** `parse_format_token` in `device_print.h` treats a lone `}` as text and reads the
  fill as one byte.
- **Format that isn't a literal.** The checks take a `const char (&)[N]`. A local pointer fails because the macro's
  lambda captures nothing, a global pointer finds no overload, and a `constexpr` array with static storage passes.
- **Spec that doesn't suit its argument.** The device checks only syntax and counts. fmt then throws on the host,
  which stops printing in test mode and ends the process otherwise. A tile's or typed array's data format is known
  only on the host.
- **Print too big for its buffer.** Nothing compares a print's size with its buffer. Above 1020 bytes the header's
  10-bit payload field overflows (`-Werror=overflow`), which also limits Quasar's bigger buffers.
- **Arguments evaluated three times.** The macro pastes the argument list into three calls.
- **Unbraced `if`/`else` and `do`.** The macro is a bare `{ ... }` block, and `DPRINT_DATA0` and `DPRINT_DATA1` are
  `if (noc_index == N) { ... }` even with printing off.
- **`CTSTR` in inline functions and templates.** Its `static` array lands in a COMDAT group, which GCC won't mix
  with the other data in `.device_print_strings`: it reports a section type conflict or silently moves the array to
  `.data`.
- **`volatile` block floats.** `bf16_t`, `bf8_t` and `bf4_t` have no constructor that takes a `volatile` object.
- **`char` prints as a number.** Plain `char` is serialized as `uint8_t`.
- **Enums in functions and anonymous namespaces.** The device names the type with `__PRETTY_FUNCTION__`. The host
  finds no DWARF entry for `kernel_main()::Color`, and the braces in `{anonymous}` end its placeholder early.
- **`#` as an enum's fill.** The host strips every `#` from an enum's spec and reads it as the alternate form.
- **`TileSlice<N>` sizes.** The generic serializer takes a `uint32_t` size and `TileSlice` an `int`, so C++17
  deduction never matches it and only the 64 and 128 specializations work. `tile_slice_from_l1` value-initializes a
  `TileSlice`, which has no default constructor outside `ENV_LLK_INFRA`.
- **Unknown typed-array formats.** The warning formats a `tt::DataFormat`, whose formatter throws
  `std::invalid_argument` for a number with no name, and the server catches only `std::runtime_error`. The host also
  reads a Quasar format number as a Wormhole and Blackhole one.
- **RISCs excluded with `TT_METAL_DPRINT_RISCVS`.** The selection only stops a RISC from announcing its kernel, so the
  RISC still writes and waits, and the host, without its kernel ELF, warns. Firmware prints aren't filtered.
- **`TT_METAL_DPRINT_ONE_FILE_PER_RISC=0`.** `rtoptions.cpp` checks only that the variable is set.
- **Closing a device.** `MetalContext::teardown()` detaches devices without a final read, and `detach_device()` only
  resets the buffers.
- **Quasar per-launch firmware prints.** `dm.cc`, `dispatch_dm.cc` and `trisc.cc` in
  `tt_metal/hw/firmware/src/tt-2xx/` call `DPRINT` on their launch paths.
- **Quasar RISC selection.** `Hal::parse_processor_set_spec` maps the names to fixed Wormhole and Blackhole indices.
  Quasar's Tensix processors 0-7 are DMs and its TRISCs start at 8, and no name adds the dispatch engines.
- **Quasar with the watcher.** The pack TRISC firmware doesn't fit its 5120 bytes with both, so
  `tt_metal/hw/firmware/src/tt-2xx/trisc.cc` stops the build with `#error`.

Users don't see one more: `await()` decides from timing, not from device state.
