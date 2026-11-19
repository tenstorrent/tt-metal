# DPRINT implementation

This is today's implementation of the [DPRINT specification](DEVICE_PRINT_spec.md).
[Device Debug Print](../../docs/source/tt-metalium/tools/device_print.rst) documents how DPRINT is used. The streaming
pipeline will replace the transport (§3).

## 1. Code

| Part | Where |
|---|---|
| Message creation (§2): macros, compile-time checks, call-site records, message layout, the shared device code (`device_print_message()`, `print_end()`, `device_print_store()`, `device_print_typed_array()`, `serialize_typed_array()`), `DEVICE_PRINT_FORCE_NOINLINE` | `tt_metal/hw/inc/api/debug/device_print.h`; `dprint.h` (the `DPRINT` names and the legacy-syntax error); `tt_metal/hw/inc/hostdev/device_print_structures.h` (message header, string-info entries) |
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

- `DEVICE_PRINT` is one statement, `do { } while (0)`, so it works in unbraced `if`/`else` branches, loops and `case`
  labels. It calls a generic lambda once, with the argument list as the lambda's arguments. The lambda takes them by
  value, so each argument is evaluated exactly once, in the caller's scope (`this->member` and lambda captures work),
  before the print takes the lock, and bitfields, packed fields and `volatile` objects are copied. The lambda runs
  the compile-time checks and calls `send_message()`, which registers the call site and hands the payload to the
  shared device code; both are `always_inline`. A `break` or `continue` in a GNU statement expression in an argument
  binds to the macro's `do { } while (0)`.
- With printing off, `DEVICE_PRINT` expands to `do { } while (0)`, so a variable or parameter that only prints use is
  left unused. The JIT build doesn't warn about unused variables or parameters and only warns about set-but-unused ones
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
  8-byte aligned. So an 8-byte value is stored into the print buffer as two 4-byte halves. That costs less memory than
  8-byte headers with every message aligned to 8 bytes, because 8-byte values are rare. Into a 16-byte-aligned image
  on the caller's stack (below), where the 8-byte values are 8-byte aligned, Quasar stores a value computed at run time
  with one store and a constant as two halves (GCC would load an 8-byte constant from `.srodata`).
- A print names its RISC with `PROCESSOR_INDEX`. The HAL defines it everywhere except on Quasar's DMs, TRISCs and
  dispatch engines, where `device_print.h` falls back to `internal_::get_hw_thread_idx()`, a thread-local index that
  firmware sets up during boot, so firmware prints only after that.
- The shared device code (the library) is in every printing ELF: `device_print_message()` takes the lock, waits for
  space, writes the header and copies the payload after it, and returns the address after the message; `print_end()`
  moves the write position there and releases the lock. A call site calls both. The payload's size in words is in the
  low 4 bits of the call site's `DevicePrintStringInfo` entry address: the entries are `aligned(16)` and a multiple of
  16 bytes, the section starts at 0x6500000 with `__device_print_strings_info_start` first, and the info id is the
  entry's distance from that symbol divided by the entry size. `send_message()` picks one of these forms from the
  argument types:
  - no payload: `print_message(entry)`;
  - words: scalar prints of up to 14 words whose values share no word, `print_message(entry + n, word 0, ...)`, the
    first 7 words in argument registers and the others in the caller's outgoing-argument area. `print_message` is a
    variadic alias of `device_print_message`, whose named parameters take all eight argument registers. On Quasar's
    data-movement RISCs a print with 8-byte values takes this form only up to 7 words. Unoptimized builds use the
    store form instead;
  - store: the other prints of up to 64 words. Each scalar and `TileSlice` is stored into a 16-byte-aligned image on
    the caller's stack, and `device_print_store(entry + high bits, image + low bits)` hands it to
    `device_print_message()` (the image's size, 1 to 64 words, is split over the free low bits of the two addresses).
    The typed arrays of such a print are copied after the call, into the reserved message;
  - typed array: a print whose first typed array (in payload order) has at most 15 words before it and at most 8
    words of other values (on Quasar's data-movement RISCs at most 2 next to an 8-byte value) calls
    `device_print_typed_array(entry + words before, first word | payload words << 24, elements)`, which reserves the
    message and copies that array under the lock, after the lock's L1 invalidation. The call site then writes the
    other arguments into the message, relative to the returned address;
  - lone `TileSlice`: the print's copy of the `TileSlice` is the image, `print_message(entry + 15, words, copy)`;
  - reserve: prints of more than 64 words, which only typed arrays and `TileSlice`s make: `device_print_message()`
    reserves the message (it copies the start of the print buffer into it, which is harmless) and the call site writes
    the payload into it.
  A typed array after the first is copied by `serialize_typed_array()`, or inline when it has 1 or 2 elements; a
  `dp_typed_array_t<0>` is one word, a scalar. The typed-array and reserve forms keep the print's other values across
  the library call, and the store form keeps its typed arrays' addresses and the message address across
  `device_print_store()` and `serialize_typed_array()`. The library functions are `static` (marked `unused`, as
  tt-llk's build errors on unused functions), `noinline` and `noclone`: a non-LTO build (the RVV recipe, tt-llk) gets
  only the ones its prints call, and GCC doesn't specialize them for one call site. Unoptimized builds compile them
  with `optimize("Os")`; optimized builds compile `device_print_message()` with `optimize("no-split-loops")`.
- A print takes at most 64 arguments (`print_argument_count<Count>`), and its scalar arguments (all but typed arrays
  and `TileSlice`s) take at most 256 bytes (`print_argument_bytes<Bytes>`); a message must fit its RISC's print
  buffer. Each is a `static_assert` whose message contains the phrase in the manual's table of compile-time checks.
- `DEVICE_PRINT_FORCE_NOINLINE` (`DPRINT_FORCE_NOINLINE` in `dprint.h`) is `__attribute__((noinline, noclone))` with
  printing on (`noinline` under clang, which has no `noclone`) and empty with it off: one copy of the function, neither
  copied into each caller nor cloned per constant argument. DPRINT's own helpers that print use it: `dprint_pages.h`'s
  `print_cb_details()`, `dprint_tensix.h`'s `dprint_data_format()`, the nine top-level dumpers of `dprint_tensix_pack.h`
  and `dprint_tensix_unpack.h` (`dprint_tensix_alu_config()`, `dprint_tensix_pack_config()` and the like) and
  `dprint_tensix_unpack_tile_descriptor_helper()`, which its dumper calls from two paths (one register, both registers)
  and which GCC otherwise copied into both (`dprint_tensix.h` defines the macro empty when tt-llk's harness includes it
  without `device_print.h`, with printing off). `dprint_pages.h`'s tile printers `print_tile_rows()` and
  `print_full_tile()` and `dump.h`'s out-of-line functions (`debug_dump_cb()`, `debug_dump_l1()`,
  `debug_dump_cb_typed_words()` and `debug_dump_cb_typed_rt()`; `debug_dump_cb_typed()` is an inline wrapper, see §5)
  use it too, so that GCC doesn't clone them per constant CB; `dprint_tile.h`'s two samplers are `static`, `noinline`
  and `noclone` the same way (they are parsed before `device_print.h` defines the macro). On the RISCs without circular
  buffers (TRISC1, ERISC, IERISC; Quasar's data-movement RISCs for `debug_dump_cb_typed()`) these functions only print
  that they are unavailable and are `noinline` only, so that GCC drops their unused arguments (with `noclone` each call
  loaded them: 8 to 12 bytes more per call). `checkpoint.h`'s barrier, `debug_checkpoint_barrier()`, and CB dump,
  `debug_checkpoint_dump_cbs_impl<>()` (which prints the checkpoint's header, the name passed as a `CTSTR`), use it (one
  barrier per kernel and one dump per template arguments, not one per checkpoint; with printing off the barrier is all a
  checkpoint does and stays `inline`, so those kernels are unchanged). The data-movement page printers
  (`print_u32_pages()` and the like) stay `inline`: marked, five of them each called once with constant arguments were
  12 bytes larger on Blackhole BRISC and NCRISC, and unmarked they are smaller than before from 1 to 8 call sites.

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
- `device_print_message()` takes the lock, waits for space, and writes the header and the payload. A message that
  doesn't fit before the end of the data goes at its start, after a header-only record with info id 0xFFFF where that
  fits. `print_end()` moves the write position past the message and releases the lock.
- A RISC's first print after a kernel starts is preceded by a header-only record with payload 1023 and the kernel id
  as its `info_id`, which tells the host which kernel's ELF decodes that RISC's messages; the wait for space writes and
  publishes it before it waits for the space of the message.
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
  waits for space or for the region's lock runs the aggregator itself (`device_print_dispatcher_execute_hook()`).
  dispatch_d shares the region and the lock with dispatch_s and has no hook, so a print on dispatch_s that waited for
  the lock without draining would deadlock with a dispatch_d that holds it while it waits for space.
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
- Per RISC, the server assembles lines, the kernel's and the firmware's apart, holds `\r` blocks until the next `\n`,
  adds the prefix (and `[FW] ` ahead of it on a firmware line with `TT_METAL_DPRINT_TAG_FIRMWARE=1`, which
  `DevicePrintFixture` turns on so that tests check kernel lines only), and writes to stdout, a file or one file per
  RISC.

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
  for a tile it can't read, it sends an error code instead, and the host prints a warning. For a circular buffer known
  at compile time, the inline constructor looks up the buffer's descriptors (data format, tile and face dimensions,
  tile size) where they fold into constants and passes them, packed into two words with the tile's byte offset, to one
  out-of-line sampler, `dprint_tile_slice_sample_words()`, and no descriptor table is loaded. Any other buffer is
  looked up out of line (`dprint_tile_slice_sample_cb()`), from the tables. Both are `noinline, noclone` (clang:
  `noinline`): one copy per kernel, which GCC still specializes in place when every call passes the same descriptors,
  but never one copy per CB or per call site (without `noclone` GCC cloned the sampler per constant: +192 to +848
  bytes with a run-time-CB `TSLICE` in a loop, or a constant-CB and a run-time-CB `TSLICE` in one kernel).
  `dump.h`'s `debug_dump_cb_typed()` does the same one level up: an inline wrapper looks a constant CB's descriptors
  up at the call site and passes them to one out-of-line dump, `debug_dump_cb_typed_words()`, whose rows are
  `TileSlice`s built from those words (a `TileSlice` constructor that takes the sampler's arguments); a run-time CB is
  looked up once, in `debug_dump_cb_typed_rt()`.
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
- tt-llk's `dprint_tensix.h` includes tt-metal's even with printing off, without `device_print.h`; tt-metal's
  `dprint_tensix.h` then defines `DEVICE_PRINT_FORCE_NOINLINE` empty under `ENV_LLK_INFRA`.

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
  test compares frame names and the `...` marker, and the checks in `debug_tools_test_utils.hpp` skip firmware lines,
  which `DevicePrintFixture` has the server tag with `[FW] `.
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
| Call site | One more copy of a print in `kernel_main` (the code-size difference between 4 and 12 copies, divided by 8), on Blackhole BRISC: 12 bytes without arguments (the previous macro 16); with values read from L1, the loads included, 16 bytes for 1 word (24), 24 for 3 (40), 48 for 8 (120), 144 for 16 (218.5), 208 for 32 (416); 40 for 3 one-byte values (40); 80 for `dp_top_callstack_t::current(0)` (308); 20 for a typed array alone, of any length from 1 (28 to 68), 16 for `dp_typed_array_t<0>` (20), 32 for two one-element typed arrays (40), 156 for a typed array next to eight 8-byte values (202); 120 to 128 for a lone `TSLICE` on Blackhole and Wormhole TRISC0 (204 to 272) |
| Shared device code | On Blackhole BRISC `device_print_message()` 388 to 500 bytes when a kernel's prints use one of its two paths (words, or an image) and 548 to 560 with both, `print_end()` 40, `device_print_store()` 16 to 24, `device_print_typed_array()` 64 to 128 (more for more shapes of typed-array prints), `serialize_typed_array()` 28 to 36; GCC drops what a kernel's prints don't use. The previous macro's was 664 bytes, plus an out-of-line serializer per argument-type list for wider prints. A constant-CB `TileSlice` adds one sampler (164 bytes on Wormhole TRISC0 for an unconfigured CB) |
| Message | a 4-byte header and the arguments padded to 4 bytes, plus one 4-byte record per RISC for each kernel that prints |
| Largest print | 64 arguments and 256 bytes of scalar values (static assertions), 1020 bytes of arguments in all (10-bit payload field), within the RISC's print buffer |
| Call sites per ELF | 65,535 (16-bit id; 0xFFFF marks a wrap) |

§3 gives the transport's numbers: the print regions' sizes and the latency.

## 9a. Code size and stack

Measured with `replay.py` (the JIT's exact compile and link commands) and `footprint.py` (every code byte of an ELF
split into DPRINT's library, DPRINT's call sites and user code), at each RISC's default optimization level, against
the previous macro.

- DPRINT's footprint: generated kernels of 104 to 150 varied prints (seeds 1, 2, 3, 71 and 83 on every target, 45
  prints on Wormhole's active ERISC; 185 kernel x target pairs): -32.5% (3,254,620 to 2,195,340 bytes; library -42.7%,
  call sites -31.9%, other loaded bytes -22.4%), smaller in every pair (-22.7% to -51.4%), other loaded bytes never
  larger; the kernels' code -24.9%, never larger. Per print (the code-size difference between 4 and 12 copies of one
  print, divided by 8), 20 shapes on Blackhole BRISC, Wormhole TRISC0 and Quasar DM (60 shape x target pairs) cost no
  more than before; the shapes that cost more are below.
- Typed arrays: a zero-length array is one word; arrays after the first with 1 or 2 elements are copied inline; a
  print whose other values take more than 8 words (on Quasar's data-movement RISCs, more than 2 words next to an
  8-byte value) stores them into an image instead of keeping them in registers across the library call. Typed array
  plus 1 to 14 words of 4-byte or 8-byte values, in `kernel_main` and in functions that are not inlined, on Blackhole
  BRISC, DRISC and ERISC0 and Wormhole TRISC0: never more per print than before between 4 and 12 copies; between 12
  and 24 copies two `u64` and a `<4>` cost more (Blackhole BRISC 77.3 to 87.3 bytes, Wormhole TRISC0 84 to 87.3).
  Costs more per print (final check 7b's sweep, 77 shapes x 8 targets between 4 and 12 copies and 12 shapes between
  12 and 24; `device_print.h` is unchanged since):
  - On Quasar DM a typed array next to more than 2 words of other values that include an 8-byte value: 6 to 108 bytes
    (a `u64`, a `u32` and a `<4>` 78 to 84 bytes; two `u64` and a `<4>` 78 to 96; eight `u64` and a `<4>` 148 to 216;
    thirteen `u64` and a `<4>` 208 to 316; in a function that is not inlined, twelve `u64` and a `<4>` 236 to 304; a
    `<2>` before three `u64` and two `u32` 114.5 to 120): such a print takes the store form, and its arrays are copied
    after `device_print_store()`, those of 3 or more elements by one more call per print, where the previous macro
    called one serializer shared by all prints with the same argument types. With fourteen or fifteen `u64` it is
    cheaper, and so is one `<1>` after 8 doubles (150 to 100).
  - Three or four typed arrays in one print: 3 arrays +16 bytes on Blackhole and Wormhole TRISC0 (52 to 68) and +17 on
    Blackhole BRISC between 12 and 24 copies (67.3 to 84.3); 4 arrays +28 on TRISC0 (60 to 88), +20.7 on Blackhole
    BRISC and +27.3 on Quasar DM between 12 and 24 copies, up to +35.5 on Quasar TRISC0 (4 x `<8>` and 4 x `<16>`, 72.5
    to 108), also with a different array address in each print. Each array after the first with 3 or more elements
    costs its own `serialize_typed_array()` call (12 bytes per array in the previous macro, 20 here). Two arrays cost
    0.7 to 2 bytes more between 12 and 24 copies, and with values between them +4 to +8 on Quasar DM and +1.5 to +4 on
    Blackhole BRISC under register pressure.
  A fix for both copies all of a print's typed arrays in one library call (the addresses and first words in argument
  registers, or in the store image); not done. No kernel in tt-metal, tt-llk or tt-blaze prints more than one typed
  array in a print or a typed array next to other values.
- `TileSlice`: 120 TileSlice kernels (1 to 8 prints on 1 to 3 constant CBs and run-time CBs) on the generic, `-fmt`
  (real CB descriptors), `-fmtx` (eight formats), RVV (no LTO) and data-movement templates: 994 kernel x target
  pairs, DPRINT code and data -39.0% (1,661,852 to 1,014,092 bytes), none larger; the manual's single-`TSLICE`
  example on Blackhole TRISC0 1008 to 752 bytes. Final check 7b's kernels with a run-time-CB `TSLICE` in a loop, or a
  constant-CB and a run-time-CB `TSLICE` in one kernel (70 kernel x target pairs): none larger, where GCC's two
  clones of the sampler made them 192 to 848 bytes larger before the sampler was `noclone` (a loop of 4 over a
  run-time CB on Blackhole TRISC0: 2132 bytes before, 2804 with the clones, 1940 now). The bytes of every message are
  unchanged (`tools/tsemu.py`, 7 targets; `dump.h`'s dumps too, except a slice's data bytes after its last value, all
  64 when its format is unsupported, which no version writes: they hold whatever the stack held). Per print (final
  check 7b's 28 shapes x 13 templates, between 4 and 12 copies), a `TSLICE` next to other values on a CB whose data
  format the build doesn't configure (the generic and RVV templates) costs more than before in 39 of 143 pairs: up to 84
  bytes with two `TSLICE`s (CBs 0 and 16 and a `u32`: 192 to 276 on the RVV recipe, 232 to 315 on Wormhole TRISC0 and
  TRISC2), up to 74.5 with one (CBs cycling 0, 1, 16 next to two `u64`: 109.5 to 184 on the RVV recipe); 12 copies of
  such prints are up to 516 bytes larger in all (9 pairs). The previous macro's per-CB constructor copy returned at once
  for a CB without a format, while here each `TSLICE` passes the CB's descriptors to the one sampler (20 bytes more per
  `TSLICE` than when GCC cloned the sampler with the descriptors as constants) and the store form copies the 80-byte
  `TileSlice` twice (the argument and the image). With real CB descriptors (`-fmt`, `-fmtx`, RVV `-fmt`) 166 of 167
  pairs cost 5.5 to 381 bytes less per print; CBs cycling next to two `u64` on the RVV `-fmt` recipe cost 2 bytes more
  (182 to 184).
- Few-print kernels that mix a typed-array print with prints of other forms: 48 of 796 kernel x target pairs (61
  kernels of 2 to 5 prints, 14 targets) are 4 to 36 bytes larger, because `device_print_typed_array()` and the second
  path of `device_print_message()` cost more than the few prints save. No kernel in the repository has this mix.
- DPRINT's own helpers (all at the default level; DPRINT code and data):
  - `print_cb_details()` with 1 to 8 constant or run-time CB ids: 616 to 624 bytes on Blackhole and Wormhole BRISC,
    NCRISC and TRISC0, where the previous macro needed 756 to 1392 (without the macro GCC copied it into every caller,
    up to 968 bytes).
  - `print_full_tile()` and `print_tile_rows()` from 1 to 8 call sites with run-time CBs, one constant CB or 8
    distinct constant CBs, alone or next to other prints, on Blackhole TRISC0/1/2, Wormhole TRISC0/1/2 and the RVV
    recipe: 154 kernel x target pairs, none larger, -33.4%, and 90 more pairs on the `-fmt` templates and with final
    check 7's kernels, none larger, -36.2% (without the attributes, GCC copied `print_full_tile()`
    into every caller: 2464 to 3716 bytes on Blackhole TRISC0 with 8 call sites; `noinline` alone still cloned it
    per constant CB, +460 to +712 bytes with 8 CBs).
  - The `dprint_tensix*.h` dumpers and `dprint_data_format()` (the repository's `writer_config_reg.cpp` for each
    register, each dumper called 1 to 5 times with constant and run-time register ids, `dprint_data_format()` from 1,
    2, 4 and 8 call sites, in a loop and with 1 to 8 constant formats): 774 kernel x target pairs on Blackhole and
    Wormhole TRISC0/1/2, none larger at the default level (`writer_config_reg.cpp` with `ALU_CONFIG` on Blackhole
    TRISC1: 1596 bytes before, 1948 without the macro, 1324 with it; two calls of `dprint_tensix_alu_config()`: 1644,
    3588, 1324; two calls of `dprint_tensix_unpack_tile_descriptor()`: 1916 before, 2048 while only the dumper was out
    of line, 1500 with its helper out of line too, on Blackhole TRISC0, and 1948, 2052, 1504 on Wormhole TRISC0). At
    `-O2`, the level `KernelSpec` builds compute kernels at (the repository's `ConfigReg*` tests), the same pairs are no
    larger except `dprint_data_format()` with constant formats: two +92 to +100 bytes, eight +4 to +8 on Wormhole (a
    124-byte jump table that `-O3` avoids by cloning the function per constant).
  - `dump.h` and `checkpoint.h` (each helper from 1, 2, 3, 4 and 8 call sites, with constant, repeated and run-time
    arguments, on Blackhole and Wormhole BRISC, NCRISC, TRISC0 and TRISC2 and Quasar DM, 592 kernel x target pairs;
    `DEBUG_CHECKPOINT_EX()` with the destination dump also on TRISC1): `debug_dump_cb()` and `debug_dump_l1()` are never
    larger. `debug_dump_cb_typed()` with CBs known at compile time is far smaller (8 distinct CBs on Blackhole TRISC0:
    1916 bytes before, 2828 when GCC cloned the function per CB, 848 now; the repository's `eltwise_copy_dump_typed.cpp`
    1968 to 904). On the RISCs without CBs the dump functions only print that they are unavailable; with `noclone` each
    call loaded their unused arguments (8 to 12 bytes more per call, larger than before from 20 calls on TRISC1), so
    they are `noinline` only: never larger (Blackhole and Wormhole TRISC1 and active ERISC, Blackhole IERISC, Quasar DM
    and TRISC1, 4 to 64 calls). Checkpoints (`DEBUG_CHECKPOINT()`, and `DEBUG_CHECKPOINT_EX()` with the same and with
    different arguments, at 1 to 64 sites on Blackhole and Wormhole BRISC, NCRISC, TRISC0 and TRISC2, 192 kernel x
    target pairs; checkpoints don't build on Quasar): every kernel is smaller than before, e.g. 8 checkpoints on
    Blackhole BRISC 4244 bytes before, 1300 now, and 64 checkpoints 10936 before, 2644 now. The header with the
    checkpoint's name is printed inside the dump, the name passed as a `CTSTR`, so a checkpoint's own code is the two
    barrier calls and the dump call (DPRINT code of 64 checkpoints on Blackhole BRISC: 960 bytes before, 576 now; with
    the destination dump on Blackhole TRISC1 the kernel is 16 bytes larger than with the header in the macro, 2528
    against 3556 before). Keeping only the dump out of
    line was not enough: GCC then copied the barrier into each site, and from about 40 sites the kernel was up to 3.2 KB
    larger than before, the size it has with printing off. With printing off every one of these kernels is
    byte-identical to before (192 pairs). The repository's 12 dump and checkpoint test kernels (114 kernel x target
    pairs, built with `DEBUG_CHECKPOINT_ENABLED`) are all smaller. Larger:
    - `debug_dump_cb_typed()` with a CB known at compile time: each call passes the CB's descriptors to
      `debug_dump_cb_typed_words()` (6 arguments), 16 to 22 bytes more per call than the previous macro's call of its
      per-CB clone (1 argument). `footprint.py` doesn't count a helper call's arguments, and its totals are smaller in
      every pair; in the kernel the dump's savings outweigh the calls up to about 55 calls on one CB (64 calls: +60
      bytes on Blackhole BRISC, +192 on Wormhole NCRISC, equal on Blackhole TRISC0).
    - `debug_dump_cb_typed()` with a CB known only at run time: 44 to 72 bytes more DPRINT code and data whatever the
      number of calls (Blackhole TRISC0 1916 to 1988, BRISC 1852 to 1896), and each call 8 bytes more in the kernel
      (its constant tile and untilize arguments are passed); with 8 calls the kernel is up to 84 bytes larger, and up
      to 140 when 4 of the 8 calls have a constant CB. The run-time-CB sampler path costs more than the previous
      per-kernel constructor copy (the same cause as the `TSLICE` exception).
  - Stack: a helper kept out of line where GCC inlined it before adds its frame to the call chain. Over 1340 kernel x
    target pairs of these helpers and of the `TSLICE` loops and mixes, 34 chains are deeper than before: by 16 to 64
    bytes (the register dumpers, the tile-descriptor helper, the checkpoint dump) and by 64 to 80 bytes for
    `DEBUG_CHECKPOINT_EX()` with the destination dump on Wormhole TRISC1 (224 to 288); 916 are shallower; the deepest
    chain is 480 bytes (DEST dumps on Wormhole TRISC1, 432 before), far below every TRISC's budget.
- The compiler decides from a function's size and stack frame whether to copy it into its callers, and a print is
  smaller than before, so a function that prints can now be copied into its callers where one copy was kept before. Each
  copy is smaller, but a kernel grows by the copies: a repository test kernel, `dfb_multi_touch_consumer.cpp` with 16
  DFBs, goes from 768 to 2188 bytes of DPRINT code on Blackhole BRISC (1636 to 2924 bytes of code in all; its `static
  inline` helper with 9 prints is copied 16 times). With the helper marked `DPRINT_FORCE_NOINLINE` it takes 744 bytes of
  DPRINT code and 1456 in all, less than with the previous macro (Blackhole and Wormhole BRISC, Blackhole NCRISC). Over
  the gate's corpora at the default levels, the DPRINT test kernels (596 builds; the test kernel that uses
  `DEVICE_PRINT_FORCE_NOINLINE` doesn't build with the previous macro), the repository's kernels with prints (425) and
  the build-environment variants (914) are never larger; 6 of 621 compute-kernel builds are (the TTNN
  `chunk_gdn_prep.cpp` on Blackhole TRISC0 by 4256 bytes: LLK functions with tt-llk's tensor-shape coverage print are
  copied into their callers), 3 of 1472 real kernels with inserted prints, 78 of 2986 generated ones and 2193 of 11,261
  of the review's inlining counterexamples (gate run of this patch; 336 of these are user functions with a `TSLICE`
  print that the compiler copies into their callers, where the previous macro kept one copy: each copy's `TSLICE` is
  cheaper than the previous macro's, but there are more copies; 16 of them were smaller while GCC cloned the sampler.
  None calls another DPRINT helper).
- Stack: a print keeps its values in the function that contains it until the print has copied them (the manual gives
  this as a caution under Limits). Six nested functions that are not inlined, each printing N values of 4 bytes read
  from memory and then calling the next, need on a Wormhole TRISC0 (912 bytes of stack between the entry `sp` and
  `__stack_base`) 208 / 304 / 496 / 880 / 1360 / 1744 / 2128 / 2896 bytes for N = 8 / 12 / 16 / 24 / 32 / 40 / 48 / 64,
  where the previous macro needed 144 / 144 / 144 / 336 / 624 / 816 / 1008 / 4368: at `-O3` such a function also saves
  all callee-saved registers. Wormhole TRISC2, Blackhole TRISC0 and Quasar TRISC2 match within 16 bytes. On the
  data-movement and Ethernet RISCs the same chains are shallower than before (Blackhole BRISC 1360 against 2064 for N =
  32, Quasar DM 1776 against 3072, Wormhole active ERISC 1424 against 2096). A recursive function that prints fits fewer
  levels.
- On Blackhole's active ERISC0 the JIT passes `-Werror=stack-usage=1912` (`bh_hal.cpp`), so a function whose frame
  grows past 1912 bytes by a print fails to compile where the previous macro's build compiled.
- tt-llk's test harness gives each TRISC a fixed stack region at the top of its local memory, above its local data
  (256 bytes on Wormhole, 512 on Blackhole); a wide print's frame can grow out of it into the local data, which tests
  don't use today. tt-llk's own builds (`ENV_LLK_INFRA`, no LTO, no `--gc-sections`) get only the library functions
  their prints call, as the library functions are `static`.

## 10. Causes of the known issues

In the manual's order:

- **Lone `}`, fill outside ASCII.** `parse_format_token` in `device_print.h` treats a lone `}` as text and reads the
  fill as one byte.
- **Format that isn't a literal.** The checks take a `const char (&)[N]`. A local pointer fails because the macro's
  lambda captures nothing, a global pointer finds no overload, and a `constexpr` array with static storage passes.
- **Spec that doesn't suit its argument.** The device checks only syntax and counts. fmt then throws on the host,
  which stops printing in test mode and ends the process otherwise. A tile's or typed array's data format is known
  only on the host.
- **Prints of more than 1020 bytes.** The header's 10-bit payload field overflows (`-Werror=overflow`), which fails
  the build with an unrelated message and also limits Quasar's bigger buffers.
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
