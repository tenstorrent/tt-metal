# DPRINT specification

[Device Debug Print](../../docs/source/tt-metalium/tools/device_print.rst), the user manual, is part of this
specification: it defines the API and the output that users rely on, and its Known issues section lists where today's
code falls short of it. This document adds what the manual doesn't cover, so that DPRINT can be rewritten from scratch
and still work as before, only better. [DEVICE_PRINT_implementation.md](DEVICE_PRINT_implementation.md) describes how
today's implementation does it. Anything users can observe that neither the manual nor this document covers stays as
it is today.

## 1. Design

### 1.1 Parts

- **Message creation** checks each print when its kernel compiles, and builds a message from its call site and
  argument values when it runs.
- **Transport** carries messages from the device to the host. The streaming pipeline will replace today's transports.
- **Decoding** finds a message's call site in the ELF that produced it and writes out the text.
- **The library**, built on the other parts, prints complex values such as tiles, typed arrays, call stacks, DEST and
  configuration registers, and the contents of circular buffers and L1.
- **Tests** check what users see, through what users use (§4).

### 1.2 Idea

Format strings never reach the device. When a kernel compiles, each print is checked, and its format string, argument
types, file and line are recorded for the host, never loaded onto the device. At run time the print sends an
identifier for its call site and the raw argument values, and the host looks up the call site and formats the text.

### 1.3 Goals

In priority order:

1. **Small footprint.** Users put hundreds of prints in one kernel, and kernel code space is small. Call sites matter
   most, because prints are many: a call site takes as few instructions as possible, as straight-line code with no
   strings on the device, and everything that isn't specific to one call site is in the library.
2. **Each argument evaluated once.** A print is one statement. Each argument is evaluated exactly once, in the
   caller's scope, before the print takes the print buffer, and not at all when printing is off.
3. **Compile-time checks.** Each print is validated when its kernel compiles: its format string, its argument types,
   and whether the placeholders match the arguments. The rules are fmt's, plus DPRINT's own limits, such as which
   types it can print and how big a print can be. The limits are public: the manual lists them, and a print beyond a
   limit is a compile error where the compiler can tell. Mistakes are compile errors
   at the print, with the messages in the manual's Compile-time checks table. Valid prints compile without warnings on
   every RISC type.
4. **No cost when off.** With printing off, prints produce no code, so they add nothing to the kernel's size or run
   time. Turning printing off causes no warnings either: a variable or parameter that only prints use needs no
   `[[maybe_unused]]`. A print that is off can still name its arguments, as long as it produces no code.
5. **Little data.** A message carries its call site, each argument once at its natural size, and just enough to tell
   which RISC sent it and where the next message starts. What stays the same between messages, such as the running
   kernel, isn't repeated in each one.
6. **Reliable delivery.** No message is lost: each one arrives once, intact and in its RISC's program order, even
   when every RISC prints as fast as it can.
7. **A robust host.** Nothing the device sends stops the host. A message it can't decode becomes a one-line
   diagnostic, and decoding goes on.

## 2. Requirements

### 2.1 Message creation

- Every RISC that can execute user code prints.
- Firmware prints like kernels do, from boot on, once it knows which RISC it runs on. It doesn't print on every kernel
  launch.
- A print works as another macro's argument (for example `MATH(...)`, which deletes it on the other TRISCs), and
  through wrapper macros that forward the whole call, format included, with or without arguments.
- Arguments are evaluated in the caller's scope, so a print can name members through `this` and lambda captures.
- DPRINT doesn't steer how the compiler inlines code: no assembly or stand-in functions that change the compiler's
  estimates, and no guesses of the optimization level. Ordinary attributes on DPRINT's own functions are fine where
  they make DPRINT's code smaller.
- Any part of the selection that is compiled into a kernel is part of its build hash.

### 2.2 Transport

- The transport carries each message to the host together with the device, core and RISC that printed it and the
  program, a kernel or the firmware, that was running there.
- It keeps what it needs on the device in the memory assigned to it, and where that memory is and how big it is are
  decided outside DPRINT. That memory always fits the largest print that the manual promises.
- Messages reach the host soon after they're printed, even if the kernel never finishes. Hang handlers reset the
  device, so a message left on it is lost.
- Prints are collected in slow and fast dispatch, during firmware boot before any kernel runs, from remote chips, and
  from dispatch cores whose own kernels print.
- The host has a call that returns as soon as everything printed so far is written out, and never before.
- Closing a device first writes out its complete lines, and a print can't make closing hang.

### 2.3 Decoding

- Decoding uses only what the transport delivers. It finds the call site in the ELF of the program that printed the
  message and formats the text.
- A complete line is written out as soon as it's decoded. The process can die, so text held in host memory is lost.

### 2.4 Library

- The library keeps its headers and the API that the manual documents, including the per-RISC `TileSlice` parameter
  lists, because the same `TileSlice` text compiles on every RISC. The register, circular-buffer and L1 dumps in
  `dprint_tensix*.h`, `dprint_pages.h` and `dump.h` keep their API too. Adding to the library is fine.
- The library's helpers that print keep one copy per kernel, not one in each caller.
- With printing off, code that uses the library compiles unchanged, and the library adds no code.

## 3. Constraints

Each of these broke DPRINT before:

- `__FILE__` and `__LINE__` don't identify a call site: one macro can expand to two prints, and one line can
  instantiate a template with several argument types.
- GCC can't bind a reference to a bitfield or a packed field, so arguments are copied.
- User code defines global names such as `dprint`, `print_tile_rows`, `DPRINT_DISPATCH`, `DPRINT_COMBINE` and
  `DEBUG_PRINT`, so DPRINT doesn't define them, nor any generic global type name: a model's `union F32` clashed with
  DPRINT's legacy `F32`, and data-movement kernels already see a global `Flags` template.

## 4. Tests

### 4.1 Rules

- Tests check what users see: the printed text, build errors, and whether programs finish. They use what users use,
  such as kernels with prints, the public host API and the documented settings, so they pass unchanged on any
  implementation.
- A reported bug first gets a test: the shortest repro that uses only what users use.
- Smaller tests that look inside DPRINT, at its device memory, the ELF or the host's state, only help find which part
  fails. They aren't requirements, and they change with the implementation.
- Tests run fast. Many cases share one kernel, and a test waits for its program or its text, never for a fixed time.
- A test compares all the text it expects, exactly, so a lost, repeated or garbled message fails it. Only what changes
  between runs or builds, such as an address, is left out. When RISCs print at the same time, each RISC's text is
  compared on its own.
- A test that expects no text also checks text printed elsewhere, so a host that prints nothing fails it.
- Tests pass every time, and stress tests keep passing when run many times in a row.

### 4.2 Coverage

Tests cover every example and rule in the manual and every requirement and constraint in this document. They run on
every RISC type that prints, in each dispatch mode that can run its kernels; what doesn't depend on the RISC, such as
how the host formats a value, is tested on one RISC. In particular, tests show that:

- no message is lost, repeated or reordered within its RISC when all the RISCs of a core and all cores print at once;
- the same holds when they print far more than DPRINT's device memory holds, in long streams of lines and in large
  dumps such as tiles, and on remote chips;
- kernels with different prints that run back to back on one RISC each print their own text;
- a line printed before a kernel hangs still appears;
- unselected cores, chips and RISCs don't hang, even when they print more than DPRINT's device memory holds.

Call-site size, the data a message carries, warning-free builds with printing on and off, and a host that survives
anything the device sends are checked during development instead.

## 5. Comparing implementations

A change to DPRINT is compared with the implementation it replaces, on every RISC type at its default optimization
level:

- Code size counts DPRINT's own footprint in a kernel: the library, the code of every print where it is written, and
  the data that DPRINT's code adds. How the compiler inlines the user's own functions around their prints is the
  user's code and doesn't count; the manual's `DPRINT_FORCE_NOINLINE` keeps one copy of a function that prints.
- For any mix of prints, the footprint is no larger than before. The library may grow when the total shrinks.
- A kernel can hold many prints of one kind, so no kind of print costs more code per print than before.
