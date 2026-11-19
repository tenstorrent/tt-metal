Device Debug Print
==================

.. note::
   Tools are only fully supported on source builds.

Overview
--------

Device debug print (DPRINT) lets kernels and firmware print formatted text from the device. The host writes the
text to the terminal or to a log file. Format strings use the syntax of fmtlib and C++20 ``std::format``, and the
compiler checks every print against its arguments.

.. code-block:: c++

    #include "api/debug/dprint.h"

    void kernel_main() {
        uint32_t n = get_arg_val<uint32_t>(0);
        DPRINT("n = {0}, hex = {0:#x}\n", n);  // prints: n = 42, hex = 0x2a
    }

How it works: the compiler stores each print's format string in the kernel binary, in a section that is never
loaded onto the device. When the print runs, the device sends the host only an identifier for the print and the
values of its arguments, and the host looks up the format string and writes the text. This has a few consequences:

- A print is cheap: a few instructions in the kernel, no device memory for its text, and only the argument values
  sent to the host.
- The host can't read device memory, so a runtime ``const char*`` prints as an address. Use ``CTSTR("...")`` for
  fixed text (see `Strings`_).
- Text appears while the kernel is still running, so DPRINT helps with debugging hangs.

Enabling
--------

DPRINT is off by default, and every print compiles to nothing. Selecting cores to print from turns it on for all
kernels and firmware built in that run. Core coordinates are logical, so worker cores and Ethernet cores both
start at (0,0).

.. code-block::

    export TT_METAL_DPRINT_CORES=0,0                    # Tensix cores: x,y OR (x1,y1),(x2,y2),(x3,y3) OR (x1,y1)-(x2,y2) OR all OR worker OR dispatch
    export TT_METAL_DPRINT_ETH_CORES=0,0                # Ethernet cores, same syntax
    export TT_METAL_DPRINT_DRAM_CORES=all               # Blackhole DRAM cores, same syntax
    export TT_METAL_DPRINT_DISPATCH_CORES=all           # Quasar dispatch-engine cores, same syntax
    export TT_METAL_DPRINT_CHIPS=0                      # optional: comma separated chip ids OR all (default)
    export TT_METAL_DPRINT_NODES="(M0,D0),(M0,D1)"      # optional: FabricNodeId list (Mn = mesh id, Dn = device id) OR all (default)
    export TT_METAL_DPRINT_MESH_COORDS="(0,0),(1,3)"    # optional: (row,col) in the global system mesh OR all (default)
    export TT_METAL_DPRINT_RISCVS=BR+TR0                # optional: any of BR,NC,TR0,TR1,TR2,TR*,ER0,ER1,ER*,DR; default all
    export TT_METAL_DPRINT_FILE=log.txt                 # optional: write to this file instead of stdout
    export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=0   # optional: on by default; starts each line with <device>:<x>-<y>:<RISC>:
    export TT_METAL_DPRINT_ONE_FILE_PER_RISC=1          # optional: one file per RISC (see Output); ignores TT_METAL_DPRINT_FILE, turns the prefix off
    export TT_METAL_DPRINT_TAG_FIRMWARE=1               # optional: off by default; starts each line that firmware printed with [FW]

- Each core list selects cores of its own type, and setting any of them turns DPRINT on. ``all`` selects every
  core of that type, ``worker`` every core that doesn't run fast-dispatch kernels, and ``dispatch`` the cores that
  do. When dispatch cores are selected, the dispatch kernels' own prints appear too.
- ``TT_METAL_DPRINT_CHIPS``, ``TT_METAL_DPRINT_NODES`` and ``TT_METAL_DPRINT_MESH_COORDS`` are mutually exclusive.
- In ``TT_METAL_DPRINT_RISCVS``, ``DR`` (Blackhole DRAM cores) is accepted only where DRAM cores are programmable.
  An Ethernet RISC whose lines show ``ER`` (see `Output`_) is selected with ``ER0`` or ``ER*``.
- Turning DPRINT on or off recompiles kernels. Cores, chips and RISCs that aren't selected print nothing and never
  wait on printing.
- DPRINT can't be enabled together with the device profiler, the streaming profiler, the NOC debug dump or NOC
  transfer recording.

Writing a print
---------------

Include ``api/debug/dprint.h``; the standard dataflow and compute kernel headers also include it.
``api/debug/device_print.h`` provides the same API, but only under the ``DEVICE_PRINT`` names.

``DPRINT(format, args...)`` prints from whichever RISC runs it, and ``DEVICE_PRINT`` is the same macro under a
longer name. These variants print only from the RISCs they name, and compile to nothing (without evaluating their
arguments) on the others:

- ``DPRINT_UNPACK``, ``DPRINT_MATH`` and ``DPRINT_PACK`` print from the unpack, math or pack TRISC of a compute
  kernel. A plain ``DPRINT`` in a compute kernel prints from all three.
- ``DPRINT_DATA0`` and ``DPRINT_DATA1`` print from a data-movement RISC whose NOC index is 0 or 1, and never from
  compute, Ethernet or DRAM RISCs.
- Each has a ``DEVICE_PRINT_`` spelling as well, for example ``DEVICE_PRINT_MATH``.

``DPRINT_FORCE_NOINLINE`` (``DEVICE_PRINT_FORCE_NOINLINE``) marks a function of yours that prints: while DPRINT is on,
the compiler keeps exactly one copy of it; while DPRINT is off, the compiler decides as usual (see `Limits`_).

A print is one statement, written with a trailing ``;``. Use it wherever a statement can stand: in unbraced ``if``
and ``else`` branches, ``case`` labels, loops, lambdas, function and class templates, member functions,
``if constexpr`` branches and inline functions in headers. A print isn't an expression, so it can't be an operand
of ``,`` or ``?:``.

Arguments:

- Each argument is evaluated once when DPRINT is on, and not at all when it is off, so don't put side effects such
  as ``i++`` in an argument.
- Arguments are evaluated where the print stands, before the print takes its core's print buffer. On Blackhole a
  RISC sees L1 that other RISCs, cores or the NOC wrote only after a fence or a cache invalidation, so call
  ``invalidate_l1_cache()`` before printing such memory. The elements of a ``dp_typed_array_t`` are read later, after
  the print has taken the buffer and invalidated the cache.
- Arguments are copied, so bitfields, fields of packed structs and values read through ``volatile`` pointers can be
  printed directly.
- A print may carry up to 128 bytes of arguments on any core, for example 32 values of 4 bytes. `Limits`_ gives the
  largest prints.

Lines:

- End each line with ``\n``. The host writes a line only when its ``\n`` arrives, so one print can write several
  lines and several prints can build one line. Text after a RISC's last ``\n`` is lost when the device closes.
- ``\r`` ends a line but holds it back. Lines ending in ``\r`` are written together with the next line that ends
  in ``\n``, so a multi-line block isn't interleaved with other RISCs' output. In the output each ``\r`` becomes a
  line break, and every line of the block gets the prefix.

Format strings
--------------

The format must be a string literal. Adjacent literals, literals built by macros or by ``#`` stringizing, UTF-8
text and escapes such as ``\t`` are fine. The syntax is fmtlib's:

- ``{}`` prints the next argument and ``{N}`` prints argument N. Don't mix the two in one format. With ``{N}`` an
  argument can be printed several times, and every argument must be used. Each argument is sent to the host once,
  however many times it is printed.
- ``{{`` and ``}}`` print braces.
- A spec follows a colon: ``[[fill]align][sign][#][0][width][.precision][L][type]``, for example ``{:#010x}``,
  ``{:>8}`` or ``{:.3f}``. The type is one of ``a A b B c d e E f F g G o p s x X ?``.
- The spec must suit the argument's type, as in fmtlib: ``{:.2f}`` on an integer and ``{:x}`` on a float don't
  compile. On enums, strings and type names, specs follow fmtlib's rules for strings.
- ``{:#}`` on an enum prints the qualified name (see `Enums`_).
- Named arguments, dynamic width or precision (``{:{}}``) and chrono specs aren't supported.

What gets printed
-----------------

Plain values are ``bool``, ``char``, integers of 8 to 64 bits, ``float`` and ``double``. They print exactly as
``fmt::format`` prints them for the same format and a value of the same type, with the type's size on the device:
``long`` and ``unsigned long`` are 32 bits, except on Quasar's 64-bit data-movement RISCs. Each "Prints" entry
below is one output line, without its ``\n``:

.. list-table::
   :header-rows: 1

   * - Call
     - Prints
   * - ``DPRINT("Hello world!\n");``
     - ``Hello world!``
   * - ``DPRINT("x = {}\n", 42u);``
     - ``x = 42``
   * - ``DPRINT("{} {} {} {}\n", int8_t(-8), uint8_t(8), int16_t(-16), uint16_t(65535));``
     - ``-8 8 -16 65535``
   * - ``DPRINT("{} {} {}\n", int32_t(INT32_MIN), uint32_t(UINT32_MAX), uint64_t(UINT64_MAX));``
     - ``-2147483648 4294967295 18446744073709551615``
   * - ``DPRINT("{:#x} {:x} {:X}\n", 255u, 255u, 255u);``
     - ``0xff ff FF``
   * - ``DPRINT("{:08x} {:#010x}\n", 0xabcu, 0xbeefu);``
     - ``00000abc 0x0000beef``
   * - ``DPRINT("{:x}\n", int32_t(-1));``
     - ``-1`` (cast to an unsigned type to get ``ffffffff``)
   * - ``DPRINT("{0:x} {0} {0:o} {0:b}\n", 15);``
     - ``f 15 17 1111``
   * - ``DPRINT("{1} {0}\n", 1, 2);``
     - ``2 1``
   * - ``DPRINT("[{:>8}] [{:<8}] [{:^8}] [{:*^9}]\n", 123, 123, 123, 7);``
     - ``[     123] [123     ] [  123   ] [****7****]``
   * - ``DPRINT("{:+} [{: }] {:+}\n", 5, 5, -5);``
     - ``+5 [ 5] -5``
   * - ``DPRINT("{:.3f} [{:8.2f}] {:e}\n", 3.14159f, 3.14159f, 1234.5);``
     - ``3.142 [    3.14] 1.234500e+03``
   * - ``DPRINT("{} {}\n", 0.1f, 6.28);``
     - ``0.1 6.28``
   * - ``DPRINT("{} {} {}\n", __builtin_nanf(""), __builtin_inff(), -0.0f);``
     - ``nan inf -0``
   * - ``DPRINT("{} {}\n", true, false);``
     - ``true false``
   * - ``DPRINT("[{:5}][{:<5}][{:>5}]\n", true, false, true);``
     - ``[true ][false][ true]``
   * - ``DPRINT("{}\n", 'a');``
     - ``a``
   * - ``DPRINT("{:c}\n", uint32_t(65));``
     - ``A``
   * - ``DPRINT("{{}} {{{}}}\n", 1);``
     - ``{} {1}``

``int8_t`` and ``uint8_t`` print as numbers, following fmtlib's rule for ``signed char`` and ``unsigned char``.
Only plain ``char`` prints as a character.

DPRINT also prints these types:

.. list-table::
   :header-rows: 1

   * - Argument
     - Prints
     - Example
   * - ``bf16_t(bits)``
     - the float whose upper 16 bits are ``bits``
     - ``bf16_t(0x3dfb)`` prints ``0.122558594``
   * - ``bf8_t(exponent, mantissa)``, ``bf4_t(exponent, mantissa)``
     - a block-float element, as a float
     - ``bf4_t(128, 1)`` prints ``0.5``; ``bf8_t(130, 3)`` prints ``0.375``
   * - ``CTSTR("text")``
     - the text (see `Strings`_)
     - ``DPRINT("[{:>6}]\n", CTSTR("ab"))`` prints ``[    ab]``
   * - ``dp_type_name_t<T>()``
     - the fully qualified name of ``T`` (see `Type names`_)
     - ``dp_type_name_t<int>()`` prints ``int``
   * - an enum
     - the enumerator's name (see `Enums`_)
     - ``Color::Green`` prints ``Green``
   * - a pointer
     - ``0x`` followed by the address in hex; any spec is ignored
     - ``reinterpret_cast<void*>(0x1234)`` prints ``0x1234``
   * - ``TSLICE(...)``, ``TileSlice<N>(...)``, ``tile_slice_from_l1<N>(...)``
     - elements sampled from a tile (see `Circular buffers and tiles`_)
     -
   * - ``dp_typed_array_t<N>(format, words)``
     - ``N`` 32-bit words decoded as the given ``DataFormat`` (see `Typed arrays`_)
     -
   * - ``dp_top_callstack_t(...)``
     - the call stack at the print (see `Call stacks`_)
     -

Strings
^^^^^^^

The host can't read device memory, so a runtime ``const char*`` prints as its address. To print fixed text, wrap
the literal in ``CTSTR()``: the text is stored in the kernel binary, never on the device, and the spec applies to
it.

.. code-block:: c++

    const char* s = "Hello world!";
    DPRINT("Pointer: {}\n", s);                   // prints: Pointer: 0x<address>
    DPRINT("String: {}\n", CTSTR("Hello!"));      // prints: String: Hello!

``CTSTR(literal)`` produces a value of type ``ct_string``, which can be stored in a variable, passed to and returned
from functions, and printed later:

.. code-block:: c++

    ct_string state_name(uint32_t state) {
        switch (state) {
            case 0: return CTSTR("idle");
            case 1: return CTSTR("busy");
            default: return CTSTR("unknown");
        }
    }

    DPRINT("state: {}\n", state_name(state));

Type names
^^^^^^^^^^

``dp_type_name_t<T>()`` prints the fully qualified name of ``T``. The name is worked out at compile time and stored
in the kernel binary, like a ``CTSTR``.

.. code-block:: c++

    namespace ns { struct MyStruct {}; }

    ns::MyStruct x;
    DPRINT("type: {}\n", dp_type_name_t<int>());          // prints: type: int
    DPRINT("type: {}\n", dp_type_name_t<decltype(x)>());  // prints: type: ns::MyStruct

Enums
^^^^^

An enum prints as the name of its enumerator. ``{:#}`` adds the fully qualified type name, and fill, align and width
apply to the printed name. The names come from the kernel's debug information, which is on by default; without it,
or for a value that has no name, DPRINT prints ``(Type)value``. The examples declare their enums at global scope.

.. code-block:: c++

    enum class Color : uint8_t { Red = 0, Green = 1, Blue = 2 };
    DPRINT("Color: {}\n", Color::Green);              // prints: Color: Green
    DPRINT("Color: {:#}\n", Color::Blue);             // prints: Color: Color::Blue
    DPRINT("[{:>10}]\n", Color::Green);               // prints: [     Green]
    DPRINT("Color: {}\n", static_cast<Color>(7));     // prints: Color: (Color)7

An enum with an ``operator|`` that returns the enum type is printed as flags, joined with ``|``:

.. code-block:: c++

    enum class Bits : uint32_t { A = 1, B = 2, C = 4 };
    constexpr Bits operator|(Bits a, Bits b) {
        return static_cast<Bits>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }
    DPRINT("Bits: {}\n", Bits::A | Bits::C);    // prints: Bits: A | C
    DPRINT("Bits: {:#}\n", Bits::A | Bits::C);  // prints: Bits: Bits::A | Bits::C

Type names are always fully qualified: inside namespace ``ns``, the examples print ``ns::Color::Blue`` and
``(ns::Color)7``.

Circular buffers and tiles
^^^^^^^^^^^^^^^^^^^^^^^^^^

``TileSlice`` samples elements of a tile in a circular buffer (CB) and prints them row by row, each element
formatted with the spec. ``TSLICE`` is ``TileSlice<64>``, which carries up to 64 bytes of tile data; use
``TileSlice<N>`` with a larger ``N`` (a multiple of 4, below 256) for bigger slices. A slice with more data than
fits is printed truncated, with a note saying so.

.. list-table::
   :header-rows: 1

   * - Argument
     - Type
     - Description
   * - ``cb_id``
     - ``uint8_t``
     - Circular buffer to print from.
   * - ``tile_idx``
     - ``int``
     - Index of the tile inside the CB, relative to the read or write pointer.
   * - ``slice_range``
     - ``SliceRange``
     - Rows and columns to sample, like a numpy slice ``[h0:h1:hs, w0:w1:ws]``. The fields are ``h0``, ``h1``,
       ``hs``, ``w0``, ``w1`` and ``ws``, all ``uint8_t``. Ready-made ranges: ``SliceRange::hw0_32_16()``,
       ``hw0_32_8()``, ``hw0_32_4()``, ``h0_w0_32()`` (row 0), ``h0_32_w0()`` (column 0), ``h0_32_w1()`` (column 1)
       and ``hw041()`` (the top-left 4x4).
   * - ``cb_type``
     - ``dprint_tslice_cb_t``
     - BRISC and NCRISC only, and required there: ``TSLICE_INPUT_CB`` or ``TSLICE_OUTPUT_CB``, which picks the CB's
       data format.
   * - ``ptr_type``
     - ``dprint_tslice_ptr_t``
     - BRISC and NCRISC only: ``TSLICE_RD_PTR`` reads from the front of the CB and ``TSLICE_WR_PTR`` from the back.
       The default is ``TSLICE_RD_PTR`` on BRISC and ``TSLICE_WR_PTR`` on NCRISC. The unpack TRISC always reads
       from the front and the pack TRISC from the back.
   * - ``endl_rows``
     - ``bool``
     - Start a new line after each row. Default ``true``.
   * - ``print_untilized``
     - ``bool``
     - Untilize the data while printing it. Default ``true``; pass ``false`` when the CB holds data that isn't
       tilized.

The arguments are positional, in the order of the table; on other RISCs, leave out ``cb_type`` and ``ptr_type``.
On BRISC and NCRISC the fourth argument is always ``cb_type``, so ``TileSlice<128>(cb_id, 0, sr, false)`` from the
compute example below means ``TSLICE_OUTPUT_CB`` there, not ``endl_rows=false``.

``TileSlice`` reads CBs only on BRISC, NCRISC and the unpack and pack TRISCs. On any other RISC, such as the math
TRISC or an Ethernet, DRAM or Quasar data-movement RISC, a warning is printed instead of the data. A warning also
replaces the data when the tile's address is outside the L1 range that circular buffers can occupy. The supported
data formats are ``Float32``, ``Float16_b``, ``Bfp8_b``, ``Bfp4_b``, ``Int8``, ``UInt8``, ``UInt16``, ``Int32``
and ``UInt32``; any other format prints a warning.

A tile's data format is known only at run time, so the host checks the spec against it then. When the spec doesn't
suit the elements, such as ``{:x}`` on a ``Float16_b`` tile, the elements print as ``{}`` would, and a warning line
comes before the print's text: ``Warning: invalid format specifier {:x} for Float16_b data, printed with {}``. A spec
that suits no number, such as ``{:s}``, doesn't compile. The same holds for ``dp_typed_array_t``.

Sampling happens relative to the CB's current read or write pointer. To print a tile from the front of a CB, print
between ``cb_wait_front`` and ``cb_pop_front``. To print a tile from the back, print between ``cb_reserve_back``
and ``cb_push_back``.

.. code-block:: c++

    // Data-movement kernel:
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    cb_wait_front(cb_id, 1);
    DPRINT("Tile:\n{}\n", TSLICE(cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR));
    cb_pop_front(cb_id, 1);

    // Compute kernel: the unpack TRISC reads from the front of the CB.
    DPRINT_UNPACK("Tile:\n{}\n", TSLICE(cb_id, 0, SliceRange::hw0_32_8()));

    // Print a whole tile, one row per print.
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_UNPACK("{}: {}\n", r, TileSlice<128>(cb_id, 0, sr, /*endl_rows=*/false));
    }

``tile_slice_from_l1<N>(l1_addr, format, slice_range)`` reads a tile straight from an L1 address and always
untilizes it. Optional trailing arguments are, in order, ``tile_dim_r``, ``tile_dim_c``, ``face_dim_r``,
``face_dim_c`` and ``num_faces``, which default to a 32x32 tile of four 16x16 faces, and ``endl_rows``, which
defaults to ``true``.

Typed arrays
^^^^^^^^^^^^

``dp_typed_array_t<N>(format, words)`` prints ``N`` 32-bit words as values of the given ``DataFormat``, each value
followed by a space and formatted with the spec. A word holds one ``Float32``, ``Tf32``, ``Int32`` or ``UInt32``
value; two ``UInt16``, ``Float16``, ``Float16_b``, ``Lf8`` or block-float (``Bfp8``, ``Bfp4``, ``Bfp2`` and their
``_b`` forms) values; or four ``Int8`` or ``UInt8`` values. Values that share a word print low bits first. Other
formats aren't supported and print a warning. Integers and ``Float32`` are read as stored in memory. The other
float formats must be in the DEST register's field order (sign, mantissa, exponent), which is how the DEST register
dumps in ``api/debug/dprint_tensix.h`` send them; ordinary ``bfloat16`` words in memory don't print as their values.

.. code-block:: c++

    uint32_t words[4] = {0x3F800000, 0x40000000, 0x40400000, 0x40800000};
    DPRINT("[{}]\n", dp_typed_array_t<4>(static_cast<uint16_t>(DataFormat::Float32), words));  // prints: [1 2 3 4 ]

Call stacks
^^^^^^^^^^^

``dp_top_callstack_t::current(skip_frames)`` captures the current position, and the host prints the call stack
there, resolved from debug information, as a tree of lines: each frame's function, then its file and line,
innermost frame first. A last leaf of ``...`` means the rest of the stack couldn't be resolved.
``dp_top_callstack_t(pc, ra, skip_frames)`` does the same for a given program counter and return address. The LLK
sanitizer uses this in its reports.

Output
------

- Each RISC's text is kept apart, and so are its kernel's and its firmware's: lines from different RISCs, or from a
  RISC's kernel and firmware, never mix within a line, but the order of lines from different RISCs is not defined.
- With the prefix on (the default), each line starts with ``<device>:<x>-<y>:<RISC>:`` and a space, in logical
  core coordinates. For example, ``DPRINT("Hello\n")`` on TRISC0 of core (1,2) on device 0 prints
  ``0:1-2:TR0: Hello``. The RISC names are ``BR``, ``NC``, ``TR0``-``TR2`` and ``DR``. An Ethernet RISC is ``ER``
  on a core that runs one ERISC (Wormhole, and Blackhole active Ethernet cores in single-ERISC mode), and
  ``ER0``/``ER1`` on a core that runs two. On Quasar the names are ``DM0``-``DM7`` and ``N0TR0``-``N3TR3`` on Tensix
  cores, and ``DE-DM0``-``DE-DM7`` on dispatch-engine cores.
- With ``TT_METAL_DPRINT_TAG_FIRMWARE=1``, each line that firmware printed starts with ``[FW]`` and a space, ahead
  of the prefix: ``[FW] 0:1-2:DM0: <text>``.
- With ``TT_METAL_DPRINT_ONE_FILE_PER_RISC``, each RISC writes
  ``generated/dprint/device-<id>_<core type>-core-<x>-<y>_<RISC>.txt`` under ``TT_METAL_LOGS_PATH`` (the current
  directory if that isn't set). The core type is ``worker``, ``ethernet``, ``DRAM`` or ``dispatch``. The RISC name
  is the long one: ``BRISC``, ``NCRISC``, ``TRISC0``-``TRISC2``, ``ERISC`` or ``ERISC0``/``ERISC1``, ``DRISC``, and
  on Quasar ``DM0``-``DM7``, ``Neo0TRISC0``-``Neo3TRISC3`` and ``DE-DM0``-``DE-DM7``. For example,
  ``device-0_worker-core-1-2_TRISC0.txt``.
- Text is written shortly after the device prints it, even while the kernel is still running or is hung.
- Fast and slow dispatch print the same text.

Compile-time checks
-------------------

A mistake in a print is a compile error that points at the print. The messages contain these phrases:

.. list-table::
   :header-rows: 1

   * - Mistake
     - Example
     - The message contains
   * - malformed braces or spec, or an unsupported feature
     - ``DPRINT("{n = {0}}}\n", 42);``
     - ``Invalid format string``
   * - mixing ``{}`` and ``{N}``
     - ``DPRINT("{} {0}\n", 42);``
     - ``Cannot mix indexed ({0}) and non-indexed ({}) placeholders``
   * - an index with no argument
     - ``DPRINT("{1}\n", 42);``
     - ``Placeholder index exceeds number of arguments``
   * - with ``{N}``, an argument that's never printed
     - ``DPRINT("{0} {0}\n", 42, 5);``
     - ``All arguments must be referenced``
   * - with ``{}``, a different number of placeholders and arguments
     - ``DPRINT("{} {}\n", 42);``
     - ``Number of {} placeholders must match number of arguments``
   * - an argument type DPRINT can't print
     - a struct
     - ``No serialization defined for this type``
   * - a spec that doesn't suit the argument
     - ``DPRINT("{:.2f}\n", 5u);``
     - ``invalid format specifier``
   * - a print too big for its print buffer
     - a large ``dp_typed_array_t``
     - ``does not fit in the print buffer``
   * - more than 64 arguments in one print (a ``dp_typed_array_t`` or a ``TileSlice`` is one)
     - 65 values
     - ``has more than 64 arguments``
   * - more than 256 bytes of values in one print (``dp_typed_array_t`` and ``TileSlice`` arguments don't count)
     - 33 values of 8 bytes
     - ``take more than 256 bytes``
   * - a format that isn't a string literal
     - ``DPRINT(format_ptr, 1);``
     - ``format must be a string literal``

Limits
------

Size:

- A print carries its arguments in one message of its RISC's print buffer: a 4-byte header and the arguments, padded
  to 4 bytes. Every RISC's buffer takes 128 bytes of arguments; a print that doesn't fit its RISC's buffer fails to
  compile with ``does not fit in the print buffer``.
- A print takes at most 64 arguments, and its values other than ``dp_typed_array_t`` and ``TileSlice`` arguments take
  at most 256 bytes, for example 64 values of 4 bytes or 32 of 8 bytes. Split a bigger print into several prints, or
  print an array as a ``dp_typed_array_t``.
- A value takes its type's size: ``bool`` and ``char`` 1 byte; ``bf16_t``, ``bf8_t`` and ``bf4_t`` 2 bytes; an enum
  its underlying type's size; a pointer, a ``CTSTR`` and a type name a pointer's size (4 bytes, 8 on Quasar's
  data-movement RISCs); a ``dp_top_callstack_t`` three pointers. Values smaller than 4 bytes share 4-byte words, so
  a print's size is the sum of its values' sizes, rounded up to 4.

Stack and code size:

- A print evaluates its arguments where it stands, and the function that contains it keeps their values, in
  registers or on its stack, until the print has copied them. A function with wide prints therefore needs more stack,
  also while it calls other functions, and deep call chains or recursion through such functions add it up. If a
  kernel runs out of stack, or a function's stack frame goes over a compile-time limit, split the print, print fewer
  values per print, or move the print into a function of its own marked ``DPRINT_FORCE_NOINLINE``.
- The compiler can copy a function that prints into each of its callers, typically a function declared ``inline``, a
  lambda or a member function, and every copy carries the function's prints. If a kernel grows too large while DPRINT
  is on, mark such functions with ``DPRINT_FORCE_NOINLINE`` (``DEVICE_PRINT_FORCE_NOINLINE``): while DPRINT is on,
  the compiler keeps exactly one copy of the function, neither copied into its callers nor specialized per constant
  argument; while DPRINT is off, the macro is empty and the compiler decides as usual.

.. code-block:: c++

    DPRINT_FORCE_NOINLINE inline void print_cb_state(uint32_t cb_id) {
        DPRINT("cb {}: rd {} wr {}\n", cb_id, get_local_cb_interface(cb_id).fifo_rd_ptr,
               get_local_cb_interface(cb_id).fifo_wr_ptr);
    }

Best practices
--------------

- **Print a value several ways with one argument.** ``DPRINT("{0} = {0:#x}\n", v)`` sends ``v`` to the host once;
  ``DPRINT("{} = {:#x}\n", v, v)`` sends it twice.
- **Put fixed text in the format string.** It stays in the kernel binary on the host and costs nothing on the
  device, however long it is.
- **Combine related values into one print.** Each print costs a few instructions in the kernel and a small header in
  the print buffer, so one print with four values is cheaper than four prints with one value each.
- **Use** ``CTSTR`` **for fixed strings.** A runtime ``const char*`` prints only its address.
- **Use an unsigned type for hex.** ``{:x}`` on a negative signed value prints a minus sign: ``int32_t(-1)`` prints
  ``-1``, not ``ffffffff``.
- **End every line with** ``\n``. A line without one isn't written until a later print ends it, and is lost if the
  device closes first.
- **Keep arguments free of side effects.** They aren't evaluated when DPRINT is off.
- **Print only what you need, and only from where you need it.** Narrow ``TT_METAL_DPRINT_CORES`` and
  ``TT_METAL_DPRINT_RISCVS``, and use the thread variants (``DPRINT_MATH`` and so on) in compute kernels. When a
  core's print buffer is full, the printing RISC waits for the host, which changes the kernel's timing.
- **Avoid prints in hot loops when timing matters,** for the same reason. Turn DPRINT off before measuring
  performance; with it off, prints compile to nothing.
