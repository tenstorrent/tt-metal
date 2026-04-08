Device Debug Print
==================

.. note::
   Tools are only fully supported on source builds.

Overview
--------

The device can optionally print to the host terminal or a log file.  This feature can be useful for printing variables,
addresses, and Circular Buffer data from kernels running on the device. Device-side prints are controlled through API
calls; the host-side is controlled through environment variables.

Enabling
--------

Device debug printing can be enabled and configured using the environment variables shown below.  The first
environment variable, ``TT_METAL_DPRINT_CORES`` specifies which cores the host-side will read print data from, and
whether this environment variable is defined determines whether printing is enabled during kernel compilation.
Note that the core coordinates are logical coordinates, so worker cores and ethernet cores both start at (0, 0).
IMPORTANT: During deprecation period, ``TT_METAL_DEVICE_PRINT`` must also be set to 1 to use the new DEVICE_PRINT system.
If only TT_METAL_DPRINT_CORES is set, the legacy DPRINT system will be used.

.. code-block::

    export TT_METAL_DPRINT_CORES=0,0                    # required, x,y OR (x1,y1),(x2,y2),(x3,y3) OR (x1,y1)-(x2,y2) OR all OR worker OR dispatch
    export TT_METAL_DPRINT_ETH_CORES=0,0                # optional, x,y OR (x1,y1),(x2,y2),(x3,y3) OR (x1,y1)-(x2,y2) OR all OR worker OR dispatch
    export TT_METAL_DPRINT_CHIPS=0                      # optional, comma separated list of chips OR all. Default is all. Mutually exclusive with TT_METAL_DPRINT_NODES and TT_METAL_DPRINT_MESH_COORDS.
    export TT_METAL_DPRINT_NODES="(M0,D0),(M0,D1)"      # optional, comma separated list of `FabricNodeId` nodes (unique node identifiers in format (Mn,Dn), where M is mesh ID and D is device ID) OR all. Default is all. Mutually exclusive with TT_METAL_DPRINT_CHIPS and TT_METAL_DPRINT_MESH_COORDS.
    export TT_METAL_DPRINT_MESH_COORDS="(0,0),(1,3)"    # optional, comma separated list of (row,col) coordinates in the global system mesh OR all. Default is all. Mutually exclusive with TT_METAL_DPRINT_CHIPS and TT_METAL_DPRINT_NODES.
    export TT_METAL_DPRINT_RISCVS=BR                    # optional, default is all RISCs. Use a subset of BR,NC,TR0,TR1,TR2,TR*,ER0,ER1,ER*
    export TT_METAL_DPRINT_FILE=log.txt                 # optional, default is to print to the screen
    export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=0   # optional, enabled by default. Prepends prints with <device id>:(<core x>, <core y>):<RISC>:.
    export TT_METAL_DPRINT_ONE_FILE_PER_RISC=1          # optional, splits DPRINT data on a per-RISC basis into files under $TT_METAL_HOME/generated/dprint/. Overrides TT_METAL_DPRINT_FILE and disables TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC.
    export TT_METAL_DEVICE_PRINT=1                      # required, use new DEVICE_PRINT system instead of legacy DPRINT. This option is available only during deprecation period of DPRINT, and will be removed in a future release.

To generate device debug prints on the device, include the ``api/debug/device_print.h`` header and use the APIs defined there.
An example with the different features available is shown below:

.. code-block:: c++

    #include "api/debug/device_print.h"  // required in all kernels using DEVICE_PRINT

    enum TestEnum { VAL0, VAL1, VAL2 };

    void kernel_main() {
        // Supported scalar types: bool (prints false/true), char, all fixed-width integer types
        // (uint8_t-uint64_t, int8_t-int64_t), float, and double.
        DEVICE_PRINT("Test string {} {} {}\n", 'a', 5, 0.123456f);
        // bool prints as false or true
        bool flag = true;
        DEVICE_PRINT("Bool value: {}\n", flag);  // prints: Bool value: true
        // BF16 type printing is supported via provided type
        bf16_t my_bf16_val(0x3dfb); // Equivalent to 0.122559
        DEVICE_PRINT("BF16 value: {}\n", my_bf16_val);

        // DEVICE_PRINT supports formatting options that are supported by fmtlib:
        DEVICE_PRINT("{:.5f}\n", 0.123456f);
        DEVICE_PRINT("{:>10}\n", 123); // right align in a field of width 10
        DEVICE_PRINT("{:<10}\n", 123); // left align in a field of width 10
        DEVICE_PRINT("{0:x} {0} {0:o} {0:b}\n", 15); // single argument print in hexadecimal, decimal, octal, and binary

        // The following prints only occur on a particular RISCV core:
        DEVICE_PRINT_MATH("this is the math kernel\n");
        DEVICE_PRINT_PACK("this is the pack kernel\n");
        DEVICE_PRINT_UNPACK("this is the unpack kernel\n");
        DEVICE_PRINT_DATA0("this is the data movement kernel on noc 0\n");
        DEVICE_PRINT_DATA1("this is the data movement kernel on noc 1\n");
    }

Strings
^^^^^^^

Runtime ``const char*`` pointers are printed as hex addresses since the host cannot read device memory.
To print the actual string content, use ``CTSTR()`` which stores the string in the ELF at compile time
so the host can resolve it:

.. code-block:: c++

    const char* s = "Hello world!";
    DEVICE_PRINT("Pointer: {}\n", s);               // prints: Pointer: 0x12345678
    DEVICE_PRINT("String: {}\n", CTSTR("Hello!"));  // prints: String: Hello!

Enums
^^^^^

Enum types are supported natively. When DWARF debug info is present in the ELF, enum values are
printed as their symbolic names. Use ``{:#}`` to include the fully-qualified type name:

.. code-block:: c++

    enum class Color : uint8_t { Red = 0, Green = 1, Blue = 2 };
    DEVICE_PRINT("Color: {}\n", Color::Green);    // prints: Color: Green
    DEVICE_PRINT("Color: {:#}\n", Color::Blue);   // prints: Color: Color::Blue

Flag enums (with ``operator|``) are detected at compile time and printed with ``|`` separators:

.. code-block:: c++

    enum class Flags : uint32_t { A = 1, B = 2, C = 4 };
    constexpr Flags operator|(Flags a, Flags b) {
        return static_cast<Flags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }
    DEVICE_PRINT("Flags: {}\n", Flags::A | Flags::C);    // prints: Flags: A | C
    DEVICE_PRINT("Flags: {:#}\n", Flags::A | Flags::C);  // prints: Flags: Flags::A | Flags::C

If no DWARF debug info is available, enum values are printed as ``(TypeName)integer``.

Circular Buffers
^^^^^^^^^^^^^^^^

Data from Circular Buffers can be printed using the ``TileSlice`` object. It can be constructed as described below, and fed directly to a ``DEVICE_PRINT`` call.

+-----------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Argument        | Type                | Description                                                                                                                                                  |
+=================+=====================+==============================================================================================================================================================+
| cb_id           | uint8_t             | Id of the Circular Buffer to print data from.                                                                                                                |
+-----------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tile_idx        | int                 | Index of tile inside the CB to print data from.                                                                                                              |
+-----------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| slice_range     | SliceRange          | A struct to describe starting index, ending index, and stride for data to print within the CB. Fields are ``h0``, ``h1``, ``hs``, ``w0``, ``w1``,            |
|                 |                     | ``ws``, all ``uint8_t``.                                                                                                                                     |
+-----------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| cb_type         | dprint_tslice_cb_t  | Only used for Data Movement RISCs, specify ``TSLICE_INPUT_CB`` or ``TSLICE_OUTPUT_CB`` depending on if the CB to print from is input or output.              |
+-----------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ptr_type        | dprint_tslice_ptr_t | Only used for Data Movement RISCs, specify ``TSLICE_RD_PTR`` to read from the front of the CB, or ``TSLICE_WR_PTR`` to read from the back of the CB.         |
|                 |                     | UNPACK RISC only reads from the front of the CB, PACK RISC only reads from the back of the CB.                                                               |
+-----------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| endl_rows       | bool                | Whether to add a newline between printed rows, default ``true``.                                                                                             |
+-----------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| print_untilized | bool                | Whether to untilize the CB data while printing it (always done for block float formats), default ``true``.                                                   |
+-----------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

An example of how to print data from a CB (in this case, ``CBIndex::c_25``) is shown below.  Note that sampling happens relative
to the current CB read or write pointer. This means that for printing a tile read from the front of the CB, the
``DEVICE_PRINT`` call has to occur between the ``cb_wait_front`` and ``cb_pop_front`` calls. For printing a tile from the
back of the CB, the ``DEVICE_PRINT`` call has to occur between the ``cb_reserve_back`` and ``cb_push_back`` calls. Currently supported data
formats for printing from CBs are ``DataFormat::Float32``, ``DataFormat::Float16_b``, ``DataFormat::Bfp8_b``, ``DataFormat::Bfp4_b``,
``DataFormat::Int8``, ``DataFormat::UInt8``, ``DataFormat::UInt16``, ``DataFormat::Int32``, and ``DataFormat::UInt32``.

.. code-block:: c++

    #include "api/debug/device_print.h"  // required in all kernels using DEVICE_PRINT

    void kernel_main() {
        // Assuming the tile we want to print from CBIndex::c_25 is from the front the CB, print must happen after
        // this call. If the tile is from the back of the CB, then print must happen after cb_reserve_back().
        cb_wait_front(CBIndex::c_25, 1);
        ...

        // Extract a numpy slice `[0:32:16, 0:32:16]` from tile `0` from `CBIndex::c_25` and print it.
        DEVICE_PRINT("{}\n", TSLICE(CBIndex::c_25, 0, SliceRange::hw0_32_16()));
        // Note that since the MATH core does not have access to CBs, so this is an invalid print:
        DEVICE_PRINT_MATH("{}\n", TSLICE(CBIndex::c_25, 0, SliceRange::hw0_32_16())); // Invalid

        // Print a full tile
        for (int32_t r = 0; r < 32; ++r) {
            SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            // On data movement RISCs, tiles can be printed from either the CB read or write pointers. Also need to specify whether
            // the CB is input or output.
            DEVICE_PRINT_DATA0("{} --READ--cin1-- {}\n", (uint)r, TileSlice(0, 0, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false));
            DEVICE_PRINT_DATA1("{} --READ--cin1-- {}\n", (uint)r, TileSlice(0, 0, sr, TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true, false));
            // Unpacker RISC only has rd_ptr and only input CBs, so no extra args
            DEVICE_PRINT_UNPACK("{} --READ--cin1-- {}\n", (uint)r, TileSlice(0, 0, sr, true, false));
            // Packer RISC only has wr_ptr
            DEVICE_PRINT_PACK("{} --READ--cin1-- {}\n", (uint)r, TileSlice(0, 0, sr, true, false));
        }

        ...
        cb_pop_front(CBIndex::c_25, 1);
    }

.. note::
    The DEVICE_PRINT buffer for a RISC is only flushed when new line character ``\n`` is read, or the device that the RISC belongs to is closed.
