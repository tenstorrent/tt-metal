Kernel Debug Print
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

Kernel debug printing can be enabled and configured using the environment variables shown below.  The first
environment variable, ``TT_METAL_DPRINT_CORES`` specifies which cores the host-side will read print data from, and
whether this environment variable is defined determines whether printing is enabled during kernel compilation.
Note that the core coordinates are logical coordinates, so worker cores and ethernet cores both start at (0, 0).

.. code-block::

    export TT_METAL_DPRINT_CORES=0,0           # required, x,y OR (x1,y1),(x2,y2),(x3,y3) OR (x1,y1)-(x2,y2) OR all OR worker OR dispatch
    export TT_METAL_DPRINT_ETH_CORES=0,0       # optional, x,y OR (x1,y1),(x2,y2),(x3,y3) OR (x1,y1)-(x2,y2) OR all OR worker OR dispatch
    export TT_METAL_DPRINT_CHIPS=0             # optional, comma separated list of chips
    export TT_METAL_DPRINT_RISCVS=BR           # optional, default is all RISCs.  Use a subset of BR,NC,TR0,TR1,TR2
    export TT_METAL_DPRINT_FILE=log.txt        # optional, default is to print to the screen
    export TT_METAL_DPRINT_ONE_FILE_PER_RISC=1 # optional, splits DPRINT data on a per-RISC basis into files under $TT_METAL_HOME/generated/dprint/. Overrides TT_METAL_DPRINT_FILE.

To generate kernel debug prints on the device, include the ``debug/dprint.h`` header and use the APIs defined there.
And example with the different features available is shown below:

.. code-block:: c++

    #include "debug/dprint.h"  // required in all kernels using DPRINT

    void kernel_main() {
        // Direct printing is supported for const char*/char/uint32_t/float
        DPRINT << "Test string" << 'a' << 5 << 0.123456f << ENDL();
        // BF16 type printing is supported via a macro
        uint16_t my_bf16_val = 0x3dfb; // Equivalent to 0.122559
        DPRINT << BF16(my_bf16_val) << ENDL();

        // dprint.h includes macros for a subset of std::ios and std::iomanip functionality:
        // SETPRECISION macro has the same behaviour as std::setprecision
        DPRINT << SETPRECISION(5) << 0.123456f << ENDL();
        // FIXED and DEFAULTFLOAT macros have the same behaviour as std::fixed and std::defaultfloat
        DPRINT << FIXED() << 0.123456f << DEFAULTFLOAT() << 0.123456f << ENDL();
        // SETW macro is the same as std::setw, but with an optional sticky flag (default true)
        DPRINT << "SETW (sticky): " << SETW(10) << 1 << 2 << ENDL();
        DPRINT << "SETW (non-sticky): " << SETW(10, false) << 1 << 2 << ENDL();
        // HEX/DEC/OCT macros corresponding to std::hex/std::dec/std::oct
        DPRINT << HEX() << 15 << DEC() << 15 << OCT() << 15 << ENDL();

        // The following prints only occur on a particular RISCV core:
        DPRINT_MATH(DPRINT << "this is the math kernel" << ENDL());
        DPRINT_PACK(DPRINT << "this is the pack kernel" << ENDL());
        DPRINT_UNPACK(DPRINT << "this is the unpack kernel" << ENDL());
        DPRINT_DATA0(DPRINT << "this is the data movement kernel on noc 0" << ENDL());
        DPRINT_DATA1(DPRINT << "this is the data movement kernel on noc 1" << ENDL());
    }

The APIs for printing data from Circular Buffers can be found in ``debug/dprint_tile.h``.  These APIs use the
``SliceRange`` struct to print tile contents with a given sample count, starting index, and stride.  An example of
how to print data from a CB (in this case, ``CB::c_intermed1``) is shown below.  Note that sampling happens relative
to the current CB read or write pointer. This means that for printing a tile read from the front of the CB, the
``DPRINT`` call has to occur between the ``cb_wait_front`` and ``cb_pop_front`` calls. For printing a tile from the
back of the CB, the ``DPRINT`` call has to occur between the ``cb_reserve_back`` and ``cb_push_back`` calls.

.. code-block:: sh

    #include "debug/dprint.h"  // required in all kernels using DPRINT

    void kernel_main() {
        // Assuming the tile we want to print from CB::c_intermed1 is from the front the CB, print must happen after
        // this call. If the tile is from the back of the CB, then print must happen after cb_reserve_back().
        cb_wait_front(CB::c_intermed1, 1);
        ...

        // Extract a numpy slice `[0:32:16, 0:32:16]` from tile `0` from `CB::c_intermed1` and print it.
        DPRINT  << TSLICE(CB::c_intermed1, 0, SliceRange::hw0_32_16()) << ENDL();
        // Note that since the MATH core does not have access to CBs, so this is an invalid print:
        DPRINT_MATH({ DPRINT  << TSLICE(CB::c_intermed1, 0, SliceRange::hw0_32_16()) << ENDL(); }); // Invalid

        // Print a full tile
        for (int32_t r = 0; r < 32; ++r) {
            SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            DPRINT << (uint)r << " --READ--cin0-- " << TileSlice(0, 0, sr, true, false) << ENDL();
        }

        ...
        cb_pop_front(CB::c_intermed1, 1);
    }
