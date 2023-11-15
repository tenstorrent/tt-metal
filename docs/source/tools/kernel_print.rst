Kernel Debug Print
==================

Overview
--------

The device can optionally print to the host terminal.  Device side prints are controlled through API calls; the host
side is controlled through environment variables.

Enabling
--------

To generate kernel debug prints on the device:

.. code-block::

    #include "debug/dprint.h"  // required in all kernels using DPRINT

    void kernel_main() {
        DPRINT << 5 << ENDL();
        DPRINT << "this is a string" << 1 << SETW(8) << F32(2.2f) << ENDL();

        // The following prints only occur on a particular RISCV core:
        DPRINT_MATH(DPRINT << "this is the math kernel" << ENDL());
        DPRINT_PACK(DPRINT << "this is the pack kernel" << ENDL());
        DPRINT_UNPACK(DPRINT << "this is the unpack kernel" << ENDL());
        DPRINT_DATA0(DPRINT << "this is the data movement kernel on noc 0" << ENDL());
        DPRINT_DATA1(DPRINT << "this is the data movement kernel on noc 1" << ENDL());

        // Print a tile slice
        DPRINT_PACK({ DPRINT  << TSLICE(CB::c_intermed1, 0, SliceRange::hw0_32_16()) << ENDL(); });

        // Print a full tile
        for (int32_t r = 0; r < 32; ++r) {
            SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            DPRINT << (uint)r << " --READ--cin0-- " << TileSlice(0, 0, sr, true, false) << ENDL();
        }
    }

The ``TSLICE`` macros support printing tile contents with a given sample count, starting index and stride.  The
first example above extracts a numpy slice ``[0:32:16, 0:32:16]`` from tile ``0`` from ``CB::c_intermed1``.

To display the kernel debug prints on the host:

.. code-block::

    export TT_METAL_DPRINT_CORES=1,1     # required, x,y OR (x1,y1),(x2,y2),(x3,y3) OR (x1,y1)-(x2,y2) OR all
    export TT_METAL_DPRINT_CHIPS=0       # optional, comma separated list of chip
    export TT_METAL_DPRINT_RISCVS=BR     # optional, default is all RISCs.  Use a subset of BR,NC,TR0,TR1,TR2
    export TT_METAL_DPRINT_FILE=log.txt  # optional, default is to print to the screen

**NOTE:** the core coordinates are currently physical NOC coordinates (not logical); the top left core is (1,1) *not*
(0,0)
