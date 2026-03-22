Device Debug Print
==================

.. note::
   Tools are only fully supported on source builds.

Overview
--------

``DEVICE_PRINT`` is an experimental feature that is meant to replace ``DPRINT``.
For more info about ``DPRINT``, see the `kernel_print` tool documentation.

Enabling
--------

To enable ``DEVICE_PRINT`` you need to first enable ``DPRINT``. Then, you should enable feature switch that will allow usage of ``DEVICE_PRINT``.

.. code-block::

    export TT_METAL_DEVICE_PRINT=1                    # required, use new DEVICE_PRINT system instead of legacy DPRINT.

To generate device debug prints, include the ``api/debug/dprint.h`` header and use the APIs defined there.
An example with the different features available is shown below:

.. code-block:: c++

    #include "api/debug/dprint.h"  // required in all kernels using DPRINT

    void kernel_main() {
        // Direct printing is supported for const char*/char/uint32_t/float
        DEVICE_PRINT("Test string {} {} {}\n", 'a', 5, 0.123456f);
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

Data from Circular Buffers can be printed using the ``TileSlice`` object. It can be constructed as described below, and fed directly to a ``DEVICE_PRINT`` call.

+-----------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Argument        | Type                | Description                                                                                                                                                  |
+=================+=====================+==============================================================================================================================================================+
| cb_id           | uint8_t             | Id of the Circular Buffer to print data form.                                                                                                                |
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
``DataFormat::Int8``, ``DataFormat::UInt8``, ``DataFormat::UInt16``, ``DataFormat::Int32``, and ``DataFormat::UInt832``.

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
