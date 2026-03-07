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
