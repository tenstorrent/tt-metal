.. _achieving_fp32_accuracy_for_computation:

Achieving FP32 Accuracy for Computation
=======================================

Tensix provides two main compute engines: the matrix engine (FPU) and the vector engine (SFPU). Each has distinct strengths and limitations that affect numerical accuracy and throughput. For a detailed overview of these engines, see :ref:`Compute Engines and Data Flow within Tensix <compute_engines_and_dataflow_within_tensix>`.

The matrix engine is built for speed and scale, handling large matrix operations efficiently. Its design favors throughput, but this comes with a trade-off: most operations use bfloat16 or TF32 formats, which offer less precision than standard IEEE 754 FP32. Additionally, the matrix engine does not handle special values (inf, NaN, ...) properly. For many machine learning tasks, this is sufficient, but it may not meet the needs of workloads that demand high numerical accuracy. For detailed information about FPU and SFPU numerical accuracy characteristics, please review the follwoing documentations:

* `SFPU FMA Numerical Accuracy <https://github.com/tenstorrent/tt-isa-documentation/blob/main/Miscellaneous/FMA/README.md#correctness-of-fma_model_ieee>`_
* `Floaring Point Bit Patterns <https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/FloatBitPatterns.md>`_
* `FPU SrcA/B and Fidelity Phases <https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md>`_

The vector engine, on the other hand, supports full 32-bit floating-point (FP32) arithmetic and is more IEEE 754-compliant (though not 100%). This makes it suitable for computations where precision is critical. However, as a vector unit, it processes data in smaller batches and at lower throughput—behavior similar to SIMD units found in conventional CPUs and GPUs.

Choosing between these engines depends on the requirements of your workload. Use the matrix engine for bulk computation where speed is the priority and the vector engine when higher accuracy is needed.

To achieve maximum accuracy with the vector engine, several conditions must be met, from host-side configuration to kernel-side implementation.

Host-Side Configuration
-----------------------

On the host, the ``DeviceComputeKernelConfig`` struct controls the precision settings for compute kernels, including both the matrix engine (FPU), the vector engine (SFPU) and other components. To ensure the highest possible accuracy, enable the following two options:

* ``fp32_dest_acc_en = true``: This setting allocates 32-bit space in the Dst registers. This is required to store intermediate and final results at FP32 precision. If disabled (``false``, the default), the Dst registers will store 16-bit data, with FP32 values automatically converted to BFP16.
* ``math_approx_mode = false``: This disables optimizations that approximate certain math operations, ensuring that calculations are performed with maximum fidelity that the kernel library provides. By default, this is ``true``.

.. note::

    The ``math_fidelity`` setting in ``DeviceComputeKernelConfig`` only applies to the matrix engine. The vector engine always performs operations in 32-bit mode.

.. code-block:: cpp

    // On the host, configure the kernel for FP32 computation
    KernelHandle compute_kernel = CreateKernel(
        program,
        "/path/to/your/kernels/compute.cpp",
        core,
        DeviceComputeKernelConfig{
            .math_approx_mode = false,
            .fp32_dest_acc_en = true,
        }
    );

Additionally, ensure that the circular buffers that will handle the FP32 data are created with the ``DataFormat::Float32`` type.

.. note::

    Some functions, most notably ``exp_tile`` and the various trigonometric functions, have inherent limitations due to their polynomial approximations. Some functions have multiple available approximations (e.g. approx and fast_and_approx template parameters for exp_tile). These limitations can lead to reduced accuracy for certain input ranges, even when using the vector engine with FP32 settings. Always validate the accuracy of results for your specific use case. The operator implementations are built to balance performance and accuracy for the intended (machine learning) workloads. If your application requires higher precision across all input ranges, consider implementing custom functions.

Kernel-Side Implementation
--------------------------

Inside the compute kernel, you must use the vector engine (SFPU) for computations and correctly configure the unpacker and packer for FP32 data.

* **Configure Unpacker and Packer**: Before moving data, you must explicitly configure the unpacker and packer to handle the FP32 format.

    * Call ``copy_tile_init()`` before unpacking data from a circular buffer into the Dst registers. This function reconfigures the unpacker to correctly interpret the 32-bit data from the circular buffer.
    * Call ``pack_reconfig_data_format()`` before packing data from Dst registers to an output circular buffer. This ensures the packer formats the data correctly for the destination.

.. warning::

    If you are unpcking or packing to multiple circular buffers of different data formats, you must call ``copy_tile_init()`` and ``pack_reconfig_data_format()`` each time you switch between circular buffers with different formats. Otherwise the data may be misinterpreted, leading to incorrect results.

The following example demonstrates a typical compute kernel structure for achieving FP32 accuracy.

.. code-block:: cpp

    #include "compute_kernel_api/common.h"
    #include "compute_kernel_api/tile_move_copy.h"
    #include "compute_kernel_api/binary.h"

    namespace NAMESPACE {
    void MAIN {
        constexpr auto cb_in0 = tt::CBIndex::c_in0;
        constexpr auto cb_in1 = tt::CBIndex::c_in1;
        constexpr auto cb_out0 = tt::CBIndex::c_out0;
        constexpr uint32_t num_tiles = 8;

        // Initialize for a binary operation on the SFPU
        init_sfpu(cb_in0, cb_out0);
        add_binary_tile_init();

        for(uint32_t i = 0; i < num_tiles; i++) {
            // Wait for input data
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            // Acquire Dst registers
            tile_regs_acquire();

            // Configure unpacker for FP32 and copy data from CB to Dst
            copy_tile_init(cb_in0);
            copy_tile(cb_in0, 0, 0); // Copy tile from cb_in0 to Dst[0]

            copy_tile_init(cb_in1);
            copy_tile(cb_in1, 0, 1); // Copy tile from cb_in1 to Dst[1]

            // Perform computation on the SFPU
            add_binary_tile(0, 1, 0); // Dst[0] = Dst[0] + Dst[1]

            // Commit results and release Dst for the packer
            tile_regs_commit();

            // Reserve space in the output CB
            cb_reserve_back(cb_out0, 1);

            // Wait for packer to be ready
            tile_regs_wait();

            // Configure packer for FP32 and pack data from Dst to CB
            // This can be hoisted out of the loop as only one output
            // exists in the kernel
            pack_reconfig_data_format(cb_out0);
            pack_tile(0, cb_out0);

            // Release Dst registers
            tile_regs_release();

            // Announce data is available in output CB
            cb_push_back(cb_out0, 1);

            // Pop from input CBs
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }
    }
    } // NAMESPACE

.. warning::
    Failing to call ``copy_tile_init()`` and ``pack_reconfig_data_format()`` will result in data being treated as 16-bit, leading to a loss of precision, even if ``fp32_dest_acc_en`` is enabled.

Distinguishing Between matrix and vector engine APIs
----------------------------------------------------

A general way to distinguish between matrix engine (FPU) and vector engine (SFPU) APIs is by their parameters.

* **matrix engine APIs** typically take circular buffer indices as arguments, as the FPU operates directly on data unpacked from circular buffers into its dedicated ``SrcA`` and ``SrcB`` registers.
* **vector engine APIs** operate on data already present in the ``Dst`` registers. Therefore, their arguments are indices into the ``Dst`` register set.

For example:

.. code-block:: cpp

    // Adding tiles using the FPU
    // Operands are specified by their location in circular buffers.
    // Result is written to Dst tile 0.
    // DO NOT use if accuracy is of concern
    add_tiles(cb_in0, cb_in1, 0, 0, 0);

    // Adding tiles using the SFPU
    // Operands are specified by their location in Dst registers.
    // Result is written back to Dst tile 0.
    add_binary_tile(0, 1, 0);
