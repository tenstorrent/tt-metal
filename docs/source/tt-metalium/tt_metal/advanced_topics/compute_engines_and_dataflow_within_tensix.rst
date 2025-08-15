.. _compute_engines_and_dataflow_within_tensix:

Compute Engines and Data Flow within Tensix
===========================================

The matrix and vector engines (FPU and SFPU) are integral to Tensix's compute capabilities. They are designed to efficiently handle a wide range of mathematical operations, particularly those common in machine learning. However, they are also highly specialized and do not work like traditional attached-to-CPU vector and matrix units. Instead of having direct access to the CPU's registers and memory, these engines operate on their own registers and rely on the unpacker and packer to manage data movement between the CPU and the engines.

Understanding the data flow between the different register sets, (un)packer, and the engines is critical to achieving optimal performance and resource utilization. Please refer to the relevant hardware and ISA documentation for more details.

* `SFPU on Wormhole <https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/VectorUnit.md>`_
* `FPU on Wormhole <https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md>`_

Component introduction
----------------------

A critical design aspect to understand is that **all registers are viewed to have a fixed element count**. Each vector register holds exactly N elements, where N depends on the kernel configuration and the specific register set. This differs from x86 SSE/AVX or ARM NEON and is more similar to some GPU warp/wavefront designs. The size of the registers also may change between generations of hardware. Thus, a program that depends on the specific dimension of a register must be rewritten to work across generations. Compute APIs provided by the Metalium kernel library like ``matmul_tiles`` and ``sin_tiles`` abstract away these details, allowing you to focus on the algorithm rather than the underlying hardware specifics.

For example, the SFPU ``LReg`` (SFPU's vector register) on Wormhole is 32 elements wide, with each element being 32 bits. A conventional SIMD register design would imply a capacity of 1024 bits per register. Thus, if int8 data were loaded into such a register, it could hold 128 elements. However, the SFPU treats the register as holding 32 elements of at most 32 bits each, regardless of the actual data type.

Several key components work together to perform computations within Tensix:

* **Unpacker**: Reads data from L1 memory and converts it into the format required by the compute engines. Into registers ``SrcA``, ``SrcB`` or ``Dst``.
* **Packer**: Takes results produced by the compute engines (from the ``Dst`` registers), formats them, and moves them back into L1.
* **Vector engine (SFPU)**: Executes complex vector operations, handling data in parallel for efficient computation.
* **Matrix engine (FPU)**: Specializes in large-scale matrix operations, optimized for high throughput.

The compute engines rely on four main register sets:

1. **SrcA**: The first source register set for the matrix engine.
2. **SrcB**: The second source register set for the matrix engine.
3. **Dst**: The destination register set for the matrix engine, also used by the vector engine to hold results. This register set is exposed in the higher-level API.
4. **LReg**: Internal registers within the SFPU for holding vector data during computation.

The following image illustrates the connection between the different components and the registers they have access to.

.. note::

    It is important to keep in mind that, although the compute kernel is a single piece of code, it is actually split and compiled into three separate binaries, each running on a different RISC-V (T0-2) core within the Tensix. Synchronization is needed to ensure correct data flow and to avoid race conditions between these components.

    Also, each of the components - including the unpacker, packer, SFPU, and FPU - is not a processing core and cannot make control flow decisions on its own. The RISC-V cores issue commands to the compute engines and manage data flow between them. Explicit synchronization may be needed to ensure the engines have finished their work before the RISC-V cores continue.

.. figure:: /images/tenstorrent-sfpu-fpu-dst-register-diagram-and-dataflow.webp
    :scale: 45%
    :alt: SFPU and FPU register diagram and data flow
    :align: center

    The connection between the unpacker, packer, SFPU, FPU, and the various registers is crucial for efficient data processing within the Tensix architecture.

Register data formats often differ from the SRAM storage format, so the unpacker and packer handle format conversion. This conversion enables efficient block floating-point computation since the matrix and vector engines support standard floating-point and integer arithmetic. This design allows a single kernel to operate on different data formats without modification.

The unpacker/packer pair provides hardware-accelerated type casting. Rather than using compute engines to expand block floating-point data (such as quantized model weights) into individual floating-point values before computation, the unpacker performs this expansion directly in hardware. This approach reduces both execution time and power consumption compared to software-based conversion on traditional processors.

Due to the separation of concerns, invoking the matrix and vector engine often comes with an initialization call to set up the pack and unpacker and configure the width of the ``Dst`` registers - in order to ensure the correctness of the result and performance optimizations.

Dst register
------------

The ``Dst`` registers are the only register set that is exposed and visible to the users through public APIs. It can either be viewed as holding 16 tiles of 16-bit data or 8 tiles of 32-bit data. The active mode used by the kernel is controlled by the ``fp32_dest_acc_en`` field in ``ComputeKernelConfig`` when setting up the kernel on the host side.

.. code-block:: c++

    auto kernel_id = tt::tt_metal::CreateKernel(
        program,
        "path/to/your/compute/kernel.cpp",
        core,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = true      // Dst is viewed as eight tiles of 32-bit data
            // fp32_dest_acc_en = false   // Dst is viewed as sixteen tiles of 16-bit data
        }
    );

Note that ``fp32_dest_acc_en`` being enabled DOES NOT guarantee that all computations are performed at 32-bit accuracy. It only means that the ``Dst`` registers will have 32 bits of storage per element for results. For example, the matrix engine may still compute and output in bfloat16, but the results are stored in 32-bit slots. If further processing is done by the vector engine, the final result will be a 32-bit value representing the processed bfloat16 data.

As the compute kernel is expected to run on all 3 cores at the same time, extra care is needed to avoid racing between the cores and users of the ``Dst`` register. Metalium provides mechanisms to ensure safe access to the ``Dst`` registers through 4 functions:

.. code-block:: c++

    // Wait for input to be available via circular buffers (e.g., using cb_wait_front).
    // ...

    // Acquire Dst registers, making them available for unpacking and math operations.
    tile_regs_acquire();

    // Perform unpacking and math operations here.
    // e.g., copy_tile(...), matmul_tiles(...), add_tiles(...)
    // ...

    // Commit the results, transferring ownership of Dst registers to the packer.
    tile_regs_commit();

    // Release input CBs and reserve space in output CBs.
    // e.g., cb_pop_front(...), cb_reserve_back(...)
    // ...

    // Wait until the packer can safely access the Dst registers.
    tile_regs_wait();

    // Perform packing operations.
    // e.g., pack_tile(...)
    // ...

    // Release the Dst registers, as they are no longer needed for this iteration.
    tile_regs_release();

    // Announce that data has been written to the output CBs.
    // e.g., cb_push_back(...)
    // ...

.. note::

    The ordering of circular buffer operations (``cb_wait_front``, ``cb_pop_front``, ``cb_reserve_back``, ``cb_push_back``) can be adjusted, but their placement is constrained by data dependencies. The pattern shown in the example is designed to minimize stalls by overlapping waiting for space and computation by different threads. Note that unpacking from a circular buffer can only occur after the ``Dst`` registers have been acquired, and packing can only begin after the packer is ready to access them.

    The ``acquire_dst`` and ``release_dst`` functions also control access to the ``Dst`` registers but are now deprecated. They offer coarse-grained control and have been superseded by the more explicit ``tile_regs_*`` family of functions, which should be used instead.

.. warning::

    Even when no packing is performed, ``tile_regs_commit`` and ``tile_regs_release`` must still be called in sequence. Failure to do so results in undefined behavior.

The above construction is often used in a loop to perform consecutive computation. Moving data from and to the ``Dst`` registers is performed using the ``copy_tile`` and ``pack_tile`` functions.


.. code-block:: c++

    // Before calling copy_tile, ensure the circular buffer contains data (use cb_wait_front).
    // copy_tile transfers a tile from the circular buffer at the specified tile_offset_in_cb
    // into the destination tile at the given index.
    copy_tile(CBIndex::c_0, /*tile_offset_in_cb*/0, /*dst_idx*/0);

    // pack_tile performs the opposite operation, transferring a tile from the Dst register
    // to the circular buffer at the specified tile_offset_in_cb.
    pack_tile(/*dst_idx*/0, CBIndex::c_16, /*tile_offset_in_cb*/0);

Matrix engine/FPU
-----------------

The matrix engine, or the FPU, performs the bulk of computation for most AI and machine learning workloads. Operations on the matrix engine take in data from ``SrcA`` and ``SrcB`` (if needed) and output or write back or even accumulate results to ``Dst``. The FPU also supports common matrix operations such as element-wise multiplication/addition/subtraction and pooling.

FPU operations require initialization before execution. This setup configures the unpacker, packer, and FPU for the specific operation (e.g., matrix multiplication). Re-initialization is not required for repeated operations with the same source, destination, and data type parameters.

The FPU has dedicated registers for each operand, and the unpacker can write directly to these registers. The API lets you specify the circular buffer index and tile offset for each operand. Since the FPU writes results to the ``Dst`` registers, you can also specify the output tile offset. It is up to the user to avoid register conflicts. Compute functions using the FPU take the following parameters, depending on the number of operands:

* Circular buffer index for the first operand and tile offset from the buffer's read head
* (If applicable) Circular buffer index for the second operand and tile offset from the buffer's read head
* Offset (in number of tiles) to write the result to the ``Dst`` registers

For example, to perform matrix multiplication pairwise:

.. code-block:: c++

    // Configure (un)packer and FPU into matmul mode
    //      cb_in0        cb_in1        cb_out
    mm_init(CBIndex::c_0, CBIndex::c_1, CBIndex::c_16);

    // Repeated computation can be performed without re-initialization
    for(int i=0;i<8;i++) {
        // Wait for data to be available in the input circular buffers
        cb_wait_front(CBIndex::c_0, 1); cb_wait_front(CBIndex::c_1, 1);

        // Make sure dst registers are available for the math core
        tile_regs_acquire();

        // Perform matrix multiplication by taking tile 0 from CB 0, tile 0 from CB 1
        // and put into Dst tile 0.
        //              cb_in0     cb_in1        in0_offset  in1_offset  dst_idx   transp
        matmul_tiles(CBIndex::c_0, CBIndex::c_1, 0         , 0         , 0      , false);

        // We are done doing math. Transfer ownership of the dst registers to the unpacker
        tile_regs_commit();

        // We are not using the tile in the input CBs anymore
        cb_pop_front(CBIndex::c_0, 1); cb_pop_front(CBIndex::c_1, 1);
        // Wait for space in the output circular buffer
        cb_reserve_back(CBIndex::c_16, 1);

        // Now we can start packing the output
        tile_regs_wait();

        // Copy tile from dst tile 0 into the output CB. This 0 is the same as
        // the Dst tile index used in matmul_tiles
        pack_tile(/*src_dst_idx*/0, CBIndex::c_16, /*tile_offset_in_cb*/0);
        // We have written the data to CB. Announce it to be done
        cb_push_back(CBIndex::c_16, 1);

        // Unpacker is done with the dst registers. Release for the next round
        tile_regs_release();
    }

.. warning::
    Note that the same input circular buffers (``cb_in0`` and ``cb_in1``) must be specified in both ``mm_init`` and ``matmul_tiles``. Using different circular buffers between these calls results in undefined behavior, as the unpacker may be interpreting the data differently or reading into invalid/undefined memory.

The information to configure the unpacker and packer is taken from the circular buffer metadata. In the above example, circular buffer 0 and 1 are used to configure the unpacker to unpack their data into ``SrcA`` and ``SrcB``. And the packer is configured to pack into the format of what circular buffer 16 is expecting.

Vector engine/SFPU
------------------

The vector engine, or the SFPU, is designed for high-throughput processing of vector data. Unlike APIs using the matrix engine, APIs using the vector engine ask the user to explicitly unpack data onto the ``Dst`` registers before performing computations and packing the results back into L1 memory. This design enables easier chaining of operations.

An initialization phase is also required. A generic ``init_sfpu`` is needed to configure the unpacker and packer to consume and produce data in the type the input and output circular buffer needed. Due to hardware limitations, there is no support for setting up the unpacker for the second operand like operations using the FPU do. Like the matrix engine, if parameters are duplicated between the initialization and computation calls, they must be the same. Otherwise, it may lead to undefined behavior.

For example, to compute the sine of a tile (duplicated comments from the above example are ignored):

.. code-block:: c++

    // Configure the (un)packer to expect data formats held by the CBs
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    add_binary_tile_init();

    for(int i=0;i<8;i++) {
        cb_wait_front(CBIndex::c_0, 1); cb_wait_front(CBIndex::c_1, 1);
        tile_regs_acquire();

        // Unpack the first tile from the CB into the first tile in DST
        // This function involves both the unpacker and math core to ensure
        // synchronization
        copy_tile(CBIndex::c_0, /*tile_offset_in_cb*/0, /*dst_idx*/0);
        // DITTO but into the second tile in Dst
        copy_tile(CBIndex::c_1, /*tile_offset_in_cb*/0, /*dst_idx*/1);

        // Add tile 0 and 1 in the dst registers together. Store result back
        // into (the first argument) tile 0. Pseudo code:
        // dst_tile[0] = dst_tile[0] + dst_tile[1]
        add_binary_tile(/*dst_idx_a*/0, /*dst_idx_b*/1);
        // More operations can be chained and performed, if desired. e.g.,
        // applying sigmoid.
        // Applies sigmoid on dst register tile 0 and writes to dst register tile 0
        // sigmoid_tile(0);

        tile_regs_commit();
        cb_pop_front(CBIndex::c_0, 1); cb_pop_front(CBIndex::c_1, 1);
        cb_reserve_back(CBIndex::c_16, 1);
        tile_regs_wait();
        pack_tile(/*dst_idx*/0, CBIndex::c_16, /*tile_offset_in_cb*/0);
        cb_push_back(CBIndex::c_16, 1);

        tile_regs_release();
    }

.. note::
    ``copy_tile_init`` can be used to re-configure the unpacker to consume different data formats from circular buffers. If ``CBIndex::c_0`` and ``CBIndex::c_1`` contain different data types, the unpacking part of the above example can be rewritten to the following:

    .. code-block:: c++

        copy_tile_init(CBIndex::c_0);
        copy_tile(CBIndex::c_0, /*tile_offset_in_cb*/0, /*dst_offset_tiles*/0);
        copy_tile_init(CBIndex::c_1);
        copy_tile(CBIndex::c_1, /*tile_offset_in_cb*/0, /*dst_offset_tiles*/1);

    Also note that ``copy_tile_init`` is always needed if you are unpacking FP32 values into 32-bit ``Dst`` registers. As ``init_sfpu`` assumes a 16-bit storage size and sets up the unpacker to unpack as bfloat16. Some accuracy will be lost if an explicit extra initialization is not done.

    Similarly, the ``pack_reconfig_data_format`` function and its variants are used to change the packer's output data format. This is necessary when a computation produces multiple tiles that must be written to circular buffers with different data formats. For example, to pack two tiles into two separate circular buffers, each with a unique data format:

    .. code-block:: c++

        pack_reconfig_data_format(CBIndex::c_16);
        pack_tile(/*src_idx*/0, CBIndex::c_16, /*tile_offset_in_cb*/0);
        pack_reconfig_data_format(CBIndex::c_17);
        pack_tile(/*src_idx*/1, CBIndex::c_17, /*tile_offset_in_cb*/0);

After data is unpacked into the ``Dst`` registers, the vector engine can load data from ``Dst`` into ``LReg`` directly, without involving other hardware blocks. For more details on programming the SFPU, see the :ref:`Low Level Kernels programming guide <llk>`. The ``dst_reg`` variable provides an ``LReg``-sized view into the ``Dst`` registers. For example, on Wormhole and Blackhole, ``LReg`` is 32 elements wide, so the first ``Dst`` tile corresponds to ``dst_reg[0:31]``. To illustrate:

.. code-block:: c++

    void sfpu_example_function() {
        vFloat vec1 = dst_reg[0]; // Load the first 32 elements of the 1st tile into LReg
        vFloat vec2 = dst_reg[32]; // Load the first 32 elements of the 2nd tile into LReg

        dst_reg[0] = vec1; // Store the result back into the 1st tile
        dst_reg[32] = vec2; // Store the result back into the 2nd tile
    }

Due to the :ref:`internal structure of tiles<internal_structure_of_a_tile>`, typically ``dst_reg[0:3]`` contains the first face of the tile. Similarly, ``dst_reg[4:7]`` contains the second face, and so on.
