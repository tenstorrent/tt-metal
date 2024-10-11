.. _Eltwise sfpu example:

Eltwise SFPU
============

We now build a program that will perform an eltwise SFPU unary operation on a
single tensor.

We'll go through any new code section by section. This builds on top of
previous examples. Note that we have this exact, full example program in
``tt_metal/programming_examples/eltwise_sfpu/eltwise_sfpu.cpp``, so you can
follow along.

To build and execute, you may use the following commands. Note that we include
the necessary environment variables here, but you may possibly need more
depending on the most up-to-date installation methods.

::

    export ARCH_NAME=<arch name>
    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh
    ./build/programming_examples/eltwise_sfpu

Circular buffers for data movement to/from compute engine
---------------------------------------------------------

The number of buffers we're using in DRAM will stay the same. However, we need
to declare some circular buffers to enable data transfer between the reader,
compute, and writer engines.

.. code-block:: cpp

    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::v0::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}}).set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = tt_metal::v0::CreateCircularBuffer(program, core, cb_output_config);

We will create one input circular buffers to accommodate our input tensor,
and an output one for the result of the eltwise sfpu operation.

Compile-time compute kernel arguments
-------------------------------------

.. code-block:: cpp

    std::vector<uint32_t> compute_kernel_args = {
        num_tiles,
        1
    };

We have to declare some compile-time arguments for compute kernel. Some default
parameters here will suffice.

These two parameters essentially tell the kernel how much data we'll be moving
in one invocation.

Compute kernel declaration and compile-time defines
---------------------------------------------------

.. code-block:: cpp

    const std::map<std::string, std::string> sfpu_defines = {
        {"SFPU_OP_EXP_INCLUDE", "1"},
        {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}
    };

    KernelHandle eltwise_sfpu_kernel_id = CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        core,
        ComputeConfig{
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = sfpu_defines,
        }
    );

We will declare what kind of compute kernel we're using.

For the eltwise SFPU compute kernel specifically, we need to use defines to
control what kind of op we're using. In this case, we need to use
``SFPU_OP_EXP_INCLUDE`` to get the exponential kernel headers included into the
kernel C++ kernel files and ``SFPU_OP_CHAIN_0`` to declare which device compute
API functions to use.

Extra runtime arguments for reader/writer
-----------------------------------------

.. code-block:: cpp

    SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            dst_dram_buffer.address(),
            static_cast<uint32_t>(dst_dram_buffer.noc_coordinates().x),
            static_cast<uint32_t>(dst_dram_buffer.noc_coordinates().y),
            num_tiles
        }
    );

In this program,  we're using a separate reader kernel to take in data from
DRAM into L1, and a separate writer kernel to write out results from the
compute engine back to the destination DRAM buffer.

That means two sets of runtime arguments for data movement kernels. In the DRAM
loopback example, we only had a single data movement kernel.

Conclusion
----------

Those are the additional steps for getting eltwise sfpu operations up and
running on the compute engine. For some complicated compute, please refer to the
:ref:`Eltwise binary example<Eltwise binary example>`.
