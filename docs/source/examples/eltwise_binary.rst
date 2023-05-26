.. _Eltwise binary example:

Eltwise binary
==============

We now build a program that will perform eltwise binary operations on a some
equal-sized tensors.

We'll go through any new code section by section. This builds on top of
previous examples. Note that we have this exact, full example program in
``programming_examples/loopback/eltwise_binary.cpp``, so you can follow along.

New buffers
-----------

In terms of DRAM buffers, We just need a new buffer for a 2nd source, because
we have two source tensors (vectors).

However, we need to declare some circular buffers to enable data transfer
between the reader, compute, and writer engines.

.. code-block:: cpp

  constexpr uint32_t src0_cb_index = CB::c_in0;
  constexpr uint32_t src0_cb_addr = 200 * 1024;
  constexpr uint32_t num_input_tiles = 2;
  CircularBuffer *cb_src0 = CreateCircularBuffer(
      program,
      device,
      src0_cb_index,
      core,
      num_input_tiles,
      num_input_tiles * single_tile_size,
      src0_cb_addr,
      tt::DataFormat::Float16_b
  );

  constexpr uint32_t src1_cb_index = CB::c_in1;
  constexpr uint32_t src1_cb_addr = 300 * 1024;
  CircularBuffer *cb_src1 = CreateCircularBuffer(
      program,
      device,
      src1_cb_index,
      core,
      num_input_tiles,
      num_input_tiles * single_tile_size,
      src1_cb_addr,
      tt::DataFormat::Float16_b
  );

  constexpr uint32_t output_cb_index = CB::c_out0;
  constexpr uint32_t output_cb_addr = 400 * 1024;
  constexpr uint32_t num_output_tiles = 2;
  CircularBuffer *cb_output = CreateCircularBuffer(
      program,
      device,
      output_cb_index,
      core,
      num_output_tiles,
      num_output_tiles * single_tile_size,
      output_cb_addr,
      tt::DataFormat::Float16_b
  );

We will create two input circular buffers to accommodate our two input tensors,
and an output one for the result of the eltwise binary operation.

Compile-time compute kernel arguments
-------------------------------------

.. code-block:: cpp

  std::vector<uint32_t> eltwise_binary_compile_args = {
      /*.per_core_block_cnt =*/ 2048,
      /*.per_core_block_size =*/ 1
  };

We have to declare some compile-time arguments for compute kernel. Some default
parameters here will suffice.

These two parameters essentially tell the kernel how much data we'll be moving
in one invocation. A high number for total block count like that specified here
is sufficient.

Compute kernel declaration and compile-time defines
---------------------------------------------------

.. code-block:: cpp

  ComputeKernel *eltwise_binary_kernel = CreateComputeKernel(
      program,
      "kernels/compute/eltwise_binary.cpp",
      core,
      eltwise_binary_args,
      MathFidelity::HiFi4,
      fp32_dest_acc_en,
      math_approx_mode
  );
  eltwise_binary_kernel->add_define("ELTWISE_OP", "add_tiles");

We will declare what kind of compute kernel we're using and further specify we
want to use the ``add_tiles`` eltwise binary op, for eltwise adding.

Extra runtime arguments and source tensor
-----------------------------------------

.. code-block:: cpp

        constexpr float val_to_add = -1.0f;
        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, val_to_add);

        WriteToBuffer(src1_dram_buffer, src1_vec);

In this program, we have a second source tensor. We will be adding this to the
first source tensor.

.. code-block:: cpp

        WriteRuntimeArgsToDevice(
          device,
          unary_writer_kernel,
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

Those are the additional steps for getting eltwise binary operations up and
running on the compute engine. We essentially repeat the same process to chain
together two operations, with one DRAM read in the middle to get the
intermediate result and hold it in a DRAM buffer.
