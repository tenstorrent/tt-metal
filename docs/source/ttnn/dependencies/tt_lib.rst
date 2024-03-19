.. _TT-LIB:

TT-LIB
######

Overview
***********

The ``tt_lib`` Python module is a
unified Python interface to the Tensor library located within ``tt_eager``. This library currently only supports 4 dimensional tensors with shape ``[W, Z, Y, X]``, in ROW_MAJOR layout, and with BFLOAT16 data type.

Some OPs in this library might change layout of input tensors and pad them to better match expectations of execution kernels on TT Accelerator device.
These OPs will unpad the result tensor before it is returned to caller.

There is a limitation that tensor in ROW_MAJOR layout on TT Accelerator device must have the size of last dimension ``X`` be divisible by 2.
You can't create these type of tensors on TT Accelerator device or send them to TT Accelerator device with ```tt_lib.tensor.Tensor.to()``.
However, you can supply these type of tensors to OPs from TT-LIB library as they can automatically pad the last dimension before moving the tensor
to TT Accelerator device. To use this functionality, you must call `tt_lib.device.SetDefaultDevice(tt_device)` to set your TT Accelerator device
as the default device that will be used to execute operations on tensors that are on host machine.

Operation Infrastructure
========================

TT-LIB has operation infrastructure which is used to launch, profile and cache operations generically.

To add a new operation that can plug in to the infrastructure, all that's needed is a struct that implements methods needed by operation interface.
Below, is an example of how to declare a new on-device operation with all of the methods required by the interface.


New Device Operation
--------------------

.. code-block:: cpp

    struct <NewOperation> {
        void validate(const std::vector<Tensor> &input_tensors) const;
        std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
        std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
        operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

        static constexpr auto attribute_names = std::make_tuple();
        const auto attribute_values() const {
            return std::make_tuple();
        }
    };

New Device Operation with a member
----------------------------------

.. code-block:: cpp

    struct <NewOperation> {
        int some_member

        void validate(const std::vector<Tensor> &input_tensors) const;
        std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
        std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
        operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

        static constexpr auto attribute_names = std::make_tuple("some_member");
        const auto attribute_values() const {
            return std::make_tuple(std::cref(some_member));
        }
    };

New Device Operation with Optional Input Tensors
------------------------------------------------

.. code-block:: cpp

    struct <NewOperation> {
        void validate(const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
        std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
        std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
        operation::ProgramWithCallbacks create_program(
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            std::vector<Tensor> &output_tensors) const;

        static constexpr auto attribute_names = std::make_tuple();
        const auto attribute_values() const {
            return std::make_tuple();
        }
    };

New Device Operation with Optional Output Tensors
-------------------------------------------------

If an operation is expected to leverage optional output tensors, please use instead the validate_with_output_tensors
and create_output_tensors with the additional parameter for the output_tensors.

.. code-block:: cpp

    struct <NewOperation> {
        void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
        std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
        std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
        operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

        static constexpr auto attribute_names = std::make_tuple();
        const auto attribute_values() const {
            return std::make_tuple();
        }
    };

New Host Operation
------------------

And below, is an example of how to declare a new on-host operation with all of the methods required by the interface.

.. code-block::

    struct <NewOperation> {
        void validate(const std::vector<Tensor> &input_tensors) const;
        std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
        std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const;

        static constexpr auto attribute_names = std::make_tuple();
        const auto attribute_values() const {
            return std::make_tuple();
        }
    };

Profiler
========

Profiler is supported out of the box for any op.

And there are 2 special methods that can be optionally implemented to set the preferred_name and parallelization_strategy.

.. code-block::

    // Implement `get_parallelization_strategy` to set the parallelization strategy on the profiler
    struct <NewOperation> {
        <ParallelizationStrategyEnum> get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    };

Fast Dispatch
=============

Fast dispatch allows programs/kernels to be enqueued to run, so host code does not have to wait for ops/programs to finish running.
The enqueued programs run asynchronously to the host code.
To wait for kernels to complete, either read a tensor from device to host with:

.. autofunction:: tt_lib.tensor.Tensor.cpu

or to perform only a wait, use:

.. autofunction:: tt_lib.device.Synchronize


Program Caching
===============

Program caching provides an ability for an operation to cache the program and simply reload it the next time the same operation is used.

It can be enabled by running:

.. code-block::

    tt::tt_metal::program_cache::enable()

And it can be disabled by running:

.. code-block::

    tt::tt_metal::program_cache::disable_and_clear()

Number of entries can be queried using:

.. code-block::

    tt::tt_metal::program_cache::num_entries()

In order for an op to be cachable, it needs to implement the following:

.. code-block::

    struct <NewOperation> {
       // Mandatory methods

        // Return type of `create_program` needs to implement override_runtime_args_callback
        // i.e.:
        operation::ProgramWithCallbacks create_program(const std::vector<Tensor> &input_tensors) const {

            Program program{};

            // ...

            auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
                const Program &program,
                const std::vector<Buffer*>& input_buffers,
                const std::vector<Buffer*>& output_buffers
            ) {

                auto src_dram_buffer = input_buffers.at(0);
                auto dst_dram_buffer = output_buffers.at(0);

                CoreCoord core = {0, 0};

                {
                    auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                    runtime_args[0] = src_dram_buffer->address();
                }

                {
                    auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                    runtime_args[0] = dst_dram_buffer->address();
                }
            };

            return {std::move(program), override_runtime_args_callback};
        }
    };

Logs
====
To see logs related to operation infrastructure, use the following environment variables:

.. code-block::

    export TT_METAL_LOGGER_TYPES=Op
    export TT_METAL_LOGGER_LEVEL=Debug

The logs will print currently running op and information related to program caching. i.e.:

.. code-block::

    Op | DEBUG    | Operation Type: silu (fallback operation)
    Op | DEBUG    | Operation Attributes: ()
    Op | DEBUG    | Input Tensors: {tt::tt_metal::Tensor(storage=tt::tt_metal::DeviceStorage(memory_config=tt::tt_metal::MemoryConfig(memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED, buffer_type=tt::tt_metal::BufferType::DRAM)), shape={1, 1, 1, 1280}, dtype=tt::tt_metal::DataType::BFLOAT16, layout=tt::tt_metal::Layout::ROW_MAJOR)}
    Op | DEBUG    | Operation Type: tt::tt_metal::LayoutConversionOnHost
    Op | DEBUG    | Operation Attributes: (target_layout=tt::tt_metal::Layout::TILE)
    Op | DEBUG    | Input Tensors: {tt::tt_metal::Tensor(storage=tt::tt_metal::OwnedStorage(), shape={1, 1, 320, 1280}, dtype=tt::tt_metal::DataType::BFLOAT16, layout=tt::tt_metal::Layout::ROW_MAJOR)}
    ...
    Op | DEBUG    | Program Cache: MISS - Compiling new program "tt::tt_metal::EltwiseUnary(op_type=tt::tt_metal::UnaryOpType::Enum::GELU, param=1)_tt::tt_metal::Tensor(storage=tt::tt_metal::DeviceStorage(memory_config=tt::tt_metal::MemoryConfig(memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED, buffer_type=tt::tt_metal::BufferType::DRAM)), shape={1, 1, 32, 32}, dtype=tt::tt_metal::DataType::BFLOAT16, layout=tt::tt_metal::Layout::TILE)"
    Op | DEBUG    | Operation Name: tt::tt_metal::EltwiseUnary
    Op | DEBUG    | Operation Attributes: (op_type=tt::tt_metal::UnaryOpType::Enum::GELU, param=0)
    Op | DEBUG    | Input Tensors: {tt::tt_metal::Tensor(storage=tt::tt_metal::DeviceStorage(memory_config=tt::tt_metal::MemoryConfig(memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED, buffer_type=tt::tt_metal::BufferType::DRAM)), shape={1, 1, 32, 32}, dtype=tt::tt_metal::DataType::BFLOAT16, layout=tt::tt_metal::Layout::TILE)}


If `OPERATION_HISTORY_CSV=<csv_file_path>` environment variable is set, then the history of all executed operations will be dumped into `<csv_file_path>`


TT-LIB API through ``tt_lib``
*****************************

Primary Operations
==================

autofunction:: tt_lib.operations.primary.matmul

.. autofunction:: tt_lib.operations.primary.layernorm

.. autofunction:: tt_lib.operations.primary.add_layernorm

.. autofunction:: tt_lib.operations.primary.softmax_in_place

.. autofunction:: tt_lib.operations.primary.moreh_softmax

.. autofunction:: tt_lib.operations.primary.moreh_softmax_backward

.. autofunction:: tt_lib.operations.primary.moreh_softmin

.. autofunction:: tt_lib.operations.primary.moreh_softmin_backward

.. autofunction:: tt_lib.operations.primary.moreh_logsoftmax

.. autofunction:: tt_lib.operations.primary.moreh_logsoftmax_backward

.. autofunction:: tt_lib.operations.primary.transformers.scale_mask_softmax_in_place

.. autofunction:: tt_lib.operations.primary.moreh_mean

.. autofunction:: tt_lib.operations.primary.moreh_mean_backward

.. autofunction:: tt_lib.operations.primary.moreh_groupnorm

.. autofunction:: tt_lib.operations.primary.moreh_groupnorm_backward

.. autofunction:: tt_lib.operations.primary.moreh_norm

.. autofunction:: tt_lib.operations.primary.moreh_norm_backward

Enums
=====

.. autoclass:: tt_lib.tensor.BcastOpMath

.. autoclass:: tt_lib.tensor.BcastOpDim

.. autoclass:: tt_lib.tensor.ReduceOpMath

.. autoclass:: tt_lib.tensor.ReduceOpDim

Tensor elementwise operations
=============================

.. autofunction:: tt_lib.tensor.add

.. autofunction:: tt_lib.tensor.sub

.. autofunction:: tt_lib.tensor.mul

.. autofunction:: tt_lib.tensor.add_unary

.. autofunction:: tt_lib.tensor.sub_unary

.. autofunction:: tt_lib.tensor.mul_unary

.. autofunction:: tt_lib.tensor.div_unary

.. autofunction:: tt_lib.tensor.gelu

.. autofunction:: tt_lib.tensor.relu

.. autofunction:: tt_lib.tensor.relu6

.. autofunction:: tt_lib.tensor.relu_min

.. autofunction:: tt_lib.tensor.relu_max

.. autofunction:: tt_lib.tensor.exp

.. autofunction:: tt_lib.tensor.recip

.. autofunction:: tt_lib.tensor.sqrt

.. autofunction:: tt_lib.tensor.log

.. autofunction:: tt_lib.tensor.log2

.. autofunction:: tt_lib.tensor.log10

.. autofunction:: tt_lib.tensor.log1p

.. autofunction:: tt_lib.tensor.tanh

.. autofunction:: tt_lib.tensor.clip

.. autofunction:: tt_lib.tensor.hardtanh

.. autofunction:: tt_lib.tensor.deg2rad

.. autofunction:: tt_lib.tensor.rad2deg

.. autofunction:: tt_lib.tensor.cbrt

.. autofunction:: tt_lib.tensor.hypot

.. autofunction:: tt_lib.tensor.softplus

.. autofunction:: tt_lib.tensor.mish

.. autofunction:: tt_lib.tensor.polyval

.. autofunction:: tt_lib.tensor.sign

.. autofunction:: tt_lib.tensor.abs

.. autofunction:: tt_lib.tensor.silu

.. autofunction:: tt_lib.tensor.square

.. autofunction:: tt_lib.tensor.neg

.. autofunction:: tt_lib.tensor.add1

.. autofunction:: tt_lib.tensor.mac

.. autofunction:: tt_lib.tensor.sigmoid

.. autofunction:: tt_lib.tensor.hardsigmoid

.. autofunction:: tt_lib.tensor.swish

.. autofunction:: tt_lib.tensor.hardswish

.. autofunction:: tt_lib.tensor.leaky_relu

.. autofunction:: tt_lib.tensor.softsign

.. autofunction:: tt_lib.tensor.softshrink

.. autofunction:: tt_lib.tensor.hardshrink

.. autofunction:: tt_lib.tensor.cos

.. autofunction:: tt_lib.tensor.sin

.. autofunction:: tt_lib.tensor.cosh

.. autofunction:: tt_lib.tensor.sinh

.. autofunction:: tt_lib.tensor.acos

.. autofunction:: tt_lib.tensor.asin

.. autofunction:: tt_lib.tensor.squared_difference

.. autofunction:: tt_lib.tensor.elu

.. autofunction:: tt_lib.tensor.exp2

.. autofunction:: tt_lib.tensor.tanhshrink

.. autofunction:: tt_lib.tensor.heaviside

.. autofunction:: tt_lib.tensor.logaddexp

.. autofunction:: tt_lib.tensor.logaddexp2

.. autofunction:: tt_lib.tensor.atan

.. autofunction:: tt_lib.tensor.atanh

.. autofunction:: tt_lib.tensor.atan2

.. autofunction:: tt_lib.tensor.logical_xor

.. autofunction:: tt_lib.tensor.logical_xori

.. autofunction:: tt_lib.tensor.logical_not_unary

.. autofunction:: tt_lib.tensor.subalpha

.. autofunction:: tt_lib.tensor.addalpha

.. autofunction:: tt_lib.tensor.ldexp

.. autofunction:: tt_lib.tensor.bias_gelu

.. autofunction:: tt_lib.tensor.bias_gelu_unary

.. autofunction:: tt_lib.tensor.isfinite

.. autofunction:: tt_lib.tensor.isinf

.. autofunction:: tt_lib.tensor.isposinf

.. autofunction:: tt_lib.tensor.isneginf

.. autofunction:: tt_lib.tensor.isnan

.. autofunction:: tt_lib.tensor.logit

.. autofunction:: tt_lib.tensor.lgamma

.. autofunction:: tt_lib.tensor.logical_and

.. autofunction:: tt_lib.tensor.logical_andi

.. autofunction:: tt_lib.tensor.erfinv

.. autofunction:: tt_lib.tensor.multigammaln

.. autofunction:: tt_lib.tensor.assign

.. autofunction:: tt_lib.tensor.isclose

.. autofunction:: tt_lib.tensor.i0

.. autofunction:: tt_lib.tensor.digamma

.. autofunction:: tt_lib.tensor.tan

.. autofunction:: tt_lib.tensor.logical_or

.. autofunction:: tt_lib.tensor.logical_ori

.. autofunction:: tt_lib.tensor.polygamma

Tensor relational operations
============================
.. autofunction:: tt_lib.tensor.gtz

.. autofunction:: tt_lib.tensor.gez

.. autofunction:: tt_lib.tensor.ltz

.. autofunction:: tt_lib.tensor.lez

.. autofunction:: tt_lib.tensor.eqz

.. autofunction:: tt_lib.tensor.nez

.. autofunction:: tt_lib.tensor.gt

.. autofunction:: tt_lib.tensor.gte

.. autofunction:: tt_lib.tensor.lt

.. autofunction:: tt_lib.tensor.lte

.. autofunction:: tt_lib.tensor.eq

.. autofunction:: tt_lib.tensor.ne

Tensor ternary operations
=========================
.. autofunction:: tt_lib.tensor.where

.. autofunction:: tt_lib.tensor.threshold

Tensor matrix math operations
=============================

.. autofunction:: tt_lib.tensor.matmul

.. autofunction:: tt_lib.tensor.bmm

Tensor manipulation operations
-=============================

These operations change the tensor shape in some way, giving it new dimensions
but in general retaining the data.

.. autofunction:: tt_lib.tensor.reshape

.. autofunction:: tt_lib.tensor.transpose

.. autofunction:: tt_lib.tensor.permute

.. autofunction:: tt_lib.tensor.tilize

.. autofunction:: tt_lib.tensor.untilize

.. autofunction:: tt_lib.tensor.tilize_with_val_padding

.. autofunction:: tt_lib.tensor.untilize_with_unpadding

.. autofunction:: tt_lib.tensor.tilize_with_zero_padding

.. autofunction:: tt_lib.tensor.pad

.. autofunction:: tt_lib.tensor.unpad

.. autofunction:: tt_lib.tensor.clone

.. autofunction:: tt_lib.tensor.typecast

.. autofunction:: tt_lib.tensor.copy

Tensor creation operations
==========================

.. autofunction:: tt_lib.tensor.arange

.. autofunction:: tt_lib.tensor.full

.. autofunction:: tt_lib.tensor.ones

.. autofunction:: tt_lib.tensor.ones_like

.. autofunction:: tt_lib.tensor.zeros

.. autofunction:: tt_lib.tensor.zeros_like

.. autofunction:: tt_lib.tensor.full_like

.. autofunction:: tt_lib.tensor.split_last_dim_two_chunks_tiled

.. autofunction:: tt_lib.tensor.empty

.. autofunction:: tt_lib.tensor.tril

.. autofunction:: tt_lib.tensor.triu

Broadcast and Reduce
====================

.. autofunction:: tt_lib.tensor.bcast

.. autofunction:: tt_lib.tensor.reduce

.. autofunction:: tt_lib.tensor.global_min

.. autofunction:: tt_lib.tensor.global_max

.. autofunction:: tt_lib.tensor.global_sum

.. autofunction:: tt_lib.tensor.global_mean

.. autofunction:: tt_lib.tensor.rpow

.. autofunction:: tt_lib.tensor.rsub

.. autofunction:: tt_lib.tensor.rdiv

Fallback Operations
*******************

These operations are currently not supported on TT accelerator device and will execute on host machine using Pytorch.

.. autofunction:: tt_lib.fallback_ops.full

.. autofunction:: tt_lib.fallback_ops.tensor_slice

.. autofunction:: tt_lib.fallback_ops.reshape

.. autofunction:: tt_lib.fallback_ops.chunk

.. autofunction:: tt_lib.fallback_ops.conv2d

.. autofunction:: tt_lib.fallback_ops.group_norm

.. autofunction:: tt_lib.fallback_ops.layer_norm

.. autofunction:: tt_lib.fallback_ops.pad

.. autofunction:: tt_lib.fallback_ops.interpolate

.. autofunction:: tt_lib.fallback_ops.repeat

.. autofunction:: tt_lib.fallback_ops.repeat_interleave

.. autofunction:: tt_lib.fallback_ops.concat

.. autofunction:: tt_lib.fallback_ops.silu

.. autofunction:: tt_lib.fallback_ops.softmax

.. autoclass:: tt_lib.fallback_ops.Conv2d

.. autoclass:: tt_lib.fallback_ops.BatchNorm2d

.. autoclass:: tt_lib.fallback_ops.GroupNorm

.. autoclass:: tt_lib.fallback_ops.LayerNorm

.. autoclass:: tt_lib.fallback_ops.MaxPool2d

.. autoclass:: tt_lib.fallback_ops.AdaptiveAvgPool2d

.. autoclass:: tt_lib.fallback_ops.ceil

.. autoclass:: tt_lib.fallback_ops.floor

.. autoclass:: tt_lib.fallback_ops.trunc

.. autoclass:: tt_lib.fallback_ops.unary_fmod

.. autoclass:: tt_lib.fallback_ops.binary_fmod

.. autoclass:: tt_lib.fallback_ops.bitwise_not

.. autoclass:: tt_lib.fallback_ops.unary_bitwise_or

.. autoclass:: tt_lib.fallback_ops.unary_bitwise_and

.. autoclass:: tt_lib.fallback_ops.unary_bitwise_xor

.. autoclass:: tt_lib.fallback_ops.binary_bitwise_or

.. autoclass:: tt_lib.fallback_ops.binary_bitwise_and

.. autoclass:: tt_lib.fallback_ops.binary_bitwise_xor

.. autoclass:: tt_lib.fallback_ops.unary_bitwise_left_shift

.. autoclass:: tt_lib.fallback_ops.unary_bitwise_right_shift

.. autoclass:: tt_lib.fallback_ops.binary_bitwise_left_shift

.. autoclass:: tt_lib.fallback_ops.binary_bitwise_right_shift

.. autoclass:: tt_lib.fallback_ops.torch_argmax

.. autoclass:: tt_lib.fallback_ops.torch_argmin

Experimental Operations
***********************

Operations in this section are experimental, don't have full support, and may behave in unexpected ways.

Fused Operations from ``tt_lib`` Mini-Graph Library
===================================================

We have a variety of common operations that require fusion of multiple
base operations together.

.. autofunction:: tt_lib.fused_ops.linear.Linear

.. autofunction:: tt_lib.fused_ops.softmax.softmax

.. autofunction:: tt_lib.fused_ops.layernorm.Layernorm

.. autofunction:: tt_lib.fused_ops.add_and_norm.AddAndNorm


Complex Operations
==================
 We use the following Tensor representation for complex tensors on device; we support complex tensor **x** as  N,H,W,C rank-4 tensor with last dim of size divisible by 64 to represent real and imaginary components
  * with indices [:,:,:,0:N/2] being real, and
  * with indices [:,:,:,N/2:N] being imaginary.

The following functions are available,


Complex arithmetic can be carried out for multiply, divide, add and subtract as follows:

.. autofunction:: tt_lib.tensor.complex_add

.. autofunction:: tt_lib.tensor.complex_sub

.. autofunction:: tt_lib.tensor.complex_mul

.. autofunction:: tt_lib.tensor.complex_div

and then unary operations for,

.. autofunction:: tt_lib.tensor.real

.. autofunction:: tt_lib.tensor.imag

.. autofunction:: tt_lib.tensor.complex_abs

.. autofunction:: tt_lib.tensor.conj

.. autofunction:: tt_lib.tensor.complex_recip

.. autofunction:: tt_lib.tensor.polar

Complex Operations (Type 2)
===========================
Type 2 Complex representation allows for more flexible storage than earlier one while providing same set of
operations; specifically this storage allows for compute without the cost of split-concat implicit in
the Type 1 contiguous representations.

Other Operations
================

.. autofunction:: tt_lib.tensor.concat

.. autofunction:: tt_lib.tensor.sum

.. autofunction:: tt_lib.tensor.log_sigmoid

.. autofunction:: tt_lib.tensor.expm1

.. autofunction:: tt_lib.tensor.asinh

.. autofunction:: tt_lib.tensor.acosh

.. autofunction:: tt_lib.tensor.erf

.. autofunction:: tt_lib.tensor.erfc

.. autofunction:: tt_lib.tensor.rsqrt

.. autofunction:: tt_lib.tensor.lerp

.. autofunction:: tt_lib.tensor.signbit

.. autofunction:: tt_lib.tensor.fill_rm

.. autofunction:: tt_lib.tensor.fill_ones_rm

.. autofunction:: tt_lib.tensor.conv

.. autofunction:: tt_lib.tensor.bert_large_fused_qkv_matmul

.. autofunction:: tt_lib.tensor.bert_large_ff1_matmul

.. autofunction:: tt_lib.tensor.bert_large_ff2_matmul

.. autofunction:: tt_lib.tensor.bert_large_selfout_matmul

.. autofunction:: tt_lib.tensor.bert_large_pre_softmax_bmm

.. autofunction:: tt_lib.tensor.bert_large_post_softmax_bmm

.. autofunction:: tt_lib.tensor.layernorm

.. autofunction:: tt_lib.tensor.groupnorm

.. autofunction:: tt_lib.tensor.rmsnorm

.. autofunction:: tt_lib.tensor.add_layernorm

.. autofunction:: tt_lib.tensor.convert_conv_weight_tensor_to_tiled_layout

.. autofunction:: tt_lib.tensor.xlogy

.. autofunction:: tt_lib.tensor.addcmul

.. autofunction:: tt_lib.tensor.addcdiv

.. autofunction:: tt_lib.tensor.mean_hw

.. autofunction:: tt_lib.tensor.var_hw

.. autofunction:: tt_lib.tensor.logical_noti

.. autofunction:: tt_lib.tensor.std_hw

.. autofunction:: tt_lib.tensor.normalize_hw

.. autofunction:: tt_lib.tensor.normalize_global

.. autofunction:: tt_lib.tensor.glu

.. autofunction:: tt_lib.tensor.geglu

.. autofunction:: tt_lib.tensor.reglu

.. autofunction:: tt_lib.tensor.swiglu

.. autofunction:: tt_lib.tensor.embeddings

.. autofunction:: tt_lib.tensor.nextafter

.. autofunction:: tt_lib.tensor.lamb_optimizer

.. autofunction:: tt_lib.tensor.repeat

.. autofunction:: tt_lib.tensor.repeat_interleave

.. autofunction:: tt_lib.tensor.pow

.. autofunction:: tt_lib.tensor.identity

.. autofunction:: tt_lib.tensor.argmax

.. autofunction:: tt_lib.tensor.argmin

Backward Operations
===================

.. autofunction:: tt_lib.tensor.addalpha_bw

.. autofunction:: tt_lib.tensor.addcmul_bw

.. autofunction:: tt_lib.tensor.addcdiv_bw

.. autofunction:: tt_lib.tensor.conj_bw

.. autofunction:: tt_lib.tensor.unary_mul_bw

.. autofunction:: tt_lib.tensor.unary_add_bw

.. autofunction:: tt_lib.tensor.unary_assign_bw

.. autofunction:: tt_lib.tensor.binary_assign_bw

.. autofunction:: tt_lib.tensor.unary_div_bw

.. autofunction:: tt_lib.tensor.div_bw

.. autofunction:: tt_lib.tensor.rdiv_bw

.. autofunction:: tt_lib.tensor.sqrt_bw

.. autofunction:: tt_lib.tensor.mul_bw

.. autofunction:: tt_lib.tensor.max_bw

.. autofunction:: tt_lib.tensor.min_bw

.. autofunction:: tt_lib.tensor.add_bw

.. autofunction:: tt_lib.tensor.tan_bw

.. autofunction:: tt_lib.tensor.exp_bw

.. autofunction:: tt_lib.tensor.exp2_bw

.. autofunction:: tt_lib.tensor.expm1_bw

.. autofunction:: tt_lib.tensor.unary_pow_bw

.. autofunction:: tt_lib.tensor.embedding_bw

.. autofunction:: tt_lib.tensor.where_bw

.. autofunction:: tt_lib.tensor.tanh_bw

.. autofunction:: tt_lib.tensor.fill_zero_bw

.. autofunction:: tt_lib.tensor.fill_bw

.. autofunction:: tt_lib.tensor.sub_bw

.. autofunction:: tt_lib.tensor.unary_sub_bw

.. autofunction:: tt_lib.tensor.log_bw

.. autofunction:: tt_lib.tensor.rsub_bw

.. autofunction:: tt_lib.tensor.abs_bw

.. autofunction:: tt_lib.tensor.rsqrt_bw

.. autofunction:: tt_lib.tensor.neg_bw

.. autofunction:: tt_lib.tensor.lt_bw

.. autofunction:: tt_lib.tensor.gt_bw

.. autofunction:: tt_lib.tensor.relu_bw

.. autofunction:: tt_lib.tensor.ne_bw

.. autofunction:: tt_lib.tensor.clamp_bw

.. autofunction:: tt_lib.tensor.clamp_min_bw

.. autofunction:: tt_lib.tensor.clamp_max_bw

.. autofunction:: tt_lib.tensor.binary_le_bw

.. autofunction:: tt_lib.tensor.atan2_bw

.. autofunction:: tt_lib.tensor.hypot_bw

.. autofunction:: tt_lib.tensor.gelu_bw

.. autofunction:: tt_lib.tensor.bias_gelu_bw

.. autofunction:: tt_lib.tensor.bias_gelu_unary_bw

.. autofunction:: tt_lib.tensor.squared_difference_bw

.. autofunction:: tt_lib.tensor.lerp_bw

.. autofunction:: tt_lib.tensor.ldexp_bw

.. autofunction:: tt_lib.tensor.xlogy_bw

.. autofunction:: tt_lib.tensor.logaddexp_bw

.. autofunction:: tt_lib.tensor.logaddexp2_bw

.. autofunction:: tt_lib.tensor.concat_bw

.. autofunction:: tt_lib.tensor.hardsigmoid_bw

.. autofunction:: tt_lib.tensor.i0_bw

.. autofunction:: tt_lib.tensor.hardshrink_bw

.. autofunction:: tt_lib.tensor.softshrink_bw

.. autofunction:: tt_lib.tensor.hardswish_bw

.. autofunction:: tt_lib.tensor.softplus_bw

.. autofunction:: tt_lib.tensor.polygamma_bw

.. autofunction:: tt_lib.tensor.atan_bw

.. autofunction:: tt_lib.tensor.atanh_bw

.. autofunction:: tt_lib.tensor.asin_bw

.. autofunction:: tt_lib.tensor.asinh_bw

.. autofunction:: tt_lib.tensor.cosh_bw

.. autofunction:: tt_lib.tensor.cos_bw

.. autofunction:: tt_lib.tensor.acosh_bw

.. autofunction:: tt_lib.tensor.acos_bw

.. autofunction:: tt_lib.tensor.erfinv_bw

.. autofunction:: tt_lib.tensor.leaky_relu_bw

.. autofunction:: tt_lib.tensor.elu_bw

.. autofunction:: tt_lib.tensor.hardtanh_bw

.. autofunction:: tt_lib.tensor.angle_bw

.. autofunction:: tt_lib.tensor.sin_bw

.. autofunction:: tt_lib.tensor.sinh_bw

.. autofunction:: tt_lib.tensor.celu_bw

.. autofunction:: tt_lib.tensor.binary_lt_bw

.. autofunction:: tt_lib.tensor.subalpha_bw

.. autofunction:: tt_lib.tensor.log10_bw

.. autofunction:: tt_lib.tensor.log1p_bw

.. autofunction:: tt_lib.tensor.binary_ne_bw

.. autofunction:: tt_lib.tensor.erf_bw

.. autofunction:: tt_lib.tensor.erfc_bw

.. autofunction:: tt_lib.tensor.digamma_bw

.. autofunction:: tt_lib.tensor.deg2rad_bw

.. autofunction:: tt_lib.tensor.rad2deg_bw

.. autofunction:: tt_lib.tensor.reciprocal_bw

.. autofunction:: tt_lib.tensor.relu6_bw

.. autofunction:: tt_lib.tensor.rpow_bw

.. autofunction:: tt_lib.tensor.silu_bw

.. autofunction:: tt_lib.tensor.selu_bw

.. autofunction:: tt_lib.tensor.binary_ge_bw

.. autofunction:: tt_lib.tensor.binary_eq_bw

.. autofunction:: tt_lib.tensor.binary_gt_bw

.. autofunction:: tt_lib.tensor.square_bw

.. autofunction:: tt_lib.tensor.lgamma_bw

.. autofunction:: tt_lib.tensor.trunc_bw

.. autofunction:: tt_lib.tensor.frac_bw

.. autofunction:: tt_lib.tensor.log_sigmoid_bw

.. autofunction:: tt_lib.tensor.tanhshrink_bw

.. autofunction:: tt_lib.tensor.threshold_bw

.. autofunction:: tt_lib.tensor.unary_eq_bw

.. autofunction:: tt_lib.tensor.logit_bw

.. autofunction:: tt_lib.tensor.logiteps_bw

.. autofunction:: tt_lib.tensor.softsign_bw

.. autofunction:: tt_lib.tensor.sign_bw

.. autofunction:: tt_lib.tensor.ceil_bw

.. autofunction:: tt_lib.tensor.log2_bw

.. autofunction:: tt_lib.tensor.ge_bw

.. autofunction:: tt_lib.tensor.le_bw

.. autofunction:: tt_lib.tensor.unary_fmod_bw

.. autofunction:: tt_lib.tensor.unary_remainder_bw

.. autofunction:: tt_lib.tensor.imag_bw

.. autofunction:: tt_lib.tensor.real_bw

.. autofunction:: tt_lib.tensor.multigammaln_bw

Loss Functions
==============

.. autofunction:: tt_lib.tensor.mseloss

.. autofunction:: tt_lib.tensor.maeloss
