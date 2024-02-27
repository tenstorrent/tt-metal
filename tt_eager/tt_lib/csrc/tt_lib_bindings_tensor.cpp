// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings.hpp"
#include "tt_dnn/op_library/downsample/downsample_op.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/conv/conv_op.hpp"
#include "tt_dnn/op_library/conv/optimized_conv_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_dnn/op_library/upsample/upsample_op.hpp"
#include "tt_dnn/op_library/groupnorm/groupnorm_op.hpp"
#include "tt_dnn/op_library/pool/average_pool.hpp"
#include "tt_dnn/op_library/pool/max_pool.hpp"
#include "tt_dnn/op_library/fully_connected/fully_connected_op.hpp"
#include "tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/split/split_last_dim_two_chunks_tiled.hpp"
#include "tt_dnn/op_library/rotate_half/rotate_half_op.hpp"
#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_op.hpp"
#include "tt_eager/tt_dnn/op_library/loss/loss_op.hpp"
#include "tt_dnn/op_library/embeddings/embeddings_op.hpp"
#include "tt_dnn/op_library/update_cache/update_cache_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/borrowed_buffer.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/serialization.hpp"
#include "type_caster.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "tt_lib_bindings_tensor.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt::tt_metal{

namespace detail{
    template<class T>
    struct DataTypeToFormatType {
        using type = T;
    };

    template<>
    struct DataTypeToFormatType<bfloat16> {
        using type = uint16_t;
    };

    template<class CppType, class DataType, class PyType>
    void implement_buffer_protocol(PyType& py_buffer_t) {
        py_buffer_t
            .def(
                "__getitem__",
                [](const CppType& self, std::size_t index) {
                    return self[index];
                }
            )
            .def(
                "__len__",
                [](const CppType& self) {
                    return self.size();
                }
            )
            .def(
                "__iter__",
                [](const CppType& self) {
                    return py::make_iterator(self.begin(), self.end());
                },
                py::keep_alive<0, 1>()
            )
            .def_buffer(
                [](CppType& self) -> py::buffer_info {
                    using FormatType = typename DataTypeToFormatType<DataType>::type;
                    return py::buffer_info(
                        self.begin(),                                /* Pointer to buffer */
                        sizeof(DataType),                            /* Size of one scalar */
                        py::format_descriptor<FormatType>::format(), /* Python struct-style format descriptor */
                        1,                                           /* Number of dimensions */
                        { self.size() },                             /* Buffer dimensions */
                        { sizeof(DataType) }                         /* Strides (in bytes) for each index */
                    );
                }
            );
    };

}

void TensorModule(py::module &m_tensor) {
    // ENUM SECTION

    // layout enums
    detail::export_enum<Layout>(m_tensor);

    detail::export_enum<DataType>(m_tensor);

    detail::export_enum<StorageType>(m_tensor);

    detail::export_enum<MathFidelity>(m_tensor);

    detail::export_enum<TensorMemoryLayout>(m_tensor);

    detail::export_enum<ShardOrientation>(m_tensor);

    detail::export_enum<LossReductionMode>(m_tensor);

    py::enum_<BufferType>(m_tensor, "BufferType")
        .value("DRAM", BufferType::DRAM)
        .value("L1", BufferType::L1);

    // Fusible Activations
    detail::export_enum<UnaryOpType>(m_tensor, "FusibleActivation");
    py::class_<UnaryWithParam>(m_tensor, "FusibleActivationWithParam")
        .def(py::init<UnaryOpType>())
        .def(py::init<UnaryOpType, float>())
        .def(py::init<>(
            [](std::pair<UnaryOpType, float> arg) {
                return UnaryWithParam{.op_type=arg.first, .param=arg.second};
            }
        ));
    // Allow implicit construction of UnaryWithParam object without user explicitly creating it
    // Can take in just the op type, or sequence container of op type and param value
    py::implicitly_convertible<UnaryOpType, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, float>, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, int>, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, bool>, UnaryWithParam>();

    detail::export_enum<EmbeddingsType>(m_tensor);

    auto py_core_coord = py::class_<CoreCoord>(m_tensor, "CoreCoord", R"doc(
        Class defining core coordinate
    )doc");

    py_core_coord
        .def(py::init<std::size_t, std::size_t>())
        .def(
            py::init<>(
                [] (std::tuple<std::size_t, std::size_t> core_coord) {
                    return CoreCoord(std::get<0>(core_coord), std::get<1>(core_coord));
                }
            )
        )
        .def("__repr__", [](const CoreCoord& self) -> std::string {
            return self.str();
        })
        .def_readonly("x", &CoreCoord::x)
        .def_readonly("y", &CoreCoord::y);
    py::implicitly_convertible<std::tuple<std::size_t, std::size_t>, CoreCoord>();

    auto py_shape = py::class_<Shape>(m_tensor, "Shape", R"doc(
        Class defining tensor shape
    )doc");

    py_shape.def(py::init<std::array<uint32_t, 4>>())
        .def(
            py::init(
                [](const std::vector<uint32_t>& shape,
                   const std::optional<std::vector<uint32_t>>& padded_shape_arg) -> Shape {
                    auto rank = shape.size();
                    std::vector<Padding::PadDimension> padding;
                    auto padded_shape = Shape{shape};
                    if (padded_shape_arg.has_value()) {
                        padded_shape = padded_shape_arg.value();
                        TT_ASSERT(shape.size() == padded_shape.rank());
                        for (auto index = 0; index < rank; index++) {
                            padding.push_back({.front = 0, .back = padded_shape[index] - shape[index]});
                        }
                    } else {
                        padding = std::vector<Padding::PadDimension>(rank, {.front = 0, .back = 0});
                    }
                    auto output = Shape{padded_shape, Padding{padding, Padding::PadValue::Any}};
                    return output;
                }),
            py::arg("shape"),
            py::arg("padded_shape") = std::nullopt)
        .def("__len__", [](const Shape& self) { return self.rank(); })
        .def("__eq__", [](const Shape& self, const Shape& other) { return self == other; })
        .def("__eq__", [](const Shape& self, const std::vector<uint32_t>& other) { return self == Shape{other}; })
        .def("__eq__", [](const Shape& self, const std::array<uint32_t, 4>& other) { return self == Shape{other}; })
        .def("__eq__", [](const Shape& self, const py::none) { return false; })
        .def("__getitem__", [](const Shape& self, const std::int64_t index) { return self[index]; })
        .def(
            "__getitem__",
            [](const Shape& self, const py::slice slice) {
                size_t start = 0, stop = 0, step = 0, slicelength = 0;
                if (!slice.compute(self.rank(), &start, &stop, &step, &slicelength)) {
                    throw std::runtime_error("Invalid slice");
                }

                std::vector<uint32_t> output;
                for (auto index = start; index < stop; index += step) {
                    output.push_back(self[index]);
                }
                return Shape{output};
            })
        .def(
            "__iter__",
            [](const Shape& self) { return py::make_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>())
        .def("__repr__", [](const Shape& self) { return fmt::format("{}", self); })
        .def("without_padding", [](const Shape& self) -> Shape { return self.without_padding(); });

    py::implicitly_convertible<std::vector<uint32_t>, Shape>();

    auto pyMemoryConfig = py::class_<MemoryConfig>(m_tensor, "MemoryConfig", R"doc(
        Class defining memory configuration for storing tensor data on TT Accelerator device.
        There are eight DRAM memory banks on TT Accelerator device, indexed as 0, 1, 2, ..., 7.
    )doc");

    pyMemoryConfig
        .def(
            py::init<>(
                [](TensorMemoryLayout memory_layout, BufferType buffer_type, std::optional<ShardSpec> shard_spec) {
                    return MemoryConfig{.memory_layout=memory_layout, .buffer_type=buffer_type, .shard_spec=shard_spec};
                }
            ),
            py::arg("memory_layout") = TensorMemoryLayout::INTERLEAVED,
            py::arg("buffer_type") = BufferType::DRAM,
            py::arg("shard_spec") = std::nullopt,
            R"doc(
                Create MemoryConfig class.
                If interleaved is set to True, tensor data will be interleaved across multiple DRAM banks on TT Accelerator device.
                Otherwise, tensor data will be stored in a DRAM bank selected by dram_channel (valid values are 0, 1, ..., 7).

                Example of creating MemoryConfig specifying that tensor data should be stored in DRAM bank 3.

                .. code-block:: python

                    mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.SINGLE_BANK)
            )doc"
        )
        .def("__repr__", [](const MemoryConfig &memory_config) -> std::string {
            return fmt::format("{}", memory_config);
        }
        )
        .def("is_sharded", &MemoryConfig::is_sharded, "Whether tensor data is sharded across multiple cores in L1")
        .def_property_readonly("interleaved", [](const MemoryConfig &memory_config) {
            return memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED;
        }, "Whether tensor data is interleaved across multiple DRAM channels"
        )
        .def_readonly("buffer_type", &MemoryConfig::buffer_type, "Buffer type to store tensor data. Can be DRAM or L1")
        .def_readonly("memory_layout", &MemoryConfig::memory_layout, "Memory layout of tensor data.")
        .def_readwrite("shard_spec", &MemoryConfig::shard_spec, "Memory layout of tensor data.")
        .def(py::self == py::self)
        .def(py::self != py::self);

    auto py_owned_buffer_for_uint16_t =
        py::class_<owned_buffer::Buffer<uint16_t>>(m_tensor, "owned_buffer_for_uint16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<uint16_t>, uint16_t>(py_owned_buffer_for_uint16_t);

    auto pyCoreRange = py::class_<CoreRange>(m_tensor, "CoreRange",  R"doc(
        Class defining a range of cores)doc"
    );
    pyCoreRange
        .def(
            py::init<>(
                [](const CoreCoord &start, const CoreCoord & end ) {
                    return CoreRange{start, end};
                }
            )
        )
        .def("__repr__", [](const CoreRange &core_range) -> std::string {
            return fmt::format("{}", core_range);
        }
        );

    auto pyCoreRangeSet = py::class_<CoreRangeSet>(m_tensor, "CoreRangeSet",  R"doc(
        Class defining a set of CoreRanges required for sharding)doc"
    );
    pyCoreRangeSet.def(py::init<>([](const std::set<CoreRange>& core_ranges) { return CoreRangeSet(core_ranges); }))
        .def("__repr__", [](const CoreRangeSet& core_range_set) -> std::string {
            return fmt::format("{}", core_range_set);
        });

    auto pyShardSpec = py::class_<ShardSpec>(m_tensor, "ShardSpec", R"doc(
        Class defining the specs required for sharding.
    )doc");

    pyShardSpec
        .def(
            py::init<>(
                [](const CoreRangeSet & core_sets,
                    const std::array<uint32_t,2> & shard_shape,
                    const ShardOrientation & shard_orientation,
                    const bool & halo
                     ) {
                    return ShardSpec(core_sets, shard_shape, shard_orientation, halo);
                }
            )
        )
        .def_readwrite("shape", &ShardSpec::shape, "Shape of shard.")
        .def_readwrite("grid", &ShardSpec::grid, "Grid to layout shards.")
        .def_readwrite("orientation", &ShardSpec::orientation, "Orientation of cores to read shards")
        .def("num_cores", &ShardSpec::num_cores, "Number of cores")
        .def(py::self == py::self)
        .def(py::self != py::self)
        ;


    auto py_owned_buffer_for_uint32_t = py::class_<owned_buffer::Buffer<uint32_t>>(m_tensor, "owned_buffer_for_uint32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<uint32_t>, uint32_t>(py_owned_buffer_for_uint32_t);

    auto py_owned_buffer_for_float32_t = py::class_<owned_buffer::Buffer<float>>(m_tensor, "owned_buffer_for_float32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<float>, float>(py_owned_buffer_for_float32_t);

    auto py_owned_buffer_for_bfloat16_t = py::class_<owned_buffer::Buffer<bfloat16>>(m_tensor, "owned_buffer_for_bfloat16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<bfloat16>, bfloat16>(py_owned_buffer_for_bfloat16_t);

    auto py_borrowed_buffer_for_uint16_t = py::class_<borrowed_buffer::Buffer<std::uint16_t>>(
        m_tensor, "borrowed_buffer_for_uint16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::uint16_t>, std::uint16_t>(
        py_borrowed_buffer_for_uint16_t);

    auto py_borrowed_buffer_for_uint32_t = py::class_<borrowed_buffer::Buffer<std::uint32_t>>(m_tensor, "borrowed_buffer_for_uint32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::uint32_t>, std::uint32_t>(py_borrowed_buffer_for_uint32_t);

    auto py_borrowed_buffer_for_float32_t = py::class_<borrowed_buffer::Buffer<float>>(m_tensor, "borrowed_buffer_for_float32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<float>, float>(py_borrowed_buffer_for_float32_t);

    auto py_borrowed_buffer_for_bfloat16_t = py::class_<borrowed_buffer::Buffer<bfloat16>>(m_tensor, "borrowed_buffer_for_bfloat16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<bfloat16>, bfloat16>(py_borrowed_buffer_for_bfloat16_t);

    py::class_<DeviceComputeKernelConfig>(m_tensor, "DeviceComputeKernelConfig");

    py::class_<GrayskullComputeKernelConfig>(m_tensor, "GrayskullComputeKernelConfig")
        .def(
            py::init<MathFidelity, bool>(),
            py::kw_only(),
            py::arg("math_fidelity") = MathFidelity::Invalid,
            py::arg("math_approx_mode") = true
        )
        .def_readwrite("math_fidelity", &GrayskullComputeKernelConfig::math_fidelity)
        .def_readwrite("math_approx_mode", &GrayskullComputeKernelConfig::math_approx_mode);

    py::class_<WormholeComputeKernelConfig>(m_tensor, "WormholeComputeKernelConfig")
        .def(
            py::init<MathFidelity, bool, bool, bool>(),
            py::kw_only(),
            py::arg("math_fidelity") = MathFidelity::Invalid,
            py::arg("math_approx_mode") = true,
            py::arg("fp32_dest_acc_en") = false,
            py::arg("packer_l1_acc") = false
        )
        .def_readwrite("math_fidelity", &WormholeComputeKernelConfig::math_fidelity)
        .def_readwrite("math_approx_mode", &WormholeComputeKernelConfig::math_approx_mode)
        .def_readwrite("fp32_dest_acc_en", &WormholeComputeKernelConfig::fp32_dest_acc_en)
        .def_readwrite("packer_l1_acc", &WormholeComputeKernelConfig::packer_l1_acc);

    detail::bind_unary_op(m_tensor, "mean_hw", tt::tt_metal::mean_hw, R"doc(  Returns a new tensor with the variance of the input tensor ``{0}`` on H,W axes.)doc");
    detail::bind_unary_op(m_tensor, "global_mean", tt::tt_metal::global_mean, R"doc(  Returns a new tensor with the mean of the input tensor ``{0}`` on all axes.)doc");
    detail::bind_unary_op(m_tensor, "global_sum", tt::tt_metal::global_sum, R"doc(  Returns a new tensor with the sum of the input tensor ``{0}`` on all axes.)doc");
    detail::bind_unary_op(m_tensor, "global_max", tt::tt_metal::global_max, R"doc(  Returns a new tensor with the max of the input tensor ``{0}`` on all axes.)doc");
    detail::bind_unary_op(m_tensor, "global_min", tt::tt_metal::global_min, R"doc(  Returns a new tensor with the min of the input tensor ``{0}`` on all axes.)doc");

    detail::bind_unary_op_with_param(
        m_tensor, "sum", &sum,
        py::arg("dim"),
        R"doc(Returns a tensor that is a sum  of input tensor with shape ``[W, Z, Y, X]`` along dimensions ``{1}``; input tensor in TILE LAYOUT.)doc",
        R"doc("dimension along which to apply sum", "int", "0, 1, 2, or 3")doc"
    );

    detail::bind_unary_op_with_param(
        m_tensor, "max", tt::tt_metal::max,
        py::arg("dim"),
        R"doc(Returns a tensor that is a max  of input tensor with shape ``[W, Z, Y, X]`` along dimensions ``{1}``; input tensor in TILE LAYOUT.)doc",
        R"doc("dimension along which to apply max", "int", "0, 1, 2, or 3")doc"
    );

    m_tensor.def("downsample", &downsample,
        py::arg().noconvert(), py::arg().noconvert(), py::arg("output_dtype").noconvert() = std::nullopt,
        R"doc(
        Performs a downsample on the input of a conv with a stride > 1 and a kernel window 1x1
        This op can be followed by a regular matmul to perform the conv1x1 with stride=1 operation

        +-------------------+-----------------------------------------------------------------------------------+---------------+-------------+----------+
        | Argument          | Description                                                                       | Data type     | Valid range | Required |
        +===================+===================================================================================+===============+=============+==========+
        | a                 | Input tensor (TILED)                                                              | uint32_t      |             | Yes      |
        | downsample_params | Params list: batch size, conv input H, conv input W, conv stride H, conv stride W | uint32_t      |             | Yes      |
        +-------------------+-----------------------------------------------------------------------------------+---------------+-------------+----------+
    )doc");

    m_tensor.def("conv", &conv, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    py::class_<OptimizedConvParallelizationConfig>(m_tensor, "OptimizedConvParallelizationConfig")
        .def(
            py::init<CoreCoord, uint32_t, uint32_t, uint32_t>(),
            py::kw_only(),
            py::arg("grid_size"),
            py::arg("num_cores_nhw"),
            py::arg("per_core_out_matrix_height_ntiles").noconvert(),
            py::arg("per_core_out_matrix_width_ntiles").noconvert()
        )
        .def_property_readonly("grid_size", [](OptimizedConvParallelizationConfig const& c) { return c.grid_size; })
        .def_property_readonly("num_cores_nhw", [](OptimizedConvParallelizationConfig const& c) { return c.num_cores_nhw; })
        .def_property_readonly("per_core_out_matrix_height_ntiles", [](OptimizedConvParallelizationConfig const& c) { return c.per_core_out_matrix_height_ntiles; })
        .def_property_readonly("per_core_out_matrix_width_ntiles", [](OptimizedConvParallelizationConfig const& c) { return c.per_core_out_matrix_width_ntiles; });

    py::class_<OptimizedConvBlockConfig>(m_tensor, "OptimizedConvBlockConfig")
        .def(
            py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
            py::kw_only(),
            py::arg("act_block_h_ntiles").noconvert(),
            py::arg("act_block_w_ntiles").noconvert(),
            py::arg("out_subblock_h_ntiles").noconvert(),
            py::arg("out_subblock_w_ntiles").noconvert()
        )
        .def_property_readonly("act_block_h_ntiles", [](OptimizedConvBlockConfig const& c) { return c.act_block_h_ntiles; })
        .def_property_readonly("act_block_w_ntiles", [](OptimizedConvBlockConfig const& c) { return c.act_block_w_ntiles; })
        .def_property_readonly("out_subblock_h_ntiles", [](OptimizedConvBlockConfig const& c) { return c.out_subblock_h_ntiles; })
        .def_property_readonly("out_subblock_w_ntiles", [](OptimizedConvBlockConfig const& c) { return c.out_subblock_w_ntiles; });

    m_tensor.def("optimized_conv", &optimized_conv,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("conv_reader_indices").noconvert() = std::nullopt,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert() = 0,
                 py::arg("output_mem_config").noconvert() = std::nullopt, py::arg("output_dtype").noconvert() = std::nullopt, py::arg("input_tensor_shape").noconvert() = std::nullopt,
                 py::arg("use_shallow_conv_variant").noconvert() = false, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def("conv_with_fast_reader", &conv_with_fast_reader,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                 py::arg().noconvert(), py::arg().noconvert(), py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def("conv_with_address_map", &conv_with_address_map, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output
        Reader kernel uses an address map which pre-computed on the host to read activations and weights

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");



    // moreh_linear
    m_tensor.def("moreh_matmul", &moreh_matmul,
        py::arg("input_a").noconvert(), py::arg("input_b").noconvert(),  py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a moreh_linear operation.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input_a", "First input tensor", "Tensor", "", "Yes"
            "input_b", "Second input tensor", "Tensor", "", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    // groupnorm
    m_tensor.def("groupnorm", &groupnorm,
        py::arg("input").noconvert(), py::arg("group_size").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a groupnorm operation on the channel dimension grouped per group_size, with optional fused with post-multiplication and addition via W-bcast.
    )doc");

    // layernorm
    m_tensor.def("layernorm", layernorm,
        py::arg("input").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
        "Performs a layernorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
    )doc");
    m_tensor.def("add_layernorm", add_layernorm,
        py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
        "Performs a layernorm(a+b)*gamma + beta operation."
    )doc");
    m_tensor.def("rmsnorm", rmsnorm,
        py::arg("input").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
        "Performs a rmsnorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
    )doc");
    m_tensor.def("add_rmsnorm", add_rmsnorm,
        py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
        "Performs a rmsnorm(a+b)*gamma + beta operation.
    )doc");
    m_tensor.def("rotate_half", &rotate_half,
        py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a rotate half operation used by RotaryEmbedding.
    )doc");
    m_tensor.def("rotary_embedding", &rotary_embedding,
        py::arg("input").noconvert(), py::arg("cos").noconvert(), py::arg("sin").noconvert(), py::arg("token_idx") = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
        "Performs rotary embedding with a given input, cos, and sin tensors. Sequence length is inferred as the second last dim of the input tensor.
        If token_idx is passed, this assumes input is transposed to [seq_len, 1, B, head_dim], and seq_len is 1.
    )doc");
    m_tensor.def("fill_cache", &fill_cache,
         py::arg("cache").noconvert(), py::arg("input").noconvert(), py::arg("batch_idx"), R"doc(
        "Fills the cache tensor in place with the values from input at the specified batch_idx.
    )doc");
    m_tensor.def("update_cache", &update_cache,
         py::arg("cache").noconvert(), py::arg("input").noconvert(), py::arg("update_idx"), py::arg("batch_offset") = 0, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
        "Updates the cache tensor in place with the values from input at the specified update_idx. When cache has batch less than 32, input is assumed to have batch padded to 32 and [batch_offset:batch_offset+batch] from dim[-2] of input is used to update the cache.
    )doc");


    // input embeddings
    m_tensor.def("embeddings", &embeddings,
        py::arg("input").noconvert(), py::arg("weights").noconvert(),
        py::arg("tilized").noconvert() = false,
        py::arg("embeddings_type").noconvert() = EmbeddingsType::GENERIC,
        py::arg("pad_token").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Returns specific indices of the embedding table specified by the input tensor

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor containing rows we want", "UInt32 Tensor", "Each element greater than 0 and less than number of embeddings in table.  Shape [batch_size, 1, 1, num_rows]", "Yes"
            "weights", "Entire embedding table", "Tensor", "Tensor shape is [1,1, num_embeddings, num_columns]. Num_columns must be divisible by 32.", "Yes"
            "tilized", "Enable fused tilize on output. Default is true.", "Bool", "", "No",
            "embeddings_type", "Version of optimized embeddings to run. PADDED requires passing pad_token. BINARY expects the indices to only be 0, 1 and weights to have 2 rows", "EmbeddingsType", "GENERIC, PADDED, BINARY", "No"
            "pad_token", "pad_token used in token ids", "uint32_t", "Default is None", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            "output_dtype", "DataType of output tensor", "DataType", "Default is weights dtype", "No"
    )doc");


    // FC
    m_tensor.def("fully_connected", &fully_connected,
        py::arg("act").noconvert(), py::arg("weights").noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Fully connected layer (linear.)

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "act", "Input activations tensor", "Tensor", "", "Yes"
            "weights", "Input weights tensor", "Tensor", "", "Yes"
            "bias", "Input bias tensor", "Tensor", "", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    // Pools
    m_tensor.def("average_pool_2d", &average_pool_2d,
        py::arg().noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,  py::arg("output_dtype").noconvert() = std::nullopt,  R"doc(
        Average Pool 2D
        It operates on tensors whose that have channels as the last dimension

        +----------+----------------------------+------------+-------------------------------+----------+
        | Argument | Description                | Data type  | Valid range                   | Required |
        +==========+============================+============+===============================+==========+
        | act      | Input activations tensor   | Tensor     |                               | Yes      |
        +----------+----------------------------+------------+-------------------------------+----------+
    )doc");

    // Upsample
    m_tensor.def("upsample", &upsample,
        py::arg("input").noconvert(),
        py::arg("scale_factor_h").noconvert(),
        py::arg("scale_factor_w").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        UpSample 2D
        It operates on tensors whose that have channels as the last dimension

        +----------+----------------------------+------------+-------------------------------+----------+
        | Argument | Description                | Data type  | Valid range                   | Required |
        +==========+============================+============+===============================+==========+
        | act      | Input activations tensor   | Tensor     |                               | Yes      |
        +----------+----------------------------+------------+-------------------------------+----------+
    )doc");

    m_tensor.def("max_pool2d", &max_pool2d,
        py::arg("input").noconvert(),
        py::arg("in_n").noconvert(),
        py::arg("in_h").noconvert(), py::arg("in_w").noconvert(),
        py::arg("kernel_h").noconvert(), py::arg("kernel_w").noconvert(),
        py::arg("stride_h") = 1, py::arg("stride_w") = 1,
        py::arg("pad_h") = 0, py::arg("pad_w") = 0,
        py::arg("dilation_h") = 1, py::arg("dilation_w") = 1,
        py::arg("output_mem_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("nblocks") = 1,
        py::arg("use_multicore") = true, R"doc(
        Max Pool 2D
        +-------------------+-------------------------------+---------------+-------------+----------+
        | Argument          | Description                   | Data type     | Valid range | Required |
        +===================+===============================+===============+=============+==========+
        | input             | Input activations tensor      | Tensor        |             | Yes      |
        | in_n              | Input nbatch                  | Tensor        |             | Yes      |
        | in_h              | Input height                  | Tensor        |             | Yes      |
        | in_w              | Input width                   | Tensor        |             | Yes      |
        | kernel_h          | kernel window height          | uint32_t      |             | Yes      |
        | kernel_w          | kernel window width           | uint32_t      |             | Yes      |
        | stride_h          | stride in height dim          | uint32_t      |             | No       |
        | stride_w          | stride in width dim           | uint32_t      |             | No       |
        | pad_h             | padding in height dim         | uint32_t      |             | No       |
        | pad_w             | padding in width dim          | uint32_t      |             | No       |
        | dilation_h        | kernel dilation in height dim | uint32_t      |             | No       |
        | dilation_w        | kernel dilation in width dim  | uint32_t      |             | No       |
        | output_mem_config | output tensor memory config   | MemoryConfig  |             | No       |
        +-------------------+-------------------------------+---------------+-------------+----------+
    )doc");

    m_tensor.def("max_pool2d_v2", &max_pool2d_v2,
        py::arg("input").noconvert(),
        py::arg("reader_indices").noconvert(),
        py::arg("in_n").noconvert(),
        py::arg("in_h").noconvert(),
        py::arg("in_w").noconvert(),
        py::arg("kernel_h").noconvert(),
        py::arg("kernel_w").noconvert(),
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("pad_h") = 0,
        py::arg("pad_w") = 0,
        py::arg("dilation_h") = 1,
        py::arg("dilation_w") = 1,
        py::arg("output_mem_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("nblocks") = 1,
        py::arg("use_multicore") = true, R"doc(
        Max Pool 2D
        +-------------------+-------------------------------+---------------+-------------+----------+
        | Argument          | Description                   | Data type     | Valid range | Required |
        +===================+===============================+===============+=============+==========+
        | input             | Input activations tensor      | Tensor        |             | Yes      |
        | in_n              | Input nbatch                  | Tensor        |             | Yes      |
        | in_h              | Input height                  | Tensor        |             | Yes      |
        | in_w              | Input width                   | Tensor        |             | Yes      |
        | kernel_h          | kernel window height          | uint32_t      |             | Yes      |
        | kernel_w          | kernel window width           | uint32_t      |             | Yes      |
        | stride_h          | stride in height dim          | uint32_t      |             | No       |
        | stride_w          | stride in width dim           | uint32_t      |             | No       |
        | pad_h             | padding in height dim         | uint32_t      |             | No       |
        | pad_w             | padding in width dim          | uint32_t      |             | No       |
        | dilation_h        | kernel dilation in height dim | uint32_t      |             | No       |
        | dilation_w        | kernel dilation in width dim  | uint32_t      |             | No       |
        | output_mem_config | output tensor memory config   | MemoryConfig  |             | No       |
        +-------------------+-------------------------------+---------------+-------------+----------+
    )doc");

    // TMs
    m_tensor.def("split_last_dim_two_chunks_tiled", &split_last_dim_two_chunks_tiled, py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Splits a tensor's last dimension in two equal sized chunks. This assumes the last dim is tile sized.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W0, Z0, Y0, X0]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

    )doc");

    m_tensor.def("split_dim_two_chunks_tiled", &split_dim_two_chunks_tiled, py::arg("input").noconvert(), py::arg("dim").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Splits a tensor's last or penultimate dimension in two equal sized chunks. This assumes the last dim is tile sized and penultimate dimension is also tile sized.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W0, Z0, Y0, X0]", "Yes"
            "dim", "Dimension", "integer", "Dimension to split over can only be 2 or 3", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    m_tensor.def("convert_conv_weight_tensor_to_tiled_layout", &convert_conv_weight_tensor_to_tiled_layout,
        py::arg("conv_weight_tensor").noconvert(), py::arg("in1_block_h"), py::arg("in1_block_w"), py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
        Converts convolution weights to 2d matrix tiled layout on host
        Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def("convert_conv_weight_tensor_to_special_padding_tiled_layout", &convert_conv_weight_tensor_to_special_padding_tiled_layout,
        py::arg("conv_weight_tensor").noconvert(), py::arg("in1_block_h"), py::arg("in1_block_w"), py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
       Converts convolution weights to 2d matrix tiled layout on host with special block height padding
       Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def(
        "format_input_tensor", &AutoFormat::format_input_tensor,
        py::arg("input").noconvert(), py::arg("device").noconvert(), py::arg("padded_shape"), py::arg("pad_value"), py::arg("target_layout").noconvert(), py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
            Formats tensor to target layout and pads to padded shape
        )doc"
    );
    m_tensor.def(
        "format_output_tensor", &AutoFormat::format_output_tensor,
        py::arg("output").noconvert(), py::arg("shape"), py::arg("device").noconvert(), py::arg("target_layout").noconvert(), py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
            Formats tensor to target layout and unpads to shape
        )doc"
    );
    m_tensor.def(
        "pad_to_tile_shape",
        [](const std::array<uint32_t, 4>& unpadded_shape,
           bool pad_c = false,
           bool pad_n = false,
           bool pad_h = true,
           bool pad_w = true) -> Shape {
            return AutoFormat::pad_to_tile_shape(unpadded_shape, pad_c, pad_n, pad_h, pad_w);
        },
        R"doc(
            Returns shape padded to tile shape
        )doc");

    m_tensor.def(
        "dump_tensor",
        &dump_tensor,
        R"doc(
            Dump tensor to file
        )doc"
    );

    m_tensor.def(
        "load_tensor",
        &load_tensor,
        R"doc(
            Load tensor to file
        )doc"
    );

    m_tensor.def(
        "num_cores_to_corerange_set",
        py::overload_cast<const uint32_t, const CoreCoord, const bool>(&num_cores_to_corerange_set),
        R"doc(
            Create a CoreRangeSet containing the specified number of cores
        )doc"
    );

    detail::TensorModuleCompositeOPs( m_tensor);
    detail::TensorModuleBackwardOPs( m_tensor);
    detail::TensorModulePyTensor ( m_tensor);
    detail::TensorModuleDMOPs ( m_tensor);
    detail::TensorModuleCustomAndBMMOPs( m_tensor );
    detail::TensorModuleXaryOPs( m_tensor );

}

}
