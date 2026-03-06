// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include <string>

#include <tt-metalium/graph_tracking.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/graph/graph_consts.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/levelized_graph.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

std::string shape_to_string(tt::stl::Span<const uint32_t> shape) {
    std::stringstream ss;
    ss << "Shape([";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i < shape.size() - 1) {
            ss << ", ";
        }
    }
    ss << "])";
    return ss.str();
}

class TestLevelizedGraphCapture : public ttnn::TTNNFixtureWithDevice {};
TEST_F(TestLevelizedGraphCapture, SimpleBinaryOp) {
    tt::tt_metal::IDevice* device = device_;

    const auto input = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{64, 128}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);
    auto operation = [](const auto& input_tensor) {
        const auto output_tensor = ttnn::add(input_tensor, input_tensor, std::nullopt, std::nullopt);
    };

    // build reference
    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(input_tensor);
        ref_json_trace = capture.end_graph_capture();
    }

    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    // Includes: 1 tensor + 1 device operation + deallocate(s)
    EXPECT_GE(levelized_graph.size(), 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    auto tensor_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("tensor") != std::string::npos && v.in_edges.empty();
    });
    ASSERT_NE(tensor_it, levelized_graph.vertices().end());

    auto binary_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("BinaryNg") != std::string::npos || v.name.find("Binary") != std::string::npos;
    });
    ASSERT_NE(binary_it, levelized_graph.vertices().end());
    auto vertex_1 = *binary_it;

    // Tensor deduplication was removed: add(a, a) now creates two separate tensor
    // nodes.  Both in_edges of the binary op must reference tensor vertices.
    EXPECT_EQ(vertex_1.in_edges.size(), 2);
    for (auto edge_id : vertex_1.in_edges) {
        const auto& edge_vertex = levelized_graph.get_vertex(edge_id);
        EXPECT_TRUE(edge_vertex.name.find("tensor") != std::string::npos);
    }
    EXPECT_TRUE(vertex_1.out_edges.empty());
    EXPECT_FALSE(vertex_1.output_shape.empty());
    EXPECT_EQ(vertex_1.output_shape[0], shape_to_string(tt::tt_metal::Array2D{64, 128}));

    // Now get the same graph but up to level 2:
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);

    EXPECT_GE(levelized_graph_2.size(), 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));

    auto binary_it_2 = std::ranges::find_if(levelized_graph_2.vertices(), [](const auto& v) {
        return v.name.find("BinaryNg") != std::string::npos || v.name.find("Binary") != std::string::npos;
    });
    ASSERT_NE(binary_it_2, levelized_graph_2.vertices().end());
    vertex_1 = *binary_it_2;

    EXPECT_EQ(vertex_1.in_edges.size(), 2);
    for (auto edge_id : vertex_1.in_edges) {
        const auto& edge_vertex = levelized_graph_2.get_vertex(edge_id);
        EXPECT_TRUE(edge_vertex.name.find("tensor") != std::string::npos);
    }
}

TEST_F(TestLevelizedGraphCapture, ReductionOp) {
    tt::tt_metal::IDevice* device = device_;

    const auto input = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{256, 128}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);

    auto operation = [](const auto& input_tensor) {
        const auto output_tensor_1 = ttnn::sum(input_tensor, 0, true);
        const auto output_tensor_2 = ttnn::mean(input_tensor, 1, false);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(input_tensor);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 3);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    // Find the input tensor vertex
    auto input_tensor_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("tensor") != std::string::npos && v.in_edges.empty();
    });
    EXPECT_NE(input_tensor_it, levelized_graph.vertices().end());
    const auto& vertex_0 = *input_tensor_it;

    // Basic structure checks - input tensor should have output edges
    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_GE(vertex_0.out_edges.size(), 1);
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array2D{256, 128}));

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
    EXPECT_TRUE(std::ranges::none_of(levelized_graph_2.vertices(), [&](const auto& vertex) {
        return vertex.output_shape.empty() && vertex.name.find("deallocate") == std::string::npos;
    }));
}

TEST_F(TestLevelizedGraphCapture, OutputLayoutInfo) {
    tt::tt_metal::IDevice* device = device_;

    const auto input = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array3D{16, 32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);

    auto operation = [](const auto& input_tensor) {
        const auto output_tensor_1 = ttnn::sum(input_tensor, 2, true);
        const auto output_tensor_2 = ttnn::softmax(output_tensor_1, -1);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(input_tensor);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 3);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find the input tensor vertex
    auto input_tensor_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("tensor") != std::string::npos && v.in_edges.empty();
    });
    EXPECT_NE(input_tensor_it, levelized_graph.vertices().end());
    const auto& vertex_0 = *input_tensor_it;

    // Find reduction and softmax operations (they should be device operations now)
    // We can't rely on specific vertex indices, so we'll do basic structure checks

    // Basic structure checks - input tensor should have output edges
    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_GE(vertex_0.out_edges.size(), 1);
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array3D{16, 32, 64}));

    auto has_reduction = std::ranges::any_of(
        levelized_graph.vertices(), [](const auto& v) { return v.name.find("Reduce") != std::string::npos; });
    auto has_softmax = std::ranges::any_of(
        levelized_graph.vertices(), [](const auto& v) { return v.name.find("Softmax") != std::string::npos; });
    EXPECT_TRUE(has_reduction || has_softmax);  // At least one should be present

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
    EXPECT_TRUE(std::ranges::none_of(levelized_graph_2.vertices(), [&](const auto& vertex) {
        return vertex.output_shape.empty() && vertex.name.find("deallocate") == std::string::npos;
    }));
}

TEST_F(TestLevelizedGraphCapture, MatmulWithBiasTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto input = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 32}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);

    auto operation = [](const auto& input_tensor) {
        const auto output_tensor_1 = ttnn::matmul(input_tensor, input_tensor);
        const auto output_tensor_2 = ttnn::add(input_tensor, output_tensor_1, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(input_tensor);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 3);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find the input tensor vertex
    auto input_tensor_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("tensor") != std::string::npos && v.in_edges.empty();
    });
    EXPECT_NE(input_tensor_it, levelized_graph.vertices().end());
    const auto& vertex_0 = *input_tensor_it;

    // Find matmul and add operations (they should be device operations now)
    auto matmul_op_it = std::ranges::find_if(
        levelized_graph.vertices(), [](const auto& v) { return v.name.find("Matmul") != std::string::npos; });
    auto add_op_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("BinaryNg") != std::string::npos || v.name.find("Binary") != std::string::npos;
    });

    // Basic structure checks

    // Basic structure checks - input tensor should have output edges
    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_GE(vertex_0.out_edges.size(), 1);  // feeds operations
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array2D{32, 32}));

    EXPECT_TRUE(matmul_op_it != levelized_graph.vertices().end() || add_op_it != levelized_graph.vertices().end());

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
    EXPECT_TRUE(std::ranges::none_of(levelized_graph_2.vertices(), [&](const auto& vertex) {
        return vertex.output_shape.empty() && vertex.name.find("deallocate") == std::string::npos;
    }));
}

TEST_F(TestLevelizedGraphCapture, CompositeOpTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto input = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{12, 19}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);

    auto operation = [](const auto& input_tensor) { const auto output_tensor_1 = ttnn::digamma(input_tensor); };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(input_tensor);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find the input tensor vertex
    auto input_tensor_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("tensor") != std::string::npos && v.in_edges.empty();
    });
    EXPECT_NE(input_tensor_it, levelized_graph.vertices().end());
    const auto& vertex_0 = *input_tensor_it;

    // Find digamma operation (should be a device operation now)
    auto digamma_op_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("Digamma") != std::string::npos || v.name.find("Unary") != std::string::npos;
    });

    // Basic structure checks

    // Basic structure checks - input tensor should have output edges
    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_GE(vertex_0.out_edges.size(), 1);
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array2D{12, 19}));

    EXPECT_TRUE(digamma_op_it != levelized_graph.vertices().end());

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
    EXPECT_TRUE(std::ranges::none_of(levelized_graph_2.vertices(), [&](const auto& vertex) {
        return vertex.output_shape.empty() && vertex.name.find("deallocate") == std::string::npos;
    }));
}

TEST_F(TestLevelizedGraphCapture, MultiplySelfTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto input = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{64, 128}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);

    auto operation = [](const auto& input_tensor) {
        const auto output_tensor = ttnn::multiply(input_tensor, input_tensor, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(input_tensor);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 2);  // tensor, device operation
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find the input tensor vertex
    auto input_tensor_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("tensor") != std::string::npos && v.in_edges.empty();
    });
    EXPECT_NE(input_tensor_it, levelized_graph.vertices().end());
    auto vertex_0 = *input_tensor_it;

    // Find the multiply operation (should be BinaryNgDeviceOperation now)
    auto multiply_op_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("BinaryNg") != std::string::npos || v.name.find("Binary") != std::string::npos;
    });
    EXPECT_NE(multiply_op_it, levelized_graph.vertices().end());
    auto vertex_1 = *multiply_op_it;

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_GE(vertex_0.out_edges.size(), 1);
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array2D{64, 128}));

    // Tensor dedup removed: multiply(a,a) creates two separate tensor vertices.
    EXPECT_EQ(vertex_1.in_edges.size(), 2);
    for (auto edge_id : vertex_1.in_edges) {
        const auto& edge_v = levelized_graph.get_vertex(edge_id);
        EXPECT_TRUE(edge_v.name.find("tensor") != std::string::npos);
    }
    EXPECT_TRUE(vertex_1.out_edges.empty());
    EXPECT_FALSE(vertex_1.output_shape.empty());
    EXPECT_EQ(vertex_1.output_shape[0], shape_to_string(tt::tt_metal::Array2D{64, 128}));

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_GE(levelized_graph_2.size(), 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
    EXPECT_TRUE(std::ranges::none_of(levelized_graph_2.vertices(), [&](const auto& vertex) {
        return vertex.output_shape.empty() && vertex.name.find("deallocate") == std::string::npos;
    }));

    // Find vertices by name since structure changed
    auto vertex_0_it = std::ranges::find_if(levelized_graph_2.vertices(), [](const auto& v) {
        return v.name.find("tensor") != std::string::npos && v.in_edges.empty();
    });
    EXPECT_NE(vertex_0_it, levelized_graph_2.vertices().end());
    vertex_0 = *vertex_0_it;

    auto vertex_1_it = std::ranges::find_if(levelized_graph_2.vertices(), [](const auto& v) {
        return v.name.find("BinaryNg") != std::string::npos || v.name.find("Binary") != std::string::npos;
    });
    EXPECT_NE(vertex_1_it, levelized_graph_2.vertices().end());
    vertex_1 = *vertex_1_it;

    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(
        vertex_1.name.find("BinaryNg") != std::string::npos || vertex_1.name.find("Binary") != std::string::npos);

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_GE(vertex_0.out_edges.size(), 1);
    // vertex_0 should connect to vertex_1
    EXPECT_TRUE(std::ranges::find(vertex_0.out_edges, vertex_1.id) != vertex_0.out_edges.end());
    EXPECT_TRUE(vertex_0.internals.empty());

    EXPECT_EQ(vertex_1.in_edges.size(), 2);
    for (auto edge_id : vertex_1.in_edges) {
        const auto& edge_v = levelized_graph_2.get_vertex(edge_id);
        EXPECT_TRUE(edge_v.name.find("tensor") != std::string::npos);
    }
    EXPECT_TRUE(vertex_1.out_edges.empty());
}

TEST_F(TestLevelizedGraphCapture, ForkTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto input = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{64, 128}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);

    auto operation = [](const auto& input_tensor) {
        const auto output_tensor_1 = ttnn::add(input_tensor, input_tensor, std::nullopt, std::nullopt);
        const auto output_tensor_2 = ttnn::subtract(input_tensor, input_tensor, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(input_tensor);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1 - fork: one input tensor feeds multiple operations
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 3);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find the input tensor vertex
    auto input_tensor_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("tensor") != std::string::npos && v.in_edges.empty();
    });
    EXPECT_NE(input_tensor_it, levelized_graph.vertices().end());
    const auto& vertex_0 = *input_tensor_it;

    // Find add and subtract operations (they should be device operations now)
    auto add_op_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("BinaryNg") != std::string::npos || v.name.find("Binary") != std::string::npos;
    });
    EXPECT_NE(add_op_it, levelized_graph.vertices().end());
    auto vertex_1 = *add_op_it;

    // There should be at least two binary operations (add and subtract)
    auto binary_ops = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->name.find("BinaryNg") != std::string::npos || it->name.find("Binary") != std::string::npos) {
            binary_ops.push_back(it);
        }
    }
    EXPECT_GE(binary_ops.size(), 1);  // At least one binary operation

    // Find a second binary operation if it exists (subtract)
    auto vertex_2_it = std::ranges::find_if(levelized_graph.vertices(), [&vertex_1](const auto& v) {
        return (v.name.find("BinaryNg") != std::string::npos || v.name.find("Binary") != std::string::npos) &&
               v.id != vertex_1.id;
    });
    const auto& vertex_2 = vertex_2_it != levelized_graph.vertices().end() ? *vertex_2_it : vertex_1;

    // Basic structure checks - input tensor should fork to multiple operations
    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_GE(vertex_0.out_edges.size(), 1);  // forks to operations
    // vertex_0 should connect to vertex_1
    EXPECT_TRUE(std::ranges::find(vertex_0.out_edges, vertex_1.id) != vertex_0.out_edges.end());

    EXPECT_EQ(vertex_1.in_edges.size(), 2);
    for (auto edge_id : vertex_1.in_edges) {
        const auto& edge_vertex = levelized_graph.get_vertex(edge_id);
        EXPECT_TRUE(edge_vertex.name.find("tensor") != std::string::npos);
    }
    EXPECT_TRUE(vertex_1.out_edges.empty());

    if (vertex_2.id != vertex_1.id) {
        EXPECT_EQ(vertex_2.in_edges.size(), 2);
        for (auto edge_id : vertex_2.in_edges) {
            const auto& edge_vertex = levelized_graph.get_vertex(edge_id);
            EXPECT_TRUE(edge_vertex.name.find("tensor") != std::string::npos);
        }
        EXPECT_TRUE(vertex_2.out_edges.empty());
    }

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
}

TEST_F(TestLevelizedGraphCapture, JoinTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto input_a = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{64, 128}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_b = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{64, 128}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor_a = tt::tt_metal::create_device_tensor(input_a, device);
    const auto input_tensor_b = tt::tt_metal::create_device_tensor(input_b, device);

    auto operation = [](const auto& input_tensor_a, const auto& input_tensor_b) {
        const auto output_tensor = ttnn::add(input_tensor_a, input_tensor_b, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(input_tensor_a, input_tensor_b);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1 - join: one operation uses multiple different input tensors
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 3);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find the two input tensor vertices
    auto tensor_vertices = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->name.find("tensor") != std::string::npos && it->in_edges.empty()) {
            tensor_vertices.push_back(it);
        }
    }
    EXPECT_GE(tensor_vertices.size(), 2);  // At least two input tensors
    const auto& vertex_0 = *tensor_vertices[0];
    const auto& vertex_1 = *tensor_vertices[1];

    // Find the add operation (should be a device operation now)
    auto add_op_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("BinaryNg") != std::string::npos || v.name.find("Binary") != std::string::npos;
    });
    EXPECT_NE(add_op_it, levelized_graph.vertices().end());
    const auto& vertex_2 = *add_op_it;

    // Basic structure checks - both tensors should join into the operation
    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_GE(vertex_0.out_edges.size(), 1);
    // vertex_0 should connect to vertex_2
    EXPECT_TRUE(std::ranges::find(vertex_0.out_edges, vertex_2.id) != vertex_0.out_edges.end());

    EXPECT_TRUE(vertex_1.in_edges.empty());
    EXPECT_GE(vertex_1.out_edges.size(), 1);
    // vertex_1 should connect to vertex_2
    EXPECT_TRUE(std::ranges::find(vertex_1.out_edges, vertex_2.id) != vertex_1.out_edges.end());

    EXPECT_EQ(vertex_2.in_edges.size(), 2);  // joins from both vertex_0 and vertex_1
    EXPECT_EQ(vertex_2.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_2.in_edges[1], vertex_1.id);
    EXPECT_TRUE(vertex_2.out_edges.empty());

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
}

TEST_F(TestLevelizedGraphCapture, OrderOfArgs) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);

    auto operation = [](const auto& a, const auto& b) {
        const auto output_tensor_1 = ttnn::subtract(a, b);
        const auto output_tensor_2 = ttnn::subtract(b, a);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tensor_a, tensor_b);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 4);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find input tensor vertices
    auto tensor_vertices = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->name.find("tensor") != std::string::npos && it->in_edges.empty()) {
            tensor_vertices.push_back(it);
        }
    }
    EXPECT_GE(tensor_vertices.size(), 2);
    const auto& tensor_a_vertex = *tensor_vertices[0];
    const auto& tensor_b_vertex = *tensor_vertices[1];

    // Find subtract operations (should be BinaryNgDeviceOperation now)
    auto binary_ops = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->name.find("BinaryNg") != std::string::npos || it->name.find("Binary") != std::string::npos) {
            binary_ops.push_back(it);
        }
    }
    EXPECT_GE(binary_ops.size(), 2);  // At least two subtract operations
    const auto& subtract_ab = *binary_ops[0];
    const auto& subtract_ba = *binary_ops[1];

    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_b_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(
        subtract_ab.name.find("BinaryNg") != std::string::npos || subtract_ab.name.find("Binary") != std::string::npos);
    EXPECT_TRUE(
        subtract_ba.name.find("BinaryNg") != std::string::npos || subtract_ba.name.find("Binary") != std::string::npos);

    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_TRUE(tensor_b_vertex.in_edges.empty());

    // First subtract(a, b) should have 2 tensor inputs
    EXPECT_EQ(subtract_ab.in_edges.size(), 2);
    const auto& sub_ab_in0 = levelized_graph.get_vertex(subtract_ab.in_edges[0]);
    const auto& sub_ab_in1 = levelized_graph.get_vertex(subtract_ab.in_edges[1]);
    EXPECT_TRUE(sub_ab_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(sub_ab_in1.name.find("tensor") != std::string::npos);
    EXPECT_NE(sub_ab_in0.name, sub_ab_in1.name);
    EXPECT_TRUE(subtract_ab.out_edges.empty());

    // Second subtract(b, a) should have 2 tensor inputs with reversed order
    EXPECT_EQ(subtract_ba.in_edges.size(), 2);
    const auto& sub_ba_in0 = levelized_graph.get_vertex(subtract_ba.in_edges[0]);
    const auto& sub_ba_in1 = levelized_graph.get_vertex(subtract_ba.in_edges[1]);
    EXPECT_TRUE(sub_ba_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(sub_ba_in1.name.find("tensor") != std::string::npos);
    EXPECT_NE(sub_ba_in0.name, sub_ba_in1.name);
    EXPECT_TRUE(subtract_ba.out_edges.empty());

    // Verify argument order is preserved: tensor names should be swapped
    EXPECT_EQ(sub_ab_in0.name, sub_ba_in1.name);
    EXPECT_EQ(sub_ab_in1.name, sub_ba_in0.name);

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
}

TEST_F(TestLevelizedGraphCapture, OrderOfArgsIntermediateTensorTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);

    auto operation = [](const auto& a, const auto& b) {
        const auto output_tensor_1 = ttnn::add(a, a);
        const auto output_tensor_2 = ttnn::add(b, b);
        const auto output_tensor_3 = ttnn::subtract(output_tensor_1, output_tensor_2);
        const auto output_tensor_4 = ttnn::subtract(output_tensor_2, output_tensor_1);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tensor_a, tensor_b);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);

    // Should have at least 6 vertices: 2 input tensors, 2 adds, 2 subtracts (as device operations)
    EXPECT_GE(levelized_graph.size(), 6);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find binary operations (add and subtract are both BinaryNgDeviceOperation now)
    auto binary_ops = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->name.find("BinaryNg") != std::string::npos || it->name.find("Binary") != std::string::npos) {
            binary_ops.push_back(it);
        }
    }
    // The first 2 binary ops (in graph order) are the adds, the last 2 are the subtracts.
    ASSERT_GE(binary_ops.size(), 4);
    const auto& add_aa = *binary_ops[0];
    const auto& add_bb = *binary_ops[1];
    const auto& subtract_12 = *binary_ops[2];
    const auto& subtract_21 = *binary_ops[3];

    // Verify add operations have tensor inputs representing the same underlying tensor
    EXPECT_EQ(add_aa.in_edges.size(), 2);
    const auto& add_aa_in0 = levelized_graph.get_vertex(add_aa.in_edges[0]);
    const auto& add_aa_in1 = levelized_graph.get_vertex(add_aa.in_edges[1]);
    EXPECT_TRUE(add_aa_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(add_aa_in1.name.find("tensor") != std::string::npos);
    EXPECT_EQ(add_aa_in0.name, add_aa_in1.name);

    EXPECT_EQ(add_bb.in_edges.size(), 2);
    const auto& add_bb_in0 = levelized_graph.get_vertex(add_bb.in_edges[0]);
    const auto& add_bb_in1 = levelized_graph.get_vertex(add_bb.in_edges[1]);
    EXPECT_TRUE(add_bb_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(add_bb_in1.name.find("tensor") != std::string::npos);
    EXPECT_EQ(add_bb_in0.name, add_bb_in1.name);

    // add_aa and add_bb use different underlying tensors
    EXPECT_NE(add_aa_in0.name, add_bb_in0.name);

    // KEY TEST: subtract operations have 2 inputs each
    EXPECT_EQ(subtract_12.in_edges.size(), 2);
    EXPECT_EQ(subtract_21.in_edges.size(), 2);

    // Verify the subtract operations have reversed argument order by comparing input names
    const auto& s12_in0 = levelized_graph.get_vertex(subtract_12.in_edges[0]);
    const auto& s12_in1 = levelized_graph.get_vertex(subtract_12.in_edges[1]);
    const auto& s21_in0 = levelized_graph.get_vertex(subtract_21.in_edges[0]);
    const auto& s21_in1 = levelized_graph.get_vertex(subtract_21.in_edges[1]);

    EXPECT_NE(s12_in0.name, s12_in1.name);
    EXPECT_NE(s21_in0.name, s21_in1.name);
    EXPECT_EQ(s12_in0.name, s21_in1.name);
    EXPECT_EQ(s12_in1.name, s21_in0.name);

    // Both subtract operations are terminal
    EXPECT_TRUE(subtract_12.out_edges.empty());
    EXPECT_TRUE(subtract_21.out_edges.empty());

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
}

TEST_F(TestLevelizedGraphCapture, SameTensorMultipleTimes) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);

    auto operation = [](const auto& a) { const auto output = ttnn::add(a, a, std::nullopt, std::nullopt); };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tensor_a);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find input tensor vertex
    auto tensor_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("tensor") != std::string::npos && v.in_edges.empty();
    });
    EXPECT_NE(tensor_it, levelized_graph.vertices().end());
    const auto& tensor_a_vertex = *tensor_it;

    // Find add operation (should be BinaryNgDeviceOperation now)
    auto add_op_it = std::ranges::find_if(levelized_graph.vertices(), [](const auto& v) {
        return v.name.find("BinaryNg") != std::string::npos || v.name.find("Binary") != std::string::npos;
    });
    EXPECT_NE(add_op_it, levelized_graph.vertices().end());
    const auto& add_aa = *add_op_it;

    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(add_aa.name.find("BinaryNg") != std::string::npos || add_aa.name.find("Binary") != std::string::npos);

    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_GE(tensor_a_vertex.out_edges.size(), 1);
    EXPECT_EQ(tensor_a_vertex.out_edges[0], add_aa.id);

    // add(a, a) should have 2 tensor inputs (dedup removed, so different vertex IDs)
    EXPECT_EQ(add_aa.in_edges.size(), 2);
    for (auto edge_id : add_aa.in_edges) {
        const auto& edge_v = levelized_graph.get_vertex(edge_id);
        EXPECT_TRUE(edge_v.name.find("tensor") != std::string::npos);
    }
    EXPECT_TRUE(add_aa.out_edges.empty());

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
}

TEST_F(TestLevelizedGraphCapture, TernaryOpDifferentOrder) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_c = tt::tt_metal::create_device_tensor(tensor_spec, device);

    auto operation = [](const auto& a, const auto& b, const auto& c) {
        const auto output1 = ttnn::addcmul(a, b, c);
        const auto output2 = ttnn::addcmul(c, b, a);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tensor_a, tensor_b, tensor_c);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 5);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find input tensor vertices
    auto tensor_vertices = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->name.find("tensor") != std::string::npos && it->in_edges.empty()) {
            tensor_vertices.push_back(it);
        }
    }
    EXPECT_GE(tensor_vertices.size(), 3);
    const auto& tensor_a_vertex = *tensor_vertices[0];
    const auto& tensor_b_vertex = *tensor_vertices[1];
    const auto& tensor_c_vertex = *tensor_vertices[2];

    // Find ternary operations (addcmul operations - these might be composite or device operations)
    // Look for operations with 3 input edges
    auto ternary_ops = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->in_edges.size() == 3) {
            ternary_ops.push_back(it);
        }
    }
    EXPECT_GE(ternary_ops.size(), 2);
    const auto& addcmul_abc = *ternary_ops[0];
    const auto& addcmul_cba = *ternary_ops[1];

    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_b_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_c_vertex.name.find("tensor") != std::string::npos);

    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_TRUE(tensor_b_vertex.in_edges.empty());
    EXPECT_TRUE(tensor_c_vertex.in_edges.empty());

    // First addcmul(a, b, c) should have 3 tensor inputs
    EXPECT_EQ(addcmul_abc.in_edges.size(), 3);
    const auto& abc_in0 = levelized_graph.get_vertex(addcmul_abc.in_edges[0]);
    const auto& abc_in1 = levelized_graph.get_vertex(addcmul_abc.in_edges[1]);
    const auto& abc_in2 = levelized_graph.get_vertex(addcmul_abc.in_edges[2]);
    EXPECT_TRUE(abc_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(abc_in1.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(abc_in2.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(addcmul_abc.out_edges.empty());

    // Second addcmul(c, b, a) should have 3 tensor inputs with reversed a,c order
    EXPECT_EQ(addcmul_cba.in_edges.size(), 3);
    const auto& cba_in0 = levelized_graph.get_vertex(addcmul_cba.in_edges[0]);
    const auto& cba_in1 = levelized_graph.get_vertex(addcmul_cba.in_edges[1]);
    const auto& cba_in2 = levelized_graph.get_vertex(addcmul_cba.in_edges[2]);
    EXPECT_TRUE(cba_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(cba_in1.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(cba_in2.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(addcmul_cba.out_edges.empty());

    // Verify argument order: addcmul(a,b,c) vs addcmul(c,b,a) - a and c swapped, b same
    EXPECT_EQ(abc_in0.name, cba_in2.name);
    EXPECT_EQ(abc_in1.name, cba_in1.name);
    EXPECT_EQ(abc_in2.name, cba_in0.name);

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
}

TEST_F(TestLevelizedGraphCapture, TernaryOpRepeatedTensors) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);

    auto operation = [](const auto& a, const auto& b) {
        const auto output1 = ttnn::addcmul(a, b, a);
        const auto output2 = ttnn::addcmul(b, a, a);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tensor_a, tensor_b);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 4);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find input tensor vertices
    auto tensor_vertices = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->name.find("tensor") != std::string::npos && it->in_edges.empty()) {
            tensor_vertices.push_back(it);
        }
    }
    EXPECT_GE(tensor_vertices.size(), 2);
    const auto& tensor_a_vertex = *tensor_vertices[0];
    const auto& tensor_b_vertex = *tensor_vertices[1];

    // Find ternary operations (addcmul operations - these might be composite or device operations)
    // Look for operations with 3 input edges
    auto ternary_ops = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->in_edges.size() == 3) {
            ternary_ops.push_back(it);
        }
    }
    EXPECT_GE(ternary_ops.size(), 2);
    const auto& addcmul_aba = *ternary_ops[0];
    const auto& addcmul_baa = *ternary_ops[1];

    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_b_vertex.name.find("tensor") != std::string::npos);

    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_TRUE(tensor_b_vertex.in_edges.empty());

    // First addcmul(a, b, a) should have 3 tensor inputs
    EXPECT_EQ(addcmul_aba.in_edges.size(), 3);
    const auto& aba_in0 = levelized_graph.get_vertex(addcmul_aba.in_edges[0]);
    const auto& aba_in1 = levelized_graph.get_vertex(addcmul_aba.in_edges[1]);
    const auto& aba_in2 = levelized_graph.get_vertex(addcmul_aba.in_edges[2]);
    EXPECT_TRUE(aba_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(aba_in1.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(aba_in2.name.find("tensor") != std::string::npos);
    EXPECT_EQ(aba_in0.name, aba_in2.name);
    EXPECT_NE(aba_in0.name, aba_in1.name);
    EXPECT_TRUE(addcmul_aba.out_edges.empty());

    // Second addcmul(b, a, a) should have 3 tensor inputs
    EXPECT_EQ(addcmul_baa.in_edges.size(), 3);
    const auto& baa_in0 = levelized_graph.get_vertex(addcmul_baa.in_edges[0]);
    const auto& baa_in1 = levelized_graph.get_vertex(addcmul_baa.in_edges[1]);
    const auto& baa_in2 = levelized_graph.get_vertex(addcmul_baa.in_edges[2]);
    EXPECT_TRUE(baa_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(baa_in1.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(baa_in2.name.find("tensor") != std::string::npos);
    EXPECT_EQ(baa_in1.name, baa_in2.name);
    EXPECT_NE(baa_in0.name, baa_in1.name);
    EXPECT_TRUE(addcmul_baa.out_edges.empty());

    // Verify argument order: addcmul(a,b,a) vs addcmul(b,a,a)
    EXPECT_EQ(aba_in0.name, baa_in1.name);
    EXPECT_EQ(aba_in1.name, baa_in0.name);
    EXPECT_EQ(aba_in2.name, baa_in2.name);

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
}

TEST_F(TestLevelizedGraphCapture, MatmulDifferentOrders) {
    tt::tt_metal::IDevice* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{64, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);

    auto operation = [](const auto& a, const auto& b) {
        const auto output1 = ttnn::matmul(a, b);
        const auto output2 = ttnn::matmul(b, a);
        const auto output3 = ttnn::matmul(a, a);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tensor_a, tensor_b);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 5);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Find input tensor vertices
    auto tensor_vertices = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->name.find("tensor") != std::string::npos && it->in_edges.empty()) {
            tensor_vertices.push_back(it);
        }
    }
    EXPECT_GE(tensor_vertices.size(), 2);
    const auto& tensor_a_vertex = *tensor_vertices[0];
    const auto& tensor_b_vertex = *tensor_vertices[1];

    // Find matmul operations (these might be MatmulDeviceOperation or similar)
    // Look for operations with 2 input edges
    auto matmul_ops = std::vector<decltype(levelized_graph.vertices().begin())>();
    for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
        if (it->in_edges.size() == 2 &&
            (it->name.find("Matmul") != std::string::npos || it->name.find("matmul") != std::string::npos)) {
            matmul_ops.push_back(it);
        }
    }
    // If we can't find by name, look for operations with 2 inputs from our tensors
    if (matmul_ops.size() < 3) {
        matmul_ops.clear();
        // Collect tensor IDs
        std::vector<decltype(tensor_a_vertex.id)> tensor_ids;
        tensor_ids.reserve(tensor_vertices.size());
for (auto t : tensor_vertices) {
            tensor_ids.push_back(t->id);
        }
        for (auto it = levelized_graph.vertices().begin(); it != levelized_graph.vertices().end(); ++it) {
            if (it->in_edges.size() == 2 && it->in_edges[0] != it->in_edges[1]) {
                // Check if both inputs are from our input tensors
                bool both_from_inputs =
                    (std::ranges::find(tensor_ids, it->in_edges[0]) != tensor_ids.end() &&
                     std::ranges::find(tensor_ids, it->in_edges[1]) != tensor_ids.end());
                if (both_from_inputs) {
                    matmul_ops.push_back(it);
                }
            }
        }
    }
    EXPECT_GE(matmul_ops.size(), 3);
    const auto& matmul_ab = *matmul_ops[0];
    const auto& matmul_ba = *matmul_ops[1];
    const auto& matmul_aa = *matmul_ops[2];

    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_b_vertex.name.find("tensor") != std::string::npos);

    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_TRUE(tensor_b_vertex.in_edges.empty());

    // First matmul(a, b) should have 2 different tensor inputs
    EXPECT_EQ(matmul_ab.in_edges.size(), 2);
    const auto& mab_in0 = levelized_graph.get_vertex(matmul_ab.in_edges[0]);
    const auto& mab_in1 = levelized_graph.get_vertex(matmul_ab.in_edges[1]);
    EXPECT_TRUE(mab_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(mab_in1.name.find("tensor") != std::string::npos);
    EXPECT_NE(mab_in0.name, mab_in1.name);
    EXPECT_TRUE(matmul_ab.out_edges.empty());

    // Second matmul(b, a) should have 2 different tensor inputs with reversed order
    EXPECT_EQ(matmul_ba.in_edges.size(), 2);
    const auto& mba_in0 = levelized_graph.get_vertex(matmul_ba.in_edges[0]);
    const auto& mba_in1 = levelized_graph.get_vertex(matmul_ba.in_edges[1]);
    EXPECT_TRUE(mba_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(mba_in1.name.find("tensor") != std::string::npos);
    EXPECT_NE(mba_in0.name, mba_in1.name);
    EXPECT_TRUE(matmul_ba.out_edges.empty());

    // Third matmul(a, a) should have 2 tensor inputs with same underlying tensor
    EXPECT_EQ(matmul_aa.in_edges.size(), 2);
    const auto& maa_in0 = levelized_graph.get_vertex(matmul_aa.in_edges[0]);
    const auto& maa_in1 = levelized_graph.get_vertex(matmul_aa.in_edges[1]);
    EXPECT_TRUE(maa_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(maa_in1.name.find("tensor") != std::string::npos);
    EXPECT_EQ(maa_in0.name, maa_in1.name);
    EXPECT_TRUE(matmul_aa.out_edges.empty());

    // Verify argument order: matmul(a,b) vs matmul(b,a) - should be swapped
    EXPECT_EQ(mab_in0.name, mba_in1.name);
    EXPECT_EQ(mab_in1.name, mba_in0.name);
    // matmul(a,a) uses the same tensor as matmul(a,b)'s first arg
    EXPECT_EQ(maa_in0.name, mab_in0.name);

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    if (levelized_graph_2.size() > 0) {
        auto level_2_vertices = levelized_graph_2.get_vertices_at_level(2);
        if (!level_2_vertices.empty()) {
            EXPECT_TRUE(
                std::ranges::all_of(level_2_vertices, [&](const auto& vertex) { return vertex.internals.empty(); }));
        }
    }
}

TEST_F(TestLevelizedGraphCapture, ExtractLevelizedGraphJsonTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto input = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);

    auto operation = [](const auto& input_tensor) {
        const auto output_tensor = ttnn::add(input_tensor, input_tensor, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(input_tensor);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test extract_levelized_graph API - level 1
    auto levelized_graph_json = ttnn::graph::extract_levelized_graph(ref_json_trace, 1);

    // Verify JSON structure
    EXPECT_TRUE(levelized_graph_json.is_array());
    EXPECT_GE(levelized_graph_json.size(), 2);

    // Verify first vertex is a tensor with expected fields
    EXPECT_TRUE(levelized_graph_json[0].is_object());
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kCounter));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kStackingLevel));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kName));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kInEdges));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kOutEdges));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kInternals));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kOutputShape));

    EXPECT_TRUE(levelized_graph_json[0][ttnn::graph::kName].get<std::string>().find("tensor") != std::string::npos);
    EXPECT_EQ(levelized_graph_json[0][ttnn::graph::kStackingLevel], 1);
    EXPECT_EQ(levelized_graph_json[0][ttnn::graph::kCounter], 0);
    EXPECT_TRUE(levelized_graph_json[0][ttnn::graph::kInEdges].is_array());
    EXPECT_TRUE(levelized_graph_json[0][ttnn::graph::kInEdges].empty());
    EXPECT_TRUE(levelized_graph_json[0][ttnn::graph::kOutEdges].is_array());
    EXPECT_GE(levelized_graph_json[0][ttnn::graph::kOutEdges].size(), 1);
    EXPECT_TRUE(levelized_graph_json[0][ttnn::graph::kInternals].is_array());
    EXPECT_TRUE(levelized_graph_json[0][ttnn::graph::kInternals].empty());
    EXPECT_TRUE(levelized_graph_json[0][ttnn::graph::kOutputShape].is_array());
    EXPECT_FALSE(levelized_graph_json[0][ttnn::graph::kOutputShape].empty());

    // Find the BinaryNgDeviceOperation vertex by name (index may vary with dedup removed)
    auto binary_json_it = std::ranges::find_if(
        levelized_graph_json, [](const auto& v) { return v[ttnn::graph::kName] == "BinaryNgDeviceOperation"; });
    ASSERT_NE(binary_json_it, levelized_graph_json.end());
    const auto& binary_json = *binary_json_it;

    EXPECT_EQ(binary_json[ttnn::graph::kStackingLevel], 1);
    EXPECT_TRUE(binary_json[ttnn::graph::kInEdges].is_array());
    EXPECT_EQ(binary_json[ttnn::graph::kInEdges].size(), 2);
    for (const auto& edge_id : binary_json[ttnn::graph::kInEdges]) {
        auto& tensor_v = levelized_graph_json[edge_id.get<int>()];
        EXPECT_TRUE(tensor_v[ttnn::graph::kName].get<std::string>().find("tensor") != std::string::npos);
    }
    EXPECT_TRUE(binary_json[ttnn::graph::kOutEdges].is_array());
    EXPECT_TRUE(binary_json[ttnn::graph::kOutEdges].empty());
    EXPECT_TRUE(binary_json[ttnn::graph::kInternals].is_array());
    EXPECT_TRUE(binary_json[ttnn::graph::kInternals].empty());
    EXPECT_TRUE(binary_json[ttnn::graph::kOutputShape].is_array());
    EXPECT_FALSE(binary_json[ttnn::graph::kOutputShape].empty());

    // Test extract_levelized_graph API - level 2
    auto levelized_graph_json_2 = ttnn::graph::extract_levelized_graph(ref_json_trace, 2);

    // Verify JSON structure for level 2
    EXPECT_TRUE(levelized_graph_json_2.is_array());
    EXPECT_GE(levelized_graph_json_2.size(), 3);  // At least 3 vertices at level 2

    // Verify that BinaryNgDeviceOperation vertex has internals at level 2
    auto add_vertex_it = std::ranges::find_if(
        levelized_graph_json_2, [](const auto& v) { return v[ttnn::graph::kName] == "BinaryNgDeviceOperation"; });
    EXPECT_NE(add_vertex_it, levelized_graph_json_2.end());
    const auto& add_vertex = *add_vertex_it;
    EXPECT_TRUE(add_vertex.contains(ttnn::graph::kInternals));
    EXPECT_TRUE(add_vertex[ttnn::graph::kInternals].is_array());
    EXPECT_FALSE(add_vertex[ttnn::graph::kInternals].empty());

    // Verify that primitives at level 2 have no internals
    for (const auto& vertex : levelized_graph_json_2) {
        if (vertex[ttnn::graph::kStackingLevel] == 2) {
            EXPECT_TRUE(vertex.contains(ttnn::graph::kInternals));
            EXPECT_TRUE(vertex[ttnn::graph::kInternals].is_array());
            EXPECT_TRUE(vertex[ttnn::graph::kInternals].empty());
        }
    }
}

TEST_F(TestLevelizedGraphCapture, MultiplyAndAddTest) {
    tt::tt_metal::IDevice* device = device_;

    const auto input_a = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor_a = tt::tt_metal::create_device_tensor(input_a, device);
    const auto input_b = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor_b = tt::tt_metal::create_device_tensor(input_b, device);
    const auto input_c = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::L1_MEMORY_CONFIG));
    const auto input_tensor_c = tt::tt_metal::create_device_tensor(input_c, device);

    auto operation = [](const auto& input_tensor_a, const auto& input_tensor_b, const auto& input_tensor_c) {
        const auto output_tensor = ttnn::multiply(input_tensor_b, input_tensor_c, std::nullopt, std::nullopt);
        const auto output_tensor_1 = ttnn::add(input_tensor_a, output_tensor, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(input_tensor_a, input_tensor_b, input_tensor_c);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test extract_levelized_graph API - level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_GE(levelized_graph.size(), 5);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));
    EXPECT_TRUE(std::ranges::none_of(levelized_graph.vertices(), [&](const auto& vertex) {
        return vertex.output_shape.empty() && vertex.name.find("deallocate") == std::string::npos;
    }));

    // Find the BinaryNgDeviceOperation vertices by name (indices may vary)
    std::vector<size_t> binary_op_ids;
    for (size_t i = 0; i < levelized_graph.size(); ++i) {
        const auto& v = levelized_graph.get_vertex(i);
        if (v.name == "BinaryNgDeviceOperation") {
            binary_op_ids.push_back(i);
        }
    }
    ASSERT_EQ(binary_op_ids.size(), 2);

    // The first binary op (in graph order) is multiply, the second is add
    const auto& v_multiply = levelized_graph.get_vertex(binary_op_ids[0]);
    const auto& v_add = levelized_graph.get_vertex(binary_op_ids[1]);

    // Multiply should have 2 tensor inputs (b and c)
    // Motivated by this issue: https://github.com/tenstorrent/tt-mlir/issues/5929
    EXPECT_EQ(v_multiply.in_edges.size(), 2);
    const auto& mul_in0 = levelized_graph.get_vertex(v_multiply.in_edges[0]);
    const auto& mul_in1 = levelized_graph.get_vertex(v_multiply.in_edges[1]);
    EXPECT_TRUE(mul_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(mul_in1.name.find("tensor") != std::string::npos);
    EXPECT_NE(mul_in0.name, mul_in1.name);

    // Add should have 2 inputs: one tensor (a) and one related to multiply's output
    EXPECT_EQ(v_add.in_edges.size(), 2);
    bool found_tensor_a = false;
    for (auto edge_id : v_add.in_edges) {
        const auto& edge_v = levelized_graph.get_vertex(edge_id);
        if (edge_v.name.find("tensor") != std::string::npos && edge_v.name != mul_in0.name &&
            edge_v.name != mul_in1.name) {
            found_tensor_a = true;
        }
    }
    EXPECT_TRUE(found_tensor_a);
    EXPECT_NE(v_add.in_edges[0], v_add.in_edges[1]);

    // Add should be the final operation
    EXPECT_TRUE(v_add.out_edges.empty());
}

TEST_F(TestLevelizedGraphCapture, MultiplyAndAddWithCapturedTensorsTest) {
    tt::tt_metal::IDevice* device = device_;

    // This test demonstrates that when create_device_tensor is captured (tensors created INSIDE the capture),
    // the levelized graph includes BOTH the create_device_tensor operations AND the tensor nodes.
    // The key difference from MultiplyAndAddTest is that tensor nodes now have in_edges pointing to
    // their producer create_device_tensor operations, rather than having empty in_edges.

    auto operation = [&device]() {
        // Create tensors INSIDE the capture
        const auto input_a = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor_a = tt::tt_metal::create_device_tensor(input_a, device);

        const auto input_b = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor_b = tt::tt_metal::create_device_tensor(input_b, device);

        const auto input_c = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor_c = tt::tt_metal::create_device_tensor(input_c, device);

        // Perform operations
        const auto output_tensor = ttnn::multiply(input_tensor_b, input_tensor_c, std::nullopt, std::nullopt);
        const auto output_tensor_1 = ttnn::add(input_tensor_a, output_tensor, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation();
        ref_json_trace = capture.end_graph_capture();
    }

    // Test extract_levelized_graph API - level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);

    EXPECT_GE(levelized_graph.size(), 8);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));
    EXPECT_TRUE(std::ranges::none_of(levelized_graph.vertices(), [&](const auto& vertex) {
        return vertex.output_shape.empty() && vertex.name.find("deallocate") == std::string::npos;
    }));

    // Categorize vertices by type
    std::vector<size_t> create_tensor_ids;
    std::vector<size_t> tensor_ids;
    std::vector<size_t> binary_op_ids;  // All BinaryNgDeviceOperation instances
    size_t multiply_id = 0, add_id = 0;

    for (size_t i = 0; i < levelized_graph.size(); ++i) {
        const auto& v = levelized_graph.get_vertex(i);
        if (v.name == "tt::tt_metal::create_device_tensor") {
            create_tensor_ids.push_back(i);
        } else if (v.name.find("tensor[") != std::string::npos) {
            tensor_ids.push_back(i);
        } else if (v.name == "BinaryNgDeviceOperation") {
            // BinaryNgDeviceOperation
            binary_op_ids.push_back(i);
        }
    }

    EXPECT_EQ(create_tensor_ids.size(), 3);
    EXPECT_GE(tensor_ids.size(), 3);
    EXPECT_EQ(binary_op_ids.size(), 2);

    // The first binary op (in graph order) is multiply, the second is add
    multiply_id = binary_op_ids[0];
    add_id = binary_op_ids[1];
    const auto& v_multiply = levelized_graph.get_vertex(multiply_id);
    const auto& v_add = levelized_graph.get_vertex(add_id);

    // Multiply should have 2 tensor inputs (b and c)
    EXPECT_EQ(v_multiply.in_edges.size(), 2);
    const auto& mul_in0 = levelized_graph.get_vertex(v_multiply.in_edges[0]);
    const auto& mul_in1 = levelized_graph.get_vertex(v_multiply.in_edges[1]);
    EXPECT_TRUE(mul_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(mul_in1.name.find("tensor") != std::string::npos);
    EXPECT_NE(mul_in0.name, mul_in1.name);

    // Add should have 2 inputs
    EXPECT_EQ(v_add.in_edges.size(), 2);
    bool found_tensor_a = false;
    for (auto edge_id : v_add.in_edges) {
        const auto& edge_v = levelized_graph.get_vertex(edge_id);
        if (edge_v.name.find("tensor") != std::string::npos && edge_v.name != mul_in0.name &&
            edge_v.name != mul_in1.name) {
            found_tensor_a = true;
        }
    }
    EXPECT_TRUE(found_tensor_a);

    // Add is the final operation
    EXPECT_TRUE(v_add.out_edges.empty());

    // Verify create_device_tensor operations exist
    for (size_t create_id : create_tensor_ids) {
        const auto& create_vertex = levelized_graph.get_vertex(create_id);
        EXPECT_FALSE(create_vertex.output_shape.empty());
    }
}

TEST_F(TestLevelizedGraphCapture, SubtractArgumentOrderWithCapturedTensorsTest) {
    tt::tt_metal::IDevice* device = device_;

    // This test verifies that when tensors are created within the capture,
    // the graph correctly tracks the argument order for non-commutative operations like subtract.
    // We test subtract(a, b) vs subtract(b, a) to ensure the order is preserved.

    auto operation = [&device]() {
        // Create tensors INSIDE the capture
        const auto tensor_spec = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));

        const auto tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
        const auto tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);

        // Perform subtract operations in different orders
        const auto output_tensor_1 = ttnn::subtract(tensor_a, tensor_b, std::nullopt, std::nullopt);
        const auto output_tensor_2 = ttnn::subtract(tensor_b, tensor_a, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation();
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);

    EXPECT_GE(levelized_graph.size(), 6);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Categorize vertices by type
    std::vector<size_t> create_tensor_ids;
    std::vector<size_t> tensor_ids;
    std::vector<size_t> subtract_ids;

    for (size_t i = 0; i < levelized_graph.size(); ++i) {
        const auto& v = levelized_graph.get_vertex(i);
        if (v.name == "tt::tt_metal::create_device_tensor") {
            create_tensor_ids.push_back(i);
        } else if (v.name.find("tensor[") != std::string::npos) {
            tensor_ids.push_back(i);
        } else if (v.name == "BinaryNgDeviceOperation") {
            subtract_ids.push_back(i);
        }
    }

    EXPECT_EQ(create_tensor_ids.size(), 2);
    EXPECT_GE(tensor_ids.size(), 2);
    EXPECT_EQ(subtract_ids.size(), 2);

    const auto& subtract_1 = levelized_graph.get_vertex(subtract_ids[0]);
    const auto& subtract_2 = levelized_graph.get_vertex(subtract_ids[1]);

    EXPECT_EQ(subtract_1.in_edges.size(), 2);
    EXPECT_EQ(subtract_2.in_edges.size(), 2);

    // Verify inputs are tensor vertices and check argument order by name
    const auto& s1_in0 = levelized_graph.get_vertex(subtract_1.in_edges[0]);
    const auto& s1_in1 = levelized_graph.get_vertex(subtract_1.in_edges[1]);
    EXPECT_TRUE(s1_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(s1_in1.name.find("tensor") != std::string::npos);
    EXPECT_NE(s1_in0.name, s1_in1.name);

    const auto& s2_in0 = levelized_graph.get_vertex(subtract_2.in_edges[0]);
    const auto& s2_in1 = levelized_graph.get_vertex(subtract_2.in_edges[1]);
    EXPECT_TRUE(s2_in0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(s2_in1.name.find("tensor") != std::string::npos);
    EXPECT_NE(s2_in0.name, s2_in1.name);

    // Argument order is preserved: names should be swapped
    EXPECT_EQ(s1_in0.name, s2_in1.name);
    EXPECT_EQ(s1_in1.name, s2_in0.name);

    EXPECT_TRUE(subtract_1.out_edges.empty());
    EXPECT_TRUE(subtract_2.out_edges.empty());
}
