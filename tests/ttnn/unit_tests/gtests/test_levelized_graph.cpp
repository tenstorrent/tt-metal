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
    // Includes: 1 tensor + 1 add operation
    EXPECT_EQ(levelized_graph.size(), 2);
    // invariants: all vertices should be at level 1 and with no internals:
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    auto vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_EQ(vertex_1.name, "ttnn::add");

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_EQ(vertex_0.out_edges.size(), 1);
    EXPECT_EQ(vertex_0.out_edges[0], vertex_1.id);
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array2D{64, 128}));
    EXPECT_TRUE(vertex_0.internals.empty());

    EXPECT_EQ(vertex_1.in_edges.size(), 2);  // 2 in edges since we're re-using the same tensor.
    EXPECT_EQ(vertex_1.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_1.in_edges[1], vertex_0.id);
    EXPECT_TRUE(vertex_1.out_edges.empty());
    EXPECT_TRUE(vertex_1.internals.empty());
    EXPECT_FALSE(vertex_1.output_shape.empty());
    EXPECT_EQ(vertex_1.output_shape[0], shape_to_string(tt::tt_metal::Array2D{64, 128}));

    // Now get the same graph but up to level 2:
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);

    EXPECT_EQ(levelized_graph_2.size(), 3);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));
    EXPECT_TRUE(std::ranges::none_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.output_shape.empty(); }));

    vertex_0 = levelized_graph_2.get_vertex(0);
    vertex_1 = levelized_graph_2.get_vertex(1);
    const auto& vertex_2 = levelized_graph_2.get_vertex(2);
    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_EQ(vertex_1.name, "ttnn::add");
    EXPECT_EQ(vertex_2.name, "BinaryNgDeviceOperation");

    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array2D{64, 128}));
    EXPECT_EQ(vertex_1.output_shape[0], shape_to_string(tt::tt_metal::Array2D{64, 128}));
    EXPECT_EQ(vertex_2.output_shape[0], shape_to_string(tt::tt_metal::Array2D{64, 128}));

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_EQ(vertex_0.out_edges.size(), 2);
    EXPECT_EQ(vertex_0.out_edges[0], vertex_1.id);
    EXPECT_EQ(vertex_0.out_edges[1], vertex_2.id);
    EXPECT_TRUE(vertex_0.internals.empty());

    EXPECT_EQ(vertex_1.in_edges.size(), 2);  // 2 in edges since we're re-using the same tensor.
    EXPECT_EQ(vertex_1.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_1.in_edges[1], vertex_0.id);
    EXPECT_TRUE(vertex_1.out_edges.empty());
    EXPECT_EQ(vertex_1.internals.size(), 1);  // we should have expanded ttnn::add to BinaryNgDeviceOperation
    EXPECT_EQ(vertex_1.internals[0], vertex_2.id);

    EXPECT_EQ(vertex_2.in_edges.size(), 2);
    EXPECT_EQ(vertex_2.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_2.in_edges[1], vertex_0.id);
    EXPECT_TRUE(vertex_2.out_edges.empty());
    EXPECT_TRUE(vertex_2.internals.empty());
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
    EXPECT_EQ(levelized_graph.size(), 3);  // tensor, sum, mean
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    const auto& vertex_2 = levelized_graph.get_vertex(2);
    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_EQ(vertex_1.name, "ttnn::sum");
    EXPECT_EQ(vertex_2.name, "ttnn::mean");

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_EQ(vertex_0.out_edges.size(), 2);
    EXPECT_EQ(vertex_0.out_edges[0], vertex_1.id);
    EXPECT_EQ(vertex_0.out_edges[1], vertex_2.id);
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array2D{256, 128}));

    EXPECT_EQ(vertex_1.in_edges.size(), 1);
    EXPECT_EQ(vertex_1.in_edges[0], vertex_0.id);
    EXPECT_TRUE(vertex_1.out_edges.empty());
    EXPECT_FALSE(vertex_1.output_shape.empty());

    EXPECT_EQ(vertex_2.in_edges.size(), 1);
    EXPECT_EQ(vertex_2.in_edges[0], vertex_0.id);
    EXPECT_TRUE(vertex_2.out_edges.empty());
    EXPECT_FALSE(vertex_2.output_shape.empty());

    EXPECT_EQ(vertex_1.output_shape[0], shape_to_string(tt::tt_metal::Array2D{1, 128}));
    EXPECT_EQ(vertex_2.output_shape[0], shape_to_string(tt::tt_metal::Array1D{256}));

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));
    EXPECT_TRUE(std::ranges::none_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.output_shape.empty(); }));

    // At level 2, sum should have internals
    vertex_1 = levelized_graph_2.get_vertex(1);
    EXPECT_FALSE(vertex_1.internals.empty());
    EXPECT_EQ(levelized_graph_2.get_vertex(vertex_1.internals[0]).name, "ttnn::fill_implicit_tile_padding");
    EXPECT_EQ(levelized_graph_2.get_vertex(vertex_1.internals[1]).name, "ttnn::reshape");
    EXPECT_EQ(levelized_graph_2.get_vertex(vertex_1.internals[2]).name, "ttnn::tilize_with_val_padding");
    EXPECT_EQ(levelized_graph_2.get_vertex(vertex_1.internals[3]).name, "ReduceDeviceOperation");
    EXPECT_EQ(levelized_graph_2.get_vertex(vertex_1.internals[4]).name, "ttnn::reshape");
    // Find mean vertex by name
    auto mean_vertex_it =
        std::ranges::find_if(levelized_graph_2.vertices(), [](const auto& v) { return v.name == "ttnn::mean"; });
    EXPECT_NE(mean_vertex_it, levelized_graph_2.vertices().end());
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->id).name, "ttnn::mean");
    EXPECT_EQ(
        levelized_graph_2.get_vertex(mean_vertex_it->id).output_shape[0], shape_to_string(tt::tt_metal::Array1D{256}));
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->id).internals.size(), 5);
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->internals[0]).name, "ttnn::fill_implicit_tile_padding");
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->internals[1]).name, "ttnn::reshape");
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->internals[2]).name, "ttnn::tilize_with_val_padding");
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->internals[3]).name, "ReduceDeviceOperation");
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->internals[4]).name, "ttnn::reshape");
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
    EXPECT_EQ(levelized_graph.size(), 3);  // tensor, sum, softmax
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    const auto& vertex_2 = levelized_graph.get_vertex(2);
    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_EQ(vertex_1.name, "ttnn::sum");
    EXPECT_EQ(vertex_2.name, "ttnn::softmax");

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_EQ(vertex_0.out_edges.size(), 1);
    EXPECT_EQ(vertex_0.out_edges[0], vertex_1.id);
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array3D{16, 32, 64}));

    EXPECT_EQ(vertex_1.in_edges.size(), 1);
    EXPECT_EQ(vertex_1.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_1.out_edges.size(), 1);
    EXPECT_EQ(vertex_1.out_edges[0], vertex_2.id);
    EXPECT_FALSE(vertex_1.output_shape.empty());
    EXPECT_EQ(vertex_1.output_shape[0], shape_to_string(tt::tt_metal::Array3D{16, 32, 1}));
    auto vertex_1_output_layout_info = vertex_1.output_info[0];
    EXPECT_NE(
        vertex_1_output_layout_info.find(
            "memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1"),
        std::string::npos);
    EXPECT_NE(vertex_1_output_layout_info.find("logical_shape=Shape([16, 32, 1])"), std::string::npos);

    EXPECT_EQ(vertex_2.in_edges.size(), 1);
    EXPECT_EQ(vertex_2.in_edges[0], vertex_1.id);
    EXPECT_TRUE(vertex_2.out_edges.empty());
    EXPECT_FALSE(vertex_2.output_shape.empty());
    EXPECT_EQ(vertex_2.output_shape[0], shape_to_string(tt::tt_metal::Array3D{16, 32, 1}));
    // Note that the output info for vertex_2 (softmax) cannot be populated since it has no consumer.

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));
    EXPECT_TRUE(std::ranges::none_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.output_shape.empty(); }));

    // At level 2, sum should have internals
    vertex_1 = levelized_graph_2.get_vertex(1);
    EXPECT_FALSE(vertex_1.internals.empty());
    // Find softmax vertex by name
    auto softmax_vertex_it =
        std::ranges::find_if(levelized_graph_2.vertices(), [](const auto& v) { return v.name == "ttnn::softmax"; });
    EXPECT_NE(softmax_vertex_it, levelized_graph_2.vertices().end());
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
    EXPECT_EQ(levelized_graph.size(), 3);  // tensor, matmul, add
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    const auto& vertex_2 = levelized_graph.get_vertex(2);
    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_EQ(vertex_1.name, "ttnn::matmul");
    EXPECT_EQ(vertex_2.name, "ttnn::add");

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_EQ(vertex_0.out_edges.size(), 2);  // feeds both matmul and add
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array2D{32, 32}));

    EXPECT_EQ(vertex_1.in_edges.size(), 2);  // both inputs are the same tensor
    EXPECT_EQ(vertex_1.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_1.in_edges[1], vertex_0.id);
    EXPECT_EQ(vertex_1.out_edges.size(), 1);
    EXPECT_EQ(vertex_1.out_edges[0], vertex_2.id);
    EXPECT_FALSE(vertex_1.output_shape.empty());
    EXPECT_EQ(vertex_1.output_shape[0], shape_to_string(tt::tt_metal::Array2D{32, 32}));

    EXPECT_EQ(vertex_2.in_edges.size(), 2);  // input_tensor and output_tensor_1
    EXPECT_EQ(vertex_2.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_2.in_edges[1], vertex_1.id);
    EXPECT_TRUE(vertex_2.out_edges.empty());
    EXPECT_FALSE(vertex_2.output_shape.empty());
    EXPECT_EQ(vertex_2.output_shape[0], shape_to_string(tt::tt_metal::Array2D{32, 32}));

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));
    EXPECT_TRUE(std::ranges::none_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.output_shape.empty(); }));

    // At level 2, matmul should have internals
    vertex_1 = levelized_graph_2.get_vertex(1);
    EXPECT_FALSE(vertex_1.internals.empty());
    // Find add vertex by name
    auto add_vertex_it =
        std::ranges::find_if(levelized_graph_2.vertices(), [](const auto& v) { return v.name == "ttnn::add"; });
    EXPECT_NE(add_vertex_it, levelized_graph_2.vertices().end());
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
    EXPECT_EQ(levelized_graph.size(), 2);  // tensor, digamma
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_EQ(vertex_1.name, "ttnn::digamma");

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_EQ(vertex_0.out_edges.size(), 1);
    EXPECT_EQ(vertex_0.out_edges[0], vertex_1.id);
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array2D{12, 19}));

    EXPECT_EQ(vertex_1.in_edges.size(), 1);
    EXPECT_EQ(vertex_1.in_edges[0], vertex_0.id);
    EXPECT_TRUE(vertex_1.out_edges.empty());
    EXPECT_FALSE(vertex_1.output_shape.empty());

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));
    EXPECT_TRUE(std::ranges::none_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.output_shape.empty(); }));

    // At level 2, digamma (composite op) should have internals
    vertex_1 = levelized_graph_2.get_vertex(1);
    EXPECT_FALSE(vertex_1.internals.empty());
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
    EXPECT_EQ(levelized_graph.size(), 2);  // tensor, multiply
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    auto vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_EQ(vertex_1.name, "ttnn::multiply");

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_EQ(vertex_0.out_edges.size(), 1);
    EXPECT_EQ(vertex_0.out_edges[0], vertex_1.id);
    EXPECT_EQ(vertex_0.output_shape[0], shape_to_string(tt::tt_metal::Array2D{64, 128}));

    EXPECT_EQ(vertex_1.in_edges.size(), 2);  // 2 in edges since we're re-using the same tensor
    EXPECT_EQ(vertex_1.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_1.in_edges[1], vertex_0.id);
    EXPECT_TRUE(vertex_1.out_edges.empty());
    EXPECT_TRUE(vertex_1.internals.empty());
    EXPECT_FALSE(vertex_1.output_shape.empty());
    EXPECT_EQ(vertex_1.output_shape[0], shape_to_string(tt::tt_metal::Array2D{64, 128}));

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_EQ(levelized_graph_2.size(), 3);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));
    EXPECT_TRUE(std::ranges::none_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.output_shape.empty(); }));

    vertex_0 = levelized_graph_2.get_vertex(0);
    vertex_1 = levelized_graph_2.get_vertex(1);
    const auto& vertex_2 = levelized_graph_2.get_vertex(2);
    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_EQ(vertex_1.name, "ttnn::multiply");
    EXPECT_EQ(vertex_2.name, "BinaryNgDeviceOperation");

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_EQ(vertex_0.out_edges.size(), 2);
    EXPECT_EQ(vertex_0.out_edges[0], vertex_1.id);
    EXPECT_EQ(vertex_0.out_edges[1], vertex_2.id);
    EXPECT_TRUE(vertex_0.internals.empty());

    EXPECT_EQ(vertex_1.in_edges.size(), 2);
    EXPECT_EQ(vertex_1.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_1.in_edges[1], vertex_0.id);
    EXPECT_TRUE(vertex_1.out_edges.empty());
    EXPECT_EQ(vertex_1.internals.size(), 1);
    EXPECT_EQ(vertex_1.internals[0], vertex_2.id);

    EXPECT_EQ(vertex_2.in_edges.size(), 2);
    EXPECT_EQ(vertex_2.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_2.in_edges[1], vertex_0.id);
    EXPECT_TRUE(vertex_2.out_edges.empty());
    EXPECT_TRUE(vertex_2.internals.empty());
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
    EXPECT_EQ(levelized_graph.size(), 3);  // tensor, add, subtract
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    const auto& vertex_2 = levelized_graph.get_vertex(2);
    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_EQ(vertex_1.name, "ttnn::add");
    EXPECT_EQ(vertex_2.name, "ttnn::subtract");

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_EQ(vertex_0.out_edges.size(), 2);  // forks to both add and subtract
    EXPECT_EQ(vertex_0.out_edges[0], vertex_1.id);
    EXPECT_EQ(vertex_0.out_edges[1], vertex_2.id);

    EXPECT_EQ(vertex_1.in_edges.size(), 2);  // both inputs from vertex_0
    EXPECT_EQ(vertex_1.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_1.in_edges[1], vertex_0.id);
    EXPECT_TRUE(vertex_1.out_edges.empty());

    EXPECT_EQ(vertex_2.in_edges.size(), 2);  // both inputs from vertex_0
    EXPECT_EQ(vertex_2.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_2.in_edges[1], vertex_0.id);
    EXPECT_TRUE(vertex_2.out_edges.empty());

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));

    // At level 2, add should have internals
    vertex_1 = levelized_graph_2.get_vertex(1);
    EXPECT_FALSE(vertex_1.internals.empty());
    // Find subtract vertex by name
    auto subtract_vertex_it =
        std::ranges::find_if(levelized_graph_2.vertices(), [](const auto& v) { return v.name == "ttnn::subtract"; });
    EXPECT_NE(subtract_vertex_it, levelized_graph_2.vertices().end());
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
    EXPECT_EQ(levelized_graph.size(), 3);  // tensor (a), tensor (b), add
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    const auto& vertex_1 = levelized_graph.get_vertex(1);
    auto vertex_2 = levelized_graph.get_vertex(2);
    EXPECT_TRUE(vertex_0.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(vertex_1.name.find("tensor") != std::string::npos);
    EXPECT_EQ(vertex_2.name, "ttnn::add");

    EXPECT_TRUE(vertex_0.in_edges.empty());
    EXPECT_EQ(vertex_0.out_edges.size(), 1);
    EXPECT_EQ(vertex_0.out_edges[0], vertex_2.id);

    EXPECT_TRUE(vertex_1.in_edges.empty());
    EXPECT_EQ(vertex_1.out_edges.size(), 1);
    EXPECT_EQ(vertex_1.out_edges[0], vertex_2.id);

    EXPECT_EQ(vertex_2.in_edges.size(), 2);  // joins from both vertex_0 and vertex_1
    EXPECT_EQ(vertex_2.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_2.in_edges[1], vertex_1.id);
    EXPECT_TRUE(vertex_2.out_edges.empty());

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));

    // At level 2, add should have internals
    vertex_2 = levelized_graph_2.get_vertex(2);
    EXPECT_FALSE(vertex_2.internals.empty());
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
    EXPECT_EQ(levelized_graph.size(), 4);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    const auto& tensor_a_vertex = levelized_graph.get_vertex(0);
    const auto& tensor_b_vertex = levelized_graph.get_vertex(1);
    const auto& subtract_ab = levelized_graph.get_vertex(2);
    const auto& subtract_ba = levelized_graph.get_vertex(3);

    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_b_vertex.name.find("tensor") != std::string::npos);
    EXPECT_EQ(subtract_ab.name, "ttnn::subtract");
    EXPECT_EQ(subtract_ba.name, "ttnn::subtract");

    // Both tensors connect to both subtract operations
    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_EQ(tensor_a_vertex.out_edges.size(), 2);
    EXPECT_TRUE(tensor_b_vertex.in_edges.empty());
    EXPECT_EQ(tensor_b_vertex.out_edges.size(), 2);

    // First subtract(a, b) should have in_edges [a, b]
    EXPECT_EQ(subtract_ab.in_edges.size(), 2);
    EXPECT_EQ(subtract_ab.in_edges[0], tensor_a_vertex.id);
    EXPECT_EQ(subtract_ab.in_edges[1], tensor_b_vertex.id);
    EXPECT_TRUE(subtract_ab.out_edges.empty());

    // Second subtract(b, a) should have in_edges [b, a]
    EXPECT_EQ(subtract_ba.in_edges.size(), 2);
    EXPECT_EQ(subtract_ba.in_edges[0], tensor_b_vertex.id);
    EXPECT_EQ(subtract_ba.in_edges[1], tensor_a_vertex.id);
    EXPECT_TRUE(subtract_ba.out_edges.empty());

    // Verify that the two subtract operations have different input orders
    EXPECT_NE(subtract_ab.in_edges, subtract_ba.in_edges);

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));

    // At level 2, at least one subtract operation should have internals
    const auto& subtract_ab_l2 = levelized_graph_2.get_vertex(2);
    EXPECT_FALSE(subtract_ab_l2.internals.empty());
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

    // Should have 6 vertices: 2 input tensors, 2 adds, 2 subtracts
    EXPECT_EQ(levelized_graph.size(), 6);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    // Get vertices
    const auto& tensor_a_vertex = levelized_graph.get_vertex(0);
    const auto& tensor_b_vertex = levelized_graph.get_vertex(1);
    const auto& add_aa = levelized_graph.get_vertex(2);       // add(a, a)
    const auto& add_bb = levelized_graph.get_vertex(3);       // add(b, b)
    const auto& subtract_12 = levelized_graph.get_vertex(4);  // subtract(output_tensor_1, output_tensor_2)
    const auto& subtract_21 = levelized_graph.get_vertex(5);  // subtract(output_tensor_2, output_tensor_1)

    // Verify tensor names
    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_b_vertex.name.find("tensor") != std::string::npos);
    EXPECT_EQ(add_aa.name, "ttnn::add");
    EXPECT_EQ(add_bb.name, "ttnn::add");
    EXPECT_EQ(subtract_12.name, "ttnn::subtract");
    EXPECT_EQ(subtract_21.name, "ttnn::subtract");

    // Verify input tensors have no incoming edges (they are runtime inputs)
    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_TRUE(tensor_b_vertex.in_edges.empty());

    // Verify first add(a, a) has correct inputs
    EXPECT_EQ(add_aa.in_edges.size(), 2);
    EXPECT_EQ(add_aa.in_edges[0], tensor_a_vertex.id);
    EXPECT_EQ(add_aa.in_edges[1], tensor_a_vertex.id);

    // Verify second add(b, b) has correct inputs
    EXPECT_EQ(add_bb.in_edges.size(), 2);
    EXPECT_EQ(add_bb.in_edges[0], tensor_b_vertex.id);
    EXPECT_EQ(add_bb.in_edges[1], tensor_b_vertex.id);

    // KEY TEST: Verify first subtract(output_tensor_1, output_tensor_2) has correct argument order
    // This tests that intermediate tensors maintain their order
    EXPECT_EQ(subtract_12.in_edges.size(), 2);
    EXPECT_EQ(subtract_12.in_edges[0], add_aa.id);  // First arg: output_tensor_1 = add(a, a)
    EXPECT_EQ(subtract_12.in_edges[1], add_bb.id);  // Second arg: output_tensor_2 = add(b, b)

    // KEY TEST: Verify second subtract(output_tensor_2, output_tensor_1) has REVERSED argument order
    // This is the critical test - argument order must be preserved for intermediate tensors
    EXPECT_EQ(subtract_21.in_edges.size(), 2);
    EXPECT_EQ(subtract_21.in_edges[0], add_bb.id);  // First arg: output_tensor_2 = add(b, b)
    EXPECT_EQ(subtract_21.in_edges[1], add_aa.id);  // Second arg: output_tensor_1 = add(a, a)

    // Verify the two subtract operations have different input orders
    EXPECT_NE(subtract_12.in_edges, subtract_21.in_edges);

    // Verify that subtract(out1, out2) has reversed order compared to subtract(out2, out1)
    EXPECT_EQ(subtract_12.in_edges[0], subtract_21.in_edges[1]);
    EXPECT_EQ(subtract_12.in_edges[1], subtract_21.in_edges[0]);

    // Verify output connections
    EXPECT_EQ(tensor_a_vertex.out_edges.size(), 1);
    EXPECT_EQ(tensor_a_vertex.out_edges[0], add_aa.id);

    EXPECT_EQ(tensor_b_vertex.out_edges.size(), 1);
    EXPECT_EQ(tensor_b_vertex.out_edges[0], add_bb.id);

    // Each add operation feeds into both subtract operations
    EXPECT_EQ(add_aa.out_edges.size(), 2);
    EXPECT_TRUE(std::ranges::find(add_aa.out_edges, subtract_12.id) != add_aa.out_edges.end());
    EXPECT_TRUE(std::ranges::find(add_aa.out_edges, subtract_21.id) != add_aa.out_edges.end());

    EXPECT_EQ(add_bb.out_edges.size(), 2);
    EXPECT_TRUE(std::ranges::find(add_bb.out_edges, subtract_12.id) != add_bb.out_edges.end());
    EXPECT_TRUE(std::ranges::find(add_bb.out_edges, subtract_21.id) != add_bb.out_edges.end());

    // Both subtract operations are terminal (no outgoing edges)
    EXPECT_TRUE(subtract_12.out_edges.empty());
    EXPECT_TRUE(subtract_21.out_edges.empty());

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));

    // At level 2, add and subtract operations should have internals
    const auto& add_aa_l2 = levelized_graph_2.get_vertex(2);
    EXPECT_FALSE(add_aa_l2.internals.empty());

    const auto& subtract_12_l2 = levelized_graph_2.get_vertex(4);
    EXPECT_FALSE(subtract_12_l2.internals.empty());
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
    EXPECT_EQ(levelized_graph.size(), 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    const auto& tensor_a_vertex = levelized_graph.get_vertex(0);
    const auto& add_aa = levelized_graph.get_vertex(1);

    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_EQ(add_aa.name, "ttnn::add");

    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_EQ(tensor_a_vertex.out_edges.size(), 1);
    EXPECT_EQ(tensor_a_vertex.out_edges[0], add_aa.id);

    // add(a, a) should have in_edges [a, a]
    EXPECT_EQ(add_aa.in_edges.size(), 2);
    EXPECT_EQ(add_aa.in_edges[0], tensor_a_vertex.id);
    EXPECT_EQ(add_aa.in_edges[1], tensor_a_vertex.id);
    EXPECT_TRUE(add_aa.out_edges.empty());

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));

    // At level 2, add should have internals (binary_ng)
    const auto& add_aa_l2 = levelized_graph_2.get_vertex(1);
    EXPECT_FALSE(add_aa_l2.internals.empty());
    EXPECT_EQ(add_aa_l2.internals.size(), 1);
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
    EXPECT_EQ(levelized_graph.size(), 5);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    const auto& tensor_a_vertex = levelized_graph.get_vertex(0);
    const auto& tensor_b_vertex = levelized_graph.get_vertex(1);
    const auto& tensor_c_vertex = levelized_graph.get_vertex(2);
    const auto& addcmul_abc = levelized_graph.get_vertex(3);
    const auto& addcmul_cba = levelized_graph.get_vertex(4);

    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_b_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_c_vertex.name.find("tensor") != std::string::npos);
    EXPECT_EQ(addcmul_abc.name, "ttnn::addcmul");
    EXPECT_EQ(addcmul_cba.name, "ttnn::addcmul");

    // All three tensors connect to both addcmul operations
    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_EQ(tensor_a_vertex.out_edges.size(), 2);
    EXPECT_TRUE(tensor_b_vertex.in_edges.empty());
    EXPECT_EQ(tensor_b_vertex.out_edges.size(), 2);
    EXPECT_TRUE(tensor_c_vertex.in_edges.empty());
    EXPECT_EQ(tensor_c_vertex.out_edges.size(), 2);

    // First addcmul(a, b, c) should have in_edges [a, b, c]
    EXPECT_EQ(addcmul_abc.in_edges.size(), 3);
    EXPECT_EQ(addcmul_abc.in_edges[0], tensor_a_vertex.id);
    EXPECT_EQ(addcmul_abc.in_edges[1], tensor_b_vertex.id);
    EXPECT_EQ(addcmul_abc.in_edges[2], tensor_c_vertex.id);
    EXPECT_TRUE(addcmul_abc.out_edges.empty());

    // Second addcmul(c, b, a) should have in_edges [c, b, a]
    EXPECT_EQ(addcmul_cba.in_edges.size(), 3);
    EXPECT_EQ(addcmul_cba.in_edges[0], tensor_c_vertex.id);
    EXPECT_EQ(addcmul_cba.in_edges[1], tensor_b_vertex.id);
    EXPECT_EQ(addcmul_cba.in_edges[2], tensor_a_vertex.id);
    EXPECT_TRUE(addcmul_cba.out_edges.empty());

    // Verify that the two addcmul operations have different input orders
    EXPECT_NE(addcmul_abc.in_edges, addcmul_cba.in_edges);

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));

    // At level 2, at least one addcmul operation should have internals
    const auto& addcmul_abc_l2 = levelized_graph_2.get_vertex(3);
    EXPECT_FALSE(addcmul_abc_l2.internals.empty());
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
    EXPECT_EQ(levelized_graph.size(), 4);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    const auto& tensor_a_vertex = levelized_graph.get_vertex(0);
    const auto& tensor_b_vertex = levelized_graph.get_vertex(1);
    const auto& addcmul_aba = levelized_graph.get_vertex(2);
    const auto& addcmul_baa = levelized_graph.get_vertex(3);

    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_b_vertex.name.find("tensor") != std::string::npos);
    EXPECT_EQ(addcmul_aba.name, "ttnn::addcmul");
    EXPECT_EQ(addcmul_baa.name, "ttnn::addcmul");

    // Both tensors connect to both addcmul operations
    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_EQ(tensor_a_vertex.out_edges.size(), 2);
    EXPECT_TRUE(tensor_b_vertex.in_edges.empty());
    EXPECT_EQ(tensor_b_vertex.out_edges.size(), 2);

    // First addcmul(a, b, a) should have in_edges [a, b, a]
    EXPECT_EQ(addcmul_aba.in_edges.size(), 3);
    EXPECT_EQ(addcmul_aba.in_edges[0], tensor_a_vertex.id);
    EXPECT_EQ(addcmul_aba.in_edges[1], tensor_b_vertex.id);
    EXPECT_EQ(addcmul_aba.in_edges[2], tensor_a_vertex.id);
    EXPECT_TRUE(addcmul_aba.out_edges.empty());

    // Second addcmul(b, a, a) should have in_edges [b, a, a]
    EXPECT_EQ(addcmul_baa.in_edges.size(), 3);
    EXPECT_EQ(addcmul_baa.in_edges[0], tensor_b_vertex.id);
    EXPECT_EQ(addcmul_baa.in_edges[1], tensor_a_vertex.id);
    EXPECT_EQ(addcmul_baa.in_edges[2], tensor_a_vertex.id);
    EXPECT_TRUE(addcmul_baa.out_edges.empty());

    // Verify that the two addcmul operations have different input orders
    EXPECT_NE(addcmul_aba.in_edges, addcmul_baa.in_edges);

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));

    // At level 2, at least one addcmul operation should have internals
    const auto& addcmul_aba_l2 = levelized_graph_2.get_vertex(2);
    EXPECT_FALSE(addcmul_aba_l2.internals.empty());
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
    EXPECT_EQ(levelized_graph.size(), 5);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    const auto& tensor_a_vertex = levelized_graph.get_vertex(0);
    const auto& tensor_b_vertex = levelized_graph.get_vertex(1);
    const auto& matmul_ab = levelized_graph.get_vertex(2);
    const auto& matmul_ba = levelized_graph.get_vertex(3);
    const auto& matmul_aa = levelized_graph.get_vertex(4);

    EXPECT_TRUE(tensor_a_vertex.name.find("tensor") != std::string::npos);
    EXPECT_TRUE(tensor_b_vertex.name.find("tensor") != std::string::npos);
    EXPECT_EQ(matmul_ab.name, "ttnn::matmul");
    EXPECT_EQ(matmul_ba.name, "ttnn::matmul");
    EXPECT_EQ(matmul_aa.name, "ttnn::matmul");

    // Both tensors connect to all three matmul operations
    EXPECT_TRUE(tensor_a_vertex.in_edges.empty());
    EXPECT_EQ(tensor_a_vertex.out_edges.size(), 3);
    EXPECT_TRUE(tensor_b_vertex.in_edges.empty());
    EXPECT_EQ(tensor_b_vertex.out_edges.size(), 2);

    // First matmul(a, b) should have in_edges [a, b]
    EXPECT_EQ(matmul_ab.in_edges.size(), 2);
    EXPECT_EQ(matmul_ab.in_edges[0], tensor_a_vertex.id);
    EXPECT_EQ(matmul_ab.in_edges[1], tensor_b_vertex.id);
    EXPECT_TRUE(matmul_ab.out_edges.empty());

    // Second matmul(b, a) should have in_edges [b, a]
    EXPECT_EQ(matmul_ba.in_edges.size(), 2);
    EXPECT_EQ(matmul_ba.in_edges[0], tensor_b_vertex.id);
    EXPECT_EQ(matmul_ba.in_edges[1], tensor_a_vertex.id);
    EXPECT_TRUE(matmul_ba.out_edges.empty());

    // Third matmul(a, a) should have in_edges [a, a]
    EXPECT_EQ(matmul_aa.in_edges.size(), 2);
    EXPECT_EQ(matmul_aa.in_edges[0], tensor_a_vertex.id);
    EXPECT_EQ(matmul_aa.in_edges[1], tensor_a_vertex.id);
    EXPECT_TRUE(matmul_aa.out_edges.empty());

    // Verify all three matmul operations have different input patterns
    EXPECT_NE(matmul_ab.in_edges, matmul_ba.in_edges);
    EXPECT_NE(matmul_ab.in_edges, matmul_aa.in_edges);
    EXPECT_NE(matmul_ba.in_edges, matmul_aa.in_edges);

    // Test level 2
    auto levelized_graph_2 = ttnn::graph::LevelizedGraph(ref_json_trace, 2);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.vertices(), [&](const auto& vertex) { return vertex.stacking_level <= 2; }));
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph_2.get_vertices_at_level(2), [&](const auto& vertex) { return vertex.internals.empty(); }));

    // At level 2, at least one matmul operation should have internals
    const auto& matmul_ab_l2 = levelized_graph_2.get_vertex(2);
    EXPECT_FALSE(matmul_ab_l2.internals.empty());
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
    EXPECT_EQ(levelized_graph_json.size(), 2);  // tensor, add

    // Verify first vertex (tensor)
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
    EXPECT_EQ(levelized_graph_json[0][ttnn::graph::kOutEdges].size(), 1);
    EXPECT_TRUE(levelized_graph_json[0][ttnn::graph::kInternals].is_array());
    EXPECT_TRUE(levelized_graph_json[0][ttnn::graph::kInternals].empty());
    EXPECT_TRUE(levelized_graph_json[0][ttnn::graph::kOutputShape].is_array());
    EXPECT_FALSE(levelized_graph_json[0][ttnn::graph::kOutputShape].empty());

    // Verify second vertex (add)
    EXPECT_TRUE(levelized_graph_json[1].is_object());
    EXPECT_EQ(levelized_graph_json[1][ttnn::graph::kName], "ttnn::add");
    EXPECT_EQ(levelized_graph_json[1][ttnn::graph::kStackingLevel], 1);
    EXPECT_EQ(levelized_graph_json[1][ttnn::graph::kCounter], 1);
    EXPECT_TRUE(levelized_graph_json[1][ttnn::graph::kInEdges].is_array());
    EXPECT_EQ(levelized_graph_json[1][ttnn::graph::kInEdges].size(), 2);  // both inputs from vertex 0
    EXPECT_EQ(levelized_graph_json[1][ttnn::graph::kInEdges][0], 0);
    EXPECT_EQ(levelized_graph_json[1][ttnn::graph::kInEdges][1], 0);
    EXPECT_TRUE(levelized_graph_json[1][ttnn::graph::kOutEdges].is_array());
    EXPECT_TRUE(levelized_graph_json[1][ttnn::graph::kOutEdges].empty());
    EXPECT_TRUE(levelized_graph_json[1][ttnn::graph::kInternals].is_array());
    EXPECT_TRUE(levelized_graph_json[1][ttnn::graph::kInternals].empty());
    EXPECT_TRUE(levelized_graph_json[1][ttnn::graph::kOutputShape].is_array());
    EXPECT_FALSE(levelized_graph_json[1][ttnn::graph::kOutputShape].empty());

    // Test extract_levelized_graph API - level 2
    auto levelized_graph_json_2 = ttnn::graph::extract_levelized_graph(ref_json_trace, 2);

    // Verify JSON structure for level 2
    EXPECT_TRUE(levelized_graph_json_2.is_array());
    EXPECT_GE(levelized_graph_json_2.size(), 3);  // At least 3 vertices at level 2

    // Verify that add vertex has internals at level 2
    auto add_vertex_it = std::ranges::find_if(
        levelized_graph_json_2, [](const auto& v) { return v[ttnn::graph::kName] == "ttnn::add"; });
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
    EXPECT_EQ(levelized_graph.size(), 5);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));
    EXPECT_TRUE(std::ranges::none_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.output_shape.empty(); }));

    // Tensor 1 is the first one that is used, thus it'll have the id 0
    const auto& v_tensor_1 = levelized_graph.get_vertex(0);
    // Tensor 2 is the second one that is used, thus it'll have the id 1
    const auto& v_tensor_2 = levelized_graph.get_vertex(1);
    // Tensor 0 is the third one that is used, thus it'll have the id 2
    const auto& v_tensor_0 = levelized_graph.get_vertex(2);
    const auto& v_multiply = levelized_graph.get_vertex(3);
    const auto& v_add = levelized_graph.get_vertex(4);
    EXPECT_NE(v_tensor_0.name.find("tensor"), std::string::npos);
    EXPECT_NE(v_tensor_1.name.find("tensor"), std::string::npos);
    EXPECT_NE(v_tensor_2.name.find("tensor"), std::string::npos);
    EXPECT_EQ(v_multiply.name, "ttnn::multiply");
    EXPECT_EQ(v_add.name, "ttnn::add");

    // Confirming that the multiply vertex has two inputs from tensors
    // Also add should have one input from tensor and one from multiply
    // Motivated by this issue: https://github.com/tenstorrent/tt-mlir/issues/5929
    EXPECT_EQ(v_multiply.in_edges.size(), 2);
    // Both inputs should be tensors
    // Note: In the current form of trace, it's not possible to distinguish between the two tensors (order of args)
    EXPECT_TRUE(v_multiply.in_edges[0] == v_tensor_1.id || v_multiply.in_edges[0] == v_tensor_2.id);
    EXPECT_TRUE(v_multiply.in_edges[1] == v_tensor_1.id || v_multiply.in_edges[1] == v_tensor_2.id);
    EXPECT_TRUE(v_multiply.in_edges[0] != v_multiply.in_edges[1]);

    EXPECT_EQ(v_add.in_edges.size(), 2);
    // One input should be a tensor, the other should be multiply
    bool has_tensor_input = v_add.in_edges[0] == v_tensor_0.id || v_add.in_edges[1] == v_tensor_0.id;
    bool has_multiply_input = v_add.in_edges[0] == v_multiply.id || v_add.in_edges[1] == v_multiply.id;
    EXPECT_TRUE(has_tensor_input);
    EXPECT_TRUE(has_multiply_input);
    EXPECT_TRUE(v_add.in_edges[0] != v_add.in_edges[1]);

    // Each tensor should have exactly one outgoing edge
    EXPECT_EQ(v_tensor_0.out_edges.size(), 1);
    EXPECT_EQ(v_tensor_0.out_edges[0], v_add.id);
    EXPECT_EQ(v_tensor_1.out_edges.size(), 1);
    EXPECT_EQ(v_tensor_1.out_edges[0], v_multiply.id);
    EXPECT_EQ(v_tensor_2.out_edges.size(), 1);
    EXPECT_EQ(v_tensor_2.out_edges[0], v_multiply.id);

    // Multiply should output to add
    EXPECT_EQ(v_multiply.out_edges.size(), 1);
    EXPECT_EQ(v_multiply.out_edges[0], v_add.id);

    // Add should be the final operation
    EXPECT_EQ(v_add.out_edges.size(), 0);
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

    // Should have 8 vertices: 3 create_device_tensor operations + 3 tensor nodes + multiply + add
    EXPECT_EQ(levelized_graph.size(), 8);
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));
    EXPECT_TRUE(std::ranges::none_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.output_shape.empty(); }));

    // Categorize vertices by type
    std::vector<size_t> create_tensor_ids;
    std::vector<size_t> tensor_ids;
    size_t multiply_id = 0, add_id = 0;

    for (size_t i = 0; i < levelized_graph.size(); ++i) {
        const auto& v = levelized_graph.get_vertex(i);
        if (v.name == "tt::tt_metal::create_device_tensor") {
            create_tensor_ids.push_back(i);
        } else if (v.name.find("tensor[") != std::string::npos) {
            tensor_ids.push_back(i);
        } else if (v.name == "ttnn::multiply") {
            multiply_id = i;
        } else if (v.name == "ttnn::add") {
            add_id = i;
        }
    }

    // Verify we have the expected number of each type
    EXPECT_EQ(create_tensor_ids.size(), 3);  // 3 create_device_tensor operations
    EXPECT_EQ(tensor_ids.size(), 3);         // 3 tensor nodes
    EXPECT_NE(multiply_id, 0);               // Found multiply
    EXPECT_NE(add_id, 0);                    // Found add

    const auto& v_multiply = levelized_graph.get_vertex(multiply_id);
    const auto& v_add = levelized_graph.get_vertex(add_id);

    // Key verification: create_device_tensor operations have no edges in the levelized graph
    // This is expected behavior - edges connect tensor nodes to operations, not operations to tensor nodes
    for (size_t create_id : create_tensor_ids) {
        const auto& create_vertex = levelized_graph.get_vertex(create_id);
        EXPECT_EQ(create_vertex.in_edges.size(), 0);   // No inputs
        EXPECT_EQ(create_vertex.out_edges.size(), 0);  // No outgoing edges (edges go through tensor nodes)
        EXPECT_FALSE(create_vertex.output_shape.empty());
    }

    // Key verification: Each tensor node should have exactly one incoming edge from create_device_tensor
    // and exactly one outgoing edge to an operation
    for (size_t tensor_id : tensor_ids) {
        const auto& tensor_vertex = levelized_graph.get_vertex(tensor_id);
        EXPECT_EQ(tensor_vertex.in_edges.size(), 1);   // One input from create_device_tensor
        EXPECT_EQ(tensor_vertex.out_edges.size(), 1);  // One output to operation
        EXPECT_FALSE(tensor_vertex.output_shape.empty());
        EXPECT_TRUE(tensor_vertex.internals.empty());

        // The input should be a create_device_tensor operation
        size_t creator = tensor_vertex.in_edges[0];
        EXPECT_NE(std::find(create_tensor_ids.begin(), create_tensor_ids.end(), creator), create_tensor_ids.end());

        // The output should be either multiply or add
        size_t consumer = tensor_vertex.out_edges[0];
        EXPECT_TRUE(consumer == multiply_id || consumer == add_id);
    }

    // Verify multiply has 2 tensor inputs
    EXPECT_EQ(v_multiply.in_edges.size(), 2);
    for (size_t input_id : v_multiply.in_edges) {
        EXPECT_NE(std::find(tensor_ids.begin(), tensor_ids.end(), input_id), tensor_ids.end());
    }
    EXPECT_EQ(v_multiply.out_edges.size(), 1);
    EXPECT_EQ(v_multiply.out_edges[0], add_id);

    // Verify add has 2 inputs: 1 tensor + 1 multiply result
    EXPECT_EQ(v_add.in_edges.size(), 2);
    int tensor_inputs = 0;
    int multiply_inputs = 0;
    for (size_t input_id : v_add.in_edges) {
        if (std::find(tensor_ids.begin(), tensor_ids.end(), input_id) != tensor_ids.end()) {
            tensor_inputs++;
        } else if (input_id == multiply_id) {
            multiply_inputs++;
        }
    }
    EXPECT_EQ(tensor_inputs, 1);           // One tensor input
    EXPECT_EQ(multiply_inputs, 1);         // One multiply result input
    EXPECT_EQ(v_add.out_edges.size(), 0);  // Final operation

    // Verify the data flow: tensor nodes track their creator and consumer
    // This demonstrates the key difference from runtime inputs:
    // - Runtime input tensors: empty in_edges
    // - Captured creation tensors: in_edges point to create_device_tensor
    for (size_t tensor_id : tensor_ids) {
        const auto& tensor_vertex = levelized_graph.get_vertex(tensor_id);

        // Verify the tensor knows its creator (create_device_tensor)
        EXPECT_EQ(tensor_vertex.in_edges.size(), 1);
        size_t creator_id = tensor_vertex.in_edges[0];
        const auto& creator = levelized_graph.get_vertex(creator_id);
        EXPECT_EQ(creator.name, "tt::tt_metal::create_device_tensor");

        // Verify the tensor feeds into an operation
        EXPECT_EQ(tensor_vertex.out_edges.size(), 1);
        size_t operation_id = tensor_vertex.out_edges[0];
        EXPECT_TRUE(operation_id == multiply_id || operation_id == add_id);
    }

    // This test demonstrates that the levelized graph algorithm correctly handles both:
    // 1. Runtime input tensors (empty in_edges) - as in MultiplyAndAddTest
    // 2. Captured tensor creation (in_edges point to creator) - as in this test
    // The key insight: tensor nodes always represent the data flow, regardless of where tensors were created
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

    // Should have: 2 create_device_tensor operations + 2 tensor nodes + 2 subtract operations
    EXPECT_EQ(levelized_graph.size(), 6);
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
        } else if (v.name == "ttnn::subtract") {
            subtract_ids.push_back(i);
        }
    }

    // Verify we have the expected number of each type
    EXPECT_EQ(create_tensor_ids.size(), 2);  // 2 create_device_tensor operations
    EXPECT_EQ(tensor_ids.size(), 2);         // 2 tensor nodes
    EXPECT_EQ(subtract_ids.size(), 2);       // 2 subtract operations

    // Get references to the subtract operations
    const auto& subtract_1 = levelized_graph.get_vertex(subtract_ids[0]);
    const auto& subtract_2 = levelized_graph.get_vertex(subtract_ids[1]);

    // Both subtract operations should have exactly 2 input edges (from tensor nodes)
    EXPECT_EQ(subtract_1.in_edges.size(), 2);
    EXPECT_EQ(subtract_2.in_edges.size(), 2);

    // Get the tensor node IDs
    size_t tensor_a_id = tensor_ids[0];
    size_t tensor_b_id = tensor_ids[1];

    // First subtract should be subtract(a, b), so in_edges should be [tensor_a, tensor_b]
    EXPECT_EQ(subtract_1.in_edges[0], tensor_a_id);
    EXPECT_EQ(subtract_1.in_edges[1], tensor_b_id);

    // Second subtract should be subtract(b, a), so in_edges should be [tensor_b, tensor_a]
    EXPECT_EQ(subtract_2.in_edges[0], tensor_b_id);
    EXPECT_EQ(subtract_2.in_edges[1], tensor_a_id);

    // Verify the two subtract operations have different input patterns (argument order matters!)
    EXPECT_NE(subtract_1.in_edges, subtract_2.in_edges);

    // Verify that subtract(a, b) has reversed order compared to subtract(b, a)
    EXPECT_EQ(subtract_1.in_edges[0], subtract_2.in_edges[1]);
    EXPECT_EQ(subtract_1.in_edges[1], subtract_2.in_edges[0]);

    // Both subtract operations should have no output edges (they are terminal operations)
    EXPECT_TRUE(subtract_1.out_edges.empty());
    EXPECT_TRUE(subtract_2.out_edges.empty());

    // Verify each tensor node feeds into exactly 2 subtract operations
    for (size_t tensor_id : tensor_ids) {
        const auto& tensor_vertex = levelized_graph.get_vertex(tensor_id);
        EXPECT_EQ(tensor_vertex.out_edges.size(), 2);  // Each tensor used in both subtract operations
    }
}
