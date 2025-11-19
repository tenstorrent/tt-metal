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

    auto operation = [&device](tt::tt_metal::DataType datatype) {
        const auto input = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{64, 128}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);
        const auto output_tensor = ttnn::add(input_tensor, input_tensor, std::nullopt, std::nullopt);
    };

    // build reference
    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        ref_json_trace = capture.end_graph_capture();
    }

    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_EQ(levelized_graph.size(), 2);  // 2 vertices: create_device_tensor, ttnn::add
    // invariants: all vertices should be at level 1 and with no internals:
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    auto vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    EXPECT_EQ(vertex_0.name, "tt::tt_metal::create_device_tensor");
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
    EXPECT_EQ(vertex_0.name, "tt::tt_metal::create_device_tensor");
    EXPECT_EQ(vertex_1.name, "ttnn::add");
    EXPECT_EQ(vertex_2.name, "ttnn::prim::binary_ng");

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
    EXPECT_EQ(vertex_1.internals.size(), 1);  // we should have expanded ttnn::add to ttnn::prim::binary_ng
    EXPECT_EQ(vertex_1.internals[0], vertex_2.id);

    EXPECT_EQ(vertex_2.in_edges.size(), 2);
    EXPECT_EQ(vertex_2.in_edges[0], vertex_0.id);
    EXPECT_EQ(vertex_2.in_edges[1], vertex_0.id);
    EXPECT_TRUE(vertex_2.out_edges.empty());
    EXPECT_TRUE(vertex_2.internals.empty());
}

TEST_F(TestLevelizedGraphCapture, ReductionOp) {
    tt::tt_metal::IDevice* device = device_;

    auto operation = [&device](tt::tt_metal::DataType datatype) {
        const auto input = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{256, 128}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);
        const auto output_tensor_1 = ttnn::sum(input_tensor, 0, true);
        const auto output_tensor_2 = ttnn::mean(input_tensor, 1, false);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_EQ(levelized_graph.size(), 3);  // create_device_tensor, sum, mean
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    const auto& vertex_2 = levelized_graph.get_vertex(2);
    EXPECT_EQ(vertex_0.name, "tt::tt_metal::create_device_tensor");
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
    EXPECT_EQ(levelized_graph_2.get_vertex(vertex_1.internals[2]).name, "ttnn::prim::old_infra_device_operation");
    EXPECT_EQ(levelized_graph_2.get_vertex(vertex_1.internals[3]).name, "ttnn::reshape");
    // Find mean vertex by name
    auto mean_vertex_it =
        std::ranges::find_if(levelized_graph_2.vertices(), [](const auto& v) { return v.name == "ttnn::mean"; });
    EXPECT_NE(mean_vertex_it, levelized_graph_2.vertices().end());
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->id).name, "ttnn::mean");
    EXPECT_EQ(
        levelized_graph_2.get_vertex(mean_vertex_it->id).output_shape[0], shape_to_string(tt::tt_metal::Array1D{256}));
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->id).internals.size(), 4);
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->internals[0]).name, "ttnn::fill_implicit_tile_padding");
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->internals[1]).name, "ttnn::reshape");
    EXPECT_EQ(
        levelized_graph_2.get_vertex(mean_vertex_it->internals[2]).name, "ttnn::prim::old_infra_device_operation");
    EXPECT_EQ(levelized_graph_2.get_vertex(mean_vertex_it->internals[3]).name, "ttnn::reshape");

    auto levelized_graph_json = ttnn::graph::extract_levelized_graph(ref_json_trace, 2);
    std::cout << levelized_graph_json.dump(4) << std::endl;
}

TEST_F(TestLevelizedGraphCapture, OutputLayoutInfo) {
    tt::tt_metal::IDevice* device = device_;

    auto operation = [&device](tt::tt_metal::DataType datatype) {
        const auto input = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array3D{16, 32, 64}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);
        const auto output_tensor_1 = ttnn::sum(input_tensor, 2, true);
        const auto output_tensor_2 = ttnn::softmax(output_tensor_1, -1);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_EQ(levelized_graph.size(), 3);  // create_device_tensor, sum, softmax
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    const auto& vertex_2 = levelized_graph.get_vertex(2);
    EXPECT_EQ(vertex_0.name, "tt::tt_metal::create_device_tensor");
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

    auto operation = [&device](tt::tt_metal::DataType datatype) {
        const auto input = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{32, 32}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);
        const auto output_tensor_1 = ttnn::matmul(input_tensor, input_tensor);
        const auto output_tensor_2 = ttnn::add(input_tensor, output_tensor_1, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_EQ(levelized_graph.size(), 3);  // create_device_tensor, matmul, add
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    const auto& vertex_2 = levelized_graph.get_vertex(2);
    EXPECT_EQ(vertex_0.name, "tt::tt_metal::create_device_tensor");
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

    auto operation = [&device](tt::tt_metal::DataType datatype) {
        const auto input = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{12, 19}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);
        const auto output_tensor_1 = ttnn::digamma(input_tensor);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_EQ(levelized_graph.size(), 2);  // create_device_tensor, digamma
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    EXPECT_EQ(vertex_0.name, "tt::tt_metal::create_device_tensor");
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

    auto operation = [&device](tt::tt_metal::DataType datatype) {
        const auto input = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{64, 128}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);
        const auto output_tensor = ttnn::multiply(input_tensor, input_tensor, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_EQ(levelized_graph.size(), 2);  // create_device_tensor, multiply
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));
    EXPECT_TRUE(
        std::ranges::all_of(levelized_graph.vertices(), [&](const auto& vertex) { return vertex.internals.empty(); }));

    auto vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    EXPECT_EQ(vertex_0.name, "tt::tt_metal::create_device_tensor");
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
    EXPECT_EQ(vertex_0.name, "tt::tt_metal::create_device_tensor");
    EXPECT_EQ(vertex_1.name, "ttnn::multiply");
    EXPECT_EQ(vertex_2.name, "ttnn::prim::binary_ng");

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

    auto operation = [&device](tt::tt_metal::DataType datatype) {
        const auto input = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{64, 128}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);
        const auto output_tensor_1 = ttnn::add(input_tensor, input_tensor, std::nullopt, std::nullopt);
        const auto output_tensor_2 = ttnn::subtract(input_tensor, input_tensor, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1 - fork: one input tensor feeds multiple operations
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_EQ(levelized_graph.size(), 3);  // create_device_tensor, add, multiply
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    auto vertex_1 = levelized_graph.get_vertex(1);
    const auto& vertex_2 = levelized_graph.get_vertex(2);
    EXPECT_EQ(vertex_0.name, "tt::tt_metal::create_device_tensor");
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

    auto operation = [&device](tt::tt_metal::DataType datatype) {
        const auto input_a = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{64, 128}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_b = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{64, 128}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor_a = tt::tt_metal::create_device_tensor(input_a, device);
        const auto input_tensor_b = tt::tt_metal::create_device_tensor(input_b, device);
        const auto output_tensor = ttnn::add(input_tensor_a, input_tensor_b, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test level 1 - join: one operation uses multiple different input tensors
    ttnn::graph::LevelizedGraph levelized_graph(ref_json_trace, 1);
    EXPECT_EQ(levelized_graph.size(), 3);  // create_device_tensor (a), create_device_tensor (b), add
    EXPECT_TRUE(std::ranges::all_of(
        levelized_graph.vertices(), [&](const auto& vertex) { return vertex.stacking_level == 1; }));

    const auto& vertex_0 = levelized_graph.get_vertex(0);
    const auto& vertex_1 = levelized_graph.get_vertex(1);
    auto vertex_2 = levelized_graph.get_vertex(2);
    EXPECT_EQ(vertex_0.name, "tt::tt_metal::create_device_tensor");
    EXPECT_EQ(vertex_1.name, "tt::tt_metal::create_device_tensor");
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

TEST_F(TestLevelizedGraphCapture, ExtractLevelizedGraphJsonTest) {
    tt::tt_metal::IDevice* device = device_;

    auto operation = [&device](tt::tt_metal::DataType datatype) {
        const auto input = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
            tt::tt_metal::TensorLayout(
                datatype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::L1_MEMORY_CONFIG));
        const auto input_tensor = tt::tt_metal::create_device_tensor(input, device);
        const auto output_tensor = ttnn::add(input_tensor, input_tensor, std::nullopt, std::nullopt);
    };

    nlohmann::json ref_json_trace;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(IGraphProcessor::RunMode::NO_DISPATCH);
        operation(tt::tt_metal::DataType::BFLOAT16);
        ref_json_trace = capture.end_graph_capture();
    }

    // Test extract_levelized_graph API - level 1
    auto levelized_graph_json = ttnn::graph::extract_levelized_graph(ref_json_trace, 1);

    // Verify JSON structure
    EXPECT_TRUE(levelized_graph_json.is_array());
    EXPECT_EQ(levelized_graph_json.size(), 2);  // create_device_tensor, add

    // Verify first vertex (create_device_tensor)
    EXPECT_TRUE(levelized_graph_json[0].is_object());
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kCounter));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kStackingLevel));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kName));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kInEdges));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kOutEdges));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kInternals));
    EXPECT_TRUE(levelized_graph_json[0].contains(ttnn::graph::kOutputShape));

    EXPECT_EQ(levelized_graph_json[0][ttnn::graph::kName], "tt::tt_metal::create_device_tensor");
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
