#include <filesystem>
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "reportify/reportify.hpp"
#include "reportify/paths.hpp"
#include "graph_deserializer/graph_deserializer.hpp"
#include "graph_utils/op_histogram.hpp"

int main() {
    using namespace tt::graphlib;
    auto graph = std::make_unique<Graph>(IRLevel::IR_PYBUDA);
    auto weights = graph->add_node(create_node<ConstantInputNode>("weights1", 0));
    auto act = graph->add_node(create_node<InputNode>("input", InputNodeType::Parameter, false));
    auto mm = graph->add_node(create_node<PyOpNode>("matmul1", OpType{.op="matmul", .attr={}, .buda_attrs={}}));
    auto out = graph->add_node(create_node<OutputNode>("output"));
    Edge act_mm(act->id(), (PortId)0, mm->id(), (PortId)0, EdgeType::kData);
    graph->add_edge(act_mm);
    Edge w_mm(weights->id(), (PortId)0, mm->id(), (PortId)1, EdgeType::kData);
    graph->add_edge(w_mm);
    Edge mm_out(mm->id(), (PortId)0, out->id(), (PortId)0, EdgeType::kData);
    graph->add_edge(mm_out);
    std::filesystem::path cwd = std::filesystem::current_path();
    tt::reportify::dump_graph(cwd.string() + "/", "test_graph_deserializer", "original_graph", graph.get());
    std::string dumped_original_graph_path = cwd.string() + "/" + "test_graph_deserializer" + "/" + tt::reportify::get_pass_reports_relative_directory() + "/original_graph.buda";
    auto new_graph = std::make_unique<Graph>(IRLevel::IR_PYBUDA);
    build_graph_from_json(new_graph.get(), dumped_original_graph_path);
    tt::reportify::dump_graph(cwd.string() + "/", "test_graph_deserializer", "deserializer_dumped_graph", new_graph.get());
    return 0;
}
