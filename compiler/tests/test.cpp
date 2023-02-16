#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "reportify/reportify.hpp"

using namespace tt::graphlib;


int main() {
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
    tt::reportify::dump_graph("test", "initial_graph", graph.get());
    return 0;
}