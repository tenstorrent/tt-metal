#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "graph_deserializer/graph_deserializer.hpp"
#include "graph_utils/op_histogram.hpp"
using namespace std;
using namespace tt::graphlib;
int main(int argc, char** argv) {
    assert(argc == 2);
    string filename = argv[1];
    auto graph = std::make_unique<Graph>(IRLevel::IR_PYBUDA);
    build_graph_from_json(graph.get(), filename);
    tt::reportify::dump_graph("test", "test_initial_graph", graph.get());
    generate_and_dump_op_histogram(graph.get());
    return 0;
}
