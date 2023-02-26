#include "op_histogram.hpp"
#include "op_reducer.hpp"
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <string>
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/common.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/utils.hpp"

void generate_and_dump_op_histogram(tt::graphlib::Graph * graph) {
    using namespace std;
    using namespace tt::graphlib;
    std::map<string, pair<string, string>> op_list;
    std::map<string, int> op_histogram;
    string pybuda_op = "";
    string reduced_set_op = "";
    for (Node* node : tt::graphlib::topological_sort(*graph)) {
        string pybuda_op = "";
        string reduced_set_op = "";
        if (node->node_type() == NodeType::kInput) {
            if (node->as<InputNode>()->is_constant()) {
                pybuda_op = node->as<InputNode>()->input_type_string();
            } else if (node->as<InputNode>()->is_accumulator()) {
                pybuda_op = "accumulator";
            } else {
                pybuda_op = "Input::" + node->as<InputNode>()->input_type_string();
            }
        } else if (node->node_type() == NodeType::kOutput) {
            pybuda_op = "Output";
        } else if (node->node_type() == NodeType::kPyOp) {
            const PyOpNode *opnode = node->as<PyOpNode>();
            pybuda_op = opnode->op_type().op;
        }
        else {
            assert(false);
        }
        std::transform(pybuda_op.begin(), pybuda_op.end(), pybuda_op.begin(), [](unsigned char c){ return std::tolower(c); });
        reduced_set_op = reduced_op_type_to_string(reduce_pybuda_op(pybuda_op));
        if(op_histogram.find(pybuda_op) != op_histogram.end()) {
            op_histogram[pybuda_op]++;
        }
        else {
            op_histogram.insert({pybuda_op, 1});
        }
        op_list.insert({ node->name(), make_pair(pybuda_op, reduced_set_op) });
    }
    for(const auto& elem : op_histogram)
    {
        std::cout << elem.first << " " << elem.second << "\n";
    }
    // Write to CSV file
    std::ofstream myfile;
    myfile.open ("op_list.csv");
    myfile << "Op List\n";
    myfile << "Op name,Pybuda Op Type,Reduced Op Type,\n";
    for (auto it = op_list.begin(); it != op_list.end(); it++)
    {
        myfile << it->first << "," << it->second.first << "," << it->second.second << ",\n";
    }
    myfile.close();
    std::ofstream myfile2;
    myfile2.open ("op_histogram.csv");
    for (auto it2 = op_histogram.begin(); it2 != op_histogram.end(); it2++)
    {
        myfile2 << it2->first << "," << it2->second << ",\n";
    }
    myfile2.close();
    
}