#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "json_fwd.hpp"
#include "json.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "reportify/reportify.hpp"

void json_to_node(tt::graphlib::Graph * graph, nlohmann::json data, std::string node_name);
void build_graph_from_json(tt::graphlib::Graph * graph, std::string json_file);
