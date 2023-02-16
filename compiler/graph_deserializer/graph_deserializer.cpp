#include "graph_deserializer.hpp"

std::unordered_map<std::string, tt::graphlib::InputNodeType> input_type_string_map = 
    {
        {"accumulator", tt::graphlib::InputNodeType::Accumulator}, 
        {"input", tt::graphlib::InputNodeType::Activation},
        {"loss", tt::graphlib::InputNodeType::Loss},
        {"parameter", tt::graphlib::InputNodeType::Parameter},
        {"constant", tt::graphlib::InputNodeType::Constant},
        {"optimizer_parameter", tt::graphlib::InputNodeType::OptimizerParameter},
        {"target", tt::graphlib::InputNodeType::Target},
    };

std::unordered_map<std::string, tt::graphlib::EdgeType> edge_type_string_map = 
    {
        {"Data", tt::graphlib::EdgeType::kData}, 
        {"Control", tt::graphlib::EdgeType::kControl},
        {"DataLoopback", tt::graphlib::EdgeType::kDataLoopback},
        {"AutogradFwdToBwd", tt::graphlib::EdgeType::kAutogradFwdToBwd},
        {"AutogradFwdToGradient", tt::graphlib::EdgeType::kAutogradFwdToGradient},
        {"AutogradFwdToOptimizer", tt::graphlib::EdgeType::kAutogradFwdToOptimizer},
        {"AutogradFwdToRecompute", tt::graphlib::EdgeType::kAutogradFwdToRecompute},
        {"kControlLoop", tt::graphlib::EdgeType::kControlLoop},
        {"kAutogradOutputToLoss", tt::graphlib::EdgeType::kAutogradOutputToLoss},
        {"AutogradInputToGradientOut", tt::graphlib::EdgeType::kAutogradInputToGradientOut},
    };


bool is_an_integer(std::string s) {
    assert(!s.empty());
    if(std::all_of(s.begin(), s.end(), ::isdigit)) {
        return true;
    }
    else if(s[0] == '-') {
        assert(std::all_of(next(s.begin()), s.end(), ::isdigit));
        return true;
    }
    return false;
}

std::vector<tt::graphlib::Edge> graph_edges;
void json_to_node(tt::graphlib::Graph * graph, nlohmann::json data, std::string node_name) {
    using namespace std;
    using namespace tt::graphlib;
    using json = nlohmann::json;
    // TODO add assert to check if node exists in data
    auto node_data = data["nodes"][node_name];
    string node_type = node_data["type"];
    string node_class = node_data["class"];
    int node_id = node_data["unique_id"];
    Node * node;
    if (node_class == "Input::") {
        string prefix = "Input::";        
        assert(node_type.compare(0, prefix.size(), prefix) == 0);
        node_type = node_type.substr(prefix.size());
        // TODO: set requires grad correctly
        node = graph->add_node(create_node<InputNode>(node_name, input_type_string_map[node_type], false));
    }
    else if (node_class == "accumulator") {
        node = graph->add_node(create_node<InputNode>(node_name, input_type_string_map[node_type], false));
    }
    else if (node_type == "Constant") {
        if (node_data.contains("constant_value")) {
            auto constant_dims = node_data["constant_dims"].get<std::vector<uint32_t>>();
            node = 
                graph->add_node(create_node<ConstantInputNode>(node_name, 
                                    stof((string) node_data["constant_value"]), 
                                    constant_dims[0], 
                                    constant_dims[1]));
        }
        else if (node_data.contains("constant_tile")) {
           node = 
                graph->add_node(create_node<ConstantInputNode>(node_name, 
                                    node_data["constant_tile"].get<std::vector<float>>())); 
        }
        else {
            assert(node_data.contains("constant_dims"));
            auto tensor_shape = Shape::create(node_data["constant_dims"].get<std::vector<uint32_t>>());
            std::shared_ptr<void> vps = std::make_shared<int>();

            node = 
                graph->add_node(create_node<ConstantInputNode>(node_name, 
                                    vps,
                                    tensor_shape)); 
        }
    }
    else if (node_data.contains("ir") && node_data["ir"] == "pybuda") {
        vector<OpType::Attr> attr;
        if(node_type != node_class) {
            assert(node_type.size() < node_class.size());
            // assert prefix exists in node class
            assert(node_class.compare(0, node_type.size()+1, node_type+"(") == 0);
            // remove prefix
            string attributes = node_class.substr(node_type.size()+1);
            // assert suffix exists
            assert(attributes.back() == ')');
            // remove suffix
            attributes.pop_back();
            size_t pos = 0;
            std::string at;
            while ((pos = attributes.find(",")) != std::string::npos) {
                at = attributes.substr(0, pos);
                assert(!at.empty());
                attributes.erase(0, pos + 1);
                if(is_an_integer(at)) {
                    attr.push_back(stoi(at));
                }
                else {
                    attr.push_back(at);
                }
            }
            if (!attributes.empty()) {
                if(is_an_integer(attributes)) {
                        attr.push_back(stoi(attributes));
                }
                else {
                    attr.push_back(attributes);
                }
            }
        }
        node = graph->add_node(create_node<PyOpNode>(node_name, OpType{.op=node_type, .attr=attr, .buda_attrs={}}));
    }
    else if (node_class == "Output") {
        node = graph->add_node(create_node<OutputNode>(node_name));
    }
    else {
        assert(false);
    }
    // DO NOT UPDATE node ID, GRAPH class does not support this, there is an internal list of nodes that are mapped by original ID. Those IDs do not get updated.
    //node->set_id(node_id);
    auto tensor_shape = Shape::create(node_data["cache"]["shape"].get<std::vector<uint32_t>>());
    node->set_shape(tensor_shape);
    // add operand edges
    int i = 0;
    for (string input_node_name : node_data["input_nodes"]) {
        // check if input node exists
        assert(data["nodes"].contains(input_node_name));
        auto input_node = graph->get_node_by_name(input_node_name);
        // get producer output port id, consumer input port id and edge type
        // TODO: add sanity check to ensure the maps, incoming_edge_port_info and input_node_to_edge_type,
        // have the same order as the "input nodes" vector
        string port_info = node_data["incoming_edge_port_info"][i];
        string port = port_info.substr(port_info.find("(")+1, port_info.find(")"));
        string p = "port_";
        int consumer_input_port_id = stoi(port.substr(p.size()));
        auto edge_info = node_data["input_node_to_edge_type"].get<std::map<string, string>>();
        graph_edges.push_back(Edge(input_node->id(), (PortId)0, node->id(), 
                (PortId)consumer_input_port_id, edge_type_string_map[edge_info[input_node_name]]));
        graph->add_edge(graph_edges.back());
        /* // collect tm ops if there are any
        auto edge_attributes = graph->get_edge_attributes(graph_edges.back());
        auto tm_array = node_data["input_tms"][i].get<json::array>();
        for (auto tm_ : tm_array) {
            edge_attributes->append_tm(Optype{.op=tm_["op_type"]["type"], .attr=attr, .buda_attrs={}})
        } */
        i++;
    }
}
void build_graph_from_json(tt::graphlib::Graph * graph, std::string json_file) {
    using json = nlohmann::json;
    std::ifstream ifs(json_file); 
    json data = json::parse(ifs);
    ifs.close();
    for (auto node_name : data["topological_sorted_nodes"]) {
        json_to_node(graph, data, node_name);
    }
}
