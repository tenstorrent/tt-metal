#include "reportify/reportify.hpp"

#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ostream>
#include <filesystem>

#include "json.hpp"

#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/common.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/utils.hpp"
// #include "common/logger.hpp"

using json = nlohmann::json;
// using tt::LogReportify;
using namespace tt::graphlib;
namespace tt {

void to_json(json& j, BudaOpAttr const& attr) {
    if (const bool* v = std::get_if<bool>(&attr)) {
        j = *v;
    } else if (const int* v = std::get_if<int>(&attr)) {
        j = *v;
    } else if (const float* v = std::get_if<float>(&attr)) {
        j = *v;
    } else if (const std::string* v = std::get_if<std::string>(&attr)) {
        j = *v;
    }
    else if (const std::vector<int>* v = std::get_if<std::vector<int>>(&attr))
    {
        j = *v;
    }
    else
    {
        TT_ASSERT(false, "Unhandled variant type for BudaOpAttr");
    }
}

void to_json(json& j, BudaOpAttrs const& attrs) {
    for (auto const& [k, attr] : attrs) {
        to_json(j[k], attr);
    }
}
void to_json(json& j, std::vector<BudaOpAttr> const& attrs) {
    j["attrs"] = json::array();
    for (auto const& attr : attrs) {
        json json_value;
        to_json(json_value, attr);
        j["attrs"].push_back(json_value);
    }
}
void to_json(json& j, graphlib::OpType const& op_type) {
    j["op_type"] = {};
    j["op_type"]["type"] = op_type.op;
    to_json(j["op_type"], op_type.attr);
    to_json(j["op_type"]["buda_attrs"], op_type.buda_attrs);
}

}  // namespace tt

// namespace tt::balancer {

// void to_json(json& j, GridShape const& grid_shape) { j = {grid_shape.r, grid_shape.c}; }

// void to_json(json& j, TStreamFactor const& tsr)
// {
//     j["dir"] = tsr.none() ? "None" : ((tsr.dir == TStreamDir::R) ? "R" : "C");
//     j["factor"] = {tsr.r, tsr.c};
// }

// void to_json(json& j, BlockShape const& block_shape) {
//     j["t"] = block_shape.t;
//     j["mblock_m"] = block_shape.mblock_m;
//     j["mblock_n"] = block_shape.mblock_n;
//     j["ublock_rt"] = block_shape.ublock.rt;
//     j["ublock_ct"] = block_shape.ublock.ct;
// }

// void to_json(json& j, BufferModel const& buffer_model) {
//     j["block_shape"] = buffer_model.block_shape;
//     j["buffer_factor"] = buffer_model.buffer_factor;
//     j["l1_size_tiles"] = buffer_model.l1_size_tiles;
//     std::stringstream ss;
//     ss << buffer_model.data_format;
//     j["data_format"] = ss.str();
// }

// void to_json(json& j, TensorShape const& shape) { j = {shape.w, shape.z, shape.rt, shape.ct}; }

// void to_json(json& j, OpModel const& op_model) {
//     j["grid_shape"] = op_model.grid_shape;
//     j["t_stream_factor"] = op_model.t_stream_factor;
//     j["inputs"] = op_model.input_buffers;
//     j["outputs"] = op_model.output_buffers;
//     j["input_shapes"] = op_model.op_shape.inputs;
// }

// }  // namespace tt::balancer

namespace tt {

namespace reportify {
std::string canonical_dirname(std::string s)
{
    static std::string const chars = "/: ";
    for (char& c : s)
    {
        if (chars.find(c) != std::string::npos)
            c = '_';
    }
    return s;
}

template <class T>
std::string stream_operator_to_string(T obj) {
  std::ostringstream oss;
  oss << obj;
  return oss.str();
}

using JsonNamePair = std::pair<json, std::string>;
using JsonNamePairs = std::vector<JsonNamePair>;



std::vector<std::string> tt_nodes_to_name_strings(
    const std::vector<graphlib::Node*>& nodes) {
  std::vector<std::string> ret_vector;
  ret_vector.reserve(nodes.size());

  for (const graphlib::Node* node : nodes) {
    ret_vector.push_back(node->name());
  }

  return ret_vector;
}

// json node_to_json(const graphlib::Node* node, const graphlib::Graph *graph, const placer::PlacerSolution *placer_solution, std::shared_ptr<balancer::BalancerSolution> balancer_solution) {
json node_to_json(const graphlib::Node* node, const graphlib::Graph *graph) {
    json ret_json;
    ret_json["pybuda"] = 1; // marker to reportify to use new colouring scheme
    ret_json["name"] = node->name();
    ret_json["unique_id"] = node->id();
    std::vector<std::string> input_nodes;
    std::vector<std::string> output_nodes;
    std::vector<std::string> port_id_to_name_incoming;
    std::vector<std::string> port_id_to_name_outgoing;
    std::unordered_map<std::string, std::string> input_node_name_to_edge_type;

    for (auto incoming_edge : graph->operand_edges(node)) {

        graphlib::NodeId incoming_node_id = incoming_edge.producer_node_id;
        graphlib::Node* incoming_node = graph->node_by_id(incoming_node_id);
        std::string edge_type_string = graphlib::edge_type_to_string(incoming_edge.edge_type);

        std::string port_key_string =
            "port_" + std::to_string(incoming_edge.consumer_input_port_id);
        std::string incoming_port_info =
            edge_type_string + ": " + incoming_node->name() + " (" + port_key_string + ")";

        if (graph->get_ir_level() == graphlib::IRLevel::IR_BUDA and
            (incoming_edge.edge_type == graphlib::EdgeType::kData or
             incoming_edge.edge_type == graphlib::EdgeType::kDataLoopback)) {
            auto edge_attrs = graph->get_edge_attributes(incoming_edge);
            switch (edge_attrs->get_ublock_order()) {
                case graphlib::UBlockOrder::R: incoming_port_info += " ublock_order(r)"; break;
                case graphlib::UBlockOrder::C: incoming_port_info += " ublock_order(c)"; break;
            }
        }

        port_id_to_name_incoming.push_back(incoming_port_info);

        if (incoming_edge.edge_type != graphlib::EdgeType::kData and
            incoming_edge.edge_type != graphlib::EdgeType::kDataLoopback and
            incoming_edge.edge_type != graphlib::EdgeType::kControlLoop)
        {
            continue; // don't display others for now
        }

        input_nodes.push_back(incoming_node->name());
        input_node_name_to_edge_type.insert({incoming_node->name(), edge_type_string});
    }

    ret_json["input_nodes"] = input_nodes;
    ret_json["incoming_edge_port_info"] = port_id_to_name_incoming;
    ret_json["input_node_to_edge_type"] = input_node_name_to_edge_type;

    for (auto outgoing_edge : graph->user_edges(node)) {
        graphlib::NodeId outgoing_node_id = outgoing_edge.consumer_node_id;
        graphlib::Node* outgoing_node = graph->node_by_id(outgoing_node_id);
        output_nodes.push_back(outgoing_node->name());

        std::string port_key_string =
            "port_" + std::to_string(outgoing_edge.producer_output_port_id);
        std::string edge_type_string = graphlib::edge_type_to_string(outgoing_edge.edge_type);
        std::string outgoing_port_info =
            edge_type_string + ": " + outgoing_node->name() + " (" + port_key_string + ")";

        port_id_to_name_outgoing.push_back(outgoing_port_info);
    }

    ret_json["opcode"] = stream_operator_to_string(node->node_type());
    ret_json["cache"]["shape"] = node->shape().as_vector();

    ret_json["epoch"] = 0;
    // if (placer_solution != nullptr) {
    //     try {
    //         if (node->node_type() == graphlib::NodeType::kBudaOp) {
    //             placer::OpPlacement placement = placer_solution->name_to_op_placement.at(node->name());
    //             ret_json["grid_start"] = {placement.placed_cores.start.row, placement.placed_cores.start.col};
    //             ret_json["grid_end"] = {placement.placed_cores.end.row, placement.placed_cores.end.col};
    //             ret_json["epoch"] = placer_solution->temporal_epoch_id(node->name());
    //             ret_json["chip_id"] = placer_solution->chip_id(node->name());
    //         }
    //         else if (node->node_type() == graphlib::NodeType::kInput) {
    //             ret_json["epoch"] = placer_solution->temporal_epoch_id(graph->data_users(node)[0]->name());
    //             ret_json["chip_id"] = placer_solution->chip_id(graph->data_users(node)[0]->name());
    //         }
    //         else if (node->node_type() == graphlib::NodeType::kOutput) {
    //             ret_json["epoch"] = placer_solution->temporal_epoch_id(graph->data_operands(node)[0]->name());
    //             ret_json["chip_id"] = placer_solution->chip_id(graph->data_operands(node)[0]->name());
    //         }

    //     } catch (std::out_of_range &e) {
    //         // log_warning(tt::LogReportify, "Node {} has no placement, skipping.", node->name());
    //     }
    // }

    // if (balancer_solution and balancer_solution->op_models.find(node->name()) != balancer_solution->op_models.end())
    // {
    //     balancer::OpModel const& op_model = balancer_solution->op_models.at(node->name());
    //     ret_json["op_model"] = op_model;
    // }

    ret_json["epoch_type"] = graphlib::node_epoch_type_to_string(node->get_epoch_type());
    ret_json["output_nodes"] = output_nodes;
    ret_json["outgoing_edge_port_info"] = port_id_to_name_outgoing;

    ret_json["type"] = stream_operator_to_string(node->node_type());
    if (node->node_type() == graphlib::NodeType::kInput) {
        // Keep constants and accumulators inside the epoch to better visualize what's happening
        if (node->as<graphlib::InputNode>()->is_constant()) {

            ret_json["class"] = node->as<graphlib::InputNode>()->input_type_string();
            ret_json["type"] = "Constant";

            const graphlib::ConstantInputNode *cnode = node->as<graphlib::ConstantInputNode>();
            if (cnode->is_single_value()) {
                ret_json["constant_value"] = std::to_string(cnode->constant_value());
                ret_json["constant_dims"] = cnode->constant_dims();
            } else if (cnode->is_single_tile()) {
                ret_json["constant_tile"] = cnode->tile_value();
            } else if (cnode->is_tensor()) {
                ret_json["constant_dims"] = cnode->tensor_shape().as_vector();
            }
        } else if (node->as<graphlib::InputNode>()->is_accumulator()) {
            ret_json["class"] = "accumulator";
            ret_json["type"] = "Accumulator";
        } else {
            ret_json["class"] = "Input::";
            ret_json["type"] = "Input::" + node->as<graphlib::InputNode>()->input_type_string();
        }
        ret_json["queue_type"] = node->as<graphlib::QueueNode>()->queue_type_string();
        ret_json["is_cross_epoch_type"] = 
            node->as<graphlib::QueueNode>()->is_epoch_to_epoch() and
            node->as<graphlib::EpochToEpochQueueNode>()->is_cross_epoch_type();
        ret_json["memory_access"] = node->as<graphlib::QueueNode>()->memory_access_type_string();
        ret_json["tile_broadcast"] = node->as<graphlib::InputNode>()->get_tile_broadcast_dims();
        ret_json["requires_grad"] = node->as<graphlib::InputNode>()->requires_grad();
    } else if (node->node_type() == graphlib::NodeType::kOutput) {
        ret_json["class"] = "Output";
        ret_json["queue_type"] = node->as<graphlib::QueueNode>()->queue_type_string();
        ret_json["is_cross_epoch_type"] = 
            node->as<graphlib::QueueNode>()->is_epoch_to_epoch() and
            node->as<graphlib::EpochToEpochQueueNode>()->is_cross_epoch_type();
        ret_json["memory_access"] = node->as<graphlib::QueueNode>()->memory_access_type_string();
    } else if (node->node_type() == graphlib::NodeType::kPyOp) {
        const graphlib::PyOpNode *opnode = node->as<graphlib::PyOpNode>();
        ret_json["ir"] = "pybuda";
        ret_json["class"] = opnode->op_type().as_string();
        ret_json["type"] = opnode->op_type().op;
        to_json(ret_json, opnode->op_type());
        ret_json["gradient_op"] = opnode->is_gradient_op();

    } else if (node->node_type() == graphlib::NodeType::kBudaOp) {
        const graphlib::BudaOpNode *opnode = node->as<graphlib::BudaOpNode>();
        ret_json["ir"] = "buda";
        ret_json["class"] = opnode->op_type().as_string();
        ret_json["type"] = opnode->op_type().op;
        ret_json["tags"] = opnode->op_type().tags;
        to_json(ret_json, opnode->op_type());
        ret_json["gradient_op"] = opnode->is_gradient_op();
        {
            std::stringstream ss; ss << opnode->intermediate_df();
            ret_json["intermediate_df"] = ss.str();
        }
        {
            std::stringstream ss; ss << opnode->math_fidelity();
            ret_json["fidelity"] = ss.str();
        }

        // if (opnode->is_fused_op())
        // {
        //     std::vector<std::vector<std::string>> schedules;

        //     auto fused_op = opnode->get_fused_op();
        //     for (const auto &schedule: fused_op->get_schedules())
        //     {
        //         std::vector<std::string> sch;
        //         for (const auto &op: schedule.ops)
        //         {
        //             auto sh = op.op_shape.outputs.at(0);
        //             std::string shape = std::to_string(sh.w) + "," + std::to_string(sh.z) 
        //                 + "," + std::to_string(sh.rt) + "," + std::to_string(sh.ct);
        //             sch.push_back(op.name + ": " + op.op_type.op + " (" + shape + "), out: " + std::to_string(op.output_buffer));
        //         }
        //         schedules.push_back(sch);
        //     }

        //     ret_json["schedules"] = schedules;
        // }
    }
    else if (node->node_type() == graphlib::NodeType::kBudaNaryTM)
    {
        const graphlib::BudaNaryTMNode* tmnode = node->as<graphlib::BudaNaryTMNode>();
        ret_json["ir"] = "buda";
        ret_json["class"] = tmnode->op_type().as_string();
        ret_json["type"] = tmnode->op_type().op;
        to_json(ret_json, tmnode->op_type());
    }
    else if (node->node_type() == graphlib::NodeType::kQueue)
    {
        ret_json["class"] = "BudaDramQueue::";
        ret_json["queue_type"] = node->as<graphlib::QueueNode>()->queue_type_string();
        ret_json["is_cross_epoch_type"] = 
            node->as<graphlib::QueueNode>()->is_epoch_to_epoch() and
            node->as<graphlib::EpochToEpochQueueNode>()->is_cross_epoch_type();
        ret_json["memory_access"] = node->as<graphlib::QueueNode>()->memory_access_type_string();

        // if (balancer_solution)
        // {
        //     auto operands = graph->data_operands(node);
        //     TT_ASSERT(operands.size() == 1);
        //     TT_ASSERT(operands[0]->node_type() == graphlib::NodeType::kBudaOp);
        //     balancer::OpModel const& op_model = balancer_solution->op_models.at(operands[0]->name());
        //     ret_json["op_model"] = {{"t_stream_factor", op_model.t_stream_factor}};
        // }
    }
    std::stringstream ss; ss << node->output_df();
    ret_json["output_df"] = ss.str();

    // Record input TMs, if any, on the input edges
    for (graphlib::Edge e : graph->operand_data_edges(node)) {
        std::vector<graphlib::OpType> tms = graph->get_edge_attributes(e)->get_tms();
        ret_json["input_tms"][e.consumer_input_port_id] = json::array();
        if (tms.size() > 0) {
            for (const auto& tm : tms)
            {
                json j;
                to_json(j, tm);
                ret_json["input_tms"][e.consumer_input_port_id].push_back(j);
            }
        }
    }

    return ret_json;
}
using JsonNamePair = std::pair<json, std::string>;
using JsonNamePairs = std::vector<JsonNamePair>;

void write_json_to_file(const std::string& path, json json_file) {
    std::ofstream o(path);
    o << std::setw(4) << json_file;
    //cout << std::setw(4) << json_file;
}

JsonNamePairs create_jsons_for_graph(
        const std::string& graph_prefix,
        const graphlib::Graph *graph,
        std::function<bool(graphlib::Node*)> node_filter = [](graphlib::Node*) { return true;});
// JsonNamePairs create_jsons_for_graph(
//         const std::string& graph_prefix,
//         const graphlib::Graph *graph,
//         const placer::PlacerSolution *placer_solution,
//         std::shared_ptr<balancer::BalancerSolution> balancer_solution,
//         std::function<bool(graphlib::Node*)> node_filter = [](graphlib::Node*) { return true;});

void dump_graph(
    const std::string& path,
    const std::string& test_name,
    const std::string& graph_prefix,
    const graphlib::Graph* graph,
    // const placer::PlacerSolution* placer_solution,
    // std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    const std::string& report_path)
{
    JsonNamePairs json_pairs = create_jsons_for_graph(graph_prefix, graph);

    initalize_reportify_directory(path, test_name);

    std::string sage_report_path = path + test_name + "/" + report_path + "/";
    std::string subgraph_path = sage_report_path + graph_prefix + "_graphs/";

    // log_debug(tt::LogReportify, "Writing graph to {}", subgraph_path);

    std::filesystem::create_directories(subgraph_path);

    json root_json = json_pairs.back().first;
    std::string root_json_name = json_pairs.back().second;

    std::string root_json_path = sage_report_path + /*graph_prefix +*/ root_json_name;
    write_json_to_file(root_json_path, root_json);
}

void dump_consteval_graph(const std::string& test_name, const std::string& graph_prefix, const graphlib::Graph* graph)
{
    return dump_graph(test_name, canonical_dirname(graph_prefix), graph, "/buda_reports/Consteval/");
}

void dump_epoch_type_graphs(
        const std::string& test_name,
        const std::string& graph_prefix, 
        const graphlib::Graph *graph,
        // const placer::PlacerSolution *placer_solution,
        // std::shared_ptr<balancer::BalancerSolution> balancer_solution,
        const std::string& directory_path) {

  std::function<bool(graphlib::Node*, NodeEpochType epoch_type, const graphlib::Graph *graph)> epoch_type_filter =
      [](graphlib::Node* node, NodeEpochType epoch_type, const graphlib::Graph *graph) {
          if (node->node_type() == graphlib::NodeType::kInput or node->node_type() == graphlib::NodeType::kQueue) {
              // we want to write out input/queue nodes if they happen to be produced/consumed
              // by a node belonging to the queried epoch_type
              for (graphlib::Node* user : graph->data_users(node)) {
                  if (user->get_epoch_type() == epoch_type) { return true; }
              }
              for (graphlib::Node* operand : graph->data_operands(node)) {
                  if (operand->get_epoch_type() == epoch_type) { return true; }
              }
              return false;
          }

          return node->get_epoch_type() == epoch_type;
      };

  initalize_reportify_directory(directory_path , test_name);

  std::string report_path = get_epoch_type_report_relative_directory();
  std::string sage_report_path = directory_path  + test_name + "/" + report_path + "/";

  std::string subgraph_path = sage_report_path + graph_prefix + "_graphs/";

//   log_debug(tt::LogReportify, "Writing graph to {}", subgraph_path);

  std::filesystem::create_directories(subgraph_path);

  for (NodeEpochType epoch_type : {NodeEpochType::Forward, NodeEpochType::Backward, NodeEpochType::Optimizer}) {
    if ( (epoch_type == NodeEpochType::Backward and not graph->contains_bwd_nodes()) or
         (epoch_type == NodeEpochType::Optimizer and not graph->contains_opt_nodes()) )
    {
        continue;
    }

    auto node_epoch_type_filter = std::bind(epoch_type_filter, std::placeholders::_1, epoch_type, graph);
    JsonNamePairs new_json_pairs = create_jsons_for_graph(
            graph_prefix + graphlib::node_epoch_type_to_string(epoch_type),
            graph,
            // placer_solution,
            // balancer_solution,
            node_epoch_type_filter);

      for (const auto& [json, json_name] : new_json_pairs) {
          std::string root_json_path = sage_report_path + graph_prefix + json_name;
          write_json_to_file(root_json_path, json);
      }
  }
}

// void dump_epoch_id_graphs(
//         const std::string& test_name,
//         const std::string& graph_prefix,
//         const graphlib::Graph *graph,
//         const placer::PlacerSolution *placer_solution,
//         std::shared_ptr<balancer::BalancerSolution> balancer_solution,
//         const std::string& directory_path) {

//   if (placer_solution == nullptr) {
//     //   log_warning(tt::LogReportify, "dump_epoch_id_graphs(..) invoked without placer_solution argument, no dumps written");
//       return;
//   }

//   std::function<bool(graphlib::Node*, uint32_t epoch_id, const graphlib::Graph *graph, const placer::PlacerSolution *placer_solution)>
//       epoch_id_filter = [](graphlib::Node* node, uint32_t epoch_id, const graphlib::Graph *graph, const placer::PlacerSolution *placer_solution) {
//           if (node->node_type() == graphlib::NodeType::kBudaOp) {
//             return placer_solution->temporal_epoch_id(node->name()) == epoch_id;
//           }

//           for (graphlib::Node* user : graph->data_users(node)) {
//               if (placer_solution->name_to_op_placement.find(user->name()) != placer_solution->name_to_op_placement.end()) {
//                   if (placer_solution->temporal_epoch_id(user->name()) == epoch_id) { return true; }
//               }

//           }
//           for (graphlib::Node* operand : graph->data_operands(node)) {
//               if (placer_solution->name_to_op_placement.find(operand->name()) != placer_solution->name_to_op_placement.end()) {
//                   if (placer_solution->temporal_epoch_id(operand->name()) == epoch_id) { return true; }
//               }
//           }
//           return false;
//       };

//   initalize_reportify_directory(directory_path , test_name);

//   std::string report_path = get_epoch_id_report_relative_directory();
//   std::string sage_report_path = directory_path  + test_name + "/" + report_path + "/";
//   std::string subgraph_path = sage_report_path + graph_prefix + "_graphs/";

// //   log_debug(tt::LogReportify, "Writing graph to {}", subgraph_path);

//   std::filesystem::create_directories(subgraph_path);

//   for (uint32_t epoch_id = 0; epoch_id < placer_solution->num_epochs; ++epoch_id) {

//     auto node_epoch_id_filter = std::bind(epoch_id_filter, std::placeholders::_1, epoch_id, graph, placer_solution);
//     JsonNamePairs new_json_pairs = create_jsons_for_graph(
//             graph_prefix + "_epoch_id_" + std::to_string(epoch_id),
//             graph,
//             placer_solution,
//             balancer_solution,
//             node_epoch_id_filter);

//       for (const auto& [json, json_name] : new_json_pairs) {
//           std::string root_json_path = sage_report_path + graph_prefix + json_name;
//           write_json_to_file(root_json_path, json);
//       }
//   }

// }

json create_json_for_graph(
    const graphlib::Graph *graph,
    // const placer::PlacerSolution *placer_solution,
    // std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    std::function<bool(graphlib::Node*)> node_filter)
{
    json this_json;
    this_json["topological_sorted_nodes"] = {};
    for (graphlib::Node* node : graphlib::topological_sort(*graph)) {
        if (node_filter(node)) {
            this_json["nodes"][node->name()] = node_to_json(node, graph);
            this_json["graph"] = std::unordered_map<std::string, std::string>();
            this_json["topological_sorted_nodes"].push_back(node->name());
        }
    }
    return this_json;
}

JsonNamePairs create_jsons_for_graph(
        const std::string& graph_prefix,
        const graphlib::Graph *graph,
        // const placer::PlacerSolution *placer_solution,
        // std::shared_ptr<balancer::BalancerSolution> balancer_solution,
        std::function<bool(graphlib::Node*)> node_filter)
{

    JsonNamePairs this_json_name_pairs;

    json this_json = create_json_for_graph(graph, node_filter);
    std::string this_name = graph_prefix + ".buda";
    JsonNamePair this_json_name_pair = std::make_pair(this_json, this_name);
    this_json_name_pairs.push_back(this_json_name_pair);

    return this_json_name_pairs;
}

void dump_graph(
    const std::string& test_name,
    const std::string& graph_prefix,
    const graphlib::Graph* graph,
    // const placer::PlacerSolution* placer_solution,
    // std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    const std::string& report_path)
{
    std::string default_dir = get_default_reportify_path("");
    dump_graph(default_dir, test_name, graph_prefix, graph, report_path);
}

}  // namespace reportify 
} // namespace tt
