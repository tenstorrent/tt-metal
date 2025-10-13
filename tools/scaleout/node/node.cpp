// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "node.hpp"
#include "node_types.hpp"

#include <enchantum/enchantum.hpp>
#include <stdexcept>

namespace tt::scaleout_tools {

namespace {
// Helper function to add a connection between two ports
void add_connection(
    google::protobuf::RepeatedPtrField<tt::scaleout_tools::cabling_generator::proto::NodeDescriptor_Connection>*
        connections,
    int tray_a,
    int port_a,
    int tray_b,
    int port_b) {
    auto* connection = connections->Add();
    connection->mutable_port_a()->set_tray_id(tray_a);
    connection->mutable_port_a()->set_port_id(port_a);
    connection->mutable_port_b()->set_tray_id(tray_b);
    connection->mutable_port_b()->set_port_id(port_b);
}

// Helper function to add boards to a node
void add_boards(
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor* node,
    const std::string& board_type,
    int start_tray,
    int end_tray) {
    auto* boards = node->mutable_boards();
    for (int i = start_tray; i <= end_tray; ++i) {
        auto* board = boards->add_board();
        board->set_tray_id(i);
        board->set_board_type(board_type);
    }
}

// Helper function to get or create port type connections
google::protobuf::RepeatedPtrField<tt::scaleout_tools::cabling_generator::proto::NodeDescriptor_Connection>*
get_port_connections(tt::scaleout_tools::cabling_generator::proto::NodeDescriptor* node, const std::string& port_type) {
    return (*node->mutable_port_type_connections())[port_type].mutable_connections();
}

}  // anonymous namespace

// N300 Node class
class N300T3KNode {
protected:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create(const std::string& motherboard) {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard(motherboard);

        // Add boards
        add_boards(&node, "N300", 1, 4);

        // Add QSFP connections
        auto* const qsfp_connections = get_port_connections(&node, "QSFP_DD");
        add_connection(qsfp_connections, 1, 1, 4, 1);
        add_connection(qsfp_connections, 2, 2, 3, 2);

        // Add WARP100 connections
        auto* const warp100_connections = get_port_connections(&node, "WARP100");
        add_connection(warp100_connections, 1, 1, 2, 1);
        add_connection(warp100_connections, 1, 2, 2, 2);
        add_connection(warp100_connections, 3, 1, 4, 1);
        add_connection(warp100_connections, 3, 2, 4, 2);

        return node;
    }
};

// N300 LB Node class
class N300LBNode : public N300T3KNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        return N300T3KNode::create("X12DPG-QT6");
    }
};

// N300 QB Node class
class N300QBNode : public N300T3KNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        return N300T3KNode::create("SIENAD8-2L2T");
    }
};

class WHGalaxyNode {
private:
    // Add X-torus QSFP connections
    static void add_x_torus_connections(tt::scaleout_tools::cabling_generator::proto::NodeDescriptor* node) {
        auto* const qsfp_connections = get_port_connections(node, "QSFP_DD");
        add_connection(qsfp_connections, 1, 3, 2, 3);
        add_connection(qsfp_connections, 1, 4, 2, 4);
        add_connection(qsfp_connections, 1, 5, 2, 5);
        add_connection(qsfp_connections, 1, 6, 2, 6);
        add_connection(qsfp_connections, 3, 6, 4, 6);
        add_connection(qsfp_connections, 3, 5, 4, 5);
        add_connection(qsfp_connections, 3, 4, 4, 4);
        add_connection(qsfp_connections, 3, 3, 4, 3);
    }

    // Add Y-torus QSFP connections
    static void add_y_torus_connections(tt::scaleout_tools::cabling_generator::proto::NodeDescriptor* node) {
        auto* const qsfp_connections = get_port_connections(node, "QSFP_DD");
        add_connection(qsfp_connections, 1, 2, 3, 2);
        add_connection(qsfp_connections, 1, 1, 3, 1);
        add_connection(qsfp_connections, 2, 1, 4, 1);
        add_connection(qsfp_connections, 2, 2, 4, 2);
    }

public:
    // WHGalaxy topology options using one-hot encoding:
    // MESH = 0 (00) - Mesh
    // X_TORUS = 0b01 - X-axis torus only (bit 0)
    // Y_TORUS = 0b10 - Y-axis torus only (bit 1)
    // XY_TORUS = 0b11 - Both X and Y torus (bits 0+1)
    enum class WHGalaxyTopology {
        MESH = 0b00,                   // Mesh
        X_TORUS = 0b01,                // X-axis torus only
        Y_TORUS = 0b10,                // Y-axis torus only
        XY_TORUS = X_TORUS | Y_TORUS,  // Both X and Y torus
    };

    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create(
        WHGalaxyTopology topology = WHGalaxyTopology::MESH) {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard("S7T-MB");

        // TODO: Use UBB_WORMHOLE instead of UBB. Currently use UBB due enum aliasing not being supported for enum
        // reflection Add boards
        add_boards(&node, "UBB", 1, 4);

        // Add LINKING_BOARD_1 connections
        auto* const lb1_connections = get_port_connections(&node, "LINKING_BOARD_1");
        add_connection(lb1_connections, 1, 1, 2, 1);
        add_connection(lb1_connections, 1, 2, 2, 2);
        add_connection(lb1_connections, 3, 1, 4, 1);
        add_connection(lb1_connections, 3, 2, 4, 2);

        // Add LINKING_BOARD_2 connections
        auto* const lb2_connections = get_port_connections(&node, "LINKING_BOARD_2");
        add_connection(lb2_connections, 1, 1, 2, 1);
        add_connection(lb2_connections, 1, 2, 2, 2);
        add_connection(lb2_connections, 3, 1, 4, 1);
        add_connection(lb2_connections, 3, 2, 4, 2);

        // Add LINKING_BOARD_3 connections
        auto* const lb3_connections = get_port_connections(&node, "LINKING_BOARD_3");
        add_connection(lb3_connections, 1, 1, 3, 1);
        add_connection(lb3_connections, 1, 2, 3, 2);
        add_connection(lb3_connections, 2, 1, 4, 1);
        add_connection(lb3_connections, 2, 2, 4, 2);

        // Add QSFP connections based on topology (one-hot encoded)
        if (static_cast<int>(topology) & static_cast<int>(WHGalaxyTopology::X_TORUS)) {  // X_TORUS bit
            add_x_torus_connections(&node);
        }
        if (static_cast<int>(topology) & static_cast<int>(WHGalaxyTopology::Y_TORUS)) {  // Y_TORUS bit
            add_y_torus_connections(&node);
        }

        return node;
    }
};

// WH Galaxy X Torus Node class
class WHGalaxyXTorusNode : public WHGalaxyNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        return WHGalaxyNode::create(WHGalaxyTopology::X_TORUS);
    }
};

// WH Galaxy Y Torus Node class
class WHGalaxyYTorusNode : public WHGalaxyNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        return WHGalaxyNode::create(WHGalaxyTopology::Y_TORUS);
    }
};

// WH Galaxy XY Torus Node class
class WHGalaxyXYTorusNode : public WHGalaxyNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        return WHGalaxyNode::create(WHGalaxyTopology::XY_TORUS);
    }
};

// P150 QB Global Node class
class P150QBGlobalNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard("SIENAD8-2L2T");

        // Add boards
        add_boards(&node, "P150", 1, 4);

        // Add QSFP connections
        auto* const qsfp_connections = get_port_connections(&node, "QSFP_DD");
        add_connection(qsfp_connections, 1, 1, 2, 1);
        add_connection(qsfp_connections, 1, 2, 2, 2);
        add_connection(qsfp_connections, 1, 3, 4, 3);
        add_connection(qsfp_connections, 1, 4, 4, 4);
        add_connection(qsfp_connections, 2, 3, 3, 3);
        add_connection(qsfp_connections, 2, 4, 3, 4);
        add_connection(qsfp_connections, 3, 1, 4, 1);
        add_connection(qsfp_connections, 3, 2, 4, 2);

        return node;
    }
};

// P300 QB America Node class
class P300QBAmericaNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard("SIENAD8-2L2T");

        // Add boards
        add_boards(&node, "P300", 1, 2);

        // Add WARP400 connections
        auto* const warp400_connections = get_port_connections(&node, "WARP400");
        add_connection(warp400_connections, 1, 1, 2, 1);
        add_connection(warp400_connections, 1, 2, 2, 2);

        return node;
    }
};

class BHGalaxyNode {
private:
    // Add X-torus QSFP connections
    static void add_x_torus_connections(tt::scaleout_tools::cabling_generator::proto::NodeDescriptor* node) {
        auto* const qsfp_connections = get_port_connections(node, "QSFP_DD");
        add_connection(qsfp_connections, 1, 3, 2, 3);
        add_connection(qsfp_connections, 1, 4, 2, 4);
        add_connection(qsfp_connections, 1, 5, 2, 5);
        add_connection(qsfp_connections, 1, 6, 2, 6);
        add_connection(qsfp_connections, 3, 6, 4, 6);
        add_connection(qsfp_connections, 3, 5, 4, 5);
        add_connection(qsfp_connections, 3, 4, 4, 4);
        add_connection(qsfp_connections, 3, 3, 4, 3);
    }

    // Add Y-torus QSFP connections
    static void add_y_torus_connections(tt::scaleout_tools::cabling_generator::proto::NodeDescriptor* node) {
        auto* const qsfp_connections = get_port_connections(node, "QSFP_DD");
        add_connection(qsfp_connections, 1, 2, 3, 2);
        add_connection(qsfp_connections, 1, 1, 3, 1);
        add_connection(qsfp_connections, 2, 1, 4, 1);
        add_connection(qsfp_connections, 2, 2, 4, 2);
    }

public:
    // BHGalaxy topology options using one-hot encoding:
    // MESH = 0 (00) - Mesh
    // X_TORUS = 0b01 - X-axis torus only (bit 0)
    // Y_TORUS = 0b10 - Y-axis torus only (bit 1)
    // XY_TORUS = 0b11 - Both X and Y torus (bits 0+1)
    enum class BHGalaxyTopology {
        MESH = 0b00,                   // Mesh
        X_TORUS = 0b01,                // X-axis torus only
        Y_TORUS = 0b10,                // Y-axis torus only
        XY_TORUS = X_TORUS | Y_TORUS,  // Both X and Y torus
    };

    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create(
        BHGalaxyTopology topology = BHGalaxyTopology::MESH) {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard("S7T-MB");

        // Add boards
        add_boards(&node, "UBB_BLACKHOLE", 1, 4);

        // Add LINKING_BOARD_1 connections
        auto* const lb1_connections = get_port_connections(&node, "LINKING_BOARD_1");
        add_connection(lb1_connections, 1, 1, 2, 1);
        add_connection(lb1_connections, 1, 2, 2, 2);
        add_connection(lb1_connections, 3, 1, 4, 1);
        add_connection(lb1_connections, 3, 2, 4, 2);

        // Add LINKING_BOARD_2 connections
        auto* const lb2_connections = get_port_connections(&node, "LINKING_BOARD_2");
        add_connection(lb2_connections, 1, 1, 2, 1);
        add_connection(lb2_connections, 1, 2, 2, 2);
        add_connection(lb2_connections, 3, 1, 4, 1);
        add_connection(lb2_connections, 3, 2, 4, 2);

        // Add LINKING_BOARD_3 connections
        auto* const lb3_connections = get_port_connections(&node, "LINKING_BOARD_3");
        add_connection(lb3_connections, 1, 1, 3, 1);
        add_connection(lb3_connections, 1, 2, 3, 2);
        add_connection(lb3_connections, 2, 1, 4, 1);
        add_connection(lb3_connections, 2, 2, 4, 2);

        // Add QSFP connections based on topology (one-hot encoded)
        if (static_cast<int>(topology) & static_cast<int>(BHGalaxyTopology::X_TORUS)) {  // X_TORUS bit
            add_x_torus_connections(&node);
        }
        if (static_cast<int>(topology) & static_cast<int>(BHGalaxyTopology::Y_TORUS)) {  // Y_TORUS bit
            add_y_torus_connections(&node);
        }

        return node;
    }
};

// BH Galaxy X Torus Node class
class BHGalaxyXTorusNode : public BHGalaxyNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        return BHGalaxyNode::create(BHGalaxyTopology::X_TORUS);
    }
};

// BH Galaxy Y Torus Node class
class BHGalaxyYTorusNode : public BHGalaxyNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        return BHGalaxyNode::create(BHGalaxyTopology::Y_TORUS);
    }
};

// BH Galaxy XY Torus Node class
class BHGalaxyXYTorusNode : public BHGalaxyNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        return BHGalaxyNode::create(BHGalaxyTopology::XY_TORUS);
    }
};

// Factory function to create node descriptors by name
tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_node_descriptor(NodeType node_type) {
    switch (node_type) {
        case NodeType::N300_LB: return N300LBNode::create();
        case NodeType::N300_QB: return N300QBNode::create();
        case NodeType::WH_GALAXY: return WHGalaxyNode::create();
        case NodeType::WH_GALAXY_X_TORUS: return WHGalaxyXTorusNode::create();
        case NodeType::WH_GALAXY_Y_TORUS: return WHGalaxyYTorusNode::create();
        case NodeType::WH_GALAXY_XY_TORUS: return WHGalaxyXYTorusNode::create();
        case NodeType::P150_QB_GLOBAL: return P150QBGlobalNode::create();
        case NodeType::P300_QB_AMERICA: return P300QBAmericaNode::create();
        case NodeType::BH_GALAXY: return BHGalaxyNode::create();
        case NodeType::BH_GALAXY_X_TORUS: return BHGalaxyXTorusNode::create();
        case NodeType::BH_GALAXY_Y_TORUS: return BHGalaxyYTorusNode::create();
        case NodeType::BH_GALAXY_XY_TORUS: return BHGalaxyXYTorusNode::create();
    }
    throw std::runtime_error("Unknown node type: " + std::string(enchantum::to_string(node_type)));
}

}  // namespace tt::scaleout_tools
