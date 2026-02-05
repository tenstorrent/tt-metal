// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "node.hpp"
#include "node_types.hpp"

#include <enchantum/enchantum.hpp>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

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
class N300T3KNode : public NodeBase {
public:
    Architecture get_architecture() const override { return Architecture::WORMHOLE; }

protected:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_impl(
        const std::string& motherboard, const bool default_cabling = false) {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard(motherboard);

        // Add boards
        add_boards(&node, "N300", 1, 4);

        // Add QSFP connections
        if (default_cabling) {
            auto* const qsfp_connections = get_port_connections(&node, "QSFP_DD");
            add_connection(qsfp_connections, 1, 1, 4, 1);
            add_connection(qsfp_connections, 2, 2, 3, 2);
        }

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
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl("X12DPG-QT6", false);
    }
};

class N300LBDefaultNode : public N300T3KNode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl("X12DPG-QT6", true);
    }
};

// N300 QB Node class
class N300QBNode : public N300T3KNode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl("SIENAD8-2L2T", false);
    }
};

class N300QBDefaultNode : public N300T3KNode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl("SIENAD8-2L2T", true);
    }
};

class WHGalaxyNode : public NodeBase {
public:
    Architecture get_architecture() const override { return Architecture::WORMHOLE; }

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

protected:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_impl(
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

public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl(WHGalaxyTopology::MESH);
    }
};

// WH Galaxy X Torus Node class
class WHGalaxyXTorusNode : public WHGalaxyNode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl(WHGalaxyTopology::X_TORUS);
    }
    Topology get_topology() const override { return Topology::X_TORUS; }
};

// WH Galaxy Y Torus Node class
class WHGalaxyYTorusNode : public WHGalaxyNode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl(WHGalaxyTopology::Y_TORUS);
    }
    Topology get_topology() const override { return Topology::Y_TORUS; }
};

// WH Galaxy XY Torus Node class
class WHGalaxyXYTorusNode : public WHGalaxyNode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl(WHGalaxyTopology::XY_TORUS);
    }
    Topology get_topology() const override { return Topology::XY_TORUS; }
};

class P150LBNode : public NodeBase {
public:
    Architecture get_architecture() const override { return Architecture::BLACKHOLE; }

    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard("H13DSG-O-CPU");

        add_boards(&node, "P150", 1, 8);

        return node;
    }
};

// P150 QB AE Node class
class P150QBAENode : public NodeBase {
public:
    Architecture get_architecture() const override { return Architecture::BLACKHOLE; }

protected:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_impl(
        const bool default_cabling = false) {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard("SIENAD8-2L2T");

        // Add boards
        add_boards(&node, "P150", 1, 4);

        // Add QSFP connections
        if (default_cabling) {
            auto* const qsfp_connections = get_port_connections(&node, "QSFP_DD");
            add_connection(qsfp_connections, 1, 1, 2, 1);
            add_connection(qsfp_connections, 1, 2, 2, 2);
            add_connection(qsfp_connections, 1, 3, 4, 3);
            add_connection(qsfp_connections, 1, 4, 4, 4);
            add_connection(qsfp_connections, 2, 3, 3, 3);
            add_connection(qsfp_connections, 2, 4, 3, 4);
            add_connection(qsfp_connections, 3, 1, 4, 1);
            add_connection(qsfp_connections, 3, 2, 4, 2);
        }

        return node;
    }

public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override { return create_impl(false); }
};

class P150QBAEDefaultNode : public P150QBAENode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override { return create_impl(true); }
};


// P300 QB GE Node class
class P300QBGENode : public NodeBase {
public:
    Architecture get_architecture() const override { return Architecture::BLACKHOLE; }

    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard("B850M-C");

        // Add boards
        add_boards(&node, "P300", 1, 2);

        // Add WARP400 connections
        auto* const warp400_connections = get_port_connections(&node, "WARP400");
        add_connection(warp400_connections, 1, 1, 2, 1);

        return node;
    }
};

class BHGalaxyNode : public NodeBase {
public:
    Architecture get_architecture() const override { return Architecture::BLACKHOLE; }

private:
    // Add X-torus QSFP connections
    static void add_x_torus_connections(tt::scaleout_tools::cabling_generator::proto::NodeDescriptor* node) {
        auto* const qsfp_connections = get_port_connections(node, "QSFP_DD");
        add_connection(qsfp_connections, 1, 3, 3, 3);
        add_connection(qsfp_connections, 1, 4, 3, 4);
        add_connection(qsfp_connections, 1, 5, 3, 5);
        add_connection(qsfp_connections, 1, 6, 3, 6);
        add_connection(qsfp_connections, 2, 6, 4, 6);
        add_connection(qsfp_connections, 2, 5, 4, 5);
        add_connection(qsfp_connections, 2, 4, 4, 4);
        add_connection(qsfp_connections, 2, 3, 4, 3);
    }

    // Add Y-torus QSFP connections
    static void add_y_torus_connections(tt::scaleout_tools::cabling_generator::proto::NodeDescriptor* node) {
        auto* const qsfp_connections = get_port_connections(node, "QSFP_DD");
        add_connection(qsfp_connections, 1, 2, 2, 2);
        add_connection(qsfp_connections, 1, 1, 2, 1);
        add_connection(qsfp_connections, 3, 1, 4, 1);
        add_connection(qsfp_connections, 3, 2, 4, 2);
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

protected:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_impl(
        BHGalaxyTopology topology = BHGalaxyTopology::MESH) {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard("S7T-MB");

        // Add boards
        add_boards(&node, "UBB_BLACKHOLE", 1, 4);

        // Add LINKING_BOARD_1 connections
        auto* const lb1_connections = get_port_connections(&node, "LINKING_BOARD_1");
        add_connection(lb1_connections, 1, 1, 3, 1);
        add_connection(lb1_connections, 1, 2, 3, 2);
        add_connection(lb1_connections, 2, 1, 4, 1);
        add_connection(lb1_connections, 2, 2, 4, 2);

        // Add LINKING_BOARD_2 connections
        auto* const lb2_connections = get_port_connections(&node, "LINKING_BOARD_2");
        add_connection(lb2_connections, 1, 1, 3, 1);
        add_connection(lb2_connections, 1, 2, 3, 2);
        add_connection(lb2_connections, 2, 1, 4, 1);
        add_connection(lb2_connections, 2, 2, 4, 2);

        // Add LINKING_BOARD_3 connections
        auto* const lb3_connections = get_port_connections(&node, "LINKING_BOARD_3");
        add_connection(lb3_connections, 1, 1, 2, 1);
        add_connection(lb3_connections, 1, 2, 2, 2);
        add_connection(lb3_connections, 3, 1, 4, 1);
        add_connection(lb3_connections, 3, 2, 4, 2);

        // Add QSFP connections based on topology (one-hot encoded)
        if (static_cast<int>(topology) & static_cast<int>(BHGalaxyTopology::X_TORUS)) {  // X_TORUS bit
            add_x_torus_connections(&node);
        }
        if (static_cast<int>(topology) & static_cast<int>(BHGalaxyTopology::Y_TORUS)) {  // Y_TORUS bit
            add_y_torus_connections(&node);
        }

        return node;
    }

public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl(BHGalaxyTopology::MESH);
    }
};

// BH Galaxy X Torus Node class
class BHGalaxyXTorusNode : public BHGalaxyNode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl(BHGalaxyTopology::X_TORUS);
    }
    Topology get_topology() const override { return Topology::X_TORUS; }
};

// BH Galaxy Y Torus Node class
class BHGalaxyYTorusNode : public BHGalaxyNode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl(BHGalaxyTopology::Y_TORUS);
    }
    Topology get_topology() const override { return Topology::Y_TORUS; }
};

// BH Galaxy XY Torus Node class
class BHGalaxyXYTorusNode : public BHGalaxyNode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl(BHGalaxyTopology::XY_TORUS);
    }
    Topology get_topology() const override { return Topology::XY_TORUS; }
};

std::unique_ptr<NodeBase> create_node_instance(NodeType node_type) {
    switch (node_type) {
        case NodeType::N300_LB: return std::make_unique<N300LBNode>();
        case NodeType::N300_LB_DEFAULT: return std::make_unique<N300LBDefaultNode>();
        case NodeType::N300_QB: return std::make_unique<N300QBNode>();
        case NodeType::N300_QB_DEFAULT: return std::make_unique<N300QBDefaultNode>();
        case NodeType::WH_GALAXY: return std::make_unique<WHGalaxyNode>();
        case NodeType::WH_GALAXY_X_TORUS: return std::make_unique<WHGalaxyXTorusNode>();
        case NodeType::WH_GALAXY_Y_TORUS: return std::make_unique<WHGalaxyYTorusNode>();
        case NodeType::WH_GALAXY_XY_TORUS: return std::make_unique<WHGalaxyXYTorusNode>();
        case NodeType::P150_LB: return std::make_unique<P150LBNode>();
        case NodeType::P150_QB_AE: return std::make_unique<P150QBAENode>();
        case NodeType::P150_QB_AE_DEFAULT: return std::make_unique<P150QBAEDefaultNode>();
        case NodeType::P300_QB_GE: return std::make_unique<P300QBGENode>();
        case NodeType::BH_GALAXY: return std::make_unique<BHGalaxyNode>();
        case NodeType::BH_GALAXY_X_TORUS: return std::make_unique<BHGalaxyXTorusNode>();
        case NodeType::BH_GALAXY_Y_TORUS: return std::make_unique<BHGalaxyYTorusNode>();
        case NodeType::BH_GALAXY_XY_TORUS: return std::make_unique<BHGalaxyXYTorusNode>();
        default: throw std::runtime_error("Unknown node type: " + std::to_string(static_cast<int>(node_type)));
    }
}

// Factory function to create node descriptors by name
tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_node_descriptor(NodeType node_type) {
    auto node = create_node_instance(node_type);
    if (!node) {
        throw std::runtime_error("Unknown node type: " + std::string(enchantum::to_string(node_type)));
    }
    return node->create();
}

// Helper function to get topology for a NodeType (uses virtual function from node instances)
Topology get_node_type_topology(NodeType node_type) {
    auto node = create_node_instance(node_type);
    return node->get_topology();
}

bool is_torus(NodeType node_type) {
    Topology topology = get_node_type_topology(node_type);
    return topology == Topology::X_TORUS || topology == Topology::Y_TORUS || topology == Topology::XY_TORUS;
}

}  // namespace tt::scaleout_tools
