// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "node.hpp"
#include "node_types.hpp"

#include <enchantum/enchantum.hpp>
#include <stdexcept>
#include <tt_stl/caseless_comparison.hpp>

namespace tt::scaleout_tools {

// N300 Node class
class N300T3KNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create(const std::string& motherboard) {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;

        node.set_motherboard(motherboard);

        // Add boards
        auto* boards = node.mutable_boards();
        for (int i = 1; i <= 4; ++i) {
            auto* board = boards->add_board();
            board->set_tray_id(i);
            board->set_board_type("N300");
        }

        // Add QSFP connections
        auto* qsfp_connections = &(*node.mutable_port_type_connections())["QSFP"];
        {
            auto* connection = qsfp_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(1);
        }
        {
            auto* connection = qsfp_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(2);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(3);
            connection->mutable_port_b()->set_port_id(2);
        }

        // Add WARP100 connections
        auto* warp100_connections = &(*node.mutable_port_type_connections())["WARP100"];
        {
            auto* connection = warp100_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(2);
            connection->mutable_port_b()->set_port_id(1);
        }
        {
            auto* connection = warp100_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(2);
            connection->mutable_port_b()->set_port_id(2);
        }
        {
            auto* connection = warp100_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(3);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(1);
        }
        {
            auto* connection = warp100_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(3);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(2);
        }

        return node;
    }
};

// N300 LB Node class
class N300LBNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        return N300T3KNode::create("X12DPG-QT6");
    }
};

// N300 QB Node class
class N300QBNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        return N300T3KNode::create("SIENAD8-2L2T");
    }
};

// P150 QB Global Node class
class P150QBGlobalNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;

        node.set_motherboard("SIENAD8-2L2T");

        // Add boards
        auto* boards = node.mutable_boards();
        for (int i = 1; i <= 4; ++i) {
            auto* board = boards->add_board();
            board->set_tray_id(i);
            board->set_board_type("P150");
        }

        // Add QSFP connections
        auto* qsfp_connections = &(*node.mutable_port_type_connections())["QSFP"];
        // Connection 1: tray 1 port 1 <-> tray 2 port 1
        {
            auto* connection = qsfp_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(2);
            connection->mutable_port_b()->set_port_id(1);
        }
        // Connection 2: tray 1 port 2 <-> tray 2 port 2
        {
            auto* connection = qsfp_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(2);
            connection->mutable_port_b()->set_port_id(2);
        }
        // Connection 3: tray 1 port 3 <-> tray 4 port 3
        {
            auto* connection = qsfp_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(3);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(3);
        }
        // Connection 4: tray 1 port 4 <-> tray 4 port 4
        {
            auto* connection = qsfp_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(4);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(4);
        }
        // Connection 5: tray 2 port 3 <-> tray 3 port 3
        {
            auto* connection = qsfp_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(2);
            connection->mutable_port_a()->set_port_id(3);
            connection->mutable_port_b()->set_tray_id(3);
            connection->mutable_port_b()->set_port_id(3);
        }
        // Connection 6: tray 2 port 4 <-> tray 3 port 4
        {
            auto* connection = qsfp_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(2);
            connection->mutable_port_a()->set_port_id(4);
            connection->mutable_port_b()->set_tray_id(3);
            connection->mutable_port_b()->set_port_id(4);
        }
        // Connection 7: tray 3 port 1 <-> tray 4 port 1
        {
            auto* connection = qsfp_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(3);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(1);
        }
        // Connection 8: tray 3 port 2 <-> tray 4 port 2
        {
            auto* connection = qsfp_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(3);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(2);
        }

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
        auto* boards = node.mutable_boards();
        for (int i = 1; i <= 2; ++i) {
            auto* board = boards->add_board();
            board->set_tray_id(i);
            board->set_board_type("P300");
        }

        // Add WARP400 connections
        auto* warp400_connections = &(*node.mutable_port_type_connections())["WARP400"];
        // Connection 1: tray 1 port 1 <-> tray 2 port 1
        {
            auto* connection = warp400_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(2);
            connection->mutable_port_b()->set_port_id(1);
        }
        // Connection 2: tray 1 port 2 <-> tray 2 port 2
        {
            auto* connection = warp400_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(2);
            connection->mutable_port_b()->set_port_id(2);
        }

        return node;
    }
};

class WHGalaxyNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create(
        bool add_x_torus = false, bool add_y_torus = false) {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;

        node.set_motherboard("S7T-MB");

        // Add boards
        auto* boards = node.mutable_boards();
        for (int i = 1; i <= 4; ++i) {
            auto* board = boards->add_board();
            board->set_tray_id(i);
            board->set_board_type("UBB");
        }

        // Add LINKING_BOARD_1 connections
        auto* lb1_connections = &(*node.mutable_port_type_connections())["LINKING_BOARD_1"];
        {
            auto* connection = lb1_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(2);
            connection->mutable_port_b()->set_port_id(1);
        }
        {
            auto* connection = lb1_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(2);
            connection->mutable_port_b()->set_port_id(2);
        }
        {
            auto* connection = lb1_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(3);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(1);
        }
        {
            auto* connection = lb1_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(3);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(2);
        }

        // Add LINKING_BOARD_2 connections
        auto* lb2_connections = &(*node.mutable_port_type_connections())["LINKING_BOARD_2"];
        {
            auto* connection = lb2_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(2);
            connection->mutable_port_b()->set_port_id(1);
        }
        {
            auto* connection = lb2_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(2);
            connection->mutable_port_b()->set_port_id(2);
        }
        {
            auto* connection = lb2_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(3);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(1);
        }
        {
            auto* connection = lb2_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(3);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(2);
        }

        // Add LINKING_BOARD_3 connections
        auto* lb3_connections = &(*node.mutable_port_type_connections())["LINKING_BOARD_3"];
        {
            auto* connection = lb3_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(3);
            connection->mutable_port_b()->set_port_id(1);
        }
        {
            auto* connection = lb3_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(1);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(3);
            connection->mutable_port_b()->set_port_id(2);
        }
        {
            auto* connection = lb3_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(2);
            connection->mutable_port_a()->set_port_id(1);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(1);
        }
        {
            auto* connection = lb3_connections->add_connections();
            connection->mutable_port_a()->set_tray_id(2);
            connection->mutable_port_a()->set_port_id(2);
            connection->mutable_port_b()->set_tray_id(4);
            connection->mutable_port_b()->set_port_id(2);
        }
        // Add QSFP connections
        auto* qsfp_connections = &(*node.mutable_port_type_connections())["QSFP"];

        if (add_x_torus) {
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(1);
                connection->mutable_port_a()->set_port_id(3);
                connection->mutable_port_b()->set_tray_id(2);
                connection->mutable_port_b()->set_port_id(3);
            }
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(1);
                connection->mutable_port_a()->set_port_id(4);
                connection->mutable_port_b()->set_tray_id(2);
                connection->mutable_port_b()->set_port_id(4);
            }
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(1);
                connection->mutable_port_a()->set_port_id(5);
                connection->mutable_port_b()->set_tray_id(2);
                connection->mutable_port_b()->set_port_id(5);
            }
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(1);
                connection->mutable_port_a()->set_port_id(6);
                connection->mutable_port_b()->set_tray_id(2);
                connection->mutable_port_b()->set_port_id(6);
            }
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(3);
                connection->mutable_port_a()->set_port_id(6);
                connection->mutable_port_b()->set_tray_id(4);
                connection->mutable_port_b()->set_port_id(6);
            }
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(3);
                connection->mutable_port_a()->set_port_id(5);
                connection->mutable_port_b()->set_tray_id(4);
                connection->mutable_port_b()->set_port_id(5);
            }
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(3);
                connection->mutable_port_a()->set_port_id(4);
                connection->mutable_port_b()->set_tray_id(4);
                connection->mutable_port_b()->set_port_id(4);
            }
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(3);
                connection->mutable_port_a()->set_port_id(3);
                connection->mutable_port_b()->set_tray_id(4);
                connection->mutable_port_b()->set_port_id(3);
            }
        }
        if (add_y_torus) {
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(1);
                connection->mutable_port_a()->set_port_id(2);
                connection->mutable_port_b()->set_tray_id(3);
                connection->mutable_port_b()->set_port_id(2);
            }
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(1);
                connection->mutable_port_a()->set_port_id(1);
                connection->mutable_port_b()->set_tray_id(3);
                connection->mutable_port_b()->set_port_id(1);
            }
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(2);
                connection->mutable_port_a()->set_port_id(1);
                connection->mutable_port_b()->set_tray_id(4);
                connection->mutable_port_b()->set_port_id(1);
            }
            {
                auto* connection = qsfp_connections->add_connections();
                connection->mutable_port_a()->set_tray_id(2);
                connection->mutable_port_a()->set_port_id(2);
                connection->mutable_port_b()->set_tray_id(4);
                connection->mutable_port_b()->set_port_id(2);
            }
        }

        return node;
    }
};

// WH Galaxy X Torus Node class
class WHGalaxyXTorusNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        auto node = WHGalaxyNode::create(true, false);
        return node;
    }
};

// WH Galaxy Y Torus Node class
class WHGalaxyYTorusNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        auto node = WHGalaxyNode::create(false, true);
        return node;
    }
};

// WH Galaxy XY Torus Node class
class WHGalaxyXYTorusNode {
public:
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() {
        auto node = WHGalaxyNode::create(true, true);
        return node;
    }
};

// Factory function to create node descriptors by name
tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_node_descriptor(NodeType node_type) {
    switch (node_type) {
        case NodeType::N300_LB: return N300LBNode::create();
        case NodeType::N300_QB: return N300QBNode::create();
        case NodeType::P150_QB_GLOBAL: return P150QBGlobalNode::create();
        case NodeType::P300_QB_AMERICA: return P300QBAmericaNode::create();
        case NodeType::WH_GALAXY: return WHGalaxyNode::create();
        case NodeType::WH_GALAXY_X_TORUS: return WHGalaxyXTorusNode::create();
        case NodeType::WH_GALAXY_Y_TORUS: return WHGalaxyYTorusNode::create();
        case NodeType::WH_GALAXY_XY_TORUS: return WHGalaxyXYTorusNode::create();
    }
    throw std::runtime_error("Unknown node type: " + std::string(enchantum::to_string(node_type)));
}

}  // namespace tt::scaleout_tools
