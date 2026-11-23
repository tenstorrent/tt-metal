// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "node.hpp"
#include "node_types.hpp"

#include <tt-umd/pcie/pci_device.hpp>
#include <tt-umd/tt_device/tt_device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <enchantum/enchantum.hpp>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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
    static tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create_impl(const std::string& motherboard) {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard(motherboard);

        // Add boards
        add_boards(&node, "N300", 1, 4);

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
        return create_impl("X12DPG-QT6");
    }
};

// N300 QB Node class
class N300QBNode : public N300T3KNode {
public:
    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        return create_impl("SIENAD8-2L2T");
    }
};

class WHGalaxyNode : public NodeBase {
public:
    Architecture get_architecture() const override { return Architecture::WORMHOLE; }

    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
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

        return node;
    }
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

    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
        tt::scaleout_tools::cabling_generator::proto::NodeDescriptor node;
        node.set_motherboard("SIENAD8-2L2T");

        // Add boards
        add_boards(&node, "P150", 1, 4);

        return node;
    }
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

class BHGalaxyRevABNode : public NodeBase {
public:
    Architecture get_architecture() const override { return Architecture::BLACKHOLE; }

    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
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

        return node;
    }
};

class BHGalaxyRevCNode : public NodeBase {
public:
    Architecture get_architecture() const override { return Architecture::BLACKHOLE; }

    tt::scaleout_tools::cabling_generator::proto::NodeDescriptor create() const override {
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

        return node;
    }
};

std::unique_ptr<NodeBase> create_node_instance(NodeType node_type) {
    switch (node_type) {
        case NodeType::N300_LB: return std::make_unique<N300LBNode>();
        case NodeType::N300_QB: return std::make_unique<N300QBNode>();
        case NodeType::WH_GALAXY: return std::make_unique<WHGalaxyNode>();
        case NodeType::P150_LB: return std::make_unique<P150LBNode>();
        case NodeType::P150_QB_AE: return std::make_unique<P150QBAENode>();
        case NodeType::P300_QB_GE: return std::make_unique<P300QBGENode>();

        case NodeType::BH_GALAXY:
            if (get_bh_glx_rev_from_system() == NodeType::BH_GALAXY_REV_C) {
                return std::make_unique<BHGalaxyRevCNode>();
            } else {
                return std::make_unique<BHGalaxyRevABNode>();
            }

        case NodeType::BH_GALAXY_REV_AB: return std::make_unique<BHGalaxyRevABNode>();
        case NodeType::BH_GALAXY_REV_C: return std::make_unique<BHGalaxyRevCNode>();
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

NodeType get_bh_glx_rev_from_system() {
    static std::optional<NodeType> cached;
    if (cached.has_value()) {
        return *cached;
    }
    uint64_t board_id;
    try {
        std::vector<int> pci_devices = tt::umd::PCIDevice::enumerate_devices();
        if (pci_devices.empty()) {
            log_warning(
                tt::LogDistributed,
                "PCIe device not accessible - cannot determine node type from system, defaulting to BH_GALAXY_REV_AB");
            cached = NodeType::BH_GALAXY_REV_AB;
            return *cached;
        }
        auto device = tt::umd::TTDevice::create(pci_devices[0]);
        device->init_tt_device();
        board_id = device->get_board_id();
    } catch (const std::exception& e) {
        log_warning(
            tt::LogDistributed,
            "Error accessing devices to determine BH Galaxy Rev ({}), defaulting to BH_GALAXY_REV_AB",
            e.what());
        cached = NodeType::BH_GALAXY_REV_AB;
        return *cached;
    }

    uint32_t revision_bits = (board_id >> 32) & 0xF;  // bits [35:32]
    if (revision_bits >= 3) {
        log_info(tt::LogDistributed, "BH Galaxy Rev C detected, using in place of BH_GALAXY");
        cached = NodeType::BH_GALAXY_REV_C;
    } else {
        log_info(tt::LogDistributed, "BH Galaxy Rev AB detected, using in place of BH_GALAXY");
        cached = NodeType::BH_GALAXY_REV_AB;
    }
    return *cached;
}

}  // namespace tt::scaleout_tools
