// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_kernels.hpp"
#include "impl/device/device_pool.hpp"
#include "tt_metal/detail/tt_metal.hpp"

#define DISPATCH_MAX_UPSTREAM 4
#define DISPATCH_MAX_DOWNSTREAM 4

typedef struct {
    int id;
    int device_id;
    uint8_t cq_id;
    DispatchWorkerType kernel_type;
    int upstream_ids[DISPATCH_MAX_UPSTREAM];
    int downstream_ids[DISPATCH_MAX_DOWNSTREAM];
    NOC my_noc;
    NOC upstream_noc;
    NOC downstream_noc;
} dispatch_kernel_node_t;

// For readablity, unset = x = -1
#define x -1

void increment_node_ids(dispatch_kernel_node_t &node, uint32_t inc) {
    node.id += inc;
    for (int &id : node.upstream_ids)
        if (id != x)
            id += inc;
    for (int &id : node.downstream_ids)
        if (id != x)
            id += inc;
}

static const std::vector<dispatch_kernel_node_t> single_card_arch_1cq = {
    {0, 0, 0, PREFETCH_HD,   { x,  x,  x,  x}, { 1,  2,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {1, 0, 0, DISPATCH_HD,   { 0,  x,  x,  x}, { 2,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {2, 0, 0, DISPATCH_S, { 0,  x,  x,  x}, { 1,  x,  x,  x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
};

static const std::vector<dispatch_kernel_node_t> single_card_arch_2cq = {
    {0, 0, 0, PREFETCH_HD, { x,  x,  x,  x}, { 1,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {1, 0, 0, DISPATCH_HD, { 0,  x,  x,  x}, { x,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {2, 0, 1, PREFETCH_HD, { x,  x,  x,  x}, { 3,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {3, 0, 1, DISPATCH_HD, { 2,  x,  x,  x}, { x,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
};

static const std::vector<dispatch_kernel_node_t> two_card_arch_1cq = {
    { 0, 0, 0, PREFETCH_HD,        { x,  x,  x,  x}, { 1,  2,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 1, 0, 0, DISPATCH_HD,        { 0,  x,  x,  x}, { 2,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    { 2, 0, 0, DISPATCH_S,         { 0,  x,  x,  x}, { 1,  x,  x,  x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},

    { 3, 0, 0, PREFETCH_H,         { x,  x,  x,  x}, { 5,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 4, 0, 0, DISPATCH_H,         { 6,  x,  x,  x}, { 3,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    { 5, 0, 0, PACKET_ROUTER_MUX,  { 3,  x,  x,  x}, { 7,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 6, 0, 0, DEMUX,              { 7,  x,  x,  x}, { 4,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 7, 0, 0, US_TUNNELER_REMOTE, { 8,  5,  x,  x}, { 8,  6,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 8, 1, 0, US_TUNNELER_LOCAL,  { 7,  9,  x,  x}, { 7, 10,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 9, 1, 0, MUX_D,              {12,  x,  x,  x}, { 8,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {10, 1, 0, PACKET_ROUTER_DEMUX,{ 8,  x,  x,  x}, {11,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {11, 1, 0, PREFETCH_D,         {10,  x,  x,  x}, {12, 13,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {12, 1, 0, DISPATCH_D,         {11,  x,  x,  x}, {13,  9,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {13, 1, 0, DISPATCH_S,         {11,  x,  x,  x}, {12,  x,  x,  x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
};

static const std::vector<dispatch_kernel_node_t> two_card_arch_2cq = {
    { 0, 0, 0, PREFETCH_HD,        { x,  x,  x,  x}, { 2,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 1, 0, 1, PREFETCH_HD,        { x,  x,  x,  x}, { 3,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 2, 0, 0, DISPATCH_HD,        { 0,  x,  x,  x}, { x,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    { 3, 0, 1, DISPATCH_HD,        { 1,  x,  x,  x}, { x,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},

    { 4, 0, 0, PREFETCH_H,         { x,  x,  x,  x}, { 8,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 5, 0, 1, PREFETCH_H,         { x,  x,  x,  x}, { 8,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 6, 0, 0, DISPATCH_H,         { 9,  x,  x,  x}, { 4,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    { 7, 0, 1, DISPATCH_H,         { 9,  x,  x,  x}, { 5,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},

    { 8, 0, 0, PACKET_ROUTER_MUX,  { 4,  5,  x,  x}, {10,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    { 9, 0, 0, DEMUX,              {10,  x,  x,  x}, { 6,  7,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {10, 0, 0, US_TUNNELER_REMOTE, {11,  8,  x,  x}, {11,  9,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {11, 1, 0, US_TUNNELER_LOCAL,  {10, 12,  x,  x}, {10, 13,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {12, 1, 0, MUX_D,              {16, 17,  x,  x}, {11,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {13, 1, 0, PACKET_ROUTER_DEMUX,{11,  x,  x,  x}, {14, 15,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    {14, 1, 0, PREFETCH_D,         {13,  x,  x,  x}, {16,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {15, 1, 1, PREFETCH_D,         {13,  x,  x,  x}, {17,  x,  x,  x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {16, 1, 0, DISPATCH_D,         {14,  x,  x,  x}, {12,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {17, 1, 1, DISPATCH_D,         {15,  x,  x,  x}, {12,  x,  x,  x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
};

std::vector<FDKernel *> node_id_to_kernel;

// Populate node_id_to_kernel and set up kernel objects. Do this once at the beginning since they (1) don't need a valid
// Device until fields are populated, (2) need to be connected to kernel objects for devices that aren't created yet,
// and (3) the table to choose depends on total number of devices, not know at Device creation.
void populate_fd_kernels(uint32_t num_devices, uint32_t num_hw_cqs) {
    tt::log_warning("FD Config: {} devices, {} HW CQs", num_devices, num_hw_cqs);
    // Select/generate the right input table. TODO: read this out of YAML instead of the structs above?
    std::vector<dispatch_kernel_node_t> nodes;
    if (num_devices == 0)
        num_devices = tt::Cluster::instance().number_of_user_devices();

    if (num_devices == 1) { // E150, N150
        nodes = (num_hw_cqs == 1) ? single_card_arch_1cq : single_card_arch_2cq;
    } else if (num_devices == 2) { // N300
        nodes = (num_hw_cqs == 1) ? two_card_arch_1cq : two_card_arch_2cq;
    } else if (num_devices == 8) { // T3K
        const std::vector<dispatch_kernel_node_t> *nodes_for_one_mmio = (num_hw_cqs == 1) ? &two_card_arch_1cq : &two_card_arch_2cq;
        // TODO: specify replication + device id mapping from struct/yaml? Just to avoid having these huge graphs typed out
        uint32_t num_mmio_devices = 4;
        uint32_t num_nodes_for_one_mmio = nodes_for_one_mmio->size();
        for (int mmio_device_id = 0; mmio_device_id < num_mmio_devices; mmio_device_id++) {
            for (dispatch_kernel_node_t node : *nodes_for_one_mmio) {
                TT_ASSERT(node.device_id == 0 || node.device_id == 1);
                if (node.device_id == 0)
                    node.device_id = mmio_device_id;
                else
                    node.device_id = mmio_device_id + num_mmio_devices;
                increment_node_ids(node, mmio_device_id * num_nodes_for_one_mmio);
                nodes.push_back(node);
            }
        }
    } else { // TG, TGG
        TT_FATAL(false, "Not yet implemented!");
    }
#if 0
    for (auto &node : nodes) {
        std::string upstream = "";
        for (int id : node.upstream_ids)
            upstream += fmt::format("{}, ", id);
        std::string downstream = "";
        for (int id : node.downstream_ids)
            downstream += fmt::format("{}, ", id);

        tt::log_info("[{}, {}, {}, {}, [{}], [{}], {}, {}, {}]", node.id, node.device_id, node.cq_id, node.kernel_type, upstream, downstream, node.my_noc, node.upstream_noc, node.downstream_noc);
    }
#endif

    // If we already had nodes from a previous run, clear them (since we could have a different # of devices or CQs).
    if (!node_id_to_kernel.empty()) {
        for (int idx = 0; idx < node_id_to_kernel.size(); idx++) {
            delete node_id_to_kernel[idx];
        }
        node_id_to_kernel.clear();
    }

    // Read the input table, create configs for each node
    for (const auto& node : nodes) {
        node_id_to_kernel.push_back(FDKernel::Generate(
            node.id, node.device_id, node.cq_id, {node.my_noc, node.upstream_noc, node.downstream_noc}, node.kernel_type));
    }

    // Connect the graph with upstream/downstream kernels
    for (const auto& node : nodes) {
        for (int idx = 0; idx < DISPATCH_MAX_UPSTREAM; idx++) {
            if (node.upstream_ids[idx] >= 0) {
                // tt::log_info("Node {} has upstream node: {}", node.id, node.upstream_ids[idx]);
                node_id_to_kernel.at(node.id)->AddUpstreamKernel(node_id_to_kernel.at(node.upstream_ids[idx]));
            }
        }
        for (int idx = 0; idx < DISPATCH_MAX_DOWNSTREAM; idx++) {
            if (node.downstream_ids[idx] >= 0) {
                // tt::log_info("Node {} has downstream node: {}", node.id, node.downstream_ids[idx]);
                node_id_to_kernel.at(node.id)->AddDownstreamKernel(node_id_to_kernel.at(node.downstream_ids[idx]));
            }
        }
    }
}

std::unique_ptr<Program> create_mmio_cq_program(Device *device) {
    TT_ASSERT(
        node_id_to_kernel.size() > 0,
        "Tried to create CQ program without nodes populated (need to run populate_fd_kernels()");

    // First pass, add device/program to all kernels for this device and generate static configs.
    auto cq_program_ptr = std::make_unique<Program>();
    // for (auto &node_and_kernel : node_id_to_kernel) {
    for (int idx = 0; idx < node_id_to_kernel.size(); idx++) {
        if (node_id_to_kernel[idx]->GetDeviceId() == device->id()) {
            node_id_to_kernel[idx]->AddDeviceAndProgram(device, cq_program_ptr.get());
            node_id_to_kernel[idx]->GenerateStaticConfigs();
            // tt::log_warning("Node {} has coord: {} (phys={})", idx, node_id_to_kernel[idx]->GetLogicalCore().str(), node_id_to_kernel[idx]->GetPhysicalCore().str());
        }
    }

    // Third pass, populate dependent configs and create kernels for each node
    // for (auto &node_and_kernel : node_id_to_kernel) {
    for (int idx = 0; idx < node_id_to_kernel.size(); idx++) {
        if (node_id_to_kernel[idx]->GetDeviceId() == device->id()) {
            node_id_to_kernel[idx]->GenerateDependentConfigs();
            node_id_to_kernel[idx]->CreateKernel();
        }
    }

    // Compile the program and return it so Device can register it
    detail::CompileProgram(device, *cq_program_ptr, /*fd_bootloader_mode=*/true);
    return cq_program_ptr;
}

void configure_dispatch_cores(Device *device) {
    // Set up completion_queue_writer core. This doesn't actually have a kernel so keep it out of the struct and config
    // it here. TODO: should this be in the struct?
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    auto &my_dispatch_constants = dispatch_constants::get(dispatch_core_type);
    uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    uint32_t cq_size = device->sysmem_manager().get_cq_size();
    std::vector<uint32_t> zero = {0x0};

    // Need to set up for all devices serviced by an mmio chip
    if (device->is_mmio_capable()) {
        for (chip_id_t serviced_device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(device->id())) {
            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(serviced_device_id);
            for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); cq_id++) {
                tt_cxy_pair completion_q_writer_location = dispatch_core_manager::instance().completion_queue_writer_core(serviced_device_id, channel, cq_id);
                Device *mmio_device = tt::DevicePool::instance().get_active_device(completion_q_writer_location.chip);
                uint32_t completion_q_wr_ptr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
                uint32_t completion_q_rd_ptr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
                uint32_t completion_q0_last_event_ptr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
                uint32_t completion_q1_last_event_ptr = my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
                // Initialize completion queue write pointer and read pointer copy
                uint32_t issue_queue_size = device->sysmem_manager().get_issue_queue_size(cq_id);
                uint32_t completion_queue_start_addr = cq_start + issue_queue_size + get_absolute_cq_offset(channel, cq_id, cq_size);
                uint32_t completion_queue_start_addr_16B = completion_queue_start_addr >> 4;
                std::vector<uint32_t> completion_queue_wr_ptr = {completion_queue_start_addr_16B};
                tt::log_warning(
                    "Configure CQ Writer (device {} core {})", mmio_device->id(), completion_q_writer_location.str());
                detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, completion_q_rd_ptr, completion_queue_wr_ptr, dispatch_core_type);
                detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, completion_q_wr_ptr, completion_queue_wr_ptr, dispatch_core_type);
                detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, completion_q0_last_event_ptr, zero, dispatch_core_type);
                detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, completion_q1_last_event_ptr, zero, dispatch_core_type);
            }
        }
    }
    // Configure cores for all nodes corresponding to this device
    for (int idx = 0; idx < node_id_to_kernel.size(); idx++) {
        if (node_id_to_kernel[idx]->GetDeviceId() == device->id()) {
            node_id_to_kernel[idx]->ConfigureCore();
        }
    }
}
