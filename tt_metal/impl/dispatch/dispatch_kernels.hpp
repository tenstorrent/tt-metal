// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "impl/device/device.hpp"
#include "impl/program/program.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"

typedef struct {
    NOC non_dispatch_noc; // For communicating with workers/DRAM/host
    NOC upstream_noc;     // For communicating with upstream dispatch modules
    NOC downstream_noc;   // For communicating with downstream dispatch modules
} noc_selection_t;

class FDKernel {
   public:
    FDKernel(int node_id, chip_id_t device_id, uint8_t cq_id, noc_selection_t noc_selection) :
        node_id(node_id), device_id(device_id), cq_id(cq_id), noc_selection(noc_selection){};
    virtual ~FDKernel() = default;
    virtual void CreateKernel() = 0;
    virtual void GenerateStaticConfigs() = 0;
    virtual void GenerateDependentConfigs() = 0;
    virtual void ConfigureCore() {}; // Overridden for specific kernels that need host-side configuration
    static FDKernel* Generate(
        int node_id, chip_id_t device_id, uint8_t cq_id, noc_selection_t noc_selection, DispatchWorkerType type);

    void AddUpstreamKernel(FDKernel *upstream) { this->upstream_kernels.push_back(upstream); }
    void AddDownstreamKernel(FDKernel *downstream) { this->downstream_kernels.push_back(downstream); }
    virtual CoreType GetCoreType() {
        return dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    }
    tt_cxy_pair GetLogicalCore() { return logical_core; }
    tt_cxy_pair GetPhysicalCore() {
        return tt_cxy_pair(logical_core.chip, tt::get_physical_core_coordinate(logical_core, GetCoreType()));
    }
    int GetUpstreamPort(FDKernel *other) { return GetPort(other, this->upstream_kernels); }
    int GetDownstreamPort(FDKernel *other) { return GetPort(other, this->downstream_kernels); }
    void AddDeviceAndProgram(Device *device, Program *program) {
        this->device = device;
        this->program = program;
    };
    chip_id_t GetDeviceId() { return this->device_id; } // Since this->device may not exist yet

   protected:
    void configure_kernel_variant(
        const string &path,
        const std::vector<uint32_t> &compile_args,
        std::map<string, string> defines_in,
        bool is_active_eth_core,
        bool send_to_brisc,
        bool force_watcher_no_inline);
    int GetPort(FDKernel *other, std::vector<FDKernel *> &kernels) {
        for (int idx = 0; idx < kernels.size(); idx++) {
            if (kernels[idx] == other)
                return idx;
        }
        TT_ASSERT(false);
        return -1;
    }

    Device *device = nullptr; // Set at configuration time by AddDeviceAndProgram()
    Program *program = nullptr;
    tt_cxy_pair logical_core;
    chip_id_t device_id;
    int node_id;
    uint8_t cq_id;
    noc_selection_t noc_selection;
    uint32_t tunnel_stop = 1; // TODO: populate this properly

    std::vector<FDKernel *> upstream_kernels;
    std::vector<FDKernel *> downstream_kernels;
};

// For each of the different dispatch kernels, a struct to hold their specific configs. Converted later on to defines +
// compile-time args.
typedef struct prefetch_config {
    std::optional<tt_cxy_pair> upstream_logical_core; // Dependant
    std::optional<tt_cxy_pair> downstream_logical_core; // Dependant
    std::optional<tt_cxy_pair> downstream_s_logical_core; // Dependant

    std::optional<uint32_t> downstream_cb_base; // Dependent
    std::optional<uint32_t> downstream_cb_log_page_size;
    std::optional<uint32_t> downstream_cb_pages;
    std::optional<uint32_t> my_downstream_cb_sem_id;
    std::optional<uint32_t> downstream_cb_sem_id; // Dependant

    std::optional<uint32_t> pcie_base;
    std::optional<uint32_t> pcie_size;
    std::optional<uint32_t> prefetch_q_base;
    std::optional<uint32_t> prefetch_q_size;
    std::optional<uint32_t> prefetch_q_rd_ptr_addr;
    std::optional<uint32_t> prefetch_q_pcie_rd_ptr_addr;

    std::optional<uint32_t> cmddat_q_base;
    std::optional<uint32_t> cmddat_q_size;

    // Used for prefetch_h
    std::optional<uint32_t> scratch_db_base;
    std::optional<uint32_t> scratch_db_size;
    std::optional<uint32_t> downstream_sync_sem_id;

    // Used for prefetch_d
    std::optional<uint32_t> cmddat_q_pages;
    std::optional<uint32_t> my_upstream_cb_sem_id;
    std::optional<uint32_t> upstream_cb_sem_id; // Dependant
    std::optional<uint32_t> cmddat_q_log_page_size;
    std::optional<uint32_t> cmddat_q_blocks;

    // Used for prefetch_d <--> dispatch_s data path
    std::optional<uint32_t> dispatch_s_buffer_base;
    std::optional<uint32_t> my_dispatch_s_cb_sem_id;
    std::optional<uint32_t> downstream_dispatch_s_cb_sem_id; // Dependant
    std::optional<uint32_t> dispatch_s_buffer_size;
    std::optional<uint32_t> dispatch_s_cb_log_page_size;

    std::optional<bool> is_d_variant;
    std::optional<bool> is_h_variant;
} prefetch_config_t;

typedef struct dispatch_config {
    std::optional<tt_cxy_pair> upstream_logical_core; // Dependant
    std::optional<tt_cxy_pair> downstream_logical_core; // Dependant
    std::optional<tt_cxy_pair> downstream_s_logical_core; // Dependant

    std::optional<uint32_t> dispatch_cb_base; // 0
    std::optional<uint32_t> dispatch_cb_log_page_size;
    std::optional<uint32_t> dispatch_cb_pages;
    std::optional<uint32_t> my_dispatch_cb_sem_id;
    std::optional<uint32_t> upstream_dispatch_cb_sem_id; // Dependant

    std::optional<uint32_t> dispatch_cb_blocks; // 5
    std::optional<uint32_t> upstream_sync_sem; // Dependant
    std::optional<uint32_t> command_queue_base_addr;
    std::optional<uint32_t> completion_queue_base_addr;
    std::optional<uint32_t> completion_queue_size;

    std::optional<uint32_t> downstream_cb_base; // 10, dependent
    std::optional<uint32_t> downstream_cb_size; // Dependent
    std::optional<uint32_t> my_downstream_cb_sem_id;
    std::optional<uint32_t> downstream_cb_sem_id; // Dependant

    std::optional<uint32_t> split_dispatch_page_preamble_size; // 14
    std::optional<uint32_t> split_prefetch;
    std::optional<uint32_t> prefetch_h_noc_xy; // Dependent
    std::optional<uint32_t> prefetch_h_local_downstream_sem_addr; // Dependent
    std::optional<uint32_t> prefetch_h_max_credits;

    std::optional<uint32_t> packed_write_max_unicast_sub_cmds; // 19
    std::optional<uint32_t> dispatch_s_sync_sem_base_addr;
    std::optional<uint32_t> max_num_worker_sems;
    std::optional<uint32_t> max_num_go_signal_noc_data_entries;
    std::optional<uint32_t> mcast_go_signal_addr;
    std::optional<uint32_t> unicast_go_signal_addr;
    std::optional<uint32_t> distributed_dispatcher;

    std::optional<uint32_t> host_completion_q_wr_ptr; // 26
    std::optional<uint32_t> dev_completion_q_wr_ptr;
    std::optional<uint32_t> dev_completion_q_rd_ptr;

    std::optional<bool> is_d_variant;
    std::optional<bool> is_h_variant;
} dispatch_config_t;

typedef struct dispatch_s_config {
    std::optional<tt_cxy_pair> upstream_logical_core; // Dependant
    std::optional<tt_cxy_pair> downstream_logical_core; // Dependant

    std::optional<uint32_t> cb_base;
    std::optional<uint32_t> cb_log_page_size;
    std::optional<uint32_t> cb_size;
    std::optional<uint32_t> my_dispatch_cb_sem_id;
    std::optional<uint32_t> upstream_dispatch_cb_sem_id; // Dependent
    std::optional<uint32_t> dispatch_s_sync_sem_base_addr;

    std::optional<uint32_t> mcast_go_signal_addr;
    std::optional<uint32_t> unicast_go_signal_addr;
    std::optional<uint32_t> distributed_dispatcher;
    std::optional<uint32_t> worker_sem_base_addr;
    std::optional<uint32_t> max_num_worker_sems;
    std::optional<uint32_t> max_num_go_signal_noc_data_entries;
} dispatch_s_config_t;

typedef struct eth_tunneler_config {
    std::optional<uint32_t> endpoint_id_start_index;
    std::optional<uint32_t> vc_count; // Dependent
    std::optional<uint32_t> in_queue_start_addr_words;
    std::optional<uint32_t> in_queue_size_words;

    std::array<std::optional<uint32_t>, MAX_TUNNEL_LANES> remote_receiver_x; // [4:13], dependent
    std::array<std::optional<uint32_t>, MAX_TUNNEL_LANES> remote_receiver_y; // [4:13], dependent
    std::array<std::optional<uint32_t>, MAX_TUNNEL_LANES> remote_receiver_queue_id; // [4:13], dependent
    std::array<std::optional<uint32_t>, MAX_TUNNEL_LANES> remote_receiver_network_type; // [4:13], dependent
    std::array<std::optional<uint32_t>, MAX_TUNNEL_LANES> remote_receiver_queue_start; // [14:2:32], dependent
    std::array<std::optional<uint32_t>, MAX_TUNNEL_LANES> remote_receiver_queue_size; // [15:2:33], dependent
    std::array<std::optional<uint32_t>, MAX_TUNNEL_LANES> remote_sender_x; // [34:43], dependent
    std::array<std::optional<uint32_t>, MAX_TUNNEL_LANES> remote_sender_y; // [34:43], dependent
    std::array<std::optional<uint32_t>, MAX_TUNNEL_LANES> remote_sender_queue_id; // [34:43], dependent
    std::array<std::optional<uint32_t>, MAX_TUNNEL_LANES> remote_sender_network_type; // [34:43], dependent

    std::optional<uint32_t> kernel_status_buf_addr_arg;
    std::optional<uint32_t> kernel_status_buf_size_bytes;
    std::optional<uint32_t> timeout_cycles;
    std::optional<uint32_t> inner_stop_mux_d_bypass; // Dependent
} eth_tunneler_config_t;

typedef struct eth_router_config {
    std::optional<uint32_t> vc_count;
    std::optional<uint32_t> rx_queue_start_addr_words; // 1
    std::optional<uint32_t> rx_queue_size_words;
    std::optional<uint32_t> router_lanes;

    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_x; // [4:7], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_y; // [4:7], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_queue_id; // [4:7], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_network_type; // [4:7], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_queue_start_addr_words; // [8:2:14], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_queue_size_words; // [9:2:15], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> remote_rx_x; // [16:19], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> remote_rx_y; // [16:19], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> remote_rx_queue_id; // [16:19]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> remote_rx_network_type; // [17:19], dependent

    std::optional<uint32_t> kernel_status_buf_addr_arg; // 22
    std::optional<uint32_t> kernel_status_buf_size_bytes;
    std::optional<uint32_t> timeout_cycles;

    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> output_depacketize; // 25, dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> output_depacketize_log_page_size; // [26:29]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> output_depacketize_downstream_sem; // [26:29], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> output_depacketize_local_sem; // [26:29]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> output_depacketize_remove_header; // [26:29]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN>  input_packetize; // [30:33]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN>  input_packetize_log_page_size; // [30:33]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN>  input_packetize_upstream_sem; // [30:33], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN>  input_packetize_local_sem; // [30:33]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN>  input_packetize_src_endpoint; // Dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN>  input_packetize_dst_endpoint; // Dependent
} eth_router_config_t;

typedef struct mux_config {
    std::optional<uint32_t> vc_count;
    std::optional<uint32_t> reserved;
    std::optional<uint32_t> rx_queue_start_addr_words;
    std::optional<uint32_t> rx_queue_size_words;
    std::optional<uint32_t> mux_fan_in; // Dependent

    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> remote_rx_x; // [4:7], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> remote_rx_y; // [4:7], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> remote_rx_queue_id; // [4:7]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> remote_rx_network_type; // [4:7]
    std::optional<uint32_t> remote_tx_queue_start_addr_words; // Dependent
    std::optional<uint32_t> remote_tx_queue_size_words; // Dependent
    std::optional<uint32_t> remote_tx_x; // Dependent
    std::optional<uint32_t> remote_tx_y; // Dependent
    std::optional<uint32_t> remote_tx_queue_id; // Dependent
    std::optional<uint32_t> tx_network_type;

    std::optional<uint32_t> test_results_buf_addr_arg;
    std::optional<uint32_t> test_results_buf_size_bytes;
    std::optional<uint32_t> timeout_cycles;
    std::optional<uint32_t> output_depacketize;
    std::optional<uint32_t> output_depacketize_info; // Packed, pack with above same is input?
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> input_packetize;
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> input_packetize_log_page_size; // Dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> input_packetize_upstream_sem; // Dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_IN> input_packetize_local_sem;
    std::optional<uint32_t> input_packetize_src_endpoint; // Packed w/ max 4 assumption
    std::optional<uint32_t> input_packetize_dest_endpoint; // Same as src
} mux_config_t;

typedef struct demux_config {
    std::optional<uint32_t> vc_count;
    std::optional<uint32_t> endpoint_id_start_index;
    std::optional<uint32_t> rx_queue_start_addr_words;
    std::optional<uint32_t> rx_queue_size_words;
    std::optional<uint32_t> demux_fan_out; // Dependent

    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_x; // [4:7], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_y; // [4:7], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_queue_id; // [4:7]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_network_type; // [4:7]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_queue_start_addr_words; // [8:2:14], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> remote_tx_queue_size_words; // [9:2:15], dependent
    std::optional<uint32_t> remote_rx_x; // Dependent
    std::optional<uint32_t> remote_rx_y; // Dependent
    std::optional<uint32_t> remote_rx_queue_id; // Dependent
    std::optional<uint32_t> remote_rx_network_type;

    std::optional<uint32_t> dest_endpoint_output_map_hi; // Dependent
    std::optional<uint32_t> dest_endpoint_output_map_lo; // Dependent
    std::optional<uint32_t> test_results_buf_addr_arg;
    std::optional<uint32_t> test_results_buf_size_bytes;
    std::optional<uint32_t> timeout_cycles;
    std::optional<uint32_t> output_depacketize; // Dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> output_depacketize_cb_log_page_size; // [26:29]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> output_depacketize_downstream_sem_id; // [26:29], dependent
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> output_depacketize_local_sem_id; // [26:29]
    std::array<std::optional<uint32_t>, MAX_SWITCH_FAN_OUT> output_depacketize_remove_header; // [26:29]
} demux_config_t;

class PrefetchKernel : public FDKernel {
   public:
    PrefetchKernel(int node_id, chip_id_t device_id, uint8_t cq_id, noc_selection_t noc_selection, bool h_variant, bool d_variant) :
        FDKernel(node_id, device_id, cq_id, noc_selection) {
        config.is_h_variant = h_variant;
        config.is_d_variant = d_variant;
    }
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;
    const prefetch_config_t &GetConfig() { return this->config; }

   private:
    prefetch_config_t config;
};

class DispatchKernel : public FDKernel {
   public:
    DispatchKernel(int node_id, chip_id_t device_id, uint8_t cq_id, noc_selection_t noc_selection, bool h_variant, bool d_variant) :
        FDKernel(node_id, device_id, cq_id, noc_selection) {
        config.is_h_variant = h_variant;
        config.is_d_variant = d_variant;
    }
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;
    const dispatch_config_t &GetConfig() { return this->config; }

   private:
    dispatch_config_t config;
};

class DispatchSKernel : public FDKernel {
   public:
    DispatchSKernel(int node_id, chip_id_t device_id, uint8_t cq_id, noc_selection_t noc_selection) : FDKernel(node_id, device_id, cq_id, noc_selection) {}
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;
    const dispatch_s_config_t &GetConfig() { return this->config; }

   private:
    dispatch_s_config_t config;
};

class MuxKernel : public FDKernel {
   public:
    MuxKernel(int node_id, chip_id_t device_id, uint8_t cq_id, noc_selection_t noc_selection) : FDKernel(node_id, device_id, cq_id, noc_selection) {}
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    const mux_config_t &GetConfig() { return this->config; }

   private:
    mux_config_t config;
};

class DemuxKernel : public FDKernel {
   public:
    DemuxKernel(int node_id, chip_id_t device_id, uint8_t cq_id, noc_selection_t noc_selection) : FDKernel(node_id, device_id, cq_id, noc_selection) {}
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    const demux_config_t &GetConfig() { return this->config; }

   private:
    demux_config_t config;
};

class EthTunnelerKernel : public FDKernel {
   public:
    EthTunnelerKernel(int node_id, chip_id_t device_id, uint8_t cq_id, noc_selection_t noc_selection, bool is_remote) :
        FDKernel(node_id, device_id, cq_id, noc_selection), is_remote(is_remote) {}
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    CoreType GetCoreType() override {
        // Tunneler kernel is the exception in that it's always on ethernet core even if dispatch is on tensix.
        return CoreType::ETH;
    }
    tt_cxy_pair GetPhysicalCore() {
        return tt_cxy_pair(logical_core.chip, tt::get_physical_core_coordinate(logical_core, GetCoreType()));
    }
    const eth_tunneler_config_t &GetConfig() { return this->config; }
    bool IsRemote() { return this->is_remote; }

   private:
    eth_tunneler_config_t config;
    uint32_t tunnel_stop;
    std::optional<chip_id_t> downstream_device_id; // TODO
    bool is_remote;
    bool is_tunnel_start = true;
    bool is_tunnel_end = true;
};

class EthRouterKernel : public FDKernel {
   public:
    EthRouterKernel(int node_id, chip_id_t device_id, uint8_t cq_id, noc_selection_t noc_selection, bool as_mux) :
        FDKernel(node_id, device_id, cq_id, noc_selection), as_mux(as_mux) {}
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    const eth_router_config_t &GetConfig() { return this->config; }

   private:
    eth_router_config_t config;
    bool as_mux;

};
