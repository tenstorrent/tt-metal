
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/test_common.hpp>

#include "ttnn/cpp/ttnn/operations/ccl/erisc_datamover_builder.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"

#include "tt_cluster.hpp"
#include "llrt.hpp"

#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

#include <memory>

using namespace tt;
using chan_id_t = std::uint8_t;

// global random number generator
std::mt19937 global_rng;

// time based seed
uint32_t time_seed;

const std::string rx_kernel_src = "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp";
const std::string tx_kernel_src = "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp";

// forward declarations
struct TestDevice;

struct TestBoard {
    std::map<chip_id_t, IDevice*> IDevice_handle_map;
    std::unordered_map<chip_id_t, std::shared_ptr<TestDevice>> test_device_map;

    void init(std::string& board_type_) {
        auto num_available_devices = tt_metal::GetNumAvailableDevices();
        uint32_t num_expected_devices;
        uint32_t device_id_offset = 0;
        std::vector<chip_id_t> device_ids;

        if ("n300" == board_type_) {
            num_expected_devices = 2;
        } else if ("t3k" == board_type_) {
            num_expected_devices = 8;
        } else if ("tg" == board_type_) {
            num_expected_devices = 32;
            device_id_offset = 4;
        } else {
            throw std::runtime_error("Unsupported board type");
        }

        if (num_available_devices != num_expected_devices) {
            log_fatal(
                LogTest,
                "Expected {} devices for {}, got: {}",
                num_expected_devices,
                board_type_,
                num_available_devices);
            throw std::runtime_error("Unexpected number of devices");
        }

        for (auto i = device_id_offset; i < num_available_devices + device_id_offset; i++) {
            device_ids.push_back(i);
        }

        IDevice_handle_map = tt::tt_metal::detail::CreateDevices(device_ids);
    }

    IDevice* get_IDevice_handle(chip_id_t chip_id) { return IDevice_handle_map[chip_id]; }

    TestDevice* init_test_device(chip_id_t chip_id) {
        auto test_device = std::make_shared<TestDevice>(chip_id);
        test_device_map[chip_id] = test_device;
        return test_device.get();
    }

    TestDevice* get_test_device(chip_id_t chip_id) {
        auto result = test_device_map.find(chip_id);
        // instantiate if not found
        if (result == test_device_map.end()) {
            return init_test_device(chip_id);
        }
        return result->second.get();
    }

    inline void close_devices() { tt::tt_metal::detail::CloseDevices(IDevice_handle_map); }

} test_board;

struct TestDevice {
    chip_id_t chip_id;
    tt_metal::IDevice* IDevice_handle;
    tt_metal::Program program_handle;
    std::vector<CoreCoord> worker_logical_cores;
    std::unordered_map<chan_id_t, CoreCoord> active_eth_chan_to_logical_core;
    std::unordered_map<chan_id_t, ttnn::ccl::FabricEriscDatamoverBuilder> edm_workers;

    TestDevice(chip_id_t chip_id_) {
        chip_id = chip_id_;
        IDevice_handle = test_board.get_IDevice_handle(chip_id);
        program_handle = tt_metal::CreateProgram();

        chan_id_t eth_chan;
        for (const auto& router_logical_core : IDevice_handle->get_active_ethernet_cores(true)) {
            eth_chan = router_logical_core.y;
            active_eth_chan_to_logical_core[eth_chan] = router_logical_core;
        }

        // initalize list of worker cores in 8X8 grid
        // TODO: remove hard-coding
        for (auto i = 0; i < 8; i++) {
            for (auto j = 0; j < 8; j++) {
                worker_logical_cores.push_back(CoreCoord({i, j}));
            }
        }
    }

    void init_EDM_worker(
        chan_id_t eth_chan,
        std::optional<std::pair<chip_id_t, chan_id_t>> backward_connection,
        std::optional<std::pair<chip_id_t, chan_id_t>> forward_connection) {
        // throw exception if EDM worker is already initialized
        if (edm_workers.find(eth_chan) != edm_workers.end()) {
            log_fatal(LogTest, "Attempt to re-init EDM worker on device id: {}, eth chan: {}", chip_id, eth_chan);
            throw std::runtime_error("Cannot initalize EDM worker more than once");
        }

        static constexpr std::size_t edm_buffer_size =
            ttnn::ccl::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes +
            sizeof(tt::fabric::PacketHeader);
        const auto edm_config = ttnn::ccl::FabricEriscDatamoverConfig(edm_buffer_size, 1, 2);

        chip_id_t remote_chip_id;
        // only one of the backward or forward connection will be to a remote chip
        if (backward_connection != std::nullopt && backward_connection.value().first != chip_id) {
            remote_chip_id = backward_connection.value().first;
        } else if (forward_connection != std::nullopt && forward_connection.value().first != chip_id) {
            remote_chip_id = forward_connection.value().first;
        } else {
            throw std::runtime_error("Cannot find remote chip for EDM worker");
        }

        if (active_eth_chan_to_logical_core.find(eth_chan) == active_eth_chan_to_logical_core.end()) {
            log_fatal(
                LogTest,
                "Attempt to init EDM worker inactive eth chan on device id: {}, eth chan: {}",
                chip_id,
                eth_chan);
            throw std::runtime_error("Cannot initalize EDM worker on inactive eth chan");
        }

        std::cout << "my chip: " << (uint32_t)chip_id << ", remote chip: " << (uint32_t)remote_chip_id << std::endl;

        auto edm_builder = ttnn::ccl::FabricEriscDatamoverBuilder::build(
            IDevice_handle,
            program_handle,
            active_eth_chan_to_logical_core[eth_chan],
            chip_id,
            remote_chip_id,
            edm_config,
            true,
            false);

        edm_workers.insert({eth_chan, edm_builder});
    }

    void connect_EDM_workers(chan_id_t eth_chan_1, chan_id_t eth_chan_2) {
        if (edm_workers.find(eth_chan_1) == edm_workers.end() || edm_workers.find(eth_chan_2) == edm_workers.end()) {
            throw std::runtime_error("Attempted to connect uninitialized EDMs");
        }

        auto& edm_worker_1 = edm_workers.at(eth_chan_1);
        auto& edm_worker_2 = edm_workers.at(eth_chan_2);

        edm_worker_1.connect_to_downstream_edm(edm_worker_2);
        edm_worker_2.connect_to_downstream_edm(edm_worker_1);
    }

    void build_EDM_kernels() {
        for (auto& [eth_chan, edm_builder] : edm_workers) {
            ttnn::ccl::generate_edm_kernel(
                program_handle, IDevice_handle, edm_builder, edm_builder.my_eth_core_logical, NOC::NOC_0);
        }
    }

    std::vector<CoreCoord> select_random_worker_cores(uint32_t count) {
        std::vector<CoreCoord> result;

        // shuffle the list of cores
        std::shuffle(worker_logical_cores.begin(), worker_logical_cores.end(), global_rng);

        // return and delete the selected cores
        for (auto i = 0; i < count; i++) {
            result.push_back(worker_logical_cores.back());
            worker_logical_cores.pop_back();
        }

        return result;
    }

    std::vector<uint32_t> generate_edm_connection_rt_args(chan_id_t eth_chan, CoreRangeSet worker_cores) {
        if (edm_workers.find(eth_chan) == edm_workers.end()) {
            log_fatal(LogTest, "EDM worker not initialized for device id: {}, eth chan: {}", chip_id, eth_chan);
            throw std::runtime_error("Attempted to connect worker to uninitialized EDM");
        }

        auto edm_connection_info = edm_workers.at(eth_chan).build_connection_to_worker_channel();
        return ttnn::ccl::worker_detail::generate_edm_connection_rt_args(
            edm_connection_info, program_handle, worker_cores);
    }

    inline uint32_t get_noc_encoding(CoreCoord& logical_core) {
        CoreCoord virt_core = IDevice_handle->worker_core_from_logical_core(logical_core);
        return tt_metal::hal.noc_xy_encoding(virt_core.x, virt_core.y);
    }

    void launch_program() { tt_metal::detail::LaunchProgram(IDevice_handle, program_handle, false); }

    void wait_for_EDM_handshake() {
        uint32_t expected_val = 0xA0B1C2D3;
        for (auto& [eth_chan, edm_builder] : edm_workers) {
            uint32_t edm_status = 0;
            uint32_t edm_status_address = edm_builder.termination_signal_ptr + 4;
            CoreCoord virtual_eth_core = tt::Cluster::instance().get_virtual_eth_core_from_channel(chip_id, eth_chan);
            while (edm_status != expected_val) {
                edm_status = tt::llrt::read_hex_vec_from_core(chip_id, virtual_eth_core, edm_status_address, 4)[0];
            }
        }
    }

    void terminate_EDM_kernels() {
        tt::fabric::TerminationSignal termination_mode = tt::fabric::TerminationSignal::GRACEFULLY_TERMINATE;
        for (auto& [eth_chan, edm_builder] : edm_workers) {
            edm_builder.teardown_from_host(IDevice_handle, termination_mode);
        }
    }

    void wait_for_program_done() { tt_metal::detail::WaitProgramDone(IDevice_handle, program_handle); }
};

struct TestFabricTraffic {
    chan_id_t tx_eth_chan;
    uint32_t num_hops;
    TestDevice* tx_device;
    std::vector<TestDevice*> rx_devices;
    CoreCoord tx_logical_core;
    CoreCoord tx_virtual_core;
    std::vector<CoreCoord> rx_logical_cores;
    std::vector<CoreCoord> rx_virtual_cores;
    std::vector<uint32_t> tx_results;
    std::vector<std::vector<uint32_t>> rx_results;

    TestFabricTraffic(
        chan_id_t tx_eth_chan_, uint32_t num_hops_, TestDevice* tx_device_, std::vector<TestDevice*> rx_devices_) {
        tx_eth_chan = tx_eth_chan_;
        num_hops = num_hops_;
        tx_device = tx_device_;
        rx_devices = rx_devices_;

        // TODO: select the optimal tx/rx worker based on the proximity from the ethernet core
        tx_logical_core = tx_device->select_random_worker_cores(1)[0];
        tx_virtual_core = tx_device->IDevice_handle->worker_core_from_logical_core(tx_logical_core);

        // TODO: for mcast, choose the same rx core
        for (auto& rx_device : rx_devices) {
            CoreCoord rx_logical_core = rx_device->select_random_worker_cores(1)[0];
            rx_logical_cores.push_back(rx_logical_core);
            rx_virtual_cores.push_back(rx_device->IDevice_handle->worker_core_from_logical_core(rx_logical_core));
        }
    }

    void build_worker_kernels() {
        // TODO get these propagated from the command line args
        uint32_t packet_header_address = 0x25000;
        uint32_t source_l1_buffer_address = 0x30000;
        uint32_t packet_payload_size_bytes = 4096;
        uint32_t num_packets = 5;
        uint32_t test_results_address = 0x100000;
        uint32_t test_results_size_bytes = 128;
        uint32_t target_address = 0x30000;
        uint32_t notfication_address = 0x24000;

        std::map<string, string> defines;
        std::vector<uint32_t> zero_buf(1, 0);

        // build sender kernel
        std::vector<uint32_t> compile_args = {test_results_address, test_results_size_bytes, target_address};

        std::vector<uint32_t> tx_runtime_args = {
            packet_header_address,
            source_l1_buffer_address,
            packet_payload_size_bytes,
            num_packets,
            num_hops,
            rx_devices[0]->get_noc_encoding(rx_logical_cores[0]),
            time_seed,
        };

        auto edm_rt_args = tx_device->generate_edm_connection_rt_args(tx_eth_chan, {tx_logical_core});
        for (auto& arg : edm_rt_args) {
            tx_runtime_args.push_back(arg);
        }

        // zero out host notification address
        tt::llrt::write_hex_vec_to_core(tx_device->chip_id, tx_virtual_core, zero_buf, notfication_address);

        auto tx_kernel = tt_metal::CreateKernel(
            tx_device->program_handle,
            tx_kernel_src,
            {tx_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = compile_args,
                .defines = defines});

        tt_metal::SetRuntimeArgs(tx_device->program_handle, tx_kernel, tx_logical_core, tx_runtime_args);

        std::cout << "num hops: " << num_hops << std::endl;

        log_info(
            LogTest,
            "[Device: Phys: {}] TX running on: logical: x={},y={}, virtual: x{},y={}, Eth chan: {}",
            tx_device->chip_id,
            tx_logical_core.x,
            tx_logical_core.y,
            tx_virtual_core.x,
            tx_virtual_core.y,
            (uint32_t)tx_eth_chan);

        // build receiver kernel(s)
        std::vector<uint32_t> rx_runtime_args = {packet_payload_size_bytes, num_packets, time_seed};

        auto rx_kernel = tt_metal::CreateKernel(
            rx_devices[0]->program_handle,
            rx_kernel_src,
            {rx_logical_cores[0]},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = compile_args,
                .defines = defines});

        tt_metal::SetRuntimeArgs(rx_devices[0]->program_handle, rx_kernel, rx_logical_cores[0], rx_runtime_args);

        log_info(
            LogTest,
            "[Device: Phys: {}] RX running on: logical: x={},y={}, virtual: x{},y={}",
            rx_devices[0]->chip_id,
            rx_logical_cores[0].x,
            rx_logical_cores[0].y,
            rx_virtual_cores[0].x,
            rx_virtual_cores[0].y);
    }

    void notify_tx_worker() {
        uint32_t notfication_address = 0x24000;
        std::vector<uint32_t> start_signal(1, 1);
        tt::llrt::write_hex_vec_to_core(tx_device->chip_id, tx_virtual_core, start_signal, notfication_address);
    }

    void wait_for_workers_to_finish() {
        uint32_t test_results_address = 0x100000;

        // wait for rx workers
        for (auto i = 0; i < rx_devices.size(); i++) {
            CoreCoord rx_virtual_core =
                rx_devices[i]->IDevice_handle->worker_core_from_logical_core(rx_logical_cores[i]);
            while (true) {
                auto rx_status =
                    tt::llrt::read_hex_vec_from_core(rx_devices[i]->chip_id, rx_virtual_core, test_results_address, 4);
                if ((rx_status[0] & 0xFFFF) != 0) {
                    break;
                }
            }
        }

        // wait for tx worker
        while (true) {
            auto tx_status =
                tt::llrt::read_hex_vec_from_core(tx_device->chip_id, tx_virtual_core, test_results_address, 4);
            if ((tx_status[0] & 0xFFFF) != 0) {
                break;
            }
        }
    }

    bool collect_results() {
        uint32_t test_results_address = 0x100000;
        bool pass = true;

        // collect tx results
        // TODO: avoid invoking the device handle directly
        CoreCoord tx_virtual_core = tx_device->IDevice_handle->worker_core_from_logical_core(tx_logical_core);
        tx_results = tt::llrt::read_hex_vec_from_core(tx_device->chip_id, tx_virtual_core, test_results_address, 128);
        log_info(
            LogTest,
            "[Device: Phys: {}] TX status = {}",
            tx_device->chip_id,
            tt_fabric_status_to_string(tx_results[TT_FABRIC_STATUS_INDEX]));
        pass &= (tx_results[TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS);

        // collect rx results
        // TODO: avoid invoking the device handle directly
        for (auto i = 0; i < rx_devices.size(); i++) {
            CoreCoord rx_virtual_core =
                rx_devices[i]->IDevice_handle->worker_core_from_logical_core(rx_logical_cores[i]);
            rx_results.push_back(
                tt::llrt::read_hex_vec_from_core(rx_devices[i]->chip_id, rx_virtual_core, test_results_address, 128));
            log_info(
                LogTest,
                "[Device: Phys: {}] RX{} status = {}",
                rx_devices[i]->chip_id,
                i,
                tt_fabric_status_to_string(rx_results[i][TT_FABRIC_STATUS_INDEX]));
            pass &= (rx_results[i][TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS);
        }

        return pass;
    }

    bool validate_results() {
        bool pass = true;
        uint64_t num_tx_bytes =
            ((uint64_t)tx_results[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | tx_results[TT_FABRIC_WORD_CNT_INDEX];
        uint64_t num_rx_bytes;

        // tally-up data words and number of packets from rx and tx
        for (auto i = 0; i < rx_results.size(); i++) {
            num_rx_bytes =
                ((uint64_t)rx_results[i][TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | rx_results[i][TT_FABRIC_WORD_CNT_INDEX];
            pass &= (num_tx_bytes == num_rx_bytes);

            if (!pass) {
                break;
            }
        }

        return pass;
    }

    void print_result_summary() {
        uint64_t num_tx_bytes =
            ((uint64_t)tx_results[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | tx_results[TT_FABRIC_WORD_CNT_INDEX];
        uint64_t tx_elapsed_cycles =
            ((uint64_t)tx_results[TT_FABRIC_CYCLES_INDEX + 1] << 32) | tx_results[TT_FABRIC_CYCLES_INDEX];
        double tx_bw = ((double)num_tx_bytes) / tx_elapsed_cycles;

        log_info(
            LogTest,
            "[Device: Phys: {}] TX bytes sent: {}, elapsed cycles: {} -> BW: {:.2f} B/cycle",
            tx_device->chip_id,
            num_tx_bytes,
            tx_elapsed_cycles,
            tx_bw);
    }
};

void generate_traffic_instances(
    std::vector<std::pair<chip_id_t, chan_id_t>>& connection_info, std::vector<TestFabricTraffic>& result) {
    auto tx_devices = connection_info;

    // maintain the order of the chips in the fabric spec
    std::unordered_set<chip_id_t> chips_in_fabric_;
    for (const auto& [chip_id, eth_chan] : connection_info) {
        chips_in_fabric_.insert(chip_id);
    }

    std::vector<chip_id_t> chips_in_fabric(chips_in_fabric_.begin(), chips_in_fabric_.end());

    // shuffle to induce randomness
    std::shuffle(tx_devices.begin(), tx_devices.end(), global_rng);

    // TODO: take in the parameter to generate n number of tx-rx pairs in a given fabric instance
    uint32_t num_tx_rx_pairs = 1;

    // TODO: take in other params like number of hops

    // TODO: take in the ucast/mcast param to figure out number of rx per tx
    // for now, generate unicast mapping

    for (auto i = 0; i < num_tx_rx_pairs; i++) {
        auto rx_chips = chips_in_fabric;
        auto tx_device = tx_devices.back();
        // remove the tx device from the list, as there can only be one sender per EDM
        tx_devices.pop_back();

        // for now avoid having the receiever on the same chip as the sender
        rx_chips.erase(std::find(rx_chips.begin(), rx_chips.end(), tx_device.first));

        std::shuffle(rx_chips.begin(), rx_chips.end(), global_rng);

        auto rx_chip_id = rx_chips.back();

        // find the distance b/w tx and rx chips
        uint32_t num_hops = std::abs(std::distance(
            std::find(chips_in_fabric.begin(), chips_in_fabric.end(), tx_device.first),
            std::find(chips_in_fabric.begin(), chips_in_fabric.end(), rx_chip_id)));

        TestFabricTraffic traffic_instance(
            tx_device.second,
            num_hops,
            test_board.get_test_device(tx_device.first),
            {test_board.get_test_device(rx_chip_id)});
        result.push_back(std::move(traffic_instance));
    }
}

struct TestLineFabric {
    std::vector<std::pair<chip_id_t, chan_id_t>> connection_info;
    std::vector<TestFabricTraffic> traffic_instances;

    TestLineFabric(std::vector<std::pair<chip_id_t, chan_id_t>>& connection_info_) {
        connection_info = connection_info_;

        // initialize EDM workers
        for (auto i = 0; i < connection_info.size(); i++) {
            auto chip_id = connection_info[i].first;
            auto eth_chan = connection_info[i].second;
            auto test_device = test_board.get_test_device(chip_id);

            if (i == 0) {
                test_device->init_EDM_worker(eth_chan, std::nullopt, connection_info[i + 1]);
            } else if (i == connection_info.size() - 1) {
                test_device->init_EDM_worker(eth_chan, connection_info[i - 1], std::nullopt);
            } else {
                test_device->init_EDM_worker(eth_chan, connection_info[i - 1], connection_info[i + 1]);
            }
        }

        // connect downstream EDMs
        for (auto i = 1; i < connection_info.size(); i++) {
            auto curr_chip_id = connection_info[i].first;
            auto prev_chip_id = connection_info[i - 1].first;

            if (curr_chip_id == prev_chip_id) {
                auto test_device = test_board.get_test_device(curr_chip_id);
                test_device->connect_EDM_workers(connection_info[i - 1].second, connection_info[i].second);
            }
        }

        // generate tx<>rx map and instantiate traffic instances
        generate_traffic_instances(connection_info, traffic_instances);
    }

    void build_kernels() {
        for (auto& instance : traffic_instances) {
            instance.build_worker_kernels();
        }
    }

    void notify_tx_workers() {
        for (auto& instance : traffic_instances) {
            instance.notify_tx_worker();
        }
    }

    void wait_for_workers_to_finish() {
        for (auto& instance : traffic_instances) {
            instance.wait_for_workers_to_finish();
        }
    }

    bool collect_results() {
        bool pass = true;
        for (auto& instance : traffic_instances) {
            pass &= instance.collect_results();
            if (!pass) {
                break;
            }
        }

        return pass;
    }

    bool validate_results() {
        bool pass = true;
        for (auto& instance : traffic_instances) {
            pass &= instance.validate_results();
            if (!pass) {
                break;
            }
        }

        return pass;
    }

    void print_result_summary() {
        for (auto& instance : traffic_instances) {
            instance.print_result_summary();
        }
    }
};

int main(int argc, char** argv) {
    constexpr uint32_t default_prng_seed = 0xFFFFFFFF;
    constexpr uint32_t default_fabric_length = 2;
    constexpr uint32_t default_num_packets = 5000;
    constexpr uint32_t default_packet_size_kb = 4;
    constexpr const char* default_board_type = "tg";

    std::vector<std::string> input_args(argv, argv + argc);
    uint32_t prng_seed = test_args::get_command_option_uint32(input_args, "--prng_seed", default_prng_seed);
    uint32_t fabric_length = test_args::get_command_option_uint32(input_args, "--fabric_length", default_fabric_length);
    uint32_t num_packets = test_args::get_command_option_uint32(input_args, "--num_packets", default_num_packets);
    uint32_t packet_size_kb =
        test_args::get_command_option_uint32(input_args, "--packet_size_kb", default_packet_size_kb);
    std::string board_type = test_args::get_command_option(input_args, "--board_type", std::string(default_board_type));

    bool pass = true;

    if (fabric_length < 2) {
        throw std::runtime_error("Minimum length of fabric is 2");
    }

    if (default_prng_seed == prng_seed) {
        std::random_device rd;
        prng_seed = rd();
    }

    global_rng.seed(prng_seed);
    time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    try {
        test_board.init(board_type);

        std::vector<std::vector<std::pair<chip_id_t, chan_id_t>>> fabric_connections;
        std::vector<TestLineFabric> line_fabrics;

        // TODO: programmatically generate these potentially using control plane
        // fabric_connections.push_back({{4, 0}, {11, 0}});  // 1 hop N<>S
        // fabric_connections.push_back({{4, 1}, {11, 1}});                                         // 1 hop N<>S
        // fabric_connections.push_back({{5, 4}, {6, 12}});                                         // 1 hop E<>W
        // fabric_connections.push_back({{17, 8}, {20, 8}, {20, 0}, {23, 0}});                      // 3 hops N<>S
        fabric_connections.push_back({{32, 12}, {27, 4}, {27, 12}, {26, 4}, {26, 12}, {25, 4}});  // 4 hops E<>W

        // init and build line fabric
        for (auto& connection_info : fabric_connections) {
            line_fabrics.emplace_back(TestLineFabric(connection_info));
        }

        // build edm kernels
        for (auto& [chip, device] : test_board.test_device_map) {
            device->build_EDM_kernels();
        }

        // build worker kernels
        for (auto& fabric : line_fabrics) {
            fabric.build_kernels();
        }

        // launch programs
        for (auto& [chip, device] : test_board.test_device_map) {
            device->launch_program();
        }

        std::cout << "seed: " << prng_seed << std::endl;

        // wait for EDM handshake to be done
        for (auto& [chip, device] : test_board.test_device_map) {
            device->wait_for_EDM_handshake();
        }

        std::cout << "handshake done" << std::endl;

        // start traffic
        for (auto& fabric : line_fabrics) {
            fabric.notify_tx_workers();
        }

        auto start = std::chrono::system_clock::now();

        // wait for all workers to finish
        // if we only wait for rx workers and there is a data mismatch, rx kernels will terminate earlier
        // while the tx kernels might still have an active connection with the EDM. This will cause the
        // test to hang
        for (auto& fabric : line_fabrics) {
            fabric.wait_for_workers_to_finish();
        }

        std::cout << "workers finished" << std::endl;

        // gracefully terminate EDM kernels
        for (auto& [chip, device] : test_board.test_device_map) {
            device->terminate_EDM_kernels();
        }

        // wait for program done
        for (auto& [chip, device] : test_board.test_device_map) {
            device->wait_for_program_done();
        }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = (end - start);
        log_info(LogTest, "Ran in {:.2f}us", elapsed_seconds.count() * 1000 * 1000);

        // collect results
        for (auto& fabric : line_fabrics) {
            pass &= fabric.collect_results();
            if (!pass) {
                log_fatal(LogTest, "Result collection failed\n");
            }
        }

        // close devices
        test_board.close_devices();

        // validate results
        for (auto& fabric : line_fabrics) {
            pass &= fabric.validate_results();
            if (!pass) {
                log_fatal(LogTest, "Result validation failed\n");
            }
        }

        // print results
        if (pass) {
            for (auto& fabric : line_fabrics) {
                fabric.print_result_summary();
            }
        }

    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    tt::llrt::RunTimeOptions::get_instance().set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }

    return 0;
}
