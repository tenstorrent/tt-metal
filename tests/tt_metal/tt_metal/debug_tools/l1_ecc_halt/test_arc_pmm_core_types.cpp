// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole ARC PMM messages from each kind of core. A core announces an event record in its L1 to ARC by writing a
// 4-byte message to ARC's PMM message register; the PMM stores the message's low 16 bits in the mailbox word picked
// by the x/y the message carries, in a 17x12 grid. Those are the sender's physical (NOC0) coordinates. Ethernet and
// DRAM cores have translated coordinates outside the grid (x >= 17 or y >= 12), so these tests check:
// 1. A message carrying the sender's physical coordinates lands in that mailbox word, from Tensix, ERISC and DRISC.
// 2. A message carrying an ERISC's or DRISC's translated coordinates changes no mailbox word.
// ARC firmware that handles PMM messages takes the record: it clears the mailbox word and then the record's magic.
// Without such firmware, both keep what was written. The tests accept either.
// Each test clears the words it uses, before and after, so it leaves no message behind for ARC firmware.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/pcie/device_event_reader.hpp>
#include <tt-umd/pcie/pci_device.hpp>
#include <tt-umd/tt_device/tt_device.hpp>

#include "device_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "hostdev/l1_ecc_halt_constants.h"
#include "tests/tt_metal/tt_metal/eth/eth_test_common.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr const char* pmm_send_kernel = "tests/tt_metal/tt_metal/test_kernels/misc/arc_pmm_send.cpp";
constexpr const char* noc_id_regs_kernel = "tests/tt_metal/tt_metal/test_kernels/misc/noc_id_regs.cpp";
constexpr uint32_t noc_id_bits = 6;
constexpr uint32_t noc_id_mask = (1u << noc_id_bits) - 1;
constexpr uint32_t arc_pmm_grid_y = 12;
// The records use the halt record's event type, with their own cause so they can't be taken for a halt.
constexpr uint32_t test_event_cause = 0x7E570001;
// How long ARC firmware that handles PMM messages may take to take a record.
constexpr auto arc_take_deadline = std::chrono::seconds(1);
// Each core kind puts its own value in the record's first data word, so a record shows who wrote it.
constexpr uint32_t tensix_marker = 0x0A1;
constexpr uint32_t eth_marker = 0x0E1;
constexpr uint32_t drisc_marker = 0x0D1;

using KernelConfig = std::variant<DataMovementConfig, EthernetConfig, DramConfig>;

// A core to send from: its logical coordinates and kernel config, and its L1 scratch for the message and the record,
// at local_addr for the core and read_addr for the host (they differ on DRISC).
struct Sender {
    std::string name;
    CoreCoord logical;
    CoreType core_type;
    KernelConfig config;
    uint32_t local_addr;
    uint32_t marker;
    uint64_t read_addr;
};

}  // namespace

class ArcPmmCoreTypeTest : public BlackholeSingleCardFixture {
protected:
    void SetUp() override {
        BlackholeSingleCardFixture::SetUp();
        if (devices_.empty()) {
            return;  // parent was skipped
        }
        mesh_device_ = devices_[0].get();
        chip_ = mesh_device_->build_id();
        const auto arc = MetalContext::instance().get_cluster().get_soc_desc(chip_).get_cores(
            tt::CoreType::ARC, CoordSystem::TRANSLATED);
        // With NOC translation on, ARC has the same coordinates on both NOCs.
        arc_ = tt_cxy_pair(chip_, arc.at(0).x, arc.at(0).y);
        // Created before any send, since a reader sees only the events that arrive after it.
        const int pci_device_num = MetalContext::instance()
                                       .get_cluster()
                                       .get_driver()
                                       ->get_tt_device(chip_)
                                       ->get_pci_device()
                                       ->get_device_num();
        try {
            events_ = std::make_unique<tt::umd::DeviceEventReader>(pci_device_num);
            if (!events_->is_firmware_attached()) {
                events_.reset();
            }
        } catch (const std::exception& e) {
            events_.reset();
            log_info(tt::LogTest, "No device events from KMD: {}", e.what());
        }
        log_info(tt::LogTest, "Device events {}", events_ ? "checked" : "not checked");
    }

    // After ARC took s's record: finds the event it raised, and checks it and the record left in s's L1.
    void expect_event(const Sender& s, const CoreCoord& msg_xy) {
        if (!events_) {
            return;
        }
        const CoreCoord t = translated(s);
        const auto source = static_cast<uint16_t>(t.x | t.y << noc_id_bits);
        const std::vector<uint32_t> record = record_words(s, msg_xy);
        const auto deadline = std::chrono::steady_clock::now() + arc_take_deadline;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            for (const tt::umd::DeviceEvent& e : events_->read_for(16, left).events) {
                if (e.source != source) {
                    continue;
                }
                EXPECT_EQ(e.type, record[1]) << s.name;
                EXPECT_EQ(e.cause, record[2]) << s.name;
                EXPECT_EQ(e.data[0], record[3]) << s.name;
                EXPECT_EQ(e.data[1], record[4]) << s.name;
                EXPECT_EQ(e.data[2], record[5]) << s.name;
                // ARC clears only the magic, so the host can read the rest of the record from the core.
                std::vector<uint32_t> in_l1(record.size(), 0);
                MetalContext::instance().get_cluster().read_core(
                    in_l1.data(),
                    in_l1.size() * sizeof(uint32_t),
                    tt_cxy_pair(chip_, t.x, t.y),
                    s.read_addr + (record_addr(s) - s.local_addr));
                std::vector<uint32_t> expected = record;
                expected[0] = 0;
                EXPECT_EQ(in_l1, expected) << s.name << ": record in L1";
                log_info(
                    tt::LogTest,
                    "{}: event seq {} source 0x{:x} type 0x{:x} cause 0x{:x} data 0x{:x} 0x{:x} 0x{:x}",
                    s.name,
                    e.seq,
                    e.source,
                    e.type,
                    e.cause,
                    e.data[0],
                    e.data[1],
                    e.data[2]);
                return;
            }
        }
        ADD_FAILURE() << s.name << ": no device event from translated (" << t.x << ", " << t.y << ")";
    }

    bool in_pmm_grid(const CoreCoord& xy) const { return xy.x < arc_pmm_grid_x && xy.y < arc_pmm_grid_y; }

    CoreCoord translated(const Sender& s) const {
        return mesh_device_->virtual_core_from_logical_core(s.logical, s.core_type);
    }

    CoreCoord noc0(const Sender& s) const {
        const CoreCoord t = translated(s);
        const auto p = MetalContext::instance().get_cluster().get_soc_desc(chip_).translate_coord_to(
            tt_xy_pair(t.x, t.y), CoordSystem::TRANSLATED, CoordSystem::NOC0);
        return CoreCoord{p.x, p.y};
    }

    uint32_t read_slot(const CoreCoord& xy) const {
        uint32_t word = 0;
        MetalContext::instance().get_cluster().read_reg(
            &word, arc_, arc_pmm_mailbox_addr + sizeof(uint32_t) * (arc_pmm_grid_x * xy.y + xy.x));
        return word;
    }

    // A message with no flags and address 0 clears the word for its x/y.
    void clear_slot(const CoreCoord& xy) const {
        const uint32_t clear = arc_pmm_message(xy.x, xy.y, 0, 0);
        MetalContext::instance().get_cluster().write_reg(&clear, arc_, arc_pmm_message_addr);
    }

    std::vector<uint32_t> snapshot() const {
        std::vector<uint32_t> words;
        for (uint32_t y = 0; y < arc_pmm_grid_y; y++) {
            for (uint32_t x = 0; x < arc_pmm_grid_x; x++) {
                words.push_back(read_slot(CoreCoord{x, y}));
            }
        }
        return words;
    }

    void run_kernel(const Sender& s, const char* kernel_path, const std::vector<uint32_t>& runtime_args) {
        Program program = CreateProgram();
        const KernelHandle kernel = std::visit(
            [&](const auto& config) { return CreateKernel(program, kernel_path, s.logical, config); }, s.config);
        SetRuntimeArgs(program, kernel, s.logical, runtime_args);
        distributed::MeshWorkload workload;
        auto zero = distributed::MeshCoordinate(0, 0);
        workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device_->mesh_command_queue());
    }

    // The record sits after the message, at the PMM's record alignment.
    static uint32_t record_addr(const Sender& s) {
        constexpr uint32_t a = l1_ecc_halt_record_alignment;
        return (s.local_addr + sizeof(uint32_t) + a - 1) / a * a;
    }

    uint32_t read_record_magic(const Sender& s) const {
        const CoreCoord t = translated(s);
        uint32_t magic = 0;
        MetalContext::instance().get_cluster().read_core(
            &magic, sizeof(magic), tt_cxy_pair(chip_, t.x, t.y), s.read_addr + (record_addr(s) - s.local_addr));
        return magic;
    }

    // ARC's tile event header (magic, type, cause), then 3 data words.
    static std::vector<uint32_t> record_words(const Sender& s, const CoreCoord& msg_xy) {
        return {
            l1_ecc_halt_record_magic,
            l1_ecc_halt_event_type,
            test_event_cause,
            s.marker,
            static_cast<uint32_t>(msg_xy.x),
            static_cast<uint32_t>(msg_xy.y)};
    }

    // Whether ARC took s's record: cleared the mailbox word at xy, then the record's magic.
    bool arc_took_record(const Sender& s, const CoreCoord& xy) const {
        const auto deadline = std::chrono::steady_clock::now() + arc_take_deadline;
        do {
            if (read_slot(xy) == 0 && read_record_magic(s) == 0) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } while (std::chrono::steady_clock::now() < deadline);
        return false;
    }

    // Writes a record into s's L1, runs the send kernel on s with a message announcing it and carrying msg_xy, and
    // returns the message.
    uint32_t send(const Sender& s, const CoreCoord& msg_xy) {
        const CoreCoord t = translated(s);
        const std::vector<uint32_t> record = record_words(s, msg_xy);
        MetalContext::instance().get_cluster().write_core(
            record.data(),
            record.size() * sizeof(uint32_t),
            tt_cxy_pair(chip_, t.x, t.y),
            s.read_addr + (record_addr(s) - s.local_addr));
        const uint32_t message = arc_pmm_message(msg_xy.x, msg_xy.y, arc_pmm_flag0, record_addr(s));
        run_kernel(
            s,
            pmm_send_kernel,
            {s.local_addr,
             message,
             static_cast<uint32_t>(arc_.x),
             static_cast<uint32_t>(arc_.y),
             arc_pmm_message_addr});
        return message;
    }

    static CoreCoord decode_noc_id(uint32_t reg) {
        return CoreCoord{reg & noc_id_mask, (reg >> noc_id_bits) & noc_id_mask};
    }

    // Reads s's NoC ID registers on the core and checks them against the SoC descriptor: NOC_NODE_ID on NOC0 must
    // be the physical (NOC0) coordinates, and NOC_ID_LOGICAL the translated ones. NOC1's are logged for reference.
    void expect_noc_id_registers(const Sender& s, bool read_noc1) {
        const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(chip_);
        const CoreCoord t = translated(s);
        const CoreCoord p = noc0(s);
        const auto n1 = soc_desc.translate_coord_to(tt_xy_pair(t.x, t.y), CoordSystem::TRANSLATED, CoordSystem::NOC1);
        std::vector<uint32_t> regs(4, 0);
        const std::vector<uint32_t> poison(4, 0xDEADBEEF);
        MetalContext::instance().get_cluster().write_core(
            poison.data(), poison.size() * sizeof(uint32_t), tt_cxy_pair(chip_, t.x, t.y), s.read_addr);
        run_kernel(s, noc_id_regs_kernel, {s.local_addr, read_noc1 ? 1u : 0u});
        MetalContext::instance().get_cluster().read_core(
            regs.data(), regs.size() * sizeof(uint32_t), tt_cxy_pair(chip_, t.x, t.y), s.read_addr);
        const CoreCoord node0 = decode_noc_id(regs[0]);
        const CoreCoord logical0 = decode_noc_id(regs[1]);
        const CoreCoord node1 = decode_noc_id(regs[2]);
        const CoreCoord logical1 = decode_noc_id(regs[3]);
        log_info(
            tt::LogTest,
            "{}: SoC descriptor translated ({}, {}), NOC0 ({}, {}), NOC1 ({}, {}) | NOC0: NODE_ID 0x{:x} = ({}, {}), "
            "ID_LOGICAL 0x{:x} = ({}, {}) | NOC1: NODE_ID 0x{:x} = ({}, {}), ID_LOGICAL 0x{:x} = ({}, {})",
            s.name,
            t.x,
            t.y,
            p.x,
            p.y,
            n1.x,
            n1.y,
            regs[0],
            node0.x,
            node0.y,
            regs[1],
            logical0.x,
            logical0.y,
            regs[2],
            node1.x,
            node1.y,
            regs[3],
            logical1.x,
            logical1.y);
        ASSERT_NE(regs[0], 0xDEADBEEF) << s.name << ": kernel did not write its results";
        EXPECT_EQ(node0, p) << s.name << ": NOC0 NOC_NODE_ID is not the physical (NOC0) coordinates";
        EXPECT_EQ(logical0, t) << s.name << ": NOC0 NOC_ID_LOGICAL is not the translated coordinates";
    }

    // Sends from s with its physical coordinates and expects the message in that mailbox word, and nowhere else.
    void expect_physical_coords_land(const Sender& s) {
        const CoreCoord t = translated(s);
        const CoreCoord p = noc0(s);
        ASSERT_TRUE(in_pmm_grid(p)) << s.name << " physical (" << p.x << ", " << p.y << ") is outside the PMM grid";
        clear_slot(p);
        const std::vector<uint32_t> before = snapshot();
        const uint32_t sent = send(s, p) & arc_pmm_mailbox_mask;
        const bool taken = arc_took_record(s, p);
        const uint32_t word = read_slot(p);
        const uint32_t magic = read_record_magic(s);
        clear_slot(p);
        log_info(
            tt::LogTest,
            "{}: translated ({}, {}), physical ({}, {}): mailbox word 0x{:x}, sent 0x{:x}, record magic 0x{:x}: {}",
            s.name,
            t.x,
            t.y,
            p.x,
            p.y,
            word,
            sent,
            magic,
            taken ? "ARC took the record" : "no ARC handler, the message is still in the mailbox");
        if (!taken) {
            EXPECT_EQ(word, sent) << s.name << ": message with physical coordinates did not land";
            EXPECT_EQ(magic, l1_ecc_halt_record_magic) << s.name << ": record magic changed";
        } else {
            expect_event(s, p);
        }
        std::vector<uint32_t> after = snapshot();
        after[arc_pmm_grid_x * p.y + p.x] = before[arc_pmm_grid_x * p.y + p.x];
        EXPECT_EQ(after, before) << s.name << ": another mailbox word changed";
    }

    // Sends from s with its translated coordinates, which are outside the grid: no mailbox word changes, and ARC never
    // sees the record.
    void expect_translated_coords_change_nothing(const Sender& s) {
        const CoreCoord t = translated(s);
        ASSERT_FALSE(in_pmm_grid(t)) << s.name << " translated (" << t.x << ", " << t.y << ") is inside the PMM grid";
        const std::vector<uint32_t> before = snapshot();
        const uint32_t sent = send(s, t) & arc_pmm_mailbox_mask;
        // Long enough for ARC firmware to take the record, had the message landed.
        std::this_thread::sleep_for(arc_take_deadline);
        const std::vector<uint32_t> after = snapshot();
        const uint32_t magic = read_record_magic(s);
        log_info(
            tt::LogTest,
            "{}: translated ({}, {}), sent 0x{:x}, record magic 0x{:x}",
            s.name,
            t.x,
            t.y,
            sent,
            magic);
        EXPECT_EQ(magic, l1_ecc_halt_record_magic) << s.name << ": ARC took a record announced from outside the grid";
        for (uint32_t i = 0; i < after.size(); i++) {
            if (after[i] != before[i]) {
                const CoreCoord xy{i % arc_pmm_grid_x, i / arc_pmm_grid_x};
                ADD_FAILURE() << s.name << ": mailbox word (" << xy.x << ", " << xy.y << ") changed from 0x" << std::hex
                              << before[i] << " to 0x" << after[i];
                clear_slot(xy);
            }
        }
    }

    Sender tensix_sender(CoreCoord logical = {0, 0}) const {
        const uint32_t l1 = mesh_device_->allocator()->get_base_allocator_addr(HalMemType::L1);
        return Sender{
            "Tensix (" + std::to_string(logical.x) + ", " + std::to_string(logical.y) + ")",
            logical,
            CoreType::WORKER,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            l1,
            tensix_marker,
            l1};
    }

    // Idle ethernet cores if the chip has any, otherwise its active ones.
    std::vector<Sender> eth_senders() const {
        auto* device = mesh_device_->impl().get_devices()[0];
        const auto& hal = MetalContext::instance().hal();
        std::unordered_set<CoreCoord> cores = device->get_inactive_ethernet_cores();
        EthernetConfig config{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .processor = DataMovementProcessor::RISCV_0};
        HalProgrammableCoreType core_type = HalProgrammableCoreType::IDLE_ETH;
        if (cores.empty()) {
            cores = device->get_active_ethernet_cores(true);
            config.eth_mode = Eth::SENDER;
            eth_test_common::set_arch_specific_eth_config(config);
            core_type = HalProgrammableCoreType::ACTIVE_ETH;
        }
        const uint32_t local_addr = hal.get_dev_addr(core_type, HalL1MemAddrType::UNRESERVED);
        std::vector<Sender> senders;
        for (const CoreCoord& core : cores) {
            senders.push_back(Sender{
                std::string(config.eth_mode == Eth::IDLE ? "idle" : "active") + " ERISC (" + std::to_string(core.x) +
                    ", " + std::to_string(core.y) + ")",
                core,
                CoreType::ETH,
                config,
                local_addr,
                eth_marker,
                local_addr});
        }
        return senders;
    }

    // One DRISC per DRAM bank, skipping each bank's NOC0 endpoint: syseng firmware owns it and keeps it in reset.
    std::vector<Sender> drisc_senders() const {
        const auto& hal = MetalContext::instance().hal();
        const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(chip_);
        const uint32_t local_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        // The host reaches DRISC L1 through a different NoC address than the core's own.
        const uint64_t read_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        const CoreCoord grid = soc_desc.get_dram_compute_grid_size();
        std::vector<Sender> senders;
        for (uint32_t bank = 0; bank < grid.x; bank++) {
            const CoreCoord noc0_endpoint =
                soc_desc.get_preferred_worker_core_for_dram_view(bank, static_cast<uint8_t>(NOC::NOC_0));
            const auto& endpoints = soc_desc.dram_bank_endpoint_coords.at(bank);
            for (uint32_t sub = 0; sub < grid.y; sub++) {
                if (sub < endpoints.size() && endpoints[sub] == noc0_endpoint) {
                    continue;
                }
                senders.push_back(Sender{
                    "DRISC bank " + std::to_string(bank) + " endpoint " + std::to_string(sub),
                    CoreCoord{bank, sub},
                    CoreType::DRAM,
                    DramConfig{.noc = NOC::NOC_0},
                    local_addr,
                    drisc_marker,
                    read_addr});
                break;
            }
        }
        return senders;
    }

    void skip_without_driscs() {
        if (!MetalContext::instance().hal().has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
    }

    distributed::MeshDevice* mesh_device_{};
    ChipId chip_{};
    tt_cxy_pair arc_{};
    // Null when KMD or ARC firmware does not publish device events.
    std::unique_ptr<tt::umd::DeviceEventReader> events_;
};

// Baseline: a Tensix kernel's message lands like the BRISC halt handler's.
TEST_F(ArcPmmCoreTypeTest, TensixPhysicalCoordsLand) { expect_physical_coords_land(tensix_sender()); }

TEST_F(ArcPmmCoreTypeTest, EthPhysicalCoordsLand) {
    const std::vector<Sender> senders = eth_senders();
    if (senders.empty()) {
        GTEST_SKIP() << "no ethernet cores";
    }
    for (const Sender& s : senders) {
        expect_physical_coords_land(s);
    }
}

TEST_F(ArcPmmCoreTypeTest, DriscPhysicalCoordsLand) {
    skip_without_driscs();
    for (const Sender& s : drisc_senders()) {
        expect_physical_coords_land(s);
    }
}

TEST_F(ArcPmmCoreTypeTest, EthTranslatedCoordsChangeNoMailboxWord) {
    const std::vector<Sender> senders = eth_senders();
    if (senders.empty()) {
        GTEST_SKIP() << "no ethernet cores";
    }
    expect_translated_coords_change_nothing(senders.front());
}

TEST_F(ArcPmmCoreTypeTest, DriscTranslatedCoordsChangeNoMailboxWord) {
    skip_without_driscs();
    expect_translated_coords_change_nothing(drisc_senders().front());
}

// Can a core find its own physical coordinates for the PMM message? Checks NOC_NODE_ID (expected physical) and
// NOC_ID_LOGICAL (expected translated) on NOC0, on Tensix (one core per column), every ERISC and one DRISC per bank.
// NOC1 registers are read and logged on Tensix and ERISC; DRISC reads only NOC0, the NOC its NIU is in stream mode on.
TEST_F(ArcPmmCoreTypeTest, NocIdRegistersMatchSocDescriptor) {
    const CoreCoord grid = mesh_device_->compute_with_storage_grid_size();
    for (uint32_t x = 0; x < grid.x; x++) {
        expect_noc_id_registers(tensix_sender(CoreCoord{x, 0}), true);
    }
    for (const Sender& s : eth_senders()) {
        expect_noc_id_registers(s, true);
    }
    if (MetalContext::instance().hal().has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        for (const Sender& s : drisc_senders()) {
            expect_noc_id_registers(s, false);
        }
    }
}
