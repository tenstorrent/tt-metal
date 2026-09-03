// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// l2cpu_fabric_forward — host orchestrator (Tasks 1, 2, 4 of the plan)
// ============================================================================
//
// A Tensix "producer" kernel on Device A stages a payload into the on-die L2CPU
// (x280) LIM and pokes the x280 mailbox. The x280 fabric-worker firmware then
// forwards the payload off-chip to Device B over one ethernet link (feeding the
// standard, unmodified EDM router), where a Tensix "receiver" kernel moves it to
// chip-B DRAM for host readback. See:
//   docs/superpowers/specs/2026-09-03-x280-fabric-worker-design.md
//   docs/superpowers/plans/2026-09-03-x280-fabric-worker.md
//
// ----------------------------------------------------------------------------
// HARDWARE STATUS
// ----------------------------------------------------------------------------
// NO HARDWARE was available when this file was written. It is structured so it
// COMPILES and is clearly organized; every step that can only be exercised or
// validated on two fabric-connected Blackhole chips is called out with a
// `HARDWARE:` note. A later agent with two chips must validate/finish those.
//
// The x280 firmware (x280/fw_fabric.c), the producer kernel (kernels/producer.cpp)
// and the receiver kernel (kernels/receiver.cpp) are authored separately (plan
// Tasks 3, 5, 6); this host references them by path and JIT-compiles them at run
// time, so it builds even before those files exist.
// ----------------------------------------------------------------------------

#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

// Shared mailbox / connection-parameter contract with the x280 firmware. Plain
// #defines, safe to include here (see x280/fabric_mbox.h).
#include "x280/fabric_mbox.h"

#if defined(FF_HAVE_UMD)
// Raw UMD is used ONLY to reach the passive L2CPU tile (mailbox) — the same
// mechanism the sibling x280_boot tool uses. See the HARDWARE note in
// X280Mailbox below about coexistence with an open tt-metal MeshDevice.
#include "umd/device/cluster.hpp"
#endif

using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

uint32_t env_or(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    return v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : fallback;
}

const char* env_str(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

// The ten+ EDM connection parameters delivered to the x280 mailbox at FF_MBOX_CONN.
// Field order mirrors the FF_CONN_* offsets in x280/fabric_mbox.h.
struct EdmConnParams {
    uint32_t edm_noc_x = 0;
    uint32_t edm_noc_y = 0;
    uint32_t edm_buffer_base_addr = 0;
    uint32_t num_buffers_per_channel = 0;
    uint32_t buffer_size_bytes = 0;
    uint32_t edm_connection_handshake_l1_addr = 0;
    uint32_t edm_worker_location_info_addr = 0;
    uint32_t edm_copy_of_wr_counter_addr = 0;
    uint32_t sender_channel_credits_stream_id = 0;
    uint32_t worker_free_slots_l1_addr = 0;
    uint32_t num_hops = 1;

    // True once source_edm_connection_params() resolved the values from the live
    // fabric API (as opposed to leaving them at their zero defaults).
    bool resolved = false;
};

// A discovered A<->B fabric link.
struct Link {
    tt::ChipId chip_a = 0;
    tt::ChipId chip_b = 0;
    tt::tt_fabric::FabricNodeId node_a{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::FabricNodeId node_b{tt::tt_fabric::MeshId{0}, 0};
    uint32_t link_idx = 0;
    std::vector<uint32_t> forwarding_link_indices;
};

// ---------------------------------------------------------------------------
// Step 1 — Fabric bring-up + link discovery (plan Task 2)
//
// Mirrors tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_smoke.cpp
// and tests/tt_metal/tt_fabric/common/fabric_fixture.hpp, but uses the PUBLIC
// free-function fabric API (tt-metalium/experimental/fabric/fabric.hpp) instead of
// MetalContext::get_control_plane(): the control-plane object and its member
// get_forwarding_eth_chans_to_chip() live behind impl/context headers that are not
// on the programming-examples public include path. get_forwarding_link_indices()
// is the reachable equivalent — a non-empty result means the link is trained AND a
// forwarding channel to the destination exists.
//
// Caller MUST have already called SetFabricConfig(FABRIC_1D). This queries the
// control plane; abort early here (BEFORE booting the x280) if the link is absent.
// ---------------------------------------------------------------------------
std::optional<Link> discover_link(tt::ChipId chip_a, tt::ChipId chip_b) {
    Link link;
    link.chip_a = chip_a;
    link.chip_b = chip_b;

    // HARDWARE: requires the fabric control plane, initialized from the live
    // cluster topology after SetFabricConfig(FABRIC_1D). Cannot run without HW.
    link.node_a = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(chip_a);
    link.node_b = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(chip_b);

    link.forwarding_link_indices = tt::tt_fabric::get_forwarding_link_indices(link.node_a, link.node_b);
    if (link.forwarding_link_indices.empty()) {
        fmt::print(
            stderr,
            "ABORT: no forwarding link from chip {} (mesh {}, chip {}) to chip {} (mesh {}, chip {}). "
            "Fabric not trained or chips not adjacent.\n",
            chip_a,
            *link.node_a.mesh_id,
            link.node_a.chip_id,
            chip_b,
            *link.node_b.mesh_id,
            link.node_b.chip_id);
        return std::nullopt;
    }
    link.link_idx = link.forwarding_link_indices.front();

    fmt::print(
        "fabric FABRIC_1D up: chip {} -> chip {} | fabric nodes (mesh {}, chip {}) -> (mesh {}, chip {}) | "
        "forwarding link indices {} (using idx {})\n",
        chip_a,
        chip_b,
        *link.node_a.mesh_id,
        link.node_a.chip_id,
        *link.node_b.mesh_id,
        link.node_b.chip_id,
        fmt::join(link.forwarding_link_indices, ","),
        link.link_idx);
    return link;
}

// ---------------------------------------------------------------------------
// Step 2 — Boot the x280 fabric-worker firmware on chip A (plan Task 4, Step 1)
//
// DESIGN CHOICE: shell out to the already-built sibling boot tool
// (metal_example_l2cpu_x280_boot) rather than replicating its raw-UMD boot
// sequence. Reasons: (a) the boot tool is the verified one-shot-reset boot path
// on this branch; (b) it runs as a SEPARATE process, so its UMD handle does not
// coexist with anything this process opens; (c) it must run while tt-metal holds
// no device, which is naturally true here because we boot BEFORE creating the
// MeshDevices. The released x280 hart keeps running afterwards (persistent),
// exactly as the sibling echo demo relies on.
//
// HARDWARE: the hart release is one-shot per chip reset. If boot fails because a
// prior run already released it, recover with `tt-smi -r`. Set FF_SKIP_BOOT=1 to
// skip in-process boot and boot the x280 manually first (the proven sibling flow).
// ---------------------------------------------------------------------------
bool boot_x280(const std::string& boot_tool, const std::string& fw_bin) {
    if (env_or("FF_SKIP_BOOT", 0)) {
        fmt::print("FF_SKIP_BOOT=1 — assuming the x280 was booted manually with {}.\n", fw_bin);
        return true;
    }
    std::string cmd = fmt::format("{} boot {}", boot_tool, fw_bin);
    fmt::print("Booting x280: {}\n", cmd);
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        fmt::print(
            stderr,
            "ABORT: x280 boot tool returned {} (cmd: {}). Build the firmware with x280/build_fw.sh and the boot "
            "tool target metal_example_l2cpu_x280_boot; recover a wedged hart with `tt-smi -r`.\n",
            rc,
            cmd);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// X280Mailbox — raw-UMD access to the passive L2CPU tile mailbox.
//
// HARDWARE (coexistence): this opens a raw UMD Cluster in-process. On this branch
// the L2CPU tile has only ever been poked via raw UMD from a SEPARATE process
// (the boot tool) or via a Tensix kernel over the NOC (the echo demo). Whether a
// second raw UMD handle can be opened while a tt-metal MeshDevice already owns the
// device is UNVALIDATED and may throw ("device busy"). If it does, the hardware
// agent should either (a) route the connection-param write through a one-time
// Tensix setup kernel (plan Task 4 fallback), or (b) sequence device open/close
// around this raw-UMD phase. Construction failures are caught and reported rather
// than aborting the whole program, so the boundary is obvious on HW.
// ---------------------------------------------------------------------------
class X280Mailbox {
public:
    X280Mailbox(tt::ChipId chip, uint32_t l2cpu_x, uint32_t l2cpu_y) : chip_(chip) {
#if defined(FF_HAVE_UMD)
        try {
            cluster_ = std::make_unique<tt::umd::Cluster>();
            core_ = tt::umd::CoreCoord(l2cpu_x, l2cpu_y, tt::CoreType::L2CPU, tt::CoordSystem::NOC0);
            ok_ = true;
        } catch (const std::exception& e) {
            fmt::print(
                stderr,
                "HARDWARE: could not open a raw UMD cluster for the L2CPU mailbox ({}). See the coexistence note in "
                "X280Mailbox — route mailbox writes through a Tensix setup kernel instead.\n",
                e.what());
            ok_ = false;
        }
#else
        (void)l2cpu_x;
        (void)l2cpu_y;
        fmt::print(stderr, "HARDWARE: built without umd::tt-umd — the L2CPU mailbox path is unavailable.\n");
#endif
    }

    bool ok() const { return ok_; }

    void write_u32(uint32_t addr, uint32_t value) {
#if defined(FF_HAVE_UMD)
        cluster_->write_to_device(&value, sizeof(value), chip_, core_, addr);
#else
        (void)addr;
        (void)value;
#endif
    }

    uint32_t read_u32(uint32_t addr) {
#if defined(FF_HAVE_UMD)
        uint32_t value = 0;
        cluster_->read_from_device(&value, chip_, core_, addr, sizeof(value));
        return value;
#else
        (void)addr;
        return 0;
#endif
    }

    uint64_t read_u64(uint32_t addr) {
#if defined(FF_HAVE_UMD)
        uint64_t value = 0;
        cluster_->read_from_device(&value, chip_, core_, addr, sizeof(value));
        return value;
#else
        (void)addr;
        return 0;
#endif
    }

private:
    tt::ChipId chip_;
    bool ok_ = false;
#if defined(FF_HAVE_UMD)
    std::unique_ptr<tt::umd::Cluster> cluster_;
    tt::umd::CoreCoord core_{0, 0, tt::CoreType::L2CPU, tt::CoordSystem::NOC0};
#endif
};

// Poll FF_MBOX_HEARTBEAT to confirm the x280 hart is actually running (plan Task 4,
// Step 1). Returns true if the heartbeat advances. HARDWARE-only.
bool poll_x280_heartbeat(X280Mailbox& mbox) {
    if (!mbox.ok()) {
        return false;
    }
    uint64_t first = mbox.read_u64(FF_MBOX_HEARTBEAT);
    uint64_t fw_state = mbox.read_u64(FF_MBOX_FW_STATE);
    // Re-read a few times; the firmware bumps the heartbeat every loop.
    uint64_t last = first;
    for (int i = 0; i < 5; i++) {
        last = mbox.read_u64(FF_MBOX_HEARTBEAT);
        if (last != first) {
            break;
        }
    }
    fmt::print(
        "x280 heartbeat: {} -> {} | fw_state=0x{:x} (expect FF_STATE_ALIVE=0x{:x})\n",
        first,
        last,
        fw_state,
        static_cast<uint64_t>(FF_STATE_ALIVE));
    return last != first;
}

// ---------------------------------------------------------------------------
// Step 3 — Source the EDM connection parameters (plan Task 4, Step 2)
//
// THIS IS THE LEAST-CERTAIN PIECE (see spec "Risks" #3). The fabric API bakes the
// connection parameters into a Tensix kernel's runtime args; here we must deliver
// them to the x280 firmware instead.
//
// APPROACH (host-side, real): call the public append_fabric_connection_rt_args()
// with core_type == ETH. For the ETH/VC2 path (fabric.cpp:239-291) this emits the
// FULL SenderWorkerAdapterSpec as twelve u32 runtime args, in the order defined by
// append_worker_to_fabric_edm_sender_rt_args (erisc_datamover_builder.cpp:542-554):
//   [0] edm_direction
//   [1] edm_noc_xy           (WorkerXY: x = v & 0xFFFF, y = v >> 16)
//   [2] edm_buffer_base_addr
//   [3] num_buffers_per_channel
//   [4] edm_l1_sem_addr
//   [5] edm_connection_handshake_addr
//   [6] edm_worker_location_info_addr
//   [7] buffer_size_bytes
//   [8] buffer_index_semaphore_id   (== edm_copy_of_wr_counter_addr)
//   [9] worker flow-control sem id
//   [10] worker teardown sem id
//   [11] worker buffer-index sem id
// We parse [1..8] into the mailbox conn block — these are REAL, live values.
//
// Two fields are NOT in that arg list and are handled explicitly:
//   * sender_channel_credits_stream_id — the x280 -> EDM credit stream register id.
//     In the device-init (VC0) worker path this is conn->worker_free_slots_stream_id
//     from the L1 connection table; the ETH/VC2 path defaults it to a compile-time
//     STREAM_ID and does not emit it. It is genuinely unresolved here.
//     >>> TODO(hardware): resolve real EDM connection params — see
//     >>>   edm_fabric_worker_adapters.hpp:112-147 (VC0 L1 conn table path).
//   * worker_free_slots_l1_addr — a HOST-CHOSEN LIM address where the EDM pushes
//     free-slot credits for the x280 to poll. Not sourced; we pick one below.
//
// HARDWARE: append_fabric_connection_rt_args() queries the live fabric builder
// context (channel allocator, eth core placement), so it only runs with fabric
// trained on real chips. It is a linked public template (instantiated for Program),
// so this compiles without HW.
// ---------------------------------------------------------------------------
EdmConnParams source_edm_connection_params(const Link& link, uint32_t worker_free_slots_lim_addr) {
    EdmConnParams params;
    params.num_hops = link.forwarding_link_indices.empty() ? 1u : 1u;  // single hop for adjacent chips
    params.worker_free_slots_l1_addr = worker_free_slots_lim_addr;

    // Host-chosen credit stream id. The value below is a placeholder default; the
    // real id must be resolved from the device-init L1 connection table on HW.
    // >>> TODO(hardware): resolve real EDM connection params — see
    // >>>   edm_fabric_worker_adapters.hpp:112-147
    params.sender_channel_credits_stream_id = env_or("FF_CREDITS_STREAM_ID", 0);

    // Emit the ETH-path runtime args against a throwaway program to read the real
    // EDM addresses/geometry. HARDWARE-only (needs the live fabric builder context).
    std::vector<uint32_t> args;
    try {
        Program throwaway = CreateProgram();
        const CoreCoord dummy_core{0, 0};
        tt::tt_fabric::append_fabric_connection_rt_args(
            link.node_a, link.node_b, link.link_idx, throwaway, dummy_core, args, tt::CoreType::ETH);
    } catch (const std::exception& e) {
        fmt::print(
            stderr,
            "HARDWARE: append_fabric_connection_rt_args(ETH) failed ({}). EDM connection params unresolved; the "
            "x280 mailbox conn block will be left at defaults. Resolve on HW.\n",
            e.what());
        return params;  // resolved stays false
    }

    // The ETH/VC2 path appends exactly twelve values; refuse to guess if the count
    // is unexpected (the arg contract may have changed).
    if (args.size() < 12) {
        fmt::print(
            stderr,
            "HARDWARE: append_fabric_connection_rt_args(ETH) returned {} args (expected >= 12). The VC2 arg contract "
            "may have changed — re-check erisc_datamover_builder.cpp. Leaving conn params at defaults.\n",
            args.size());
        return params;
    }

    const tt::tt_fabric::WorkerXY edm_xy = tt::tt_fabric::WorkerXY::from_uint32(args[1]);
    params.edm_noc_x = edm_xy.x;
    params.edm_noc_y = edm_xy.y;
    params.edm_buffer_base_addr = args[2];
    params.num_buffers_per_channel = args[3];
    params.edm_connection_handshake_l1_addr = args[5];
    params.edm_worker_location_info_addr = args[6];
    params.buffer_size_bytes = args[7];
    params.edm_copy_of_wr_counter_addr = args[8];
    params.resolved = true;
    return params;
}

// Write the resolved conn params into the x280 mailbox at FF_MBOX_CONN, then read
// them back and print (plan Task 4, Steps 3-4). HARDWARE-only.
void write_conn_params_to_x280(X280Mailbox& mbox, const EdmConnParams& p) {
    if (!mbox.ok()) {
        fmt::print(stderr, "HARDWARE: mailbox unavailable — cannot deliver EDM connection params to the x280.\n");
        return;
    }
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_EDM_NOC_X, p.edm_noc_x);
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_EDM_NOC_Y, p.edm_noc_y);
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_EDM_BUFFER_BASE, p.edm_buffer_base_addr);
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_NUM_BUFFERS, p.num_buffers_per_channel);
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_BUFFER_SIZE, p.buffer_size_bytes);
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_HANDSHAKE_ADDR, p.edm_connection_handshake_l1_addr);
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_WORKER_LOC_INFO, p.edm_worker_location_info_addr);
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_WR_COUNTER_ADDR, p.edm_copy_of_wr_counter_addr);
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_CREDITS_STREAM_ID, p.sender_channel_credits_stream_id);
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_WORKER_FREESLOTS_L1, p.worker_free_slots_l1_addr);
    mbox.write_u32(FF_MBOX_CONN + FF_CONN_NUM_HOPS, p.num_hops);

    fmt::print(
        "EDM connection params delivered to x280 mailbox @ 0x{:08x} ({}):\n",
        FF_MBOX_CONN,
        p.resolved ? "resolved from live fabric API" : "DEFAULTS — unresolved, see TODO(hardware)");
    fmt::print(
        "  edm_noc_x/y            = ({}, {})  [read-back ({}, {})]\n",
        p.edm_noc_x,
        p.edm_noc_y,
        mbox.read_u32(FF_MBOX_CONN + FF_CONN_EDM_NOC_X),
        mbox.read_u32(FF_MBOX_CONN + FF_CONN_EDM_NOC_Y));
    fmt::print("  edm_buffer_base_addr   = 0x{:08x}\n", p.edm_buffer_base_addr);
    fmt::print("  num_buffers_per_channel= {}\n", p.num_buffers_per_channel);
    fmt::print("  buffer_size_bytes      = {}\n", p.buffer_size_bytes);
    fmt::print("  handshake_l1_addr      = 0x{:08x}\n", p.edm_connection_handshake_l1_addr);
    fmt::print("  worker_location_info   = 0x{:08x}\n", p.edm_worker_location_info_addr);
    fmt::print("  copy_of_wr_counter     = 0x{:08x}\n", p.edm_copy_of_wr_counter_addr);
    fmt::print(
        "  credits_stream_id      = {}  (TODO(hardware): resolve real id)\n", p.sender_channel_credits_stream_id);
    fmt::print("  worker_free_slots_l1   = 0x{:08x}  (host-chosen LIM sink)\n", p.worker_free_slots_l1_addr);
    fmt::print("  num_hops               = {}\n", p.num_hops);
}

}  // namespace

// ===========================================================================
// main
// ===========================================================================
int main() {
    // ----- Tunables (env-overridable, mirroring the sibling example) ---------
    const uint32_t l2cpu_x = env_or("L2CPU_X", 8);
    const uint32_t l2cpu_y = env_or("L2CPU_Y", 3);
    const uint32_t payload_size = env_or("FF_PAYLOAD_SIZE", 2048);  // bytes, <= EDM buffer slot
    // x280 physical LIM address of the payload the producer stages (clear of boot code).
    const uint32_t lim_payload_addr = env_or("FF_LIM_PAYLOAD_ADDR", 0x0800'0000u + 0x1'0000u);
    // LIM address the EDM pushes free-slot credits into for the x280 to poll.
    const uint32_t worker_free_slots_lim_addr = env_or("FF_LIM_FREESLOTS_ADDR", 0x0800'0000u + 0x2'0000u);
    // chip-B receiver L1 destination for the delivered payload.
    const uint32_t dest_l1_addr = env_or("FF_DEST_L1_ADDR", 0);  // 0 => use the allocated L1 buffer address
    const std::string boot_tool = env_str("FF_BOOT_TOOL", "./build/programming_examples/metal_example_l2cpu_x280_boot");
    const std::string fw_bin =
        env_str("FF_FW_BIN", "tt_metal/programming_examples/l2cpu_fabric_forward/x280/build/fw_fabric.bin");
    const uint32_t seq = static_cast<uint32_t>(time(nullptr)) | 1u;  // unique + nonzero per run

    bool pass = false;

    try {
        // ---- Enumerate chips; need at least two --------------------------------
        auto num_devices = GetNumAvailableDevices();
        if (num_devices < 2) {
            fmt::print(stderr, "ABORT: need 2 chips for the A->B fabric example, found {}.\n", num_devices);
            return 1;
        }
        const tt::ChipId chip_a = 0;
        const tt::ChipId chip_b = 1;

        // ---- Fabric bring-up (BEFORE creating devices) -------------------------
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);

        // ---- Link discovery + early abort (BEFORE booting the x280) ------------
        // HARDWARE: get_forwarding_link_indices() needs the control plane, which is
        // built from the live cluster after SetFabricConfig. If on HW this proves to
        // require a created MeshDevice, the hardware agent can move discovery just
        // after create_unit_meshes() below (accepting that a no-link abort would then
        // happen after boot, costing the one-shot reset — recover with `tt-smi -r`).
        auto maybe_link = discover_link(chip_a, chip_b);
        if (!maybe_link.has_value()) {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
            return 1;
        }
        const Link link = maybe_link.value();

        // ---- Boot the x280 fabric-worker firmware on chip A --------------------
        // Done here, before create_unit_meshes(), so the boot subprocess has the
        // device to itself. The released hart keeps running afterwards.
        if (!boot_x280(boot_tool, fw_bin)) {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
            return 1;
        }
        // Confirm the hart is alive via its heartbeat (raw UMD to the L2CPU tile;
        // no tt-metal device open yet, so no coexistence concern at this point).
        {
            X280Mailbox mbox(chip_a, l2cpu_x, l2cpu_y);
            if (!poll_x280_heartbeat(mbox)) {
                fmt::print(stderr, "HARDWARE: x280 heartbeat not advancing — hart may not be running.\n");
                // Not fatal for the structural run; the HW agent must ensure it advances.
            }
        }

        // ---- Create the two unit meshes ----------------------------------------
        // Device-init populates each Tensix core's L1 fabric connection table and
        // (with FABRIC_1D) trains the EDM routers.
        auto meshes = distributed::MeshDevice::create_unit_meshes({static_cast<int>(chip_a), static_cast<int>(chip_b)});
        auto device_a = meshes.at(static_cast<int>(chip_a));
        auto device_b = meshes.at(static_cast<int>(chip_b));

        if (device_a->arch() != tt::ARCH::BLACKHOLE) {
            fmt::print(stderr, "ABORT: this example targets Blackhole (L2CPU tiles only exist there).\n");
            for (auto& [id, dev] : meshes) {
                dev->close();
            }
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
            return 1;
        }

        // ---- Source EDM connection params + deliver to the x280 ----------------
        EdmConnParams conn = source_edm_connection_params(link, worker_free_slots_lim_addr);
        {
            X280Mailbox mbox(chip_a, l2cpu_x, l2cpu_y);
            write_conn_params_to_x280(mbox, conn);
        }

        // ---- Stage payload + wire up producer (chip A) / receiver (chip B) -----
        const uint32_t num_words = payload_size / sizeof(uint32_t);

        distributed::MeshCommandQueue& cq_a = device_a->mesh_command_queue();
        distributed::MeshCommandQueue& cq_b = device_b->mesh_command_queue();

        // Chip-A source DRAM (staged payload) + L1 scratch for the producer.
        distributed::DeviceLocalBufferConfig dram_cfg{.page_size = payload_size, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig l1_cfg{.page_size = payload_size, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig buf_size{.size = payload_size};

        auto src_dram_a = distributed::MeshBuffer::create(buf_size, dram_cfg, device_a.get());
        auto l1_scratch_a = distributed::MeshBuffer::create(buf_size, l1_cfg, device_a.get());

        // Chip-B destination L1 (fabric delivery target) + DRAM (host readback).
        auto dst_l1_b = distributed::MeshBuffer::create(buf_size, l1_cfg, device_b.get());
        auto dst_dram_b = distributed::MeshBuffer::create(buf_size, dram_cfg, device_b.get());

        const uint32_t receiver_l1_addr = dest_l1_addr != 0 ? dest_l1_addr : static_cast<uint32_t>(dst_l1_b->address());

        // Receiver NOC coords on chip B (where the fabric delivers). The producer
        // encodes these into the request so the x280 builds the packet header.
        constexpr CoreCoord producer_core{0, 0};
        constexpr CoreCoord receiver_core{0, 0};
        const CoreCoord receiver_noc = device_b->worker_core_from_logical_core(receiver_core);

        // Stage the known pattern into chip-A source DRAM.
        std::vector<uint32_t> input(num_words);
        for (uint32_t i = 0; i < num_words; i++) {
            input[i] = 0xF00D'0000u + i;
        }
        distributed::EnqueueWriteMeshBuffer(cq_a, src_dram_a, input, /*blocking=*/false);

        // ---- Producer program (chip A) -----------------------------------------
        // INTEGRATION ASSUMPTION: producer.cpp (authored separately, plan Task 3)
        // mirrors l2cpu_noc_transfer/kernels/l2cpu_rw.cpp, so it consumes a
        // TensorAccessorArgs block for the source DRAM as compile args, then the
        // runtime args below (order fixed by the plan).
        Program producer_program = CreateProgram();
        std::vector<uint32_t> producer_ct_args;
        TensorAccessorArgs(*src_dram_a->get_backing_buffer()).append_to(producer_ct_args);
        KernelHandle producer_kernel = CreateKernel(
            producer_program,
            OVERRIDE_KERNEL_PREFIX "l2cpu_fabric_forward/kernels/producer.cpp",
            producer_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = producer_ct_args});
        SetRuntimeArgs(
            producer_program,
            producer_kernel,
            producer_core,
            {static_cast<uint32_t>(l1_scratch_a->address()),  // l1_src
             static_cast<uint32_t>(src_dram_a->address()),    // dram_src
             payload_size,                                    // size
             lim_payload_addr,                                // lim_addr
             l2cpu_x,                                         // l2cpu_x
             l2cpu_y,                                         // l2cpu_y
             FF_MBOX,                                         // mbox_base
             seq,                                             // seq
             static_cast<uint32_t>(receiver_noc.x),           // dest_noc_x
             static_cast<uint32_t>(receiver_noc.y),           // dest_noc_y
             receiver_l1_addr});                              // dest_l1_addr

        // ---- Receiver program (chip B) -----------------------------------------
        // INTEGRATION ASSUMPTION: receiver.cpp reads its own L1 at dst_l1_addr and
        // writes to DRAM (TensorAccessorArgs for the dest DRAM as compile args).
        Program receiver_program = CreateProgram();
        std::vector<uint32_t> receiver_ct_args;
        TensorAccessorArgs(*dst_dram_b->get_backing_buffer()).append_to(receiver_ct_args);
        KernelHandle receiver_kernel = CreateKernel(
            receiver_program,
            OVERRIDE_KERNEL_PREFIX "l2cpu_fabric_forward/kernels/receiver.cpp",
            receiver_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = receiver_ct_args});
        SetRuntimeArgs(
            receiver_program,
            receiver_kernel,
            receiver_core,
            {receiver_l1_addr,                              // dst_l1_addr
             static_cast<uint32_t>(dst_dram_b->address()),  // dram_dst
             payload_size});                                // size

        fmt::print(
            "Launching: producer on chip {} core ({},{}) -> x280 -> fabric -> receiver on chip {} L1 0x{:08x} "
            "(NOC {},{}), {} bytes, seq=0x{:08x}\n",
            chip_a,
            producer_core.x,
            producer_core.y,
            chip_b,
            receiver_l1_addr,
            receiver_noc.x,
            receiver_noc.y,
            payload_size,
            seq);

        // Launch receiver first (it waits for delivery), then the producer that
        // kicks off the x280 send.
        // HARDWARE: end-to-end delivery depends on the x280 firmware (plan Tasks
        // 5-6) actually opening the EDM connection and pushing the packet. Until
        // that firmware exists, the receiver will time out / read stale L1.
        distributed::MeshWorkload receiver_workload;
        receiver_workload.add_program(distributed::MeshCoordinateRange(device_b->shape()), std::move(receiver_program));
        distributed::EnqueueMeshWorkload(cq_b, receiver_workload, /*blocking=*/false);

        distributed::MeshWorkload producer_workload;
        producer_workload.add_program(distributed::MeshCoordinateRange(device_a->shape()), std::move(producer_program));
        distributed::EnqueueMeshWorkload(cq_a, producer_workload, /*blocking=*/false);

        distributed::Finish(cq_a);
        distributed::Finish(cq_b);

        // ---- Read back chip-B DRAM and verify ----------------------------------
        std::vector<uint32_t> result;
        distributed::EnqueueReadMeshBuffer(cq_b, result, dst_dram_b, /*blocking=*/true);

        uint32_t mismatches = 0;
        for (uint32_t i = 0; i < num_words && i < result.size(); i++) {
            if (result[i] != input[i]) {
                if (mismatches < 8) {
                    fmt::print(stderr, "  word {:4d}: expected 0x{:08x}, got 0x{:08x}\n", i, input[i], result[i]);
                }
                mismatches++;
            }
        }
        pass = (mismatches == 0) && (result.size() >= num_words);
        fmt::print(
            "chip-B readback: {} words | {}\n",
            num_words,
            pass ? "all match" : fmt::format("{} MISMATCHES", mismatches));

        // ---- Teardown ----------------------------------------------------------
        for (auto& [id, dev] : meshes) {
            if (!dev->close()) {
                pass = false;
            }
        }
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    } catch (const std::exception& e) {
        fmt::print(stderr, "Failed with exception: {}\n", e.what());
        // Best-effort fabric teardown before propagating.
        try {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
        } catch (...) {
        }
        throw;
    }

    fmt::print("{}\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
