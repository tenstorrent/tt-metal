// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/drisc_service/service.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/chip_helpers/sysmem_buffer.hpp>
#include <tt-umd/cluster_descriptor.hpp>
#include <tt-umd/driver_atomics.hpp>
#include <tt-umd/arch/blackhole_implementation.hpp>
#include <tt-umd/io_window/io_window.hpp>
#include <tt-umd/pcie/pci_device.hpp>
#include <tt-umd/pcie/tlb_window.hpp>
#include <tt-umd-workload/cluster.hpp>

#include "impl/buffers/dram_sender_topology.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_elffile.hpp"

namespace tt::tt_metal::drisc_service {

namespace {

// The kernel's bounce slots are 64 B aligned and the NoC cannot shift bytes across lanes, so DRAM and PCIe sources
// must be too. Destinations are held to the same rule.
constexpr uint64_t command_alignment = 64;

// The Blackhole PCIe tile passes NoC addresses below this to PCIe unchanged (KMD's noc_dma_limit); a device address at
// or above it needs an iATU region.
constexpr uint64_t pcie_direct_window_end = 1ULL << 58;

// Covers a launching client's whole attach, the longest any client holds the gate.
constexpr std::chrono::milliseconds gate_timeout{4 * abi::BOOT_TIMEOUT_MS};

// Issue windows this process holds, per chip; at most max_windows_per_chip each.
std::mutex window_count_mutex;
std::unordered_map<ChipId, uint32_t> window_count;

uint32_t pcie_noc_xy(tt::umd::Cluster& cluster, ChipId chip) {
    const auto pcie = cluster.get_soc_descriptor(chip).get_cores(CoreType::PCIE, CoordSystem::TRANSLATED).front();
    return abi::noc_xy(pcie.x, pcie.y);
}

// The kernel trusts every field: a CMD_LOCAL size off the 16 B grid is truncated yet reported done, and an unknown
// kind runs as a NoC copy.
void validate_command(const abi::KernelCommand& command, uint64_t bank_size) {
    const abi::CommandKind kind = abi::command_kind(command);
    TT_FATAL(
        kind == abi::CMD_SINGLE || kind == abi::CMD_LOCAL,
        "drisc_service: command kind {} is neither CMD_SINGLE nor CMD_LOCAL",
        static_cast<uint32_t>(kind));
    TT_FATAL(command.size != 0, "drisc_service: command size is 0");
    TT_FATAL(
        command.src_noc_xy_addr >> abi::CMD_KIND_SHIFT == 0,
        "drisc_service: source coords {:#x} do not fit in {} bits",
        command.src_noc_xy_addr,
        abi::CMD_KIND_SHIFT);
    TT_FATAL(
        command.src_addr % command_alignment == 0 && command.dst_addr % command_alignment == 0,
        "drisc_service: src {:#x} and dst {:#x} must be {} B aligned",
        command.src_addr,
        command.dst_addr,
        command_alignment);
    const uint64_t distance =
        command.src_addr < command.dst_addr ? command.dst_addr - command.src_addr : command.src_addr - command.dst_addr;
    TT_FATAL(
        (kind == abi::CMD_SINGLE && command.src_noc_xy_addr != command.dst_noc_xy_addr) || distance >= command.size,
        "drisc_service: src {:#x} and dst {:#x} overlap over {} B",
        command.src_addr,
        command.dst_addr,
        command.size);
    if (kind == abi::CMD_LOCAL) {
        TT_FATAL(
            command.size % abi::LOCAL_SIZE_ALIGN == 0,
            "drisc_service: CMD_LOCAL size {} is not a multiple of {}",
            command.size,
            abi::LOCAL_SIZE_ALIGN);
        TT_FATAL(
            command.size <= bank_size && command.src_addr <= bank_size - command.size &&
                command.dst_addr <= bank_size - command.size,
            "drisc_service: CMD_LOCAL src {:#x} or dst {:#x} + {} B runs past the {:#x} B bank",
            command.src_addr,
            command.dst_addr,
            command.size,
            bank_size);
    }
}

template <typename Done>
bool wait_until(std::chrono::milliseconds timeout, Done done) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!done()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

std::string elf_path(const std::string& root, uint32_t noc) {
    return fmt::format("{}runtime/hw/lib/blackhole/drisc_service_noc{}.elf", root, noc);
}

ll_api::ElfFile load_service_image(const std::string& path) {
    ll_api::ElfFile image;
    image.ReadImage(path);
    uint64_t build_vaddr = 0;
    const std::span<std::byte> bytes = image.GetSectionContents(abi::BUILD_BLOCK_SECTION, build_vaddr);
    TT_FATAL(
        bytes.size() >= sizeof(abi::KernelBuildBlock),
        "drisc_service: {} has no {} section",
        path,
        abi::BUILD_BLOCK_SECTION);
    abi::KernelBuildBlock build{};
    std::memcpy(&build, bytes.data(), sizeof(build));
    TT_FATAL(build.magic == abi::BUILD_BLOCK_MAGIC, "drisc_service: {} has an unstamped build block", path);
    TT_FATAL(
        build.abi_version == abi::ABI_VERSION,
        "drisc_service: {} was built for ABI version {}, host is {}",
        path,
        build.abi_version,
        abi::ABI_VERSION);
    return image;
}

}  // namespace

// 2 MiB windows on other chips that the seats of one chip write into over PCIe, each mapped into that chip's IOMMU
// domain. KMD unmaps only when the chip's device closes, so windows are freed only with the client.
class PeerWindows {
public:
    PeerWindows(tt::umd::Cluster& cluster, ChipId chip) :
        cluster_(cluster), chip_(chip), pcie_noc_xy_addr_(pcie_noc_xy(cluster, chip)) {}
    ~PeerWindows() {
        if (leak_) {
            for (Window& window : windows_) {
                (void)window.io.release();
            }
        }
    }
    PeerWindows(const PeerWindows&) = delete;
    PeerWindows& operator=(const PeerWindows&) = delete;
    PeerWindows(PeerWindows&&) = delete;
    PeerWindows& operator=(PeerWindows&&) = delete;

    abi::NocEndpoint endpoint(ChipId peer, const tt::umd::CoreCoord& core, uint64_t addr, uint64_t size);

    // True when [addr, addr + size) on this chip's PCIe tile touches a peer window.
    bool overlaps(uint32_t noc_xy_addr, uint64_t addr, uint64_t size) const {
        if (noc_xy_addr != pcie_noc_xy_addr_) {
            return false;
        }
        for (const Window& window : windows_) {
            if (addr < window.dma_addr + window_size && window.dma_addr < addr + size) {
                return true;
            }
        }
        return false;
    }

    // A seat whose slot was never freed may still write into the windows: keep them allocated.
    void leak() { leak_ = true; }

private:
    // A 4 GiB window would need MAP_PEER_BAR's 32-bit offset and length to reach past BAR4's first 4 GiB, and no
    // mapping that size fits below 4 GiB of IOVA space, the only addresses under the PCIe tile's direct limit.
    static constexpr uint64_t window_size = tt::umd::blackhole::TLB_2M_SIZE;
    struct Window {
        ChipId peer;
        tt::umd::CoreCoord core;
        uint64_t base;      // first address of `core` the window reaches, 2 MiB aligned
        uint64_t dma_addr;  // the address this chip's PCIe tile names for `base`
        std::unique_ptr<tt::umd::IoWindow> io;
    };
    tt::umd::Cluster& cluster_;
    ChipId chip_;
    uint32_t pcie_noc_xy_addr_;
    std::vector<Window> windows_;
    bool leak_ = false;
};

abi::NocEndpoint PeerWindows::endpoint(ChipId peer, const tt::umd::CoreCoord& core, uint64_t addr, uint64_t size) {
    TT_FATAL(peer != chip_, "drisc_service: chip {} cannot be its own peer", peer);
    const uint64_t base = addr - addr % window_size;
    TT_FATAL(
        size <= window_size && addr - base <= window_size - size,
        "drisc_service: chip {} {} [{:#x}, +{:#x}) crosses a 2 MiB block; a peer range must fit one 2 MiB window",
        peer,
        core.str(),
        addr,
        size);
    for (const Window& window : windows_) {
        if (window.peer == peer && window.core == core && window.base == base) {
            return {window.dma_addr + (addr - base), pcie_noc_xy_addr_};
        }
    }
    std::unique_ptr<tt::umd::IoWindow> io = cluster_.create_io_window(
        peer,
        core,
        base,
        tt::umd::HostIoWindowConfig{tt::umd::HostMemoryCaching::WC, window_size},
        tt::umd::IoOrdering::Posted,
        std::nullopt,
        tt::umd::WindowFlags::UnicastWrite);
    // KMD places the 2 MiB TLBs at the start of BAR0, each at its index times its size.
    const int tlb_id = dynamic_cast<tt::umd::TlbWindow&>(*io).handle_ref().get_tlb_id();
    const uint64_t dma_addr = cluster_.get_tt_device(chip_)->get_pci_device()->map_peer_bar(
        *cluster_.get_tt_device(peer)->get_pci_device(),
        0,
        static_cast<uint32_t>(tlb_id * window_size),
        static_cast<uint32_t>(window_size));
    // Linux maps below 4 GiB until that fails once for a device, then hands it 64-bit addresses until it is probed
    // again: KMD sets a 64-bit DMA mask.
    TT_FATAL(
        dma_addr + window_size <= pcie_direct_window_end,
        "drisc_service: chip {}'s window on chip {} maps to {:#x}, past the {:#x} its PCIe tile reaches without an "
        "iATU region, which MAP_PEER_BAR cannot set up; the chip's IOMMU now hands out 64-bit addresses until it is "
        "probed again",
        chip_,
        peer,
        dma_addr,
        pcie_direct_window_end);
    windows_.push_back(Window{peer, core, base, dma_addr, std::move(io)});
    return {dma_addr + (addr - base), pcie_noc_xy_addr_};
}

std::unique_ptr<tt::umd::SysmemBuffer> allocate_host_buffer(
    tt::umd::Cluster& cluster, ChipId chip, size_t size, uint64_t& device_addr) {
    auto* sysmem = cluster.get_chip(chip)->get_sysmem_manager();
    TT_FATAL(sysmem != nullptr, "drisc_service: chip {} has no sysmem manager", chip);
    auto buffer = sysmem->allocate_sysmem_buffer(size, false);
    if (buffer->get_iova() + size > pcie_direct_window_end) {
        buffer.reset();
        buffer = sysmem->allocate_sysmem_buffer(size, true);
        TT_FATAL(
            buffer->get_noc_address().has_value(),
            "drisc_service: host buffer on chip {} needs an iATU mapping and got none",
            chip);
    }
    device_addr = buffer->get_noc_address().value_or(buffer->get_iova());
    std::memset(buffer->get_va(), 0, size);
    return buffer;
}

std::array<Seat, 2> bank_seats(tt::umd::Cluster& cluster, ChipId chip, const metal_SocDescriptor& soc, uint32_t bank) {
    TT_FATAL(
        bank < soc.get_num_dram_views(),
        "drisc_service: bank {} is out of range; chip {} has {} DRAM banks",
        bank,
        chip,
        soc.get_num_dram_views());
    const std::vector<CoreCoord> logical = dram_sender_logical_cores(soc, bank, chip);
    const uint64_t asic_id = cluster.get_cluster_description()->get_chip_unique_ids().at(chip);
    TT_FATAL(
        soc.get_grid_size(tt::CoreType::DRAM).y == abi::BANK_ENDPOINTS,
        "drisc_service: chip {} has {} endpoints per DRAM bank, not {}",
        chip,
        soc.get_grid_size(tt::CoreType::DRAM).y,
        abi::BANK_ENDPOINTS);
    const int channel = static_cast<int>(soc.get_channel_for_dram_view(static_cast<int>(bank)));
    std::array<uint16_t, abi::BANK_ENDPOINTS> bank_noc_xy_addr{};
    for (uint32_t sub = 0; sub < bank_noc_xy_addr.size(); sub++) {
        const tt::umd::CoreCoord c =
            soc.get_dram_core_for_channel(channel, static_cast<int>(sub), tt::CoordSystem::TRANSLATED);
        bank_noc_xy_addr[sub] = static_cast<uint16_t>(abi::noc_xy(c.x, c.y));
    }
    std::array<Seat, 2> seats{};
    for (uint32_t noc = 0; noc < seats.size(); noc++) {
        const CoreCoord translated = soc.get_physical_dram_core_from_logical(logical.at(1 - noc));
        const tt::umd::CoreCoord core{translated.x, translated.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED};
        const tt::umd::CoreCoord noc0 = soc.translate_coord_to(core, tt::CoordSystem::NOC0);
        const uint8_t mask = soc.get_dram_endpoint_noc_mask(translated);
        TT_FATAL(
            ((mask >> noc) & 1u) == 0,
            "drisc_service: chip {} bank {} NoC{} seat {} must keep that NIU in NOC2AXI mode (mask {:#x})",
            chip,
            bank,
            noc,
            core.str(),
            mask);
        seats[noc] = Seat{
            .chip = chip,
            .bank = bank,
            .noc = noc,
            .translated = core,
            .id =
                {.asic_id = asic_id, .noc0_x = static_cast<uint32_t>(noc0.x), .noc0_y = static_cast<uint32_t>(noc0.y)},
            .noc2axi_mask = mask,
            .bank_size = soc.dram_view_size,
            .bank_noc_xy_addr = bank_noc_xy_addr};
    }
    return seats;
}

const std::string& runtime_root() {
    static const std::string root = tt::llrt::RunTimeOptions().get_root_dir();
    return root;
}

metal_SocDescriptor soc_descriptor(tt::umd::Cluster& cluster, ChipId chip, const std::string& root) {
    tt::umd::SocDescriptor soc = cluster.get_soc_descriptor(chip);
    TT_FATAL(soc.arch == tt::ARCH::BLACKHOLE, "drisc_service: chip {} is not Blackhole", chip);
    soc.device_descriptor_file_path = root + "tt_metal/soc_descriptors/blackhole_140_arch.yaml";
    return metal_SocDescriptor(soc, tt::BoardType::UNKNOWN, read_mrisc_noc2axi_ports(cluster.get_tt_device(chip)));
}

SeatClient::SeatClient(
    tt::umd::Cluster& cluster, const Seat& seat, const ll_api::ElfFile& image, IssuePath issue_path) :
    seat_(seat), core_(cluster, seat.chip, seat.translated), issue_path_(issue_path) {
    control_page_buffer_ = allocate_host_buffer(cluster, seat.chip, abi::CONTROL_PAGE_SIZE, control_page_device_addr_);
    batch_buffer_ = allocate_host_buffer(cluster, seat.chip, abi::BATCH_REGION_SIZE, batch_device_addr_);
    control_page_ = static_cast<abi::ControlPage*>(control_page_buffer_->get_va());
    batch_slots_ = static_cast<abi::KernelCommand*>(batch_buffer_->get_va());
    pcie_noc_xy_addr_ = pcie_noc_xy(cluster, seat.chip);

    SeatGate gate(seat.id, gate_timeout);
    lock_ = ClientSeatLock::try_acquire(seat.id);
    TT_FATAL(
        lock_.has_value(),
        "drisc_service: chip {} bank {} NoC{} seat {} is held exclusively (tt-metal DRAM programmable cores such as "
        "the tensor prefetcher, or tt-d-gen's own kernel)",
        seat.chip,
        seat.bank,
        seat.noc,
        seat.translated.str());
    gen_ = gate.next_gen(abi::SLOT_GEN_MASK);

    switch (classify()) {
        case Kernel::Stopped: launch(image, 0); break;
        case Kernel::Running: join(image); break;
        case Kernel::Hung:
            TT_FATAL(
                lock_->sole(),
                "drisc_service: the kernel on chip {} DRISC {} is hung or another build, and serves other clients: {}",
                seat.chip,
                seat.translated.str(),
                describe_state());
            // A core that never ran the service (fresh boot, or metal used it last) is not worth a warning.
            if (abi::is_service_state(state())) {
                log_warning(
                    tt::LogMetal,
                    "drisc_service: chip {} DRISC {} has no client and a hung kernel; relaunching: {}",
                    seat.chip,
                    seat.translated.str(),
                    describe_state());
            }
            launch(image, 0);
            break;
    }
    lock_->share();
}

// Running means the kernel serves slots now, with a build of this ABI. Anything else that is not STOPPED, such as a
// launch that died before the kernel booted, is hung.
SeatClient::Kernel SeatClient::classify() const {
    if (state() == abi::SERVICE_STOPPED) {
        return Kernel::Stopped;
    }
    const uint32_t before = heartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(abi::ALIVE_PROBE_MS));
    const uint32_t now = state();
    if (now == abi::SERVICE_STOPPED) {
        return Kernel::Stopped;
    }
    abi::KernelBuildBlock build{};
    core_.read_l1(abi::L1CommandBlockLayout::build_addr, &build, sizeof(build));
    const bool serving = now == abi::SERVICE_RUNNING || now == abi::SERVICE_STOPPING;
    const bool ours = build.magic == abi::BUILD_BLOCK_MAGIC && build.abi_version == abi::ABI_VERSION;
    return serving && ours && heartbeat() != before ? Kernel::Running : Kernel::Hung;
}

// A slot whose ack is not ATTACH is free: its last client was freed, or its request was never answered.
void SeatClient::join(const ll_api::ElfFile& image) {
    std::optional<uint32_t> slot;
    const bool found = wait_until(std::chrono::milliseconds(abi::ATTACH_TIMEOUT_MS), [&] {
        for (uint32_t s = 0; s < abi::MAX_CLIENTS; s++) {
            if (abi::slot_word_type(core_.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(s))) != abi::SLOT_ATTACH) {
                slot = s;
                return true;
            }
        }
        return false;
    });
    TT_FATAL(
        found,
        "drisc_service: all {} client slots on chip {} DRISC {} are in use",
        abi::MAX_CLIENTS,
        seat_.chip,
        seat_.translated.str());
    request_attach(*slot);
    // The kernel attaches the request or, having missed it in its last scan, stops; then this client launches it.
    bool stopped = false;
    const bool answered = wait_until(std::chrono::milliseconds(abi::ATTACH_TIMEOUT_MS), [&] {
        if (acked()) {
            return true;
        }
        stopped = state() == abi::SERVICE_STOPPED;
        return stopped;
    });
    if (stopped) {
        exit_write_.reset();
        launch(image, *slot);
        return;
    }
    if (!answered) {
        exit_write_.reset();
        TT_THROW(
            "drisc_service: chip {} DRISC {} did not ack slot {} gen {}: {}",
            seat_.chip,
            seat_.translated.str(),
            *slot,
            gen_,
            describe_state());
    }
}

void SeatClient::launch(const ll_api::ElfFile& image, uint32_t slot) {
    core_.force_stop(seat_.noc2axi_mask);
    core_.load(image);
    abi::L1CommandBlock block{};
    block.header.bank_size = seat_.bank_size;
    std::copy(seat_.bank_noc_xy_addr.begin(), seat_.bank_noc_xy_addr.end(), block.header.bank_noc_xy_addr);
    core_.write_l1(abi::L1CommandBlockLayout::base_addr, &block, sizeof(block));
    request_attach(slot);
    core_.write_l1_u32(abi::L1CommandBlockLayout::state_addr, abi::SERVICE_BOOTING);
    core_.start();
    launched_ = true;
    if (!wait_until(std::chrono::milliseconds(abi::BOOT_TIMEOUT_MS), [&] { return acked(); })) {
        const std::string state_text = describe_state();
        exit_write_.reset();
        core_.force_stop(seat_.noc2axi_mask);
        core_.write_l1_u32(abi::L1CommandBlockLayout::state_addr, abi::SERVICE_STOPPED);
        TT_THROW(
            "drisc_service: the kernel launched on chip {} DRISC {} did not ack slot {}: {}",
            seat_.chip,
            seat_.translated.str(),
            slot,
            state_text);
    }
}

// The exit write is armed before the request, so a client that dies from here on is detached by the KMD.
void SeatClient::request_attach(uint32_t slot) {
    slot_ = slot;
    exit_write_.emplace(
        core_, abi::L1CommandBlockLayout::exit_mailbox_addr(slot), abi::slot_word(abi::SLOT_DETACH, gen_));
    const abi::NocEndpoint control_page{control_page_device_addr_, pcie_noc_xy_addr_};
    core_.write_l1(abi::L1CommandBlockLayout::control_page_addr(slot), &control_page, sizeof(control_page));
    core_.write_l1_u32(abi::L1CommandBlockLayout::doorbell_addr(slot), 0);
    tt_driver_atomics::sfence();
    core_.write_l1_u32(abi::L1CommandBlockLayout::request_addr(slot), abi::slot_word(abi::SLOT_ATTACH, gen_));
}

bool SeatClient::acked() const {
    return core_.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(slot_)) == abi::slot_word(abi::SLOT_ATTACH, gen_);
}

SeatClient::~SeatClient() {
    try {
        drain(std::chrono::milliseconds(abi::DETACH_TIMEOUT_MS));
    } catch (const std::exception& e) {
        log_error(tt::LogMetal, "drisc_service: drain at detach failed: {}", e.what());
    }
    exit_write_.reset();
    bool freed = false;
    try {
        freed = wait_until(std::chrono::milliseconds(abi::DETACH_TIMEOUT_MS), [&] { return !acked(); });
        if (!freed) {
            log_error(
                tt::LogMetal,
                "drisc_service: chip {} DRISC {} did not free slot {} gen {}; leaking its host buffers: {}",
                seat_.chip,
                seat_.translated.str(),
                slot_,
                gen_,
                describe_state());
        }
    } catch (const std::exception& e) {
        log_error(tt::LogMetal, "drisc_service: detach failed; leaking its host buffers: {}", e.what());
    }
    if (!freed) {
        // The kernel may still write into them.
        (void)control_page_buffer_.release();
        (void)batch_buffer_.release();
        if (peers_ != nullptr) {
            peers_->leak();
        }
    }
    if (window_ != nullptr) {
        window_.reset();
        const std::lock_guard lock(window_count_mutex);
        window_count[seat_.chip]--;
    }
}

void SeatClient::validate(const abi::KernelCommand& command) const {
    validate_command(command, seat_.bank_size);
    // A peer window's static VC carries writes; a read through it crashes the host (UMD architecture_tlbs.cpp).
    TT_FATAL(
        peers_ == nullptr || !peers_->overlaps(command.src_noc_xy_addr, command.src_addr, command.size),
        "drisc_service: source {:#x} + {} B lies in a peer window, which must never be read through",
        command.src_addr,
        command.size);
}

bool SeatClient::issue(const abi::KernelCommand& command) {
    validate(command);
    return post(command);
}

bool SeatClient::post(const abi::KernelCommand& command) {
    if (in_flight() >= abi::CLIENT_QUEUE_CAPACITY) {
        return false;
    }
    const uint32_t entry_addr = abi::L1CommandBlockLayout::command_queue_addr(slot_) +
                                (issued_ % abi::CLIENT_QUEUE_CAPACITY) * sizeof(abi::KernelCommand);
    const uint32_t doorbell = static_cast<uint32_t>(issued_ + 1);
    if (issue_path_ == IssuePath::Window && !window_tried_) {
        window_tried_ = true;
        const std::lock_guard lock(window_count_mutex);
        uint32_t& count = window_count[seat_.chip];
        if (count < max_windows_per_chip) {
            window_ = core_.map_l1(tt::umd::HostMemoryCaching::WC, abi::L1CommandBlockLayout::end_addr);
            count += window_ != nullptr ? 1 : 0;
        }
    }
    if (window_ != nullptr) {
        // Each write leaves as one PCIe burst. The first sfence puts the entry ahead of the doorbell; the second sends
        // the doorbell now rather than whenever the write-combining buffer drains.
        window_->write_aligned(entry_addr, &command, sizeof(command));
        tt_driver_atomics::sfence();
        window_->write_aligned(abi::L1CommandBlockLayout::doorbell_addr(slot_), &doorbell, sizeof(doorbell));
        tt_driver_atomics::sfence();
    } else {
        core_.write_l1(entry_addr, &command, sizeof(command));
        tt_driver_atomics::sfence();
        core_.write_l1_u32(abi::L1CommandBlockLayout::doorbell_addr(slot_), doorbell);
    }
    issued_++;
    return true;
}

bool SeatClient::issue_batch(std::span<const abi::KernelCommand> commands) {
    TT_FATAL(
        !commands.empty() && commands.size() <= abi::BATCH_MAX_COMMANDS,
        "drisc_service: a batch carries 1 to {} commands, got {}",
        abi::BATCH_MAX_COMMANDS,
        commands.size());
    for (const abi::KernelCommand& command : commands) {
        validate(command);
    }
    if (in_flight() >= abi::CLIENT_QUEUE_CAPACITY) {
        return false;
    }
    const uint32_t entry = issued_ % abi::CLIENT_QUEUE_CAPACITY;
    std::copy(commands.begin(), commands.end(), batch_slots_ + entry * abi::BATCH_MAX_COMMANDS);
    const abi::NocEndpoint batch_slot{batch_device_addr_ + entry * abi::BATCH_SLOT_SIZE, pcie_noc_xy_addr_};
    return post(abi::make_batch_command(batch_slot, static_cast<uint32_t>(commands.size())));
}

uint32_t SeatClient::in_flight() const {
    const uint32_t done = std::atomic_ref(control_page_->done_count).load(std::memory_order_acquire);
    return static_cast<uint32_t>(issued_) - done;
}

void SeatClient::drain(std::chrono::milliseconds timeout) const {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (in_flight() > 0) {
        TT_FATAL(
            std::chrono::steady_clock::now() < deadline,
            "drisc_service: drain timed out on chip {} DRISC {}: {} issued, {} in flight, {}",
            seat_.chip,
            seat_.translated.str(),
            issued_,
            in_flight(),
            describe_state());
    }
}

uint32_t SeatClient::state() const { return core_.read_l1_u32(abi::L1CommandBlockLayout::state_addr); }

uint32_t SeatClient::heartbeat() const { return core_.read_l1_u32(abi::L1CommandBlockLayout::heartbeat_addr); }

abi::NocEndpoint SeatClient::host_endpoint(uint64_t device_addr) const { return {device_addr, pcie_noc_xy_addr_}; }

std::string SeatClient::describe_state() const {
    return fmt::format(
        "state {:#x} heartbeat {} free pull TRIDs {} slot {} request {:#x} ack {:#x} exit {:#x} done_count {}",
        state(),
        heartbeat(),
        core_.read_l1_u32(abi::L1CommandBlockLayout::free_pull_trid_count_addr),
        slot_,
        core_.read_l1_u32(abi::L1CommandBlockLayout::request_addr(slot_)),
        core_.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(slot_)),
        core_.read_l1_u32(abi::L1CommandBlockLayout::exit_mailbox_addr(slot_)),
        std::atomic_ref(control_page_->done_count).load(std::memory_order_acquire));
}

Client::Client(
    tt::umd::Cluster& cluster,
    ChipId chip,
    std::vector<uint32_t> banks,
    const std::string& root,
    IssuePath issue_path) :
    banks_(std::move(banks)), peers_(std::make_unique<PeerWindows>(cluster, chip)) {
    TT_FATAL(!banks_.empty(), "drisc_service: a client needs at least one bank");
    std::sort(banks_.begin(), banks_.end());
    banks_.erase(std::unique(banks_.begin(), banks_.end()), banks_.end());
    const metal_SocDescriptor soc = soc_descriptor(cluster, chip, root);
    const char* disable_precompiled = std::getenv("TT_METAL_DISABLE_PRECOMPILED_FW");
    if (disable_precompiled != nullptr && disable_precompiled[0] == '1') {
        static std::once_flag warned;
        std::call_once(warned, [&] {
            log_warning(
                tt::LogMetal,
                "drisc_service: TT_METAL_DISABLE_PRECOMPILED_FW does not rebuild {}; after editing the service kernel, "
                "rebuild the drisc_service_elf target",
                elf_path(root, 0));
        });
    }
    const std::array<ll_api::ElfFile, 2> images = {
        load_service_image(elf_path(root, 0)), load_service_image(elf_path(root, 1))};
    std::vector<Seat> seats;
    seats.reserve(2 * banks_.size());
    for (const uint32_t bank : banks_) {
        const std::array<Seat, 2> pair = bank_seats(cluster, chip, soc, bank);
        seats.insert(seats.end(), pair.begin(), pair.end());
    }
    seats_.reserve(seats.size());
    for (const Seat& seat : seats) {
        seats_.push_back(std::make_unique<SeatClient>(cluster, seat, images[seat.noc], issue_path));
        seats_.back()->peers_ = peers_.get();
    }
}

Client::~Client() = default;

void Client::drain(std::chrono::milliseconds timeout) const {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (const auto& seat : seats_) {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        seat->drain(std::max(left, std::chrono::milliseconds(0)));
    }
}

abi::NocEndpoint Client::peer_endpoint(ChipId peer, const tt::umd::CoreCoord& core, uint64_t addr, uint64_t size) {
    return peers_->endpoint(peer, core, addr, size);
}

// A read-only window: a static VC shared by reads and writes crashes the host (UMD architecture_tlbs.cpp).
PeerFlag::PeerFlag(tt::umd::Cluster& cluster, ChipId chip, const tt::umd::CoreCoord& core, uint64_t addr) :
    window_(cluster.create_io_window(
        chip,
        core,
        addr,
        tt::umd::HostIoWindowConfig{tt::umd::HostMemoryCaching::UC, sizeof(uint32_t)},
        tt::umd::IoOrdering::Strict,
        std::nullopt,
        tt::umd::WindowFlags::UnicastRead)) {}

PeerFlag::~PeerFlag() = default;

uint32_t PeerFlag::value() const { return window_->read32(0); }

void PeerFlag::wait(uint32_t seq, std::chrono::milliseconds timeout) const {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    uint32_t now = value();
    while (static_cast<int32_t>(now - seq) < 0) {
        TT_FATAL(
            std::chrono::steady_clock::now() < deadline,
            "drisc_service: flag stayed at {:#x}, short of {:#x}, for {} ms",
            now,
            seq,
            timeout.count());
        now = value();
    }
}

SeatClient& Client::seat(uint32_t bank, uint32_t noc) {
    const auto it = std::lower_bound(banks_.begin(), banks_.end(), bank);
    TT_FATAL(it != banks_.end() && *it == bank, "drisc_service: bank {} is not one of this client's banks", bank);
    TT_FATAL(noc < 2, "drisc_service: NoC {} is not 0 or 1", noc);
    return *seats_[2 * (it - banks_.begin()) + noc];
}

}  // namespace tt::tt_metal::drisc_service
