// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/drisc_service/service.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <sys/mman.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/chip_helpers/sysmem_buffer.hpp>
#include <tt-umd/cluster_descriptor.hpp>
#include <tt-umd/driver_atomics.hpp>
#include <tt-umd/io_window/io_window.hpp>
#include <tt-umd-workload/cluster.hpp>

#include "impl/buffers/dram_sender_topology.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_elffile.hpp"

namespace tt::tt_metal::internal::drisc_service {

namespace {

// The kernel's bounce slots are 64 B aligned and the NoC cannot shift bytes across lanes, so DRAM and PCIe sources
// must be too. Destinations are held to the same rule.
constexpr uint64_t command_alignment = 64;

// The Blackhole PCIe tile passes NoC addresses below this to PCIe unchanged (KMD's noc_dma_limit).
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

void UnmapPages::operator()(void* va) const noexcept { ::munmap(va, size); }

namespace {

std::unique_ptr<void, UnmapPages> map_host_pages(size_t size) {
    void* va = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TT_FATAL(va != MAP_FAILED, "drisc_service: mapping {} bytes of host memory failed: {}", size, std::strerror(errno));
    return std::unique_ptr<void, UnmapPages>(va, UnmapPages{size});
}

uint64_t pin_host_pages(ExitWrite& exit_write, void* va, size_t size) {
    const uint64_t iova = exit_write.pin(va, size);
    TT_FATAL(
        iova + size <= pcie_direct_window_end,
        "drisc_service: host pages got IOVA {:#x}, past the {:#x} the PCIe tile passes through",
        iova,
        pcie_direct_window_end);
    return iova;
}

}  // namespace

std::unique_ptr<tt::umd::SysmemBuffer> allocate_host_buffer(
    tt::umd::Cluster& cluster, ChipId chip, size_t size, uint64_t& device_addr) {
    auto* sysmem = cluster.get_chip(chip)->get_sysmem_manager();
    TT_FATAL(sysmem != nullptr, "drisc_service: chip {} has no sysmem manager", chip);
    // Never mapped to the NoC: KMD hands out iATU regions from address 0 up, and tt-metal requires its own at 0.
    auto buffer = sysmem->allocate_sysmem_buffer(size, false);
    TT_FATAL(
        buffer->get_iova() + size <= pcie_direct_window_end,
        "drisc_service: host buffer on chip {} got IOVA {:#x}, past the {:#x} its PCIe tile passes through. KMD 2.9 "
        "hands out 64-bit IOVAs once a chip's low 4 GiB runs out (tt-kmd#268): use KMD 2.10, or dma_address_bits=58",
        chip,
        buffer->get_iova(),
        pcie_direct_window_end);
    device_addr = buffer->get_iova();
    std::memset(buffer->get_va(), 0, size);
    return buffer;
}

namespace {

uint16_t noc_xy_addr(const tt::umd::CoreCoord& core) { return static_cast<uint16_t>(abi::noc_xy(core.x, core.y)); }

// Every NoC endpoint of DRAM view `bank`, TRANSLATED.
std::array<uint16_t, abi::BANK_ENDPOINTS> bank_endpoints(const metal_SocDescriptor& soc, ChipId chip, uint32_t bank) {
    TT_FATAL(
        soc.get_grid_size(tt::CoreType::DRAM).y == abi::BANK_ENDPOINTS,
        "drisc_service: chip {} has {} endpoints per DRAM bank, not {}",
        chip,
        soc.get_grid_size(tt::CoreType::DRAM).y,
        abi::BANK_ENDPOINTS);
    const int channel = static_cast<int>(soc.get_channel_for_dram_view(static_cast<int>(bank)));
    std::array<uint16_t, abi::BANK_ENDPOINTS> endpoints{};
    for (uint32_t sub = 0; sub < endpoints.size(); sub++) {
        endpoints[sub] =
            noc_xy_addr(soc.get_dram_core_for_channel(channel, static_cast<int>(sub), tt::CoordSystem::TRANSLATED));
    }
    return endpoints;
}

// The chip's DRAM cores whose NIU for `noc` is in stream mode, TRANSLATED.
std::vector<uint16_t> stream_endpoints(const metal_SocDescriptor& soc, uint32_t noc) {
    std::vector<uint16_t> endpoints;
    for (uint32_t view = 0; view < soc.get_num_dram_views(); view++) {
        const int channel = static_cast<int>(soc.get_channel_for_dram_view(static_cast<int>(view)));
        for (uint32_t sub = 0; sub < abi::BANK_ENDPOINTS; sub++) {
            const tt::umd::CoreCoord c =
                soc.get_dram_core_for_channel(channel, static_cast<int>(sub), tt::CoordSystem::TRANSLATED);
            if (((soc.get_dram_endpoint_noc_mask(CoreCoord(c.x, c.y)) >> noc) & 1u) == 0) {
                endpoints.push_back(noc_xy_addr(c));
            }
        }
    }
    return endpoints;
}

// `bank_fields` placed on the DRISC at `logical`, driving `noc`. Throws when DRISC firmware keeps that NoC's NIU in
// NOC2AXI mode: only a stream-mode NIU lets the DRISC start NoC transfers.
Seat place_seat(const metal_SocDescriptor& soc, Seat bank_fields, uint32_t noc, const CoreCoord& logical) {
    const CoreCoord translated = soc.get_physical_dram_core_from_logical(logical);
    const tt::umd::CoreCoord core{translated.x, translated.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED};
    const uint8_t mask = soc.get_dram_endpoint_noc_mask(translated);
    TT_FATAL(
        ((mask >> noc) & 1u) == 0,
        "drisc_service: chip {} bank {} NoC{} seat {} must keep that NIU in NOC2AXI mode (mask {:#x})",
        bank_fields.chip,
        bank_fields.bank,
        noc,
        core.str(),
        mask);
    const tt::umd::CoreCoord noc0 = soc.translate_coord_to(core, tt::CoordSystem::NOC0);
    Seat seat = bank_fields;
    seat.noc = noc;
    seat.translated = core;
    seat.id.noc0_x = static_cast<uint32_t>(noc0.x);
    seat.id.noc0_y = static_cast<uint32_t>(noc0.y);
    seat.noc2axi_mask = mask;
    const CoreCoord endpoint = soc.get_preferred_worker_core_for_dram_view(static_cast<int>(seat.bank), noc);
    seat.bank_base = {soc.get_address_offset(static_cast<int>(seat.bank)), abi::noc_xy(endpoint.x, endpoint.y)};
    seat.stream_dram_noc_xy_addr = stream_endpoints(soc, noc);
    return seat;
}

}  // namespace

std::array<Seat, NUM_NOCS> bank_seats(
    tt::umd::Cluster& cluster, ChipId chip, const metal_SocDescriptor& soc, uint32_t bank) {
    TT_FATAL(
        bank < soc.get_num_dram_views(),
        "drisc_service: bank {} is out of range; chip {} has {} DRAM banks",
        bank,
        chip,
        soc.get_num_dram_views());
    const std::vector<CoreCoord> logical = dram_sender_logical_cores(soc, bank, chip);
    const Seat bank_fields{
        .chip = chip,
        .bank = bank,
        .id = {.asic_id = cluster.get_cluster_description()->get_chip_unique_ids().at(chip)},
        .bank_size = soc.dram_view_size,
        .bank_noc_xy_addr = bank_endpoints(soc, chip, bank)};
    std::array<Seat, NUM_NOCS> seats{};
    for (uint32_t noc = 0; noc < seats.size(); noc++) {
        seats[noc] = place_seat(soc, bank_fields, noc, logical.at(1 - noc));
    }
    return seats;
}

const tt::llrt::RunTimeOptions& runtime_options() {
    static const tt::llrt::RunTimeOptions options;
    return options;
}

const std::string& runtime_root() { return runtime_options().get_root_dir(); }

metal_SocDescriptor soc_descriptor(tt::umd::Cluster& cluster, ChipId chip) {
    tt::umd::SocDescriptor soc = cluster.get_soc_descriptor(chip);
    TT_FATAL(soc.arch == tt::ARCH::BLACKHOLE, "drisc_service: chip {} is not Blackhole", chip);
    soc.device_descriptor_file_path =
        get_soc_description_file(tt::ARCH::BLACKHOLE, tt::TargetDevice::Silicon, runtime_options());
    return metal_SocDescriptor(soc, tt::BoardType::UNKNOWN, read_mrisc_noc2axi_ports(cluster.get_tt_device(chip)));
}

SeatClientImpl::SeatClientImpl(
    tt::umd::Cluster& cluster, const Seat& seat, const ll_api::ElfFile& image, IssuePath issue_path) :
    seat_(seat), core_(cluster, seat.chip, seat.translated), issue_path_(issue_path) {
    control_page_memory_ = map_host_pages(abi::CONTROL_PAGE_SIZE);
    batch_memory_ = map_host_pages(abi::BATCH_REGION_SIZE);
    control_page_ = static_cast<abi::ControlPage*>(control_page_memory_.get());
    batch_slots_ = static_cast<abi::KernelCommand*>(batch_memory_.get());
    pcie_noc_xy_addr_ = pcie_noc_xy(cluster, seat.chip);
    for (const uint16_t noc_xy_addr : seat.bank_noc_xy_addr) {
        endpoint_flags_[noc_xy_addr] |= own_bank_endpoint;
    }
    for (const uint16_t noc_xy_addr : seat.stream_dram_noc_xy_addr) {
        endpoint_flags_[noc_xy_addr] |= stream_mode_endpoint;
    }

    SeatGate gate(seat.id, gate_timeout);
    lock_ = SeatLock::try_acquire_client(seat.id);
    TT_FATAL(
        lock_.has_value(),
        "drisc_service: chip {} bank {} NoC{} seat {} is held exclusively (tt-metal DRAM programmable cores such as "
        "the tensor prefetcher)",
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
SeatClientImpl::Kernel SeatClientImpl::classify() const {
    if (state() == abi::SERVICE_STOPPED) {
        return Kernel::Stopped;
    }
    const uint32_t before = l1_heartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(abi::ALIVE_PROBE_MS));
    const uint32_t now = state();
    if (now == abi::SERVICE_STOPPED) {
        return Kernel::Stopped;
    }
    abi::KernelBuildBlock build{};
    core_.read_l1(abi::L1CommandBlockLayout::build_addr, &build, sizeof(build));
    const bool serving = now == abi::SERVICE_RUNNING || now == abi::SERVICE_STOPPING;
    const bool ours = build.magic == abi::BUILD_BLOCK_MAGIC && build.abi_version == abi::ABI_VERSION;
    return serving && ours && l1_heartbeat() != before ? Kernel::Running : Kernel::Hung;
}

// A slot whose ack is not ATTACH is free: its last client was freed, or its request was never answered.
void SeatClientImpl::join(const ll_api::ElfFile& image) {
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

void SeatClientImpl::launch(const ll_api::ElfFile& image, uint32_t slot) {
    core_.force_stop();
    core_.load(image);
    abi::L1CommandBlock block{};
    block.header.bank_size = seat_.bank_size;
    std::copy(seat_.bank_noc_xy_addr.begin(), seat_.bank_noc_xy_addr.end(), block.header.bank_noc_xy_addr);
    block.header.noc2axi_niu_mask = seat_.noc2axi_mask;
    core_.write_l1(abi::L1CommandBlockLayout::base_addr, &block, sizeof(block));
    request_attach(slot);
    core_.write_l1_u32(abi::L1CommandBlockLayout::state_addr, abi::SERVICE_BOOTING);
    core_.start();
    launched_ = true;
    if (!wait_until(std::chrono::milliseconds(abi::BOOT_TIMEOUT_MS), [&] { return acked(); })) {
        const std::string state_text = describe_state();
        exit_write_.reset();
        core_.force_stop();
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
void SeatClientImpl::request_attach(uint32_t slot) {
    slot_ = slot;
    exit_write_.emplace(
        core_, abi::L1CommandBlockLayout::exit_mailbox_addr(slot), abi::slot_word(abi::SLOT_DETACH, gen_));
    control_page_device_addr_ = pin_host_pages(*exit_write_, control_page_memory_.get(), abi::CONTROL_PAGE_SIZE);
    batch_device_addr_ = pin_host_pages(*exit_write_, batch_memory_.get(), abi::BATCH_REGION_SIZE);
    const abi::NocEndpoint control_page{control_page_device_addr_, pcie_noc_xy_addr_};
    core_.write_l1(abi::L1CommandBlockLayout::control_page_addr(slot), &control_page, sizeof(control_page));
    core_.write_l1_u32(abi::L1CommandBlockLayout::doorbell_addr(slot), 0);
    tt_driver_atomics::sfence();
    core_.write_l1_u32(abi::L1CommandBlockLayout::request_addr(slot), abi::slot_word(abi::SLOT_ATTACH, gen_));
}

bool SeatClientImpl::acked() const {
    return core_.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(slot_)) == abi::slot_word(abi::SLOT_ATTACH, gen_);
}

SeatClientImpl::~SeatClientImpl() {
    try {
        drain(std::chrono::milliseconds(abi::DETACH_TIMEOUT_MS));
    } catch (const std::exception& e) {
        log_error(tt::LogMetal, "drisc_service: drain at detach failed: {}", e.what());
    }
    bool freed = false;
    // Held until the KMD's own DETACH is sent, so no client takes the freed slot and has its DETACH overwritten.
    std::optional<SeatGate> gate;
    try {
        gate.emplace(seat_.id, gate_timeout);
        // Not through exit_write_: closing its fd unpins the control page and batch, which must wait for FREED.
        core_.write_l1_u32(abi::L1CommandBlockLayout::exit_mailbox_addr(slot_), abi::slot_word(abi::SLOT_DETACH, gen_));
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
    exit_write_.reset();
    gate.reset();
    if (!freed) {
        // The kernel may still write into them.
        (void)control_page_memory_.release();
        (void)batch_memory_.release();
    }
    if (window_ != nullptr) {
        window_.reset();
        const std::lock_guard lock(window_count_mutex);
        window_count[seat_.chip]--;
    }
}

void SeatClientImpl::validate(const abi::KernelCommand& command, bool src_in_own_bank) const {
    validate_command(command, seat_.bank_size);
    if (abi::command_kind(command) != abi::CMD_SINGLE) {
        return;
    }
    // The kernel reads such a source with the GDDR DMA engine, whatever endpoint it names (its dma_readable()).
    const bool src_by_dma = command.size % abi::LOCAL_SIZE_ALIGN == 0 && src_in_own_bank;
    const auto expect_noc2axi = [&](const char* side, uint16_t noc_xy_addr) {
        TT_FATAL(
            (endpoint_flags(noc_xy_addr) & stream_mode_endpoint) == 0,
            "drisc_service: {} coords {:#x} are a DRAM core whose NIU for NoC{} is in stream mode, so the access would "
            "reach that DRISC's L1, not DRAM; name the bank's NoC{} endpoint (get_preferred_worker_core_for_dram_view)",
            side,
            noc_xy_addr,
            seat_.noc,
            seat_.noc);
    };
    if (!src_by_dma) {
        expect_noc2axi("source", command.src_noc_xy_addr);
    }
    expect_noc2axi("destination", command.dst_noc_xy_addr);
}

uint8_t SeatClientImpl::endpoint_flags(uint16_t noc_xy_addr) const {
    return noc_xy_addr < endpoint_flags_.size() ? endpoint_flags_[noc_xy_addr] : 0;
}

bool SeatClientImpl::in_own_bank(uint16_t noc_xy_addr, uint64_t addr, uint32_t size) const {
    return (endpoint_flags(noc_xy_addr) & own_bank_endpoint) != 0 && size <= seat_.bank_size &&
           addr <= seat_.bank_size - size;
}

abi::KernelCommand SeatClientImpl::prepare(const abi::KernelCommand& command) const {
    const bool src_in_own_bank = in_own_bank(command.src_noc_xy_addr, command.src_addr, command.size);
    const bool local = abi::command_kind(command) == abi::CMD_SINGLE && command.size % abi::LOCAL_SIZE_ALIGN == 0 &&
                       src_in_own_bank && in_own_bank(command.dst_noc_xy_addr, command.dst_addr, command.size);
    const abi::KernelCommand sent =
        local ? abi::make_local_command(command.src_addr, command.dst_addr, command.size) : command;
    validate(sent, src_in_own_bank);
    return sent;
}

std::optional<uint64_t> SeatClientImpl::issue(const abi::KernelCommand& command) { return post(prepare(command)); }

std::optional<uint64_t> SeatClientImpl::post(const abi::KernelCommand& command) {
    if (in_flight() >= abi::CLIENT_QUEUE_CAPACITY) {
        return std::nullopt;
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
    return ++issued_;
}

std::optional<uint64_t> SeatClientImpl::issue_batch(std::span<const abi::KernelCommand> commands) {
    TT_FATAL(
        !commands.empty() && commands.size() <= abi::BATCH_MAX_COMMANDS,
        "drisc_service: a batch carries 1 to {} commands, got {}",
        abi::BATCH_MAX_COMMANDS,
        commands.size());
    if (in_flight() >= abi::CLIENT_QUEUE_CAPACITY) {
        return std::nullopt;
    }
    const uint32_t entry = issued_ % abi::CLIENT_QUEUE_CAPACITY;
    // The entry is free, and nothing is posted unless every command passes.
    std::ranges::transform(commands, batch_slots_ + entry * abi::BATCH_MAX_COMMANDS, [this](const auto& command) {
        return prepare(command);
    });
    const abi::NocEndpoint batch_slot{batch_device_addr_ + entry * abi::BATCH_SLOT_SIZE, pcie_noc_xy_addr_};
    return post(abi::make_batch_command(batch_slot, static_cast<uint32_t>(commands.size())));
}

uint32_t SeatClientImpl::in_flight() const { return static_cast<uint32_t>(issued_) - done_count(); }

uint32_t SeatClientImpl::done_count() const {
    return std::atomic_ref(control_page_->done_count).load(std::memory_order_acquire);
}

void SeatClientImpl::drain(std::chrono::milliseconds timeout) const {
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

uint32_t SeatClientImpl::state() const { return core_.read_l1_u32(abi::L1CommandBlockLayout::state_addr); }

uint32_t SeatClientImpl::heartbeat() const {
    return std::atomic_ref(control_page_->heartbeat).load(std::memory_order_relaxed);
}

uint32_t SeatClientImpl::l1_heartbeat() const { return core_.read_l1_u32(abi::L1CommandBlockLayout::heartbeat_addr); }

abi::NocEndpoint SeatClientImpl::host_endpoint(uint64_t device_addr) const { return {device_addr, pcie_noc_xy_addr_}; }

std::string SeatClientImpl::describe_state() const {
    return fmt::format(
        "state {:#x} heartbeat {} free pull TRIDs {} slot {} request {:#x} ack {:#x} exit {:#x} done_count {}",
        state(),
        l1_heartbeat(),
        core_.read_l1_u32(abi::L1CommandBlockLayout::free_pull_trid_count_addr),
        slot_,
        core_.read_l1_u32(abi::L1CommandBlockLayout::request_addr(slot_)),
        core_.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(slot_)),
        core_.read_l1_u32(abi::L1CommandBlockLayout::exit_mailbox_addr(slot_)),
        done_count());
}

SeatClient::SeatClient(std::unique_ptr<SeatClientImpl> impl) : impl_(std::move(impl)) {}
SeatClient::~SeatClient() = default;
bool SeatClient::has_window() const { return impl_->has_window(); }
std::optional<uint64_t> SeatClient::issue(const abi::KernelCommand& command) { return impl_->issue(command); }
std::optional<uint64_t> SeatClient::issue_batch(std::span<const abi::KernelCommand> commands) {
    return impl_->issue_batch(commands);
}
uint32_t SeatClient::in_flight() const { return impl_->in_flight(); }
uint32_t SeatClient::done_count() const { return impl_->done_count(); }
uint64_t SeatClient::completed() const { return impl_->completed(); }
bool SeatClient::done(uint64_t ticket) const { return impl_->done(ticket); }
void SeatClient::drain(std::chrono::milliseconds timeout) const { impl_->drain(timeout); }
uint32_t SeatClient::state() const { return impl_->state(); }
uint32_t SeatClient::heartbeat() const { return impl_->heartbeat(); }
std::string SeatClient::describe_state() const { return impl_->describe_state(); }
abi::NocEndpoint SeatClient::host_endpoint(uint64_t device_addr) const { return impl_->host_endpoint(device_addr); }
abi::NocEndpoint SeatClient::bank_endpoint(uint64_t offset) const { return impl_->bank_endpoint(offset); }
uint32_t SeatClient::slot() const { return impl_->slot(); }
uint32_t SeatClient::gen() const { return impl_->gen(); }
bool SeatClient::launched() const { return impl_->launched(); }

Client::Client(
    tt::umd::Cluster& cluster,
    ChipId chip,
    std::vector<uint32_t> banks,
    const std::string& root,
    IssuePath issue_path) :
    banks_(std::move(banks)) {
    TT_FATAL(!banks_.empty(), "drisc_service: a client needs at least one bank");
    std::sort(banks_.begin(), banks_.end());
    banks_.erase(std::unique(banks_.begin(), banks_.end()), banks_.end());
    const metal_SocDescriptor soc = soc_descriptor(cluster, chip);
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
        const std::array<Seat, NUM_NOCS> pair = bank_seats(cluster, chip, soc, bank);
        seats.insert(seats.end(), pair.begin(), pair.end());
    }
    seats_.reserve(seats.size());
    for (const Seat& seat : seats) {
        seats_.push_back(std::unique_ptr<SeatClient>(
            new SeatClient(std::make_unique<SeatClientImpl>(cluster, seat, images[seat.noc], issue_path))));
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

SeatClient& Client::seat(uint32_t bank, uint32_t noc) {
    const auto it = std::lower_bound(banks_.begin(), banks_.end(), bank);
    TT_FATAL(it != banks_.end() && *it == bank, "drisc_service: bank {} is not one of this client's banks", bank);
    TT_FATAL(noc < 2, "drisc_service: NoC {} is not 0 or 1", noc);
    return *seats_[2 * (it - banks_.begin()) + noc];
}

}  // namespace tt::tt_metal::internal::drisc_service
