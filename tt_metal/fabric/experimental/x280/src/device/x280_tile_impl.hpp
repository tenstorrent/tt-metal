// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Private implementation of one X280 L2CPU tile.
//
// TileImpl owns tile-specific hardware semantics. RuntimeImpl owns one TileImpl
// per physical tile and therefore outlives these non-owning runtime and chip
// back-pointers.

#pragma once

#include "x280/device/x280_tile.hpp"
#include "x280_chip_impl.hpp"
#include "x280_pinned_input_record.hpp"
#include "x280_runtime_impl.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace x280::detail {

enum class SessionAttachOutcome {
    Attached,
    Busy,
    Unhealthy,
};

class TileImpl {
public:
    TileImpl(RuntimeImpl& runtime, ChipImpl& chip, TileIdentity identity);

    const TileIdentity& id() const { return identity_; }

    TileState inspect() const;
    TileState inspect_state() const;
    BootResult boot(const BootRequest& request);
    LimRange reserved_lim() const;
    LimRange available_lim() const;
    std::vector<uint8_t> read_available_lim(uint64_t address, size_t size) const;
    void write_available_lim(uint64_t address, std::span<const uint8_t> data);
    /**
     * Programs host-block window @p index onto @p target, resolved for the NoC
     * window.noc1 selects, and returns what the entry read back as. Refuses a
     * firmware-owned index or an input encode() refuses, without writing.
     */
    NocWindowResult program_noc_window(uint32_t index, const TranslatedCoord& target, const NocWindow2M& window);
    NocWindowResult read_noc_window(uint32_t index) const;

    // Latched false by a restoration that could not confirm resident idle, or
    // by a submission that failed after clearing the per-hart records. Never
    // cleared in software.
    bool healthy() const;
    std::optional<std::array<HartState, kHartsPerTile>> failure_hartstates() const;

    PinnedInputState pinned_input_state() const;
    PinnedInputPrepareResult prepare_pinned_input(const PinnedInputPrepareRequest& request);

    // Runtime initialization owns sequencing and recovery reporting. These
    // perform the one-tile body under the chip lock the caller holds.
    void prime_lim_ecc_unlocked(bool& wayenable_committed);
    void set_lim_shape_unlocked(uint32_t target_wayenable, bool& commit_attempted);

    // Cold-boot stages called under the chip lock: classify every tile,
    // prepare every selected tile, release all resets, then confirm each.

    // Return true only when the tile is observable, primed, reset-held, and
    // ready to prepare. Otherwise fill `result`; live idle reports
    // AlreadyRunning.
    bool classify_cold_boot_unlocked(BootResult& result) const;

    // Prepares and verifies the image, vectors, and protocol without releasing
    // reset. Returns the image size.
    size_t prepare_cold_boot_unlocked(const std::string& idle_image_path);

    // Fills `result` and returns whether resident idle was confirmed within
    // `timeout`. Reset stays released on failure.
    bool confirm_resident_idle_unlocked(std::chrono::milliseconds timeout, BootResult& result);

private:
    friend class ActiveSessionImpl;
    friend class TileTestAccess;
    friend class RuntimeImpl;

    struct LimEccObservation {
        uint32_t raw_wayenable = 0;
        uint64_t raw_marker = 0;
        uint64_t raw_extent = 0;
        LimEccState state = LimEccState::Invalid;
        std::optional<uint64_t> primed_bytes;
    };

    X280MetalTransport& transport() { return chip_->transport(); }
    const X280MetalTransport& transport() const { return chip_->transport(); }
    tt::umd::CoreCoord core() const { return chip_->core_for(id().hw_index); }

    static std::vector<uint8_t> read_image_file(const std::string& path, const char* what, const char* build_hint);

    // Active-protocol traffic. Each of these eight reaches the transport and
    // acquires no lock itself: the caller must hold ChipImpl::lock() across the
    // call, as the _unlocked suffix marks elsewhere in this class. Holding it
    // across the sequence is the point — UMD makes each register operation
    // atomic, but nothing else keeps a clear/publish/JUMP sequence from
    // interleaving with a concurrent STOP/poll/RETURN_ACK on the same tile. It
    // does not serialise against Metal's own threads.
    std::array<HartState, kHartsPerTile> snapshot_active_run() const;
    std::array<uint64_t, kHartsPerTile> read_hart_phases() const;
    std::array<uint32_t, kHartsPerTile> read_hart_statuses() const;
    void clear_active_run_state();
    void publish_active_run_entry(uint64_t run_value, uint64_t entry_addr, uint64_t args_addr, uint64_t args_bytes);
    void publish_active_command(uint32_t hart, uint64_t command);
    void publish_active_command_all(uint64_t command);
    // primed_bytes is the tile's LimEccObservation::primed_bytes; the image is
    // refused past the window it bounds.
    void load_active_image(const std::vector<uint8_t>& image, std::optional<uint64_t> primed_bytes);

    // Session slot state. These take session_mutex_ themselves and need no
    // chip lock, which is what lets the enqueue pre-check run before the chip
    // is serialised.
    //
    // attach_session is the reservation authority for the one execution slot
    // this tile has. The runtime mints the id; this only commits or refuses it.
    SessionAttachOutcome attach_session(SessionId id);
    bool complete_session(SessionId id, bool restored, const std::array<HartState, kHartsPerTile>& hartstates);
    bool session_active() const;

    LimEccObservation read_lim_ecc_unlocked() const;
    LimRange available_lim_unlocked(const TileState& state) const;

    // Caller must hold ChipImpl::lock(). Reads all three records for one hart —
    // control, state, trap — so command comes from the host-owned side while
    // the rest is firmware-published.
    HartState read_hart_state_unlocked(uint32_t hart) const;

    // Caller must hold ChipImpl::lock(). One composite snapshot of all four
    // harts: holding the lock across the whole sweep is what stops the four
    // records being read from different points in the protocol.
    std::array<HartState, kHartsPerTile> read_hart_states_unlocked() const;

    void load_idle_image_unlocked(const std::vector<uint8_t>& image);
    void set_reset_vectors_unlocked(uint64_t entry_addr);

    // Caller must hold ChipImpl::lock(). Zeroes all twelve protocol cache
    // lines, which only a cold boot may do: resident idle publishes its four
    // IDLE records once on entry, so clearing them on a live tile leaves
    // nothing that will republish them.
    void zero_hart_protocol_unlocked();

    // Caller must hold ChipImpl::lock(). The pinned-input record in the tile's
    // general-purpose scratch, one 32-bit register access per word; the write
    // covers words [first, end).
    PinnedInputRecord read_pinned_input_record_unlocked() const;
    void write_pinned_input_words_unlocked(const PinnedInputRecord& words, size_t first, size_t end);

    RuntimeImpl* runtime_;
    ChipImpl* chip_;
    TileIdentity identity_;
    mutable std::mutex session_mutex_;
    bool session_active_ = false;
    SessionId active_session_id_;
    bool healthy_ = true;
    std::optional<std::array<HartState, kHartsPerTile>> failure_hartstates_;
    // Serialises pinned_input_state and prepare_pinned_input; taken before the
    // chip lock.
    mutable std::mutex pinned_input_mutex_;
};

// Device tests need to create reversible hardware states that the semantic
// public API intentionally cannot express. This private adapter uses the same
// TileImpl, chip lock, NOC coordinates, and transport as production code.
class TileTestAccess {
public:
    explicit TileTestAccess(X280Tile& tile);

    uint64_t read_u64(uint64_t addr) const;
    void write_u64(uint64_t addr, uint64_t value);
    std::vector<uint8_t> read_block(uint64_t addr, size_t size) const;
    void write_block(uint64_t addr, const std::vector<uint8_t>& data);
    NocWindowWords read_noc_window_words(uint32_t index) const;
    void write_noc_window_words(uint32_t index, const NocWindowWords& words);
    void load_idle_image(const std::vector<uint8_t>& image);
    void load_active_image(const std::vector<uint8_t>& image);
    uint32_t read_l2cpu_reset_raw() const;

private:
    TileImpl* tile_;
};

}  // namespace x280::detail
