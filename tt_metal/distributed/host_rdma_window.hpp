// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// One RMA window over the whole pinned region. Displacements are offsets both sides derive
// from the shared layout, so there is nothing to attach and no address to exchange.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace tt::tt_metal::experimental {

class RdmaWindow {
public:
    // Collective, once per rank, over an already-pinned region that must not be remapped.
    // expect_rank/expect_size are host id and count: host ids ARE the MPI ranks here.
    static std::unique_ptr<RdmaWindow> create(
        uint8_t* region_base, uint64_t region_bytes, uint32_t expect_rank, uint32_t expect_size, std::string& err);
    ~RdmaWindow();

    RdmaWindow(const RdmaWindow&) = delete;
    RdmaWindow& operator=(const RdmaWindow&) = delete;

    // Zero means "no operation outstanding".
    struct Op {
        uint32_t id = 0;
        bool valid() const { return id != 0; }
    };

    // Origin is ordinary memory inside the region; only the target has to be in the window.
    std::string put(const void* src, uint64_t bytes, uint32_t peer_rank, uint64_t target_offset, Op& op);

    // Stages the value, since an origin must outlive the call and a temporary does not.
    // Unused since credits moved to CreditPublisher's coalesced multi-word put.
    std::string put_word(uint64_t value, uint32_t peer_rank, uint64_t target_offset);

    // True once the ORIGIN buffer is reusable, which is what frees the page behind it.
    // Consumes the op when it completes.
    bool test(Op& op);

    // Pushes outstanding operations toward the peer and turns the progress engine.
    std::string flush(uint32_t peer_rank);
    std::string barrier();

    std::string describe() const;

    // True only when every rank passed. Collective, so a rank that already failed locally must
    // still call it: RdmaWindow::create is collective too and would hang the ranks that passed.
    static bool agree(bool local_ok, std::string& err);

    // True only when every rank passed the SAME value. Collective, with agree()'s rule: a rank
    // that skips it strands the peers. For geometry both sides must lay out identically.
    static bool agree_value(uint64_t local, std::string& err);

private:
    RdmaWindow();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
