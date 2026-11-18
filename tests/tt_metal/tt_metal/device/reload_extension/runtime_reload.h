// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Metal's own BRISC firmware extension, built only by test_firmware_extension.cpp. It walks
// reload_test_table_t using nothing but internal::fw_ext, so a Metal change that breaks the extension
// contract breaks this test.

#pragma once

#include <cstddef>
#include <cstdint>

#include "reload_test_table.h"

// Per-chip barrier. Every core counts itself in on the coordinator; the coordinator multicasts `verdict`
// as the release, and the others return the verdict it sent.
inline uint32_t reload_test_barrier(
    volatile tt_l1_ptr reload_test_table_t* t, uint32_t table_addr, uint32_t noc, uint32_t verdict) {
    namespace fw = internal::fw_ext;
    const uint32_t arrive = table_addr + offsetof(reload_test_table_t, arrive);
    const uint32_t release = table_addr + offsetof(reload_test_table_t, release);
    fw::atomic_inc(noc, fw::noc_xy_addr(noc, t->coord_x, t->coord_y, arrive), 1);
    if (t->is_coord) {
        fw::wait_local_ge(arrive, t->num_cores);
        t->arrive = 0;
        t->release = verdict;
        if (t->mcast_dests != 0) {
            const fw::Rect dst{
                static_cast<uint8_t>(t->mcast_start_x),
                static_cast<uint8_t>(t->mcast_start_y),
                static_cast<uint8_t>(t->mcast_end_x),
                static_cast<uint8_t>(t->mcast_end_y)};
            fw::write_multicast(noc, release, dst, release, sizeof(uint32_t), t->mcast_dests);
        }
        return verdict;
    }
    verdict = fw::wait_local_ge(release, 1);
    t->release = 0;
    return verdict;
}

inline bool reload_next_stage(internal::fw_ext::Launch& launch, uint32_t& stage, uint32_t& round) {
    namespace fw = internal::fw_ext;
    const uint32_t table_addr = launch.extension_table();
    if (table_addr == 0) {
        return false;
    }
    auto* t = reinterpret_cast<volatile tt_l1_ptr reload_test_table_t*>(table_addr);
    const uint32_t noc = launch.noc();
    t->phase = RELOAD_TEST_PHASE_ENTERED;
    fw::quiesce_core_noc();
    t->phase = RELOAD_TEST_PHASE_QUIESCED;

    const bool at_boundary = stage + 1 == t->num_images;
    // Only the coordinator's verdict is used, so only it reads the host's stop word.
    bool stop = false;
    if (at_boundary) {
        stop = t->persistent ? t->is_coord && fw::load_fresh(table_addr + offsetof(reload_test_table_t, stop)) != 0
                             : round + 1 == t->num_rounds;
    }
    const uint32_t verdict = stop ? RELOAD_TEST_STOP : RELOAD_TEST_ADVANCE;
    const uint32_t decided = reload_test_barrier(t, table_addr, noc, verdict);
    t->phase = RELOAD_TEST_PHASE_BARRIER1;
    if (decided == RELOAD_TEST_STOP) {
        fw::flush();
        t->phase = RELOAD_TEST_PHASE_STOPPED;
        return false;
    }
    const uint32_t next = at_boundary ? 0 : stage + 1;

    volatile tt_l1_ptr reload_test_image_t& image = t->images[next];
    fw::read(noc, fw::noc_xy_addr(noc, t->src_x, t->src_y, image.block_src), launch.kernel_config_base(), image.block_bytes, 1);
    fw::read(noc, fw::noc_xy_addr(noc, t->src_x, t->src_y, image.config_src), t->scratch_addr, image.config_bytes, 1);
    fw::flush();
    t->phase = RELOAD_TEST_PHASE_FETCHED;
    // After every core's image has landed, so no core's new image writes into a peer still loading.
    reload_test_barrier(t, table_addr, noc, RELOAD_TEST_ADVANCE);
    fw::flush();
    t->phase = RELOAD_TEST_PHASE_BARRIER2;

    if (!launch.install(t->scratch_addr, image.config_bytes)) {
        return false;
    }
    t->installs = t->installs + 1;
    t->phase = RELOAD_TEST_PHASE_INSTALLED;
    if (at_boundary) {
        round++;
    }
    stage = next;
    return true;
}
