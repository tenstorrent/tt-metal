// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

// #43563 COLLECTIVE-COUPLING ISOLATION probe. When 1: for core_num==2 (the S3 worker) ONLY, drop
// its last KV chunk so it processes 4 chunks instead of 5. This makes S2 (core_num==1) the lone deep
// worker at a position where, un-hacked, BOTH S2 and S3 are deep (e.g. pos4351). All other cores —
// including S2 — are left byte-identical. If S2's iter1-vs-iter2 divergence disappears, S3's depth
// (the 2nd deep worker) is the causal trigger (collective coupling, position-independent). If S2
// still diverges, the position/total-chunks drives it. 0 = off (kernel unchanged). Defaulted OFF.
#define HACK_S3_SHALLOW_43563 0

// #43563 DECISIVE PER-CORE TEST. When 1: for core_num==1 (the S2 worker) ONLY, drop its last KV
// chunk so it processes one fewer chunk (e.g. 4 instead of 5), leaving every other core byte-identical.
// Mirrors HACK_S3_SHALLOW_43563 but targets S2. If S2's iter1-vs-iter2 divergence disappears, the
// trigger is S2's OWN accumulation depth (its 5th FULL chunk) — per-core, not collective. 0 = off.
#define HACK_S2_SHALLOW_43563 0

inline uint32_t nearest_n(uint32_t x, uint32_t n) { return ((x + n - 1) / n) * n; }

// Given a global current position and this device's SP parameters, determine:
//   - skip_attention: device has no data yet (global pos < this device's first slot)
//   - skip_kv_cache_update: device has data but doesn't own the new write slot
//   - local_cur_pos: the device-local position for KV cache / MLA
//
// Tokens are assigned in device_chunk_size groups round-robin across SP devices:
//   [0, dcs) -> dev 0, [dcs, 2*dcs) -> dev 1, ...
// The owning device is (cur_pos / dcs) % num_sp_devices.
// local_cur_pos = local_seq_len for the owning device (write slot),
//              = local_seq_len - 1 for non-owning devices (last valid entry).
inline std::tuple<bool, bool, uint32_t> get_device_mla_work_assignment(
    uint32_t cur_pos, uint32_t sp_device_idx, uint32_t device_chunk_size, uint32_t num_sp_devices) {
    if (cur_pos < sp_device_idx * device_chunk_size) {
        return {true, true, 0};
    }

    uint32_t sp_block = device_chunk_size * num_sp_devices;
    uint32_t num_full_blocks = cur_pos / sp_block;
    uint32_t remainder = cur_pos % sp_block;

    uint32_t dev_start = sp_device_idx * device_chunk_size;
    uint32_t dev_contrib = (remainder > dev_start) ? remainder - dev_start : 0;
    if (dev_contrib > device_chunk_size) {
        dev_contrib = device_chunk_size;
    }
    uint32_t local_seq_len = num_full_blocks * device_chunk_size + dev_contrib;

    uint32_t owning_device = (cur_pos / device_chunk_size) % num_sp_devices;
    bool skip_kv_cache_update = (sp_device_idx != owning_device);
    uint32_t local_cur_pos = skip_kv_cache_update ? local_seq_len - 1 : local_seq_len;

    return {false, skip_kv_cache_update, local_cur_pos};
}

inline std::tuple<uint32_t, uint32_t, uint32_t> get_runtime_args(
    int cur_pos, int cur_batch, int core_num, int num_cores_per_batch, uint32_t k_chunk_size) {
    // No sliding window: process from beginning up to cur_pos
    uint32_t valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);
    uint32_t num_chunks_value = valid_seq_len / k_chunk_size;

    uint32_t k_chunk_start = 0;
    uint32_t k_chunk_end = 0;

    // Strided chunk distribution: core N gets chunks N, N+num_cores, N+2*num_cores, ...
    // This ensures each core reads from the same DRAM bank (round-robin sharding)
    // E.g., with 8 cores and 16 chunks:
    //   core 0 gets chunks 0, 8 (both on bank 1)
    //   core 1 gets chunks 1, 9 (both on bank 3)
    //   etc.
    if (num_cores_per_batch > int(num_chunks_value)) {
        // More cores than chunks: each active core gets 1 chunk
        int chunks_per_core = (core_num < int(num_chunks_value)) ? 1 : 0;
        k_chunk_start = core_num;
        k_chunk_end = k_chunk_start + chunks_per_core;
    } else {
        // More chunks than cores: strided distribution
        // k_chunk_start = core_num (first chunk index)
        // k_chunk_end = first chunk + (num_chunks - 1) * stride + 1
        // Kernel iterates: for (k = start; k < end; k += stride)
        // where stride = num_cores_per_batch
        int chunks_per_core = num_chunks_value / num_cores_per_batch;
        int residuals = num_chunks_value % num_cores_per_batch;
        int num_chunks_for_core = chunks_per_core + (core_num < residuals ? 1 : 0);

        k_chunk_start = core_num;
        k_chunk_end =
            k_chunk_start + (num_chunks_for_core > 0 ? (num_chunks_for_core - 1) * num_cores_per_batch + 1 : 0);

#if defined(HACK_S3_SHALLOW_43563) && HACK_S3_SHALLOW_43563
        // #43563 probe: force ONLY core_num==2 (S3) to do one fewer chunk (drop its last chunk),
        // leaving every other core (incl. core_num==1 / S2) untouched. Only fires in the strided
        // (more-chunks-than-cores) regime and only when S3 actually has >1 chunk.
        if (core_num == 2 && num_chunks_for_core > 1) {
            k_chunk_end = k_chunk_start + (num_chunks_for_core - 2) * num_cores_per_batch + 1;
        }
#endif

#if defined(HACK_S2_SHALLOW_43563) && HACK_S2_SHALLOW_43563
        // #43563 decisive per-core test: force ONLY core_num==1 (S2) to do one fewer chunk (drop its
        // last chunk), leaving every other core (incl. core_num==0 / S1 and core_num==2 / S3) untouched.
        // Only fires in the strided (more-chunks-than-cores) regime and only when S2 has >1 chunk.
        if (core_num == 1 && num_chunks_for_core > 1) {
            k_chunk_end = k_chunk_start + (num_chunks_for_core - 2) * num_cores_per_batch + 1;
        }
#endif
    }

    return {num_chunks_value, k_chunk_start, k_chunk_end};
}
