// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused SigLIP attention sub-block kernel.
//
// Eventual target: LN1 → QKV → 16× SDPA-head → O-proj → residual in one
// TRISC dispatch on a 128-core BH chip grid (16 × 8 = 128 with 2 spare).
//
// State (task #10):
//   Commit 1: LN1 + residual on the 8-core LN1 row, math: out = LN1(x) + x.
//   Commit 2 (this file): expand grid to LN1 ∪ QKV = 38 distinct cores.
//                         NCRISC implements an 8→36 receiver-pull mcast of
//                         LN1's output (ln_out_cb) into the 36 QKV receivers'
//                         qkv_act_cb. No QKV matmul yet — Commit 3.
//   Commit 3: QKV EncoderMatmul::Op on the 36-core receiver grid.

#include "../../unified_kernels/ln.h"
#include "../../unified_kernels/matmul.h"
#include "../../unified_kernels/residual_add.h"
#include "../../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
// mcast.hpp left included as it advertises deepseek_b1_ops::Mcast::Op for any
// future sender-push variant. The receiver-pull design used here doesn't
// instantiate Mcast::Op — it uses the bare noc_semaphore_inc + noc_async_read
// primitives directly. The header still must compile in our context, so the
// include is load-bearing as a plumbing check.
#include "../../../../../demos/deepseek_v3_b1/unified_kernels/mcast.hpp"

// =============================================================================
// Role gating (runtime)
// -----------------------------------------------------------------------------
// The kernel runs on the union LN1 ∪ QKV = 38 distinct cores. Each phase only
// participates on a sub-grid; we use runtime if-gates on get_relative_logical_*
// rather than compile-time per-range gating because the union has only a few
// roles (ln1_only=2, both=6, qkv_only=30) and runtime branching keeps the host
// descriptor to a single kernel over the union. Compile-time per-range gating
// can be reintroduced as a perf pass once the math is locked in.
//
// Roles (x = logical_x, y = logical_y):
//   is_ln1_core:  y == 0 && x < ln1_num_cores                          (8 cores)
//   is_qkv_core:  y < qkv_grid_y && x < qkv_grid_x                     (36 cores)
//
// "Both" cores (y=0, x=0..5) participate in LN1 AND QKV — they sender-incr
// every receiver including themselves, and as receivers they noc_async_read
// their own L1 over the NoC loopback. That's expected; NoC loopback reads are
// well-defined.
// =============================================================================

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    // LN1 inputs (live on LN1 row only).
    constexpr uint32_t ln_in_cb = get_named_compile_time_arg_val("ln_in_cb");
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("gamma_cb");
    constexpr uint32_t beta_cb = get_named_compile_time_arg_val("beta_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t ones_cb = get_named_compile_time_arg_val("ones_cb");
    // Residual second input + LN1 chaining + final residual output (LN1 row only).
    constexpr uint32_t x_residual_cb = get_named_compile_time_arg_val("x_residual_cb");
    constexpr uint32_t final_out_cb = get_named_compile_time_arg_val("final_out_cb");
    constexpr uint32_t ln_out_cb = get_named_compile_time_arg_val("ln_out_cb");
    // QKV receiver-pull destination CB (QKV grid only).
    constexpr uint32_t qkv_act_cb = get_named_compile_time_arg_val("qkv_act_cb");
    // 1-tile sync CB on the LN1 row: TRISC pushes after LN1's cb_push_back of
    // ln_out_cb; NCRISC sender waits + pops before issuing 36 atomic-incs.
    // Avoids a multi-consumer race on ln_out_cb (residual + sender would both
    // poll pages_received - pages_acked, and residual's pop would starve the
    // sender's wait_front).
    constexpr uint32_t ln_done_trigger_cb = get_named_compile_time_arg_val("ln_done_trigger_cb");

    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");
    constexpr uint32_t gamma_tiles = get_named_compile_time_arg_val("gamma_tiles");
    constexpr uint32_t ln1_num_cores = get_named_compile_time_arg_val("ln1_num_cores");
    constexpr uint32_t qkv_grid_x = get_named_compile_time_arg_val("qkv_grid_x");
    constexpr uint32_t qkv_grid_y = get_named_compile_time_arg_val("qkv_grid_y");
    constexpr uint32_t qkv_act_tiles_per_core = get_named_compile_time_arg_val("qkv_act_tiles_per_core");
    constexpr uint32_t counter_sem_id = get_named_compile_time_arg_val("counter_sem_id");
    // SDPA grid bounds — task #11 Commit 2 plumbing. No body yet; the role
    // flag is computed below but no SDPA work runs on it. Compute commits
    // (#11 commits 3+) will add NCRISC reader for per-head Q/K/V plus TRISC
    // QK^T / softmax / Attn@V Op-struct calls under is_sdpa_core_*.
    constexpr uint32_t sdpa_grid_x = get_named_compile_time_arg_val("sdpa_grid_x");
    constexpr uint32_t sdpa_grid_y = get_named_compile_time_arg_val("sdpa_grid_y");
    constexpr uint32_t sdpa_y_offset = get_named_compile_time_arg_val("sdpa_y_offset");
    constexpr uint32_t sdpa_x_offset = get_named_compile_time_arg_val("sdpa_x_offset");
    // Commit 3: QKV matmul weight CB lives on the 36-core QKV grid. Pre-loaded
    // bfp8 weight; NCRISC marks all 108 tiles pushed so TRISC matmul's
    // cb_wait_front returns immediately. The matmul Op-struct doesn't pop the
    // weight CB (weights stay L1-resident).
    constexpr uint32_t qkv_w_cb = get_named_compile_time_arg_val("qkv_w_cb");
    constexpr uint32_t qkv_weight_tiles = get_named_compile_time_arg_val("qkv_weight_tiles");
    // #11 Commit 3: per-head Q delivery from QKV → SDPA workers.
    constexpr uint32_t qkv_done_trigger_cb = get_named_compile_time_arg_val("qkv_done_trigger_cb");
    constexpr uint32_t sdpa_q_cb = get_named_compile_time_arg_val("sdpa_q_cb");
    constexpr uint32_t sdpa_qkv_ready_sem_id = get_named_compile_time_arg_val("sdpa_qkv_ready_sem_id");
    constexpr uint32_t num_qkv_signals_per_worker = get_named_compile_time_arg_val("num_qkv_signals_per_worker");
    constexpr uint32_t num_heads = get_named_compile_time_arg_val("num_heads");
    constexpr uint32_t num_sdpa_workers_per_head = get_named_compile_time_arg_val("num_sdpa_workers_per_head");
    constexpr uint32_t m_tiles_per_sdpa_worker = get_named_compile_time_arg_val("m_tiles_per_sdpa_worker");
    constexpr uint32_t sdpa_q_tiles_per_worker = get_named_compile_time_arg_val("sdpa_q_tiles_per_worker");
    // #11 Commit 6: streaming K row 0 (3 tiles) → first QK^T tile probe.
    // NCRISC streams 3 K tiles (the head_dim_padded=96 cols at m=0) into
    // sdpa_k_partial_cb, then TRISC matmuls Q[0,:] × K[0,:]^T into the
    // 1-tile sdpa_qk_probe_cb. The same noc_async_read shape extends to
    // other K rows in future commits.
    constexpr uint32_t sdpa_k_partial_cb = get_named_compile_time_arg_val("sdpa_k_partial_cb");
    constexpr uint32_t sdpa_qk_probe_cb = get_named_compile_time_arg_val("sdpa_qk_probe_cb");
    constexpr uint32_t sdpa_k_partial_tiles = get_named_compile_time_arg_val("sdpa_k_partial_tiles");

    const bool is_ln1_core_nc = (get_relative_logical_y() == 0) && (get_relative_logical_x() < ln1_num_cores);
    const bool is_qkv_core_nc = (get_relative_logical_y() < qkv_grid_y) && (get_relative_logical_x() < qkv_grid_x);
    // SDPA role flag — grid at logical (x_offset..x_offset+grid_x-1,
    // y_offset..y_offset+grid_y-1) = (8..11, 0..7). Disjoint from LN1+QKV
    // (which live in x=0..7), so SDPA cores have the full per-core L1 to
    // themselves for Q + K + V CBs.
    const uint32_t my_logical_y_nc = get_relative_logical_y();
    const uint32_t my_logical_x_nc = get_relative_logical_x();
    const bool is_sdpa_core_nc = (my_logical_y_nc >= sdpa_y_offset) &&
                                 (my_logical_y_nc < sdpa_y_offset + sdpa_grid_y) &&
                                 (my_logical_x_nc >= sdpa_x_offset) && (my_logical_x_nc < sdpa_x_offset + sdpa_grid_x);

    // setup_sharded_buffer pre-pushes the LN1 row's input CBs so TRISC's
    // cb_wait_front returns immediately. These CBs only exist on the LN1 row;
    // calling setup on a core that lacks the CB is undefined, so gate on role.
    if (is_ln1_core_nc) {
        unified_kernels::setup_sharded_buffer(ln_in_cb, in_tiles);
        unified_kernels::setup_sharded_buffer(gamma_cb, gamma_tiles);
        unified_kernels::setup_sharded_buffer(beta_cb, gamma_tiles);
        unified_kernels::setup_sharded_buffer(scaler_cb, 1);
        unified_kernels::setup_sharded_buffer(ones_cb, 1);
        unified_kernels::setup_sharded_buffer(x_residual_cb, in_tiles);
        unified_kernels::setup_sharded_buffer(final_out_cb, in_tiles);
    }

    // QKV weight is pre-loaded; mark all 108 tiles pushed on every QKV core so
    // the TRISC matmul's cb_wait_front returns immediately. Skipped on LN1-only
    // cores (no qkv_w_cb allocated there).
    if (is_qkv_core_nc) {
        unified_kernels::setup_sharded_buffer(qkv_w_cb, qkv_weight_tiles);
    }

    // =========================================================================
    // SENDER (LN1 cores): after TRISC has pushed ln_out_cb, unicast an atomic
    // +1 to every QKV receiver's counter semaphore. Do NOT pop ln_out_cb —
    // receivers will read from it via noc_async_read.
    //
    // 8 senders × 36 receivers = 288 tiny atomic NoC writes. The atomic-inc
    // path is the same pattern used by deepseek_b1_ops::Gather (NCRISC sender,
    // noc_semaphore_inc per receiver). Note: BH does not support posted atomics
    // (HW bug), so we leave the default non-posted variant.
    // =========================================================================
    if (is_ln1_core_nc) {
        // Wait for TRISC LN1 to signal "ln_out_cb pushed" via the 1-tile sync
        // CB. Using ln_done_trigger_cb (single-consumer) instead of ln_out_cb
        // (residual's consumer) keeps NCRISC sender immune to residual's pop.
        cb_wait_front(ln_done_trigger_cb, 1);
        cb_pop_front(ln_done_trigger_cb, 1);

        const uint32_t counter_sem_l1_addr = get_semaphore(counter_sem_id);
        const uint32_t phys_origin_x = get_common_arg_val<uint32_t>(1);  // worker_phys_origin_x
        const uint32_t phys_origin_y = get_common_arg_val<uint32_t>(2);  // worker_phys_origin_y
        // 8 senders × 36 receivers = 288 atomic-incs. Pass physical NoC coords
        // (host-supplied origin + logical) — BH atomic-inc responses don't
        // route through the HW logical→physical translation table for atomics
        // the way data writes do, so the kernel must address physical coords
        // directly (matches deepseek's gather op host-side pattern).
        for (uint32_t rx_y = 0; rx_y < qkv_grid_y; ++rx_y) {
            for (uint32_t rx_x = 0; rx_x < qkv_grid_x; ++rx_x) {
                const uint64_t rx_sem_noc_addr =
                    get_noc_addr(phys_origin_x + rx_x, phys_origin_y + rx_y, counter_sem_l1_addr);
                noc_semaphore_inc(rx_sem_noc_addr, 1);
            }
        }
        // Barrier ensures all 36 incs have left this core before we proceed —
        // important since the same core (if "both" role) is about to wait on
        // its own counter as a receiver in the block below.
        noc_async_atomic_barrier();
    }

    // =========================================================================
    // RECEIVER (QKV cores): wait for all 8 senders, then noc_async_read each
    // LN1 shard into the 8 contiguous slots of qkv_act_cb. Sharded persistent
    // buffer ⇒ every LN1 core's ln_out_cb shard sits at the same L1 address,
    // passed in as the NCRISC common RT arg `ln_out_l1_addr` (index 0).
    // =========================================================================
    if (is_qkv_core_nc) {
        const uint32_t counter_sem_l1_addr = get_semaphore(counter_sem_id);
        volatile tt_l1_ptr uint32_t* counter_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_sem_l1_addr);

        cb_reserve_back(qkv_act_cb, qkv_act_tiles_per_core);
        noc_semaphore_wait_min(counter_sem_ptr, ln1_num_cores);
        // Reset for the next iteration (kernel reentry). Non-atomic reset is
        // safe here: all senders have already incremented and won't touch this
        // sem again within this dispatch.
        noc_semaphore_set(counter_sem_ptr, 0);

        const uint32_t ln_out_l1_addr = get_common_arg_val<uint32_t>(0);  // ln_out_l1_addr
        // LN1 row sits at logical y=0; physical y is the worker grid origin's y
        // (the y dimension stays contiguous for the cores we care about).
        const uint32_t ln1_phys_y = get_common_arg_val<uint32_t>(2);  // worker_phys_origin_y
        // shard_bytes = in_tiles × bytes-per-tile. qkv_act_cb's page size is
        // exactly one tile, so this matches the producer's contribution.
        const uint32_t shard_bytes = in_tiles * get_tile_size(qkv_act_cb);
        const uint32_t qkv_act_write_base = get_write_ptr(qkv_act_cb);
        // LN1 row logical x=0..7 maps to a non-contiguous physical x range on
        // BH (eth/PCIe columns split the worker grid: logical x=7 → physical
        // x=10, not 8). Pull the 8 physical x coords from the host-supplied
        // array (named "ln1_phys_x") rather than computing origin + offset.
        // Array slot 0 sits at common-arg-index 4 (after the 4 scalar named
        // common args: ln_out_l1_addr, worker_phys_origin_x,
        // worker_phys_origin_y, qkv_out_l1_addr).
        constexpr uint32_t LN1_PHYS_X_BASE = 4;
        for (uint32_t i = 0; i < ln1_num_cores; ++i) {
            const uint32_t ln1_phys_x = get_common_arg_val<uint32_t>(LN1_PHYS_X_BASE + i);
            const uint64_t src = get_noc_addr(ln1_phys_x, ln1_phys_y, ln_out_l1_addr);
            const uint32_t dst = qkv_act_write_base + i * shard_bytes;
            noc_async_read(src, dst, shard_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(qkv_act_cb, qkv_act_tiles_per_core);
    }

    // =========================================================================
    // PHASE 4a (Commit 3 of #11): QKV → SDPA per-head fan-out.
    //
    // After TRISC pushes qkv_done_trigger_cb (right after EncoderMatmul writes
    // qkv_out_cb), QKV NCRISC determines its head h and which 4 SDPA workers
    // need that head's slice, then atomic-incs each worker's sdpa_qkv_ready_sem.
    //
    // QKV shard layout (Commit 1 of #11): linear shard index s = y*qkv_grid_x + x.
    //   s ∈ [0..15]: Q heads 0..15
    //   s ∈ [16..31]: K heads 0..15
    //   s ∈ [32..47]: V heads 0..15
    // For this commit only Q producers signal (num_qkv_signals_per_worker=1);
    // K and V producers will be added in subsequent commits.
    //
    // SDPA worker mapping (Commit 2 of #11):
    //   head h → workers at logical (head_col = h%sdpa_grid_x,
    //                                head_row = h/sdpa_grid_x,
    //                                y = head_row*num_sdpa_workers_per_head + w)
    //   for w ∈ [0..num_sdpa_workers_per_head-1]. Note sdpa_grid_x is the head
    //   count per head-row (8) — same numerical value as the SDPA grid x, by
    //   construction.
    // =========================================================================
    if (is_qkv_core_nc) {
        cb_wait_front(qkv_done_trigger_cb, 1);
        cb_pop_front(qkv_done_trigger_cb, 1);

        const uint32_t my_qkv_shard_idx = get_relative_logical_y() * qkv_grid_x + get_relative_logical_x();
        // All 3 producers signal (Q, K, V). SDPA workers wait for sem ≥ 3
        // before any reads. Linear QKV-shard layout:
        //   [0..num_heads-1]:             Q heads
        //   [num_heads..2*num_heads-1]:   K heads
        //   [2*num_heads..3*num_heads-1]: V heads
        if (my_qkv_shard_idx < 3 * num_heads) {
            const uint32_t head_idx = my_qkv_shard_idx % num_heads;
            const uint32_t head_col = head_idx % sdpa_grid_x;  // x within SDPA (0..3)
            const uint32_t head_row = head_idx / sdpa_grid_x;  // y/num_workers within SDPA (0..3)
            // Workers of head h are at logical
            //   (x = sdpa_x_offset + head_col,
            //    y = sdpa_y_offset + head_row * num_workers_per_head + w).
            const uint32_t sdpa_y_base = sdpa_y_offset + head_row * num_sdpa_workers_per_head;

            // sdpa_phys_x carries the SDPA grid's physical NoC x coords (one
            // per head_col), sitting after ln1_phys_x in the common RT arg
            // slots. With sdpa_grid_x=4 (16 heads × 2 workers/head, #11
            // Commit 4) the array has 4 entries; ln1_phys_x still has 8.
            //   slot  0: ln_out_l1_addr
            //   slot  1: worker_phys_origin_x
            //   slot  2: worker_phys_origin_y
            //   slot  3: qkv_out_l1_addr
            //   slots 4..11:  ln1_phys_x[0..7]
            //   slots 12..15: sdpa_phys_x[0..3]
            constexpr uint32_t SDPA_PHYS_X_BASE = 4 + 8;
            const uint32_t sdpa_worker_phys_x = get_common_arg_val<uint32_t>(SDPA_PHYS_X_BASE + head_col);
            const uint32_t phys_origin_y_local = get_common_arg_val<uint32_t>(2);
            const uint32_t sem_l1_addr = get_semaphore(sdpa_qkv_ready_sem_id);
            for (uint32_t w = 0; w < num_sdpa_workers_per_head; ++w) {
                const uint32_t sdpa_worker_phys_y = phys_origin_y_local + (sdpa_y_base + w);
                const uint64_t sem_noc_addr = get_noc_addr(sdpa_worker_phys_x, sdpa_worker_phys_y, sem_l1_addr);
                noc_semaphore_inc(sem_noc_addr, 1);
            }
            noc_async_atomic_barrier();
        }
    }

    // =========================================================================
    // PHASE 4b (#11 Commit 4): SDPA NCRISC reader — Q-only this commit.
    // Q: M-parallel slice (rows_per_worker, head_dim_padded) from the head's
    // QKV source core's qkv_out_cb at byte offset = worker_idx * Q-bytes.
    //
    // K and V deliveries deferred to a follow-up commit (global L1 budget
    // currently maxed; needs LN1 intermediate CB reuse or bfp8 packing).
    // =========================================================================
    if (is_sdpa_core_nc) {
        // Translate this core's logical (x, y) back to (head_idx, worker_idx).
        const uint32_t head_col_self = get_relative_logical_x() - sdpa_x_offset;
        const uint32_t sdpa_local_y = get_relative_logical_y() - sdpa_y_offset;
        const uint32_t head_row_self = sdpa_local_y / num_sdpa_workers_per_head;
        const uint32_t worker_idx_self = sdpa_local_y % num_sdpa_workers_per_head;
        const uint32_t head_idx_self = head_row_self * sdpa_grid_x + head_col_self;

        const uint32_t sdpa_sem_l1_addr = get_semaphore(sdpa_qkv_ready_sem_id);
        volatile tt_l1_ptr uint32_t* sdpa_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sdpa_sem_l1_addr);

        cb_reserve_back(sdpa_q_cb, sdpa_q_tiles_per_worker);
        noc_semaphore_wait_min(sdpa_sem_ptr, num_qkv_signals_per_worker);
        noc_semaphore_set(sdpa_sem_ptr, 0);

        const uint32_t phys_origin_x_local = get_common_arg_val<uint32_t>(1);
        const uint32_t phys_origin_y_local = get_common_arg_val<uint32_t>(2);
        const uint32_t qkv_out_l1_addr = get_common_arg_val<uint32_t>(3);
        const uint32_t tile_size_bytes = get_tile_size(sdpa_q_cb);

        const uint32_t q_shard_idx = head_idx_self;  // shards [0..num_heads-1]
        const uint32_t q_src_phys_x = phys_origin_x_local + (q_shard_idx % qkv_grid_x);
        const uint32_t q_src_phys_y = phys_origin_y_local + (q_shard_idx / qkv_grid_x);
        const uint32_t q_bytes_per_worker = sdpa_q_tiles_per_worker * tile_size_bytes;
        const uint32_t q_byte_offset = worker_idx_self * q_bytes_per_worker;
        const uint64_t q_src_noc_addr = get_noc_addr(q_src_phys_x, q_src_phys_y, qkv_out_l1_addr + q_byte_offset);
        noc_async_read(q_src_noc_addr, get_write_ptr(sdpa_q_cb), q_bytes_per_worker);

        // #11 Commit 6: stream K row 0 (3 head_dim tiles = 6 KB) from the
        // head's K source core into sdpa_k_partial_cb. Sized to cover the
        // tiles needed for the first QK^T output tile; the same call shape
        // will be iterated for other K rows in future commits.
        cb_reserve_back(sdpa_k_partial_cb, sdpa_k_partial_tiles);
        const uint32_t k_shard_idx = num_heads + head_idx_self;  // shards [16..31]
        const uint32_t k_src_phys_x = phys_origin_x_local + (k_shard_idx % qkv_grid_x);
        const uint32_t k_src_phys_y = phys_origin_y_local + (k_shard_idx / qkv_grid_x);
        const uint64_t k_src_noc_addr = get_noc_addr(k_src_phys_x, k_src_phys_y, qkv_out_l1_addr);
        const uint32_t k_partial_bytes = sdpa_k_partial_tiles * tile_size_bytes;
        noc_async_read(k_src_noc_addr, get_write_ptr(sdpa_k_partial_cb), k_partial_bytes);

        noc_async_read_barrier();
        cb_push_back(sdpa_q_cb, sdpa_q_tiles_per_worker);
        cb_push_back(sdpa_k_partial_cb, sdpa_k_partial_tiles);
    }
#endif  // COMPILE_FOR_NCRISC

#if defined(COMPILE_FOR_BRISC)
    // no-op: all data is sharded L1; no DRAM streaming in this fused path.
#endif

#if defined(COMPILE_FOR_TRISC)
    constexpr uint32_t ln_in_cb = get_named_compile_time_arg_val("ln_in_cb");
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("gamma_cb");
    constexpr uint32_t beta_cb = get_named_compile_time_arg_val("beta_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t ones_cb = get_named_compile_time_arg_val("ones_cb");
    constexpr uint32_t accum_cb = get_named_compile_time_arg_val("accum_cb");
    constexpr uint32_t xmm_cb = get_named_compile_time_arg_val("xmm_cb");
    constexpr uint32_t xmm2_cb = get_named_compile_time_arg_val("xmm2_cb");
    constexpr uint32_t mean_cb = get_named_compile_time_arg_val("mean_cb");
    constexpr uint32_t var_cb = get_named_compile_time_arg_val("var_cb");
    constexpr uint32_t ivar_cb = get_named_compile_time_arg_val("ivar_cb");
    // ln_out_cb is the chaining buffer: LN1 writes it, residual reads it as a_cb.
    constexpr uint32_t ln_out_cb = get_named_compile_time_arg_val("ln_out_cb");
    constexpr uint32_t x_residual_cb = get_named_compile_time_arg_val("x_residual_cb");
    constexpr uint32_t final_out_cb = get_named_compile_time_arg_val("final_out_cb");
    // 1-tile sync CB pushed at the end of LN1 to release the NCRISC sender —
    // see NCRISC's matching wait_front/pop above.
    constexpr uint32_t ln_done_trigger_cb = get_named_compile_time_arg_val("ln_done_trigger_cb");
    // QKV matmul CBs + shape params (Commit 3). The matmul Op-struct waits on
    // qkv_act_cb (288 tiles, pushed by NCRISC receiver) and qkv_w_cb (108
    // tiles, pre-loaded), and writes 24 tiles to qkv_out_cb.
    constexpr uint32_t qkv_act_cb_tr = get_named_compile_time_arg_val("qkv_act_cb");
    constexpr uint32_t qkv_w_cb_tr = get_named_compile_time_arg_val("qkv_w_cb");
    constexpr uint32_t qkv_out_cb_tr = get_named_compile_time_arg_val("qkv_out_cb");
    constexpr uint32_t qkv_m_tiles = get_named_compile_time_arg_val("qkv_m_tiles");
    constexpr uint32_t qkv_k_tiles = get_named_compile_time_arg_val("qkv_k_tiles");
    constexpr uint32_t qkv_n_tiles_per_core = get_named_compile_time_arg_val("qkv_n_tiles_per_core");
    constexpr uint32_t qkv_act_total_tiles = get_named_compile_time_arg_val("qkv_act_total_tiles");
    constexpr uint32_t qkv_weight_tiles_tr = get_named_compile_time_arg_val("qkv_weight_tiles");

    constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");
    constexpr uint32_t eps_bits = get_named_compile_time_arg_val("eps_bits");

    // Runtime role flags. Same logic as NCRISC; named with _tr suffix to keep
    // is-ln1-core unambiguous if the NCRISC and TRISC bodies are ever read
    // side-by-side.
    // TRISC role gating: keep `ln1_num_cores` and `qkv_grid_{x,y}` as CT-args
    // pulled from the same named-arg list NCRISC uses (lifted into TRISC at the
    // top of this function), so the role bounds stay consistent if the host
    // descriptor changes them. Hardcoded `8` / `6` previously made this body
    // silently drift from NCRISC's role check when the QKV grid expanded.
    constexpr uint32_t ln1_num_cores_tr = get_named_compile_time_arg_val("ln1_num_cores");
    constexpr uint32_t qkv_grid_x_tr = get_named_compile_time_arg_val("qkv_grid_x");
    constexpr uint32_t qkv_grid_y_tr = get_named_compile_time_arg_val("qkv_grid_y");
    constexpr uint32_t sdpa_grid_x_tr = get_named_compile_time_arg_val("sdpa_grid_x");
    constexpr uint32_t sdpa_grid_y_tr = get_named_compile_time_arg_val("sdpa_grid_y");
    constexpr uint32_t sdpa_y_offset_tr = get_named_compile_time_arg_val("sdpa_y_offset");
    constexpr uint32_t sdpa_x_offset_tr = get_named_compile_time_arg_val("sdpa_x_offset");
    // #11 Commit 6: TRISC SDPA worker computes first QK^T tile per worker.
    constexpr uint32_t sdpa_q_cb_tr = get_named_compile_time_arg_val("sdpa_q_cb");
    constexpr uint32_t sdpa_k_partial_cb_tr = get_named_compile_time_arg_val("sdpa_k_partial_cb");
    constexpr uint32_t sdpa_qk_probe_cb_tr = get_named_compile_time_arg_val("sdpa_qk_probe_cb");
    constexpr uint32_t sdpa_q_tiles_per_worker_tr = get_named_compile_time_arg_val("sdpa_q_tiles_per_worker");
    constexpr uint32_t sdpa_k_partial_tiles_tr = get_named_compile_time_arg_val("sdpa_k_partial_tiles");
    const bool is_ln1_core_tr = (get_relative_logical_y() == 0) && (get_relative_logical_x() < ln1_num_cores_tr);
    const bool is_residual_core_tr = is_ln1_core_tr;
    const bool is_qkv_core_tr =
        (get_relative_logical_y() < qkv_grid_y_tr) && (get_relative_logical_x() < qkv_grid_x_tr);
    // SDPA role flag — grid at logical (sdpa_x_offset..+grid_x-1,
    // sdpa_y_offset..+grid_y-1). Disjoint from LN1∪QKV; compute commits
    // (#11 commits 5+) gate QK^T/softmax/Attn@V calls on it.
    const uint32_t my_logical_y_tr = get_relative_logical_y();
    const uint32_t my_logical_x_tr = get_relative_logical_x();
    const bool is_sdpa_core_tr =
        (my_logical_y_tr >= sdpa_y_offset_tr) && (my_logical_y_tr < sdpa_y_offset_tr + sdpa_grid_y_tr) &&
        (my_logical_x_tr >= sdpa_x_offset_tr) && (my_logical_x_tr < sdpa_x_offset_tr + sdpa_grid_x_tr);

    // =========================================================================
    // PHASE 1: LN1 — y = ((x - mean) / sqrt(var + eps)) * gamma + beta
    //          Reads: ln_in_cb, gamma_cb, beta_cb, scaler_cb, ones_cb
    //          Writes: ln_out_cb (consumed by Phase 2 + NCRISC sender)
    // =========================================================================
    if (is_ln1_core_tr) {
        using LNCTArgs = pi05_siglip_ops::LayerNorm::ComputeCTArgs<
            ln_in_cb,
            gamma_cb,
            beta_cb,
            scaler_cb,
            ones_cb,
            accum_cb,
            xmm_cb,
            xmm2_cb,
            mean_cb,
            var_cb,
            ivar_cb,
            ln_out_cb,
            d_tiles,
            in_tiles,
            eps_bits>;

        pi05_siglip_ops::LayerNorm::Op<LNCTArgs, true> ln1;
        pi05_siglip_ops::LayerNorm::RTArgs ln_args{};
        ln1(ln_args);

        // Release NCRISC sender. Single-tile sync push must happen AFTER LN1's
        // final cb_push_back(ln_out_cb) so the L1 data is committed before the
        // sender starts unicasting +1s to QKV receivers (receivers then
        // noc_async_read this L1 region).
        cb_reserve_back(ln_done_trigger_cb, 1);
        cb_push_back(ln_done_trigger_cb, 1);
    }

    // =========================================================================
    // PHASE 2: residual — out = ln_out + x_residual
    //          Reads: ln_out_cb (from Phase 1), x_residual_cb (separate L1 copy of x)
    //          Writes: final_out_cb
    // =========================================================================
    if (is_residual_core_tr) {
        using ResCTArgs = pi05_siglip_ops::ResidualAdd::ComputeCTArgs<ln_out_cb, x_residual_cb, final_out_cb, in_tiles>;

        pi05_siglip_ops::ResidualAdd::Op<ResCTArgs, true> residual;
        pi05_siglip_ops::ResidualAdd::RTArgs res_args{};
        residual(res_args);
    }

    // =========================================================================
    // PHASE 3 (Commit 3): QKV matmul on the 36-core receiver grid.
    //   Activation: qkv_act_cb (8×36 = 288 tiles, full (256, 1152) per core,
    //               produced by NCRISC receiver-pull).
    //   Weight:     qkv_w_cb   (36×3 = 108 tiles, (1152, 96) per core, bfp8,
    //               pre-loaded; NCRISC ran setup_sharded_buffer on the 36-core
    //               grid to mark all tiles pushed).
    //   Output:     qkv_out_cb (8×3 = 24 tiles, (256, 96) per core, bf16). Left
    //               for #11 SDPA to consume; Commit 3 only validates the math.
    //
    // The matmul Op-struct pops act_cb after the matmul and leaves weight_cb
    // L1-resident — both matching the standalone qkv_matmul kernel.
    // =========================================================================
    if (is_qkv_core_tr) {
        using QkvMmCTArgs = pi05_siglip_ops::EncoderMatmul::ComputeCTArgs<
            qkv_act_cb_tr,
            qkv_w_cb_tr,
            qkv_out_cb_tr,
            qkv_m_tiles,
            qkv_k_tiles,
            qkv_n_tiles_per_core,
            qkv_act_total_tiles,
            qkv_weight_tiles_tr>;

        pi05_siglip_ops::EncoderMatmul::Op<QkvMmCTArgs, true> qkv_mm;
        pi05_siglip_ops::EncoderMatmul::RTArgs qkv_args{};
        qkv_mm(qkv_args);

        // #11 Commit 3: release the QKV→SDPA fan-out atomic-inc. The matmul
        // Op-struct ended with cb_push_back(qkv_out_cb), so the (M, 96) head
        // slice is now in L1; we just need to tell NCRISC to fire its incs.
        constexpr uint32_t qkv_done_trigger_cb_tr = get_named_compile_time_arg_val("qkv_done_trigger_cb");
        cb_reserve_back(qkv_done_trigger_cb_tr, 1);
        cb_push_back(qkv_done_trigger_cb_tr, 1);
    }

    // =========================================================================
    // PHASE 5 (#11 Commit 6): SDPA QK^T first-tile compute.
    //
    // Each SDPA worker computes the (0, 0) tile of its M-slice of QK^T:
    //   output_tile = Q[m=0, :] @ K[m=0, :]^T   (32 × 32 bf16)
    //
    // Inputs:
    //   sdpa_q_cb:        Q tiles (M_per_worker × head_dim_padded) = 12 tiles
    //   sdpa_k_partial_cb: K row 0 (32 × head_dim_padded) = 3 tiles
    // Output:
    //   sdpa_qk_probe_cb: 1 tile (32, 32) = first QK^T output tile for this
    //                     worker's M-slice.
    //
    // Matmul iteration: inner loop over k_inner ∈ [0..2] (the 3 head_dim
    // tiles) with matmul_block(transpose=true on K), reading Q tile k_inner
    // (Q[0, k_inner], i.e. Q's row 0 col k_inner) and K tile k_inner (K[0,
    // k_inner]). transpose=true makes the matmul compute Q @ K^T directly.
    //
    // Future SDPA compute commits iterate this pattern over all (m_out, n_out)
    // pairs of QK^T (8 × 8 = 64 output tiles per worker for SigLIP shape) with
    // streaming K-tile reads inside the loop.
    // =========================================================================
    if (is_sdpa_core_tr) {
        cb_wait_front(sdpa_q_cb_tr, sdpa_q_tiles_per_worker_tr);
        cb_wait_front(sdpa_k_partial_cb_tr, sdpa_k_partial_tiles_tr);
        cb_reserve_back(sdpa_qk_probe_cb_tr, 1);

        constexpr uint32_t SUBBLOCK_H = 1;
        constexpr uint32_t SUBBLOCK_W = 1;

        reconfig_data_format(sdpa_k_partial_cb_tr, sdpa_q_cb_tr);
        pack_reconfig_data_format(sdpa_qk_probe_cb_tr);
        mm_init(sdpa_q_cb_tr, sdpa_k_partial_cb_tr, sdpa_qk_probe_cb_tr);
        mm_block_init_short(
            sdpa_q_cb_tr,
            sdpa_k_partial_cb_tr,
            /*transpose=*/1,
            /*ct_dim=*/SUBBLOCK_W,
            /*rt_dim=*/SUBBLOCK_H,
            /*kt_dim=*/sdpa_k_partial_tiles_tr);

        tile_regs_acquire();
        for (uint32_t k = 0; k < sdpa_k_partial_tiles_tr; ++k) {
            matmul_block(
                sdpa_q_cb_tr,
                sdpa_k_partial_cb_tr,
                /*in0_index=*/k,  // Q[m=0, k] — Q's row 0 col k
                /*in1_index=*/k,  // K[m=0, k] — K's row 0 col k (transposed inside)
                /*idst=*/0,
                /*transpose=*/true,
                /*ct_dim=*/SUBBLOCK_W,
                /*rt_dim=*/SUBBLOCK_H,
                /*kt_dim=*/sdpa_k_partial_tiles_tr);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile<true>(0, sdpa_qk_probe_cb_tr, 0);
        tile_regs_release();

        cb_push_back(sdpa_qk_probe_cb_tr, 1);
        cb_pop_front(sdpa_q_cb_tr, sdpa_q_tiles_per_worker_tr);
        cb_pop_front(sdpa_k_partial_cb_tr, sdpa_k_partial_tiles_tr);
    }
#endif
}
