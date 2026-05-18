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

    const bool is_ln1_core_nc = (get_relative_logical_y() == 0) && (get_relative_logical_x() < ln1_num_cores);
    const bool is_qkv_core_nc = (get_relative_logical_y() < qkv_grid_y) && (get_relative_logical_x() < qkv_grid_x);

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
        // Array slot 0 sits at common-arg-index 3 (after ln_out_l1_addr,
        // worker_phys_origin_x, worker_phys_origin_y).
        constexpr uint32_t LN1_PHYS_X_BASE = 3;
        for (uint32_t i = 0; i < ln1_num_cores; ++i) {
            const uint32_t ln1_phys_x = get_common_arg_val<uint32_t>(LN1_PHYS_X_BASE + i);
            const uint64_t src = get_noc_addr(ln1_phys_x, ln1_phys_y, ln_out_l1_addr);
            const uint32_t dst = qkv_act_write_base + i * shard_bytes;
            noc_async_read(src, dst, shard_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(qkv_act_cb, qkv_act_tiles_per_core);
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

    constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");
    constexpr uint32_t eps_bits = get_named_compile_time_arg_val("eps_bits");

    // Runtime role flags. Same logic as NCRISC; named with _tr suffix to keep
    // is-ln1-core unambiguous if the NCRISC and TRISC bodies are ever read
    // side-by-side.
    const bool is_ln1_core_tr = (get_relative_logical_y() == 0) && (get_relative_logical_x() < 8);
    const bool is_residual_core_tr = is_ln1_core_tr;
    // is_qkv_core_tr is computed but currently unused at TRISC level — Commit 3
    // will gate the QKV EncoderMatmul::Op call on it. Marked maybe_unused so
    // the compiler doesn't warn pre-Commit 3.
    [[maybe_unused]] const bool is_qkv_core_tr = (get_relative_logical_y() < 6) && (get_relative_logical_x() < 6);

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
#endif
}
