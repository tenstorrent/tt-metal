// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Exact scan with only the final persistent-state pack changed to BF16.
template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
inline void scan_step_state_bf16(const GdnScanCbs& cb, uint32_t cur_S, uint32_t dst) {
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t kc = Kt * Ct;
    // Every CB the scan touches is fp32 (hand-off intermediates, state, scratch, o), so the
    // unpacker/packer formats set once at kernel start (compute_kernel_hw_startup on fp32 CBs) hold
    // for the whole step: skip the per-call reconfigs (the producer does the same under
    // GDN_HOIST_RECONFIG). Formats are identical either way => bit-exact.
    constexpr bool H = true;

    // v_new = T_inv @ (v_beta + nkd@S), nkd = -(k_beta*decay_exp) from prep -- apply the inverse
    // after the subtraction so the WY inverse's fp error is not amplified by the cancellation (vs the
    // u - w@S form). diff is formed in ONE DST pass of matmul accumulation: nkd @ S, then I @ v_beta
    // onto the same DST tile (MVMUL adds into DST; only the packer zeroes it at release). This replaces
    // the packed kd@S, its re-unpack and the eltwise block of the two-block form; the identity matmul
    // keeps the window inside one op class (no eltwise init or reconfig between the products). One
    // rounding moves: v_beta enters the sum through srcA like every other matmul operand instead of
    // through the eltwise add. diff itself must be packed: it is the in1 operand of T_inv @ diff, and
    // the WAIT on cb.ointer below is what keeps that block's unpacker behind this block's packer.
    {
        GDN_ZONE("st_diff");
        WAIT(cb.nkd, ck);
        WAIT(cb.vbeta, cv);
        WAIT(cur_S, kv);
        CircularBuffer(cb.ointer).reserve_back(cv);
        matmul_init(cb.nkd, cur_S, 0);  // one init serves both products: matmul_tiles binds its CBs per call
        for (uint32_t t0 = 0; t0 < cv; t0 += kDstTiles) {
            const uint32_t nb = (cv - t0 < kDstTiles) ? (cv - t0) : kDstTiles;
            tile_regs_acquire();
            for (uint32_t j = 0; j < nb; j++) {
                const uint32_t t = t0 + j;
                const uint32_t mi = t / Vt;
                const uint32_t ni = t - mi * Vt;
                for (uint32_t ki = 0; ki < Kt; ki++) {
                    matmul_tiles(cb.nkd, cur_S, mi * Kt + ki, ki * Vt + ni, j);  // DST[j]  = nkd @ S
                }
                matmul_tiles(cb.eye, cb.vbeta, 0, t, j);  // DST[j] += I @ v_beta
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nb; j++) {
                pack_tile(j, cb.ointer, t0 + j);
            }
            tile_regs_release();
        }
        CircularBuffer(cb.ointer).push_back(cv);
        POP(cb.nkd, ck);
        WAIT(cb.ointer, cv);  // the next block's unpacker must not run ahead of this block's packer
        POP(cb.vbeta, cv);
    }
    {
        GDN_ZONE("st_vnew");
        WAIT(cb.Tinv, cc);
        mm(cb.Tinv, cb.ointer, cb.vnew, Ct, Ct, Vt, false, H);  // v_new = T_inv @ diff -> vnew
        WAIT(cb.vnew, cv);
        POP(cb.Tinv, cc);
        POP(cb.ointer, cv);
    }

    // The two outputs are each a sum of two matmuls; both sums stay in DST: the second matmul
    // accumulates onto the first (MVMUL adds into DST, which only the packer zeroes at release), so
    // there is no packed intermediate, no re-unpack and no eltwise block. matmul_tiles binds its
    // operand CBs per call, so one matmul_init serves both products of a block.
    // o = q_decay @ S + intra @ v_new -> cb_out (drained by the writer)
    {
        GDN_ZONE("st_o");
        WAIT(cb.qdecay, ck);
        WAIT(cb.intra, cc);
        CircularBuffer(cb.out).reserve_back(cv);
        matmul_init(cb.qdecay, cur_S, 0);
        for (uint32_t t0 = 0; t0 < cv; t0 += kDstTiles) {
            const uint32_t nb = (cv - t0 < kDstTiles) ? (cv - t0) : kDstTiles;
            tile_regs_acquire();
            for (uint32_t j = 0; j < nb; j++) {
                const uint32_t t = t0 + j;
                const uint32_t mi = t / Vt;
                const uint32_t ni = t - mi * Vt;
                for (uint32_t ki = 0; ki < Kt; ki++) {
                    matmul_tiles(cb.qdecay, cur_S, mi * Kt + ki, ki * Vt + ni, j);
                }
                for (uint32_t kc_ = 0; kc_ < Ct; kc_++) {
                    matmul_tiles(cb.intra, cb.vnew, mi * Ct + kc_, kc_ * Vt + ni, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nb; j++) {
                pack_tile(j, cb.out, t0 + j);
            }
            tile_regs_release();
        }
        CircularBuffer(cb.out).push_back(cv);
        POP(cb.qdecay, ck);
        POP(cb.intra, cc);
    }
    // S_new = (dl*I) @ S + k_dec_t @ v_new -> dst (the next chunk's cur_S, or the final state).
    // The decay is block-diagonal, so every state tile (i,j) uses the single dl*I tile as in0.
    {
        GDN_ZONE("st_snew");
        WAIT(cb.kdec_t, kc);
        WAIT(cb.dl, 1);
        pack_reconfig_data_format(dst);
        CircularBuffer(dst).reserve_back(kv);
        matmul_init(cb.dl, cur_S, 0);
        for (uint32_t t0 = 0; t0 < kv; t0 += kDstTiles) {
            const uint32_t nb = (kv - t0 < kDstTiles) ? (kv - t0) : kDstTiles;
            tile_regs_acquire();
            for (uint32_t j = 0; j < nb; j++) {
                matmul_tiles(cb.dl, cur_S, 0, t0 + j, j);
            }
            for (uint32_t j = 0; j < nb; j++) {
                const uint32_t t = t0 + j;
                const uint32_t mi = t / Vt;
                const uint32_t ni = t - mi * Vt;
                for (uint32_t kc_ = 0; kc_ < Ct; kc_++) {
                    matmul_tiles(cb.kdec_t, cb.vnew, mi * Ct + kc_, kc_ * Vt + ni, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nb; j++) {
                pack_tile(j, dst, t0 + j);
            }
            tile_regs_release();
        }
        CircularBuffer(dst).push_back(kv);
        POP(cb.kdec_t, kc);
        POP(cb.dl, 1);
    }
    // v_new and the input state fed both blocks; release them once everything is packed.
    POP(cb.vnew, cv);
    POP(cur_S, kv);
}
