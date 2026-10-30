// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

using hal::mop::MopTemplate;
void t1_program_const() { mop::program(mop::loop(1 /*count*/, mop::loop(8, TT_OP_NOP))); }
void t1_program_rt(const std::uint32_t n) { mop::program(mop::loop(1 /*count*/, mop::loop(n, TT_OP_NOP))); }
void t1_run() { mop::Runner<MopTemplate::Template1>::run(); }
void t1_prog_run(const std::uint32_t n) {
    mop::program(mop::loop(2, mop::loop(n, TT_OP_NOP)));
    mop::Runner<MopTemplate::Template1>::run();
    mop::Runner<MopTemplate::Template1>::run();
}
void t0_run_const() { mop::Runner<MopTemplate::Template0>::run<4, 0xf>(); }
void t0_run_highmask() { mop::Runner<MopTemplate::Template0>::run<32, 0xaaaa5555u>(); }
void t0_run_rtcount(const std::uint32_t n) { mop::Runner<MopTemplate::Template0>::run<0xaaaa5555u>(n); }
void t0_run_rtmask(const std::uint32_t m) { mop::Runner<MopTemplate::Template0>::run_with_runtime_mask<8>(m); }
void t0_run_rtboth(const std::uint32_t n, const std::uint32_t m) { mop::Runner<MopTemplate::Template0>::run(n, m); }
void t0_cfg_mask_high_const() { mop::Runner<MopTemplate::Template0>::configure_mask_high<0x12340000u>(); }
void t0_cfg_mask_high_rt(const std::uint32_t m) { mop::Runner<MopTemplate::Template0>::configure_mask_high(m); }
void mop_update_rt(const std::uint32_t n) { mop::update(mop::outer_count, n); }
void mop_op_noissue() { volatile std::uint32_t x = mop::Runner<MopTemplate::Template1>{}.operation(); (void)x; }
