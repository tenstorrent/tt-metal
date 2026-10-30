// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

using replay::BufferRange;
void rec_const() {
    replay::record<BufferRange{0, 3}>([] { TTI_NOP; TTI_NOP; TTI_SETC16(1, 2); });
    replay::run<BufferRange{0, 3}>();
}
void rec_exec() {
    replay::record<BufferRange{4, 2}, replay::RecordBehavior::RecordAndExecute>([] { TTI_NOP; TTI_NOP; });
}
void rec_count64() {
    replay::record<BufferRange{0, 32}>([] { for (int i = 0; i < 32; ++i) { TTI_NOP; } });
    replay::run<BufferRange{0, 32}>();
}
void rec_wrap() {
    replay::record<BufferRange{30, 4}>([] { TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; });
    replay::run<BufferRange{30, 4}>();
}
void rec_rt(const std::uint32_t n) {
    replay::record(BufferRange{0, n}, [&] { for (std::uint32_t i = 0; i < n; ++i) { TTI_NOP; } });
    replay::run(BufferRange{0, n});
}
void rec_rt_cond(const bool partial) {
    replay::record(BufferRange{0, partial ? 12u : 10u}, [&] {
        if (partial) { TTI_NOP; TTI_NOP; }
        for (int i = 0; i < 10; ++i) TTI_NOP;
    });
    replay::run(BufferRange{0, partial ? 12u : 10u});
}
void run_rt_start_const(const std::uint32_t n) { replay::run(BufferRange{5, n}); }
void replay_op_noissue() { volatile std::uint32_t x = replay::get_operation<BufferRange{0, 3}>(); (void)x; }
void replay_op_rt(const std::uint32_t n) { volatile std::uint32_t x = replay::get_operation(BufferRange{0, n}); (void)x; }
void raw_replay() { TTI_REPLAY(1, 5, 0, 0); TT_REPLAY(2, 6, 1, 0); }
void raw_replay_rt(const std::uint32_t n) { TT_REPLAY(2, n, 0, 0); }
