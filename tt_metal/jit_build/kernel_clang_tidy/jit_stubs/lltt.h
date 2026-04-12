// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stub for lltt.h (clang-tidy analysis only).
//
// lltt.h is referenced by llk_math_eltwise_binary.h for tracing/telemetry
// (lltt::record<lltt::NoExec>(...)).  The file does not exist in the repo
// tree checked by clang-tidy, so we provide this minimal stub that satisfies
// the call site without pulling in any hardware-specific code.

#pragma once

namespace lltt {

struct NoExec {};
struct Exec {};

// record<Mode>(start, length) — telemetry stub
template <typename Mode = NoExec>
[[gnu::always_inline]] inline void record(unsigned, unsigned) {}

// replay(start, length) — instruction replay stub
[[gnu::always_inline]] inline void replay(unsigned, unsigned) {}

// replay_insn — compile-time replay encoding stub
[[gnu::always_inline]] constexpr unsigned int replay_insn(unsigned, unsigned) { return 0; }

}  // namespace lltt
