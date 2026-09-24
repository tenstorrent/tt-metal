// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by the two single-candidate translation units, codegen_tu.cpp (item 6) and
// compile_time_tu.cpp (item 7). Each is compiled once per candidate with -DCANDIDATE_{STD,ZOO,FU2}
// and exposes it as Fn<Signature>.
//
// The skips matter: these TUs measure one library, so they must not pay to include the others.
// For item 7 that is the measurement itself; without them every row would carry the parse cost of
// all three.

#if defined(CANDIDATE_STD)
#define BENCH_SKIP_ZOO
#define BENCH_SKIP_FU2
#elif defined(CANDIDATE_ZOO)
#define BENCH_SKIP_FU2
#elif defined(CANDIDATE_FU2)
#define BENCH_SKIP_ZOO
#else
#error "define one of CANDIDATE_STD / CANDIDATE_ZOO / CANDIDATE_FU2"
#endif

#include "candidates.hpp"

#if defined(CANDIDATE_STD)
template <typename Signature>
using Fn = bench::StdFunction<Signature>;
#elif defined(CANDIDATE_ZOO)
template <typename Signature>
using Fn = bench::ZooFunction<Signature>;
#elif defined(CANDIDATE_FU2)
template <typename Signature>
using Fn = bench::Fu2Function<Signature>;
#endif
