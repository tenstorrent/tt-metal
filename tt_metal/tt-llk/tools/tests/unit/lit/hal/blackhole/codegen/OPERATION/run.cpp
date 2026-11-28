// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/tables.cpp -o %t/tables.o
// RUN: %{blackhole_objdump} -drz %t/tables.o > %t/tables.dump
// RUN: %{blackhole_compare_codegen} %t/tables.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/unpack.cpp -o %t/unpack.o
// RUN: %{blackhole_objdump} -drz %t/unpack.o > %t/unpack.dump
// RUN: %{blackhole_compare_codegen} %t/unpack.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/atomic.cpp -o %t/atomic.o
// RUN: %{blackhole_objdump} -drz %t/atomic.o > %t/atomic.dump
// RUN: %{blackhole_compare_codegen} %t/atomic.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// tables.cpp: a run-time-selected table of hal::Operation issues like production's raw-word table.
// unpack.cpp, atomic.cpp: object methods with compile-time operands issue immediates; a fallback
// to the instruction buffer fails the comparison with TTI_INSN of the same encoding.

//--- tables.cpp
#include <cstdint>

#include "hal/mop.h"
#include "hal/replay.h"

namespace mop    = hal::mop;
namespace replay = hal::replay;

using Template0 = mop::Runner<mop::MopTemplate::Template0>;

// A decode table mixing REPLAY and Template 0 MOP words, as the compressed matmul kernels use.
inline constexpr hal::Operation table[4] = {
    replay::get_operation(replay::BufferRange {0, 4}),
    replay::get_operation(replay::BufferRange {4, 6}),
    Template0 {.count = 1, .mask = 0x1}.operation(),
    replay::get_operation(replay::BufferRange {10, 3})};

// The table holds production's words.
static_assert(table[0] == TT_OP_REPLAY(0 /*start_idx*/, 4 /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/));
static_assert(table[1] == TT_OP_REPLAY(4 /*start_idx*/, 6 /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/));
static_assert(table[2] == TT_OP_MOP(0 /*mop_type*/, 0 /*loop_count*/, 0x1 /*zmask_lo16*/));
static_assert(table[3] == TT_OP_REPLAY(10 /*start_idx*/, 3 /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/));

static_assert(sizeof(hal::Operation) == sizeof(std::uint32_t), "an Operation is one instruction word");

extern "C" __attribute__((noinline, used)) void run_table(const std::uint32_t meta)
{
    hal::run(table[meta & 3]);
    hal::run(table[(meta >> 2) & 3]);
}

extern "C" __attribute__((noinline, used)) void reference_run_table(const std::uint32_t meta)
{
    // Production issues a raw word from its table.
    TT_INSN(static_cast<std::uint32_t>(table[meta & 3]));
    TT_INSN(static_cast<std::uint32_t>(table[(meta >> 2) & 3]));
}

extern "C" __attribute__((noinline, used)) void run_constant()
{
    hal::run(replay::get_operation<replay::BufferRange {4, 6}>());
}

extern "C" __attribute__((noinline, used)) void reference_run_constant()
{
    TTI_INSN(TT_OP_REPLAY(4 /*start_idx*/, 6 /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/));
}

//--- unpack.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

inline constexpr unpack::Transfer face {.increments = {.channel0 = {.z = 1}}, .handoff = unpack::SrcHandoff::FlipAndSetDataValid};
inline constexpr unpack::Clear zero {.handoff = unpack::SrcHandoff::Keep};

extern "C" __attribute__((noinline, used)) void named_unpacker_template_forms()
{
    unpack::Unpacker0.transfer<face>();
    unpack::Unpacker1.clear<zero>();
    unpack::Unpacker0.increment_context_counter();
    unpack::Unpacker1.flush_row_start_cache<unpack::CacheScope::AllEntries>();
    unpack::experimental::clear_both_banks<unpack::experimental::BothBanksClear {}>(unpack::Unpacker0);
}

extern "C" __attribute__((noinline, used)) void reference_named_unpacker_template_forms()
{
    TTI_INSN(unpack::Unpacker0.operation().transfer<face>());
    TTI_INSN(unpack::Unpacker1.operation().clear<zero>());
    TTI_INSN(unpack::Unpacker0.operation().increment_context_counter());
    TTI_INSN(unpack::Unpacker1.operation().flush_row_start_cache<unpack::CacheScope::AllEntries>());
    TTI_INSN(unpack::experimental::clear_both_banks_operation<unpack::experimental::BothBanksClear {}>(unpack::Unpacker0));
}

//--- atomic.cpp
#include "hal/atomic.h"

namespace atomic = hal::atomic;

inline constexpr auto counter = atomic::counter<32>(hal::gpr<1>(), 3 /*word*/);
inline constexpr auto lock    = atomic::Lock {hal::gpr<7>(), 2 /*word*/};
inline constexpr auto fifo    = atomic::fifo<3>(hal::gpr<3>());

extern "C" __attribute__((noinline, used)) void atomic_object_template_forms()
{
    counter.fetch_add(hal::gpr<2>());
    lock.acquire();
    lock.compare_set<3 /*Compare*/, 4 /*Set*/>();
    lock.release();
    fifo.pop<1 /*IncrementLog2*/>(hal::gpr<4>());
    fifo.push(hal::gpr<4>());
    fifo.wait_while<atomic::FifoState::Empty>(hal::gpr<4>());
}

extern "C" __attribute__((noinline, used)) void reference_atomic_object_template_forms()
{
    TTI_INSN(counter.operation().fetch_add(hal::gpr<2>()));
    TTI_INSN(lock.operation().acquire());
    TTI_INSN((lock.operation().compare_set<3 /*Compare*/, 4 /*Set*/>()));
    TTI_INSN(lock.operation().release());
    TTI_INSN(fifo.operation().pop<1 /*IncrementLog2*/>(hal::gpr<4>()));
    TTI_INSN(fifo.operation().push(hal::gpr<4>()));
    TTI_INSN(fifo.operation().wait_while<atomic::FifoState::Empty>(hal::gpr<4>()));
}
