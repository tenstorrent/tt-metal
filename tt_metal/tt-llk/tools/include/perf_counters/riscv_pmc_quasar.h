// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/riscv_pmc_types.h"

// Quasar select tables: the two TRISC events Blackhole's baby cores lack, and the data-movement cores.

// Same placement rule as quasar.h: LLK_PERF_TABLES_IN_TEXT keeps the tables out of the DM firmware data region.
#ifndef LLK_PERF_TABLE_SECTION
#if defined(LLK_PERF_TABLES_IN_TEXT)
#define LLK_PERF_TABLE_SECTION __attribute__((section(".text.perf_counter_tables")))
#else
#define LLK_PERF_TABLE_SECTION
#endif
#endif

namespace llk::perf::riscv
{

// clang-format off
// The TRISC events are the Blackhole baby_events plus these two; the host parser joins them.
constexpr std::array<Entry, 2> trisc_extra_events LLK_PERF_TABLE_SECTION = {
    {
    {PmcEvent::IF_STALL_NO_IF_RTS, 0x70}, // front end starved, decode ready but nothing to give
    {PmcEvent::IF_ID_STALL_LD, 0x80}}};   // decode stalled on a load-use hazard

constexpr std::array<Entry, 24> dm_events LLK_PERF_TABLE_SECTION = {
    {
    {PmcEvent::DM_COMMIT_EXCEPTION, 0x100},
    {PmcEvent::DM_COMMIT_LOAD, 0x200},
    {PmcEvent::DM_COMMIT_STORE, 0x400},
    {PmcEvent::DM_COMMIT_AMO, 0x800},
    {PmcEvent::DM_COMMIT_SYSTEM, 0x1000},
    {PmcEvent::DM_COMMIT_ARITH, 0x2000},
    {PmcEvent::DM_COMMIT_BRANCH, 0x4000},
    {PmcEvent::DM_COMMIT_JAL, 0x8000},
    {PmcEvent::DM_COMMIT_JALR, 0x10000},
    {PmcEvent::DM_COMMIT_MUL, 0x20000},
    {PmcEvent::DM_COMMIT_DIV, 0x40000},
    {PmcEvent::DM_PIPE_LOAD_USE, 0x101},
    {PmcEvent::DM_PIPE_LONG_LATENCY, 0x201},
    {PmcEvent::DM_PIPE_CSR_INTERLOCK, 0x401},
    {PmcEvent::DM_PIPE_ICACHE_BLOCKED, 0x801},
    {PmcEvent::DM_PIPE_DCACHE_BLOCKED, 0x1001},
    {PmcEvent::DM_PIPE_BRANCH_MISPREDICT, 0x2001},
    {PmcEvent::DM_PIPE_TARGET_MISPREDICT, 0x4001},
    {PmcEvent::DM_PIPE_FLUSH, 0x8001},
    {PmcEvent::DM_PIPE_REPLAY, 0x10001},
    {PmcEvent::DM_PIPE_MULDIV_INTERLOCK, 0x20001},
    {PmcEvent::DM_MEM_ICACHE_MISS, 0x102},
    {PmcEvent::DM_MEM_DCACHE_MISS, 0x202},
    {PmcEvent::DM_MEM_DCACHE_RELEASE, 0x402}}};
// clang-format on

} // namespace llk::perf::riscv
