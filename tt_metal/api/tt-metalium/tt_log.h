// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Define TT_LOG and it's derivatives so the compile passes.
// If we are running workloads on hardware, TT_LOG will already have been defined.
#ifndef TT_LOG_DEFINED
#define TT_LOG_DEFINED
#define TT_LOG(...) (void)sizeof(__VA_ARGS__)
#define TT_LOG_NB(...) (void)sizeof(__VA_ARGS__)
#define TT_PAUSE(...) (void)sizeof(__VA_ARGS__)
#define TT_RISC_ASSERT(...) (void)sizeof(__VA_ARGS__)
#define TT_LLK_DUMP(...) (void)sizeof(__VA_ARGS__)
#define TT_DUMP_LOG(...) (void)sizeof(__VA_ARGS__)
#define TT_DUMP_ASSERT(...) (void)sizeof(__VA_ARGS__)
#endif
