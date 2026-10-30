// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"
void run_only(const std::uint32_t n) { replay::run(replay::BufferRange {4, n}); }
void run_twice(const std::uint32_t n, const std::uint32_t m) { replay::run(replay::BufferRange {4, n}); replay::run(replay::BufferRange {5, m}); }
void run_n_n(const std::uint32_t n) { replay::run(replay::BufferRange {4, n}); replay::run(replay::BufferRange {5, n}); }
void run_n_loop(const std::uint32_t n, const std::uint32_t k) { for (std::uint32_t i = 0; i < k; i++) replay::run(replay::BufferRange {4, n}); }
void run_n_ge(const std::uint32_t n) { replay::run(replay::BufferRange {4, n + 1}); }
void run_n64(const std::uint32_t n) { replay::run(replay::BufferRange {4, 64u * (n != 0) + 3u * (n == 0)}); }
