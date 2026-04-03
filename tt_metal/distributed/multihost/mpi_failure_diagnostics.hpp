// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <mpi.h>
#include "mpi_ulfm_config.hpp"
#include "mpi_distributed_context.hpp"

namespace tt::tt_metal::distributed::multihost {

// Shared ULFM fast-fail exit code (EX_SOFTWARE / ttrun exit 70).
inline constexpr int k_mpi_ulfm_fast_fail_exit_code = 70;

inline constexpr std::string_view k_mpi_unknown_world_rank_name = "unknown-world-rank";
inline constexpr std::string_view k_mpi_unknown_hostname_sv = "unknown-hostname";

[[nodiscard]] bool can_query_mpi_process_identity() noexcept;

[[nodiscard]] std::string_view best_effort_local_rank_hostname_view(
    std::array<char, MPI_MAX_PROCESSOR_NAME>& processor_name_buffer) noexcept;

[[nodiscard]] std::string best_effort_local_rank_hostname();

[[nodiscard]] int best_effort_local_world_rank() noexcept;

#if OMPI_HAS_ULFM
[[nodiscard]] std::vector<std::string> best_effort_gather_rank_hostnames(MPI_Comm comm, int local_rank, int size);
#endif

[[nodiscard]] std::string format_world_rank_name(int world_rank);

[[nodiscard]] std::string_view failure_policy_name(FailurePolicy policy) noexcept;

[[nodiscard]] std::string_view hostname_for_rank(std::span<const std::string> rank_hostnames, int rank) noexcept;

#if OMPI_HAS_ULFM
void emit_rank_failure_diagnostics(
    const std::vector<Rank>& failed_ranks,
    std::string_view failed_ranks_text,
    std::span<const std::string> rank_hostnames,
    int detecting_rank,
    std::string_view operation,
    int error_code,
    FailurePolicy policy);
#endif

void capture_terminate_reason(char* buf, std::size_t buf_cap) noexcept;

void emit_local_mpi_process_fatal_diagnostic(
    int world_rank, std::string_view hostname, std::string_view operation, const char* reason_cstr) noexcept;

}  // namespace tt::tt_metal::distributed::multihost
