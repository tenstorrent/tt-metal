// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mpi_failure_diagnostics.hpp"
#include "mpi_ulfm_config.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <numeric>
#include <span>
#include <unistd.h>

#include <fmt/format.h>

namespace tt::tt_metal::distributed::multihost {

namespace {

// getenv() requires a null-terminated name; keep a C string for Open MPI / POSIX.
static constexpr char k_runner_name_env_var[] = "RUNNER_NAME";

[[nodiscard]] bool hostname_is_generic_for_rank_diagnostics(std::string_view hostname) noexcept {
    return hostname.empty() || hostname == k_mpi_unknown_hostname_sv || hostname == "localhost" ||
           hostname == "localhost.localdomain" || hostname == "mpirun-host";
}

[[nodiscard]] std::string_view select_best_effort_local_rank_hostname(
    std::string_view mpi_processor_name, std::string_view runner_name) noexcept {
    if (!hostname_is_generic_for_rank_diagnostics(mpi_processor_name)) {
        return mpi_processor_name;
    }

    if (!runner_name.empty()) {
        return runner_name;
    }

    if (!mpi_processor_name.empty()) {
        return mpi_processor_name;
    }

    return k_mpi_unknown_hostname_sv;
}

// Drop stack/backtrace noise from exception .what() for one-line CI diagnostics.
[[nodiscard]] std::string_view truncate_exception_text_at_backtrace(std::string_view text) noexcept {
    constexpr static std::string_view needle = "backtrace";
    for (std::size_t i = 0; i + needle.size() <= text.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < needle.size(); ++j) {
            if (std::tolower(static_cast<unsigned char>(text[i + j])) != static_cast<unsigned char>(needle[j])) {
                match = false;
                break;
            }
        }
        if (match) {
            std::size_t end = i;
            while (end > 0 && (text[end - 1] == ' ' || text[end - 1] == '\t')) {
                --end;
            }
            return text.substr(0, end);
        }
    }
    return text;
}

// Copy printable ASCII into a bounded buffer for a single-line diagnostic (semicolons/newlines → space).
[[nodiscard]] std::size_t sanitize_exception_text_into_buffer(
    std::string_view text, char* buf, std::size_t buf_cap) noexcept {
    if (buf_cap == 0) {
        return 0;
    }
    std::size_t out = 0;
    for (unsigned char c : text) {
        if (out + 1 >= buf_cap) {
            break;
        }
        if (c == '\n' || c == '\r' || c == ';') {
            buf[out++] = ' ';
        } else if (c == '\t') {
            buf[out++] = ' ';
        } else if (c >= 32 && c < 127) {
            buf[out++] = static_cast<char>(c);
        }
    }
    while (out > 0 && buf[out - 1] == ' ') {
        --out;
    }
    buf[out] = '\0';
    return out;
}

// Null-terminated copy; avoids strncpy padding quirks.
void copy_literal_to_buffer(char* buf, std::size_t buf_cap, std::string_view literal) noexcept {
    if (buf_cap == 0) {
        return;
    }
    const std::size_t n = std::min(literal.size(), buf_cap - 1);
    if (n > 0) {
        literal.copy(buf, n);
    }
    buf[n] = '\0';
}

}  // namespace

bool can_query_mpi_process_identity() noexcept {
    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
        return false;
    }

    int finalized = 0;
    return MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0;
}

std::string_view best_effort_local_rank_hostname_view(
    std::array<char, MPI_MAX_PROCESSOR_NAME>& processor_name_buffer) noexcept {
    std::string_view mpi_processor_name;
    if (can_query_mpi_process_identity()) {
        // MPI_Get_processor_name writes a NUL-terminated string; resultlen is strlen(name), excluding the NUL.
        int resultlen = 0;
        if (MPI_Get_processor_name(processor_name_buffer.data(), &resultlen) == MPI_SUCCESS && resultlen > 0) {
            const int capped = std::clamp(resultlen, 0, static_cast<int>(MPI_MAX_PROCESSOR_NAME) - 1);
            processor_name_buffer[static_cast<std::size_t>(capped)] = '\0';
            mpi_processor_name = std::string_view{processor_name_buffer.data(), static_cast<std::size_t>(capped)};
        }
    }

    std::string_view runner_name;
    if (const char* value = std::getenv(k_runner_name_env_var); value != nullptr && *value != '\0') {
        runner_name = value;
    }

    return select_best_effort_local_rank_hostname(mpi_processor_name, runner_name);
}

std::string best_effort_local_rank_hostname() {
    std::array<char, MPI_MAX_PROCESSOR_NAME> processor_name_buffer{};
    return std::string{best_effort_local_rank_hostname_view(processor_name_buffer)};
}

int best_effort_local_world_rank() noexcept {
    if (!can_query_mpi_process_identity()) {
        return -1;
    }

    // If MPI_Comm_rank fails, the output rank is undefined; keep a sentinel.
    int rank = -1;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
        return -1;
    }

    return rank;
}

#if OMPI_HAS_ULFM

std::vector<std::string> best_effort_gather_rank_hostnames(MPI_Comm comm, int local_rank, int size) {
    if (size <= 0) {
        return {};
    }

    // Fixed per-rank payload size for MPI_CHAR Allgather (MPI 4.x: counts are int).
    static constexpr int k_hostname_field_bytes = 256;
    const auto usize = static_cast<std::size_t>(size);

    std::vector<std::string> rank_hostnames(usize, std::string{k_mpi_unknown_hostname_sv});
    std::array<char, k_hostname_field_bytes> local_hostname{};
    const std::string local_hostname_value = best_effort_local_rank_hostname();
    const auto copied_length = std::min(local_hostname_value.size(), local_hostname.size() - 1);
    std::copy_n(local_hostname_value.data(), copied_length, local_hostname.data());
    local_hostname[copied_length] = '\0';
    if (local_rank >= 0 && local_rank < size) {
        rank_hostnames[static_cast<std::size_t>(local_rank)] = local_hostname.data();
    }

    std::vector<char> gathered_hostnames(usize * static_cast<std::size_t>(k_hostname_field_bytes), '\0');
    if (MPI_Allgather(
            local_hostname.data(),
            k_hostname_field_bytes,
            MPI_CHAR,
            gathered_hostnames.data(),
            k_hostname_field_bytes,
            MPI_CHAR,
            comm) != MPI_SUCCESS) {
        return rank_hostnames;
    }

    for (int rank = 0; rank < size; ++rank) {
        const auto offset = static_cast<std::size_t>(rank) * static_cast<std::size_t>(k_hostname_field_bytes);
        const std::span<const char> field{gathered_hostnames.data() + offset, k_hostname_field_bytes};
        const auto nul = std::find(field.begin(), field.end(), '\0');
        if (nul != field.begin()) {
            rank_hostnames[static_cast<std::size_t>(rank)] = std::string(field.begin(), nul);
        }
    }

    return rank_hostnames;
}

#endif  // OMPI_HAS_ULFM

std::string format_world_rank_name(int world_rank) { return fmt::format(FMT_STRING("world-rank-{}"), world_rank); }

std::string_view failure_policy_name(FailurePolicy policy) noexcept {
    switch (policy) {
        case FailurePolicy::FAST_FAIL: return "fast_fail";
        case FailurePolicy::FAULT_TOLERANT: return "fault_tolerant";
        default: return "fast_fail";
    }
}

std::string_view hostname_for_rank(std::span<const std::string> rank_hostnames, int rank) noexcept {
    if (rank < 0) {
        return k_mpi_unknown_hostname_sv;
    }
    const auto rank_index = static_cast<std::size_t>(rank);
    if (rank_index >= rank_hostnames.size() || rank_hostnames[rank_index].empty()) {
        return k_mpi_unknown_hostname_sv;
    }
    return rank_hostnames[rank_index];
}

#if OMPI_HAS_ULFM

// Logs stable key=value fields to stderr for CI and multihost smoke tests (same
// field names as the former GitHub annotation payload). Local std::terminate
// fatals use emit_local_mpi_process_fatal_diagnostic() with the same key layout.
void emit_rank_failure_diagnostics(
    const std::vector<Rank>& failed_ranks,
    std::string_view failed_ranks_text,
    std::span<const std::string> rank_hostnames,
    int detecting_rank,
    std::string_view operation,
    int error_code,
    FailurePolicy policy) {
    const std::string detecting_rank_name = format_world_rank_name(detecting_rank);
    const std::string_view detecting_hostname = hostname_for_rank(rank_hostnames, detecting_rank);
    const std::string_view policy_sv = failure_policy_name(policy);

    auto emit_line = [&](std::string_view failed_rank_name, std::string_view failed_hostname) {
        fmt::print(
            stderr,
            FMT_STRING("  ULFM detected a rank failure; failed_rank_name={}; failed_hostname={}; failed_ranks={}; "
                       "detecting_rank_name={}; detecting_hostname={}; operation={}; error_code={}; policy={}\n"),
            failed_rank_name,
            failed_hostname,
            failed_ranks_text,
            detecting_rank_name,
            detecting_hostname,
            operation,
            error_code,
            policy_sv);
    };

    if (failed_ranks.empty()) {
        emit_line(k_mpi_unknown_world_rank_name, k_mpi_unknown_hostname_sv);
    } else {
        for (const auto failed_rank : failed_ranks) {
            const std::string failed_rank_name = format_world_rank_name(*failed_rank);
            emit_line(failed_rank_name, hostname_for_rank(rank_hostnames, *failed_rank));
        }
    }
}

#endif  // OMPI_HAS_ULFM

void capture_terminate_reason(char* buf, std::size_t buf_cap) noexcept {
    if (buf_cap == 0) {
        return;
    }
    buf[0] = '\0';

    auto write_literal = [&](std::string_view literal) noexcept { copy_literal_to_buffer(buf, buf_cap, literal); };

    if (!std::current_exception()) {
        write_literal("uncaught_exception_or_explicit_terminate");
        return;
    }
    try {
        std::rethrow_exception(std::current_exception());
    } catch (const std::exception& e) {
        const char* w = e.what();
        const std::string_view raw = (w != nullptr) ? std::string_view{w} : std::string_view{};
        const std::string_view what_sv = truncate_exception_text_at_backtrace(raw);
        const std::size_t n = sanitize_exception_text_into_buffer(what_sv, buf, buf_cap);
        if (n == 0) {
            write_literal("std_exception_empty_what");
        }
    } catch (...) {
        write_literal("non_std_exception");
    }
}

void emit_local_mpi_process_fatal_diagnostic(
    int world_rank, std::string_view hostname, std::string_view operation, const char* reason_cstr) noexcept {
    // noexcept: fmt::format_to_n on stack buffers should not allocate; on failure, skip the structured line.
    try {
        std::array<char, 64> rank_name_storage{};
        std::array<char, 32> failed_ranks_storage{};
        std::string_view rank_name_sv;
        std::string_view failed_ranks_sv;

        if (world_rank >= 0) {
            {
                const auto r = fmt::format_to_n(
                    rank_name_storage.data(), rank_name_storage.size() - 1, FMT_STRING("world-rank-{}"), world_rank);
                *r.out = '\0';
                rank_name_sv = std::string_view{rank_name_storage.data(), r.size};
            }
            {
                const auto r = fmt::format_to_n(
                    failed_ranks_storage.data(), failed_ranks_storage.size() - 1, FMT_STRING("{}"), world_rank);
                *r.out = '\0';
                failed_ranks_sv = std::string_view{failed_ranks_storage.data(), r.size};
            }
        } else {
            rank_name_sv = k_mpi_unknown_world_rank_name;
            failed_ranks_sv = "unknown";
        }

        std::string_view host_sv = hostname.empty() ? k_mpi_unknown_hostname_sv : hostname;
        if (host_sv.size() > static_cast<std::size_t>(MPI_MAX_PROCESSOR_NAME) - 1) {
            host_sv = host_sv.substr(0, static_cast<std::size_t>(MPI_MAX_PROCESSOR_NAME) - 1);
        }

        const std::string_view reason_sv = (reason_cstr != nullptr && reason_cstr[0] != '\0')
                                               ? std::string_view{reason_cstr}
                                               : std::string_view{"unspecified"};

        std::array<char, 2048> line{};
        const auto result = fmt::format_to_n(
            line.data(),
            line.size() - 1,
            FMT_STRING("  ULFM detected a rank failure; failed_rank_name={}; failed_hostname={}; failed_ranks={}; "
                       "detecting_rank_name={}; detecting_hostname={}; operation={}; error_code={}; policy=fast_fail; "
                       "reason={}\n"),
            rank_name_sv,
            host_sv,
            failed_ranks_sv,
            rank_name_sv,
            host_sv,
            operation,
            k_mpi_ulfm_fast_fail_exit_code,
            reason_sv);
        *result.out = '\0';
        [[maybe_unused]] const auto w = write(STDERR_FILENO, line.data(), result.size);
    } catch (...) {
        // std::terminate path: never propagate.
    }
}

}  // namespace tt::tt_metal::distributed::multihost

#undef OMPI_HAS_ULFM
