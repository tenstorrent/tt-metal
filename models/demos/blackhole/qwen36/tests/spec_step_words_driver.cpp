// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Host driver of tt/kernels/spec_step_words.hpp for tests/test_spec_kernels_host.py: runs the kernels' arithmetic
// (accept_words, chain_words) and L1 layouts on the host. Reads cases from stdin as whitespace-separated integers,
// writes one line of integers per case:
//   accept B K pack_words top draws | ids[R] [draws[R]] tokens[R] user_rows[6 * B]
//     -> pack[(2K + 5) * pack_words] real[R]
//   chain B K nv groups step_words ctrl_words top sampler | ids[R] [draws[R]] anchor[B] overlay[9 * B]
//                                                          drafts[K * B] cand[R * 2 * nv * groups] [seed_cand[R * T]]
//     -> step[9 * step_words] ctrl[groups * ctrl_words] [seeds[R]]
//   accept_layout B K pack_words chunk -> bytes
//   chain_layout B K nv groups step_words ctrl_words sampler -> bytes

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "spec_step_words.hpp"

using namespace qwen36_spec;

namespace {

long long read_int() {
    long long v = 0;
    if (std::scanf("%lld", &v) != 1) {
        std::exit(2);
    }
    return v;
}

std::vector<uint32_t> read_u32(size_t n) {
    std::vector<uint32_t> v(n);
    for (auto& x : v) {
        x = static_cast<uint32_t>(read_int());
    }
    return v;
}

std::vector<int32_t> read_i32(size_t n) {
    std::vector<int32_t> v(n);
    for (auto& x : v) {
        x = static_cast<int32_t>(read_int());
    }
    return v;
}

template <typename W>
void print_words(const std::vector<W>& v) {
    for (const auto& x : v) {
        std::printf(" %lld", static_cast<long long>(x));
    }
}

void accept_case() {
    const uint32_t B = read_int(), K = read_int(), pack_words = read_int();
    const int32_t top = read_int();
    const bool draws = read_int() != 0;
    const uint32_t R = B * (K + 1);
    const auto ids = read_u32(R);
    const auto dr = draws ? read_u32(R) : std::vector<uint32_t>();
    const auto tokens = read_u32(R);
    const auto users = read_u32(STEP_USER_ROWS * B);
    std::vector<uint32_t> pack(pack_rows(K) * pack_words, 0xDEADBEEFu), real(R, 7u);
    accept_words(
        AcceptShape{B, K, pack_words, top},
        AcceptIn{ids.data(), draws ? dr.data() : nullptr, tokens.data(), users.data(), B},
        AcceptOut{pack.data(), pack_words, real.data()});
    print_words(pack);
    print_words(real);
    std::printf("\n");
}

void chain_case() {
    const uint32_t B = read_int(), K = read_int(), nv = read_int(), groups = read_int();
    const uint32_t step_words = read_int(), ctrl_words = read_int();
    const int32_t top = read_int();
    const bool sampler = read_int() != 0;
    const uint32_t T = K + 1, R = B * T, cand_words = 2 * nv * groups;
    const auto ids = read_u32(R);
    const auto draws = sampler ? read_u32(R) : std::vector<uint32_t>();
    const auto anchor = read_u32(B);
    const auto overlay = read_i32(CHAIN_OVERLAY_ROWS * B);
    const auto drafts = read_u32(K * B);
    const auto cand = read_i32(R * cand_words);
    const auto seed_cand = sampler ? read_i32(R * T) : std::vector<int32_t>();
    std::vector<uint32_t> step(VERIFY_STEP_ROWS * step_words, 0xDEADBEEFu), seeds(R, 0xDEADBEEFu);
    std::vector<int32_t> ctrl(groups * ctrl_words, 0x5A5A5A5A);
    chain_words(
        ChainShape{B, K, nv, groups, step_words, ctrl_words, top},
        ChainIn{
            ids.data(),
            sampler ? draws.data() : nullptr,
            anchor.data(),
            overlay.data(),
            B,
            drafts.data(),
            1,
            cand.data(),
            cand_words,
            sampler ? seed_cand.data() : nullptr,
            T},
        ChainOut{step.data(), step_words, ctrl.data(), ctrl_words, sampler ? seeds.data() : nullptr});
    print_words(step);
    print_words(ctrl);
    if (sampler) {
        print_words(seeds);
    }
    std::printf("\n");
}

}  // namespace

int main() {
    char mode[32];
    while (std::scanf("%31s", mode) == 1) {
        if (std::strcmp(mode, "accept") == 0) {
            accept_case();
        } else if (std::strcmp(mode, "chain") == 0) {
            chain_case();
        } else if (std::strcmp(mode, "accept_layout") == 0) {
            const uint32_t B = read_int(), K = read_int(), pack_words = read_int(), chunk = read_int();
            std::printf("%u\n", AcceptLayout(B, K, pack_words, chunk).bytes);
        } else if (std::strcmp(mode, "chain_layout") == 0) {
            const uint32_t B = read_int(), K = read_int(), nv = read_int(), groups = read_int();
            const uint32_t step_words = read_int(), ctrl_words = read_int();
            const bool sampler = read_int() != 0;
            std::printf("%u\n", ChainLayout(B, K, nv, groups, step_words, ctrl_words, sampler).bytes);
        } else {
            return 3;
        }
    }
    return 0;
}
