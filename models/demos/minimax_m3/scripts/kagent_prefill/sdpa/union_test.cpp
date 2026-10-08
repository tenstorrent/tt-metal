// Host test: hashed union (sparse_sdpa_msa_packed_union.hpp) == the linear-scan union of the first packed reader.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include "sparse_sdpa_msa_packed_union.hpp"

static uint32_t linear(
    const std::vector<uint32_t>& rows,
    const uint32_t* nv,
    uint32_t g_max,
    uint32_t topk,
    std::vector<uint32_t>& uid,
    std::vector<uint32_t>& um) {
    uid.assign(rows.begin(), rows.begin() + nv[0]);
    um.assign(nv[0], 1u);
    uint32_t g = 1;
    for (uint32_t j = 1; j < g_max; ++j) {
        uint32_t bit = 1u << j, n_before = uid.size();
        for (uint32_t c = 0; c < nv[j]; ++c) {
            uint32_t b = rows[j * topk + c], e = 0;
            while (e < uid.size() && !(uid[e] == b && (um[e] & bit) == 0)) {
                ++e;
            }
            if (e == uid.size()) {
                uid.push_back(b);
                um.push_back(0);
            }
            um[e] |= bit;
        }
        uint32_t all = (bit << 1) - 1;
        bool lead = false;
        for (uint32_t e = 0; e < n_before && !lead; ++e) {
            lead = (um[e] & all) == all;
        }
        if (!lead) {
            uid.resize(n_before);
            um.resize(n_before);
            for (auto& m : um) {
                m &= ~bit;
            }
            break;
        }
        g = j + 1;
    }
    uint32_t all = (1u << g) - 1, lead = 0;
    while ((um[lead] & all) != all) {
        ++lead;
    }
    if (lead) {
        uint32_t a = uid[lead], m = um[lead];
        uid.erase(uid.begin() + lead);
        um.erase(um.begin() + lead);
        uid.insert(uid.begin(), a);
        um.insert(um.begin(), m);
    }
    return g;
}

int main() {
    std::mt19937 rng(1);
    const uint32_t topk = 16;
    long cases = 0;
    for (int trial = 0; trial < 200000; ++trial) {
        uint32_t G = 2 + 2 * (rng() % 6);  // 2..12
        uint32_t g_max = 1 + rng() % G;
        uint32_t span = (rng() % 3 == 0) ? 20 : (rng() % 2 ? 600 : 9000);
        std::vector<uint32_t> rows(G * topk, 0xFFFFFFFFu);
        uint32_t nv[16];
        uint32_t base = rng() % 9000;
        for (uint32_t j = 0; j < g_max; ++j) {
            nv[j] = 1 + rng() % topk;
            for (uint32_t c = 0; c < nv[j]; ++c) {
                uint32_t r = rng() % 10;
                uint32_t b = (r < 3) ? base : (r < 6 ? (base + rng() % 8) : (rng() % span));
                if (rng() % 20 == 0 && c) {
                    b = rows[j * topk + rng() % c];  // duplicate
                }
                rows[j * topk + c] = b;
            }
        }
        std::vector<uint32_t> uid, um;
        uint32_t g1 = linear(rows, nv, g_max, topk, uid, um);
        std::vector<uint32_t> id(G * topk), mask(G * topk);
        std::vector<uint16_t> head(256), tail(256), next(G * topk);
        sparse_sdpa_msa_union::Union<uint32_t*, uint16_t*> u{
            id.data(), mask.data(), head.data(), tail.data(), next.data()};
        uint32_t g2 = sparse_sdpa_msa_union::build_group(u, rows.data(), nv, g_max, topk);
        bool ok = g1 == g2 && u.n == uid.size();
        for (uint32_t e = 0; ok && e < u.n; ++e) {
            ok = id[e] == uid[e] && mask[e] == um[e];
        }
        // diag lookup == linear first-owned scan
        for (uint32_t s = 0; ok && s < g2; ++s) {
            uint32_t b = rows[s * topk + rng() % nv[s]];
            uint32_t lin = 0;
            while (lin < uid.size() && !(uid[lin] == b && (um[lin] >> s & 1))) {
                ++lin;
            }
            ok = u.find_owned(b, 1u << s) == lin;
        }
        if (!ok) {
            printf("MISMATCH trial %d G=%u g_max=%u g1=%u g2=%u n=%zu/%u\n", trial, G, g_max, g1, g2, uid.size(), u.n);
            return 1;
        }
        ++cases;
    }
    printf("union_test: %ld cases OK\n", cases);
    return 0;
}
