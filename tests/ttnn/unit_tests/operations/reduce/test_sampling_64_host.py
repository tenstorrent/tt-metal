# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run directly with Python: actual writer helper checks, without importing TTNN.

These CPU checks do not validate SFPU top-k, dataflow synchronization, or device
execution. The companion device tests are required before enabling this path.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class Sampling64HostTests(unittest.TestCase):
    def test_production_address_and_draw_helpers(self):
        root = Path(__file__).resolve().parents[5]
        compiler = shutil.which("clang++-20") or shutil.which("c++")
        self.assertIsNotNone(compiler)
        code = r"""
#include <array>
#include <cassert>
#include <cmath>
#include "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/sampling_64_utils.hpp"

int main() {
    // Construct the physical two-tile/four-face layout independently of the
    // production offset expression, including every user and candidate rank.
    std::array<unsigned, 2048> packed{};
    unsigned offset = 0;
    for (unsigned tile = 0; tile < 2; ++tile)
        for (unsigned face_row = 0; face_row < 2; ++face_row)
            for (unsigned face_col = 0; face_col < 2; ++face_col)
                for (unsigned r = 0; r < 16; ++r)
                    for (unsigned c = 0; c < 16; ++c)
                        packed[offset++] = (face_row * 16 + r) * 1000 + tile * 32 + face_col * 16 + c;
    for (unsigned user = 0; user < 32; ++user)
        for (unsigned rank = 0; rank < 64; ++rank)
            assert(packed[sampling_64::candidate_offset(user, rank)] == user * 1000 + rank);

    // Deterministic thresholds select every rank, including 32..63. The old
    // two-face writer addresses these as adjacent users, which the check above
    // rejects. No PRNG or hardware approximation is involved in this oracle.
    for (unsigned k : {1u, 16u, 17u, 31u, 32u, 33u, 40u, 48u, 49u, 64u}) {
        auto prob = [k](unsigned) { return 1.0f / k; };
        for (unsigned rank = 0; rank < k; ++rank) {
            float threshold = (rank + 0.5f) / k;
            assert(sampling_64::sample_rank(k, 1.0f, threshold, prob) == rank);
        }
        assert(sampling_64::sample_rank(k, 0.0f, 0.999f, prob) == 0);
    }
    // Exact binary probabilities make strict > cutoffs unambiguous: p=.5
    // keeps 33 of 64 candidates; BF16(.95)=.94921875 keeps 61 of 64.
    auto uniform64 = [](unsigned) { return 1.0f / 64; };
    assert(sampling_64::sample_rank(64, .5f, .999f, uniform64) == 32);
    assert(sampling_64::sample_rank(64, .94921875f, .999f, uniform64) == 60);
    assert(sampling_64::sample_rank(64, 1.0f, 1.0f - std::ldexp(1.0f, -24), uniform64) == 63);

    // Unequal probabilities, cutoff after tile 0, checked against an independent
    // explicit CDF and midpoint thresholds for every retained candidate.
    std::array<float, 64> probs{};
    float total = 0;
    for (unsigned i = 0; i < 64; ++i) { probs[i] = 128.0f - i; total += probs[i]; }
    for (auto& p : probs) p /= total;
    float mass = 0; unsigned kept = 0;
    while (kept < 64 && mass <= .94921875f) mass += probs[kept++];
    assert(kept > 32);
    float cumulative = 0;
    for (unsigned i = 0; i < kept; ++i) {
        float threshold = (cumulative + probs[i] / 2) / mass;
        assert(sampling_64::sample_rank(64, .94921875f, threshold, [&](unsigned j) { return probs[j]; }) == i);
        cumulative += probs[i];
    }
}
"""
        with tempfile.TemporaryDirectory(prefix="sampling64-host-") as temp:
            source = Path(temp) / "check.cpp"
            executable = Path(temp) / "check"
            source.write_text(code)
            subprocess.run(
                [
                    compiler,
                    "-std=c++20",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fsanitize=address,undefined",
                    "-I",
                    str(root),
                    str(source),
                    "-o",
                    str(executable),
                ],
                check=True,
            )
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
