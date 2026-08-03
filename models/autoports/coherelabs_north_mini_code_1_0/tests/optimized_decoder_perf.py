# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed batch-1 performance harness for the optimized North-Mini decoder."""

import sys

from models.autoports.coherelabs_north_mini_code_1_0.tests import functional_decoder_perf as harness
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder


class _CandidateDecoder(OptimizedDecoder):
    candidate = "default"

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        return super().from_state_dict(state_dict, candidate=cls.candidate, **kwargs)


def main():
    candidate = "default"
    if "--candidate" in sys.argv:
        index = sys.argv.index("--candidate")
        candidate = sys.argv[index + 1]
        del sys.argv[index : index + 2]
    _CandidateDecoder.candidate = candidate
    harness.FunctionalDecoder = _CandidateDecoder
    harness.main()


if __name__ == "__main__":
    main()
