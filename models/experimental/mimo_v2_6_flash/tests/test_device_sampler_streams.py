# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeviceSampler's random streams, host side (no device): a seeded row's draw at each decode position depends only on
(seed, position), so reloading it at any step continues the same sequence; unseeded rows get distinct random streams.
"""

import torch

from models.experimental.mimo_v2_6_flash.tt.device_sampler import MASK64, splitmix_next, stream_states


def unsigned(x):
    return int(x) & MASK64


def draws(state, n):
    out = []
    for _ in range(n):
        state, u = splitmix_next(state)
        out.append(u)
    return out


def test_seeded_stream_depends_on_seed_and_position_only():
    for seed in (0, 7, -3, 2**40 + 5, 2**64 + 11):
        start = unsigned(stream_states([seed], [100])[0])
        sequence = draws(start, 20)
        # Reloaded at a later position: the same draws from there on.
        for k in (1, 5, 19):
            later = unsigned(stream_states([seed], [100 + k])[0])
            assert draws(later, 20 - k) == sequence[k:], f"seed {seed}: reload at +{k} changed the draws"
        # Batched with other rows, or reloaded alongside unseeded ones: the same state.
        rows = stream_states([None, seed, 3], [5, 100, 7], generator=torch.Generator().manual_seed(1))
        assert unsigned(rows[1]) == start


def test_streams_differ():
    a = stream_states([1, 2, None, None], [10, 10, 10, 10], generator=torch.Generator().manual_seed(0))
    assert len({unsigned(x) for x in a}) == 4
    assert draws(unsigned(stream_states([1], [0])[0]), 4) != draws(unsigned(stream_states([2], [0])[0]), 4)
    # Uniforms are spread over [0, 1).
    u = draws(unsigned(stream_states([42], [0])[0]), 4096)
    assert 0.0 <= min(u) and max(u) < 1.0 and abs(sum(u) / len(u) - 0.5) < 0.02
