# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import torch

from models.demos.deepseek_v3_d_p.tt.v4 import pp_pack


def test_pack_unpack_round_trip():
    streams = [torch.randn(1, 1, 64, 1024) for _ in range(4)]
    packed = pp_pack.pack_streams_torch(streams)
    assert tuple(packed.shape) == (1, 1, 64, 4096)
    back = pp_pack.unpack_streams_torch(packed)
    for a, b in zip(streams, back):
        torch.testing.assert_close(a, b)
    # stream-major on the last dim: stream h occupies columns [h*D_l, (h+1)*D_l)
    torch.testing.assert_close(packed[..., 1024:2048], streams[1])
