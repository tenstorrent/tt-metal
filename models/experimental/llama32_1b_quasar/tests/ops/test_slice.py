# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.slice``.

Model call sites (models/llama32_1b/model.py):
  * L906  forward           — last-token slice before the LM head
  * L919  post_process_prefill_output — same pattern

Both trim a prefill hidden-state tensor [1, 1, seq, DIM] to the single
tile-row window that contains the last token:

    get_last_token_floor = (get_last_token // 32) * 32
    x = ttnn.slice(x, (0, 0, floor, 0), (1, 1, floor + 32, x.shape[-1]))

``ttnn.slice`` end coords are exclusive, so this yields a [1, 1, 32, DIM] window.
Value-preserving -> torch reference is a plain narrow; PCC 0.999.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# (id, seq_len, last_token) grounded in real prefill sequence lengths (U.PREFILL_SEQ_LENS).
_SLICE_SITES = [
    ("seq128_last127", 128, 127),
    ("seq512_last200", 512, 200),
    ("seq1024_last1023", 1024, 1023),
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, seq_len, last_token",
    [pytest.param(*s, id=s[0]) for s in _SLICE_SITES],
)
def test_slice(ttnn_mesh_device, reset_seeds, name, seq_len, last_token):
    mesh = ttnn_mesh_device
    floor = (last_token // 32) * 32

    x_torch = U.torch_rand((1, 1, seq_len, U.DIM))
    x = U.to_tt(x_torch, mesh)

    out = ttnn.slice(x, (0, 0, floor, 0), (1, 1, floor + 32, x.shape[-1]))

    ref = x_torch.float()[0:1, 0:1, floor : floor + 32, : U.DIM]
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)
