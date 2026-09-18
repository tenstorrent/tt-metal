# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Run with the target environment's normal TTNN/architecture configuration."""
import pytest
import torch
from tt_bfp_quant import search_linear, search_packed, gptq_search, validate_repacking

ttnn = pytest.importorskip("ttnn")
if not hasattr(ttnn, "from_torch"):
    pytest.skip("Requires an installed TTNN build, not the source namespace", allow_module_level=True)
pytestmark = pytest.mark.ttnn


@pytest.mark.parametrize("bits", [4, 8])
def test_native_pack_and_partial_output_shards(bits):
    torch.manual_seed(20)
    w = torch.randn(80, 65)
    q, _ = search_linear(w, bits, output_splits=[40, 40])
    validate_repacking(q, bits, output_splits=[40, 40], native=True)
    for shard in w.chunk(2):
        expected = ttnn.to_torch(
            ttnn.from_torch(
                shard.T.contiguous(), dtype=ttnn.bfloat4_b if bits == 4 else ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )
        ).float()
        ordinary, _ = search_packed(shard.T, bits, (0,))
        assert torch.equal(expected, ordinary)


def test_gptq_native_packing():
    torch.manual_seed(21)
    w, x = torch.randn(80, 65), torch.randn(256, 65)
    q, _ = gptq_search(w, x.T @ x / len(x), output_splits=[40, 40])
    validate_repacking(q, output_splits=[40, 40], native=True)
