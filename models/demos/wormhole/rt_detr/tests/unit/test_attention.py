# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import torch
import ttnn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from tt.attention import self_attention, multihead_attention
from tt.weight_utils import Params, _linear_params
from models.common.utility_functions import comp_pcc

REF = Path(__file__).parent.parent / "reference_outputs.pt"
PCC_THRESHOLD = 0.99


@pytest.fixture(scope="module")
def ref():
    return torch.load(REF, map_location="cpu")


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    yield dev
    ttnn.close_device(dev)


def _attn_params_from_ref(ref, device):
    """Build attention Params from the saved AIFI MHA weight tensors."""
    return Params(
        q_proj  =_linear_params(ref["attn_q_weight"], ref["attn_q_bias"],   device),
        k_proj  =_linear_params(ref["attn_k_weight"], ref["attn_k_bias"],   device),
        v_proj  =_linear_params(ref["attn_v_weight"], ref["attn_v_bias"],   device),
        out_proj=_linear_params(ref["attn_out_weight"], ref["attn_out_bias"], device),
    )


def _to_device(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
    )


class TestMultiheadAttention:
    def test_self_attention_pcc(self, ref, device):
        """AIFI self-attention: 400 tokens, real weights, compare vs torch ref."""
        params = _attn_params_from_ref(ref, device)
        x_tt = _to_device(ref["attn_input"], device)

        out_tt = self_attention(x_tt, params, device, num_heads=ref["attn_num_heads"].item())
        out = ttnn.to_torch(out_tt).squeeze(1)

        pcc, msg = comp_pcc(ref["attn_self_output"], out, PCC_THRESHOLD)
        print(f"\nself_attention PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"self_attention PCC {pcc:.4f} < {PCC_THRESHOLD} — {msg}"

    def test_decoder_self_attention_pcc(self, ref, device):
        """Decoder self-attention path: 300 queries, validates shape and no runtime error.

        Uses AIFI weights (same hidden dim=256, same num_heads=8) with the
        decoder init query as input. This confirms multihead_attention handles
        300-token input correctly. PCC against decoder weights requires saving
        them in generate_reference_outputs.py — tracked as pre-Stage-2 task.
        """
        params = _attn_params_from_ref(ref, device)

        # decoder_init_query shape: (1, 1, 300, 256)
        q = ref["decoder_init_query"]
        q_tt = _to_device(q, device)

        out_tt = self_attention(q_tt, params, device, num_heads=ref["attn_num_heads"].item())
        out = ttnn.to_torch(out_tt).squeeze(1)

        assert out.shape == (1, 300, 256), \
            f"expected (1, 300, 256), got {tuple(out.shape)}"
        print(f"\ndecoder self-attention shape: {tuple(out.shape)} — correct")

    def test_self_attention_is_deterministic(self, ref, device):
        """Same input should produce identical output on two runs."""
        params = _attn_params_from_ref(ref, device)
        x_tt = _to_device(ref["attn_input"], device)

        out1 = ttnn.to_torch(self_attention(x_tt, params, device, num_heads=ref["attn_num_heads"].item()))
        out2 = ttnn.to_torch(self_attention(x_tt, params, device, num_heads=ref["attn_num_heads"].item()))

        max_diff = (out1 - out2).abs().max().item()
        print(f"\ndeterminism max |diff|: {max_diff:.2e}")
        assert max_diff == 0.0, f"non-deterministic output, max diff: {max_diff:.2e}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        r = torch.load(REF, map_location="cpu")
        t = TestMultiheadAttention()
        t.test_self_attention_pcc(r, dev)
        t.test_decoder_self_attention_pcc(r, dev)
        t.test_self_attention_is_deterministic(r, dev)
        print("passed")
    finally:
        ttnn.close_device(dev)