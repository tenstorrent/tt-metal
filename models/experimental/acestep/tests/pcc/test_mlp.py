# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 MLP vs TTTv2 MLP1D (pure reuse, no custom class).

ACE-Step DiT and encoder layers use Qwen3MLP: SwiGLU with gate/up/down projections
and SiLU activation, no bias. TTTv2 MLP1D(w1=gate, w2=down, w3=up) with SILU is a
direct match. Validated against the real HF Qwen3MLP at ACE-Step's dims.

Weight layout: HF nn.Linear stores [out, in]. MLP1D expects w1/w3 as (dim, hidden_dim)
and w2 as (hidden_dim, dim) -> transpose HF weights on load.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from models.common.modules.mlp.mlp_1d import MLP1D
from models.experimental.acestep.tests.test_utils import (
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    SEQUENCE_LENGTHS,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

BATCH_SIZE = 1


@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{seq_len}" for seq_len in SEQUENCE_LENGTHS])
def test_mlp_vs_qwen3(device, seq_len):
    require_single_device(device)
    torch.manual_seed(42)

    cfg = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act="silu",
    )
    reference_layer = Qwen3MLP(cfg).eval()
    with torch.no_grad():
        for lin in (reference_layer.gate_proj, reference_layer.up_proj, reference_layer.down_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))

    x = torch.randn((BATCH_SIZE, 1, seq_len, HIDDEN_SIZE), dtype=torch.float32)

    # HF Linear weight is [out, in]; MLP1D wants [in, out] -> transpose.
    def w(t):
        return make_lazy_weight(
            t.detach().clone().transpose(-1, -2).contiguous(),
            device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    tt_model = MLP1D(
        w1=w(reference_layer.gate_proj.weight),
        w2=w(reference_layer.down_proj.weight),
        w3=w(reference_layer.up_proj.weight),
    )

    tt_output = tt_model.forward(to_ttnn_tensor(x, device), mode="prefill")
    tt_output_torch = to_torch(tt_output, expected_shape=(BATCH_SIZE, 1, seq_len, HIDDEN_SIZE))

    with torch.no_grad():
        reference_output = reference_layer(x.to(torch.float32)).to(torch.float32)

    assert_pcc(reference_output, tt_output_torch, 0.999)
