# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.functional import swiglu_mlp as reference_swiglu_mlp
from models.experimental.voxtraltts.utils.test_common import load_acoustic_fm_layer_weights_or_skip
from models.experimental.voxtraltts.tt.mlp import VoxtralTTMLP


@torch.no_grad()
def test_voxtral_mlp_pcc(device, reset_seeds):
    """SwiGLU MLP PCC at production FM dims (3072 → 9216) with checkpoint weights."""
    layer_weights = load_acoustic_fm_layer_weights_or_skip(0)
    w1 = layer_weights["feed_forward.w1.weight"]
    w2 = layer_weights["feed_forward.w2.weight"]
    w3 = layer_weights["feed_forward.w3.weight"]

    hidden = int(w1.shape[1])
    intermediate = int(w1.shape[0])
    batch = 1
    seq_len = 64

    torch.manual_seed(0)
    torch_input = torch.randn(batch, 1, seq_len, hidden, dtype=torch.bfloat16)

    reference_output = reference_swiglu_mlp(torch_input.squeeze(1), w1, w2, w3)

    tt_model = VoxtralTTMLP(
        device=device,
        state_dict={
            "mlp.w1.weight": w1,
            "mlp.w2.weight": w2,
            "mlp.w3.weight": w3,
        },
        w1_key="mlp.w1",
        w2_key="mlp.w2",
        w3_key="mlp.w3",
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(1)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc=0.995)
    assert passing, f"Voxtral MLP PCC failed: {pcc_message}"
