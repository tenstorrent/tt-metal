# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN GeGLU MLP vs the torch reference."""

import pytest
import torch

import ttnn
from models.experimental.modernbert.common import build_inputs, load_config, load_torch_model
from models.experimental.modernbert.reference.modernbert import ModernBertModel
from models.experimental.modernbert.tests.pcc_utils import pcc
from models.experimental.modernbert.tt.model_config import ACTIVATIONS_DTYPE
from models.experimental.modernbert.tt.modernbert_mlp import TtnnModernBertMLP
from models.experimental.modernbert.tt.weights import prepare_weights

MLP_PCC = 0.999


@pytest.fixture(scope="module")
def torch_ref():
    config = load_config()
    hf = load_torch_model()
    ref = ModernBertModel(config)
    ref.load_state_dict(hf.state_dict(), strict=True)
    ref.eval()
    return config, ref


def _mlp_input(ref, seq_len, batch_size):
    """Real activations entering the MLP of layer 0, not random noise: the GELU
    is nonlinear, so the input distribution matters."""
    ids, _ = build_inputs(seq_len=seq_len, batch_size=batch_size)
    with torch.no_grad():
        hidden = ref.embeddings(ids)
        return ref.layers[0].mlp_norm(hidden)


@pytest.mark.parametrize("seq_len,batch_size", [(256, 1), (512, 1), (256, 2)])
def test_ttnn_mlp_matches_reference(device, torch_ref, seq_len, batch_size):
    config, ref = torch_ref
    x = _mlp_input(ref, seq_len, batch_size)

    with torch.no_grad():
        expected = ref.layers[0].mlp(x)

    params = prepare_weights(ref, device)
    module = TtnnModernBertMLP(params["layers"][0]["mlp"])

    tt_x = ttnn.from_torch(x, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    out = module(tt_x)
    got = ttnn.to_torch(out).reshape(expected.shape)

    p = pcc(expected, got.float())
    print(f"\n[mlp seq={seq_len} batch={batch_size}] PCC={p:.8f}")
    assert p >= MLP_PCC, f"MLP PCC {p:.8f} < {MLP_PCC}"


def test_negative_control_swapped_gate(device, torch_ref):
    """Applying the activation to the second half instead of the first must
    collapse PCC. This is the single easiest ModernBERT detail to get wrong."""
    config, ref = torch_ref
    x = _mlp_input(ref, 256, 1)

    with torch.no_grad():
        expected = ref.layers[0].mlp(x)

    params = prepare_weights(ref, device)
    tt_x = ttnn.from_torch(x, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)

    mlp = params["layers"][0]["mlp"]
    act_half = ttnn.linear(tt_x, mlp["Wi_act"])
    gate_half = ttnn.linear(tt_x, mlp["Wi_gate"])

    # deliberately inverted: gelu on the gate half instead of the activated half
    swapped = ttnn.mul(ttnn.gelu(gate_half, fast_and_approximate_mode=False), act_half)
    got = ttnn.to_torch(ttnn.linear(swapped, mlp["Wo"])).reshape(expected.shape)

    p = pcc(expected, got.float())
    print(f"\n[NC mlp-swapped-gate] PCC={p:.8f} (must be < {MLP_PCC})")
    assert p < MLP_PCC, "swapping the GeGLU gate did not change the output - test is blind"


def test_gelu_exact_vs_approx_matters(device, torch_ref):
    """Documents whether the tanh-approximate GELU is good enough here."""
    _, ref = torch_ref
    x = _mlp_input(ref, 256, 1)
    with torch.no_grad():
        expected = torch.nn.functional.gelu(x, approximate="none")

    tt_x = ttnn.from_torch(x, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    exact = ttnn.to_torch(ttnn.gelu(tt_x, fast_and_approximate_mode=False)).reshape(expected.shape)
    approx = ttnn.to_torch(ttnn.gelu(tt_x, fast_and_approximate_mode=True)).reshape(expected.shape)

    p_exact = pcc(expected, exact.float())
    p_approx = pcc(expected, approx.float())
    print(f"\n[gelu] exact={p_exact:.8f}  approx={p_approx:.8f}")
    assert p_exact >= 0.999, f"exact gelu unexpectedly poor: {p_exact:.8f}"
