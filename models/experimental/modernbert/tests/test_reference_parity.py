# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The torch reference must reproduce HuggingFace exactly."""

import pytest
import torch

from models.experimental.modernbert.common import build_inputs, load_config, load_torch_model
from models.experimental.modernbert.reference.modernbert import ModernBertModel
from models.experimental.modernbert.tests.pcc_utils import max_abs_err, pcc

PARITY_PCC = 0.9999
# Local attention is only meaningfully exercised well above the band width: the
# band spans [i-64, i+64], so every row sees the whole sequence when seq_len <= 65
# and a broken band is undetectable there.
SEQ_LENS = [256, 512]


def build_reference(config, ablate=None):
    ref = ModernBertModel(config, ablate=ablate)
    ref.eval()
    return ref


def load_into_reference(ref, hf_model):
    """strict=True is the weight-map proof: it raises on any missing, unexpected
    or misshaped key. Notably it enforces that layer 0 has no attn_norm.weight."""
    return ref.load_state_dict(hf_model.state_dict(), strict=True)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def hf_model():
    return load_torch_model()


def test_state_dict_maps_exactly(config, hf_model):
    """Weight mapping is exact - no key renaming, nothing missing or extra."""
    ref = build_reference(config)
    load_into_reference(ref, hf_model)  # raises if mapping is wrong

    hf_keys = set(hf_model.state_dict().keys())
    ref_keys = set(ref.state_dict().keys())
    assert ref_keys == hf_keys, f"missing={hf_keys - ref_keys} extra={ref_keys - hf_keys}"
    # measured: 2 embeddings + 1 final_norm + 5 (layer 0, no attn_norm) + 6*21
    assert len(hf_keys) == 134, f"expected 134 tensors, got {len(hf_keys)}"
    assert "layers.0.attn_norm.weight" not in hf_keys, "layer 0 must use Identity"
    assert "layers.1.attn_norm.weight" in hf_keys, "layers >=1 must have LayerNorm"


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_reference_matches_hf(config, hf_model, seq_len):
    """Final output and every intermediate hidden state match HF."""
    ref = build_reference(config)
    load_into_reference(ref, hf_model)

    ids, mask = build_inputs(seq_len=seq_len)
    with torch.no_grad():
        hf_out = hf_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        ref_out, ref_hidden = ref(ids, mask, output_hidden_states=True)

    final = pcc(hf_out.last_hidden_state, ref_out)
    print(f"\n[seq={seq_len}] final PCC={final:.10f} max_abs_err={max_abs_err(hf_out.last_hidden_state, ref_out):.3e}")

    worst = (1.0, None)
    for i, (h, r) in enumerate(zip(hf_out.hidden_states, ref_hidden)):
        p = pcc(h, r)
        if p < worst[0]:
            worst = (p, i)
        assert p >= PARITY_PCC, f"layer {i} PCC {p:.10f} < {PARITY_PCC}"
    print(f"[seq={seq_len}] worst layer PCC={worst[0]:.10f} at layer {worst[1]}")

    assert final >= PARITY_PCC, f"final PCC {final:.10f} < {PARITY_PCC}"


# ---------------------------------------------------------------------------
# Negative controls. Each breaks exactly one measured quirk; PCC MUST collapse.
# If a control does not trip, the parity test above is blind to that bug class.
# ---------------------------------------------------------------------------

NEGATIVE_CONTROLS = [
    ("NC1_swap_geglu_gate", {"swap_gate": True}, 256),
    ("NC2_wrong_window_65", {"half_window_override": 65 // 2}, 256),
    ("NC3_norm_at_layer0", {"norm_at_layer0": True}, 256),
    ("NC4_qkv_permuted", {"qkv_permute": True}, 256),
    ("NC5_single_rope_theta", {"single_theta": True}, 256),
]


@pytest.mark.parametrize("name,ablate,seq_len", NEGATIVE_CONTROLS)
def test_negative_control_detects_break(config, hf_model, name, ablate, seq_len):
    """A deliberately broken reference must NOT reach the parity threshold."""
    ref = build_reference(config, ablate=ablate)
    # NC3 adds layers.0.attn_norm.weight, which strict loading correctly rejects.
    # That rejection is itself the detection, so allow it as a pass.
    try:
        load_into_reference(ref, hf_model)
    except RuntimeError as e:
        print(f"\n[{name}] detected via strict state_dict rejection: {str(e)[:120]}")
        return

    ids, mask = build_inputs(seq_len=seq_len)
    with torch.no_grad():
        hf_out = hf_model(input_ids=ids, attention_mask=mask)
        ref_out = ref(ids, mask)

    p = pcc(hf_out.last_hidden_state, ref_out)
    print(f"\n[{name}] PCC={p:.10f} (must be < {PARITY_PCC})")
    assert p < PARITY_PCC, f"{name} did NOT change the output - the test suite is blind to this bug"
