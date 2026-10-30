# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN embeddings vs the torch reference."""

import pytest
import torch

import ttnn
from models.experimental.modernbert.common import build_inputs, load_config, load_torch_model
from models.experimental.modernbert.reference.modernbert import ModernBertModel
from models.experimental.modernbert.tests.pcc_utils import outlier_report, pcc
from models.experimental.modernbert.tt.modernbert_embeddings import TtnnModernBertEmbeddings
from models.experimental.modernbert.tt.weights import prepare_weights

EMBEDDINGS_PCC = 0.999


@pytest.fixture(scope="module")
def torch_ref():
    config = load_config()
    hf = load_torch_model()
    ref = ModernBertModel(config)
    ref.load_state_dict(hf.state_dict(), strict=True)
    ref.eval()
    return config, ref


def _to_device_ids(ids, device):
    return ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


@pytest.mark.parametrize("seq_len", [256, 512])
def test_ttnn_embeddings_matches_reference(device, torch_ref, seq_len):
    config, ref = torch_ref
    ids, _ = build_inputs(seq_len=seq_len)

    with torch.no_grad():
        expected = ref.embeddings(ids)

    params = prepare_weights(ref, device)
    module = TtnnModernBertEmbeddings(params["embeddings"], config)
    out = module(_to_device_ids(ids, device))
    got = ttnn.to_torch(out).reshape(expected.shape)

    p = pcc(expected, got.float())
    print(f"\n[embeddings seq={seq_len}] PCC={p:.8f}")
    print(f"  outliers: {outlier_report(expected)}")
    assert p >= EMBEDDINGS_PCC, f"embeddings PCC {p:.8f} < {EMBEDDINGS_PCC}"


def test_negative_control_missing_layernorm(device, torch_ref):
    """Skipping the LayerNorm must collapse PCC."""
    config, ref = torch_ref
    ids, _ = build_inputs(seq_len=256)

    with torch.no_grad():
        expected = ref.embeddings(ids)

    params = prepare_weights(ref, device)
    tt_ids = _to_device_ids(ids, device)

    # embedding lookup only - deliberately omit the LayerNorm
    raw = ttnn.embedding(tt_ids, params["embeddings"]["tok_embeddings"], layout=ttnn.TILE_LAYOUT)
    got = ttnn.to_torch(raw).reshape(expected.shape)

    p = pcc(expected, got.float())
    print(f"\n[NC embeddings-without-layernorm] PCC={p:.8f} (must be < {EMBEDDINGS_PCC})")
    assert p < EMBEDDINGS_PCC, "dropping LayerNorm did not change the output - test is blind"


def test_weight_prep_rejects_wrong_tensor_count(torch_ref, expect_error):
    """prepare_weights guards on the 134-tensor invariant."""
    _, ref = torch_ref
    import torch.nn as nn

    broken = ModernBertModel(load_config())
    broken.load_state_dict(ref.state_dict(), strict=True)
    # add a stray parameter so the count no longer matches
    broken.extra = nn.Parameter(torch.zeros(1))
    with expect_error(ValueError, "expected 134 tensors"):
        prepare_weights(broken, None)
