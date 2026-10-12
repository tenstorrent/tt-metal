# SPDX-License-Identifier: MIT
"""End-to-end model test: full 33-layer ESM-2 on device with the real checkpoint."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.experimental.esm2.tt.esm2.reference_layers import Esm2Model
from models.experimental.esm2.tt.esm2.ttnn_backend import TtnnEsm2

TEST_SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
MASKED_POSITIONS = [5, 21, 37]
HIDDEN_NRMSE_GATE = 0.035
LOGITS_NRMSE_GATE = 0.035


def _load_vocab(path):
    vocab = {}
    with open(path) as f:
        for i, line in enumerate(f):
            token = line.strip()
            if token:
                vocab[token] = i
    return vocab


def _tokenize(sequence, vocab):
    ids = [0]
    for c in sequence:
        ids.append(vocab.get(c, vocab.get("<unk>", 0)))
    ids.append(2)
    return ids


def _row_nrmse(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b**2)), 1e-12))


@pytest.fixture(scope="module")
def ref_model(config, weights):
    m = Esm2Model(config, weights)
    m.eval()
    return m


@pytest.fixture(scope="module")
def vocab(checkpoint):
    return _load_vocab(str(checkpoint / "vocab.txt"))


@pytest.fixture(scope="module")
def input_ids(config, vocab):
    return torch.tensor([_tokenize(TEST_SEQUENCE, vocab)], dtype=torch.long)


@pytest.fixture(scope="module")
def masked_ids(config, input_ids):
    ids = input_ids.clone()
    for pos in MASKED_POSITIONS:
        ids[0, pos + 1] = config.mask_token_id
    return ids


@pytest.fixture(scope="module")
def attention_mask(input_ids):
    return torch.ones_like(input_ids)


@pytest.fixture(scope="module")
def reference_logits(ref_model, masked_ids, attention_mask):
    with torch.no_grad():
        logits, hidden = ref_model(masked_ids, attention_mask)
    return logits, hidden


@pytest.fixture(scope="module")
def backend(config, weights, device):
    b = TtnnEsm2(config, weights, device=device, precision="bf16")
    b.build()
    return b


@pytest.fixture(scope="module")
def backend_output(backend, masked_ids, attention_mask):
    return backend.forward(masked_ids, attention_mask)


class TestEndToEnd:
    def test_hidden_nrmse(self, backend_output, reference_logits):
        _, ref_hidden = reference_logits
        got = np.asarray(backend_output["hidden"])
        nrmse = _row_nrmse(got, ref_hidden.numpy())
        print(f"\n[e2e] hidden NRMSE = {nrmse:.6f} (gate {HIDDEN_NRMSE_GATE})")
        assert nrmse <= HIDDEN_NRMSE_GATE, f"hidden NRMSE {nrmse:.6f} > {HIDDEN_NRMSE_GATE}"

    def test_logits_nrmse(self, backend_output, reference_logits):
        ref_logits, _ = reference_logits
        got = np.asarray(backend_output["logits"])
        nrmse = _row_nrmse(got, ref_logits.numpy())
        print(f"\n[e2e] logits NRMSE = {nrmse:.6f} (gate {LOGITS_NRMSE_GATE})")
        assert nrmse <= LOGITS_NRMSE_GATE, f"logits NRMSE {nrmse:.6f} > {LOGITS_NRMSE_GATE}"

    def test_masked_argmax(self, backend_output, reference_logits):
        ref_logits, _ = reference_logits
        got_logits = np.asarray(backend_output["logits"])
        for pos in MASKED_POSITIONS:
            idx = pos + 1  # +1 for <cls>
            expected = int(ref_logits[0, idx].argmax())
            got = int(got_logits[0, idx].argmax())
            assert got == expected, f"masked pos {pos}: got {got}, expected {expected}"
