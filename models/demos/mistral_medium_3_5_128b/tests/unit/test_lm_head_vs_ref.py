# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M2 row: the column-parallel lm head vs ``torch.nn.Linear``.

Full width on the target mesh: ``[12288, 131072]`` sharded over the TP cols, 32768 vocab columns
per chip, sequence SP-sharded over the rows. The head is the only weight in the model whose output
dimension is the vocab, so it is the only place a vocab-sharding mistake can appear. A wrong shard
*order* is caught by the main PCC row rather than by a separate test: the comparison is
element-wise against the reference at the same token ids, so a permuted quarter puts unrelated
values at every index and PCC collapses to ~0. (It would *not* be caught by comparing sorted
logits, or by any statistic over the value distribution — the same 131072 numbers are present
either way. Comparing at the id is the whole point.)

What does need its own row is the **top of the distribution**, which is where bf8 weights cost the
most and where the head's consumer looks. With random weights the top-1 and top-2 of 131072
near-Gaussian logits are separated by ~0.2 sigma on average, so a 0.02-sigma perturbation flips
about 11% of positions — measured 0.887 agreement here, which is the predicted value and not a
defect. :func:`test_top1_disagreements_are_near_ties` states the property that actually matters:
where the argmax moves, it moves between near-ties.

The sequence is short (256 tokens). This is the one op where the host reference is the expensive
side: ``256 x 12288 x 131072`` is 412 GFLOP in torch, against a matmul the mesh does not notice.
The vocab dimension is what is under test and it is at full size; the sequence dimension is not.

The head is bf8 on device against a bf16 reference, so unlike the embedding row this is a PCC
comparison, at the spec's bounds.
"""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.modeling import REF_DTYPE
from models.demos.mistral_medium_3_5_128b.tests.device_utils import assert_pcc, from_mesh_2d, to_mesh
from models.demos.mistral_medium_3_5_128b.tt.lm_head import LMHead

SEQ = 256  # multiple of TILE_SIZE * sp = 256; see the module docstring on why it is small


@pytest.fixture(scope="module")
def head_weight(cfg):
    """``[vocab_size, hidden_size]`` in HF orientation, scaled like a trained head."""
    torch.manual_seed(2)
    return (torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.float32) / cfg.hidden_size**0.5).to(REF_DTYPE)


@pytest.fixture(scope="module")
def activations(cfg):
    torch.manual_seed(3)
    return (torch.randn(1, 1, SEQ, cfg.hidden_size) * 0.5).to(REF_DTYPE)


@pytest.fixture(scope="module")
def reference(head_weight, activations):
    """``x @ W.T`` in fp32 — the head is linear, so there is nothing else to model."""
    return (activations.float() @ head_weight.float().T).to(REF_DTYPE)


@pytest.fixture(scope="module")
def device_logits(galaxy, mesh_config, ccl, cfg, head_weight, activations):
    """Full ``[1, 1, SEQ, vocab_size]`` logits, gathered back off the mesh once."""
    head = LMHead(galaxy, cfg, {"weight": head_weight}, mesh_config, ccl)
    sharded = head(to_mesh(galaxy, activations, dims=[-2, None]))
    # The shards come back with the sequence on the SP rows and the vocab on the TP cols.
    return from_mesh_2d(galaxy, sharded, dims=(2, 3))


def test_lm_head_vs_ref(cfg, reference, device_logits):
    """Logits at full vocab width against the torch reference."""
    assert device_logits.shape == (1, 1, SEQ, cfg.vocab_size), tuple(device_logits.shape)
    assert_pcc("lm_head", reference, device_logits)


#: How far below the reference's own maximum the device's pick may fall, in units of the logit
#: standard deviation. Measured here: mean 0.0072, max 0.1875 over 256 positions, against a PCC of
#: 0.9998 (a per-logit perturbation of ~0.02 sigma, so ~0.028 sigma between two of them and a
#: worst-of-256 tail a few times that). The bound is set at roughly twice the measured maximum,
#: which is loose enough to survive a different random seed and far too tight for a head that is
#: systematically wrong about any token.
MAX_TIE_GAP_SIGMA = 0.4


def test_top1_disagreements_are_near_ties(reference, device_logits):
    """Where bf8 moves the argmax, it moves it between logits that were nearly equal.

    This is the statement a sampler cares about and the one a value-level PCC cannot make: 0.9998
    PCC is compatible both with "the top of the distribution is preserved" and with "the head is
    systematically wrong about a subset of tokens". Measuring the reference's own gap at the
    device's choice separates them.
    """
    ref = reference.float()
    ref_top = ref.argmax(dim=-1)
    out_top = device_logits.float().argmax(dim=-1)
    agree = (ref_top == out_top).float().mean().item()

    sigma = ref.std().item()
    chosen = ref.gather(-1, out_top.unsqueeze(-1)).squeeze(-1)
    gap = (ref.max(dim=-1).values - chosen) / sigma
    print(
        f"lm_head top-1 agreement {agree:.4f} over {ref_top.numel()} positions; "
        f"reference gap at the device's pick: mean {gap.mean():.4f} max {gap.max():.4f} sigma"
    )
    assert gap.max().item() < MAX_TIE_GAP_SIGMA, (
        f"the device picked a token {gap.max():.3f} sigma below the reference's best — that is a "
        f"systematic head error, not a near-tie"
    )


def test_gather_is_a_permutation_free_concat(galaxy, mesh_config, ccl, cfg, head_weight, activations):
    """``gather`` on device must give the same tensor as composing the shards on host.

    :meth:`~...tt.lm_head.LMHead.gather` is used by anything that wants full-width logits inside a
    run; every other test here composes on the host instead. If the two disagree, one of them is
    wrong and the choice of read-back path would silently change results.
    """
    head = LMHead(galaxy, cfg, {"weight": head_weight}, mesh_config, ccl)
    sharded = head(to_mesh(galaxy, activations, dims=[-2, None]))
    host_composed = from_mesh_2d(galaxy, sharded, dims=(2, 3))

    gathered = head.gather(sharded)
    assert gathered.shape[-1] == cfg.vocab_size, tuple(gathered.shape)
    # After the TP all-gather every col holds the full vocab, so the cols are replicas.
    on_device = from_mesh_2d(galaxy, gathered, dims=(2, 1))[:, :1]
    torch.testing.assert_close(on_device, host_composed, rtol=0, atol=0)
