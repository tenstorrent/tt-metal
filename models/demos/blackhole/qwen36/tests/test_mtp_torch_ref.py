# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only guard on the MTP torch reference (tests/mtp_torch_ref.py).

The acceptance oracle in mtp_cpu_check.py draws its conclusions from two code paths that must
agree: the batched causal ``forward_sequence`` (used for the warmed-prefix KV and the depth-1
sweep) and the incremental ``forward_step`` (used for the autoregressive draft chain). If they
disagree, every number the oracle prints is suspect — so pin them here.

Random weights at toy dims: no checkpoint, no device, runs in well under a second.

    pytest models/demos/blackhole/qwen36/tests/test_mtp_torch_ref.py -v
"""
import pytest
import torch

from models.demos.blackhole.qwen36.tests.mtp_torch_ref import LAYER, MTPTorchHead

DIM = 64
HEAD_DIM = 16
N_HEADS = 4
N_KV_HEADS = 2
VOCAB = 97
FFN = 128
ROPE_DIM = 8  # partial rope: only the first 8 of each 16-wide head is rotated
ROPE_THETA = 10_000.0


def _random_sd(seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g) * 0.05

    return {
        "tok_embeddings.weight": r(VOCAB, DIM),
        "output.weight": r(VOCAB, DIM),
        "mtp.fc.weight": r(DIM, 2 * DIM),
        "mtp.pre_fc_norm_embedding.weight": r(DIM),
        "mtp.pre_fc_norm_hidden.weight": r(DIM),
        "mtp.norm.weight": r(DIM),
        LAYER + "input_layernorm.weight": r(DIM),
        LAYER + "post_attention_layernorm.weight": r(DIM),
        LAYER + "self_attn.q_proj.weight": r(N_HEADS * 2 * HEAD_DIM, DIM),
        LAYER + "self_attn.k_proj.weight": r(N_KV_HEADS * HEAD_DIM, DIM),
        LAYER + "self_attn.v_proj.weight": r(N_KV_HEADS * HEAD_DIM, DIM),
        LAYER + "self_attn.o_proj.weight": r(DIM, N_HEADS * HEAD_DIM),
        LAYER + "self_attn.q_norm.weight": r(HEAD_DIM),
        LAYER + "self_attn.k_norm.weight": r(HEAD_DIM),
        LAYER + "mlp.gate_proj.weight": r(FFN, DIM),
        LAYER + "mlp.up_proj.weight": r(FFN, DIM),
        LAYER + "mlp.down_proj.weight": r(DIM, FFN),
    }


def _head(chain_postnorm=False):
    return MTPTorchHead(_random_sd(), rope_dim=ROPE_DIM, rope_theta=ROPE_THETA, chain_postnorm=chain_postnorm)


@torch.no_grad()
def test_dims_derived_from_state_dict():
    """Shapes alone must pin dim / head_dim / head counts (the oracle takes no config for these)."""
    head = _head()
    assert (head.dim, head.head_dim) == (DIM, HEAD_DIM)
    assert (head.n_heads, head.n_kv_heads, head.group) == (N_HEADS, N_KV_HEADS, N_HEADS // N_KV_HEADS)
    assert head.vocab_size == VOCAB


@torch.no_grad()
@pytest.mark.parametrize("chain_postnorm", (False, True))
def test_incremental_matches_sequence(chain_postnorm):
    """Stepping slot-by-slot with a growing K/V cache == one causal pass over all slots.

    Under both chain contracts (V0 raw block output, V3 mtp.norm output)."""
    head = _head(chain_postnorm)
    S = 7
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(S, DIM, generator=g)
    tokens = torch.randint(0, VOCAB, (S,), generator=g)

    seq_logits, seq_block, seq_k, seq_v = head.forward_sequence(hidden, tokens)

    k_all = v_all = None
    for i in range(S):
        logits_i, block_i, k_all, v_all = head.forward_step(hidden[i], int(tokens[i]), i, k_all, v_all)
        assert torch.allclose(logits_i, seq_logits[i], atol=1e-5), f"logits mismatch at slot {i}"
        assert torch.allclose(block_i, seq_block[i], atol=1e-5), f"block output mismatch at slot {i}"
    assert torch.allclose(k_all, seq_k, atol=1e-6)
    assert torch.allclose(v_all, seq_v, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("chain_postnorm", (False, True))
def test_step_from_warm_prefix_matches_sequence(chain_postnorm):
    """The oracle's chain pattern: take a warmed prefix cache from forward_sequence, then step.

    Stepping slot P from the prefix cache [0..P-1] must equal row P of a full sequence pass, which
    is what makes 'warm the drafter over the prompt, then draft' a faithful simulation. Under both
    chain contracts (V0 raw block output, V3 mtp.norm output).
    """
    head = _head(chain_postnorm)
    S, P = 9, 6
    g = torch.Generator().manual_seed(2)
    hidden = torch.randn(S, DIM, generator=g)
    tokens = torch.randint(0, VOCAB, (S,), generator=g)

    seq_logits, seq_block, _, _ = head.forward_sequence(hidden, tokens)
    _, _, pk, pv = head.forward_sequence(hidden[:P], tokens[:P], want_logits=False)

    logits_p, block_p, _, _ = head.forward_step(hidden[P], int(tokens[P]), P, pk, pv)
    assert torch.allclose(logits_p, seq_logits[P], atol=1e-5)
    assert torch.allclose(block_p, seq_block[P], atol=1e-5)


@torch.no_grad()
def test_causal_mask_is_strictly_causal():
    """Slot i must not see slot i+1: perturbing a later token cannot change an earlier row."""
    head = _head()
    S = 6
    g = torch.Generator().manual_seed(3)
    hidden = torch.randn(S, DIM, generator=g)
    tokens = torch.randint(0, VOCAB, (S,), generator=g)

    base, _, _, _ = head.forward_sequence(hidden, tokens)
    perturbed_tokens = tokens.clone()
    perturbed_tokens[-1] = (perturbed_tokens[-1] + 1) % VOCAB
    perturbed_hidden = hidden.clone()
    perturbed_hidden[-1] += 1.0
    got, _, _, _ = head.forward_sequence(perturbed_hidden, perturbed_tokens)

    assert torch.allclose(base[:-1], got[:-1], atol=1e-6), "an earlier slot saw a later slot"
    assert not torch.allclose(base[-1], got[-1], atol=1e-3), "the perturbed slot did not change"


@torch.no_grad()
def test_rope_only_sees_relative_position():
    """RoPE rotates q and k alike, so only the OFFSET between a slot and its cache matters.

    Two consequences the oracle depends on:
      * a slot with no prefix (attending only to itself) is position-invariant;
      * the same (hidden, token) against the same prefix at a different offset is NOT.

    This is why a uniform shift of every MTP slot index is harmless, and why only the
    (hidden, token) PAIRING — not the absolute position — is at stake in the alignment question.
    """
    head = _head()
    g = torch.Generator().manual_seed(4)
    hidden = torch.randn(3, DIM, generator=g)

    alone_at_0, _, _, _ = head.forward_step(hidden[0], 5, 0)
    alone_at_7, _, _, _ = head.forward_step(hidden[0], 5, 7)
    assert torch.allclose(alone_at_0, alone_at_7, atol=1e-6), "self-attention should be position-invariant"

    # Same prefix, same (hidden, token), different distance from it.
    _, _, pk, pv = head.forward_sequence(hidden[:2], torch.tensor([11, 12]), want_logits=False)
    near, _, _, _ = head.forward_step(hidden[2], 5, 2, pk, pv)
    far, _, _, _ = head.forward_step(hidden[2], 5, 9, pk, pv)
    assert not torch.allclose(near, far, atol=1e-4), "distance to the cached prefix should matter"
