# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading and TP fracturing for the DFlash drafter.

The ``fc`` permutation tests are pure CPU index algebra and are the most valuable tests in
this file: a wrong permutation passes every other drafter test (Milestone 1 feeds
``target_hidden`` from a host fixture, where any consistent ordering "works") and only
produces garbage once Milestone 2 wires up real on-device taps. Proving it algebraically now
is much cheaper than discovering it then.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.blackhole.qwen36.tt.dflash.weights import (
    fc_input_permutation,
    layer_keys,
    permute_fc_input_activation,
    read_state_dict,
)

TP = 8


# ---- fc input permutation (no device) ------------------------------------------------


def test_fc_permutation_is_a_bijection(drafter_cfg):
    perm = fc_input_permutation(TP, drafter_cfg.hidden_size, len(drafter_cfg.target_layer_ids))
    assert perm.shape == (drafter_cfg.target_feature_size,)
    assert torch.equal(perm.sort().values, torch.arange(drafter_cfg.target_feature_size))


def test_fc_permutation_matches_device_tap_layout(drafter_cfg):
    """The permuted-and-contiguously-sharded weight must reproduce the dense ``fc``.

    Emulates the real data flow exactly: chip ``d`` holds ``tap_i[d*640:(d+1)*640]`` for
    every tap ``i`` (because the target's residual stream is ``dim/tp`` sharded), computes a
    partial against its contiguous 3200-row slice of the permuted weight, and the partials
    are summed by the all-reduce. That sum must equal ``fc(concat(taps))``.
    """
    torch.manual_seed(0)
    hidden = drafter_cfg.hidden_size
    n_taps = len(drafter_cfg.target_layer_ids)
    per = hidden // TP
    seq = 4

    # Small integers in float64: every partial sum is exactly representable, so this is a
    # test of index algebra with no floating-point tolerance to argue about. (In float32 the
    # 8-partial sum and the dense sum differ by ~4e-4 purely from accumulation order, which
    # says nothing about whether the permutation is right.)
    def ints(*shape):
        return torch.randint(-4, 5, shape, dtype=torch.int64).to(torch.float64)

    taps = [ints(seq, hidden) for _ in range(n_taps)]
    x = torch.cat(taps, dim=-1)  # what the reference fc consumes
    w = ints(hidden, drafter_cfg.target_feature_size)

    expected = x @ w.T

    perm = fc_input_permutation(TP, hidden, n_taps)
    w_perm = w.index_select(1, perm)

    # Sum of per-device partials, each built from what that device actually holds.
    total = torch.zeros_like(expected)
    shard = n_taps * per
    for d in range(TP):
        act_d = torch.cat([t[:, d * per : (d + 1) * per] for t in taps], dim=-1)
        assert act_d.shape == (seq, shard)
        w_d = w_perm[:, d * shard : (d + 1) * shard]
        total += act_d @ w_d.T

    assert torch.equal(total, expected), (total - expected).abs().max()


def test_permute_activation_agrees_with_device_slices(drafter_cfg):
    """``permute_fc_input_activation`` reorders a fixture activation into per-chip blocks.

    Block ``d`` of the permuted activation must equal what chip ``d`` physically holds.
    """
    torch.manual_seed(0)
    hidden = drafter_cfg.hidden_size
    n_taps = len(drafter_cfg.target_layer_ids)
    per = hidden // TP
    shard = n_taps * per

    taps = [torch.randn(3, hidden) for _ in range(n_taps)]
    x = torch.cat(taps, dim=-1)
    xp = permute_fc_input_activation(x, TP, hidden, n_taps)

    assert xp.shape == x.shape
    for d in range(TP):
        on_chip = torch.cat([t[:, d * per : (d + 1) * per] for t in taps], dim=-1)
        torch.testing.assert_close(xp[:, d * shard : (d + 1) * shard], on_chip)


def test_contiguous_sharding_would_be_wrong(drafter_cfg):
    """Guard the guard: the naive ordering really does differ, so the test above has teeth.

    If sharding the raw concatenation contiguously happened to match the tap layout, the
    permutation would be dead code and its test vacuous.
    """
    perm = fc_input_permutation(TP, drafter_cfg.hidden_size, len(drafter_cfg.target_layer_ids))
    assert not torch.equal(perm, torch.arange(drafter_cfg.target_feature_size))


# ---- checkpoint reading (needs the checkpoint, no device) ----------------------------


def _skip_without_checkpoint():
    try:
        return read_state_dict(keys=["fc.weight"])
    except Exception as exc:
        pytest.skip(f"drafter checkpoint unavailable: {type(exc).__name__}: {exc}")


def test_read_state_dict_is_lazy(drafter_cfg):
    """A single-tensor read must return exactly that tensor, not the whole 3.46 GB."""
    sd = _skip_without_checkpoint()
    assert list(sd) == ["fc.weight"]
    assert sd["fc.weight"].shape == (drafter_cfg.hidden_size, drafter_cfg.target_feature_size)


def test_layer_keys_all_present(drafter_cfg):
    _skip_without_checkpoint()
    for layer_idx in range(drafter_cfg.num_hidden_layers):
        sd = read_state_dict(keys=layer_keys(layer_idx))
        assert len(sd) == 11, sorted(sd)


def test_checkpoint_shapes_match_config(drafter_cfg):
    """Every weight's shape must follow from the config, with no hardcoded dims."""
    _skip_without_checkpoint()
    c = drafter_cfg
    sd = read_state_dict(keys=layer_keys(0) + ["fc.weight", "hidden_norm.weight", "norm.weight"])
    expected = {
        "layers.0.self_attn.q_proj.weight": (c.q_dim, c.hidden_size),
        "layers.0.self_attn.k_proj.weight": (c.kv_dim, c.hidden_size),
        "layers.0.self_attn.v_proj.weight": (c.kv_dim, c.hidden_size),
        "layers.0.self_attn.o_proj.weight": (c.hidden_size, c.q_dim),
        "layers.0.self_attn.q_norm.weight": (c.head_dim,),
        "layers.0.self_attn.k_norm.weight": (c.head_dim,),
        "layers.0.input_layernorm.weight": (c.hidden_size,),
        "layers.0.post_attention_layernorm.weight": (c.hidden_size,),
        "layers.0.mlp.gate_proj.weight": (c.intermediate_size, c.hidden_size),
        "layers.0.mlp.up_proj.weight": (c.intermediate_size, c.hidden_size),
        "layers.0.mlp.down_proj.weight": (c.hidden_size, c.intermediate_size),
        "fc.weight": (c.hidden_size, c.target_feature_size),
        "hidden_norm.weight": (c.hidden_size,),
        "norm.weight": (c.hidden_size,),
    }
    for key, shape in expected.items():
        assert tuple(sd[key].shape) == shape, f"{key}: {tuple(sd[key].shape)} != {shape}"


def test_norm_gains_are_not_pre_offset(drafter_cfg):
    """The drafter's norm gains must be near 1.0, i.e. NOT the target's "+1"-folded form.

    qwen36's own loaders pre-add 1.0 to every gain for its zero-centered RMSNorm. The
    drafter is plain ``Qwen3RMSNorm``, so if a gain ever arrives centred near 0 instead of 1
    someone has applied the target's convention to the drafter's weights.
    """
    _skip_without_checkpoint()
    sd = read_state_dict(keys=["hidden_norm.weight", "norm.weight", "layers.0.input_layernorm.weight"])
    for key, w in sd.items():
        assert w.float().mean() > 0.3, f"{key} mean {w.float().mean():.4f} looks zero-centered"
