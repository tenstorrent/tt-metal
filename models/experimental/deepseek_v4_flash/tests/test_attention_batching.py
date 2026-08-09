# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Batch-invariance test for the ttnn ``DeepSeekV4Attention`` decode path.

``test_attention_real_weights.py`` checks the block against an HF full-prefill, which
answers "is the arithmetic right" but bounds the answer at the accuracy of a bf4 weight
path (~0.96 PCC). That is too loose to say anything sharp about *batching*, and it cannot
cover a CSA layer at all: the HF CSA module carries Lightning-Indexer weights this
checkpoint does not ship, so its strict state_dict load fails.

This test asks the narrower question that batching actually raises, and asks it of every
layer kind: **a B-user step must hand each user exactly what a ``B == 1`` run of that
user's own tokens hands it.** Both sides are the same ttnn block on the same weights, so
the comparison is tight (unlike the reference PCC, which is dominated by bf4 error) and it
fails loudly on the two ways batching goes wrong -- users leaking into each other through a
shared tile-row, and per-user cache state landing in the wrong slot.

No HF reference and no system-interpreter subprocess: the config comes off the checkpoint's
``config.json`` and the RoPE tables are synthesised, which is sound here because both sides
consume the identical tables.

Run (ttnn venv)::

    pytest -s models/experimental/deepseek_v4_flash/tests/test_attention_batching.py
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tt.attention import (
    DeepSeekV4Attention,
    build_static_layer_cache,
    host_decode_mask,
    int32_pos_tensor,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

from models.experimental.deepseek_v4_flash.tests.test_attention_real_weights import (
    _DEFAULT_MODEL_DIR,
    _WEIGHT_DTYPE,
    _build_attn_weights,
    _checkpoint_available,
    _to_tt,
    _weight_cache,
)

# The two sides run the same ops on the same weights, so they should agree to numerical
# noise rather than to model accuracy. Not asserted as exact equality because a batched
# step does not decompose into the same reductions: SDPA-decode splits its KV reduction
# over cores per (head, user), so the batched run sums the same terms in a different order.
INVARIANCE_PCC = 0.999


def _config_from_checkpoint(loader: DeepseekV4WeightLoader):
    """The handful of attributes :class:`DeepSeekV4Attention` reads, off ``config.json``.

    Avoids ``transformers`` entirely (the cached install with ``deepseek_v4`` imports only
    under the system interpreter). ``layer_types`` / ``compress_rates`` are not stored
    directly: the checkpoint carries a per-layer ``compress_ratios``, of which 0 means a
    sliding-only layer and the two non-zero values are the CSA and HCA compress rates.
    """
    with (Path(loader.snapshot_dir) / "config.json").open() as fh:
        cfg = json.load(fh)

    ratios = cfg["compress_ratios"]
    non_zero = sorted({r for r in ratios if r})
    assert len(non_zero) == 2, f"expected one CSA and one HCA compress rate, got {non_zero}"
    csa_rate, hca_rate = non_zero
    kind = {0: "sliding_attention", csa_rate: "compressed_sparse_attention", hca_rate: "heavily_compressed_attention"}

    return type(
        "AttnConfig",
        (),
        {
            "hidden_size": cfg["hidden_size"],
            "num_attention_heads": cfg["num_attention_heads"],
            "head_dim": cfg["head_dim"],
            "qk_rope_head_dim": cfg["qk_rope_head_dim"],
            "o_groups": cfg["o_groups"],
            "o_lora_rank": cfg["o_lora_rank"],
            "rms_norm_eps": cfg["rms_norm_eps"],
            "sliding_window": cfg["sliding_window"],
            "layer_types": [kind[r] for r in ratios],
            "compress_rates": {
                "compressed_sparse_attention": csa_rate,
                "heavily_compressed_attention": hca_rate,
            },
        },
    )()


def _rope_half_tables(positions: torch.Tensor, rope_dim: int, theta: float = 10000.0):
    """``(cos_half, sin_half)`` ``[len(positions), rope_dim // 2]`` for ``positions``.

    The exact table does not matter here -- both sides of the comparison consume the same
    one -- but a real rotary table keeps the values in the range the kernel sees in
    production rather than exercising it on arbitrary magnitudes.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return angles.cos(), angles.sin()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two flattened tensors, for reporting the margin."""
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _rope_rows(cos_half: torch.Tensor, sin_half: torch.Tensor, device):
    cos_full, sin_full = make_rope_table(cos_half, sin_half)
    return _to_tt(cos_full, device), _to_tt(sin_full, device), _to_tt(-sin_full, device)


def _replay_decode(attn, cfg, layer_type, hidden, device, capture_from: int) -> dict[int, torch.Tensor]:
    """Feed ``hidden`` ``[B, S, D]`` through ``decode`` one position at a time.

    Returns the outputs at positions ``>= capture_from`` as ``{pos: [B, D]}``. Each call
    builds its own cache, so a ``B == 1`` replay is a genuinely independent session rather
    than a slice of the batched one.
    """
    batch, seq_len, hidden_size = hidden.shape
    is_compressor = layer_type != "sliding_attention"
    cr = cfg.compress_rates[layer_type] if is_compressor else None

    kv_cache = build_static_layer_cache(
        device, cfg.sliding_window, layer_type, cfg.head_dim, seq_len, cfg.compress_rates, batch=batch
    )

    cos_half, sin_half = _rope_half_tables(torch.arange(seq_len), cfg.qk_rope_head_dim)
    if is_compressor:
        win_cos_half, win_sin_half = _rope_half_tables(torch.arange(seq_len // cr + 1) * cr, cfg.qk_rope_head_dim)

    captured: dict[int, torch.Tensor] = {}
    for pos in range(seq_len):
        attn.prefetch_weights()
        cos_d, sin_d, neg_sin_d = _rope_rows(cos_half[pos : pos + 1], sin_half[pos : pos + 1], device)

        cos_win_d = sin_win_d = win_slot = win_row = None
        pool = False
        if is_compressor:
            wi = max((pos + 1) // cr - 1, 0)
            pool = (pos + 1) % cr == 0
            cw, sw = make_rope_table(win_cos_half[wi : wi + 1], win_sin_half[wi : wi + 1])
            cos_win_d = _to_tt(cw, device)
            sin_win_d = _to_tt(sw, device)
            win_slot = int32_pos_tensor(pos % cr, device, batch)
            win_row = int32_pos_tensor(cfg.sliding_window + wi, device, batch)

        out_tt = attn.decode(
            _to_tt(hidden[:, pos : pos + 1].reshape(batch, 1, 1, hidden_size), device),
            cos_d,
            sin_d,
            neg_sin_d,
            cos_win_d,
            sin_win_d,
            host_decode_mask(cfg.sliding_window, layer_type, cr, pos, seq_len, device),
            kv_cache,
            int32_pos_tensor(pos % cfg.sliding_window, device, batch),
            int32_pos_tensor(pos, device, batch),
            pool_compressor=pool,
            win_slot=win_slot,
            win_row=win_row,
        )
        if pos >= capture_from:
            captured[pos] = ttnn.to_torch(out_tt).reshape(batch, hidden_size).to(torch.float32)
    return captured


def _build_attention(device, layer_idx: int):
    """``(attn, cfg, layer_type, use_prefetcher)`` for one layer off the real checkpoint."""
    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    cfg = _config_from_checkpoint(loader)
    layer_type = cfg.layer_types[layer_idx]
    logger.info(f"layer {layer_idx} is {layer_type}")

    use_prefetcher = ttnn.experimental.is_tensor_prefetcher_supported(device)
    attn = DeepSeekV4Attention(
        cfg,
        layer_idx,
        _build_attn_weights(loader, layer_idx, layer_type),
        device,
        cache=_weight_cache(layer_idx),
        weight_dtype=_WEIGHT_DTYPE,
        use_prefetcher=use_prefetcher,
    )
    return attn, cfg, layer_type, use_prefetcher


# One layer of each kind. 2 = CSA (compress_rate 4, so a 16-token run closes four windows
# and exercises the Ca/Cb overlap and the per-user window retire); 5 = HCA (compress_rate
# 128, whose single window closes on the last step of a 128-token run).
_LAYER_CASES = dict(
    argnames="layer_idx, seq_len", argvalues=((1, 16), (2, 16), (5, 128)), ids=["sliding", "csa", "hca"]
)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(**_LAYER_CASES)
@pytest.mark.parametrize("batch_size", (8,))
def test_attention_decode_batch_invariance(device, reset_seeds, layer_idx: int, seq_len: int, batch_size: int) -> None:
    """A batched decode step must reproduce every user's own ``B == 1`` answer.

    Replays ``batch_size`` independent sequences through one batched run, then replays each
    of them alone, and compares the last few positions user by user. The per-user replay is
    the control: same block, same weights, same RoPE tables, same positions -- the only
    difference is how many users share the step.

    Changing the batch size does move the arithmetic a little: SDPA-decode splits its KV
    reduction across cores per (head, user), so a B-user step merges the flash-attention
    partials in a different order than a one-user step and the bf16 rounding differs. That
    shows up as a few percent of relative RMS on top of a PCC well past 0.999. It is the
    batch *size* that does it, not which user sits where --
    :func:`test_attention_decode_user_permutation_invariance` pins that down by holding the
    batch size fixed.
    """
    attn, cfg, layer_type, use_prefetcher = _build_attention(device, layer_idx)

    torch.manual_seed(1234)
    hidden = torch.randn(batch_size, seq_len, cfg.hidden_size)
    # Only the tail is compared; the earlier steps exist to build up cache state (and, on a
    # compressor layer, to close at least one window) so the comparison covers a populated
    # cache rather than a cold one.
    capture_from = max(seq_len - 4, 0)

    with contextlib.ExitStack() as prefetcher:
        if use_prefetcher:
            prefetcher.enter_context(tensor_prefetcher_session(device))
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)

        batched = _replay_decode(attn, cfg, layer_type, hidden, device, capture_from)
        single = [
            _replay_decode(attn, cfg, layer_type, hidden[user : user + 1], device, capture_from)
            for user in range(batch_size)
        ]

    worst_pcc, worst_rel = 1.0, 0.0
    for pos in sorted(batched):
        for user in range(batch_size):
            got = batched[pos][user]
            want = single[user][pos][0]
            passing, pcc_message = comp_pcc(want, got, pcc=INVARIANCE_PCC)
            # Reported alongside PCC because PCC is scale-free: it would not notice the two
            # sides differing by a constant factor, which a mis-shared reduction could cause.
            # RMS rather than peak, because a single outlier channel out of 4096 says little.
            relative = ((want - got).pow(2).mean().sqrt() / want.pow(2).mean().sqrt()).item()
            worst_pcc, worst_rel = min(worst_pcc, _pcc(want, got)), max(worst_rel, relative)
            assert passing, (
                f"layer {layer_idx} ({layer_type}) pos {pos} user {user}: a batch of {batch_size} did not "
                f"reproduce this user's B==1 output (relative RMS {relative:.3%}): {pcc_message}"
            )
    logger.info(
        f"layer {layer_idx} ({layer_type}) batch-invariance holds over {len(batched)} positions "
        f"x {batch_size} users; worst PCC {worst_pcc:.6f}, worst relative RMS {worst_rel:.3%}"
    )


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(**_LAYER_CASES)
# 8 is the supported ceiling, and it is L1 that sets it rather than the 32-user tile-row cap
# in ``_pack_tokens``: the width-sharded query is the one thing that grows with B, and by 16
# users it collides with SDPA-decode's static circular buffers. See the attention.py header.
@pytest.mark.parametrize("batch_size", (8,))
def test_attention_decode_user_independence(device, reset_seeds, layer_idx: int, seq_len: int, batch_size: int) -> None:
    """A user's answer must not depend on what the *other* users in the step are decoding.

    Replays the batch twice with slot 0 carrying the identical sequence both times and every
    other slot re-randomised, then compares slot 0 against itself. This is the sharp form of
    "the users of a step are independent": it isolates cross-user contamination from the two
    things that legitimately perturb the arithmetic -- the batch size (which changes how
    SDPA-decode splits its KV reduction) and the slot index (which changes which cores serve
    that user) -- by holding both fixed. Nothing is shared between users in a correct
    implementation, so this is asserted as bit equality rather than a tolerance: the users sit
    on separate rows of the packed tile, separate rows of every cache, and separate SDPA
    batch slots, and any coupling at all is a bug.
    """
    attn, cfg, layer_type, use_prefetcher = _build_attention(device, layer_idx)

    torch.manual_seed(1234)
    hidden = torch.randn(batch_size, seq_len, cfg.hidden_size)
    other = torch.randn(batch_size, seq_len, cfg.hidden_size)
    other[0] = hidden[0]  # slot 0 unchanged; every other slot decodes something else
    capture_from = max(seq_len - 4, 0)

    with contextlib.ExitStack() as prefetcher:
        if use_prefetcher:
            prefetcher.enter_context(tensor_prefetcher_session(device))
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)

        first = _replay_decode(attn, cfg, layer_type, hidden, device, capture_from)
        second = _replay_decode(attn, cfg, layer_type, other, device, capture_from)

    for pos in sorted(first):
        want, got = first[pos][0], second[pos][0]
        relative = ((want - got).pow(2).mean().sqrt() / want.pow(2).mean().sqrt()).item()
        assert torch.equal(want, got), (
            f"layer {layer_idx} ({layer_type}) pos {pos}: slot 0's output moved by {relative:.3%} "
            f"relative RMS when only the other users' tokens changed -- users are not independent"
        )
    logger.info(
        f"layer {layer_idx} ({layer_type}) users are independent: slot 0 is bit-identical across "
        f"{len(first)} positions while the other {batch_size - 1} users decode different tokens"
    )
