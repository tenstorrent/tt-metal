# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verifies the fix to DFlash's growing-context attention mask (see
generate.py's module docstring, "FIXED (was a KNOWN LATENT LIMITATION...)"):
the noise block's query position is now ``context_valid_len_tt + local_row``
(the sequence's true absolute position), not the old ``ctx_len + local_row``
(the FIXED buffer width -- wrong whenever the real context hasn't filled the
whole buffer, which is effectively always during growth).

Three things checked, at configs where ``sliding_window`` is deliberately
made comparable to or smaller than ``ctx_len`` (the regime where the old
formula was provably wrong, unlike every config exercised before this fix,
where ``sliding_window`` comfortably exceeded ``ctx_len``):

1. ``build_attention_mask_additive_device_dynamic`` (eager growing-context
   path) matches a host-torch reference built with TRUE absolute positions.
2. ``build_attention_mask_static_parts`` + ``combine_attention_mask_dynamic``
   (the trace-safe split used by the traced steady-state loop) matches the
   same reference.
3. The OLD (pre-fix) formula is reproduced by hand for one clearly divergent
   config, to concretely demonstrate the bug was real, not hypothetical --
   its output must NOT match the reference, while both fixed paths above must.

Run: pytest models/demos/gemma4/tests/dflash/test_dflash_sliding_window_mask.py -k 1x8 -s
"""

import torch
from loguru import logger

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric

# (ctx_len, q_len, is_causal, sliding_window, start, valid_len) -- start < ctx_len in
# every row (the real growing-context case: the buffer is wider than the real content
# so far), and sliding_window <= ctx_len in most rows (the regime where the old
# ctx_len-relative formula and the new context_valid_len_tt-relative formula disagree).
CASES = [
    # ctx_len, q_len, is_causal, sliding_window, start,  valid_len
    (256, 16, True, 64, 32, 32),  # start << ctx_len, narrow window
    (256, 16, True, 64, 200, 200),  # start close to ctx_len, narrow window
    (256, 16, True, 128, 100, 100),  # start ~ window, mid ctx_len
    (256, 16, False, 64, 32, 32),  # non-causal (full-attention layer type)
    (2048, 16, True, 2048, 500, 500),  # DFlash's real drafter sliding_window=2048, start well below ctx_len
    (
        2048,
        16,
        True,
        2048,
        2032,
        2032,
    ),  # start near ctx_len -- old formula's error term near 0 here (sanity: should agree)
    (256, 16, True, None, 100, 100),  # sliding_window=None (full-attention layer type)
]


def _torch_reference_mask(ctx_len, q_len, is_causal, sliding_window, start, valid_len, q_len_padded=None):
    """Correct reference: query position = start + local_row (TRUE absolute position)."""
    total_real = ctx_len + q_len
    total = ctx_len + (q_len_padded if q_len_padded else q_len)
    query_position = start + torch.arange(q_len)[:, None]
    key_position = torch.arange(total)[None, :]
    visible = torch.ones((q_len, total), dtype=torch.bool)
    if is_causal:
        visible &= key_position <= query_position
    if sliding_window is not None:
        visible &= (query_position - key_position) < sliding_window
        if not is_causal:
            visible &= (key_position - query_position) < sliding_window
    is_valid_context_col = key_position < valid_len
    is_noise_col = key_position >= ctx_len
    visible &= is_valid_context_col | is_noise_col
    if total > total_real:
        visible &= key_position < total_real
    mask = torch.where(visible, torch.zeros(1), torch.full((1,), -1e4))
    return mask.unsqueeze(0).unsqueeze(0)  # [1,1,q_len,total]


def _torch_old_buggy_mask(ctx_len, q_len, is_causal, sliding_window, valid_len):
    """Reproduces the OLD (pre-fix) formula by hand: query position = ctx_len + local_row,
    NOT the true start. Used only to demonstrate the bug was real for at least one config
    where it and the reference diverge."""
    total_real = ctx_len + q_len
    query_position = ctx_len + torch.arange(q_len)[:, None]  # the bug: ctx_len, not start
    key_position = torch.arange(total_real)[None, :]
    visible = torch.ones((q_len, total_real), dtype=torch.bool)
    if is_causal:
        visible &= key_position <= query_position
    if sliding_window is not None:
        visible &= (query_position - key_position) < sliding_window
        if not is_causal:
            visible &= (key_position - query_position) < sliding_window
    is_valid_context_col = key_position < valid_len
    is_noise_col = key_position >= ctx_len
    visible &= is_valid_context_col | is_noise_col
    return visible


@parametrize_mesh_with_fabric([(1, 8)])
def test_dflash_sliding_window_mask_fixed(mesh_device, device_params, reset_seeds):
    from models.demos.gemma4.tt.dflash.attention import (
        build_attention_mask_additive_device_dynamic,
        build_attention_mask_static_parts,
        combine_attention_mask_dynamic,
    )

    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if hasattr(mesh_device, "shape") else None

    def _read(t):
        out = ttnn.to_torch(t, mesh_composer=composer)
        return out[0:1] if composer is not None else out

    all_ok = True
    for ctx_len, q_len, is_causal, sliding_window, start, valid_len in CASES:
        ref = _torch_reference_mask(ctx_len, q_len, is_causal, sliding_window, start, valid_len)
        ref_bf16 = ref.to(torch.bfloat16).float()

        valid_len_tt = ttnn.from_torch(
            torch.tensor([[valid_len]], dtype=torch.int32), device=mesh_device, dtype=ttnn.int32, mesh_mapper=mapper
        )

        # Path 1: eager growing-context builder.
        eager_mask = build_attention_mask_additive_device_dynamic(
            mesh_device, ctx_len, q_len, is_causal, sliding_window, valid_len_tt
        )
        eager_out = _read(eager_mask).float()
        eager_diff = (eager_out - ref_bf16).abs().max().item()

        # Path 2: trace-safe static/dynamic split.
        static_parts = build_attention_mask_static_parts(mesh_device, ctx_len, q_len, is_causal, sliding_window)
        combined_mask = combine_attention_mask_dynamic(static_parts, valid_len_tt)
        combined_out = _read(combined_mask).float()
        combined_diff = (combined_out - ref_bf16).abs().max().item()

        ok = eager_diff == 0.0 and combined_diff == 0.0
        all_ok &= ok
        logger.info(
            f"[sliding-window-mask] ctx_len={ctx_len:>5} q_len={q_len:>3} causal={is_causal!s:<5} "
            f"window={str(sliding_window):>6} start={start:>5} valid_len={valid_len:>5}  "
            f"eager_diff={eager_diff}  combined_diff={combined_diff}  {'PASS' if ok else 'FAIL'}"
        )

        valid_len_tt.deallocate(True)
        eager_mask.deallocate(True)
        combined_mask.deallocate(True)

    assert all_ok, "Fixed sliding-window mask paths diverged from the true-absolute-position reference"


def test_dflash_sliding_window_mask_bug_was_real():
    """Host-only (no device): confirms the OLD formula genuinely disagreed with the
    reference for at least one config -- the bug being fixed was real, not hypothetical."""
    ctx_len, q_len, is_causal, sliding_window, start, valid_len = CASES[0]  # (256, 16, True, 64, 32, 32)
    ref = _torch_reference_mask(ctx_len, q_len, is_causal, sliding_window, start, valid_len)
    ref_visible = ref[0, 0] > -1.0  # 0.0 where visible, -1e4 where not
    old_visible = _torch_old_buggy_mask(ctx_len, q_len, is_causal, sliding_window, valid_len)
    # Old formula only computed total_real columns; compare on that overlap.
    total_real = ctx_len + q_len
    mismatch = (ref_visible[:, :total_real] != old_visible).sum().item()
    logger.info(f"[sliding-window-mask] old-formula vs reference mismatched cells: {mismatch} (expected > 0)")
    assert mismatch > 0, "Expected the old ctx_len-relative formula to diverge from the true-position reference here"
