# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Indexed YaRN RoPE on device vs the reference rotation (HF half-split), compared in the Meta layout
the device uses. Covers the whole-cache block-cyclic tables at the one-shot geometry and at later
chunks of a chunked cache, where the per-chip start row is derived on device from kv_actual."""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.model import apply_rope, rope_cos_sin
from models.demos.mistral_medium_3_5_128b.tt.rope import RopeSetup, hf_to_meta_perm, permute_qk_rows

from .common import CFG, assert_pcc, heads_to_torch, randn, to_heads


def test_meta_permutation_commutes_with_rope():
    """Host-only algebra: permuting q/k rows HF -> Meta and rotating Meta-style == rotating HF-style."""
    d = CFG.head_dim
    perm = hf_to_meta_perm(d)
    w = randn(2 * d, 64, seed=5, dtype=torch.float32)
    x = randn(7, 64, seed=6, dtype=torch.float32)
    wm = permute_qk_rows(w, d)
    assert torch.equal(wm.view(2, d, 64), w.view(2, d, 64)[:, perm])
    pos = torch.arange(100, 107)
    cos, sin = rope_cos_sin(CFG, pos, dtype=torch.float32)
    hf = apply_rope((x @ w.t()).view(7, 2, d).transpose(0, 1), cos, sin)
    # Meta rotation of interleaved pairs (2j, 2j+1) with the same angles.
    xm = (x @ wm.t()).view(7, 2, d).transpose(0, 1)
    cm, sm = cos[:, perm], sin[:, perm]
    rot = torch.stack([-xm[..., 1::2], xm[..., 0::2]], dim=-1).flatten(-2)
    assert torch.allclose(xm * cm + rot * sm, hf[..., perm], atol=1e-4)


@pytest.mark.parametrize(
    "max_seq_len, chunk, start",
    [(10240, 10240, 0), (10240, 5120, 5120), (20480, 5120, 10240)],
    ids=["one_shot", "chunk1_of_2", "chunk2_of_4"],
)
def test_indexed_rope_vs_ref(galaxy_mesh, mesh_config, max_seq_len, chunk, start):
    heads = CFG.num_key_value_heads
    x = randn(1, heads, chunk, CFG.head_dim, seed=7)
    cos, sin = rope_cos_sin(CFG, torch.arange(start, start + chunk))
    perm = hf_to_meta_perm(CFG.head_dim)
    ref = apply_rope(x, cos, sin)[..., perm]

    rope = RopeSetup(galaxy_mesh, mesh_config, CFG, max_seq_len=max_seq_len, chunk_size=chunk)
    out = heads_to_torch(
        rope(to_heads(x[..., perm], galaxy_mesh, mesh_config), kv_actual=start), galaxy_mesh, mesh_config
    )
    assert_pcc(f"indexed_rope[{start}:{start + chunk}]", ref, out)
    assert (out - ref.float()).abs().max() < 0.1
