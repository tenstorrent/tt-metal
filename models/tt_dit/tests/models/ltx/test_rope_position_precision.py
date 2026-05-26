# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""RoPE must use fp32 positions before cos/sin — bf16 positions destroy high-frequency channels."""


import torch

from models.tt_dit.models.transformers.ltx.rope_ltx import LTXRopeType, precompute_freqs_cis
from models.tt_dit.utils.ltx import AudioLatentShape, audio_get_patch_grid_bounds


def test_audio_rope_bf16_positions_much_worse_than_fp32():
    audio_n_real = 126
    a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_n_real, mel_bins=16)
    a_positions = audio_get_patch_grid_bounds(a_shape).float()

    kwargs = dict(
        dim=2048,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=LTXRopeType.SPLIT,
    )
    cos_fp32, _ = precompute_freqs_cis(a_positions, **kwargs)
    cos_bf16, _ = precompute_freqs_cis(a_positions.bfloat16(), **kwargs)

    max_diff = (cos_fp32 - cos_bf16).abs().max().item()
    # bf16 fractional positions lose ~1e-3 resolution; high RoPE freqs amplify to O(1) phase error.
    assert max_diff > 0.1, f"expected large cos error from bf16 positions, got max|diff|={max_diff}"
