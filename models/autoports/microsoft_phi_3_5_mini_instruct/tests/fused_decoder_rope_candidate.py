# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal evidence for rejecting the material dedicated-RoPE candidates."""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_rotary_embedding_hf_native_and_padded_candidates(mesh_device, expect_error):
    generator = torch.Generator().manual_seed(96048)
    value = torch.randn(1, 32, 32, 96, generator=generator, dtype=torch.bfloat16)
    cos = torch.randn(1, 1, 32, 96, generator=generator, dtype=torch.bfloat16)
    sin = torch.randn(1, 1, 32, 96, generator=generator, dtype=torch.bfloat16)
    tt_value = ttnn.from_torch(value, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_cos = ttnn.from_torch(cos, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_sin = ttnn.from_torch(sin, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    with expect_error(RuntimeError, "divisible by 64"):
        ttnn.experimental.rotary_embedding_hf(tt_value, tt_cos, tt_sin)
    print("ROPE_CANDIDATE native_hf96=REJECTED reason='padded width 96 is not divisible by 64'")

    # Adapt the width to the next legal size. This executes, but the kernel
    # rotates at padded midpoint 64, not Phi's logical midpoint 48.
    padded_value = torch.nn.functional.pad(value, (0, 32))
    padded_cos = torch.nn.functional.pad(cos, (0, 32))
    padded_sin = torch.nn.functional.pad(sin, (0, 32))
    tt_padded = ttnn.from_torch(padded_value, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_padded_cos = ttnn.from_torch(padded_cos, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_padded_sin = ttnn.from_torch(padded_sin, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    padded_output = ttnn.to_torch(ttnn.experimental.rotary_embedding_hf(tt_padded, tt_padded_cos, tt_padded_sin))[
        ..., :96
    ]

    first, second = value[..., :48], value[..., 48:]
    phi_rotated = torch.cat((-second, first), dim=-1)
    reference = value * cos + phi_rotated * sin
    pcc = torch.corrcoef(torch.stack((reference.float().flatten(), padded_output.float().flatten())))[0, 1]
    assert pcc < 0.995
    print(
        "ROPE_CANDIDATE padded_hf128=REJECTED "
        "reason='legal padding moves rotate-half midpoint from logical 48 to padded 64'"
    )


def test_llama_rotary_semantic_candidate():
    # The llama kernel's transformation matrix is one 32x32 tile repeated
    # across the width. Its public helper rotates adjacent even/odd pairs.
    # Phi/HF rotates two 48-wide halves, which crosses a tile boundary.
    value = torch.arange(96, dtype=torch.float32)
    phi_rotate_half = torch.cat((-value[48:], value[:48]))
    llama_rotate_pairs = torch.stack((-value[1::2], value[0::2]), dim=-1).flatten()
    assert not torch.equal(phi_rotate_half, llama_rotate_pairs)
    print(
        "ROPE_CANDIDATE llama_transformation_tile=REJECTED "
        "reason='32x32 repeated adjacent-pair transform cannot express a 48-wide HF half rotation'"
    )
