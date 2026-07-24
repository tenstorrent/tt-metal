# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Host-only test for Rope2DSetup.

Step 6 covers just the precompute path: our `get_freqs_cis(grid_hws)`
should match HF `Rope2DPosEmb.get_freqs_cis(grid_hws)` exactly. Both
run on CPU; the on-device application is verified in step 7 inside
the attention test.

Also exercises the auxiliary `get_cos_sin` method which builds the
real-valued cos/sin tensors that ttnn rotary_embedding_llama consumes.
A `get_cos_sin` round-trip — multiply Q by (cos, sin) using the
consecutive-pair RoPE formula — should match HF's `apply_rope` output.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_rope2d.py -v
"""
from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.rope import Rope2DSetup

# ----------------------------------------------------------------------
# (1) Complex-tensor PCC vs HF Rope2DPosEmb.get_freqs_cis.


@torch.no_grad()
@pytest.mark.parametrize(
    "grid_hws",
    [
        [[16, 16]],
        [[32, 24]],
        [[64, 64]],
        [[80, 80]],
        [[16, 16], [32, 24]],
    ],
)
def test_rope2d_freqs_cis_matches_hf(model_args, grid_hws):
    """Our get_freqs_cis output matches HF (PCC on real and imag parts)."""
    pcc_threshold = 0.9999

    ref = model_args.reference_rope_2d()
    tt = Rope2DSetup.from_torch(ref)
    assert ref.dim == tt.dim == model_args.head_dim
    assert ref.max_height == tt.max_height
    assert ref.max_width == tt.max_width

    grid_tensor = torch.tensor(grid_hws, dtype=torch.long)
    ref_complex = ref.get_freqs_cis(grid_tensor)
    tt_complex = tt.get_freqs_cis(grid_tensor)

    assert ref_complex.shape == tt_complex.shape, (ref_complex.shape, tt_complex.shape)
    assert ref_complex.dtype == torch.complex64
    assert tt_complex.dtype == torch.complex64

    # PCC on real and imaginary parts independently (comp_pcc takes floats).
    real_pass, real_msg = comp_pcc(ref_complex.real, tt_complex.real, pcc_threshold)
    imag_pass, imag_msg = comp_pcc(ref_complex.imag, tt_complex.imag, pcc_threshold)
    logger.info(
        f"[grid_hws={grid_hws}] real={real_msg} imag={imag_msg} "
        f"|{comp_allclose(ref_complex.real, tt_complex.real)}|"
    )
    assert real_pass and imag_pass, f"PCC mismatch real={real_msg} imag={imag_msg}"


# ----------------------------------------------------------------------
# (2) get_cos_sin produces tensors that, applied via the consecutive-pair
#     RoPE formula, match HF apply_rope output. This pins down both the
#     freqs scheme AND the cos/sin layout we'll hand to ttnn in step 7.


def _consecutive_pair_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply consecutive-pair RoPE: pairs (x[..., 2i], x[..., 2i+1]) get rotated.

    cos/sin have shape (..., head_dim) where each pair shares one value.
    This is the standard layout for ttnn.experimental.rotary_embedding_llama
    AFTER its internal "rotate half" handling — here we model the math
    directly on host to compare against HF apply_rope.

    Formula per pair (a, b):
        a' = a*cos - b*sin
        b' = a*sin + b*cos
    """
    # x shape: (..., num_heads, head_dim). cos/sin: (..., head_dim).
    # Broadcast cos/sin over the heads dim.
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    a = x[..., 0::2]
    b = x[..., 1::2]
    cos_p = cos[..., 0::2]  # same as cos[..., 1::2] by construction
    sin_p = sin[..., 0::2]
    a_new = a * cos_p - b * sin_p
    b_new = a * sin_p + b * cos_p
    # Re-interleave back into head_dim.
    out = torch.empty_like(x)
    out[..., 0::2] = a_new
    out[..., 1::2] = b_new
    return out


@torch.no_grad()
@pytest.mark.parametrize("grid_hws", [[[16, 16]], [[32, 24]]])
def test_rope2d_apply_matches_hf(model_args, grid_hws):
    """Applying our cos/sin via the pair-rotation formula matches HF apply_rope."""
    pcc_threshold = 0.999

    ref = model_args.reference_rope_2d()
    tt = Rope2DSetup.from_torch(ref)

    grid_tensor = torch.tensor(grid_hws, dtype=torch.long)
    L = int(grid_tensor.prod(dim=1).sum().item())
    num_heads = model_args.num_attention_heads
    head_dim = model_args.head_dim

    # Synthetic Q/K. We use fp32 for the apply step then compare against the
    # HF reference run in the same precision.
    torch.manual_seed(0)
    xq = torch.randn(L, num_heads, head_dim, dtype=torch.float32)
    xk = torch.randn(L, num_heads, head_dim, dtype=torch.float32)

    # HF apply_rope path. We need to import their function — easiest via the
    # already-loaded modeling module on the HF cache.
    import sys

    apply_rope = None
    for name, mod in sys.modules.items():
        if name.endswith("modeling_kimi_k25") and hasattr(mod, "apply_rope"):
            apply_rope = mod.apply_rope
            break
    if apply_rope is None:
        pytest.skip("apply_rope not yet registered in sys.modules; load HF model first")

    freqs_cis = ref.get_freqs_cis(grid_tensor)
    xq_ref, xk_ref = apply_rope(xq, xk, freqs_cis)

    # Our path: get_cos_sin then consecutive-pair apply in fp32.
    cos, sin = tt.get_cos_sin(grid_tensor, dtype=torch.float32)
    xq_ours = _consecutive_pair_apply(xq, cos, sin)
    xk_ours = _consecutive_pair_apply(xk, cos, sin)

    for name, ref_t, our_t in [("xq", xq_ref, xq_ours), ("xk", xk_ref, xk_ours)]:
        passing, pcc_msg = comp_pcc(ref_t, our_t, pcc_threshold)
        logger.info(f"[grid_hws={grid_hws} {name}] {comp_allclose(ref_t, our_t)} {pcc_msg}")
        assert passing, f"{name} PCC mismatch for grid_hws={grid_hws}: {pcc_msg}"
