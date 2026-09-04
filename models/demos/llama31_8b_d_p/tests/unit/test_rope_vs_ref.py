# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/rope.py` + `ttnn.experimental.rotary_embedding_llama` vs the HF `rotate_half` path. Gate: `G-ROPE`.

The block: llama3-scaled RoPE applied to a `[1, n_heads, S, 128]` tensor. Full rotary
(`rotary_dim == head_dim`), theta 500000.0, llama3 scaling factor 8.0 with
`original_max_position_embeddings` 8192 (`bringup_log/00_MODEL_CARD.md` §2).

**The two conventions, and why the test is built the way it is.** `tt/rope.py` builds Meta /
interleaved cos/sin; HF's reference rotates halves. Both tables are derived from **one** set of
frequencies, exactly as `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py::_build_cos_sin`
does, so the test cannot silently compare two different RoPEs and call it a pass. The input is
generated in HF layout, the reference rotates it there, and both the input and the reference output
are mapped into Meta layout by `_hf_to_meta` — the activation-space equivalent of the weight
`reverse_permute` at `models/tt_transformers/tt/load_checkpoints.py:891`.

* **Input distribution:** standard normal, per head, over `[1, n_heads, S, 128]`.
* **Reference dtype policy:** fp32 input, fp32 cos/sin, fp32 arithmetic. Only what the device
  *stores* — the bf16 input and the bf16 cos/sin — is quantised, and only for the floor.
* **Threshold:** PCC >= 0.999 (`BRINGUP_RECIPE.md:1752`), expect ~0.99999. Ratio to the floor is
  recorded.
* **Negative controls, two:** an HF-layout tensor fed straight into the Meta op must collapse
  (without it, 0.99999 could mean "both sides are wrong the same way"); and the llama3 scaling
  must be provably active, because a test that passes with scaling silently disabled is worthless
  — asserted on the piecewise **band structure** of the frequencies, not on
  `original_max_position_embeddings` divergence, for the measured reason in
  `test_llama3_scaling_is_active`'s docstring.

**What this does NOT prove.** That the Q/K projection weights are `reverse_permute`d on the real
load path — the permutation is applied here by the test, not by `tt/attention/weights.py`, which
does not exist until P5.5. `G-ATTN`'s "loaded without the Meta permute" control is what closes
that.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_rope_vs_ref.py -x -q
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import err_ratio, llama_config_dims, quantize_like_device
from models.demos.llama31_8b_d_p.tt.config import default_compute_kernel_config, derive_head_dim
from models.demos.llama31_8b_d_p.tt.rope import (
    assert_llama3_factors,
    build_indexed_rope,
    build_prefill_rope,
    build_transformation_mat,
    llama3_freqs,
    rope_params,
)

PCC_THRESHOLD = 0.999
SEQ_LENS = [32, 512, 4096]
N_HEADS = 4  # local Q heads per chip at the deployment TP=8 (32/8)
ACTIVATION_DTYPE = ttnn.bfloat16


def _hf_to_meta(x):
    """HF head layout `[r_0..r_63, i_0..i_63]` -> Meta interleaved `[r_0, i_0, r_1, i_1, ...]`.

    The activation-space equivalent of `reverse_permute`
    (`models/tt_transformers/tt/load_checkpoints.py:891`), which does the same reordering to the
    rows of a Q/K projection weight.
    """
    half = x.shape[-1] // 2
    return torch.stack([x[..., :half], x[..., half:]], dim=-1).flatten(-2)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _rope_hf(x, cos_hf, sin_hf):
    """HF convention: `x * cos + rotate_half(x) * sin`, all fp32."""
    return x.float() * cos_hf.float() + _rotate_half(x.float()) * sin_hf.float()


def _cos_sin_both_conventions(hf, seq_len):
    """One frequency set -> `(cos_hf, sin_hf)` for the reference and Meta tables for the device."""
    cos_half, sin_half = llama3_freqs(hf, seq_len)
    cos_hf = torch.cat([cos_half, cos_half], dim=-1)[None, None]  # [1,1,S,head_dim]
    sin_hf = torch.cat([sin_half, sin_half], dim=-1)[None, None]
    cos_meta = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)[None, None]
    sin_meta = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)[None, None]
    return (cos_hf, sin_hf), (cos_meta, sin_meta)


def _apply_on_device(mesh_device, hf, x_meta, seq_len):
    """Run `rotary_embedding_llama` with the module's own tables and transformation matrix."""
    rope_mats = build_prefill_rope(mesh_device, hf, seq_len)
    trans_mat = build_transformation_mat(mesh_device)
    tt_x = ttnn.from_torch(
        x_meta,
        device=mesh_device,
        dtype=ACTIVATION_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = ttnn.experimental.rotary_embedding_llama(
        tt_x,
        rope_mats[0],
        rope_mats[1],
        trans_mat,
        is_decode_mode=False,
        compute_kernel_config=default_compute_kernel_config(mesh_device),
    )
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    for t in (tt_x, tt_out, trans_mat, *rope_mats):
        t.deallocate(True)
    return out


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"s{s}")
def test_rope_vs_ref(mesh_device, seq_len, reset_seeds):
    """Meta-convention device RoPE == HF-convention torch RoPE on the correspondingly permuted input."""
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)

    x_hf = torch.randn(1, N_HEADS, seq_len, head_dim)
    (cos_hf, sin_hf), _ = _cos_sin_both_conventions(hf, seq_len)

    ref_meta = _hf_to_meta(_rope_hf(x_hf, cos_hf, sin_hf))

    # Noise floor: quantise the three tensors the device stores (input, cos, sin), fp32 arithmetic.
    x_q = quantize_like_device(x_hf, ACTIVATION_DTYPE)
    cos_q = quantize_like_device(cos_hf, ACTIVATION_DTYPE)
    sin_q = quantize_like_device(sin_hf, ACTIVATION_DTYPE)
    _, floor = comp_pcc(ref_meta, _hf_to_meta(_rope_hf(x_q, cos_q, sin_q)), 0.0)

    out = _apply_on_device(mesh_device, hf, _hf_to_meta(x_hf), seq_len)

    passing, pcc = comp_pcc(ref_meta, out, PCC_THRESHOLD)
    ratio = err_ratio(float(pcc), float(floor))
    logger.info(f"[G-ROPE] seq={seq_len}: PCC={float(pcc):.7f} floor={float(floor):.7f} ratio={ratio:.2f}x")
    assert passing, f"below threshold {PCC_THRESHOLD}: {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"s{s}")
def test_prefill_tables_are_meta_convention(mesh_device, seq_len, reset_seeds):
    """`build_prefill_rope`'s device tables are **bit-identical** to the test's own Meta tables.

    This is what makes the "one frequency set, two conventions" claim in the module docstring
    checkable: without it, the positive test compares the module's tables against a reference built
    from the test's tables and nothing proves the two agree. Gated on `torch.equal`, not PCC —
    a table layout is a mapping claim (recipe §2.5).
    """
    hf = llama_config_dims()
    _, (cos_meta, sin_meta) = _cos_sin_both_conventions(hf, seq_len)

    rope_mats = build_prefill_rope(mesh_device, hf, seq_len)
    dev_cos = ttnn.to_torch(ttnn.get_device_tensors(rope_mats[0])[0]).float()
    dev_sin = ttnn.to_torch(ttnn.get_device_tensors(rope_mats[1])[0]).float()
    for t in rope_mats:
        t.deallocate(True)

    exp_cos = quantize_like_device(cos_meta, ACTIVATION_DTYPE)
    exp_sin = quantize_like_device(sin_meta, ACTIVATION_DTYPE)
    logger.info(
        f"[G-ROPE] tables seq={seq_len}: shape {tuple(dev_cos.shape)}; "
        f"max|cos delta|={float((dev_cos - exp_cos).abs().max())} "
        f"max|sin delta|={float((dev_sin - exp_sin).abs().max())}"
    )
    assert torch.equal(dev_cos, exp_cos), "build_prefill_rope's cos table is not the Meta interleave"
    assert torch.equal(dev_sin, exp_sin), "build_prefill_rope's sin table is not the Meta interleave"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_rope_hf_layout_into_meta_op_collapses(mesh_device, reset_seeds):
    """**Negative control 1.** An HF-layout tensor into the Meta op must collapse.

    This is the classic RoPE bug — halves-concatenated activations rotated as interleaved pairs.
    The recipe measured 0.01296 for it. Without this control the positive test's 0.99999 could
    equally mean both sides are wrong the same way.
    """
    hf = llama_config_dims()
    seq_len, head_dim = 512, derive_head_dim(hf)

    x_hf = torch.randn(1, N_HEADS, seq_len, head_dim)
    (cos_hf, sin_hf), _ = _cos_sin_both_conventions(hf, seq_len)
    ref_meta = _hf_to_meta(_rope_hf(x_hf, cos_hf, sin_hf))

    # The mistake: hand the op the HF-layout tensor, skipping `_hf_to_meta`.
    out = _apply_on_device(mesh_device, hf, x_hf, seq_len)

    _, pcc = comp_pcc(ref_meta, out, 0.0)
    logger.info(f"[G-ROPE] control 1: HF-layout tensor into the Meta op -> PCC={float(pcc):.5f}")
    assert float(pcc) < 0.99, f"the control did not collapse ({pcc}) — the gate is not sensitive to layout"


@torch.no_grad()
def test_llama3_scaling_is_active():
    """**Negative control 2.** The llama3 piecewise scaling must measurably change the tables.

    Three claims, all asserted rather than assumed:

    1. the piecewise band structure of `compute_llama3_parameters` is real — long-wavelength
       components (`wavelen > orig/low_freq_factor`) divided by exactly `factor`,
       short-wavelength ones (`wavelen < orig/high_freq_factor`) untouched, and a non-empty middle
       band strictly between the two;
    2. the scaled and unscaled cos tables actually differ, recorded **both** inside and beyond
       `original_max_position_embeddings`. Measured, the inside-window delta saturates at ~2.0 as
       well, because llama3 scaling divides the long-wavelength frequencies at *every* position and
       `cos` oscillates — so `BRINGUP_RECIPE.md:1093-1095`'s framing ("the scaled `inv_freq` must
       differ from the unscaled one for positions **beyond**
       `original_max_position_embeddings`") is not a discriminator on its own: an implementation
       that scaled everything, or nothing beyond the window, would pass it. Claim 1 is the sharp
       test; this one is kept as the recorded number the recipe asks for.

    Claim 1 is checked on the frequencies themselves rather than recovered from a cos table by
    `arccos`: for the lowest frequency `cos(1 * f)` rounds to 1.0 in fp32, so the inversion is
    `0/0 = nan`. Measuring the input to the table instead of inverting its output is both exact and
    a direct test of the imported math.
    """
    from models.tt_transformers.tt.common import apply_scaling

    hf = llama_config_dims()
    theta, factor, orig_context_len = rope_params(hf)
    assert (theta, factor, orig_context_len) == (500000.0, 8.0, 8192), (theta, factor, orig_context_len)
    assert_llama3_factors(hf)

    head_dim = derive_head_dim(hf)
    beyond = 2 * orig_context_len  # 16384 positions: half inside the original window, half past it

    cos_scaled, _ = llama3_freqs(hf, beyond, scaled=True)
    cos_unscaled, _ = llama3_freqs(hf, beyond, scaled=False)

    # (1) the bands, on the frequencies `precompute_freqs` feeds to `apply_scaling`.
    base = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    scaled = apply_scaling(base.clone(), factor, orig_context_len, rope_type="llama3")
    wavelen = 2 * torch.pi / base
    low_band = wavelen > orig_context_len / 1.0  # low_freq_wavelen  = orig / low_freq_factor
    high_band = wavelen < orig_context_len / 4.0  # high_freq_wavelen = orig / high_freq_factor
    mid_band = ~(low_band | high_band)
    logger.info(
        f"[G-ROPE] control 2: bands over {head_dim // 2} freqs — "
        f"low(scaled by 1/{factor})={int(low_band.sum())} mid(interpolated)={int(mid_band.sum())} "
        f"high(untouched)={int(high_band.sum())}"
    )
    assert bool(low_band.any()) and bool(mid_band.any()) and bool(high_band.any()), "a scaling band is empty"
    assert torch.allclose(scaled[low_band], base[low_band] / factor), "long-wavelength freqs are not divided"
    assert torch.equal(scaled[high_band], base[high_band]), "short-wavelength freqs were altered"
    assert (
        (scaled[mid_band] > base[mid_band] / factor) & (scaled[mid_band] < base[mid_band])
    ).all(), "the middle band is not strictly interpolated between the two"

    # (2) divergence past the original window.
    delta_inside = (cos_scaled[:orig_context_len] - cos_unscaled[:orig_context_len]).abs().max()
    delta_beyond = (cos_scaled[orig_context_len:] - cos_unscaled[orig_context_len:]).abs().max()
    logger.info(
        f"[G-ROPE] control 2: max|cos_scaled - cos_unscaled| inside={float(delta_inside):.5f} "
        f"beyond={float(delta_beyond):.5f} (both saturate — see this test's docstring)"
    )
    assert float(delta_beyond) > 0.5, "scaled and unscaled tables barely differ past the original window"


@torch.no_grad()
def test_contiguous_builder_refuses_a_chunked_start_pos(expect_error):
    """`build_prefill_rope` must refuse `start_pos > seq_len` rather than read out of bounds.

    `models/tt_transformers/tt/common.py:525` `gather_cos_sin` indexes a table of `seq_len * 2`
    rows, so a chunked call raises `RuntimeError: index N is out of bounds` from inside a helper
    whose message names neither RoPE nor chunking. Failing at the boundary instead is the point.
    """
    with expect_error(AssertionError, "build_indexed_rope"):
        build_prefill_rope(None, llama_config_dims(), 128, start_pos=256)


@torch.no_grad()
def test_transformation_mat_is_dhead_independent():
    """`get_rot_transformation_mat` ignores its argument — call it with none (recipe P1 trap 4).

    `models/tt_transformers/tt/common.py:564` reassigns `dhead = 32` regardless, so a `dhead=128`
    call is silently a 32x32 matrix. Asserted so the module's "call it with no args" comment is a
    measured fact rather than a claim.
    """
    from models.tt_transformers.tt.common import get_rot_transformation_mat

    default = get_rot_transformation_mat()
    with_head_dim = get_rot_transformation_mat(dhead=128)
    assert tuple(default.shape)[-2:] == (32, 32), tuple(default.shape)
    assert torch.equal(default, with_head_dim), "get_rot_transformation_mat now honours dhead — recheck P1 trap 4"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_indexed_builder_structure_and_refusals(mesh_device, expect_error, reset_seeds):
    """`build_indexed_rope` — the structural half only. `G-CHUNK` (P7) is what proves it numerically.

    At SP=1 the block-cyclic reorder is the identity, so the whole-cache table must equal the plain
    Meta table for `max_seq_len` positions; that pins the convention and the position ordering
    without claiming anything about the SP>1 layout, which no `(1,1)` gate can see. The two shape
    constraints are asserted as refusals because violating them is the "chunk write asserts" row of
    Appendix B, and failing at table-build time is cheaper than failing inside the op.

    Written in P5.3 rather than P7 because `BRINGUP_RECIPE.md:1091-1093` puts the builder here; the
    numbers belong to the phase that uses it.
    """
    hf = llama_config_dims()
    max_seq_len, chunk_size = 1024, 256

    _, (cos_meta, sin_meta) = _cos_sin_both_conventions(hf, max_seq_len)
    mats = build_indexed_rope(mesh_device, hf, max_seq_len=max_seq_len, chunk_size=chunk_size)
    dev_cos = ttnn.to_torch(ttnn.get_device_tensors(mats[0])[0]).float()
    dev_sin = ttnn.to_torch(ttnn.get_device_tensors(mats[1])[0]).float()
    for t in mats:
        t.deallocate(True)

    logger.info(f"[G-ROPE] indexed table at SP=1: shape {tuple(dev_cos.shape)} (max_seq_len={max_seq_len})")
    assert tuple(dev_cos.shape) == (1, 1, max_seq_len, derive_head_dim(hf))
    assert torch.equal(dev_cos, quantize_like_device(cos_meta, ACTIVATION_DTYPE))
    assert torch.equal(dev_sin, quantize_like_device(sin_meta, ACTIVATION_DTYPE))

    with expect_error(AssertionError, "must be a multiple of TILE_SIZE"):
        build_indexed_rope(mesh_device, hf, max_seq_len=1024, chunk_size=100)
    with expect_error(AssertionError, "must be a multiple of chunk_size"):
        build_indexed_rope(mesh_device, hf, max_seq_len=1000, chunk_size=256)
