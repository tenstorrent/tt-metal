# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-ROPE` — `tt/rope.py` on device vs the HF `rotate_half` path, `(1,1)` mesh.

Two independent things are proved here, and the gate needs **both**:

1. **The rotation is right.** ``ttnn.experimental.rotary_embedding_llama`` with the Meta
   interleaved cos/sin and the ``[1,1,32,32]`` transformation matrix, applied to
   ``[1, n_heads, S, 128]``, equals HF's ``x * cos + rotate_half(x) * sin`` applied to the
   correspondingly-laid-out input. **PCC >= 0.999.**

   Both cos/sin conventions are derived from **one** frequency set, exactly as
   ``models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:83`` ``_build_cos_sin`` does. That
   structure is the point: a test that built the device tables from one frequency set and the
   reference tables from another could not tell a convention bug from a pass.

   The layout map between the conventions, derived rather than assumed. Meta rotates *adjacent
   pairs*; HF rotates element ``i`` against element ``i + D/2``. For pair angle ``θ_i``, Meta gives
   ``(a, b) -> (a·cos - b·sin, a·sin + b·cos)`` and HF gives exactly the same on
   ``(x_i, x_{i+D/2})``. Hence ``x_meta[2i] = x_hf[i]`` and ``x_meta[2i+1] = x_hf[i + D/2]`` — an
   interleave of the halves, and the same relation the Q/K weight ``reverse_permute``
   (``models/tt_transformers/tt/load_checkpoints.py:891``) encodes at load time for P5.5.

2. **The llama3 scaling is actually active.** A RoPE test that passes with scaling silently
   disabled is worthless (`BRINGUP_RECIPE.md:650-652`): the unscaled and scaled tables agree at
   position 0 and diverge slowly, so a plain-θ RoPE still scores a high PCC against itself. The
   scaled ``inv_freq`` must therefore differ from the unscaled one for every frequency whose
   wavelength exceeds ``original_max_position_embeddings`` (8192), with the analytic maximum
   relative deviation ``1 - 1/factor = 0.875``. P1 measured **35 / 64** slots differing at
   ``max rel dev = 0.875000`` (`06_GATES.md` G-REF); this reproduces that number through
   ``tt/rope.py``'s own delegate chain rather than through HF.

Run:
    pytest models/demos/llama32_8b_d_p/tests/unit/test_rope_vs_ref.py -x -q
"""

from __future__ import annotations

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.llama32_8b_d_p.tests.test_factory import TestFactory
from models.demos.llama32_8b_d_p.tests.unit.test_reference_model import _apply_rope_hf
from models.demos.llama32_8b_d_p.tt.rope import build_meta_cos_sin, build_prefill_rope, build_transformation_mat

PCC_THRESHOLD = 0.999

# P1's measured llama3-scaling signature (06_GATES.md, G-REF detail block).
EXPECTED_SCALED_SLOTS = 35
EXPECTED_HEAD_DIM_HALF = 64


def _meta_to_hf_layout(x: torch.Tensor) -> torch.Tensor:
    """``x_meta[..., 2i] -> x_hf[..., i]``, ``x_meta[..., 2i+1] -> x_hf[..., i + D/2]``."""
    return torch.cat([x[..., 0::2], x[..., 1::2]], dim=-1)


def _hf_to_meta_layout(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_meta_to_hf_layout` — interleave the two halves."""
    half = x.shape[-1] // 2
    return torch.stack([x[..., :half], x[..., half:]], dim=-1).flatten(-2)


def _hf_cos_sin_from_meta(cos_meta: torch.Tensor, sin_meta: torch.Tensor):
    """The HF-convention pair for the SAME frequencies: ``cat(halves)`` instead of interleave.

    ``cos_meta`` is ``[1, 1, S, D]`` holding ``[c0, c0, c1, c1, ...]``, so every even slot is one
    distinct frequency; ``cat`` of that half with itself is HF's expansion.
    """
    cos_half = cos_meta[0, 0, :, 0::2]  # [S, D/2]
    sin_half = sin_meta[0, 0, :, 0::2]
    return torch.cat([cos_half, cos_half], dim=-1), torch.cat([sin_half, sin_half], dim=-1)


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("seq_len", [128, 512, 8192], ids=["s128", "s512", "s8192"])
@torch.no_grad()
def test_rope_vs_ref(mesh_device, seq_len, reset_seeds):
    """Meta RoPE on device vs the HF `rotate_half` path on the same frequencies. PCC >= 0.999."""
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    head_dim = hf_config.head_dim
    n_heads = hf_config.num_attention_heads

    assert head_dim == 128, f"expected full-rotary head_dim 128, got {head_dim}"

    # One frequency set -> both conventions.
    cos_meta, sin_meta = build_meta_cos_sin(hf_config, seq_len)
    cos_hf, sin_hf = _hf_cos_sin_from_meta(cos_meta, sin_meta)

    # The device tables built by tt/rope.py must be that same host pair.
    rot_mats = build_prefill_rope(mesh_device, hf_config, seq_len=seq_len)
    dev_cos = ttnn.to_torch(rot_mats[0], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]
    torch.testing.assert_close(dev_cos.float(), cos_meta.to(torch.bfloat16).float(), rtol=0.0, atol=0.0)

    x_meta = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float32)

    # --- reference: HF convention, on the de-interleaved input, then re-interleaved.
    x_hf = _meta_to_hf_layout(x_meta)
    ref_hf = _apply_rope_hf(x_hf, cos_hf.unsqueeze(0), sin_hf.unsqueeze(0))
    ref = _hf_to_meta_layout(ref_hf)

    # --- device: Meta convention
    tt_x = ttnn.from_torch(
        x_meta,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    trans_mat = build_transformation_mat(mesh_device)
    assert tuple(trans_mat.shape) == (
        1,
        1,
        32,
        32,
    ), f"transformation mat is {tuple(trans_mat.shape)}, expected (1,1,32,32)"

    tt_out = ttnn.experimental.rotary_embedding_llama(tt_x, rot_mats[0], rot_mats[1], trans_mat, is_decode_mode=False)
    out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]

    assert out.shape == ref.shape == (1, n_heads, seq_len, head_dim)

    passing, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    logger.info(comp_allclose(ref, out))
    logger.info(f"[G-ROPE] seq_len={seq_len}: PCC = {pcc} (threshold {PCC_THRESHOLD})")
    assert passing, f"[G-ROPE] seq_len={seq_len} below {PCC_THRESHOLD}: {pcc}"


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_wrong_convention_fails(mesh_device, reset_seeds):
    """The negative control: feeding the device the HF-layout input scores far below the gate.

    Without this, `test_rope_vs_ref` could be passing because both sides are wrong in the same way.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    seq_len = 128

    cos_meta, sin_meta = build_meta_cos_sin(hf_config, seq_len)
    cos_hf, sin_hf = _hf_cos_sin_from_meta(cos_meta, sin_meta)
    x_meta = torch.randn(1, hf_config.num_attention_heads, seq_len, hf_config.head_dim, dtype=torch.float32)

    ref = _hf_to_meta_layout(_apply_rope_hf(_meta_to_hf_layout(x_meta), cos_hf.unsqueeze(0), sin_hf.unsqueeze(0)))

    rot_mats = build_prefill_rope(mesh_device, hf_config, seq_len=seq_len)
    # Deliberately hand the device the HF-laid-out tensor instead of the Meta-laid-out one.
    tt_x = ttnn.from_torch(
        _meta_to_hf_layout(x_meta),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = ttnn.experimental.rotary_embedding_llama(
        tt_x, rot_mats[0], rot_mats[1], build_transformation_mat(mesh_device), is_decode_mode=False
    )
    out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]

    _, pcc = comp_pcc(ref, out, 0.0)
    logger.info(
        f"[G-ROPE] negative control (HF layout fed to the Meta op): PCC = {pcc} — must be far below {PCC_THRESHOLD}"
    )
    assert float(pcc) < 0.99, f"the wrong convention scored {pcc}; the positive test proves nothing"


@torch.no_grad()
def test_llama3_scaling_is_active_in_our_path():
    """The delegate chain `tt/rope.py` uses really applies the llama3 schedule.

    Host-only. Checked on `inv_freq` — the one place the schedule is unambiguous — through
    `models/tt_transformers/tt/common.py:437` `apply_scaling`, the exact call `precompute_freqs`
    (`:504`) makes for us.
    """
    from models.demos.llama32_8b_d_p.tests.test_factory import llama_config_dims
    from models.demos.llama32_8b_d_p.tt.model_config import llama_hf_config
    from models.tt_transformers.tt.common import apply_scaling

    hf_config = llama_hf_config(llama_config_dims())

    dim = hf_config.head_dim
    factor = hf_config.rope_scaling_factor
    orig = hf_config.rope_orig_context_len

    # precompute_freqs:501, verbatim.
    unscaled = 1.0 / (hf_config.rope_theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    scaled = apply_scaling(unscaled.clone(), factor, orig, rope_type=hf_config.rope_type)

    assert unscaled.shape == scaled.shape == (dim // 2,) == (EXPECTED_HEAD_DIM_HALF,)

    rel = (scaled.double() - unscaled.double()).abs() / unscaled.double()
    n_diff = int((rel > 1e-12).sum())
    wavelen = 2 * math.pi / unscaled.double()
    low_wl = orig / hf_config.rope_low_freq_factor  # 8192
    high_wl = orig / hf_config.rope_high_freq_factor  # 2048

    logger.info(
        f"[G-ROPE] llama3 scaling: theta={hf_config.rope_theta}, factor={factor}, orig_max_pos={orig}, "
        f"low_freq_wavelen={low_wl}, high_freq_wavelen={high_wl}"
    )
    logger.info(
        f"[G-ROPE] inv_freq slots differing scaled-vs-unscaled: {n_diff}/{dim // 2}; "
        f"max rel dev = {rel.max().item():.6f} (analytic 1 - 1/factor = {1 - 1 / factor:.6f})"
    )

    assert n_diff > 0, "llama3 scaling had NO effect — it is silently disabled"
    assert (
        n_diff == EXPECTED_SCALED_SLOTS
    ), f"expected {EXPECTED_SCALED_SLOTS} differing inv_freq slots (P1 measured that); got {n_diff}"

    # Every frequency whose wavelength exceeds original_max_position_embeddings (8192) is divided by
    # exactly `factor`; everything below high_freq_wavelen is untouched. This is the positive check
    # the gate demands: "differs beyond original_max_position_embeddings".
    low_limb = wavelen > low_wl
    assert low_limb.any(), "no frequency has a wavelength beyond orig_max_pos; the test proves nothing"
    torch.testing.assert_close(scaled.double()[low_limb], unscaled.double()[low_limb] / factor, rtol=1e-12, atol=0.0)

    high_limb = wavelen < high_wl
    assert high_limb.any()
    torch.testing.assert_close(scaled.double()[high_limb], unscaled.double()[high_limb], rtol=0.0, atol=0.0)

    torch.testing.assert_close(rel.max().item(), 1.0 - 1.0 / factor, rtol=1e-9, atol=1e-12)


@torch.no_grad()
def test_scaled_cos_sin_tables_differ_from_unscaled():
    """The scaling survives into the cos/sin tables `tt/rope.py` actually emits.

    Complements the `inv_freq` check: the schedule could be right and still be dropped on the way
    to the tables (e.g. by passing `scale_factor=None` through). Compared at a position past
    `original_max_position_embeddings`, where the low-frequency limb's 8x phase difference is large.
    """
    from dataclasses import replace

    from models.demos.llama32_8b_d_p.tests.test_factory import llama_config_dims
    from models.demos.llama32_8b_d_p.tt.model_config import llama_hf_config

    hf_config = llama_hf_config(llama_config_dims())
    seq_len = 16384
    assert seq_len > hf_config.rope_orig_context_len

    scaled_cos, _ = build_meta_cos_sin(hf_config, seq_len)
    # factor 1.0 is the identity schedule: the same code path with scaling neutralised.
    unscaled_cos, _ = build_meta_cos_sin(replace(hf_config, rope_scaling_factor=1.0), seq_len)

    tail = slice(hf_config.rope_orig_context_len, seq_len)
    max_dev = (scaled_cos[:, :, tail, :] - unscaled_cos[:, :, tail, :]).abs().max().item()
    logger.info(
        f"[G-ROPE] max|cos_scaled - cos_unscaled| over positions [{hf_config.rope_orig_context_len}, {seq_len}) = {max_dev:.6f}"
    )
    assert (
        max_dev > 0.5
    ), f"the scaled and unscaled cos tables barely differ ({max_dev}); scaling is not reaching the tables"


# --------------------------------------------------------------------------------------
# build_indexed_rope — the P7/P8 chunked-prefill table. Not part of the G-ROPE PCC number,
# but smoke-tested here so the function is not shipped unexercised.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_build_indexed_rope_shapes_and_constraints(mesh_device, reset_seeds, expect_error):
    """`build_indexed_rope` builds a whole-cache table on `(1,1)` (SP=1) and enforces its two
    divisibility constraints.

    At SP=1 the block-cyclic reorder is the identity, so the table must equal the plain
    whole-cache Meta table — which is exactly the check that the reorder is not applied twice or
    with the wrong `chunk_local`.
    """
    from models.demos.llama32_8b_d_p.tests.test_factory import llama_config_dims
    from models.demos.llama32_8b_d_p.tt.model_config import llama_hf_config
    from models.demos.llama32_8b_d_p.tt.rope import build_indexed_rope

    hf_config = llama_hf_config(llama_config_dims())
    max_seq_len, chunk_size = 1024, 256

    rope_mats = build_indexed_rope(mesh_device, hf_config, max_seq_len=max_seq_len, chunk_size=chunk_size)
    assert len(rope_mats) == 2
    for t in rope_mats:
        assert tuple(t.shape) == (1, 1, max_seq_len, hf_config.head_dim), tuple(t.shape)

    expected_cos, _ = build_meta_cos_sin(hf_config, max_seq_len)
    dev_cos = ttnn.to_torch(rope_mats[0], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]
    torch.testing.assert_close(dev_cos.float(), expected_cos.to(torch.bfloat16).float(), rtol=0.0, atol=0.0)
    logger.info(f"[G-ROPE] build_indexed_rope on (1,1): cos/sin {tuple(rope_mats[0].shape)}, identity reorder OK")

    with expect_error(AssertionError, "multiple of TILE_SIZE"):
        build_indexed_rope(mesh_device, hf_config, max_seq_len=1024, chunk_size=100)
    with expect_error(AssertionError, "multiple of chunk_size"):
        build_indexed_rope(mesh_device, hf_config, max_seq_len=1000, chunk_size=256)


@torch.no_grad()
def test_block_cyclic_layout_maps_local_rows_to_global_positions():
    """Host-only: at SP=4 each chip's contiguous cos/sin shard must carry the global positions the
    KV-cache writer will put in the same local rows.

    Asserted without a 32-device mesh, because the layout is pure host arithmetic and getting it
    wrong is the failure `G-MESH-KV` would surface three phases later: chip `c`'s local row `lr`
    holds global position `(lr // chunk_local) * chunk_size + c * chunk_local + (lr % chunk_local)`
    (`models/demos/deepseek_v3_d_p/tt/mla/utils.py:88-90`).
    """
    from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
    from models.demos.llama32_8b_d_p.tests.test_factory import llama_config_dims
    from models.demos.llama32_8b_d_p.tt.model_config import llama_hf_config

    hf_config = llama_hf_config(llama_config_dims())
    sp, chunk_size, max_seq_len = TestFactory.TARGET_SP, 512, 4096
    chunk_local = chunk_size // sp

    cos, _ = build_meta_cos_sin(hf_config, max_seq_len)
    reordered = block_cyclic_reorder(cos, chunk_local, sp, seq_dim=2)
    per_chip = max_seq_len // sp
    assert reordered.shape == cos.shape

    for c in range(sp):
        shard = reordered[:, :, c * per_chip : (c + 1) * per_chip, :]
        for lr in range(0, per_chip, 37):  # stride to keep the check cheap but non-aligned
            global_pos = (lr // chunk_local) * chunk_size + c * chunk_local + (lr % chunk_local)
            torch.testing.assert_close(shard[0, 0, lr], cos[0, 0, global_pos], rtol=0.0, atol=0.0)
    logger.info(
        f"[G-ROPE] block-cyclic layout verified for sp={sp}, chunk_size={chunk_size}, max_seq_len={max_seq_len}"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_prefill_rope_start_pos_bound(mesh_device, reset_seeds, expect_error):
    """`build_prefill_rope` refuses `start_pos > seq_len` with a message naming the alternative.

    Measured, not assumed: `get_prefill_rot_mat` precomputes only `seq_len * 2` positions
    (`models/tt_transformers/tt/common.py:536`) and gathers `[start_pos, start_pos + seq_len)` from
    them (`:538`), so `start_pos = 2 * chunk` — the third chunk of a chunked prefill — raises
    `RuntimeError: index N is out of bounds` from inside `gather_cos_sin`. `DEC-029`.
    """
    from models.demos.llama32_8b_d_p.tests.test_factory import llama_config_dims
    from models.demos.llama32_8b_d_p.tt.model_config import llama_hf_config

    hf_config = llama_hf_config(llama_config_dims())

    # start_pos == seq_len is the last legal offset, and it must produce the right positions.
    rot = build_prefill_rope(mesh_device, hf_config, seq_len=128, start_pos=128)
    dev_cos = ttnn.to_torch(rot[0], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]
    expected, _ = build_meta_cos_sin(hf_config, 128, start_pos=128)
    torch.testing.assert_close(dev_cos.float(), expected.to(torch.bfloat16).float(), rtol=0.0, atol=0.0)

    with expect_error(AssertionError, "start_pos must be <= seq_len"):
        build_prefill_rope(mesh_device, hf_config, seq_len=128, start_pos=256)
    logger.info("[G-ROPE] build_prefill_rope refuses start_pos > seq_len (P7 must use build_indexed_rope)")
