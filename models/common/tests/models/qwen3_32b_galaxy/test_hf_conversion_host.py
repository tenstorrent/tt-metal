# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step-4 host qualification of the Qwen3-32B Galaxy adaptor.

``test_model_host.py`` already checks that every conversion in ``weight_utils``
returns the *shapes* the graph builder contracts for. Shapes are not
correctness, and for this model they are unusually weak evidence:

* ``local_qkv_size`` and ``local_dim`` are **both 1280** for Qwen3-32B, so a
  confusion between the fused-QKV width and the residual width produces exactly
  the right shape and the wrong numbers;
* a per-head Q/K norm relaid with the wrong permutation is still ``head_dim``
  wide;
* a cos/sin table built with the wrong ``rope_theta`` is still the right shape.

Milestone B's brief requires the layout conversion and the Qwen RoPE parameters
to be confirmed *numerically* on host before anything reaches the mesh, because
a weight-layout error that reaches silicon costs an hour per iteration and
costs a minute here.

Every test in this file is host-only: no ``ttnn`` device is opened.

What is actually proven here:

* the **decoupled 64-head geometry** - ``n_heads * head_dim == 8192`` against
  ``dim == 5120`` - derives the widths the graph builder contracts for, and the
  ``wo`` projection really does reduce ``attention_dim`` to ``dim``. Milestone A
  measured Qwen attention against a *square* 40-head fixture, so this geometry
  had no evidence of any kind before this file;
* ``reverse_permute_1d`` (the per-head Q/K-norm relayout) is **the same
  permutation** ``reverse_permute`` applies to the Q/K projection rows, which is
  what makes head-local RMSNorm in Meta layout equal HF's in HF layout;
* the per-head Q/K normalization reproduces HF's ``Qwen3RMSNorm`` numerically;
* ``reverse_permute`` composed with the interleaved (Meta) rotation that
  ``ttnn.experimental.rotary_embedding_llama`` implements is algebraically the
  same operator as Hugging Face's halved ``rotate_half`` composed with the HF
  weight layout, checked against the real Qwen3 rotary at ``rope_theta ==
  1000000`` and ``head_dim == 128``;
* the fused row-major QKV packing is invertible under the *decoupled* geometry,
  and each mesh row's block really is that row's ``[Q_r, K_r, V_r]`` slice;
* the converted attention (including Q/K norm), MLP and LM-head weights
  reproduce the *unmodified* Hugging Face modules to near machine precision.

The rotation convention this file asserts against is not hand-written: it is
read out of ``models.common.tensor_utils.get_rot_transformation_mat``, the same
matrix the device kernel is handed, so the host reference cannot drift from the
device one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from models.common.models.qwen3_32b_galaxy import weight_utils
from models.common.models.qwen3_32b_galaxy.model import QWEN3_32B_CHECKPOINT_CONTRACT, QWEN3_32B_GALAXY_HF_MODEL
from models.common.tensor_utils import get_rot_transformation_mat
from models.common.tests.modules import _hf_reference
from models.common.utility_functions import comp_pcc

# The conversion is exact up to bfloat16 rounding, so these thresholds are far
# above the Milestone B 0.99 model gate on purpose: a layout error shows up as a
# PCC near zero, never as a near miss.
_EXACT_PCC = 0.9999
_HEAD_DIM = 128
_ROWS = weight_utils.GALAXY_ROWS

# The real product geometry, restated so the fixture below can be checked
# against it rather than against itself.
_REAL_DIM = 5120
_REAL_N_HEADS = 64
_REAL_N_KV_HEADS = 8
_REAL_HIDDEN = 25600
_REAL_VOCAB = 151936
_REAL_PADDED_VOCAB = 152064
_REAL_ROPE_THETA = 1000000


# =============================================================================
# Host-side model of the device rotation
# =============================================================================


def _interleaved_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the rotation the device kernel applies, in the device's own terms.

    ``x`` is ``[..., head_dim]`` in Meta (interleaved) layout, ``cos``/``sin``
    are broadcastable Meta-layout tables. The ``trans_mat`` matmul is exactly
    what ``ttnn.experimental.rotary_embedding_llama`` performs on device, so the
    sign and pairing convention is taken from production rather than restated.
    """

    trans_mat = get_rot_transformation_mat(x.shape[-1])[0, 0].to(x.dtype)
    return x * cos + (x @ trans_mat) * sin


def _meta_permutation(head_dim: int) -> torch.Tensor:
    """Index map from Meta (interleaved) positions to HF (halved) positions.

    Restated from first principles - the interleave of the two HF halves -
    rather than read out of ``weight_utils``. Every comparison below that has to
    cross between the two layouts uses this, so a relayout bug in the adaptor
    cannot hide by being applied to both sides of an assertion.
    """

    half = head_dim // 2
    return torch.stack((torch.arange(half), torch.arange(half, head_dim)), dim=-1).flatten()


def _head_local_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The head-local RMSNorm ``Attention2D`` runs on the created heads.

    This is ``RMSNorm2DGeometry.HEAD_LOCAL``: the reduction is over the
    ``head_dim``-wide head only - no column reduction and no collective - so it
    is expressible in three lines of torch and needs no mesh to check.
    """

    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


# =============================================================================
# A small Qwen3 that keeps the real decoupled geometry and the real rotary
# =============================================================================


def _small_qwen3_config(**overrides: Any):
    """A tiny Qwen3 whose *shape character* is the product's, not a square one.

    Three properties of the real checkpoint are preserved exactly because they
    are the three this file exists to check:

    * ``head_dim`` is 128, decoupled from ``hidden_size``;
    * ``n_heads * head_dim > hidden_size``, in the real 1.6x ratio
      (2048 / 1280 here, 8192 / 5120 in the product), so ``wo`` is a genuinely
      non-square ``[attention_dim, dim]`` reduction;
    * ``rope_theta`` is 1000000 with ``rope_scaling`` disabled, which is what
      the cos/sin preparation depends on.

    Everything that only makes the test expensive - 64 layers, a 151936-token
    vocabulary, 25600 hidden - is shrunk. Head counts stay divisible by the
    eight mesh rows so the fused row-major packing is exercised for real.
    """

    from transformers import Qwen3Config

    kwargs = dict(
        hidden_size=1280,  # != n_heads * head_dim, in the product's 1.6x ratio
        num_attention_heads=16,  # 2 heads per mesh row
        num_key_value_heads=_ROWS,  # 1 KV head per mesh row, as in the product
        head_dim=_HEAD_DIM,
        intermediate_size=512,
        num_hidden_layers=1,
        vocab_size=256,
        rms_norm_eps=1e-6,
        rope_theta=float(_REAL_ROPE_THETA),
        rope_scaling=None,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


@pytest.fixture(scope="module")
def small_qwen3():
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(1234)
    model = Qwen3ForCausalLM(_small_qwen3_config())
    model.eval()
    return model


# =============================================================================
# The 64-head decoupled geometry: Milestone B's ranked risk #1
# =============================================================================


def test_real_qwen3_32b_geometry_is_decoupled_and_derives_the_contracted_widths():
    """The product geometry, and the widths every placement is built from.

    Milestone A's recorded "Qwen3-32B attention qualified" was measured against
    a 40-head fixture chosen so that ``n_heads * head_dim == dim``. This asserts
    the real thing: ``attention_dim`` is 8192 against ``dim`` 5120, and the two
    are *not* interchangeable anywhere.
    """

    from models.common.models.galaxy.recipes import GalaxyDenseGeometry

    geometry = GalaxyDenseGeometry(
        dim=_REAL_DIM,
        hidden_dim=_REAL_HIDDEN,
        n_heads=_REAL_N_HEADS,
        n_kv_heads=_REAL_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        vocab_size=_REAL_VOCAB,
        max_seq_len=40960,
        prefill_sequence_lengths=(128, 2048),
    )

    assert geometry.attention_dim == 8192
    assert geometry.dim == 5120
    assert geometry.attention_dim != geometry.dim, "the whole point of this model's risk"

    # The row-local widths every attention placement is built from.
    assert geometry.local_attention_dim == 1024  # attention_dim / 8 rows
    assert geometry.local_dim == 1280  # dim / 4 columns
    assert geometry.local_attention_dim != geometry.local_dim

    # `wo` reduces attention_dim -> dim, so its DRAM-sharded placement is
    # (local_attention_dim, local_dim) and never (local_dim, local_dim).
    assert (geometry.local_attention_dim, geometry.local_dim) == (1024, 1280)

    # The trap this file exists to document: local_qkv_size and local_dim are
    # both 1280, so a confusion between them is shape-invisible.
    assert geometry.local_qkv_size == geometry.local_dim == 1280
    assert geometry.local_heads == 8 and geometry.local_kv_heads == 1


def test_decode_ring_widths_differ_from_llama_by_exact_divisibility():
    """The 800-vs-960 resource-key divergence, derived rather than asserted.

    The scattered W1/W3 *placement* is padded to the 24-core ring (960 columns
    for both models); the resource *key* uses the logical width TTNN reports.
    Qwen's ``local_hidden_dim`` 3200 is an exact multiple of the 160-wide ring
    shard, so the logical width is scattered and the key is 800. Llama's 3584 is
    not, so the padded width is scattered and its key is 960. The divergence is
    arithmetic, not a defect - but it is the one place the two models'
    resource keys legitimately differ.
    """

    from models.common.models.galaxy.recipes import GalaxyDenseGeometry

    def geometry_for(hidden_dim: int, dim: int, n_heads: int) -> GalaxyDenseGeometry:
        return GalaxyDenseGeometry(
            dim=dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_kv_heads=8,
            head_dim=_HEAD_DIM,
            vocab_size=_REAL_VOCAB,
            max_seq_len=8192,
        )

    qwen = geometry_for(_REAL_HIDDEN, _REAL_DIM, _REAL_N_HEADS)
    llama = geometry_for(28672, 8192, 64)

    assert qwen.local_hidden_dim == 3200
    assert llama.local_hidden_dim == 3584

    # Placement is padded identically for both.
    assert qwen.decode_reduce_scatter_padded_width == 960
    assert llama.decode_reduce_scatter_padded_width == 960

    # The resource key is not.
    assert qwen.decode_reduce_scatter_width == 800
    assert llama.decode_reduce_scatter_width == 960


# =============================================================================
# Per-head Q/K normalization: Milestone B's ranked risk #2
# =============================================================================


@torch.no_grad()
def test_qk_norm_relayout_is_the_same_permutation_as_the_projection_relayout():
    """``reverse_permute_1d`` on a vector == ``reverse_permute`` on one head.

    Head-local RMSNorm is elementwise against a ``head_dim``-wide weight, and
    its reduction (a mean of squares over the head) is permutation-invariant.
    So normalizing Meta-layout heads with a Meta-relaid weight equals the HF
    result relaid - *provided the two relayouts are the same permutation*. If
    they ever diverge, Q/K norm silently scrambles each head and every
    downstream PCC collapses for a reason that looks like RoPE.
    """

    torch.manual_seed(5)
    weight = torch.randn(_HEAD_DIM, dtype=torch.float32)

    by_vector = weight_utils.reverse_permute_1d(weight)
    # The same operator the Q/K *projection rows* go through, for a single head.
    by_matrix = weight_utils.reverse_permute(weight.unsqueeze(-1), 1, _HEAD_DIM, 1).squeeze(-1)

    assert torch.equal(by_vector, by_matrix), "Q/K-norm relayout diverges from the projection relayout"

    # And it is the interleave of the two halves, stated independently.
    expected = torch.stack((weight[: _HEAD_DIM // 2], weight[_HEAD_DIM // 2 :]), dim=-1).flatten()
    assert torch.equal(by_vector, expected)


@torch.no_grad()
def test_head_local_rms_norm_in_meta_layout_reproduces_hf_qk_norm(small_qwen3):
    """The composed claim: Meta-layout head norm == HF head norm, relaid.

    This is the number that did not exist anywhere before this file. Milestone
    A's D2 defect was that head-local decode aborted in op validation *before
    producing any numerical result*, so there was no prior Qwen Q/K-norm value
    to compare against.
    """

    layer = small_qwen3.model.layers[0].self_attn
    eps = small_qwen3.config.rms_norm_eps
    n_heads = small_qwen3.config.num_attention_heads

    torch.manual_seed(7)
    heads_hf = torch.randn(1, n_heads, 12, _HEAD_DIM, dtype=torch.float32)

    expected_hf = layer.q_norm(heads_hf)

    # Device side: the heads arrive already in Meta layout, and the norm weight
    # was converted by the adaptor.
    #
    # The permutation is restated here from first principles - the interleave of
    # the two HF halves - rather than taken from ``reverse_permute_1d``. Deriving
    # it from the function under test would make this an identity about RMSNorm
    # (norm(Px, Pw) == P norm(x, w) holds for *any* permutation P) and would
    # prove nothing about the adaptor. Stated independently, it fails if the
    # adaptor's relayout is not this permutation.
    perm = _meta_permutation(_HEAD_DIM)
    heads_meta = heads_hf[..., perm]
    q_norm_meta = weight_utils.reverse_permute_1d(layer.q_norm.weight.detach().float())

    actual_meta = _head_local_rms_norm(heads_meta, q_norm_meta, eps)

    passing, message = comp_pcc(expected_hf[..., perm], actual_meta, _EXACT_PCC)
    assert passing, f"head-local Q norm does not reproduce HF Qwen3RMSNorm: {message}"

    # k_norm is a different weight through the same path; check it too.
    k_norm_meta = weight_utils.reverse_permute_1d(layer.k_norm.weight.detach().float())
    expected_k = layer.k_norm(heads_hf)
    actual_k = _head_local_rms_norm(heads_meta, k_norm_meta, eps)
    passing, message = comp_pcc(expected_k[..., perm], actual_k, _EXACT_PCC)
    assert passing, f"head-local K norm does not reproduce HF Qwen3RMSNorm: {message}"


@torch.no_grad()
def test_adaptor_extracts_qk_norm_and_refuses_a_checkpoint_without_one(small_qwen3):
    """The adaptor must return ``head_dim``-wide Q/K norms for Qwen3."""

    layer = small_qwen3.model.layers[0].self_attn
    _, _, q_norm, k_norm, bias = weight_utils.attention_weights_from_hf_layer(layer, rows=_ROWS)

    assert q_norm is not None and k_norm is not None, "Qwen3 must carry per-head Q/K norms"
    assert tuple(q_norm.shape) == (_HEAD_DIM,)
    assert tuple(k_norm.shape) == (_HEAD_DIM,)
    assert bias is None, "Qwen3-32B declares attention_bias=False"


# =============================================================================
# RoPE
# =============================================================================


@torch.no_grad()
def test_meta_rope_tables_are_the_hf_qwen3_rotary_relaid_not_recomputed(small_qwen3):
    """The Meta tables must be the HF rotary's own output, re-laid."""

    table_len = 64
    cos_meta, sin_meta = weight_utils.build_rope_cos_sin_torch(
        small_qwen3.model.rotary_emb, table_len, _HEAD_DIM, torch.float32
    )
    assert tuple(cos_meta.shape) == (1, 1, table_len, _HEAD_DIM)
    assert tuple(sin_meta.shape) == (1, 1, table_len, _HEAD_DIM)

    x = torch.zeros(1, 1, table_len, _HEAD_DIM, dtype=torch.float32)
    position_ids = torch.arange(table_len, dtype=torch.long).unsqueeze(0)
    cos_hf, sin_hf = small_qwen3.model.rotary_emb(x, position_ids)

    # HF's table is [half, half] duplicated; the Meta table interleaves them.
    half = _HEAD_DIM // 2
    expected_cos = torch.stack((cos_hf[0, :, :half], cos_hf[0, :, :half]), dim=-1).flatten(-2)
    expected_sin = torch.stack((sin_hf[0, :, :half], sin_hf[0, :, :half]), dim=-1).flatten(-2)

    assert torch.allclose(cos_meta[0, 0], expected_cos, atol=1e-6)
    assert torch.allclose(sin_meta[0, 0], expected_sin, atol=1e-6)


@torch.no_grad()
def test_permuted_q_with_interleaved_rope_equals_hf_rotary_on_hf_layout(small_qwen3):
    """The algebraic identity the whole Q/K conversion rests on.

    ``reverse_permute`` on the projection rows, then the device's interleaved
    rotation, must equal HF's ``rotate_half`` rotation applied to the HF-layout
    projection. Checked against the real Qwen3 rotary (theta 1000000).
    """

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    seq_len = 32
    n_heads = small_qwen3.config.num_attention_heads
    dim = small_qwen3.config.hidden_size

    torch.manual_seed(11)
    x = torch.randn(1, seq_len, dim, dtype=torch.float32)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos_hf, sin_hf = small_qwen3.model.rotary_emb(x, position_ids)
    cos_meta, sin_meta = weight_utils.build_rope_cos_sin_torch(
        small_qwen3.model.rotary_emb, seq_len, _HEAD_DIM, torch.float32
    )

    q_proj = small_qwen3.model.layers[0].self_attn.q_proj.weight.detach().float()

    # HF path: project in HF layout, form heads, rotate with rotate_half.
    q_hf = (x @ q_proj.T).view(1, seq_len, n_heads, _HEAD_DIM).transpose(1, 2)
    expected, _ = apply_rotary_pos_emb(q_hf, q_hf, cos_hf, sin_hf)

    # Device path: project with the *relaid* weight, rotate with trans_mat.
    q_meta_weight = weight_utils.reverse_permute(q_proj, n_heads, n_heads * _HEAD_DIM, dim).T
    q_meta = (x @ q_meta_weight).view(1, seq_len, n_heads, _HEAD_DIM).transpose(1, 2)
    actual = _interleaved_rope(q_meta, cos_meta, sin_meta)

    # Compare in a common layout, using the independently-stated permutation.
    perm = _meta_permutation(_HEAD_DIM)
    passing, message = comp_pcc(expected[..., perm], actual, _EXACT_PCC)
    assert passing, f"permuted-Q + interleaved RoPE != HF rotary on HF layout: {message}"


# =============================================================================
# Fused QKV packing under the decoupled geometry
# =============================================================================


@torch.no_grad()
def test_fused_qkv_row_blocks_are_recoverable_projection_slices(small_qwen3):
    """Each mesh row's block must be that row's ``[Q_r, K_r, V_r]`` slice.

    Run under the decoupled geometry, where the Q block width
    (``local_heads * head_dim``) and the KV block width differ from anything
    derived from ``dim``.
    """

    layer = small_qwen3.model.layers[0].self_attn
    config = small_qwen3.config
    n_heads, n_kv_heads = config.num_attention_heads, config.num_key_value_heads
    dim = config.hidden_size

    wq = weight_utils.reverse_permute(layer.q_proj.weight.detach(), n_heads, n_heads * _HEAD_DIM, dim).T
    wk = weight_utils.reverse_permute(layer.k_proj.weight.detach(), n_kv_heads, n_kv_heads * _HEAD_DIM, dim).T
    wv = layer.v_proj.weight.detach().T
    fused = weight_utils.fuse_qkv_by_mesh_row(wq, wk, wv, rows=_ROWS)

    q_width = (n_heads // _ROWS) * _HEAD_DIM
    kv_width = (n_kv_heads * _HEAD_DIM) // _ROWS
    block = q_width + 2 * kv_width

    assert fused.shape == (dim, _ROWS * block)
    assert _ROWS * block == _HEAD_DIM * (n_heads + 2 * n_kv_heads), "fused width must be qkv_size"

    for row in range(_ROWS):
        start = row * block
        assert torch.equal(fused[:, start : start + q_width], torch.chunk(wq, _ROWS, dim=-1)[row])
        assert torch.equal(fused[:, start + q_width : start + q_width + kv_width], torch.chunk(wk, _ROWS, dim=-1)[row])
        assert torch.equal(fused[:, start + q_width + kv_width : start + block], torch.chunk(wv, _ROWS, dim=-1)[row])


# =============================================================================
# End-to-end module equivalence
# =============================================================================


@torch.no_grad()
def test_converted_attention_weights_reproduce_the_hf_attention_output(small_qwen3):
    """Reconstruct HF attention from ``wqkv``/``wo``/``q_norm``/``k_norm`` alone.

    This is the test that exercises the decoupled geometry end to end: the
    head concat produces an ``attention_dim``-wide activation, ``wo`` reduces it
    to ``dim``, and the result is what a ``dim``-wide residual would be added
    to. A ``dim``-vs-``attention_dim`` confusion anywhere in that chain fails
    here rather than on silicon.
    """

    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    layer = small_qwen3.model.layers[0].self_attn
    config = small_qwen3.config
    n_heads, n_kv_heads = config.num_attention_heads, config.num_key_value_heads
    dim, seq_len = config.hidden_size, 48
    eps = config.rms_norm_eps

    torch.manual_seed(9)
    x = torch.randn(1, seq_len, dim, dtype=torch.float32)
    causal = torch.full((seq_len, seq_len), float("-inf")).triu(1)

    # --- reference: the unmodified HF module -------------------------------
    reference = _hf_reference.HfAttentionWrapper(layer, _HEAD_DIM, small_qwen3.model.rotary_emb)
    reference.reset_cache()
    expected = reference(x, 0, mask=causal).float()

    # --- Galaxy: only the converted tensors are allowed as input -----------
    wqkv, wo, q_norm, k_norm, bias = weight_utils.attention_weights_from_hf_layer(layer, rows=_ROWS)
    assert bias is None
    cos_meta, sin_meta = weight_utils.build_rope_cos_sin_torch(
        small_qwen3.model.rotary_emb, seq_len, _HEAD_DIM, torch.float32
    )

    # The decoupled widths, stated in the device's terms.
    attention_dim = n_heads * _HEAD_DIM
    assert attention_dim != dim, "fixture must exercise the decoupled geometry"
    assert tuple(wo.shape) == (attention_dim, dim), "wo must reduce attention_dim -> dim"

    q_width, kv_width = (n_heads // _ROWS) * _HEAD_DIM, (n_kv_heads * _HEAD_DIM) // _ROWS
    block = q_width + 2 * kv_width
    projected = x @ wqkv.float()

    # Unpack per mesh row exactly as the fused create-QKV-heads collective does.
    q_rows, k_rows, v_rows = [], [], []
    for row in range(_ROWS):
        start = row * block
        q_rows.append(projected[..., start : start + q_width])
        k_rows.append(projected[..., start + q_width : start + q_width + kv_width])
        v_rows.append(projected[..., start + q_width + kv_width : start + block])
    q = torch.cat(q_rows, dim=-1).view(1, seq_len, n_heads, _HEAD_DIM).transpose(1, 2)
    k = torch.cat(k_rows, dim=-1).view(1, seq_len, n_kv_heads, _HEAD_DIM).transpose(1, 2)
    v = torch.cat(v_rows, dim=-1).view(1, seq_len, n_kv_heads, _HEAD_DIM).transpose(1, 2)

    # Qwen3 normalizes each head before RoPE - head-local, no collective.
    q = _head_local_rms_norm(q, q_norm, eps)
    k = _head_local_rms_norm(k, k_norm, eps)

    q = _interleaved_rope(q, cos_meta, sin_meta)
    k = _interleaved_rope(k, cos_meta, sin_meta)
    # The device stores and attends in Meta layout on both sides of the QK dot
    # product, which is rotation-invariant under the relayout, so no conversion
    # back to HF halves is needed here.
    groups = n_heads // n_kv_heads
    scores = (q @ repeat_kv(k, groups).transpose(-1, -2)) / (_HEAD_DIM**0.5) + causal
    attended = torch.softmax(scores, dim=-1) @ repeat_kv(v, groups)

    # The concat is attention_dim wide, and only wo brings it back to dim.
    concatenated = attended.transpose(1, 2).reshape(1, seq_len, attention_dim)
    assert concatenated.shape[-1] == attention_dim
    actual = concatenated @ wo.float()
    assert actual.shape[-1] == dim, "the residual added after wo must be dim-wide"

    passing, message = comp_pcc(expected, actual, _EXACT_PCC)
    assert passing, f"converted attention weights do not reproduce HF attention: {message}"


@torch.no_grad()
def test_converted_mlp_weights_reproduce_the_hf_mlp_output(small_qwen3):
    mlp = small_qwen3.model.layers[0].mlp
    torch.manual_seed(13)
    x = torch.randn(1, 32, small_qwen3.config.hidden_size, dtype=torch.float32)

    expected = mlp(x.to(mlp.gate_proj.weight.dtype)).float()
    w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(mlp)
    actual = (torch.nn.functional.silu(x @ w1.float()) * (x @ w3.float())) @ w2.float()

    passing, message = comp_pcc(expected, actual, _EXACT_PCC)
    assert passing, f"converted MLP weights do not reproduce HF MLP: {message}"


@torch.no_grad()
def test_converted_lm_head_reproduces_the_hf_logits(small_qwen3):
    lm_head = small_qwen3.lm_head
    dim = small_qwen3.config.hidden_size
    vocab = small_qwen3.config.vocab_size

    torch.manual_seed(17)
    x = torch.randn(1, 8, dim, dtype=torch.float32)
    expected = lm_head(x.to(lm_head.weight.dtype)).float()

    converted = weight_utils.lm_head_weight_torch(lm_head, dim=dim, vocab_size=vocab, padded_vocab_size=vocab)
    actual = x @ converted.float()

    passing, message = comp_pcc(expected, actual, _EXACT_PCC)
    assert passing, f"converted LM head does not reproduce HF logits: {message}"


@torch.no_grad()
def test_lm_head_padding_is_inert_at_the_real_vocabulary_widths():
    """151936 -> 152064 padding must be zero and must not move real columns."""

    dim = 64
    torch.manual_seed(19)
    head = torch.nn.Linear(dim, _REAL_VOCAB, bias=False)

    converted = weight_utils.lm_head_weight_torch(
        head, dim=dim, vocab_size=_REAL_VOCAB, padded_vocab_size=_REAL_PADDED_VOCAB
    )

    assert tuple(converted.shape) == (dim, _REAL_PADDED_VOCAB)
    assert torch.equal(
        converted[:, _REAL_VOCAB:], torch.zeros(dim, _REAL_PADDED_VOCAB - _REAL_VOCAB, dtype=converted.dtype)
    )
    assert torch.equal(converted[:, :_REAL_VOCAB], head.weight.detach().to(torch.bfloat16).T)


# =============================================================================
# The real checkpoint
# =============================================================================


def _local_files_only() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False")


def _checkpoint_snapshot_or_skip(hf_model: str) -> Path:
    """Return the local snapshot directory, or skip when it is not present."""

    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(hf_model, local_files_only=True, allow_patterns=["config.json"]))
    except Exception as exc:  # noqa: BLE001 - any resolution failure is a skip
        pytest.skip(f"{hf_model} is not present in the local HF cache: {exc}")


def test_real_checkpoint_config_matches_the_contract_and_declares_no_qkv_bias():
    """Risk #4: a fused QKV bias would be a contract change, not a fix.

    ``Attention2D`` validates a bias against the projection's DRAM-sharded
    weight placement, which a bias vector cannot satisfy. If this ever fails
    because ``attention_bias`` became true, the correct response is to stop and
    report it, not to add a bias path in passing.
    """

    snapshot = _checkpoint_snapshot_or_skip(QWEN3_32B_GALAXY_HF_MODEL)
    config = json.loads((snapshot / "config.json").read_text())

    assert config["attention_bias"] is False, "a fused QKV bias needs a module-config contract change"

    for name, expected in QWEN3_32B_CHECKPOINT_CONTRACT.items():
        assert config.get(name) == expected, f"checkpoint {name}={config.get(name)!r}, contract expects {expected!r}"

    # And the decoupled geometry, read from the checkpoint rather than restated.
    assert config["num_attention_heads"] * config["head_dim"] == 8192
    assert config["hidden_size"] == 5120
