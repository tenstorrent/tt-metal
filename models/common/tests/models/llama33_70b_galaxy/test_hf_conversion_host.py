# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step-1 host qualification of the Llama-3.3-70B Galaxy adaptor.

``test_model_host.py`` already checks that every conversion in
``weight_utils`` returns the *shapes* the graph builder contracts for. Shapes
are not correctness: a Q/K permutation that transposes the wrong pair of axes,
or a cos/sin table built with the unscaled rotary, produces exactly the right
shape and the wrong numbers. Milestone B's brief requires the layout conversion
and the Llama 3 scaled-RoPE preparation to be confirmed *numerically* on host
before anything reaches the mesh, because a weight-layout error that reaches
silicon costs an hour per iteration and costs a minute here.

Every test in this file is host-only: no ``ttnn`` device is opened.

What is actually proven here:

* ``reverse_permute`` composed with the interleaved (Meta) rotation that
  ``ttnn.experimental.rotary_embedding_llama`` implements is *algebraically the
  same operator* as Hugging Face's halved ``rotate_half`` rotation composed with
  the HF weight layout. This is checked against the real Llama-3.3-70B rotary
  module, with its ``llama3`` scaling, at ``head_dim == 128``.
* the fused row-major QKV packing is invertible, and each mesh row's block
  really is that row's ``[Q_r, K_r, V_r]`` slice;
* the converted attention and MLP weights reproduce the output of the
  *unmodified* Hugging Face modules to near machine precision;
* the real checkpoint's layer 0 converts to the contracted shapes, read
  directly from the safetensors shards rather than by materializing all 141 GB.

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

from models.common.models.llama33_70b_galaxy import weight_utils
from models.common.models.llama33_70b_galaxy.model import LLAMA33_70B_GALAXY_HF_MODEL, parameters_from_hf_config
from models.common.tensor_utils import get_rot_transformation_mat
from models.common.tests.modules import _hf_reference
from models.common.utility_functions import comp_pcc

# The conversion is exact up to bfloat16 rounding, so these thresholds are far
# above the Milestone B 0.99 model gate on purpose: a layout error shows up as a
# PCC near zero, never as a near miss.
_EXACT_PCC = 0.9999
_HEAD_DIM = 128
_ROWS = weight_utils.GALAXY_ROWS


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


def _meta_to_hf_halves(x: torch.Tensor) -> torch.Tensor:
    """Re-lay a Meta-interleaved ``[..., head_dim]`` vector into HF halves."""

    return torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)


# =============================================================================
# A small Llama that keeps the real Llama-3.3-70B rotary configuration
# =============================================================================


def _small_llama_config(**overrides: Any):
    """A tiny Llama whose *rotary* configuration is the product's, verbatim.

    ``head_dim`` and ``rope_scaling`` are the values from the real
    ``meta-llama/Llama-3.3-70B-Instruct`` config, because those are the two
    things the RoPE preparation depends on. Everything that only makes the test
    expensive - 80 layers, 128k vocabulary, 28672 hidden - is shrunk. Head
    counts stay divisible by the eight mesh rows so the fused packing is
    exercised for real.
    """

    from transformers import LlamaConfig

    kwargs = dict(
        hidden_size=_ROWS * _HEAD_DIM,  # 1024
        num_attention_heads=_ROWS,  # 8 heads, one per mesh row
        num_key_value_heads=2,
        head_dim=_HEAD_DIM,
        intermediate_size=512,
        num_hidden_layers=1,
        vocab_size=256,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    kwargs.update(overrides)
    return LlamaConfig(**kwargs)


@pytest.fixture(scope="module")
def small_llama():
    from transformers import LlamaForCausalLM

    torch.manual_seed(1234)
    model = LlamaForCausalLM(_small_llama_config())
    model.eval()
    return model


# =============================================================================
# RoPE: the Milestone B author's ranked risk #1, checked on host
# =============================================================================


@torch.no_grad()
def test_meta_rope_tables_are_the_hf_rotary_relaid_not_recomputed(small_llama):
    """The Meta tables must carry the *scaled* HF frequencies, pair-duplicated."""

    rotary = small_llama.model.rotary_emb
    table_len = 256
    cos, sin = weight_utils.build_rope_cos_sin_torch(rotary, table_len, _HEAD_DIM, torch.float32)
    assert cos.shape == (1, 1, table_len, _HEAD_DIM)
    assert sin.shape == (1, 1, table_len, _HEAD_DIM)

    position_ids = torch.arange(table_len).unsqueeze(0)
    cos_hf, sin_hf = rotary(torch.zeros(1, 1, table_len, _HEAD_DIM), position_ids)
    cos_hf, sin_hf = cos_hf[0].float(), sin_hf[0].float()

    # Meta pair k must carry HF frequency k, in both slots of the pair.
    for table, reference, name in ((cos[0, 0], cos_hf, "cos"), (sin[0, 0], sin_hf, "sin")):
        assert torch.equal(table[:, 0::2], reference[:, : _HEAD_DIM // 2]), f"{name} even slots"
        assert torch.equal(table[:, 1::2], reference[:, : _HEAD_DIM // 2]), f"{name} odd slots"

    # And the scaling really is applied: llama3 scaling must move the table away
    # from the unscaled theta table, or the whole check above is vacuous.
    unscaled_config = _small_llama_config(rope_scaling=None)
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    unscaled = LlamaRotaryEmbedding(config=unscaled_config)
    cos_unscaled, _ = weight_utils.build_rope_cos_sin_torch(unscaled, table_len, _HEAD_DIM, torch.float32)
    assert not torch.allclose(cos, cos_unscaled, atol=1e-3), "llama3 RoPE scaling was not applied"


@torch.no_grad()
def test_permuted_q_with_interleaved_rope_equals_hf_rotary_on_hf_layout(small_llama):
    """``reverse_permute`` + interleaved rotation == HF layout + ``rotate_half``.

    This is the pairing Milestone A never qualified: attention was qualified
    with an *identity* rotary, and ``RotarySetup2D`` was qualified standalone.
    Here the two are composed on host, against the real scaled rotary, so a
    convention mismatch is a host failure rather than a silicon PCC mystery.
    """

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    layer = small_llama.model.layers[0].self_attn
    config = small_llama.config
    n_heads, n_kv_heads = config.num_attention_heads, config.num_key_value_heads
    dim, seq_len = config.hidden_size, 64

    torch.manual_seed(7)
    x = torch.randn(1, seq_len, dim, dtype=torch.float32)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos_hf, sin_hf = small_llama.model.rotary_emb(x, position_ids)

    # --- Hugging Face side, untouched --------------------------------------
    q_hf = (x @ layer.q_proj.weight.float().T).view(1, seq_len, n_heads, _HEAD_DIM).transpose(1, 2)
    k_hf = (x @ layer.k_proj.weight.float().T).view(1, seq_len, n_kv_heads, _HEAD_DIM).transpose(1, 2)
    q_hf_rot, k_hf_rot = apply_rotary_pos_emb(q_hf, k_hf, cos_hf.float(), sin_hf.float())

    # --- Galaxy side: production permutation, production rotation ----------
    cos_meta, sin_meta = weight_utils.build_rope_cos_sin_torch(
        small_llama.model.rotary_emb, seq_len, _HEAD_DIM, torch.float32
    )
    wq_meta = weight_utils.reverse_permute(layer.q_proj.weight.float(), n_heads, n_heads * _HEAD_DIM, dim).T
    wk_meta = weight_utils.reverse_permute(layer.k_proj.weight.float(), n_kv_heads, n_kv_heads * _HEAD_DIM, dim).T
    q_meta = (x @ wq_meta).view(1, seq_len, n_heads, _HEAD_DIM).transpose(1, 2)
    k_meta = (x @ wk_meta).view(1, seq_len, n_kv_heads, _HEAD_DIM).transpose(1, 2)
    q_meta_rot = _interleaved_rope(q_meta, cos_meta, sin_meta)
    k_meta_rot = _interleaved_rope(k_meta, cos_meta, sin_meta)

    # The two differ only by the within-head layout, so relaying one must
    # reproduce the other exactly.
    torch.testing.assert_close(_meta_to_hf_halves(q_meta_rot), q_hf_rot, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(_meta_to_hf_halves(k_meta_rot), k_hf_rot, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_production_reverse_permute_matches_the_shared_1d_reference():
    """The 2D adaptor and the qualified 1D suites must permute identically."""

    torch.manual_seed(3)
    weight = torch.randn(_ROWS * _HEAD_DIM, 64)
    assert torch.equal(
        weight_utils.reverse_permute(weight, _ROWS, _ROWS * _HEAD_DIM, 64),
        _hf_reference.reverse_permute(weight, _ROWS, _ROWS * _HEAD_DIM, 64),
    )


# =============================================================================
# Fused QKV packing
# =============================================================================


@torch.no_grad()
def test_fused_qkv_row_blocks_are_recoverable_projection_slices():
    """Row ``r`` of the fused weight must be exactly ``[Q_r, K_r, V_r]``."""

    torch.manual_seed(5)
    dim = 256
    wq = torch.randn(dim, _ROWS * _HEAD_DIM)
    wk = torch.randn(dim, 2 * _HEAD_DIM)
    wv = torch.randn(dim, 2 * _HEAD_DIM)
    fused = weight_utils.fuse_qkv_by_mesh_row(wq, wk, wv, rows=_ROWS)

    q_width, k_width, v_width = (tensor.shape[-1] // _ROWS for tensor in (wq, wk, wv))
    block = q_width + k_width + v_width
    assert fused.shape == (dim, block * _ROWS)
    for row in range(_ROWS):
        start = row * block
        torch.testing.assert_close(fused[:, start : start + q_width], wq[:, row * q_width : (row + 1) * q_width])
        start += q_width
        torch.testing.assert_close(fused[:, start : start + k_width], wk[:, row * k_width : (row + 1) * k_width])
        start += k_width
        torch.testing.assert_close(fused[:, start : start + v_width], wv[:, row * v_width : (row + 1) * v_width])


# =============================================================================
# Whole-module numerical equivalence
# =============================================================================


@torch.no_grad()
def test_converted_attention_weights_reproduce_the_hf_attention_output(small_llama):
    """Reconstruct HF attention from ``wqkv``/``wo`` alone and compare."""

    from transformers.models.llama.modeling_llama import repeat_kv

    layer = small_llama.model.layers[0].self_attn
    config = small_llama.config
    n_heads, n_kv_heads = config.num_attention_heads, config.num_key_value_heads
    dim, seq_len = config.hidden_size, 48

    torch.manual_seed(9)
    x = torch.randn(1, seq_len, dim, dtype=torch.float32)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos_hf, sin_hf = small_llama.model.rotary_emb(x, position_ids)
    causal = torch.full((seq_len, seq_len), float("-inf")).triu(1)

    # --- reference: the unmodified HF module -------------------------------
    reference = _hf_reference.HfAttentionWrapper(layer, _HEAD_DIM, small_llama.model.rotary_emb)
    reference.reset_cache()
    expected = reference(x, 0, mask=causal).float()

    # --- Galaxy: only the converted tensors are allowed as input -----------
    wqkv, wo = weight_utils.attention_weights_from_hf_layer(layer, rows=_ROWS)
    cos_meta, sin_meta = weight_utils.build_rope_cos_sin_torch(
        small_llama.model.rotary_emb, seq_len, _HEAD_DIM, torch.float32
    )

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

    q = _interleaved_rope(q, cos_meta, sin_meta)
    k = _interleaved_rope(k, cos_meta, sin_meta)
    # The device stores and attends in Meta layout on both sides of the QK dot
    # product, which is rotation-invariant under the relayout, so no conversion
    # back to HF halves is needed here.
    groups = n_heads // n_kv_heads
    scores = (q @ repeat_kv(k, groups).transpose(-1, -2)) / (_HEAD_DIM**0.5) + causal
    attended = torch.softmax(scores, dim=-1) @ repeat_kv(v, groups)
    actual = attended.transpose(1, 2).reshape(1, seq_len, n_heads * _HEAD_DIM) @ wo.float()

    passing, message = comp_pcc(expected, actual, _EXACT_PCC)
    assert passing, f"converted attention weights do not reproduce HF attention: {message}"


@torch.no_grad()
def test_converted_mlp_weights_reproduce_the_hf_mlp_output(small_llama):
    mlp = small_llama.model.layers[0].mlp
    torch.manual_seed(13)
    x = torch.randn(1, 32, small_llama.config.hidden_size, dtype=torch.float32)

    expected = mlp(x.to(mlp.gate_proj.weight.dtype)).float()
    w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(mlp)
    actual = (torch.nn.functional.silu(x @ w1.float()) * (x @ w3.float())) @ w2.float()

    passing, message = comp_pcc(expected, actual, _EXACT_PCC)
    assert passing, f"converted MLP weights do not reproduce HF MLP: {message}"


@torch.no_grad()
def test_converted_lm_head_reproduces_the_hf_logits_on_the_real_vocabulary(small_llama):
    """Padding columns must be inert, and real columns must be untouched."""

    params_dim = small_llama.config.hidden_size
    vocab = small_llama.config.vocab_size
    padded = vocab + 64
    torch.manual_seed(17)
    x = torch.randn(4, params_dim, dtype=torch.float32)

    expected = small_llama.lm_head(x.to(small_llama.lm_head.weight.dtype)).float()
    weight = weight_utils.lm_head_weight_torch(
        small_llama.lm_head, dim=params_dim, vocab_size=vocab, padded_vocab_size=padded
    )
    assert weight.shape == (params_dim, padded)
    actual = x @ weight.float()
    assert torch.count_nonzero(actual[:, vocab:]) == 0, "padding columns are not inert"

    passing, message = comp_pcc(expected, actual[:, :vocab], _EXACT_PCC)
    assert passing, f"converted LM head does not reproduce HF logits: {message}"


# =============================================================================
# The real checkpoint
# =============================================================================


def _local_files_only() -> bool:
    return any(
        os.getenv(name, "").lower() in {"1", "true", "yes"} for name in ("CI", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _checkpoint_snapshot_or_skip(hf_model: str) -> Path:
    """Locate the resolved snapshot directory, or skip.

    Reading the safetensors shards directly is what makes a real-checkpoint
    assertion affordable: layer 0 plus the embedding and LM head live in three
    of the thirty shards, so this costs ~12 GB of I/O instead of 141 GB.
    """

    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                hf_model,
                allow_patterns=["config.json", "model.safetensors.index.json"],
                local_files_only=_local_files_only(),
            )
        )
    except BaseException as error:  # noqa: BLE001 - any resolution failure is a skip, not a defect
        pytest.skip(f"checkpoint {hf_model!r} is unavailable: {error}")


def _hf_config_or_skip(hf_model: str):
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(hf_model, local_files_only=_local_files_only())
    except BaseException as error:  # noqa: BLE001
        pytest.skip(f"checkpoint {hf_model!r} is unavailable: {error}")


@pytest.fixture(scope="module")
def real_checkpoint():
    hf_model = os.getenv("LLAMA33_70B_HF_MODEL", LLAMA33_70B_GALAXY_HF_MODEL)
    return hf_model, _hf_config_or_skip(hf_model), _checkpoint_snapshot_or_skip(hf_model)


@torch.no_grad()
def test_real_checkpoint_rope_tables_match_an_independent_llama3_scaling(real_checkpoint):
    """Rebuild the llama3-scaled frequencies from the paper and compare.

    ``build_rope_cos_sin_torch`` trusts the HF rotary module to have applied
    Llama 3's piecewise wavelength scaling. That trust is what this test
    removes: the expected inverse frequencies are recomputed here from
    ``rope_theta``, ``factor``, the two frequency factors and the original
    context length, with no HF rope helper involved.
    """

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    _, hf_config, _ = real_checkpoint
    params = parameters_from_hf_config(hf_config, n_layers=1)
    scaling = hf_config.rope_scaling
    table_len, head_dim = 512, params.head_dim

    cos, sin = weight_utils.build_rope_cos_sin_torch(
        LlamaRotaryEmbedding(config=hf_config), table_len, head_dim, torch.float32
    )

    # --- independent reference: Llama 3 RoPE scaling, from its definition ---
    base, factor = params.rope_theta, scaling["factor"]
    low_factor, high_factor = scaling["low_freq_factor"], scaling["high_freq_factor"]
    original = scaling["original_max_position_embeddings"]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    wavelength = 2 * torch.pi / inv_freq
    low_wavelength, high_wavelength = original / low_factor, original / high_factor
    smooth = (original / wavelength - low_factor) / (high_factor - low_factor)
    scaled = torch.where(wavelength > low_wavelength, inv_freq / factor, inv_freq)
    interpolated = (1 - smooth) * scaled / factor + smooth * scaled
    scaled = torch.where(
        (wavelength >= high_wavelength) & (wavelength <= low_wavelength),
        interpolated,
        scaled,
    )
    angles = torch.arange(table_len, dtype=torch.float64).unsqueeze(-1) * scaled.unsqueeze(0)

    # Meta layout duplicates each frequency into an adjacent pair.
    expected_cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).float()
    expected_sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).float()

    torch.testing.assert_close(cos[0, 0], expected_cos, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(sin[0, 0], expected_sin, rtol=2e-3, atol=2e-3)

    assert params.rope_scaling_factor == factor
    assert params.original_context_len == original


@torch.no_grad()
def test_real_checkpoint_layer0_converts_to_the_contracted_shapes(real_checkpoint):
    """Convert the real layer 0, read straight from its safetensors shards."""

    from safetensors.torch import load_file

    hf_model, hf_config, snapshot = real_checkpoint
    params = parameters_from_hf_config(hf_config, n_layers=1)
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip(f"no safetensors index in {snapshot}")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    wanted = {
        name: weight_map[name]
        for name in weight_map
        if name.startswith("model.layers.0.") or name in {"model.norm.weight"}
    }
    missing = [shard for shard in set(wanted.values()) if not (snapshot / shard).exists()]
    if missing:
        pytest.skip(f"checkpoint shards are not materialized locally: {sorted(missing)}")

    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(wanted.values())):
        loaded = load_file(str(snapshot / shard))
        tensors.update({name: loaded[name] for name in wanted if wanted[name] == shard})
        del loaded

    class _Attn:
        def __init__(self) -> None:
            self.config = hf_config
            self.q_proj = torch.nn.Linear(params.dim, params.n_heads * params.head_dim, bias=False)
            self.k_proj = torch.nn.Linear(params.dim, params.n_kv_heads * params.head_dim, bias=False)
            self.v_proj = torch.nn.Linear(params.dim, params.n_kv_heads * params.head_dim, bias=False)
            self.o_proj = torch.nn.Linear(params.n_heads * params.head_dim, params.dim, bias=False)

    attn = _Attn()
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(attn, projection).weight = torch.nn.Parameter(
            tensors[f"model.layers.0.self_attn.{projection}.weight"], requires_grad=False
        )

    wqkv, wo = weight_utils.attention_weights_from_hf_layer(attn, rows=_ROWS)
    geometry = params.geometry()
    assert wqkv.shape == (params.dim, geometry.qkv_size), (wqkv.shape, geometry.qkv_size)
    assert wo.shape == (params.n_heads * params.head_dim, params.dim)
    assert torch.isfinite(wqkv.float()).all() and torch.isfinite(wo.float()).all()

    class _Mlp:
        pass

    mlp = _Mlp()
    for projection in ("gate_proj", "up_proj", "down_proj"):
        linear = torch.nn.Linear(1, 1, bias=False)
        linear.weight = torch.nn.Parameter(tensors[f"model.layers.0.mlp.{projection}.weight"], requires_grad=False)
        setattr(mlp, projection, linear)

    w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(mlp)
    assert w1.shape == (params.dim, params.hidden_dim)
    assert w3.shape == (params.dim, params.hidden_dim)
    assert w2.shape == (params.hidden_dim, params.dim)

    norm = torch.nn.RMSNorm(params.dim)
    norm.weight = torch.nn.Parameter(tensors["model.norm.weight"], requires_grad=False)
    assert weight_utils.rms_weight_torch(norm).shape == (params.dim,)
