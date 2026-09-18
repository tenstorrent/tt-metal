# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the Llama-3.1-8B prefill decoder layer (tt-blaze#4146).

The layer is the first place the TP-sharded residual contract is actually exercised end to end:
``tt/mlp.py`` and ``tt/attention.py`` were each written to consume a replicated full-width
activation and return a ``reduce_scatter``-ed shard, and ``tt/rms_norm.py`` is what closes the loop
by widening the shard back. A mismatch anywhere in that chain is a shape error, but a mismatch in
*which* collective runs where is not — it is a PCC failure, which is what these grade.

Device cases:

  1. ``single-card-tp1`` — one chip, TP=1. No collective at all: the norms see full width already
     and both submodules return before their reduce-scatter. Isolates the layer's arithmetic
     (pre-norm placement, the two residual adds) from the distribution.

  2. ``tp8-1x8`` — eight chips, TP=8: the production per-chip width, a TP-sharded residual
     (512/chip), and all four per-layer collectives live.

  3. ``two-layers-tp8`` — the same 1x8 mesh run through two stacked layers with different weights,
     against two stacked reference layers. One layer cannot catch an error in the *output* layout,
     because the test composes the output back to full width before comparing either way; stacking
     makes layer 1 consume layer 0's actual device output, so a residual that came back in the
     wrong layout fails here instead of in the full model.

Run (single card):
    pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_decoder_vs_ref.py
Eight-chip loudbox (all):
    pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_decoder_vs_ref.py -k tp8
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.reference.model import Llama31DecoderLayer, Llama31RMSNorm, build_hf_cos_sin
from models.demos.llama_3p1_8b_d_p.tests.mesh_profiles import drop_sp_replicas, galaxy_torus_xy_device_params
from models.demos.llama_3p1_8b_d_p.tt.ccl import CCLManager
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.decoder import TtLlamaDecoderLayer
from models.demos.llama_3p1_8b_d_p.tt.rms_norm import TtLlamaRMSNorm
from models.demos.llama_3p1_8b_d_p.tt.rope import build_llama3_cos_sin, build_transformation_mat

EMB_DIM = Llama31_8BConfig.EMB_SIZE
PCC_REQUIRED = 0.99
# The norm is graded tighter than the layer: it is one reduction over 4096 elements with no matmul
# in it, so anything below this is a real arithmetic difference rather than accumulated bf16 error.
PCC_REQUIRED_NORM = 0.999
# bf16 activations move the output RMS by well under a percent; 2% leaves room without admitting the
# order-of-magnitude rescale _assert_scale_matches exists to catch.
NORM_SCALE_RTOL = 0.02


def _assert_scale_matches(expected: torch.Tensor, got: torch.Tensor, *, rtol: float, what: str) -> None:
    """Reject a uniformly rescaled output, which ``comp_pcc`` cannot see (tt-blaze#1314).

    A double bit-cast of epsilon makes the denominator ``sqrt(garbage)`` instead of the row RMS, so
    the op returns ``x * gamma * tiny_const``. That correlates ~1.0 with the true norm, so a
    PCC-only test passes at ~0.9998 while the normalisation and its reduce path never ran at all.
    Comparing absolute scale is what separates "correlates" from "is equal", and it is cheap.
    """
    expected_rms = expected.float().pow(2).mean().sqrt()
    got_rms = got.float().pow(2).mean().sqrt()
    ratio = (got_rms / expected_rms).item()
    assert abs(ratio - 1.0) <= rtol, (
        f"{what}: output RMS is {ratio:.4g}x the reference's. The values correlate but the scale is "
        f"wrong, which is the tt-blaze#1314 epsilon false-pass — PCC alone cannot see it."
    )


def _replicated(t: torch.Tensor, mesh_device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(
        t, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device), layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=dtype
    )


def _residual_to_device(t: torch.Tensor, mesh_device, mesh_config, tp: int) -> ttnn.Tensor:
    """Put a full-width ``[1, 1, seq, emb]`` residual on the mesh in the layer's input layout."""
    if tp == 1:
        return _replicated(t, mesh_device)
    return ttnn.from_torch(
        t,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )


def _residual_to_host(tt: ttnn.Tensor, mesh_device, tp: int, shape) -> torch.Tensor:
    if tp == 1:
        out = ttnn.to_torch(ttnn.get_device_tensors(tt)[0])
    else:
        out = drop_sp_replicas(
            ttnn.to_torch(
                tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1))
            ),
            mesh_device.shape[0],
        )
    return out.reshape(shape).to(torch.float32)


def _rope_and_ccl(mesh_device, seq_len: int, tp: int):
    cos, sin = build_llama3_cos_sin(seq_len)
    rope_mats = [_replicated(cos, mesh_device), _replicated(sin, mesh_device)]
    return rope_mats, build_transformation_mat(mesh_device), (CCLManager(mesh_device, num_links=1) if tp > 1 else None)


# =====================================================================================
# Device-free
# =====================================================================================
def test_decoder_module_is_import_light():
    """Importing ``tt.decoder`` must not drag in reference modelling or checkpoint loading.

    This is the module the prefill model imports, so it is the one that actually has to stay cheap;
    it transitively imports attention, the MLP, the norm and the KV cache, so a reference import
    sneaking into any of them fails here.
    """
    forbidden = ("safetensors", "transformers", "models.demos.llama_3p1_8b_d_p.reference.model")
    probe = (
        "import sys;"
        "import models.demos.llama_3p1_8b_d_p.tt.decoder;"
        f"print(','.join(m for m in {forbidden!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"tt.decoder import pulled in {out}"


def test_layer_weight_slice_matches_reference_parameter_names(expect_error):
    """``torch_weights`` is keyed exactly as HF names a layer's parameters.

    The reference's own ``named_parameters()`` were chosen to match HF, so if the two agree here a
    checkpoint slice can be handed to the device layer with no renaming — and any future rename on
    either side is caught immediately rather than becoming a silently random weight.
    """
    reference_keys = set(dict(Llama31DecoderLayer().named_parameters()))
    expected = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }
    assert reference_keys == expected

    state_dict = {f"model.layers.7.{key}": torch.zeros(1) for key in expected}
    state_dict["model.layers.8.mlp.up_proj.weight"] = torch.zeros(1)  # a neighbour that must not leak
    sliced = TtLlamaDecoderLayer.weights_from_layer_state_dict(state_dict, 7)
    assert set(sliced) == expected

    with expect_error(KeyError, "model.layers.31"):
        TtLlamaDecoderLayer.weights_from_layer_state_dict(state_dict, 31)


def test_scale_check_catches_the_epsilon_false_pass(expect_error):
    """The guard in ``test_norm_vs_ref`` is itself tested, against the bug it is named for.

    tt-blaze#1314: a double bit-cast of epsilon left a garbage-huge denominator, so RMSNorm returned
    ``x * gamma * tiny_const`` and the micro-op test passed at ~0.9998 without ever exercising the
    normalisation. Reproduce that output exactly and assert the two halves of the claim: ``comp_pcc``
    accepts it, and ``_assert_scale_matches`` does not.

    Without this, the guard could be vacuous — comparing a ratio that is 1 by construction, say —
    and ``test_norm_vs_ref`` would look protected while being exactly as blind as before.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 1, 128, EMB_DIM)
    gamma = 1.0 + torch.randn(EMB_DIM) * 0.05

    reference = Llama31RMSNorm().eval()
    with torch.no_grad():
        reference.weight.copy_(gamma)
        good = reference(x)

    # The bug's output: the denominator is sqrt(garbage epsilon) rather than each row's RMS, which
    # is a single constant factor across the whole tensor.
    bad = x * gamma / math.sqrt(1e30)

    passing, pcc = comp_pcc(good, bad, PCC_REQUIRED_NORM)
    assert passing, f"precondition failed: PCC should be blind to a pure rescale, got {pcc}"

    with expect_error(AssertionError, "output RMS is"):
        _assert_scale_matches(good, bad, rtol=NORM_SCALE_RTOL, what="RMSNorm")


# =====================================================================================
# Device PCC
# =====================================================================================
@pytest.mark.parametrize("seq_len", [256], ids=["s256"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((1, 1), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-card-tp1"),
        pytest.param((1, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="tp8-1x8"),
        # Production TP=8 with the four SP rows replicating; the only arm a Galaxy can open.
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_norm_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """RMSNorm alone: TP-sharded in, replicated full-width out, vs HF's float32-upcast norm.

    Graded separately from the layer because the norm is the one submodule whose *output* layout is
    the opposite of everything else's, and a norm that returned a shard would still let the layer
    run (attention would reject it, but only on the last-dim check — not on numerics).
    """
    torch.manual_seed(0)
    rows, cols = mesh_device.shape
    tp = cols
    mesh_config = MeshConfig((rows, cols), tp=tp, tp_axis=1)

    reference = Llama31RMSNorm().eval()
    with torch.no_grad():
        reference.weight.copy_(1.0 + torch.randn(EMB_DIM) * 0.05)

    torch_input = torch.randn(1, 1, seq_len, EMB_DIM)
    with torch.no_grad():
        torch_output = reference(torch_input)

    tt_norm = TtLlamaRMSNorm(
        mesh_device=mesh_device, mesh_config=mesh_config, torch_weight=reference.weight.detach(), emb_dim=EMB_DIM
    )
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    tt_out = tt_norm(_residual_to_device(torch_input, mesh_device, mesh_config, tp), ccl_manager)
    ttnn.synchronize_device(mesh_device)

    assert tt_out.shape[-1] == EMB_DIM, f"norm must return full width for attention/MLP, got {tt_out.shape[-1]}"
    got = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(torch_output.shape).to(torch.float32)

    passing, pcc = comp_pcc(torch_output, got, PCC_REQUIRED_NORM)
    logger.info(f"RMSNorm PCC: {pcc}")
    assert passing, f"RMSNorm PCC {pcc} below {PCC_REQUIRED_NORM}"
    # PCC is scale-invariant, so it grades direction only. Without this the whole normalisation
    # could be off by a constant factor and this test would still pass — see tt-blaze#1314.
    _assert_scale_matches(torch_output, got, rtol=NORM_SCALE_RTOL, what="RMSNorm")


@pytest.mark.parametrize("seq_len", [256], ids=["s256"])
@pytest.mark.parametrize("num_layers", [1, 2], ids=["one-layer", "two-layers"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((1, 1), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-card-tp1"),
        pytest.param((1, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="tp8-1x8"),
        # Production TP=8 with the four SP rows replicating; the only arm a Galaxy can open.
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_decoder_vs_ref(mesh_device, device_params, seq_len, num_layers, reset_seeds):
    """Decoder layer(s) vs the torch reference, PCC >= 0.99, residual TP-sharded throughout."""
    torch.manual_seed(0)
    rows, cols = mesh_device.shape
    tp = cols
    mesh_config = MeshConfig((rows, cols), tp=tp, tp_axis=1)

    references = []
    for _ in range(num_layers):
        layer = Llama31DecoderLayer().eval()
        with torch.no_grad():
            # Default-initialised norms are all ones, which makes gamma a no-op and would hide a
            # dropped or misplaced norm weight.
            layer.input_layernorm.weight.copy_(1.0 + torch.randn(EMB_DIM) * 0.05)
            layer.post_attention_layernorm.weight.copy_(1.0 + torch.randn(EMB_DIM) * 0.05)
        references.append(layer)

    tt_layers = [
        TtLlamaDecoderLayer(
            mesh_device=mesh_device,
            mesh_config=mesh_config,
            torch_weights={k: v.detach() for k, v in reference.named_parameters()},
            layer_idx=index,
        )
        for index, reference in enumerate(references)
    ]
    logger.info(f"mesh={rows}x{cols} tp={tp} layers={num_layers} residual {tt_layers[0].emb_dim_per_chip}/chip")

    torch_input = torch.randn(1, seq_len, EMB_DIM, dtype=torch.float32)
    cos_hf, sin_hf = build_hf_cos_sin(torch.arange(seq_len))
    with torch.no_grad():
        torch_output = torch_input
        for reference in references:
            torch_output, _ = reference(torch_output, cos_hf, sin_hf)

    rope_mats, transformation_mat, ccl_manager = _rope_and_ccl(mesh_device, seq_len, tp)
    tt_x = _residual_to_device(torch_input.unsqueeze(0), mesh_device, mesh_config, tp)
    for tt_layer in tt_layers:
        tt_x = tt_layer(tt_x, rope_mats, transformation_mat, ccl_manager=ccl_manager)
    ttnn.synchronize_device(mesh_device)

    expected_width = EMB_DIM if tp == 1 else EMB_DIM // tp
    assert tt_x.shape[-1] == expected_width, (
        f"the layer must return the residual in the layout it consumed ({expected_width}/chip), "
        f"got {tt_x.shape[-1]} — stacking layers depends on this"
    )

    got = _residual_to_host(tt_x, mesh_device, tp, torch_output.unsqueeze(0).shape).reshape(torch_output.shape)
    assert not torch.isnan(got).any(), "NaN in decoder output"
    assert not torch.isinf(got).any(), "Inf in decoder output"

    passing, pcc = comp_pcc(torch_output, got, PCC_REQUIRED)
    logger.info(f"decoder ({num_layers} layer(s), tp={tp}) PCC: {pcc}")
    assert passing, f"decoder PCC {pcc} below {PCC_REQUIRED}"
