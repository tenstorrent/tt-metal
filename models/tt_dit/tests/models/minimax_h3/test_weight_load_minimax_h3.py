# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Weight-load gate for the MiniMax-H3 DiT at TP=4 / SP=8.

Loading is checked before any forward because the two layout fixups it performs
are silent when wrong -- they produce plausible video, not an error:

- fused QKV arrives per-head interleaved and must become ``[q_all; k_all; v_all]``
  *and then* be interleaved per device, or a device gets a quarter of ``q``
  instead of a slice of each of q/k/v;
- ``fc1`` arrives as ``[gate; value]`` while tt_dit's swiglu reads ``[value; gate]``.

Both are asserted directly against hand-computed expectations rather than only
through the absence of load errors.

The synthetic state dict mirrors the real checkpoint's key set and shapes exactly
(verified against ``FL2VA/transformer``: 535 tensors, 522 bf16 + 13 fp32), so this
exercises the whole traversal without reading 40 GB.
"""

import pytest
import torch

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import reorder_interleaved_qkv
from ....models.transformers.minimax_h3.transformer_minimax_h3 import (
    MINIMAX_H3_HOST_OWNED_PREFIXES,
    MINIMAX_H3_HOST_OWNED_SUFFIXES,
    MiniMaxH3Transformer3DModel,
)
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.test import ring_params

HIDDEN = 5376
NUM_HEADS = 56
HEAD_DIM = 128
INNER = NUM_HEADS * HEAD_DIM
FFN = 14336
TEXT_DIM = 5120
VIDEO_PATCH = 24 * 1 * 2 * 2
AUDIO_LATENTS = 32
TIME_EMBED_DIM = 2688
ADALN_OUT = 18 * HIDDEN

# Kept small: the fixups and the key traversal do not depend on depth, and a
# 50-layer instantiation on a 4x8 mesh is minutes of allocation for no signal.
NUM_LAYERS = 2
REFINER_LAYERS = 2

FP32_KEYS = frozenset(
    {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)


def _block_keys(prefix):
    return {
        f"{prefix}.norm1.weight": [HIDDEN],
        f"{prefix}.norm2.weight": [HIDDEN],
        f"{prefix}.attn.qkv_proj.weight": [3 * INNER, HIDDEN],
        f"{prefix}.attn.out_proj.weight": [HIDDEN, INNER],
        f"{prefix}.attn.q_norm.weight": [HEAD_DIM],
        f"{prefix}.attn.k_norm.weight": [HEAD_DIM],
        f"{prefix}.mlp.fc1.weight": [2 * FFN, HIDDEN],
        f"{prefix}.mlp.fc2.weight": [HIDDEN, FFN],
    }


def checkpoint_shapes(num_layers=NUM_LAYERS, refiner_layers=REFINER_LAYERS, include_host_owned=True):
    """The real checkpoint's key set and shapes, at a reduced layer count."""
    shapes = {
        "video_patch_proj.weight": [HIDDEN, VIDEO_PATCH],
        "video_patch_proj.bias": [HIDDEN],
        "audio_patch_proj.weight": [HIDDEN, AUDIO_LATENTS],
        "audio_patch_proj.bias": [HIDDEN],
        "condition_proj.weight": [HIDDEN, TEXT_DIM],
        "condition_proj.bias": [HIDDEN],
        "token_refiner.final_norm.weight": [HIDDEN],
        "final_layer.norm.weight": [HIDDEN],
        "final_layer.video_out.weight": [VIDEO_PATCH, HIDDEN],
        "final_layer.video_out.bias": [VIDEO_PATCH],
        "final_layer.audio_out.weight": [AUDIO_LATENTS, HIDDEN],
        "final_layer.audio_out.bias": [AUDIO_LATENTS],
    }
    for layer in range(num_layers):
        shapes.update(_block_keys(f"blocks.{layer}"))
    for layer in range(refiner_layers):
        shapes.update(_block_keys(f"token_refiner.blocks.{layer}"))

    if include_host_owned:
        # Present in the checkpoint but owned by the host-side precompute.
        shapes.update(
            {
                "time_embedder.proj_in.weight": [HIDDEN, 256],
                "time_embedder.proj_in.bias": [HIDDEN],
                "time_embedder.proj_out.weight": [TIME_EMBED_DIM, HIDDEN],
                "time_embedder.proj_out.bias": [TIME_EMBED_DIM],
                "rope.inv_freq": [16],
                "final_layer.adaln_proj.linear.weight": [2 * HIDDEN, TIME_EMBED_DIM],
                "final_layer.adaln_proj.linear.bias": [2 * HIDDEN],
            }
        )
        for layer in range(num_layers):
            shapes[f"blocks.{layer}.adaln_proj.linear.weight"] = [ADALN_OUT, TIME_EMBED_DIM]
            shapes[f"blocks.{layer}.adaln_proj.linear.bias"] = [ADALN_OUT]
    return shapes


def synthetic_state(seed=0, **kwargs):
    generator = torch.Generator().manual_seed(seed)
    return {
        key: torch.randn(shape, generator=generator, dtype=torch.float32 if key in FP32_KEYS else torch.float32).to(
            torch.float32 if key in FP32_KEYS else torch.bfloat16
        )
        for key, shape in checkpoint_shapes(**kwargs).items()
    }


def _build(mesh_device, mesh_shape, sp_axis, tp_axis, num_links):
    parallel_config = DiTParallelConfig.from_tuples(
        cfg=(1, 0), sp=(mesh_shape[sp_axis], sp_axis), tp=(mesh_shape[tp_axis], tp_axis)
    )
    ccl_manager = CCLManager(mesh_device, num_links=num_links, topology=ttnn.Topology.Ring)
    model = MiniMaxH3Transformer3DModel(
        num_layers=NUM_LAYERS,
        token_refiner_num_layers=REFINER_LAYERS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    return model, parallel_config


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params"),
    [pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, id="bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
def test_weight_load_consumes_every_key(mesh_device, mesh_shape, sp_axis, tp_axis, num_links):
    """Strict load: no missing keys, no unexpected keys."""
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    model, _ = _build(mesh_device, mesh_shape, sp_axis, tp_axis, num_links)

    # strict=True raises on any missing or unexpected key, which is the gate.
    incompatible = model.load_torch_state_dict(synthetic_state(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert model.is_loaded()


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params"),
    [pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, id="bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
def test_shard_shapes_and_dtypes(mesh_device, mesh_shape, sp_axis, tp_axis, num_links):
    """Per-device shard widths and the fp32 parameters that must stay fp32."""
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    model, _ = _build(mesh_device, mesh_shape, sp_axis, tp_axis, num_links)
    model.load_torch_state_dict(synthetic_state(), strict=True)

    tp = mesh_shape[tp_axis]
    attention = model.blocks[0].attn
    mlp = model.blocks[0].mlp

    # ColParallel weights are stored [in, out] with out sharded on the TP axis.
    assert attention.qkv_proj.weight.local_shape == (HIDDEN, 3 * INNER // tp)
    assert attention.out_proj.weight.local_shape == (INNER, HIDDEN // tp)
    assert mlp.fc1.weight.local_shape == (HIDDEN, 2 * FFN // tp)
    # RowParallel shards the input dim instead.
    assert mlp.fc2.weight.local_shape == (FFN // tp, HIDDEN)

    # 5376/4 = 1344 and 14336/4 = 3584: both tile-aligned, per the port plan.
    assert HIDDEN % (32 * tp) == 0
    assert (2 * FFN) % (32 * tp) == 0

    # Replicated on purpose: 96 and 32 columns would go sub-tile across TP=4.
    assert model.final_layer.video_out.weight.local_shape == (HIDDEN, VIDEO_PATCH)
    assert model.final_layer.audio_out.weight.local_shape == (HIDDEN, AUDIO_LATENTS)
    assert model.video_patch_proj.weight.local_shape == (VIDEO_PATCH, HIDDEN)

    for parameter in (
        model.video_patch_proj.weight,
        model.video_patch_proj.bias,
        model.audio_patch_proj.weight,
        model.final_layer.video_out.weight,
        model.final_layer.audio_out.weight,
    ):
        assert parameter.dtype == ttnn.float32


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params"),
    [pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, id="bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
def test_qkv_and_fc1_fixups_land_on_device(mesh_device, mesh_shape, sp_axis, tp_axis, num_links):
    """Read the loaded shards back and check both fixups actually happened.

    Device 0 of the TP axis must hold the first ``inner/tp`` columns of each of
    q, k and v -- not the first quarter of q -- and ``fc1``'s first half must be
    the checkpoint's *second* half.
    """
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    model, _ = _build(mesh_device, mesh_shape, sp_axis, tp_axis, num_links)
    state = synthetic_state()
    model.load_torch_state_dict({k: v.clone() for k, v in state.items()}, strict=True)

    tp = mesh_shape[tp_axis]
    local_inner = INNER // tp

    reordered = reorder_interleaved_qkv(state["blocks.0.attn.qkv_proj.weight"], NUM_HEADS, HEAD_DIM)
    query, key, value = reordered.split(INNER, dim=0)
    expected_shard0 = torch.cat([part[:local_inner] for part in (query, key, value)], dim=0).transpose(0, 1)

    loaded = ttnn.to_torch(
        model.blocks[0].attn.qkv_proj.weight.data,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_shape)),
    )
    # dims=(0,1) tiles the replicated SP axis along rows; one TP shard is the
    # first `HIDDEN` rows and `3 * local_inner` columns.
    shard0 = loaded[:HIDDEN, : 3 * local_inner].to(torch.bfloat16)
    assert torch.equal(shard0, expected_shard0.to(torch.bfloat16))

    # A naive column shard would have given q's first quarter instead.
    naive = query[: 3 * local_inner].transpose(0, 1).to(torch.bfloat16)
    assert not torch.equal(shard0, naive)

    gate, value_half = state["blocks.0.mlp.fc1.weight"].chunk(2, dim=0)
    fc1 = ttnn.to_torch(
        model.blocks[0].mlp.fc1.weight.data,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_shape)),
    )[:HIDDEN]
    # ColParallelLinear interleaves [value | gate] per device, so device 0's
    # columns are value[:FFN/tp] then gate[:FFN/tp].
    per_device = FFN // tp
    assert torch.equal(
        fc1[:, :per_device].to(torch.bfloat16), value_half[:per_device].transpose(0, 1).to(torch.bfloat16)
    )
    assert torch.equal(
        fc1[:, per_device : 2 * per_device].to(torch.bfloat16),
        gate[:per_device].transpose(0, 1).to(torch.bfloat16),
    )


def test_host_owned_keys_are_dropped_not_silently_accepted():
    """The AdaLN, time-embedder and rope keys must be excluded deliberately."""
    shapes = checkpoint_shapes(include_host_owned=True)
    host_owned = [
        key
        for key in shapes
        if key.startswith(MINIMAX_H3_HOST_OWNED_PREFIXES) or key.endswith(MINIMAX_H3_HOST_OWNED_SUFFIXES)
    ]
    # 4 time embedder + rope.inv_freq + 2 final adaln + 2 per block.
    assert len(host_owned) == 4 + 1 + 2 + 2 * NUM_LAYERS
    assert "blocks.0.adaln_proj.linear.weight" in host_owned
    assert "time_embedder.proj_in.weight" in host_owned
    assert "rope.inv_freq" in host_owned
    # Refiner blocks have no AdaLN in the checkpoint either.
    assert not any("token_refiner" in key and "adaln" in key for key in shapes)
