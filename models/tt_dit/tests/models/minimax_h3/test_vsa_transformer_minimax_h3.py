# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Model-level VSA integration on the 4x8 galaxy: the whole (depth-reduced) transformer with the
VSA path on, exercising the pack/unpack row gathers and tile-order metadata end to end.

R6a at model level: sparsity 0 + zero gate matches the dense (ring) model at PCC >= 0.9995 on the
same inputs -- ragged prefix tiles and pad tiles included, since both paths attend exactly the real
rows. R6d: striped placement equals identity placement on the final (unpacked) outputs.
"""

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3Transformer3DModel as TorchMiniMaxH3Transformer,
)
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSAConfig, MiniMaxH3VSACoarseStage
from ....pipelines.minimax_h3.vsa_geometry import build_vsa_geometry
from ....utils.check import assert_quality
from ....utils.tensor import from_torch
from ....utils.test import skip_if_unsupported_num_links
from .common import GALAXY_RING, randomize_norm_weights, upload_rope
from .test_transformer_minimax_h3 import (
    ATTENTION_HEAD_DIM,
    AUDIO_IN_CHANNELS,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TEXT_DIM,
    TRANSFORMER_CONFIG,
    VIDEO_PATCH_DIM,
    _modality_metadata,
    _prepare_tt_inputs,
)

# ragged prefix tiles AND pad tiles: text 512 -> 8 full tiles, audio 270 -> 5 tiles (tail 14),
# video 1280 on a (8, 8) frame grid -> 3D grid (20, 8, 8) -> 20 full tiles; 33 tiles -> pad to 40.
_SHAPE = dict(num_text=512, num_audio=270, num_video=1280, grid=(8, 8))


def _vsa_metadata(inputs, geometry, mesh_device, sp_axis):
    """Tile-order rope/adaln/timestep tensors for the VSA model, from the packed-order host arrays."""
    position_ids = geometry.permute_metadata(inputs.position_ids, dim=0)
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    with torch.no_grad():
        cos, sin = rope(position_ids)
    cos, sin = prepare_rope_tables(cos, sin, ATTENTION_HEAD_DIM)
    tt_cos, tt_sin = upload_rope(cos, sin, mesh_device=mesh_device, sp_axis=sp_axis)

    def rows(arr: torch.Tensor) -> ttnn.Tensor:
        tiled = geometry.permute_metadata(arr, dim=0)
        return from_torch(
            tiled.to(torch.int32).reshape(1, 1, 1, geometry.padded_len),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, sp_axis],
        )

    from .common import TAG_VIDEO  # noqa: F401  (tags are already merged into adaln indices)

    adaln = inputs.ts_idx * 3 + inputs.tags.clamp(min=0)
    return dict(rope_cos=tt_cos, rope_sin=tt_sin, adaln_indices=rows(adaln), timestep_indices=rows(inputs.ts_idx))


def _build_tt_model(inputs, mesh_device, is_fsdp, state, vsa_config, geometry, sp_axis):
    model = MiniMaxH3Transformer3DModel(
        **TRANSFORMER_CONFIG,
        mesh_device=mesh_device,
        ccl_manager=inputs.ccl_manager,
        parallel_config=inputs.parallel_config,
        is_fsdp=is_fsdp,
        vsa_config=vsa_config,
    )
    model.load_torch_state_dict(state)
    if vsa_config is not None:
        stage = MiniMaxH3VSACoarseStage(
            geometry,
            sparsity=vsa_config.sparsity,
            head_dim=ATTENTION_HEAD_DIM,
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            ccl_manager=inputs.ccl_manager,
        )
        model.set_vsa_stage(stage)
    return model


def _compose_replicated(mesh_device, t: ttnn.Tensor) -> torch.Tensor:
    out = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape))
    )
    return out.reshape(-1, *out.shape[2:])[:1]


@GALAXY_RING
@pytest.mark.timeout(2700)
def test_vsa_transformer_sparsity0_matches_dense(
    mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, reset_seeds
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)
    if tuple(mesh_device.shape)[sp_axis] != 8:
        pytest.skip("VSA v0 targets 4x8")
    MIN_PCC = 0.9995

    per_modality = _modality_metadata(_SHAPE["num_text"], _SHAPE["num_audio"], _SHAPE["num_video"], _SHAPE["grid"])
    inputs = _prepare_tt_inputs(
        mesh_device, sp_axis, tp_axis, num_links, topology, per_modality,
        text_dim=TEXT_DIM, video_patch_dim=VIDEO_PATCH_DIM, audio_channels=AUDIO_IN_CHANNELS,
        head_dim=ATTENTION_HEAD_DIM, rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA,
    )  # fmt: skip

    grid_t = _SHAPE["num_video"] // (_SHAPE["grid"][0] * _SHAPE["grid"][1])
    geometry = build_vsa_geometry(
        (_SHAPE["num_text"], 0, _SHAPE["num_audio"]), (grid_t, *_SHAPE["grid"]), sp_factor=8
    )
    assert geometry.n_pad_tiles > 0  # the shape is chosen to cover pad tiles and a ragged prefix tail

    torch_model = TorchMiniMaxH3Transformer(**TRANSFORMER_CONFIG, rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    torch_model = torch_model.to(torch.float32)
    randomize_norm_weights(torch_model)
    state = torch_model.state_dict()
    del torch_model

    dense = _build_tt_model(inputs, mesh_device, is_fsdp, state, None, None, sp_axis)
    dense_video, dense_audio = dense.forward(**inputs.tt)
    dense_video = _compose_replicated(mesh_device, dense_video)
    dense_audio = _compose_replicated(mesh_device, dense_audio)
    del dense

    vsa_model = _build_tt_model(
        inputs, mesh_device, is_fsdp, state, MiniMaxH3VSAConfig(sparsity=0.0), geometry, sp_axis
    )
    vsa_tt = {**inputs.tt, **_vsa_metadata(inputs, geometry, mesh_device, sp_axis)}
    vsa_video, vsa_audio = vsa_model.forward(**vsa_tt)
    vsa_video = _compose_replicated(mesh_device, vsa_video)
    vsa_audio = _compose_replicated(mesh_device, vsa_audio)

    logger.info("checking video output")
    assert_quality(dense_video, vsa_video, pcc=MIN_PCC)
    logger.info("checking audio output")
    assert_quality(dense_audio, vsa_audio, pcc=MIN_PCC)


@GALAXY_RING
@pytest.mark.timeout(2700)
def test_vsa_transformer_striped_matches_identity(
    mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, reset_seeds
) -> None:
    """R6d at model level: striped placement gives the same outputs as identity after unpacking."""
    skip_if_unsupported_num_links(mesh_device, num_links)
    if tuple(mesh_device.shape)[sp_axis] != 8:
        pytest.skip("VSA v0 targets 4x8")
    MIN_PCC = 0.9999  # same math, different device order -> bf16 reduction-order noise only

    per_modality = _modality_metadata(_SHAPE["num_text"], _SHAPE["num_audio"], _SHAPE["num_video"], _SHAPE["grid"])
    inputs = _prepare_tt_inputs(
        mesh_device, sp_axis, tp_axis, num_links, topology, per_modality,
        text_dim=TEXT_DIM, video_patch_dim=VIDEO_PATCH_DIM, audio_channels=AUDIO_IN_CHANNELS,
        head_dim=ATTENTION_HEAD_DIM, rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA,
    )  # fmt: skip

    torch_model = TorchMiniMaxH3Transformer(**TRANSFORMER_CONFIG, rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    torch_model = torch_model.to(torch.float32)
    randomize_norm_weights(torch_model)
    state = dict(torch_model.state_dict())
    torch.manual_seed(3)
    for name in list(state):
        # give every VSA block a nonzero gate to exercise O_c too (not the token refiner's attention)
        if name.startswith("transformer_blocks.") and name.endswith("attn.to_q.weight"):
            state[name.replace("to_q", "to_gate_compress")] = 0.02 * torch.randn(56 * 128, 5376)
    del torch_model

    grid_t = _SHAPE["num_video"] // (_SHAPE["grid"][0] * _SHAPE["grid"][1])
    outs = {}
    for placement in ("identity", "striped"):
        geometry = build_vsa_geometry(
            (_SHAPE["num_text"], 0, _SHAPE["num_audio"]), (grid_t, *_SHAPE["grid"]), sp_factor=8, placement=placement
        )
        model = _build_tt_model(
            inputs, mesh_device, is_fsdp, state,
            MiniMaxH3VSAConfig(sparsity=0.75, placement=placement, k_chunk_blocks=2), geometry, sp_axis,
        )  # fmt: skip
        assert not model.transformer_blocks[0].attn.gate_compress_is_zero
        vsa_tt = {**inputs.tt, **_vsa_metadata(inputs, geometry, mesh_device, sp_axis)}
        video, audio = model.forward(**vsa_tt)
        outs[placement] = (
            _compose_replicated(mesh_device, video),
            _compose_replicated(mesh_device, audio),
        )
        del model

    logger.info("checking video output")
    assert_quality(outs["identity"][0], outs["striped"][0], pcc=MIN_PCC)
    logger.info("checking audio output")
    assert_quality(outs["identity"][1], outs["striped"][1], pcc=MIN_PCC)
