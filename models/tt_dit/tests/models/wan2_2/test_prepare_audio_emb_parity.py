# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity test for the S2V audio path.

Compares our on-device :class:`CausalAudioEncoder` + the slice/reshape logic
from :meth:`WanS2VTransformer3DModel.prepare_audio_emb` against the
reference :class:`wan.modules.s2v.audio_utils.CausalAudioEncoder` followed
by the slice lines from ``WanModel_S2V.forward`` (model_s2v.py:682-694).

Test bar: PCC ≥ 0.99.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.wan2_2.audio_utils_wan import CausalAudioEncoder
from ....utils.check import assert_quality
from ....utils.tensor import local_device_to_torch
from ....utils.test import line_params
from .test_s2v_components import _install_wan_ref_stubs

_REF_REPO = Path("/home/kevinmi/wan2_2_ref")


# Reduced config — same shape contract as production but smaller.
# Production:  audio_dim=1024, num_layers=25, out_dim=5120, num_token=4.
# We shrink out_dim (the DiT inner dim — only affects MotionEncoder's last
# conv channel count) for fast CPU reference.
AUDIO_DIM = 1024
NUM_LAYERS = 25
OUT_DIM = 256
NUM_TOKEN = 4
MOTION_FRAMES = (17, 5)  # production [encoded_frames, latent_frames]
T_AUDIO = 80  # ≥ motion_frames[0] + small video extent at video_rate


def _ref_to_tt_audio_encoder_state(ref_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate reference CausalAudioEncoder state-dict to our naming.

    Reference and ours share the same module names — ``weights``,
    ``encoder.conv1_local.conv.{weight,bias}``, ``encoder.conv2...``,
    ``encoder.conv3...``, ``encoder.padding_tokens``. The conv submodules
    nest the actual nn.Conv1d as ``.conv``; our CausalConv1d
    `_prepare_torch_state` handles `conv.weight` -> `weight` and inserts the
    spatial-1 dims.
    """
    return dict(ref_state)


@pytest.mark.skipif(
    not (_REF_REPO / "wan" / "modules" / "s2v" / "audio_utils.py").exists(),
    reason="Wan-Video/Wan2.2 reference repo not at /home/kevinmi/wan2_2_ref",
)
@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8),
            (4, 8),
            1,
            0,
            2,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_4x8sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_prepare_audio_emb_parity(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """End-to-end audio path: CausalAudioEncoder + slice off motion_frames[1].

    Compares the ``[B, T_video, num_token+1, out_dim]`` tensor that the
    reference stores in ``self.merged_audio_emb`` (and that we flatten to
    ``[1, B, T_video*(num_token+1), out_dim]`` for cross-attn K/V).
    """
    torch.manual_seed(0)
    if str(_REF_REPO) not in sys.path:
        sys.path.insert(0, str(_REF_REPO))
    _install_wan_ref_stubs()

    from wan.modules.s2v.audio_utils import CausalAudioEncoder as RefCausalAudioEncoder

    # ---- Build reference (CPU) ----
    ref = (
        RefCausalAudioEncoder(
            dim=AUDIO_DIM, num_layers=NUM_LAYERS, out_dim=OUT_DIM, num_token=NUM_TOKEN, need_global=False
        )
        .eval()
        .to(torch.float32)
    )
    logger.info(f"Reference CausalAudioEncoder built: dim={AUDIO_DIM}, num_layers={NUM_LAYERS}, out_dim={OUT_DIM}")

    # ---- Build ours (on device) ----
    tt = CausalAudioEncoder(
        dim=AUDIO_DIM,
        num_layers=NUM_LAYERS,
        out_dim=OUT_DIM,
        num_token=NUM_TOKEN,
        need_global=False,
        mesh_device=mesh_device,
    )

    # Weight transfer ref -> ours. Names line up 1:1 modulo our CausalConv1d
    # _prepare_torch_state handling of ``conv.weight`` -> ``weight`` (unsqueeze
    # spatial 1×1). The aggregation weights and padding_tokens transfer raw.
    ref_sd = ref.state_dict()
    incompat = tt.load_torch_state_dict(_ref_to_tt_audio_encoder_state(ref_sd), strict=False)
    logger.info(
        f"TT CausalAudioEncoder load: missing={len(incompat.missing_keys)} "
        f"unexpected={len(incompat.unexpected_keys)}"
    )
    if incompat.missing_keys:
        logger.warning(f"missing keys: {incompat.missing_keys[:5]}")
    if incompat.unexpected_keys:
        logger.warning(f"unexpected keys: {incompat.unexpected_keys[:5]}")

    # ---- Same audio input on both sides ----
    audio_input_torch = torch.randn(1, NUM_LAYERS, AUDIO_DIM, T_AUDIO, dtype=torch.float32)

    # Replicate the reference's left-pad-by-motion_frames[0] step
    # (wan/modules/s2v/model_s2v.py:683):
    pre = audio_input_torch[..., :1].expand(-1, -1, -1, MOTION_FRAMES[0])
    audio_input_padded = torch.cat([pre, audio_input_torch], dim=-1)

    # ---- Reference forward ----
    with torch.no_grad():
        ref_out = ref(audio_input_padded)  # [B, T, num_token+1, out_dim]
        ref_merged = ref_out[:, MOTION_FRAMES[1] :, :]  # slice off motion-latent prefix
    logger.info(f"Reference merged_audio_emb: shape={tuple(ref_merged.shape)}")

    # ---- Our forward ----
    tt_out_dev = tt(audio_input_padded)
    tt_out_torch = local_device_to_torch(tt_out_dev)  # [B, T, num_token+1, out_dim]
    # Slice off motion-latent prefix (matches our prepare_audio_emb).
    tt_merged = tt_out_torch[:, MOTION_FRAMES[1] :, :, :]
    logger.info(f"TT merged_audio_emb: shape={tuple(tt_merged.shape)}")

    # ---- Compare ----
    assert (
        tt_merged.shape == ref_merged.shape
    ), f"shape mismatch: tt={tuple(tt_merged.shape)} ref={tuple(ref_merged.shape)}"
    assert_quality(tt_merged.float(), ref_merged.float(), pcc=0.99)
