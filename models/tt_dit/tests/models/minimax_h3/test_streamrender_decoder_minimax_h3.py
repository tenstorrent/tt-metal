# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""StreamRender's causal ViT-24 decoder on TT against Sana's torch `CausalViTDecoder.decode_chunk`.

Six consecutive chunks cover an empty, a half and a full K/V cache, both alignment-frame trims (latents 0 and 5)
and the steady-state trace's capture and replays. Needs a Sana `models/streamrender_h3` checkout and the released student weights:

    STREAMRENDER_SANA_ROOT=.../streamrender_h3 STREAMRENDER_DECODER_WEIGHTS=.../causal_vit24_step6000.safetensors
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn

from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.streamrender.causal_decoder import CausalViTDecoderTT
from ....pipelines.minimax_h3.streamrender.sana_host import _pixel_revert, mount_sana
from ....utils.check import assert_quality
from ....utils.test import ring_params_8k

SANA_ROOT = os.environ.get("STREAMRENDER_SANA_ROOT")
WEIGHTS = os.environ.get("STREAMRENDER_DECODER_WEIGHTS")


@pytest.mark.skipif(not (SANA_ROOT and WEIGHTS), reason="needs STREAMRENDER_SANA_ROOT and STREAMRENDER_DECODER_WEIGHTS")
@pytest.mark.parametrize(
    ("mesh_device", "device_params", "sp_axis", "height", "width", "rgb"),
    [
        pytest.param((1, 1), {"l1_small_size": 65536}, None, 24, 40, False, id="1x1-24x40"),
        pytest.param((1, 1), {"l1_small_size": 65536}, None, 48, 84, False, id="1x1-768p"),
        # Windows sharded over the 8-device axis, replicated over the 4-device one, as the worker runs it.
        pytest.param((4, 8), {**ring_params_8k, "l1_small_size": 65536}, 1, 48, 84, False, id="4x8-768p"),
        pytest.param((4, 8), {**ring_params_8k, "l1_small_size": 65536}, 1, 48, 84, True, id="4x8-768p-rgb"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_streamrender_causal_decoder(
    mesh_device: ttnn.MeshDevice, sp_axis: int | None, height: int, width: int, rgb: bool, reset_seeds
) -> None:
    MIN_PCC = 0.99
    root = Path(SANA_ROOT)
    mount_sana(root)
    from causal_vit_codec import CausalViTVideoDecoder

    settings = json.loads((root / "configs/runtime.json").read_text())
    args = dict(settings["inference"]["models"]["video_decoder"]["args"])
    codec = CausalViTVideoDecoder(args["decoder_config"], WEIGHTS, cuda_graph=False)
    reference = codec.vit.float().eval()

    ccl_manager = (
        None if sp_axis is None else CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    )
    tt_decoder = CausalViTDecoderTT(
        reference,
        height=height,
        width=width,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        sp_axis=sp_axis,
        rgb=_pixel_revert(codec.processor) if rgb else None,
        trace=sp_axis is not None,
    )
    tt_decoder.load_reference_weights()
    state = reference.init_state()
    # Chunks 3+ run with a full cache, which the decoder captures and then replays as a trace.
    for chunk in range(6):
        z = torch.randn(1, reference.chunk, reference.latents_mean.numel(), height, width)
        with torch.no_grad():
            expected = reference.decode_chunk(z, state)
        actual = tt_decoder.decode_chunk(z)
        assert actual.shape == expected.shape, f"chunk {chunk}: {tuple(actual.shape)} != {tuple(expected.shape)}"
        if rgb:
            # What the host path publishes from the float output: revert, clamp, `* 255`, truncate.
            assert actual.dtype == torch.uint8
            expected, actual = codec.processor.revert_tensor(expected.float()).mul(255).byte().float(), actual.float()
        assert_quality(expected, actual, pcc=MIN_PCC)
