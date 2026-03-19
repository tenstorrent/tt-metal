# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from loguru import logger

import ttnn

from ....pipelines.qwenimageedit.pipeline_qwenimageedit import QwenImageEditPipeline


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 47000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 50)])
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, encoder_tp, vae_tp, topology, num_links",
    [
        [(2, 4), (2, 0), (1, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=[
        "2x4sp1tp4",
        "4x8sp4tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "use_torch_text_encoder",
    [
        pytest.param(False, id="encoder_device"),
    ],
)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(True, id="traced"),
    ],
)
def test_qwenimageedit_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    num_inference_steps: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    encoder_tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    no_prompt: bool,
    use_torch_text_encoder: bool,
    traced: bool,
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    pipeline = QwenImageEditPipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        use_torch_text_encoder=use_torch_text_encoder,
        use_torch_vae_decoder=False,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
    )

    edit_prompt = (
        "Change the sky to a dramatic sunset with orange and purple clouds. " "Keep the foreground elements unchanged."
    )

    _, cfg_axis = cfg
    _, sp_axis = sp
    _, tp_axis = tp

    result = pipeline.run_single_edit(
        prompt=edit_prompt,
        image=None,
        negative_prompt="" if not no_prompt else None,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=4.0,
        seed=42,
        traced=traced,
    )

    logger.info(f"Generated {len(result)} edited image(s)")

    for idx, img in enumerate(result):
        assert img is not None, f"Edited image {idx} is None"
        assert img.size[0] > 0 and img.size[1] > 0, f"Edited image {idx} has invalid dimensions"
        logger.info(f"Edited image {idx}: size={img.size}")
