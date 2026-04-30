# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os
from pathlib import Path

import pytest
from loguru import logger
from PIL import Image

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
    "mesh_device, cfg, sp, tp, encoder_tp, vae_tp, topology, num_links, mesh_test_id",
    [
        pytest.param(
            (2, 4), (2, 0), (1, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1, "2x4sp1tp4", id="2x4sp1tp4"
        ),
        pytest.param(
            (4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4, "4x8sp4tp4", id="4x8sp4tp4"
        ),
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
    mesh_test_id: str,
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

    input_path = os.environ.get("QWEN_EDIT_INPUT")
    input_image: Image.Image | None = None
    if input_path:
        input_image = Image.open(input_path).convert("RGB")
        logger.info(f"Loaded input image from {input_path}: size={input_image.size}")
        if input_image.size != (width, height):
            logger.info(f"Resizing input image from {input_image.size} to ({width}, {height})")
            input_image = input_image.resize((width, height), Image.LANCZOS)

    default_prompt = os.environ.get("QWEN_EDIT_PROMPT") or (
        "Change the sky to a dramatic sunset with orange and purple clouds. Keep the foreground elements unchanged."
    )
    outdir = Path(os.environ.get("QWEN_EDIT_OUTDIR", "."))
    outdir.mkdir(parents=True, exist_ok=True)

    def run_once(prompt: str, number: int, seed: int) -> None:
        result = pipeline.run_single_edit(
            prompt=prompt,
            image=input_image,
            negative_prompt="" if not no_prompt else None,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=4.0,
            seed=seed,
            traced=traced,
        )
        assert result and result[0] is not None, "pipeline returned no image"
        assert result[0].size[0] > 0 and result[0].size[1] > 0, f"Edited image has invalid size {result[0].size}"
        out_path = outdir / f"qwen_edit_{mesh_test_id}_{number}.png"
        result[0].save(out_path)
        logger.info(f"Saved {out_path} (size={result[0].size})")

    if no_prompt:
        run_once(default_prompt, 0, seed=42)
        return

    prompt = default_prompt
    for i in itertools.count():
        new_prompt = input("Enter the edit prompt (empty = reuse previous, q = quit): ")
        if new_prompt.strip().lower().startswith("q"):
            break
        if new_prompt:
            prompt = new_prompt
        run_once(prompt, i, seed=i)
