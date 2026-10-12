# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The Qwen2.5-VL vision tower on device against the `transformers` reference.

Uses the vision weights of Qwen-Image's text encoder, which Qwen-Image-Edit shares, and random
patches on the grids the edit pipelines produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import huggingface_hub
import PIL.Image
import pytest
import safetensors
import torch
import transformers
import transformers.vision_utils
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

import ttnn
from models.tt_dit.encoders.qwen25vl.vision_qwen25vl import Qwen25VlVisionModel, vision_window_index
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

CHECKPOINT = "Qwen/Qwen-Image"
SUBFOLDER = "text_encoder"
PREFIX = "visual."
EDIT_CHECKPOINT = "Qwen/Qwen-Image-Edit"
PHOTO = Path(__file__).parents[4] / "experimental/retinanet/resources/dog_800x800.jpg"

# (t, h, w) in patches of 14 px.
GRIDS = {
    # Qwen-Image-Edit passes the input image resized to ~1024² (1036 x 1036 after the processor).
    "edit_1to1": [[1, 74, 74]],
    "edit_16to9": [[1, 54, 98]],
    # Qwen-Image-Edit-2509 and 2511 resize each condition image to ~384² (392 x 392).
    "plus_1x": [[1, 28, 28]],
    "plus_3x": [[1, 28, 28], [1, 24, 32], [1, 32, 24]],
}


@pytest.fixture(scope="module")
def hf_config() -> Qwen2_5_VLVisionConfig:
    return transformers.Qwen2_5_VLConfig.from_pretrained(CHECKPOINT, subfolder=SUBFOLDER).vision_config


@pytest.fixture(scope="module")
def state_dict() -> dict[str, torch.Tensor]:
    index = huggingface_hub.hf_hub_download(CHECKPOINT, f"{SUBFOLDER}/model.safetensors.index.json")
    weight_map = json.loads(Path(index).read_text())["weight_map"]
    shards = sorted({shard for key, shard in weight_map.items() if key.startswith(PREFIX)})

    state = {}
    for shard in shards:
        path = huggingface_hub.hf_hub_download(CHECKPOINT, f"{SUBFOLDER}/{shard}")
        with safetensors.safe_open(path, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith(PREFIX):
                    state[key.removeprefix(PREFIX)] = f.get_tensor(key)
    return state


@pytest.fixture(scope="module")
def reference(hf_config: Qwen2_5_VLVisionConfig, state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    model_class = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel
    model = model_class._from_config(hf_config, torch_dtype=torch.float32).eval()  # noqa: SLF001
    model.load_state_dict({k: v.float() for k, v in state_dict.items()})
    return model


@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS.keys())
def test_window_index(hf_config: Qwen2_5_VLVisionConfig, grid: list[list[int]]) -> None:
    grid_thw = torch.tensor(grid)
    expected_index, expected_bounds = transformers.vision_utils.get_vision_window_index(
        grid_thw,
        spatial_merge_size=hf_config.spatial_merge_size,
        window_size=hf_config.window_size,
        patch_size=hf_config.patch_size,
    )
    index, bounds = vision_window_index(
        grid_thw,
        spatial_merge_size=hf_config.spatial_merge_size,
        window_size=hf_config.window_size,
        patch_size=hf_config.patch_size,
    )
    assert torch.equal(index, expected_index)
    assert bounds == tuple(expected_bounds.tolist())


@pytest.mark.parametrize(
    ("mesh_device", "tp", "device_params"),
    [
        pytest.param((1, 1), 1, {}, id="1x1"),
        pytest.param((1, 4), 4, {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="1x4tp4"),
        pytest.param((1, 8), 8, {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="1x8tp8"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS.keys())
def test_tower(
    *,
    mesh_device: ttnn.MeshDevice,
    tp: int,
    grid: list[list[int]],
    hf_config: Qwen2_5_VLVisionConfig,
    state_dict: dict[str, torch.Tensor],
    reference: torch.nn.Module,
) -> None:
    """Random patches on the grids of the edit pipelines."""
    grid_thw = torch.tensor(grid)
    num_patches = int(grid_thw.prod(dim=-1).sum())
    patch_dim = hf_config.in_channels * hf_config.temporal_patch_size * hf_config.patch_size**2

    torch.manual_seed(0)
    pixel_values = torch.randn(num_patches, patch_dim)
    with torch.no_grad():
        expected = reference(pixel_values, grid_thw=grid_thw).pooler_output

    tower = _tower(hf_config, state_dict, mesh_device=mesh_device, tp=tp)
    actual = _run(tower, pixel_values, grid_thw, mesh_device=mesh_device)

    assert actual.shape == expected.shape
    assert_quality(expected, actual, pcc=0.999)


@pytest.mark.parametrize(
    ("mesh_device", "tp", "device_params"),
    [
        pytest.param((1, 1), 1, {}, id="1x1"),
        pytest.param((1, 4), 4, {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="1x4tp4"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("size", "pcc"), [(384, 0.997), (1024, 0.988)], ids=["384", "1024"])
def test_tower_photo(
    *,
    mesh_device: ttnn.MeshDevice,
    tp: int,
    size: int,
    pcc: float,
    hf_config: Qwen2_5_VLVisionConfig,
    state_dict: dict[str, torch.Tensor],
    reference: torch.nn.Module,
) -> None:
    """A photo, at the condition image size of Qwen-Image-Edit-2509 and the input size of Qwen-Image-Edit.

    Photos are harder than random patches: from block 17 on, a few tokens carry activations around
    10^4, and which tokens those are is sensitive to rounding. Images with large flat areas are
    worse still (~95 %), since many near-identical patches compete for those tokens.
    """
    processor = transformers.Qwen2VLProcessor.from_pretrained(EDIT_CHECKPOINT, subfolder="processor")
    inputs = processor.image_processor(images=[photo(size)], return_tensors="pt")
    with torch.no_grad():
        expected = reference(inputs.pixel_values, grid_thw=inputs.image_grid_thw).pooler_output

    tower = _tower(hf_config, state_dict, mesh_device=mesh_device, tp=tp)
    actual = _run(tower, inputs.pixel_values, inputs.image_grid_thw, mesh_device=mesh_device)

    assert actual.shape == expected.shape
    assert_quality(expected, actual, pcc=pcc)


def photo(size: int) -> PIL.Image.Image:
    return PIL.Image.open(PHOTO).convert("RGB").resize((size, size), PIL.Image.BICUBIC)


def _tower(
    hf_config: Qwen2_5_VLVisionConfig, state_dict: dict[str, torch.Tensor], *, mesh_device: ttnn.MeshDevice, tp: int
) -> Qwen25VlVisionModel:
    """The tower, loaded from `state_dict` without the weight cache."""
    parallel_config, ccl_manager = None, None
    if tp > 1:
        parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp, mesh_axis=1))
        ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    tower = Qwen25VlVisionModel.from_hf_config(
        hf_config,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tower.load_torch_state_dict(dict(state_dict))
    return tower


def _run(
    tower: Qwen25VlVisionModel, pixel_values: torch.Tensor, grid_thw: torch.Tensor, *, mesh_device: ttnn.MeshDevice
) -> torch.Tensor:
    inputs = tower.prepare_inputs(pixel_values, grid_thw)
    output = tower.forward(
        tensor.from_torch(inputs.patches, device=mesh_device),
        rope=(tensor.from_torch(inputs.cos, device=mesh_device), tensor.from_torch(inputs.sin, device=mesh_device)),
        reverse_index=tensor.from_torch(
            inputs.reverse_index.unsqueeze(0), device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        cu_seqlens=inputs.cu_seqlens,
        cu_window_seqlens=inputs.cu_window_seqlens,
    )
    return tensor.to_torch(output, mesh_axes=[None, None])
