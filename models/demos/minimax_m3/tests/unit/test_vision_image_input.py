# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side checks of M3VLVisionModel's input, which forward_tower checks with check_image_input
(tt/vision/image_input.py) before any device work, and of num_tokens, which check_image_input uses. No device. The
tests check that:
  - forward_tower, and forward through it, raise ValueError for input that is not one image, before any ttnn call;
  - a good image's pixel values go to the device as bf16 whatever the weights' dtype;
  - num_tokens counts the vision tokens of one image's grid in each form it takes, and rejects the grids that
    check_image_input stops before it calls num_tokens.

These tests have no mesh id; the 1-card CI leg selects them by name (its -k includes "test_vision_").

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_image_input.py
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.image_input import num_tokens
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel
from models.demos.minimax_m3.tt.vision.patch_embed import M3VLPatchEmbed

CONFIG = MiniMaxM3VisionConfig()  # the checkpoint's: 1176 values per patch
ROWS = torch.zeros(16, 1176)  # the patch rows of an image of 4 x 4 patches
GRID = torch.tensor([[1, 4, 4]])


def _rows_ending_in(*values):
    """ROWS with its last values replaced by values."""
    rows = ROWS.clone()
    rows.view(-1)[-len(values) :] = torch.tensor(values)
    return rows


def _device_work(*args, **kwargs):
    raise AssertionError("device work before the input check")


# id: (pixel_values, image_grid_thw, message). Every case is the image of ROWS and GRID or a mistake on it.
MODEL_BAD_INPUTS = {
    "rows_3d": (ROWS[..., None], GRID, r"\(num_patches, 1176\)"),
    "uint8_rows": (ROWS.to(torch.uint8), GRID, "float patch rows"),
    "row_width": (ROWS[:, :1000], GRID, r"\(num_patches, 1176\)"),
    "rows_numpy": (ROWS.numpy(), GRID, "torch tensors"),
    "grid_list": (ROWS, GRID.tolist(), "torch tensors"),
    "float_grid": (ROWS, GRID.float(), "integer tensor"),
    "bool_grid": (ROWS, GRID.bool(), "integer tensor"),
    "complex_grid": (ROWS, GRID.to(torch.complex64), "integer tensor"),
    "grid_1d": (ROWS, GRID[0], r"\(1, 3\)"),
    "two_images": (torch.cat([ROWS, ROWS]), torch.cat([GRID, GRID]), "grid of one image"),
    "video": (torch.cat([ROWS, ROWS]), torch.tensor([[2, 4, 4]]), "t = 1"),
    "odd_grid": (ROWS[:12], torch.tensor([[1, 3, 4]]), "multiples of 2"),
    "zero_grid": (ROWS[:0], torch.tensor([[1, 0, 4]]), "multiples of 2"),
    "more_rows": (torch.zeros(20, 1176), GRID, "20 patch rows .* 16 patches"),
    "fewer_rows": (ROWS[:12], GRID, "12 patch rows .* 16 patches"),
    "fewer_rows_4x6": (ROWS, torch.tensor([[1, 4, 6]]), "16 patch rows .* 24 patches"),
    "nan_pixel": (_rows_ending_in(float("nan")), GRID, "must be finite, got a sum of nan"),
    "inf_pixel": (_rows_ending_in(float("inf")), GRID, "must be finite, got a sum of inf"),
    "inf_pair": (_rows_ending_in(float("inf"), float("-inf")), GRID, "must be finite, got a sum of nan"),
}


@pytest.mark.parametrize("case", MODEL_BAD_INPUTS)
def test_vision_model_rejects_bad_input(case, expect_error, monkeypatch):
    """M3VLVisionModel.forward_tower, and forward through it, raise ValueError for input that is not one image before
    any ttnn call. No device: ttnn.from_torch and the modules are stand-ins that fail the test if they run.

    more_rows is the mismatch the device would not catch: the RoPE op gives zero q and k for the rows past the grid's
    patches. fewer_rows_4x6 has the rows of a 4 x 4 grid, which a token count of h * h would take for a 4 x 6 one.
    two_images would let the images attend to each other, and one NaN or Inf value would reach every patch: nan_pixel
    and inf_pixel have it in the last value only, and the two Inf values of inf_pair add up to NaN."""
    pixel_values, grid_thw, message = MODEL_BAD_INPUTS[case]
    monkeypatch.setattr(ttnn, "from_torch", _device_work)
    model = M3VLVisionModel(
        None,
        CONFIG,
        padded_head_dim=96,  # head_dim 80 in whole tiles; no case gets as far as the RoPE tables
        patch_embed=_device_work,
        pre_layrnorm=_device_work,
        blocks=[],
        projector=_device_work,
    )
    for run in (model.forward_tower, model.forward):
        with expect_error(ValueError, message):
            run(pixel_values, grid_thw)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32_rows", "fp16_rows"])
def test_vision_model_uploads_pixels_as_bf16(dtype, expect_error, monkeypatch):
    """A model built with bfloat8_b weights takes one good image, and its pixel values go to the device as bf16, the
    only dtype the RoPE op takes. The image's grid of 4 x 6 patches is not square, so a token count of h * h would
    reject it, and its 24 x 1176 values of 4.0 add up to more than fp16 holds, so the fp16 case also checks that the
    finite check does not overflow. No device: the module builders, the mesh mapper and ttnn.from_torch are stand-ins,
    and from_torch records what it gets and stops the forward."""
    weight_dtypes, uploads = [], []

    def build(mesh_device, reference_module, dtype):
        weight_dtypes.append(dtype)
        return _device_work

    def from_torch(tensor, **kwargs):
        uploads.append((tensor.dtype, tuple(tensor.shape), kwargs["dtype"]))
        raise RuntimeError("stop after the pixel upload")

    monkeypatch.setattr(M3VLPatchEmbed, "from_torch", build)
    monkeypatch.setattr(M3VLLayerNorm, "from_torch", build)
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh_device: None)
    reference = SimpleNamespace(config=CONFIG, patch_embed=None, pre_layrnorm=None, layers=[])
    model = M3VLVisionModel.from_torch(None, reference, with_projector=False, dtype=ttnn.bfloat8_b)
    with expect_error(RuntimeError, "stop after the pixel upload"):
        model.forward_tower(torch.full((24, 1176), 4.0, dtype=dtype), torch.tensor([[1, 4, 6]]))
    assert weight_dtypes == [ttnn.bfloat8_b, ttnn.bfloat8_b]
    assert uploads == [(torch.bfloat16, (1, 1, 24, 1176), ttnn.bfloat16)]


# id: (grid_thw, vision tokens): the grid of one image in each form num_tokens takes. h and w differ, so a count that
# used one of them twice would be wrong.
NUM_TOKENS_GRIDS = {
    "tensor_1x3": (torch.tensor([[1, 4, 6]]), 6),
    "tensor_3_int32": (torch.tensor([1, 34, 46], dtype=torch.int32), 391),
    "list": ([1, 46, 34], 391),
}

# id: (grid_thw, error, message). check_image_input stops these grids before it calls num_tokens.
NUM_TOKENS_BAD_GRIDS = {
    "float_grid": ([1.0, 4.0, 4.0], TypeError, "integers"),
    "bool_grid": (GRID.bool(), TypeError, "integers"),
    "complex_grid": (GRID.to(torch.complex64), TypeError, "integers"),
    "two_images": (torch.cat([GRID, GRID]), ValueError, "grid of one image"),
}


@pytest.mark.parametrize("case", NUM_TOKENS_GRIDS)
def test_vision_num_tokens(case):
    """num_tokens gives h * w / 4 vision tokens for the grid (1, h, w) of one image."""
    grid_thw, tokens = NUM_TOKENS_GRIDS[case]
    assert num_tokens(grid_thw) == tokens


@pytest.mark.parametrize("case", NUM_TOKENS_BAD_GRIDS)
def test_vision_num_tokens_rejects_bad_grid(case, expect_error):
    """num_tokens raises TypeError for a grid that does not hold integers and ValueError for the grids of two images."""
    grid_thw, error, message = NUM_TOKENS_BAD_GRIDS[case]
    with expect_error(error, message):
        num_tokens(grid_thw)
