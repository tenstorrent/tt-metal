# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_quasar


@pytest.fixture(autouse=True)
def quasar_host_cast_fp32_to_bf16(monkeypatch):
    """On Quasar, cast float32 torch tensors to bfloat16 on the host before ttnn.from_torch uploads them.

    When the source is float32 and the target dtype is bfloat16, from_torch converts on the device: it
    tilizes in float32 and then typecasts. The float32 tilize faults on Quasar with UNPACKER_0
    ILLEGAL_FORMAT_CONVERSION (issue #57780). Casting on the host makes the source dtype match the
    target dtype, so the upload runs no device conversion. ttnn.as_tensor calls ttnn.from_torch, so
    weight uploads are covered too.
    """
    if not is_quasar():
        return

    orig_from_torch = ttnn.from_torch

    def _from_torch(tensor, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dtype == torch.float32
            and dtype == ttnn.bfloat16
            and kwargs.get("device") is not None
        ):
            tensor = tensor.to(torch.bfloat16)
        return orig_from_torch(tensor, *args, **kwargs)

    monkeypatch.setattr(ttnn, "from_torch", _from_torch)


def pytest_addoption(parser):
    parser.addoption("--skip-model-load", action="store_true", default=False, help="Skip loading the model state dict")


@pytest.fixture(scope="session")
def state_dict(request):
    load_model = not request.config.getoption("--skip-model-load")
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None or not load_model:
        # Explicit skip: build weights purely from the ttnn cache.
        return {}
    # Defer the (expensive) HF weight load to create_tt_model, which knows the mesh shape +
    # dtype and can skip it when a warm ttnn cache is already on disk (see
    # ModelArgs.weight_cache_is_complete). Returning None signals "load if needed".
    return None


@pytest.fixture
def test_thresholds(request):
    with open("models/experimental/ops/quasar/gpt_oss/unit_test_thresholds.json", "r") as f:
        thresholds = json.load(f)
    return thresholds
