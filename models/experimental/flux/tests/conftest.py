from __future__ import annotations

import pytest
import torch

from ..reference import FluxTransformer


@pytest.fixture(scope="module")
def parent_torch_model() -> FluxTransformer:
    model = FluxTransformer.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model
