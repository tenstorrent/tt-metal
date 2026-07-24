# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Shared pytest fixtures for the HunyuanImage-3.0 PCC tests.

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unit_host: host-only unit tests (mock logits); excluded from on-device PCC sweeps",
    )
    config.addinivalue_line(
        "markers",
        "e2e_random_inputs: integration test with random latent/text embeds; opt-in via HY_RUN_E2E_RANDOM=1",
    )


def pytest_collection_modifyitems(items):
    """Production slow tests (32L load) exceed the global 300s pytest.ini timeout."""
    for item in items:
        if "slow" in item.keywords and not any(m.name == "timeout" for m in item.iter_markers()):
            item.add_marker(pytest.mark.timeout(10800))


@pytest.fixture(scope="function")
def device():
    """Function-scoped device — safe for single-device and mesh tests."""
    import ttnn

    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# SigLIP2 PCC fixtures (moved from tests/vision/conftest.py)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_dir():
    from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR

    index = MODEL_DIR / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip(f"Hunyuan checkpoint not found at {MODEL_DIR} (set HUNYUAN_MODEL_DIR)")
    return MODEL_DIR


@pytest.fixture(scope="module")
def vision_inputs():
    from models.experimental.hunyuan_image_3_0.tests.pcc.siglip2_helpers import make_smoke_vision_inputs

    return make_smoke_vision_inputs()


@pytest.fixture
def tt_vision_inputs(device, vision_inputs):
    from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import Siglip2VisionInputs
    from models.experimental.hunyuan_image_3_0.tests.pcc.siglip2_helpers import (
        spatial_shapes_to_hw,
        upload_attention_mask,
        upload_pixel_values,
    )

    pixel_values, spatial_shapes, pixel_attention_mask = vision_inputs
    return Siglip2VisionInputs.create(
        upload_pixel_values(device, pixel_values),
        spatial_shapes_to_hw(spatial_shapes),
        upload_attention_mask(device, pixel_attention_mask),
    )


@pytest.fixture(scope="module")
def ref_vision(model_dir):
    from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import load_siglip2_vision
    from models.experimental.hunyuan_image_3_0.tests.pcc.siglip2_helpers import NUM_LAYERS

    return load_siglip2_vision(model_dir, num_layers=NUM_LAYERS)


@pytest.fixture(scope="module")
def ref_aligner(model_dir):
    from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import load_aligner

    return load_aligner(model_dir)


@pytest.fixture(scope="module")
def vision_state_dict(ref_vision):
    return ref_vision.state_dict()


@pytest.fixture(scope="module")
def aligner_state_dict(ref_aligner):
    return ref_aligner.state_dict()
