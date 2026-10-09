# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vision tower runtime contract: no host fallback in the forward, deterministic output.

- ``test_forward_stays_on_device``: after setup (weights loaded, image prepared and uploaded, one
  warm-up pass), one whole tower forward runs inside ``tests/runtime_audit.count_host_calls``; the
  counter (``ttnn.from_torch`` / ``to_torch`` / host copies / any torch op) must stay empty. v01 is an
  exact bucket (256, no slice); v02 is 936 patches in the 1024 bucket (mask + output slice).
- ``test_deterministic``: the same image twice gives bit-identical features.

Also the watcher target (doc/vision/work_log.md has the command)::

    TT_METAL_WATCHER=10 pytest .../test_vision_runtime.py -k "test_forward_stays_on_device"
"""

import pytest
import torch

import ttnn
from models.demos.pplx_decider_v1_27b.tests.runtime_audit import count_host_calls
from models.demos.pplx_decider_v1_27b.tests.vision.vision_test_utils import DEVICE_PARAMS, build_tower, prepare

pytestmark = pytest.mark.use_module_device(DEVICE_PARAMS)
AUDIT_IMAGES = ("v01_dominant_color", "v02_count_circles")


@pytest.fixture(scope="module")
def tower(_device_module_impl):
    return build_tower(_device_module_impl)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("image", AUDIT_IMAGES, ids=["v01_256_exact", "v02_936_in_1024"])
def test_forward_stays_on_device(tower, image):
    device = tower.mesh_device
    inputs = prepare(tower, image)
    ttnn.deallocate(tower(inputs))  # warm-up: program compile happens here
    ttnn.synchronize_device(device)

    with count_host_calls() as counts:
        out = tower(inputs)
        ttnn.synchronize_device(device)
    assert ttnn.is_tensor_storage_on_device(out)
    assert tuple(out.shape) == (1, 1, inputs.num_tokens, 5120)
    assert not counts, f"Host calls inside the measured tower forward: {dict(counts)}"

    # Positive control: the counter does see host conversions and torch ops.
    with count_host_calls() as control:
        torch.ones(2).sum()
        ttnn.to_torch(out)
    assert control["ttnn.to_torch"] == 1 and any(k.startswith("torch_function:") for k in control), dict(control)
    inputs.deallocate()


@pytest.mark.timeout(900)
@pytest.mark.parametrize("image", AUDIT_IMAGES, ids=["v01_256_exact", "v02_936_in_1024"])
def test_deterministic(tower, image):
    inputs = prepare(tower, image)
    first = ttnn.to_torch(tower(inputs))
    second = ttnn.to_torch(tower(inputs))
    inputs.deallocate()
    # A fresh upload of the same image as well (no stale state carried between requests).
    inputs = prepare(tower, image)
    third = ttnn.to_torch(tower(inputs))
    inputs.deallocate()
    assert torch.equal(first, second) and torch.equal(first, third), (
        f"max diff {float((first.float() - second.float()).abs().max())}, "
        f"{float((first.float() - third.float()).abs().max())}"
    )
