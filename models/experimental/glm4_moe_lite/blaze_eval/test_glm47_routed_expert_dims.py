# SPDX-License-Identifier: Apache-2.0
"""Run blaze's GLMRoutedExpert at GLM-4.7-Flash dims with a properly authored config.

test_glm5_routed_expert_blaze is parameterized on (embedding_dim, hidden_dim, num_experts,
eps, scaling_factor) but reads gate-MM cores and sender_core from GLM5_BLAZE_CONFIG. This
substitutes GLM4_FLASH_BLAZE_CONFIG instead -- a real authored config that passes
`sanity_check_model_config`, not a mutated GLM-5 one. BlazeConfig forbids `__replace__`
precisely because piecemeal patching yields inconsistent objects: truncating gate cores to 2
while the model_config still declared 256 experts tripped
`num_gate_mm_cores >= ceil(experts/32)`, and leaving 8 gate cores wired in for 64 experts
HANGS the device with no error (20+ min, against ~190 s for the GLM-5 shape).

  GLM-5.1        embedding 6144, moe intermediate 2048, 256 experts, scale 2.827, 8 gate cores
  GLM-4.7-Flash  embedding 2048, moe intermediate 1536,  64 experts, scale 1.8,   2 gate cores

Needs no plugin: the authored config already carries sender_core=(11, 9) and column-11 gate
cores for a 12x10 grid.

    pytest tests/blaze/glm5_1/test_glm47_routed_expert_dims.py -q
"""

import pytest

import blaze_tests.glm5_1.test_glm5_routed_expert as glm5_routed
from blaze.models.glm4_flash.glm4_flash_blaze_config import GLM4_FLASH_BLAZE_CONFIG

# From zai-org/GLM-4.7-Flash config.json: hidden_size, moe_intermediate_size,
# n_routed_experts, routed_scaling_factor. eps matches the GLM-5 test's router eps.
EMBEDDING_DIM = 2048
MOE_INTERMEDIATE_DIM = 1536
NUM_EXPERTS = 64
ROUTER_EPS = 1e-20
ROUTED_SCALING_FACTOR = 1.8


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_routed_expert_at_glm47_dims(mesh_device, monkeypatch):
    # Substitute the authored GLM-4.7-Flash config for the module-level GLM-5 one that the
    # test reads its gate-MM cores and sender core from.
    monkeypatch.setattr(glm5_routed, "GLM5_BLAZE_CONFIG", GLM4_FLASH_BLAZE_CONFIG)

    assert len(GLM4_FLASH_BLAZE_CONFIG.moe_router_gate_mm_cores) == NUM_EXPERTS // 32, (
        "gate-MM core count must equal num_experts // 32 -- a mismatch hangs the device " "rather than raising"
    )

    glm5_routed.test_glm5_routed_expert_blaze(
        mesh_device,
        0,  # device_id
        EMBEDDING_DIM,
        MOE_INTERMEDIATE_DIM,
        NUM_EXPERTS,
        ROUTER_EPS,
        ROUTED_SCALING_FACTOR,
    )
