# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Session fixtures for the e2e gates: ONE mesh, ONE reference, ONE pipeline.

Why session scope
-----------------
Building the pipeline stages 9.08 B parameters onto the 1x8 mesh, and the
reference is 9.08 B float32 parameters on host. The three gate files
(`test_e2e_denoise_step.py`, `test_e2e_denoise_latents.py`,
`test_trace_contract.py`) all want the same objects, and the gate runs them in a
single `pytest` invocation, so they are built once here and shared. A
function-scoped device would rebuild everything per test and pay that cost three
times over for no extra coverage.

This fixture is the SOLE device opener for the e2e gates -- `tt/pipeline.py`
never opens one, it runs on the `device` handed to `build_pipeline`. The mesh is
opened with `num_command_queues=1` and a real `trace_region_size`, which is what
makes `ttnn.begin_trace_capture` usable in `test_trace_contract.py`.

Shapes
------
`e2e_plan.json::shapes.gate_test`: 256x256 -> 16x16 latent grid -> S_img=256,
S_txt=64, S_joint=320, 4 Euler steps. Depth defaults to the checkpoint's full
8 dual + 24 single blocks; `TT_FLUX2_E2E_{LAYERS,DUAL_LAYERS,SINGLE_LAYERS}` cap
it for a fast wiring loop, and every test reports the depth it actually ran at.
"""

from __future__ import annotations

import os

import pytest

import ttnn
from models.demos.flux_2_klein_9b.transformer.tt import inputs as tt_inputs
from models.demos.flux_2_klein_9b.transformer.tt import pipeline as tt_pipeline
from models.demos.flux_2_klein_9b.transformer.tt import reference as tt_reference

# e2e_plan.json::device -- T3K, mesh 1x8, TP=8, FABRIC_1D, l1_small_size 24576.
TP = int(os.environ.get("TT_FLUX2_E2E_TP", "8"))
L1_SMALL_SIZE = 24576

# The traced unit here is a whole 32-block forward over the joint sequence, which
# is a lot of dispatch commands; sized generously and overridable, because a
# too-small region shows up as a capture failure rather than a wrong number.
TRACE_REGION_SIZE = int(os.environ.get("TT_FLUX2_E2E_TRACE_REGION_SIZE", str(768 * 1024 * 1024)))

# e2e_plan.json::shapes.gate_test
GATE_HEIGHT = int(os.environ.get("TT_FLUX2_E2E_HEIGHT", "256"))
GATE_WIDTH = int(os.environ.get("TT_FLUX2_E2E_WIDTH", "256"))
GATE_TXT_LEN = int(os.environ.get("TT_FLUX2_E2E_TXT_LEN", "64"))
GATE_STEPS = int(os.environ.get("TT_FLUX2_E2E_STEPS", "4"))
GATE_SEED = int(os.environ.get("TT_FLUX2_E2E_SEED", "0"))

# e2e_plan.json::gates.gate_3 -- the tool-enforced threshold is 0.95; the
# per-component PCCs are 0.99991 .. 1.0 at TP=8, so anything much below 0.99 here
# would mean a wiring bug rather than accumulated bfloat16 rounding.
PCC_TARGET = float(os.environ.get("TT_FLUX2_E2E_PCC", "0.95"))


@pytest.fixture(scope="session")
def flux2_shapes():
    return {
        "height": GATE_HEIGHT,
        "width": GATE_WIDTH,
        "txt_len": GATE_TXT_LEN,
        "steps": GATE_STEPS,
        "seed": GATE_SEED,
        "pcc": PCC_TARGET,
    }


@pytest.fixture(scope="session")
def flux2_device():
    """The 1x8 mesh, opened once, with fabric and a trace region."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, TP),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=TRACE_REGION_SIZE,
        num_command_queues=1,
    )
    yield device
    ttnn.close_mesh_device(device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="session")
def flux2_reference():
    """The HF golden module (real shipped weights, float32, eval mode)."""
    return tt_reference.load_reference_model()


@pytest.fixture(scope="session")
def flux2_pipeline(flux2_device, flux2_reference, flux2_shapes):
    """The resident TT pipeline, built once at the depth the env asks for."""
    return tt_pipeline.build_pipeline(
        flux2_device,
        model=flux2_reference,
        height=flux2_shapes["height"],
        width=flux2_shapes["width"],
        txt_len=flux2_shapes["txt_len"],
        **tt_pipeline.depth_from_env(),
    )


@pytest.fixture(scope="session")
def flux2_inputs(flux2_shapes):
    """The seeded, Source-A-recipe input set BOTH sides of the PCC receive."""
    return tt_inputs.build_inputs(
        height=flux2_shapes["height"],
        width=flux2_shapes["width"],
        txt_len=flux2_shapes["txt_len"],
        batch=1,
        seed=flux2_shapes["seed"],
    )
