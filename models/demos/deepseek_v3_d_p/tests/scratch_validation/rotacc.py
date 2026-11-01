# Scratch (not for commit): expanded-K MLA chunked prefill vs the CPU reference on rotated / partial / unaligned-prefix
# scenarios, on the 2D torus (XY by default, Y with SCRATCH_TORUS=y).
import pytest

import models.demos.deepseek_v3_d_p.tests.test_mla as T
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
import os

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    moe_fabric_payload,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

SCENARIOS = [(f"rot-{i}", dict(iters_isl=l)) for i, l in zip(T.ROTATED_VALID_IDS, T.ROTATED_VALID_LISTS)] + [
    ("maxedge-1u", dict(iters_isl=[2560, 2592, 5120])),
    ("padoverflow-1u", dict(iters_isl=[5120 - 32] * 2 + [32], tight_cache=True)),
    ("deep-unaligned", dict(iters_isl=[5120], prefill_len=51200 + 2048)),
]


def _torus_device_params(**kw):
    """Full torus (TORUS_XY) by default; SCRATCH_TORUS=y for a Galaxy whose TP rings do not close."""
    return (torus_y_device_params if os.environ.get("SCRATCH_TORUS", "xy") == "y" else torus_xy_device_params)(**kw)


def _mesh(cfg, variant_id):
    return pytest.param(
        (8, 4),
        _torus_device_params(l1_small_size=1152, fabric_payload_size=moe_fabric_payload(cfg)),
        variant_id,
        id=f"torus-8x4-{variant_id}",
    )


@pytest.mark.parametrize(
    "mesh_device,device_params,variant",
    [_mesh(KimiK3Config, "kimi_k3"), _mesh(KimiK27Config, "kimi_k2_7")],
    indirect=["mesh_device", "device_params", "variant"],
)
@pytest.mark.parametrize("kwargs", [kw for _, kw in SCENARIOS], ids=[sid for sid, _ in SCENARIOS])
@pytest.mark.timeout(0)
def test_rotacc(request, mesh_device, device_params, variant, kwargs):
    T._run_chunked_prefill(
        request,
        mesh_device,
        reference="cpu",
        topology=per_axis_topology(device_params["fabric_config"]),
        use_metadata_tensor=True,
        **kwargs,
    )
