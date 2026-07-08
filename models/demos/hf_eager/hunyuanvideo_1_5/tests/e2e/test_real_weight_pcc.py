# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight PCC: ttnn DiT vs the diffusers golden, both loaded with the REAL
720p_t2v checkpoint via the hyvideo->diffusers converter.

    HY_N=54 HY_BF16=1 pytest tests/e2e/test_real_weight_pcc.py -s

Results on one Blackhole (see real_weights/README.md):
  N=2 fp32 0.999998 · N=24 fp32 0.999913 · N=54 fp32 OOM · N=54 bf16 0.967557 (PASS)
"""
import os

import pytest

import ttnn
from models.demos.hf_eager.hunyuanvideo_1_5.real_weights.weights import build_real_transformer
from models.demos.hf_eager.hunyuanvideo_1_5.tt import pipeline as P


def coerce_bf16():
    """Run the DiT in bf16 (weights+activations) so full 54 layers fit one chip:
    map every ttnn.float32 request -> bfloat16 (stubs keep fp32 dest-accum)."""
    import ttnn

    _from, _tc = ttnn.from_torch, ttnn.typecast
    ttnn.from_torch = lambda t, *a, **k: _from(
        t, *a, **{**k, "dtype": ttnn.bfloat16} if k.get("dtype") == ttnn.float32 else k
    )
    ttnn.typecast = lambda t, dt, *a, **k: _tc(t, ttnn.bfloat16 if dt == ttnn.float32 else dt, *a, **k)


def test_real_weight_pcc(device):
    if os.environ.get("HY_BF16") == "1":
        coerce_bf16()
    N = int(os.environ.get("HY_N", "2"))
    print(f"\n[real-weight PCC] building real DiT at num_layers={N} ...", flush=True)
    model = build_real_transformer(num_layers=N)
    pipe = P.build_pipeline(device, model)
    inputs = P.build_inputs(model.config, task="t2v")
    golden = P.hf_reference(model, inputs)
    out = pipe.run(inputs, granularity="composite")
    val = P.pcc(golden, out)
    print(f"\n===== REAL-WEIGHT e2e PCC (N={N}, t2v) = {val:.6f} =====\n", flush=True)
    assert val >= 0.95, f"real-weight PCC {val:.6f} < 0.95"


@pytest.mark.timeout(1800)  # 4x54-layer build + upload across a fresh mesh is slower than pytest.ini's 300s default
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
def test_real_weight_pcc_mesh(mesh_device):
    """QB2 flat-TP=4 variant of :func:`test_real_weight_pcc` -- same real
    checkpoint, same golden, but the DiT runs sharded across all 4 mesh
    devices (see real_weights/README.md "RESUME ON QB2"). PCC should stay
    comparable to the single-chip baseline (bf16, N=54 -> 0.967557); sharding
    changes WHERE the math runs, not the math itself."""
    if os.environ.get("HY_BF16") == "1":
        coerce_bf16()
    N = int(os.environ.get("HY_N", "2"))
    print(
        f"\n[real-weight PCC mesh] building real DiT at num_layers={N}, tp={mesh_device.get_num_devices()} ...",
        flush=True,
    )
    model = build_real_transformer(num_layers=N)
    pipe = P.build_pipeline(mesh_device, model)
    inputs = P.build_inputs(model.config, task="t2v")
    golden = P.hf_reference(model, inputs)
    out = pipe.run(inputs, granularity="composite")
    val = P.pcc(golden, out)
    print(
        f"\n===== REAL-WEIGHT e2e PCC mesh (N={N}, tp={mesh_device.get_num_devices()}, t2v) = {val:.6f} =====\n",
        flush=True,
    )
    assert val >= 0.95, f"real-weight mesh PCC {val:.6f} < 0.95"
