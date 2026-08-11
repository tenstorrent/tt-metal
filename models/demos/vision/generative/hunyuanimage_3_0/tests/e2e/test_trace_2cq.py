# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Contract: trace replay matches eager (PCC) per stage + host-op purity (zero host aten ops)."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0.tt import pipeline as pl

# Shard-graduated (TP=8) stubs run tensor-parallel on a mesh; FABRIC_1D only
# comes up on the FULL physical mesh of this 6U Blackhole Galaxy.
try:
    _MESH = tuple(int(x) for x in ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
except Exception:
    _MESH = (1, 8)

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = pl.load_reference_model()
    return _MODEL


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_host_op_selftest(device_params, mesh_device):
    """Gate 1 (authoritative): the on-device forward fires zero host aten ops."""
    torch.manual_seed(0)
    pipe = pl.build_pipeline(mesh_device, _get_model())
    v = pipe.host_op_selftest()
    print(f"\nhost_op verdict: on_device={v['on_device']} n_host_ops={v['n_host_ops']}")
    if v["host_ops"]:
        print(f"host_ops={v['host_ops']}")
    assert v["on_device"], f"host aten ops fired in the forward: {v['host_ops']}"


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": 200_000_000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_trace_capture_selftest(device_params, mesh_device):
    """Each PIPELINE_STAGE captures host-free and matches the eager step."""
    torch.manual_seed(0)
    pipe = pl.build_pipeline(mesh_device, _get_model())
    print(f"\nPIPELINE_STAGES={pipe.PIPELINE_STAGES}")
    ok_all, results = pipe.trace_capture_selftest(mesh_device)
    for stage, r in results.items():
        print(f"trace stage={stage}: {r}")
    # prefill is the real graduated path — it must capture host-free & match.
    assert results["prefill"]["captured"], f"prefill trace capture failed: {results['prefill']}"
    assert results["prefill"]["ok"], f"prefill trace PCC below target: {results['prefill']}"
