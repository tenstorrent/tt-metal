# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare all recipes on identical captured real-model BF16 attention inputs."""

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch
import ttnn

import device_attention as kernel


@pytest.fixture
def device_params():
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
@pytest.mark.timeout(600)
def test_captured_inputs(mesh_device):
    source = Path(os.environ["FLUX2_CAPTURE_MANIFEST"])
    output = Path(os.environ["FLUX2_CAPTURE_REPORT"])
    if output.exists():
        raise ValueError("Use a fresh report path")
    manifest = json.loads(source.read_text())
    assert manifest["variant"] == "D" and manifest["status"] == "completed"
    torch.set_num_threads(8)
    rows = []
    for name, record in manifest["real_input_captures"].items():
        path = source.parent / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        values = torch.load(path, weights_only=True)
        # Include both image and text queries, using all recorded KV tokens.
        q = torch.cat((values["q"][:, :, :256], values["q"][:, :, -256:]), dim=2)
        original = [q, values["k"], values["v"]]
        for value in original:
            assert bool(torch.isfinite(value).all())
        for value in original[1:]:
            kernel.B4.validate_input(value)
        q64, k64, v64 = [x.double() for x in original]
        reference = (q64 @ k64.transpose(-2, -1) / math.sqrt(128)).softmax(-1) @ v64
        tensors = [ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT,
                                  mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) for x in original]
        for variant in kernel.VARIANTS:
            prepared = [kernel.prepare(mesh_device, x, variant, is_q=i == 0, cores=8)
                        for i, x in enumerate(tensors)]
            result = kernel.attention(mesh_device, *prepared, variant, max_cores=8)
            actual = ttnn.to_torch(ttnn.get_device_tensors(result)[0]).double()
            assert bool(torch.isfinite(actual).all()), (name, variant)
            delta = actual - reference
            rms = reference.square().mean().sqrt()
            row = dict(block=name, variant=variant,
                       l2_pct=100 * float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)),
                       pcc=float(torch.corrcoef(torch.stack([actual.flatten(), reference.flatten()]))[0, 1]),
                       max_abs=float(delta.abs().max()),
                       max_relative_rms_floor_1e_minus3=float((delta.abs() / reference.abs().clamp_min(rms * 1e-3)).max()),
                       query_rows="first 256 and last 256 recorded tokens", input_sha256=record["sha256"])
            rows.append(row)
            print("REAL_INPUT_QUALIFICATION", json.dumps(row), flush=True)
            output.write_text(json.dumps(dict(source=str(source), status="running", rows=rows), indent=2) + "\n")
    assert len(rows) == 28
    output.write_text(json.dumps(dict(source=str(source), status="completed", rows=rows), indent=2) + "\n")
