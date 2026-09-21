# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare the six recipes on identical real Wan inputs and an FP64 reference."""

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch
import ttnn

from attention import kernel


@pytest.fixture
def device_params():
    return dict(fabric_config=ttnn.FabricConfig.FABRIC_1D, l1_small_size=65536)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.timeout(900)
def test_real_qkv(mesh_device):
    torch.set_num_threads(8)
    source = Path(os.environ["WAN_CAPTURE_MANIFEST"])
    target = Path(os.environ["WAN_CAPTURE_REPORT"])
    assert not target.exists()
    manifest = json.loads(source.read_text())
    assert manifest["variant"] == "D" and manifest["status"] == "completed"
    rows = []
    for name, record in manifest["captures"].items():
        path = source.parent / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        values = torch.load(path, weights_only=True)
        originals = [values[key] for key in ("q", "k", "v")]
        n = values["logical_n"]
        q, k, v = [x.double() for x in originals]
        reference = (q @ k[:, :, :n].transpose(-1, -2) / math.sqrt(128)).softmax(-1) @ v[:, :, :n]
        tensors = [
            ttnn.from_torch(
                x, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
            )
            for x in originals
        ]
        for variant in "DCBEFG":
            prepared = [kernel.prepare(mesh_device, x, variant, is_q=i == 0, cores=8) for i, x in enumerate(tensors)]
            out = kernel.attention(mesh_device, *prepared, variant, logical_k=n, max_cores=8)
            actual = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).double()
            assert torch.isfinite(actual).all()
            delta = actual - reference
            rms = reference.square().mean().sqrt()
            row = dict(
                block=name,
                variant=variant,
                l2_pct=100 * float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)),
                pcc=float(torch.corrcoef(torch.stack([actual.flatten(), reference.flatten()]))[0, 1]),
                max_abs=float(delta.abs().max()),
                max_relative_rms_floor_1e_minus3=float((delta.abs() / reference.abs().clamp_min(rms * 1e-3)).max()),
            )
            rows.append(row)
            print("WAN_REAL_QKV", json.dumps(row), flush=True)
            target.write_text(json.dumps(dict(status="running", source=str(source), rows=rows), indent=2) + "\n")
            for tensor in prepared:
                if all(tensor is not original for original in tensors):
                    ttnn.deallocate(tensor)
            ttnn.deallocate(out)
        for tensor in tensors:
            ttnn.deallocate(tensor)
    assert len(rows) == 24
    target.write_text(json.dumps(dict(status="completed", source=str(source), rows=rows), indent=2) + "\n")
