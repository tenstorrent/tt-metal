# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full per-device model shape replay probe on captured real inputs."""

import json
import os
from pathlib import Path

import pytest
import torch
import ttnn

import device_attention as kernel


@pytest.fixture
def device_params():
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 64 * 1024**2}


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
@pytest.mark.timeout(600)
def test_replay_probe(mesh_device):
    torch.set_num_threads(8)
    source = Path(os.environ["FLUX2_CAPTURE_MANIFEST"])
    output = Path(os.environ["FLUX2_REPLAY_REPORT"])
    assert not output.exists()
    manifest = json.loads(source.read_text())
    rows = []
    for name in ("dual.0", "single.47"):
        values = torch.load(source.parent / manifest["real_input_captures"][name]["file"], weights_only=True)
        inputs = [values[key].repeat(1, 3, 1, 1).contiguous() for key in ("q", "k", "v")]
        inputs[0] = inputs[0][:, :, :2304].contiguous()
        assert [tuple(x.shape) for x in inputs] == [(1, 12, n, 128) for n in (2304, 4608, 4608)]
        tensors = [
            ttnn.from_torch(
                x, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
            )
            for x in inputs
        ]
        for variant in ("B", "A", "E", "F", "G", "C", "D"):

            def invoke():
                prepared = [
                    kernel.prepare(mesh_device, x, variant, is_q=i == 0, cores=108) for i, x in enumerate(tensors)
                ]
                return prepared, kernel.attention(mesh_device, *prepared, variant, max_cores=108)

            def read(value):
                shards = ttnn.get_device_tensors(value)
                if os.environ.get("FLUX2_REPLAY_ALL_DEVICES") == "1":
                    return torch.stack([ttnn.to_torch(x) for x in shards])
                return ttnn.to_torch(shards[0]).clone()

            prepared, result = invoke()
            reference = read(result)
            assert bool(torch.isfinite(reference).all())

            def record(mode, index, value):
                actual = read(value)
                delta = actual.float() - reference.float()
                row = dict(
                    block=name,
                    variant=variant,
                    mode=mode,
                    repeat=index,
                    exact=torch.equal(actual, reference),
                    l2_pct=100 * float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.float())),
                    max_abs=float(delta.abs().max()),
                    unequal=int(torch.count_nonzero(actual != reference)),
                )
                rows.append(row)
                print("REPLAY_PROBE", json.dumps(row), flush=True)
                output.write_text(json.dumps(dict(status="running", rows=rows), indent=2) + "\n")

            for index in range(3):
                # Interleave a different SFPU program and DRAM allocations.
                scratch = ttnn.exp(tensors[0])
                if os.environ.get("FLUX2_REPLAY_FP32_MATMUL") == "1":
                    scratch_mm = ttnn.matmul(
                        tensors[0], tensors[1], transpose_b=True,
                        compute_kernel_config=ttnn.init_device_compute_kernel_config(
                            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
                    )
                if os.environ.get("FLUX2_REPLAY_ALTERNATE") == "1":
                    saved_inputs = tensors
                    tensors = [ttnn.neg(x) for x in saved_inputs]
                    other_prepared, other_result = invoke()
                    tensors = saved_inputs
                if os.environ.get("FLUX2_REPLAY_RELOCATE") == "1":
                    tensors = [ttnn.clone(x) for x in tensors]
                prepared, result = invoke()
                record("untraced_after_exp", index, result)
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            prepared_trace, traced = invoke()
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            try:
                for index in range(5):
                    scratch = ttnn.exp(tensors[0])
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    record("traced_after_exp", index, traced)
            finally:
                ttnn.release_trace(mesh_device, trace)
    output.write_text(json.dumps(dict(status="completed", rows=rows), indent=2) + "\n")
    assert all(r["exact"] for r in rows), "Replay dependence found; inspect report"
