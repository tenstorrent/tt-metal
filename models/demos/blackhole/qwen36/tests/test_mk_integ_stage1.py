"""Megakernel integration stage1 (QWEN36_R16_MEGAKERNEL=stage1) host path with the identity stand-in engine
(engine/build.py, QWEN36_R16_MK_ENGINE_KIND=identity): x_in -> one generic_op -> persistent x_out per bucket, eagerly and
inside a captured trace with replays. Every device gets distinct x_in data, so a copy that read the wrong device's
buffer, or did not run, fails. No model weights needed.

  pytest models/demos/blackhole/qwen36/tests/test_mk_integ_stage1.py -s -x   (full-mesh conftest: QWEN36_TEST_FULL_MESH=1)
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.megakernel.engine.build import DIM, build_engine, rows_R

FA = [(i % 4) == 3 for i in range(64)]
T = 4


def test_rows_R():
    assert [rows_R(b) for b in (1, 2, 3, 4, 5, 8)] == [4, 8, 16, 16, 32, 32]


def _host(n_dev, rows, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_dev, 1, rows, DIM, generator=g).to(torch.bfloat16)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_mk_stage1_identity_traced(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_R16_MK_ENGINE_KIND", "identity")
    n_dev = mesh_device.get_num_devices()
    types = ["attention" if f else "gdn" for f in FA]
    buckets = [1, 2, 4, 8]
    progs = {b: build_engine(mesh_device, types, b, None, [], [], None, T=T) for b in buckets}
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    compose = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    def put(t, dst=None):
        h = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard)
        if dst is None:
            return ttnn.to_device(h, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy_host_to_device_tensor(h, dst)

    x_in = {b: put(_host(n_dev, b * T, b)) for b in buckets}
    x_out = {b: progs[b].alloc_x_out() for b in buckets}

    def check(tag, seed_of):
        for b in buckets:
            got = ttnn.to_torch(x_out[b], mesh_composer=compose)
            want = _host(n_dev, b * T, seed_of(b))
            bad = [d for d in range(n_dev) if not torch.equal(got[d], want[d])]
            assert not bad, f"{tag}: bucket {b} x_out != x_in on devices {bad}"

    for b in buckets:
        progs[b].launch(x_in[b], x_out[b], {})
    ttnn.synchronize_device(mesh_device)
    check("eager", lambda b: b)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for b in buckets:
        progs[b].launch(x_in[b], x_out[b], {})
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for it in range(3):
        for b in buckets:  # new input each replay, x_out zeroed: a stale value cannot pass
            put(_host(n_dev, b * T, 100 * (it + 1) + b), x_in[b])
            put(torch.zeros(n_dev, 1, b * T, DIM, dtype=torch.bfloat16), x_out[b])
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        check(f"replay {it}", lambda b: 100 * (it + 1) + b)
    ttnn.release_trace(mesh_device, tid)
    print(f"mk stage1 identity OK: {len(buckets)} buckets x {n_dev} devices, eager + 3 trace replays")
