"""Megakernel integration stage0 (QWEN36_R16_MEGAKERNEL=stage0): per-bucket phase tables + the probe generic_op,
eagerly and inside a captured trace with replays. No model weights needed.

  pytest models/demos/blackhole/qwen36/tests/test_mk_integ_stage0.py -s -x   (full-mesh conftest: QWEN36_TEST_FULL_MESH=1)

The model-level check is the step harness with QWEN36_R16_MEGAKERNEL=stage0: the probe runs inside every verify
bucket's trace and the outputs must match the flag-off md5.
"""

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.megakernel.integ.phase_table import build_phase_table, checksum, summary
from models.demos.blackhole.qwen36.tt.megakernel.integ.verify_mk import MegakernelVerify

# Qwen3.8-27B: 64 layers, full attention every 4th (layers 3, 7, ...)
FA = [(i % 4) == 3 for i in range(64)]


def test_phase_table_host():
    """Host-only: layer pattern → phase counts and checksum stability."""
    t8, t1 = build_phase_table(FA, 8), build_phase_table(FA, 1)
    s8, s1 = summary(t8), summary(t1)
    assert s8["SDPA"] == 16 and s8["GDN_PREFIX"] == 48 and s8["GDN_STEP"] == 48 and "GDN_STEP_U1" not in s8
    assert s1["GDN_STEP_U1"] == 48 and "GDN_PREFIX" not in s1
    assert s8["AR_SEND"] == s8["AR_RECV"] == 128  # 2 ARs per layer × 64
    assert checksum(t8) != checksum(t1)
    assert checksum(t8) == checksum(build_phase_table(FA, 8))


class _FakeLayer:
    def __init__(self, fa):
        self.is_full_attention = fa


class _FakeModel:
    def __init__(self, mesh):
        self.mesh_device = self.device = mesh
        self.layers = [_FakeLayer(f) for f in FA]


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_mk_stage0_probe_traced(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_R16_MEGAKERNEL", "stage0")
    mk = MegakernelVerify(_FakeModel(mesh_device))
    buckets = [1, 2, 4, 8]
    mk.prepare(buckets)

    # eager
    for b in buckets:
        mk.pre_layers(b)
    ttnn.synchronize_device(mesh_device)
    for b in buckets:
        want, n_dev, bad = mk.check_probe(b)
        assert not bad, f"eager probe mismatch bucket {b}: devices {bad} (want {want})"

    # inside a captured trace, replayed 3×: clear the outputs between replays so a stale value can't pass
    import torch

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for b in buckets:
        mk.pre_layers(b)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    zeros = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
    for it in range(3):
        for b in buckets:
            host = ttnn.from_torch(zeros, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
            ttnn.copy_host_to_device_tensor(host, mk.probe_out[b])
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        for b in buckets:
            want, n_dev, bad = mk.check_probe(b)
            assert not bad, f"traced probe mismatch replay {it} bucket {b}: devices {bad} (want {want})"
    ttnn.release_trace(mesh_device, tid)
    print(f"mk stage0 OK: {len(buckets)} buckets × {n_dev} devices, eager + 3 trace replays")
