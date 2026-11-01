# Scratch (not for merge): MLA layer forward time for the vs-main matrix. Runs unchanged on main (mode=baseline, the
# model's ring_mla; copy this file into a main checkout) and on the branch (mode=expandedV / expandedKV). Galaxy 8x4 on
# the 2D torus (XY, or Y with SCRATCH_TORUS=y) at each model's fabric packet size. The chunk forward on a preloaded
# prefix runs once eagerly, is captured in a trace, replayed once to warm up, then timed (host wall clock) over 5
# replays 2 s apart; the result is their median.
import time

import pytest
from loguru import logger

import models.demos.deepseek_v3_d_p.tests.test_mla as T
import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
import os

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    moe_fabric_payload,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

MODES = (
    ["expandedV", "expandedKV"]
    if "mla_kwargs" in T._run_chunked_prefill.__code__.co_varnames
    else ["baseline", "baselineTuned"]
)


def _torus_device_params(**kw):
    """Full torus (TORUS_XY) by default; SCRATCH_TORUS=y for a Galaxy whose TP rings do not close."""
    return (torus_y_device_params if os.environ.get("SCRATCH_TORUS", "xy") == "y" else torus_xy_device_params)(**kw)


def _mesh(cfg, variant_id):
    return pytest.param(
        (8, 4),
        _torus_device_params(
            l1_small_size=1152, trace_region_size=64 * 1024 * 1024, fabric_payload_size=moe_fabric_payload(cfg)
        ),
        variant_id,
        id=f"torus-8x4-{variant_id}",
    )


@pytest.mark.parametrize(
    "mesh_device,device_params,variant",
    [_mesh(KimiK3Config, "kimi_k3"), _mesh(KimiK27Config, "kimi_k2_7")],
    indirect=["mesh_device", "device_params", "variant"],
)
@pytest.mark.parametrize("chunk", [5120, 2048], ids=["c5k", "c2k"])
@pytest.mark.parametrize("fidelity", ["HiFi2", "LoFi"], ids=["hifi2", "lofi"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("metadata", [True], ids=["meta"])
@pytest.mark.parametrize("prefill_len", [51200, 512000], ids=["kv50k", "kv500k"])
@pytest.mark.timeout(0)
def test_matrix_vs_main(
    request, mesh_device, device_params, variant, chunk, fidelity, mode, prefill_len, metadata, monkeypatch
):
    init = ttMLA.__init__

    def init_with_fidelity(self, *a, **kw):
        init(self, *a, **kw)
        self.sdpa_matmul_fidelity = getattr(ttnn.MathFidelity, fidelity)

    monkeypatch.setattr(ttMLA, "__init__", init_with_fidelity)
    if mode == "baselineTuned":
        # Main has no ring_mla SDPA config for 256 local rows (2k chunks) and falls back to q32 / k32; use the fastest
        # config of a q / k sweep at 512k (q32 / k512).
        if chunk != 2048:
            pytest.skip("baselineTuned only differs from baseline for 2k chunks")
        import models.demos.deepseek_v3_d_p.tt.mla.mla_config as C

        monkeypatch.setitem(
            C.MLA_SDPA_CONFIG, 256, [{"q_chunk_size": 32, "k_chunk_size": 512, "num_heads": None, "chunked_only": True}]
        )
    times = []

    def traced_forward_ns(mesh_device, run_fn):
        ttnn.deallocate(run_fn())
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = run_fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        replay_ns = []
        for _ in range(5):
            time.sleep(2)
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            replay_ns.append(time.perf_counter_ns() - start)
        ttnn.release_trace(mesh_device, tid)
        times.extend(replay_ns)
        return out, sorted(replay_ns)[2]

    monkeypatch.setattr(T, "_rt_profile_forward_ns", traced_forward_ns)
    kwargs = {} if mode.startswith("baseline") else dict(mla_kwargs=dict(expanded_k=mode == "expandedKV"))
    T._run_chunked_prefill(
        request,
        mesh_device,
        reference=None,
        topology=per_axis_topology(device_params["fabric_config"]),
        iters_isl=[chunk],
        chunk_size_global=chunk,
        prefill_len=prefill_len,
        use_metadata_tensor=metadata,
        profile=True,
        **kwargs,
    )
    t = sorted(times)[2]
    logger.info(
        f"RESULT {mode} {variant.name} chunk={chunk} {fidelity} kv={prefill_len} meta={metadata}: {t / 1e6:.3f} ms "
        f"(replays {[round(x / 1e6, 3) for x in times]})"
    )
