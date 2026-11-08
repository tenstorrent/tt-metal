# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse MLA (DSA) chunked-prefill perf: host-observed time, untraced and traced. No device-side profiling:
sparse MLA overlaps the indexer top-k and the KV gather on separate sub-devices, which per-program device times
don't capture.

Scenarios (Galaxy sizes; smaller boxes scale the chunk and cache by SP/8 so each chip does Galaxy's work):
  * warm: one 5k chunk over a 50k cache (zero-init caches; contents don't change op shapes or timing).
  * cold: every chunk from an empty cache up to the warm step, with real forwards that fill the caches.
  * long: one 5k chunk over a 500k cache.

test_sparse_mla_perf: the CI gate (Galaxy, LoudBox). Warm and long with the production scaled-FP8 KVPE cache
(SPxTP-sharded), torus XY on Galaxy and Fabric2D on LoudBox, for GATE_CASES: K3.1 (last DSA layer's slot of 24, at
K3.1's packet size), GLM-5.3 (last full indexer layer) and a GLM-5.3 shared layer (no indexer; reuses that layer's
top-k). Untraced: fastest of UNTRACED_SAMPLES calls after UNTRACED_WARMUP discarded ones. Traced: one timed replay of
the captured trace. Each (case, box, scenario, mode) is gated two-sided against PERF_TARGETS_MS ± PERF_MARGIN[mode] (a
case without targets only reports), and the traced output must match the untraced one (one extra untimed run of each,
for triage, plus one untimed eager forward with a per-op host profile). Each layer runs standalone (no next-layer KV
prefetch overlapped with its attention), as the first layer of a stage.

test_mla_chunked_perf: the exploration matrix (QuietBox, LoudBox, Galaxy; skipped on CI), reports only.
GLM-5.3 (last full indexer layer, 21-slot index cache) and K3.1 x warm/cold/long x sparse with a BF16 or
scaled-FP8 cache / dense ring-MLA baseline. Narrow with -k, e.g. -k "glm_5_3 and cold and kv_scaled_fp8".
"""

import copy
import gc
import os
import resource
import time
from dataclasses import dataclass

import pytest

import ttnn
from models.common.utility_functions import comp_pcc, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    moe_fabric_payload,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import make_hidden
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from tt_py_test_utils_common.perf.cpu_clock import current_cpu, hold_full_cpu_clock

GALAXY_SP = 8
# Galaxy-global scenario sizes: a 5k chunk over a 50k (warm) or 500k (long) cache.
CHUNK_TOKENS = 5120
WARM_CACHE_TOKENS = 51200
LONG_CACHE_TOKENS = 512000
# Untraced: discarded calls (the first compiles), then timed calls, each through sync; the gate takes the fastest.
UNTRACED_WARMUP = 10
UNTRACED_SAMPLES = 10
SCENARIO_CACHE_TOKENS = {"warm": WARM_CACHE_TOKENS, "long": LONG_CACHE_TOKENS}
MODES = ("untraced", "traced")
# Gate cases: (variant, shared_layer). GLM-5.3 also gates a shared layer, which reuses a full layer's top-k and runs
# no indexer.
GATE_CASES = {
    "kimi_k3_1": ("kimi_k3_1", False),
    "glm_5_3": ("glm_5_3", False),
    "glm_5_3-no_indexer": ("glm_5_3", True),
}
# Host time of one chunk forward through device synchronization, in ms, per case: (warm untraced, warm traced, long
# untraced, long traced). From CI (LOGURU_LEVEL=ERROR, TT_METAL_SHM_TRACKING_DISABLED=1, one pytest process per model as
# the jobs run them), 2-3 runs per case on 2026-11-07. Traced is their mean. Untraced is the fastest of the runs'
# per-run minimums: some runs landed in a slow mode that adds 1-2 ms to every sample of a test (the dispatching core left
# at a low CPU frequency; hold_full_cpu_clock() requests full clock under schedutil). The GLM-5.3 shared layer has no
# cache-length-dependent work, and its warm case has only been seen in the slow mode on Galaxy, so it takes the long
# case's untraced time. LoudBox GLM-5.3 untraced is from the first clamped CI run (warm 2.854, long 3.904 ms) on 2026-11-08:
# the earlier targets (4.54, 4.88) came from unclamped slow-mode runs. LoudBox K3.1 untraced is from two clamped CI runs (warm 3.019, 3.014;
# long 4.234, 4.234 ms) on 2026-11-07: the earlier warm target was a slow-mode run, and long is re-centred with it. Its
# traced targets keep the earlier means (the clamped runs were within ~1.5% below them).
_TARGETS_MS = {
    "Galaxy": {
        "kimi_k3_1": (3.99, 3.10, 7.23, 6.71),
        "glm_5_3": (3.95, 2.70, 7.08, 6.30),
        "glm_5_3-no_indexer": (2.93, 2.19, 2.93, 2.25),
    },
    "LoudBox": {
        "kimi_k3_1": (3.01, 2.77, 4.23, 4.04),
        "glm_5_3": (2.85, 2.40, 3.90, 3.60),
        "glm_5_3-no_indexer": (2.18, 2.10, 2.83, 2.72),
    },
}
PERF_TARGETS_MS = {
    (case, box, scenario, mode): ms
    for box, rows in _TARGETS_MS.items()
    for case, row in rows.items()
    for (scenario, mode), ms in zip([(s, m) for s in SCENARIO_CACHE_TOKENS for m in MODES], row)
}
# Both modes are gated two-sided. Untraced is host-bound and takes the fastest of UNTRACED_SAMPLES calls, so it gets the
# wider band.
PERF_MARGIN = {"untraced": 0.05, "traced": 0.03}
# Traced output must match the untraced output of the same forward.
TRACE_PCC = 0.9999
CI_RUN = os.environ.get("CI") == "true"


def local_cache_tokens(galaxy_cache: int, sp: int) -> int:
    """Box-local cache length: the Galaxy cache scaled by SP/GALAXY_SP like the chunk, so every box runs
    Galaxy's per-chip sequence depth."""
    return galaxy_cache * sp // GALAXY_SP


@dataclass
class SparsePerfCase:
    mla: ttMLA
    x: ttnn.Tensor
    rope: dict
    kvpe_cache: ttnn.Tensor
    index_kv_cache: ttnn.Tensor | None
    cache_layer_idx: int
    # A GLM-5.3 shared layer runs no indexer: it takes a full layer's top-k indices.
    indexer_indices: ttnn.Tensor | None = None

    def forward(self, start: int) -> ttnn.Tensor:
        """One chunk forward at `start`; the caller owns (and deallocates) the output."""
        return self.mla.forward(
            self.x,
            self.rope,
            self.kvpe_cache,
            actual_start=start,
            cache_layer_idx=self.cache_layer_idx,
            index_kv_cache=self.index_kv_cache,
            indexer_indices=self.indexer_indices,
        )


def build_sparse_perf_case(
    mesh_device,
    variant,
    config_only,
    *,
    cache: int,
    chunk: int,
    topology,
    attn_mode: str = "sparse",
    kv_cache_format=MlaKvCacheFormat.SCALED_FP8,
    tp_shard_kv: bool = True,
    shared_layer: bool = False,
) -> SparsePerfCase:
    """ttMLA, its caches and one chunk of input for forwards over a `cache + chunk` prefix.

    sparse: DSA indexer + sparse SDPA over `kv_cache_format`, caches SPxTP-sharded when `tp_shard_kv`.
    dense: the v3.1 baseline, no indexer, full-prefix ring MLA over the tiled BFP8 cache.
    shared_layer (GLM-5.3, sparse): the last shared layer, which runs no indexer and reuses the top-k indices of a
    full layer's forward at `cache`.
    """
    has_indexer = attn_mode == "sparse"
    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_device.shape
    kv_tp_axis = tp_axis if tp_shard_kv else None
    total = cache + chunk
    # The cold loop steps by `chunk`, and the indexed rope table needs total to be a whole number of chunks.
    assert cache % chunk == 0, f"cache {cache} must be a whole number of {chunk}-token chunks"
    assert total % sp == 0 and (total // sp) % 32 == 0, f"total {total} must be tile-aligned per SP={sp} chip"
    if tp_shard_kv:
        for name, n in (("cache", total), ("chunk", chunk)):
            assert (
                n % (sp * tp) == 0 and (n // (sp * tp)) % 32 == 0
            ), f"tp_shard_kv: {name} {n} must split into tile-aligned per-chip stripes over sp {sp} x tp {tp}"
    assert config_only.num_attention_heads % tp == 0 and config_only.index_n_heads % tp == 0, (
        f"{variant.name} heads (MLA={config_only.num_attention_heads}, index={config_only.index_n_heads}) "
        f"must be divisible by TP={tp}"
    )

    config = copy.deepcopy(config_only)
    config.max_seq_len = total
    # GLM-5.3 packs its index-K cache by full indexer layers (21 slots): build the last full layer (74 -> slot
    # 20) so the ring score selects a real slot, not slot 0 of a one-slot proxy.
    full_indexer_layers = num_full_indexer_layers(config) if has_indexer else None
    index_cache_layers = full_indexer_layers or 1
    mla_layer_idx = 0
    if full_indexer_layers:
        full_layer_indices = [i for i, mode in enumerate(config.indexer_types) if mode == "full"]
        assert len(full_layer_indices) == full_indexer_layers
        mla_layer_idx = full_layer_indices[-1]
    # Kimi-K3.1: the last DSA layer's slot of a 24-slot cache.
    mla_layer_num, cache_layer_idx = 1, 0
    if variant.name == "kimi_k3_1" and has_indexer:
        mla_layer_ids = variant.model_config.mla_layer_ids()
        mla_layer_num, mla_layer_idx = len(mla_layer_ids), mla_layer_ids[-1]
        cache_layer_idx = variant.model_config.mla_kv_slot(mla_layer_idx)
        index_cache_layers = mla_layer_num

    mla = ttMLA(
        config,
        dict(random_mla_weights(config)),
        mesh_device,
        topology=topology,
        layer_idx=mla_layer_idx,
        seq_len=total,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=mla_layer_num,
        has_indexer=has_indexer,
        sparse_kv_cache_format=kv_cache_format if has_indexer else MlaKvCacheFormat.BF16_RM,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(total, chunk)
    kvpe_cache = init_mla_kv_cache(
        cache_format=kv_cache_format if has_indexer else MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=mla_layer_num if has_indexer else 1,
        tp_axis=kv_tp_axis if has_indexer else None,
    )
    # Both caches start at zero: the ring indexer always covers the full prefix, so cache contents don't change
    # op shapes or timing. (cold fills them with real chunk forwards.)
    index_kv_cache = None
    if has_indexer:
        index_kv_cache = init_kvpe_cache(
            kvpe_cache_head_dim=mla._indexer.index_args.index_head_dim,
            mesh_device=mesh_device,
            seq_len=total,
            mesh_shape=list(mesh_device.shape),
            sp_axis=sp_axis,
            num_kvpe_cache_layers=index_cache_layers,
            num_users=1,
            dtype=ttnn.bfloat8_b,
            tp_axis=kv_tp_axis,
        )
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    x = ttnn.from_torch(
        make_hidden(chunk, config.hidden_size, seed=42)[:, :chunk].unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    case = SparsePerfCase(mla, x, rope, kvpe_cache, index_kv_cache, cache_layer_idx)
    if not shared_layer:
        return case

    assert has_indexer and full_indexer_layers, f"{variant.name} has no shared (indexer-reuse) layers"
    out, indices = mla.forward(
        x,
        rope,
        kvpe_cache,
        actual_start=cache,
        cache_layer_idx=cache_layer_idx,
        index_kv_cache=index_kv_cache,
        return_indexer_indices=True,
    )
    ttnn.deallocate(out)
    # Free the full layer before building the shared one: ttMLA holds reference cycles, so without a collection its
    # buffers would linger and could be freed mid-measurement.
    ttnn.deallocate(index_kv_cache)
    ttnn.deallocate(kvpe_cache.storage)
    del case, mla, kvpe_cache, index_kv_cache
    gc.collect()
    shared_layer_idx = [i for i, mode in enumerate(config.indexer_types) if mode == "shared"][-1]
    shared = ttMLA(
        config,
        dict(random_mla_weights(config)),
        mesh_device,
        topology=topology,
        layer_idx=shared_layer_idx,
        seq_len=total,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=1,
        has_indexer=True,
        sparse_kv_cache_format=kv_cache_format,
    )
    shared_kvpe_cache = init_mla_kv_cache(
        cache_format=kv_cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        tp_axis=kv_tp_axis,
    )
    return SparsePerfCase(shared, x, rope, shared_kvpe_cache, None, 0, indexer_indices=indices)


_BOX_BY_DEVICE_COUNT = {
    32: ("Galaxy", (8, 4), torus_xy_device_params),
    8: ("LoudBox", (2, 4), fabric2d_device_params),
}
_BOX = _BOX_BY_DEVICE_COUNT.get(detect_num_devices())
# Each model opens the fabric at its production packet size.
_FABRIC_PAYLOAD = {"kimi_k3_1": KimiK31Config, "glm_5_3": GLM53Config}


def _gate_params():
    if _BOX is None:
        return [
            pytest.param(
                (8, 4),
                {},
                "kimi_k3_1",
                "kimi_k3_1",
                marks=pytest.mark.skip(reason="sparse MLA perf gate runs on Galaxy and LoudBox only"),
            )
        ]
    name, mesh, device_params = _BOX
    params = []
    for case, (variant, _) in GATE_CASES.items():
        payload = _FABRIC_PAYLOAD.get(variant)
        dp = device_params(
            fabric_payload_size=moe_fabric_payload(payload) if payload else None,
            worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            trace_region_size=16 * 1024 * 1024,
        )
        params.append(pytest.param(mesh, dp, variant, case, id=f"{name.lower()}-{mesh[0]}x{mesh[1]}-{case}"))
    return params


def _device_tensors(tensor):
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]


def _cpu_ranges(cpus):
    out, run = [], []
    for cpu in cpus:
        if run and cpu != run[-1] + 1:
            out.append(run)
            run = []
        run.append(cpu)
    if run:
        out.append(run)
    return ",".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else f"{r[0]}" for r in out)


def _host_state():
    try:
        affinity = sorted(os.sched_getaffinity(0))
        return (
            f"{len(affinity)} CPUs allowed ({_cpu_ranges(affinity)}), {os.cpu_count()} online, "
            f"load {os.getloadavg()[0]:.1f}"
        )
    except OSError as e:
        return f"unavailable ({e})"


def _thp_state():
    """Transparent huge page settings and this process's anonymous huge-page memory: a CI diagnostic."""
    out = []
    for name in ("enabled", "defrag"):
        try:
            with open(f"/sys/kernel/mm/transparent_hugepage/{name}") as f:
                out.append(f"thp {name} {f.read().strip()}")
        except OSError:
            pass
    try:
        with open("/proc/self/smaps_rollup") as f:
            kib = {}
            for line in f:
                key, _, value = line.partition(":")
                if value.split():
                    kib[key] = value.split()[0]
        rss, huge = (int(kib.get(k, -1024)) // 1024 for k in ("Rss", "AnonHugePages"))
        out.append(f"rss {rss} MiB, anon huge {huge} MiB")
    except Exception:
        pass
    return "; ".join(out)


_VMSTAT_KEYS = ("compact_stall", "compact_fail", "thp_fault_alloc", "thp_fault_fallback", "thp_collapse_alloc")


def _rusage():
    """(minor faults, involuntary context switches) of the whole process, all threads."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_minflt, ru.ru_nivcsw


def _vmstat():
    """System-wide compaction and THP allocation counters (/proc/vmstat), or {} if unavailable."""
    try:
        with open("/proc/vmstat") as f:
            pairs = (line.split() for line in f)
            return {k: int(v) for k, v, *_ in pairs if k in _VMSTAT_KEYS}
    except Exception:
        return {}


def _time_untraced(mesh_device, case, start, cpus=None, repeat=None, enqueue=None, host=None):
    """Host ms of each timed eager forward through sync, and the last timed forward's per-device output. Appends the CPU
    the main thread ran on after each sample to `cpus`, the ms until forward() returned (before the sync) to `enqueue`,
    per-sample host counters to `host` (process minor faults during enqueue and during the sync, involuntary context
    switches, and /proc/vmstat compaction/THP deltas), and the output of one more, untimed forward to `repeat`, if
    given."""
    for _ in range(UNTRACED_WARMUP):  # the first call compiles
        ttnn.deallocate(case.forward(start))
        ttnn.synchronize_device(mesh_device)
    samples = []
    for i in range(UNTRACED_SAMPLES):
        if host is not None:
            vm0, (f0, c0) = _vmstat(), _rusage()
        t0 = time.perf_counter()
        out = case.forward(start)
        t1 = time.perf_counter()
        if host is not None:
            f1, _ = _rusage()  # outside the enqueue window, ~1 us inside the total one
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - t0) * 1e3)
        if host is not None:
            (f2, c2), vm2 = _rusage(), _vmstat()
            if vm0 and vm2:
                vm = " ".join(f"{k}={vm2[k] - vm0[k]}" for k in sorted(vm0.keys() & vm2.keys()) if vm2[k] != vm0[k])
            else:
                vm = "vmstat unavailable"
            host.append(f"{f1 - f0}/{f2 - f1}/{c2 - c0}" + (f" {vm}" if vm else ""))
        if enqueue is not None:
            enqueue.append((t1 - t0) * 1e3)
        if cpus is not None:
            cpus.append(current_cpu())
        if i == UNTRACED_SAMPLES - 1:
            reference = _device_tensors(out)
        ttnn.deallocate(out)
    if repeat is not None:
        out = case.forward(start)
        repeat.append(_device_tensors(out))
        ttnn.deallocate(out)
    return samples, reference


def _op_host_profile(mesh_device, case, start, top=12):
    """One untimed eager forward with each ttnn op call timed on the host: a CI diagnostic for untraced host-time noise
    (which ops grow when a run is slow). Times are inclusive (a Python op that calls other ttnn ops counts them too) and
    cover only ops called through ttnn's FastOperation wrapper (CI runs in fast-runtime mode). Returns a summary."""
    from ttnn.decorators import FastOperation

    per_op, original = {}, FastOperation.__call__

    def timed(self, *args, **kwargs):
        t = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            calls, total = per_op.get(self.python_fully_qualified_name, (0, 0.0))
            per_op[self.python_fully_qualified_name] = (calls + 1, total + time.perf_counter() - t)

    entries, out = mesh_device.num_program_cache_entries(), None
    try:
        FastOperation.__call__ = timed
        try:
            t0 = time.perf_counter()
            out = case.forward(start)
            t1 = time.perf_counter()
        finally:
            FastOperation.__call__ = original
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
    finally:
        if out is not None:
            ttnn.deallocate(out)
    rows = sorted(per_op.items(), key=lambda kv: -kv[1][1])[:top]
    ops = ", ".join(f"{name.rsplit('.', 1)[-1]} {calls}x {total * 1e3:.3f}" for name, (calls, total) in rows)
    return (
        f"forward() {(t1 - t0) * 1e3:.3f} ms, sync {(t2 - t1) * 1e3:.3f} ms, "
        f"{sum(c for c, _ in per_op.values())} op calls, "
        f"{sum(t for _, t in per_op.values()) * 1e3:.3f} ms in ops (inclusive), "
        f"program cache +{mesh_device.num_program_cache_entries() - entries}; top ops (ms): {ops}"
    )


def _time_traced(mesh_device, case, start, repeat=None):
    """Host ms of one segmented trace replay (it ends in a device sync), and the trace's output. If `repeat` is given,
    replays once more, untimed, and appends that replay's output to it."""
    controller = SubDeviceTraceController(mesh_device)
    case.mla.set_trace_controller(controller)
    capture_out, capturing = None, False
    try:
        ttnn.deallocate(case.forward(start))  # compile under the controller
        ttnn.synchronize_device(mesh_device)
        controller.begin_capture()
        capturing = True
        capture_out = case.forward(start)
        controller.end_capture()
        capturing = False
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        controller.replay()
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        out = _device_tensors(capture_out)
        if repeat is not None:
            controller.replay()
            repeat.append(_device_tensors(capture_out))
        return [elapsed_ms], out
    finally:
        if capturing:
            try:
                controller.end_capture()
            except Exception:
                pass
        try:
            controller.release()
        finally:
            case.mla.set_trace_controller(None)
            if capture_out is not None:
                ttnn.deallocate(capture_out)


@skip_with_llk_assert("LLK assertions perturb kernel timing")
@skip_with_watcher("Watcher perturbs kernel timing")
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name",
    _gate_params(),
    indirect=["mesh_device", "device_params", "variant"],
)
@pytest.mark.parametrize("scenario", list(SCENARIO_CACHE_TOKENS))
def test_sparse_mla_perf(request, mesh_device, device_params, variant, case_name, config_only, scenario):
    box = _BOX[0]
    shared_layer = GATE_CASES[case_name][1]
    label = f"{case_name} sparse MLA {box} {scenario}"
    if box == "Galaxy":
        require_high_power(f"{label} perf")
    sp, tp = mesh_device.shape
    chunk = CHUNK_TOKENS * sp // GALAXY_SP
    cache = local_cache_tokens(SCENARIO_CACHE_TOKENS[scenario], sp)
    case = build_sparse_perf_case(
        mesh_device,
        variant,
        config_only,
        cache=cache,
        chunk=chunk,
        topology=per_axis_topology(device_params["fabric_config"]),
        shared_layer=shared_layer,
    )
    start = cache
    clock = hold_full_cpu_clock()
    request.addfinalizer(clock.restore)
    print(f"{label} host: {clock.status}")
    samples = {}
    cpus = []
    eager_repeat, traced_repeat = [], []
    enqueue, host = [], []
    thp_before = _thp_state()
    samples["untraced"], eager_out = _time_untraced(mesh_device, case, start, cpus, eager_repeat, enqueue, host)
    print(f"{label} host: {_host_state()}; untraced sample CPUs {' '.join(map(str, cpus))}")
    print(f"{label} untraced enqueue (forward() returns, ms): {' '.join(f'{x:.3f}' for x in enqueue)}")
    print(
        f"{label} untraced host per sample (process-wide minor faults in enqueue/sync, involuntary ctx switches; "
        f"system-wide vmstat deltas; windows approximate): {' | '.join(host)}; "
        f"before: {thp_before}; after: {_thp_state()}"
    )
    print(f"{label} untraced op profile: {_op_host_profile(mesh_device, case, start)}")
    samples["traced"], traced_out = _time_traced(mesh_device, case, start, traced_repeat)

    def min_pcc(xs, ys):
        return min(comp_pcc(a, b, TRACE_PCC)[1] for a, b in zip(xs, ys))

    trace_pcc = min_pcc(eager_out, traced_out)

    failures, unmeasured = [], []
    for mode in MODES:
        measured_ms = min(samples[mode])
        target_ms = PERF_TARGETS_MS.get((case_name, box, scenario, mode))
        margin = PERF_MARGIN[mode]
        within = target_ms is not None and abs(measured_ms - target_ms) <= target_ms * margin
        print(  # print, not logger: CI runs perf jobs at LOGURU_LEVEL=ERROR
            f"{label} {mode}: {measured_ms:.3f} ms "
            f"(max {max(samples[mode]):.3f}); target {target_ms} ms ±{margin:.0%}; "
            f"samples {' '.join(f'{x:.3f}' for x in samples[mode])}"
        )
        emit_perf_metrics(
            f"{case_name.replace('-', '_')}_sparse_mla_{box.lower()}_{scenario}_{mode}",
            f"{label} ({mode})",
            {
                "box": box,
                "mesh": f"{sp}x{tp}",
                "scenario": scenario,
                "mode": mode,
                "chunk_tokens": chunk,
                "cache_tokens": cache,
                "fabric": device_params["fabric_config"].name,
                "fabric_payload_bytes": (
                    moe_fabric_payload(_FABRIC_PAYLOAD[variant.name]) if variant.name in _FABRIC_PAYLOAD else None
                ),
                "kv_cache_format": "scaled_fp8",
                "weights": "random",
                "samples_ms": samples[mode],
                "measured_ms": measured_ms,
                "target_ms": target_ms,
                "margin_pct": margin * 100.0,
                "within_band": within,
            },
        )
        if target_ms is None:
            unmeasured.append(f"{mode} {measured_ms:.3f} ms")
        elif not within:
            lower, upper = target_ms * (1 - margin), target_ms * (1 + margin)
            direction = "faster than the target, re-centre it" if measured_ms < lower else "slower than the target"
            failures.append(
                f"{mode}: {measured_ms:.3f} ms outside [{lower:.3f}, {upper:.3f}] "
                f"(target {target_ms:.3f} ms ±{margin:.0%}); {direction}"
            )
    if trace_pcc < TRACE_PCC:
        # Within-mode repeatability, for triage: if both pairs match, that suggests a consistent eager/trace
        # discrepancy; a pair that does not match means that mode failed this repeatability check.
        failures.append(
            f"traced output differs from untraced: min per-device PCC {trace_pcc:.6f} (eager vs eager "
            f"{min_pcc(eager_repeat[0], eager_out):.6f}, replay vs replay {min_pcc(traced_repeat[0], traced_out):.6f})"
        )
    assert not failures, f"{label}: " + "; ".join(failures)
    if unmeasured:  # report-only until this case has targets on this box
        pytest.skip(f"{label}: no targets yet; measured " + ", ".join(unmeasured))


# ============================================================================
# Exploration matrix: every variant x scenario x cache/attention case, host time only, no gate.
# ============================================================================
# cold: forward every chunk start=0,chunk,...,cache from empty caches (the last is the warm step); the region
# spans all chunks, so it is the whole cold-start prefill cost.
COLD_CACHE_TOKENS = WARM_CACHE_TOKENS
MATRIX_SCENARIOS = {"warm": WARM_CACHE_TOKENS, "cold": COLD_CACHE_TOKENS, "long": LONG_CACHE_TOKENS}
MATRIX_VARIANTS = ("glm_5_3", "kimi_k3_1")
# (id, attn_mode, kv_cache_format, tp_shard_kv). Dense ring MLA keeps its tiled BFP8 cache and has no dedup.
MATRIX_CASES = (
    ("sparse-kv_bf16", "sparse", MlaKvCacheFormat.BF16_RM, True),
    ("sparse-kv_scaled_fp8", "sparse", MlaKvCacheFormat.SCALED_FP8, True),
    ("dense", "dense", MlaKvCacheFormat.BF16_RM, False),
)
_MATRIX_BOX_BY_DEVICE_COUNT = {**_BOX_BY_DEVICE_COUNT, 4: ("QuietBox", (2, 2), fabric2d_device_params)}
_MATRIX_BOX = _MATRIX_BOX_BY_DEVICE_COUNT.get(detect_num_devices())


def _matrix_case_runs(variant: str, scenario: str, case_id: str) -> bool:
    """Kimi-K3.1 runs its production cache (scaled FP8) everywhere; its comparison points only in long."""
    return variant != "kimi_k3_1" or case_id == "sparse-kv_scaled_fp8" or scenario == "long"


def _matrix_params():
    if _MATRIX_BOX is None:
        return [
            pytest.param(
                (8, 4),
                {},
                "glm_5_3",
                "warm",
                "sparse",
                MlaKvCacheFormat.SCALED_FP8,
                True,
                marks=pytest.mark.skip(reason="sparse MLA perf runs on Blackhole QuietBox, LoudBox and Galaxy only"),
            )
        ]
    name, mesh, device_params = _MATRIX_BOX
    params = []
    for variant in MATRIX_VARIANTS:
        payload = _FABRIC_PAYLOAD.get(variant)
        dp = device_params(
            fabric_payload_size=moe_fabric_payload(payload) if payload else None,
            worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            trace_region_size=16 * 1024 * 1024,
        )
        for scenario in MATRIX_SCENARIOS:
            for case_id, attn_mode, kv_cache_format, tp_shard_kv in MATRIX_CASES:
                if _matrix_case_runs(variant, scenario, case_id):
                    params.append(
                        pytest.param(
                            mesh,
                            dp,
                            variant,
                            scenario,
                            attn_mode,
                            kv_cache_format,
                            tp_shard_kv,
                            id=f"{name.lower()}-{mesh[0]}x{mesh[1]}-{variant}-{scenario}-{case_id}",
                        )
                    )
    return params


@skip_with_llk_assert("LLK assertions perturb kernel timing")
@skip_with_watcher("Watcher perturbs kernel timing")
@pytest.mark.skipif(CI_RUN, reason="exploration matrix; CI runs test_sparse_mla_perf")
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, scenario, attn_mode, kv_cache_format, tp_shard_kv",
    _matrix_params(),
    indirect=["mesh_device", "device_params", "variant"],
)
def test_mla_chunked_perf(
    mesh_device, device_params, variant, config_only, scenario, attn_mode, kv_cache_format, tp_shard_kv
):
    """Host time of one chunk (warm, long) or of the whole cold fill, untraced and traced. Reports only."""
    sp, _ = mesh_device.shape
    chunk = CHUNK_TOKENS * sp // GALAXY_SP
    cache = local_cache_tokens(MATRIX_SCENARIOS[scenario], sp)
    case = build_sparse_perf_case(
        mesh_device,
        variant,
        config_only,
        cache=cache,
        chunk=chunk,
        topology=per_axis_topology(device_params["fabric_config"]),
        attn_mode=attn_mode,
        kv_cache_format=kv_cache_format,
        tp_shard_kv=tp_shard_kv,
    )
    label = f"{_MATRIX_BOX[0]} {variant.name} {scenario} {attn_mode}"
    if attn_mode == "sparse":
        label += f" {kv_cache_format.name} {'tp-sharded' if tp_shard_kv else 'tp-replicated'} KV"

    if scenario != "cold":
        untraced, _ = _time_untraced(mesh_device, case, cache)
        traced, _ = _time_traced(mesh_device, case, cache)
        print(f"\n{label}: untraced {min(untraced):.3f} ms (fastest of {len(untraced)}), traced {traced[0]:.3f} ms")
        return

    # cold: one untimed fill compiles every start; then one timed untraced fill, then each chunk traced.
    starts = list(range(0, cache + chunk, chunk))
    for start in starts:
        ttnn.deallocate(case.forward(start))
    ttnn.synchronize_device(mesh_device)
    untraced_rows = []
    for start in starts:
        t0 = time.perf_counter()
        out = case.forward(start)
        ttnn.synchronize_device(mesh_device)
        untraced_rows.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(out)
    traced_rows = [_time_traced(mesh_device, case, start)[0][0] for start in starts]
    rows = list(zip(starts, untraced_rows, traced_rows))
    print(
        f"\n{label}: {len(starts)} chunks, untraced {sum(r[1] for r in rows):.3f} ms, "
        f"traced {sum(r[2] for r in rows):.3f} ms"
    )
    for start, untraced_ms, traced_ms in rows:
        print(f"  cache depth {start:>7}: untraced {untraced_ms:.3f} ms, traced {traced_ms:.3f} ms")
