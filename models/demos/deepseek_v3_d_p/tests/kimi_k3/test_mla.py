# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 MLA correctness and performance for one attention layer.

Coverage:
* ``test_mla_loudbox_correctness``: SP4 x TP2, full 96 heads, 5,120 fresh tokens
  on a 51,200-token prefix. Uses Galaxy's CPU-reference output/KV PCC checks, with no timing gate.
* ``test_mla_loudbox_compute_perf``: SP4 x TP2, 48 heads, 2,560 fresh tokens
  on a 51,200-token prefix. Matches Galaxy's 640 queries, 24 heads, and prefix-attention arithmetic
  per chip; gates device time. Four larger KV shards change ring communication and kernel shapes.
* ``test_mla_chunked_perf``: existing cold/warm/long sweep on QuietBox, LoudBox, or Galaxy.
  Chunk and prefix scale with SP/8 and global heads with TP/4, preserving per-ring-step shapes.
  LoudBox uses SP4 x TP2, 48 heads, 2,560 fresh tokens, and a 25,600-token warm prefix.

Performance measures warmed trace replays. Host E2E includes device synchronization; device time
sums each program's MAX duration across chips. Correctness uses the shared Galaxy test driver.

Run the fixed LoudBox cases::

    scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_mla.py -k correctness -s
    scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_mla.py -m perf -k compute_perf -s

Select the sweep with ``-m perf -k chunked_perf``; narrow further with ``warm``, ``cold``, or ``long``.
Sweep sizes can be overridden with ``K3_MLA_PERF_CACHE``, ``K3_MLA_PERF_CHUNK``, and
``K3_MLA_PERF_LONG_CACHE`` (Galaxy-global sizes, scaled by SP). Fixed LoudBox cases ignore these
size overrides. Profiler outputs live under ``generated/profiler/kimi_k3_dense_mla_perf``.
"""

import copy
import csv
import datetime
import json
import os
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import pandas as pd
import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_plugin import is_marker_explicitly_selected
from models.demos.deepseek_v3_d_p.tests.test_mla import _run_chunked_prefill
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS, PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

pytestmark = [
    pytest.mark.parametrize("variant", ["kimi_k3"], ids=["k3"], indirect=True),
    pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 MLA tests require Blackhole"),
    pytest.mark.timeout(0),
]

# Workload configuration

LOUDBOX_PREFIX_TOKENS = 50 * 1024
LOUDBOX_MESH_SHAPE = (4, 2)
LOUDBOX_PERF_CHUNK_TOKENS = LOUDBOX_MESH_SHAPE[0] * PREFILL_CHUNK_TOKENS_PER_CHIP
LOUDBOX_PERF_SAMPLES = 3
# Blackhole LoudBox, FABRIC_2D, 2026-10-13: 10.018 / 10.067 / 10.082 ms traced samples.
LOUDBOX_PERF_TARGET_NS = 10_100_000
LOUDBOX_PERF_MARGIN = 0.03
LOUDBOX_DEVICE_PARAMS = fabric2d_device_params(
    fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE,
    l1_small_size=KimiK3Config.L1_SMALL_SIZE,
    trace_region_size=16 * 1024 * 1024,
)

GALAXY_SP = 8
GALAXY_TP = 4
CACHE_TOKENS = int(os.environ.get("K3_MLA_PERF_CACHE", 50 * 1024))
CHUNK_TOKENS = int(os.environ.get("K3_MLA_PERF_CHUNK", 5 * 1024))
LONG_CACHE_TOKENS = int(os.environ.get("K3_MLA_PERF_LONG_CACHE", 500 * 1024))
RT_RECORD_TIMEOUT_S = float(os.environ.get("K3_MLA_PERF_RT_TIMEOUT", 30.0))
WRITE_OPS_DUMP = os.environ.get("K3_MLA_PERF_OPS_DUMP", "") not in ("", "0", "false")

SCENARIOS = {
    "warm": {"cache": CACHE_TOKENS, "loop": False},
    "cold": {"cache": CACHE_TOKENS, "loop": True},
    "long": {"cache": LONG_CACHE_TOKENS, "loop": False},
}
SCENARIO = os.environ.get("K3_MLA_PERF_SCENARIO", "warm")

GALAXY_HEADS_PER_DEVICE = KimiK3Config.NUM_ATTENTION_HEADS // GALAXY_TP


@dataclass(frozen=True)
class PerfWorkload:
    system_name: str
    num_devices: int
    mesh_shape: tuple[int, int]
    chunk_tokens: int
    num_attention_heads: int

    @property
    def sp(self) -> int:
        return self.mesh_shape[0]

    @property
    def tp(self) -> int:
        return self.mesh_shape[1]

    @property
    def local_attention_heads(self) -> int:
        return self.num_attention_heads // self.tp

    @property
    def id(self) -> str:
        return f"{self.system_name.lower()}_sp{self.sp}xtp{self.tp}"


_SYSTEM_BY_DEVICE_COUNT = {
    4: ("QuietBox", (4, 1)),
    8: ("LoudBox", (4, 2)),
    32: ("Galaxy", (8, 4)),
}


def _exact_div(numerator: int, denominator: int, label: str) -> int:
    if numerator % denominator != 0:
        raise ValueError(f"{label}={numerator} must be divisible by {denominator}")
    return numerator // denominator


def _detect_perf_workload() -> tuple[PerfWorkload, str | None]:
    num_devices = detect_num_devices()
    system = _SYSTEM_BY_DEVICE_COUNT.get(num_devices)
    if system is None:
        placeholder = PerfWorkload("unsupported", num_devices, (1, 1), CHUNK_TOKENS, GALAXY_HEADS_PER_DEVICE)
        return (
            placeholder,
            f"K3 MLA perf supports Blackhole QuietBox/LoudBox/Galaxy only (detected {num_devices} chips)",
        )

    system_name, mesh_shape = system
    sp, tp = mesh_shape
    local_chunk = _exact_div(CHUNK_TOKENS, GALAXY_SP, "K3_MLA_PERF_CHUNK")
    workload = PerfWorkload(
        system_name=system_name,
        num_devices=num_devices,
        mesh_shape=mesh_shape,
        chunk_tokens=local_chunk * sp,
        num_attention_heads=GALAXY_HEADS_PER_DEVICE * tp,
    )
    return workload, None


PERF_WORKLOAD, PERF_SKIP_REASON = _detect_perf_workload()
PERF_FABRIC = (
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    if PERF_WORKLOAD.mesh_shape == (GALAXY_SP, GALAXY_TP)
    else ttnn.FabricConfig.FABRIC_2D
)

_DEVICE_PARAMS = (
    torus_xy_device_params if PERF_FABRIC == ttnn.FabricConfig.FABRIC_2D_TORUS_XY else fabric2d_device_params
)(
    fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE,
    l1_small_size=KimiK3Config.L1_SMALL_SIZE,
    trace_region_size=16 * 1024 * 1024,
)
assert _DEVICE_PARAMS["fabric_config"] == PERF_FABRIC

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[4]
_OUTPUT_DIR = _REPO_ROOT / "generated" / "profiler" / "kimi_k3_dense_mla_perf"


@pytest.fixture(autouse=True)
def _require_perf(request):
    if request.node.get_closest_marker("perf") is None or is_marker_explicitly_selected(request.config, "perf"):
        return
    pytest.skip("K3 MLA perf tests require explicit marker selection: pytest -m perf")


def _local_tokens(galaxy_tokens: int, sp: int) -> int:
    return _exact_div(galaxy_tokens, GALAXY_SP, "Galaxy token count") * sp


def _shape_normalized_config_and_weights(fixture_config, fixture_weights, tp: int):
    """Keep K3's Galaxy head count per TP rank when running on a smaller mesh."""
    config = copy.deepcopy(fixture_config)
    source_heads = config.num_attention_heads
    target_heads = GALAXY_HEADS_PER_DEVICE * tp
    assert source_heads == KimiK3Config.NUM_ATTENTION_HEADS
    assert 0 < target_heads <= source_heads

    weights = dict(fixture_weights)
    if target_heads != source_heads:
        q_width = config.qk_nope_head_dim + config.qk_rope_head_dim
        kv_width = config.qk_nope_head_dim + config.v_head_dim
        v_width = config.v_head_dim
        weights["q_b_proj.weight"] = weights["q_b_proj.weight"][: target_heads * q_width].contiguous()
        weights["kv_b_proj.weight"] = weights["kv_b_proj.weight"][: target_heads * kv_width].contiguous()
        weights["o_proj.weight"] = weights["o_proj.weight"][:, : target_heads * v_width].contiguous()
        weights["g_proj.weight"] = weights["g_proj.weight"][: target_heads * v_width].contiguous()

    config.num_attention_heads = target_heads
    config.num_key_value_heads = target_heads
    return config, weights


def _require_rt_profiler() -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("realtime profiler is inactive (K3 MLA perf requires eligible Blackhole hardware)")


# Shared trace profiling and reports

_OP_CODE_RULES = (
    ("/ring_joint_sdpa", "RingJointSDPA"),
    ("/ring_attention_all_gather", "RingJointSDPA"),
    ("/ccl/all_gather_async/", "AllGatherAsync"),
    ("/ccl/reduce_scatter_minimal_async/", "ReduceScatterMinimalAsync"),
    ("/nlp_create_qkv_heads", "NlpCreateHeads"),
    ("/nlp_concat_heads", "NlpConcatHeads"),
    ("/update_padded_kv_cache/", "UpdateCache"),
    ("/fast_reduce_nc/", "FastReduceNC"),
    ("/matmul/", "Matmul"),
    ("/layernorm/", "LayerNorm"),
    ("/untilize_with_unpadding/", "UntilizeWithUnpadding"),
    ("/tilize_with_val_padding/", "TilizeWithValPadding"),
    ("/untilize/", "Untilize"),
    ("/tilize/", "Tilize"),
    ("/concat/", "Concat"),
    ("/permute/", "Permute"),
    ("/slice/", "Slice"),
    ("/typecast/", "Typecast"),
    ("/copy/", "Copy"),
    ("/binary", "BinaryNg"),
    ("/unary", "UnaryEltwise"),
)


def _op_code(kernel_sources) -> str:
    paths = "\n".join(source.replace("\\", "/") for source in kernel_sources)
    for needle, code in _OP_CODE_RULES:
        if needle in paths:
            return code
    names = set()
    for source in kernel_sources:
        parts = source.replace("\\", "/").split("/")
        if "operations" in parts:
            index = parts.index("operations") + 1
            if index < len(parts):
                name = parts[index]
                if name == "experimental" and index + 1 < len(parts):
                    name = parts[index + 1]
                names.add(name)
    return "+".join(sorted(names)) if names else "unknown"


def _profile_forward(mesh_device, run_fn) -> dict:
    host_duration_ns = 0

    def measured():
        nonlocal host_duration_ns
        start_ns = time.perf_counter_ns()
        result = run_fn()
        ttnn.synchronize_device(mesh_device)
        host_duration_ns = time.perf_counter_ns() - start_ns
        return result

    _, records = profile_realtime_program(
        mesh_device,
        measured,
        collect_all=True,
        record_timeout_seconds=RT_RECORD_TIMEOUT_S,
    )
    programs = OrderedDict()
    for record in records:
        runtime_id = record["runtime_id"]
        if not runtime_id:
            continue
        duration_ns = float(record["duration_ns"])
        current = programs.get(runtime_id)
        if current is None:
            programs[runtime_id] = {
                "duration_ns": duration_ns,
                "kernel_sources": record["kernel_sources"],
            }
        else:
            current["duration_ns"] = max(current["duration_ns"], duration_ns)
    assert programs, "realtime profiler returned no valid K3 MLA program records"
    return {"programs": programs, "host_duration_ns": host_duration_ns}


def _profile_traced_forward(mesh_device, mla, run_fn) -> dict:
    """Compile/capture/warm/measure one forward and release every trace allocation."""
    controller = SubDeviceTraceController(mesh_device)
    compile_out = None
    capture_out = None
    capture_started = False
    capture_ended = False
    mla.set_trace_controller(controller)
    try:
        compile_out = run_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(compile_out)
        compile_out = None

        controller.begin_capture()
        capture_started = True
        capture_out = run_fn()
        controller.end_capture()
        capture_ended = True
        ttnn.synchronize_device(mesh_device)

        _profile_forward(mesh_device, controller.replay)  # exclude first-replay jitter
        measured = _profile_forward(mesh_device, controller.replay)
        measured["trace_segments"] = controller.num_segments
        return measured
    finally:
        if capture_started and not capture_ended:
            try:
                controller.end_capture()
            except Exception:
                pass
        try:
            controller.release()
        finally:
            mla.set_trace_controller(None)
            if capture_out is not None:
                ttnn.deallocate(capture_out)
            if compile_out is not None:
                ttnn.deallocate(compile_out)


def _programs_to_frame(forward: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "OP CODE": _op_code(info["kernel_sources"]),
                "DEVICE KERNEL DURATION [ns]": info["duration_ns"],
            }
            for info in forward["programs"].values()
        ]
    )


def _by_op(frame: pd.DataFrame) -> pd.DataFrame:
    duration = "DEVICE KERNEL DURATION [ns]"
    return (
        frame.groupby("OP CODE")
        .agg(count=(duration, "count"), inclusive_ns=(duration, "sum"), avg_inclusive_ns=(duration, "mean"))
        .sort_values("inclusive_ns", ascending=False)
    )


def _write_manifest(*, scenario: str, cache: int, chunk: int, forwards: list[dict]) -> None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = (
            subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=_REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError):
        commit, branch = None, None
    manifest = {
        "schema_version": 1,
        "execution": "traced_replay",
        "profiler": "realtime",
        "variant": "kimi_k3",
        "attention": "dense_mla",
        "kv_cache_format": MlaKvCacheFormat.BFP8_TILE.value,
        "scenario": scenario,
        "commit": commit,
        "branch": branch,
        "device": {
            "num_devices": PERF_WORKLOAD.num_devices,
            "box": PERF_WORKLOAD.system_name,
            "mesh_sp": PERF_WORKLOAD.sp,
            "mesh_tp": PERF_WORKLOAD.tp,
            "fabric": getattr(PERF_FABRIC, "name", str(PERF_FABRIC)),
        },
        "workload": {
            "galaxy_cache_tokens": SCENARIOS[scenario]["cache"],
            "galaxy_chunk_tokens": CHUNK_TOKENS,
            "local_cache_tokens": cache,
            "local_chunk_tokens": chunk,
            "attention_heads": PERF_WORKLOAD.num_attention_heads,
            "local_attention_heads": PERF_WORKLOAD.local_attention_heads,
            "forward_count": len(forwards),
            "trace_segments": [forward["trace_segments"] for forward in forwards],
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": (
            "scripts/run_safe_pytest.sh "
            "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_mla.py "
            f"-m perf -k {scenario} -s"
        ),
    }
    (_OUTPUT_DIR / f"run_manifest_{scenario}.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _write_outputs(*, scenario: str, cache: int, chunk: int, forwards: list[dict], by_op: pd.DataFrame) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = _OUTPUT_DIR / f"kimi_k3_dense_mla_perf_{scenario}"
    by_op.reset_index().to_csv(stem.with_suffix(".csv"), index=False)
    pd.DataFrame(
        [
            {
                "scenario": scenario,
                "HOST E2E DURATION [ns]": sum(item["host_duration_ns"] for item in forwards),
                "forward_count": len(forwards),
                "trace_segments": [item["trace_segments"] for item in forwards],
            }
        ]
    ).to_csv(f"{stem}_e2e.csv", index=False)

    if scenario == "cold":
        rows = []
        for iteration, forward in enumerate(forwards):
            grouped = _by_op(_programs_to_frame(forward)).reset_index()
            grouped.insert(0, "cache_depth_tokens", iteration * chunk)
            grouped.insert(0, "iteration", iteration)
            rows.append(grouped)
        pd.concat(rows, ignore_index=True).to_csv(f"{stem}_by_iter.csv", index=False)

    if WRITE_OPS_DUMP:
        with open(f"{stem}_ops.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["forward", "seq", "runtime_id", "OP CODE", "DEVICE KERNEL DURATION [ns]", "kernel_sources"]
            )
            for forward_index, forward in enumerate(forwards):
                for sequence, (runtime_id, info) in enumerate(forward["programs"].items()):
                    writer.writerow(
                        [
                            forward_index,
                            sequence,
                            runtime_id,
                            _op_code(info["kernel_sources"]),
                            info["duration_ns"],
                            "|".join(info["kernel_sources"]),
                        ]
                    )
    _write_manifest(scenario=scenario, cache=cache, chunk=chunk, forwards=forwards)


# Shape-normalized performance sweep


@pytest.mark.perf
@pytest.mark.parametrize("mesh_device", [PERF_WORKLOAD.mesh_shape], ids=[PERF_WORKLOAD.id], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [_DEVICE_PARAMS],
    ids=["torus-xy" if PERF_FABRIC == ttnn.FabricConfig.FABRIC_2D_TORUS_XY else "fabric2d"],
    indirect=True,
)
@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=list(SCENARIOS))
@skip_with_llk_assert("LLK assertions perturb kernel timing; K3 MLA perf measurements require them disabled")
@skip_with_watcher("Watcher perturbs kernel timing; K3 MLA perf measurements require it disabled")
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="K3 MLA perf requires dedicated Blackhole hardware")
def test_mla_chunked_perf(mesh_device, device_params, scenario, variant, random_weights):
    if PERF_SKIP_REASON:
        pytest.skip(PERF_SKIP_REASON)
    _require_rt_profiler()

    sp_axis, tp_axis = 0, 1
    sp, tp = tuple(mesh_device.shape)
    assert (sp, tp) == PERF_WORKLOAD.mesh_shape
    assert PERF_WORKLOAD.num_attention_heads % tp == 0

    cache = _local_tokens(SCENARIOS[scenario]["cache"], sp)
    chunk = PERF_WORKLOAD.chunk_tokens
    total = cache + chunk
    assert cache % chunk == 0, f"cache {cache} must be a whole number of {chunk}-token chunks"
    assert chunk % (sp * ttnn.TILE_SIZE) == 0
    is_cold = SCENARIOS[scenario]["loop"]

    fixture_config, weights = random_weights
    config, weights = _shape_normalized_config_and_weights(fixture_config, weights, tp)
    config.max_seq_len = total
    config.max_position_embeddings = total
    assert config.mla_use_nope and config.mla_use_output_gate

    mla = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=KimiK3Config.mla_layer_ids()[0],
        seq_len=total,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        topology=per_axis_topology(PERF_FABRIC),
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=1,
        has_indexer=False,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(total, chunk)
    assert rope == {}, "K3 NoPE must not allocate a long-context rotary table"
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    generator = torch.Generator().manual_seed(42)
    hidden = torch.randn(1, 1, chunk, config.hidden_size, generator=generator, dtype=torch.bfloat16)
    shard_dims = [None, None]
    shard_dims[sp_axis], shard_dims[tp_axis] = -2, -1
    tt_hidden = ttnn.from_torch(
        hidden,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    starts = list(range(0, cache + chunk, chunk)) if is_cold else [cache]
    logger.info(
        f"profiling {PERF_WORKLOAD.system_name} SP={sp}xTP={tp} K3 dense/{scenario} proxy: "
        f"{len(starts)} x {chunk}-token forward(s), cache={cache}, total={total}, "
        f"local_query_rows={chunk // sp}, local_heads={config.num_attention_heads // tp}"
    )

    def one_forward(start):
        return mla.forward(tt_hidden, rope, kvpe_cache, actual_start=start)

    forwards = []
    for start in starts:
        ttnn.synchronize_device(mesh_device)
        forwards.append(_profile_traced_forward(mesh_device, mla, lambda start=start: one_forward(start)))

    frame = pd.concat([_programs_to_frame(forward) for forward in forwards], ignore_index=True)
    by_op = _by_op(frame)
    host_ns = sum(forward["host_duration_ns"] for forward in forwards)
    header = f"{'OP CODE':<42}{'count':>7}{'device_ms':>15}{'avg_us':>12}"
    rows = [
        f"{name:<42}{int(row['count']):>7}{row['inclusive_ns'] / 1e6:>15.3f}" f"{row['avg_inclusive_ns'] / 1e3:>12.1f}"
        for name, row in by_op.iterrows()
    ]
    report = "\n".join(
        [
            f"kimi_k3 MLA chunked perf [dense/{scenario}] — {PERF_WORKLOAD.system_name} proxy SP={sp}xTP={tp}",
            f"Galaxy-equivalent target: chunk={CHUNK_TOKENS}, cache={SCENARIOS[scenario]['cache']}",
            f"traced host E2E: {host_ns / 1e6:.3f} ms over {len(forwards)} forward(s)",
            header,
            "-" * len(header),
            *rows,
        ]
    )
    logger.info("\n" + report)
    print("\n" + report)
    if is_cold:
        per_iteration = [forward["host_duration_ns"] / 1e6 for forward in forwards]
        logger.info("cold per-iteration traced host ms: " + " ".join(f"{value:.3f}" for value in per_iteration))

    _write_outputs(scenario=scenario, cache=cache, chunk=chunk, forwards=forwards, by_op=by_op)


# LoudBox correctness and compute proxy


@pytest.mark.parametrize("mesh_device", [LOUDBOX_MESH_SHAPE], ids=["sp4xtp2"], indirect=True)
@pytest.mark.parametrize("device_params", [LOUDBOX_DEVICE_PARAMS], ids=["fabric2d"], indirect=True)
@pytest.mark.skipif(detect_num_devices() != 8, reason="These workloads require an eight-device LoudBox")
def test_mla_loudbox_correctness(request, mesh_device, device_params, variant):
    """Full K3 dimensions and 50k+5k; Galaxy's reference/PCC checks, with no timing gate."""
    _run_chunked_prefill(
        request,
        mesh_device,
        reference="cpu",
        use_pretrained=bool(os.environ.get(variant.env_var)),
        topology=per_axis_topology(device_params["fabric_config"]),
        chunk_size_global=PREFILL_CHUNK_TOKENS,
        iters_isl=[PREFILL_CHUNK_TOKENS],
        prefill_len=LOUDBOX_PREFIX_TOKENS,
    )


@pytest.mark.parametrize("mesh_device", [LOUDBOX_MESH_SHAPE], ids=["sp4xtp2"], indirect=True)
@pytest.mark.parametrize("device_params", [LOUDBOX_DEVICE_PARAMS], ids=["fabric2d"], indirect=True)
@pytest.mark.skipif(detect_num_devices() != 8, reason="These workloads require an eight-device LoudBox")
@pytest.mark.perf
@skip_with_llk_assert("LLK assertions perturb performance measurements")
@skip_with_watcher("Watcher perturbs performance measurements")
def test_mla_loudbox_compute_perf(mesh_device, device_params, random_weights):
    """Gate the median sum of per-program MAX chip durations from three warmed trace replays."""
    _require_rt_profiler()
    config, weights = random_weights
    assert tuple(mesh_device.shape) == LOUDBOX_MESH_SHAPE
    config, weights = _shape_normalized_config_and_weights(config, weights, LOUDBOX_MESH_SHAPE[1])
    assert config.num_attention_heads == 48
    total = LOUDBOX_PREFIX_TOKENS + LOUDBOX_PERF_CHUNK_TOKENS
    config.max_seq_len = config.max_position_embeddings = total

    mla = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=KimiK3Config.mla_layer_ids()[0],
        seq_len=total,
        sp_axis=0,
        tp_axis=1,
        topology=per_axis_topology(device_params["fabric_config"]),
        is_chunked=True,
        active_seq_len=LOUDBOX_PERF_CHUNK_TOKENS,
        layer_num=1,
        has_indexer=False,
    )
    kv_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=0,
        num_kvpe_cache_layers=1,
    )
    hidden = torch.randn(
        1,
        1,
        LOUDBOX_PERF_CHUNK_TOKENS,
        config.hidden_size,
        generator=torch.Generator().manual_seed(42),
        dtype=torch.bfloat16,
    )
    tt_hidden = ttnn.from_torch(
        hidden,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=LOUDBOX_MESH_SHAPE, dims=(-2, -1)),
    )
    samples = []
    for sample in range(LOUDBOX_PERF_SAMPLES):
        forward = _profile_traced_forward(
            mesh_device, mla, lambda: mla.forward(tt_hidden, {}, kv_cache, actual_start=LOUDBOX_PREFIX_TOKENS)
        )
        ring_ns = sum(
            program["duration_ns"]
            for program in forward["programs"].values()
            if _op_code(program["kernel_sources"]) == "RingJointSDPA"
        )
        assert ring_ns > 0, "realtime profiler returned no ring MLA program"
        samples.append(
            {
                "sample": sample,
                "device_ns": sum(program["duration_ns"] for program in forward["programs"].values()),
                "ring_mla_ns": ring_ns,
                "host_ns": forward["host_duration_ns"],
            }
        )
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_csv(_OUTPUT_DIR / "loudbox_sp4xtp2_compute_samples.csv", index=False)
    measured_ns = median(sample["device_ns"] for sample in samples)
    limit_ns = LOUDBOX_PERF_TARGET_NS * (1 + LOUDBOX_PERF_MARGIN)
    logger.info(
        f"K3 LoudBox SP4xTP2: chunk={LOUDBOX_PERF_CHUNK_TOKENS}, prefix={LOUDBOX_PREFIX_TOKENS}, heads=48; "
        f"device samples (ns)={[sample['device_ns'] for sample in samples]}, median={measured_ns:,.0f} ns; "
        f"ring MLA samples (ns)={[sample['ring_mla_ns'] for sample in samples]}; "
        f"target={LOUDBOX_PERF_TARGET_NS:,} ns, regression limit={limit_ns:,.0f} ns"
    )
    assert 0 < measured_ns <= limit_ns, (
        f"K3 LoudBox device time {measured_ns:,.0f} ns exceeds {limit_ns:,.0f} ns "
        f"(target {LOUDBOX_PERF_TARGET_NS:,} ns + {LOUDBOX_PERF_MARGIN:.0%})"
    )
