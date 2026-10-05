# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Baseline of the current TP GDN prefill layer (qwen36 ``TPGatedDeltaNet`` -> ``ttnn.transformer.chunk_gated_delta_rule``)
on a 1x4 Blackhole mesh (TP4, FABRIC_1D), production configuration (``Qwen36ModelArgs`` defaults, op dispatch left to
its cost model).

* ``test_gdn_baseline_accuracy`` (R7 chained chunks, R8 ragged last chunk, R9 accuracy, R10 determinism, R11 trace):
  per case (``cases.py``) the chunks run eagerly twice with the layer's carried state (``_stable_state``); every
  chunk's output, recurrent state and conv carry is gated against the FP32 CPU reference (``accuracy.py``) and the
  second pass must equal the first bit for bit. ``chain`` cases then capture one chunk as a trace and replay all
  three chunks into the persistent input buffer (replays 1 and 2 see inputs that differ from capture); each replay
  must equal the eager pass bit for bit and is gated against the reference. ``ragged`` cases run eager only: the
  masked ``valid_len`` path uploads a host mask (``fused_chunk.py``), which trace capture rejects.
* ``test_gdn_baseline_perf`` (R12): synthetic weights, no reference; median of five synchronized samples of 100
  back-to-back trace replays of one chunk (the KDA layer-perf method, which uses 10 replays per sample), for the
  default dispatch and forced phased.
* ``test_gdn_baseline_profile`` (R14 driver, run under ``run_safe_pytest.sh --profile``): one compile chunk, two warm
  eager chunks and three trace replays, separated by Tracy signposts.

Mesh: the fixture opens the LoudBox 2x4 system mesh and the layer runs on its 1x4 row-0 submesh. A standalone 1x4
FABRIC_1D mesh on LoudBox fails fabric init (device 0 ethernet routers toward the unopened row never handshake), so
fabric must be initialized on all eight chips; the layer's TP4 collectives stay within the 1x4 row.

CPU preparation first (fills the reference cache; the accuracy test fails on a miss):
    python -m models.demos.blackhole.qwen36.tests.gdn_baseline.prepare --case <case name>
Results are printed as ``GDN_BASELINE_*=<json>`` lines and written to ``generated/gdn_baseline/``.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.gdn_baseline import cases as gc
from models.demos.blackhole.qwen36.tests.gdn_baseline.accuracy import measure, per_head_rel_rmse
from models.demos.blackhole.qwen36.tests.test_factory import shard_to_device, tp_composer
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical

SYSTEM_MESH_SHAPE = (2, 4)
MESH_SHAPE = (1, 4)
TIMING_SAMPLES = 5
# 100 back-to-back replays per sample: a ~1 ms layer timed over 10 replays (the KDA layer-perf setting, ~9.5 ms
# layers) showed up to 62% sample spread from host jitter on a loaded host.
TIMING_REPETITIONS = 100
RESULTS_DIR = gc.REPOSITORY_ROOT / "generated" / "gdn_baseline"

_DEVICE_PARAMS = pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
            "trace_region_size": 268435456,
        }
    ],
    indirect=True,
)
_MESH = pytest.mark.parametrize("mesh_device", [pytest.param(SYSTEM_MESH_SHAPE, id="1x4of2x4")], indirect=True)


def _tp_mesh(system_mesh):
    """The 1x4 TP mesh: row 0 of the system mesh (fabric is initialized on the whole system mesh)."""
    return system_mesh.create_submesh(ttnn.MeshShape(*MESH_SHAPE), offset=ttnn.MeshCoordinate(0, 0))


def _emit(kind: str, name: str, result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{kind}-{name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=1, default=float))
    print(f"GDN_BASELINE_{kind.upper()}=" + json.dumps(result, default=float))
    logger.info(f"GDN baseline {kind} {name}: written {path}")


def _environment(mesh, args) -> dict:
    grid = mesh.compute_with_storage_grid_size()
    return {
        "mesh_shape": list(mesh.shape),
        "system_mesh_shape": list(SYSTEM_MESH_SHAPE),
        "fabric_config": ttnn.get_fabric_config().name,
        "compute_with_storage_grid_size": [grid.x, grid.y],
        "gdn_program_config": repr(args.gdn_program_config),
        "conv_impl": os.environ.get("QWEN_GDN_CONV", "kda"),
        "heads_per_chip": {"k": args.gdn_nk_tp, "v": args.gdn_nv_tp},
        "host_load_average_1_5_15": list(os.getloadavg()),
        "host_cpus": os.cpu_count(),
    }


def _build_layer(mesh, model: str, state_dict: dict, tokens: int, program_config=None):
    """Production TP GDN layer with the carried-state (chunk-outer) mode the model's chunked prefill uses."""
    from models.tt_transformers.tt.ccl import TT_CCL

    os.environ["HF_MODEL"] = str(gc.model_dir(model))
    args = Qwen36ModelArgs(mesh, max_batch_size=1, max_seq_len=tokens * gc.CHAINED_CHUNKS)
    if program_config is not None:
        args.gdn_program_config = program_config
    start = time.perf_counter()
    tw = load_gdn_weights_tp(mesh, state_dict, args)
    ttnn.synchronize_device(mesh)
    logger.info(f"GDN weights converted and uploaded in {time.perf_counter() - start:.2f} s")
    gdn = TPGatedDeltaNet(mesh, args, tw, TT_CCL(mesh))
    gdn._stable_state = True
    gdn.reset_state()
    return args, gdn


class _Chunks:
    """A persistent K-sharded input buffer (address baked into a trace) and the host chunks copied into it."""

    def __init__(self, mesh, x: torch.Tensor, tokens: int):
        self.mesh = mesh
        self.chunks = [x[c * tokens : (c + 1) * tokens][None, None] for c in range(x.shape[0] // tokens)]
        self.buffer = shard_to_device(mesh, self.chunks[0], dim=-1)

    def load(self, c: int) -> None:
        src = shard_to_device(self.mesh, self.chunks[c], dim=-1)
        ttnn.copy(src, self.buffer)
        ttnn.deallocate(src)


def _read_chunk(mesh, gdn, out, valid: int) -> dict:
    hidden = gdn.args.dim
    output = ttnn.to_torch(out, mesh_composer=tp_composer(mesh)).reshape(-1, hidden)[:valid]
    recurrent = ttnn.to_torch(gdn.rec_state, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))[0]
    conv = ttnn.to_torch(gdn.conv_carry, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))[0]
    return {"output": output, "recurrent": recurrent, "conv": conv}


def _eager_pass(mesh, args, gdn, chunks: _Chunks, case: gc.GdnCase) -> list[dict]:
    gdn.reset_state_inplace()
    results = []
    for c, valid in enumerate(case.valid_lengths):
        chunks.load(c)
        out = gdn.forward_prefill(
            chunks.buffer, chunk_size=args.gdn_chunk_size, valid_len=None if valid == case.tokens else valid
        )
        results.append(_read_chunk(mesh, gdn, out, valid))
        ttnn.deallocate(out)
    return results


def _accuracy(case: gc.GdnCase, reference: list[dict], device: list[dict], label: str, tp: int) -> dict:
    shape = gc.gdn_shape(case.model)
    tensors = []
    for c, (ref, dev) in enumerate(zip(reference, device)):
        tensors.append(measure(ref["output"], dev["output"], f"{label}.chunk{c}.output"))
        tensors.append(measure(ref["recurrent"], dev["recurrent"], f"{label}.chunk{c}.recurrent_state"))
        ref_conv = gc.per_device_conv_columns(ref["conv"], shape, tp)
        tensors.append(measure(ref_conv, dev["conv"], f"{label}.chunk{c}.conv_state"))
    tensors.append(
        measure(
            torch.cat([r["output"] for r in reference]),
            torch.cat([d["output"] for d in device]),
            f"{label}.sequence.output",
        )
    )
    heads = per_head_rel_rmse(reference[-1]["recurrent"], device[-1]["recurrent"])
    worst = sorted(range(len(heads)), key=lambda h: -heads[h])[:5]
    return {
        "tensors": tensors,
        "final_state_worst_heads": [{"head": h, "rel_rmse": heads[h]} for h in worst],
        "final_state_ref_norm_of_worst": [float(reference[-1]["recurrent"][h].norm()) for h in worst],
    }


def _bit_identical(first: list[dict], second: list[dict], label: str) -> list[str]:
    failures = []
    for c, (a, b) in enumerate(zip(first, second)):
        for key in ("output", "recurrent", "conv"):
            try:
                assert_bit_identical(a[key], b[key], name=f"{label}.chunk{c}.{key}")
            except AssertionError as error:
                differing = int((a[key] != b[key]).sum()) if a[key].shape == b[key].shape else -1
                failures.append(f"{str(error).splitlines()[0]} ({differing} elements differ)")
    return failures


def _decay_stats(case: gc.GdnCase, state_dict: dict, x: torch.Tensor) -> dict:
    """Per-chunk cumulative decay |G_last| of the case's input (what strong-decay heads reach), host only."""
    import torch.nn.functional as F

    a = x.float() @ state_dict["linear_attn.in_proj_a.weight"].float().T
    g = -state_dict["linear_attn.A_log"].float().exp() * F.softplus(a + state_dict["linear_attn.dt_bias"].float())
    valid = torch.cat([g[c * case.tokens : c * case.tokens + n] for c, n in enumerate(case.valid_lengths)])
    usable = valid.shape[0] // 32 * 32
    g_last = valid[:usable].reshape(-1, 32, g.shape[-1]).sum(1).abs()
    return {"max_abs_G_last_per_32": float(g_last.max()), "max_abs_g_per_token": float(valid.abs().max())}


@torch.no_grad()
@_DEVICE_PARAMS
@_MESH
@pytest.mark.parametrize("case_name", list(gc.CASES))
def test_gdn_baseline_accuracy(mesh_device, case_name, reset_seeds, ensure_gc):
    mesh = _tp_mesh(mesh_device)
    case = gc.CASES[case_name]
    state_dict = gc.load_layer_weights(case.model, case.weights)
    entry = gc.load_case(case, state_dict)
    reference = entry["chunks"]
    args, gdn = _build_layer(mesh, case.model, state_dict, case.tokens)
    tp = mesh.get_num_devices()
    chunks = _Chunks(mesh, entry["inputs"], case.tokens)
    result = {"case": case_name, "environment": _environment(mesh, args), "valid_lengths": case.valid_lengths}
    result["decay"] = _decay_stats(case, state_dict, entry["inputs"])

    start = time.perf_counter()
    first = _eager_pass(mesh, args, gdn, chunks, case)
    result["first_eager_pass_s"] = time.perf_counter() - start
    second = _eager_pass(mesh, args, gdn, chunks, case)
    result["eager"] = _accuracy(case, reference, first, "eager", tp)
    result["determinism_failures"] = _bit_identical(first, second, "eager_repeat")

    if case.plan == "chain":
        gdn.reset_state_inplace()
        chunks.load(0)
        ttnn.synchronize_device(mesh)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = gdn.forward_prefill(chunks.buffer, chunk_size=args.gdn_chunk_size)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        replays = []
        try:
            for c, valid in enumerate(case.valid_lengths):
                chunks.load(c)
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                replays.append(_read_chunk(mesh, gdn, out, valid))
        finally:
            ttnn.release_trace(mesh, trace_id)
        result["trace"] = _accuracy(case, reference, replays, "trace", tp)
        result["trace_vs_eager_failures"] = _bit_identical(first, replays, "trace_vs_eager")
    else:
        result["trace"] = "not run: the masked valid_len path uploads a host mask, rejected under trace capture"

    tensors = result["eager"]["tensors"] + (result["trace"]["tensors"] if isinstance(result["trace"], dict) else [])
    accuracy_failures = [f for t in tensors for f in t["failures"]]
    for t in tensors:
        logger.info(
            f"{t['name']}: pcc={t.get('pcc', float('nan')):.6f} rel_rmse={t.get('rel_rmse', float('nan')):.3e} "
            f"rel_linf={t.get('rel_linf', float('nan')):.3e} norm_ratio={t.get('norm_ratio', float('nan')):.4f} "
            f"{'PASS' if t['passed'] else 'FAIL'}"
        )
    _emit("accuracy", case_name, result)
    problems = accuracy_failures + result["determinism_failures"] + result.get("trace_vs_eager_failures", [])
    assert not problems, f"{case_name}: {len(problems)} failures:\n" + "\n".join(problems)


def _synthetic_layer(mesh, model: str, tokens: int, program_config=None):
    state_dict = gc.load_layer_weights(model, "synthetic")
    args, gdn = _build_layer(mesh, model, state_dict, tokens, program_config)
    generator = torch.Generator().manual_seed(gc.RANDN_SEED)
    x = torch.randn(gc.CHAINED_CHUNKS * tokens, args.dim, generator=generator).to(torch.bfloat16)
    return args, gdn, _Chunks(mesh, x, tokens)


_PERF_PARAMS = [pytest.param(model, tokens, id=f"{model}-T{tokens}") for model in gc.MODELS for tokens in gc.TOKENS]


@torch.no_grad()
@_DEVICE_PARAMS
@_MESH
@pytest.mark.parametrize("dispatch", ["default", "phased"])
@pytest.mark.parametrize("model, tokens", _PERF_PARAMS)
def test_gdn_baseline_perf(mesh_device, model, tokens, dispatch, reset_seeds, ensure_gc):
    mesh = _tp_mesh(mesh_device)
    program_config = ttnn.ChunkGdnPhasedProgramConfig() if dispatch == "phased" else None
    args, gdn, chunks = _synthetic_layer(mesh, model, tokens, program_config)
    # Two eager chunks establish persistent resources (AGMM gather buffer, program cache) before capture.
    for c in range(2):
        chunks.load(c)
        ttnn.deallocate(gdn.forward_prefill(chunks.buffer, chunk_size=args.gdn_chunk_size))
    gdn.reset_state_inplace()
    chunks.load(0)
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = gdn.forward_prefill(chunks.buffer, chunk_size=args.gdn_chunk_size)
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        samples_ms = []
        for _ in range(TIMING_SAMPLES):
            start = time.perf_counter()
            for _ in range(TIMING_REPETITIONS):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples_ms.append((time.perf_counter() - start) * 1e3 / TIMING_REPETITIONS)
        final = ttnn.to_torch(out, mesh_composer=tp_composer(mesh)).float()
    finally:
        ttnn.release_trace(mesh, trace_id)
    assert torch.isfinite(final).all(), "trace replay produced non-finite output"
    result = {
        "model": model,
        "tokens": tokens,
        "dispatch": dispatch,
        "weights": "synthetic (random_gdn_state_dict seed 0)",
        "environment": _environment(mesh, args),
        "repetitions": TIMING_REPETITIONS,
        "trace_wall_samples_ms": samples_ms,
        "median_trace_wall_ms": statistics.median(samples_ms),
        "min_trace_wall_ms": min(samples_ms),
        "max_trace_wall_ms": max(samples_ms),
    }
    _emit("perf", f"{model}-T{tokens}-{dispatch}", result)


@torch.no_grad()
@_DEVICE_PARAMS
@_MESH
@pytest.mark.parametrize("model, tokens", _PERF_PARAMS)
def test_gdn_baseline_profile(mesh_device, model, tokens, reset_seeds, ensure_gc):
    from tracy import signpost

    mesh = _tp_mesh(mesh_device)
    args, gdn, chunks = _synthetic_layer(mesh, model, tokens)
    signpost("gdn_compile")
    chunks.load(0)
    ttnn.deallocate(gdn.forward_prefill(chunks.buffer, chunk_size=args.gdn_chunk_size))
    ttnn.synchronize_device(mesh)
    for c in (1, 2):
        chunks.load(c)
        ttnn.synchronize_device(mesh)
        signpost(f"gdn_eager_chunk{c}")
        ttnn.deallocate(gdn.forward_prefill(chunks.buffer, chunk_size=args.gdn_chunk_size))
        ttnn.synchronize_device(mesh)
    gdn.reset_state_inplace()
    chunks.load(0)
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = gdn.forward_prefill(chunks.buffer, chunk_size=args.gdn_chunk_size)
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    try:
        for c in range(gc.CHAINED_CHUNKS):
            chunks.load(c)
            ttnn.synchronize_device(mesh)
            signpost(f"gdn_trace_replay{c}")
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        signpost("gdn_end")
        ttnn.ReadDeviceProfiler(mesh)
    finally:
        ttnn.release_trace(mesh, trace_id)
    ttnn.deallocate(out)
    _emit("profile", f"{model}-T{tokens}", {"model": model, "tokens": tokens, "environment": _environment(mesh, args)})
