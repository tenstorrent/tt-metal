# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How fast is Kimi-K3 prefill at 1, 5, 12 and 24 layers?

Two numbers per depth, because on this mesh they answer different questions.

**Device kernel time** is the sum over programs of that program's critical path across the 32 chips.
The real-time profiler reports one record per program per chip, and the chips run a program
concurrently, so the program costs what its slowest chip costs; programs then execute in sequence,
so the sum is the time the device spends inside kernels. It excludes the dispatch gaps between
programs, which is exactly what makes it the right number to compare against another model's kernels
and the wrong number to quote as throughput.

**Eager wall-clock** is a synchronized multi-iteration measurement of `forward` as the tests
actually call it, host dispatch included. The gap between the two is the host overhead that trace
capture removes, so reporting both says how much there is to win from tracing rather than leaving it
implicit.

Both are reported per token as well, since a depth comparison in milliseconds says little when the
layer counts differ by 24x.

The weight cache matters here more than anywhere: without it each depth spends over an hour reading
routed experts before the first token moves, and a perf sweep re-reads them at every rung.
"""

import cProfile
import io
import os
import pstats
import time
from collections import defaultdict
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import (
    PLACEMENTS,
    SEQ_LEN,
    SP_AXIS,
    TP_AXIS,
    _model_state_dict,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

DEPTHS = [1, 5, 12, 24]
# 1 chunk is 5120 tokens; 11 chunks is 56320 — the "55k" leg. The second is not just a longer run:
# MLA attends over every token cached so far, so chunk N costs more than chunk N-1, and the
# per-chunk curve is the only thing that shows how prefill scales with context.
CHUNK_COUNTS = [1, 11]
ITERATIONS = 5


def _kernel_label(source: str) -> str:
    """A kernel source -> a short name: a file path's last two parts, or an inline kernel's first #define."""
    src = str(source)
    if "\n" not in src and ("/" in src or src.endswith((".cpp", ".hpp", ".h"))):
        parts = [q for q in src.split("/") if q]
        return "/".join(parts[-2:])
    for line in src.splitlines():
        if line.startswith("#define "):
            return "inline:" + line.split()[1]
    return "inline:" + src.strip()[:24].replace("\n", " ")


def _op_breakdown(num_layers, critical_path, sources, device_ms, chip_mean=None):
    """Device time by program kind (programs grouped by their kernels' names): total ms, count, share - and, with
    $K3_PERF_DUMP, every program's (runtime id, ns, kernel names) as json."""
    import json
    import os

    chip_mean = chip_mean or {}
    groups = defaultdict(lambda: [0.0, 0, 0.0])
    rows = []
    for rid, ns in critical_path.items():
        names = tuple(sorted({_kernel_label(k) for k in sources.get(rid, ())}))
        key = " + ".join(names[:4]) + (f" (+{len(names) - 4})" if len(names) > 4 else "")
        groups[key][0] += ns
        groups[key][1] += 1
        groups[key][2] += chip_mean.get(rid, 0.0)
        rows.append({"runtime_id": rid, "ns": ns, "mean_ns": chip_mean.get(rid), "kernels": names})
    mean_ms = sum(chip_mean.values()) / 1e6
    lines = [
        f"  {ns / 1e6:9.2f} ms {ns / 1e6 / max(device_ms, 1e-9) * 100:5.1f}%  (chip mean {mn / 1e6:8.2f})  x{n:4d}  {key}"
        for key, (ns, n, mn) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:40]
    ]
    logger.info(
        f"PERF_OPS L{num_layers:2d}: device {device_ms:.2f} ms (slowest chip a program, summed; the mean over chips "
        f"summed {mean_ms:.2f} ms) by program kind (top 40):\n" + "\n".join(lines)
    )
    path = os.environ.get("K3_PERF_DUMP")
    if path:
        with open(path, "w") as f:
            json.dump(rows, f)


def _traced(model, tokens, kvpe, mesh_device, num_layers, eager_ms, replays=10):
    """One forward captured as a trace (SubDeviceTraceController: TtMoe's shared-expert / dispatch sub-device swaps
    split it into segments, as the runtime's traced prefill does) and replayed: each replay's host wall, synced."""
    import torch

    from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata
    from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

    def meta1(v):  # (as the runtime's: a persistent 1-element uint32 the trace reads at its captured address)
        return ttnn.from_torch(
            torch.tensor([v], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # (the chunk's bounds on the device - the eager path uploads them from the host each forward, illegal in a capture)
    metadata = ChunkMetadata(meta1(0), meta1(0), meta1(SEQ_LEN))

    def fwd():
        out = model.forward(tokens, kvpe_cache=kvpe, metadata=metadata)
        if out is not None:
            ttnn.deallocate(out)

    controller = SubDeviceTraceController(mesh_device)
    model.set_trace_controller(controller)
    capturing = False
    try:
        model.reset_streams()
        fwd()  # (warm with the controller idle: the programs compiled, its sub-device managers created)
        ttnn.synchronize_device(mesh_device)
        controller.begin_capture()
        capturing = True
        fwd()
        controller.end_capture()
        capturing = False
        ms = []
        for _ in range(replays):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            controller.replay()
            ttnn.synchronize_device(mesh_device)
            ms.append((time.perf_counter() - t0) * 1e3)
        rest = sorted(ms[1:]) or ms
        logger.info(
            f"PERF_TRACE L{num_layers}: traced replay first {ms[0]:.2f} ms, median {rest[len(rest) // 2]:.2f} ms, min "
            f"{rest[0]:.2f} ms over {len(ms)} ({controller.num_segments} segments, {controller.trace_bytes() / 2**20:.1f} "
            f"MiB of trace); eager {eager_ms:.2f} ms"
        )
    finally:
        if capturing:
            controller.end_capture()
        controller.release()
        model.set_trace_controller(None)


def _host_breakdown(model, once, mesh_device, num_layers, eager_ms, iters=5):
    """Where an eager forward's host time goes: each layer's parts timed on the host (no syncs - the enqueue cost, as
    the eager wall pays it while the device keeps up), then one forward under cProfile (the top functions by own
    time, the .prof next to K3_PERF_DUMP)."""
    acc = defaultdict(float)

    def timed(fn, key):
        def call(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                acc[key] += time.perf_counter() - t0

        return call

    saved = []
    for blk in model.layers:
        attn = "MLA" if blk.attention.writes_kv else "KDA"
        ffn = "moe" if blk.is_moe else "dense"
        for obj, name, key in (
            (blk, "forward", "layer"),
            (blk, "attn_norm", "attn_norm"),
            (blk.attention, "forward", f"attn {attn}"),
            (blk, "ffn_norm", "ffn_norm"),
            (blk, "_ffn_path", f"ffn {ffn}"),
        ):
            orig = getattr(obj, name)
            had = name in vars(obj)
            saved.append((obj, name, orig, had))
            setattr(obj, name, timed(orig, key))
    try:
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(iters):
            acc["forward"] -= time.perf_counter()
            once()
            acc["forward"] += time.perf_counter()
        t_enq = time.perf_counter() - t0
        ttnn.synchronize_device(mesh_device)
        wall = time.perf_counter() - t0
    finally:
        for obj, name, orig, had in reversed(saved):
            if had:
                setattr(obj, name, orig)
            else:
                delattr(obj, name)
    ms = {k: v / iters * 1e3 for k, v in acc.items()}
    parts = sum(v for k, v in ms.items() if k not in ("forward", "layer"))
    logger.info(
        f"PERF_HOST L{num_layers}: host enqueue {t_enq / iters * 1e3:.1f} ms a forward (wall with the final sync "
        f"{wall / iters * 1e3:.1f}, eager {eager_ms:.1f}); layers {ms['layer']:.1f} ms of it, outside them "
        f"{ms['forward'] - ms['layer']:.1f}; inside layers: "
        + ", ".join(f"{k} {v:.1f}" for k, v in sorted(ms.items(), key=lambda kv: -kv[1]) if k not in ("forward", "layer"))
        + f", the rest of the layer (residual / KV ack) {ms['layer'] - parts:.1f} ms"
    )

    prof = cProfile.Profile()
    ttnn.synchronize_device(mesh_device)
    prof.enable()
    once()
    prof.disable()
    ttnn.synchronize_device(mesh_device)
    dump = os.environ.get("K3_PERF_DUMP")
    if dump:
        prof.dump_stats(str(Path(dump).with_suffix(".host.prof")))
    for key in ("tottime", "cumulative"):
        out = io.StringIO()
        pstats.Stats(prof, stream=out).sort_stats(key).print_stats(45)
        logger.info(f"PERF_HOST_PROF L{num_layers} by {key}:\n{out.getvalue()}")


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize("num_layers", DEPTHS, ids=[f"L{n}" for n in DEPTHS])
@pytest.mark.parametrize("num_chunks", CHUNK_COUNTS, ids=[f"{n}chunk" for n in CHUNK_COUNTS])
# pytest.ini caps every test at 300 s. The deeper rungs build and run a multi-layer model and blow
# straight through that, so the cap kills the run rather than the test reporting a number.
@pytest.mark.timeout(4800)
def test_prefill_cost(mesh_device, device_params, num_layers, num_chunks):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    total_len = SEQ_LEN * num_chunks
    config = kimi_k3_hf_config(max_seq=total_len)
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)

    attn_res = TtAttnRes(
        mesh_device,
        hidden_size=KimiK3Config.EMB_SIZE,
        eps=KimiK3Config.RMS_NORM_EPS,
        tp_axis=TP_AXIS,
        weights=load_attn_res_weights(
            mesh_device,
            load_attn_res_state_dict(checkpoint, num_layers, root),
            None,
            num_layers=num_layers,
            tensor_parallel_axis=TP_AXIS,
            prefix=root,
        ),
    )

    def residual_factory(hidden, block_residual=None):
        # Single-rank test: nothing is inherited, so the second argument is always None.
        return TtAttnResResidual(
            TtAttnResWalk(
                attn_res,
                hidden,
                list(attn_res.weights.pre),
                list(attn_res.weights.post),
                attn_res.weights.output,
                num_layers,
            )
        )

    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        _model_state_dict(checkpoint, num_layers, root, cache),
        num_layers=num_layers,
        seq_len=SEQ_LEN,
        residual_factory=residual_factory,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=total_len,
        # `actual_start` is only accepted by a model built for chunking, and ttMLA asserts the pair
        # match. A single-chunk run passes neither.
        is_chunked=num_chunks > 1,
        weight_cache_path=cache,
        # Dense KV dedup needs the chunked path.
        tp_shard_kv=num_chunks > 1,
    )

    kvpe = None
    if model.schedule.num_mla_layers:
        kvpe = allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=total_len,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            num_users=1,
            tp_axis=TP_AXIS if num_chunks > 1 else None,
        )
    chunks = [
        prepare_prefill_input_tensor(
            trace.token_ids(SEQ_LEN, SEQ_LEN * c)[0].tolist(),
            mesh_device,
            tuple(mesh_device.shape)[SP_AXIS],
            False,
            tuple(mesh_device.shape),
            SP_AXIS,
        )
        for c in range(num_chunks)
    ]

    def once():
        """One whole prefill: every chunk, in order, as a request would arrive."""
        model.reset_streams()
        for index, tokens in enumerate(chunks):
            start = index * SEQ_LEN if num_chunks > 1 else None
            out = model.forward(tokens, kvpe_cache=kvpe, actual_start=start)
            if out is not None:
                ttnn.deallocate(out)

    try:
        for _ in range(3):  # warm the program cache; the first pass compiles
            once()
        ttnn.synchronize_device(mesh_device)

        start = time.perf_counter()
        for _ in range(ITERATIONS):
            once()
        ttnn.synchronize_device(mesh_device)
        eager_ms = (time.perf_counter() - start) / ITERATIONS * 1e3

        if os.environ.get("K3_PERF_HOST") == "1":
            _host_breakdown(model, once, mesh_device, num_layers, eager_ms)
        if os.environ.get("K3_PERF_TRACE") == "1" and num_chunks == 1:
            _traced(model, chunks[0], kvpe, mesh_device, num_layers, eager_ms)

        device_ms = None
        try:
            _, records = profile_realtime_program(mesh_device, once, collect_all=True)
            # One record per (program, chip). The chips run a program concurrently, so the program
            # costs its slowest chip; programs then run in sequence, so the sum is device kernel time.
            critical_path = defaultdict(float)
            sources = {}
            chip_sum, chip_n = defaultdict(float), defaultdict(int)  # (the mean over chips too: other profiles' metric)
            for record in records:
                critical_path[record["runtime_id"]] = max(critical_path[record["runtime_id"]], record["duration_ns"])
                sources.setdefault(record["runtime_id"], record.get("kernel_sources", ()))
                chip_sum[record["runtime_id"]] += record["duration_ns"]
                chip_n[record["runtime_id"]] += 1
            device_ms = sum(critical_path.values()) / 1e6
            programs = len(critical_path)
            chip_mean = {rid: chip_sum[rid] / chip_n[rid] for rid in chip_sum}
            _op_breakdown(num_layers, critical_path, sources, device_ms, chip_mean)
        except RuntimeError as error:
            # Dropping records makes the set partial, and a partial sum under-reports. Say so rather
            # than quote a number that looks like a measurement.
            logger.warning(f"  L{num_layers}: device kernel time unavailable — {error}")
            programs = 0
        # Per-chunk device time, when there is more than one chunk. The total divided by the count is an
        # average, and the average hides the thing worth knowing: MLA attends over every token cached so
        # far, so chunk N should cost more than chunk N-1 while the KDA layers stay flat. Profiling each
        # chunk separately is the only way to see that curve.
        if num_chunks > 1:
            model.reset_streams()
            per_chunk = []
            for index, tokens in enumerate(chunks):

                def one(_tokens=tokens, _index=index):
                    out = model.forward(_tokens, kvpe_cache=kvpe, actual_start=_index * SEQ_LEN)
                    if out is not None:
                        ttnn.deallocate(out)

                try:
                    _, records = profile_realtime_program(mesh_device, one, collect_all=True)
                    path = defaultdict(float)
                    for record in records:
                        path[record["runtime_id"]] = max(path[record["runtime_id"]], record["duration_ns"])
                    per_chunk.append(sum(path.values()) / 1e6)
                except RuntimeError:
                    per_chunk.append(float("nan"))
            logger.info(
                f"PERF L{num_layers:2d} per-chunk device ms: "
                + " ".join(f"{c:.1f}" for c in per_chunk)
                + f"  (first {per_chunk[0]:.1f}, last {per_chunk[-1]:.1f}, "
                + f"growth {per_chunk[-1] - per_chunk[0]:+.1f} ms over {num_chunks} chunks)"
            )

    finally:
        if model.kda_states is not None:
            model.kda_states.deallocate()

    logger.info(
        f"PERF L{num_layers:2d} x {num_chunks:2d}chunk ({total_len:6d} tok): eager {eager_ms:9.2f} ms "
        f"({total_len / eager_ms * 1e3:8.0f} tok/s, {eager_ms / num_layers:7.2f} ms/layer)"
    )
    if device_ms is not None:
        logger.info(
            f"PERF L{num_layers:2d} x {num_chunks:2d}chunk ({total_len:6d} tok): device {device_ms:9.2f} ms "
            f"over {programs} programs ({total_len / device_ms * 1e3:8.0f} tok/s, "
            f"{device_ms / num_layers:7.2f} ms/layer) -- host {eager_ms - device_ms:7.2f} ms "
            f"({(1 - device_ms / eager_ms) * 100:3.0f}%)"
        )
