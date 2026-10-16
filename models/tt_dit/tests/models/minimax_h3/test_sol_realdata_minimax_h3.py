# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn ring SDPA on REAL captured H3 activations (15 s fox prompt): timing at the real block density
and error vs the fp32 torch reference, per tau.

Inputs: `/data/losullivan/sol_attn_captures/15s/full/s{step}_l{layer}.pt` ({q,k,v} [56, 109312, 128] bf16),
one TP shard's 14 heads (SOL_HEADS=0 -> heads 0..13) sharded over SP exactly as the model does
(8 x 13664 rows). Timing follows ~/h3_attn_timing: synchronize_device before/after, 1 compile run then
REPS timed runs; the pre-pass and the ring op are timed separately. The reference is evaluated on
SOL_REF_ROWS query rows chosen to cover the prefix, the video start, a middle shard and the ragged tail.

Env: SOL_CAPTURE (default s10_l12), SOL_TAU (default 1.0,1.5,2.0), SOL_SINK_TOKENS (default 1245),
SOL_REPS (3), SOL_REF=0 to skip the torch reference.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import SolGeometry, SolPrepass
from ....models.transformers.minimax_h3.sol_prepass_fused import SolPrepassFused
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor, from_torch
from .common import GALAXY_RING
from .test_transformer_minimax_h3 import logical_length_tensor

sys.path.insert(0, os.environ.get("SOL_ATTN_WORK", "/home/losullivan/sol_attn_work"))  # /data copy on Slurm nodes
import sol_ref as R  # noqa: E402

# Workload geometry (sol-vid3, 2026-10-16): SOL_GEOM=5s|10s|15s selects the capture set; rows/logical_n/video_start come
# from that capture's stats.jsonl; the chunk sizes default to the module's measured point for that clip length.
GEOM = os.environ.get("SOL_GEOM", "15s")
CAPTURES = Path(f"/data/losullivan/sol_attn_captures/{GEOM}/full")
RESULTS = Path(os.environ.get("SOL_RESULTS", "/home/losullivan/sol_attn_work/bench/results_realdata.jsonl"))
_stats_path = CAPTURES.parent / "stats.jsonl"
if _stats_path.exists():
    _st = json.loads(_stats_path.read_text().splitlines()[0])
    LOGICAL_N, ROWS, VIDEO_START = int(_st["logical_n"]), int(_st["rung"]) // 8, int(_st["video_start"])
else:  # 15 s constants (captures written before stats.jsonl existed)
    LOGICAL_N, ROWS, VIDEO_START = 109101, 13664, 1245
_DEFAULT_CHUNKS = {"5s": (320, 384), "10s": (256, 512), "15s": (256, 512)}[GEOM]
Q_CHUNK = int(os.environ.get("SOL_Q_CHUNK", str(_DEFAULT_CHUNKS[0])))
K_CHUNK = int(os.environ.get("SOL_K_CHUNK", str(_DEFAULT_CHUNKS[1])))


_SOL_MODE_LINE = "RingJointSDPA sol mode resolved: "


def _sol_mode_guard(capfd, mode, state):
    """Fail when a requested Sol mode did not reach the kernels (lead's request after the docstring-gate incident).

    The program factory logs "RingJointSDPA sol mode resolved: <mode> ..." once per program creation, i.e. exactly when
    the kernels are created with that mode's named compile-time args. Read it back from the captured stderr after the
    first sparse launch; the captured text is re-emitted so job logs keep every line.
    """
    if state.get("checked"):
        return
    state["checked"] = True
    out, err = capfd.readouterr()
    with capfd.disabled():
        sys.stdout.write(out)
        sys.stderr.write(err)
        sys.stdout.flush()
        sys.stderr.flush()
    lines = [l for l in (out + err).splitlines() if _SOL_MODE_LINE in l]
    assert (
        lines
    ), "no 'RingJointSDPA sol mode resolved' line captured: the factory log is missing (old build) or filtered"
    resolved = {l.split(_SOL_MODE_LINE, 1)[1].split()[0] for l in lines}
    expected = "gather" if mode in (None, "auto") else mode
    assert resolved == {
        expected
    }, f"requested sol mode {expected!r} but the op resolved {sorted(resolved)}: {lines[-1]}"
    logger.info(f"[sol] mode guard: op resolved {expected!r} ({len(lines)} program creation(s))")


def _sol_mode_kwargs(mode):
    """Pass sol_mode only when the installed binding has it (older builds infer gather/chunk from the mask width).

    The op's docstring never lists keyword names (a check on it is always False), so probe the raw nanobind function:
    its argument error text carries the full signature. "compact" cannot be inferred, so a binding without the keyword
    is an error there instead of a silent gather run (that silently happened once; see DESIGN_BLOCK_COMPACT.md).
    """
    if mode in (None, "auto"):
        return {}
    raw = ttnn._ttnn.operations.transformer.ring_joint_scaled_dot_product_attention
    try:
        raw()
        has = False
    except TypeError as e:
        has = "sol_mode" in str(e)
    if not has:
        if mode == "compact":
            raise RuntimeError(
                "the installed ring_joint_scaled_dot_product_attention binding has no sol_mode keyword: "
                "a 'compact' run would silently execute in gather mode"
            )
        logger.warning(f"binding has no sol_mode keyword; mode {mode!r} is inferred from the mask width")
        return {}
    logger.info(f"[sol] passing sol_mode={mode!r} to the ring op")
    return {"sol_mode": mode}


def _timed(fn, mesh_device, reps):
    out = None
    times = []
    for _ in range(reps):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e3)
    return times, out


@GALAXY_RING
def test_sol_realdata(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, capfd):
    S = tuple(mesh_device.shape)[sp_axis]
    assert S == 8, "captures are for SP=8"
    H, D = 14, 128
    scale = D**-0.5
    cap = os.environ.get("SOL_CAPTURE", "s10_l12")
    taus = [float(t) for t in os.environ.get("SOL_TAU", "1.0,1.5,2.0").split(",")]
    sink_tokens = int(os.environ.get("SOL_SINK_TOKENS", str(VIDEO_START)))
    reps = int(os.environ.get("SOL_REPS", "3"))
    head0 = int(os.environ.get("SOL_HEADS", "0")) * H
    do_ref = os.environ.get("SOL_REF", "1") == "1"
    mode = os.environ.get("SOL_MODE", "gather")
    dense_prefix = os.environ.get("SOL_DENSE_PREFIX", "1") == "1"
    gen_bias = os.environ.get("SOL_GEN_BIAS", "0") == "1"  # ring op generates the pooled-column bias on device
    fused = (
        os.environ.get("SOL_FUSED", "0") == "1"
    )  # sol-video: fused generic_op pre-pass (SolPrepassFused) instead of the ttnn ops

    blob = torch.load(CAPTURES / f"{cap}.pt", map_location="cpu", mmap=True)
    assert blob["q"].shape == (56, S * ROWS, D), blob["q"].shape

    def shard(x):  # [H, S*rows, D] -> [S, H, rows, D]
        return x[head0 : head0 + H].view(H, S, ROWS, D).permute(1, 0, 2, 3).contiguous()

    q, k, v = (shard(blob[n]).bfloat16() for n in ("q", "k", "v"))
    geo = SolGeometry(shards=S, rows=ROWS, logical_n=LOGICAL_N, q_chunk=Q_CHUNK, k_chunk=K_CHUNK)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    def upload(x):
        return from_torch(
            x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None], layout=ttnn.TILE_LAYOUT
        )

    tq, tk, tv = upload(q), upload(k), upload(v)
    grid = mesh_device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(
            int(os.environ.get("SOL_GRID_X", grid.x - 1)), int(os.environ.get("SOL_GRID_Y", grid.y))
        ),
        q_chunk_size=Q_CHUNK,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    dummy_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)
    kshape = tq.shape
    ln = logical_length_tensor(mesh_device, LOGICAL_N)

    mode_guard_state = {}

    def run_ring(**sol_kwargs):
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            dummy_joint,
            dummy_joint,
            dummy_joint,
            persistent_output_buffer_k=ccl_manager.get_ag_ping_pong_buffer(kshape, 2, sp_axis, dtype=ttnn.bfloat16),
            persistent_output_buffer_v=ccl_manager.get_ag_ping_pong_buffer(kshape, 2, sp_axis, dtype=ttnn.bfloat16),
            joint_strategy="rear",
            logical_n=ln,
            program_config=program_config,
            compute_kernel_config=ckc,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=ccl_manager.ccl_sub_device_id,
            ccl_core_grid_offset=(grid.x - 1, 0),
            use_column_major_ccl=True,
            **sol_kwargs,
        )
        if sol_kwargs:
            _sol_mode_guard(capfd, mode, mode_guard_state)
        return out

    def to_host(t):
        host = ttnn.to_torch(
            t,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=[1, 2] if sp_axis == 1 else [2, 1], mesh_shape=tuple(mesh_device.shape)
            ),
        )
        return host[0, :H].float()  # [H, S*rows, D]

    # reference rows: prefix (text+cond), video start, a middle shard, the ragged tail of the last shard.
    # SOL_REF_ROWS=all evaluates every query row (hours of CPU); SOL_REF_ROWS=a:b,c:d picks ranges.
    ref_spec = os.environ.get("SOL_REF_ROWS", "")
    if ref_spec == "all":
        ref_ranges = [(a, min(a + 4096, S * ROWS)) for a in range(0, S * ROWS, 4096)]
    elif ref_spec:
        ref_ranges = [tuple(int(x) for x in r.split(":")) for r in ref_spec.split(",")]
    else:
        ref_ranges = [(0, 1024), (1280, 2304), (3 * ROWS + 5120, 3 * ROWS + 6144), (S * ROWS - 1280, S * ROWS)]
    ref_rows = torch.cat([torch.arange(a, b) for a, b in ref_ranges])
    ref_rows = ref_rows[ref_rows < LOGICAL_N]
    # named regions for the per-region breakdown (token layout of the 15 s fox prompt: text 39, prefix 1245)
    regions = {
        "text": (0, 39),
        "cond_prefix": (39, sink_tokens),
        "video_head": (sink_tokens, sink_tokens + 2048),
        "video_mid": (3 * ROWS, 5 * ROWS),
        "video_tail": (S * ROWS - 4096, LOGICAL_N),
    }

    def region_stats(ref, test):
        out = {}
        for name, (a, b) in regions.items():
            rows = ref_rows[(ref_rows >= a) & (ref_rows < b)]
            if rows.numel() == 0:
                continue
            e = R.error_stats(ref[:, rows], test[:, rows])
            cos = torch.nn.functional.cosine_similarity(ref[:, rows].float(), test[:, rows].float(), dim=-1)
            e["rows"] = int(rows.numel())
            e["row_cos_p1"] = float(cos.flatten().kthvalue(max(1, int(0.01 * cos.numel()))).values)
            e["frac_rows_cos_lt_0.9"] = float((cos < 0.9).float().mean())
            out[name] = e
        return out

    def head_stats(ref, test):
        a, b = ref[:, ref_rows].float(), test[:, ref_rows].float()
        return [round(float((b[h] - a[h]).norm() / a[h].norm().clamp_min(1e-12)), 4) for h in range(a.shape[0])]

    records = []
    # ---- dense
    _, _ = _timed(lambda: run_ring(), mesh_device, 1)
    dense_ms, dense_out = _timed(lambda: run_ring(), mesh_device, reps)
    dense_host = to_host(dense_out)
    logger.info(f"[{cap}] dense op {' / '.join(f'{m:.2f}' for m in dense_ms)} ms")
    rgeo = R.Geometry(
        shards=S,
        rows=ROWS,
        logical_n=LOGICAL_N,
        q_chunk=Q_CHUNK,
        k_chunk=64 if mode in ("gather", "compact") else K_CHUNK,
    )
    qg = q.permute(1, 0, 2, 3).reshape(H, S * ROWS, D).float()
    kg = k.permute(1, 0, 2, 3).reshape(H, S * ROWS, D).float()
    vg = v.permute(1, 0, 2, 3).reshape(H, S * ROWS, D).float()
    ref_dense = None
    if do_ref:
        ref_dense = torch.zeros_like(qg)
        for a, b in ref_ranges:
            ref_dense += R.dense_attention(qg, kg, vg, rgeo, rows=slice(a, b))
        e_dense = R.error_stats(ref_dense[:, ref_rows], dense_host[:, ref_rows])
        logger.info(f"[{cap}] dense kernel vs fp32 dense: {e_dense}")
        dense_regions = region_stats(ref_dense, dense_host)
        logger.info(f"[{cap}] dense kernel vs fp32 dense per region: {dense_regions}")
    else:
        e_dense = None
        dense_regions = None
    records.append(
        {
            "capture": cap,
            "mode": "dense",
            "tau": None,
            "op_ms": dense_ms,
            "prepass_ms": [],
            "density": None,
            "err_vs_dense_ref": e_dense,
            "regions_vs_dense_ref": dense_regions,
            "ref_ranges": ref_ranges,
        }
    )

    # ---- sparse per tau
    for tau in taus:
        pre = (SolPrepassFused if fused else SolPrepass)(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            sp_axis=sp_axis,
            tau=tau,
            mode=mode,
            dense_prefix_queries=dense_prefix,
            emit_bias=not gen_bias,
        )
        pre.prepare(geo, sink_tokens, scale)
        sol = pre(tq, tk, tv, scale=scale, report_density=True)
        density = pre.last_density

        def sol_kwargs(d):
            kw = dict(
                sol_block_mask=d["block_mask"],
                sol_pooled_k=d["pooled_k"],
                sol_pooled_v=d["pooled_v"],
                **_sol_mode_kwargs(mode),
            )
            if d["bias"] is not None:
                kw["sol_bias"] = d["bias"]
            return kw

        _, _ = _timed(lambda: run_ring(**sol_kwargs(sol)), mesh_device, 1)
        pre_ms, sol2 = _timed(lambda: pre(tq, tk, tv, scale=scale, report_density=False), mesh_device, reps)
        op_ms, sol_out = _timed(lambda: run_ring(**sol_kwargs(sol2)), mesh_device, reps)
        sol_host = to_host(sol_out)
        e_sol_vs_dense_kernel = R.error_stats(dense_host[:, ref_rows], sol_host[:, ref_rows])
        rec = {
            "capture": cap,
            "mode": "sol",
            "prepass": "fused" if fused else "ttnn",
            "sol_mode": mode,
            "dense_prefix": dense_prefix,
            "gen_bias": gen_bias,
            "tau": tau,
            "op_ms": op_ms,
            "prepass_ms": pre_ms,
            "density": density,
            "err_vs_dense_kernel": e_sol_vs_dense_kernel,
        }
        logger.info(
            f"[{cap}] {mode}{'' if dense_prefix else '/dp0'}{'/genbias' if gen_bias else ''} tau={tau}: kept density {density:.4f}; prepass {' / '.join(f'{m:.2f}' for m in pre_ms)} ms; "
            f"op {' / '.join(f'{m:.2f}' for m in op_ms)} ms (dense {min(dense_ms):.2f}); sol kernel vs dense kernel {e_sol_vs_dense_kernel}"
        )
        if do_ref:
            routing = R.route(
                qg,
                kg,
                rgeo,
                tau=tau,
                scale=scale,
                band=1,
                sink_tokens=sink_tokens,
                tt_kernel_rules=True,
                dense_prefix_queries=dense_prefix,
                full_blocks_only=(mode in ("gather", "compact")),
                even_up_bpc=(K_CHUNK // 64)
                if (
                    mode == "compact"
                    and (os.environ.get("TT_SOL_EVENUP", "0") == "1" or os.environ.get("TT_SOL_EVENUP_ALL", "0") == "1")
                )
                else 0,
            )
            if mode == "compact" and os.environ.get("TT_SOL_Q_PAIRS", "0") == "1":
                n_cores = int(os.environ.get("SOL_GRID_X", grid.x - 1)) * int(os.environ.get("SOL_GRID_Y", grid.y))
                routing = dict(routing)
                routing["chunk_mask"] = R.pair_union_rows(routing["chunk_mask"], -(-ROWS // Q_CHUNK), S, n_cores)
            ref_sol = torch.zeros_like(qg)
            for a, b in ref_ranges:
                ref_sol += R.sol_attention(qg, kg, vg, rgeo, routing, granularity="chunk", rows=slice(a, b))
            rec["ref_density"] = float(routing["chunk_mask"].float().mean())
            rec["err_kernel_vs_sol_ref"] = R.error_stats(ref_sol[:, ref_rows], sol_host[:, ref_rows])
            rec["err_sol_ref_vs_dense_ref"] = R.error_stats(ref_dense[:, ref_rows], ref_sol[:, ref_rows])
            rec["err_kernel_vs_dense_ref"] = R.error_stats(ref_dense[:, ref_rows], sol_host[:, ref_rows])
            logger.info(
                f"[{cap}] tau={tau}: ref density {rec['ref_density']:.4f}; kernel vs sol ref {rec['err_kernel_vs_sol_ref']}; "
                f"sol ref vs dense ref {rec['err_sol_ref_vs_dense_ref']}; kernel vs dense ref {rec['err_kernel_vs_dense_ref']}"
            )
            rec["regions_kernel_vs_sol_ref"] = region_stats(ref_sol, sol_host)
            rec["regions_sol_ref_vs_dense_ref"] = region_stats(ref_dense, ref_sol)
            rec["heads_kernel_vs_sol_ref"] = head_stats(ref_sol, sol_host)
            rec["heads_sol_ref_vs_dense_ref"] = head_stats(ref_dense, ref_sol)
            for name in rec["regions_kernel_vs_sol_ref"]:
                k_ = rec["regions_kernel_vs_sol_ref"][name]
                a_ = rec["regions_sol_ref_vs_dense_ref"][name]
                logger.info(
                    f"[{cap}] tau={tau} region {name:>11} ({k_['rows']} rows): kernel vs sol ref relL2 {k_['rel_l2']:.4f} "
                    f"(cos p1 {k_['row_cos_p1']:.3f}); sol vs dense relL2 {a_['rel_l2']:.4f} (cos p1 {a_['row_cos_p1']:.3f}, "
                    f"rows cos<0.9 {a_['frac_rows_cos_lt_0.9']:.3%})"
                )
            logger.info(f"[{cap}] tau={tau} per-head relL2 kernel vs sol ref {rec['heads_kernel_vs_sol_ref']}")
            logger.info(f"[{cap}] tau={tau} per-head relL2 sol ref vs dense {rec['heads_sol_ref_vs_dense_ref']}")
        records.append(rec)
        for t in sol.values():
            if t is not None:
                ttnn.deallocate(t)
        for t in sol2.values():
            if t is not None:
                ttnn.deallocate(t)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"wrote {len(records)} records to {RESULTS}")
