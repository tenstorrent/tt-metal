# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-call attention benchmark of the block-sparse ring attention (`ttnn.transformer.sparse_ring_sdpa`) on a 4x8
Blackhole Galaxy (SP=8 on the ring axis x TP=4 on heads), on SGLang-format inputs.

Backends (SGLang names):
  fa                 the dense ring op (ring_joint_scaled_dot_product_attention), the baseline.
  sol_attn           Sol pre-pass on device (block mask + pooled K / V) + sparse_ring_sdpa, pooled route.
  video_sparse_attn  VSAPrepass on device (block top-k mask + coarse branch) + sparse_ring_sdpa kept-only, on the
                     sequence in VSA's balanced 4x4x4 tile layout.
  cube_sparse_attn   the same block top-k routing without the coarse branch (top-k ratio of the video cubes).

Inputs: --inputs FILE|DIR ... (safetensors with q / k / v [B, T, H, D] bf16, see README.md) or --synthetic
--seconds 5|10|15 (seeded randn at the MiniMax-H3 768p t2va shapes; --save-inputs writes them in the same format).
Each (input, backend) appends one JSONL record (README.md, "Result schema") to --output. The sparse backends' time
is scoring + attention, both on device, in one trace; each part is in `extra`.

Run from the tt-metal root:
  python models/tt_dit/benchmarks/sparse_ring_attention/bench_tt.py --synthetic --seconds 5 \
      --attention-backend fa,sol_attn,video_sparse_attn,cube_sparse_attn --output results_tt.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import glob
import json
import math
import os
import subprocess
import sys
import time

import torch
from loguru import logger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, ROOT)

import ttnn  # noqa: E402
from models.tt_dit.models.transformers.minimax_h3.sol_prepass import SolGeometry, SolPrepass  # noqa: E402
from models.tt_dit.models.transformers.minimax_h3.vsa_prepass import VSAPrepass, vsa_routing_request  # noqa: E402
from models.tt_dit.parallel.manager import CCLManager  # noqa: E402
from models.tt_dit.pipelines.minimax_h3.packing import (  # noqa: E402
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from models.tt_dit.pipelines.minimax_h3.policy import align_num_frames  # noqa: E402
from models.tt_dit.pipelines.minimax_h3.vsa_geometry import (  # noqa: E402
    VSA_TILE_TOKENS,
    build_vsa_geometry,
    chop_prefix_segments,
    video_tile_partition_indices,
    video_tile_valid_counts,
    vsa_min_padded_len,
)
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (  # noqa: E402
    MESH_CONFIG,
    call_sdpa,
    close_ring_joint_sdpa_runtime,
    open_ring_joint_sdpa_runtime,
)
from tests.nightly.blackhole.sdpa.test_sparse_ring_sdpa import (  # noqa: E402
    BLOCK,
    HEADS_PER_DEVICE,
    Case,
    D,
    Geometry,
    _density,
    _fit_scale,
    _pcc,
    _rel_l2,
    dense_reference,
    pool,
    round_significand,
    sparse_reference,
)
from tests.nightly.blackhole.sdpa.test_sparse_ring_sdpa_perf import DENSE_CHUNKS, TRACE_REGION_SIZE  # noqa: E402

BACKENDS = ("fa", "sol_attn", "video_sparse_attn", "cube_sparse_attn")
TOPK_BACKENDS = ("video_sparse_attn", "cube_sparse_attn")  # VSAPrepass routing on VSA's tile layout
WORKING_ROWS = {5: 4736, 10: 9184, 15: 13664}  # rows per device on SP=8: 37,888 / 73,472 / 109,312 padded tokens
TEXT_TOKENS = 39  # prompt tokens of the H3 perf tests (CALIBRATED_FOX_PROMPT_NUM_TOKENS)
DEFAULTS = {
    "fa": {},
    # SGLang sol_attn defaults; band / dense_prefix_queries / k_chunk are this implementation's knobs.
    "sol_attn": {
        "tau": 1.0,
        "thresh_type": "diag",
        "kv_splits": "auto",
        "sink_start": 0,
        "sink_tokens": 0,
        "band": 1,
        "dense_prefix_queries": False,
        "k_chunk": 256,
    },
    # shuffle: VSA's shard-balanced tile placement (False: the tiles fill shard 0 first); rung: the pipeline's
    # bucket ladder rung, or "min" (the fewest whole blocks that hold the tiles).
    "video_sparse_attn": {"VSA_sparsity": 0.9, "vsa_mode": "exempt", "k_chunk": 256, "shuffle": True, "rung": "ladder"},
    "cube_sparse_attn": {
        "topk_ratio": 0.5,
        "local_cube_size": [4, 4, 4],
        "k_chunk": 256,
        "shuffle": True,
        "rung": "ladder",
    },
}
PIPELINE_KEYS = ("dense_steps", "dense_layers", "dense_backend")  # applied by the pipeline, not per call
FA_KEYS = ("rung",)  # the only config keys fa takes: "ladder" / "min" runs it at the VSA / cube padded geometry


# ----------------------------------------------------------------------------- inputs


def parse_config(text):
    """SGLang's --attention-backend-config: JSON, or k=v,k=v."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    out = {}
    for kv in text.split(","):
        k, v = kv.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v.strip()
    return out


def h3_t2va_shape(seconds, text_tokens=TEXT_TOKENS):
    """Packed segment lengths ([text | condition | audio | video]) and the video token grid (t, h, w) of `seconds` of
    768p (16:9) t2va, from the pipeline's packing helpers (as test_transformer_minimax_h3._packed_sizes)."""
    height, width = resolve_canvas_size(16, 9)
    num_frames = align_num_frames(int(seconds * MINIMAX_H3_FPS))
    grid = (video_latent_num_frames(num_frames), height // 16 // 2, width // 16 // 2)  # VAE x16, patch (1, 2, 2)
    audio = audio_latent_num_frames(num_frames) * MINIMAX_H3_AUDIO_CHANNELS
    return {"text": text_tokens, "condition": 0, "audio": audio, "video": math.prod(grid)}, grid


def synthetic_inputs(seconds, seed):
    """[1, T, 56, 128] bf16 q / k / v with T the packed t2va length at `seconds` (padded to the working point on
    device) and its metadata."""
    segments, grid = h3_t2va_shape(seconds)
    T = sum(segments.values())
    H = HEADS_PER_DEVICE * MESH_CONFIG.tp_size
    g = torch.Generator().manual_seed(seed)
    qkv = {n: torch.randn(1, T, H, D, generator=g).bfloat16() for n in "qkv"}
    meta = {
        "seconds": str(seconds),
        "segments": json.dumps(segments),
        "video_grid": json.dumps(list(grid)),
        "text_len": str(segments["text"]),
        "dense_tokens": str(T - segments["video"]),
        "source": f"synthetic seed {seed}",
    }
    return qkv, meta


def load_inputs(path):
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
        qkv = {n: f.get_tensor(n) for n in "qkv"}
    return qkv, meta


def save_inputs(path, qkv, meta):
    from safetensors.torch import save_file

    save_file({n: t.contiguous() for n, t in qkv.items()}, path, metadata=meta)


def expand_inputs(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, "*.safetensors"))) if os.path.isdir(p) else [p]
    return out


def segment_lengths(meta):
    return json.loads(meta["segments"]) if "segments" in meta else None


def text_len(meta):
    if "text_len" in meta:
        return int(meta["text_len"])
    seg = segment_lengths(meta)
    return seg.get("text") if seg else None


def non_video_tokens(meta, override):
    """Tokens before the video segment (packed [text | condition | audio | video]): dense rows / exempt keys."""
    if override is not None:
        return override
    seg = segment_lengths(meta)
    if seg:
        return sum(n for name, n in seg.items() if name != "video") if "video" in seg else 0
    return int(meta.get("dense_tokens", 0))


def to_device_layout(qkv, batch_index, rows):
    """BTHD -> [1, H, S * rows, D] float (bf16 values), zero padding after the T real tokens."""
    S = MESH_CONFIG.sp_size
    out = []
    for n in "qkv":
        x = qkv[n][batch_index].transpose(0, 1).float()  # [H, T, D]
        x = torch.nn.functional.pad(x, (0, 0, 0, S * rows - x.shape[1]))
        out.append(x.unsqueeze(0).contiguous())
    return out


def rows_for(T):
    """Rows per device: the smallest working point that holds T tokens, else T / SP rounded up to a block."""
    S = MESH_CONFIG.sp_size
    for rows in sorted(WORKING_ROWS.values()):
        if rows * S >= T:
            return rows
    return -(-T // (S * BLOCK)) * BLOCK


def video_grid(meta):
    """(t, h, w) video token grid from the metadata (`video_grid`: JSON list or "t,h,w"), else inferred from the
    video segment at 768p 16:9 (24 x 42 tokens per latent frame)."""
    if "video_grid" in meta:
        g = meta["video_grid"]
        return tuple(json.loads(g) if g.lstrip().startswith("[") else (int(x) for x in g.split(",")))
    seg = segment_lengths(meta)
    _, (_, gh, gw) = h3_t2va_shape(5)
    if seg and "video" in seg and seg["video"] % (gh * gw) == 0:
        logger.warning(f"no video_grid in the input metadata: assuming {gh} x {gw} tokens per latent frame")
        return (seg["video"] // (gh * gw), gh, gw)
    raise ValueError("the VSA / cube backends need the input's segments and video_grid metadata (README.md)")


def prefix_segments(meta):
    """Lengths of the packed segments before the video (the last segment), in order."""
    seg = segment_lengths(meta)
    if not seg or list(seg)[-1] != "video":
        raise ValueError(f"the VSA / cube backends need `segments` ending with video, got {seg}")
    return tuple(n for name, n in seg.items() if name != "video")


# ----------------------------------------------------------------------------- VSA tile layout


def unshuffled_vsa_geometry(segments, grid, *, sp_factor, padded_len):
    """build_vsa_geometry's tiles without its shard balancing (the layout before "add data shuffling to balance
    attention"): prefix tiles then video tiles fill the whole blocks of shard 0, then shard 1, ... Comparison only."""
    geo = build_vsa_geometry(segments, grid, sp_factor=sp_factor, padded_len=padded_len)
    prefix_sizes = chop_prefix_segments(segments)
    video_counts = video_tile_valid_counts(grid)
    sizes = torch.cat([torch.tensor(prefix_sizes, dtype=torch.long), video_counts])
    slots = geo.rows_per_shard // VSA_TILE_TOKENS
    tile = torch.arange(sizes.numel())
    shard, slot = tile // slots, tile % slots
    tile_block = shard * geo.blocks_per_shard + slot
    tile_start = shard * geo.rows_per_shard + slot * VSA_TILE_TOKENS
    n_prefix = len(prefix_sizes)
    valid_counts = torch.zeros(geo.n_blocks, dtype=torch.long)
    valid_counts[tile_block] = sizes
    is_exempt = torch.zeros(geo.n_blocks, dtype=torch.bool)
    is_exempt[tile_block[:n_prefix]] = True
    is_candidate = torch.zeros(geo.n_blocks, dtype=torch.bool)
    is_candidate[tile_block[n_prefix:]] = True
    prefix_len = sum(segments)
    tile_rows = list(torch.arange(prefix_len, dtype=torch.long).split(prefix_sizes))
    tile_rows += list((video_tile_partition_indices(grid) + prefix_len).split(video_counts.tolist()))
    gather_index = torch.full((padded_len,), -1, dtype=torch.long)
    row_source = torch.zeros(padded_len, dtype=torch.long)
    for start, rows in zip(tile_start.tolist(), tile_rows):
        gather_index[start : start + rows.numel()] = rows
        row_source[start : start + VSA_TILE_TOKENS] = rows[0]
        row_source[start : start + rows.numel()] = rows
    untile_index = torch.empty(geo.seq_len, dtype=torch.long)
    valid_rows = torch.nonzero(gather_index >= 0, as_tuple=False).reshape(-1)
    untile_index[gather_index[valid_rows]] = valid_rows
    return dataclasses.replace(
        geo,
        valid_counts=valid_counts,
        is_exempt=is_exempt,
        is_candidate=is_candidate,
        gather_index=gather_index,
        untile_index=untile_index,
        row_source=row_source,
    )


def vsa_layout(meta, T, cfg):
    """The request's VSA tile geometry (MiniMaxH3VSAGeometry) at the rung the pipeline would run it at."""
    segments, grid = prefix_segments(meta), video_grid(meta)
    if sum(segments) + math.prod(grid) != T:
        raise ValueError(f"prefix {segments} + video grid {grid} do not cover the {T} tokens")
    S = MESH_CONFIG.sp_size
    rung = vsa_min_padded_len(segments, grid, S)
    if cfg.get("rung", "ladder") == "ladder":
        # The pipeline module is heavy (encoders, VAE), so it is imported only here.
        from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import MINIMAX_H3_BUCKET_LADDER_4X8, select_bucket

        rung = select_bucket(rung, MINIMAX_H3_BUCKET_LADDER_4X8)
    build = build_vsa_geometry if cfg.get("shuffle", True) else unshuffled_vsa_geometry
    return build(segments, grid, sp_factor=S, padded_len=rung)


# ----------------------------------------------------------------------------- timing


def quantiles(times):
    xs = sorted(times)

    def q(p):
        i = p * (len(xs) - 1)
        lo, hi = math.floor(i), math.ceil(i)
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

    return q(0.5), q(0.1), q(0.9)


def traced(mesh_device, fn, reps, warmup, readback=None):
    """Compile call, one trace capture, warmup + reps timed replays (each followed by a synchronize).

    fn(hold) returns its output tensor and appends intermediates it allocates to `hold`; they stay allocated until
    the trace is released. readback(out) runs on the last replay's output. Returns (replay ms, readback result).
    """
    hold = []
    _free(fn(hold), hold)
    ttnn.synchronize_device(mesh_device)
    hold = []
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn(hold)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    times = []
    for i in range(warmup + reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    result = readback(out) if readback is not None else None
    ttnn.release_trace(mesh_device, trace_id)
    _free(out, hold)
    return times, result


def _free(out, hold=()):
    for t in [out, *hold]:
        if t is not None and t.is_allocated():
            ttnn.deallocate(t)


# ----------------------------------------------------------------------------- bench


class Bench:
    def __init__(self, rt, args):
        self.rt, self.args = rt, args
        self.ccl = CCLManager(mesh_device=rt.mesh_device, num_links=rt.num_links, topology=rt.topology)
        self.mesh = rt.mesh_device
        self.S, self.H = MESH_CONFIG.sp_size, HEADS_PER_DEVICE * MESH_CONFIG.tp_size
        self.heads = [int(h) for h in args.ref_heads.split(",")]
        self.shards = [int(s) for s in args.ref_shards.split(",")]
        self.kv_dtype = ttnn.bfloat8_b if args.kv_dtype == "bf8" else ttnn.bfloat16
        self._topk_prepasses = {}

    def composer(self, sp_dim, tp_dim):
        dims = [None, None]
        dims[self.rt.sp_axis], dims[self.rt.tp_axis] = sp_dim, tp_dim
        return ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=dims)

    def output_heads(self, t):
        """The reference heads of a [1, H, N, D] device output, on host (float)."""
        return ttnn.to_torch(t, mesh_composer=self.composer(2, 1))[:, self.heads].float()

    def accuracy(self, out_heads, exact, logical_n, extra_ref=None):
        """rel L2 / PCC / fitted scale vs exact fp64 dense attention on the reference slice."""
        o = out_heads[:, :, :logical_n]
        acc = {
            "rel_l2_vs_dense": _rel_l2(exact, o),
            "pcc_vs_dense": _pcc(exact, o),
            "scale_vs_dense": _fit_scale(exact, o),
        }
        if extra_ref is not None:  # the op against its own block-sparse definition on the same mask
            acc["pcc_vs_block_sparse_ref"] = _pcc(extra_ref, o)
        return acc

    def run(self, backend, cfg, q, k, v, logical_n, meta, exact):
        """One backend on one input: eager call for accuracy, then traced timing. Returns (times, fields, extra)."""
        if backend in TOPK_BACKENDS:
            return self._topk(backend, cfg, q, k, v, logical_n, meta, exact)
        if backend == "fa" and cfg.get("rung"):
            return self._fa_at_rung(cfg, q, k, v, logical_n, meta)
        rt, S = self.rt, self.S
        rows = q.shape[2] // S
        sparse = backend != "fa"
        k_chunk = int(cfg.get("k_chunk", 256))
        geo = Geometry(rows, logical_n, k_chunk, dense_chunks=DENSE_CHUNKS.get(rows, (256, 512)))
        lofi = sparse and self.args.lofi
        case = Case(rt, geo, dtype=self.kv_dtype if sparse else ttnn.bfloat16, lofi=lofi)
        tq = case._upload(q, case.seq_dims, dtype=ttnn.bfloat16)
        tk, tv = (case._upload(x, case.seq_dims, dtype=case.dtype) for x in (k, v))  # as a QKV matmul would emit
        device_inputs = [tq, tk, tv]
        sl = dict(heads=self.heads, shards=self.shards)
        fields, extra = {"kept_fraction": 1.0}, {"rows_per_device": rows, "k_chunk": k_chunk if sparse else None}
        try:
            if backend == "fa":
                fn = lambda hold: self._fa_call(case, tq, tk, tv)  # noqa: E731
                extra["dense_chunks"] = list(geo.dense_chunks)
                eager = fn([])
                ttnn.synchronize_device(self.mesh)
                eager_heads = self.output_heads(eager)
                ttnn.deallocate(eager)
                fields.update(self.accuracy(eager_heads, exact, logical_n))
                times, traced_heads = traced(self.mesh, fn, self.args.reps, self.args.warmup, self.output_heads)
            else:
                times, traced_heads, eager_heads = self._sol(
                    case, cfg, tq, tk, tv, q, k, v, meta, exact, fields, extra, sl
                )
        finally:
            for t in device_inputs + [case.buf_k, case.buf_v]:
                ttnn.deallocate(t)
        extra["max_abs_diff_traced_vs_eager"] = float((traced_heads - eager_heads).abs().max())
        return times, fields, extra

    def _fa_call(self, case, tq, tk, tv):
        rt = self.rt
        return call_sdpa(
            tq,
            tk,
            tv,
            case.geo.logical_n,
            False,
            False,
            case.buf_k,
            case.buf_v,
            case.dense_config,
            rt.compute_kernel_config,
            rt.ccl_semaphore_handles,
            rt.num_links,
            rt.sp_axis,
            self.mesh,
            rt.topology,
            rt.worker_sub_device_id,
            rt.ccl_column,
        )

    def _fa_at_rung(self, cfg, q, k, v, logical_n, meta):
        """The dense op on the VSA / cube padded geometry: Q / K / V in the same tile layout at the same rung, with
        logical_n = the rung as VSA / cube get it, so it does the same padded work. It then attends the pad rows too,
        so its output is not the real tokens' attention: the accuracy fields are null."""
        lay = vsa_layout(meta, logical_n, {"rung": cfg["rung"], "shuffle": True})
        rows = lay.rows_per_shard
        geo = Geometry(rows, lay.padded_len, 256, dense_chunks=DENSE_CHUNKS.get(rows, (256, 512)))
        case = Case(self.rt, geo)
        packed = [lay.pack_rows(x[0, :, :logical_n], dim=1).unsqueeze(0) for x in (q, k, v)]
        tq, tk, tv = (case._upload(x, case.seq_dims, dtype=ttnn.bfloat16) for x in packed)
        del packed
        try:
            fn = lambda hold: self._fa_call(case, tq, tk, tv)  # noqa: E731
            eager = fn([])
            ttnn.synchronize_device(self.mesh)
            eager_heads = self.output_heads(eager)
            ttnn.deallocate(eager)
            times, traced_heads = traced(self.mesh, fn, self.args.reps, self.args.warmup, self.output_heads)
        finally:
            for t in (tq, tk, tv, case.buf_k, case.buf_v):
                ttnn.deallocate(t)
        fields = {"kept_fraction": 1.0, "rel_l2_vs_dense": None, "pcc_vs_dense": None, "scale_vs_dense": None}
        extra = {
            "rows_per_device": rows,
            "rung": lay.padded_len,
            "dense_chunks": list(geo.dense_chunks),
            "accuracy": "null: logical_n = rung, so the pad rows are attended (timing baseline only)",
            "max_abs_diff_traced_vs_eager": float((traced_heads - eager_heads).abs().max()),
        }
        return times, fields, extra

    def _sparse_call(self, case, q, k, v, mask, pk, pv, counts=None):
        rt = self.rt
        return ttnn.transformer.sparse_ring_sdpa(
            q,
            k,
            v,
            mask,
            pk,
            pv,
            persistent_output_buffer_k=case.buf_k,
            persistent_output_buffer_v=case.buf_v,
            logical_n=case.geo.logical_n,
            program_config=case.sparse_config,
            compute_kernel_config=rt.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=rt.ccl_semaphore_handles,
            num_links=rt.num_links,
            cluster_axis=rt.sp_axis,
            mesh_device=self.mesh,
            topology=rt.topology,
            subdevice_id=rt.worker_sub_device_id,
            ccl_core_grid_offset=(rt.ccl_column, 0),
            use_column_major_ccl=True,
            block_size=BLOCK,
            block_counts=counts,
        )

    def _ref_inputs(self, case, q, k, v):
        """The values the op reads (LoFi rounding as in test_sparse_ring_sdpa.Case.prepare), for its own reference."""
        if case.lofi:
            return round_significand(q, 7), *(case.quantize(x, bits=5) for x in (k, v))
        return q, *(case.quantize(x) for x in (k, v))

    def _sol(self, case, cfg, tq, tk, tv, q, k, v, meta, exact, fields, extra, sl):
        geo, S = case.geo, self.S
        assert int(cfg.get("sink_start", 0)) == 0, "sink_start != 0 is not supported (the prefix is packed first)"
        assert cfg.get("thresh_type", "diag") == "diag", "the pre-pass implements the diag threshold"
        sink = cfg.get("sink_tokens", 0)
        if sink == "text_len":
            sink = text_len(meta)
            if sink is None:
                sink = non_video_tokens(meta, self.args.dense_tokens)
                logger.warning(f"no text_len in the input metadata: sink_tokens = non-video prefix {sink}")
        elif sink == "prefix":
            sink = non_video_tokens(meta, self.args.dense_tokens)
        sink = int(sink)
        extra["sink_tokens_resolved"] = sink
        pre = SolPrepass(
            mesh_device=self.mesh,
            ccl_manager=self.ccl,
            sp_axis=self.rt.sp_axis,
            tau=float(cfg.get("tau", 1.0)),
            thresh_type="diag",
            band=int(cfg.get("band", 1)),
            dense_prefix_queries=bool(cfg.get("dense_prefix_queries", False)),
        )
        scale = case.scale
        pre.prepare(SolGeometry(shards=S, rows=geo.rows, logical_n=geo.logical_n), sink, scale)

        # The pre-pass reads the op's inputs (bf16 Q, K / V in the op's dtype); pooled K / V come back in K's dtype.
        def full(hold):
            o = pre(tq, tk, tv, scale=scale)
            hold += [o["block_mask"], o["pooled_k"], o["pooled_v"]]
            return self._sparse_call(case, tq, tk, tv, o["block_mask"], o["pooled_k"], o["pooled_v"])

        # Eager: the mask and output for the accuracy checks (also compiles both programs).
        o = pre(tq, tk, tv, scale=scale, report_density=True)
        mask_t, pk_t, pv_t = o["block_mask"], o["pooled_k"], o["pooled_v"]
        eager = self._sparse_call(case, tq, tk, tv, mask_t, pk_t, pv_t)
        ttnn.synchronize_device(self.mesh)
        mask = ttnn.to_torch(mask_t, mesh_composer=self.composer(0, 1)).to(torch.int64)  # [S, H, nb_pad, S*nb_pad]
        eager_heads = self.output_heads(eager)
        ttnn.deallocate(eager)
        fields["kept_fraction"] = _density(mask, geo, S)
        rq, rk, rv = self._ref_inputs(case, q, k, v)
        pk_h, pv_h = (case.quantize(pool(x, geo, S), bits=case.kv_bits()) for x in (rk, rv))
        ref = sparse_reference(rq, rk, rv, mask, pk_h, pv_h, geo, S, scale, **sl)[:, self.heads, : geo.logical_n]
        fields.update(self.accuracy(eager_heads, exact, geo.logical_n, ref))
        del ref, mask

        times, traced_heads = traced(self.mesh, full, self.args.reps, self.args.warmup, self.output_heads)
        # The two halves alone: the pre-pass, and the attention call on the eager call's mask / pooled K / V.
        pre_times, _ = traced(
            self.mesh, lambda h: _hold_all(pre(tq, tk, tv, scale=scale), h), self.args.reps, self.args.warmup
        )
        attn_times, _ = traced(
            self.mesh,
            lambda h: self._sparse_call(case, tq, tk, tv, mask_t, pk_t, pv_t),
            self.args.reps,
            self.args.warmup,
        )
        _free(None, [mask_t, pk_t, pv_t])
        pre._release_state()
        extra["scoring"] = "device (SolPrepass: pool, pooled K / V all-gathers, route)"
        extra["scoring_ms_median"] = extra["prepass_ms_median"] = quantiles(pre_times)[0]
        extra["attention_only_ms_median"] = quantiles(attn_times)[0]
        return times, traced_heads, eager_heads

    def _topk_prepass(self, rung, sparsity):
        """One VSAPrepass per (rung, sparsity), kept for the run: its routing buffers have no release call."""
        key = (rung, sparsity)
        if key not in self._topk_prepasses:
            self._topk_prepasses[key] = VSAPrepass(
                rung=rung,
                sp_factor=self.S,
                sparsity=sparsity,
                num_heads=HEADS_PER_DEVICE,
                head_dim=D,
                mesh_device=self.mesh,
                ccl_manager=self.ccl,
                sp_axis=self.rt.sp_axis,
            )
        return self._topk_prepasses[key]

    def _topk(self, backend, cfg, q, k, v, logical_n, meta, exact):
        """video_sparse_attn / cube_sparse_attn: the sequence in VSA's tile layout (the pipeline reorders it once on
        the host at model input, so the reorder is not timed), VSAPrepass routing on device, the sparse op kept-only
        with block_counts. Prefix (non-video) Q tiles are dense rows and prefix K tiles always kept, by the mask."""
        rt, S = self.rt, self.S
        cube = backend == "cube_sparse_attn"
        if cube:
            assert list(cfg.get("local_cube_size", [4, 4, 4])) == [4, 4, 4], "VSA's tiles are 4 x 4 x 4 cubes"
            sparsity = 1.0 - float(cfg["topk_ratio"])
        else:
            assert cfg.get("vsa_mode", "exempt") == "exempt", "VSAPrepass implements exempt prefix tiles only"
            sparsity = float(cfg["VSA_sparsity"])
        coarse = not cube
        k_chunk = int(cfg.get("k_chunk", 256))
        lay = vsa_layout(meta, logical_n, cfg)
        rows = lay.rows_per_shard
        # logical_n = the rung, as the model passes it; block_counts mask the pad rows.
        geo = Geometry(rows, lay.padded_len, k_chunk)
        case = Case(rt, geo, dtype=self.kv_dtype, lofi=self.args.lofi)
        packed = [lay.pack_rows(x[0, :, :logical_n], dim=1).unsqueeze(0) for x in (q, k, v)]  # [1, H, rung, D]
        tq, tk, tv = (case._upload(x, case.seq_dims, dtype=ttnn.bfloat16) for x in packed)
        tk_op, tv_op = tk, tv
        if case.dtype != ttnn.bfloat16:  # block_pool reads bf16 only: the op gets its own K / V copies
            tk_op, tv_op = (case._upload(x, case.seq_dims, dtype=case.dtype) for x in packed[1:])
        device_inputs = [tq, tk, tv] + ([tk_op, tv_op] if tk_op is not tk else [])
        pre = self._topk_prepass(lay.padded_len, sparsity)
        pre.update(lay, traced=False)
        n_cand = int(lay.is_candidate.sum())
        request = vsa_routing_request(lay, pre.nb_pad, 1)
        counts = request.counts.reshape(-1).to(torch.int64)  # [S * nb_pad], as the device block_counts
        fields = {}
        extra = {
            "rows_per_device": rows,
            "k_chunk": k_chunk,
            "rung": lay.padded_len,
            "shuffle": bool(cfg.get("shuffle", True)),
            "video_grid": list(lay.video_grid),
            "prefix_segments": list(prefix_segments(meta)),
            "prefix_tiles": lay.n_prefix_tiles,
            "video_tiles": lay.n_video_tiles,
            "topk": max(1, min(math.ceil((1 - sparsity) * n_cand), n_cand)),
            "coarse_branch": coarse,
            "scoring_inputs": "bf16" if case.dtype == ttnn.bfloat16 else "bf16 (block_pool reads bf16 only)",
            "non_video_tokens": logical_n - math.prod(lay.video_grid),
        }

        def score(hold):
            r = pre(tq, tk, tv, coarse=coarse)
            hold += [t for t in (r["block_mask"], r["coarse_out"]) if t is not None]
            return r

        def full(hold):
            r = score(hold)
            return self._sparse_call(case, tq, tk_op, tv_op, r["block_mask"], None, None, r["block_counts"])

        def unpacked_heads(t):
            return lay.unpack_rows(self.output_heads(t), dim=2)  # [1, heads, T, D] in packed (input) order

        try:
            hold = []
            r = score(hold)
            eager = self._sparse_call(case, tq, tk_op, tv_op, r["block_mask"], None, None, r["block_counts"])
            ttnn.synchronize_device(self.mesh)
            mask = ttnn.to_torch(r["block_mask"], mesh_composer=self.composer(0, 1)).to(torch.int64)
            eager_heads = unpacked_heads(eager)
            ttnn.deallocate(eager)
            fields["kept_fraction"] = _density(mask, geo, S, counts)
            rq, rk, rv = self._ref_inputs(case, *packed)
            zeros = torch.zeros(1, rq.shape[1], S * geo.nb_pad(), D)  # no pooled entries in kept-only mode
            ref = sparse_reference(
                rq, rk, rv, mask, zeros, zeros, geo, S, case.scale, pooled=False, counts=counts,
                heads=self.heads, shards=self.shards,
            )  # fmt: skip
            del rq, rk, rv, mask
            ref = lay.unpack_rows(ref[:, self.heads], dim=2)
            fields.update(self.accuracy(eager_heads, exact, logical_n, ref))
            del ref

            times, traced_heads = traced(self.mesh, full, self.args.reps, self.args.warmup, unpacked_heads)
            score_times, _ = traced(self.mesh, lambda h: score(h)["block_mask"], self.args.reps, self.args.warmup)
            attn_times, _ = traced(
                self.mesh,
                lambda h: self._sparse_call(case, tq, tk_op, tv_op, r["block_mask"], None, None, r["block_counts"]),
                self.args.reps,
                self.args.warmup,
            )
            _free(None, hold)
        finally:
            for t in device_inputs + [case.buf_k, case.buf_v]:
                ttnn.deallocate(t)
        extra["scoring"] = "device (VSAPrepass: block_pool, K / V means all-gather, block_topk_route)"
        extra["scoring_ms_median"] = quantiles(score_times)[0]
        extra["attention_only_ms_median"] = quantiles(attn_times)[0]
        extra["max_abs_diff_traced_vs_eager"] = float((traced_heads - eager_heads).abs().max())
        return times, fields, extra


def _hold_all(out, hold):
    hold += [t for t in out.values() if t is not None]
    return None


def commit():
    try:
        sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "-uno"], text=True).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", nargs="+", help="safetensors files or directories of them")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--seconds", type=int, choices=sorted(WORKING_ROWS), help="synthetic length; recorded for inputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-inputs", help="write the synthetic inputs to this safetensors path")
    p.add_argument("--attention-backend", default="sol_attn", help=f"comma list of {', '.join(BACKENDS)}")
    p.add_argument("--attention-backend-config", default="", help="JSON or k=v,k=v (applies to the sparse backends)")
    p.add_argument("--lofi", action="store_true", help="LoFi matmuls in the sparse op")
    p.add_argument("--kv-dtype", choices=("bf16", "bf8"), default="bf16", help="K / V dtype of the sparse op")
    p.add_argument("--dense-tokens", type=int, default=None, help="non-video prefix length when inputs lack it")
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--ref-heads", default=None, help="heads of the accuracy slice (default first and last)")
    p.add_argument("--ref-shards", default=None, help="SP shards of the accuracy slice (default first and last)")
    p.add_argument("--output", default="results_tt.jsonl")
    args = p.parse_args()
    if args.synthetic and args.seconds is None:
        p.error("--synthetic needs --seconds")
    backends = [b.strip() for b in args.attention_backend.split(",") if b.strip()]
    for b in backends:
        if b not in BACKENDS:
            p.error(f"unknown backend {b}")
    user_cfg = parse_config(args.attention_backend_config)
    H = HEADS_PER_DEVICE * MESH_CONFIG.tp_size
    args.ref_heads = args.ref_heads or f"0,{H - 1}"
    args.ref_shards = args.ref_shards or f"0,{MESH_CONFIG.sp_size - 1}"

    if args.synthetic:
        qkv, meta = synthetic_inputs(args.seconds, args.seed)
        if args.save_inputs:
            save_inputs(args.save_inputs, qkv, meta)
            logger.info(f"synthetic inputs written to {args.save_inputs}")
        sources = [(f"synthetic_{args.seconds}s_seed{args.seed}", lambda: (qkv, meta))]
    else:
        sources = [(os.path.basename(f), (lambda f=f: load_inputs(f))) for f in expand_inputs(args.inputs)]

    rt = open_ring_joint_sdpa_runtime(MESH_CONFIG, trace_region_size=TRACE_REGION_SIZE)
    try:
        bench = Bench(rt, args)
        for name, load in sources:
            qkv, meta = load()
            B, T, Hq, Dq = qkv["q"].shape
            if (Hq, Dq) != (H, D):
                raise ValueError(f"{name}: [B, T, H, D] = {tuple(qkv['q'].shape)}; expected H={H}, D={D}")
            rows = rows_for(T)
            q, k, v = to_device_layout(qkv, args.batch_index, rows)
            del qkv
            scale = D**-0.5
            sl = dict(heads=bench.heads, shards=bench.shards)
            t0 = time.perf_counter()
            exact = dense_reference(q, k, v, T, scale, rows=rows, **sl)[:, bench.heads, :T]
            logger.info(f"{name}: T={T}, {rows} rows per device, exact reference {time.perf_counter() - t0:.0f} s")
            seconds = args.seconds if args.seconds is not None else meta.get("seconds")
            for backend in backends:
                fa_cfg = {key: val for key, val in user_cfg.items() if key in FA_KEYS}
                cfg = {**DEFAULTS[backend], **(user_cfg if backend != "fa" else fa_cfg)}
                times, fields, extra = bench.run(backend, cfg, q, k, v, T, meta, exact)
                med, p10, p90 = quantiles(times)
                ignored = [key for key in cfg if key in PIPELINE_KEYS]
                if ignored:
                    extra["pipeline_keys_not_applied"] = ignored
                extra["padded_tokens"] = extra["rows_per_device"] * MESH_CONFIG.sp_size
                dtype = "bf16"
                if backend != "fa":
                    dtype = ("bf16" if args.kv_dtype == "bf16" else "q bf16, kv bf8") + (" lofi" if args.lofi else "")
                rec = {
                    "platform": "tenstorrent",
                    "device": f"Blackhole Galaxy {MESH_CONFIG.tp_size}x{MESH_CONFIG.sp_size} "
                    f"(TP={MESH_CONFIG.tp_size} heads, SP={MESH_CONFIG.sp_size} sequence)",
                    "backend": backend,
                    "config": cfg,
                    "seconds": None if seconds is None else int(seconds),
                    "tokens": T,
                    "heads": H,
                    "head_dim": D,
                    "dtype": dtype,
                    "ms_median": med,
                    "ms_p10": p10,
                    "ms_p90": p90,
                    "kept_fraction": fields.pop("kept_fraction"),
                    "rel_l2_vs_dense": fields.pop("rel_l2_vs_dense"),
                    "pcc_vs_dense": fields.pop("pcc_vs_dense"),
                    "scale_vs_dense": fields.pop("scale_vs_dense"),
                    "commit": commit(),
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "input": name,
                    "layer": meta.get("layer"),
                    "step": meta.get("step"),
                    "reps": args.reps,
                    "extra": {**fields, **extra},
                }
                acc = ", ".join(
                    f"{label} {'-' if rec[key] is None else format(rec[key], fmt)}"
                    for label, key, fmt in (
                        ("rel L2", "rel_l2_vs_dense", ".4f"),
                        ("pcc", "pcc_vs_dense", ".5f"),
                        ("scale", "scale_vs_dense", ".4f"),
                    )
                )
                logger.info(
                    f"{name} {backend}: {med:.3f} ms (p10 {p10:.3f}, p90 {p90:.3f}), kept {rec['kept_fraction']:.3f}, "
                    f"{acc}; {rec['extra']}"
                )
                with open(args.output, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            del q, k, v, exact
    finally:
        close_ring_joint_sdpa_runtime(rt)


if __name__ == "__main__":
    main()
