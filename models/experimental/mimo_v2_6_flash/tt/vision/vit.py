# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo ViT (``visual.*``) on one 1x8 row, tensor-parallel (TP=8).

Per chip: 4 of the 32 Q heads and 1 of the 8 KV heads (chip c holds Q heads 4c..4c+3 and the KV head they share),
a 576-wide slice of the SwiGLU MLP and a 640-wide slice of the merger MLP. In BF16 that is 201 MB of weights per chip
(full replication would be ~1.4 GB).

Sequence parallel: the residual stream [N, 1280] is split by rows over the row's chips (chip c holds rows
c * N / 8 ..), always in row (processor) order. A block's RMSNorm runs on the chip's rows and is all-gathered (bf16)
as the input of the column-parallel linears (QKV, MLP gate / up); the row-parallel linears' partial sums (attention
output, MLP down) are reduce-scattered back to row slices and added to the residual there. Same CCL volume as an
all-reduce of a replicated stream, but the fp32 residual adds and norms touch 1/8 of the rows. The pixels arrive
row-sharded too, so the patch embedding needs no collective; the merger gathers its normed input.

Attention (see reference/vision.py for the semantics):

* Full-attention blocks (0/9/18/27): flash SDPA with ``cu_window_seqlens``, one window per frame / image.
* Window blocks: |i - j| <= 64 inside each frame plus a per-head sink on the frame's first key. Queries run in blocks
  of 64 against the 192 keys around them (the block and 64 on each side): [blocks, 1, 4 heads x 64, 192] scores
  from one batched matmul, an additive mask (band, frame boundaries, padding; the sink as an indicator times the
  head's sink), softmax, and a batched matmul with V. The scores cost 1.5x the band's useful work, nothing else
  grows with the sequence.
* Column-order blocks (5-8, 14-17, 23-26): HF permutes the residual stream into column order for each run of them.
  Only attention depends on the order (norms and MLPs are per row), so here the gathered norm output is permuted
  into column order and the attention's concatenated heads back (embedding lookups of their rows, exact); RoPE
  tables come in both orders. The band mask is the same in both orders (frames keep their row range).

The 1/sqrt(64) attention scale is folded into the Q weights and bias (a power of two: exact in bf16).

Every attention sequence (a frame pair of a video or an image) is independent, so a long video runs as several
passes of whole frames (``max_rows`` patch rows per pass); a single image always runs in one pass.

Bounded programs: a pass is padded to the next of ``ROW_BUCKETS`` row counts (padding is its own masked window,
dropped from the output), and ``cu_window_seqlens`` always has ``CU_ENTRIES`` entries (empty windows at the end), so
every program a call runs is one of the ``len(ROW_BUCKETS)`` sets that ``warm()`` compiles up front. A pass longer
than the largest bucket (only if the processor allows images over 32768 patches) falls back to a shape of its own.
"""

import os
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch
from loguru import logger

import ttnn

from ...reference import vision as ref
from ..common import MeshPlan, as_weight, cache_fill, cache_name, compute_config
from ..decoder import TtRMSNorm

BLOCK = 64  # query rows per window-attention block; keys per block: BLOCK + 2 * window
MASK_NEG = -1.0e9
DEFAULT_MAX_ROWS = 16384
# Pass row buckets: four per octave from 1024 (at most 25% padding, ~14% at the top of an octave; multiples of 256:
# SDPA chunks, 64-row attention blocks, 32-row merged tiles). 32768 is the processor's largest image (8388608 pixels
# / 256 per patch). A 32-frame 640x360 video pass (14080 rows) lands in 14336.
ROW_BUCKETS = tuple(
    b for octave in (1024, 2048, 4096, 8192, 16384) for b in (octave, octave * 5 // 4, octave * 3 // 2, octave * 7 // 4)
) + (32768,)
# cu_window_seqlens entries per pass (fixed so the SDPA programs do not depend on the number of frames): up to
# CU_ENTRIES - 2 sequences plus the padding window.
CU_ENTRIES = 256
MAX_SEQS_PER_PASS = CU_ENTRIES - 2
# Buckets whose passes are traced by default (prepare_traces): up to a 1024x1024 image. A trace replay costs ~0 host
# time against the eager pass's ~40 us of enqueue per op (~1000 ops: the whole small-image latency).
TRACE_MAX_ROWS = 4096
TRACE_BUCKETS = tuple(b for b in ROW_BUCKETS if b <= TRACE_MAX_ROWS)
# Pass geometries whose host input tensors (PassInputs.host) stay cached, least recently used out first: at most
# this many, and at most HOST_INPUT_CACHE_BYTES of them (MIMO_VIT_HOST_CACHE_MB, default 128). One geometry is
# ~0.7 KB a row per device copy (11 MB at 16384 rows, 21 MB at 32768: the band mask, cos, sin, index tensors), so 64
# large ones (many image sizes / video lengths) were 0.7-1.3 GB of host RAM per row engine, and more where a
# replicated host tensor keeps a copy per device. The byte bound counts one copy.
HOST_INPUT_CACHE = 64
HOST_INPUT_CACHE_BYTES = int(float(os.environ.get("MIMO_VIT_HOST_CACHE_MB", "128")) * 2**20)
_DTYPE_BYTES = {ttnn.bfloat16: 2, ttnn.uint16: 2, ttnn.uint8: 1}  # every other dtype used here is 4 bytes


def host_input_bytes(host):
    """Bytes of one copy of a pass's host inputs ({name: host tensor}): elements times element size."""
    total = 0
    for t in host.values():
        elements = 1
        for d in tuple(t.shape):
            elements *= int(d)
        total += elements * _DTYPE_BYTES.get(t.dtype, 4)
    return total


class HostInputCache:
    """Pass geometry -> PassInputs.host, least recently used out first, bounded by entries and by bytes
    (``host_input_bytes``). The entry just used always stays, whatever its size."""

    def __init__(self, max_entries=HOST_INPUT_CACHE, max_bytes=None, size_of=host_input_bytes):
        self.max_entries = max_entries
        self.max_bytes = HOST_INPUT_CACHE_BYTES if max_bytes is None else max_bytes
        self.size_of = size_of
        self.entries = OrderedDict()  # key -> (host, bytes)
        self.bytes = 0

    def __len__(self):
        return len(self.entries)

    def get(self, key, make):
        """The cached inputs of ``key``, made with ``make()`` on a miss."""
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
            return entry[0]
        host = make()
        size = self.size_of(host)
        self.entries[key] = (host, size)
        self.bytes += size
        while len(self.entries) > 1 and (len(self.entries) > self.max_entries or self.bytes > self.max_bytes):
            _, (_, dropped) = self.entries.popitem(last=False)
            self.bytes -= dropped
        return host


# The column-parallel linears (QKV, MLP gate / up) of passes up to this many rows take a 2D-multicast program config
# (1.4-1.7x faster than the auto config at 1024 / 4096 rows); larger passes keep the auto config (within 10% there,
# and the per-core blocks outgrow L1).
LINEAR_2D_MAX_ROWS = 4096


@dataclass(frozen=True)
class VisionNumerics:
    """``matmul_fidelity``: the linears' math fidelity (attention matmuls, softmax, RoPE and SDPA always run HiFi4
    with fp32 accumulation). ``residual_dtype``: the residual stream between blocks. ``reduce_dtype``: the partial
    sums of the row-parallel projections (attention output, MLP down, merger) and their reduce-scatter.

    An fp32 ``ttnn.all_reduce`` (reduce-scatter + all-gather) on the 1x8 row was seen handing its output to the next
    op before every chip's share had landed ([1, 1, 1024, 1280] and [1, 1, 1280, 1280]: random results run to run,
    exact with a device sync after it; tests/perf/allreduce_fp32_race.py checks it standalone), so fp32 reduces are
    followed by ``ttnn.synchronize_device``. bf16 reduces showed no such race (every row bucket is checked bit for
    bit run to run in tests/test_vision_tp8.py)."""

    matmul_fidelity: object = ttnn.MathFidelity.HiFi2
    residual_dtype: object = ttnn.bfloat16
    reduce_dtype: object = ttnn.bfloat16


NUMERICS = {
    "fast": VisionNumerics(),  # fails the 0.99 per-block bar on some inputs
    "balanced": VisionNumerics(ttnn.MathFidelity.HiFi4, ttnn.float32, ttnn.bfloat16),  # the default
    "accurate": VisionNumerics(ttnn.MathFidelity.HiFi4, ttnn.float32, ttnn.float32),  # + a sync per fp32 reduce
}
DEFAULT_NUMERICS = "balanced"


def reduce_scatter(plan, x, dim):
    """Sum of the row's partial ``x`` (deallocated), chip c keeping the c-th slice of ``dim``; an fp32 one is
    followed by a device sync (see VisionNumerics)."""
    out = ttnn.reduce_scatter(
        x,
        dim,
        cluster_axis=plan.tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_links=plan.num_links,
        topology=plan.topology,
    )
    x.deallocate(True)
    if out.dtype == ttnn.float32:
        ttnn.synchronize_device(plan.mesh_device)
    return out


def col_linear_program(grid, rows, k, n):
    """2D-multicast config of a [rows, k] x [k, n] linear on ``grid`` (in0 blocks of 8 tiles), or None (auto)."""
    mt, nt, kt = rows // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE, k // ttnn.TILE_SIZE
    if rows > LINEAR_2D_MAX_ROWS or kt % 8:
        return None
    pm, pn = -(-mt // grid.y), -(-nt // grid.x)
    sw = max(w for w in (4, 2, 1) if pn % w == 0)
    sh = max(h for h in (4, 2, 1) if pm % h == 0 and h * sw <= 4)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=8,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def band_program(grid, rows, nt, kt):
    """The window attention's batched matmuls ([blocks, 1, 256, *]: 8 tile rows per block, ``nt`` x ``kt`` tiles):
    blocks of 4 (8 above 2048 rows) tile rows x all ``nt`` columns spread over the grid. The auto config puts one
    block per core (16 cores at 1024 rows): the scores matmul runs 2-2.7x and the probabilities x V one 7-10x faster
    this way, bit-identical."""
    pm = 4 if rows <= 2048 else 8
    sw = max(w for w in (4, 3, 2, 1) if nt % w == 0)
    sh = max(h for h in (4, 2, 1) if pm % h == 0 and h * sw <= 4)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=kt,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=nt,
    )


def gather_rows(x, index):
    """bf16 [1, 1, rows, d] -> its rows in ``index`` order (an embedding lookup into the tensor itself)."""
    rows, d = x.shape[-2], x.shape[-1]
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    table = ttnn.reshape(rm, (rows, d))
    out = ttnn.embedding(index, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rm.deallocate(True)
    return ttnn.reshape(out, (1, 1, rows, d))


def padded_rows(n, buckets=ROW_BUCKETS):
    """Rows a pass of ``n`` patch rows runs with: the smallest bucket that holds it. Without buckets (or past the
    largest) a multiple of 128 (SDPA chunks, 64-row attention blocks, 32-row merged tiles), coarser as ``n`` grows
    (at most ~1/16 padding)."""
    for b in buckets or ():
        if b >= n:
            return b
    step = 128
    while n > 16 * step:
        step *= 2
    return -(-n // step) * step


@dataclass
class PassGeometry:
    """Host-side layout of one pass: its sequences (frames / images) back to back, padded to ``rows``."""

    grids: list  # [(t, h, w)] in row order; every temporal step of an entry is its own sequence
    n: int  # real patch rows
    rows: int  # padded rows

    @classmethod
    def build(cls, grids, buckets=ROW_BUCKETS):
        n = sum(t * h * w for t, h, w in grids)
        return cls(grids=list(grids), n=n, rows=padded_rows(n, buckets))

    def seq_bounds(self):
        """[0, ..., n, rows]: the sequences, then the padding as one more window."""
        cu = ref.cu_seqlens(self.grids).tolist()
        return cu + ([self.rows] if self.rows > self.n else [])

    def cu_window_seqlens(self):
        """seq_bounds padded to CU_ENTRIES with empty windows at ``rows`` (the SDPA skips them)."""
        bounds = self.seq_bounds()
        assert len(bounds) <= CU_ENTRIES, f"{len(bounds) - 1} windows in one pass (at most {CU_ENTRIES - 1})"
        return bounds + [self.rows] * (CU_ENTRIES - len(bounds))


def plan_passes(grid_thw, max_rows, max_seqs=MAX_SEQS_PER_PASS):
    """Split items into passes of whole frames, at most ``max_rows`` patch rows and ``max_seqs`` frames / images each
    (a single frame / image larger than ``max_rows`` gets a pass of its own).
    Returns [(row_start, row_end, [(t, h, w), ...])]."""
    passes, cur, start, rows, seqs = [], [], 0, 0, 0
    for t, h, w in ref.as_grid_list(grid_thw):
        frame = h * w
        left = t
        while left:
            fit = max(1, (max_rows - rows) // frame) if rows else max(1, max_rows // frame)
            take = min(left, fit, max_seqs - seqs)
            if rows and (take <= 0 or rows + take * frame > max_rows):
                passes.append((start, start + rows, cur))
                start, rows, seqs, cur = start + rows, 0, 0, []
                continue
            cur.append((take, h, w))
            rows += take * frame
            seqs += take
            left -= take
    if cur:
        passes.append((start, start + rows, cur))
    return passes


def band_masks(geo, window):
    """Window-attention mask and sink indicator, each [blocks, 1, BLOCK, BLOCK + 2 * window] bf16 (shared by the
    heads: the scores are viewed as [blocks, heads, BLOCK, span]).

    Block b's queries are rows b*BLOCK .. +BLOCK, its keys rows b*BLOCK - window .. b*BLOCK + BLOCK + window.
    ``mask``: 0 where key j is visible from query i (same sequence, |i - j| <= window, 0 <= j < rows), MASK_NEG
    elsewhere. ``sink``: 1 where a visible key is its sequence's first row."""
    rows = geo.rows
    nb = rows // BLOCK
    span = BLOCK + 2 * window
    bounds = torch.tensor(geo.seq_bounds(), dtype=torch.int32)
    seq_id = torch.bucketize(torch.arange(rows, dtype=torch.int32), bounds[1:], right=True)
    is_first = torch.zeros(rows + 1, dtype=torch.bool)
    is_first[bounds[:-1][bounds[:-1] < geo.n].long()] = True
    i = torch.arange(rows, dtype=torch.int32).view(nb, BLOCK, 1)
    j = (torch.arange(nb, dtype=torch.int32) * BLOCK - window).view(nb, 1, 1) + torch.arange(
        span, dtype=torch.int32
    ).view(1, 1, span)
    jc = j.clamp(0, rows - 1).long()
    visible = (j >= 0) & (j < rows) & ((i - j).abs() <= window) & (seq_id[i.long()] == seq_id[jc])
    mask = torch.zeros(visible.shape, dtype=torch.bfloat16).masked_fill_(~visible, MASK_NEG)
    sink = (visible & is_first[jc]).to(torch.bfloat16)
    return mask.unsqueeze(1), sink.unsqueeze(1)


class PassInputs:
    """Device tensors describing one pass (replicated on the row): RoPE tables in both orders, the order
    permutations, the SDPA windows and the band masks.

    ``host(plan, cfg, geo)`` builds what is uploaded, as host tensors: they depend on the pass geometry only, never on
    the pixels, so the model caches them per geometry. ``upload`` puts a host set on the device; ``copy_from`` writes a
    host set of the same row count into existing device tensors (a traced pass's persistent inputs). Uploaded are the
    row-order RoPE tables and one ``band`` tensor (0: visible key, 1: visible sink key, MASK_NEG: hidden); ``derive``
    makes the rest on the device at the start of a pass (column-order tables: the row-order ones gathered into column
    order; the mask and sink indicator: band - relu(band) and relu(band)), all exact, and ``release`` frees them."""

    def __init__(self, geo, tensors):
        self.geo = geo
        self.tensors = tensors
        self.order = {True: tensors["to_col"], False: tensors["to_row"]}
        self.cu = tensors["cu"]
        self.window_rows = tensors["window_rows"]
        self.cos = self.sin = self.mask = self.sink = None

    @staticmethod
    def host(plan, cfg, geo):
        rows = geo.rows
        rep = plan.replicate()

        def host(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=rep)

        def index(t):
            return host(t.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

        pad = rows - geo.n
        freqs = ref.rotary_freqs(geo.grids, cfg.head_dim, cfg.spatial_merge_size)
        emb = torch.cat((freqs, freqs), dim=-1)

        def table(fn, fill):
            t = torch.cat([fn(emb), torch.full((pad, emb.shape[-1]), fill)]) if pad else fn(emb)
            return host(t.reshape(1, 1, rows, -1))

        # Row permutations (padding rows stay in place): to column order and back.
        to_col = ref.patch_row_index(ref.col_window_index(geo.grids, cfg.spatial_merge_size), cfg.merge_unit)
        to_col = torch.cat([to_col, torch.arange(geo.n, rows)])
        mask, sink = band_masks(geo, cfg.window_size)
        # Window blocks' key rows, block after block (b*BLOCK - window .. + span; rows outside the pass are masked,
        # so they read any row: the nearest one).
        nb, span = rows // BLOCK, BLOCK + 2 * cfg.window_size
        keys = (torch.arange(nb) * BLOCK - cfg.window_size).view(nb, 1) + torch.arange(span).view(1, span)
        cu = torch.tensor(geo.cu_window_seqlens(), dtype=torch.int32)
        return {
            "cos": table(torch.cos, 1.0),
            "sin": table(torch.sin, 0.0),
            "to_col": index(to_col),
            "to_row": index(torch.argsort(to_col)),
            "cu": host(cu, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "band": host(mask + sink),
            "window_rows": index(keys.clamp(0, rows - 1)),
        }

    @classmethod
    def upload(cls, plan, geo, host):
        mesh = plan.mesh_device
        return cls(geo, {k: ttnn.to_device(t, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) for k, t in host.items()})

    def copy_from(self, host):
        for k, t in host.items():
            ttnn.copy_host_to_device_tensor(t, self.tensors[k])

    def derive(self):
        """The device-made inputs."""
        t = self.tensors
        dram = ttnn.DRAM_MEMORY_CONFIG
        self.cos = {False: t["cos"], True: gather_rows(t["cos"], t["to_col"])}
        self.sin = {False: t["sin"], True: gather_rows(t["sin"], t["to_col"])}
        self.sink = ttnn.relu(t["band"], memory_config=dram)
        self.mask = ttnn.subtract(t["band"], self.sink, memory_config=dram)

    def release(self):
        for x in (self.cos[True], self.sin[True], self.sink, self.mask):
            x.deallocate(True)
        self.cos = self.sin = self.mask = self.sink = None

    def deallocate(self):
        for t in self.tensors.values():
            t.deallocate(True)


def norm_bf16(norm, x):
    """RMSNorm of the residual stream, as bf16 matmul input."""
    h = norm(x, prefill=True)
    if h.dtype == ttnn.bfloat16:
        return h
    out = ttnn.typecast(h, ttnn.bfloat16)
    h.deallocate(True)
    return out


def gathered_norm(plan, norm, x):
    """RMSNorm of this chip's residual rows, all-gathered: bf16 [1, 1, rows, hidden] on every chip."""
    h = norm_bf16(norm, x)
    out = plan.all_gather(h, dim=2)
    h.deallocate(True)
    return out


class TtVisionBlock:
    def __init__(self, plan, cfg, i, sd, cache_dir, weight_dtype, numerics, mm_config, attn_config):
        self.plan = plan
        self.cfg = cfg
        self.numerics = numerics
        self.i = i
        self.full = cfg.is_full(i)
        self.col = cfg.is_col(i)
        self.has_sink = cfg.has_sink(i)
        self.mm_config = mm_config
        self.attn_config = attn_config
        self.grid = plan.mesh_device.compute_with_storage_grid_size()
        # The tuned linear and SDPA configs (col_linear_program, _sdpa_program) change rounding. With bf16 weights
        # every variant measured keeps the final PCC >= 0.9985; with bfloat8_b weights the final PCC of a 448x448
        # image swings between 0.989 and 0.9985 with such changes (block 27's massive activations), so those keep
        # the auto linears and the earlier SDPA chunks, measured >= 0.998.
        self.tuned = weight_dtype == ttnn.bfloat16
        tp = plan.tp
        hd = cfg.head_dim
        self.local_heads = cfg.num_heads // tp
        self.local_kv = cfg.num_kv_heads // tp
        assert self.local_kv == 1 and self.local_heads * tp == cfg.num_heads, "one KV head per chip"
        prefix = f"vit_tp{tp}_b{i}"
        name = lambda n: cache_name(cache_dir, prefix, n)
        p = f"blocks.{i}."
        w = {}
        if sd is not None:
            hidden = cfg.hidden_size
            scale = hd**-0.5
            qkv_w, qkv_b = sd[p + "attn.qkv.weight"], sd[p + "attn.qkv.bias"]
            q_dim, kv_dim = cfg.num_heads * hd, cfg.num_kv_heads * hd
            q_w, k_w, v_w = qkv_w[:q_dim] * scale, qkv_w[q_dim : q_dim + kv_dim], qkv_w[q_dim + kv_dim :]
            q_b, k_b, v_b = qkv_b[:q_dim] * scale, qkv_b[q_dim : q_dim + kv_dim], qkv_b[q_dim + kv_dim :]
            qh, kh = self.local_heads * hd, hd
            chips_w, chips_b = [], []
            for c in range(tp):
                chips_w += [q_w[c * qh : (c + 1) * qh], k_w[c * kh : (c + 1) * kh], v_w[c * kh : (c + 1) * kh]]
                chips_b += [q_b[c * qh : (c + 1) * qh], k_b[c * kh : (c + 1) * kh], v_b[c * kh : (c + 1) * kh]]
            w["wqkv"] = torch.cat(chips_w).T.contiguous()  # [hidden, tp * (qh + 2 kh)], chip blocks side by side
            w["bqkv"] = torch.cat(chips_b).reshape(1, -1)
            w["wo"] = sd[p + "attn.proj.weight"].T.contiguous()  # [heads * hd, hidden]: chip c's rows = its heads
            w["bo"] = self._first_chip_bias(sd[p + "attn.proj.bias"], tp)
            w["wg"] = sd[p + "mlp.gate_proj.weight"].T.contiguous()
            w["bg"] = sd[p + "mlp.gate_proj.bias"].reshape(1, -1)
            w["wu"] = sd[p + "mlp.up_proj.weight"].T.contiguous()
            w["bu"] = sd[p + "mlp.up_proj.bias"].reshape(1, -1)
            w["wd"] = sd[p + "mlp.down_proj.weight"].T.contiguous()
            w["bd"] = self._first_chip_bias(sd[p + "mlp.down_proj.bias"], tp)
            if self.has_sink:
                sinks = sd[p + "attn.sinks"].float()
                span = BLOCK + 2 * cfg.window_size
                col = sinks.view(tp, self.local_heads, 1).repeat_interleave(BLOCK, dim=1)  # [tp, heads * 64, 1]
                w["sink"] = col.expand(tp, self.local_heads * BLOCK, span).permute(1, 0, 2).reshape(1, 1, -1, tp * span)
            assert hidden == w["wo"].shape[1]

        def load(key, mapper, dtype=weight_dtype):
            return as_weight(w.get(key), plan, mapper, dtype, name(key))

        self.wqkv = load("wqkv", plan.shard(-1))
        self.bqkv = load("bqkv", plan.shard(-1), ttnn.bfloat16)
        self.wo = load("wo", plan.shard(-2))
        self.bo = load("bo", plan.shard(-1), ttnn.bfloat16)
        self.wg = load("wg", plan.shard(-1))
        self.bg = load("bg", plan.shard(-1), ttnn.bfloat16)
        self.wu = load("wu", plan.shard(-1))
        self.bu = load("bu", plan.shard(-1), ttnn.bfloat16)
        self.wd = load("wd", plan.shard(-2))
        self.bd = load("bd", plan.shard(-1), ttnn.bfloat16)
        self.sink = None
        if self.has_sink:  # [1, heads, BLOCK, span]: head h's sink everywhere in its slice
            sink = load("sink", plan.shard(-1), ttnn.bfloat16)
            self.sink = ttnn.reshape(sink, (1, self.local_heads, BLOCK, BLOCK + 2 * cfg.window_size))
        eps = cfg.rms_norm_eps
        self.norm1 = TtRMSNorm(plan, sd and sd[p + "norm1.weight"], eps, name("norm1"))
        self.norm2 = TtRMSNorm(plan, sd and sd[p + "norm2.weight"], eps, name("norm2"))

    @staticmethod
    def _first_chip_bias(bias, tp):
        """[1, tp * n]: the bias in chip 0's slice, zeros elsewhere, added once by the reduce."""
        return torch.cat([bias.reshape(1, -1), torch.zeros(1, bias.numel() * (tp - 1))], dim=1)

    def linear(self, x, w, b=None, dtype=ttnn.bfloat16, column=False):
        """``column``: a column-parallel linear of the gathered rows (2D-multicast config up to LINEAR_2D_MAX_ROWS)."""
        tuned = column and self.tuned
        program = col_linear_program(self.grid, x.shape[-2], w.shape[-2], w.shape[-1]) if tuned else None
        return ttnn.linear(
            x,
            w,
            bias=b,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
            program_config=program,
        )

    def __call__(self, x, inp):
        """``x``: this chip's rows of the residual stream [1, 1, rows / tp, hidden] (row order) -> the block's."""
        res = self.numerics.residual_dtype
        h = gathered_norm(self.plan, self.norm1, x)
        attn = self.attention(h, inp)
        h.deallocate(True)
        x2 = ttnn.add(x, attn, dtype=res, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        h = gathered_norm(self.plan, self.norm2, x2)
        gate = self.linear(h, self.wg, self.bg, column=True)
        up = self.linear(h, self.wu, self.bu, column=True)
        h.deallocate(True)
        act = ttnn.mul(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate.deallocate(True)
        up.deallocate(True)
        down = self.linear(act, self.wd, self.bd, dtype=self.numerics.reduce_dtype)
        act.deallocate(True)
        down = reduce_scatter(self.plan, down, 2)
        out = ttnn.add(x2, down, dtype=res, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x2.deallocate(True)
        down.deallocate(True)
        return out

    def attention(self, h, inp):
        """``h``: the gathered norm output (row order) -> this chip's rows of the attention output (reduced)."""
        if self.col:
            h = gather_rows(h, inp.order[True])
        qkv = self.linear(h, self.wqkv, self.bqkv, column=True)
        if self.col:
            h.deallocate(True)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.local_heads,
            num_kv_heads=self.local_kv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv.deallocate(True)
        cos, sin = inp.cos[self.col], inp.sin[self.col]
        q_rot = ttnn.experimental.rotary_embedding_hf(q, cos, sin, compute_kernel_config=self.attn_config)
        k_rot = ttnn.experimental.rotary_embedding_hf(k, cos, sin, compute_kernel_config=self.attn_config)
        q.deallocate(True)
        k.deallocate(True)
        if self.full:
            o = ttnn.transformer.scaled_dot_product_attention(
                q_rot,
                k_rot,
                v,
                is_causal=False,
                scale=1.0,
                cu_window_seqlens=inp.cu,
                program_config=self._sdpa_program(inp.geo.rows),
                compute_kernel_config=self.attn_config,
            )
        else:
            o = self.band_attention(q_rot, k_rot, v, inp)
        for t in (q_rot, k_rot, v):
            t.deallocate(True)
        cat = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o.deallocate(True)
        if self.col:
            cat_col = cat
            cat = gather_rows(cat_col, inp.order[False])
            cat_col.deallocate(True)
        out = self.linear(cat, self.wo, self.bo, dtype=self.numerics.reduce_dtype)
        cat.deallocate(True)
        return reduce_scatter(self.plan, out, 2)

    def _sdpa_program(self, rows):
        """Chunks from a sweep (4 heads x head dim 64 per chip): 32 x 64 up to 2048 rows (230 vs 382 us at 128 x 128
        for 1024 rows: 4x the work units on the grid), 64 x 256 above (4096-row image 1115 vs 1369 us at 256 x 256,
        16 frames of 880 rows 1256 vs 1490 us). Untuned (bfloat8_b weights): 128 x 128 below 2048 rows, else 256."""
        if self.tuned:
            q_chunk, k_chunk = (32, 64) if rows <= 2048 else (64, 256)
        else:
            q_chunk = k_chunk = 256 if rows >= 2048 else 128
        while rows % k_chunk:
            k_chunk //= 2
        while rows % q_chunk:
            q_chunk //= 2
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    def _windows(self, t, nb, window_rows):
        """[1, 1, rows, d] -> [nb, 1, BLOCK + 2 * window, d]: block b's keys, rows b*BLOCK - window .. +span (one
        embedding lookup of the rows ``window_rows`` lists)."""
        rows, d = t.shape[-2], t.shape[-1]
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.embedding(
            window_rows, ttnn.reshape(rm, (rows, d)), layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        rm.deallocate(True)
        return ttnn.reshape(out, (nb, 1, -1, d))

    def band_attention(self, q, k, v, inp):
        rows = q.shape[-2]
        nb = rows // BLOCK
        heads, hd = self.local_heads, self.cfg.head_dim
        qb = ttnn.reshape(q, (heads, nb, BLOCK, hd))
        qb_t = ttnn.permute(qb, (1, 0, 2, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qb = ttnn.reshape(qb_t, (nb, 1, heads * BLOCK, hd))
        kw = self._windows(k, nb, inp.window_rows)
        vw = self._windows(v, nb, inp.window_rows)
        span_t = kw.shape[-2] // ttnn.TILE_SIZE
        scores = ttnn.matmul(
            qb,
            kw,
            transpose_b=True,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attn_config,
            program_config=band_program(self.grid, rows, span_t, hd // ttnn.TILE_SIZE),
        )
        qb_t.deallocate(True)
        kw.deallocate(True)
        span = scores.shape[-1]
        scores = ttnn.reshape(scores, (nb, heads, BLOCK, span))  # the same tiles, heads as a batch dim
        mask = inp.mask
        if self.sink is not None:  # the mask plus each head's sink on its frames' first keys
            sink = ttnn.mul(inp.sink, self.sink, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            mask = ttnn.add(sink, inp.mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sink.deallocate(True)
        masked = ttnn.add(scores, mask, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores.deallocate(True)
        if mask is not inp.mask:
            mask.deallocate(True)
        probs = ttnn.softmax(
            masked,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attn_config,
        )
        masked.deallocate(True)
        probs_bf16 = ttnn.reshape(ttnn.typecast(probs, ttnn.bfloat16), (nb, 1, heads * BLOCK, span))
        probs.deallocate(True)
        ob = ttnn.matmul(
            probs_bf16,
            vw,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attn_config,
            program_config=band_program(self.grid, rows, hd // ttnn.TILE_SIZE, span // ttnn.TILE_SIZE),
        )
        probs_bf16.deallocate(True)
        vw.deallocate(True)
        ob = ttnn.reshape(ob, (nb, heads, BLOCK, hd))
        o = ttnn.permute(ob, (1, 0, 2, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ob.deallocate(True)
        return ttnn.reshape(o, (1, heads, rows, hd))


class TtMiMoVisionTransformer:
    """The ViT on one 1x8 mesh (or submesh). ``forward(pixel_values, grid_thw)`` -> host bf16 [N / 4, 4096].

    ``state_dict``: ``reference.vision.load_vision_state_dict`` output, or None to load everything from a warm
    ``cache_dir``. ``weight_dtype``: matmul weights (norms, biases and sinks stay bf16). ``buckets``: pass row
    counts (``ROW_BUCKETS``; None pads each pass to a multiple of 128 instead, one program set per size)."""

    def __init__(
        self,
        mesh_device,
        cfg,
        state_dict=None,
        cache_dir=None,
        weight_dtype=ttnn.bfloat16,
        max_rows=DEFAULT_MAX_ROWS,
        num_links=2,
        numerics=NUMERICS[DEFAULT_NUMERICS],
        buckets=ROW_BUCKETS,
    ):
        assert tuple(mesh_device.shape)[0] == 1, f"one row (1 x TP), got {tuple(mesh_device.shape)}"
        self.plan = MeshPlan(mesh_device, num_links=num_links)
        self.cfg = cfg
        self.buckets = tuple(sorted(buckets)) if buckets else ()
        assert not self.buckets or max_rows <= self.buckets[-1], "a video pass must fit the largest bucket"
        assert all(b % 256 == 0 for b in self.buckets), self.buckets
        self.max_rows = max_rows
        tp = self.plan.tp
        sd = state_dict
        self.numerics = numerics
        self.mm_config = compute_config(numerics.matmul_fidelity)
        self.attn_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)
        name = lambda n: cache_name(cache_dir, f"vit_tp{tp}", n)
        w = {}
        if sd is not None:
            w["patch"] = sd["patch_embed.proj.weight"].reshape(cfg.hidden_size, -1).T.contiguous()
            w["m0"] = sd["merger.mlp.0.weight"].T.contiguous()
            w["m2"] = sd["merger.mlp.2.weight"].T.contiguous()
        load = lambda key, mapper: as_weight(w.get(key), self.plan, mapper, weight_dtype, name(key))
        self.patch_w = load("patch", self.plan.replicate())
        self.merger_w0 = load("m0", self.plan.shard(-1))
        self.merger_w2 = load("m2", self.plan.shard(-2))
        self.merger_norm = TtRMSNorm(self.plan, sd and sd["merger.ln_q.weight"], 1e-6, name("ln_q"))
        self.blocks = [
            TtVisionBlock(self.plan, cfg, i, sd, cache_dir, weight_dtype, numerics, self.mm_config, self.attn_config)
            for i in range(cfg.depth)
        ]
        self._host_cache = HostInputCache()  # (grids, rows) -> PassInputs.host(...)
        self._traces = {}  # rows -> PassTrace (prepare_traces)

    @classmethod
    def from_checkpoint(cls, mesh_device, model_path, cache_dir=None, **kwargs):
        """Converts from the checkpoint once per ``cache_dir`` (``cache_fill`` lock, as the LLM's weights), then
        loads the cached tensors."""
        cfg = ref.VisionConfig.from_model_path(model_path)
        tp = tuple(mesh_device.shape)[1]
        weight_dtype = kwargs.get("weight_dtype", ttnn.bfloat16)
        with cache_fill(cache_dir, f"vit_tp{tp}_{weight_dtype.name.lower()}") as need:
            sd = ref.load_vision_state_dict(model_path) if need else None
            return cls(mesh_device, cfg, sd, cache_dir, **kwargs)

    # -- one pass ------------------------------------------------------------------------------------------------

    def _merger(self, x):
        cfg = self.cfg
        h = gathered_norm(self.plan, self.merger_norm, x)
        rows = h.shape[-2]
        rm = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h.deallocate(True)
        merged_rm = ttnn.reshape(rm, (1, 1, rows // cfg.merge_unit, cfg.merged_dim))
        merged = ttnn.to_layout(merged_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rm.deallocate(True)
        kw = dict(dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.mm_config)
        h0 = ttnn.linear(merged, self.merger_w0, **kw)
        merged.deallocate(True)
        g = ttnn.gelu(h0, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h0.deallocate(True)
        out = ttnn.linear(g, self.merger_w2, **{**kw, "dtype": self.numerics.reduce_dtype})
        g.deallocate(True)
        return reduce_scatter(self.plan, out, 3)  # chip c: output columns c * 512 .. (the host concatenates them)

    def _host_rows(self, x, geo):
        """The row-sharded residual stream -> torch [n, d] (row order)."""
        t = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(self.plan.mesh_device, dim=2))
        return t.reshape(geo.rows, -1)[: geo.n]

    def _host_inputs(self, geo):
        """PassInputs.host for ``geo``, cached per pass geometry (they never depend on the pixels)."""
        key = (tuple(geo.grids), geo.rows)
        return self._host_cache.get(key, lambda: PassInputs.host(self.plan, self.cfg, geo))

    def _host_pixels(self, pixel_values, geo):
        """Pixel rows [geo.n .. geo.rows, patch_dim] -> host bf16 [1, 1, rows, patch_dim] row-major tensor, rows
        sharded over the row's chips (the pass tilizes and all-gathers them: 1/8 of a replicated upload, no host
        tilize)."""
        pv = pixel_values.to(torch.bfloat16)
        assert geo.n <= pv.shape[0] <= geo.rows, (pv.shape, geo.n, geo.rows)
        if geo.rows > pv.shape[0]:
            pv = torch.cat([pv, torch.zeros(geo.rows - pv.shape[0], pv.shape[-1], dtype=pv.dtype)])
        return ttnn.from_torch(
            pv.reshape(1, 1, geo.rows, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.plan.shard(2),
        )

    def _read_out(self, out, geo):
        """The merger output on device (row-major) -> host bf16 [geo.n / 4, out_hidden]."""
        unit = self.cfg.merge_unit
        # One read of every chip's columns (3-4x faster than a to_torch per chip).
        t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.plan.mesh_device, dim=3))
        return t.reshape(geo.rows // unit, -1)[: geo.n // unit].to(torch.bfloat16)

    def _pass(self, pixels, inp, own_inputs=True, hidden=None):
        """The device pass: row-sharded pixels + PassInputs -> the merger output (reduce-scattered: chip c holds
        output columns c * 512 ..), row-major. ``own_inputs``: deallocate the inputs as soon as they are used (an eager pass); a
        traced pass keeps them (persistent). ``hidden``: see forward_pass (eager only: it reads every block's
        output)."""
        geo = inp.geo
        dram = ttnn.DRAM_MEMORY_CONFIG
        tiles = ttnn.to_layout(pixels, ttnn.TILE_LAYOUT, memory_config=dram)
        if own_inputs:
            pixels.deallocate(True)
        inp.derive()
        x = ttnn.linear(  # this chip's rows of the residual stream
            tiles,
            self.patch_w,
            dtype=self.numerics.residual_dtype,
            memory_config=dram,
            compute_kernel_config=self.mm_config,
        )
        tiles.deallocate(True)
        for blk in self.blocks:
            x_new = blk(x, inp)
            x.deallocate(True)
            x = x_new
            if hidden is not None:
                hidden.append(self._host_rows(x, geo).float())
        inp.release()
        out = self._merger(x)
        x.deallocate(True)
        if own_inputs:
            inp.deallocate()
        out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        out.deallocate(True)
        return out_rm

    def forward_pass(self, pixel_values, geo, hidden=None, trace=True):
        """One pass: pixel rows [geo.n, patch_dim] -> host bf16 [geo.n / 4, out_hidden]. Rows past ``geo.n`` (up to
        ``geo.rows``; zeros when not given) are padding: they never reach the real rows. ``hidden``: a list that
        collects each block's output (host fp32 [n, hidden], row order) for tests (eager). ``trace``: replay the
        bucket's trace when it has one (prepare_traces; bit-identical to the eager pass)."""
        pixels = self._host_pixels(pixel_values, geo)
        host = self._host_inputs(geo)
        traced = self._traces.get(geo.rows) if trace and hidden is None else None
        if traced is not None and self._replay(traced, pixels, host):
            return self._read_out(traced.out, geo)
        mesh = self.plan.mesh_device
        inp = PassInputs.upload(self.plan, geo, host)
        out = self._pass(ttnn.to_device(pixels, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG), inp, hidden=hidden)
        result = self._read_out(out, geo)
        out.deallocate(True)
        return result

    def forward(self, pixel_values, grid_thw, hidden=None):
        """pixel_values [N, patch_dim] (processor layout, any float dtype), grid_thw [items, 3] -> host bf16
        [N / 4, out_hidden_size], items in order."""
        outs = []
        for start, end, grids in plan_passes(grid_thw, self.max_rows):
            geo = PassGeometry.build(grids, self.buckets)
            outs.append(self.forward_pass(pixel_values[start:end], geo, hidden))
        return torch.cat(outs, dim=0)

    __call__ = forward

    def warm(self, buckets=None):
        """Compile the programs of every pass size up front (each bucket's are fixed: see the module docstring), one
        zero-pixel pass per bucket. ``buckets``: a subset of ``self.buckets`` (default: all). Returns {rows: seconds}.
        Run it before the LLM captures its traces: compiling later allocates program binaries in DRAM."""
        times = {}
        for rows in buckets if buckets is not None else self.buckets:
            assert rows in self.buckets, f"{rows} is not one of the buckets {self.buckets}"
            t0 = time.time()
            self.forward_pass(torch.zeros(rows, self.cfg.patch_dim), self._warm_geometry(rows), trace=False)
            times[rows] = time.time() - t0
        return times

    def _warm_geometry(self, rows):
        return PassGeometry.build([(1, 32, rows // 32)], self.buckets)  # one image of exactly ``rows`` patches

    # -- traced passes -------------------------------------------------------------------------------------------

    def traceable(self):
        """Whether passes can be traced: the device has a trace region (without one a capture would take the DRAM
        heap), and no fp32 reduce (it needs a device sync after it, VisionNumerics, which a trace cannot hold)."""
        return self.numerics.reduce_dtype != ttnn.float32 and _trace_region(self.plan.mesh_device)[1] > 0

    def prepare_traces(self, buckets=TRACE_BUCKETS):
        """Allocate the persistent inputs and output of a traced pass for each of ``buckets`` and compile everything
        it runs (one eager run of its body). Call it before the LLM captures its traces (as warm()), so their
        replays never write over these buffers. The traces themselves are captured later (capture_traces, or on
        first use). Returns the persistent DRAM per chip (bytes)."""
        if not self.traceable():
            return 0
        mesh = self.plan.mesh_device
        before = _dram_allocated(mesh)
        new = []
        for rows in buckets:
            assert rows in self.buckets, f"{rows} is not one of the buckets {self.buckets}"
            if rows in self._traces:
                continue
            geo = self._warm_geometry(rows)
            pixels = self._host_pixels(torch.zeros(rows, self.cfg.patch_dim), geo)
            pixels = ttnn.to_device(pixels, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            inp = PassInputs.upload(self.plan, geo, self._host_inputs(geo))
            unit, out_dim = self.cfg.merge_unit, self.cfg.out_hidden_size
            out_cols = out_dim // self.plan.tp
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows // unit, out_cols]),
                self.numerics.reduce_dtype,
                ttnn.ROW_MAJOR_LAYOUT,
                mesh,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._traces[rows] = PassTrace(rows, pixels, inp, out)
            new.append(rows)
        persistent = _dram_allocated(mesh) - before  # (the bodies below can add program binaries)
        for rows in new:  # after every persistent buffer, so the bodies' temporaries do not split them up
            self._trace_body(self._traces[rows])
        ttnn.synchronize_device(mesh)
        return persistent

    def prepared_traces(self):
        """The buckets whose persistent trace buffers are allocated (prepare_traces)."""
        return tuple(self._traces)

    def _trace_body(self, traced):
        out = self._pass(traced.pixels, traced.inp, own_inputs=False)
        ttnn.copy(out, traced.out)
        out.deallocate(True)

    def capture_traces(self, buckets=None):
        """Capture the prepared traces not captured yet (``buckets``: a subset; default all prepared). Capture after
        every long-lived allocation of the process (the LLM's traces): a replay writes the trace's temporaries where
        they were at capture. Returns {rows: trace region bytes per chip}; a bucket whose capture fails (trace region
        full) runs eagerly, with a warning."""
        sizes = {}
        for rows, traced in self._traces.items():
            if traced.id is None and (buckets is None or rows in buckets):
                sizes[rows] = self._capture(traced)
        return sizes

    def _capture(self, traced):
        mesh = self.plan.mesh_device
        before = _trace_allocated(mesh)
        trace_id = None
        try:
            ttnn.synchronize_device(mesh)
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                self._trace_body(traced)
            finally:
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh)
        except Exception as e:  # e.g. the trace region is full: that bucket runs eagerly
            if trace_id is not None:
                try:
                    ttnn.release_trace(mesh, trace_id)
                except Exception:
                    pass
            logger.warning(f"MiMo ViT: capturing the {traced.rows}-row pass failed, it runs eagerly: {e}")
            traced.failed = True
            return 0
        traced.id = trace_id
        traced.programs = mesh.num_program_cache_entries()
        traced.bytes = _trace_allocated(mesh) - before
        return traced.bytes

    def _replay(self, traced, pixels, host):
        """Copy a pass's inputs into its trace's buffers and replay it (non-blocking; the output read waits). Captures
        it first if it is not yet (or again if programs were compiled since its capture: their binaries can sit where
        its temporaries are). Returns False when the bucket has no usable trace (it then runs eagerly)."""
        if traced.failed:
            return False
        mesh = self.plan.mesh_device
        if traced.id is not None and traced.programs != mesh.num_program_cache_entries():
            ttnn.release_trace(mesh, traced.id)
            traced.id = None
            traced.recaptures += 1
        if traced.id is None and not self._capture(traced):
            return False
        ttnn.copy_host_to_device_tensor(pixels, traced.pixels)
        traced.inp.copy_from(host)
        ttnn.execute_trace(mesh, traced.id, cq_id=0, blocking=False)
        return True

    def release_traces(self):
        """Release every trace and free its persistent buffers."""
        mesh = self.plan.mesh_device
        for traced in self._traces.values():
            if traced.id is not None:
                ttnn.release_trace(mesh, traced.id)
            traced.pixels.deallocate(True)
            traced.inp.deallocate()
            traced.out.deallocate(True)
        self._traces = {}

    def trace_info(self):
        """{rows: (captured, trace region bytes per chip, recaptures)} of the prepared traces."""
        return {r: (t.id is not None, t.bytes, t.recaptures) for r, t in self._traces.items()}


@dataclass
class PassTrace:
    """A bucket's traced pass: persistent sharded pixel rows, PassInputs and output, and the trace."""

    rows: int
    pixels: object
    inp: PassInputs
    out: object
    id: object = None
    programs: int = -1  # program-cache entries at capture
    bytes: int = 0  # trace region per chip
    recaptures: int = 0
    failed: bool = False


def _dram_allocated(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


def _trace_region(mesh):
    """(allocated, total) bytes of the trace region per chip; (0, 0) without one."""
    try:
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
    except Exception:
        return 0, 0
    return view.total_bytes_allocated_per_bank * view.num_banks, view.total_bytes_per_bank * view.num_banks


def _trace_allocated(mesh):
    return _trace_region(mesh)[0]
