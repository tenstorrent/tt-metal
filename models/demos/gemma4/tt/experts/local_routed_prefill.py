# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local routed-expert prefill for the whole-expert TP line (no token leaves its chip inside the experts).

On the 4-chip line (the (4,1) column mesh the 26B-A4B serves on, TP=4 over the rows) the dense TP all-reduce leaves
the residual REPLICATED: every chip already holds every token of the call, and each chip owns 32 whole experts of the
decode copy (``experts/weights.py``: tp index ``c`` holds experts ``[c * E_loc, (c + 1) * E_loc)``). The token
dispatch / combine of ``dispatch_prefill.py`` therefore moves tokens that are already everywhere: on a 4096-token
chunk at position 0 (device profiler, served env, 2026-11-17) its dispatch and ``untilize_combine`` programs took
~229 ms of the ~554 ms chunk, 5.4x the expert compute. Here every chip computes the routed pairs of ITS experts from
its own copy of the chunk and the caller's TP all-reduce sums the per-chip partials (as for the sparse_matmul path):

1. ``custom_ops/local_routed_gather``: from the router's top-k ``(values, ids)`` (TILE, straight from its top-k: no
   dense ``[S, E]`` scatter upstream, no second top-k here) the cores split the tokens, count the pairs per local
   expert, the leader lays out tile-aligned expert regions, and each core copies its tokens' rows into the
   expert-major buffer (``[M, H]`` bf16 ROW_MAJOR, ``M = T * K + 32 * E_loc`` the drop-free bound) and writes the
   combine's per-token row index (``SKIP_ROW`` on other chips' slots) and slot weights (zero there);
2. ``custom_ops/geglu_routed_expert`` (``GegluRoutedExpert.from_stacked``) runs each local expert over its region,
   reading the routed experts' DECODE copy in place: one copy of the routed experts per chip
   (``GEMMA4_MOE_DISPATCH_SHARED_WEIGHTS`` semantics; nothing is loaded or cached for this path, so a warm-cache
   placeholder state dict never reaches a converter);
3. ``custom_ops/local_routed_untilize`` converts only the regions' real rows of the bf8 TILE expert output back to
   bf16 rows, in place over the consumed buffer;
4. ``deepseek_prefill.post_combine_reduce`` with ``combine_row_index``: per token the weighted sum of its local slots'
   rows read straight from that table (other chips' slots are neither read nor computed) -> this chip's ``[S, H]``
   partial; ``Gemma4Experts`` all-reduces it over TP.

No fabric op runs inside the experts. Calls longer than ``GEMMA4_MOE_LOCAL_ROUTED_CHUNK`` (default 4096) are processed
in chunks; any row count works: the programs are built per row bucket (``row_bucket``: the count rounded up to 256,
the real count a runtime argument), the padding tokens route nowhere and their rows are sliced off. The path is the default where the dispatch prefill was
(``mesh_orientation.local_routed_prefill_default``: a Blackhole (4,1) column mesh); ``GEMMA4_MOE_LOCAL_ROUTED_PREFILL=0``
returns to the dispatch path, ``=1`` asks for it on another whole-expert tp-only line (``local_routed_reason``).

History: the first local path (2026-10-23, its own per-expert weight copy, an index pipeline of ``ttnn.sort`` /
``masked_bincount`` / fp32 prefix-sum matmuls / ``scatter`` / two ``embedding`` gathers over EVERY expert's pairs)
failed the 2K TTFT gate (last-token logits PCC 0.522 vs the base, top-1 mismatch). Its region offsets (an fp32 matmul
prefix sum) went wrong once an expert held 2048+ rows of a chunk, which real prompts reach in 4K chunks; the 2K
position itself is ill-conditioned for every path (``local_routed_index.py`` has the probes). The gather kernel's
integer offsets replace that pipeline.
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.custom_ops.geglu_routed_expert import GegluRoutedExpert
from models.demos.gemma4.custom_ops.local_routed_gather import LocalRoutedGather
from models.demos.gemma4.custom_ops.local_routed_untilize import LocalRoutedUntilize

TILE = 32
LOCAL_ROUTED_ENV = "GEMMA4_MOE_LOCAL_ROUTED_PREFILL"
LOCAL_ROUTED_MIN_TOKENS_ENV = "GEMMA4_MOE_LOCAL_ROUTED_MIN_TOKENS"
LOCAL_ROUTED_CHUNK_ENV = "GEMMA4_MOE_LOCAL_ROUTED_CHUNK"
# A call's row count is rounded up to a multiple of this for the gather / GeGLU programs (the dispatch path's fit-chunk
# unit on the 4-chip line): every prompt length of a bucket shares one compiled program set, the real count is a
# runtime argument; the padding tokens route nowhere.
ROW_BUCKET = 256


def row_bucket(rows: int) -> int:
    """The program row count of a ``rows``-row chunk: ``rows`` rounded up to ``ROW_BUCKET``. Pure host logic."""
    return -(-int(rows) // ROW_BUCKET) * ROW_BUCKET


def local_routed_chunk() -> int:
    """Tokens per local-routed chunk (``GEMMA4_MOE_LOCAL_ROUTED_CHUNK``, default 4096, a multiple of 32): bounds the
    expert buffer (``chunk * K + 32 * E_loc`` rows) and the kernels' per-call token count."""
    chunk = int(os.environ.get(LOCAL_ROUTED_CHUNK_ENV, "4096"))
    assert chunk > 0 and chunk % TILE == 0, f"{LOCAL_ROUTED_CHUNK_ENV} must be a positive multiple of 32: {chunk}"
    return chunk


def local_routed_min_tokens(default: int) -> int:
    """Rows from which a prefill call takes the local path: ``GEMMA4_MOE_LOCAL_ROUTED_MIN_TOKENS`` when set, else
    ``default`` (the dispatch path's threshold; shorter calls keep the decode / small-row paths)."""
    env = os.environ.get(LOCAL_ROUTED_MIN_TOKENS_ENV)
    if env is not None and env.strip() != "":
        return int(env)
    return int(default)


def local_routed_reason(
    mesh_shape,
    *,
    ep: int,
    tp: int,
    tp_axis: int,
    num_local_experts: int,
    num_experts: int,
    top_k: int,
    hidden: int,
    lane_sharded: bool = False,
    interleaved: bool = True,
):
    """None when the local routed prefill can run on this build, else why not. Pure host logic (unit-tested).

    Every chip must hold every token (the dense residual replicated: no lanes) and the decode copy's whole experts
    must partition the expert set over the chips of ONE line (ep 1, the tp group spanning the mesh), so the per-chip
    partials summed by the TP all-reduce cover every routed pair exactly once. The GeGLU op addresses the stacked
    decode copy by page id (DRAM-interleaved TILE)."""
    rows, cols = (int(d) for d in mesh_shape)
    if rows * cols < 2:
        return "a single chip: the experts run whole on it"
    if lane_sharded:
        return "lane-sharded rows: chips hold different tokens"
    if int(ep) != 1:
        return f"the decode copy is expert-parallel over the lanes (ep {ep})"
    if int(tp) != rows * cols or int(mesh_shape[int(tp_axis)]) != int(tp):
        return f"the tp group ({tp} chips on axis {tp_axis}) is not the whole {rows}x{cols} mesh"
    if int(num_local_experts) * int(tp) != int(num_experts):
        return f"{num_local_experts} experts per chip x tp {tp} != {num_experts}"
    if not LocalRoutedGather.supports(num_experts, num_local_experts, top_k, hidden):
        return f"the gather kernel does not take E {num_experts} / E_loc {num_local_experts} / K {top_k} / H {hidden}"
    if not interleaved:
        return "the decode copy is not DRAM-interleaved TILE"
    return None


def local_routed_reason_for(mesh_device, mesh_config, expert_weights, config):
    """``local_routed_reason`` of a mesh, its MeshConfig and the loaded decode copy (``ExpertWeights``)."""
    if expert_weights is None or not hasattr(mesh_device, "shape"):
        return "no decode copy on a mesh"
    from models.demos.gemma4.tt.experts.weights import _expert_parallel_factors

    _, _, _, tp_axis = _expert_parallel_factors(mesh_device, mesh_config)
    interleaved = all(
        w.layout == ttnn.TILE_LAYOUT
        and w.memory_config().buffer_type == ttnn.BufferType.DRAM
        and w.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        for w in (expert_weights.gate_up_proj, expert_weights.down_proj)
    )
    return local_routed_reason(
        tuple(mesh_device.shape),
        ep=expert_weights.ep,
        tp=expert_weights.tp,
        tp_axis=tp_axis,
        num_local_experts=expert_weights.num_local_experts,
        num_experts=config.num_experts,
        top_k=config.top_k,
        hidden=config.hidden_size,
        lane_sharded=bool(mesh_config is not None and getattr(mesh_config, "lane_sharded", False)),
        interleaved=interleaved,
    )


def chip_expert_bases(mesh_shape, tp_axis: int, num_local: int):
    """``[rows][cols]`` first global expert of every chip: the decode copy's placement (tp index along ``tp_axis``
    times ``E_loc``; experts/weights.py with ep 1)."""
    rows, cols = (int(d) for d in mesh_shape)
    return [[(r if int(tp_axis) == 0 else c) * int(num_local) for c in range(cols)] for r in range(rows)]


class _Plan:
    """The per-bucket ops: the gather sized for the bucket's rows and the GeGLU op whose per-expert capacity is the
    bucket's row count (a token names an expert at most once)."""

    def __init__(self, owner, T: int):
        self.T = int(T)
        self.gather = LocalRoutedGather(self.T, owner.E, owner.E_loc, owner.K, owner.H)
        self.geglu = GegluRoutedExpert.from_stacked(
            owner.mesh_device,
            owner.weights.gate_up_proj,
            owner.weights.down_proj,
            owner.idx_table,
            max_tokens=self.T,
        )


class Gemma4LocalRoutedExperts:
    """Routed experts for prefill on a whole-expert TP line: gather -> GeGLU (decode copy) -> untilize -> combine."""

    @staticmethod
    def unsupported_reason(mesh_device):
        """None when the kernels can run on ``mesh_device`` (the Blackhole-only GeGLU routed-expert op,
        ``GegluRoutedExpert.unsupported_reason``); the caller keeps another expert path with one warning."""
        return GegluRoutedExpert.unsupported_reason(mesh_device)

    def __init__(self, mesh_device, config, expert_weights, mesh_config=None, min_tokens: int = 128):
        reason = self.unsupported_reason(mesh_device) or local_routed_reason_for(
            mesh_device, mesh_config, expert_weights, config
        )
        assert reason is None, f"local routed prefill: {reason}"
        from models.demos.gemma4.tt.experts.weights import _expert_parallel_factors

        self.mesh_device = mesh_device
        self.weights = expert_weights
        self.E, self.K, self.H = int(config.num_experts), int(config.top_k), int(config.hidden_size)
        self.E_loc = int(expert_weights.num_local_experts)
        self._min_tokens = local_routed_min_tokens(min_tokens)
        _, _, _, tp_axis = _expert_parallel_factors(mesh_device, mesh_config)
        rows, cols = (int(d) for d in mesh_device.shape)
        bases = chip_expert_bases((rows, cols), tp_axis, self.E_loc)
        table = [[[b + j for j in range(self.E_loc)] for b in row] for row in bases]
        idx = ttnn.from_torch(
            torch.tensor(table, dtype=torch.int32),  # [rows, cols, E_loc]
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, 1)),
        )
        self.idx_table = ttnn.squeeze(ttnn.squeeze(idx, 0), 0)  # [E_loc] per chip: local slot -> global id
        self.untilize = LocalRoutedUntilize()
        self._plans: dict = {}
        logger.info(
            f"Gemma4LocalRoutedExperts: mesh {rows}x{cols} tp_axis {tp_axis}, {self.E_loc} experts/chip (the decode "
            f"copy), min {self._min_tokens} rows, chunk {local_routed_chunk()}"
        )

    def min_tokens(self) -> int:
        return self._min_tokens

    def _plan(self, T_real: int) -> _Plan:
        """The plan of a ``T_real``-row chunk (its row bucket's, built on first use)."""
        T = row_bucket(T_real)
        plan = self._plans.get(T)
        if plan is None:
            plan = self._plans[T] = _Plan(self, T)
        return plan

    def forward(self, hidden, routing):
        """``hidden`` [1, 1, S, H] TILE bf16 replicated; ``routing`` the dense [1, 1, S, E] matrix (final per-expert
        weights, zeros off the top-k) or the router's ``(values, indices)`` [1, 1, S, K] (bf16 / uint16 TILE).
        Returns this chip's partial [1, 1, S, H] TILE bf16 (the sum over the slots of its experts); the caller
        all-reduces over TP."""
        S = int(hidden.shape[2])
        chunk = local_routed_chunk()
        if S <= chunk:
            return self._forward_chunk(hidden, routing)
        outs = []
        dense = not isinstance(routing, tuple)
        for start in range(0, S, chunk):
            end = min(start + chunk, S)
            h = ttnn.slice(hidden, [0, 0, start, 0], [1, 1, end, self.H])
            parts = (routing,) if dense else tuple(routing)
            parts = tuple(ttnn.slice(t, [0, 0, start, 0], [1, 1, end, int(t.shape[-1])]) for t in parts)
            outs.append(self._forward_chunk(h, parts[0] if dense else parts))
            ttnn.deallocate(h)
            for t in parts:
                ttnn.deallocate(t)
        out = ttnn.concat(outs, dim=2)
        for o in outs:
            ttnn.deallocate(o)
        return out

    def _forward_chunk(self, hidden, routing):
        T_real = int(hidden.shape[2])
        plan = self._plan(T_real)
        owned = []
        if isinstance(routing, tuple):
            values, ids = routing
        else:
            values, ids = ttnn.topk(routing, k=self.K, dim=-1)  # [1, 1, S, K] bf16 / uint16 TILE
            owned += [values, ids]
        if ids.dtype != ttnn.uint16:
            ids = ttnn.typecast(ids, ttnn.uint16)
            owned.append(ids)
        if values.dtype != ttnn.bfloat16:
            values = ttnn.typecast(values, ttnn.bfloat16)
            owned.append(values)
        x_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)  # the caller's tensor when already ROW_MAJOR
        if x_rm.dtype != ttnn.bfloat16:
            x_tc = ttnn.typecast(x_rm, ttnn.bfloat16)
            if x_rm is not hidden:
                ttnn.deallocate(x_rm)
            x_rm = x_tc
        buffer, counts, offsets, row_index, weights = plan.gather.op(x_rm, ids, values, self.idx_table)
        if x_rm is not hidden:
            ttnn.deallocate(x_rm)
        for t in owned:
            ttnn.deallocate(t)
        expert_out = plan.geglu(buffer, counts, offsets)  # [M, H] bf8 TILE, the local regions written
        self.untilize.op(expert_out, counts, offsets, self.idx_table, buffer)  # real rows back into the buffer
        ttnn.deallocate(expert_out)
        y = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            buffer,
            weights,
            expert_dim=3,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            combine_row_index=row_index,
            skip_row=LocalRoutedGather.SKIP_ROW,
        )  # [1, 1, T, H] TILE bf16
        for t in (buffer, counts, offsets, row_index, weights):
            ttnn.deallocate(t)
        if plan.T != T_real:
            y_real = ttnn.slice(y, [0, 0, 0, 0], [1, 1, T_real, self.H])
            ttnn.deallocate(y)
            y = y_real
        return y
