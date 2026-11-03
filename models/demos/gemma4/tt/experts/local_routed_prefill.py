# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local token-sorted routed-expert prefill for the whole-expert TP shapes (no collective inside the experts).

On the 4-chip line every chip already holds every token of the call (the dense TP all-reduce leaves the residual
replicated) and 32 whole experts. The shipped prefill (``experts/prefill.py``) runs all 32 local experts on every
token through masked-dense ``sparse_matmul`` chunks: 16x the routed (token, expert) pairs (14x with the 32-token
tile mask), 86% of the 2048-token device time on P150x4 (A7 profile, 2026-10-23). This module computes only the
routed pairs, all on device:

1. ``ttnn.sort`` the ``S * K`` (token, slot) pairs by the fp32 key ``gid * S * K + t * K + k`` (unique, so the
   sorted keys themselves carry every row's expert, token and pair index exactly: no integer typecast of the sort's
   index output) -> ``pos[b]`` = pair at sorted row ``b``, expert-major, tokens ascending inside an expert;
2. ``masked_bincount`` the per-expert pair counts (all experts for the region offsets, this chip's experts for the
   kernel's counts); the kernel addresses regions in whole tile-rows, so every region starts at the exclusive prefix
   sum (an fp32 triangular matmul) of the counts rounded up to 32, and a sorted row ``b`` of expert ``g`` lands at
   buffer row ``b + padshift[g]`` (``padshift`` = aligned minus raw prefix sum, gathered per row);
3. ``ttnn.embedding`` gathers the tokens' hidden rows into the expert-major buffer ``[S * K + 32 * E, H]`` (the pad
   rows read token 0 and are never consumed);
4. the model-local GeGLU routed-expert kernel (``custom_ops/geglu_routed_expert``, the dispatch stack's expert op)
   runs each local expert over its contiguous region of the buffer;
5. the inverse permutation (``ttnn.scatter`` of an iota) and a second ``ttnn.embedding`` bring the expert rows back
   per (token, slot); ``deepseek_prefill.post_combine_reduce`` applies the routing weights (zeroed on the slots of
   other chips' experts, whose rows this chip never wrote) and sums the slots -> this chip's partial ``[S, H]``.

The caller (``Gemma4Experts``) all-reduces the partials over TP exactly as for the sparse_matmul path. No fabric op
runs inside the experts, no 4096-token dispatch padding, and the buffers follow the call length. Opt-in with
``GEMMA4_MOE_LOCAL_ROUTED_PREFILL=1`` for calls of at least ``GEMMA4_MOE_LOCAL_ROUTED_MIN_TOKENS`` rows (default 256;
``masked_bincount`` needs a multiple of 64) on the tp-only shapes (ep == 1, no lanes); shorter calls (decode frames,
short prefills) keep the sparse_matmul path. Calls longer than ``GEMMA4_MOE_LOCAL_ROUTED_CHUNK`` (default 4096) are
processed in chunks.

Expert weights are loaded through the DeepSeek ``TtRoutedExpert`` loader (per-expert [H, I] / [I, H] TILE tensors,
DRAM ND-sharded on Blackhole, its own cache files ``layer_<i>.local_routed_<rows>x<cols>.local_<e>_{gate,up,down}``),
on the ``ExpertMapping`` column-major placement: chip (r, c) holds global experts ``[(c * rows + r) * E_loc, ...)``,
which is the ``ExpertWeights`` layout of a line in either orientation.
"""

import glob
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.custom_ops.geglu_routed_expert import GegluRoutedExpert
from models.demos.gemma4.tt.experts.local_routed_index import local_expert_mask

TILE = 32


def local_routed_prefill_enabled() -> bool:
    return os.environ.get("GEMMA4_MOE_LOCAL_ROUTED_PREFILL", "0") == "1"


def local_routed_min_tokens() -> int:
    return int(os.environ.get("GEMMA4_MOE_LOCAL_ROUTED_MIN_TOKENS", "256"))


def local_routed_chunk() -> int:
    return int(os.environ.get("GEMMA4_MOE_LOCAL_ROUTED_CHUNK", "4096"))


def hf_expert_list(gate_up_proj, down_proj):
    """HF fused gate_up_proj [E, 2I, H] (gate rows first) + down_proj [E, H, I] -> list of E (out, in) dicts for the
    TtRoutedExpert loader (which transposes to the device [K, N] orientation)."""
    E, two_i, _ = gate_up_proj.shape
    I = two_i // 2
    return [
        {
            "gate_proj": gate_up_proj[e, :I, :].float(),
            "up_proj": gate_up_proj[e, I:, :].float(),
            "down_proj": down_proj[e].float(),
        }
        for e in range(E)
    ]


# One expert-output buffer per (mesh, rows, hidden) shared by every layer: the layers run sequentially and each
# converts the kernel output before the next layer overwrites it. Rows the kernel never wrote stay zero (filled at
# creation); rows written by an earlier call hold finite stale values that the zeroed slot weights never read.
# The cache owns every buffer it hands out for the life of the process (it is the ``output`` tensor of the GeGLU
# generic op, so nothing may free it under an enqueued call) and keys it by the mesh's own id (``MeshDevice.id()``, a
# process-wide counter that is never reused), never by ``id(mesh_device)``: a mesh closed and reopened between two
# calls (every pytest case) can reuse the Python object's address, and the entry then names a buffer whose mesh is
# gone -- ``ttnn.generic_op`` raised "Can't get device from mesh buffer, already deallocated" on the 2026-10-25 CI
# (test_experts_local_routed_chunked after test_experts_local_routed[prefill_128]: both use the 5120-row buffer).
_OUT_BUFFERS: dict = {}


def _out_buffer_key(mesh_device, rows: int, hidden: int) -> tuple:
    """Cache key of the shared expert-output buffer: the mesh's own id (never the Python ``id()``), rows, hidden."""
    return (int(mesh_device.id()), int(rows), int(hidden))


class Gemma4LocalRoutedExperts:
    """Routed experts for prefill on a whole-expert TP mesh: sort -> gather -> GeGLU kernel -> gather -> weighted sum."""

    @staticmethod
    def unsupported_reason(mesh_device):
        """None when this path can run on ``mesh_device``, else why not (the Blackhole-only GeGLU routed-expert op,
        ``GegluRoutedExpert.unsupported_reason``); the caller falls back to sparse_matmul with one warning."""
        return GegluRoutedExpert.unsupported_reason(mesh_device)

    def __init__(
        self,
        mesh_device,
        config,
        expert_weights,  # list of E dicts {gate_proj [I,H], up_proj [I,H], down_proj [H,I]} (HF (out,in)), or None
        mesh_config=None,
        tensor_cache_path=None,
        layer_idx=0,
        weights_dtype=ttnn.bfloat8_b,
    ):
        from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

        reason = self.unsupported_reason(mesh_device)
        assert reason is None, f"local routed prefill: {reason}"
        self.mesh_device = mesh_device
        rows, cols = (int(d) for d in mesh_device.shape)
        self.rows, self.cols = rows, cols
        self.E = int(config.num_experts)
        self.K = int(config.top_k)
        self.H = int(config.hidden_size)
        self.I = int(config.moe_intermediate_size)
        n_chips = rows * cols
        assert n_chips > 1, "the local routed prefill serves the multi-chip whole-expert shapes"
        assert self.E % n_chips == 0, f"{self.E} experts over {n_chips} chips"
        self.E_loc = self.E // n_chips
        ep = int(getattr(mesh_config, "ep", 1) or 1) if mesh_config is not None else 1
        assert ep == 1, "the local routed prefill serves the whole-expert tp-only shapes (ep == 1)"
        # chip (r, c) holds experts [base, base + E_loc): the ExpertMapping column-major order the TtRoutedExpert
        # loader distributes (group = column, chip = row), equal to the ExpertWeights layout on a line
        bases = torch.tensor(
            [[(c * rows + r) * self.E_loc for c in range(cols)] for r in range(rows)], dtype=torch.int64
        )
        self._bases = bases
        mapper2d = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, 1))
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        def _rm(t, dtype):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper2d if t.shape[:2] == (rows, cols) else replicate,
            )

        # local slot -> global expert id, [E_loc] uint32 per chip
        idx_table = (bases.reshape(rows, cols, 1) + torch.arange(self.E_loc).reshape(1, 1, -1)).to(torch.int32)
        self.idx_table = ttnn.squeeze(ttnn.squeeze(_rm(idx_table, ttnn.uint32), 0), 0)
        # masked_bincount masks [1, E + 1] int32: this chip's experts / every expert (the sentinel column is -1)
        mask_local = torch.stack(
            [
                torch.stack([local_expert_mask(self.E, int(bases[r, c]), self.E_loc) for c in range(cols)])
                for r in range(rows)
            ]
        )
        self.mask_local = ttnn.reshape(_rm(mask_local, ttnn.int32), (1, self.E + 1))
        mask_all = torch.zeros((1, 1, self.E + 1), dtype=torch.int32)
        mask_all[..., self.E] = -1
        self.mask_all = ttnn.reshape(_rm(mask_all, ttnn.int32), (1, self.E + 1))
        # strictly-upper-triangular ones: counts @ U = exclusive prefix sum (fp32, exact below 2^24)
        self.upper = ttnn.from_torch(
            torch.triu(torch.ones(self.E, self.E), diagonal=1).reshape(1, 1, self.E, self.E),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.fp32_acc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.weights_dtype = weights_dtype
        self.cache_prefix = f"layer_{layer_idx}.local_routed_{rows}x{cols}"
        cache_path = Path(tensor_cache_path) if tensor_cache_path else None
        if expert_weights is None:
            # a warm-cache placeholder state dict: refuse to let the loader persist dataless tensors on a miss
            # (weights.py::_refuse_placeholder_on_cache_miss)
            if cache_path is None or not self._cache_complete(cache_path):
                raise RuntimeError(
                    f"local routed prefill: expert cache miss for {self.cache_prefix} under {cache_path} with a "
                    "placeholder state dict; cold-load the checkpoint (the flag is part of the warm-cache identity)"
                )
        self.routed_expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=self.E_loc,
            global_expert_idx_table=self.idx_table,
            emb_dim=self.H,
            hidden_dim=self.I,
            max_tokens=local_routed_chunk(),
            torch_weights=expert_weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=weights_dtype,
            weight_cache_path=cache_path,
            cache_name_prefix=self.cache_prefix,
            # loader and cache only: the GeGLU op below runs the experts (Silu is the value the loader validates)
            activation=ttnn.RoutedExpertActivation.Silu,
            hybrid_token_threshold=None,
        )
        self._ops: dict = {}  # S -> GegluRoutedExpert (max_tokens = S: every token may pick the same expert)
        self._consts: dict = {}  # S -> iota / zeros / per-chip base tensors
        logger.info(
            f"Gemma4LocalRoutedExperts: mesh {rows}x{cols}, {self.E_loc} experts/chip, min {local_routed_min_tokens()} "
            f"rows, chunk {local_routed_chunk()}, weights {weights_dtype}"
        )

    @staticmethod
    def min_tokens() -> int:
        return local_routed_min_tokens()

    def _cache_complete(self, cache_path: Path) -> bool:
        name = self.weights_dtype.name
        for e in range(self.E_loc):
            for proj in ("gate", "up", "down"):
                if not glob.glob(str(cache_path / f"{self.cache_prefix}.local_{e}_{proj}_dtype_{name}_*.tensorbin")):
                    return False
        return True

    # ------------------------------------------------------------------------------------------------ per-length state
    def _op_for(self, S: int) -> GegluRoutedExpert:
        if S not in self._ops:
            re = self.routed_expert
            self._ops[S] = GegluRoutedExpert(
                self.mesh_device, re.gate_projs, re.up_projs, re.down_projs, self.idx_table, max_tokens=S
            )
        return self._ops[S]

    def _consts_for(self, S: int) -> dict:
        if S not in self._consts:
            N = S * self.K
            dev, dram = self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
            replicate = ttnn.ReplicateTensorToMesh(dev)
            mapper2d = ttnn.ShardTensor2dMesh(dev, mesh_shape=dev.shape, dims=(0, 1))
            assert N < (1 << 24), f"sort keys must stay exact in fp32: {S} rows x {self.K} slots"

            def _f32(t):
                return ttnn.from_torch(
                    t.to(torch.float32),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=dev,
                    memory_config=dram,
                    mesh_mapper=replicate,
                )

            M = N + TILE * self.E  # buffer rows: every expert region rounded up to a whole tile
            iota = _f32(torch.arange(N).reshape(1, N))  # sorted row index b
            # scatter targets: ROW_MAJOR fp32 (ttnn.scatter refuses fp32 TILE tensors and int32 TILE rows > 256)
            zeros = ttnn.to_layout(_f32(torch.zeros(1, N)), ttnn.ROW_MAJOR_LAYOUT)
            zeros_buf = ttnn.to_layout(_f32(torch.zeros(1, M)), ttnn.ROW_MAJOR_LAYOUT)
            pair = _f32(torch.arange(N).reshape(1, 1, S, self.K))  # pair index t * K + k per (token, slot)
            # this chip's first expert id, [1, 1, S, K] (same shape as the slot ids: no broadcast)
            base = ttnn.from_torch(
                self._bases.reshape(self.rows, self.cols, 1, 1)
                .expand(self.rows, self.cols, S, self.K)
                .to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=dram,
                mesh_mapper=mapper2d,
            )
            self._consts[S] = {"iota": iota, "zeros": zeros, "zeros_buf": zeros_buf, "pair": pair, "base": base, "M": M}
        return self._consts[S]

    def _out_buffer(self, N: int):
        key = _out_buffer_key(self.mesh_device, N, self.H)
        if key not in _OUT_BUFFERS:
            z16 = ttnn.zeros(
                (N, self.H),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _OUT_BUFFERS[key] = ttnn.typecast(z16, ttnn.bfloat8_b)
            ttnn.deallocate(z16)
        return _OUT_BUFFERS[key]

    def _exclusive_cumsum_f32(self, counts_f):
        """[1, 1, 1, E] fp32 TILE counts -> same shape, exclusive prefix sum (exact below 2^24)."""
        return ttnn.matmul(
            counts_f,
            self.upper,
            dtype=ttnn.float32,
            compute_kernel_config=self.fp32_acc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _region_layout(self, counts):
        """[E] uint32 ROW_MAJOR pair counts of every expert -> (aligned region offsets [1, E] uint32 ROW_MAJOR for the
        kernel, padshift [1, 1, 1, E] fp32 TILE = aligned minus raw exclusive prefix sum)."""
        t = ttnn.to_layout(ttnn.reshape(counts, (1, 1, 1, self.E)), ttnn.TILE_LAYOUT)
        f = ttnn.typecast(t, ttnn.float32)
        ttnn.deallocate(t)
        raw = self._exclusive_cumsum_f32(f)
        aligned_counts = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(f, TILE - 0.5), 1.0 / TILE)), float(TILE))
        ttnn.deallocate(f)
        aligned = self._exclusive_cumsum_f32(aligned_counts)
        ttnn.deallocate(aligned_counts)
        padshift = ttnn.subtract(aligned, raw)
        ttnn.deallocate(raw)
        u = ttnn.typecast(aligned, ttnn.uint32)
        ttnn.deallocate(aligned)
        offsets = ttnn.reshape(ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT), (1, self.E))
        ttnn.deallocate(u)
        return offsets, padshift

    # ------------------------------------------------------------------------------------------------ forward
    def forward(self, hidden, routing):
        """``hidden`` [1, 1, S, H] TILE bf16 replicated; ``routing`` either the dense [1, 1, S, E] matrix (final
        per-expert weights, zeros off the top-k) or the router's ``(values, indices)`` [1, 1, S, K]. Returns this
        chip's partial [1, 1, S, H] TILE bf16 (sum over the slots of its experts); the caller all-reduces over TP."""
        S = int(hidden.shape[2])
        chunk = local_routed_chunk()
        if S <= chunk:
            return self._forward_chunk(hidden, routing)
        outs = []
        dense = not isinstance(routing, tuple)
        for start in range(0, S, chunk):
            end = min(start + chunk, S)
            h = ttnn.slice(hidden, [0, 0, start, 0], [1, 1, end, self.H])
            if dense:
                r = ttnn.slice(routing, [0, 0, start, 0], [1, 1, end, self.E])
                parts = (r,)
            else:
                parts = tuple(ttnn.slice(t, [0, 0, start, 0], [1, 1, end, int(t.shape[-1])]) for t in routing)
            outs.append(self._forward_chunk(h, parts[0] if dense else parts))
            ttnn.deallocate(h)
            for t in parts:
                ttnn.deallocate(t)
        out = ttnn.concat(outs, dim=2)
        for o in outs:
            ttnn.deallocate(o)
        return out

    def _forward_chunk(self, hidden, routing):
        S, K, E, H = int(hidden.shape[2]), self.K, self.E, self.H
        assert S % 64 == 0, f"local routed prefill needs a multiple of 64 rows, got {S}"
        N = S * K
        owned = []
        if isinstance(routing, tuple):
            values, idx = routing
        else:
            values, idx = ttnn.topk(routing, k=K, dim=-1)  # [1, 1, S, K] bf16 / uint16 TILE
            owned += [values, idx]
        c = self._consts_for(S)
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        dram = ttnn.DRAM_MEMORY_CONFIG

        # per-expert pair counts: every expert (region offsets) and this chip's experts (the kernel's counts)
        idx2 = ttnn.reshape(idx, (S, K))
        full_counts = ttnn.experimental.deepseek_prefill.masked_bincount(idx2, self.mask_all, E, K)
        local_counts = ttnn.experimental.deepseek_prefill.masked_bincount(idx2, self.mask_local, E, K)
        offsets, padshift = self._region_layout(full_counts)
        ttnn.deallocate(full_counts)

        # (token, slot) pairs sorted by global expert id through the unique fp32 key gid * N + (t * K + k): the sorted
        # keys give every row's pair index exactly (floor((key + 0.5) / N) is the expert, the rest the pair)
        idx_b = ttnn.typecast(idx, ttnn.bfloat16)  # exact: ids < 256
        idx_f = ttnn.typecast(idx_b, ttnn.float32)
        key = ttnn.add(ttnn.multiply(idx_f, float(N)), c["pair"])  # [1, 1, S, K] fp32
        ttnn.deallocate(idx_f)
        key_rm = ttnn.reshape(ttnn.to_layout(key, rm), (1, N))
        ttnn.deallocate(key)
        sorted_keys, _pos = ttnn.sort(key_rm, dim=-1)
        ttnn.deallocate(_pos)
        ttnn.deallocate(key_rm)
        sk = ttnn.to_layout(sorted_keys, tile)  # [1, N] fp32
        ttnn.deallocate(sorted_keys)
        gid_f = ttnn.floor(ttnn.multiply(ttnn.add(sk, 0.5), 1.0 / N))
        pos_f = ttnn.subtract(sk, ttnn.multiply(gid_f, float(N)))  # pair index t * K + k of sorted row b (exact)
        ttnn.deallocate(sk)
        tok_f = ttnn.floor(ttnn.multiply(ttnn.add(pos_f, 0.5), 1.0 / K))  # token of sorted row b
        # buffer row of sorted row b: b + padshift[gid] (tile-aligned expert regions)
        gid_u = ttnn.reshape(ttnn.typecast(gid_f, ttnn.uint32), (1, 1, 1, N))
        ttnn.deallocate(gid_f)
        shift = ttnn.reshape(ttnn.gather(padshift, -1, gid_u), (1, N))
        ttnn.deallocate(gid_u)
        ttnn.deallocate(padshift)
        dest_f = ttnn.add(c["iota"], shift)
        ttnn.deallocate(shift)
        dest_u = ttnn.typecast(dest_f, ttnn.uint32)
        dest_u_rm = ttnn.to_layout(dest_u, rm)
        ttnn.deallocate(dest_u)
        dest_rm = ttnn.to_layout(dest_f, rm)
        ttnn.deallocate(dest_f)
        tok_rm = ttnn.to_layout(tok_f, rm)
        ttnn.deallocate(tok_f)
        # token id per buffer row (pad rows keep token 0: valid, never consumed); fp32 ROW_MAJOR scatter
        tok_buf = ttnn.scatter(c["zeros_buf"], -1, dest_u_rm, tok_rm)  # [1, M] fp32 RM
        ttnn.deallocate(tok_rm)
        tok_t = ttnn.to_layout(tok_buf, tile)
        ttnn.deallocate(tok_buf)
        tok_u = ttnn.to_layout(ttnn.typecast(tok_t, ttnn.uint32), rm)  # [1, M]
        ttnn.deallocate(tok_t)

        # expert-major token buffer [M, H] (row-major bf16, tilized to bf8 in the kernel)
        M = c["M"]
        x_rm = ttnn.to_layout(ttnn.reshape(hidden, (S, H)), rm)
        xbuf = ttnn.reshape(ttnn.embedding(tok_u, x_rm, layout=rm, dtype=ttnn.bfloat16, memory_config=dram), (M, H))
        ttnn.deallocate(x_rm)
        ttnn.deallocate(tok_u)
        out = self._op_for(S)(xbuf, local_counts, offsets, output=self._out_buffer(M))
        ttnn.deallocate(xbuf)
        ttnn.deallocate(local_counts)
        ttnn.deallocate(offsets)

        # rows back to (token, slot) order: inv[p] = buffer row of pair p (fp32 ROW_MAJOR scatter, exact below 2^24)
        pos_u = ttnn.typecast(pos_f, ttnn.uint32)
        ttnn.deallocate(pos_f)
        pos_u_rm = ttnn.to_layout(pos_u, rm)
        ttnn.deallocate(pos_u)
        inv_f = ttnn.scatter(c["zeros"], -1, pos_u_rm, dest_rm)  # [1, N] fp32 RM
        ttnn.deallocate(pos_u_rm)
        ttnn.deallocate(dest_u_rm)
        ttnn.deallocate(dest_rm)
        inv_t = ttnn.to_layout(inv_f, tile)
        ttnn.deallocate(inv_f)
        inv_u = ttnn.typecast(inv_t, ttnn.uint32)
        ttnn.deallocate(inv_t)
        inv_rm = ttnn.to_layout(inv_u, rm)
        ttnn.deallocate(inv_u)
        out16 = ttnn.typecast(out, ttnn.bfloat16)
        out_rm = ttnn.to_layout(out16, rm)
        ttnn.deallocate(out16)
        comb = ttnn.reshape(
            ttnn.embedding(inv_rm, out_rm, layout=rm, dtype=ttnn.bfloat16, memory_config=dram), (1, 1, S, K, H)
        )
        ttnn.deallocate(out_rm)
        ttnn.deallocate(inv_rm)

        # slot weights, zero on the slots of other chips' experts (their rows were never written for this call)
        d = ttnn.subtract(idx_b, c["base"])
        ttnn.deallocate(idx_b)
        is_local = ttnn.multiply(ttnn.ge(d, 0.0), ttnn.lt(d, float(self.E_loc)))
        ttnn.deallocate(d)
        w = ttnn.multiply(values, is_local)
        ttnn.deallocate(is_local)
        w_rm = ttnn.reshape(ttnn.to_layout(w, rm), (1, 1, S, K, 1))
        ttnn.deallocate(w)
        y = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            comb, w_rm, None, None, expert_dim=3, output_memory_config=dram
        )  # [1, 1, S, H] TILE bf16
        ttnn.deallocate(comb)
        ttnn.deallocate(w_rm)
        for t in owned:
            ttnn.deallocate(t)
        return y
