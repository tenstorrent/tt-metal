# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 DSpark prefill seeding (beads 9.1 / 9.2; graph nodes N6, D1, D2).

Prefill leaves exactly one piece of DSpark state: each DSpark layer's window KV ring (``inference/model.py``
``DSparkBlock.forward`` / ``DSparkAttention.forward`` at ``start_pos == 0``)::

    tap_l   = mean over the hc copies of layer l's block input streams (after Engram), l = 37, 38, 39
    main_x  = main_norm(main_proj(concat(tap_37, tap_38, tap_39)))        FP8 GEMM: FP8 activation QDQ first
    kv_k    = QDQ_fp8(RoPE(kv_norm(wkv_k(main_x))))                      per DSpark layer k; own weights;
                                                                          RoPE theta 1e4 without YaRN
    ring_k[p % window] = kv_k[p]      for the last ``window`` positions p

Every step is row-local, so a chunked prefill seeds the rings per chunk (graph.md §4 rule 6): each chunk writes
its last ``min(length, window)`` real positions to their slots, and slots it does not reach keep the previous
chunk's rows. The ring is therefore the only cross-chunk state; no taps are carried, and positions at or beyond
the chunk's valid length write nothing (rule 8).

Layout. Streams are ``[1, 1, S/sp, hc * hidden/tp]`` fp32 per chip (its hidden slice of every copy, as in
``mhc.py``); a tap is ``[1, 1, S/sp, hidden/tp]`` bf16. The seeder gathers the taps over SP, keeps the
tile-aligned rows around the last ``window`` real positions (<= window + 32 rows), and runs main_proj
row-parallel over TP (each chip holds the rows of its hidden slice of every tap) followed by one TP all-reduce;
main_norm and the DSpark projections then run replicated. Rings are replicated ``[1, 1, window, head_dim]``
bf16 row-major tensors holding the reference's quantize-dequantized values (KV format stage 1).

RoPE tables are device tensors per chunk (``seed_rope``; the transformer uploads every chunk's before the chunk loop,
so a chunk's seeding writes nothing from the host) and cover the rope channels only: the nope channels pass
through unchanged, as the full-width identity (cos 1, sin 0) would give them.

Weights are the exact bf16 dequantization of the FP8 checkpoint tensors (bf16 rather than bfp8: the GEMMs see
at most window + 32 rows once per chunk, so weight precision costs little and keeps them exact).
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp8_qdq
from models.demos.deepseek_v3_d_p.tt.v41.rope import cos_sin

SP_AXIS = 0
TILE = 32


class TtV41DSpark(LightweightModule):
    def __init__(self, mesh_device, config, weights: dict):
        """``weights`` (torch, checkpoint ``[out, in]`` orientation, FP8 tensors dequantized): ``main_proj``
        ``[hidden, n_taps * hidden]``, ``main_norm`` ``[hidden]``, and ``layers``: one dict per DSpark layer with
        ``wkv`` ``[head_dim, hidden]`` and ``kv_norm`` ``[head_dim]``."""
        self.mesh_device, self.config = mesh_device, config
        self.sp, self.tp = mesh_device.shape[SP_AXIS], mesh_device.shape[1]
        self.hidden, self.hc = config.EMB_SIZE, config.HC_MULT
        self.n_taps = len(config.DSPARK_TARGET_LAYER_IDS)
        self.head_dim, self.rope_dim = config.HEAD_DIM, config.QK_ROPE_HEAD_DIM
        self.window, self.eps = config.SLIDING_WINDOW, config.RMS_NORM_EPS
        assert len(weights["layers"]) == config.NUM_DSPARK_LAYERS
        assert self.hidden % (self.tp * TILE) == 0, "each chip's hidden slice of a tap must be whole tiles"
        self.ccl = V41Collectives(mesh_device)
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        shape = tuple(mesh_device.shape)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        def upload(t, mapper=replicate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, device=mesh_device, dtype=dtype, layout=layout, mesh_mapper=mapper)

        def row(w):
            return upload(w.detach().to(torch.bfloat16).reshape(1, 1, 1, -1))

        # main_proj input index (tap, chip, j) -> chip-major (chip, tap, j), so that TP shard t of the weight's rows
        # multiplies the local concatenation [tap_0 slice t | tap_1 slice t | ...].
        w = weights["main_proj"].detach().to(torch.bfloat16).T
        assert tuple(w.shape) == (self.n_taps * self.hidden, self.hidden)
        w = w.reshape(self.n_taps, self.tp, self.hidden // self.tp, self.hidden).transpose(0, 1)
        self.main_proj = upload(
            w.reshape(1, 1, self.n_taps * self.hidden, self.hidden).contiguous(),
            ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(None, 2)),
        )
        self.main_norm = row(weights["main_norm"])
        self.layers = [
            {
                "wkv": upload(l["wkv"].detach().to(torch.bfloat16).T.contiguous()[None, None]),
                "kv_norm": row(l["kv_norm"]),
            }
            for l in weights["layers"]
        ]
        # (x @ pair_swap)[2i] = -x[2i+1], [2i+1] = x[2i] on the rope channels
        swap = torch.zeros(self.rope_dim, self.rope_dim)
        for i in range(0, self.rope_dim, 2):
            swap[i + 1, i], swap[i, i + 1] = -1.0, 1.0
        self.pair_swap = upload(swap[None, None])

    # --- N6 --------------------------------------------------------------------------------------------------
    def tap(self, streams):
        """Block input streams ``[1, 1, S/sp, hc * hidden/tp]`` fp32 -> tap ``[1, 1, S/sp, hidden/tp]`` bf16,
        the mean over the hc copies."""
        rows, width = streams.shape[2], streams.shape[3] // self.hc
        total = None
        for c in range(self.hc):
            part = ttnn.slice(streams, [0, 0, 0, c * width], [1, 1, rows, (c + 1) * width])
            total = part if total is None else ttnn.add(total, part)
        return ttnn.typecast(ttnn.multiply(total, 1.0 / self.hc), ttnn.bfloat16)

    # --- D1 --------------------------------------------------------------------------------------------------
    def _seed_rows(self, length: int) -> tuple[int, int, int]:
        """(lo, a, b): the real positions [lo, length) a chunk seeds and the tile-aligned rows [a, b) around them."""
        lo = max(0, length - self.window)
        return lo, lo // TILE * TILE, -(-length // TILE) * TILE

    def project(self, taps: list, length: int):
        """``taps`` (one per target layer, in order) of a chunk with ``length`` valid tokens -> (main_x, a):
        ``main_x`` ``[1, 1, b - a, hidden]`` bf16 (replicated) holds main_x of chunk rows [a, b), which cover the
        last ``min(length, window)`` valid rows."""
        assert len(taps) == self.n_taps
        padded = taps[0].shape[2] * self.sp
        assert 0 < length <= padded, f"valid length {length} outside the chunk's {padded} rows"
        _, a, b = self._seed_rows(length)
        x = self._sp_all_gather(ttnn.concat(taps, dim=3))
        x = fp8_qdq(ttnn.slice(x, [0, 0, a, 0], [1, 1, b, x.shape[3]]))
        y = ttnn.linear(x, self.main_proj, dtype=ttnn.float32, compute_kernel_config=self.compute_kernel_config)
        y = ttnn.typecast(self.ccl.tp_all_reduce(y), ttnn.bfloat16)  # the reference GEMM rounds once to bf16
        return ttnn.rms_norm(y, weight=self.main_norm, epsilon=self.eps), a

    # --- D2 --------------------------------------------------------------------------------------------------
    def new_rings(self) -> list:
        """Fresh (all-zero) DSpark window rings, one per DSpark layer."""
        return [
            ttnn.from_torch(
                torch.zeros(1, 1, self.window, self.head_dim),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for _ in self.layers
        ]

    def seed(self, taps: list, start: int, length: int, rings: list, rope: tuple | None = None) -> None:
        """Write the DSpark window KV of the chunk's last ``min(length, window)`` real positions (absolute
        ``start + p``) into ``rings`` slot ``(start + p) % window``, in place. Call once per chunk, in order.
        ``rope``: this chunk's ``seed_rope(start, length)``, uploaded ahead (uploaded here when None)."""
        assert len(rings) == len(self.layers)
        main_x, a = self.project(taps, length)
        lo, _, b = self._seed_rows(length)
        cos, sin = rope if rope is not None else self.seed_rope(start, length)
        assert cos.shape[2] == b - a, f"RoPE table of {cos.shape[2]} rows for seed rows [{a}, {b})"
        x = fp8_qdq(main_x)
        for layer, ring in zip(self.layers, rings):
            kv = ttnn.linear(x, layer["wkv"], compute_kernel_config=self.compute_kernel_config)
            kv = self._rope(ttnn.rms_norm(kv, weight=layer["kv_norm"], epsilon=self.eps), cos, sin)
            kv = ttnn.to_layout(fp8_qdq(kv), ttnn.ROW_MAJOR_LAYOUT)
            self._write_ring(ring, ttnn.slice(kv, [0, 0, lo - a, 0], [1, 1, length - a, self.head_dim]), start + lo)

    def seed_rope(self, start: int, length: int) -> tuple:
        """Replicated fp32 cos/sin ``[1, 1, b - a, rope_dim]`` (ratio-0 table) of the seed rows [a, b) of the chunk
        at ``start`` with ``length`` valid tokens (``_seed_rows``); depends on the chunk's position only."""
        _, a, b = self._seed_rows(length)
        return tuple(
            ttnn.from_torch(
                t,
                device=self.mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for t in cos_sin(self.config, False, torch.arange(start + a, start + b))
        )

    def _rope(self, kv, cos, sin):
        """Interleaved RoPE of the rope channels in fp32 like the reference (``x * cos + swap(x) * sin``), rounded
        once to bf16; the nope channels pass through. The pair swap is a ±1 permutation GEMM of bf16 values into
        fp32, hence exact."""
        rows, nope = kv.shape[2], self.head_dim - self.rope_dim
        pe = ttnn.slice(kv, [0, 0, 0, nope], [1, 1, rows, self.head_dim])
        swapped = ttnn.linear(pe, self.pair_swap, dtype=ttnn.float32, compute_kernel_config=self.compute_kernel_config)
        out = ttnn.add(ttnn.multiply(ttnn.typecast(pe, ttnn.float32), cos), ttnn.multiply(swapped, sin))
        return ttnn.concat([ttnn.slice(kv, [0, 0, 0, 0], [1, 1, rows, nope]), ttnn.typecast(out, ttnn.bfloat16)], dim=3)

    def _write_ring(self, ring, rows, first_position: int) -> None:
        """Write ``rows`` [1, 1, n, head_dim] (n <= window) of consecutive positions from ``first_position`` into
        their ring slots: at most two contiguous slot ranges (the second wraps to slot 0)."""
        n, slot = rows.shape[2], first_position % self.window
        head = min(n, self.window - slot)
        segments = [(0, head, slot)] + ([(head, n, 0)] if n > head else [])
        for begin, end, dst in segments:
            part = rows if (begin, end) == (0, n) else ttnn.slice(rows, [0, 0, begin, 0], [1, 1, end, self.head_dim])
            ttnn.experimental.slice_write(
                part, ring, [0, 0, dst, 0], [1, 1, dst + end - begin, self.head_dim], [1, 1, 1, 1]
            )

    def _sp_all_gather(self, t):
        """[1, 1, rows/sp, W] per SP rank (contiguous token order) -> [1, 1, rows, W] on every rank."""
        if self.sp == 1:
            return t
        tt_ccl = self.ccl.tt_ccl
        return ttnn.experimental.all_gather_async(
            t,
            dim=2,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=SP_AXIS),
            num_links=self.ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl.sp_topology,
            cluster_axis=SP_AXIS,
        )
