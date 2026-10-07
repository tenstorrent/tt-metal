# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash model on the 4x8 mesh: embedding (replicated table) -> N decoder layers -> final RMSNorm -> lm_head (vocab sharded over mesh columns).

Rows of the mesh are independent user groups (DP); every row runs the same program on its own tokens. Only the first n_layers layers are built for
partial-model smokes (n_layers=48 is the full model).
"""
import time

import torch

import ttnn
from models.demos.blackhole.mimo_v26.reference.loader import Ckpt
from models.demos.blackhole.mimo_v26.reference.loader import is_swa as Ld_is_swa
from models.demos.blackhole.mimo_v26.tt import weights as W
from models.demos.blackhole.mimo_v26.tt.layer import EPS, MiMoDecoderLayer, _k_norm

HIDDEN, VOCAB = W.HIDDEN, W.VOCAB


class MiMoModel:
    def __init__(
        self,
        mesh,
        n_layers=48,
        max_pos=512,
        users=32,
        ring_blocks=10,
        ck=None,
        wc=None,
        expert_op=None,
        log=print,
        layout=None,
    ):
        if layout is not None:
            max_pos, ring_blocks, users = layout.max_pos, layout.ring_blocks, 32
        self.mesh, self.n_layers, self.users, self.layout = mesh, n_layers, users, layout
        self.ck = ck or Ckpt()
        self.wc = wc or W.WeightCache(mesh)
        rep = W.rep_mapper(mesh)
        t0 = time.time()
        self.embed = ttnn.from_torch(
            self.ck.get("model.embed_tokens.weight").bfloat16(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.w_norm = ttnn.from_torch(
            W.norm_weight(self.ck, "model.norm.weight"),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        lm = self.ck.get("lm_head.weight").bfloat16().t().contiguous().reshape(1, 1, HIDDEN, VOCAB)  # [4096, V]
        self.lm_head = ttnn.from_torch(
            lm,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=W.col_mapper(mesh, 3),
        )
        del lm
        log(f"MODEL embed+norm+lm_head loaded in {time.time() - t0:.1f}s")
        self.layers = []
        for L in range(n_layers):
            t0 = time.time()
            self.layers.append(
                MiMoDecoderLayer(
                    mesh,
                    L,
                    self.ck,
                    self.wc,
                    max_pos,
                    users=users,
                    ring_blocks=ring_blocks,
                    expert_op=expert_op,
                    page_table=(None if layout is None else layout.tables()["swa" if Ld_is_swa(L) else "ga"]),
                )
            )
            log(f"MODEL layer {L} built in {time.time() - t0:.1f}s cache={self.wc.stats}")

    def embed_tokens(self, tokens):
        """tokens torch int64 [rows, N] -> device [1,1,N,4096] bf16 tile per row."""
        rows, N = tokens.shape
        tt = ttnn.from_torch(
            tokens.to(torch.int32),
            device=self.mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, self.mesh.shape, dims=(0, None)),
        )
        e = ttnn.embedding(tt, self.embed, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(tt)
        return ttnn.reshape(e, (1, 1, N, HIDDEN))

    def final_logits(self, h_last):
        """h_last [1,1,32,4096] bf16 tile (padded token rows). Returns host logits [rows, 32, VOCAB] float."""
        x = ttnn.rms_norm(h_last, epsilon=EPS, weight=self.w_norm, compute_kernel_config=_k_norm())
        lg = ttnn.linear(
            x,
            self.lm_head,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            ),
        )
        out = ttnn.to_torch(
            lg, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, dims=(0, 3), mesh_shape=tuple(self.mesh.shape))
        ).float()  # [rows,1,32,V]
        ttnn.deallocate(lg)
        ttnn.deallocate(x)
        return out[:, 0]

    def upload_rows(self, x):
        """torch [rows, T, 4096] -> device [1,1,T,4096] bf16 tile per row (T padded to a multiple of 32 by the caller)."""
        rows, T, _ = x.shape
        return ttnn.from_torch(
            x.reshape(rows, 1, T, HIDDEN).bfloat16(),
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, self.mesh.shape, dims=(0, None)),
        )

    def rows_to_host(self, t, col=0):
        """device tensor [1,1,T,D] replicated over columns -> torch [rows, T, D] float (column `col`)."""
        rows, cols = tuple(self.mesh.shape)
        dts = ttnn.get_device_tensors(t)
        return torch.stack([ttnn.to_torch(dts[r * cols + col]).float().reshape(t.shape[2], -1) for r in range(rows)])

    def prefill_first_token(self, tokens, lens, S, U):
        """tokens [rows, U*S] int64 (each user right-padded to S), lens [rows, U] real lengths. Returns (first-token ids [rows,U], h_out host-free)."""
        h = self.embed_tokens(tokens)
        for lyr in self.layers:
            h = self._step(lyr, h, S, U)
        hh = self.rows_to_host(h)  # [rows, U*S, 4096]
        ttnn.deallocate(h)
        last = torch.zeros(tokens.shape[0], 32, HIDDEN)
        for r in range(tokens.shape[0]):
            for u in range(U):
                last[r, u] = hh[r, u * S + int(lens[r][u]) - 1]
        lg = self.final_logits(self.upload_rows(last))
        return lg[:, :U].argmax(-1), lg[:, :U]

    def decode_step(self, tokens, positions):
        """One decode step for 32 slots per row. tokens int64 [rows,32] (dead slots: any id), positions [32] (same for every row; dead slots 0).
        Returns host logits [rows, 32, VOCAB]."""
        if not hasattr(self, "dio"):
            self.dio = [lyr.attn.decode_io() for lyr in self.layers]
        h = self.embed_tokens(tokens)
        for lyr, io in zip(self.layers, self.dio):
            lyr.attn.decode_set(io, positions)
            h2 = lyr.decode(h, io["cur"], io["idx"])
            ttnn.deallocate(h)
            h = h2
        lg = self.final_logits(h)
        ttnn.deallocate(h)
        return lg

    @staticmethod
    def _step(lyr, h, S, U):
        h2 = lyr.prefill(h, S, U)
        ttnn.deallocate(h)
        return h2
