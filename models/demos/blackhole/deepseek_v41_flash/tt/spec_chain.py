# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer builder for speculative verification blocks: ``DSV41DecodeChain`` with the paged spec attention (n = 1 + k token rows per user)."""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt.layer import DSV41Layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.spec_attention import SpecCompressedAttention, SpecWindowAttention


class SpecChain(DSV41DecodeChain):
    def __init__(self, mesh_device, users_per_row=4, n=2, max_comp=128, log=print):
        super().__init__(mesh_device, users_per_row=users_per_row * n, max_comp=max_comp, log=log)
        self.U, self.n = users_per_row, n
        self.n_users = self.rows * users_per_row  # B (tokens) = n_users * n

    def build_layer(self, L, chain, w=None):
        from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer

        w = w if w is not None else load_layer(L)
        meta, state, S = w["meta"], chain["state"], chain["S"]
        kw = dict(users_per_row=self.U, n=self.n)
        if meta["ratio"] == 0:
            attn = SpecWindowAttention(
                self.md, self.mesh_config, self.ccl, w["attn"], w["freqs_cis"], max_seq=256, **kw
            )
            attn.load_window(self._ring_to_linear(state["window"], S))
        elif meta["is_kv_source"]:
            attn = SpecCompressedAttention(
                self.md,
                self.mesh_config,
                self.ccl,
                w["attn"],
                w["freqs_cis"],
                meta["ratio"],
                w["compressor"],
                max_comp=self.max_comp,
                **kw,
            )
            attn.load_state(state["window"], state["comp"], state.get("kv_state"), state.get("score_state"), S=S)
            self.sources[L] = attn
        else:
            attn = SpecCompressedAttention(
                self.md,
                self.mesh_config,
                self.ccl,
                w["attn"],
                w["freqs_cis"],
                meta["ratio"],
                None,
                max_comp=self.max_comp,
                source=self.sources[meta["kv_source"]],
                **kw,
            )
            attn.load_state(state["window"], None, None, None, S=S)
        layer = DSV41Layer(
            self.md,
            self.mesh_config,
            self.ccl,
            attn,
            w["norms"],
            w["mhc"],
            w["moe"],
            gate_bias_shift=chain["gate_cutoff"],
            users_per_row=self.T,
            moe_buffers=self.moe_buffers,
        )
        if self.moe_buffers is None:
            self.moe_buffers = layer.moe.decode.buffers
        return layer, attn

    @staticmethod
    def _ring_to_linear(window, S):
        """reference window ring [B,128,512] after S tokens -> linear [B, S, 512] (S <= 128 here: slot == position)."""
        assert S <= 128
        return window[:, :S]

    def tok_dev(self, per_step, dims_last, steps):
        """per_step: list over block index j of [n_users, ...] torch tensors -> device tensor with token rows user-major (u*n + j)."""
        t = torch.stack([per_step[j].reshape(self.n_users, -1) for j in steps], dim=1)  # [B, n, F]
        return self.to_dev(t.reshape(self.n_users * self.n, -1), dims_last)

    def pos_dev(self, base):
        import ttnn

        pos = (torch.as_tensor(base).reshape(-1, 1) + torch.arange(self.n).reshape(1, -1)).reshape(-1).to(torch.int32)
        return ttnn.from_torch(
            pos,
            device=self.md,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols)),
        )
