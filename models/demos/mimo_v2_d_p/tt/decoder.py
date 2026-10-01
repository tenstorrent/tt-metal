# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 decoder layer: x + attn(input_norm(x)) -> r; r + ffn(post_attn_norm(r))."""

import ttnn
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.tt.attention.attention import TtAttention
from models.demos.mimo_v2_d_p.tt.ccl import resolve_num_links
from models.demos.mimo_v2_d_p.tt.ffn import TtDenseMLP, TtMoE, TtRMSNorm, all_gather_tp, partition_tp
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions


class TtDecoderLayer:
    def __init__(
        self,
        mesh_device,
        cfg: MiMoTextConfig,
        layer_idx: int,
        layer_sd: dict,
        *,
        ccl,
        sp_topology,
        seq_len_per_chip,
        options: MiMoRuntimeOptions | None = None,
    ):
        self.layer_idx = layer_idx
        options = options or MiMoRuntimeOptions()
        self.mesh_device = mesh_device
        # sequence-parallel residual: the residual stream holds this col's S/TP rows; norms / adds run on them, each block
        # all-gathers its normed input and reduce-scatters its output (the all-reduce's two halves, same CCL volume)
        self.sp = options.sp_residual and mesh_device.shape[1] > 1
        self.kind = cfg.layer_type(layer_idx)
        sub = lambda p: {k[len(p) + 1 :]: v for k, v in layer_sd.items() if k.startswith(p + ".")}
        eps = cfg.layernorm_epsilon
        self.input_norm = TtRMSNorm(mesh_device, layer_sd["input_layernorm.weight"], eps)
        self.post_attn_norm = TtRMSNorm(mesh_device, layer_sd["post_attention_layernorm.weight"], eps)
        cp = f"L{layer_idx}"
        self.attn = TtAttention(mesh_device, cfg, layer_idx, sub("self_attn"), ccl, cache_prefix=cp, options=options)
        if cfg.is_moe(layer_idx):
            self.ffn = TtMoE(
                mesh_device,
                sub("mlp"),
                cfg,
                seq_len_per_chip=seq_len_per_chip,
                num_links=resolve_num_links(mesh_device, options.num_links),
                topology=sp_topology,
                cache_prefix=cp,
                options=options,
            )
        else:
            self.ffn = TtDenseMLP(mesh_device, sub("mlp"), cache_prefix=cp, options=options)

    def __call__(self, x, rope, trans_mat, kv_cache, *, cache_layer, kv_actual, user=0, valid_end=None):
        """x [1,1,S_local,H] replicated over TP -> the same (a sequence-parallel layer partitions and gathers here; the
        model chains ``forward_sp`` without them)."""
        if self.sp:
            xs = partition_tp(x, self.mesh_device)
            ys = self.forward_sp(
                xs,
                rope,
                trans_mat,
                kv_cache,
                cache_layer=cache_layer,
                kv_actual=kv_actual,
                user=user,
                valid_end=valid_end,
            )
            xs.deallocate(True)
            out = all_gather_tp(ys, self.mesh_device)
            ys.deallocate(True)
            return out
        h = self.input_norm(x)
        a = self.attn(
            h, rope, trans_mat, kv_cache, cache_layer=cache_layer, kv_actual=kv_actual, user=user, valid_end=valid_end
        )
        h.deallocate(True)
        r = ttnn.add(x, a)
        a.deallocate(True)
        h = self.post_attn_norm(r)
        f = self.ffn(h)
        h.deallocate(True)
        out = ttnn.add(r, f)
        r.deallocate(True)
        f.deallocate(True)
        return out

    def forward_sp(self, x, rope, trans_mat, kv_cache, *, cache_layer, kv_actual, user=0, valid_end=None):
        """Sequence-parallel residual: x [1,1,S_local/TP,H] (this col's rows) -> the same rows of the layer's output."""
        h = self.input_norm(x)
        hf = all_gather_tp(h, self.mesh_device)
        h.deallocate(True)
        a = self.attn(
            hf,
            rope,
            trans_mat,
            kv_cache,
            cache_layer=cache_layer,
            kv_actual=kv_actual,
            user=user,
            valid_end=valid_end,
            tp_out="scattered",
        )
        hf.deallocate(True)
        r = ttnn.add(x, a)
        a.deallocate(True)
        h = self.post_attn_norm(r)
        hf = all_gather_tp(h, self.mesh_device)
        h.deallocate(True)
        f = self.ffn(hf, tp_out="scattered")
        hf.deallocate(True)
        out = ttnn.add(r, f)
        r.deallocate(True)
        f.deallocate(True)
        return out
