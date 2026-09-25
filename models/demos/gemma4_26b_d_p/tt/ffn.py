# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 FFN sub-layer: dense GELU-tanh MLP in parallel with a 128-expert top-8 MoE.

    m1  = post_ff_norm_1(dense_mlp(pre_ff_norm(r)))
    idx, w = router(r)                                  # router sees the PRE-norm residual
    m2  = post_ff_norm_2(moe(pre_ff_norm_2(r), idx, w))
    out = (r + post_ff_norm(m1 + m2)) * layer_scalar

Hidden states are ``[1, 1, S_local, H]``: sequence block-cyclic over SP rows, replicated over TP cols,
so every norm is a plain token-local ``rms_norm``.

* Dense MLP: gate/up column-parallel, down row-parallel + TP all-reduce. The intermediate (2112) is
  zero-padded to a multiple of 32*TP (2176) so per-col shards are tile aligned (exact: gelu(0)*0 = 0).
* Router: ``softmax(128) -> top8 -> renormalise`` == ``top8(logits) -> softmax over the 8``, so it is
  rms_norm(weight = scale / sqrt(H)) -> linear -> topk -> softmax(k). ``per_expert_scale`` is folded
  exactly into each expert's down_proj.
* MoE: DeepSeek EP substrate (routing_setup -> dispatch -> unified_routed_expert_ffn(GeluTanh) -> combine
  -> reduce), experts spread over all chips (E / (rows*cols) per chip). Dispatch capacity factor
  = min(top_k, experts_per_chip): the exact worst case (all of a token's experts on one chip).
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants, extract_mesh_config, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.gemma4_26b_d_p.reference.config import Gemma4TextConfig
from models.demos.gpt_oss_d_p.tt.moe.router import route_tokens_to_experts


def _rep(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


def norm_weight(w: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        w.float().reshape(1, 1, -1, ttnn.TILE_SIZE), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=_rep(mesh_device)
    )


class TtRMSNorm:
    def __init__(self, mesh_device, weight: torch.Tensor | None, eps: float):
        self.w = norm_weight(weight, mesh_device) if weight is not None else None
        self.eps = eps
        self.cfg = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)

    def __call__(self, x):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.w, compute_kernel_config=self.cfg)


def all_reduce_tp(x, mesh_device):
    if mesh_device.shape[1] == 1:
        return x
    out = ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x.deallocate(True)
    return out


class TtDenseMLP:
    def __init__(self, mesh_device, sd, hidden: int, inter: int, weight_dtype=ttnn.bfloat8_b):
        self.mesh_device = mesh_device
        tp = mesh_device.shape[1]
        align = ttnn.TILE_SIZE * tp
        inter_p = (inter + align - 1) // align * align
        pad = inter_p - inter
        g = torch.nn.functional.pad(sd["gate_proj.weight"].float(), (0, 0, 0, pad))
        u = torch.nn.functional.pad(sd["up_proj.weight"].float(), (0, 0, 0, pad))
        d = torch.nn.functional.pad(sd["down_proj.weight"].float(), (0, pad))
        col = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 3))
        row = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 2))
        to = lambda t, m: ttnn.from_torch(t[None, None], device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=weight_dtype, mesh_mapper=m, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.w_gate, self.w_up, self.w_down = to(g.T, col), to(u.T, col), to(d.T, row)
        self.cfg = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True)

    def __call__(self, x):
        gate = ttnn.linear(x, self.w_gate, dtype=ttnn.bfloat16, compute_kernel_config=self.cfg)
        up = ttnn.linear(x, self.w_up, dtype=ttnn.bfloat16, compute_kernel_config=self.cfg)
        act = ttnn.gelu(gate, fast_and_approximate_mode=False)
        h = ttnn.mul(act, up)
        for t in (gate, up, act):
            t.deallocate(True)
        out = ttnn.linear(h, self.w_down, dtype=ttnn.bfloat16, compute_kernel_config=self.cfg)
        h.deallocate(True)
        return all_reduce_tp(out, self.mesh_device)


class TtRouter:
    """rms_norm(no scale) * scale * H^-0.5 -> proj -> top8 -> softmax over the 8 (see module doc)."""

    def __init__(self, mesh_device, sd, cfg: Gemma4TextConfig):
        H = cfg.hidden_size
        self.H = H
        self.top_k = cfg.top_k_experts
        self.norm = TtRMSNorm(mesh_device, sd["scale"].float() * H**-0.5, cfg.rms_norm_eps)
        self.w = ttnn.from_torch(
            sd["proj.weight"].float().T.contiguous(), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=_rep(mesh_device)
        )
        self.lin_cfg = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        self.sm_cfg = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True)

    def __call__(self, r):
        h = self.norm(r)
        h2 = ttnn.reshape(h, (-1, self.H))
        logits = ttnn.linear(h2, self.w, dtype=ttnn.bfloat16, compute_kernel_config=self.lin_cfg)
        h.deallocate(True)
        idx, w = route_tokens_to_experts(logits, self.top_k, self.sm_cfg)
        logits.deallocate(True)
        return idx, w


def prepare_expert_weights(sd, cfg: Gemma4TextConfig, per_expert_scale: torch.Tensor):
    """HF fused ``gate_up_proj [E, 2I, H]`` (gate rows first) / ``down_proj [E, H, I]`` -> per-expert
    {gate_proj, up_proj, down_proj} in HF (out, in) layout, global id order, per-expert scale folded."""
    I = cfg.moe_intermediate_size
    gu, dn = sd["gate_up_proj"].float(), sd["down_proj"].float()
    return [
        {"gate_proj": gu[e, :I].contiguous(), "up_proj": gu[e, I:].contiguous(), "down_proj": (dn[e] * per_expert_scale[e].float()).contiguous()}
        for e in range(cfg.num_experts)
    ]


class TtMoE:
    """Expert-parallel routed experts (DeepSeek substrate, GELU-tanh fused expert FFN)."""

    def __init__(self, mesh_device, sd, cfg: Gemma4TextConfig, *, seq_len_per_chip, per_expert_scale, num_links=1,
                 topology=ttnn.Topology.Linear, weights_dtype=ttnn.bfloat8_b, layer_idx=0):
        self.mesh_device = mesh_device
        E, K, H, I = cfg.num_experts, cfg.top_k_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.E, self.K, self.H = E, K, H
        mc = extract_mesh_config(mesh_device)
        dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
        experts_per_chip = E // mesh_device.get_num_devices()
        cap = min(K, experts_per_chip)
        experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(seq_len_per_chip, E, K, mesh_device.get_num_devices(), dgs, cap)
        table = ExpertMapping.create_dispatch_table(E, dgs, ndg)
        self.routing_setup = TtMoERoutingSetup(mesh_device, table, num_links=num_links, experts_per_chip=experts_per_chip)
        self.tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, table, dispatch_axis=0)
        self.dispatch = TtDispatchModule(
            mesh_device=mesh_device, dispatch_group_size=dgs, experts_per_chip=experts_per_chip, num_routed_experts=E,
            num_experts_per_tok=K, metadata_len=metadata_len, max_dispatch_buffer_token_size=max_buf,
            seq_len_per_chip=seq_len_per_chip, emb_dim=H, cluster_axis=0, num_links=num_links, topology=topology, subdevice_id=None,
        )
        self.combine = TtCombineModule(
            mesh_device=mesh_device, dispatch_group_size=dgs, num_dispatch_groups=ndg, experts_per_chip=experts_per_chip,
            num_experts_per_tok=K, seq_len_per_chip=seq_len_per_chip, cluster_axis=0, num_links=num_links, topology=topology, init_zeros=True,
        )
        gidx = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(experts_per_chip=experts_per_chip, dispatch_group_size=dgs, num_dispatch_groups=ndg),
            mesh_mapper=get_ep_mesh_mapper(mesh_device), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint32,
        )
        gidx = ttnn.squeeze(ttnn.squeeze(gidx, 0), 0)
        self.expert = TtRoutedExpert(
            mesh_device=mesh_device, experts_per_chip=experts_per_chip, global_expert_idx_table=gidx, emb_dim=H, hidden_dim=I,
            max_tokens=max_tok, torch_weights=prepare_expert_weights(sd, cfg, per_expert_scale), activations_dtype=ttnn.bfloat8_b,
            weights_dtype=weights_dtype, activation=ttnn.RoutedExpertActivation.GeluTanh,
        )
        self.reduce = TtReduceModule(mesh_device=mesh_device, topk_dim=3, cluster_axis=1, num_links=num_links, topology=topology)

    def __call__(self, x, idx, w):
        """x [1,1,S,H] (replicated over TP) -> [1,1,S,H]."""
        offsets, counts, regions, _ = self.routing_setup(ttnn_top_k_experts_indices=idx, num_routed_experts=self.E, num_experts_per_tok=self.K)
        S = x.shape[2]
        idx = ttnn.reshape(ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT), (1, S, self.K))
        w = ttnn.reshape(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (1, S, self.K))
        x3 = ttnn.squeeze(x, 0)
        buf, meta = self.dispatch(x3, w, idx, offsets, self.tt_table)
        tiled = ttnn.to_layout(ttnn.squeeze(ttnn.squeeze(buf, 0), 0), ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(buf)
        y = self.expert(tiled, counts, regions)
        y = ttnn.unsqueeze(ttnn.unsqueeze(y, 0), 0)
        comb = self.combine(y, meta, counts, regions, seq_len_per_chip=S)
        w = ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG)
        idx = ttnn.to_memory_config(idx, ttnn.DRAM_MEMORY_CONFIG)
        out = self.reduce(comb, weights=w, indices=idx, expert_dispatch_table=self.tt_table)  # [1,S,H/tp] reduce-scattered over TP
        out = ttnn.unsqueeze(ttnn.squeeze(out, 0), 0) if len(out.shape) == 4 else ttnn.unsqueeze(out, 0)
        if self.mesh_device.shape[1] > 1 and out.shape[-1] < self.H:
            out = ttnn.all_gather(out, dim=-1, cluster_axis=1, topology=ttnn.Topology.Linear)
        return out


class TtFFN:
    def __init__(self, mesh_device, cfg: Gemma4TextConfig, layer_sd: dict, *, seq_len_per_chip, num_links=1, sp_topology=ttnn.Topology.Linear,
                 expert_dtype=ttnn.bfloat8_b, layer_idx=0):
        sub = lambda p: {k[len(p) + 1 :]: v for k, v in layer_sd.items() if k.startswith(p + ".")}
        eps = cfg.rms_norm_eps
        self.pre_ff_norm = TtRMSNorm(mesh_device, layer_sd["pre_feedforward_layernorm.weight"], eps)
        self.post_ff_norm_1 = TtRMSNorm(mesh_device, layer_sd["post_feedforward_layernorm_1.weight"], eps)
        self.pre_ff_norm_2 = TtRMSNorm(mesh_device, layer_sd["pre_feedforward_layernorm_2.weight"], eps)
        self.post_ff_norm_2 = TtRMSNorm(mesh_device, layer_sd["post_feedforward_layernorm_2.weight"], eps)
        self.post_ff_norm = TtRMSNorm(mesh_device, layer_sd["post_feedforward_layernorm.weight"], eps)
        self.mlp = TtDenseMLP(mesh_device, sub("mlp"), cfg.hidden_size, cfg.intermediate_size)
        rsd = sub("router")
        self.router = TtRouter(mesh_device, rsd, cfg)
        self.moe = TtMoE(mesh_device, sub("experts"), cfg, seq_len_per_chip=seq_len_per_chip, per_expert_scale=rsd["per_expert_scale"],
                         num_links=num_links, topology=sp_topology, weights_dtype=expert_dtype, layer_idx=layer_idx)
        self.layer_scalar = float(layer_sd["layer_scalar"].float().item()) if "layer_scalar" in layer_sd else 1.0

    def __call__(self, r):
        x1 = self.pre_ff_norm(r)
        m1 = self.mlp(x1)
        x1.deallocate(True)
        m1n = self.post_ff_norm_1(m1)
        m1.deallocate(True)
        idx, w = self.router(r)
        x2 = self.pre_ff_norm_2(r)
        m2 = self.moe(x2, idx, w)
        x2.deallocate(True)
        m2n = self.post_ff_norm_2(m2)
        m2.deallocate(True)
        s = ttnn.add(m1n, m2n)
        m1n.deallocate(True)
        m2n.deallocate(True)
        f = self.post_ff_norm(s)
        s.deallocate(True)
        out = ttnn.add(r, f)
        f.deallocate(True)
        if self.layer_scalar != 1.0:
            scaled = ttnn.mul(out, self.layer_scalar)
            out.deallocate(True)
            out = scaled
        return out
