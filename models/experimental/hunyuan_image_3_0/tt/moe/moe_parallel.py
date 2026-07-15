# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Expert-parallel MoE — shards the 64 experts across a mesh axis so each device
# holds 64/ndev experts RESIDENT in bf8 (no per-forward host streaming). This is
# the dominant residency lever for the 80B model (experts are ~97% of per-layer
# weight memory). See MEMORY_FIT_PLAN.md.
#
# Math (mirrors the dense single-device HunyuanTtMoE):
#     combined = sum_e expert_e(x) * combine_w[:, e]
# Each device computes the partial sum over ITS experts, then we all-reduce the
# partial across the mesh axis to get the full combined output on every device.
#
# Per-device expert identity in SPMD: an arange(E) sharded along the mesh axis
# gives each device the GLOBAL ids of its local experts, so the combine-weight
# selection (topk_idx == global_id) picks the right column per device.

import os
import time

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

from .gate import HunyuanTtTopKGate
from .mlp import HunyuanTtMLP
from ..cache import cache_file
from ..matmul_utils import l1_sharded_linear
from ..parallel_utils import moe_full_seq_mem_config, resid_mem_config, sp_gather, sp_shard


class HunyuanTtMoEParallel(LightweightModule):
    def __init__(
        self,
        mesh_device,
        ccl_manager,
        state_dict,
        prefix,
        *,
        num_experts: int,
        hidden_size: int,
        moe_topk: int,
        num_shared_expert: int = 1,
        norm_topk_prob: bool = True,
        use_mixed_mlp_moe: bool = True,
        mesh_axis: int = 0,
        weight_dtype=ttnn.bfloat8_b,
        gate_dtype=ttnn.bfloat8_b,
        sp_axis: int = 0,
        sp_factor: int = 1,
        weight_cache_path=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.ccl = ccl_manager
        self.mesh_axis = mesh_axis
        # SP: tokens arrive sequence-sharded on sp_axis. Experts live on all 4
        # devices, so a local token may route to a remote expert — rather than an
        # all-to-all, we gather tokens to the full sequence (replicated), run the
        # existing full-mesh EP, then reshard the combined output back to S/sp. The
        # EP all-reduce is unchanged (its precondition — replicated tokens — holds
        # after the gather).
        self.sp_axis = sp_axis
        self.sp_factor = sp_factor
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe

        # Expert parallel spans the FULL mesh: ShardTensorToMesh(dim=0) below splits
        # the 64 experts row-major across every device, so on a 1x4 mesh each device
        # holds 16 experts and on a 2x2 mesh each device ALSO holds 16 (4-way EP) —
        # this is the only layout that fits the 80GB bf8 model (16 experts ~= 19GB).
        # The partial sum is therefore reduced over every non-trivial mesh axis (the
        # `mesh_axis` arg is retained for API compatibility but no longer gates the
        # shard — see forward()). On a 1x4 mesh only axis 1 is non-trivial, so this
        # reduces to the original single-axis behavior.
        ndev = mesh_device.get_num_devices()
        assert num_experts % ndev == 0, f"{num_experts} experts not divisible by mesh device count {ndev}"
        self.ndev = ndev
        self.experts_per_dev = num_experts // ndev
        mesh_shape = tuple(mesh_device.shape)
        self.ep_reduce_axes = [a for a, n in enumerate(mesh_shape) if n > 1]

        # --- stacked expert weights, sharded along the expert dim ------------
        # gate_and_up: [E, H, 2I]  down: [E, I, H]  (transposed for ttnn.linear)
        verbose = os.environ.get("HY_VERBOSE", "1") != "0"
        if verbose:
            print(f"[backbone]   stacking {num_experts} expert weights on host ...", flush=True)
        wgu = torch.stack(
            [
                state_dict[f"{prefix}.experts.{e}.gate_and_up_proj.weight"].transpose(0, 1).contiguous()
                for e in range(num_experts)
            ]
        )
        wdn = torch.stack(
            [
                state_dict[f"{prefix}.experts.{e}.down_proj.weight"].transpose(0, 1).contiguous()
                for e in range(num_experts)
            ]
        )
        self.inter2 = wgu.shape[-1]  # 2I
        if verbose:
            print(
                f"[backbone]   uploading {num_experts} experts ({weight_dtype}) to mesh ...",
                flush=True,
            )
        t_upload = time.time()
        expert_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        self.w_gate_up = ttnn.as_tensor(
            wgu,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=expert_mapper,
            cache_file_name=cache_file(weight_cache_path, f"{prefix}.experts.gate_and_up_stacked"),
        )  # per-device [epd, H, 2I]
        self.w_down = ttnn.as_tensor(
            wdn,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=expert_mapper,
            cache_file_name=cache_file(weight_cache_path, f"{prefix}.experts.down_stacked"),
        )  # per-device [epd, I, H]
        del wgu, wdn
        if verbose:
            print(f"[backbone]   expert upload done ({time.time() - t_upload:.1f}s)", flush=True)

        # Pre-slice the stacked expert weights into per-expert [A, B] tensors ONCE.
        # The forward path used to re-slice the full DRAM weight stack on the expert
        # dim for every expert, every layer, every decode step — a ~25MB DRAM copy per
        # slice that dominated the AR decode step. The weights are static, so we take
        # those slices here and keep them resident; the hot loop just indexes a list.
        # Same total bytes (the stack is freed right after), no per-token copy.
        self.w_gate_up_experts = [self._slice_expert(self.w_gate_up, el) for el in range(self.experts_per_dev)]
        self.w_down_experts = [self._slice_expert(self.w_down, el) for el in range(self.experts_per_dev)]
        ttnn.deallocate(self.w_gate_up)
        ttnn.deallocate(self.w_down)

        # Per-device GLOBAL expert ids: arange(E) sharded along the mesh axis.
        # bf16 so it compares against the (bf16-cast) topk indices; ids <= 63 exact.
        ids = torch.arange(num_experts, dtype=torch.float32).reshape(num_experts, 1)
        self.local_ids = ttnn.from_torch(
            ids,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )  # per-device [epd, 1]
        # Pre-slice once: local_ids is static, so re-slicing inside the per-forward
        # expert loop would be a pure repeated cost.
        self.local_ids_experts = [self.local_ids[el] for el in range(self.experts_per_dev)]

        # Gate + shared MLP run replicated (input is replicated). Gate signature is
        # positional: (device, num_experts, moe_topk, state_dict, weight_key, ...).
        self.gate = HunyuanTtTopKGate(
            mesh_device,
            hidden_size,
            num_experts,
            moe_topk,
            state_dict,
            f"{prefix}.gate.wg",
            norm_topk_prob=norm_topk_prob,
            weight_dtype=gate_dtype,
            weight_cache_path=weight_cache_path,
        )
        self.shared_mlp = None
        if use_mixed_mlp_moe:
            mlp_dtype = weight_dtype
            self.shared_mlp = HunyuanTtMLP(
                mesh_device,
                hidden_size,
                state_dict,
                f"{prefix}.shared_mlp",
                weight_dtype=mlp_dtype,
                weight_cache_path=weight_cache_path,
            )

        # Expert matmuls are memory/compute-bound on low-precision weights (bf8/bf16),
        # so HiFi4's 4 math passes and fp32 dest accumulation buy no accuracy the
        # weights can carry — HiFi2 (2 passes) + bf16 accumulation roughly halves the
        # compute-bound expert-matmul time. PCC-gated by the kv-cache decode/prefill tests.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _slice_expert(self, w, el):
        """Slice local expert el from a [epd, A, B] sharded weight -> [A, B]."""
        epd, A, Bd = w.shape
        s = ttnn.slice(w, [el, 0, 0], [el + 1, A, Bd])
        return ttnn.reshape(s, (A, Bd))

    def _expert(self, x, el):
        """Run local expert `el` (its weight slice differs per device)."""
        wgu = self.w_gate_up_experts[el]  # [H, 2I] (different global expert per device)
        wdn = self.w_down_experts[el]  # [I, H]
        # l1_sharded_linear self-gates the two matmuls (small-M -> L1 width-sharded,
        # large full-sequence M -> DRAM). The transient SwiGLU multiply runs on the
        # full gathered sequence, so its L1 buffer scales with S and clashes with the
        # ops' static CBs past the measured bound — follow the seq-length L1->DRAM gate.
        mc = moe_full_seq_mem_config(x.shape[1])
        gu = l1_sharded_linear(x, wgu, compute_kernel_config=self.compute_kernel_config)
        x1, x2 = ttnn.chunk(gu, 2, dim=-1)
        ttnn.deallocate(gu)
        # SwiGLU x1 * silu(x2), with SiLU folded into the multiply's LHS activation
        # (one fused BinaryNg instead of a separate silu Unary + multiply). SILU must
        # apply to x2 (the up half); multiply(x2, x1, a_acts=[SILU]) == x1 * silu(x2).
        h = ttnn.multiply(x2, x1, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mc)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        # Down-proj (M x 3072 x 4096) schedule is M-dependent (measured,
        # tests/perf/test_expert_down_sweep.py): Mt==1 decode -> width-sharded L1,
        # mid M (Mt 2-7) -> ttnn auto (25-33% faster than wide_mm), large M (Mt >= 8)
        # -> rectangular-grid 2D-mcast. l1_sharded_linear applies exactly this policy.
        out = l1_sharded_linear(h, wdn, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(h)
        return out

    def _ep_all_reduce(self, partial: ttnn.Tensor) -> ttnn.Tensor:
        """Sum a [B,S,H] per-device partial over every non-trivial mesh axis.

        Each device computed the partial sum over ITS local experts; reducing over
        the axes the experts are sharded on yields the full combined output on
        every device. Done as all-gather(dim=0)+sum per axis.
        """
        out = partial
        for axis in self.ep_reduce_axes:
            n = self.mesh_device.shape[axis]
            gathered = self.ccl.all_gather(out, dim=0, mesh_axis=axis, use_hyperparams=False)  # [n*B,S,H]
            ttnn.deallocate(out)
            B = gathered.shape[0] // n
            S, H = gathered.shape[1], gathered.shape[2]
            gathered = ttnn.reshape(gathered, (n, B, S, H))
            out = ttnn.sum(gathered, dim=0, memory_config=moe_full_seq_mem_config(S))  # [B,S,H]
            ttnn.deallocate(gathered)
        return out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # SP: gather the sequence-sharded tokens to the full (replicated) sequence so
        # the full-mesh EP below is valid; reshard the combined output back at the end.
        # `xf` is a fresh tensor — the caller-owned `x` is left untouched.
        sp = self.sp_factor > 1
        # gathered tokens feed the router matmul next: land L1-resident up to the
        # full-sequence CB-clash bound (the gathered tensor is the GLOBAL seq), else DRAM.
        global_seq = x.shape[1] * self.sp_factor if sp else x.shape[1]
        xf = (
            sp_gather(
                self.ccl,
                x,
                dim=1,
                mesh_axis=self.sp_axis,
                n=self.sp_factor,
                out_memory_config=moe_full_seq_mem_config(global_seq),
            )
            if sp
            else x
        )

        out = self._forward_full(xf)
        if sp:
            ttnn.deallocate(xf)
            # MoE output feeds the layer's residual add next: land it L1-resident up to
            # the per-device residual bound (the reshard output is the per-device seq).
            out = sp_shard(
                self.ccl,
                out,
                dim=1,
                mesh_axis=self.sp_axis,
                n=self.sp_factor,
                out_memory_config=resid_mem_config(out.shape[1] // self.sp_factor),
            )
        return out

    def _forward_full(self, x: ttnn.Tensor) -> ttnn.Tensor:
        topk_w, topk_idx_raw = self.gate(x)  # [B,S,k] each, replicated
        topk_idx = ttnn.typecast(topk_idx_raw, ttnn.bfloat16)  # ids <= 63 exact in bf16
        ttnn.deallocate(topk_idx_raw)

        # Full-S combine accumulator [B,S,H] scales with sequence like the expert body,
        # so it follows the same measured L1->DRAM gate.
        mc = moe_full_seq_mem_config(x.shape[1])
        partial = None
        for el in range(self.experts_per_dev):
            gid = self.local_ids_experts[el]  # [1] — global id of this device's el-th expert
            sel = ttnn.eq(topk_idx, gid)  # [B,S,k] (broadcast scalar differs per device)
            contrib = ttnn.multiply(sel, topk_w)
            ttnn.deallocate(sel)
            w_e = ttnn.sum(contrib, dim=-1, keepdim=True)  # [B,S,1]
            ttnn.deallocate(contrib)

            oe = self._expert(x, el)
            if partial is None:
                partial = ttnn.multiply(oe, w_e, memory_config=mc)  # [B,S,H]
            else:
                # addcmul(partial, oe, w_e) = partial + oe * w_e, fused in one op
                # instead of a separate multiply + add.
                tmp = ttnn.addcmul(partial, oe, w_e, memory_config=mc)
                ttnn.deallocate(partial)
                partial = tmp
            ttnn.deallocate(oe)
            ttnn.deallocate(w_e)

        ttnn.deallocate(topk_w)
        ttnn.deallocate(topk_idx)

        # All-reduce the per-device partial sums over the full expert-shard mesh.
        # Experts are split across EVERY non-trivial mesh axis (4-way on a 2x2,
        # 4-way on a 1x4), so we reduce over each such axis in turn: all-gather the
        # partials along the axis and sum. After the final axis every device holds
        # the full combined output. (On a 1x4 this is a single reduce over axis 1.)
        combined = self._ep_all_reduce(partial)

        if self.shared_mlp is not None:
            shared = self.shared_mlp(x)
            out = ttnn.add(shared, combined, memory_config=moe_full_seq_mem_config(x.shape[1]))
            ttnn.deallocate(shared)
            ttnn.deallocate(combined)
            return out
        return combined
