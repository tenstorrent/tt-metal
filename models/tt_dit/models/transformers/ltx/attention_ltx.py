# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ....layers.linear import ColParallelLinear, LoRAColParallelLinear, maybe_cast_activation, resolve_output_dtype
from ....layers.module import Module
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.matmul import get_fabric_agmm_config, get_matmul_config
from ....utils import sdpa_recipe
from ....utils.sdpa_recipe import (
    prepare_recipe_inputs,
    recipe_config,
    sdpa_kwargs,
    validate_recipe_args,
)
from ....utils.substate import pop_substate, rename_substate
from ....utils.tensor import bf16_tensor
from .quant_config import LtxQuantProfile

# to_gate_logits and to_q/to_qkv are both ColParallelLinear fed the SAME activation, and each fuses
# its own TP all-gather of it (all_gather_minimal_matmul_async) — the activation crosses the fabric
# twice. The fused gather barely overlaps its matmul, so gathering once explicitly and running both
# projections as plain matmuls on the gathered tensor is the same math for one gather less.
# Set to 0 to restore the double gather for an A/B.
LTX_DEDUP_GATE_GATHER = os.environ.get("LTX_DEDUP_GATE_GATHER", "1") in ("1", "true", "True")


class LTXAttention(Module):
    # Named SDPA recipe of every SDPA call in this module on Blackhole. DiT models default to
    # FAST (legacy streaming numerics with the approximate exponential; user decision
    # 2026-09-25): at the models' shapes it is as accurate as the legacy HiFi2 / BF16-dest /
    # exact-exp setup within a few percent and at least as fast. Pass sdpa_precision to opt
    # up (e.g. BALANCED). See tests/ttnn/unit_tests/operations/sdpa/test_sdpa_dit_recipe_parity.py.
    sdpa_precision_default = ttnn.SDPAPrecision.FAST

    # Legacy ring SDPA chunks (non-Blackhole only): (is_blackhole, sp_factor, tp_factor) -> (q, k).
    sdpa_chunk_size_map = {
        (False, 2, 4): (256, 256),
        (False, 8, 4): (256, 256),
    }
    default_sdpa_chunk_size = (256, 256)

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        is_self: bool = True,
        context_dim: int | None = None,
        query_input_dim: int | None = None,
        output_dim: int | None = None,
        apply_gated_attention: bool = False,
        quant_config: LtxQuantProfile | None = None,
        lora_enabled: bool = False,
        sdpa_precision: ttnn.SDPAPrecision | None = None,
        sdpa_kv_dtype: ttnn.DataType | None = None,
    ) -> None:
        """``sdpa_precision``/``sdpa_kv_dtype`` override the named SDPA recipe of every SDPA call
        (Blackhole, D64/D128/D256, noncausal; see models/tt_dit/utils/sdpa_recipe.py). ``None`` selects
        ``sdpa_precision_default`` (or, for self-attention, the quant profile's recipe); off Blackhole the
        legacy SDPA configuration is used. In LTX-2 the transformer block passes them to every
        attention: the D128 video self/text attentions and the D64 audio self/text, A2V and V2A
        attentions. The padded audio self-attn's key-column mask is replaced under a recipe by slicing
        K/V to the logical key length (``forward(attn_kv_len=...)``); any other mask is passed to the recipe
        as ``attn_mask``."""
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        blackhole = is_blackhole()
        if sdpa_precision is None and is_self and quant_config is not None and blackhole:
            # The quant profile's self-attention SDPA recipe (None: the default below).
            sdpa_precision, sdpa_kv_dtype = quant_config.sdpa_self_recipe()
        self.sdpa_precision = sdpa_recipe.resolve_precision(
            sdpa_precision, self.sdpa_precision_default, blackhole=blackhole, model="LTX-2"
        )
        self.sdpa_kv_dtype = validate_recipe_args(
            self.sdpa_precision, sdpa_kv_dtype, head_dim=self.head_dim, model="LTX-2", is_blackhole=blackhole
        )
        self.qk_norm = qk_norm
        self.eps = eps
        self.is_self = is_self
        self.query_input_dim = query_input_dim or dim
        self.output_dim = output_dim or dim

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.n_local_heads = self.num_heads // self.parallel_config.tensor_parallel.factor

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        rms_kwargs = {
            "embedding_dim": dim,
            "norm_eps": eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "ccl_manager": ccl_manager,
        }

        self.norm_q = DistributedRMSNorm(**rms_kwargs)
        self.norm_k = DistributedRMSNorm(**rms_kwargs)

        col_parallel_kwargs = {
            "bias": True,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "fsdp_mesh_axis": fsdp_mesh_axis,
            "ccl_manager": ccl_manager,
        }

        self.kv_input_dim = context_dim if (context_dim is not None and not is_self) else dim

        # Per-linear precision comes entirely from the quant profile (None => bf16 everywhere, matching
        # the unquantized model). The profile owns the to_out carve-out and the LTX_QUANT_ACTIVATIONS
        # gating; here each role's linear just spreads the kwargs the profile hands it.
        def qk(role):
            return quant_config.linear_kwargs(role) if quant_config is not None else {}

        # Fuse-mode LoRA lives in weight.data, so the chunked (to_qkv/to_kv) and
        # fused-addcmul (to_out) paths work unchanged; runtime mode is unsupported here. The quant
        # kwargs forward through LoRAColParallelLinear's **kwargs to the base linear.
        ColCls = LoRAColParallelLinear if lora_enabled else ColParallelLinear

        # Fuse the per-head gate into the QKV/Q matmul via variable-width chunks
        tp_factor = parallel_config.tensor_parallel.factor
        uses_fused_agmm = tp_factor > 1 and ccl_manager is not None and ccl_manager.topology == ttnn.Topology.Ring
        # A fused linear carries a single weight dtype, but the quant profile holds the gate at bf16
        # while quantizing qkv/q, so the two cannot share one matmul: fusion is bf16-only.
        self.fuse_gate = apply_gated_attention and uses_fused_agmm and quant_config is None

        # Gate is num_heads/TP columns per device (sub-tile); pad to a whole tile so it's a legal chunk.
        self.gate_width_per_device = self.n_local_heads
        self.gate_padded_per_device = ((self.n_local_heads + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        gate_fused_width = self.gate_padded_per_device * tp_factor

        if is_self:
            if self.fuse_gate:
                self.to_qkv = ColCls(
                    dim,
                    3 * dim + gate_fused_width,
                    chunks=4,
                    chunk_sizes=[dim, dim, dim, gate_fused_width],
                    **col_parallel_kwargs,
                    **qk("qkv"),
                )
            else:
                self.to_qkv = ColCls(dim, 3 * dim, chunks=3, **col_parallel_kwargs, **qk("qkv"))
        else:
            if self.fuse_gate:
                self.to_q = ColCls(
                    self.query_input_dim,
                    dim + gate_fused_width,
                    chunks=2,
                    chunk_sizes=[dim, gate_fused_width],
                    **col_parallel_kwargs,
                    **qk("q"),
                )
            else:
                self.to_q = ColCls(self.query_input_dim, dim, **col_parallel_kwargs, **qk("q"))
            self.to_kv = ColCls(self.kv_input_dim, 2 * dim, chunks=2, **col_parallel_kwargs, **qk("kv"))

        self.to_out = ColCls(
            dim,
            self.output_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
            **qk("out"),
        )

        # Per-head gate, sharded on num_heads to match the SDPA-output head layout
        self.apply_gated_attention = apply_gated_attention
        if apply_gated_attention and not self.fuse_gate:
            # Standalone gate weight stays bf16 (the model's working dtype) while consuming the shared
            # bf8 activation, so its output is pinned back to bf16 like the quantized projections.
            self.to_gate_logits = ColParallelLinear(
                in_features=self.query_input_dim,
                out_features=self.num_heads,
                bias=True,
                dtype=ttnn.bfloat16,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                fsdp_mesh_axis=fsdp_mesh_axis,
                ccl_manager=ccl_manager,
                **qk("gate"),
            )

        self.dummy_joint_input = bf16_tensor(torch.zeros((1, self.n_local_heads, 0, self.head_dim)), device=mesh_device)

        full_grid = self.mesh_device.compute_with_storage_grid_size()
        self.sdpa_worker_grid = (full_grid.x - 1, full_grid.y)
        if self.sdpa_precision is not None:
            # The recipe owns the numerics; SDPA chooses the chunks for each grid and shape.
            self.sdpa_program_config = recipe_config(full_grid)
            self.ring_sdpa_program_config = recipe_config(self.sdpa_worker_grid)
            self.cross_ring_sdpa_program_config = self.ring_sdpa_program_config
            self.sdpa_compute_kernel_config = None
        else:
            # Legacy SDPA (non-Blackhole).
            self.sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=full_grid,
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            mesh_key = (
                blackhole,
                self.parallel_config.sequence_parallel.factor,
                self.parallel_config.tensor_parallel.factor,
            )
            ring_sdpa_chunk_size = self.sdpa_chunk_size_map.get(mesh_key, self.default_sdpa_chunk_size)
            self.ring_sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.sdpa_worker_grid,
                q_chunk_size=ring_sdpa_chunk_size[0],
                k_chunk_size=ring_sdpa_chunk_size[1],
                exp_approx_mode=False,
            )
            self.cross_ring_sdpa_program_config = self.ring_sdpa_program_config
            # All SDPA (ring + cross) runs HiFi2, matching the Wan attention config.
            self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
            )

        self.rope_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

        # Attention QKV/out matmuls run HiFi2, matching the Wan attention config.
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Under a quant preset, override the attention compute configs to the profile's fidelity.
        # Legacy (non-Blackhole) self-attn additionally swaps in the ring-SDPA compute and narrows its
        # SDPA inputs (on Blackhole the profile's recipe was selected above); cross-attn leaves SDPA and
        # _sdpa_input_dtype unset, so forward's getattr(self, "_sdpa_input_dtype", None) keeps cross
        # SDPA at bf16.
        if quant_config is not None:
            arch = self.mesh_device.arch()
            self.mm_compute_kernel_config = quant_config.mm_compute_config(arch)
            if self.is_self and self.sdpa_precision is None:
                self.sdpa_compute_kernel_config, self._sdpa_input_dtype = quant_config.sdpa_self_config(arch)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "to_out.0", "to_out")
        rename_substate(state, "q_norm", "norm_q")
        rename_substate(state, "k_norm", "norm_k")

        # Permute Q/K (and norm_q/norm_k) channels per head from checkpoint SPLIT rotation to
        # the INTERLEAVED layout rotary_embedding_llama expects: even output lanes take the
        # first half of each head's channels, odd lanes the second half.
        D = self.head_dim
        D_half = D // 2
        perm = torch.empty(D, dtype=torch.long)
        perm[0::2] = torch.arange(D_half)
        perm[1::2] = torch.arange(D_half, D)

        def _permute_qk(t: torch.Tensor) -> torch.Tensor:
            rest = t.shape[1:]
            t = t.reshape(self.num_heads, D, *rest).index_select(1, perm)
            return t.reshape(self.num_heads * D, *rest)

        for nk in ("norm_q.weight", "norm_k.weight"):
            if nk in state:
                state[nk] = _permute_qk(state[nk])

        def _interleave_heads(tensors: list[torch.Tensor]):
            n_dev = self.parallel_config.tensor_parallel.factor
            tensors = [t.T for t in tensors]
            tensors = [t.reshape(t.shape[0], n_dev, self.n_local_heads, self.head_dim) for t in tensors]
            merged = torch.cat(tensors, dim=2)
            merged = merged.reshape(merged.shape[0], len(tensors) * self.num_heads * self.head_dim)
            return merged.T

        def _pad_gate_to_tile(t: torch.Tensor) -> torch.Tensor:
            """Zero-pad each device's gate block from n_local_heads up to a whole tile (device-major)."""
            n_dev = self.parallel_config.tensor_parallel.factor
            pad = self.gate_padded_per_device - self.n_local_heads
            if pad == 0:
                return t
            blocks = []
            for d in range(n_dev):
                blocks.append(t[d * self.n_local_heads : (d + 1) * self.n_local_heads])
                blocks.append(torch.zeros(pad, *t.shape[1:], dtype=t.dtype))
            return torch.cat(blocks, dim=0)

        def _gate_state_or_zeros(reference: torch.Tensor, want_bias: bool) -> dict[str, torch.Tensor]:
            """Gate substate, zero-filled when the checkpoint has none (zeros -> identity gating)."""
            g = pop_substate(state, "to_gate_logits")
            if "weight" not in g:
                g["weight"] = torch.zeros(self.num_heads, reference.shape[1], dtype=reference.dtype)
            if want_bias and "bias" not in g:
                g["bias"] = torch.zeros(self.num_heads, dtype=reference.dtype)
            return g

        def _interleave_device_major(tensors: list[torch.Tensor]):
            """Like _interleave_heads but for per-device widths that differ (q/k/v head_dim vs gate)."""
            n_dev = self.parallel_config.tensor_parallel.factor
            reshaped = []
            for t in tensors:
                t = t.T  # [in, out]
                assert t.shape[1] % n_dev == 0, f"projection width {t.shape[1]} not divisible by TP={n_dev}"
                reshaped.append(t.reshape(t.shape[0], n_dev, t.shape[1] // n_dev))
            merged = torch.cat(reshaped, dim=2)
            return merged.reshape(merged.shape[0], -1).T

        if self.is_self:
            q_state = pop_substate(state, "to_q")
            k_state = pop_substate(state, "to_k")
            v_state = pop_substate(state, "to_v")

            q_state["weight"] = _permute_qk(q_state["weight"])
            k_state["weight"] = _permute_qk(k_state["weight"])
            if "bias" in q_state:
                q_state["bias"] = _permute_qk(q_state["bias"])
            if "bias" in k_state:
                k_state["bias"] = _permute_qk(k_state["bias"])

            if self.fuse_gate:
                # Fold the gate in as a 4th chunk, device-major: device d holds [q_d | k_d | v_d | gate_d].
                g_state = _gate_state_or_zeros(q_state["weight"], "bias" in q_state)
                g_state = {k: _pad_gate_to_tile(v) for k, v in g_state.items()}
                state["to_qkv.weight"] = _interleave_device_major(
                    [q_state["weight"], k_state["weight"], v_state["weight"], g_state["weight"]]
                )
                if "bias" in q_state:
                    bias = _interleave_device_major(
                        [
                            q_state["bias"].unsqueeze(-1),
                            k_state["bias"].unsqueeze(-1),
                            v_state["bias"].unsqueeze(-1),
                            g_state["bias"].unsqueeze(-1),
                        ]
                    )
                    state["to_qkv.bias"] = bias.squeeze(-1)
            else:
                state["to_qkv.weight"] = _interleave_heads([q_state["weight"], k_state["weight"], v_state["weight"]])
                if "bias" in q_state:
                    bias = _interleave_heads(
                        [q_state["bias"].unsqueeze(-1), k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)]
                    )
                    state["to_qkv.bias"] = bias.squeeze(-1)
        else:
            k_state = pop_substate(state, "to_k")
            v_state = pop_substate(state, "to_v")

            k_state["weight"] = _permute_qk(k_state["weight"])
            if "bias" in k_state:
                k_state["bias"] = _permute_qk(k_state["bias"])

            if "to_q.weight" in state:
                state["to_q.weight"] = _permute_qk(state["to_q.weight"])
            if "to_q.bias" in state:
                state["to_q.bias"] = _permute_qk(state["to_q.bias"])

            if self.fuse_gate and "to_q.weight" in state:
                # Cross-attn: gate rides along with Q as a 2nd chunk.
                g_state = _gate_state_or_zeros(state["to_q.weight"], "to_q.bias" in state)
                g_state = {k: _pad_gate_to_tile(v) for k, v in g_state.items()}
                state["to_q.weight"] = _interleave_device_major([state["to_q.weight"], g_state["weight"]])
                if "to_q.bias" in state and "bias" in g_state:
                    bias = _interleave_device_major([state["to_q.bias"].unsqueeze(-1), g_state["bias"].unsqueeze(-1)])
                    state["to_q.bias"] = bias.squeeze(-1)

            state["to_kv.weight"] = _interleave_heads([k_state["weight"], v_state["weight"]])
            if "bias" in k_state:
                bias = _interleave_heads([k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)])
                state["to_kv.bias"] = bias.squeeze(-1)

    def _to_out_fused_addcmul(
        self,
        x: ttnn.Tensor,
        addcmul_residual: ttnn.Tensor,
        addcmul_gate: ttnn.Tensor,
        compute_kernel_config=None,
        parallel_config: DiTParallelConfig | None = None,
        dtype=None,
    ) -> ttnn.Tensor:
        """Fused to_out projection + addcmul: output = residual + (matmul(x, W) + bias) * gate."""
        to_out = self.to_out
        # to_out inlines the AG-matmul rather than calling ColParallelLinear.forward, so it has to
        # honour the activation cast itself. The addcmul residual/gate stay bf16 — the kernel ties
        # their tile size to the weight's, not the activation's — and the output is the residual
        # stream, so it must be pinned back to bf16 rather than inheriting the bf8 activation.
        x = maybe_cast_activation(x, to_out.activation_dtype)
        if to_out.pin_output_bf16:
            dtype = resolve_output_dtype(dtype, x)

        if to_out.fsdp_mesh_axis is not None and to_out.mesh_device.shape[to_out.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(to_out.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=2, mesh_axis=to_out.fsdp_mesh_axis
            )
            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = to_out.weight.data

        if parallel_config is not None and parallel_config.tensor_parallel.factor > 1:
            M, K, N_out = x.padded_shape[-2], weight.padded_shape[-2], weight.padded_shape[-1]
            full_grid = self.mesh_device.compute_with_storage_grid_size()

            # Known shapes route the addcmul to_out to the strided AG-matmul (out = a + scalar*matmul*b)
            fabric_cfg = get_fabric_agmm_config(M, K, N_out, 1, full_grid)
            if fabric_cfg is not None:
                tp_axis = parallel_config.tensor_parallel.mesh_axis
                dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
                fabric_matmul_config = ttnn.MinimalMatmulConfig(
                    M_block_size=fabric_cfg.M_block_size,
                    K_block_size=fabric_cfg.K_block_size,
                    N_block_size=fabric_cfg.N_block_size,
                    subblock_h=fabric_cfg.subblock_h,
                    subblock_w=fabric_cfg.subblock_w,
                    compute_with_storage_grid_size=fabric_cfg.mm_core_grid,
                )
                outputs = ttnn.experimental.strided_all_gather_minimal_matmul_async(
                    x,
                    weight,
                    persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                        x.shape, 3, tp_axis, dtype=x.get_dtype()
                    ),
                    dim=3,
                    multi_device_global_semaphore=self.ccl_manager.get_strided_ag_mm_semaphore(
                        tp_axis, fabric_cfg.num_workers_per_link
                    ),
                    strided_all_gather_core_grid_offset=fabric_cfg.ag_core_grid_offset,
                    num_links=self.ccl_manager.num_links,
                    memory_config_ag=dram,
                    topology=self.ccl_manager.topology,
                    cluster_axis=tp_axis,
                    bias=to_out.bias.data if to_out.bias is not None else None,
                    config=fabric_matmul_config,
                    memory_config_mm=dram,
                    compute_kernel_config=compute_kernel_config or to_out.compute_config,
                    num_workers_per_link=fabric_cfg.num_workers_per_link,
                    num_buffers_per_channel=fabric_cfg.num_buffers_per_channel,
                    read_local_slice_from_input=True,
                    fused_ternary_input_a=addcmul_residual,
                    fused_ternary_input_b=addcmul_gate,
                    fused_ternary_scalar=1.0,
                    chunks=1,
                )
                # Op returns [all_gather_output, matmul_chunk_0]; take the single matmul chunk.
                return outputs[1]

            core_grid = ttnn.CoreCoord(full_grid.x, full_grid.y - 1)
            matmul_config = get_matmul_config(M, K, N_out, core_grid)

            ag_persistent_buffer = self.ccl_manager.get_ag_ping_pong_buffer(
                x.shape, 3, parallel_config.tensor_parallel.mesh_axis, dtype=x.get_dtype()
            )
            ag_global_semaphores = self.ccl_manager.get_ag_ping_pong_semaphore(
                parallel_config.tensor_parallel.mesh_axis
            )
            output = ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=to_out.bias.data if to_out.bias is not None else None,
                config=matmul_config,
                compute_kernel_config=compute_kernel_config or to_out.compute_config,
                persistent_output_buffer=ag_persistent_buffer,
                multi_device_global_semaphore=ag_global_semaphores,
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=parallel_config.tensor_parallel.mesh_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=full_grid.x // self.ccl_manager.num_links,
                num_buffers_per_channel=48 if not is_blackhole() else 24,
                scalar=1.0,
                addcmul_input_tensor1=addcmul_residual,
                addcmul_input_tensor2=addcmul_gate,
                dtype=dtype,
            )[0]
        else:
            M, K, N_out = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
            core_grid = self.mesh_device.compute_with_storage_grid_size()
            matmul_config = get_matmul_config(M, K, N_out, core_grid)

            output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x,
                weight,
                1.0,
                addcmul_residual,
                addcmul_gate,
                bias_tensor=to_out.bias.data if to_out.bias is not None else None,
                config=matmul_config,
                compute_kernel_config=compute_kernel_config or to_out.compute_config,
                dtype=dtype,
            )
        return output

    def _gate_is_live(self) -> bool:
        """True when the gate projection will actually run (and so will gather its input).

        A fused gate has no projection of its own — it falls out of the QKV/Q matmul — so it neither
        runs nor gathers, and ``to_gate_logits`` is never built.
        """
        return self.apply_gated_attention and not self.fuse_gate and self.to_gate_logits.weight._data is not None

    def _compute_gate(
        self, spatial_1BND: ttnn.Tensor, qkv_parallel_config: DiTParallelConfig | None
    ) -> ttnn.Tensor | None:
        """Per-head gate 2 * sigmoid(to_gate_logits(x)); returns (B, H_local, N, 1) or None."""
        if not self._gate_is_live():
            return None

        gate_logits = self.to_gate_logits(spatial_1BND, parallel_config=qkv_parallel_config)
        return self._gate_from_logits(gate_logits)

    def _unpad_gate_logits(self, gate_logits: ttnn.Tensor) -> ttnn.Tensor:
        """Drop the tile padding from the fused gate chunk, leaving n_local_heads real columns."""
        if self.gate_padded_per_device == self.gate_width_per_device:
            return gate_logits
        shape = gate_logits.shape
        return ttnn.slice(
            gate_logits,
            [0, 0, 0, 0],
            [shape[0], shape[1], shape[2], self.gate_width_per_device],
        )

    def _gate_from_logits(self, gate_logits: ttnn.Tensor) -> ttnn.Tensor:
        """2 * sigmoid(logits) as (B, H_local, N, 1); shared by the fused and unfused paths."""
        gate = ttnn.multiply(ttnn.sigmoid(gate_logits), 2.0)
        return ttnn.permute(gate, (1, 3, 2, 0))

    def _sdpa_kwargs(self) -> dict:
        """Recipe kwargs, or (non-Blackhole) the legacy compute config read at call time."""
        return sdpa_kwargs(self.sdpa_precision, self.sdpa_compute_kernel_config)

    def _ring_program_config(self, N: int) -> ttnn.SDPAProgramConfig:
        """Self-attn ring SDPA config (recipe: op-selected chunks for the worker grid)."""
        del N
        return self.ring_sdpa_program_config

    def _dense_program_config(self) -> ttnn.SDPAProgramConfig:
        """Dense self-attn SDPA config (SP=1)."""
        return self.sdpa_program_config

    def _cross_program_config(self, q_seq: int, kv_seq: int) -> ttnn.SDPAProgramConfig:
        """Local cross-attn SDPA config (recipe: the op sizes the chunks for the shape)."""
        del q_seq, kv_seq
        return self.sdpa_program_config

    def _gathered_program_config(self, q_len: int) -> ttnn.SDPAProgramConfig:
        """Padded audio self-attn with gathered K/V (SP>1)."""
        del q_len
        return self.sdpa_program_config

    def _cross_ring_program_config(self, q_len: int) -> ttnn.SDPAProgramConfig:
        """V2A ring cross (is_cross) SDPA config (recipe: the op sizes Q chunks for the audio Q shard)."""
        del q_len
        return self.cross_ring_sdpa_program_config

    def _recipe_mask_kv_len(self, attn_mask, attn_kv_len: int | None) -> int | None:
        """Under a recipe, the logical key length that replaces a key-column padding mask.

        The one LTX-2 mask (padded audio self-attn, built by ``build_audio_masks``) only bars keys
        ``>= audio_N_real``, so a recipe slices K/V to that length instead (same result, less work); the
        caller must state it via ``attn_kv_len`` since it can't be read off the mask. Any other mask (no
        ``attn_kv_len``, or a cross-attention mask) goes to the recipe as ``attn_mask``. ``None`` = no
        slicing."""
        if self.sdpa_precision is None or attn_mask is None:
            return None
        if attn_kv_len is None or not self.is_self:
            return None
        if attn_kv_len <= 0:
            raise ValueError(f"LTX-2: attn_kv_len must be positive (got {attn_kv_len})")
        return attn_kv_len

    @staticmethod
    def _slice_kv(k_BHNE: ttnn.Tensor, v_BHNE: ttnn.Tensor, kv_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Keep the first ``kv_len`` keys/values (the unpadded audio tokens)."""
        n = k_BHNE.shape[2]
        if kv_len > n:
            raise ValueError(f"LTX-2: attn_kv_len={kv_len} exceeds the key length {n}")
        if kv_len == n:
            return k_BHNE, v_BHNE
        b, h, _, d = k_BHNE.shape
        return (
            ttnn.slice(k_BHNE, [0, 0, 0, 0], [b, h, kv_len, d]),
            ttnn.slice(v_BHNE, [0, 0, 0, 0], [b, h, kv_len, d]),
        )

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        N: int,
        prompt_1BLP: ttnn.Tensor | None = None,
        rope_cos: ttnn.Tensor | None = None,
        rope_sin: ttnn.Tensor | None = None,
        trans_mat: ttnn.Tensor | None = None,
        addcmul_residual: ttnn.Tensor | None = None,
        addcmul_gate: ttnn.Tensor | None = None,
        k_rope_cos: ttnn.Tensor | None = None,
        k_rope_sin: ttnn.Tensor | None = None,
        attn_mask: ttnn.Tensor | None = None,
        skip_qk: bool = False,
        kv_replicated: bool = False,
        kv_logical_n: int | None = None,
        attn_kv_len: int | None = None,
    ) -> ttnn.Tensor:
        """Same interface as WanAttention.forward(); pass k_rope_cos/sin for separate K RoPE
        in A2V/V2A cross-attention. ``attn_kv_len`` is the logical key length of a key-column
        ``attn_mask`` (padded audio self-attn); only a recipe uses it (it slices K/V instead of masking)."""
        # Under a recipe a key-length mask becomes a K/V slice; other masks go to the recipe SDPA.
        recipe_kv_len = self._recipe_mask_kv_len(attn_mask, attn_kv_len)
        if rope_cos is not None:
            assert rope_sin is not None
            assert trans_mat is not None, "INTERLEAVED RoPE requires trans_mat (load-time Q/K permute assumes it)"

        use_nonfused_agmm = (self.ccl_manager.topology == ttnn.Topology.Linear) and (
            self.parallel_config.tensor_parallel.factor > 1
        )
        # With the gate on, the gate and Q/QKV projections would each fuse a gather of the same
        # activation; hoisting one explicit gather feeds both and halves the fabric traffic here.
        dedup_gate_gather = (
            LTX_DEDUP_GATE_GATHER
            and not use_nonfused_agmm
            and self._gate_is_live()
            and self.parallel_config.tensor_parallel.factor > 1
        )
        if use_nonfused_agmm or dedup_gate_gather:
            # Cast BEFORE the gather, not inside the linears downstream of it: this path hoists the
            # gather out so Q/QKV (and the gate) can share it, so the fabric payload is this tensor.
            # Casting it at the linear would happen on the already-gathered result and shrink nothing.
            qkv_linear = self.to_qkv if self.is_self else self.to_q
            spatial_1BND = maybe_cast_activation(spatial_1BND, qkv_linear.activation_dtype)
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        qkv_parallel_config = None if (use_nonfused_agmm or dedup_gate_gather) else self.parallel_config

        # When fused, the gate falls out of the QKV/Q matmul below; otherwise it's its own projection.
        gate_bhne = None if self.fuse_gate else self._compute_gate(spatial_1BND, qkv_parallel_config)

        if self.is_self:
            if self.fuse_gate:
                q_1BNF, k_1BNF, v_1BNF, gate_logits = self.to_qkv(
                    spatial_1BND,
                    compute_kernel_config=self.mm_compute_kernel_config,
                    parallel_config=qkv_parallel_config,
                )
                gate_bhne = self._gate_from_logits(self._unpad_gate_logits(gate_logits))
            else:
                q_1BNF, k_1BNF, v_1BNF = self.to_qkv(
                    spatial_1BND,
                    compute_kernel_config=self.mm_compute_kernel_config,
                    parallel_config=qkv_parallel_config,
                )
        else:
            kv_input = prompt_1BLP if prompt_1BLP is not None else spatial_1BND
            # Cross K/V: gather TP-sharded context for to_kv (replicated text prompt is already full).
            kv_parallel_config = None
            if prompt_1BLP is not None and self.parallel_config.tensor_parallel.factor > 1:
                local_k = kv_input.shape[-1]
                kv_is_tp_sharded = local_k * self.parallel_config.tensor_parallel.factor == self.kv_input_dim
                if kv_is_tp_sharded:
                    if use_nonfused_agmm:
                        kv_input = self.ccl_manager.all_gather_persistent_buffer(
                            kv_input, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                        )
                    else:
                        kv_parallel_config = self.parallel_config
            if self.fuse_gate:
                q_1BNF, gate_logits = self.to_q(
                    spatial_1BND,
                    compute_kernel_config=self.mm_compute_kernel_config,
                    parallel_config=qkv_parallel_config,
                )
                gate_bhne = self._gate_from_logits(self._unpad_gate_logits(gate_logits))
            else:
                q_1BNF = self.to_q(
                    spatial_1BND,
                    compute_kernel_config=self.mm_compute_kernel_config,
                    parallel_config=qkv_parallel_config,
                )
            k_1BNF, v_1BNF = self.to_kv(
                kv_input,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=kv_parallel_config,
            )

        # RMSNorm on Q/K fused with the head split (emits BHNE via num_heads_per_device).
        q_BHNE = self.norm_q(q_1BNF, num_heads_per_device=self.n_local_heads)
        k_BHNE = self.norm_k(k_1BNF, num_heads_per_device=self.n_local_heads)

        def create_heads(inp):
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                inp, num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
            )
            return out

        # V still goes through the explicit reshape (no norm to fuse with).
        v_BHNE = create_heads(v_1BNF)

        # Cross-attn K/V must be full-seq for SDPA; gather across SP only when genuinely sharded.
        is_cross = prompt_1BLP is not None
        sp_factor = self.parallel_config.sequence_parallel.factor
        _k_cos_pe = k_rope_cos if k_rope_cos is not None else rope_cos
        # V2A cross: K/V stay SP-sharded (caller passes the sharded K-rope and kv_logical_n) so the
        # ring SDPA fuses the gather instead of an explicit K/V all-gather + local SDPA.
        use_ring_cross = is_cross and sp_factor > 1 and not kv_replicated and kv_logical_n is not None
        if is_cross and sp_factor > 1 and not use_ring_cross:
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis
            if kv_replicated:
                need_gather = False
            elif _k_cos_pe is not None:
                need_gather = k_BHNE.shape[2] < _k_cos_pe.shape[2]
            else:
                # Unknown sharded context with no rope reference: gather conservatively.
                need_gather = True
            if need_gather:
                k_BHNE = self.ccl_manager.all_gather_persistent_buffer(k_BHNE, dim=2, mesh_axis=sp_axis)
                v_BHNE = self.ccl_manager.all_gather_persistent_buffer(v_BHNE, dim=2, mesh_axis=sp_axis)

        if rope_cos is not None:
            _k_cos = _k_cos_pe
            _k_sin = k_rope_sin if k_rope_sin is not None else rope_sin
            q_BHNE = ttnn.experimental.rotary_embedding_llama(
                q_BHNE, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
            )
            k_BHNE = ttnn.experimental.rotary_embedding_llama(
                k_BHNE, _k_cos, _k_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
            )

        # SDPA input quant, applied after RoPE so the rotation still runs at full precision. On the
        # ring paths K/V are the fabric payload (SDPA fuses their SP gather), so this shrinks a
        # collective as well as the QK^T/PV matmuls; dummy_joint is a real SDPA input and must carry
        # the same dtype. Kept separate from the linear activation cast: SDPA inputs have the widest
        # dynamic range in the block and are the likeliest place for bf8 to break accuracy.
        # A recipe owns its SDPA input dtypes (BF16, or LOW_PRECISION-prepared), so skip the quant cast.
        sdpa_input_dtype = None if self.sdpa_precision is not None else getattr(self, "_sdpa_input_dtype", None)
        dummy_joint = self.dummy_joint_input
        if sdpa_input_dtype is not None:
            q_BHNE = maybe_cast_activation(q_BHNE, sdpa_input_dtype)
            k_BHNE = maybe_cast_activation(k_BHNE, sdpa_input_dtype)
            v_BHNE = maybe_cast_activation(v_BHNE, sdpa_input_dtype)
            dummy_joint = maybe_cast_activation(dummy_joint, sdpa_input_dtype)

        if not skip_qk:
            # LOW_PRECISION: after norm/RoPE, before the SDPA call (which fuses the ring K/V gather).
            q_BHNE, k_BHNE, v_BHNE = prepare_recipe_inputs(
                self.sdpa_precision, self.sdpa_kv_dtype, q_BHNE, k_BHNE, v_BHNE
            )

        if skip_qk:
            # STG perturbation: skip Q/K attention, use V passthrough.
            spatial_BHNE = v_BHNE
        elif prompt_1BLP is None:
            if sp_factor > 1 and attn_mask is None:
                spatial_BHNE, _prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                    q_BHNE,
                    k_BHNE,
                    v_BHNE,
                    dummy_joint,
                    dummy_joint,
                    dummy_joint,
                    # The gather buffer must be allocated at the gathered tensor's dtype: the fabric
                    # writes raw tiles into it, so a dtype mismatch is silent corruption, not a cast.
                    persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                        k_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis, dtype=k_BHNE.get_dtype()
                    ),
                    persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                        v_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis, dtype=v_BHNE.get_dtype()
                    ),
                    joint_strategy="rear",
                    logical_n=N,
                    program_config=self._ring_program_config(N),
                    **self._sdpa_kwargs(),
                    dim=2,
                    multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                        self.parallel_config.sequence_parallel.mesh_axis
                    ),
                    num_links=self.ccl_manager.num_links,
                    cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                    mesh_device=self.mesh_device,
                    topology=self.ccl_manager.topology,
                    subdevice_id=self.ccl_manager.ccl_sub_device_id,
                    ccl_core_grid_offset=(self.sdpa_worker_grid[0], 0),
                    use_column_major_ccl=True,
                )
            elif sp_factor > 1 and recipe_kv_len is not None:
                # Recipe on the padded audio self-attn: gather K/V, drop the padded keys, run unmasked.
                sp_axis = self.parallel_config.sequence_parallel.mesh_axis
                k_full = self.ccl_manager.all_gather_persistent_buffer(k_BHNE, dim=2, mesh_axis=sp_axis)
                v_full = self.ccl_manager.all_gather_persistent_buffer(v_BHNE, dim=2, mesh_axis=sp_axis)
                k_full, v_full = self._slice_kv(k_full, v_full, recipe_kv_len)
                spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                    q_BHNE,
                    k_full,
                    v_full,
                    is_causal=False,
                    program_config=self._gathered_program_config(q_BHNE.shape[2]),
                    **self._sdpa_kwargs(),
                )
            elif sp_factor > 1:
                # Masked audio self-attn: gather K/V, keep Q sharded; gather+local SDPA beats ring-joint here.
                sp_axis = self.parallel_config.sequence_parallel.mesh_axis
                k_full = self.ccl_manager.all_gather_persistent_buffer(k_BHNE, dim=2, mesh_axis=sp_axis)
                v_full = self.ccl_manager.all_gather_persistent_buffer(v_BHNE, dim=2, mesh_axis=sp_axis)
                spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                    q_BHNE,
                    k_full,
                    v_full,
                    attn_mask=attn_mask,
                    is_causal=False,
                    program_config=self._gathered_program_config(q_BHNE.shape[2]),
                    **self._sdpa_kwargs(),
                )
            elif recipe_kv_len is not None:
                # Recipe on the padded audio self-attn (SP=1): drop the padded keys, run unmasked.
                k_BHNE, v_BHNE = self._slice_kv(k_BHNE, v_BHNE, recipe_kv_len)
                spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                    q_BHNE,
                    k_BHNE,
                    v_BHNE,
                    is_causal=False,
                    program_config=self._dense_program_config(),
                    **self._sdpa_kwargs(),
                )
            else:
                spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                    q_BHNE,
                    k_BHNE,
                    v_BHNE,
                    attn_mask=attn_mask,
                    is_causal=False,
                    program_config=self._dense_program_config(),
                    **self._sdpa_kwargs(),
                )
        elif use_ring_cross:
            # Short audio Q attends non-causally to the SP-sharded video K/V; is_cross fuses the
            # K/V gather into the ring SDPA. Output is the per-device Q shard (same as local SDPA).
            sp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis
            spatial_BHNE, _prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                dummy_joint,
                dummy_joint,
                dummy_joint,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k_BHNE.shape, 2, sp_mesh_axis, dtype=k_BHNE.get_dtype()
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v_BHNE.shape, 2, sp_mesh_axis, dtype=v_BHNE.get_dtype()
                ),
                joint_strategy="rear",
                logical_n=kv_logical_n,
                is_cross=True,
                program_config=self._cross_ring_program_config(q_BHNE.shape[2]),
                **self._sdpa_kwargs(),
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(sp_mesh_axis),
                num_links=self.ccl_manager.num_links,
                cluster_axis=sp_mesh_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(self.sdpa_worker_grid[0], 0),
                use_column_major_ccl=True,
            )
        else:
            # Cross-attention: K/V full-seq, Q SP-sharded so local SDPA returns the local shard.
            spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                attn_mask=attn_mask,
                is_causal=False,
                program_config=self._cross_program_config(q_BHNE.shape[2], k_BHNE.shape[2]),
                **self._sdpa_kwargs(),
            )

        # Apply per-head gate in BHNE space.
        if gate_bhne is not None:
            spatial_BHNE = ttnn.multiply(spatial_BHNE, gate_bhne)

        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        # Ring fuses the TP all-gather into the to_out matmul; only Linear needs explicit AG.
        addcmul_fused = addcmul_residual is not None and addcmul_gate is not None
        to_out_explicit_ag = self.parallel_config.tensor_parallel.factor > 1 and use_nonfused_agmm
        if to_out_explicit_ag:
            # Same ordering rule as the QKV gather above: shrink the tensor before it crosses.
            spatial_1BND = maybe_cast_activation(spatial_1BND, self.to_out.activation_dtype)
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        if addcmul_fused:
            spatial_1BND = self._to_out_fused_addcmul(
                spatial_1BND,
                addcmul_residual,
                addcmul_gate,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=None if use_nonfused_agmm else self.parallel_config,
            )
        else:
            spatial_1BND = self.to_out(
                spatial_1BND,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=None if to_out_explicit_ag else self.parallel_config,
            )

        return spatial_1BND
