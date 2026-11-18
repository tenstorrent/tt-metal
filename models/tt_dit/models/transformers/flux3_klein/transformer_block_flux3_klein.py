# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ....blocks.attention_opt import Attention
from ....layers.linear import ColParallelLinear, RowParallelLinear
from ....layers.module import Module
from ....layers.normalization import DistributedLayerNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.matmul import get_agmm_config
from ....utils.tensor import prepare_weight_for_concatenated_input

HIDDEN = 3072
HEADS = 24
HEAD_DIM = 128
MLP_RATIO = 3
# Ring SDPA (q, k) chunks by the rows each device holds (image + text), where the shared table keyed by the logical
# length picks badly; other row counts keep the Attention defaults.
# At the 4096 rows on 116 cores (12 x 9 + the CCL row's 8 free cores): 2 rounds of q 224, and 3 rounds of q 160 at
# 4096 + 1 reference (2048-row text bucket).
RING_SDPA_CHUNKS_BY_LOCAL_ROWS: dict[int, tuple[int, int]] = {
    2560: (160, 512),
    2688: (160, 512),
    2816: (160, 512),
    3072: (192, 512),
    8192: (224, 512),
    8320: (224, 512),
    8960: (160, 512),
    # DP4 (SP4 x TP2) at 4096, every image / joined row count: SP4 has no shared-table key, so its default k chunk is 256 where
    # SP8 x TP4's is 512, and the extra softmax merges cost 7 dB of image vs SP8 x TP4 (job 219220: 28.4 -> 35.3 dB with these)
    **{rows: (224, 512) for rows in (16384, 16640, 16896, 17408, 17920, 18432, 20736, 20992, 21504)},
}
# The ring all-gather keeps the CCL row's cores x < 4 with 2 links
RING_SDPA_CCL_ROW_START_X = 4
# linear1 (QKV + MLP-in as one all-gather matmul): N cores of its transposed grid, and the rows from which it runs
LINEAR1_N_CORES = 9
LINEAR1_MIN_ROWS = 8192


class Flux3KleinAttention(Attention):
    """Ring SDPA on the streaming compute path with fp32 dest accumulation (bf16 score block), also on the CCL row's
    cores from RING_SDPA_CCL_ROW_START_X on, with RING_SDPA_CHUNKS_BY_LOCAL_ROWS's chunks where it has the rows."""

    def get_ring_sdpa_program_config(self, per_device_seq_len: int) -> ttnn.SDPAProgramConfig:
        rows = per_device_seq_len // self.parallel_config.sequence_parallel.factor
        key = ("local", rows)
        if key not in self.ring_sdpa_program_config:
            chunks = RING_SDPA_CHUNKS_BY_LOCAL_ROWS.get(rows)
            if chunks is None:
                config = super().get_ring_sdpa_program_config(per_device_seq_len)
            else:
                config = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.get_ring_sdpa_core_grid(per_device_seq_len),
                    q_chunk_size=chunks[0],
                    k_chunk_size=chunks[1],
                    exp_approx_mode=self.ring_sdpa_exp_approx,
                )
            config.streaming_fp32_dest = True
            config.ccl_row_sdpa_start_x = RING_SDPA_CCL_ROW_START_X
            self.ring_sdpa_program_config[key] = config
        return self.ring_sdpa_program_config[key]


class Flux3KleinTransformerBlock(Module):
    """Single-stream block: one LayerNorm modulated by (shift, scale + 1), QKV + SwiGLU MLP from the same input,
    and one output projection over [attn | mlp] gated back into the residual.

    ``linear1`` runs QKV and MLP-in as one all-gather matmul over [W_qkv | W_mlp] (swiglu_last_chunk, grouped N split) from
    LINEAR1_MIN_ROWS rows on, where the two resolve the same K block: byte-identical to the two ops."""

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        linear1: bool = False,
    ) -> None:
        super().__init__()
        tp_ring = ccl_manager.axis_topology(parallel_config.tensor_parallel.mesh_axis) == ttnn.Topology.Ring
        if linear1 and (is_fsdp or not tp_ring):
            raise ValueError("linear1 needs a ring TP axis and resident (non-FSDP) weights")
        if linear1 and device.compute_with_storage_grid_size().y < LINEAR1_N_CORES:
            raise ValueError(f"linear1 needs {LINEAR1_N_CORES} rows of cores")

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        mlp_hidden_dim = MLP_RATIO * HIDDEN
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.attn = Flux3KleinAttention(
            query_dim=HIDDEN,
            head_dim=HEAD_DIM,
            heads=HEADS,
            out_dim=HIDDEN,
            added_kv_proj_dim=0,
            pre_only=True,
            proj_bias=False,
            eps=1e-6,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=None,
            use_spatial_weights_for_prompt=True,
            per_head_norm=True,
            is_fsdp=is_fsdp,
            shard_prompt=False,
            ring_sdpa_q_chunk_max=128,
            ring_sdpa_exp_approx=True,
            sdpa_fp32_dest_acc=True,
        )

        self.norm = DistributedLayerNorm(
            HIDDEN,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=device,
            ccl_manager=ccl_manager,
        )
        self.proj_mlp = ColParallelLinear(
            HIDDEN,
            mlp_hidden_dim,
            bias=False,
            activation_fn="swiglu",
            mesh_device=device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.proj_out = RowParallelLinear(
            HIDDEN + mlp_hidden_dim,
            HIDDEN,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        self._mlp_hidden_dim = mlp_hidden_dim
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._mesh_device = device
        self._linear1 = linear1
        self._linear1_k_blocks: dict[int, int | None] = {}
        self._linear1_weight = None
        self._barrier_tile = None
        self.register_eviction_hook(self._deallocate_linear1)

    def _deallocate_linear1(self) -> None:
        for t in (self._linear1_weight, self._barrier_tile):
            if t is not None:
                ttnn.deallocate(t)
        self._linear1_weight = self._barrier_tile = None

    def _tp_barrier(self) -> None:
        """A one-tile all-gather over the TP axis: no device of the group gets past it before every device reached it."""
        if self._barrier_tile is None:
            self._barrier_tile = ttnn.from_torch(
                torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self._mesh_device
            )
        self._ccl_manager.all_gather(
            self._barrier_tile,
            dim=3,
            mesh_axis=self._parallel_config.tensor_parallel.mesh_axis,
            use_hyperparams=False,
            use_persistent_buffer=True,
        )

    def _agmm_config(self, rows: int, n: int, core_grid: ttnn.CoreCoord, **kwargs):
        tp = self._parallel_config.tensor_parallel
        return get_agmm_config(
            rows,
            HIDDEN,
            n,
            full_grid=self._mesh_device.compute_with_storage_grid_size(),
            cluster_size=tp.factor,
            num_links=self._ccl_manager.num_links,
            core_grid=core_grid,
            use_heuristic=True,
            **kwargs,
        )

    def _linear1_k_block(self, rows: int) -> int | None:
        """The K block QKV and MLP-in resolve at this row count, None where they differ (linear1 does not run)."""
        if rows not in self._linear1_k_blocks:
            grid = Attention.get_core_grid(rows, self._mesh_device.compute_with_storage_grid_size())
            _, cfg_q, _ = self._agmm_config(rows, self.attn.to_qkv.weight.data.padded_shape[-1], grid)
            _, cfg_m, _ = self._agmm_config(
                rows, self.proj_mlp.weight.data.padded_shape[-1], grid, fuse_swiglu=self.proj_mlp.fuse_swiglu
            )
            self._linear1_k_blocks[rows] = cfg_q.K_block_size if cfg_q.K_block_size == cfg_m.K_block_size else None
        return self._linear1_k_blocks[rows]

    def _linear1_qkv_mlp(self, x_norm: ttnn.Tensor, k_block: int) -> list[ttnn.Tensor]:
        """[q, k, v, mlp] from one all-gather matmul with QKV's blocking (at the shared K block) and MLP-in's compute
        config, on LINEAR1_N_CORES N cores so the 72 QKV / 144 MLP tiles per device split evenly."""
        attn = self.attn
        if self._linear1_weight is None:
            self._linear1_weight = ttnn.concat([attn.to_qkv.weight.data, self.proj_mlp.weight.data], dim=-1)
        tp = self._parallel_config.tensor_parallel
        n_qkv = attn.to_qkv.weight.data.padded_shape[-1]
        n_mlp = self.proj_mlp.weight.data.padded_shape[-1]
        grid = ttnn.CoreCoord(self._mesh_device.compute_with_storage_grid_size().x, LINEAR1_N_CORES)
        _, cfg, workers = self._agmm_config(x_norm.padded_shape[-2], n_qkv, grid)
        # swiglu_last_chunk: an even N block that divides both groups' per-core widths (8 QKV / 16 MLP tiles), and
        # subblock_w dividing it
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=cfg.M_block_size,
            K_block_size=k_block,
            N_block_size=cfg.N_block_size if cfg.N_block_size in (4, 8) else 8,
            subblock_h=cfg.subblock_h,
            subblock_w=min(cfg.subblock_w, 4),
            compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
        )
        third = n_qkv // 3
        return list(
            ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x_norm,
                weight_tensor=self._linear1_weight,
                bias_tensor=None,
                config=cfg,
                compute_kernel_config=self.proj_mlp.compute_config,
                persistent_output_buffer=self._ccl_manager.get_ag_ping_pong_buffer(
                    x_norm.shape, -1, tp.mesh_axis, dtype=x_norm.get_dtype()
                ),
                multi_device_global_semaphore=self._ccl_manager.get_ag_ping_pong_semaphore(tp.mesh_axis),
                num_links=self._ccl_manager.num_links,
                topology=self._ccl_manager.topology,
                cluster_axis=tp.mesh_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=workers,
                num_buffers_per_channel=attn.to_qkv._ag_num_buffers_per_channel,
                chunks=4,
                chunk_sizes=[third, third, third, n_mlp],
                swiglu_last_chunk=True,
            )
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for src, dst in (("q_proj", "attn.to_q"), ("k_proj", "attn.to_k"), ("v_proj", "attn.to_v")):
            state[f"{dst}.weight"] = state.pop(f"{src}.weight")
        for src, dst in (
            ("norm.query_norm.scale", "attn.norm_q.weight"),
            ("norm.key_norm.scale", "attn.norm_k.weight"),
        ):
            state[dst] = state.pop(src)
        w = state.pop("mlp_in.weight")
        state["proj_mlp.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)
        state["proj_out.weight"] = prepare_weight_for_concatenated_input(
            torch.cat([state.pop("attn_out.weight"), state.pop("mlp_out.weight")], dim=1),
            [HIDDEN, self._mlp_hidden_dim],
            device_count=self._tp_factor,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        mod: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
        logical_length: int | ttnn.Tensor,
    ) -> ttnn.Tensor:
        shift, scale_plus_one, gate = mod
        x_norm = ttnn.squeeze(self.norm(ttnn.unsqueeze(x, 0), dynamic_weight=scale_plus_one, dynamic_bias=shift), 0)
        rows = int(x_norm.padded_shape[-2])
        k_block = self._linear1_k_block(rows) if self._linear1 and rows >= LINEAR1_MIN_ROWS else None
        if k_block is not None:
            q, k, v, mlp = self._linear1_qkv_mlp(x_norm, k_block)
            attn, _ = self.attn.forward(
                sequence_1=x_norm,
                sequence_1_rope=rope,
                sequence_1_length=logical_length,
                precomputed_qkv_flat=(q, k, v),
            )
            # MLP-in's all-gather now runs before the attention: restore a TP sync point before proj_out, whose
            # reduce-scatter also handshakes with its peers before writing into them
            self._tp_barrier()
            return self.proj_out.forward_fused_addcmul([attn, mlp], x, gate, scalar=1.0, rs_barrier=True)
        attn, _ = self.attn.forward(sequence_1=x_norm, sequence_1_rope=rope, sequence_1_length=logical_length)
        mlp = self.proj_mlp(
            x_norm,
            parallel_config=self._parallel_config,
            use_heuristic_mmcfg=True,
            core_grid=Attention.get_core_grid(x_norm.shape[-2], self._mesh_device.compute_with_storage_grid_size()),
        )
        if self._ccl_manager.axis_topology(self._parallel_config.tensor_parallel.mesh_axis) == ttnn.Topology.Ring:
            return self.proj_out.forward_fused_addcmul([attn, mlp], x, gate, scalar=1.0)
        return ttnn.addcmul(x, self.proj_out([attn, mlp]), gate)
