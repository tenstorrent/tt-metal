# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from loguru import logger

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
SDPA_Q_CHUNK_MAX = 128
# Ring SDPA (q, k) chunks by per-device rows (SP8 x TP4): q fills whole rounds of (head, q chunk) units on the SDPA
# grid, k divides the shard. Rows not listed keep the shared rule (q clamped to SDPA_Q_CHUNK_MAX).
RING_SDPA_CHUNKS_BY_LOCAL_ROWS: dict[int, tuple[int, int]] = {
    # 2048x2048: 6 heads x 16 q chunks = 96 units, one round
    2560: (160, 512),
    # 2048x2048 + 1 reference: 96 units, one round
    3072: (192, 512),
    # 2048x2048 + 1 reference, 1024-row text bucket: 102 units, one round
    2688: (160, 512),
    # 4096x4096, 1024-row text bucket: k 640 divides the shard
    8320: (128, 640),
    # 2048x2048 + 1 reference, 2048-row text bucket: 108 units, one round
    2816: (160, 512),
    # 4096x4096 + 1 reference, 2048-row text bucket: k 640 divides the shard
    8960: (128, 640),
}

# Streaming path on the 116-core grid (12 x 9 + the CCL row's 8 free cores): 2 rounds of q 224 at the 4096 rows,
# 3 rounds of q 160 at 4096 + 1 reference (2048-row text bucket).
RING_SDPA_CHUNKS_STREAMING_BY_LOCAL_ROWS: dict[int, tuple[int, int]] = {
    8192: (224, 512),
    8320: (224, 512),
    8960: (160, 512),
}
# SDPA also runs on the CCL row's cores from this x (the all-gather keeps x < 4 with 2 links)
RING_SDPA_CCL_ROW_START_X = 4
LINEAR1_N_CORES = 9  # N cores of linear1's transposed grid
LINEAR1_MIN_ROWS = 8192  # linear1 only at the 4096 row counts, where it is bit-identical to QKV + MLP-in
_LINEAR1_K_BLOCK: dict[int, int | None] = {}  # rows -> the K block QKV and MLP-in share there (None: they differ)


class Flux3KleinAttention(Attention):
    # ring SDPA on the streaming compute path with fp32 dest (bf16 score block); False: the compute_common.hpp path
    ring_sdpa_streaming_fp32_dest = True

    def forward(self, *, sequence_1: ttnn.Tensor, **kwargs):
        self._local_rows = int(sequence_1.shape[-2])
        return super().forward(sequence_1=sequence_1, **kwargs)

    def get_ring_sdpa_program_config(self, per_device_seq_len: int) -> ttnn.SDPAProgramConfig:
        rows = getattr(self, "_local_rows", None)
        chunks = (self.ring_sdpa_streaming_fp32_dest and RING_SDPA_CHUNKS_STREAMING_BY_LOCAL_ROWS.get(rows)) or (
            RING_SDPA_CHUNKS_BY_LOCAL_ROWS.get(rows)
        )
        if chunks is None:
            config = super().get_ring_sdpa_program_config(per_device_seq_len)
        else:
            key = ("local", rows, chunks)
            if key not in self.ring_sdpa_program_config:
                self.ring_sdpa_program_config[key] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.get_ring_sdpa_core_grid(per_device_seq_len),
                    q_chunk_size=chunks[0],
                    k_chunk_size=chunks[1],
                    exp_approx_mode=self.ring_sdpa_exp_approx,
                )
            config = self.ring_sdpa_program_config[key]
        config.streaming_fp32_dest = self.ring_sdpa_streaming_fp32_dest
        config.ccl_row_sdpa_start_x = RING_SDPA_CCL_ROW_START_X if self.ring_sdpa_streaming_fp32_dest else None
        return config


class Flux3KleinTransformerBlock(Module):
    """Single-stream block: one LayerNorm modulated by (shift, scale + 1), QKV + SwiGLU MLP from the same input,
    and one output projection over [attn | mlp] gated back into the residual."""

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

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
        )
        self.attn.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.attn.ring_sdpa_exp_approx = True
        self.attn.ring_sdpa_chunk_size_map = {
            key: {length: (min(q, SDPA_Q_CHUNK_MAX), k) for length, (q, k) in table.items()}
            for key, table in Attention.ring_sdpa_chunk_size_map.items()
        }
        self.attn.default_ring_sdpa_chunk_size = {
            length: (min(q, SDPA_Q_CHUNK_MAX), k) for length, (q, k) in Attention.default_ring_sdpa_chunk_size.items()
        }

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
        self.proj_mlp.swiglu_overlap = True
        self._mesh_device = device
        self._barrier_tile = None
        self._l1_weight = None  # [W_qkv | W_mlp], built once from the loaded weights, outside the state dict
        self._l1_disabled = False

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

    def _linear1_k_block(self, rows: int, n_qkv: int, n_mlp: int, full) -> int | None:
        """The K block QKV and MLP-in resolve at this row count, or None if they differ. linear1 sums K in the same
        blocks and order for every N block, so its outputs are bit-identical to the two separate ops."""
        kb = _LINEAR1_K_BLOCK.get(rows)
        if kb is None and rows not in _LINEAR1_K_BLOCK:
            tp = self._parallel_config.tensor_parallel
            grid = Attention.get_core_grid(rows, full)
            common = {
                "full_grid": full,
                "cluster_size": tp.factor,
                "num_links": self._ccl_manager.num_links,
                "core_grid": grid,
                "use_heuristic": True,
            }
            _, cfg_q, _ = get_agmm_config(rows, HIDDEN, n_qkv, **common)
            _, cfg_m, _ = get_agmm_config(rows, HIDDEN, n_mlp, fuse_swiglu=self.proj_mlp.fuse_swiglu, **common)
            kb = cfg_q.K_block_size if cfg_q.K_block_size == cfg_m.K_block_size else None
            _LINEAR1_K_BLOCK[rows] = kb
            logger.info(f"linear1 rows {rows}: K block QKV {cfg_q.K_block_size} / MLP-in {cfg_m.K_block_size}")
        return kb

    def _linear1(self, x_norm: ttnn.Tensor) -> list[ttnn.Tensor] | None:
        """QKV + MLP-in as one all-gather-matmul over [W_qkv | W_mlp] per device (swiglu_chunk 3, grouped N split),
        with QKV's blocking and grid and MLP-in's compute config. Returns [q, k, v, mlp]."""
        attn = self.attn
        if self._l1_weight is None:
            self._l1_weight = ttnn.concat([attn.to_qkv.weight.data, self.proj_mlp.weight.data], dim=-1)
        tp = self._parallel_config.tensor_parallel
        n_qkv = attn.to_qkv.weight.data.padded_shape[-1]
        n_mlp = self.proj_mlp.weight.data.padded_shape[-1]
        full = self._mesh_device.compute_with_storage_grid_size()
        # the grouped N split needs the 72 QKV / 144 MLP tiles per device to split evenly over the N cores
        if full.y < LINEAR1_N_CORES:
            raise RuntimeError(f"linear1 needs {LINEAR1_N_CORES} N cores, grid {full.x}x{full.y}")
        grid = ttnn.CoreCoord(full.x, LINEAR1_N_CORES)
        k_block = self._linear1_k_block(int(x_norm.padded_shape[-2]), n_qkv, n_mlp, full)
        if k_block is None:
            return None
        _, cfg, workers = get_agmm_config(
            x_norm.padded_shape[-2],
            HIDDEN,
            n_qkv,
            full_grid=full,
            cluster_size=tp.factor,
            num_links=self._ccl_manager.num_links,
            core_grid=grid,
            use_heuristic=True,
        )
        if cfg.N_block_size not in (4, 8) or cfg.K_block_size != k_block:
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=cfg.M_block_size,
                K_block_size=k_block,
                N_block_size=cfg.N_block_size if cfg.N_block_size in (4, 8) else 8,
                subblock_h=cfg.subblock_h,
                subblock_w=min(cfg.subblock_w, 4),
                compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
            )
        buf = self._ccl_manager.get_ag_ping_pong_buffer(x_norm.shape, -1, tp.mesh_axis, dtype=x_norm.get_dtype())
        third = n_qkv // 3
        return list(
            ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x_norm,
                weight_tensor=self._l1_weight,
                bias_tensor=None,
                config=cfg,
                compute_kernel_config=self.proj_mlp.compute_config,
                persistent_output_buffer=buf,
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
                swiglu_chunk=3,
                swiglu_overlap=True,
            )
        )

    def _forward_linear1(self, x, x_norm, rope, gate, logical_length) -> ttnn.Tensor | None:
        try:
            l1_out = self._linear1(x_norm)
        except RuntimeError as err:
            logger.warning(f"linear1 rejected for {tuple(x_norm.shape)} ({str(err)[:200]}); using QKV + MLP-in")
            self._l1_disabled = True
            if self._l1_weight is not None:
                ttnn.deallocate(self._l1_weight)
                self._l1_weight = None
            return None
        if l1_out is None:
            return None
        q, k, v, mlp = l1_out
        attn, _ = self.attn.forward(
            sequence_1=x_norm,
            sequence_1_rope=rope,
            sequence_1_length=logical_length,
            precomputed_qkv_flat=(q, k, v),
        )
        # linear1 moved MLP-in's all-gather ahead of the attention: restore the TP ordering it gave before proj_out,
        # and let proj_out's reduce-scatter handshake with its peers before writing into them
        self._tp_barrier()
        return self.proj_out.forward_fused_addcmul([attn, mlp], x, gate, scalar=1.0, rs_barrier=True)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for src, dst in (("q_proj", "attn.to_q"), ("k_proj", "attn.to_k"), ("v_proj", "attn.to_v")):
            if f"{src}.weight" in state:
                state[f"{dst}.weight"] = state.pop(f"{src}.weight")
        for src, dst in (
            ("norm.query_norm.scale", "attn.norm_q.weight"),
            ("norm.key_norm.scale", "attn.norm_k.weight"),
        ):
            if src in state:
                state[dst] = state.pop(src)
        if "mlp_in.weight" in state:
            w = state.pop("mlp_in.weight")
            state["proj_mlp.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)
        if "attn_out.weight" in state and "mlp_out.weight" in state:
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
        if (
            int(x_norm.padded_shape[-2]) >= LINEAR1_MIN_ROWS
            and self._ccl_manager.topology == ttnn.Topology.Ring
            and not self._l1_disabled
            and rope is not None
        ):
            out = self._forward_linear1(x, x_norm, rope, gate, logical_length)
            if out is not None:
                return out
        attn, _ = self.attn.forward(sequence_1=x_norm, sequence_1_rope=rope, sequence_1_length=logical_length)
        mlp = self.proj_mlp(
            x_norm,
            parallel_config=self._parallel_config,
            use_heuristic_mmcfg=True,
            core_grid=Attention.get_core_grid(x_norm.shape[-2], self._mesh_device.compute_with_storage_grid_size()),
        )
        if self._ccl_manager.topology == ttnn.Topology.Ring:
            return self.proj_out.forward_fused_addcmul([attn, mlp], x, gate, scalar=1.0)
        return ttnn.addcmul(x, self.proj_out([attn, mlp]), gate)
