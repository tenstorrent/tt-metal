# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ....layers.linear import ColParallelLinear
from ....layers.module import Module
from ....layers.normalization import RMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.substate import pop_substate, rename_substate
from ....utils.tensor import bf16_tensor


class MiniMaxH3Attention(Module):
    """Full self-attention over one packed sequence. MiniMax-H3 has no cross-attention.

    Two things differ from `WanAttention` and drive the shape of this module:

    * The attention inner dim (`num_heads * head_dim` = 7168) is *larger* than the residual stream
      (`hidden_size` = 5376), so `to_q/k/v` widen 5376 -> 7168 and `to_out` narrows 7168 -> 5376.
      Nothing here may assume `inner_dim == hidden_size`. Every projection is bias-free.
    * The query/key norms are RMSNorms over `head_dim` (128), not over the TP-sharded residual
      stream. Once heads are split, each head's 128 channels live entirely on one device, so this is
      a plain local `RMSNorm` and needs no CCL -- unlike Wan's `DistributedRMSNorm`.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        qk_norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        is_sequence_parallel: bool = True,
    ) -> None:
        super().__init__()

        # is_sequence_parallel=False means the sequence is *replicated* on the SP axis rather than
        # fractured across it, so attention runs locally with plain SDPA and no ring all-gather. The
        # token refiner uses that: its text stream is short and every SP device holds all of it.
        self.is_sequence_parallel = is_sequence_parallel

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.qk_norm_eps = qk_norm_eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        tp_factor = parallel_config.tensor_parallel.factor
        assert num_heads % tp_factor == 0, f"{num_heads} heads must divide across TP={tp_factor}"
        self.n_local_heads = num_heads // tp_factor
        self.tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis
        self.sp_mesh_axis = parallel_config.sequence_parallel.mesh_axis

        fsdp_mesh_axis = self.sp_mesh_axis if is_fsdp else None

        # Fused QKV: one matmul, output split into three. The state dict is rearranged in
        # `_prepare_torch_state` so that column-parallel fracturing hands each device the same
        # 14 heads of q, k and v.
        self.to_qkv = ColParallelLinear(
            hidden_size,
            3 * self.inner_dim,
            chunks=3,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.to_out = ColParallelLinear(
            self.inner_dim,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Per-head norms: last dim is head_dim, so these are local.
        self.norm_q = RMSNorm(
            embedding_dim=head_dim,
            norm_eps=qk_norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.norm_k = RMSNorm(
            embedding_dim=head_dim,
            norm_eps=qk_norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )

        # Ring SDPA reuses the joint-attention entry point with empty joint inputs, as WanAttention does.
        self.dummy_joint_input = bf16_tensor(torch.zeros((1, self.n_local_heads, 0, head_dim)), device=mesh_device)

        full_grid = mesh_device.compute_with_storage_grid_size()
        self.full_grid = full_grid
        self.sdpa_worker_grid = (full_grid.x - 1, full_grid.y)  # reserve last column for CCL
        self._sdpa_program_configs: dict[tuple[int, bool], ttnn.SDPAProgramConfig] = {}

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
        )
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------ weights

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "to_out.0", "to_out")

        def _interleave_heads(tensors: list[torch.Tensor]) -> torch.Tensor:
            """Reorder [out, in] weights so TP column-fracturing gives each device matching heads.

            Out dim is `num_heads * head_dim`. Reshaping it to [n_dev, n_local_heads, head_dim] and
            concatenating the tensors on the heads axis puts device `d`'s q, k and v heads
            contiguously inside shard `d`, which is also the order `chunks=3` splits them back out in.
            Device `d` therefore owns canonical heads `[d * n_local, (d + 1) * n_local)`, so simply
            all-gathering the attention output on TP rebuilds the canonical 7168-channel order that
            `to_out` expects.
            """
            n_dev = self.parallel_config.tensor_parallel.factor
            tensors = [t.T for t in tensors]  # -> [in, out]
            tensors = [t.reshape(t.shape[0], n_dev, self.n_local_heads, self.head_dim) for t in tensors]
            merged = torch.cat(tensors, dim=2)
            merged = merged.reshape(merged.shape[0], len(tensors) * self.inner_dim)
            return merged.T

        q_state = pop_substate(state, "to_q")
        k_state = pop_substate(state, "to_k")
        v_state = pop_substate(state, "to_v")
        state["to_qkv.weight"] = _interleave_heads([q_state["weight"], k_state["weight"], v_state["weight"]])

    # ------------------------------------------------------------------ helpers

    def _sdpa_program_config(self, seq_local: int, *, ring: bool) -> ttnn.SDPAProgramConfig:
        """Chunk sizes capped to the local sequence length, so short sequences stay legal."""
        key = (seq_local, ring)
        if key not in self._sdpa_program_configs:
            tile = ttnn.TILE_SIZE
            q_chunk = max(tile, min(256, (seq_local // tile) * tile))
            k_chunk = max(tile, min(512, (seq_local // tile) * tile))
            grid = (
                ttnn.CoreCoord(*self.sdpa_worker_grid) if ring else ttnn.CoreCoord(self.full_grid.x, self.full_grid.y)
            )
            self._sdpa_program_configs[key] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,  # NOTE: False is more correct
            )
        return self._sdpa_program_configs[key]

    def _apply_rope(self, x_BHNE: ttnn.Tensor, rope_cos: ttnn.Tensor, rope_sin: ttnn.Tensor) -> ttnn.Tensor:
        """MiniMax-H3 partial rotary embedding.

        Rotates the leading `rotary_dim` (= 2 * 3 * rope_freq_dim = 96) channels of every head and
        passes the remaining `head_dim - rotary_dim` (= 32) through unchanged. The rotate-half split
        is over those 96 channels -- pairing channel `i` with `i + 48` -- and *not* over the full
        head_dim.

        That is why the fused rope path Wan uses is unavailable here:
        `ttnn.experimental.rotary_embedding_llama` applies its `trans_mat` rotate-half across the
        whole head_dim, and `ttnn.experimental.rotate_half` rejects a 96-wide input outright
        ("Input X dimension (96) must be divisible by 64 for tiling") because a 48-channel half is
        not tile-aligned. So the rotation is decomposed here. Slicing at 48 is legal and exact even
        though it is not tile-aligned; the numerics match the reference to bf16 rounding.
        """
        b, h, n, _ = x_BHNE.shape
        rotary_dim = rope_cos.shape[-1]
        half = rotary_dim // 2

        rot = ttnn.slice(x_BHNE, [0, 0, 0, 0], [b, h, n, rotary_dim])
        x1 = ttnn.slice(rot, [0, 0, 0, 0], [b, h, n, half])
        x2 = ttnn.slice(rot, [0, 0, 0, half], [b, h, n, rotary_dim])
        rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)

        # cos/sin are [1, 1, n, rotary_dim] and broadcast over the heads axis.
        out = ttnn.add(ttnn.mul(rot, rope_cos), ttnn.mul(rotated, rope_sin))

        if rotary_dim == self.head_dim:
            return out
        passthrough = ttnn.slice(x_BHNE, [0, 0, 0, rotary_dim], [b, h, n, self.head_dim])
        return ttnn.concat([out, passthrough], dim=-1)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        N: int | None = None,
        rope_cos: ttnn.Tensor | None = None,
        rope_sin: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        spatial_1BND: fractured hidden_size on TP; fractured N on SP when `is_sequence_parallel`,
            otherwise replicated on SP.
        rope_cos/rope_sin: [1, 1, N_local, rotary_dim], fractured N on SP, replicated on TP. Both
            None skips the rotary embedding entirely, as the token refiner requires.
        N: logical (unfractured) sequence length. Only needed for ring attention.

        Returns the attention output with the same distribution as the input.
        """
        assert (rope_cos is None) == (rope_sin is None), "rope_cos and rope_sin must be given together"

        tp_factor = self.parallel_config.tensor_parallel.factor
        sp_factor = self.parallel_config.sequence_parallel.factor
        use_ring = self.is_sequence_parallel and sp_factor > 1
        assert not (use_ring and N is None), "ring attention needs the logical sequence length N"

        # NOTE: bringup takes the unfused path -- an explicit all-gather feeding a plain
        # column-parallel matmul. Folding these into `all_gather_minimal_matmul_async` (as
        # WanAttention does on ring topologies) is the performance follow-up.
        if tp_factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.tp_mesh_axis
            )

        q_1BNF, k_1BNF, v_1BNF = self.to_qkv(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)

        def create_heads(inp: ttnn.Tensor) -> ttnn.Tensor:
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                inp,
                num_heads=self.n_local_heads,
                num_kv_heads=0,
                transpose_k_heads=False,
            )
            return out

        q_BHNE = create_heads(q_1BNF)
        k_BHNE = create_heads(k_1BNF)
        v_BHNE = create_heads(v_1BNF)

        # QK-norm over head_dim, then partial rope. Order matches the reference processor.
        q_BHNE = self.norm_q(q_BHNE)
        k_BHNE = self.norm_k(k_BHNE)

        if rope_cos is not None:
            q_BHNE = self._apply_rope(q_BHNE, rope_cos, rope_sin)
            k_BHNE = self._apply_rope(k_BHNE, rope_cos, rope_sin)

        if use_ring:
            # Sequence is fractured across SP, so attention must gather K/V around the ring.
            # The packed sequence is one attention document (the test keeps it padless), so no mask.
            spatial_BHNE, _prompt, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                self.dummy_joint_input,
                self.dummy_joint_input,
                self.dummy_joint_input,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k_BHNE.shape, 2, self.sp_mesh_axis, dtype=k_BHNE.dtype
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v_BHNE.shape, 2, self.sp_mesh_axis, dtype=v_BHNE.dtype
                ),
                joint_strategy="rear",
                logical_n=N,
                program_config=self._sdpa_program_config(q_BHNE.shape[2], ring=True),
                compute_kernel_config=self.sdpa_compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(self.sp_mesh_axis),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.sp_mesh_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(self.sdpa_worker_grid[0], 0),
                use_column_major_ccl=True,
            )
        else:
            spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                is_causal=False,
                program_config=self._sdpa_program_config(q_BHNE.shape[2], ring=False),
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )

        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        # Each device holds canonical heads [d * n_local, (d+1) * n_local), so gathering on TP
        # rebuilds the full inner_dim in canonical order for to_out.
        if tp_factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.tp_mesh_axis
            )

        return self.to_out(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)
