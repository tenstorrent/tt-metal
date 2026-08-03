# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused single-device North-Mini decoder layer.

This stage deliberately preserves the public contract and the already-fused
attention/cache topology of :class:`FunctionalDecoder`.  It specializes the
remaining MLP graph:

* dense and expert gate/up projections share one packed weight and one matmul;
* SiLU is folded into the consuming multiply;
* sparse gate/up evaluates only top-k routes, while the current A-sparse
  contract uses exact all-expert down projection over already-zero inactive
  rows;
* the tiled all-expert path fuses routing-score multiplication and reduction.

Weight packing happens once in ``from_state_dict``.  Runtime forwards contain
no host conversion and never dispatch the functional all-expert MLP.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import FunctionalDecoder


def _largest_rectangular_divisor(tile_count: int, grid) -> tuple[int, int, int]:
    """Return ``(cores, x, y)`` for the largest legal divisor of N tiles."""
    limit = min(tile_count, grid.x * grid.y)
    for cores in range(limit, 0, -1):
        if tile_count % cores:
            continue
        for x in range(min(grid.x, cores), 0, -1):
            if cores % x == 0 and cores // x <= grid.y:
                return cores, x, cores // x
    raise RuntimeError(f"no legal core rectangle for {tile_count} output tiles on grid {grid}")


def _sparse_matmul_config(*, m: int, k: int, n: int, grid):
    n_tiles = math.ceil(n / ttnn.TILE_SIZE)
    cores, grid_x, grid_y = _largest_rectangular_divisor(n_tiles, grid)
    k_tiles = math.ceil(k / ttnn.TILE_SIZE)
    # A moderate K block avoids the tiny-block functional default without
    # overcommitting L1.  Both North-Mini K dimensions divide eight tiles.
    in0_block_w = next(value for value in (8, 4, 2, 1) if k_tiles % value == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=n_tiles // cores,
        per_core_M=max(ttnn.TILE_SIZE, m) // ttnn.TILE_SIZE,
        per_core_N=n_tiles // cores,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class FusedDecoder(FunctionalDecoder):
    """North-Mini decoder whose remaining MLP graph is fused and routed."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        storage_grid = decoder.mesh_device.compute_with_storage_grid_size()
        key_grid = ttnn.num_cores_to_corerangeset(
            decoder.batch,
            storage_grid,
            row_wise=True,
        )
        value_coords = [
            ttnn.CoreCoord(x, y)
            for y in range(storage_grid.y)
            for x in range(storage_grid.x)
            if not key_grid.contains(ttnn.CoreCoord(x, y))
        ][: decoder.batch]
        if len(value_coords) == decoder.batch:
            value_grid = ttnn.CoreRangeSet({ttnn.CoreRange(coord, coord) for coord in value_coords})
            decoder.decode_value_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, decoder.head_dim),
                core_grid=value_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            decoder.decode_value_memory_config = None
        if decoder.mlp_type == "dense":
            decoder.weights["gate_up"] = ttnn.concat(
                [decoder.weights["gate_proj"], decoder.weights["up_proj"]],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            decoder.weights["gate_proj"].deallocate(True)
            decoder.weights["up_proj"].deallocate(True)
            del decoder.weights["gate_proj"]
            del decoder.weights["up_proj"]
        else:
            decoder.weights["expert_gate_up"] = ttnn.concat(
                [decoder.weights["expert_gate"], decoder.weights["expert_up"]],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            decoder.weights["expert_gate"].deallocate(True)
            decoder.weights["expert_up"].deallocate(True)
            del decoder.weights["expert_gate"]
            del decoder.weights["expert_up"]

            decoder.weights["expert_gate_up"] = ttnn.reshape(
                decoder.weights["expert_gate_up"],
                (1, decoder.num_experts, decoder.hidden_size, 2 * decoder.intermediate_size),
            )
            decoder.weights["expert_down"] = ttnn.reshape(
                decoder.weights["expert_down"],
                (1, decoder.num_experts, decoder.intermediate_size, decoder.hidden_size),
            )
            grid = decoder.mesh_device.compute_with_storage_grid_size()
            decoder.sparse_gate_up_program_config = _sparse_matmul_config(
                m=1,
                k=decoder.hidden_size,
                n=2 * decoder.intermediate_size,
                grid=grid,
            )
            decoder.sparse_down_program_config = _sparse_matmul_config(
                m=1,
                k=decoder.intermediate_size,
                n=decoder.hidden_size,
                grid=grid,
            )
            decoder.all_expert_sparsity = ttnn.ones(
                (1, 1, 1, decoder.num_experts),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=decoder.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            identity = torch.arange(decoder.num_experts, dtype=torch.int32).reshape(1, 1, 1, decoder.num_experts)
            decoder.fused_reduce_indices = ttnn.from_torch(
                identity.repeat(ttnn.TILE_SIZE, 1, 1, 1),
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=decoder.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            decoder.fused_reduce_mapping = ttnn.from_torch(
                torch.zeros((1, decoder.num_experts), dtype=torch.int32),
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=decoder.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return decoder

    @staticmethod
    def _fused_swiglu(gate, up):
        return ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _split_gate_up(gate_up, intermediate_size):
        shape = tuple(gate_up.shape)
        starts = [0] * len(shape)
        gate_end = list(shape)
        gate_end[-1] = intermediate_size
        up_start = list(starts)
        up_start[-1] = intermediate_size
        return (
            ttnn.slice(gate_up, tuple(starts), tuple(gate_end)),
            ttnn.slice(gate_up, tuple(up_start), shape),
        )

    def _dense_mlp(self, normalized):
        gate_up = ttnn.linear(
            normalized,
            self.weights["gate_up"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate, up = self._split_gate_up(gate_up, self.intermediate_size)
        activated = self._fused_swiglu(gate, up)
        return ttnn.linear(
            activated,
            self.weights["down_proj"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _attention_decode(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        position_cos,
        position_sin,
    ):
        query, key, value = self._qkv_decode(normalized, position_cos, position_sin)
        if self.decode_value_memory_config is None:
            raise RuntimeError("paged fused cache update requires a disjoint value core grid")
        value = ttnn.to_memory_config(value, self.decode_value_memory_config)
        ttnn.experimental.paged_fused_update_cache(
            key_cache,
            key,
            value_cache,
            value,
            update_idxs_tensor=current_positions,
            page_table=page_table,
        )
        attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_positions,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self.decode_concat_memory_config)
        attended = ttnn.experimental.nlp_concat_heads_decode(
            attended,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_sub_core_grids,
        )
        projected = ttnn.linear(
            attended,
            self.weights["o"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        projected = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, self.hidden_size))
        return ttnn.permute(projected, (0, 2, 1, 3))

    def _sparse_moe_chunk(self, normalized, token_count):
        flat = ttnn.reshape(normalized, (token_count, self.hidden_size))
        logits = ttnn.linear(
            flat,
            self.weights["router"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        top_values, top_indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.sigmoid(top_values)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        if token_count >= ttnn.TILE_SIZE:
            return self._packed_all_expert_moe(flat, routing, token_count)

        sparsity = ttnn.to_layout(
            ttnn.reshape(routing, (1, token_count, 1, self.num_experts)),
            ttnn.ROW_MAJOR_LAYOUT,
        )

        # Treat tokens as independent sparse batches with M=1.  This preserves
        # per-token routing rather than broadening a tile's routes to their
        # union, and works unchanged for decode batch lanes and prefill tokens.
        expert_input = ttnn.reshape(flat, (1, token_count, 1, self.hidden_size))
        gate_up = ttnn.sparse_matmul(
            expert_input,
            self.weights["expert_gate_up"],
            sparsity=sparsity,
            nnz=token_count * self.top_k,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=self.sparse_gate_up_program_config,
            dtype=ttnn.bfloat16,
        )
        gate_up = ttnn.reshape(
            gate_up,
            (token_count, self.num_experts, 2 * self.intermediate_size),
        )
        gate, up = self._split_gate_up(gate_up, self.intermediate_size)
        activated = self._fused_swiglu(gate, up)
        activated = ttnn.reshape(
            ttnn.transpose(activated, 1, 0),
            (1, self.num_experts, token_count, self.intermediate_size),
        )
        # Sparse-matmul's A-sparse mode has one mask per expert batch, not per
        # M row.  Gate/up already zeroed inactive token/expert pairs, so an
        # all-expert down mask remains numerically exact while batching all
        # tokens in M and avoiding per-token tile waste.
        expert_output = ttnn.sparse_matmul(
            activated,
            self.weights["expert_down"],
            sparsity=self.all_expert_sparsity,
            nnz=self.num_experts,
            is_input_a_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=self.sparse_down_program_config,
            dtype=ttnn.bfloat16,
        )
        expert_output = ttnn.reshape(
            ttnn.permute(expert_output, (0, 2, 1, 3)),
            (token_count, self.num_experts, self.hidden_size),
        )
        routing = ttnn.reshape(routing, (token_count, self.num_experts, 1))
        expert_output = ttnn.multiply(expert_output, routing)
        return ttnn.sum(expert_output, dim=1)

    def _packed_all_expert_moe(self, flat, routing, token_count):
        """Tile-efficient path for 32+ tokens with packed gate/up dispatch."""
        expert_input = ttnn.repeat(
            ttnn.reshape(flat, (1, token_count, self.hidden_size)),
            ttnn.Shape((self.num_experts, 1, 1)),
        )
        gate_up = ttnn.matmul(
            expert_input,
            ttnn.reshape(
                self.weights["expert_gate_up"],
                (self.num_experts, self.hidden_size, 2 * self.intermediate_size),
            ),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate, up = self._split_gate_up(gate_up, self.intermediate_size)
        activated = self._fused_swiglu(gate, up)
        expert_output = ttnn.matmul(
            activated,
            ttnn.reshape(
                self.weights["expert_down"],
                (self.num_experts, self.intermediate_size, self.hidden_size),
            ),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if token_count != ttnn.TILE_SIZE:
            routing = ttnn.reshape(
                ttnn.permute(routing, (1, 0)),
                (self.num_experts, token_count, 1),
            )
            return ttnn.sum(ttnn.multiply(expert_output, routing), dim=0)

        expert_output = ttnn.reshape(
            expert_output,
            (self.num_experts, 1, token_count, self.hidden_size),
        )
        expert_output = ttnn.to_memory_config(expert_output, ttnn.L1_MEMORY_CONFIG)
        scores = ttnn.to_layout(
            ttnn.reshape(routing, (token_count, 1, 1, self.num_experts)),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        indices = ttnn.slice(
            self.fused_reduce_indices,
            (0, 0, 0, 0),
            (token_count, 1, 1, self.num_experts),
        )
        outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            expert_output,
            indices,
            self.fused_reduce_mapping,
            reduce_dim=0,
            split_size=self.hidden_size,
            cluster_axis=0,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            scores_tensor=scores,
        )
        return ttnn.reshape(outputs[0], (token_count, self.hidden_size))
