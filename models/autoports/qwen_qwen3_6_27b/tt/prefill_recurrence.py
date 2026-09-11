# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Native chunk GDN integration with model-owned constant tiles."""

import torch

import ttnn


class NativeGatedDeltaRule:
    """Model-owned constants for the native chunk GDN implementation."""

    def __init__(self, mesh):
        grid = mesh.compute_with_storage_grid_size()
        self.compute_cores = grid.x * grid.y
        rows = torch.arange(32).unsqueeze(1) < 16
        columns = torch.arange(32).unsqueeze(0) < 16
        masks = torch.cat((rows & columns, ~rows & ~columns, ~rows & columns), dim=1).float()
        self.constants = tuple(
            ttnn.from_torch(
                value.reshape(1, 1, *value.shape),
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for value in (torch.eye(32), torch.ones(32, 32).tril(), torch.ones(32, 32), masks)
        )

    def __call__(
        self, query, key, value, beta, decay, *, initial_state, groups, sequence, value_dim, batch, value_heads
    ):
        def token_heads(tensor):
            return ttnn.permute(ttnn.reshape(tensor, (batch, value_heads, sequence, value_dim)), (0, 2, 1, 3))

        def gates(tensor):
            tensor = ttnn.reshape(tensor, (batch, value_heads, sequence))
            return ttnn.permute(ttnn.typecast(tensor, ttnn.float32), (0, 2, 1))

        output, state = self._forward_batches(
            token_heads(query),
            token_heads(key),
            token_heads(value),
            ttnn.log(gates(decay)),
            gates(beta),
            initial_state,
            1.0,
        )
        return ttnn.reshape(output, (groups, sequence, 1, value_dim)), state

    def close(self):
        self.constants = ()

    def flat_forward(self, query, key, value, log_decay, beta, initial_state, scale):
        # Rank-3 Q/K selects the phased reader's head mapping and in-kernel
        # L2 normalization. use_qk_l2norm=True is NOT supported by this API.
        return self._forward_batches(query, key, value, log_decay, beta, initial_state, scale)

    def _forward_batches(self, query, key, value, log_decay, beta, initial_state, scale):
        # Native phased scan needs at least one core per batch/value-head pair.
        # Keep the common B1 graph unchanged; larger batches remain on device
        # and preserve independent per-slot state through batch-axis slices.
        batch = query.shape[0]
        value_heads = beta.shape[-1]
        max_batch = self.compute_cores // value_heads
        if max_batch < 1:
            raise ValueError(f"Native GDN value heads {value_heads} exceed compute cores {self.compute_cores}")
        if batch <= max_batch:
            return self._forward(query, key, value, log_decay, beta, initial_state, scale)
        outputs, states = [], []
        for start in range(0, batch, max_batch):
            stop = min(start + max_batch, batch)
            inputs = [tensor[start:stop] for tensor in (query, key, value, log_decay, beta, initial_state)]
            output, state = self._forward(*inputs, scale)
            outputs.append(output)
            states.append(state)
        return ttnn.concat(outputs, dim=0), ttnn.concat(states, dim=0)

    def _forward(self, query, key, value, log_decay, beta, initial_state, scale):
        return ttnn.transformer.chunk_gated_delta_rule(
            query,
            key,
            value,
            log_decay,
            beta,
            scale=scale,
            initial_state=ttnn.typecast(initial_state, ttnn.float32),
            output_final_state=True,
            chunk_size=32,
            use_qk_l2norm=False,
            output_head_major=True,
            eye=self.constants[0],
            tril=self.constants[1],
            ones=self.constants[2],
            masks=self.constants[3],
        )
