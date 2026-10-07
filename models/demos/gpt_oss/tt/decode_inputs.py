# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode step inputs of the fused decode path (tt/fused_decode.py) in one op (kernels/decode_inputs.cpp).

From the persistent decode trace inputs (token ids, RoPE row indices) it writes the token's embedding row (the
[1, 1, 1, hidden] BF16 row-major input of the first layer's entry boundary, decode_boundary.py) and the cos / sin rows of
the fused Q/K rotary op (height-sharded [1, 2 * batch, 1, head_dim] BF16 tiles on the RoPE setup's batch grid, row 0 of
each shard: the op broadcasts row 0). The unfused decode path builds the same tensors with ttnn.embedding (+ a slice of
the 32-slot token buffer) and RotarySetup.get_rot_mats (2 embeddings, 2 transposes, 2 slices, 2 interleaved-to-sharded
copies): 10 ops per token.
"""

import ttnn

from .decode_boundary import KERNEL_DIR

CORE = ttnn.CoreCoord(0, 0)


def _page_bytes(t):
    return t.shape[-1] * (2 if t.dtype == ttnn.bfloat16 else 4)


class DecodeInputs:
    def __init__(self, mesh_device, embedding_weight, rope_setup, hidden):
        self.mesh_device = mesh_device
        self.embedding_weight = embedding_weight
        self.cos_table, self.sin_table = rope_setup.cos_matrix, rope_setup.sin_matrix
        self.hidden = hidden
        self.head_dim = rope_setup.head_dim
        assert rope_setup.use_qk_fused and rope_setup.prefetcher is None
        self.users = rope_setup.doubled_batch_size
        self.rope_cores = ttnn.corerange_to_cores(rope_setup.batch_grid, row_wise=True)[: self.users]
        self.rope_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=rope_setup.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])

    def __call__(self, tokens, rot_idxs):
        """tokens: the [1, 1, 1, 32] UINT32 decode token buffer (slot 0 is the user); rot_idxs: the [1, 32] UINT32
        RoPE row indices (Q users then K users). Returns (embedding row, [cos, sin])."""
        mesh = self.mesh_device
        emb = ttnn.empty(
            [1, 1, 1, self.hidden],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos, sin = (
            ttnn.empty(
                [1, self.users, 1, self.head_dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=self.rope_memory_config,
            )
            for _ in range(2)
        )
        row_bytes = self.hidden * 2
        tok_page, rot_page = _page_bytes(tokens), _page_bytes(rot_idxs)
        dim_tiles = self.head_dim // ttnn.TILE_SIZE
        scratch = 0
        for b in (tok_page, rot_page, row_bytes):
            scratch += -(-b // 64) * 64
        scratch += 2 * self.users * self.head_dim * 2
        ct = [0, row_bytes, tok_page, rot_page, self.users, dim_tiles]
        for t in (tokens, self.embedding_weight, emb, rot_idxs, self.cos_table, self.sin_table):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        args = [
            tokens.buffer_address(),
            self.embedding_weight.buffer_address(),
            emb.buffer_address(),
            rot_idxs.buffer_address(),
            self.cos_table.buffer_address(),
            self.sin_table.buffer_address(),
            cos.buffer_address(),
            sin.buffer_address(),
        ]
        for c in self.rope_cores:
            p = mesh.worker_core_from_logical_core(c)
            args += [p.x, p.y]
        rt[CORE.x][CORE.y] = args
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "decode_inputs.cpp"),
            core_ranges=self.cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        cb = ttnn.CBDescriptor(
            total_size=scratch,
            core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=scratch)],
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
        ttnn.generic_op(
            [tokens, self.embedding_weight, rot_idxs, self.cos_table, self.sin_table, emb, cos, sin], program
        )
        return emb, [cos, sin]
