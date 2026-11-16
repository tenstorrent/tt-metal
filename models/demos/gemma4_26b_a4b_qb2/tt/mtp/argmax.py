# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fast on-device argmax of row 0 of vocab-sharded logits, for the drafter's draft tokens (WS6).

The common sampler's greedy path costs ~0.52 ms (a top-k over every 65536-wide shard row); a draft step needs one
row. Here (per chip, vocab shard ``c``):

1. transpose the logits to ``[V, 32]`` (vocab on rows) and view them as ``[64, 1024, 32]``: the max over dim -2 then
   runs on 64 cores, a second max over the 64 partials gives the chip's row maxima;
2. all-gather the four chips' maxima and take the global max ``G`` per row;
3. ``mask = logits >= G`` (only the winner's entries are 1) and one batched matmul of the mask with a constant
   index table sums ``[id // 256 % 256, id % 256, count, chip]`` per row (every factor < 256, so bf16 operands
   and fp32 accumulation are exact);
4. all-reduce over the chips and assemble ``id = 65536 chip + 256 hi + lo``.

An exact tie between two vocab entries sums their ids (a wrong but in-range draft that the verify then rejects); the
verify itself keeps the plain decode's sampler, so ties never change the committed tokens.
"""

import torch

import ttnn


class DraftArgmax:
    SPLIT = 64

    def __init__(self, model, vocab_per_chip):
        self.model, self.mesh = model, model.mesh
        self.V = vocab_per_chip
        self.block = vocab_per_chip // self.SPLIT
        chips = self.mesh.get_num_devices()
        # Index table [chips * SPLIT, 32, block]: rows 0..3 = hi, lo, 1, chip of every vocab entry.
        table = torch.zeros(chips, self.SPLIT, 32, self.block)
        local = torch.arange(vocab_per_chip).reshape(self.SPLIT, self.block)
        for chip in range(chips):
            table[chip, :, 0] = (local // 256) % 256
            table[chip, :, 1] = local % 256
            table[chip, :, 2] = 1
            table[chip, :, 3] = chip
        self.index = ttnn.from_torch(
            table.reshape(chips * self.SPLIT, 32, self.block)[None].bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=1),
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.ccl = model.layers[0]

    def __call__(self, logits):
        """``logits`` [1, 1, R, V/4] bf16 TILE (row 0 used). Returns the row-0 id as ``[1, 1]`` uint32 ROW_MAJOR."""
        cols = ttnn.transpose(logits, -2, -1)  # [1, 1, V/4, R]: vocab on rows, logit rows on columns
        rows = int(cols.shape[-1])
        blocks = ttnn.reshape(cols, (1, self.SPLIT, self.block, rows))
        local = ttnn.max(ttnn.max(blocks, dim=-2, keepdim=True), dim=1, keepdim=True)  # [1, 1, 1, 32]
        gathered = self.ccl.gather(local)  # [1, 1, 1, 4R]: chip c's maxima at columns Rc..Rc+R-1
        best = ttnn.max(ttnn.reshape(gathered, (1, 1, 4, rows)), dim=-2, keepdim=True)  # [1, 1, 1, R]
        mask = ttnn.reshape(ttnn.ge(cols, best), (1, self.SPLIT, self.block, rows))
        sums = ttnn.matmul(self.index, mask, dtype=ttnn.float32, compute_kernel_config=self.compute)  # [1, S, 32, 32]
        sums = ttnn.sum(sums, dim=1, keepdim=True)  # [1, 1, 32 (hi, lo, count, chip, 0..), 32 (logit rows)]
        row = ttnn.slice(ttnn.transpose(sums, -2, -1), (0, 0, 0, 0), (1, 1, 1, 32))  # logit row 0: [hi, lo, n, c]
        # Only the winning chip has nonzero sums: add the four chips' rows (small integers: exact).
        row = ttnn.sum(ttnn.reshape(self.ccl.gather(row), (1, 1, 4, 32)), dim=-2, keepdim=True)
        hi, lo, chip = (ttnn.slice(row, (0, 0, 0, i), (1, 1, 1, i + 1)) for i in (0, 1, 3))
        ids = ttnn.add(ttnn.add(ttnn.mul(hi, 256.0), lo), ttnn.mul(chip, float(self.V)))  # fp32 SFPU: exact
        # A tie sums ids; keep the (rejected) draft inside the vocabulary.
        ids = ttnn.typecast(ttnn.clamp(ids, 0.0, float(4 * self.V - 1)), ttnn.uint32)
        return ttnn.reshape(ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT), (1, 1))
