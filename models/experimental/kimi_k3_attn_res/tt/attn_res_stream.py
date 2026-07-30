# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-side `block_residual` lifecycle for Kimi K3 attention residuals.

Deliberately interface-compatible with `torch_functional.attn_res.AttnResStream`
— same `prefix_sum` / `num_sealed` / `read` / `seal` / `accumulate` / `block_size`
— so the depth harness can drive both backends through one shared walk and the
two orders provably cannot diverge.
"""

import ttnn

BLOCK_SIZE = 12


class TtAttnResStream(object):
    """One live stream and write-once snapshots, on device.

    Writes are plain `+=` with weight one; AttnRes rewrites only the read.
    `prefix_sum` is None between a seal and the next `accumulate` — the layer
    pipeline places no read site in that window, so `read` asserts rather than
    guessing.

    **The stream owns its tensors.** Construction transfers ownership of
    `hidden_states`, and `accumulate` takes ownership of `module_out`. That makes
    the first `seal` a zero-copy ownership move into `block_residual` instead of a
    clone, and lets every later `seal` free what it superseded.

    Args:
        op: a `TtAttnRes`.
        hidden_states: `[1, 1, N, d]` token embeddings, the first live stream.
        block_size: layers per block; seals fire at `layer_idx % block_size == 0`.
    """

    def __init__(self, op, hidden_states, block_size=BLOCK_SIZE):
        self.op = op
        self.prefix_sum = hidden_states
        self.block_residual = None
        self.block_size = block_size

    @property
    def num_sealed(self):
        return 0 if self.block_residual is None else self.block_residual.shape[1]

    def read(self, q):
        assert self.prefix_sum is not None, "no live stream between seal and accumulate"
        return self.op.forward(self.prefix_sum, self.block_residual, q)

    def seal(self):
        assert self.prefix_sum is not None, "nothing to seal"
        if self.block_residual is None:
            self.block_residual = self.prefix_sum
        else:
            grown = ttnn.concat([self.block_residual, self.prefix_sum], dim=1)
            ttnn.deallocate(self.block_residual)
            ttnn.deallocate(self.prefix_sum)
            self.block_residual = grown
        self.prefix_sum = None

    def accumulate(self, module_out):
        if self.prefix_sum is None:
            self.prefix_sum = module_out
        else:
            total = ttnn.add(self.prefix_sum, module_out)
            ttnn.deallocate(self.prefix_sum)
            ttnn.deallocate(module_out)
            self.prefix_sum = total

    def deallocate(self):
        for tensor in (self.prefix_sum, self.block_residual):
            if tensor is not None:
                ttnn.deallocate(tensor)
        self.prefix_sum = None
        self.block_residual = None
