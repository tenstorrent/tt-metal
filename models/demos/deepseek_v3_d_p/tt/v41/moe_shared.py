# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 decode MoE: the shared expert folded into the routed path's tail (``V41_SHARED_IN_FFN``).

``TtSharedExpert.forward`` runs the shared expert on the decode tile as three ``ttnn.matmul`` at M = 32 (gate,
up,
down on the chip's TP slice of the intermediate: ``[5120, 576]``, ``[5120, 576]``, ``[576, 5120]`` bfp8, bf16
activations), a ``clamped_silu_glu`` and a TP reduce-scatter of the ``[32, 5120]`` partial: five programs,
~100 us
a layer. :class:`DecodeSharedExpert` drops the reduce-scatter: the partial is handed
to :class:`moe_tail.DecodeReduce`, which adds it to the routed sum before the routed output's TP
reduce-scatter (one
``[32, 5120]`` add in place of a 28 us collective), and ``forward`` returns ``None`` so ``TtMoe.forward``
skips the
final add. Two modes:

- ``V41_SHARED_IN_FFN=rs``: the three matmuls and the activation as today (bf16 activations; the only
difference
  from today's output is the bf16 rounding of the sum before the reduce-scatter instead of after it).
- ``V41_SHARED_IN_FFN=ffn``: the projections and the activation as one ``unified_routed_expert_moe`` program
(the
  routed experts' op, on its 11 x 8 grid) with the shared expert's three device tensors as a one-expert weight
  list
  (no second copy at bfp8: the op takes any ``[K, N]`` bfp8 TILE DRAM tensors; N = 576 is 18 tiles over 11
  columns), ``counts`` the whole tile, the activation the routed experts' clamped SiLU-GLU (the same limit),
  on a
  bfp8 copy of the all-gathered tile (the op works in place; ``x`` goes on to the dispatch). The op packs its
  input to bfp8 (PCC 0.9997 against the bf16 matmuls on random tiles) and at bfp8 weights reads no faster than
  the
  three matmuls (54 us against 56); ``V41_SHARED_IN_FFN_DTYPE=bfp4`` runs it on a bfp4
  copy of the weights (4.4 MB a chip a layer).

The shared expert is a "13th expert every token selects": a 13th program rather than a 13th
region of the dispatch buffer, since the shared expert's intermediate is a quarter of a routed expert's and
the
dispatch buffer's regions are one width.
"""

import os

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import (
    ACTIVATION_CLAMPED_SILU_GLU,
    clamped_silu_glu,
    get_program_configs,
)

SHARED_IN_FFN = os.environ.get("V41_SHARED_IN_FFN", "0")
assert SHARED_IN_FFN in ("0", "rs", "ffn"), SHARED_IN_FFN
_TILE = 32


class DecodeSharedExpert:
    """``TtSharedExpert`` for the decode tile: one routed-expert FFN program on the chip's TP slice of the
    shared
    expert, the partial handed to ``reduce`` (a :class:`moe_tail.DecodeReduce`) instead of
    reduce-scattered."""

    def __init__(self, shared, routed, reduce, mode: str = "rs", weights_dtype=None):
        assert mode in ("rs", "ffn"), mode
        assert shared.activation == ACTIVATION_CLAMPED_SILU_GLU, shared.activation
        assert float(shared.clamped_silu_glu_limit) == 10.0, shared.clamped_silu_glu_limit  # the op bakes 10
        assert shared.subdevice_cores is None, "the decode shared expert runs on the whole grid"
        self._shared, self._routed, self._reduce, self.mode = shared, routed, reduce, mode
        device = shared.mesh_device
        if mode == "rs":
            self.hidden, self.inter = int(shared.gate_proj.shape[-2]), int(shared.gate_proj.shape[-1])
            return
        two_d = lambda t: ttnn.reshape(t, tuple(t.shape)[-2:]) if len(t.shape) != 2 else t
        self.gate, self.up, self.down = (two_d(shared.gate_proj), two_d(shared.up_proj), two_d(shared.down_proj))
        if weights_dtype is not None and self.gate.dtype != weights_dtype:
            # V41_SHARED_IN_FFN_DTYPE=bfp4: a second copy of the three tensors (4.4 MB a chip a layer)
            cast = lambda t: ttnn.typecast(t, weights_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.gate, self.up, self.down = cast(self.gate), cast(self.up), cast(self.down)
        self.hidden, self.inter = int(self.gate.shape[-2]), int(self.gate.shape[-1])
        assert int(self.down.shape[-2]) == self.inter and int(self.down.shape[-1]) == self.hidden
        rep = lambda t: ttnn.from_torch(
            t,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        # one "expert" (global id 0): its region at row 0 of the input, every row of the tile in it
        self.table = rep(torch.zeros(1, dtype=torch.int32))
        self.region = rep(torch.zeros(1, 1, dtype=torch.int32))
        self.counts = rep(torch.full((1, 1), _TILE, dtype=torch.int32))
        # build the program-cache entry on the empty L1 (a routed expert FFN program compiled
        # after the decode tail's ``ttnn.reduce_scatter`` is rejected against a block it leaves)
        zeros = ttnn.from_torch(
            torch.zeros(1, _TILE, self.hidden),
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        ttnn.deallocate(self._ffn(zeros))

    def __getattr__(self, name):
        return getattr(self._shared, name)

    def _ffn(self, x):
        routed = self._routed
        return ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            x,
            self.region,
            self.counts,
            self.table,
            [self.gate],
            [self.up],
            [self.down],
            max_dispatched_tokens_per_expert=_TILE,
            compute_kernel_config=routed.compute_kernel_config,
            activation=routed.activation,
        )

    def _matmuls(self, x):
        """``TtSharedExpert.forward``'s steps 1 to 3 (the gate and up matmuls, the clamped SiLU-GLU, the down
        matmul), without its reduce-scatter."""
        s = self._shared
        grid = s.mesh_device.compute_with_storage_grid_size()
        tiles = lambda t, d: t.padded_shape[d] // ttnn.TILE_SIZE
        gate_cfg, up_cfg, down_cfg = get_program_configs(
            grid,
            tiles(x, -2),
            tiles(s.gate_proj, -2),
            tiles(s.gate_proj, -1),
            tiles(s.down_proj, -2),
            tiles(s.down_proj, -1),
            fuse_silu=False,
        )
        mm = lambda a, w, cfg: ttnn.matmul(a, w, program_config=cfg, compute_kernel_config=s.compute_kernel_config)
        act = clamped_silu_glu(mm(x, s.gate_proj, gate_cfg), mm(x, s.up_proj, up_cfg), s.clamped_silu_glu_limit)
        out = mm(act, s.down_proj, down_cfg)
        ttnn.deallocate(act)
        return out

    def __call__(self, x):
        """``x``: the all-gathered tile, bf16 TILE ``[1, T, hidden]`` (``T <= 32``) -> ``None``; the chip's
        partial
        of the shared expert's output (TILE, the same shape; bf16 in ``rs`` mode, bfp8 in ``ffn`` mode) waits
        in
        ``reduce.pending_shared``."""
        assert self._reduce.pending_shared is None, "the previous partial was not consumed by the reduce"
        T, H = int(x.shape[-2]), int(x.shape[-1])
        assert T <= _TILE and H == self.hidden, (tuple(x.shape), self.hidden)
        if self.mode == "rs":
            self._reduce.pending_shared = self._matmuls(x)
        else:
            self._reduce.pending_shared = self._ffn(
                ttnn.typecast(x, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
        return None

    forward = __call__


def install_shared_in_ffn(v41_moe, mode: str = "rs", weights_dtype=None) -> None:
    """Swap ``v41_moe.moe``'s shared expert for :class:`DecodeSharedExpert`; needs the decode tail installed."""
    from models.demos.deepseek_v3_d_p.tt.v41.moe_tail import DecodeReduce

    moe = v41_moe.moe
    assert isinstance(moe.reduce_module, DecodeReduce), "V41_SHARED_IN_FFN needs V41_DECODE_TAIL=1"
    if not isinstance(moe.shared_expert, DecodeSharedExpert):
        moe.shared_expert = DecodeSharedExpert(
            moe.shared_expert, moe.routed_expert, moe.reduce_module, mode, weights_dtype
        )
