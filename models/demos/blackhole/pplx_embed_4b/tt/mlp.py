# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""pplx-embed-v1-4B MLP with fused SwiGLU on the batched prefill path.

The stock MLP runs FF1 (gate) and FF3 (up) as two separate matmuls and then a
third op to combine them::

    w1_out = matmul(x, w1)                     # gate
    w3_out = matmul(x, w3)                     # up
    w2_in  = mul(w1_out, w3_out, act=[silu])   # BinaryNg

``ttnn.experimental.minimal_matmul`` can do all three in one op via
``fuse_swiglu=True``, which collapses gate/up tile-pairs to ``silu(gate)*up``
as it packs. At batch>=8 that matters: BinaryNg is 78.5 ms of the 502 ms
bs=32 device total (15.6%, the largest non-matmul cost), the SwiGLU multiply is
one of its three ops per layer, and the two intermediates are DRAM-resident, so
fusing also removes their round-trips and one matmul launch.

Weight layout
-------------
The op's contract (minimal_matmul_device_operation.cpp): "The weight packs
gate/up tile-pairs along N, so its width must be an even number of tiles; the
matmul output collapses each pair to one tile (silu(gate)*up)."

That is *tile-pair interleaved*, not a plain ``[gate | up]`` concat — 32-column
tiles alternate g0,u0,g1,u1,... Verified at real 4B shapes (K=2560, N=9728,
M=512): interleaved gives PCC 0.9997 against
``silu(x @ gate) * (x @ up)`` while plain concat gives -0.0002, i.e. the wrong
layout is silent numerical garbage rather than an error.

Scope
-----
Only the prefill path that already routes to ``minimal_matmul`` is fused. Decode,
galaxy/multi-device, the legacy MatmulMultiCoreReuseMultiCast path used at bs=1,
and anything whose shape fails the op's constraints fall through to the stock
implementation. Opt out entirely with ``QWEN_FUSE_SWIGLU=0``.
"""

import os

import torch

import ttnn
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import Mode, OpGroup

TILE = 32


def _pack_gate_up_tile_pairs(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Interleave gate/up along N in ``TILE``-column groups.

    gate, up: ``[K, N]`` -> ``[K, 2N]`` ordered g0,u0,g1,u1,... per 32-col tile.
    """
    k, n = gate.shape
    assert n % TILE == 0, f"N={n} must be tile-aligned"
    g = gate.reshape(k, n // TILE, TILE)
    u = up.reshape(k, n // TILE, TILE)
    return torch.stack([g, u], dim=2).reshape(k, 2 * n)


class PplxFusedSwigluMLP(MLP):
    """MLP whose batched-prefill FF1/FF3/mul collapse into one fused matmul."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default off: the demo's apply_workload_env opts in for the batch sizes
        # where it measured faster (see its comment). Anything constructing this
        # class directly gets stock behaviour unless it asks for the fusion.
        self.fuse_swiglu_enabled = os.getenv("QWEN_FUSE_SWIGLU", "0") == "1"
        self.w_gate_up = None
        if self.fuse_swiglu_enabled:
            self._build_packed_gate_up(kwargs.get("state_dict"), kwargs.get("weight_cache_path"))

    # ------------------------------------------------------------------ #
    def _build_packed_gate_up(self, state_dict, weight_cache_path):
        """Pack w1/w3 into one tile-pair-interleaved device tensor.

        Packing happens once at construction. If anything about the checkpoint
        layout does not match expectations we leave ``w_gate_up`` as None and the
        forward path silently keeps using the stock two-matmul implementation.
        """
        try:
            # Must match the base class exactly: it derives the prefix from
            # self.__class__.__name__, which is why this class is renamed to "MLP"
            # at the bottom of this file. Ask for "MLP" explicitly here so the
            # lookup cannot drift if that alias is ever removed.
            prefix = self.args.get_state_dict_prefix("MLP", self.layer_num)
            w1 = state_dict[f"{prefix}.w1.weight"]
            w3 = state_dict[f"{prefix}.w3.weight"]
        except (KeyError, TypeError, AttributeError):
            return

        # Checkpoint stores [out_features, in_features]; the matmul wants [K, N].
        gate, up = w1.transpose(-2, -1), w3.transpose(-2, -1)
        if gate.shape != up.shape or gate.shape[-1] % TILE != 0:
            return

        packed = _pack_gate_up_tile_pairs(gate, up).unsqueeze(0).unsqueeze(0)
        # The base __init__ has already resolved this for w1/w3; reuse it so the
        # packed tensor cannot drift from the unpacked ones.
        self.w_gate_up = ttnn.as_tensor(
            packed,
            dtype=self.ff1_3_dtype or ttnn.bfloat8_b,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=(
                None if weight_cache_path is None else str(weight_cache_path / f"{prefix}.w_gate_up_swiglu")
            ),
        )

    # ------------------------------------------------------------------ #
    def _can_fuse(self, x, mode, seq_len) -> bool:
        if not self.fuse_swiglu_enabled or self.w_gate_up is None:
            return False
        if mode != Mode.PREFILL:
            return False
        if self.args.is_galaxy or self.args.num_devices > 1:
            return False
        # Only where the stock path would already have used minimal_matmul; the
        # legacy 2D path (bs=1 short-seq on this model) is faster left alone.
        if not self.args.use_minimal_prefill_matmul(seq_len):
            return False
        # fuse_swiglu needs the packed width to be an even number of tiles.
        return (2 * self.args.hidden_dim) % (2 * TILE) == 0

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        seq_len = x.shape[-2]
        if not self._can_fuse(x, mode, seq_len):
            return super().forward(x, mode)

        layer = max(self.layer_num, 0)
        ckc = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )
        pc = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        if not isinstance(pc, ttnn.MinimalMatmulConfig):
            return super().forward(x, mode)

        prefill_mem_seq = int(seq_len)
        w2_in = ttnn.experimental.minimal_matmul(
            x,
            self.w_gate_up,
            config=pc,
            compute_kernel_config=ckc,
            memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher, prefill_seq_len=prefill_mem_seq),
            fuse_swiglu=True,
        )
        ttnn.deallocate(x)
        return self._down_project(w2_in, mode, seq_len)

    # ------------------------------------------------------------------ #
    def _down_project(self, w2_in, mode, seq_len):
        """FF2 plus the tail the stock forward does after the SwiGLU combine."""
        layer = max(self.layer_num, 0)
        li_ff2_ckc = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer, op=OpGroup.LI_FF2, configuration=self.args
        )
        prefill_mem_seq = int(seq_len)
        pc_2 = self.args.get_mlp_ff2_prg_config(mode, seq_len, self.prefetcher)
        out_mem = self.args.get_mlp_ff2_mem_config(mode, self.prefetcher, prefill_seq_len=prefill_mem_seq)

        if isinstance(pc_2, ttnn.MinimalMatmulConfig):
            w2_out = ttnn.experimental.minimal_matmul(
                w2_in, self.w2, compute_kernel_config=li_ff2_ckc, config=pc_2, memory_config=out_mem
            )
        else:
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_ckc,
                program_config=pc_2,
                memory_config=out_mem,
                core_grid=None,
            )
        ttnn.deallocate(w2_in)

        original_shape = w2_out.shape
        w2_out = ttnn.reshape(
            w2_out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        return w2_out


# The base MLP derives its state_dict prefix from ``self.__class__.__name__``
# (args.get_state_dict_prefix(self.__class__.__name__, layer_num)), so a subclass
# with a different name would look up weights under the wrong prefix and fail to
# load. tt/attention.py does the same for PplxBidirectionalAttention.
PplxFusedSwigluMLP.__name__ = "MLP"
