# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN rotary embeddings for ModernBERT.

Two rotary thetas, selected by layer type:

    full_attention     rope_theta = 160000.0
    sliding_attention  rope_theta =  10000.0

Both live in `config.rope_parameters[layer_type]["rope_theta"]`; the
`global_rope_theta` / `local_rope_theta` names do not exist in transformers 5.x.

This is an encoder, so the cos/sin caches are constant per sequence length and are
built on host once rather than per layer.

`rotary_embedding_hf` wants caches shaped (1, 1, seq_len, head_dim) and implements
the HF rotate_half convention. `rotary_embedding_llama` is the interleaved Meta
convention and would be silently wrong here.
"""

import torch

import ttnn
from models.experimental.modernbert.reference.modernbert import ModernBertRotaryEmbedding
from models.experimental.modernbert.tt import model_config as _cfg

# The prefill program factory sizes its input CB at 2 tiles for an interleaved
# tensor and at the whole shard for a sharded one, so sharding is worth a lot to
# the kernel - but it costs two reshards per call, which is near-fixed while the
# gain scales with the data. It therefore only pays at the largest shape:
#
#     b1s256  +50%   b1s512  +8%   b4s256  +6%   b8s256  -4.2%
#
# Hence a row threshold rather than an on/off. Forcing it on below the threshold
# costs 14.17 -> 17.43 ms at b4s256. "keep" is not available - SDPA rejects
# sharded operands, so the tensor has to come back before attention.
#   "off"        interleaved
#   "roundtrip"  reshard in, rotary sharded, reshard back out for SDPA
#   "keep"       unavailable, see above
SHARD_ROTARY = "roundtrip"
# Flattened rows are batch * n_heads * seq_len: 24576 at b8s256, 12288 at b4s256.
_SHARD_ROTARY_MIN_ROWS = 24576


class TtnnModernBertRotary:
    """Holds one cos/sin cache pair per layer type for a fixed sequence length."""

    def __init__(self, config, device, seq_len, batch_size=1, dtype=ttnn.bfloat16):
        self.seq_len = seq_len
        head_dim = config.hidden_size // config.num_attention_heads
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # Resolved once, not per call. __call__ runs 44 times per pass and reading
        # a tensor property crosses the pybind boundary each time: computing this
        # inline cost 0.70 ms of an 8.78 ms b1s256 pass, and memoising on
        # tuple(tensor.shape) still cost 0.43 ms. The call path inspects no shapes.
        self._shard_mem = self._shard_config((batch_size, config.num_attention_heads, seq_len, head_dim))
        self._interleaved = _cfg.attention_interleaved()

        self.caches = {}
        for layer_type in set(config.layer_types):
            theta = config.rope_parameters[layer_type]["rope_theta"]
            # Reuse the reference generator so host-side cache construction has a
            # single definition and cannot drift from the validated torch path.
            cos, sin = ModernBertRotaryEmbedding(head_dim, theta)(position_ids, torch.float32)
            self.caches[layer_type] = (
                self._upload(cos, device, dtype),
                self._upload(sin, device, dtype),
            )

    @staticmethod
    def _upload(t, device, dtype):
        # (B, S, head_dim) -> (1, 1, S, head_dim); the op broadcasts over heads.
        return ttnn.from_torch(
            t[:1].unsqueeze(1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, tensor, layer_type):
        """Apply rotary embedding to a (B, n_heads, seq_len, head_dim) tensor.

        No `compute_kernel_config` on purpose: every fidelity lands inside
        run-to-run noise, so there is no time to buy, and the HiFi4 default scores
        best on accuracy.
        """
        cos, sin = self.caches[layer_type]
        if SHARD_ROTARY == "off":
            return ttnn.experimental.rotary_embedding_hf(
                tensor, cos, sin, is_decode_mode=False, memory_config=self._interleaved
            )

        mem = self._shard_mem
        if mem is None:
            return ttnn.experimental.rotary_embedding_hf(
                tensor, cos, sin, is_decode_mode=False, memory_config=self._interleaved
            )
        sh = ttnn.to_memory_config(tensor, mem)
        out = ttnn.experimental.rotary_embedding_hf(sh, cos, sin, is_decode_mode=False, memory_config=mem)
        ttnn.deallocate(sh)
        if SHARD_ROTARY == "keep":
            return out
        # Back to whichever interleaved space the chain lives in; handing SDPA a
        # DRAM tensor mid-L1-chain would reintroduce the round trip.
        il = ttnn.to_memory_config(out, self._interleaved)
        ttnn.deallocate(out)
        return il

    @staticmethod
    def _shard_config(shape):
        """Height-sharded config for a (B, H, S, D) rotary input, or None.

        Height sharding needs the tile-row count to divide across the grid, and it
        does not at every shape: b8s256 gives 768 tile-rows (64 x 12) but b1s256
        gives 96, which 64 does not divide. Fall back through narrower grids, and
        below the row threshold stay interleaved regardless.
        """
        rows = 1
        for d in shape[:-1]:
            rows *= int(d)
        tile_rows = rows // 32
        grid = next(((8, y) for y in (8, 6, 4, 2) if tile_rows % (8 * y) == 0), None)
        if grid is None or rows < _SHARD_ROTARY_MIN_ROWS:
            return None
        return ttnn.create_sharded_memory_config(
            shape=(1, rows, int(shape[-1])),
            core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    def deallocate(self):
        for cos, sin in self.caches.values():
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        self.caches = {}
