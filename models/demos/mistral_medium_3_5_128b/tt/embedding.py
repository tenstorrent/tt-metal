# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Token embedding on the 8x4 mesh, from ``minimax_m3/tt/parallel_embedding.py``.

Two sharding modes of the ``[vocab, hidden]`` table:

* **1D (default)**: hidden sharded over TP, vocab replicated over SP (``[vocab, hidden/tp]`` = 768 MiB
  per chip in bf16). Each chip looks up its own sequence shard's tokens in its slice; the result is
  already the sharded residual ``[1, 1, s_local, hidden/tp]``, with no CCL.
* **2D** (``shard_vocab_on_sp=True``): vocab also sharded over SP (``[vocab/sp, hidden/tp]`` per chip):
  SP all-gather of the tokens, sentinel-shifted local lookup (each shard padded with a zero row at both
  ends so out-of-shard ids clamp onto zeros), SP reduce-scatter on the sequence.

The table stays bf16: ``ttnn.embedding`` only takes a BFLOAT16 ROW_MAJOR table (see bringup_log M3), and
bf16 is the checkpoint's own dtype for it, so the lookup is exact.
"""

import torch

import ttnn

from .common import cache_name


class Embedding:
    def __init__(
        self,
        mesh_device,
        mesh_config,
        ccl_manager,
        weight,
        *,
        shard_vocab_on_sp: bool = False,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
    ):
        """``weight``: HF ``[vocab, hidden]`` table (None when loading from ``tensor_cache_path``)."""
        assert dtype == ttnn.bfloat16, "ttnn.embedding requires a bf16 table"
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.shard_vocab_on_sp = shard_vocab_on_sp
        sp = mesh_config.sp
        torch_weight = None if weight is None else weight.to(torch.bfloat16)[None, None]
        if shard_vocab_on_sp:
            vocab = None if weight is None else weight.shape[0]
            assert vocab is None or vocab % sp == 0, f"vocab {vocab} must divide by sp {sp}"
            mapper = mesh_config.mapper(mesh_device, sp_dim=2, tp_dim=3)
            tag = f"2d_sp{sp}_tp{mesh_config.tp}"
        else:
            mapper = mesh_config.mapper(mesh_device, tp_dim=3)
            tag = f"1d_tp{mesh_config.tp}"
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
            cache_file_name=cache_name(tensor_cache_path, f"weight_{tag}"),
        )
        if shard_vocab_on_sp:
            self.vocab_local = self.weight.shape[-2]
            # Zero sentinel rows: [0 | real shard | 0] -> in-shard ids land on rows 1..vocab_local.
            self.weight = ttnn.pad(self.weight, [(0, 0), (0, 0), (1, 1), (0, 0)], value=0.0)
            starts = torch.arange(sp, dtype=torch.float32).reshape(sp, 1, 1, 1) * float(self.vocab_local) - 1.0
            self.vocab_start = ttnn.from_torch(
                starts,
                device=mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_config.mapper(mesh_device, sp_dim=0),
            )

    def _lookup(self, idx):
        emb = ttnn.embedding(idx, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return ttnn.unsqueeze_to_4D(emb) if len(emb.shape) == 3 else emb

    def __call__(self, tokens):
        """``tokens`` uint32 ROW_MAJOR, per chip ``[1, 1, s_local]`` (sequence over SP, replicated on TP)
        -> sharded residual ``[1, 1, s_local, hidden/tp]`` bf16."""
        if not self.shard_vocab_on_sp:
            return self._lookup(tokens)
        mc, ccl = self.mesh_config, self.ccl_manager
        tok = ttnn.reshape(tokens, [1, 1, 1, tokens.shape[-1]])
        tok = mc.allgather(tok, ccl, axis=mc.sp_axis, dim=3)  # every SP row sees all positions
        s_total = tok.shape[-1]
        local = ttnn.subtract(ttnn.typecast(tok, ttnn.float32), self.vocab_start)
        local = ttnn.minimum(ttnn.maximum(local, 0.0), float(self.vocab_local + 1))
        emb = self._lookup(ttnn.reshape(ttnn.typecast(local, ttnn.uint32), [1, 1, s_total]))
        # Sum over the vocab shards (each id resolved by exactly one row) and scatter the sequence back.
        out = mc.reduce_scatter(emb, ccl, axis=mc.sp_axis, dim=2)
        emb.deallocate(True)
        return out
