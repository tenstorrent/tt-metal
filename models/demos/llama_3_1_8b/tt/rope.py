# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RoPE setup: Meta-interleaved cos/sin plus the 32x32 rotation-transformation matrix.

Reuses the shared ``tt_transformers`` rope math by **import** — ``gather_cos_sin`` (which produces
the Meta interleaved duplication ``[c0, c0, c1, c1, ...]``) and ``get_rot_transformation_mat``. The
full ``RotarySetup`` class is deliberately NOT used: it additionally builds batch-sharded decode
transformation matrices on a per-batch core grid, and this package is prefill-only.

**One source of frequencies.** The inverse frequencies come from ``reference/model.py``'s
``llama3_inv_freq`` — the same function the torch reference uses — so the device and the oracle
cannot disagree about the rotation by construction. That function is pinned against HuggingFace's own
``ROPE_INIT_FUNCTIONS["llama3"]`` and against ``tt_transformers.apply_scaling`` in
``tests/torch_ref/test_reference_llama.py``. ``tt_transformers.precompute_freqs`` is deliberately not
called for the angles: it forms ``position * inv_freq`` in fp32, which at position 10239 on the
unscaled j=1 frequency is an ~3300 rad angle whose fp32 ulp is 2.4e-4. That is harmless next to the
bf16 the tables are stored in, but computing the outer product in fp64 costs nothing and removes a
term from every rope comparison.

``LLAMA_ROPE_FREQ_FP16=1`` rounds the inverse frequencies to float16 before the outer product. That
is **off by default** and is not how Llama-3.1 is defined — HuggingFace keeps ``inv_freq`` in fp32
regardless of model dtype, and rounding it introduces a phase error that grows *linearly with
position* (about 1.7 rad by token 10240 on the fastest unscaled frequency) rather than staying
bounded. It exists because the shipped golden trace was generated that way, which is the single
reason the acceptance K numbers sit below ``pcc_target``; see
``tests/torch_ref/test_golden_trace_rope_precision.py``, which demonstrates the equivalence exactly,
and the README's PCC section.

Two consumers, one table:

* **per-chunk** (``chunk_cos_sin``): cos/sin sliced to one chunk's absolute positions and
  SP-sharded contiguously, fed to ``ttnn.experimental.rotary_embedding_llama``. Used by the unit
  tests and by the model when the runtime does not supply a whole-cache table.
* **whole-cache indexed** (``build_indexed_rope``): the entire cache's cos/sin, block-cyclic
  reordered by ``chunk_local`` then SP-sharded, fed to
  ``ttnn.experimental.deepseek_prefill.rotary_embedding_indexed``, which derives each chunk's start
  row on device from ``kv_actual_global`` plus the chip's SP coordinate. Built once; no per-chunk
  host reshard. This is the path the runtime uses, so chunk 0 and chunk N run identical code.

Both produce the same rotation. ``tests/unit/test_rope_vs_ref.py`` checks them against each other
and against the torch reference's HF-layout rotation.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.utils import block_cyclic_reorder
from models.tt_transformers.tt.common import gather_cos_sin, get_rot_transformation_mat

# The SAME frequency function the torch reference uses — see the module docstring.
from ..reference.model import llama3_inv_freq as inv_freq


def rope_tables(cfg, seq_len: int, start_pos: int = 0):
    """Host Meta-interleaved ``(cos, sin)``, each ``[1, 1, seq_len, head_dim]``.

    The angles are formed in fp64 and only the cos/sin values are narrowed; ``gather_cos_sin`` then
    does the Meta interleaved duplication that ``rotary_embedding_llama`` expects.
    """
    t = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq(cfg))  # [seq_len, head_dim/2]
    cos, sin = freqs.cos().float(), freqs.sin().float()
    # gather_cos_sin gathers by position into a table indexed from 0; ours already starts at
    # start_pos, so the gather is the identity and only the interleaving is used.
    return gather_cos_sin(torch.arange(seq_len), cos, sin)


class RopeSetup:
    """Owns the transformation matrix and hands out cos/sin in whichever layout is asked for."""

    def __init__(self, mesh_device, cfg, mesh_config, dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.mesh_config = mesh_config
        self.dtype = dtype
        # rotary_embedding_llama's transformation matrix is a single 32x32 tile regardless of
        # head_dim (get_rot_transformation_mat pins dhead=32 internally).
        self.transformation_mat = ttnn.from_torch(
            get_rot_transformation_mat(dhead=cfg.head_dim),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self._chunk_cache = {}

    def replicated(self, seq_len: int, start_pos: int = 0):
        """cos/sin for ``[start_pos, start_pos+seq_len)`` replicated on every device (sp=1 tests)."""
        cos, sin = rope_tables(self.cfg, seq_len, start_pos)
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return [
            ttnn.from_torch(t, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=self.dtype, mesh_mapper=mapper)
            for t in (cos, sin)
        ]

    def chunk_cos_sin(self, chunk_size: int, start_pos: int = 0):
        """SP-sharded cos/sin for one chunk: row ``r`` gets positions
        ``[start_pos + r*s_local, start_pos + (r+1)*s_local)`` — the contiguous split
        ``make_chunk_input`` gives the tokens, so the rope a device applies matches the tokens it holds.

        Cached per ``(chunk_size, start_pos)``: a served loop revisits the same starts, and each
        build is a host outer-product plus an H2D.
        """
        key = (chunk_size, start_pos)
        if key not in self._chunk_cache:
            cos, sin = rope_tables(self.cfg, chunk_size, start_pos)
            mapper = self.mesh_config.sequence_parallel(self.mesh_device, seq_dim=2)
            self._chunk_cache[key] = [
                ttnn.from_torch(
                    t, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=self.dtype, mesh_mapper=mapper
                )
                for t in (cos, sin)
            ]
        return self._chunk_cache[key]

    def build_indexed_rope(self, cache_seq: int, chunk_size: int):
        """Whole-cache block-cyclic SP-sharded cos/sin for ``rotary_embedding_indexed``.

        ``block_cyclic_reorder`` (imported from ``models/common/utils.py``, the same helper the KV
        writer's inverse ``blockcyclic_positions`` comes from) permutes the table so that after a
        plain SP shard on the seq dim, device ``c``'s contiguous rows hold — in local-cache-row order
        — the rope for every global position that device will ever carry. The op then reads a
        contiguous window starting at a row it derives from ``kv_actual_global`` and the chip's SP
        coordinate, which is the same arithmetic ``update_padded_kv_cache`` uses to place the tokens.
        """
        sp = self.mesh_config.sp
        assert cache_seq % chunk_size == 0, f"cache {cache_seq} must be a multiple of chunk {chunk_size}"
        chunk_local = chunk_size // sp
        cos, sin = rope_tables(self.cfg, cache_seq, 0)
        mapper = self.mesh_config.sequence_parallel(self.mesh_device, seq_dim=2)
        return [
            ttnn.from_torch(
                block_cyclic_reorder(t, chunk_local, sp, seq_dim=2),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.dtype,
                mesh_mapper=mapper,
            )
            for t in (cos, sin)
        ]
