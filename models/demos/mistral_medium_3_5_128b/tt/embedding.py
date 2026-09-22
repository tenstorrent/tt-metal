# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parallel token embedding. Ported from ``minimax_m3/tt/parallel_embedding.py``.

The table is ``[131072, 12288]`` — 1.6 G parameters, more than a decoder layer. Replicating it on
all 32 chips would cost 3 GiB each, so it is sharded, in one of two layouts:

1. **1D** (``shard_vocab_on_sp=False``) — hidden across the TP cols, replicated down the SP rows.
   Each chip stores ``[131072, 3072]`` = 0.75 GiB. The lookup is local; one TP all-gather rebuilds
   the full hidden width.
2. **2D** (``shard_vocab_on_sp=True``) — vocab across the SP rows *as well*, so each chip stores
   ``[16384, 3072]`` = 96 MiB. This is the Megatron vocab-parallel pattern and costs two SP-axis
   collectives per chunk; the source package measured it perf-neutral and bit-identical to 1D.

**1D is the default here because the 2D path is numerically broken on this mesh.** Its closing
SP reduce-scatter corrupts a fixed 12-row window — local sequence index 1088..1099, identical in
every one of the 8 shards, independent of token id and of the table contents. The damaged rows
come back either exactly zero or holding an unrelated table row, which is what rules out the
index arithmetic in :meth:`_lookup_2d`: a clamp or shift bug would mis-map *as a function of the
id*, deterministically, not wreck a positional window. 76 of 10240 positions are wrong, ~0.7%.

That fraction is small enough to look like precision and is not. It cost layer 0 ``k 0.9960 /
v 0.9961`` against the golden trace where the bfloat8_b weight floor is 0.99999, and because a
wrong embedding row is wrong in the *residual stream*, every later layer inherited it — the
full-depth run bottomed out at ``v 0.552``, far below the spec's ``pcc_lower_bound``. On 1D the
same layer 0 measures ``k 0.9999711 / v 0.9999297``, i.e. at the weight floor.

``tests/unit/test_embedding_vs_ref.py`` asserted both layouts bit-exact and passed throughout,
because it drew ids from ``torch.randint(0, vocab_size)``. That is the lesson worth keeping: a
uniform random id is a *worse* test than a real prompt here, and the defect is positional, so it
only showed up at a sequence length the unit test never ran. ``test_embed_real_prompt_ids`` now
covers both layouts at the acceptance length on the trace's own ids, and xfails 2D.

The cost of 1D is memory: ``[131072, 3072]`` bf16 = 768 MiB/chip against 96 MiB, on a budget
where the 88 layers already take ~30.1 GiB of ~31.8 GiB. It fits, and the full-depth acceptance
run below is the measurement that says so.

The 2D lookup's trick is worth stating because it is not obvious: each vocab shard is stored padded
with **one zero row above and below**, and the token is shifted by ``row*vocab_local - 1`` and
clamped to ``[0, vocab_local+1]``. A token this row owns lands on a real row; a token it does not
own clamps onto a zero sentinel. The SP reduce-scatter then sums across rows — exactly one row
contributed a non-zero vector — so no output mask is needed anywhere.

**The table is bfloat16, not the spec's ``bfloat8_b``.** ``ttnn.embedding`` needs a ROW_MAJOR weight
and ``bfloat8_b`` is a tiled-only format, so this one tensor cannot carry the spec's weight
dataformat. It costs 2x the memory of a bf8 table and is the only weight in the model stored wider
than the spec asks; the alternative would be a gather composed from a one-hot matmul, which at
vocab 131072 is 10240x131072 of ones and zeros per chunk.

Both layouts write distinct cache keys (``_1d`` / ``_2d``, neither a prefix of the other) so a
stale layout can never be loaded as the other.
"""

import os

import torch

import ttnn
from models.demos.mistral_medium_3_5_128b.utils.general_utils import get_cache_file_name

#: 1D (hidden-only) sharding is the default: it is the layout that is *correct* on this mesh.
#: 2D would save 672 MiB/chip but corrupts a 12-row window per shard — see the module docstring.
DEFAULT_SHARD_VOCAB_ON_SP = False


def embed_shard_vocab_on_sp() -> bool:
    """The single toggle between the two layouts. ``MISTRAL_EMBED_SHARD_VOCAB=1`` selects 2D,
    which is retained only so the defect stays reproducible — it is not correct on this mesh."""
    v = os.getenv("MISTRAL_EMBED_SHARD_VOCAB")
    if v is None:
        return DEFAULT_SHARD_VOCAB_ON_SP
    return v.strip().lower() in ("1", "true", "yes", "on")


def embed_cache_name(shard_vocab_on_sp: bool) -> str:
    """Layout-tagged cache stem. The two names must not prefix each other."""
    return "embed_tokens_2d" if shard_vocab_on_sp else "embed_tokens_1d"


class ParallelEmbedding:
    """Token lookup producing the package's activation layout: ``[1, 1, tokens_local, hidden]``,
    sequence SP-sharded, hidden replicated across TP."""

    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        *,
        shard_vocab_on_sp: bool | None = None,
        weight_dtype=ttnn.bfloat16,
        tensor_cache_path=None,
    ):
        """
        Args:
            mesh_device: the open mesh.
            config: a :class:`~...reference.model_config.MistralMediumConfig`.
            state_dict: ``{"weight": [vocab_size, hidden_size]}``. Empty dict => cache-only load.
            mesh_config: :class:`~...tt.config.MeshConfig`.
            ccl_manager: required unless the mesh is 1x1 — every path ends in a collective.
            shard_vocab_on_sp: layout override; defaults to :func:`embed_shard_vocab_on_sp`.
            weight_dtype: must be a ROW_MAJOR-capable dtype (see the module docstring).
            tensor_cache_path: directory for the weight cache, or None.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.dtype = weight_dtype
        self.shard_vocab_on_sp = embed_shard_vocab_on_sp() if shard_vocab_on_sp is None else shard_vocab_on_sp

        tp, sp = mesh_config.tp, mesh_config.sp
        assert self.hidden_size % tp == 0, f"hidden_size ({self.hidden_size}) must divide TP ({tp})"
        assert tp == 1 or ccl_manager is not None, "the closing TP all-gather needs a CCLManager"

        shard_dims = [None, None]
        shard_dims[mesh_config.tp_axis] = -1  # hidden
        if self.shard_vocab_on_sp:
            assert (
                self.vocab_size % sp == 0
            ), f"2D embedding: vocab_size ({self.vocab_size}) must divide SP ({sp}); pad the table first"
            shard_dims[mesh_config.sp_axis] = 0  # vocab
            self.vocab_local = self.vocab_size // sp
            self.vocab_start = self._build_vocab_start(sp, self.vocab_local)
        else:
            self.vocab_local = self.vocab_size
            self.vocab_start = None

        torch_weight = state_dict["weight"].reshape(self.vocab_size, self.hidden_size) if state_dict else None
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=tuple(shard_dims)),
            cache_file_name=get_cache_file_name(tensor_cache_path, embed_cache_name(self.shard_vocab_on_sp)),
        )
        if self.shard_vocab_on_sp:
            # The zero sentinel rows. Padded on device after the load so the cached tensor stays
            # unpadded and is reusable as-is.
            pad = [(0, 0)] * len(self.weight.shape)
            pad[-2] = (1, 1)
            self.weight = ttnn.pad(self.weight, pad, value=0.0)

    def _build_vocab_start(self, sp: int, vocab_local: int):
        """``[1,1,1,1]`` holding ``row*vocab_local - 1`` on SP row ``row``, replicated across TP.

        fp32 because the subtraction goes negative for a token the row does not own, and is exact
        for every id below 2**24 — the vocab is 131072.
        """
        starts = torch.arange(sp, dtype=torch.float32).reshape(sp, 1, 1, 1) * float(vocab_local) - 1.0
        dims = [None, None]
        dims[self.mesh_config.sp_axis] = 0
        return ttnn.from_torch(
            starts,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=self.mesh_device.shape, dims=tuple(dims)),
        )

    def __call__(self, tokens):
        """``tokens``: uint32 ``[1, 1, tokens_local]``, SP-sharded on the sequence, TP-replicated.

        Returns ``[1, 1, tokens_local, hidden_size]`` — the residual-stream layout every decoder
        layer expects.
        """
        emb = self._lookup_2d(tokens) if self.shard_vocab_on_sp else self._lookup_1d(tokens)
        if self.mesh_config.tp > 1:
            gathered = self.mesh_config.allgather(emb, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=3)
            emb.deallocate(True)
            emb = gathered
        return emb

    def _lookup_1d(self, tokens):
        emb = ttnn.embedding(tokens, self.weight, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
        return ttnn.unsqueeze_to_4D(emb) if len(emb.shape) == 3 else emb

    def _lookup_2d(self, tokens):
        sp = self.mesh_config.sp
        # 1) SP all-gather the ids (uint32, a few KB) so every row can try every position.
        tok4d = ttnn.reshape(tokens, [1, 1, 1, tokens.shape[-1]])
        if sp > 1:
            tok4d = self.mesh_config.allgather(tok4d, self.ccl_manager, axis=self.mesh_config.sp_axis, dim=3)
        s_total = tok4d.shape[-1]

        # 2) Sentinel lookup: shift into this row's vocab window, clamp the rest onto the zero rows.
        local = ttnn.subtract(ttnn.typecast(tok4d, ttnn.float32), self.vocab_start)
        local = ttnn.minimum(ttnn.maximum(local, 0.0), float(self.vocab_local + 1))
        local_idx = ttnn.reshape(ttnn.typecast(local, ttnn.uint32), [1, 1, s_total])
        emb = ttnn.embedding(local_idx, self.weight, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)

        # 3) SP reduce-scatter on the sequence: sums the one real contribution per token and puts
        #    the sequence back on its owning row in a single collective.
        if sp > 1:
            emb = self.mesh_config.reduce_scatter(emb, self.ccl_manager, dim=2, axis=self.mesh_config.sp_axis)
        return emb
