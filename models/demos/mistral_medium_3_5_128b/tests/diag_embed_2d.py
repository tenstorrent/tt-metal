# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: reproduce the 2D embedding defect on the real trace token ids, in isolation.

``tests/unit/test_embedding_vs_ref.py`` asserts the 2D path bit-exact and passes — it draws ids
from ``torch.randint(0, vocab_size)``, uniform over 131072. Real prompt ids are not uniform; they
cluster in the low vocab, which is where the per-row shard boundaries (multiples of
``vocab_local = 16384``) actually are. This runs the same comparison on the trace's own ids.
"""

import torch
from loguru import logger

import ttnn
from models.demos.mistral_medium_3_5_128b.reference.golden import GoldenTrace
from models.demos.mistral_medium_3_5_128b.tt.embedding import ParallelEmbedding

SEQ = 10240


def test_embed_2d_on_real_ids(galaxy, cfg, mesh_config, ccl):
    torch.manual_seed(0)
    table = (torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    ids = GoldenTrace.from_env().token_ids(SEQ).to(torch.int32)

    vocab_local = cfg.vocab_size // mesh_config.sp
    logger.info(f"vocab_local={vocab_local}; id range {int(ids.min())}..{int(ids.max())}")
    for r in range(mesh_config.sp):
        n = int(((ids >= r * vocab_local) & (ids < (r + 1) * vocab_local)).sum())
        logger.info(f"  ids owned by SP row {r}: {n}")

    ref = table[ids[0].long()][None, None]  # [1,1,SEQ,hidden]

    for shard in (False, True):
        emb = ParallelEmbedding(galaxy, cfg, {"weight": table}, mesh_config, ccl, shard_vocab_on_sp=shard)
        tok = ttnn.from_torch(
            ids.reshape(1, 1, SEQ),
            device=galaxy,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                galaxy,
                mesh_shape=galaxy.shape,
                dims=(-1, None) if mesh_config.sp_axis == 0 else (None, -1),
            ),
        )
        out = ttnn.to_torch(
            emb(tok),
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                galaxy, mesh_shape=galaxy.shape, dims=(2, 3) if mesh_config.sp_axis == 0 else (3, 2)
            ),
        )[..., : cfg.hidden_size].to(torch.bfloat16)

        bad = (out != ref).any(dim=-1)[0, 0].nonzero().flatten()
        name = "2D" if shard else "1D"
        logger.info(f"{name}: {bad.numel()} / {SEQ} positions differ from the table")
        if bad.numel():
            pos = bad[:16].tolist()
            logger.info(f"{name} first bad positions: {pos}")
            logger.info(f"{name} their token ids:     {[int(ids[0, p]) for p in pos]}")
            allbad = sorted({int(ids[0, p]) for p in bad.tolist()})
            logger.info(f"{name} distinct bad ids ({len(allbad)}): {allbad[:24]}")
            logger.info(f"{name} bad ids mod {vocab_local}: {sorted({i % vocab_local for i in allbad})[:24]}")
            logger.info(f"{name} bad id // {vocab_local}: {sorted({i // vocab_local for i in allbad})}")

            # Local index within each SP shard: is the damage positional or id-dependent?
            per_row = SEQ // mesh_config.sp
            logger.info(f"{name} bad local idx: {sorted({p % per_row for p in bad.tolist()})}")

            # Zero output => the real contribution was lost (clamp or collective dropped it).
            # Non-zero but wrong => the index arithmetic picked the wrong table row.
            got_bad = out[0, 0, bad]
            n_zero = int((got_bad == 0).all(dim=-1).sum())
            logger.info(f"{name} bad rows that are exactly zero: {n_zero} / {bad.numel()}")
            if n_zero < bad.numel():
                nz = bad[(got_bad != 0).any(dim=-1)][:4].tolist()
                for p in nz:
                    match = (table == out[0, 0, p]).all(dim=-1).nonzero().flatten()
                    logger.info(f"{name}   pos {p} (id {int(ids[0,p])}) -> table row {match[:4].tolist()}")
