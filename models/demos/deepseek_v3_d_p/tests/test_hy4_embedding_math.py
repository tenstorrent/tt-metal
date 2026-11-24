# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 prefill vocab-sharded embedding (#65892, tt/hy4/embedding.py): the sharding math and the cache rules, on CPU.

The device test is test_hy4_prefill_embedding.py."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ttnn")

from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config, Hy4TP32Config  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.hy4 import embedding as emb  # noqa: E402
from models.experimental.hy4_preview.reference.lm_head import VocabShards  # noqa: E402

SC1 = (8, 4)


def _weight(vocab: int, hidden: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # BF16-representable and never zero, so "one non-zero owner" is visible.
    return (torch.rand(vocab, hidden, generator=g) + 0.5).to(torch.bfloat16).float()


def test_hy4_split_sizes():
    shards = VocabShards(Hy4Config.VOCAB_SIZE, emb.num_chips(SC1))
    assert shards.rows_per_chip == 3776 and shards.table_rows == 3778
    new = shards.table_rows * Hy4Config.EMB_SIZE * 2
    old = Hy4Config.VOCAB_SIZE * (Hy4Config.EMB_SIZE // SC1[1]) * 2  # TtParallelEmbedding: hidden / 4, all ids
    assert new == 46_424_064 and old == 371_195_904
    assert old / new > 7.99


def test_chip_order_is_row_major():
    assert [emb.chip_of(SC1, (r, c)) for r in range(8) for c in range(4)] == list(range(32))
    assert emb.chip_of(SC1, (2, 3)) == 11


def test_host_table_places_each_chips_slice():
    w = _weight(32 * 5, 64)
    t = emb.host_table(w, SC1)
    assert tuple(t.shape) == (8, 4, 7, 64)
    for r in range(8):
        for c in range(4):
            chip = 4 * r + c
            assert torch.equal(t[r, c, 1:-1], w[5 * chip : 5 * (chip + 1)])
            assert not t[r, c, 0].any() and not t[r, c, -1].any()


def test_each_token_has_exactly_one_owner():
    vocab = 32 * 5
    w = _weight(vocab, 64)
    t = emb.host_table(w, SC1)
    shards = VocabShards(vocab, 32)
    ids = torch.arange(vocab)
    rows = shards.table_rows_for(ids)  # [32, vocab]
    looked = torch.stack([t[r, c][rows[4 * r + c]] for r in range(8) for c in range(4)])  # [32, vocab, 64]
    nonzero = looked.abs().sum(-1) > 0  # [32, vocab]
    assert torch.equal(nonzero.sum(0), torch.ones(vocab, dtype=torch.long))
    assert torch.equal(nonzero.float().argmax(0), ids // 5)
    assert torch.equal(looked.sum(0), w)  # the sum over all chips is the row, exactly


@pytest.mark.parametrize("mesh, sp_axis", [((8, 4), 0), ((4, 8), 1), ((2, 4), 0)])
def test_reduce_scatter_dataflow_is_the_residual_layout(mesh, sp_axis):
    """Chip (r, c) ends with its SP slice of the sequence and its TP slice of the hidden, bit for bit."""
    chips = mesh[0] * mesh[1]
    vocab, hidden = chips * 7, 4 * 32
    w = _weight(vocab, hidden, seed=1)
    sp, tp = mesh[sp_axis], mesh[1 - sp_axis]
    S = sp * 32
    ids = torch.randint(0, vocab, (S,), generator=torch.Generator().manual_seed(2))
    ids[:4] = torch.tensor([0, vocab - 1, 6, 7])  # first / last id, a chip edge
    full = w[ids]  # [S, hidden]
    out = emb.reference_output(w, ids, mesh, sp_axis)
    assert len(out) == chips
    for (r, c), got in out.items():
        i_sp, i_tp = (r, c)[sp_axis], (r, c)[1 - sp_axis]
        want = full[i_sp * (S // sp) : (i_sp + 1) * (S // sp), i_tp * (hidden // tp) : (i_tp + 1) * (hidden // tp)]
        assert torch.equal(got, want), (r, c)


def test_collective_bytes_per_chunk():
    b = emb.collective_bytes(4096, 6144, SC1, 0)
    assert b["lookup_out"] == 4096 * 6144 * 2 == 50_331_648
    assert b["rs_sp"] == 50_331_648 * 7 // 8
    assert b["rs_tp"] == (50_331_648 // 8) * 3 // 4
    assert b["id_gather"] == 7 * 512 * 4


def test_model_config_switch(expect_error):
    assert not emb.vocab_sharded(Hy4Config) and emb.vocab_sharded(Hy4TP32Config)
    assert emb.embedding_mesh(Hy4TP32Config) == ((8, 4), 0)
    with expect_error(ValueError, "EMBED_VOCAB_SHARDED"):
        emb.embedding_mesh(Hy4Config)
    assert not emb.vocab_sharded(None)


def test_cache_names_and_stamp(tmp_path, expect_error):
    import ttnn

    assert emb.cache_name(SC1) == "embed_vocab32_8x4"
    f = emb.cache_file(tmp_path, SC1)
    assert f.name == "embed_vocab32_8x4_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin"
    guard = emb.cache_guard(tmp_path, SC1, 0, Hy4Config.VOCAB_SIZE)
    assert guard.descriptor["module"] == "TtHy4PrefillEmbedding"
    assert guard.descriptor["dtypes"] == {"table": str(ttnn.bfloat16)}
    # Another mesh / split never shares the stamp or the descriptor.
    other = emb.cache_guard(tmp_path, (4, 8), 1, Hy4Config.VOCAB_SIZE)
    assert other.name != guard.name and other.descriptor != guard.descriptor
    assert emb.cache_guard(tmp_path, SC1, 0, 32 * 10).descriptor != guard.descriptor  # another vocab split

    check = lambda: emb.TtHy4PrefillEmbedding.check_cache_complete(tmp_path, SC1, 0, Hy4Config.VOCAB_SIZE)  # noqa
    assert not check()  # cold cache
    guard.check()  # cold: fine
    with expect_error(RuntimeError, "Refusing to fill the cache from placeholders"):
        guard.require_cached([f])  # a load without weights never caches torch.empty
    f.write_bytes(b"x")
    assert not check()  # a file with no stamp does not count
    with expect_error(RuntimeError, "no layout stamp"):
        guard.check()
    guard.write()  # what the module does after converting real weights
    assert check()
    guard.require_cached([f])
    other_cache = emb.cache_guard(tmp_path, SC1, 0, 32 * 10)
    with expect_error(RuntimeError, "another layout|records layout"):
        other_cache.check()


def test_embedding_cache_check_follows_the_model_config(tmp_path):
    from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import embedding_cache_complete
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

    init_checker(tmp_path)
    assert not embedding_cache_complete(tmp_path, Hy4TP32Config)
    assert not embedding_cache_complete(tmp_path, Hy4Config)

    def missing(cfg):
        return wc.missing_trunk_categories(
            tmp_path,
            num_layers=0,
            first_layer_idx=0,
            experts_per_chip=8,
            model_cfg=cfg,
            routed_expert_weights_dtype=None,
            is_first_rank=True,
            is_last_rank=False,
            mtp_levels=0,
            kv_only_last_layer=False,
        )

    assert wc.EMBEDDING in missing(Hy4TP32Config)
    # The vocab-32 table with its stamp satisfies the tp32 config only; the dev config still wants embed_weight.
    emb.cache_file(tmp_path, SC1).write_bytes(b"x")
    emb.cache_guard(tmp_path, SC1, 0, Hy4Config.VOCAB_SIZE).write()
    init_checker(tmp_path)
    assert embedding_cache_complete(tmp_path, Hy4TP32Config)
    assert not embedding_cache_complete(tmp_path, Hy4Config)
    assert wc.EMBEDDING not in missing(Hy4TP32Config)
    assert wc.EMBEDDING in missing(Hy4Config)


def test_kv_only_last_layer_on_2d_tp_needs_its_stamped_layer(tmp_path):
    """A kv-only 2D-TP last layer (tp32) reads the cache only under the layer's stamp, which only a full build writes;
    with no stamped 2D shards the runner must report MLA missing for it (and hand it the checkpoint weights), not
    load placeholders. The dev layout's kv-only check (its two KV files) is unchanged."""
    from models.demos.deepseek_v3_d_p.reference.hy4_config import hy4_hf_config
    from models.demos.deepseek_v3_d_p.tt.mla.mla_2d import MLA_2D_TP_FLAG
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE
    from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

    # Layer 4's dev (ttMLA) KV files, as the dev layout's cache holds them.
    for name in ("kv_a_layernorm", "kv_a_proj_with_mqa"):
        (tmp_path / f"layer_4.mla.{name}_dtype_BFLOAT16_layout_TILE.tensorbin").write_bytes(b"x")
    init_checker(tmp_path)

    def mla_missing(cfg, hf):
        return 4 in wc.missing_trunk_categories(
            tmp_path,
            num_layers=5,
            first_layer_idx=0,
            experts_per_chip=8,
            model_cfg=cfg,
            routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
            is_first_rank=False,
            is_last_rank=True,
            mtp_levels=0,
            kv_only_last_layer=True,
            hf_config=hf,
            tp_factor=4,
        ).get(wc.MLA, [])

    dev = hy4_hf_config()
    assert not mla_missing(Hy4Config, dev)  # layer 4 is a shared indexer layer: its KV files are enough
    tp32 = hy4_hf_config()
    setattr(tp32, MLA_2D_TP_FLAG, True)
    assert mla_missing(Hy4TP32Config, tp32)
