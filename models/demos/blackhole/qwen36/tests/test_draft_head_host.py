# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the drafter's bfloat4_b DRAM-sharded LM head (tt/draft_head.py).

* DraftHeadPlan at the Qwen3.8 TP=8 shape (head shard [5120, 31040], 8 DRAM banks, 11x10 grid): 122 tiles per bank,
  4 readers per bank, 2-tile K blocks, 10 logits tiles on each of 97 cores, the pipeline layout and L1 per core, and
  the copy's DRAM size.
* The program config, the activation and weight memory configs DraftHead and build_draft_head pass to ttnn.
* build_draft_head loads a bfloat4_b copy under its own cache name, vocab-sharded like the shared head, moves it to
  the DRAM-sharded layout and frees the interleaved copy; shapes the plan refuses load nothing and keep the shared head.
* Qwen36MTP.forward_decode: the per-shard draft logits of up to one tile of rows read the copy with the drafter's fp32
  compute config; more rows, the bf16 shard branch and a drafter without the copy read the shared head.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import draft_head as dh
from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP

K, VOCAB, TP = 5120, 248320, 8
N = VOCAB // TP
BLACKHOLE = dict(banks=8, grid=(11, 10))


def test_the_plan_at_the_tp8_shape():
    p = dh.DraftHeadPlan(K, N, **BLACKHOLE)
    assert (p.k_tiles, p.n_tiles, p.bank_tiles) == (160, 970, 122)
    assert p.bank_tiles * p.banks - p.n_tiles == 6  # padding tiles in the last bank, past N
    assert (p.readers, p.in0_block_w, p.act_shard_w) == (4, 2, 640)
    assert (p.per_core_N, p.out_cores) == (10, 97)
    assert p.layout == dict(cn=4, ck=1, p=4, ncp=8, kb=2, max_own=31, max_blocks=80, work=32)
    l1 = p.l1_bytes()
    # whole-K activation 160 bf16 tiles, 2 streams x 3 blocks x 2 K x 8 columns of bfloat4_b, 31 float32 output tiles
    assert l1["cb"] == 160 * 2048 + 2 * 3 * 2 * 8 * 576 + 31 * 4096 == 509952
    assert (l1["act_shard"], l1["out_shard"]) == (20 * 2048, 10 * 4096)
    assert l1["total"] == 591872 <= dh.L1_BANK_BYTES - dh.L1_RESERVED_BYTES
    # DRAM per device: the copy adds 89.9 MB; the shared bfloat8_b head (160 x 970 tiles of 1088 B) stays
    assert p.weight_bytes() == 160 * 976 * 576 == 89_948_160
    assert 160 * 970 * 1088 == 168_857_600


def _record(name):
    return lambda *args, **kw: SimpleNamespace(kind=name, args=args, **kw)


@pytest.fixture
def ttnn_configs(monkeypatch):
    """ttnn's config constructors return records of their arguments."""
    for name in (
        "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
        "create_sharded_memory_config",
        "MemoryConfig",
        "ShardSpec",
        "CoreRangeSet",
        "CoreGrid",
    ):
        monkeypatch.setattr(ttnn, name, _record(name))
    monkeypatch.setattr(ttnn, "CoreRange", lambda a, b: (a, b))
    monkeypatch.setattr(ttnn, "CoreCoord", lambda x, y: (x, y))


def test_the_program_and_memory_configs(ttnn_configs):
    p = dh.DraftHeadPlan(K, N, **BLACKHOLE)
    head = dh.DraftHead("w", p)
    pc = head.progcfg
    assert (pc.in0_block_w, pc.per_core_M, pc.per_core_N, pc.fused_activation) == (2, 1, 10, None)
    assert pc.num_workers_per_dram_bank == 4 and pc.use_multicore_pipeline is True
    act = head.act_memcfg
    assert act.shape == (32, 640) and (act.core_grid.x, act.core_grid.y) == (8, 1)
    assert act.strategy == ttnn.ShardStrategy.WIDTH and act.orientation == ttnn.ShardOrientation.ROW_MAJOR
    assert act.use_height_and_width_as_shard_shape is True
    w = p.weight_memory_config()
    assert w.args[:2] == (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM)
    grid, shape, orientation = w.args[2].args
    assert shape == [K, 122 * 32] and orientation == ttnn.ShardOrientation.ROW_MAJOR
    assert grid.args == ({((0, 0), (7, 0))},)  # one shard per DRAM bank


def _mesh(shape=(1, TP), grid=(11, 10), banks=8):
    return SimpleNamespace(
        shape=shape,
        get_num_devices=lambda: shape[0] * shape[1],
        compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid[0], y=grid[1]),
        dram_grid_size=lambda: SimpleNamespace(x=banks),
    )


@pytest.fixture
def loads(monkeypatch, ttnn_configs):
    """ttnn.as_tensor, to_memory_config and deallocate record their calls; tp_shard_mapper returns its arguments."""
    log = []

    def as_tensor(t, **kw):
        log.append(("as_tensor", t, kw))
        vocab, k = t.shape
        return SimpleNamespace(name="interleaved", shape=(k, vocab // TP))

    monkeypatch.setattr(ttnn, "as_tensor", as_tensor)
    monkeypatch.setattr(ttnn, "to_memory_config", lambda t, mc: log.append(("to_memory_config", t, mc)) or "sharded")
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("deallocate", t)))
    monkeypatch.setattr(dh.tpc, "tp_shard_mapper", lambda mesh, dim: ("tp_shard_mapper", dim))
    return log


def test_build_loads_a_bfloat4_b_copy_and_frees_the_interleaved_one(loads):
    weight = SimpleNamespace(shape=(VOCAB, K))
    head = dh.build_draft_head(weight, _mesh(), "cache/output.weight.vshard.draft_bfp4")
    assert head is not None and head.weight == "sharded"
    (op, src, kw), (op2, moved, mc), (op3, freed) = loads
    assert (op, op2, op3) == ("as_tensor", "to_memory_config", "deallocate")
    assert src is weight and kw["dtype"] == ttnn.bfloat4_b and kw["layout"] == ttnn.TILE_LAYOUT
    assert kw["memory_config"] == ttnn.DRAM_MEMORY_CONFIG
    assert kw["cache_file_name"] == "cache/output.weight.vshard.draft_bfp4"
    assert kw["mesh_mapper"] == ("tp_shard_mapper", -1)  # the shared head's vocab split
    assert kw["preprocess"](torch.zeros(4, 2)).shape == (2, 4)  # [vocab, k] -> [k, vocab]
    assert moved.name == "interleaved" and freed is moved
    grid, shape, _ = mc.args[2].args
    assert mc.args[:2] == (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM) and shape == [K, 3904]


def test_build_raises_when_the_loaded_shard_is_not_the_planned_one(loads, monkeypatch, expect_error):
    monkeypatch.setattr(ttnn, "as_tensor", lambda t, **kw: SimpleNamespace(shape=(K, VOCAB)))
    with expect_error(ValueError, "expected"):
        dh.build_draft_head(SimpleNamespace(shape=(VOCAB, K)), _mesh(), None)


@pytest.mark.parametrize(
    "vocab, k, mesh, reason",
    [
        (VOCAB + 1, K, _mesh(), "does not split"),
        (VOCAB, 5100, _mesh(), "whole number of tiles"),
        (VOCAB, K, _mesh(banks=7), "activation shards"),
        (VOCAB, K, _mesh(grid=(8, 8)), "logits cores"),
        (VOCAB, K, _mesh(shape=(1, 4)), "logits cores"),
        (VOCAB, 4 * K, _mesh(), "KB of L1"),
    ],
    ids=["vocab-not-split", "k-not-tiles", "7-dram-banks", "64-core-grid", "tp4", "l1"],
)
def test_shapes_the_plan_refuses_keep_the_shared_head(loads, expect_error, vocab, k, mesh, reason):
    tp = mesh.shape[1]
    if vocab % tp == 0:
        g = mesh.compute_with_storage_grid_size()
        with expect_error(ValueError, reason):
            dh.DraftHeadPlan(k, vocab // tp, mesh.dram_grid_size().x, (g.x, g.y))
    assert dh.build_draft_head(SimpleNamespace(shape=(vocab, k)), mesh, None) is None
    assert loads == []  # nothing loaded


def test_a_2d_mesh_splits_the_vocab_over_its_tp_axis(loads):
    head = dh.build_draft_head(SimpleNamespace(shape=(VOCAB, K)), _mesh(shape=(4, TP)), None)
    assert head is not None and head.plan.n_tiles == 970


def test_the_head_applies_to_one_tile_of_full_width_rows():
    head = dh.DraftHead.__new__(dh.DraftHead)
    head.plan = SimpleNamespace(k_tiles=160)
    x = lambda shape, layout=ttnn.TILE_LAYOUT: SimpleNamespace(shape=shape, layout=layout)
    assert all(head.applies(x((1, 1, r, K))) for r in (1, 8, 32))
    assert not head.applies(x((1, 1, 33, K)))
    assert not head.applies(x((1, 1, 8, K // TP)))
    assert not head.applies(x((1, 8, K)))
    assert not head.applies(x((1, 1, 8, K), ttnn.ROW_MAJOR_LAYOUT))


def _drafter(monkeypatch, rows, head):
    """A Qwen36MTP whose forward_decode runs to the head on ``rows`` rows; ttnn calls are logged."""
    log = []
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("deallocate", t)))
    monkeypatch.setattr(ttnn, "mesh_partition", lambda t, **kw: "next_hidden")
    monkeypatch.setattr(ttnn, "to_memory_config", lambda t, mc: log.append(("to_memory_config", t, mc)) or f"in {mc}")
    monkeypatch.setattr(ttnn, "linear", lambda x, w, **kw: log.append(("linear", x, w, kw)) or "logits")
    normed = SimpleNamespace(shape=(1, 1, rows, K), layout=ttnn.TILE_LAYOUT)
    mtp = Qwen36MTP.__new__(Qwen36MTP)
    mtp.num_devices = TP
    mtp.args = SimpleNamespace(get_norm_config=lambda key, mode: {})
    mtp._embed_rows = lambda ids: "emb"
    mtp._fuse = lambda emb, hidden, mode: "fused"
    mtp.decoder = SimpleNamespace(forward=lambda *a, **kw: "block")
    mtp.head_norm = lambda x, mode, norm_config: normed
    mtp._sharded_argmax, mtp._fp32_cfg = True, "fp32-cfg"
    mtp.lm_head_weight, mtp.draft_head = "shared-bf8", head
    return mtp, normed, log


def _bfp4_head():
    head = dh.DraftHead.__new__(dh.DraftHead)
    head.weight, head.plan = "bfp4-copy", SimpleNamespace(k_tiles=160)
    head.act_memcfg, head.progcfg = "act-shards", "mc-program"
    return head


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_draft_logits_of_one_tile_of_rows_read_the_bfp4_copy(monkeypatch, rows):
    mtp, normed, log = _drafter(monkeypatch, rows, _bfp4_head())
    logits, hidden = mtp.forward_decode("h", "ids", "pos", "cos", "sin", "pt")
    assert (logits, hidden) == (f"in {ttnn.DRAM_MEMORY_CONFIG}", "next_hidden")
    head_calls = [c for c in log if c[0] != "deallocate" or c[1] in ("in act-shards", "logits")]
    assert head_calls == [
        ("to_memory_config", normed, "act-shards"),
        (
            "linear",
            "in act-shards",
            "bfp4-copy",
            dict(
                program_config="mc-program",
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config="fp32-cfg",
                dtype=ttnn.float32,
            ),
        ),
        ("deallocate", "in act-shards"),
        ("to_memory_config", "logits", ttnn.DRAM_MEMORY_CONFIG),
        ("deallocate", "logits"),
    ]


@pytest.mark.parametrize(
    "rows, head, sharded_lm_head, want",
    [
        (33, "bfp4", False, dict(dtype=ttnn.float32, compute_kernel_config="fp32-cfg")),
        (8, None, False, dict(dtype=ttnn.float32, compute_kernel_config="fp32-cfg")),
        (8, "bfp4", True, {}),
    ],
    ids=["33-rows", "no-copy", "bf16-shard-branch"],
)
def test_other_draft_logits_read_the_shared_head(monkeypatch, rows, head, sharded_lm_head, want):
    mtp, normed, log = _drafter(monkeypatch, rows, _bfp4_head() if head else None)
    mtp.forward_decode("h", "ids", "pos", "cos", "sin", "pt", sharded_lm_head=sharded_lm_head)
    linears = [c for c in log if c[0] == "linear"]
    assert linears == [("linear", normed, "shared-bf8", want)]
    assert not any(c[0] == "to_memory_config" for c in log)
