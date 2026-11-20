# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The layout arithmetic of the context-parallel KV cache (tt/kv_cache.py), tenstorrent/tt-metal#65886.
Pure CPU.

* The byte accounting against the figures of the issue.
* The prefill plan: padding to whole rounds and dealing a round's blocks to the chips (what
  ttnn.mesh_partition does on the device, modelled here in torch) is kv_layout's split, from any
  start; the keep mask protects exactly the positions before the start.
* The decode index table and the selection table against kv_layout, and the fp32 arithmetic that
  ``locate`` runs on the device (locate_rows, the same lines on torch fp32 tensors) against the
  selection table, up to 2**24 positions.
* What goes into the cache is what HF caches: HYV4Attention with a DynamicCache, prefill then
  decode, holds (k_pass, k_rot) and the indexer keys per position; the reference latent equals it,
  and striping it through the cache's write arithmetic and merging returns it bit for bit.
"""

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention, causal_mask, rope_cos_sin
from models.experimental.hy4_preview.tt.kv_cache import (
    HY4_INDEXER_LAYERS,
    INDEX_KEY,
    LATENT,
    KvCacheConfig,
    append_index_table,
    host_stripes,
    locate_rows,
    prefill_keep_mask,
    prefill_plan,
    selection_table,
)
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

LAYOUTS = [BlockCyclicLayout(), BlockCyclicLayout(8, 128), BlockCyclicLayout(4, 32), BlockCyclicLayout(1, 128)]
layouts = pytest.mark.parametrize("layout", LAYOUTS, ids=lambda layout: f"{layout.num_chips}x{layout.block}")


# ---------------------------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------------------------
def test_values_and_bytes_per_token():
    cfg = KvCacheConfig()
    assert len(cfg.indexer_layers) == 21
    assert cfg.values_per_token == 78 * 576 + 21 * 128 == 47616
    assert cfg.bytes_per_token == 47616 * 1088 / 1024 == 50592  # the issue's "~51 KB" in BFP8
    assert KvCacheConfig(dtype="bf16").bytes_per_token == 95232  # "~95 KB" in BF16
    # with the MTP layer, which has its own indexer
    mtp = KvCacheConfig(num_layers=79, indexer_layers=HY4_INDEXER_LAYERS + (78,))
    assert mtp.values_per_token == 48320


def test_indexer_layers_match_the_checkpoint_config():
    root = os.environ.get("HY4_HF_MODEL")
    if not root or not (Path(root) / "config.json").is_file():
        pytest.skip("HY4_HF_MODEL/config.json not found")
    config = json.loads((Path(root) / "config.json").read_text())
    full = tuple(i for i, kind in enumerate(config["indexer_types"]) if kind == "full")
    assert HY4_INDEXER_LAYERS == full
    cfg = KvCacheConfig()
    assert cfg.num_layers == config["num_hidden_layers"]
    assert cfg.latent_dim == config["kv_lora_rank"] + config["qk_rope_head_dim"]
    assert cfg.index_key_dim == config["index_head_dim"]


def test_storage_per_chip_is_one_32nd():
    for max_context, users in ((131072, 1), (1048576, 1), (1000000, 3)):
        cfg = KvCacheConfig(max_users=users, max_context=max_context)
        assert cfg.slots == cfg.layout.slots(max_context)
        assert cfg.capacity == cfg.slots * 32 >= max_context
        assert cfg.capacity - max_context < 32 * 128
        # every chip stores capacity / 32 positions of every user, at bytes_per_token each
        assert cfg.bytes_per_chip() == users * cfg.capacity // 32 * cfg.bytes_per_token
        assert cfg.bytes_per_chip() == sum(
            cfg.stripe_bytes(kind) for layer in range(cfg.num_layers) for kind in cfg.kinds(layer)
        )
    one_million = KvCacheConfig(max_context=1048576)
    assert one_million.slots == 32768
    assert one_million.bytes_per_chip() * 32 == 1048576 * 50592  # 53.0 GB: the issue's "~54 GB"
    assert KvCacheConfig().kinds(0) == (LATENT, INDEX_KEY) and KvCacheConfig().kinds(2) == (LATENT,)


def test_config_rejects_what_the_storage_cannot_hold():
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        KvCacheConfig(dtype="bfp4")
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        KvCacheConfig(latent_dim=570)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        KvCacheConfig(num_layers=4, indexer_layers=(0, 5))
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        prefill_plan(BlockCyclicLayout(), start=4000, length=200, capacity=4096)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        append_index_table(BlockCyclicLayout(), [4096], capacity=4096)


# ---------------------------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------------------------
def _device_stripes(layout, x, start):
    """write_prefill's device ops in torch: pad to whole rounds, view as [rounds, chips * block, W],
    and give chip c the c-th slice of every round (ttnn.mesh_partition on that dim)."""
    plan = prefill_plan(layout, start, x.shape[0])
    padded = torch.cat([x.new_zeros(plan.lead, x.shape[1]), x, x.new_zeros(plan.tail, x.shape[1])])
    rounds = padded.reshape(plan.rounds, layout.num_chips, layout.block, x.shape[1])
    return rounds.transpose(0, 1).reshape(layout.num_chips, plan.slots, x.shape[1])


def _store(layout, stripes, x, start):
    """TtHy4KvCache._store on a host model of one user's stripes [chips, slots, W]."""
    plan = prefill_plan(layout, start, x.shape[0], stripes.shape[1] * layout.num_chips)
    new = _device_stripes(layout, x, start)
    assert torch.equal(new, host_stripes(layout, x, start))
    if plan.lead:
        keep = prefill_keep_mask(layout, plan)[..., None]
        old = stripes[:, plan.first_slot : plan.first_slot + layout.block]
        new[:, : layout.block] = torch.where(keep, old, new[:, : layout.block])
    stripes[:, plan.first_slot : plan.first_slot + plan.slots] = new


@layouts
def test_prefill_plan(layout):
    round_len = layout.num_chips * layout.block
    for start, length in ((0, 1), (0, round_len), (0, round_len + 1), (5, 3), (round_len, 7), (3 * round_len - 1, 2)):
        plan = prefill_plan(layout, start, length)
        assert plan.lead + plan.length + plan.tail == plan.rounds * round_len
        assert 0 <= plan.lead < round_len and 0 <= plan.tail < round_len
        assert plan.slots == plan.rounds * layout.block
        first_round_position = start - plan.lead
        assert layout.slot(first_round_position) == plan.first_slot and layout.chip(first_round_position) == 0
        assert plan.first_slot % 32 == 0  # fill_cache writes at a tile-aligned row
        assert plan.first_slot + plan.slots == layout.slots(start + length)


@layouts
def test_prefill_from_zero_is_kv_layout_split(layout):
    round_len = layout.num_chips * layout.block
    for length in (1, layout.block - 1, layout.block, round_len - 3, round_len, round_len + 1, 2 * round_len + 77):
        x = torch.randn(length, 6, generator=torch.Generator().manual_seed(length))
        stripes = host_stripes(layout, x)
        assert torch.equal(stripes, layout.split(x))
        assert torch.equal(_device_stripes(layout, x, 0), stripes)
        assert torch.equal(layout.merge(stripes, length), x)


@layouts
def test_prefill_from_any_start(layout):
    round_len = layout.num_chips * layout.block
    total = 3 * round_len
    x = torch.randn(total, 5, generator=torch.Generator().manual_seed(1))
    for start, length in (
        (0, 10),
        (7, 1),
        (layout.block - 1, 2),
        (layout.block, layout.block),
        (round_len - 5, 9),
        (round_len, round_len),
        (round_len + 300 % round_len, round_len + 11),
        (2 * round_len - 1, round_len + 1),
    ):
        plan = prefill_plan(layout, start, length)
        stripes = host_stripes(layout, x[start : start + length], start)
        assert torch.equal(_device_stripes(layout, x[start : start + length], start), stripes)
        chips = torch.arange(layout.num_chips)[:, None]
        position = layout.position(chips, plan.first_slot + torch.arange(plan.slots)[None, :])
        inside = (position >= start) & (position < start + length)
        assert inside.sum() == length
        assert torch.equal(stripes[inside], x[position[inside]])
        assert (stripes[~inside] == 0).all()
        keep = prefill_keep_mask(layout, plan)
        assert torch.equal(keep, position[:, : layout.block] < start)
        assert keep.sum() == plan.lead


@layouts
def test_writes_compose_into_one_context(layout):
    """Prefill, decode appends and prefill continuations from arbitrary starts leave the stripes of
    the whole sequence: earlier positions survive, later ones land where kv_layout says."""
    round_len = layout.num_chips * layout.block
    total = 3 * round_len
    x = torch.randn(total, 4, generator=torch.Generator().manual_seed(2))
    stripes = torch.zeros(layout.num_chips, layout.slots(total), 4)
    length = 0
    g = torch.Generator().manual_seed(3)
    while length < total:
        if torch.rand(1, generator=g).item() < 0.5:  # a few decode steps
            for _ in range(min(3, total - length)):
                table = append_index_table(layout, [length])
                (chip,) = (table[:, 0] >= 0).nonzero().flatten().tolist()
                stripes[chip, table[chip, 0]] = x[length]
                length += 1
        else:  # a prefill chunk of any size
            chunk = min(int(torch.randint(1, round_len + 200, (1,), generator=g)), total - length)
            _store(layout, stripes, x[length : length + chunk], length)
            length += chunk
        assert torch.equal(layout.merge(stripes, length), x[:length]), length
        assert torch.equal(
            stripes[layout.filled(length, stripes.shape[1])], layout.split(x[:length])[layout.filled(length)]
        )


# ---------------------------------------------------------------------------------------------
# decode and selection tables
# ---------------------------------------------------------------------------------------------
@layouts
def test_append_index_table(layout):
    round_len = layout.num_chips * layout.block
    positions = [0, layout.block - 1, layout.block, round_len - 1, round_len, 2 * round_len + 5, None, -1]
    table = append_index_table(layout, positions)
    assert table.shape == (layout.num_chips, len(positions)) and table.dtype == torch.int32
    for user, position in enumerate(positions):
        column = table[:, user]
        if position is None or position < 0:
            assert (column == -1).all()
            continue
        assert (column >= 0).sum() == 1, "exactly one chip writes"
        chip = int(column.argmax())
        assert chip == layout.chip(position) and column[chip] == layout.slot(position)
        assert layout.position(chip, int(column[chip])) == position


def test_appends_cross_block_chip_and_round_boundaries():
    layout = BlockCyclicLayout()
    owners = [
        (int(c), int(table[c, 0]))
        for p in (127, 128, 1023, 1024, 4095, 4096)
        for table in [append_index_table(layout, [p])]
        for c in (table[:, 0] >= 0).nonzero().flatten()
    ]
    #          block | chip      mesh row 1 | 2 (8x4) round 0 | 1
    assert owners == [(0, 127), (1, 0), (7, 127), (8, 0), (31, 127), (0, 128)]


@layouts
def test_selection_table(layout):
    total = 3 * layout.num_chips * layout.block
    picks = torch.randperm(total, generator=torch.Generator().manual_seed(4))[:200]
    slot, mine = selection_table(layout, picks)
    assert slot.shape == mine.shape == (layout.num_chips, 200)
    assert (mine.sum(dim=0) == 1).all(), "one chip owns each selected position"
    chip = mine.float().argmax(dim=0)
    assert torch.equal(chip, layout.chip(picks))
    assert torch.equal(slot[chip, torch.arange(200)], layout.slot(picks))
    assert (slot[~mine] == 0).all()
    # gathering by (chip, slot) from the stripes returns the selected rows
    x = torch.randn(total, 3, generator=torch.Generator().manual_seed(5))
    stripes = layout.split(x)
    gathered = torch.stack([stripes[c][slot[c]] for c in range(layout.num_chips)])  # [chips, K, W]
    assert torch.equal((gathered * mine[..., None]).sum(dim=0), x[picks])
    # batched selections keep their leading dims
    slot2, mine2 = selection_table(layout, picks.reshape(4, 50))
    assert slot2.shape == (layout.num_chips, 4, 50) and torch.equal(slot2.reshape(-1, 200), slot)
    assert torch.equal(mine2.reshape(-1, 200), mine)


# locate_rows takes its elementwise ops from a namespace: ttnn on the device, torch here. ttnn.eq
# returns 0/1 in the operands' dtype, torch.eq a bool.
TORCH_FP32 = SimpleNamespace(
    floor=torch.floor,
    multiply=torch.multiply,
    subtract=torch.subtract,
    add=torch.add,
    eq=lambda a, b: torch.eq(a, b).to(a.dtype),
)


@layouts
def test_locate_rows_is_exact_in_fp32(layout):
    """The arithmetic of TtHy4KvCache.locate in fp32, for S queries of K picks, at every position a
    cache may hold (capacity is asserted <= 2**24) and with a user's row offset: the rows and the
    ownership flags are selection_table's. This pins the formula and its fp32 exactness, not the
    device kernels that run it."""
    capacity = 2**24
    round_len = layout.num_chips * layout.block
    g = torch.Generator().manual_seed(6)
    edges = [0, 1, layout.block - 1, layout.block, round_len - 1, round_len, 2**20 - 1, 2**20, capacity - 1]
    picks = torch.cat([torch.randint(0, capacity, (4 * 4096 - len(edges),), generator=g), torch.tensor(edges)])
    picks = picks.reshape(4, 4096)  # S = 4 queries, K = 4096
    positions = picks.to(torch.float32).reshape(1, 1, 4, 4096)
    assert torch.equal(positions.long().reshape(4, 4096), picks), "the positions themselves are exact in fp32"
    chip_id = torch.arange(layout.num_chips, dtype=torch.float32).reshape(-1, 1, 1, 1, 1)
    want_slot, want_mine = selection_table(layout, picks)  # [chips, 4, 4096]
    # user 0, and the last user that keeps every row below 2**24 (the cache asserts users * slots <= 2**24)
    slots = layout.slots(capacity)
    for row_offset in sorted({0, capacity - slots}):
        assert row_offset % slots == 0 and row_offset + slots <= capacity
        rows, mine = locate_rows(TORCH_FP32, positions, chip_id, layout, row_offset)
        assert rows.dtype == mine.dtype == torch.float32
        assert rows.shape == mine.shape == (layout.num_chips, 1, 1, 4, 4096)
        assert torch.equal(mine[:, 0, 0].bool(), want_mine)
        assert (mine.sum(dim=0) == 1).all(), "one chip owns each position"
        assert torch.equal(rows[:, 0, 0].long(), want_slot + row_offset)


def test_locate_rows_needs_powers_of_two():
    positions = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
    for layout in (BlockCyclicLayout(3, 128), BlockCyclicLayout(4, 96)):
        with pytest.raises(AssertionError, match="power-of-two"):  # allow-pytest.raises: host-side check
            locate_rows(TORCH_FP32, positions, torch.zeros(1, 1, 1, 1), layout)


# ---------------------------------------------------------------------------------------------
# what the cache holds is what HF caches
# ---------------------------------------------------------------------------------------------
SMALL = AttentionConfig(
    hidden_size=96,
    num_heads=4,
    q_lora_rank=48,
    kv_lora_rank=32,
    qk_nope_head_dim=24,
    qk_rope_head_dim=8,
    v_head_dim=16,
)
HF_NAMES = {
    "q_a_proj": "q_a_proj.weight",
    "q_a_layernorm": "q_a_layernorm.weight",
    "q_b_proj": "q_b_proj.weight",
    "kv_a_proj_with_mqa": "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm": "kv_a_layernorm.weight",
    "kv_b_proj": "kv_b_proj.weight",
    "o_proj": "o_proj.weight",
    "linear_gate": "gate_proj.weight",
    "learnable_sink_param": "sinks",
}


def test_cache_entries_are_what_hf_caches():
    hy_v4 = pytest.importorskip("transformers.models.hy_v4.modeling_hy_v4")
    from transformers.cache_utils import DynamicCache
    from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config

    cfg, index_dim = SMALL, 32
    hf_cfg = HYV4Config(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_heads,
        head_dim=cfg.qk_rope_head_dim,
        num_hidden_layers=2,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        index_n_heads=2,
        index_head_dim=index_dim,
        index_topk=16,
        indexer_types=["full", "shared"],
        mlp_layer_types=["dense", "sparse"],
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        attn_implementation="eager",
    )
    hf = hy_v4.HYV4Attention(hf_cfg, layer_idx=0).eval()  # a "full" layer: it has an indexer and caches index keys
    ref = Hy4Attention(cfg).eval()
    g = torch.Generator().manual_seed(11)
    hf_state = dict(hf.named_parameters())
    with torch.no_grad():
        for name, hf_name in HF_NAMES.items():
            p = getattr(ref, name)
            value = (
                torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5
                if p.dim() == 2
                else 1.0 + 0.5 * torch.randn(p.shape, generator=g)
            )
            p.copy_(value)
            hf_state[hf_name].copy_(value)
        for name, p in hf.indexer.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (p.shape[-1] ** -0.5 if p.dim() == 2 else 1.0))

    # a 12-chip-block toy layout so that 40 positions cross blocks, chips and a round
    layout = BlockCyclicLayout(num_chips=4, block=8)
    prompt, steps = 27, 13
    total = prompt + steps
    x = torch.randn(total, cfg.hidden_size, generator=g)
    positions = torch.arange(total)
    cache = DynamicCache(config=hf_cfg)
    # the keyword the module takes its cache by was renamed between transformers releases
    cache_kwarg = "cache" if "cache" in inspect.signature(hf.forward).parameters else "past_key_values"
    rope = hy_v4.HYV4RotaryEmbedding(hf_cfg)
    with torch.no_grad():
        hf(
            x[None, :prompt],
            position_embeddings=rope(x[None, :prompt], positions[None, :prompt]),
            attention_mask=causal_mask(prompt)[None, None],
            **{cache_kwarg: cache},
        )
        for t in range(prompt, total):
            hf(
                x[None, t : t + 1],
                position_embeddings=rope(x[None, t : t + 1], positions[None, t : t + 1]),
                attention_mask=torch.zeros(1, 1, 1, t + 1),
                **{cache_kwarg: cache},
            )
        cos, sin = rope_cos_sin(positions, cfg)
        _, k_pass, k_rope = ref.latent(x, cos, sin)
    hf_layer = cache.layers[0]
    hf_latent = torch.cat([hf_layer.keys[0, 0], hf_layer.values[0, 0]], dim=-1)  # (k_pass | k_rot) per position
    hf_index_keys = hf_layer.indexer_keys[0]
    assert hf_latent.shape == (total, cfg.latent_dim) and hf_index_keys.shape == (total, index_dim)

    # the reference latent is HF's cache entry (the prefill + decode split does not change it)
    latent = torch.cat([k_pass, k_rope], dim=-1)
    max_diff = (latent - hf_latent).abs().max().item()
    assert max_diff <= 1e-5 * hf_latent.abs().max().item(), f"max|Δ|={max_diff:.3e}"

    # written the way the cache writes (a prompt, then one token per step), HF's entries come back bit for bit
    for entries in (hf_latent, hf_index_keys):
        stripes = torch.zeros(layout.num_chips, layout.slots(total), entries.shape[1])
        _store(layout, stripes, entries[:prompt], 0)
        for t in range(prompt, total):
            table = append_index_table(layout, [t])
            chip = int(table[:, 0].argmax())
            stripes[chip, table[chip, 0]] = entries[t]
        assert torch.equal(layout.merge(stripes, total), entries)
        per_chip = layout.filled(total).sum(dim=1)
        assert per_chip.sum() == total and per_chip.max() - per_chip.min() <= layout.block
