# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of the global KV pair split (reference/kv_pair_split.py; no device): the position mappings, the
prefill fill tables and decode positions driven through a torch mirror of the paged cache ops, the LSE merge against
full attention, the prefill padding and warmup under the split, the adapter's KV capacity (through the plugin's pool
sizing and vLLM's planner where they are installed), its per-group block sizes and page-table widths, and that a
chunk start prefill_forward moves back reads only sliding-window blocks vLLM still holds."""

import math
import random
from types import SimpleNamespace

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference import kv_pair_split as kps
from models.experimental.mimo_v2_6_flash.tt import kv as kv_lib
from models.experimental.mimo_v2_6_flash.tt.attention import TtMiMoAttention
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel, prefill_padded_len, prefill_warmup_chunks

SUB, PAIR, TP = kps.SUB, kps.PAIR, 8
MAX_1M = 1 << 20


def brute_visible(p, c):
    return sum(1 for q in range(p + 1) if ((q >> 6) & 1) == c) if p >= 0 else 0


def visible_by_count(p, c):
    """Local keys of parity c at global positions <= p, counted with tensor ops (any p up to 1M)."""
    if p < 0:
        return 0
    q = torch.arange(p + 1, dtype=torch.int64)
    return int((((q >> 6) & 1) == c).sum())


# ---------------------------------------------------------------------- mappings
@pytest.mark.parametrize("p", [0, 1, 62, 63, 64, 65, 126, 127, 128, 129, 191, 192, 255, 256, 1000])
def test_mappings_at_the_boundaries(p):
    c_own = (p >> 6) & 1
    assert kps.own(p, c_own) and not kps.own(p, 1 - c_own)
    assert kps.global_position(kps.local(p), c_own) == p
    for c in (0, 1):
        assert kps.visible(p, c) == brute_visible(p, c)
        assert kps.write_position(p, c) == (kps.local(p) if c == c_own else -1)
        s = kps.sdpa_position(p, c)
        assert s == kps.visible(p, c) - 1
        # The SDPA position is the last local key at or before p: the next one is past p.
        if s >= 0:
            assert kps.global_position(s, c) <= p < kps.global_position(s + 1, c)
        else:
            assert kps.global_position(0, c) > p
    # The pair's two visible counts add up to every position.
    assert kps.visible(p, 0) + kps.visible(p, 1) == p + 1


def test_mappings_named_cases():
    # Chip 0 keeps 0..63, 128..191, ...; chip 1 keeps 64..127, 192..255, ...
    assert [kps.local(p) for p in (0, 63, 64, 127, 128, 191, 192, 255, 256)] == [0, 63, 0, 63, 64, 127, 64, 127, 128]
    assert [kps.sdpa_position(p, 0) for p in (-1, 0, 63, 64, 127, 128, 255, 256)] == [-1, 0, 63, 63, 63, 64, 127, 128]
    assert [kps.sdpa_position(p, 1) for p in (-1, 0, 63, 64, 127, 128, 255, 256)] == [-1, -1, -1, 0, 63, 63, 127, 127]
    assert kps.write_position(-1, 0) == kps.write_position(-1, 1) == -1


def test_mappings_random_up_to_1m():
    rng = random.Random(5)
    for p in [rng.randrange(MAX_1M) for _ in range(40)] + [MAX_1M - 1, MAX_1M - 64, MAX_1M - 65, MAX_1M - 128]:
        c_own = (p >> 6) & 1
        assert kps.global_position(kps.local(p), c_own) == p
        for c in (0, 1):
            assert kps.visible(p, c) == visible_by_count(p, c), p
        assert kps.visible(p, 0) + kps.visible(p, 1) == p + 1
    # local is a bijection from each parity's positions onto 0, 1, 2, ...
    for c in (0, 1):
        positions = [p for p in range(4 * PAIR + 37) if kps.own(p, c)]
        assert [kps.local(p) for p in positions] == list(range(len(positions)))


def test_positions_step_is_the_scalar_functions():
    rng = random.Random(9)
    positions = [-1, 0, 63, 64, 127, 128, MAX_1M - 1] + [rng.randrange(-1, MAX_1M) for _ in range(25)]
    for c in (0, 1):
        write, sdpa, serial = kps.positions_step(positions, c, serial_slots=4)
        assert write == [kps.write_position(p, c) for p in positions]
        assert sdpa == [kps.sdpa_position(p, c) for p in positions]
        for j, row in enumerate(serial):
            assert row == [write[k] if k % 4 == j else -1 for k in range(len(positions))]
    w, s = kps.chip_positions(torch.tensor(positions, dtype=torch.int32), TP)
    assert w.shape == s.shape == (TP, len(positions)) and w.dtype == torch.int32
    for chip in range(TP):
        assert w[chip].tolist() == kps.positions_step(positions, chip % 2)[0]


# ---------------------------------------------------------------------- the paged cache, mirrored
class PairCache:
    """Each chip parity's paged cache (pages of SUB rows, one value per row: the global position written there) and
    torch mirrors of the paged ops through vLLM's global table (one block per PAIR positions)."""

    def __init__(self, num_blocks):
        self.pages = {c: torch.full((num_blocks, SUB), -7, dtype=torch.int64) for c in (0, 1)}

    def fill(self, c, table, values):
        """paged_fill_cache: input row i -> page table[i // SUB], row i % SUB."""
        for i, v in enumerate(values):
            self.pages[c][table[i // SUB], i % SUB] = v

    def update(self, c, row, pos, value):
        """paged_update_cache of one user at local position ``pos`` (-1: skipped)."""
        if pos >= 0:
            self.pages[c][row[pos // SUB], pos % SUB] = value

    def read(self, c, row, last):
        """What paged SDPA reads for a user at local position ``last``: local indices 0 .. last."""
        return [int(self.pages[c][row[l // SUB], l % SUB]) for l in range(last + 1)]


@pytest.mark.parametrize("length, chunk", [(1, 4096), (63, 4096), (64, 4096), (129, 4096), (700, 256), (1333, 512)])
def test_fill_and_decode_through_the_tables(length, chunk):
    """A request prefilled the way TtMiMoModel chunks it (first chunk at 0, later ones at multiples of 128, padded as
    prefill_padded_len pads them under the split) and decoded for a few steps: every chip's cache holds exactly its
    positions at their local indices, padding and the other chip's halves land in the scratch block, and each decode
    step's SDPA reads exactly the chip's positions up to the user's."""
    steps = 70
    blocks = math.ceil((length + steps + kv_lib.PREFILL_MAX_PAD + PAIR) / PAIR)
    row = torch.tensor(random.Random(length).sample(range(1, 4 * blocks), blocks), dtype=torch.int32)
    table = row.reshape(1, 1, -1).expand(TP, 1, -1)
    cache = PairCache(4 * blocks)
    for s in range(0, length, chunk):
        n = min(chunk, length - s)
        padded = prefill_padded_len(n, s, pair_split=True)
        assert padded % (PAIR if s else 32) == 0 and padded >= n
        fills = kv_lib.pair_fill_rows(table, 0, s, padded, scratch=0)
        assert fills.shape == (TP, 1, -(-padded // SUB))
        for chip in range(TP):
            assert torch.equal(fills[chip], fills[chip % 2])  # one table per parity
        for c in (0, 1):
            # padding rows write position -1 (they are rewritten by decode before any read)
            cache.fill(c, fills[c, 0].tolist(), [s + i if i < n else -1 for i in range(padded)])
    for step in range(steps):
        p = length + step
        for c in (0, 1):
            cache.update(c, row, kps.write_position(p, c), p)
            got = cache.read(c, row, kps.sdpa_position(p, c))
            assert got == [q for q in range(p + 1) if kps.own(q, c)], (p, c)
    # Nothing past the context's halves was written into a real block of the other chip.
    for c in (0, 1):
        for l in range(kps.visible(length + steps - 1, c)):
            assert int(cache.pages[c][row[l // SUB], l % SUB]) == kps.global_position(l, c)


def test_fill_tables_scratch_and_batches(expect_error):
    row = torch.arange(10, 20, dtype=torch.int32)  # 10 global blocks: positions 0 .. 1279
    assert kps.fill_table(row, 0, 32, 0, scratch=0) == [10]
    assert kps.fill_table(row, 0, 32, 1, scratch=0) == [0]
    assert kps.fill_table(row, 0, 96, 1, scratch=0) == [0, 10]
    assert kps.fill_table(row, 128, 256, 0, scratch=-3) == [11, -3, 12, -3]
    assert kps.fill_table(row, 128, 256, 1, scratch=-3) == [-3, 11, -3, 12]
    # Past the end of the row: scratch.
    assert kps.fill_table(row, 1152, 256, 0, scratch=0) == [19, 0, 0, 0]
    with expect_error(AssertionError, "chunk starts on a"):
        kps.fill_table(row, 64, 128, 0)
    table = torch.stack([row + 100 * s for s in range(3)]).unsqueeze(0).expand(TP, -1, -1)
    batch = kv_lib.batch_fill_rows(table, [2, 0], 512, SUB, 0, pair_split=True)
    assert batch.shape == (TP, 2, 8)
    for chip in range(TP):
        for r, slot in enumerate([2, 0]):
            assert batch[chip, r].tolist() == kps.fill_table(table[chip, slot], 0, 512, chip % 2, 0)
    # Without the split the batch rows are unchanged.
    plain = kv_lib.batch_fill_rows(table, [2, 0], 512, SUB, 0)
    assert torch.equal(plain[:, 0], table[:, 2, :8]) and torch.equal(plain[:, 1], table[:, 0, :8])


# ---------------------------------------------------------------------- attention and the merge
def test_merge_weights_device_form():
    g = torch.Generator().manual_seed(3)
    a, b = torch.randn(4096, generator=g) * 20, torch.randn(4096, generator=g) * 20
    ref, dev = kps.merge_weights(a.double(), b.double()), kps.merge_weights_device(a.double(), b.double())
    assert (ref - dev).abs().max() < 1e-30 + 1e-12
    none = torch.tensor(kps.NO_KEY_LSE, dtype=torch.float32)
    x = torch.tensor(3.5, dtype=torch.float32)
    for weights, nothing in ((kps.merge_weights, 0.0), (kps.merge_weights_device, 2e-35)):
        # No key on this chip: (about) 0, and its partial output is 0; no key on the partner: 1; on neither: 1/2.
        assert 0.0 <= float(weights(none, x)) <= nothing and float(weights(x, none)) == 1.0
        assert float(weights(none, none)) == 0.5
        assert float(weights(x, x)) == 0.5


@pytest.mark.parametrize("weights", [kps.merge_weights, kps.merge_weights_device], ids=["spec", "device"])
def test_pair_attention_is_full_attention(weights):
    g = torch.Generator().manual_seed(1)
    heads, d, dv, scale = 4, 24, 16, 0.3
    T = 700
    k = torch.randn(T, d, generator=g, dtype=torch.float64)
    v = torch.randn(T, dv, generator=g, dtype=torch.float64)
    # Decode: one query per user at positions straddling the 64 / 128 boundaries (each its own context).
    for p in [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 699]:
        q = torch.randn(heads, 1, d, generator=g, dtype=torch.float64)
        want = kps.full_attention(q, k[: p + 1], v[: p + 1], scale, [p])
        got = kps.pair_attention(q, k[: p + 1], v[: p + 1], scale, [p], weights)
        assert torch.allclose(got, want, atol=1e-12), p
    # Prefill chunks: queries start .. start + S - 1 over the context up to the chunk's end.
    for start, S in [(0, 128), (128, 128), (256, 384), (512, 128)]:
        q = torch.randn(heads, S, d, generator=g, dtype=torch.float64)
        pos = list(range(start, start + S))
        end = start + S
        want = kps.full_attention(q, k[:end], v[:end], scale, pos)
        got = kps.pair_attention(q, k[:end], v[:end], scale, pos, weights)
        assert torch.allclose(got, want, atol=1e-12), (start, S)


def test_partial_with_no_visible_key():
    g = torch.Generator().manual_seed(2)
    k, v = torch.randn(10, 8, generator=g), torch.randn(10, 4, generator=g)
    q = torch.randn(2, 1, 8, generator=g)
    o, lse = kps.partial_attention(q, k, v, 0.5, [9], 1)  # positions 0..9 are all chip 0's
    assert torch.equal(o, torch.zeros_like(o)) and (lse == kps.NO_KEY_LSE).all()
    o0, lse0 = kps.partial_attention(q, k, v, 0.5, [9], 0)
    assert torch.allclose(kps.merge(o0, lse0, o, lse), kps.full_attention(q, k, v, 0.5, [9]), atol=1e-6)


# ---------------------------------------------------------------------- prefill padding and warmup
def pair_program_key(padded, start, window=128):
    lead = start - max(0, start - window) // SUB * SUB
    return lead, TtMiMoAttention._prefill_sdpa_program(None, padded, start).q_chunk_size


def test_pair_padding_and_sdpa_chunks():
    for n in range(1, 4096 + 128):
        assert prefill_padded_len(n, 0, True) == prefill_padded_len(n)  # first chunks unchanged
        padded = prefill_padded_len(n, PAIR, True)
        assert padded % PAIR == 0 and n <= padded <= n + kv_lib.PREFILL_MAX_PAD
        assert prefill_padded_len(n, PAIR, False) == prefill_padded_len(n)
    for padded in [128, 256, 384, 1920, 2048, 2304, 4096]:
        for start in range(PAIR, 8192, PAIR):
            program = TtMiMoAttention._prefill_sdpa_program(None, padded, start)
            q, k = program.q_chunk_size, program.k_chunk_size
            # The pair-split op's tested configurations: Q and K chunks equal, dividing the start and the length.
            assert q == k and padded % q == 0 and start % q == 0


@pytest.mark.parametrize("max_tokens, max_seq", [(2048 + PAIR - 1, 8192), (4096 + PAIR - 1, 1 << 20), (300, 300)])
def test_pair_warmup_covers_every_chunk(max_tokens, max_seq):
    """Every (padded length, program key) a pair-split chunk of up to the warmup's tokens can run is in the list."""
    window = 128
    listed = prefill_warmup_chunks(max_tokens, max_seq, SUB, window, pair_split=True)
    first = {prefill_padded_len(n) for n, s in listed if not s}
    leads, chunks_ = {}, {}
    for n, s in listed:
        if s:
            assert s % PAIR == 0
            padded = prefill_padded_len(n, s, True)
            lead, chunk = pair_program_key(padded, s, window)
            leads.setdefault(padded, set()).add(lead)
            chunks_.setdefault(padded, set()).add(chunk)
    for n in range(1, min(max_tokens, max_seq) + 1):
        assert prefill_padded_len(n) in first
        padded = prefill_padded_len(n, PAIR, True)
        for start in range(PAIR, min(max_seq - n + 1, 8192), PAIR):
            lead, chunk = pair_program_key(padded, start, window)
            assert lead in leads[padded] and chunk in chunks_[padded], (n, start)


# ---------------------------------------------------------------------- the adapter
def test_default_pool_holds_4x1m_and_a_short_cap_warns(monkeypatch):
    """With the pair split and MIMO_KV_GB_PER_CHIP unset the pool holds 4 x 1M per lane; at 6.6 (3.95 requests) the
    adapter warns and names the least cap that holds them."""
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv

    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", True)
    warnings = []
    monkeypatch.setattr(gv.logger, "warning", warnings.append)
    monkeypatch.delenv("MIMO_KV_GB_PER_CHIP", raising=False)
    demand = 4 * (kps.global_pool_tokens(MAX_1M) + gv.swa_tokens_per_user())
    assert gv.MiMoV2ForCausalLM.get_max_tokens_all_users(max_model_len=MAX_1M, max_num_seqs=4) == demand
    assert not warnings
    monkeypatch.setenv("MIMO_KV_GB_PER_CHIP", "6.6")
    tokens = gv.MiMoV2ForCausalLM.get_max_tokens_all_users(max_model_len=MAX_1M, max_num_seqs=4)
    assert tokens < demand and len(warnings) == 1
    assert "holds 3.95 requests" in warnings[0] and "MIMO_KV_GB_PER_CHIP >= 6.69" in warnings[0], warnings[0]


def test_capacity_4x1m_at_6p7_gb(monkeypatch):
    """The model's share of the pool (get_max_tokens_all_users) holds 4 x 1M per lane within MIMO_KV_GB_PER_CHIP=6.7;
    the plugin's sizing on top of it is test_plugin_pool_admits_4x1m."""
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv

    monkeypatch.setenv("MIMO_KV_GB_PER_CHIP", "6.7")
    budget_tokens = int(6.7e9 // gv.KV_BYTES_PER_TOKEN)
    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", True)
    tokens = gv.MiMoV2ForCausalLM.get_max_tokens_all_users(max_model_len=MAX_1M, max_num_seqs=4)
    demand = 4 * (kps.global_pool_tokens(MAX_1M) + gv.swa_tokens_per_user())
    assert tokens == demand <= budget_tokens, (tokens, demand, budget_tokens)
    # One 1M request's global group takes half the pages it did.
    assert kps.global_pool_tokens(MAX_1M) == MAX_1M // 2
    # Without the split the same budget holds only two 1M requests.
    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", False)
    tokens = gv.MiMoV2ForCausalLM.get_max_tokens_all_users(max_model_len=MAX_1M, max_num_seqs=4)
    assert tokens == budget_tokens < 4 * MAX_1M and 2 * (MAX_1M + gv.swa_tokens_per_user()) <= tokens


class _LayoutCacheConfig(SimpleNamespace):
    """vllm.config.CacheConfig before the engine core resolves the KV cache layout (as at the plugin's pool sizing)."""

    def get_resolved_kv_cache_layout(self):
        from vllm.v1.kv_cache_layout import KVCacheLayout

        if self.kv_cache_layout is None:
            raise ValueError("KV cache layout has not been resolved yet")
        return KVCacheLayout[self.kv_cache_layout]


def test_plugin_pool_admits_4x1m(monkeypatch):
    """At MIMO_KV_GB_PER_CHIP=6.7 the plugin's own pool sizing (vllm_tt_plugin.worker.get_num_available_blocks_tt,
    with MiMo's budget and specs) gives a lane a pool that vLLM's planner accepts and that holds 4 concurrent 1M
    requests by vLLM's per-request accounting (each group's spec max_memory_usage_bytes: a sliding-window group's
    admission blocks with a 2048-token step budget and async scheduling's two batches in flight, as serve.sh launches
    it)."""
    pytest.importorskip("vllm")
    worker = pytest.importorskip("vllm_tt_plugin.worker")
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs, get_kv_cache_groups
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm_tt_plugin import config as tt_config

    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv

    monkeypatch.setenv("MIMO_KV_GB_PER_CHIP", "6.7")
    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", True)
    monkeypatch.setattr(gv, "PAIR_SPLIT_OPS", True)
    monkeypatch.setattr(gv.MiMoV2ForCausalLM, "_kv_cache_spec_source", None)
    monkeypatch.setattr(worker, "get_model_architecture", lambda model_config: (gv.MiMoV2ForCausalLM, "MiMo"))
    lanes, per_lane = 4, 4
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            model="XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
            max_model_len=MAX_1M,
            original_max_model_len=MAX_1M,
            get_sliding_window=lambda: 128,
            hf_config=mimo_hf_config(),
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=lanes * per_lane, max_num_batched_tokens=2048, disable_hybrid_kv_cache_manager=False
        ),
        # Lane mode: the plugin folds --data-parallel-size 4 into four in-process lanes of one engine.
        parallel_config=SimpleNamespace(data_parallel_size=1, decode_context_parallel_size=1),
        cache_config=_LayoutCacheConfig(
            block_size=SUB, num_gpu_blocks_override=None, kv_cache_layout=None, prefix_cache_retention_interval=None
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        additional_config={tt_config._RESOLVED_LANE_COUNT_KEY: lanes},
        speculative_config=None,
        kv_transfer_config=None,
        max_in_flight_tokens=2 * 2048,
    )
    assert tt_config.get_tt_per_lane_max_num_seqs(cfg) == per_lane
    blocks = worker.get_num_available_blocks_tt(cfg, TP * lanes)
    pool_bytes = blocks * SUB * gv.KV_BYTES_PER_TOKEN

    # vLLM's per-request accounting of the groups it makes from MiMo's specs.
    specs = gv.MiMoV2ForCausalLM.get_kv_cache_spec(cfg)
    cfg.cache_config.kv_cache_layout = KVCacheLayout.LBNHC.name  # the engine core resolves it before planning
    groups = get_kv_cache_groups(cfg, dict(specs))
    per_request = sum(g.kv_cache_spec.max_memory_usage_bytes(cfg) // g.kv_cache_spec.page_size_bytes for g in groups)
    null_block = 1
    assert per_lane * per_request + null_block <= blocks, (per_request, blocks)
    # The engine's planner takes that pool (the plugin hands it over as a byte budget).
    cfg.cache_config.num_gpu_blocks_override = blocks
    available = worker._available_kv_cache_memory_bytes_for_num_blocks(cfg, specs, blocks)
    [planned] = get_kv_cache_configs(cfg, [specs], [available])
    assert planned.num_blocks == blocks and cfg.model_config.max_model_len == MAX_1M
    # The model's share stays within the 6.7 GB (MIMO_KV_GB_PER_CHIP caps it); the plugin adds its output reservation
    # and sliding-window headroom on top.
    model_tokens = gv.MiMoV2ForCausalLM.get_max_tokens_all_users(max_model_len=MAX_1M, max_num_seqs=per_lane)
    assert model_tokens * gv.KV_BYTES_PER_TOKEN <= 6.7e9 and blocks * SUB >= model_tokens
    print(f"pool {blocks} blocks per lane, {pool_bytes / 1e9:.4f} GB per chip; {per_request} blocks per 1M request")


class FakeModel:
    def __init__(self, kinds):
        self.layers = [
            SimpleNamespace(is_swa=swa, layer_idx=i, attn=SimpleNamespace(head_dim=192, v_head_dim=128))
            for i, swa in enumerate(kinds)
        ]
        self.mtp = self.dflash = None
        self.setup = None

    def setup_vllm_kv_cache(self, users, caches, groups, widths, pair_split=False):
        self.setup = (users, len(caches), list(groups), widths, pair_split)


@pytest.mark.parametrize("split", [True, False])
def test_allocate_per_group_block_sizes(monkeypatch, split, expect_error):
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv

    kinds = [False, True, True, False, True, True]  # global, SWA, SWA, ...
    groups = [0, 1, 2, 0, 1, 2]
    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", split)
    monkeypatch.setattr(gv, "PAIR_SPLIT_OPS", True)
    monkeypatch.setattr(gv, "DEVICE_ROPE", False)
    monkeypatch.setattr(gv.MiMoV2ForCausalLM, "_layer_groups", groups)
    monkeypatch.setattr(gv.MiMoV2ForCausalLM, "_buffer", staticmethod(lambda model, buffers, key, n, dim: key))
    models = [FakeModel(kinds) for _ in range(2)]
    gen = SimpleNamespace(models=models, dp=2, users_per_replica=4, cfg=SimpleNamespace(vocab_size=16))
    adapter = gv.MiMoV2ForCausalLM(gen, 32768)
    global_block = PAIR if split else SUB
    specs = [((1000, 1, global_block if not swa else SUB, 192), torch.bfloat16, i // 3) for i, swa in enumerate(kinds)]
    adapter.allocate_kv_cache_per_layer(specs)
    for model in models:
        users, n, got_groups, widths, pair_split = model.setup
        assert (users, n, got_groups, pair_split) == (4, 6, groups, split)
        assert widths == {0: 32768 // global_block, 1: 512, 2: 512}
    assert adapter.pair_split == split and adapter._chunk_align() == (PAIR if split else SUB)
    assert adapter.group_block_sizes == {0: global_block, 1: SUB, 2: SUB}
    # A global layer at the other block size is refused.
    wrong = [((1000, 1, SUB if split else PAIR, 192), torch.bfloat16, 0)] + specs[1:]
    with expect_error(ValueError, "block size"):
        adapter.allocate_kv_cache_per_layer(wrong)


def test_set_vllm_page_tables_trims_padded_rows(monkeypatch, expect_error):
    import ttnn

    copies = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t.clone())
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, dev: copies.append((host, dev)))
    stub = SimpleNamespace(
        vllm_tables={0: torch.zeros(4, 16, dtype=torch.int32), 1: torch.zeros(4, 32, dtype=torch.int32)},
        vllm_table_lengths={(g, s): 0 for g in (0, 1) for s in range(4)},
        page_table_tt={0: "global", 1: "swa"},
        plan=SimpleNamespace(replicate=lambda: None),
    )
    row = torch.zeros(32, dtype=torch.int32)
    row[:10] = torch.arange(1, 11)
    TtMiMoModel.set_vllm_page_tables(stub, {2: {0: row, 1: row}})
    assert stub.vllm_tables[0][2, :10].tolist() == list(range(1, 11)) and stub.vllm_table_lengths[(0, 2)] == 16
    assert stub.vllm_tables[1][2, :10].tolist() == list(range(1, 11)) and stub.vllm_table_lengths[(1, 2)] == 32
    assert sorted(dev for _, dev in copies) == ["global", "swa"]
    bad = row.clone()
    bad[20] = 5  # a block past the global table's width
    with expect_error(ValueError, "past the table"):
        TtMiMoModel.set_vllm_page_tables(stub, {1: {0: bad}})


def mimo_hf_config(**over):
    """The KV fields of MiMo-V2.6-Flash's config.json get_kv_cache_spec reads (9 global, 39 SWA layers)."""
    pattern = [0, 1, 1, 1, 1] + [0, 1, 1, 1, 1, 1] * 7 + [0]
    assert len(pattern) == 48 and pattern.count(0) == 9
    hf = dict(
        num_hidden_layers=48,
        hybrid_layer_pattern=pattern,
        num_key_value_heads=4,
        head_dim=192,
        v_head_dim=128,
        swa_num_key_value_heads=8,
        swa_head_dim=192,
        swa_v_head_dim=128,
        sliding_window_size=128,
        sliding_window=128,
    )
    hf.update(over)
    return SimpleNamespace(**hf)


def vllm_config(block_size=64):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=mimo_hf_config()),
        cache_config=SimpleNamespace(block_size=block_size),
    )


@pytest.fixture
def spec_classes(monkeypatch):
    """vLLM's two spec classes, or stand-ins recording their arguments when vLLM is not installed."""
    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec  # noqa: F401
    except ImportError:
        import sys
        import types

        def spec(name):
            def init(self, **kw):
                self.__dict__.update(kw)
                self.kind = name

            return type(name, (), {"__init__": init})

        module = types.ModuleType("vllm.v1.kv_cache_interface")
        module.FullAttentionSpec, module.SlidingWindowSpec = spec("FullAttentionSpec"), spec("SlidingWindowSpec")
        for name in ("vllm", "vllm.v1"):
            monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", module)


def page_bytes(spec):
    return spec.block_size * spec.num_kv_heads * (spec.head_size + spec.head_size_v) * 2


@pytest.mark.parametrize("split", [True, False])
def test_kv_cache_spec_block_sizes(monkeypatch, spec_classes, split):
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv

    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", split)
    monkeypatch.setattr(gv, "PAIR_SPLIT_OPS", True)
    monkeypatch.setattr(gv.MiMoV2ForCausalLM, "_kv_cache_spec_source", None)
    specs = gv.MiMoV2ForCausalLM.get_kv_cache_spec(vllm_config())
    hf = mimo_hf_config()
    pages = set()
    for i in range(hf.num_hidden_layers):
        spec = specs[f"model.layers.{i}.self_attn"]
        swa = hf.hybrid_layer_pattern[i] == 1
        assert spec.block_size == (SUB if swa or not split else PAIR)
        assert spec.page_size_padded == SUB * 8 * (192 + 128) * 2  # the SWA page
        if swa:  # the model's window plus the positions a moved-back chunk start recomputes (swa_spec_window)
            assert spec.sliding_window == 128 + (PAIR if split else SUB) - 1
        pages.add(page_bytes(spec) if swa or split else spec.page_size_padded)
    # The split's 128-token global page is the SWA page itself: no padding.
    assert len(pages) == 1
    if split:
        assert page_bytes(specs["model.layers.0.self_attn"]) == page_bytes(specs["model.layers.1.self_attn"])


def test_kv_cache_spec_needs_the_lse_ops(monkeypatch, spec_classes, expect_error):
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv

    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", True)
    monkeypatch.setattr(gv, "PAIR_SPLIT_OPS", False)
    monkeypatch.setattr(gv.MiMoV2ForCausalLM, "_kv_cache_spec_source", None)
    with expect_error(ValueError, "MIMO_KV_PAIR_SPLIT=0"):
        gv.MiMoV2ForCausalLM.get_kv_cache_spec(vllm_config())


def test_vllm_groups_the_split_specs(monkeypatch):
    """vLLM's hybrid manager (when installed) keeps the 128-token global group beside the 64-token SWA groups on one
    page size, and schedules on lcm(64, 128) = 128."""
    pytest.importorskip("vllm")
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv

    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", True)
    monkeypatch.setattr(gv, "PAIR_SPLIT_OPS", True)
    monkeypatch.setattr(gv.MiMoV2ForCausalLM, "_kv_cache_spec_source", None)
    specs = gv.MiMoV2ForCausalLM.get_kv_cache_spec(vllm_config())

    from vllm.v1.kv_cache_layout import KVCacheLayout

    class Cache:  # the plugin advertises LBNHC alone (vllm_tt_plugin.worker)
        kv_cache_layout = KVCacheLayout.LBNHC.name

        def get_resolved_kv_cache_layout(self):
            return KVCacheLayout.LBNHC

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=Cache(),
        speculative_config=None,
        model_config=None,
        parallel_config=None,
        kv_transfer_config=None,
    )
    groups = get_kv_cache_groups(config, dict(specs))
    sizes = [g.kv_cache_spec.block_size for g in groups]
    assert sizes == [PAIR] + [SUB] * 5, sizes
    assert len({g.kv_cache_spec.page_size_bytes for g in groups}) == 1
    assert math.lcm(*sizes) == PAIR


# ---------------------------------------------------------------------- a chunk start moved back
def vllm_skipped_blocks(window, processed):
    """The sliding-window blocks vLLM has freed (set to its null block 0) for a request with ``processed`` positions
    computed: SlidingWindowManager.remove_skipped_blocks with the spec's window (vLLM's own function when installed)."""
    try:
        from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
    except ImportError:
        tokens = max(0, processed - window + 1)  # vLLM 0.30's get_num_skipped_tokens without extra_retained_tokens
    else:
        manager = SimpleNamespace(sliding_window=window, extra_retained_tokens=0)
        tokens = SlidingWindowManager.get_num_skipped_tokens(manager, processed)
    return tokens // SUB


class WindowReplica:
    """A replica for prefill_forward on the host: its page tables take vLLM's rows (set_vllm_page_tables), and every
    later chunk reads its SWA windows through the real TtMiMoModel._window_blocks, which asserts that none of them is
    vLLM's null block."""

    PREFILL_BATCH_MAX_PADDED = TtMiMoModel.PREFILL_BATCH_MAX_PADDED
    _host_table = TtMiMoModel._host_table
    _table_key = TtMiMoModel._table_key
    _window_blocks = TtMiMoModel._window_blocks

    def __init__(self, kinds, groups, widths, max_batch=4):
        self.layers = [
            SimpleNamespace(is_swa=swa, attn=SimpleNamespace(sliding_window=128 if swa else None)) for swa in kinds
        ]
        self.layer_groups = list(groups)
        self.block_size = SUB
        self.plan = SimpleNamespace(tp=TP)
        self.vllm_tables = {g: torch.zeros(max_batch, w, dtype=torch.int32) for g, w in widths.items()}
        self.mtp = self.dflash = self.decode_trace = None
        self.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: 0)
        self.chunks = []

    def chunk_traced(self, slot, start_pos, num_tokens, chunk_size=None):
        return False  # no later-chunk traces: every chunk runs eagerly

    def set_vllm_page_tables(self, rows):
        for slot, tables in rows.items():
            for g, row in tables.items():
                host = self.vllm_tables[g]
                n = min(row.numel(), host.shape[1])
                host[slot] = 0
                host[slot, :n] = row[:n]

    def prefill_enqueue(self, chunk, slot, start_pos=0, **kwargs):
        if start_pos > 0:
            for i, layer in enumerate(self.layers):
                if layer.is_swa:
                    self._window_blocks(self._table_key(i), slot, start_pos)
        self.chunks.append((slot, start_pos, chunk.numel()))
        return len(self.chunks)

    def prefill_read(self, handle):
        return [torch.zeros(1, 8)]


def run_moved_back_chunk(gv, split, window, scheduled, end, slot=1):
    """prefill_forward of one chunk [scheduled, end) whose vLLM tables are what vLLM holds at that point: the global
    group's blocks all live, the SWA groups' blocks below the spec ``window``'s retention vLLM's null block 0. Returns
    the start the replica ran (prefill_forward moves an unaligned one back)."""
    kinds = [False, True, True, False, True, True]
    groups = [0, 1, 2, 0, 1, 2]
    block = {0: PAIR if split else SUB, 1: SUB, 2: SUB}
    max_seq = 1 << 18
    widths = {g: max_seq // b for g, b in block.items()}
    replica = WindowReplica(kinds, groups, widths)
    gen = SimpleNamespace(models=[replica], dp=1, users_per_replica=4, cfg=SimpleNamespace(sliding_window=128))
    adapter = gv.MiMoV2ForCausalLM(gen, max_seq)
    adapter.layer_groups, adapter.pair_split, adapter.group_block_sizes = groups, split, block
    freed = vllm_skipped_blocks(window, scheduled)
    rows = {}
    for g, b in block.items():
        row = torch.zeros(widths[g], dtype=torch.int32)
        n = -(-end // b)
        row[:n] = torch.arange(1, n + 1, dtype=torch.int32) + 10000 * g
        if g != 0:
            row[:freed] = 0  # vLLM's null block
        rows[g] = row
    tables = [rows[g].unsqueeze(0) for g in groups]
    tokens = torch.arange(end, dtype=torch.int32).unsqueeze(0)
    adapter.prefill_forward(tokens, [scheduled], [end], None, None, empty_slots=[slot], page_tables_per_layer=tables)
    [(got_slot, start, n)] = replica.chunks
    assert got_slot == slot and start + n == end
    return start


@pytest.mark.parametrize("split", [True, False])
@pytest.mark.parametrize("base", [4096, 128 * 1000, 256])
def test_a_moved_back_chunk_start_reads_live_window_blocks(monkeypatch, spec_classes, split, base):
    """A running request the plugin's scheduler gave a short chunk on leftover budget starts its next chunk off the
    128-token grid; prefill_forward moves the start back up to 127 positions (63 without the pair split) and the
    recomputed chunk reads the SWA window before it. With the spec window get_kv_cache_spec declares, vLLM still holds
    every one of those blocks at every offset; with the model's own window (128) it has freed the first one for the
    larger offsets (the null block, which _window_blocks refuses)."""
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv

    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", split)
    monkeypatch.setattr(gv, "PAIR_SPLIT_OPS", True)
    monkeypatch.setattr(gv, "TIMING", False)
    monkeypatch.setattr(gv.MiMoV2ForCausalLM, "_kv_cache_spec_source", None)
    specs = gv.MiMoV2ForCausalLM.get_kv_cache_spec(vllm_config())
    window = specs["model.layers.1.self_attn"].sliding_window
    align = PAIR if split else SUB
    assert window == gv.swa_spec_window(128) == 128 + align - 1
    end = base + align + 1000  # past the prefill batch's lengths: an eager later chunk
    for offset in range(align):
        assert run_moved_back_chunk(gv, split, window, base + offset, end) == base, offset
    # The model's own window as vLLM's: an offset of 63 and up (0 .. 62 keep the block) frees the window's first block.
    tripped = []
    for offset in range(align):
        try:
            run_moved_back_chunk(gv, split, 128, base + offset, end)
        except AssertionError as e:
            assert "SWA window" in str(e)
            tripped.append(offset)
    assert tripped == list(range(SUB - 1, align)), tripped


def test_a_chain_table_hole_is_logged_at_its_position(monkeypatch):
    """_check_chain_tables reports where a group's uncovered range starts in positions: a pair-split global table has
    one entry per 128 positions."""
    from loguru import logger

    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv

    gen = SimpleNamespace(models=[None], dp=1, users_per_replica=4, cfg=SimpleNamespace(sliding_window=128))
    adapter = gv.MiMoV2ForCausalLM(gen, 1 << 16)
    adapter.layer_groups, adapter.group_block_sizes = [0, 1], {0: PAIR, 1: SUB}
    low, high = 4000, 4100
    global_row = torch.arange(1, 65, dtype=torch.int32)
    global_row[(low - 127) // PAIR] = 0  # a hole in the global window
    swa_row = torch.arange(1, 129, dtype=torch.int32)
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        adapter._check_chain_tables([global_row.unsqueeze(0), swa_row.unsqueeze(0)], 0, low, high)
    finally:
        logger.remove(sink)
    assert adapter.chain_stats["uncovered"] == 1 and len(messages) == 1
    first = (low - 127) // PAIR * PAIR
    assert f"group 0 page table has no block for some of positions {first} .. {high}" in messages[0], messages[0]
