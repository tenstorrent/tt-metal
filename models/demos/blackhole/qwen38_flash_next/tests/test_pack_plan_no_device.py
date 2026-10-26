# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P3d-4 host tests (no device): the pack planner (break-even, mixes, ineligible rows, force, determinism, group limits), the
bucket defaults by lane count, the DRAM estimate with the allocated context, the knob parsing and the adapter's grouped
prefill on a fake lane decoder (row order, per-row bookkeeping, sampling from the pack logits)."""

from __future__ import annotations

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.tests import test_vllm_batch_adapter_no_device as adapter
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_adapter_no_device import (  # noqa: F401  (fixtures)
    FakeDecoder,
    Rig,
    _prompt,
    construction,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_batch_decode as batch
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_pack_plan as plan_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_pack_plan import (
    default_pack_buckets,
    pack_force_enabled,
    pack_ms,
    plan_cost_ms,
    plan_prefill_groups,
    single_ms,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    Qwen38HostSamplingPolicy,
    greedy_batch_sampling_params,
    position_generator,
    sample_from_row,
)

ALL = (128, 256, 512, 1024)
LANE_BUCKETS = (256, 512, 1024)


def _rows(groups):
    return sorted(i for g in groups for i in (g[2] if g[0] == "pack" else g[1]))


# -- the cost model ------------------------------------------------------------------------------------------------------


def test_single_cost_matches_the_measured_requests():
    # P3d-3 measurements: P32 122 ms, P128 216 ms, P1024 766 ms
    assert single_ms(32) == pytest.approx(122.0)
    assert single_ms(128) == pytest.approx(216.0)
    assert single_ms(1024) == pytest.approx(766.0)


def test_pack_cost_is_fixed_plus_landing_per_user():
    assert pack_ms(128, 16) == pytest.approx(657.0 + 16 * 5.8)  # measured 750 ms total
    assert pack_ms(1024, 2) == pytest.approx(661.1 + 2 * 12.1)  # measured 686 ms total


# -- the planner ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("users,packs", [(1, False), (2, False), (3, False), (4, True), (8, True), (16, True)])
def test_break_even_at_100_tokens(users, packs):
    groups = plan_prefill_groups([100] * users, ALL, 16)
    assert _rows(groups) == list(range(users))
    if packs:
        assert groups == [("pack", 128, list(range(users)))]
    else:
        assert groups == [("single", [i]) for i in range(users)]


def test_the_break_even_with_the_lane_default_buckets_is_also_four_users():
    assert plan_prefill_groups([100] * 3, LANE_BUCKETS, 8) == [("single", [i]) for i in range(3)]
    assert plan_prefill_groups([100] * 4, LANE_BUCKETS, 8) == [("pack", 256, [0, 1, 2, 3])]


def test_rows_beyond_the_largest_bucket_are_always_single():
    groups = plan_prefill_groups([1025, 2000, 30000, 100, 100, 100, 100], ALL, 16)
    assert [g for g in groups if g[0] == "single" and g[1][0] in (0, 1, 2)] == [
        ("single", [0]),
        ("single", [1]),
        ("single", [2]),
    ]
    assert ("pack", 128, [3, 4, 5, 6]) in groups
    assert plan_prefill_groups([600] * 8, (256,), 8) == [("single", [i]) for i in range(8)]
    assert plan_prefill_groups([300] * 8, (128, 256), 8) == [("single", [i]) for i in range(8)]  # 300 > 256
    assert plan_prefill_groups([200] * 8, (128, 256), 8) == [("pack", 256, list(range(8)))]


def test_a_pack_holds_at_most_g_users_and_at_most_the_lanes(expect_error):
    groups = plan_prefill_groups([1000] * 8, ALL, 8)
    for g in groups:
        if g[0] == "pack":
            assert len(g[2]) <= 2 and g[1] == 1024
    assert _rows(groups) == list(range(8))
    groups = plan_prefill_groups([500] * 12, ALL, 16)
    assert all(len(g[2]) <= 4 for g in groups if g[0] == "pack")
    groups = plan_prefill_groups([50] * 32, ALL, 32)
    assert all(len(g[2]) <= 16 for g in groups if g[0] == "pack") and _rows(groups) == list(range(32))
    assert [len(g[2]) for g in groups if g[0] == "pack"] == [16, 16]
    with expect_error(ValueError):
        plan_prefill_groups([10] * 9, ALL, 8)


def test_short_and_long_mix_keeps_the_long_rows_out_of_the_short_pack():
    lengths = [100] * 8 + [900, 40, 1000]
    groups = plan_prefill_groups(lengths, ALL, 16)
    assert _rows(groups) == list(range(11))
    cost = plan_cost_ms(groups, lengths)
    packs = [g for g in groups if g[0] == "pack"]
    assert any(g[1] == 128 and set(g[2]) >= set(range(8)) for g in packs)
    # never worse than all single and than any one fixed alternative
    assert cost <= sum(single_ms(n) for n in lengths)
    assert cost <= plan_cost_ms([("pack", 1024, list(range(11)))][:0] or groups, lengths)


def test_two_long_rows_pack_only_when_the_pack_is_cheaper():
    # 2 x 1000: pack 1024 (661.1 + 24.2 = 685.3) against 2 x 766 single; one row: 673 against 766, the pack wins
    # (the model says even a lone row near 1024 tokens is cheaper in the pack: 661 ms fixed against the 648 + 118 of a slab)
    assert plan_prefill_groups([1000, 1000], ALL, 8) == [("pack", 1024, [0, 1])]
    assert plan_prefill_groups([1000], ALL, 8) == [("pack", 1024, [0])]
    assert plan_prefill_groups([600], ALL, 8) == [("pack", 1024, [0])]
    assert plan_prefill_groups([400], ALL, 8) == [("single", [0])]  # single 296 ms against a pack of 657 + 9


def test_the_plan_beats_or_ties_every_all_single_and_all_pack_alternative():
    import random

    rng = random.Random(7)
    for _ in range(40):
        n = rng.randint(1, 16)
        lengths = [rng.choice([5, 31, 64, 100, 128, 200, 256, 500, 700, 1024, 1500]) for _ in range(n)]
        groups = plan_prefill_groups(lengths, ALL, 16)
        assert _rows(groups) == list(range(n))
        assert plan_cost_ms(groups, lengths) <= sum(single_ms(x) for x in lengths) + 1e-9


def test_force_packs_every_eligible_row_even_alone():
    assert plan_prefill_groups([100], ALL, 8, force=True) == [("pack", 128, [0])]
    assert plan_prefill_groups([100, 2000], ALL, 8, force=True) == [("pack", 128, [0]), ("single", [1])]
    groups = plan_prefill_groups([100, 300, 900], ALL, 8, force=True)
    assert [g[0] for g in groups] == ["pack"] * len(groups)
    assert _rows(groups) == [0, 1, 2]
    assert all(g[1] >= max([100, 300, 900][i] for i in g[2]) for g in groups)


def test_the_bucket_is_the_smallest_enabled_one_that_holds_the_longest_row():
    assert plan_prefill_groups([100, 129, 130, 131], ALL, 8, force=True) == [("pack", 256, [0, 1, 2, 3])]
    assert plan_prefill_groups([100] * 4, (512, 1024), 8) == [("pack", 512, [0, 1, 2, 3])]


def test_the_plan_is_deterministic_and_ties_keep_row_order():
    lengths = [100, 700, 100, 33, 100, 100, 1024, 100]
    first = plan_prefill_groups(lengths, ALL, 16)
    for _ in range(5):
        assert plan_prefill_groups(list(lengths), list(ALL), 16) == first
    # equal lengths: members in index order
    assert plan_prefill_groups([100] * 6, ALL, 8) == [("pack", 128, [0, 1, 2, 3, 4, 5])]


def test_ties_go_to_the_single_path(monkeypatch):
    monkeypatch.setattr(plan_module, "PACK_FIXED_MS", {128: 100.0, 256: 100.0, 512: 100.0, 1024: 100.0})
    monkeypatch.setattr(plan_module, "PACK_LAND_MS", {128: 0.0, 256: 0.0, 512: 0.0, 1024: 0.0})
    monkeypatch.setattr(plan_module, "single_ms", lambda length, **kw: 50.0)
    assert plan_prefill_groups([10, 10], ALL, 8) == [("single", [0]), ("single", [1])]  # 100 against 100


def test_the_group_rows_cover_the_input_once():
    lengths = [3, 5000, 128, 129, 1024, 1025, 64]
    groups = plan_prefill_groups(lengths, ALL, 8)
    assert _rows(groups) == list(range(7))
    assert plan_prefill_groups([], ALL, 8) == []


# -- buckets by lane count, DRAM, knobs ----------------------------------------------------------------------------------


def test_default_buckets_depend_on_the_lane_count():
    assert default_pack_buckets(8) == LANE_BUCKETS and default_pack_buckets(2) == LANE_BUCKETS
    assert default_pack_buckets(9) == ALL and default_pack_buckets(32) == ALL and default_pack_buckets(None) == ALL
    assert batch.pack_buckets_from_env({}, lanes=8) == LANE_BUCKETS
    assert batch.pack_buckets_from_env({}, lanes=16) == ALL
    assert batch.pack_buckets_from_env({"QWEN38_PACK_BUCKETS": "128,512"}, lanes=8) == (128, 512)  # the env wins


def test_the_decoder_picks_its_default_buckets_from_its_lane_count(monkeypatch):
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "1")
    monkeypatch.delenv("QWEN38_PACK_BUCKETS", raising=False)
    assert batch.Qwen38PlainLaneDecoder(8).pack_buckets == LANE_BUCKETS
    assert batch.Qwen38PlainLaneDecoder(16).pack_buckets == ALL


def test_pack_dram_scales_with_the_allocated_context():
    mib = 1 << 20
    at_32k = batch.pack_dram_bytes(ALL, 32768)
    assert at_32k == batch.pack_dram_bytes(ALL)
    assert 905 * mib < at_32k < 935 * mib  # measured 918.6 MiB
    extra = batch.pack_dram_bytes(ALL, 65536) - at_32k
    assert extra == 4 * (65536 - 32768) // 4 * 256  # four buckets x the extra compressed rows x 128 bf16
    assert batch.pack_dram_bytes((256, 512), 65536) - batch.pack_dram_bytes((256, 512), 32768) == 2 * 8192 * 256


def test_the_pool_cap_counts_the_pack_with_the_context(monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm

    monkeypatch.delenv("QWEN38_PREFILL_PACK", raising=False)
    monkeypatch.delenv("QWEN38_PACK_BUCKETS", raising=False)
    for max_len, seqs in ((32704, 8), (65472, 8), (32704, 32)):
        context = vllm.resident_allocated_context(max_len)
        monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")
        off = vllm.batch_pool_token_cap(max_len, seqs)
        monkeypatch.setenv("QWEN38_PREFILL_PACK", "1")
        on = vllm.batch_pool_token_cap(max_len, seqs)
        want = batch.pack_dram_bytes(batch.pack_buckets_from_env({}, lanes=seqs), context)
        assert abs((off - on) * vllm.KV_BYTES_PER_TOKEN - want) < 2 * vllm.KV_BYTES_PER_TOKEN


def test_the_force_knob_parses_strictly(expect_error):
    assert pack_force_enabled({}) is False
    assert pack_force_enabled({"QWEN38_PACK_FORCE": "0"}) is False
    assert pack_force_enabled({"QWEN38_PACK_FORCE": "1"}) is True
    for bad in ("2", "true", "", "on"):
        with expect_error(ValueError):
            pack_force_enabled({"QWEN38_PACK_FORCE": bad})


# -- the adapter's grouped prefill on a fake decoder ---------------------------------------------------------------------


class PackDecoder(FakeDecoder):
    """The fake lane decoder plus the pack surface: ``admit_pack`` lands the lanes like the single import and returns logits
    that are a function of the prompt (so the test can tell which user's row each output came from)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pack_buckets = LANE_BUCKETS if self.lanes <= 8 else ALL
        self.packs: list = []
        self.last_pack = {}

    def admit_pack(self, lanes, prompts, rows_per_user, pages=None, users=None):
        self.packs.append((list(lanes), [len(p) for p in prompts], rows_per_user, [list(p) for p in pages]))
        for lane, prompt, rows in zip(lanes, prompts, pages):
            assert self.positions[lane] is None and not self.page_rows[lane]
            self.set_page_row(lane, rows)
            self.positions[lane] = len(prompt)
            self.ls.position.positions[lane] = len(prompt)
            self.contexts[lane] = (prompt[-2] if len(prompt) > 1 else 0, prompt[-1])
        self.page_table_dirty = False
        return torch.stack([self.pack_logits(p) for p in prompts])

    @staticmethod
    def pack_logits(prompt):
        generator = torch.Generator().manual_seed(sum(prompt))
        row = torch.randn(VOCAB_SIZE, generator=generator)
        row[(sum(prompt) * 3) % 4000 + 50] += 15.0
        row[TOKENIZER_SIZE + 3] = 100.0  # a padding column that must be masked
        return row


@pytest.fixture
def pack_rig(monkeypatch):
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "1")
    monkeypatch.delenv("QWEN38_PACK_BUCKETS", raising=False)
    monkeypatch.delenv("QWEN38_PACK_FORCE", raising=False)
    monkeypatch.setattr(adapter.vllm_module, "Qwen38PlainLaneDecoder", PackDecoder)
    rig = Rig.__new__(Rig)
    # Rig.__init__ installs FakeDecoder: build it by hand with the pack decoder
    FakeDecoder.instances.clear()
    from models.demos.blackhole.qwen38_flash_next.tests.test_qwen38_vllm_no_device import ALLOCATED_CONTEXT, FakeChain
    from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM

    rig.chain = FakeChain()
    rig.chain.mesh = "mesh"
    rig.order = []
    adapter._fake_dram(monkeypatch)

    def opener(warm_hook):
        warm_hook(rig.chain)
        return rig.chain

    rig.model = Qwen38ForCausalLM(None, allocated_context=ALLOCATED_CONTEXT, batch_lanes=8, chain_opener=opener)
    rig.lanes, rig.block_size, rig.num_blocks = 8, 64, 2048
    rig.model.allocate_kv_cache((2048, 1, 64, 128), None, 12)
    rig.decoder = FakeDecoder.instances[-1]
    rig.width = -(-ALLOCATED_CONTEXT // 64)
    return rig


def test_the_grouped_prefill_packs_short_rows_keeps_long_ones_single_and_the_row_order(pack_rig, monkeypatch):
    rig = pack_rig
    calls = []
    original = rig.model._prefill_one

    def tracked(ids, *, greedy_token):
        calls.append(len(ids))
        return original(ids, greedy_token=greedy_token)

    monkeypatch.setattr(rig.model, "_prefill_one", tracked)
    prompts = [_prompt(100, 10), _prompt(1500, 20), _prompt(90, 30), _prompt(80, 40), _prompt(70, 50), _prompt(60, 60)]
    blocks = [[1 + i * 30 + j for j in range(-(-len(p) // 64))] for i, p in enumerate(prompts)]
    out, deltas = rig.prefill(prompts, [3, 0, 5, 1, 7, 2], blocks, greedy_batch_sampling_params(6))
    assert calls == [1500]  # only the long row ran alone
    (lanes, lengths, bucket, pages) = rig.decoder.packs[0]
    assert bucket == 256 and sorted(lengths) == [60, 70, 80, 90, 100] and len(rig.decoder.packs) == 1
    assert rig.model.last_prefill_groups[-1] == ("single", [1]) and rig.model.last_prefill_groups[0][0] == "pack"
    assert out.shape == (6, 1) and out.dtype == torch.int32 and deltas.tolist() == [0] * 6
    for i in (0, 2, 3, 4, 5):  # packed rows: the host argmax of the masked logits, in ROW order
        row = PackDecoder.pack_logits(prompts[i]).clone()
        row[TOKENIZER_SIZE:] = float("-inf")
        assert int(out[i, 0]) == int(row.argmax()) and int(out[i, 0]) < TOKENIZER_SIZE
    lane_of = {3: 0, 0: 1, 5: 2, 1: 3, 7: 4, 2: 5}  # slot -> row
    for slot, i in lane_of.items():
        lane = rig.model.lane_of_slot[slot]
        assert rig.decoder.positions[lane] == len(prompts[i]) and rig.decoder.page_rows[lane] == blocks[i]


def test_the_grouped_prefill_without_sampling_params_returns_masked_logits(pack_rig):
    rig = pack_rig
    prompts = [_prompt(100, 10 + i) for i in range(4)]
    blocks = [[1 + i * 3 + j for j in range(2)] for i in range(4)]
    out, _ = rig.prefill(prompts, [0, 1, 2, 3], blocks, None)
    assert out.shape == (4, 1, VOCAB_SIZE) and len(rig.decoder.packs) == 1
    for i in range(4):
        want = PackDecoder.pack_logits(prompts[i])
        assert torch.equal(out[i, 0, :TOKENIZER_SIZE], want[:TOKENIZER_SIZE])
        assert torch.isinf(out[i, 0, TOKENIZER_SIZE:]).all()


def test_a_sampled_pack_row_draws_like_the_single_path_and_adopts_its_policy(pack_rig):
    rig = pack_rig
    prompts = [_prompt(100, 10 + i) for i in range(4)]
    blocks = [[1 + i * 3, 2 + i * 3] for i in range(4)]
    params = greedy_batch_sampling_params(4)
    params.temperature = [0.8] * 4
    params.top_k = [0] * 4
    params.top_p = [1.0] * 4
    params.seed = [11, 12, 13, 14]
    out, _ = rig.prefill(prompts, [0, 1, 2, 3], blocks, params)
    for i in range(4):
        policy = Qwen38HostSamplingPolicy.from_contract(params, i)
        row = PackDecoder.pack_logits(prompts[i]).clone()
        row[TOKENIZER_SIZE:] = float("-inf")
        want = sample_from_row(row, policy, position_generator(policy.seed, len(prompts[i]) - 1))
        assert int(out[i, 0]) == want
        assert rig.model.lane_slots[rig.model.lane_of_slot[i]] is not None


def test_a_live_lane_is_released_before_a_pack_and_a_pack_failure_poisons_the_batch(
    pack_rig, monkeypatch, expect_error
):
    rig = pack_rig
    prompts = [_prompt(100, 10 + i) for i in range(4)]
    blocks = [[1 + i * 3, 2 + i * 3] for i in range(4)]
    rig.prefill(prompts, [0, 1, 2, 3], blocks, greedy_batch_sampling_params(4))
    rig.decoder.log.clear()
    rig.prefill(prompts, [0, 1, 2, 3], blocks, greedy_batch_sampling_params(4))
    assert [e for e in rig.decoder.log if e[0] == "release_lane"] == [("release_lane", lane) for lane in range(4)]

    def boom(*args, **kwargs):
        raise RuntimeError("landing failed")

    monkeypatch.setattr(rig.decoder, "admit_pack", boom)
    with expect_error(RuntimeError):
        rig.prefill(prompts, [4, 5, 6, 7], blocks, greedy_batch_sampling_params(4))
    assert rig.model.batch_failed is True


def test_force_packs_a_single_row(pack_rig, monkeypatch):
    monkeypatch.setenv("QWEN38_PACK_FORCE", "1")
    rig = pack_rig
    out, _ = rig.prefill([_prompt(100, 10)], [2], [[1, 2]], greedy_batch_sampling_params(1))
    assert len(rig.decoder.packs) == 1 and rig.decoder.packs[0][1] == [100] and out.shape == (1, 1)


def test_the_default_packs_with_eight_lanes(pack_rig, monkeypatch):
    monkeypatch.delenv("QWEN38_PREFILL_PACK", raising=False)  # P3e: unset is on with at least 8 lanes
    rig = pack_rig
    blocks = [[1 + i * 30, 2 + i * 30] for i in range(4)]
    rig.prefill([_prompt(100, 10 + 10 * i) for i in range(4)], [0, 1, 2, 3], blocks, greedy_batch_sampling_params(4))
    assert len(rig.decoder.packs) == 1 and rig.decoder.packs[0][1] == [100] * 4  # 4 users of 100 tokens: one pack


def test_the_default_never_plans_with_fewer_than_eight_lanes(monkeypatch):
    monkeypatch.delenv("QWEN38_PREFILL_PACK", raising=False)
    rig = Rig(monkeypatch, lanes=4)
    monkeypatch.setattr(
        adapter.vllm_module, "plan_prefill_groups", lambda *a, **k: pytest.fail("the planner ran with 4 lanes")
    )
    rig.prefill([_prompt(40, 10), _prompt(70, 20)], [0, 1], [[5], [9, 3]], greedy_batch_sampling_params(2))


def test_the_knob_off_path_never_plans(monkeypatch):
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")
    rig = Rig(monkeypatch)
    monkeypatch.setattr(
        adapter.vllm_module, "plan_prefill_groups", lambda *a, **k: pytest.fail("the planner ran with the knob off")
    )
    rig.prefill([_prompt(40, 10), _prompt(70, 20)], [0, 1], [[5], [9, 3]], greedy_batch_sampling_params(2))
