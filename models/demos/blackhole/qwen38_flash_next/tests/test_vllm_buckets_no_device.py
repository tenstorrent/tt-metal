# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter with decode-width buckets (W4) on a fake lane decoder (no device).

The ``QWEN38_DECODE_BUCKETS`` knob; the lowest-free-lane admission (``_take_lane``) with the slot -> lane swap; a decode
step whose bucket R is below the lane count reads only rows below R on every route (greedy ids, candidates, normalizer,
row read, full gather, logprobs); warmup replays every bucket's sampling traces through ``warm_bucket_reads``; the DRAM term
for the extra buckets.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.tests import test_vllm_batch_adapter_no_device as adapter_tests
from models.demos.blackhole.qwen38_flash_next.tests.test_batch_candidates_no_device import (
    CandidateDecoder,
    candidate_row_of,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_qwen38_vllm_no_device import construction  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_adapter_no_device import (  # noqa: F401  (fixture)
    FakeDecoder,
    Rig,
    rig,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import Qwen38PlainLaneDecoder, row_read_buckets
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    BUCKET_FIXED_BYTES_32K,
    BUCKET_PER_LANE_BYTES_32K,
    DECODE_BUCKETS_VARIABLE,
    KV_BYTES_PER_TOKEN,
    Qwen38ForCausalLM,
    batch_pool_token_cap,
    bucket_dram_bytes,
    decode_buckets_from_env,
    greedy_batch_sampling_params,
    position_generator,
    sample_from_row,
)

LANES = 8
BUCKETS = (1, 2, 4, 8)
BUCKETS_32 = (1, 2, 4, 8, 16, 32)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")
    monkeypatch.delenv(DECODE_BUCKETS_VARIABLE, raising=False)


class BucketDecoder(CandidateDecoder):
    """The candidate fake with the bucket surface: a step replays the smallest bucket holding the highest live lane, advances
    only lanes below R and returns -1 past R; every read returns R rows and a read of a row at R or above fails."""

    has_row_read = True

    def __init__(self, lanes, *args, buckets=None, **kwargs) -> None:
        super().__init__(lanes, *args, **kwargs)
        self.buckets = tuple(buckets or (lanes,))
        self.active_r = lanes
        self.bucket_steps = {r: 0 for r in self.buckets}
        self.row_read_buckets = row_read_buckets(lanes)
        self.row_reads: list[list[int]] = []
        self.warm_calls = 0

    @property
    def active(self):
        return SimpleNamespace(lanes=self.active_r)

    def step(self, tokens) -> list[int]:
        live = [u for u in range(self.lanes) if self.positions[u] is not None]
        need = max(live) + 1 if live else 1
        r = next(b for b in self.buckets if b >= need)
        self.active_r = r
        self.bucket_steps[r] += 1
        self.row_read_buckets = row_read_buckets(r)
        positions = list(self.ls.position.positions)
        self.steps.append(
            {"tokens": list(tokens), "positions": positions, "r": r, "rows": [list(row) for row in self.page_rows]}
        )
        ids = [self.next_id(t, p) for t, p in zip(tokens[:r], positions[:r])]
        self.last_ids = ids + [0] * (self.lanes - r)
        self.ls.position.positions = [p + 1 if u < r else p for u, p in enumerate(positions)]
        return ids + [-1] * (self.lanes - r)

    def _rows(self, lanes) -> list[int]:
        lanes = list(lanes)
        assert all(0 <= lane < self.active_r for lane in lanes), f"row read past R={self.active_r}: {lanes}"
        return lanes

    def read_full_logits(self) -> torch.Tensor:
        self.full_reads += 1
        return torch.stack([self.logits_for(lane, len(self.steps)) for lane in range(self.active_r)])

    def read_candidates(self) -> torch.Tensor:
        self.candidate_reads += 1
        return torch.stack(
            [candidate_row_of(self.logits_for(u, len(self.steps))) for u in range(self.active_r)]
        ).clone()

    def read_candidates_normalized(self, inv_temperatures):
        assert len(inv_temperatures) == self.lanes  # the decoder takes the first R
        self.normalizer_reads.append([float(value) for value in inv_temperatures])
        rows = self.read_candidates()
        shard = VOCAB_SIZE // 4
        lse = torch.zeros(self.active_r, 4)
        for lane in range(self.active_r):
            scaled = self.logits_for(lane, len(self.steps)) * torch.tensor(inv_temperatures[lane], dtype=torch.float32)
            scaled[TOKENIZER_SIZE:] = float("-inf")
            lse[lane] = torch.stack([torch.logsumexp(scaled[d * shard : (d + 1) * shard], 0) for d in range(4)])
        return rows, lse

    def read_logits_rows(self, lanes) -> torch.Tensor:
        lanes = self._rows(lanes)
        self.row_reads.append(lanes)
        return torch.stack([self.logits_for(lane, len(self.steps)) for lane in lanes]).clone()

    def compaction_moves(self) -> list[tuple[int, int]]:
        return Qwen38PlainLaneDecoder.compaction_moves(self)  # the real selection over the fake's mirrors

    def move_lane(self, src, dst) -> None:
        """The host effects of the real ``move_lane`` (mirrors, page row, position row)."""

        self.log.append(("move", src, dst))
        assert self.positions[src] is not None and self.positions[dst] is None and not self.page_rows[dst]
        self.positions[dst], self.positions[src] = self.positions[src], None
        self.contexts[dst], self.contexts[src] = self.contexts[src], None
        self.page_rows[dst], self.page_rows[src] = self.page_rows[src], []
        self.page_table_dirty = True
        position = self.ls.position
        position.positions[dst], position.positions[src] = position.positions[src], 0
        self.last_move = {"src": src, "dst": dst, "ms": 1.5}

    def warm_bucket_reads(self) -> None:
        self.warm_calls += 1
        self.active_r = self.lanes


@pytest.fixture
def brig(monkeypatch) -> Rig:
    monkeypatch.setattr(adapter_tests, "FakeDecoder", BucketDecoder)
    rig = Rig(monkeypatch)
    rig.model.decode_buckets = BUCKETS  # 8 lanes: the adapter's default is off, the fake takes any bucket list
    rig.decoder.buckets = BUCKETS
    rig.model.decode_compact = False  # the compaction tests turn it on (crig)
    rig.decoder.bucket_steps = {r: 0 for r in BUCKETS}
    return rig


# -- the knob ---------------------------------------------------------------------------------------------------------


def test_the_knob_defaults_to_the_six_buckets_with_32_lanes_and_one_trace_otherwise() -> None:
    assert decode_buckets_from_env(32, {}) == (1, 2, 4, 8, 16, 32)
    assert decode_buckets_from_env(32, {DECODE_BUCKETS_VARIABLE: ""}) == (1, 2, 4, 8, 16, 32)
    assert decode_buckets_from_env(8, {}) == (8,)
    assert decode_buckets_from_env(2, {DECODE_BUCKETS_VARIABLE: " "}) == (2,)


@pytest.mark.parametrize("value", ["off", "0", "OFF"])
def test_the_knob_off_is_the_full_width_only(value) -> None:
    assert decode_buckets_from_env(32, {DECODE_BUCKETS_VARIABLE: value}) == (32,)
    assert decode_buckets_from_env(8, {DECODE_BUCKETS_VARIABLE: value}) == (8,)


def test_the_knob_takes_a_list_that_ends_in_the_lanes() -> None:
    assert decode_buckets_from_env(32, {DECODE_BUCKETS_VARIABLE: "4,16,32"}) == (4, 16, 32)
    assert decode_buckets_from_env(32, {DECODE_BUCKETS_VARIABLE: "32"}) == (32,)
    assert decode_buckets_from_env(8, {DECODE_BUCKETS_VARIABLE: "8"}) == (8,)


@pytest.mark.parametrize("value", ["1,2,4", "4,2,32", "2,2,32", "0,32", "1,64", "a,32", "1,,32", "-1,32", "1;32"])
def test_a_bad_knob_value_is_refused_naming_the_variable(value, expect_error) -> None:
    with expect_error(ValueError, DECODE_BUCKETS_VARIABLE):
        decode_buckets_from_env(32, {DECODE_BUCKETS_VARIABLE: value})


def test_more_than_one_bucket_needs_32_lanes(expect_error) -> None:
    with expect_error(ValueError, DECODE_BUCKETS_VARIABLE):
        decode_buckets_from_env(8, {DECODE_BUCKETS_VARIABLE: "4,8"})


def test_the_adapter_reads_the_knob_and_hands_the_buckets_to_the_decoder(monkeypatch, expect_error) -> None:
    monkeypatch.setattr(adapter_tests, "FakeDecoder", BucketDecoder)
    seen = {}
    real = BucketDecoder.__init__

    def spy(self, lanes, *args, **kwargs):
        seen["buckets"] = kwargs.get("buckets")
        real(self, lanes, *args, **kwargs)

    monkeypatch.setattr(BucketDecoder, "__init__", spy)
    rig = Rig(monkeypatch, lanes=32)
    assert rig.model.decode_buckets == (1, 2, 4, 8, 16, 32) and seen["buckets"] == (1, 2, 4, 8, 16, 32)
    monkeypatch.setenv(DECODE_BUCKETS_VARIABLE, "off")
    rig = Rig(monkeypatch, lanes=32)
    assert rig.model.decode_buckets == (32,) and seen["buckets"] == (32,)
    monkeypatch.setenv(DECODE_BUCKETS_VARIABLE, "2,16")
    with expect_error(ValueError, DECODE_BUCKETS_VARIABLE):
        Rig(monkeypatch, lanes=32)


# -- _take_lane -------------------------------------------------------------------------------------------------------


def test_take_lane_with_the_buckets_off_is_the_slots_own_lane(rig: Rig) -> None:
    assert rig.model.decode_buckets == (LANES,)
    claimed: set[int] = set()
    assert rig.model._take_lane(rig.decoder, 5, claimed) == 5 and claimed == {5}
    assert rig.model.lane_of_slot == list(range(LANES))


def test_take_lane_with_the_buckets_off_releases_a_live_lane_first(rig: Rig) -> None:
    rig.decoder.positions[5] = 3
    assert rig.model._take_lane(rig.decoder, 5, set()) == 5
    assert ("release_lane", 5) in rig.decoder.log and rig.decoder.positions[5] is None


def test_take_lane_takes_the_lowest_free_lane_and_swaps_the_map(brig: Rig) -> None:
    model, decoder = brig.model, brig.decoder
    claimed: set[int] = set()
    assert model._take_lane(decoder, 7, claimed) == 0
    assert model.lane_of_slot[7] == 0 and model.lane_of_slot[0] == 7 and claimed == {0}
    assert model._take_lane(decoder, 3, claimed) == 1
    assert model.lane_of_slot[3] == 1 and model.lane_of_slot[1] == 3 and claimed == {0, 1}
    assert sorted(model.lane_of_slot) == list(range(LANES))  # still a permutation
    assert model._take_lane(decoder, 1, claimed) == 2  # slot 1's lane is 3 now; the lowest unclaimed free lane is 2
    assert sorted(model.lane_of_slot) == list(range(LANES)) and model.lane_of_slot[1] == 2


def test_take_lane_skips_live_and_claimed_lanes(brig: Rig) -> None:
    model, decoder = brig.model, brig.decoder
    decoder.positions[0] = decoder.positions[2] = 10
    assert model._take_lane(decoder, 6, set()) == 1
    assert model._take_lane(decoder, 6, {1}) == 3
    assert model.lane_of_slot[6] == 3


def test_take_lane_keeps_a_free_slot_lane_that_is_already_the_lowest(brig: Rig) -> None:
    assert brig.model._take_lane(brig.decoder, 0, set()) == 0
    assert brig.model.lane_of_slot == list(range(LANES))


def test_take_lane_releases_a_live_lane_in_the_slot_before_it_looks(brig: Rig) -> None:
    model, decoder = brig.model, brig.decoder
    decoder.positions[0] = 9
    decoder.page_rows[0] = [4]
    assert model._take_lane(decoder, 0, set()) == 0  # released, then the lowest free lane is lane 0 again
    assert ("release_lane", 0) in decoder.log and decoder.positions[0] is None and decoder.page_rows[0] == []


def test_two_new_requests_in_one_batch_get_lanes_0_and_1_even_from_slots_7_and_3(brig: Rig) -> None:
    prompts = [adapter_tests._prompt(20, 100), adapter_tests._prompt(30, 200)]
    brig.prefill(prompts, [7, 3], [[1], [2]], greedy_batch_sampling_params(2))
    imports = [entry for entry in brig.decoder.log if isinstance(entry, tuple) and entry[0] == "import"]
    assert [(i[1], i[2]) for i in imports] == [(0, 20), (1, 30)]
    assert brig.model.lane_of_slot[7] == 0 and brig.model.lane_of_slot[3] == 1
    assert brig.decoder.positions[:3] == [20, 30, None]


def test_a_later_request_takes_the_lowest_lane_freed_by_a_release(brig: Rig) -> None:
    prompts = [adapter_tests._prompt(20, 100), adapter_tests._prompt(30, 200), adapter_tests._prompt(25, 300)]
    brig.prefill(prompts, [0, 1, 2], [[1], [2], [3]], greedy_batch_sampling_params(3))
    brig.model.release_request(0)  # lane 0 is free; lanes 1 and 2 stay
    brig.prefill([adapter_tests._prompt(22, 400)], [5], [[4]], greedy_batch_sampling_params(1))
    assert brig.model.lane_of_slot[5] == 0 and brig.decoder.positions[:4] == [22, 30, 25, None]


def test_the_grouped_prefill_shares_one_claimed_set_with_its_single_rows(brig: Rig, monkeypatch) -> None:
    decoder = brig.decoder
    decoder.pack_buckets = (256,)
    admitted = []

    def admit_pack(lanes, prompts, bucket, pages=None):
        admitted.append(list(lanes))
        for lane, prompt, page in zip(lanes, prompts, pages):
            decoder.positions[lane] = len(prompt)
            decoder.page_rows[lane] = list(page)
            decoder.ls.position.positions[lane] = len(prompt)
        return torch.zeros(len(lanes), VOCAB_SIZE)

    decoder.admit_pack = admit_pack
    decoder.last_pack = None
    monkeypatch.setattr(
        vllm_module, "plan_prefill_groups", lambda *a, **k: [("pack", 256, [0, 1]), ("single", [2]), ("pack", 256, [3])]
    )
    prompts = [adapter_tests._prompt(20 + i, 100 * (i + 1)) for i in range(4)]
    policies = [None] * 4
    brig.model._prefill_batch_grouped(prompts, [[1], [2], [3], [4]], policies, [7, 3, 5, 1])
    assert admitted == [[0, 1], [3]]  # the single row took lane 2 between the packs
    assert decoder.positions[:5] == [20, 21, 22, 23, None]
    assert [brig.model.lane_of_slot[s] for s in (7, 3, 5, 1)] == [0, 1, 2, 3]
    assert sorted(brig.model.lane_of_slot) == list(range(LANES))


# -- a step below the full width reads only rows below R --------------------------------------------------------------


def _three_in_odd_slots(rig: Rig):
    """Requests in slots 7, 3 and 5 (lanes 0, 1, 2 after the lowest-free-lane rule; R = 4)."""

    prompts = {7: adapter_tests._prompt(24, 700), 3: adapter_tests._prompt(28, 300), 5: adapter_tests._prompt(32, 500)}
    blocks = {7: [1], 3: [2], 5: [3]}
    rig.prefill(
        [prompts[s] for s in (7, 3, 5)], [7, 3, 5], [blocks[s] for s in (7, 3, 5)], greedy_batch_sampling_params(3)
    )
    assert [rig.model.lane_of_slot[s] for s in (7, 3, 5)] == [0, 1, 2]
    return prompts, blocks


def _rows(prompts, blocks):
    rows = [None] * LANES
    for slot, prompt in prompts.items():
        rows[slot] = (1000 + slot, len(prompt), blocks[slot])
    return rows


def test_a_greedy_step_runs_the_small_bucket_and_never_reads_logits(brig: Rig) -> None:
    prompts, blocks = _three_in_odd_slots(brig)
    out = brig.decode(_rows(prompts, blocks))
    decoder = brig.decoder
    assert decoder.steps[-1]["r"] == 4 and decoder.bucket_steps[4] == 1
    for slot, lane in ((7, 0), (3, 1), (5, 2)):
        assert out[slot, 0] == FakeDecoder.next_id(1000 + slot, len(prompts[slot]))
    assert decoder.full_reads == 0 and decoder.candidate_reads == 0


def test_a_host_logits_step_reads_the_full_gather_of_r_rows(brig: Rig) -> None:
    prompts, blocks = _three_in_odd_slots(brig)
    out = brig.decode(_rows(prompts, blocks), sampling_params=None)
    decoder = brig.decoder
    assert out.shape == (LANES, 1, VOCAB_SIZE) and decoder.full_reads == 1
    for slot, lane in ((7, 0), (3, 1), (5, 2)):
        expected = decoder.logits_for(lane, len(decoder.steps)).clone()
        expected[TOKENIZER_SIZE:] = float("-inf")
        assert torch.equal(out[slot, 0], expected)
    assert float(out[0, 0].abs().sum()) == 0.0  # padding rows stay zero


def _sampled_params():
    """Slot 7 top-k off (normalizer route), slot 3 top_k 20 (candidates), slot 5 top_k 50 (a fallback row), logprobs on 3, 5."""

    params = adapter_tests._sampling(
        LANES,
        temperature=[0.0] * 3 + [1.0] + [0.0, 1.0, 0.0, 1.0],
        top_k=[1] * 3 + [20] + [1, 50, 1, VOCAB_SIZE],
        top_p=[0.0] * 3 + [0.95] + [0.0, 0.9, 0.0, 0.95],
        seed=[None] * 3 + [11] + [None, 33, None, 22],
    )
    params.enable_log_probs = [slot in (3, 5) for slot in range(LANES)]
    return params


def _expected_full(decoder, lane: int, position: int, params, slot: int) -> int:
    policy = vllm_module.Qwen38HostSamplingPolicy.from_contract(params, slot)
    row = decoder.logits_for(lane, len(decoder.steps)).clone()
    row[TOKENIZER_SIZE:] = float("-inf")
    return sample_from_row(row, policy, position_generator(policy.seed, position))


def test_a_sampled_step_reads_candidates_normalizer_rows_and_logprobs_below_r(brig: Rig) -> None:
    prompts, blocks = _three_in_odd_slots(brig)
    params = _sampled_params()
    out, log_probs = brig.decode(
        _rows(prompts, blocks),
        sampling_params=params,
        prompt_tokens=torch.full((LANES, 4), -1, dtype=torch.int32),
        output_tokens=torch.full((LANES, 4), -1, dtype=torch.int32),
    )
    decoder = brig.decoder
    assert decoder.steps[-1]["r"] == 4
    assert len(decoder.normalizer_reads) == 1 and len(decoder.normalizer_reads[0]) == LANES  # slot 7's lane 0 is 1 / T
    assert decoder.normalizer_reads[0][0] == 1.0
    assert len(decoder.row_reads) == 1 and {1, 2} <= set(decoder.row_reads[0]) <= {
        0,
        1,
        2,
    }  # the fallback + logprobs lanes
    assert decoder.full_reads == 0
    assert out[3, 0] == _expected_full(decoder, 1, len(prompts[3]), params, 3)
    assert out[5, 0] == _expected_full(decoder, 2, len(prompts[5]), params, 5)
    assert 0 <= int(out[7, 0]) < TOKENIZER_SIZE
    for slot, lane in ((3, 1), (5, 2)):
        row = decoder.logits_for(lane, len(decoder.steps)).clone()
        row[TOKENIZER_SIZE:] = float("-inf")
        assert float(log_probs[slot]) == pytest.approx(vllm_module.raw_logprob(row, int(out[slot, 0])), abs=1e-5)
    assert float(log_probs[7]) == 0.0


def test_a_logprobs_only_step_reads_just_the_logprobs_lanes_below_r(brig: Rig) -> None:
    prompts, blocks = _three_in_odd_slots(brig)
    params = adapter_tests._sampling(LANES)
    params.enable_log_probs = [slot == 5 for slot in range(LANES)]
    out, log_probs = brig.decode(_rows(prompts, blocks), sampling_params=params)
    assert brig.decoder.row_reads == [[2]] and brig.decoder.full_reads == 0
    assert out[5, 0] == FakeDecoder.next_id(1005, len(prompts[5])) and float(log_probs[5]) != 0.0


def test_the_full_gather_fallback_serves_rows_below_r(brig: Rig, monkeypatch) -> None:
    monkeypatch.setattr(brig.model, "batch_row_read", False)
    prompts, blocks = _three_in_odd_slots(brig)
    params = _sampled_params()
    out, _ = brig.decode(
        _rows(prompts, blocks),
        sampling_params=params,
        prompt_tokens=torch.full((LANES, 4), -1, dtype=torch.int32),
        output_tokens=torch.full((LANES, 4), -1, dtype=torch.int32),
    )
    assert brig.decoder.full_reads == 1 and brig.decoder.row_reads == []
    assert out[5, 0] == _expected_full(brig.decoder, 2, len(prompts[5]), params, 5)


def test_the_step_stats_carry_r_and_the_decoder_counts_the_buckets(brig: Rig) -> None:
    prompts, blocks = _three_in_odd_slots(brig)
    brig.decode(_rows(prompts, blocks))
    assert brig.model.decode_stats[-1]["bucket"] == 4
    assert brig.decoder.bucket_steps == {1: 0, 2: 0, 4: 1, 8: 0}


def test_the_periodic_log_line_names_r_and_the_steps_per_bucket(brig: Rig, monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(vllm_module.logger, "info", lambda message, *a, **k: lines.append(str(message)))
    monkeypatch.setattr(vllm_module, "DECODE_LOG_INTERVAL", 1)
    prompts, blocks = _three_in_odd_slots(brig)
    brig.decode(_rows(prompts, blocks))
    line = next(line for line in lines if line.startswith("batch decode:"))
    assert "bucket R 4.0" in line and "steps per bucket {1: 0, 2: 0, 4: 1, 8: 0}" in line


# -- warmup -----------------------------------------------------------------------------------------------------------


def _warm(rig: Rig) -> None:
    rig.model.warmup_model_prefill(enable_trace=True, kv_cache=None, can_sample_on_device=False)
    rig.model.warmup_model_decode(
        enable_trace=True, kv_cache=None, max_batch_size=rig.lanes, num_blocks=rig.width, can_sample_on_device=True
    )


def test_warmup_with_buckets_on_replays_every_bucket_through_warm_bucket_reads(monkeypatch) -> None:
    monkeypatch.setattr(adapter_tests, "FakeDecoder", BucketDecoder)
    rig = Rig(monkeypatch, lanes=32)
    _warm(rig)
    decoder = rig.decoder
    assert decoder.warm_calls == 1
    assert decoder.candidate_reads == 0 and decoder.normalizer_reads == [] and decoder.row_reads == []
    assert decoder.active_r == 32 and rig.model.lane_of_slot == list(range(32))


def test_warmup_with_buckets_off_keeps_the_per_trace_replays(monkeypatch) -> None:
    monkeypatch.setenv(DECODE_BUCKETS_VARIABLE, "off")
    monkeypatch.setattr(adapter_tests, "FakeDecoder", BucketDecoder)
    rig = Rig(monkeypatch, lanes=32)
    _warm(rig)
    decoder = rig.decoder
    assert decoder.warm_calls == 0 and decoder.candidate_reads >= 1
    assert decoder.normalizer_reads == [[1.0] * 32]
    assert decoder.row_reads == [list(range(size)) for size in row_read_buckets(32)]


# -- the DRAM term ----------------------------------------------------------------------------------------------------


def test_the_pool_cap_loses_the_extra_buckets_bytes() -> None:
    base_cap = batch_pool_token_cap(32704, 32)
    assert batch_pool_token_cap(32704, 32, ()) == base_cap
    five = batch_pool_token_cap(32704, 32, BUCKETS_32)
    assert batch_pool_token_cap(32704, 32, (32,)) == base_cap  # the full width has no extra term
    assert base_cap - five == pytest.approx(bucket_dram_bytes(BUCKETS_32, 32) / KV_BYTES_PER_TOKEN, abs=1)


def test_the_plugins_token_budget_counts_the_default_buckets(monkeypatch) -> None:
    monkeypatch.delenv("QWEN38_MAX_TOKENS_ALL_USERS", raising=False)
    kwargs = dict(max_model_len=32704, max_num_seqs=32)
    assert Qwen38ForCausalLM.get_max_tokens_all_users(**kwargs) == batch_pool_token_cap(32704, 32, BUCKETS_32)
    monkeypatch.setenv(DECODE_BUCKETS_VARIABLE, "off")
    assert Qwen38ForCausalLM.get_max_tokens_all_users(**kwargs) == batch_pool_token_cap(32704, 32, (32,))


def test_the_pool_admission_counts_the_extra_buckets(monkeypatch, expect_error) -> None:
    monkeypatch.setattr(adapter_tests, "FakeDecoder", BucketDecoder)
    monkeypatch.setattr(
        vllm_module, "BUCKET_FIXED_BYTES_32K", 40 * 2**30
    )  # five extra buckets: 200 GiB beside 128 GiB
    with expect_error(ValueError, "QWEN38_MAX_TOKENS_ALL_USERS"):
        Rig(monkeypatch, lanes=32)
    monkeypatch.setenv(DECODE_BUCKETS_VARIABLE, "off")
    Rig(monkeypatch, lanes=32)  # the full width alone has no extra term


def test_bucket_dram_bytes_values() -> None:
    assert BUCKET_FIXED_BYTES_32K == 16_000_000 and BUCKET_PER_LANE_BYTES_32K == 2_600_000
    assert bucket_dram_bytes((32,), 32) == 0 and bucket_dram_bytes((), 32) == 0
    assert bucket_dram_bytes((1, 2, 4, 8, 16, 32), 32) == 5 * 16_000_000 + 31 * 2_600_000
    assert bucket_dram_bytes((4, 16, 32), 32) == 2 * 16_000_000 + 20 * 2_600_000
    assert bucket_dram_bytes((8,), 8) == 0


# -- W5: compaction in the decode step --------------------------------------------------------------------------------


@pytest.fixture
def crig(brig: Rig) -> Rig:
    """``brig`` with compaction on."""

    brig.model.decode_compact = True
    return brig


def _three_then_release_middle(rig: Rig):
    """Slots 0, 1, 2 in lanes 0, 1, 2; slot 1 leaves: live lanes {0, 2} (R = 4 without compaction)."""

    prompts = {0: adapter_tests._prompt(24, 700), 1: adapter_tests._prompt(28, 300), 2: adapter_tests._prompt(32, 500)}
    blocks = {0: [1], 1: [2], 2: [3]}
    rig.prefill(
        [prompts[s] for s in (0, 1, 2)], [0, 1, 2], [blocks[s] for s in (0, 1, 2)], greedy_batch_sampling_params(3)
    )
    rig.model.release_request(1)
    return prompts, blocks


def test_the_compact_knob() -> None:
    from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
        DECODE_COMPACT_VARIABLE,
        decode_compact_enabled,
    )

    assert DECODE_COMPACT_VARIABLE == "QWEN38_DECODE_COMPACT"
    assert decode_compact_enabled({}) and decode_compact_enabled({DECODE_COMPACT_VARIABLE: "1"})
    assert not decode_compact_enabled({DECODE_COMPACT_VARIABLE: "0"})
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        ValueError, match=DECODE_COMPACT_VARIABLE
    ):
        decode_compact_enabled({DECODE_COMPACT_VARIABLE: "yes"})


def test_the_adapter_reads_the_compact_knob_only_with_buckets_on(monkeypatch) -> None:
    monkeypatch.setattr(adapter_tests, "FakeDecoder", BucketDecoder)
    assert Rig(monkeypatch, lanes=32).model.decode_compact is True
    monkeypatch.setenv("QWEN38_DECODE_COMPACT", "0")
    assert Rig(monkeypatch, lanes=32).model.decode_compact is False
    monkeypatch.delenv("QWEN38_DECODE_COMPACT")
    monkeypatch.setenv(DECODE_BUCKETS_VARIABLE, "off")
    assert Rig(monkeypatch, lanes=32).model.decode_compact is False


def test_a_step_moves_a_high_lane_down_before_the_remap_and_the_slot_follows(crig: Rig) -> None:
    prompts, blocks = _three_then_release_middle(crig)
    model, decoder = crig.model, crig.decoder
    marker = model.lane_slots[2]
    assert [model.lane_of_slot[s] for s in (0, 2)] == [0, 2]
    # a remap that swaps rows 0 and 2: row 0 is slot 2's request, row 2 is slot 0's; the map it resolves against is the moved one
    remap = list(range(LANES))
    remap[0], remap[2] = 2, 0
    rows = [(1002, len(prompts[2]), blocks[2]), None, (1000, len(prompts[0]), blocks[0])]
    out = crig.decode(rows, slot_remap=remap)
    assert ("move", 2, 1) in decoder.log and decoder.log.count(("move", 2, 1)) == 1
    assert model.lane_moves == 1
    assert decoder.steps[-1]["r"] == 2 and decoder.bucket_steps[2] == 1 and decoder.bucket_steps[4] == 0
    assert out[0, 0] == FakeDecoder.next_id(1002, len(prompts[2])) and out[2, 0] == FakeDecoder.next_id(
        1000, len(prompts[0])
    )
    # slot 2's request lives in lane 1 and slot 1 (free) took lane 2: still a permutation, then the remap applied on top
    assert sorted(model.lane_of_slot) == list(range(LANES))
    assert model.lane_of_slot[0] == 1 and model.lane_of_slot[2] == 0 and model.lane_of_slot[1] == 2
    assert model.lane_slots[1] is marker and model.lane_slots[2] is not marker
    assert decoder.positions[:3] == [len(prompts[0]), len(prompts[2]), None]
    assert decoder.steps[-1]["rows"][1] == blocks[2] and decoder.steps[-1]["rows"][2] == []


def test_the_step_after_a_move_uses_the_moved_lane_for_the_slots_row(crig: Rig) -> None:
    prompts, blocks = _three_then_release_middle(crig)
    model, decoder = crig.model, crig.decoder
    rows = [(1000, len(prompts[0]), blocks[0]), None, (1002, len(prompts[2]), blocks[2])]
    crig.decode(rows)
    assert model.lane_of_slot[2] == 1 and decoder.positions[:3] == [len(prompts[0]), len(prompts[2]), None]
    assert decoder.steps[-1]["tokens"][1] == 1002 and decoder.steps[-1]["rows"][1] == [
        3
    ]  # lane 1 carries slot 2's page
    assert decoder.steps[-1]["rows"][2] == []
    rows = [(5000, len(prompts[0]) + 1, blocks[0]), None, (5002, len(prompts[2]) + 1, blocks[2])]
    out = crig.decode(rows)  # no second move, the same lanes
    assert model.lane_moves == 1 and decoder.log.count(("move", 2, 1)) == 1
    assert decoder.steps[-1]["tokens"][:2] == [5000, 5002]
    assert out[2, 0] == FakeDecoder.next_id(5002, len(prompts[2]) + 1)


def test_compaction_off_makes_no_move(crig: Rig) -> None:
    prompts, blocks = _three_then_release_middle(crig)
    crig.model.decode_compact = False
    crig.decode([(1000, len(prompts[0]), blocks[0]), None, (1002, len(prompts[2]), blocks[2])])
    assert not any(isinstance(e, tuple) and e[0] == "move" for e in crig.decoder.log)
    assert crig.model.lane_moves == 0 and crig.decoder.steps[-1]["r"] == 4
    assert crig.model.lane_of_slot[2] == 2


def test_a_failed_move_marks_the_batch_failed(crig: Rig, monkeypatch) -> None:
    prompts, blocks = _three_then_release_middle(crig)

    def boom(self, src, dst):
        raise RuntimeError("device")

    monkeypatch.setattr(BucketDecoder, "move_lane", boom)
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        RuntimeError, match="device"
    ):
        crig.decode([(1000, len(prompts[0]), blocks[0]), None, (1002, len(prompts[2]), blocks[2])])
    assert crig.model.batch_failed and crig.model.lane_of_slot[2] == 2


def test_the_decode_stat_reports_the_active_bucket_width(crig: Rig) -> None:
    prompts, blocks = _three_then_release_middle(crig)
    crig.decode([(1000, len(prompts[0]), blocks[0]), None, (1002, len(prompts[2]), blocks[2])])
    assert crig.model.decode_stats[-1]["bucket"] == 2 and crig.decoder.active.lanes == 2
