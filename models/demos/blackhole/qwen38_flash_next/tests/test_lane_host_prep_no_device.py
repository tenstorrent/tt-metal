# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The decode step's host prep: ``_page_row`` (tensor-side trim), the per-step sampling-contract rows and the lazy
``park_idle`` give what the per-row Python conversions, the per-row field conversions and the park-every-step gave."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import PARK_DRIFT_MARGIN, Qwen38PlainLaneDecoder
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38HostSamplingPolicy

NUM_BLOCKS = 600
MAX_BLOCKS = 511


def old_page_row(adapter, row, *, needed, label):
    """The full-row Python conversion ``_page_row`` had before the tensor-side trim."""

    ids = [int(block) for block in torch.as_tensor(row).reshape(-1).tolist()]
    count = len(ids)
    while count and ids[count - 1] == 0:
        count -= 1
    pages = ids[:count]
    if 0 in pages:
        raise ValueError(f"{label}: block 0 (vLLM's null block) inside the page-table row {pages}")
    if pages and (min(pages) < 0 or max(pages) >= adapter.num_blocks):
        raise ValueError(f"{label}: block ids must be in [1, {adapter.num_blocks}), got {pages}")
    if len(pages) > adapter.decoder.paging.max_blocks:
        raise ValueError(f"{label}: {len(pages)} blocks exceed the lane table's {adapter.decoder.paging.max_blocks}")
    if len(pages) < needed:
        raise ValueError(f"{label}: needs {needed} blocks for its tokens, the page-table row holds {len(pages)}")
    return pages


def error_text(call):
    """The ValueError message ``call`` raises."""

    try:
        call()
    except ValueError as error:
        return str(error)
    raise AssertionError("expected a ValueError")


def adapter_stub(max_blocks=MAX_BLOCKS):
    return SimpleNamespace(
        num_blocks=NUM_BLOCKS, decoder=SimpleNamespace(paging=SimpleNamespace(max_blocks=max_blocks))
    )


def padded(pages, width=MAX_BLOCKS, dtype=torch.int32):
    row = torch.zeros(width, dtype=dtype)
    row[: len(pages)] = torch.tensor(pages, dtype=dtype)
    return row


PAGE_ROWS = {
    "trailing zeros": (padded([5, 9, 3]), 2, None),
    "full row": (padded(list(range(1, MAX_BLOCKS + 1))), 3, None),
    "empty row": (padded([]), 0, None),
    "empty row needs one": (padded([]), 1, "needs 1 blocks"),
    "block 0 inside": (padded([5, 0, 3]), 1, "block 0"),
    "negative id": (padded([5, -2, 3]), 1, "block ids must be in"),
    "out of range id": (padded([5, NUM_BLOCKS, 3]), 1, "block ids must be in"),
    "too few blocks": (padded([5, 9]), 3, "needs 3 blocks"),
    "int64 row": (padded([7, 8], dtype=torch.int64), 2, None),
    "2-d row": (padded([7, 8]).reshape(1, -1), 2, None),
}


@pytest.mark.parametrize("name", PAGE_ROWS)
def test_page_row_matches_the_full_row_conversion(name):
    row, needed, error = PAGE_ROWS[name]
    adapter = adapter_stub()
    new = lambda: vllm_module.Qwen38ForCausalLM._page_row(adapter, row, needed=needed, label="row 3")  # noqa: E731
    old = lambda: old_page_row(adapter, row, needed=needed, label="row 3")  # noqa: E731
    if error is None:
        pages = new()
        assert pages == old() and all(type(page) is int for page in pages)
    else:
        message = error_text(new)
        assert error in message and message == error_text(old)


def test_page_row_refuses_more_blocks_than_the_lane_table_holds():
    adapter = adapter_stub(max_blocks=4)
    row = padded([1, 2, 3, 4, 5], width=8)
    message = error_text(lambda: vllm_module.Qwen38ForCausalLM._page_row(adapter, row, needed=1, label="row 0"))
    assert message == error_text(lambda: old_page_row(adapter, row, needed=1, label="row 0"))
    assert "exceed the lane table" in message


# -- the sampling contract's rows ---------------------------------------------------------------------------


def sampling_contract(tensors=True):
    """The plugin's padded per-row lists (``submit_decode``): 32 rows of greedy, sampled and penalised requests."""

    rows = range(32)
    fields = {
        "temperature": [0.0 if row % 3 == 0 else 0.7 + 0.01 * row for row in rows],
        "top_k": [1 if row % 3 == 0 else 20 + row for row in rows],
        "top_p": [1.0 if row % 3 == 0 else 0.9 for row in rows],
        "presence_penalty": [0.5 if row % 4 == 1 else 0.0 for row in rows],
        "frequency_penalty": [0.25 if row % 5 == 2 else 0.0 for row in rows],
        "repetition_penalty": [1.1 if row % 7 == 3 else 1.0 for row in rows],
        "seed": [row + 100 if row % 2 else -1 for row in rows],
        "enable_log_probs": [row % 6 == 4 for row in rows],
        "num_logprobs": [0] * 32,
    }
    if tensors:
        dtypes = {
            "top_k": torch.int32,
            "seed": torch.int64,
            "enable_log_probs": torch.bool,
            "num_logprobs": torch.int32,
        }
        fields = {name: torch.tensor(value, dtype=dtypes.get(name, torch.float32)) for name, value in fields.items()}
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("tensors", [True, False])
def test_contract_rows_build_the_same_policies_for_every_row(tensors):
    params = sampling_contract(tensors)
    rows = Qwen38HostSamplingPolicy.contract_rows(params)
    assert all(isinstance(value, list) for value in vars(rows).values())
    for row in range(32):
        assert Qwen38HostSamplingPolicy.from_contract(rows, row) == Qwen38HostSamplingPolicy.from_contract(params, row)
    policies = [Qwen38HostSamplingPolicy.from_contract(rows, row) for row in range(32)]
    assert any(p.greedy for p in policies) and any(p.penalized for p in policies) and any(p.logprobs for p in policies)


def test_contract_rows_keep_scalars_missing_fields_and_errors():
    params = SimpleNamespace(temperature=0.8, top_k=torch.tensor([5, 6]), top_p=0.95)  # no seed, penalties or logprobs
    rows = Qwen38HostSamplingPolicy.contract_rows(params)
    assert not hasattr(rows, "seed") and rows.temperature == 0.8
    for row in range(2):
        assert Qwen38HostSamplingPolicy.from_contract(rows, row) == Qwen38HostSamplingPolicy.from_contract(params, row)
    message = error_text(lambda: Qwen38HostSamplingPolicy.from_contract(rows, 2))
    assert "row 2 is missing" in message
    assert message == error_text(lambda: Qwen38HostSamplingPolicy.from_contract(params, 2))


# -- park_idle ----------------------------------------------------------------------------------------------


class Park:
    """A plain decoder over a stub position row and bucket: which lanes ``park_idle`` parks."""

    def __init__(self, replayed, buckets=(1, 2, 4, 32), allocated_context=32768):
        self.decoder = Qwen38PlainLaneDecoder.__new__(Qwen38PlainLaneDecoder)
        self.decoder.model = SimpleNamespace(allocated_context=allocated_context)
        self.decoder.buckets = buckets
        self.decoder.bucket_for_step = lambda: replayed
        self.parked = []
        self.position = SimpleNamespace(
            positions=[0] * 32, shifts=[0] * 32, park=lambda lanes: self.parked.append(list(lanes))
        )
        self.decoder.ls = SimpleNamespace(position=self.position)

    def run(self, idle=None):
        idle = list(range(1, 32)) if idle is None else idle
        return self.decoder.park_idle(idle), self.parked


def old_park_idle(position, lanes):
    return any(position.positions[lane] or position.shifts[lane] for lane in lanes)


def test_park_idle_skips_idle_lanes_that_drifted_above_the_replayed_bucket():
    rig = Park(replayed=1)
    for lane in range(1, 32):
        rig.position.positions[lane] = 3 + lane  # small drift: the old rule would park
    assert old_park_idle(rig.position, range(1, 32))
    assert rig.run() == (False, [])


def test_park_idle_parks_all_idle_lanes_when_one_below_the_bucket_is_off_zero():
    rig = Park(replayed=4)
    rig.position.positions[2] = 5
    rig.position.positions[20] = 9
    wrote, parked = rig.run()
    assert wrote and parked == [list(range(1, 32))]


def test_park_idle_checks_the_shift_of_a_lane_below_the_bucket():
    rig = Park(replayed=4)
    rig.position.shifts[3] = 7
    assert rig.run()[0] is True
    rig = Park(replayed=4)
    rig.position.shifts[9] = 7  # above the bucket: not computed
    assert rig.run() == (False, [])


def test_park_idle_parks_all_when_a_lane_above_the_bucket_reaches_the_bound():
    bound = 32768 - PARK_DRIFT_MARGIN
    rig = Park(replayed=1)
    rig.position.positions[17] = bound - 1
    assert rig.run() == (False, [])
    rig.position.positions[17] = bound
    wrote, parked = rig.run()
    assert wrote and parked == [list(range(1, 32))]


@pytest.mark.parametrize("positions", [{}, {5: 1}, {31: 2}, {0: 4}])
def test_park_idle_with_one_bucket_is_the_old_rule(positions):
    rig = Park(replayed=32, buckets=(32,))
    for lane, value in positions.items():
        rig.position.positions[lane] = value
    idle = list(range(32))
    wrote, parked = rig.run(idle)
    assert wrote == old_park_idle(rig.position, idle) and parked == ([idle] if wrote else [])
