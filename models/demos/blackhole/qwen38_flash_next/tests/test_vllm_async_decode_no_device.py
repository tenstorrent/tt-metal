# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The adapter's async decode (``QWEN38_ASYNC_DECODE``) on a fake lane decoder, a fake chain and a fake ttnn (events, the
non-blocking read, the trace replay): no device.

Knob off is today's capability and behaviour.  A steady lane step (``reload_inputs=False``, plain greedy, device sampling)
does its pre-step host work, waits for the previous step's token read, writes the PLE rows of those tokens, skips the token
write (the lane trace's epilogue left each lane's greedy token in the row), replays without blocking, enqueues the token
row read and two events, and returns a handle the plugin's ``read_decode_output`` / ``process_decode_output_host`` take.  Any
other call waits for the step in flight first and runs the blocking path.  The chain route (the hybrid's lone row, and the
single slot) enqueues HEAD before it waits.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE
from models.demos.blackhole.qwen38_flash_next.tests import test_vllm_batch_adapter_no_device as batch_tests
from models.demos.blackhole.qwen38_flash_next.tests.test_qwen38_vllm_no_device import (  # noqa: F401  (fixture)
    ALLOCATED_CONTEXT,
    PAGE_TABLE,
    FakeChain,
    construction,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_adapter_no_device import (  # noqa: F401
    FakeDecoder,
    Rig,
    _prompt,
    _single_path_prefill,
    _three,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import WARM_TOKEN_ID
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    Qwen38AsyncStep,
    Qwen38AsyncTokenError,
    Qwen38ForCausalLM,
    greedy_batch_sampling_params,
)

LANES = batch_tests.LANES
ORDER: list = []  # the host-visible order of the fake device operations of every fake below


class AsyncDecoder(FakeDecoder):
    """The fake lane decoder with the async surface: bucketed steps whose 'device' keeps a token row the way the lane
    trace does (the epilogue writes each computed lane's greedy token), an enqueue/finish split, the token row read."""

    BUCKETS = (1, 2, 4, 8, 16, 32)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.row = [0] * self.lanes  # the device token row
        self.consumed: list[list[int]] = []  # per replay: the tokens the lanes actually consumed
        self.ple_tokens: list[list[int]] = []
        self.reads = 0
        self.last_keep: list | None = None

    def bucket_for_step(self) -> int:
        live = [lane for lane in range(self.lanes) if self.positions[lane] is not None]
        need = max(live) + 1 if live else 1
        return next(r for r in self.BUCKETS if r >= need)

    def sync_page_table(self, keep=None) -> bool:
        wrote = super().sync_page_table()
        if wrote:
            ORDER.append("sync_table")
            if keep is not None:
                keep.append("table-host")
        return wrote

    def park_idle(self, lanes) -> bool:
        ORDER.append("park")
        return super().park_idle(lanes)

    def import_lane(self, vm, lane, length, context, pages) -> None:
        ORDER.append("import")
        super().import_lane(vm, lane, length, context, pages)

    def release_lane_external(self, lane) -> bool:
        ORDER.append("release_lane")
        return super().release_lane_external(lane)

    def move_lane(self, src: int, dst: int) -> None:
        ORDER.append(f"move:{src}->{dst}")
        self.positions[dst], self.positions[src] = self.positions[src], None
        self.page_rows[dst], self.page_rows[src] = self.page_rows[src], []
        self.contexts[dst], self.contexts[src] = self.contexts[src], None
        position = self.ls.position.positions
        position[dst], position[src] = position[src], 0
        self.page_table_dirty = True
        self.last_move = {"src": src, "dst": dst, "ms": 0.0}
        # the pager moves the lane's tensors but not its token row entry

    def compaction_moves(self):
        live = sorted(lane for lane in range(self.lanes) if self.positions[lane] is not None)
        b = next(r for r in self.BUCKETS if r >= max(len(live), 1))
        high = [lane for lane in reversed(live) if lane >= b]
        free = [lane for lane in range(b) if self.positions[lane] is None and not self.page_rows[lane]]
        return list(zip(high, free))

    def enqueue_step(self, tokens, *, blocking=True, write_tokens=True, keep=None) -> int:
        r = self.bucket_for_step()
        ORDER.append(f"ple:{list(tokens[:r])}")
        self.ple_tokens.append(list(tokens[:r]))
        if keep is not None:
            keep.append("ple-host")
        if write_tokens:
            ORDER.append("token_write")
            self.row[:r] = list(tokens[:r])
            if keep is not None:
                keep.append("token-host")
        ORDER.append("replay:" + ("blocking" if blocking else "nonblocking"))
        positions = list(self.ls.position.positions)
        self.consumed.append(list(self.row[:r]))
        self.steps.append(
            {
                "tokens": list(tokens),
                "positions": positions,
                "rows": [list(row) for row in self.page_rows],
                "live": [p is not None for p in self.positions],
            }
        )
        self.row[:r] = [self.next_id(self.row[lane], positions[lane]) for lane in range(r)]  # the resolve epilogue
        self.last_ids = list(self.row)
        self.ls.position.positions = [p + 1 for p in positions]
        self.step_split = {"ple": 0.0, "tok": 0.0, "exe": 0.0, "adv": 0.0, "read": 0.0}
        return r

    def finish_step(self, r: int) -> list[int]:
        return [int(v) for v in self.row[:r]] + [-1] * (self.lanes - r)

    def step(self, tokens) -> list[int]:
        return self.finish_step(self.enqueue_step(tokens))

    def enqueue_token_read(self):
        self.reads += 1
        ORDER.append("read")
        host = torch.tensor([float(v) for v in self.row], dtype=torch.float32).reshape(1, 1, 1, -1)
        ORDER.append("events")
        return host, SimpleNamespace(name=f"a{self.reads}"), SimpleNamespace(name=f"p{self.reads}")


class AsyncChain(FakeChain):
    """The fake chain with the async primitives; every op logs into ``log`` (a list shared with ``ORDER``)."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ORDER
        self.row_value: int | None = None
        self.reads = 0

    def feed_token_row(self, token_id: int) -> bool:
        if self.row_value == token_id:
            self.log.append(f"feed_skip:{token_id}")
            return False
        self.log.append(f"feed:{token_id}")
        self.write_token_row(token_id)
        return True

    def write_token_row(self, token_id: int) -> None:
        self.row_value = None
        super().write_token_row(token_id)

    def execute_head(self, residue: int) -> None:
        self.row_value = None
        super().execute_head(residue)

    def read_token_row(self) -> int:
        value = super().read_token_row()
        self.row_value = value
        return value

    def read_token_row_lean(self) -> int:
        return self.read_token_row()

    def mark_ple_fenced(self) -> None:
        self.log.append("fence")
        self.fences = getattr(self, "fences", 0) + 1

    def read_token_row_nonblocking(self):
        self.reads += 1
        self.log.append("read_nb")
        return torch.tensor([float(self.row)] + [0.0] * 31, dtype=torch.float32).reshape(1, 1, 1, 32)

    def record_event(self):
        self.log.append("event")
        return SimpleNamespace(name=f"c{len(self.log)}")

    def event_synchronize(self, event) -> None:
        self.log.append(f"sync:{event.name}")

    @staticmethod
    def pending_value(host) -> int:
        return int(host.reshape(-1)[0])

    def note_token_row_value(self, value: int) -> None:
        self.log.append(f"note:{value}")
        self.row_value = value


@pytest.fixture(autouse=True)
def _fake_ttnn(monkeypatch):
    ORDER.clear()
    ttnn = vllm_module.ttnn
    monkeypatch.setattr(ttnn, "event_synchronize", lambda event: ORDER.append(f"wait:{event.name}"), raising=False)
    monkeypatch.setattr(ttnn, "to_torch", lambda host: host, raising=False)


@pytest.fixture
def async_on(monkeypatch):
    monkeypatch.setattr(vllm_module, "ASYNC_DECODE", True)
    monkeypatch.setattr(batch_tests, "FakeDecoder", AsyncDecoder)
    monkeypatch.setattr(batch_tests, "FakeChain", AsyncChain)


@pytest.fixture
def rig(async_on, monkeypatch) -> Rig:
    return Rig(monkeypatch)


@pytest.fixture
def hybrid_rig(async_on, monkeypatch) -> Rig:
    rig = Rig(monkeypatch)
    rig.model.hybrid_single = True
    return rig


def _start(rig: Rig, requests=((40, [5]), (70, [9, 3]), (130, [20, 11, 7]))):
    """Prefill requests into slots 0..n-1; returns ``(first tokens, positions, blocks)``."""

    prompts = [_prompt(length, 1000 * (i + 1)) for i, (length, _blocks) in enumerate(requests)]
    blocks = [b for _length, b in requests]
    out, _ = rig.prefill(prompts, list(range(len(requests))), blocks, greedy_batch_sampling_params(len(requests)))
    return [int(out[i, 0]) for i in range(len(requests))], [length for length, _b in requests], blocks


def _blocking(rig: Rig, tokens, positions, blocks, **overrides):
    return rig.decode([(t, p, b) for t, p, b in zip(tokens, positions, blocks)], **overrides)


def _steady(rig: Rig, blocks, **overrides):
    """A steady call: the plugin's tokens and positions are stale (token 777, position 1) and must be ignored."""

    return rig.decode([(777, 1, b) for b in blocks], reload_inputs=False, **overrides)


def _tokens(rig: Rig, handle, rows: int) -> list[int]:
    host, events = rig.model.read_decode_output(handle, async_read=True)
    assert events == [handle.event]
    out = rig.model.process_decode_output_host(host, is_tokens=True)
    return [int(out[i, 0]) for i in range(rows)]


def _expected(tokens, positions):
    return [FakeDecoder.next_id(t, p) for t, p in zip(tokens, positions)]


# -- knob off -----------------------------------------------------------------------------------------------------------


def _capability_in_subprocess(env_value: str | None) -> str:
    import os

    env = {k: v for k, v in os.environ.items() if k != "QWEN38_ASYNC_DECODE"}
    env["QWEN38_RING"] = "0"  # explicit, so this no-device test never probes the device cluster
    if env_value is not None:
        env["QWEN38_ASYNC_DECODE"] = env_value
    code = (
        "from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM as M; "
        "print('CAP', M.model_capabilities['supports_async_decode'])"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    return next(line for line in result.stdout.splitlines() if line.startswith("CAP")).split()[1]


def test_the_capability_follows_the_knob_read_at_import() -> None:
    assert _capability_in_subprocess(None) == "False"
    assert _capability_in_subprocess("0") == "False"
    assert _capability_in_subprocess("1") == "True"


def test_knob_off_is_the_old_behaviour(monkeypatch, expect_error) -> None:
    monkeypatch.setattr(vllm_module, "ASYNC_DECODE", False)
    monkeypatch.setattr(batch_tests, "FakeDecoder", AsyncDecoder)
    rig = Rig(monkeypatch)
    assert rig.model.async_decode is False and Qwen38ForCausalLM.model_capabilities["supports_async_decode"] is False
    tokens, positions, blocks = _start(rig)
    with expect_error(ValueError, "reload_inputs must be True"):
        _steady(rig, blocks)
    out = _blocking(rig, tokens, positions, blocks)  # the blocking path: a host tensor, no event, nothing in flight
    assert isinstance(out, torch.Tensor) and rig.model._inflight == [] and rig.model._async_pending is None
    assert [int(out[i, 0]) for i in range(3)] == _expected(tokens, positions)
    assert "token_write" in ORDER and "replay:blocking" in ORDER and "read" not in ORDER
    assert not any(entry.startswith("wait:") for entry in ORDER)
    assert rig.model.read_decode_output(out, async_read=True) == (out, [])


# -- the steady lane step -----------------------------------------------------------------------------------------------


def test_a_steady_lane_step_orders_the_host_work_wait_ple_nonblocking_replay_read_events(rig: Rig) -> None:
    model, decoder = rig.model, rig.decoder
    tokens, positions, blocks = _start(rig, ((40, [5]), (70, [9, 3]), (130, [20, 11, 7]), (50, [30])))
    first = _blocking(rig, tokens, positions, blocks)  # reload_inputs=True: blocking, a host tensor
    assert isinstance(first, torch.Tensor) and model._async_pending is None
    n1 = _expected(tokens, positions)
    assert [int(first[i, 0]) for i in range(4)] == n1

    ORDER.clear()
    h1 = _steady(rig, blocks)  # stale tokens ignored; the previous step was blocking: its returned tokens feed
    assert isinstance(h1, Qwen38AsyncStep) and h1.kind == "lane" and h1.rows == LANES and h1.lanes == 4
    assert h1.row_lane == {0: 0, 1: 1, 2: 2, 3: 3} and model._async_pending is h1 and model._inflight == [h1]
    # all four lanes of bucket 4 are live and the previous call's tokens are not in the row (blocking path wrote them):
    # the row holds them, so no token write
    assert ORDER[-4:] == [f"ple:{n1}", "replay:nonblocking", "read", "events"] and "token_write" not in ORDER
    assert decoder.consumed[-1] == n1  # the replay consumed what the previous step produced
    assert h1.keep == ["ple-host"]  # held until the adapter waits the step's event
    n2 = _expected(n1, [p + 1 for p in positions])

    ORDER.clear()
    h2 = _steady(rig, blocks)  # issued BEFORE the previous output is processed, like the plugin
    pre = ORDER[: ORDER.index("wait:a1")]
    assert "wait:a1" in ORDER and f"ple:{n1}" not in pre  # the pre-step work (park) comes first; the wait before PLE
    assert ORDER[ORDER.index("wait:a1") :] == ["wait:a1", f"ple:{n2}", "replay:nonblocking", "read", "events"]
    assert decoder.ple_tokens[-1] == n2 and "token_write" not in ORDER and decoder.consumed[-1] == n2
    assert model._inflight == [h2] and h1.keep == ["ple-host"]  # step 1's holds are released with the list
    assert _tokens(rig, h2, 4) == _expected(n2, [p + 2 for p in positions])
    assert _tokens(rig, h1, 4) == _expected(n1, [p + 1 for p in positions])  # the older output still reads after N+1


def test_an_idle_lane_inside_the_bucket_keeps_the_token_write(rig: Rig) -> None:
    tokens, positions, blocks = _start(rig)  # three live lanes, bucket 4: lane 3 is idle but computed
    _blocking(rig, tokens, positions, blocks)
    ORDER.clear()
    _steady(rig, blocks)
    assert "token_write" in ORDER  # lane 3 takes the warm token, as on the blocking path
    assert rig.decoder.ple_tokens[-1][3] == WARM_TOKEN_ID and rig.decoder.consumed[-1][3] == WARM_TOKEN_ID


def test_page_rows_and_positions_come_from_the_adapter_not_the_plugin(rig: Rig) -> None:
    tokens, positions, blocks = _start(rig, ((63, [5]), (70, [9, 3])))
    _blocking(rig, tokens, positions, blocks)  # lane 0 is now at 64: its next step needs a second block
    ORDER.clear()
    _steady(rig, [[5, 8], [9, 3]])
    assert rig.decoder.page_rows[0] == [5, 8] and rig.decoder.steps[-1]["positions"][:2] == [64, 71]
    assert "sync_table" in ORDER and ORDER.index("sync_table") < ORDER.index("replay:nonblocking")
    handle = rig.model._async_pending
    assert any(item == "table-host" for item in handle.keep)  # the page-table upload's host tensor is held


def test_process_decode_output_host_maps_rows_through_a_slot_remap(rig: Rig) -> None:
    model = rig.model
    tokens, positions, blocks = _start(rig, ((40, [5]), (70, [9, 3])))
    _blocking(rig, tokens, positions, blocks)
    swap = list(range(LANES))
    swap[0], swap[1] = 1, 0  # row 0 now holds slot 1's state (lane 1), row 1 slot 0's (lane 0)
    handle = _steady(rig, [[9, 3], [5]], slot_remap=swap)
    assert handle.row_lane == {0: 1, 1: 0}
    host, _events = model.read_decode_output(handle, async_read=True)
    out = model.process_decode_output_host(host, is_tokens=True)
    n1 = _expected(tokens, positions)
    step2 = _expected(n1, [p + 1 for p in positions])  # by lane
    assert tuple(out.shape) == (LANES, 1) and out.dtype == torch.int32
    assert int(out[0, 0]) == step2[1] and int(out[1, 0]) == step2[0] and int(out[2:].abs().sum()) == 0


def test_a_compaction_move_forces_the_token_write_and_moves_the_token(async_on, monkeypatch) -> None:
    rig = Rig(monkeypatch, lanes=32)  # decode buckets need 32 lanes
    model, decoder = rig.model, rig.decoder
    tokens, positions, blocks = _start(rig, ((40, [5]), (70, [9, 3]), (50, [30]), (60, [31]), (45, [32])))
    _blocking(rig, tokens, positions, blocks)
    first = _steady(rig, blocks)  # five live: bucket 8, nothing moves
    assert first.lanes == 8 and not any(entry.startswith("move:") for entry in ORDER)
    n1 = _expected(tokens, positions)
    n2 = _expected(n1, [p + 1 for p in positions])
    # slot 1 finishes: four live requests are left, lane 4 is above bucket 4 and moves into the free lane 1
    model.release_request(1)
    ORDER.clear()
    rows = [blocks[0], None, blocks[2], blocks[3], blocks[4]]
    handle = rig.decode([(777, 1, b) if b is not None else None for b in rows], reload_inputs=False)
    assert ORDER.index("move:4->1") < ORDER.index("replay:nonblocking") and "token_write" in ORDER
    assert decoder.positions[1] is not None and decoder.positions[4] is None
    assert isinstance(handle, Qwen38AsyncStep) and handle.row_lane == {0: 0, 2: 2, 3: 3, 4: 1}
    assert decoder.consumed[-1][1] == n2[4]  # the moved lane feeds the token its request produced in lane 4


def test_a_discarded_step_needs_no_rollback_the_next_prefill_waits_it(rig: Rig) -> None:
    model, decoder = rig.model, rig.decoder
    tokens, positions, blocks = _start(rig, ((40, [5]), (70, [9, 3])))
    _blocking(rig, tokens, positions, blocks)
    handle = _steady(rig, blocks)
    ORDER.clear()
    model.release_request(0)  # the request finished: the plugin discards `handle`'s output, nothing is rolled back
    assert ORDER[:2] == ["wait:a1", "release_lane"] and model._async_pending is None and model._inflight == []
    n2 = _expected(_expected(tokens, positions), [p + 1 for p in positions])
    assert model._last_tokens == {
        1: n2[1]
    }  # the discarded step's tokens are the adapter's record, minus the released lane
    ORDER.clear()
    rig.prefill([_prompt(20, 7000)], [0], [[40]], greedy_batch_sampling_params(1))
    assert "import" in ORDER and decoder.positions[0] == 20


# -- quiesce ------------------------------------------------------------------------------------------------------------


def test_reload_inputs_true_waits_the_step_in_flight_then_runs_the_blocking_path(rig: Rig) -> None:
    model, decoder = rig.model, rig.decoder
    tokens, positions, blocks = _start(rig, ((40, [5]), (70, [9, 3])))
    _blocking(rig, tokens, positions, blocks)
    h1 = _steady(rig, blocks)
    n1 = _expected(tokens, positions)
    n2 = _expected(n1, [p + 1 for p in positions])
    ORDER.clear()
    out = _blocking(rig, n2, [p + 2 for p in positions], blocks)  # a reload: the plugin's tokens are authoritative
    assert ORDER[0] == "wait:a1" and "replay:blocking" in ORDER and "read" not in ORDER
    assert isinstance(out, torch.Tensor) and model._async_pending is None and model._inflight == []
    assert [int(out[i, 0]) for i in range(2)] == _expected(n2, [p + 2 for p in positions])
    assert model.read_decode_output(out, async_read=True) == (out, [])
    assert model.process_decode_output_host(out, is_tokens=True) is out
    # the handle of the step the reload superseded still reads fine on the output thread
    assert _tokens(rig, h1, 2) == n2


def test_a_sampled_row_without_reload_waits_and_feeds_the_adapters_own_token(rig: Rig) -> None:
    model, decoder = rig.model, rig.decoder
    tokens, positions, blocks = _start(rig, ((40, [5]), (70, [9, 3])))
    _blocking(rig, tokens, positions, blocks)
    _steady(rig, blocks)
    n1 = _expected(tokens, positions)
    sampled = greedy_batch_sampling_params(LANES)
    sampled.temperature = [0.7] * LANES
    sampled.top_k = [50] * LANES
    ORDER.clear()
    out = _steady(rig, blocks, sampling_params=sampled)  # not steady: sampled rows
    assert isinstance(out, torch.Tensor) and ORDER[0] == "wait:a1"
    assert decoder.consumed[-1][:2] == _expected(n1, [p + 1 for p in positions]) and "replay:blocking" in ORDER
    assert "token_write" in ORDER and model._inflight == []


@pytest.mark.parametrize("which", ("prefill", "release", "persistent"))
def test_each_entry_point_quiesces_first(rig: Rig, which: str) -> None:
    model = rig.model
    tokens, positions, blocks = _start(rig, ((40, [5]),))
    _blocking(rig, tokens, positions, blocks)
    _steady(rig, blocks)
    ORDER.clear()
    if which == "prefill":
        rig.prefill([_prompt(20, 9000)], [1], [[50]], greedy_batch_sampling_params(1))
    elif which == "release":
        model.release_request(0)
    else:
        model.release_persistent_capture()
    assert ORDER[0] == "wait:a1" and model._async_pending is None and model._inflight == []


def test_a_token_outside_the_tokenizer_raises_before_the_replay_and_keeps_the_lanes(rig: Rig, expect_error) -> None:
    model, decoder = rig.model, rig.decoder
    tokens, positions, blocks = _start(rig, ((40, [5]),))
    _blocking(rig, tokens, positions, blocks)
    _steady(rig, blocks)
    model._async_pending.host.reshape(-1)[0] = float(TOKENIZER_SIZE + 3)  # the step in flight read a padded LM-head id
    steps = len(decoder.steps)
    with expect_error(Qwen38AsyncTokenError, "outside the tokenizer"):
        _steady(rig, blocks)
    assert len(decoder.steps) == steps and model.batch_failed is False and model._async_pending is None


# -- the chain route ----------------------------------------------------------------------------------------------------


def _chain_ops(log: list, start: int) -> list:
    return [entry for entry in log[start:] if not entry.startswith(("guard", "unguard"))]


def _hybrid_blocking_reference(hybrid_rig_factory, steps: int):
    """The same request through blocking steps only: its committed, ple_context and tokens after each step."""

    ref = hybrid_rig_factory()
    token, _pos, blocks = _start(ref, ((40, [5]),))
    token = token[0]
    history = []
    for step in range(steps):
        out = _blocking(ref, [token], [40 + step], blocks)
        token = int(out[0, 0])
        history.append((list(ref.model.slot.committed), ref.model.slot.ple_context, token))
    return history


def test_the_hybrid_chain_route_enqueues_head_before_the_wait_and_matches_the_blocking_path(
    async_on, monkeypatch
) -> None:
    def make():
        rig = Rig(monkeypatch)
        rig.model.hybrid_single = True
        return rig

    reference = _hybrid_blocking_reference(make, 6)
    ORDER.clear()
    rig = make()
    model, chain = rig.model, rig.chain
    (token,), _positions, blocks = _start(rig, ((40, [5]),))
    out = _blocking(rig, [token], [40], blocks)  # reload_inputs=True: blocking
    assert (
        isinstance(out, torch.Tensor) and (model.slot.committed, model.slot.ple_context, int(out[0, 0])) == reference[0]
    )
    handles = []
    # step 2: nothing in flight, the blocking step's token feeds (the row already holds it: no feed), then the read
    mark = len(ORDER)
    h = _steady(rig, [blocks[0]])
    ops = _chain_ops(ORDER, mark)
    assert isinstance(h, Qwen38AsyncStep) and h.kind == "chain" and h.rows == LANES and h.row_lane == {0: 0}
    assert ops[0].startswith("feed") and ops[-3:] == ["read_nb", "event", "event"]
    handles.append(h)
    assert (model.slot.committed, model.slot.ple_context) == reference[1][:2]
    for step in range(2, 6):
        mark = len(ORDER)
        h = _steady(rig, [blocks[0]])  # stale tokens ignored
        ops = _chain_ops(ORDER, mark)
        residue = (40 + step) % 4
        head = ops.index(f"head:{residue}")
        wait = next(i for i, entry in enumerate(ops) if entry.startswith("sync:"))
        assert head < wait  # HEAD(N+1) is enqueued before the wait
        assert ops[wait:] == [
            ops[wait],
            "fence",
            f"ple_prepare:{reference[step - 1][2]}",
            f"ple_upload:row:{reference[step - 1][2]}",
            f"tail:{residue}",
            "read_nb",
            "event",
            "event",
        ]
        assert "guard" not in ORDER[mark:]  # no loop_guard on the async path
        # committed / ple_context after step N+1 are the blocking path's
        assert (model.slot.committed, model.slot.ple_context) == reference[step][:2]
        handles.append(h)
    assert model._inflight == [handles[-1]]
    for step, handle in enumerate(handles, start=1):
        host, events = model.read_decode_output(handle, async_read=True)
        assert [int(model.process_decode_output_host(host, is_tokens=True)[0, 0])] == [reference[step][2]]
    assert chain.fences >= 4


def test_the_chain_token_outside_the_tokenizer_fails_the_slot_with_a_clear_error(hybrid_rig: Rig, expect_error) -> None:
    rig = hybrid_rig
    (token,), _p, blocks = _start(rig, ((40, [5]),))
    _blocking(rig, [token], [40], blocks)
    _steady(rig, [blocks[0]])
    rig.chain.row = TOKENIZER_SIZE + 5  # what the step in flight's TAIL resolved
    # the read of that step happened at enqueue: patch the pending host copy as the device would have written it
    rig.model._async_pending.host.reshape(-1)[0] = float(TOKENIZER_SIZE + 5)
    with expect_error(Qwen38AsyncTokenError, "outside the tokenizer"):
        _steady(rig, [blocks[0]])
    assert rig.model.slot.failed is True


def test_demote_quiesces_and_the_lane_continues_from_the_chains_token(hybrid_rig: Rig) -> None:
    rig = hybrid_rig
    model, chain, decoder = rig.model, rig.chain, rig.decoder
    (token,), _p, blocks = _start(rig, ((40, [5]),))
    _blocking(rig, [token], [40], blocks)
    _steady(rig, [blocks[0]])
    h2 = _steady(rig, [blocks[0]])
    chain_token = int(chain.pending_value(h2.host))
    committed = len(model.slot.committed)
    mark = len(ORDER)
    rig.prefill([_prompt(20, 5000)], [1], [[7]], greedy_batch_sampling_params(1))  # the second request: demote
    ops = ORDER[mark:]
    assert ops[0].startswith("sync:") and ops.index("import") > 0 and model.hybrid_slot is None
    assert model._async_pending is None and model._inflight == []
    assert model._last_tokens == {0: chain_token} and decoder.positions[0] == committed
    # reload_inputs=False for the demoted request alone is not valid here (a new request arrived): a blocking reload
    out = rig.decode([(chain_token, committed, [5]), (9, 20, [7])])
    assert int(out[0, 0]) == FakeDecoder.next_id(chain_token, committed)


def test_a_hybrid_from_a_logits_prefill_holds_no_token_and_its_first_decode_reloads(
    hybrid_rig: Rig, expect_error
) -> None:
    rig = hybrid_rig
    model, chain = rig.model, rig.chain
    model._last_tokens, model._last_synced = {0: 123}, True  # a previous request's record
    out, _ = rig.prefill([_prompt(40, 1000)], [0], [[5]], None)  # the server's prefill: logits, no sampling params
    assert out.dim() == 3 and model.hybrid_slot == 0 and model._last_tokens is None and not model._last_synced
    with expect_error(Exception, "holds no token"):
        _steady(rig, [[5]])  # a non-reloading first decode has no token to feed
    token = int(torch.argmax(out[0, 0]))  # the plugin's host sampling
    mark = len(ORDER)
    result = _blocking(rig, [token], [40], [[5]])  # reload_inputs=True: a blocking step of the plugin's token
    assert isinstance(result, torch.Tensor) and f"feed:{token}" in _chain_ops(ORDER, mark)
    assert model._last_tokens == {0: int(result[0, 0])} and model._last_synced
    assert isinstance(_steady(rig, [[5]]), Qwen38AsyncStep)  # steady from the second decode on


def test_the_single_slot_chain_steady_steps(async_on) -> None:
    chain = AsyncChain()
    model = Qwen38ForCausalLM(chain, allocated_context=ALLOCATED_CONTEXT)
    ids = list(range(1000, 1010))
    logits, _ = model.prefill_forward(
        tokens=torch.tensor([ids], dtype=torch.int32),
        page_table=PAGE_TABLE,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(ids)],
        start_pos=torch.zeros(1, dtype=torch.int32),
        empty_slots=[0],
        sampling_params=greedy_batch_sampling_params(1),
    )

    def call(token, position, **overrides):
        kwargs = dict(
            tokens=torch.tensor([[token]], dtype=torch.int32),
            start_pos=torch.tensor([position], dtype=torch.int32),
            page_table=PAGE_TABLE,
            kv_cache=None,
            enable_trace=True,
            read_from_device=False,
            sampling_params=greedy_batch_sampling_params(1),
        )
        kwargs.update(overrides)
        return model.decode_forward(**kwargs)

    out = call(int(logits[0, 0]), len(ids))
    assert isinstance(out, torch.Tensor)
    h1 = call(0, 0, reload_inputs=False)
    h2 = call(0, 0, reload_inputs=False)
    assert isinstance(h2, Qwen38AsyncStep) and h2.rows == 1 and len(model.slot.committed) == len(ids) + 3
    token1 = int(
        model.process_decode_output_host(model.read_decode_output(h1, async_read=True)[0], is_tokens=True)[0, 0]
    )
    assert tuple(model.process_decode_output_host(h2, is_tokens=True).shape) == (1, 1)
    assert model.slot.committed[-1] == token1  # step 2's input was step 1's token (taken from the device read)
    # a release waits the step in flight and clears the host state
    ORDER.clear()
    model.release_request(0)
    assert ORDER[0].startswith("sync:") and model.slot.committed == [] and model._inflight == []


# -- QWEN38_STEP_SPLIT --------------------------------------------------------------------------------------------------


def test_the_step_split_has_the_wait_and_the_enqueue_parts(rig: Rig, monkeypatch) -> None:
    from collections import deque

    monkeypatch.setattr(vllm_module, "STEP_SPLIT", True)
    rig.model._split_outside = deque(maxlen=4)
    tokens, positions, blocks = _start(rig, ((40, [5]),))
    _blocking(rig, tokens, positions, blocks)
    _steady(rig, blocks)
    _steady(rig, blocks)
    split = rig.model.decode_stats[-1]["step_split"]
    assert {"wait", "exe"} <= set(split) and split["wait"] >= 0.0
    text = rig.model._split_log_text(list(rig.model.decode_stats))
    assert "wait " in text and "exe " in text
