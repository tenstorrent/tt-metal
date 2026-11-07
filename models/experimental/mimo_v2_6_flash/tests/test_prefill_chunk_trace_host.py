# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of the traced later prefill chunks (TtMiMoModel.setup_prefill_chunk_traces; no device): a row longer
than one chunk is traced only when every chunk replays, vLLM's prefill_forward runs a row that would mix replays and
eager chunks all eager and after the lane's replays, and the traces are refused on a build whose chunked MLA prefill
takes no runtime start."""

import os
from types import SimpleNamespace

import pytest
import torch

from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv
from models.experimental.mimo_v2_6_flash.tt import model as model_lib
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel
from models.experimental.mimo_v2_6_flash.tt.rope import MiMoRotarySetup

CHUNK = 4096
ALIGN = 128  # the pair split's global block


def traced_lengths(model, lengths):
    """A fake TtMiMoModel._chunk_trace: a trace for every later chunk of one of ``lengths`` tokens."""
    model._chunk_trace = lambda slot, start, tokens: object() if start > 0 and tokens.numel() in lengths else None


def bare_model(lengths=(CHUNK,)):
    model = TtMiMoModel.__new__(TtMiMoModel)
    model.max_prefill_chunk = CHUNK
    traced_lengths(model, lengths)
    return model


def test_a_row_is_traced_only_when_every_chunk_replays():
    model = bare_model()
    assert model.chunk_traced(0, 8192, CHUNK)
    assert model.chunk_traced(0, 8192, 2 * CHUNK), "two replays"
    # An unaligned start moved back 64 positions past the budget: a replay of 4096 tokens, then a 64-token eager tail.
    assert not model.chunk_traced(0, 8192, CHUNK + 64)
    assert not model.chunk_traced(0, 8192, 1000)
    assert not model.chunk_traced(0, 8192, 0)
    assert bare_model((CHUNK, 64)).chunk_traced(0, 8192, CHUNK + 64)


class Replica:
    """A replica whose prefill_enqueue splits a row into chunks as TtMiMoModel's does and records, per chunk, whether
    it replays (a trace and ``allow_trace``) or runs eagerly."""

    PREFILL_BATCH_MAX_PADDED = TtMiMoModel.PREFILL_BATCH_MAX_PADDED
    chunk_traced = TtMiMoModel.chunk_traced

    def __init__(self):
        self.max_prefill_chunk = CHUNK
        traced_lengths(self, (CHUNK,))
        self.mtp = self.dflash = self.decode_trace = None
        self.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: 0)
        self.events = []  # (slot, start, tokens, "replay" | "eager")

    def set_vllm_page_tables(self, rows):
        pass

    def prefill_enqueue(self, chunk, slot, start_pos=0, history_continues=False, draft_group=0, allow_trace=True):
        n = chunk.numel()
        for s in range(0, n, CHUNK):
            piece = chunk[s : s + CHUNK]
            replay = allow_trace and self._chunk_trace(slot, start_pos + s, piece) is not None
            self.events.append((slot, start_pos + s, piece.numel(), "replay" if replay else "eager"))
        return n

    def prefill_read(self, handle):
        return [torch.zeros(1, 8)]


def test_a_mixed_row_runs_all_eager_after_the_lane_replays(monkeypatch):
    """Three requests on one lane: an eager row (a short chunk), a mixed row (an unaligned start moved back past the
    4096-token budget: 4160 tokens) and a traced row. The traced row replays first; the mixed row runs all eager, so
    no replay follows an eager pass of the lane (its trace could overwrite that pass's pending logits)."""
    monkeypatch.setattr(gv, "PREFILL_CHUNK_TRACE", True)
    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", True)
    monkeypatch.setattr(gv, "TIMING", False)
    replica = Replica()
    gen = SimpleNamespace(models=[replica], dp=1, users_per_replica=4, cfg=SimpleNamespace(sliding_window=128))
    adapter = gv.MiMoV2ForCausalLM(gen, 1 << 16)
    adapter.layer_groups, adapter.pair_split = [0], True
    monkeypatch.setattr(adapter, "_layer_tables", lambda *a: None)
    monkeypatch.setattr(adapter, "_group_rows", lambda *a: {})
    end = 3 * CHUNK + 4 * ALIGN
    tokens = torch.zeros(3, end, dtype=torch.int32)
    # Row 0: 1000 tokens at 8192 (eager); row 1: 4096 tokens scheduled at 8256, moved back to 8192 (4160 tokens);
    # row 2: 4096 tokens at 8192 (traced).
    starts = [2 * CHUNK, 2 * CHUNK + 64, 2 * CHUNK]
    ends = [2 * CHUNK + 1000, 3 * CHUNK + 64, 3 * CHUNK]
    adapter.prefill_forward(tokens, starts, ends, None, None, empty_slots=[0, 1, 2])
    assert replica.events == [
        (2, 2 * CHUNK, CHUNK, "replay"),
        (0, 2 * CHUNK, 1000, "eager"),
        (1, 2 * CHUNK, CHUNK, "eager"),
        (1, 3 * CHUNK, 64, "eager"),
    ], replica.events
    kinds = [e[3] for e in replica.events]
    assert "replay" not in kinds[kinds.index("eager") :], "a replay after an eager pass of the lane"


def layers(pair_split):
    pair = object() if pair_split else None
    return [
        SimpleNamespace(is_swa=True, attn=SimpleNamespace(pair=None)),
        SimpleNamespace(is_swa=False, attn=SimpleNamespace(pair=pair)),
    ]


@pytest.mark.parametrize("pair_split", [False, True], ids=["owner_chunk", "pair_split"])
@pytest.mark.parametrize("runtime_start", [False, True], ids=["scalar_start", "runtime_start"])
def test_chunk_traces_need_a_runtime_start(monkeypatch, expect_error, pair_split, runtime_start):
    """Without chunked_flash_mla_prefill's chunk_start_idx_tensor the global layers' later chunks (except the KV pair
    split's) key on the scalar start the trace was captured at: setup_prefill_chunk_traces refuses before it allocates
    anything, and the vLLM adapter's warmup runs later chunks eagerly."""
    monkeypatch.setattr(model_lib.attention_lib, "MLA_RUNTIME_START", runtime_start)
    model = TtMiMoModel.__new__(TtMiMoModel)
    model.layers = layers(pair_split)
    supported = runtime_start or pair_split
    assert model.chunk_traces_supported() == supported
    if supported:
        return
    model.kv_caches, model.dflash, model.chunk_logits, model.chunk_traces = [object()], None, None, {}
    model._logits_row = lambda: pytest.fail("allocated a logits row")
    with expect_error(RuntimeError, "chunk_start_idx_tensor"):
        model.setup_prefill_chunk_traces([CHUNK])
    assert model.chunk_logits is None and model.chunk_traces == {}


def test_chunk_traces_are_on_by_default():
    if "MIMO_PREFILL_CHUNK_TRACE" in os.environ:
        pytest.skip("MIMO_PREFILL_CHUNK_TRACE is set")
    assert model_lib.PREFILL_CHUNK_TRACE and gv.PREFILL_CHUNK_TRACE


class FakeTraces:
    """The ttnn trace calls capture_prefill_chunk_traces makes, on a trace region of ``total`` bytes where each capture
    takes ``size`` bytes; capture number ``fail_at`` (from 1) raises as the region-size check does. A capture that
    raises takes nothing from the region (tt-metal's MeshTrace::populate_mesh_buffer counts a trace only once it fits).
    """

    BufferType = SimpleNamespace(TRACE="trace")

    def __init__(self, total, size, used=0, fail_at=None):
        self.total, self.size, self.used, self.fail_at = total, size, used, fail_at
        self.captures, self.live, self.released = 0, set(), []

    def synchronize_device(self, device):
        pass

    def begin_trace_capture(self, device, cq_id):
        self.captures += 1
        return self.captures

    def end_trace_capture(self, device, trace_id, cq_id):
        self.live.add(trace_id)
        if trace_id == self.fail_at:
            raise RuntimeError(
                "Creating trace buffers of size 2B on MeshDevice 1, but only 1B is allocated for trace region."
            )
        self.used += self.size

    def release_trace(self, device, trace_id):
        self.live.discard(trace_id)
        self.released.append(trace_id)

    def get_memory_view(self, device, buffer_type):
        assert buffer_type == "trace"
        return SimpleNamespace(num_banks=1, total_bytes_per_bank=self.total, total_bytes_allocated_per_bank=self.used)


def traced_model(monkeypatch, fake, count=3, first_estimate=100):
    monkeypatch.setattr(model_lib, "ttnn", fake)
    monkeypatch.setattr(model_lib, "CHUNK_TRACE_FIRST_ESTIMATE", first_estimate)
    model = TtMiMoModel.__new__(TtMiMoModel)
    model.mesh_device = object()
    model.chunk_traces = {k: {"id": None, "key": k} for k in range(count)}
    model.chunk_inputs, model.chunk_logits = {0: {}}, [object()]
    model._chunk_trace_body = lambda trace: None
    return model


def test_chunk_traces_that_fit_are_captured(monkeypatch):
    fake = FakeTraces(total=1000, size=100, used=500)
    model = traced_model(monkeypatch, fake)
    assert model.capture_prefill_chunk_traces()
    assert [t["id"] for t in model.chunk_traces.values()] == [1, 2, 3] and fake.used == 800


@pytest.mark.parametrize("fail_at", [1, 2], ids=["first", "second"])
def test_a_failed_chunk_capture_falls_back_to_eager(monkeypatch, fail_at):
    """A capture that throws (the region-size check, say) releases every later-chunk trace, frees their buffers and
    leaves later chunks eager; startup goes on."""
    fake = FakeTraces(total=1000, size=100, fail_at=fail_at)
    model = traced_model(monkeypatch, fake)
    assert not model.capture_prefill_chunk_traces()
    assert model.chunk_traces == {} and model.chunk_inputs == {} and model.chunk_logits is None
    assert fake.live == set(), "a captured or failed trace is still live"
    assert model._chunk_trace(0, 8192, torch.empty(CHUNK)) is None


def test_chunk_traces_stop_before_the_region_is_full(monkeypatch):
    """Two captures of 400 bytes fit in 1000 with 100 used; the third would not (100 left): every later-chunk trace is
    dropped before that capture is tried."""
    fake = FakeTraces(total=1000, size=400, used=100)
    model = traced_model(monkeypatch, fake)
    assert not model.capture_prefill_chunk_traces()
    assert fake.captures == 2 and fake.released == [1, 2] and fake.live == set()
    assert model.chunk_traces == {}


def test_the_first_chunk_capture_is_checked_too(monkeypatch):
    """Before the first capture of a replica there is no captured size to go by: it is checked against
    CHUNK_TRACE_FIRST_ESTIMATE, so a region without room for one chunk trace drops them without trying a capture."""
    fake = FakeTraces(total=1000, size=100, used=900)
    model = traced_model(monkeypatch, fake, first_estimate=200)
    assert not model.capture_prefill_chunk_traces()
    assert fake.captures == 0 and model.chunk_traces == {}


def rope_setup():
    setup = MiMoRotarySetup.__new__(MiMoRotarySetup)
    setup.prefill_cache, setup.prefill_pinned = None, set()
    return setup


class Failing:
    """A ttnn whose embedding (the first op of a traced chunk) raises."""

    TILE_LAYOUT, bfloat16 = "tile", "bf16"

    def embedding(self, *args, **kwargs):
        raise RuntimeError("out of memory")


@pytest.mark.parametrize("pinned", [False, True], ids=["eager", "pinned"])
def test_a_prefill_that_raises_leaves_no_rope_chunk_open(expect_error, pinned):
    """_prefill_rope ends every RoPE setup's chunk however the pass exits, so the next prefill can begin its own (an
    open chunk fails the next one: "prefill chunks do not nest")."""
    model = TtMiMoModel.__new__(TtMiMoModel)
    model.rope_setups = {"global": rope_setup(), "swa": rope_setup()}
    rope = {name: {(4096, CHUNK): ["cos", "sin"]} for name in model.rope_setups} if pinned else None
    with expect_error(RuntimeError, "boom"):
        with model._prefill_rope(rope):
            assert all(s.prefill_cache is not None for s in model.rope_setups.values())
            raise RuntimeError("boom")
    assert all(s.prefill_cache is None for s in model.rope_setups.values())
    with model._prefill_rope():  # a later prefill begins its chunk
        pass


def test_a_failed_chunk_trace_body_leaves_no_rope_chunk_open(monkeypatch, expect_error):
    """A traced chunk's body that raises (its warm pass, or inside a capture) leaves the RoPE setups closed."""
    monkeypatch.setattr(model_lib, "ttnn", Failing())
    model = TtMiMoModel.__new__(TtMiMoModel)
    model.rope_setups = {"global": rope_setup(), "swa": rope_setup()}
    model.chunk_inputs = {CHUNK: {"rope": {name: ["cos", "sin"] for name in model.rope_setups}, "tokens": None}}
    model.embedding = None
    trace = {"key": (CHUNK, 256, True, ALIGN, 0), "start": 2 * CHUNK}
    with expect_error(RuntimeError, "out of memory"):
        model._chunk_trace_body(trace)
    assert all(s.prefill_cache is None for s in model.rope_setups.values())


def optional_traced_model(monkeypatch, fake, optional):
    """traced_model with tuple keys (padded, k) and an ``optional`` flag per trace."""
    model = traced_model(monkeypatch, fake, count=0)
    model.chunk_traces = {}
    for k, opt in enumerate(optional):
        key = (256 if opt else CHUNK, k)
        model.chunk_traces[key] = {"id": None, "key": key, "optional": opt}
    model.chunk_inputs = {CHUNK: {}, 256: {}}
    return model


def test_optional_chunk_traces_that_do_not_fit_drop_only_themselves(monkeypatch):
    """The optional traces (the multimodal prompts' first and short later chunks) are captured after the others; when
    the region runs out only the optional ones left are dropped: the others stay captured and their chunks replay."""
    fake = FakeTraces(total=1000, size=300, used=100)
    model = optional_traced_model(monkeypatch, fake, [True, False, False, True])
    assert model.capture_prefill_chunk_traces()
    assert fake.captures == 2 and fake.released == [] and fake.used == 700
    assert sorted(t["optional"] for t in model.chunk_traces.values()) == [False, False]
    assert all(t["id"] is not None for t in model.chunk_traces.values())
    assert list(model.chunk_inputs) == [CHUNK], "the dropped traces' inputs are freed"


def test_a_failed_optional_capture_keeps_the_others(monkeypatch):
    fake = FakeTraces(total=10000, size=100, fail_at=3)
    model = optional_traced_model(monkeypatch, fake, [False, False, True, True])
    assert model.capture_prefill_chunk_traces()
    assert [t["id"] for t in model.chunk_traces.values()] == [1, 2] and fake.live == {1, 2}


def key_model(rep=2):
    model = TtMiMoModel.__new__(TtMiMoModel)
    model.block_size, model.pair, model.dflash = 64, None, None
    model.layers = [
        SimpleNamespace(is_swa=True, attn=SimpleNamespace(sliding_window=128, pair=None)),
        SimpleNamespace(is_swa=False, attn=SimpleNamespace(pair=None, split_users=True, kv_both_chips=False, rep=rep)),
    ]
    return model


def test_first_chunk_traces_key_on_the_length_only():
    """A first chunk (start 0) reads no window, no context and no owner mask: one traced first chunk per padded length
    serves every slot, and _chunk_trace finds it for a start-0 chunk (later chunks keep their slot parity)."""
    model = key_model()
    key = model._chunk_trace_key(1, 0, 1024)
    assert key == (1024, 128, False, 0, 0) == model._chunk_trace_key(0, 0, 1024)
    assert model._chunk_trace_key(1, 1024, 1024)[3:] == (128, 1)
    model.chunk_traces = {key: {"id": 7}}
    assert model._chunk_trace(3, 0, torch.empty(900))["id"] == 7
    assert model._chunk_trace(3, 0, torch.empty(1100)) is None, "1100 tokens pad to 1152"
    assert model._chunk_trace(3, 1024, torch.empty(900)) is None, "a later chunk has its own traces"


class Segment:
    """A row buffer segment (TtMiMoModel.trace_rows): only its shape is read on the host."""

    def __init__(self, name, rows):
        self.name, self.shape = name, (rows, 4)

    def __repr__(self):
        return self.name


def test_write_merge_points_the_gather_at_the_host_rows(monkeypatch):
    """A replay's merge inputs (_write_merge): row r of the pass gathers token row r, or row n + k of [token rows; row
    buffer] when it is the pass's k-th host row (the host rows are packed in pass order into the row buffer's
    segments, and only the segments they reach are uploaded). A replay without host rows resets the index to the
    identity once, and uploads nothing while it stays the identity."""
    uploads = []
    monkeypatch.setattr(model_lib.kv_lib, "replicated", lambda plan, t, dtype, device=True: t.clone())
    fake = SimpleNamespace(copy_host_to_device_tensor=lambda h, d: uploads.append((d, h)), uint32="u32", bfloat16="bf")
    monkeypatch.setattr(model_lib, "ttnn", fake)
    model = TtMiMoModel.__new__(TtMiMoModel)
    model.plan, model.cfg = None, SimpleNamespace(hidden_size=4)
    seg0, seg1, seg2 = Segment("seg0", 4), Segment("seg1", 4), Segment("seg2", 8)
    model.trace_rows = [seg0, seg1, seg2]
    merge = {"index": "index", "segments": 2, "identity": True}
    n = 8
    emb_a = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) + 1
    emb_b = -torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)
    rows_b = torch.tensor([False, True, True, False])
    # Users given out of order: rows go to the buffer in pass order.
    model._write_merge(merge, n, [(4, emb_b, rows_b), (0, emb_a, None), (6, None, None)])
    got = dict((str(d), h) for d, h in uploads)
    assert sorted(got) == ["index", "seg0", "seg1"], "the third segment is not reached"
    assert got["index"].reshape(-1).tolist() == [8, 9, 10, 3, 4, 11, 12, 7]
    packed = torch.cat([got["seg0"], got["seg1"]]).float()
    assert got["seg0"].dtype == torch.bfloat16 and got["seg0"].shape == (4, 4)
    assert torch.equal(packed[:3], emb_a) and torch.equal(packed[3:5], emb_b[1:3])
    assert not merge["identity"]
    uploads.clear()
    model._write_merge(merge, n, [(0, emb_a[:2], None)])
    assert sorted(str(d) for d, _ in uploads) == ["index", "seg0"], "two rows reach one segment"
    uploads.clear()
    model._write_merge(merge, n, [(0, None, None)])
    assert [d for d, _ in uploads] == ["index"] and uploads[0][1].reshape(-1).tolist() == list(range(n))
    uploads.clear()
    model._write_merge(merge, n, [])
    assert uploads == [] and merge["identity"]


def test_row_buffer_segments_cover_the_longest_pass(monkeypatch):
    """A pass of n rows reads the first segments that together hold n rows; a pass longer than the configured
    segments adds one of the missing rows (set up before any capture)."""
    monkeypatch.setattr(model_lib.kv_lib, "replicated", lambda plan, t, dtype, device=True: Segment("s", t.shape[0]))
    monkeypatch.setattr(model_lib, "ttnn", SimpleNamespace(bfloat16="bf"))
    model = TtMiMoModel.__new__(TtMiMoModel)
    model.plan, model.cfg = None, SimpleNamespace(hidden_size=4)
    model.trace_row_sizes, model.trace_rows = (256, 768), []
    assert model._trace_row_segments(100) == 1 and [t.shape[0] for t in model.trace_rows] == [256]
    assert model._trace_row_segments(1024) == 2 and [t.shape[0] for t in model.trace_rows] == [256, 768]
    assert model._trace_row_segments(1500) == 3 and [t.shape[0] for t in model.trace_rows] == [256, 768, 476]
    assert model._trace_row_segments(257) == 2
