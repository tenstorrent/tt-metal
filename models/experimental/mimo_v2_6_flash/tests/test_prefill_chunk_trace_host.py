# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of the traced later prefill chunks (TtMiMoModel.setup_prefill_chunk_traces; no device): a row longer
than one chunk is traced only when every chunk replays, vLLM's prefill_forward runs a row that would mix replays and
eager chunks all eager and after the lane's replays, and the traces are refused on a build whose chunked MLA prefill
takes no runtime start."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv
from models.experimental.mimo_v2_6_flash.tt import model as model_lib
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel

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
