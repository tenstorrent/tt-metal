# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of MiMoV2OmniForCausalLM.prefill_forward's routing (no device): on replicas whose traced passes take no
multimodal rows a chunk holding an item's embedding rows runs eagerly on its own with the item rows merged (never in a
prefill batch, never replayed), and on replicas whose traced passes take them (TtMiMoModel.enable_trace_embeds) it
batches and replays as a text chunk does; a cached prefix in front of an item is not recomputed in a batch, text rows
and text-only steps run as in the text class, and malformed multimodal input is served as text instead of raising out
of prefill."""

from types import SimpleNamespace

import torch

from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel

HIDDEN = 8
IMAGE = 151655
EOS = 7  # the fake replica's logits have 8 columns


class Replica:
    """Records each prefill pass: ("batch", [(slot, tokens)]) or ("row", slot, start, tokens, embed rows or None)."""

    PREFILL_BATCH_MAX_PADDED = TtMiMoModel.PREFILL_BATCH_MAX_PADDED
    chunk_traced = TtMiMoModel.chunk_traced

    def __init__(self):
        self.max_prefill_chunk = 4096
        self._chunk_trace = lambda slot, start, tokens: None
        self.mtp = self.dflash = self.decode_trace = None
        self.cfg = SimpleNamespace(hidden_size=HIDDEN)
        self.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: 0)
        self.passes = []
        self.embeds = {}

    def set_vllm_page_tables(self, rows):
        pass

    def prefill_enqueue(self, chunk, slot, start_pos=0, history_continues=False, draft_group=0, allow_trace=True, **mm):
        rows = mm.get("embed_rows")
        self.passes.append(("row", slot, start_pos, chunk.numel(), None if rows is None else rows.nonzero().flatten()))
        if "embeds" in mm:
            assert not allow_trace, "a multimodal chunk must not replay a trace"
            self.embeds[slot] = mm["embeds"]
        return 1

    def prefill_enqueue_batch(self, chunks, layers=None):
        self.passes.append(("batch", [(c[1], c[0].numel()) for c in chunks]))
        return len(chunks)

    def prefill_read(self, handle):
        return [torch.zeros(handle, 8)]


class Backend:
    def __init__(self):
        self.calls = 0

    def embed_image(self, pixel_values, grid_thw):
        self.calls += 1
        n = int(torch.as_tensor(grid_thw).prod(-1).sum()) // 4
        return torch.full((n, HIDDEN), float(pixel_values[0, 0]), dtype=torch.bfloat16)

    embed_video = embed_image

    def embed_audio(self, mel):
        self.calls += 1
        return torch.full((int(mel.shape[0]) // 4, HIDDEN), float(mel[0, 0]), dtype=torch.bfloat16)


def adapter(monkeypatch, cls=gv.MiMoV2OmniForCausalLM, replica=None):
    monkeypatch.setattr(gv, "PREFILL_BATCH", True)
    monkeypatch.setattr(gv, "PREFILL_CHUNK_TRACE", True)
    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", True)
    replica = Replica() if replica is None else replica
    gen = SimpleNamespace(models=[replica], dp=1, users_per_replica=4, cfg=SimpleNamespace(sliding_window=128))
    a = cls(gen, 1 << 16)
    a.layer_groups, a.pair_split = [0], True
    monkeypatch.setattr(a, "_layer_tables", lambda *args: None)
    monkeypatch.setattr(a, "_group_rows", lambda *args: {})
    if cls is gv.MiMoV2OmniForCausalLM:
        hf = SimpleNamespace(image_token_id=IMAGE, vision_config={"spatial_merge_size": 2}, eos_token_id=EOS)
        a.setup_multimodal(None, hf, backends=[Backend()])
    return a, replica


class TracedReplica(Replica):
    """A replica whose traced passes take multimodal rows (TtMiMoModel.enable_trace_embeds), with traced later chunks
    of the padded lengths in ``later``. Records each pass with whether it replays and the rows it merges."""

    def __init__(self, later=(128, 256)):
        super().__init__()
        self.trace_rows = None
        self._chunk_trace = lambda slot, start, tokens: (
            object() if start > 0 and gv.prefill_padded_len(tokens.numel(), start, True) in later else None
        )

    def enable_trace_embeds(self):
        self.trace_rows = []

    def prefill_enqueue(self, chunk, slot, start_pos=0, history_continues=False, draft_group=0, allow_trace=True, **mm):
        rows = mm.get("embed_rows")
        replay = allow_trace and self._chunk_trace(slot, start_pos, chunk) is not None
        merged = None if rows is None else rows.nonzero().flatten()
        self.passes.append(("row", slot, start_pos, chunk.numel(), merged, "replay" if replay else "eager"))
        if "embeds" in mm:
            self.embeds[slot] = mm["embeds"]
        return 1

    def prefill_enqueue_batch(self, chunks, layers=None):
        merged = [None if len(c) < 4 or c[3] is None else c[4].nonzero().flatten().tolist() for c in chunks]
        self.passes.append(("batch", [(c[1], c[0].numel()) for c in chunks], merged))
        for c in chunks:
            if len(c) > 3 and c[3] is not None:
                self.embeds[c[1]] = c[3]
        return len(chunks)


def test_multimodal_rows_batch_and_replay_when_traces_take_them(monkeypatch):
    """With traced passes that take multimodal rows (MIMO_MM_TRACE), a chunk with an item's rows runs as a text chunk
    does: a first chunk batched with the lane's other first chunks of its padded length (each user's rows merged),
    a later chunk replaying its trace with its rows; a cached prefix in front of an item is still not recomputed in a
    batch (it would encode the cached items again)."""
    monkeypatch.setattr(gv, "PREFILL_TRACE", True)
    monkeypatch.setattr(gv, "MM_TRACE", True)
    monkeypatch.setattr(gv, "TIMING", True)
    replica = TracedReplica()
    a, _ = adapter(monkeypatch, replica=replica)
    assert a._mm_traced() and replica.trace_rows == []
    tokens = torch.zeros(4, 512, dtype=torch.int32)
    # Row 0: text (300 tokens: padded 384). Row 1: an image at [10, 18) in a 300-token first chunk. Row 2: a prefix
    # hit at 256 with an image at [300, 308) in its chunk (later, 144 tokens: padded 256, traced). Row 3: text, 100.
    mm_items = [None, [image(1, 10)], [image(2, 300)], []]
    a.prefill_forward(
        tokens, [0, 0, 256, 0], [300, 300, 400, 100], None, None, empty_slots=[0, 1, 2, 3], mm_items=mm_items
    )
    replays = [p for p in replica.passes if p[0] == "row"]
    batches = {tuple(s for s, _ in p[1]): p for p in replica.passes if p[0] == "batch"}
    assert set(batches) == {(0, 1), (3,)}
    assert batches[(0, 1)][2] == [None, list(range(10, 18))]
    assert torch.equal(replica.embeds[1][10:18], torch.full((8, HIDDEN), 1.0, dtype=torch.bfloat16))
    assert len(replays) == 1 and replays[0][1:4] == (2, 256, 144) and replays[0][5] == "replay"
    assert replays[0][4].tolist() == list(range(44, 52))
    assert replica.passes[0] is replays[0], "a lane's replayed later chunks go before its batches"


def test_multimodal_rows_stay_eager_without_trace_embeds(monkeypatch):
    """MIMO_MM_TRACE=0: the replicas' traced passes take no rows, so a multimodal chunk runs eagerly on its own."""
    monkeypatch.setattr(gv, "MM_TRACE", False)
    replica = TracedReplica()
    a, _ = adapter(monkeypatch, replica=replica)
    assert not a._mm_traced() and replica.trace_rows is None
    tokens = torch.zeros(1, 512, dtype=torch.int32)
    a.prefill_forward(tokens, [256], [400], None, None, empty_slots=[0], mm_items=[[image(2, 300)]])
    assert replica.passes[-1][5] == "eager"


def image(tag, offset, h=4, w=8, stripped=False):
    grid = torch.tensor([1, h, w])
    data = None if stripped else {"pixel_values": torch.full((h * w, 6), float(tag)), "image_grid_thw": grid}
    return {"modality": "image", "identifier": f"img{tag}", "offset": offset, "length": h * w // 4, "data": data}


def test_multimodal_rows_run_eagerly_with_their_embeddings(monkeypatch):
    a, replica = adapter(monkeypatch)
    tokens = torch.zeros(4, 512, dtype=torch.int32)
    # Row 0: a text prompt (batched). Row 1: an image at [10, 18) in its first chunk. Row 2: a prefix hit at 256 whose
    # image at [20, 28) lies in the cached prefix (stripped): its chunk has no item row. Row 3: text, batched with row 0.
    mm_items = [None, [image(1, 10)], [image(2, 20, stripped=True)], []]
    a.prefill_forward(
        tokens, [0, 0, 256, 0], [300, 300, 400, 290], None, None, empty_slots=[0, 1, 2, 3], mm_items=mm_items
    )
    assert replica.passes[0] == ("batch", [(0, 300), (3, 290)])
    rows = {p[1]: p for p in replica.passes[1:]}
    assert set(rows) == {1, 2}
    assert rows[1][2:4] == (0, 300) and rows[1][4].tolist() == list(range(10, 18))
    assert torch.equal(replica.embeds[1][10:18], torch.full((8, HIDDEN), 1.0, dtype=torch.bfloat16))
    # Row 2 is not recomputed from 0 in a batch (that would need the stripped image): it continues at 256, as text.
    assert rows[2][2:5] == (256, 144, None)


def test_text_only_steps_match_the_text_class(monkeypatch):
    tokens = torch.zeros(3, 600, dtype=torch.int32)
    args = (tokens, [0, 256, 0], [300, 600, 100], None, None)
    passes = []
    for cls in (gv.MiMoV2ForCausalLM, gv.MiMoV2OmniForCausalLM):
        a, replica = adapter(monkeypatch, cls)
        a.prefill_forward(*args, empty_slots=[0, 1, 2], mm_items=[None, None, None])
        passes.append(replica.passes)
    assert passes[0] == passes[1]


def test_image_lists_fallback_and_cache(monkeypatch):
    """The plugin's image-only kwargs: items found by their placeholder runs; the next chunk of the request, moved
    back into the image, reuses its cached embeddings."""
    a, replica = adapter(monkeypatch)
    tokens = torch.zeros(1, 400, dtype=torch.int32)
    tokens[0, 100:108] = IMAGE
    grid = torch.tensor([1, 4, 8])
    kw = dict(pixel_values=[[torch.full((32, 6), 5.0)]], image_grid_thw=[[grid]])
    a.prefill_forward(tokens, [0], [104], None, None, empty_slots=[0], **kw)  # a chunk ending inside the image
    assert replica.passes[-1][4].tolist() == list(range(100, 104))
    calls = a.mm_encoders[0].backend.calls
    a.prefill_forward(tokens, [104], [400], None, None, empty_slots=[0], **kw)  # moved back to 0 (pair split: 128)
    assert replica.passes[-1][2:4] == (0, 400) and replica.passes[-1][4].tolist() == list(range(100, 108))
    assert a.mm_encoders[0].backend.calls == calls, "the image was encoded again"


def test_bad_input_fails_the_row(monkeypatch):
    """Input that cannot be served is never answered from placeholder embeddings (MIMO_MM_STRICT): the row's first
    token is forced to the end of the turn and take_prefill_errors() reports it; nothing raises out of prefill."""
    monkeypatch.setattr(gv, "MM_STRICT", True)
    a, replica = adapter(monkeypatch)
    tokens = torch.zeros(4, 300, dtype=torch.int32)
    mm_items = [
        [{"modality": "smell", "identifier": "x", "offset": 0, "length": 4}],  # not a served modality
        [{**image(1, 10), "length": 3}],  # the encoder gives 8 rows for 3 placeholders
        [image(2, 4), image(3, 6)],  # overlapping items
        None,  # a text request in the same step is served as usual
    ]
    logits = a.prefill_forward(tokens, [0, 0, 0, 0], [200] * 4, None, None, empty_slots=[0, 1, 2, 3], mm_items=mm_items)
    errors = a.take_prefill_errors()
    assert sorted(errors) == [0, 1, 2] and a.take_prefill_errors() == {}
    for row in (0, 1, 2):
        assert int(logits[row].reshape(-1).argmax()) == EOS and int(torch.isinf(logits[row]).sum()) == 7
    assert not torch.isinf(logits[3]).any()
    # A later request whose cached prefix (start 128) holds the item that failed is failed too: that KV is placeholder.
    tokens = torch.zeros(1, 400, dtype=torch.int32)
    a.prefill_forward(tokens, [128], [400], None, None, empty_slots=[1], mm_items=[[{**image(1, 10), "length": 3}]])
    assert list(a.take_prefill_errors()) == [0]


def test_bad_input_served_degraded_when_not_strict(monkeypatch):
    monkeypatch.setattr(gv, "MM_STRICT", False)
    a, replica = adapter(monkeypatch)
    tokens = torch.zeros(1, 300, dtype=torch.int32)
    items = [[{**image(1, 10), "length": 3}]]
    logits = a.prefill_forward(tokens, [0], [200], None, None, empty_slots=[0], mm_items=items)
    assert list(a.take_prefill_errors()) == [0] and not torch.isinf(logits).any()
    assert replica.passes[-1][:2] == ("row", 0) and replica.passes[-1][4] is None


def test_items_straddling_chunks_and_slot_reuse(monkeypatch):
    """An item cut by the chunk end (vLLM's chunked multimodal input) is encoded once and scattered over both chunks;
    when its slot takes another request, its cached embeddings go unless another slot's request holds the item."""
    a, replica = adapter(monkeypatch)
    tokens = torch.zeros(2, 512, dtype=torch.int32)
    big = image(4, 120, h=8, w=16)  # 32 rows at [120, 152)
    a.prefill_forward(tokens[:1], [0], [128], None, None, empty_slots=[0], mm_items=[[big]])
    assert replica.passes[-1][4].tolist() == list(range(120, 128))
    calls = a.mm_encoders[0].backend.calls
    a.prefill_forward(tokens[:1], [128], [300], None, None, empty_slots=[0], mm_items=[[big]])
    assert replica.passes[-1][2] == 128 and replica.passes[-1][4].tolist() == list(range(0, 24))
    assert a.mm_encoders[0].backend.calls == calls and "img4" in a.mm_cache
    # Slot 1 takes a request with the same image; slot 0 a text request: the image stays (slot 1 holds it).
    a.prefill_forward(tokens, [0, 0], [200, 200], None, None, empty_slots=[0, 1], mm_items=[None, [big]])
    assert "img4" in a.mm_cache
    # Slot 1 takes a text request too: nothing holds the image any more.
    a.prefill_forward(tokens[:1], [0], [100], None, None, empty_slots=[1], mm_items=[None])
    assert "img4" not in a.mm_cache


def test_an_item_that_encodes_again_is_no_longer_failed(monkeypatch):
    """One failed encoder pass (a fragmented DRAM, say) does not fail the item's requests for the life of the engine:
    while only the failure is known, a request whose cached prefix covers the item is failed (that KV is placeholder);
    once a request encodes the item and computes its positions itself, the mark is gone and prefix hits on it are
    served."""
    monkeypatch.setattr(gv, "MM_STRICT", True)
    a, replica = adapter(monkeypatch)
    backend = a.mm_encoders[0].backend
    good, state = backend.embed_image, {"fail": 1}

    def flaky(pixel_values, grid_thw):
        if state["fail"]:
            state["fail"] -= 1
            raise RuntimeError("out of memory")
        return good(pixel_values, grid_thw)

    backend.embed_image = flaky
    tokens = torch.zeros(1, 400, dtype=torch.int32)
    kw = dict(empty_slots=[0], mm_items=[[image(1, 10)]])
    a.prefill_forward(tokens, [0], [200], None, None, **kw)
    assert list(a.take_prefill_errors()) == [0] and "img1" in a.mm_failed_ids
    a.prefill_forward(tokens, [128], [400], None, None, **kw)  # a prefix hit over the failed item
    assert list(a.take_prefill_errors()) == [0] and "img1" in a.mm_failed_ids
    a.prefill_forward(tokens, [0], [200], None, None, **kw)  # recomputed from the start: the encoder works again
    assert a.take_prefill_errors() == {} and "img1" not in a.mm_failed_ids and "img1" not in a.mm_encoders[0].failed
    a.prefill_forward(tokens, [128], [400], None, None, **kw)  # the prefix now holds the item's real KV
    assert a.take_prefill_errors() == {}
    # A failure keeps only the recent identifiers, and never an object id.
    recent = gv.mm.RecentSet(capacity=3)
    recent.update("abcd")
    recent.add("b")
    assert list(recent) == ["c", "d", "b"] and "a" not in recent and len(recent) == 3


def test_programs_an_encoder_compiles_are_recaptured_around_before_any_pass(monkeypatch):
    """An encoder that compiles programs while serving (an input size first seen) puts their buffers where the
    replica's traces write on replay: the traces are captured again before this step's passes (and their replays),
    not only after them, and once."""
    events = []

    class Compiling(TracedReplica):
        def __init__(self):
            super().__init__()
            self.programs = 10
            self.decode_trace = object()
            self.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: self.programs)
            for name in (
                "release_spec_traces",
                "release_decode_trace",
                "release_prefill_chunk_traces",
                "recapture_prefill_batch_traces",
                "capture_prefill_chunk_traces",
            ):
                setattr(self, name, lambda: None)

        def capture_decode_trace(self):
            events.append("recapture")

        def set_vllm_page_tables(self, rows):
            events.append("page tables")

        def prefill_enqueue_batch(self, chunks, layers=None):
            events.append("pass")
            return super().prefill_enqueue_batch(chunks, layers)

        def prefill_enqueue(self, *args, **kwargs):
            events.append("pass")
            return super().prefill_enqueue(*args, **kwargs)

    a, replica = adapter(monkeypatch, replica=Compiling())
    backend = a.mm_encoders[0].backend
    good = backend.embed_image

    def compiling(pixel_values, grid_thw):
        events.append("encode")
        replica.programs += 3  # a pass size the ViT had not compiled
        return good(pixel_values, grid_thw)

    backend.embed_image = compiling
    tokens = torch.zeros(2, 300, dtype=torch.int32)
    before = a.decode_recaptures
    a.prefill_forward(tokens, [0, 0], [200, 200], None, None, empty_slots=[0, 1], mm_items=[[image(1, 10)], None])
    assert events[:2] == ["encode", "recapture"] and "pass" in events[2:], events
    assert events.count("recapture") == 1 and a.decode_recaptures == before + 1  # not again after the passes
    # The cached image compiles nothing: no recapture.
    events.clear()
    a.prefill_forward(tokens, [0, 0], [200, 200], None, None, empty_slots=[0, 1], mm_items=[[image(1, 10)], None])
    assert "recapture" not in events and "encode" not in events and "pass" in events
