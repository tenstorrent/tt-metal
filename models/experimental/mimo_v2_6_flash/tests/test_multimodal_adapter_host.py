# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of MiMoV2OmniForCausalLM.prefill_forward's routing (no device): a chunk holding an item's embedding
rows runs eagerly on its own with the item rows merged (never in a prefill batch, never replayed), a cached prefix in
front of an item is not recomputed in a batch, text rows and text-only steps run as in the text class, and malformed
multimodal input is served as text instead of raising out of prefill."""

from types import SimpleNamespace

import torch

from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel

HIDDEN = 8
IMAGE = 151655


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


def adapter(monkeypatch, cls=gv.MiMoV2OmniForCausalLM):
    monkeypatch.setattr(gv, "PREFILL_BATCH", True)
    monkeypatch.setattr(gv, "PREFILL_CHUNK_TRACE", True)
    monkeypatch.setattr(gv, "KV_PAIR_SPLIT", True)
    replica = Replica()
    gen = SimpleNamespace(models=[replica], dp=1, users_per_replica=4, cfg=SimpleNamespace(sliding_window=128))
    a = cls(gen, 1 << 16)
    a.layer_groups, a.pair_split = [0], True
    monkeypatch.setattr(a, "_layer_tables", lambda *args: None)
    monkeypatch.setattr(a, "_group_rows", lambda *args: {})
    if cls is gv.MiMoV2OmniForCausalLM:
        hf = SimpleNamespace(image_token_id=IMAGE, vision_config={"spatial_merge_size": 2})
        a.setup_multimodal(None, hf, backends=[Backend()])
    return a, replica


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


def test_bad_input_is_served_as_text(monkeypatch):
    a, replica = adapter(monkeypatch)
    tokens = torch.zeros(3, 300, dtype=torch.int32)
    mm_items = [
        [{"modality": "smell", "identifier": "x", "offset": 0, "length": 4}],  # not a served modality
        [{**image(1, 10), "length": 3}],  # the encoder gives 8 rows for 3 placeholders
        [image(2, 4), image(3, 6)],  # overlapping items
    ]
    a.prefill_forward(tokens, [0, 0, 0], [200, 200, 200], None, None, empty_slots=[0, 1, 2], mm_items=mm_items)
    # Rows 0 and 2 are text (batched); row 1 keeps its placeholder tokens, eagerly (its item still has rows there).
    assert replica.passes[0] == ("batch", [(0, 200), (2, 200)])
    assert replica.passes[1][:2] == ("row", 1) and replica.passes[1][4] is None


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
