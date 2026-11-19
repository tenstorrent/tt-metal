# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post-splice embeddings on one chip and on a galaxy: TtParallelEmbedding over an expanded prompt with image rows spliced chunk by
chunk through TtPrefillRuntime._media_splice, against a torch merge and against Hugging Face minimax_m3_vl's merge.

Run: pytest models/demos/minimax_m3/tests/unit/test_media_splice_vs_ref.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.multimedia_store.reader import FetchRequest, Missing
from models.demos.common.prefill.chunk_layout import pack_forward_tokens
from models.demos.common.prefill.media_embedding import plan_media_splice, table_height
from models.demos.common.prefill.prefill_chunk import MediaRef, PrefillChunk
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.multimodal import format as media_format
from models.demos.minimax_m3.multimodal.stub_reader import StubEmbeddingReader
from models.demos.minimax_m3.tt import tt_prefill_runtime
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.parallel_embedding import TtParallelEmbedding
from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

VOCAB, HIDDEN, CHUNK = 1024, 256, 128
IMAGE_TOKEN, VISION_START, VISION_END = 1000, 1001, 1002


class Store:
    """Serves each image's embeddings as a stored value in format.py's layout, keyed by media key."""

    def __init__(self, embs: dict):
        self.values = {key: self.value(rows) for key, rows in embs.items()}

    @staticmethod
    def value(rows: torch.Tensor) -> bytes:
        buf = memoryview(bytearray(media_format.stored_nbytes(*rows.shape)))
        media_format.Header("image").write(buf)
        media_format.embeddings(buf, rows.shape[1])[:] = rows.to(torch.bfloat16).view(torch.int16).numpy()
        return bytes(buf)

    def fetch_batch(self, requests) -> list:
        return [memoryview(self.values[FetchRequest(*r).media_key]) for r in requests]


class SharedBufferStore(Store):
    """Like the store's reads from another host: each fetch_batch overwrites one buffer that its views point into."""

    def __init__(self, embs: dict):
        super().__init__(embs)
        self.buf = bytearray(1 << 22)
        self.calls = 0

    def fetch_batch(self, requests) -> list:
        self.calls += 1
        self.buf[:] = bytes(len(self.buf))
        views, at = [], 0
        for r in requests:
            value = self.values[FetchRequest(*r).media_key]
            self.buf[at : at + len(value)] = value
            views.append(memoryview(self.buf)[at : at + len(value)])
            at += len(value)
        return views


def runtime_with(store, *, sp, chunk_size, mesh_shape, mesh_device=None, hidden=HIDDEN, shard_vocab_on_sp=True):
    return SimpleNamespace(
        config=SimpleNamespace(
            sp_factor=sp,
            chunk_size=chunk_size,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
        ),
        hf_config=SimpleNamespace(hidden_size=hidden),
        media_embedding_reader=store,
        mesh_device=mesh_device,
        model=SimpleNamespace(embedding=SimpleNamespace(shard_vocab_on_sp=shard_vocab_on_sp)),
        _media_splice_buffers=None,
    )


def embedding(mesh_device, table, shard_vocab_on_sp):
    mesh_shape = tuple(mesh_device.shape)
    return TtParallelEmbedding(
        mesh_device,
        VOCAB,
        HIDDEN,
        MeshConfig(mesh_shape, tp=mesh_shape[1]),
        CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear),
        torch_weight=table,
        dtype=ttnn.bfloat16,
        shard_vocab_on_sp=shard_vocab_on_sp,
    )


def device_tokens(mesh_device, tokens):
    return ttnn.from_torch(
        tokens.reshape(1, 1, -1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None)),
    )


def splice_chunks(mesh_device, table, tokens, images, store, first_chunk_start, shard_vocab_on_sp=False):
    """Embeddings of tokens[first_chunk_start:] after splicing, prefilled CHUNK tokens at a time.
    images: (start, num_tokens, key) per image."""
    mesh_shape = tuple(mesh_device.shape)
    emb = embedding(mesh_device, table, shard_vocab_on_sp)
    runtime = runtime_with(
        store,
        sp=mesh_shape[0],
        chunk_size=CHUNK,
        mesh_shape=mesh_shape,
        mesh_device=mesh_device,
        shard_vocab_on_sp=shard_vocab_on_sp,
    )
    out = []
    for start in range(first_chunk_start, len(tokens), CHUNK):
        end = min(start + CHUNK, len(tokens))
        chunk_tokens = torch.zeros(CHUNK, dtype=torch.int32)
        chunk_tokens[: end - start] = tokens[start:end]
        refs = [MediaRef(0, s, s + n, key) for s, n, key in images if s < end and s + n > start]
        chunks = [PrefillChunk(0, start, end)]
        media = TtPrefillRuntime._media_splice(runtime, chunks, refs)
        x = emb.forward(
            device_tokens(mesh_device, chunk_tokens), media_fn=(lambda: media) if media is not None else None
        )
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)
        # Inverts activation_mapper. A replicated residual repeats hidden once per TP rank, so keep the first.
        rows = ttnn.to_torch(
            x, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 3))
        ).reshape(CHUNK, -1)[:, :HIDDEN]
        out.append(rows[: end - start])
        ttnn.deallocate(x)
    return torch.cat(out)


def image_mask(images, start, end):
    positions = torch.arange(start, end)
    mask = torch.zeros(end - start, dtype=torch.bool)
    for s, n, _ in images:
        mask |= (positions >= s) & (positions < s + n)
    return mask


def check(got, want, images, first_chunk_start, seq_len):
    is_image = image_mask(images, first_chunk_start, seq_len)
    assert torch.equal(got[~is_image], want[~is_image]), "a text position changed"
    assert torch.equal(got[is_image], want[is_image]), "an image row is not its embedding"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (2, 4), (8, 4)])
@pytest.mark.parametrize("sharded_residual", ["1", "0"], ids=["tp_sharded", "replicated"])
@pytest.mark.parametrize("shard_vocab_on_sp", [True, False], ids=["vocab_sharded_2d", "by_position_1d"])
@pytest.mark.parametrize(
    "seq_len,layout,first_chunk_start",
    [
        pytest.param(128, [(10, 40)], 0, id="one_image"),
        pytest.param(128, [(3, 20), (40, 30), (100, 12)], 0, id="several_images"),
        pytest.param(384, [(100, 200)], 0, id="image_across_chunks"),
        pytest.param(384, [(64, 200)], 128, id="resume_mid_image"),
        pytest.param(256, [], 0, id="text_only"),
    ],
)
def test_post_splice_embeddings(
    mesh_device, device_params, monkeypatch, sharded_residual, shard_vocab_on_sp, seq_len, layout, first_chunk_start
):
    monkeypatch.setenv("M3_SHARDED_RESIDUAL", sharded_residual)
    torch.manual_seed(0)
    table = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    tokens = torch.randint(0, VOCAB, (seq_len,), dtype=torch.int32)
    images = [(s, n, bytes([i + 1]) * 32) for i, (s, n) in enumerate(layout)]
    embs = {key: torch.randn(n, HIDDEN).to(torch.bfloat16) for _, n, key in images}
    want = F.embedding(tokens.long(), table)
    for s, n, key in images:
        want[s : s + n] = embs[key]

    got = splice_chunks(mesh_device, table, tokens, images, Store(embs), first_chunk_start, shard_vocab_on_sp)
    check(got, want[first_chunk_start:], images, first_chunk_start, seq_len)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (2, 4), (8, 4)])
@pytest.mark.parametrize("shard_vocab_on_sp", [True, False], ids=["vocab_sharded_2d", "by_position_1d"])
def test_packed_forward(mesh_device, device_params, shard_vocab_on_sp):
    """Two slots' chunks in one forward: each position gets its image row or its token embedding, in the
    forward's chip-major token order."""
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    table = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    chunks = [PrefillChunk(0, 0, CHUNK), PrefillChunk(1, CHUNK, 2 * CHUNK)]
    tokens = torch.randint(0, VOCAB, (2, CHUNK), dtype=torch.int32)
    images = {bytes([1]) * 32: (0, 10, 40), bytes([2]) * 32: (1, CHUNK + 100, 28)}
    embs = {key: torch.randn(n, HIDDEN).to(torch.bfloat16) for key, (_, _, n) in images.items()}
    refs = [MediaRef(k, s, s + n, key) for key, (k, s, n) in images.items()]
    runtime = runtime_with(
        Store(embs),
        sp=mesh_shape[0],
        chunk_size=CHUNK,
        mesh_shape=mesh_shape,
        mesh_device=mesh_device,
        shard_vocab_on_sp=shard_vocab_on_sp,
    )
    cells = pack_forward_tokens(
        [[(k, c.actual_start + i) for i in range(CHUNK)] for k, c in enumerate(chunks)],
        [c.actual_start for c in chunks],
        mesh_shape[0],
    )
    ids = torch.tensor([tokens[k, p - chunks[k].actual_start] for k, p in cells], dtype=torch.int32)
    want = F.embedding(ids.long(), table)
    for j, (k, p) in enumerate(cells):
        for key, (rk, s, n) in images.items():
            if rk == k and s <= p < s + n:
                want[j] = embs[key][p - s]

    media = TtPrefillRuntime._media_splice(runtime, chunks, refs)
    x = embedding(mesh_device, table, shard_vocab_on_sp).forward(
        device_tokens(mesh_device, ids), media_fn=lambda: media
    )
    got = ttnn.to_torch(
        x, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 3))
    ).reshape(len(cells), -1)[:, :HIDDEN]
    assert torch.equal(got, want)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (2, 4), (8, 4)])
@pytest.mark.parametrize("shard_vocab_on_sp", [True, False], ids=["vocab_sharded_2d", "by_position_1d"])
def test_same_bucket_compiles_once(mesh_device, device_params, shard_vocab_on_sp):
    """Image counts that snap to the same table height reuse the program — what table_height is for.

    One 10-token image and two totalling 20 both snap to a 32-row table at sp 1, 2 and 8. Their exact
    heights differ (11 vs 21, 6 vs 11, 3 vs 4), so without the snapping the second forward would compile
    its own embedding and where."""
    torch.manual_seed(0)
    table = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    tokens = torch.randint(0, VOCAB, (CHUNK,), dtype=torch.int32)
    one = [(10, 10, bytes([1]) * 32)]
    two = [(10, 12, bytes([2]) * 32), (40, 8, bytes([3]) * 32)]
    embs = {key: torch.randn(n, HIDDEN).to(torch.bfloat16) for _, n, key in one + two}

    splice_chunks(mesh_device, table, tokens, one, Store(embs), 0, shard_vocab_on_sp=shard_vocab_on_sp)
    entries = mesh_device.num_program_cache_entries()
    splice_chunks(mesh_device, table, tokens, two, Store(embs), 0, shard_vocab_on_sp=shard_vocab_on_sp)

    assert mesh_device.num_program_cache_entries() == entries, "a same-bucket image count recompiled"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (2, 4), (8, 4)])
@pytest.mark.parametrize("shard_vocab_on_sp", [True, False], ids=["vocab_sharded_2d", "by_position_1d"])
def test_warm_media_leaves_nothing_to_compile(mesh_device, device_params, shard_vocab_on_sp):
    """After compile()'s media warm-up, serving images from one token up to a whole chunk compiles no program."""
    sp, chunk = mesh_device.shape[0], 2048
    torch.manual_seed(0)
    emb = embedding(mesh_device, torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16), shard_vocab_on_sp)
    sizes = [1, 50, 300, 1000, chunk - 7]
    embs = {bytes([i + 1]) * 32: torch.randn(n, HIDDEN).to(torch.bfloat16) for i, n in enumerate(sizes)}
    runtime = runtime_with(
        Store(embs),
        sp=sp,
        chunk_size=chunk,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        shard_vocab_on_sp=shard_vocab_on_sp,
    )
    runtime.make_forward_input = lambda tokens, starts: device_tokens(
        mesh_device, torch.tensor(tokens[0], dtype=torch.int32)
    )
    runtime._embed_tokens = lambda inp, media_fn: emb.forward(inp, media_fn=media_fn)
    runtime._media_splice = lambda chunks, refs: TtPrefillRuntime._media_splice(runtime, chunks, refs)
    TtPrefillRuntime._warm_media(runtime, chunk, 1)
    entries = mesh_device.num_program_cache_entries()

    chunks = [PrefillChunk(0, 0, chunk)]
    for key, rows in embs.items():
        refs = [MediaRef(0, 7, 7 + len(rows), key)]
        tokens = device_tokens(mesh_device, torch.randint(0, VOCAB, (chunk,), dtype=torch.int32))
        ttnn.deallocate(emb.forward(tokens, media_fn=lambda: runtime._media_splice(chunks, refs)))

    assert mesh_device.num_program_cache_entries() == entries, "a served image size compiled a program"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (2, 4), (8, 4)])
def test_full_size_tables_refill_in_place(mesh_device, device_params):
    """Two forwards at one table height reuse its host table. At full hidden size the table is over the 32 MB above
    which copy_host_to_device_tensor takes its pinned, blocking path where the device can pin; each forward's device
    table must hold its own image rows, on the right chip for every SP row and TP column."""
    sp, tp = tuple(mesh_device.shape)
    hidden, chunk = 6144, 4096
    chunks = [PrefillChunk(0, 0, chunk)]
    embs = {bytes([i]) * 32: torch.randn(n, hidden).to(torch.bfloat16) for i, n in ((1, 3000), (2, 2900))}
    runtime = runtime_with(
        Store(embs), sp=sp, chunk_size=chunk, mesh_shape=(sp, tp), mesh_device=mesh_device, hidden=hidden
    )
    shapes = []
    for key, rows in embs.items():
        refs = [MediaRef(0, 0, len(rows), key)]
        media = TtPrefillRuntime._media_splice(runtime, chunks, refs)
        got = ttnn.to_torch(
            media.table, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(0, 1))
        )
        shapes.append(got.shape)
        got = got.view(sp, tp, -1, hidden // tp).transpose(1, 2).reshape(sp, -1, hidden)
        assert got.shape[1] * sp * hidden * 2 > 32 << 20, "the table is too small to reach the pinned write path"
        assert not got[:, 0].any(), "the sentinel row must stay zero"
        u16 = lambda r: embs[r.media_key].view(torch.int16).numpy().view(np.uint16)
        for sp_row, slot, want in plan_media_splice(refs, chunks, u16, sp, chunk // sp).runs:
            assert torch.equal(
                got[sp_row, slot : slot + len(want)], torch.from_numpy(want.view(np.int16)).view(torch.bfloat16)
            )
    assert shapes[0] == shapes[1], "the two forwards must share one table height to reuse its host table"


@pytest.mark.parametrize("shard_vocab_on_sp", [True, False], ids=["vocab_sharded_2d", "by_position_1d"])
def test_warm_media_reaches_packed_forwards(monkeypatch, shard_vocab_on_sp):
    """compile() warms every table height a packed forward can reach: images can fill all its chunks, not just one."""
    monkeypatch.setattr(tt_prefill_runtime.ttnn, "deallocate", lambda t: None)
    sp, chunk, n_chunks = 2, CHUNK, 2
    zeros = lambda r: np.zeros((r.num_tokens, HIDDEN), np.uint16)
    heights = []
    runtime = runtime_with(None, sp=sp, chunk_size=chunk, mesh_shape=(sp, 1), shard_vocab_on_sp=shard_vocab_on_sp)
    runtime.make_forward_input = lambda tokens, starts: None
    runtime._embed_tokens = lambda inp, media_fn: media_fn()
    runtime._media_splice = lambda chunks, refs: heights.append(
        plan_media_splice(refs, chunks, zeros, sp, chunk // sp, not shard_vocab_on_sp).height
    )
    runtime._media_splice_buffers = SimpleNamespace(release_tables=lambda: None)
    TtPrefillRuntime._warm_media(runtime, chunk, n_chunks)
    assert heights[-1] == table_height(n_chunks * chunk // sp + 1)


@pytest.mark.parametrize("shard_vocab_on_sp", [True, False], ids=["vocab_sharded_2d", "by_position_1d"])
def test_one_batched_fetch_per_forward(shard_vocab_on_sp):
    """A packed forward naming one image in two chunks reads every value in one fetch, before the next fetch could
    overwrite it."""
    sp, chunk = 2, 128
    a, b = bytes([1]) * 32, bytes([2]) * 32
    embs = {a: torch.randn(100, HIDDEN).to(torch.bfloat16), b: torch.randn(20, HIDDEN).to(torch.bfloat16)}
    chunks = [PrefillChunk(0, 0, chunk), PrefillChunk(0, chunk, 2 * chunk)]
    refs = [MediaRef(0, 100, 200, a), MediaRef(1, 100, 200, a), MediaRef(1, 220, 240, b)]
    store = SharedBufferStore(embs)
    runtime = runtime_with(store, sp=sp, chunk_size=chunk, mesh_shape=(sp, 1), shard_vocab_on_sp=shard_vocab_on_sp)
    got = []
    runtime._media_splice_buffers = SimpleNamespace(
        refill=lambda plan: got.extend(run.rows.copy() for run in plan.runs)
    )

    TtPrefillRuntime._media_splice(runtime, chunks, refs)

    u16 = lambda r: embs[r.media_key].view(torch.int16).numpy().view(np.uint16)
    want = [run.rows for run in plan_media_splice(refs, chunks, u16, sp, chunk // sp, not shard_vocab_on_sp).runs]
    assert store.calls == 1
    assert len(got) == len(want) and all(np.array_equal(g, w) for g, w in zip(got, want))


def test_failed_fetch_frees_the_input(monkeypatch, expect_error):
    """A forward whose image value is missing raises, and still frees the token input it consumes."""
    freed = []
    monkeypatch.setattr(tt_prefill_runtime.ttnn, "deallocate", freed.append)
    runtime = TtPrefillRuntime.__new__(TtPrefillRuntime)
    vars(runtime).update(vars(runtime_with(StubEmbeddingReader(HIDDEN), sp=1, chunk_size=CHUNK, mesh_shape=(1, 1))))
    vars(runtime.config).update(forward_chunks=1, num_users=1, max_seq_len=CHUNK, is_first_rank=True)
    runtime.model_built = True
    runtime._embed_tokens = lambda tokens, media_fn: media_fn()
    tokens = object()

    with expect_error(Missing, bytes(32).hex()):
        runtime.prefill_forward(tokens, None, [PrefillChunk(0, 0, CHUNK)], media_refs=[MediaRef(0, 0, 10, bytes(32))])
    assert freed == [tokens]


def tiny_hf_model():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLModel

    config = MiniMaxM3VLConfig(
        text_config=dict(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            num_local_experts=2,
            num_experts_per_tok=1,
            dense_intermediate_size=64,
            shared_intermediate_size=64,
            rotary_dim=32,
            index_head_dim=32,
            bos_token_id=None,
            eos_token_id=None,
        ),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2),
        projector_hidden_size=64,
        image_token_index=IMAGE_TOKEN,
    )
    return MiniMaxM3VLModel(config).eval().to(torch.bfloat16)


def hf_prompt(layout, model):
    """Expanded token ids, (start, num_tokens, key) per image and the HF image features, for a layout of text
    lengths and (h, w) images."""
    from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import MiniMaxM3VLImageProcessor

    processor = MiniMaxM3VLImageProcessor()
    merge = model.config.vision_config.spatial_merge_size
    rng = np.random.default_rng(0)
    ids, images, features = [], [], []
    for part in layout:
        if isinstance(part, int):
            ids += rng.integers(0, IMAGE_TOKEN, part).tolist()
            continue
        image = processor(images=[rng.integers(0, 256, (*part, 3), dtype=np.uint8)], return_tensors="pt")
        n = int(image["image_grid_thw"].prod()) // merge**2
        with torch.no_grad():
            features.append(
                model.get_image_features(
                    pixel_values=image["pixel_values"].to(torch.bfloat16), image_grid_thw=image["image_grid_thw"]
                ).pooler_output
            )
        images.append((len(ids) + 1, n, bytes([len(images) + 1]) * 32))
        ids += [VISION_START] + [IMAGE_TOKEN] * n + [VISION_END]
    return torch.tensor(ids), images, features


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (2, 4), (8, 4)])
@pytest.mark.parametrize(
    "layout,first_chunk_start",
    [
        pytest.param([3, (64, 96), 5, (84, 84), 10, (56, 112), 4], 0, id="several_images"),
        pytest.param([100, (280, 448), 40], 0, id="image_across_chunks"),
        pytest.param([100, (280, 448), 40], 128, id="resume_mid_image"),
    ],
)
def test_post_splice_vs_hf(mesh_device, device_params, monkeypatch, layout, first_chunk_start):
    pytest.importorskip("transformers.models.minimax_m3_vl", reason="needs transformers with minimax_m3_vl")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(0)
    model = tiny_hf_model()
    ids, images, features = hf_prompt(layout, model)

    with torch.no_grad():
        embeds = model.get_input_embeddings()(ids[None])
        all_features = torch.cat(features)
        mask, _ = model.get_placeholder_mask(ids[None], embeds, image_features=all_features)
        want = embeds.masked_scatter(mask, all_features)[0]

    store = Store({key: f for (_, _, key), f in zip(images, features)})
    table = model.get_input_embeddings().weight.detach()
    got = splice_chunks(mesh_device, table, ids.to(torch.int32), images, store, first_chunk_start)

    is_image = image_mask(images, first_chunk_start, len(ids))
    assert torch.equal(is_image, ids[first_chunk_start:] == IMAGE_TOKEN), "image positions differ from HF's"
    check(got, want[first_chunk_start:], images, first_chunk_start, len(ids))
