# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post-splice embeddings on one chip: TtParallelEmbedding over an expanded prompt with image rows spliced chunk by
chunk through TtPrefillRuntime._splice_media, against a torch merge and against Hugging Face minimax_m3_vl's merge.

Run: pytest models/demos/minimax_m3/tests/unit/test_media_splice_vs_ref.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.multimedia_store.reader import FetchRequest
from models.demos.common.prefill.media import build_injection, store_key
from models.demos.common.prefill.prefill_chunk import MediaRef, PrefillChunk
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt import tt_prefill_runtime
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.parallel_embedding import TtParallelEmbedding
from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime
from models.demos.minimax_m3.utils.general_utils import get_default_num_links
from models.demos.minimax_m3.vision import format

from ..test_factory import parametrize_mesh_with_fabric

VOCAB, HIDDEN, CHUNK = 1024, 256, 128
IMAGE_TOKEN, VISION_START, VISION_END = 1000, 1001, 1002


class Store:
    """Serves each image's embeddings as a stored value in format.py's layout, keyed by store_key."""

    def __init__(self, embs: dict):
        self.values = {store_key(key): self.value(rows) for key, rows in embs.items()}

    @staticmethod
    def value(rows: torch.Tensor) -> bytes:
        rows = rows.to(torch.bfloat16)
        h = format.Header(1, rows.shape[0], rows.shape[0], rows.shape[1])
        buf = memoryview(bytearray(format.value_size(h.n_tokens, h.hidden)))
        format.write_header(buf, h)
        format.embeddings(buf, h)[:] = rows.view(torch.int16).numpy()
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


def runtime_with(store, *, sp, chunk_size, mesh_shape, mesh_device=None):
    return SimpleNamespace(
        config=SimpleNamespace(sp_factor=sp, chunk_size=chunk_size, mesh_shape=mesh_shape, sp_axis=0, tp_axis=1),
        hf_config=SimpleNamespace(hidden_size=HIDDEN),
        media_embedding_reader=store,
        mesh_device=mesh_device,
    )


def splice_chunks(mesh_device, table, tokens, images, store, first_chunk_start):
    """Embeddings of tokens[first_chunk_start:] after splicing, prefilled CHUNK tokens at a time.
    images: (start, num_tokens, key) per image."""
    mesh_shape = tuple(mesh_device.shape)
    emb = TtParallelEmbedding(
        mesh_device,
        VOCAB,
        HIDDEN,
        MeshConfig(mesh_shape, tp=mesh_shape[1]),
        CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear),
        torch_weight=table,
        dtype=ttnn.bfloat16,
        shard_vocab_on_sp=False,
    )
    runtime = runtime_with(store, sp=1, chunk_size=CHUNK, mesh_shape=mesh_shape, mesh_device=mesh_device)
    out = []
    for start in range(first_chunk_start, len(tokens), CHUNK):
        end = min(start + CHUNK, len(tokens))
        chunk_tokens = torch.zeros(CHUNK, dtype=torch.int32)
        chunk_tokens[: end - start] = tokens[start:end]
        x = emb.forward(
            ttnn.from_torch(
                chunk_tokens.reshape(1, 1, CHUNK),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(0, None)),
            )
        )
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)
        refs = [MediaRef(0, s, n, key) for s, n, key in images if s < end and s + n > start]
        if refs:
            x = TtPrefillRuntime._splice_media(runtime, x, [PrefillChunk(0, start, end)], refs)
        out.append(ttnn.to_torch(ttnn.get_device_tensors(x)[0]).reshape(CHUNK, -1)[: end - start])
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


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("sharded_residual", ["1", "0"], ids=["tp_sharded", "replicated"])
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
    mesh_device, device_params, monkeypatch, sharded_residual, seq_len, layout, first_chunk_start
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

    got = splice_chunks(mesh_device, table, tokens, images, Store(embs), first_chunk_start)
    check(got, want[first_chunk_start:], images, first_chunk_start, seq_len)


def test_one_batched_fetch_per_forward(monkeypatch):
    """A packed forward naming one image in two chunks reads every value in one fetch, before the next fetch could
    overwrite it."""
    monkeypatch.setattr(tt_prefill_runtime, "inject_media_embeddings", lambda dev, x, rows, mask, **kw: (rows, mask))
    sp, chunk = 2, 128
    a, b = bytes([1]) * 32, bytes([2]) * 32
    embs = {a: torch.randn(100, HIDDEN).to(torch.bfloat16), b: torch.randn(20, HIDDEN).to(torch.bfloat16)}
    chunks = [PrefillChunk(0, 0, chunk), PrefillChunk(0, chunk, 2 * chunk)]
    refs = [MediaRef(0, 100, 100, a), MediaRef(1, 100, 100, a), MediaRef(1, 220, 20, b)]
    store = SharedBufferStore(embs)
    runtime = runtime_with(store, sp=sp, chunk_size=chunk, mesh_shape=(sp, 1))

    rows, mask = TtPrefillRuntime._splice_media(runtime, None, chunks, refs)

    want_rows, want_mask = build_injection(
        refs, chunks, lambda r: embs[r.media_key].view(torch.int16).numpy(), sp, chunk // sp
    )
    assert store.calls == 1
    assert torch.equal(mask, want_mask)
    assert torch.equal(rows, want_rows)


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


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
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
