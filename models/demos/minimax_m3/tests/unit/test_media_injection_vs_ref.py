# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post-injection embeddings on one chip: TtParallelEmbedding over an expanded prompt with image rows injected chunk
by chunk, against a torch merge of stub rows and against Hugging Face minimax_m3_vl's own merge.

Run: pytest models/demos/minimax_m3/tests/unit/test_media_injection_vs_ref.py
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.common.prefill.media import MediaItem, build_injection, media_rows_for_chunk, stub_embeddings
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.media_injection import inject_media_embeddings
from models.demos.minimax_m3.tt.parallel_embedding import TtParallelEmbedding
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

VOCAB, HIDDEN, CHUNK = 1024, 256, 128
IMAGE_TOKEN, VISION_START, VISION_END = 1000, 1001, 1002


def item(start, n, seed):
    return MediaItem(start, n, bytes([seed]) * 32)


def inject_chunks(mesh_device, table, tokens, items, rows_of, emb_tp_sharded, first_chunk_start):
    """Embeddings of tokens[first_chunk_start:] after injection, prefilled CHUNK tokens at a time."""
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
        media_rows = build_injection(media_rows_for_chunk(items, start, end, rows_of), start, end, 1, CHUNK)
        if media_rows is not None:
            x = inject_media_embeddings(
                mesh_device,
                x,
                *media_rows,
                mesh_shape=mesh_shape,
                sp_axis=0,
                tp_axis=1,
                emb_tp_sharded=emb_tp_sharded,
            )
        out.append(ttnn.to_torch(ttnn.get_device_tensors(x)[0]).reshape(CHUNK, -1)[: end - start])
        ttnn.deallocate(x)
    return torch.cat(out)


def image_mask(items, start, end):
    positions = torch.arange(start, end)
    mask = torch.zeros(end - start, dtype=torch.bool)
    for m in items:
        mask |= (positions >= m.start_position) & (positions < m.end_position)
    return mask


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("emb_tp_sharded", [True, False], ids=["tp_sharded", "replicated"])
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
def test_post_injection_embeddings(mesh_device, device_params, emb_tp_sharded, seq_len, layout, first_chunk_start):
    torch.manual_seed(0)
    table = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    tokens = torch.randint(0, VOCAB, (seq_len,), dtype=torch.int32)
    items = [item(s, n, i + 1) for i, (s, n) in enumerate(layout)]
    embs = {m.hash: stub_embeddings(m.hash, m.num_tokens, HIDDEN) for m in items}
    ref = F.embedding(tokens.long(), table)
    for m in items:
        ref[m.start_position : m.end_position] = embs[m.hash]

    got = inject_chunks(
        mesh_device, table, tokens, items, lambda m, lo, hi: embs[m.hash][lo:hi], emb_tp_sharded, first_chunk_start
    )

    want = ref[first_chunk_start:]
    is_image = image_mask(items, first_chunk_start, seq_len)
    assert torch.equal(got[~is_image], want[~is_image]), "a text position changed"
    assert torch.equal(got[is_image], want[is_image]), "an image row is not its embedding"


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
    """Expanded token ids, our MediaItems and the HF image features for a layout of text lengths and (h, w) images."""
    from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import MiniMaxM3VLImageProcessor

    processor = MiniMaxM3VLImageProcessor()
    merge = model.config.vision_config.spatial_merge_size
    rng = np.random.default_rng(0)
    ids, items, features = [], [], []
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
        items.append(item(len(ids) + 1, n, len(items) + 1))
        ids += [VISION_START] + [IMAGE_TOKEN] * n + [VISION_END]
    return torch.tensor(ids), items, features


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("emb_tp_sharded", [True, False], ids=["tp_sharded", "replicated"])
@pytest.mark.parametrize(
    "layout,first_chunk_start",
    [
        pytest.param([10, (64, 96), 30], 0, id="one_image"),
        pytest.param([3, (64, 96), 5, (84, 84), 10, (56, 112), 4], 0, id="several_images"),
        pytest.param([100, (280, 448), 40], 0, id="image_across_chunks"),
        pytest.param([100, (280, 448), 40], 128, id="resume_mid_image"),
        pytest.param([200], 0, id="text_only"),
    ],
)
def test_post_injection_vs_hf(mesh_device, device_params, monkeypatch, emb_tp_sharded, layout, first_chunk_start):
    pytest.importorskip("transformers.models.minimax_m3_vl", reason="needs transformers with minimax_m3_vl")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(0)
    model = tiny_hf_model()
    ids, items, features = hf_prompt(layout, model)

    with torch.no_grad():
        embeds = model.get_input_embeddings()(ids[None])
        all_features = torch.cat(features) if features else None
        mask, _ = model.get_placeholder_mask(ids[None], embeds, image_features=all_features)
        ref = (embeds.masked_scatter(mask, all_features) if features else embeds)[0]

    rows = dict(zip((m.hash for m in items), features))
    got = inject_chunks(
        mesh_device,
        model.get_input_embeddings().weight.detach(),
        ids.to(torch.int32),
        items,
        lambda m, lo, hi: rows[m.hash][lo:hi],
        emb_tp_sharded,
        first_chunk_start,
    )

    want = ref[first_chunk_start:]
    is_image = image_mask(items, first_chunk_start, len(ids))
    assert torch.equal(is_image, ids[first_chunk_start:] == IMAGE_TOKEN), "image positions differ from HF's"
    assert torch.equal(got[~is_image], want[~is_image]), "a text position differs from HF"
    assert torch.equal(got[is_image], want[is_image]), "an image row differs from HF"
