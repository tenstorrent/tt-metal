# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post-injection embeddings on one chip: TtParallelEmbedding over an expanded prompt, then the image rows injected
chunk by chunk, against the torch merge (token embeddings with each image's rows written at its positions).

Text positions must match exactly and image positions must equal the injected embeddings. Covers one image,
several images in one chunk, an image across chunk boundaries, a resume that starts mid-image, and text only.
Embeddings here are stubs; the real vision encoder is tt-metal#62156.

Run: pytest models/demos/minimax_m3/tests/unit/test_media_injection_vs_ref.py -s
"""

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


def item(start, n, seed):
    return MediaItem(start, n, bytes([seed]) * 32)


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

    ref = F.embedding(tokens.long(), table).clone()
    for m in items:
        ref[m.start_position : m.end_position] = embs[m.hash]

    rows_mesh, cols_mesh = tuple(mesh_device.shape)
    emb = TtParallelEmbedding(
        mesh_device,
        VOCAB,
        HIDDEN,
        MeshConfig((rows_mesh, cols_mesh), tp=cols_mesh),
        CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear),
        torch_weight=table,
        dtype=ttnn.bfloat16,
        shard_vocab_on_sp=False,
    )

    injected_any = False
    for start in range(first_chunk_start, seq_len, CHUNK):
        end = min(start + CHUNK, seq_len)
        chunk_tokens = torch.zeros(CHUNK, dtype=torch.int32)
        chunk_tokens[: end - start] = tokens[start:end]
        tt_tokens = ttnn.from_torch(
            chunk_tokens.reshape(1, 1, CHUNK),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows_mesh, cols_mesh), dims=(0, None)),
        )
        x = emb.forward(tt_tokens)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)

        slices = media_rows_for_chunk(items, start, end, lambda m, lo, hi: embs[m.hash][lo:hi])
        media_rows = build_injection(slices, start, end, sp=1, chunk_local=CHUNK)
        if media_rows is not None:
            injected_any = True
            x = inject_media_embeddings(
                mesh_device,
                x,
                *media_rows,
                mesh_shape=(rows_mesh, cols_mesh),
                sp_axis=0,
                tp_axis=1,
                emb_tp_sharded=emb_tp_sharded,
            )

        got = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).reshape(CHUNK, -1)[: end - start]
        ttnn.deallocate(x)
        want = ref[start:end]
        assert got.shape == want.shape, f"chunk at {start}: {tuple(got.shape)} vs {tuple(want.shape)}"

        positions = torch.arange(start, end)
        is_image = torch.zeros(end - start, dtype=torch.bool)
        for m in items:
            is_image |= (positions >= m.start_position) & (positions < m.end_position)
        assert torch.equal(got[~is_image], want[~is_image]), f"chunk at {start}: a text position changed"
        assert torch.equal(got[is_image], want[is_image]), f"chunk at {start}: an image row is not its embedding"

    assert injected_any == bool(items)
