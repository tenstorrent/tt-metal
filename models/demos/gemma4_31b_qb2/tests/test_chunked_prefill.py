# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Resume checkpoint-weight prefills across aligned and partial cache pages."""
import pytest
import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
from models.demos.gemma4_31b_qb2.tests.test_decoder import ReferenceCache, check_rows, reference
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.model import WeightReader, checkpoint_path


@pytest.mark.parametrize("layer_index", [0, 5], ids=["sliding", "full"])
@pytest.mark.parametrize(
    "ends",
    [(1024, 2048, 2177), (127, 1023, 1025, 2177), (1024, 2049, 3073, 4098), (1201, 2317, 3467, 4571)],
    ids=["aligned", "partial-pages", "after-full-window", "ragged-windows"],
)
def test_resumed_prefill_matches_hf(qb2_mesh, layer_index, ends):
    torch.set_num_threads(8)
    rng = torch.Generator().manual_seed(3600 + layer_index)
    path = checkpoint_path()
    config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    reader = WeightReader(path)
    weights = reader.layer(layer_index)
    prefix = f"model.language_model.layers.{layer_index}."
    with torch.device("meta"):
        hf = Gemma4TextDecoderLayer(config, layer_index)
    hf.load_state_dict({k[len(prefix) :]: v for k, v in weights.items()}, assign=True, strict=True)
    hf.eval()
    rope = Gemma4TextRotaryEmbedding(config)
    layer = Decoder.from_state_dict(weights, hf_config=config, layer_idx=layer_index, mesh_device=qb2_mesh)
    layer._prepare_prefill_norm_stats()
    pages = (ends[-1] + 128) // 128
    table = torch.randperm(pages + 2, generator=rng)[2:].reshape(1, pages).int()
    cache = layer.allocate_cache(physical_pages=pages + 2)
    ref_cache = ReferenceCache(1, layer.kv_heads * 4, pages * 128, layer.head_dim)
    embedding = reader.get("model.language_model.embed_tokens.weight")
    ids = torch.randint(100, config.vocab_size, (1, ends[-1] + 1), generator=rng)
    inputs = embedding[ids] * torch.tensor(config.hidden_size**0.5, dtype=embedding.dtype)
    del embedding
    pt = ttnn.from_torch(
        table,
        device=qb2_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(qb2_mesh),
    )
    previous_end, history = 0, None
    for end in ends:
        aligned_start = previous_end // 128 * 128
        previous = None
        if history is not None:
            valid = history[0].shape[2] - (previous_end - aligned_start)
            begin = max(0, valid - 1024)
            previous = tuple(t[:, :, begin:valid, :] for t in history) if valid else None
        x = inputs[:, aligned_start:end]
        sharded = layer.prefill_input_width(end - aligned_start) != config.hidden_size
        tx = ttnn.from_torch(
            x.unsqueeze(1).contiguous(),
            device=qb2_mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(qb2_mesh, dim=3) if sharded else ttnn.ReplicateTensorToMesh(qb2_mesh),
        )
        out, history = layer.prefill_forward(
            tx, page_table=pt, kv_cache=cache, start_pos=aligned_start, history=previous, return_history=True
        )
        parts = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
        actual = (torch.cat(parts, dim=-1) if sharded else parts[0]).squeeze(1)
        if not sharded:
            assert all(torch.equal(parts[0], part) for part in parts[1:])
        expected = reference(
            hf, rope, x, torch.arange(aligned_start, end).reshape(1, -1), ref_cache, layer.kind, layer.window
        )
        print(f"COMPARE layer={layer_index} start={aligned_start} end={end}", flush=True)
        check_rows(expected, actual)
        previous_end = end
    dx = inputs[:, -1:]
    tx = ttnn.from_torch(
        dx.reshape(1, 1, 1, -1),
        device=qb2_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(qb2_mesh),
    )
    positions = torch.tensor([ends[-1]], dtype=torch.int32)
    tp = ttnn.from_torch(
        positions,
        device=qb2_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(qb2_mesh),
    )
    out = layer.decode_forward(tx, positions=tp, page_table=pt, kv_cache=cache, cyclic_cache=False)
    actual = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(1, 1, -1)
    expected = reference(hf, rope, dx, positions[:, None], ref_cache, layer.kind, layer.window)
    check_rows(expected, actual)
