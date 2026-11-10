# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A speculative block is exact: draft columns of one user in one decode call equal the same steps run one by one.

The columns write K/V at consecutive positions, which share a cache tile; the decoder updates the cache once per
column so the writes do not race.
"""

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.model import WeightReader, checkpoint_path

COLUMNS = 4
FIRST_POSITION = 1500  # inside a cache tile, so the block's positions share it


@pytest.mark.parametrize("layer_index", [0, 5], ids=["sliding", "full"])
def test_draft_columns_equal_sequential_decode_steps(qb2_mesh, layer_index):
    rng = torch.Generator().manual_seed(5)
    path = checkpoint_path()
    config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
    reader = WeightReader(path)
    layer = Decoder.from_state_dict(
        reader.layer(layer_index), hf_config=config, layer_idx=layer_index, mesh_device=qb2_mesh
    )
    embedding = reader.get("model.language_model.embed_tokens.weight")
    pages = (FIRST_POSITION + COLUMNS + 127) // 128 + 1
    physical = pages + COLUMNS + 2
    cache = layer.allocate_cache(physical_pages=physical)
    history = [torch.randn(tuple(t.shape), generator=rng).bfloat16() * 0.5 for t in cache]
    table = (torch.randperm(physical - COLUMNS - 2, generator=rng)[:pages] + COLUMNS + 2).int()
    ids = torch.randint(100, config.vocab_size, (COLUMNS,), generator=rng)
    hidden = (embedding[ids] * torch.tensor(config.hidden_size**0.5, dtype=embedding.dtype)).float()
    replicate = ttnn.ReplicateTensorToMesh(qb2_mesh)

    def reset_cache():
        for tensor, host in zip(cache, history):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=tensor.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate), tensor
            )

    def device(values, integer=False):
        return ttnn.from_torch(
            values.contiguous(),
            device=qb2_mesh,
            dtype=ttnn.int32 if integer else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )

    def decode(rows, positions, *, per_column_updates=False):
        updates = None
        if per_column_updates:
            updates = [
                device(torch.where(torch.arange(COLUMNS) == c, positions, -1).int(), True) for c in range(COLUMNS)
            ]
        out = layer.decode_forward(
            device(rows.reshape(1, 1, COLUMNS, -1)),
            positions=device(positions, True),
            page_table=device(table.unsqueeze(0).repeat(COLUMNS, 1), True),
            kv_cache=cache,
            cyclic_cache=False,
            update_positions=updates,
        )
        return (
            ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(qb2_mesh, dim=0))
            .float()
            .reshape(4, COLUMNS, -1)[0]
        )

    reset_cache()
    block = decode(
        hidden, torch.arange(FIRST_POSITION, FIRST_POSITION + COLUMNS, dtype=torch.int32), per_column_updates=True
    )
    reset_cache()
    for column in range(COLUMNS):
        rows = torch.zeros(COLUMNS, hidden.shape[1])
        rows[0] = hidden[column]
        positions = torch.full((COLUMNS,), -1, dtype=torch.int32)
        positions[0] = FIRST_POSITION + column
        assert torch.equal(
            block[column], decode(rows, positions)[0]
        ), f"column {column} differs from its sequential step"
