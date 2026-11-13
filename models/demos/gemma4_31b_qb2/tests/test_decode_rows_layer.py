# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A decode step of a layer with few rows agrees with the 32-row step to within the layer accuracy bound.

The decode SDPA and the sharded memory configs depend on the row count (a lone row spreads its KV range over more cores),
so the results are not bit-identical across traces; they agree to PCC >= PCC_MIN, well inside the PCC >= 0.995 bound the
decoder tests hold against the HF reference. The end-to-end check is the real-prompt panel (greedy and seeded outputs
of the 32-slot server against the 1-slot server) and the teacher-forced KL over recorded agent prompts.
"""

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.model import WeightReader, checkpoint_path

PCC_MIN = 0.998
ROWS = (1, 4, 8, 16, 32)


@pytest.mark.parametrize("layer_index", [0, 5], ids=["sliding", "full"])
def test_decode_rows_agree_with_the_32_row_trace(qb2_mesh, layer_index):
    rng = torch.Generator().manual_seed(11)
    path = checkpoint_path()
    config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
    reader = WeightReader(path)
    layer = Decoder.from_state_dict(
        reader.layer(layer_index), hf_config=config, layer_idx=layer_index, mesh_device=qb2_mesh
    )
    embedding = reader.get("model.language_model.embed_tokens.weight")
    pages = 16
    physical = pages + 40
    cache = layer.allocate_cache(physical_pages=physical)
    history = [torch.randn(tuple(t.shape), generator=rng).bfloat16() * 0.5 for t in cache]
    table = (torch.randperm(physical - 40, generator=rng)[:pages] + 34).int()
    replicate = ttnn.ReplicateTensorToMesh(qb2_mesh)

    def device(values, integer=False):
        return ttnn.from_torch(
            values.contiguous(),
            device=qb2_mesh,
            dtype=ttnn.int32 if integer else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )

    def decode(rows, hidden, position):
        for tensor, host in zip(cache, history):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=tensor.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate), tensor
            )
        x = torch.zeros(1, 1, rows, config.hidden_size)
        x[0, 0, 0] = hidden
        positions = torch.full((rows,), -1, dtype=torch.int32)
        positions[0] = position
        out = layer.decode_forward(
            device(x),
            positions=device(positions, True),
            page_table=device(table.unsqueeze(0).repeat(rows, 1), True),
            kv_cache=cache,
            cyclic_cache=False,
        )
        return (
            ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(qb2_mesh, dim=0))[0]
            .reshape(-1, config.hidden_size)[0]
            .float()
        )

    worst = 1.0
    for _ in range(4):
        token = int(torch.randint(100, config.vocab_size, (1,), generator=rng))
        position = int(torch.randint(150, 1900, (1,), generator=rng))
        hidden = (embedding[token] * torch.tensor(config.hidden_size**0.5, dtype=embedding.dtype)).float()
        reference = decode(32, hidden, position)
        for rows in ROWS[:-1]:
            pcc = float(torch.corrcoef(torch.stack([decode(rows, hidden, position), reference]))[0, 1])
            worst = min(worst, pcc)
            assert pcc >= PCC_MIN, f"{rows}-row trace vs 32-row trace: PCC {pcc:.5f} at position {position}"
    print(f"layer {layer_index}: worst PCC vs the 32-row trace {worst:.5f}")
