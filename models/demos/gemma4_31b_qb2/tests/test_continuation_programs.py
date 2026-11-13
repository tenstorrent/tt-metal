# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A continuation prefill compiles no program for a (length, start) it has not seen, and its layer output equals the per-length one bit for bit.

The position-dependent ops of a prefill chunk (rope rows, cache-fill page ids, chunked-SDPA start, last-token tile) take
runtime values and the prompt is padded to whole cache pages, so the programs depend only on the padded chunk length.
warm_prefill_shapes compiles those; any later signature then adds nothing and cannot retire the decode traces.
"""

import pytest
import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.generator import build_generator

CAPACITY = 5120  # 40 pages: the chunked-SDPA page-table row must be a multiple of 32 bytes
# (length, start): unaligned lengths, aligned and unaligned starts, one and several chunks, never seen by the warm-up.
SIGNATURES = [(2327, 1152), (2701, 2304), (3000, 1152), (1300, 1152), (4100, 2048), (1537, 1024)]


@pytest.fixture
def generator(qb2_mesh):
    gen = build_generator(qb2_mesh, layer_indices=[0, 5], max_seq_len=CAPACITY)
    yield gen
    gen.close()


def test_novel_continuation_signatures_add_no_programs(generator, qb2_mesh):
    gen = generator
    state = gen.allocate_cache(batch=1, capacity=CAPACITY)
    tokens = torch.randint(1000, 50000, (1, CAPACITY), generator=torch.Generator().manual_seed(1))
    gen.warm_prefill_shapes(state, state.page_tables, CAPACITY)
    key = iter(range(1, 1000))

    def run(length, start, **kwargs):
        history = next(key)
        if start:
            gen.prefill_forward(
                tokens, page_table=state.page_tables, kv_cache=state, prompt_lens=[start], history_keys=[history]
            )
        return gen.prefill_forward(
            tokens,
            page_table=state.page_tables,
            kv_cache=state,
            prompt_lens=[length],
            start_positions=[start],
            history_keys=[history],
            **kwargs,
        )

    for length, start in SIGNATURES:
        programs = qb2_mesh.num_program_cache_entries()
        run(length, start)
        assert qb2_mesh.num_program_cache_entries() == programs, f"({length}, {start}) compiled programs"


@pytest.mark.parametrize(
    "layer_index", [0, 1], ids=["sliding", "full"]
)  # the loaded layers are checkpoint layers 0 and 5
@pytest.mark.parametrize("length, start", [(1175, 1152), (1300, 1280), (300, 1024)])
def test_runtime_chunk_inputs_give_the_same_layer_output_bit_for_bit(generator, qb2_mesh, layer_index, length, start):
    """Layer level: the continuation chunk with runtime-valued inputs (padded to whole pages) equals the per-length one."""
    layer = generator.model.layers[layer_index]
    layer._prepare_prefill_norm_stats()
    # The chunked-SDPA page-table row must be a multiple of 32 bytes: a multiple of 8 pages.
    pages = ((start + length + 127) // 128 + 2 + 7) // 8 * 8
    torch.manual_seed(layer_index)
    table = (torch.randperm(pages) + 1).reshape(1, pages).int()
    cache = layer.allocate_cache(physical_pages=pages + 2)
    replicate = ttnn.ReplicateTensorToMesh(qb2_mesh)
    pt = ttnn.from_torch(table, device=qb2_mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate)

    def hidden(rows):
        x = torch.randn(1, 1, rows, layer.hidden_size).to(torch.bfloat16)
        return x

    def upload(x):
        sharded = layer.prefill_input_width(x.shape[2]) != layer.hidden_size
        return (
            ttnn.from_torch(
                x,
                device=qb2_mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(qb2_mesh, dim=3) if sharded else replicate,
            ),
            sharded,
        )

    def read(out, sharded):
        parts = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
        return torch.cat(parts, dim=-1) if sharded else parts[0]

    # The preceding chunk: its cache pages and (sliding) history tail.
    first, sharded = upload(hidden(start))
    _, history = layer.prefill_forward(first, page_table=pt, kv_cache=cache, start_pos=0, return_history=True)
    previous = None
    if history is not None:
        valid = history[0].shape[2]
        begin = max(0, valid - 1024)
        previous = tuple(t[:, :, begin:valid, :] for t in history)
    x = hidden(length)
    padded = (length + 127) // 128 * 128
    x_padded = torch.zeros(1, 1, padded, layer.hidden_size, dtype=torch.bfloat16)
    x_padded[:, :, :length] = x

    tx, sharded = upload(x)
    expected, _ = layer.prefill_forward(
        tx, page_table=pt, kv_cache=cache, start_pos=start, history=previous, return_history=True
    )
    expected = read(expected, sharded)[:, :, :length]

    tx, sharded = upload(x_padded)
    # As Gemma4Model.prefill_device builds them: rope rows per layer kind, fill pages and start per page-table row.
    rope = layer.prefill_chunk_inputs(padded, start)
    fills = layer.prefill_chunk_inputs(padded, start, table)
    inputs = [dict(f, rope=r["rope"]) for r, f in zip(rope, fills)]
    actual, _ = layer.prefill_forward(
        tx,
        page_table=pt,
        kv_cache=cache,
        start_pos=start,
        history=previous,
        return_history=True,
        chunk_inputs=inputs,
        padded_tails=True,
    )
    actual = read(actual, sharded)[:, :, :length]
    assert torch.equal(actual, expected), f"layer {layer_index} ({length}, {start}): outputs differ"
