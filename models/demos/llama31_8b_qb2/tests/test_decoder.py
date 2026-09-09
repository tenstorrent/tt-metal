# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight decoder checks for chunk tails, batch families and trace replay."""

from copy import copy
from dataclasses import replace

import pytest
import torch
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

import ttnn
from models.common.modules.tt_ccl import TT_CCL
from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder
from models.demos.llama31_8b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.llama31_8b_qb2.tt.precision import load_precision_config


def to_device(value, mesh, *, shard=False):
    integer = not value.is_floating_point()
    return ttnn.from_torch(
        value.contiguous(),
        device=mesh,
        dtype=ttnn.int32 if integer else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1) if shard else ttnn.ReplicateTensorToMesh(mesh),
    )


def to_host(value):
    return torch.cat([ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(value)], dim=-1)


def assert_correlated(actual, expected, threshold=0.99):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    a, b = actual.float().flatten(), expected.float().flatten()
    pcc = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
    assert pcc >= threshold, f"PCC {pcc:.6f} < {threshold}"
    return pcc


@pytest.mark.parametrize("batch,length", [(1, 129), (9, 1025), (32, 128)])
def test_decoder(qb2_mesh, batch, length):
    """Compare with HF, then replay shared families through a page remapping."""
    torch.set_num_threads(8)
    mesh = qb2_mesh
    folder = checkpoint_path()
    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    config._attn_implementation = "sdpa"
    checkpoint = Checkpoint(folder)
    names = [name for name in checkpoint.index if name.startswith("model.layers.0.")]
    weights = checkpoint.load(names)
    reference = LlamaDecoderLayer(config, layer_idx=0).bfloat16().eval()
    reference.load_state_dict({name.removeprefix("model.layers.0."): value for name, value in weights.items()})
    rope = LlamaRotaryEmbedding(config)
    decoder = LlamaDecoder.from_state_dict(
        weights,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh,
        precision_policy=load_precision_config(),
        ccl=TT_CCL(mesh),
    )
    workspace = decoder.prepare_decode(batch)
    embedding = checkpoint.load(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    ids = torch.randint(0, config.vocab_size, (batch, length + 1), generator=torch.Generator().manual_seed(35))
    inputs = embedding[ids]
    pages = (length + 128) // 128
    table = torch.arange(1, 1 + batch * pages, dtype=torch.int32).reshape(batch, pages)
    cache = decoder.allocate_cache(num_physical_pages=1 + batch * pages)

    with torch.no_grad():
        positions = torch.arange(length + 1)[None].expand(batch, -1)
        mask = torch.full((length + 1, length + 1), torch.finfo(torch.bfloat16).min).triu(1)[None, None]
        expected = reference(inputs, attention_mask=mask, position_embeddings=rope(inputs, positions))

    def prefill(mapping):
        return decoder.prefill_forward(
            to_device(inputs[:, None, :length], mesh, shard=True),
            page_table=to_device(mapping, mesh),
            kv_cache=cache,
        )

    actual = to_host(prefill(table))[:, 0]
    print(f"HF prefill PCC: {assert_correlated(actual, expected[:, :length]):.6f}", flush=True)
    # Refill the same logical prefixes through a different physical page order.
    remapped = table.flip(1).contiguous()
    torch.testing.assert_close(to_host(prefill(remapped))[:, 0], actual, rtol=0, atol=0)
    families = {batch: decoder}
    if batch > 1:
        single = copy(decoder)
        single.decode_workspace = None
        buffers = {
            name: ttnn.reshape(
                tensor, ttnn.Shape((1, 1, 1, tensor.shape[-1])), tensor.padded_shape, skip_padding_fill=True
            )
            for name, tensor in workspace.buffers.items()
        }
        single.prepare_decode(1, workspace=replace(workspace, batch=1, buffers=buffers))
        families[1] = single
    traces = {}
    outputs = {}
    calls = {}
    for size, layer in families.items():
        hidden = to_device(inputs[:size, None, length:].reshape(1, 1, size, 4096), mesh, shard=True)
        pos = to_device(torch.full((size,), length, dtype=torch.int32), mesh)
        mapping = to_device(remapped[:size], mesh)
        calls[size] = lambda layer=layer, hidden=hidden, pos=pos, mapping=mapping: layer.decode_forward(
            hidden, current_pos=pos, page_table=mapping, kv_cache=cache
        )
        eager = to_host(calls[size]())
        print(
            f"HF decode B{size} PCC: {assert_correlated(eager.reshape(size, 4096), expected[:size, -1]):.6f}",
            flush=True,
        )
        outputs[size] = eager
    # Compile every family before any trace captures allocator addresses.
    try:
        for size, call in calls.items():
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            output = call()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            traces[size] = (tid, output)
        for size in [batch, 1, batch]:
            tid, output = traces[size]
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            torch.testing.assert_close(to_host(output), outputs[size], rtol=0, atol=0)
    finally:
        for tid, _ in traces.values():
            ttnn.release_trace(mesh, tid)
