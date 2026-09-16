# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checkpoint-weight HF comparisons at padding, page, and sliding-window boundaries."""

import pytest
import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.model import WeightReader, checkpoint_path


class ReferenceCache:
    def __init__(self, batch, heads, capacity, dim):
        self.k = torch.zeros(batch, heads, capacity, dim, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.positions = None
        self.limit = 0

    def update(self, k, v, layer_idx):
        for b in range(k.shape[0]):
            self.k[b, :, self.positions[b]] = k[b]
            self.v[b, :, self.positions[b]] = v[b]
        self.limit = max(self.limit, int(self.positions.max()) + 1)
        return self.k[:, :, : self.limit], self.v[:, :, : self.limit]


def reference(layer, rope, x, positions, cache, kind, window):
    cache.positions = positions
    limit = max(cache.limit, int(positions.max()) + 1)
    keys = torch.arange(limit).reshape(1, 1, 1, -1)
    pos = positions[:, None, :, None]
    allowed = keys <= pos
    if window:
        allowed &= keys > pos - window
    mask = torch.where(allowed, 0.0, float("-inf")).bfloat16()
    with torch.no_grad():
        return layer(
            x,
            position_embeddings=rope(x, positions, layer_type=kind),
            attention_mask=mask,
            past_key_values=cache,
            shared_kv_states={},
        )


def check_rows(expected, actual):
    assert expected.shape == actual.shape
    assert torch.isfinite(actual).all()
    for a, b in zip(expected, actual):
        passed, message = comp_pcc(a, b, pcc=0.995)
        assert passed, message


@pytest.mark.parametrize("layer_index", [0, 5], ids=["sliding", "full"])
@pytest.mark.parametrize("length,batch", [(1, 13), (127, 1), (129, 3), (1025, 2)])
def test_prefill_and_live_decode_trace(qb2_mesh, layer_index, length, batch):
    """Every logical row is scored; changed inputs, positions and slot maps reach the trace."""
    torch.set_num_threads(8)
    rng = torch.Generator().manual_seed(3600 + layer_index + length)
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
    capacity = ((length + 4 + 127) // 128) * 128
    pages = capacity // 128
    table = torch.randperm(batch * pages + 2, generator=rng)[2:].reshape(batch, pages).int()
    cache = layer.allocate_cache(physical_pages=batch * pages + 2)
    ref_cache = ReferenceCache(batch, layer.kv_heads * 4, capacity, layer.head_dim)

    def device(x, *, integer=False, sharded=False):
        return ttnn.from_torch(
            x.contiguous(),
            device=qb2_mesh,
            dtype=ttnn.int32 if integer else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(qb2_mesh, dim=3) if sharded else ttnn.ReplicateTensorToMesh(qb2_mesh),
        )

    def host(x, *, sharded=False):
        parts = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(x)]
        if sharded:
            return torch.cat(parts, dim=-1)
        for part in parts[1:]:
            assert torch.equal(parts[0], part), "Replicated outputs differ across mesh ranks"
        return parts[0]

    def refresh(value, target, *, integer=False):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                value.contiguous(),
                dtype=ttnn.int32 if integer else ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            ),
            target,
        )

    # Actual embeddings give reproducible, non-Gaussian inputs without a fixture tied to a worker path.
    embedding = reader.get("model.language_model.embed_tokens.weight")
    token_ids = torch.randint(100, config.vocab_size, (batch, length + 4), generator=rng)
    inputs = embedding[token_ids] * torch.tensor(config.hidden_size**0.5, dtype=embedding.dtype)
    del embedding
    x = inputs[:, :length]
    width_sharded = layer.prefill_input_width(length) != config.hidden_size
    tx = device(x.unsqueeze(1), sharded=width_sharded)
    tt_table = device(table, integer=True)
    actual = host(layer.prefill_forward(tx, page_table=tt_table, kv_cache=cache), sharded=width_sharded).squeeze(1)
    expected = reference(hf, rope, x, torch.arange(length).expand(batch, -1), ref_cache, layer.kind, layer.window)
    check_rows(expected, actual)

    positions = torch.full((batch,), length, dtype=torch.int32)
    dx = inputs[:, length].reshape(1, 1, batch, -1)
    td, tp = device(dx), device(positions, integer=True)

    def forward():
        return layer.decode_forward(td, positions=tp, page_table=tt_table, kv_cache=cache, cyclic_cache=False)

    eager = host(forward()).reshape(batch, 1, -1)
    expected = reference(hf, rope, dx.reshape(batch, 1, -1), positions[:, None], ref_cache, layer.kind, layer.window)
    check_rows(expected, eager)
    trace = ttnn.begin_trace_capture(qb2_mesh, cq_id=0)
    output = forward()
    ttnn.end_trace_capture(qb2_mesh, trace, cq_id=0)
    try:
        for step in range(3):
            positions.fill_(length + step)
            dx = inputs[:, length + step].reshape(1, 1, batch, -1)
            if step == 1 and batch > 1:
                # Move whole requests between scheduler slots, including their live histories.
                table = table.flip(0).contiguous()
                ref_cache.k = ref_cache.k.flip(0).contiguous()
                ref_cache.v = ref_cache.v.flip(0).contiguous()
                refresh(table, tt_table, integer=True)
            refresh(dx, td)
            refresh(positions, tp, integer=True)
            ttnn.execute_trace(qb2_mesh, trace, cq_id=0, blocking=True)
            actual = host(output).reshape(batch, 1, -1)
            expected = reference(
                hf, rope, dx.reshape(batch, 1, -1), positions[:, None], ref_cache, layer.kind, layer.window
            )
            check_rows(expected, actual)
            # Replaying unchanged inputs must be deterministic and overwrite only the same cache position.
            ttnn.execute_trace(qb2_mesh, trace, cq_id=0, blocking=True)
            assert torch.equal(actual, host(output).reshape(batch, 1, -1))
    finally:
        ttnn.release_trace(qb2_mesh, trace)
