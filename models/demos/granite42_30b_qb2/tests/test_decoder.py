# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Checkpoint-weight HF comparisons for padded prefill, prefix reuse and traced decode."""

import json

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.granite.modeling_granite import GraniteDecoderLayer, GraniteRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc

from ..tt.checkpoint import resolve_checkpoint
from ..tt.decoder import GraniteDecoder
from ..tt.precision import layer_policy, load_precision


class ReferenceCache:
    def __init__(self, capacity):
        self.k = torch.zeros(1, 8, capacity, 128, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.positions = None
        self.limit = 0

    def update(self, k, v, layer_idx, **kwargs):
        self.k[:, :, self.positions] = k
        self.v[:, :, self.positions] = v
        self.limit = max(self.limit, int(self.positions.max()) + 1)
        return self.k[:, :, : self.limit], self.v[:, :, : self.limit]


def reference(layer, rope, x, positions, cache):
    cache.positions = positions
    limit = max(cache.limit, int(positions.max()) + 1)
    allowed = torch.arange(limit)[None, :] <= positions[:, None]
    mask = torch.where(allowed, 0.0, float("-inf"))[None, None].bfloat16()
    with torch.no_grad():
        return layer(
            x,
            position_embeddings=rope(x, positions[None]),
            attention_mask=mask,
            past_key_values=cache,
        )


def check_output(expected, actual, pcc):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    passed, message = comp_pcc(expected, actual, pcc=pcc)
    print(f"Granite decoder PCC: {message}", flush=True)
    assert passed, message


@pytest.fixture(scope="module", params=["selected", "bfp8-control"])
def decoder_pair(request):
    torch.set_num_threads(8)
    path = resolve_checkpoint()
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config._attn_implementation = "sdpa"
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.layers.0."
    weights = {}
    for shard in {index[key] for key in index if key.startswith(prefix)}:
        with safe_open(path / shard, framework="pt") as f:
            weights.update({key: f.get_tensor(key) for key in index if key.startswith(prefix) and index[key] == shard})
    with torch.device("meta"):
        hf = GraniteDecoderLayer(config, 0)
    hf.load_state_dict({key[len(prefix) :]: value for key, value in weights.items()}, assign=True, strict=True)
    hf.eval()
    rope = GraniteRotaryEmbedding(config)
    precision = load_precision()
    # Selected BFP4 projections are compared to unquantized HF BF16. In the
    # 127-token case, decode position 129 measures 0.9931 PCC; changing only
    # projection weights to BFP8 raises it to 0.9985. Use 0.99 for the selected
    # policy and 0.995 for that control, rather than weakening the reference or
    # changing the release precision to satisfy a new test's arbitrary bar.
    pcc = 0.99
    if request.param == "bfp8-control":
        for role in ("qkv", "o", "gate", "up", "down"):
            precision["weight_groups"][role] = "bfloat8_b"
        pcc = 0.995
    key = "model.embed_tokens.weight"
    with safe_open(path / index[key], framework="pt") as f:
        embeddings = f.get_slice(key)[:512].clone() * config.embedding_multiplier
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    try:
        layer = GraniteDecoder.from_state_dict(
            weights, hf_config=config, layer_idx=0, mesh_device=mesh, policy=layer_policy(precision, 0)
        )
        yield mesh, layer, hf, rope, embeddings, pcc
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("prefix_length,length", [(0, 127), (0, 129), (129, 3)])
def test_checkpoint_prefill_and_traced_decode(decoder_pair, prefix_length, length):
    mesh, layer, hf, rope, embeddings, pcc = decoder_pair
    rng = torch.Generator().manual_seed(1380 + prefix_length + length)
    total = prefix_length + length
    inputs = embeddings[torch.randint(100, 512, (1, total + 3), generator=rng)]
    # Cover padded bulk-prefill storage as well as the logical decode positions.
    pages = ((total + 1023) // 1024) * 32
    table = torch.randperm(pages + 3, generator=rng)[3:].reshape(1, pages).int()
    kv = [
        ttnn.zeros([pages + 3, 2, 32, 128], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh)
        for _ in range(2)
    ]
    cache = ReferenceCache(pages * 32)

    def host(value):
        parts = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(value)]
        for part in parts[1:]:
            assert torch.equal(parts[0], part), "Replicated decoder outputs differ across ranks"
        return parts[0]

    def prefill(start, count):
        x = inputs[:, start : start + count]
        entries = layer.prepare_prefill(x, page_table=table, start_pos=start)
        outputs = layer.prefill_forward(entries, kv_cache=kv)
        actual = torch.cat([host(y).squeeze(1) for y in outputs], dim=1)
        expected = reference(hf, rope, x, torch.arange(start, start + count), cache)
        check_output(expected, actual, pcc)

    if prefix_length:
        prefill(0, prefix_length)
    prefill(prefix_length, length)

    # A token entry supplies the same stable input layouts as the production
    # decoder path. Change both hidden state and absolute position on replay.
    entry = layer.prepare_prefill(inputs[:, total : total + 1], page_table=table, start_pos=total)[0]
    assert entry.write_page_table is None

    def forward():
        return layer.decode_forward(
            entry.x, current_pos=entry.position, page_table=entry.page_table, kv_cache=kv, cos=entry.cos, sin=entry.sin
        )

    expected_rows, eager_rows = [], []
    for step in range(3):
        position = total + step
        x = inputs[:, position : position + 1]
        layer.refresh_prefill_entry(entry, x, page_table=table, start_pos=position)
        expected_rows.append(reference(hf, rope, x, torch.tensor([position]), cache))
        eager_rows.append(host(forward()).squeeze(1))
        print(f"eager decode position={position}", flush=True)
        check_output(expected_rows[-1], eager_rows[-1], pcc)
    layer.refresh_prefill_entry(entry, inputs[:, total : total + 1], page_table=table, start_pos=total)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    output = forward()
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        for step in range(3):
            position = total + step
            x = inputs[:, position : position + 1]
            layer.refresh_prefill_entry(entry, x, page_table=table, start_pos=position)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = host(output).squeeze(1)
            print(f"traced decode position={position}", flush=True)
            check_output(expected_rows[step], actual, pcc)
            assert torch.equal(eager_rows[step], actual), "Trace replay differs from eager decode"
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            assert torch.equal(actual, host(output).squeeze(1))
    finally:
        ttnn.release_trace(mesh, trace)
