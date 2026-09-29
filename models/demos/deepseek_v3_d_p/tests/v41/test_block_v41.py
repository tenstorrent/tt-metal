# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 blocks on device state (beads F3, F7): real-dims schedule 2 -> 3 -> 20 -> 21 -> 24.

Each block runs on device with the oracle's inputs (teacher-forced streams and pre_mix); everything else is
device-produced and device-held: window KV, compressed KV and index keys written by the KV sources, top-k
published by the index sources and reused by consumers, candidates of layer 20 constraining layer 24.
Checks per layer: block output and next pre_mix vs the oracle, compressed-KV and index-K rows vs the oracle's
shared state, determinism (a second pass over a fresh state is bit-identical); selection recall is reported.
Bars (G1): block >= 0.99 synthetic / >= 0.98 real; caches >= 0.998.
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.prototype_oracle import device_weights
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config, small_spec
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from tests.ttnn.utils_for_testing import comp_pcc

LAYERS = (2, 3, 20, 21, 24)
SEQ = 2048
BLOCK_PCC = {"small": 0.99, "synthetic": 0.99, "real": 0.98}
SMALL_SEQ = 512
CACHE_PCC = 0.998
WEIGHT_CACHE = Path(os.environ.get("TT_V41_WEIGHT_CACHE", Path.home() / ".cache" / "tt-v41-weights"))


def _pack(x, tp):
    s, n, d = x.shape
    return x.reshape(s, n, tp, d // tp).permute(0, 2, 1, 3).reshape(1, 1, s, n * d)


def _unpack(t, n, tp):
    s, width = t.shape[-2], t.shape[-1]
    return t.reshape(s, tp, n, width // n // tp).permute(0, 2, 1, 3).reshape(s, n, -1)


def _pcc(a, b):
    return comp_pcc(a.float(), b.float(), 0.0)[1]


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("chunks", [1, 2], ids=["single_chunk", "two_chunks"])
@pytest.mark.parametrize("weights", ["small", "synthetic", "real"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        )
    ],
    indirect=True,
)
def test_v41_blocks_on_device_state(mesh_device, device_params, weights, chunks):
    ckpt = resolve_checkpoint() if weights == "real" else None
    if weights == "real" and ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    if weights == "small":  # same code path at small dims: seconds of CPU oracle, minutes of device time
        seq = SMALL_SEQ
        spec, cfg = small_spec(LAYERS, seq), SmallV41Config
    else:
        seq = SEQ
        spec = orc.real_spec(LAYERS, seq, candidate_topk_blocks=96, checkpoint=ckpt.root if ckpt else None)
        cfg = type("V41TestConfig", (C,), {"CANDIDATE_TOPK_BLOCKS": 96})  # match the oracle's candidate count
    tokens = orc.random_tokens(spec)
    reference = orc.build_reference(spec) if ckpt is None else None
    result = orc.oracle(spec, tokens, model=reference)
    shape, (sp, tp), n = tuple(mesh_device.shape), tuple(mesh_device.shape), cfg.HC_MULT
    topology = per_axis_topology(device_params["fabric_config"])[1]
    down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))
    host = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()[0, 0]

    # MoE device tensors are cached on disk: the first build converts 1152 expert matrices per layer on the
    # host (minutes); later builds load them. A marker records a completed layer.
    cache_root = WEIGHT_CACHE / f"{weights}-{orc.cache_path(spec, tokens).stem}"
    cache_root.mkdir(parents=True, exist_ok=True)
    init_checker(cache_root)
    blocks = {}
    for i, layer in enumerate(LAYERS):
        marker = cache_root / f"layer_{layer}.complete"
        if ckpt is not None:
            w = load_layer_dense(ckpt, layer) if marker.exists() else load_layer(ckpt, layer)
        else:
            w = device_weights(reference, i)
        if marker.exists():
            w = {
                k: v
                for k, v in w.items()
                if k not in ("routed_expert_weights", "shared_expert_weights", "gate_weights")
            }
        blocks[layer] = TtV41Block(
            mesh_device, cfg, layer, w, seq // chunks, topology=topology, weight_cache_path=cache_root
        )
        marker.touch()
        del w

    def run():
        """All chunks through all layers in execution order; layer inputs teacher-forced per chunk."""
        chunk = seq // chunks
        state = V41PrefillState(mesh_device, cfg, seq, chunk, list(LAYERS))
        outs = {layer: ([], []) for layer in LAYERS}
        for c in range(chunks):
            rows = slice(c * chunk, (c + 1) * chunk)
            for layer in LAYERS:
                rec = result["blocks"][layer]
                x = ttnn.from_torch(
                    _pack(rec["x_in"][rows].float(), tp),
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
                )
                pre = ttnn.from_torch(
                    rec["pre_in"][rows].float()[None, None],
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, None)),
                )
                x_out, pre_out = blocks[layer](x, pre, state, chunk)
                outs[layer][0].append(_unpack(down(x_out)[0, 0], n, tp))
                outs[layer][1].append(down(pre_out)[0, 0, :, :n])
            state.advance(chunk)
        return state, {layer: (torch.cat(a), torch.cat(b)) for layer, (a, b) in outs.items()}

    state, first = run()
    _, second = run()
    report = {}
    # the first layer's attention alone on the oracle's attention input (a KV source reads only its own rows)
    head = LAYERS[0]
    attn_state = V41PrefillState(mesh_device, cfg, seq, seq, [head])
    attn_in = ttnn.from_torch(
        result["blocks"][head]["attn_in"][None, None],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
    )
    attention_pcc = _pcc(result["blocks"][head]["attn_out"], down(blocks[head].attn(attn_in, attn_state, seq))[0, 0])
    for layer in LAYERS:
        rec, (x_out, pre_out) = result["blocks"][layer], first[layer]
        entry = {
            "block_type": C.block_type(layer).value,
            "block_out": _pcc(rec["x_out"], x_out),
            "pre_mix": _pcc(rec["pre_out"], pre_out),
            "deterministic": torch.equal(x_out, second[layer][0]) and torch.equal(pre_out, second[layer][1]),
        }
        if layer in C.KV_SOURCE_LAYERS:
            pub = result["shared"][layer]
            rows = pub["compress_kv"].shape[0]
            w0 = state.geometry.window_rows
            entry["compressed_kv"] = _pcc(pub["compress_kv"], host(state.kv[layer])[w0 : w0 + rows])
            entry["index_k"] = _pcc(pub["index_k"], host(state.index_k[layer])[:rows])
        if layer == head and chunks == 1:
            entry["attention_alone"] = attention_pcc
        report[layer] = entry
        logger.info(f"layer {layer} ({weights}): {entry}")
    out = os.environ.get("TT_V41_BLOCK_REPORT")
    if out:
        Path(out).write_text(repr(report))
    for layer, entry in report.items():
        assert entry["deterministic"], (layer, entry)
        assert entry["block_out"] >= BLOCK_PCC[weights], (layer, entry)
        for key in ("compressed_kv", "index_k"):
            if key in entry:
                assert entry[key] >= CACHE_PCC, (layer, key, entry)
