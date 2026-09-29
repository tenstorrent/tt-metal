# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 Engram (bead F5): host hash + lookup, device wkv + gated stream update, vs the reference.

Expected results come from ``engram_reference.engram_case`` (the vendored reference's ``NgramHashState``,
``ParallelEngramEmbedding`` on synthetic rows and ``Engram.forward``, cached on disk; precompute with
``python -m models.demos.deepseek_v3_d_p.tests.v41.engram_reference``). Checks:

* host hash ids == reference, single-shot and chunked with the carried history (chunks shorter than an n-gram,
  an image span across a chunk boundary); the history is the last 3 compressed ids;
* host lookup rows == reference lookup (bit-exact);
* device update (G1 new elementwise/linear ops >= 0.999): the Engram increment ``out - x`` per stream copy and
  the output vs the reference's fp32 update; image-token rows pass through bit-exactly; repeat bit-identical;
  one and two chunks, and a padded last chunk.

Real Engram weights and tables (shards 47/48, 101 GB each) are not downloaded: synthetic weights only.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41 import engram_reference as er
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.engram import DEAD, TtV41Engram, V41EngramHash, V41EngramTable
from models.demos.deepseek_v3_d_p.tt.v41.weights import dequant_fp8_block
from tests.ttnn.utils_for_testing import comp_pcc

ENGRAM_PCC = 0.999
LAYER = 1


def _config(case: str):
    return DeepSeekV41FlashConfig if case == "real" else SmallV41Config


def _pcc(a, b):
    return comp_pcc(a.float(), b.float(), 0.0)[1]


def _pack(x, tp):
    s, n, d = x.shape
    return x.reshape(s, n, tp, d // tp).permute(0, 2, 1, 3).reshape(1, 1, s, n * d)


def _unpack(t, n, tp):
    s, width = t.shape[-2], t.shape[-1]
    return t.reshape(s, tp, n, width // n // tp).permute(0, 2, 1, 3).reshape(s, n, -1)


def _hasher(case: str) -> V41EngramHash:
    from transformers import AutoTokenizer

    return V41EngramHash.from_tokenizer(_config(case), AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT))


@pytest.mark.parametrize("case", ["small", "small_padded", "small_image", "real"])
def test_engram_host(case, expect_error):
    """Hash ids (chunked, carried history) and row lookup are bit-exact vs the reference."""
    spec, tokens, mask = er.case_inputs(case)
    expected = er.engram_case(spec, tokens, mask)
    hasher = _hasher(case)
    column = hasher.layer_index(LAYER)
    total = tokens.numel()
    compressed = hasher.token_map[tokens]
    if mask is not None:
        compressed = torch.where(mask, compressed, DEAD)
    splits = {"single": [total], "halves": [256, total - 256], "short": [1, 2, 1, 3, 250, total - 257]}
    for name, split in splits.items():
        history, ids, start = hasher.new_history(), [], 0
        for length in split:
            chunk_mask = None if mask is None else mask[start : start + length]
            chunk_ids, history = hasher(tokens[start : start + length], history, chunk_mask)
            ids.append(chunk_ids[:, column])
            start += length
            assert torch.equal(history, compressed[max(0, start - 3) : start]), (name, start)
        assert torch.equal(torch.cat(ids), expected["hash_ids"]), name
    if case == "small":  # the helper's Engram input/output are the full oracle's (text-only) captures
        block = orc.oracle(spec, tokens[None])["blocks"][LAYER]
        assert torch.equal(block["engram_hash_ids"], expected["hash_ids"])
        assert torch.equal(block["engram_in"], expected["x_in"]) and torch.equal(block["x_in"], expected["out"])
    table = V41EngramTable(**expected["table"])
    started = time.perf_counter()
    rows = table.lookup(expected["hash_ids"]).flatten(-2)
    logger.info(f"{case}: host lookup of {total} tokens {1e3 * (time.perf_counter() - started):.1f} ms")
    assert torch.equal(rows, expected["rows"])
    with expect_error(KeyError, "does not hold"):
        table.lookup(expected["table"]["row_ids"][-1:] + 1)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("chunks", [1, 2], ids=["single_chunk", "two_chunks"])
@pytest.mark.parametrize("case", ["small", "small_padded", "small_image", "real"])
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
def test_engram_device(mesh_device, device_params, case, chunks):
    cfg = _config(case)
    spec, tokens, mask = er.case_inputs(case)
    expected = er.engram_case(spec, tokens, mask)
    reference = orc.build_reference(spec)
    engram = reference.layers[0].engram
    weights = {
        "wkv": dequant_fp8_block(engram.wkv.weight.detach(), engram.wkv.scale.detach()),
        "q_weight": engram.q_weight.detach(),
        "k_weight": engram.k_weight.detach(),
    }
    hasher = V41EngramHash(cfg, reference.engram_hash.token_map)
    column = hasher.layer_index(LAYER)
    shape, (sp, tp), n = tuple(mesh_device.shape), tuple(mesh_device.shape), cfg.HC_MULT
    topology = per_axis_topology(device_params["fabric_config"])[1]
    module = TtV41Engram(mesh_device, cfg, LAYER, weights, V41EngramTable(**expected["table"]), topology)
    del reference, weights

    total = tokens.numel()
    chunk = -(-total // chunks // 128) * 128  # a multiple of 2 * 32 * sp
    x_in = expected["x_in"].float()

    def run():
        outs, history, host_ms, device_ms = [], hasher.new_history(), [], []
        for start in range(0, total, chunk):
            length = min(chunk, total - start)
            chunk_mask = None if mask is None else mask[start : start + length]
            began = time.perf_counter()
            ids, history = hasher(tokens[start : start + length], history, chunk_mask)
            inputs = module.prepare(ids[:, column], chunk, chunk_mask)
            ttnn.synchronize_device(mesh_device)
            host_ms.append(1e3 * (time.perf_counter() - began))
            rows = torch.zeros(chunk, n, x_in.shape[-1])
            rows[:length] = x_in[start : start + length]
            x = ttnn.from_torch(
                _pack(rows, tp),
                device=mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
            )
            ttnn.synchronize_device(mesh_device)
            began = time.perf_counter()
            out = module(x, inputs)
            ttnn.synchronize_device(mesh_device)
            device_ms.append(1e3 * (time.perf_counter() - began))
            full = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))
            outs.append(_unpack(full[0, 0], n, tp)[:length])
        return torch.cat(outs), host_ms, device_ms

    first, host_ms, device_ms = run()
    second, host_ms2, device_ms2 = run()
    want = expected["out_fp32"]
    delta, want_delta = first - x_in, want - x_in
    report = {
        "out": _pcc(want, first),
        "out_vs_bf16_reference": _pcc(expected["out"], first),
        "delta_per_copy": [_pcc(want_delta[:, c], delta[:, c]) for c in range(n)],
        "delta_max_abs_err": float((delta - want_delta).abs().max()),
        "deterministic": torch.equal(first, second),
        "host_prepare_ms": [round(t, 1) for t in host_ms2],
        "device_forward_ms": [round(t, 1) for t in device_ms2],
    }
    if mask is not None:
        report["image_rows_unchanged"] = torch.equal(first[~mask], x_in[~mask])
    logger.info(f"engram {case} {chunks} chunk(s): {report}")
    assert report["deterministic"]
    assert report.get("image_rows_unchanged", True)
    assert report["out"] >= ENGRAM_PCC and min(report["delta_per_copy"]) >= ENGRAM_PCC, report
