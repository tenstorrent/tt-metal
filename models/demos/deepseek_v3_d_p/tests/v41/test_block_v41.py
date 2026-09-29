# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 blocks on device state (beads F3, F7): real-dims schedule 2 -> 3 -> 20 -> 21 -> 24.

Each block runs on device with the oracle's inputs (teacher-forced streams and pre_mix); everything else is
device-produced and device-held: window KV, compressed KV and index keys written by the KV sources, top-k
published by the index sources and reused by consumers, candidates of layer 20 constraining layer 24.
Checks per layer: block output and next pre_mix vs the oracle, compressed-KV and index-K rows vs the oracle's
shared state, the window-KV carry vs the oracle's window KV, determinism (a second pass over a fresh state is
bit-identical); selection recall is reported.
Bars (G1): block >= 0.99 synthetic / >= 0.98 real; caches >= 0.998.

``kv_format`` selects the stage of the compressed layers' KV (epic KV FORMAT): ``bf16`` stores the reference's
QDQ values, so compressed KV is compared with the oracle's FP4-QDQ rows; ``scaled_fp8`` encodes the unrounded
KV, so its compressed rows are compared with the reference's compressed KV before the FP4 QDQ (the value the
format encodes); the PCC against the FP4-QDQ rows is reported (``compressed_kv_vs_fp4``, no bar).
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import kernel_cpu
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.reference_weights import MOE_KEYS, device_weights
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config, small_spec
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import (  # noqa: F401 (WEIGHT_CACHE re-export)
    WEIGHT_CACHE,
    begin_layer,
    complete_layer,
    host_weights,
    weight_cache_dir,
)
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import WINDOW_SLOT, V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from tests.ttnn.utils_for_testing import comp_pcc

# every sharing role (2->3, 20->21->24) and the sliding-window-only type (layer 0; layer 1 needs Engram, F5)
SCHEDULES = {"stack": (2, 3, 20, 21, 24), "swa": (0,)}
SEQ = 2048
BLOCK_PCC = {"small": 0.99, "synthetic": 0.99, "real": 0.98}
# User decision 2026-09-29 (bars by data source): synthetic production-shape index-source blocks are hypersensitive
# to the intrinsic top-k disagreement of FP4-quantized selection (reference self-recall 0.993 at bf16-level input
# perturbation; synthetic compressed rows uncorrelated), so they are gated at the real-weight bar.
SYNTHETIC_INDEX_SOURCE_PCC = 0.98
SMALL_SEQ = 512
CACHE_PCC = 0.998


def _pack(x, tp):
    s, n, d = x.shape
    return x.reshape(s, n, tp, d // tp).permute(0, 2, 1, 3).reshape(1, 1, s, n * d)


def _unpack(t, n, tp):
    s, width = t.shape[-2], t.shape[-1]
    return t.reshape(s, tp, n, width // n // tp).permute(0, 2, 1, 3).reshape(s, n, -1)


def _pcc(a, b):
    return comp_pcc(a.float(), b.float(), 0.0)[1]


KV_FORMATS = {"bf16": MlaKvCacheFormat.BF16_RM, "scaled_fp8": MlaKvCacheFormat.SCALED_FP8}


@torch.no_grad()
def _unrounded_compressed_kv(model, spec, layer: int, attn_in: torch.Tensor) -> torch.Tensor:
    """The reference compressed KV of KV source ``layer`` with RoPE and before its FP4 QDQ (``Attention._compress_kv``
    at start_pos 0), from the recorded attention input: what the SCALED_FP8 cache encodes."""
    attn = model.layers[spec.layer_ids.index(layer)].attn
    seq, ratio = attn_in.shape[0], attn.compress_ratio
    with v41.set_dtype(torch.bfloat16):
        latent = attn.compressor(attn_in[None], 0)
        if latent is None:  # shorter than one group
            return attn_in.new_zeros(0, attn.head_dim)
        v41.apply_rotary_emb(latent[..., -attn.rope_head_dim :], attn.freqs_cis[: seq - seq % ratio : ratio])
    return latent[0]


def setup_blocks(mesh_device, weights, chunks, schedule, prompt, kv_format):
    """Everything before the device forward: oracle result, reference-derived expectations, device weights (MoE
    tensors from / into the weight cache) and the blocks. Shared by the test and the CPU-only cache prepare step
    (tests/v41/prepare_caches.py), which runs it on a mock mesh so the device run only reads caches."""
    LAYERS = SCHEDULES[schedule]
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
    # prompt lengths: full chunks, a padded last chunk (graph.md rule 8), and a prompt shorter than a ratio-2 group
    # pair plus window edge (empty or single compressed row: top-k empty)
    valid = {"full": seq, "padded": seq - 12, "tiny": 3}[prompt]
    if prompt != "full" and weights != "small":
        pytest.skip("prompt-length edge cases run at small dims")
    tokens = orc.random_tokens(spec, valid)
    # built only on an oracle or weight-cache miss (warm runs construct no reference model)
    reference = orc.LazyReference(spec)
    result = orc.oracle(spec, tokens, model=reference)
    fmt = KV_FORMATS[kv_format]
    unrounded = {}
    if fmt == MlaKvCacheFormat.SCALED_FP8 and any(l in C.KV_SOURCE_LAYERS for l in LAYERS):
        for l in (l for l in LAYERS if l in C.KV_SOURCE_LAYERS):
            base = orc.cache_path(spec, tokens)
            unrounded[l] = host_weights(
                base.parent,
                f"{base.stem}-unrounded-kv{l}",
                lambda l=l: _unrounded_compressed_kv(reference(), spec, l, result["blocks"][l]["attn_in"]),
            )
            # the derivation is the reference's: its FP4 QDQ reproduces the oracle's compressed rows bit for bit
            fp4 = kernel_cpu.fp4_act_quant(unrounded[l].clone(), 16, True, scale_dtype=torch.float8_e4m3fn)
            assert torch.equal(fp4, result["shared"][l]["compress_kv"]), l

    # MoE device tensors are cached on disk (tests/v41/weight_cache.py key): the first build converts 1152 expert
    # matrices per layer on the host (minutes); later builds load them. A marker records a completed layer.
    cache_root = weight_cache_dir(spec, mesh_device.shape)
    cache_root.mkdir(parents=True, exist_ok=True)
    init_checker(cache_root)
    blocks = {}
    for i, layer in enumerate(LAYERS):
        cached = begin_layer(cache_root, layer)
        if ckpt is not None:
            w = load_layer_dense(ckpt, layer) if cached else load_layer(ckpt, layer)
        elif cached:
            w = host_weights(
                cache_root, f"layer_{layer}.dense", lambda i=i: device_weights(reference(), i, include_moe=False)
            )
        else:
            w = device_weights(reference(), i)
            host_weights(
                cache_root, f"layer_{layer}.dense", lambda w=w: {k: v for k, v in w.items() if k not in MOE_KEYS}
            )
        blocks[layer] = TtV41Block(mesh_device, cfg, layer, w, seq // chunks, weight_cache_path=cache_root)
        complete_layer(cache_root, layer)
        del w
    logger.info(f"reference model built: {reference.built}")

    return SimpleNamespace(
        LAYERS=LAYERS,
        seq=seq,
        spec=spec,
        cfg=cfg,
        valid=valid,
        tokens=tokens,
        result=result,
        fmt=fmt,
        unrounded=unrounded,
        blocks=blocks,
        reference=reference,
    )


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kv_format", list(KV_FORMATS))
@pytest.mark.parametrize("prompt", ["full", "padded", "tiny"])
@pytest.mark.parametrize("schedule", ["stack", "swa"])
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
def test_v41_blocks_on_device_state(mesh_device, device_params, weights, chunks, schedule, prompt, kv_format):
    setup = setup_blocks(mesh_device, weights, chunks, schedule, prompt, kv_format)
    LAYERS, seq, spec, cfg, valid, tokens = setup.LAYERS, setup.seq, setup.spec, setup.cfg, setup.valid, setup.tokens
    result, fmt, unrounded, blocks = setup.result, setup.fmt, setup.unrounded, setup.blocks
    shape, (sp, tp), n = tuple(mesh_device.shape), tuple(mesh_device.shape), cfg.HC_MULT
    down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))

    def run():
        """All chunks through all layers in execution order; layer inputs teacher-forced per chunk, padded rows
        of the last chunk zero (their outputs are not compared)."""
        chunk = seq // chunks
        state = V41PrefillState(mesh_device, cfg, seq, chunk, list(LAYERS), kv_format=fmt)
        outs = {layer: ([], []) for layer in LAYERS}
        for c in range(chunks):
            start = c * chunk
            length = min(chunk, valid - start)
            if length <= 0:
                break
            for layer in LAYERS:
                rec = result["blocks"][layer]
                x_rows = torch.zeros(chunk, *rec["x_in"].shape[1:])
                x_rows[:length] = rec["x_in"][start : start + length].float()
                pre_rows = torch.zeros(chunk, rec["pre_in"].shape[-1])
                pre_rows[:length] = rec["pre_in"][start : start + length].float()
                x = ttnn.from_torch(
                    _pack(x_rows, tp),
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
                )
                pre = ttnn.from_torch(
                    pre_rows[None, None],
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, None)),
                )
                x_out, pre_out = blocks[layer](x, pre, state, length)
                outs[layer][0].append(_unpack(down(x_out)[0, 0], n, tp)[:length])
                outs[layer][1].append(down(pre_out)[0, 0, :length, :n])
            state.advance(length)
        return state, {layer: (torch.cat(a), torch.cat(b)) for layer, (a, b) in outs.items()}

    state, first = run()
    _, second = run()
    report = {}
    # the first layer's attention alone on the oracle's attention input (a KV source reads only its own rows)
    head = LAYERS[0]
    attention_pcc = None
    if prompt == "full" and chunks == 1:
        # the first layer's attention alone on the oracle's attention input (a KV source reads only its own rows)
        attn_state = V41PrefillState(mesh_device, cfg, seq, seq, [head], kv_format=fmt)
        attn_in = ttnn.from_torch(
            result["blocks"][head]["attn_in"][None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
        )
        attention_pcc = _pcc(
            result["blocks"][head]["attn_out"], down(blocks[head].attn(attn_in, attn_state, seq))[0, 0]
        )
    for layer in LAYERS:
        rec, (x_out, pre_out) = result["blocks"][layer], first[layer]
        entry = {
            "block_type": C.block_type(layer).value,
            "block_out": _pcc(rec["x_out"], x_out),
            "pre_mix": _pcc(rec["pre_out"], pre_out),
            "deterministic": torch.equal(x_out, second[layer][0]) and torch.equal(pre_out, second[layer][1]),
        }
        # window KV contents: the carry holds the last WINDOW_SLOT positions' post-RoPE FP8-QDQ rows, right-aligned
        tail = rec["window_kv"][max(0, valid - WINDOW_SLOT) : valid].float()
        expected_carry = torch.zeros(WINDOW_SLOT, tail.shape[-1])
        expected_carry[WINDOW_SLOT - tail.shape[0] :] = tail
        entry["window_kv"] = _pcc(expected_carry, state.to_host(state.window_carry[layer]))
        if layer in C.KV_SOURCE_LAYERS:
            pub = result["shared"][layer]
            rows = pub["compress_kv"].shape[0]
            w0 = state.geometry.window_rows
            stored = state.to_host(state.kv[layer])[w0 : w0 + rows]
            if fmt == MlaKvCacheFormat.SCALED_FP8:
                entry["compressed_kv"] = _pcc(unrounded[layer], stored)
                entry["compressed_kv_vs_fp4"] = _pcc(pub["compress_kv"], stored)
            else:
                entry["compressed_kv"] = _pcc(pub["compress_kv"], stored)
            entry["index_k"] = _pcc(pub["index_k"], state.to_host(state.index_k[layer])[:rows])
        if layer == head and attention_pcc is not None:
            entry["attention_alone"] = attention_pcc
        report[layer] = entry
        logger.info(f"layer {layer} ({weights}): {entry}")
    out = os.environ.get("TT_V41_BLOCK_REPORT")
    if out:
        Path(out).write_text(repr(report))
    for layer, entry in report.items():
        assert entry["deterministic"], (layer, entry)
        bar = BLOCK_PCC[weights]
        if weights == "synthetic" and layer in C.INDEX_SOURCE_LAYERS:
            bar = SYNTHETIC_INDEX_SOURCE_PCC
        assert entry["block_out"] >= bar, (layer, bar, entry)
        for key in ("compressed_kv", "index_k", "window_kv"):
            if key in entry:
                assert entry[key] >= CACHE_PCC, (layer, key, entry)
