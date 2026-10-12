# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics for the DeepSeek-V4 Flash full-depth drift, against the CPU reference and the vLLM golden.

``test_v4_flash_depth_local_error`` runs chunk 0 through every layer on device, then re-runs selected
layers on the CPU reference from the device's own input to that layer. The gap between the device's
output and the reference's is the error that layer adds, separate from what it inherited. Chunk 0 is
enough: prefill is causal, so its tokens depend on nothing later, and the golden's largest-norm
tokens all sit in it.

``test_v4_flash_layer3_routing`` feeds layer 3, the first top-k MoE layer, the golden's layer-2 output
over the whole prompt and splits the device's expert choice against the golden's into the gate
top-k, the gate matmul, and the gate's input.

Both log and assert nothing about accuracy.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.block import v4_block_forward
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.tests.v4 import golden
from models.demos.deepseek_v3_d_p.tests.v4.test_transformer import streams_to_host, upload_tokens
from models.demos.deepseek_v3_d_p.tests.v4.test_transformer_depth import (
    _GALAXY,
    _GOLDEN,
    _OUTLIERS,
    CHUNK,
    N_CHUNKS,
    SEQ_CACHE,
    _log_compressed_kv,
    _topk_agreement,
    _upload_streams,
)
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import score_activation
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Transformer
from models.demos.deepseek_v3_d_p.tt.v4.weights import (
    V4CheckpointLayers,
    v4_layer_from_checkpoint,
    v4_model_from_checkpoint,
)

_FLASH = DeepSeekV4FlashConfig
# Baseline layers with golden input and output (1, 3, 9), two mid-depth, and the run-up to the last.
_PROBE_LAYERS = [1, 3, 9, 20, 30, 36, 37, 38, 39, 40, 41, 42]
_ROUTING_LAYER = 3


def _probe_layers():
    env = os.getenv("V4_PROBE_LAYERS")
    return sorted(int(s) for s in env.split(",") if s.strip()) if env else _PROBE_LAYERS


def _inputs(model_config):
    gold = _GOLDEN[model_config]
    trace = golden.resolve_trace(gold)
    assert trace is not None, f"no golden trace at {gold.trace}; ${gold.trace_env} overrides the path"
    checkpoint = golden.resolve_checkpoint(gold)
    assert checkpoint is not None, f"no checkpoint at {gold.checkpoint}; ${' / $'.join(gold.ckpt_envs)} override it"
    return trace, checkpoint


def _build(mesh_device, device_params, num_links, model_config, config, checkpoint, first, count):
    t0 = time.monotonic()
    last = first + count
    is_first, is_last = first == 0, last == model_config.NUM_LAYERS
    state_dict = v4_model_from_checkpoint(config, checkpoint, load_embed=is_first, load_tail=is_last)
    state_dict["layers"] = V4CheckpointLayers(config, checkpoint, first, count)
    model = TtV4Transformer(
        mesh_device,
        config,
        model_config,
        state_dict,
        count,
        CHUNK,
        first_layer_idx=first,
        is_first_rank=is_first,
        is_last_rank=is_last,
        max_seq_len=SEQ_CACHE,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
    )
    logger.info(f"built layers [{first}, {last}) in {(time.monotonic() - t0) / 60:.1f} min")
    return model


def _capture_gate(gate, k, sink):
    """Wrap ``gate`` so each call appends its input, logits and top-k ids to ``sink``, in sequence order.

    The block uploads the sequence as a contiguous SP shard, so the SP concat the gate's own logits
    composer does is already token order; the ids go through the same composer.
    """
    forward = gate.forward

    def wrapped(x, *args, **kwargs):
        scores, indices, logits = forward(x, *args, **kwargs)
        host_logits = gate._compose_logits_to_host(logits).float()
        host_logits = host_logits.reshape(-1, host_logits.shape[-1])
        sink.append(
            {
                "x": gate._compose_x_to_host(x).float().reshape(host_logits.shape[0], -1),
                "logits": host_logits,
                "ids": gate._compose_logits_to_host(indices).to(torch.int64).reshape(-1, k),
            }
        )
        return scores, indices, logits

    gate.forward = wrapped


def _capture_ref_gate(ref):
    """Hook the reference router; returns the dict it fills with the gate's input, logits and ids."""
    out = {}

    def hook(module, args, result):
        out["x"] = args[0].reshape(-1, args[0].shape[-1]).float()
        out["logits"] = result[0].float()
        out["ids"] = result[2].to(torch.int64)

    ref["mlp"].gate.register_forward_hook(hook)
    return out


def _gate_breakdown(dev, ref_gate, config, k, label, truth_ids=None, truth_logits=None, ref=None):
    """Log where the device's top-k sets part from the others'.

    ``dev`` holds the device gate's input ``x``, ``logits`` and ``ids``. The chain is: device ids vs a
    host top-k on the device's logits with the bias rounded to bf16 as the device holds it (the top-k
    op), that vs the same with the fp32 bias (the bias rounding), that vs a host top-k on fp32 logits
    from the device's ``x`` (the bf16 matmul), and the fp32 host choice vs the golden's or the
    reference's (the gate's input). ``truth_*`` are the golden's ids and pre-activation logits for the same tokens, ``ref`` the
    reference gate's capture.
    """
    weight = ref_gate.weight.detach().float()
    bias = ref_gate.e_score_correction_bias.detach().float()

    def top(logits, bias=bias):
        return (score_activation(logits, config.scoring_func) + bias).topk(k, dim=-1).indices

    def agree(a, b):
        return _topk_agreement(a, b)[0].float().mean()

    host_logits = dev["x"] @ weight.T
    on_dev_bf16 = top(dev["logits"], bias.bfloat16().float())
    on_dev_logits, on_host_logits = top(dev["logits"]), top(host_logits)
    _, logits_pcc = comp_pcc(host_logits, dev["logits"])
    msg = (
        f"dev~topk(dev logits, bf16 bias) {agree(dev['ids'], on_dev_bf16):.4f} "
        f"bf16 bias~fp32 bias {agree(on_dev_bf16, on_dev_logits):.4f} "
        f"topk(dev logits)~topk(host fp32 logits) {agree(on_dev_logits, on_host_logits):.4f} "
        f"logits dev vs host fp32 PCC {logits_pcc:.6f} max|d| {(host_logits - dev['logits']).abs().max():.3e}"
    )
    if truth_ids is not None:
        exact, overlap = _topk_agreement(dev["ids"], truth_ids)
        _, gold_pcc = comp_pcc(truth_logits, dev["logits"])
        msg += (
            f"; vs golden: dev {exact.float().mean():.4f} (overlap {overlap.mean():.4f}) "
            f"host fp32 {agree(on_host_logits, truth_ids):.4f} golden-logits recompute {agree(top(truth_logits), truth_ids):.4f} "
            f"host fp32~golden-logits recompute {agree(on_host_logits, top(truth_logits)):.4f} "
            f"logits dev vs golden PCC {gold_pcc:.6f}"
        )
    if ref is not None:
        _, x_pcc = comp_pcc(ref["x"], dev["x"])
        _, ref_logits_pcc = comp_pcc(ref["logits"], dev["logits"])
        msg += (
            f"; vs reference: gate input PCC {x_pcc:.6f} logits PCC {ref_logits_pcc:.6f} "
            f"dev {agree(dev['ids'], ref['ids']):.4f} host fp32 {agree(on_host_logits, ref['ids']):.4f}"
        )
        if truth_ids is not None:
            msg += f" reference~golden {agree(ref['ids'], truth_ids):.4f}"
    logger.info(f"[{label}] gate: {msg}")


def _norm_at(streams, positions):
    norms = streams[0].flatten(1).norm(dim=-1)
    return {p: float(norms[p]) for p in positions if p < norms.numel()}


def _log_local_error(x_in, dev_out, ref_out, outliers, label):
    """Log the device output against the reference's, both from ``x_in``; all are ``[1, S, n, D]``.

    The stream PCC is mostly the residual both sides carry through unchanged, so the layer's own
    increment over its input is graded too, and each token's error is relative to that increment.
    """
    _, stream_pcc = comp_pcc(ref_out, dev_out)
    d_ref, d_dev = (ref_out - x_in)[0].flatten(1), (dev_out - x_in)[0].flatten(1)
    _, delta_pcc = comp_pcc(d_ref, d_dev)
    cos = torch.nn.functional.cosine_similarity(ref_out[0].flatten(1), dev_out[0].flatten(1), dim=-1)
    rel = (d_dev - d_ref).norm(dim=-1) / d_ref.norm(dim=-1).clamp_min(1e-12)
    ref_n, dev_n, in_n = _norm_at(ref_out, outliers), _norm_at(dev_out, outliers), _norm_at(x_in, outliers)
    at = " ".join(
        f"{p}:in {in_n[p]:.1f} ref {ref_n[p]:.1f} dev {dev_n[p]:.1f} rel {rel[p]:.3f} cos {cos[p]:.4f}" for p in ref_n
    )
    logger.info(
        f"[{label}] local: stream PCC {stream_pcc:.6f} increment PCC {delta_pcc:.6f}; per-token cosine mean "
        f"{cos.mean():.4f} p1 {torch.quantile(cos, 0.01):.4f} min {cos.min():.4f} at {int(cos.argmin())}; "
        f"increment rel err median {rel.median():.4f} p99 {torch.quantile(rel, 0.99):.4f} max {rel.max():.4f} "
        f"at {int(rel.argmax())}; outliers {at}"
    )


@pytest.mark.parametrize("mesh_device, device_params, num_links", _GALAXY, indirect=["mesh_device", "device_params"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_flash_depth_local_error(mesh_device, device_params, num_links):
    trace, checkpoint = _inputs(_FLASH)
    probes = _probe_layers()
    assert probes and probes[0] >= 1, f"probe layers {probes}: layer 0's input is the embedding, not a stream"
    label = f"v4 probe {_FLASH.__name__}"
    config = v4_hf_config(_FLASH, _FLASH.NUM_LAYERS, max_seq_len=SEQ_CACHE)
    n, k = config.hc_mult, config.num_experts_per_tok
    outliers = _OUTLIERS[_FLASH]
    topk_probes = [i for i in probes if config.mlp_layer_types[i] != "hash_moe"]

    model = _build(mesh_device, device_params, num_links, _FLASH, config, checkpoint, 0, _FLASH.NUM_LAYERS)
    captures = {i: [] for i in topk_probes}
    for i in topk_probes:
        _capture_gate(model.layers[i].ffn.gate, k, captures[i])

    keep = set(probes) | {i - 1 for i in probes}
    streams = {}

    def tap(i, h):
        if i in keep:
            streams[i] = streams_to_host(mesh_device, h, n)

    token_ids = trace.token_ids(CHUNK, 0)
    t0 = time.monotonic()
    tt_in = upload_tokens(mesh_device, token_ids)
    out = model(tt_in, actual_isl=CHUNK, actual_start=0, layer_tap=tap)
    ttnn.deallocate(tt_in)
    ttnn.deallocate(out)
    logger.info(f"chunk 0 through {_FLASH.NUM_LAYERS} layers in {time.monotonic() - t0:.0f} s")

    for i in probes:
        t0 = time.monotonic()
        ref = v4_layer_from_checkpoint(config, i, checkpoint)
        ref_gate = _capture_ref_gate(ref)
        ref_out = v4_block_forward(ref, config, streams[i - 1], token_ids)
        layer_label = f"{label} layer {i} {config.layer_types[i]}/{config.mlp_layer_types[i]}"
        _log_local_error(streams[i - 1], streams[i], ref_out, outliers, layer_label)
        if i in captures:
            _gate_breakdown(captures[i][0], ref["mlp"].gate, config, k, layer_label, ref=ref_gate)
        if {i - 1, i} <= set(trace.kept_layers):
            # The golden's own input: how close the reference itself is to vLLM, against the device's error
            # on the same tokens after i layers of its own.
            gold_in = trace.decoder_output(i - 1, 0, CHUNK).reshape(1, CHUNK, n, -1)
            gold_out = trace.decoder_output(i, 0, CHUNK).reshape(1, CHUNK, n, -1)
            _, ref_pcc = comp_pcc(gold_out, v4_block_forward(ref, config, gold_in, token_ids))
            _, dev_pcc = comp_pcc(gold_out, streams[i])
            logger.info(
                f"[{layer_label}] chunk 0 vs golden: reference from golden input {ref_pcc:.6f}, "
                f"device free-running {dev_pcc:.6f}"
            )
        del ref, ref_out
        logger.info(f"  layer {i} reference done in {time.monotonic() - t0:.0f} s")


@pytest.mark.parametrize("mesh_device, device_params, num_links", _GALAXY, indirect=["mesh_device", "device_params"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_flash_layer3_routing(mesh_device, device_params, num_links):
    trace, checkpoint = _inputs(_FLASH)
    i = _ROUTING_LAYER
    label = f"v4 probe {_FLASH.__name__} layer {i}"
    config = v4_hf_config(_FLASH, i + 1, max_seq_len=SEQ_CACHE)
    assert config.mlp_layer_types[i] != "hash_moe", f"layer {i} is hash-routed; the gate split needs a top-k layer"
    n, hidden, k = config.hc_mult, config.hidden_size, config.num_experts_per_tok

    model = _build(mesh_device, device_params, num_links, _FLASH, config, checkpoint, i, 1)
    captured = []
    _capture_gate(model.layers[0].ffn.gate, k, captured)

    dev_out = torch.zeros(1, SEQ_CACHE, n, hidden)
    for chunk in range(N_CHUNKS):
        start = chunk * CHUNK
        tt_in = _upload_streams(mesh_device, trace.layer_input(i, start, start + CHUNK).reshape(1, CHUNK, n, hidden))
        out = model(tt_in, actual_isl=CHUNK, actual_start=start)
        dev_out[:, start : start + CHUNK] = streams_to_host(mesh_device, out, n)
        ttnn.deallocate(tt_in)
        ttnn.deallocate(out)
    assert len(captured) == N_CHUNKS, f"gate ran {len(captured)} times over {N_CHUNKS} chunks"
    dev = {key: torch.cat([c[key] for c in captured]) for key in ("x", "logits", "ids")}

    truth_out = trace.decoder_output(i).reshape(1, SEQ_CACHE, n, hidden)
    _, pcc = comp_pcc(truth_out, dev_out)
    logger.info(f"[{label}] output PCC vs golden, teacher-forced: {pcc:.6f}")
    _log_compressed_kv(trace, config, model, label)

    ref = v4_layer_from_checkpoint(config, i, checkpoint)
    _gate_breakdown(dev, ref["mlp"].gate, config, k, f"{label} all chunks", trace.expert_ids(i), trace.router_logits(i))

    ref_gate = _capture_ref_gate(ref)
    gold_in = trace.layer_input(i, 0, CHUNK).reshape(1, CHUNK, n, hidden)
    ref_out = v4_block_forward(ref, config, gold_in, trace.token_ids(CHUNK, 0))
    c0 = {key: v[:CHUNK] for key, v in dev.items()}
    _gate_breakdown(
        c0,
        ref["mlp"].gate,
        config,
        k,
        f"{label} chunk 0",
        trace.expert_ids(i, 0, CHUNK),
        trace.router_logits(i, 0, CHUNK),
        ref=ref_gate,
    )
    _, ref_pcc = comp_pcc(truth_out[:, :CHUNK], ref_out)
    _, dev_pcc = comp_pcc(truth_out[:, :CHUNK], dev_out[:, :CHUNK])
    _, dev_ref_pcc = comp_pcc(ref_out, dev_out[:, :CHUNK])
    logger.info(
        f"[{label}] chunk 0 output PCC: reference vs golden {ref_pcc:.6f}, device vs golden {dev_pcc:.6f}, "
        f"device vs reference {dev_ref_pcc:.6f}"
    )
