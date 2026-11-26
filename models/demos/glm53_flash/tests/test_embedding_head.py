# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Token embedding / mHC entry and the final head (mean -> norm -> lm_head -> argmax) vs HF, real weights.

Head inputs: the model's real final residual is layer 44's output, which no dump holds yet. The
layer-4 output streams of the R4097 dump (random-token prompt) are used as a STAND-IN distribution
for the head tests; they exercise the arithmetic (stream mean, RMSNorm, 154880-wide LM head,
argmax) on realistic magnitudes, not the model's true next-token predictions.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_embedding_head.py -s
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    default_snapshot_dir,
    load_tensors,
    shard_headers,
)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.embedding import TtEmbedding, padded_rows
from models.demos.glm53_flash.tt.functional_decoder import device_streams_to_host, host_streams_to_device
from models.demos.glm53_flash.tt.model_head import TtModelHead
from models.demos.glm53_flash.tt.sampler import TtSampler

DUMP = Path("bringup/references/glm53_flash/R4097")
HEAD_LAYER = 4  # stand-in streams (see the module docstring)


def _cfg():
    return GLM53FlashConfig.from_snapshot()


def _head_weights():
    snapshot = default_snapshot_dir()
    w = load_tensors(snapshot, [FINAL_NORM_KEY, LM_HEAD_KEY], shard_headers(snapshot))
    return w[FINAL_NORM_KEY], w[LM_HEAD_KEY]


def hf_head_reference(streams: torch.Tensor, norm_w: torch.Tensor, lm_w: torch.Tensor, cfg, dtype) -> torch.Tensor:
    """``lm_head(norm(hc_head(streams)))`` with the transformers modules in ``dtype``.

    ``dtype=float32``: the host reference the device is graded against. ``dtype=bfloat16``: the
    model's own dtype path (hc_head and norm exactly as HF runs them in bf16); its matmul is done in
    fp32 on the bf16-rounded operands and rounded to bf16, i.e. a bf16 GEMM with fp32 accumulation.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperHead, Glm5NextTextRMSNorm

    norm = Glm5NextTextRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = norm_w.to(dtype)
    with torch.no_grad():
        h = norm(Glm5NextTextHyperHead()(streams.to(dtype)))  # [1, S, H] in dtype
        logits = h.float() @ lm_w.to(dtype).float().t()
        return logits.to(dtype).float()


def _decode_streams(first: int, n: int) -> torch.Tensor:
    """``[1, n, 4, 4096]`` bf16: row k = layer-4 decode output of dump step ``first + k``."""
    return torch.cat([torch.load(DUMP / f"layer{HEAD_LAYER}_decode{first + k}.pt")["x_out"] for k in range(n)], dim=1)


def _grade(name: str, ref: torch.Tensor, got: torch.Tensor) -> float:
    """PCC, top-1 agreement and top-5 overlap over the rows of ``[rows, vocab]`` logits."""
    p = pcc(ref, got)
    top1 = float((ref.argmax(-1) == got.argmax(-1)).float().mean())
    k = 5
    r5 = ref.topk(k, dim=-1).indices
    g5 = got.topk(k, dim=-1).indices
    top5 = float((r5.unsqueeze(-1) == g5.unsqueeze(-2)).any(-1).float().sum(-1).mean() / k)
    logger.info(
        f"{name}: pcc {p:.6f}, top-1 agreement {top1:.4f}, top-5 overlap {top5:.4f}, "
        f"max abs diff {(ref - got).abs().max().item():.4f} (logit range {ref.abs().max().item():.2f})"
    )
    return p


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_embedding_vs_hf(mesh_device):
    """Packed streams of the embedding vs HF ``embed_tokens(ids)`` expanded to 4 streams: exact."""
    cfg = _cfg()
    snapshot = default_snapshot_dir()
    table = load_tensors(snapshot, [EMBED_KEY], shard_headers(snapshot))[EMBED_KEY]  # bf16 [V, H]
    emb = TtEmbedding(mesh_device, table, cfg.hc_mult, pad_token_id=cfg.pad_token_id)
    pre = torch.load(DUMP / "prefill_input.pt")
    ids, x0 = pre["prompt_ids"], pre["x0"]  # [1, 4097] int64; HF expanded embedding [1, 4097, 4, 4096] bf16
    S = ids.shape[1]

    # prefill-shaped: [1, S] ids -> [1, 1, T_pad, 16384]
    T_pad = padded_rows(S)
    ids_dev = emb.tokens_to_device(ids, pad_to=T_pad)
    emb(ids_dev)  # compile
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    streams = emb(ids_dev)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"embedding S={S} (T_pad={T_pad}): {(time.perf_counter() - t0) * 1e3:.1f} ms warm")
    assert tuple(streams.shape) == (1, 1, T_pad, cfg.hc_mult * cfg.hidden_size) and streams.dtype == ttnn.float32
    out = device_streams_to_host(streams, cfg.hc_mult, S)  # [1, S, 4, 4096]
    gather = table[ids[0].long()].float().reshape(1, S, 1, -1).expand(-1, -1, cfg.hc_mult, -1)
    for name, ref in [("embed_tokens(ids) expanded", gather), ("dump x0 (HF layer-0 input)", x0.float())]:
        p, d = pcc(ref, out), (ref - out).abs().max().item()
        logger.info(f"prefill embedding vs {name}: pcc {p:.6f}, max abs diff {d}")
        assert d == 0.0 and p > 0.9999
    for i in range(1, cfg.hc_mult):
        assert torch.equal(out[:, :, 0], out[:, :, i])

    # decode-shaped: [1, 1, 1, B] ids (the 8 next tokens of the dump) -> [1, 1, B, 16384]
    B = 8
    toks = torch.tensor([[int(torch.load(DUMP / f"layer0_decode{k}.pt")["token"]) for k in range(B)]])
    streams_d = emb(emb.tokens_to_device(toks))
    assert tuple(streams_d.shape) == (1, 1, B, cfg.hc_mult * cfg.hidden_size)
    out_d = device_streams_to_host(streams_d, cfg.hc_mult, B)
    ref_d = table[toks[0].long()].float().reshape(1, B, 1, -1).expand(-1, -1, cfg.hc_mult, -1)
    d = (ref_d - out_d).abs().max().item()
    logger.info(f"decode embedding B={B}: pcc {pcc(ref_d, out_d):.6f}, max abs diff {d}")
    assert d == 0.0
    # host-side id shapes all flatten to the same row order
    assert torch.equal(emb.tokens_to_host(toks), emb.tokens_to_host(toks.reshape(1, 1, 1, B)))
    assert torch.equal(emb.tokens_to_host(ids[:, :64].reshape(2, 32)), emb.tokens_to_host(ids[:, :64]))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("rows", [256, 4097])
def test_head_vs_hf_prefill(mesh_device, rows):
    """Head on the layer-4 prefill streams (a 256-row slice and all 4097 rows) vs the fp32 HF formula."""
    cfg = _cfg()
    norm_w, lm_w = _head_weights()
    head = TtModelHead(mesh_device, cfg, norm_w, lm_w)
    x = torch.load(DUMP / f"layer{HEAD_LAYER}_prefill.pt")["x_out"][:, :rows]  # [1, rows, 4, 4096] bf16
    streams = host_streams_to_device(x, mesh_device)  # [1, 1, T_pad, 16384] fp32

    head(streams)  # compile
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    logits_dev = head(streams)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"head rows={rows} (T_pad={streams.shape[-2]}): {(time.perf_counter() - t0) * 1e3:.1f} ms warm")
    assert tuple(logits_dev.shape) == (1, 1, streams.shape[-2], cfg.vocab_size) and logits_dev.dtype == ttnn.bfloat16
    got = ttnn.to_torch(logits_dev).float()[0, 0, :rows]  # [rows, V]

    ref32 = hf_head_reference(x, norm_w, lm_w, cfg, torch.float32)[0]
    p = _grade(f"head rows={rows} vs HF fp32", ref32, got)
    assert p >= 0.995
    if rows <= 256:
        ref16 = hf_head_reference(x, norm_w, lm_w, cfg, torch.bfloat16)[0]
        _grade(f"head rows={rows} vs HF bf16 path", ref16, got)
        _grade(f"(HF bf16 path vs HF fp32, host only)", ref32, ref16)

    # last-token helper: slice the stream row first, then the head
    last = ttnn.to_torch(head.last_token_logits(streams, rows)).float()[0, 0, 0]
    p_last = pcc(ref32[rows - 1], last)
    logger.info(
        f"last_token_logits row {rows - 1}: pcc vs HF fp32 {p_last:.6f}, argmax {int(last.argmax())} "
        f"(HF {int(ref32[rows - 1].argmax())}, full-row device {int(got[rows - 1].argmax())}), "
        f"max abs diff vs the full-row device logits {(last - got[rows - 1]).abs().max().item():.4f}"
    )
    assert p_last >= 0.995 and int(last.argmax()) == int(got[rows - 1].argmax())


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_head_decode_shaped(mesh_device):
    """Decode-shaped ``[1, 1, B=8, 16384]`` rows (layer-4 decode outputs): logits + greedy ids."""
    cfg = _cfg()
    norm_w, lm_w = _head_weights()
    head = TtModelHead(mesh_device, cfg, norm_w, lm_w)
    B = 8
    x = _decode_streams(0, B)  # [1, 8, 4, 4096]
    streams = host_streams_to_device(x, mesh_device, pad_to=B)
    assert tuple(streams.shape) == (1, 1, B, cfg.hc_mult * cfg.hidden_size)

    logits_dev, ids_dev = head.forward_greedy(streams)
    assert tuple(logits_dev.shape) == (1, 1, B, cfg.vocab_size)
    assert tuple(ids_dev.shape) == (1, 1, B, 1) and ids_dev.dtype == ttnn.uint32
    got = ttnn.to_torch(logits_dev).float()[0, 0]
    ids = ttnn.to_torch(ids_dev).view(torch.int32).long().reshape(-1)

    ref32 = hf_head_reference(x, norm_w, lm_w, cfg, torch.float32)[0]
    p = _grade(f"head decode B={B} vs HF fp32", ref32, got)
    assert p >= 0.995
    _grade(f"head decode B={B} vs HF bf16 path", hf_head_reference(x, norm_w, lm_w, cfg, torch.bfloat16)[0], got)
    logger.info(f"greedy ids {ids.tolist()}; HF fp32 argmax {ref32.argmax(-1).tolist()}")
    assert torch.equal(ids, got.argmax(-1)), "device argmax must agree with the device logits"
    logger.info(f"greedy id agreement with HF fp32: {int((ids == ref32.argmax(-1)).sum())}/{B}")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_head_greedy_trace(mesh_device):
    """Capture (head + argmax) once on a persistent ``[1, 1, B, 16384]`` input, replay on new rows;
    replays must be bit-identical to the eager results of the same inputs."""
    cfg = _cfg()
    norm_w, lm_w = _head_weights()
    head = TtModelHead(mesh_device, cfg, norm_w, lm_w)
    B = 8
    xa, xb = _decode_streams(0, B), _decode_streams(4, B)  # two different row sets (steps 0..7 and 4..11)

    def host(x):
        packed = torch.zeros(1, 1, B, cfg.hc_mult * cfg.hidden_size, dtype=torch.float32)
        packed[0, 0] = x.reshape(B, -1).float()
        return ttnn.from_torch(packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    inp = host_streams_to_device(xa, mesh_device, pad_to=B)  # persistent decode input

    def read(logits, ids):
        return ttnn.to_torch(logits).float()[0, 0], ttnn.to_torch(ids).view(torch.int32).long().reshape(-1)

    # eager results (also compiles the programs)
    eager = {}
    for name, x in [("a", xa), ("b", xb)]:
        ttnn.copy_host_to_device_tensor(host(x), inp)
        eager[name] = read(*head.forward_greedy(inp))
    assert not torch.equal(eager["a"][1], eager["b"][1]) or not torch.equal(eager["a"][0], eager["b"][0])

    # capture on input a, replay on a and b
    ttnn.copy_host_to_device_tensor(host(xa), inp)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    logits_tr, ids_tr = head.forward_greedy(inp)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    for name, x in [("a", xa), ("b", xb), ("a", xa)]:
        ttnn.copy_host_to_device_tensor(host(x), inp)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        logits, ids = read(logits_tr, ids_tr)
        assert torch.equal(logits, eager[name][0]), f"replay logits differ from eager ({name})"
        assert torch.equal(ids, eager[name][1]), f"replay ids differ from eager ({name})"
        logger.info(f"traced greedy replay on rows {name}: ids {ids.tolist()} == eager (logits bit-identical)")

    n = 10
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"traced head+argmax B={B}: {(time.perf_counter() - t0) / n * 1e3:.2f} ms per replay")
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_head_sampler(mesh_device):
    """``tt/sampler.py`` on the head's decode logits (B = 8 real rows): all-greedy parameters reproduce the head's
    argmax ids bit-exactly; sampled rows draw inside their top-k, identically under the same seeds and differently
    under new seeds (hot rows); the sampler's eager step cost is logged."""
    cfg = _cfg()
    norm_w, lm_w = _head_weights()
    head = TtModelHead(mesh_device, cfg, norm_w, lm_w)
    B = 8
    streams = host_streams_to_device(_decode_streams(0, B), mesh_device, pad_to=B)
    logits_dev, ids_dev = head.forward_greedy(streams)
    ids = ttnn.to_torch(ids_dev).view(torch.int32).long().reshape(-1)
    logits = ttnn.to_torch(logits_dev).float()[0, 0]
    top32 = torch.topk(logits, 32, dim=-1).indices

    def run():
        return ttnn.to_torch(sampler(logits_dev, ids_dev)).view(torch.int32).long().reshape(-1)

    sampler = TtSampler(mesh_device, users=B)
    sampler.set_params(k=[1] * B, p=[1.0] * B, inv_temp=[1.0] * B, greedy=[True] * B)
    sampler.set_seeds(list(range(1, B + 1)))
    out = run()
    assert torch.equal(out, ids), f"all-greedy sampler must return the head's argmax: {out.tolist()} vs {ids.tolist()}"
    # rows 0-3 greedy, rows 4-7 sampled hot (1/T = 0.1, top-p 0.95), row 7 with top-k 10
    k = [1] * 4 + [32, 32, 32, 10]
    sampler.set_params(k=k, p=[0.95] * B, inv_temp=[1.0] * 4 + [0.1] * 4, greedy=[True] * 4 + [False] * 4)
    draws = []
    for s in range(4):
        sampler.set_seeds([100 + s * 31 + u for u in range(B)])
        draws.append(run())
    d = torch.stack(draws)
    assert torch.equal(d[:, :4], ids[:4].expand(4, 4)), "greedy rows keep the argmax beside sampled companions"
    for r in range(4, B):
        allowed = set(top32[r, : k[r]].tolist())
        assert all(int(v) in allowed for v in d[:, r]), f"row {r} drew outside its top-{k[r]}: {d[:, r].tolist()}"
    sampler.set_seeds([100 + u for u in range(B)])
    assert torch.equal(run(), d[0]), "the same seeds must reproduce the draw"
    varied = int((d[:, 4:] != d[0, 4:]).any(0).sum())
    logger.info(
        f"sampled rows over 4 seed sets: {d[:, 4:].tolist()}; {varied}/4 rows varied; greedy rows {ids[:4].tolist()}"
    )
    assert varied >= 1, "hot rows (1/T = 0.1, top-p 0.95) must vary across seeds"
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(10):
        sampler(logits_dev, ids_dev)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"sampler step B={B}: {(time.perf_counter() - t0) / 10 * 1e3:.2f} ms (eager, one chip)")
