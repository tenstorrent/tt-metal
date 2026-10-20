# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-expert GPTQ for the bfloat4_b experts, on the host (no device).

The fp32 reference runs calibration text (C4) layer by layer. For every MoE layer it collects each expert's input
second moments, weighted by the routing weight its output is scaled with (tt-quant's per-expert MoE GPTQ):
``H_gate_up = sum w^2 x x^T`` over the tokens routed to the expert, ``H_down = sum w^2 a a^T`` with
``a = silu(gate) * up``. GPTQ (Frantar et al.) then quantizes each expert in its device layout [K, N] (16-value
blocks along N), row by row along K, with the bfloat4_b projector of tt.common.bfp_project_blocks restricted to
the exponents {e, e + 1} (tt-quant's ``gptq-noclip``) and the inverse-Hessian error feedback. All 256 experts of
a layer run as one batch. Experts with fewer than ``--min-tokens`` routed tokens are MSE-prepared instead.

The output is the same as tests/perf/prep_experts.py's (<out>/layer<i>.pt with w_gate_up / w_down in device
layout, every value on the bfloat4_b grid), for the cache conversion with MIMO_EXPERT_QUANT=gptq and
MIMO_EXPERT_PREP_DIR=<out>.

    python models/experimental/mimo_v2_6_flash/tests/perf/gptq_experts.py --out DIR --calib c4-train.00000-of-01024.json.gz
"""

import argparse
import gzip
import json
import os
import time

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tt.common import BFP_MANTISSA_BITS, bfp_mse_prepare, bfp_project_blocks
from models.experimental.mimo_v2_6_flash.tt.moe import expert_weights_host


def calibration_sequences(tok, path, num, length):
    """``num`` sequences of ``length`` tokens from C4 documents, concatenated with EOS between them."""
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    stream, out = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            stream.extend(tok(json.loads(line)["text"])["input_ids"] + [eos])
            while len(stream) >= length:
                out.append(torch.tensor(stream[:length]))
                stream = stream[length:]
                if len(out) == num:
                    return out
    return out


def project_rows(w, bits, offsets):
    """[E, N] rows -> on-grid values, blocks of 16 along N."""
    e, n = w.shape
    return bfp_project_blocks(w.reshape(-1, 16), bits, offsets).reshape(e, n)


def gptq_batched(w, h, bits, damping=0.3, block=128, inner=16, offsets=(0, 1)):
    """w [E, K, N] fp32 (device layout), h [E, K, K] -> on-grid [E, K, N]. act-order per expert; lazy updates in two
    levels (``inner`` rows by rank-1 updates, ``block`` rows by one batched matmul) to keep the memory traffic of
    the [E, K, N] updates down."""
    E, K, N = w.shape
    w = w.clone()
    h = h.double().clone()
    diag = torch.diagonal(h, dim1=1, dim2=2)
    dead = diag == 0
    diag[dead] = 1.0
    w[dead] = 0.0
    perm = torch.argsort(diag, dim=1, descending=True)
    w = w.gather(1, perm[:, :, None].expand(E, K, N))
    h = h.gather(1, perm[:, :, None].expand(E, K, K)).gather(2, perm[:, None, :].expand(E, K, K))
    damp = damping * torch.diagonal(h, dim1=1, dim2=2).mean(-1)
    h += damp[:, None, None] * torch.eye(K, dtype=h.dtype)
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True).float()
    del h
    q = torch.empty_like(w)
    for b1 in range(0, K, block):
        b2 = min(b1 + block, K)
        wb = w[:, b1:b2, :].clone()
        errb = torch.empty_like(wb)
        for i1 in range(0, b2 - b1, inner):
            i2 = min(i1 + inner, b2 - b1)
            u1 = u[:, b1 + i1 : b1 + i2, b1 + i1 : b1 + i2]
            for j in range(i2 - i1):
                row = wb[:, i1 + j, :]
                qr = project_rows(row, bits, offsets)
                q[:, b1 + i1 + j, :] = qr
                e = (row - qr) / u1[:, j, j, None]
                if j + 1 < i2 - i1:
                    wb[:, i1 + j + 1 : i2, :] -= u1[:, j, j + 1 :, None] * e[:, None, :]
                errb[:, i1 + j, :] = e
            if i2 < b2 - b1:
                ui = u[:, b1 + i1 : b1 + i2, b1 + i2 : b2]
                wb[:, i2:, :] -= torch.bmm(ui.transpose(1, 2), errb[:, i1:i2, :])
        if b2 < K:
            w[:, b2:, :] -= torch.bmm(u[:, b1:b2, b2:].transpose(1, 2), errb)
    inv = torch.argsort(perm, dim=1)
    return q.gather(1, inv[:, :, None].expand(E, K, N))


def expert_hessians(x, idx, wts, experts, expert_ids):
    """x [T, H] fp32 MoE input, idx / wts [T, k]. -> (H_gate_up [n, H, H], H_down [n, I, I], tokens [n]) for the
    experts ``expert_ids`` (n of them)."""
    hidden, inter = x.shape[1], experts["down_proj"].shape[-1]
    n = len(expert_ids)
    h_gu = torch.zeros(n, hidden, hidden)
    h_d = torch.zeros(n, inter, inter)
    count = torch.zeros(n, dtype=torch.long)
    for j, e in enumerate(expert_ids):
        t, slot = torch.where(idx == e)
        if t.numel() == 0:
            continue
        xe = x[t]
        we = wts[t, slot, None].float()
        act = F.silu(xe @ experts["gate_proj"][e].float().T) * (xe @ experts["up_proj"][e].float().T)
        xw, aw = xe * we, act * we
        h_gu[j] = xw.T @ xw
        h_d[j] = aw.T @ aw
        count[j] = t.numel()
    return h_gu, h_d, count


def quantize_layer(cfg, args, experts, x, idx, wts, bits, inter_local, dst, i, t0, chunk=32):
    """GPTQ one MoE layer's experts from its captured input x [T, H] and routing (idx, wts [T, k]); save to dst.
    Experts go ``chunk`` at a time to bound memory (~60 GB per layer instead of ~200 GB)."""
    t1 = time.time()
    # Device layout: gate/up -> [E, H, tp * 2 * I_local] (chip c: [gate_c | up_c]); down -> [E, I, H].
    w_gate_up, w_down = expert_weights_host(experts, args.tp, inter_local, ttnn.bfloat4_b, mse=False)
    w_gate_up, w_down = w_gate_up[0], w_down[0]
    q_gate_up, q_down = torch.empty_like(w_gate_up), torch.empty_like(w_down)  # bf16 holds bfloat4_b values
    num_experts, few, t_h = w_gate_up.shape[0], 0, 0.0
    for c0 in range(0, num_experts, chunk):
        ids = list(range(c0, min(c0 + chunk, num_experts)))
        th = time.time()
        h_gu, h_d, count = expert_hessians(x, idx, wts, experts, ids)
        t_h += time.time() - th
        wg, wd = w_gate_up[ids].float(), w_down[ids].float()
        qg = gptq_batched(wg, h_gu, bits, args.damping)
        del h_gu
        qd = gptq_batched(wd, h_d, bits, args.damping)
        del h_d
        low = count < args.min_tokens
        if low.any():
            qg[low] = bfp_mse_prepare(wg[low], ttnn.bfloat4_b).float()
            qd[low] = bfp_mse_prepare(wd[low], ttnn.bfloat4_b).float()
        few += int(low.sum())
        q_gate_up[ids] = qg.to(torch.bfloat16)
        q_down[ids] = qd.to(torch.bfloat16)
    torch.save({"w_gate_up": q_gate_up.unsqueeze(0), "w_down": q_down.unsqueeze(0)}, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    logger.info(
        f"layer {i}: inputs {t1 - t0:.0f}s, hessians {t_h:.0f}s, gptq {time.time() - t1 - t_h:.0f}s; "
        f"{few} experts under {args.min_tokens} tokens"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--calib", required=True, help="C4 json.gz shard")
    parser.add_argument("--seqs", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--layers", default="1-47", help="MoE layers to quantize (a-b or a comma list)")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--damping", type=float, default=0.3)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument(
        "--capture-dir",
        default=None,
        help="stage 1: run the reference and save every MoE layer's input and routing here (no GPTQ)",
    )
    parser.add_argument(
        "--from-capture",
        default=None,
        help="stage 2: GPTQ the --layers from a stage-1 capture dir (layers are independent: run several at once)",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if "-" in args.layers:
        a, b = map(int, args.layers.split("-"))
        layers = set(range(a, b + 1))
    else:
        layers = {int(x) for x in args.layers.split(",")}
    os.makedirs(args.out, exist_ok=True)

    path = default_model_path()
    cfg = MiMoConfig.from_model_path(path)
    ckpt = MiMoCheckpoint(path, cfg)
    bits = BFP_MANTISSA_BITS[ttnn.bfloat4_b]
    inter_local = cfg.moe_intermediate_size // args.tp
    if args.from_capture:
        for i in sorted(layers):
            dst = os.path.join(args.out, f"layer{i}.pt")
            if not cfg.is_moe(i) or os.path.exists(dst):
                continue
            t0 = time.time()
            cap = torch.load(os.path.join(args.from_capture, f"layer{i}.pt"))
            experts = ckpt.experts(i)
            quantize_layer(cfg, args, experts, cap["x"].float(), cap["idx"], cap["wts"], bits, inter_local, dst, i, t0)
        return
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    seqs = calibration_sequences(tok, args.calib, args.seqs, args.seq_len)
    logger.info(f"{len(seqs)} calibration sequences of {args.seq_len} tokens")
    embedding = ckpt.embedding()
    hidden = [embedding[t].float() for t in seqs]
    del embedding
    captured = []
    plain_moe = ref.moe

    def capturing_moe(x, router, experts, cfg_):
        idx, wts = ref.route(x, router, cfg_)
        captured.append((x.float(), idx, wts))
        return plain_moe(x, router, experts, cfg_)

    last = max(layers)
    for i in range(last + 1):
        t0 = time.time()
        quantize = cfg.is_moe(i) and i in layers and not os.path.exists(os.path.join(args.out, f"layer{i}.pt"))
        layer = ref.ReferenceDecoderLayer(cfg, i, ckpt=ckpt)
        captured.clear()
        lw = layer.w
        with torch.no_grad():
            # Attention per sequence, then the FFN over every token at once (one pass over the experts).
            post, lens = [], []
            for s_, t in enumerate(seqs):
                xs = hidden[s_]
                hn = ref.rms_norm(xs, lw["input_layernorm"], cfg.layernorm_epsilon)
                xs = xs + ref.attention(hn, lw["attn"], layer.dims, cfg.attention_value_scale, torch.arange(t.numel()))
                hidden[s_] = xs
                post.append(ref.rms_norm(xs, lw["post_attention_layernorm"], cfg.layernorm_epsilon))
                lens.append(t.numel())
            hcat = torch.cat(post)
            if layer.is_moe:
                ffn = (capturing_moe if quantize else plain_moe)(hcat, lw["router"], lw["experts"], cfg)
            else:
                ffn = ref.dense_mlp(hcat, lw["mlp"])
            for s_, part in enumerate(torch.split(ffn, lens)):
                hidden[s_] = hidden[s_] + part
        t1 = time.time()
        if quantize:
            x = torch.cat([c[0] for c in captured])
            idx = torch.cat([c[1] for c in captured])
            wts = torch.cat([c[2] for c in captured])
            if args.capture_dir:
                os.makedirs(args.capture_dir, exist_ok=True)
                torch.save(
                    dict(x=x.to(torch.bfloat16), idx=idx, wts=wts), os.path.join(args.capture_dir, f"layer{i}.pt")
                )
                logger.info(f"layer {i}: forward {t1 - t0:.0f}s, captured {x.shape[0]} tokens")
            else:
                dst = os.path.join(args.out, f"layer{i}.pt")
                quantize_layer(cfg, args, layer.w["experts"], x, idx, wts, bits, inter_local, dst, i, t0)
        else:
            logger.info(f"layer {i}: forward {t1 - t0:.0f}s")
        del layer


if __name__ == "__main__":
    main()
