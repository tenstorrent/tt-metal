# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The official arithmetic per step (F80): DeepSeek's inference/model.py at its shipped dtypes, tapped per step.

``capture(model_path, layers, tokens, chunk, chunk_size)`` builds the official oracle (``hf_oracle.build_oracle``,
bf16 default dtype = the GPU configuration of ``generate.py``) with the first ``max(layers) + 1`` blocks, prefills
``tokens[: (chunk + 1) * chunk_size]`` from position 0 in one call (model.py's only prefill path: a start_pos > 0 is a
one-token decode step) and returns, for the rows of chunk ``chunk``, every step of ds41_ref.block_steps:

    {layer: {step: {"inputs": [the framework step's inputs, in its order, as model.py held them], "output": ...}}}

model.py is not changed: the taps are forward (pre-)hooks on its modules and per-instance wrappers of Block.hc_mixes /
hc_pre / hc_post (Block.forward calls them through self). Map (framework step -> model.py):

  engram_rows    Engram.embed output (bf16 [S, 24, 256] -> [S, 6144])
  engram_kv      Engram.wkv (input: the flattened rows; output bf16)
  engram_add     Engram.forward (input streams + the block's pending pre_mix, packed; output streams bf16, same pre)
  attn_hc        Block.hc_mixes with fn = hc_attn_fn (input: the block's streams + pre_mix; output fp32 [pre|post|comb])
  attn_collapse  1st Block.hc_pre (streams, pre_mix) -> bf16
  attn_norm      Block.attn_norm
  q_a            Attention.q_norm output (q_norm(wq_a(x)))
  kv             Attention.kv_norm output object: RoPE'd and fp8 q-dq'd in place by _window_kv (read after the run)
  compressor     the rows this chunk wrote to Attention.compress_kv_cache (RoPE'd latent, fp4 x e4m3 per 16) and
                 Indexer.k_cache (RoPE'd index key, fp4 x ue8m0 per 32), concatenated: [m, D + 128]
  indexer        Indexer.forward output minus its ``offset`` (-1 kept), padded to index_topk with -1
  attention      Attention.forward output; inputs q_resid, kv, (compressed), (topk) as above
  attn_residual  1st Block.hc_post (attn_out, residual streams, post, comb); inputs packed / mixes / attn_out
  ffn_hc         Block.hc_mixes with fn = hc_ffn_fn (input: h_mid, the 1st hc_post output)
  ffn_collapse   2nd Block.hc_pre (h_mid, attn pre) -> bf16
  ffn_norm       Block.ffn_norm
  router         MoE.gate: (weights, indices) scattered dense [S, n_experts] fp32
  experts        MoE.forward's fp32 ``y`` before the shared expert: rebuilt from the routed experts' outputs in the
                 same order with the same ops (``y[idx] += expert(...)``, experts in id order), so it is bit-equal
  shared_expert  MoE.shared_experts output (bf16)
  moe_add        MoE.forward output (``(y + shared).type_as(x)``, bf16); inputs experts (fp32), shared (bf16)
  ffn_residual   2nd Block.hc_post; output streams packed with the FFN pre (the next block's pre_mix)

Packed boundaries ([S, 4H + 4]: the 4 streams, then the 4 pending pre weights) hold bf16 streams and fp32 pre weights
in model.py; the framework takes one tensor (official_taps.yaml marks it bf16, the streams' dtype).
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config, block_steps


def _pack(streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """streams [S, 4, H] (bf16) + pre [S, 4] (fp32) -> [S, 4H + 4] fp32 (the values unchanged)."""
    return torch.cat([streams.flatten(1).float(), pre.float()], dim=-1)


def _mixes(pre, post, comb) -> torch.Tensor:
    return torch.cat([pre, post, comb.flatten(-2)], dim=-1)


def capture(model_path: str, layers, tokens: torch.Tensor, chunk: int, chunk_size: int, states: bool = False):
    """The taps (module docstring); with ``states``, also ``{layer: {"start": .., "end": ..}}``: the official code's
    own state tensors (DS41Reference.state_tensors format) as of the chunk's start and end, for a context built from
    the official prefix instead of the golden's (diagnostics)."""
    from models.demos.deepseek_v41_flash_d_p.reference.hf_oracle import build_oracle

    cfg = Config.from_dir(model_path)
    layers = sorted(set(int(i) for i in layers))
    n = (chunk + 1) * chunk_size
    r0, r1 = chunk * chunk_size, n
    tokens = torch.as_tensor(tokens).long().reshape(-1)[:n]
    assert tokens.numel() == n, f"{tokens.numel()} tokens < (chunk + 1) * chunk_size = {n}"
    orc = build_oracle(model_path, num_layers=max(layers) + 1, dtype=torch.bfloat16, max_seq_len=n)

    def rows(t):  # [1, n, ...] -> this chunk's rows, own storage
        return t[0, r0:r1].clone()

    cap = {L: {} for L in layers}
    for L in layers:
        blk, c = orc.model.layers[L], cap[L]
        calls = {"pre": 0, "post": 0}
        om, opre, opost = blk.hc_mixes, blk.hc_pre, blk.hc_post

        def mixes(x, fn, sc, base, blk=blk, c=c, om=om):
            out = om(x, fn, sc, base)
            tag = "attn_hc" if fn is blk.hc_attn_fn else "ffn_hc"
            c[tag] = dict(x=rows(x), out=rows(_mixes(*out)))
            return out

        def pre(x, pm, c=c, calls=calls, opre=opre):
            y = opre(x, pm)
            k = calls["pre"]
            calls["pre"] += 1
            if k < 2:  # a 3rd call is Transformer's final collapse after the last block
                c[("attn_collapse", "ffn_collapse")[k]] = dict(x=rows(x), pre=rows(pm), out=rows(y))
            return y

        def post(x, res, p, cb, c=c, calls=calls, opost=opost):
            y = opost(x, res, p, cb)
            k = calls["post"]
            calls["post"] += 1
            if k < 2:
                c[("attn_residual", "ffn_residual")[k]] = dict(y=rows(x), res=rows(res), out=rows(y))
            return y

        blk.hc_mixes, blk.hc_pre, blk.hc_post = mixes, pre, post

        def put(tag, what="out", c=c):
            def h(m, inp, out):
                c.setdefault(tag, {})[what] = rows(out)
                c[tag]["in"] = rows(inp[0])

            return h

        blk.register_forward_pre_hook(lambda m, args, c=c: c.__setitem__("pre_mix", rows(args[2])))
        blk.attn_norm.register_forward_hook(put("attn_norm"))
        blk.ffn_norm.register_forward_hook(put("ffn_norm"))
        blk.attn.register_forward_hook(put("attention"))
        blk.attn.q_norm.register_forward_hook(lambda m, i, out, c=c: c.__setitem__("q_resid", rows(out)))
        # kv_norm's output is RoPE'd and fp8-quantized in place by _window_kv afterwards: keep it, read it after the run
        blk.attn.kv_norm.register_forward_hook(lambda m, i, out, c=c: c.__setitem__("kv_obj", out))
        if blk.attn.indexer is not None:
            blk.attn.indexer.register_forward_pre_hook(lambda m, args, c=c: c.__setitem__("idx_offset", int(args[4])))
            blk.attn.indexer.register_forward_hook(
                lambda m, i, out, c=c: c.update(topk_raw=rows(out), topk_full=out[0].clone())
            )

        moe = blk.ffn

        def gate_hook(m, inp, out, c=c, moe=moe):
            w, idx = out
            dense = torch.zeros(w.shape[0], m.weight.shape[0], dtype=torch.float32).scatter_(1, idx.long(), w.float())
            c["router"] = dict(out=dense[r0:r1].clone())
            # the routed experts' fp32 sum, rebuilt as MoE.forward builds it (the gate runs again in forward after the
            # prefetch pre-hook, so the last call is the one the experts use)
            c["_indices"] = idx
            c["_y"] = torch.zeros(w.shape[0], moe.dim, dtype=torch.float32)

        moe.gate.register_forward_hook(gate_hook)
        for e, ex in enumerate(moe.experts):
            if ex is None:
                continue

            def ex_hook(m, inp, out, e=e, c=c):
                idx, _ = torch.where(c["_indices"] == e)
                c["_y"][idx] += out

            ex.register_forward_hook(ex_hook)
        moe.shared_experts.register_forward_hook(
            lambda m, i, out, c=c: c.__setitem__("shared", dict(out=out[r0:r1].clone()))
        )
        moe.register_forward_hook(lambda m, i, out, c=c: c.__setitem__("moe", dict(out=rows(out))))

        if blk.engram is not None:
            eg = blk.engram
            eg.embed.register_forward_hook(lambda m, i, out, c=c: c.__setitem__("engram_rows", rows(out).flatten(1)))
            eg.wkv.register_forward_hook(put("engram_kv"))
            eg.register_forward_pre_hook(lambda m, args, c=c: c.__setitem__("engram_in", rows(args[0])))
            eg.register_forward_hook(lambda m, i, out, c=c: c.__setitem__("engram_out", rows(out)))

    mod = orc.mod
    if states and cfg.cand_source in layers:
        oi = orc.model.layers[cfg.cand_source].attn.indexer

        def cand_hook(m, i, o):
            cap[cfg.cand_source]["cand_mask"] = mod.shared_attn.candidates[0].clone()

        oi.register_forward_hook(cand_hook)
    orc._run(tokens[None], 0, full_logits=False)

    out = {}
    for L in layers:
        blk, c = orc.model.layers[L], cap[L]
        r = cfg.ratio(L)
        c["kv_full"] = c["kv_obj"][0]
        kv = c.pop("kv_obj")[0, r0:r1].clone()
        compressed = topk = None
        if cfg.is_kv_source(L):
            g0, g1 = r0 // r, r1 // r
            ckv = blk.attn.compress_kv_cache[0, g0:g1].clone()
            ik = blk.attn.indexer.k_cache[0, g0:g1].clone()
            compressed = torch.cat([ckv.float(), ik.float()], dim=-1)
        if cfg.is_index_source(L):
            t = c["topk_raw"]
            t = torch.where(t >= 0, t - c["idx_offset"], torch.full_like(t, -1))
            topk = torch.full((t.shape[0], cfg.index_topk), -1, dtype=torch.int32)
            topk[:, : t.shape[1]] = t.to(torch.int32)
        pm = c["pre_mix"]
        ah, fh = c["attn_hc"], c["ffn_hc"]
        ac, fc = c["attn_collapse"], c["ffn_collapse"]
        ar, fr = c["attn_residual"], c["ffn_residual"]
        experts_out = c["_y"][r0:r1].clone()
        steps = {}
        if cfg.is_engram(L):
            steps["engram_rows"] = dict(inputs=[], output=c["engram_rows"])
            steps["engram_kv"] = dict(inputs=[c["engram_kv"]["in"]], output=c["engram_kv"]["out"])
            steps["engram_add"] = dict(
                inputs=[_pack(c["engram_in"], pm), c["engram_kv"]["out"]], output=_pack(c["engram_out"], pm)
            )
        steps["attn_hc"] = dict(inputs=[_pack(ah["x"], pm)], output=ah["out"])
        steps["attn_collapse"] = dict(inputs=[_pack(ac["x"], ac["pre"])], output=ac["out"])
        steps["attn_norm"] = dict(inputs=[c["attn_norm"]["in"]], output=c["attn_norm"]["out"])
        steps["q_a"] = dict(inputs=[c["attention"]["in"]], output=c["q_resid"])
        steps["kv"] = dict(inputs=[c["attention"]["in"]], output=kv)
        attn_in = [c["q_resid"], kv]
        if compressed is not None:
            steps["compressor"] = dict(inputs=[c["attention"]["in"]], output=compressed)
            attn_in.append(compressed)
        if topk is not None:
            idx_in = [c["attention"]["in"], c["q_resid"]] + ([compressed] if compressed is not None else [])
            steps["indexer"] = dict(inputs=idx_in, output=topk)
            attn_in.append(topk)
        steps["attention"] = dict(inputs=attn_in, output=c["attention"]["out"])
        steps["attn_residual"] = dict(inputs=[_pack(ar["res"], pm), ah["out"], ar["y"]], output=ar["out"].flatten(1))
        steps["ffn_hc"] = dict(inputs=[fh["x"].flatten(1)], output=fh["out"])
        steps["ffn_collapse"] = dict(inputs=[fc["x"].flatten(1), ah["out"]], output=fc["out"])
        steps["ffn_norm"] = dict(inputs=[c["ffn_norm"]["in"]], output=c["ffn_norm"]["out"])
        steps["router"] = dict(inputs=[c["ffn_norm"]["out"]], output=c["router"]["out"])
        steps["experts"] = dict(inputs=[c["ffn_norm"]["out"], c["router"]["out"]], output=experts_out)
        steps["shared_expert"] = dict(inputs=[c["ffn_norm"]["out"]], output=c["shared"]["out"])
        steps["moe_add"] = dict(inputs=[experts_out, c["shared"]["out"]], output=c["moe"]["out"])
        steps["ffn_residual"] = dict(
            inputs=[fr["res"].flatten(1), fh["out"], fr["y"]], output=_pack(fr["out"], fh["out"][:, : cfg.hc])
        )
        names = [st.name for st in block_steps(cfg, L)]
        assert sorted(steps) == sorted(names), (L, sorted(set(steps) ^ set(names)))
        out[L] = {k: steps[k] for k in names}
    st = _states(orc, cfg, cap, layers, tokens, r0, r1) if states else None
    del orc
    return (out, st) if states else out


def _states(orc, cfg, cap, layers, tokens, r0, r1) -> dict:
    """The official state of each layer at r0 and r1 in ds41_ref's State layout (engram_tail: the token ids')."""
    res = {}
    for L in layers:
        blk, c, r = orc.model.layers[L], cap[L], cfg.ratio(L)
        res[L] = {}
        for tag, length in (("start", r0), ("end", r1)):
            d = {}
            for n in cfg.state_names(L):
                if n == "window_kv":
                    w = torch.zeros(cfg.window, cfg.head_dim)
                    pos = torch.arange(max(0, length - cfg.window), length)
                    w[pos % cfg.window] = c["kv_full"][pos].float()
                    d[n] = w
                elif n == "compress_kv":
                    d[n] = blk.attn.compress_kv_cache[0, : length // r].float().clone()
                elif n == "index_k":
                    d[n] = blk.attn.indexer.k_cache[0, : length // r].float().clone()
                elif n == "compress_partial":
                    assert length % r == 0
                    d[n] = torch.zeros(2, r, cfg.head_dim)
                elif n == "topk_idx":
                    t = c["topk_full"][:length]
                    t = torch.where(t >= 0, t - c["idx_offset"], torch.full_like(t, -1))
                    d[n] = torch.full((length, cfg.index_topk), -1, dtype=torch.int32)
                    d[n][:, : t.shape[1]] = t.to(torch.int32)
                elif n == "candidates":
                    m = c["cand_mask"][:length]  # [length, width] bool over compressed positions
                    nb = -(-m.shape[1] // cfg.cand_block)
                    m = torch.nn.functional.pad(m, (0, nb * cfg.cand_block - m.shape[1])).unflatten(-1, (nb, -1))
                    blocks = m.any(-1)
                    ids = torch.full((length, cfg.cand_topk_blocks), -1, dtype=torch.int32)
                    for q in range(length):
                        b = blocks[q].nonzero().flatten()
                        ids[q, : b.numel()] = b.to(torch.int32)
                    d[n] = ids
                elif n == "engram_tail":
                    d[n] = None  # the token ids' (identical in the golden): filled by the caller
            res[L][tag] = d
    return res
