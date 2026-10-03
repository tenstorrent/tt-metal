# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference chain for DeepSeek-V4.1-Flash decode: prefill S tokens for B users through all layers, then one
decode token, saving per layer what the device chain needs.

Per layer L (files under --out):
    layer_{L}.pt = {
      "dec_in": decode-step input streams [B,1,4,D], "pre_in": [B,1,4],
      "dec_out": decode-step output streams, "pre_out": ffn pre,
      "state": window ring / compressed cache / compressor state snapshot taken after prefill,
      "gate_cutoff": calibration of the router bias shift (mean 6th-ranked score+bias over the prefill FFN inputs),
    }
and ``final.pt`` = {"logits": [B, vocab], "tokens": ..., "decode_tokens": ...} when all layers and the head ran.

Run:  python -m models.demos.blackhole.deepseek_v41_flash.reference.ref_chain --layers 0-39 --engram
"""

import argparse
import gc
import os
import time

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards

OUT = "/mnt/tt-data/ssinghal/dsv4-chain"


def parse_layers(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--S", type=int, default=9, help="prefill tokens per user (decode runs at position S)")
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--engram", action="store_true")
    ap.add_argument("--head", action="store_true", help="also apply the final norm + LM head (needs shard 43)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--steps", type=int, default=1, help="decode steps (teacher-forced random tokens) after the prefill"
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "32")))
    ref_kernels.FAKE_QUANT = False  # unquantised reference: fp8/fp4 activation rounding is not reproduced on device
    g = torch.Generator().manual_seed(a.seed)
    pre_tok = torch.randint(1000, 100000, (a.B, a.S), generator=g)
    dec_tok = torch.randint(1000, 100000, (a.B, a.steps), generator=g)
    torch.save({"prefill_tokens": pre_tok, "decode_tokens": dec_tok}, os.path.join(a.out, "tokens.pt"))

    h_pre, pm_pre = R.embed_tokens(pre_tok)
    emb = [R.embed_tokens(dec_tok[:, i : i + 1]) for i in range(a.steps)]
    h_decs, pm_decs = [e[0] for e in emb], [e[1] for e in emb]  # per decode step
    engram = hashes_pre = None
    hashes_decs = []
    if a.engram:
        from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostEngram

        engram = HostEngram()
        hashes_pre = engram.hashes(pre_tok, 0)
        hashes_decs = [
            engram.hashes(dec_tok[:, i : i + 1], a.S + i) for i in range(a.steps)
        ]  # sequential: the hash state rolls

    sh = _Shards()
    layers = parse_layers(a.layers)
    # model.py keeps cross-layer state (compressed KV, index keys, top-k, candidates) in one module-level object that a
    # forward pass fills layer by layer. Prefill and decode are interleaved per layer here, so each phase gets its own
    # copy of that state, swapped in around every call.
    sa = R.load_model_module().shared_attn
    fields = ("compress_kv", "index_k", "topk_idxs", "candidates")
    phase = {"pre": {k: None for k in fields}}
    phase.update(
        {("dec", i): {k: None for k in fields} for i in range(a.steps)}
    )  # one cross-layer state per decode step

    def enter(name):
        for k in fields:
            v = phase[name][k]
            if (
                v is None and name != "pre"
            ):  # a decode step that has not seen this field yet starts from the post-prefill state
                v = phase["pre"][k]
            setattr(sa, k, v)

    def leave(name):
        phase[name] = {k: getattr(sa, k) for k in fields}

    t_all = time.time()
    for L in layers:
        t0 = time.time()
        blk = R.build_layer(L, max_batch_size=a.B, max_seq_len=256)
        if engram is not None and L in engram.mods:
            h_pre = engram.apply(L, h_pre, hashes_pre)
            h_decs = [engram.apply(L, h, hs) for h, hs in zip(h_decs, hashes_decs)]
        cap = {}
        hook = blk.ffn.register_forward_hook(lambda m, i, o: cap.setdefault("x", i[0].detach()))
        gate_out = {}  # the decode step's routing (set below; the hook is replaced between prefill and decode)
        route_steps = []
        enter("pre")
        h_pre, pm_pre = blk(h_pre, 0, pm_pre, None)  # prefill (fills this layer's caches)
        leave("pre")
        hook.remove()
        # router calibration from the prefill FFN inputs
        x = cap["x"].reshape(-1, 5120).float()
        w = sh.get(f"layers.{L}.ffn.gate.weight").float()
        b = sh.get(f"layers.{L}.ffn.gate.bias").float()
        rank = torch.nn.functional.softplus(x @ w.T).sqrt() + b
        K = float(rank.topk(7, -1).values[:, 5].mean())
        ratio = blk.attn.compress_ratio
        state = {"window": blk.attn.window_kv_cache.clone().float()}
        if ratio and blk.attn.is_kv_source:
            state["comp"] = blk.attn.compress_kv_cache[:, : a.S // ratio].clone().float()
            if ratio > 1:
                state["kv_state"] = blk.attn.compressor.kv_state.clone()
                state["score_state"] = blk.attn.compressor.score_state.clone()
        dec_in, pre_in = h_decs[0].clone(), pm_decs[0].clone()
        steps_in = [(h.clone(), p.clone()) for h, p in zip(h_decs, pm_decs)]

        def _gh(m, i, o):
            route_steps.append((o[0].detach().clone(), o[1].detach().clone()))
            gate_out.setdefault("w", o[0].detach().clone())
            gate_out.setdefault("idx", o[1].detach().clone())

        ghook = blk.ffn.gate.register_forward_hook(_gh)
        steps_out = []
        for i in range(
            a.steps
        ):  # decode tokens at positions S, S+1, ... (the hook keeps the routing of the FIRST step)
            enter(("dec", i))
            h_decs[i], pm_decs[i] = blk(h_decs[i], a.S + i, pm_decs[i], None)
            leave(("dec", i))
            steps_out.append((h_decs[i].clone(), pm_decs[i].clone()))
        ghook.remove()
        h_dec, pm_dec = h_decs[0], pm_decs[0]
        torch.save(
            {
                "dec_in": dec_in,
                "pre_in": pre_in,
                "dec_out": steps_out[0][0],
                "pre_out": steps_out[0][1],
                "dec_in_steps": steps_in,
                "dec_out_steps": steps_out,
                "state": state,
                "gate_cutoff": K,
                "routing_idx": gate_out["idx"].reshape(a.B, -1),
                "routing_wt": gate_out["w"].reshape(a.B, -1),
                "routing_steps": [(w_.reshape(a.B, -1), i_.reshape(a.B, -1)) for w_, i_ in route_steps],
                "ratio": ratio,
                "is_kv_source": bool(ratio and blk.attn.is_kv_source),
            },
            os.path.join(a.out, f"layer_{L}.pt"),
        )
        print(
            f"layer {L:2d} ratio {ratio} K={K:.3f}  {time.time() - t0:.0f}s  (total {time.time() - t_all:.0f}s)",
            flush=True,
        )
        del blk, cap, x, rank
        gc.collect()

    if a.head and layers[-1] == 39:
        from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostHead

        head = HostHead()
        all_logits = [head(h, p) for h, p in zip(h_decs, pm_decs)]
        logits = all_logits[0]
        torch.save(
            {
                "logits": logits,
                "argmax": logits.argmax(-1),
                "logits_steps": torch.stack(all_logits),
                "argmax_steps": torch.stack([l.argmax(-1) for l in all_logits]),
            },
            os.path.join(a.out, "final.pt"),
        )
        print("final argmax tokens (step 0):", logits.argmax(-1).tolist())


if __name__ == "__main__":
    main()
