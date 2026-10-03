# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU oracle of the PREFILL compressed-sparse attention selection of the checkpoint (``Attention.forward`` at start_pos 0) for layers of a prefill dump
(``ref_prefill_dump.py``): per layer the attention output, the top-512 compressed entries of every query and the dense attention probabilities of the
compressed entries (head-summed, sink and window in the normaliser) so that the attention-mass coverage of ANY selected set can be evaluated.

Layers are run in ascending order inside one process (index / kv sources publish their state through the checkpoint's ``shared_attn`` runtime; a layer
that reads a source needs the source in the list too). Output ``{out}/sel_L{layer}.pt``:
    out [S,5120] bf16, ids [S,K] int32 (entry ids, -1 = unreachable; window part removed), P [S,L] fp16 head-summed probability of every compressed entry
    (dense softmax over window + ALL visible compressed entries + sink), pwin [S] head-summed window mass, nvis [S] visible entries per query.
Run: python -m ...ref_prefill_sel --dir /mnt/tt-data/ssinghal/dsv4-prefill-s2048b1 --layers 2,3,8,14,20 [--qblock 256]
"""

import argparse
import json
import os
import time

import torch
from safetensors import safe_open

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R


def build_attn(layer_id, S):
    mod = R.load_model_module()
    torch.set_default_dtype(torch.bfloat16)
    args = R.model_args(1, S)
    at = mod.Attention(layer_id, args)
    index = json.load(open(os.path.join(R.CKPT_DIR, "model.safetensors.index.json")))["weight_map"]
    prefix = f"layers.{layer_id}.attn."
    handles = {}

    def get(key):
        path = R._shard_for(index, key)
        if path not in handles:
            handles[path] = safe_open(path, "pt")
        return handles[path].get_tensor(key)

    with torch.no_grad():
        for name, p in at.named_parameters():
            key = prefix + name
            t = get(key)
            if p.dtype == torch.float4_e2m1fn_x2:
                t = t.view(torch.float4_e2m1fn_x2)
            elif t.dtype == torch.float8_e4m3fn and p.dtype != torch.float8_e4m3fn:
                s = get(key.replace(".weight", ".scale"))
                bo = t.size(0) // s.size(0)
                t = ref_kernels.dequant_fp8_weight(t, s, bo).to(p.dtype)
            elif t.dtype != p.dtype:
                t = t.to(p.dtype)
            assert t.shape == p.shape, (key, t.shape, p.shape)
            p.data = t
    at.eval()
    return at


class Cap:
    """hook around ``sparse_attn`` of one prefill call: dense probabilities of the compressed entries (head summed)."""

    def __init__(self, mod, ratio, qblock):
        self.mod, self.ratio, self.qb = mod, ratio, qblock
        self.P = self.pwin = None
        self._orig = mod.sparse_attn

        def hook(q, kv, attn_sink, topk_idxs, softmax_scale):
            S = q.shape[1]
            if S > 1 and ratio:
                with torch.no_grad():
                    L = kv.shape[1] - S
                    P = torch.zeros(S, L, dtype=torch.float16)
                    pw = torch.zeros(S)
                    for q0 in range(0, S, self.qb):
                        q1 = min(S, q0 + self.qb)
                        t = torch.arange(q0, q1).view(-1, 1)
                        s = (
                            torch.einsum("mhd,nd->hmn", q[0, q0:q1].float(), kv[0].float()) * softmax_scale
                        )  # [H, qb, S+L]
                        kp = torch.arange(S).view(1, -1)
                        ok_raw = (kp <= t) & (kp > t - 128)
                        jc = torch.arange(L).view(1, -1)
                        ok_c = jc < (t + 1) // ratio
                        ok = torch.cat([ok_raw, ok_c], dim=1)
                        s = s.masked_fill(~ok.unsqueeze(0), float("-inf"))
                        s = torch.cat([s, attn_sink.float().view(-1, 1, 1).expand(-1, q1 - q0, 1)], dim=-1)
                        p = torch.softmax(s, dim=-1)[..., :-1]
                        P[q0:q1] = p[..., S:].sum(0).half()
                        pw[q0:q1] = p[..., :S].sum(0).sum(-1)
                    self.P, self.pwin = P, pw
            return self._orig(q, kv, attn_sink, topk_idxs, softmax_scale)

        mod.sparse_attn = hook

    def release(self):
        self.mod.sparse_attn = self._orig


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--qblock", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or a.dir
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "32")))
    ref_kernels.FAKE_QUANT = False
    mod = R.load_model_module()
    sa = mod.shared_attn
    for L in [int(x) for x in a.layers.split(",")]:
        t0 = time.time()
        d = torch.load(os.path.join(a.dir, f"layer_{L}.pt"), map_location="cpu")
        x = d["prefill"]["attn_in"][:1].to(torch.bfloat16)
        S = x.shape[1]
        at = build_attn(L, S)
        ratio = at.compress_ratio
        cap = Cap(mod, ratio, a.qblock)
        # the selection of this layer = shared_attn.topk_idxs after the call (window part is concatenated inside ``forward``)
        out = at(x, 0)
        cap.release()
        rec = {"out": out[0].to(torch.bfloat16), "layer": L, "ratio": ratio}
        if ratio:
            raw = sa.topk_idxs[
                0
            ].clone()  # [S, K]: entry id + S (the raw-kv length is the offset in ``Attention.forward``), -1 = unreachable
            rec["ids"] = torch.where(raw >= 0, raw - S, torch.full_like(raw, -1))
            rec["P"], rec["pwin"] = cap.P, cap.pwin
            t = torch.arange(S)
            rec["nvis"] = (t + 1) // ratio
        torch.save(rec, os.path.join(out_dir, f"sel_L{L}.pt"))
        print(f"layer {L} ratio {ratio}: {time.time() - t0:.0f}s", flush=True)
        del at, cap, rec, d


if __name__ == "__main__":
    main()
