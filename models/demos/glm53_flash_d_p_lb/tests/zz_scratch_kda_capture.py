# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch (accuracy investigation 2026-10-27): KDA state drift and context-state swaps on the whole model.

Mode capture (GLM_KC_OUT=dir, default): the s56320 prompt in 5120 chunks from 0 on the device. Saves
  - x_<chunk>.pt: every KDA layer's attention input (attn_norm, [S, 4096] bf16) per chunk (GLM_KC_X=0 skips),
  - y_10.pt: every KDA layer's attention output of the last chunk,
  - state_<end>.pt: every KDA layer's carried state (recurrent fp32, conv) after each chunk,
  - full_51200.pt: every layer's state before the last chunk (KDA + MLA latent + indexer keys),
  - last.pt: the last chunk's top-5 ids / values and final hidden,
then runs the variants (GLM_KC_VARIANTS) of the last chunk.
Mode variants (GLM_KC_BASE=full_51200.pt): no prefill; each variant loads the base context and runs the last chunk.
Variants: base (the saved / just-built device context), gold (every state from the golden at 51200), gold_kda,
gold_dsa, kda:<file> (KDA states from a {layer: {kda_recurrent, kda_conv}} file, the rest base),
kdarec:<file> (only kda_recurrent from the file).
Per variant: top-1 vs text (all rows), top-1 agreement with the CPU golden's argmax (all rows), final hidden PCC.

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    GLM_KC_OUT=<dir> scripts/run_safe_pytest.sh --run-all --no-precompile \\
    models/demos/glm53_flash_d_p_lb/tests/zz_scratch_kda_capture.py -s
"""

import os
import time
from pathlib import Path

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
KDA = ("kda_recurrent", "kda_conv")
DSA = ("kv_latent", "index_key")


def _pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


@mesh_parametrize
def test_kda_capture(mesh_device):
    from models.demos.common.bringup.reference.golden import Golden
    from models.demos.glm53_flash_d_p.tt.common import split_to_host
    from models.demos.glm53_flash_d_p.tt.kda_attention import KDA_DECAY

    if os.environ.get("GLM_KC_MASK"):  # probe: chunk preparation's bf16 outputs (bit 1 kd, 2 q_decay, 5 final_decay)
        from models.demos.deepseek_v3_d_p.tt.kda import recurrence

        recurrence.KDA_PREP_OUTPUT_BF16_MASK = int(os.environ["GLM_KC_MASK"], 0)
        print(f"[kc] prep bf16 mask {recurrence.KDA_PREP_OUTPUT_BF16_MASK:#x}", flush=True)
    g = Golden.for_rung(S, "s56320")
    tokens = g.tokens().to(torch.long)
    seq, chunk = 56320, 5120
    last = seq - chunk
    layers = S.layers()
    t0 = time.time()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=True)
    print(f"[kc] loaded in {time.time() - t0:.0f}s, KDA decay {KDA_DECAY}", flush=True)
    kda_layers = [i for i in layers if model.cfg.is_kda(i)]
    base_file = os.environ.get("GLM_KC_BASE")
    out = Path(os.environ.get("GLM_KC_OUT", "/tmp/kc")) if not base_file else None
    cap_x = os.environ.get("GLM_KC_X", "1") == "1"
    cap = {"x": None, "y": None, "host": None}

    for i in kda_layers:
        blk = model.blocks[i]
        fn = blk.overrides["attention"]

        def wrapped(ctx, x, fn=fn, i=i):
            if cap["host"] is not None:  # the exact fp32 reference KDA on the host from this variant's loaded state
                import ttnn
                from models.demos.glm53_flash_d_p.tt.common import split_from_host

                ref, hst = cap["host"]
                xh = split_to_host(x).reshape(-1, x.shape[-1]).float()
                y = ref._kda(i, type("C", (), {"start": ctx.start, "state": hst})(), xh)
                ttnn.deallocate(x)
                return split_from_host(mesh_device, y.reshape(1, 1, -1, y.shape[-1]).to(torch.bfloat16))
            if cap["x"] is not None:
                cap["x"][i] = split_to_host(x).reshape(-1, x.shape[-1]).to(torch.bfloat16)
            y = fn(ctx, x)
            if cap["y"] is not None:
                cap["y"][i] = split_to_host(y).reshape(-1, y.shape[-1]).to(torch.bfloat16)
            return y

        blk.overrides["attention"] = wrapped

    def run_chunk(start):
        h = model.embed(tokens[start : start + chunk])
        for i in layers:
            h2 = model.layer(i, h, start, None)
            model.free(h)
            h = h2
        hid = model.final_norm(h)
        model.free(h)
        return hid

    want = tokens[last + 1 : seq]
    gm = g.model(10)
    gtop = gm["top32_ids"][: chunk - 1, 0].long()

    def score(hid, tag):
        allr = list(range(chunk - 1))
        ids, vals = [], []
        for r0 in range(0, len(allr), 512):
            lg = model.logits(hid, allr[r0 : r0 + 512]).float()
            tk = lg.topk(5, dim=-1)
            ids.append(tk.indices)
            vals.append(tk.values)
        ids, vals = torch.cat(ids), torch.cat(vals)
        hh = model.to_host(hid).float()
        t1 = float((ids[:, 0] == want).float().mean())
        t5 = float((ids == want[:, None]).any(-1).float().mean())
        a1 = float((ids[:, 0] == gtop).float().mean())
        fh = _pcc(hh, gm["final_norm"].float())
        print(
            f"[kc] {tag:28s} top1 vs text {t1:.4f} top5 {t5:.4f}  top1 vs CPU argmax {a1:.4f}  final hidden pcc {fh:.4f}",
            flush=True,
        )
        return {"ids": ids, "vals": vals, "hid": hh.to(torch.bfloat16), "top1": t1, "top5": t5, "agree": a1}

    if base_file is None:
        out.mkdir(parents=True, exist_ok=True)
        for start in range(0, seq, chunk):
            t = time.time()
            cap["x"] = {} if cap_x else None
            cap["y"] = {} if start == last else None
            if start == last:
                full = {i: {k: v.clone() for k, v in model.blocks[i].state_torch(last).items()} for i in layers}
                torch.save(full, out / f"full_{last}.pt")
            hid = run_chunk(start)
            model.sync()
            dt = time.time() - t
            st = {i: {k: v.clone() for k, v in model.blocks[i].state_torch(start + chunk).items()} for i in kda_layers}
            torch.save(st, out / f"state_{start + chunk}.pt")
            if cap["x"] is not None:
                torch.save(cap["x"], out / f"x_{start // chunk:02d}.pt")
            if cap["y"] is not None:
                torch.save(cap["y"], out / f"y_{start // chunk:02d}.pt")
            if start == last:
                res = score(hid, f"run ({KDA_DECAY})")
                torch.save(res, out / "last.pt")
            model.free(hid)
            print(
                f"[kc] chunk [{start},{start + chunk}) {dt * 1e3:.0f} ms (+ saves {time.time() - t - dt:.1f}s)",
                flush=True,
            )
        cap["x"] = cap["y"] = None
        base = full
    else:
        base = torch.load(base_file)

    gold = {}
    variants = [v for v in os.environ.get("GLM_KC_VARIANTS", "").split(",") if v]
    files = {}
    href = None
    for v in variants:
        v, host = (v[:-5], True) if v.endswith("+host") else (v, False)
        if v in ("gold", "gold_kda", "gold_dsa") and not gold:
            gold = {i: g.state(i, at=last) for i in layers}
        st_all = {}
        for i in layers:
            st = dict(base[i])
            if v == "gold" or (v == "gold_kda" and i in kda_layers) or (v == "gold_dsa" and i not in kda_layers):
                st = {k: gold[i][k].float() for k in st}
            elif v.startswith(("kda:", "kdarec:")) and i in kda_layers:
                path = v.split(":", 1)[1]
                if path not in files:
                    files[path] = torch.load(path)
                src = files[path][i]
                names = KDA if v.startswith("kda:") else ("kda_recurrent",)
                st.update({k: src[k].float() for k in names})
            st_all[i] = st
        for i in layers:
            model.blocks[i].load_state(st_all[i], last)
        if host:
            from models.demos.common.bringup.reference.golden import hf_path
            from models.demos.glm53_flash_d_p.reference.glm_ref import GlmReference, State, load_layer
            from models.demos.glm53_flash_d_p.reference.weights import WeightLoader

            if href is None:
                href = GlmReference.__new__(GlmReference)
                href.cfg, href.dtype, href.w = model.cfg, torch.float32, {}
                wl = WeightLoader(hf_path(S))
                for i in kda_layers:
                    w = load_layer(wl, model.cfg, i, torch.float32, eager_experts=False)
                    w["conv"] = torch.cat([w.pop(f"self_attn.{n}_conv1d.weight") for n in "qkv"])[:, 0].float()
                    href.w[i] = {k: t for k, t in w.items() if k.startswith("self_attn.") or k == "conv"}
            hst = State(model.cfg, kda_layers, seq)
            for i in kda_layers:
                hst.rec[i].copy_(st_all[i]["kda_recurrent"].reshape(hst.rec[i].shape))
                hst.conv[i].copy_(st_all[i]["kda_conv"].reshape(hst.conv[i].shape))
            cap["host"] = (href, hst)
        hid = run_chunk(last)
        cap["host"] = None
        v = v + ("+host" if host else "")
        score(hid, v if len(v) < 28 else v[:12] + ".." + v[-14:])
        model.free(hid)
