# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only (HY4_MOCK=1: no board). The host controls of tests/test_chain.py::test_control as ONE pass over a list of
variants, row by row: the reference free-running from the golden's embedding with a chosen source of cache rows and a
chosen format of the experts, against the golden and against the device's stages of test_accuracy (its dump).

    HY4_ROPE_PAIRING=interleaved HY4_GOLDEN_DIR=generated/hy4_preview/golden_interleaved HY4_LATENT_SCALE=1.0430 \
    HY4_CHAIN_LAYERS=78 HY4_CHAIN_PROMPT=31 HY4_CTL_ROWS=0 TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>

A variant is "<rows>.<experts>":
  rows     hf        the golden's own cache rows
           stored    the rows the chips held (the packed storage read back: truncating FP8 cast)
           storedx   the same, their latent part x HY4_LATENT_SCALE at read (what AttnLayer(latent_scale=) folds in)
           near      the golden's rows through the host model of the pack with a cast that rounds to NEAREST
           trunc     the same host model with the truncating cast (a check of the model against "stored")
  experts  exact     the checkpoint's BF16 values
           bfp4      what the decode streams hold: routed experts one BFP4 term, the shared expert and layer 0's
                     dense FFN two terms
           2tA-B     bfp4, the routed experts of layers A..B with a SECOND BFP4 term (2tall: every layer)
           shx       bfp4 routed experts, the shared expert and the dense FFN exact
           p:<K>:<format>:A-B   as f:, but <format> only for each layer's K most used experts (HY4_CTL_POPULAR)
           f:<format>:A-B   bfp4, the routed experts of layers A..B in <format> (expert_value below): b4s (one term,
                     block exponents by least squared error), b4+b2, b4+b4, gu2 / dn2 (the second term on gate|up or on
                     down only), b4+b4top50 (the second term on the half of the intermediate channels that weigh most) ...
tests/test_control_rows.py::test_expert_formats: one layer at a time, the golden's inputs and picks: every format's MoE
output against the exact weights' (what a format buys before any free-running control is run), and the host model of
the BFP rounding against ttnn's own conversion.
Only the experts a row picks are read from the checkpoint (a slice per expert), so a layer costs seconds.
HY4_CTL_ROWS: rows of the dump (batch * 8 + user), default all; HY4_CTL_KEEP: rows whose stages are saved in full.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.ihc_site.op import golden_post, golden_pre
from models.experimental.hy4_preview.fused.layer_mlp.tests.test_mlp_layer import DenseWeights
from models.experimental.hy4_preview.fused.model.tests.test_chain import (
    DUMP,
    LAYERS,
    LATENT_SCALE,
    PROMPT,
    gain,
    head_reference,
    host_logits,
    site,
)
from models.experimental.hy4_preview.fused.moe import layout as ML
from models.experimental.hy4_preview.fused.sparse_mla.tests.test_attend import _checkpoint_reference
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import rms_norm
from models.experimental.hy4_preview.reference.ihc import IHCConfig
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc

E, I, HIDDEN, LAT = 256, ML.INTERMEDIATE, 6144, 512
DEFAULT = (
    "hf.exact,stored.exact,storedx.exact,hf.bfp4,storedx.bfp4,"
    "storedx.2t66-77,storedx.2t54-77,storedx.2t42-77,storedx.2t1-41,storedx.2tall,storedx.shx,near.bfp4,near.exact,stored.bfp4"
)
VARIANTS = os.environ.get("HY4_CTL_VARIANTS", DEFAULT).split(",")
TAG = os.environ.get("HY4_CTL_TAG", "")


def e4m3(v, mode):
    """v as an e4m3 value, the magnitude cut toward zero ("trunc", the packer) or rounded to nearest."""
    a = v.abs().double()
    e = torch.floor(torch.log2(a.clamp(min=2.0**-6)))
    step = torch.where(a < 2.0**-6, torch.full_like(a, 2.0**-9), 2.0 ** (e - 3))
    q = (torch.floor(a / step) if mode == "trunc" else torch.round(a / step)) * step
    return (torch.sign(v).double() * q.clamp(max=448.0)).float()


def store(rows, mode="trunc"):
    """[n, 576] (latent 512 | RoPE 64) as the packed storage holds it (the host model of
    fused/layer_attn/tests/host_fp8_rows.py): the BF16 input of the cast, 4 power-of-two scales per row, e4m3, BF16
    RoPE part."""
    p = rows[:, :LAT].to(torch.bfloat16).float().reshape(-1, 4, 128)
    amax = p.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = e4m3(p / scale, mode) * scale
    return torch.cat([q.reshape(-1, LAT), rows[:, LAT:].to(torch.bfloat16).float()], dim=1)


def routed_format(experts, layer):
    """The format (expert_value) of the routed experts of ``layer`` under the variant's experts part."""
    if experts == "exact":
        return "x"
    if experts.startswith("2t"):
        if experts == "2tall":
            return "b4+b4"
        a, b = (int(v) for v in experts[2:].split("-"))
        return "b4+b4" if a <= layer <= b else "b4"
    if experts.startswith("f:"):
        _, fmt, span = experts.split(":")
        a, b = (int(v) for v in span.split("-"))
        return fmt if a <= layer <= b else "b4"
    if experts.startswith("p:"):  # p:<K>:<format>:A-B: <format> only for the layer's K most used experts
        _, k, fmt, span = experts.split(":")
        a, b = (int(v) for v in span.split("-"))
        return f"pop{k}/{fmt}" if a <= layer <= b else "b4"
    return "b4"


def plain_format(experts):
    """The shared expert and the dense FFN: exact, or two BFP4 terms."""
    return "x" if experts in ("exact", "shx") else "b4+b4"


BFP_BITS = {"b2": 2, "b4": 4, "b8": 8}


def bfp_term(w, bits, mode="max"):
    """w [a, b] as ONE block-float term the way the device stores a matrix (tiles of w.T: a shared exponent per 16
    consecutive values of a row of w.T, sign + (bits - 1) mantissa bits, round to nearest even, the mantissa clamped
    at its maximum): the host model of tt_metal/impl/data_format/blockfloat_common.cpp convert_u32_to_bfp. mode "max":
    the block's exponent is its largest value's (ttnn's converter); "sse": of the exponents max - 1, max, max + 1 the
    one with the least squared error of the block (the same format, another choice at conversion time)."""
    # (fp32 is exact here: the values are BF16, the steps powers of two, and a value far below its block's step rounds
    # to zero; test_expert_formats compares with ttnn's conversion value by value)
    t = w.t().contiguous().float()
    blocks = t.reshape(-1, 16)
    mag = blocks.abs()
    amax = mag.amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp(min=1e-30)))
    top = float((1 << (bits - 1)) - 1)

    def quant(exp):
        step = torch.pow(2.0, exp - (bits - 2))
        return torch.round(mag / step).clamp_(max=top).mul_(step)

    if mode == "max":
        q = quant(e)
    else:
        best = None
        for d in (-1.0, 0.0, 1.0):
            cand = quant(e + d)
            err = (cand - mag).pow_(2).sum(-1, keepdim=True)
            if best is None:
                q, best = cand, err
            else:
                better = err < best
                q, best = torch.where(better, cand, q), torch.where(better, err, best)
    q = torch.where(amax > 0, q, torch.zeros_like(q)) * torch.sign(blocks)
    return q.reshape(t.shape).t().contiguous()


def bfp_sum(w, fmt):
    """w as a sum of block-float terms, each the rounding of what the terms before left: "b4", "b4s" (s: exponents by
    least squared error), "b4+b4", "b4+b2", "b4s+b2s", "b8" ..."""
    q = torch.zeros_like(w)
    for part in fmt.split("+"):
        mode = "sse" if part.endswith("s") else "max"
        q = q + bfp_term(w - q, BFP_BITS[part.rstrip("s")], mode)
    return q


def expert_value(gu, dn, fmt):
    """(gate_up [4096, 6144], down [6144, 2048]) exact -> the values a format computes with.
    x                exact
    b4, b4+b4        what the streams hold today / with a second term (ttnn's conversion, by its host model)
    gu2, dn2         the second term on gate|up only, on down only (gub2, dnb2: that second term as BFP2)
    <sum>            bfp_sum on both matrices (b4s, b4+b2, b4s+b2s, b8, ...)
    gu=<sum>,dn=<sum>   per matrix
    <sum>topNN       the terms after the first only on the NN % of the intermediate channels with the largest
                     |up row| x |down column| (a channel: its gate row, its up row, its down column)
    <sum>*g          as <sum>, then gate, up and down each multiplied by the scalar that makes its rounding's
                     projection on the exact matrix 1 (a gain folded into the weights)"""
    # (the host model, value for value ttnn's conversion - test_expert_formats - and several times faster on a busy host)
    one = lambda w: bfp_term(w, 4)
    two = lambda w: bfp_sum(w, "b4+b4")
    if fmt == "x":
        return gu, dn
    if fmt == "b4":
        return one(gu), one(dn)
    if fmt == "b4+b4":
        return two(gu), two(dn)
    if fmt == "gu2":
        return two(gu), one(dn)
    if fmt == "dn2":
        return one(gu), two(dn)
    if fmt in ("gub2", "dnb2"):  # the second term as BFP2, on gate|up only / on down only
        return bfp_sum(gu, "b4+b2" if fmt == "gub2" else "b4"), bfp_sum(dn, "b4+b2" if fmt == "dnb2" else "b4")
    if fmt.endswith("*g"):
        qg, qd = expert_value(gu, dn, fmt[:-2])
        fit = lambda w, q: q * float((w.double() * q.double()).sum() / (q.double() * q.double()).sum())
        return torch.cat((fit(gu[:I], qg[:I]), fit(gu[I:], qg[I:])), dim=0), fit(dn, qd)
    if "top" in fmt:
        base, share = fmt.split("top")
        first = base.split("+")[0]
        g1, d1 = bfp_sum(gu, first), bfp_sum(dn, first)
        g2, d2 = bfp_sum(gu, base), bfp_sum(dn, base)
        weight = gu[I:].norm(dim=1) * dn.norm(dim=0)  # [2048]
        chosen = torch.zeros(I, dtype=torch.bool)
        chosen[weight.topk(int(I * int(share) / 100)).indices] = True
        rows = torch.cat((chosen, chosen))
        return torch.where(rows[:, None], g2, g1), torch.where(chosen[None, :], d2, d1)
    if fmt.startswith("gu="):
        f_gu, f_dn = (part.split("=")[1] for part in fmt.split(","))
        return bfp_sum(gu, f_gu), bfp_sum(dn, f_dn)
    return bfp_sum(gu, fmt), bfp_sum(dn, fmt)


class Experts:
    """A layer's routed and shared experts, read from the checkpoint one expert at a time, in three formats."""

    def __init__(self, layer):
        import json
        import safetensors

        prefix = f"model.layers.{layer}.mlp."
        root = Path(os.environ["HY4_HF_MODEL"])
        weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.names = (prefix + "experts.gate_up_proj", prefix + "experts.down_proj")
        self.files = [safetensors.safe_open(str(root / weight_map[n]), framework="pt") for n in self.names]
        self.shared_names = [prefix + f"shared_experts.{n}_proj.weight" for n in ("gate", "up", "down")]
        self.shared_files = [str(root / weight_map[n]) for n in self.shared_names]
        self.gamma, self.router_w, self.router_b = hy4_tensors(
            [
                f"model.layers.{layer}.post_attention_layernorm.weight",
                prefix + "gate.weight",
                prefix + "gate.e_score_correction_bias",
            ]
        )
        self.cache = {}
        self.layer, self._popular = layer, None

    def _exact(self, e):
        if (e, "x") not in self.cache:
            if e < E:
                gu, dn = (f.get_slice(n)[e : e + 1][0].float() for f, n in zip(self.files, self.names))
            else:
                import safetensors

                g, u, d = [
                    safetensors.safe_open(p, framework="pt").get_tensor(n).float()
                    for p, n in zip(self.shared_files, self.shared_names)
                ]
                gu, dn = torch.cat((g, u), dim=0), d
            self.cache[(e, "x")] = (gu, dn)
        return self.cache[(e, "x")]

    def get(self, e, fmt):
        """(gate_up [4096, 6144], down [6144, 2048]) of expert e (256: the shared) as format fmt (expert_value), fp32."""
        if fmt.startswith("pop"):  # the format after the slash for the layer's K most used experts, one BFP4 term else
            k, inner = fmt[3:].split("/")
            fmt = inner if e < E and e in self.popular(int(k)) else "b4"
        if (e, fmt) not in self.cache:
            self.cache[(e, fmt)] = expert_value(*self._exact(e), fmt)
        return self.cache[(e, fmt)]

    def popular(self, k):
        """The layer's k most used experts (HY4_CTL_POPULAR: a file {layer: expert ids, most used first}, made by
        mk11_scripts/mk11_popular.py from the router's picks at every position of texts other than the tested one)."""
        if self._popular is None:
            self._popular = [int(v) for v in torch.load(os.environ["HY4_CTL_POPULAR"], weights_only=False)[self.layer]]
        return set(self._popular[:k])

    def moe(self, x, ids, w, routed, plain):
        out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float64)
        xf = x.float()
        for e in sorted(set(ids.flatten().tolist())) + [E]:
            gu, dn = self.get(e, routed if e < E else plain)
            if e < E:
                rows = (ids == e).any(dim=-1).nonzero().flatten()
                we = (w * (ids == e))[rows].sum(dim=-1, keepdim=True).float()
                limit = 10.0
            else:
                rows, we, limit = torch.arange(x.shape[0]), 1.0, float("inf")
            pre = xf[rows] @ gu.T
            act = ML.swiglu(pre[:, :I], pre[:, I:], limit) * we
            out[rows] += (act @ dn.T).double()
        return out.float()


def rowwise(a, b, f=pcc):
    return [round(f(a[r], b[r]), 5) for r in range(a.shape[0])]


def test_control_rows():
    if not (os.environ.get("HY4_CTL_VARIANTS") or os.environ.get("HY4_CTL_ROWS") or os.environ.get("HY4_CTL_TAG")):
        pytest.skip(
            "a host measurement, not a regression test: give HY4_CTL_ROWS / HY4_CTL_VARIANTS / HY4_CTL_TAG (see the module's text)"
        )
    if not os.path.isfile(DUMP):
        pytest.skip(f"{DUMP}: run test_accuracy on the board first")
    torch.set_num_threads(int(os.environ.get("HY4_CTL_THREADS", "8")))
    dump = torch.load(DUMP, weights_only=False)
    g = hf_harness.find_decode_golden(LAYERS, PROMPT)
    cfg = IHCConfig()
    steps = dump["steps"]
    every = [(b, u) for b in range(len(steps)) for u in range(8)]
    chosen = (
        [int(v) for v in os.environ["HY4_CTL_ROWS"].split(",")]
        if os.environ.get("HY4_CTL_ROWS")
        else list(range(len(every)))
    )
    keep_rows = [int(v) for v in os.environ.get("HY4_CTL_KEEP", str(chosen[0])).split(",")]
    rows = [every[i] for i in chosen]
    n = len(rows)
    J = [steps[b]["J"][u] for b, u in rows]
    pos = [steps[b]["positions"][u] for b, u in rows]
    gold = lambda layer, stage: g["layers"][layer][stage][J].float()
    dev = lambda layer, stage: torch.stack([steps[b]["out"][(layer, stage)][u] for b, u in rows])
    emb = gold(0, "streams_in")
    scale = LATENT_SCALE or 1.0430
    streams = {name: emb.clone() for name in VARIANTS}
    keep = {
        "J": J,
        "rows": chosen,
        "positions": pos,
        "variants": VARIANTS,
        "metrics": {},
        "stages": {},
        "logits": {},
        "keep_rows": keep_rows,
    }
    kept = [chosen.index(r) for r in keep_rows if r in chosen]
    t0 = time.perf_counter()
    for layer in range(LAYERS):
        ref = _checkpoint_reference(layer)
        (gamma_attn,) = hy4_tensors([f"model.layers.{layer}.input_layernorm.weight"])
        s_attn, s_mlp = site(layer, "hc_attn_layer"), site(layer, "hc_mlp_layer")
        w = DenseWeights(0) if layer == 0 else Experts(layer)
        hf_rows = [g["cache"][layer]["latent"][: p + 1].float() for p in pos]
        cache_rows = {"hf": hf_rows}
        kinds = {name.split(".")[0] for name in VARIANTS}
        if kinds & {"stored", "storedx", "trunc"}:
            cache_rows["stored"] = [steps[b]["stored"][(layer, u)] for b, u in rows]
        if "storedx" in kinds:
            cache_rows["storedx"] = []
            for lat in cache_rows["stored"]:
                lat = lat.clone()
                lat[:, :LAT] *= scale
                cache_rows["storedx"].append(lat)
        if "near" in kinds:
            cache_rows["near"] = [store(lat, "near") for lat in hf_rows]
        if "trunc" in kinds:
            cache_rows["trunc"] = [store(lat, "trunc") for lat in hf_rows]
            if layer in (0, 1, 40, 77):
                logger.info(
                    f"HY4_MK11 ctlrows P{PROMPT} layer {layer}: the host model of the truncating pack against the chips' rows: max abs "
                    f"{max(float((a - b).abs().max()) for a, b in zip(cache_rows['trunc'], cache_rows['stored'])):.3e}, bit-equal rows "
                    f"{sum(int((a == b).all(dim=-1).sum()) for a, b in zip(cache_rows['trunc'], cache_rows['stored']))} of {sum(a.shape[0] for a in hf_rows)}"
                )
        stage = {}
        with torch.no_grad():
            for name in VARIANTS:  # the attention half and the router
                rows_kind, experts = name.split(".")
                y, _, post, _ = golden_pre(cfg, *s_attn, streams[name])
                x = rms_norm(y, gamma_attn.float(), cfg.norm_eps)
                outs = []
                for i in range(n):
                    token = ref.new_token(x[i : i + 1], pos[i])
                    outs.append(ref.absorbed_stages(x[i : i + 1], token, cache_rows[rows_kind][i])["out"].float())
                f_attn = torch.cat(outs)
                mid = golden_post(cfg, streams[name], f_attn, post)
                y2, _, post2, _ = golden_pre(cfg, *s_mlp, mid)
                x2 = rms_norm(y2, w.gamma.float(), cfg.norm_eps)
                ids = weights = None
                if layer:
                    _, ids, weights = ML.route_golden(x2, w.router_w, w.router_b)
                stage[name] = {
                    "attn_in": x,
                    "attn_out": f_attn,
                    "mid": mid,
                    "post2": post2,
                    "mlp_in": x2,
                    "ids": ids,
                    "weights": weights,
                }
            if layer:  # every (expert, format) any variant needs, read and rounded once, 16 at a time
                need = set()
                for name in VARIANTS:
                    experts = name.split(".")[1]
                    need |= {(e, routed_format(experts, layer)) for e in set(stage[name]["ids"].flatten().tolist())}
                    need.add((E, plain_format(experts)))
                with ThreadPoolExecutor(16) as pool:
                    list(pool.map(w._exact, sorted({e for e, _ in need})))
                    list(pool.map(lambda k: w.get(*k), sorted(need)))
            for name in VARIANTS:
                experts = name.split(".")[1]
                st_ = stage[name]
                if layer == 0:
                    f_mlp, _ = w.ffn(st_["mlp_in"], rounded=plain_format(experts) != "x")
                else:
                    f_mlp = w.moe(
                        st_["mlp_in"], st_["ids"], st_["weights"], routed_format(experts, layer), plain_format(experts)
                    )
                st_["mlp_out"] = f_mlp
                streams[name] = st_["streams_out"] = golden_post(cfg, st_["mid"], f_mlp, st_.pop("post2"))
        del w
        hf_ids = g["layers"][layer]["router_indices"][J] if layer else None
        d_ids = torch.stack([steps[b]["out"][(layer, "picks")]["ids"][u].long() for b, u in rows]) if layer else None
        shared = lambda a, b: [len(set(a[r].tolist()) & set(b[r].tolist())) for r in range(n)]
        m = {}
        for name, st_ in stage.items():
            m[name] = {
                "streams": rowwise(gold(layer, "streams_out"), st_["streams_out"]),
                "streams_gain": rowwise(gold(layer, "streams_out"), st_["streams_out"], gain),
                "attn_out": rowwise(gold(layer, "attn_out"), st_["attn_out"]),
                "mlp_out": rowwise(gold(layer, "mlp_out"), st_["mlp_out"]),
                "mlp_in": rowwise(gold(layer, "mlp_in"), st_["mlp_in"]),
                "dev_streams": rowwise(st_["streams_out"], dev(layer, "streams_out")),
                "dev_attn_in": rowwise(st_["attn_in"], dev(layer, "attn_in")),
                "dev_attn_out": rowwise(st_["attn_out"], dev(layer, "attn_out")),
                "dev_attn_out_gain": rowwise(st_["attn_out"], dev(layer, "attn_out"), gain),
                "dev_mlp_in": rowwise(st_["mlp_in"], dev(layer, "mlp_in")),
                "dev_mlp_out": rowwise(st_["mlp_out"], dev(layer, "mlp_out")),
                "dev_mlp_out_gain": rowwise(st_["mlp_out"], dev(layer, "mlp_out"), gain),
            }
            if layer:
                m[name]["picks_hf"] = shared(hf_ids, st_["ids"])
                m[name]["picks_dev"] = shared(d_ids, st_["ids"])
        m["device"] = {
            "streams": rowwise(gold(layer, "streams_out"), dev(layer, "streams_out")),
            "streams_gain": rowwise(gold(layer, "streams_out"), dev(layer, "streams_out"), gain),
            "attn_out": rowwise(gold(layer, "attn_out"), dev(layer, "attn_out")),
            "mlp_out": rowwise(gold(layer, "mlp_out"), dev(layer, "mlp_out")),
            "mlp_in": rowwise(gold(layer, "mlp_in"), dev(layer, "mlp_in")),
        }
        if layer:
            m["device"]["picks_hf"] = shared(hf_ids, d_ids)
        keep["metrics"][layer] = m
        keep["stages"][layer] = {
            name: {k: (v[kept].clone() if isinstance(v, torch.Tensor) else v) for k, v in st_.items() if k != "mid"}
            for name, st_ in stage.items()
        }
        first = 0
        logger.info(
            f"HY4_MK11 ctlrows P{PROMPT} layer {layer} ({time.perf_counter() - t0:.0f} s) row {chosen[first]} streams/mlp_out pcc vs the golden"
            f"{' [picks shared of 8]' if layer else ''}: "
            + "; ".join(
                f"{name} {v['streams'][first]:.4f}/{v['mlp_out'][first]:.3f}"
                + (f" [{v['picks_hf'][first]}]" if layer else "")
                for name, v in m.items()
            )
            + (
                f" | mean over {n} rows, streams: "
                + ", ".join(f"{name} {sum(v['streams']) / n:.4f}" for name, v in m.items())
                if n > 1
                else ""
            )
        )
    head, lm = head_reference()
    gl = g["logits"][J].float()
    want = g["cached_logits_argmax"][J]
    margin = [float(gl[r].topk(2).values[0] - gl[r].topk(2).values[1]) for r in range(n)]
    summary = {}
    for name, s in list(streams.items()) + [("device", dev(LAYERS - 1, "streams_out"))]:
        _, logits = host_logits(head, lm, s)
        got = logits.argmax(-1)
        rows_pcc = [pcc(gl[r], logits[r]) for r in range(n)]
        top5 = [int(want[r]) in logits[r].topk(5).indices.tolist() for r in range(n)]
        # how far the golden's token is below the variant's top token, in the variant's own logits
        deficit = [float(logits[r].max() - logits[r, want[r]]) for r in range(n)]
        summary[name] = {
            "logits_pcc": pcc(gl, logits),
            "rows_pcc": rows_pcc,
            "top1": [bool(got[r] == want[r]) for r in range(n)],
            "top5": top5,
            "got": got.tolist(),
            "deficit": deficit,
        }
        keep["logits"][name] = logits[kept].clone()
        logger.info(
            f"HY4_MK11 ctlrows P{PROMPT} head on {name}: logits vs the golden pcc={pcc(gl, logits):.6f} (per row min {min(rows_pcc):.6f}); "
            f"top-1 the golden's on {int((got == want).sum())} of {n} rows, in the top 5 on {sum(top5)}; row {chosen[0]}: logits pcc "
            f"{rows_pcc[0]:.4f}, top-1 {'kept' if got[0] == want[0] else 'LOST'} (golden {int(want[0])}, got {int(got[0])}, the golden's margin "
            f"{margin[0]:.3f}, the golden's token {deficit[0]:.3f} below the top here); differing (row, golden, got, the golden's margin) "
            f"{[(chosen[r], int(want[r]), int(got[r]), round(margin[r], 3)) for r in range(n) if got[r] != want[r]]}"
        )
    keep["summary"], keep["want"], keep["margin"] = summary, want.tolist(), margin
    keep["gold_logits"] = gl[kept].clone()
    path = DUMP[:-3] + f"_ctlrows{TAG}.pt"
    torch.save(keep, path)
    logger.info(
        f"HY4_MK11 ctlrows P{PROMPT}: metrics of {n} rows x {len(VARIANTS)} variants, the stages of rows {keep_rows} and the logits saved to {path}"
    )


FORMATS = os.environ.get(
    "HY4_FMT_FORMATS",
    "b4,b4s,b4*g,b4s*g,gu=b4s,dn=b4,gu=b4,dn=b4s,gu2,dn2,b4+b2,b4s+b2s,gu=b4+b2,dn=b4,gu=b4,dn=b4+b2,b4+b4top50,b4+b4top25,b4+b2top50,b4+b4,b8",
)


def split_formats(text):
    """A comma list of formats, where a per-matrix format "gu=..,dn=.." holds a comma itself."""
    out, parts = [], text.split(",")
    while parts:
        part = parts.pop(0)
        out.append(part + "," + parts.pop(0) if part.startswith("gu=") else part)
    return out


def test_expert_formats():
    """Host only. Per layer of HY4_FMT_LAYERS, on the golden's own MLP inputs and picks of HY4_FMT_ROWS decode steps:
    the routed experts' summed output under every format of HY4_FMT_FORMATS against the exact weights' (pcc, gain, the
    squared error relative to one BFP4 term's), and each matrix's own rounding error and gain. First: the host model of
    the rounding against ttnn's conversion on one expert."""
    if not os.environ.get("HY4_FMT_LAYERS"):
        pytest.skip(
            "a host measurement, not a regression test: name the layers with HY4_FMT_LAYERS (e.g. 30,55,72 at HY4_CHAIN_LAYERS=78)"
        )
    torch.set_num_threads(int(os.environ.get("HY4_CTL_THREADS", "8")))
    g = hf_harness.find_decode_golden(LAYERS, PROMPT)
    if g is None:
        pytest.skip(f"no {LAYERS}-layer decode golden of a {PROMPT}-token prompt under {hf_harness.GOLDEN_ROOT}")
    layers = [int(v) for v in os.environ["HY4_FMT_LAYERS"].split(",")]
    absent = [layer for layer in layers if not 1 <= layer < len(g["layers"])]
    if absent:
        pytest.skip(f"HY4_FMT_LAYERS names layers {absent}; the golden has MoE layers 1 .. {len(g['layers']) - 1}")
    steps = g["layers"][1]["mlp_in"].shape[0]
    rows = [int(v) for v in os.environ.get("HY4_FMT_ROWS", ",".join(str(v) for v in range(min(steps, 12)))).split(",")]
    formats = split_formats(FORMATS)
    table = {}
    for n, layer in enumerate(layers):
        w = Experts(layer)
        x = g["layers"][layer]["mlp_in"][rows].float()
        hf_ids = g["layers"][layer]["router_indices"][rows]
        _, ids, weights = ML.route_golden(x, w.router_w, w.router_b)
        union = sorted(set(ids.flatten().tolist()))
        if n == 0:
            gu, dn = w._exact(union[0])
            for name, mine, theirs in (
                ("gate|up b4", bfp_term(gu, 4), ML.bfp4_round(gu.t().contiguous()).t()),
                ("down b4", bfp_term(dn, 4), ML.bfp4_round(dn.t().contiguous()).t()),
                ("gate|up b4+b4", bfp_sum(gu, "b4+b4"), ML.two_terms_value(gu[None])[0]),
            ):
                logger.info(
                    f"HY4_MK11 formats: the host model of the rounding against ttnn's conversion, layer {layer} expert {union[0]}, {name}: "
                    f"values that differ {int((mine != theirs).sum())} of {mine.numel()}, max abs {float((mine - theirs).abs().max()):.3e}"
                )
        for e in union:
            w._exact(e)

        def routed(fmt):
            with ThreadPoolExecutor(8) as pool:
                list(pool.map(lambda e: w.get(e, fmt), union))
            out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float64)
            for e in union:
                gu, dn = w.get(e, fmt)
                sel = (ids == e).any(dim=-1).nonzero().flatten()
                we = (weights * (ids == e))[sel].sum(dim=-1, keepdim=True).float()
                pre = x[sel] @ gu.T
                out[sel] += ((ML.swiglu(pre[:, :I], pre[:, I:], 10.0) * we) @ dn.T).double()
            for e in union:
                if fmt != "x":
                    del w.cache[(e, fmt)]
            return out

        exact = routed("x")
        base = None
        stats = {}
        for fmt in formats:
            # the matrices' own error, over the union: |q - w|^2 / |w|^2 and the projection of q on w
            num = {"gu": 0.0, "dn": 0.0}
            den = {"gu": 0.0, "dn": 0.0}
            dot = {"gu": 0.0, "dn": 0.0}
            for e in union[:8]:
                (gu, dn), (qg, qd) = w._exact(e), w.get(e, fmt)
                for key, a, q in (("gu", gu, qg), ("dn", dn, qd)):
                    num[key] += float((q - a).double().pow(2).sum())
                    den[key] += float(a.double().pow(2).sum())
                    dot[key] += float((q.double() * a.double()).sum())
            out = routed(fmt)
            err = float((out - exact).pow(2).sum() / exact.pow(2).sum())
            base = err if base is None else base
            rows_pcc = [pcc(exact[r], out[r]) for r in range(len(rows))]
            stats[fmt] = {
                "pcc": pcc(exact, out),
                "pcc_min": min(rows_pcc),
                "gain": gain(exact, out),
                "err": err,
                "rel": err / base,
                "w_err": {k: num[k] / den[k] for k in num},
                "w_gain": {k: dot[k] / den[k] for k in num},
            }
            logger.info(
                f"HY4_MK11 formats P{PROMPT} layer {layer}, {len(rows)} rows, {len(union)} experts (the router on the golden's x: the golden's "
                f"8 on {sum(set(ids[r].tolist()) == set(hf_ids[r].tolist()) for r in range(len(rows)))} rows), routed output vs exact weights "
                f"[{fmt}]: pcc {stats[fmt]['pcc']:.5f} (row min {min(rows_pcc):.5f}) gain {stats[fmt]['gain']:.4f}, squared error "
                f"{err:.3e} = {err / base:.3f} of one BFP4 term's; the matrices' own: gate|up error {num['gu'] / den['gu']:.3e} gain "
                f"{dot['gu'] / den['gu']:.4f}, down error {num['dn'] / den['dn']:.3e} gain {dot['dn'] / den['dn']:.4f}"
            )
        table[layer] = stats
        del w
    logger.info(
        f"HY4_MK11 formats P{PROMPT} TABLE (squared error of the routed output relative to one BFP4 term's, layers {layers}): "
        + "; ".join(f"{fmt}: " + " ".join(f"{table[layer][fmt]['rel']:.3f}" for layer in layers) for fmt in formats)
    )
    torch.save(table, DUMP[:-3] + "_formats.pt")
