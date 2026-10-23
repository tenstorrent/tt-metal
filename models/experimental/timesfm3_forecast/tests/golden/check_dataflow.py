# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Check the CPU dataflow (``dataflow.py``) and the host pipeline (``common/host.py``) against the pinned
reference. Runs in the reference venv, offline; exits non-zero if any check fails:

    env -u PYTHONPATH -u VIRTUAL_ENV HF_HUB_OFFLINE=1 /home/andy/bringup/venvs/timesfm3/bin/python \
        models/experimental/timesfm3_forecast/tests/golden/check_dataflow.py

Writes ``dataflow_report.json`` next to this script. Metric: ``rel`` = max|a - b| / max|b| over the compared
elements.

- fp64 module checks (reference ``.double()`` vs the dataflow in fp64, real weights, random states with
  left padding): the decomposition is exact, so these sit at fp64 rounding (<= 1e-12).
- the QK-norm eps of the padded head, at a large eps (0.1) where the factor is visible: 80/96 is exact,
  96/80 is not.
- the 20-layer network in fp64 from the reference's own token features (<= 1e-10).
- the host pipeline (fp64): decode inputs, token features and running statistics against the fp64 reference
  (the running mean relative to each row's scale), and the post-processing of the fp32 reference's own head output
  (CPM refine, stitching, trend, sort, symmetric averaging, make_positive) against its quantiles; <= 1e-5.
- end to end in fp32 (host + dataflow in fp32): the head output against the fp64 reference, and the final
  quantiles against the fp32 reference's post-processing, <= 1e-5. The fp32 reference's own gap to fp64
  (up to ~5e-5 on series with a large level / std ratio) is printed next to each check.
- exact topology rewrites for later rounds (fp64): dropping the trailing tokens after the last read-out,
  cutting the last layer to the read-out queries, and a finite mask fill instead of -inf.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
sys.path.insert(0, str(HERE))

import dataflow as df  # noqa: E402
import host  # noqa: E402
import make_goldens as mg  # noqa: E402

REPORT = HERE / "dataflow_report.json"
results: list[dict] = []


def rel(a: torch.Tensor, b: torch.Tensor, where: torch.Tensor | None = None) -> float:
    a, b = a.to(torch.float64), b.to(torch.float64)
    if where is not None:
        a, b = a[where], b[where]
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def sigma_rel(a: torch.Tensor, b: torch.Tensor, sigma: torch.Tensor) -> float:
    """max over rows of max|a - b| / max(sigma): the error in units of the row's data scale (rows, n)."""
    a, b, sigma = a.to(torch.float64), b.to(torch.float64), sigma.to(torch.float64)
    scale = sigma.max(-1).values
    keep = scale > 0
    return float(((a - b).abs().max(-1).values[keep] / scale[keep]).max())


def check(name: str, value: float, bound: float, note: str = "", expect_fail: bool = False):
    ok = (value <= bound) != expect_fail
    results.append({"check": name, "rel": value, "bound": bound, "pass": ok, "note": note, "expect_above": expect_fail})
    print(
        f"[{'ok' if ok else 'FAIL'}] {name}: rel {value:.3e} ({'>' if expect_fail else '<='} {bound:.0e}) {note}",
        flush=True,
    )


def module_checks(model64, w64):
    """fp64: each sublayer of the device decomposition against the reference modules."""
    rng = np.random.default_rng(7)
    cases = {
        "uni": (6, 1, 66, [[0], [0], [3], [10], [33], [65]]),
        "mv8": (2, 8, 66, [[0] * 8, [5, 5, 0, 12, 5, 66, 0, 2]]),
        "mv32": (1, 32, 34, [[int(k) for k in rng.integers(0, 8, 32)]]),
    }
    with torch.inference_mode():
        tok = torch.from_numpy(rng.normal(0, 1, (4, 66, 192)))
        check("fp64 tokenizer", rel(df.tokenizer(tok, w64), model64.pre_transformer_resblock(tok)), 1e-12)
        hx = torch.from_numpy(rng.normal(0, 3, (4, 66, 1280)))
        check("fp64 head", rel(hx @ w64.head_w.T + w64.head_b, model64.output_head(hx)), 1e-12)
        for li in (0, 19):
            layer, lw = model64.transformer_stack.layers[li], w64.layers[li]
            for name, (b, v, n, front) in cases.items():
                x, pm = mg.random_module_inputs(rng, b, v, n, front)
                x = x.double()
                eff = torch.cumprod(pm.int(), dim=2).bool()
                ref_out, _, _ = layer(x, eff)
                s_out, _, _ = layer.seq_attn(
                    layer.pre_seq_attn_ln(x).reshape(b * v, n, 1280), patch_mask=eff.reshape(b * v, n)
                )
                ref_h1 = layer.post_seq_attn_ln(s_out.view(b, v, n, 1280)) + x
                v_in = layer.pre_var_attn_ln(ref_h1).permute(0, 2, 1, 3).reshape(b * n, v, 1280)
                v_out, _, _ = layer.var_attn(v_in, patch_mask=eff.permute(0, 2, 1).reshape(b * n, v))
                ref_h2 = layer.post_var_attn_ln(v_out.view(b, n, v, 1280).permute(0, 2, 1, 3)) + ref_h1

                nfm = eff.reshape(b * v, n).sum(-1)
                masks = df.Masks.build(nfm, v, n)
                cos, sin = df.rope_tables(n, torch.float64)
                xr = x.reshape(b * v, n, 1280)
                h1 = df.seq_attention(xr, lw.seq, masks, cos, sin)
                h2 = df.var_attention(ref_h1.reshape(b * v, n, 1280), lw.var, masks, masks.token_valid)
                out = df.ffn(ref_h2.reshape(b * v, n, 1280), lw)
                full, _ = df.layer(xr, lw, masks, cos, sin)
                path = "V=1 shortcut" if v == 1 else f"general V={v}"
                check(
                    f"fp64 layer{li} {name} seq attention (RoPE, padded QK-norm, folded scale, masks)",
                    rel(h1, ref_h1.reshape(b * v, n, 1280)),
                    1e-12,
                )
                check(f"fp64 layer{li} {name} var attention ({path})", rel(h2, ref_h2.reshape(b * v, n, 1280)), 1e-12)
                check(f"fp64 layer{li} {name} ffn", rel(out, ref_out.reshape(b * v, n, 1280)), 1e-12)
                check(f"fp64 layer{li} {name} full layer", rel(full, ref_out.reshape(b * v, n, 1280)), 1e-12)


def eps_proof(model64, sd):
    """The padded QK-norm eps: with eps 0.1 on query_ln/key_ln the factor matters; 80/96 is exact."""
    rng = np.random.default_rng(8)
    layer = copy.deepcopy(model64.transformer_stack.layers[0])
    layer.seq_attn.query_ln.eps = layer.seq_attn.key_ln.eps = 0.1
    x, pm = mg.random_module_inputs(rng, 4, 1, 66, [[0], [2], [0], [9]])
    x = x.double()
    eff = torch.cumprod(pm.int(), dim=2).bool()
    with torch.inference_mode():
        s_out, _, _ = layer.seq_attn(layer.pre_seq_attn_ln(x).reshape(4, 66, 1280), patch_mask=eff.reshape(4, 66))
        ref = layer.post_seq_attn_ln(s_out.view(4, 1, 66, 1280)) + x
        w64 = df.load_weights(sd, torch.float64, num_layers=1)
        masks = df.Masks.build(eff.reshape(4, 66).sum(-1), 1, 66)
        cos, sin = df.rope_tables(66, torch.float64)
        df.QK_EPS80 = 0.1
        try:
            for factor, label, expect_fail in (
                (80 / 96, "80/96", False),
                (96 / 80, "96/80 (contract text)", True),
                (1.0, "unscaled", True),
            ):
                df.QK_EPS_FACTOR = factor
                got = df.seq_attention(x.reshape(4, 66, 1280), w64.layers[0].seq, masks, cos, sin)
                check(
                    f"fp64 padded QK-norm eps x {label} at eps 0.1",
                    rel(got, ref.reshape(4, 66, 1280)),
                    1e-12,
                    expect_fail=expect_fail,
                )
        finally:
            df.QK_EPS80, df.QK_EPS_FACTOR = None, df.QK_EPS_PAD


def network_fp64(model64, w64, cases):
    """20 layers in fp64 from the reference's own token features: transformer output and head."""
    for name, (target, mask, horizon) in cases.items():
        with torch.inference_mode():
            _, aux, _, raw = mg.decode_capture(model64, target.double(), mask, horizon, layers=())
        tok = aux["__call__:resblock_input"]
        b, v, n, _ = tok.shape
        pm = torch.cumprod(tok[..., 96:].eq(1).all(-1).int(), dim=2)
        nfm = pm.reshape(b * v, n).sum(-1)
        cap = {}
        head = df.forward(tok.reshape(b * v, n, -1), nfm, v, w64, capture=cap)
        valid = df.Masks.build(nfm, v, n).token_valid[..., None].expand(-1, -1, 1280)
        ref_out = aux["__call__:transformer_output"].reshape(b * v, n, -1)
        check(f"fp64 network {name}: transformer output (all tokens)", rel(cap["stack_output"], ref_out), 1e-10)
        check(
            f"fp64 network {name}: head output (valid tokens)",
            rel(head, raw.reshape(b * v, n, -1), valid[..., :1].expand(-1, -1, 576)),
            1e-10,
        )


def host_checks(fc, model64, cases):
    """Host pipeline (fp64) against the reference forecaster: the decode inputs, token features and running
    statistics against the fp64 reference (the fp32 reference's own rounding of (x - mu) / sigma reaches
    ~4e-5 on series with a large level / std ratio, reported as its gap to fp64), and the post-processing
    of the fp32 reference's head output against its final quantiles."""
    for name, (contexts, horizon, kw) in cases.items():
        queries = len(contexts) * (2 if kw.get("use_symmetric_averaging") else 1)
        q_ref, _, _, calls = mg.predict_recording(mg.with_batch(fc, queries), contexts, horizon, **kw)
        target, mask, h = calls[0]
        hi = host.prepare(contexts, horizon, **kw)
        check(
            f"host {name}: decode inputs (target, mask) == predict_batch's",
            float(
                (torch.from_numpy(hi.batch.target) != target).any() | (torch.from_numpy(hi.batch.mask) != mask).any()
            ),
            0.0,
        )
        _, aux, _, raw = mg.decode_capture(fc.model, target, mask, h, layers=())
        _, aux64, _, _ = mg.decode_capture(model64, target.double(), mask, h, layers=())
        tok32, tok64 = aux["__call__:resblock_input"], aux64["__call__:resblock_input"]
        b, v, n, _ = tok32.shape
        flat = lambda t: t.reshape(b * v, n, -1)  # noqa: E731
        noise = rel(flat(tok32), flat(tok64))
        check(
            f"host {name}: token features vs fp64 reference",
            rel(hi.tokens, flat(tok64)),
            1e-5,
            f"(vs fp32 reference: {rel(hi.tokens, flat(tok32)):.2e}; fp32 vs fp64 reference: {noise:.2e})",
        )
        mu64, mu32 = (a["revin_stats"][0].reshape(b * v, n) for a in (aux64, aux))
        sig64, sig32 = (a["revin_stats"][1].reshape(b * v, n) for a in (aux64, aux))
        # The running mean relative to the row's data scale (its largest running std): a detrended mean sits
        # near 0, so max|mu| is an ill-conditioned scale. The fp64 reference still detrends with fp32 sums
        # and an fp32 time axis, which leaves ~1e-5 of the scale on short or interpolated rows.
        check(
            f"host {name}: running mean (cumsum) vs fp64 reference (Chan), relative to the row scale",
            sigma_rel(hi.running_mu, mu64, sig64),
            1e-5,
            f"(vs fp32 reference: {sigma_rel(hi.running_mu, mu32, sig64):.2e})",
        )
        check(
            f"host {name}: running std (cumsum) vs fp64 reference (Chan)",
            rel(hi.running_sigma, sig64),
            1e-5,
            f"(vs fp32 reference: {rel(hi.running_sigma, sig32):.2e})",
        )
        start, count = hi.readout_start, hi.num_forecast_patches
        head_ref = raw[:, :, start : start + count].reshape(b * v, count, -1)
        q = host.postprocess(head_ref, hi)
        check(
            f"host {name}: postprocess of the reference head == final quantiles",
            rel(torch.from_numpy(q), torch.from_numpy(q_ref)),
            1e-5,
            kw.get("note", ""),
        )


def running_stats_fp64():
    """cumsum running stats vs the reference Chan loop, both in fp64, on left-padded data with a big level."""
    from timesfm3.torch import util

    rng = np.random.default_rng(9)
    vals = torch.from_numpy(rng.normal(1e4, 3.0, (5, 2, 66, 32)))
    masks = torch.zeros(5, 2, 66, 32, dtype=torch.bool)
    masks[1, :, :10] = True
    masks[2, 0, :3, :17] = True
    masks[3, :, 64:] = True
    masks[4, 1, ::7, ::3] = True
    rn, rmu, rs = util.get_running_stats(vals, masks)
    n, mu, s = host.running_stats(vals.reshape(10, 66, 32), masks.reshape(10, 66, 32))
    check("fp64 running count", rel(n, rn.reshape(10, 66)), 0.0)
    check("fp64 running mean (level 1e4)", rel(mu, rmu.reshape(10, 66)), 1e-12)
    check("fp64 running std (level 1e4, std 3)", rel(s, rs.reshape(10, 66)), 1e-9)


def end_to_end_fp32(model32, model64, w32, cases):
    """Host (fp64 -> fp32 tokens) + dataflow fp32: the head against the fp64 reference (the fp32 reference's
    own gap to it is printed), the quantiles against the fp32 reference's post-processing."""
    for name, (contexts, horizon, kw) in cases.items():
        hi = host.prepare(contexts, horizon, **kw)
        target, mask = torch.from_numpy(hi.batch.target), torch.from_numpy(hi.batch.mask)
        with torch.inference_mode():
            _, aux32, _, raw32 = mg.decode_capture(model32, target, mask, hi.horizon, layers=())
            _, _, _, raw64 = mg.decode_capture(model64, target.double(), mask, hi.horizon, layers=())
            cap = {}
            head = df.forward(hi.tokens, hi.num_front_masked, hi.variates, w32, hi.readout_tokens, capture=cap)
        start, count = hi.readout_start, hi.num_forecast_patches
        r32 = raw32[:, :, start : start + count].reshape(hi.rows, count, -1)
        r64 = raw64[:, :, start : start + count].reshape(hi.rows, count, -1)
        floor = rel(r32, r64)
        check(
            f"fp32 e2e {name}: head output at read-outs vs fp64 reference",
            rel(head, r64),
            1e-5,
            f"(vs fp32 reference: {rel(head, r32):.2e}; fp32 vs fp64 reference: {floor:.2e})",
        )
        q = host.postprocess(head, hi)
        q32 = host.postprocess(r32, hi)
        check(f"fp32 e2e {name}: final quantiles", rel(torch.from_numpy(q), torch.from_numpy(q32)), 1e-5)


def topology_checks(w64, cases):
    """Exact rewrites (fp64): trailing-token drop, last-layer cut, finite mask fill."""
    for name, (contexts, horizon, kw) in cases.items():
        hi = host.prepare(contexts, horizon, **kw)
        ro = list(hi.readout_tokens)
        base = df.forward(hi.tokens, hi.num_front_masked, hi.variates, w64, ro)
        keep = ro[-1] + 1
        cut = df.forward(hi.tokens[:, :keep], hi.num_front_masked, hi.variates, w64, ro)
        check(
            f"fp64 {name}: drop the {hi.num_patches - keep} tokens after the last read-out ({hi.num_patches} -> {keep})",
            rel(cut, base),
            1e-12,
        )
        last = df.forward(hi.tokens, hi.num_front_masked, hi.variates, w64, ro, last_layer_cut=True)
        check(f"fp64 {name}: last layer cut to the {len(ro)} read-out queries", rel(last, base), 1e-12)
        df.MASK_FILL = -1e9
        try:
            cap = {}
            fin = df.forward(hi.tokens, hi.num_front_masked, hi.variates, w64, ro, capture=cap)
        finally:
            df.MASK_FILL = float("-inf")
        finite = all(torch.isfinite(l["out"]).all() for l in cap["layers"])
        check(
            f"fp64 {name}: finite mask fill (-1e9) leaves the read-outs unchanged (all tokens finite: {finite})",
            rel(fin, base) if finite else float("inf"),
            1e-12,
        )


def main():
    torch.set_num_threads(mg.THREADS)
    fc = mg.load_forecaster()
    model32 = fc.model
    model64 = copy.deepcopy(model32).double()
    # The refine runs after the head and hard-codes fp32 carries (its einsum fails in fp64); the network
    # outputs and the head hook do not depend on it.
    model64.use_iterative_cpm_revin = False
    from safetensors.torch import load_file

    sd = load_file(str(mg.weights_file()))
    w32 = df.load_weights(sd, torch.float32)
    w64 = df.load_weights(sd, torch.float64)

    rng = np.random.default_rng(0)
    t = np.arange(2048)
    single = (10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / 24) + rng.standard_normal(2048)).astype(np.float32)
    full16, mv4 = mg.full16_contexts(), mg.mv4_contexts()
    gift = mg.gift_contexts()[:16]
    long2 = list(mg.synthetic(np.random.default_rng(11), 2, 15360))
    e2e = {
        "single 2048->64": ([single], 64, {}),
        "16 x 2048->64 (left-padded, NaN)": (full16, 64, {}),
        "16 x 2048->128 (CPM refine, stitching)": (full16, 128, {}),
        "4 items x 8 variates": (mv4, 64, {}),
        "gift 16 (symmetric, make_positive) ->48": (gift, 48, {"use_symmetric_averaging": True, "make_positive": True}),
        "2 x 15360->1024 (long context)": (long2, 1024, {}),
    }

    module_checks(model64, w64)
    eps_proof(model64, sd)
    running_stats_fp64()
    host_checks(fc, model64, e2e)
    fmt = {k: host.prepare(*v[:2], **v[2]).batch for k, v in list(e2e.items())[:4]}
    network_fp64(
        model64, w64, {k: (torch.from_numpy(b.target), torch.from_numpy(b.mask), b.horizon) for k, b in fmt.items()}
    )
    end_to_end_fp32(model32, model64, w32, e2e)
    topology_checks(w64, {k: e2e[k] for k in list(e2e)[1:4]})

    failed = [r for r in results if not r["pass"]]
    REPORT.write_text(
        json.dumps(
            {
                "reference": f"{mg.REPO}@{mg.COMMIT}",
                "checkpoint": f"{mg.HF_REPO}@{mg.HF_REVISION}",
                "torch": torch.__version__,
                "metric": "max|a-b| / max|b|",
                "checks": results,
                "failed": len(failed),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(results) - len(failed)}/{len(results)} checks pass; report {REPORT}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
