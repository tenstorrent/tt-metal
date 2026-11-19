# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The fused iHC site (fused/ihc_site/op.py) on one chip against reference/ihc.py and against the ttnn form
(tt/ihc.py, tt/norm.py) on the same inputs.

On the mock Galaxy (HY4_MOCK=1) every case builds its programs, compiles the kernels and checks CB / L1 /
kernel sizes; nothing runs and nothing is compared. On the board, smallest first:

    pytest <this file> -k test_probe_lanes
    pytest <this file> -k "C256 and K1"
    pytest <this file> -k "test_pre or test_post or test_post_pre or test_layers"
    pytest <this file> -k test_latency

Log lines start with HY4_IHC_SITE. Thresholds: PCC >= 0.999 on every output (tests/test_ihc.py's), and the bound
tests/test_norm.py puts on an fp32 norm, 1e-5 relative, on the statistic 1 / RMS(y).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.ihc_site import layout
from models.experimental.hy4_preview.fused.ihc_site.op import (
    IhcSite,
    SiteRef,
    probe_device,
    probe_projection,
    golden_post,
    golden_pre,
    probe_pages,
)
from models.experimental.hy4_preview.reference.ihc import IHCConfig
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.ihc import TtIHCWrap
from models.experimental.hy4_preview.tt.norm import TtRMSNorm

MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))
PCC = 0.999
STAT_BOUND = 1e-5
C_HY4 = IHCConfig().dim
TRACE_REGION_BYTES = 8 << 20

# One device for the whole module (every case is one chip; the latency cases need the trace region).
pytestmark = pytest.mark.use_module_device({"trace_region_size": TRACE_REGION_BYTES})

_PROBE = {}


def _probe(device) -> dict:
    """What the SFPU probe finds on this device (once per run; the first program a run executes)."""
    if not _PROBE:
        _PROBE.update(probe_device(device))
        logger.info(
            f"HY4_IHC_SITE SFPU lanes on this device: {_PROBE['lanes']}; unpacker reaches DEST slots 4..7: {_PROBE['unpack_high']}"
        )
        low = not _PROBE["unpack_high"] or bool(os.environ.get("HY4_IHC_LOW_UNPACK"))
        proj = probe_projection(device, _PROBE["lanes"], low)
        # the [16, 32] x [32, 16] projection unless it is off and the full-tile one is not (19-bit operands: 5e-4)
        _PROBE["mm32"] = (proj["mm16"] > 5e-3 and proj["mm32"] <= 5e-3) or bool(os.environ.get("HY4_IHC_MM32"))
        logger.info(
            f"HY4_IHC_SITE projection probe: relative error [16, 32] x [32, 16] {proj['mm16']:.3e}, "
            f"[32, 32] x [32, 32] {proj['mm32']:.3e}; the cases run with mm32={_PROBE['mm32']}"
        )
    return _PROBE


def _lanes(device) -> str:
    return _probe(device)["lanes"]


def _site(device, cfg, **kw):
    """An IhcSite laid out as the probe says this device needs it. The cases name their projection: without
    sfpu_proj=True it is the FPU one (the ttnn form's), which is not IhcSite's default."""
    kw.setdefault("sfpu_proj", False)
    found = _probe(device)
    low = not found["unpack_high"] or bool(os.environ.get("HY4_IHC_LOW_UNPACK"))  # the variable: mock compile of it
    if low and kw.get("num_cores", 8) > 32:
        pytest.skip("low_unpack: the root core takes at most 4 exchange pages (32 cores)")
    fallbacks = {} if kw["sfpu_proj"] else {"low_unpack": low, "mm32": found["mm32"]}
    return IhcSite(device, cfg, lanes=found["lanes"], **fallbacks, **kw)


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def _random_params(cfg, scale_val, seed):
    """tests/test_ihc.py's random parameters; fn rounded to BF16, the dtype the site keeps it in (and the
    checkpoint's)."""
    g = torch.Generator().manual_seed(seed)
    fn = _bf16(torch.randn(cfg.mix_hc, cfg.n * cfg.dim, generator=g) * 0.02)
    base = torch.randn(cfg.mix_hc, generator=g)
    scale = torch.tensor([float(scale_val), 0.5 * float(scale_val)])
    return fn, base, scale


def _checkpoint_params(layer, site):
    prefix = f"model.layers.{layer}.{site}.hc_pre."
    return hy4_tensors([prefix + "hc_fn", prefix + "hc_base", prefix + "hc_scale"])


def _params(cfg, weights, seed):
    if weights == "random":
        return _random_params(cfg, 0.01 if cfg.dim == C_HY4 else 1.0, seed)
    layer, site = weights
    return _checkpoint_params(layer, site)


class Checks:
    """Every comparison of a case is logged (lines start with HY4_IHC_SITE) and the case fails once, at the end, so
    one board run shows every output even when one of them is off."""

    def __init__(self, tag):
        self.tag, self.failed = tag, []

    def pcc(self, name, ref, dev, threshold=PCC):
        ref, dev = ref.double(), dev.double()
        err = (ref - dev).abs().max().item()
        try:
            score = pcc(ref, dev)
        except AssertionError:
            score = float("nan")
        logger.info(f"HY4_IHC_SITE {self.tag} {name}: pcc={score:.7f} max_abs={err:.3e}")
        if not score >= threshold:
            self.failed.append(f"{name}: pcc={score:.7f} max_abs={err:.3e} (threshold {threshold})")
        return score, err

    def rel(self, name, ref, dev, bound):
        err = ((dev.double() - ref.double()).abs() / ref.double().abs()).max().item()
        logger.info(f"HY4_IHC_SITE {self.tag} {name}: max_rel={err:.3e} (bound {bound:.0e})")
        if not err <= bound:
            self.failed.append(f"{name}: max relative error {err:.3e} > {bound:.0e}")
        return err

    def true(self, name, ok):
        if not ok:
            logger.info(f"HY4_IHC_SITE {self.tag} {name}: FAILED")
            self.failed.append(name)

    def done(self):
        assert not self.failed, f"{self.tag}: " + "; ".join(self.failed)


def _scratch_kept(site) -> bool:
    """With the root exchange and more than 40 cores the root core's sixth exchange page passes through its scratch
    slot, so its debug page no longer holds its own partial sums: the intermediates are not reported then."""
    return not (site.root_exchange and site.K > 40)


def _debug_report(tag, site, cfg, fn, x, T):
    """The intermediates every core left (debug=True) against the host in float64: the FPU projection of the
    core's own columns, its SFPU sums of squares, the normalised mixes and the parametrised coefficients."""
    if not _scratch_kept(site):
        return
    K, C = site.K, cfg.dim
    xd = torch.zeros(layout.ROWS, cfg.n, C, dtype=torch.float64)
    xd[:T] = x.double()
    fnd = fn.double().reshape(8, cfg.n, C)
    worst = {"proj": 0.0, "ss": 0.0}
    scale = xd.reshape(layout.ROWS, -1).norm(dim=-1).max().item() * fnd.reshape(8, -1).norm(dim=-1).max().item()
    for core in range(K):
        d = site.debug_to_torch(core)
        cols = slice(core * C // K, (core + 1) * C // K)
        r_proj = torch.einsum("rjc,ojc->ro", xd[:, :, cols], fnd[:, :, cols])
        r_ss = xd[:, :, cols].pow(2).sum(dim=(1, 2))
        worst["proj"] = max(worst["proj"], (d["proj"].double() - r_proj).abs().max().item())
        ss = d["ss"].double().sum(-1)
        worst["ss"] = max(worst["ss"], ((ss - r_ss).abs() / r_ss.clamp_min(1e-30))[:T].max().item())
        if core == 0:
            flat = xd.reshape(layout.ROWS, -1)
            r_mix = (flat @ fnd.reshape(8, -1).t()) * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + cfg.norm_eps)
            mix_err = (d["mix"].double() - r_mix)[:T].abs().max().item()
            mix_rel = ((d["mix"].double() - r_mix)[:T].norm() / r_mix[:T].norm()).item()
    logger.info(
        f"HY4_IHC_SITE {tag} intermediates: projection per core max_abs={worst['proj']:.3e} "
        f"(|x| |fn| = {scale:.3e}), sum of squares per core max_rel={worst['ss']:.3e}, "
        f"normalised mixes max_abs={mix_err:.3e} rel={mix_rel:.3e}"
    )


def _stage_report(tag, site, cfg, base, scale_ab):
    """Every stage after the projection against the host applied to the device's own inputs of that stage, so a
    stage that is off shows by itself whatever the stages before it did: the exchange and normalisation (sum of the
    cores' partials), the sigmoid parametrisation, the broadcast into the coefficient page, the statistic."""
    if not _scratch_kept(site):
        return
    K, C, n = site.K, cfg.dim, cfg.n
    d = [site.debug_to_torch(core) for core in range(K)]
    sent = max(
        max((dk["part_mix"] - dk["proj"]).abs().max().item(), (dk["part_ss"] - dk["ss"]).abs().max().item()) for dk in d
    )
    mix_sum = sum(dk["part_mix"].double() for dk in d)
    ss_sum = sum(dk["part_ss"].double() for dk in d).sum(-1, keepdim=True)
    mix_exp = mix_sum * torch.rsqrt(ss_sum / (n * C) + cfg.norm_eps)
    mix_dev, g_dev = d[0]["mix"].double(), d[0]["g"].double()
    ones = torch.ones(n, dtype=torch.float64)
    scale_row = torch.cat([ones * float(scale_ab[0]), ones * float(scale_ab[1])])
    gain = torch.cat([ones, ones * float(cfg.magnitude)])
    g_exp = torch.sigmoid(mix_dev * scale_row + base.double()) * gain + cfg.eps
    pre, post = site.coef_to_torch(layout.ROWS)
    y2 = sum(dk["y2"].double() for dk in d).sum(-1)
    stat_exp = torch.rsqrt(y2 / C + cfg.norm_eps)
    stat = site.stat_to_torch(layout.ROWS).double()
    logger.info(
        f"HY4_IHC_SITE {tag} stages on the device's own data: partial as sent max_abs={sent:.3e}, "
        f"exchange + normalise max_abs={(mix_dev - mix_exp).abs().max():.3e}, "
        f"sigmoid max_abs={(g_dev - g_exp).abs().max():.3e}, "
        f"broadcast max_abs={(torch.cat([pre, post], -1).double() - g_dev).abs().max():.3e}, "
        f"statistic max_rel={((stat - stat_exp).abs() / stat_exp).max():.3e}"
    )


# ---------------------------------------------------------------------------------------------
# What the SFPU delivers in fp32: the mixes replayed on the host with one fp32 rounding per multiply-add
# ---------------------------------------------------------------------------------------------
def _round32(v: torch.Tensor, mode: str) -> torch.Tensor:
    """A float64 value to fp32: to nearest even, or truncated towards zero."""
    f = v.float()
    if mode == "nearest":
        return f
    over = f.double().abs() > v.abs()
    return torch.where(over, torch.nextafter(f, torch.zeros_like(f)), f)


def _round_bf16_truncate(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> BF16 by dropping the low 16 bits (truncation towards zero)."""
    bits = v.float().contiguous().view(torch.int32) & ~0xFFFF
    return bits.view(torch.float32)


def _mad32(a, b, c, mode):
    """a * b + c with one rounding (the product of two fp32 values is exact in float64)."""
    return _round32(a.double() * b.double() + c.double(), mode)


def _keep_bits(v: torch.Tensor, bits: int, nearest: bool) -> torch.Tensor:
    """A float64 value cut to ``bits`` significant bits, truncated towards zero or rounded to nearest."""
    m, e = torch.frexp(v)
    scaled = m * float(2**bits)
    scaled = torch.round(scaled) if nearest else torch.trunc(scaled)
    return torch.ldexp(scaled / float(2**bits), e)


def _report_mad(tag, a, b, c, dev):
    """One SFPU multiply-add a * b + c against host models of how it rounds: the exact product added and rounded
    once (a fused multiply-add), or the product first cut to 24 + g bits (truncated or rounded), then the sum rounded
    to nearest or truncated. The share of bit-equal results per model; a model that is the hardware's gives 100 %."""
    exact = a.double() * b.double()
    cd = c.double()
    models = {
        "fused, to nearest": _round32(exact + cd, "nearest"),
        "fused, truncated": _round32(exact + cd, "truncate"),
    }
    for g in (0, 1, 2, 3, 4, 8):
        models[f"product truncated to {24 + g} bits, sum to nearest"] = _round32(
            _keep_bits(exact, 24 + g, False) + cd, "nearest"
        )
        models[f"product rounded to {24 + g} bits, sum to nearest"] = _round32(
            _keep_bits(exact, 24 + g, True) + cd, "nearest"
        )
    models["product truncated to 24 bits, sum truncated"] = _round32(_keep_bits(exact, 24, False) + cd, "truncate")
    scores = {name: (dev == value).float().mean().item() for name, value in models.items()}
    best = sorted(scores.items(), key=lambda kv: -kv[1])
    logger.info(
        f"HY4_IHC_SITE {tag} SFPU multiply-add, bit-equal results per model: "
        + "; ".join(f"{name}: {100 * score:.3f}%" for name, score in best[:6])
    )
    fused = models["fused, to nearest"]
    off = dev != fused
    if off.any():
        ulp = (torch.nextafter(fused.abs(), torch.full_like(fused, float("inf"))) - fused.abs()).double()
        steps = ((dev.double() - fused.double()) / ulp)[off]
        # how far apart the exponents of product and addend are where the result differs from the fused one
        gap = (torch.frexp(exact)[1] - torch.frexp(cd)[1])[off].double()
        logger.info(
            f"HY4_IHC_SITE {tag} SFPU multiply-add where it differs from fused-to-nearest ({100 * off.float().mean():.3f}%): "
            f"difference in ulps min {steps.min():.1f} median {steps.median():.1f} max {steps.max():.1f}, "
            f"share below the fused result {100 * (steps < 0).float().mean():.1f}%, "
            f"exponent(product) - exponent(addend) min {gap.min():.0f} median {gap.median():.0f} max {gap.max():.0f}"
        )
    return scores


def _ulps(ref: torch.Tensor, dev: torch.Tensor):
    """(fraction of bit-equal values, largest difference in fp32 ulps of the reference)."""
    ulp = (torch.nextafter(ref.abs(), torch.full_like(ref, float("inf"))) - ref.abs()).double()
    diff = (dev.double() - ref.double()).abs() / ulp
    return (dev == ref).float().mean().item(), diff.max().item()


def _sfpu_model_mix(x, pre, mode):
    """y = ((x0 c0 + x1 c1) + x2 c2) + x3 c3 as the mix sweep issues it: a multiply, then three multiply-adds."""
    c = pre.float().unsqueeze(-1)  # [T, 4, 1]
    y = _round32(x[:, 0].double() * c[:, 0].double(), mode)
    for j in range(1, 4):
        y = _mad32(x[:, j], c[:, j], y, mode)
    return y


def _sfpu_model_post(x, f, post, mode):
    """x_j + post_j * F: one multiply-add per value."""
    return _mad32(f.float().unsqueeze(1), post.float().unsqueeze(-1), x.float(), mode)


def _report_sfpu(tag, dev, model):
    """Which rounding the SFPU multiply-add applies: the device against both host models."""
    found = {mode: _ulps(model(mode), dev) for mode in ("nearest", "truncate")}
    logger.info(
        f"HY4_IHC_SITE {tag} SFPU vs fp32 model: "
        + ", ".join(f"{mode}: {100 * eq:.2f}% bit-equal, max {ulp:.2f} ulp" for mode, (eq, ulp) in found.items())
    )
    return found


# ---------------------------------------------------------------------------------------------
# Relative errors against float64 and against the ttnn form
# ---------------------------------------------------------------------------------------------
def _fp64(cfg, fn, base, scale, x):
    """reference/ihc.py's hc_pre and the RMSNorm statistic in float64, in the reference's order:
    (P * inverse_rms) * scale + base, streams summed 0 first."""
    n = cfg.n
    flat = x.double().reshape(x.shape[0], -1)
    mixes = (flat @ fn.double().t()) * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + cfg.norm_eps)
    pre = torch.sigmoid(mixes[:, :n] * float(scale[0]) + base.double()[:n]) + cfg.eps
    post = cfg.magnitude * torch.sigmoid(mixes[:, n:] * float(scale[1]) + base.double()[n:]) + cfg.eps
    y = torch.zeros(x.shape[0], cfg.dim, dtype=torch.float64)
    for j in range(n):
        y = y + pre[:, j : j + 1] * x.double()[:, j]
    return {"1/rms": torch.rsqrt(y.pow(2).mean(-1) + cfg.norm_eps), "pre": pre, "post": post, "y": y, "mixes": mixes}


def _max_rel(ref, dev):
    return ((dev.double() - ref.double()).abs() / ref.double().abs()).max().item()


def _row_err(ref, dev):
    """Largest row error of a [T, C] output: |dev - ref| / |ref| per token row (tests/test_norm.py's measure)."""
    return ((dev.double() - ref.double()).norm(dim=-1) / ref.double().norm(dim=-1)).max().item()


def _precision_report(tag, ref, dev, ttf):
    """1 / RMS, the pre and post coefficients and the collapsed output y: largest relative error of the site and of
    the ttnn form against float64, and of the site against the ttnn form (y: largest row error). Returns the
    site's errors against float64."""
    out = {}
    parts = []
    for name in ("1/rms", "pre", "post", "y"):
        err = _row_err if name == "y" else _max_rel
        out[name] = err(ref[name], dev[name])
        parts.append(
            f"{name}: site {out[name]:.2e}, ttnn {err(ref[name], ttf[name]):.2e}, site vs ttnn {err(ttf[name], dev[name]):.2e}"
        )
    logger.info(f"HY4_IHC_SITE precision {tag} max relative error vs float64 | " + " | ".join(parts))
    return out


# ---------------------------------------------------------------------------------------------
# The ttnn form of the same stages on the same chip (tt/ihc.py with no hidden split, tt/norm.py)
# ---------------------------------------------------------------------------------------------
class TtForm:
    def __init__(self, device, cfg, fn, base, scale):
        self.plan = MeshPlan(device, hidden_axis=None)
        self.cfg = cfg
        self.wrap = TtIHCWrap(self.plan, cfg, fn, base, scale)
        self.norm = TtRMSNorm(self.plan, torch.ones(cfg.dim), cfg.norm_eps)

    def up(self, t):
        return ttnn.from_torch(
            t.float().contiguous(), layout=ttnn.TILE_LAYOUT, device=self.plan.mesh_device, dtype=ttnn.float32
        )

    def up_x(self, x):  # [T, n, C] -> [1, 1, T, n C]
        return self.up(x.reshape(1, 1, x.shape[0], -1))

    def up_rows(self, v):  # [T, C] -> [1, 1, T, C]
        return self.up(v.reshape(1, 1, *v.shape))

    def pre(self, d_x):
        y, post = self.wrap.hc_pre(d_x)
        return y, post, self.norm.inverse_rms(y)

    def post(self, d_f, d_x, d_post):
        return self.wrap.hc_post(d_f, d_x, d_post)

    def pre_outputs(self, x):
        """x [T, n, C] -> {"1/rms" [T], "pre" [T, n], "post" [T, n], "y" [T, C]} as the ttnn form computes them."""
        n = self.cfg.n
        d_x = self.up_x(x)
        g = ttnn.to_torch(self.wrap.parametrize(self.wrap.project(d_x))).float()[0, 0]
        y, _ = self.wrap.hc_pre(d_x)
        stat = ttnn.to_torch(self.norm.inverse_rms(y)).float()[0, 0, :, 0]
        return {"1/rms": stat, "pre": g[:, :n], "post": g[:, n:], "y": ttnn.to_torch(y).float()[0, 0]}


# ---------------------------------------------------------------------------------------------
# Probe: what an SFPU register holds, the lane rotation, DEST slots 6 / 7, DEST kept across sections
# ---------------------------------------------------------------------------------------------
def test_probe_lanes(device):
    cfg = IHCConfig(dim=128)
    site = IhcSite(device, cfg, num_cores=1, sfpu_proj=False)
    page, const = probe_pages()
    out, stat = site.probe(page, const)
    if MOCK:
        return
    chk = Checks("probe")
    face_in, face_out = page[:16, :16], out[:16, :16]
    logger.info(f"HY4_IHC_SITE probe in  rows 0..1: {face_in[0].tolist()} | {face_in[1].tolist()}")
    for r in range(0, 16, 4):
        logger.info(f"HY4_IHC_SITE probe out row {r}: {[round(v, 5) for v in face_out[r].tolist()]}")
    logger.info(f"HY4_IHC_SITE probe stat row 0: {[round(v, 5) for v in stat[0, :16].tolist()]}")
    # unit 0 is stored where it was loaded from: its lane set is the columns that came back unchanged
    same = tuple(c for c in range(16) if torch.equal(face_out[0:4, c], face_in[0:4, c]))
    logger.info(f"HY4_IHC_SITE probe lane set 0 columns: {same}")
    lanes = {tuple(range(8)): "half", tuple(range(0, 16, 2)): "evenodd"}.get(same)
    assert lanes is not None, f"unit 0 came back on columns {same}"
    logger.info(f"HY4_IHC_SITE probe SFPU lanes: {lanes}")
    a = layout.unit_get(page, 0, lanes)
    # the register loaded from unit 1 (stored to unit 2) is the other lane set of the same rows
    chk.true(
        "unit 1 is the other lane set", torch.equal(layout.unit_get(out, 2, lanes), layout.unit_get(page, 1, lanes))
    )
    # rotation: lane l of the rotated register holds lane (l + shift) % 8 of the original
    rot = layout.unit_get(out, 1, lanes)
    shift = [k for k in range(8) if torch.equal(rot, torch.roll(a, -k, dims=1))]
    logger.info(f"HY4_IHC_SITE probe rotate: lane l <- lane (l + {shift}) % 8")
    chk.true("rotation by one lane within the 8 lanes of a row", len(shift) == 1 and shift[0] in (1, 7))
    hs = layout.unit_get(out, 3, lanes)
    ref = a.double().sum(dim=1, keepdim=True).expand(4, 8)
    err = ((hs.double() - ref).abs() / ref).max().item()
    logger.info(f"HY4_IHC_SITE probe hsum8: max_rel={err:.3e} all lanes equal={bool((hs == hs[:, :1]).all())}")
    chk.true("hsum8 leaves the row sum on every lane", err < 1e-6)
    chk.true("x * 3 + x is exact", torch.equal(layout.unit_get(out, 4, lanes), 4.0 * a))
    high = torch.equal(layout.unit_get(out, 5, lanes), layout.unit_get(const, 0, lanes))
    logger.info(f"HY4_IHC_SITE probe unpacker reaches DEST slot 7: {high} (if not, the cases run with low_unpack)")
    for unit, slot in ((8, 7), (9, 5), (10, 4)):
        chk.true(f"SFPU store and load in DEST slot {slot}", torch.equal(layout.unit_get(out, unit, lanes), a))
    rs = layout.unit_get(out, 6, lanes)
    err_rs = ((rs.double() - a.double().rsqrt()).abs() * a.double().sqrt()).max().item()
    sg = layout.unit_get(out, 7, lanes)
    err_sg = (sg.double() - torch.sigmoid(a.double())).abs().max().item()
    logger.info(f"HY4_IHC_SITE probe rsqrt max_rel={err_rs:.3e} sigmoid max_abs={err_sg:.3e}")
    chk.true("rsqrt", err_rs < 1e-5)
    chk.true("sigmoid", err_sg < 1e-6)
    chk.true("DEST slot 6 kept across sections", torch.equal(layout.unit_get(stat, 0, lanes), a))
    chk.true(
        "DEST slot 7 kept across sections",
        torch.equal(layout.unit_get(stat, 1, lanes), layout.unit_get(const, 1, lanes)),
    )
    chk.true(f"layout.SFPU_LANES is {lanes!r}", lanes == layout.SFPU_LANES)
    chk.done()


# ---------------------------------------------------------------------------------------------
# pre, post, post + pre against the reference and the ttnn form
# ---------------------------------------------------------------------------------------------
SHAPES = [(256, 1), (256, 2), (C_HY4, 2), (C_HY4, 8), (C_HY4, 16)]
SHAPE_IDS = [f"C{c}-K{k}" for c, k in SHAPES]
WEIGHTS = ["random"] + [(layer, site) for layer in range(6) for site in ("hc_attn_layer", "hc_mlp_layer")]
WEIGHT_IDS = ["random"] + [f"L{layer}-{site[3:-6]}" for layer in range(6) for site in ("hc_attn_layer", "hc_mlp_layer")]


def _cases():
    """(C, K, weights): random weights on every shape, the checkpoint's on the Hy4 width with 8 cores."""
    out, ids = [], []
    for (c, k), sid in zip(SHAPES, SHAPE_IDS):
        for w, wid in zip(WEIGHTS, WEIGHT_IDS):
            if w == "random" or (c == C_HY4 and k == 8):
                out.append((c, k, w))
                ids.append(f"{sid}-{wid}")
    return out, ids


CASES, CASE_IDS = _cases()


def _inputs(cfg, T, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, cfg.n, cfg.dim, generator=g)
    f = torch.randn(T, cfg.dim, generator=g)
    return x, f


def _wname(weights):
    return weights if weights == "random" else f"L{weights[0]}-{weights[1][3:-6]}"


@pytest.mark.parametrize("T", [1, 8], ids=["T1", "T8"])
@pytest.mark.parametrize("C, K, weights", CASES, ids=CASE_IDS)
def test_pre(device, C, K, weights, T):
    cfg = IHCConfig(dim=C)
    fn, base, scale = _params(cfg, weights, seed=3)
    x, _ = _inputs(cfg, T, seed=0)
    site = _site(device, cfg, num_cores=K)
    params = site.site_params(fn, base, scale)
    d_x = site.streams_to_device(x)
    d_y = site.pre(d_x, params, debug=True)
    if MOCK:
        return
    chk = Checks(f"pre C{C} K{K} T{T} {_wname(weights)}")
    r_y, r_pre, r_post, r_stat = golden_pre(cfg, fn, base, scale, x)
    y = site.rows_to_torch(d_y, T)
    pre, post = site.coef_to_torch(T)
    stat = site.stat_to_torch(T)
    _debug_report(chk.tag, site, cfg, fn, x, T)
    _stage_report(chk.tag, site, cfg, base, scale)
    chk.pcc("y vs ref", r_y, y)
    logger.info(
        f"HY4_IHC_SITE {chk.tag} coefficients vs ref: max_abs pre={(pre - r_pre).abs().max():.3e} "
        f"post={(post - r_post).abs().max():.3e}"
    )
    chk.true("pre within 1e-3 of the reference", (pre - r_pre).abs().max().item() < 1e-3)
    chk.true("post within 2e-3 of the reference", (post - r_post).abs().max().item() < 2e-3)
    if T > 1:
        chk.pcc("post vs ref", r_post, post)
    chk.rel("1/rms(y) vs ref", r_stat, stat, 5e-3)
    # the statistic against the sum of squares of the y the site itself produced: the fp32 norm bound
    chk.rel("1/rms(y) vs own y", torch.rsqrt(y.double().pow(2).mean(-1) + cfg.norm_eps), stat, STAT_BOUND)
    # the mix itself, given the coefficients the device computed: fp32 multiply-adds
    _report_sfpu(f"{chk.tag} y", y, lambda mode: _sfpu_model_mix(x, pre, mode))
    chk.true("streams untouched", torch.equal(site.streams_to_torch(d_x, T), x))
    same = all(
        torch.equal(c_pre, pre) and torch.equal(c_post, post)
        for c_pre, c_post in map(lambda k: site.coef_to_torch(T, k), range(1, K))
    )
    chk.true("every core holds the same coefficients", same)
    # the ttnn form on the same inputs
    tt = TtForm(device, cfg, fn, base, scale)
    t_y, t_post, t_stat = tt.pre(tt.up_x(x))
    t_y = ttnn.to_torch(t_y).float()[0, 0]
    t_post = ttnn.to_torch(t_post).float()[0, 0]
    t_stat = ttnn.to_torch(t_stat).float()[0, 0, :, 0]
    chk.pcc("y vs ttnn", t_y, y)
    logger.info(
        f"HY4_IHC_SITE {chk.tag} ttnn form vs ref: y pcc={pcc(r_y, t_y):.7f} max_abs={(r_y - t_y).abs().max():.3e} "
        f"post max_abs={(r_post - t_post).abs().max():.3e} "
        f"1/rms max_rel={((t_stat - r_stat).abs() / r_stat).max():.3e}"
    )
    chk.done()


@pytest.mark.parametrize("f_dtype", [ttnn.float32, ttnn.bfloat16], ids=["F-fp32", "F-bf16"])
@pytest.mark.parametrize("T", [1, 8], ids=["T1", "T8"])
@pytest.mark.parametrize("C, K", SHAPES, ids=SHAPE_IDS)
def test_post(device, C, K, T, f_dtype):
    cfg = IHCConfig(dim=C)
    x, f = _inputs(cfg, T, seed=4)
    if f_dtype == ttnn.bfloat16:
        f = _bf16(f)
    g = torch.Generator().manual_seed(5)
    pre = torch.sigmoid(torch.randn(T, cfg.n, generator=g)) + cfg.eps
    post = cfg.magnitude * torch.sigmoid(torch.randn(T, cfg.n, generator=g)) + cfg.eps
    site = _site(device, cfg, num_cores=K, f_dtype=f_dtype)
    site.coef_to_device(pre, post)
    d_x = site.streams_to_device(x)
    site.post(d_x, site.rows_to_device(f))
    if MOCK:
        return
    chk = Checks(f"post C{C} K{K} T{T} F-{'fp32' if f_dtype == ttnn.float32 else 'bf16'}")
    out = site.streams_to_torch(d_x, T)
    ref = golden_post(cfg, x, f, post)
    chk.pcc("streams vs ref", ref, out)
    _report_sfpu(f"{chk.tag} streams", out, lambda mode: _sfpu_model_post(x, f, post, mode))
    if T == 8 and f_dtype == ttnn.float32:
        # one multiply-add per value: stream j = F * post_j + stream j
        _report_mad(
            chk.tag,
            f.float().unsqueeze(1).expand(-1, cfg.n, -1),
            post.float().unsqueeze(-1).expand(-1, -1, C),
            x.float(),
            out,
        )
    tt = TtForm(device, cfg, *_random_params(cfg, 1.0, seed=3))
    t_out = tt.post(tt.up_rows(f), tt.up_x(x), tt.up(post.reshape(1, 1, T, cfg.n)))
    t_out = ttnn.to_torch(t_out).float()[0, 0].reshape(T, cfg.n, C)
    chk.pcc("streams vs ttnn", t_out, out)
    logger.info(f"HY4_IHC_SITE {chk.tag} ttnn form vs ref: max_abs={(ref - t_out).abs().max():.3e}")
    chk.done()


@pytest.mark.parametrize("y_dtype", [ttnn.float32, ttnn.bfloat16], ids=["y-fp32", "y-bf16"])
@pytest.mark.parametrize("T", [1, 8], ids=["T1", "T8"])
@pytest.mark.parametrize("C, K, weights", CASES, ids=CASE_IDS)
def test_post_pre(device, C, K, weights, T, y_dtype):
    """post of site a with the coefficients its pre left, then pre of site b, as one program."""
    _post_pre_case(device, C, K, weights, T, y_dtype)


def _post_pre_case(device, C, K, weights, T, y_dtype, label="", **site_kw):
    cfg = IHCConfig(dim=C)
    fn_a, base_a, scale_a = _random_params(cfg, 0.01 if C == C_HY4 else 1.0, seed=11)
    fn_b, base_b, scale_b = _params(cfg, weights, seed=12)
    x, f = _inputs(cfg, T, seed=6)
    site = _site(device, cfg, num_cores=K, y_dtype=y_dtype, **site_kw)
    params_a = site.site_params(fn_a, base_a, scale_a)
    params_b = site.site_params(fn_b, base_b, scale_b)
    d_x = site.streams_to_device(x)
    site.pre(d_x, params_a)
    if not MOCK:
        _, post_a = site.coef_to_torch(T)
    d_y = site.post_pre(d_x, site.rows_to_device(f), params_b, debug=True)
    if MOCK:
        return
    chk = Checks(f"post_pre{label} C{C} K{K} T{T} {_wname(weights)} y-{'bf16' if y_dtype == ttnn.bfloat16 else 'fp32'}")
    _, _, r_post_a, _ = golden_pre(cfg, fn_a, base_a, scale_a, x)
    r_x = golden_post(cfg, x, f, r_post_a)
    r_y, r_pre, r_post, r_stat = golden_pre(cfg, fn_b, base_b, scale_b, r_x)
    out = site.streams_to_torch(d_x, T)
    y = site.rows_to_torch(d_y, T)
    pre, post = site.coef_to_torch(T)
    stat = site.stat_to_torch(T)
    _debug_report(chk.tag, site, cfg, fn_b, out, T)
    _stage_report(chk.tag, site, cfg, base_b, scale_b)
    chk.pcc("streams vs ref", r_x, out)
    _report_sfpu(f"{chk.tag} streams", out, lambda mode: _sfpu_model_post(x, f, post_a, mode))
    chk.pcc("y vs ref", r_y, y)
    logger.info(
        f"HY4_IHC_SITE {chk.tag} coefficients vs ref: max_abs pre={(pre - r_pre).abs().max():.3e} "
        f"post={(post - r_post).abs().max():.3e}"
    )
    chk.true("pre within 1e-3 of the reference", (pre - r_pre).abs().max().item() < 1e-3)
    chk.true("post within 2e-3 of the reference", (post - r_post).abs().max().item() < 2e-3)
    chk.rel("1/rms(y) vs ref", r_stat, stat, 5e-3)
    if y_dtype == ttnn.float32:
        chk.rel("1/rms(y) vs own y", torch.rsqrt(y.double().pow(2).mean(-1) + cfg.norm_eps), stat, STAT_BOUND)
        _report_sfpu(f"{chk.tag} y", y, lambda mode: _sfpu_model_mix(out, pre, mode))
    else:
        exact = _sfpu_model_mix(out, pre, "nearest")
        near, trunc = _bf16(exact), _round_bf16_truncate(exact)
        logger.info(
            f"HY4_IHC_SITE {chk.tag} BF16 y vs the fp32 mix rounded: to nearest {100 * (y == near).float().mean():.2f}% "
            f"bit-equal, truncated {100 * (y == trunc).float().mean():.2f}% bit-equal, "
            f"max_abs vs nearest={(y - near).abs().max():.3e}"
        )
    chk.done()


# ---------------------------------------------------------------------------------------------
# The one FPU step: the fn projection (19-bit operands, HiFi4, fp32 accumulation)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("width, K", [(256, 1), (C_HY4, 8)], ids=["proj256", f"proj{C_HY4}"])
def test_precision_projection(device, width, K):
    """The normalised mixes with scale 1 and base 0 against reference/ihc.py; tests/test_ihc.py's bound for the
    ttnn projection is PCC 0.998. The ttnn form's own projection of the same input is logged next to it."""
    cfg = IHCConfig(dim=width)
    T = 8
    g = torch.Generator().manual_seed(31)
    fn = _bf16(torch.randn(cfg.mix_hc, cfg.n * cfg.dim, generator=g) * 0.02)
    base, scale = torch.zeros(cfg.mix_hc), torch.ones(2)
    x = torch.randn(T, cfg.n, cfg.dim, generator=g)
    site = _site(device, cfg, num_cores=K)
    site.pre(site.streams_to_device(x), site.site_params(fn, base, scale), debug=True)
    if MOCK:
        return
    chk = Checks(f"projection C{width} K{K}")
    flat = x.reshape(T, -1).double()
    r_mixes = (flat @ fn.double().t()) * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + cfg.norm_eps)
    mixes = site.debug_to_torch(0)["mix"].double()[:T]
    _debug_report(chk.tag, site, cfg, fn, x, T)
    chk.pcc("mixes vs ref", r_mixes, mixes, 0.998)
    rel = ((mixes - r_mixes).norm() / r_mixes.norm()).item()
    ratio = (mixes.norm() / r_mixes.norm()).item()
    logger.info(f"HY4_IHC_SITE {chk.tag}: relative error {rel:.3e}, norm ratio {ratio:.6f}")
    tt = TtForm(device, cfg, fn, base, scale)
    t_mixes = ttnn.to_torch(tt.wrap.project(tt.up_x(x))).double()[0, 0]
    logger.info(
        f"HY4_IHC_SITE {chk.tag} ttnn form: pcc={pcc(r_mixes, t_mixes):.7f} "
        f"max_abs={(r_mixes - t_mixes).abs().max():.3e} "
        f"relative error {((t_mixes - r_mixes).norm() / r_mixes.norm()).item():.3e} "
        f"norm ratio {(t_mixes.norm() / r_mixes.norm()).item():.6f}"
    )
    chk.done()


# ---------------------------------------------------------------------------------------------
# Layers 0-5 of the checkpoint chained through the two fused programs
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("proj", ["fpu", "sfpu"])
@pytest.mark.parametrize("T", [8], ids=["T8"])
def test_layers(device, T, proj):
    """expand -> 6 x (attention site, MLP site) with the checkpoint's iHC parameters; every sublayer is the same
    bounded map applied on the host to the y the device produced, so what accumulates is the site's own arithmetic:
    pre(L0 attn), then post(attn) + pre(MLP) and post(MLP) + pre(next attn) alternating."""
    cfg = IHCConfig()
    K = 8
    site = _site(device, cfg, num_cores=K, sfpu_proj=proj == "sfpu")
    sites = [(layer, s) for layer in range(6) for s in ("hc_attn_layer", "hc_mlp_layer")]
    weights = [_checkpoint_params(layer, s) for layer, s in sites]
    if proj == "sfpu":
        # 12 sites of fp32 fn pages do not fit the cores' L1: they come from DRAM
        table = site.site_table(weights)
        params = [SiteRef(table, i) for i in range(len(sites))]
    else:
        params = [site.site_params(*w) for w in weights]
    g = torch.Generator().manual_seed(21)
    emb = torch.randn(T, cfg.dim, generator=g)
    perm = torch.randperm(cfg.dim, generator=g)
    sub = lambda h: 0.7 * torch.tanh(h[:, perm])  # a fixed sublayer stand-in, applied to the device's own y
    r_x = emb.unsqueeze(1).expand(-1, cfg.n, -1).contiguous()
    d_x = site.streams_to_device(r_x)
    d_y = site.pre(d_x, params[0])
    if MOCK:
        site.post_pre(d_x, site.rows_to_device(emb), params[1])
        site.post(d_x, site.rows_to_device(emb))
        return
    chk = Checks(f"layers {proj}")
    for i, (layer, s) in enumerate(sites):
        r_y, _, r_post, r_stat = golden_pre(cfg, *weights[i], r_x)
        y = site.rows_to_torch(d_y, T)
        stat = site.stat_to_torch(T)
        f = sub(y)
        r_x = golden_post(cfg, r_x, sub(r_y), r_post)
        if i + 1 < len(sites):
            d_y = site.post_pre(d_x, site.rows_to_device(f), params[i + 1])
        else:
            site.post(d_x, site.rows_to_device(f))
        out = site.streams_to_torch(d_x, T)
        tag = f"L{layer} {s[3:-6]}"
        chk.pcc(f"{tag} y vs ref", r_y, y)
        chk.pcc(f"{tag} streams vs ref", r_x, out)
        chk.rel(f"{tag} 1/rms(y) vs ref", r_stat, stat, 5e-3)
    chk.done()


# ---------------------------------------------------------------------------------------------
# Traced latency next to the ttnn form of the same stages
# ---------------------------------------------------------------------------------------------
def _latency(device, K, root_exchange, label, **site_kw):
    cfg = IHCConfig()
    T = 8
    fn, base, scale = _random_params(cfg, 0.01, seed=3)
    x, f = _inputs(cfg, T, seed=0)
    site = _site(device, cfg, num_cores=K, root_exchange=root_exchange, **site_kw)
    params = site.site_params(fn, base, scale)
    d_x = site.streams_to_device(x)
    d_f = site.rows_to_device(f * 1e-3)  # small, so the replayed post-mixes keep the streams finite
    d_y = site.empty_rows()
    programs = {
        "empty program (launch only)": lambda: site.empty(d_x, d_f, params, d_y),
        "pre": lambda: site.pre(d_x, params, d_y),
        "post": lambda: site.post(d_x, d_f),
        "post_pre": lambda: site.post_pre(d_x, d_f, params, d_y),
        "pre cut after ss": lambda: site.pre(d_x, params, d_y, stop_after=1),
        "pre cut after project": lambda: site.pre(d_x, params, d_y, stop_after=2),
        "pre cut after param": lambda: site.pre(d_x, params, d_y, stop_after=3),
        "pre cut after mix": lambda: site.pre(d_x, params, d_y, stop_after=4),
        "post_pre cut after post": lambda: site.post_pre(d_x, d_f, params, d_y, stop_after=1),
    }
    # The same programs unpacking the streams one page per unpacker run (IhcSite(block_unpack=False)): its own site
    # object (its own coefficient state) on the same tensors, checked bit for bit against the default before it is
    # timed.
    paged = _site(device, cfg, num_cores=K, block_unpack=False, root_exchange=root_exchange, **site_kw)
    paged_programs = {
        "pre (one page per unpack)": lambda: paged.pre(d_x, params, d_y),
        "post (one page per unpack)": lambda: paged.post(d_x, d_f),
        "post_pre (one page per unpack)": lambda: paged.post_pre(d_x, d_f, params, d_y),
    }
    if MOCK:
        for run in list(programs.values()) + list(paged_programs.values()):
            run()
        return
    idle = tracing.idle_sync_ms(device)
    logger.info(f"HY4_IHC_SITE latency {label}: idle synchronize {1e3 * idle:.1f} us")

    def timed(name, run):
        t = tracing.traced_ms(device, run)
        logger.info(
            f"HY4_IHC_SITE latency {label} {name}: {1e3 * t['replay_ms']:.1f} us per replay "
            f"(rounds {[round(1e3 * v, 1) for v in t['rounds_ms']]})"
        )

    for name, run in programs.items():
        timed(name, run)

    # one page per unpack against the default path, on fresh streams
    ttnn.deallocate(d_x)
    d_x = site.streams_to_device(x)
    d_xb = site.streams_to_device(x)
    y_a = site.rows_to_torch(site.pre(d_x, params, d_y))
    stat_a, coef_a = site.stat_to_torch(), site.coef_to_torch()
    y_b = paged.rows_to_torch(paged.pre(d_x, params, d_y))
    stat_b, coef_b = paged.stat_to_torch(), paged.coef_to_torch()
    same_pre = (
        torch.equal(y_a, y_b) and torch.equal(stat_a, stat_b) and all(torch.equal(a, b) for a, b in zip(coef_a, coef_b))
    )
    y_a = site.rows_to_torch(site.post_pre(d_x, d_f, params, d_y))
    x_a = site.streams_to_torch(d_x)
    y_b = paged.rows_to_torch(paged.post_pre(d_xb, d_f, params, d_y))
    x_b = paged.streams_to_torch(d_xb)
    same_fused = torch.equal(y_a, y_b) and torch.equal(x_a, x_b)
    logger.info(
        f"HY4_IHC_SITE latency {label} one page per unpack bit-equal to the default path: pre {same_pre}, "
        f"post_pre {same_fused} (max_abs y {(y_a - y_b).abs().max():.3e}, streams {(x_a - x_b).abs().max():.3e})"
    )
    ttnn.deallocate(d_xb)
    for name, run in paged_programs.items():
        timed(name + ("" if same_pre and same_fused else " [DIFFERENT RESULTS]"), run)
    assert same_pre and same_fused, "the two unpack paths must agree bit for bit"


@pytest.mark.parametrize("K", [2, 4, 8, 16], ids=lambda k: f"K{k}")
def test_latency(device, K):
    """Each program as a traced replay (20 back to back, one synchronize), the pre program cut after each of its
    stages for the breakdown, and the program with empty kernels for the launch floor; every core sends its partial
    to every core. PERF.md 3.2, ttnn form on the 8x4 mesh, 8 rows, traced: pre-mix 0.303 ms, norm 0.097 ms,
    post-mix 0.086 ms."""
    _latency(device, K, False, f"K{K} all-to-all")


def test_latency_ttnn_form(device):
    """The ttnn form of the same stages on this one chip (no hidden split, so no collective), 8 rows, traced."""
    cfg = IHCConfig()
    T = 8
    fn, base, scale = _random_params(cfg, 0.01, seed=3)
    x, f = _inputs(cfg, T, seed=0)
    if MOCK:
        return
    tt = TtForm(device, cfg, fn, base, scale)
    d_x, d_f = tt.up_x(x), tt.up_rows(f)
    d_post = tt.up(torch.ones(1, 1, T, cfg.n))
    stages = {
        "pre-mix": lambda: tt.wrap.hc_pre(d_x),
        "post-mix": lambda: tt.wrap.hc_post(d_f, d_x, d_post),
        "norm statistic": lambda: tt.norm.inverse_rms(d_f),
        "norm": lambda: tt.norm(d_f),
    }
    for name, run in stages.items():
        t = tracing.traced_ms(device, run)
        logger.info(f"HY4_IHC_SITE latency ttnn form {name}: {1e3 * t['replay_ms']:.1f} us per replay")


# ---------------------------------------------------------------------------------------------
# The root exchange (more than 16 cores): partials to the root core, coefficients back by one multicast.
# These cases run after everything above (tests/conftest.py): they are the only ones with a NoC multicast.
# ---------------------------------------------------------------------------------------------
ROOT_SHAPES = [(256, 1), (256, 2), (C_HY4, 8), (C_HY4, 24), (C_HY4, 48)]


@pytest.mark.parametrize("C, K", ROOT_SHAPES, ids=[f"C{c}-K{k}" for c, k in ROOT_SHAPES])
def test_root_exchange(device, C, K):
    """pre, then post + pre, with the root exchange, 8 rows, random weights, against the reference."""
    cfg = IHCConfig(dim=C)
    T = 8
    fn_a, base_a, scale_a = _random_params(cfg, 0.01 if C == C_HY4 else 1.0, seed=11)
    fn_b, base_b, scale_b = _random_params(cfg, 0.01 if C == C_HY4 else 1.0, seed=12)
    x, f = _inputs(cfg, T, seed=6)
    site = _site(device, cfg, num_cores=K, root_exchange=True)
    params_a = site.site_params(fn_a, base_a, scale_a)
    params_b = site.site_params(fn_b, base_b, scale_b)
    d_x = site.streams_to_device(x)
    d_y = site.pre(d_x, params_a, debug=True)
    if MOCK:
        site.post_pre(d_x, site.rows_to_device(f), params_b, debug=True)
        return
    chk = Checks(f"root exchange C{C} K{K}")
    r_y, r_pre, r_post_a, r_stat = golden_pre(cfg, fn_a, base_a, scale_a, x)
    y, stat = site.rows_to_torch(d_y, T), site.stat_to_torch(T)
    pre, post = site.coef_to_torch(T)
    _debug_report(f"{chk.tag} pre", site, cfg, fn_a, x, T)
    _stage_report(f"{chk.tag} pre", site, cfg, base_a, scale_a)
    chk.pcc("pre: y vs ref", r_y, y)
    chk.true(
        "pre: coefficients within 2e-3 of the reference",
        max((pre - r_pre).abs().max().item(), (post - r_post_a).abs().max().item()) < 2e-3,
    )
    chk.rel("pre: 1/rms(y) vs own y", torch.rsqrt(y.double().pow(2).mean(-1) + cfg.norm_eps), stat, STAT_BOUND)
    same = all(
        torch.equal(c_pre, pre) and torch.equal(c_post, post)
        for c_pre, c_post in map(lambda k: site.coef_to_torch(T, k), range(1, K))
    )
    chk.true("pre: every core holds the root's coefficients", same)
    d_y = site.post_pre(d_x, site.rows_to_device(f), params_b, debug=True)
    r_x = golden_post(cfg, x, f, r_post_a)
    r_y, r_pre, r_post, r_stat = golden_pre(cfg, fn_b, base_b, scale_b, r_x)
    out, y, stat = site.streams_to_torch(d_x, T), site.rows_to_torch(d_y, T), site.stat_to_torch(T)
    pre, post = site.coef_to_torch(T)
    _debug_report(f"{chk.tag} post_pre", site, cfg, fn_b, out, T)
    _stage_report(f"{chk.tag} post_pre", site, cfg, base_b, scale_b)
    chk.pcc("post_pre: streams vs ref", r_x, out)
    chk.pcc("post_pre: y vs ref", r_y, y)
    chk.true(
        "post_pre: coefficients within 2e-3 of the reference",
        max((pre - r_pre).abs().max().item(), (post - r_post).abs().max().item()) < 2e-3,
    )
    chk.rel("post_pre: 1/rms(y) vs ref", r_stat, stat, 5e-3)
    chk.rel("post_pre: 1/rms(y) vs own y", torch.rsqrt(y.double().pow(2).mean(-1) + cfg.norm_eps), stat, STAT_BOUND)
    chk.done()


@pytest.mark.parametrize("K", [8, 16, 24, 48], ids=lambda k: f"K{k}")
def test_latency_root_exchange(device, K):
    _latency(device, K, True, f"K{K} root exchange")


def test_mock_compiles_the_ttnn_form(device):
    """Mock Galaxy only: run the ttnn form of every shape the board cases compare against, so that its kernels are
    in the JIT cache before a board run (which then spends its time on the site, not on compiling ttnn ops)."""
    if not MOCK:
        pytest.skip("mock Galaxy only")
    for C in (256, C_HY4):
        cfg = IHCConfig(dim=C)
        for T in (1, 8):
            x, f = _inputs(cfg, T, seed=0)
            tt = TtForm(device, cfg, *_random_params(cfg, 1.0, seed=3))
            _, post, _ = tt.pre(tt.up_x(x))
            tt.post(tt.up_rows(f), tt.up_x(x), post)
            tt.wrap.project(tt.up_x(x))
            tt.norm(tt.up_rows(f))


# ---------------------------------------------------------------------------------------------
# The sites' data from DRAM: one program for every site, the site as runtime arguments.
# Runs after the cases above (tests/conftest.py).
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("proj", ["fpu", "sfpu"])
@pytest.mark.parametrize("C, K", [(256, 2), (C_HY4, 8)], ids=["C256-K2", f"C{C_HY4}-K8"])
def test_site_dram(device, C, K, proj):
    """Three sites' fn^T and constants in two DRAM tensors; pre of site 2, then post + pre of site 0 and of site 1,
    each against the reference and against the same site's L1 tensors bit for bit."""
    cfg = IHCConfig(dim=C)
    T = 8
    weights = [_random_params(cfg, 0.01 if C == C_HY4 else 1.0, seed=40 + i) for i in range(3)]
    x, f = _inputs(cfg, T, seed=8)
    site = _site(device, cfg, num_cores=K, sfpu_proj=proj == "sfpu")
    table = site.site_table(weights)
    chk = Checks(f"site dram {proj} C{C} K{K}")
    d_x, d_l = site.streams_to_device(x), site.streams_to_device(x)
    d_f = site.rows_to_device(f)
    r_x = x
    for step, index in enumerate((2, 0, 1)):
        if step == 0:
            d_y = site.pre(d_x, SiteRef(table, index))
        else:
            d_y = site.post_pre(d_x, d_f, SiteRef(table, index))
        if MOCK:
            continue
        y, stat, coef = site.rows_to_torch(d_y, T), site.stat_to_torch(T), site.coef_to_torch(T)
        out = site.streams_to_torch(d_x, T)
        # the same step with the site's L1 tensors (one site at a time: they are large), on a second copy of
        # the streams
        local = site.site_params(*weights[index])
        if step == 0:
            l_y = site.pre(d_l, local)
        else:
            site.coef_to_device(*prev_coef)
            l_y = site.post_pre(d_l, d_f, local)
        same = (
            torch.equal(site.rows_to_torch(l_y, T), y)
            and torch.equal(site.stat_to_torch(T), stat)
            and all(torch.equal(a, b) for a, b in zip(site.coef_to_torch(T), coef))
            and torch.equal(site.streams_to_torch(d_l, T), out)
        )
        for t in (local.w, local.const, l_y, d_y):
            ttnn.deallocate(t)
        chk.true(f"site {index}: bit-equal to the L1 form", same)
        if step > 0:
            r_x = golden_post(cfg, r_x, f, r_post)
        r_y, _, r_post, r_stat = golden_pre(cfg, *weights[index], r_x)
        chk.pcc(f"site {index}: y vs ref", r_y, y)
        chk.pcc(f"site {index}: streams vs ref", r_x, out)
        chk.rel(f"site {index}: 1/rms(y) vs ref", r_stat, stat, 5e-3)
        prev_coef = coef
    if not MOCK:
        chk.done()


def test_latency_site_dram(device):
    """pre and post + pre with the site's data fetched from DRAM, next to the L1 form, 8 cores."""
    cfg = IHCConfig()
    K, T = 8, 8
    weights = [_random_params(cfg, 0.01, seed=40 + i) for i in range(2)]
    x, f = _inputs(cfg, T, seed=0)
    site = _site(device, cfg, num_cores=K)
    table = site.site_table(weights)
    local = site.site_params(*weights[1])
    d_x, d_f, d_y = site.streams_to_device(x), site.rows_to_device(f * 1e-3), site.empty_rows()
    programs = {
        "pre, site data in L1": lambda: site.pre(d_x, local, d_y),
        "pre, site data from DRAM": lambda: site.pre(d_x, SiteRef(table, 1), d_y),
        "post_pre, site data in L1": lambda: site.post_pre(d_x, d_f, local, d_y),
        "post_pre, site data from DRAM": lambda: site.post_pre(d_x, d_f, SiteRef(table, 1), d_y),
    }
    if MOCK:
        for run in programs.values():
            run()
        return
    for name, run in programs.items():
        t = tracing.traced_ms(device, run)
        logger.info(f"HY4_IHC_SITE latency K{K} {name}: {1e3 * t['replay_ms']:.1f} us per replay")


# ---------------------------------------------------------------------------------------------
# The fn projection in the SFPU (IhcSite(sfpu_proj=True)) next to the FPU one: relative errors of every output
# against float64 and against the ttnn form, the fused program, and what it costs.
# ---------------------------------------------------------------------------------------------
PRECISION = [(256, 1, "random"), (C_HY4, 8, "random"), (C_HY4, 16, "random")] + [
    (C_HY4, 8, (layer, site)) for layer in range(6) for site in ("hc_attn_layer", "hc_mlp_layer")
]


@pytest.mark.parametrize("proj", ["fpu", "sfpu"])
@pytest.mark.parametrize("C, K, weights", PRECISION, ids=[f"C{c}-K{k}-{_wname(w)}" for c, k, w in PRECISION])
def test_precision(device, C, K, weights, proj):
    """pre for 8 rows: 1 / RMS, pre, post and y of the site and of the ttnn form against float64, and against each
    other. With the FPU projection both carry its 19-bit operands (5e-4 on the mixes); with the SFPU projection the
    site is fp32 throughout."""
    cfg = IHCConfig(dim=C)
    T = 8
    fn, base, scale = _params(cfg, weights, seed=3)
    x, _ = _inputs(cfg, T, seed=0)
    site = _site(device, cfg, num_cores=K, sfpu_proj=proj == "sfpu")
    d_y = site.pre(site.streams_to_device(x), site.site_params(fn, base, scale), debug=True)
    if MOCK:
        return
    chk = Checks(f"{proj} C{C} K{K} {_wname(weights)}")
    ref = _fp64(cfg, fn, base, scale, x)
    pre, post = site.coef_to_torch(T)
    dev = {"1/rms": site.stat_to_torch(T), "pre": pre, "post": post, "y": site.rows_to_torch(d_y, T)}
    ttf = TtForm(device, cfg, fn, base, scale).pre_outputs(x)
    found = _precision_report(chk.tag, ref, dev, ttf)
    mixes = site.debug_to_torch(0)["mix"].double()[:T]
    logger.info(
        f"HY4_IHC_SITE precision {chk.tag} normalised mixes vs float64: max_rel "
        f"{((mixes - ref['mixes']).abs() / ref['mixes'].abs()).max():.2e}, relative error "
        f"{((mixes - ref['mixes']).norm() / ref['mixes'].norm()):.2e}"
    )
    # the reference module itself (fp32) against float64, for scale
    r_y, r_pre, r_post, r_stat = golden_pre(cfg, fn, base, scale, x)
    _precision_report(
        f"{chk.tag} [reference/ihc.py in fp32, in place of the site]",
        ref,
        {"1/rms": r_stat, "pre": r_pre, "post": r_post, "y": r_y},
        ttf,
    )
    _stage_report(chk.tag, site, cfg, base, scale)
    chk.pcc("y vs ref", ref["y"], dev["y"])
    if proj == "sfpu":
        for name, bound in (("1/rms", 2e-6), ("pre", 5e-6), ("post", 5e-6), ("y", 5e-6)):
            chk.true(
                f"{name} within {bound:.0e} of float64 with the SFPU projection ({found[name]:.2e})",
                found[name] <= bound,
            )
    chk.done()


@pytest.mark.parametrize("y_dtype", [ttnn.float32, ttnn.bfloat16], ids=["y-fp32", "y-bf16"])
@pytest.mark.parametrize("C, K", [(256, 1), (256, 2), (C_HY4, 8), (C_HY4, 16)], ids=lambda v: str(v))
def test_post_pre_sfpu_projection(device, C, K, y_dtype):
    """The fused post + pre program with the SFPU projection, 8 rows, against the reference."""
    _post_pre_case(device, C, K, "random", 8, y_dtype, label=" sfpu", sfpu_proj=True)


@pytest.mark.parametrize("K", [8, 16, 24, 48], ids=lambda k: f"K{k}")
def test_latency_sfpu_projection(device, K):
    _latency(device, K, None, f"K{K} sfpu projection", sfpu_proj=True)
