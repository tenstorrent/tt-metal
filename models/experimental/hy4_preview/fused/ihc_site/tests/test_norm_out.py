# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The layer program's form of the pre-mix on one chip (IhcSite.program norm_out): after the pre-mix the site cores
apply the sublayer's RMSNorm, x = y * (1 / RMS(y) of the row) * gamma, in the SFPU, and hand x as BF16 row tiles to
the chip's sender core. Here no sender runs: the staging buffer and the words are read back by the host.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=20 pytest <this file> -x

Checks: x against reference.attention.rms_norm of the reference's y (PCC, max abs error), and against the fp32
model of what the device computes from its OWN y and statistic ((y * s) * gamma in fp32, rounded to BF16): bit-equal
or within one BF16 step, counted; the words (ready bit set once per run, gather word back at zero, no status).
Cases: the site as the MLP layer program places it (16 cores as 2 columns x 8 rows at (5, 0), CB ids from 32) and
the default rectangle; pre alone and post + pre; random and checkpoint (layer 1, MLP site) parameters.
Log lines start with HY4_MK8."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.collective.collective import to_bf16
from models.experimental.hy4_preview.fused.ihc_site.op import IhcSite, NormOut, SiteRef, golden_post, golden_pre
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.fused.moe.layout import from_row_tiles
from models.experimental.hy4_preview.reference.attention import rms_norm
from models.experimental.hy4_preview.reference.ihc import IHCConfig
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc

MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))
SENDER = (6, 9)
C = 6144
LAYOUTS = {"layer": dict(num_cores=16, origin=(5, 0), per_row=2, cb_base=32), "rect": dict(num_cores=16)}


def params_of(weights, cfg):
    if weights == "random":
        g = torch.Generator().manual_seed(21)
        fn = (torch.randn(cfg.mix_hc, cfg.n * cfg.dim, generator=g) * 0.02).to(torch.bfloat16).float()
        gamma = (1.0 + 0.2 * torch.randn(cfg.dim, generator=g)).to(torch.bfloat16).float()
        return (fn, torch.randn(cfg.mix_hc, generator=g), torch.tensor([0.01, 0.005])), gamma
    prefix = "model.layers.1."
    fn, base, scale, gamma = hy4_tensors(
        [prefix + "hc_mlp_layer.hc_pre." + n for n in ("hc_fn", "hc_base", "hc_scale")]
        + [prefix + "post_attention_layernorm.weight"]
    )
    return (fn, base, scale), gamma


@pytest.mark.parametrize("gamma_from", ["l1", "table"])
@pytest.mark.parametrize("weights", ["random", "checkpoint"])
@pytest.mark.parametrize("where", list(LAYOUTS))
def test_norm_out(device, where, weights, gamma_from):
    """gamma_from "table": the norm's weight comes from the site table in DRAM (site_table(gammas=...),
    NormOut(gamma=None)), as site 1 of two so that the page arithmetic is exercised."""
    cfg = IHCConfig(dim=C)
    T = 8
    (fn, base, scale), gamma = params_of(weights, cfg)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(T, cfg.n, C, generator=g)
    f = torch.randn(T, C, generator=g)
    site = IhcSite(device, cfg, **LAYOUTS[where])
    index = 0
    if gamma_from == "table":
        other = (fn.flip(1), base.flip(0), scale)
        table = site.site_table([other, (fn, base, scale)], gammas=[gamma.flip(0) * 3.0, gamma])
        index = 1
    else:
        table = site.site_table([(fn, base, scale)])
    words = tp.core_buffer(device, tp.grid_set(device), 4 * tp.WORDS, ttnn.uint32, width=tp.WORDS)
    staging = tp.core_buffer(device, [SENDER], 2 * 8 * C)
    grid = device.compute_with_storage_grid_size()
    sender = device.worker_core_from_logical_core(ttnn.CoreCoord(*SENDER))
    w = words.buffer_address()
    norm = NormOut(
        gamma=site.gamma_to_device(gamma) if gamma_from == "l1" else None,
        xn=site.empty_rows(ttnn.bfloat16),
        sender=sender.x | (sender.y << 16),
        staging=staging.buffer_address(),
        ready=w + 4 * tp.W_READY,
        ready_bit=1,
        gather=w + 4 * tp.W_GATHER,
        status=w + 4 * tp.W_STATUS,
        wait_limit=20_000_000,
    )
    d_x = site.streams_to_device(x)
    d_f = site.rows_to_device(f, ttnn.bfloat16)
    d_y = site.empty_rows()
    run = ttnn.experimental.prepare_generic_op if MOCK else ttnn.generic_op
    failed = []
    r_x = x
    for step in ("pre", "post_pre"):
        io, desc = site.program(
            d_x, f=d_f if step == "post_pre" else None, params=SiteRef(table, index), y=d_y, norm_out=norm
        )
        run(io, desc)
        if MOCK:
            continue
        if step == "post_pre":
            r_x = golden_post(cfg, r_x, f.to(torch.bfloat16).float(), r_post)
        r_y, _, r_post, r_stat = golden_pre(cfg, fn, base, scale, r_x)
        ref = rms_norm(r_y, gamma.float(), cfg.norm_eps)
        y, stat = site.rows_to_torch(d_y, T), site.stat_to_torch(T)
        got = from_row_tiles(ttnn.to_torch(staging).float().reshape(-1, 256))
        model = to_bf16((y * stat[:, None]) * gamma.float()[None, :], "away")
        step16 = torch.pow(2.0, torch.floor(torch.log2(model.abs().clamp_min(1e-30))) - 7)
        differ = int((got != model).sum())
        beyond = int(((got - model).abs() > step16).sum())
        wd = ttnn.to_torch(words).reshape(-1, tp.WORDS).to(torch.int64) & 0xFFFFFFFF
        ready = int(wd[SENDER[1] * grid.x + SENDER[0], tp.W_READY])
        root = site.cores[0]
        gather = int(wd[root.y * grid.x + root.x, tp.W_GATHER])
        status = int(wd[root.y * grid.x + root.x, tp.W_STATUS])
        logger.info(
            f"HY4_MK8 norm_out {where} {weights} gamma from {gamma_from} {step}: x vs reference norm pcc={pcc(ref, got):.7f} "
            f"max_abs={(ref - got).abs().max():.3e} (BF16 rounding of the reference itself: "
            f"{(ref - ref.to(torch.bfloat16).float()).abs().max():.3e}); vs the fp32 model of the device's own y and "
            f"statistic: {differ} of {got.numel()} elements differ, {beyond} by more than one BF16 step; "
            f"y pcc={pcc(r_y, y):.7f}; words: ready {ready:#x}, gather {gather}, status {status:#x}"
        )
        want_ready = 1 if step == "pre" else 2  # no sender runs here: the bits add up
        if pcc(ref, got) < 0.9999:
            failed.append(f"{step}: pcc {pcc(ref, got)}")
        if beyond:
            failed.append(f"{step}: {beyond} elements off the fp32 model by more than a BF16 step")
        if (ready, gather, status) != (want_ready, 0, 0):
            failed.append(f"{step}: words ready {ready} gather {gather} status {status:#x}")
    assert not failed, "; ".join(failed)
