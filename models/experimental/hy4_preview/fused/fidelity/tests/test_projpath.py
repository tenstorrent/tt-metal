# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""hifi_mm inside the projections piece's own programs (projpath.py): their units, DRAM weight streams, hub, SFPU
finishing and BF16 outputs, with only the matmul call replaced.

    test_stream_probe      the piece's one-hot operand probe (8 K tiles, one output tile) reading EVERY weight row,
                           through its own compute kernel and through the hifi_mm one, against the weights the
                           stream tensor holds (a sharded block-format tensor is rounded to BF16 first) and against
                           the piece's reference (one rounding on the host)
    test_pre_accuracy      the pre group on checkpoint weights (layers $HY4_FID_LAYERS, chip $HY4_FID_CHIP, BFP8)
                           against the piece's golden: q_a and the gate, original kernel and each pass schedule
    test_pre_latency       traced latency of the pre group (random weights, BFP8), original kernel and each pass
                           schedule, with and without the DRAM reads

"custom_mm" is the piece's kernel as it is. Log lines start with HY4_MK9. On the mock Galaxy the programs are
compiled only."""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.common.tracing import traced_ms
from models.experimental.hy4_preview.fused.fidelity import lab as LAB
from models.experimental.hy4_preview.fused.fidelity import probes as PR
from models.experimental.hy4_preview.fused.fidelity.projpath import use_hifi
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import program as P

OUT_DIR = Path(os.environ.get("HY4_PROBE_DIR", "generated/hy4_preview/logs/mk9_probe"))
CHIP = int(os.environ.get("HY4_FID_CHIP", "13"))
LAYERS = [int(v) for v in os.environ.get("HY4_FID_LAYERS", "0,1,2,3,4,5").split(",")]
TRACE_REGION = 64 << 20
# form: (fidelity or None for the piece's own kernel, how the split accumulators are handled: projpath.SPLIT)
FORMS = {
    "custom_mm": (None, None),
    "lofi finalize": ("lofi", "finalize"),
    "lofi": ("lofi", "merge"),
    "hifi2": ("hifi2", "merge"),
    "hifi3": ("hifi3", "merge"),
    "hifi4": ("hifi4", "merge"),
    "p02": ("p02", "merge"),
    "hifi2 plain": ("hifi2", "plain"),
    "hifi4 plain": ("hifi4", "plain"),
    "hifi4 finalize": ("hifi4", "finalize"),
}
_forms = [f for f in os.environ.get("HY4_FID_PATH_FORMS", ",".join(FORMS)).split(",") if f]
K_TILES = 8


def apply(op, form):
    fidelity, split = FORMS[form]
    return op if fidelity is None else use_hifi(op, fidelity, split)


@pytest.mark.parametrize("form", ["custom_mm", "lofi finalize", "lofi", "hifi2", "hifi4", "hifi4 plain"])
@pytest.mark.parametrize("fmt", ["bfp8", "bfp4"])
def test_stream_probe(device, fmt, form):
    g = torch.Generator().manual_seed(4)
    wt = torch.randn(L.TILE * K_TILES, L.TILE, generator=g) * 0.05
    op = P.ProjProgram(device, [P.Unit("probe", wt, 1, fin=L.FIN_NONE, rnd=P.RND_NONE)], K_TILES, 1, fmt=fmt)
    apply(op, form)
    host = L.stored(wt, fmt)  # the piece's reference: one rounding on the host
    stored = LAB.stored_sharded(wt, fmt)  # what a sharded tensor holds: BF16 first (tests/test_stream_reference.py)
    seen = torch.zeros_like(stored)
    for run in range(L.TILE * K_TILES // L.ROWS):
        rows = [8 * run + r for r in range(L.ROWS)]
        x = torch.zeros(L.ROWS, L.TILE * K_TILES)
        x[torch.arange(L.ROWS), rows] = 1.0
        op.set_x(x)
        op()
        if P.MOCK:
            return
        out, _, status = op.read()
        op.check_status(status)
        seen[rows] = out
    cut5 = PR.trunc_sig(stored, 5)
    want = cut5 if form in ("custom_mm", "lofi finalize", "lofi") else stored
    odd = seen != want
    logger.info(
        f"HY4_MK9 stream probe {fmt} [{form}]: against the weights a sharded tensor holds (BF16, then {fmt}): equal "
        f"{float((seen == stored).float().mean()):.4f}, equal to them cut to 5 significant bits "
        f"{float((seen == cut5).float().mean()):.4f}, {int(odd.sum())} of {odd.numel()} entries are not what this "
        f"form should give; against the host's single rounding: equal {float((seen == host).float().mean()):.4f}, "
        f"equal to it cut to 5 significant bits {float((seen == PR.trunc_sig(host, 5)).float().mean()):.4f}"
    )
    pairs = [(stored[i, j].item(), seen[i, j].item(), i, j) for i, j in odd.nonzero()[:12].tolist()]
    logger.info(f"HY4_MK9 stream probe {fmt} [{form}]: first (stored, device, K row, column): {pairs}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(dict(wt=wt, stored=stored, host=host, seen=seen), OUT_DIR / f"stream_{fmt}_{form.replace(' ', '_')}.pt")


@pytest.mark.parametrize("layer", LAYERS)
def test_pre_accuracy(device, layer):
    if not os.environ.get("HY4_HF_MODEL"):
        pytest.skip("HY4_HF_MODEL is not set")
    from models.experimental.hy4_preview.fused.projections import prepare
    from models.experimental.hy4_preview.fused.projections.tests.harness import check, random_input, tables

    lw = prepare.layer_weights(layer)
    w = L.chip_weights(lw, CHIP)
    y, scale = random_input(seed=10 + layer)
    _, _, d_cos, d_sin = tables()
    ref = L.pre_golden(y, scale, w, d_cos, d_sin)
    # the prepared streams are sharded tensors: BF16 first, then BFP8
    stored_w = {k: LAB.stored_sharded(w[k].contiguous(), "bfp8") for k in ("q_a", "gate")}
    ref_stored = L.pre_golden(y, scale, {**w, **stored_w}, d_cos, d_sin)
    for form in _forms:
        op = P.pre_program(
            device, w, fmt="bfp8", cache=prepare.stream_cache(prepare.DEFAULT_OUT, layer, "bfp8", CHIP, "pre")
        )
        apply(op, form)
        op.set_x(torch.cat((y, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1))
        op.set_params(scale=scale)
        op()
        if P.MOCK:
            continue
        out, _, status = op.read()
        op.check_status(status)
        tag = f"pre L{layer} chip {CHIP}"
        q_a = out[:, : 2 * L.TILE]
        gate = out[:, P.PRE_GATE * L.TILE : P.PRE_IDX * L.TILE]
        check(f"{tag} q_a [{form}] vs checkpoint weights", ref["q_a"], q_a, prefix="HY4_MK9")
        check(f"{tag} q_a [{form}] vs stored weights", ref_stored["q_a"], q_a, prefix="HY4_MK9")
        check(f"{tag} gate [{form}] vs checkpoint weights", ref["gate"], gate, prefix="HY4_MK9")
        check(f"{tag} gate [{form}] vs stored weights", ref_stored["gate"], gate, prefix="HY4_MK9")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
@pytest.mark.parametrize("variant", ["", "noread"])
@pytest.mark.parametrize("form", _forms)
def test_pre_latency(device, form, variant):
    from models.experimental.hy4_preview.fused.projections.tests.test_latency import random_share

    calls = int(os.environ.get("HY4_LATENCY_CALLS", "8"))
    w = random_share(0, False)
    defines = [("HY4_NO_READ", "1")] if variant else []
    op = P.pre_program(device, w, fmt="bfp8", defines=defines)
    apply(op, form)
    g = torch.Generator().manual_seed(1)
    op.set_x(torch.randn(L.ROWS, L.TILE * op.x_tiles, generator=g))
    op.set_params(lanes=torch.full((L.ROWS, 8), 256.0))
    if P.MOCK:
        op()
        return
    t = traced_ms(device, op, calls=calls)
    _, _, status = op.read()
    op.check_status(status)
    logger.info(
        f"HY4_MK9 pre latency [{form}]{' ' + variant if variant else ''} ({calls} per trace): "
        f"{t['replay_ms'] * 1e3:.1f} us per call (rounds {[round(v * 1e3, 1) for v in t['rounds_ms']]})"
    )
