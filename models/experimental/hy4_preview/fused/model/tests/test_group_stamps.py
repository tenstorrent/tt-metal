# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where a step's MLP programs spend their time chip by chip, layer by layer, in either expert layout
(HY4_MLP_SPLIT=tp32 | tp8g4; fused/layer_mlp/PLAN.md section 15): ONE untraced step of tests/test_chain.py's Chain,
its programs launched layer by layer; after every MoE layer's MLP program the hubs' stamps of all 32 chips (x
arrived, expert cores released, partial complete: data-movement stamps the MLP half has, layer_mlp/kernels/
mlp_hub.cpp) and the rows' picks are read. A chip's stamps are on its own clock, so "x arrived -> partial complete"
is valid in an untraced run. NOT a latency figure of the step (the stamp build, untraced, a host read after every
layer): it shows which chips and which group a layer waits for.

    HY4_MOCK=1 HY4_MLP_SPLIT=tp8g4 pytest <this file>
    HY4_CHAIN_LAYERS=78 HY4_MLP_SPLIT=tp8g4 HY4_ROPE_PAIRING=interleaved HY4_GOLDEN_DIR=... HY4_LATENT_SCALE=1.0430 \\
        TT_METAL_OPERATION_TIMEOUT_SECONDS=30 pytest <this file> -x

Only the MLP half is built with stamps (the Stack's ``timing`` is set after its AttnLayer exists): the attention
programs are the binaries of every other run. Log lines start with HY4_TP8."""

import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.fused.model.tests import test_chain as TC
from models.experimental.hy4_preview.tests.common import pcc

CYCLE_US = 1.0 / 1350  # the cores' wall clock counts AICLK cycles (1350 MHz)


def us(a, b):
    return ((int(b) - int(a)) & 0xFFFFFFFF) * CYCLE_US


@TC.galaxy
def test_group_stamps(mesh_device, monkeypatch):
    def stack_with_mlp_stamps(*args, **kwargs):
        st = Stack(*args, **kwargs)
        st.timing = True  # the MlpLayers and their transport are made later, by the first add_layer
        return st

    monkeypatch.setattr(TC, "Stack", stack_with_mlp_stamps)
    chain = TC.Chain(mesh_device, wait_limit=30_000_000)
    st, mesh = chain.st, mesh_device
    try:
        chain.set_batch(0)
        if (
            not is_mock()
        ):  # the L1 of the step as it is compiled here: tensors on the board, the programs' largest buffer sets
            report = st.l1_report(st.layers[-1])
            logger.info(
                f"HY4_TP8 stamps L1 on the board ({TC.MLP_SPLIT}, MLP stamp build): tensors {report['tensors']} of {report['core']} B per core; "
                + "; ".join(
                    f"{name} chip {coord}: largest buffer set {v[0]} B on core {v[1]}, {report['core'] - report['tensors'] - v[0]} B free"
                    for (name, coord), v in ((k, v) for k, v in report.items() if isinstance(k, tuple))
                )
            )
        programs = st.step_programs()
        label = f"stamps {chain.label} {chain.n} layers"
        root_chip = st.topo.chip(st.topo.root)
        hub = st.hub
        rows = []
        for n, state in enumerate(st.layers):
            Stack.launch(programs[3 * n : 3 * n + 3])
            if is_mock() or state.mlp.dense:
                continue
            ttnn.synchronize_device(mesh)
            words = st.mt.read_words()
            stamp = {
                c: [int(v) for v in st.mt.core_words(words[c], hub)[tp.W_STAMP : tp.W_STAMP + 4]] for c in st.topo.chips
            }
            picks = ttnn.to_torch(ttnn.get_device_tensors(state.mlp.op.picks)[root_chip]).to(torch.int32) & 0xFFFF
            union = sorted(set(picks[:, :8].flatten().tolist()))
            groups = [sum(1 for e in union if e // 64 == g) for g in range(4)]
            x2p = {c: us(v[1], v[3]) for c, v in stamp.items()}  # x arrived -> partial complete
            router = sorted(us(v[1], v[2]) for v in stamp.values())
            cols = [sorted(x2p[(r, c)] for r in range(8)) for c in range(4)]
            slowest = max(x2p, key=x2p.get)
            rows.append(
                dict(
                    layer=state.key,
                    union=len(union),
                    groups=groups,
                    cols=cols,
                    slowest=slowest,
                    worst=x2p[slowest],
                    median=sorted(x2p.values())[16],
                )
            )
            if n < 6 or n % 12 == 5 or n == len(st.layers) - 1:
                logger.info(
                    f"HY4_TP8 {label} layer {state.key}: union {len(union)} (per group {groups}); router phase median {router[16]:.0f} us; "
                    f"x to partial by group (min / median / max over its 8 chips): "
                    + "; ".join(f"g{c} {v[0]:.0f} / {v[4]:.0f} / {v[-1]:.0f}" for c, v in enumerate(cols))
                    + f"; slowest chip {slowest} {x2p[slowest]:.0f} us"
                )
        Stack.launch(programs[3 * len(st.layers) :])
        if is_mock():
            return
        ttnn.synchronize_device(mesh)
        status = TC.full_status(chain)
        assert not status, f"a status is left after the step: {status}"
        streams = st.read_streams()
        gold = chain.gold(chain.n - 1, "streams_out")
        m = len(rows)
        mean = lambda xs: sum(xs) / max(len(xs), 1)
        by_group = [sum(1 for r in rows if r["slowest"][1] == g) for g in range(4)]
        busiest = [max(r["groups"]) for r in rows]
        logger.info(
            f"HY4_TP8 {label} SUMMARY over {m} MoE layers (one untraced step, MLP stamp build): streams after the last layer vs the "
            f"golden pcc={pcc(gold, streams):.6f}; union mean {mean([r['union'] for r in rows]):.2f} (min {min(r['union'] for r in rows)}, max "
            f"{max(r['union'] for r in rows)}); busiest group mean {mean(busiest):.2f} (max {max(busiest)}); x to partial: the slowest chip of a "
            f"layer mean {mean([r['worst'] for r in rows]):.1f} us (max {max(r['worst'] for r in rows):.0f}), the median chip of a layer mean "
            f"{mean([r['median'] for r in rows]):.1f} us; the slowest chip is in group 0 / 1 / 2 / 3 on {by_group} layers; group median "
            f"x to partial, mean over the layers: {[round(mean([r['cols'][g][4] for r in rows]), 1) for g in range(4)]}"
        )
        worst = max(rows, key=lambda r: r["worst"])
        logger.info(
            f"HY4_TP8 {label} the layer that waited longest: layer {worst['layer']}, union {worst['union']} (per group {worst['groups']}), "
            f"slowest chip {worst['slowest']} {worst['worst']:.0f} us"
        )
    finally:
        chain.free()
