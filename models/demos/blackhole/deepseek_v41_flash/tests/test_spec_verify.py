# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1: verify blocks of n = 1 + k tokens per user in ONE layer call (paged cache, causal inside the block), layer by layer, teacher-forced on
the reference chain (dsv4-chain-m: S=9 prefill, 6 decode steps, 16 users). Per layer: PCC of every block token's output vs the reference
decode output of that step, for consecutive blocks (state carried: caches, ratio-2 prev_cs via commit()).
Env: DSV41_LAYERS (default 0-3), DSV41_K (default 1), DSV41_CHAIN, DSV41_BLOCKS (default all that fit in the saved steps)."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.spec_chain import SpecChain
from models.demos.blackhole.deepseek_v41_flash.tt.spec_state import SpecStepState

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-m")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_spec_verify(mesh_device):
    md = mesh_device
    layer_ids = []
    for part in os.environ.get("DSV41_LAYERS", "0-3").split(
        ","
    ):  # e.g. "0-3,20-21"; readers need their kv-source layer in the list
        a, _, b = part.partition("-")
        layer_ids += list(range(int(a), int(b or a) + 1))
    k = int(os.environ.get("DSV41_K", "1"))
    n = 1 + k
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S = toks["prefill_tokens"].shape[1]
    log = lambda m: print(m, flush=True)
    orig = os.environ.get("DSV41_ORIG") == "1"  # baseline: the original single-token attention classes (n must be 1)
    if orig:
        from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
        from models.demos.blackhole.deepseek_v41_flash.tt.step_state import DSV41StepState

        assert n == 1
        chain = DSV41DecodeChain(md, users_per_row=4, log=log)
        chain.n, chain.n_users = 1, chain.B
        chain.tok_dev = lambda per_step, d, steps: chain.to_dev(per_step[steps[0]].reshape(chain.B, -1), d)
        chain.pos_dev = lambda base: ttnn.from_torch(
            torch.as_tensor(base).to(torch.int32),
            device=md,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
        )
    else:
        chain = SpecChain(md, users_per_row=4, n=n, log=log)
    B = chain.n_users
    assert B == 16
    steps_avail = toks["decode_tokens"].shape[1]
    nblocks = min(steps_avail // n, int(os.environ.get("DSV41_BLOCKS", "9")))
    log(f"layers {layer_ids[0]}..{layer_ids[-1]} k={k} n={n}: {nblocks} consecutive blocks of {n} tokens/user (S={S})")
    worst = {}
    built = []
    for (
        L
    ) in (
        layer_ids
    ):  # all layers resident: blocks run layer by layer in model order, so readers see their source's latent of the SAME block
        ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
        ref["S"] = S
        t0 = time.time()
        layer, attn = chain.build_layer(L, ref, load_layer(L))
        ss = DSV41StepState(attn) if orig else SpecStepState(attn)
        if orig:
            attn.commit = lambda m=None: None
        built.append((L, ref, layer, attn, ss))
        log(f"layer {L} built in {time.time() - t0:.0f}s")
    for blk in range(nblocks):
        steps = list(range(blk * n, blk * n + n))
        for L, ref, layer, attn, ss in built:
            tx = chain.tok_dev([ref["dec_in_steps"][s][0] for s in range(len(ref["dec_in_steps"]))], 4 * 5120, steps)
            tp = chain.tok_dev([ref["dec_in_steps"][s][1] for s in range(len(ref["dec_in_steps"]))], 4, steps)
            st = ss.build(chain.pos_dev(S + blk * n * torch.ones(B, dtype=torch.long)))
            t0 = time.time()
            out, nxt = layer.forward(tx, tp, st)
            ttnn.synchronize_device(md)
            t_fwd = time.time() - t0
            got = chain.to_host(out, 4 * 5120).reshape(B, n, 4 * 5120)
            attn.commit()
            pj = [
                R.pcc(got[:, j], ref["dec_out_steps"][s][0].reshape(B, 4 * 5120).float()) for j, s in enumerate(steps)
            ]
            worst[L] = min(worst.get(L, 1.0), min(pj))
            log(
                f"L{L} block {blk} (steps {steps[0]}..{steps[-1]}): PCC per block index {[round(p, 5) for p in pj]}  eager fwd {t_fwd:.2f}s"
            )
    log("WORST per layer: " + str({l: round(v, 5) for l, v in worst.items()}))
    assert all(v > float(os.environ.get("DSV41_MIN_PCC", "0.99")) for v in worst.values()), worst
