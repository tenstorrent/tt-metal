# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Spec decode (k=5, drafter seeded from the prefill taps) analogue of tests/test_decode_batch_sensitivity.py, 40 layers: does the greedy text of user 1 depend on what the OTHER users of the batch are
(their prompt lengths / positions)? Same prompt P in user 1 in every run; the others change. Reports the first diverging generated token and the top1-top2 logit gap (near-tie evidence from
``SpecRunner.run``) of the reference at that token.

    DSV41_LAYERS=0-39 DSV41_POOL_DTYPE=fp8 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_spec_batch_sensitivity.py -s
"""

import os

import pytest
import torch

import ttnn

U, K, CHUNK = 4, 5, 512
PLEN = int(os.environ.get("DSV41_SS_PLEN", "720"))
NEW = int(os.environ.get("DSV41_SS_NEW", "48"))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 1600000000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_spec_greedy_text_vs_batch_composition(mesh_device):
    os.environ.update(
        {
            "DSV41_PF_UMASK": "1",
            "DSV41_PREFILL_UP": "1",
            "DSV41_MO_OVERLAP": "0",
            "DSV41_SPEC": str(K),
            "DSV41_RING_ROWS": "288",
            "DSV41_IDX_SWITCH": "262144",
            "MOE_COMPUTE_FP32_ACC": "1",
            "MOE_COMPUTE_BFP8_WEIGHTS": "1",
        }
    )
    from models.demos.blackhole.deepseek_v41_flash.tt import spec_policy
    from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
    from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    md, B, S_pad = mesh_device, 4 * U, 2048
    layer_ids = list(range(40))
    pp = default_page_params(S_pad + 256, U)
    args, m, _, _ = create_tt_model(
        md,
        B,
        S_pad + 256,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=print,
    )
    gen = Generator([m], [args], md, tokenizer=args.tokenizer)
    gen.spec_choice = spec_policy.SpecChoice("fixed", K, [], False, "test", True, ring_rows=spec_policy.RING_ROWS_SPEC)
    gen.enable_spec(K)
    sp = gen.spec
    m._admit_idle()
    sp.prepare(torch.zeros(B, sp.n, dtype=torch.long), torch.full((B,), 130, dtype=torch.long))
    g = torch.Generator().manual_seed(31)
    rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g, dtype=torch.int64)
    P = rnd(PLEN)
    others = [rnd(PLEN) for _ in range(B)]

    def run(prompts, obs=1):
        for u in range(B):
            m.release_user(u)
        res = m.prefill_interleaved(
            [(u, p, 0, int(p.numel())) for u, p in prompts.items()], CHUNK, s_pad_max=S_pad, want_logits=False
        )
        lens = torch.tensor([int(prompts[u].numel()) for u in range(B)])
        first = torch.tensor([int(res[u][0]) for u in range(B)])
        X0, base = sp.seed_from_prefill(lens, first)
        out, stats = sp.run(X0, base, NEW)
        return out[obs][:NEW], list(sp.gaps[obs]), stats

    def full(extra):
        d = {u: others[u] for u in range(B)}
        d[1] = P
        d.update(extra)
        return d

    ref, gaps, st0 = run(full({}))
    print(f"reference text ids: {ref[:16]}...; spec stats {st0['tok_per_round']:.2f} tok/round", flush=True)
    results = {}
    for name, extra in {
        "repeat": {},
        "row-mate user 0 one token shorter": {0: rnd(PLEN - 1)},
        "row-mate user 0 100 shorter": {0: rnd(PLEN - 100)},
        "other row user 7 much longer (1500)": {7: rnd(1500)},
        "all other users 100 longer": {u: rnd(PLEN + 100) for u in range(B) if u != 1},
    }.items():
        out, _, _ = run(full(extra))
        n = min(len(out), len(ref))
        j = next((i for i in range(n) if out[i] != ref[i]), None)
        gap = None if j is None or j - 1 >= len(gaps) or j < 1 else gaps[j - 1]
        results[name] = (j, gap)
        print(
            f"SPECSENS {name}: identical {j is None} (first diff at generated token {j}, top1-top2 gap of the reference there: {gap})",
            flush=True,
        )
    print(
        f"SPECSENS reference near-tie gaps: min {min(gaps):.3f}, tokens with gap < 0.1: {sum(1 for x in gaps if x < 0.1)}/{len(gaps)}",
        flush=True,
    )
