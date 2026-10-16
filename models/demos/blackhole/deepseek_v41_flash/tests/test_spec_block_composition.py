# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same prompt P in user 1, greedy text produced by (a) the k=5 spec runner, (b) sibling runners with k=1 and k=3 (different VERIFY BLOCK compositions: the committed tokens are rows of blocks of 2 / 4 / 6
tokens), (c) plain eager decode (the K=0 path). Reports the first diverging generated token and the top1-top2 logit gap there: do different block compositions of the same greedy stream change tokens,
and only at near-ties?

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
def test_spec_greedy_text_vs_block_composition(mesh_device):
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
    print(f"SPECBLK reference k=5: {ref[:12]}... {st0['tok_per_round']:.2f} tok/round", flush=True)
    texts = {"k=5 (natural acceptance)": (ref, gaps)}

    def forced_run(force):
        """the SAME k=5 runner, accept count of every round forced to ``force`` (0 = one token per round: a plain-like stream through the 6-row verify block)"""
        for u in range(B):
            m.release_user(u)
        prompts = full({})
        res = m.prefill_interleaved(
            [(u, p, 0, int(p.numel())) for u, p in prompts.items()], CHUNK, s_pad_max=S_pad, want_logits=False
        )
        lens = torch.tensor([int(prompts[u].numel()) for u in range(B)])
        first = torch.tensor([int(res[u][0]) for u in range(B)])
        X, base = sp.seed_from_prefill(lens, first)
        X, base = X.clone(), base.clone()
        out, gp = [int(X[1, 0])], []
        while len(out) < NEW:
            a_, mm, d = sp._round(X, base, force=force)
            v2 = torch.cat(
                [
                    ttnn.to_torch(ttnn.get_device_tensors(sp.dec.top2)[r * sp.cols]).reshape(sp.U * sp.n, -1)
                    for r in range(sp.rows)
                ]
            ).float()
            t2 = v2.topk(2, dim=-1).values.reshape(B, sp.n, 2)
            for b_ in range(B):
                if b_ == 1:
                    toks = [int(x) for x in a_[b_, : int(mm[b_]) + 1]]
                    out += toks
                    gp += [float(t2[b_, j, 0] - t2[b_, j, 1]) for j in range(len(toks))]
                base[b_] += int(mm[b_]) + 1
                X[b_, 0] = a_[b_, int(mm[b_])]
                X[b_, 1 : 1 + sp.k] = d[b_, : sp.k]
        return out[:NEW], gp

    for f in (0, 2):
        texts[f"k=5 runner, accept forced to {f}"] = forced_run(f)
    # plain eager decode of the same stream
    for u in range(B):
        m.release_user(u)
    prompts = full({})
    res = m.prefill_interleaved(
        [(u, p, 0, int(p.numel())) for u, p in prompts.items()], CHUNK, s_pad_max=S_pad, want_logits=False
    )
    cur = torch.tensor([int(res[u][0]) for u in range(B)])
    plain, pg = [int(cur[1])], []
    for k_ in range(NEW - 1):
        pos = torch.tensor([int(prompts[u].numel()) + k_ for u in range(B)])
        nxt = m.decode_forward(cur, pos, enable_trace=False, reload_inputs=True).clone()
        lg = m.read_logits()[1].float().topk(2).values
        pg.append(float(lg[0] - lg[1]))
        cur = nxt
        plain.append(int(nxt[1]))
    texts["plain"] = (plain, pg)
    base_t, base_g = texts["k=5 (natural acceptance)"]
    for name, (t, g_) in texts.items():
        n = min(len(t), len(base_t))
        j = next((i for i in range(n) if t[i] != base_t[i]), None)
        gap5 = None if j is None or j < 1 or j - 1 >= len(base_g) else base_g[j - 1]
        gapp = None if j is None or j < 1 or j - 1 >= len(pg) else pg[j - 1]
        print(
            f"SPECBLK {name} vs k=5: identical {j is None}, first diff at token {j}, top1-top2 gap there: k=5 stream {gap5}, plain stream {gapp}",
            flush=True,
        )
