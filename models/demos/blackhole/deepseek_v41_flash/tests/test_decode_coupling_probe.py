# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: where does the decode of layer 2 of user 1 start to depend on the position of its row-mate user 0 (tests/test_decode_batch_sensitivity.py)? Eager decode, per-op capture of the
compressed layer's tensors; prints, per captured tensor, which rows (users of the mesh row 0) differ between the reference run and the run with user 0 one token shorter.

    DSV41_LAYERS=0-2 pytest ...test_decode_coupling_probe.py -s
"""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

U, PLEN, CHUNK = 4, 720, 512


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
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_probe(mesh_device):
    os.environ.update({"DSV41_PF_UMASK": "1", "DSV41_PREFILL_UP": "1", "DSV41_MO_OVERLAP": "0"})
    md, B, S_pad = mesh_device, 4 * U, 2048
    pp = default_page_params(S_pad + 256, U)
    args, m, _, _ = create_tt_model(
        md,
        B,
        S_pad + 256,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=[0, 1, 2],
        log=print,
    )
    g = torch.Generator().manual_seed(5)
    rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g, dtype=torch.int64)
    P, others, other_user0 = rnd(PLEN), [rnd(PLEN) for _ in range(B)], rnd(PLEN - 1)
    forced = rnd(3)
    cap = {}
    at = m.attns[2]
    from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as PO

    def rec(name, t):
        if isinstance(t, ttnn.Tensor):
            cap.setdefault(name, []).append(ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().clone())

    orig_cs, orig_att, orig_kv = at._compress_step, at._paged_attend, PO.paged_kv_step

    def cs(x, st):
        rec("x_in", x)
        lat = orig_cs(x, st)
        rec("lat", lat)
        rec("prev_cs_after", at.prev_cs)
        return lat

    def att(q, lat, st, ratio, src_off, topk_out):
        rec("q", q)
        rec("pos", st["pos"])
        rec("topk_ids", st.get("topk_ids"))
        rec("nvalid", st.get("nvalid"))
        rec("valid", st.get("valid"))
        o = orig_att(q, lat, st, ratio, src_off, topk_out)
        rec("o", o)
        return o

    def kv(*a, **k):
        idx = orig_kv(*a, **k)
        rec("sdpa_idx", idx)
        return idx

    at._compress_step, at._paged_attend = cs, att
    PO.paged_kv_step = kv
    import models.demos.blackhole.deepseek_v41_flash.tt.paged_attention as PA

    PA.P.paged_kv_step = kv

    def run(u0):
        cap.clear()
        for u in range(B):
            m.release_user(u)
        prompts = {u: others[u] for u in range(B)}
        prompts[1], prompts[0] = P, u0
        m.prefill_interleaved(
            [(u, p, 0, int(p.numel())) for u, p in prompts.items()], CHUNK, s_pad_max=S_pad, want_logits=True
        )
        for k in range(2):
            tok = torch.randint(1000, 100000, (B,), generator=torch.Generator().manual_seed(9 + k), dtype=torch.int64)
            tok[1] = forced[k]
            pos = torch.tensor([int(prompts[u].numel()) + k for u in range(B)])
            m.decode_forward(tok, pos, enable_trace=False, reload_inputs=True)
        return {k: [x.clone() for x in v] for k, v in cap.items()}

    run(others[0])  # (compile / warm-up pass: its captures are dropped)
    A = run(others[0])
    A = run(P.clone()[:PLEN].clone()) if False else A
    Bc = run(other_user0)
    for name in A:
        print(f"PROBE counts {name}: {len(A[name])} vs {len(Bc[name])}")
        for k, (x, y) in enumerate(zip(A[name][-2:], Bc[name][-2:])):
            if x.shape != y.shape:
                print(f"{name}[{k}] shape {tuple(x.shape)} vs {tuple(y.shape)}")
                continue
            d = x != y
            info = []
            for dim in range(d.dim()):
                if d.shape[dim] == U:
                    ad = (x - y).abs().transpose(dim, 0).reshape(U, -1).amax(dim=1)
                    info.append((dim, [round(float(v), 4) for v in ad]))
            print(
                f"PROBE {name}[{k}] shape {tuple(x.shape)} differs: {bool(d.any())} users-dim diffs {info}", flush=True
            )
