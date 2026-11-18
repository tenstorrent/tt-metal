"""B40 unit test (1x8 slot, MESH_DEVICE=1x8): spec_accept / spec_state one-core programs vs the stock ttnn op glue of hy4_spec_trace (bit-exact). RES lines."""
import json
import os
import random

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt import hy4_derive_fused as B
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_spec_trace import accept

W16 = 16


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def snap0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()


def main():
    S, k = 4, 3
    full, md = open_row()
    try:
        rnd = random.Random(11)
        f32 = lambda v, sh: upload(md, torch.tensor(v, dtype=torch.float32).reshape(*sh), ttnn.float32)
        ar16 = f32(list(range(16)), (1, 1, 1, 16))
        arS = f32(list(range(S)), (1, 1, S, 1))
        padvec = f32([120002.0] + [0.0] * (S - 1), (1, 1, 1, S))
        neg_pad = f32([-1.0] * (W16 - k), (1, 1, 1, W16 - k))
        zeros_pad = f32([0.0] * (W16 - S), (1, 1, 1, W16 - S))
        # persistent outputs
        z = lambda dt=ttnn.float32, sh=(1, 1, 1, 1), lay=ttnn.TILE_LAYOUT: upload(md, torch.zeros(*sh), dt, lay)
        a_out, m_out, p_new, nxt = z(sh=(1, 1, 1, 16)), z(), z(), z()
        oh_col = z(ttnn.bfloat16, (1, 1, S, 1))
        d_row, pos, tokt = (
            z(sh=(1, 1, 1, 16)),
            z(),
            upload(md, torch.zeros(1, S, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        )
        bad_a, bad_s = 0, 0
        N = int(os.environ.get("N", "80"))
        for t in range(N):
            drafts = [rnd.randrange(120000) if rnd.random() < 0.5 else rnd.randrange(5) for _ in range(k)]
            tgt = [d if rnd.random() < 0.65 else rnd.randrange(5) for d in drafts] + [rnd.randrange(120000)]
            cap = rnd.choice([0, 1, 2, 3, 5])
            P = rnd.choice([0, 1, 513, 65535, 1048000 + rnd.randrange(500)])
            dvec = torch.full((16,), -1.0)
            dvec[:k] = torch.tensor(drafts, dtype=torch.float32)
            a_tok = upload(md, torch.tensor(tgt, dtype=torch.int32).reshape(S, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            d_t = upload(md, dvec.reshape(1, 1, 1, 16), ttnn.float32)
            cap_t, pos_t = f32([float(cap)], (1, 1, 1, 1)), f32([float(P)], (1, 1, 1, 1))
            # stock
            a_f = ttnn.typecast(ttnn.to_layout(ttnn.reshape(a_tok, [1, 1, 1, S]), ttnn.TILE_LAYOUT), ttnn.float32)
            a_row = ttnn.concat([a_f, zeros_pad], dim=3)
            m, onehot, nx = accept(a_row, d_t, cap_t, k, ar16, None, None)
            oh_ref = ttnn.typecast(ttnn.eq(arS, m), ttnn.bfloat16)
            pn_ref = ttnn.add(ttnn.add(pos_t, m), 1.0)
            B.spec_accept(a_tok, d_t, cap_t, pos_t, a_out, m_out, oh_col, p_new, nxt, S, k)
            ttnn.synchronize_device(md)
            ok = (
                torch.equal(snap0(a_out).reshape(-1)[:16], snap0(a_row).reshape(-1)[:16])
                and snap0(m_out).reshape(-1)[0] == snap0(m).reshape(-1)[0]
                and snap0(p_new).reshape(-1)[0] == snap0(pn_ref).reshape(-1)[0]
                and snap0(nxt).reshape(-1)[0] == snap0(nx).reshape(-1)[0]
                and torch.equal(snap0(oh_col).reshape(S), snap0(oh_ref).reshape(S))
            )
            if not ok:
                bad_a += 1
                if bad_a <= 3:
                    res(
                        event="accept_bad",
                        tgt=tgt,
                        drafts=drafts,
                        cap=cap,
                        P=P,
                        got=[snap0(x).reshape(-1)[:16].tolist() for x in (a_out, m_out, p_new, nxt)],
                        oh=snap0(oh_col).reshape(S).tolist(),
                        ref=[snap0(x).reshape(-1)[:4].tolist() for x in (m, pn_ref, nx)],
                    )
            # state
            for park in (0.0, 1.0):
                for use_pv in (False, True):
                    pvv = float(rnd.randrange(0, 5000))
                    park_t = f32([park], (1, 1, 1, 1))
                    pv_t = f32([pvv], (1, 1, 1, 1)) if use_pv else None
                    dts = [f32([float(d)], (1, 1, 1, 1)) for d in drafts]
                    keep = ttnn.subtract(ttnn.multiply(park_t, -1.0), -1.0)
                    dcat = ttnn.concat(dts, dim=3)
                    d_pad = ttnn.concat([dcat, neg_pad], dim=3)
                    d_eff = ttnn.subtract(ttnn.multiply(d_pad, keep), park_t)
                    tok_f = ttnn.concat([nx, dcat], dim=3)
                    tok_eff = ttnn.add(ttnn.multiply(tok_f, keep), ttnn.multiply(padvec, park_t))
                    tok_ref = ttnn.to_layout(ttnn.typecast(tok_eff, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
                    p_eff = ttnn.multiply(pn_ref, keep)
                    if pv_t is not None:
                        p_eff = ttnn.add(p_eff, ttnn.multiply(pv_t, park_t))
                    B.spec_state(dts, nxt, p_new, park_t, pv_t, d_row, tokt, pos, S, k)
                    ttnn.synchronize_device(md)
                    ok = (
                        torch.equal(snap0(d_row).reshape(-1)[:16], snap0(d_eff).reshape(-1)[:16])
                        and torch.equal(
                            snap0(tokt).reshape(-1)[:S].to(torch.int64), snap0(tok_ref).reshape(-1)[:S].to(torch.int64)
                        )
                        and snap0(pos).reshape(-1)[0] == snap0(p_eff).reshape(-1)[0]
                    )
                    if not ok:
                        bad_s += 1
                        if bad_s <= 3:
                            res(
                                event="state_bad",
                                park=park,
                                pv=pvv if use_pv else None,
                                got=[
                                    snap0(d_row).reshape(-1)[:16].tolist(),
                                    snap0(tokt).reshape(-1)[:S].tolist(),
                                    snap0(pos).reshape(-1)[0].item(),
                                ],
                                ref=[
                                    snap0(d_eff).reshape(-1)[:16].tolist(),
                                    snap0(tok_ref).reshape(-1)[:S].tolist(),
                                    snap0(p_eff).reshape(-1)[0].item(),
                                ],
                            )
        res(event="tail", cases=N, accept_bad=bad_a, state_bad=bad_s, ok=bad_a == 0 and bad_s == 0)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
