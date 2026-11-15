"""CPU check of the DEVICE-RESIDENT step state of the sampling design: every per-step input of the pool decoder (write row, key counts, col info, rope tables) is a pure function of the
position; the device derives them from ONE resident position tensor with fp32 exact arithmetic (multiply by powers of two, floor, subtract, clamp, compare), so no per-step upload.
Checks (all positions 0 .. 2**20 + 63): (1) derived write_rows / key_counts / colinfo == hy4_pool.write_rows / key_counts / colid_host bit for bit; (2) two-level rope tables
cos(a_hi + a_lo) = cos a_hi cos a_lo - sin a_hi sin a_lo (1024 + 1024 rows) vs the fp64 truth and vs the host fp32 table rope_tables (the one the model uses today), after the bf16 rounding the device tables have."""
import sys

import torch

sys.path.insert(0, "/mnt/tt-data/ssinghal/wt/hy4_samp")
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_rope import rope_tables

CHUNK, COLS, SLICE = HP.CHUNK, HP.COLS, HP.SLICE


def derive(pos):  # pos fp32 [N] (exact integers < 2**24); returns wr [N,COLS], kv [N,COLS], n [N]
    p = pos.float()
    hi = torch.floor(p * (1.0 / CHUNK))
    lo = p - hi * CHUNK
    col = torch.floor(lo * (1.0 / SLICE))
    loc = hi * SLICE + (lo - col * SLICE)
    cid = torch.arange(COLS, dtype=torch.float32)[None, :]
    wr = torch.where(col[:, None] == cid, loc[:, None], torch.full((1, 1), -1.0))
    n = p + 1
    full = torch.floor(n * (1.0 / CHUNK))
    rem = n - full * CHUNK
    kv = full[:, None] * SLICE + torch.clamp(rem[:, None] - SLICE * cid, 0, SLICE)
    return wr.int(), kv.int(), n.int()


def main():
    N = 2**20 + 64
    pos = torch.arange(N)
    wr, kv, n = derive(pos.float())
    bad = 0
    for s in range(0, N, 1 << 16):
        e = min(N, s + (1 << 16))
        pl = pos[s:e].tolist()
        wr_h = HP.write_rows(pl, len(pl))[:, :, 0, 0]
        kv_h = HP.key_counts(pl, len(pl))[:, :, 0, 0]
        ci_h = HP.colid_host(len(pl), [p + 1 for p in pl])[:, 0, 0, 1]
        bad += int((wr_h != wr[s:e]).sum() + (kv_h != kv[s:e]).sum() + (ci_h != n[s:e]).sum())
    print("derived write_rows / key_counts / colinfo vs host helpers: mismatches =", bad, "over", N, "positions")
    # rope: two-level
    rope_dim, theta = 64, 1e7
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    B = 1024
    hi_ang = torch.outer(torch.arange(B, dtype=torch.float64) * B, inv)
    lo_ang = torch.outer(torch.arange(B, dtype=torch.float64), inv)
    ch, sh, cl, sl = [t.float() for t in (hi_ang.cos(), hi_ang.sin(), lo_ang.cos(), lo_ang.sin())]
    sample = torch.cat(
        [torch.arange(0, 4096), torch.randint(0, 2**20, (20000,)), torch.arange(2**20 - 64, 2**20)]
    )
    h, l = sample // B, sample % B
    c2 = ch[h] * cl[l] - sh[h] * sl[l]
    s2 = sh[h] * cl[l] + ch[h] * sl[l]
    ang = torch.outer(sample.double(), inv)
    ct, st = ang.cos(), ang.sin()
    c_host, s_host = rope_tables(sample, rope_dim, theta, interleave=False)
    c_host, s_host = c_host[:, :32].double(), s_host[:, :32].double()
    bf = lambda x: x.float().bfloat16().double()
    for name, c, s in (("two-level fp32", c2.double(), s2.double()), ("host fp32 rope_tables", c_host, s_host)):
        print(
            "%-22s max|cos err| fp32 %.2e  after bf16 %.2e   max|sin err| fp32 %.2e  after bf16 %.2e   (vs fp64 truth)"
            % (name, (c - ct).abs().max(), (bf(c) - ct).abs().max(), (s - st).abs().max(), (bf(s) - st).abs().max())
        )
    m = sample < 2048
    print(
        "positions < 2048: bf16 tables equal host path: cos %.4f%% sin %.4f%% of entries"
        % (100 * (bf(c2)[m] == bf(c_host)[m]).float().mean(), 100 * (bf(s2)[m] == bf(s_host)[m]).float().mean())
    )


if __name__ == "__main__":
    main()
