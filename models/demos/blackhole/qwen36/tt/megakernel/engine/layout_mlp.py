"""v12 (fm12) split-K MLP weight layout for one chip of TP8 (spec v0.6-v0.9; -mk-v12 V12_MLP_RESULTS.md).

STREAM core (n, k) of the 10 x 8 rect (logical x = n, y = k) owns, per layer, ONE contiguous BFP4 slice of SLOT = 420
tiles in its nearest DRAM bank (10 cores per bank):
    gate [20 K-tiles of row k][nn h-tiles of column n] | up [20][nn] | down [4 sub-blocks][nn][5 out tiles]
with nn = 7 (n < 8) or 6 (n = 8, 9) (68 h tiles = 7 x 8 + 6 x 2), K-slice k = x tiles 20k..20k+19, down out-slice k =
out tiles 20k..20k+19 (sub-block b = tiles 20k + 5b .. +4).
Optional γ / NORM_FIX fold (post-attention RMSNorm weight) into gate and up rows: W' = diag(γ · norm_fix) W.

Pure torch (no device): returns host page tensors in the interleaved-DRAM page order (page = slot_tile * banks + bank)
plus the runtime-arg fields the engine kernels need."""
import math, torch

TILE = 32
GX, GY, KTR = 10, 8, 20
NN = [7] * 8 + [6] * 2
HB = [sum(NN[:n]) for n in range(GX)]
NNM = 7
SLOT = 3 * KTR * NNM   # 420


def assign_banks(core_xy, bank_loc):
    """core_xy: list of logical (x, y) of the 80 STREAM cores in STREAM index order (i = n + 10 k); bank_loc: per bank,
    the logical (x, y) of its preferred worker. Nearest bank first, 80 / banks cores per bank.
    Returns (bank_of[i], rank_in_bank[i], slots_per_bank)."""
    banks = len(bank_loc)
    caps = [len(core_xy) // banks + (1 if b < len(core_xy) % banks else 0) for b in range(banks)]
    pairs = sorted((abs(x - bx) + abs(y - by), i, b) for i, (x, y) in enumerate(core_xy) for b, (bx, by) in enumerate(bank_loc))
    bank_of = {}
    for d, i, b in pairs:
        if i in bank_of or caps[b] == 0: continue
        bank_of[i] = b; caps[b] -= 1
    rank = {}
    for b in range(banks):
        for r, i in enumerate([i for i in range(len(core_xy)) if bank_of[i] == b]): rank[i] = r
    return bank_of, rank, max(rank.values()) + 1


def tiles(w):
    R, C = w.shape[0] // TILE, w.shape[1] // TILE
    return w.reshape(R, TILE, C, TILE).permute(0, 2, 1, 3)


def pack_layer(wg, wu, wd, bank_of, rank, nslot_b, banks, gamma=None, norm_fix=1.0):
    """wg, wu: [H, I], wd: [I, H] (float, already the values to place, e.g. BFP4-dequantized). Returns a bf16 page
    tensor [nslot_b * SLOT * banks, 32, 32] (DRAM page order) for one layer."""
    if gamma is not None:
        s = (gamma.float() * norm_fix)[:, None]
        wg, wu = wg * s, wu * s
    tg, tu, td = tiles(wg), tiles(wu), tiles(wd)
    P = torch.zeros(nslot_b * SLOT, banks, TILE, TILE, dtype=torch.bfloat16)
    for i in range(GX * GY):
        n, k = i % GX, i // GX
        nn, hb = NN[n], HB[n]
        g = tg[k * KTR:(k + 1) * KTR, hb:hb + nn].reshape(-1, TILE, TILE)
        u = tu[k * KTR:(k + 1) * KTR, hb:hb + nn].reshape(-1, TILE, TILE)
        d = td[hb:hb + nn, k * KTR:(k + 1) * KTR].reshape(nn, 4, 5, TILE, TILE).permute(1, 0, 2, 3, 4).reshape(-1, TILE, TILE)
        sl = torch.cat([g, u, d]); s0 = rank[i] * SLOT
        P[s0:s0 + sl.shape[0], bank_of[i]] = sl.bfloat16()
    return P.reshape(-1, TILE, TILE)


def core_fields(i, bank_of, rank, banks):
    """Runtime-arg fields for STREAM core i: (first page of its slice, tiles in its slice, nn)."""
    n = i % GX
    return rank[i] * SLOT * banks + bank_of[i], 3 * KTR * NN[n], NN[n]


def pages_per_layer(nslot_b, banks):
    return nslot_b * SLOT * banks
