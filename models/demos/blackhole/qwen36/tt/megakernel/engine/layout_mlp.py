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


def assign_banks_balanced(core_xy, bank_loc):
    """MK_V12_NOCBAL (MK80, DESIGN_mk70 §9.1): like assign_banks (nearest first, 80 / banks cores per bank) but every bank
    also gets the same number of nn = 6 columns' cores (16 / 8 = 2), so every bank streams the same bytes per layer
    (8 x 420 + 2 x 360 tiles). Same return value as assign_banks."""
    banks = len(bank_loc)
    small = [i for i, (x, y) in enumerate(core_xy) if NN[x] < NNM]
    n = len(core_xy)
    caps = {False: [(n - len(small)) // banks + (1 if b < (n - len(small)) % banks else 0) for b in range(banks)],
            True: [len(small) // banks + (1 if b < len(small) % banks else 0) for b in range(banks)]}
    pairs = sorted((abs(x - bx) + abs(y - by), i, b) for i, (x, y) in enumerate(core_xy) for b, (bx, by) in enumerate(bank_loc))
    bank_of, sm = {}, set(small)
    for d, i, b in pairs:
        if i in bank_of or caps[i in sm][b] == 0: continue
        bank_of[i] = b; caps[i in sm][b] -= 1
    rank = {}
    for b in range(banks):
        for r, i in enumerate([i for i in range(len(core_xy)) if bank_of[i] == b]): rank[i] = r
    return bank_of, rank, max(rank.values()) + 1


def bank_tiles(bank_of, banks):
    """tiles streamed per bank per layer (bytes balance check)"""
    t = [0] * banks
    for i, b in bank_of.items(): t[b] += 3 * KTR * NN[i % GX]
    return t


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


# ---------------------------------------------------------------------------------------------------------------------
# HF checkpoint -> per-device v12 MLP weights (TP8 column/row parallel exactly as prod tt/mlp.py: gate/up column-parallel
# on the intermediate dim, down row-parallel on it; ShardTensorToMesh gives device d the contiguous slice d).
H_DIM = 5120
# AR contract (spec §12 / AR_INTERFACE.md): rstd' = rsqrt(Σ_j P_j + 1.25 eps), P_j = Σ x² · 2^-12, so
# rstd' = rstd / NORM_FIX with NORM_FIX = sqrt(H / 4096) = sqrt(1.25); fold γ · NORM_FIX into gate and up rows.
NORM_FIX = math.sqrt(H_DIM / 4096)


def from_hf(state_dict, layer, dev, tp=8, prefix="model.layers.{L}.", fold_norm=False):
    """REFERENCE / TEST path (prod is GPTQ BFP4: use from_tt). Return (wg [H, I], wu [H, I], wd [I, H]) float32 for
    device `dev` of layer `layer` (I = I_total / tp). fold_norm (default OFF, spec §19: the AR owner applies
    γ·NORM_FIX to x at delivery) folds the zero-centered post-attention RMSNorm (1 + w) x NORM_FIX into gate / up."""
    p = prefix.format(L=layer)
    g = state_dict[p + "mlp.gate_proj.weight"].float()   # [I_total, H]
    u = state_dict[p + "mlp.up_proj.weight"].float()
    d = state_dict[p + "mlp.down_proj.weight"].float()   # [H, I_total]
    It = g.shape[0]; assert It % tp == 0 and g.shape[1] == H_DIM, g.shape
    I = It // tp; sl = slice(dev * I, (dev + 1) * I)
    wg, wu, wd = g[sl].T.contiguous(), u[sl].T.contiguous(), d[:, sl].T.contiguous()
    if fold_norm:
        gam = (1.0 + state_dict[p + "post_attention_layernorm.weight"].float()) * NORM_FIX   # zero-centered RMSNorm
        wg, wu = wg * gam[:, None], wu * gam[:, None]
    return wg, wu, wd


def gu_split(gu_dev):
    """Prod per-device w_gate_up [H, 2I] (prepare_for_fused_swiglu(gate_is_first=True): tile-pair interleaved
    [g_t0, u_t0, g_t1, u_t1, ...] within the device block) -> (wg [H, I], wu [H, I])."""
    Hh, two = gu_dev.shape
    I = two // 2
    t = gu_dev.reshape(Hh, I // TILE, 2, TILE)
    return t[:, :, 0].reshape(Hh, I), t[:, :, 1].reshape(Hh, I)


def from_tt(w_gate_up, w2, dev=None):
    """DEFAULT path (spec §19): LOSSLESS tile-granular repack of the model's own TT weights for one device.
    w_gate_up: prod MLPWeights.w_gate_up, w2: MLPWeights.w2 (GPTQ BFP4). Each is either the device's host tensor
    (torch [H, 2I] / [I, H], e.g. ttnn.to_torch(ttnn.get_device_tensors(t)[dev])) or a ttnn mesh tensor + `dev`.
    BFP4 -> bf16 is exact and every v12 tile is a whole source tile (BFP4 exponents are shared within a tile's
    16-element face rows), so re-packing as BFP4 reproduces the source blocks bit for bit (check: bfp4_roundtrip_exact).
    No γ fold: the MLP consumes x already scaled by γ·NORM_FIX at AR delivery. Returns (wg [H, I], wu [H, I], wd [I, H])."""
    def host(t):
        if isinstance(t, torch.Tensor): return t
        import ttnn
        return ttnn.to_torch(ttnn.get_device_tensors(t)[dev])
    gu, d = host(w_gate_up).float(), host(w2).float()
    gu = gu.reshape(gu.shape[-2], gu.shape[-1]); d = d.reshape(d.shape[-2], d.shape[-1])
    wg, wu = gu_split(gu)
    assert d.shape == (wg.shape[1], wg.shape[0]), (d.shape, wg.shape)
    return wg, wu, d


def bfp4_roundtrip_exact(pages_bf16):
    """True if packing the bf16 pages as BFP4 reproduces them exactly (lossless repack check, needs ttnn)."""
    import ttnn
    t = ttnn.from_torch(pages_bf16.reshape(1, 1, -1, TILE), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    return torch.equal(ttnn.to_torch(t).float().reshape(pages_bf16.shape), pages_bf16.float())


def build_device_pages(layers_w, bank_loc, quant=None, nocbal=False):
    """layers_w: list over layers of (wg, wu, wd) for ONE device. Returns (pages bf16 [NL * PVL, 32, 32] in DRAM page
    order, fields) with fields = dict(pvl, slot, core -> (first_page, ntiles, nn)). `quant(w)` optionally maps a weight
    to its device-dequantized values (e.g. BFP4 round trip) before packing; packing itself never changes values.
    nocbal=True: the MK_V12_NOCBAL page placement (assign_banks_balanced); must match the engine build's flag."""
    banks = len(bank_loc)
    core_xy = [(i % GX, i // GX) for i in range(GX * GY)]
    bank_of, rank, nsb = (assign_banks_balanced if nocbal else assign_banks)(core_xy, bank_loc)
    pages = []
    for wg, wu, wd in layers_w:
        if quant is not None: wg, wu, wd = quant(wg), quant(wu), quant(wd)
        pages.append(pack_layer(wg, wu, wd, bank_of, rank, nsb, banks))
    fields = {i: core_fields(i, bank_of, rank, banks) for i in range(GX * GY)}
    return torch.cat(pages), dict(pvl=pages_per_layer(nsb, banks), slot=SLOT, cores=fields, bank_of=bank_of, rank=rank)


def unpack_check(pages, fields, banks, wg, wu, wd, layer=0):
    """Host self-check: rebuild the dense [gate|up|down] of one layer from the page tensor and compare exactly."""
    P = pages.reshape(-1, TILE, TILE)[layer * fields["pvl"]:(layer + 1) * fields["pvl"]].float()
    tg, tu, td = tiles(wg), tiles(wu), tiles(wd)
    for i in range(GX * GY):
        n, k = i % GX, i // GX; nn, hb = NN[n], HB[n]
        fp, nt, _ = fields["cores"][i]
        sl = torch.stack([P[fp + t * banks] for t in range(nt)])
        g = sl[:KTR * nn].reshape(KTR, nn, TILE, TILE); u = sl[KTR * nn:2 * KTR * nn].reshape(KTR, nn, TILE, TILE)
        d = sl[2 * KTR * nn:].reshape(4, nn, 5, TILE, TILE).permute(1, 0, 2, 3, 4).reshape(nn, KTR, TILE, TILE)
        assert torch.equal(g, tg[k * KTR:(k + 1) * KTR, hb:hb + nn].bfloat16().float()), (i, "gate")
        assert torch.equal(u, tu[k * KTR:(k + 1) * KTR, hb:hb + nn].bfloat16().float()), (i, "up")
        assert torch.equal(d, td[hb:hb + nn, k * KTR:(k + 1) * KTR].bfloat16().float()), (i, "down")
    return True


if __name__ == "__main__":   # host-only self-test with a synthetic checkpoint (no device)
    torch.manual_seed(0)
    It, tp = 2176 * 8, 8
    sd = {"model.layers.0.mlp.gate_proj.weight": torch.randn(It, H_DIM), "model.layers.0.mlp.up_proj.weight": torch.randn(It, H_DIM),
          "model.layers.0.mlp.down_proj.weight": torch.randn(H_DIM, It), "model.layers.0.post_attention_layernorm.weight": 0.1 * torch.randn(H_DIM)}
    import ttnn
    q4 = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
    wg, wu, wd = from_hf(sd, 0, dev=3, tp=tp)
    bank_loc = [(0, 0), (0, 4), (4, 0), (4, 4), (8, 0), (8, 4), (9, 2), (2, 6)]
    # from_tt path on a prod-style BFP4 device layout: [gate|up] tile-pair interleaved + w2
    wgq, wuq, wdq = q4(wg), q4(wu), q4(wd)
    gu = torch.stack([wgq.reshape(H_DIM, -1, TILE), wuq.reshape(H_DIM, -1, TILE)], dim=2).reshape(H_DIM, -1)
    tg, tu, td = from_tt(gu, wdq)
    assert torch.equal(tg, wgq) and torch.equal(tu, wuq) and torch.equal(td, wdq)
    pages, f = build_device_pages([(tg, tu, td)], bank_loc)
    print("pages", tuple(pages.shape), "pvl", f["pvl"], "unpack", unpack_check(pages, f, len(bank_loc), tg, tu, td),
          "bfp4 lossless", bfp4_roundtrip_exact(pages))
