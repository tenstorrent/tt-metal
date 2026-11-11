"""Lossless engine weight layouts, attention track: qkv split-K (spec §18), the wo unit (owners) and gdn_out (the wo unit
on the GDN role, G-c).

Inputs are the per-device, per-layer matrices in integ/weights_mk.py's extraction orientation:
    qkv     [H, 112*32]      weights_mk.qkv_mats()     [q pair 0-47 | k 48-55 | v 56-63 | gate pair 64-111]
    wo      [48*32, 160*32]  weights_mk.wo_mats()      [6*256, H]: pair heads x hd rows (= A columns), H columns
    gdn_out [24*32, 160*32]  weights_mk.gdn_out_mats() [local value dim, H]
Packing only moves whole 32x32 tiles, so a BFP8 / BFP4 tile keeps its 16-value exponent blocks: quantizing the packed
pages gives the same bytes as the model's tiles (test_weight_layouts.py checks it through ttnn's host quantizer).

Each layout follows integ's repack protocol: .pack(mats, layer_idx) -> torch [pages_per_layer, 32, 32], .pages_per_layer,
.dtype; plus .unpack(pages) -> the matrix (the inverse; program.py builds its golds from it) and the placement fields the
engine's runtime args use. pack_qkv_splitk / pack_wo_unit / pack_gdn_out concatenate a list of per-layer matrices (layer
L at L * pages_per_layer) and optionally build the DRAM tensor program.py binds (wq_all / wo_all / wg_all).
"""

TILE = 32
QN = 112          # qkv(g) output tiles per chip
WN = 160          # H / 32: wo / gdn_out output tiles
CHUNK = 10        # output tiles per owner chunk (owner j holds tiles 10j .. 10j + 9)
N_CHUNKS = 16


def _layout_attn():
    try:
        from models.demos.blackhole.qwen36.tt.megakernel.engine import layout_attn as la
    except ImportError:   # bench path: engine/ is on sys.path
        import layout_attn as la
    return la


def _tile(m, r, c):
    return m[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]


def rot_order(n_chunks=N_CHUNKS):
    """Owner chunk order for device 0: owning device o = j // 2, farthest around the 8-ring first, own (o = 0) last."""
    return sorted(range(n_chunks), key=lambda j: (-min(j // 2, 8 - j // 2), j))


# ------------------------------------------------------------------------------------------------------------------- #
# qkv split-K (QSK)
# ------------------------------------------------------------------------------------------------------------------- #
class QkvSplitKLayout:
    """80 STREAM slices (column c = logical x, K-row k = logical y): 20 K tiles x the column's N-list (12, 12, 11 x 8),
    k-major pages, byte-balanced nearest-bank placement (layout_attn.splitk_bank_assign). Fields for the runtime args:
    slices[(c, k)], bank[(c, k)], off[(c, k)] (bank-local page offset), pages_per_layer (PQL)."""

    dtype = "bfloat8_b"

    def __init__(self, bank_loc, banks, gamma_fold=None):
        la = _layout_attn()
        self.la, self.banks, self.gamma_fold = la, banks, gamma_fold
        self.slices = la.qkv_splitk_slices()
        self.bank, self.off, self.pages_per_layer = la.splitk_bank_assign(self.slices, {k: k for k in self.slices},
                                                                          bank_loc, banks)

    def page(self, key, kl, j):
        sl = self.slices[key]
        return (self.off[key] + sl.page_of(kl, j)) * self.banks + self.bank[key]

    def pack(self, w, layer_idx=None, scale=1.0):
        if self.gamma_fold is not None:   # optional (spec §19: off by default, γ is applied by the AR owner)
            g, nf = self.gamma_fold(layer_idx)
            w = self.la.fold_gamma(w, g, nf)
        assert tuple(w.shape) == (self.la.H, QN * TILE), w.shape
        return self.la.pack_qkv_splitk_layer(w, self.slices, self.bank, self.off, self.banks, self.pages_per_layer, scale)

    def unpack(self, pg):
        """[..., pages_per_layer, 32, 32] -> [..., H, QN*32] (leading dims kept, e.g. NL)."""
        import torch

        w = torch.zeros(*pg.shape[:-3], self.la.H, QN * TILE, dtype=pg.dtype)
        for key, sl in self.slices.items():
            for kl in range(sl.pages // sl.nt):
                for j in range(sl.nt):
                    k, n = sl.k0 + kl, sl.first + j
                    w[..., k * TILE:(k + 1) * TILE, n * TILE:(n + 1) * TILE] = pg[..., self.page(key, kl, j), :, :]
        return w


# ------------------------------------------------------------------------------------------------------------------- #
# wo unit (owners) / gdn_out (GDN role)
# ------------------------------------------------------------------------------------------------------------------- #
class WoUnitLayout:
    """Split-N wo unit slices. cores = [(core id, logical x, logical y)] in the engine's core order (wo: program.py's
    ext_list / the 16 owners; gdn_out: GDNC, the 24 GDN cores row-major). Core c computes output tiles
    tiles[c] = flat[c::len(cores)] with flat = the owner chunks in rot order; its slice is wk x nt pages, k-major
    (page kl * nt + jj), in ONE bank (nearest bank first, byte-balanced caps), rows packed per bank in core order.
    Fields: tiles[c], bank[c], row[c] (bank-local page offset), base_page(c) = row * banks + bank, pages_per_layer."""

    def __init__(self, cores, wk, bank_loc, banks, dtype="bfloat8_b", maxt=None, wn=WN):
        self.cores, self.wk, self.wn, self.banks, self.dtype = list(cores), wk, wn, banks, dtype
        n = len(self.cores)
        flat = [t for j in rot_order() for t in range(CHUNK * j, CHUNK * j + CHUNK)]
        assert len(flat) == wn
        self.tiles = [flat[c::n] for c in range(n)]
        if maxt is not None:
            assert max(len(t) for t in self.tiles) <= maxt, (max(len(t) for t in self.tiles), maxt)
        caps = [n // banks + (1 if b < n % banks else 0) for b in range(banks)]
        # same tie order as program.py: (distance, core id, bank)
        pairs = sorted((abs(x - bx) + abs(y - by), cid, c, b) for c, (cid, x, y) in enumerate(self.cores)
                       for b, (bx, by) in enumerate(bank_loc))
        self.bank = [None] * n
        for _, _, c, b in pairs:
            if self.bank[c] is not None or caps[b] == 0:
                continue
            self.bank[c] = b
            caps[b] -= 1
        tot = [0] * banks
        self.row = [0] * n
        for c in range(n):
            self.row[c] = tot[self.bank[c]]
            tot[self.bank[c]] += len(self.tiles[c]) * wk
        self.pages_per_layer = max(tot) * banks

    def base_page(self, c):
        return self.row[c] * self.banks + self.bank[c]

    def page(self, c, kl, jj):
        return (self.row[c] + kl * len(self.tiles[c]) + jj) * self.banks + self.bank[c]

    def pack(self, w, layer_idx=None, scale=1.0):
        import torch

        assert tuple(w.shape) == (self.wk * TILE, self.wn * TILE), (w.shape, self.wk, self.wn)
        pg = torch.zeros(self.pages_per_layer, TILE, TILE, dtype=w.dtype)
        for c, ts in enumerate(self.tiles):
            for kl in range(self.wk):
                for jj, t in enumerate(ts):
                    pg[self.page(c, kl, jj)] = _tile(w, kl, t) * scale
        return pg

    def unpack(self, pg):
        """[..., pages_per_layer, 32, 32] -> [..., wk*32, wn*32]."""
        import torch

        w = torch.zeros(*pg.shape[:-3], self.wk * TILE, self.wn * TILE, dtype=pg.dtype)
        for c, ts in enumerate(self.tiles):
            for kl in range(self.wk):
                for jj, t in enumerate(ts):
                    w[..., kl * TILE:(kl + 1) * TILE, t * TILE:(t + 1) * TILE] = pg[..., self.page(c, kl, jj), :, :]
        return w


def wo_unit_layout(cores, bank_loc, banks, maxt=10):
    """Attention wo on the owners: WK = 48 (6 pair heads x 256 / 32), BFP8."""
    return WoUnitLayout(cores, 48, bank_loc, banks, dtype="bfloat8_b", maxt=maxt)


def gdn_out_layout(cores, bank_loc, banks, maxt=7):
    """GDN out_proj on the 24 GDN cores (gdn_out fork's layout, -mk-gdnout @d30f0b5ae8a): WK = 24 (local value dim 768),
    BFP4 like prod's out-proj."""
    return WoUnitLayout(cores, 24, bank_loc, banks, dtype="bfloat4_b", maxt=maxt)


def engine_cores(env=None, grid=(12, 10), stream_rect=(10, 8)):
    """The engine's core lists as a pure function of its env (so integ can build layouts before setup_bench):
    dict(wo=[(fi, x, y)] = ext_list (the non-STREAM, non-reserved cores in fi order; GDN24: the 16 owners x 10-11, rows
    0-7), gdn_out=[(fi, x, y)] = GDNC (GDN24: rows 8-9 row-major; else empty)). fi = y * grid_x + x (program.py fcores).
    setup_bench asserts equality with what it computes."""
    import os

    env = os.environ if env is None else env
    gxf, gyf = grid
    gx, gy = stream_rect
    gdn24 = env.get("MKA_GDN24", "0") == "1"
    arc = env.get("MKA_ARCORES", "106,107,118,119")
    resv = set() if (gdn24 or arc in ("", "none")) else set(int(v) for v in arc.split(","))
    gdnc = [fi for fi in range(gxf * gyf) if gdn24 and fi // gxf >= gy]
    resv |= set(gdnc)
    xy = lambda fi: (fi, fi % gxf, fi // gxf)
    ext = [fi for fi in range(gxf * gyf) if not (fi % gxf < gx and fi // gxf < gy) and fi not in resv]
    return dict(wo=[xy(fi) for fi in ext], gdn_out=[xy(fi) for fi in gdnc])


# ------------------------------------------------------------------------------------------------------------------- #
# whole-pass packers (layer L at L * pages_per_layer)
# ------------------------------------------------------------------------------------------------------------------- #
def pack_layers(ws, layout, scales=None, layer_ids=None):
    """List of per-layer matrices -> torch [len(ws) * pages_per_layer, 32, 32]."""
    import torch

    return torch.cat([layout.pack(w, None if layer_ids is None else layer_ids[i], 1.0 if scales is None else scales[i])
                      for i, w in enumerate(ws)])


def to_device(pages, layout, dev):
    """The DRAM-interleaved tensor program.py binds ([1, 1, pages*32, 32], layout.dtype)."""
    import torch
    import ttnn

    return ttnn.from_torch(pages.reshape(1, 1, -1, TILE).to(torch.bfloat16), dtype=getattr(ttnn, layout.dtype),
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def pack_qkv_splitk(ws, layout, scales=None, dev=None):
    pg = pack_layers(ws, layout, scales)
    return pg if dev is None else to_device(pg, layout, dev)


def pack_wo_unit(ws, layout, scales=None, dev=None):
    pg = pack_layers(ws, layout, scales)
    return pg if dev is None else to_device(pg, layout, dev)


def pack_gdn_out(ws, layout, scales=None, dev=None):
    pg = pack_layers(ws, layout, scales)
    return pg if dev is None else to_device(pg, layout, dev)


def roundtrip(layout, w, quantize=True):
    """Lossless check: unpack(pack(w)) == w, and (quantize) the packed pages quantized on host by ttnn give exactly the
    tiles of w quantized whole (same BFP blocks). Returns (exact_fp, exact_quantized)."""
    import torch

    pg = layout.pack(w)
    ok_fp = torch.equal(layout.unpack(pg), w)
    if not quantize:
        return ok_fp, None
    import ttnn

    dt = getattr(ttnn, layout.dtype)
    q = lambda t: ttnn.to_torch(ttnn.from_torch(t.reshape(1, 1, -1, t.shape[-1]).to(torch.bfloat16), dtype=dt,
                                                layout=ttnn.TILE_LAYOUT)).float().reshape(t.shape)
    return ok_fp, torch.equal(layout.unpack(q(pg)), q(w))


# ------------------------------------------------------------------------------------------------------------------- #
# gdn_in (G-a, MK_GDN_INPROJ, v12 unit): the GDN in-projection through the same STREAM split-K as qkv
# ------------------------------------------------------------------------------------------------------------------- #
GIN_TILES, GIN_COLS = 65, 2060   # per device: tw["qkvz"] = fused [q k v z (a b)], 2060 columns = 64 tiles + 12


def gdn_in_dense(w):
    """One device's gdn_in [H, 2060] (weights_mk.gdn_in_mats, or tile-padded [H, 2080]) -> float [H, 2080], pad = 0."""
    import torch

    w = w.float().reshape(-1, w.shape[-1])
    assert w.shape[1] in (GIN_COLS, GIN_TILES * TILE), w.shape
    if w.shape[1] == GIN_COLS:
        w = torch.nn.functional.pad(w, (0, GIN_TILES * TILE - GIN_COLS))
    assert not w[:, GIN_COLS:].any(), "gdn_in pad columns must be zero"
    return w


class GdnInSplitKLayout(QkvSplitKLayout):
    """gdn_in split-K slices over the GX STREAM columns: 65 tiles (7 x 5 + 6 x 5 at GX = 10), 20 K tiles per K-row,
    k-major pages, nearest-bank placement; BFP4 (CB 15 on the GDN-type layers). No γ fold (x arrives γ-scaled from the
    AR owner, spec §19). The owning K-row r < nt of column c writes output tile first(c) + r to GDN core (c, 8 + r % 2),
    slot r // 2 (MK_GDN_INPROJ writer)."""

    dtype = "bfloat4_b"

    def __init__(self, bank_loc, banks, gx=10):
        la = _layout_attn()
        self.la, self.banks, self.gamma_fold = la, banks, None
        self.slices = la.qkv_splitk_slices(qn=GIN_TILES, cols=gx)
        self.bank, self.off, self.pages_per_layer = la.splitk_bank_assign(self.slices, {k: k for k in self.slices},
                                                                          bank_loc, banks)

    def pack(self, w, layer_idx=None, scale=1.0):
        w = gdn_in_dense(w)
        return self.la.pack_qkv_splitk_layer(w, self.slices, self.bank, self.off, self.banks, self.pages_per_layer, scale)

    def unpack(self, pg):
        """[..., pages_per_layer, 32, 32] -> [..., H, 2080]."""
        import torch

        w = torch.zeros(*pg.shape[:-3], self.la.H, GIN_TILES * TILE, dtype=pg.dtype)
        for key, sl in self.slices.items():
            for kl in range(sl.pages // sl.nt):
                for j in range(sl.nt):
                    k, n = sl.k0 + kl, sl.first + j
                    w[..., k * TILE:(k + 1) * TILE, n * TILE:(n + 1) * TILE] = pg[..., self.page(key, kl, j), :, :]
        return w

    def core_fields(self, col, krow):
        """STREAM core (col, krow): (first page, column nt, owns a tile, owned tile index first + krow)."""
        sl = self.slices[(col, krow)]
        return self.off[(col, krow)] * self.banks + self.bank[(col, krow)], sl.nt, int(krow < sl.nt), sl.first + krow


def pack_gdn_in(ws, layout, scales=None, dev=None):
    """Per-GDN-layer gdn_in matrices (GDN ordinal order: layer g at g * pages_per_layer) -> pages / the wg_all tensor."""
    pg = pack_layers(ws, layout, scales)
    return pg if dev is None else to_device(pg, layout, dev)
