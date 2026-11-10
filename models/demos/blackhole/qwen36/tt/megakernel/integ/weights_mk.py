"""Load-time weight repack for the megakernel (DESIGN_integration.md §4 / §10, spec: repack source = the model's TT tensors).

The engine reads one DRAM tensor per weight KIND, device-distinct (ShardTensorToMesh), with layer L's pages at
L * pages_per_layer (the bench_mka2 layout: wgu_all / wd_all / wq_all). Its values are exactly the model's own decode
weights (GPTQ BFP4 / BFP8 as prod loads them), moved tile for tile, so the repack is lossless: a whole 32x32 tile keeps
its 16-value exponent blocks and re-quantizes to the same bytes. Only the optional γ·NORM_FIX fold
(QWEN36_R16_MK_GAMMA_FOLD) changes values.

Three layers of code, so each track owns its own piece:
  1. extraction (device -> host, per device): the per-chip matrices in a fixed, documented orientation
       mlp : (wg [H, I], wu [H, I], wd [I, H])     from w_gate_up (prepare_for_fused_swiglu gate/up tile pairs) + w2
       qkv : [H, 112*32] = [q pair 0-47 | k 48-55 | v 56-63 | gate pair 64-111]  from tw["wqkvg_pair"]
       wo  : [6*256, H]  pair heads x hd rows, same head order as qkv          from tw["wo_pair"]
  2. packing (pure torch, per device per layer): LAYOUTS[kind](mats, ctx) -> pages [n, 32, 32]. These call the tracks'
     layout functions (engine/layout_mlp.py, engine/layout_attn.py, later layout_gdn / ar_phase).
  3. assembly: per kind, concatenate the layers, quantize per device on host and aggregate into one mesh tensor.
  Plus kind 'gamma' (spec §19): the per-layer input / post-attention RMSNorm (1+w) gammas, replicated (gamma_pages).

Host memory: one device's pages of one kind at a time (MLP at BFP4 ~1.1 GB per device for 64 layers).
"""

import os

TILE = 32
H = 5120
QN = 112  # qkv(g) output tiles per chip (layout_attn.QN)


# ---------------------------------------------------------------------------------------------------------------- #
# 1. extraction (device -> host)
# ---------------------------------------------------------------------------------------------------------------- #
def device_shards(t):
    """Per-device host float32 copies of a mesh tensor, as 2-D [rows, cols] (leading unit dims dropped)."""
    import ttnn

    out = []
    for dt in ttnn.get_device_tensors(t):
        x = ttnn.to_torch(dt).float()
        out.append(x.reshape(-1, x.shape[-1]))
    return out


def ungroup_gate_up(gu):
    """One device's prepare_for_fused_swiglu block [H, 2*I] = [g_t0, u_t0, g_t1, u_t1, ...] (tile pairs) -> (wg, wu)."""
    rows, two_i = gu.shape
    t = gu.reshape(rows, two_i // (2 * TILE), 2, TILE)
    return t[:, :, 0, :].reshape(rows, -1), t[:, :, 1, :].reshape(rows, -1)


def mlp_mats(layer):
    w = layer.feed_forward.weights
    assert w.w_gate_up is not None, "megakernel repack needs the packed gate|up weight (prod dsmc path)"
    gus, wds = device_shards(w.w_gate_up), device_shards(w.w2)
    out = []
    for gu, wd in zip(gus, wds):
        wg, wu = ungroup_gate_up(gu)
        assert wg.shape[0] == H and wd.shape == (wg.shape[1], H), (wg.shape, wd.shape)
        out.append((wg, wu, wd))
    return out


def qkv_mats(layer):
    tw = layer.attention.tw
    assert "wqkvg_pair" in tw, "megakernel repack needs the KV-split pair weight (QWEN36_ATTN_KV_SPLIT=user)"
    out = device_shards(tw["wqkvg_pair"])
    assert all(m.shape == (H, QN * TILE) for m in out), [m.shape for m in out]
    return out


def wo_mats(layer):
    out = device_shards(layer.attention.tw["wo_pair"])
    assert all(m.shape[1] == H for m in out), [m.shape for m in out]
    return out


# kind -> (which layers carry it, extractor)
SOURCES = {
    "mlp": (lambda l: True, mlp_mats),
    "qkv": (lambda l: l.is_full_attention, qkv_mats),
    "wo": (lambda l: l.is_full_attention, wo_mats),
}


# ---------------------------------------------------------------------------------------------------------------- #
# 2. packing (pure torch): the tracks' layout functions
# ---------------------------------------------------------------------------------------------------------------- #
class MlpLayout:
    """v12 split-K MLP pages (engine/layout_mlp.py). ctx: core_xy (80 STREAM cores, STREAM index order), bank_loc."""

    dtype = "bfloat4_b"

    def __init__(self, core_xy, bank_loc, gamma_fold=None):
        from models.demos.blackhole.qwen36.tt.megakernel.engine import layout_mlp as lm

        self.lm = lm
        self.banks = len(bank_loc)
        self.bank_of, self.rank, self.nslot_b = lm.assign_banks(core_xy, bank_loc)
        self.pages_per_layer = lm.pages_per_layer(self.nslot_b, self.banks)
        self.gamma_fold = gamma_fold  # None, or (layer_idx -> (gamma [H], norm_fix))

    def pack(self, mats, layer_idx):
        wg, wu, wd = mats
        g = self.gamma_fold(layer_idx) if self.gamma_fold else (None, 1.0)
        return self.lm.pack_layer(wg, wu, wd, self.bank_of, self.rank, self.nslot_b, self.banks, gamma=g[0], norm_fix=g[1])


class QkvLayout:
    """Attention qkv(g) pages (engine/layout_attn.py; spec §18 moves it to split-K, layout_attn stays the source)."""

    dtype = "bfloat8_b"

    def __init__(self, plan, bank_loc, banks, gamma_fold=None):
        from models.demos.blackhole.qwen36.tt.megakernel.engine import layout_attn as la

        self.la = la
        self.roles = la.qkv_roles(plan)
        self.bank, self.off, self.pages_per_layer = la.bank_assign(self.roles, plan, bank_loc, banks)
        self.banks = banks
        self.gamma_fold = gamma_fold

    def pack(self, w, layer_idx):
        if self.gamma_fold:
            g, nf = self.gamma_fold(layer_idx)
            w = self.la.fold_gamma(w, g, nf)
        return self.la.pack_qkv_layer(w, self.roles, self.bank, self.off, self.banks, self.pages_per_layer)


class TileOrderLayout:
    """Test layout: the per-chip matrices' tiles in row-major order (mlp: wg, wu, wd), no reordering. Lets a device test
    read the repacked tensor back and compare it with the model's own weights (extraction + quantize round trip)."""

    def __init__(self, kind, dtype):
        self.kind, self.dtype = kind, dtype
        self.pages_per_layer = {"mlp": 3 * 160 * 68, "qkv": 160 * QN, "wo": 48 * 160}[kind]

    @staticmethod
    def tiles(m):
        r, c = m.shape[0] // TILE, m.shape[1] // TILE
        return m.reshape(r, TILE, c, TILE).permute(0, 2, 1, 3).reshape(-1, TILE, TILE)

    def pack(self, mats, layer_idx):
        import torch

        ms = mats if isinstance(mats, tuple) else (mats,)
        return torch.cat([self.tiles(m) for m in ms])


def norm_gamma(norm):
    """The full-width (1+w) RMSNorm gamma the DAR path uses (DecodeResidualAllReduce.rms_norm: ``norm.weight``), as a
    float [H] host vector (replicated, so device 0's copy)."""
    g = device_shards(norm.weight)[0].reshape(-1)
    if g.numel() == TILE * H:  # tile-padded [32, H]: row 0 holds gamma
        g = g.reshape(TILE, H)[0]
    assert g.numel() == H, g.shape
    return g


def gamma_pages(model):
    """kind 'gamma' (spec §19; the AR owner applies gamma at x delivery): per layer L and norm n (0 input_layernorm =
    attention_norm, 1 post_attention_layernorm = ffn_norm) one tile row [32, H] bf16 with gamma in row 0 (rows 1-31
    zero) -> page ((L * 2 + n) * 160 + chunk * 10 + t) = owner chunk ``chunk`` (16 x 10 tiles), tile t. Identical on
    every device (each device reads its own owner chunks)."""
    import torch

    rows = []
    for l in model.layers:
        for norm in (l.attention_norm.norm, l.ffn_norm.norm):
            r = torch.zeros(TILE, H)
            r[0] = norm_gamma(norm)
            rows.append(r)
    return torch.cat(rows).reshape(1, 1, -1, H)


# ---------------------------------------------------------------------------------------------------------------- #
# 3. assembly
# ---------------------------------------------------------------------------------------------------------------- #
def gamma_fold_enabled():
    return os.environ.get("QWEN36_R16_MK_GAMMA_FOLD", "0") == "1"


def repack(model, layouts=None, kinds=None, only_layers=None):
    """-> {kind: dict(tensor=mesh tensor, pages_per_layer=int, layers=[model layer indices in page order])}.

    ``layouts``: kind -> layout object with .pack(mats, layer_idx) -> torch [pages, 32, 32], .pages_per_layer, .dtype.
    The engine builder (engine/program.py, M3) constructs them from its grid plan and the device's bank map. With no
    layouts there is nothing to pack (the identity engine): NotImplementedError, which Stage1Verify treats as "no
    weights needed"."""
    if not layouts:
        raise NotImplementedError("megakernel weight repack: no layouts given (identity engine needs none)")
    import torch
    import ttnn

    mesh = model.mesh_device
    out = {}
    if "gamma" in (kinds or layouts):  # replicated, bf16, its own fixed layout (gamma_pages)
        t = ttnn.from_torch(gamma_pages(model).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        out["gamma"] = dict(tensor=t, pages_per_layer=2 * (H // TILE), layers=list(range(len(model.layers))),
                            dtype="bfloat16")
    for kind in [k for k in (kinds or sorted(layouts)) if k != "gamma"]:
        lay = layouts[kind]
        want, extract = SOURCES[kind]
        idx = [i for i, l in enumerate(model.layers) if want(l) and (only_layers is None or i in only_layers)]
        per_dev = None  # device -> list of per-layer page tensors (host bf16)
        for i in idx:
            mats = extract(model.layers[i])
            if per_dev is None:
                per_dev = [[] for _ in mats]
            for d, m in enumerate(mats):
                p = lay.pack(m, i)
                assert p.shape[0] == lay.pages_per_layer, (kind, p.shape, lay.pages_per_layer)
                per_dev[d].append(p.to(torch.bfloat16))
        dtype = getattr(ttnn, lay.dtype)
        shards = []
        for d in range(len(per_dev)):  # quantize one device at a time on host
            pages = torch.cat(per_dev[d]).reshape(1, 1, -1, TILE)
            per_dev[d] = None
            shards.append(ttnn.from_torch(pages, dtype=dtype, layout=ttnn.TILE_LAYOUT))
        t = ttnn.to_device(ttnn.from_host_shards(shards, mesh.shape), mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out[kind] = dict(tensor=t, pages_per_layer=lay.pages_per_layer, layers=idx, dtype=lay.dtype)
    return out
