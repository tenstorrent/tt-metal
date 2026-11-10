"""Attention-block layouts for the megakernel engine (attention track, mixed3).

Pure planning + packing (no device calls), shared by attn/bench_mka2.py and the integration builder (engine/build.py).

Covers, per chip (TP8, one P150, 12x10 full grid; logical coords):
  * core roles: STREAM rectangle (x < 10, y < 8), STREAM_EXT (the other cores), AR-only link senders (spec v0.9.1);
  * the real qkv PROJ (MKA_RQKV, spec v0.4 deferred norm + v0.5 STREAM_EXT): 112 output tiles (q 0-47, k 48-55,
    v 56-63, gate 64-111), STREAM_EXT cores 2 tiles at full K (348 KB ring), the rest as STREAM K-half pairs
    (87 KB, owner + partner), byte-balanced bank-local slices, optional input-norm gamma fold;
  * the SDPA KV split (own users x contiguous block ranges per core) and the 2-level pull tree;
  * the KV cache descriptor.
wo is still the stand-in (real wo -> CB_XO per AR_INTERFACE.md section 1 is the next attention-track step).
"""

from dataclasses import dataclass, field

TILE = 32
H = 5120
KT = H // TILE           # 160 K tiles of x
QN = 112                 # qkv(g) output tiles per chip: q 0-47, k 48-55, v 56-63, gate 64-111
DH = 8                   # head_dim 256 = 8 tiles
STREAM_W, STREAM_H = 10, 8
AR_SENDER_CORES = ((10, 8), (11, 8), (10, 9), (11, 9))   # AR_INTERFACE.md section 0: fwd/bwd x link 0/1

QR_NONE, QR_OWNER, QR_PARTNER, QR_EXT = 0, 1, 2, 3


@dataclass
class GridPlan:
    gx: int
    gy: int
    stream: dict                     # fi -> STREAM index (mk1 numbering x + 10 y)
    ar_only: set                     # fi of AR-only cores (no MKA2 kernel)
    active: list = field(default_factory=list)   # fi running MKA2 kernels

    def fi(self, x, y):
        return y * self.gx + x

    def xy(self, fi):
        return fi % self.gx, fi // self.gx

    @property
    def ext(self):
        return [f for f in self.active if f not in self.stream]


def grid_plan(gx=12, gy=10, ar_only_xy=()):
    stream = {}
    for fi in range(gx * gy):
        x, y = fi % gx, fi // gx
        if x < STREAM_W and y < STREAM_H:
            stream[fi] = x + STREAM_W * y
    ar = {y * gx + x for x, y in ar_only_xy}
    return GridPlan(gx, gy, stream, ar, [f for f in range(gx * gy) if f not in ar])


@dataclass(frozen=True)
class QRole:
    role: int      # QR_*
    n0: int        # first output tile
    nt: int        # output tiles
    k0: int        # first K tile
    nk: int        # K tiles
    peer: int      # owner <-> partner fi (self for EXT)


def qkv_roles(plan, qn=QN, kt=KT):
    """STREAM_EXT cores: 2 tiles each at full K; the remaining tiles as K-half pairs on STREAM cores (index 2i owner,
    2i + 1 partner: neighbours in x)."""
    ext = plan.ext
    s_by_i = {i: f for f, i in plan.stream.items()}
    n_ext = 2 * len(ext)
    n_half = qn - n_ext
    if n_half < 0 or 2 * n_half > len(plan.stream):
        raise ValueError(f"qkv split: {len(ext)} ext cores x 2 tiles + {n_half} K-half pairs does not fit")
    roles = {}
    for e, f in enumerate(ext):
        roles[f] = QRole(QR_EXT, 2 * e, 2, 0, kt, f)
    for i in range(2 * n_half):
        f, h = s_by_i[i], i % 2
        roles[f] = QRole(QR_OWNER if h == 0 else QR_PARTNER, n_ext + i // 2, 1, h * (kt // 2), kt // 2,
                         s_by_i[i + 1] if h == 0 else s_by_i[i - 1])
    return roles


def qkv_producers(roles):
    """Cores that write qkv output tiles (and bump MIXDONE on the Q root): owners + ext."""
    return sum(1 for r in roles.values() if r.role in (QR_OWNER, QR_EXT))


def bank_assign(roles, plan, bank_loc, banks):
    """Byte-balanced bank-local placement: each slice class (ext 2 x K tiles, STREAM half K tiles) is spread evenly
    over the banks, nearest bank first within the class. A count-only cap put most ext slices next to the ext cores
    (hot banks: u1 attention 290 vs 260 us). bank_loc[b] = (x, y) logical worker preferred by bank b.
    Returns bank{fi}, offset{fi} (bank-local page offset), pages_per_layer (multiple of banks)."""
    cores = sorted(roles)
    bank = {}
    for cls in ([f for f in cores if roles[f].role == QR_EXT], [f for f in cores if roles[f].role != QR_EXT]):
        caps = [len(cls) // banks + (1 if b < len(cls) % banks else 0) for b in range(banks)]
        pairs = sorted((abs(plan.xy(f)[0] - bx) + abs(plan.xy(f)[1] - by), f, b)
                       for f in cls for b, (bx, by) in enumerate(bank_loc))
        for _, f, b in pairs:
            if f in bank or caps[b] == 0:
                continue
            bank[f] = b
            caps[b] -= 1
    off, tot = {}, [0] * banks
    for f in cores:
        off[f] = tot[bank[f]]
        tot[bank[f]] += roles[f].nt * roles[f].nk
    return bank, off, max(tot) * banks


def base_page(fi, bank, off, banks):
    """Interleaved page index of the first page of fi's slice (bank-local contiguous run)."""
    return off[fi] * banks + bank[fi]


def fold_gamma(w, gamma, norm_fix=1.0):
    """Spec v0.4: RMSNorm(x) gamma @ W = rstd * (x @ diag(gamma * NORM_FIX) W). w [H, N], gamma [H]."""
    return (gamma * norm_fix)[:, None] * w


def pack_qkv_layer(w, roles, bank, off, banks, pages_per_layer, scale=1.0):
    """One layer's page block [pages_per_layer, 32, 32] (torch) of w [H, QN*32]; page order within a slice is
    k-major (k * nt + j) to match matmul_block(in1 index k * nt)."""
    import torch

    pg = torch.zeros(pages_per_layer, TILE, TILE)
    for f, r in roles.items():
        for kl in range(r.nk):
            k = r.k0 + kl
            for j in range(r.nt):
                n = r.n0 + j
                pg[(off[f] + kl * r.nt + j) * banks + bank[f]] = w[k * TILE:(k + 1) * TILE, n * TILE:(n + 1) * TILE] * scale
    return pg


@dataclass
class KVGroups:
    members: list    # per own user: fi list (rank order)
    rank: dict       # fi -> rank in its user's group
    blocks: dict     # fi -> (b0, b1) contiguous 32-position block range of its user's context
    fan: int


def kv_groups(plan, n_users_own, n_blocks):
    """Own users over the active cores in fi order (contiguous groups), equal block ranges. 2-level pull tree with
    fanout FAN: rank r % FAN != 0 -> parent rank r - r % FAN (level 0); r % FAN == 0, r > 0 -> root (level 1)."""
    act = plan.active
    grp = {f: j * n_users_own // len(act) for j, f in enumerate(act)}
    members = [[f for f in act if grp[f] == u] for u in range(n_users_own)]
    rank = {f: members[grp[f]].index(f) for f in act}
    fan = 6
    while fan * fan < max(len(m) for m in members):
        fan += 1
    blocks = {}
    for m in members:
        ed = [round(n_blocks * k / len(m)) for k in range(len(m) + 1)]
        for k, f in enumerate(m):
            blocks[f] = (ed[k], max(ed[k + 1], ed[k] + 1))
        blocks[m[-1]] = (blocks[m[-1]][0], n_blocks)
    return KVGroups(members, rank, blocks, fan)


def tree_links(groups, fi):
    """(has_parent, parent fi, parent level, children level 0, children level 1, my index in the parent's list)."""
    u = next(i for i, m in enumerate(groups.members) if fi in m)
    m, r, fan = groups.members[u], groups.rank[fi], groups.fan
    if r == 0:
        par, lvl, k = None, 0, 0
    elif r % fan:
        par, lvl, k = m[r - r % fan], 0, r % fan - 1
    else:
        par, lvl, k = m[0], 1, r // fan - 1
    ch0 = [m[j] for j in range(r + 1, min(r + fan, len(m)))] if r % fan == 0 else []
    ch1 = [m[j] for j in range(fan, len(m), fan)] if r == 0 else []
    return par is not None, par, lvl, ch0, ch1, k


@dataclass(frozen=True)
class KVCacheDesc:
    """Per-layer KV cache as the engine reads it (MKA_KVPROD = the prod vLLM cache, read in place, no re-allocation).

    prod (paged=True): TWO tensors per attention layer, paged_k and paged_v, each [num_blocks, n_kv_heads, block_size,
        head_dim] TILE, dtype bfloat8_b (BFP4 failed the GPQA output-length gate), DRAM interleaved, replicated on every
        device. Tile page of (block p, head h, position tile r, dim tile c) = ((p * n_kv_heads + h) * (block_size // 32)
        + r) * 8 + c. This chip reads head h = chip // 2. SDPA 32-position block b of user u: p = page_table[u][b //
        (block_size // 32)], r = b % (block_size // 32). Bank spread: dim tile c lands on bank c (8 banks), as before.
    bench (paged=False): one tensor per layer, K / V interleaved per 32-position block:
        ((u * n_blocks + b) * 2 + kv) * 8 + c.
    k_addr / v_addr / num_blocks / n_kv_heads / block_size come per layer from integ/stage1.kv_descriptors."""
    dtype: str = "bfp8"
    head_dim: int = 256
    block_size: int = 64
    n_kv_heads: int = 4
    paged: bool = True

    @property
    def pos_tiles_per_block(self):
        assert self.block_size % TILE == 0
        return self.block_size // TILE

    def page(self, p, h, r, c):
        return ((p * self.n_kv_heads + h) * self.pos_tiles_per_block + r) * (self.head_dim // TILE) + c

    def kernel_defines(self, head):
        """Compile-time defines for the mka2 reader / writer (head = chip // 2)."""
        return [("MKA_KVPROD", "1"), ("MKA_NKVH", str(self.n_kv_heads)), ("MKA_BS32", str(self.pos_tiles_per_block)),
                ("MKA_KVH", str(head))]


def rows_R(n_users):
    r = 4 * int(n_users)
    for c in (4, 8, 16, 32):
        if r <= c:
            return c
    raise ValueError(f"n_users={n_users}: R={r} > 32")


# ---------------------------------------------------------------------------------------------------------------------
# Spec §18 (v1.2+): qkv(+gate) split-K on the 80 STREAM cores, consumed through v12's shared split-K + column-RS unit.
# Core (column c, K-row k) of the 10 x 8 STREAM rectangle computes the partial of its column's N-list over K tiles
# 20k .. 20k+19 (x from CB_XK), then the column reduce-scatter gives row r the tiles {r, r + 8} of the N-list.
QKV_COLS, QKV_KROWS, QKV_KPR = 10, 8, KT // 8   # 20 K tiles per K-row


@dataclass(frozen=True)
class SplitKSlice:
    col: int
    krow: int
    first: int       # first qkv tile of the column's N-list
    nt: int          # tiles in the N-list (11 or 12)
    k0: int          # first K tile (20 * krow)

    @property
    def pages(self):
        return self.nt * QKV_KPR

    def page_of(self, kl, j):
        """k-major order inside the slice: matmul_block in1 index kl * nt + j."""
        return kl * self.nt + j


def qkv_splitk_slices(qn=QN, cols=QKV_COLS, krows=QKV_KROWS):
    """{(col, krow): SplitKSlice}. N-lists are contiguous qkv tile ranges: columns 0..(qn % cols)-1 get one extra tile
    (112 over 10: 12, 12, 11 x 8)."""
    base, extra = divmod(qn, cols)
    out, first = {}, 0
    for c in range(cols):
        nt = base + (1 if c < extra else 0)
        for k in range(krows):
            out[(c, k)] = SplitKSlice(c, k, first, nt, k * QKV_KPR)
        first += nt
    assert first == qn
    return out


def qkv_splitk_owner(sl, tile):
    """Row of the column that holds the reduced tile `tile` after the column RS: row r owns N-list entries {r, r + 8}."""
    return (tile - sl.first) % QKV_KROWS


def qkv_kind(tile):
    """Route class of a qkv tile: 'q' (0-47, head tile // 8), 'k' (48-55), 'v' (56-63), 'gate' (64-111)."""
    return 'q' if tile < 48 else ('k' if tile < 56 else ('v' if tile < 64 else 'gate'))


def splitk_bank_assign(slices, stream_xy, bank_loc, banks):
    """Byte-balanced bank-local placement of the 80 slices (nearest bank first). stream_xy[(col, krow)] = logical (x, y)
    of that STREAM core. Returns bank{key}, offset{key} (bank-local page offset), pages_per_layer (multiple of banks)."""
    keys = sorted(slices)
    caps = [len(keys) // banks + (1 if b < len(keys) % banks else 0) for b in range(banks)]
    pairs = sorted((abs(stream_xy[key][0] - bx) + abs(stream_xy[key][1] - by), key, b)
                   for key in keys for b, (bx, by) in enumerate(bank_loc))
    bank = {}
    for _, key, b in pairs:
        if key in bank or caps[b] == 0:
            continue
        bank[key] = b
        caps[b] -= 1
    off, tot = {}, [0] * banks
    for key in keys:
        off[key] = tot[bank[key]]
        tot[bank[key]] += slices[key].pages
    return bank, off, max(tot) * banks


def pack_qkv_splitk_layer(w, slices, bank, off, banks, pages_per_layer, scale=1.0):
    """One layer's page block (torch [pages_per_layer, 32, 32]) of w [H, QN*32] (gamma-folded) in split-K order."""
    import torch

    pg = torch.zeros(pages_per_layer, TILE, TILE)
    for key, sl in slices.items():
        for kl in range(QKV_KPR):
            k = sl.k0 + kl
            for j in range(sl.nt):
                n = sl.first + j
                pg[(off[key] + sl.page_of(kl, j)) * banks + bank[key]] = w[k * TILE:(k + 1) * TILE, n * TILE:(n + 1) * TILE] * scale
    return pg


def splitk_filler_args(slices, bank, off, banks, pages_per_layer, page_bytes=1088):
    """Per-core inputs for v12's slot filler: {(col, krow): (base page, bytes, per-layer page stride)}."""
    return {key: (off[key] * banks + bank[key], sl.pages * page_bytes, pages_per_layer) for key, sl in slices.items()}
