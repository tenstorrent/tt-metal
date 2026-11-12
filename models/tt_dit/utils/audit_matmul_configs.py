# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only audit of which matmul config source each production DiT matmul resolves to.

For every production matmul of MiniMax-H3 and FLUX 3 Klein this resolves, without a device, the
config source the lookup hits -- exact table entry, same-M-per-core / same-M-blocks fallback,
`default_block_size` (and which swept entry that came from), the AGMM v3 / MMRS v2.3 rule engines,
the warned generic fallback -- plus the resulting grid and blocking, and flags the weak paths.

It mirrors the call path in `models/tt_dit/layers/linear.py` (fabric strided-AGMM gate,
`all_gather_minimal_matmul_async` via `get_agmm_config`, RowParallel fused MM+RS via
`get_fused_mmrs_config`, plain `minimal_matmul` via `get_matmul_config`) and the model call sites
that feed it (`default_block_size`, `force_transpose`, `core_grid`, `use_heuristic_mmcfg`). The source
label is derived by a mirror of the lookup precedence; the blocking printed is what the *real*
library function returns, and any disagreement between the two is reported as an audit MISMATCH.

Run from the repo root (pure Python, no hardware; stubs `ttnn` / `loguru` only when they are not
importable, so it also runs on a laptop):

    python3 models/tt_dit/utils/audit_matmul_configs.py            # full table
    python3 models/tt_dit/utils/audit_matmul_configs.py --weak     # weak paths only

The FLUX 3 Klein rows apply the `register_matmul_configs` table of PR #64101 (copied below) since
that model is not on main yet; drop `KLEIN_REGISTERED` once it lands and import it instead.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

TILE = 32


# ------------------------------------------------------------------------------------------ stubs
def _install_stubs_if_needed() -> bool:
    """Install minimal `ttnn` / `loguru` / `models.common.utility_functions` stand-ins if absent.

    Only the names `utils/matmul.py` and `minimax_h3/mmrs_config.py` touch at import or lookup time.
    Returns True when stubs were installed.
    """
    stubbed = False
    try:
        import ttnn

        # The repo root's `ttnn/` source dir imports as an empty namespace package without a build.
        if not hasattr(ttnn, "CoreCoord"):
            raise ImportError("ttnn is not built")
    except Exception:
        sys.modules.pop("ttnn", None)
        stubbed = True
        ttnn = types.ModuleType("ttnn")

        @dataclass(frozen=True)
        class CoreCoord:
            x: int
            y: int

        class _Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def __repr__(self):
                return f"{type(self).__name__}({self.__dict__})"

        class MinimalMatmulConfig(_Config):
            pass

        class MatmulMultiCoreReuseMultiCast1DProgramConfig(_Config):
            pass

        class SDPAProgramConfig(_Config):
            pass

        ttnn.CoreCoord = CoreCoord
        ttnn.MinimalMatmulConfig = MinimalMatmulConfig
        ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig = MatmulMultiCoreReuseMultiCast1DProgramConfig
        ttnn.SDPAProgramConfig = SDPAProgramConfig
        ttnn.TILE_SIZE = TILE
        ttnn.device = types.SimpleNamespace(is_blackhole=lambda: True)
        sys.modules["ttnn"] = ttnn
    try:
        import loguru  # noqa: F401
    except Exception:
        stubbed = True
        loguru = types.ModuleType("loguru")
        loguru.logger = types.SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
        sys.modules["loguru"] = loguru
    try:
        from models.common.utility_functions import is_blackhole  # noqa: F401
    except Exception:
        stubbed = True
        uf = types.ModuleType("models.common.utility_functions")
        uf.is_blackhole = lambda: True
        uf.is_wormhole_b0 = lambda: False
        sys.modules["models.common.utility_functions"] = uf
    return stubbed


STUBBED = _install_stubs_if_needed()

import ttnn  # noqa: E402
from models.tt_dit.models.transformers.minimax_h3 import agmm_config, mmrs_config  # noqa: E402
from models.tt_dit.utils import agmm_rules, matmul, mmrs_rules  # noqa: E402

# ------------------------------------------------------------------------------------ model facts
H3_DIR = _REPO / "models/tt_dit/models/transformers/minimax_h3"
H3_PIPELINE = _REPO / "models/tt_dit/pipelines/minimax_h3/pipeline_minimax_h3.py"


def _ast_constant(path: Path, name: str, cls: str | None = None):
    """Read a literal module (or class) constant without importing its torch-heavy module."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    for node in body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


H3_LADDER_4X8 = _ast_constant(H3_PIPELINE, "MINIMAX_H3_BUCKET_LADDER_4X8")
H3_SDPA_MEASURED = _ast_constant(H3_DIR / "attention_minimax_h3.py", "measured_sdpa_chunk_sizes", "MiniMaxH3Attention")

# MiniMax-H3 (hidden 5376, inner 7168 = 56 heads x 128, ffn 14336), TP=4 / SP=8 on a 4x8 Galaxy.
H3_TP, H3_SP = 4, 8
H3_HIDDEN, H3_INNER, H3_FFN = 5376, 7168, 14336
H3_HEADS_LOCAL = 56 // H3_TP
# Per-device packed length at 768P with SP * TILE padding only (the keys the tables were swept at).
H3_SP_TILE_M = {"5s": 4736, "10s": 9184, "15s": 13664}

# FLUX 3 Klein (hidden 3072, MLP 9216 SwiGLU), BH Galaxy SP8 x TP4.
KLEIN_HIDDEN, KLEIN_MLP, KLEIN_TP = 3072, 9216, 4
# register_matmul_configs() table of PR #64101 (transformer_flux3_klein.py), copied verbatim.
KLEIN_REGISTERED = {
    "12x9": {
        (2560, 3072, 4608): (4, 8, 16, (1, 4)),
        (3072, 3072, 4608): (4, 8, 16, (1, 4)),
        (8192, 3072, 2304): (11, 8, 8, (1, 4)),
        (8192, 3072, 4608): (6, 8, 16, (1, 4)),
        (8704, 3072, 2304): (8, 8, 8, (1, 4)),
        (8704, 3072, 4608): (6, 8, 16, (1, 4)),
        (9216, 3072, 2304): (8, 8, 8, (1, 4)),
        (9216, 3072, 4608): (6, 8, 16, (1, 4)),
    }
}
# Per-device M: single stream = image/8 + 512 text (+512 per 1 MP reference); early stacks run the
# text (512), image (image/8) and reference (512) modes separately.
KLEIN_SINGLE_M = {1024: 1024, 2048: 2560, 4096: 8704}
KLEIN_EARLY_IMAGE_M = {1024: 512, 2048: 2048, 4096: 8192}

ARCH = {
    # full device worker grid, num_links (pipeline presets)
    "BH": dict(full_grid=(12, 10), num_links=2, galaxy_matmul_grid=(11, 10)),
    "WH": dict(full_grid=(8, 9), num_links=4, galaxy_matmul_grid=(8, 9)),
}
# `default_fused_mmrs_config` is chosen at import by arch (utils/matmul.py); mirror both.
_DEFAULT_MMRS = {
    "BH": matmul.FusedMMRSConfig(ttnn.CoreCoord(12, 8), 8, 4, 8, 2, 1, None, 1),
    "WH": matmul.FusedMMRSConfig(ttnn.CoreCoord(8, 7), 2, 8, 8, 1, 1, None, 1),
}


@contextlib.contextmanager
def arch_context(arch: str):
    bh = arch == "BH"
    saved = (matmul.is_blackhole, matmul.default_fused_mmrs_config, ttnn.device.is_blackhole)
    matmul.is_blackhole = lambda: bh
    matmul.default_fused_mmrs_config = _DEFAULT_MMRS[arch]
    try:
        ttnn.device.is_blackhole = lambda: bh
    except Exception:
        pass
    try:
        yield
    finally:
        matmul.is_blackhole, matmul.default_fused_mmrs_config = saved[0], saved[1]
        with contextlib.suppress(Exception):
            ttnn.device.is_blackhole = saved[2]


# ------------------------------------------------------------------------------- lookup mirrors
_HAS_M_CORES = "m_cores" in inspect.signature(matmul.get_matmul_config).parameters
_BLOCKS_MATCH = getattr(matmul, "_same_m_blocks_match", None)
_AGMM_SIG = inspect.signature(agmm_config.agmm_block_size).parameters


def _cc(g):
    return ttnn.CoreCoord(*g)


def _xy(c):
    return (c.x, c.y)


def mirror_matmul_lookup(M, K, N, grid, dbs=None, use_heuristic=False, m_cores=None):
    """Mirror of get_matmul_config's precedence. Returns (source_label, (Mb, Kb, Nb, (sh, sw)))."""
    gx, gy = grid
    table = matmul._grid_config_lookup.get((gx, gy))
    mc = m_cores if (_HAS_M_CORES and m_cores) else gx
    hit, label = None, None
    if table is not None:
        if (M, K, N) in table:
            hit, label = table[(M, K, N)], f"table[{gx}x{gy}] exact"
        else:
            near = matmul._same_m_per_core_match(table, M, K, N, mc)
            if near is not None:
                hit, label = near[1], f"table[{gx}x{gy}] same-M/core(M={near[0]})"
            elif _BLOCKS_MATCH is not None:
                near = _BLOCKS_MATCH(table, M, K, N, mc)
                if near is not None:
                    hit, label = near[1], f"table[{gx}x{gy}] same-M-blocks(M={near[0]})"
    if hit is not None:
        sb = hit[3] if len(hit) == 4 else (2, 2)
        return label, (hit[0], hit[1], hit[2], sb)
    sb = (2, 2)
    if dbs is not None:
        mb, kb, nb = dbs
        label = "default_block_size"
    elif use_heuristic:
        mb, kb, nb, sb = matmul._compute_heuristic_blocking(M, K, N, gx, gy)
        label = "heuristic"
    else:
        mb, kb, nb = 8, 8, 8
        label = "GENERIC 8x8x8"
    if math.ceil(M / TILE) < mb:
        mb = sb[0]
    if math.ceil(N / TILE) < nb:
        nb = sb[1]
    return label, (mb, kb, nb, sb)


def resolve_agmm(arch, M, K, N, *, dbs, core_grid=None, use_heuristic=False, fuse_swiglu=False, use_addcmul=False,
                 force_transpose=True, cluster=4):  # fmt: skip
    a = ARCH[arch]
    full = _cc(a["full_grid"])
    transpose = force_transpose or M > N
    legacy = core_grid or _xy(matmul.agmm_worker_grid(full, transpose, a["num_links"]))
    table = matmul._grid_config_lookup.get(legacy, {})
    m_cores = legacy[0] if transpose else legacy[1]
    table_hit = (M, K, N) in table or matmul._same_m_per_core_match(table, M, K, N, m_cores) is not None
    if _BLOCKS_MATCH is not None and not table_hit:
        table_hit = _BLOCKS_MATCH(table, M, K, N, m_cores) is not None
    if core_grid is not None or dbs is not None or use_heuristic or table_hit:
        label, blk = mirror_matmul_lookup(M, K, N, legacy, dbs, use_heuristic, m_cores)
        grid = legacy
    else:
        v3 = None
        if arch == "BH" and transpose == (M > N):
            v3 = agmm_rules.pick_v3(M, K, N, cluster_size=cluster, full_grid=a["full_grid"], fuse_swiglu=fuse_swiglu,
                                    use_addcmul=use_addcmul)  # fmt: skip
        if v3 is None:
            label, blk = mirror_matmul_lookup(M, K, N, legacy)
            kt = K // TILE // cluster
            if kt % blk[1]:
                blk = (blk[0], max(d for d in range(1, blk[1] + 1) if kt % d == 0), blk[2], blk[3])
                label += " +ring-safe K"
            grid = legacy
        else:
            label = "agmm v3 rules"
            blk = (*v3["blocks"], v3["subblock"])
            grid = v3["core_grid"]
            transpose = v3["transposed"]
    # Ground truth from the real library.
    with arch_context(arch):
        real_grid, cfg, _ = matmul.get_agmm_config(
            M, K, N, full_grid=full, cluster_size=cluster, num_links=a["num_links"],
            core_grid=_cc(core_grid) if core_grid else None, default_block_size=dbs, use_heuristic=use_heuristic,
            fuse_swiglu=fuse_swiglu, use_addcmul=use_addcmul, force_transpose=force_transpose,
        )  # fmt: skip
    real = (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size, (cfg.subblock_h, cfg.subblock_w))
    if real != blk or _xy(real_grid) != tuple(grid):
        label = f"MISMATCH(mirror {label} {grid} {blk})"
    return label, tuple(_xy(real_grid)), real, transpose


def resolve_mmrs(arch, M, K, N):
    a = ARCH[arch]
    full = _cc(a["full_grid"])
    with arch_context(arch):
        entry = matmul.fused_mmrs_configs.get(full, {}).get((M, K, N))
        if entry is not None:
            label = "mmrs table exact"
        elif arch == "BH" and M % 32 == 0 and mmrs_rules.pick_v23(M, K, N, full_grid=a["full_grid"]) is not None:
            label = "mmrs v2.3 rules"
        else:
            label = "MMRS DEFAULT"
        params = matmul.get_fused_mmrs_config(M, K, N, full, a["num_links"])
    c = params["config"]
    blk = (c.M_block_size, c.K_block_size, c.N_block_size, (c.subblock_h, c.subblock_w))
    return label, _xy(c.compute_with_storage_grid_size), blk, M > N, params["mm_window_blocks"]


def agmm_dbs(k, n, m, arch):
    """H3 call sites: `agmm_block_size(K, N, M)` (attention_minimax_h3.py / transformer_block_minimax_h3.py)."""
    kwargs = {}
    if "rules_fallback" in _AGMM_SIG:
        kwargs["rules_fallback"] = arch == "BH"
    dbs = agmm_config.agmm_block_size(k, n, m, **kwargs)
    pcm = agmm_config._per_core_m(m, n)
    if dbs is None:
        return None, f"agmm_block_size: None (pcm {pcm})"
    swept = sorted(p for (kk, nn, p) in agmm_config.AGMM_BLOCK_SIZES if kk == k and nn == n)
    src = max(p for p in swept if pcm % p == 0) if pcm not in swept else pcm
    how = "exact" if src == pcm else f"BORROWED from pcm {src}"
    return dbs, f"agmm_block_size pcm {pcm} {how}"


# --------------------------------------------------------------------------------- weak checks
def per_core_flags(M, N, grid, blk, transposed):
    gx, gy = grid
    m_cores, n_cores = (gx, gy) if transposed else (gy, gx)
    pcm = math.ceil(math.ceil(M / TILE) / m_cores)
    pcn = math.ceil(math.ceil(N / TILE) / n_cores)
    mb, _, nb, _ = blk
    m_eff = pcm / (math.ceil(pcm / mb) * mb)
    n_eff = pcn / (math.ceil(pcn / nb) * nb)
    flags = []
    if m_eff < 0.85:
        flags.append(f"M-pad {1 - m_eff:.0%}")
    if n_eff < 0.85:
        flags.append(f"N-pad {1 - n_eff:.0%}")
    if nb > pcn:
        flags.append("N_block>pcN")
    tail = pcn - (math.ceil(pcn / nb) - 1) * nb
    if math.ceil(pcn / nb) > 1 and tail < nb / 2:
        flags.append(f"N-tail {nb}+..+{tail}")
    if mb <= 2 and pcm > 12:
        flags.append("tiny M_block")
    return pcm, pcn, flags


# A row is weak when its source is unmeasured for that shape, or its blocking computes a padded N
# strip wider than the core's share. Padding flags alone are informational: the rule engines trade
# some padding for block balance by design and were measured within 5% of the swept optimum.
WEAK_SOURCES = ("GENERIC", "MMRS DEFAULT", "BORROWED", "heuristic", "MISMATCH", "BH-tuned on WH", "extrapolated")
HARD_FLAGS = ("N_block>pcN", "tiny M_block")


@dataclass
class Row:
    model: str
    arch: str
    case: str
    op: str
    path: str
    M: int
    K: int
    N: int
    grid: tuple
    blk: tuple
    source: str
    flags: list

    @property
    def weak(self) -> bool:
        return any(s in self.source for s in WEAK_SOURCES) or any(f in HARD_FLAGS for f in self.flags)


def h3_rows():
    rows = []
    for arch in ("BH", "WH"):
        for dur, m_sp in H3_SP_TILE_M.items():
            bucket = next(r for r in H3_LADDER_4X8 if r >= m_sp * H3_SP) // H3_SP
            for pad, M in (("sp*tile", m_sp), ("bucket", bucket)):
                case = f"{dur} {pad}"
                # attention to_qkv: chunks=3 -> never the fabric path; force_transpose=False.
                for op, K, N, extra in (
                    ("qkv", H3_HIDDEN, 3 * H3_INNER // H3_TP, {}),
                    ("to_out", H3_INNER, H3_HIDDEN // H3_TP, {"use_addcmul": True}),
                    ("ff1", H3_HIDDEN, 2 * H3_FFN // H3_TP, {"fuse_swiglu": True}),
                ):
                    with arch_context(arch):
                        fab = matmul.get_fabric_agmm_config(M, K, N, 1, _cc(ARCH[arch]["full_grid"]))
                    assert fab is None or op == "qkv", f"unexpected fabric hit for H3 {op}"
                    dbs, dbs_label = agmm_dbs(K, N, M, arch)
                    label, grid, blk, tr = resolve_agmm(arch, M, K, N, dbs=dbs, force_transpose=False, **extra)
                    if label == "default_block_size":
                        label = dbs_label + (" [BH-tuned on WH]" if arch == "WH" else "")
                        label = label.replace("[BH-tuned on WH]", "BH-tuned on WH")
                    pcm, pcn, flags = per_core_flags(M, N, grid, blk, tr)
                    rows.append(Row("H3", arch, case, op, "agmm", M, K, N, grid, blk, label, flags))
                # ff2: fused MM+RS when has_mmrs_config, else unfused RowParallel minimal_matmul.
                K, N = H3_FFN // H3_TP, H3_HIDDEN
                with arch_context(arch):
                    fused = mmrs_config.has_mmrs_config(M, K, N, _cc(ARCH[arch]["full_grid"]))
                    if fused:
                        mmrs_config.register_mmrs_config(M, K, N, _cc(ARCH[arch]["full_grid"]))
                if fused:
                    label, grid, blk, tr, win = resolve_mmrs(arch, M, K, N)
                    path = f"mmrs(win={win})"
                else:
                    grid = ARCH[arch]["galaxy_matmul_grid"]
                    label, blk = mirror_matmul_lookup(M, K, N, grid)
                    with arch_context(arch):
                        cfg = matmul.get_matmul_config(M, K, N, _cc(grid))
                    real = (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size, (cfg.subblock_h, cfg.subblock_w))
                    if real != blk:
                        label = f"MISMATCH(mirror {label} {blk})"
                        blk = real
                    tr, path = M > N, "minimal_matmul"
                pcm, pcn, flags = per_core_flags(M, N, grid, blk, tr)
                rows.append(Row("H3", arch, case, "ff2", path, M, K, N, grid, blk, label, flags))
    return rows


def klein_core_grid(M, full=(12, 10)):
    """`Attention.get_core_grid` (blocks/attention_opt.py)."""
    return (full[0], full[1] - 2) if M <= 1152 else (full[0], full[1] - 1)


def klein_rows():
    matmul.register_matmul_configs(KLEIN_REGISTERED)
    rows = []
    cases = []
    for size in (1024, 2048, 4096):
        for refs in (0, 1):
            cases.append((f"{size}^2 r{refs} single", KLEIN_SINGLE_M[size] + 512 * refs))
        cases.append((f"{size}^2 early-image", KLEIN_EARLY_IMAGE_M[size]))
    cases.append(("all early-text/ref", 512))
    seen = set()
    for case, M in cases:
        if M in seen:
            case += " (dup M)"
        seen.add(M)
        grid = klein_core_grid(M)
        # qkv (chunks=3) and proj_mlp (packed SwiGLU): explicit core_grid + use_heuristic_mmcfg,
        # force_transpose left at ColParallelLinear's default True.
        klein_ops = (
            ("qkv", 3 * KLEIN_HIDDEN // KLEIN_TP, {}),
            ("proj_mlp", 2 * KLEIN_MLP // KLEIN_TP, {"fuse_swiglu": True}),
        )
        for op, N, extra in klein_ops:
            with arch_context("BH"):
                fab = matmul.get_fabric_agmm_config(M, KLEIN_HIDDEN, N, 1, _cc(ARCH["BH"]["full_grid"]))
            path = "agmm"
            if fab is not None and op == "proj_mlp":
                path = "FABRIC"
            label, g, blk, tr = resolve_agmm(
                "BH", M, KLEIN_HIDDEN, N, dbs=None, core_grid=grid, use_heuristic=True, **extra
            )
            _, _, flags = per_core_flags(M, N, g, blk, tr)
            rows.append(Row("Klein", "BH", case, op, path, M, KLEIN_HIDDEN, N, g, blk, label, flags))
        if "early" in case:
            continue  # early blocks share the single-stream block; proj_out M listed with the single rows
        K = (KLEIN_HIDDEN + KLEIN_MLP) // KLEIN_TP
        label, g, blk, tr, win = resolve_mmrs("BH", M, K, KLEIN_HIDDEN)
        if label == "mmrs v2.3 rules":
            label += " (extrapolated: N=3072 family not in fit)"
        _, _, flags = per_core_flags(M, KLEIN_HIDDEN, g, blk, tr)
        rows.append(Row("Klein", "BH", case, "proj_out", f"mmrs(win={win})", M, K, KLEIN_HIDDEN, g, blk, label, flags))
    # Early blocks at M=512/2048/8192 also run proj_out through MMRS.
    for M in sorted({512, *KLEIN_EARLY_IMAGE_M.values()} - {r.M for r in rows if r.op == "proj_out"}):
        K = (KLEIN_HIDDEN + KLEIN_MLP) // KLEIN_TP
        label, g, blk, tr, win = resolve_mmrs("BH", M, K, KLEIN_HIDDEN)
        if label == "mmrs v2.3 rules":
            label += " (extrapolated: N=3072 family not in fit)"
        _, _, flags = per_core_flags(M, KLEIN_HIDDEN, g, blk, tr)
        path = f"mmrs(win={win})"
        rows.append(Row("Klein", "BH", f"early M={M}", "proj_out", path, M, K, KLEIN_HIDDEN, g, blk, label, flags))
    return rows


def unreachable_h3_entries():
    """H3-shaped entries in grid tables that no H3 call can look up (force_transpose=False on BH 12x10)."""
    out = []
    families = {(H3_HIDDEN, 3 * H3_INNER // H3_TP), (H3_INNER, H3_HIDDEN // H3_TP), (H3_HIDDEN, 2 * H3_FFN // H3_TP)}
    for grid, table in matmul._grid_config_lookup.items():
        for M, K, N in table:
            if (K, N) not in families or grid not in ((12, 9), (11, 10)):
                continue
            want = _xy(matmul.agmm_worker_grid(_cc((12, 10)), M > N, 2))
            if want != grid:
                out.append(((M, K, N), grid, want))
    return out


def sdpa_rows():
    rows = []
    cores = (ARCH["BH"]["full_grid"][0] - 1) * ARCH["BH"]["full_grid"][1]
    for dur, m_sp in H3_SP_TILE_M.items():
        bucket = next(r for r in H3_LADDER_4X8 if r >= m_sp * H3_SP) // H3_SP
        for pad, s in (("sp*tile", m_sp), ("bucket", bucket)):
            measured = H3_SDPA_MEASURED.get(s)
            fallback = (max(TILE, min(256, s // TILE * TILE)), max(TILE, min(512, s // TILE * TILE)))
            q, k = measured or fallback
            items = H3_HEADS_LOCAL * math.ceil(s / q)
            per_core = math.ceil(items / cores)
            rows.append((f"{dur} {pad}", s, "table" if measured else "FALLBACK", (q, k), items, per_core, per_core * q))
    return rows


def fmt_blk(b):
    return f"({b[0]},{b[1]},{b[2]}) sb{b[3][0]}x{b[3][1]}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weak", action="store_true", help="only print rows on a weak path")
    args = ap.parse_args(argv)

    rows = h3_rows() + klein_rows()
    hdr = (
        f"{'model':5} {'arch':4} {'case':22} {'op':8} {'path':15} {'M':>6} {'K':>5} {'N':>5} {'grid':6} "
        f"{'blocking':18} source / flags"
    )
    print(f"# matmul config audit (stubs={'on' if STUBBED else 'off'})")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if args.weak and not r.weak:
            continue
        mark = "!" if r.weak else " "
        flags = f"  [{', '.join(r.flags)}]" if r.flags else ""
        print(
            f"{r.model:5} {r.arch:4} {r.case:22} {r.op:8} {r.path:15} {r.M:6} {r.K:5} {r.N:5} "
            f"{r.grid[0]}x{r.grid[1]:<4} {fmt_blk(r.blk):18} {mark}{r.source}{flags}"
        )
    print()
    print("# H3 ring SDPA chunk sizes (BH 4x8, 11x10 SDPA grid, 14 local heads)")
    print(f"{'case':12} {'seq_local':>9} {'source':9} {'(q,k)':11} {'items':>5} {'/core':>5} {'rows/core':>9}")
    for case, s, src, qk, items, pc, rpc in sdpa_rows():
        print(f"{case:12} {s:9} {src:9} {str(qk):11} {items:5} {pc:5} {rpc:9}")
    print()
    dead = unreachable_h3_entries()
    print("# H3-shaped table entries no H3 call can reach (AGMM grid is chosen by M > N with force_transpose=False)")
    for key, grid, want in dead:
        print(f"  {key} keyed under {grid[0]}x{grid[1]}, but that shape runs on {want[0]}x{want[1]}")
    if not dead:
        print("  none")
    weak = sum(r.weak for r in rows)
    print(f"\n{weak} of {len(rows)} production matmul lookups on a weak path ('!').")


if __name__ == "__main__":
    main()
