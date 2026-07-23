#!/usr/bin/env python3
"""Offline mirror of the Regime-A planner + production picker (device-free).

Faithful Python reproduction of:
  - build_plan() geometry + feasibility  (regime_a_matmul_plan.hpp)
  - auto_select_config() production picker (regime_a_matmul_config.cpp)

Used by the picker-generalization campaign to (a) enumerate the PLANNER-feasible config set for a shape
without launching the device, (b) compute the current production picker's choice for comparison, and
(c) host the PROPOSED hierarchical heuristic (propose_config) developed in analyze.py. NO C++ change is
made from this phase — this module is the offline laboratory only.

See AUDIT.md for the prose description of everything mirrored here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---- platform constants (plan.hpp single source of truth) ----
G = KNUMBANKS = 8
TILE_BF16 = 2048
TILE_FP32 = 4096
L1_BUDGET = 1440 * 1024
MIN_CORES = 16
MAX_CORES = 104
GRID_CELLS = 110  # BH p150b 11x10, no holes in v1
AVAILABLE = GRID_CELLS


def cdiv(a, b):
    return (a + b - 1) // b


def rup(x, y):
    return cdiv(x, y) * y


def nt_width_shard_feasible(Nt):
    """plan.hpp nt_width_shard_feasible: 7*ceil(Nt/8) < Nt."""
    return 7 * cdiv(Nt, 8) < Nt


# ------------------------------------------------------------------------------------------------
# Geometry (mirror of build_plan's Geometry + CB sizing)
# ------------------------------------------------------------------------------------------------
@dataclass
class Geo:
    Mt: int
    Kt: int
    Nt: int
    Pk: int
    Ns: int
    Sm: int
    kb: int
    nsb: int
    # derived
    N_band: int
    K_slice_capacity: int
    K_num_blocks_eff: int
    W: int
    M_block_capacity: int
    N_own: int
    N_sub: int
    N_bpc: int
    N_slice_capacity: int
    preaders: int
    mfac: int
    num_cores: int
    waste_k: float
    waste_n: float
    # CBs
    cb0: int
    cb1: int
    cb2: int
    cb3: int
    cb7: int
    l1_bytes: int


def geometry(Mt, Kt, Nt, Pk, Ns, Sm, kb, nsb):
    """Compute the planner geometry (does not check feasibility). Mirrors build_plan lines 261-317."""
    Pk = max(1, Pk)
    Ns = max(1, Ns)
    Sm = max(1, Sm)
    kb = max(1, kb)
    N_band = cdiv(Nt, 8)
    K_slice_capacity = rup(cdiv(Kt, Pk), kb * 8)
    K_num_blocks_eff = K_slice_capacity // kb
    W = K_num_blocks_eff // 8
    M_block_capacity = cdiv(Mt, Sm)
    N_own = cdiv(N_band, Ns)
    N_sub = nsb if nsb else N_own
    N_bpc = cdiv(N_own, N_sub)
    N_slice_capacity = N_bpc * N_sub
    preaders = Pk * Ns * Sm
    mfac = Ns * Sm
    num_cores = 8 * preaders
    waste_k = Pk * K_slice_capacity / Kt - 1.0
    waste_n = 8 * Ns * N_slice_capacity / Nt - 1.0
    cb0 = M_block_capacity * K_slice_capacity
    cb1 = 4 * kb * N_sub
    cb2 = 2 * M_block_capacity * N_sub
    cb3 = M_block_capacity * N_sub
    cb7 = (2 * M_block_capacity * N_sub) if Pk > 1 else 0
    l1_bytes = (cb0 + cb1 + cb2 + cb7) * TILE_BF16 + cb3 * TILE_FP32
    return Geo(
        Mt,
        Kt,
        Nt,
        Pk,
        Ns,
        Sm,
        kb,
        nsb,
        N_band,
        K_slice_capacity,
        K_num_blocks_eff,
        W,
        M_block_capacity,
        N_own,
        N_sub,
        N_bpc,
        N_slice_capacity,
        preaders,
        mfac,
        num_cores,
        waste_k,
        waste_n,
        cb0,
        cb1,
        cb2,
        cb3,
        cb7,
        l1_bytes,
    )


def plan_feasible(Mt, Kt, Nt, Pk, Ns, Sm, kb, nsb, available=AVAILABLE, l1_budget=L1_BUDGET):
    """Return (ok: bool, reason: str) mirroring build_plan's reject set (section 3 of AUDIT.md).

    This is the DEVICE-LAUNCHABLE set (planner), NOT the picker's stricter pick_plan set. The sweep
    enumerates over THIS so no candidate the device could run is silently pruned.
    """
    Pk = max(1, Pk)
    Ns = max(1, Ns)
    Sm = max(1, Sm)
    kb = max(1, kb)
    if Mt == 0 or Kt == 0 or Nt == 0:
        return False, "Mt/Kt/Nt must be > 0"
    if Sm > Mt:
        return False, "Sm > Mt"
    if Pk > Kt:
        return False, "Pk > Kt"
    if not nt_width_shard_feasible(Nt):
        return False, "Nt too small to width-shard 8 banks"
    g = geometry(Mt, Kt, Nt, Pk, Ns, Sm, kb, nsb)
    if nsb and nsb > g.N_own:
        return False, "nsb > N_own"
    if g.K_num_blocks_eff % 8 != 0:
        return False, "K_num_blocks_eff % 8"
    if g.num_cores > available:
        return False, f"needs {g.num_cores} cores > {available}"
    if g.l1_bytes > l1_budget:
        return False, f"L1 over budget {g.l1_bytes} > {l1_budget}"
    # empty-ownership: balanced ranges over Pk/Sm/Ns(valid band) must be non-empty for every core.
    # valid_k>=1 iff Pk<=Kt (checked). valid_m>=1 iff Sm<=Mt (checked). valid_n: each bank's valid band
    # width b_valid subdivided over Ns must be non-empty for the LAST used bank. Bank b valid width =
    # clamp(Nt - b*N_band, 0, N_band); smallest nonzero is the last bank. Require its Ns-split nonempty.
    last_bank_valid = max(0, min(Nt - 7 * g.N_band, g.N_band))
    if last_bank_valid == 0:
        return False, "empty ownership (bank)"
    # balanced split of last_bank_valid over Ns: owner Ns-1 gets floor(Ns*bv/Ns)-floor((Ns-1)*bv/Ns)
    lo = (Ns - 1) * last_bank_valid // Ns
    hi = Ns * last_bank_valid // Ns
    if hi - lo == 0:
        return False, "empty ownership (n-slice)"
    return True, "ok"


def nsb_lattice(N_own):
    """Geometric + boundary lattice of nsb values in [1, N_own] (NOT the picker's set).

    N_sub is the N-subblock width fed to compute; perf is smooth in nsb with real optima only at small
    widths (matmul DST / subblock efficiency + CB1 depth 4*kb*nsb) and at the single-block case
    (nsb=N_own). We sample {1..8} densely, then divisors of N_own <= 16 (even N_bpc pipelining
    boundaries), then N_own itself. This captures every meaningful block-size optimum while cutting the
    wide-N nsb explosion (~24 -> ~10). Justified in the manifest; verified against a full-nsb control
    shape in the sweep.
    """
    s = set(range(1, min(8, N_own) + 1))
    for d in range(1, min(16, N_own) + 1):
        if N_own % d == 0:
            s.add(d)
    s.add(N_own)
    return sorted(v for v in s if 1 <= v <= N_own)


def enumerate_feasible(Mt, Kt, Nt, pk_max=13, ns_max=8, kb_set=(1, 2, 4, 8), available=AVAILABLE, nsb_mode="full"):
    """Every PLANNER-feasible (Pk,Ns,Sm,kb,nsb) for a shape. Ranges are the union of what the device can
    launch given the 104-core cap (Pk*Ns*Sm<=13) — deliberately WIDER than the picker's pick_plan set.

    nsb ranges 1..N_own for the (Ns) it is evaluated under. Returns list of dict rows with geometry.
    """
    out = []
    for Pk in range(1, min(pk_max, Kt) + 1):
        for Ns in range(1, ns_max + 1):
            for Sm in range(1, Mt + 1):
                if Pk * Ns * Sm * 8 > available:
                    continue
                N_band = cdiv(Nt, 8)
                N_own = cdiv(N_band, Ns)
                nsb_vals = nsb_lattice(N_own) if nsb_mode == "lattice" else range(1, N_own + 1)
                for kb in kb_set:
                    for nsb in nsb_vals:
                        ok, why = plan_feasible(Mt, Kt, Nt, Pk, Ns, Sm, kb, nsb, available)
                        if not ok:
                            continue
                        g = geometry(Mt, Kt, Nt, Pk, Ns, Sm, kb, nsb)
                        out.append(
                            {
                                "Pk": Pk,
                                "Ns": Ns,
                                "Sm": Sm,
                                "kb": kb,
                                "nsb": nsb,
                                "cores": g.num_cores,
                                "W": g.W,
                                "K_slice_capacity": g.K_slice_capacity,
                                "M_block_capacity": g.M_block_capacity,
                                "N_own": g.N_own,
                                "N_sub": g.N_sub,
                                "N_bpc": g.N_bpc,
                                "N_band": g.N_band,
                                "waste_k": round(g.waste_k, 4),
                                "waste_n": round(g.waste_n, 4),
                                "l1_bytes": g.l1_bytes,
                            }
                        )
    return out


# ------------------------------------------------------------------------------------------------
# Production picker (mirror of auto_select_config)
# ------------------------------------------------------------------------------------------------
# Cost-model params (config.cpp lines 37-44).
CSAT, ACAP, KBCAP = 24, 6, 2
KK, AA, OVL, START, WST = 0.5, 2.0, 1.0, 0.0, 0.5
RK, MSPLIT_MARGIN, NBAND_MAX = 0.8, 0.03, 2

# Oracle lookup table, verbatim from config.cpp kTable (keyed (Mt,Kt,Nt) -> (Pk,Ns,Sm,kb,nsb)).
KTABLE = {
    (1, 64, 16): (4, 2, 1, 2, 1),
    (1, 64, 48): (2, 2, 1, 4, 3),
    (1, 192, 48): (6, 1, 1, 4, 2),
    (1, 64, 64): (2, 2, 1, 4, 4),
    (1, 192, 72): (3, 1, 1, 4, 5),
    (1, 192, 96): (3, 1, 1, 4, 6),
    (1, 8, 192): (1, 3, 1, 1, 8),
    (1, 192, 192): (6, 1, 1, 4, 2),
    (1, 192, 288): (3, 1, 1, 4, 6),
    (2, 192, 48): (3, 1, 1, 8, 2),
    (2, 480, 48): (6, 1, 1, 2, 3),
    (2, 192, 144): (6, 1, 1, 4, 2),
    (2, 144, 192): (3, 2, 1, 2, 3),
    (2, 192, 288): (6, 1, 1, 4, 2),
    (4, 192, 24): (12, 1, 1, 2, 1),
    (4, 480, 24): (6, 1, 1, 2, 3),
    (4, 192, 72): (12, 1, 1, 2, 1),
    (4, 192, 144): (12, 1, 1, 2, 1),
    (4, 72, 192): (3, 2, 1, 1, 6),
    (16, 192, 48): (12, 1, 1, 2, 1),
    (8, 64, 32): (4, 1, 2, 2, 4),
    (8, 480, 48): (6, 1, 2, 2, 6),
    (8, 64, 16): (4, 1, 3, 2, 2),
    (1, 480, 24): (6, 1, 1, 2, 3),
    (8, 72, 192): (3, 4, 1, 1, 3),
    (4, 64, 16): (4, 1, 2, 2, 2),
    (4, 480, 48): (12, 1, 1, 1, 3),
    (2, 64, 16): (4, 2, 1, 2, 1),
    (8, 64, 48): (4, 1, 3, 2, 3),
    (1, 64, 32): (2, 4, 1, 4, 1),
    (2, 64, 32): (4, 2, 1, 2, 2),
    (8, 480, 24): (6, 1, 2, 2, 3),
    (8, 64, 64): (4, 1, 3, 2, 4),
    (4, 64, 32): (4, 1, 2, 2, 4),
    (8, 192, 48): (6, 1, 2, 4, 2),
    (1, 72, 192): (3, 2, 1, 1, 6),
    (2, 64, 64): (2, 3, 1, 2, 3),
    (1, 480, 48): (6, 1, 1, 2, 3),
    (8, 192, 192): (6, 1, 2, 4, 2),
    (4, 64, 64): (4, 3, 1, 2, 3),
    (2, 480, 24): (10, 1, 1, 2, 3),
    (1, 192, 24): (6, 1, 1, 2, 3),
    (8, 192, 144): (6, 1, 2, 4, 2),
}


def _pick_plan(Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb):
    """Mirror of pick_plan (config.cpp): the picker's STRICTER feasibility (adds core-window + waste gates
    on top of the planner's L1 check). Returns Geo-like dict or None."""
    if not nt_width_shard_feasible(Nt):
        return None
    cores = 8 * Pk * Ns * Sm
    if cores < MIN_CORES or cores > MAX_CORES:
        return None
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    wasteK = Pk * Ktl / Kt - 1.0
    if wasteK > 0.20:
        return None
    Mblk = cdiv(Mt, Sm)
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    if nsb > Nown:
        return None
    Nbpc = cdiv(Nown, nsb)
    wasteN = 8 * Ns * Nbpc * nsb / Nt - 1.0
    if wasteN > 0.20:
        return None
    cb0 = Ktl * Mblk * TILE_BF16
    cb1 = 4 * kb * nsb * TILE_BF16
    cb2 = 2 * Mblk * nsb * TILE_BF16
    cb3 = Mblk * nsb * TILE_FP32
    cb7 = 2 * Mblk * nsb * TILE_BF16
    if cb0 + cb1 + cb2 + cb3 + cb7 > L1_BUDGET:
        return None
    return {"cores": cores, "Ktl": Ktl, "Mblk": Mblk, "Nown": Nown, "Nbpc": Nbpc, "wasteK": wasteK, "wasteN": wasteN}


def _pick_cost(Kt, Nt, kb, nsb, g):
    readT = Kt * Nt / min(g["cores"], CSAT)
    comp_pc = g["Mblk"] * g["Nown"] * g["Ktl"]
    area = min(g["Mblk"] * nsb, ACAP)
    kbe = min(kb, KBCAP)
    compT = comp_pc / ((kbe / (kbe + KK)) * (area / (area + AA)))
    ovlT = OVL * comp_pc / g["Nbpc"]
    base = max(readT, compT) + ovlT + START * g["Ktl"]
    return base * (1.0 + WST * (g["wasteK"] + g["wasteN"]))


def _pick_cost_v3(Kt, Nt, Pk, kb, nsb, g):
    reduce = RK * (Pk - 1 if Pk > 1 else 0) * g["Mblk"] * g["Nown"]
    return _pick_cost(Kt, Nt, kb, nsb, g) + reduce


def production_pick(Mt, Kt, Nt):
    """Exact mirror of auto_select_config: table hit else two-step cost-model fallback.
    Returns (Pk,Ns,Sm,kb,nsb, source) where source in {'table','fallback-anchor','fallback-msplit'}."""
    if (Mt, Kt, Nt) in KTABLE:
        return (*KTABLE[(Mt, Kt, Nt)], "table")
    Nband = cdiv(Nt, 8)
    # Step 1: Sm=1 anchor
    anchor, anchor_cost, anchor_g = None, math.inf, None
    for Pk in range(1, 13):
        for Ns in range(1, 7):
            Nown = cdiv(Nband, Ns)
            for kb in (1, 2, 4, 8):
                for nsb in range(1, Nown + 1):
                    g = _pick_plan(Mt, Kt, Nt, Ns, Pk, 1, kb, nsb)
                    if g is None:
                        continue
                    c = _pick_cost(Kt, Nt, kb, nsb, g)
                    if c < anchor_cost:
                        anchor_cost, anchor_g = c, g
                        anchor = (Pk, Ns, 1, kb, nsb)
    if anchor is None:
        raise RuntimeError(f"no feasible config Mt={Mt} Kt={Kt} Nt={Nt}")
    # Step 2: narrow-N M-split hysteresis
    if Nband > NBAND_MAX or Mt < 2:
        return (*anchor, "fallback-anchor")
    bestG, bestG_cost = None, math.inf
    for Pk in range(1, 13):
        for Ns in range(1, 7):
            Nown = cdiv(Nband, Ns)
            for Sm in range(2, Mt + 1):
                for kb in (1, 2, 4, 8):
                    for nsb in range(1, Nown + 1):
                        g = _pick_plan(Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb)
                        if g is None:
                            continue
                        c = _pick_cost_v3(Kt, Nt, Pk, kb, nsb, g)
                        if c < bestG_cost:
                            bestG_cost = c
                            bestG = (Pk, Ns, Sm, kb, nsb)
    anchor_cost_v3 = _pick_cost_v3(Kt, Nt, anchor[0], anchor[3], anchor[4], anchor_g)
    if bestG is not None and bestG_cost < anchor_cost_v3 * (1.0 - MSPLIT_MARGIN):
        return (*bestG, "fallback-msplit")
    return (*anchor, "fallback-anchor")


if __name__ == "__main__":
    # smoke: feasibility count + production pick for a couple shapes.
    for Mt, Kt, Nt in [(8, 64, 32), (1, 192, 192), (4, 480, 48), (8, 480, 480)]:
        feas = enumerate_feasible(Mt, Kt, Nt)
        pick = production_pick(Mt, Kt, Nt)
        print(f"Mt={Mt} Kt={Kt} Nt={Nt}: {len(feas)} feasible configs; prod pick {pick}")
