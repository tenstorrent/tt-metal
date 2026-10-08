#!/usr/bin/env python3
"""Host model of the packed sparse_sdpa_msa kernels: a line-by-line port of the reader's grouping / union / lead /
diagonal logic and of the compute's per-row mask modes, checked against the per-token reference attention
(fp64) on real captured indices and on adversarial edge cases. Also reports union statistics (blocks fetched,
row-steps, mask stamps) for the given work split."""
import random
import sys

import torch

SENT = 0xFFFFFFFF
BS = 128
KPT = 32
SKT = BS // KPT


def reader_groups(rows_ids, work_start, work_count, S, G, chunk_start, straddle_row=0, straddle_jump=0, causal=True):
    """rows_ids: list over linearized work items (kv*S + tok) of lists of ids (with SENT tail). Yields groups."""
    tok = work_start % S
    kv = work_start // S
    rem = work_count
    while rem > 0:
        g_max = min(G, rem, S - tok)
        rows = [rows_ids[kv * S + tok + j] for j in range(g_max)]
        nv = []
        for r in rows:
            lo, hi = 0, len(r)
            while lo < hi:
                mid = (lo + hi) >> 1
                if r[mid] == SENT:
                    hi = mid
                else:
                    lo = mid + 1
            nv.append(1 if lo == 0 else lo)
        uid = list(rows[0][: nv[0]])
        umask = [1] * nv[0]
        g = 1
        for j in range(1, g_max):
            bit = 1 << j
            n_before = len(uid)
            for c in range(nv[j]):
                b = rows[j][c]
                e = 0
                while e < len(uid) and not (uid[e] == b and (umask[e] & bit) == 0):
                    e += 1
                if e == len(uid):
                    uid.append(b)
                    umask.append(0)
                umask[e] |= bit
            allm = (bit << 1) - 1
            if not any((umask[e] & allm) == allm for e in range(n_before)):
                del uid[n_before:]
                del umask[n_before:]
                umask = [m & ~bit for m in umask]
                break
            g = j + 1
        allm = (1 << g) - 1
        lead = next(e for e in range(len(uid)) if (umask[e] & allm) == allm)
        if lead:
            uid.insert(0, uid.pop(lead))
            umask.insert(0, umask.pop(lead))
        geo = [None] * g
        diag_e = [None] * g
        if causal:
            for s in range(g):
                t = tok + s
                p = chunk_start + t + (straddle_jump if t >= straddle_row else 0)
                db = p // BS
                fm = p % BS + 1
                geo[s] = fm
                for e in range(len(uid)):
                    if uid[e] == db and (umask[e] >> s) & 1:
                        diag_e[s] = e
                        break
        yield dict(kv=kv, tok=tok, g=g, uid=uid, umask=umask, diag_e=diag_e, first_masked=geo, nv=nv[:g], rows=rows[:g])
        tok += g
        rem -= g
        if tok == S:
            tok = 0
            kv += 1


def check_group(grp, q, k, v, scale, chunk_start, causal):
    """Compute the packed result for each token of the group from the modes and compare with the reference."""
    g = grp["g"]
    uid = grp["uid"]
    umask = grp["umask"]
    errs = []
    n_rows = (g + 1) // 2
    # lead invariant: entry 0 selected by all
    assert umask[0] & ((1 << g) - 1) == (1 << g) - 1, "lead not common"
    # every row's first active block is entry 0
    for r in range(n_rows):
        first = next(e for e in range(len(uid)) if (umask[e] >> (2 * r)) & 3)
        assert first == 0, "row starts on a non-lead block"
    for s in range(g):
        keys = []
        mask = []
        for e, b in enumerate(uid):
            if not (umask[e] >> s) & 1:
                continue  # hidden: P = 0 exactly (state untouched / -inf)
            ks = torch.arange(b * BS, (b + 1) * BS)
            m = torch.zeros(BS, dtype=torch.bool)
            if causal and grp["diag_e"][s] == e:
                m = torch.arange(BS) >= grp["first_masked"][s]
            keys.append(ks)
            mask.append(m)
        keys = torch.cat(keys)
        mask = torch.cat(mask)
        t = grp["tok"] + s
        sc = (q[t] * scale) @ k[keys].T
        sc = sc.masked_fill(mask, float("-inf"))
        out = sc.softmax(-1) @ v[keys]
        # legacy reference: the token's own row in order, diag = first occurrence of its diag block
        ids = grp["rows"][s][: grp["nv"][s]]
        rk = []
        rm = []
        dfound = False
        for b in ids:
            rk.append(torch.arange(b * BS, (b + 1) * BS))
            m = torch.zeros(BS, dtype=torch.bool)
            if causal and not dfound:
                p = chunk_start + t
                if b == p // BS:
                    m = torch.arange(BS) >= (p % BS + 1)
                    dfound = True
            rm.append(m)
        rk = torch.cat(rk)
        rm = torch.cat(rm)
        rs = ((q[t] * scale) @ k[rk].T).masked_fill(rm, float("-inf"))
        ref = rs.softmax(-1) @ v[rk]
        err = (out - ref).abs().max().item()
        if not err < 1e-9:
            errs.append((t, err))
    return errs


def run_case(name, ids_lin, S, n_kv, G, chunk_start, causal, ncores=130, T=None, seed=0):
    torch.manual_seed(seed)
    nb = (T or ((max(max(i for i in r if i != SENT) for r in ids_lin) + 1) * BS)) // BS
    T = nb * BS
    q = torch.randn(n_kv * S, 16, 64, dtype=torch.float64)  # per work item (kv, tok)
    k = torch.randn(T, 64, dtype=torch.float64)
    v = torch.randn(T, 64, dtype=torch.float64)
    total = S * n_kv
    base, extra = total // ncores, total % ncores
    st = 0
    ngroups = 0
    nblk = 0
    nstep = 0
    bad = []
    gsz = {}
    for c in range(ncores):
        cnt = base + (1 if c < extra else 0)
        for grp in reader_groups(ids_lin, st, cnt, S, G, chunk_start, causal=causal):
            ngroups += 1
            nblk += len(grp["uid"])
            gsz[grp["g"]] = gsz.get(grp["g"], 0) + 1
            for e in range(len(grp["uid"])):
                for r in range((grp["g"] + 1) // 2):
                    if (grp["umask"][e] >> (2 * r)) & 3:
                        nstep += 1
            # per kv-group tensors: q indexed by linear work item
            bad += check_group(grp, q[grp["kv"] * S : (grp["kv"] + 1) * S], k, v, 0.1, chunk_start, causal)
        st += cnt
    sel = sum(min(len([i for i in r if i != SENT]), 16) or 1 for r in ids_lin)
    print(
        f"[{name}] G={G}: groups={ngroups} sizes={dict(sorted(gsz.items()))} blocks={nblk} ({nblk/sel:.3f} of legacy) "
        f"row-steps={nstep} ({nstep/sel:.3f}) mismatches={len(bad)} {bad[:3]}"
    )
    return len(bad) == 0


if __name__ == "__main__":
    ok = True
    cap = torch.load("/mnt/data/kernel-agent/dev/prefill/runs/c5120-v2/dump/cuts_ev/msa_block_ids.pt")
    for layer, rank in ((30, 0), (3, 3), (59, 1)):
        rows = cap[layer]["block_ids"][0][rank * 1280 : (rank + 1) * 1280].tolist()
        rows = [[(i if 0 <= i < 0xFFFFFFF0 else SENT) for i in r] for r in rows]
        for G in (2, 4, 8):
            ok &= run_case(f"L{layer} r{rank}", rows, 1280, 1, G, 51200 + rank * 1280, True, T=56320)
    # synthetic random causal (bench pattern)
    gen = torch.Generator().manual_seed(7)
    S = 1280
    q0 = 51200
    rows = []
    for s in range(S):
        p = q0 + s
        local = p // BS
        pool = torch.randperm(local, generator=gen)[:15]
        rows.append(torch.cat([pool, torch.tensor([local])]).sort().values.tolist())
    for G in (2, 8):
        ok &= run_case("synthetic", rows, S, 1, G, q0, True, T=56320)
    # adversarial: non-causal random (no common block), duplicates, sentinel tails, GQA with odd S, straddling
    rnd = random.Random(3)
    for trial in range(30):
        S = rnd.choice([1, 2, 3, 5, 8, 9, 12, 33])
        n_kv = rnd.choice([1, 2, 4])
        nblk = rnd.choice([16, 32])
        causal = rnd.random() < 0.5
        cs = rnd.choice([0, 128, 1000, 4064]) if causal else 0
        if causal:
            nblk = max(nblk, (cs + S) // BS + 1)
        rows = []
        for item in range(S * n_kv):
            t = item % S
            if causal:
                db = (cs + t) // BS
                cand = list(range(db + 1))
            else:
                cand = list(range(nblk))
            n = rnd.randint(1, min(16, len(cand)))
            r = rnd.sample(cand, n)
            if causal and rnd.random() < 0.8 and db not in r:
                r[0] = db
            if rnd.random() < 0.2 and n > 1:
                r[-1] = r[0]  # duplicate
            if rnd.random() < 0.5:
                rnd.shuffle(r)
            rows.append(r + [SENT] * (16 - len(r)))
        for G in (2, 4, 8):
            ok &= run_case(
                f"adv{trial} S={S} nkv={n_kv} causal={causal}",
                rows,
                S,
                n_kv,
                G,
                cs,
                causal,
                ncores=rnd.choice([1, 3, 130]),
                T=nblk * BS,
            )
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
