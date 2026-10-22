# Scratch, not for merge. Hang and wrong-output stress for ttnn.experimental.topk_large_indices.
# usage: python stress.py glm|follow [--iters N] [--cases a,b] [--mixed] [--seed S] [--deadline SEC]
import argparse, os, sys, time, random, importlib.util
import numpy as np
import torch
import ttnn

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["glm", "follow", "smoke"])
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--cases", default="")
ap.add_argument("--mixed", action="store_true")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--deadline", type=float, default=1e9)
ap.add_argument("--variants", type=int, default=3)
args = ap.parse_args()
T0 = time.time()
BAD = []


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# (name, rows, n, k, valid_length)
CASES = [
    ("r160_n524288_k2048", 160, 524288, 2048, None),
    ("r160_n102400_k2048", 160, 102400, 2048, None),
    ("r640_n51200_k1536", 640, 51200, 1536, None),
    ("bounded_r2_n102400_k1536_vl56320", 2, 102400, 1536, 56320),
    ("r3_n102400_k2048", 3, 102400, 2048, None),
    ("r37_n102400_k2048", 37, 102400, 2048, None),
    ("r111_n102400_k2048", 111, 102400, 2048, None),
    ("r161_n102400_k2048", 161, 102400, 2048, None),
    ("r161_n32868_k1024", 161, 32768 + 100, 1024, None),
    ("r37_n102400_k2048_vl77777", 37, 102400, 2048, 77777),
    ("r160_n102400_k2048_vl50000", 160, 102400, 2048, 50000),
    ("r111_n65536_k512_vl40001", 111, 65536, 512, 40001),
]


def make_case(rng, rows, n, k, vl):
    lim = vl if vl is not None else n
    arr = np.zeros((rows, n), dtype=np.uint16)
    if lim < n:
        arr[:, lim:] = 0x5000  # stale tail larger than any winner; must be ignored
    pos = np.empty((rows, k), dtype=np.int64)
    perm = np.empty((rows, k), dtype=np.int64)
    for r in range(rows):
        pos[r] = rng.choice(lim, k, replace=False)
        perm[r] = rng.permutation(k)
        arr[r, pos[r]] = (0x3F80 + perm[r]).astype(np.uint16)
    exp = np.empty((rows, k), dtype=np.int64)
    np.put_along_axis(exp, k - 1 - perm, pos, axis=1)
    t = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return t, exp


def to_dev(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)


def run_topk(x, k, vl):
    return ttnn.experimental.topk_large_indices(x, k=k, valid_length=vl) if vl is not None else ttnn.experimental.topk_large_indices(x, k=k)


def check_topk(name, out, exp, tag):
    got = ttnn.to_torch(out, dtype=torch.uint32).to(torch.int64).numpy()
    if got.shape != exp.shape or not np.array_equal(got, exp):
        nbad = int((got != exp).any(axis=1).sum()) if got.shape == exp.shape else -1
        log(f"MISMATCH {name} {tag} rows_bad={nbad}")
        BAD.append((name, tag))
        return False
    return True


dev = ttnn.open_device(device_id=0)
dev.enable_program_cache()
log("device open", "watcher=", os.environ.get("TT_METAL_WATCHER"), "llk_asserts=", os.environ.get("TT_METAL_LLK_ASSERTS"))
rng = np.random.default_rng(args.seed)
random.seed(args.seed)


def prepare(sel):
    prepared = []
    for (name, rows, n, k, vl) in CASES:
        if sel and name not in sel:
            continue
        vs = []
        for v in range(args.variants):
            t, exp = make_case(rng, rows, n, k, vl)
            vs.append((to_dev(t, dev), exp))
            del t
        prepared.append((name, k, vl, vs))
        log("prepared", name)
    return prepared


def glm():
    sel = set(args.cases.split(",")) if args.cases else None
    prep = prepare(sel)
    counts = {p[0]: 0 for p in prep}
    modes = [None]
    if args.mixed:
        modes = ["split", "whole"]
        e0 = dev.num_program_cache_entries()
        x, _ = prep[0][3][0]
        os.environ.pop("TOPK_FORCE_WHOLE", None)
        run_topk(x, prep[0][1], prep[0][2]); e1 = dev.num_program_cache_entries()
        os.environ["TOPK_FORCE_WHOLE"] = "1"
        run_topk(x, prep[0][1], prep[0][2]); e2 = dev.num_program_cache_entries()
        os.environ.pop("TOPK_FORCE_WHOLE", None)
        log(f"mixed probe: cache entries {e0} -> split {e1} -> whole {e2} (whole must add an entry when the override exists)")
    for it in range(args.iters):
        if time.time() - T0 > args.deadline:
            log(f"deadline reached after {it} iterations"); break
        order = prep[:]
        random.shuffle(order)
        for (name, k, vl, vs) in order:
            x, exp = vs[it % len(vs)]
            for m in modes:
                if m == "whole":
                    os.environ["TOPK_FORCE_WHOLE"] = "1"
                else:
                    os.environ.pop("TOPK_FORCE_WHOLE", None)
                out = run_topk(x, k, vl)
                check_topk(name, out, exp, f"it{it}{'/' + m if m else ''}")
                counts[name] += 1
        if it % 20 == 0:
            log(f"iter {it} ok-so-far bad={len(BAD)}")
    os.environ.pop("TOPK_FORCE_WHOLE", None)
    for n_, c in counts.items():
        log(f"COUNT {n_} {c}")


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def follow():
    spec = importlib.util.spec_from_file_location(
        "tis", os.path.join(os.environ.get("TT_METAL_HOME", "."), "tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py"))
    tis = importlib.util.module_from_spec(spec)
    have_idx = True
    try:
        spec.loader.exec_module(tis)
    except Exception as e:  # noqa
        log("indexer_score helpers unavailable", repr(e)[:200]); have_idx = False
    g = torch.Generator().manual_seed(args.seed)
    R = 130 * 32
    a_i = torch.randint(-8, 8, (1, 1, R, 1024), generator=g).to(torch.bfloat16)
    b_i = torch.randint(-8, 8, (1, 1, R, 1024), generator=g).to(torch.bfloat16)
    mm_a = (torch.randint(-1, 2, (1, 1, 2048, 512), generator=g)).to(torch.bfloat16)
    mm_b = (torch.randint(-1, 2, (1, 1, 512, 1024), generator=g)).to(torch.bfloat16)
    sm_x = torch.randn(1, 1, R, 512, generator=g).to(torch.bfloat16)
    rm_x = torch.randn(1, 1, R, 512, generator=g).to(torch.bfloat16)
    tk_vals = (0x3F80 + torch.stack([torch.randperm(4096, generator=g) for _ in range(32)])).to(torch.int32)
    tk_x = (tk_vals.to(torch.int16)).view(torch.bfloat16).reshape(1, 1, 32, 4096)
    TL = ttnn.TILE_LAYOUT
    d = lambda t, lay=TL: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=dev)
    ta, tb = d(a_i), d(b_i)
    tma, tmb = d(mm_a), d(mm_b)
    tsm, trm = d(sm_x), d(rm_x)
    ttk = d(tk_x)
    trow = d(rm_x, ttnn.ROW_MAJOR_LAYOUT)
    ex_add = a_i + b_i
    ex_mm = (mm_a.float() @ mm_b.float())
    ex_sm = torch.softmax(sm_x.float(), -1)
    ex_sum = rm_x.float().sum(-1, keepdim=True)
    ex_tkv, ex_tki = torch.topk(tk_x.float(), 32, dim=-1)

    def f_add():
        return torch.equal(ttnn.to_torch(ttnn.add(ta, tb)), ex_add)

    def f_mm():
        return pcc(ttnn.to_torch(ttnn.matmul(tma, tmb)), ex_mm) > 0.999

    def f_sm():
        return pcc(ttnn.to_torch(ttnn.softmax(tsm, dim=-1)), ex_sm) > 0.999

    def f_tilize():
        rt = ttnn.to_layout(trow, TL)
        back = ttnn.to_layout(rt, ttnn.ROW_MAJOR_LAYOUT)
        return torch.equal(ttnn.to_torch(back), rm_x) and torch.equal(ttnn.to_torch(rt), rm_x)

    def f_reduce():
        return pcc(ttnn.to_torch(ttnn.sum(trm, dim=-1, keepdim=True))[..., :1], ex_sum) > 0.999

    def f_topk():
        v, i = ttnn.topk(ttk, 32, dim=-1)
        return torch.equal(ttnn.to_torch(v).float(), ex_tkv) and torch.equal(ttnn.to_torch(i).to(torch.int64), ex_tki)

    fl = {"add": f_add, "matmul": f_mm, "softmax": f_sm, "tilize_untilize": f_tilize, "reduce": f_reduce, "ttnn_topk": f_topk}
    if have_idx:
        q, k_, w = tis.make_inputs(8, 64, 64, 256)
        iq, ik, iw = tis.to_device(q, dev), tis.to_device(k_, dev), tis.to_device(w, dev)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
        iref = tis.indexer_score_dsa_ref(q, k_, w, 192)

        def f_idx():
            out = ttnn.experimental.indexer_score_dsa(iq, ik, iw, chunk_start_idx=192, program_config=cfg)
            try:
                tis.assert_indexer_match(ttnn.to_torch(out), iref, 64, 256, check_neg=True)
                return True
            except AssertionError:
                return False
        fl["indexer_score"] = f_idx

    tcases = {}
    for (name, rows, n, k, vl) in CASES:
        if name in ("r160_n102400_k2048", "r37_n102400_k2048_vl77777", "r640_n51200_k1536", "r161_n32868_k1024",
                    "r3_n102400_k2048", "bounded_r2_n102400_k1536_vl56320", "r160_n524288_k2048"):
            t, exp = make_case(rng, rows, n, k, vl)
            tcases[name] = (to_dev(t, dev), k, vl, exp)
    tnames = list(tcases)
    stats = {}

    def T(name):
        x, k, vl, exp = tcases[name]
        ok = check_topk(name, run_topk(x, k, vl), exp, "follow")
        stats[("topk", name)] = stats.get(("topk", name), 0) + 1
        return ok

    def F(name):
        ok = fl[name]()
        stats[("follower", name)] = stats.get(("follower", name), 0) + 1
        if not ok:
            log(f"MISMATCH follower {name}"); BAD.append((name, "follower"))
        return ok

    for name in fl:  # baseline
        F(name)
    log("baseline followers done", len(BAD))
    n_order = max(1, args.iters // 5)
    for it in range(n_order):  # topk first, then each follower on the same cores
        for tn in tnames:
            T(tn)
            for fn in fl:
                F(fn)
        if time.time() - T0 > args.deadline: break
    log("topk-first done", len(BAD))
    for it in range(n_order):  # topk last: follower then topk
        for fn in fl:
            F(fn)
            for tn in random.sample(tnames, 2):
                T(tn)
        if time.time() - T0 > args.deadline: break
    log("topk-last done", len(BAD))
    ops = [("T", n) for n in tnames] + [("F", n) for n in fl]
    for it in range(args.iters):  # interleaved
        seq = random.choices(ops, k=14)
        for kind, n in seq:
            (T if kind == "T" else F)(n)
        if time.time() - T0 > args.deadline: log("deadline"); break
    log("interleaved done", len(BAD))
    for it in range(args.iters):  # program cache switch: different shapes back to back
        a, b = random.sample(tnames, 2)
        T(a); T(b); T(a)
    for (kind, n), c in sorted(stats.items()):
        log(f"COUNT {kind} {n} {c}")


try:
    if args.mode == "glm":
        glm()
    elif args.mode == "follow":
        follow()
    else:
        args.iters = 2; args.variants = 1; args.cases = "r160_n102400_k2048,r37_n102400_k2048_vl77777"; glm()
finally:
    ttnn.close_device(dev)
log(f"DONE bad={len(BAD)} {BAD[:10]}")
sys.exit(1 if BAD else 0)
