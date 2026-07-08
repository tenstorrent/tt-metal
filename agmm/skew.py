import csv, sys, statistics
from collections import defaultdict

OP = "AllGatherMinimalMatmulAsyncOp"

def analyze(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        cols = {c.strip(): c for c in r.fieldnames}
        gc = cols["GLOBAL CALL COUNT"]
        dev = cols["DEVICE ID"]
        kd = cols["DEVICE KERNEL DURATION [ns]"]
        opc = cols["OP CODE"]
        # shape columns for M x K x N (input0 Y=M, input1 X=N, input0 X=K)
        i0y = cols["INPUT_0_Y_PAD[LOGICAL]"]
        i0x = cols["INPUT_0_X_PAD[LOGICAL]"]
        i1x = cols["INPUT_1_X_PAD[LOGICAL]"]
        for row in r:
            if row[opc].strip() != OP:
                continue
            rows.append({
                "gc": int(row[gc]),
                "dev": int(row[dev]),
                "kd": float(row[kd]),
                "M": row[i0y].strip(), "K": row[i0x].strip(), "N": row[i1x].strip(),
            })

    # per device, order AGMM ops by global call count -> instance index
    perdev = defaultdict(list)
    for x in rows:
        perdev[x["dev"]].append(x)
    for d in perdev:
        perdev[d].sort(key=lambda z: z["gc"])

    ndev = len(perdev)
    counts = {len(v) for v in perdev.values()}
    assert len(counts) == 1, f"uneven op counts per device: {counts}"
    ninst = counts.pop()

    print(f"\n=== {path.split('/')[-1]} ===")
    print(f"devices={ndev}  AGMM instances per device={ninst}")
    print(f"{'inst':>4} {'shape (MxKxN)':>22} {'min us':>8} {'max us':>8} {'mean us':>8} "
          f"{'max-min us':>10} {'skew x':>7} {'slow dev':>8} {'fast dev':>8}")
    print("-"*100)

    ratios = []
    abs_gaps = []
    for i in range(ninst):
        insts = [perdev[d][i] for d in perdev]
        durs = [(x["kd"], x["dev"]) for x in insts]
        durs.sort()
        fast, fdev = durs[0]
        slow, sdev = durs[-1]
        mean = statistics.mean(d for d, _ in durs)
        shape = f"{insts[0]['M']}x{insts[0]['K']}x{insts[0]['N']}"
        ratio = slow / fast if fast > 0 else float('nan')
        ratios.append(ratio)
        abs_gaps.append((slow - fast) / 1000)
        print(f"{i:>4} {shape:>22} {fast/1000:>8.1f} {slow/1000:>8.1f} {mean/1000:>8.1f} "
              f"{(slow-fast)/1000:>10.1f} {ratio:>6.2f}x {sdev:>8} {fdev:>8}")

    print("-"*100)
    print(f"skew ratio (slow/fast): min={min(ratios):.2f}x  median={statistics.median(ratios):.2f}x  "
          f"mean={statistics.mean(ratios):.2f}x  max={max(ratios):.2f}x")
    print(f"absolute gap (max-min): min={min(abs_gaps):.1f}us  median={statistics.median(abs_gaps):.1f}us  "
          f"max={max(abs_gaps):.1f}us")

for p in sys.argv[1:]:
    analyze(p)
