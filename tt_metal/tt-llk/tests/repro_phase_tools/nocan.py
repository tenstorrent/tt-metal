import sys


def n(v):
    try:
        return int(v, 2)
    except:
        return 0


f = open(sys.argv[1])
keys = f.readline().split()
ix = {k: i for i, k in enumerate(keys)}
rd = []
noc = []
reg = []
noc_in = []
for line in f:
    v = line.split()
    rd.append(n(v[ix["xb_rden"]]))
    noc.append(n(v[ix["noc_l1_rden"]]))
    noc_in.append(n(v[ix["noc_l1_in_rden"]]))
    reg.append(n(v[ix["noc_reg_rden"]]))
act = [i for i, x in enumerate(rd) if x]
a, b = act[0], act[-1]
print("cycles", len(rd), "packing window", a, b, b - a)
for lab, arr in (
    ("NoC reads from L1 (o_mem_out_rden)", noc),
    ("NoC mem_in rden", noc_in),
    ("NoC register reads", reg),
):
    tot = [i for i, x in enumerate(arr) if x]
    inside = [i for i in tot if a <= i <= b]
    print(
        lab,
        "total",
        len(tot),
        "inside packing",
        len(inside),
        "first/last",
        tot[:3],
        tot[-3:],
    )
    # gaps of activity clusters
    cl = []
    for i in tot:
        if not cl or i - cl[-1][1] > 50:
            cl.append([i, i])
        else:
            cl[-1][1] = i
    print("  clusters", len(cl), [(x, y) for x, y in cl[:12]])
