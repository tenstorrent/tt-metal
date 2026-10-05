"""python cmp_ids.py <tagA> <tagB> [S=4096 C=1024 layers=2,3,8,9,14,15] -- top-512 selection sets saved by tests/test_prefill_sparse_device.py (DSV41_PS_TAG) of two runs: per layer,
mean / min set overlap over the queries with > 512 visible entries and the share of queries with an identical set, plus the output PCC between the two runs."""
import sys

import torch

a, b = sys.argv[1], sys.argv[2]
S, C = 4096, 1024
layers = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "2,3,8,9,14,15").split(",")]
for L in layers:
    f = lambda t: torch.load(f"/mnt/tt-data/ssinghal/dsv4-logs/h46x_dev_S{S}_C{C}_auto_L{L}{t}.pt")
    A, B = f(a), f(b)
    ia, ib = A["ids"], B["ids"]
    ov, same = [], 0
    qs = [i for i in range(S) if (ia[i] >= 0).sum() > 0]
    for i in qs:
        x, y = set(ia[i][ia[i] >= 0].tolist()), set(ib[i][ib[i] >= 0].tolist())
        ov.append(len(x & y) / max(1, len(x)))
        same += x == y
    ga, gb = A["got"].float(), B["got"].float()
    pcc = torch.corrcoef(torch.stack([ga.flatten(), gb.flatten()]))[0, 1].item()
    print(
        f"layer {L}: {len(qs)} queries with selections, overlap mean {sum(ov) / len(ov):.4f} min {min(ov):.4f}, identical sets {same / len(qs):.3f}; output PCC(A,B) {pcc:.6f}"
    )
