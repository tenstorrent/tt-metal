# t41: fraction of ring-joint SDPA work left with a temporal band, at chunk granularity.
# Stage 2, 1080p/145f, 4x8: N=38760 real (38912 padded), SP=8 shard 4864, q_chunk 192, k_chunk 512.
# Work unit = (q chunk, k chunk) pair; a pair is skipped when no key in the k chunk is in the band
# of any query in the q chunk. Also the ring hops each device still needs (K/V gather distance).
N, NP, SP, TPF, QC, KC = 38760, 38912, 8, 2040, 192, 512
S = NP // SP
for W in (1, 2, 3, 4, 6):
    per_dev, hops = [], []
    for d in range(SP):
        done = total = 0
        need = set()
        for q0 in range(d * S, (d + 1) * S, QC):
            q1 = min(q0 + QC, (d + 1) * S, N)
            if q0 >= N:
                break
            lo = max(q0 // TPF - W, 0) * TPF
            hi = min(((q1 - 1) // TPF + W + 1) * TPF, N)
            for src in range(SP):
                for k0 in range(src * S, (src + 1) * S, KC):
                    k1 = min(k0 + KC, (src + 1) * S)
                    total += 1
                    if k0 < hi and k1 > lo and k0 < N:
                        done += 1
                        need.add(src)
        per_dev.append(done / total)
        hops.append(max(abs(s - d) for s in need))
    print(
        f"W{W}: work max {max(per_dev):.3f} mean {sum(per_dev)/SP:.3f} | per-dev "
        + " ".join(f"{x:.2f}" for x in per_dev)
        + f" | max hops {max(hops)}"
    )
