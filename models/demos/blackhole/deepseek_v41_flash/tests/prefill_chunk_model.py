"""CPU model of the 40-layer prefill TTFT of ONE prompt (device time only, all slots of its window busy with similar prompts) vs the chunk size per slot.
python tests/prefill_chunk_model.py
Input: measured pure device time of the captured traces on a 4-layer A (+embedding) / 2-layer B (+ragged head) subset (dsv4-logs/prefill_opt/cs2_*), scaled to 21 / 19 layers:
  TA(n) = EMB + 21/4 (A4(n) - EMB), TB(n) = HEAD + 19/2 (B4(n) - HEAD)      (EMB 2.6 ms embedding, HEAD 6.4 ms ragged head + sampling per B window)
Prompt of S tokens, replay zone R=2560 (OPTIMIZATIONS 8.6): windows that end at or before p0 = floor((S-R)/C)*C run trace A only, the others A + B; the last window is padded to a
full chunk (cost of a full window). The 'dynamic' rule picks the largest chunk that still ends at or before p0' = floor((S-R)/512)*512 (A-only), then 512 chunks for the rest."""
import math

EMB, HEAD, R = 2.6, 6.4, 2560
# (A4, B4) ms per window, key = row-tokens = U * C
DATA = {
    2: {1024: (48.5, 36.2), 2048: (88.7, 60.0), 4096: (163.8, 103.8)},  # Up=2 (B=8): C = 512, 1024, 2048
    1: {512: (28.8, 19.4), 1024: (49.0, 30.3), 2048: (85.7, 51.0)},  # Up=1 (B=4 / B=16): C = 512, 1024, 2048
}
EXTRA = {}  # filled from the C=4096 runs: {Up: {row_tokens: (A4, B4)}}


def ta(up, n):
    d = {**DATA[up], **EXTRA.get(up, {})}
    return EMB + 21 / 4 * (d[n][0] - EMB)


def tb(up, n):
    d = {**DATA[up], **EXTRA.get(up, {})}
    return HEAD + 19 / 2 * (d[n][1] - HEAD)


def fixed_chunk(S, C, up):
    """TTFT (s) with one chunk size C per slot."""
    n = up * C
    p0 = max(0, (S - R) // C * C)
    nw = math.ceil(S / C)
    nz = math.ceil((S - p0) / C)
    return ((nw - nz) * ta(up, n) + nz * (ta(up, n) + tb(up, n))) / 1e3


def dynamic(S, up, sizes):
    """sizes: descending chunk sizes allowed (all multiples of 512, the last one 512). Big chunks while the window ends at or before p0 (A only), then smaller, zone at 512."""
    p0 = max(0, (S - R) // 512 * 512)
    pos, t = 0, 0.0
    for C in sizes[:-1]:
        while pos + C <= p0:
            t += ta(up, up * C)
            pos += C
    C = sizes[-1]
    while pos < S:  # 512 windows: A-only until p0, then A + B (the last one padded)
        t += ta(up, up * C) + (tb(up, up * C) if pos + C > p0 else 0.0)
        pos += C
    return t / 1e3


if __name__ == "__main__":
    for up, name in ((2, "B=8 (Up=2, loaded)"), (1, "B=16 / B=4 (Up=1)")):
        cs = [c for c in (512, 1024, 2048, 4096) if up * c in {**DATA[up], **EXTRA.get(up, {})}]
        print(f"\n== {name}: device-time TTFT (s) of one S-token prompt, chunk C per slot")
        print("S        " + "".join(f"C={c:<8d}" for c in cs) + "dyn(2048,512)  dyn(1024,512)  dyn(4096,1024,512)")
        for S in (8192, 32768, 131072, 8192 + 300, 32768 + 300, 131072 + 300):
            base = fixed_chunk(S, 512, up)
            row = f"{S:<8d} " + "".join(
                f"{fixed_chunk(S, c, up):6.2f} {100 * (fixed_chunk(S, c, up) / base - 1):+5.1f}% " for c in cs
            )
            dd = []
            for sz in ((2048, 512), (1024, 512), (4096, 1024, 512)):
                if all(up * c in {**DATA[up], **EXTRA.get(up, {})} for c in sz):
                    v = dynamic(S, up, list(sz))
                    dd.append(f"{v:6.2f} {100 * (v / base - 1):+5.1f}%")
                else:
                    dd.append("   n/a      ")
            print(row + "  ".join(dd))
