"""Simulate the vol2col_rm CB pointer the way the reader pushes and compute pops it (#149, #152).

cb_push_back / cb_pop_front (dataflow_api.h) add the pages and wrap only when the pointer == limit.
Each block pushes 32-page chunks plus a num_patches % 32 tail; compute pops the same sizes.
old sizing (t149 and before): min(n, 32) pages if n % 32 == 0, else min(n, 64).
new sizing (t152): min(n, 32) pages if n % 32 == 0, else n, so every block ends at the CB end.
Prints old vs new for the job 273-285 blockings and checks n = 1..2048 never straddle with the new sizing.
Run: python tt-project/t149/sim_vol2col_cb.py (CPU only, no imports).
"""


def old_pages(n):
    return min(n, 32) if n % 32 == 0 else min(n, 64)


def new_pages(n):
    return min(n, 32) if n % 32 == 0 else n


def first_overrun(n, pages, blocks=300):
    ptr = 0
    for b in range(blocks):
        left = n
        while left:
            chunk = min(left, 32)
            ptr += chunk
            if ptr == pages:
                ptr = 0
            elif ptr > pages:
                return b, ptr
            left -= chunk
    return None


def fmt(n, pages):
    r = first_overrun(n, pages)
    return f"{pages:4d} pages ok" if r is None else f"{pages:4d} pages OVERRUN in block {r[0]}: ptr {r[1]}"


for t, h, w in [(5, 4, 4), (5, 8, 2), (7, 4, 4), (7, 8, 2), (3, 8, 8), (3, 16, 4), (3, 4, 4), (3, 8, 2), (6, 8, 2)]:
    n = t * h * w
    print(f"{(t, h, w)!s:12} n={n:4d}  old: {fmt(n, old_pages(n)):40}  new: {fmt(n, new_pages(n))}")

old_bad = [n for n in range(1, 2049) if first_overrun(n, old_pages(n))]
new_bad = [n for n in range(1, 2049) if first_overrun(n, new_pages(n))]
print(f"n=1..2048: old sizing straddles for {len(old_bad)} counts (first {old_bad[:5]}), new sizing for {len(new_bad)}")
assert all(n > 64 and n % 32 for n in old_bad)
assert new_bad == []
