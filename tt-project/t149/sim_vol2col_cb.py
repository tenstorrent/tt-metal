"""Simulate the vol2col_rm CB write pointer the way the reader and cb_push_back move it (#149).

cb_push_back (dataflow_api.h) adds the pushed pages and wraps only when wr_ptr == limit.
The factory sizes the CB at min(n, 32) pages if n % 32 == 0, else min(n, 64) pages.
Prints the first block whose push leaves wr_ptr past the CB end, or "ok".
"""


def first_overrun(n, blocks=300):
    pages = min(n, 32) if n % 32 == 0 else min(n, 64)
    wr = 0
    for b in range(blocks):
        left = n
        while left:
            chunk = min(left, 32)
            wr += chunk
            if wr == pages:
                wr = 0
            elif wr > pages:
                return b, pages, wr
            left -= chunk
    return None


for t, h, w in [(5, 4, 4), (5, 8, 2), (7, 4, 4), (7, 8, 2), (3, 8, 8), (3, 16, 4), (3, 4, 4), (3, 8, 2), (6, 8, 2)]:
    r = first_overrun(t * h * w)
    print((t, h, w), t * h * w, "ok" if r is None else f"overrun in block {r[0]}: wr_ptr {r[2]} > {r[1]} pages")
