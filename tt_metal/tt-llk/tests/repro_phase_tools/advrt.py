import sys

sys.argv = [sys.argv[0], "/tmp/adv"] + sys.argv[1:]
exec(open("/tmp/advan.py").read().split("for p in sys.argv[2:]")[0])
from collections import Counter

for p in sys.argv[2:]:
    a, b = p.split(">")
    A, B = get(a), get(b)
    ks = [k for k in A if k in B and A[k] and abs(B[k] / A[k] - 1) > 0.02]
    print(p, len(ks), dict(Counter(k[-1][5:-1] for k in ks)))
