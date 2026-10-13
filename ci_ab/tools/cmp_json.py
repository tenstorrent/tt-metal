# Compare two variants' case hashes (all.json): identical, differing and missing cases.
import json, sys
a, b = (json.load(open(f"{d}/all.json")) for d in sys.argv[1:3])
diff = sorted(k for k in a if k in b and a[k] != b[k]); miss = sorted(set(a) ^ set(b))
print(f"cases {len(a)} / {len(b)}: identical {sum(1 for k in a if b.get(k) == a[k])}, differ {len(diff)}, missing {len(miss)}")
for k in diff[:30]: print("  DIFF", k)
for k in miss[:10]: print("  MISSING", k)
