# Compare two elf_hash_cfg.py listings where one key (kernel, generated-files hash, ELF) can hold several builds that differ
# only in compile-time arguments: the key matches when the sorted lists of load-image hashes are equal, and the count of
# builds whose image has an identical counterpart is reported. usage: cmp_multi.py <main.txt> <branch.txt>
import collections, sys
def load(p):
    d = collections.defaultdict(list)
    for l in open(p):
        k, h = l.split()[:2]; d[k].append(h)
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
per = collections.defaultdict(collections.Counter)
for k in sorted(set(a) | set(b)):
    kern = k.split("/")[0]
    ca, cb = collections.Counter(a.get(k, [])), collections.Counter(b.get(k, []))
    same = sum((ca & cb).values()); n = max(sum(ca.values()), sum(cb.values()))
    per[kern]["builds"] += n; per[kern]["identical"] += same
    if same < n:
        per[kern]["differ"] += n - same
        print("DIFF", k, f"{same}/{n}")
tot = collections.Counter()
for kern, c in sorted(per.items()):
    tot.update(c); print(f"{kern}: {c['identical']} of {c['builds']} builds identical" + ("   <--" if c["differ"] else ""))
print("TOTAL", f"{tot['identical']} of {tot['builds']} builds identical, {tot['differ']} differ")
