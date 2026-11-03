"""ranges.py ELF: per value live in E1's pass loop: the share of the pass loop's instructions where it is live, in a
register or a slot, from loopvars.liveness at every pass instruction."""
import sys, collections, re
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L
e = sys.argv[1]
fns = L.functions(e); name, _ = L.pick(fns, "receiver"); body = fns[name]
r0 = L.liveness(e, "receiver")
# all pass addresses: recompute the pass set the same way, by asking for every body address and keeping the counted ones
r = L.liveness(e, "receiver", at=[a for a, *_ in body])
passes = r0["pass_insns"]
cnt = collections.Counter(); where = collections.defaultdict(set)
for d in r["at"]:
    for v in d["values"]:
        key = (v["names"][0].split("~")[0] if v["names"] else v["origin"][:70])
        key = re.sub(r"@\w+<\d+>|<\d+(,\d+)*>", "", key)
        cnt[key] += 1; where[key].add(v["kind"])
tot = len(r["at"])
print(f"{e.split('/')[-4] if '/builds/' in e else e}: {tot} instructions in the function; pass loop {passes}")
for k, n in cnt.most_common(45):
    print(f"  {100*n/tot:5.1f}%  {'/'.join(sorted(where[k])):12s} {k}")
