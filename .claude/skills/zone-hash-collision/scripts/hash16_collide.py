#!/usr/bin/env python3
"""Zone-collision analysis at two granularities:
  (A) full source-location string  "NAME,file,line,KERNEL_PROFILER"  (what tt-metal hashes today)
  (B) name-only  field[0]          (the 'hash the semantic name only' proposal)
Reports distinct counts + 16-bit collisions for each, using the exact hash16CT.
Usage: hash16_v2.py <new_zone_src_locations.log> [more...]
"""
import sys, collections
DELIM = "'#pragma message: "

def hash32(s):
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h

def hash16(s):
    r = hash32(s)
    return ((r & 0xFFFF) ^ ((r & 0xFFFF0000) >> 16)) & 0xFFFF

def extract(paths):
    out = []
    for p in paths:
        for line in open(p, errors="replace"):
            i = line.find(DELIM)
            if i == -1: continue
            payload = line[i+len(DELIM):].rstrip("\n")
            if payload.endswith("'"): payload = payload[:-1]
            out.append(payload)
    return out

def collisions_for(keys):
    """keys: iterable of hashable strings to hash. Returns (distinct, occupied, collision_pairs)."""
    seen=set(); hmap={}; col=[]
    for k in keys:
        if k in seen: continue
        seen.add(k)
        h=hash16(k)
        if h in hmap: col.append((h,hmap[h],k))
        else: hmap[h]=k
    return len(seen), len(hmap), col

def main():
    paths=sys.argv[1:]
    strings=extract(paths)
    names=[s.split(",",1)[0] for s in strings]

    ds,os_,cs = collisions_for(strings)
    dn,on_,cn = collisions_for(names)

    print(f"files: {paths}")
    print(f"total #pragma lines parsed: {len(strings)}")
    print("=== (A) FULL STRING  (current tt-metal behavior) ===")
    print(f"  distinct strings: {ds}")
    print(f"  16-bit collisions: {len(cs)}")
    for h,a,b in cs:
        print(f"    0x{h:04x}: kept={a}")
        print(f"             drop={b}")
    print("=== (B) NAME ONLY  (proposed: hash field[0] only) ===")
    print(f"  distinct names: {dn}")
    print(f"  16-bit collisions: {len(cn)}")
    for h,a,b in cn:
        print(f"    0x{h:04x}: {a}  <->  {b}")
    # how many distinct strings share each name (churn factor)
    per=collections.Counter(names)
    top=per.most_common(8)
    print("=== name -> #distinct-strings (top multiplicity) ===")
    for nm,c in top: print(f"  {c:3d} x  {nm}")
    import statistics
    mult=list(per.values())
    print(f"  mean strings-per-name={statistics.mean(mult):.2f}  max={max(mult)}")

if __name__=="__main__": main()
