#!/usr/bin/env python3
"""Write a human-readable zone-hash collision report from a zone-src log.
Usage: report_collisions.py <new_zone_src_locations.log> <out.txt>
Shows, using the exact tt-metal hash16CT:
  (A) FULL-STRING hashing (current default): distinct strings, collisions, raw lines
  (B) NAME-ONLY hashing (proposed opt-in):  distinct names, collisions
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

def extract(path):
    out = []
    for line in open(path, errors="replace"):
        i = line.find(DELIM)
        if i == -1:
            continue
        payload = line[i + len(DELIM):].rstrip("\n")
        if payload.endswith("'"):
            payload = payload[:-1]
        out.append(payload)
    return out

def find_collisions(keys):
    seen, hmap, col = set(), {}, []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        h = hash16(k)
        if h in hmap:
            col.append((h, hmap[h], k))
        else:
            hmap[h] = k
    return len(seen), col

def main():
    log, out = sys.argv[1], sys.argv[2]
    strings = extract(log)
    names = [s.split(",", 1)[0] for s in strings]
    ds, cs = find_collisions(strings)
    dn, cn = find_collisions(names)

    L = []
    L.append("GPT-OSS-120B global@128 fused layer — device-profiler zone-hash collision report")
    L.append(f"source log : {log}")
    L.append(f"raw #pragma lines parsed: {len(strings)}  (same zone repeats across cores/RISCs)")
    L.append("hash: tt-metal hash16CT = FNV-1a-32 folded to 16 bits ((lo^hi) & 0xFFFF), space = 65536")
    L.append("")
    L.append("A) CURRENT DEFAULT — hash the full source string 'name,file,line,KERNEL_PROFILER'")
    L.append(f"   distinct source strings : {ds}")
    L.append(f"   16-bit COLLISIONS       : {len(cs)}   (birthday expectation at {ds} ~ {int((1-2.718281828**(-ds*ds/2/65536))*100)}%)")
    for h, a, b in cs:
        L.append(f"   collision @ hash 0x{h:04x}:")
        L.append(f"       A: {a}")
        L.append(f"       B: {b}")
        L.append(f"       (both stamped with the same 16-bit id {h} on-device; host keeps one, the other's")
        L.append("        samples are attributed to the kept name.)")
    L.append("")
    L.append("B) PROPOSED OPT-IN — hash the zone NAME only (field[0]); #pragma still logs file/line")
    L.append(f"   distinct zone names     : {dn}   (same op appears under ~{ds/dn:.1f} content-hashed kernel files)")
    L.append(f"   16-bit COLLISIONS       : {len(cn)}   (birthday expectation at {dn} ~ {int((1-2.718281828**(-dn*dn/2/65536))*100)}%)")
    for h, a, b in cn:
        L.append(f"   collision @ hash 0x{h:04x}: {a}  <->  {b}")
    if not cn:
        L.append("   (no different-named zones collide -> the current collision disappears)")
    L.append("")
    L.append("Interpretation: the collision is between two DIFFERENT zone names")
    L.append("(ATTN_Q vs an injected fabric-barrier zone). Full-string hashing also creates")
    L.append(f"{ds}-{dn}={ds-dn} same-name duplicates from per-file content hashes, inflating the count")
    L.append("into the ~50%+ birthday zone. Name-only hashing removes the duplicates and the collision.")

    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)
    print("\n".join(L))

if __name__ == "__main__":
    main()
