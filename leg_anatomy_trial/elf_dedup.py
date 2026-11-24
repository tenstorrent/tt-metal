"""Hash the loadable content of each variant's ELFs (PT_LOAD segments + entry), ignoring debug info and paths.
Writes {variant_dir: {elf: hash}, ...} plus summary counts."""
import hashlib, json, os, struct, sys
root, out = sys.argv[1], sys.argv[2]
def load_hash(p):
    with open(p, 'rb') as f: d = f.read()
    if d[:4] != b'\x7fELF' or d[4] != 1: return hashlib.sha256(d).hexdigest()  # 32-bit RISC-V expected
    e = '<'
    entry, phoff = struct.unpack_from(e+'II', d, 24)
    phentsize, phnum = struct.unpack_from(e+'HH', d, 42)
    h = hashlib.sha256(struct.pack('<I', entry))
    for i in range(phnum):
        t, off, vaddr, paddr, filesz, memsz, flags, align = struct.unpack_from(e+'8I', d, phoff+i*phentsize)
        if t == 1:
            h.update(struct.pack('<4I', vaddr, filesz, memsz, flags)); h.update(d[off:off+filesz])
    return h.hexdigest()
V = {}
for dp, dn, fn in os.walk(root):
    if dp.endswith('/elf') and '/shared/' not in dp + '/':
        var = os.path.dirname(dp)
        V[os.path.relpath(var, root)] = {f: load_hash(os.path.join(dp, f)) for f in sorted(fn) if f.endswith('.elf')}
trip = {}
for v, m in V.items(): trip.setdefault(json.dumps(m, sort_keys=True), []).append(v)
elfs = [h for m in V.values() for h in m.values()]
per_kind = {}
for m in V.values():
    for k, h in m.items(): per_kind.setdefault(k, set()).add(h)
s = {'variants': len(V), 'unique_variants_by_load_content': len(trip), 'elfs': len(elfs), 'unique_elfs': len(set(elfs)),
     'unique_per_kind': {k: len(v) for k, v in per_kind.items()},
     'largest_dup_groups': sorted(([len(v), v[:4]] for v in trip.values()), reverse=True)[:15]}
json.dump({'summary': s, 'variants': V}, open(out, 'w'))
print(json.dumps(s)[:3000])
