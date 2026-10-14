"""TILE_LOOP rows of counters CSVs: parameter columns and every counter of the four run types, as CHDR/CROW lines."""
import csv, re, sys
RT = re.compile(r"^(L1_TO_L1|UNPACK_ISOLATE|MATH_ISOLATE|PACK_ISOLATE|L1_CONGESTION)_")
KEEP = re.compile(r"INSTRN_THREAD\.|FPU\.|TDMA_PACK\.|TDMA_UNPACK\.|L1\.")
for f in sys.argv[1:]:
    rows = list(csv.reader(open(f)))
    if not rows:
        continue
    h = rows[0]
    idx = [i for i, c in enumerate(h) if (not RT.match(c) and not c.startswith("TEXT_SIZE")) or (RT.match(c) and KEEP.search(RT.sub("", c)))]
    mk = h.index("marker") if "marker" in h else None
    print("CHDR," + ",".join(h[i] for i in idx))
    for r in rows[1:]:
        if mk is None or r[mk] == "TILE_LOOP":
            print("CROW," + ",".join(r[i] if i < len(r) else "" for i in idx))
