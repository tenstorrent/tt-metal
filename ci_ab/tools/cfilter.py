"""Prints the TILE_LOOP rows of counters CSVs with the parameter columns and the hand-off counters only."""
import csv, re, sys
RT = re.compile(r"^(L1_TO_L1|UNPACK_ISOLATE|MATH_ISOLATE|PACK_ISOLATE|L1_CONGESTION)_")
KEEP = re.compile(r"INSTRN_THREAD\.(THREAD_INSTRUCTIONS_[012]|THREAD_STALLS_[012]|WAITING_FOR_(PACK_IDLE|NONZERO_SEM|NONFULL_SEM|MATH_IDLE|UNPACK_IDLE|CFG_IDLE|SFPU_IDLE)_[012]|WAITING_FOR_SRC[AB]_(VALID|CLEAR))$"
                  r"|FPU\.(FPU|MATH|SFPU)_COUNTER$|TDMA_PACK\.(PACKER_BUSY|PACKER0_DEST_READ_REQ|DEST_READ_GRANTED_0)$"
                  r"|TDMA_UNPACK\.(UNPACK0_BUSY_THREAD0|UNPACK1_BUSY_THREAD0|MATH_INSTRN_STARTED|SRCA_WRITE_REQ)$|INSTRN_THREAD\.THREAD_INSTRUCTIONS_0\.cycles$")
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
