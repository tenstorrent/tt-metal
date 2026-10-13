# reshuffle_rows op level: the replayed body (L), with the replay recorded on the first live group and no restore without
# one (Z), and without the stall before each offset write (N), each against the previous head (P).
BITS_VARIANTS="P Z N"
bits() {
  python3 $S/tools/exh_rr.py $1/rr.json
  python3 -c "import json,sys; d=json.load(open('$1/rr.json')); json.dump(d, open('$1/all.json','w'))"
}
COMPARE=("P Z" "P N")
COMPARE_CMD="python3 $S/tools/cmp_json.py"
PROF_PAIRS=("P L" "P Z" "P N")
PROF_CASES="emb_1x2048x2048x32000 emb_8x128x768x30522 emb_2x1024x4096x3200 emb_2x64x160x96"
PROF_SCRIPT=prof_rr.py
