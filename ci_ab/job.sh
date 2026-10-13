# reshuffle_rows: the replayed body (L) against the previous head's (P); C is L adding into the wrong register (positive
# control); P2 repeats P (A/A). The ttnn functional test runs on L.
BITS_VARIANTS="P L P2 C"
bits() {
  python3 $S/tools/exh_rr.py $1/rr.json
  python3 -c "import json,sys; d=json.load(open('$1/rr.json')); json.dump(d, open('$1/all.json','w'))"
  [[ $CUR == L ]] && timeout -s INT -k 60 1800 python3 -m pytest -q -p no:cacheprovider tests/ttnn/unit_tests/operations/data_movement/test_backward_embedding.py
}
COMPARE=("P L" "P P2" "P C")
COMPARE_CMD="python3 $S/tools/cmp_json.py"
PROF_PAIRS=("P L")
PROF_CASES="emb_2x1024x4096x3200 emb_1x2048x2048x32000 emb_8x128x768x30522 emb_2x64x160x96"
PROF_SCRIPT=prof_rr.py
