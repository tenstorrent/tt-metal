# binary and comparisons: the stacked bf16 rounding and the multiply, int32 compare levers (L) against the previous heads'
# files (P); C is L with a wrong rounding addend and compare shift (positive control); P2 repeats P (A/A).
BITS_VARIANTS="P L P2 C"
bits() {
  python3 $S/tools/exh_dump.py $1/exh.json add,subtract,multiply,rsub,divide,xlogy
  python3 $S/tools/exh_sct.py $1/sct.json
  R3_BITS_OUT=$1/ucmp timeout -s INT -k 60 1800 python3 -m pytest -q -p no:cacheprovider $S/tools/bits_ucmp.py
  python3 - $1 <<'PYB'
import glob, hashlib, json, os, sys
import numpy as np
d = sys.argv[1]; out = {}
for f in ("exh.json", "sct.json"):
    if os.path.exists(os.path.join(d, f)):
        out.update({f"{f}:{k}": v for k, v in json.load(open(os.path.join(d, f))).items() if k not in ("other", "lanes_per_case")})
for f in sorted(glob.glob(d + "/ucmp/*.npz")):
    z = np.load(f); h = hashlib.sha256()
    for k in sorted(z.files):
        h.update(k.encode()); h.update(np.ascontiguousarray(z[k]).tobytes())
    out["ucmp:" + os.path.basename(f)[:-4]] = h.hexdigest()[:20]
json.dump(out, open(d + "/all.json", "w"), indent=0)
print("cases", len(out))
PYB
}
COMPARE=()
PROF_PAIRS=("P L")
PROF_CASES="add_bf16_hs add_bf16_dram_1024x1024 sub_bf16_hs rsub_bf16_hs mul_bf16_hs mul_bf16_dram_1024x1024 mul_f32_hs ucmp_lt_hs ucmp_gt_hs"
PROF_SCRIPT=prof_bin.py
COMPARE=("P L" "P P2" "P C")
COMPARE_CMD="python3 $S/tools/cmp_json.py"
