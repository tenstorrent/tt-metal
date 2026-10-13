# #58179 alone on current main (S, conflict in exp.h resolved by keeping main's three-constant overload as a wrapper)
# against main's files (M): sigmoid and silu op level; M2 repeats M (A/A). No bits.
BITS_VARIANTS=""
COMPARE=()
PROF_PAIRS=("M S" "M M2")
PROF_CASES="sigmoid_dram_1024x1024 silu_dram_1024x1024 sigmoid_dram_1024x4096 sigmoid_hs"
PROF_SCRIPT=prof_su.py
