# #58179 (S) against its own base's files (M), on its head: op level of sigmoid at several shapes, silu and exp; M2
# repeats M (A/A). No bits.
BITS_VARIANTS=""
COMPARE=()
PROF_PAIRS=("M S" "M M2")
PROF_CASES="sigmoid_dram_1024x1024 sigmoid_dram_1024x4096 sigmoid_dram_256x1024 sigmoid_hs silu_dram_1024x1024 exp_dram_1024x1024"
PROF_SCRIPT=prof_su.py
