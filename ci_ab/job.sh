# unary stack (F) against its previous head (P), both merged with main: op level of sigmoid at several shapes, silu and
# exp; P2 repeats P (A/A). No bits (run 38429066540 compared them).
BITS_VARIANTS=""
COMPARE=()
PROF_PAIRS=("P F" "P P2")
PROF_CASES="sigmoid_dram_1024x1024 sigmoid_dram_1024x4096 sigmoid_dram_256x1024 sigmoid_hs silu_dram_1024x1024 exp_dram_1024x1024"
PROF_SCRIPT=prof_su.py
