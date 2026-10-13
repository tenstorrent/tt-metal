# sigmoid and silu on DRAM 1x1x1024x1024, one change at a time: F (stack) against P (previous head); Fh = F with the
# exp constants as literals; Fi = F with 1/ln2 as a literal and no Prgm1 write; Pp = P with one SFPNOP per sigmoid row.
BITS_VARIANTS=""
COMPARE=()
PROF_PAIRS=("P F" "P Fh" "P Fi" "P Pp")
PROF_CASES="sigmoid_dram_1024x1024 silu_dram_1024x1024 sigmoid_hs"
PROF_SCRIPT=prof_su.py
