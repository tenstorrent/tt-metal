# sigmoid and silu on DRAM 1x1x1024x1024: the stack (F) against the previous head (P), and F with three SFPNOPs per
# sigmoid row (Fn, about P's per-tile time), with per-RISC device columns; gelu at the same shape (our lever only).
BITS_VARIANTS=""
COMPARE=()
PROF_PAIRS=("P F" "P Fn")
PROF_CASES="sigmoid_dram_1024x1024 silu_dram_1024x1024 gelu_dram_1024x1024 sigmoid_dram_1024x4096 sigmoid_hs"
PROF_SCRIPT=prof_su.py
