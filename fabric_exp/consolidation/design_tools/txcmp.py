# Copied from the session scratchpad (2026-11-01): rxcmp.py (design_a) applied to the sender loop function.
# rxcmp.py's comparison applied to the sender loop (two ERISCs: sender::run; one ERISC: single::sender_pass).
import sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation/design_a")
import rxcmp
rxcmp.RX_KEYS = ("sender3run", "sender_pass")
rxcmp.main()
