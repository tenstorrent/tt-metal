"""Run the traced conv decode A/B with the s4_res / s1_up 4x8 blockings of arm $T100_ARM patched into _BLOCKINGS.

Arms: table (unchanged), exact (sweep winner with the table's C_in_block, expected bit-identical),
best (sweep winner, smaller C_in_block). Usage: python ab100.py <pytest args>
"""
import os
import sys

import pytest

from models.tt_dit.utils import conv3d

S4 = (4, 8, 128, 128, (3, 3, 3), 147, 68, 60)
S1 = (4, 8, 512, 4096, (3, 3, 3), 39, 17, 15)
ARMS = {
    "table": {},
    "exact": {S4: (128, 64, 6, 4, 8), S1: (128, 64, 5, 2, 16)},
    "best": {S4: (64, 128, 6, 4, 8), S1: (64, 256, 1, 2, 16)},
}
arm = os.environ["T100_ARM"].rstrip("0123456789")
conv3d._BLOCKINGS.update(ARMS[arm])
print(f"T100_ARM {os.environ['T100_ARM']} s4={conv3d._BLOCKINGS[S4]} s1_up={conv3d._BLOCKINGS[S1]}", flush=True)
sys.exit(pytest.main(sys.argv[1:]))
