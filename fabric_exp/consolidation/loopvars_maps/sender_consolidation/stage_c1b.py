"""stage_c1b.py: v4txc1b = v4txc1 + stage C1b of V4_SENDER_CONSOLIDATION.md (user's answer 3): a VC's free-slots
registers are read in the pass's first read group where this router serves the VC's worker channel
(reads_free_slots_at_pass_start = has_worker_channel), not wherever the host has a worker position. worker_positions
keeps the worker-channel mask and the one-worker assert. Changes only images with a worker position this router does not
serve (worker-trimmed shapes)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txc1", "v4txc1b", "c1b.patch")
