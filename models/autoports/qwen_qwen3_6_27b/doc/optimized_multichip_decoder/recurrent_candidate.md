# Recurrence candidates

The isolated investigator's source diagnosis is in debug_recurrent/AUTODEBUG.md.
The prepared recurrent_variants.py and probe_multichip_recurrent.py survived the
session transition; their agent had not written a final report. Parent inspection
confirmed all source transformation anchors match current code for grouped and
ungrouped execution. No hardware correctness is claimed from this check.

The probe compares candidate output and recurrent state bit-for-bit against
sources extracted from the accepted baseline commit, across changed-input eager
steps and trace replay. Individual candidates remove dead padding, unused
constants, duplicate grouped Q/K reads, repeated gate reads, or reduce exp work.
Timing uses alternating baseline/candidate order on the same TP4 mesh. Initial
smoke covers B32 BF16, followed by wider batches/dtypes/poison tests if it passes.
