# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-module sequence ceilings for Tracy / device-profiler capture (Blackhole 1×4).

The ``test_*_prof_capture_seq_pcc`` tests run one forward at ``seq ==`` the constant below.
That workload is small enough for::

    python -m tracy -p --op-support-count 10000 -r -v -m pytest <prof_capture_test> -k 1x4

to finish without Tracy's 32K source-location overflow, Metal DRAM marker drops, or
capture-process segfault. The next power-of-two step above each constant typically fails
one of those checks (or PCC at 0.99).

Limits were validated by sweeping powers of two only (…, 64, 128, 256, 512, …), not
single-step binary search. They are separate from the PCC ``MAX_*`` constants in each
test file (hardware / numerical correctness ceilings).

Re-validate when profiler buffers or Seamless op counts change materially.
"""

# PCC ≥ 0.99 @ seed=0; mel_seq=64/128 fail PCC (bf16 drift); mel_seq=256+ breaks tracy -r -v.
SPEECH_ENCODER_MEL_SEQ = 32

# Tracy-safe tokenized source length for text-decoder profiling (decoder seed = 2).
TEXT_DECODER_ENC_SEQ = 32

# Highest power-of-two T2U encoder forward that clears PCC and tracy on BH (1024 hits L1).
TEXT_TO_UNIT_ENCODER_SEQ = 512

# Highest power-of-two unit_seq that clears PCC and tracy; 256 overflows 32K source locations.
CODE_HIFIGAN_UNIT_SEQ = 128
