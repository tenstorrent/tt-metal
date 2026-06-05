# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy-safe sequence lengths for ``test_*_prof_capture_*`` (BH 1×4).

Separate from per-module ``MAX_*`` PCC limits in each test file.
"""

# PCC ≥ 0.99 @ seed=0; mel_seq=64/128 fail PCC (bf16 drift); mel_seq=256+ breaks tracy -r -v.
SPEECH_ENCODER_MEL_SEQ = 32

# Tracy-safe tokenized source length for text-decoder profiling (decoder seed = 2).
TEXT_DECODER_ENC_SEQ = 32

# Highest power-of-two T2U encoder forward that clears PCC and tracy on BH (1024 hits L1).
TEXT_TO_UNIT_ENCODER_SEQ = 512

# Highest power-of-two unit_seq that clears PCC and tracy; 256 overflows 32K source locations.
CODE_HIFIGAN_UNIT_SEQ = 128
