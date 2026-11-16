# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Prefill padding policy. Pure python (no ttnn) so it can be checked without a device.

LLAMA_GALAXY_PREFILL_PAD selects how a prompt longer than 1024 tokens is padded:
  unset / "0" / "pow2": the next power of two (the original behaviour).
  "1024": the next multiple of 1024, but only onto the lengths in PAD_1024_EXTRA_SEQLENS.
          Every other length still pads to the next power of two.

Each extra length gets persistent prefill CCL buffers (llama_ccl.TT_CCL.support_seqlens), a warmup
compile, and sp0/sp1 prefill traces, so the set is kept small. 7168 is left out: inputs in
6145-8192 keep padding to 8192.
"""

import os

PREFILL_PAD_ENV = "LLAMA_GALAXY_PREFILL_PAD"
PAD_1024_EXTRA_SEQLENS = (3072, 5120, 6144)


def prefill_extra_seqlens(env_value=None):
    """Non-power-of-two prefill lengths enabled by LLAMA_GALAXY_PREFILL_PAD (empty when unset)."""
    value = os.environ.get(PREFILL_PAD_ENV, "") if env_value is None else env_value
    value = value.strip()
    if value in ("", "0", "pow2"):
        return ()
    if value == "1024":
        return PAD_1024_EXTRA_SEQLENS
    raise ValueError(f"{PREFILL_PAD_ENV}={value!r} is not supported; use 1024, or leave it unset for powers of two")


def padded_prefill_len(seq_len: int, extra_seqlens=()) -> int:
    """128 up to 128 tokens, 1024 up to 1024, then the smallest of the next power of two and any
    extra length that holds seq_len."""
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    padded = 2 ** (seq_len - 1).bit_length()
    for extra in extra_seqlens:
        if seq_len <= extra < padded:
            padded = extra
    return padded
