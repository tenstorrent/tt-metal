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


# LLAMA_GALAXY_PREFILL_SLAB1024 (opt-in): ring SDPA q_chunk_size for the 4096/6144/8192 buckets.
# ring_distributed_sdpa splits each device's two seqlen/8 slices into 8 heads x q-chunks and, when the
# chunk count per slice is even, hands them out in pairs, one pair per core. With q_chunk 256 a 4096
# prefill keeps 8 of 70 cores busy with 512 Q rows each; 5120 (q_chunk 128, 5 chunks, odd) keeps 40
# cores busy with 128 rows each. The value is the q_chunk_size to use at those lengths:
#   unset / "0": original (256 at every power of two >= 2048).
#   "1" / "128": 128, the 5120 setting (4096 and 8192 halve rows per core; 6144 is unchanged).
#   "64": 64, 128 Q rows per busy core at 4096, 6144 and 8192 (32 / 48 / 64 cores).
PREFILL_SLAB1024_ENV = "LLAMA_GALAXY_PREFILL_SLAB1024"
PREFILL_SLAB1024_SEQLENS = (4096, 6144, 8192)


def prefill_slab1024_q_chunk(env_value=None):
    """Ring SDPA q_chunk_size selected by LLAMA_GALAXY_PREFILL_SLAB1024, or None when unset."""
    value = os.environ.get(PREFILL_SLAB1024_ENV, "") if env_value is None else env_value
    value = value.strip()
    if value in ("", "0"):
        return None
    if value in ("1", "128"):
        return 128
    if value == "64":
        return 64
    raise ValueError(f"{PREFILL_SLAB1024_ENV}={value!r} is not supported; use 1, 128 or 64, or leave it unset")


def ring_sdpa_q_chunk_override(seqlen, default, q_chunk=None):
    """q_chunk for ring SDPA at seqlen: `q_chunk` for the PREFILL_SLAB1024_SEQLENS when it divides the
    per-device slice (seqlen/8), else `default`. q_chunk None keeps `default` everywhere."""
    if q_chunk is None or seqlen not in PREFILL_SLAB1024_SEQLENS or (seqlen // 8) % q_chunk != 0:
        return default
    return q_chunk


# LLAMA_GALAXY_PREFILL_SDPA_Q64_2048=1 (opt-in): ring SDPA q_chunk_size 64 for the cold 2048 bucket.
# At 2048 the default q_chunk 256 gives each device 8 heads x 2 slices = 16 work units for 70 cores;
# 64 gives 64. Same mechanism as LLAMA_GALAXY_PREFILL_SLAB1024=64 at 4096/6144/8192. Read at call time.
PREFILL_SDPA_Q64_2048_ENV = "LLAMA_GALAXY_PREFILL_SDPA_Q64_2048"


def prefill_sdpa_q64_2048_enabled():
    return os.environ.get(PREFILL_SDPA_Q64_2048_ENV, "0").strip() == "1"


# LLAMA_GALAXY_PREFILL_MINMM (opt-in, read at call time): move prefill matmuls from ttnn.linear to
# ttnn.experimental.minimal_matmul (FF2: the fused all_gather_minimal_matmul_async), which reaches
# 137-155 TFLOP/s on FF1 where linear reaches 81-105. Comma (or +) list of parts, or "1" for all:
#   qkv     : the QKV matmul at every batch-1 prefill length >= 2048 (unslabbed [1, 1, S, 2048] input)
#   mlp2048 : FF1/FF3 (minimal) and FF2 (fused AG + minimal) at 2048
#   mlp3072 : FF1/FF3 (minimal) and FF2 (fused AG + minimal) at 3072 (not part of "1")
#   mlp512  : the same at the CPF_PAD512 lengths below 4096 (2560, 3584; not part of "1")
PREFILL_MINMM_ENV = "LLAMA_GALAXY_PREFILL_MINMM"
PREFILL_MINMM_PARTS = ("qkv", "mlp2048", "mlp3072", "mlp512")
PREFILL_MINMM_ALL = ("qkv", "mlp2048")  # what "1" turns on

# Settings that were env-tunable probes, fixed at the values every serving run used.
CPF_RING_WARM_PREFIXES = (2048,)  # prefix lengths whose CPF ring traces are warmed; only these take the ring path
CPF_RING_MIN_NEW = 2048  # fewest padded new rows that take the CPF ring path
CPF_PAD512_Q_CHUNK = 64  # ring SDPA q_chunk_size at the CPF_PAD512 lengths (their seqlen/8 is a multiple of 64 only)
MINMM_QKV_CONFIG = (8, 8, 4, 1, 4, 7, 10)  # QKV minimal_matmul: M/K/N block, subblock h/w, grid x/y (in tiles)
PREFILL_CCL_BUFS_KINDS = ("rs",)  # prefill CCLs that get PREFILL_CCL_BUFS buffers per channel: the FF1/FF3 ring RS


# LLAMA_GALAXY_PREFILL_CPF_PAD512 (opt-in, read at TT_CCL construction and call time): a prefill over a cached
# prefix that takes the CPF ring path pads its new tokens to the next multiple of 512 when that is one of the
# enabled lengths (and shorter than the usual padding), instead of the next 1024. "1" enables CPF_PAD512_SEQLENS,
# a comma list enables a subset (e.g. "2560"). These lengths get prefill CCL buffers and ring(+replica) traces only:
# the cold path keeps its padding and warmup. Their matmuls run on 512-row slabs (prefill_slab_rows).
CPF_PAD512_ENV = "LLAMA_GALAXY_PREFILL_CPF_PAD512"
CPF_PAD512_SEQLENS = (2560, 3584, 4608)


def cpf_pad512_seqlens():
    """The ring-only lengths LLAMA_GALAXY_PREFILL_CPF_PAD512 enables (empty when unset or "0")."""
    value = os.environ.get(CPF_PAD512_ENV, "").strip()
    if value in ("", "0"):
        return ()
    if value == "1":
        return CPF_PAD512_SEQLENS
    lens = tuple(int(v) for v in value.replace("+", ",").split(",") if v.strip())
    for s in lens:
        if s not in CPF_PAD512_SEQLENS:
            raise ValueError(f"{CPF_PAD512_ENV}={value!r}: {s} is not one of {CPF_PAD512_SEQLENS}")
    return lens


def cpf_pad512_on():
    return bool(cpf_pad512_seqlens())


def prefill_slab_rows(seq_len):
    """Rows per slab of the slabbed 2D prefill matmuls: 1024, or 512 for the lengths that are not multiples of 1024
    (CPF_PAD512_SEQLENS)."""
    return 1024 if seq_len % 1024 == 0 else 512


def prefill_minmm(part):
    value = os.environ.get(PREFILL_MINMM_ENV, "").strip()
    if value in ("", "0"):
        return False
    parts = PREFILL_MINMM_ALL if value == "1" else tuple(p.strip() for p in value.replace("+", ",").split(","))
    for p in parts:
        if p not in PREFILL_MINMM_PARTS:
            raise ValueError(
                f"{PREFILL_MINMM_ENV}={value!r}: unknown part {p!r}; use 1 or a list of {PREFILL_MINMM_PARTS}"
            )
    return part in parts
