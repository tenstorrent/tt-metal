# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Prefill configuration of Llama-3.3-70B on the 8x4 Wormhole Galaxy (TP32). Pure python (no ttnn), so every rule
here can be checked without a device.

These settings are the model's prefill path, not options: no environment variable changes them. They apply to Llama
only. Qwen (qwen_model_config.py) shares the tt/ modules and keeps its original prefill path: every caller checks
is_qwen, or passes it to the functions below.
"""

# --- Padding ---
PAD_1024_EXTRA_SEQLENS = (3072, 5120, 6144)  # cold prefill pads to these between the powers of two (7168: no)
CPF_PAD512_SEQLENS = (2560, 3584)  # new rows over a cached prefix on the CPF ring path may also pad to these


def prefill_extra_seqlens(is_qwen=False):
    """Non-power-of-two prefill lengths: each gets persistent prefill CCL buffers (TT_CCL.support_seqlens), a warmup
    compile and sp0/sp1 prefill traces. None for Qwen."""
    return () if is_qwen else PAD_1024_EXTRA_SEQLENS


def cpf_pad512_seqlens(is_qwen=False):
    """Ring-only lengths (CCL buffers and CPF ring traces only; the cold path never pads to them). None for Qwen."""
    return () if is_qwen else CPF_PAD512_SEQLENS


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


def prefill_slab_rows(seq_len):
    """Rows per slab of the slabbed 2D prefill matmuls: 1024, or 512 for the lengths that are not multiples of 1024
    (CPF_PAD512_SEQLENS)."""
    return 1024 if seq_len % 1024 == 0 else 512


# --- Ring SDPA ---
# ring_distributed_sdpa splits each device's two seqlen/8 slices into 8 heads x q-chunks. With q_chunk 256 a 4096
# prefill keeps 8 of 70 cores busy with 512 Q rows each; q_chunk 64 gives 128 Q rows per busy core.
RING_SDPA_Q64_SEQLENS = (2048, 4096, 6144, 8192)  # cold ring SDPA lengths that run q_chunk 64 (not 256 or 128)
CPF_PAD512_Q_CHUNK = 64  # ring SDPA q_chunk_size at the CPF_PAD512 lengths (their seqlen/8 is a multiple of 64 only)


def ring_sdpa_q_chunk_size(seqlen):
    """q_chunk_size for ring_distributed_sdpa at seqlen >= 2048 (ring_size 4: each device takes two seqlen/8 slices).
    The op places a slice at slice_index * (seqlen/8) // q_chunk_size, so q_chunk_size must divide seqlen/8."""
    if seqlen in RING_SDPA_Q64_SEQLENS:
        return 64
    if (seqlen // 8) % 256 == 0:
        return 256
    if (seqlen // 8) % 128 == 0:
        return 128
    return CPF_PAD512_Q_CHUNK


# --- minimal_matmul (ttnn.experimental.minimal_matmul; FF2: the fused all_gather_minimal_matmul_async) ---
MINMM_QKV_MIN_SEQLEN = 2048  # batch-1 prefill QKV matmul runs as minimal_matmul (unslabbed input) from this length
MINMM_QKV_CONFIG = (8, 8, 4, 1, 4, 7, 10)  # QKV minimal_matmul: M/K/N block, subblock h/w, grid x/y (in tiles)
MINMM_MLP_SEQLENS = (2048, 2560, 3072, 3584)  # batch-1 MLP lengths below 4096 that run minimal matmuls, as >= 4096


def prefill_minmm_qkv(seq_len, batch_size):
    """True when the prefill QKV matmul runs as minimal_matmul (callers also require the prefetcher and Llama)."""
    return batch_size == 1 and seq_len >= MINMM_QKV_MIN_SEQLEN


def prefill_minmm_mlp(seq_len, batch_size):
    """True when a prefill MLP shorter than 4096 rows runs FF1/FF3 and FF2 as minimal matmuls (callers also require
    the prefetcher, Wormhole and Llama)."""
    return batch_size == 1 and seq_len in MINMM_MLP_SEQLENS


# --- CCLs ---
PREFILL_RS_BUFS_PER_CHANNEL = 32  # num_buffers_per_channel of the Wormhole prefill ring reduce-scatter (FF1/FF3)
WO_AG_MAX = 0  # Llama prefill WO all-reduce: all-gather + local sum (key WO_AG) up to this length, else RS + AG
WO_AG_MAX_QWEN = 4096  # the same split for Qwen (its original value)
PREFILL_SEQSPLIT_MIN = 2048  # shortest batch-1 prefill whose residual add + RMSNorm run on seqlen/8 rows per device
PREFILL_SEQSPLIT_MAX = 8192  # longest one

# --- Prefill over a cached prefix: ring SDPA on all 4 mesh columns (CPF ring) ---
CPF_RING_GATHER_Q_CHUNK = 32  # q chunk of the one-hot SDPA that gathers the owner's paged prefix K/V
CPF_RING_GATHER_K_CHUNK = 256  # its k chunk (the prefix length must be a multiple of it)
CPF_RING_GATHER_SCALE = 16.0  # its softmax scale (score gap between the matching row and any other row >= 2 * scale)
CPF_RING_ALIGN = 1024  # the generator rounds a cached prefix down to a multiple of this to take the ring path
CPF_RING_WARM_PREFIXES = (2048,)  # prefix lengths whose CPF ring traces are warmed; only these take the ring path
CPF_RING_MIN_NEW = 2048  # fewest padded new rows that take the CPF ring path
CPF_RING_GROUP_BLOCKS = 1  # paged blocks per prefix-gather group: each gathered row scores this many blocks of keys
# Only 1, 2 and 4 build: at 8 the page-table split yields one part, concat aliases it and the gather
# reads a freed tensor (TT_THROW 'Tensor is not allocated').
assert CPF_RING_GROUP_BLOCKS in (1, 2, 4), CPF_RING_GROUP_BLOCKS
CPF_REPLICA_MAX = 5120  # longest padded new-row count that uses the CPF prefix replica (2 ring traces per length)


def cpf_ring_eligible(seq_len, prefix_len, block_size, support_seqlens):
    """The one CPF ring rule for a batch-1 Llama prefill of seq_len new rows over prefix_len cached rows, shared by
    TtLlamaAttention._cpf_ring_enabled (which path runs) and the generator (trace key, skip rule, rounding, warmup),
    so a trace keyed for the ring path always holds it and no other trace does. block_size is the paged KV cache's
    (kv tensor dim 2: vLLM passes no paged_attention_config); support_seqlens the prefill TT_CCL's. Callers rule out
    Qwen, Blackhole and batch > 1 first."""
    P = int(prefix_len)
    if P <= 0 or seq_len < CPF_RING_MIN_NEW:
        return False
    return (
        P in CPF_RING_WARM_PREFIXES
        and P in support_seqlens  # the K/V broadcast uses its ATTN_REPLICATE buffers
        and P % CPF_RING_ALIGN == 0
        and P % CPF_RING_GATHER_K_CHUNK == 0
        and P % block_size == 0
        and P % ring_sdpa_q_chunk_size(seq_len) == 0
        and (P + seq_len) % 512 == 0  # SDPA_PROGCFG k_chunk_size for seq_len >= 2048
        and seq_len % block_size == 0
    )
