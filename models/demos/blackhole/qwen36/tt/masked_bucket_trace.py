# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side (pure torch) value builders for the traced masked-bucket prefill.

The eager masked-bucket prefill rebuilds ~151 small host tensors per request (GDN beta/g and
q/k/v validity masks in every GDN layer, the FIR decode-window one-hot in every GDN layer, the
KV-fill page table, the logits one-hot). Each of those is a ``ttnn.from_torch`` host write, which
is illegal inside a captured trace ("Writes are not supported during trace capture") and is also
the bulk of the per-request host dispatch cost.

The traced path hoists all of them into ONE model-owned set of persistent device buffers per
bucket whose SHAPES depend only on the bucket, so a single captured program set serves every
``actual_len``; only the VALUES change per request and are DMA'd in with
``ttnn.copy_host_to_device_tensor`` before ``ttnn.execute_trace``.

Everything here is pure torch (no ttnn import) so the value logic — the part that is easy to get
subtly wrong — is unit-testable on a host without a Tenstorrent device.
"""

import torch

# Default paged-KV block size for the Qwen3.6 serving path (get_block_size(kv_caches)).
DEFAULT_BLOCK_SIZE = 64


def host_masks(actual_len, bucket):
    """GDN validity mask [1, bucket, 1] float32: 1.0 for t < actual_len else 0.0.

    One tensor feeds BOTH device buffers: the fp32 beta/g mask and the bf16 q/k/v mask (the eager
    path in gdn/fused_chunk.py builds the same float32 ``_mt`` and uploads it twice, once as
    float32 and once as ``q.dtype``). 0.0/1.0 are exact in bf16, so the traced values are
    bit-identical to the eager ones.
    """
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    m = torch.zeros(1, bucket, 1, dtype=torch.float32)
    m[:, :actual_len, :] = 1.0
    return m


def host_conv_sel(actual_len, bucket, kernel_size):
    """FIR decode-window one-hot [1, K-1, bucket+K-1] float32 (sel[0, j, actual_len+j] = 1.0).

    Selects rows ``x_padded[:, actual_len : actual_len+(K-1)]`` — i.e. the last K-1 REAL conv
    inputs ``x[actual_len-(K-1) : actual_len]`` — via a matmul instead of a static slice, so the
    program depends only on shapes (fixed per bucket) and only the values depend on actual_len.
    Mirrors the one-hot built inside ``_causal_conv1d_fir`` for the eager int-valid_len path.
    """
    assert kernel_size >= 2, f"kernel_size {kernel_size} must be >= 2"
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    total_len = (kernel_size - 1) + bucket
    sel = torch.zeros(1, kernel_size - 1, total_len, dtype=torch.float32)
    for j in range(kernel_size - 1):
        sel[:, j, actual_len + j] = 1.0
    return sel


def host_logit_sel(actual_len, bucket):
    """Last-real-row one-hot [1, 1, 1, bucket] float32 for the TP logits select (row actual_len-1)."""
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    sel = torch.zeros(1, 1, 1, bucket, dtype=torch.float32)
    sel[0, 0, 0, actual_len - 1] = 1.0
    return sel


def fill_pt_row(page_table, chunk_start, actual_len, bucket, pad_block, block_size=DEFAULT_BLOCK_SIZE):
    """FIXED-WIDTH KV-fill page table [1, bucket//block_size] int32 for the traced bucket body.

    The eager path sizes this table to the REAL blocks only (``page_table[:, blk0:blkN]`` with
    blkN = ceil((chunk_start+actual_len)/block_size)), so its width — and hence the paged_fill_cache
    program plus the ``ttnn.slice`` that trims K/V to page_len — changes with actual_len. A trace
    needs one fixed shape, so the traced body always fills the whole bucket: width = bucket/block_size
    makes ``page_len == S``, which also removes the slice entirely.

    Entries ``[0, nreal)`` are the request's real blocks. Trailing entries hold K/V for PAD rows,
    which causal SDPA never reads (k <= q < actual_len) and decode later overwrites via
    paged_update_cache; they must nevertheless point somewhere harmless:
      * the request's OWN mapped block at that index when it is non-zero (the natural full-bucket
        fill — no extra block needed, and the demo/tests' arange page tables always hit this), else
      * ``pad_block``, a scratch physical block no request owns (default: the last KV block, env
        QWEN36_PREFILL_BUCKET_PAD_BLOCK). Never 0, because block 0 is a real request's block for a
        zero-padded page-table row.
    """
    assert bucket % block_size == 0, f"bucket {bucket} must be a multiple of block_size {block_size}"
    assert chunk_start % block_size == 0, f"chunk_start {chunk_start} must be block-aligned"
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    assert int(pad_block) != 0, "pad_block must not be block 0 (it aliases a real request's block)"
    width = bucket // block_size
    blk0 = chunk_start // block_size
    nreal = -(-(chunk_start + actual_len) // block_size) - blk0  # ceil div
    nreal = max(0, min(nreal, width))
    pt = page_table.reshape(1, -1)
    assert blk0 + nreal <= pt.shape[1], (
        f"page table row of {pt.shape[1]} blocks does not cover the {nreal} real block(s) at "
        f"offset {blk0} for chunk_start={chunk_start}, actual_len={actual_len}"
    )
    row = torch.full((1, width), int(pad_block), dtype=torch.int32)
    for j in range(width):
        idx = blk0 + j
        if idx >= pt.shape[1]:
            continue  # page-table row ends before the bucket -> scratch block
        v = int(pt[0, idx])
        if j < nreal or v != 0:
            row[0, j] = v
    return row


def parse_bucket_trace_gate(value, all_buckets):
    """Parse QWEN36_PREFILL_BUCKET_TRACE into a sorted tuple of buckets to trace.

    "0"/""/unset -> () (today's eager masked path, byte-for-byte). "1" -> every bucket.
    A comma list ("128", "128,256") -> exactly those buckets, so the 128 bucket can ship alone
    while the trace-region footprint of the larger ones is measured.
    """
    if value is None:
        return ()
    value = value.strip()
    if value in ("", "0", "off", "false"):
        return ()
    if value in ("1", "all", "true"):
        return tuple(sorted(all_buckets))
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        b = int(part)
        assert b in all_buckets, f"QWEN36_PREFILL_BUCKET_TRACE bucket {b} is not one of {tuple(all_buckets)}"
        out.append(b)
    return tuple(sorted(set(out)))
