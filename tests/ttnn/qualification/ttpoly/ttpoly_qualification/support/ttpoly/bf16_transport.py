# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Canonical raw BF16 transport, independent of fitting and CLI machinery.

NumPy, Torch and TTNN are supplied by the caller. This module moves encodings;
it defines neither an evaluator nor a golden reference.
"""
import hashlib
from pathlib import Path
import sys


BF16_PATTERN_COUNT = 1 << 16


def _write_all(handle, data: bytes) -> None:
    """Write a complete chunk even when a FIFO reports a short write."""

    remaining = memoryview(data)
    while remaining:
        written = handle.write(remaining)
        if written is None:
            # Buffered streams may return None only in non-blocking mode, which
            # this producer never requests.  Treat it as a hard short write.
            raise OSError("raw FP32 output stream made no write progress")
        if written <= 0:
            raise OSError("raw FP32 output stream ended during a chunk write")
        remaining = remaining[written:]


def dispatch_bf16_raw_words(np, torch, ttnn, apply, device, bits):
    """Transport tile-aligned BF16 encodings without a floating-point conversion."""
    bits = np.asarray(bits)
    if bits.dtype != np.dtype(np.uint16) or bits.ndim != 1 or not bits.size or bits.size % 1024:
        raise ValueError("raw BF16 dispatch requires a nonempty tile-aligned uint16 vector")
    host_bits = torch.from_numpy(bits.view(np.int16).copy())
    host_input = host_bits.view(torch.bfloat16)
    device_input = ttnn.from_torch(
        host_input.reshape(1, 1, 32, -1),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    out_t = ttnn.to_torch(apply(device_input)).reshape(-1)
    if out_t.numel() != bits.size:
        raise RuntimeError(f"bf16 exhaustive output size {out_t.numel()} != {bits.size}")
    return out_t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16).copy()


def run_bf16_exhaustive_accuracy(np, torch, ttnn, apply, device, args):
    """Push all 65,536 BF16 bit patterns through one raw-bit dispatch."""

    device.enable_program_cache()
    output_path = Path(args.out_bf16_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    with output_path.open("wb", buffering=0) as output_stream:
        bits = np.arange(BF16_PATTERN_COUNT, dtype=np.uint16)
        out_bits = dispatch_bf16_raw_words(np, torch, ttnn, apply, device, bits)
        raw_output = out_bits.astype(np.dtype("<u2"), copy=False).tobytes(order="C")
        _write_all(output_stream, raw_output)
        output_digest.update(raw_output)
        output_stream.flush()

    expected_bytes = BF16_PATTERN_COUNT * 2
    if len(raw_output) != expected_bytes:
        raise RuntimeError(f"internal bf16 exhaustive output mismatch: " f"bytes={len(raw_output)}/{expected_bytes}")
    input_sum = int(np.sum(bits.astype(np.uint64), dtype=np.uint64)) & ((1 << 64) - 1)
    input_xor = int(np.bitwise_xor.reduce(bits))
    print(
        f"BF16_EXHAUSTIVE_PROGRESS,count={BF16_PATTERN_COUNT}," f"total={BF16_PATTERN_COUNT},chunks=1",
        file=sys.stderr,
        flush=True,
    )
    return {
        "schema": "ttnn_bf16_exhaustive_producer_v1",
        "complete": True,
        "input_encoding": "sequential_ieee754_bfloat16_bit_patterns",
        "output_encoding": "raw_little_endian_bfloat16",
        "start_bit": 0,
        "end_bit_exclusive": BF16_PATTERN_COUNT,
        "count": BF16_PATTERN_COUNT,
        "chunks": 1,
        "expected_bytes": expected_bytes,
        "observed_bytes": len(raw_output),
        "input_bit_sum_mod_2_64": input_sum,
        "input_bit_xor": input_xor,
        "output_checksum": output_digest.hexdigest(),
    }
