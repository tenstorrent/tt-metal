"""Exact device-side n-gram IDs and request-owned token history."""
from pathlib import Path

import ttnn


def ngram_indices(tokens, positions, slots, state, *, valid_lengths, output, multipliers, primes, offsets, eos):
    """Hash UINT32 tokens [B,S] into [B*S,16], preserving a request's last two IDs.

    positions/slots/valid_lengths are UINT32 [B,32] (first element used).
    state is UINT32 [max_requests,32]: previous-older, previous-newer, next-pos.
    Position 0 starts a fresh request; continuation must match saved next-pos.
    Padded input tokens neither advance state nor become part of the hash history.
    All tensors and output must be allocated before capture.
    """
    batch, physical = tokens.shape
    grid = tokens.device().compute_with_storage_grid_size()
    if batch > grid.x * grid.y:
        raise ValueError("Hashing requires one worker per batch lane")
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(batch)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    config = [physical, eos, *[part for m in multipliers for part in (m & 0xFFFFFFFF, m >> 32)], *primes, *offsets]
    for tensor in (tokens, positions, slots, state, valid_lengths, output):
        if tensor.dtype != ttnn.uint32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("Hash inputs and output must be UINT32 ROW_MAJOR")
        config.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    cb = ttnn.CBDescriptor(
        total_size=((physical * 4 + 63) // 64) * 64 + 1024,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0, data_format=ttnn.uint32, page_size=((physical * 4 + 63) // 64) * 64 + 1024
            )
        ],
    )
    tensors = [tokens, positions, slots, state, valid_lengths, output]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).with_name("kernels") / "ple_hash.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=config,
        runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
    return output
