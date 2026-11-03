# SPDX-License-Identifier: Apache-2.0
"""Parallel n-gram (PLE) convolution taps for single-lane multi-token prefill.

multichip_decoder._local_ple_conv runs one worker per lane: for a serial prefill row that is one core
copying 4 taps x seq rows (read, wait, write, wait) and then shifting the request's 9-row history. Here
the tap copies are spread over the grid (each (tap, token) job writes its own output row) and the history
shift runs afterwards as a second single-core program, so no tap read can race the history update.
Both programs are derived from _LOCAL_PLE_CONV: same sources, rows and bytes.
Toggle: QWEN_PREFILL_PLE_PAR=0 restores the single-core kernel (default on).
"""

import os

import ttnn

from .multichip_decoder import _LOCAL_PLE_CONV


def ple_parallel_enabled():
    return os.environ.get("QWEN_PREFILL_PLE_PAR", "1") != "0"


_TAP_LOOP = "    for (uint32_t tap = 0; tap < 4; ++tap) {\n" "        for (uint32_t t = 0; t < seq; ++t) {\n"
_HISTORY = "    // Advance by logical length."


def _derive():
    if _LOCAL_PLE_CONV.count(_TAP_LOOP) != 1 or _LOCAL_PLE_CONV.count(_HISTORY) != 1:
        raise ValueError("Unexpected _LOCAL_PLE_CONV layout")
    head, rest = _LOCAL_PLE_CONV.split(_TAP_LOOP)
    taps, history = rest.split(_HISTORY)
    jobs = (
        "    const uint32_t begin = get_arg_val<uint32_t>(7), count = get_arg_val<uint32_t>(8),"
        " stride = get_arg_val<uint32_t>(9);\n"
        "    for (uint32_t item = 0; item < count; ++item) {\n"
        "        {\n"
        "            const uint32_t tap = (begin + item * stride) / seq, t = (begin + item * stride) % seq;\n"
    )
    return head + jobs + taps + "}\n", head + _HISTORY + history


TAPS_SOURCE, HISTORY_SOURCE = _derive()


def local_ple_conv_parallel(x, positions, slots, valid_lengths, history, *, output, seq):
    """_local_ple_conv for batch 1: tap copies over the grid, then the history shift on one core."""
    grid = x.device().compute_with_storage_grid_size()
    tensors = [x, positions, slots, valid_lengths, history, output]
    args = [1, seq]
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    addresses = [t.buffer_address() for t in tensors]

    def run(source, workers, extra):
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        cb = ttnn.CBDescriptor(
            total_size=5184,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=5184)],
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=[((c.x, c.y), [*addresses, 0, *extra(i, workers)]) for i, c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))

    work = 4 * seq
    workers = min(work, grid.x * grid.y)
    run(TAPS_SOURCE, workers, lambda i, w: [i, (work + w - 1 - i) // w, w])
    run(HISTORY_SOURCE, 1, lambda i, w: [])
