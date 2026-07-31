# SPDX-License-Identifier: Apache-2.0
"""Fused MoE router (top-k + softmax + scatter) in one generic_op launch.

Replaces the 16-launch, 58.8 us/layer post-matmul router path with a single
kernel. See router_fused_reader.cpp for the full rationale and the op list it
subsumes.

Contract (identical to the path it replaces):
    in:  router_logits    [1,1,32,E] bf16 TILE   (only row 0 is real)
    out: routing_weights  [1,1,1,E]  bf16 TILE   dense, 0 outside the top-k
         expert_ids       [1,1,1,k]  uint32 ROW_MAJOR
"""
import ttnn

_KDIR = "models/demos/gpt_oss/kernels"
_TILE_BYTES = 2 * 1024
_CACHE = {}
_PROG = {}


def fused_router(logits, weights_out, ids_out, num_experts, top_k):
    key = (id(logits.device()), num_experts, top_k)
    if key not in _CACHE:
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        cb_in, cb_w, cb_id = 0, 1, 2
        ct = [cb_in, cb_w, cb_id, num_experts, top_k]
        ct += ttnn.TensorAccessorArgs(logits).get_compile_time_args()
        ct += ttnn.TensorAccessorArgs(weights_out).get_compile_time_args()
        ct += ttnn.TensorAccessorArgs(ids_out).get_compile_time_args()
        _CACHE[key] = (core, cb_in, cb_w, cb_id, ct)

    core, cb_in, cb_w, cb_id, ct = _CACHE[key]

    addrs = (logits.buffer_address(), weights_out.buffer_address(), ids_out.buffer_address())
    pk = (key, addrs)
    if pk in _PROG:
        return ttnn.generic_op([logits, weights_out, ids_out], _PROG[pk])

    # ids are uint16 TILE, so one full tile page
    id_bytes = _TILE_BYTES

    def cb(i, page):
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=page, core_ranges=core, format_descriptors=[fmt])

    cbs = [cb(cb_in, _TILE_BYTES), cb(cb_w, _TILE_BYTES), cb(cb_id, id_bytes)]

    rt = ttnn.RuntimeArgs()
    rt[0][0] = [logits.buffer_address()]

    rt_w = ttnn.RuntimeArgs()
    rt_w[0][0] = [weights_out.buffer_address(), ids_out.buffer_address()]

    reader = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/router_split_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/router_split_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=ct,
        runtime_args=rt_w,
        config=ttnn.WriterConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    _PROG[pk] = prog
    return ttnn.generic_op([logits, weights_out, ids_out], prog)
