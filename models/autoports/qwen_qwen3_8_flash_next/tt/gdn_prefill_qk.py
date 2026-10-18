# SPDX-License-Identifier: Apache-2.0
"""Qualified contiguous FP32 Q/K reads for the serial serving prefill view.

Only normalized operand storage and its reader change. Original arithmetic,
state layout and public banks remain owned by ParallelGatedDeltaNet.
"""

import ttnn

from .multichip_decoder import _BROADCAST_DECODE_COMPUTE, _BROADCAST_PREFILL_READER, ParallelGatedDeltaNet


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one original reader fragment: {old}")
    return source.replace(old, new)


def _make_reader(source):
    source = _replace_once(
        source,
        """            for (uint32_t tile=0;tile<(which==3?1:4);++tile) {""",
        """            if(which!=3) {
                const uint32_t row=(b*length+token)*4+h/3;
                if(which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),512,{.page_id=row},{});
                else noc.async_read(k,CoreLocalMem<uint32_t>(scratch),512,{.page_id=row},{});
                noc.async_read_barrier();
            }
            for (uint32_t tile=0;tile<(which==3?1:4);++tile) {""",
    )
    for which, name in ((1, "q"), (2, "k")):
        source = _replace_once(
            source,
            f"                if (which=={which}) noc.async_read({name},CoreLocalMem<uint32_t>(scratch),4096,{{.page_id=page}},{{}});\n",
            "",
        )
    return _replace_once(source, "const uint32_t value=src[off(h/3,r)];", "const uint32_t value=src[tile*32+r];")


_RM_QK_READER = _make_reader(_BROADCAST_PREFILL_READER)


def _rm_recurrence(self, q, k, v, g, beta):
    """Exact token recurrence; prefill keeps FP32 state in aliased local buffers."""
    batch, length = q.shape[0], q.shape[1]
    if not (self.recurrence_broadcast and batch == 1 and length in (128, 256, 512)):
        raise ValueError("ROW_MAJOR Q/K requires the qualified serial prefill geometry")
    signature = dict(sequence=length)
    values = []
    for name, value in (("q", q), ("k", k)):
        converted = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT, memory_config=value.memory_config())
        if value.dtype != ttnn.float32 or converted.dtype != ttnn.float32:
            raise ValueError("ROW_MAJOR Q/K must preserve normalized FP32 operands")
        signature[name] = dict(
            logical_shape=list(value.shape),
            tile_padded_shape=list(value.padded_shape),
            rm_padded_shape=list(converted.padded_shape),
            tile_dtype=str(value.dtype),
            rm_dtype=str(converted.dtype),
            tile_memory_config=str(value.memory_config()),
            rm_memory_config=str(converted.memory_config()),
            tile_layout=str(value.layout),
            rm_layout=str(converted.layout),
        )
        values.append(converted)
    self.prefill_qk_rm_signatures[str(length)] = signature
    q, k = values
    del value, converted, values
    # Query normalization already applied the scale as its output activation.
    decay = g  # Already exponentiated by the multiply output activation.
    output = ttnn.empty(
        (batch, 12, length, 128),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=self.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    final = self.recurrent_state
    tensors = [q, k, v, decay, beta, self.recurrent_state, output, final]
    grid = self.device.compute_with_storage_grid_size()
    jobs = batch * 48
    workers = min(jobs, grid.x * grid.y)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = []
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    assignments = [(i * jobs // workers, (i + 1) * jobs // workers - i * jobs // workers) for i in range(workers)]
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
    )
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    reader = _RM_QK_READER
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=reader,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            config=ttnn.ReaderConfigDescriptor(),
            runtime_args=[
                ((c.x, c.y), [*[t.buffer_address() for t in tensors], *job, length])
                for c, job in zip(coords, assignments)
            ],
        ),
        ttnn.KernelDescriptor(
            kernel_source=_BROADCAST_DECODE_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            config=config,
            runtime_args=[((c.x, c.y), [job[1] * length]) for c, job in zip(coords, assignments)],
        ),
    ]
    sizes = [4, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4, 1, 1]
    if length > 1:
        sizes.append(1)
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * 4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
        )
        for i, count in enumerate(sizes)
        if not (length > 1 and i in (0, 12))
    ]
    if length > 1:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=4 * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096) for i in (0, 12)
                ],
            )
        )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
    return output, final


class PrefillQKRowMajorGDN(ParallelGatedDeltaNet):
    prefill_qk_rm_enabled = False

    def _recurrence(self, q, k, v, g, beta):
        b, s, _ = q.shape
        if not (self.prefill_qk_rm_enabled and self.recurrence_broadcast and b == 1 and s in (128, 256, 512)):
            return super()._recurrence(q, k, v, g, beta)
        q = self._heads(q, b, s, 4)
        k = self._heads(k, b, s, 4)
        q = ttnn.rms_norm(
            ttnn.typecast(q, ttnn.float32), weight=self.qscale, epsilon=1e-06 / 128, compute_kernel_config=self.compute
        )
        k = ttnn.rms_norm(
            ttnn.typecast(k, ttnn.float32), weight=self.kscale, epsilon=1e-06 / 128, compute_kernel_config=self.compute
        )
        v = ttnn.typecast(self._heads(v, b, s, 12), ttnn.float32)
        return _rm_recurrence(self, q, k, v, g, beta)
