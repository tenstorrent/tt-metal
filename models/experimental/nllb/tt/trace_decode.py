# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Request-local NLLB decoder traces and packed vocabulary projection."""

import json
import sys

import numpy as np
import torch
import ttnn


class LastWarmupReuse:
    """Preparation metadata only; explicitly owned cache lifetime, never tensors.

    No automatic cache-mutation detection is available. The owner guarantees
    an enabled, uninterrupted program cache, invalidates BEFORE clear/disable or
    configuration mutation, and leaves this scope before device teardown.
    This scope is sequential: concurrent requests/cache mutation are unsupported.
    """

    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.variants = set()
        self.entered = False
        self.cleanup_errors = []

    def __enter__(self):
        b = self.model
        if self.entered or getattr(b, "_last_warmup_reuse", None) is not None:
            raise RuntimeError("LAST warmup reuse already owned")
        b.invalidate_last_warmup()
        if b.device is not self.device:
            raise RuntimeError("LAST warmup device changed")
        self.variants.clear()
        b._last_warmup_reuse = self
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        # A failed invalidation must retain scope/trace ownership. Keep the
        # secondary diagnostic without replacing a primary request exception.
        try:
            self.model.invalidate_last_warmup()
        except BaseException as cleanup_error:
            self.cleanup_errors.append(cleanup_error)
            if exc is None:
                raise
            return False
        self.model._last_warmup_reuse = None
        self.entered = False
        return False

    @staticmethod
    def tensor_spec(tensor):
        return (
            tuple(tensor.shape),
            tuple(tensor.padded_shape),
            str(tensor.dtype),
            str(tensor.layout),
            repr(tensor.memory_config()),
        )

    def warm(self, output, rows, encoder):
        b = self.model
        if not self.entered or getattr(b, "_last_warmup_reuse", None) is not self:
            raise RuntimeError("LAST warmup scope is not owned")
        if b.device is not self.device:
            raise RuntimeError("LAST warmup device changed")
        # All operator geometry/configuration, including physical padding,
        # dtype/layout/memory, row offsets, projection policy and kernel options.
        # Addresses and prompt values are deliberately not retained.
        geometry = (
            self.tensor_spec(output),
            self.tensor_spec(b.lm_weight),
            self.tensor_spec(encoder),
            b.dim,
            b.vocab,
            b.generation_projection,
            str(b.matrix_dtype),
            json.dumps(b.config, sort_keys=True),
            json.dumps(b.precision_policy, sort_keys=True),
            id(b.kernel),
            repr(b.kernel),
            id(type(b).project_decoder),
            "LAST-v1",
        )
        pending = [(geometry, row) for row in range(1, rows + 1) if (geometry, row) not in self.variants]
        for _, row in pending:
            b.project_decoder(output, row, final_token_only=True)
        # Commit atomically only after every missing variant executes and syncs.
        # A failed enqueue/readback/sync leaves no newly reusable variants.
        ttnn.synchronize_device(self.device)
        self.variants.update(pending)


class DecoderTrace:
    """Request-owned trained full-prefix body; no vocabulary projection in capture."""

    def __init__(self, model, encoder, valid, cross_kv):
        model._guard_trace_ownership()
        self.model = model
        self.device = model.device
        self.encoder, self.valid = model.decoder_memory(encoder, valid)
        self.cross_kv = cross_kv
        self.inputs = []
        self.output = None
        self.trace_id = None
        self.bucket = None
        self.replays = 0
        self.events = []
        self.unresolved = False
        self.end_attempted = False

    def host_inputs(self, ids):
        b = self.model
        padded = b.padded(ids, b.pad)
        n = padded.shape[1]
        tokens, positions = b.embedding_host_inputs(padded)

        def mask(valid, causal=False):
            allowed = np.broadcast_to(valid.astype(bool)[None, :], (n, len(valid))).copy()
            if causal:
                allowed &= np.arange(len(valid))[None, :] <= np.arange(n)[:, None]
            return torch.from_numpy(np.where(allowed, 0.0, -1e9).astype(np.float32)).reshape(1, 1, n, -1)

        values = (tokens, positions, mask(np.arange(n) < ids.shape[1], True), mask(self.valid))
        return [
            ttnn.from_torch(
                v.contiguous(),
                dtype=ttnn.uint32 if i == 0 else ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if i == 0 else ttnn.TILE_LAYOUT,
            )
            for i, v in enumerate(values)
        ]

    def body(self):
        a, p, causal, cross = self.inputs
        return self.model.decoder_body(a, p, self.encoder, causal, cross, cross_kv=self.cross_kv)

    def end(self):
        self.end_attempted = True
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        self.events.append("end")

    def prepare(self, ids, hosts):
        self.model._guard_trace_ownership()
        n = ((ids.shape[1] + 31) // 32) * 32
        if n == self.bucket:
            return
        self.close()
        if not hasattr(self.model, "_last_warmup_owners"):
            self.model._last_warmup_owners = set()
        self.model._last_warmup_owners.add(self)
        self.inputs = [ttnn.to_device(h, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for h in hosts]
        self.bucket = n
        # Warm body and every real-row LAST slice shape before capture. No
        # full-prefix vocabulary projection is performed, including warmup.
        warm = self.body()
        scope = getattr(self.model, "_last_warmup_reuse", None)
        if scope is None:
            for length in range(1, n + 1):
                self.model.project_decoder(warm, length, final_token_only=True)
        else:
            scope.warm(warm, n, self.encoder)
        del warm
        ttnn.synchronize_device(self.device)
        self.capture()

    def capture(self):
        self.end_attempted = False
        try:
            self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        except BaseException as error:
            # Native begin may mutate capture state before returning an ID.
            # No ID means no supported end/release target; retain every buffer.
            # An inactive-capture observation alone cannot establish recovery.
            self.unresolved = True
            self.events.append("begin_error:" + repr(error))
            self.close(preserve_exception=True)
            raise
        self.events.append("begin:" + str(self.bucket))
        try:
            self.output = self.body()
            self.end()
        except BaseException:
            if not self.end_attempted:
                try:
                    self.end()
                except BaseException as error:
                    self.unresolved = True
                    self.events.append("end_error:" + repr(error))
            else:
                self.unresolved = True
            self.close(preserve_exception=True)
            raise

    def decode(self, ids):
        self.model._guard_trace_ownership()
        hosts = self.host_inputs(ids)
        self.prepare(ids, hosts)
        addresses = [x.buffer_address() for x in self.inputs]
        for src, dst in zip(hosts, self.inputs):
            ttnn.copy_host_to_device_tensor(src, dst, cq_id=0)
        assert addresses == [x.buffer_address() for x in self.inputs]
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=True)
        self.replays += 1
        self.events.append("replay")
        # This helper returns only copied host data; its TT temporaries expire.
        return self.model.project_decoder(self.output, ids.shape[1], final_token_only=True)

    def close(self, preserve_exception=False):
        if self.trace_id is not None:
            try:
                ttnn.release_trace(self.device, self.trace_id)
                self.events.append("release")
                self.trace_id = None
            except BaseException as error:
                self.unresolved = True
                self.events.append("release_error:" + repr(error))
                if self not in self.model._trace_failures:
                    self.model._trace_failures.append(self)
                if not preserve_exception:
                    raise
        if self.unresolved:
            if self not in self.model._trace_failures:
                self.model._trace_failures.append(self)
            return  # Preserve all ownership after uncertain native cleanup.
        getattr(self.model, "_last_warmup_owners", set()).discard(self)
        self.output = None
        self.inputs.clear()
        self.bucket = None
        self.events.append("drop_buffers")


_RAW_SELECTOR_KERNEL = r"""
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(1);
    const auto hidden = TensorAccessor(TensorAccessorArgs<2>(), get_arg_val<uint32_t>(0));
    const auto index = TensorAccessor(TensorAccessorArgs<INDEX_ARGS>(), get_arg_val<uint32_t>(1));
    const auto output = TensorAccessor(TensorAccessorArgs<OUTPUT_ARGS>(), get_arg_val<uint32_t>(2));
    cb_reserve_back(0, 1);
    cb_reserve_back(1, 1);
    cb_reserve_back(2, 1);
    const uint32_t src = get_write_ptr(0);
    const uint32_t dst = get_write_ptr(1);
    const uint32_t idx = get_write_ptr(2);
    noc_async_read_page(0, index, idx);
    noc_async_read_barrier();
    const uint32_t row = *reinterpret_cast<volatile uint32_t*>(idx);
    volatile uint16_t* s = reinterpret_cast<volatile uint16_t*>(src);
    volatile uint16_t* d = reinterpret_cast<volatile uint16_t*>(dst);
    // Invalid indices produce zeros without reading the hidden table.
    for (uint32_t tile = 0; tile < width_tiles; ++tile) {
        for (uint32_t j = 0; j < 1024; ++j) d[j] = 0;
        if (row < rows) {
            noc_async_read_page((row / 32) * width_tiles + tile, hidden, src);
            noc_async_read_barrier();
            for (uint32_t col = 0; col < 32; ++col)
                d[get_tilized_idx(0, col)] = s[get_tilized_idx(row, col)];
        }
        noc_async_write_page(tile, output, dst);
        noc_async_write_barrier();
    }
}
"""


def raw_selector(hidden, index, output):
    """Raw BF16 selector verified by receipt 59da8b07618f46cd93c2c76d9f5e7d8a; index contents stay on device."""
    rows, width = tuple(hidden.shape)[-2:]
    assert rows in (32, 64, 96, 128, 160, 192, 224, 256) and width in (1024, 2048)
    assert tuple(hidden.shape) == tuple(hidden.padded_shape) == (1, 1, rows, width)
    assert tuple(index.shape) == tuple(index.padded_shape) == (1, 1)
    assert index.dtype == ttnn.uint32 and index.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tuple(output.shape) == (1, 1, 1, width)
    assert tuple(output.padded_shape) == (1, 1, 32, width)
    for tensor in (hidden, output):
        assert tensor.dtype == ttnn.bfloat16 and tensor.layout == ttnn.TILE_LAYOUT
    for tensor in (hidden, index, output):
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert len({t.buffer_address() for t in (hidden, index, output)}) == 3
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    ha, ia, oa = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (hidden, index, output)]
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t.buffer_address() for t in (hidden, index, output)]
    kernel = ttnn.KernelDescriptor(
        kernel_source=_RAW_SELECTOR_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core,
        compile_time_args=[rows, width // 32] + ha + ia + oa,
        defines=[("INDEX_ARGS", str(2 + len(ha))), ("OUTPUT_ARGS", str(2 + len(ha) + len(ia)))],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    cbs = [
        ttnn.CBDescriptor(
            total_size=2048, core_ranges=core, format_descriptors=[ttnn.CBFormatDescriptor(i, ttnn.bfloat16, 2048)]
        )
        for i in range(3)
    ]
    return ttnn.generic_op([hidden, index, output], ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs))


class ProjectedDecoderTrace(DecoderTrace):
    """Request-owned decoder, dynamic row selector, one-row LM and layout."""

    def __init__(self, *args):
        super().__init__(*args)
        self.selected = None
        self.warm_results = 0

    def host_inputs(self, ids):
        hosts = super().host_inputs(ids)
        # Real final position, not a count of non-PAD entries.
        index = torch.tensor([[ids.shape[1] - 1]], dtype=torch.int32)
        hosts.append(ttnn.from_torch(index, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT))
        return hosts

    def body(self):
        a, p, causal, cross, index = self.inputs
        hidden = self.model.decoder_body(a, p, self.encoder, causal, cross, cross_kv=self.cross_kv)
        raw_selector(hidden, index, self.selected)
        return self.model.project_selected_device(self.selected)

    def prepare(self, ids, hosts):
        self.model._guard_trace_ownership()
        n = ((ids.shape[1] + 31) // 32) * 32
        if n == self.bucket:
            return False
        self.close()
        if not hasattr(self.model, "_last_warmup_owners"):
            self.model._last_warmup_owners = set()
        self.model._last_warmup_owners.add(self)
        self.inputs = [ttnn.to_device(h, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for h in hosts]
        self.selected = self.model.upload(torch.zeros(1, 1, 1, self.model.dim))
        self.bucket = n
        self.output = self.body()
        return True

    def decode(self, ids):
        self.model._guard_trace_ownership()
        hosts = self.host_inputs(ids)
        if self.prepare(ids, hosts):
            # Ordinary execution is the first result for this bucket. Defer
            # capture until another call: EOS/cap therefore needs no capture.
            try:
                result = self.model.read_decoder_row(self.output)
                self.warm_results += 1
                self.events.append("warm_result")
                return result
            finally:
                self.output = None
        addresses = [x.buffer_address() for x in self.inputs]
        for src, dst in zip(hosts, self.inputs):
            ttnn.copy_host_to_device_tensor(src, dst, cq_id=0)
        assert addresses == [x.buffer_address() for x in self.inputs]
        if self.trace_id is None:
            self.capture()
        # Capture records commands; only explicit execution yields logits.
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=True)
        self.replays += 1
        self.events.append("replay")
        return self.model.read_decoder_row(self.output)

    def close(self, preserve_exception=False):
        super().close(preserve_exception=preserve_exception)
        if not self.unresolved:
            self.selected = None


def pack_last_rows(hidden, indices, output):
    """Copy independent LAST vectors into height rows of one BF16 query tile."""
    batch = len(hidden)
    assert 1 <= batch <= 4 and len(indices) == batch
    width = int(hidden[0].shape[-1])
    assert width % 32 == 0
    assert tuple(output.shape) == (1, 1, batch, width)
    assert tuple(output.padded_shape) == (1, 1, 32, width)
    tensors = list(hidden) + list(indices) + [output]
    for x in list(hidden) + [output]:
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
        assert x.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    for x in hidden:
        assert tuple(x.shape) == tuple(x.padded_shape)
        assert tuple(x.shape)[:2] == (1, 1) and x.shape[-1] == width
        assert x.shape[-2] % 32 == 0
    for x in indices:
        assert tuple(x.shape) == tuple(x.padded_shape) == (1, 1)
        assert x.dtype == ttnn.uint32 and x.layout == ttnn.ROW_MAJOR_LAYOUT
        assert x.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert len({x.buffer_address() for x in tensors}) == len(tensors)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    args, declarations = [], []
    for i, x in enumerate(tensors):
        declarations.append(
            f"const auto a{i} = TensorAccessor(TensorAccessorArgs<{len(args)}>(), get_arg_val<uint32_t>({i}));"
        )
        args += list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    copies = []
    for i, x in enumerate(hidden):
        copies.append(
            f"""
        noc_async_read_page(0, a{batch + i}, idx);
        noc_async_read_barrier();
        const uint32_t row{i} = *reinterpret_cast<volatile uint32_t*>(idx);
        if (row{i} < {x.shape[-2]}) {{
            noc_async_read_page((row{i}/32)*{width // 32}+tile, a{i}, src);
            noc_async_read_barrier();
            for (uint32_t col=0; col<32; ++col)
                d[get_tilized_idx({i},col)] = s[get_tilized_idx(row{i},col)];
        }}
        """
        )
    source = (
        """
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
void kernel_main() {
"""
        + "\n".join(declarations)
        + """
    cb_reserve_back(0,1); cb_reserve_back(1,1); cb_reserve_back(2,1);
    const uint32_t src=get_write_ptr(0), dst=get_write_ptr(1), idx=get_write_ptr(2);
    volatile uint16_t* s=reinterpret_cast<volatile uint16_t*>(src);
    volatile uint16_t* d=reinterpret_cast<volatile uint16_t*>(dst);
"""
        + f"for (uint32_t tile=0; tile<{width // 32}; ++tile) {{"
        + """
    for (uint32_t j=0; j<1024; ++j) d[j]=0;
"""
        + "\n".join(copies)
        + f"""
    noc_async_write_page(tile,a{2 * batch},dst); noc_async_write_barrier();
    }}
}}
"""
    )
    runtime = ttnn.RuntimeArgs()
    runtime[0][0] = [x.buffer_address() for x in tensors]
    kernel = ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core,
        compile_time_args=args,
        runtime_args=runtime,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    cbs = [
        ttnn.CBDescriptor(
            total_size=2048, core_ranges=core, format_descriptors=[ttnn.CBFormatDescriptor(i, ttnn.bfloat16, 2048)]
        )
        for i in range(3)
    ]
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs))
    return output


class PersistentRowTrace(DecoderTrace):
    """Decoder-only trace writing into a buffer allocated before any capture."""

    def body(self):
        hidden = super().body()
        # Same-layout device copy preserves BF16 payloads, including signed zero.
        ttnn.copy(hidden, self.output)
        return self.output


class PackedLMRequest(DecoderTrace):
    """Request-owned row traces and one fixed-lane shared physical LAST tile."""

    def __init__(self, model, input_ids, attention_mask):
        super().__init__(model, None, np.zeros(0), {})
        self.rows = []
        self.active = []
        self.selected = None
        self.warm_results = 0
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def initialize(self):
        for row in range(len(self.input_ids)):
            encoder, valid = self.model.encode(self.input_ids[row : row + 1], self.attention_mask[row : row + 1])
            self.rows.append(PersistentRowTrace(self.model, encoder, valid, {}))

    def body(self):
        pack_last_rows([row.output for row in self.rows], self.inputs, self.selected)
        logits = ttnn.linear(
            self.selected,
            self.model.lm_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.model.kernel,
        )
        logits = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, len(self.rows), self.model.vocab))
        logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        # Installed copy supports preallocated row-major output; add does not.
        ttnn.copy(logits, self.output)
        return self.output

    def index_host(self, position):
        return ttnn.from_torch(
            torch.tensor([[position]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def prepare_rows(self, prefixes, active):
        self.close()
        self.model._last_warmup_owners = getattr(self.model, "_last_warmup_owners", set())
        self.model._last_warmup_owners.add(self)
        batch = len(self.rows)
        # Allocate ALL persistent inputs and outputs before the first capture.
        for i, row in enumerate(self.rows):
            ids = np.array([prefixes[i]], dtype=np.int64)
            row.bucket = ((ids.shape[1] + 31) // 32) * 32
            row.inputs = [
                ttnn.to_device(h, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for h in row.host_inputs(ids)
            ]
            row.output = self.model.upload(torch.zeros(1, 1, row.bucket, self.model.dim))
        self.inputs = [
            ttnn.to_device(
                self.index_host(len(prefixes[i]) - 1 if i in active else self.rows[i].bucket),
                self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for i in range(batch)
        ]
        self.selected = self.model.upload(torch.zeros(1, 1, batch, self.model.dim))
        host = ttnn.from_torch(
            torch.zeros(1, 1, batch, self.model.vocab), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.output = ttnn.to_device(host, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.bucket = tuple(row.bucket for row in self.rows)
        self.active = list(active)
        # Populate every cross-KV and compile every body before live traces.
        for row in self.rows:
            row.body()
        self.body()
        ttnn.synchronize_device(self.device)
        self.warm_results += 1

    def decode_rows(self, prefixes, active):
        self.model._guard_trace_ownership()
        if self.bucket is None or any(((len(prefixes[i]) + 31) // 32) * 32 != self.rows[i].bucket for i in active):
            self.prepare_rows(prefixes, active)
            return self.read()
        self.active = list(active)
        for i, row in enumerate(self.rows):
            if i in active:
                hosts = row.host_inputs(np.array([prefixes[i]], dtype=np.int64))
                for src, dst in zip(hosts, row.inputs):
                    ttnn.copy_host_to_device_tensor(src, dst, cq_id=0)
            # Invalid LAST index zeroes an inactive lane, including stale hidden.
            ttnn.copy_host_to_device_tensor(
                self.index_host(len(prefixes[i]) - 1 if i in active else row.bucket), self.inputs[i], cq_id=0
            )
        for i in active:
            row = self.rows[i]
            if row.trace_id is None:
                row.capture()
            ttnn.execute_trace(self.device, row.trace_id, cq_id=0, blocking=True)
            row.replays += 1
        if self.trace_id is None:
            self.capture()
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=True)
        self.replays += 1
        return self.read()

    def read(self):
        values = ttnn.to_torch(self.output).float().numpy().reshape(len(self.rows), self.model.vocab)
        return values[self.active].copy()

    def close(self, preserve_exception=False):
        errors = []
        # Release every trace before dropping any shared persistent buffer.
        for owner in [self] + self.rows:
            if owner.trace_id is not None:
                try:
                    ttnn.release_trace(self.device, owner.trace_id)
                    owner.trace_id = None
                    owner.events.append("release")
                except BaseException as error:
                    owner.unresolved = True
                    owner.events.append("release_error:" + repr(error))
                    errors.append(error)
            if owner.unresolved and owner not in self.model._trace_failures:
                self.model._trace_failures.append(owner)
        if any(owner.unresolved for owner in [self] + self.rows):
            self.unresolved = True
            if self not in self.model._trace_failures:
                self.model._trace_failures.append(self)
        else:
            for row in self.rows:
                row.inputs.clear()
                row.output = None
                row.bucket = None
            super().close(preserve_exception=preserve_exception)
            self.selected = None
            self.active = []
        if errors and not preserve_exception:
            raise errors[0]

    def finish(self, preserve_exception=False):
        try:
            self.close(preserve_exception=preserve_exception)
        finally:
            if not self.unresolved:
                for row in self.rows:
                    row.cross_kv.clear()
                    row.encoder = None
                self.rows.clear()
                self.input_ids = self.attention_mask = None


def generate_packed(model, input_ids, attention_mask, target_id, max_new_tokens):
    owner = PackedLMRequest(model, input_ids, attention_mask)
    model._batch_trace = owner
    outputs = [[2, int(target_id)] for _ in range(len(input_ids))]
    active = list(range(len(outputs)))
    try:
        owner.initialize()
        for _ in range(1, max_new_tokens):
            logits = owner.decode_rows(outputs, active)
            if logits.shape != (len(active), model.vocab) or not np.isfinite(logits).all():
                raise FloatingPointError("Invalid packed last-token logits")
            continuing = []
            for slot, row in enumerate(active):
                token = int(np.argmax(logits[slot]))
                outputs[row].append(token)
                if token != 2:
                    continuing.append(row)
            active = continuing
            if not active:
                break
    finally:
        try:
            owner.finish(preserve_exception=sys.exc_info()[0] is not None)
        finally:
            model._batch_trace = None
    result = np.full((len(outputs), max(map(len, outputs))), model.pad, dtype=np.int64)
    for row, tokens in enumerate(outputs):
        result[row, : len(tokens)] = tokens
    return result
