# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integer-only request seed progression for the existing common sampler.

No sampled-token mathematics lives here. The generic op writes the common
sampler's existing seed buffer before its own manual_seed + sampling calls.
All slot state is persistent UINT32 ROW_MAJOR DRAM. Actual common-sampler
seeded replay/lifecycle proof is in doc/full_model/seeded_sampling_device_v2.json
(synthetic immutable logits; it does not establish full-model serving acceptance).
"""

from __future__ import annotations

import operator
import secrets

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
DEVICE_SEED_MAX = 1_000_000


def seed_for_draw(seed, counter):
    """Exact common _hash_request_seed_to_device_seed(..., salt=0)."""
    value = (int(seed) & MASK64) ^ ((int(counter) + 0x9E3779B97F4A7C15) & MASK64)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return ((value ^ (value >> 31)) % DEVICE_SEED_MAX) + 1


def reset_words(seed, counter, *, entropy_seed=None):
    counter = operator.index(counter)
    if not 0 <= counter <= MASK64:
        raise ValueError("Draw counter must be an unsigned 64-bit integer")
    if seed is None:
        value = secrets.randbelow(DEVICE_SEED_MAX) + 1 if entropy_seed is None else operator.index(entropy_seed)
        if not 1 <= value <= DEVICE_SEED_MAX:
            raise ValueError("Unseeded initialization must be in [1,1000000]")
        mode = 2
    else:
        value, mode = operator.index(seed) & MASK64, 1
    return [value & MASK32, value >> 32, counter & MASK32, counter >> 32, mode] + [0] * 11


_INTEGER_SOURCE = r"""
struct Seed64 { uint32_t lo; uint32_t hi; };
inline Seed64 seed_xor(Seed64 a, Seed64 b) { return {a.lo ^ b.lo, a.hi ^ b.hi}; }
inline Seed64 seed_right(Seed64 a, uint32_t n) {
    return {(a.lo >> n) | (a.hi << (32 - n)), a.hi >> n};
}
inline Seed64 seed_mul32(uint32_t a, uint32_t b) {
    const uint32_t a0 = a & 65535u, a1 = a >> 16;
    const uint32_t b0 = b & 65535u, b1 = b >> 16;
    const uint32_t w0 = a0 * b0;
    const uint32_t t = a1 * b0 + (w0 >> 16);
    const uint32_t w1 = a0 * b1 + (t & 65535u);
    return {(w1 << 16) | (w0 & 65535u), a1 * b1 + (t >> 16) + (w1 >> 16)};
}
inline Seed64 seed_mul(Seed64 a, Seed64 b) {
    Seed64 result = seed_mul32(a.lo, b.lo);
    result.hi += a.lo * b.hi + a.hi * b.lo;
    return result;
}
inline Seed64 seed_add(Seed64 a, Seed64 b) {
    const uint32_t lo = a.lo + b.lo;
    return {lo, a.hi + b.hi + (lo < a.lo ? 1u : 0u)};
}
inline uint32_t request_seed_for_draw(Seed64 seed, Seed64 counter) {
    Seed64 value = seed_xor(seed, seed_add(counter, {0x7F4A7C15u, 0x9E3779B9u}));
    value = seed_mul(seed_xor(value, seed_right(value, 30)), {0x1CE4E5B9u, 0xBF58476Du});
    value = seed_mul(seed_xor(value, seed_right(value, 27)), {0x133111EBu, 0x94D049BBu});
    value = seed_xor(value, seed_right(value, 31));
    uint32_t remainder = 0;
    for (int bit = 31; bit >= 0; --bit) {
        remainder = (remainder << 1) | ((value.hi >> bit) & 1u);
        if (remainder >= 1000000u) remainder -= 1000000u;
    }
    for (int bit = 31; bit >= 0; --bit) {
        remainder = (remainder << 1) | ((value.lo >> bit) & 1u);
        if (remainder >= 1000000u) remainder -= 1000000u;
    }
    return remainder + 1u;
}
// Mode 1 is seeded; mode 2 initializes an unseeded hardware stream once.
// Mode 0 leaves the hardware RNG advancing. Inactive rows do not advance.
inline uint32_t advance_request_seed(volatile uint32_t* row, bool active) {
    if (!active) return 0xFFFFFFFFu;
    if (row[4] == 2u) { row[4] = 0u; return row[0]; }
    if (row[4] != 1u) return 0xFFFFFFFFu;
    const uint32_t value = request_seed_for_draw({row[0], row[1]}, {row[2], row[3]});
    row[2] += 1u;
    if (row[2] == 0u) row[3] += 1u;
    return value;
}
"""


def kernel_source():
    return (
        '#include <stdint.h>\n#include "api/dataflow/dataflow_api.h"\n'
        + _INTEGER_SOURCE
        + r"""
void kernel_main() {
    constexpr uint32_t DRAW = get_compile_time_arg_val(0);
    constexpr auto sa = TensorAccessorArgs<1>();
    constexpr auto ua = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto aa = TensorAccessorArgs<ua.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(0), 64);
    const auto updates = TensorAccessor(ua, get_arg_val<uint32_t>(1), 64);
    const auto active = TensorAccessor(aa, get_arg_val<uint32_t>(2), 128);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(3), 128);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint32_t* metadata = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 2048);
    volatile tt_l1_ptr uint32_t* seeds = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 4096);
    for (uint32_t row = 0; row < 32; ++row)
        noc_async_read(state.get_noc_addr(row), scratch + row * 64, 64);
    if constexpr (DRAW) {
        noc_async_read(active.get_noc_addr(0), scratch + 2048, 128);
    } else {
        for (uint32_t row = 0; row < 32; ++row)
            noc_async_read(updates.get_noc_addr(row), scratch + 2048 + row * 64, 64);
    }
    noc_async_read_barrier();
    for (uint32_t row = 0; row < 32; ++row) {
        if constexpr (DRAW) {
            ASSERT(rows[row * 16 + 4] <= 2u);
            ASSERT(metadata[row] <= 1u);
            seeds[row] = advance_request_seed(rows + row * 16, metadata[row] != 0u);
        } else if (metadata[row * 16] != 0u) {
            ASSERT(metadata[row * 16] == 1u);
            for (uint32_t word = 0; word < 5; ++word)
                rows[row * 16 + word] = metadata[row * 16 + word + 1];
        }
        noc_async_write(scratch + row * 64, state.get_noc_addr(row), 64);
    }
    if constexpr (DRAW) noc_async_write(scratch + 4096, output.get_noc_addr(0), 128);
    noc_async_write_barrier();
}
"""
    )


class DeviceSeedState:
    """Setup-owned seed state and transition-only host APIs; no token readback.

    prepare() warms reset/draw ops with initially inactive state. reset_slots() runs
    only at authoritative request transitions; set_active_slots() uploads only
    when membership changes. Put forward() inside the sampling trace directly
    before common _run_sampling. Do not also advance common SeedManager.
    """

    def __init__(self, mesh, seed_output):
        import torch

        import ttnn

        self._ttnn, self._torch, self.mesh = ttnn, torch, mesh
        if tuple(mesh.shape) != (1, 4):
            raise ValueError("Seed helper requires the current replicated TP4 sampler")
        if tuple(seed_output.shape) != (32,):
            raise ValueError("Expected common sampler seed output shape [32]")

        def zeros(shape):
            return ttnn.from_torch(
                torch.zeros(shape, dtype=torch.uint32),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.state, self.updates, self.active = zeros((32, 16)), zeros((32, 16)), zeros((32,))
        self.output = seed_output
        self.tensors = [self.state, self.updates, self.active, self.output]
        for tensor in self.tensors:
            if tensor.dtype != ttnn.uint32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise ValueError("Seed buffers require UINT32 ROW_MAJOR")
            if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise ValueError("Seed buffers require interleaved DRAM")
        self.programs = [self._program(draw) for draw in (0, 1)]
        self._active_slots = ()
        self._prepared = False

    def _program(self, draw):
        ttnn = self._ttnn
        core = ttnn.CoreCoord(0, 0)
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        runtime = ttnn.RuntimeArgs()
        runtime[0][0] = [tensor.buffer_address() for tensor in self.tensors]
        compile_args = [draw]
        for tensor in self.tensors:
            compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        return ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=kernel_source(),
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ranges,
                    compile_time_args=compile_args,
                    runtime_args=runtime,
                    config=ttnn.WriterConfigDescriptor(),
                )
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=4224,
                    core_ranges=ranges,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=4224)
                    ],
                )
            ],
            semaphores=[],
        )

    def _upload(self, target, values):
        source = self._ttnn.from_torch(values, dtype=self._ttnn.uint32, layout=self._ttnn.ROW_MAJOR_LAYOUT)
        self._ttnn.copy_host_to_device_tensor(source, target)

    def prepare(self):
        if self._prepared:
            return
        for program in self.programs:
            self._ttnn.experimental.prepare_generic_op(self.tensors, program)
            # prepare_generic_op populates the host program cache but does not
            # load its binaries onto the mesh. The first actual enqueue must
            # precede trace capture. Initial zero updates/active masks make
            # these setup executions leave request state unchanged.
            self._ttnn.generic_op(self.tensors, program)
        self._prepared = True

    def reset_slots(self, seeds, counters, slots):
        slots = tuple(operator.index(slot) for slot in slots)
        if len(set(slots)) != len(slots) or any(not 0 <= slot < 32 for slot in slots):
            raise ValueError("Reset slots must be unique physical32 indices")
        if len(seeds) != len(slots) or len(counters) != len(slots):
            raise ValueError("Seeds/counters must be in reset-slot order")
        updates = self._torch.zeros((32, 16), dtype=self._torch.uint32)
        for slot, seed, counter in zip(slots, seeds, counters, strict=True):
            updates[slot, :6] = self._torch.tensor([1] + reset_words(seed, counter)[:5], dtype=self._torch.uint32)
        self._upload(self.updates, updates)
        self._ttnn.generic_op(self.tensors, self.programs[0])

    def set_active_slots(self, slots):
        slots = tuple(sorted(operator.index(slot) for slot in slots))
        if len(set(slots)) != len(slots) or any(not 0 <= slot < 32 for slot in slots):
            raise ValueError("Active slots must be unique physical32 indices")
        if slots != self._active_slots:
            active = self._torch.zeros(32, dtype=self._torch.uint32)
            for slot in slots:
                active[slot] = 1
            self._upload(self.active, active)
            self._active_slots = slots

    def remap(self, remap, *, warmup=False):
        from .slot_state import validate_remap

        remap = validate_remap(remap, len(remap))
        if len(remap) > 32:
            raise ValueError("At most 32 seed slots")
        if not warmup and all(source == destination for destination, source in enumerate(remap)):
            return
        full = remap + tuple(range(len(remap), 32))
        rows = [self.state[source : source + 1, :] for source in full]
        gathered = self._ttnn.concat(rows, dim=0)
        self._ttnn.copy(gathered, self.state)
        # active is scheduling control, supplied for the new submission; it is
        # not request state. Force the next membership call to refresh it.
        self._active_slots = None

    def forward(self):
        return self._ttnn.generic_op(self.tensors, self.programs[1])
