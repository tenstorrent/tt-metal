# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exact, device-resident Qwen3.8-Flash-Next PLE n-gram hashing.

The checkpoint uses positive signed-I64 products. Two UINT32 limbs preserve
those products without floating-point conversion; bit-serial integer division
then produces the exact remainder. This is specific to the pinned Qwen4Exp
text config (PLE ordinal zero), not a general signed-I64 hash implementation.

Buffers are preallocated UINT32, interleaved DRAM, ROW_MAJOR tensors:

* tokens: [B, S, 16]. Word 0 is a token, word 1 is conv-mask validity (0/1).
* control: [B, 16]. Words 0/1 are logical length and reset-history (0/1).
* history: [B, 16]. Words 0/1 store oldest/newest previous token, in place.
* output: [B, S, 16]. Sixteen global embedding row IDs per physical token.

S is a prepared physical bucket; lengths can change without recapture. Invalid
tail rows produce zero and do not advance history. A masked *logical* token
is EOS and advances history, matching the HF model's conv_mask boundary.
Set reset-history for a fresh request; otherwise history must contain the
request's preceding two tokens, including raw EOS tokens. Each row has its
own independent length/reset/history. Maximum batch is 32, maximum S 262144.

Prepare every bucket before trace capture. ``forward`` launches one generic
TTNN operation with no tensor allocation, host conversion, or token inspection.
The last tensor is the generic op output; history is an explicit in-place
side effect. Keep all four buffers alive for every trace using this object.
"""

MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)
PRIMES = (
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
)
OFFSETS = tuple(sum(PRIMES[:i]) for i in range(16))
VOCAB_SIZE = 248320
EOS = 248044


# Kept in this module so the exact host arithmetic probe can compile the same
# implementation as ordinary C++, without importing TTNN or touching a device.
_INTEGER_SOURCE = r"""
struct Limbs { uint32_t lo; uint32_t hi; };

inline Limbs mul32(uint32_t a, uint32_t b) {
    const uint32_t a0 = a & 65535u, a1 = a >> 16;
    const uint32_t b0 = b & 65535u, b1 = b >> 16;
    const uint32_t w0 = a0 * b0;
    const uint32_t t = a1 * b0 + (w0 >> 16);
    const uint32_t w1 = a0 * b1 + (t & 65535u);
    return {(w1 << 16) | (w0 & 65535u), a1 * b1 + (t >> 16) + (w1 >> 16)};
}

inline Limbs product(uint32_t token, uint32_t multiplier_lo, uint32_t multiplier_hi) {
    Limbs result = mul32(token, multiplier_lo);
    result.hi += token * multiplier_hi;
    return result;
}

inline uint32_t remainder(Limbs value, uint32_t prime) {
    uint32_t r = 0;
    // r < prime <= 20000171: doubling never overflows a UINT32.
    for (int bit = 31; bit >= 0; --bit) {
        r = (r << 1) | ((value.hi >> bit) & 1u);
        if (r >= prime) r -= prime;
    }
    for (int bit = 31; bit >= 0; --bit) {
        r = (r << 1) | ((value.lo >> bit) & 1u);
        if (r >= prime) r -= prime;
    }
    return r;
}
"""


def kernel_source():
    """Build static source from the pinned checkpoint constants (setup only)."""
    lo = ", ".join(f"{x & 0xFFFFFFFF}u" for x in MULTIPLIERS)
    hi = ", ".join(f"{x >> 32}u" for x in MULTIPLIERS)
    primes = ", ".join(f"{x}u" for x in PRIMES)
    offsets = ", ".join(f"{x}u" for x in OFFSETS)
    return (
        '#include <stdint.h>\n#include "api/dataflow/dataflow_api.h"\n'
        + _INTEGER_SOURCE
        + f"\nconstexpr uint32_t ML[3] = {{{lo}}};\n"
        + f"constexpr uint32_t MH[3] = {{{hi}}};\n"
        + f"constexpr uint32_t P[16] = {{{primes}}};\n"
        + f"constexpr uint32_t O[16] = {{{offsets}}};\n"
        + f"constexpr uint32_t EOS = {EOS}u;\n"
        + r"""
void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr auto ta = TensorAccessorArgs<1>();
    constexpr auto ca = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto tokens = TensorAccessor(ta, get_arg_val<uint32_t>(0), 64);
    const auto control = TensorAccessor(ca, get_arg_val<uint32_t>(1), 64);
    const auto history = TensorAccessor(ha, get_arg_val<uint32_t>(2), 64);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(3), 64);
    const uint32_t row = get_arg_val<uint32_t>(4);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    volatile tt_l1_ptr uint32_t* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 64);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 128);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 192);
    noc_async_read(history.get_noc_addr(row), scratch, 64);
    noc_async_read(control.get_noc_addr(row), scratch + 64, 64);
    noc_async_read_barrier();
    ASSERT(c[0] <= S);
    ASSERT(c[1] <= 1);
    uint32_t oldest = c[1] ? EOS : h[0];
    uint32_t newest = c[1] ? EOS : h[1];
    ASSERT(oldest < 248320u && newest < 248320u);
    for (uint32_t position = 0; position < S; ++position) {
        if (position < c[0]) {
            noc_async_read(tokens.get_noc_addr(row * S + position), scratch + 128, 64);
            noc_async_read_barrier();
            ASSERT(t[1] <= 1);
            const uint32_t current = t[1] ? t[0] : EOS;
            ASSERT(current < 248320u);
            const Limbs a = product(current, ML[0], MH[0]);
            const Limbs b = product(newest, ML[1], MH[1]);
            // A previous EOS blocks older history. A *current* EOS does not.
            const Limbs d = product(newest == EOS ? EOS : oldest, ML[2], MH[2]);
            const Limbs mixed2 = {a.lo ^ b.lo, a.hi ^ b.hi};
            const Limbs mixed3 = {mixed2.lo ^ d.lo, mixed2.hi ^ d.hi};
            for (uint32_t head = 0; head < 16; ++head)
                o[head] = remainder(head < 8 ? mixed2 : mixed3, P[head]) + O[head];
            oldest = newest;
            newest = current;
        } else {
            for (uint32_t head = 0; head < 16; ++head) o[head] = 0;
        }
        noc_async_write(scratch + 192, output.get_noc_addr(row * S + position), 64);
        // Reusing this scratch row requires write completion, including tails.
        noc_async_write_barrier();
    }
    h[0] = oldest;
    h[1] = newest;
    noc_async_write(scratch, history.get_noc_addr(row), 64);
    noc_async_write_barrier();
}
"""
    )


class DeviceNGramHash:
    """Prepared hash bucket with caller-owned, stable device buffers."""

    def __init__(self, *, tokens, control, history, output):
        import ttnn

        self._ttnn = ttnn
        self.tensors = [tokens, control, history, output]
        if len(tokens.shape) != 3 or tokens.shape[-1] != 16:
            raise ValueError("tokens must have physical shape [B,S,16]")
        batch, sequence, _ = tuple(tokens.shape)
        if not (1 <= batch <= 32 and 1 <= sequence <= 262144):
            raise ValueError("Expected 1 <= B <= 32 and 1 <= S <= 262144")
        if tuple(control.shape) != (batch, 16) or tuple(history.shape) != (batch, 16):
            raise ValueError("control and history must be [B,16]")
        if tuple(output.shape) != tuple(tokens.shape):
            raise ValueError("output must be [B,S,16]")
        for tensor in self.tensors:
            if tensor.dtype != ttnn.uint32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise ValueError("Hash buffers must be UINT32 ROW_MAJOR")
            if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise ValueError("Hash buffers must be interleaved DRAM")
        addresses = [t.buffer_address() for t in self.tensors]
        if len(set(addresses)) != 4:
            raise ValueError("Hash buffers must be separate allocations")
        grid = tokens.device().compute_with_storage_grid_size()
        if grid.x * grid.y < batch:
            raise ValueError("One core per batch row is required")
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(batch)]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
        args = ttnn.RuntimeArgs()
        for row, core in enumerate(cores):
            args[core.x][core.y] = [*addresses, row]
        compile_args = [sequence]
        for tensor in self.tensors:
            compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        self.program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=kernel_source(),
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ranges,
                    compile_time_args=compile_args,
                    runtime_args=args,
                    config=ttnn.WriterConfigDescriptor(),
                )
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=256,
                    core_ranges=ranges,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=0,
                            data_format=ttnn.uint32,
                            page_size=256,
                        )
                    ],
                )
            ],
            semaphores=[],
        )

    def prepare(self):
        """Compile/cache without advancing history; call before any capture."""
        self._ttnn.experimental.prepare_generic_op(self.tensors, self.program)

    def forward(self):
        """Update explicit history and return the preallocated [B,S,16] IDs."""
        return self._ttnn.generic_op(self.tensors, self.program)
