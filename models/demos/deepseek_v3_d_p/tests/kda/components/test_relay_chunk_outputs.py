# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The SP relay must leave every rank its chunk outputs, each formed from the summary's terms and the entry state the
gather + chain gives the rank (the separated tail's from the relayed final state), and the final carry on every rank,
call after call and bit-identically: on torus, unwrapped 2D and 1D ring fabrics, with Kimi-K3's per-chip workload (640
rows of 24 heads) and fabric payload, and within its device-time reference.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import assert_accurate, make_actual_start
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    DeviceMismatch,
    assert_device_time,
    closed_form_tensor,
    distributed_prefix,
    gated_rms_norm_reference,
    kda_mesh_param,
    median_device_us,
    pack_complement,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import _sp_links
from models.demos.deepseek_v3_d_p.tt.kda.relay_chunk_outputs import (
    OutputNorm,
    RelayChunkOutputs,
    reset_persistent_semaphores,
)
from tt_py_test_utils_common.realtime_profiler_utils import require_realtime_profiler

pytestmark = run_for_blackhole()

_ROWS, _HEADS, _KEY = 640, 24, 128
_CHUNKS = _ROWS // 32
# The gated output norm: its gate starts a tile column into a [1, T, K + H*V] gate tensor.
_GATE_OFFSET, _NORM_EPSILON = _KEY, 1e-6
# The model's relay runs at the affine prefix's fidelity, on the model's cores.
_PROGRAM_CONFIG = kimi_k3_program_config(active_seq_len_local=_ROWS, tp_ccl_topology=ttnn.Topology.Linear).recurrence
_MATH_FIDELITY = _PROGRAM_CONFIG.affine_prefix_math_fidelity


def _fabric_1d_ring_params() -> dict:
    """FABRIC_1D_RING with K3's payload."""
    return fabric_1d_device_params(
        fabric_config=ttnn.FabricConfig.FABRIC_1D_RING, fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE
    )


# SP takes mesh axis 0.
_MESH = pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        kda_mesh_param((8, 4), "mesh-8x4", "SP8xTP4"),
        kda_mesh_param(
            (8, 4),
            "mesh-8x4",
            "SP8xTP4-fabric-2d",
            fabric2d_device_params(fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE),
        ),
        # FABRIC_1D: on a mesh descriptor without wrap links the last rank reaches the first sp_size - 1 hops back.
        kda_mesh_param(
            (8, 4),
            "mesh-8x4",
            "SP8xTP4-fabric-1d-line",
            fabric_1d_device_params(fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE),
        ),
        kda_mesh_param((8, 1), "ring", "loudbox-SP8xTP1-ring"),
        kda_mesh_param((8, 1), "ring", "loudbox-SP8xTP1-fabric-1d-ring", _fabric_1d_ring_params()),
        kda_mesh_param((4, 2), "mesh-4x2", "loudbox-SP4xTP2-fabric-2d"),
        kda_mesh_param((2, 4), "mesh-2x4", "loudbox-SP2xTP4-fabric-2d"),
        kda_mesh_param((4, 1), "ring", "quietbox-SP4xTP1-ring"),
        kda_mesh_param((4, 1), "ring", "quietbox-SP4xTP1-fabric-1d-ring", _fabric_1d_ring_params()),
    ],
    indirect=True,
)


def _rank_chunks(sp_size: int, start: int, end: int | None) -> dict[int, tuple[int, int]]:
    """Chronology by hand: per rank, its valid chunks and the chunk its separated tail starts at (0: no tail). Only the
    rank holding the first token splits, and only when the interval reaches back into it."""
    offset = start % _ROWS
    first_rank = (start // _ROWS) % sp_size
    head_rows = _ROWS - offset
    length = None if end is None else end - start
    chunks = {}
    for rank in range(sp_size):
        step = (rank - first_rank) % sp_size
        if length is None:
            valid_rows = _ROWS
        elif step == 0:
            valid_rows = min(length, head_rows) + max(0, length - head_rows - (sp_size - 1) * _ROWS)
        else:
            valid_rows = min(max(0, length - head_rows - (step - 1) * _ROWS), _ROWS)
        reset = head_rows // 32 if step == 0 and offset != 0 and valid_rows > head_rows else 0
        chunks[rank] = ((valid_rows + 31) // 32, reset)
    return chunks


class _Transforms:
    """One transform per rank, its packed complement form [E | B] (E = A - I) on device, the chunk terms, the final
    transition and the output norm, and the FP64 reference of the relay around the ring."""

    def __init__(self, mesh_device, relay, seed: int | None, batch: int = 1, output_dtype=ttnn.bfloat16):
        """Random transforms from ``seed``, or closed-form ones (for the determinism check) without one; the heads are
        ``batch`` batches in the norm's gate and output."""
        self.sp_size, self.tp_size = tuple(mesh_device.shape)
        self.batch, gate_width = batch, _GATE_OFFSET + _HEADS // batch * _KEY
        if seed is None:
            shape = (self.sp_size, _HEADS, _KEY, _KEY)
            a, b = closed_form_tensor(*shape, offset=-0.125), closed_form_tensor(*shape, offset=0.0625)
            self.initial = closed_form_tensor(_HEADS, _KEY, _KEY, offset=0.03)
            self.zero = closed_form_tensor(_HEADS, _CHUNKS, 32, _KEY, offset=-0.125)
            self.identity = self.zero.flip(-1)
            self.final_e = closed_form_tensor(_HEADS, _KEY, _KEY, offset=-0.1)
            self.final_b = closed_form_tensor(_HEADS, _KEY, _KEY, offset=0.05)
            self.gate = closed_form_tensor(batch, _ROWS, gate_width, offset=0.2)
            self.weight = closed_form_tensor(_KEY, offset=1.0)
        else:
            generator = torch.Generator().manual_seed(seed)
            a = 0.1 * torch.randn(self.sp_size, _HEADS, _KEY, _KEY, generator=generator)
            b = 0.1 * torch.randn(self.sp_size, _HEADS, _KEY, _KEY, generator=generator)
            self.initial = 0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator)
            self.zero = 0.1 * torch.randn(_HEADS, _CHUNKS, 32, _KEY, generator=generator)
            self.identity = 0.1 * torch.randn(_HEADS, _CHUNKS, 32, _KEY, generator=generator)
            self.final_e = 0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator)
            self.final_b = 0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator)
            self.gate = torch.randn(batch, _ROWS, gate_width, generator=generator)
            self.weight = 1 + 0.1 * torch.randn(_KEY, generator=generator)
        self.gate, self.weight = self.gate.bfloat16().float(), self.weight.bfloat16().float()
        # A BF16-representable E and B make the BF16 transport exact for both paths; A = E + I exactly.
        e, self.b = (a - torch.eye(_KEY)).bfloat16().float(), b.bfloat16().float()
        self.a = e + torch.eye(_KEY)
        self.packed = _pack(mesh_device, self.a, self.b)
        self.initial_tt = _replicated(mesh_device, self.initial)
        self.inputs = _inputs(mesh_device, relay, self.zero, self.identity, self.final_e, self.final_b)
        self.norm = OutputNorm(
            gate=_replicated(mesh_device, self.gate, dtype=ttnn.bfloat16),
            weight=_replicated(mesh_device, self.weight, dtype=ttnn.bfloat16),
            num_heads=_HEADS // batch,
            gate_column_offset=_GATE_OFFSET,
            norm_epsilon=_NORM_EPSILON,
            output_dtype=output_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=_MATH_FIDELITY, fp32_dest_acc_en=True, packer_l1_acc=False
        )

    def head_major(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token-major [B, T, H*V] as [B*H, chunks, 32, V]."""
        heads = _HEADS // self.batch
        return (
            tokens.reshape(self.batch, _CHUNKS, 32, heads, _KEY)
            .permute(0, 3, 1, 2, 4)
            .reshape(_HEADS, _CHUNKS, 32, _KEY)
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """The gated output norm of [B*H, chunks, 32, V] chunk outputs."""
        gate = self.head_major(self.gate[:, :, _GATE_OFFSET:])
        return gated_rms_norm_reference(x, gate, self.weight, _NORM_EPSILON)

    def reference(self, start: int):
        """Each rank's entry state and the relayed final state P."""
        first_rank = (start // _ROWS) % self.sp_size
        carry = self.initial.double()
        entries = {}
        for step in range(self.sp_size):
            rank = (first_rank + step) % self.sp_size
            entries[rank] = carry
            carry = self.a[rank].double() @ carry + self.b[rank].double()
        return entries, carry

    def assert_matches_reference(self, output, carry, actual_start, start: int, end: int | None = None):
        """Each rank's valid chunk outputs from its entry state (a separated tail's from P), and every rank's carry:
        P + (E P + B) with the final transition when the interval ends in a separated tail, else P. An unsplit carry
        is P itself, which must be no further from the FP64 reference than the gather + chain's final state."""
        entries, final = self.reference(start)
        chunks = _rank_chunks(self.sp_size, start, end)
        split = any(reset for _, reset in chunks.values())
        expected_carry = final + (self.final_e.double() @ final + self.final_b.double()) if split else final
        zero, identity = self.zero.double(), self.identity.double()
        chained_final = None
        if not split:
            _, chained = distributed_prefix(
                self.packed,
                self.initial_tt,
                sequence_parallel_axis=0,
                compute_config=self.compute_config,
                actual_start=actual_start,
                local_rows=_ROWS,
            )
            chained_final = ttnn.to_torch(ttnn.get_device_tensors(chained)[0]).double()
        for index, (local_output, local_carry) in enumerate(
            zip(ttnn.get_device_tensors(output), ttnn.get_device_tensors(carry), strict=True)
        ):
            rank, tp = divmod(index, self.tp_size)
            name = f"start={start} end={end} rank={rank} tp={tp}"
            valid, reset = chunks[rank]
            expected = zero + (identity - zero) @ entries[rank].unsqueeze(1)
            if reset:
                expected[:, reset:] = (zero + (identity - zero) @ final.unsqueeze(1))[:, reset:]
            expected = self.normalize(expected)
            got = self.head_major(ttnn.to_torch(local_output).double())
            assert_accurate(
                expected[:, :valid].float(),
                got[:, :valid].float(),
                name=f"output {name}",
                pcc_threshold=0.9999,
                nrmse_threshold=0.01,
            )
            got_carry = ttnn.to_torch(local_carry).double()
            assert_accurate(expected_carry.float(), got_carry.float(), name=f"carry {name}", nrmse_threshold=0.03)
            if chained_final is not None:
                relay_error = torch.linalg.vector_norm(got_carry - expected_carry).item()
                chain_error = torch.linalg.vector_norm(chained_final - expected_carry).item()
                assert relay_error <= 1.01 * chain_error, f"carry {name}: relay {relay_error} > chain {chain_error}"


def _replicated(
    mesh_device, host: torch.Tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32
) -> ttnn.Tensor:
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _inputs(mesh_device, relay, zero, identity, final_e, final_b) -> tuple[ttnn.Tensor, ...]:
    """The chunk terms F(0) and F(I) in the relay's terms layout (each core's chunk rows of F(0), then of F(I), in its
    shard; or per head, streamed), and the final transition."""
    blocks = _HEADS * relay.cores_per_head if relay.terms_memory_config.is_sharded() else _HEADS
    terms = torch.stack([zero.reshape(blocks, -1, _KEY), identity.reshape(blocks, -1, _KEY)], dim=1)
    return (
        _replicated(mesh_device, terms, relay.terms_memory_config),
        _replicated(mesh_device, final_e),
        _replicated(mesh_device, final_b),
    )


def _pack(mesh_device, a: torch.Tensor, b: torch.Tensor) -> ttnn.Tensor:
    """Each rank's [sp_size, heads, K, K] A and B as its BF16 [1, heads, K, 2K] rows of [E | B], replicated over TP."""
    sp_size, tp_size = tuple(mesh_device.shape)
    return pack_complement(
        a, b, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(sp_size, tp_size))
    )


def _relay(
    mesh_device, math_fidelity=_MATH_FIDELITY, max_cores=_PROGRAM_CONFIG.entry_correction_max_cores
) -> RelayChunkOutputs:
    return RelayChunkOutputs(
        mesh_device,
        sequence_parallel_axis=0,
        batch_heads=_HEADS,
        key_dim=_KEY,
        value_dim=_KEY,
        num_links=_sp_links(mesh_device, 0),
        local_rows=_ROWS,
        max_cores=max_cores,
        math_fidelity=math_fidelity,
    )


def _run(relay: RelayChunkOutputs, transforms: _Transforms, actual_start, actual_end=None):
    terms, final_e, final_b = transforms.inputs
    return relay(
        transforms.packed, terms, transforms.initial_tt, final_e, final_b, transforms.norm, actual_start, actual_end
    )


# Only the rank holding the first token orders the relay; "split" checks that an offset within that rank is ignored by
# the relay and starts the tail's chunks (and the carry's final transition) from the final state. The bounded intervals
# end two chunks before the last rank's end, and three chunks into a split rank's tail.
_INTERVALS = {
    "aligned": (0, None),
    "rotated": (640, None),
    "split": (672, None),
    "late-split": (3232, None),
    "last-rank": (4480, None),
    "bounded": (0, -64),
    "split-bounded": (992, -256),
}


def _bounds(mesh_device, name: str):
    """The interval's start and optional exclusive end, and their device scalars."""
    start, length_delta = _INTERVALS[name]
    end = None if length_delta is None else start + tuple(mesh_device.shape)[0] * _ROWS + length_delta
    return (
        start,
        end,
        make_actual_start(mesh_device, start),
        None if end is None else make_actual_start(mesh_device, end),
    )


@_MESH
@pytest.mark.parametrize("interval", list(_INTERVALS))
def test_relay_chunk_outputs_matches_reference(mesh_device, interval):
    start, end, actual_start, actual_end = _bounds(mesh_device, interval)
    relay = _relay(mesh_device)
    transforms = _Transforms(mesh_device, relay, seed=start + (end or 0))
    output, carry = _run(relay, transforms, actual_start, actual_end)
    transforms.assert_matches_reference(output, carry, actual_start, start, end)


# Core layouts besides K3's four cores per head: one core per head, whose share of the terms is over the L1 budget and
# streams from DRAM (and whose hop core corrects every chunk), and two, with the terms in place.
_LAYOUTS = {"one-core-per-head": (_HEADS, ttnn.BufferType.DRAM), "two-cores-per-head": (2 * _HEADS, ttnn.BufferType.L1)}


@_MESH
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_relay_chunk_outputs_core_layouts(mesh_device, layout):
    max_cores, terms_buffer = _LAYOUTS[layout]
    relay = _relay(mesh_device, max_cores=max_cores)
    assert relay.terms_memory_config.buffer_type == terms_buffer
    start, end, actual_start, actual_end = _bounds(mesh_device, "split-bounded")
    transforms = _Transforms(mesh_device, relay, seed=start + end)
    output, carry = _run(relay, transforms, actual_start, actual_end)
    transforms.assert_matches_reference(output, carry, actual_start, start, end)


# The norm's output in FP32, and the heads as two batches (each with its own gate rows and output rows).
_NORM_CASES = {"float32-output": (1, ttnn.float32), "two-batches": (2, ttnn.bfloat16)}


@_MESH
@pytest.mark.parametrize("case", list(_NORM_CASES))
def test_relay_chunk_outputs_output_norm(mesh_device, case):
    batch, output_dtype = _NORM_CASES[case]
    start, end, actual_start, actual_end = _bounds(mesh_device, "split-bounded")
    relay = _relay(mesh_device)
    transforms = _Transforms(mesh_device, relay, seed=start + end, batch=batch, output_dtype=output_dtype)
    output, carry = _run(relay, transforms, actual_start, actual_end)
    assert output.dtype == output_dtype
    transforms.assert_matches_reference(output, carry, actual_start, start, end)


@_MESH
def test_relay_chunk_outputs_applies_the_complement_form(mesh_device):
    """Hand-checkable relay of the complement form: rank r carries A = I / 2 (E = -I / 2) and B = (r + 1) I, except
    rank 3, which carries the identity map, (0, 0) as an empty rank's summary publishes it. From S = 10 I every entry
    state is a small dyadic multiple of I, exact in FP32 and in the matmuls' source registers, so every rank's carry must
    match the hand value bit for bit. A relay that read E as A would send -4 I from rank 0, and 0 from rank 3."""
    sp_size, tp_size = tuple(mesh_device.shape)
    eye = torch.eye(_KEY)
    scales = [(0.0, 0.0) if rank == 3 else (-0.5, rank + 1.0) for rank in range(sp_size)]
    relay = _relay(mesh_device)
    transforms = _Transforms(mesh_device, relay, seed=0)
    e = torch.stack([torch.stack([e_scale * eye] * _HEADS) for e_scale, _ in scales])
    b = torch.stack([torch.stack([b_scale * eye] * _HEADS) for _, b_scale in scales])
    transforms.packed = _pack(mesh_device, e + eye, b)
    transforms.initial_tt = _replicated(mesh_device, 10 * eye.expand(_HEADS, _KEY, _KEY).contiguous())
    zero = torch.zeros(_HEADS, _KEY, _KEY)
    transforms.inputs = _inputs(mesh_device, relay, transforms.zero, transforms.identity, zero, zero)
    # From rank 0: 10, 6, 5, 5.5, 5.5 (identity), 7.75, 9.875, 11.9375 for 8 ranks; the hand recurrence below.
    entries, carry = [], 10.0
    for e_scale, b_scale in scales:
        entries.append(carry)
        carry = carry + e_scale * carry + b_scale
    if sp_size == 8:
        assert entries == [10.0, 6.0, 5.0, 5.5, 5.5, 7.75, 9.875, 11.9375] and carry == 13.96875
    _, final = _run(relay, transforms, make_actual_start(mesh_device, 0))
    for index, local_final in enumerate(ttnn.get_device_tensors(final)):
        rank = index // tp_size
        torch.testing.assert_close(
            ttnn.to_torch(local_final),
            carry * eye.expand(_HEADS, _KEY, _KEY),
            rtol=0,
            atol=0,
            msg=f"carry rank={rank}",
        )


# (start, end) as functions of the SP span: the interval's tail separates from its head on the first rank only when
# the start is not rank-aligned and the interval reaches past the start of that tail (span - 32 rows after a 32-row
# offset into the first rank).
_TAIL_BOUNDS = {
    "aligned": lambda span: (_ROWS, None),
    "split": lambda span: (_ROWS + 32, None),
    "end-past-tail": lambda span: (_ROWS + 32, _ROWS + 32 + span),
    "end-at-tail": lambda span: (32, span),
    "end-before-tail": lambda span: (32, span - 32),
}


@_MESH
@pytest.mark.parametrize("bounds", list(_TAIL_BOUNDS.values()), ids=list(_TAIL_BOUNDS))
def test_relay_chunk_outputs_applies_the_final_transition_of_a_separated_tail(mesh_device, bounds):
    """The carry is the relayed final state P, or P + E P + B with the first rank's final transition (E, B) when the
    interval ends in a separated tail. Every rank holds a distinct, non-zero final transition, so a carry formed
    from the wrong rank's transition, or one applied (or skipped) on the wrong side of the tail's start, cannot
    match."""
    sp_size, tp_size = tuple(mesh_device.shape)
    start, end = bounds(sp_size * _ROWS)
    relay = _relay(mesh_device)
    transforms = _Transforms(mesh_device, relay, seed=start)
    generator = torch.Generator().manual_seed(start + 1)
    final_e = 0.1 * torch.randn(sp_size, _HEADS, _KEY, _KEY, generator=generator)
    final_b = 0.1 * torch.randn(sp_size, _HEADS, _KEY, _KEY, generator=generator)
    final_e_tt, final_b_tt = (
        ttnn.from_torch(
            t.reshape(sp_size * _HEADS, _KEY, _KEY),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(sp_size, tp_size)),
        )
        for t in (final_e, final_b)
    )
    actual_start = make_actual_start(mesh_device, start)
    actual_end = None if end is None else make_actual_start(mesh_device, end)
    terms, _, _ = transforms.inputs
    output, carry = relay(
        transforms.packed,
        terms,
        transforms.initial_tt,
        final_e_tt,
        final_b_tt,
        transforms.norm,
        actual_start,
        actual_end,
    )

    _, prefix = transforms.reference(start)
    offset, first_rank = start % _ROWS, (start // _ROWS) % sp_size
    separated = offset != 0 and (end is None or end - start > (_ROWS - offset) + (sp_size - 1) * _ROWS)
    expected = prefix.double()
    if separated:
        expected = expected + final_e[first_rank].double() @ expected + final_b[first_rank].double()
    for index, local_carry in enumerate(ttnn.get_device_tensors(carry)):
        rank, tp = divmod(index, tp_size)
        assert_accurate(
            expected.float(),
            ttnn.to_torch(local_carry),
            name=f"carry start={start} end={end} rank={rank} tp={tp}",
            nrmse_threshold=0.03,
        )
    for tensor in (output, carry, final_e_tt, final_b_tt, actual_start, actual_end):
        if tensor is not None:
            ttnn.deallocate(tensor)


@_MESH
def test_relay_chunk_outputs_adds_the_entry_state_and_b_in_fp32(mesh_device):
    """With E = 0 each hop is S + B: a random FP32 initial state and BF16 B with full mantissas must reach the carry as
    their FP32 sum over every hop, to FP32 rounding. A source register would truncate the entry state to TF32, and B
    in srcB to 6 mantissa bits at HiFi2: either is an error of order 1e-4 here, against the 1e-6 allowed."""
    sp_size, _ = tuple(mesh_device.shape)
    relay = _relay(mesh_device)
    transforms = _Transforms(mesh_device, relay, seed=17)
    generator = torch.Generator().manual_seed(17)
    b = (0.1 * torch.randn(sp_size, _HEADS, _KEY, _KEY, generator=generator)).bfloat16().float()
    initial = 0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator)
    transforms.packed = _pack(mesh_device, torch.eye(_KEY).expand_as(b), b)
    transforms.initial_tt = _replicated(mesh_device, initial)
    _, carry = _run(relay, transforms, make_actual_start(mesh_device, 0))
    expected = initial + b.sum(dim=0)
    for index, local_carry in enumerate(ttnn.get_device_tensors(carry)):
        torch.testing.assert_close(ttnn.to_torch(local_carry), expected, rtol=0, atol=1e-6, msg=f"carry device={index}")


@_MESH
def test_relay_chunk_outputs_rejects_low_fidelity(mesh_device, expect_error):
    """Below HiFi3 the norm's FPU products truncate 1 / rms and the squares: HiFi2 is refused."""
    relay = _relay(mesh_device, math_fidelity=ttnn.MathFidelity.HiFi2)
    transforms = _Transforms(mesh_device, relay, seed=0)
    with expect_error(RuntimeError, "needs HiFi3 or above"):
        _run(relay, transforms, make_actual_start(mesh_device, 0))


@_MESH
def test_relay_chunk_outputs_back_to_back(mesh_device):
    """Back-to-back calls, separated only by every rank waiting for the previous call's final carry: the second call
    must be as exact as the first, from the landing buffers and arrival counters the first left."""
    # Rank-aligned starts, the second rotated.
    starts = (0, 640)
    relay = _relay(mesh_device)
    transforms = _Transforms(mesh_device, relay, seed=starts[0])
    results = []
    for start in starts:
        actual_start = make_actual_start(mesh_device, start)
        results.append((start, actual_start, *_run(relay, transforms, actual_start)))
    for start, actual_start, output, carry in results:
        transforms.assert_matches_reference(output, carry, actual_start, start)


@_MESH
def test_relay_chunk_outputs_after_counter_reset(mesh_device):
    """Counts an aborted call left on the persistent counters are zeroed by reset_persistent_semaphores, and the next
    call is exact."""
    start = 640
    relay = _relay(mesh_device)
    transforms = _Transforms(mesh_device, relay, seed=start)
    for semaphore in relay.semaphores:
        ttnn.reset_global_semaphore_value(semaphore, 3)
    reset_persistent_semaphores(mesh_device)
    actual_start = make_actual_start(mesh_device, start)
    output, carry = _run(relay, transforms, actual_start)
    transforms.assert_matches_reference(output, carry, actual_start, start)


_DETERMINISM_CALLS = 3


@_MESH
def test_relay_chunk_outputs_determinism(mesh_device):
    """Every full interval's calls return bit-identical outputs and carries (a bounded interval leaves its invalid
    chunks unwritten)."""
    relay = _relay(mesh_device)
    transforms = _Transforms(mesh_device, relay, seed=None)
    mismatch = DeviceMismatch(mesh_device)
    for name, (start, length_delta) in _INTERVALS.items():
        if length_delta is not None:
            continue
        actual_start = make_actual_start(mesh_device, start)
        first = None
        for _ in range(_DETERMINISM_CALLS):
            outputs = _run(relay, transforms, actual_start)
            if first is None:
                first = outputs
                continue
            mismatch.compare(first[0], outputs[0], f"{name} output")
            mismatch.compare(first[1], outputs[1], f"{name} carry")
            for tensor in outputs:
                ttnn.deallocate(tensor)
        for tensor in (*first, actual_start):
            ttnn.deallocate(tensor)
    mismatch.assert_none()


# Device time of one relay call: its program's slowest chip under the realtime profiler, the median over single calls,
# each after a discarded first and never back to back (a device synchronize separates them, the barrier consecutive
# calls need). A few single calls run slow while the ring waits on its predecessors; the median of 15 does not.
_PERF_SAMPLES = 15
# A full chunk that starts 320 rows into the second rank, as the layer gates place it: the relay starts mid-ring.
_PERF_START = _ROWS + 320
_PERF_KERNEL = "relay_chunk_outputs"


def _perf_case(shape: tuple[int, int], topology: str, case_id: str, reference_us: float, margin: float):
    """A mesh case with its device-time reference and margin."""
    case = kda_mesh_param(shape, topology, case_id)
    return pytest.param(*case.values, reference_us, margin, marks=case.marks, id=case_id)


# Device-time references in us, with K3's payload: Galaxy on TorusXY, LoudBox and QuietBox on TorusY.
_PERF_CASES = [
    _perf_case((8, 4), "mesh-8x4", "SP8xTP4", 174.0, 0.10),
    _perf_case((8, 1), "ring", "loudbox-SP8xTP1-ring", 388.0, 0.12),
    _perf_case((4, 1), "ring", "quietbox-SP4xTP1-ring", 75.5, 0.05),
]


@pytest.mark.parametrize(
    "mesh_device,device_params,reference_us,margin", _PERF_CASES, indirect=["mesh_device", "device_params"]
)
def test_relay_chunk_outputs_kimi_k3_perf(mesh_device, reference_us, margin):
    """The model's relay: 24 heads per chip, the model's fidelity and cores, the terms in L1."""
    require_realtime_profiler("the relay_chunk_outputs device-time gate")
    layout = "SP{}xTP{}".format(*tuple(mesh_device.shape))
    relay = _relay(mesh_device)
    transforms = _Transforms(mesh_device, relay, seed=_PERF_START)
    actual_start = make_actual_start(mesh_device, _PERF_START)

    def call():
        outputs = _run(relay, transforms, actual_start)
        ttnn.synchronize_device(mesh_device)
        return outputs

    output, carry = call()
    transforms.assert_matches_reference(output, carry, actual_start, _PERF_START)
    for tensor in (output, carry):
        ttnn.deallocate(tensor)

    median_us = median_device_us(mesh_device, call, _PERF_KERNEL, _PERF_SAMPLES)
    assert_device_time(median_us, reference_us, margin, f"relay_chunk_outputs Kimi-K3 {layout}")
