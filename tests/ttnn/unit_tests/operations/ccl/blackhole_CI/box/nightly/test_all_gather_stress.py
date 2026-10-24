# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# all_gather stress on a BH (1,8) box, every algorithm forced through TTNN_ALL_GATHER_FORCE. Bit-exact on every
# device.
#
# Traced: one gather captured in a trace and replayed back to back in blocks of BLOCK; before each block the input
# gets new data and the output a sentinel, after it the output is checked. A replay that starts before the previous
# one's handshakes have drained, or a signal counted twice, hangs or corrupts the block's last replay. rdelay /
# wdelay stall the reader / writer per slot, shifting which side waits on which semaphore. Each test then replays a
# second and a third shape's traces (the third with narrow row-major rows: about 30 short packets per slot) and returns
# to the first. The output is persistent, or allocated by the op at capture.
#
# Overlap: cached gathers back to back with no sync, a different input each time, into one reused persistent output,
# with every device but one stalled after each launch and the output copied out behind it. Without the init
# handshake a fast device's next launch overwrites a neighbor's output before that neighbor has relayed or copied
# it.
#
# L1-heavy: L1-interleaved input and output, pushed down towards the L1 base by a filler tensor that leaves about a
# quarter of L1 free. The tensors must still lie above L1_FLOOR (past any mux map); the test asserts that
# precondition, then replays traced.
#
# Mux L1 guard: the unicast mux map is raw L1 from the allocator base. With an L1 tensor reaching below its end (but
# above the slot buffers), unicast must refuse to launch, and multicast (no mux) must still run.

import pytest
import ttnn
from loguru import logger

from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.all_gather_test_utils import (
    L1,
    LINE,
    RING,
    RM,
    BF16,
    Case,
    check,
    device_input,
    fabric_params,
    gather,
    host_input,
    sentinel_host,
    sentinel_output,
    set_force,
)

BLOCK = 100
TOPOLOGIES = [fabric_params(LINE, 500_000), fabric_params(RING, 500_000)]
TOPOLOGY_IDS = ["line", "ring"]
ALGOS = ["uc_n2", "uc_n3", "mc"]

SHAPE_A = Case([1, 1, 256, 8192], 3)  # 4 MB: several slots per iteration, relays on every device
SHAPE_B = Case([1, 1, 224, 2816], 3)  # 11 tiles x 7 rows per device: odd slices that start mid-row
SHAPE_K = Case([1, 1, 2048, 1024], 3, layout=RM)  # 256 B per device per row: 1 KB packet cap, k = 30 per slot

# id -> (replays of shape A, rdelay, wdelay, marks)
DELAYS = {
    "d0_x1000": (1000, 0, 0, [pytest.mark.slow]),
    "r300": (BLOCK, 300, 0, []),
    "r1000": (BLOCK, 1000, 0, []),
    "w300": (BLOCK, 0, 300, []),
    "w1000": (BLOCK, 0, 1000, []),
}


class Traced:
    """One gather captured in a trace, with its own input and output: persistent, or allocated by the op at capture
    (the trace then writes that buffer on every replay)."""

    def __init__(self, mesh, case, persistent=True):
        self.mesh, self.case = mesh, case
        self.tt_in, expected = device_input(mesh, case, seed=0)
        out = sentinel_output(mesh, case) if persistent else None
        check(gather(self.tt_in, case, out), expected, "compile run")
        self.trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.out = gather(self.tt_in, case, out)
        ttnn.end_trace_capture(mesh, self.trace, cq_id=0)
        ttnn.synchronize_device(mesh)
        self.seed = 1

    def replay(self, replays, label):
        for start in range(0, replays, BLOCK):
            host_in, expected = host_input(self.mesh, self.case, seed=self.seed)
            self.seed += 1
            ttnn.copy_host_to_device_tensor(host_in, self.tt_in)
            ttnn.copy_host_to_device_tensor(sentinel_host(self.mesh, self.case), self.out)
            for _ in range(min(BLOCK, replays - start)):
                ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            check(self.out, expected, f"{label}: replays {start}..{min(start + BLOCK, replays) - 1}")

    def release(self):
        ttnn.release_trace(self.mesh, self.trace)


@skip_for_wormhole_b0()
@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", TOPOLOGIES, indirect=True, ids=TOPOLOGY_IDS)
@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("delay_id", [pytest.param(d, marks=marks, id=d) for d, (_, _, _, marks) in DELAYS.items()])
@pytest.mark.parametrize("output", ["persistent", "fresh"])
def test_all_gather_traced_stress(mesh_device, device_params, algo, delay_id, output, monkeypatch):
    replays, rdelay, wdelay, _ = DELAYS[delay_id]
    set_force(monkeypatch, algo, rdelay, wdelay)
    persistent = output == "persistent"
    label = f"{algo} {delay_id} {output}"
    a = Traced(mesh_device, SHAPE_A, persistent)
    a.replay(replays, f"A {label}")
    b = Traced(mesh_device, SHAPE_B, persistent)
    b.replay(BLOCK, f"B {label}")
    k = Traced(mesh_device, SHAPE_K, persistent)
    k.replay(BLOCK, f"K {label}")
    a.replay(BLOCK, f"A again {label}")
    k.release()
    b.release()
    a.release()


OVERLAP_CASE = Case([1, 1, 1024, 1024], 3)  # 2 MB output
OVERLAP_ITERS = 10
OVERLAP_DELAY = 2_000_000  # cycles: about 1.5 ms on BH, far longer than one gather


@skip_for_wormhole_b0()
@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", TOPOLOGIES, indirect=True, ids=TOPOLOGY_IDS)
@pytest.mark.parametrize("algo", ["uc_n2", "mc"])
def test_all_gather_overlap(mesh_device, device_params, algo, monkeypatch):
    set_force(monkeypatch, algo)
    case = OVERLAP_CASE
    inputs = [device_input(mesh_device, case, seed=i) for i in range(OVERLAP_ITERS)]
    out = sentinel_output(mesh_device, case)
    captures = [sentinel_output(mesh_device, case) for _ in range(OVERLAP_ITERS)]
    gather(inputs[0][0], case, out)  # compile, so every launch below is a cached one
    ttnn.synchronize_device(mesh_device)
    # Stall every device but (0, 0) after each launch, so the fast device's next gather could write into outputs the
    # others have not yet relayed or copied out.
    rows, cols = tuple(mesh_device.shape)
    delays = [[0 if (r, c) == (0, 0) else OVERLAP_DELAY for c in range(cols)] for r in range(rows)]
    for i in range(OVERLAP_ITERS):
        result = gather(inputs[i][0], case, out)
        ttnn.apply_device_delay(mesh_device, delays)
        ttnn.copy(result, captures[i])
    ttnn.synchronize_device(mesh_device)
    for i in range(OVERLAP_ITERS):
        check(captures[i], inputs[i][1], f"overlap {algo} iteration {i}")


# The lowest address the op's L1 tensors may take here: above any mux map (a few tens of KB) with margin.
L1_FLOOR = 256 * 1024
L1_FREE = 0.25  # fraction of L1 left free once filler, input and output are allocated
L1_CASE = Case([1, 1, 256, 4096], 3, mem_in=L1, mem_out=L1)  # 2 MB output
FILLER_ROW = 2048  # filler page: one bf16 row-major row of 1024 elements


def _l1_bytes_per_bank(case, num_devices, num_banks):
    pages_out = case.shape[0] * case.shape[1] * case.shape[2] * case.shape[3] // 1024  # bf16 tiles, 2 KB each
    pages_in = pages_out // num_devices
    return sum(-(-pages // num_banks) * 2048 for pages in (pages_in, pages_out))


@skip_for_wormhole_b0()
@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", TOPOLOGIES, indirect=True, ids=TOPOLOGY_IDS)
@pytest.mark.parametrize("algo", ["uc_n3", "mc"])
def test_all_gather_l1_heavy(mesh_device, device_params, algo, monkeypatch):
    set_force(monkeypatch, algo)
    l1_base = ttnn.get_allocator_base_address(mesh_device, ttnn.BufferType.L1)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    total, free = view.total_bytes_per_bank, view.total_bytes_free_per_bank
    ops_bytes = _l1_bytes_per_bank(L1_CASE, mesh_device.get_num_devices(), view.num_banks)
    filler_bytes = free - int(L1_FREE * total) - ops_bytes
    assert filler_bytes > 0, f"L1 is already fuller than the test wants ({free} of {total} B free per bank)"
    # Allocated first, so it takes the top of L1 and the op's tensors land below it.
    filler = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, filler_bytes // FILLER_ROW * view.num_banks, FILLER_ROW // 2]),
        BF16,
        RM,
        mesh_device,
        L1,
    )
    a = Traced(mesh_device, L1_CASE)
    lowest = min(a.tt_in.buffer_address(), a.out.buffer_address())
    left = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_free_per_bank
    logger.info(
        f"L1 per bank: base {l1_base:#x}, {total} B; op tensors from {lowest:#x} (base + {lowest - l1_base} B); "
        f"{left} B ({100 * left / total:.0f}%) free"
    )
    assert lowest >= l1_base + L1_FLOOR, (
        f"the op's L1 tensors start at base + {lowest - l1_base} B, below base + {L1_FLOOR} B: this test only "
        "covers the regime above any mux map"
    )
    a.replay(BLOCK, f"L1-heavy {algo}")
    a.release()
    filler.deallocate()


# Below the end of a 3-channel mux map (about 95 KB past the base at P = 15168) but above any 2-slot worker CB.
GUARD_GAP = 72 * 1024
GUARD_CASE = Case([1, 1, 256, 2048], 3)  # DRAM in and out: only the filler lives in L1


@skip_for_wormhole_b0()
@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric_params(LINE)], indirect=True, ids=["line"])
def test_all_gather_mux_l1_guard(mesh_device, device_params, monkeypatch, expect_error):
    l1_base = ttnn.get_allocator_base_address(mesh_device, ttnn.BufferType.L1)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    rows_per_bank = (view.total_bytes_free_per_bank - GUARD_GAP) // FILLER_ROW
    filler = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows_per_bank * view.num_banks, FILLER_ROW // 2]), BF16, RM, mesh_device, L1
    )
    gap = filler.buffer_address() - l1_base
    logger.info(f"L1 filler from base + {gap} B")
    assert 64 * 1024 <= gap <= 80 * 1024, f"the filler starts at base + {gap} B, outside the window this test needs"
    tt_in, expected = device_input(mesh_device, GUARD_CASE, seed=0)
    set_force(monkeypatch, "uc_n3")
    with expect_error(RuntimeError, "clash with L1 buffers"):
        gather(tt_in, GUARD_CASE)
    set_force(monkeypatch, "mc")
    out = gather(tt_in, GUARD_CASE)
    ttnn.synchronize_device(mesh_device)
    check(out, expected, "multicast under the guard")
    filler.deallocate()
