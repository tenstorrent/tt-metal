# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""NOC data integrity on OpTestBase: a Python port of tests/tt_metal/tt_metal/deployment/noc/test_noc_data_integrity.cpp.

Temporary port so the NOC integrity screen gets the didt harness: determinism checks against a reference and the VF
sweep (--didt-vf-sweep, --enable-vdd-offsets). The device kernel is the deployment one
(tests/tt_metal/tt_metal/deployment/kernels/noc_integrity_kernel.cpp); everything on the host is here.

Per chip, on NOC0 and then NOC1, every source core writes six data patterns (walking 1s/0s, checkerboard and inverse,
address, PRBS) to its own slice of every destination, in packets with several in flight, reads them back and compares on
the device. One OpTestBase iteration is one such full pass. A destination pair fails on any mismatch, a write/read
barrier timeout, or not being reached at all.

Sources (the `sources` parameter):
  eth     every Ethernet core, active and idle (C++ Noc00). Needs TT_METAL_SLOW_DISPATCH_MODE=1.
  tensix  the four corner Tensix cores (C++ Noc01).
  all     both.
Destinations: every Tensix core (rows limited by NOC_TEST_TENSIX_ROWS), the free L1 of every Ethernet core, and every
DRAM bank (addressed by bank through the firmware's per-NOC bank table).

    TT_METAL_SLOW_DISPATCH_MODE=1 pytest -sv tests/didt/test_noc_integrity.py -k "galaxy and eth" \\
      --didt-workload-iterations 20 --determinism-check-interval 5
    # with the clock x VDD sweep:
    TT_METAL_SLOW_DISPATCH_MODE=1 pytest -sv tests/didt/test_noc_integrity.py -k "galaxy and all" \\
      --determinism-check-interval 5 --didt-vf-sweep --sweep-iterations 20 --enable-vdd-offsets

Settings (same names and defaults as the C++ test): NOC_TEST_ITERATIONS (pattern passes per pair, 1),
NOC_TEST_SLICE_SIZE (4096), NOC_TEST_PACKET_SIZE (2048), NOC_TEST_BURST (8), NOC_TEST_VC (2), NOC_TEST_TENSIX_ROWS
(0 = all), NOC_TEST_TIMEOUT_MS (2000). NOC_TEST_DRY_RUN=1 launches no kernels, to check that every pair then fails.
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.op_test_base import OpParameter, OpTestBase

KERNEL = "tests/tt_metal/tt_metal/deployment/kernels/noc_integrity_kernel.cpp"

# Mirrors tests/tt_metal/tt_metal/deployment/kernels/noc_integrity_types.hpp
RESULT_WORDS = 8  # status, mismatches, first offset, expected, actual, pattern, iteration, transfers
STATUS = {0: "not run", 1: "done", 2: "write timeout", 3: "read timeout", 0xBAD0BAD0: "kernel never ran"}
STATUS_DONE = 1
STATUS_STALE = 0xBAD0BAD0  # written over every result before a pass: a kernel that never starts cannot pass on old data
PATTERNS = ["WalkingOnes", "WalkingZeroes", "Checkerboard", "CheckerboardInv", "Address", "PRBS"]
DEST_DRAM_BANK = 0xFFFF

MAX_DESTS = 64  # destinations per kernel launch (Ethernet kernels are limited in runtime args)
ALIGN = 64
UNUSED_GALAXY_ETH_CHANNELS = (10, 11)  # unused by design on a Blackhole Galaxy: failures there are only warnings
MAX_LOGGED_FAILURES = 40

SLOW_DISPATCH = os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") == "1"


def env_int(name, default):
    return int(os.environ.get(name, default))


def align(x):
    return (x + ALIGN - 1) // ALIGN * ALIGN


def core_layout(base, max_dests, slice_size, num_slots):
    """Addresses on a source/destination core: progress, results (both NOCs), source pattern, readback, destination
    slots (one per source of the chip)."""
    progress = align(base)
    results = progress + ALIGN
    src = align(results + 2 * max_dests * RESULT_WORDS * 4)
    readback = src + slice_size
    dest = readback + slice_size
    return {
        "progress": progress,
        "results": results,
        "src": src,
        "readback": readback,
        "dest": dest,
        "end": dest + num_slots * slice_size,
    }


class Source:
    def __init__(self, kind, logical, virtual, active=False, channel=-1):
        self.kind = kind  # "eth" or "tensix"
        self.logical = logical
        self.virtual = virtual
        self.active = active
        self.channel = channel
        self.slot = None

    def __str__(self):
        if self.kind == "eth":
            return f"eth ch {self.channel} [{'active' if self.active else 'idle'}, logical {self.logical.x}-{self.logical.y}]"
        return f"tensix [logical {self.logical.x}-{self.logical.y}, virtual {self.virtual.x}-{self.virtual.y}]"


class Dest:
    def __init__(self, kind, x, y, label, channel=-1, logical=None):
        self.kind = kind  # "eth", "tensix" or "dram"
        self.x, self.y = x, y  # NOC coordinate, or (DEST_DRAM_BANK, bank) for DRAM
        self.label = label
        self.channel = channel
        self.logical = logical

    def __str__(self):
        return self.label


class NocIntegrityTest(OpTestBase):
    def __init__(self, mesh_device, sources, *args, **kwargs):
        super().__init__(mesh_device, *args, **kwargs)
        self.sources_kind = sources
        self.p = {
            "iterations": env_int("NOC_TEST_ITERATIONS", 1),
            "slice_size": env_int("NOC_TEST_SLICE_SIZE", 4096),
            "timeout_ms": env_int("NOC_TEST_TIMEOUT_MS", 2000),
            "packet_size": env_int("NOC_TEST_PACKET_SIZE", 2048),
            "burst": env_int("NOC_TEST_BURST", 8),
            "vc": env_int("NOC_TEST_VC", 2),
            "tensix_rows": env_int("NOC_TEST_TENSIX_ROWS", 0),
        }
        assert self.p["slice_size"] % ALIGN == 0, "NOC_TEST_SLICE_SIZE must be a multiple of 64"
        assert self.p["packet_size"] % 16 == 0, "NOC_TEST_PACKET_SIZE must be a multiple of 16"
        self.blackhole_galaxy = ttnn.get_arch_name() == "blackhole" and ttnn.distributed.get_num_devices() == 32
        self._plan()

    # ---------------------------------------------------------------- plan
    def _coords(self):
        # Not list(MeshCoordinateRange(shape)): its Python iterator yields one coordinate object, mutated in place.
        shape = self.mesh_device.shape
        return [ttnn.MeshCoordinate(r, c) for r in range(shape[0]) for c in range(shape[1])]

    def _plan(self):
        md = self.mesh_device
        grid = md.compute_with_storage_grid_size()
        rows = min(self.p["tensix_rows"], grid.y) if self.p["tensix_rows"] else grid.y
        self.grid = grid
        self.eth_risc = ttnn._ttnn.device.ExperimentalGetEthernetRiscInfo()
        eth_start, eth_end = ttnn._ttnn.device.ExperimentalGetEthernetL1Range()

        tensix_dests = []
        for y in range(rows):
            for x in range(grid.x):
                v = md.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                tensix_dests.append(
                    Dest("tensix", v.x, v.y, f"tensix [logical {x}-{y}, virtual {v.x}-{v.y}]", logical=(x, y))
                )
        num_banks = md.dram_grid_size().x
        dram_dests = [Dest("dram", DEST_DRAM_BANK, b, f"dram bank {b}") for b in range(num_banks)]

        corners = sorted(
            {(0, 0), (grid.x - 1, 0), (0, grid.y - 1), (grid.x - 1, grid.y - 1)}, key=lambda c: (c[1], c[0])
        )

        # Per chip: Ethernet cores differ (active/idle, which channels exist), Tensix and DRAM do not.
        self.chips = {}
        for coord in self._coords():
            device_id = md.get_device_id(coord)
            eth = ttnn._ttnn.device.ExperimentalGetEthernetCores(md, coord)
            sources = []
            if self.sources_kind in ("eth", "all"):
                sources += [Source("eth", e["logical"], e["virtual"], e["active"], e["channel"]) for e in eth]
            if self.sources_kind in ("tensix", "all"):
                for x, y in corners:
                    sources.append(
                        Source("tensix", ttnn.CoreCoord(x, y), md.worker_core_from_logical_core(ttnn.CoreCoord(x, y)))
                    )
            for slot, s in enumerate(sources):
                s.slot = slot
            eth_dests = [
                Dest(
                    "eth",
                    e["virtual"].x,
                    e["virtual"].y,
                    f"eth ch {e['channel']} [{'active' if e['active'] else 'idle'}]",
                    channel=e["channel"],
                )
                for e in eth
            ]
            self.chips[device_id] = {"coord": coord, "sources": sources, "eth_dests": eth_dests}

        self.num_slots = max(len(c["sources"]) for c in self.chips.values())
        self.max_dests = len(tensix_dests) + len(dram_dests) + max(len(c["eth_dests"]) for c in self.chips.values())
        self.num_batches = math.ceil(self.max_dests / MAX_DESTS)
        slice_size = self.p["slice_size"]

        # Ethernet L1 has no allocator: use the free range every Ethernet core leaves to kernels, as the C++ test does.
        self.eth_layout = core_layout(eth_start, self.max_dests, slice_size, self.num_slots)
        assert self.eth_layout["end"] <= eth_end, (
            f"Ethernet L1 layout needs {self.eth_layout['end'] - eth_start} B, only {eth_end - eth_start} B free; "
            "lower NOC_TEST_SLICE_SIZE"
        )
        # Tensix L1 and DRAM come from the allocator, through tensors that stay alive for the whole test.
        tensix_bytes = core_layout(0, self.max_dests, slice_size, self.num_slots)["end"]
        grid_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        shard_rows = align(tensix_bytes) // ALIGN
        l1_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid_set, [shard_rows, ALIGN // 4], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.tensix_tensor = self._zeros([grid.x * grid.y * shard_rows, ALIGN // 4], l1_config)
        self.tensix_layout = core_layout(
            self.tensix_tensor.buffer_address(), self.max_dests, slice_size, self.num_slots
        )
        dram_rows = num_banks * self.num_slots * slice_size // ALIGN
        self.dram_tensor = self._zeros([dram_rows, ALIGN // 4], ttnn.DRAM_MEMORY_CONFIG)
        dram_base = self.dram_tensor.buffer_address()

        # Destination lists per chip and source: everything but the source itself.
        for chip in self.chips.values():
            for s in chip["sources"]:
                dests = []
                for d in tensix_dests:
                    if not (s.kind == "tensix" and (d.x, d.y) == (s.virtual.x, s.virtual.y)):
                        dests.append((d, self.tensix_layout["dest"] + s.slot * slice_size))
                for d in chip["eth_dests"]:
                    if not (s.kind == "eth" and (d.x, d.y) == (s.virtual.x, s.virtual.y)):
                        dests.append((d, self.eth_layout["dest"] + s.slot * slice_size))
                for d in dram_dests:
                    dests.append((d, dram_base + s.slot * slice_size))
                s.dests = dests

        n_eth = sum(1 for s in next(iter(self.chips.values()))["sources"] if s.kind == "eth")
        logger.info(
            f"NOC integrity: sources {self.sources_kind} ({n_eth} Ethernet + "
            f"{len(next(iter(self.chips.values()))['sources']) - n_eth} Tensix per chip), up to {self.max_dests} "
            f"destinations each ({len(tensix_dests)} Tensix, {len(dram_dests)} DRAM banks, Ethernet), "
            f"{self.num_batches} batch(es) x 2 NOCs per pass, slice {slice_size} B, "
            f"{self.p['iterations']} pattern pass(es) per pair"
        )
        self.programs = [[self._mesh_program(noc, batch) for batch in range(self.num_batches)] for noc in (0, 1)]

    def _zeros(self, shape, memory_config):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=memory_config,
            mesh_mapper=self.from_torch_mesh_mapper,
        )

    def _compile_args(self, layout, noc, batch):
        results = layout["results"] + (noc * self.max_dests + batch * MAX_DESTS) * RESULT_WORDS * 4
        p = self.p
        return [
            layout["progress"],
            results,
            layout["src"],
            layout["readback"],
            p["slice_size"],
            p["iterations"],
            p["timeout_ms"],
            p["packet_size"],
            p["burst"],
            p["vc"],
        ]

    def _eth_config(self, active, noc):
        """Processor and NOC for an Ethernet kernel, as the C++ test picks them (set_arch_specific_eth_config)."""
        info = self.eth_risc
        processor = noc if (active or info["idle_eth_riscs"] > 1) else 0
        kernel_noc = noc
        if active and not info["active_fw_cooperative"]:
            kernel_noc = processor if info["active_eth_riscs"] > 1 else 1
        return ttnn.EthernetConfigDescriptor(
            eth_mode=ttnn.EthernetKernelMode.SENDER if active else ttnn.EthernetKernelMode.IDLE,
            noc=ttnn.NOC.RISCV_1_default if kernel_noc else ttnn.NOC.RISCV_0_default,
            processor=ttnn.DataMovementProcessor.RISCV_1 if processor else ttnn.DataMovementProcessor.RISCV_0,
        )

    def _mesh_program(self, noc, batch):
        mesh_program = ttnn.MeshProgramDescriptor()
        for device_id, chip in self.chips.items():
            groups = {}  # one kernel per (core kind, active) group
            for s in chip["sources"]:
                groups.setdefault((s.kind, s.active), []).append(s)
            kernels = []
            for (kind, active), sources in groups.items():
                rt_args = ttnn.RuntimeArgs()
                for s in sources:
                    batch_dests = s.dests[batch * MAX_DESTS : (batch + 1) * MAX_DESTS]
                    seed = (0x9E3779B9 ^ (device_id << 24) ^ (s.slot << 8) ^ noc) & 0xFFFFFFFF
                    args = [seed, len(batch_dests)]
                    for d, addr in batch_dests:
                        args += [d.x, d.y, addr]
                    rt_args[s.logical.x][s.logical.y] = args
                layout = self.eth_layout if kind == "eth" else self.tensix_layout
                config = (
                    self._eth_config(active, noc)
                    if kind == "eth"
                    else ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1 if noc else ttnn.DataMovementProcessor.RISCV_0,
                        noc=ttnn.NOC.RISCV_1_default if noc else ttnn.NOC.RISCV_0_default,
                    )
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=KERNEL,
                        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(s.logical, s.logical) for s in sources]),
                        compile_time_args=self._compile_args(layout, noc, batch),
                        runtime_args=rt_args,
                        config=config,
                    )
                )
            program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
            mesh_program[ttnn.MeshCoordinateRange(chip["coord"], chip["coord"])] = program
        return mesh_program

    # ---------------------------------------------------------------- OpTestBase hooks
    def _clear_results(self):
        """Overwrite every source's result records, so stale ones from an earlier pass cannot look like a pass."""
        for device_idx in self.device_ids:
            ttnn.device.synchronize_device(self.get_device(device_idx))
        stale = [STATUS_STALE] * (2 * self.max_dests * RESULT_WORDS)
        for chip in self.chips.values():
            for s in chip["sources"]:
                layout = self.eth_layout if s.kind == "eth" else self.tensix_layout
                ttnn._ttnn.device.ExperimentalWriteL1(
                    self.mesh_device, chip["coord"], s.logical, s.kind == "eth", layout["results"], stale
                )

    def run_device_operation(self):
        self._clear_results()
        if os.environ.get("NOC_TEST_DRY_RUN") == "1":  # self-check: launch nothing, every pair must fail
            return None
        for noc in (0, 1):
            for batch in range(self.num_batches):
                ttnn.generic_op([self.tensix_tensor, self.dram_tensor], self.programs[noc][batch])
        return None

    def release_output(self, out):
        pass

    def read_outputs(self, out, pool):
        """Per device: every source's result records for both NOCs, concatenated (int64)."""

        def read_chip(device_id):
            chip = self.chips[device_id]
            words = []
            for s in chip["sources"]:
                layout = self.eth_layout if s.kind == "eth" else self.tensix_layout
                words += ttnn._ttnn.device.ExperimentalReadL1(
                    self.mesh_device,
                    chip["coord"],
                    s.logical,
                    s.kind == "eth",
                    layout["results"],
                    2 * self.max_dests * RESULT_WORDS * 4,
                )
            return torch.tensor(words, dtype=torch.int64)

        return list(pool.map(read_chip, self.device_ids))

    def output_failures(self, outputs):
        bad = []
        logged = 0
        for output_id, device_id in enumerate(self.device_ids):
            words = outputs[output_id].tolist()
            per_source = 2 * self.max_dests * RESULT_WORDS
            failed = False
            for i, s in enumerate(self.chips[device_id]["sources"]):
                for noc in (0, 1):
                    for d_idx, (d, addr) in enumerate(s.dests):
                        o = i * per_source + (noc * self.max_dests + d_idx) * RESULT_WORDS
                        status, mismatches, first, expected, actual, pattern, iteration, transfers = words[o : o + 8]
                        if status == STATUS_DONE and mismatches == 0:
                            continue
                        unused = self.blackhole_galaxy and (
                            (s.kind == "eth" and s.channel in UNUSED_GALAXY_ETH_CHANNELS)
                            or (d.kind == "eth" and d.channel in UNUSED_GALAXY_ETH_CHANNELS)
                        )
                        if not unused:
                            failed = True
                        if logged < MAX_LOGGED_FAILURES:
                            logged += 1
                            what = (
                                f"{mismatches} words in {transfers} transfers, first at +{first:#x}, expected "
                                f"{expected:#010x}, got {actual:#010x} ({PATTERNS[pattern % len(PATTERNS)]}, "
                                f"iteration {iteration})"
                                if status == STATUS_DONE
                                else STATUS.get(status, f"status {status}")
                            )
                            (logger.warning if unused else logger.error)(
                                f"NOC {'mismatch' if status == STATUS_DONE else 'failure'}: chip {device_id} src {s} "
                                f"-> dst {d} @{addr:#x} noc {noc}: {what}" + (" [unused channel]" if unused else "")
                            )
            bad.append(failed)
        return bad


@pytest.mark.parametrize("sources", ["eth", "tensix", "all"])
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((8, 4), id="galaxy"), pytest.param(1, id="1chips")],
    indirect=["mesh_device"],
)
def test_noc_integrity(mesh_device, sources, didt_workload_iterations, determinism_check_interval):
    if sources in ("eth", "all") and not SLOW_DISPATCH:
        pytest.skip("Ethernet sources need TT_METAL_SLOW_DISPATCH_MODE=1 (kernels on active Ethernet cores)")
    test = NocIntegrityTest(
        mesh_device,
        sources,
        OpParameter([32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),  # unused activation
        [],
        out_mem_config=None,
        out_dtype=None,
        program_config=None,
        compute_config=None,
        loop_count=didt_workload_iterations,
        # The NOC results are judged at determinism checks, so without an interval check every pass.
        determinism_check_enabled=True,
        determinism_check_interval=determinism_check_interval if determinism_check_interval > 0 else 1,
    )
    test.run_op_test()
