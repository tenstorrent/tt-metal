# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Transient DRAM SiLU LUT, final multiply, and head packing component.

CPU preparation is the default; device execution requires --run-device.
This is a component diagnostic, not decoder acceptance or a production path.
"""

import argparse
import ast
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_gated_norm_tail_fusion_probe as norm
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_silu_local_lookup_probe import READER as BASE_READER

E = norm.E
NORM_SHA = "47d50c370b8038bc1b10f24f86c5e58f61222e21b6cc7a1bb22d6318055961bd"
LOCAL_READER_SHA = "10d1206e9686bee0dd7515a27be1b305aa67e3224d0ac8fa46c53cf3cf328e24"


def reader_source(strategy):
    assert strategy in ("stream", "blocks")
    source = BASE_READER
    source = source.replace(
        " const auto* table=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2));",
        " auto table_args=TensorAccessorArgs<z_args.next_compile_time_args_offset(),z_args.next_common_runtime_args_offset()>();\n"
        " const auto table_accessor=TensorAccessor(table_args,get_arg_val<uint32_t>(2));",
    )
    source = source.replace(
        " Noc noc; DataflowBuffer weights(0), gates(1), codes(2);",
        " Noc noc; DataflowBuffer weights(0), gates(1), codes(2), lookup_scratch(3);",
    )
    if strategy == "stream":
        setup = """
 lookup_scratch.reserve_back(65);
 const uint32_t table_bias=(64-(get_write_ptr(3)&63))&63;
 for(uint32_t offset=0;offset<262144;offset+=8192) {
  noc.async_read(table_accessor,lookup_scratch,8192,
    {.page_id=0,.offset_bytes=offset},{.offset_bytes=table_bias+offset});
 }
 noc.async_read_barrier();
 const auto* table=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3)+table_bias);
"""
    else:
        setup = """
 lookup_scratch.reserve_back(1);
 const uint32_t table_bias=(64-(get_write_ptr(3)&63))&63;
 const auto* blocks=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3)+table_bias);
"""
        old = "  for(uint32_t j=0;j<32;++j) g[(j/16)*256+j%16]=table[c[(j/16)*32+j%16]];"
        new = """  for(uint32_t j=0;j<32;++j) {
   const uint32_t code=c[(j/16)*32+j%16];
   noc.async_read(table_accessor,lookup_scratch,64,
      {.page_id=0,.offset_bytes=(code>>4)*64},{.offset_bytes=table_bias+j*64});
  }
  noc.async_read_barrier();
  for(uint32_t j=0;j<32;++j) {
   const uint32_t code=c[(j/16)*32+j%16];
   g[(j/16)*256+j%16]=blocks[j*16+(code&15)];
  }"""
        assert source.count(old) == 1
        source = source.replace(old, new)
    source = source.replace(" codes.reserve_back(1);", setup + " codes.reserve_back(1);")
    assert source.count("table_accessor=TensorAccessor") == 1
    return source


COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu.h"
#endif
void kernel_main() {
 constexpr bool debug=get_compile_time_arg_val(0)!=0;
 CircularBuffer weighted(0),gate(1),output(16),debug_gate(17);
 compute_kernel_hw_startup(0,16);copy_init(0);
 for(uint32_t tile=0;tile<4;++tile) {
  weighted.wait_front(1);gate.wait_front(1);
  tile_regs_acquire_math_clear();copy_tile(0,0,0);copy_tile(1,0,1);
#ifdef TRISC_MATH
  sfpu::_init_sfpu_config_reg();
  addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);
  math::reset_counters(p_setrwc::SET_ABD_F);_llk_math_eltwise_sfpu_start_(0);
  for(uint32_t lane=0;lane<32;++lane) {
   {sfpi::vFloat w=sfpi::dst_reg[0],g=sfpi::dst_reg[32];sfpi::dst_reg[0]=w*g;}
   {sfpi::vFloat product=sfpi::dst_reg[0];sfpi::vUInt bits=sfpi::as<sfpi::vUInt>(product);
    sfpi::vUInt tie=(bits>>16)&1;bits=bits+0x7fff;bits=bits+tie;
    sfpi::dst_reg[0]=sfpi::as<sfpi::vFloat>(bits&0xffff0000);}
   sfpi::dst_reg++;
  }
  _llk_math_eltwise_sfpu_done_();
#endif
  tile_regs_commit();tile_regs_wait();
  pack_reconfig_data_format(16);output.reserve_back(1);pack_tile(0,16);output.push_back(1);
  if constexpr(debug) {
   pack_reconfig_data_format(17);debug_gate.reserve_back(1);pack_tile(1,17);debug_gate.push_back(1);
  }
  weighted.pop_front(1);gate.pop_front(1);tile_regs_release_math_clear();
 }
}
"""


class LocalSiluMultiply:
    """32 workers, exact borrowed DRAM LUT, no runtime tensor conversions.

    Weighted FP32 carriers [1,32,128] have already been rounded to BF16.
    Z is BF16 TILE [1,1,4096]. Output directly matches native head concat:
    BF16 TILE [1,1,4096], padded [1,32,4096], 32 width shards [32,128].
    """

    def __init__(self, t, mesh, table=None, strategy="stream", debug=False):
        self.t, self.mesh, self.debug = t, mesh, debug
        self.strategy = strategy
        self.reader = reader_source(strategy)
        grid = mesh.compute_with_storage_grid_size()
        assert grid.x * grid.y >= 32
        self.coordinates = [(i % grid.x, i // grid.x) for i in range(32)]
        self.cores = t.num_cores_to_corerangeset(32, grid, row_wise=True)
        self.output_memory = t.MemoryConfig(
            t.TensorMemoryLayout.WIDTH_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 128), t.ShardOrientation.ROW_MAJOR),
        )
        # Borrow the already-needed prefill table when available. No persistent
        # L1 allocation; an isolated component can upload one DRAM copy in setup.
        self.table = table
        if self.table is None:
            self.table = t.from_torch(
                make_table(),
                dtype=t.float32,
                layout=t.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=t.DRAM_MEMORY_CONFIG,
            )
        assert tuple(self.table.shape) == (1, 65536)
        assert self.table.dtype == t.float32 and self.table.layout == t.ROW_MAJOR_LAYOUT
        assert self.table.memory_config() == t.DRAM_MEMORY_CONFIG
        self.config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
        )
        modes = [t.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = t.UnpackToDestMode.UnpackToDestFp32
        self.config.unpack_to_dest_mode = modes

    def __call__(self, weighted, z):
        t = self.t
        assert tuple(weighted.shape) == (1, 32, 128) and weighted.dtype == t.float32
        assert tuple(z.shape) == (1, 1, 4096) and z.dtype == t.bfloat16
        assert weighted.layout == z.layout == t.TILE_LAYOUT
        # TensorAccessor handles the input placement; no implicit layout copy.
        output = t.empty(
            (1, 1, 4096), dtype=t.bfloat16, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.output_memory
        )
        bindings = [(16, output)]
        gate = None
        if self.debug:
            gate = t.empty(
                (1, 1, 4096), dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.output_memory
            )
            bindings.append((17, gate))
        runtime = t.RuntimeArgs()
        for head, (x, y) in enumerate(self.coordinates):
            runtime[x][y] = [weighted.buffer_address(), z.buffer_address(), self.table.buffer_address(), head]
        accessor_args = t.TensorAccessorArgs(weighted).get_compile_time_args()
        accessor_args += t.TensorAccessorArgs(z).get_compile_time_args()
        accessor_args += t.TensorAccessorArgs(self.table).get_compile_time_args()

        def kernel(source, config, compile_args, runtime_args):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=self.cores,
                config=config,
                compile_time_args=compile_args,
                runtime_args=runtime_args,
            )

        cbs = [t.cb_descriptor_from_sharded_tensor(i, value) for i, value in bindings]
        for index, dtype, size, pages in (
            (0, t.float32, 4096, 2),
            (1, t.float32, 4096, 2),
            (2, t.bfloat16, 2048, 1),
            (3, t.float32, 4096, 65 if self.strategy == "stream" else 1),
        ):
            cbs.append(
                t.CBDescriptor(
                    total_size=pages * size,
                    core_ranges=self.cores,
                    format_descriptors=[t.CBFormatDescriptor(index, dtype, size)],
                )
            )
        program = t.ProgramDescriptor(
            kernels=[
                kernel(self.reader, t.ReaderConfigDescriptor(), accessor_args, runtime),
                kernel(COMPUTE, self.config, [int(self.debug)], []),
            ],
            cbs=cbs,
        )
        t.generic_op([weighted, z, self.table, *[v for _, v in bindings]], program)
        return output, gate


def make_table():
    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return torch.nn.functional.silu(codes.float()).reshape(1, 65536)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    integer = torch.int32 if a.dtype == torch.float32 else torch.int16
    different = a.contiguous().view(integer) != b.contiguous().view(integer)
    both_nan = a.isnan() & b.isnan()
    return dict(
        bit_differences=int(different.sum()),
        differences_excluding_nan_payload=int((different & ~both_nan).sum()),
        nan_class_differences=int((a.isnan() != b.isnan()).sum()),
        nan_payload_only=int((different & both_nan).sum()),
    )


def prepare(args):
    assert verify_source(norm.__file__, NORM_SHA)
    assert verify_source(Path(__file__).with_name("v6_silu_local_lookup_probe.py"), LOCAL_READER_SHA)
    gamma, originals = norm.fixtures(args.checkpoint, args.random_cases)
    metadata_path = E / "v6_gated_norm_tail_fusion_device.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["device_closed"] and metadata["exact_native_tail"]
    assert metadata["script_sha256"] == NORM_SHA
    tensors_path = Path(metadata["device_tensors"])
    assert norm.sha(tensors_path) == metadata["device_tensors_sha256"]
    saved = torch.load(tensors_path, weights_only=False)
    cases = []
    table = make_table()
    raw = torch.arange(65536, dtype=torch.int64)
    # Exact block-address/lane mapping covers every possible BF16 payload.
    block_lookup = table.reshape(4096, 16)[raw >> 4, raw & 15]
    assert torch.equal(block_lookup.view(torch.int32), table.flatten().view(torch.int32))
    lookup_alignment_checks = 0
    for residue in (0, 16, 32, 48):
        bias = (64 - residue) & 63
        for offset in range(0, 262144, 8192):
            assert (residue + bias + offset) % 64 == offset % 64
            assert bias + offset + 8192 <= 65 * 4096
            lookup_alignment_checks += 1
        for j in range(32):
            assert (residue + bias + j * 64) % 64 == 0
            assert bias + j * 64 + 64 <= 4096
            lookup_alignment_checks += 1
    for fixture in originals:
        gate = torch.nn.functional.silu(fixture["z"].float())
        _, cpu = norm.cpu_tail(fixture["core"], fixture["core"].float().square().sum(-1, keepdim=True), gamma, gate)
        actual = fixture["name"].startswith("position")
        weighted = saved[fixture["name"]]["native"]["weighted_fp32_carrier"] if actual else cpu["weighted_fp32_carrier"]
        assert torch.equal(weighted, weighted.bfloat16().float())
        cases.append(
            dict(name=fixture["name"], weighted=weighted, z=fixture["z"].reshape(1, 1, 4096), exhaustive=False)
        )
    for block in range(16):
        raw = torch.arange(block * 4096, (block + 1) * 4096, dtype=torch.int32).to(torch.int16)
        # Exactly representable nontrivial BF16 carriers exercise both signs and
        # product rounding while the debug gate separately checks every LUT bit.
        weights = torch.tensor([0.75, -1.25, 1.0, 1.5, -0.5, 0.0, -2.5, 0.125]).repeat(512)
        cases.append(
            dict(
                name=f"all_codes_{block:02d}",
                weighted=weights.reshape(1, 32, 128),
                z=raw.view(torch.bfloat16).reshape(1, 1, 4096),
                exhaustive=True,
            )
        )
    # Independently check physical face addresses for every head and column.
    coordinates = torch.arange(4096).reshape(32, 128)
    tiles = torch.stack(
        [coordinates[:, j : j + 32].reshape(2, 16, 2, 16).permute(0, 2, 1, 3).reshape(-1) for j in range(0, 128, 32)]
    )
    reads = torch.empty_like(coordinates)
    for head in range(32):
        for tile in range(4):
            for col in range(32):
                reads[head, tile * 32 + col] = tiles[
                    tile, (head // 16) * 512 + (head % 16) * 16 + (col // 16) * 256 + col % 16
                ]
    assert torch.equal(reads, coordinates)
    # Model both half-row transactions for every possible L1-aligned CB base.
    # DRAM reads require the source and destination low six bits to match;
    # the original contiguous 32-byte scratch slots violated this for half1.
    alignment_checked = 0
    for base_mod64 in (0, 16, 32, 48):
        bias = (64 - base_mod64) & 63
        for head in range(32):
            for tile in range(4):
                scratch = torch.full((1024,), -1, dtype=torch.int64)
                for half in range(2):
                    source_offset = half * 512
                    destination_offset = bias + half * 64
                    assert source_offset % 64 == (base_mod64 + destination_offset) % 64
                    assert destination_offset + 32 <= 2048
                    start = destination_offset // 2
                    scratch[start : start + 16] = torch.arange(
                        head * 128 + tile * 32 + half * 16, head * 128 + tile * 32 + half * 16 + 16
                    )
                    alignment_checked += 1
                recovered = torch.tensor([scratch[bias // 2 + (j // 16) * 32 + j % 16] for j in range(32)])
                assert torch.equal(recovered, torch.arange(head * 128 + tile * 32, head * 128 + tile * 32 + 32))
    for fixture in cases:
        codes = fixture["z"].view(torch.int16).long() & 65535
        direct = torch.nn.functional.silu(fixture["z"].float())
        assert compare(table.flatten()[codes], direct)["differences_excluding_nan_payload"] == 0
    tree = ast.parse(Path(__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LocalSiluMultiply")
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    calls = [ast.unparse(n.func) for n in ast.walk(call) if isinstance(n, ast.Call)]
    prohibited = (".from_torch", ".to_torch", ".to_layout", ".to_memory_config", ".gather")
    assert not any(name.startswith("torch.") or name.endswith(prohibited) for name in calls)
    result = dict(
        classification="Transient DRAM SiLU lookup, FP32 product, BF16 rounding, and head packing component diagnostic",
        acceptance_eligible=False,
        oracle_assisted=False,
        status="cpu_setup_pass",
        device_closed=False,
        source_sha256=norm.sha(__file__),
        norm_helper_sha256=raw_sha256(norm.__file__),
        norm_helper_historical_sha256=NORM_SHA,
        source_dependencies={str(Path(norm.__file__)): raw_sha256(norm.__file__)},
        source_dependencies_historical_sha256={str(Path(norm.__file__)): NORM_SHA},
        reader_base_sha256=raw_sha256(Path(__file__).with_name("v6_silu_local_lookup_probe.py")),
        reader_base_historical_sha256=LOCAL_READER_SHA,
        exact_dram_block_code_mappings=65536,
        lookup_transaction_alignment_checks=lookup_alignment_checks,
        weighted_fixture_source=str(tensors_path),
        weighted_fixture_source_sha256=norm.sha(tensors_path),
        table_sha256=digest(table),
        table_entries=65536,
        strategy=args.strategy,
        persistent_l1_table_bytes_per_worker=0,
        temporary_table_cb_bytes_per_worker=266240 if args.strategy == "stream" else 4096,
        table_workers=32,
        dram_table_bytes=262144,
        per_decode_lut_read_bytes=8388608 if args.strategy == "stream" else 262144,
        borrowed_existing_prefill_table=True,
        face_mapping_checked_elements=4096,
        dram_halfrow_alignment_checks=alignment_checked,
        dram_read_alignment_bytes=64,
        dram_alignment_source="tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379; tt_metal/hw/inc/internal/debug/sanitize.h:746",
        exhaustive_table_codes=65536,
        runtime_tensor_conversions=[],
        output_contract="BF16 TILE logical [1,1,4096], padded [1,32,4096], 32 row-major workers with WIDTH_SHARDED [32,128]",
        timing_scope="weighted BF16-rounded FP32 carrier + original BF16 Z -> merged BF16; excludes normalization and setup; includes all runtime input handling",
        limitations="No speedup or full decoder capacity claim. No persistent L1 table; stream mode uses temporary 260 KiB/core scratch; blocks mode 4 KiB/core. Exhaustive unusual/nonfinite inputs are separately recorded; exact per-token decoder gates remain unchanged.",
        cases=[],
        timing=[],
    )
    return table, cases, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--strategy", choices=("stream", "blocks"), default="stream")
    parser.add_argument(
        "--time", action="store_true", help="Trace this component after correctness; watcher/profiler must be off"
    )
    parser.add_argument("--random-cases", type=int, default=8)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / norm.REVISION,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.replays >= 100 and args.samples > 0 and args.warmup > 0
    assert not args.output.exists(), "Preserve evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    table, cases, result = prepare(args)
    for name, content in (
        ("source.py", Path(__file__).read_text()),
        ("reader.cpp", reader_source(args.strategy)),
        ("compute.cpp", COMPUTE),
    ):
        snapshot = Path(str(args.output) + f".{name}.snapshot")
        snapshot.write_text(content)
        result[name + "_snapshot"] = dict(path=str(snapshot), sha256=norm.sha(snapshot))
    result["command"] = sys.argv

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_DRAM_SILU_SETUP_PASS", len(cases), "cases; all 65536 LUT codes and 4096 face positions")
        return
    if args.time:
        for variable in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
            assert os.environ.get(variable) in (None, "", "0"), "Separate timing from watcher/profiler"
    from ttnn.tools import trace_allocation_tracker

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    record(args.output)
    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=16777216
        )
        lut = t.from_torch(
            table, dtype=t.float32, layout=t.ROW_MAJOR_LAYOUT, device=mesh, memory_config=t.DRAM_MEMORY_CONFIG
        )
        candidate = LocalSiluMultiply(t, mesh, table=lut, strategy=args.strategy, debug=True)
        head_shard = t.create_sharded_memory_config(
            (32, 128),
            t.CoreGrid(y=1, x=1),
            t.ShardStrategy.HEIGHT,
            t.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        def baseline(weighted, z):
            gate = norm.device_lookup(t, z, lut)
            gated = t.multiply(
                weighted, gate, dtype=t.bfloat16, fast_and_approximate_mode=False, memory_config=t.L1_MEMORY_CONFIG
            )
            sharded = t.to_memory_config(t.reshape(gated, (1, 1, 32, 128)), head_shard)
            return t.reshape(t.experimental.nlp_concat_heads_decode(sharded, num_heads=32), (1, 1, 4096), (1, 32, 4096))

        def upload(fixture):
            return (
                t.from_torch(
                    fixture["weighted"],
                    dtype=t.float32,
                    layout=t.TILE_LAYOUT,
                    device=mesh,
                    memory_config=t.L1_MEMORY_CONFIG,
                ),
                t.from_torch(
                    fixture["z"],
                    dtype=t.bfloat16,
                    layout=t.TILE_LAYOUT,
                    device=mesh,
                    memory_config=t.DRAM_MEMORY_CONFIG,
                ),
            )

        diagnostic_tensors = {}
        for fixture in cases:
            weighted, z = upload(fixture)
            with device_only():
                expected = baseline(weighted, z)
                actual, gate = candidate(weighted, z)
            host = t.to_torch(actual).clone()
            gate_host = t.to_torch(gate).clone()
            gate_cpu = table.flatten()[fixture["z"].view(torch.int16).long() & 65535]
            check = compare(t.to_torch(expected), host)
            if fixture["name"] in ("position128", "position3271", "all_codes_00"):
                diagnostic_tensors[fixture["name"]] = dict(
                    weighted=fixture["weighted"],
                    z=fixture["z"],
                    gate_cpu=gate_cpu,
                    gate_device=gate_host,
                    output_device=host,
                    output_native=t.to_torch(expected).clone(),
                )
            row = dict(
                name=fixture["name"],
                exhaustive=fixture["exhaustive"],
                output_vs_native=check,
                gate_vs_table=compare(gate_cpu, gate_host),
            )
            result["cases"].append(row)
            save()
            print(fixture["name"], row, flush=True)
            del weighted, z, expected, actual, gate
        tensor_path = Path(str(args.output) + ".diagnostic_tensors.pt")
        torch.save(diagnostic_tensors, tensor_path)
        result["diagnostic_tensors"] = dict(path=str(tensor_path), sha256=norm.sha(tensor_path))
        ordinary = [row for row in result["cases"] if not row["exhaustive"]]
        result["ordinary_cases_bitexact"] = all(
            row["output_vs_native"]["bit_differences"] == 0 and row["gate_vs_table"]["bit_differences"] == 0
            for row in ordinary
        )
        result["all_code_cases_semantically_exact"] = all(
            row[key]["differences_excluding_nan_payload"] == 0
            for row in result["cases"]
            for key in ("output_vs_native", "gate_vs_table")
        )
        assert result["ordinary_cases_bitexact"], "Actual/random finite cases must remain bitexact"
        if args.time:
            candidate.debug = False  # Only changes the compile-time debug branch; no debug output allocated.
            fixture = next(f for f in cases if f["name"] == "position3271")
            weighted, z = upload(fixture)
            reference = t.to_torch(baseline(weighted, z)).clone()
            for name, operation in (
                ("lookup_multiply_concat", baseline),
                ("local_lookup_multiply_pack", lambda w, z: candidate(w, z)[0]),
            ):
                for _ in range(args.warmup):
                    with device_only():
                        output = operation(weighted, z)
                    t.synchronize_device(mesh)
                    del output
                trace_id = None
                capturing = False
                try:
                    trace_id = t.begin_trace_capture(mesh, cq_id=0)
                    capturing = True
                    with device_only():
                        output = operation(weighted, z)
                    t.end_trace_capture(mesh, trace_id, cq_id=0)
                    capturing = False
                    trace_allocation_tracker.acknowledge_corruptible(output)
                    for _ in range(args.warmup):
                        t.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    first = t.to_torch(output).clone()
                    assert compare(reference, first)["bit_differences"] == 0
                    samples = []
                    for _ in range(args.samples):
                        t.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            t.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                        t.synchronize_device(mesh)
                        samples.append((time.perf_counter() - start) * 1000 / args.replays)
                    last = t.to_torch(output).clone()
                    assert compare(first, last)["bit_differences"] == 0
                    result["timing"].append(
                        dict(
                            variant=name,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            replays_per_sample=args.replays,
                            first_vs_last=compare(first, last),
                            output_memory_config=str(output.memory_config()),
                        )
                    )
                    save()
                finally:
                    if trace_id is not None:
                        if capturing:
                            t.end_trace_capture(mesh, trace_id, cq_id=0)
                        t.synchronize_device(mesh)
                        t.release_trace(mesh, trace_id)
                    del output
        result["status"] = "completed_diagnostic"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


if __name__ == "__main__":
    main()
