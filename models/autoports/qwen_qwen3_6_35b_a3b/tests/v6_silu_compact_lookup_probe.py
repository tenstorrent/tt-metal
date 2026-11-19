# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact compact immutable SiLU LUT, final multiply, and head packing component.

CPU preparation is the default; device execution requires --run-device.
This is a component diagnostic, not decoder acceptance or a production path.
"""

import argparse
import ast
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_gated_norm_tail_fusion_probe as norm
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_silu_local_lookup_probe import COMPUTE as FULL_COMPUTE
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_silu_local_lookup_probe import READER as FULL_READER

E = norm.E
NORM_SHA = "47d50c370b8038bc1b10f24f86c5e58f61222e21b6cc7a1bb22d6318055961bd"

FULL_SOURCE_SHA = "10d1206e9686bee0dd7515a27be1b305aa67e3224d0ac8fa46c53cf3cf328e24"
POS_LO, POS_HI, NEG_LO, NEG_HI = 0x33C1, 0x4185, 0xB440, 0xC2B1
POS_COUNT, NEG_COUNT = 3525, 3698
COMPACT_ENTRIES = 7488
COMPACT_BYTES = COMPACT_ENTRIES * 4
LOOKUP_CPP = r"""
inline uint32_t silu_lookup_bits(uint32_t code, const volatile tt_l1_ptr uint32_t* table) {
 if(code<0x8000) {
  if(code>=0x33c1 && code<=0x4185) return table[code-0x33c1];
  if(code>0x4185 && code<0x7f80) return code<<16;
  if(code>=0x7f80) return table[7223+code-0x7f80];
 } else {
  if(code>=0xb440 && code<=0xc2b1) return table[3525+code-0xb440];
  if(code>0xc2b1 && code<0xff80) return 0x80000000;
  if(code>=0xff80) return table[7351+code-0xff80];
 }
 // Exactly representable FP32 x/2 for BF16 x, including FP32 subnormals.
 const uint32_t magnitude=(code&0x7fff)<<16, sign=(code&0x8000)<<16;
 return sign | (magnitude>=0x01000000 ? magnitude-0x00800000 : magnitude>>1);
}
"""
READER = FULL_READER.replace("void kernel_main() {", LOOKUP_CPP + "\nvoid kernel_main() {")
READER = READER.replace("table[c[(j/16)*32+j%16]]", "silu_lookup_bits(c[(j/16)*32+j%16],table)")


def compact_table(full):
    raw = torch.arange(65536, dtype=torch.int64)
    values = full.flatten().contiguous().view(torch.int32).long() & 0xFFFFFFFF
    magnitude, sign = (raw & 32767) << 16, (raw & 32768) << 16
    half = sign | torch.where(magnitude >= 0x01000000, magnitude - 0x00800000, magnitude >> 1)
    derived = []
    for start in (0, 32768):
        ids = raw[start : start + 32640]
        far = (ids << 16) if start == 0 else torch.full_like(ids, 0x80000000)
        lo = int(ids[values[ids] != half[ids]].min())
        hi = int(ids[values[ids] != far].max())
        derived.extend((lo, hi))
    assert derived == [POS_LO, POS_HI, NEG_LO, NEG_HI], derived
    parts = [values[POS_LO : POS_HI + 1], values[NEG_LO : NEG_HI + 1], values[0x7F80:0x8000], values[0xFF80:]]
    packed = torch.cat(parts)
    assert packed.numel() == 7479
    padded = torch.cat((packed, torch.zeros(COMPACT_ENTRIES - packed.numel(), dtype=torch.int64)))
    reconstructed = half.clone()
    reconstructed[POS_LO : POS_HI + 1] = padded[:POS_COUNT]
    reconstructed[POS_HI + 1 : 0x7F80] = raw[POS_HI + 1 : 0x7F80] << 16
    reconstructed[NEG_LO : NEG_HI + 1] = padded[POS_COUNT : POS_COUNT + NEG_COUNT]
    reconstructed[NEG_HI + 1 : 0xFF80] = 0x80000000
    reconstructed[0x7F80:0x8000] = padded[7223:7351]
    reconstructed[0xFF80:] = padded[7351:7479]
    assert torch.equal(reconstructed, values), int((reconstructed != values).sum())
    return padded.to(torch.int32).view(torch.float32).reshape(1, COMPACT_ENTRIES)


def compiled_mapping_proof(full, compact):
    source = (
        "#include <stdint.h>\n#define tt_l1_ptr\n"
        + LOOKUP_CPP
        + r"""
extern "C" void reconstruct(const uint32_t* table, uint32_t* output) {
 for(uint32_t code=0;code<65536;++code) output[code]=silu_lookup_bits(code,table);
}
"""
    )
    with tempfile.TemporaryDirectory(prefix="compact_silu_cpu_", dir=E) as directory:
        cpp, library = Path(directory) / "mapping.cpp", Path(directory) / "mapping.so"
        cpp.write_text(source)
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "-shared", "-fPIC", str(cpp), "-o", str(library)],
            check=True,
            capture_output=True,
        )
        dll = ctypes.CDLL(str(library))
        pointer = ctypes.POINTER(ctypes.c_uint32)
        dll.reconstruct.argtypes = [pointer, pointer]
        table = compact.contiguous().view(torch.int32).numpy().view(np.uint32).reshape(-1)
        output = np.empty(65536, dtype=np.uint32)
        dll.reconstruct(table.ctypes.data_as(pointer), output.ctypes.data_as(pointer))
        expected = full.contiguous().view(torch.int32).numpy().view(np.uint32).reshape(-1)
        assert np.array_equal(output, expected), int(np.count_nonzero(output != expected))
    return dict(
        compiled_cpp_bit_differences=0,
        codes_checked=65536,
        cpp_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source=source,
    )


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
    """32 workers, compact immutable LUT, no runtime tensor conversions.

    Weighted FP32 carriers [1,32,128] have already been rounded to BF16.
    Z is BF16 TILE [1,1,4096]. Output directly matches native head concat:
    BF16 TILE [1,1,4096], padded [1,32,4096], 32 width shards [32,128].
    """

    def __init__(self, t, mesh, debug=False):
        self.t, self.mesh, self.debug = t, mesh, debug
        grid = mesh.compute_with_storage_grid_size()
        assert grid.x * grid.y >= 32
        self.coordinates = [(i % grid.x, i // grid.x) for i in range(32)]
        self.cores = t.num_cores_to_corerangeset(32, grid, row_wise=True)
        self.output_memory = t.MemoryConfig(
            t.TensorMemoryLayout.WIDTH_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 128), t.ShardOrientation.ROW_MAJOR),
        )
        table_memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (1, COMPACT_ENTRIES), t.ShardOrientation.ROW_MAJOR),
        )
        self.table = t.from_torch(
            compact_table(make_table()).expand(32, COMPACT_ENTRIES).contiguous(),
            dtype=t.float32,
            layout=t.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=table_memory,
        )
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
        for index, dtype, size, pages in ((0, t.float32, 4096, 2), (1, t.float32, 4096, 2), (2, t.bfloat16, 2048, 1)):
            cbs.append(
                t.CBDescriptor(
                    total_size=pages * size,
                    core_ranges=self.cores,
                    format_descriptors=[t.CBFormatDescriptor(index, dtype, size)],
                )
            )
        program = t.ProgramDescriptor(
            kernels=[
                kernel(READER, t.ReaderConfigDescriptor(), accessor_args, runtime),
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
    assert verify_source(Path(__file__).with_name("v6_silu_local_lookup_probe.py"), FULL_SOURCE_SHA)
    assert COMPUTE == FULL_COMPUTE
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
    compact = compact_table(table)
    proof = compiled_mapping_proof(table, compact)
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
        classification="Exact compact SiLU lookup, FP32 product, BF16 rounding, and head packing component diagnostic",
        acceptance_eligible=False,
        oracle_assisted=False,
        status="cpu_setup_pass",
        device_closed=False,
        source_sha256=norm.sha(__file__),
        norm_helper_sha256=raw_sha256(norm.__file__),
        norm_helper_historical_sha256=NORM_SHA,
        source_dependencies={str(Path(norm.__file__)): raw_sha256(norm.__file__)},
        source_dependencies_historical_sha256={str(Path(norm.__file__)): NORM_SHA},
        weighted_fixture_source=str(tensors_path),
        weighted_fixture_source_sha256=norm.sha(tensors_path),
        table_sha256=digest(table),
        table_entries=65536,
        table_bytes_per_worker=COMPACT_BYTES,
        compact_entries=COMPACT_ENTRIES,
        compact_used_entries=7479,
        compact_table_sha256=digest(compact),
        compiled_mapping={k: v for k, v in proof.items() if k != "source"},
        derived_ranges=dict(positive=[POS_LO, POS_HI], negative=[NEG_LO, NEG_HI]),
        table_workers=32,
        table_total_bytes=COMPACT_BYTES * 32,
        face_mapping_checked_elements=4096,
        dram_halfrow_alignment_checks=alignment_checked,
        dram_read_alignment_bytes=64,
        dram_alignment_source="tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379; tt_metal/hw/inc/internal/debug/sanitize.h:746",
        exhaustive_table_codes=65536,
        runtime_tensor_conversions=[],
        output_contract="BF16 TILE logical [1,1,4096], padded [1,32,4096], 32 row-major workers with WIDTH_SHARDED [32,128]",
        timing_scope="weighted BF16-rounded FP32 carrier + original BF16 Z -> merged BF16; excludes normalization and setup; includes all runtime input handling",
        limitations="No speedup or full decoder capacity claim. Exact compact table requires 29,952 bytes/core persistent L1; all original FP32 table bits reconstructed. Exhaustive unusual/nonfinite inputs are separately recorded; exact per-token decoder gates remain unchanged.",
        cases=[],
        timing=[],
    )
    return table, cases, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-device", action="store_true")
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
    for name, content in (("source.py", Path(__file__).read_text()), ("reader.cpp", READER), ("compute.cpp", COMPUTE)):
        snapshot = Path(str(args.output) + f".{name}.snapshot")
        snapshot.write_text(content)
        result[name + "_snapshot"] = dict(path=str(snapshot), sha256=norm.sha(snapshot))
    result["command"] = sys.argv

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_COMPACT_SILU_SETUP_PASS", len(cases), "cases; all 65536 LUT codes and 4096 face positions")
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
        candidate = LocalSiluMultiply(t, mesh, debug=True)
        lut = t.from_torch(
            table, dtype=t.float32, layout=t.ROW_MAJOR_LAYOUT, device=mesh, memory_config=t.DRAM_MEMORY_CONFIG
        )
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
