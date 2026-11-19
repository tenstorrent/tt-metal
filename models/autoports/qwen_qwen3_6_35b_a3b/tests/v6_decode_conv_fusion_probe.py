# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode-only exact convolution/SiLU/carry component; hardware opt-in only.

Native MUL, ADDCMUL, TYPECAST and SiLU kernel APIs are reused, including FP32
pack/unpack between tap accumulations. No IEEE-FMA equivalence is assumed.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
BASE_SHA = "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
constexpr uint32_t face(uint32_t c) {return (c/16)*256+c%16;}
void kernel_main() {
 constexpr uint32_t so=get_compile_time_arg_val(0),w0o=get_compile_time_arg_val(1);
 constexpr uint32_t w1o=get_compile_time_arg_val(2),w2o=get_compile_time_arg_val(3),w3o=get_compile_time_arg_val(4);
 const auto q=TensorAccessor(TensorAccessorArgs<5>(),get_arg_val<uint32_t>(0));
 const auto state=TensorAccessor(TensorAccessorArgs<so>(),get_arg_val<uint32_t>(1));
 const auto w0=TensorAccessor(TensorAccessorArgs<w0o>(),get_arg_val<uint32_t>(2));
 const auto w1=TensorAccessor(TensorAccessorArgs<w1o>(),get_arg_val<uint32_t>(3));
 const auto w2=TensorAccessor(TensorAccessorArgs<w2o>(),get_arg_val<uint32_t>(4));
 const auto w3=TensorAccessor(TensorAccessorArgs<w3o>(),get_arg_val<uint32_t>(5));
 const uint32_t start=get_arg_val<uint32_t>(6),count=get_arg_val<uint32_t>(7);
 DataflowBuffer input(0),weight(1),scratch(3);Noc noc;
 scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(3)&63))&63;
 auto* raw=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(3)+bias);
 auto* wide=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3)+bias);
 input.reserve_back(2);weight.reserve_back(2);
 auto* iz=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
 auto* wz=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
 for(uint32_t n=0;n<2048;++n){iz[n]=0;wz[n]=0;}
 for(uint32_t tile=start;tile<start+count;++tile) {
  // DRAM reads retain matching64-byte offsets; source half-row blocks include padding.
  noc.async_read(q,scratch,64,{.page_id=tile,.offset_bytes=0},{.offset_bytes=bias});
  noc.async_read(q,scratch,64,{.page_id=tile,.offset_bytes=512},{.offset_bytes=bias+64});
  for(uint32_t row=0;row<3;++row)
   noc.async_read(state,scratch,64,{.page_id=row,.offset_bytes=tile*64},{.offset_bytes=bias+128+row*64});
  noc.async_read(w0,scratch,64,{.page_id=tile,.offset_bytes=0},{.offset_bytes=bias+384});
  noc.async_read(w0,scratch,64,{.page_id=tile,.offset_bytes=1024},{.offset_bytes=bias+448});
  noc.async_read(w1,scratch,64,{.page_id=tile,.offset_bytes=0},{.offset_bytes=bias+512});
  noc.async_read(w1,scratch,64,{.page_id=tile,.offset_bytes=1024},{.offset_bytes=bias+576});
  noc.async_read(w2,scratch,64,{.page_id=tile,.offset_bytes=0},{.offset_bytes=bias+640});
  noc.async_read(w2,scratch,64,{.page_id=tile,.offset_bytes=1024},{.offset_bytes=bias+704});
  noc.async_read(w3,scratch,64,{.page_id=tile,.offset_bytes=0},{.offset_bytes=bias+768});
  noc.async_read(w3,scratch,64,{.page_id=tile,.offset_bytes=1024},{.offset_bytes=bias+832});
  noc.async_read_barrier();
  for(uint32_t c=0;c<32;++c) raw[160+c]=raw[c<16?c:32+c-16];
  for(uint32_t j=0;j<4;++j) {
   input.reserve_back(1);weight.reserve_back(1);
   auto* in=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
   auto* wt=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
   for(uint32_t c=0;c<32;++c) {
    const uint32_t code=raw[(j<3?64+j*32:160)+c];
    // Current folded RM->FP32 TILE mapping: all finite exponent-zero codes map to+0.
    in[face(c)]=(code&0x7f80u)?code<<16:0u;
    wt[face(c)]=wide[96+j*32+c];
   }
   input.push_back(1);weight.push_back(1);
  }
  // All source history words were fetched before any in-place carry writes.
  noc.async_write(scratch,state,64,{.offset_bytes=bias+192},{.page_id=0,.offset_bytes=tile*64});
  noc.async_write(scratch,state,64,{.offset_bytes=bias+256},{.page_id=1,.offset_bytes=tile*64});
  noc.async_write(scratch,state,64,{.offset_bytes=bias+320},{.page_id=2,.offset_bytes=tile*64});
  noc.async_write_barrier();
 }
}
"""

COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
 constexpr bool debug=get_compile_time_arg_val(0)!=0;
 const uint32_t count=get_arg_val<uint32_t>(0);
 CircularBuffer input(0),weight(1),feedback(24),output(16),partials(17);
 compute_kernel_hw_startup(0,24);copy_init(0);
 for(uint32_t tile=0;tile<count;++tile) {
  for(uint32_t j=0;j<4;++j) {
   tile_regs_acquire_math_clear();input.wait_front(1);weight.wait_front(1);
   copy_tile(0,0,1);copy_tile(1,0,2);
   if(j==0){mul_binary_tile_init();mul_binary_tile<true>(1,2,0);}
   else {
    feedback.wait_front(1);copy_tile(24,0,0);feedback.pop_front(1);
    addcmul_tile_init();addcmul_tile<DataFormat::Float32,true>(0,1,2,0,0x3f800000);
   }
   input.pop_front(1);weight.pop_front(1);tile_regs_commit();tile_regs_wait();
   pack_reconfig_data_format(24);feedback.reserve_back(1);pack_tile(0,24);feedback.push_back(1);
   if constexpr(debug){partials.reserve_back(1);pack_tile(0,17);partials.push_back(1);}
   tile_regs_release_math_clear();
  }
  // Preserve final accumulator's native FP32 pack/unpack before unary chain.
  tile_regs_acquire_math_clear();feedback.wait_front(1);copy_tile(24,0,0);feedback.pop_front(1);
  typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
  typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b),true>(0);
  typecast_tile<static_cast<uint32_t>(DataFormat::Float16_b),static_cast<uint32_t>(DataFormat::Float32),true>(0);
  silu_tile_init();silu_tile<true>(0);
  typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b),true>(0);
  tile_regs_commit();tile_regs_wait();pack_reconfig_data_format(16);
  output.reserve_back(1);pack_tile(0,16);output.push_back(1);tile_regs_release_math_clear();
 }
}
"""
WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main(){
 constexpr bool debug=get_compile_time_arg_val(0)!=0;
 constexpr uint32_t po=get_compile_time_arg_val(1);
 const auto output=TensorAccessor(TensorAccessorArgs<2>(),get_arg_val<uint32_t>(0));
 const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
 DataflowBuffer out(16),partials(17);Noc noc;
 for(uint32_t tile=start;tile<start+count;++tile){
  if constexpr(debug){
   const auto p=TensorAccessor(TensorAccessorArgs<po>(),get_arg_val<uint32_t>(1));
   for(uint32_t j=0;j<4;++j){partials.wait_front(1);noc.async_write(partials,p,4096,{}, {.page_id=j*256+tile});noc.async_write_barrier();partials.pop_front(1);}
  }
  out.wait_front(1);noc.async_write(out,output,2048,{}, {.page_id=tile});noc.async_write_barrier();out.pop_front(1);
 }
}
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DecodeConv:
    def __init__(self, t, mesh, taps, max_cores=64):
        self.t, self.mesh, self.taps, self.max_cores = t, mesh, taps, max_cores

    def __call__(self, qkv, state, debug=False):
        t = self.t
        assert tuple(qkv.shape) == (1, 1, 8192) and qkv.dtype == t.bfloat16 and qkv.layout == t.TILE_LAYOUT
        assert tuple(state.shape) == (1, 3, 8192) and state.dtype == t.bfloat16 and state.layout == t.ROW_MAJOR_LAYOUT
        assert state.memory_config() == t.DRAM_MEMORY_CONFIG
        assert qkv.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        assert all(
            tuple(w.shape) == (1, 1, 8192) and w.dtype == t.float32 and w.layout == t.TILE_LAYOUT for w in self.taps
        )
        grid = self.mesh.compute_with_storage_grid_size()
        active = min(256, grid.x * grid.y, self.max_cores)
        partitions = []
        start = 0
        for h in range(active):
            count = 256 // active + (h < 256 % active)
            partitions.append((h % grid.x, h // grid.x, start, count))
            start += count
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partitions])
        out = t.empty(
            (1, 1, 8192), device=self.mesh, dtype=t.bfloat16, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        partial = (
            t.empty(
                (4, 1, 1, 8192),
                device=self.mesh,
                dtype=t.float32,
                layout=t.TILE_LAYOUT,
                memory_config=t.DRAM_MEMORY_CONFIG,
            )
            if debug
            else None
        )
        rd, cr, wr = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
        for x, y, start, count in partitions:
            rd[x][y] = [
                qkv.buffer_address(),
                state.buffer_address(),
                *[w.buffer_address() for w in self.taps],
                start,
                count,
            ]
            cr[x][y] = [count]
            wr[x][y] = [out.buffer_address(), partial.buffer_address() if debug else 0, start, count]
        accessors = [list(t.TensorAccessorArgs(x).get_compile_time_args()) for x in (qkv, state, *self.taps)]
        offsets = []
        cursor = 5
        for a in accessors:
            offsets.append(cursor)
            cursor += len(a)
        rdargs = offsets[1:] + sum(accessors, [])
        oa = list(t.TensorAccessorArgs(out).get_compile_time_args())
        pa = list(t.TensorAccessorArgs(partial).get_compile_time_args()) if debug else []
        cfg = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 24):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes

        def kernel(source, config, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=config,
                compile_time_args=args,
                runtime_args=runtime,
            )

        cb_specs = [
            (0, 2, 4096, t.float32),
            (1, 2, 4096, t.float32),
            (3, 1, 4096, t.float32),
            (24, 2, 4096, t.float32),
            (16, 1, 2048, t.bfloat16),
        ]
        if debug:
            cb_specs.append((17, 1, 4096, t.float32))
        program = t.ProgramDescriptor(
            kernels=[
                kernel(READER, t.ReaderConfigDescriptor(), rdargs, rd),
                kernel(COMPUTE, cfg, [int(debug)], cr),
                kernel(WRITER, t.WriterConfigDescriptor(), [int(debug), 2 + len(oa)] + oa + pa, wr),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=pages * size,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(i, dtype, size)],
                )
                for i, pages, size, dtype in cb_specs
            ],
        )
        t.generic_op([qkv, state, *self.taps, out] + ([partial] if debug else []), program)
        return out, partial


def native_function(t):
    assert sha(BASE) == BASE_SHA
    tree = ast.parse(BASE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_linear_attention")
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "history" for x in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(method.body[start:], start)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "ttnn.slice"
        and any(k.arg == "output_tensor" for k in n.value.keywords)
    )
    original = copy.deepcopy(method.body[start : end + 1])
    block = copy.deepcopy(original)

    class Observe(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(x, ast.Name) and x.id == "accum" for x in node.targets):
                return [node, ast.parse("stages.append(accum)").body[0]]
            return node

    changed = Observe().visit(ast.Module(block, type_ignores=[]))
    fn = ast.parse("def convolution(self,qkv,state):\n s=1\n stages=[]\n return conv,stages\n").body[0]
    fn.body = fn.body[:2] + changed.body + fn.body[-1:]
    module = ast.fix_missing_locations(ast.Module([fn], type_ignores=[]))
    scope = {"ttnn": t}
    exec(compile(module, str(BASE) + ":conv", "exec"), scope)
    return scope["convolution"], hashlib.sha256(ast.dump(ast.Module(original, type_ignores=[])).encode()).hexdigest()


def compare(a, b):
    import torch

    assert a.shape == b.shape and a.dtype == b.dtype
    dtype = torch.int32 if a.dtype == torch.float32 else torch.int16
    unequal = a.contiguous().view(dtype) != b.contiguous().view(dtype)
    finite = torch.isfinite(a) & torch.isfinite(b)
    return dict(
        bits_different=int(unequal.sum()),
        elements=a.numel(),
        max_finite_abs=float((a[finite].float() - b[finite].float()).abs().max()) if bool(finite.any()) else 0.0,
        first_indices=unequal.nonzero()[:8].tolist(),
    )


def prepare(args):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(4)
    fixtures = []
    provenance = {}
    weights = {}
    captures = {
        3271: "v6_prefix_control_stream_probe.json.pos3271.pt",
        227: "v6_inline_precision_stream_probe.json.pos227.pt",
        738: "v6_inline_precision_stream_probe.json.pos738.pt",
        5799: "v6_inline_precision_stream_probe.json.pos5799.pt",
    }
    generator = torch.Generator().manual_seed(2026111304)
    for name in args.cases.split(","):
        if name.startswith("actual"):
            position = int(name.removeprefix("actual"))
            path = E / captures[position]
            capture = torch.load(path, weights_only=False, map_location="cpu")
            v = capture["tt_values"]
            fixture = dict(
                name=name,
                layer=capture["layer"],
                qkv=v["projection_qkv"].contiguous(),
                carry=v["conv_input_state"].contiguous(),
                recorded_output=v["conv_output"],
                recorded_carry=v["conv_state"],
            )
            provenance[str(path)] = sha(path)
        elif name == "random":
            fixture = dict(
                name=name,
                layer=args.layer,
                qkv=(torch.randn(1, 1, 8192, generator=generator) * 3).bfloat16(),
                carry=(torch.randn(1, 3, 8192, generator=generator) * 3).bfloat16(),
            )
        elif name == "zeros":
            fixture = dict(
                name=name,
                layer=args.layer,
                qkv=(torch.randn(1, 1, 8192, generator=generator) * 0).bfloat16(),
                carry=(torch.randn(1, 3, 8192, generator=generator) * 0).bfloat16(),
            )
        elif name == "finite":
            codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
            codes = codes[torch.isfinite(codes)]
            padded = torch.cat((codes, torch.zeros(256, dtype=torch.bfloat16))).reshape(2, 4, 8192)
            for i in range(2):
                fixtures.append(
                    dict(
                        name=f"finite{i}",
                        layer=args.layer,
                        qkv=padded[i : i + 1, 3:].contiguous(),
                        carry=padded[i : i + 1, :3].contiguous(),
                    )
                )
            continue
        elif name == "tiny":
            codes = (
                torch.tensor(
                    [
                        0,
                        0x8000,
                        1,
                        0x8001,
                        0x7F,
                        0x807F,
                        0x80,
                        0x8080,
                        0x81,
                        0x8081,
                        0x100,
                        0x8100,
                        0x5FF,
                        0x85FF,
                        0x600,
                        0x8600,
                    ],
                    dtype=torch.int32,
                )
                .to(torch.int16)
                .view(torch.bfloat16)
            )
            for shift in range(4):
                x = codes.roll(shift).repeat(2048).reshape(1, 4, 8192)
                fixtures.append(
                    dict(name=f"tiny{shift}", layer=args.layer, qkv=x[:, 3:].contiguous(), carry=x[:, :3].contiguous())
                )
            continue
        else:
            raise ValueError(name)
        fixtures.append(fixture)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    for layer in sorted({f["layer"] for f in fixtures}):
        key = f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"
        with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as opened:
            weights[layer] = opened.get_tensor(key)
        assert weights[layer].shape == (8192, 1, 4) and weights[layer].dtype == torch.bfloat16
    # This observed mapping is checked exhaustively against the prior folded-route capture.
    mapping_path = E / "v4_conv_widen_boundary_probe.conversion.pt"
    captured = torch.load(mapping_path, weights_only=False, map_location="cpu")
    raw = captured["input"].view(torch.int16).to(torch.int32) & 65535
    predicted = torch.where((raw & 0x7F80) == 0, 0, raw << 16)
    assert torch.equal(predicted, captured["candidate"].view(torch.int32))
    provenance[str(mapping_path)] = sha(mapping_path)
    _, native_sha = native_function(None)
    return (
        weights,
        fixtures,
        dict(
            source_sha256=sha(__file__),
            base_sha256=BASE_SHA,
            native_conv_ast_sha256=native_sha,
            fixture_sources=provenance,
            folded_finite_widen_mapping_bits_different=0,
            weight_sha256={
                layer: hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest() for layer, w in weights.items()
            },
            fixture_names=[f["name"] for f in fixtures],
            status="cpu_setup_pass",
            device_executed=args.run_device,
            device_closed=False,
            cases=[],
            timing=[],
            acceptance_eligible=False,
            oracle_assisted=False,
            classification="Same-operand native-vs-fused decode convolution diagnostic; no decoder accuracy claim",
            timing_scope="BF16 QKV TILE + BF16 three-row RM carry + existing four FP32 taps -> BF16 convolution TILE + mutated RM carry, including matching Q/K/V slice adapters; no input placement conversions excluded",
            persistent_l1_bytes=0,
            cb_bytes_per_core=30720,
            debug_cb_bytes_per_core=34816,
        ),
    )


def run(args):
    assert not args.output.exists()
    weights, fixtures, result = prepare(args)
    args.output.with_suffix(".source.py.snapshot").write_bytes(Path(__file__).read_bytes())
    for name, source in [("reader", READER), ("compute", COMPUTE), ("writer", WRITER)]:
        path = args.output.with_suffix(f".{name}.cpp.snapshot")
        path.write_text(source)
        result[name + "_sha256"] = sha(path)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("DECODE_CONV_CPU_SETUP_PASS", sha(__file__))
        return
    for variable in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(variable) in (None, "", "0"), variable
    import torch

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=16 * 1024 * 1024
        )

        def upload(x, dtype=t.bfloat16, layout=t.TILE_LAYOUT, memory=t.DRAM_MEMORY_CONFIG):
            return t.from_torch(x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=memory)

        native, _ = native_function(t)
        narrow = t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.float32.value), float(t.bfloat16.value))
        widen = t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.bfloat16.value), float(t.float32.value))
        adapters = {}
        for layer, weight in weights.items():
            taps = [upload(weight[:, 0, j].float().reshape(1, 1, 8192), t.float32) for j in range(4)]
            adapters[layer] = (
                SimpleNamespace(
                    taps_fp32=taps, conv_post_chain=[narrow, widen, t.UnaryWithParam(t.UnaryOpType.SILU), narrow]
                ),
                DecodeConv(t, mesh, taps, args.max_cores),
            )
        saved = {}
        for fixture in fixtures:
            name, layer = fixture["name"], fixture["layer"]
            obj, candidate = adapters[layer]
            qkv = upload(fixture["qkv"], memory=t.L1_MEMORY_CONFIG)
            oldcarry = upload(fixture["carry"], layout=t.ROW_MAJOR_LAYOUT)
            newcarry = upload(fixture["carry"], layout=t.ROW_MAJOR_LAYOUT)
            with device_only():
                old, oldpartials = native(obj, qkv, {"conv": oldcarry})
                new, newpartials = candidate(qkv, newcarry, debug=True)
                # Native extraction includes these slices; keep same adapter cost.
                q, k, v = new[:, :, :2048], new[:, :, 2048:4096], new[:, :, 4096:]
            t.synchronize_device(mesh)
            oldhost, newhost = t.to_torch(old), t.to_torch(new)
            oldstate, newstate = t.to_torch(oldcarry), t.to_torch(newcarry)
            partial = t.to_torch(newpartials)
            row = dict(
                name=name,
                layer=layer,
                output=compare(oldhost, newhost),
                carry=compare(oldstate, newstate),
                partial=[compare(t.to_torch(p), partial[j]) for j, p in enumerate(oldpartials)],
            )
            if "recorded_output" in fixture:
                row.update(
                    native_vs_recorded_output=compare(fixture["recorded_output"], oldhost),
                    native_vs_recorded_carry=compare(fixture["recorded_carry"], oldstate),
                )
            result["cases"].append(row)
            saved[name] = dict(
                old=oldhost,
                new=newhost,
                oldcarry=oldstate,
                newcarry=newstate,
                partial=partial,
                oldpartial=[t.to_torch(p) for p in oldpartials],
            )
            save()
            print(json.dumps(row), flush=True)
        capture = args.output.with_suffix(".device.pt")
        torch.save(saved, capture)
        result.update(device_tensors=str(capture), device_tensors_sha256=sha(capture))
        exact = all(
            r["output"]["bits_different"] == r["carry"]["bits_different"] == 0
            and all(p["bits_different"] == 0 for p in r["partial"])
            for r in result["cases"]
        )
        result["status"] = "component_exact" if exact else "component_differences"
        save()
        assert exact, result["cases"]
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--cases", default="actual3271")
    p.add_argument("--layer", type=int, default=34)
    p.add_argument("--max-cores", type=int, default=64)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
