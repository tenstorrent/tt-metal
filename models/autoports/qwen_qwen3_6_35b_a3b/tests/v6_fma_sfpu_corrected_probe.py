# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dynamic direct-unpack inline SFPU FMA diagnostic. Hardware is opt-in."""

import argparse
import ctypes
import hashlib
import json
import struct
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_fma_sfpu_primitive_probe import fixtures as midpoint_fixtures

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
F32 = lambda x: ctypes.c_float(x).value
BITS = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
FLOAT = lambda x: struct.unpack("<f", struct.pack("<I", x & 0xFFFFFFFF))[0]


class Graph:
    def __init__(self):
        self.nodes = [("input",), ("input",), ("input",), ("constant", 4097.0), ("constant", 0.0), ("constant", 0.5)]

    def op(self, kind, *args):
        self.nodes.append((kind, *args))
        return len(self.nodes) - 1

    def sum(self, a, b):
        s = self.op("+", a, b)
        z = self.op("-", s, a)
        e = self.op("+", self.op("-", a, self.op("-", s, z)), self.op("-", b, z))
        return s, e

    def grow(self, expansion, b):
        out = []
        q = b
        for term in expansion:
            q, error = self.sum(q, term)
            out.append(error)
        return out + [q]

    def split(self, x):
        t = self.op("*", 3, x)
        hi = self.op("-", t, self.op("-", t, x))
        return hi, self.op("-", x, hi)

    def build(self):
        ah, al = self.split(0)
        bh, bl = self.split(1)
        p = self.op("*", 0, 1)
        e1 = self.op("-", p, self.op("*", ah, bh))
        e2 = self.op("-", e1, self.op("*", al, bh))
        e3 = self.op("-", e2, self.op("*", ah, bl))
        pe = self.op("-", self.op("*", al, bl), e3)
        s, se = self.sum(p, 2)
        r = self.op("+", s, self.op("+", se, pe))
        exact = self.grow([pe, p], 2)
        rem = self.grow(exact, self.op("-", 4, r))
        direction = self.op("sign", *rem)
        neighbor = self.op("neighbor", r, direction)
        half = self.op("*", self.op("-", neighbor, r), 5)
        comparison = self.grow(rem, self.op("-", 4, half))
        cmp_sign = self.op("sign", *comparison)
        self.result = self.op("select", r, neighbor, direction, cmp_sign)
        return self

    @staticmethod
    def dst(slot):
        return (0, 32, 64)[slot] if slot < 3 else slot + 93

    def evaluate(self, a, b, c):
        values = [a, b, c]
        for node in self.nodes[3:]:
            kind, *ids = node
            if kind == "constant":
                value = ids[0]
            elif kind in ("+", "-", "*"):
                x, y = (values[i] for i in ids)
                value = F32(x + y if kind == "+" else x - y if kind == "-" else x * y)
            elif kind == "sign":
                value = 0.0
                for i in ids:
                    if values[i] != 0:
                        value = 1.0 if values[i] > 0 else -1.0
            elif kind == "neighbor":
                r, direction = (values[i] for i in ids)
                if direction == 0:
                    value = r
                elif r == 0:
                    value = FLOAT(1 if direction > 0 else 0x80000001)
                else:
                    step = 1 if (r > 0) == (direction > 0) else -1
                    value = FLOAT(BITS(r) + step)
            elif kind == "select":
                r, neighbor, direction, cmp_sign = (values[i] for i in ids)
                cmp = direction * cmp_sign
                value = neighbor if cmp > 0 or (cmp == 0 and direction != 0 and BITS(r) & 1) else r
                if direction == 0 and r == 0:
                    negative = (a == 0 or b == 0) and c == 0 and ((BITS(a) ^ BITS(b)) >> 31) and (BITS(c) >> 31)
                    value = FLOAT(0x80000000) if negative else 0.0
            else:
                raise AssertionError(kind)
            values.append(value)
        return values[self.result]

    def cpp(self):
        lines = []
        for slot, node in enumerate(self.nodes):
            if slot < 3:
                continue
            kind, *ids = node
            dest = self.dst(slot)
            reads = [self.dst(i) for i in ids] if kind != "constant" else []
            if kind == "constant":
                lines.append(f"sfpi::dst_reg[{dest}] = sfpi::vFloat({float(ids[0]).hex()}f);")
            elif kind in ("+", "-", "*"):
                lines.append(
                    f"{{ sfpi::vFloat a=sfpi::dst_reg[{reads[0]}]; sfpi::vFloat b=sfpi::dst_reg[{reads[1]}]; sfpi::dst_reg[{dest}]=a {kind} b; }}"
                )
            elif kind == "sign":
                body = ["sfpi::vFloat direction=0.0f;"]
                for address in reads:
                    body.append(
                        f"{{ sfpi::vFloat x=sfpi::dst_reg[{address}]; sfpi::vInt bits=sfpi::as<sfpi::vInt>(x); v_if((bits & 0x7fffffff) != 0) {{ direction=1.0f; }} v_endif; v_if((bits & 0x7fffffff) != 0 && bits < 0) {{ direction=-1.0f; }} v_endif; }}"
                    )
                body.append(f"sfpi::dst_reg[{dest}]=direction;")
                lines.append("{ " + " ".join(body) + " }")
            elif kind == "neighbor":
                lines.append(
                    f"""{{
 sfpi::vFloat r=sfpi::dst_reg[{reads[0]}]; sfpi::vFloat direction=sfpi::dst_reg[{reads[1]}];
 sfpi::vInt bits=sfpi::as<sfpi::vInt>(r); sfpi::vInt step=0;
 v_if(direction>0.0f && bits>=0) {{ step=1; }} v_endif;
 v_if(direction>0.0f && bits<0) {{ step=-1; }} v_endif;
 v_if(direction<0.0f && bits>=0) {{ step=-1; }} v_endif;
 v_if(direction<0.0f && bits<0) {{ step=1; }} v_endif;
 sfpi::vInt nextbits=bits+step;
 v_if((bits & 0x7fffffff)==0 && direction>0.0f) {{ nextbits=1; }} v_endif;
 v_if((bits & 0x7fffffff)==0 && direction<0.0f) {{ nextbits=sfpi::vInt(-2147483647); }} v_endif;
 sfpi::dst_reg[{dest}]=sfpi::as<sfpi::vFloat>(nextbits);
}}"""
                )
            elif kind == "select":
                lines.append(
                    f"""{{
 sfpi::vFloat r=sfpi::dst_reg[{reads[0]}]; sfpi::vFloat neighbor=sfpi::dst_reg[{reads[1]}];
 sfpi::vFloat direction=sfpi::dst_reg[{reads[2]}]; sfpi::vFloat cmp_sign=sfpi::dst_reg[{reads[3]}];
 sfpi::vFloat cmp=direction*cmp_sign; sfpi::vFloat selected=r; sfpi::vInt rb=sfpi::as<sfpi::vInt>(r);
 v_if(cmp>0.0f) {{ selected=neighbor; }} v_endif;
 v_if(cmp==0.0f && direction!=0.0f && (rb & 1)!=0) {{ selected=neighbor; }} v_endif;
 v_if(direction==0.0f && (rb & 0x7fffffff)==0) {{ selected=0.0f; }} v_endif;
 sfpi::dst_reg[{dest}]=selected;
}}
{{
 sfpi::vFloat selected=sfpi::dst_reg[{dest}]; sfpi::vFloat direction=sfpi::dst_reg[{reads[2]}];
 sfpi::vInt rb=sfpi::as<sfpi::vInt>(selected);
 sfpi::vFloat a=sfpi::dst_reg[0]; sfpi::vInt ab=sfpi::as<sfpi::vInt>(a);
 sfpi::vFloat b=sfpi::dst_reg[32]; sfpi::vInt bb=sfpi::as<sfpi::vInt>(b);
 sfpi::vFloat c=sfpi::dst_reg[64]; sfpi::vInt cb=sfpi::as<sfpi::vInt>(c);
 // Flatten the signed-zero predicate: SFPI 7.95 rvtt_cc ICEs on the mixed
 // six-way conjunction/disjunction above a predicated assignment.
 sfpi::vInt eligible=1; sfpi::vInt zero_operand=0;
 v_if((ab & 0x7fffffff)==0) {{ zero_operand=1; }} v_endif;
 v_if((bb & 0x7fffffff)==0) {{ zero_operand=1; }} v_endif;
 v_if(direction!=0.0f) {{ eligible=0; }} v_endif;
 v_if((rb & 0x7fffffff)!=0) {{ eligible=0; }} v_endif;
 v_if(zero_operand==0) {{ eligible=0; }} v_endif;
 v_if(cb!=sfpi::vInt(-2147483647-1)) {{ eligible=0; }} v_endif;
 v_if((ab ^ bb)>=0) {{ eligible=0; }} v_endif;
 v_if(eligible!=0) {{ selected=sfpi::as<sfpi::vFloat>(sfpi::vInt(-2147483647-1)); }} v_endif;
 sfpi::dst_reg[{dest}]=selected;
}}"""
                )
        return "\n".join(lines)


def compute_source(graph):
    # Eight FP32 DST tiles: operands 0/1/2, scratch tiles 3..6, result tile 7.
    assert graph.dst(len(graph.nodes) - 1) + 31 < 224
    return r"""
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
 CircularBuffer input(0); CircularBuffer output(16);
 compute_kernel_hw_startup(0,16); copy_init(0);
 constexpr uint32_t count=get_compile_time_arg_val(0);
 input.reserve_back(3*count); input.push_back(3*count);
 for(uint32_t t=0;t<count;++t) {
  tile_regs_acquire_math_clear(); input.wait_front(3);
  copy_tile(0,0,0); copy_tile(0,1,1); copy_tile(0,2,2);
#ifdef TRISC_MATH
  sfpu::_init_sfpu_config_reg();
  addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);
  math::reset_counters(p_setrwc::SET_ABD_F);
  _llk_math_eltwise_sfpu_start_(0);
  for(uint32_t lane=0;lane<32;++lane) {
   GRAPH
   { sfpi::vFloat result=sfpi::dst_reg[RESULT_SLOT]; sfpi::dst_reg[224]=result; }
   sfpi::dst_reg++;
  }
  _llk_math_eltwise_sfpu_done_();
#endif
  tile_regs_commit(); tile_regs_wait(); output.reserve_back(4);
  pack_tile(7,16); pack_tile(0,16); pack_tile(1,16); pack_tile(2,16);
  output.push_back(4); input.pop_front(3); tile_regs_release_math_clear();
 }
}
""".replace(
        "GRAPH", graph.cpp()
    ).replace(
        "RESULT_SLOT", str(graph.dst(graph.result))
    )


def make_fixtures(random_tiles):
    base = midpoint_fixtures()
    base.extend(
        [
            dict(name="zero_positive", a=0.0, b=-1.0, c=0.0),
            dict(name="zero_negative", a=-0.0, b=1.0, c=-0.0),
            dict(name="exact_cancel", a=1.0, b=-1.0, c=1.0),
        ]
    )
    triples = torch.tensor([[x[k] for k in ("a", "b", "c")] for x in base], dtype=torch.float32)
    index = torch.arange(1024) % len(base)
    first = triples[index]
    labels = [base[i]["name"] for i in index.tolist()]
    categories = [
        "subnormal_control" if "subnormal" in x else "zero_control" if "zero" in x else "actual_or_bounded"
        for x in labels
    ]
    generator = torch.Generator().manual_seed(603614)
    all_values = [first]
    for tile in range(random_tiles):
        parts = []
        for lo, hi in ((-28, 29), (-28, 29), (-58, 59)):
            bits = torch.randint(0, 1 << 23, (1024,), generator=generator, dtype=torch.int64)
            bits |= (torch.randint(lo, hi, (1024,), generator=generator, dtype=torch.int64) + 127) << 23
            bits |= torch.randint(0, 2, (1024,), generator=generator, dtype=torch.int64) << 31
            parts.append(bits.to(torch.int32).view(torch.float32))
        values = torch.stack(parts, dim=1)
        values[::3, 2] = -(values[::3, 0] * values[::3, 1])
        all_values.append(values)
        labels.extend([f"random_tile{tile}"] * 1024)
        categories.extend(["bounded_random"] * 1024)
    # Tensor ordering is a tile, b tile, c tile for each batch element.
    values = torch.stack(all_values)
    packed = values.transpose(1, 2).reshape(-1, 1, 32, 32).contiguous()
    libm = ctypes.CDLL("libm.so.6")
    libm.fmaf.argtypes = [ctypes.c_float] * 3
    libm.fmaf.restype = ctypes.c_float
    golden = torch.tensor(
        [libm.fmaf(*triple) for triple in values.reshape(-1, 3).tolist()], dtype=torch.float32
    ).reshape(-1, 32, 32)
    return packed, golden, values, labels, categories


def summarize(actual, golden, categories):
    actual = actual.flatten()
    golden = golden.flatten()
    result = {}
    for category in sorted(set(categories)):
        selected = torch.tensor([x == category for x in categories])
        mismatch = actual.view(torch.int32) != golden.view(torch.int32)
        numeric = actual != golden
        indices = torch.nonzero(selected & mismatch).flatten()
        result[category] = dict(
            count=int(selected.sum()),
            bit_different=int((selected & mismatch).sum()),
            numeric_different=int((selected & numeric).sum()),
            examples=[
                dict(
                    index=i,
                    expected=float(golden[i]),
                    actual=float(actual[i]),
                    expected_bits=BITS(float(golden[i])),
                    actual_bits=BITS(float(actual[i])),
                )
                for i in indices[:12].tolist()
            ],
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--random-tiles", type=int, default=8)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    torch.set_num_threads(2)
    graph = Graph().build()
    source = compute_source(graph)
    packed, golden, values, labels, categories = make_fixtures(args.random_tiles)
    cpu = torch.tensor([graph.evaluate(*x) for x in values.reshape(-1, 3).tolist()], dtype=torch.float32).reshape_as(
        golden
    )
    result = dict(
        status="source_emitted_unexecuted",
        classification="bounded FMA component, not decoder acceptance",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        compute_sha256=hashlib.sha256(source.encode()).hexdigest(),
        node_count=len(graph.nodes),
        last_scratch_slot=graph.dst(len(graph.nodes) - 1),
        random_seed=603614,
        tiles=len(values),
        cpu_graph=summarize(cpu, golden, categories),
        limits="Bounded normal finite arithmetic only; subnormal controls are reported separately. Dynamic input unpack and all lanes are checked. No runtime host arithmetic. Performance and whole-chunk integration remain unqualified.",
    )
    args.output.with_suffix(".compute.cpp.snapshot").write_text(source)
    fixture_path = args.output.with_suffix(".fixtures.pt")
    torch.save(dict(packed=packed, golden=golden, values=values, labels=labels, categories=categories), fixture_path)
    result["fixtures_sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    assert all(v["bit_different"] == 0 for k, v in result["cpu_graph"].items() if k != "subnormal_control")
    if not args.run_device:
        return
    import ttnn

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        core = ttnn.CoreCoord(0, 0)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

        def memory(tile_count):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, (tile_count * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
            )

        inp = ttnn.from_torch(
            packed, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=memory(len(values) * 3)
        )
        out = ttnn.empty(
            (len(values) * 4, 1, 32, 32),
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory(len(values) * 4),
        )
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=source,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    compile_time_args=[len(values)],
                    runtime_args=[],
                    config=config,
                )
            ],
            semaphores=[],
            cbs=[ttnn.cb_descriptor_from_sharded_tensor(0, inp), ttnn.cb_descriptor_from_sharded_tensor(16, out)],
        )
        ttnn.generic_op([inp, out], program)
        host = ttnn.to_torch(out).reshape(len(values), 4, 32, 32)
        result["device"] = summarize(host[:, 0], golden, categories)
        result["input_roundtrip_bits_different"] = [
            int(
                (
                    host[:, i + 1].contiguous().view(torch.int32)
                    != values[:, :, i].reshape(-1, 32, 32).contiguous().view(torch.int32)
                ).sum()
            )
            for i in range(3)
        ]
        result["bounded_pass"] = all(
            v["bit_different"] == 0 for k, v in result["device"].items() if k != "subnormal_control"
        ) and not any(result["input_roundtrip_bits_different"])
        result["status"] = "component_pass" if result["bounded_pass"] else "component_fail"
        torch.save(dict(actual=host, golden=golden), args.output.with_suffix(".device.pt"))
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


if __name__ == "__main__":
    main()
