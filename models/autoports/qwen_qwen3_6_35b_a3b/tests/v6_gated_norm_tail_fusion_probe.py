# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Post-sum gated-normalization fusion diagnostic. Hardware is opt-in.

Native explicit FP32 squares/sum and the proven complete BF16-domain SiLU
lookup are unchanged. This first candidate fuses only the post-sum arithmetic.
Input placement copies are runtime costs and must not be excluded from timing.
"""

import argparse
import ast
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_gated_norm_same_operand_probe import REVISION
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_state_probe import code_compare, sha
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_silu_lookup_probe import device_lookup

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
EPSILON = 1e-6


def kernel_source():
    return r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu.h"
sfpi_inline sfpi::vFloat round_bf16(sfpi::vFloat x) {
 sfpi::vUInt bits=sfpi::as<sfpi::vUInt>(x);
 sfpi::vUInt tie=(bits>>16)&1;
 bits=bits+0x7fff; bits=bits+tie; bits=bits&0xffff0000;
 return sfpi::as<sfpi::vFloat>(bits);
}
#endif
void kernel_main() {
 constexpr bool debug=get_compile_time_arg_val(0)!=0;
 CircularBuffer core(0), sums(1), gamma(2), gate(3), output(16);
 CircularBuffer norm_debug(17), weight_debug(18), inverse_debug(19);
 compute_kernel_hw_startup(0,16); copy_init(0);
 core.reserve_back(4); core.push_back(4);
 sums.reserve_back(1); sums.push_back(1);
 gamma.reserve_back(4); gamma.push_back(4);
 gate.reserve_back(4); gate.push_back(4);
 core.wait_front(4); sums.wait_front(1); gamma.wait_front(4); gate.wait_front(4);
 for(uint32_t tile=0;tile<4;++tile) {
  tile_regs_acquire_math_clear();
  copy_tile(0,tile,0); copy_tile(1,0,1); copy_tile(2,tile,2); copy_tile(3,tile,3);
#ifdef TRISC_MATH
  sfpu::_init_sfpu_config_reg();
  addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);
  math::reset_counters(p_setrwc::SET_ABD_F);
  _llk_math_eltwise_sfpu_start_(1);
  for(uint32_t lane=0;lane<32;++lane) {
   // Explicit DST store/load separates multiply and add, preserving rounding.
   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::dst_reg[0]=x*0x1p-7f; }
   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::dst_reg[0]=x+EPSILON_HEX; }
   sfpi::dst_reg++;
  }
  _llk_math_eltwise_sfpu_done_();
#endif
  rsqrt_tile_init(); rsqrt_tile<RsqrtMode::Default,true>(1);
  sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col<true>(0,1);
#ifdef TRISC_MATH
  math::reset_counters(p_setrwc::SET_ABD_F);
  _llk_math_eltwise_sfpu_start_(0);
  for(uint32_t lane=0;lane<32;++lane) {
   { sfpi::vFloat x=sfpi::dst_reg[0]; if constexpr(debug) sfpi::dst_reg[128]=x; sfpi::dst_reg[0]=round_bf16(x); }
   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::vFloat w=sfpi::dst_reg[64]; sfpi::dst_reg[0]=x*w; }
   { sfpi::vFloat x=sfpi::dst_reg[0]; x=round_bf16(x); sfpi::dst_reg[0]=x; if constexpr(debug) sfpi::dst_reg[160]=x; }
   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::vFloat g=sfpi::dst_reg[96]; sfpi::dst_reg[0]=x*g; }
   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::dst_reg[0]=round_bf16(x); }
   sfpi::dst_reg++;
  }
  _llk_math_eltwise_sfpu_done_();
#endif
  tile_regs_commit(); tile_regs_wait();
  pack_reconfig_data_format(16);
  output.reserve_back(1); pack_tile(0,16); output.push_back(1);
  if constexpr(debug) {
   pack_reconfig_data_format(17);
   norm_debug.reserve_back(1); pack_tile(4,17); norm_debug.push_back(1);
   weight_debug.reserve_back(1); pack_tile(5,18); weight_debug.push_back(1);
   inverse_debug.reserve_back(1); pack_tile(1,19); inverse_debug.push_back(1);
  }
  tile_regs_release_math_clear();
 }
 core.pop_front(4); sums.pop_front(1); gamma.pop_front(4); gate.pop_front(4);
}
""".replace(
        "EPSILON_HEX", float(EPSILON).hex() + "f"
    )


class GatedNormTail:
    """Device-only tail after setup, accepting the unchanged native sum and LUT.

    Shapes: core/gate [1,32,128], sums [1,32,1]. All carriers are FP32.
    Single-core sharding is diagnostic. Existing input placement is restored
    only by the caller if its next consumer requires it. No numerical fallback.
    """

    def __init__(self, t, mesh, gamma, debug=False):
        self.t, self.mesh, self.debug = t, mesh, debug
        self.cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(0, 0), t.CoreCoord(0, 0))])
        self.memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 128), t.ShardOrientation.ROW_MAJOR),
        )
        self.sum_memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 32), t.ShardOrientation.ROW_MAJOR),
        )
        expanded_gamma = gamma.float().reshape(1, 1, 128).expand(1, 32, 128).contiguous()
        self.gamma = t.from_torch(
            expanded_gamma, dtype=t.float32, layout=t.TILE_LAYOUT, device=mesh, memory_config=self.memory
        )
        self.config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2, 3):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        self.config.unpack_to_dest_mode = modes
        self.source = kernel_source()

    def __call__(self, core, sums, gate):
        t = self.t
        assert tuple(core.shape) == tuple(gate.shape) == (1, 32, 128)
        assert tuple(sums.shape) == (1, 32, 1)
        assert core.dtype == sums.dtype == gate.dtype == t.float32
        core = t.to_memory_config(core, self.memory)
        sums = t.to_memory_config(sums, self.sum_memory)
        gate = t.to_memory_config(gate, self.memory)
        output = t.empty(
            (1, 32, 128), dtype=t.bfloat16, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.memory
        )
        bindings = [(0, core), (1, sums), (2, self.gamma), (3, gate), (16, output)]
        debug_outputs = {}
        if self.debug:
            for index, name in ((17, "normalized_fp32"), (18, "weighted_fp32_carrier"), (19, "inverse_repeated")):
                tensor = t.empty(
                    (1, 32, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.memory
                )
                bindings.append((index, tensor))
                debug_outputs[name] = tensor
        descriptor = t.KernelDescriptor(
            kernel_source=self.source,
            source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self.cores,
            compile_time_args=[int(self.debug)],
            runtime_args=[],
            config=self.config,
        )
        program = t.ProgramDescriptor(
            kernels=[descriptor],
            semaphores=[],
            cbs=[t.cb_descriptor_from_sharded_tensor(i, value) for i, value in bindings],
        )
        t.generic_op([value for _, value in bindings], program)
        return output, debug_outputs


def cpu_tail(core, sums, gamma, gate):
    variance = sums * (1 / 128)
    shifted = variance + EPSILON
    inverse = torch.rsqrt(shifted)
    normalized = core.float() * inverse
    weighted = (normalized.bfloat16().float() * gamma.float().reshape(1, 1, 128)).bfloat16().float()
    return (weighted * gate).bfloat16(), dict(
        normalized_fp32=normalized, weighted_fp32_carrier=weighted, inverse=inverse
    )


def fixtures(checkpoint, random_cases):
    key = "model.language_model.layers.34.linear_attn.norm.weight"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(checkpoint / index[key], framework="pt", device="cpu") as handle:
        gamma = handle.get_tensor(key)
    assert gamma.dtype == torch.bfloat16 and tuple(gamma.shape) == (128,)
    result = []
    for name in ("v5_gated_norm_same_operand_device.json", "v6_gated_norm_3271_device.json"):
        meta = json.loads((E / name).read_text())
        stages = torch.load(meta["tensors"], weights_only=False)
        for row in meta["cases"]:
            capture = torch.load(row["capture"], weights_only=False)
            cpu = stages[str(row["position"])]["cpu_reference"]
            result.append(
                dict(
                    name=f"position{row['position']}",
                    core=cpu["core_bf16"],
                    z=capture["tt_values"]["projection_z"].reshape(1, 32, 128).contiguous(),
                    expected=cpu["gated_bf16"],
                    evidence=name,
                    evidence_sha256=sha(E / name),
                    capture_sha256=sha(row["capture"]),
                )
            )
    generator = torch.Generator().manual_seed(613684)
    for i in range(random_cases):
        core = (torch.randn(1, 32, 128, generator=generator) * (10 ** (-4 + i % 5))).bfloat16()
        z = (torch.randn(1, 32, 128, generator=generator) * 3).bfloat16()
        gate = torch.nn.functional.silu(z.float())
        expected, _ = cpu_tail(core, core.float().square().sum(-1, keepdim=True), gamma, gate)
        result.append(dict(name=f"random{i}", core=core, z=z, expected=expected))
    return gamma, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--random-cases", type=int, default=8)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists(), "Preserve evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    gamma, cases = fixtures(args.checkpoint, args.random_cases)
    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    table = torch.nn.functional.silu(codes.float()).reshape(1, 65536)
    result = dict(
        classification="Post-reduction tail component diagnostic; not decoder acceptance",
        acceptance_eligible=False,
        oracle_assisted=False,
        status="cpu_setup_pass",
        script_sha256=sha(__file__),
        kernel_sha256=hashlib.sha256(kernel_source().encode()).hexdigest(),
        device_closed=False,
        cases=[],
        limitations="Native square/sum and existing SiLU lookup remain unchanged. Single-core tail requires input placement copies; no speedup claim. BF16 core, normalized, weighted and final boundaries are retained. Finite inputs only.",
    )
    result["gamma_sha256"] = hashlib.sha256(gamma.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    result["table_sha256"] = hashlib.sha256(table.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    result["lookup_source_sha256"] = sha(Path(__file__).with_name("v6_silu_lookup_probe.py"))
    method = next(
        n
        for n in ast.walk(ast.parse(Path(__file__).read_text()))
        if isinstance(n, ast.ClassDef) and n.name == "GatedNormTail"
    )
    call = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    calls = [ast.unparse(n.func) for n in ast.walk(call) if isinstance(n, ast.Call)]
    assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in calls)
    result["runtime_audit"] = dict(
        host_conversion_calls=[],
        placement_copies=3,
        source_ast_sha256=hashlib.sha256(ast.dump(call).encode()).hexdigest(),
    )
    for case in cases:
        lookup = table.flatten()[case["z"].view(torch.int16).long() & 65535]
        assert torch.equal(lookup, torch.nn.functional.silu(case["z"].float()))
        actual, _ = cpu_tail(case["core"], case["core"].float().square().sum(-1, keepdim=True), gamma, lookup)
        row = dict(name=case["name"], cpu_math_vs_reference=code_compare(case["expected"], actual))
        row.update({name: case[name] for name in ("evidence", "evidence_sha256", "capture_sha256") if name in case})
        assert row["cpu_math_vs_reference"]["bf16_code_differences"] == 0
        result["cases"].append(row)
    snapshot = Path(str(args.output) + ".source.py.snapshot")
    snapshot.write_bytes(Path(__file__).read_bytes())
    Path(str(args.output) + ".compute.cpp.snapshot").write_text(kernel_source())
    result["source_snapshot"] = str(snapshot)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_GATED_NORM_TAIL_PASS", len(cases))
        return
    import ttnn as t

    mesh = None
    try:
        mesh = t.open_mesh_device(t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        mem = t.L1_MEMORY_CONFIG
        compute = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(x, dtype=t.float32, layout=t.TILE_LAYOUT):
            return t.from_torch(
                x.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=t.DRAM_MEMORY_CONFIG
            )

        lut = upload(table, layout=t.ROW_MAJOR_LAYOUT)
        weight = upload(gamma.float().reshape(1, 1, 128))
        helper = GatedNormTail(t, mesh, gamma, debug=True)
        round_chain = [
            t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.float32.value), float(t.bfloat16.value)),
            t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.bfloat16.value), float(t.float32.value)),
        ]
        device_tensors = {}
        for case, row in zip(cases, result["cases"]):
            core = upload(case["core"], t.bfloat16)
            wide = t.typecast(core, t.float32, memory_config=mem)
            squares = t.multiply(wide, wide, dtype=t.float32, fast_and_approximate_mode=False, memory_config=mem)
            sums = t.sum(squares, dim=-1, keepdim=True, compute_kernel_config=compute, memory_config=mem)
            gate = device_lookup(t, upload(case["z"], t.bfloat16), lut)
            variance = t.multiply(sums, 1 / 128, dtype=t.float32, fast_and_approximate_mode=False, memory_config=mem)
            shifted = t.add(variance, EPSILON, memory_config=mem)
            inverse = t.rsqrt(shifted, fast_and_approximate_mode=False, memory_config=mem)
            normalized = t.multiply(wide, inverse, dtype=t.float32, fast_and_approximate_mode=False, memory_config=mem)
            normalized_bf16 = t.typecast(normalized, t.bfloat16, memory_config=mem)
            weighted = t.multiply(
                normalized_bf16,
                weight,
                dtype=t.float32,
                fast_and_approximate_mode=False,
                activations=round_chain,
                memory_config=mem,
            )
            expected = t.multiply(weighted, gate, dtype=t.bfloat16, fast_and_approximate_mode=False, memory_config=mem)
            output, diagnostic = helper(wide, sums, gate)
            host = {name: t.to_torch(value).clone() for name, value in diagnostic.items()}
            host["gated_bf16"] = t.to_torch(output).clone()
            native = {
                name: t.to_torch(value).clone()
                for name, value in dict(
                    normalized_fp32=normalized, weighted_fp32_carrier=weighted, inverse=inverse, gated_bf16=expected
                ).items()
            }
            row["fused_vs_native"] = {
                name: code_compare(native[name], host[name])
                for name in ("normalized_fp32", "weighted_fp32_carrier", "gated_bf16")
            }
            row["fused_vs_native"]["inverse"] = code_compare(native["inverse"], host["inverse_repeated"][..., :1])
            row["native_vs_cpu_reference"] = code_compare(case["expected"], native["gated_bf16"])
            device_tensors[case["name"]] = dict(fused=host, native=native, sums=t.to_torch(sums).clone())
            save()
            print(
                case["name"],
                {name: value["fp32_code_differences"] for name, value in row["fused_vs_native"].items()},
                flush=True,
            )
        tensor_path = Path(str(args.output) + ".device_tensors.pt")
        torch.save(device_tensors, tensor_path)
        result.update(
            status="completed_diagnostic",
            device_tensors=str(tensor_path),
            device_tensors_sha256=sha(tensor_path),
            exact_native_tail=all(
                v["fp32_code_differences"] == 0 for row in result["cases"] for v in row["fused_vs_native"].values()
            ),
        )
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
