# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU exp algorithm and saved-device boundary study; no hardware or TTNN."""
import ctypes
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_state_probe import code_compare

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
CPP = r"""
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <immintrin.h>
inline float op(float a,float b,float c,int mode) {
 if(mode==4) return std::fma(a,b,c);
 volatile float product=a*b; return product+c;
}
extern "C" void eval_exp(const float* input,float* output,int count,int mode,const char* library) {
 void* handle=dlopen(library,RTLD_LAZY);
 auto f8=(__m256(*)(__m256))dlsym(handle,"Sleef_expf8_u10");
 auto f16=(__m512(*)(__m512))dlsym(handle,"Sleef_expf16_u10");
 for(int i=0;i<count;) {
  if(mode==2 && i+8<=count) { _mm256_storeu_ps(output+i,f8(_mm256_loadu_ps(input+i)));i+=8;continue; }
  if(mode==3 && i+16<=count) { _mm512_storeu_ps(output+i,f16(_mm512_loadu_ps(input+i)));i+=16;continue; }
  float x=input[i],result;
  if(mode==0) result=std::exp(x);
  else if(mode==1) result=(float)std::exp((double)x);
  else if(mode==4 || mode==5) {
    float j=std::nearbyint(1.4426950216293334961f*x);
    float f=op(j,-6.93145752e-1f,x,mode);
    f=op(j,-1.42860677e-6f,f,mode);
    float r=op(1.37805939e-3f,f,8.37312452e-3f,mode);
    r=op(r,f,4.16695364e-2f,mode);
    r=op(r,f,1.66664720e-1f,mode);
    r=op(r,f,4.99999851e-1f,mode);
    float y=op(r,f,1.0f,mode); r=op(y,f,1.0f,mode);
    result=std::ldexp(r,(int)j);
    if(result<0x1p-126f)result=0.0f;
  } else result=std::exp(x);
  output[i++]=result;
 }
 dlclose(handle);
}
"""


def metric(expected, actual):
    expected, actual = expected.contiguous().float(), actual.contiguous().float()
    codes = (expected.view(torch.int32).long() - actual.view(torch.int32).long()).abs()
    normal = expected >= torch.finfo(torch.float32).tiny
    return dict(
        fp32_code_differences=int((codes != 0).sum()),
        bf16_code_differences=int((expected.bfloat16().view(torch.int16) != actual.bfloat16().view(torch.int16)).sum()),
        normal_output_code_differences=int(((codes != 0) & normal).sum()),
        max_normal_ulp=int(codes[normal].max()) if normal.any() else 0,
        max_abs=float((expected - actual).abs().max()),
        expected_subnormal_count=int(((expected > 0) & ~normal).sum()),
    )


def main():
    torch.set_num_threads(4)
    source = E / "v6_chunk64_ordered_state_device.json"
    meta = json.loads(source.read_text())
    cpu_path, device_path = Path(meta["cpu_matrix_snapshot"]), Path(meta["device_matrix_snapshot"])
    cpu = torch.load(cpu_path, weights_only=False)
    case = cpu["cases"][128]
    dev = torch.load(device_path, weights_only=False)[128]["intermediates"]
    rows, boundaries, state_controls = [], [], []
    saved_predictions = {}
    library_path = str(Path(torch.__file__).parent / "lib/libtorch_cpu.so")
    ptr = ctypes.POINTER(ctypes.c_float)
    with tempfile.TemporaryDirectory(prefix="v6_exp_") as directory:
        cpp, binary = Path(directory) / "probe.cpp", Path(directory) / "probe.so"
        cpp.write_text(CPP)
        subprocess.run(
            [
                "g++-13",
                "-shared",
                "-fPIC",
                "-O2",
                "-ffp-contract=off",
                "-mavx512f",
                "-mavx2",
                "-mfma",
                str(cpp),
                "-ldl",
                "-o",
                str(binary),
            ],
            check=True,
        )
        lib = ctypes.CDLL(str(binary))
        fn = lib.eval_exp
        fn.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        variants = (
            "libm_expf",
            "libm_double_exp_rounded",
            "SLEEF8_u10",
            "SLEEF16_u10",
            "Juffa_true_fma_simulation",
            "Juffa_unfused_simulation",
        )
        for chunk in range(2):
            prefix = f"chunk{chunk}/"
            cg = case["matrices"][prefix + "cumulative_g"]
            dg = dev[prefix + "cumulative_g"]
            for name, operand in (
                ("canonical_cumulative", cg),
                ("canonical_lower_difference", (cg - cg.transpose(-1, -2)).tril()),
                ("canonical_last_difference", cg[..., -1:, :] - cg),
                ("device_cumulative", dg),
                ("device_lower_difference", (dg - dg.transpose(-1, -2)).tril()),
                ("device_last_difference", dg[..., -1:, :] - dg),
            ):
                operand = operand.contiguous()
                expected = operand.exp()
                record = dict(
                    chunk=chunk,
                    argument=name,
                    shape=list(operand.shape),
                    minimum=float(operand.min()),
                    maximum=float(operand.max()),
                    comparisons={},
                )
                for mode, variant in enumerate(variants):
                    out = torch.empty_like(operand)
                    fn(
                        ctypes.cast(operand.data_ptr(), ptr),
                        ctypes.cast(out.data_ptr(), ptr),
                        operand.numel(),
                        mode,
                        library_path.encode(),
                    )
                    record["comparisons"][variant] = metric(expected, out)
                    if name == "device_lower_difference" and mode == 4:
                        saved_predictions[chunk] = out.tril().clone()
                record["comparisons"]["torch_double_exp_rounded"] = metric(expected, operand.double().exp().float())
                rows.append(record)
            canonical_on_actual = (dg - dg.transpose(-1, -2)).tril().exp().tril()
            k = case["operands"]["key"][..., chunk * 64 : (chunk + 1) * 64, :]
            weighted_on_actual = k * (dg[..., -1:, :] - dg).exp()
            boundaries.append(
                dict(
                    chunk=chunk,
                    decay_mask_vs_torch_on_actual_device_cumsum=metric(canonical_on_actual, dev[prefix + "decay_mask"]),
                    native_decay_mask_vs_juffa_true_fma_transcription=metric(
                        saved_predictions[chunk], dev[prefix + "decay_mask"]
                    ),
                    weighted_k_vs_torch_on_actual_device_cumsum=code_compare(
                        weighted_on_actual, dev[prefix + "weighted_k"]
                    ),
                    limitation="Includes device subtraction/masking/multiplication; standalone device exp output was not captured.",
                )
            )
        sites = ("decay_mask", "cumulative_g", "last_difference", "last_g")
        controls = [(mode, sites) for mode in (-1, 1, 2, 4, 5)]
        controls += [(mode, (site,)) for mode in (1, 4) for site in sites]
        for mode, replaced_sites in controls:

            def exp(value, site):
                if mode == -1 or site not in replaced_sites:
                    return value.exp()
                value = value.contiguous()
                out = torch.empty_like(value)
                fn(
                    ctypes.cast(value.data_ptr(), ptr),
                    ctypes.cast(out.data_ptr(), ptr),
                    value.numel(),
                    mode,
                    library_path.encode(),
                )
                return out

            state = cpu["initial"].clone()
            for start in (0, 64):
                key, val, raw_g, beta = [
                    case["operands"][name][..., start : start + 64, :] for name in ("key", "value", "g", "beta")
                ]
                kb, vb = key * beta, val * beta
                cg = raw_g.cumsum(-2)
                decay = exp((cg - cg.transpose(-1, -2)).tril(), "decay_mask").tril()
                strict = torch.ones(64, 64, dtype=torch.bool).tril(-1)
                a = -((kb @ key.transpose(-1, -2)) * decay).masked_fill(~strict, 0)
                for index in range(1, 64):
                    row, sub = a[..., index, :index].clone(), a[..., :index, :index].clone()
                    a[..., index, :index] = row + (row.unsqueeze(-1) * sub).sum(-2)
                a = a + torch.eye(64)
                new = a @ vb - (a @ (kb * exp(cg, "cumulative_g"))) @ state
                last = cg[..., -1:, :]
                wk = key * exp(last - cg, "last_difference")
                state = state * exp(last, "last_g") + wk.transpose(-1, -2) @ new
            state_controls.append(
                dict(
                    exp_variant="Torch_MKL" if mode == -1 else variants[mode],
                    replaced_exp_sites=list(replaced_sites),
                    final=code_compare(case["canonical"], state),
                )
            )
    result = dict(
        classification="CPU exp controls and saved-device boundary audit; not acceptance or direct native-exp measurement",
        torch_build=torch.__config__.show(),
        torch_exp_path="MKL vmsExp, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE",
        inspected_source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(torch.__file__).parent / "include/ATen/Config.h",
                Path(torch.__file__).parent / "include/ATen/cpu/vml.h",
                Path(torch.__file__).parent / "include/ATen/cpu/vec/vec256/vec256_float.h",
                Path(torch.__file__).parent / "include/ATen/cpu/vec/vec512/vec512_float.h",
                E / "v6_torch_UnaryOpsKernel.cpp.snapshot",
                Path("tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h"),
            )
        },
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpp_sha256=hashlib.sha256(CPP.encode()).hexdigest(),
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, cpu_path, device_path)},
        rows=rows,
        full_state_exp_only_controls=state_controls,
        saved_device_boundaries=boundaries,
    )
    (E / "v6_exp_boundary_cpu_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(
            row["chunk"],
            row["argument"],
            {k: (v["fp32_code_differences"], v["max_normal_ulp"]) for k, v in row["comparisons"].items()},
        )
    print("boundaries", json.dumps(boundaries))
    print("states", json.dumps(state_controls))


if __name__ == "__main__":
    main()
