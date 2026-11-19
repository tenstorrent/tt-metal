# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only split-product/expansion FMA experiment; no device acceptance claim."""

import ctypes
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import torch

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
INPUT = DOC / "v6_chunk64_sfpu_fma_compensated_device_v2.json.cpu_matrices.pt"
CPP = r"""
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
struct Pair {float hi,lo;};
static inline float add(float a,float b) {volatile float t=a+b;return t;}
static inline float sub(float a,float b) {volatile float t=a-b;return t;}
static inline float mul(float a,float b) {volatile float t=a*b;return t;}
static inline Pair sum(float a,float b) {float s=add(a,b),z=sub(s,a);return {s,add(sub(a,sub(s,z)),sub(b,z))};}
static inline Pair split(float a) {float t=mul(4097.f,a),h=sub(t,sub(t,a));return {h,sub(a,h)};}
static inline Pair product(float a,float b) {
 auto x=split(a),y=split(b);float p=mul(a,b);
 float e1=sub(p,mul(x.hi,y.hi));
 float e2=sub(e1,mul(x.lo,y.hi));
 float e3=sub(e2,mul(x.hi,y.lo));
 return {p,sub(mul(x.lo,y.lo),e3)};
}
// A zero-eliminated expansion in increasing magnitude; fixed capacity is sufficient.
struct Expansion {float x[8];int n;};
static inline Expansion grow(Expansion e,float b) {
 Expansion h={{},0};float q=b;
 for(int i=0;i<e.n;++i) {auto t=sum(q,e.x[i]);if(t.lo!=0)h.x[h.n++]=t.lo;q=t.hi;}
 if(q!=0 || h.n==0)h.x[h.n++]=q;return h;
}
static inline int sign(Expansion e) {float h=e.x[e.n-1];return (h>0)-(h<0);}
static inline uint32_t bits(float a) {uint32_t u;std::memcpy(&u,&a,4);return u;}
static inline float corrected(float a,float b,float c,int* fixes=nullptr) {
 auto p=product(a,b);auto s=sum(p.hi,c);float r=add(s.hi,add(s.lo,p.lo));
 Expansion e={{p.lo,p.hi},2};e=grow(e,c);auto rem=grow(e,-r);int direction=sign(rem);
 if(!direction) {if(r==0) return (mul(a,b)==0 && c==0)?add(mul(a,b),c):0.f;return r;}
 float next=std::nextafter(r,direction>0?INFINITY:-INFINITY);
 // Valid as a normal-domain FP32 operation. Subnormal half-spacing is deliberately not repaired here.
 float half=mul(sub(next,r),.5f);
 int cmp=sign(grow(rem,-half))*direction;
 if(cmp>0 || (cmp==0 && (bits(r)&1))) {if(fixes)++*fixes;return next;}
 return r;
}
static inline bool bounded(float a,float b,float c) {
 auto ab=[](float x,int lo,int hi) {return x==0 || (std::fabs(x)>=std::ldexp(1.f,lo) && std::fabs(x)<=std::ldexp(1.f,hi));};
 return ab(a,-30,30)&&ab(b,-30,30)&&ab(c,-60,60);
}
static inline float ftz(float a) {return std::fpclassify(a)==FP_SUBNORMAL?std::copysign(0.f,a):a;}
extern "C" void matrix(const float* a,const float* b,float* out,float* oracle,float* flushed,int batch,int m,int k,int n,uint64_t* stats,float* examples) {
 for(int h=0;h<batch;++h)for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
  float c=0,ref=0,f=0;
  for(int p=0;p<k;++p) {
   float x=a[(h*m+i)*k+p],y=b[(h*k+p)*n+j];float expected=std::fma(x,y,ref);int fix=0;
   float actual=corrected(x,y,ref,&fix);bool safe=bounded(x,y,ref);
   if(fix && stats[4]<4) {int t=5*stats[4];examples[t]=x;examples[t+1]=y;examples[t+2]=ref;examples[t+3]=expected;examples[t+4]=actual;}
   ++stats[0];stats[1]+=safe;stats[2]+=(actual!=expected);stats[3]+=safe&&(actual!=expected);stats[4]+=fix;
   stats[5]+=(bits(actual)!=bits(expected));stats[6]+=safe&&(bits(actual)!=bits(expected));
   ref=expected;c=corrected(x,y,c);f=ftz(std::fma(ftz(x),ftz(y),ftz(f)));
  }
  int at=(h*m+i)*n+j;out[at]=c;oracle[at]=ref;flushed[at]=f;
 }
}
static uint64_t state=0x28edab39f26d9c37ULL;
static uint32_t rng(){state^=state<<13;state^=state>>7;state^=state<<17;return uint32_t(state);}
static float random_value(int emin,int emax) {uint32_t u=(rng()&0x807fffffu)|((127+emin+int(rng()%uint32_t(emax-emin+1)))<<23);float f;std::memcpy(&f,&u,4);return f;}
extern "C" void midpoint_probe(float* result) {
 float a=1.f+std::ldexp(1.f,-23),b=1.f-std::ldexp(1.f,-23),c=16777218.f;
 int fixes=0;auto p=product(a,b);auto s=sum(p.hi,c);
 result[0]=a;result[1]=b;result[2]=c;result[3]=add(s.hi,add(s.lo,p.lo));
 result[4]=corrected(a,b,c,&fixes);result[5]=std::fma(a,b,c);result[6]=float(fixes);
}
extern "C" void random_probe(uint64_t count,uint64_t* stats) {
 for(uint64_t i=0;i<count;++i) {
  float a=random_value(-29,29),b=random_value(-29,29),c=random_value(-59,59);
  if(i%3==0)c=-mul(a,b); // Exact-product cancellation stresses the retained tail.
  if(i%3==1) {float p=mul(a,b);float next=std::nextafter(p,INFINITY);c=mul(sub(next,p),.5f);}
  int fixes=0;float r=corrected(a,b,c,&fixes),expected=std::fma(a,b,c);
  ++stats[0];stats[1]+=bounded(a,b,c);stats[2]+=(r!=expected);stats[3]+=(bits(r)!=bits(expected));stats[4]+=fixes;
 }
}
"""


def compare(a, b):
    return {
        "fp32_bits_different": int((a.view(torch.int32) != b.view(torch.int32)).sum()),
        "numeric_different": int((a != b).sum()),
        "bf16_different": int((a.bfloat16() != b.bfloat16()).sum()),
        "max_abs": float((a - b).abs().max()),
    }


def main():
    torch.set_num_threads(2)
    saved = torch.load(INPUT, weights_only=False)
    case = saved["cases"][128]
    matrices = case["matrices"]
    key = case["operands"]["key"]
    fixtures = []
    for chunk in range(2):
        get = lambda name: matrices[f"chunk{chunk}/{name}"]
        k = key[:, :, chunk * 64 : (chunk + 1) * 64]
        state = saved["initial"] if chunk == 0 else matrices["chunk0/state"]
        fixtures.extend(
            [
                (f"chunk{chunk}/gram", get("k_beta"), k.transpose(-1, -2)),
                (f"chunk{chunk}/transformed_v", get("A_final"), get("v_beta")),
                (f"chunk{chunk}/transformed_k", get("A_final"), get("k_beta") * get("cumulative_g").exp()),
                (f"chunk{chunk}/state_contribution", get("transformed_k"), state),
                (f"chunk{chunk}/state_update", get("weighted_k").transpose(-1, -2), get("v_new")),
            ]
        )
    result = {
        "classification": "CPU arithmetic experiment; no device acceptance",
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cpp_sha256": hashlib.sha256(CPP.encode()).hexdigest(),
        "bounded_domain": "a,b zero or abs in [2^-30,2^30]; c zero or abs in [2^-60,2^60]",
        "fixtures": {},
    }
    ptr = ctypes.POINTER(ctypes.c_float)
    counts = ctypes.POINTER(ctypes.c_uint64)
    with tempfile.TemporaryDirectory(prefix="fma_expansion_") as directory:
        src = Path(directory) / "probe.cpp"
        binary = Path(directory) / "probe.so"
        src.write_text(CPP)
        command = [
            "g++-13",
            "-O2",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-march=native",
            str(src),
            "-o",
            str(binary),
        ]
        subprocess.run(command, check=True)
        lib = ctypes.CDLL(str(binary))
        lib.matrix.argtypes = [ptr] * 5 + [ctypes.c_int] * 4 + [counts, ptr]
        lib.random_probe.argtypes = [ctypes.c_uint64, counts]
        lib.midpoint_probe.argtypes = [ptr]
        midpoint = (ctypes.c_float * 7)()
        lib.midpoint_probe(midpoint)
        result["midpoint_counterexample"] = dict(
            zip(["a", "b", "c", "uncorrected", "corrected", "std_fma", "correction_count"], midpoint)
        )
        random_stats = (ctypes.c_uint64 * 5)()
        lib.random_probe(2000000, random_stats)
        result["random_2m"] = dict(
            zip(["triples", "bounded", "numeric_different", "bit_different", "midpoint_corrections"], random_stats)
        )
        print(result["random_2m"], flush=True)
        for name, a, b in fixtures:
            m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
            # All 32 actual heads; all K steps, not cherry-picked numerical triples.
            a = a.reshape(-1, m, k)[:32].contiguous()
            b = b.reshape(-1, k, n)[:32].contiguous()
            out = torch.empty(a.shape[0], m, n)
            oracle = torch.empty_like(out)
            flushed = torch.empty_like(out)
            stats = (ctypes.c_uint64 * 7)()
            examples = (ctypes.c_float * 20)()
            lib.matrix(
                *[ctypes.cast(t.data_ptr(), ptr) for t in (a, b, out, oracle, flushed)],
                a.shape[0],
                m,
                k,
                n,
                stats,
                examples,
            )
            row = dict(
                shape=[a.shape[0], m, k, n],
                midpoint_examples=[
                    dict(zip(["a", "b", "acc", "std_fma", "corrected"], examples[5 * i : 5 * i + 5]))
                    for i in range(min(4, stats[4]))
                ],
                triples=dict(
                    zip(
                        [
                            "count",
                            "bounded",
                            "numeric_different",
                            "bounded_numeric_different",
                            "midpoint_corrections",
                            "bit_different",
                            "bounded_bit_different",
                        ],
                        stats,
                    )
                ),
                sequential_expansion=compare(oracle, out),
                flush_subnormals_fma=compare(oracle, flushed),
            )
            result["fixtures"][name] = row
            print(name, row, flush=True)
    (DOC / "v6_fma_expansion_proof.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["random_2m"]["bit_different"] == 0
    assert all(row["triples"]["bounded_bit_different"] == 0 for row in result["fixtures"].values())
    assert result["midpoint_counterexample"]["corrected"] == result["midpoint_counterexample"]["std_fma"]
    assert result["midpoint_counterexample"]["uncorrected"] != result["midpoint_counterexample"]["std_fma"]


if __name__ == "__main__":
    main()
