# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only measured FP32 GEMM accumulation-order comparison on real prefix operands."""

import ctypes
import hashlib
import inspect
import json
import subprocess
import tempfile
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm

from models.autoports.qwen_qwen3_6_35b_a3b.tests import reference

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
CPP = r"""
#include <cmath>
extern "C" void dot_probe(const float* a,const float* b,float* out,int batch,int m,int k,int n,int groups,int mode,int combine) {
 for(int h=0;h<batch;++h) for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
  float acc[16]={};
  for(int p=0;p<k;++p) {
   float x=a[(h*m+i)*k+p], y=b[(h*k+p)*n+j]; int g=p%groups;
   if(mode==0) acc[g]=std::fma(x,y,acc[g]);
   else { volatile float product=x*y; acc[g]=acc[g]+product; }
  }
  if(combine==0) { float value=acc[0]; for(int g=1;g<groups;++g)value=value+acc[g]; out[(h*m+i)*n+j]=value; }
  else { for(int stride=1;stride<groups;stride*=2) for(int g=0;g+stride<groups;g+=2*stride) acc[g]=acc[g]+acc[g+stride]; out[(h*m+i)*n+j]=acc[0]; }
 }
}
"""


def compare(expected, actual):
    diff = expected - actual
    return dict(
        fp32_different=int((expected.view(torch.int32) != actual.view(torch.int32)).sum()),
        numeric_different=int((expected != actual).sum()),
        bf16_different=int((expected.bfloat16() != actual.bfloat16()).sum()),
        max_abs=float(diff.abs().max()),
        relative_l2=float(diff.norm() / expected.norm()),
    )


def main():
    torch.set_num_threads(4)
    meta = json.loads((DOC / "v5_combined_layer34_stream_probe.json").read_text())
    oracle = torch.load(meta["oracle"], weights_only=False)
    hf = reference.layer(reference.weights(34), 34)
    cache = DynamicCache(config=reference.config())
    captured = {}
    old = hf.linear_attn.chunk_gated_delta_rule

    def capture(q, k, v, **kw):
        captured.update(q=q.clone(), k=k.clone(), v=v.clone(), kw=kw)
        return old(q, k, v, **kw)

    hf.linear_attn.chunk_gated_delta_rule = capture
    with torch.no_grad():
        hf.linear_attn(hf.input_layernorm(oracle["prefix"]), cache_params=cache)
    q, k, v = [captured[x] for x in ("q", "k", "v")]
    k = l2norm(k, dim=-1, eps=1e-6).transpose(1, 2).float().contiguous()
    v = v.transpose(1, 2).float().contiguous()
    beta = captured["kw"]["beta"].transpose(1, 2).float().contiguous()
    raw_g = captured["kw"]["g"].transpose(1, 2).float().contiguous().reshape(1, 32, 2, 64)
    g = raw_g.cumsum(-1)
    sequential_g = torch.empty_like(g)
    partial_g = torch.zeros_like(g[..., 0])
    for index in range(64):
        partial_g = partial_g + raw_g[..., index]
        sequential_g[..., index] = partial_g
    cumulative_log_decay = {
        "sequential_fp32": compare(g, sequential_g),
        "fp64_then_fp32": compare(g, raw_g.double().cumsum(-1).float()),
    }
    k = k.reshape(1, 32, 2, 64, 128)
    v = v.reshape(1, 32, 2, 64, 128)
    kb = k * beta.reshape(1, 32, 2, 64, 1)
    vb = v * beta.reshape(1, 32, 2, 64, 1)
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    gram = kb @ k.transpose(-1, -2)
    a = -(gram * decay).masked_fill(torch.ones(64, 64, dtype=torch.bool).triu(), 0)
    triangular_reductions = {}
    for i in range(1, 64):
        row = a[..., i, :i].clone()
        sub = a[..., :i, :i].clone()
        products = row.unsqueeze(-1) * sub
        reduced = products.sum(-2)
        if i in (7, 31, 63):
            choices = {}
            for groups in (1, 2, 4, 8):
                accumulators = [torch.zeros_like(reduced) for _ in range(groups)]
                for index in range(i):
                    accumulators[index % groups] = accumulators[index % groups] + products[..., index, :]
                value = accumulators[0]
                for extra in accumulators[1:]:
                    value = value + extra
                choices[f"groups{groups}_linear"] = compare(reduced, value)
            for groups in (2, 4, 8):
                for combination in ("linear", "tree"):
                    accumulators = [torch.zeros_like(reduced) for _ in range(groups)]
                    end = i // groups * groups
                    for index in range(end):
                        accumulators[index % groups] = accumulators[index % groups] + products[..., index, :]
                    if combination == "linear":
                        value = accumulators[0]
                        for extra in accumulators[1:]:
                            value = value + extra
                    else:
                        while len(accumulators) > 1:
                            accumulators = [
                                accumulators[j] + accumulators[j + 1] for j in range(0, len(accumulators), 2)
                            ]
                        value = accumulators[0]
                    for index in range(end, i):
                        value = value + products[..., index, :]
                    choices[f"groups{groups}_{combination}_then_tail"] = compare(reduced, value)
            triangular_reductions[str(i)] = choices
        a[..., i, :i] = row + reduced
    a = a + torch.eye(64)
    vv = a @ vb
    kg = kb * g.exp().unsqueeze(-1)
    kc = a @ kg
    state = torch.zeros(1, 32, 128, 128)
    fixtures = [
        ("k_beta_key_transpose", kb, k.transpose(-1, -2), gram),
        ("a_v_beta", a, vb, vv),
        ("a_k_beta_cumdecay", a, kg, kc),
    ]
    for chunk in range(2):
        prime = kc[:, :, chunk] @ state
        new = vv[:, :, chunk] - prime
        sk = (k[:, :, chunk] * (g[:, :, chunk, -1, None] - g[:, :, chunk]).exp()[..., None]).transpose(-1, -2)
        update = sk @ new
        fixtures.append((f"final_update_chunk{chunk}", sk, new, update))
        if chunk:
            fixtures.append(("k_cumdecay_state", kc[:, :, chunk], state, prime))
        state = state * g[:, :, chunk, -1, None, None].exp() + update
    result = dict(
        classification="CPU GEMM ordering diagnostic, not device acceptance",
        torch_version=torch.__version__,
        torch_build=torch.__config__.show(),
        source_hashes={
            str(Path(path)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (reference.__file__, inspect.getfile(old))
        },
        prefix_sha256=hashlib.sha256(oracle["prefix"].view(torch.uint8).numpy().tobytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpp_sha256=hashlib.sha256(CPP.encode()).hexdigest(),
        fixtures={},
        cumulative_log_decay=cumulative_log_decay,
        triangular_reductions=triangular_reductions,
    )
    ptr = ctypes.POINTER(ctypes.c_float)
    with tempfile.TemporaryDirectory(prefix="chunk_order_") as directory:
        source = Path(directory) / "order.cpp"
        binary = Path(directory) / "order.so"
        source.write_text(CPP)
        command = [
            "g++-13",
            "-shared",
            "-fPIC",
            "-O2",
            "-ffp-contract=off",
            "-march=native",
            str(source),
            "-o",
            str(binary),
        ]
        subprocess.run(command, check=True)
        library = ctypes.CDLL(str(binary))
        fn = library.dot_probe
        fn.argtypes = [ptr, ptr, ptr] + [ctypes.c_int] * 7
        for name, left, right, expected in fixtures:
            # Preserve batched Torch matrix multiplication shapes for expected values.
            m, kk, n = left.shape[-2], left.shape[-1], right.shape[-1]
            left = left.contiguous().reshape(-1, m, kk)
            right = right.contiguous().reshape(-1, kk, n)
            expected = expected.contiguous().reshape(-1, m, n)
            full_output = torch.empty_like(expected)
            fn(
                ctypes.cast(left.data_ptr(), ptr),
                ctypes.cast(right.data_ptr(), ptr),
                ctypes.cast(full_output.data_ptr(), ptr),
                left.shape[0],
                m,
                kk,
                n,
                1,
                0,
                0,
            )
            full_comparison = compare(expected, full_output)
            full_shape = [left.shape[0], m, kk, n]
            # Four actual heads from each chunk suffice to distinguish order; never claim whole-model proof.
            count = min(8, left.shape[0])
            left = left[:count].contiguous()
            right = right[:count].contiguous()
            expected = expected[:count].contiguous()
            record = dict(
                shape=[count, m, kk, n],
                full_head_shape=full_shape,
                full_head_sequential_fma=full_comparison,
                comparisons={},
                input_sha256={
                    label: hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
                    for label, t in [("left", left), ("right", right), ("expected", expected)]
                },
            )
            for groups in (1, 2, 4, 8, 16):
                for mode in (0, 1):
                    for combine in (0,) if groups <= 2 else (0, 1):
                        output = torch.empty_like(expected)
                        fn(
                            ctypes.cast(left.data_ptr(), ptr),
                            ctypes.cast(right.data_ptr(), ptr),
                            ctypes.cast(output.data_ptr(), ptr),
                            count,
                            m,
                            kk,
                            n,
                            groups,
                            mode,
                            combine,
                        )
                        record["comparisons"][
                            f'{"fma" if mode==0 else "mul_add"}_groups{groups}_{"linear" if combine==0 else "tree"}'
                        ] = compare(expected, output)
            # Explicit full product followed by Torch reduction, matching the proposed SFPU graph topology only.
            reduced = torch.stack([(left[h, :, :, None] * right[h, None, :, :]).sum(1) for h in range(count)])
            record["comparisons"]["products_torch_sum"] = compare(expected, reduced)
            result["fixtures"][name] = record
            print(name, {k: v["fp32_different"] for k, v in record["comparisons"].items()}, flush=True)
    result["compiler_command"] = command
    (DOC / "v6_chunk_cpu_order_probe.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
