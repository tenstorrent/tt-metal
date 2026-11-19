# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU causal/order proof for an in-place chunk64 triangular recurrence kernel.

No TTNN import or hardware. This is not evidence of device arithmetic exactness.
"""

import argparse
import ast
import ctypes
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_triangular_sum_order_probe import canonical_outer_sum

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_streaming_scan_norm_tail_fused_matmul.py.snapshot"
BASE_SHA = "8cda999c18f9245c872e40d7f67e027838b56396056ede43e162b7a8386f3faa"
ROWS = (1, 2, 31, 63)
CPP = r"""
#include <stdint.h>
static inline float rounded_mul(float a,float b) { volatile float x=a*b;return x; }
static inline float rounded_add(float a,float b) { volatile float x=a+b;return x; }
extern "C" void triangular(const float* input,float* output,float* snapshots,uint32_t heads) {
 for(uint32_t q=0;q<heads*4096;++q) output[q]=input[q];
 for(uint32_t h=0;h<heads;++h) {
  float* matrix=output+h*4096;
  for(uint32_t i=1;i<64;++i) {
   // A row update is logically simultaneous. Preserve every old coefficient
   // before writing even the first column/tile of the new row.
   float coefficients[64];
   for(uint32_t k=0;k<64;++k) coefficients[k]=matrix[i*64+k];
   uint32_t cascade_columns=i>=8 ? (i/32)*32 : (i/4)*4;
   for(uint32_t column_tile=0;column_tile<(i+31)/32;++column_tile) {
    for(uint32_t j=column_tile*32;j<i && j<(column_tile+1)*32;++j) {
     float correction;
     if(j<cascade_columns) {
      float completed=0.0f;
      uint32_t end=(i/16)*16;
      for(uint32_t start=0;start<end;start+=16) {
       float partial=0.0f;
       for(uint32_t k=start;k<start+16;++k)
        partial=rounded_add(partial,rounded_mul(coefficients[k],matrix[k*64+j]));
       completed=rounded_add(completed,partial);
      }
      float tail=0.0f;
      for(uint32_t k=end;k<i;++k)
       tail=rounded_add(tail,rounded_mul(coefficients[k],matrix[k*64+j]));
      correction=rounded_add(0.0f,rounded_add(tail,completed));
     } else {
      float p[4]={0.0f,0.0f,0.0f,0.0f};
      uint32_t end=(i/4)*4;
      for(uint32_t k=0;k<end;++k)
       p[k%4]=rounded_add(p[k%4],rounded_mul(coefficients[k],matrix[k*64+j]));
      for(uint32_t k=end;k<i;++k)
       p[0]=rounded_add(p[0],rounded_mul(coefficients[k],matrix[k*64+j]));
      for(uint32_t k=1;k<4;++k) p[0]=rounded_add(p[0],p[k]);
      correction=rounded_add(0.0f,p[0]);
     }
     matrix[i*64+j]=rounded_add(coefficients[j],correction);
    }
   }
   uint32_t slot=i==1?0:i==2?1:i==31?2:i==63?3:4;
   if(slot<4) for(uint32_t j=0;j<4096;++j) snapshots[(slot*heads+h)*4096+j]=matrix[j];
  }
 }
}
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
    return int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum())


def reference(initial):
    matrix = initial.clone()
    snapshots = {}
    for i in range(1, 64):
        row = matrix[..., i, :i].clone()
        product = row.unsqueeze(-1) * matrix[..., :i, :i]
        correction = canonical_outer_sum(product.contiguous())
        matrix[..., i, :i] = row + correction
        if i in ROWS:
            snapshots[i] = matrix.clone()
    return matrix, snapshots


def operation_inventory():
    """Count baseline Python tensor API calls, not profiler device kernels."""
    tree = ast.parse(BASE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Chunk64StateGraph")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "triangular_sum")
    counts = Counter()

    class Tensor:
        def __init__(self, shape):
            self.shape = shape

    class Adapter:
        float32, TILE_LAYOUT = "fp32", "tile"

        @staticmethod
        def zeros(shape, **kw):
            counts["zeros"] += 1
            return Tensor(shape)

        @staticmethod
        def slice(tensor, starts, ends, **kw):
            counts["slice"] += 1
            return Tensor(tuple(b - a for a, b in zip(starts, ends)))

        @staticmethod
        def concat(tensors, dim, **kw):
            counts["concat"] += 1
            shape = list(tensors[0].shape)
            shape[dim] = sum(t.shape[dim] for t in tensors)
            return Tensor(tuple(shape))

    class Graph:
        t, mesh, mem, sum_order = Adapter, None, None, "canonical"

        @staticmethod
        def add(a, b):
            counts["add"] += 1
            return Tensor(a.shape)

    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<baseline_sum_inventory>", "exec"), scope)
    graph = Graph()
    for i in range(1, 64):
        scope["triangular_sum"](graph, Tensor((1, 32, i, i)))
    helper = dict(counts)
    counts.update(add=63, multiply=63, transpose=63, slice=63 * 4 + 62, concat=126)
    return dict(
        triangular_sum_only=helper,
        full_63_row_update_calls=dict(counts),
        total_api_calls=sum(counts.values()),
        caveat="Python tensor API calls, not measured dispatches; some structural operations may be no-ops",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=E / "v6_triangular_recurrence_fusion_cpu.json")
    args = p.parse_args()
    assert not args.output.exists()
    assert sha(BASE) == BASE_SHA
    torch.set_num_threads(4)
    fixtures, evidence = [], {}
    for tag in ("zero", "nonzero"):
        meta_path = E / f"v6_chunk64_fused_corrected_output_{tag}_device.json"
        meta = json.loads(meta_path.read_text())
        assert meta["device_closed"] and meta["status"] == "completed_diagnostic"
        path = Path(meta["device_matrix_snapshot"])
        assert sha(path) == meta["device_matrix_snapshot_sha256"]
        cases = torch.load(path, map_location="cpu", weights_only=False)
        evidence[str(meta_path)] = sha(meta_path)
        evidence[str(path)] = sha(path)
        for length, case in cases.items():
            saved = case["intermediates"]
            prefixes = sorted(k.removesuffix("A_initial") for k in saved if k.endswith("/A_initial"))
            assert prefixes
            for prefix in prefixes:
                fixtures.append(
                    dict(
                        name=f"actual_{tag}_S{length}_{prefix.rstrip('/')}",
                        initial=saved[prefix + "A_initial"],
                        native={i: saved[prefix + f"A_row{i}"] for i in ROWS},
                        final=saved[prefix + "A_final"],
                    )
                )
    generator = torch.Generator().manual_seed(2026111302)
    for i in range(8):
        matrix = torch.randn(1, 32, 64, 64, generator=generator) * (2 ** (-7 + i % 3))
        if i >= 4:
            matrix *= torch.pow(2.0, torch.randint(-8, 5, matrix.shape, generator=generator))
        matrix = -matrix.masked_fill(torch.ones(64, 64, dtype=torch.bool).triu(), 0.0)
        fixtures.append(dict(name=f"random{i}", initial=matrix.contiguous()))
    fixtures.append(dict(name="signed_zeros", initial=torch.randn(1, 32, 64, 64, generator=generator) * 0.0))
    cpp_path = Path(str(args.output) + ".cpp.snapshot")
    cpp_path.write_text(CPP)
    result = dict(
        classification="CPU/source proof for exact ordered triangular recurrence fusion; no hardware",
        acceptance_eligible=False,
        oracle_assisted=False,
        status="cpu_proof_started",
        script_sha256=sha(__file__),
        base_sha256=BASE_SHA,
        cpp_sha256=sha(cpp_path),
        evidence=evidence,
        inventory=operation_inventory(),
        cases=[],
        numerical_contract="Each FP32 product rounds before addition; four-way and16-row cascade schedules unchanged; zero additions retained; original row coefficients preserved before writes",
        device_claim="none; device SFPU arithmetic, packing, streaming synchronization and complete chunk output/state still require separate exact gates",
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="triangular_cpu_", dir=E) as directory:
        library = Path(directory) / "triangular.so"
        command = [
            "g++",
            "-O2",
            "-std=c++17",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-shared",
            "-fPIC",
            "-x",
            "c++",
            str(cpp_path),
            "-o",
            str(library),
        ]
        subprocess.run(command, check=True, capture_output=True)
        dll = ctypes.CDLL(str(library))
        pointer = ctypes.POINTER(ctypes.c_float)
        dll.triangular.argtypes = [pointer, pointer, pointer, ctypes.c_uint32]
        result["compiler_command"] = command[:-1] + ["<temporary shared object>"]
        for fixture in fixtures:
            initial = fixture["initial"].contiguous()
            expected, rows = reference(initial)
            input_np = initial.numpy().reshape(32, 64, 64)
            output = np.empty_like(input_np)
            saved = np.empty((4, 32, 64, 64), dtype=np.float32)
            dll.triangular(
                input_np.ctypes.data_as(pointer), output.ctypes.data_as(pointer), saved.ctypes.data_as(pointer), 32
            )
            actual = torch.from_numpy(output).reshape_as(initial)
            snapshots = {i: torch.from_numpy(saved[index]).reshape_as(initial) for index, i in enumerate(ROWS)}
            row = dict(
                name=fixture["name"],
                cpp_vs_canonical_final=bits(expected, actual),
                cpp_vs_canonical_rows={i: bits(rows[i], snapshots[i]) for i in ROWS},
            )
            if "native" in fixture:
                row["cpp_vs_saved_native_rows"] = {i: bits(fixture["native"][i], snapshots[i]) for i in ROWS}
                row["cpp_plus_eye_vs_saved_native_final"] = bits(fixture["final"], actual + torch.eye(64))
            result["cases"].append(row)
            save()
            assert row["cpp_vs_canonical_final"] == 0 and all(
                v == 0 for v in row["cpp_vs_canonical_rows"].values()
            ), row
    result["status"] = "cpu_proof_pass"
    result["saved_native_exact"] = all(
        row.get("cpp_plus_eye_vs_saved_native_final", 0) == 0
        and all(v == 0 for v in row.get("cpp_vs_saved_native_rows", {}).values())
        for row in result["cases"]
    )
    save()
    assert "ttnn" not in sys.modules
    print(
        result["status"],
        "native_exact",
        result["saved_native_exact"],
        "baseline_API_calls",
        result["inventory"]["total_api_calls"],
    )


if __name__ == "__main__":
    main()
