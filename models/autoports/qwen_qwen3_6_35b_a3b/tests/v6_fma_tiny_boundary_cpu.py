# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only FTZ hypotheses for saved corrected-FMA matrix discrepancies."""

import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_state_probe import code_compare, sha
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_fma_sfpu_corrected_probe import Graph

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"


def source():
    graph = Graph().build()
    lines = []
    for slot, node in enumerate(graph.nodes):
        if slot < 3:
            continue
        kind, *ids = node
        if kind == "constant":
            expression = f"{float(ids[0]).hex()}f"
        elif kind in ("+", "-", "*"):
            expression = f"v[{ids[0]}] {kind} v[{ids[1]}]"
        elif kind == "sign":
            lines.append(f"v[{slot}]=0.0f;")
            for i in ids:
                lines.append(f"if(v[{i}]!=0) v[{slot}]=v[{i}]>0?1.0f:-1.0f;")
            continue
        elif kind == "neighbor":
            expression = f"neighbor(v[{ids[0]}],v[{ids[1]}])"
        elif kind == "select":
            r, neighbor, direction, cmp = ids
            expression = f"select(v[{r}],v[{neighbor}],v[{direction}],v[{cmp}],a,b,c)"
        else:
            raise AssertionError(kind)
        lines.append(f"v[{slot}]={expression};")
        if kind in ("+", "-", "*"):
            lines.append(f"if(mode&2) v[{slot}]=flush(v[{slot}]);")
    return (
        r"""
#include <cmath>
#include <cstdint>
#include <cstring>
static uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
static float value(uint32_t u){float x;std::memcpy(&x,&u,4);return x;}
static float flush(float x){uint32_t u=bits(x);return (u&0x7f800000u)==0?value(u&0x80000000u):x;}
static float neighbor(float r,float d){if(d==0)return r;if(r==0)return value(d>0?1:0x80000001u);return value(bits(r)+(((r>0)==(d>0))?1:-1));}
static float select(float r,float n,float d,float cs,float a,float b,float c){
 float cmp=d*cs;float v=(cmp>0 || (cmp==0 && d!=0 && (bits(r)&1)))?n:r;
 if(d==0 && r==0){bool neg=(a==0||b==0)&&c==0&&((bits(a)^bits(b))>>31)&&(bits(c)>>31);v=value(neg?0x80000000u:0u);}return v;
}
static float corrected(float a,float b,float c,int mode){
 float v[NODES];v[0]=a;v[1]=b;v[2]=c;
 GRAPH
 return v[RESULT];
}
extern "C" void run(const float* aa,const float* bb,float* out,int count,int kmax,int mode){
 for(int i=0;i<count;++i){float acc=0.0f;
  for(int k=0;k<kmax;++k){float a=aa[i*kmax+k],b=bb[i*kmax+k];
   if(mode&1){a=flush(a);b=flush(b);}
   if(mode&4){acc=std::fma(a,b,acc);if(mode&2)acc=flush(acc);}
   else acc=corrected(a,b,acc,mode);
  }out[i]=acc;
 }
}
""".replace(
            "NODES", str(len(graph.nodes))
        )
        .replace("GRAPH", "\n".join(lines))
        .replace("RESULT", str(graph.result))
    )


def main():
    torch.set_num_threads(4)
    fixture_path = E / "v6_corrected_fma_matrices_device.fixtures.pt"
    device_path = E / "v6_corrected_fma_matrices_device.device.pt"
    meta_path = E / "v6_corrected_fma_matrices_device.json"
    fixtures = torch.load(fixture_path, weights_only=False)
    measured = torch.load(device_path, weights_only=False)
    metadata = json.loads(meta_path.read_text())
    assert sha(fixture_path) == metadata["fixtures_sha256"]
    cpp = source()
    cpp_path = E / "v6_fma_tiny_boundary_cpu.cpp.snapshot"
    cpp_path.write_text(cpp)
    result = dict(
        classification="CPU arithmetic hypotheses on saved device tensors; no hardware or acceptance claim",
        script_sha256=sha(__file__),
        fixture_sha256=sha(fixture_path),
        device_tensor_sha256=sha(device_path),
        device_metadata_sha256=sha(meta_path),
        graph_source_sha256=sha(Path(__file__).with_name("v6_fma_sfpu_corrected_probe.py")),
        cpp_sha256=sha(cpp_path),
        cases=[],
    )
    modes = {
        0: "graph_ieee",
        1: "graph_input_ftz",
        2: "graph_arithmetic_ftz",
        3: "graph_input_and_arithmetic_ftz",
        4: "true_fma",
        5: "true_fma_input_ftz",
        6: "true_fma_output_ftz",
        7: "true_fma_input_output_ftz",
    }
    with tempfile.TemporaryDirectory(prefix="v6_fma_tiny_") as temporary:
        library = Path(temporary) / "proof.so"
        command = [
            "g++",
            "-shared",
            "-fPIC",
            "-O2",
            "-std=c++17",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-x",
            "c++",
            str(cpp_path),
            "-o",
            str(library),
        ]
        subprocess.run(command, check=True)
        result["compiler"] = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
        result["compiler_flags"] = command[1:8]
        lib = ctypes.CDLL(str(library))
        pointer = ctypes.c_void_p
        lib.run.argtypes = [pointer, pointer, pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        for fixture in fixtures:
            name, expected = fixture["name"], fixture["expected"]
            actual = measured[name]
            wrong = expected.contiguous().view(torch.int32) != actual.contiguous().view(torch.int32)
            if not wrong.any():
                result["cases"].append(dict(name=name, device_exact=True))
                continue
            selected = wrong.clone()
            correct_indices = torch.nonzero(~wrong.reshape(-1)).flatten()[:: max(1, int((~wrong).sum()) // 512)][:512]
            selected.reshape(-1)[correct_indices] = True
            indices = selected.nonzero(as_tuple=True)
            batch, head, row, column = indices
            aa = fixture["a"][batch, head, row, :].contiguous()
            bb = fixture["b"].transpose(-1, -2)[batch, head, column, :].contiguous()
            want, device = expected[indices].contiguous(), actual[indices].contiguous()
            row_result = dict(
                name=name,
                selected=int(selected.sum()),
                device_mismatch_count=int(wrong.sum()),
                matched_controls=len(correct_indices),
                input_subnormal_counts={
                    key: int(((fixture[key].abs() < torch.finfo(torch.float32).tiny) & (fixture[key] != 0)).sum())
                    for key in ("a", "b")
                },
                predictions=[],
            )
            difference = (expected - actual).abs()
            flat_index = int(difference.argmax())
            row_result["largest_error"] = dict(
                expected=float(expected.reshape(-1)[flat_index]),
                actual=float(actual.reshape(-1)[flat_index]),
                max_abs=float(difference.max()),
                expected_bits=hex(int(expected.reshape(-1)[flat_index].view(torch.int32)) & 0xFFFFFFFF),
                actual_bits=hex(int(actual.reshape(-1)[flat_index].view(torch.int32)) & 0xFFFFFFFF),
            )
            for mode, label in modes.items():
                out = torch.empty(aa.shape[0], dtype=torch.float32)
                lib.run(aa.data_ptr(), bb.data_ptr(), out.data_ptr(), aa.shape[0], aa.shape[1], mode)
                matches_device = out.view(torch.int32) == device.view(torch.int32)
                wrong_selected = wrong[indices]
                row_result["predictions"].append(
                    dict(
                        mode=label,
                        vs_torch=code_compare(want, out),
                        vs_device=code_compare(device, out),
                        explained_device_mismatches=int((matches_device & wrong_selected).sum()),
                        correct_controls_preserved=int((matches_device & ~wrong_selected).sum()),
                    )
                )
            result["cases"].append(row_result)
            print(
                name,
                [
                    (r["mode"], r["explained_device_mismatches"], r["vs_device"]["max_abs"])
                    for r in row_result["predictions"]
                ],
                flush=True,
            )

    saved = torch.load(E / "v6_chunk64_sfpu_fma_compensated_device_v2.json.cpu_matrices.pt", weights_only=False)
    case = saved["cases"][128]
    mat = case["matrices"]
    controls = []
    for policy in ("transformed_k", "state_contribution", "both"):
        state = saved["initial"].clone()
        for chunk in range(2):
            get = lambda name: mat[f"chunk{chunk}/{name}"]
            transformed_k = (
                measured[f"chunk{chunk}_transformed_k"] if policy in ("transformed_k", "both") else get("transformed_k")
            )
            contribution = transformed_k @ state
            if policy in ("state_contribution", "both"):
                contribution = measured[f"chunk{chunk}_state_contribution"]
            new = get("transformed_v") - contribution
            state = state * get("cumulative_g")[..., -1:, :].exp() + get("weighted_k").transpose(-1, -2) @ new
        controls.append(dict(replaced=policy, final=code_compare(case["canonical"], state)))
    result["downstream_explicit_fixture_interventions"] = controls
    result[
        "intervention_limits"
    ] = "State-contribution replacement uses the captured independent canonical-state operand fixture; both is not a compositional transformed-K intervention. All are diagnostic only. No recurrence or acceptance waiver."
    full_graph_path = E / "v6_chunk64_corrected_fma_output_device.json"
    if full_graph_path.exists():
        full_graph = json.loads(full_graph_path.read_text())
        result["parent_device_full_graph_evidence"] = dict(
            path=str(full_graph_path),
            sha256=sha(full_graph_path),
            status=full_graph["status"],
            device_closed=full_graph["device_closed"],
        )
    assert "ttnn" not in sys.modules, "This analysis must never import TTNN"
    result["ttnn_imported"] = False
    output = E / "v6_fma_tiny_boundary_cpu.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(controls, indent=2))


if __name__ == "__main__":
    main()
