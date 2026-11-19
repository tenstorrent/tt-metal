# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic HF chunk64 final-state graph, using identical canonical operands.

--oracle-only performs CPU setup without importing TTNN. Device runs are owned
by the parent. This component is not a decoder, acceptance test, or timing run.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    l2norm,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from models.autoports.qwen_qwen3_6_35b_a3b.tests import reference
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_prefill_state_hypothesis import compare

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
ROW_SNAPSHOTS = (1, 2, 31, 63)
REPO = Path(__file__).resolve().parents[4]
ADDCMUL_SOURCES = (
    "ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.cpp",
    "ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.cpp",
    "ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h",
    "tests/ttnn/unit_tests/operations/eltwise/test_ternary_category1_float32.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def code_compare(expected, actual):
    result = compare(expected, actual)
    if not torch.isfinite(torch.tensor(result["relative_l2"])):
        result["relative_l2"] = 0.0 if torch.equal(expected, actual) else None
    if not torch.isfinite(torch.tensor(result["pcc"])):
        result["pcc"] = None
    result["fp32_code_differences"] = int(
        (expected.float().contiguous().view(torch.int32) != actual.float().contiguous().view(torch.int32)).sum()
    )
    result["bf16_code_differences"] = int(
        (expected.bfloat16().contiguous().view(torch.int16) != actual.bfloat16().contiguous().view(torch.int16)).sum()
    )
    return result


def cpu_chunk64(key, value, g, beta, initial):
    """State-only transcription; inputs are normalized FP32 [B,H,S,D]."""
    n = key.shape[-2]
    pad = (-n) % 64
    key, value, g, beta = [torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (key, value, g, beta)]
    lower = torch.ones(64, 64, dtype=torch.bool).tril()
    strict = lower.tril(-1)
    eye = torch.eye(64)
    state = initial.clone()
    saved = {}
    for start in range(0, n, 64):
        prefix = f"chunk{start // 64}/"
        k, v, gg, bb = [t[..., start : start + 64, :] for t in (key, value, g, beta)]
        kb, vb = k * bb, v * bb
        cumulative = gg.cumsum(-2)
        differences = cumulative - cumulative.transpose(-1, -2)
        decay = differences.tril().exp().tril()
        a = -((kb @ k.transpose(-1, -2)) * decay).masked_fill(~strict, 0)
        for name, tensor in (
            ("k_beta", kb),
            ("v_beta", vb),
            ("cumulative_g", cumulative),
            ("decay_mask", decay),
            ("A_initial", a),
        ):
            saved[prefix + name] = tensor.clone()
        for i in range(1, 64):
            row, sub = a[..., i, :i].clone(), a[..., :i, :i].clone()
            a[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            if i in ROW_SNAPSHOTS:
                saved[prefix + f"A_row{i}"] = a.clone()
        a = a + eye
        transformed_v = a @ vb
        transformed_k = a @ (kb * cumulative.exp())
        v_new = transformed_v - transformed_k @ state
        last = cumulative[..., -1:, :]
        weighted_k = k * (last - cumulative).exp()
        state = state * last.exp() + weighted_k.transpose(-1, -2) @ v_new
        for name, tensor in (
            ("A_final", a),
            ("transformed_v", transformed_v),
            ("transformed_k", transformed_k),
            ("v_new", v_new),
            ("weighted_k", weighted_k),
            ("state", state),
        ):
            saved[prefix + name] = tensor.clone()
    return state, saved


class Chunk64StateGraph:
    """Device-only state graph after setup, optionally instrumented by observer.

    Native mode uses HiFi4 FP32-destination matmul: full FP32 operand precision
    is NOT claimed. sfpu_reduce mode forms FP32 products explicitly and reduces
    them; reduction order still need not equal Torch CPU matrix multiplication.
    sfpu_fma is a sequential addcmul control. Despite its SFPMAD instruction,
    the native full-mantissa tests document that it is not IEEE single-rounding
    FP32 FMA. Its CLI name identifies the proposed candidate, not an exactness claim.
    """

    def __init__(self, ttnn, mesh, upload, variant="native", exp_approx=False, cumsum="plain"):
        self.t = ttnn
        self.mesh = mesh
        self.variant = variant
        self.exp_approx = exp_approx
        assert cumsum in ("plain", "compensated")
        self.cumsum = cumsum
        self.mem = ttnn.DRAM_MEMORY_CONFIG
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.lower = upload(torch.ones(1, 1, 64, 64).tril())
        self.strict = upload(torch.ones(1, 1, 64, 64).tril(-1))
        self.eye = upload(torch.eye(64).reshape(1, 1, 64, 64))

    def add(self, a, b):
        return self.t.add(a, b, dtype=self.t.float32, memory_config=self.mem)

    def sub(self, a, b):
        return self.t.subtract(a, b, dtype=self.t.float32, memory_config=self.mem)

    def mul(self, a, b):
        return self.t.multiply(a, b, dtype=self.t.float32, fast_and_approximate_mode=False, memory_config=self.mem)

    def exp(self, a):
        return self.t.exp(a, fast_and_approximate_mode=self.exp_approx, memory_config=self.mem)

    def mm(self, a, b):
        t = self.t
        if self.variant == "native":
            return t.matmul(a, b, dtype=t.float32, compute_kernel_config=self.compute, memory_config=self.mem)
        if self.variant == "sfpu_fma":
            assert a.shape[-1] == b.shape[-2] and tuple(a.shape)[:-2] == tuple(b.shape)[:-2]
            shape = (*tuple(a.shape)[:-2], a.shape[-2], b.shape[-1])
            result = t.zeros(shape, dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.mem)
            for k in range(a.shape[-1]):
                # FP32 full(M,N), col(M,1), row(1,N) dispatches native ROW_COL_BCAST.
                # No invalid-broadcast or block-float composite fallback applies.
                result = t.addcmul(
                    result, a[..., :, k : k + 1], b[..., k : k + 1, :], value=1.0, memory_config=self.mem
                )
            return result
        assert self.variant == "sfpu_reduce"
        rows = []
        for i in range(a.shape[-2]):
            coeff = t.transpose(a[..., i : i + 1, :], -1, -2)
            product = self.mul(coeff, b)
            rows.append(
                t.sum(product, dim=-2, keepdim=True, compute_kernel_config=self.compute, memory_config=self.mem)
            )
        return t.concat(rows, dim=-2, memory_config=self.mem)

    def __call__(self, key, value, g, beta, initial, observer=None):
        """Support every positive logical S, padding only its final chunk."""
        t = self.t
        n = key.shape[-2]
        assert n > 0 and all(x.shape[-2] == n for x in (value, g, beta))
        pad = (-n) % 64
        if pad:
            key, value, g, beta = [
                t.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0, memory_config=self.mem) for x in (key, value, g, beta)
            ]
        state = initial
        for start in range(0, n, 64):
            prefix = f"chunk{start // 64}/"
            k, v, gg, bb = [x[..., start : start + 64, :] for x in (key, value, g, beta)]
            kb, vb = self.mul(k, bb), self.mul(v, bb)
            cumulative = t.cumsum(
                gg, -2, dtype=t.float32, disable_compensated_sum=self.cumsum == "plain", memory_config=self.mem
            )
            differences = self.sub(cumulative, t.transpose(cumulative, -1, -2))
            masked_diff = t.where(self.lower, differences, 0.0, memory_config=self.mem)
            decay = self.mul(self.exp(masked_diff), self.lower)
            a = t.where(
                self.strict,
                self.mul(self.mm(kb, t.transpose(k, -1, -2)), decay),
                0.0,
                memory_config=self.mem,
            )
            a = self.mul(a, -1.0)
            if observer:
                for name, tensor in (
                    ("k_beta", kb),
                    ("v_beta", vb),
                    ("cumulative_g", cumulative),
                    ("decay_mask", decay),
                    ("A_initial", a),
                ):
                    observer(prefix + name, tensor)
            for i in range(1, 64):
                row, sub = a[..., i : i + 1, :i], a[..., :i, :i]
                product = self.mul(t.transpose(row, -1, -2), sub)
                correction = t.sum(
                    product, dim=-2, keepdim=True, compute_kernel_config=self.compute, memory_config=self.mem
                )
                updated = self.add(row, correction)
                full_row = t.concat([updated, a[..., i : i + 1, i:]], dim=-1, memory_config=self.mem)
                parts = [a[..., :i, :], full_row]
                if i < 63:
                    parts.append(a[..., i + 1 :, :])
                a = t.concat(parts, dim=-2, memory_config=self.mem)
                if observer and i in ROW_SNAPSHOTS:
                    observer(prefix + f"A_row{i}", a)
            a = self.add(a, self.eye)
            transformed_v = self.mm(a, vb)
            transformed_k = self.mm(a, self.mul(kb, self.exp(cumulative)))
            v_new = self.sub(transformed_v, self.mm(transformed_k, state))
            last = cumulative[..., 63:64, :]
            weighted_k = self.mul(k, self.exp(self.sub(last, cumulative)))
            state = self.add(self.mul(state, self.exp(last)), self.mm(t.transpose(weighted_k, -1, -2), v_new))
            if observer:
                for name, tensor in (
                    ("A_final", a),
                    ("transformed_v", transformed_v),
                    ("transformed_k", transformed_k),
                    ("v_new", v_new),
                    ("weighted_k", weighted_k),
                    ("state", state),
                ):
                    observer(prefix + name, tensor)
        return state


def cpu_setup(args):
    meta_path = E / "v5_combined_layer34_stream_probe.json"
    meta = json.loads(meta_path.read_text())
    oracle_path = Path(meta["oracle"])
    oracle = torch.load(oracle_path, weights_only=False)
    cap_path = E / "v5_combined_layer34_stream_probe.json.pos128.pt"
    capture = torch.load(cap_path, weights_only=False)
    hf = reference.layer(reference.weights(34), 34)
    cache = DynamicCache(config=reference.config())
    saved = {}
    original = hf.linear_attn.chunk_gated_delta_rule

    def intercept(q, k, v, **kw):
        saved.update(q=q.clone(), k=k.clone(), v=v.clone(), kw=kw.copy())
        result = original(q, k, v, **kw)
        saved["state"] = result[1].clone()
        return result

    hf.linear_attn.chunk_gated_delta_rule = intercept
    with torch.no_grad():
        hf.linear_attn(hf.input_layernorm(oracle["prefix"]), cache_params=cache)
    assert torch.equal(saved["state"].bfloat16(), capture["hf_values"]["delta_state_input"])
    raw_q, raw_k, raw_v = saved["q"], saved["k"], saved["v"]
    kw = saved["kw"]
    normalized_q = l2norm(raw_q, dim=-1, eps=1e-6) if kw["use_qk_l2norm_in_kernel"] else raw_q
    normalized_k = l2norm(raw_k, dim=-1, eps=1e-6) if kw["use_qk_l2norm_in_kernel"] else raw_k
    initial = kw["initial_state"]
    if initial is None:
        initial = torch.zeros(raw_k.shape[0], raw_k.shape[2], raw_k.shape[-1], raw_v.shape[-1])
    if args.initial_state == "prefix64":
        _, initial = torch_chunk_gated_delta_rule(
            normalized_q[:, :64],
            normalized_k[:, :64],
            raw_v[:, :64],
            g=kw["g"][:, :64],
            beta=kw["beta"][:, :64],
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
    payload = dict(
        cases={}, initial=initial.clone(), raw_operands=dict(q=raw_q, k=raw_k, v=raw_v, g=kw["g"], beta=kw["beta"])
    )
    result = dict(
        classification="same-operand component diagnostic only",
        acceptance_eligible=False,
        script_sha256=sha(__file__),
        hf_source_sha256=sha(inspect.getfile(original)),
        oracle=str(oracle_path),
        oracle_sha256=sha(oracle_path),
        prefix_sha256=tensor_sha(oracle["prefix"]),
        capture=str(cap_path),
        capture_sha256=sha(cap_path),
        frozen_hf_prefix_cache_reproduced=True,
        initial_state=args.initial_state,
        operand_policy="Canonical HF normalized BF16 key/value widened to FP32; actual beta and log decay; setup-only host extraction",
        normalized_operand_sha256={"key": tensor_sha(normalized_k), "query": tensor_sha(normalized_q)},
        cases=[],
        device_closed=None,
        timing="not measured",
        variant=args.variant,
        cumsum=args.cumsum,
        cumsum_precision_claim="CPU HF cumsum uses a wider accumulator; plain/compensated device variants are measured independently, neither is assumed exact",
        precision_claim="Native FP32 matmul may narrow operands; sfpu_reduce uses explicit FP32 products but a different reduction order; sfpu_fma is sequential native addcmul, which local full-mantissa tests explicitly do not claim is IEEE single-rounding FMA",
        addcmul_source_contract={
            "source_sha256": {name: sha(REPO / name) for name in ADDCMUL_SOURCES},
            "dispatch": "Same-leading-dimension FP32 full(M,N),col(M,1),row(1,N) selects native ROW_COL_BCAST; no invalid-broadcast/block-float fallback",
            "native_fp32_unpack": "ternary_program_factory enables FP32 dest and UnpackToDest for every FP32 input",
            "single_rounding_claim": False,
            "numerical_limitation": "test_ternary_category1_float32.py:54-69 explicitly describes intermediate product rounding for full-mantissa addcmul",
        },
    )
    for n in args.lengths:
        assert 1 <= n <= raw_q.shape[1]
        q, k, v = normalized_q[:, :n], normalized_k[:, :n], raw_v[:, :n]
        g, beta = kw["g"][:, :n], kw["beta"][:, :n]
        with torch.no_grad():
            _, canonical = torch_chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta, initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=False
            )
            _, sequential = torch_recurrent_gated_delta_rule(
                q, k, v, g=g, beta=beta, initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=False
            )
            operands = dict(
                key=k.transpose(1, 2).contiguous().float(),
                value=v.transpose(1, 2).contiguous().float(),
                g=g.transpose(1, 2).contiguous().float().unsqueeze(-1),
                beta=beta.transpose(1, 2).contiguous().float().unsqueeze(-1),
            )
            state, matrices = cpu_chunk64(**operands, initial=initial)
        assert torch.equal(state, canonical), "CPU state-only transcription must reproduce HF FP32 state exactly"
        payload["cases"][n] = dict(operands=operands, canonical=canonical, sequential=sequential, matrices=matrices)
        result["cases"].append(
            dict(
                length=n,
                internal_padding=(-n) % 64,
                cpu_transcription_vs_hf=code_compare(canonical, state),
                hf_chunk64_vs_sequential=code_compare(canonical, sequential),
                device_intermediates={},
            )
        )
    tree = ast.parse(Path(__file__).read_text())
    graph = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Chunk64StateGraph")
    runtime = [node for node in graph.body if isinstance(node, ast.FunctionDef) and node.name != "__init__"]
    forbidden = []
    for method in runtime:
        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                if call.startswith("torch.") or call.endswith((".from_torch", ".to_torch")):
                    forbidden.append(call)
    assert not forbidden
    result["runtime_ast_audit"] = {
        "torch_or_host_conversion_calls": forbidden,
        "observer_policy": "Optional diagnostic callback; omit for device-only execution",
        "positive_logical_length_required": True,
        "alignment_restriction": False,
    }
    return result, payload


def run(args):
    torch.set_num_threads(4)
    assert not args.output.exists(), "Preserve previous evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result, payload = cpu_setup(args)
    result["status"] = "cpu_setup_pass"
    matrices_path = Path(str(args.output) + ".cpu_matrices.pt")
    torch.save(payload, matrices_path)
    result.update(cpu_matrix_snapshot=str(matrices_path), cpu_matrix_snapshot_sha256=sha(matrices_path))
    source = Path(str(args.output) + ".source.py.snapshot")
    source.write_bytes(Path(__file__).read_bytes())
    result.update(source_snapshot=str(source), source_snapshot_sha256=sha(source))

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    record(args.output)
    save()
    if args.oracle_only:
        assert "ttnn" not in sys.modules
        print("CPU_CHUNK64_SETUP_PASS", args.lengths)
        return
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(key) in (None, "", "0"), key
    import ttnn

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        result.update(status="running", device_closed=False)

        def upload(x):
            return ttnn.from_torch(
                x.contiguous(),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        graph = Chunk64StateGraph(ttnn, mesh, upload, variant=args.variant, cumsum=args.cumsum)
        device_payload = {}
        for row in result["cases"]:
            n = row["length"]
            item = payload["cases"][n]
            outputs = {}

            def observe(name, tensor):
                actual = ttnn.to_torch(tensor).clone()
                assert torch.isfinite(actual).all(), name
                outputs[name] = actual
                row["device_intermediates"][name] = code_compare(item["matrices"][name], actual)
                save()

            inputs = {key: upload(value) for key, value in item["operands"].items()}
            got = ttnn.to_torch(graph(**inputs, initial=upload(payload["initial"]), observer=observe)).clone()
            assert torch.isfinite(got).all()
            row["device_vs_hf_chunk64"] = code_compare(item["canonical"], got)
            row["device_vs_cpu_sequential"] = code_compare(item["sequential"], got)
            device_payload[n] = dict(final_state=got, intermediates=outputs)
            save()
        output_tensors = Path(str(args.output) + ".device_matrices.pt")
        torch.save(device_payload, output_tensors)
        result.update(
            status="completed_diagnostic",
            device_matrix_snapshot=str(output_tensors),
            device_matrix_snapshot_sha256=sha(output_tensors),
        )
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 127, 65])
    p.add_argument("--initial-state", choices=["zero", "prefix64"], default="zero")
    p.add_argument("--variant", choices=["native", "sfpu_reduce", "sfpu_fma"], default="native")
    p.add_argument("--cumsum", choices=["plain", "compensated"], default="plain")
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
