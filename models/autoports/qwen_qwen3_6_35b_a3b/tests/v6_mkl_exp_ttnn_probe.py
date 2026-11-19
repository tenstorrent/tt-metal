# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic device transcription of the installed MKL normal-range exp loop.

The 128-entry table is mathematical, generated independently using Decimal at
setup. No model data or reference outputs enter the runtime helper. The current
contract is finite nonpositive inputs; values below -87.33654022216797 flush to
zero. This is deliberately not a general exp implementation or acceptance run.
"""

import argparse
import ast
import hashlib
import json
import os
import struct
import sys
import traceback
from decimal import Decimal, localcontext
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_state_probe import code_compare, sha

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
SAFE_ABS = 87.33654022216797
CONSTANT_BITS = dict(
    scale=0x4338AA3B,
    magic=0x4B400000,
    negative_magic=0xCB400000,
    table_mask=0x3F8,
    ln2_hi_by128=0x3BB18000,
    negative_ln2_lo_by128=0x35DE8083,
    half=0x3F000000,
    one_bits=0x3F800000,
    exponent_mask=0x7F800000,
    safe_abs=0x42AEAC4F,
)


def mathematical_table():
    """Generic 2**(j/128) split into nearest FP32 high and residual low."""

    def f32(value):
        return struct.unpack("f", struct.pack("f", float(value)))[0]

    rows = []
    with localcontext() as context:
        context.prec = 100
        log2 = Decimal(2).ln()
        for index in range(128):
            exact = (log2 * Decimal(index) / Decimal(128)).exp()
            high = f32(exact)
            low = f32(exact - Decimal.from_float(high))
            rows.append((high, low))
    return torch.tensor(rows, dtype=torch.float32)


class MKLNegativeExp:
    """TTNN-only runtime, with independent rounded multiply/add/sub operations.

    Tables are replicated at setup to avoid reshaping sub-tile logical indices
    or changing layouts inside the diagnostic path. Default table allocation is
    2 MiB total (two FP32 [1,32,64,128] tables), containing only 256 unique values.
    Input is FP32 [B,H,R,C], B<=1,H<=32,R<=64. Width C is not constrained by 128.
    No device exactness is claimed until this component has been measured.
    """

    def __init__(self, t, mesh, upload, memory_config, max_shape=(1, 32, 64)):
        self.t, self.mesh, self.mem = t, mesh, memory_config
        self.max_shape = tuple(max_shape)
        table = mathematical_table()
        self.high = upload(table[:, 0].reshape(1, 1, 1, 128).expand(*self.max_shape, 128).contiguous())
        self.low = upload(table[:, 1].reshape(1, 1, 1, 128).expand(*self.max_shape, 128).contiguous())
        self.one_bits = t.from_torch(
            torch.tensor([[[[0x3F800000]]]], dtype=torch.int64),
            dtype=t.uint32,
            layout=t.TILE_LAYOUT,
            device=mesh,
            memory_config=memory_config,
        )

    def __call__(self, x, observer=None):
        t, mem = self.t, self.mem
        shape = tuple(x.shape)
        assert len(shape) == 4 and x.dtype == t.float32
        assert all(got <= maximum for got, maximum in zip(shape[:3], self.max_shape))

        def add(a, b):
            return t.add(a, b, dtype=t.float32, memory_config=mem)

        def sub(a, b):
            return t.subtract(a, b, dtype=t.float32, memory_config=mem)

        def mul(a, b):
            return t.multiply(a, b, dtype=t.float32, fast_and_approximate_mode=False, memory_config=mem)

        safe = t.ge(x, -SAFE_ABS, memory_config=mem)
        bounded = t.where(safe, x, -SAFE_ABS, memory_config=mem)
        z = add(mul(bounded, 184.6649627685547), 12582912.0)
        raw = t.bitcast(z, t.uint32, memory_config=mem)
        index = t.bitwise_and(raw, 127, memory_config=mem)
        high = t.gather(self.high, dim=-1, index=index, memory_config=mem)
        low = t.gather(self.low, dim=-1, index=index, memory_config=mem)
        n = add(z, -12582912.0)
        r = sub(bounded, mul(n, 0.0054168701171875))
        r = add(r, mul(n, 1.6577690757912933e-06))
        polynomial = add(mul(mul(r, r), 0.5), r)
        reduced = add(add(mul(polynomial, high), low), high)
        shifted = t.bitwise_left_shift(raw, 16, memory_config=mem)
        exponent_bits = t.bitwise_and(
            t.add(shifted, self.one_bits, dtype=t.uint32, memory_config=mem), 0x7F800000, memory_config=mem
        )
        exponent = t.bitcast(exponent_bits, t.float32, memory_config=mem)
        value = mul(reduced, exponent)
        result = t.where(safe, value, 0.0, memory_config=mem)
        if observer:
            for name, tensor in (
                ("bounded", bounded),
                ("z", z),
                ("raw", raw),
                ("index", index),
                ("high", high),
                ("low", low),
                ("n", n),
                ("r", r),
                ("polynomial", polynomial),
                ("reduced", reduced),
                ("exponent_bits", exponent_bits),
                ("exponent", exponent),
                ("result", result),
            ):
                observer(name, tensor)
        return result


class CPUAdapter:
    """Exercise exactly the device helper graph without importing TTNN."""

    float32, uint32, TILE_LAYOUT = torch.float32, torch.int32, "tile"

    @staticmethod
    def from_torch(x, dtype, **kw):
        return x.to(dtype)

    @staticmethod
    def add(a, b, **kw):
        return a + b

    @staticmethod
    def subtract(a, b, **kw):
        return a - b

    @staticmethod
    def multiply(a, b, **kw):
        return a * b

    @staticmethod
    def ge(a, b, **kw):
        return a >= b

    @staticmethod
    def where(condition, a, b, **kw):
        return torch.where(condition, a, b)

    @staticmethod
    def bitcast(x, dtype, **kw):
        return x.contiguous().view(dtype)

    @staticmethod
    def bitwise_and(x, mask, **kw):
        return x & mask

    @staticmethod
    def bitwise_left_shift(x, amount, **kw):
        return x << amount

    @staticmethod
    def gather(table, dim, index, **kw):
        assert dim == -1
        table = table[tuple(slice(0, size) for size in index.shape[:-1]) + (slice(None),)]
        return table.gather(-1, index.long())


def audit():
    tree = ast.parse(Path(__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MKLNegativeExp")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    calls = [ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)]
    assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in calls)
    proof_path = E / "v6_mkl_exp_dispatch_cpu.json"
    proof = json.loads(proof_path.read_text())
    assert CONSTANT_BITS == proof["constants_bits"]
    table = mathematical_table()
    table_codes = table.view(torch.int32).reshape(-1).tolist()
    expected_codes = [code if code < 2**31 else code - 2**32 for code in proof["table_hi_lo_bits"]]
    assert table_codes == expected_codes
    return dict(
        acceptance_eligible=False,
        oracle_assisted=False,
        classification="Arithmetic component diagnostic; not decoder acceptance or performance evidence",
        helper_source=str(Path(__file__).resolve()),
        helper_sha256=sha(__file__),
        helper_runtime_ast_sha256=hashlib.sha256(ast.dump(method).encode()).hexdigest(),
        runtime_audit="No torch/from_torch/to_torch, tilize/untilize, reshard or host fallback; optional observer is diagnostic only",
        math_table_generation="Decimal precision100; hi=RN32(2**(j/128)), lo=RN32(exact-hi), j=0..127",
        math_table_unique_fp32_values=256,
        math_table_matches_installed_mkl_codes=256,
        math_table_raw_sha256=hashlib.sha256(table.numpy().tobytes()).hexdigest(),
        dispatch_evidence=str(proof_path),
        dispatch_evidence_sha256=sha(proof_path),
        dispatch_symbol=proof["selected_symbol"],
        constants_bits=CONSTANT_BITS,
        domain="Finite nonpositive FP32 inputs; x>=-87.33654022216797 uses normal loop, smaller values flush zero. Positive/NaN/Inf unsupported.",
        accuracy_claim="CPU exact on tested normal arguments; TTNN arithmetic and lookup require device measurement",
        table_device_bytes=2 * 1 * 32 * 64 * 128 * 4,
    )


def operands(payload):
    for length, case in payload["cases"].items():
        for name, cg in case["matrices"].items():
            if not name.endswith("/cumulative_g"):
                continue
            prefix = f"length{length}/{name.rsplit('/', 1)[0]}"
            yield prefix + "/cumulative", cg
            yield prefix + "/lower_difference", (cg - cg.transpose(-1, -2)).tril()
            yield prefix + "/last_difference", cg[..., -1:, :] - cg
            yield prefix + "/last", cg[..., -1:, :]
    generator = torch.Generator().manual_seed(20261113)
    yield "random_normal", -87 * torch.rand(1, 32, 64, 64, generator=generator)
    # Explicit boundaries, mixed table indices and underflow, with logical width1.
    boundary = torch.tensor([0.0, -0.0, -0.001, -0.005, -0.1, -1, -87, -SAFE_ABS, -88, -100, -1000])
    yield "boundary", boundary.reshape(1, 1, 11, 1).repeat(1, 32, 1, 1)


def cpu_check(payload):
    helper = MKLNegativeExp(CPUAdapter, None, lambda x: x.clone(), None)
    rows = []
    for name, x in operands(payload):
        assert torch.isfinite(x).all() and (x <= 0).all(), name
        actual = helper(x)
        safe = x >= -SAFE_ABS
        expected = x.exp()
        differences = int((actual[safe].view(torch.int32) != expected[safe].view(torch.int32)).sum())
        assert differences == 0, (name, differences)
        rows.append(
            dict(
                name=name,
                shape=list(x.shape),
                safe_elements=int(safe.sum()),
                rare_elements=int((~safe).sum()),
                normal_fp32_code_differences=differences,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-matrices", type=Path, default=E / "v6_chunk64_double_single_device.json.cpu_matrices.pt")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert not args.output.exists(), "Preserve previous evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.cpu_matrices, weights_only=False)
    result = audit()
    result.update(
        cpu_matrix_snapshot=str(args.cpu_matrices),
        cpu_matrix_snapshot_sha256=sha(args.cpu_matrices),
        cpu_checks=cpu_check(payload),
        status="cpu_setup_pass",
    )
    snapshot = Path(str(args.output) + ".helper.py.snapshot")
    snapshot.write_bytes(Path(__file__).read_bytes())
    result.update(helper_snapshot=str(snapshot), helper_snapshot_sha256=sha(snapshot))

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    record(args.output)
    save()
    if args.oracle_only:
        assert "ttnn" not in sys.modules
        print("CPU_MKL_EXP_HELPER_PASS", len(result["cpu_checks"]))
        return
    for name in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(name) in (None, "", "0"), name
    import ttnn

    mesh = None
    device_tensors = {}
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        result.update(status="running", device_closed=False, device_checks=[])

        def upload(x):
            return ttnn.from_torch(
                x.contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        helper = MKLNegativeExp(ttnn, mesh, upload, ttnn.DRAM_MEMORY_CONFIG)
        cpu_helper = MKLNegativeExp(CPUAdapter, None, lambda x: x.clone(), None)
        for name, x in operands(payload):
            collected = {}

            def observe(label, tensor):
                collected[label] = ttnn.to_torch(tensor).clone()

            uploaded = upload(x)
            actual = ttnn.to_torch(helper(uploaded, observer=observe)).clone()
            native = ttnn.to_torch(ttnn.exp(uploaded, fast_and_approximate_mode=False)).clone()
            safe = x >= -SAFE_ABS
            cpu_intermediates = {}
            expected = cpu_helper(x, observer=lambda label, value: cpu_intermediates.update({label: value.clone()}))
            result["device_checks"].append(
                dict(
                    name=name,
                    shape=list(x.shape),
                    helper_vs_cpu_transcription=code_compare(expected, actual),
                    helper_vs_torch_normal=code_compare(x.exp()[safe], actual[safe]),
                    native_vs_torch_normal=code_compare(x.exp()[safe], native[safe]),
                    intermediate_code_differences={
                        label: int(
                            (
                                cpu_intermediates[label].contiguous().view(torch.int32)
                                != value.contiguous().view(torch.int32)
                            ).sum()
                        )
                        for label, value in collected.items()
                    },
                    rare_elements=int((~safe).sum()),
                )
            )
            device_tensors[name] = dict(input=x, output=actual, native=native, intermediates=collected)
            save()
        target = Path(str(args.output) + ".device_tensors.pt")
        torch.save(device_tensors, target)
        result.update(
            status="completed_diagnostic", device_tensor_snapshot=str(target), device_tensor_snapshot_sha256=sha(target)
        )
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


if __name__ == "__main__":
    main()
