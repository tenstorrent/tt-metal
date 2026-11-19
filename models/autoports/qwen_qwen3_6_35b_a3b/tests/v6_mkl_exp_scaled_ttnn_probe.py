# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Separate MKL exp candidate avoiding the observed tiny-float bitcast defect.

The original measured helper is immutable. Shift the power-of-two scale up 16
exponents before bitcast and exactly scale the order-one reduced polynomial
down by 2**16. The finite nonpositive domain and rare flush policy are unchanged.
"""

import ast
import copy
import hashlib
import inspect
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_mkl_exp_ttnn_probe as original
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

EXPONENT_SHIFT = 16
ORIGINAL_HELPER_SHA256 = "8b262be19cf72ca69d553a65de656398a3efa8388f5e3ed45e9a1df5113787fa"


class MKLNegativeExp(original.MKLNegativeExp):
    def __init__(self, t, mesh, upload, memory_config, max_shape=(1, 32, 64)):
        super().__init__(t, mesh, upload, memory_config, max_shape)
        self.exponent_bias = t.from_torch(
            torch.tensor([[[[EXPONENT_SHIFT << 23]]]], dtype=torch.int64),
            dtype=t.uint32,
            layout=t.TILE_LAYOUT,
            device=mesh,
            memory_config=memory_config,
        )


def attach_call():
    assert verify_source(original.__file__, ORIGINAL_HELPER_SHA256)
    tree = ast.parse(Path(original.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MKLNegativeExp")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    before = copy.deepcopy(method)
    changed = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == "exponent_bits":
            old = node.value
            replacement = ast.parse(
                "t.add(ORIGINAL, self.exponent_bias, dtype=t.uint32, memory_config=mem)", mode="eval"
            ).body
            replacement.args[0] = copy.deepcopy(old)
            changed.append((node, old))
            node.value = replacement
        elif name == "value":
            assert ast.unparse(node.value) == "mul(reduced, exponent)"
            changed.append((node, node.value))
            node.value = ast.parse("mul(mul(reduced, 0.0000152587890625), exponent)", mode="eval").body
    assert len(changed) == 2
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    runtime_hash = hashlib.sha256(ast.dump(method).encode()).hexdigest()
    calls = [ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)]
    assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in calls)
    scope = dict(vars(original))
    exec(compile(module, str(Path(__file__).resolve()) + ":scaled_exp", "exec"), scope)
    MKLNegativeExp.__call__ = scope["__call__"]
    for node, old in changed:
        node.value = old
    assert ast.dump(method) == ast.dump(before)
    return dict(
        changed_assignments=["exponent_bits", "value"], original_ast_restored=True, runtime_ast_sha256=runtime_hash
    )


PATCH_PROOF = attach_call()


def audit():
    result = original.audit()
    result.update(
        helper_source=str(Path(__file__).resolve()),
        helper_sha256=original.sha(__file__),
        original_helper_source=str(Path(original.__file__).resolve()),
        original_helper_sha256=raw_sha256(original.__file__),
        original_helper_historical_sha256=ORIGINAL_HELPER_SHA256,
        scaled_exp_patch_proof=PATCH_PROOF,
        exponent_shift=EXPONENT_SHIFT,
        exponent_bias_bits=EXPONENT_SHIFT << 23,
        reduced_scale=2.0**-EXPONENT_SHIFT,
        correction_reason="Measured integer-to-FP32 bitcast changed normal scale codes 0x00800000..0x05800000. A one-exponent shift cannot avoid that range.",
        arithmetic_proof="For finite nonpositive inputs, original exponent range is -126..0; shifted scale is -110..16, reduced is order one scaled by2^-16. Both scaling steps are exact binary powers; final product and rounding remain identical on CPU.",
        original_device_evidence=str(original.E / "v6_mkl_exp_ttnn_device.json"),
        original_device_evidence_sha256=original.sha(original.E / "v6_mkl_exp_ttnn_device.json"),
        device_claim="Untested candidate; explicit device measurement required",
    )
    result["helper_runtime_ast_sha256"] = PATCH_PROOF["runtime_ast_sha256"]
    return result


def cpu_check(payload):
    helper = MKLNegativeExp(original.CPUAdapter, None, lambda x: x.clone(), None)
    unscaled = original.MKLNegativeExp(original.CPUAdapter, None, lambda x: x.clone(), None)
    rows = []
    for name, x in original.operands(payload):
        assert torch.isfinite(x).all() and (x <= 0).all(), name
        actual = helper(x)
        previous = unscaled(x)
        safe = x >= -original.SAFE_ABS
        expected = x.exp()
        differences = int((actual[safe].view(torch.int32) != expected[safe].view(torch.int32)).sum())
        delta = int((actual.contiguous().view(torch.int32) != previous.contiguous().view(torch.int32)).sum())
        assert differences == delta == 0, (name, differences, delta)
        rows.append(
            dict(
                name=name,
                shape=list(x.shape),
                safe_elements=int(safe.sum()),
                rare_elements=int((~safe).sum()),
                normal_fp32_code_differences=differences,
                all_codes_different_from_original=delta,
            )
        )
    # Every rounded scale exponent and table entry, with normal arguments only.
    lattice = (
        torch.arange(-16000, 1, dtype=torch.float32) * (torch.log(torch.tensor(2.0, dtype=torch.float64)) / 128).float()
    )
    for offset in (0.0, -0.00001, 0.00001):
        x = (lattice + offset).clamp(max=0).reshape(1, 1, 1, -1)
        actual, expected = helper(x), unscaled(x)
        delta = int((actual.contiguous().view(torch.int32) != expected.contiguous().view(torch.int32)).sum())
        assert delta == 0
        rows.append(
            dict(
                name=f"scale_table_lattice_{offset}",
                safe_elements=x.numel(),
                rare_elements=0,
                all_codes_different_from_original=delta,
            )
        )
    return rows


def main():
    # Reuse the measured component harness verbatim, replacing only helper/audit
    # bindings. It records this module's source snapshot and the original hash.
    source = inspect.getsource(original.main)
    scope = dict(vars(original), __file__=__file__, MKLNegativeExp=MKLNegativeExp, audit=audit, cpu_check=cpu_check)
    exec(compile(source, str(Path(__file__).resolve()) + ":unchanged_component_harness", "exec"), scope)
    scope["main"]()


if __name__ == "__main__":
    main()
