# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Source/capture-only recurrent reduction schedule audit; no TTNN import."""
import ast
import hashlib
import json
from pathlib import Path

import torch

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
BASE_SHA = "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fold(values):
    total = values[0].clone()
    for value in values[1:]:
        total = total + value
    return total


def compare(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
    return dict(
        bits_different=int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum()),
        elements=a.numel(),
        max_abs=float((a - b).abs().max()),
    )


def main():
    torch.set_num_threads(4)
    output = E / "v6_recurrent_fusion_contract_cpu.json"
    assert not output.exists() and sha(BASE) == BASE_SHA
    repo = Path(__file__).resolve().parents[4]
    sources = [
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.cpp",
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp",
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp",
        "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp",
        "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp",
        "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp",
    ]
    tree = ast.parse(BASE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    result = dict(
        status="cpu_saved_capture_audit",
        source_sha256=sha(__file__),
        device_executed=False,
        acceptance_eligible=False,
        base_sha256=BASE_SHA,
        native_sources={str(repo / p): sha(repo / p) for p in sources},
        method_ast_sha256={
            name: hashlib.sha256(ast.dump(methods[name]).encode()).hexdigest()
            for name in ("_recurrent_sum", "_recurrent_step")
        },
        cases=[],
        interpretation="CPU addition checks saved device product tensors only; no general SFPU/IEEE equivalence or new device fusion is asserted.",
    )
    for position in (3271, 227, 738, 5799):
        prefix = "v6_prefix_control_stream_probe" if position == 3271 else "v6_inline_precision_stream_probe"
        path = E / f"{prefix}.json.pos{position}.pt"
        capture = torch.load(path, weights_only=False, map_location="cpu")
        values = capture["tt_values"]
        row = dict(
            position=position,
            capture=str(path),
            capture_sha256=sha(path),
            round_state=values["round_state"],
            output_state_is_input=values["output_state_is_input"],
            inputs={
                name: dict(shape=list(values[name].shape), dtype=str(values[name].dtype))
                for name in ("delta_q", "delta_k", "delta_v", "delta_beta", "delta_decay", "delta_state_input")
            },
            reductions={},
        )
        for product, target in (("memory_products", "memory"), ("reduced_output_products", "reduced_output")):
            grouped = values[product].permute(0, 2, 1, 3).reshape(8, 16, 32, 128)
            partial = torch.stack([fold(group) for group in grouped])
            ordered = fold(partial).reshape(1, 32, 1, 128)
            sequential128 = fold(grouped.reshape(128, 32, 128)).reshape(1, 32, 1, 128)
            row["reductions"][target] = dict(
                ordered16_then8=compare(values[target], ordered), single128=compare(values[target], sequential128)
            )
        result["cases"].append(row)
    output.write_text(json.dumps(result, indent=2) + "\n")
    output.with_suffix(".source.py.snapshot").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({r["position"]: r["reductions"] for r in result["cases"]}, indent=2))


if __name__ == "__main__":
    main()
