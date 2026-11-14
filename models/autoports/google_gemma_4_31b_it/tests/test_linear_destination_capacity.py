# SPDX-License-Identifier: Apache-2.0
"""Exercise actual projection config selection without opening a device."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("fp32,bf16_weight", [(False, False), (True, True), (False, True), (True, False)])
@pytest.mark.parametrize("decoding,per_core_n", [(False, 8), (True, 8), (True, 6)])
def test_projection_subblock_respects_destination_capacity(fp32, bf16_weight, decoding, per_core_n):
    source = Path(__file__).parents[1] / "tt/optimized_decoder.py"
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_linear")
    configs = []
    kernel = SimpleNamespace(fp32_dest_acc_en=fp32)
    x = SimpleNamespace(shape=(1, 1, 64, 128))
    weight = SimpleNamespace(shape=(1, 1, 128, 32 * 8 * per_core_n), dtype="BF16" if bf16_weight else "BFP8")

    def make_config(**kwargs):
        configs.append(kwargs)
        return kwargs

    def matmul(*args, **kwargs):
        assert kwargs["compute_kernel_config"] is kernel
        config = kwargs.get("config", kwargs.get("program_config"))
        h = config.get("subblock_h", config.get("out_subblock_h"))
        w = config.get("subblock_w", config.get("out_subblock_w"))
        assert h * w <= (4 if fp32 else 8)
        if decoding:
            assert config["per_core_N"] % w == 0
        else:
            assert config["N_block_size"] == (8 if bf16_weight or fp32 else 16)
        return x

    ttnn = SimpleNamespace(
        to_memory_config=lambda tensor, config: tensor,
        L1_MEMORY_CONFIG="L1",
        DRAM_MEMORY_CONFIG="DRAM",
        CoreCoord=lambda x, y: (x, y),
        MinimalMatmulConfig=make_config,
        MatmulMultiCoreReuseMultiCastProgramConfig=make_config,
        experimental=SimpleNamespace(minimal_matmul=matmul),
        linear=matmul,
        bfloat8_b="BFP8",
        bfloat16="BF16",
    )
    env = {"ttnn": ttnn, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), env)
    decoder = SimpleNamespace(
        weights={"mlp.gate_proj.weight": weight},
        linear_kernels={"mlp": kernel},
        decoding=decoding,
        prefill_l1=False,
        mesh_device=SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=10)),
        output_padding=set(),
        banks=8,
    )
    assert env["_linear"](decoder, x, "mlp.gate_proj") is x
    assert len(configs) == 1
    selected = configs[0].get("subblock_w", configs[0].get("out_subblock_w"))
    assert selected == (2 if decoding and per_core_n == 6 else 4 if fp32 else 8)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("sliding", [False, True])
def test_decode_geometry_guard_uses_actual_weight_dtype(bf16, sliding):
    source = Path(__file__).parents[1] / "tt/multichip_decoder.py"
    tree = ast.parse(source.read_text())
    guard = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Tuple)
        and [getattr(t, "id", None) for t in n.target.elts] == ["name", "block"]
    )
    weights = {"self_attn.o_proj.weight": SimpleNamespace(dtype="BF16" if bf16 else "BFP4")}
    if not sliding:
        weights["self_attn.qk_proj.weight"] = SimpleNamespace(dtype="BF16" if bf16 else "BFP4")
    obj = SimpleNamespace(
        weights=weights,
        geometry={
            "self_attn.qk_proj": {"block": 21},
            "self_attn.o_proj": {"block": 8 if sliding else 16},
            "mlp.down_proj": {"block": 11},
        },
    )
    exec(
        compile(ast.Module(body=[guard], type_ignores=[]), str(source), "exec"),
        {"obj": obj, "ttnn": SimpleNamespace(bfloat16="BF16")},
    )
    assert obj.geometry["self_attn.qk_proj"]["block"] == (3 if bf16 and not sliding else 21)
    assert obj.geometry["self_attn.o_proj"]["block"] == (8 if bf16 or sliding else 16)
    assert obj.geometry["mlp.down_proj"]["block"] == 11
