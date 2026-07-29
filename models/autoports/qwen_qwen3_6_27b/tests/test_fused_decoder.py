# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect

from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder


def test_fused_decoder_is_a_distinct_runtime_path():
    assert FusedDecoder is not FunctionalDecoder
    assert "_mlp" in FusedDecoder.__dict__
    assert "_full_attention_prefill" in FusedDecoder.__dict__
    assert "_full_attention_decode" in FusedDecoder.__dict__
    assert "_linear_attention_prefill_chunk" in FusedDecoder.__dict__
    assert "_linear_attention_decode" in FusedDecoder.__dict__


def test_mlp_fuses_silu_into_multiply():
    source = inspect.getsource(FusedDecoder._mlp)
    tree = ast.parse(source.lstrip())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    assert "silu" not in attributes
    multiply = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "multiply")
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in multiply.keywords}
    assert keywords["input_tensor_a_activations"] == "[ttnn.UnaryOpType.SILU]"


def test_fused_runtime_methods_have_no_host_transfer():
    runtime = "\n".join(
        inspect.getsource(method)
        for method in (
            FusedDecoder._mlp,
            FusedDecoder._full_attention_prefill,
            FusedDecoder._full_attention_decode,
            FusedDecoder._linear_attention_prefill_chunk,
            FusedDecoder._linear_attention_decode,
            FusedDecoder.prefill_forward,
            FusedDecoder.decode_forward,
        )
    )
    for forbidden in ("torch.", "ttnn.from_torch", "ttnn.to_torch"):
        assert forbidden not in runtime


def test_shared_lhs_projections_are_packed():
    setup = inspect.getsource(FusedDecoder.from_state_dict)
    assert '"packed_qkv"' in setup
    assert '"packed_linear_inputs"' in setup
    full = inspect.getsource(FusedDecoder._full_attention_decode)
    linear = inspect.getsource(FusedDecoder._linear_attention_decode)
    assert full.count("ttnn.linear(") == 2  # packed input projection + output projection
    assert linear.count("ttnn.linear(") == 2
