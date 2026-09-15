"""model_map (ast skeleton) — deterministic, no key/hardware."""

from agent.model_map import build_model_map, render_skeleton

SRC = """\
import ttnn

class Attn:
    def forward(self, x):
        qkv = ttnn.linear(x, self.w)
        n = ttnn.layer_norm(qkv)
        return n

def helper(a, b):
    return ttnn.matmul(a, b)
"""


def _map(tmp_path):
    f = tmp_path / "attn.py"
    f.write_text(SRC)
    return build_model_map([f], root=tmp_path)


def test_extracts_classes_methods_functions(tmp_path):
    info = _map(tmp_path)["files"]["attn.py"]
    assert info["classes"][0]["name"] == "Attn"
    assert info["classes"][0]["methods"][0]["name"] == "forward"
    assert any(fn["name"] == "helper" for fn in info["functions"])


def test_extracts_ttnn_ops_with_assignment_and_scope(tmp_path):
    ops = _map(tmp_path)["files"]["attn.py"]["ops"]
    by_name = {o["name"]: o for o in ops}
    assert by_name["ttnn.linear"]["assigned_to"] == "qkv"
    assert by_name["ttnn.linear"]["scope"] == "Attn.forward"
    assert by_name["ttnn.matmul"]["scope"] == "helper"


def test_render_filters_by_op_substrings(tmp_path):
    sk = render_skeleton(_map(tmp_path), op_substrings=["linear", "matmul"])
    assert "ttnn.linear" in sk and "ttnn.matmul" in sk
    assert "ttnn.layer_norm" not in sk  # filtered out for a matmul lever


def test_parse_error_is_captured_not_raised(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def (:\n")
    info = build_model_map([f], root=tmp_path)["files"]["bad.py"]
    assert "error" in info
