import torch
from models.experimental.opt_transfer.trace import trace_module
from models.experimental.opt_transfer.references.seamless_m4t_v2 import SeamlessBlock
from models.experimental.opt_transfer.structural import validate
from models.experimental.opt_transfer.schema import FusionProposal, KBEntry, PatternKind


def _graph():
    return trace_module(SeamlessBlock(1024, 16), (torch.randn(1, 8, 1024),))


def _merge_entry():
    return KBEntry(
        id="qkv_merge",
        fused_op="op",
        category="attention.qkv",
        pattern_kind=PatternKind.HORIZONTAL_MERGE,
        torch_pattern=["linear", "linear", "linear"],
        signature={},
        config_template={},
        weight_transform="concat_qkv",
        source="x",
    )


def test_horizontal_merge_accepts_siblings_sharing_input():
    p = FusionProposal("qkv_merge", "op", ["q_proj", "k_proj", "v_proj"], {}, "concat_qkv", "", "x")
    ok, reason = validate(_graph(), p, _merge_entry())
    assert ok, reason


def test_horizontal_merge_rejects_non_shared_input():
    p = FusionProposal("qkv_merge", "op", ["q_proj", "k_proj", "out_proj"], {}, "concat_qkv", "", "x")
    ok, reason = validate(_graph(), p, _merge_entry())
    assert not ok and "input" in reason
