# tests/test_e2e_device.py
import pytest
from models.experimental.opt_transfer.graph import build_graph, RealImpl
from models.experimental.opt_transfer.schema import KBEntry, PatternKind, FusionProposal


class StubMatcher:
    """Stands in for the LLM during the device acceptance test so the gate is
    deterministic and key-free."""

    def propose(self, graph_summary, kb, diagnosis=None):
        return [
            FusionProposal(
                entry_id="qkv_merge",
                fused_op="ttnn.experimental.nlp_create_qkv_heads",
                matched_nodes=["q_proj", "k_proj", "v_proj"],
                config={"num_heads": "{H}", "num_kv_heads": "{H}", "transpose_k_heads": False},
                weight_transform="concat_qkv",
                rationale="siblings share input",
                source="x",
            )
        ]


def _kb():
    return [
        KBEntry(
            id="qkv_merge",
            fused_op="ttnn.experimental.nlp_create_qkv_heads",
            category="attention.qkv",
            pattern_kind=PatternKind.HORIZONTAL_MERGE,
            torch_pattern=["linear", "linear", "linear"],
            signature={"input_rank": 4},
            config_template={"num_heads": "{H}", "num_kv_heads": "{H}"},
            weight_transform="concat_qkv",
            source="x",
        )
    ]


@pytest.mark.device
def test_e2e_seamless_qkv_fusion_passes_pcc():
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        impl = RealImpl("seamless_m4t_v2", device, StubMatcher(), _kb())
        out = build_graph(impl).invoke({"model": "seamless_m4t_v2", "iteration": 0})
        assert out["status"] == "pass", out
        assert out["full_pcc"] > 0.99
    finally:
        ttnn.close_device(device)
