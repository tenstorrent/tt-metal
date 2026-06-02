import json
from models.experimental.opt_transfer.matcher import Matcher
from models.experimental.opt_transfer.schema import KBEntry, PatternKind, FusionProposal


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


class FakeTransport:
    def __init__(self):
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        payload = [
            FusionProposal(
                entry_id="qkv_merge",
                fused_op="ttnn.experimental.nlp_create_qkv_heads",
                matched_nodes=["q_proj", "k_proj", "v_proj"],
                config={"num_heads": "{H}", "num_kv_heads": "{H}"},
                weight_transform="concat_qkv",
                rationale="siblings share input",
                source="x",
            ).__dict__
        ]
        return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def test_matcher_returns_proposals():
    t = FakeTransport()
    m = Matcher(transport=t)
    graph_summary = [
        {"name": "q_proj", "kind": "linear", "inputs": ["h"]},
        {"name": "k_proj", "kind": "linear", "inputs": ["h"]},
        {"name": "v_proj", "kind": "linear", "inputs": ["h"]},
    ]
    props = m.propose(graph_summary, _kb())
    assert props[0].matched_nodes == ["q_proj", "k_proj", "v_proj"]


def test_matcher_marks_kb_for_prompt_caching():
    t = FakeTransport()
    Matcher(transport=t).propose([], _kb())
    sys_blocks = t.last_request["system"]
    assert any(b.get("cache_control", {}).get("type") == "ephemeral" for b in sys_blocks)
