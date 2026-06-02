from models.experimental.opt_transfer.kb.miner import inventory_ops
from models.experimental.opt_transfer.config import CONFIG


def test_inventory_covers_the_supported_op_surface():
    inv = inventory_ops(CONFIG)
    assert len(inv) > 30
    assert "nlp_create_qkv_heads" in inv
    assert "scaled_dot_product_attention" in inv
    assert inv["nlp_create_qkv_heads"]["tests"]
    assert any("nlp_create_qkv_heads" in s for s in inv["nlp_create_qkv_heads"]["examples"])


from models.experimental.opt_transfer.kb.miner import scan_usage


def test_usage_scan_finds_model_callsites_with_provenance():
    usage = scan_usage(CONFIG)
    assert "nlp_create_qkv_heads" in usage
    hit = usage["nlp_create_qkv_heads"][0]
    assert any(r in hit["source"] for r in ("tt_transformers", "tt_dit", "demos"))
    assert "nlp_create_qkv_heads" in hit["snippet"]


from models.experimental.opt_transfer.kb.miner import build_kb
from models.experimental.opt_transfer.schema import KBEntry, PatternKind


class FakeClient:
    def __init__(self):
        self.calls = 0

    def extract_entries(self, op, available, used, golden_src) -> list[dict]:
        self.calls += 1
        pattern = (golden_src or (available["examples"][0] if available["examples"] else op)).split("\n")
        return [
            KBEntry(
                id=op,
                fused_op=f"ttnn.{op}",
                category="auto",
                pattern_kind=PatternKind.CHAIN,
                torch_pattern=pattern[:3],
                signature={},
                config_template={},
                weight_transform=None,
                source=(used[0]["source"] if used else (available["tests"][0] if available["tests"] else "tests")),
                usage_examples=available["examples"][:1],
            ).to_dict()
        ]


def test_build_kb_captures_available_used_and_provenance(tmp_path):
    client = FakeClient()
    # full build (no limit) so a specific op is guaranteed present, not cut by alpha-sort
    entries = build_kb(client=client, cache_root=tmp_path / "c", kb_root=tmp_path / "kb")
    by_id = {e.id: e for e in entries}
    assert len(by_id) > 25
    e = by_id["nlp_create_qkv_heads"]
    assert e.status == "in_use"
    assert e.unit_test_refs and any("test" in t for t in e.unit_test_refs)
    assert all(x.status in ("in_use", "supported_unused") for x in entries)
    assert all(x.pattern_source in ("golden", "unit_test", "llm") for x in entries)
    if "rms_norm" in by_id:
        assert by_id["rms_norm"].pattern_source == "golden"


def test_build_kb_is_cached(tmp_path):
    client = FakeClient()
    build_kb(client=client, cache_root=tmp_path / "c", kb_root=tmp_path / "kb", limit_ops=20)
    n = client.calls
    build_kb(client=client, cache_root=tmp_path / "c", kb_root=tmp_path / "kb", limit_ops=20)
    assert client.calls == n
