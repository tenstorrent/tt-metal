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
