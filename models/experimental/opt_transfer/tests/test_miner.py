from models.experimental.opt_transfer.kb.miner import inventory_ops
from models.experimental.opt_transfer.config import CONFIG


def test_inventory_covers_the_supported_op_surface():
    inv = inventory_ops(CONFIG)
    assert len(inv) > 30
    assert "nlp_create_qkv_heads" in inv
    assert "scaled_dot_product_attention" in inv
    assert inv["nlp_create_qkv_heads"]["tests"]
    assert any("nlp_create_qkv_heads" in s for s in inv["nlp_create_qkv_heads"]["examples"])
