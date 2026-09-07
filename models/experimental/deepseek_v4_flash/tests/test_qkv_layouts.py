import torch

from models.experimental.deepseek_v4_flash.tt.attention import _fold_norm_gamma
from models.experimental.deepseek_v4_flash.tt.decode_prefetch import DECODE_GCB_GROUP, DECODE_LAYOUTS
from models.experimental.deepseek_v4_flash.tt.l1_placement import placement_for


def test_q_a_norm_gamma_folds_exactly_into_q_b():
    torch.manual_seed(0)
    q_a = torch.randn(3, 8)
    gamma = torch.randn(8)
    q_b_weight = torch.randn(12, 8)

    folded = _fold_norm_gamma(q_b_weight, gamma)()

    torch.testing.assert_close((q_a * gamma) @ q_b_weight.T, q_a @ folded.T)


def test_q_a_uses_full_width_32_core_layout():
    assert DECODE_LAYOUTS["q_a_proj"] == {"K": 4096, "N": 1024, "n_blocks": 32}


def test_q_a_uses_a_private_prefetch_ring():
    assert "q_a_proj" not in DECODE_GCB_GROUP


def test_packed_q_a_matches_full_width_layout():
    placement = placement_for("q_a_proj")
    assert placement.zone == "Z1"
    assert placement.k_blocks is None
    assert placement.n_blocks == 32
    assert placement.shard_shape == (4096, 32)
