import pytest, torch

from models.experimental.boltz1.tenstorrent_moritz_SDPA_jun18 import (
    filter_dict,
    MSAModule,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
import time

from models.experimental.boltz1.reference.model.modules.trunk import (
    MSAModule as MSAModuleTorch,
)

torch.set_grad_enabled(False)
torch.manual_seed(893)

# state_dict = torch.load(
#    "/home/yfan/.boltz/boltz1_conf.ckpt", map_location="cpu", mmap=True
# )["state_dict"]
state_dict = torch.load("models/experimental/boltz1/boltz1_conf_dict.pth")


def median_relative_error(a, b):
    return ((a - b).abs() / b.abs()).median().item()


@pytest.mark.parametrize("n_blocks", [4])
@pytest.mark.parametrize("seq_len", [100, 500, 1000])
def test_msa(device, seq_len, n_blocks):
    n_sequences = 100
    msa = MSAModule(
        device,
        n_blocks=n_blocks,
        avg_head_dim=32,
        avg_n_heads=8,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
    )

    msa_torch = MSAModuleTorch(
        msa_s=64, token_z=128, s_input_dim=455, msa_blocks=n_blocks, msa_dropout=0, z_dropout=0
    ).eval()
    msa_state_dict = filter_dict(state_dict, "msa_module")

    print("Loading ttnn weights")
    msa.load_state_dict(msa_state_dict)

    print("Loading torch weights")
    msa_torch.load_state_dict(msa_state_dict)

    z = 7 * torch.randn(1, seq_len, seq_len, 128)
    emb = torch.ones(1, seq_len, 455)

    feats = {
        "msa": torch.nn.functional.one_hot(torch.randint(33, (1, n_sequences, seq_len)), 33),
        "has_deletion": torch.zeros((1, n_sequences, seq_len), dtype=torch.bool),
        "deletion_value": torch.zeros((1, n_sequences, seq_len)),
        "msa_paired": torch.zeros((1, n_sequences, seq_len)),
        "msa_mask": torch.ones((1, n_sequences, seq_len)),
        "token_pad_mask": torch.ones((1, seq_len)),
    }

    print("forward pass with torch")
    z_ref = msa_torch(z, emb, feats)

    print("forward pass with ttnn")
    start = time.time()
    z_tt = msa(z, emb, feats)
    end = time.time()

    print(f"$$$YF: MSAModule time: {end - start:.4f} seconds")

    assert_with_pcc(z_tt, z_ref, 0.99)
