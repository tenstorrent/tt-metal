import pytest, torch
import ttnn
import os

from models.experimental.boltz1.tenstorrent_moritz_SDPA_jun18 import PairformerModule, filter_dict
from tests.ttnn.utils_for_testing import assert_with_pcc
import time

from models.experimental.boltz1.reference.model.modules.trunk import PairformerModule as PairformerModuleTorch


torch.set_grad_enabled(False)
torch.manual_seed(893)


state_dict = torch.load("models/experimental/boltz1/boltz1_conf_dict.pth")


def median_relative_error(a, b):
    return ((a - b).abs() / b.abs()).median().item()


@pytest.mark.parametrize("seq_len", [128, 192, 256, 512, 686, 768, 1024])
@pytest.mark.parametrize("n_blocks", [1, 48])
def test_pairformer(device, seq_len, n_blocks):
    ttnn.device.EnablePersistentKernelCache()  # be careful, can lead to bugs when profiling etc.
    device.enable_program_cache()

    pairformer = PairformerModule(
        device,
        n_blocks=n_blocks,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=24,
        att_n_heads=16,
    )

    pairformer_torch = PairformerModuleTorch(
        token_s=384,
        token_z=128,
        num_blocks=n_blocks,
        num_heads=16,
        pairwise_head_width=32,
        pairwise_num_heads=4,
    ).eval()

    pairformer_state_dict = filter_dict(state_dict, "pairformer_module")

    pairformer.load_state_dict(
        pairformer_state_dict,
        strict=False,
    )
    pairformer_torch.load_state_dict(pairformer_state_dict, strict=False)

    tensor_cache_dir = "models/experimental/boltz1/tensor_cache/"
    if not os.path.exists(tensor_cache_dir):
        os.makedirs(tensor_cache_dir)

    if not (
        os.path.isfile(tensor_cache_dir + "tensor_s_" + str(seq_len) + "_" + str(n_blocks) + ".pt")
        and os.path.isfile(tensor_cache_dir + "tensor_z_" + str(seq_len) + "_" + str(n_blocks) + ".pt")
    ):
        print("reference tensors not found, computing now")
        s = 8 * torch.randn(1, seq_len, 384)
        z = 26 * torch.randn(1, seq_len, seq_len, 128)
        mask = torch.ones(1, seq_len)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        torch.save(s, tensor_cache_dir + "tensor_s_" + str(seq_len) + "_" + str(n_blocks) + ".pt")
        torch.save(z, tensor_cache_dir + "tensor_z_" + str(seq_len) + "_" + str(n_blocks) + ".pt")
    else:
        print("loading reference tensors from cache")

        mask = torch.ones(1, seq_len)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s = torch.load(tensor_cache_dir + "tensor_s_" + str(seq_len) + "_" + str(n_blocks) + ".pt")
        z = torch.load(tensor_cache_dir + "tensor_z_" + str(seq_len) + "_" + str(n_blocks) + ".pt")

    s_ref, z_ref = pairformer_torch(s, z, mask, pair_mask)

    start = time.time()
    s_tt, z_tt = pairformer(s, z, mask, pair_mask)
    end = time.time()
    print(f"$$$YF: Pairformer time: {end - start:.4f} seconds")

    assert_with_pcc(z_tt, z_ref, 0.99)
    assert_with_pcc(s_tt, s_ref, 0.98)
