import pytest, torch
import ttnn

# from .tenstorrent import (
# from  .tenstorrent_bfp16_L1 import (
# from .tenstorrent_dram_fp16_may25 import (
# from .tenstorrent_dram_i2s_fp32_may25 import (
# from .tenstorrent_moritz_jun7 import (
# from .tenstorrent_attn_i2s_jun9 import (
# from .tenstorrent_moritz_L1_jun9 import (
# from .tenstorrent_moritz_L1_jun11 import (
# from .tenstorrent_moritz_SDPA_jun13 import (
from .tenstorrent_moritz_SDPA_jun18 import (
    filter_dict,
    PairformerModule,
    DiffusionTransformerModule,
    # MSAModule,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
import time

# from boltz.model.modules.trunk import PairformerModule as PairformerModuleTorch
# from boltz.model.modules.diffusion import (
#    DiffusionTransformer as DiffusionTransformerTorch,
# )

torch.set_grad_enabled(False)
torch.manual_seed(893)

# state_dict = torch.load(
#    "/home/yfan/.boltz/boltz1_conf.ckpt", map_location="cpu", mmap=True
# )["state_dict"]
state_dict = torch.load("models/experimental/boltz1/boltz1_conf_dict.pth")


def median_relative_error(a, b):
    return ((a - b).abs() / b.abs()).median().item()


@pytest.mark.parametrize("seq_len", [128, 192, 256, 512, 686, 768, 1024])
def test_pairformer(device, seq_len):
    ttnn.device.EnablePersistentKernelCache()  # be careful, can lead to bugs when profiling etc.
    device.enable_program_cache()
    pairformer = PairformerModule(
        device,
        n_blocks=2,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=24,
        att_n_heads=16,
    )

    # pairformer_torch = PairformerModuleTorch(
    #    token_s=384, token_z=128, num_blocks=2
    # ).eval()
    pairformer_state_dict = filter_dict(state_dict, "pairformer_module")
    pairformer.load_state_dict(
        pairformer_state_dict,
        strict=False,
    )
    # pairformer_torch.load_state_dict(pairformer_state_dict, strict=False)

    gen_ref_tensors = 0

    if gen_ref_tensors:
        s = 8 * torch.randn(1, seq_len, 384)
        z = 26 * torch.randn(1, seq_len, seq_len, 128)
        mask = torch.ones(1, seq_len)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        seq_len = str(seq_len)
        torch.save(s, "models/experimental/boltz1/tensor_s_" + seq_len + ".pt")
        torch.save(z, "models/experimental/boltz1/tensor_z_" + seq_len + ".pt")
    else:
        mask = torch.ones(1, seq_len)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        seq_len = str(seq_len)
        s = torch.load("models/experimental/boltz1/tensor_s_" + seq_len + ".pt")
        z = torch.load("models/experimental/boltz1/tensor_z_" + seq_len + ".pt")

    # s_tt, z_tt = pairformer(s, z, mask, pair_mask)

    start = time.time()
    s_tt, z_tt = pairformer(s, z, mask, pair_mask)
    end = time.time()
    print(f"$$$YF: pairformer time: {end - start:.4f} seconds")

    if not gen_ref_tensors:
        s_tt_ref = torch.load("models/experimental/boltz1/tensor_s_out_" + seq_len + "_dram.pt")  # .to(torch.bfloat16)
        z_tt_ref = torch.load("models/experimental/boltz1/tensor_z_out_" + seq_len + "_dram.pt")  # .to(torch.bfloat16)
    else:
        torch.save(s_tt, "models/experimental/boltz1/tensor_s_out_" + seq_len + "_dram.pt")  # .to(torch.bfloat16)
        torch.save(z_tt, "models/experimental/boltz1/tensor_z_out_" + seq_len + "_dram.pt")  # .to(torch.bfloat16)

    if not gen_ref_tensors:
        # assert median_relative_error(s_tt, s_tt_ref) < 1e-1, "s not accurate"
        # assert median_relative_error(z_tt, z_tt_ref) < 1e-1, "z not accurate"

        assert_with_pcc(z_tt, z_tt_ref, 0.9988)
        assert_with_pcc(s_tt, s_tt_ref, 0.99)

    """
    s_sharded = ttnn.to_memory_config(
        s,
        memory_config=ttnn.create_sharded_memory_config(
            s.shape,
            ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat16,

    ttnn.deallocate(s)

    z_sharded = ttnn.to_memory_config(
        z,
        memory_config=ttnn.create_sharded_memory_config(
            s.shape,
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(z)
    s_tt, z_tt = pairformer(s_sharded, z_sharded, mask, pair_mask)
    """

    # s_torch, z_torch = pairformer_torch(s, z, mask, pair_mask)
    # assert median_relative_error(s_tt, s_torch) < 1e-1, "s not accurate"
    # assert median_relative_error(z_tt, z_torch) < 1e-1, "z not accurate"


@pytest.mark.parametrize("seq_len", [128, 512, 768, 1024])
def test_token_transformer(device, seq_len):
    token_transformer = DiffusionTransformerModule(
        device,
        n_layers=2,
        dim=768,
        n_heads=16,
    )
    # token_transformer_torch = DiffusionTransformerTorch(
    #    depth=2, heads=16, dim=768, dim_single_cond=768, dim_pairwise=128
    # ).eval()
    token_transformer_state_dict = filter_dict(state_dict, "structure_module.score_model.token_transformer")
    token_transformer.load_state_dict(
        token_transformer_state_dict,
        strict=False,
    )
    # token_transformer_torch.load_state_dict(token_transformer_state_dict, strict=False)
    a = 3 + 5 * torch.randn(1, seq_len, 768)
    s = -2 + 42 * torch.randn(1, seq_len, 768)
    z = 10 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    a_tt = token_transformer(
        a,
        s,
        z,
        mask,
    )
    # a_torch = token_transformer_torch(
    #    a,
    #    s,
    #    z,
    #    mask,
    # )
    # assert median_relative_error(a_tt, a_torch) < 1e-1, "a not accurate"


@pytest.mark.parametrize("seq_len", [100, 500, 1000])
def test_msa(seq_len):
    n_sequences = 100
    msa = MSAModule(
        n_blocks=4,
        avg_head_dim=32,
        avg_n_heads=8,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
    )
    # msa_torch = MSAModuleTorch(msa_s=64, token_z=128, s_input_dim=455, msa_blocks=4, msa_dropout=0, z_dropout=0).eval()
    msa_state_dict = filter_dict(state_dict, "msa_module")
    msa.load_state_dict(msa_state_dict)
    # msa_torch.load_state_dict(msa_state_dict)
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
    z_tt = msa(z, emb, feats)
    # z_torch = msa_torch(z, emb, feats)
    # assert median_relative_error(z_tt, z_torch) < 1e-1, "z not accurate"
