import torch
import ttnn

# import pytest
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualDownBlock
from models.experimental.lingbot_va.tt.wan_residual_down_block import WanResidualDownBlock as WanResidualDownBlockTTNN
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.parallel.config import VaeHWParallelConfig, ParallelFactor
from models.common.metrics import compute_pcc


# Test configuration constants
CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints_robotwin/vae"
BATCH_SIZE = 1
PCC_THRESHOLD = 0.99
H = 128  # Height
W = 160  # Width
T = 1  # Frames


def load_autoencoder_klwan(checkpoint_path):
    print("Loading AutoencoderKLWan...")
    autoencoder_klwan = AutoencoderKLWan.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    ).to(device="cpu")
    autoencoder_klwan.eval()
    return autoencoder_klwan


def test_wan_res_down_block():
    autoencoder_klwan = load_autoencoder_klwan(CHECKPOINT_PATH)
    wan_res_down_block_weights = autoencoder_klwan.encoder.down_blocks[1].state_dict()

    in_dim = 160
    out_dim = 320
    temperal_downsample = True
    down_flag = True
    wan_res_down_block_torch = WanResidualDownBlock(
        in_dim=in_dim,
        out_dim=out_dim,
        dropout=0.0,
        num_res_blocks=2,
        temperal_downsample=temperal_downsample,
        down_flag=down_flag,
    )
    wan_res_down_block_torch.load_state_dict(wan_res_down_block_weights)
    wan_res_down_block_torch.eval()

    # Create input tensor in (B, C, T, H, W) format
    input_tensor = torch.randn(BATCH_SIZE, in_dim, T, H, W, dtype=torch.float32)
    feat_cache = [torch.randn(1, 12, 1, 128, 160) for _ in range(26)]
    input_tensor = input_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
    print("Running WanResidualDownBlock forward pass...")
    with torch.no_grad():
        wan_res_down_block_out = wan_res_down_block_torch(input_tensor, feat_cache, feat_idx=[1])
    print("WanResidualDownBlock output shape:", wan_res_down_block_out.shape)
    mesh_device = None

    try:
        # ── 1. Create mesh device and parallel config ──
        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=1, mesh_axis=0),
            width_parallel=ParallelFactor(factor=1, mesh_axis=1),
        )

        wan_res_block_ttnn = WanResidualDownBlockTTNN(
            in_dim=in_dim,
            out_dim=out_dim,
            num_res_blocks=2,
            temperal_downsample=temperal_downsample,
            down_flag=down_flag,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        wan_res_block_ttnn.load_torch_state_dict(wan_res_down_block_weights)

        input_tensor = input_tensor.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)

        tt_input_tensor = ttnn.from_torch(
            input_tensor, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        tt_feat_cache = [
            ttnn.from_torch(feat, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
            for feat in feat_cache
        ]
        # Run TTNN forward pass
        tt_output = wan_res_block_ttnn(tt_input_tensor, tt_feat_cache, feat_idx=[1])

        tt_output = ttnn.permute(tt_output, (0, 4, 1, 2, 3))  # (B, C, T, H, W)
        print("WanResidualDownBlock TTNN output shape:", tt_output.shape)

        tt_wan_res_block_out_torch = ttnn.to_torch(tt_output)

        # ── 2. Compare outputs ──
        print("Comparing outputs...")
        pcc = compute_pcc(wan_res_down_block_out, tt_wan_res_block_out_torch)
        print(f"PCC: {pcc}")
        assert pcc >= PCC_THRESHOLD, f"PCC {pcc} is less than {PCC_THRESHOLD}"
    except Exception as e:
        print(f"Error: {e}")
        # pytest.fail(f"Error: {e}")

    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    test_wan_res_down_block()
