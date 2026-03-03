# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.metrics import compute_pcc
from models.experimental.lingbot_va.tt.wan_time_text_image_embedding import TtWanTimeTextImageEmbedding
from models.experimental.lingbot_va.reference.model import WanTimeTextImageEmbedding
from loguru import logger
from models.experimental.lingbot_va.reference.model import WanTransformer3DModel


def test_time_text_image_embedding():
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    device = torch.device("cpu")
    dtype = torch.bfloat16
    ckpt_path = "models/experimental/lingbot_va/reference/checkpoints/transformer"
    transformer = WanTransformer3DModel.from_pretrained(ckpt_path, torch_dtype=dtype, attn_mode="torch").to(device)
    condition_embedder_weights = transformer.condition_embedder.state_dict()
    transformer.eval()
    # -------------------------------
    # 1️⃣ Create Torch model with dummy weights
    # -------------------------------
    torch_embedding = WanTimeTextImageEmbedding(
        dim=3072,
        time_freq_dim=256,
        time_proj_dim=18432,
        text_embed_dim=4096,
        pos_embed_seq_len=None,
    ).to(device)
    torch_embedding.load_state_dict(condition_embedder_weights)
    torch_embedding = torch_embedding.to(dtype=torch.bfloat16)
    torch_embedding.eval()

    # import pdb; pdb.set_trace()
    # -------------------------------
    # 2️⃣ Create TT model
    # -------------------------------
    tt_embedding = TtWanTimeTextImageEmbedding(
        dim=3072,
        time_freq_dim=256,
        time_proj_dim=18432,
        text_embed_dim=4096,
        mesh_device=mesh_device,
    )
    tt_state_dict = {k: v for k, v in torch_embedding.state_dict().items() if not k.startswith("text_embedder.")}
    tt_embedding.load_torch_state_dict(tt_state_dict)

    # -------------------------------
    # 3️⃣ Create dummy input
    # -------------------------------
    # Create timestep input
    B = 1
    L = 5
    timestep = torch.randint(0, 1000, (B, L), dtype=torch.float32)
    timestep_tt = ttnn.from_torch(timestep, dtype=ttnn.float32, device=mesh_device)
    # Fix: Use float32_tensor utility and prepare as 4D tensor
    # Reshape to (B*L,) first, then unsqueeze to 4D (1, 1, 1, B*L)
    # timestep_flat = timestep.reshape(-1)  # (B*L,)
    # timestep_4d = timestep_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, B*L)
    # timestep_tt = float32_tensor(timestep_4d, device=mesh_device)
    # -------------------------------
    # 4️⃣ Forward pass
    # -------------------------------
    torch_temb, torch_timestep_proj = torch_embedding(timestep, dtype=torch.bfloat16)
    import pdb

    pdb.set_trace()
    tt_temb, tt_timestep_proj = tt_embedding(timestep_tt)
    import pdb

    pdb.set_trace()
    tt_temb_torch = ttnn.to_torch(tt_temb, dtype=torch.bfloat16)
    tt_timestep_proj_torch = ttnn.to_torch(tt_timestep_proj, dtype=torch.bfloat16)
    # -------------------------------
    # 5️⃣ Compare outputs
    # -------------------------------
    pcc_temb = compute_pcc(torch_temb, tt_temb_torch)
    pcc_timestep_proj = compute_pcc(torch_timestep_proj, tt_timestep_proj_torch)

    logger.info(f"PCC for temb: {pcc_temb:.6f}")
    logger.info(f"PCC for timestep_proj: {pcc_timestep_proj:.6f}")

    assert pcc_temb > 0.999, f"temb PCC {pcc_temb:.6f} is below threshold 0.999"
    assert pcc_timestep_proj > 0.999, f"timestep_proj PCC {pcc_timestep_proj:.6f} is below threshold 0.999"


if __name__ == "__main__":
    test_time_text_image_embedding()
