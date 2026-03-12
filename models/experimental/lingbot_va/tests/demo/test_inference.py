# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.lingbot_va.reference.WanTransformer3D import WanTransformer3DModel
from transformers import T5Tokenizer, T5EncoderModel
from models.experimental.lingbot_va.reference.get_mesh_id import get_mesh_id

# ----------------------------------------------------------------------
# 1) Device / dtype
# ----------------------------------------------------------------------
device = torch.device("cpu")
dtype = torch.bfloat16

# ----------------------------------------------------------------------
# 2) Load real checkpoint (e.g., lingbot-va-base)
# ----------------------------------------------------------------------
ckpt_path = "models/experimental/lingbot_va/reference/checkpoints/transformer"
transformer = WanTransformer3DModel.from_pretrained(ckpt_path, torch_dtype=dtype, attn_mode="torch").to(device)
transformer.eval()

# ----------------------------------------------------------------------
# 3) Minimal sample inputs matching transformer.forward signature
# ----------------------------------------------------------------------
B = 1
C = 48
F = 8
H, W = 24, 24
action_dim = 30

# Get patch size from model config
patch_size = transformer.config.patch_size  # e.g., [1, 2, 2]
patch_f, patch_h, patch_w = patch_size

# Dummy noisy latents (video mode) - shape should account for patch size
noisy_latents = torch.randn(B, C, F, H, W, dtype=dtype, device=device)

# Dummy text embeddings (e.g., from T5)
tokenizer = T5Tokenizer.from_pretrained("models/experimental/lingbot_va/reference/checkpoints/tokenizer")
text_encoder = T5EncoderModel.from_pretrained(
    "models/experimental/lingbot_va/reference/checkpoints/text_encoder", torch_dtype=dtype
).to(device)
prompt = "Pick up the bottle."
text_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=64).input_ids.to(device)
with torch.no_grad():
    text_emb = text_encoder(text_inputs).last_hidden_state  # (1, seq_text, 4096)

# Calculate dimensions after patching
F_patched = F // patch_f  # 8 // 1 = 8
H_patched = H // patch_h  # 24 // 2 = 12
W_patched = W // patch_w  # 24 // 2 = 12

# Timesteps should be per frame: (B, F_patched)
# Each frame's timestep will be repeated across spatial dimensions (H × W)
timesteps = torch.randint(0, 1000, (B, F_patched), device=device, dtype=torch.float32)  # Shape: (B, 8)

# Create grid_id using get_mesh_id (shape: [4, seq_len], then expand to [B, 4, seq_len])
# The function expects dimensions after patching
latent_grid_id = get_mesh_id(
    f=F // patch_f,  # number of frames after patching
    h=H // patch_h,  # height after patching
    w=W // patch_w,  # width after patching
    t=0,  # timestep (0 for video mode)
    f_w=1,
    f_shift=0,
    action=False,
).to(
    device
)  # Shape: [4, seq_len]

# Expand to batch dimension: [B, 4, seq_len]
# Note: rope.forward only uses first 3 dims (f, h, w), so [B, 3, seq_len] would also work
latent_grid_id = latent_grid_id[:3].unsqueeze(0).repeat(B, 1, 1)  # [B, 3, seq_len]

input_dict = {
    "noisy_latents": noisy_latents,
    "text_emb": text_emb,
    "timesteps": timesteps,
    "grid_id": latent_grid_id,  # Now has shape [B, 3, seq_len]
}

# ----------------------------------------------------------------------
# 4) Single forward pass: video mode
# ----------------------------------------------------------------------
with torch.no_grad():
    out_video = transformer(
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
    )
print("Video output shape:", out_video.shape)
print(out_video)
# ----------------------------------------------------------------------
# 5) Single forward pass: action mode
# ----------------------------------------------------------------------
action_per_frame = 16  # From config
F_action = 8  # Number of action frames

# Action tokens should be: (B, action_dim, F, action_per_frame, 1)
action_tokens = torch.randn(B, action_dim, F_action, action_per_frame, 1, dtype=dtype, device=device)
# Shape: (B, 30, 8, 16, 1)

# After rearrange: (B, 30, 8, 16, 1) -> (B, 8*16*1, 30) = (B, 128, 30)
# So grid_id should have sequence length = F * action_per_frame = 8 * 16 = 128

# Create grid_id for action mode with correct dimensions
action_grid_id = get_mesh_id(
    f=F_action,  # 8 frames
    h=action_per_frame,  # 16 (not 1!)
    w=1,
    t=1,  # t=1 for action mode
    f_w=1,
    f_shift=0,
    action=True,  # Important: set action=True
).to(device)

action_grid_id = action_grid_id[:3].unsqueeze(0).repeat(B, 1, 1)  # [B, 3, 128]

# Timesteps for action mode - should be per frame
action_timesteps = torch.randint(0, 1000, (B, F_action), device=device, dtype=torch.float32)

input_dict_action = {
    "noisy_latents": action_tokens,  # Shape: (B, 30, 8, 16, 1)
    "text_emb": text_emb,
    "timesteps": action_timesteps,  # Shape: (B, 8)
    "grid_id": action_grid_id,  # Shape: (B, 3, 128)
}

with torch.no_grad():
    out_action = transformer(
        input_dict_action,
        update_cache=0,
        cache_name="pos",
        action_mode=True,
    )
print("Action output shape:", out_action.shape)
print(out_action)
