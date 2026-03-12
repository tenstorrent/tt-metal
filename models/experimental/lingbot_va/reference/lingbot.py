from diffusers import AutoencoderKLWan
from transformers import UMT5EncoderModel
from transformers import T5TokenizerFast
from models.experimental.lingbot_va.reference.WanTransformer3D import WanTransformer3DModel
from models.experimental.lingbot_va.reference.WanScheduler import FlowMatchScheduler
import numpy as np
import torch
import torch.nn.functional as F
import torch
import numpy as np
from diffusers import AutoencoderKLWan

import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def save_async(obj, file_path):
    """
    todo
    """
    if torch.is_tensor(obj) or (isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values())):
        if torch.is_tensor(obj):
            if obj.is_cuda:
                obj = obj.cpu()
        elif isinstance(obj, dict):
            obj = {k: v.cpu() if torch.is_tensor(v) else v for k, v in obj.items()}
        executor.submit(torch.save, obj, file_path)
    elif isinstance(obj, np.ndarray):
        obj_copy = obj.copy()
        executor.submit(np.save, file_path, obj_copy)
    else:
        executor.submit(torch.save, obj, file_path)


def get_mesh_id(f, h, w, t, f_w=1, f_shift=0, action=False):
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat(
        [
            ff.unsqueeze(0),
            hh.unsqueeze(0),
            ww.unsqueeze(0),
        ],
        dim=0,
    ).flatten(1)
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)
    return grid_id


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(
    tokenizer_path,
):
    tokenizer = T5TokenizerFast.from_pretrained(
        tokenizer_path,
    )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
):
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
    )
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)
    return x


def pre_process_images(images, height=224, width=224):
    # 1. Process images: convert to tensor, normalize, resize
    image_list = []
    for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        if key in images:
            img = images[key]

            # Convert numpy array to tensor
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()

            # Handle shape: [H, W, 3] -> [1, 3, H, W]
            if img.dim() == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            elif img.dim() == 4:
                img = img[:1]  # Take first batch if needed

            # Normalize from uint8 [0, 255] to [-1, 1]
            if img.max() > 1.0:
                img = img / 255.0 * 2.0 - 1.0

            # Resize to target size
            target_h = height
            target_w = width
            img = F.interpolate(img, size=(target_h, target_w), mode="bilinear", align_corners=False)

            image_list.append(img)
    return image_list


class WanVAEStreamingWrapper:
    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
            # import pdb; pdb.set_trace()
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc


class Lingbot:
    def __init__(self, config):
        self.config = config
        self.vae = load_vae(config.vae_path, config.torch_dtype, config.device)
        self.tt_text_encoder = TT_text_encoder()
        # self.text_encoder = load_text_encoder(config.text_encoder_path, config.torch_dtype, config.device)
        self.tokenizer = load_tokenizer(config.tokenizer_path)
        self.transformer = load_transformer(config.transformer_path, config.torch_dtype, config.device)
        self.transformer.eval()
        self.scheduler = FlowMatchScheduler(
            num_inference_steps=getattr(self.config, "num_inference_steps", 100),
            num_train_timesteps=getattr(self.config, "num_train_timesteps", 1000),
            shift=getattr(self.config, "shift", 3.0),
            sigma_max=getattr(self.config, "sigma_max", 1.0),
            sigma_min=getattr(self.config, "sigma_min", 0.003 / 1.002),
        )
        self.scheduler.set_timesteps(getattr(self.config, "num_inference_steps", 100))

        # Initialize VAE wrapper for streaming
        self.vae_wrapper = WanVAEStreamingWrapper(self.vae)

    def infer(self, obs, action_mode=False):
        """
        Single forward pass through the Lingbot model.

        Args:
            obs: Dictionary containing:
                - image: Dict with camera views (numpy arrays, uint8, shape [H, W, 3]):
                    - "base_0_rgb": Base camera view
                    - "left_wrist_0_rgb": Left wrist camera view (optional)
                    - "right_wrist_0_rgb": Right wrist camera view (optional)
                - prompt: Language instruction string
                - state: Robot state array (optional)

        Returns:
            Dictionary with transformer output latents
        """
        # Extract inputs
        images = obs.get("image", {})
        prompt = obs.get("prompt", "")
        state = obs.get("state", None)

        # Handle empty images case
        if not images or len(images) == 0:
            self.vae_wrapper.clear_cache()
            return {"latent": None, "reset": True}

        # 1. Process images: convert to tensor, normalize, resize
        # image_list = []
        # for key in ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']:
        #     if key in images:
        #         img = images[key]

        #         # Convert numpy array to tensor
        #         if isinstance(img, np.ndarray):
        #             img = torch.from_numpy(img).float()

        #         # Handle shape: [H, W, 3] -> [1, 3, H, W]
        #         if img.dim() == 3 and img.shape[-1] == 3:
        #             img = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        #         elif img.dim() == 4:
        #             img = img[:1]  # Take first batch if needed

        #         # Normalize from uint8 [0, 255] to [-1, 1]
        #         if img.max() > 1.0:
        #             img = img / 255.0 * 2.0 - 1.0

        #         # Resize to target size
        #         target_h = getattr(self.config, 'height', 224)
        #         target_w = getattr(self.config, 'width', 224)
        #         img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)

        #         image_list.append(img)
        target_h = getattr(self.config, "height", 224)
        target_w = getattr(self.config, "width", 224)
        image_list = pre_process_images(images, height=target_h, width=target_w)

        if len(image_list) == 0:
            self.vae_wrapper.clear_cache()
            return {"latent": None, "reset": True}

        # Stack images: [num_cams, 1, 3, H, W]
        stacked_images = torch.stack(image_list, dim=0)

        videos = stacked_images.squeeze(1)
        # Permute to [1, 3, num_cams, H, W] = [B, C, T, H, W]
        videos = videos.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, num_cams, H, W]

        num_cams = videos.shape[2]
        # Use only 2 frames to avoid temporal dimension mismatch
        # The VAE encoder's temporal downsampling expects compatible dimensions
        if num_cams >= 2:
            # Use only the first 2 cameras
            videos = videos[:, :, :2, :, :]  # [1, 3, 2, H, W]
        elif num_cams == 1:
            # Duplicate to make it 2 (minimum for temporal downsampling)
            videos = videos.repeat(1, 1, 2, 1, 1)  # [1, 3, 2, H, W]

        vae_device = next(self.vae.parameters()).device
        videos = videos.to(vae_device).to(self.config.torch_dtype)

        # Encode chunk
        enc_out = self.vae_wrapper.encode_chunk(videos)  # [1, 2*C, 1, H', W']
        mu, logvar = torch.chunk(enc_out, 2, dim=1)

        # Normalize latents
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        video_latent = (mu.float() - latents_mean.view(1, -1, 1, 1, 1)) * latents_std.view(1, -1, 1, 1, 1)
        # Convert to transformer dtype and device
        video_latent = video_latent.to(dtype=self.config.torch_dtype, device=self.config.device)

        # 3. Encode text prompt
        text_inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=64
        ).input_ids.to(self.config.device)

        with torch.no_grad():
            tt_text_input = ttnn.from_torch()
            text_emb = self.text_encoder(text_inputs).last_hidden_state  # [1, seq_len, text_dim]
            to_torch
            # Convert to transformer dtype
            text_emb = text_emb.to(dtype=self.config.torch_dtype)

        # 4. Prepare transformer input
        patch_size = getattr(self.config, "patch_size", [1, 2, 2])

        # Get grid IDs for positional encoding
        latent_f = video_latent.shape[2]  # temporal dimension
        latent_h = video_latent.shape[-2]
        latent_w = video_latent.shape[-1]

        grid_id = get_mesh_id(
            latent_f // patch_size[0], latent_h // patch_size[1], latent_w // patch_size[2], 0, 1, 0
        ).to(self.config.device)
        grid_id = grid_id[:3].unsqueeze(0)  # [1, 3, seq_len]

        # Prepare timesteps (zeros for single-pass inference)
        timesteps = torch.ones([1, video_latent.shape[2]], dtype=torch.float32, device=self.config.device) * 1

        # 5. Forward through transformer
        input_dict = {
            "noisy_latents": video_latent,  # [B, C, F, H, W]
            "text_emb": text_emb,  # [B, seq_text, text_dim]
            "grid_id": grid_id,  # [B, 3, seq_len]
            "timesteps": timesteps,  # [B, F_patched]
        }

        # video mode
        with torch.no_grad():
            latent_output = self.transformer(
                input_dict, update_cache=0, cache_name="pos", action_mode=False, train_mode=False
            )
        print("Transformer video output shape:", latent_output.shape)
        print(latent_output)

        action_per_frame = getattr(self.config, "action_per_frame", 16)  # From config
        F_action = getattr(self.config, "F_action", 8)  # Number of action frames
        action_dim = getattr(self.config, "action_dim", 30)

        # Action tokens should be: (B, action_dim, F, action_per_frame, 1)
        action_tokens = torch.randn(
            1, action_dim, F_action, action_per_frame, 1, dtype=self.config.torch_dtype, device=self.config.device
        )
        # Shape: (B, 30, 8, 16, 1)

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
        ).to(self.config.device)

        action_grid_id = action_grid_id[:3].unsqueeze(0)

        # Timesteps for action mode - should be per frame
        action_timesteps = torch.ones([1, action_tokens.shape[2]], dtype=torch.float32, device=self.config.device) * 1
        input_dict_action = {
            "noisy_latents": action_tokens,
            "text_emb": text_emb,
            "timesteps": action_timesteps,
            "grid_id": action_grid_id,
        }
        with torch.no_grad():
            action_output = self.transformer(
                input_dict_action, update_cache=0, cache_name="pos", action_mode=True, train_mode=False
            )
        print("Transformer action output shape:", action_output.shape)
        print(action_output)

        save_async(latent_output, "models/experimental/lingbot_va/reference/latent_output.pt")
        save_async(action_output, "models/experimental/lingbot_va/reference/action_output.pt")
        return {"latent": latent_output, "action": action_output}
