# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN Lingbot-VA staged wrapper: ``TtLingbotVA`` runs text encoder → VAE encode → Wan transformer per forward.

All random **torch** inputs are drawn in :meth:`__init__` and uploaded to the mesh there. :meth:`forward` only
consumes those device tensors (no ``torch.*`` and no ``ttnn.from_torch`` in the forward path). Wan noise
normalizes robotwin fused latent H×W via :func:`_robotwin_tshape_latent_hw` so ctor Wan noise matches
:func:`demo._encode_obs_ttnn` / :func:`demo._reset_infer_state` even if an older demo helper is loaded.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import torch
import ttnn
from models.experimental.lingbot_va.reference.utils import VA_CONFIGS

from models.experimental.lingbot_va.tests.demo import demo as lingbot_demo

# Match demo / UMT5 / Wan (see ``models.experimental.lingbot_va.tt.transformer_wan.TEXT_DIM``).
_TEXT_SEQ_LEN = 512
_WAN_LATENT_CHANNELS = 48


def _robotwin_tshape_latent_hw(height: int, width: int) -> tuple[int, int]:
    """Spatial H×W of fused latents from ``_encode_obs_ttnn`` for ``robotwin_tshape``.

    Matches ``demo._reset_infer_state`` and the corrected ``_prepare_state_for_vae_encode``
    (``latent_width`` is ``width // 16``, not ``((width // 16) * 3) // 2``).
    """
    lh = ((height // 16) * 3) // 2
    lw = width // 16
    return lh, lw


class TtLingbotVA:
    """Loads shared phase-1 assets once; each :meth:`forward` cycles TT text encoder → VAE → Wan (each freed after use)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        mesh_device: ttnn.MeshDevice,
        *,
        save_dir: str | Path | None = None,
        frame_chunk_size: int = 2,
        random_seed: int = 0,
    ) -> None:
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_path}")

        lingbot_demo._set_seed()
        os.chdir(lingbot_demo._REPO_ROOT)
        self.config = deepcopy(VA_CONFIGS["robotwin"])
        self.config.wan22_pretrained_model_name_or_path = str(checkpoint_path)
        self.config.local_rank = 0
        self.config.rank = 0
        self.config.world_size = 1
        self.config.num_chunks_to_infer = 1
        self.config.frame_chunk_size = frame_chunk_size
        if save_dir is None:
            save_dir = lingbot_demo._SCRIPT_DIR
        self.config.save_root = str(save_dir)

        self.mesh_device = mesh_device
        self.models = lingbot_demo._load_models_phase1(self.config, load_text_encoder=False, mesh_device=mesh_device)

        g = torch.Generator().manual_seed(random_seed)

        # Spatial sizes for VAE / latents; override with fused layout so Wan noise matches ``init_latent``
        # even if an older ``demo._prepare_state_for_vae_encode`` (wrong ``latent_width``) is on PYTHONPATH.
        self._spatial_state: dict = {}
        lingbot_demo._prepare_state_for_vae_encode(self._spatial_state, self.config)
        F = frame_chunk_size
        H, W = self.config.height, self.config.width
        keys = self.config.obs_cam_keys
        if self.config.env_type != "robotwin_tshape" or len(keys) != 3:
            raise ValueError("TtLingbotVA random VAE path expects robotwin_tshape with three obs_cam_keys")

        lh, lw = _robotwin_tshape_latent_hw(H, W)
        self._spatial_state["latent_height"] = lh
        self._spatial_state["latent_width"] = lw

        # 1) Text encoder inputs (torch in ctor only; device tensors used in :meth:`forward`).
        text_input_ids_cpu = torch.randint(0, 32_000, (1, _TEXT_SEQ_LEN), dtype=torch.int32, generator=g)
        text_attention_mask_cpu = torch.ones(1, _TEXT_SEQ_LEN, dtype=torch.bfloat16)
        self._tt_text_input_ids = ttnn.from_torch(text_input_ids_cpu, dtype=ttnn.uint32, device=mesh_device)
        self._tt_text_attention_mask = ttnn.from_torch(text_attention_mask_cpu, dtype=ttnn.bfloat16, device=mesh_device)

        # 2) VAE encoder: BCTHW like ``_encode_obs_ttnn`` ``obs_ttnn`` branch (values ~[0, 255] before *2/255-1).
        high = torch.rand((1, 3, F, H, W), dtype=torch.bfloat16, generator=g) * 255.0
        lr_h, lr_w = H // 2, W // 2
        left = torch.rand((1, 3, F, lr_h, lr_w), dtype=torch.bfloat16, generator=g) * 255.0
        right = torch.rand((1, 3, F, lr_h, lr_w), dtype=torch.bfloat16, generator=g) * 255.0
        self._vae_obs = {
            "obs_ttnn": [
                {
                    keys[0]: ttnn.from_torch(
                        high, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device
                    ),
                    keys[1]: ttnn.from_torch(
                        left, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device
                    ),
                    keys[2]: ttnn.from_torch(
                        right, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device
                    ),
                }
            ]
        }

        # 3) Wan noisy latents: same shape rule as ``demo._randn_ttnn`` in ``_infer_impl``.
        wan_noise_cpu = torch.randn((1, _WAN_LATENT_CHANNELS, F, lh, lw), dtype=torch.bfloat16, generator=g)
        self._tt_wan_spatial_noise = ttnn.from_torch(
            wan_noise_cpu.contiguous(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=mesh_device,
        )

    def forward(self) -> ttnn.Tensor:
        """Load text encoder → encode → free; load VAE → encode → free; load Wan → one video forward → free.

        Uses only pre-uploaded ``ttnn`` inputs from :meth:`__init__` (no torch, no ``from_torch`` here).
        """
        cfg = self.config
        models = self.models

        lingbot_demo._load_text_encoder_into_models(models, cfg)
        prompt_embeds, neg_embeds = lingbot_demo._encode_prompt_ttnn(
            models,
            self._tt_text_input_ids,
            self._tt_text_attention_mask,
            do_classifier_free_guidance=(cfg.guidance_scale > 1),
            max_sequence_length=_TEXT_SEQ_LEN,
        )
        lingbot_demo._free_tt_model(models, "text_encoder")

        st = dict(self._spatial_state)
        lingbot_demo._load_tt_vae_into_models(models)
        init_latent = lingbot_demo._encode_obs_ttnn(models, st, self._vae_obs)
        lingbot_demo._free_tt_vae_from_models(models)

        lingbot_demo._load_transformer_into_models(models, cfg)
        transformer = models["transformer"]
        cache_name = models["cache_name"]
        dtype = models["dtype"]
        device = models["device"]

        st["use_cfg"] = False
        st["frame_st_id"] = 0
        st["prompt_embeds"] = prompt_embeds
        st["negative_prompt_embeds"] = neg_embeds
        st["action_per_frame"] = cfg.action_per_frame
        st["action_mask"] = lingbot_demo._ttnn_action_channel_mask_vector(
            self.mesh_device, cfg.action_dim, cfg.used_action_channel_ids
        )
        st["actions_q01"] = lingbot_demo._ttnn_quantile_table_c11(self.mesh_device, cfg.norm_stat["q01"])
        st["actions_q99"] = lingbot_demo._ttnn_quantile_table_c11(self.mesh_device, cfg.norm_stat["q99"])
        st["action_norm_method"] = cfg.action_norm_method

        transformer.clear_cache(cache_name)
        patch_size = cfg.patch_size
        _b, _c, _tf, lh_act, lw_act = (int(x) for x in init_latent.shape)
        exp_lh, exp_lw = _robotwin_tshape_latent_hw(cfg.height, cfg.width)
        if lh_act != exp_lh or lw_act != exp_lw:
            raise RuntimeError(
                "VAE init_latent spatial dims do not match fused robotwin_tshape layout; "
                f"got ({lh_act}, {lw_act}), expected ({exp_lh}, {exp_lw})."
            )
        st["latent_height"] = lh_act
        st["latent_width"] = lw_act
        latent_token_per_chunk = (cfg.frame_chunk_size * lh_act * lw_act) // (
            patch_size[0] * patch_size[1] * patch_size[2]
        )
        action_token_per_chunk = cfg.frame_chunk_size * st["action_per_frame"]
        transformer.create_empty_cache(
            cache_name,
            cfg.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=dtype,
            device=device,
            batch_size=1,
        )

        spatial = self._tt_wan_spatial_noise
        latent_cond = init_latent[:, :, 0:1, :, :]
        input_dict = lingbot_demo._prepare_latent_input_ttnn(
            models,
            st,
            spatial,
            None,
            latent_t=0.0,
            action_t=0.0,
            latent_cond=latent_cond,
            action_cond=None,
            frame_st_id=0,
            patch_size=patch_size,
        )
        out = transformer(
            lingbot_demo._repeat_input_for_cfg_ttnn(models, st, input_dict["latent_res_lst"]),
            update_cache=1,
            cache_name=cache_name,
            action_mode=False,
            dump_iter=None,
        )
        lingbot_demo._free_tt_model(models, "transformer")
        return out

    def __call__(self, l1_input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Invoked by ``tt_cnn`` once per ``pipeline.compile`` and once per ``enqueue`` host tensor."""
        _ = l1_input_tensor
        return self.forward()

    @classmethod
    def prepare(
        cls,
        checkpoint_path: str | Path,
        message: dict | None = None,
        mesh_device: ttnn.MeshDevice | None = None,
        save_dir: str | Path | None = None,
        *,
        num_inference_steps: int | None = None,
        action_num_inference_steps: int | None = None,
        frame_chunk_size: int = 2,
        random_seed: int = 0,
        **kwargs,
    ) -> TtLingbotVA:
        """Factory compatible with tests; ``message`` and step kwargs are ignored (fixed random tensors in ``__init__``)."""
        _ = message, num_inference_steps, action_num_inference_steps, kwargs
        return cls(
            checkpoint_path,
            mesh_device,
            save_dir=save_dir,
            frame_chunk_size=frame_chunk_size,
            random_seed=random_seed,
        )
