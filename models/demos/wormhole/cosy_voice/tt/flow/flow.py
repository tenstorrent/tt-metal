# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice3 Flow Decoder — top-level orchestrator.

Implements CausalMaskedDiffWithDiT which:
1. Embeds speech tokens and projects through PreLookaheadLayer
2. Projects speaker embeddings
3. Runs 10-step Euler ODE with CFG via DiT estimator
4. Outputs mel spectrogram features

All host-side except for DiT block evaluations which run on device.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.wormhole.cosy_voice.tt.flow.dit import TtDiT


class TtCausalMaskedDiffWithDiT(LightweightModule):
    """
    Top-level flow decoder for CosyVoice3.

    Inference flow:
        1. Embed speech tokens → PreLookaheadLayer → repeat_interleave(2×)
        2. Normalize + project speaker embedding (192→80)
        3. Build conditioning (prompt mel features)
        4. Run 10-step Euler ODE with classifier-free guidance
        5. Return generated mel features (excluding prompt portion)
    """

    def __init__(self, device, state_dict, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device

        # --- Config (from cosyvoice3.yaml) ---
        self.input_size = 80
        self.output_size = 80
        self.vocab_size = 6561
        self.token_mel_ratio = 2
        self.pre_lookahead_len = 3
        self.inference_cfg_rate = 0.7
        self.n_timesteps = 10

        # --- Host-side modules ---

        # input_embedding: Embedding(6561, 80)
        self.input_embedding = nn.Embedding(self.vocab_size, self.input_size)
        self.input_embedding.load_state_dict({"weight": state_dict["input_embedding.weight"]})

        # spk_embed_affine_layer: Linear(192, 80)
        self.spk_embed_affine_layer = nn.Linear(192, self.output_size)
        self.spk_embed_affine_layer.load_state_dict(
            {
                "weight": state_dict["spk_embed_affine_layer.weight"],
                "bias": state_dict["spk_embed_affine_layer.bias"],
            }
        )

        # PreLookaheadLayer: Conv1d(80,1024,k=4) → LeakyReLU → Conv1d(1024,80,k=3) + residual
        self.pll_conv1_w = state_dict["pre_lookahead_layer.conv1.weight"]
        self.pll_conv1_b = state_dict["pre_lookahead_layer.conv1.bias"]
        self.pll_conv2_w = state_dict["pre_lookahead_layer.conv2.weight"]
        self.pll_conv2_b = state_dict["pre_lookahead_layer.conv2.bias"]

        # Fixed random noise (from CausalConditionalCFM — deterministic for reproducibility)
        torch.manual_seed(0)
        self.rand_noise = torch.randn(1, 80, 50 * 300)

        # --- Device-side module ---
        logger.info("Loading DiT estimator...")
        self.dit = TtDiT(device, state_dict, dtype=dtype)
        logger.info("Flow decoder initialized")

    def _pre_lookahead_forward(self, inputs):
        """Host-side PreLookaheadLayer."""
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        outputs = F.leaky_relu(F.conv1d(outputs, self.pll_conv1_w, self.pll_conv1_b))
        outputs = F.pad(outputs, (self.pll_conv2_w.shape[2] - 1, 0), mode="constant", value=0.0)
        outputs = F.conv1d(outputs, self.pll_conv2_w, self.pll_conv2_b)
        outputs = outputs.transpose(1, 2).contiguous()
        return outputs + inputs

    def _solve_euler(self, z, mu, mask, spks, cond):
        """
        10-step Euler ODE solver with classifier-free guidance.

        At each step, evaluates the DiT estimator with batch=2:
        - Index 0: conditional (with mu, spks, cond)
        - Index 1: unconditional (zeros)
        Then blends: dphi = (1 + cfg_rate) * cond - cfg_rate * uncond
        """
        n = self.n_timesteps
        t_span = torch.linspace(0, 1, n + 1, device=z.device, dtype=z.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * math.pi)  # cosine schedule

        x = z.clone()
        t, dt = t_span[0].unsqueeze(0), t_span[1] - t_span[0]
        seq_len = x.size(2)

        # Pre-allocate batch=2 tensors for CFG
        x_in = torch.zeros(2, 80, seq_len, dtype=z.dtype)
        mask_in = torch.zeros(2, 1, seq_len, dtype=z.dtype)
        mu_in = torch.zeros(2, 80, seq_len, dtype=z.dtype)
        t_in = torch.zeros(2, dtype=z.dtype)
        spks_in = torch.zeros(2, 80, dtype=z.dtype)
        cond_in = torch.zeros(2, 80, seq_len, dtype=z.dtype)

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t
            spks_in[0] = spks
            cond_in[0] = cond

            dphi_dt = self.dit(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_cond, dphi_uncond = torch.split(dphi_dt, [1, 1], dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_cond - self.inference_cfg_rate * dphi_uncond

            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]

        return x.float()

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        streaming=False,
        finalize=True,
    ):
        """
        Run flow decoder inference.

        Args:
            token: (1, seq) - target speech tokens
            token_len: (1,) - target length
            prompt_token: (1, prompt_seq) - prompt speech tokens
            prompt_token_len: (1,) - prompt length
            prompt_feat: (1, mel_frames, 80) - prompt mel features
            prompt_feat_len: (1,) - prompt mel length
            embedding: (1, 192) - speaker embedding
            streaming: bool - streaming mode (unused for now)
            finalize: bool - final chunk flag

        Returns:
            feat: (1, 80, generated_mel_frames) - generated mel features
            cache: None
        """
        assert token.shape[0] == 1, "Batch size must be 1"

        # 1. Speaker embedding projection
        spk = F.normalize(embedding, dim=1)
        spk = self.spk_embed_affine_layer(spk)  # (1, 80)

        # 2. Embed and encode speech tokens
        all_tokens = torch.cat([prompt_token, token], dim=1)
        all_token_len = prompt_token_len + token_len
        mask = torch.ones(1, 1, all_tokens.shape[1]).to(spk)
        token_emb = self.input_embedding(torch.clamp(all_tokens, min=0)) * mask.transpose(1, 2)

        # 3. PreLookaheadLayer
        h = self._pre_lookahead_forward(token_emb)

        # 4. Upsample by token_mel_ratio (repeat_interleave)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1

        # 5. Build conditioning
        conds = torch.zeros(1, mel_len1 + mel_len2, self.output_size, dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)  # (1, 80, total_mel_len)

        # 6. Build mask and mu
        mu = h.transpose(1, 2).contiguous()  # (1, 80, total_mel_len)
        mask = torch.ones(1, 1, mel_len1 + mel_len2).to(h)

        # 7. Initialize noise
        z = self.rand_noise[:, :, : mu.size(2)].to(mu.device).to(mu.dtype)

        # 8. Run Euler ODE solver
        feat = self._solve_euler(z, mu, mask, spk.squeeze(0), conds)

        print(
            f"DEBUG: Full feat stats before slicing: min={feat.min().item():.4f}, max={feat.max().item():.4f}, mean={feat.mean().item():.4f}"
        )

        # 9. Extract generated portion (exclude prompt)
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None

    @classmethod
    def from_pretrained(cls, weights_dir, device, dtype=ttnn.bfloat16):
        """Load from pretrained weights directory."""
        import os

        flow_path = os.path.join(weights_dir, "flow.pt")
        logger.info(f"Loading flow weights from {flow_path}")
        state_dict = torch.load(flow_path, map_location="cpu")
        return cls(device, state_dict, dtype)
