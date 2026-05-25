# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU Code Predictor using HF Qwen3ForCausalLM with KV cache.

Runs the 5-layer Code Predictor on CPU in float32 for numerical accuracy.
The TT bfloat16 Code Predictor introduces enough error in CB1-15 tokens
to cause the Talker to diverge from the HF reference after ~8 decode steps.
"""

import glob

import torch
import torch.nn.functional as F
from loguru import logger
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3ForCausalLM


class CPUCodePredictor:
    """Code Predictor that runs on CPU using HF Qwen3ForCausalLM."""

    def __init__(self, model, proj_w, proj_b, lm_heads, codec_embeddings, num_code_groups=16):
        self.model = model
        self.proj_w = proj_w
        self.proj_b = proj_b
        self.lm_heads = lm_heads
        self.codec_embeddings = codec_embeddings
        self.num_cb_predict = num_code_groups - 1

    @classmethod
    def from_pretrained(cls, model_path: str):
        cfg = Qwen3Config(
            hidden_size=1024,
            intermediate_size=3072,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            num_hidden_layers=5,
            rms_norm_eps=1e-6,
            rope_theta=1000000,
            vocab_size=2048,
            max_position_embeddings=65536,
        )
        model = Qwen3ForCausalLM(cfg)

        snap_dirs = glob.glob(f"/root/.cache/huggingface/hub/models--{model_path.replace('/', '--')}/snapshots/*/")
        if not snap_dirs:
            from huggingface_hub import snapshot_download
            snapshot_download(model_path)
            snap_dirs = glob.glob(f"/root/.cache/huggingface/hub/models--{model_path.replace('/', '--')}/snapshots/*/")
        snap_dir = snap_dirs[0]

        full_sd = {}
        for f in sorted(glob.glob(snap_dir + "*.safetensors")):
            full_sd.update(load_file(f))

        cp_prefix = "talker.code_predictor.model."
        new_sd = {}
        for key in full_sd:
            if key.startswith(cp_prefix):
                suffix = key[len(cp_prefix):]
                if not suffix.startswith("codec_embedding"):
                    new_sd["model." + suffix] = full_sd[key].float()

        norm_key = "talker.code_predictor.model.norm.weight"
        if norm_key in full_sd:
            new_sd["model.norm.weight"] = full_sd[norm_key].float()

        new_sd["lm_head.weight"] = torch.zeros(2048, 1024)
        missing, _ = model.load_state_dict(new_sd, strict=False)
        logger.info(f"CPU Code Predictor: missing keys={len(missing)} (expected: embed_tokens)")
        model.eval()

        proj_w = full_sd["talker.code_predictor.small_to_mtp_projection.weight"].float()
        proj_b = full_sd["talker.code_predictor.small_to_mtp_projection.bias"].float()

        lm_heads = []
        for i in range(15):
            lm_heads.append(full_sd[f"talker.code_predictor.lm_head.{i}.weight"].float())

        codec_embeddings = []
        for i in range(15):
            codec_embeddings.append(
                full_sd[f"talker.code_predictor.model.codec_embedding.{i}.weight"].float()
            )

        return cls(model, proj_w, proj_b, lm_heads, codec_embeddings)

    @torch.no_grad()
    def predict_codebooks(
        self,
        talker_hidden,
        cb0_token,
        talker_codec_embedding_weight,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    ):
        """Generate CB1-CB15 using HF KV-cache approach on CPU.

        Args:
            talker_hidden: [B, 1, 2048] pre-norm hidden from Talker
            cb0_token: [B] CB0 token ID
            talker_codec_embedding_weight: [vocab, 2048] Talker's codec embedding

        Returns:
            [B, 16] CB0 + CB1..CB15
        """
        B = talker_hidden.shape[0]
        th = talker_hidden.float()
        ce = talker_codec_embedding_weight.float()

        cb0_emb = F.embedding(cb0_token.unsqueeze(-1), ce)
        context = torch.cat([th, cb0_emb], dim=1)
        projected = F.linear(context, self.proj_w, self.proj_b)

        out = self.model.model(inputs_embeds=projected, use_cache=True)
        hidden = out.last_hidden_state
        kv_cache = out.past_key_values

        generated = [cb0_token.unsqueeze(-1)]

        last_h = hidden[:, -1:, :]
        logits = F.linear(last_h, self.lm_heads[0])
        tok = self._sample(logits[:, -1, :], temperature, top_k, top_p)
        generated.append(tok.unsqueeze(-1))

        for step in range(1, self.num_cb_predict):
            prev_emb = F.embedding(generated[-1], self.codec_embeddings[step - 1])
            prev_proj = F.linear(prev_emb, self.proj_w, self.proj_b)

            out = self.model.model(inputs_embeds=prev_proj, past_key_values=kv_cache, use_cache=True)
            hidden = out.last_hidden_state
            kv_cache = out.past_key_values

            logits = F.linear(hidden[:, -1:, :], self.lm_heads[step])
            tok = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            generated.append(tok.unsqueeze(-1))

        return torch.cat(generated, dim=-1)

    @staticmethod
    def _sample(logits, temperature, top_k, top_p):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
