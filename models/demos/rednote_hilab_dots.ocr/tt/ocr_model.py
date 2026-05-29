# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN end-to-end OCR pipeline for rednote-hilab/dots.ocr (image -> text).

This is the ``ocr`` use_case assembly. It does NOT re-implement any block maths --
it imports and COMPOSES the already-verified TTNN component modules:

    vision_patch_embed  (host Conv2d+RMSNorm, the documented host boundary)
    vision_rmsnorm      (post-trunk / per-block RMSNorm)
    vision_attention    (fused-QKV 2D-RoPE attention)
    vision_mlp          (SwiGLU FFN)
    vision_block         (pre-norm residual vision block)
    vision_patch_merger  (LayerNorm + GELU MLP, 2x2 merge)
    vision_tower         (full DotsVisionTransformer assembly)  -- tt/vision_tower.py
    embedding            (LM token-embedding gather)
    rmsnorm              (Qwen2RMSNorm)
    rope                 (1D rotary tables)
    attention            (GQA self-attention)
    mlp                  (SwiGLU FFN)
    decoder_layer        (pre-norm residual decoder block)
    lm_head              (untied hidden->vocab projection)
    language_model       (full Qwen2 trunk assembly)            -- tt/language_model.py

Pipeline (mirrors HF ``DotsOCRForCausalLM``):

    pixel_values --(host patch_embed)--> patch tokens --(TtVisionTower)--> vision_embeds
    input_ids --(embed_tokens)--> text_embeds
    inputs_embeds = masked_scatter(text_embeds, vision_embeds @ <|imgpad|> positions)
    logits = language_model_trunk(inputs_embeds)          # embed -> N layers -> norm -> lm_head
    next_token = argmax(logits[-1])                        # greedy decode

The vision->text masked_scatter (at ``config.image_token_id`` = 151665) is the
host/glue op documented in the use_case ``hybrid_notes``; everything heavy
(vision trunk + LM trunk) runs on device.

AR decode: the verified ``TtLanguageModel`` bakes the causal mask / RoPE tables
into its decoder layers at the construction ``seq_len`` and carries NO KV cache
(it is the validated full-causal assembly). For greedy AR generation we therefore
re-instantiate the LM trunk per total sequence length and run a full forward,
taking the final position's logits. This is correct (it is exactly the HF
no-cache forward) and is acceptable for the short OCR generations validated here.

The model dir name (rednote_hilab_dots.ocr) contains a dot, so sibling modules
are imported by file path with importlib (the project convention).
"""
import importlib.util
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

_TT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_TT_DIR, file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Compose (import, never copy) the verified component assemblies.
_vt = _load_sibling("dots_ocr_vision_tower", "vision_tower.py")
_lm = _load_sibling("dots_ocr_language_model", "language_model.py")
_loader_mod = _load_sibling("dots_ocr_weight_loader", "weight_loader.py")
_kvc = _load_sibling("dots_ocr_kv_cache", "kv_cache.py")

TtVisionTower = _vt.TtVisionTower
TtLanguageModel = _lm.TtLanguageModel
SelfAttentionKVCache = _kvc.SelfAttentionKVCache
load_vision_tower_weights = _loader_mod.load_vision_tower_weights
load_language_model_weights = _loader_mod.load_language_model_weights
load_vision_patch_embed_weights = _loader_mod.load_vision_patch_embed_weights

IMAGE_TOKEN_ID = 151665  # config.image_token_id (<|imgpad|>)


class TtOcrModel(LightweightModule):
    """End-to-end TTNN OCR model: image + prompt tokens -> text token logits.

    Composes :class:`TtVisionTower` (vision trunk) and :class:`TtLanguageModel`
    (Qwen2 trunk). The vision embeddings are scattered into the text embedding
    stream at the image-token positions (host masked_scatter), then the LM trunk
    runs on device. Greedy AR decode is driven by :meth:`generate`.

    Args:
        device: ttnn Device or MeshDevice.
        lm_state_dict: flat LM-trunk state_dict (load_language_model_weights).
        vision_state_dict: flat vision-tower state_dict (load_vision_tower_weights).
        grid_thw: torch.Tensor [num_images, 3] patch grid (t, h, w).
        lm_num_layers / vision_num_layers: depth to assemble.
        hidden_size: 1536. num_heads/num_kv_heads/head_dim: 12/2/128.
        rope_theta 1e6, eps 1e-6, vocab_size 151936.
    """

    def __init__(
        self,
        device,
        lm_state_dict,
        vision_state_dict,
        grid_thw,
        lm_num_layers: int,
        vision_num_layers: int,
        hidden_size: int = 1536,
        num_heads: int = 12,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
        eps: float = 1e-6,
        bias: bool = True,
        vocab_size: int = 151936,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.lm_state_dict = {k: v.to(torch.float32) for k, v in lm_state_dict.items()}
        self.vision_state_dict = vision_state_dict
        self.grid_thw = torch.as_tensor(grid_thw)
        self.lm_num_layers = lm_num_layers
        self.vision_num_layers = vision_num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.eps = eps
        self.bias = bias
        self.vocab_size = vocab_size
        self.dtype = dtype

        # The LM input token-embedding table (kept on host for the scatter glue).
        self._embed_table = self.lm_state_dict["embed_tokens.weight"].to(torch.float32)

        # Build the vision tower once (image is processed once per generation).
        self.vision_tower = TtVisionTower(
            device=device,
            state_dict=self.vision_state_dict,
            grid_thw=self.grid_thw,
            num_layers=vision_num_layers,
            dtype=dtype,
        )

        # Cache LM trunks per total seq_len (RoPE/causal mask baked at build).
        self._lm_cache = {}

    # -- vision -------------------------------------------------------------
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values -> merged vision embeddings [num_vis_tokens, hidden] (torch)."""
        patch_tokens = self.vision_tower.patch_embed(pixel_values)  # host Conv2d+RMSNorm
        tt_in = ttnn.from_torch(
            patch_tokens.to(torch.float32),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_out = self.vision_tower(tt_in)
        vis = ttnn.to_torch(tt_out).to(torch.float32)
        return vis.reshape(-1, self.hidden_size)

    # -- embedding + scatter (host glue, per hybrid_notes) ------------------
    def build_inputs_embeds(self, input_ids: torch.Tensor, vision_embeds: torch.Tensor) -> torch.Tensor:
        """Embed text tokens and masked_scatter vision embeds at image positions.

        Mirrors DotsOCRForCausalLM.prepare_inputs_embeds: gather text embeds, then
        scatter the vision tower output into the <|imgpad|> (image_token_id 151665)
        slots. Returns inputs_embeds [seq, hidden] (torch fp32).
        """
        ids = input_ids.reshape(-1).to(torch.int64)
        text_embeds = self._embed_table[ids]  # [seq, hidden]
        img_mask = ids == IMAGE_TOKEN_ID
        n_img = int(img_mask.sum())
        if n_img > 0:
            assert vision_embeds.size(0) >= n_img, f"vision_embeds {vision_embeds.size(0)} < img slots {n_img}"
            text_embeds = text_embeds.clone()
            text_embeds[img_mask] = vision_embeds[:n_img].to(text_embeds.dtype)
        return text_embeds

    # -- LM trunk on device -------------------------------------------------
    def _lm_for_seq(self, seq_len: int, max_seq_len: int = None):
        key = (seq_len, max_seq_len)
        lm = self._lm_cache.get(key)
        if lm is None:
            lm = TtLanguageModel(
                device=self.device,
                state_dict=self.lm_state_dict,
                num_layers=self.lm_num_layers,
                seq_len=seq_len,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                rope_theta=self.rope_theta,
                eps=self.eps,
                bias=self.bias,
                dtype=self.dtype,
                max_seq_len=max_seq_len if max_seq_len is not None else seq_len,
            )
            self._lm_cache[key] = lm
        return lm

    def lm_logits(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """inputs_embeds [seq, hidden] (torch) -> logits [seq, vocab] (torch).

        Runs the verified LM trunk on device starting from embeddings (skipping
        embed_tokens, which already happened in the host scatter). Reuses the
        TtLanguageModel decoder stack / final norm / lm_head modules directly.
        """
        seq_len = inputs_embeds.shape[0]
        lm = self._lm_for_seq(seq_len)
        hidden = ttnn.from_torch(
            inputs_embeds.to(torch.float32),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for layer in lm.layers:
            hidden = layer(hidden)
        hidden = lm.norm(hidden)
        logits = lm.lm_head(hidden)
        return ttnn.to_torch(logits).to(torch.float32).reshape(seq_len, self.vocab_size)

    def _embed_token(self, token_id: int) -> ttnn.Tensor:
        """Embed a single token id -> device tensor [1, hidden] (TILE)."""
        vec = self._embed_table[int(token_id)].reshape(1, self.hidden_size).to(torch.float32)
        return ttnn.from_torch(
            vec, device=self.device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 16,
        eos_token_ids=(151643, 151673),
    ):
        """Greedy AR decode with a KV cache: image + prompt -> token ids (list[int]).

        Pipeline:
          1. vision tower -> masked_scatter -> inputs_embeds [prompt_len, hidden].
          2. PREFILL: one full-causal forward over the prompt that ALSO writes
             every layer's K/V into a :class:`SelfAttentionKVCache`. The last
             position's logits give the first generated token.
          3. DECODE: per step, embed the single new token, run ONE cached decode
             step (O(1) layer-runs via flash-decode SDPA against the cache, GQA
             handled by the op), argmax. The cache makes each step O(1) instead
             of re-running the whole O(N) trunk -- this is the perf win that
             unlocks full-depth generation.
        """
        eos = set(eos_token_ids)
        vision_embeds = self.encode_image(pixel_values)
        ids = input_ids.reshape(-1).to(torch.int64).tolist()
        prompt_len = len(ids)
        max_seq_len = prompt_len + max_new_tokens + 1

        lm = self._lm_for_seq(prompt_len, max_seq_len=max_seq_len)
        kv_cache = SelfAttentionKVCache(
            device=self.device,
            num_layers=self.lm_num_layers,
            batch=1,
            num_kv_heads=self.num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=self.head_dim,
            dtype=self.dtype,
        )

        # ---- Prefill: scatter vision embeds, run trunk, populate cache. ----
        cur = torch.tensor(ids, dtype=torch.int64)
        inputs_embeds = self.build_inputs_embeds(cur, vision_embeds)  # [prompt_len, hidden]
        hidden = ttnn.from_torch(
            inputs_embeds.to(torch.float32),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = lm.prefill_from_embeds(hidden, kv_cache)
        logits = ttnn.to_torch(logits).to(torch.float32).reshape(-1)  # last-position logits [vocab]
        next_id = int(torch.argmax(logits).item())

        generated = [next_id]
        ids.append(next_id)
        if next_id in eos:
            return generated

        # ---- Cached decode: O(1) per step. ----
        for step in range(1, max_new_tokens):
            pos = prompt_len + step - 1  # sequence index of the token being fed
            tok_embed = self._embed_token(next_id)  # [1, hidden]
            step_logits = lm.decode_step(tok_embed, pos, kv_cache)
            step_logits = ttnn.to_torch(step_logits).to(torch.float32).reshape(self.vocab_size)
            next_id = int(torch.argmax(step_logits).item())
            generated.append(next_id)
            ids.append(next_id)
            if next_id in eos:
                break
        return generated
