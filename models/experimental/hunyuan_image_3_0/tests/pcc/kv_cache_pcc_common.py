# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared fixtures/helpers for the HunyuanImage-3.0 KV-cache PCC tests
# (test_kv_cache_prefill.py, test_kv_cache_decode.py).
#
# Mirrors the devstral2 decoder_pcc_common.py split (prefill vs decode, PCC-gated
# vs a full-precision reference) but adapted to the Hunyuan backbone's KV-cache
# path (tt/model.py forward(kv_cache=..., decode_step=...), fed by tt/generate.py):
#
#   * PREFILL — one full-sequence forward at ISL S with use_cache=True; this
#     populates the per-layer K/V cache. Scored against the fp32 host reference at
#     the same S (mathematically the non-cached full forward).
#
#   * DECODE — after a prefill at MAX_ISL, single-token steps that append to the
#     cache. Teacher-forced with reference-greedy tokens (like tt_transformers
#     test_model.py) so the TT decode never drifts off the reference trajectory;
#     each step's [1, H] hidden and [1, V] logits are scored against a fresh fp32
#     full-sequence forward at that length.
#
# Both granularities are reported: the backbone hidden state (post-ln_f, the KV
# cache's own output) AND the LM-head logits ("check the pcc of logits as well").
#
# Env knobs:
#   HY_NUM_LAYERS   backbone layers to build (default 2 for a fast gate).
#   HY_MAX_ISL      max input sequence length to prefill/decode at (default 256).
#   HY_DECODE_STEPS decode steps to run after the MAX_ISL prefill (default 8).

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
import ttnn

from models.experimental.hunyuan_image_3_0.ref.attention.mask import (
    build_attention_mask,
    build_attention_mask_query_row,
    to_additive,
)
from models.experimental.hunyuan_image_3_0.ref.attention.rms_norm import HunyuanRMSNorm
from models.experimental.hunyuan_image_3_0.ref.lm_head import lm_head_logits
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer, prepare_recaption_inputs
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, load_tensors
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tt.kv_cache import HunyuanTtKvCache
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel

from hunyuan_image_3.modeling_hunyuan_image_3 import build_batch_2d_rope

PROMPT = "a cat on a mat"
BOT_TASK = "recaption"
SEQUENCE_TEMPLATE = "instruct"


def _hf_max_isl() -> int:
    """Max input sequence length as defined by the HF checkpoint: the backbone's
    ``max_position_embeddings`` (== ``generation_config.max_length`` for this model).
    This is the true ISL ceiling — the RoPE tables / position ids are only defined
    below it — so every ISL used here is clamped to it."""
    return int(json.load(open(h.I2I_WEIGHTS / "config.json"))["max_position_embeddings"])


HF_MAX_ISL = _hf_max_isl()  # 22800 for HunyuanImage-3.0

NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))
# Default the gate ISL to a practical length; the full HF ceiling is exercised by
# the @slow sweep. Never exceed HF_MAX_ISL (positions past it are undefined).
MAX_ISL = min(int(os.environ.get("HY_MAX_ISL", "512")), HF_MAX_ISL)
DECODE_GENERATION_LENGTH = int(os.environ.get("HY_DECODE_STEPS", "8"))

# Deep layer counts cannot hold every expert resident (64 experts x 32 layers is
# ~77B params). Stream them (load per-expert during forward) and build/free the
# fp32 reference one layer at a time so neither side holds all NUM_LAYERS at once.
# Default on past a handful of layers; overridable. Small runs keep experts
# resident (faster, and matches test_recaption_on_device).
STREAM_EXPERTS = os.environ.get("HY_STREAM_EXPERTS", "1" if NUM_LAYERS > 8 else "0") != "0"


def _sweep_seq_lengths() -> list[int]:
    """Powers of two from 128 up to (but not past) HF_MAX_ISL, then HF_MAX_ISL itself."""
    lens, s = [], 128
    while s < HF_MAX_ISL:
        lens.append(s)
        s *= 2
    lens.append(HF_MAX_ISL)
    return lens


# Prefill ISLs to sweep. Sanity = a short length plus MAX_ISL (fast CI gate); the
# sweep climbs to the HF ceiling. Text-only recaption uses a pure causal mask, so
# any ISL >= the real prompt prefix is valid (the prompt is padded with a token).
PREFILL_SANITY_SEQ_LENGTHS = sorted({128, MAX_ISL})
PREFILL_SWEEP_SEQ_LENGTHS = _sweep_seq_lengths()

# Chained multi-layer bf16 accumulation loosens with depth (see
# test_model_teacher_forced.py). Match test_recaption_on_device's tiers.
PCC_REQUIRED = 0.95 if NUM_LAYERS <= 2 else 0.85


def _pad_ids_to(ids: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a [1, S] id row to [1, target_len] by repeating the last token."""
    cur = int(ids.shape[1])
    if target_len <= cur:
        return ids[:, :target_len]
    pad = ids[:, -1:].repeat(1, target_len - cur)
    return torch.cat([ids, pad], dim=1)


class KvCachePccContext:
    """Backbone + LM head + fp32 reference, built once and shared across ISLs/steps."""

    def __init__(self, device):
        self.device = device
        self.c = h.model_cfg()

        tok = HunyuanTokenizer.from_model_dir(INSTRUCT_MODEL_DIR, sequence_template=SEQUENCE_TEMPLATE)
        bundle = prepare_recaption_inputs(tok, PROMPT, bot_task=BOT_TASK, sequence_template=SEQUENCE_TEMPLATE)
        self.prompt_ids = bundle.input_ids  # [1, prefix_len]

        self.wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
        self.lm_w = load_tensors(INSTRUCT_MODEL_DIR, ["lm_head.weight"])["lm_head.weight"].float()
        self.ln_f_w = h.load_tensor("model.ln_f.weight")

        # Text-only recaption: pure causal mask, standard 1D-ish 2D-RoPE (no images).
        self.image_infos = None
        self.attn_slices = [[]]

        layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in h.load_prefix(f"model.layers.{i}").items()}
        # sp_factor=1 is REQUIRED for the KV-cache path (tt/generate.py): every device
        # must see the full K/V sequence. bf16 weights so the gate reflects the KV
        # cache / RoPE-slice logic, not weight quantization.
        self.backbone = HunyuanTtModel(
            device,
            num_layers=NUM_LAYERS,
            hidden_size=self.c["H"],
            num_heads=self.c["HEADS"],
            num_kv_heads=self.c["KV_HEADS"],
            head_dim=self.c["HEAD_DIM"],
            num_experts=self.c["NUM_EXPERTS"],
            moe_topk=self.c["MOE_TOPK"],
            use_qk_norm=self.c["USE_QK_NORM"],
            use_mixed_mlp_moe=self.c["USE_MIXED"],
            norm_topk_prob=self.c["NORM_TOPK"],
            rms_norm_eps=self.c["EPS"],
            stream_experts=STREAM_EXPERTS,
            layer_loader=layer_loader,
            embed_state_dict={"model.wte.weight": self.wte},
            norm_state_dict={"model.ln_f.weight": self.ln_f_w},
            apply_final_norm=True,
            weight_dtype=ttnn.bfloat16,
            sp_factor=1,
        )
        self.lm_head = HunyuanTtLMHead(device, {"lm_head.weight": h.load_tensor("lm_head.weight")})

        # fp32 reference stack (full-sequence forward each call). When streaming,
        # layers are built/freed per forward (never all NUM_LAYERS fp32 at once).
        self.wte_f = self.wte.float()
        self.ref_layers = None if STREAM_EXPERTS else [h.ref_layer(self.c, i) for i in range(NUM_LAYERS)]
        self.ref_ln_f = HunyuanRMSNorm(self.c["H"], eps=self.c["EPS"])
        self.ref_ln_f.weight.data = self.ln_f_w.float()

    def _ref_layer_forward(self, i, hidden, mask, cos, sin):
        if self.ref_layers is not None:
            return self.ref_layers[i](hidden, attention_mask=mask, custom_pos_emb=(cos, sin))
        # Streaming: build layer i fresh, forward, free it (bypasses h.ref_layer's cache).
        import gc

        from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer

        sd = h.load_prefix(f"model.layers.{i}")
        layer = RefLayer(
            hidden_size=self.c["H"],
            num_attention_heads=self.c["HEADS"],
            num_key_value_heads=self.c["KV_HEADS"],
            attention_head_dim=self.c["HEAD_DIM"],
            num_experts=self.c["NUM_EXPERTS"],
            moe_topk=self.c["MOE_TOPK"],
            moe_intermediate_size=self.c["MOE_INTER"],
            num_shared_expert=self.c["NUM_SHARED"],
            use_mixed_mlp_moe=self.c["USE_MIXED"],
            norm_topk_prob=self.c["NORM_TOPK"],
            use_qk_norm=self.c["USE_QK_NORM"],
            rms_norm_eps=self.c["EPS"],
            layer_idx=i,
        )
        layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
        layer.eval()
        out = layer(hidden, attention_mask=mask, custom_pos_emb=(cos, sin))
        del layer, sd
        gc.collect()
        return out

    # -- reference (fp32, non-cached full forward) --------------------------
    @torch.no_grad()
    def reference_forward(self, ids: torch.Tensor):
        """fp32 host forward over NUM_LAYERS + ln_f + lm_head.

        Returns (hidden_last [1, H], logits_last [1, V]) at the final position.
        """
        S = int(ids.shape[1])
        hidden = F.embedding(ids.long(), self.wte_f)
        mask = to_additive(build_attention_mask(S, self.attn_slices, bsz=1)).reshape(1, 1, S, S)
        cos, sin = build_batch_2d_rope(
            image_infos=self.image_infos,
            seq_len=S,
            n_elem=self.c["HEAD_DIM"],
            device=hidden.device,
        )
        for i in range(NUM_LAYERS):
            hidden = self._ref_layer_forward(i, hidden, mask, cos, sin)
        hidden = self.ref_ln_f(hidden)
        logits = lm_head_logits(hidden, self.lm_w)
        return hidden[:, -1, :], logits[:, -1, :]

    # -- TTNN KV-cache path -------------------------------------------------
    def _upload_embeds(self, ids: torch.Tensor) -> ttnn.Tensor:
        emb = F.embedding(ids.long(), self.wte_f)
        return ttnn.from_torch(
            emb,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _upload_mask_full(self, S: int) -> ttnn.Tensor:
        mask_add = to_additive(build_attention_mask(S, self.attn_slices, bsz=1), dtype=torch.bfloat16).reshape(
            1, 1, S, S
        )
        return ttnn.from_torch(
            mask_add,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _upload_mask_row(self, query_pos: int, total_len: int) -> ttnn.Tensor:
        mask_add = to_additive(
            build_attention_mask_query_row(total_len, query_pos, self.attn_slices, bsz=1),
            dtype=torch.bfloat16,
        ).reshape(1, 1, 1, total_len)
        return ttnn.from_torch(
            mask_add,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _to_logits(self, hidden_tt: ttnn.Tensor):
        logits_tt = self.lm_head(hidden_tt, last_token_only=True)
        logits = ttnn.to_torch(logits_tt).float().squeeze(1)  # [1, V]
        ttnn.deallocate(logits_tt)
        return logits

    def new_kv_state(self, max_cache_len: int):
        """Fresh KV cache + full cos/sin tables sized for the whole run."""
        rope = self.backbone.layers[0].self_attn.rope
        cos_full, sin_full = rope.prepare_cos_sin(max_cache_len, image_infos=self.image_infos)
        return {
            "kv": HunyuanTtKvCache(len(self.backbone.layers)),
            "cos": cos_full,
            "sin": sin_full,
        }

    def free_kv_state(self, state):
        state["kv"].clear()
        ttnn.deallocate(state["cos"])
        ttnn.deallocate(state["sin"])

    @torch.no_grad()
    def prefill(self, state, ids: torch.Tensor):
        """KV-cache prefill at ISL S. Returns (hidden_last [1, H], logits_last [1, V])."""
        S = int(ids.shape[1])
        hidden_tt = self._upload_embeds(ids)
        cos_tt = ttnn.slice(state["cos"], [0, 0, 0, 0], [1, 1, S, state["cos"].shape[-1]])
        sin_tt = ttnn.slice(state["sin"], [0, 0, 0, 0], [1, 1, S, state["sin"].shape[-1]])
        mask_tt = self._upload_mask_full(S)
        hidden = self.backbone.forward(
            inputs_embeds=hidden_tt,
            seq_len=S,
            image_infos=self.image_infos,
            attention_mask=mask_tt,
            kv_cache=state["kv"],
            use_cache=True,
            decode_step=False,
            cos_sin=(cos_tt, sin_tt),
        )
        state["kv"].seq_len = S
        # Read the backbone hidden BEFORE lm_head: with last_token_only the head
        # slices the last position, and for an S==1 tensor that slice aliases the
        # input buffer, which the head then deallocates.
        hidden_last = ttnn.to_torch(hidden)[:, -1, :].float()
        logits = self._to_logits(hidden)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mask_tt)
        ttnn.deallocate(hidden_tt)
        return hidden_last, logits

    @torch.no_grad()
    def decode(self, state, ids: torch.Tensor):
        """One decode step for the last token of `ids` (prefill must have run).

        Returns (hidden_last [1, H], logits_last [1, V]) for the query position.
        """
        S = int(ids.shape[1])
        query_pos = S - 1
        hidden_tt = self._upload_embeds(ids[:, -1:])
        rope = self.backbone.layers[0].self_attn.rope
        cos_tt, sin_tt = rope.slice_cos_sin(state["cos"], state["sin"], query_pos)
        mask_tt = self._upload_mask_row(query_pos, S)
        hidden = self.backbone.forward(
            inputs_embeds=hidden_tt,
            seq_len=S,
            image_infos=self.image_infos,
            attention_mask=mask_tt,
            kv_cache=state["kv"],
            use_cache=True,
            decode_step=True,
            cos_sin=(cos_tt, sin_tt),
        )
        # Read hidden before lm_head: last_token_only slices the last position and
        # for this S==1 decode tensor that slice aliases the buffer the head frees.
        hidden_last = ttnn.to_torch(hidden)[:, -1, :].float()
        logits = self._to_logits(hidden)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mask_tt)
        ttnn.deallocate(hidden_tt)
        return hidden_last, logits


def build_context(device) -> KvCachePccContext:
    return KvCachePccContext(device)
