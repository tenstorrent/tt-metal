# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end TTNN OCR model for dots.ocr (the ``ocr`` use case).

Composes the brought-up TTNN components — TtVisionTransformer (which
itself composes vision_patch_embed, vision_rmsnorm, vision_attention,
vision_mlp, vision_block and patch_merger), TtEmbedding (embedding),
TtDecoderLayer x28 (which composes text_rmsnorm, text_attention and
text_mlp), the stack-level TtTextRMSNorm (text_rmsnorm final norm) and
TtLMHead (lm_head) — into the DotsOCRForCausalLM pipeline:

    image patches -> vision tower -> merged vision embeddings
    token ids     -> text embedding --\
                                       splice at <|imgpad|> positions
    -> 28 x decoder layer -> final RMSNorm -> lm_head -> greedy AR loop

Host/device boundary (architecture_inventory hybrid_notes): image
preprocessing (resize/patchify/grid_thw) and the tokenizer/chat template
stay on the HF host path; rope tables, cu_seqlens and the causal mask are
computed on host and staged once per ``ocr()`` call. The vision-embedding
splice mirrors HF's ``masked_scatter`` and runs on host between the two
device stages.

AR loop: fixed-shape full-sequence recompute. The sequence buffer is
padded once to ``L = roundup(prompt + max_new_tokens, 32)`` and every
decode step re-runs the decoder stack over the same [1, 1, L, H] tensor
(constant shapes -> every step after the first hits the device program
cache; the plain causal mask makes pad rows unread and unattended). A
paged-KV-cache decode path is deliberately deferred to the perf phase —
parity first, then the traced single-token step (see skills/perf).
"""

import importlib.util
import sys
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

# Dir name contains a dot -> not importable as a package; load siblings by path.
_TT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _TT_DIR.parent


def _load_by_path(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _load_sibling(stem):
    return _load_by_path(f"dots_ocr_tt_{stem}", _TT_DIR / f"{stem}.py")


# Components used (use_case.components_used): vision_patch_embed,
# vision_rmsnorm, vision_attention, vision_mlp, vision_block and
# patch_merger are imported and composed inside vision_transformer;
# text_attention and text_mlp inside decoder_layer.
TtVisionTransformer = _load_sibling("vision_transformer").TtVisionTransformer
TtEmbedding = _load_sibling("embedding").TtEmbedding
TtTextRMSNorm = _load_sibling("text_rmsnorm").TtTextRMSNorm
TtDecoderLayer = _load_sibling("decoder_layer").TtDecoderLayer
TtLMHead = _load_sibling("lm_head").TtLMHead

wl = _load_by_path("dots_ocr_weight_loader", _TT_DIR / "weight_loader.py")
ref = _load_by_path("dots_ocr_reference_functional", _MODEL_DIR / "reference" / "functional.py")

IMAGE_TOKEN_ID = 151665
DEFAULT_EOS_IDS = (151643, 151673)  # generation_config.json eos_token_id
DEFAULT_PROMPT = "Extract the text content from the image."


class TtOCRModel(LightweightModule):
    """dots.ocr end-to-end OCR: vision tower + text decoder + greedy AR loop.

    Construction loads real HF checkpoint weights ONCE through
    tt/weight_loader.py; ``ocr()`` is re-entrant across calls.

    Args:
        mesh_device: ttnn mesh device handle (1xN line; vision replicated,
            decoder TP-sharded per the recorded parallelism plan).
        tokenizer: HF tokenizer for rednote-hilab/dots.ocr.
        image_processor: HF Qwen2VLImageProcessor for the checkpoint.
        chat_template: the checkpoint's chat template string.
        num_text_layers / num_vision_layers: production 28 / 42.
    """

    def __init__(
        self,
        mesh_device,
        tokenizer,
        image_processor,
        chat_template,
        num_text_layers=wl.TEXT_NUM_LAYERS,
        num_vision_layers=wl.VISION_NUM_BLOCKS,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.chat_template = chat_template
        self.spatial_merge_size = 2

        # Vision tower: fp32 residual stream, all weights replicated.
        self.vision = TtVisionTransformer(
            mesh_device,
            wl.vision_transformer_weights(num_layers=num_vision_layers),
            num_layers=num_vision_layers,
            num_heads=12,
            dtype=ttnn.float32,
        )
        # Text token embedding (embedding component): bf16 hidden-sharded table.
        self.embedding = TtEmbedding(mesh_device, wl.embedding_weights())
        # Decoder stack: fp32-mandatory attention path (Qwen2 attention sink).
        self.layers = [
            TtDecoderLayer(
                mesh_device,
                wl.decoder_layer_weights(layer_idx=i),
                num_heads=12,
                num_kv_heads=2,
                dtype=ttnn.float32,
            )
            for i in range(num_text_layers)
        ]
        # Stack-level final norm (text_rmsnorm component, model.norm).
        self.final_norm = TtTextRMSNorm(
            mesh_device, wl.text_rmsnorm_weights(which="final_norm"), dtype=ttnn.float32, eps=1e-6
        )
        # Untied vocab projection (lm_head component), vocab-sharded. fp32
        # weights instead of the block's bf8b default: greedy decode rides on
        # exact argmax and reduced-precision logits flip near-tie tokens (e2e
        # showed one subword flip vs HF at bf8b AND one at bf16 — fp32 logits
        # remove the quantization tie-break entirely); the perf phase owns speed.
        self.lm_head = TtLMHead(mesh_device, wl.lm_head_weights(), dtype=ttnn.float32)

    # ------------------------------------------------------------------
    # Host-side preprocessing (hybrid_notes: stays on the HF host path)
    # ------------------------------------------------------------------
    def preprocess(self, image, prompt=DEFAULT_PROMPT):
        """image + prompt -> (input_ids [1, S], pixel_values [P, 588], grid_thw [1, 3])."""
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.tokenizer.apply_chat_template(
            messages, chat_template=self.chat_template, add_generation_prompt=True, tokenize=False
        )
        vis = self.image_processor(images=[image], return_tensors="pt")
        grid_thw = vis["image_grid_thw"]
        t, h, w = grid_thw[0].tolist()
        merged = (t * h * w) // self.spatial_merge_size**2
        text = text.replace("<|imgpad|>", "<|imgpad|>" * merged)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        return input_ids, vis["pixel_values"].float(), grid_thw

    # ------------------------------------------------------------------
    # Device stage 1: vision tower (run once per image)
    # ------------------------------------------------------------------
    def encode_image(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """pixel_values [P, C*14*14] -> merged vision embeddings [P/4, 1536] (host fp32)."""
        seq, patch_dim = pixel_values.shape
        padded_seq = ((seq + 127) // 128) * 128
        x_pad = torch.cat([pixel_values, torch.zeros(padded_seq - seq, patch_dim)], dim=0)
        x_tt = ttnn.from_torch(
            x_pad.reshape(1, 1, padded_seq, patch_dim),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # Host rope tables + UNPADDED window boundaries (hybrid_notes).
        rope = ref.vision_rot_pos_emb(grid_thw, head_dim=128, spatial_merge_size=self.spatial_merge_size)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        rot_mats = self.vision.prepare_rope(rope, padded_seq)
        cu_tt = self.vision.prepare_cu_seqlens(cu_seqlens)
        out_tt = self.vision.forward(x_tt, rot_mats, cu_tt)
        ttnn.deallocate(x_tt)
        merged_seq = seq // self.spatial_merge_size**2
        out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()[:merged_seq]
        ttnn.deallocate(out_tt)
        return out

    # ------------------------------------------------------------------
    # Device stage 2 helpers
    # ------------------------------------------------------------------
    def _embed_tokens(self, ids: list[int]) -> torch.Tensor:
        """Token ids -> [len(ids), 1536] host fp32 via the TTNN embedding block."""
        n = len(ids)
        padded = ((n + 31) // 32) * 32
        ids_t = torch.zeros(1, 1, 1, padded, dtype=torch.int32)
        ids_t[0, 0, 0, :n] = torch.tensor(ids, dtype=torch.int32)
        ids_tt = ttnn.from_torch(
            ids_t,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        e_tt = self.embedding.forward(ids_tt)
        ttnn.deallocate(ids_tt)
        e = ttnn.to_torch(ttnn.get_device_tensors(e_tt)[0]).float().reshape(padded, -1)[:n]
        ttnn.deallocate(e_tt)
        return e

    def _decode_logits_row(self, embeds: torch.Tensor, rot_mats, causal_mask, row: int) -> torch.Tensor:
        """Run the decoder stack over [L, H] host embeds; return logits [vocab] at ``row``."""
        L, H = embeds.shape
        h = ttnn.from_torch(
            embeds.reshape(1, 1, L, H),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        for layer in self.layers:
            h = layer.forward(h, rot_mats, causal_mask)
        normed = self.final_norm(h)
        ttnn.deallocate(h)
        # lm_head only needs the tile window holding ``row``; keep it fp32 —
        # a bf16 typecast here rounds the hidden state and flips near-tie
        # argmax tokens (the "Tenstorrent"/"Tenstorment" e2e drift).
        start = (row // 32) * 32
        window = ttnn.slice(normed, [0, 0, start, 0], [1, 1, start + 32, H])
        ttnn.deallocate(normed)
        logits_tt = self.lm_head.forward(window)
        ttnn.deallocate(window)
        logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)).float()[
            0, 0, row - start
        ]
        ttnn.deallocate(logits_tt)
        return logits

    # ------------------------------------------------------------------
    # The use-case verb
    # ------------------------------------------------------------------
    def ocr(self, image, prompt=DEFAULT_PROMPT, max_new_tokens=32, eos_ids=DEFAULT_EOS_IDS):
        """Greedy OCR decode. Returns the generated text (special tokens stripped)."""
        input_ids, pixel_values, grid_thw = self.preprocess(image, prompt)
        tokens = input_ids[0].tolist()
        prompt_len = len(tokens)
        L = ((prompt_len + max_new_tokens + 31) // 32) * 32

        # Stage 1: vision tower, then host splice at <|imgpad|> positions
        # (HF masked_scatter equivalent).
        vision_embeds = self.encode_image(pixel_values, grid_thw)
        embeds = torch.zeros(L, vision_embeds.shape[-1])
        embeds[:prompt_len] = self._embed_tokens(tokens)
        img_pos = [i for i, t in enumerate(tokens) if t == IMAGE_TOKEN_ID]
        assert len(img_pos) == vision_embeds.shape[0], f"{len(img_pos)} != {vision_embeds.shape[0]}"
        embeds[torch.tensor(img_pos)] = vision_embeds

        # Constant per-call device inputs: rope tables + plain causal mask
        # (pad rows beyond the live length are causally unattendable).
        cos, sin = ref.text_rope_cos_sin(torch.arange(L).unsqueeze(0))
        rot_mats = self.layers[0].prepare_rope(cos, sin)
        causal_mask = self.layers[0].prepare_causal_mask(L)

        # Stage 2: greedy AR loop, fixed-shape full-sequence recompute.
        generated = []
        for _ in range(max_new_tokens):
            row = len(tokens) - 1
            logits = self._decode_logits_row(embeds, rot_mats, causal_mask, row)
            next_token = int(torch.argmax(logits).item())
            tokens.append(next_token)
            generated.append(next_token)
            if next_token in eos_ids:
                break
            embeds[len(tokens) - 1] = self._embed_tokens([next_token])[0]

        for t in (*rot_mats, causal_mask):
            ttnn.deallocate(t)
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
