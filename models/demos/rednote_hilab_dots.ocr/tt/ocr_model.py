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

AR loop (decode_strategy=kv_cache): prefill runs ONCE over the padded
prompt ``P32 = roundup(prompt_len, 32)`` and populates persistent per-layer
fp32 K/V caches (``ttnn.fill_cache`` inside text_attention); every decode
step then processes ONE token row [1, 1, 1, H] against the caches
(``ttnn.experimental.paged_update_cache`` + chip-local fp32 single-row
attention; no CCL in attention — the KV cache shards with the TP plan, Q
heads 4-way, kv_replication=2 so each chip's Q-head group holds a full
copy of its KV head's cache). Per-token device cost is O(1) in sequence
length (cache reads scan the fixed cache window, never the recompute).
First sampled token comes from the prefill logits at row prompt_len-1.

Perf phase note: the prior full-seq metal trace was retired with the
recompute loop; the token step has fixed shapes + persistent caches/pos
buffers, so the perf redo captures it as a single trace per step. The
``use_trace`` kwarg is accepted and ignored for now.
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
        max_cache_seq: persistent KV-cache capacity (slots) per layer;
            bounds prompt_len + max_new_tokens for every ocr() call.
    """

    def __init__(
        self,
        mesh_device,
        tokenizer,
        image_processor,
        chat_template,
        num_text_layers=wl.TEXT_NUM_LAYERS,
        num_vision_layers=wl.VISION_NUM_BLOCKS,
        max_cache_seq=3200,
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
        # Persistent per-layer fp32 KV caches (decode_strategy=kv_cache):
        # allocated once; prefill refills slots 0..P32-1 each call, decode
        # steps overwrite one slot per token. Buffers persist across calls
        # (trace-capture safe for the perf phase).
        self.max_cache_seq = max_cache_seq
        self.kv_caches = [layer.init_kv_cache(max_cache_seq) for layer in self.layers]
        # All layers share one persistent slot buffer: one H2D copy per step.
        for kv in self.kv_caches[1:]:
            ttnn.deallocate(kv["pos"])
            kv["pos"] = self.kv_caches[0]["pos"]

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

    def _lm_head_argmax(self, normed: ttnn.Tensor, row: int) -> int:
        """fp32 lm_head over the tile window holding ``row``; host argmax -> token id."""
        start = (row // 32) * 32
        window = ttnn.slice(normed, [0, 0, start, 0], [1, 1, start + 32, normed.shape[-1]])
        logits_tt = self.lm_head.forward(window)
        ttnn.deallocate(window)
        logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)).float()[
            0, 0, row - start
        ]
        ttnn.deallocate(logits_tt)
        return int(torch.argmax(logits).item())

    def _prefill(self, embeds: torch.Tensor, prompt_len: int) -> int:
        """Run the padded prompt once, populating every layer's KV cache.

        embeds: [P32, H] host fp32 (pad rows zero). Returns the first
        sampled token (argmax of the logits at row prompt_len - 1).
        """
        P32, H = embeds.shape
        h = ttnn.from_torch(
            embeds.reshape(1, 1, P32, H),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        cos, sin = ref.text_rope_cos_sin(torch.arange(P32).unsqueeze(0))
        rot_mats = self.layers[0].prepare_rope(cos, sin)
        causal_mask = self.layers[0].prepare_causal_mask(P32)
        for layer, kv in zip(self.layers, self.kv_caches):
            h_next = layer.forward(h, rot_mats, causal_mask, kv_cache=kv)
            ttnn.deallocate(h)
            h = h_next
        normed = self.final_norm(h)
        ttnn.deallocate(h)
        tok = self._lm_head_argmax(normed, prompt_len - 1)
        ttnn.deallocate(normed)
        for t in (*rot_mats, causal_mask):
            ttnn.deallocate(t)
        return tok

    def _decode_step(self, token: int, slot: int) -> int:
        """One KV-cached token step at cache slot/position ``slot``; returns next token."""
        emb = self._embed_tokens([token])[0]
        x = ttnn.from_torch(
            emb.reshape(1, 1, 1, -1).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        rot_step = self.layers[0].prepare_decode_rope(slot)
        decode_mask = self.layers[0].prepare_decode_mask(slot, self.max_cache_seq)
        pos_host = ttnn.from_torch(torch.tensor([slot], dtype=torch.int32), dtype=ttnn.int32)
        ttnn.copy_host_to_device_tensor(pos_host, self.kv_caches[0]["pos"])  # shared by all layers
        h = x
        for layer, kv in zip(self.layers, self.kv_caches):
            h_next = layer.forward_decode(h, kv, rot_step, decode_mask)
            ttnn.deallocate(h)
            h = h_next
        normed = self.final_norm(h)
        ttnn.deallocate(h)
        tok = self._lm_head_argmax(normed, 0)
        ttnn.deallocate(normed)
        for t in (*rot_step, decode_mask):
            ttnn.deallocate(t)
        return tok

    # ------------------------------------------------------------------
    # The use-case verb
    # ------------------------------------------------------------------
    def ocr(
        self,
        image,
        prompt=DEFAULT_PROMPT,
        max_new_tokens=32,
        eos_ids=DEFAULT_EOS_IDS,
        use_trace=False,
        step_callback=None,
    ):
        """Greedy OCR decode. Returns the generated text (special tokens stripped).

        use_trace: accepted for caller compatibility, currently ignored —
            the KV-cached token step is fixed-shape with persistent caches
            and slot buffer; the perf phase owns the metal-trace capture.
        step_callback: optional ``f(position, ms, kind)`` hook,
            kind in {"prefill", "decode"}; "decode" is steady state.
        """
        import time as _time

        input_ids, pixel_values, grid_thw = self.preprocess(image, prompt)
        tokens = input_ids[0].tolist()
        prompt_len = len(tokens)
        P32 = ((prompt_len + 31) // 32) * 32
        assert (
            prompt_len + max_new_tokens <= self.max_cache_seq
        ), f"prompt {prompt_len} + max_new_tokens {max_new_tokens} exceeds KV-cache capacity {self.max_cache_seq}"

        # Stage 1: vision tower, then host splice at <|imgpad|> positions
        # (HF masked_scatter equivalent).
        vision_embeds = self.encode_image(pixel_values, grid_thw)
        embeds = torch.zeros(P32, vision_embeds.shape[-1])
        embeds[:prompt_len] = self._embed_tokens(tokens)
        img_pos = [i for i, t in enumerate(tokens) if t == IMAGE_TOKEN_ID]
        assert len(img_pos) == vision_embeds.shape[0], f"{len(img_pos)} != {vision_embeds.shape[0]}"
        embeds[torch.tensor(img_pos)] = vision_embeds

        # Stage 2: prefill once (populates the KV caches; pad rows up to P32
        # hold garbage K/V but stay masked until a decode step overwrites
        # them — the first decode slot IS prompt_len), then KV-cached steps.
        t0 = _time.perf_counter()
        next_token = self._prefill(embeds, prompt_len)
        if step_callback:
            step_callback(prompt_len - 1, (_time.perf_counter() - t0) * 1000.0, "prefill")
        generated = [next_token]
        for step in range(1, max_new_tokens):
            if next_token in eos_ids:
                break
            t0 = _time.perf_counter()
            next_token = self._decode_step(next_token, prompt_len + step - 1)
            if step_callback:
                step_callback(prompt_len + step - 1, (_time.perf_counter() - t0) * 1000.0, "decode")
            generated.append(next_token)
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
