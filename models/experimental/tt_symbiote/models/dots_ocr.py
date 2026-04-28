# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone TTNN inference pipeline for dots.ocr.

Replaces HF model.generate() entirely, combining scatter-merge on device,
argmax on device, and a custom generation loop.  No monkey-patching, no
HF generate dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import ttnn

from transformers.modeling_outputs import BaseModelOutputWithPast

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import TracedRun
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.dots_ocr_decoder_layer import (
    TTNNDotsOCRDecoderLayer,
    TTNNDotsOCRLayerStack,
)
from models.experimental.tt_symbiote.modules.dots_ocr_vision import TTNNDotsOCRVisionTower
from models.experimental.tt_symbiote.modules.embedding import TTNNEmbedding
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWAllReduced
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.utils.device_management import DeviceInit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for TTNNDotsOCRPipeline."""

    vocab_size: int = 151936
    hidden_size: int = 1536
    image_token_id: int = 151665
    eos_token_ids: List[int] = field(default_factory=lambda: [151643, 151673])
    max_new_tokens: int = 512
    num_devices: int = 2  # N300


# ---------------------------------------------------------------------------
# Helper: create paged KV cache (mirrors test helper)
# ---------------------------------------------------------------------------


def _create_paged_kv_cache(model_config, device, batch_size: int = 1):
    """Create a paged attention KV cache for dots.ocr."""
    head_dim = getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=256,
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        config=config,
        device=None,
    ).to_device(device)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TTNNDotsOCRPipeline(TTNNModule):
    """Standalone TTNN inference pipeline for dots.ocr.

    Orchestrates embedding, vision tower, decoder stack, final norm, and
    lm_head entirely through TTNN -- no HF ``model.generate()`` dependency.
    Scatter-merge and argmax run on device.

    Inherits from TTNNModule so that ``set_device()`` can recursively
    initialize all child modules (device, device_state, bypass flags,
    weight preprocessing).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        embedding: TTNNEmbedding,
        vision_tower: TTNNDotsOCRVisionTower,
        decoder_stack: TTNNDotsOCRLayerStack,
        final_norm: TTNNDistributedRMSNorm,
        lm_head: TTNNLinearIColShardedWAllReduced,
        paged_cache: TTNNPagedAttentionKVCache,
        device: "ttnn.MeshDevice",
        config: PipelineConfig,
    ):
        super().__init__()
        self._bypass_tensor_wrapping = True

        self.embedding = embedding
        self.vision_tower = vision_tower
        self.decoder_stack = decoder_stack
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.paged_cache = paged_cache
        self._device = device
        self.config = config

        # Decode-loop buffer (allocated on first decode call, reused thereafter)
        self._decode_cache_position: Optional[ttnn.Tensor] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_hf_model(
        cls,
        model_path: str,
        device: "ttnn.MeshDevice",
        batch_size: int = 1,
    ) -> "TTNNDotsOCRPipeline":
        """Build pipeline from a HuggingFace model path.

        Loads the HF model, extracts components, creates TTNN modules,
        and assembles the pipeline.
        """
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # --- Create TTNN modules from HF components ---
        # _bypass_tensor_wrapping is set later by set_device() which sees
        # that the parent (the pipeline) is a TTNNModule and propagates
        # bypass=True to all children.

        embedding = TTNNEmbedding.from_torch(hf_model.model.embed_tokens)
        embedding._unique_name = "model.embed_tokens"

        vision_tower = TTNNDotsOCRVisionTower.from_torch(hf_model.vision_tower)
        vision_tower._unique_name = "vision_tower"
        vision_tower.override_children_module_names()

        decoder_layers = []
        for i, hf_layer in enumerate(hf_model.model.layers):
            layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer)
            layer._unique_name = f"model.layers.{i}"
            layer.override_children_module_names()
            decoder_layers.append(layer)
        decoder_stack = TTNNDotsOCRLayerStack(decoder_layers)
        decoder_stack._unique_name = "model.layer_stack"

        final_norm = TTNNDistributedRMSNorm.from_torch(hf_model.model.norm)
        final_norm._unique_name = "model.norm"

        # AllReduced gives full vocab on each device for argmax
        lm_head = TTNNLinearIColShardedWAllReduced.from_torch(hf_model.lm_head)
        lm_head._unique_name = "lm_head"

        # Paged KV cache
        paged_cache = _create_paged_kv_cache(hf_model.config, device, batch_size)

        # Pipeline config
        eos_token_ids = [151643, 151673]
        if hasattr(hf_model, "generation_config") and hf_model.generation_config is not None:
            gc_eos = getattr(hf_model.generation_config, "eos_token_id", None)
            if gc_eos is not None:
                eos_token_ids = gc_eos if isinstance(gc_eos, list) else [gc_eos]

        config = PipelineConfig(
            vocab_size=hf_model.config.vocab_size,
            hidden_size=hf_model.config.hidden_size,
            image_token_id=getattr(hf_model.config, "image_token_id", 151665),
            eos_token_ids=eos_token_ids,
            num_devices=device.get_num_devices(),
        )

        pipeline = cls(
            embedding=embedding,
            vision_tower=vision_tower,
            decoder_stack=decoder_stack,
            final_norm=final_norm,
            lm_head=lm_head,
            paged_cache=paged_cache,
            device=device,
            config=config,
        )
        pipeline._unique_name = "dots_ocr_pipeline"

        # Set device and preprocess weights
        pipeline._set_device_and_preprocess(device)

        return pipeline

    # ------------------------------------------------------------------
    # Device / weight setup
    # ------------------------------------------------------------------

    def _set_device_and_preprocess(self, device: "ttnn.MeshDevice") -> None:
        """Recursively set device/device_state on all children, then preprocess weights."""
        from models.experimental.tt_symbiote.utils.device_management import set_device

        TracedRun.configure(device=device)

        # set_device recursively propagates device, device_state, and
        # _bypass_tensor_wrapping to every TTNNModule reachable from self.
        set_device(self, device, device_init=DeviceInit, register_forward_hook=False, dump_visualization=False)

        # Preprocess and move weights for every leaf TTNNModule.
        for module in self._collect_ttnn_modules():
            module.preprocess_weights()
            module.move_weights_to_device()

    def _collect_ttnn_modules(self) -> List[TTNNModule]:
        """Recursively collect all TTNNModule instances from pipeline components."""
        found: List[TTNNModule] = []
        visited: set = set()

        def _recurse(obj):
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)
            if isinstance(obj, TTNNModule):
                found.append(obj)
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    value = getattr(obj, attr_name)
                except Exception:
                    continue
                if isinstance(value, TTNNModule):
                    _recurse(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, TTNNModule):
                            _recurse(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        if isinstance(item, TTNNModule):
                            _recurse(item)

        for component in [self.embedding, self.vision_tower, self.decoder_stack, self.final_norm, self.lm_head]:
            _recurse(component)
        return found

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> int:
        """Run prefill (first forward pass) and return the first generated token.

        Args:
            input_ids: ``[1, S]`` int64/int32 token IDs on host.
            pixel_values: Optional vision input for multimodal prefill.
            image_grid_thw: Optional grid info for vision input.

        Returns:
            First predicted token ID (int).
        """
        seq_len = input_ids.shape[-1]

        # --- Embedding ---
        # All children have _bypass_tensor_wrapping=True (pipeline is a
        # TTNNModule parent), so we convert input_ids to ttnn ourselves.
        tt_input_ids = ttnn.from_torch(
            input_ids.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        text_embeds = self.embedding(tt_input_ids)

        # --- Vision scatter-merge (if applicable) ---
        if pixel_values is not None:
            hidden_states = self._scatter_merge_on_device(text_embeds, pixel_values, image_grid_thw, input_ids)
        else:
            hidden_states = text_embeds

        # --- Cache position for prefill ---
        cache_position = torch.arange(0, seq_len, dtype=torch.int32)
        tt_cache_position = ttnn.from_torch(
            cache_position,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # --- Decoder stack (via __call__ for TracedRun lifecycle) ---
        hidden_states = self.decoder_stack(
            hidden_states,
            past_key_value=self.paged_cache,
            cache_position=tt_cache_position,
        )

        # --- Final norm ---
        hidden_states = self.final_norm(hidden_states)

        # Slice to last token BEFORE lm_head to avoid projecting the full
        # sequence to vocab_size (which OOMs on long vision prefills).
        if seq_len > 1:
            h_dim = hidden_states.shape[-1]
            hidden_states = ttnn.slice(
                hidden_states,
                [0, seq_len - 1, 0],
                [1, seq_len, h_dim],
            )

        # --- lm_head ---
        logits = self.lm_head(hidden_states)
        # logits shape: [1, 1, vocab_size] (rank 3, full vocab, replicated)

        # --- Argmax on device ---
        token_id_tt = self._argmax_on_device(logits)

        # --- Read to host ---
        token_id_torch = ttnn.to_torch(
            token_id_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )
        first_token = int(token_id_torch.flatten()[0].item())

        # Clean up prefill-only tensors
        ttnn.deallocate(tt_cache_position)

        return first_token

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode_step(self, prev_token_id: int) -> int:
        """Execute one decode step entirely on device.

        Args:
            prev_token_id: Previous token ID (int, on host).

        Returns:
            Next predicted token ID (int).
        """
        # --- Embedding ---
        tt_token = ttnn.from_torch(
            torch.tensor([[prev_token_id]], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        embed = self.embedding(tt_token)

        # --- Cache position management ---
        # Follow the pattern from TTNNDotsOCRModel.call():
        # Pre-allocate _decode_cache_position buffer on first decode,
        # reuse via ttnn.copy on subsequent calls.
        cur_seq_len = self.paged_cache.get_seq_length(layer_idx=0)
        cache_pos_val = torch.tensor([cur_seq_len], dtype=torch.int32)
        tt_cache_pos_new = ttnn.from_torch(
            cache_pos_val,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if self._decode_cache_position is not None:
            ttnn.copy(tt_cache_pos_new, self._decode_cache_position)
            ttnn.deallocate(tt_cache_pos_new)
            tt_cache_position = self._decode_cache_position
        else:
            self._decode_cache_position = tt_cache_pos_new
            tt_cache_position = self._decode_cache_position

        # --- Decoder stack ---
        hidden_states = self.decoder_stack(
            embed,
            past_key_value=self.paged_cache,
            cache_position=tt_cache_position,
        )

        # --- Final norm ---
        hidden_states = self.final_norm(hidden_states)

        # --- lm_head ---
        logits = self.lm_head(hidden_states)
        # For decode, logits is already [1, 1, 1, vocab_size]

        # --- Argmax on device ---
        token_id_tt = self._argmax_on_device(logits)

        # --- Read to host ---
        token_id_torch = ttnn.to_torch(
            token_id_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )
        return int(token_id_torch.flatten()[0].item())

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
    ) -> List[int]:
        """Full generation: prefill + decode loop.

        Args:
            input_ids: ``[1, S]`` token IDs on host.
            pixel_values: Optional vision input.
            image_grid_thw: Optional grid info for vision.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            List of generated token IDs (not including prompt).
        """
        # Reset cache for fresh generation
        self.paged_cache.reset()
        self._decode_cache_position = None

        # Prefill
        first_token_id = self.prefill(input_ids, pixel_values, image_grid_thw)

        # Decode loop
        generated = [first_token_id]
        current_token = first_token_id

        for _ in range(max_new_tokens - 1):
            next_token = self.decode_step(current_token)
            generated.append(next_token)

            # Check EOS
            if next_token in self.config.eos_token_ids:
                break

            current_token = next_token

        return generated

    # ------------------------------------------------------------------
    # Scatter-merge (vision + text on device)
    # ------------------------------------------------------------------

    def _scatter_merge_on_device(
        self,
        text_embeds: ttnn.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> ttnn.Tensor:
        """Merge vision embeddings into text embeddings at image token positions.

        Args:
            text_embeds: ``[1, S, H_per_device]`` bf16 on device (col-sharded, rank 3).
            pixel_values: Vision input (torch on host).
            image_grid_thw: Grid info (torch on host).
            input_ids: ``[1, S]`` token IDs (torch on host, for building mask).

        Returns:
            fused_embeds: same shape as text_embeds, bf16 on device.
        """
        S = text_embeds.shape[-2]
        H = self.config.hidden_size

        num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1

        # =====================================================================
        # Step A: Vision tower produces col-sharded output
        # =====================================================================
        vision_tt = self.vision_tower.forward(pixel_values, image_grid_thw)

        # =====================================================================
        # Step B: Build vision table on host, re-upload as col-sharded.
        # Download col-sharded vision_tt, reassemble full H on host,
        # prepend zero row, re-upload with ShardTensor2dMesh so the
        # embedding table is col-sharded [N_vision+1, H/num_devices]
        # per device.
        # =====================================================================
        N_vision = int(vision_tt.shape[2])

        if num_devices > 1:
            # Col-sharded: concat on last dim to reassemble full H
            composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
        else:
            composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        vision_host = ttnn.to_torch(vision_tt, mesh_composer=composer).to(torch.bfloat16)
        ttnn.deallocate(vision_tt)

        if num_devices <= 1 and vision_host.shape[0] > 1:
            per = vision_host.shape[0] // max(num_devices, 1)
            vision_host = vision_host[:per]

        vision_host = vision_host.squeeze(0).squeeze(0)  # [N_vision, H]

        zero_row = torch.zeros(1, H, dtype=torch.bfloat16)
        vision_table_host = torch.cat([zero_row, vision_host], dim=0)  # [N_vision+1, H]

        if num_devices > 1:
            # Upload as col-sharded so embedding gather produces col-sharded output
            table_mapper = ttnn.ShardTensor2dMesh(
                self.device,
                dims=(None, -1),
                mesh_shape=list(self.device.shape),
            )
        else:
            table_mapper = ttnn.ReplicateTensorToMesh(self.device)

        vision_table = ttnn.from_torch(
            vision_table_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=table_mapper,
        )

        # =====================================================================
        # Step C: Build gather index on host (tiny: ~19 KB) and upload
        # gather_idx[0, pos] = 0 for non-image tokens (gathers zero row)
        #                    = 1..N_vision for image tokens
        # =====================================================================
        img_mask = input_ids == self.config.image_token_id  # [1, S] bool
        img_positions = img_mask.squeeze().nonzero(as_tuple=True)[0]
        n_img = min(len(img_positions), N_vision)

        gather_idx = torch.zeros(1, S, dtype=torch.int32)
        gather_idx[0, img_positions[:n_img]] = torch.arange(1, n_img + 1, dtype=torch.int32)

        tt_idx = ttnn.from_torch(
            gather_idx,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

        # =====================================================================
        # Step D: Device-side gather via ttnn.embedding
        # With col-sharded table [N_vision+1, H/num_devices] and replicated
        # index, each device independently gathers its own H/num_devices
        # columns -> [1, S, H/num_devices] col-sharded output.
        # No identity matmul needed!
        # =====================================================================
        full_vision_col_sharded = ttnn.embedding(tt_idx, vision_table, layout=ttnn.TILE_LAYOUT)
        # full_vision_col_sharded: [1, S, H/num_devices] col-sharded, TILE_LAYOUT
        ttnn.deallocate(tt_idx)
        ttnn.deallocate(vision_table)

        # =====================================================================
        # Step E: Build mask and merge
        # =====================================================================
        mask_float = img_mask.float().unsqueeze(-1)  # [1, S, 1]
        tt_mask = ttnn.from_torch(
            mask_float,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

        fused = ttnn.where(tt_mask, full_vision_col_sharded, text_embeds)

        return fused

    # ------------------------------------------------------------------
    # Argmax on device
    # ------------------------------------------------------------------

    def _argmax_on_device(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """Run argmax on device.

        Args:
            logits: ``[1, 1, vocab_size]`` bf16 TILE_LAYOUT on device (rank 3).

        Returns:
            token_id tensor on device.
        """
        logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        token_id = ttnn.argmax(
            logits_rm,
            dim=-1,
            keepdim=True,
            use_multicore=True,
        )
        return token_id

    # ------------------------------------------------------------------
    # Warmup / trace management
    # ------------------------------------------------------------------

    def warmup(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> None:
        """Run warmup passes to prime JIT and capture traces.

        Call sequence:
          1. Warmup run (no trace): primes TTNN JIT, CCL, allocators.
          2. TracedRun.release_all(): release any traces from warmup.
          3. Trace capture run: captures decode trace.
          4. Reset: clear KV cache, reset positions.
        """
        # Pass 1: Warmup (TracedRun phase 1 -- no trace capture)
        self.generate(input_ids, pixel_values, image_grid_thw, max_new_tokens=2)
        self.paged_cache.reset()
        self._decode_cache_position = None

        # Release all traces so run 2 starts clean
        TracedRun.release_all()

        # Pass 2: Trace capture (TracedRun phase 2)
        self.generate(input_ids, pixel_values, image_grid_thw, max_new_tokens=4)
        self.paged_cache.reset()
        self._decode_cache_position = None

    def release(self) -> None:
        """Release all traced runs and deallocate pre-allocated buffers."""
        TracedRun.release_all()
        self._decode_cache_position = None


# ---------------------------------------------------------------------------
# HF-compatible model wrapper (used by HF model.generate() path)
# ---------------------------------------------------------------------------


class TTNNDotsOCRModel(TTNNModule):
    @staticmethod
    def from_torch(model):
        new_model = TTNNDotsOCRModel()
        new_model.model = model
        new_model._decode_cache_position = None

        for layer in model.layers:
            if isinstance(layer, TTNNModule):
                layer._bypass_tensor_wrapping = True
        if isinstance(model.norm, TTNNModule):
            model.norm._bypass_tensor_wrapping = True

        ttnn_layers = [l for l in model.layers if isinstance(l, TTNNModule)]
        new_model.layer_stack = TTNNDotsOCRLayerStack(ttnn_layers)
        new_model.layer_stack._bypass_tensor_wrapping = True

        return new_model

    def to_device(self, device):
        super().to_device(device)
        self.layer_stack.to_device(device)
        return self

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            pass
        return getattr(self.__dict__["model"], name)

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ):
        ttnn_self = self
        hf_model = self.model
        use_cache = use_cache if use_cache is not None else hf_model.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = hf_model.embed_tokens(input_ids)
        elif isinstance(inputs_embeds, torch.Tensor) and type(inputs_embeds) is torch.Tensor:
            inputs_embeds = ttnn.from_torch(
                inputs_embeds,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_self.device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    ttnn_self.device,
                    dims=(None, -1),
                    mesh_shape=list(ttnn_self.device.shape),
                ),
            )

        if use_cache and past_key_values is None:
            from transformers import DynamicCache

            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], dtype=torch.int32
            )

        if not isinstance(cache_position, ttnn.Tensor):
            cp = cache_position
            if hasattr(cp, "cpu"):
                cp = cp.cpu()
            if isinstance(cp, torch.Tensor):
                cp = cp.to(torch.int32)
            else:
                cp = torch.tensor(cp, dtype=torch.int32)
            mesh_mapper = (
                ttnn.ReplicateTensorToMesh(ttnn_self.device) if ttnn_self.device.get_num_devices() > 1 else None
            )
            is_decode = inputs_embeds.shape[1] == 1
            if is_decode and ttnn_self._decode_cache_position is not None:
                cp_temp = ttnn.from_torch(
                    cp,
                    device=ttnn_self.device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.copy(cp_temp, ttnn_self._decode_cache_position)
                cache_position = ttnn_self._decode_cache_position
            else:
                cache_position = ttnn.from_torch(
                    cp,
                    device=ttnn_self.device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if is_decode:
                    ttnn_self._decode_cache_position = cache_position

        hidden_states = inputs_embeds

        hidden_states = ttnn_self.layer_stack(
            hidden_states,
            past_key_value=past_key_values,
            cache_position=cache_position,
        )

        hidden_states = hf_model.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
