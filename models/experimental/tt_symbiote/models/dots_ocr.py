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


from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import TracedRun, trace_enabled
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
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIColShardedWAllReduced,
    TTNNLinearLLamaIColShardedWAllReduced,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.utils.device_management import timed_call


def _argmax_token_on_device(logits: ttnn.Tensor) -> ttnn.Tensor:
    logits_rm = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    token = ttnn.argmax(
        logits_rm,
        dim=-1,
        keepdim=True,
        use_multicore=True,
    )
    return ttnn.reshape(token, (1, 1))


@trace_enabled
class TTNNDotsOCRPrefillGraph(TTNNModule):
    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def __init__(self, decoder_stack, final_norm, lm_head):
        super().__init__()
        self._p_stack = decoder_stack
        self._p_norm = final_norm
        self._p_lm = lm_head

    def forward(self, hidden_states, cache_position, past_key_value):
        h = self._p_stack.forward(hidden_states, past_key_value=past_key_value, cache_position=cache_position)
        h = self._p_norm.forward(h)
        sl = int(h.shape[-2])
        if sl > 1:
            hd = int(h.shape[-1])
            h = ttnn.slice(h, [0, sl - 1, 0], [1, sl, hd])
        logits = self._p_lm.forward(h)
        return _argmax_token_on_device(logits)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        hid = func_args[0]
        seq_len = int(hid.shape[-2])
        for layer in self._p_stack.layers:
            past_key_value.update_seq_length(layer_idx=layer.self_attn.layer_idx, seq_len=seq_len)


@trace_enabled
class TTNNDotsOCRDecodeGraph(TTNNModule):
    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def __init__(self, embedding, decoder_stack, final_norm, lm_head):
        super().__init__()
        self._d_embed = embedding
        self._d_stack = decoder_stack
        self._d_norm = final_norm
        self._d_lm = lm_head

    def forward(self, token_ids, cache_position, past_key_value):
        emb = self._d_embed.forward(token_ids)
        h = self._d_stack.forward(emb, past_key_value=past_key_value, cache_position=cache_position)
        h = self._d_norm.forward(h)
        logits = self._d_lm.forward(h)
        return _argmax_token_on_device(logits)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        tok = func_args[0]
        seq_len = int(tok.shape[-1])
        for layer in self._d_stack.layers:
            past_key_value.update_seq_length(layer_idx=layer.self_attn.layer_idx, seq_len=seq_len)


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
        graph_prefill: TTNNDotsOCRPrefillGraph,
        graph_decode: TTNNDotsOCRDecodeGraph,
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
        self.graph_prefill = graph_prefill
        self.graph_decode = graph_decode
        self.paged_cache = paged_cache
        self._device = device
        self.config = config

        # Decode-loop buffer (allocated on first decode call, reused thereafter)
        self._decode_cache_position: Optional[ttnn.Tensor] = None
        self._decode_token_buffer: Optional[ttnn.Tensor] = None
        self._decode_token_host: Optional[ttnn.Tensor] = None
        self._decode_cache_pos_host: Optional[ttnn.Tensor] = None
        self._decode_seq_counter: int = 0

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
        lm_head = TTNNLinearLLamaIColShardedWAllReduced.from_torch(hf_model.lm_head)
        lm_head._unique_name = "lm_head"

        # Paged KV cache
        paged_cache = _create_paged_kv_cache(hf_model.config, device, batch_size)

        graph_prefill = TTNNDotsOCRPrefillGraph(decoder_stack, final_norm, lm_head)
        graph_prefill._unique_name = "dots_ocr_graph_prefill"
        graph_decode = TTNNDotsOCRDecodeGraph(embedding, decoder_stack, final_norm, lm_head)
        graph_decode._unique_name = "dots_ocr_graph_decode"

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
            graph_prefill=graph_prefill,
            graph_decode=graph_decode,
            device=device,
            config=config,
        )
        pipeline._unique_name = "dots_ocr_pipeline"

        # Set device and preprocess weights
        pipeline._set_device_and_preprocess(device)
        pipeline.prefill = timed_call(pipeline.prefill, "prefill", "TTNNDotsOCRPipelinePrefill")
        pipeline.decode_step = timed_call(pipeline.decode_step, "decode", "TTNNDotsOCRPipelineDecode")
        return pipeline

    # ------------------------------------------------------------------
    # Device / weight setup
    # ------------------------------------------------------------------

    def _set_device_and_preprocess(self, device: "ttnn.MeshDevice") -> None:
        """Recursively set device/device_state on all children, then preprocess weights."""
        from models.experimental.tt_symbiote.utils.device_management import set_device

        set_device(self, device, register_forward_hook=False, dump_visualization=False)

        # Preprocess and move weights for every leaf TTNNModule.
        for module in self._collect_ttnn_modules():
            module.preprocess_weights()
            module.move_weights_to_device()

        # LM head: HiFi2 + packer_l1_acc for decode
        self.lm_head.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

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

        for component in [
            self.embedding,
            self.vision_tower,
            self.decoder_stack,
            self.final_norm,
            self.lm_head,
            self.graph_prefill,
            self.graph_decode,
        ]:
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

        # Traced prefill graph includes decoder stack, final norm, lm_head, and argmax.
        token_id_tt = self.graph_prefill(hidden_states, tt_cache_position, past_key_value=self.paged_cache)

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

    def _init_decode_buffers(self, prev_token_id: int):
        """Allocate reusable device buffers for decode loop on first call."""
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        self._decode_token_host = torch.tensor([[prev_token_id]], dtype=torch.int32)
        self._decode_token_buffer = ttnn.from_torch(
            self._decode_token_host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
        )
        self._decode_seq_counter = self.paged_cache.get_seq_length(layer_idx=0)
        self._decode_cache_pos_host = torch.tensor([self._decode_seq_counter], dtype=torch.int32)
        self._decode_cache_position = ttnn.from_torch(
            self._decode_cache_pos_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_step(self, prev_token_id: int) -> int:
        """Execute one decode step entirely on device.

        Args:
            prev_token_id: Previous token ID (int, on host).

        Returns:
            Next predicted token ID (int).
        """
        if self._decode_token_buffer is None:
            self._init_decode_buffers(prev_token_id)
        else:
            self._decode_token_host[0][0] = prev_token_id
            token_host_tt = ttnn.from_torch(
                self._decode_token_host,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(token_host_tt, self._decode_token_buffer)

            self._decode_seq_counter += 1
            self._decode_cache_pos_host[0] = self._decode_seq_counter
            cache_host_tt = ttnn.from_torch(
                self._decode_cache_pos_host,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(cache_host_tt, self._decode_cache_position)

        token_id_tt = self.graph_decode(
            self._decode_token_buffer, self._decode_cache_position, past_key_value=self.paged_cache
        )

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
        self._decode_token_buffer = None
        self._decode_seq_counter = 0

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
        # Step B: Build vision table ON DEVICE (no host round-trip).
        # Convert TILE -> ROW_MAJOR, reshape to 2D, prepend a tiny
        # zero row, so the embedding table is [N_vision+1, H/num_devices]
        # per device, col-sharded.
        # =====================================================================
        N_vision = int(vision_tt.shape[2])
        H_per_device = int(vision_tt.shape[3])

        # B.1: TILE -> ROW_MAJOR on device (needed by ttnn.embedding)
        vision_rm = ttnn.to_layout(vision_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(vision_tt)

        # B.2: Reshape [1, 1, N_vision, H_per_device] -> [N_vision, H_per_device]
        vision_2d = ttnn.reshape(vision_rm, (N_vision, H_per_device))

        # B.3: Create zero row (tiny: ~3 KB upload, not a bottleneck)
        if num_devices > 1:
            zero_row_mapper = ttnn.ShardTensor2dMesh(
                self.device,
                dims=(None, -1),
                mesh_shape=list(self.device.shape),
            )
        else:
            zero_row_mapper = ttnn.ReplicateTensorToMesh(self.device)

        zero_row_tt = ttnn.from_torch(
            torch.zeros(1, H, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=zero_row_mapper,
        )

        # B.4: Concat zero row + vision embeddings -> vision table
        vision_table = ttnn.concat([zero_row_tt, vision_2d], dim=0)
        ttnn.deallocate(zero_row_tt)
        ttnn.deallocate(vision_2d)

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
        return _argmax_token_on_device(logits)

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
