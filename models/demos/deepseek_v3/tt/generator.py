# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.model_1d import Model1D
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_helpers import MAX_BATCH_SIZE
from models.demos.deepseek_v3.utils.hf_model_utils import load_model_weights
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import add_inv_scale_to_state_dict


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 0.0


def _strip_model_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a copy of the HF state_dict with leading 'model.' stripped.

    Deepseek TT modules expect keys like 'embed_tokens.', 'layers.', 'norm.',
    but HF weights are under 'model.'.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            out[k[len("model.") :]] = v
        else:
            out[k] = v
    return out


class DeepseekGenerator:
    """
    Simple generator that wires Model1D + LMHead for decode-only inference.

    Notes:
    - Prefill at the model level is not fully implemented in Model1D; we emulate
      prefill by iterating decode steps over the prompt tokens (updates caches).
    - Batch size in configs is tied to MAX_BATCH_SIZE; for simplicity we decode
      up to that many sequences. If fewer are provided, we pad/ignore extras.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        model_path: str | Path,
        cache_dir: str | Path | None = None,
        batch_size: int = MAX_BATCH_SIZE,
        tokenizer=None,
        random_weights: bool = False,
        dense_layers: int | None = None,
        override_num_layers: int | None = None,
        single_layer: str | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.model_path = str(model_path)
        self.batch_size = min(MAX_BATCH_SIZE, batch_size)

        # Load HF config + tokenizer
        self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self._ensure_max_seq_len(self.hf_config)
        # Optional overrides for layer counts before building states
        if override_num_layers is not None:
            try:
                self.hf_config.num_hidden_layers = int(override_num_layers)
            except Exception:
                pass
        if dense_layers is not None:
            try:
                self.hf_config.first_k_dense_replace = int(dense_layers)
            except Exception:
                pass
        # Tokenizer is optional; caller can pass a tokenizer or handle failure.
        self.tokenizer = tokenizer

        # Runtime helpers
        self.ccl = CCL1D(mesh_device)
        mesh_shape = list(mesh_device.shape)
        self.dp_factor = mesh_shape[1]

        # Paged attention setup
        self.paged_config = MLA1D.get_valid_paged_config(self.hf_config.max_seq_len, MAX_BATCH_SIZE, self.dp_factor)
        self.page_table_tt, _ = MLA1D.create_page_table(
            MAX_BATCH_SIZE, dp_factor=self.dp_factor, config=self.paged_config, mesh_device=mesh_device
        )
        self.rope = RotarySetup(device=mesh_device, batch_size=MAX_BATCH_SIZE, hf_config=self.hf_config)

        # Prepare weights/configs
        self.random_weights = random_weights
        self.single_layer = single_layer
        self._prepare_run_configs(cache_dir)

    @staticmethod
    def _ensure_max_seq_len(hf_config) -> None:
        if getattr(hf_config, "max_seq_len", None) is not None:
            return
        try:
            if getattr(hf_config, "rope_scaling", None):
                factor = hf_config.rope_scaling.get("factor")
                orig = hf_config.rope_scaling.get("original_max_position_embeddings")
                if factor and orig:
                    hf_config.max_seq_len = int(factor * orig)
                    return
            if getattr(hf_config, "max_position_embeddings", None):
                hf_config.max_seq_len = int(hf_config.max_position_embeddings)
                return
        except Exception:
            pass
        hf_config.max_seq_len = 4096

    def _prepare_run_configs(self, cache_dir: str | Path | None) -> None:
        cache_root = Path(cache_dir) if cache_dir is not None else Path("generated/deepseek_v3")
        weights_out = cache_root / "weights"
        weights_out.mkdir(parents=True, exist_ok=True)

        if self.random_weights:
            if self.single_layer and self.single_layer.lower() == "moe":
                raise NotImplementedError(
                    "Random weights with 'moe' single layer is not supported by Model1D demo yet. Use 'mlp' or disable random mode."
                )
            logger.info("Building random weights from HF reference model (ForCausalLM)...")
            ref_model = DeepseekV3ForCausalLM(self.hf_config).eval()
            # Ensure parameter/buffer dtype matches downstream expectations (bfloat16)
            ref_model = ref_model.to(dtype=torch.bfloat16)
            torch_state = ref_model.state_dict()
            # Quantize MLP weights as expected by TT converters
            torch_state = add_inv_scale_to_state_dict(
                torch_state,
                block_shape=self.hf_config.quantization_config["weight_block_size"],
            )
            stripped = _strip_model_prefix(torch_state)
            model1d_state = {
                k: v
                for k, v in stripped.items()
                if k.startswith("embed_tokens.")
                or k.startswith("layers.")
                or k.startswith("norm.")
                or k.startswith("lm_head.")
            }
        else:
            logger.info("Loading HF weights (this may take a while)...")
            hf_weights = load_model_weights(self.model_path)
            logger.info("HF weights loaded")

            # Split weight dicts for Model1D and LMHead
            stripped = _strip_model_prefix(hf_weights)
            if "lm_head.weight" not in stripped:
                raise RuntimeError(
                    "No HF safetensors found in model path or missing 'lm_head.weight'. "
                    "Set DEEPSEEK_V3_HF_MODEL to a directory containing DeepSeek-V3 safetensors, or pass --model-path."
                )
            model1d_state = {
                k: v
                for k, v in stripped.items()
                if k.startswith("embed_tokens.")
                or k.startswith("layers.")
                or k.startswith("norm.")
                or k.startswith("lm_head.")
            }
        # Convert weights to TT tensors-on-disk and build weight_config
        logger.info("Converting weights to TTNN SavedWeight format (Model1D)...")
        self.model1d_weight_config = Model1D.convert_weights(
            self.hf_config, model1d_state, weights_out / "model_1d", self.mesh_device
        )  # Build model and head decode configs + states
        model_decode_cfg = Model1D.decode_model_config(self.hf_config, self.mesh_device)
        model_state = Model1D.create_state(
            self.hf_config, self.mesh_device, paged_config=self.paged_config, ccl=self.ccl
        )
        model_shared_state = Model1D.create_shared_state(self.hf_config, self.mesh_device)
        self.model1d_run_config = create_run_config(
            model_decode_cfg, self.model1d_weight_config, model_state, model_shared_state
        )

    def _tt_from_tokens_step(self, tokens_step: torch.Tensor) -> ttnn.Tensor:
        """Tokens step: [B] -> TTNN tensor [1, 1, B] uint32, replicated to mesh."""
        assert tokens_step.dim() == 1
        x = tokens_step.view(1, 1, -1).to(torch.int32)
        return ttnn.from_torch(
            x,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _tt_from_positions(self, positions: torch.Tensor) -> Tuple[dict, ttnn.Tensor]:
        """Return rope tensors dict and TTNN positions shard for decode.

        positions: [B] int tensor
        returns: (rope_tensors, tt_positions)
        """
        # Build RoPE tensors for current positions
        rope_mats = self.rope.get_rot_mats(positions.to(torch.int32))
        rope_tensors = {"cos_matrix": rope_mats[0], "sin_matrix": rope_mats[1], "trans_matrix": rope_mats[2]}

        # Create TTNN position tensor as INT32 with the same sharding pattern used in tests
        mesh_shape = list(self.mesh_device.shape)
        tt_positions = ttnn.from_torch(
            positions.to(torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 0), mesh_shape=mesh_shape),
            dtype=ttnn.int32,
        )
        return rope_tensors, tt_positions

    def _decode_step(self, tokens_step: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Run a single decode step and return logits on host as torch tensor [1, 1, B, V]."""
        # Prepare TT inputs
        tt_tokens = self._tt_from_tokens_step(tokens_step)
        rope_tensors, tt_positions = self._tt_from_positions(positions)

        # Model forward
        logits_tt = Model1D.forward_decode(
            tt_tokens, tt_positions, rope_tensors, self.page_table_tt, self.model1d_run_config
        )
        # Gather to host
        logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))

        # Free device tensors for this step
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(logits_tt)

        return logits  # [1, 1, B, V]

    def _sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [1, 1, B, V]
        return torch.argmax(logits[0, 0], dim=-1)  # [B]

    def _pad_batch(self, tokens_list: List[List[int]]) -> Tuple[torch.Tensor, List[int]]:
        """Pad/pack a list of token id sequences to batch of size MAX_BATCH_SIZE.

        Returns
            tokens_packed: torch.LongTensor [MAX_BATCH_SIZE, S]
            valid_counts: list of actual sequence lengths for first N sequences
        """
        assert len(tokens_list) > 0 and len(tokens_list) <= MAX_BATCH_SIZE
        max_len = max(len(t) for t in tokens_list)
        B = MAX_BATCH_SIZE
        out = torch.full((B, max_len), 0, dtype=torch.long)
        valid = []
        for i, seq in enumerate(tokens_list):
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            valid.append(len(seq))
        return out, valid

    def generate(
        self,
        prompts: Iterable[str],
        max_new_tokens: int = 32,
        sampling: SamplingParams | None = None,
        teacher_forcing=None,
    ) -> List[List[int]]:
        """Generate tokens for the given prompts using greedy decode by default.

        Returns: list of generated token id lists for the provided prompts (order preserved).
        """
        prompts = list(prompts)
        assert 1 <= len(prompts) <= MAX_BATCH_SIZE, f"Supports 1..{MAX_BATCH_SIZE} prompts"

        # Tokenize using HF chat template
        encoded: List[List[int]] = [self._encode_prompt(p) for p in prompts]
        tokens_batched, lengths = self._pad_batch(encoded)  # [MAX_BATCH_SIZE, S]

        # Prefill via repeated decode steps over prompt tokens
        B = MAX_BATCH_SIZE
        positions = torch.zeros(B, dtype=torch.int32)
        last_logits = None
        for step in range(tokens_batched.shape[1]):
            step_tokens = tokens_batched[:, step]  # [B]
            last_logits = self._decode_step(step_tokens, positions)
            positions += 1

        assert last_logits is not None
        # First sampled token after prompt
        next_tokens = self._sample_greedy(last_logits)
        # If teacher forcing is enabled, collect the model's predicted token and force GT for next step (single prompt)
        if teacher_forcing is not None:
            # Only enforce for the first user to keep scope minimal
            forced = teacher_forcing.collect_predicted_tokens(int(next_tokens[0].item()))
            next_tokens[0] = int(forced)

        generations: List[List[int]] = [[] for _ in range(len(prompts))]
        for gen_idx in range(max_new_tokens):
            # Decode one step with previous next_tokens
            logits = self._decode_step(next_tokens, positions)
            pred_tokens = self._sample_greedy(logits)
            if teacher_forcing is not None:
                forced = teacher_forcing.collect_predicted_tokens(int(pred_tokens[0].item()))
                pred_tokens[0] = int(forced)
            next_tokens = pred_tokens
            positions += 1

            # Collect only for the original batch size
            for i in range(len(prompts)):
                generations[i].append(int(next_tokens[i].item()))

        return generations

    def _encode_prompt(self, prompt: str) -> List[int]:
        # Use HF chat template if a tokenizer is provided; otherwise synthesize simple token ids
        if self.tokenizer is not None:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
            )
            return list(ids)

        # Fallback: deterministic dummy tokenization for random-weights mode
        vocab = int(getattr(self.hf_config, "vocab_size", 32768))
        # Start with a BOS-like token 0
        out: List[int] = [0]
        data = prompt.encode("utf-8")[:16]
        for i, b in enumerate(data):
            out.append(int((b + i) % vocab))
        # Append an EOS-like token 1 if within vocab
        if vocab > 1:
            out.append(1)
        return out


__all__ = ["DeepseekGenerator", "SamplingParams"]
