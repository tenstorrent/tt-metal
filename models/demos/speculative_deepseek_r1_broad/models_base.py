# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import random
from typing import Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models.demos.speculative_deepseek_r1_broad.base_runtime import (
    DecodeState,
    clone_past_key_values,
    normalize_past_key_values,
    resolve_dtype,
)
from models.demos.speculative_deepseek_r1_broad.base_verification import (
    PathVerification,
    verify_paths_batched_single_pass,
    verify_paths_flattened_tree,
    verify_paths_from_decode_state,
)
from models.demos.speculative_deepseek_r1_broad.reference.configuration_deepseek_r1 import load_reference_config
from models.demos.speculative_deepseek_r1_broad.reference.reference_utils import build_reference_bundle, summarize_model_structure

logger = logging.getLogger(__name__)


class DeepSeekBaseAdapter:
    """HF-backed base model adapter with path-batched verification."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        torch_dtype: str = "float32",
        trust_remote_code: bool = False,
        base_impl: str = "reference",
        tp_size: int = 1,
        tp_backend: str = "deepspeed",
        local_rank: int | None = None,
    ) -> None:
        if tp_size > 1:
            if not torch.cuda.is_available():
                raise RuntimeError("Tensor parallel requires CUDA-capable environment.")
            local_rank_resolved = int(local_rank if local_rank is not None and local_rank >= 0 else os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{local_rank_resolved}")
        else:
            self.device = torch.device(device)
        resolved_dtype = resolve_dtype(torch_dtype)
        self.base_impl = base_impl
        self.tp_size = int(tp_size)
        self.tp_backend = tp_backend
        logger.info(
            "Loading base model '%s' impl=%s tp_size=%d tp_backend=%s on device=%s dtype=%s trust_remote_code=%s",
            model_id,
            base_impl,
            self.tp_size,
            self.tp_backend,
            device,
            torch_dtype,
            trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

        if base_impl == "accelerate":
            self._load_with_accelerate(model_id, trust_remote_code, resolved_dtype)
        elif self.tp_size > 1:
            if self.tp_backend != "deepspeed":
                raise ValueError(f"Unsupported tp_backend='{self.tp_backend}'.")
            try:
                import deepspeed
            except Exception as exc:
                raise RuntimeError("deepspeed is required for tp_size > 1. Please install deepspeed.") from exc

            if base_impl == "reference":
                try:
                    ref_cfg = load_reference_config(model_id, trust_remote_code=trust_remote_code)
                    logger.info(
                        "Reference base structure: architecture=%s hidden=%d layers=%d heads=%d vocab=%d max_pos=%d",
                        ref_cfg.architecture,
                        ref_cfg.hidden_size,
                        ref_cfg.num_hidden_layers,
                        ref_cfg.num_attention_heads,
                        ref_cfg.vocab_size,
                        ref_cfg.max_position_embeddings,
                    )
                except Exception as exc:
                    logger.warning("Could not load reference config summary for '%s': %s", model_id, exc)
            try:
                raw_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=resolved_dtype,
                    low_cpu_mem_usage=True,
                )
                raw_model.eval()
                ds_engine = deepspeed.init_inference(
                    raw_model,
                    mp_size=self.tp_size,
                    dtype=resolved_dtype,
                    replace_method="auto",
                    replace_with_kernel_inject=True,
                )
                self._ds_engine = ds_engine
                self.model = ds_engine.module if hasattr(ds_engine, "module") else ds_engine
            except RuntimeError as exc:
                msg = str(exc)
                if "FP8 quantization" in msg and "GPU or XPU" in msg:
                    raise RuntimeError(
                        f"Base model '{model_id}' requires GPU/XPU for FP8 quantization and cannot run in this path."
                    ) from exc
                raise
        elif base_impl == "reference":
            try:
                bundle = build_reference_bundle(
                    model_id,
                    device=device,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                )
            except RuntimeError as exc:
                msg = str(exc)
                if "FP8 quantization" in msg and "GPU or XPU" in msg:
                    raise RuntimeError(
                        f"Base model '{model_id}' requires GPU/XPU for FP8 quantization and cannot run in this CPU path."
                    ) from exc
                raise
            self.tokenizer = bundle.tokenizer
            self.model = bundle.model
            logger.info("Reference base structure: %s", summarize_model_structure(bundle))
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=resolved_dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device)
            except RuntimeError as exc:
                msg = str(exc)
                if "FP8 quantization" in msg and "GPU or XPU" in msg:
                    raise RuntimeError(
                        f"Base model '{model_id}' requires GPU/XPU for FP8 quantization and cannot run in this CPU path."
                    ) from exc
                raise
        self.model.eval()
        logger.info("Base model '%s' loaded successfully (impl=%s)", model_id, base_impl)

    def _load_with_accelerate(self, model_id: str, trust_remote_code: bool, resolved_dtype: torch.dtype) -> None:
        """Load large models across multiple GPUs using accelerate.

        This handles models like DeepSeek-R1-0528 that are too large for a single
        GPU and have custom architectures (MLA, MoE, MTP) that DeepSpeed can't
        kernel-inject. It creates the model structure first (skipping MTP layers),
        then loads matching weights directly to the correct devices.
        """
        from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
        from huggingface_hub import snapshot_download

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        n_mtp = int(getattr(config, "num_nextn_predict_layers", 0))
        if n_mtp > 0:
            logger.info("Disabling %d MTP layers for accelerate loading (not needed for base inference)", n_mtp)
            config.num_nextn_predict_layers = 0

        logger.info("Creating empty model from config...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_gpus > 0:
            mem = {i: "75GiB" for i in range(n_gpus)}
            mem["cpu"] = "500GiB"
            logger.info("Computing device map for %d GPUs + CPU...", n_gpus)
        else:
            mem = {"cpu": "500GiB"}
            logger.info("Computing device map for CPU only...")

        no_split = ["DeepseekV3DecoderLayer", "DeepseekR1DecoderLayer"]
        device_map = infer_auto_device_map(model, max_memory=mem, no_split_module_classes=no_split)

        gpu_layers = sum(1 for v in device_map.values() if isinstance(v, int) or (isinstance(v, str) and v.startswith("cuda")))
        cpu_layers = sum(1 for v in device_map.values() if v == "cpu")
        logger.info("Device map: %d modules on GPU, %d on CPU", gpu_layers, cpu_layers)

        logger.info("Downloading/resolving model path...")
        model_path = snapshot_download(model_id)

        logger.info("Loading and dispatching weights...")
        model = load_checkpoint_and_dispatch(
            model, model_path,
            device_map=device_map,
            no_split_module_classes=no_split,
            dtype=resolved_dtype,
        )

        self.model = model
        self.device = next(model.parameters()).device
        logger.info("Model loaded with accelerate on device=%s", self.device)

    def encode_prompt(self, prompt: str) -> list[int]:
        encoded = self.tokenizer(prompt, return_tensors=None, add_special_tokens=True)
        input_ids = encoded["input_ids"]
        if len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if not input_ids:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                raise ValueError("Tokenizer returned an empty prompt and has no bos_token_id.")
            return [int(bos)]
        return [int(token_id) for token_id in input_ids]

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    # Demo-style naming: explicit prefill/decode API.
    def forward_prefill(self, prefix_token_ids: Sequence[int]) -> DecodeState:
        return self.create_decode_state(prefix_token_ids)

    def forward_decode(self, state: DecodeState, token_id: int) -> DecodeState:
        return self.advance_decode_state(state, token_id)

    @torch.no_grad()
    def greedy_next_token(self, prefix_token_ids: Sequence[int]) -> int:
        if len(prefix_token_ids) == 0:
            raise ValueError("prefix_token_ids must be non-empty.")
        input_tensor = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=self.device)
        logits = self.model(input_ids=input_tensor).logits[:, -1, :]
        next_token_id = int(torch.argmax(logits, dim=-1).item())
        return next_token_id

    @torch.no_grad()
    def create_decode_state(self, prefix_token_ids: Sequence[int]) -> DecodeState:
        """Prefill model once and return an incremental decode state."""
        if len(prefix_token_ids) == 0:
            raise ValueError("prefix_token_ids must be non-empty.")
        input_tensor = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=self.device)
        outputs = self.model(input_ids=input_tensor, use_cache=True, output_hidden_states=True)
        return DecodeState(
            past_key_values=normalize_past_key_values(outputs.past_key_values),
            next_token_logits=outputs.logits[:, -1, :],
            last_hidden_state=outputs.hidden_states[-1][:, -1, :],
            multi_layer_hidden=self._extract_multi_layer_hidden(outputs.hidden_states),
        )

    def decode_state_next_token(self, state: DecodeState) -> int:
        return int(torch.argmax(state.next_token_logits, dim=-1).item())

    def clone_decode_state(self, state: DecodeState) -> DecodeState:
        return DecodeState(
            past_key_values=clone_past_key_values(state.past_key_values),
            next_token_logits=state.next_token_logits.clone(),
            last_hidden_state=state.last_hidden_state.clone(),
            multi_layer_hidden=state.multi_layer_hidden.clone() if state.multi_layer_hidden is not None else None,
        )

    eagle3_layer_indices: tuple[int, int, int] | None = None

    def set_eagle3_layer_indices(self, indices: tuple[int, int, int] | list[int]) -> None:
        self.eagle3_layer_indices = (int(indices[0]), int(indices[1]), int(indices[2]))
        logger.info("EAGLE3 feature layer indices set to %s", self.eagle3_layer_indices)

    def _extract_multi_layer_hidden(self, hidden_states: tuple) -> torch.Tensor | None:
        """Extract hidden states from 3 layers for EAGLE3 feature fusion.

        Uses eagle3_layer_indices if set (from draft model config, e.g. [1,29,57]),
        otherwise falls back to (7,16,27) for LLaMA-8B or evenly-spaced for others.
        """
        n = len(hidden_states)
        if n < 3:
            return None
        if self.eagle3_layer_indices is not None:
            low_idx, mid_idx, high_idx = self.eagle3_layer_indices
        else:
            low_idx, mid_idx, high_idx = 7, 16, 27
        if high_idx >= n:
            low_idx = 0
            mid_idx = n // 2
            high_idx = n - 1
        return torch.cat([
            hidden_states[low_idx][:, -1, :],
            hidden_states[mid_idx][:, -1, :],
            hidden_states[high_idx][:, -1, :],
        ], dim=-1)

    @torch.no_grad()
    def advance_decode_state(self, state: DecodeState, token_id: int) -> DecodeState:
        """Append one committed token and compute logits for the next position."""
        input_tensor = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        outputs = self.model(
            input_ids=input_tensor,
            past_key_values=state.past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        return DecodeState(
            past_key_values=normalize_past_key_values(outputs.past_key_values),
            next_token_logits=outputs.logits[:, -1, :],
            last_hidden_state=outputs.hidden_states[-1][:, -1, :],
            multi_layer_hidden=self._extract_multi_layer_hidden(outputs.hidden_states),
        )

    def verify_paths_from_decode_state(
        self,
        decode_state: DecodeState,
        proposed_paths: Sequence[Sequence[int]],
        *,
        acceptance_mode: str = "argmax",
        rng: random.Random | None = None,
        draft_probs_per_path: Sequence[Sequence[float]] | None = None,
        return_base_argmax: bool = False,
    ) -> PathVerification:
        return verify_paths_from_decode_state(
            decode_state,
            proposed_paths,
            clone_decode_state=self.clone_decode_state,
            advance_decode_state=self.advance_decode_state,
            acceptance_mode=acceptance_mode,
            rng=rng,
            draft_probs_per_path=draft_probs_per_path,
            return_base_argmax=return_base_argmax,
        )

    @torch.no_grad()
    def verify_paths_batched_single_pass(
        self,
        prefix_token_ids: Sequence[int],
        proposed_paths: Sequence[Sequence[int]],
        *,
        acceptance_mode: str = "argmax",
        rng: random.Random | None = None,
        draft_probs_per_path: Sequence[Sequence[float]] | None = None,
        return_base_argmax: bool = False,
        per_path_forward: bool = False,
    ) -> PathVerification:
        """Verify paths against the base model.

        When per_path_forward is False (default), one batched forward per round is used.
        When True, one forward per path (for backends that do not batch correctly).
        """
        return verify_paths_batched_single_pass(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            prefix_token_ids=prefix_token_ids,
            proposed_paths=proposed_paths,
            acceptance_mode=acceptance_mode,
            rng=rng,
            draft_probs_per_path=draft_probs_per_path,
            return_base_argmax=return_base_argmax,
            per_path_forward=per_path_forward,
        )

    @torch.no_grad()
    def verify_paths_flattened_tree(
        self,
        prefix_token_ids: Sequence[int],
        proposed_paths: Sequence[Sequence[int]],
        *,
        acceptance_mode: str = "argmax",
        rng: random.Random | None = None,
        draft_probs_per_path: Sequence[Sequence[float]] | None = None,
        return_base_argmax: bool = False,
    ) -> PathVerification:
        """Verify paths in one forward over a flattened tree with a tree-shaped attention mask."""
        return verify_paths_flattened_tree(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            prefix_token_ids=prefix_token_ids,
            proposed_paths=proposed_paths,
            acceptance_mode=acceptance_mode,
            rng=rng,
            draft_probs_per_path=draft_probs_per_path,
            return_base_argmax=return_base_argmax,
        )

    @torch.no_grad()
    def verify_paths(self, prefix_token_ids: Sequence[int], proposed_paths: Sequence[Sequence[int]]) -> PathVerification:
        """Compatibility alias for single-pass batched verification."""
        return self.verify_paths_batched_single_pass(prefix_token_ids, proposed_paths)

