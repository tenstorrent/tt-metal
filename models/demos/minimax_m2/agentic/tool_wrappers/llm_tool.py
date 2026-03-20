# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LLMTool: Orchestrator LLM using Llama 3.2 3B Instruct on TTNN.

Loads the model via tt_transformers Generator on N300.
Requires:
  - HF_MODEL env var (default: meta-llama/Llama-3.2-3B-Instruct)
  - Model weights available (HuggingFace or local cache)
  - N300 mesh device passed at init time

Generation uses internal KV cache (kv_cache=None to Generator calls),
so no external paged-attention cache needs to be allocated.
Prefill trace is compiled during warmup; decode trace is captured
lazily on the first decode step.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def _extract_outermost_json(text: str) -> Optional[str]:
    """
    Extract the first outermost JSON object from *text* by counting braces.
    Returns the JSON string or None if not found.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Try to extract a tool call from the LLM output text.

    Handles:
      - Llama 3.2 <|python_tag|>{...} format
      - ```json {...} ``` code blocks
      - Bare JSON objects with "name" key

    Returns (tool_name, args_dict) or None if no tool call is found.
    """
    candidates = []

    # 1. python_tag prefix
    tag_match = re.search(r"<\|python_tag\|>\s*(\{)", text, re.DOTALL)
    if tag_match:
        candidates.append(text[tag_match.start(1) :])

    # 2. JSON code block
    block_match = re.search(r"```(?:json)?\s*(\{)", text, re.DOTALL)
    if block_match:
        candidates.append(text[block_match.start(1) :])

    # 3. Any bare JSON object in the text
    candidates.append(text)

    for candidate in candidates:
        json_str = _extract_outermost_json(candidate)
        if json_str is None:
            continue
        try:
            obj = json.loads(json_str)
            name = obj.get("name") or obj.get("tool_name") or obj.get("function")
            args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
            if name:
                return name, args
        except json.JSONDecodeError:
            continue
    return None


class LLMTool:
    """
    Orchestrator LLM wrapper using TTNN Llama 3.2 3B Instruct.

    The model is loaded once at startup via tt_transformers Generator.
    Generation uses prefill_forward_text + decode_forward with the
    model's internal (non-paged) KV cache.
    """

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self._generator = None
        self._model_args = None
        self._init_ttnn(mesh_device)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_ttnn(self, mesh_device):
        """Load Llama 3.2 3B via tt_transformers on N300."""
        from transformers import AutoTokenizer

        import ttnn as _ttnn
        from models.tt_transformers.tt.generator import Generator
        from models.tt_transformers.tt.model import Transformer
        from models.tt_transformers.tt.model_config import ModelArgs

        hf_model = os.getenv("HF_MODEL", LLAMA_MODEL_ID)
        os.environ.setdefault("HF_MODEL", hf_model)

        logger.info(f"Loading LLM via TTNN backend: {hf_model}")

        model_args = ModelArgs(
            mesh_device,
            instruct=True,
            max_batch_size=1,
            max_seq_len=4096,
        )

        state_dict = model_args.load_state_dict()

        tt_model = Transformer(
            args=model_args,
            mesh_device=mesh_device,
            dtype=_ttnn.bfloat8_b,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(_ttnn.bfloat8_b),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self._generator = Generator([tt_model], [model_args], mesh_device, tokenizer=self.tokenizer)
        self._model_args = model_args

        # Warmup: compile prefill kernels for all supported sequence lengths.
        # Decode trace is captured lazily on the first decode_forward call.
        # kv_cache=None → model uses its own pre-allocated internal KV cache.
        self._generator.warmup_model_prefill(
            kv_cache=None,
            enable_trace=True,
            can_sample_on_device=False,
            non_greedy_decoding_on_device=False,
        )
        logger.info("LLM (TTNN Llama 3B) ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_response(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a response for the given message history.

        Args:
            messages:       List of chat messages (OpenAI format).
            tools:          Optional list of tool schemas for function calling.
            max_new_tokens: Maximum tokens to generate.
            temperature:    Sampling temperature (0 = greedy).

        Returns:
            Generated text string (may contain a tool call JSON).
        """
        return self._generate_ttnn(messages, tools, max_new_tokens, temperature)

    def parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Parse a tool call from the LLM output. Returns (name, args) or None."""
        return _parse_tool_call(text)

    def close(self):
        """
        Release traces before device close to prevent segfault.

        Must be called before ttnn.close_mesh_device() to ensure proper cleanup.
        The Generator's __del__ releases traces, but if called after device close
        it causes a segfault. Explicitly deleting the generator forces cleanup.
        """
        if self._generator is not None:
            # Force the generator's __del__ to run now by deleting it
            del self._generator
            self._generator = None
        logger.info("LLMTool closed (traces released).")

    # ------------------------------------------------------------------
    # TTNN generation
    # ------------------------------------------------------------------

    def _generate_ttnn(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """
        Run prefill_forward_text → decode_forward loop on device.

        prefill_forward_text returns a torch tensor of shape [batch, 1, vocab_size].
        decode_forward returns (logits [batch, 1, vocab_size], log_probs) as torch tensors.
        Both calls use kv_cache=None so the model's internal pre-allocated cache is used.
        """
        template_kwargs = dict(add_generation_prompt=True, tokenize=True, return_tensors="pt")
        if tools:
            template_kwargs["tools"] = tools

        input_ids = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]

        prompt_tokens = input_ids[0].tolist()
        prompt_len = len(prompt_tokens)

        tokens_tensor = torch.tensor([prompt_tokens], dtype=torch.long)  # [1, S]
        prompt_lens = torch.tensor([prompt_len], dtype=torch.long)

        # --- Prefill ---
        # warmup_prefill=True triggers kernel compilation on first call.
        # Returns torch tensor [1, 1, vocab_size] with logits for last prompt token.
        prefill_logits = self._generator.prefill_forward_text(
            tokens=tokens_tensor,
            kv_cache=None,
            prompt_lens=prompt_lens,
            empty_slots=[0],
            enable_trace=True,
            warmup_prefill=True,
        )

        # First generated token from prefill output
        next_token = int(prefill_logits[0, 0, :].argmax())

        generated_tokens: List[int] = []
        current_pos = torch.tensor([prompt_len], dtype=torch.int32)

        # --- Decode loop ---
        # decode_forward with enable_trace=True captures decode trace on first call.
        # Returns (logits [1, 1, vocab_size], log_probs) as torch tensors.
        for _ in range(max_new_tokens):
            if next_token == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)  # [1, 1]
            decode_out = self._generator.decode_forward(
                tokens=next_token_tensor,
                start_pos=current_pos,
                kv_cache=None,
                enable_trace=True,
            )
            # decode_forward returns (logits, log_probs) when read_from_device=True
            decode_logits = decode_out[0] if isinstance(decode_out, (tuple, list)) else decode_out
            current_pos = current_pos + 1
            next_token = int(decode_logits[0, 0, :].argmax())

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()
