# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM-compatible generator interface for efficient serving"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

import ttnn

from .generator import Generator


@dataclass
class SamplingParams:
    """vLLM-compatible sampling parameters"""

    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    max_tokens: Optional[int] = 16
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True


@dataclass
class RequestOutput:
    """Output for a single request"""

    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List["CompletionOutput"]
    finished: bool


@dataclass
class CompletionOutput:
    """Output for a single completion"""

    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: float
    logprobs: Optional[List[Dict[int, float]]] = None
    finish_reason: Optional[str] = None


class VLLMGenerator(Generator):
    """
    vLLM-compatible generator interface.

    Provides efficient batch generation with continuous batching support.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: ttnn.Device,
        max_batch_size: int = 256,
        max_model_len: int = 2048,
        enable_chunked_prefill: bool = True,
        max_num_batched_tokens: Optional[int] = None,
    ):
        super().__init__(model, tokenizer, device)
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_batched_tokens = max_num_batched_tokens or max_model_len

        # Request tracking
        self.active_requests: Dict[str, "RequestState"] = {}
        self.request_counter = 0

    def add_request(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Add a generation request to the queue.

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            request_id: Optional request ID

        Returns:
            request_id: ID of the added request
        """
        if request_id is None:
            request_id = f"req_{self.request_counter}"
            self.request_counter += 1

        sampling_params = sampling_params or SamplingParams()

        # Tokenize prompt
        token_ids = self.tokenizer.encode(prompt)

        # Create request state
        request_state = RequestState(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            generated_tokens=[],
            finished=False,
        )

        self.active_requests[request_id] = request_state
        return request_id

    def step(self) -> List[RequestOutput]:
        """
        Execute one generation step for all active requests.

        Returns:
            List of request outputs with generated tokens
        """
        if not self.active_requests:
            return []

        # Batch active requests
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        request_ids = []

        for req_id, req_state in self.active_requests.items():
            if not req_state.finished:
                # Prepare inputs
                input_ids = req_state.get_input_ids()
                attention_mask = req_state.get_attention_mask()
                position_ids = req_state.get_position_ids()

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_position_ids.append(position_ids)
                request_ids.append(req_id)

        if not batch_input_ids:
            return []

        # Pad and create batch
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = self._pad_sequences(batch_input_ids, max_len)
        padded_attention_mask = self._pad_sequences(batch_attention_mask, max_len, pad_value=0)
        padded_position_ids = self._pad_sequences(batch_position_ids, max_len)

        # Convert to tensors
        input_ids_tensor = ttnn.from_torch(torch.tensor(padded_input_ids), device=self.device)
        attention_mask_tensor = ttnn.from_torch(torch.tensor(padded_attention_mask), device=self.device)
        position_ids_tensor = ttnn.from_torch(torch.tensor(padded_position_ids), device=self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                position_ids=position_ids_tensor,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]  # Get last token logits

        # Sample next tokens for each request
        outputs = []
        for i, req_id in enumerate(request_ids):
            req_state = self.active_requests[req_id]
            req_logits = logits[i]

            # Apply sampling
            next_token_id = self._sample_token(req_logits, req_state.sampling_params)

            # Update request state
            req_state.add_token(next_token_id)

            # Check stopping conditions
            if self._should_stop(req_state):
                req_state.finished = True

            # Create output
            output = self._create_request_output(req_state)
            outputs.append(output)

        return outputs

    def generate_batch(
        self,
        prompts: List[str],
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
    ) -> List[List[CompletionOutput]]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters (single or per-prompt)

        Returns:
            List of completion outputs for each prompt
        """
        # Add all requests
        request_ids = []
        if isinstance(sampling_params, SamplingParams) or sampling_params is None:
            # Use same params for all prompts
            params = sampling_params or SamplingParams()
            for prompt in prompts:
                req_id = self.add_request(prompt, params)
                request_ids.append(req_id)
        else:
            # Use individual params
            for prompt, params in zip(prompts, sampling_params):
                req_id = self.add_request(prompt, params)
                request_ids.append(req_id)

        # Generate until all requests finish
        results = {req_id: [] for req_id in request_ids}
        while any(not self.active_requests[req_id].finished for req_id in request_ids):
            outputs = self.step()
            for output in outputs:
                if output.request_id in results:
                    results[output.request_id] = output.outputs

        # Clean up and return
        for req_id in request_ids:
            del self.active_requests[req_id]

        return [results[req_id] for req_id in request_ids]

    def _sample_token(
        self,
        logits: ttnn.Tensor,
        sampling_params: SamplingParams,
    ) -> int:
        """Sample next token based on sampling parameters"""
        # Apply temperature
        if sampling_params.temperature > 0:
            logits = logits / sampling_params.temperature

        # Apply penalties
        # TODO: Implement presence and frequency penalties

        # Apply top-k filtering
        if sampling_params.top_k > 0:
            top_k_logits, top_k_indices = ttnn.topk(logits, k=min(sampling_params.top_k, logits.shape[-1]))
            logits = ttnn.full_like(logits, float("-inf"))
            logits = ttnn.scatter(logits, -1, top_k_indices, top_k_logits)

        # Apply top-p filtering
        if sampling_params.top_p < 1.0:
            sorted_logits, sorted_indices = ttnn.sort(logits, descending=True)
            cumulative_probs = ttnn.cumsum(ttnn.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > sampling_params.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = ttnn.zeros_like(logits, dtype=ttnn.bool)
            indices_to_remove = ttnn.scatter(indices_to_remove, -1, sorted_indices, sorted_indices_to_remove)
            logits = ttnn.where(indices_to_remove, float("-inf"), logits)

        # Sample
        if sampling_params.temperature == 0:
            # Greedy
            next_token = ttnn.argmax(logits, dim=-1)
        else:
            # Sample from distribution
            probs = ttnn.softmax(logits, dim=-1)
            next_token = ttnn.multinomial(probs, num_samples=1)

        return int(next_token.item())

    def _should_stop(self, request_state: "RequestState") -> bool:
        """Check if generation should stop for a request"""
        params = request_state.sampling_params

        # Check max tokens
        if params.max_tokens and len(request_state.generated_tokens) >= params.max_tokens:
            return True

        # Check stop tokens
        if params.stop_token_ids:
            last_token = request_state.generated_tokens[-1]
            if last_token in params.stop_token_ids:
                return True

        # Check EOS token
        if hasattr(self.tokenizer, "eos_token_id"):
            if request_state.generated_tokens[-1] == self.tokenizer.eos_token_id:
                return True

        return False

    def _create_request_output(self, request_state: "RequestState") -> RequestOutput:
        """Create output for a request"""
        # Decode generated tokens
        generated_text = self.tokenizer.decode(
            request_state.generated_tokens, skip_special_tokens=request_state.sampling_params.skip_special_tokens
        )

        completion = CompletionOutput(
            index=0,
            text=generated_text,
            token_ids=request_state.generated_tokens.copy(),
            cumulative_logprob=0.0,  # TODO: Track logprobs
            finish_reason="stop" if request_state.finished else None,
        )

        return RequestOutput(
            request_id=request_state.request_id,
            prompt=request_state.prompt,
            prompt_token_ids=request_state.prompt_token_ids,
            outputs=[completion],
            finished=request_state.finished,
        )

    def _pad_sequences(
        self,
        sequences: List[List[int]],
        max_len: int,
        pad_value: int = 0,
    ) -> List[List[int]]:
        """Pad sequences to same length"""
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                seq = seq + [pad_value] * (max_len - len(seq))
            padded.append(seq)
        return padded


class RequestState:
    """Internal state for a generation request"""

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        generated_tokens: List[int],
        finished: bool = False,
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.sampling_params = sampling_params
        self.generated_tokens = generated_tokens
        self.finished = finished
        self.past_key_values = None

    def get_input_ids(self) -> List[int]:
        """Get full input IDs including generated tokens"""
        return self.prompt_token_ids + self.generated_tokens

    def get_attention_mask(self) -> List[int]:
        """Get attention mask"""
        seq_len = len(self.prompt_token_ids) + len(self.generated_tokens)
        return [1] * seq_len

    def get_position_ids(self) -> List[int]:
        """Get position IDs"""
        seq_len = len(self.prompt_token_ids) + len(self.generated_tokens)
        return list(range(seq_len))

    def add_token(self, token_id: int):
        """Add generated token"""
        self.generated_tokens.append(token_id)
