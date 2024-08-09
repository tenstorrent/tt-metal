# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from loguru import logger
from dataclasses import dataclass, field
from functools import partial
from copy import deepcopy
import torch


class AttentionMaskConverter:
    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, "str"] = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # If shape is not cached, create a new causal mask and cache it
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )

        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window + 1

            context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.Tensor, attention_mask: torch.Tensor, unmasked_value: Union[bool, float]
    ):
        # fmt: off
        """
        Attend to all tokens in masked rows from the expanded attention mask, for example the relevant first rows when
        using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        Details: https://github.com/pytorch/pytorch/issues/110213

        `expanded_mask` is [bsz, num_masks, tgt_seq_len, src_seq_len] or [bsz, tgt_seq_len, src_seq_len].
        `attention_mask` is [bsz, src_seq_len].

        The dimension num_masks of `expanded_mask` is most often 1, but it can also be the number of heads in the case of alibi attention bias.

        For example, if `attention_mask` is
        ```
        [[0, 0, 1],
         [1, 1, 1],
         [0, 1, 1]]
        ```
        and `expanded_mask` is (e.g. here left-padding case)
        ```
        [[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[0, 0, 0],
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        then the modified `expanded_mask` will be
        ```
        [[[[1, 1, 1],   <-- modified
           [1, 1, 1],   <-- modified
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[1, 1, 1],   <-- modified
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        """
        # fmt: on

        # Get the index of the first non-zero value for every sample in the batch.
        # In the above example, indices = [[2], [0], [1]]]
        tmp = torch.arange(attention_mask.shape[1], 0, -1)
        indices = torch.argmax(attention_mask.cpu() * tmp, 1, keepdim=True)

        # Find the batch indexes that have unattended tokens on the leftmost side (e.g. [0, 0, 1, 1, 1]), for which the first rows of the
        # expanded mask will be completely unattended.
        left_masked_rows = torch.where(indices > 0)[0]

        if left_masked_rows.shape[0] == 0:
            return expanded_mask
        indices = indices[left_masked_rows]

        max_len = torch.max(indices)
        range_tensor = torch.arange(max_len).unsqueeze(0)
        range_tensor = range_tensor.repeat(indices.size(0), 1)

        # Avoid unmasking tokens at relevant target positions (on the row axis), by rather unmasking possibly several times the first row that should always be unmasked as we filtered out the batch above.
        range_tensor[range_tensor >= indices] = 0

        # TODO: we may drop support for 3D attention mask as the refactor from Patrick maybe dropped this case
        if expanded_mask.dim() == 4:
            num_masks = expanded_mask.shape[1]
            if num_masks == 1:
                # Broadcast [left_masked_rows, 1], [left_masked_rows, max_len]
                mask_slice = (left_masked_rows[:, None], 0, range_tensor)
            else:
                # Broadcast [left_masked_rows, 1, 1], [1, num_masks, 1], [left_masked_rows, 1, max_len]
                mask_slice = (
                    left_masked_rows[:, None, None],
                    torch.arange(num_masks)[None, :, None],
                    range_tensor[:, None, :],
                )
        else:
            # Broadcast [left_masked_rows, 1], [left_masked_rows, max_len]
            mask_slice = (left_masked_rows[:, None], range_tensor)

        expanded_mask[mask_slice] = unmasked_value

        return expanded_mask


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


class LogitsProcessorList(list):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


from abc import ABC


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation.

    If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
    output_scores=True` to `generate`.
    """

    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return is_done


class MaxTimeCriteria(StoppingCriteria):
    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return time.time() - self.initial_timestamp > self.max_time


class StoppingCriteriaList(list):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None
