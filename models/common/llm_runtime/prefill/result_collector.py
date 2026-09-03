# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Streaming synchronized readback and prefill result collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch

import ttnn
from models.common.llm_runtime.prefill.config import PrefillRuntimeConfig
from models.common.llm_runtime.prefill.postprocess import PrefillPostprocessor
from models.common.llm_runtime.prefill.sampling_helpers import _TILE_SIZE, _merge_log_probs, _select_sample_log_prob
from models.common.llm_runtime.prefill.signatures import PreparedPrefill
from models.common.llm_runtime.tensor_resources import attach_cleanup_failures, raise_cleanup_failures
from models.common.sampling import SamplingParams


@dataclass(frozen=True)
class InvocationResult:
    value: Any
    owned: Any
    replay_ownership: Any | None = None


class PrefillResultAssembler:
    """Read each result before reuse, restore source rows, and release it."""

    def __init__(
        self,
        config: PrefillRuntimeConfig,
        *,
        postprocessor: PrefillPostprocessor,
        release_transient: Callable[[Any], list[BaseException]],
    ) -> None:
        self.config = config
        self.postprocessor = postprocessor
        self._release_transient = release_transient

    def configure(self, config: PrefillRuntimeConfig) -> None:
        self.config = config

    def assemble(
        self,
        prepared_results: Iterable[tuple[PreparedPrefill, InvocationResult]],
        *,
        batch_size: int,
        sampling_params: SamplingParams | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        sampled = sampling_params is not None
        config = self.config
        vocab_size = int(config.model.vocab_size)
        cluster_shape = config.cluster_shape
        output_logits = torch.zeros(batch_size, 1, vocab_size)
        output_tokens = torch.zeros(batch_size, dtype=torch.int64)
        row_log_probs: list[tuple[tuple[int, ...], Any]] = []

        # Keep this loop streaming. Trace replay can reuse one persistent
        # sampled-output buffer, so each value must be consumed before the
        # caller advances its prepared-results generator.
        for prepared, result in prepared_results:
            request = prepared.request
            if sampled != (prepared.sampling_params is not None):
                raise ValueError("prefill result sampling path disagrees with the public request")
            try:
                host_output = config.output_reader.read_synchronized(result.value)
                if isinstance(host_output, tuple):
                    if len(host_output) != 2:
                        raise TypeError("runtime output tuple must contain (output, log_probs)")
                    host_primary, host_log_probs = host_output
                else:
                    host_primary, host_log_probs = host_output, None
                if sampled:
                    uses_static_q128 = self.postprocessor.uses_static_q128_topk(
                        request,
                        prepared.sampling_path,
                    )
                    output_rows = _TILE_SIZE if uses_static_q128 else self.postprocessor.sampling_batch_size(request)
                    sampled_tokens = process_output_tokens(host_primary, output_rows, cluster_shape)
                    for local_row, source_row in enumerate(request.source_rows):
                        if request.kind == "batched":
                            token_index = local_row
                        elif uses_static_q128:
                            token_index = (request.last_token_indices[0] - request.cached_tokens[0]) % _TILE_SIZE
                        else:
                            token_index = 0
                        output_tokens[source_row] = sampled_tokens.reshape(-1)[token_index].to(torch.int64)
                    if host_log_probs is not None:
                        if uses_static_q128:
                            host_log_probs = _select_sample_log_prob(host_log_probs, token_index)
                        row_log_probs.append((request.source_rows, host_log_probs))
                elif request.kind == "batched":
                    if isinstance(host_primary, list):
                        for local_row, (source_row, last_token, cached_tokens) in enumerate(
                            zip(request.source_rows, request.last_token_indices, request.cached_tokens)
                        ):
                            output_logits[source_row] = process_output_prefill(
                                host_primary[local_row],
                                (last_token - cached_tokens) % _TILE_SIZE,
                                vocab_size,
                                cluster_shape,
                            )
                    else:
                        if (
                            isinstance(host_primary, ttnn.Tensor)
                            and host_primary.storage_type() != ttnn.StorageType.HOST
                        ):
                            raise ValueError("prefill output must be on host")
                        combined = concat_host_output(host_primary, cluster_shape)
                        for local_row, source_row in enumerate(request.source_rows):
                            output_logits[source_row] = combined[0, 0, local_row, :vocab_size].float()
                else:
                    relative_last = (request.last_token_indices[0] - request.cached_tokens[0]) % _TILE_SIZE
                    if request.kind == "single" and not request.uses_chunked_prefill:
                        relative_last = 0
                    output_logits[request.source_rows[0]] = process_output_prefill(
                        host_primary,
                        relative_last,
                        vocab_size,
                        cluster_shape,
                    )
            except BaseException as primary:
                failures = self._release_transient(result.owned)
                attach_cleanup_failures(primary, failures)
                raise
            failures = self._release_transient(result.owned)
            if failures:
                raise_cleanup_failures(failures)

        if sampled:
            return output_tokens, _merge_log_probs(row_log_probs, batch_size)
        return output_logits


def concat_host_output(value, cluster_shape):
    if isinstance(value, torch.Tensor):
        return value
    tensors = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(value)]
    rows, columns = cluster_shape
    mesh = [tensors[index : index + columns] for index in range(0, len(tensors), columns)]
    return torch.cat([torch.cat(row, dim=-1) for row in mesh], dim=1)


def process_output_prefill(value, row, vocab_size, cluster_shape):
    if isinstance(value, ttnn.Tensor) and value.storage_type() != ttnn.StorageType.HOST:
        raise ValueError("prefill output must be on host")
    output = concat_host_output(value, cluster_shape)
    return output[0, 0, int(row), :vocab_size].float()


def process_output_tokens(value, batch_size, cluster_shape):
    if isinstance(value, ttnn.Tensor):
        replicas = ttnn.get_device_tensors(value)
        if not replicas:
            raise ValueError("sampled prefill output has no device tensors")
        # Sampling outputs are replicated. Convert only the first replica.
        output = ttnn.to_torch(replicas[0])
    else:
        output = value
    if output.ndim >= 4:
        if int(output.shape[2]) >= batch_size:
            output = output[0, 0, :batch_size, 0]
        elif int(output.shape[3]) >= batch_size:
            output = output[0, 0, 0, :batch_size]
    return output.reshape(-1)[:batch_size].to(torch.int64)
