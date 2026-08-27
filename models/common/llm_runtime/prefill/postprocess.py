# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill postprocessing and device-sampling behavior."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

import ttnn
from models.common.llm_runtime.prefill.config import PrefillRuntimeConfig
from models.common.llm_runtime.prefill.inputs import PrefillPositionInputs
from models.common.llm_runtime.prefill.plan import PrefillRequest
from models.common.llm_runtime.prefill.sampling_helpers import _TILE_SIZE, SamplingPath, _formatted_sampling_values
from models.common.llm_runtime.prefill.signatures import PreparedPrefill
from models.common.sampling import SamplingParams

KPTSignature = tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]] | None


class PrefillPostprocessor:
    """Own prefill output selection, sampling tensors, and finalization."""

    def __init__(
        self,
        config: PrefillRuntimeConfig,
        *,
        allocate_device_tensors: Callable[[Any], Any],
        copy_into_device_tensors: Callable[[Any, Any], Any],
    ) -> None:
        self.config = config
        self._allocate_device_tensors = allocate_device_tensors
        self._copy_into_device_tensors = copy_into_device_tensors

    def configure(self, config: PrefillRuntimeConfig) -> None:
        self.config = config

    def validate_sampling_request(self, sampling_params: SamplingParams | None) -> None:
        if sampling_params is not None and not self.config.device_sampling_enabled:
            raise ValueError("sampling parameters were supplied while device sampling is disabled")

    def classify_sampling_path(
        self,
        request: PrefillRequest,
        sampling_params: SamplingParams | None,
    ) -> SamplingPath:
        if sampling_params is None:
            return "logits"
        if self.config.allow_force_argmax and request.kind == "single":
            values = _formatted_sampling_values(sampling_params, self.sampling_batch_size(request))
            if values[3]:
                return "argmax"
        return "topk"

    def sampling_batch_size(self, request: PrefillRequest) -> int:
        if self.config.device_sampling_enabled:
            return self.config.sampling_batch_size
        return request.padded_batch_size

    def sampling_output_rows(self, prepared: PreparedPrefill) -> int:
        # TT sampling validates K/P/T against the physical logits row count.
        # The static Q128 path retains one complete tile and selects the exact
        # logical row on the host, so its sampling tensors must span that tile.
        if self.uses_static_q128_topk(prepared.request, prepared.sampling_path):
            return _TILE_SIZE
        return self.sampling_batch_size(prepared.request)

    def uses_static_q128_topk(
        self,
        request: PrefillRequest,
        sampling_path: SamplingPath,
    ) -> bool:
        return (
            self.config.static_q128_topk_supported
            and sampling_path == "topk"
            and request.kind == "single"
            and not request.uses_chunked_prefill
            and request.padded_sequence_length == 128
        )

    def make_device_kpt(
        self,
        sampling_params: SamplingParams | None,
        batch_size: int,
        force_topk: bool,
    ) -> tuple[Any, Any, Any] | None:
        host = self.make_host_kpt(sampling_params, batch_size, force_topk)
        if host is None:
            return None
        return tuple(self._allocate_device_tensors(host))

    def make_host_kpt(
        self,
        sampling_params: SamplingParams | None,
        batch_size: int,
        force_topk: bool,
    ) -> tuple[Any, Any, Any] | None:
        if sampling_params is None:
            return None
        values = _formatted_sampling_values(sampling_params, batch_size)
        if self.config.allow_force_argmax and not force_topk and values[3]:
            return None
        k, p, temperature, _ = values
        mapper = ttnn.ReplicateTensorToMesh(self.config.mesh_device)
        return (
            ttnn.from_torch(
                torch.tensor(k, dtype=torch.int32),
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.tensor(p, dtype=torch.float32),
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.tensor(temperature, dtype=torch.float32),
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
        )

    def refresh_kpt(
        self,
        device_kpt: tuple[Any, Any, Any] | None,
        sampling_params: SamplingParams | None,
        batch_size: int,
        force_topk: bool,
    ) -> None:
        host_kpt = self.make_host_kpt(sampling_params, batch_size, force_topk)
        if (host_kpt is None) != (device_kpt is None):
            raise RuntimeError("sampling parameters changed the compiled sampling path")
        if host_kpt is not None:
            self._copy_into_device_tensors(host_kpt, device_kpt)

    def refresh_workspace_sampling(
        self,
        prepared: PreparedPrefill,
        *,
        kpt: tuple[Any, Any, Any] | None,
        kpt_signature: KPTSignature,
    ) -> KPTSignature:
        if prepared.sampling_path != "topk":
            return kpt_signature
        sampling_batch_size = self.sampling_output_rows(prepared)
        if prepared.sampling_params is None:
            kpt_value = None
        else:
            k, p, temperature, _ = _formatted_sampling_values(prepared.sampling_params, sampling_batch_size)
            kpt_value = k, p, temperature
        if kpt_signature != kpt_value:
            self.refresh_kpt(
                kpt,
                prepared.sampling_params,
                sampling_batch_size,
                force_topk=True,
            )
            return kpt_value
        return kpt_signature

    def make_sampling_output(self, batch_size: int) -> Any:
        return ttnn.from_torch(
            torch.zeros((1, 1, 1, int(batch_size)), dtype=torch.int32),
            device=self.config.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.config.mesh_device),
        )

    def sample_device(
        self,
        logits: Any,
        kpt: tuple[Any, Any, Any] | None,
        sampled_output: Any | None = None,
    ) -> Any:
        if kpt is None:
            return self.config.model.sampling.decode_forward(logits, tt_out_tok=sampled_output)
        return self.config.model.sampling.decode_forward(
            logits,
            k=kpt[0],
            p=kpt[1],
            temp=kpt[2],
            tt_out_tok=sampled_output,
        )

    def finish_regular_prefill(
        self,
        prepared: PreparedPrefill,
        hidden: Any,
        kpt: tuple[Any, Any, Any] | None,
        position_inputs: PrefillPositionInputs,
        *,
        sampled_output: Any | None = None,
        owned: list[Any] | None = None,
    ) -> Any:
        request = prepared.request
        relative_last = [last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens)]
        if request.kind == "batched" and not self.config.batched_prefill_batched_extract:
            hidden = ttnn.reshape(
                hidden,
                [request.padded_batch_size, 1, request.padded_sequence_length, int(hidden.shape[-1])],
            )
            outputs = []
            for local_row, last_token in enumerate(relative_last):
                logits = self.config.model.post_process_prefill_output(
                    hidden[local_row : local_row + 1],
                    last_token,
                )
                retain_owned(owned, logits)
                output = ttnn.untilize(logits, use_multicore=True)
                retain_owned(owned, output)
                outputs.append(output)
            return outputs
        if request.kind == "batched":
            padded_last = list(relative_last) + [0] * (request.padded_batch_size - len(relative_last))
            logits = self.config.model.post_process_batched_prefill_output(
                hidden,
                padded_last,
                request.padded_batch_size,
                request.padded_sequence_length,
            )
        elif self.uses_static_q128_topk(request, prepared.sampling_path) or (
            prepared.sampling_params is None and request.kind == "single" and not request.uses_chunked_prefill
        ):
            logits = self.config.model.post_process_prefill_output(hidden, relative_last[0])
        else:
            logits = self.config.model.post_process_prefill_output(
                hidden,
                relative_last[0],
                last_token_slice=(position_inputs.slice_start, position_inputs.slice_end),
                last_token_index=(position_inputs.row_index if prepared.sampling_params is not None else None),
            )
        retain_owned(owned, logits)
        if prepared.sampling_params is not None:
            selected = fit_prefill_sampling_logits(logits, self.sampling_output_rows(prepared))
            retain_owned(owned, selected)
            output = self.sample_device(selected, kpt, sampled_output)
        else:
            output = ttnn.untilize(logits, use_multicore=True)
            if request.kind == "single" and not request.uses_chunked_prefill:
                retain_owned(owned, output)
                row = relative_last[0] % _TILE_SIZE
                output = ttnn.slice(output, (0, 0, row, 0), (1, 1, row + 1, int(output.shape[-1])))
        retain_owned(owned, output)
        return output

    def finish_prefill_sequence(
        self,
        prepared: PreparedPrefill,
        final_step_output: Any,
        kpt: tuple[Any, Any, Any] | None,
        position_inputs: PrefillPositionInputs,
        *,
        sampled_output: Any | None,
        owned: list[Any],
    ) -> Any:
        if not prepared.request.uses_chunked_prefill:
            return self.finish_regular_prefill(
                prepared,
                final_step_output,
                kpt,
                position_inputs,
                sampled_output=sampled_output,
                owned=owned,
            )
        if prepared.sampling_params is not None:
            selected = fit_prefill_sampling_logits(final_step_output, self.sampling_output_rows(prepared))
            retain_owned(owned, selected)
            output = self.sample_device(selected, kpt, sampled_output)
        else:
            output = ttnn.untilize(final_step_output, use_multicore=True)
        retain_owned(owned, output)
        return output


def retain_owned(owned: list[Any] | None, value: Any) -> None:
    if owned is None or value is None or any(existing is value for existing in owned):
        return
    owned.append(value)


def without_borrowed(values: Iterable[Any], borrowed: Iterable[Any]) -> tuple[Any, ...]:
    """Remove trace/workspace-owned leaves from replay-local ownership."""

    borrowed_ids = {id(value) for value in borrowed if value is not None}

    def prune(value):
        if value is None or id(value) in borrowed_ids:
            return None
        if isinstance(value, tuple):
            kept = tuple(item for item in (prune(item) for item in value) if item is not None)
            return kept or None
        if isinstance(value, list):
            kept = [item for item in (prune(item) for item in value) if item is not None]
            return kept or None
        return value

    return tuple(item for item in (prune(value) for value in values) if item is not None)


def new_logprob_output(output: Any, persistent_sampled_output: Any | None) -> Any | None:
    if persistent_sampled_output is None:
        return None
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("sampled prefill output must contain (tokens, log_probs)")
    return output[1]


def fit_prefill_sampling_logits(logits, target_batch: int):
    target_batch = int(target_batch)
    if target_batch <= 0:
        raise ValueError("prefill sampling target batch must be positive")
    current_batch = int(logits.shape[2])
    if current_batch == target_batch:
        return logits
    if current_batch > target_batch:
        return ttnn.slice(
            logits,
            (0, 0, 0, 0),
            (logits.shape[0], logits.shape[1], target_batch, logits.shape[3]),
        )
    return ttnn.pad(logits, [(0, 0), (0, 0), (0, target_batch - current_batch), (0, 0)], value=0.0)
