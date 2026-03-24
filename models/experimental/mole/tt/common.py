# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from dataclasses import dataclass
from typing import Any, Callable
import weakref

import torch
import ttnn


DEFAULT_DTYPE = ttnn.float32
LOW_FIDELITY_LINEAR_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
<<<<<<< HEAD
class _TraceReleaseHookTable:
    """Weakref hooks keyed by ``id(device)`` — avoids a module-level dict literal."""

    __slots__ = ("_by_device_id",)

    def __init__(self) -> None:
        self._by_device_id: dict[int, list[weakref.ReferenceType]] = {}

    def register(self, *, device: Any, hook: Callable[[], None]) -> None:
        key = id(device)
        hooks = self._by_device_id.get(key)
        if hooks is None:
            hooks = []
            self._by_device_id[key] = hooks
        if hasattr(hook, "__self__") and hook.__self__ is not None:
            hook_ref: weakref.ReferenceType = weakref.WeakMethod(hook)
        else:
            hook_ref = weakref.ref(hook)
        hooks.append(hook_ref)

    def release_for_device(self, *, device: Any) -> None:
        key = id(device)
        hooks = self._by_device_id.get(key, [])
        live_refs = []
        for hook_ref in hooks:
            hook = hook_ref()
            if hook is None:
                continue
            live_refs.append(hook_ref)
            with contextlib.suppress(Exception):
                hook()
        self._by_device_id[key] = live_refs


_TRACE_RELEASE_HOOKS = _TraceReleaseHookTable()
=======
_ACTIVE_TRACE_RELEASE_HOOKS: dict[int, list[weakref.ReferenceType]] = {}
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f


def register_trace_release_hook(*, device: Any, hook: Callable[[], None]) -> None:
    _TRACE_RELEASE_HOOKS.register(device=device, hook=hook)


def release_active_traces_for_device(*, device: Any) -> None:
    _TRACE_RELEASE_HOOKS.release_for_device(device=device)


@dataclass(frozen=True)
class TtRuntimeOptions:
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG
    activation_memory_config: ttnn.MemoryConfig | None = None
    dtype: ttnn.DataType = DEFAULT_DTYPE


def timeseries_input_to_device(
    torch_input: torch.Tensor,
    *,
    device: Any,
    dtype: ttnn.DataType = DEFAULT_DTYPE,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
) -> ttnn.Tensor:
    if torch_input.ndim != 3:
        raise ValueError(f"expected [batch, seq_len, channels], got {tuple(torch_input.shape)}")

    return ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def time_marks_input_to_device(
    torch_input: torch.Tensor,
    *,
    device: Any,
    dtype: ttnn.DataType = DEFAULT_DTYPE,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
) -> ttnn.Tensor:
    if torch_input.ndim != 3:
        raise ValueError(f"expected [batch, seq_len, features], got {tuple(torch_input.shape)}")

    return ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def upload_timeseries_and_marks_to_device(
    *,
    model: Any,
    device: Any,
    torch_input: torch.Tensor,
    torch_input_mark: torch.Tensor,
    memory_config: ttnn.MemoryConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    input_signature = (
        int(torch_input.data_ptr()),
        tuple(torch_input.shape),
        tuple(torch_input.stride()),
        str(torch_input.dtype),
        str(torch_input.device),
        int(getattr(torch_input, "_version", -1)),
        int(torch_input_mark.data_ptr()),
        tuple(torch_input_mark.shape),
        tuple(torch_input_mark.stride()),
        str(torch_input_mark.dtype),
        str(torch_input_mark.device),
        int(getattr(torch_input_mark, "_version", -1)),
    )

    cache_key = (
        id(device),
        tuple(torch_input.shape),
        tuple(torch_input_mark.shape),
        model.dtype,
        id(memory_config),
    )
    cache = getattr(model, "_input_buffer_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_input_buffer_cache", cache)
    signature_cache = getattr(model, "_input_signature_cache", None)
    if signature_cache is None:
        signature_cache = {}
        setattr(model, "_input_signature_cache", signature_cache)

    cached = cache.get(cache_key)
    if cached is not None and signature_cache.get(cache_key) == input_signature:
        return cached

    host_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        dtype=model.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )
    host_marks = ttnn.from_torch(
        torch_input_mark.unsqueeze(0),
        dtype=model.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )

    if cached is None:
        try:
            # Active traces can make subsequent buffer allocations unsafe; release
            # trace caches before allocating new device input buffers.
            release_active_traces_for_device(device=device)
            cached = (
                ttnn.allocate_tensor_on_device(host_input.spec, device),
                ttnn.allocate_tensor_on_device(host_marks.spec, device),
            )
            cache[cache_key] = cached
        except Exception:
            return (
                timeseries_input_to_device(
                    torch_input,
                    device=device,
                    dtype=model.dtype,
                    memory_config=memory_config,
                ),
                time_marks_input_to_device(
                    torch_input_mark,
                    device=device,
                    dtype=model.dtype,
                    memory_config=memory_config,
                ),
            )

    tt_input, tt_marks = cached
    ttnn.copy_host_to_device_tensor(host_input, tt_input)
    ttnn.copy_host_to_device_tensor(host_marks, tt_marks)
    signature_cache[cache_key] = input_signature
    return tt_input, tt_marks


def to_torch_with_cached_host(
    *,
    model: Any,
    device_tensor: ttnn.Tensor,
    cache_name: str,
) -> torch.Tensor:
    cache = getattr(model, "_output_host_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_output_host_cache", cache)

    key = (cache_name, tuple(device_tensor.shape))
    host_tensor = cache.get(key)
    if host_tensor is None:
        try:
            host_tensor = ttnn.allocate_tensor_on_host(device_tensor.spec)
            cache[key] = host_tensor
        except Exception:
            return ttnn.to_torch(device_tensor)

    try:
        ttnn.copy_device_to_host_tensor(device_tensor, host_tensor)
        return ttnn.to_torch(host_tensor)
    except Exception:
        return ttnn.to_torch(device_tensor)


def validate_timeseries_input(
    input_tensor: ttnn.Tensor, *, seq_len: int | None = None, input_dim: int | None = None
) -> None:
    if len(input_tensor.shape) != 4:
        raise ValueError(f"expected [1, batch, seq_len, channels], got {tuple(input_tensor.shape)}")
    if input_tensor.shape[0] != 1:
        raise ValueError(f"expected leading dimension 1, got {input_tensor.shape[0]}")
    if seq_len is not None and input_tensor.shape[2] != seq_len:
        raise ValueError(f"expected seq_len={seq_len}, got seq_len={input_tensor.shape[2]}")
    if input_dim is not None and input_tensor.shape[3] != input_dim:
        raise ValueError(f"expected input_dim={input_dim}, got input_dim={input_tensor.shape[3]}")


def validate_time_marks(
    input_marks: ttnn.Tensor, *, seq_len: int | None = None, expected_features: int | None = None
) -> None:
    if len(input_marks.shape) != 4:
        raise ValueError(f"expected [1, batch, seq_len, features], got {tuple(input_marks.shape)}")
    if input_marks.shape[0] != 1:
        raise ValueError(f"expected leading dimension 1, got {input_marks.shape[0]}")
    if seq_len is not None and input_marks.shape[2] != seq_len:
        raise ValueError(f"expected seq_len={seq_len}, got seq_len={input_marks.shape[2]}")
    if expected_features is not None and input_marks.shape[3] != expected_features:
        raise ValueError(f"expected {expected_features} time features, got {input_marks.shape[3]}")


def upload_linear(
    module_or_weight,
    bias=None,
    *,
    device: object,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
) -> dict[str, ttnn.Tensor]:
    if isinstance(module_or_weight, torch.nn.Linear):
        weight, bias = module_or_weight.weight, module_or_weight.bias
    else:
        weight = module_or_weight
    if bias is None:
        raise ValueError("upload_linear requires a bias tensor; nn.Linear(bias=False) is not supported")
    return {
        "weight": ttnn.from_torch(
            weight.detach(),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        ),
        "bias": ttnn.from_torch(
            bias.detach().reshape(1, -1),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        ),
    }


def upload_vector(
    vector: torch.Tensor,
    *,
    device: object,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        vector.detach().reshape(1, 1, 1, -1),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def default_activation_memory_config() -> ttnn.MemoryConfig:
    # Keep intermediates in L1 even when parameters are in DRAM.
    return ttnn.L1_MEMORY_CONFIG


def moving_average_projection_matrix(*, seq_len: int, kernel_size: int) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    pad = (kernel_size - 1) // 2
    projection = torch.zeros((seq_len, seq_len), dtype=torch.float32)
    scale = 1.0 / kernel_size
    for output_index in range(seq_len):
        for offset in range(-pad, pad + 1):
            input_index = min(max(output_index + offset, 0), seq_len - 1)
            projection[output_index, input_index] += scale
    return projection


def router_time_major_to_channel_major_permutation(*, channels: int, num_predictions: int) -> torch.Tensor:
    permutation = []
    for channel_index in range(channels):
        for prediction_index in range(num_predictions):
            permutation.append(prediction_index * channels + channel_index)
    return torch.tensor(permutation, dtype=torch.long)


def permute_temporal_router_mlp_to_channel_major(
    module: torch.nn.Sequential,
    *,
    channels: int,
    num_predictions: int,
) -> dict[str, torch.Tensor]:
    permutation = router_time_major_to_channel_major_permutation(channels=channels, num_predictions=num_predictions)
    permutation_device = module[0].weight.device
    permutation = permutation.to(permutation_device)

    linear_1 = module[0]
    linear_2 = module[2]

    linear_1_weight = linear_1.weight.detach().index_select(0, permutation)
    linear_1_bias = linear_1.bias.detach().index_select(0, permutation)

    linear_2_weight = linear_2.weight.detach().index_select(0, permutation).index_select(1, permutation)
    linear_2_bias = linear_2.bias.detach().index_select(0, permutation)

    return {
        "linear_1_weight": linear_1_weight,
        "linear_1_bias": linear_1_bias,
        "linear_2_weight": linear_2_weight,
        "linear_2_bias": linear_2_bias,
    }


def apply_linear(
    input_tensor: ttnn.Tensor,
    params: dict[str, ttnn.Tensor],
    *,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    kwargs = {
        "transpose_b": True,
        "memory_config": memory_config,
    }
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config
    return ttnn.linear(
        input_tensor,
        params["weight"],
        bias=params["bias"],
        **kwargs,
    )


def apply_two_layer_mlp(
    input_tensor: ttnn.Tensor,
    params: dict[str, dict[str, ttnn.Tensor]],
    *,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    hidden = apply_linear(
        input_tensor,
        params["linear_1"],
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    hidden = ttnn.relu(hidden)
    return apply_linear(
        hidden,
        params["linear_2"],
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )


def extract_initial_marks(input_marks: ttnn.Tensor) -> ttnn.Tensor:
    batch_size = input_marks.shape[1]
    feature_dim = input_marks.shape[3]
    initial = ttnn.slice(input_marks, (0, 0, 0, 0), (1, batch_size, 1, feature_dim))
    return ttnn.reshape(initial, (1, 1, batch_size, feature_dim))


def project_individual_channels(
    input_tensor: ttnn.Tensor,
    params: list[dict[str, ttnn.Tensor]],
    *,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
) -> ttnn.Tensor:
    outputs = []
    for channel_index, channel_params in enumerate(params):
        channel_input = ttnn.slice(
            input_tensor,
            (0, channel_index, 0, 0),
            (1, channel_index + 1, input_tensor.shape[2], input_tensor.shape[3]),
        )
        outputs.append(apply_linear(channel_input, channel_params, memory_config=memory_config))
    return ttnn.concat(outputs, dim=1, memory_config=memory_config)


def moving_average_1d(
    input_tensor: ttnn.Tensor, kernel_size: int, *, memory_config=ttnn.L1_MEMORY_CONFIG
) -> ttnn.Tensor:
    batch_size = input_tensor.shape[1]
    seq_len = input_tensor.shape[2]
    channels = input_tensor.shape[3]
    pad = (kernel_size - 1) // 2

    if pad > 0:
        left = ttnn.slice(input_tensor, (0, 0, 0, 0), (1, batch_size, 1, channels))
        left = ttnn.repeat(left, ttnn.Shape((1, 1, pad, 1)))
        right = ttnn.slice(input_tensor, (0, 0, seq_len - 1, 0), (1, batch_size, seq_len, channels))
        right = ttnn.repeat(right, ttnn.Shape((1, 1, pad, 1)))
        padded = ttnn.concat([left, input_tensor, right], dim=2, memory_config=memory_config)
    else:
        padded = input_tensor

    # Build moving-window sums with a prefix sum to avoid O(kernel_size) slice/add TMs.
    prefix = ttnn.cumsum(padded, dim=2)

    zero_prefix = ttnn.slice(prefix, (0, 0, 0, 0), (1, batch_size, 1, channels))
    zero_prefix = ttnn.multiply(zero_prefix, 0.0, memory_config=memory_config)
    prefix_with_zero = ttnn.concat([zero_prefix, prefix], dim=2, memory_config=memory_config)

    window_end = ttnn.slice(
        prefix_with_zero,
        (0, 0, kernel_size, 0),
        (1, batch_size, kernel_size + seq_len, channels),
    )
    window_start = ttnn.slice(prefix_with_zero, (0, 0, 0, 0), (1, batch_size, seq_len, channels))
    running_sum = ttnn.subtract(window_end, window_start, memory_config=memory_config)

    return ttnn.multiply(running_sum, 1.0 / kernel_size, memory_config=memory_config)


def temporal_gating(
    logits: ttnn.Tensor,
    *,
    batch_size: int,
    channels: int,
    num_predictions: int,
    return_channelwise_weights: bool = True,
) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
    gating_channel_major = ttnn.reshape(logits, (1, 1, batch_size, channels, num_predictions))
    gating_channel_major = ttnn.softmax(gating_channel_major, dim=-1)
    gating_channel_major = ttnn.reshape(gating_channel_major, (1, batch_size, channels, num_predictions))
    gating_flat = ttnn.reshape(gating_channel_major, (1, batch_size * channels, 1, num_predictions))
    if not return_channelwise_weights:
        return gating_flat, None
    gating_weights = ttnn.permute(gating_channel_major, (0, 1, 3, 2))
    return gating_flat, gating_weights


def _reduce_weighted_heads_core(
    projected: ttnn.Tensor,
    gating_flat: ttnn.Tensor,
    *,
    batch_size: int,
    channels: int,
    pred_len: int,
    memory_config=ttnn.L1_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Shared tail: matmul against column gating then reshape back to NTC layout."""
    gating_column = ttnn.permute(gating_flat, (0, 1, 3, 2))
    reduced = ttnn.matmul(projected, gating_column, memory_config=memory_config)
    reduced = ttnn.reshape(reduced, (1, batch_size, channels, pred_len))
    return ttnn.permute(reduced, (0, 1, 3, 2))


def reduce_weighted_heads_batch_major(
    projected: ttnn.Tensor,
    gating_flat: ttnn.Tensor,
    *,
    batch_size: int,
    channels: int,
    pred_len: int,
    num_predictions: int,
    memory_config=ttnn.L1_MEMORY_CONFIG,
) -> ttnn.Tensor:
    projected = ttnn.reshape(projected, (1, batch_size * channels, pred_len, num_predictions))
    return _reduce_weighted_heads_core(
        projected,
        gating_flat,
        batch_size=batch_size,
        channels=channels,
        pred_len=pred_len,
        memory_config=memory_config,
    )


@dataclass
class _ExpertPredictionTraceState:
    trace_id: int
    prediction: ttnn.Tensor
    input_ids: tuple[int, int]
    device: Any


class TtExpertBase:
    """Shared init, trace capture, and torch I/O for all TT expert classes."""

    def _init_base(self, config, reference_model, options):
        """
        Base initialization sequence for device specialized expert layers.
        Sets up memory configurations and captures data type parameters.
        """
        options = options or TtRuntimeOptions()
        self.config = config
        self.reference_model = reference_model
        self.parameter_memory_config = options.memory_config
        self.activation_memory_config = (
            default_activation_memory_config()
            if options.activation_memory_config is None
            else options.activation_memory_config
        )
        self.memory_config = self.activation_memory_config
        self.dtype = options.dtype
        self._cached_device = None
        self._tt_parameters = None
        self._prediction_trace_state = None
        self._trace_capture_enabled = True

    def _release_prediction_trace(self) -> None:
        state = self._prediction_trace_state
        if state is None:
            return
        with contextlib.suppress(Exception):
            ttnn.release_trace(state.device, state.trace_id)
        self._prediction_trace_state = None

    def __del__(self):
        self._release_prediction_trace()

    def forward_prediction(self, input_tensor: ttnn.Tensor, input_marks: ttnn.Tensor) -> ttnn.Tensor:
        device = input_tensor.device()
        if not self._trace_capture_enabled:
            return self.forward(input_tensor, input_marks)

        state = self._prediction_trace_state
        current_ids = (id(input_tensor), id(input_marks))

        if state is None or state.input_ids != current_ids or state.device is not device:
            self._release_prediction_trace()
            prediction = self.forward(input_tensor, input_marks)
            ttnn.synchronize_device(device)

            trace_id = None
            try:
                trace_id = ttnn.begin_trace_capture(device, cq_id=0)
                prediction = self.forward(input_tensor, input_marks)
                ttnn.end_trace_capture(device, trace_id, cq_id=0)
            except Exception:
                self._trace_capture_enabled = False
                if trace_id is not None:
                    with contextlib.suppress(Exception):
                        ttnn.end_trace_capture(device, trace_id, cq_id=0)
                        ttnn.release_trace(device, trace_id)
                self._release_prediction_trace()
                return prediction
            self._prediction_trace_state = _ExpertPredictionTraceState(
                trace_id=trace_id,
                prediction=prediction,
                input_ids=current_ids,
                device=device,
            )

        st = self._prediction_trace_state
        ttnn.execute_trace(device, st.trace_id, cq_id=0, blocking=False)
        return st.prediction

    def forward_from_torch_input(self, torch_input: torch.Tensor, *, input_marks: torch.Tensor, device) -> ttnn.Tensor:
        tt_input, tt_marks = upload_timeseries_and_marks_to_device(
            model=self,
            device=device,
            torch_input=torch_input,
            torch_input_mark=input_marks,
            memory_config=self.activation_memory_config,
        )
        return self.forward_prediction(tt_input, tt_marks)
