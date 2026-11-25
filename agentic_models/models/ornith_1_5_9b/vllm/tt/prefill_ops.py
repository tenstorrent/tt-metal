# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-profile prefill operators preserving the reviewed decoder policy.

Runtime owners allocate every profile before capture, refresh masks/maps outside
capture, and pass physical residual tensors. No test-module imports or implicit
cache resets occur here. Canonical request profiles are selected by prefill_plan.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import ceil

import torch

import ttnn
from agentic_models.models.ornith_1_5_9b.vllm.tt.multichip_decoder import DecoderState
from agentic_models.models.ornith_1_5_9b.vllm.tt.prefill_plan import (
    FULL_PREFILL_BUCKETS,
    LINEAR_BUCKETS,
    physical_bucket,
)


def select_bucket(valid_count):
    return physical_bucket("linear_attention", valid_count)


class NormalizedLinearPrefill:
    """Wrap an existing real-weight decoder without changing its policy knobs."""

    def __init__(self, decoder, *, buckets=LINEAR_BUCKETS):
        if decoder.kind != "linear_attention":
            raise ValueError("Expected a linear-attention decoder")
        self.decoder = decoder
        self.device = decoder.device
        self.profiles = {}
        for n in buckets:
            if n not in LINEAR_BUCKETS or n > decoder.chunk_size:
                raise ValueError(f"Unreviewed physical profile {n}")
            self.profiles[n] = {
                "g_mask": self.upload(torch.ones(1, n, 8), dtype=ttnn.float32),
                "beta_mask": self.upload(torch.ones(1, n, 8)),
                "history_indices": self.upload(
                    self.history_indices(n), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                ),
                "history_output": self.upload(torch.zeros(1, 1, 2048, 3), layout=ttnn.ROW_MAJOR_LAYOUT),
                "valid_count": n,  # Constructor masks/indices already describe all n rows.
            }

    @staticmethod
    def history_indices(valid_count):
        return (
            torch.arange(valid_count, valid_count + 3, dtype=torch.int64)
            .reshape(1, 1, 1, 3)
            .expand(1, 1, 2048, 3)
            .contiguous()
        )

    def upload(self, value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=True):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.device if device else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def refresh(self, valid_count):
        """Outside capture: select the prepared profile and refresh changed inputs."""
        n = select_bucket(valid_count)
        self._refresh_profile(valid_count=valid_count, bucket=n)
        return n

    def _refresh_profile(self, *, valid_count, bucket):
        if not 0 < valid_count <= bucket or bucket not in self.profiles:
            raise ValueError("Invalid canonical profile segment")
        profile = self.profiles[bucket]
        # Each layer/profile owns these persistent tensors. They depend only on
        # valid_count, not token values, absolute position, request or cache slot.
        if profile["valid_count"] == valid_count:
            return
        # A failed partial transfer must not leave the old metadata reusable.
        profile["valid_count"] = None
        mask = (torch.arange(bucket) < valid_count).float().reshape(1, bucket, 1).expand(1, bucket, 8).contiguous()
        for name, value, dtype, layout in (
            ("g_mask", mask, ttnn.float32, ttnn.TILE_LAYOUT),
            ("beta_mask", mask, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ("history_indices", self.history_indices(valid_count), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        ):
            host = self.upload(value, dtype=dtype, layout=layout, device=False)
            ttnn.copy_host_to_device_tensor(host, profile[name])
        profile["valid_count"] = valid_count

    def attention(self, x, state, *, profile):
        d = self.decoder
        if getattr(state, "state_row", None) is not None:
            local = DecoderState(
                conv=d.read_linear_state(state, "conv"), recurrent=d.read_linear_state(state, "recurrent")
            )
            out = self.attention(x, local, profile=profile)
            d.write_linear_state(state, "conv", local.conv)
            d.write_linear_state(state, "recurrent", local.recurrent)
            return out

        n = x.shape[2]
        packed = d.linear(x, "packed_input")
        z = packed[:, :, :, 2048:3072]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 3072:3080]), (1, n, 8))
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 3104:3112], (1, n, 8)), ttnn.float32)
        g = ttnn.multiply(
            ttnn.add(a, d.dt_bias),
            d.decay,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        )
        # Preserve the original FP32 g / BF16 beta policy. A finite inactive
        # projected row contributes exactly exp(0)=1 and beta=0 to recurrence.
        g = ttnn.multiply(g, profile["g_mask"])
        beta = ttnn.multiply(beta, profile["beta_mask"])
        old_history = ttnn.reshape(ttnn.to_layout(state.conv, ttnn.ROW_MAJOR_LAYOUT), (1, 3, 2048))
        padded = packed
        if n % 32:
            padded = ttnn.pad(packed, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            ttnn.reshape(padded, (1, padded.shape[2], 3136)),
            old_history,
            *d.taps,
            512,
            512,
            1024,
            actual_start=d.conv_start,
            predecessor_carry=old_history,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
            compute_kernel_config=d.conv_compute,
        )
        # All tensor shapes/slice offsets below depend on the physical profile,
        # never valid_count. BF16 + RM matches gather's supported value/index
        # layouts; full-channel indices avoid relying on nonexistent broadcast.
        projected = ttnn.to_layout(packed[:, :, :, :2048], ttnn.ROW_MAJOR_LAYOUT)
        history_source = ttnn.concat([ttnn.reshape(old_history, (1, 1, 3, 2048)), projected], dim=2)
        history_source = ttnn.transpose(history_source, 2, 3)
        history = ttnn.gather(history_source, 3, profile["history_indices"], out=profile["history_output"])
        ttnn.copy(ttnn.transpose(history, 2, 3), state.conv)

        if n % 32:
            g = ttnn.pad(g, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
            beta = ttnn.pad(beta, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        out, new_state = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state.recurrent,
            output_final_state=True,
            compute_kernel_config=d.compute,
            chunk_size=32,
            num_key_heads=4,
            output_head_major=True,
            **d.gdn_constants,
        )
        ttnn.copy(new_state, state.recurrent)
        out = d.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm")
        out = ttnn.transformer.concatenate_heads(ttnn.reshape(out, (1, 8, padded.shape[2], 128)))
        out = ttnn.reshape(out, (1, 1, padded.shape[2], 1024))[:, :, :n, :]
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return d.reduce(d.linear(out, "linear_attn.out_proj"))

    def forward(self, residual, *, state):
        """Device-only one-bucket forward. Call refresh before request admission."""
        n = residual.shape[2]
        if residual.shape[-1] != self.decoder.hidden_width:
            raise ValueError("Preserve the decoder's inter-layer residual layout")
        profile = self.profiles[n]
        h = self.decoder.norm(residual, "input_layernorm")
        mixed = self.attention(h, state, profile=profile)
        # Match the reviewed decoder's lifetime: MLP/down projection needs the
        # L1 space formerly held by the attention-normalized residual.
        del h
        return ttnn.to_memory_config(self.decoder.finish(residual, mixed), ttnn.DRAM_MEMORY_CONFIG)


class RequestProfileLinear(NormalizedLinearPrefill):
    """Use the request's physical bucket with the segment's actual valid rows."""

    def refresh_profile(self, *, valid_count, bucket):
        self._refresh_profile(valid_count=valid_count, bucket=bucket)


@dataclass(frozen=True)
class AttentionInputs:
    rope_indices: tuple[int, ...]
    fill_pages: tuple[int, ...]
    attention_pages: tuple[int, ...]
    start: int
    valid_count: int


def make_attention_inputs(
    *, physical_bucket, supported_context, local_page_count, start, valid_count, page_mapping, page_offset=0
):
    """Pure-host plan; page_mapping has local IDs, never pool-global IDs.

    Fill -1 is inserted AFTER adding the pool offset. SDPA instead receives safe
    aliases for unused virtual pages. Extra virtual capacity is context pages
    plus a whole physical bucket, rounded to eight pages: 8704 pages for the
    262144 context / 16384 bucket. It allocates no extra physical K/V storage.
    """
    if physical_bucket not in FULL_PREFILL_BUCKETS:
        raise ValueError(f"Unprepared full-attention bucket: {physical_bucket}")
    if supported_context <= 0 or local_page_count < ceil(supported_context / 32):
        raise ValueError("Cache region cannot cover supported_context")
    if start < 0 or start % 32 or not 0 < valid_count <= physical_bucket:
        raise ValueError("Aligned body required; consume the unaligned prologue with one-token decode")
    if start + valid_count > supported_context:
        raise ValueError("Logical interval exceeds supported context")
    # Snapshot mutable scheduler maps before looking up a pure-host result.
    mapping = tuple(int(page) for page in page_mapping)
    return _cached_attention_inputs(
        physical_bucket, supported_context, local_page_count, start, valid_count, mapping, page_offset
    )


@lru_cache(maxsize=8)
def _cached_attention_inputs(
    physical_bucket, supported_context, local_page_count, start, valid_count, mapping, page_offset
):
    """Bounded immutable host plans; the key includes every validation input."""
    if len(mapping) != ceil(supported_context / 32):
        raise ValueError("Page mapping must have the configured fixed logical capacity")
    if page_offset < 0 or any(page < -1 or page >= local_page_count for page in mapping):
        raise ValueError("Page IDs must be local to the state's private/pool region, or -1 for unused pages")
    first_page, valid_end_page = start // 32, ceil((start + valid_count) / 32)
    if any(page < 0 for page in mapping[:valid_end_page]):
        raise ValueError("All causal prefix and newly written pages must have valid mappings")
    fill = tuple(
        mapping[first_page + index] + page_offset if index < ceil(valid_count / 32) else -1
        for index in range(physical_bucket // 32)
    )
    attention_capacity = ceil((ceil(supported_context / 32) + physical_bucket // 32) / 8) * 8
    # Local page zero belongs to this state, including for pooled cache regions.
    attention = tuple(page + page_offset if page >= 0 else page_offset for page in mapping)
    attention += (page_offset,) * (attention_capacity - len(attention))
    # Dummy query positions may exceed max context; embedding must never do so.
    rope = tuple(min(start + index, supported_context - 1) for index in range(physical_bucket))
    return AttentionInputs(rope, fill, attention, start, valid_count)


class NormalizedAttentionPrefill:
    """One stable binding and bucket; preserves the decoder's math/collectives.

    ``state`` remains caller-owned. This helper neither clears caches nor replaces
    their backing allocations. ``forward`` uses the decoder's existing norm,
    projection, rotary, reduction, residual and MLP implementations unchanged.
    """

    def __init__(self, layer, *, state, physical_bucket, supported_context=None):
        import torch

        import ttnn

        if layer.kind != "full_attention" or physical_bucket > layer.chunk_size:
            raise ValueError("Expected full-attention decoder with its existing chunk cap")
        self.layer, self.state, self.bucket = layer, state, physical_bucket
        self.context = supported_context or layer.config.max_position_embeddings
        if self.context > layer.config.max_position_embeddings:
            raise ValueError("Context exceeds the loaded RoPE table")
        self.global_pages = bool(getattr(state, "global_pages", False))
        self.page_count = state.key.shape[0] if self.global_pages else state.page_count or state.key.shape[0]
        self.page_offset = 0 if self.global_pages else state.page_offset
        if self.page_offset + self.page_count > state.key.shape[0]:
            raise ValueError("State region extends outside physical K/V cache")
        if state.key.shape != state.value.shape:
            raise ValueError("K/V pool geometry differs")
        self._cache_binding = (id(state.key), id(state.value), state.page_offset, state.page_count, self.global_pages)
        initial = make_attention_inputs(
            physical_bucket=self.bucket,
            supported_context=self.context,
            local_page_count=self.page_count,
            start=0,
            valid_count=1,
            page_mapping=range(ceil(self.context / 32)),
            page_offset=self.page_offset,
        )
        self._specs = {
            "rope_indices": (ttnn.uint32, lambda p: torch.tensor(p.rope_indices, dtype=torch.int32).reshape(1, -1)),
            "fill_pages": (ttnn.int32, lambda p: torch.tensor(p.fill_pages, dtype=torch.int32).reshape(1, -1)),
            "attention_pages": (
                ttnn.int32,
                lambda p: torch.tensor(p.attention_pages, dtype=torch.int32).reshape(1, -1),
            ),
            "start": (ttnn.int32, lambda p: torch.tensor([p.start], dtype=torch.int32)),
        }
        self.inputs = {
            name: ttnn.from_torch(
                pack(initial),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=layer.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(layer.device),
            )
            for name, (dtype, pack) in self._specs.items()
        }
        # These values describe the actual constructor uploads. Per-adapter
        # ownership matches inputs; the shared pure-plan cache owns no devices.
        self._input_values = {name: getattr(initial, name) for name in self._specs}
        self.current = None

    def refresh(self, *, start, valid_count, page_mapping):
        """Host-side runtime update; no device allocation and no request slices."""
        import ttnn

        state = self.state
        if self._cache_binding != (
            id(state.key),
            id(state.value),
            state.page_offset,
            state.page_count,
            bool(getattr(state, "global_pages", False)),
        ):
            raise ValueError("Cache binding changed; construct and prepare a separate adapter before sealing")
        plan = make_attention_inputs(
            physical_bucket=self.bucket,
            supported_context=self.context,
            local_page_count=self.page_count,
            start=start,
            valid_count=valid_count,
            page_mapping=page_mapping,
            page_offset=self.page_offset,
        )
        # A failed partial refresh must never authorize attention on mixed inputs.
        self.current = None
        for name, (dtype, pack) in self._specs.items():
            value = getattr(plan, name)
            if self._input_values.get(name) == value:
                continue
            self._input_values.pop(name, None)
            host = ttnn.from_torch(
                pack(plan),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.layer.device),
            )
            ttnn.copy_host_to_device_tensor(host, self.inputs[name])
            self._input_values[name] = value
        self.current = plan
        return plan

    def attention(self, normalized_hidden, *, project=True):
        """Device-only prefill attention with fixed-shape runtime position/maps."""
        import ttnn

        if self.current is None:
            raise RuntimeError("refresh must initialize the request before forward/warmup")
        layer, state, n = self.layer, self.state, self.bucket
        if tuple(normalized_hidden.shape) != (1, 1, n, layer.hidden_width):
            raise ValueError("Caller must supply the fixed physical hidden bucket")
        packed = layer.linear(normalized_hidden, "packed_input")
        qg, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(packed, (1, n, 2560)), num_heads=8, num_kv_heads=1, transpose_key=False
        )
        q = layer.norm(qg[:, :4, :, :], "self_attn.q_norm")
        k = layer.norm(k, "self_attn.k_norm")
        gate = ttnn.reshape(ttnn.permute(qg[:, 4:8, :, :], (0, 2, 1, 3)), (1, 1, n, 1024))
        rope = ttnn.reshape(
            ttnn.embedding(self.inputs["rope_indices"], layer.rope_decode, layout=ttnn.TILE_LAYOUT), (1, 1, n, 128)
        )
        q, k = (
            layer.rope(q, rope[:, :, :, :64], rope[:, :, :, 64:]),
            layer.rope(k, rope[:, :, :, :64], rope[:, :, :, 64:]),
        )
        # Cast to each actual cache dtype, preserving any paged-cache dtype split.
        ttnn.experimental.paged_fill_cache(state.key, ttnn.typecast(k, state.key.dtype), self.inputs["fill_pages"])
        ttnn.experimental.paged_fill_cache(state.value, ttnn.typecast(v, state.value.dtype), self.inputs["fill_pages"])
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            state.key,
            state.value,
            self.inputs["attention_pages"],
            chunk_start_idx_tensor=self.inputs["start"],
            scale=1 / 16,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=128
            ),
            compute_kernel_config=layer.compute,
        )
        out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, 1, n, 1024))
        out = ttnn.multiply(gate, out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return layer.reduce(layer.linear(out, "self_attn.o_proj")) if project else out

    def forward(self, hidden):
        """One complete layer, retaining physical output height including dummy rows."""
        import ttnn

        normalized = self.layer.norm(hidden, "input_layernorm")
        mixed = self.attention(normalized)
        del normalized
        return ttnn.to_memory_config(self.layer.finish(hidden, mixed), ttnn.DRAM_MEMORY_CONFIG)
