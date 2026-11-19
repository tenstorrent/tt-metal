# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Universal eager prefill over fixed, device-resident residual workspaces.

The executor preserves the accepted decoder's attention and MoE grouping. Its
persistent storage depends on configured context, never on a logical request.
Only bounded temporary tensors exist during a call; callers must consume logits
and drop them before replaying a trace which may reuse their allocations.

Integration requires MultichipDecoder._full_attention to recognize
positions['fill_pages'] and pass positions['current_pos'] as the flexible SDPA
chunk_start_idx_tensor. No other attention arithmetic is replaced here.
"""

from collections.abc import Callable

import torch

import ttnn

from .prefill_transport import PrefillTransport

DRAM = ttnn.DRAM_MEMORY_CONFIG


def attention_chunks(kind, length, start):
    """The FusedDecoder planner, without any request-keyed device allocation."""
    if length <= 0 or start < 0:
        raise ValueError("Prefill needs a positive length and nonnegative start")
    if kind == "linear_attention":
        for offset in range(0, length, 64):
            yield offset, min(64, length - offset), start + offset
    elif kind == "full_attention":
        offset = 0
        while offset < length:
            absolute, remaining = start + offset, length - offset
            n = min(256, remaining // 32 * 32) if absolute % 256 == 0 and remaining >= 32 else 1
            yield offset, n, absolute
            offset += n
    else:
        raise ValueError(f"Unsupported attention kind: {kind}")


class BoundedPrefill:
    """One serial prefill executor for a configured model/cache allocation.

    Public token IDs stay on the host until copied into a fixed 64-token input.
    Activations, cache contents and logits are never reconstructed on the host.
    `on_logits(offset, valid_rows, tensor)` is the explicit compatibility output
    boundary; it must consume a chunk synchronously and must not replay decode.
    Without that callback only the final valid row receives the LM head.
    """

    TOKEN_CHUNK = 64
    LOGIT_CHUNK = 32

    def __init__(self, model):
        self.model = model
        self.mesh = model.mesh
        self.capacity = ((model.max_seq_len + 31) // 32) * 32
        self.transport = PrefillTransport(self.mesh)
        self.residual = [self._empty(self.capacity) for _ in range(3)]
        self.tokens = self._upload(torch.zeros((1, self.TOKEN_CHUNK), dtype=torch.int32), ttnn.uint32)
        self.position = self._upload(torch.zeros((1,), dtype=torch.int32), ttnn.int32)
        self.fill_pages = self._upload(torch.zeros((1, 8), dtype=torch.int32), ttnn.int32)
        self.rope_indices = {
            n: self._upload(torch.zeros((1, n), dtype=torch.int32), ttnn.uint32) for n in (1, *range(32, 257, 32))
        }
        self.prepared = False
        self._busy = False
        self._prepared_cache_signature = None
        # Exactly three capacity-sized, replicated BF16 workspaces. Include
        # their allocation plus transport/position helpers in model.census().
        self.workspace_bytes_per_chip = 3 * self.capacity * 2048 * 2

    def _upload(self, value, dtype):
        return ttnn.from_torch(
            value,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _empty(self, rows):
        return ttnn.empty(
            (1, rows, 2048),
            dtype=self.model.boundary_dtypes["residual_dtype"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=DRAM,
        )

    @staticmethod
    def _refresh(destination, value, dtype):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(value.contiguous(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT), destination
        )

    def _read(self, source, offset, rows, *, valid_rows=None):
        valid_rows = rows if valid_rows is None else valid_rows
        output = self._empty(rows)
        self.transport.rows(source, output, source_row=offset, count=valid_rows, extent=rows)
        result = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        self.model.observe_boundary(result, "activation_dtype", "bounded_prefill.decoder_input")
        return self.model.observe_boundary(result, "residual_dtype", "bounded_prefill.residual_read")

    def _write(self, source, destination, offset, rows):
        self.model.observe_boundary(source, "residual_dtype", "bounded_prefill.residual_write_source")
        self.model.observe_boundary(destination, "residual_dtype", "bounded_prefill.residual_write_buffer")
        source = ttnn.to_layout(ttnn.to_memory_config(source, DRAM), ttnn.ROW_MAJOR_LAYOUT)
        self.transport.rows(source, destination, destination_row=offset, count=rows)

    def _positions(self, state, absolute, rows):
        self._refresh(self.position, torch.tensor([absolute], dtype=torch.int32), ttnn.int32)
        index = self.rope_indices[rows]
        self._refresh(index, torch.arange(absolute, absolute + rows, dtype=torch.int32)[None, :], ttnn.uint32)
        cos = ttnn.embedding(index, self.model.rope_cos, layout=ttnn.ROW_MAJOR_LAYOUT)
        sin = ttnn.embedding(index, self.model.rope_sin, layout=ttnn.ROW_MAJOR_LAYOUT)
        result = {
            "cos": ttnn.to_layout(ttnn.reshape(cos, (1, rows, 1, 64)), ttnn.TILE_LAYOUT),
            "sin": ttnn.to_layout(ttnn.reshape(sin, (1, rows, 1, 64)), ttnn.TILE_LAYOUT),
            "current_pos": self.position,
        }
        if rows > 1:
            self.transport.page_window(state["page_table"], self.fill_pages, absolute // 32)
            result["fill_pages"] = self.fill_pages
        return result

    @staticmethod
    def _begin_layer(layer, state):
        if layer.kind == "linear_attention":
            # Exactly once per public call, matching MultichipDecoder.prefill_forward.
            ttnn.copy(ttnn.typecast(ttnn.typecast(state["recurrent"], ttnn.bfloat16), ttnn.float32), state["recurrent"])

    def _attention(self, layer, state, source, offset, rows, absolute, public_length):
        x = self._read(source, offset, rows)
        positions = self._positions(state, absolute, rows) if layer.kind == "full_attention" else None
        return layer._attention_residual(
            x,
            state=state,
            decode=rows == 1 and (layer.kind == "full_attention" or public_length == 1),
            positions=positions,
            start_pos=absolute,
            cache_boundary=False,
            projection_k=1024 if public_length > 1 else 512,
        )

    @staticmethod
    def _moe(layer, x):
        # Singleton retains its sparse route. Nonsingleton inputs already have
        # the decoder's next-multiple-of-32 zero padding, moved before the
        # row-independent norm; transport commits only the valid rows.
        return ttnn.add(x, layer._reduce(layer._moe(layer._normalized_input(x, "post_attention_layernorm.weight"))))

    @staticmethod
    def _cache_signature(cache):
        return tuple(
            tuple(
                (name, tuple(t.shape), tuple(t.padded_shape), str(t.dtype), str(t.layout), str(t.memory_config()))
                for name, t in sorted(states[0].items())
                if name in ("k", "v", "page_table", "recurrent", "conv")
            )
            for states in cache
        )

    def prefill(self, token_ids, *, cache, slot=0, start=0, on_logits: Callable | None = None, logit_positions=None):
        """Fill cache for exact logical tokens and return the last-row logits.

        A callback additionally receives every prefill logit in bounded chunks;
        its valid_rows parameter excludes internal final-chunk padding. The
        final returned tensor is a separate single-row terminal result.
        """
        if not self.prepared:
            raise RuntimeError("Prepare every prefill physical variant before capturing decode")
        if self._prepared_cache_signature != self._cache_signature(cache):
            raise ValueError("KV/cache geometry changed after global preparation")
        if self._busy:
            raise RuntimeError("BoundedPrefill is a serial workspace")
        if isinstance(token_ids, torch.Tensor):
            if token_ids.device.type != "cpu":
                raise ValueError("The public prefill boundary accepts CPU token IDs")
            token_ids = token_ids.reshape(-1).to(torch.int32)
        else:
            token_ids = torch.tensor(token_ids, dtype=torch.int32).reshape(-1)
        length = token_ids.numel()
        if not 0 <= start < start + length <= self.model.max_seq_len:
            raise ValueError("Logical prefill exceeds the configured context")
        if logit_positions is not None and any(not 0 <= p < length for p in logit_positions):
            raise ValueError("Requested logit position is outside the logical prefill suffix")
        if len(cache) != len(self.model.layers) or any(not 0 <= slot < len(states) for states in cache):
            raise ValueError("Invalid fixed cache slot")
        if any(
            layer.kind == "full_attention" and start + length > states[slot]["context_length"]
            for layer, states in zip(self.model.layers, cache)
        ):
            raise ValueError("Logical prefill exceeds allocated KV context")
        self._busy = True
        try:
            source, attended, destination = self.residual
            for offset in range(0, length, self.TOKEN_CHUNK):
                n = min(self.TOKEN_CHUNK, length - offset)
                host = torch.zeros((1, self.TOKEN_CHUNK), dtype=torch.int32)
                host[0, :n] = token_ids[offset : offset + n]
                self._refresh(self.tokens, host, ttnn.uint32)
                embedded = ttnn.reshape(self.model.embed(self.tokens), (1, self.TOKEN_CHUNK, 2048))
                self._write(embedded, source, offset, n)
                del embedded
            for layer, states in zip(self.model.layers, cache):
                state = states[slot]
                self._begin_layer(layer, state)
                for offset, n, absolute in attention_chunks(layer.kind, length, start):
                    result = self._attention(layer, state, source, offset, n, absolute, length)
                    self._write(result, attended, offset, n)
                    del result
                for offset in range(0, length, layer.policy.moe_chunk):
                    n = min(layer.policy.moe_chunk, length - offset)
                    physical = 1 if n == 1 else ((n + 31) // 32) * 32
                    x = self._read(attended, offset, physical, valid_rows=n)
                    result = self._moe(layer, x)
                    self._write(result, destination, offset, n)
                    del result, x
                source, destination = destination, source
            if on_logits is not None:
                offsets = (
                    range(0, length, self.LOGIT_CHUNK)
                    if logit_positions is None
                    else sorted({p // self.LOGIT_CHUNK * self.LOGIT_CHUNK for p in logit_positions})
                )
                for offset in offsets:
                    n = min(self.LOGIT_CHUNK, length - offset)
                    hidden = self._read(source, offset, self.LOGIT_CHUNK, valid_rows=n)
                    logits = self.model.terminal(hidden)
                    on_logits(offset, n, logits)
                    del logits, hidden
            return self.model.terminal(self._read(source, length - 1, 1))
        finally:
            self._busy = False

    @staticmethod
    def _layer_key(layer):
        # Layers sharing these selected settings use identical operations; the
        # weights' addresses are runtime bindings. Include all public scalar
        # policy switches, excluding the layer index itself.
        switches = tuple(
            sorted(
                (name, value)
                for name, value in vars(layer).items()
                if isinstance(value, (bool, int, float, str)) and name not in ("layer_idx", "layer_index")
            )
        )

        # Compute config repr includes its Python address. Warm one representative
        # for each actual configuration, rather than every separately allocated
        # but equivalent layer object. Include every exposed kernel field.
        def config_key(value):
            if hasattr(value, "math_fidelity"):
                return type(value).__name__, tuple(
                    (field, str(getattr(value, field)))
                    for field in (
                        "math_fidelity",
                        "math_approx_mode",
                        "fp32_dest_acc_en",
                        "packer_l1_acc",
                        "dst_full_sync_en",
                        "throttle_level",
                    )
                    if hasattr(value, field)
                )
            if isinstance(value, dict):
                return tuple((key, config_key(item)) for key, item in sorted(value.items()))
            if isinstance(value, (tuple, list)):
                return tuple(config_key(item) for item in value)
            return str(value)

        configs = tuple(
            (name, config_key(getattr(layer, name, None)))
            for name in (
                "compute",
                "expert_compute",
                "expert_down_compute",
                "grouped_compute",
                "shared_compute",
                "candidate_compute",
                "projection_computes",
                "fused_projection_compute",
                "shared_scalar_compute",
                "policy",
                "sparse_geometry",
            )
        )
        # Actual storage dtypes and resolved per-layer policy distinguish
        # quantized exceptions even when tensor shapes match the baseline.
        tensors = []

        def collect(prefix, value):
            if isinstance(value, ttnn.Tensor):
                tensors.append(
                    (prefix, tuple(value.shape), str(value.dtype), str(value.layout), str(value.memory_config()))
                )
            elif isinstance(value, dict):
                for key, item in sorted(value.items()):
                    collect(f"{prefix}.{key}", item)
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    collect(f"{prefix}.{index}", item)

        for name in (
            "mesh_weights",
            "mesh_decode_weights",
            "ep_weights",
            "egu",
            "ed",
            "shared_scalar_weight",
            "scan_combined_weight",
            "scan_attention_weight",
            "scan_output_weight",
            "scan_attention_out_weight",
        ):
            collect(name, getattr(layer, name, None))
        return switches, configs, tuple(tensors), config_key(getattr(layer, "runtime_precision", None))

    def prepare(self, *, cache, reset_state: Callable, progress: Callable | None = None):
        """Warm components, never a sweep of whole-model logical requests.

        Must run before any trace capture. reset_state restores all request,
        recurrent/conv/KV and sampler state changed by warmup. Nonsingleton
        MoE tails use the selected next-multiple-of-32 padding throughout the
        row-independent norm/MoE/residual operations; only transport sees the
        logical tail count. No scalar tail crop needs a separate program.
        """
        signature = self._cache_signature(cache)
        if self.prepared:
            if signature != self._prepared_cache_signature:
                raise ValueError("Prepared prefill cannot change cache geometry")
            return
        if len(cache) != len(self.model.layers):
            raise ValueError("One cache entry is required per layer")
        representatives = {}
        for layer, states in zip(self.model.layers, cache):
            if not states:
                raise ValueError("Every layer needs at least one fixed cache slot")
            if layer.policy.moe_chunk != 2048 or layer.residual_layout != "replicated":
                raise ValueError("Bounded prefill requires the selected replicated/MoE2048 policy")
            representatives.setdefault(self._layer_key(layer), (layer, states[0]))
            largest_moe = min(2048, ((self.model.max_seq_len + 31) // 32) * 32)
            for n in (32, 64, 128, 256, 512, 1024, 2048):
                if n <= max(32, 1 << (largest_moe - 1).bit_length()):
                    layer.prepare_moe(n)
        try:
            # Establish initialized transport inputs without allocating a full
            # host activation. This zero block is only a warmup fixture.
            warm_rows = min(2048, self.capacity)
            zeros = self._upload(
                torch.zeros((1, warm_rows, 2048), dtype=torch.bfloat16), self.model.boundary_dtypes["residual_dtype"]
            )
            for buffer in self.residual:
                self.transport.rows(zeros, buffer, count=warm_rows)
            del zeros
            embedded = ttnn.reshape(self.model.embed(self.tokens), (1, self.TOKEN_CHUNK, 2048))
            self._write(embedded, self.residual[0], 0, min(self.capacity, self.TOKEN_CHUNK))
            del embedded
            for ordinal, (layer, state) in enumerate(representatives.values()):
                lengths = range(1, 65) if layer.kind == "linear_attention" else (1, *range(32, 257, 32))
                for n in lengths:
                    if n > self.model.max_seq_len:
                        continue
                    self._begin_layer(layer, state)
                    for source in (self.residual[0], self.residual[2]):
                        result = self._attention(layer, state, source, 0, n, 0, max(2, n))
                        self._write(result, self.residual[1], 0, n)
                        del result
                    if n == 1 and layer.kind == "linear_attention":
                        result = self._attention(layer, state, self.residual[0], 0, 1, 0, 1)
                        self._write(result, self.residual[1], 0, 1)
                        del result
                    if progress is not None and (n == 1 or n % 16 == 0):
                        progress(
                            {"phase": "prefill_attention", "representative": ordinal, "kind": layer.kind, "rows": n}
                        )
                largest_moe = min(2048, ((self.model.max_seq_len + 31) // 32) * 32)
                moe_lengths = (1, *range(32, largest_moe + 1, 32)) if self.model.max_seq_len > 1 else (1,)
                for n in moe_lengths:
                    x = self._read(self.residual[1], 0, n)
                    result = self._moe(layer, x)
                    for destination in (self.residual[0], self.residual[2]):
                        self._write(result, destination, 0, n)
                    del result, x
                    if progress is not None and (n % 32 == 0 or n == 1):
                        progress(
                            {
                                "phase": "prefill_components",
                                "representative": ordinal,
                                "kind": layer.kind,
                                "logical_moe_rows": n,
                            }
                        )
            for source in (self.residual[0], self.residual[2]):
                for n in (1, self.LOGIT_CHUNK):
                    hidden = self._read(source, 0, n, valid_rows=min(n, self.model.max_seq_len))
                    logits = self.model.terminal(hidden)
                    del logits, hidden
            ttnn.synchronize_device(self.mesh)
        finally:
            reset_state()
        self._prepared_cache_signature = signature
        self.prepared = True
