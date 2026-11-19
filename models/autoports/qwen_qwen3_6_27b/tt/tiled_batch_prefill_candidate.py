# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in physical-batch prefill; importing this module is CPU-only.

The native fused-batch matmul consumes B*ceil32(S) physical rows without
compacting logical tokens. Temporal scheduling and cache/state updates stay
with the inherited decoder. This is an unmeasured candidate, not a default.
"""

import math


def row_accounting(batch, seq):
    if min(batch, seq) < 1:
        raise ValueError("Batch and sequence must be positive")
    compact = math.ceil(batch * seq / 32) * 32
    physical = batch * math.ceil(seq / 32) * 32
    return dict(
        batch=batch,
        seq=seq,
        logical_rows=batch * seq,
        compact_padded_rows=compact,
        tiled_batch_rows=physical,
        row_work_and_payload_ratio=physical / compact,
    )


def matmul_plan(physical_rows, width, physical_n, policy, options=None):
    """Mirror inherited 2D geometry, using padded physical M, never logical B*S."""
    options = options or {}
    unknown = set(options) - {"grid", "in0_block_w", "subblock_h", "subblock_w", "input_l1", "out_block_h", "row_cap"}
    if unknown:
        raise ValueError(f"Unknown prefill options: {sorted(unknown)}")
    gx, gy = options.get("grid", policy.get("prefill_grid", (8, 8)))
    if min(physical_rows, width, physical_n, gx, gy) < 1 or any(v % 32 for v in (physical_rows, width, physical_n)):
        raise ValueError("Physical M/K/N must be positive tile multiples")
    pm, pn = math.ceil(physical_rows / 32 / gy), math.ceil(physical_n / 32 / gx)
    sw = options.get("subblock_w", max(v for v in (1, 2, 4) if pn % v == 0))
    # Physical batch padding can nearly double M at S33. Using all per-core
    # M as one block then reserves an equally enlarged FP32 partials buffer.
    # Keep row ownership unchanged and loop over bounded height blocks instead.
    requested_bh = options.get("out_block_h", policy.get("tiled_prefill_out_block_h", 1))
    if requested_bh < 1 or sw not in (1, 2, 4) or pn % sw:
        raise ValueError("Invalid output block geometry")
    bh = max(v for v in range(1, pm + 1) if v <= requested_bh and pm % v == 0)
    requested_h = min(bh, options.get("subblock_h", 4 if policy.get("prefill_subblock_h", False) else 1))
    valid_h = [v for v in (1, 2, 4) if v <= requested_h and bh % v == 0 and v * sw <= 4]
    kb = options.get("in0_block_w", 2 if physical_n > 20000 else policy.get("prefill_k", 4))
    if not valid_h or kb < 1 or (width // 32) % kb:
        raise ValueError("Invalid K block or subblock geometry")
    return dict(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=max(valid_h),
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=pn,
        out_block_h=bh,
        out_block_w=pn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def matmul_buffer_bytes(
    plan, *, input_tile_bytes=2048, weight_tile_bytes=1088, output_tile_bytes=2048, partial_tile_bytes=4096
):
    """Major native 2D CB payloads, for multi-K-block interleaved DRAM IO.

    From the native 2D factory: two input buffers, one output and one partials
    block; output/partials alias when their formats match. Excludes firmware,
    descriptors, semaphores and any live allocator buffers; not an L1-fit proof.
    """
    height, width, reduction = plan["out_block_h"], plan["out_block_w"], plan["in0_block_w"]
    result = {
        "input": 2 * height * reduction * input_tile_bytes,
        "weight": 2 * width * reduction * weight_tile_bytes,
        "output": height * width * output_tile_bytes,
        "partials": 0 if output_tile_bytes == partial_tile_bytes else height * width * partial_tile_bytes,
    }
    return {**result, "total": sum(result.values())}


def make_decoder(base):
    """Create a real subclass: cloned resource owners use their own tensors."""
    import ttnn
    from .optimized_decoder import OptimizedDecoder
    from .packed_correction_candidate import KEY

    class TiledBatchPrefillDecoder(base):
        @classmethod
        def from_state_dict(cls, state_dict, *, policy=None, **kwargs):
            policy = dict(policy or {})
            cap = policy.get("tiled_prefill_row_cap", 4096)
            if not isinstance(cap, int) or cap < 32 or cap % 32:
                raise ValueError("tiled_prefill_row_cap must be a positive multiple of 32")
            block_height = policy.get("tiled_prefill_out_block_h", 1)
            if not isinstance(block_height, int) or block_height < 1:
                raise ValueError("tiled_prefill_out_block_h must be a positive integer")
            policy.update(
                flat_prefill=False,
                flat_finish=False,
                flat_norm=False,
                flat_ccl=False,
                minimal_swiglu=False,
                prefill_minimal=False,
                flat_prefill_correction=False,
                chunk_add_prefill_correction=False,
                tiled_prefill_row_cap=cap,
                tiled_prefill_out_block_h=block_height,
                tiled_batch_prefill_candidate={
                    "scope": "B>1,S>1 dense projections; inherited temporal scheduler",
                    "mmrs": "bypassed for batched prefill only",
                    "mlp": "existing contiguous packed weights and kind-specific epilogue",
                    "default_output_block_height": block_height,
                    "status": "experimental, hardware validation required",
                },
            )
            return super().from_state_dict(state_dict, policy=policy, **kwargs)

        def _linear(self, x, key):
            if x.shape[0] <= 1 or x.shape[1] <= 1:
                return super()._linear(x, key)
            if key == "packed_gdn" and self._packed_correction_candidate is not None:
                candidate = self._packed_correction_candidate
                users = max(1, self.policy["tiled_prefill_row_cap"] // x.padded_shape[-2])
                if x.shape[0] > users:
                    # Combine each FP32 joint piece before concatenation, so
                    # the complete batch does not acquire an FP32 joint copy.
                    return ttnn.concat(
                        [
                            candidate._finish_projection(
                                self, self._tiled_linear(x[start : min(start + users, x.shape[0]), :, :], KEY)
                            )
                            for start in range(0, x.shape[0], users)
                        ],
                        dim=0,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                return candidate._finish_projection(self, self._tiled_linear(x, KEY))
            # Deliberately bypass only the multichip MMRS/flatten branch;
            # inherited correction dispatch still calls this owner's raw op.
            return OptimizedDecoder._linear(self, x, key)

        def _linear_raw(self, x, key):
            if x.shape[0] <= 1 or x.shape[1] <= 1:
                return super()._linear_raw(x, key)
            return self._tiled_linear(x, key)

        def _tiled_linear(self, x, key):
            batch, seq, width = tuple(x.shape)
            padded_seq = x.padded_shape[-2]
            # Split only whole users, never flatten/repad their token rows.
            users = max(1, self.policy["tiled_prefill_row_cap"] // padded_seq)
            if batch > users:
                return ttnn.concat(
                    [
                        self._tiled_linear(x[start : min(start + users, batch), :, :], key)
                        for start in range(0, batch, users)
                    ],
                    dim=0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            group = "mlp" if key.startswith("mlp.") else "attention"
            target = getattr(ttnn, self.policy.get(group + "_activation_dtype", "bfloat16"))
            if x.dtype != target:
                x = ttnn.typecast(x, target)
            weight = self.weights[key]
            configs = self.policy.get("prefill_projection_configs", {})
            options = configs.get(key, configs.get(key.removesuffix("_residual"), {}))
            if key == KEY:
                options = {**configs.get("packed_gdn", {}), **self._packed_correction_candidate.prefill_config}
            # Candidate-only overrides must not retune inherited B1 matmuls.
            tiled_configs = self.policy.get("tiled_prefill_projection_configs", {})
            options = {**options, **tiled_configs.get(key, tiled_configs.get(key.removesuffix("_residual"), {}))}
            plan = matmul_plan(batch * padded_seq, width, weight.padded_shape[-1], self.policy, options)
            if options.get("input_l1", self.policy.get("prefill_input_l1", False)):
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            fp32 = key == KEY or key in self.weight_corrections or key.endswith("_residual")
            y = ttnn.linear(
                x,
                weight,
                program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**plan),
                compute_kernel_config=self.matmul_configs[group],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.float32 if fp32 else ttnn.bfloat16,
            )
            # Joint combine reads the uncut physical stride plus column offset.
            if key != KEY and key in self.projection_info:
                logical_n = self.projection_info[key]["logical_n"]
                if y.shape[-1] != logical_n:
                    y = y[..., :logical_n]
            if tuple(y.shape)[:2] != (batch, seq):
                raise AssertionError("Native matmul changed logical batch/sequence")
            return y

    return TiledBatchPrefillDecoder
