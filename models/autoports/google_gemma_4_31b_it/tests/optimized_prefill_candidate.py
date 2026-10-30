"""Phase-specific large-M program and chunk candidates."""

import math

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_lifetime_candidate import LifetimeCandidate


class PrefillCandidate(LifetimeCandidate):
    def _linear(self, x, name):
        if self.decoding or not self.geometry.get("tune_prefill"):
            return super()._linear(x, name)
        # Packing goes through its splitter so split costs remain in the window.
        if (self.pack_attention and name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")) or (
            self.pack_mlp and name in ("mlp.gate_proj", "mlp.up_proj")
        ):
            return super()._linear(x, name)
        weight = self.weights[name + ".weight"]
        k, n = tuple(weight.shape)[-2:]
        m = x.padded_shape[-2]
        group = "attention" if name.startswith("self_attn.") else "down" if name == "mlp.down_proj" else "mlp"
        kernel = self.linear_kernels[group]
        if self.geometry.get("attn_act8" if group == "attention" else "mlp_act8") and x.dtype != ttnn.bfloat8_b:
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        if self.geometry.get("prefill_l1") or self.geometry.get("prefill_l1_carry"):
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        if self.geometry.get("minimal"):
            block = self.geometry.get("minimal_roles", {}).get(name, self.geometry.get("minimal_block", [4, 8, 8]))
            sub = self.geometry.get("minimal_sub", [1, 8])
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=block[0],
                K_block_size=block[1],
                N_block_size=block[2],
                subblock_h=sub[0],
                subblock_w=sub[1],
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            )
            return ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=weight,
                config=cfg,
                compute_kernel_config=kernel,
                dtype=(
                    ttnn.bfloat8_b
                    if self.geometry.get("prefill_mlp_act8")
                    and name in ("mlp.gate_proj", "mlp.up_proj", "mlp.gate_up_proj")
                    else ttnn.bfloat16
                ),
                memory_config=(
                    ttnn.L1_MEMORY_CONFIG
                    if self.geometry.get("prefill_l1_carry") and (name.startswith("mlp.") or name == "self_attn.o_proj")
                    else ttnn.DRAM_MEMORY_CONFIG
                ),
            )
        gy = min(self.geometry.get("prefill_grid_y", 8), max(1, m // 32))
        gx = self.banks
        pn = math.ceil(n / 32 / gx)
        pm = math.ceil(m / 32 / gy)
        oh = max(v for v in range(1, min(pm, self.geometry.get("prefill_out_h", pm)) + 1) if pm % v == 0)
        ow = max(v for v in range(1, min(pn, self.geometry.get("prefill_out_w", pn)) + 1) if pn % v == 0)
        sh = max(v for v in range(1, min(oh, self.geometry.get("prefill_sub_h", 1)) + 1) if oh % v == 0)
        sw = max(v for v in range(1, 8 // sh + 1) if ow % v == 0)
        block = self.geometry.get("prefill_block", 8)
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=block,
            out_subblock_h=sh,
            out_subblock_w=sw,
            out_block_h=oh,
            out_block_w=ow,
            per_core_M=pm,
            per_core_N=pn,
            transpose_mcast=False,
            fuse_batch=False,
            fused_activation=(
                ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
                if self.geometry.get("fused_gate") and name == "mlp.gate_proj"
                else None
            ),
        )
        return ttnn.linear(
            x,
            weight,
            program_config=pc,
            compute_kernel_config=kernel,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def prefill_forward(self, x, *, cos, sin, kv_cache, page_table, chunk_start=0):
        """Prefill arbitrary logical [B,1,S,H] in bounded 1024-token chunks.

        RoPE inputs are [B,1,S,D], already evaluated at absolute positions.
        Explicit continuation starts at a page boundary and retains prefix pages.
        All chunk planning uses shape metadata; tensor work stays on device.
        """
        b, _, length, _ = tuple(x.shape)
        page_size = kv_cache[0].shape[2]
        if length < 1 or chunk_start % page_size:
            raise ValueError("Empty prefill or continuation outside a page boundary")
        if chunk_start + length > self.config.max_position_embeddings:
            raise ValueError("Context exceeds pinned HF contract")
        batch_idx_tensor = self._prefill_indices(b)
        outputs = []
        chunk_size = self.geometry.get("chunk", 1024)
        for offset in range(0, length, chunk_size):
            valid = min(chunk_size, length - offset)
            physical = (valid + 31) // 32 * 32

            def chunk(t):
                physical_input = self._sequence_view(t, t.padded_shape[2])
                return ttnn.slice(physical_input, (0, 0, offset, 0), (b, 1, offset + physical, t.shape[3]))

            first_page = (chunk_start + offset) // page_size
            table = ttnn.slice(page_table, (0, first_page), (b, first_page + (physical + page_size - 1) // page_size))
            outputs.append(
                self.prefill_chunk_forward(
                    chunk(x),
                    cos=chunk(cos),
                    sin=chunk(sin),
                    kv_cache=kv_cache,
                    page_table=page_table,
                    chunk_page_table=table,
                    chunk_start=chunk_start + offset,
                    valid_tokens=valid,
                    batch_idx_tensor=batch_idx_tensor,
                )
            )
        physical_outputs = [self._sequence_view(y, y.padded_shape[2]) for y in outputs]
        combined = physical_outputs[0] if len(physical_outputs) == 1 else ttnn.concat(physical_outputs, dim=2)
        return self._sequence_view(combined, length)
