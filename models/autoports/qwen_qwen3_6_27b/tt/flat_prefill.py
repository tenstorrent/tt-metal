# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP4 prefill with compact rows between the existing temporal boundaries.

Requires explicit TP4 _qkv(logical_shape), _gdn(packed, flat_output), and
_collective(placement_seq) hooks; see flat_prefill_candidate.md. No policy or
method is mutated during forward. The reference optimized decoder is unchanged.
"""

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import RecurrentState


def _can_flatten(layer, batch, seq):
    # Retain the original dispatch for single-token tails and uncommon policy
    # overrides whose decisions depend on physical batch rather than row count.
    return (
        batch > 1
        and seq > 1
        and layer.hidden_width == 1280
        and layer.policy.get("flat_norm", False)
        and layer.policy.get("flat_ccl", False)
        and (
            layer.policy.get("gather_norm", False) in (False, "decode", "single", "decode_or_single")
            or layer.policy.get("gather_norm") is True
            and layer.policy.get("flat_gather_prefill", False)
        )
        and not layer.policy.get("chunk_add_prefill_correction", False)
        # L1 MLP dispatch only changes the one-tile-row path. Larger flat
        # prefill uses the original DRAM matmuls and can retain compaction.
        and (not layer.policy.get("l1_mlp", False) or batch * seq > 32)
    )


def _reduce_flat(layer, x, seq):
    if x.shape[-1] == 1280:
        return x
    # The old reduction chose placement before packing B separate users. Norm
    # collectives already used compact rows and must not receive this override.
    return layer._collective(x, reduce=True, placement_seq=seq)


def _finish_flat(layer, x, attention, batch, seq):
    x = layer._residual_add(x, _reduce_flat(layer, attention, seq), "residual_after_attention")
    h = layer._norm(x, "post_attention_layernorm.weight", logical_batch=batch)
    layer._observe_precision("normalized_mlp_input", h, "activation_dtype")
    if layer.swiglu_weight is not None:
        if batch * seq > 512 and layer.policy.get("prefill_swiglu_options"):
            act = layer._prefill_swiglu_with_options(h)
        else:
            rows = batch * seq
            small = 32 < rows <= 512
            act = ttnn.experimental.minimal_matmul(
                layer._mlp_input(ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)),
                layer.swiglu_weight,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=2 if small else 4,
                    K_block_size=8 if small else 4,
                    N_block_size=26 if small else 16,
                    subblock_h=1,
                    subblock_w=2 if small else 4,
                    compute_with_storage_grid_size=(11, 8) if small else (11, 10),
                ),
                fuse_swiglu=True,
                compute_kernel_config=layer.matmul_configs.get("mlp_gate_up", layer.matmul_configs["mlp"]),
                dtype=getattr(ttnn, layer.policy.get("activation_dtype", "bfloat16")),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    else:
        if layer.policy.get("packed_mlp", False):
            packed = layer._linear(h, "mlp.gate_up.weight")
            gate, up = packed[:, :, :4352], packed[:, :, 4352:]
        else:
            gate = layer._linear(h, "mlp.gate_proj.weight")
            up = layer._linear(h, "mlp.up_proj.weight")
        act = (
            ttnn.multiply(ttnn.silu(gate), up)
            if layer.policy["round_silu"]
            else ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        )
    layer._observe_precision("mlp_activation_output", act, "activation_dtype")
    return layer._residual_add(
        x, _reduce_flat(layer, layer._linear(act, "mlp.down_proj.weight"), seq), "residual_after_mlp"
    )


def _attention_flat(layer, h, state, *, batch, seq, start_pos, rope):
    """Keep the reference cache/SDPA operations on their original user axes."""
    assert start_pos % layer._attention_chunk(state) == 0
    q, k, v, gate = layer._qkv(h, rope, prefill=True, logical_shape=(batch, seq))
    padded = (seq + 31) // 32 * 32
    if seq != padded:
        q, k, v = (ttnn.pad(t, ((0, 0), (0, 0), (0, padded - seq), (0, 0)), value=0.0) for t in (q, k, v))
    k, v = (ttnn.typecast(t, state.key.dtype) if t.dtype != state.key.dtype else t for t in (k, v))
    layer._prefill_cache_fill(state, k, v, start_pos=start_pos)
    out = ttnn.transformer.chunked_scaled_dot_product_attention(
        q,
        state.key,
        state.value,
        layer._attention_page_table(state),
        chunk_start_idx=start_pos,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*layer.policy.get("sdpa_grid", [8, 4])),
            q_chunk_size=32,
            k_chunk_size=layer._attention_chunk(state),
            exp_approx_mode=False,
        ),
        compute_kernel_config=layer.ckc,
    )
    out = ttnn.reshape(ttnn.transformer.concatenate_heads(out[:, :, :seq, :]), (1, batch * seq, 1536))
    out = ttnn.multiply(
        layer._linear(h, "gate") if gate is None else gate,
        out,
        input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
    )
    return layer._linear(out, "self_attn.o_proj.weight")


def _flat_chunk(layer, chunk, state, *, start_pos, rope):
    batch, seq, width = tuple(chunk.shape)
    x = ttnn.reshape(ttnn.to_memory_config(chunk, ttnn.DRAM_MEMORY_CONFIG), (1, batch * seq, width))
    h = layer._norm(x, "input_layernorm.weight", logical_batch=batch)
    if layer.kind == "linear_attention":
        # B>1 historically bypasses correction_l1. Calling _linear on physical
        # batch1 could enable a different combine for R<=32, so preserve the
        # two raw FP32 partials and single BF16 combine explicitly at all sizes.
        key = "packed_gdn"
        if hasattr(layer, "_prefill_packed_gdn"):
            packed = layer._prefill_packed_gdn(h)
        elif key in layer.weight_corrections:
            packed = layer._combine_correction(
                layer._linear_raw(h, key),
                layer._linear_raw(h, layer.weight_corrections[key]),
                key,
            )
        else:
            packed = layer._linear(h, key)
        if layer.policy.get("flat_packed_gdn", False):
            from .flat_gdn_prefill_candidate import flat_packed_gdn

            attention = flat_packed_gdn(layer, packed, state, batch=batch, seq=seq)
        elif hasattr(layer, "_flat_packed_gdn"):
            attention = layer._flat_packed_gdn(packed, state, batch=batch, seq=seq)
        else:
            packed = ttnn.reshape(packed, (batch, seq, packed.shape[-1]))
            attention = layer._gdn(h, state, packed=packed, flat_output=True)
    else:
        attention = _attention_flat(layer, h, state, batch=batch, seq=seq, start_pos=start_pos, rope=rope)
    if layer.kind == "full_attention" and layer.policy.get("norm_projection_l1", True) and h.is_sharded():
        ttnn.deallocate(h)
    return ttnn.reshape(_finish_flat(layer, x, attention, batch, seq), (batch, seq, width))


def _legacy_chunk(layer, chunk, state, *, start_pos, rope, unaligned):
    """Use the original path without publishing/rounding a live chunk state."""
    h = layer._norm(chunk, "input_layernorm.weight")
    if layer.kind == "linear_attention":
        attention = layer._gdn(h, state)
    elif unaligned:
        position = ttnn.reshape(
            ttnn.moreh_full((1, 32), start_pos, layer.mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)[
                :, : chunk.shape[0]
            ],
            (chunk.shape[0],),
        )
        attention = layer._attention_decode(h, state, position, rope)
    else:
        attention = layer._attention_prefill(h, state, start_pos=start_pos, rope=rope)
    if layer.kind == "full_attention" and layer.policy.get("norm_projection_l1", True) and h.is_sharded():
        ttnn.deallocate(h)
    return layer._finish(chunk, attention, prefill=True)


def flat_prefill_forward(layer, x, *, state, start_pos=0, rope=None):
    """Preserve the inherited scheduler and state publication around flat chunks."""
    if isinstance(x, (list, tuple)):
        starts = [start_pos] * len(x) if isinstance(start_pos, int) else start_pos
        rotations = [None] * len(x) if rope is None else rope
        if not x or not (len(x) == len(state) == len(starts) == len(rotations)):
            raise ValueError("Each prefill row needs its own state, start and rotations")
        return [
            flat_prefill_forward(layer, t, state=s, start_pos=p, rope=r)
            for t, s, p, r in zip(x, state, starts, rotations)
        ]
    length = x.shape[1]
    if not (
        0 <= start_pos < layer.config.max_position_embeddings
        and 1 <= length <= layer.config.max_position_embeddings - start_pos
    ):
        raise ValueError("Prefill is outside the HF context contract")
    published_state = state
    if layer.kind == "linear_attention" and state.recurrent.dtype == ttnn.bfloat16:
        state = RecurrentState(state.conv, ttnn.typecast(state.recurrent, ttnn.float32))
    outputs = []
    offset = 0
    alignment = layer._attention_chunk(state) if layer.kind == "full_attention" else 1
    while offset < length:
        absolute = start_pos + offset
        n = 1 if absolute % alignment else min(layer.policy.get("prefill_chunk", 512), length - offset)
        chunk = x[:, offset : offset + n]
        rotations = tuple(t[:, :, offset : offset + n, :] for t in rope) if layer.kind == "full_attention" else None
        if _can_flatten(layer, x.shape[0], n):
            out = _flat_chunk(layer, chunk, state, start_pos=absolute, rope=rotations)
        else:
            out = _legacy_chunk(
                layer, chunk, state, start_pos=absolute, rope=rotations, unaligned=bool(absolute % alignment)
            )
        outputs.append(out)
        offset += n
    if state is not published_state:
        ttnn.typecast(state.recurrent, published_state.recurrent.dtype, output_tensor=published_state.recurrent)
    else:
        layer._round_recurrent_state(state)
    return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)
