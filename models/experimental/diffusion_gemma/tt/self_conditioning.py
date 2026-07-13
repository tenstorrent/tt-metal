# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device (ttnn) self-conditioning gated MLP — the net-new diffusion weight module (#47461 loader / #47463 runtime).

Device mirror of ``reference/self_conditioning.py`` (the pure-torch oracle) and of
transformers ``DiffusionGemmaSelfConditioning``::

    forward = post_norm(inputs_embeds + down(gelu_tanh(gate(pre_norm(signal))) * up(pre_norm(signal))))

- ``pre_norm``  — scaled RMSNorm (`model.decoder.self_conditioning.pre_norm.weight`)
- ``post_norm`` — **scaleless** RMSNorm (no checkpoint weight; absent by design)
- gate/up/down  — bias-free linears, gemma4 GeGLU pattern (`tt/shared_mlp.py`)

RMSNorm uses the weight **directly** (NOT the Gemma2/3 ``1+weight`` convention) —
matches both `ttnn.rms_norm` and the reference, so weights load verbatim.

The module is small (2816→2112→2816) and is kept replicated for the current QB2
integration path. Weights come from a ``weight_mapping.remap_state_dict``
self-conditioning sub-dict (short keys ``{pre_norm,gate_proj,up_proj,down_proj}.weight``).
Validated on QB2 vs the reference oracle by ``tests/test_device_self_conditioning.py`` and
as part of the mesh denoise logits wrapper in ``tests/test_device_bidirectional_attention_integration.py``.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import ttnn

from models.experimental.diffusion_gemma.weight_mapping import expected_self_conditioning_shapes


class ChunkedEmbeddingWeight(NamedTuple):
    chunks: tuple
    shape: tuple
    chunk_size: int


def self_conditioning_embedding_prechunk_enabled() -> bool:
    return os.getenv("DG_SELFCOND_PRECHUNK_EMBED", "1") != "0"


def self_conditioning_logits_l1_mode() -> str:
    mode = os.getenv("DG_SELFCOND_LOGITS_L1", "chain").lower()
    if mode not in {"off", "chain"}:
        raise ValueError("DG_SELFCOND_LOGITS_L1 must be one of: off, chain")
    return mode


def _config_value(config, name: str):
    if isinstance(config, dict):
        return config[name]
    return getattr(config, name)


def validate_self_conditioning_state(state_dict, *, hidden_size: int, intermediate_size: int) -> None:
    """Validate remapped self-conditioning weights before moving them to device."""
    expected = expected_self_conditioning_shapes(hidden_size, intermediate_size)
    missing = sorted(set(expected) - set(state_dict))
    if missing:
        raise ValueError(f"missing self-conditioning weights: {missing}")
    for key, shape in expected.items():
        if tuple(state_dict[key].shape) != shape:
            raise ValueError(f"{key} has shape {tuple(state_dict[key].shape)}, expected {shape}")


def build_self_conditioning(
    device,
    state_dict,
    *,
    config=None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    eps: float | None = None,
    dtype=ttnn.bfloat16,
    module_cls=None,
):
    """Build ``TtSelfConditioning`` from remapped checkpoint weights and config."""
    if config is not None:
        hidden_size = hidden_size if hidden_size is not None else _config_value(config, "hidden_size")
        intermediate_size = (
            intermediate_size if intermediate_size is not None else _config_value(config, "intermediate_size")
        )
        eps = eps if eps is not None else _config_value(config, "rms_norm_eps")
    if hidden_size is None or intermediate_size is None:
        raise ValueError("hidden_size and intermediate_size are required")
    eps = 1e-6 if eps is None else eps
    validate_self_conditioning_state(state_dict, hidden_size=hidden_size, intermediate_size=intermediate_size)
    cls = TtSelfConditioning if module_cls is None else module_cls
    return cls(
        device,
        state_dict,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        eps=eps,
        dtype=dtype,
    )


def build_self_conditioning_embedding_weight(
    device,
    embedding_weight,
    *,
    hidden_size: int | None = None,
    dtype=ttnn.bfloat16,
    tensor_fn=ttnn.as_tensor,
):
    """Move tied token embedding weights to the self-conditioning matmul layout."""
    if len(embedding_weight.shape) != 2:
        raise ValueError("embedding_weight must have shape [vocab, hidden]")
    if hidden_size is not None and embedding_weight.shape[-1] != hidden_size:
        raise ValueError(f"embedding hidden size {embedding_weight.shape[-1]} does not match expected {hidden_size}")
    tensor_kwargs = {
        "device": device,
        "dtype": dtype,
        "layout": ttnn.TILE_LAYOUT,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    if self_conditioning_embedding_prechunk_enabled():
        chunk_size = 8192
        chunks = tuple(
            tensor_fn(
                embedding_weight[start : min(start + chunk_size, embedding_weight.shape[0])]
                .unsqueeze(0)
                .unsqueeze(0)
                .contiguous(),
                **tensor_kwargs,
            )
            for start in range(0, embedding_weight.shape[0], chunk_size)
        )
        return ChunkedEmbeddingWeight(
            chunks=chunks,
            shape=(1, 1, embedding_weight.shape[0], embedding_weight.shape[1]),
            chunk_size=chunk_size,
        )
    return tensor_fn(embedding_weight.unsqueeze(0).unsqueeze(0), **tensor_kwargs)


def build_self_conditioning_embedding_weight_vocab_sharded(
    device,
    embedding_weight,
    mesh_config,
    *,
    hidden_size: int | None = None,
    dtype=ttnn.bfloat16,
    tensor_fn=ttnn.from_torch,
):
    """Tied token embedding table ROW-sharded on vocab for the sharded soft-embedding (E7).

    ``row_parallel`` shards the vocab (dim -2) contiguously in device-column order, so device ``c``
    holds rows ``[c*per_dev, (c+1)*per_dev)`` — the SAME vocab block the column-parallel lm_head
    scores on device ``c`` (both use ``ShardTensor2dMesh`` with the tp mesh axis, identical device
    ordering). This aligns each device's embedding rows with its logit shard, so
    :meth:`TtSelfConditioning.soft_embedding_sharded` can contract the sharded vocab dim locally and
    combine with one ``all_reduce``. It also SAVES ~1 GB/chip versus the replicated table build.
    Build ONCE, OUTSIDE any trace.
    """
    if len(embedding_weight.shape) != 2:
        raise ValueError("embedding_weight must have shape [vocab, hidden]")
    if hidden_size is not None and embedding_weight.shape[-1] != hidden_size:
        raise ValueError(f"embedding hidden size {embedding_weight.shape[-1]} does not match expected {hidden_size}")
    full = embedding_weight.unsqueeze(0).unsqueeze(0).contiguous()  # [1,1,vocab,hidden]
    return tensor_fn(
        full,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.row_parallel(device),
    )


def _dram_for_rms_norm(tensor):
    memory_config = tensor.memory_config()
    if memory_config.buffer_type == ttnn.BufferType.DRAM and not memory_config.is_sharded():
        return tensor
    return ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)


def _norm_shard_core_count(hidden_size: int) -> int:
    tile_size = getattr(ttnn, "TILE_SIZE", 32)
    tile_cols = hidden_size // tile_size
    for cores in (8, 4, 2):
        if tile_cols % cores == 0:
            return cores
    return 1


def _norm_subblock_w(block_w: int) -> int:
    for subblock_w in range(4, 0, -1):
        if block_w % subblock_w == 0:
            return subblock_w
    return 1


def _width_sharded_rms_norm(chunk, *, weight=None, epsilon: float):
    hidden_size = chunk.shape[-1]
    tile_size = getattr(ttnn, "TILE_SIZE", 32)
    if hidden_size % tile_size != 0 or chunk.shape[-2] != tile_size:
        kwargs = {"epsilon": epsilon, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
        if weight is not None:
            kwargs["weight"] = weight
        return ttnn.rms_norm(chunk, **kwargs)
    tile_cols = hidden_size // tile_size
    cores = _norm_shard_core_count(hidden_size)
    if cores == 1:
        kwargs = {"epsilon": epsilon, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
        if weight is not None:
            kwargs["weight"] = weight
        return ttnn.rms_norm(chunk, **kwargs)

    grid = ttnn.CoreGrid(x=cores, y=1)
    sharded_mem = ttnn.create_sharded_memory_config(
        (tile_size, hidden_size),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(cores, 1),
        subblock_w=_norm_subblock_w(tile_cols // cores),
        block_h=1,
        block_w=tile_cols // cores,
        inplace=False,
    )

    chunk_sharded = ttnn.to_memory_config(chunk, sharded_mem)
    weight_sharded = None
    if weight is not None:
        weight_sharded = ttnn.to_memory_config(weight, sharded_mem)
    out_sharded = ttnn.rms_norm(
        chunk_sharded,
        weight=weight_sharded,
        epsilon=epsilon,
        program_config=program_config,
        memory_config=sharded_mem,
    )
    out = ttnn.sharded_to_interleaved(out_sharded, ttnn.DRAM_MEMORY_CONFIG)
    out_sharded.deallocate(True)
    chunk_sharded.deallocate(True)
    if weight_sharded is not None and weight_sharded is not weight:
        weight_sharded.deallocate(True)
    return out


def _rms_norm_dram(tensor, *, weight=None, epsilon: float, chunk_size: int = 32):
    norm_input = _dram_for_rms_norm(tensor)
    seq_len = norm_input.shape[-2]
    if seq_len <= chunk_size:
        out = _width_sharded_rms_norm(norm_input, weight=weight, epsilon=epsilon)
        if norm_input is not tensor:
            norm_input.deallocate(True)
        return out

    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = ttnn.slice(
            norm_input,
            [0, 0, start, 0],
            [norm_input.shape[0], norm_input.shape[1], end, norm_input.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        chunks.append(_width_sharded_rms_norm(chunk, weight=weight, epsilon=epsilon))
        chunk.deallocate(True)
    out = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for chunk in chunks:
        chunk.deallocate(True)
    if norm_input is not tensor:
        norm_input.deallocate(True)
    return out


class TtSelfConditioning:
    def __init__(
        self,
        device,
        state_dict,
        *,
        hidden_size,
        intermediate_size,
        eps=1e-6,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.eps = eps
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # scaled pre_norm weight for self-conditioning RMSNorm.
        pre_w = state_dict["pre_norm.weight"].reshape((1, 1, 1, hidden_size))
        self.pre_norm_weight = ttnn.as_tensor(
            pre_w,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # post_norm is scaleless — no checkpoint weight, no tensor built.

        # gate/up/down linears: HF [out,in] -> [1,1,in,out], TILE, DRAM.
        def _lin(key):
            w = state_dict[key].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                w,
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.gate_proj = _lin("gate_proj.weight")
        self.up_proj = _lin("up_proj.weight")
        self.down_proj = _lin("down_proj.weight")

    def forward(self, inputs_embeds_tt, signal_tt):
        """``inputs_embeds_tt`` / ``signal_tt``: ``[1,1,L,hidden]`` TILE_LAYOUT.

        Zero signal -> ``post_norm(inputs_embeds)`` (NOT inputs_embeds), matching the
        decoder: it always post-normalizes its input embeddings.
        """
        normed = _rms_norm_dram(signal_tt, weight=self.pre_norm_weight, epsilon=self.eps)

        gate = ttnn.linear(normed, self.gate_proj)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True)  # gelu_pytorch_tanh
        up = ttnn.linear(normed, self.up_proj)
        normed.deallocate(True)

        hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        sc = ttnn.linear(hidden, self.down_proj)
        hidden.deallocate(True)

        summed = ttnn.add(inputs_embeds_tt, sc)
        sc.deallocate(True)
        out = _rms_norm_dram(summed, epsilon=self.eps)  # scaleless post_norm
        summed.deallocate(True)
        return out

    def soft_embedding(self, prev_logits_tt, embedding_weight_tt, *, compute_kernel_config=None):
        """Probability-weighted token embedding from prev-step logits — the decoder's
        soft-embedding step (modeling: ``softmax(logits, dim=-1) @ embed_tokens.weight``).

        ``prev_logits_tt`` ``[1,1,L,vocab]`` (TILE), ``embedding_weight_tt`` the tied
        table ``[1,1,vocab,hidden]`` (TILE). Returns the signal ``[1,1,L,hidden]``.
        ``compute_kernel_config`` applies to the moderate-vocabulary full-softmax
        branch. The production 262144-vocabulary path uses the ordered online
        chunk reduction below and retains its established BF16 arithmetic; it
        does not forward that full-softmax kernel configuration.
        """
        vocab_size = prev_logits_tt.shape[-1]
        vocab_chunk_size = 8192
        if isinstance(embedding_weight_tt, ChunkedEmbeddingWeight) or vocab_size > vocab_chunk_size:
            return self._soft_embedding_chunked(
                prev_logits_tt,
                embedding_weight_tt,
                vocab_chunk_size=vocab_chunk_size,
            )
        if compute_kernel_config is not None:
            probs = ttnn.softmax(
                prev_logits_tt, dim=-1, numeric_stable=True, compute_kernel_config=compute_kernel_config
            )
        else:
            probs = ttnn.softmax(prev_logits_tt, dim=-1)
        signal = ttnn.matmul(probs, embedding_weight_tt)  # [1,1,L,vocab] @ [1,1,vocab,hidden] -> [1,1,L,hidden]
        probs.deallocate(True)
        # canonical: * embed_scale = hidden_size**0.5 (the tied embedding's scale). The pre_norm eps
        # floor does NOT absorb this at the tiny soft-RMS of a 262k-vocab softmax, so it is load-bearing.
        scaled = ttnn.multiply(signal, float(self.hidden_size) ** 0.5)
        signal.deallocate(True)
        return scaled

    def _soft_embedding_chunked(self, prev_logits_tt, embedding_weight_tt, *, vocab_chunk_size: int):
        """Streaming ``softmax(prev_logits) @ embedding`` over vocab chunks.

        This avoids materializing a production-vocab probability tensor whose
        softmax program has a large static circular-buffer footprint.
        """
        logits_max = ttnn.max(prev_logits_tt, dim=-1, keepdim=True)
        numerator = None
        denominator = None
        vocab_size = prev_logits_tt.shape[-1]
        embedding_chunks = None
        if isinstance(embedding_weight_tt, ChunkedEmbeddingWeight):
            embedding_chunks = embedding_weight_tt.chunks
            vocab_chunk_size = embedding_weight_tt.chunk_size
        hidden_size = embedding_weight_tt.shape[-1]
        logits_l1_mode = self_conditioning_logits_l1_mode()
        for start in range(0, vocab_size, vocab_chunk_size):
            end = min(start + vocab_chunk_size, vocab_size)
            logits_chunk = ttnn.slice(
                prev_logits_tt,
                [0, 0, 0, start],
                [prev_logits_tt.shape[0], prev_logits_tt.shape[1], prev_logits_tt.shape[2], end],
                memory_config=ttnn.DRAM_MEMORY_CONFIG if logits_l1_mode == "off" else ttnn.L1_MEMORY_CONFIG,
            )
            if logits_l1_mode == "chain":
                shifted = ttnn.subtract(logits_chunk, logits_max, memory_config=ttnn.L1_MEMORY_CONFIG)
            else:
                shifted = ttnn.subtract(logits_chunk, logits_max)
            logits_chunk.deallocate(True)
            if logits_l1_mode == "chain":
                exp_chunk = ttnn.exp(shifted, memory_config=ttnn.L1_MEMORY_CONFIG)
            else:
                exp_chunk = ttnn.exp(shifted)
            shifted.deallocate(True)
            # TTNN reductions and binary ops inherit the first input's memory
            # config. In chain mode this intentionally carries the denominator
            # reduction/accumulator through L1 without changing operation order;
            # the DRAM matmul keeps the numerator and final divide in DRAM.
            denom_chunk = ttnn.sum(exp_chunk, dim=-1, keepdim=True)
            if embedding_chunks is None:
                embed_chunk = ttnn.slice(
                    embedding_weight_tt,
                    [0, 0, start, 0],
                    [embedding_weight_tt.shape[0], embedding_weight_tt.shape[1], end, hidden_size],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                embed_chunk = embedding_chunks[start // vocab_chunk_size]
            numer_chunk = ttnn.matmul(exp_chunk, embed_chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            exp_chunk.deallocate(True)
            if embedding_chunks is None:
                embed_chunk.deallocate(True)
            if numerator is None:
                numerator = numer_chunk
                denominator = denom_chunk
            else:
                next_numerator = ttnn.add(numerator, numer_chunk)
                numerator.deallocate(True)
                numer_chunk.deallocate(True)
                numerator = next_numerator
                next_denominator = ttnn.add(denominator, denom_chunk)
                denominator.deallocate(True)
                denom_chunk.deallocate(True)
                denominator = next_denominator
        logits_max.deallocate(True)
        signal = ttnn.div(numerator, denominator)
        numerator.deallocate(True)
        denominator.deallocate(True)
        scaled = ttnn.multiply(signal, float(self.hidden_size) ** 0.5)
        signal.deallocate(True)
        return scaled

    def soft_embedding_sharded(
        self,
        prev_logits_shard,
        embedding_weight_sharded,
        *,
        mesh_config,
        ccl_manager,
        global_max=None,
    ):
        """Probability-weighted token embedding from TP-sharded prev-step logits (E7).

        ``prev_logits_shard`` is the per-device vocab shard ``[1,1,S,vocab/TP]`` (the lm_head output
        with ``return_sharded=True``); ``embedding_weight_sharded`` is the vocab-ROW-sharded tied
        table ``[1,1,vocab/TP,hidden]`` from
        :func:`build_self_conditioning_embedding_weight_vocab_sharded` (device ``c`` rows aligned to
        the logit shard). Computes ``softmax(prev_logits) @ embed`` distributed over TP:

          numerator_c = Σ_v exp(z_v - M)·embed_v   (a matmul contracting the sharded vocab dim -> a
                        per-device PARTIAL [1,1,S,hidden], the standard row-parallel pattern)
          denominator_c = Σ_v exp(z_v - M)          (per-device partial [1,1,S,1])

        with the shared EXACT global max ``M`` (bit-identical to the replicated path). The partials
        are accumulated/all-reduced in fp32, then ``signal = (numer/denom)·sqrt(hidden)`` is
        REPLICATED on every device — a drop-in for the replicated :meth:`soft_embedding` signal.

        NOT bit-identical: only the vocab sum is re-associated as TP partials (same class as
        :func:`token_entropy_sharded`); fp32 partials + fp32 all-reduce keep it near-exact, and the
        signal passes through RMSNorm so its decision impact is softer than entropy's. The load-
        bearing ``sqrt(hidden)`` embed scale (self_conditioning.py) is preserved. Decision-gated.
        """
        from models.demos.gemma4.tt.ccl import ccl_allreduce
        from models.experimental.diffusion_gemma.tt.sampling import global_vocab_max

        owns_max = global_max is None
        max_t = (
            global_vocab_max(prev_logits_shard, mesh_config=mesh_config, ccl_manager=ccl_manager)
            if owns_max
            else global_max
        )
        shifted = ttnn.subtract(prev_logits_shard, max_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if owns_max:
            max_t.deallocate(True)
        exp_shard = ttnn.exp(shifted, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shifted.deallocate(True)
        # Contract the sharded vocab dim -> per-device PARTIAL numerator/denominator (row-parallel).
        denom_partial = ttnn.sum(exp_shard, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        numer_partial = ttnn.matmul(exp_shard, embedding_weight_sharded, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        exp_shard.deallocate(True)
        # fp32 combine: all-reduce(SUM) the partials across TP (deallocates its inputs).
        numer_f = ttnn.typecast(numer_partial, ttnn.float32)
        numer_partial.deallocate(True)
        denom_f = ttnn.typecast(denom_partial, ttnn.float32)
        denom_partial.deallocate(True)
        numerator = ccl_allreduce(numer_f, mesh_config, ccl_manager)
        denominator = ccl_allreduce(denom_f, mesh_config, ccl_manager)
        signal = ttnn.div(numerator, denominator)
        numerator.deallocate(True)
        denominator.deallocate(True)
        scaled = ttnn.multiply(signal, float(self.hidden_size) ** 0.5)
        signal.deallocate(True)
        if scaled.get_dtype() != ttnn.bfloat16:
            out = ttnn.typecast(scaled, ttnn.bfloat16)
            scaled.deallocate(True)
            return out
        return scaled

    def condition(self, inputs_embeds_tt, prev_logits_tt, embedding_weight_tt, *, compute_kernel_config=None):
        """Full self-conditioning step: soft-embed prev logits, then apply the module
        (mirrors the reference ``SelfConditioning.condition`` / decoder forward).

        ``prev_logits_tt is None`` (first step / encoder pass) -> zero signal, so the
        result is ``post_norm(inputs_embeds)``.
        """
        if prev_logits_tt is None:
            return _rms_norm_dram(inputs_embeds_tt, epsilon=self.eps)
        signal = self.soft_embedding(prev_logits_tt, embedding_weight_tt, compute_kernel_config=compute_kernel_config)
        out = self.forward(inputs_embeds_tt, signal)
        signal.deallocate(True)
        return out
