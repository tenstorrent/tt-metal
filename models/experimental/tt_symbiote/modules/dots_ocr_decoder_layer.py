# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNLayerStack, TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.dots_ocr_attention import TTNNDotsOCRAttention
from models.experimental.tt_symbiote.modules.dots_ocr_mlp import TTNNDotsOCRMLP, TTNNDotsOCRMLPColParallelFusedGateUp
from models.experimental.tt_symbiote.modules.linear import (
    _decode_rmsnorm_program_config,
    _decode_width_sharded_input_memory_config,
    _ccl_num_links,
    _ccl_worker_kwargs,
    _tp_requires_ccl,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm


def _mesh_dp_batch_sharded(device, batch_size: int) -> bool:
    if not hasattr(device, "get_num_devices") or int(device.get_num_devices()) <= 1:
        return False
    if not hasattr(device, "shape"):
        return False
    mesh_shape = [int(x) for x in device.shape]
    if len(mesh_shape) != 2:
        return False
    return (mesh_shape[0] == int(batch_size) and mesh_shape[0] > 1) or (
        mesh_shape[1] == int(batch_size) and mesh_shape[0] == 1
    )


def _take_local_dp_batch(hidden_states, device):
    if len(hidden_states.shape) != 3 or int(hidden_states.shape[0]) <= 1:
        return hidden_states
    if not _mesh_dp_batch_sharded(device, int(hidden_states.shape[0])):
        return hidden_states
    return ttnn.slice(
        hidden_states,
        [0, 0, 0],
        [1, int(hidden_states.shape[-2]), int(hidden_states.shape[-1])],
    )


def _match_residual_width(tensor, residual, device):
    if int(tensor.shape[-1]) == int(residual.shape[-1]):
        return tensor
    if not _tp_requires_ccl(device):
        return tensor
    return ttnn.all_gather(
        tensor,
        dim=len(tensor.shape) - 1,
        num_links=_ccl_num_links(device),
        cluster_axis=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        **_ccl_worker_kwargs("all_gather"),
    )


def _gather_tp_hidden_if_needed(tensor, device, hidden_size: int):
    if int(tensor.shape[-1]) == int(hidden_size):
        return tensor
    if not _tp_requires_ccl(device):
        return tensor
    return ttnn.all_gather(
        tensor,
        dim=len(tensor.shape) - 1,
        num_links=_ccl_num_links(device),
        cluster_axis=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        **_ccl_worker_kwargs("all_gather"),
    )


def _col_parallel_rmsnorm_mode() -> str:
    # Default to the known-correct full-hidden sharded RMSNorm path. Local
    # shard-only RMSNorm is intentionally unsupported because it normalizes over
    # H/TP instead of full H and can corrupt long OCR/table decode output.
    mode = os.environ.get("DOTS_OCR_COL_PARALLEL_RMSNORM_MODE", "full_multicore").lower()
    if mode not in {"full_multicore", "full_single", "distributed"}:
        raise ValueError(
            "DOTS_OCR_COL_PARALLEL_RMSNORM_MODE must be one of "
            "{'full_multicore', 'full_single', 'distributed'}; "
            f"got {mode!r}"
        )
    return mode


def _partition_tp_hidden(tensor, device):
    if not _tp_requires_ccl(device):
        return tensor
    return ttnn.mesh_partition(
        tensor,
        dim=len(tensor.shape) - 1,
        cluster_axis=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _full_hidden_rmsnorm_then_maybe_partition(
    norm,
    tensor,
    device,
    hidden_size: int,
    partition_output: bool,
    use_multicore: bool,
):
    full_hidden = _gather_tp_hidden_if_needed(tensor, device, hidden_size)
    normed = (
        norm._forward_decode_sharded(full_hidden, full_hidden.shape)
        if use_multicore
        else TTNNDistributedRMSNorm.forward(norm, full_hidden)
    )
    if full_hidden is not tensor:
        ttnn.deallocate(full_hidden)
    if not partition_output:
        if normed.memory_config().is_sharded():
            interleaved = ttnn.sharded_to_interleaved(normed, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(normed)
            normed = interleaved
        if len(normed.shape) == 4 and int(normed.shape[1]) == 1:
            normed = ttnn.reshape(normed, [normed.shape[0], normed.shape[2], normed.shape[3]])
        return normed
    partitioned = _partition_tp_hidden(normed, device)
    if partitioned is not normed:
        ttnn.deallocate(normed)
    if len(partitioned.shape) == 4 and int(partitioned.shape[1]) == 1:
        partitioned = ttnn.reshape(partitioned, [partitioned.shape[0], partitioned.shape[2], partitioned.shape[3]])
    return partitioned


def _use_bfp8_decoder_weights(layer_idx) -> bool:
    if layer_idx is None:
        return False
    layer_idx = int(layer_idx)
    # Layers 0..6 stay BFP4 for decode speed; later layers are more sensitive
    # for OCR spelling/table tokens.
    return layer_idx >= 7


class TTNNDotsOCRLocalShardRMSNorm(TTNNDistributedRMSNorm):
    def move_weights_to_device_impl(self):
        # Inherit the distributed-RMSNorm weight setup (weight_distributed +
        # compute_kernel_config) from the parent so the interleaved fallback
        # path still works unchanged.
        super().move_weights_to_device_impl()
        # Build a replicated per-device tile-layout weight sized [32, padded_dim]
        # that the sharded multi-core RMSNorm kernel consumes directly. This
        # is the same shape/layout the parent uses for its single-device
        # ``tt_weight_local`` fallback, just promoted to work on every device
        # of the mesh (no mesh_mapper -> replication).
        dim = int(self.torch_layer.weight.shape[0])
        padded_dim = ((dim + 31) // 32) * 32
        weight = self.torch_layer.weight
        if padded_dim != dim:
            weight = torch.nn.functional.pad(weight, (0, padded_dim - dim), value=1.0)
        self.tt_weight_sharded = ttnn.from_torch(
            weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_weight_sharded = ttnn.to_device(self.tt_weight_sharded, self.device)

    def _forward_decode_sharded(self, inp, original_shape):
        # Sharded LayerNorm fast path (decode only, non-TP-CCL). Runs the
        # multi-core sharded RMSNorm kernel by re-sharding the input to the
        # exact ``shard_in_cfg`` the program config expects, then re-sharding
        # the output back to DRAM_MEMORY_CONFIG so downstream QKV/MLP matmuls
        # see the same interleaved input layout as on the unsharded path.
        # That second reshard is what isolates this optimization: nothing
        # outside this method has to change.
        hidden_size = int(self.torch_layer.weight.shape[0])
        eps = getattr(self.torch_layer, "variance_epsilon", getattr(self.torch_layer, "eps", 1e-6))
        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shard_in_cfg = _decode_width_sharded_input_memory_config(hidden_size)
        if inp.memory_config() != shard_in_cfg:
            if inp.is_sharded():
                inp = ttnn.to_memory_config(inp, shard_in_cfg)
            else:
                inp = ttnn.interleaved_to_sharded(inp, shard_in_cfg)
        # HiFi4 + FP32 dest accumulator + packer L1 accumulator. The multi-core
        # sharded RMSNorm combines partial variances across cores; the partial
        # sums sit in the dest register and the L1 packer buffer between
        # cores, so both knobs need maximum precision to keep the cross-core
        # variance combine accurate. Without ``packer_l1_acc=True`` we saw
        # downstream tokenization drift (``Hodgkin`` -> ``Hodgin`` and
        # ``(reference group`` -> ``( (reference group``).
        sharded_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        print("Using RMSNorm")
        print("decode sharded inp.shape:", inp.shape)
        tt_out = ttnn.rms_norm(
            inp,
            epsilon=eps,
            weight=self.tt_weight_sharded,
            program_config=_decode_rmsnorm_program_config(hidden_size),
            memory_config=shard_in_cfg,
            compute_kernel_config=sharded_compute_kernel_config,
        )
        # Leave the output L1 width-sharded on the same 16c 8x2 grid we
        # computed on. Downstream consumers (QKV and gate-up matmuls in
        # decode) are configured as DRAM-sharded matmuls that take this
        # exact layout as input; the previous trailing
        # ``sharded_to_interleaved`` here is what removed the umbrella
        # across LN -> matmul. If a caller still needs an interleaved
        # tensor (prefill, non-sharded QKV path), it will reshard itself
        # via ``to_memory_config``.
        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        return tt_out

    def _forward_prefill_sharded(self, inp, original_shape):
        # Block-sharded prefill RMSNorm (op 16). Consumes the BLOCK_SHARDED
        # attention-residual sum produced on the o_proj op-14 8x8 grid and
        # normalizes in place on that grid, so there is no reshard around the
        # post-attention LN. The gamma is the same [32, padded_dim] TILE-layout
        # ``tt_weight_sharded`` the decode path uses (the sharded LN kernel
        # accepts a [TILE_HEIGHT, width] TILE gamma; see
        # layernorm_device_operation.cpp gamma validation).
        eps = getattr(self.torch_layer, "variance_epsilon", getattr(self.torch_layer, "eps", 1e-6))
        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)
        shard_spec = inp.memory_config().shard_spec
        bbox = shard_spec.grid.bounding_box()
        grid_x = int(bbox.end.x - bbox.start.x + 1)
        grid_y = int(bbox.end.y - bbox.start.y + 1)
        block_h = int(shard_spec.shape[0]) // ttnn.TILE_SIZE
        block_w = int(shard_spec.shape[1]) // ttnn.TILE_SIZE
        subblock_w = min(4, block_w)
        while subblock_w > 1 and block_w % subblock_w != 0:
            subblock_w -= 1
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid_x, grid_y],
            subblock_w=subblock_w,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )
        sharded_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        print("Using RMSNorm")
        print("prefill sharded inp.shape:", inp.shape)
        tt_out = ttnn.rms_norm(
            inp,
            epsilon=eps,
            weight=self.tt_weight_sharded,
            program_config=program_config,
            memory_config=inp.memory_config(),
            compute_kernel_config=sharded_compute_kernel_config,
        )
        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        return tt_out

    def forward(self, inp):
        original_shape = inp.shape
        hidden_size = int(self.torch_layer.weight.shape[0])
        if _tp_requires_ccl(self.device) and int(original_shape[-1]) != hidden_size:
            out = super().forward(inp)
            if out.memory_config().buffer_type != ttnn.BufferType.L1:
                out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
            return out

        # Sharded LN fast path: decode-shape (M=1) and single-device or pure DP
        # (no TP CCL). TP decode uses the parent distributed path above when
        # the activation is already the local hidden shard (e.g. 384 on 1x4).
        is_decode = len(original_shape) >= 2 and int(original_shape[-2]) == 1
        if is_decode and not _tp_requires_ccl(self.device):
            return self._forward_decode_sharded(inp, original_shape)

        # Prefill block-sharded fast path: input already lives BLOCK_SHARDED on
        # the o_proj op-14 grid, so normalize there (no reshard). Single-device
        # / pure-DP only; TP and interleaved inputs fall through.
        if not _tp_requires_ccl(self.device) and inp.memory_config().is_sharded():
            return self._forward_prefill_sharded(inp, original_shape)

        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if _tp_requires_ccl(self.device) and int(inp.shape[-1]) != hidden_size:
            tt_out = super().forward(inp)
            if tt_out.memory_config().buffer_type != ttnn.BufferType.L1:
                tt_out = ttnn.to_memory_config(tt_out, ttnn.L1_MEMORY_CONFIG)
            if len(original_shape) == 3 and len(tt_out.shape) == 4:
                tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
            return tt_out
        eps = getattr(self.torch_layer, "variance_epsilon", getattr(self.torch_layer, "eps", 1e-6))
        print("Using RMSNorm")
        print("prefill inp.shape:", inp.shape)
        tt_out = ttnn.rms_norm(
            inp,
            epsilon=eps,
            weight=self.tt_weight_sharded if int(inp.shape[-1]) == hidden_size else self.weight_distributed,
            compute_kernel_config=self.compute_kernel_config,
        )
        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        return tt_out


@trace_enabled
class TTNNDotsOCRDecoderLayer(TTNNModule):
    def __init__(self):
        super().__init__()
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.self_attn = None
        self.mlp = None

    @classmethod
    def from_torch(cls, torch_layer, tp_decode_scheme: str = "row"):
        new_layer = cls()
        new_layer._fallback_torch_layer = torch_layer
        new_layer.attention_type = getattr(torch_layer, "attention_type", "full_attention")
        new_layer.tp_decode_scheme = tp_decode_scheme
        new_layer.input_layernorm = TTNNDotsOCRLocalShardRMSNorm.from_torch(torch_layer.input_layernorm)
        new_layer.post_attention_layernorm = TTNNDotsOCRLocalShardRMSNorm.from_torch(
            torch_layer.post_attention_layernorm
        )
        if tp_decode_scheme == "row":
            new_layer.self_attn = TTNNDotsOCRAttention.from_torch(torch_layer.self_attn)
            new_layer.mlp = TTNNDotsOCRMLP.from_torch(torch_layer.mlp)
        elif tp_decode_scheme == "col_parallel":
            # TP4-distributed (SP+TP) decode: the hidden/residual stays
            # TP-sharded (e.g. 384 on a 1x4 mesh) end to end. Keep attention on
            # the proven K-parallel QKV path by default; the N-parallel QKV A/B
            # path is opt-in because it collapses greedy OCR decode on structured
            # HTML/table tokens. The MLP still uses column-parallel gate/up and a
            # reduce_scatter-only down so the residual add stays hidden-sharded.
            use_n_parallel_attn = os.environ.get("DOTS_OCR_COL_PARALLEL_USE_N_PARALLEL_ATTN", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            new_layer._col_parallel_use_n_parallel_attn = use_n_parallel_attn
            new_layer.self_attn = TTNNDotsOCRAttention.from_torch(
                torch_layer.self_attn, qkv_n_parallel=use_n_parallel_attn
            )
            new_layer.mlp = TTNNDotsOCRMLPColParallelFusedGateUp.from_torch(torch_layer.mlp, replicated_output=False)
            new_layer.mlp_prefill = TTNNDotsOCRMLP.from_torch(torch_layer.mlp)
        elif tp_decode_scheme == "head_parallel":
            # Devstral-style HEAD-PARALLEL decode (TP=2, KV heads divide TP). The
            # attention block runs N-parallel head-local QKV (no reassembly
            # gather), head-local SDPA/cache, and a row-parallel reduce_scatter
            # o_proj that returns the hidden/TP shard. That output contract is
            # IDENTICAL to col_parallel (hidden-sharded residual), so the
            # RMSNorm + MLP wiring is shared with col_parallel; the only deltas
            # are head_parallel=True on the attention and that full hidden must be
            # replicated into the attention QKV for PREFILL too (handled in
            # forward via the n_parallel gather, which col_parallel only did for
            # decode).
            new_layer._col_parallel_use_n_parallel_attn = True
            new_layer._head_parallel = True
            new_layer.self_attn = TTNNDotsOCRAttention.from_torch(torch_layer.self_attn, head_parallel=True)
            new_layer.mlp = TTNNDotsOCRMLPColParallelFusedGateUp.from_torch(torch_layer.mlp, replicated_output=False)
            new_layer.mlp_prefill = TTNNDotsOCRMLP.from_torch(torch_layer.mlp)
        else:
            raise ValueError(f"Unsupported dots.ocr decoder TP decode scheme: {tp_decode_scheme}")
        if tp_decode_scheme in {"col_parallel", "head_parallel"} or _use_bfp8_decoder_weights(
            getattr(new_layer.self_attn, "layer_idx", None)
        ):
            new_layer.mlp.fused_gate_up_proj.set_weight_dtype(ttnn.bfloat8_b)
            if hasattr(new_layer, "mlp_prefill"):
                new_layer.mlp_prefill.set_weight_dtype(ttnn.bfloat8_b)
        return new_layer

    def call(self, *args, **kwds):
        # Keep only kwargs used by forward — unused kwargs with incompatible
        # dtypes (e.g. UINT8 from bool masks) cause ttnn.copy failures in trace replay.
        filtered = {k: kwds[k] for k in ("past_key_value", "cache_position") if k in kwds}
        return super().call(*args, **filtered)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        hidden_states = func_args[0]
        seq_len = hidden_states.shape[-2]
        layer_idx = self.self_attn.layer_idx
        past_key_value.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs,
    ):
        hs = _take_local_dp_batch(hidden_states, self.device)
        scheme = getattr(self, "tp_decode_scheme", "row")
        # head_parallel shares col_parallel's hidden/TP residual contract and
        # RMSNorm/MLP wiring; the attention block differs (head-local).
        is_col_parallel = scheme in {"col_parallel", "head_parallel"}
        is_head_parallel = scheme == "head_parallel"

        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hs.dtype != ttnn.bfloat16:
            hs = ttnn.typecast(hs, ttnn.bfloat16)

        hidden_dim = int(self.input_layernorm.torch_layer.weight.shape[0])
        rmsnorm_mode = _col_parallel_rmsnorm_mode() if is_col_parallel else "distributed"

        # Attention block. The residual stays in the TP-sharded hidden layout
        # (e.g. 384 on a 1x4 mesh); input_layernorm runs distributed (TP4) on
        # that shard. For col_parallel decode the normed shard is all-gathered
        # to full hidden so the column-parallel QKV contracts over full hidden.
        residual = hs
        seq_len = hs.shape[-2]
        is_decode = seq_len == 1
        decode_l1_mc = ttnn.L1_MEMORY_CONFIG if is_decode else None
        if is_col_parallel and is_decode and rmsnorm_mode == "full_multicore":
            hs = _full_hidden_rmsnorm_then_maybe_partition(
                self.input_layernorm,
                hs,
                self.device,
                hidden_dim,
                partition_output=not getattr(self, "_col_parallel_use_n_parallel_attn", False),
                use_multicore=True,
            )
        elif is_col_parallel and is_decode and rmsnorm_mode == "full_single":
            hs = _full_hidden_rmsnorm_then_maybe_partition(
                self.input_layernorm,
                hs,
                self.device,
                hidden_dim,
                partition_output=not getattr(self, "_col_parallel_use_n_parallel_attn", False),
                use_multicore=False,
            )
        else:
            hs = self.input_layernorm(hs)
        # Replicate full hidden into the N-parallel DECODE QKV (col_parallel and
        # head_parallel both consume full hidden in decode). Decode-only: for
        # PREFILL, head_parallel dispatches to the attention's head-sharded
        # ``_forward_prefill_head_sharded``, which takes the hidden/TP shard and
        # all_gathers internally -- gathering here too would DOUBLE-gather
        # (hidden/TP -> full hidden -> 2x full). col_parallel prefill is K-parallel
        # and also consumes the hidden/TP shard. So neither scheme gathers in prefill.
        if is_col_parallel and is_decode and getattr(self, "_col_parallel_use_n_parallel_attn", False):
            hs = _gather_tp_hidden_if_needed(hs, self.device, hidden_dim)
        attn_out, _ = self.self_attn(
            hidden_states=hs,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=past_key_value,
            cache_position=kwargs.get("cache_position"),
            decode_cur_pos_tt=kwargs.get("decode_cur_pos_tt"),
            decode_cos_sin=kwargs.get("decode_cos_sin"),
        )

        # Prefill block-sharded region (ops 14-16): the o_proj returns its
        # output BLOCK_SHARDED on the 8x8 grid, so do the attention residual-add
        # and post-attention RMSNorm on that grid (no reshards). The pre-norm
        # sum H1 is kept block-sharded as the MLP residual.
        if not is_decode and attn_out.memory_config().is_sharded():
            attn_mc = attn_out.memory_config()
            residual_bs = ttnn.to_memory_config(residual, attn_mc)
            ttnn.deallocate(residual)
            hs = ttnn.add(residual_bs, attn_out, memory_config=attn_mc)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(residual_bs)

            residual = hs  # H1 (block-sharded)
            normed = self.post_attention_layernorm(hs)

            # Op 17: the gate-up matmul consumes the BLOCK_SHARDED LN output
            # directly (block-sharded in0 -> DRAM out), so the LN->gate-up
            # reshard is gone. down_proj still emits interleaved, so op 22
            # reshards the H1 residual once to match mlp_out.
            mlp = getattr(self, "mlp_prefill", self.mlp)
            mlp_out = mlp(normed)
            ttnn.deallocate(normed)

            residual_il = ttnn.sharded_to_interleaved(residual, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(residual)
            hs = ttnn.add(residual_il, mlp_out)
            ttnn.deallocate(mlp_out)
            ttnn.deallocate(residual_il)
            return (hs,)

        attn_out = _match_residual_width(attn_out, residual, self.device)
        hs = (
            ttnn.add(residual, attn_out, memory_config=decode_l1_mc)
            if decode_l1_mc is not None
            else ttnn.add(residual, attn_out)
        )
        ttnn.deallocate(attn_out)

        # MLP block. Same SP+TP contract as the attention block: distributed
        # (TP4) RMSNorm on the sharded residual, then (col_parallel decode)
        # all-gather to full hidden for the column-parallel gate/up. The
        # row-parallel down does reduce_scatter only, so mlp_out comes back
        # hidden-sharded and the residual add stays on the sharded hidden.
        residual = hs
        if is_col_parallel and is_decode and rmsnorm_mode == "full_multicore":
            hs = _full_hidden_rmsnorm_then_maybe_partition(
                self.post_attention_layernorm,
                hs,
                self.device,
                hidden_dim,
                partition_output=False,
                use_multicore=True,
            )
        elif is_col_parallel and is_decode and rmsnorm_mode == "full_single":
            hs = _full_hidden_rmsnorm_then_maybe_partition(
                self.post_attention_layernorm,
                hs,
                self.device,
                hidden_dim,
                partition_output=False,
                use_multicore=False,
            )
        else:
            hs = self.post_attention_layernorm(hs)
        if is_col_parallel and is_decode and rmsnorm_mode == "distributed":
            hs = _gather_tp_hidden_if_needed(hs, self.device, hidden_dim)
        mlp = self.mlp if is_decode else getattr(self, "mlp_prefill", self.mlp)
        mlp_out = mlp(hs)

        hs = (
            ttnn.add(residual, mlp_out, memory_config=decode_l1_mc)
            if decode_l1_mc is not None
            else ttnn.add(residual, mlp_out)
        )
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        # CRITICAL: Return tuple — Qwen2Model does layer_outputs[0]
        return (hs,)


class TTNNDotsOCRLayerStack(TTNNLayerStack):
    def call(self, *args, **kwds):
        filtered = {k: kwds[k] for k in ("past_key_value", "cache_position") if k in kwds}
        return super().call(*args, **filtered)

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()
        shared_buf = None
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "_decode_cur_pos") and attn._decode_cur_pos is not None:
                if shared_buf is None:
                    shared_buf = attn._decode_cur_pos
                else:
                    ttnn.deallocate(attn._decode_cur_pos)
                    attn._decode_cur_pos = shared_buf
        self._shared_decode_cur_pos = shared_buf

    def forward(self, hidden_states, **kwargs):
        seq_len = hidden_states.shape[-2]
        if (
            seq_len == 1
            and getattr(self, "_shared_decode_cur_pos", None) is not None
            and self.layers
            and "decode_cos_sin" not in kwargs
        ):
            attn0 = getattr(self.layers[0], "self_attn", None)
            rotary_setup = getattr(attn0, "_rotary_setup", None) if attn0 is not None else None
            cache_position = kwargs.get("cache_position")
            if rotary_setup is not None and cache_position is not None:
                cur_pos_tt = self._materialize_shared_cur_pos(cache_position)
                if cur_pos_tt is not None:
                    kwargs["decode_cur_pos_tt"] = cur_pos_tt
                    kwargs["decode_cos_sin"] = rotary_setup.get_cos_sin_for_decode(cur_pos_tt)

        for layer in self.layers:
            layer_output = layer.forward(hidden_states, **kwargs)
            hidden_states = layer_output[0]
        return hidden_states

    def _materialize_shared_cur_pos(self, cache_position):
        cp = cache_position
        if hasattr(cp, "ttnn_tensor") and cp.ttnn_tensor is not None:
            cp = cp.ttnn_tensor
        if not isinstance(cp, ttnn.Tensor):
            return None
        if len(cp.shape) > 1:
            total_elems = 1
            for d in cp.shape:
                total_elems *= d
            cp = ttnn.reshape(cp, (total_elems,))
        if cp.shape[0] > 1:
            cp = ttnn.slice(cp, [0], [1])
        ttnn.copy(cp, self._shared_decode_cur_pos)
        return self._shared_decode_cur_pos

    def pre_trace_execute(self, func_args, func_kwargs):
        cache_position = func_kwargs.get("cache_position")
        if cache_position is None:
            return

        cp = cache_position
        if hasattr(cp, "ttnn_tensor") and cp.ttnn_tensor is not None:
            cp = cp.ttnn_tensor

        if len(cp.shape) > 1:
            total = 1
            for d in cp.shape:
                total *= d
            cp = ttnn.reshape(cp, (total,))

        if cp.shape[0] > 1:
            cp = ttnn.slice(cp, [0], [1])

        if hasattr(self, "_shared_decode_cur_pos") and self._shared_decode_cur_pos is not None:
            ttnn.copy(cp, self._shared_decode_cur_pos)
            return

        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "_decode_cur_pos") and attn._decode_cur_pos is not None:
                ttnn.copy(cp, attn._decode_cur_pos)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        hidden_states = func_args[0]
        seq_len = hidden_states.shape[-2]
        for layer in self.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer_idx = layer.self_attn.layer_idx
                past_key_value.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)
