# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Routed Experts module.

Decode (seq_len=1): on-device via sparse_matmul with top-k sparsity
Prefill (seq_len>1): on-device via sparse_matmul with all-ones sparsity (gpt_oss pattern)
"""

import os

import torch
import ttnn
from loguru import logger

from .decode import FusedRouting, PackedRouting, decode_forward, decode_forward_fused
from .prefill import create_prefill_sparsity, prefill_forward
from .weights import ExpertWeights, load_expert_weights
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.ccl import ccl_allreduce, ccl_lane_gather_rows, ccl_lane_scatter_rows
from models.demos.gemma4.tt.mesh_orientation import dispatch_min_tokens_default, dispatch_prefill_default

_WARNED: set = set()


def _warn_once(key: str, message: str):
    """One warning per process for a kernel path that refuses this device (every layer would repeat it)."""
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)


class Gemma4ExpertConfig:
    """Configuration for the routed experts, derived from HF config."""

    def __init__(self, hf_config):
        self.hidden_size = hf_config.hidden_size
        self.num_experts = hf_config.num_experts
        self.top_k = hf_config.top_k_experts
        self.moe_intermediate_size = hf_config.moe_intermediate_size


class Gemma4Experts:
    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        ccl_manager,
        mesh_config,
        program_config,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        layer_idx=0,
        num_layers=1,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config

        # Load weights to device for sparse_matmul
        self.weights = load_expert_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )
        # Whole-expert sharding: every chip holds E / (ep * tp) experts at full width (ep over the
        # lane axis, tp over the other), and the dense routing is partitioned to its expert
        # columns on device. The ep-axis combine sums the lanes' partial outputs, the tp-axis
        # all-reduce the columns' partials, exactly as before; only the per-chip expert set and
        # the tensor widths changed (Qwen3.6 MoE layout).
        self.ep = self.weights.ep
        self.num_local_experts = self.weights.num_local_experts
        self.local_expert_mask = None
        # Prefill sparsity: all-ones over the local experts (reused for every prefill call)
        self.prefill_sparsity = create_prefill_sparsity(mesh_device, self.num_local_experts)
        # Optional token-dispatch prefill (DeepSeek dispatch/combine + fused GeGLU experts): prompts of at
        # least GEMMA4_MOE_DISPATCH_MIN_TOKENS rows route tokens to the chips owning their experts instead of
        # running every chip's experts on every token; shorter calls (the traced decode frame) keep the
        # sparse_matmul path above.
        # The path is the default on a Blackhole (4,1) column mesh (the 4-chip line as the vLLM build views it,
        # see tt/mesh_orientation.py), min tokens 128 there; GEMMA4_MOE_DISPATCH_PREFILL / _MIN_TOKENS override.
        self.dispatch_experts = None
        is_mesh = hasattr(mesh_device, "shape")
        _shape = tuple(mesh_device.shape) if is_mesh else None
        self.dispatch_min_tokens = dispatch_min_tokens_default(_shape, blackhole=is_blackhole())
        if dispatch_prefill_default(_shape, blackhole=is_blackhole()) and is_mesh and mesh_device.get_num_devices() > 1:
            # imported here: the DeepSeek dispatch/combine stack is only needed on this path
            from models.demos.gemma4.tt.experts.dispatch_prefill import Gemma4DispatchExperts

            reason = Gemma4DispatchExperts.unsupported_reason(mesh_device)
            if reason is not None:
                _warn_once("dispatch_prefill", f"dispatch prefill off, sparse_matmul experts: {reason}")
            else:
                placeholder = bool(getattr(state_dict, "is_placeholder", False))
                hf_list = (
                    None
                    if placeholder
                    else Gemma4DispatchExperts.hf_expert_list(state_dict["gate_up_proj"], state_dict["down_proj"])
                )
                self.dispatch_experts = Gemma4DispatchExperts(
                    mesh_device,
                    config,
                    hf_list,
                    tensor_cache_path=tensor_cache_path,
                    weights_dtype=weight_dtype,
                    ccl_manager=self.ccl_manager,
                )
        # Optional local token-sorted routed prefill (GEMMA4_MOE_LOCAL_ROUTED_PREFILL=1, experts/local_routed_prefill.py):
        # on the whole-expert tp-only shapes every chip holds every token, so the routed (token, expert) pairs are
        # sorted and computed locally (GeGLU routed-expert kernel) with no collective inside the experts; calls of at
        # least GEMMA4_MOE_LOCAL_ROUTED_MIN_TOKENS rows (a multiple of 64) take it, shorter ones keep sparse_matmul.
        self.local_routed = None
        if (
            os.environ.get("GEMMA4_MOE_LOCAL_ROUTED_PREFILL", "0") == "1"
            and is_mesh
            and mesh_device.get_num_devices() > 1
            and self.ep == 1
        ):
            from models.demos.gemma4.tt.experts.local_routed_prefill import Gemma4LocalRoutedExperts, hf_expert_list

            reason = Gemma4LocalRoutedExperts.unsupported_reason(mesh_device)
            if reason is not None:
                _warn_once("local_routed", f"local routed prefill off, sparse_matmul experts: {reason}")
            else:
                placeholder = bool(getattr(state_dict, "is_placeholder", False))
                hf_list = None if placeholder else hf_expert_list(state_dict["gate_up_proj"], state_dict["down_proj"])
                self.local_routed = Gemma4LocalRoutedExperts(
                    mesh_device,
                    config,
                    hf_list,
                    mesh_config=mesh_config,
                    tensor_cache_path=tensor_cache_path,
                    layer_idx=layer_idx,
                    weights_dtype=weight_dtype,
                )
        # Optional decode MoE kernel for the lane decode frame (token dispatch within the column, moe_compute):
        # calls of at most the frame's rows take it; longer prefill calls keep the paths above.
        self.moe_decode = None
        if os.environ.get("GEMMA4_MOE_DECODE_KERNEL", "0") == "1" and is_mesh and mesh_device.get_num_devices() > 1:
            from models.demos.gemma4.tt.experts.moe_decode import Gemma4MoEDecode

            assert not getattr(state_dict, "is_placeholder", False), "the decode MoE kernel needs real expert weights"
            self.moe_decode = Gemma4MoEDecode(
                mesh_device,
                config,
                state_dict["gate_up_proj"],
                state_dict["down_proj"],
                layer_idx=layer_idx,
                num_layers=num_layers,
            )

    def wants_sparse_routing(self, hidden_states):
        """True when this call will run the decode MoE kernel, which takes the router's (values, indices)
        directly (``GEMMA4_SPARSE_ROUTING=0`` keeps the dense routing matrix everywhere)."""
        if os.environ.get("GEMMA4_SPARSE_ROUTING", "1") == "0":
            return False
        if self.moe_decode is None:
            # the local routed prefill takes the router's (values, indices) too: no dense scatter, no second top-k
            return self._use_local_routed(hidden_states) or self._use_dispatch_sparse_routing(hidden_states)
        rows = int(hidden_states.shape[2])
        lane_sharded = bool(self.mesh_config is not None and getattr(self.mesh_config, "lane_sharded", False))
        if lane_sharded and self.ep > 1:
            if getattr(self.mesh_config, "_g4_single_lane_prefill_owner", None) is not None:
                return False  # single-user prefill: the owner block feeds the kernel its dense routing matrix
            rows = (rows + (-rows) % 32) * self.mesh_config.lanes
        return 1 < rows <= self.moe_decode.frame_rows

    @staticmethod
    def _routing_parts(routing):
        """(values, indices) for sparse routing, (dense, None) for the dense matrix."""
        return (routing[0], routing[1]) if isinstance(routing, tuple) else (routing, None)

    @staticmethod
    def _owned_copy(tensor, source):
        """True when ``tensor`` is a new buffer this module made from the caller's ``source`` (a pad that grew the
        padded shape) and may be freed after the lane gather; False when it is ``source`` itself or a pad within the
        tile padding, which ``ttnn.pad`` returns as a view of the caller's buffer (only the logical shape changes).
        The caller owns ``source``: the decoder layer frees ``expert_input`` after the MoE call, and the MoE unit
        tests run the same input through the block twice."""
        if tensor is source:
            return False
        try:
            return tuple(tensor.padded_shape) != tuple(source.padded_shape)
        except Exception:  # noqa: BLE001 - no padded shape on this storage: keep the caller's tensor alive
            return False

    def _forward_packed(self, hidden_states, routing, lane_sharded):
        """GEMMA4_MOE_FUSED=1 on the decode MoE kernel rows: the router's packed (weights | ids) rows travel as ONE
        tensor (one lane gather instead of values + indices + their typecasts), and ``moe_decode.forward`` builds the
        kernel's dispatch inputs with ``custom_ops/moe_frame_prep`` (no pad / layout / partition glue)."""
        assert self.moe_decode is not None, "the packed routing only feeds the decode MoE kernel"
        rows = int(hidden_states.shape[2])
        pad_rows = (-rows) % 32
        packed = routing.packed
        if lane_sharded and self.ep > 1:
            assert getattr(self.mesh_config, "_g4_single_lane_prefill_owner", None) is None, "owner prefill is dense"
            x_in = hidden_states  # the caller's tensor: never freed here
            if pad_rows:
                hidden_states = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, pad_rows), (0, 0)], value=0.0)
            if int(packed.shape[2]) != rows + pad_rows:  # the router op always writes one row tile (32 rows)
                packed = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, rows + pad_rows, int(packed.shape[3])])
            hidden_states = ccl_lane_gather_rows(
                hidden_states,
                self.mesh_config,
                consume=self._owned_copy(hidden_states, x_in),
                ccl_manager=self.ccl_manager,
            )
            if os.environ.get("GEMMA4_PACKED_GATHER_TILE", "0") == "1":
                # fallback if the lane all_gather refuses the 32-byte ROW_MAJOR pages: gather one tile per lane
                packed = ttnn.to_layout(packed, ttnn.TILE_LAYOUT)
                packed = ccl_lane_gather_rows(packed, self.mesh_config, ccl_manager=self.ccl_manager)
                packed = ttnn.to_layout(packed, ttnn.ROW_MAJOR_LAYOUT)
            else:
                packed = ccl_lane_gather_rows(packed, self.mesh_config, ccl_manager=self.ccl_manager)
            assert self._use_decode_kernel(hidden_states), "the gathered rows exceed the kernel frame"
            out = self.moe_decode.forward(hidden_states, PackedRouting(packed, routing.k))
            out = ttnn.mesh_partition(out, dim=2, cluster_axis=self.mesh_config.ep_axis)
            if pad_rows:
                out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, rows, out.shape[-1]])
            return out
        assert self._use_decode_kernel(hidden_states), "packed routing without the kernel"
        if pad_rows:  # the frame-prep op reads whole row tiles: zero the tile padding explicitly
            hidden_states = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, pad_rows), (0, 0)], value=0.0)
        out = self.moe_decode.forward(hidden_states, PackedRouting(packed, routing.k))
        if pad_rows:
            out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, rows, out.shape[-1]])
        return out

    def __call__(self, hidden_states, dense_routing):
        lane_sharded = bool(self.mesh_config is not None and getattr(self.mesh_config, "lane_sharded", False))
        if isinstance(dense_routing, PackedRouting):
            return self._forward_packed(hidden_states, dense_routing, lane_sharded)
        if isinstance(dense_routing, FusedRouting):
            # GEMMA4_MOE_FUSED=1: the fused router's tables, decode frame on whole-expert chips (no lane exchange)
            assert self.ep == 1 and not lane_sharded, "the fused MoE decode path serves the tp-only shapes"
            return decode_forward_fused(
                hidden_states,
                dense_routing,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
            )
        # With EP=1 every lane holds the full expert set, so lane-local rows need no exchange.
        if lane_sharded and self.ep > 1:
            # Lanes hold different rows: gather every lane's rows (and their routing), run the
            # local experts on all of them, then reduce_scatter the rows back to their lanes
            # (summing the ep partials) and all-reduce over tp -- the SharedMLP choreography.
            # Decode hands each lane 1..32 rows; the gathered tensor then has lanes*rows rows and
            # takes the (tile-granular) prefill path, so pad every lane to a tile of rows first and
            # slice the lane's own rows back after the scatter. Zero routing on the pad rows keeps
            # them out of the expert sums.
            rows = hidden_states.shape[2]
            pad_rows = (-rows) % 32
            x_in = hidden_states  # the caller's tensor: never freed here
            dense_routing, sparse_idx = self._routing_parts(dense_routing)
            if sparse_idx is not None:
                sparse_idx = ttnn.typecast(sparse_idx, ttnn.bfloat16)  # exact for expert ids; the gather wants bf16
            if pad_rows:
                hidden_states = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, pad_rows), (0, 0)], value=0.0)
                dense_routing = ttnn.pad(dense_routing, [(0, 0), (0, 0), (0, pad_rows), (0, 0)], value=0.0)
                if sparse_idx is not None:
                    # Distinct ids per k on the pad rows (a row of equal ids hangs the decode kernel's dispatch).
                    k = int(sparse_idx.shape[-1])
                    pad_ids = ttnn.from_torch(
                        torch.arange(k, dtype=torch.float32).repeat(pad_rows, 1).reshape(1, 1, pad_rows, k),
                        device=self.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    )
                    sparse_idx = ttnn.concat([ttnn.to_layout(sparse_idx, ttnn.ROW_MAJOR_LAYOUT), pad_ids], dim=2)
                    sparse_idx = ttnn.to_layout(sparse_idx, ttnn.TILE_LAYOUT)
            hidden_states = ccl_lane_gather_rows(
                hidden_states,
                self.mesh_config,
                consume=self._owned_copy(hidden_states, x_in),
                ccl_manager=self.ccl_manager,
            )
            dense_routing = ccl_lane_gather_rows(dense_routing, self.mesh_config, ccl_manager=self.ccl_manager)
            if sparse_idx is not None:
                sparse_idx = ccl_lane_gather_rows(sparse_idx, self.mesh_config, ccl_manager=self.ccl_manager)
                if os.environ.get("GEMMA4_SPARSE_DEBUG") == "1":
                    ttnn.synchronize_device(self.mesh_device)
                    logger.info(
                        f"[sparse-dbg] lane-gathered indices: shape={sparse_idx.shape} dtype={sparse_idx.dtype}"
                    )
            owner = getattr(self.mesh_config, "_g4_single_lane_prefill_owner", None)
            if owner is not None and os.environ.get("GEMMA4_EP_LANE_FASTPATH", "1") != "0":
                # Single-user prefill: only the owner lane's rows are real. Every lane still needs
                # them (the experts are sharded over the lanes), but running the local experts on
                # the other lanes' scratch rows would be 1 - 1/lanes wasted work. Slice the owner's
                # block out of the gather, compute 1x rows, and all-reduce the ep partials so every
                # lane ends up holding the owner's result (the other lanes' rows are discarded
                # downstream anyway).
                block = int(hidden_states.shape[2]) // self.mesh_config.lanes
                lo, hi = owner * block, (owner + 1) * block
                hidden_owner = ttnn.slice(hidden_states, [0, 0, lo, 0], [1, 1, hi, hidden_states.shape[-1]])
                routing_owner = ttnn.slice(dense_routing, [0, 0, lo, 0], [1, 1, hi, dense_routing.shape[-1]])
                if sparse_idx is not None:
                    idx_owner = ttnn.slice(sparse_idx, [0, 0, lo, 0], [1, 1, hi, sparse_idx.shape[-1]])
                    sparse_idx.deallocate(True)
                    routing_owner = (routing_owner, ttnn.typecast(idx_owner, ttnn.uint16))
                hidden_states.deallocate(True)
                dense_routing.deallocate(True)
                # The owner block keeps the dense routing matrix: the kernel's sparse (values, indices) input is
                # for decode frames (served A/B 2026-10-23: kernel vs dispatch for 128-token prefills within 3%).
                assert not isinstance(routing_owner, tuple), "sparse routing only feeds the decode frames"
                if self._use_decode_kernel(hidden_owner):
                    # the single-user prefill's rows: the shared op + ROW_MAJOR tail (decode=False)
                    out = self.moe_decode.forward(hidden_owner, routing_owner, decode=False)  # full sum on every chip
                elif self._use_dispatch(hidden_owner):
                    # already the full sum on every chip
                    out = self.dispatch_experts.forward(hidden_owner, routing_owner)
                else:
                    out = self._forward(hidden_owner, routing_owner, combine=False)
                    out = ccl_allreduce(
                        out, self.mesh_config, self.ccl_manager, axis=self.mesh_config.ep_axis, group_size=self.ep
                    )
                    if self.mesh_config.tp > 1:
                        out = ccl_allreduce(out, self.mesh_config, self.ccl_manager)
            elif self._use_dispatch(hidden_states) or self._use_decode_kernel(hidden_states):
                # full sum of every lane's rows on every chip: each lane keeps its own block of rows
                if self._use_decode_kernel(hidden_states):
                    if sparse_idx is not None:
                        dense_routing = (dense_routing, ttnn.typecast(sparse_idx, ttnn.uint16))
                    out = self.moe_decode.forward(hidden_states, dense_routing)
                else:
                    assert sparse_idx is None, "sparse routing only feeds the decode MoE kernel"
                    out = self.dispatch_experts.forward(hidden_states, dense_routing)
                out = ttnn.mesh_partition(out, dim=2, cluster_axis=self.mesh_config.ep_axis)
            else:
                out = self._forward(hidden_states, dense_routing, combine=False)
                out = ccl_lane_scatter_rows(out, self.mesh_config, ccl_manager=self.ccl_manager)
            if pad_rows:
                out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, rows, out.shape[-1]])
            return out
        if self._use_decode_kernel(hidden_states):
            from models.demos.gemma4.tt.experts.moe_decode import kernel_decode_frame

            return self.moe_decode.forward(  # dense matrix or (values, indices)
                hidden_states,
                dense_routing,
                decode=kernel_decode_frame(self.mesh_config, int(hidden_states.shape[2])),
            )
        if self._use_local_routed(hidden_states):
            out = self.local_routed.forward(hidden_states, dense_routing)  # dense matrix or (values, indices)
            if self.mesh_config is not None and self.mesh_config.tp > 1:
                out = ccl_allreduce(out, self.mesh_config, self.ccl_manager)
            return out
        if self._use_dispatch(hidden_states):
            # dense matrix, or the router's (values, indices) with GEMMA4_MOE_DISPATCH_SPARSE_ROUTING=1
            return self.dispatch_experts.forward(hidden_states, dense_routing)
        assert not isinstance(dense_routing, tuple), "sparse routing only feeds the decode MoE kernel"
        return self._forward(hidden_states, dense_routing, combine=True)

    def _use_decode_kernel(self, hidden_states):
        return self.moe_decode is not None and 1 < int(hidden_states.shape[2]) <= self.moe_decode.frame_rows

    def _use_local_routed(self, hidden_states):
        if self.local_routed is None:
            return False
        if bool(self.mesh_config is not None and getattr(self.mesh_config, "lane_sharded", False)):
            return False
        rows = int(hidden_states.shape[2])
        return rows >= self.local_routed.min_tokens() and rows % 64 == 0

    def _use_dispatch(self, hidden_states):
        return self.dispatch_experts is not None and int(hidden_states.shape[2]) >= self.dispatch_min_tokens

    def _use_dispatch_sparse_routing(self, hidden_states):
        """GEMMA4_MOE_DISPATCH_SPARSE_ROUTING=1: a dispatch-prefill call takes the router's (values, indices)
        straight from its top-k (no dense [S, E] scatter in the router, no second top-k in the dispatch chunk);
        the lane paths keep the dense matrix (their gathers and owner slices are written for it)."""
        if os.environ.get("GEMMA4_MOE_DISPATCH_SPARSE_ROUTING", "0") != "1":
            return False
        if bool(self.mesh_config is not None and getattr(self.mesh_config, "lane_sharded", False)):
            return False
        return self._use_dispatch(hidden_states)

    def _forward(self, hidden_states, dense_routing, combine=True):
        """
        Expert forward — fully on-device.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            dense_routing: [1, 1, seq_len, num_experts] on device

        Returns:
            output: [1, 1, seq_len, hidden_size] on device
        """
        seq_len = hidden_states.shape[2]

        if seq_len == 1:
            return decode_forward(
                local_expert_mask=self.local_expert_mask,
                combine=combine,
                hidden_states=hidden_states,
                routing_weights=dense_routing,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
            )
        else:
            assert seq_len % 32 == 0, f"Prefill seq_len must be a multiple of 32, got {seq_len}"
            return prefill_forward(
                num_local_experts=self.num_local_experts,
                combine=combine,
                hidden_states=hidden_states,
                routing_weights=dense_routing,
                weights=self.weights,
                config=self.config,
                prefill_sparsity=self.prefill_sparsity,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
            )
