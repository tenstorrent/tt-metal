# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Top-K Router implementation for Mixture of Experts (MoE) in GPT-OSS.

This module implements the routing mechanism that selects which experts should
process each token in the MoE architecture. The router uses a learned linear
transformation followed by top-k selection to assign tokens to experts.

"""

import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from .fused_decode.config import ROUTER_DECODE_WEIGHT_DTYPE
from .fused_decode.stream import NBIAS, as_stream_tensor, linear_stream_rows, stream_linear_layout


def topk_router(g, experts_per_token, use_throughput_experts, softmax_compute_config=None):
    typecast_needed = False
    if g.dtype != ttnn.bfloat16:
        g_og = g
        typecast_needed = True
        g = ttnn.typecast(g, dtype=ttnn.bfloat16)

    expert_weights, expert_indices = ttnn.topk(g, k=experts_per_token, dim=-1, sorted=True)
    # ttnn.scatter needs input and src in one dtype and the weights are bfloat16, so take the zeros from the bfloat16 g.
    # zeros_like(g, dtype=...) on a bfloat8_b g would fill on host and write to device, which trace capture rejects.
    dense_zeros = None if use_throughput_experts else ttnn.zeros_like(g)
    if typecast_needed:
        g.deallocate(True)
        g = g_og
    if softmax_compute_config is None:
        softmax_compute_config = ttnn.init_device_compute_kernel_config(
            g.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    expert_weights = ttnn.softmax(
        expert_weights, dim=1, numeric_stable=True, compute_kernel_config=softmax_compute_config
    )
    if use_throughput_experts:
        return expert_indices, expert_weights
    else:
        return expert_indices, ttnn.scatter(dense_zeros, dim=1, index=expert_indices, src=expert_weights)


# Rows of 32 per core of the batched router's full-K program (_batched_router_linear): bf16 in0 blocks of per_core_M x
# 90 tiles (180 KiB per row of tiles) plus the 90-tile weight; 32 users x 128 tokens need 2 on 11x10 cores.
_BATCHED_ROUTER_MAX_PER_CORE_M = 2


class TopKRouter:
    def __init__(
        self, mesh_device, hf_config, state_dict, tensor_cache_path=None, indexed_decode=False, ccl_manager=None
    ):
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.tensor_cache_path = tensor_cache_path
        torch_weight = state_dict["weight"].transpose(0, 1) if state_dict else None
        torch_bias = state_dict["bias"].unsqueeze(0) if state_dict else None
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias = ttnn.as_tensor(
            torch_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Keep compute_config=None for linear (known quality-safe default)
        # Custom compute configs were previously found to cause quality degradation
        self.compute_config = None
        self._batched_router_kernel_config = None  # _batched_router_linear (built on first use)

        # Cache softmax compute config (same as what topk_router creates per-call)
        self.softmax_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        if indexed_decode:
            assert ccl_manager is not None, "fused decode routing shares its output buffers through the CCL manager"
            self._init_indexed_decode(mesh_device, torch_weight, torch_bias, tensor_cache_path, ccl_manager)

        # Fused op support: matmul + topk + softmax in one kernel
        # The fused kernel uses 4 groups of 3 cores, one per N-tile (32 experts
        # each), so it requires exactly 128 experts and 12 DRAM-aligned cores.
        # Blackhole has only 8 DRAM banks; use the generic router on that architecture.
        # Issue for native 8 bank BH support: https://github.com/tenstorrent/tt-metal/issues/57186
        self.use_fused_op = self.num_experts == 128 and not ttnn.device.is_blackhole(mesh_device)
        self._fused_bias = None
        # Keep the original unsharded bias for fused op initialization
        # (ttnn.as_tensor shards self.bias across the mesh, but the fused op
        # needs the full [1, num_experts] bias replicated on every device)
        if self.use_fused_op and state_dict:
            self._bias_torch = state_dict["bias"].unsqueeze(0).to(torch.bfloat16)
        else:
            self._bias_torch = None

    def _init_indexed_decode(self, mesh_device, torch_weight, torch_bias, tensor_cache_path, ccl_manager):
        # Fused decode router (fused_decode/stream.py: LinearStream out_mode 4): [hidden, num_experts] + bias streamed
        # from DRAM, then top-k and the softmax over the selected logits in the same op.
        self.ccl_manager = ccl_manager
        banks = mesh_device.dram_grid_size().x
        layout = None
        if torch_weight is not None:
            layout = stream_linear_layout(torch_weight, torch_bias.reshape(-1), banks)
        self.weight_stream = as_stream_tensor(
            mesh_device,
            [layout] * mesh_device.shape[1] if layout is not None else None,
            linear_stream_rows(mesh_device, self.hidden_dim, self.num_experts),
            ROUTER_DECODE_WEIGHT_DTYPE,
            get_cache_file_name(tensor_cache_path, f"weight_stream_e{self.num_experts}_b{banks}_nb{NBIAS}"),
        )

    def _init_fused_op(self, device, B):
        """Lazily initialize fused op tensors (bias broadcast + output pre-alloc)."""
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None

        if self._fused_bias is None:
            if self._bias_torch is not None:
                # Use the original unsharded bias (self._bias_torch is [1, num_experts])
                # and broadcast to [B, num_experts] so every tile row has the bias vector.
                bias_bcast = self._bias_torch.expand(B, -1).contiguous()
            else:
                bias_bcast = None
            self._fused_bias = ttnn.as_tensor(
                bias_bcast,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
                cache_file_name=get_cache_file_name(self.tensor_cache_path, f"fused_bias_B{B}"),
            )

    def with_expert_order(self, order):
        """A router whose output column v is this router's expert order[v] (order: a permutation of the experts).

        Shares everything but the [hidden, E] weight and [1, E] bias, which are column-permuted copies (bf16 in and
        out, replicated like the originals), so top-k returns the same experts renumbered (expert order[v] as id v) up
        to rounding: the prefill logits come out of the linear in the input's bfloat8_b (one exponent per 16 columns
        of a row), so permuting the columns regroups the shared exponents and near-tied logits can round, and then
        rank, the other way (measured on layer 12: ~10% of logits differ by one bf16 ulp; 4.5% of tokens change one
        top-4 expert, the MoE output matches HF equally well). Used by the routed prefill experts, whose devices hold
        the experts in that order (experts/prefill_routed.py)."""
        import copy

        perm = torch.as_tensor(order, dtype=torch.long)
        assert sorted(perm.tolist()) == list(range(self.num_experts)), f"not a permutation of the experts: {order}"
        out = copy.copy(self)
        for name in ("weight", "bias"):
            t = getattr(self, name)
            host = ttnn.to_torch(ttnn.get_device_tensors(t)[0])
            host = host.reshape(-1, host.shape[-1])[:, perm]
            setattr(
                out,
                name,
                ttnn.from_torch(
                    host.contiguous(),
                    device=t.device(),
                    dtype=t.dtype,
                    layout=t.layout,
                    memory_config=t.memory_config(),
                    mesh_mapper=ttnn.ReplicateTensorToMesh(t.device()),
                ),
            )
        return out

    def _batched_router_linear(self, x4):
        """Router logits of a batched prefill of short users, x4 [B, 1, S, H] (S <= 128) -> [B, 1, S, E] in DRAM.

        Bit-identical to the batch-1 linear of each user (L1 output: MatmulMultiCoreProgramConfig, one output tile per
        step, all K tiles accumulated in the dest register, then the bias): a 1D in1-multicast program with
        in0_block_w = all of K (one K block, so the same in-dest sum) and the batch-1 call's kernel config (HiFi2,
        bf16 dest; ttnn's default for a user-given program differs). Measured on BH 1x4, 32 users x 128 tokens
        (l20/mb_router2.py): 0.200 -> 0.118 ms per linear, 0 / 131072 logits differ; the DRAM output also keeps the
        top-k's pad multi-core (from L1 it runs on one core, 80 us). Users of a partial tile of rows, or more rows per
        core than _BATCHED_ROUTER_MAX_PER_CORE_M (L1 of the full-K blocks), keep the [B, 1, S, H] linear with the L1
        output (ttnn picks the batch-1 call's program), copied to DRAM."""
        M_tiles = x4.shape[0] * (x4.shape[-2] // ttnn.TILE_SIZE)
        k_tiles = self.hidden_dim // ttnn.TILE_SIZE
        grid = x4.device().compute_with_storage_grid_size()
        per_core_M = -(-M_tiles // (grid.x * grid.y))
        if (
            x4.shape[-2] % ttnn.TILE_SIZE
            or self.hidden_dim % ttnn.TILE_SIZE
            or self.num_experts != ttnn.TILE_SIZE
            or per_core_M > _BATCHED_ROUTER_MAX_PER_CORE_M
        ):
            logits = ttnn.linear(
                x4,
                self.weight,
                bias=self.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
            )
            logits_dram = ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(logits)
            return logits_dram
        if self._batched_router_kernel_config is None:
            self._batched_router_kernel_config = ttnn.init_device_compute_kernel_config(
                x4.device().arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=k_tiles,
            out_subblock_h=max(h for h in (1, 2, 4) if per_core_M % h == 0),
            out_subblock_w=1,
            per_core_M=per_core_M,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        return ttnn.linear(
            x4,
            self.weight,
            bias=self.bias,
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._batched_router_kernel_config,
        )

    def __call__(self, hidden_states, use_throughput_experts, batch_size=1):
        # Compute actual token count from the tensor volume before reshape,
        # since shape[0] after reshape returns the tile-padded dimension
        # (e.g. 8 tokens padded to 32 in TILE_LAYOUT).
        actual_tokens = hidden_states.volume() // self.hidden_dim
        if batch_size > 1 and actual_tokens // batch_size <= 128:
            # Batched prefill of short users: each user's rows must equal its batch-1 router (<= 128 tokens: a linear
            # with an L1 output, which ttnn runs as MatmulMultiCoreProgramConfig: every output tile sums all K tiles
            # in the dest register in order, HiFi2, bf16 dest). A B*S-row call with ttnn's default program would round
            # differently. _batched_router_linear runs the same sum on a full-K 1D program (tests/unit/
            # test_batched_prefill.py checks ids and weights against the batch-1 router of each user).
            per_user = actual_tokens // batch_size
            x4 = ttnn.reshape(hidden_states, (batch_size, 1, per_user, self.hidden_dim))
            router_logits = ttnn.reshape(self._batched_router_linear(x4), (actual_tokens, self.num_experts))
            expert_indices, expert_weights = topk_router(
                router_logits, self.top_k, use_throughput_experts, self.softmax_compute_config
            )
            ttnn.deallocate(router_logits)
            return expert_indices, expert_weights
        if (
            batch_size == 1
            and 32 < actual_tokens <= 128
            and actual_tokens % ttnn.TILE_SIZE == 0
            and self.num_experts == ttnn.TILE_SIZE
            and ttnn.device.is_blackhole(hidden_states.device())
        ):
            # Batch-1 prefill of up to 128 tokens: the same full-K program as the batched path above, bit-identical to
            # the default linear (one K block summed in dest) and reading the K tiles in one block instead of one by
            # one (measured 77.5 -> 12.3 us at 128 tokens). Not at 256 rows: there the default linear sums K in more
            # than one block and the logits differ.
            x4 = ttnn.reshape(hidden_states, (1, 1, actual_tokens, self.hidden_dim))
            router_logits = ttnn.reshape(self._batched_router_linear(x4), (actual_tokens, self.num_experts))
            expert_indices, expert_weights = topk_router(
                router_logits, self.top_k, use_throughput_experts, self.softmax_compute_config
            )
            ttnn.deallocate(router_logits)
            return expert_indices, expert_weights
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))

        # Fused op only supports decode mode (B=32, seq_len=1 → shape [32, hidden_dim])
        if self.use_fused_op and actual_tokens == 32 and use_throughput_experts:
            return self._fused_call(hidden_states, use_throughput_experts)

        # Use L1 for decode (small tensors), DRAM for prefill (large sequences)
        is_decode = actual_tokens <= 128
        mem_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        router_logits = ttnn.linear(
            hidden_states,
            self.weight,
            bias=self.bias,
            memory_config=mem_config,
            compute_kernel_config=self.compute_config,
        )

        expert_indices, expert_weights = topk_router(
            router_logits, self.top_k, use_throughput_experts, self.softmax_compute_config
        )
        ttnn.deallocate(router_logits)
        return expert_indices, expert_weights

    def _dense_route(self, logits):
        """[32, E] BF16 logits tile -> the dense routing weights (dense_route.cpp; K_SEL = top_k, every logit tied with
        the top_k-th routed, as dense_decode's ge(logits, kth))."""
        from .fused_decode.boundary import KERNEL_DIR

        mesh = logits.device()
        gx = mesh.compute_with_storage_grid_size().x
        cores = ttnn.num_cores_to_corerangeset(ttnn.TILE_SIZE, mesh.compute_with_storage_grid_size(), row_wise=True)
        w = ttnn.empty(list(logits.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                       memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        ct = [0, gx, self.top_k, self.num_experts]  # (up to every expert routed: ge(logits, kth) keeps every tie)
        for t in (logits, w):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dense_route.cpp"), core_ranges=cores, compile_time_args=ct,
            runtime_args=[], common_runtime_args=[logits.buffer_address(), w.buffer_address()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )  # fmt: skip
        cb = ttnn.CBDescriptor(total_size=128, core_ranges=cores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=128)])  # fmt: skip
        ttnn.generic_op([logits, w], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))
        return w

    def dense_decode(self, hidden_states):
        """Batched decode routing (one 32-token tile per device, fused_decode.config.dense_moe_decode_supported).

        hidden_states: [1, 1, 32, hidden] L1 interleaved. Returns the dense [32, num_experts] routing weights (each
        token's softmax over its top-k logits, zeros elsewhere) that experts.decode_forward_dense consumes.

        The logits matmul runs on one core with the whole K in one block (in0_block_w = K tiles): the same tile
        accumulation order as the auto-configured ttnn.linear, so the logits are bitwise identical, in ~8 us instead of
        ~62 us, with the bias fused. The routing weights are built from a top-k threshold mask instead of
        softmax + scatter (whose row-major index round trip is most of the original routing time)."""
        k_tiles = self.hidden_dim // ttnn.TILE_SIZE
        tokens = hidden_states.shape[-2]
        if not hasattr(self, "_dense_logits_program_config"):
            self._dense_logits_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
                in0_block_w=k_tiles,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=-(-self.num_experts // ttnn.TILE_SIZE),
                per_core_M=-(-tokens // ttnn.TILE_SIZE),
                per_core_N=-(-self.num_experts // ttnn.TILE_SIZE),
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            # The compute config ttnn.linear picks without a program config (BFP8 x BF16: HiFi2, no FP32 accumulation,
            # packer L1 accumulation); an explicit program config would otherwise default to LoFi.
            self._dense_logits_compute_config = ttnn.init_device_compute_kernel_config(
                hidden_states.device().arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        logits = ttnn.linear(
            hidden_states,
            self.weight,
            bias=self.bias,
            program_config=self._dense_logits_program_config,
            compute_kernel_config=self._dense_logits_compute_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        if tokens == ttnn.TILE_SIZE and self.num_experts == ttnn.TILE_SIZE:
            # one program, one core per token (fused_decode/kernels/dense_route.cpp): the same selection, the softmax
            # in fixed point on the RISC-V cores (as the DFlash verify's route program)
            weights = self._dense_route(logits)
            logits.deallocate(True)
            return ttnn.reshape(weights, (tokens, self.num_experts))
        values, indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        indices.deallocate(True)
        # Reductions, not column slices: a slice inside a tile becomes untilize + slice + tilize.
        kth = ttnn.min(values, dim=-1, keepdim=True)
        values.deallocate(True)
        top = ttnn.max(logits, dim=-1, keepdim=True)
        selected = ttnn.ge(logits, kth)
        kth.deallocate(True)
        e = ttnn.exp(ttnn.subtract(logits, top, output_tensor=logits), output_tensor=logits)
        top.deallocate(True)
        e = ttnn.multiply(e, selected, output_tensor=e)
        selected.deallocate(True)
        total = ttnn.sum(e, dim=-1, keepdim=True)
        weights = ttnn.divide(e, total)
        total.deallocate(True)
        e.deallocate(True)
        return ttnn.reshape(weights, (tokens, self.num_experts))

    def decode_indexed(self, hidden_states):
        """Fused decode routing for one token.

        hidden_states: the flat norm output of the layer boundary (fused_decode/boundary.py).
        Returns persistent [1, 32] UINT16 expert ids and BF16 softmax weights holding the top-k in their first k
        entries (shared by every layer; not to be deallocated)."""
        router = self.ccl_manager.get_decode_linear_stream(
            "router",
            self.hidden_dim // ttnn.TILE_SIZE,
            self.num_experts,
            ROUTER_DECODE_WEIGHT_DTYPE,
            readers=1,
            out_mode=4,
            out_page_bytes=64,
            heads=(self.top_k, self.num_experts),
        )
        return router(hidden_states, self.weight_stream, self.ccl_manager.get_decode_router_out())

    def _fused_call(self, hidden_states, use_throughput_experts):
        """Forward pass using fused matmul+topk+softmax kernel.

        Note: Fused op only supports throughput experts (sparse [B,k] output).
        """
        if not use_throughput_experts:
            raise ValueError("Fused topk_router_gpt requires use_throughput_experts=True")

        # Typecast to bf16 if needed (fused op requires bf16 input)
        needs_typecast = hidden_states.dtype != ttnn.bfloat16
        if needs_typecast:
            hidden_states_bf16 = ttnn.typecast(hidden_states, dtype=ttnn.bfloat16)
        else:
            hidden_states_bf16 = hidden_states

        B = hidden_states_bf16.shape[0]
        device = hidden_states_bf16.device()

        # Initialize fused op buffers (once)
        self._init_fused_op(device, B)

        # Run fused matmul + topk + softmax
        indices_rm, weights_rm = ttnn.experimental.topk_router_gpt(
            hidden_states_bf16,
            weight_tensor=self.weight,
            bias_tensor=self._fused_bias,
            k=self.top_k,
            num_experts=self.num_experts,
        )

        if needs_typecast:
            ttnn.deallocate(hidden_states_bf16)

        # The op already reports uint16 / bf16 RM [B, top_k] (32 physical rows), so return its outputs as they are;
        # fused_decode.py handles RM input natively (zero-cost reshape to 4D). Do not slice + deallocate here: a slice
        # that covers the whole logical shape is a no-op that aliases its input, and ttnn.deallocate (force=True)
        # would then free the tensors returned to the caller.
        return indices_rm, weights_rm
