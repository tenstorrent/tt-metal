# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
LM Head Sampling: CCL Broadcast + Mcast + Matmul fused operation for LM head vocab projection.

In multi-device mode, CCL broadcasts input_tensor from the sender device to all devices,
then on each device multicasts from the sender core to all cores in the device grid,
then each matmul core computes a local matmul with its vocab weight shard.

In single-device mode (skip_ccl=True), the CCL broadcast is skipped and the
input_tensor is used directly.

- input_tensor (in0): [1, K] on sender core
- vocab_tensor (in1): [K, N_total] width-sharded across matmul cores as [K, N_per_core]
- output: [1, N_total] width-sharded across matmul cores as [1, N_per_core]

CB Layout:
- CB 0:  mcast_src (input_tensor on sender core, tensor-backed)
- CB 1:  mcast_dst / matmul_in0 (all device grid cores, intermediate)
- CB 2:  matmul_in1 (vocab weights on matmul cores, tensor-backed)
- CB 16: matmul_out (output on matmul cores, tensor-backed)
- CB 30: bcast_pkt (CCL broadcast packet buffer, only in multi-device mode)
"""


import math

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.moe_routed_expert.op import MESH_LEAF, MESH_ROOT1, MESH_ROOT2, MESH_ROOT3
from models.demos.deepseek_v3_b1.metadata.metadata import METADATA_TENSOR_BYTES
from models.demos.deepseek_v3_b1.micro_ops.ccl_broadcast.op import DeepseekMinimalBroadcast
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import get_max_page_size_and_num_pages
from models.demos.deepseek_v3_b1.micro_ops.reduce_to_one_b1.op import get_device_role as get_reduce_device_role
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import (
    float_to_bfloat16_packed,
    float_to_uint32,
    get_pinned_optimal_dram_bank_to_logical_worker_assignment,
    merge_kernel_defines,
)


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _is_singleton_prefix_shape(shape, expected_last_dim: int) -> bool:
    dims = tuple(int(d) for d in shape)
    if not dims or dims[-1] != expected_last_dim:
        return False
    return all(d == 1 for d in dims[:-1])


# SpecLMHeadStage pads the broadcast/intermediate tensor to (M, K + N) BF16 columns; the last N columns
# hold token metadata (see demo/stage.py). RMSNorm + gamma still apply only to the first K activations.
# If we use input_shape[-1] for RMS tile picking, (K+32)//32 can make (num_cols % 32) != 0 and incorrectly
# select the 16x32 RMS path while rms_num_tiles stays 7 (seven 32x32 tiles = K elements) — wrong norm.
_SPEC_VERIFY_METADATA_BF16_COLS = 32
ACTIVATION_DIM = 7168


class LMHeadSampling:
    """
    LM head sampling vocab projection: CCL broadcast + mcast + matmul via ttnn.generic_op.

    In multi-device mode, CCL broadcasts input_tensor from the sender device to all
    devices, then on each device the sender core multicasts to all device cores, and
    each matmul core computes [1, K] x [K, N_per_core] with its width-sharded vocab
    weight shard.
    """

    @staticmethod
    def golden(
        input_tensor,
        gamma_tensor,
        vocab_tensor,
        indices: torch.Tensor | None = None,
        k: int = 1,
        temperature: float = 1.0,
        p: float = 1.0,
        epsilon: float = 1e-6,
        fuse_mtp: bool = False,
        fuse_mtp_verification: bool = False,
        reference_token: torch.Tensor | None = None,
        embedding_tensor: torch.Tensor | None = None,
        h_gamma_tensor: torch.Tensor | None = None,
        e_gamma_tensor: torch.Tensor | None = None,
        eh_projection_tensor: torch.Tensor | None = None,
    ):
        """
        PyTorch reference implementation for fused LM-head + sampling golden.

        This function performs the following operations:
        1. RMSNorm on input_tensor (hidden states) using gamma_tensor
        2. Matmul with vocab_tensor to get logits/scores
        3. Top-k filtering, temperature-scaled softmax, top-p filtering,
           then random categorical sampling (falls back to argmax when k=1)

        When fuse_mtp=True, additionally performs MTP (Multi-Token Prediction) fusion:
        4. Look up token embedding from embedding_tensor using sampled index
        5. RMSNorm on hidden states using h_gamma_tensor
        6. RMSNorm on token embedding using e_gamma_tensor
        7. Concatenate normalized embedding and hidden states
        8. Project through eh_projection_tensor to get MTP input

        When fuse_mtp_verification=True, performs speculative decoding verification:
        4v. Compare sampled token (T_spec) with reference_token (T_base)
        5v. Return verification result (match=1, no_match=0)

        fuse_mtp and fuse_mtp_verification are mutually exclusive post-argmax paths.

        Args:
            input_tensor: Input hidden states tensor (torch.Tensor) [1, hidden_dim]
            gamma_tensor: RMSNorm gamma/weight tensor for LM head (torch.Tensor) [hidden_dim]
            vocab_tensor: Vocab projection tensor (torch.Tensor) [hidden_dim, vocab_size]
            indices: Optional indices tensor used by fused sampling. If provided,
                golden returns sampled index tensor [1, 1]. If omitted, returns scores.
            k: Top-k sampling parameter. k=1 is greedy argmax.
            temperature: Sampling temperature applied as logits / temperature before
                softmax.  Only used when k > 1.  Must be > 0.
            p: Top-p (nucleus) threshold.  After softmax the cumulative probability
                is cut at p and the remaining mass is renormalised.
            epsilon: Small value to avoid division by zero in RMS norm.
            fuse_mtp: If True, perform MTP fusion after sampling.
            fuse_mtp_verification: If True, perform verification against reference_token.
            reference_token: Reference token (T_base) for verification [1, 1] uint32.
            embedding_tensor: Token embedding table (torch.Tensor) [vocab_size, embedding_dim]
            h_gamma_tensor: RMSNorm gamma for hidden states in MTP (torch.Tensor) [hidden_dim]
            e_gamma_tensor: RMSNorm gamma for embeddings in MTP (torch.Tensor) [embedding_dim]
            eh_projection_tensor: Projection matrix for concatenated [e, h] (torch.Tensor) [embedding_dim + hidden_dim, output_dim]

        Returns:
            - If neither fuse_mtp nor fuse_mtp_verification: (sampled_index [1,1], None)
            - If fuse_mtp: (sampled_index [1,1], mtp_projection_output [1, output_dim])
            - If fuse_mtp_verification: (sampled_index [1,1], verification_result uint32 (1=match, 0=no_match))
        """
        assert not (fuse_mtp and fuse_mtp_verification), "fuse_mtp and fuse_mtp_verification are mutually exclusive"

        if fuse_mtp:
            assert embedding_tensor is not None, "embedding_tensor is required for fused MTP"
            assert eh_projection_tensor is not None, "eh_projection_tensor is required for fused MTP"

        if fuse_mtp_verification:
            assert reference_token is not None, "reference_token is required for MTP verification"

        # Step 1: RMSNorm on input hidden states for LM head
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        normalized = input_tensor * torch.rsqrt(variance + epsilon)
        rmsnorm_out = normalized * gamma_tensor if gamma_tensor is not None else normalized

        # Step 2: Matmul with vocab tensor to get logits
        scores = rmsnorm_out @ vocab_tensor

        scores_f32 = scores.float().reshape(-1)
        indices_i64 = indices.to(torch.int64).reshape(-1)
        if scores_f32.numel() != indices_i64.numel():
            raise ValueError(
                f"scores and indices must have the same number of elements, got {scores_f32.numel()} and {indices_i64.numel()}"
            )

        if k == 1:
            # Step 3a: Greedy argmax — temperature is irrelevant for k=1.
            max_score = torch.max(scores_f32)
            tied_mask = scores_f32 == max_score
            selected_index = torch.min(indices_i64[tied_mask]).to(torch.uint32)
        else:
            # Step 3b: Top-k → temperature-scaled softmax → top-p → categorical.
            # Mirrors the device kernel flow in sampling.hpp.
            assert temperature > 0, f"temperature must be > 0, got {temperature}"
            topk_scores, topk_positions = torch.topk(scores_f32, k, largest=True, sorted=True)
            topk_indices = indices_i64[topk_positions]

            scaled = topk_scores / temperature
            probs = torch.softmax(scaled, dim=-1)

            # Top-p filtering: keep smallest set whose cumulative mass exceeds p.
            if p < 1.0:
                cum_probs = torch.cumsum(probs, dim=-1)
                cutoff_mask = cum_probs > p
                # Always keep at least one token.
                cutoff_mask[0] = False
                probs[cutoff_mask] = 0.0
                probs = probs / probs.sum()

            selected_pos = torch.multinomial(probs, num_samples=1).item()
            selected_index = topk_indices[selected_pos].to(torch.uint32)

        if fuse_mtp_verification:
            spec_token = selected_index.item()
            ref_token = reference_token.reshape(-1)[0].item()
            match = torch.tensor(1 if spec_token == ref_token else 0, dtype=torch.uint32).reshape(1, 1)
            return selected_index.reshape(1, 1), match

        if not fuse_mtp:
            return selected_index.reshape(1, 1), None

        # Step 4: Look up token embedding from embedding table
        token_id = selected_index.to(torch.int64).item()
        token_embedding = embedding_tensor[token_id, :].unsqueeze(0)  # [1, embedding_dim]

        # Step 5: RMSNorm on hidden states using h_gamma_tensor (skip gamma if folded)
        h_variance = input_tensor.pow(2).mean(-1, keepdim=True)
        h_normalized = input_tensor * torch.rsqrt(h_variance + epsilon)
        h_rmsnorm_out = h_normalized * h_gamma_tensor if h_gamma_tensor is not None else h_normalized

        # Step 6: RMSNorm on token embedding using e_gamma_tensor (skip gamma if folded)
        e_variance = token_embedding.pow(2).mean(-1, keepdim=True)
        e_normalized = token_embedding * torch.rsqrt(e_variance + epsilon)
        e_rmsnorm_out = e_normalized * e_gamma_tensor if e_gamma_tensor is not None else e_normalized

        # Step 7: Concatenate normalized embedding then hidden states.
        # The MTP checkpoint projection expects the [e_norm | h_norm] layout.
        concat_he = torch.cat([e_rmsnorm_out, h_rmsnorm_out], dim=-1)

        # Step 8: Project through eh_projection_tensor
        mtp_output = concat_he @ eh_projection_tensor
        return selected_index.reshape(1, 1), mtp_output

    @staticmethod
    def _build_reduce_brisc_per_core_args(enable, rp, device_role, device, device_idx):
        if not enable:
            return []
        role_name = {MESH_LEAF: "LEAF", MESH_ROOT3: "ROOT3", MESH_ROOT2: "ROOT2", MESH_ROOT1: "ROOT1"}
        fused_base = rp["fused_per_device"][device_idx].buffer_address()
        intermediate_base = fused_base + rp["cb20_offset_bytes"]
        payload = rp["payload_size_bytes"]
        if device_role == MESH_LEAF:
            dst_l1, dst_sem = intermediate_base, rp["sem_addrs"][0]
        elif device_role == MESH_ROOT3:
            dst_l1, dst_sem = intermediate_base + payload, rp["sem_addrs"][1]
        elif device_role == MESH_ROOT2:
            dst_l1, dst_sem = intermediate_base + 2 * payload, rp["sem_addrs"][2]
        else:
            dst_l1, dst_sem = 0, rp["sem_addrs"][3]
        out_addr = fused_base
        args = []
        for core in rp["worker_cores"]:
            fc = rp["col_to_fabric"][core.x]
            fc_phys = device.worker_core_from_logical_core(fc)
            slot = rp["core_to_slot"][(core.x, core.y)]
            shard = rp["core_to_shard"][(core.x, core.y)]
            args.append(
                (
                    core,
                    [
                        int(fc_phys.x),
                        int(fc_phys.y),
                        slot,
                        rp["wf_ready_sem_addr"],
                        int(dst_l1),
                        int(dst_sem),
                        int(out_addr),
                        shard,
                    ],
                )
            )
        for fc in rp["fabric_cores"]:
            args.append((fc, [rp["wf_ready_sem_addr"]]))
        return args

    @staticmethod
    def op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        gamma_tensor,
        vocab_tensor,
        output_tensor,
        sender_coord,
        eh_mm_fused_buffer=None,
        embedding_tensor=None,
        h_gamma_tensor=None,
        e_gamma_tensor=None,
        eh_projection_tensor=None,
        indices_tensor=None,
        output_index_tensor=None,
        argmax_final_core_coord=None,
        argmax_final_mesh_coord=None,
        global_semaphore=None,
        global_stage2_semaphore=None,
        scores_scratch_tensor=None,
        indices_scratch_tensor=None,
        bcast_semaphores=None,
        bcast_num_links=1,
        seed=2005,
        fp32_dest_acc_en=False,
        epsilon=1e-6,
        rsqrt_fast_approx=False,
        skip_ccl=None,
        socket_input=None,
        socket_output=None,
        persistent_mode=False,
        termination_semaphore=None,
        persistent_next_iter_semaphore=None,
        *,
        fabric_config=None,
        broadcast_topology_override=None,
        is_mtp_base_stage=False,
        is_mtp_verify_stage=False,
        metadata_tensor=None,
        eh_subblock_k=None,
        reduce_semaphores=None,
        mtp_bcast_semaphores=None,
        base_token_buffer=None,
        k=32,
    ):
        logger.debug(f"broadcast sender_coord={sender_coord}")
        """
        Execute LM head sampling CCL broadcast + mcast + matmul operation using generic_op.

        In multi-device mode, CCL broadcasts input_tensor from the sender device to all
        devices via the fabric, then on each device the sender core multicasts to all
        device cores, and each matmul core computes a local matmul with its vocab weight
        shard.

        Args:
            input_tensor_mesh: Input tensor mesh [1, K] height-sharded in L1 on a single sender core
            intermediate_tensor_mesh: Intermediate mesh tensor for CCL broadcast destination
            gamma_tensor: RMSNorm gamma tensor [1, K], same tile/layout as input
            output_mtp_tensor: Pre-allocated output for MTP [1, output_dim] width-sharded across matmul cores
            embedding_tensor: Token embedding table [vocab_size, embedding_dim]
            h_gamma_tensor: RMSNorm gamma tensor for hidden states [hidden_dim]
            e_gamma_tensor: RMSNorm gamma tensor for embeddings [embedding_dim]
            eh_projection_tensor: Projection matrix [hidden_dim + embedding_dim, output_dim]
            vocab_tensor: Vocab weights [K, N_total] width-sharded across matmul cores as [K, N_per_core]
            output_tensor: Pre-allocated output [1, N_total] width-sharded across matmul cores
            indices_tensor: Optional pre-cached global indices tensor, width-sharded like output scores
            output_index_tensor: Optional pre-allocated [1, 1] uint32 tensor for fused argmax output
            argmax_final_core_coord: Optional final core for fused argmax reduction (defaults to first matmul core)
            sender_coord: Tuple (row, col) of sender device in mesh
            bcast_semaphores: Per-link global semaphores for neighbor-exchange CCL broadcast.
                Must contain exactly `bcast_num_links` entries in CCL mode.
            bcast_num_links: Number of fabric links for CCL broadcast
            fp32_dest_acc_en: Whether to enable FP32 accumulation
            skip_ccl: Whether to skip CCL broadcast. If None, defaults to True for single-device meshes.
            socket_input: Optional socket input endpoint. Supports ttnn.MeshSocket receiver endpoint (D2D input).
            socket_output: Optional socket output endpoint. Supports ttnn.D2HSocket (host output) and
                ttnn.MeshSocket sender endpoint (D2D output).
            persistent_mode: Enable persistent execution loop in kernel.
            termination_semaphore: Global semaphore used to terminate persistent loop (terminate when set to 1).
            persistent_next_iter_semaphore: Global semaphore used to gate next persistent iteration.
        Returns:
            Output tensor with matmul result. If fused argmax is enabled, output_index_tensor is written in-place.
        """

        logger.debug(
            f"[OP] entered mtp_base={is_mtp_base_stage} mtp_verify={is_mtp_verify_stage} persistent={persistent_mode}",
        )
        # LMHeadSampling is always fused with k=1 sampling (argmax fast path).
        enable_argmax = True
        fold_rmsnorm = gamma_tensor is None
        is_mtp_base_stage = (
            is_mtp_base_stage
            and eh_mm_fused_buffer is not None
            and embedding_tensor is not None
            and eh_projection_tensor is not None
        )
        # Reduce is always on for MTP (ReduceToOneB1 replaces the old single-device Gather)
        enable_reduce_to_one = is_mtp_base_stage
        # MTP Verify stage is enabled if the verification tensors are provided
        is_mtp_verify_stage = is_mtp_verify_stage and metadata_tensor is not None
        assert not (
            is_mtp_base_stage and is_mtp_verify_stage
        ), "is_mtp_base_stage and is_mtp_verify_stage are mutually exclusive"
        # Socket output for MTP logits (exit device only); used for has_mtp_logits_socket_on_device in device loop.
        socket_mode_none = 0
        socket_mode_d2h = 1
        socket_mode_d2d = 2
        socket_page_size_bytes = METADATA_TENSOR_BYTES
        input_socket_mode_none = 0
        input_socket_mode_d2d = 2

        if socket_input is None:
            input_socket_mode_selected = input_socket_mode_none
        elif isinstance(socket_input, ttnn.MeshSocket):
            input_socket_mode_selected = input_socket_mode_d2d
        else:
            raise TypeError(
                f"Unsupported socket_input type for lm_head_sampling: {type(socket_input)}. "
                "Expected ttnn.MeshSocket."
            )

        if input_socket_mode_selected == 1:
            raise AssertionError("lm_head_sampling input socket mode=1 is invalid (reserved for output d2h mode)")
        enable_socket_input = socket_input is not None
        if socket_output is None:
            socket_mode_selected = socket_mode_none
        elif isinstance(socket_output, ttnn.D2HSocket):
            socket_mode_selected = socket_mode_d2h
        elif isinstance(socket_output, ttnn.MeshSocket):
            socket_mode_selected = socket_mode_d2d
        else:
            raise TypeError(
                f"Unsupported socket_output type for lm_head_sampling: {type(socket_output)}. "
                "Expected ttnn.D2HSocket or ttnn.MeshSocket."
            )

        enable_socket_output = socket_output is not None
        if indices_tensor is None or output_index_tensor is None:
            raise ValueError("indices_tensor and output_index_tensor are required for fused LM-head + sampling")
        # Get mesh/device info
        mesh_device = input_tensor_mesh.device()

        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]
        sender_row = int(sender_coord[0])
        sender_col = int(sender_coord[1])
        if skip_ccl is None:
            skip_ccl = mesh_rows * mesh_cols == 1
        if enable_socket_input:
            active_input_socket_cores = socket_input.get_active_cores()
            if len(active_input_socket_cores) != 1:
                raise ValueError("socket input for lm_head_sampling must have exactly one active core")
            input_socket_core = active_input_socket_cores[0]
        if enable_socket_output:
            # only for d2h sockets
            if isinstance(socket_output, ttnn.D2HSocket):
                socket_output.set_page_size(socket_page_size_bytes)
            active_socket_cores = socket_output.get_active_cores()
            if len(active_socket_cores) != 1:
                raise ValueError("socket output for lm_head_sampling must have exactly one active core")
            socket_core = active_socket_cores[0]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        intermediate_tensors_per_device = ttnn.get_device_tensors(intermediate_tensor_mesh)
        gamma_tensors_per_device = ttnn.get_device_tensors(gamma_tensor) if gamma_tensor is not None else None
        vocab_tensors_per_device = ttnn.get_device_tensors(vocab_tensor)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        indices_tensors_per_device = ttnn.get_device_tensors(indices_tensor) if enable_argmax else None
        output_index_tensors_per_device = ttnn.get_device_tensors(output_index_tensor) if enable_argmax else None
        eh_mm_fused_per_device = ttnn.get_device_tensors(eh_mm_fused_buffer) if is_mtp_base_stage else None
        embedding_tensors_per_device = ttnn.get_device_tensors(embedding_tensor) if is_mtp_base_stage else None
        h_gamma_tensors_per_device = (
            ttnn.get_device_tensors(h_gamma_tensor) if is_mtp_base_stage and h_gamma_tensor is not None else None
        )
        e_gamma_tensors_per_device = (
            ttnn.get_device_tensors(e_gamma_tensor) if is_mtp_base_stage and e_gamma_tensor is not None else None
        )
        eh_proj_tensors_per_device = ttnn.get_device_tensors(eh_projection_tensor) if is_mtp_base_stage else None
        base_token_tensors_per_device = (
            ttnn.get_device_tensors(base_token_buffer)
            if (is_mtp_base_stage and base_token_buffer is not None)
            else None
        )
        # [Sampling] Per-device scratch tensors for mesh stage-1/stage-2 gather.
        # scores_scratch_tensor: bf16-logical scratch (height-sharded on argmax_final_core_grid)
        # indices_scratch_tensor: uint32 scratch (height-sharded on argmax_final_core_grid)
        # These replace the single `fabric_scratch_tensor` that argmax used.
        scores_scratch_tensors_per_device = (
            ttnn.get_device_tensors(scores_scratch_tensor)
            if (enable_argmax and not skip_ccl and scores_scratch_tensor is not None)
            else None
        )
        indices_scratch_tensors_per_device = (
            ttnn.get_device_tensors(indices_scratch_tensor)
            if (enable_argmax and not skip_ccl and indices_scratch_tensor is not None)
            else None
        )
        if enable_argmax and not skip_ccl:
            if (
                global_semaphore is None
                or global_stage2_semaphore is None
                or scores_scratch_tensor is None
                or indices_scratch_tensor is None
            ):
                raise ValueError(
                    "global_semaphore, global_stage2_semaphore, scores_scratch_tensor, and "
                    "indices_scratch_tensor are required for mesh sampling"
                )
            if mesh_rows < 2 or mesh_cols != 2:
                raise NotImplementedError(
                    f"Fused LM-head mesh argmax currently supports only (R,2) with R>=2, got {mesh_shape}"
                )
            if argmax_final_mesh_coord is None:
                raise ValueError("argmax_final_mesh_coord is required for mesh argmax")
        if bcast_semaphores is None:
            bcast_semaphores = []
        else:
            bcast_semaphores = list(bcast_semaphores)
        if not skip_ccl and len(bcast_semaphores) != bcast_num_links:
            raise ValueError(f"Expected exactly {bcast_num_links} broadcast semaphore(s), got {len(bcast_semaphores)}")
        if persistent_mode and persistent_next_iter_semaphore is None:
            raise ValueError(
                "persistent_next_iter_semaphore is required when persistent_mode=True "
                "(must be a global semaphore on the full device grid)"
            )
        if persistent_mode and termination_semaphore is None:
            raise ValueError(
                "termination_semaphore is required when persistent_mode=True "
                "(must be a global semaphore on the full device grid)"
            )
        global_sem_addr = (
            int(ttnn.get_global_semaphore_address(global_semaphore)) if (enable_argmax and not skip_ccl) else 0
        )
        global_stage2_sem_addr = (
            int(ttnn.get_global_semaphore_address(global_stage2_semaphore)) if (enable_argmax and not skip_ccl) else 0
        )
        persistent_next_iter_global_sem_addr = (
            int(ttnn.get_global_semaphore_address(persistent_next_iter_semaphore)) if persistent_mode else 0
        )
        termination_global_sem_addr = (
            int(ttnn.get_global_semaphore_address(termination_semaphore)) if persistent_mode else 0
        )
        # Calculate packet size and page info for CCL broadcast

        # Get tile info from input tensor (use a sample device tensor)
        input_tensor_sample = input_tensors_per_device[0]
        in0_tile = input_tensor_sample.get_tile()
        input_shape = input_tensor_sample.shape
        data_format = input_tensor_sample.dtype
        numel = 7168
        scalar_packed = float_to_uint32(1.0 / math.sqrt(float(numel)))
        epsilon_packed = float_to_uint32(epsilon)

        # CCL broadcast page info
        # Matmul shape info from input and vocab tensors
        num_tiles_k = 224
        # RMS tile geometry follows activation width K, not padded (K + metadata) broadcast width.
        activation_cols_for_rms = int(input_shape[1])
        if is_mtp_verify_stage:
            activation_cols_for_rms -= _SPEC_VERIFY_METADATA_BF16_COLS
        # RMSNorm in this path must match broadcast_rms tile/page interpretation.
        rms_interpreted_tile = ttnn.Tile((32, 32))
        rms_tile_height, rms_tile_width = rms_interpreted_tile.tile_shape
        rms_tile_size = rms_interpreted_tile.get_tile_size(data_format)
        rms_num_tiles = 7  # (input_shape[0] * input_shape[1]) // (rms_tile_height * rms_tile_width)
        # Get output tile info
        output_tensor_sample = output_tensors_per_device[0]
        out_tile = output_tensor_sample.get_tile()
        # Get vocab weights info (per-core output width)
        vocab_tensor_sample = vocab_tensors_per_device[0]
        weights_shard_spec = vocab_tensor_sample.memory_config().shard_spec
        n_per_core = weights_shard_spec.shape[1]
        out_w_per_core = n_per_core // out_tile.tile_shape[1]

        # Input tile size for mcast data (must be defined before MTP block uses it)
        # Everywhere that uses input_tile_size is wrong
        input_tile_size = in0_tile.get_tile_size(data_format)
        mcast_data_size_bytes = num_tiles_k * input_tile_size

        # [MTP base stage] Get eh projection matmul info (per-core output width)
        if is_mtp_base_stage:
            eh_projection_tensor_sample = eh_proj_tensors_per_device[0]
            eh_weights_shard_spec = eh_projection_tensor_sample.memory_config().shard_spec
            eh_n_per_core = eh_weights_shard_spec.shape[1]
            eh_out_w_per_core = eh_n_per_core // eh_projection_tensor_sample.get_tile().tile_shape[1]
            # Embedding CB size (single row from DRAM embedding table)
            embedding_tensor_sample = embedding_tensors_per_device[0]
            embedding_shape = embedding_tensor_sample.shape
            embedding_dim = int(embedding_shape[-1])
            e_num_tiles = rms_num_tiles  # embedding_dim // rms_tile_width  # tiles for e_rmsnorm

            # EH matmul k dimension: concat [h_norm | e_norm] has shape [1, hidden_dim + embedding_dim]
            # eh_concat_rms_tiles: packed (32x32) tile count for concat/mcast source CB
            # eh_num_tiles_k: standard tile count for the matmul K dimension (must match weight tile-rows)
            eh_concat_rms_tiles = 2  # device K dim by number of devices in stage (8)
            eh_num_tiles_k = (2 * embedding_dim // 8) // in0_tile.tile_shape[
                1
            ]  # device K dim by number of devices in stage (8)
            eh_mcast_data_size_bytes = eh_num_tiles_k * input_tile_size  # per-device K slice: 56 * 64 = 3584 bytes

            # EH matmul DRAM streaming parameters
            eh_projection_tensor_sample = eh_proj_tensors_per_device[0]
            eh_dtype = eh_projection_tensor_sample.dtype
            eh_proj_tile = eh_projection_tensor_sample.get_tile()
            eh_proj_tile_size = eh_proj_tile.get_tile_size(eh_dtype)
            eh_subblock_k = (ACTIVATION_DIM // 8) // eh_proj_tile.tile_shape[1]
            eh_num_subblocks_k = 2
            eh_out_num_tiles = eh_out_w_per_core

            # Compute subblock_w: max dest tiles that evenly divide per_core_n
            # With fp32_dest_acc_en: max_dest=4 (half sync), else max_dest=8
            _max_dest = 4 if fp32_dest_acc_en else 8
            _max_subblock_w = min(_max_dest, eh_out_w_per_core)
            eh_subblock_w = _max_subblock_w
            while eh_subblock_w > 1 and eh_out_w_per_core % eh_subblock_w != 0:
                eh_subblock_w -= 1
            sample_device = eh_projection_tensor_sample.device()
            eh_page_size, eh_num_pages = get_max_page_size_and_num_pages(
                sample_device, eh_subblock_k, eh_proj_tile_size
            )
            eh_in1_block_size_bytes = eh_subblock_k * eh_proj_tile_size
            eh_num_in1_buffers = 3  # Triple buffering
            eh_in1_CB_tiles = eh_subblock_k * eh_num_in1_buffers
            eh_in1_CB_size = eh_in1_CB_tiles * eh_proj_tile_size
        else:
            eh_out_w_per_core = 0
            embedding_dim = 0
            e_num_tiles = 0
            eh_num_tiles_k = 0
            eh_mcast_data_size_bytes = 0

        # ====================================================================
        # CB indices
        # ====================================================================
        rmsnorm_input_cb = 0  # raw input on sender core (tensor-backed)
        mcast_dst_cb = 1  # Mcast destination = matmul in0 (all mcast grid cores, intermediate)
        matmul_in1_cb = 2  # vocab_tensor weights on matmul cores (tensor-backed)
        rmsnorm_gamma_cb = 7  # RMSNorm gamma weights on sender core (tensor-backed)
        mcast_src_cb = 8  # RMSNorm output on sender core (intermediate), consumed by mcast sender
        matmul_eh_cb = 9  # [MTP] EH projection weights on matmul cores (tensor-backed)
        embedding_cb = 10  # [MTP] Embedding CB — mcast consumes it before embedding DRAM read
        h_gamma_cb = 11  # [MTP] RMSNorm gamma weights for hidden states on sender core (tensor-backed)
        e_gamma_cb = 12  # [MTP] RMSNorm gamma weights for embeddings on sender core (tensor-backed)
        mcast_eh_src_cb = 15  # [MTP] Fused [h_norm|e_norm] on sender core, both RMSNorms write here directly
        # ----------------------------------------------------------------------
        # Sampling (top-K + top-P + softmax) CB indices — replace the 4 legacy
        # argmax CBs (winner/gather/indices/socket) with the full CB layout that
        # models/demos/deepseek_v3_b1/micro_ops/sampling/op.py :: _op_mesh_topk
        # uses. IDs are chosen to avoid conflicts with existing LM-head CBs
        # (taken: 0,1,2,7-13,15-23,30) and leave headroom for future CBs.
        #
        # Per-core CBs (live on every matmul/argmax_core_grid core):
        sampling_winner_cb = 3  # local top-K winners (scores + indices, packed)
        sampling_topk_in_scores_cb = 4  # LLK topk input scores (phase1 local + phase2 gather)
        sampling_topk_in_indices_cb = 5  # LLK topk input indices (phase1 local + phase2 gather)
        sampling_topk_out_scores_cb = 6  # LLK topk output scores (1 tile)
        sampling_topk_out_indices_cb = 14  # LLK topk output indices (1 tile)
        # Final-core-only compute CBs (softmax / top-P / RNG — 1 bf16 tile each):
        sampling_softmax_in_cb = 24
        sampling_softmax_out_cb = 25
        sampling_softmax_exp_cb = 26
        sampling_softmax_sub_cb = 27
        sampling_max_cb = 28
        sampling_sum_cb = 29
        sampling_scaler_cb = 31
        sampling_temp_cb = 32
        sampling_rand_cb = 33
        # Mesh-stage scratch CBs (live on stage-1 / stage-2 receiver = final core):
        sampling_mesh_stage_scores_cb = 34
        sampling_mesh_stage_indices_cb = 35
        # Deferred-socket output CB (final core only):
        sampling_socket_cb = 36
        matmul_out_cb = 16  # Matmul output on matmul cores (tensor-backed)
        matmul_out_eh_cb = 17  # [MTP] EH matmul output on matmul cores (tensor-backed)
        mcast_eh_dst_cb = 18  # [MTP] Second mcast destination
        eh_gather_dst_cb = 19  # [MTP] EH output gather destination on argmax_final_core
        # [Reduce] CB indices — reuse existing CBs where possible to save L1
        reduce_local_cb = 21  # Aliased to CB17's L1 but with payload_size page_size for reduce
        reduce_received_cb = 20  # tensor-backed by reduce_intermediate_tensor
        reduce_scratch_cb = 23  # Dedicated scratch for reduce (CANNOT reuse CB18: BRISC races TRISC's DSMM)
        reduce_output_cb = reduce_scratch_cb  # alias; compute packs to scratch, writer reads scratch
        reduce_packet_cb = 22  # new: fabric send staging buffer

        # CB indices for CCL broadcast (use separate CBs to avoid conflicts)
        bcast_pkt_cb = 30  # Packet buffer for CCL broadcast
        mtp_bcast_pkt_cb = 13  # [MTP] Packet buffer for MTP token broadcast

        # Sync CB for h_rmsnorm and lm head norm on TRISC
        hnorm_ready_cb = 37
        # [MTP] NCRISC -> BRISC sync CB on input_core: NCRISC pushes after
        # `token_bcast_receiver` returns, BRISC waits before issuing `mcast_eh`.
        # The token broadcast traces back to argmax_final_core's sampling
        # completion, which in turn requires every matmul core's TRISC matmul
        # to have popped CB 1.  Gating BRISC's `mcast_eh` on this push therefore
        # guarantees that the second mcast does not write into receivers' L1
        # while their CB 1 (= same backing region as CB 18 today) is still
        # being read by matmul.  Sender-only, sized to the L1 minimum.
        mcast_eh_ready_cb = 38

        # ====================================================================
        # Semaphore IDs (for intra-device mcast)
        # ====================================================================
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1
        argmax_receiver_semaphore_id = 2
        argmax_local_ready_semaphore_id = 3
        fabric_gate_bcast_turn_semaphore_id = 4
        fabric_gate_argmax_turn_semaphore_id = 5
        # [MTP] Semaphore IDs for second mcast (EH projection matmul)
        # Separate receiver semaphore from first mcast to avoid linked-VC posted write race
        mtp_ready_semaphore_id = 6
        mcast_eh_data_sender_semaphore_id = mcast_data_sender_semaphore_id
        mcast_eh_data_receiver_semaphore_id = 8
        mtp_done_semaphore_id = 9
        eh_matmul_done_semaphore_id = 10
        reduce_gate_semaphore_id = 11
        # [MTP] Semaphore IDs for singalling metadata unicast in spec stage
        metadata_ready_semaphore_id = 12
        bcast_config = DeepseekMinimalBroadcast.configure(
            mesh_device=mesh_device,
            input_tensor_mesh=input_tensor_mesh,
            output_tensor=intermediate_tensor_mesh,
            sender_coord=sender_coord,
            semaphores=bcast_semaphores,
            socket=socket_input,
            skip_ccl=skip_ccl,
            chunk_size_bytes=None,
            bcast_cb_id=bcast_pkt_cb,
            num_links=bcast_num_links,
            fabric_config=fabric_config,
            broadcast_topology_override=broadcast_topology_override,
        )

        # ====================================================================
        # MTP token broadcast config (exit device → all devices, 4-byte token)
        # ====================================================================
        mtp_bcast_config = None
        if is_mtp_base_stage and not skip_ccl:
            if mtp_bcast_semaphores is None:
                raise ValueError("mtp_bcast_semaphores required when is_mtp_base_stage=True and skip_ccl=False")
            mtp_bcast_config = DeepseekMinimalBroadcast.configure(
                mesh_device=mesh_device,
                input_tensor_mesh=base_token_buffer,
                output_tensor=base_token_buffer,
                sender_coord=argmax_final_mesh_coord,
                semaphores=mtp_bcast_semaphores,
                skip_ccl=False,
                chunk_size_bytes=None,
                bcast_cb_id=mtp_bcast_pkt_cb,
                num_links=1,
                fabric_config=fabric_config,
                broadcast_topology_override=broadcast_topology_override,
                tensor_size_bytes=32,
            )

        # ====================================================================
        # ReduceToOne setup (cross-device reduce for TP EH matmul)
        # ====================================================================
        reduce_params = {}
        if enable_reduce_to_one:
            reduce_root_coord = argmax_final_mesh_coord
            reduce_sem_addrs = [int(ttnn.get_global_semaphore_address(s)) for s in reduce_semaphores]
            reduce_shard_shape = [1, eh_n_per_core]
            reduce_element_size = 2  # bfloat16
            reduce_shard_elements = reduce_shard_shape[0] * reduce_shard_shape[1]
            reduce_payload_size_bytes = reduce_shard_elements * reduce_element_size
            reduce_compute_tile_size = 32 * 32 * reduce_element_size
            reduce_num_tiles = (reduce_shard_elements + 32 * 32 - 1) // (32 * 32)

            reduce_packet_header_size = ttnn.get_tt_fabric_packet_header_size_bytes()
            reduce_slot_size_bytes = reduce_packet_header_size + reduce_payload_size_bytes

            # CB17 (matmul out) size per core — the offset where CB20 (reduce received) begins
            eh_mm_cb17_shard_bytes = reduce_shard_elements * reduce_element_size

            # Worker cores = eh_matmul cores (DRAM bank cores, same on every device)
            eh_matmul_sample_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(
                mesh_device, ttnn.NOC.NOC_0
            )
            eh_matmul_sample_grid = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in eh_matmul_sample_cores]
            )
            reduce_worker_cores = ttnn.corerange_to_cores(eh_matmul_sample_grid, row_wise=True)
            reduce_col_to_cores = {}
            for c in reduce_worker_cores:
                reduce_col_to_cores.setdefault(c.x, []).append(c)
            reduce_sorted_cols = sorted(reduce_col_to_cores.keys())
            for x in reduce_sorted_cols:
                reduce_col_to_cores[x].sort(key=lambda c: c.y)
            reduce_num_workers_per_column = len(reduce_col_to_cores[reduce_sorted_cols[0]])

            # Fabric cores: one per column, placed to the right of the bottom worker
            reduce_fabric_cores = []
            reduce_col_to_fabric = {}
            for x in reduce_sorted_cols:
                bottom = max(reduce_col_to_cores[x], key=lambda c: c.y)
                fc = ttnn.CoreCoord(bottom.x + 1, bottom.y)
                reduce_fabric_cores.append(fc)
                reduce_col_to_fabric[x] = fc

            # Slot and shard index maps
            reduce_core_to_slot = {}
            for x in reduce_sorted_cols:
                for si, c in enumerate(reduce_col_to_cores[x]):
                    reduce_core_to_slot[(c.x, c.y)] = si
            reduce_core_to_shard = {(c.x, c.y): i for i, c in enumerate(reduce_worker_cores)}

            reduce_output_core = argmax_final_core_coord

            # Worker→fabric global semaphores (created once, shared across devices)
            device_grid_size = mesh_device.compute_with_storage_grid_size()
            reduce_wf_sem_cores = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
            )
            reduce_wf_ready_sem = ttnn.create_global_semaphore(mesh_device, reduce_wf_sem_cores, 0)
            reduce_wf_ready_sem_addr = int(ttnn.get_global_semaphore_address(reduce_wf_ready_sem))

            reduce_params = dict(
                sem_addrs=reduce_sem_addrs,
                fused_per_device=eh_mm_fused_per_device,
                cb17_offset_bytes=0,
                cb20_offset_bytes=eh_mm_cb17_shard_bytes,
                num_tiles=reduce_num_tiles,
                payload_size_bytes=reduce_payload_size_bytes,
                slot_size_bytes=reduce_slot_size_bytes,
                compute_tile_size=reduce_compute_tile_size,
                worker_cores=reduce_worker_cores,
                num_workers_per_column=reduce_num_workers_per_column,
                num_columns=len(reduce_sorted_cols),
                fabric_cores=reduce_fabric_cores,
                col_to_fabric=reduce_col_to_fabric,
                core_to_slot=reduce_core_to_slot,
                core_to_shard=reduce_core_to_shard,
                output_core=reduce_output_core,
                wf_ready_sem_addr=reduce_wf_ready_sem_addr,
                root_coord=reduce_root_coord,
            )

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        device_programs = []
        worker_ncrisc_arg_refs = []
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col
                # is_exit_device: whether this device is the exit device
                # enable_mtp_on_device: whether this device is the exit device and we run MTP on it
                if skip_ccl or argmax_final_mesh_coord is None:
                    is_exit_device = True
                else:
                    is_exit_device = row == int(argmax_final_mesh_coord[0]) and col == int(argmax_final_mesh_coord[1])

                enable_mtp_on_device = is_mtp_base_stage

                num_devices_in_stage = mesh_rows * mesh_cols
                is_e_norm_device = enable_mtp_on_device and (device_idx < num_devices_in_stage // 2)
                eh_norm_slice_idx = (
                    device_idx
                    if is_e_norm_device
                    else (device_idx - num_devices_in_stage // 2)
                    if enable_mtp_on_device
                    else 0
                )
                eh_norm_slice_offset_bytes = (
                    eh_norm_slice_idx * eh_num_tiles_k * input_tile_size if enable_mtp_on_device else 0
                )
                # Get per-device tensors
                input_tensor_device = input_tensors_per_device[device_idx]
                intermediate_tensor_device = intermediate_tensors_per_device[device_idx]
                gamma_tensor_device = (
                    gamma_tensors_per_device[device_idx] if gamma_tensors_per_device is not None else None
                )
                vocab_tensor_device = vocab_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]
                indices_tensor_device = indices_tensors_per_device[device_idx] if enable_argmax else None
                output_index_tensor_device = output_index_tensors_per_device[device_idx] if enable_argmax else None

                # Get device handle
                device = input_tensor_device.device()

                # Broadcast worker core from config (root/non-root consistent).
                worker_core = bcast_config.get_worker_core(coord)
                bcast_worker_core_phys = device.worker_core_from_logical_core(worker_core)

                # ================================================================
                # Core grid configuration (per-device)
                # ================================================================
                # Sender core: from input_tensor (must be single core)
                mcast_sender_core_grid = input_tensor_device.memory_config().shard_spec.grid
                assert mcast_sender_core_grid.num_cores() == 1, "input_tensor must be sharded on a single sender core"
                mcast_sender_core = list(mcast_sender_core_grid.ranges())[0].start
                recv_socket_on_this_device = False
                if enable_socket_input:
                    recv_socket_on_this_device = (
                        input_socket_core.device_coord == ttnn.MeshCoordinate(row, col)
                        and input_socket_core.core_coord.x == mcast_sender_core.x
                        and input_socket_core.core_coord.y == mcast_sender_core.y
                    )
                if enable_socket_input and not skip_ccl:
                    if (row == sender_row and col == sender_col) and not recv_socket_on_this_device:
                        raise ValueError(
                            "socket input active core must match sender device/core in multi-device lm_head_sampling"
                        )
                    if recv_socket_on_this_device and not (row == sender_row and col == sender_col):
                        raise ValueError(
                            "socket input active core must be on sender device/core in multi-device lm_head_sampling"
                        )
                input_socket_mode = input_socket_mode_selected if recv_socket_on_this_device else input_socket_mode_none
                # Matmul cores: from vocab_tensor (multiple cores with weight shards)
                matmul_core_grid = vocab_tensor_device.memory_config().shard_spec.grid
                argmax_core_grid = matmul_core_grid
                argmax_cores_row_wise = ttnn.corerange_to_cores(argmax_core_grid, row_wise=True)
                matmul_bbox = matmul_core_grid.bounding_box()
                mcast_grid = ttnn.CoreRange(
                    ttnn.CoreCoord(
                        min(matmul_bbox.start.x, mcast_sender_core.x),
                        min(matmul_bbox.start.y, mcast_sender_core.y),
                    ),
                    ttnn.CoreCoord(
                        max(matmul_bbox.end.x, mcast_sender_core.x),
                        max(matmul_bbox.end.y, mcast_sender_core.y),
                    ),
                )
                mcast_grid_set = ttnn.CoreRangeSet([mcast_grid])
                num_mcast_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y

                # Compute per-core bank_id and vc for EH DRAM streaming matmul
                if enable_mtp_on_device:
                    eh_matmul_noc = ttnn.NOC.NOC_0
                    eh_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(
                        mesh_device, eh_matmul_noc
                    )
                    eh_matmul_core_grid = eh_matmul_sample_grid
                    eh_bank_id_core_values = []
                    eh_vc_core_values = []
                    eh_bank_ids = []
                    for idx, core in enumerate(eh_worker_cores):
                        bank_id = idx % len(eh_worker_cores)
                        vc = bank_id & 0x3
                        for j in range(idx):
                            if eh_worker_cores[j].y == core.y and (eh_bank_ids[j] & 0x3) == (bank_id & 0x3):
                                vc = (vc + 1) & 0x3
                                break
                        eh_bank_ids.append(bank_id)
                        eh_bank_id_core_values.append((core, bank_id))
                        eh_vc_core_values.append((core, vc))

                # Build mcast receiver grid = mcast grid minus sender core
                mcast_receiver_ranges = []
                for r in range(mcast_grid.start.y, mcast_grid.end.y + 1):
                    for c in range(mcast_grid.start.x, mcast_grid.end.x + 1):
                        if c == mcast_sender_core.x and r == mcast_sender_core.y:
                            continue
                        mcast_receiver_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(c, r), ttnn.CoreCoord(c, r)))
                mcast_receiver_grid = ttnn.CoreRangeSet(mcast_receiver_ranges)

                # All cores = mcast grid (sender is already included)
                all_cores = mcast_grid_set

                if enable_argmax:
                    if indices_tensor_device.memory_config().shard_spec.grid != argmax_core_grid:
                        raise ValueError("indices_tensor must be width-sharded on the same core grid as LM-head scores")
                    if (
                        indices_tensor_device.memory_config().shard_spec.shape
                        != output_tensor_device.memory_config().shard_spec.shape
                    ):
                        raise ValueError("indices_tensor shard shape must match output_tensor shard shape")
                    if indices_tensor_device.dtype != ttnn.uint32:
                        raise ValueError("indices_tensor must be uint32")
                    if output_index_tensor_device.memory_config().shard_spec.grid.num_cores() != 1:
                        raise ValueError("output_index_tensor must be sharded on a single final core")
                    if not _is_singleton_prefix_shape(output_index_tensor_device.shape, 1):
                        raise ValueError(
                            "output_index_tensor must be singleton-prefix with last dim 1 (per-device logical shape (1,1))"
                        )

                    output_index_core = output_index_tensor_device.memory_config().shard_spec.grid.ranges()[0].start
                    argmax_final_core = (
                        output_index_core if argmax_final_core_coord is None else argmax_final_core_coord
                    )
                    if not any(
                        c.x == argmax_final_core.x and c.y == argmax_final_core.y for c in argmax_cores_row_wise
                    ):
                        raise ValueError("argmax_final_core_coord must be within the matmul core grid")
                    if output_index_core.x != argmax_final_core.x or output_index_core.y != argmax_final_core.y:
                        raise ValueError("output_index_tensor shard core must match argmax_final_core_coord")
                    emit_socket_on_this_device = bool(enable_socket_output)
                    argmax_num_values = n_per_core
                    argmax_num_senders = len(argmax_cores_row_wise)
                    argmax_expected_remote_incs = argmax_num_senders - 1
                    argmax_sender_idx_lookup = {(c.x, c.y): idx for idx, c in enumerate(argmax_cores_row_wise)}
                    # `sampling_winner_page_bytes` is computed below from
                    # `sampling_topk_scores_slot_bytes + sampling_topk_indices_slot_bytes`
                    # (sampling.hpp uses it to size the local winner-CB NOC writes).
                    argmax_mesh_mode = 0
                    argmax_stage1_sender = 0
                    argmax_stage1_receiver = 0
                    argmax_stage2_sender = 0
                    argmax_stage2_receiver = 0
                    argmax_stage1_slot_base_offset = 0
                    argmax_stage1_num_slots = 0
                    argmax_stage1_expected_remote_incs = 0
                    argmax_stage1_local_slot_offset = 0
                    argmax_stage2_slot_base_offset = 0
                    argmax_stage2_num_slots = 0
                    argmax_stage2_expected_remote_incs = 0
                    argmax_stage2_local_slot_offset = 0
                    argmax_mesh_local_send_slot_offset = 0
                    is_argmax_mesh_sender_core = False
                    sender_link_idx = 0
                    dest_coord = ttnn.MeshCoordinate(row, col)
                    per_core_brisc_runtime_args = []

                    # ----------------------------------------------------------------
                    # [Sampling] hyperparameter pre-computation.
                    # Keep definitions identical to the mesh top-K micro-op
                    # (micro_ops/sampling/op.py :: _op_mesh_topk) so the mesh
                    # sender addressing in the per-core RT args below lines up
                    # with the CB layout assembled later in this function.
                    # ----------------------------------------------------------------
                    sampling_topk_k = 32
                    sampling_topk_min_alignment = 32
                    sampling_l1_alignment = 16
                    sampling_bf16_tile_size = 2 * 32 * 32  # 2048
                    sampling_uint32_tile_size = 4 * 32 * 32  # 4096
                    sampling_topk_scores_slot_bytes = (
                        ((sampling_topk_min_alignment * 2) + sampling_l1_alignment - 1) // sampling_l1_alignment
                    ) * sampling_l1_alignment
                    sampling_topk_indices_slot_bytes = (
                        ((sampling_topk_min_alignment * 4) + sampling_l1_alignment - 1) // sampling_l1_alignment
                    ) * sampling_l1_alignment
                    sampling_winner_page_bytes = sampling_topk_scores_slot_bytes + sampling_topk_indices_slot_bytes
                    # Stage-1 scores/indices precede stage-2 in the receiver's
                    # scratch tensors (one tensor for scores, one for indices).
                    sampling_stage1_num_slots = mesh_rows
                    sampling_stage2_num_slots = mesh_cols
                    sampling_stage1_mesh_tiles = (
                        sampling_stage1_num_slots * sampling_topk_min_alignment + 1023
                    ) // 1024
                    sampling_stage2_scores_scratch_offset = sampling_stage1_mesh_tiles * sampling_bf16_tile_size
                    sampling_stage2_indices_scratch_offset = sampling_stage1_mesh_tiles * sampling_uint32_tile_size
                    # Phase-2 offsets inside the per-core topk_in_{scores,indices}_cb.
                    sampling_phase1_tiles = (n_per_core + 1023) // 1024
                    sampling_phase2_scores_byte_offset = sampling_phase1_tiles * sampling_bf16_tile_size
                    sampling_phase2_indices_byte_offset = sampling_phase1_tiles * sampling_uint32_tile_size
                    sampling_stage2_mesh_tiles = (
                        sampling_stage2_num_slots * sampling_topk_min_alignment + 1023
                    ) // 1024
                    # Per-stage CT-arg values:
                    #   * BaseLMHeadStage (default + MTP base): k=32, enable_metadata=1,
                    #     copy_probabilities=1; k/p/temperature actually consumed at
                    #     runtime from the metadata packet.
                    #   * SpecLMHeadStage (is_mtp_verify_stage): k=1, no metadata copy.
                    if is_mtp_verify_stage:
                        sampling_topk_k_value = 1
                        sampling_enable_metadata_value = 0
                        sampling_copy_probabilities_value = 0
                    else:
                        sampling_topk_k_value = sampling_topk_k
                        sampling_enable_metadata_value = 1
                        sampling_copy_probabilities_value = 1
                    # Canonical packings, matching micro_ops/sampling/op.py:
                    #   * inv_temp_bf16: two copies of bf16(1/temp) packed into uint32
                    #     (the LLK softmax recip helper consumes both halves).
                    #   * p_bf16: float32 bit pattern of `p`. (Naming is legacy; the
                    #     kernel does __builtin_bit_cast<float>(uint32_t).)
                    # These defaults only matter when sampling_enable_metadata=0
                    # (spec stage); base stage overwrites k/p/temperature from the
                    # runtime metadata packet.
                    sampling_inv_temp_bf16 = float_to_bfloat16_packed(1.0)
                    sampling_p_bf16 = float_to_uint32(1.0)
                    # sampling.hpp dereferences `CTArgs::output_addr` unconditionally
                    # (writes the selected token index to L1) — so passing 0 would
                    # crash. The matching RT slot in ReaderArgs is declared but never
                    # consumed inside sampling.hpp. We thread the per-device output
                    # tensor address here. Rand output remains 0 (kernel-side guarded
                    # by `if constexpr (rand_output_addr != 0)`).
                    sampling_output_addr_ct = (
                        int(output_index_tensor_device.buffer_address())
                        if output_index_tensor_device is not None
                        else 0
                    )
                    # Scratch tensor addresses are program-build-time-known and used
                    # as CT args by sampling.hpp (NCRISC reader). Mirrors how
                    # micro_ops/sampling/op.py wires them.
                    sampling_scores_scratch_addr_ct = (
                        int(scores_scratch_tensors_per_device[device_idx].buffer_address())
                        if (not skip_ccl and scores_scratch_tensors_per_device is not None)
                        else 0
                    )
                    sampling_indices_scratch_addr_ct = (
                        int(indices_scratch_tensors_per_device[device_idx].buffer_address())
                        if (not skip_ccl and indices_scratch_tensors_per_device is not None)
                        else 0
                    )

                    if not skip_ccl:
                        target_row = int(argmax_final_mesh_coord[0])
                        target_col = int(argmax_final_mesh_coord[1])
                        if not (0 <= target_row < mesh_rows and 0 <= target_col < mesh_cols):
                            raise ValueError(
                                f"argmax_final_mesh_coord {argmax_final_mesh_coord} out of bounds for mesh shape {mesh_shape}"
                            )
                        emit_socket_on_this_device = bool(
                            enable_socket_output and row == target_row and col == target_col
                        )

                        def _x_axis_link_idx_for_stage1_sender(sender_row_local: int) -> int:
                            linear_distance = abs(int(sender_row_local) - target_row)
                            ring_distance = min(linear_distance, mesh_rows - linear_distance)
                            max_ring_distance = mesh_rows // 2
                            first_half_threshold = (max_ring_distance + 1) // 2
                            return 0 if ring_distance <= first_half_threshold else 1

                        argmax_mesh_mode = 1
                        argmax_stage1_slot_base_offset = 0
                        argmax_stage1_num_slots = mesh_rows
                        argmax_stage2_slot_base_offset = 0
                        argmax_stage2_num_slots = mesh_cols
                        argmax_stage1_expected_remote_incs = mesh_rows - 1
                        argmax_stage2_expected_remote_incs = mesh_cols - 1
                        argmax_stage1_sender = 1 if row != target_row else 0
                        argmax_stage1_receiver = 1 if row == target_row else 0
                        argmax_stage2_sender = 1 if (row == target_row and col != target_col) else 0
                        argmax_stage2_receiver = 1 if (row == target_row and col == target_col) else 0
                        argmax_stage1_local_slot_offset = row
                        argmax_stage2_local_slot_offset = col
                        is_argmax_mesh_sender_core = bool(argmax_stage1_sender or argmax_stage2_sender)
                        argmax_mesh_local_send_slot_offset = 0

                        if is_argmax_mesh_sender_core:
                            if argmax_stage1_sender:
                                dest_coord = ttnn.MeshCoordinate(target_row, col)
                                sender_dst_sem_addr = global_sem_addr
                                sender_link_idx = _x_axis_link_idx_for_stage1_sender(row)
                            else:
                                dest_coord = ttnn.MeshCoordinate(target_row, target_col)
                                sender_dst_sem_addr = global_stage2_sem_addr
                                sender_link_idx = 0

                            dest_idx = int(dest_coord[0]) * mesh_cols + int(dest_coord[1])
                            # [Sampling] mesh-sender writes its local top-K winner
                            # (scores + indices) into the receiver's two scratch
                            # tensors — scores into `scores_scratch`, indices into
                            # `indices_scratch`. Slot layout mirrors _op_mesh_topk.
                            dest_scores_scratch_base = int(scores_scratch_tensors_per_device[dest_idx].buffer_address())
                            dest_indices_scratch_base = int(
                                indices_scratch_tensors_per_device[dest_idx].buffer_address()
                            )
                            if argmax_stage1_sender:
                                dst_scores_addr = dest_scores_scratch_base + row * sampling_topk_scores_slot_bytes
                                dst_indices_addr = dest_indices_scratch_base + row * sampling_topk_indices_slot_bytes
                            else:
                                dst_scores_addr = (
                                    dest_scores_scratch_base
                                    + sampling_stage2_scores_scratch_offset
                                    + col * sampling_topk_scores_slot_bytes
                                )
                                dst_indices_addr = (
                                    dest_indices_scratch_base
                                    + sampling_stage2_indices_scratch_offset
                                    + col * sampling_topk_indices_slot_bytes
                                )
                            per_core_brisc_runtime_args.append(
                                (
                                    argmax_final_core,
                                    [
                                        int(mesh_device.get_fabric_node_id(dest_coord).mesh_id),
                                        int(mesh_device.get_fabric_node_id(dest_coord).chip_id),
                                        int(dst_scores_addr),
                                        int(dst_indices_addr),
                                        int(sender_dst_sem_addr),
                                    ],
                                )
                            )

                    if emit_socket_on_this_device:
                        if (
                            socket_core.device_coord != ttnn.MeshCoordinate(row, col)
                            or socket_core.core_coord.x != argmax_final_core.x
                            or socket_core.core_coord.y != argmax_final_core.y
                        ):
                            raise ValueError(
                                "socket output active core must match argmax final core and emitting mesh device for lm_head_sampling"
                            )

                    argmax_socket_mode = socket_mode_selected if emit_socket_on_this_device else socket_mode_none
                    final_core_phys = device.worker_core_from_logical_core(argmax_final_core)

                # Determine if sender is part of the mcast rectangle
                is_part_of_receiver_grid = mcast_grid.contains(mcast_sender_core)
                persistent_target_mesh_coord = ttnn.MeshCoordinate(sender_row, sender_col)
                persistent_target_device_idx = sender_row * mesh_cols + sender_col
                persistent_target_device = input_tensors_per_device[persistent_target_device_idx].device()
                persistent_target_input_core_phys = persistent_target_device.worker_core_from_logical_core(
                    mcast_sender_core
                )
                persistent_target_node = mesh_device.get_fabric_node_id(persistent_target_mesh_coord)
                persistent_enable = int(persistent_mode and emit_socket_on_this_device)

                # broadcast_rms-style BRISC source selection:
                # - CCL path: packet CB
                # - skip_ccl + socket path: rmsnorm input CB
                # - otherwise BRISC broadcast path is idle
                # Get NOC coordinates for mcast destination
                mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
                mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

                # [MTP] NOC coords of argmax final core (for eh_matmul_done semaphore target)
                if enable_argmax:
                    argmax_core_phys = device.worker_core_from_logical_core(argmax_final_core)
                    argmax_core_noc_x = int(argmax_core_phys.x)
                    argmax_core_noc_y = int(argmax_core_phys.y)
                else:
                    argmax_core_noc_x = 0
                    argmax_core_noc_y = 0

                # [MTP] Number of EH matmul cores (each incs eh_matmul_done semaphore once).
                # Must match the core set that has is_eh_matmul_core=1 (matmul_core_grid on exit device).
                eh_matmul_num_cores = eh_matmul_core_grid.num_cores() if enable_mtp_on_device else 0

                # [MTP] EH output parameters (used by socket send for gather_dst_cb sizing)
                eh_output_tile_size = out_tile.get_tile_size(data_format) if enable_mtp_on_device else 1
                eh_gather_dst_num_pages = eh_matmul_num_cores * eh_out_w_per_core if enable_mtp_on_device else 0
                eh_gather_send_total_bytes = (
                    (eh_gather_dst_num_pages + socket_page_size_bytes // (eh_output_tile_size)) * eh_output_tile_size
                    if enable_mtp_on_device
                    else 0
                )
                # MTP metadata landing buffer on argmax final core (NCRISC unicast from exit input core).
                metadata_output_l1_addr = 0
                if metadata_tensor is not None:
                    _metadata_buf = ttnn.get_device_tensors(metadata_tensor)[device_idx]
                    metadata_output_l1_addr = int(_metadata_buf.buffer_address())

                sender_core_phys_for_mtp = device.worker_core_from_logical_core(mcast_sender_core)
                mtp_embedding_dram_base = (
                    int(embedding_tensors_per_device[device_idx].buffer_address()) if enable_mtp_on_device else 0
                )
                base_token_device = (
                    base_token_tensors_per_device[device_idx] if base_token_tensors_per_device is not None else None
                )
                if enable_mtp_on_device:
                    if base_token_device is not None:
                        mtp_token_l1_addr = int(base_token_device.buffer_address())
                    else:
                        mtp_token_l1_addr = int(intermediate_tensor_device.buffer_address())
                else:
                    mtp_token_l1_addr = 0
                mtp_input_core_noc_x = int(sender_core_phys_for_mtp.x) if enable_mtp_on_device else 0
                mtp_input_core_noc_y = int(sender_core_phys_for_mtp.y) if enable_mtp_on_device else 0
                mtp_argmax_output_l1_addr = (
                    int(output_index_tensor_device.buffer_address()) if enable_mtp_on_device else 0
                )

                # ================================================================
                # ReduceToOne per-device setup
                # ================================================================
                reduce_device_role = 0
                reduce_dest_coord = coord
                reduce_dest_fabric_node_id = None
                reduce_output_core_phys_x = 0
                reduce_output_core_phys_y = 0
                if enable_reduce_to_one:
                    reduce_root_row = int(reduce_params["root_coord"][0])
                    reduce_use_torus = reduce_root_row in [0, 3]
                    reduce_device_role = get_reduce_device_role(coord, reduce_params["root_coord"], reduce_use_torus)
                    if reduce_device_role == MESH_LEAF:
                        if reduce_use_torus:
                            reduce_dest_coord = ttnn.MeshCoordinate(row - 1 if row == 1 else row + 1, col)
                        else:
                            reduce_dest_coord = ttnn.MeshCoordinate(row + 1 if row == 0 else row - 1, col)
                    elif reduce_device_role == MESH_ROOT3:
                        reduce_dest_coord = ttnn.MeshCoordinate(int(reduce_params["root_coord"][0]), col)
                    elif reduce_device_role == MESH_ROOT2:
                        reduce_dest_coord = reduce_params["root_coord"]
                    else:
                        reduce_dest_coord = reduce_params["root_coord"]
                    reduce_dest_fabric_node_id = mesh_device.get_fabric_node_id(reduce_dest_coord)
                    roc_phys = device.worker_core_from_logical_core(reduce_params["output_core"])
                    reduce_output_core_phys_x = int(roc_phys.x)
                    reduce_output_core_phys_y = int(roc_phys.y)
                    role_name = {MESH_LEAF: "LEAF", MESH_ROOT3: "ROOT3", MESH_ROOT2: "ROOT2", MESH_ROOT1: "ROOT1"}

                reduce_num_fabric_and_worker_cores = (
                    len(reduce_params["fabric_cores"]) + len(reduce_params["worker_cores"])
                    if (enable_reduce_to_one and enable_mtp_on_device)
                    else 0
                )

                reduce_fc_nocs = [(0, 0)] * 2
                if enable_reduce_to_one and enable_mtp_on_device:
                    for i, fc in enumerate(reduce_params["fabric_cores"][:2]):
                        fc_phys = device.worker_core_from_logical_core(fc)
                        reduce_fc_nocs[i] = (int(fc_phys.x), int(fc_phys.y))

                # ================================================================
                # NCRISC compile-time args
                # ================================================================
                ncrisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("enable_argmax", 1),
                    ("input_socket_mode", input_socket_mode),
                    # Mcast source (for setup_sharded_buffer on sender core)
                    ("mcast_src_cb", mcast_src_cb),
                    ("mcast_src_num_pages", rms_num_tiles),
                    ("rmsnorm_input_cb", rmsnorm_input_cb),
                    ("rmsnorm_gamma_cb", rmsnorm_gamma_cb),
                    ("rmsnorm_num_tiles", rms_num_tiles),
                    # Mcast receiver
                    ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
                    ("mcast_dst_cb", mcast_dst_cb),
                    ("mcast_dst_num_pages", num_tiles_k),
                    # Matmul
                    ("matmul_in0", mcast_dst_cb),  # Matmul reads from mcast destination
                    ("matmul_in1", matmul_in1_cb),
                    ("matmul_out", matmul_out_cb),
                    ("matmul_k_num_tiles", num_tiles_k),
                    ("matmul_out_w", out_w_per_core),
                    # Argmax sampling
                    ("sampling_num_values", argmax_num_values),
                    ("sampling_winner_page_bytes", sampling_winner_page_bytes),
                    ("sampling_num_senders", argmax_num_senders),
                    ("sampling_expected_remote_incs", argmax_expected_remote_incs),
                    ("sampling_receiver_semaphore_id", argmax_receiver_semaphore_id),
                    ("sampling_local_ready_semaphore_id", argmax_local_ready_semaphore_id),
                    ("sampling_mesh_mode", argmax_mesh_mode),
                    ("sampling_stage1_sender", argmax_stage1_sender),
                    ("sampling_stage1_receiver", argmax_stage1_receiver),
                    ("sampling_stage2_sender", argmax_stage2_sender),
                    ("sampling_stage2_receiver", argmax_stage2_receiver),
                    ("sampling_stage1_slot_base_offset", argmax_stage1_slot_base_offset),
                    ("sampling_stage1_num_slots", argmax_stage1_num_slots),
                    ("sampling_stage1_expected_remote_incs", argmax_stage1_expected_remote_incs),
                    ("sampling_stage1_local_slot_offset", row),
                    ("sampling_stage2_slot_base_offset", argmax_stage2_slot_base_offset),
                    ("sampling_stage2_num_slots", argmax_stage2_num_slots),
                    ("sampling_stage2_expected_remote_incs", argmax_stage2_expected_remote_incs),
                    ("sampling_stage2_local_slot_offset", col),
                    ("sampling_mesh_local_send_slot_offset", argmax_mesh_local_send_slot_offset),
                    ("sampling_socket_mode", argmax_socket_mode),
                    ("sampling_socket_cb", sampling_socket_cb if enable_socket_output else 0),
                    ("sampling_socket_page_size_bytes", socket_page_size_bytes if enable_socket_output else 0),
                    # ── Sampling kernel additions (sampling.hpp ReaderCTArgs) ────
                    ("sampling_topk_k", sampling_topk_k_value),
                    ("sampling_winner_cb", sampling_winner_cb),
                    ("sampling_softmax_in_cb", sampling_softmax_in_cb),
                    ("sampling_softmax_out_cb", sampling_softmax_out_cb),
                    ("sampling_softmax_exp_cb", sampling_softmax_exp_cb),
                    ("sampling_scaler_cb", sampling_scaler_cb),
                    ("sampling_temp_cb", sampling_temp_cb),
                    ("sampling_inv_temp_bf16", sampling_inv_temp_bf16),
                    ("sampling_topk_in_scores_cb", sampling_topk_in_scores_cb),
                    ("sampling_topk_in_indices_cb", sampling_topk_in_indices_cb),
                    ("sampling_topk_out_scores_cb", sampling_topk_out_scores_cb),
                    ("sampling_topk_out_indices_cb", sampling_topk_out_indices_cb),
                    ("sampling_phase2_scores_byte_offset", sampling_phase2_scores_byte_offset),
                    ("sampling_phase2_indices_byte_offset", sampling_phase2_indices_byte_offset),
                    ("sampling_mesh_stage_scores_cb", sampling_mesh_stage_scores_cb),
                    ("sampling_mesh_stage_indices_cb", sampling_mesh_stage_indices_cb),
                    ("sampling_scores_scratch_stage2_offset", sampling_stage2_scores_scratch_offset),
                    ("sampling_indices_scratch_stage2_offset", sampling_stage2_indices_scratch_offset),
                    ("sampling_scores_scratch_addr", sampling_scores_scratch_addr_ct),
                    ("sampling_indices_scratch_addr", sampling_indices_scratch_addr_ct),
                    ("persistent_mode", 1 if persistent_mode else 0),
                    ("termination_semaphore_addr", termination_global_sem_addr),
                    ("fabric_gate_bcast_turn_semaphore_id", fabric_gate_bcast_turn_semaphore_id),
                    ("fabric_gate_argmax_turn_semaphore_id", fabric_gate_argmax_turn_semaphore_id),
                    ("fabric_gate_bcast_noc_x", int(bcast_worker_core_phys.x)),
                    ("fabric_gate_bcast_noc_y", int(bcast_worker_core_phys.y)),
                    ("fabric_gate_argmax_noc_x", int(final_core_phys.x)),
                    ("fabric_gate_argmax_noc_y", int(final_core_phys.y)),
                    ("mesh_row", row),
                    ("mesh_col", col),
                    # [MTP] is_eh_matmul_core must be in base args so non-exit devices get it (0); exit device descriptor overrides per-core.
                    ("matmul_eh_in0", mcast_eh_dst_cb),
                    ("matmul_eh_in1", matmul_eh_cb),
                    ("matmul_eh_out", matmul_out_eh_cb),
                    ("matmul_eh_k_num_tiles", eh_num_tiles_k if enable_mtp_on_device else 0),
                    ("matmul_eh_out_w", eh_out_w_per_core if enable_mtp_on_device else 0),
                    (
                        "matmul_eh_dram_in1_tensor_addr",
                        eh_proj_tensors_per_device[device_idx].buffer_address() if enable_mtp_on_device else 0,
                    ),
                    ("matmul_eh_dram_in1_page_size", eh_page_size if enable_mtp_on_device else 0),
                    ("matmul_eh_dram_in1_num_pages", eh_num_pages if enable_mtp_on_device else 0),
                    ("matmul_eh_dram_in1_block_size_bytes", eh_in1_block_size_bytes if enable_mtp_on_device else 0),
                    ("matmul_eh_subblock_k", eh_subblock_k if enable_mtp_on_device else 0),
                    ("matmul_eh_num_subblocks_k", eh_num_subblocks_k if enable_mtp_on_device else 0),
                    ("matmul_eh_out_num_tiles", eh_out_num_tiles if enable_mtp_on_device else 0),
                    ("embedding_cb", embedding_cb),
                    ("h_gamma_cb", h_gamma_cb),
                    ("e_gamma_cb", e_gamma_cb),
                    ("mcast_eh_src_cb", mcast_eh_src_cb),
                    ("mcast_eh_dst_cb", mcast_eh_dst_cb),
                    ("rmsnorm_h_input_cb", rmsnorm_input_cb),
                    ("rmsnorm_h_output_cb", mcast_eh_src_cb),
                    ("rmsnorm_e_input_cb", embedding_cb),
                    ("rmsnorm_e_output_cb", mcast_eh_src_cb),
                    ("hnorm_ready_cb", hnorm_ready_cb),
                    ("mcast_eh_ready_cb", mcast_eh_ready_cb if enable_mtp_on_device else 0),
                    # [MTP] semaphores
                    ("mtp_ready_semaphore_id", mtp_ready_semaphore_id),
                    ("metadata_ready_semaphore_id", metadata_ready_semaphore_id),
                    (
                        "mcast_eh_data_sender_semaphore",
                        mcast_eh_data_sender_semaphore_id if enable_mtp_on_device else 0,
                    ),
                    ("mcast_eh_data_receiver_semaphore", mcast_eh_data_receiver_semaphore_id),
                    ("mcast_eh_src_num_pages", rms_num_tiles if enable_mtp_on_device else 0),
                    ("mcast_eh_dst_num_pages", eh_num_tiles_k if enable_mtp_on_device else 0),
                    ("rmsnorm_h_num_tiles", rms_num_tiles),
                    ("rmsnorm_e_num_tiles", e_num_tiles if enable_mtp_on_device else 0),
                    ("embedding_size_bytes", embedding_dim * 2 if enable_mtp_on_device else 0),
                    # Sender core NOC for L1-to-L1 copy (embedding region in mcast_eh_src_cb -> embedding_cb)
                    ("reduce_gate_semaphore_id", reduce_gate_semaphore_id if enable_mtp_on_device else 0),
                    ("sampling_defer_socket_output", 1 if enable_socket_output else 0),
                    ("argmax_core_noc_x", argmax_core_noc_x if (enable_mtp_on_device or is_mtp_verify_stage) else 0),
                    ("argmax_core_noc_y", argmax_core_noc_y if (enable_mtp_on_device or is_mtp_verify_stage) else 0),
                    ("has_bypass_socket_output", 0),
                    ("has_bypass_socket_input", 0),
                    ("mtp_embedding_dram_base", mtp_embedding_dram_base),
                    ("mtp_token_l1_addr", mtp_token_l1_addr),
                    ("mtp_input_core_noc_x", mtp_input_core_noc_x),
                    ("mtp_input_core_noc_y", mtp_input_core_noc_y),
                    ("mtp_argmax_output_addr", mtp_argmax_output_l1_addr),
                    ("metadata_output_l1_addr", metadata_output_l1_addr),
                    ("is_e_norm_device", 1 if is_e_norm_device else 0),
                    # [Reduce] NCRISC CT args
                    ("reduce_device_role", reduce_device_role),
                    ("reduce_num_tiles", reduce_params["num_tiles"] if enable_reduce_to_one else 0),
                    ("reduce_local_cb", reduce_local_cb),
                    ("reduce_received_cb", reduce_received_cb),
                    # [Reduce] Pre-reduce gate: BRISC on argmax_final_core signals 2 fabric cores after sampling
                    ("reduce_fc_noc_0_x", reduce_fc_nocs[0][0]),
                    ("reduce_fc_noc_0_y", reduce_fc_nocs[0][1]),
                    ("reduce_fc_noc_1_x", reduce_fc_nocs[1][0]),
                    ("reduce_fc_noc_1_y", reduce_fc_nocs[1][1]),
                ]
                ncrisc_named_compile_time_args.extend(bcast_config.get_ncrisc_named_ct_args(coord))
                if mtp_bcast_config is not None:
                    ncrisc_named_compile_time_args.extend(
                        [(f"mtp_{k}", v) for k, v in mtp_bcast_config.get_ncrisc_named_ct_args(coord)]
                    )
                else:
                    ncrisc_named_compile_time_args.extend(
                        [
                            ("mtp_bcast_data_cb_id", 0),
                            ("mtp_bcast_num_pages_to_read", 0),
                            ("mtp_bcast_tensor0_page_size", 0),
                            ("mtp_bcast_num_neighbors", 0),
                            ("mtp_bcast_num_links", 1),
                            ("mtp_bcast_is_root", 0),
                            ("mtp_bcast_chunk_size_bytes", 1),
                            ("mtp_bcast_last_chunk_size_bytes", 1),
                            ("mtp_bcast_num_chunks", 1),
                        ]
                    )

                # ================================================================
                # BRISC compile-time args
                # ================================================================
                brisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("enable_argmax", 1),
                    ("input_socket_mode", input_socket_mode),
                    ("mtp_token_l1_addr", mtp_token_l1_addr),
                    ("mtp_input_core_noc_x", mtp_input_core_noc_x),
                    ("mtp_input_core_noc_y", mtp_input_core_noc_y),
                    ("mtp_argmax_output_addr", mtp_argmax_output_l1_addr),
                    # Mcast sender
                    ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
                    ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
                    ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
                    ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
                    ("mcast_num_cores", num_mcast_cores),
                    ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
                    ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
                    ("mcast_data_size_bytes", mcast_data_size_bytes),
                    ("rmsnorm_input_cb", rmsnorm_input_cb),
                    ("rmsnorm_num_tiles", rms_num_tiles),
                    ("mcast_src_cb", mcast_src_cb),
                    ("mcast_src_num_pages", rms_num_tiles),
                    ("mcast_dst_cb", mcast_dst_cb),
                    ("matmul_eh_out", matmul_out_eh_cb),
                    ("matmul_eh_out_w", eh_out_w_per_core if enable_mtp_on_device else 0),
                    ("mcast_is_part_of_receiver_grid", is_part_of_receiver_grid),
                    ("sampling_winner_page_bytes", sampling_winner_page_bytes),
                    ("sampling_local_ready_semaphore_id", argmax_local_ready_semaphore_id),
                    ("sampling_socket_mode", argmax_socket_mode),
                    ("sampling_socket_cb", sampling_socket_cb if enable_socket_output else 0),
                    ("sampling_socket_page_size_bytes", socket_page_size_bytes if enable_socket_output else 0),
                    # ── Sampling kernel additions (sampling.hpp WriterCTArgs) ────
                    ("sampling_topk_k", sampling_topk_k_value),
                    ("sampling_softmax_out_cb", sampling_softmax_out_cb),
                    ("sampling_rand_cb", sampling_rand_cb),
                    ("sampling_winner_cb", sampling_winner_cb),
                    ("sampling_p_bf16", sampling_p_bf16),
                    ("sampling_topk_scores_slot_bytes", sampling_topk_scores_slot_bytes),
                    ("sampling_mesh_mode", argmax_mesh_mode),
                    ("sampling_stage2_receiver", argmax_stage2_receiver),
                    ("sampling_output_addr", sampling_output_addr_ct),
                    ("sampling_rand_output_addr", 0),
                    ("sampling_inv_temp_bf16", sampling_inv_temp_bf16),
                    ("sampling_softmax_in_cb", sampling_softmax_in_cb),
                    ("sampling_temp_cb", sampling_temp_cb),
                    ("sampling_enable_metadata", sampling_enable_metadata_value),
                    ("sampling_copy_probabilities", sampling_copy_probabilities_value),
                    ("persistent_mode", 1 if persistent_mode else 0),
                    ("termination_semaphore_addr", termination_global_sem_addr),
                    ("fabric_gate_bcast_turn_semaphore_id", fabric_gate_bcast_turn_semaphore_id),
                    ("fabric_gate_argmax_turn_semaphore_id", fabric_gate_argmax_turn_semaphore_id),
                    ("fabric_gate_bcast_noc_x", int(bcast_worker_core_phys.x)),
                    ("fabric_gate_bcast_noc_y", int(bcast_worker_core_phys.y)),
                    ("fabric_gate_argmax_noc_x", int(final_core_phys.x)),
                    ("fabric_gate_argmax_noc_y", int(final_core_phys.y)),
                    ("mesh_row", row),
                    ("mesh_col", col),
                    # [MTP] Second mcast (EH projection input); is_eh_matmul_core in base so non-exit gets 0.
                    ("mcast_eh_dest_noc_start_x", mcast_dest_noc_start.x if enable_mtp_on_device else 0),
                    ("mcast_eh_dest_noc_start_y", mcast_dest_noc_start.y if enable_mtp_on_device else 0),
                    ("mcast_eh_dest_noc_end_x", mcast_dest_noc_end.x if enable_mtp_on_device else 0),
                    ("mcast_eh_dest_noc_end_y", mcast_dest_noc_end.y if enable_mtp_on_device else 0),
                    ("mcast_eh_num_cores", num_mcast_cores if enable_mtp_on_device else 0),
                    (
                        "mcast_eh_data_sender_semaphore",
                        mcast_eh_data_sender_semaphore_id if enable_mtp_on_device else 0,
                    ),
                    (
                        "mcast_eh_data_receiver_semaphore",
                        mcast_eh_data_receiver_semaphore_id if enable_mtp_on_device else 0,
                    ),
                    ("hnorm_ready_cb", hnorm_ready_cb),
                    ("mcast_eh_ready_cb", mcast_eh_ready_cb if enable_mtp_on_device else 0),
                    ("mcast_eh_src_cb", mcast_eh_src_cb if enable_mtp_on_device else 0),
                    ("mcast_eh_dst_cb", mcast_eh_dst_cb if enable_mtp_on_device else 0),
                    ("mcast_eh_dst_num_pages", eh_num_tiles_k if enable_mtp_on_device else 0),
                    ("mcast_eh_data_size_bytes", eh_mcast_data_size_bytes if enable_mtp_on_device else 0),
                    ("mcast_eh_src_num_pages", rms_num_tiles if enable_mtp_on_device else 0),
                    ("reduce_gate_semaphore_id", reduce_gate_semaphore_id if enable_mtp_on_device else 0),
                    ("reduce_gate_num_targets", reduce_num_fabric_and_worker_cores),
                    ("sampling_defer_socket_output", 1 if enable_socket_output else 0),
                    ("argmax_core_noc_x", argmax_core_noc_x if (enable_mtp_on_device or is_mtp_verify_stage) else 0),
                    ("argmax_core_noc_y", argmax_core_noc_y if (enable_mtp_on_device or is_mtp_verify_stage) else 0),
                    ("eh_matmul_num_cores", eh_matmul_num_cores if enable_mtp_on_device else 0),
                    ("mtp_ready_semaphore_id", mtp_ready_semaphore_id),
                    ("metadata_ready_semaphore_id", metadata_ready_semaphore_id),
                    ("gather_dst_cb", eh_gather_dst_cb if enable_mtp_on_device else 0),
                    ("gather_dst_num_pages", eh_gather_dst_num_pages),
                    ("gather_send_total_bytes", eh_gather_send_total_bytes),
                    ("metadata_output_l1_addr", metadata_output_l1_addr),
                    ("is_e_norm_device", 1 if is_e_norm_device else 0),
                    ("eh_norm_slice_offset_bytes", eh_norm_slice_offset_bytes),
                    # [Reduce] BRISC CT args
                    ("reduce_device_role", reduce_device_role),
                    ("reduce_num_tiles", reduce_params["num_tiles"] if enable_reduce_to_one else 0),
                    ("reduce_payload_size_bytes", reduce_params["payload_size_bytes"] if enable_reduce_to_one else 0),
                    ("reduce_local_cb", reduce_local_cb),
                    ("reduce_scratch_cb", reduce_scratch_cb),
                    ("reduce_packet_cb", reduce_packet_cb),
                    ("reduce_num_hops", 1),
                    (
                        "reduce_dst_fabric_node_chip_id",
                        int(reduce_dest_fabric_node_id.chip_id) if enable_reduce_to_one else 0,
                    ),
                    (
                        "reduce_dst_fabric_node_mesh_id",
                        int(reduce_dest_fabric_node_id.mesh_id) if enable_reduce_to_one else 0,
                    ),
                    ("reduce_output_core_noc_x", reduce_output_core_phys_x),
                    ("reduce_output_core_noc_y", reduce_output_core_phys_y),
                    ("reduce_num_workers", reduce_params["num_workers_per_column"] if enable_reduce_to_one else 0),
                    # [Reduce] Pre-reduce gate: BRISC on argmax_final_core signals 2 fabric cores after sampling
                    ("reduce_fc_noc_0_x", reduce_fc_nocs[0][0]),
                    ("reduce_fc_noc_0_y", reduce_fc_nocs[0][1]),
                    ("reduce_fc_noc_1_x", reduce_fc_nocs[1][0]),
                    ("reduce_fc_noc_1_y", reduce_fc_nocs[1][1]),
                    ("mtp_token_l1_addr", mtp_token_l1_addr),
                    ("mtp_input_core_noc_x", mtp_input_core_noc_x),
                    ("mtp_input_core_noc_y", mtp_input_core_noc_y),
                    ("mtp_argmax_output_addr", mtp_argmax_output_l1_addr),
                    ("hnorm_ready_cb", hnorm_ready_cb),
                ]
                brisc_named_compile_time_args.extend(bcast_config.get_brisc_named_ct_args(coord))
                if mtp_bcast_config is not None:
                    brisc_named_compile_time_args.extend(
                        [(f"mtp_{k}", v) for k, v in mtp_bcast_config.get_brisc_named_ct_args(coord)]
                    )
                else:
                    brisc_named_compile_time_args.extend(
                        [
                            ("mtp_bcast_data_cb_id", 0),
                            ("mtp_bcast_num_pages_to_read", 0),
                            ("mtp_bcast_is_root", 0),
                            ("mtp_bcast_use_socket", 0),
                        ]
                    )

                # ================================================================
                # TRISC compile-time args
                # ================================================================
                trisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("enable_argmax", 1),
                    ("input_socket_mode", input_socket_mode),
                    ("rmsnorm_input_cb", rmsnorm_input_cb),
                    ("rmsnorm_gamma_cb", rmsnorm_gamma_cb),
                    ("rmsnorm_output_cb", mcast_src_cb),
                    ("rmsnorm_fp32_acc", 1 if fp32_dest_acc_en else 0),
                    ("rmsnorm_num_tiles", rms_num_tiles),
                    ("rmsnorm_rsqrt_fast_approx", 1 if rsqrt_fast_approx else 0),
                    ("matmul_in0", mcast_dst_cb),  # Matmul reads from mcast destination
                    ("matmul_in1", matmul_in1_cb),
                    ("matmul_out", matmul_out_cb),
                    ("matmul_k_num_tiles", num_tiles_k),
                    ("matmul_out_w", out_w_per_core),
                    ("persistent_mode", 1 if persistent_mode else 0),
                    ("termination_semaphore_addr", termination_global_sem_addr),
                    ("fabric_gate_bcast_turn_semaphore_id", fabric_gate_bcast_turn_semaphore_id),
                    ("fabric_gate_argmax_turn_semaphore_id", fabric_gate_argmax_turn_semaphore_id),
                    ("fabric_gate_bcast_noc_x", int(bcast_worker_core_phys.x)),
                    ("fabric_gate_bcast_noc_y", int(bcast_worker_core_phys.y)),
                    ("fabric_gate_argmax_noc_x", int(final_core_phys.x)),
                    ("fabric_gate_argmax_noc_y", int(final_core_phys.y)),
                    ("mesh_row", row),
                    ("mesh_col", col),
                    # [MTP] h_rmsnorm and e_rmsnorm CTArgs; is_eh_matmul_core in base so non-exit gets 0.
                    # Both RMSNorms write directly to mcast_eh_src_cb to avoid concat copy:
                    # h_rmsnorm writes first half (h_tiles), e_rmsnorm writes second half (e_tiles)
                    ("rmsnorm_h_input_cb", rmsnorm_input_cb),
                    ("rmsnorm_h_gamma_cb", h_gamma_cb),
                    ("rmsnorm_h_output_cb", mcast_eh_src_cb),
                    ("rmsnorm_h_num_tiles", rms_num_tiles),
                    ("rmsnorm_e_input_cb", embedding_cb),
                    ("rmsnorm_e_gamma_cb", e_gamma_cb),
                    ("rmsnorm_e_output_cb", mcast_eh_src_cb),
                    ("rmsnorm_e_num_tiles", e_num_tiles if enable_mtp_on_device else 0),
                    # [MTP] EH matmul CTArgs
                    ("matmul_eh_in0", mcast_eh_dst_cb),
                    ("matmul_eh_in1", matmul_eh_cb),
                    ("matmul_eh_out", matmul_out_eh_cb),
                    ("matmul_eh_k_num_tiles", eh_num_tiles_k if enable_mtp_on_device else 0),
                    ("matmul_eh_out_w", eh_out_w_per_core if enable_mtp_on_device else 0),
                    # [MTP] EH matmul DRAM streaming CTArgs (TRISC compute)
                    ("matmul_eh_subblock_w", eh_subblock_w if enable_mtp_on_device else 0),
                    ("matmul_eh_subblock_k", eh_subblock_k if enable_mtp_on_device else 0),
                    ("matmul_eh_num_subblocks_k", eh_num_subblocks_k if enable_mtp_on_device else 0),
                    ("reduce_gate_semaphore_id", reduce_gate_semaphore_id if enable_mtp_on_device else 0),
                    ("argmax_core_noc_x", argmax_core_noc_x if (enable_mtp_on_device or is_mtp_verify_stage) else 0),
                    ("argmax_core_noc_y", argmax_core_noc_y if (enable_mtp_on_device or is_mtp_verify_stage) else 0),
                    # [Reduce] TRISC CT args
                    ("reduce_device_role", reduce_device_role),
                    ("reduce_num_tiles", reduce_params["num_tiles"] if enable_reduce_to_one else 0),
                    ("reduce_local_cb", reduce_local_cb),
                    ("reduce_received_cb", reduce_received_cb),
                    ("reduce_output_cb", reduce_output_cb),
                    ("reduce_scratch_cb", reduce_scratch_cb),
                    ("hnorm_ready_cb", hnorm_ready_cb),
                    ("is_e_norm_device", 1 if is_e_norm_device else 0),
                    # ── Sampling kernel additions (sampling.hpp ComputeCTArgs) ──
                    ("sampling_softmax_in_cb", sampling_softmax_in_cb),
                    ("sampling_softmax_out_cb", sampling_softmax_out_cb),
                    ("sampling_softmax_exp_cb", sampling_softmax_exp_cb),
                    ("sampling_softmax_sub_cb", sampling_softmax_sub_cb),
                    ("sampling_max_cb", sampling_max_cb),
                    ("sampling_sum_cb", sampling_sum_cb),
                    ("sampling_scaler_cb", sampling_scaler_cb),
                    ("sampling_temp_cb", sampling_temp_cb),
                    ("sampling_rand_cb", sampling_rand_cb),
                    ("sampling_seed", int(seed) & 0xFFFFFFFF),
                    ("sampling_topk_k", sampling_topk_k_value),
                    ("sampling_mesh_mode", argmax_mesh_mode),
                    ("sampling_stage1_receiver", argmax_stage1_receiver),
                    ("sampling_stage2_receiver", argmax_stage2_receiver),
                    ("sampling_num_values", argmax_num_values),
                    ("sampling_num_senders", argmax_num_senders),
                    ("sampling_topk_in_scores_cb", sampling_topk_in_scores_cb),
                    ("sampling_topk_in_indices_cb", sampling_topk_in_indices_cb),
                    ("sampling_topk_out_scores_cb", sampling_topk_out_scores_cb),
                    ("sampling_topk_out_indices_cb", sampling_topk_out_indices_cb),
                    ("sampling_mesh_stage_scores_cb", sampling_mesh_stage_scores_cb),
                    ("sampling_mesh_stage_indices_cb", sampling_mesh_stage_indices_cb),
                    ("sampling_stage1_row_elements", sampling_stage1_num_slots * sampling_topk_min_alignment),
                    ("sampling_stage1_num_input_tiles", sampling_stage1_mesh_tiles),
                    ("sampling_stage2_row_elements", sampling_stage2_num_slots * sampling_topk_min_alignment),
                    ("sampling_stage2_num_input_tiles", sampling_stage2_mesh_tiles),
                ]

                # ================================================================
                # CCL Broadcast common runtime args
                # ================================================================
                # [Sampling] argmax's single scratch_addr (RT) is dropped here:
                # in sampling.hpp the scores_scratch_addr / indices_scratch_addr
                # are passed as CT args (added to the NCRISC named CT-arg list
                # below). RT layout follows sampling.hpp::ReaderArgs exactly:
                #   scores_addr, indices_addr, output_addr, final_noc_x,
                #   final_noc_y, global_sem_addr, global_stage2_sem_addr
                sampling_scores_scratch_addr = (
                    int(scores_scratch_tensors_per_device[device_idx].buffer_address())
                    if (not skip_ccl and scores_scratch_tensors_per_device is not None)
                    else 0
                )
                sampling_indices_scratch_addr = (
                    int(indices_scratch_tensors_per_device[device_idx].buffer_address())
                    if (not skip_ccl and indices_scratch_tensors_per_device is not None)
                    else 0
                )
                sampling_scores_addr = int(output_tensor_device.buffer_address())

                ncrisc_bcast_common_args = (
                    bcast_config.get_ncrisc_common_rt_args(coord)
                    + [
                        sampling_scores_addr,
                        int(indices_tensor_device.buffer_address()),
                        int(output_index_tensor_device.buffer_address()),
                        int(final_core_phys.x),
                        int(final_core_phys.y),
                        global_sem_addr,
                        global_stage2_sem_addr,
                    ]
                    + (reduce_params["sem_addrs"][:3] if enable_reduce_to_one else [0, 0, 0])
                    + (
                        mtp_bcast_config.get_ncrisc_common_rt_args(coord)
                        if mtp_bcast_config is not None
                        else [0, 0, 0, 0, 0]
                    )
                )

                brisc_bcast_common_args = (
                    bcast_config.get_brisc_common_rt_args(coord)
                    + [
                        int(final_core_phys.x),
                        int(final_core_phys.y),
                        int(socket_output.get_config_buffer_address()) if enable_socket_output else 0,
                        persistent_enable,
                        int(persistent_target_input_core_phys.x),
                        int(persistent_target_input_core_phys.y),
                        int(persistent_target_node.mesh_id),
                        int(persistent_target_node.chip_id),
                        persistent_next_iter_global_sem_addr,
                    ]
                    + (mtp_bcast_config.get_brisc_common_rt_args(coord) if mtp_bcast_config is not None else [0, 0, 0])
                )

                # ================================================================
                # Circular buffer descriptors
                # ================================================================
                # CB 0: RMSNorm input source — In multi-device mode, backed by intermediate_tensor
                #       (where CCL broadcast placed the data). In single-device mode,
                #       backed by input_tensor directly.
                rmsnorm_input_backing_tensor = input_tensor_device if skip_ccl else intermediate_tensor_device
                rmsnorm_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    rmsnorm_input_cb, rmsnorm_input_backing_tensor
                )
                rms_tile_descriptor = ttnn.TileDescriptor(rms_interpreted_tile)
                rmsnorm_input_cb_descriptor.format_descriptors[0].tile = rms_tile_descriptor
                rmsnorm_input_cb_descriptor.format_descriptors[0].page_size = rms_tile_size
                rmsnorm_input_cb_descriptor.total_size = (rms_num_tiles) * rms_tile_size

                # CB 7: RMSNorm gamma — tensor-backed on sender core (None when folded into vocab weights).
                rmsnorm_gamma_cb_descriptor = None
                if gamma_tensor_device is not None:
                    rmsnorm_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        rmsnorm_gamma_cb, gamma_tensor_device
                    )
                    rmsnorm_gamma_cb_descriptor.format_descriptors[0].tile = rms_tile_descriptor
                    rmsnorm_gamma_cb_descriptor.format_descriptors[0].page_size = rms_tile_size

                # CB 8: RMSNorm output on sender core (mcast source)
                rms_out_tile_descriptor = ttnn.TileDescriptor(rms_interpreted_tile)
                rmsnorm_out_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=mcast_src_cb,
                    data_format=data_format,
                    page_size=rms_tile_size,
                    tile=rms_out_tile_descriptor,
                )
                rmsnorm_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    mcast_src_cb,
                    rmsnorm_input_backing_tensor,
                    address_offset=0,
                    total_size=rms_num_tiles * rms_tile_size,
                    core_ranges=mcast_sender_core_grid,
                )
                rmsnorm_out_cb_descriptor.format_descriptors = [rmsnorm_out_cb_format]

                # CB 1: Mcast destination — on receiver cores this is the matmul in0 buffer.
                mcast_dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
                mcast_dst_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=mcast_dst_cb,
                    data_format=data_format,
                    page_size=input_tile_size,
                    tile=mcast_dst_tile_descriptor,
                )
                mcast_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    mcast_dst_cb,
                    rmsnorm_input_backing_tensor,
                    address_offset=0,
                    total_size=num_tiles_k * input_tile_size,
                    core_ranges=all_cores,
                )
                mcast_dst_cb_descriptor.format_descriptors = [mcast_dst_cb_format]

                # CB 2: Matmul weights — vocab_tensor, tensor-backed on matmul cores
                matmul_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_in1_cb, vocab_tensor_device)

                # CB 16: Matmul output — tensor-backed on matmul cores
                matmul_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_out_cb, output_tensor_device)

                # [MTP] CB descriptors (only if is_mtp_base_stage)
                mtp_cb_descriptors = []
                if enable_mtp_on_device:
                    # CB 11: h_gamma - tensor-backed on sender core (h_norm devices only, None when folded)
                    h_gamma_cb_descriptor = None
                    if not is_e_norm_device and h_gamma_tensors_per_device is not None:
                        h_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                            h_gamma_cb, h_gamma_tensors_per_device[device_idx]
                        )
                        h_gamma_cb_descriptor.format_descriptors[0].tile = rms_tile_descriptor
                        h_gamma_cb_descriptor.format_descriptors[0].page_size = rms_tile_size

                    # CB 12: e_gamma - tensor-backed on sender core (e_norm devices only, None when folded)
                    e_gamma_cb_descriptor = None
                    if is_e_norm_device and e_gamma_tensors_per_device is not None:
                        e_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                            e_gamma_cb, e_gamma_tensors_per_device[device_idx]
                        )
                        e_gamma_cb_descriptor.format_descriptors[0].tile = rms_tile_descriptor
                        e_gamma_cb_descriptor.format_descriptors[0].page_size = rms_tile_size

                    # CB 10: Embedding — overlapped onto fused buffer on sender core
                    embedding_cb_descriptor = None
                    if is_e_norm_device:
                        embedding_tile_descriptor = ttnn.TileDescriptor(rms_interpreted_tile)
                        embedding_cb_format = ttnn.CBFormatDescriptor(
                            buffer_index=embedding_cb,
                            data_format=data_format,
                            page_size=rms_tile_size,
                            tile=embedding_tile_descriptor,
                        )
                        embedding_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                            embedding_cb,
                            rmsnorm_input_backing_tensor,
                            address_offset=0,
                            total_size=rms_num_tiles * rms_tile_size,
                            core_ranges=mcast_sender_core_grid,
                        )
                        embedding_cb_descriptor.format_descriptors = [embedding_cb_format]

                    # CB 15: mcast_eh_src — overlapped onto fused buffer on sender core.
                    # Sized for 7 tiles of 32x32; mcast sends a 1792-datum slice via byte offset.
                    mcast_eh_tile_descriptor = ttnn.TileDescriptor(rms_interpreted_tile)
                    mcast_eh_src_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=mcast_eh_src_cb,
                        data_format=data_format,
                        page_size=rms_tile_size,
                        tile=mcast_eh_tile_descriptor,
                    )
                    mcast_eh_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        mcast_eh_src_cb,
                        rmsnorm_input_backing_tensor,
                        address_offset=0,
                        total_size=rms_num_tiles * rms_tile_size,
                        core_ranges=mcast_sender_core_grid,
                    )
                    mcast_eh_src_cb_descriptor.format_descriptors = [mcast_eh_src_cb_format]

                    # CB 18: mcast_eh_dst — overlapped onto fused buffer on sender core,
                    # normal intermediate on receiver cores.
                    eh_dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
                    mcast_eh_dst_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=mcast_eh_dst_cb,
                        data_format=data_format,
                        page_size=input_tile_size,
                        tile=eh_dst_tile_descriptor,
                    )
                    mcast_eh_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        mcast_eh_dst_cb,
                        rmsnorm_input_backing_tensor,
                        address_offset=0,
                        total_size=eh_num_tiles_k * input_tile_size,
                        core_ranges=all_cores,
                    )
                    mcast_eh_dst_cb_descriptor.format_descriptors = [mcast_eh_dst_cb_format]

                    # CB 9: EH projection weights (in1) buffer - CB-backed working buffer for DRAM streaming
                    eh_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=matmul_eh_cb,
                        data_format=eh_dtype,
                        page_size=eh_proj_tile_size,
                        tile=ttnn.TileDescriptor(eh_proj_tile),
                    )
                    matmul_eh_cb_descriptor = ttnn.CBDescriptor(
                        total_size=eh_in1_CB_size,
                        core_ranges=eh_matmul_core_grid,
                        format_descriptors=[eh_cb_format],
                    )

                    # CB 17: EH matmul output — offset 0 in fused buffer on matmul cores
                    matmul_out_eh_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        matmul_out_eh_cb, eh_mm_fused_per_device[device_idx], address_offset=0
                    )
                    matmul_out_eh_cb_descriptor.core_ranges = eh_matmul_core_grid
                    matmul_out_eh_cb_descriptor.total_size = eh_out_w_per_core * eh_output_tile_size

                    # CB 19: EH output gather — offset 0 in fused buffer on argmax core
                    eh_gather_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        eh_gather_dst_cb, eh_mm_fused_per_device[device_idx], address_offset=0
                    )
                    eh_gather_cb_descriptor.core_ranges = ttnn.CoreRangeSet(
                        [ttnn.CoreRange(argmax_final_core, argmax_final_core)]
                    )
                    eh_gather_cb_descriptor.total_size = (eh_gather_dst_num_pages + 4) * eh_output_tile_size

                    # CB 37: Sync CB for h_rmsnorm and lm head norm on TRISC
                    hnorm_ready_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=hnorm_ready_cb,
                        data_format=ttnn.uint32,
                        page_size=16,
                    )
                    hnorm_ready_cb_descriptor = ttnn.CBDescriptor(
                        total_size=16,
                        core_ranges=mcast_sender_core_grid,
                        format_descriptors=[hnorm_ready_cb_format],
                    )

                    # CB 38: NCRISC -> BRISC sync on input_core. Pushed by NCRISC
                    # right after `token_bcast_receiver` returns; consumed by
                    # BRISC immediately before issuing `mcast_eh`.
                    mcast_eh_ready_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=mcast_eh_ready_cb,
                        data_format=ttnn.uint32,
                        page_size=16,
                    )
                    mcast_eh_ready_cb_descriptor = ttnn.CBDescriptor(
                        total_size=16,
                        core_ranges=mcast_sender_core_grid,
                        format_descriptors=[mcast_eh_ready_cb_format],
                    )

                    mtp_cb_descriptors = [
                        mcast_eh_src_cb_descriptor,
                        matmul_eh_cb_descriptor,
                        matmul_out_eh_cb_descriptor,
                        eh_gather_cb_descriptor,
                        mcast_eh_dst_cb_descriptor,
                        hnorm_ready_cb_descriptor,
                        mcast_eh_ready_cb_descriptor,
                    ]
                    if h_gamma_cb_descriptor is not None:
                        mtp_cb_descriptors.append(h_gamma_cb_descriptor)
                    if e_gamma_cb_descriptor is not None:
                        mtp_cb_descriptors.append(e_gamma_cb_descriptor)
                    if embedding_cb_descriptor is not None:
                        mtp_cb_descriptors.append(embedding_cb_descriptor)

                # CB list
                cbs_list = [
                    rmsnorm_input_cb_descriptor,
                    matmul_in1_cb_descriptor,
                    rmsnorm_out_cb_descriptor,
                    matmul_out_cb_descriptor,
                    mcast_dst_cb_descriptor,
                ]
                if rmsnorm_gamma_cb_descriptor is not None:
                    cbs_list.append(rmsnorm_gamma_cb_descriptor)
                if enable_mtp_on_device:
                    cbs_list.extend(mtp_cb_descriptors)
                if enable_argmax:
                    # ----------------------------------------------------------------
                    # Sampling (top-K + top-P + softmax) CB descriptors.
                    # Layout + sizing is copied from
                    #   models/demos/deepseek_v3_b1/micro_ops/sampling/op.py
                    # :: _op_mesh_topk (the multi-device mesh path).
                    # ----------------------------------------------------------------
                    sampling_l1_alignment = 16
                    sampling_bf16_tile_size = 2 * 32 * 32  # 2048
                    sampling_uint32_tile_size = 4 * 32 * 32  # 4096
                    sampling_topk_min_alignment = 32

                    def _sampling_round_up(value: int, alignment: int) -> int:
                        return ((value + alignment - 1) // alignment) * alignment

                    # Winner = K scores (bf16) + K indices (uint32), each padded up
                    # to 16B L1 alignment. Matches micro-op's topk_min_alignment=32.
                    sampling_topk_scores_slot_bytes = _sampling_round_up(
                        sampling_topk_min_alignment * 2, sampling_l1_alignment
                    )
                    sampling_topk_indices_slot_bytes = _sampling_round_up(
                        sampling_topk_min_alignment * 4, sampling_l1_alignment
                    )
                    sampling_winner_page_bytes = sampling_topk_scores_slot_bytes + sampling_topk_indices_slot_bytes

                    # Phase-1 input = local per-core scores shard (num_values = n_per_core);
                    # Phase-2 input = gathered top-K scores from all cores on this device.
                    sampling_phase1_tiles = (n_per_core + 1023) // 1024
                    sampling_phase2_tiles = (sampling_topk_min_alignment * argmax_num_senders + 1023) // 1024
                    sampling_total_input_tiles = sampling_phase1_tiles + sampling_phase2_tiles

                    # Mesh-stage scratch sizing (stage-1 merges across mesh rows,
                    # stage-2 merges across mesh cols on the final receiver core).
                    sampling_stage1_num_slots = mesh_rows
                    sampling_stage2_num_slots = mesh_cols
                    sampling_stage1_mesh_tiles = (
                        sampling_stage1_num_slots * sampling_topk_min_alignment + 1023
                    ) // 1024
                    sampling_stage2_mesh_tiles = (
                        sampling_stage2_num_slots * sampling_topk_min_alignment + 1023
                    ) // 1024
                    sampling_total_mesh_stage_tiles = sampling_stage1_mesh_tiles + sampling_stage2_mesh_tiles

                    sampling_final_core_crs = ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)])

                    # --- Per-core CBs (all matmul / argmax_core_grid cores) -----------
                    sampling_winner_cb_descriptor = ttnn.CBDescriptor(
                        total_size=sampling_winner_page_bytes,
                        core_ranges=argmax_core_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=sampling_winner_cb,
                                data_format=ttnn.uint32,
                                page_size=sampling_winner_page_bytes,
                            )
                        ],
                    )

                    if enable_mtp_on_device:
                        sampling_topk_in_scores_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                            sampling_topk_in_scores_cb,
                            eh_mm_fused_per_device[device_idx],
                            address_offset=0,
                            total_size=sampling_total_input_tiles * sampling_bf16_tile_size,
                            core_ranges=argmax_core_grid,
                        )
                        sampling_topk_in_scores_cb_descriptor.format_descriptors = [
                            ttnn.CBFormatDescriptor(
                                buffer_index=sampling_topk_in_scores_cb,
                                data_format=ttnn.bfloat16,
                                page_size=sampling_bf16_tile_size,
                            )
                        ]
                        sampling_topk_in_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                            sampling_topk_in_indices_cb,
                            eh_mm_fused_per_device[device_idx],
                            address_offset=sampling_total_input_tiles * sampling_bf16_tile_size,
                            total_size=sampling_total_input_tiles * sampling_uint32_tile_size,
                            core_ranges=argmax_core_grid,
                        )
                        sampling_topk_in_indices_cb_descriptor.format_descriptors = [
                            ttnn.CBFormatDescriptor(
                                buffer_index=sampling_topk_in_indices_cb,
                                data_format=ttnn.uint32,
                                page_size=sampling_uint32_tile_size,
                            )
                        ]
                    else:
                        sampling_topk_in_scores_cb_descriptor = ttnn.CBDescriptor(
                            total_size=sampling_total_input_tiles * sampling_bf16_tile_size,
                            core_ranges=argmax_core_grid,
                            format_descriptors=[
                                ttnn.CBFormatDescriptor(
                                    buffer_index=sampling_topk_in_scores_cb,
                                    data_format=ttnn.bfloat16,
                                    page_size=sampling_bf16_tile_size,
                                )
                            ],
                        )
                        sampling_topk_in_indices_cb_descriptor = ttnn.CBDescriptor(
                            total_size=sampling_total_input_tiles * sampling_uint32_tile_size,
                            core_ranges=argmax_core_grid,
                            format_descriptors=[
                                ttnn.CBFormatDescriptor(
                                    buffer_index=sampling_topk_in_indices_cb,
                                    data_format=ttnn.uint32,
                                    page_size=sampling_uint32_tile_size,
                                )
                            ],
                        )

                    sampling_topk_out_scores_cb_descriptor = ttnn.CBDescriptor(
                        total_size=sampling_bf16_tile_size,
                        core_ranges=argmax_core_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=sampling_topk_out_scores_cb,
                                data_format=ttnn.bfloat16,
                                page_size=sampling_bf16_tile_size,
                            )
                        ],
                    )
                    sampling_topk_out_indices_cb_descriptor = ttnn.CBDescriptor(
                        total_size=sampling_uint32_tile_size,
                        core_ranges=argmax_core_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=sampling_topk_out_indices_cb,
                                data_format=ttnn.uint32,
                                page_size=sampling_uint32_tile_size,
                            )
                        ],
                    )

                    cbs_list.extend(
                        [
                            sampling_winner_cb_descriptor,
                            sampling_topk_in_scores_cb_descriptor,
                            sampling_topk_in_indices_cb_descriptor,
                            sampling_topk_out_scores_cb_descriptor,
                            sampling_topk_out_indices_cb_descriptor,
                        ]
                    )

                    # --- Final-core-only softmax / top-P / RNG compute CBs ------------
                    # Each is a single bf16 tile (2 KB). Mirrors the final-device
                    # block in _op_mesh_topk under `is_final_mesh_device`.
                    for _sampling_compute_cb_id in (
                        sampling_softmax_in_cb,
                        sampling_softmax_out_cb,
                        sampling_softmax_exp_cb,
                        sampling_softmax_sub_cb,
                        sampling_max_cb,
                        sampling_sum_cb,
                        sampling_scaler_cb,
                        sampling_temp_cb,
                        sampling_rand_cb,
                    ):
                        cbs_list.append(
                            ttnn.CBDescriptor(
                                total_size=sampling_bf16_tile_size,
                                core_ranges=sampling_final_core_crs,
                                format_descriptors=[
                                    ttnn.CBFormatDescriptor(
                                        buffer_index=_sampling_compute_cb_id,
                                        data_format=ttnn.bfloat16,
                                        page_size=sampling_bf16_tile_size,
                                    )
                                ],
                            )
                        )

                    # --- Mesh-stage scratch CBs (stage-1 / stage-2 receiver) ----------
                    # Back these CBs with the caller-supplied scratch tensors so that
                    # NCRISC write_topk_slot (which writes to scratch_addr) and TRISC
                    # cb_wait_front (which reads from the CB) access the same memory.
                    #
                    # The CB total_size must match the number of tiles the device
                    # actually pushes/pops per iteration so the FIFO wraps correctly.
                    # Stage-1-only devices push 1 tile; devices running both stages
                    # push 2 tiles.
                    if mesh_rows > 1 or mesh_cols > 1:
                        scores_scratch_device = scores_scratch_tensors_per_device[device_idx]
                        indices_scratch_device = indices_scratch_tensors_per_device[device_idx]

                        device_mesh_tiles = sampling_stage1_mesh_tiles
                        if argmax_stage2_receiver:
                            device_mesh_tiles += sampling_stage2_mesh_tiles

                        mesh_scores_cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
                            sampling_mesh_stage_scores_cb,
                            scores_scratch_device,
                            address_offset=0,
                            total_size=device_mesh_tiles * sampling_bf16_tile_size,
                        )
                        mesh_scores_cb_desc.core_ranges = sampling_final_core_crs
                        mesh_scores_cb_desc.format_descriptors = [
                            ttnn.CBFormatDescriptor(
                                buffer_index=sampling_mesh_stage_scores_cb,
                                data_format=ttnn.bfloat16,
                                page_size=sampling_bf16_tile_size,
                                tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                            )
                        ]
                        cbs_list.append(mesh_scores_cb_desc)

                        mesh_indices_cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
                            sampling_mesh_stage_indices_cb,
                            indices_scratch_device,
                            address_offset=0,
                            total_size=device_mesh_tiles * sampling_uint32_tile_size,
                        )
                        mesh_indices_cb_desc.core_ranges = sampling_final_core_crs
                        mesh_indices_cb_desc.format_descriptors = [
                            ttnn.CBFormatDescriptor(
                                buffer_index=sampling_mesh_stage_indices_cb,
                                data_format=ttnn.uint32,
                                page_size=sampling_uint32_tile_size,
                                tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                            )
                        ]
                        cbs_list.append(mesh_indices_cb_desc)

                    # --- Deferred socket output CB (final core, same page/role
                    # as the old argmax_socket_cb) ------------------------------------
                    if enable_socket_output:
                        sampling_socket_cb_descriptor = ttnn.CBDescriptor(
                            total_size=socket_page_size_bytes,
                            core_ranges=sampling_final_core_crs,
                            format_descriptors=[
                                ttnn.CBFormatDescriptor(
                                    buffer_index=sampling_socket_cb,
                                    data_format=ttnn.uint32,
                                    page_size=socket_page_size_bytes,
                                )
                            ],
                        )
                        cbs_list.append(sampling_socket_cb_descriptor)

                bcast_pkt_cb_descriptor = bcast_config.get_cb_descriptor(coord)
                if bcast_pkt_cb_descriptor is not None:
                    cbs_list.append(bcast_pkt_cb_descriptor)

                if mtp_bcast_config is not None:
                    mtp_bcast_cb_descriptor = mtp_bcast_config.get_cb_descriptor(coord)
                    if mtp_bcast_cb_descriptor is not None:
                        cbs_list.append(mtp_bcast_cb_descriptor)

                # [Reduce] CB descriptors
                if enable_reduce_to_one:
                    rp = reduce_params
                    reduce_worker_cores_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in rp["worker_cores"]])
                    reduce_all_cores_set = ttnn.CoreRangeSet(
                        [ttnn.CoreRange(c, c) for c in rp["worker_cores"]]
                        + [ttnn.CoreRange(c, c) for c in rp["fabric_cores"]]
                    )
                    reduce_tile_desc = ttnn.TileDescriptor(32, 32)
                    # CB 20: received (3 pages for 3 reduce rounds) — offset past CB17 in fused buffer
                    reduce_cb_recv = ttnn.cb_descriptor_from_sharded_tensor(
                        reduce_received_cb,
                        rp["fused_per_device"][device_idx],
                        address_offset=rp["cb20_offset_bytes"],
                    )
                    reduce_cb_recv.core_ranges = reduce_worker_cores_set
                    reduce_cb_recv.total_size = 3 * rp["payload_size_bytes"]
                    reduce_cb_recv.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=reduce_received_cb,
                            data_format=ttnn.bfloat16,
                            page_size=rp["payload_size_bytes"],
                            tile=reduce_tile_desc,
                        )
                    ]
                    # CB 18 (scratch) already allocated as mcast_eh_dst_cb — reused, no new descriptor
                    # CB 22: packet (fabric send buffer) — both worker + fabric cores
                    reduce_cb_pkt = ttnn.CBDescriptor(
                        total_size=rp["slot_size_bytes"] * rp["num_workers_per_column"],
                        core_ranges=reduce_all_cores_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=reduce_packet_cb,
                                data_format=ttnn.bfloat16,
                                page_size=rp["slot_size_bytes"],
                            )
                        ],
                    )
                    # CB 21: reduce local — aliased to CB17 (offset 0 in fused buffer),
                    # but with page_size=payload_size_bytes so reduce sees full shard as 1 page.
                    # The TRISC bridge pops CB17 (28 tiles) and pushes CB21 (1 page) between
                    # the EH matmul and reduce, so reduce operates with num_compute_tiles=1.
                    reduce_local_cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
                        reduce_local_cb, rp["fused_per_device"][device_idx], address_offset=0
                    )
                    reduce_local_cb_desc.core_ranges = reduce_worker_cores_set
                    reduce_local_cb_desc.total_size = rp["payload_size_bytes"]
                    reduce_local_cb_desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=reduce_local_cb,
                            data_format=ttnn.bfloat16,
                            page_size=rp["payload_size_bytes"],
                            tile=reduce_tile_desc,
                        )
                    ]
                    # CB 23: reduce scratch — dedicated intermediate for reduce TRISC→BRISC handoff.
                    # Cannot reuse CB18 (mcast_eh_dst_cb): BRISC enters reduce before TRISC's
                    # DSMM consumes the mcast data in CB18, causing a race condition.
                    reduce_scratch_cb_desc = ttnn.CBDescriptor(
                        total_size=rp["num_tiles"] * rp["compute_tile_size"],
                        core_ranges=reduce_worker_cores_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=reduce_scratch_cb,
                                data_format=ttnn.bfloat16,
                                page_size=rp["compute_tile_size"],
                                tile=reduce_tile_desc,
                            )
                        ],
                    )
                    cbs_list.extend([reduce_cb_recv, reduce_cb_pkt, reduce_local_cb_desc, reduce_scratch_cb_desc])

                # ================================================================
                # Semaphore descriptors (for intra-device mcast)
                # ================================================================
                semaphore_descriptors = [
                    ttnn.SemaphoreDescriptor(
                        id=mcast_data_sender_semaphore_id,
                        core_ranges=all_cores,
                        initial_value=0,
                    ),
                    ttnn.SemaphoreDescriptor(
                        id=mcast_data_receiver_semaphore_id,
                        core_ranges=all_cores,
                        initial_value=0,
                    ),
                ]
                if enable_argmax:
                    semaphore_descriptors.extend(
                        [
                            ttnn.SemaphoreDescriptor(
                                id=argmax_receiver_semaphore_id,
                                core_ranges=argmax_core_grid,
                                initial_value=0,
                            ),
                            ttnn.SemaphoreDescriptor(
                                id=argmax_local_ready_semaphore_id,
                                core_ranges=argmax_core_grid,
                                initial_value=0,
                            ),
                            ttnn.SemaphoreDescriptor(
                                id=fabric_gate_bcast_turn_semaphore_id,
                                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)]),
                                initial_value=1,
                            ),
                            ttnn.SemaphoreDescriptor(
                                id=fabric_gate_argmax_turn_semaphore_id,
                                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)]),
                                initial_value=0,
                            ),
                        ]
                    )
                # Semaphores for MTP in Base Stage
                if enable_mtp_on_device:
                    semaphore_descriptors.extend(
                        [
                            ttnn.SemaphoreDescriptor(
                                id=mtp_ready_semaphore_id,
                                core_ranges=mcast_sender_core_grid,
                                initial_value=0,
                            ),
                            ttnn.SemaphoreDescriptor(
                                id=mcast_eh_data_receiver_semaphore_id,
                                core_ranges=all_cores,
                                initial_value=0,
                            ),
                            ttnn.SemaphoreDescriptor(
                                id=reduce_gate_semaphore_id,
                                core_ranges=all_cores,
                                initial_value=0,
                            ),
                        ]
                    )
                if is_exit_device:
                    semaphore_descriptors.extend(
                        [
                            ttnn.SemaphoreDescriptor(
                                id=metadata_ready_semaphore_id,
                                core_ranges=all_cores,
                                initial_value=0,
                            ),
                        ]
                    )
                # ================================================================
                # Unified kernel descriptor
                # ================================================================
                # Broadcast contributes the current define set. If this fused op
                # adds extra defines later, merge/de-dupe at the op layer.
                kernel_defines = merge_kernel_defines(
                    bcast_config.get_kernel_defines(coord),
                    [("ENABLE_SOCKET_READER", "1")] if enable_socket_input else [],
                )
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/lm_head_sampling/kernels/lm_head_sampling_kernel.cpp",
                    core_ranges=all_cores,
                    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
                    brisc_named_compile_time_args=brisc_named_compile_time_args,
                    trisc_named_compile_time_args=trisc_named_compile_time_args,
                    defines=kernel_defines,
                    ncrisc_common_runtime_args=ncrisc_bcast_common_args,
                    brisc_common_runtime_args=brisc_bcast_common_args,
                    noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=fp32_dest_acc_en,
                        dst_full_sync_en=fp32_dest_acc_en,
                    ),
                    trisc_common_runtime_args=[
                        epsilon_packed,
                        scalar_packed,
                        float_to_uint32(1.0 / math.sqrt(float(embedding_dim))) if enable_mtp_on_device else 0,
                    ],
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_input_core",
                            core_range=mcast_sender_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_mcast_receiver_core",
                            core_range=mcast_receiver_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_mcast_grid_core",
                            core_range=mcast_grid_set,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_matmul_core",
                            core_range=matmul_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_rmsnorm_core",
                            core_range=mcast_sender_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="sampling_is_active_core",
                            core_range=argmax_core_grid,
                            value=1 if enable_argmax else 0,
                            other_value=0,
                        ),
                        # reduce to one core descriptors
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_reduce_worker_core",
                            core_range=eh_matmul_core_grid if enable_reduce_to_one else ttnn.CoreRangeSet([]),
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_reduce_fabric_core",
                            core_range=(
                                ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in reduce_params["fabric_cores"]])
                                if enable_reduce_to_one
                                else ttnn.CoreRangeSet([])
                            ),
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="sampling_is_final_core",
                            core_range=argmax_final_core,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="sampling_mesh_sender_core",
                            core_range=argmax_final_core,
                            value=1 if is_argmax_mesh_sender_core else 0,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="enable_mtp",
                            core_range=all_cores,
                            value=1 if enable_mtp_on_device else 0,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="fold_rmsnorm",
                            core_range=all_cores,
                            value=1 if fold_rmsnorm else 0,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_mtp_base_stage",
                            core_range=all_cores,
                            value=1 if is_mtp_base_stage else 0,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_mtp_verify_stage",
                            core_range=all_cores,
                            value=1 if is_mtp_verify_stage else 0,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_exit_device",
                            core_range=all_cores,
                            value=1 if is_exit_device else 0,
                            other_value=0,
                        ),
                    ]
                    + (
                        [
                            UnifiedCompileTimeCoreDescriptor(
                                named_compile_time_arg="is_eh_matmul_core",
                                core_range=eh_matmul_core_grid,
                                value=1,
                                other_value=0,
                            ),
                        ]
                        if enable_mtp_on_device
                        else []
                    ),
                    per_core_compile_time_descriptors=(
                        (
                            []
                            if not enable_argmax
                            else [
                                PerCoreCompileTimeDescriptor(
                                    named_compile_time_arg="sampling_sender_idx",
                                    core_values=[
                                        (core, argmax_sender_idx_lookup[(core.x, core.y)])
                                        for core in argmax_cores_row_wise
                                    ],
                                    other_value=0,
                                )
                            ]
                        )
                        + [
                            PerCoreCompileTimeDescriptor(
                                named_compile_time_arg="matmul_eh_bank_id",
                                core_values=eh_bank_id_core_values if enable_mtp_on_device else [],
                                other_value=0,
                            ),
                            PerCoreCompileTimeDescriptor(
                                named_compile_time_arg="matmul_eh_vc",
                                core_values=eh_vc_core_values if enable_mtp_on_device else [],
                                other_value=0,
                            ),
                        ]
                    ),
                    # Per-core runtime args: mesh argmax senders get BRISC sender metadata.
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(worker_core, [])],
                        brisc_args=[(worker_core, [])]
                        + per_core_brisc_runtime_args
                        + LMHeadSampling._build_reduce_brisc_per_core_args(
                            enable_reduce_to_one, reduce_params, reduce_device_role, device, device_idx
                        ),
                    ),
                )

                # ================================================================
                # Program descriptor
                # ================================================================
                kernel_result = unified_kernel.get_kernel_descriptors()
                input_role_cores = set()
                mcast_receiver_role_cores = set()
                matmul_role_cores = set()
                argmax_role_cores = set()
                argmax_final_role_cores = set()
                for group in kernel_result.groups:
                    group_cores = ttnn.corerange_to_cores(group.core_range_set)
                    if group.compile_time_arg_values.get("is_input_core", 0) == 1:
                        input_role_cores.update((c.x, c.y) for c in group_cores)
                    if group.compile_time_arg_values.get("is_mcast_receiver_core", 0) == 1:
                        mcast_receiver_role_cores.update((c.x, c.y) for c in group_cores)
                    if group.compile_time_arg_values.get("is_matmul_core", 0) == 1:
                        matmul_role_cores.update((c.x, c.y) for c in group_cores)
                    if group.compile_time_arg_values.get("sampling_is_active_core", 0) == 1:
                        argmax_role_cores.update((c.x, c.y) for c in group_cores)
                    if group.compile_time_arg_values.get("sampling_is_final_core", 0) == 1:
                        argmax_final_role_cores.update((c.x, c.y) for c in group_cores)

                expected_input_role_cores = {(c.x, c.y) for c in ttnn.corerange_to_cores(mcast_sender_core_grid)}
                if input_role_cores != expected_input_role_cores:
                    missing = sorted(expected_input_role_cores - input_role_cores)[:16]
                    extra = sorted(input_role_cores - expected_input_role_cores)[:16]
                    raise RuntimeError(
                        "Unified kernel role mapping mismatch: "
                        f"is_input_core core-set mismatch. missing={missing}, extra={extra}"
                    )

                expected_mcast_receiver_role_cores = {(c.x, c.y) for c in ttnn.corerange_to_cores(mcast_receiver_grid)}
                if mcast_receiver_role_cores != expected_mcast_receiver_role_cores:
                    missing = sorted(expected_mcast_receiver_role_cores - mcast_receiver_role_cores)[:16]
                    extra = sorted(mcast_receiver_role_cores - expected_mcast_receiver_role_cores)[:16]
                    raise RuntimeError(
                        "Unified kernel role mapping mismatch: "
                        f"is_mcast_receiver_core core-set mismatch. missing={missing}, extra={extra}"
                    )

                expected_matmul_role_cores = {(c.x, c.y) for c in ttnn.corerange_to_cores(matmul_core_grid)}
                if matmul_role_cores != expected_matmul_role_cores:
                    missing = sorted(expected_matmul_role_cores - matmul_role_cores)[:16]
                    extra = sorted(matmul_role_cores - expected_matmul_role_cores)[:16]
                    raise RuntimeError(
                        "Unified kernel role mapping mismatch: "
                        f"is_matmul_core core-set mismatch. missing={missing}, extra={extra}"
                    )
                if enable_argmax:
                    if argmax_role_cores != expected_matmul_role_cores:
                        missing = sorted(expected_matmul_role_cores - argmax_role_cores)[:16]
                        extra = sorted(argmax_role_cores - expected_matmul_role_cores)[:16]
                        raise RuntimeError(
                            "Unified kernel role mapping mismatch: "
                            f"sampling_is_active_core core-set mismatch. missing={missing}, extra={extra}"
                        )
                    if len(argmax_final_role_cores) != 1:
                        raise RuntimeError(
                            "Unified kernel role mapping mismatch: "
                            "sampling_is_final_core must map to exactly one core"
                        )

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    cbs=cbs_list,
                    semaphores=semaphore_descriptors,
                )

                # Append CCL routing args to the broadcast writer kernel (NCRISC in current broadcast split).
                # Find the is_input_core NCRISC kernel group (used for both input and MTP broadcasts)
                writer_kernel_idx = None
                writer_rt_args_ref = None
                if not skip_ccl:
                    ccl_writer_group = None
                    for group in kernel_result.groups:
                        if group.compile_time_arg_values.get("is_input_core", 0) == 1 and group.core_range_set.contains(
                            worker_core
                        ):
                            ccl_writer_group = group
                            break
                    if ccl_writer_group is None:
                        raise RuntimeError("Missing is_input_core kernel group for CCL writer fabric append")
                    writer_kernel_idx = ccl_writer_group.ncrisc_kernel_index
                    writer_rt_args_ref = program.kernels[writer_kernel_idx].runtime_args[worker_core.x][worker_core.y]
                    writer_rt_args_ref.extend(bcast_config.get_ncrisc_per_core_rt_args(coord, program, worker_core))

                # [MTP] Append MTP broadcast tree per-core routing args
                if mtp_bcast_config is not None:
                    mtp_bcast_worker = mtp_bcast_config.get_worker_core(coord)
                    writer_rt_args_ref.extend(
                        mtp_bcast_config.get_ncrisc_per_core_rt_args(coord, program, mtp_bcast_worker)
                    )

                if not skip_ccl and is_argmax_mesh_sender_core:
                    sender_group = kernel_result.get_group_by_arg("sampling_mesh_sender_core", 1)
                    if sender_group is None:
                        raise RuntimeError("Missing argmax mesh sender kernel group for BRISC fabric append")
                    sender_kernel_idx = sender_group.brisc_kernel_index
                    fabric_rt_args = ttnn.setup_fabric_connection(
                        src_fabric_node_id=mesh_device.get_fabric_node_id(coord),
                        dst_fabric_node_id=mesh_device.get_fabric_node_id(dest_coord),
                        link_idx=sender_link_idx,
                        program_descriptor=program,
                        worker_core=argmax_final_core,
                    )
                    program.kernels[sender_kernel_idx].runtime_args[argmax_final_core.x][argmax_final_core.y].extend(
                        fabric_rt_args
                    )
                if persistent_enable:
                    persistent_group = kernel_result.get_group_by_arg("sampling_is_final_core", 1)
                    if persistent_group is None:
                        raise RuntimeError("Missing argmax final core kernel group for persistent fabric append")
                    persistent_kernel_idx = persistent_group.brisc_kernel_index
                    persistent_fabric_rt_args = ttnn.setup_fabric_connection(
                        src_fabric_node_id=mesh_device.get_fabric_node_id(coord),
                        dst_fabric_node_id=persistent_target_node,
                        link_idx=0,
                        program_descriptor=program,
                        worker_core=argmax_final_core,
                    )
                    program.kernels[persistent_kernel_idx].runtime_args[argmax_final_core.x][
                        argmax_final_core.y
                    ].extend(persistent_fabric_rt_args)

                # [Reduce] Setup fabric connections for reduce fabric cores (non-ROOT1 only)
                if enable_reduce_to_one and reduce_device_role != MESH_ROOT1:
                    src_node = mesh_device.get_fabric_node_id(coord)
                    num_cols = reduce_params["num_columns"]
                    for fc_i, fc in enumerate(reduce_params["fabric_cores"]):
                        fc_group = None
                        for g in kernel_result.groups:
                            if g.compile_time_arg_values.get(
                                "is_reduce_fabric_core", 0
                            ) == 1 and g.core_range_set.contains(fc):
                                fc_group = g
                                break
                        rfk_idx = fc_group.brisc_kernel_index
                        link = 0 if fc_i < num_cols // 2 else 1
                        rf_args = ttnn.setup_fabric_connection(
                            src_node,
                            reduce_dest_fabric_node_id,
                            link,
                            program,
                            fc,
                        )
                        program.kernels[rfk_idx].runtime_args[fc.x][fc.y].extend(rf_args)

                input_group_for_pad = kernel_result.get_group_by_arg("is_input_core", 1)
                if input_group_for_pad is not None:
                    pad_ncrisc_idx = input_group_for_pad.ncrisc_kernel_index
                    worker_ncrisc_arg_refs.append(
                        program.kernels[pad_ncrisc_idx].runtime_args[worker_core.x][worker_core.y]
                    )
                device_programs.append((coord, program))

        if worker_ncrisc_arg_refs:
            max_ncrisc_len = max(len(ref) for ref in worker_ncrisc_arg_refs)
            for ref in worker_ncrisc_arg_refs:
                pad_count = max_ncrisc_len - len(ref)
                if pad_count > 0:
                    ref.extend([0] * pad_count)

        for dp_coord, dp_program in device_programs:
            mesh_program_descriptor[ttnn.MeshCoordinateRange(dp_coord, dp_coord)] = dp_program
        io_tensors = [input_tensor_mesh, intermediate_tensor_mesh, vocab_tensor, output_tensor]
        if gamma_tensor is not None:
            io_tensors.append(gamma_tensor)
        io_tensors.extend([indices_tensor, output_index_tensor])
        if is_mtp_base_stage:
            io_tensors.extend([eh_mm_fused_buffer, embedding_tensor, eh_projection_tensor])
            if h_gamma_tensor is not None:
                io_tensors.append(h_gamma_tensor)
            if e_gamma_tensor is not None:
                io_tensors.append(e_gamma_tensor)
        if metadata_tensor is not None:
            io_tensors.append(metadata_tensor)
        if not skip_ccl:
            if scores_scratch_tensor is not None:
                io_tensors.append(scores_scratch_tensor)
            if indices_scratch_tensor is not None:
                io_tensors.append(indices_scratch_tensor)
        if base_token_buffer is not None:
            io_tensors.append(base_token_buffer)
        result = ttnn.generic_op(io_tensors, mesh_program_descriptor)
        logger.debug("[OP] generic_op returned")
        return result
