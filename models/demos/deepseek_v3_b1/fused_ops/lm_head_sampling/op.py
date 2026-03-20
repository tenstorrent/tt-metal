# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import get_max_page_size_and_num_pages
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_uint32


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _is_singleton_prefix_shape(shape, expected_last_dim: int) -> bool:
    dims = tuple(int(d) for d in shape)
    if not dims or dims[-1] != expected_last_dim:
        return False
    return all(d == 1 for d in dims[:-1])


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
        3. Sample token index using argmax (greedy k=1)

        When fuse_mtp=True, additionally performs MTP (Multi-Token Prediction) fusion:
        4. Look up token embedding from embedding_tensor using sampled index
        5. RMSNorm on hidden states using h_gamma_tensor
        6. RMSNorm on token embedding using e_gamma_tensor
        7. Concatenate normalized hidden states and embedding
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
            k: Sampling k; currently only k=1 supported when indices is provided.
            p: Top-p threshold (unused for k=1 path).
            epsilon: Small value to avoid division by zero in RMS norm.
            fuse_mtp: If True, perform MTP fusion after sampling.
            fuse_mtp_verification: If True, perform verification against reference_token.
            reference_token: Reference token (T_base) for verification [1, 1] uint32.
            embedding_tensor: Token embedding table (torch.Tensor) [vocab_size, embedding_dim]
            h_gamma_tensor: RMSNorm gamma for hidden states in MTP (torch.Tensor) [hidden_dim]
            e_gamma_tensor: RMSNorm gamma for embeddings in MTP (torch.Tensor) [embedding_dim]
            eh_projection_tensor: Projection matrix for concatenated [h, e] (torch.Tensor) [hidden_dim + embedding_dim, output_dim]

        Returns:
            - If neither fuse_mtp nor fuse_mtp_verification: (sampled_index [1,1], None)
            - If fuse_mtp: (sampled_index [1,1], mtp_projection_output [1, output_dim])
            - If fuse_mtp_verification: (sampled_index [1,1], verification_result uint32 (1=match, 0=no_match))
        """
        assert not (fuse_mtp and fuse_mtp_verification), "fuse_mtp and fuse_mtp_verification are mutually exclusive"

        if fuse_mtp:
            assert embedding_tensor is not None, "embedding_tensor is required for fused MTP"
            assert h_gamma_tensor is not None, "h_gamma_tensor is required for fused MTP"
            assert e_gamma_tensor is not None, "e_gamma_tensor is required for fused MTP"
            assert eh_projection_tensor is not None, "eh_projection_tensor is required for fused MTP"

        if fuse_mtp_verification:
            assert reference_token is not None, "reference_token is required for MTP verification"

        # Step 1: RMSNorm on input hidden states for LM head
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        normalized = input_tensor * torch.rsqrt(variance + epsilon)
        rmsnorm_out = normalized * gamma_tensor

        # Step 2: Matmul with vocab tensor to get logits
        scores = rmsnorm_out @ vocab_tensor

        if k != 1:
            raise NotImplementedError("LMHeadSampling fused golden currently supports only k=1")

        # p is intentionally unused in k=1 path; keep for API compatibility.
        _ = p

        # Step 3: Sample token using argmax (greedy)
        scores_f32 = scores.float().reshape(-1)
        indices_i64 = indices.to(torch.int64).reshape(-1)
        if scores_f32.numel() != indices_i64.numel():
            raise ValueError(
                f"scores and indices must have the same number of elements, got {scores_f32.numel()} and {indices_i64.numel()}"
            )

        max_score = torch.max(scores_f32)
        tied_mask = scores_f32 == max_score
        selected_index = torch.min(indices_i64[tied_mask]).to(torch.uint32)

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

        # Step 5: RMSNorm on hidden states using h_gamma_tensor
        h_variance = input_tensor.pow(2).mean(-1, keepdim=True)
        h_normalized = input_tensor * torch.rsqrt(h_variance + epsilon)
        h_rmsnorm_out = h_normalized * h_gamma_tensor  # [1, hidden_dim]

        # Step 6: RMSNorm on token embedding using e_gamma_tensor
        e_variance = token_embedding.pow(2).mean(-1, keepdim=True)
        e_normalized = token_embedding * torch.rsqrt(e_variance + epsilon)
        e_rmsnorm_out = e_normalized * e_gamma_tensor  # [1, embedding_dim]

        # Step 7: Concatenate normalized hidden states and embedding
        concat_he = torch.cat([h_rmsnorm_out, e_rmsnorm_out], dim=-1)

        # Step 8: Project through eh_projection_tensor
        mtp_output = concat_he @ eh_projection_tensor
        return selected_index.reshape(1, 1), mtp_output

    @staticmethod
    def op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        gamma_tensor,
        vocab_tensor,
        output_tensor,
        sender_coord,
        output_mtp_tensor=None,
        embedding_tensor=None,
        h_gamma_tensor=None,
        e_gamma_tensor=None,
        eh_projection_tensor=None,
        eh_proj_working_buf_tensor=None,
        mcast_dst_working_buf_tensor=None,
        mcast_eh_dst_working_buf_tensor=None,
        indices_tensor=None,
        output_index_tensor=None,
        argmax_final_core_coord=None,
        argmax_final_mesh_coord=None,
        global_semaphore=None,
        global_stage2_semaphore=None,
        fabric_scratch_tensor=None,
        semaphores=None,
        cluster_axis=0,
        secondary_cluster_axis=None,
        num_links=1,
        fp32_dest_acc_en=False,
        epsilon=1e-6,
        rsqrt_fast_approx=False,
        skip_ccl=None,
        socket_input=None,
        socket_output=None,
        persistent_mode=False,
        termination_semaphore=None,
        persistent_next_iter_semaphore=None,
        enable_mtp=False,
        enable_mtp_verification=False,
        reference_token_tensor=None,
        unverified_spec_tensor=None,
        verified_spec_tensor=None,
        eh_subblock_k=None,
        mtp_logits_socket_output=None,
        verify_output_staging_tensor=None,
    ):
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
            semaphores: List of global semaphores [out_ready, barrier, secondary_sync] for CCL
            cluster_axis: Primary axis for CCL broadcast (0=row, 1=col)
            secondary_cluster_axis: Secondary axis for CCL broadcast (optional)
            num_links: Number of fabric links for CCL
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
        # LMHeadSampling is always fused with k=1 sampling (argmax fast path).

        enable_argmax = True
        # MTP fusion is enabled if all MTP tensors are provided
        enable_mtp = (
            enable_mtp
            and output_mtp_tensor is not None
            and embedding_tensor is not None
            and h_gamma_tensor is not None
            and e_gamma_tensor is not None
            and eh_projection_tensor is not None
        )
        # MTP verification is enabled if the verification tensors are provided
        enable_mtp_verification = enable_mtp_verification and reference_token_tensor is not None
        assert not (
            enable_mtp and enable_mtp_verification
        ), "enable_mtp and enable_mtp_verification are mutually exclusive"
        has_mtp_logits_socket_output = enable_mtp and mtp_logits_socket_output is not None
        socket_mode_none = 0
        socket_mode_d2h = 1
        socket_mode_d2d = 2
        socket_page_size_bytes = 64
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
        # In multi-column meshes, enable secondary-axis relay by default so broadcast reaches
        # all clusters. Callers can still override explicitly if needed.
        if not skip_ccl and mesh_cols > 1 and secondary_cluster_axis is None:
            secondary_cluster_axis = 1
        sender_row = sender_coord[0]
        sender_col = sender_coord[1]
        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        intermediate_tensors_per_device = ttnn.get_device_tensors(intermediate_tensor_mesh)
        gamma_tensors_per_device = ttnn.get_device_tensors(gamma_tensor)
        vocab_tensors_per_device = ttnn.get_device_tensors(vocab_tensor)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        indices_tensors_per_device = ttnn.get_device_tensors(indices_tensor) if enable_argmax else None
        output_index_tensors_per_device = ttnn.get_device_tensors(output_index_tensor) if enable_argmax else None
        output_mtp_tensors_per_device = ttnn.get_device_tensors(output_mtp_tensor) if enable_mtp else None
        embedding_tensors_per_device = ttnn.get_device_tensors(embedding_tensor) if enable_mtp else None
        h_gamma_tensors_per_device = ttnn.get_device_tensors(h_gamma_tensor) if enable_mtp else None
        e_gamma_tensors_per_device = ttnn.get_device_tensors(e_gamma_tensor) if enable_mtp else None
        eh_proj_tensors_per_device = ttnn.get_device_tensors(eh_projection_tensor) if enable_mtp else None
        scratch_tensors_per_device = (
            ttnn.get_device_tensors(fabric_scratch_tensor) if (enable_argmax and not skip_ccl) else None
        )
        if enable_argmax and not skip_ccl:
            if global_semaphore is None or global_stage2_semaphore is None or fabric_scratch_tensor is None:
                raise ValueError(
                    "global_semaphore, global_stage2_semaphore, and fabric_scratch_tensor are required for mesh argmax"
                )
            if mesh_rows < 2 or mesh_cols != 2:
                raise NotImplementedError(
                    f"Fused LM-head mesh argmax currently supports only (R,2) with R>=2, got {mesh_shape}"
                )
            if argmax_final_mesh_coord is None:
                raise ValueError("argmax_final_mesh_coord is required for mesh argmax")
        if persistent_mode and persistent_next_iter_semaphore is None:
            raise ValueError(
                "persistent_next_iter_semaphore is required when persistent_mode=True "
                "(must be a global semaphore on the full device grid)"
            )
        # Semaphore addresses (only needed for CCL mode)
        out_ready_sem_addr = 0
        barrier_sem_addr = 0
        secondary_sync_sem_addr = 0
        if not skip_ccl and semaphores is not None:
            out_ready_semaphore = semaphores[0]
            barrier_semaphore = semaphores[1]
            secondary_sync_semaphore = semaphores[2]
            out_ready_sem_addr = ttnn.get_global_semaphore_address(out_ready_semaphore)
            barrier_sem_addr = ttnn.get_global_semaphore_address(barrier_semaphore)
            secondary_sync_sem_addr = ttnn.get_global_semaphore_address(secondary_sync_semaphore)
        global_sem_addr = (
            int(ttnn.get_global_semaphore_address(global_semaphore)) if (enable_argmax and not skip_ccl) else 0
        )
        global_stage2_sem_addr = (
            int(ttnn.get_global_semaphore_address(global_stage2_semaphore)) if (enable_argmax and not skip_ccl) else 0
        )
        persistent_next_iter_global_sem_addr = (
            int(ttnn.get_global_semaphore_address(persistent_next_iter_semaphore)) if persistent_mode else 0
        )
        # Calculate packet size and page info for CCL broadcast
        packet_size_bytes = 14336  # 14 KB packets for (1, 7168) input

        # Get tile info from input tensor (use a sample device tensor)
        input_tensor_sample = input_tensors_per_device[0]
        in0_tile = input_tensor_sample.get_tile()
        input_shape = input_tensor_sample.shape
        data_format = input_tensor_sample.dtype
        element_size = 2
        tile_id_start = 0
        numel = int(input_shape[0]) * int(input_shape[1])
        scalar_packed = float_to_uint32(1.0 / math.sqrt(float(numel)))
        epsilon_packed = float_to_uint32(epsilon)
        # CCL broadcast page info
        bcast_page_size_bytes = 32 * 32 * element_size  # interpret as 32x32 tile
        bcast_num_pages = input_shape[0] * input_shape[1] * element_size // bcast_page_size_bytes
        num_pages_per_packet = packet_size_bytes // bcast_page_size_bytes

        # Matmul shape info from input and vocab tensors
        num_tiles_k = input_shape[1] // in0_tile.tile_shape[1]
        # RMSNorm in this path must match broadcast_rms tile/page interpretation.
        full_32x32_tile = ttnn.Tile((32, 32))
        half_16x32_tile = ttnn.Tile((16, 32))
        is_16x32_tile = (input_shape[1] // full_32x32_tile.tile_shape[1]) % full_32x32_tile.tile_shape[0] != 0
        rms_interpreted_tile = half_16x32_tile if is_16x32_tile else full_32x32_tile
        rms_tile_height, rms_tile_width = rms_interpreted_tile.tile_shape
        rms_tile_size = rms_interpreted_tile.get_tile_size(data_format)
        rms_num_tiles = (input_shape[0] * input_shape[1]) // (rms_tile_height * rms_tile_width)
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

        # [MTP] Get eh projection matmul info (per-core output width)
        if enable_mtp:
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
            eh_concat_rms_tiles = rms_num_tiles * 2
            eh_num_tiles_k = (int(input_shape[1]) + embedding_dim) // in0_tile.tile_shape[1]
            eh_mcast_data_size_bytes = eh_concat_rms_tiles * rms_tile_size

            # EH matmul DRAM streaming parameters
            eh_projection_tensor_sample = eh_proj_tensors_per_device[0]
            eh_dtype = eh_projection_tensor_sample.dtype
            eh_proj_tile = eh_projection_tensor_sample.get_tile()
            eh_proj_tile_size = eh_proj_tile.get_tile_size(eh_dtype)
            eh_subblock_k = eh_subblock_k or eh_concat_rms_tiles
            eh_num_subblocks_k = eh_num_tiles_k // eh_subblock_k
            eh_out_num_tiles = eh_out_w_per_core  # M=1, so out tiles = per_core_n
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
            eh_num_in1_buffers = 2  # Double buffering (must match NumBuffers=2 in dram_streaming_matmul.hpp)
            eh_in1_CB_tiles = eh_subblock_k * eh_num_in1_buffers
            eh_in1_CB_size = eh_in1_CB_tiles * eh_proj_tile_size
            # Token ID buffer (8 bytes aligned: 4 bytes token_id + 4 bytes padding)
            mtp_token_size_bytes = 8
        else:
            eh_out_w_per_core = 0
            embedding_dim = 0
            e_num_tiles = 0
            eh_num_tiles_k = 0
            eh_mcast_data_size_bytes = 0
            mtp_token_size_bytes = 0

        # ====================================================================
        # CB indices
        # ====================================================================
        rmsnorm_input_cb = 0  # raw input on sender core (tensor-backed)
        mcast_dst_cb = 1  # Mcast destination = matmul in0 (all mcast grid cores, intermediate)
        matmul_in1_cb = 2  # vocab_tensor weights on matmul cores (tensor-backed)
        rmsnorm_gamma_cb = 7  # RMSNorm gamma weights on sender core (tensor-backed)
        mcast_src_cb = 8  # RMSNorm output on sender core (intermediate), consumed by mcast sender
        matmul_eh_cb = 9  # [MTP] EH projection weights on matmul cores (tensor-backed)
        embedding_cb = mcast_src_cb  # [MTP] Reuses CB 8 — mcast consumes it before embedding DRAM read
        h_gamma_cb = 11  # [MTP] RMSNorm gamma weights for hidden states on sender core (tensor-backed)
        e_gamma_cb = 12  # [MTP] RMSNorm gamma weights for embeddings on sender core (tensor-backed)
        mcast_eh_src_cb = 15  # [MTP] Fused [h_norm|e_norm] on sender core, both RMSNorms write here directly
        mtp_embedding_done_cb = 20  # [MTP] Signal CB: NCRISC pushes after embedding, TRISC waits before e_rmsnorm
        argmax_winner_cb = 3
        argmax_gather_cb = 4
        argmax_indices_cb = 5
        argmax_socket_cb = 6
        matmul_out_cb = 16  # Matmul output on matmul cores (tensor-backed)
        matmul_out_eh_cb = 17  # [MTP] EH matmul output on matmul cores (tensor-backed)
        mcast_eh_dst_cb = (
            18  # [MTP] Second mcast destination (concat [h_norm|e_norm]) on all mcast cores (intermediate)
        )

        # CB indices for CCL broadcast (use separate CBs to avoid conflicts)
        bcast_pkt_cb = 30  # Packet buffer for CCL broadcast

        # ====================================================================
        # Semaphore IDs (for intra-device mcast)
        # ====================================================================
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1
        argmax_receiver_semaphore_id = 2
        argmax_local_ready_semaphore_id = 3
        # [MTP] Semaphore IDs for second mcast (EH projection matmul)
        mtp_ready_semaphore_id = 4
        mcast_eh_data_sender_semaphore_id = 5
        mcast_eh_data_receiver_semaphore_id = 6
        verify_output_ready_semaphore_id = 7
        skip_pipeline_semaphore_id = 8
        eh_matmul_done_semaphore_id = 9  # [MTP] EH matmul cores inc this on argmax final core when shard is pushed

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                # Sender identity is fixed by sender_coord even when skip_ccl=True.
                is_sender = (row == sender_row) and (col == sender_col)

                # Gate MTP pipeline to exit device only.
                # MTP (embedding lookup, EH matmul, logits gather/send) only makes
                # sense on the device that holds the global argmax winner.  Non-exit
                # devices skip MTP entirely — they loop back to the CCL broadcast
                # which is gated by persistent_next_iter from the exit device.
                # Note: MTP token CCL broadcast config (further below) still uses the
                # original enable_mtp so ALL devices participate in the broadcast.
                if skip_ccl or argmax_final_mesh_coord is None:
                    is_exit_device = True
                else:
                    is_exit_device = row == int(argmax_final_mesh_coord[0]) and col == int(argmax_final_mesh_coord[1])
                enable_mtp_on_device = enable_mtp and is_exit_device
                has_mtp_logits_socket_on_device = has_mtp_logits_socket_output and is_exit_device
                if skip_ccl:
                    is_secondary_sender = False
                    is_receiver = False
                else:
                    is_secondary_sender = (
                        secondary_cluster_axis is not None and (row == sender_row) and (col != sender_col)
                    )
                    is_receiver = not is_sender and not is_secondary_sender

                # Get per-device tensors
                input_tensor_device = input_tensors_per_device[device_idx]
                intermediate_tensor_device = intermediate_tensors_per_device[device_idx]
                gamma_tensor_device = gamma_tensors_per_device[device_idx]
                vocab_tensor_device = vocab_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]
                indices_tensor_device = indices_tensors_per_device[device_idx] if enable_argmax else None
                output_index_tensor_device = output_index_tensors_per_device[device_idx] if enable_argmax else None

                # Get device handle
                device = input_tensor_device.device()

                # ================================================================
                # CCL broadcast: physical core and routing info
                # ================================================================
                # Worker core from input tensor shard grid (single core)
                input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                shard_grid_start = input_shard_grid.bounding_box().start
                worker_core = ttnn.CoreCoord(shard_grid_start.x, shard_grid_start.y)

                # Get physical core for NOC addressing
                data_core_physical = device.worker_core_from_logical_core(worker_core)
                core_noc_x = data_core_physical.x
                core_noc_y = data_core_physical.y

                # Calculate ring index and targets for primary axis (column)
                ring_size = mesh_rows
                ring_index = row

                # For Linear topology, calculate forward and backward targets
                num_targets_forward = ring_size - ring_index - 1
                num_targets_backward = ring_index

                # Determine if this device has secondary axis connections
                has_secondary_target = is_sender and (mesh_cols > 1) and (secondary_cluster_axis is not None)

                # Calculate mcast distances
                start_distance_forward = 1 if num_targets_forward > 0 else 0
                range_hops_forward = num_targets_forward
                start_distance_backward = 1 if num_targets_backward > 0 else 0
                range_hops_backward = num_targets_backward

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

                # Mcast grid = bounding box of (matmul participants U sender core).
                # This avoids reserving the full device grid while still ensuring
                # is_input_core is inside unified kernel core_ranges.
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
                    eh_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(eh_matmul_noc)
                    eh_matmul_core_grid = output_mtp_tensors_per_device[device_idx].memory_config().shard_spec.grid
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

                # Pre-compute CB 1 backing address for BRISC mcast and MTP logits gather
                mcast_receiver_data_addr = (
                    int(ttnn.get_device_tensors(mcast_dst_working_buf_tensor)[device_idx].buffer_address())
                    if mcast_dst_working_buf_tensor is not None
                    else 0
                )

                # Pre-compute MTP logits gather metadata (only on exit device)
                eh_output_tile_size = 0
                eh_shard_size_bytes = 0
                num_eh_matmul_cores = 0
                if enable_mtp_on_device:
                    eh_output_tile_size = out_tile.get_tile_size(data_format)
                    eh_shard_size_bytes = eh_out_w_per_core * eh_output_tile_size
                    num_eh_matmul_cores = len(eh_worker_cores)

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
                    argmax_final_core_phys = device.worker_core_from_logical_core(argmax_final_core)
                    emit_socket_on_this_device = bool(enable_socket_output)
                    argmax_num_values = n_per_core
                    argmax_num_senders = len(argmax_cores_row_wise)
                    argmax_expected_remote_incs = argmax_num_senders - 1
                    argmax_sender_idx_lookup = {(c.x, c.y): idx for idx, c in enumerate(argmax_cores_row_wise)}
                    argmax_winner_page_bytes = 16
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
                        argmax_stage2_slot_base_offset = (
                            argmax_stage1_slot_base_offset + argmax_stage1_num_slots * argmax_winner_page_bytes
                        )
                        argmax_stage2_num_slots = mesh_cols
                        argmax_stage1_expected_remote_incs = mesh_rows - 1
                        argmax_stage2_expected_remote_incs = mesh_cols - 1
                        argmax_stage1_sender = 1 if row != target_row else 0
                        argmax_stage1_receiver = 1 if row == target_row else 0
                        argmax_stage2_sender = 1 if (row == target_row and col != target_col) else 0
                        argmax_stage2_receiver = 1 if (row == target_row and col == target_col) else 0
                        argmax_stage1_local_slot_offset = (
                            argmax_stage1_slot_base_offset + row * argmax_winner_page_bytes
                        )
                        argmax_stage2_local_slot_offset = (
                            argmax_stage2_slot_base_offset + col * argmax_winner_page_bytes
                        )
                        is_argmax_mesh_sender_core = bool(argmax_stage1_sender or argmax_stage2_sender)
                        argmax_mesh_local_send_slot_offset = (
                            argmax_stage1_local_slot_offset if argmax_stage1_sender else argmax_stage2_local_slot_offset
                        )

                        if is_argmax_mesh_sender_core:
                            if argmax_stage1_sender:
                                dest_coord = ttnn.MeshCoordinate(target_row, col)
                                send_slot_offset = argmax_stage1_slot_base_offset + row * argmax_winner_page_bytes
                                sender_dst_sem_addr = global_sem_addr
                                sender_link_idx = _x_axis_link_idx_for_stage1_sender(row)
                            else:
                                dest_coord = ttnn.MeshCoordinate(target_row, target_col)
                                send_slot_offset = argmax_stage2_slot_base_offset + col * argmax_winner_page_bytes
                                sender_dst_sem_addr = global_stage2_sem_addr
                                sender_link_idx = 0

                            dest_idx = int(dest_coord[0]) * mesh_cols + int(dest_coord[1])
                            per_core_brisc_runtime_args.append(
                                (
                                    argmax_final_core,
                                    [
                                        int(argmax_mesh_local_send_slot_offset),
                                        int(mesh_device.get_fabric_node_id(dest_coord).mesh_id),
                                        int(mesh_device.get_fabric_node_id(dest_coord).chip_id),
                                        int(scratch_tensors_per_device[dest_idx].buffer_address())
                                        + int(send_slot_offset),
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

                    if enable_mtp_on_device:
                        argmax_socket_mode = socket_mode_none
                    else:
                        argmax_socket_mode = socket_mode_selected if emit_socket_on_this_device else socket_mode_none

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

                if not skip_ccl:
                    brisc_bcast_cb = bcast_pkt_cb
                    brisc_bcast_num_pages_to_read = bcast_num_pages
                elif recv_socket_on_this_device:
                    brisc_bcast_cb = rmsnorm_input_cb
                    brisc_bcast_num_pages_to_read = rms_num_tiles
                else:
                    brisc_bcast_cb = 0
                    brisc_bcast_num_pages_to_read = 0
                brisc_is_active = (not skip_ccl) or recv_socket_on_this_device

                # Get NOC coordinates for mcast destination
                mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
                mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)
                bcast_num_pages_to_read = bcast_num_pages

                # ================================================================
                # [MTP] Token CCL Broadcast configuration
                # Broadcast token from argmax_final_mesh_coord device to all devices.
                # Similar to input logits broadcast but from a different sender.
                # ================================================================
                mtp_bcast_is_sender = 0
                mtp_bcast_is_secondary_sender = 0
                mtp_bcast_is_receiver = 0
                mtp_bcast_num_targets_forward = 0
                mtp_bcast_num_targets_backward = 0
                mtp_bcast_has_secondary_target = 0
                mtp_bcast_start_distance_forward = 0
                mtp_bcast_range_hops_forward = 0
                mtp_bcast_start_distance_backward = 0
                mtp_bcast_range_hops_backward = 0

                if enable_mtp and not skip_ccl and argmax_final_mesh_coord is not None:
                    mtp_sender_row = int(argmax_final_mesh_coord[0])
                    mtp_sender_col = int(argmax_final_mesh_coord[1])

                    # Determine role in MTP token broadcast (sender broadcasts to all other devices)
                    mtp_bcast_is_sender = 1 if (row == mtp_sender_row and col == mtp_sender_col) else 0
                    # Secondary sender: same row as sender, different column (for column broadcast)
                    mtp_bcast_is_secondary_sender = (
                        1
                        if (secondary_cluster_axis is not None and row == mtp_sender_row and col != mtp_sender_col)
                        else 0
                    )
                    mtp_bcast_is_receiver = 1 if not mtp_bcast_is_sender and not mtp_bcast_is_secondary_sender else 0

                    # Calculate forward/backward targets from sender's perspective
                    # Forward = devices with row > sender_row, Backward = devices with row < sender_row
                    mtp_bcast_num_targets_forward = mesh_rows - mtp_sender_row - 1
                    mtp_bcast_num_targets_backward = mtp_sender_row

                    # Secondary target: sender broadcasts to secondary senders in other columns
                    mtp_bcast_has_secondary_target = (
                        1 if mtp_bcast_is_sender and mesh_cols > 1 and secondary_cluster_axis is not None else 0
                    )

                    # Distance calculations (from current device's perspective)
                    if mtp_bcast_is_sender:
                        mtp_bcast_start_distance_forward = 1 if mtp_bcast_num_targets_forward > 0 else 0
                        mtp_bcast_range_hops_forward = mtp_bcast_num_targets_forward
                        mtp_bcast_start_distance_backward = 1 if mtp_bcast_num_targets_backward > 0 else 0
                        mtp_bcast_range_hops_backward = mtp_bcast_num_targets_backward
                    elif mtp_bcast_is_secondary_sender:
                        # Secondary sender broadcasts within its column
                        mtp_bcast_start_distance_forward = 1 if mtp_bcast_num_targets_forward > 0 else 0
                        mtp_bcast_range_hops_forward = mtp_bcast_num_targets_forward
                        mtp_bcast_start_distance_backward = 1 if mtp_bcast_num_targets_backward > 0 else 0
                        mtp_bcast_range_hops_backward = mtp_bcast_num_targets_backward

                # ================================================================
                # NCRISC compile-time args
                # ================================================================
                ncrisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("enable_argmax", 1),
                    ("bcast_cb0_id", bcast_pkt_cb if not skip_ccl else 0),
                    ("bcast_num_pages_to_read", bcast_num_pages_to_read if not skip_ccl else 0),
                    ("bcast_tensor0_page_size", bcast_page_size_bytes if not skip_ccl else 0),
                    ("bcast_num_targets_forward_direction", num_targets_forward if not skip_ccl else 0),
                    ("bcast_num_targets_backward_direction", num_targets_backward if not skip_ccl else 0),
                    ("bcast_is_sender", int(is_sender) if not skip_ccl else 0),
                    ("bcast_core_noc_x", core_noc_x if not skip_ccl else 0),
                    ("bcast_core_noc_y", core_noc_y if not skip_ccl else 0),
                    ("bcast_is_secondary_sender", int(is_secondary_sender) if not skip_ccl else 0),
                    ("bcast_has_secondary_target", int(has_secondary_target) if not skip_ccl else 0),
                    ("bcast_start_distance_in_hops_forward", start_distance_forward if not skip_ccl else 0),
                    ("bcast_range_hops_forward", range_hops_forward if not skip_ccl else 0),
                    ("bcast_start_distance_in_hops_backward", start_distance_backward if not skip_ccl else 0),
                    ("bcast_range_hops_backward", range_hops_backward if not skip_ccl else 0),
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
                    ("argmax_num_values", argmax_num_values),
                    ("argmax_winner_page_bytes", argmax_winner_page_bytes),
                    ("argmax_num_senders", argmax_num_senders),
                    ("argmax_expected_remote_incs", argmax_expected_remote_incs),
                    ("argmax_receiver_semaphore_id", argmax_receiver_semaphore_id),
                    ("argmax_local_ready_semaphore_id", argmax_local_ready_semaphore_id),
                    ("argmax_mesh_mode", argmax_mesh_mode),
                    ("argmax_stage1_sender", argmax_stage1_sender),
                    ("argmax_stage1_receiver", argmax_stage1_receiver),
                    ("argmax_stage2_sender", argmax_stage2_sender),
                    ("argmax_stage2_receiver", argmax_stage2_receiver),
                    ("argmax_stage1_slot_base_offset", argmax_stage1_slot_base_offset),
                    ("argmax_stage1_num_slots", argmax_stage1_num_slots),
                    ("argmax_stage1_expected_remote_incs", argmax_stage1_expected_remote_incs),
                    ("argmax_stage1_local_slot_offset", argmax_stage1_local_slot_offset),
                    ("argmax_stage2_slot_base_offset", argmax_stage2_slot_base_offset),
                    ("argmax_stage2_num_slots", argmax_stage2_num_slots),
                    ("argmax_stage2_expected_remote_incs", argmax_stage2_expected_remote_incs),
                    ("argmax_stage2_local_slot_offset", argmax_stage2_local_slot_offset),
                    ("argmax_mesh_local_send_slot_offset", argmax_mesh_local_send_slot_offset),
                    ("argmax_gather_cb", argmax_gather_cb),
                    ("argmax_indices_cb", argmax_indices_cb),
                    ("argmax_socket_mode", argmax_socket_mode),
                    ("argmax_socket_cb", argmax_socket_cb if enable_socket_output else 0),
                    ("argmax_socket_page_size_bytes", socket_page_size_bytes if enable_socket_output else 0),
                    ("persistent_mode", 1 if persistent_mode else 0),
                    ("mesh_row", row),
                    ("mesh_col", col),
                    # [MTP]
                    ("enable_mtp", 1 if enable_mtp_on_device else 0),
                    ("enable_mtp_verification", 1 if enable_mtp_verification else 0),
                    # MTP matmul, embedding, rmsnorm CBs
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
                    (
                        "eh_matmul_cb_in1_buf_addr",
                        ttnn.get_device_tensors(eh_proj_working_buf_tensor)[device_idx].buffer_address()
                        if (enable_mtp_on_device and eh_proj_working_buf_tensor is not None)
                        else 0,
                    ),
                    ("embedding_cb", embedding_cb),
                    ("h_gamma_cb", h_gamma_cb),
                    ("e_gamma_cb", e_gamma_cb),
                    ("mcast_eh_src_cb", mcast_eh_src_cb),
                    ("mcast_eh_dst_cb", mcast_eh_dst_cb),
                    ("rmsnorm_h_input_cb", rmsnorm_input_cb),
                    ("rmsnorm_h_output_cb", mcast_eh_src_cb),
                    ("rmsnorm_e_input_cb", embedding_cb),
                    ("rmsnorm_e_output_cb", mcast_eh_src_cb),
                    # [MTP] semaphores
                    ("mtp_ready_semaphore_id", mtp_ready_semaphore_id),
                    (
                        "mcast_eh_data_sender_semaphore",
                        mcast_eh_data_sender_semaphore_id if enable_mtp_on_device else 0,
                    ),
                    ("mcast_eh_data_receiver_semaphore", mcast_eh_data_receiver_semaphore_id),
                    ("mcast_eh_src_num_pages", eh_concat_rms_tiles if enable_mtp_on_device else 0),
                    ("mcast_eh_dst_num_pages", eh_num_tiles_k if enable_mtp_on_device else 0),
                    ("rmsnorm_h_num_tiles", rms_num_tiles),
                    ("rmsnorm_e_num_tiles", e_num_tiles if enable_mtp_on_device else 0),
                    ("mcast_eh_tile_size_bytes", 2048 if enable_mtp_on_device else 0),
                    ("embedding_size_bytes", embedding_dim * element_size if enable_mtp_on_device else 0),
                    # Sender core NOC for L1-to-L1 copy (embedding region in mcast_eh_src_cb -> embedding_cb)
                    ("sender_noc_x", int(core_noc_x) if enable_mtp_on_device else 0),
                    ("sender_noc_y", int(core_noc_y) if enable_mtp_on_device else 0),
                    ("mtp_embedding_done_cb", mtp_embedding_done_cb if enable_mtp_on_device else 0),
                    # [MTP Logits Socket]
                    ("has_mtp_logits_socket_output", 1 if has_mtp_logits_socket_on_device else 0),
                    ("mtp_logits_num_eh_cores", num_eh_matmul_cores if enable_mtp_on_device else 0),
                    ("mtp_logits_eh_shard_size", eh_shard_size_bytes if enable_mtp_on_device else 0),
                    (
                        "mtp_logits_payload_size",
                        packet_size_bytes if (enable_mtp_on_device or enable_mtp_verification) else 0,
                    ),
                    # [Argmax socket deferral — main kernel handles all socket sends]
                    ("argmax_defer_socket_output", 1 if enable_socket_output else 0),
                    # [Cross-core verification semaphore]
                    (
                        "verify_output_ready_semaphore_id",
                        verify_output_ready_semaphore_id if enable_mtp_verification else 0,
                    ),
                    # [Skip pipeline semaphore (broadcast-based skip)]
                    (
                        "skip_pipeline_semaphore_id",
                        skip_pipeline_semaphore_id if enable_mtp_verification else 0,
                    ),
                    # [MTP] EH matmul done semaphore (push-based logits gather)
                    ("eh_matmul_done_semaphore_id", eh_matmul_done_semaphore_id if enable_mtp_on_device else 0),
                ]

                # ================================================================
                # BRISC compile-time args
                # ================================================================
                brisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("enable_argmax", 1),
                    ("bcast_cb0_id", brisc_bcast_cb),
                    ("bcast_num_pages_to_read", brisc_bcast_num_pages_to_read),
                    ("bcast_is_sender", int(is_sender) if brisc_is_active else 0),
                    ("input_socket_mode", input_socket_mode),
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
                    ("mcast_is_part_of_receiver_grid", is_part_of_receiver_grid),
                    ("argmax_winner_page_bytes", argmax_winner_page_bytes),
                    ("argmax_local_ready_semaphore_id", argmax_local_ready_semaphore_id),
                    ("argmax_socket_mode", argmax_socket_mode),
                    ("argmax_socket_cb", argmax_socket_cb if enable_socket_output else 0),
                    ("argmax_socket_page_size_bytes", socket_page_size_bytes if enable_socket_output else 0),
                    ("persistent_mode", 1 if persistent_mode else 0),
                    ("mesh_row", row),
                    ("mesh_col", col),
                    # [MTP] Second mcast (EH projection input)
                    ("enable_mtp", 1 if enable_mtp_on_device else 0),
                    ("enable_mtp_verification", 1 if enable_mtp_verification else 0),
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
                    ("mcast_eh_src_cb", mcast_eh_src_cb if enable_mtp_on_device else 0),
                    ("mcast_eh_dst_cb", mcast_eh_dst_cb if enable_mtp_on_device else 0),
                    ("mcast_eh_data_size_bytes", eh_mcast_data_size_bytes if enable_mtp_on_device else 0),
                    ("mcast_eh_src_num_pages", eh_concat_rms_tiles if enable_mtp_on_device else 0),
                    ("has_mtp_logits_socket_output", 1 if has_mtp_logits_socket_on_device else 0),
                    # [MTP] EH matmul output CB (for BRISC push)
                    ("matmul_eh_out", matmul_out_eh_cb),
                    ("matmul_eh_out_w", eh_out_w_per_core if enable_mtp_on_device else 0),
                    ("mtp_logits_num_eh_cores", num_eh_matmul_cores if enable_mtp_on_device else 0),
                    ("mtp_logits_eh_shard_size", eh_shard_size_bytes if enable_mtp_on_device else 0),
                    (
                        "mtp_logits_payload_size",
                        packet_size_bytes if (enable_mtp_on_device or enable_mtp_verification) else 0,
                    ),
                    ("argmax_defer_socket_output", 1 if enable_socket_output else 0),
                    (
                        "verify_output_ready_semaphore_id",
                        verify_output_ready_semaphore_id if enable_mtp_verification else 0,
                    ),
                    # [Skip pipeline semaphore (broadcast-based skip)]
                    (
                        "skip_pipeline_semaphore_id",
                        skip_pipeline_semaphore_id if enable_mtp_verification else 0,
                    ),
                    # [MTP] Push-based logits gather: EH cores write shards to argmax core
                    ("eh_matmul_done_semaphore_id", eh_matmul_done_semaphore_id if enable_mtp_on_device else 0),
                    ("mtp_logits_argmax_core_noc_x", int(argmax_final_core_phys.x) if enable_mtp_on_device else 0),
                    ("mtp_logits_argmax_core_noc_y", int(argmax_final_core_phys.y) if enable_mtp_on_device else 0),
                    ("mtp_logits_argmax_staging_addr", mcast_receiver_data_addr if enable_mtp_on_device else 0),
                ]

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
                    ("mesh_row", row),
                    ("mesh_col", col),
                    # [MTP] h_rmsnorm and e_rmsnorm CTArgs
                    # Both RMSNorms write directly to mcast_eh_src_cb to avoid concat copy:
                    # h_rmsnorm writes first half (h_tiles), e_rmsnorm writes second half (e_tiles)
                    ("enable_mtp", 1 if enable_mtp_on_device else 0),
                    ("enable_mtp_verification", 1 if enable_mtp_verification else 0),
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
                    (
                        "eh_matmul_cb_in1_buf_addr",
                        ttnn.get_device_tensors(eh_proj_working_buf_tensor)[device_idx].buffer_address()
                        if (enable_mtp_on_device and eh_proj_working_buf_tensor is not None)
                        else 0,
                    ),
                    ("mtp_embedding_done_cb", mtp_embedding_done_cb if enable_mtp_on_device else 0),
                    ("has_mtp_logits_socket_output", 1 if has_mtp_logits_socket_on_device else 0),
                    ("mtp_logits_num_eh_cores", 0),
                    ("mtp_logits_eh_shard_size", 0),
                    (
                        "mtp_logits_payload_size",
                        packet_size_bytes if (enable_mtp_on_device or enable_mtp_verification) else 0,
                    ),
                    ("argmax_defer_socket_output", 1 if enable_socket_output else 0),
                    (
                        "verify_output_ready_semaphore_id",
                        verify_output_ready_semaphore_id if enable_mtp_verification else 0,
                    ),
                    # [Skip pipeline semaphore (broadcast-based skip)]
                    (
                        "skip_pipeline_semaphore_id",
                        skip_pipeline_semaphore_id if enable_mtp_verification else 0,
                    ),
                ]

                # ================================================================
                # CCL Broadcast common runtime args
                # ================================================================
                if skip_ccl:
                    final_core_phys = argmax_final_core_phys
                    ncrisc_bcast_common_args = [
                        int(indices_tensor_device.buffer_address()),
                        int(output_index_tensor_device.buffer_address()),
                        int(final_core_phys.x),
                        int(final_core_phys.y),
                        0,
                        0,
                        0,
                    ]
                    # [MTP] All cores consume 4 alignment args + input_core gets embedding_base
                    if enable_mtp_on_device:
                        sender_core_phys = device.worker_core_from_logical_core(mcast_sender_core)
                        mtp_token_addr = int(intermediate_tensor_device.buffer_address())
                        embedding_tensor_device = embedding_tensors_per_device[device_idx]
                        ncrisc_bcast_common_args += [
                            int(sender_core_phys.x),  # input_core_noc_x
                            int(sender_core_phys.y),  # input_core_noc_y
                            mtp_token_addr,  # mtp_token_addr = intermediate_tensor buffer
                            int(embedding_tensor_device.buffer_address()),  # embedding DRAM base addr
                        ]
                    # [MTP LOGITS GATHER] staging addr for logits + metadata on argmax core
                    if enable_mtp_on_device:
                        ncrisc_bcast_common_args += [
                            mcast_receiver_data_addr,
                        ]
                    # [MTP LOGITS SOCKET] socket config addr (only when socket is present)
                    if has_mtp_logits_socket_on_device:
                        ncrisc_bcast_common_args += [
                            int(mtp_logits_socket_output.get_config_buffer_address()),
                        ]
                    # [MTP Verification] reference token addr, unverified spec addr, verified spec addr
                    if enable_mtp_verification:
                        ref_token_dev = ttnn.get_device_tensors(reference_token_tensor)[device_idx]
                        ncrisc_bcast_common_args.append(int(ref_token_dev.buffer_address()))
                        if unverified_spec_tensor is not None:
                            unverified_dev = ttnn.get_device_tensors(unverified_spec_tensor)[device_idx]
                            ncrisc_bcast_common_args.append(int(unverified_dev.buffer_address()))
                        else:
                            ncrisc_bcast_common_args.append(0)
                        if verified_spec_tensor is not None:
                            verified_dev = ttnn.get_device_tensors(verified_spec_tensor)[device_idx]
                            ncrisc_bcast_common_args.append(int(verified_dev.buffer_address()))
                        else:
                            ncrisc_bcast_common_args.append(0)
                        # Cross-core verification: staging buffer addr, argmax output addr,
                        # argmax noc coords, input_core noc coords
                        if verify_output_staging_tensor is not None:
                            staging_dev = ttnn.get_device_tensors(verify_output_staging_tensor)[device_idx]
                            ncrisc_bcast_common_args.append(int(staging_dev.buffer_address()))
                        else:
                            ncrisc_bcast_common_args.append(0)
                        ncrisc_bcast_common_args.append(int(output_index_tensor_device.buffer_address()))
                        ncrisc_bcast_common_args.append(int(final_core_phys.x))
                        ncrisc_bcast_common_args.append(int(final_core_phys.y))
                        input_core_phys = device.worker_core_from_logical_core(mcast_sender_core)
                        ncrisc_bcast_common_args.append(int(input_core_phys.x))
                        ncrisc_bcast_common_args.append(int(input_core_phys.y))
                    # Compute combined page size for spec stage (logits + 64 bytes metadata)
                    bcast_socket_page_size = packet_size_bytes
                    if enable_mtp_verification and recv_socket_on_this_device:
                        bcast_socket_page_size = packet_size_bytes + 64
                    brisc_bcast_common_args = [
                        int(final_core_phys.x),
                        int(final_core_phys.y),
                        0,
                        int(socket_output.get_config_buffer_address()) if enable_socket_output else 0,
                        int(socket_input.get_config_buffer_address()) if recv_socket_on_this_device else 0,
                        bcast_socket_page_size if recv_socket_on_this_device else 0,
                        1 if recv_socket_on_this_device else 0,
                        persistent_enable,
                        int(persistent_target_input_core_phys.x),
                        int(persistent_target_input_core_phys.y),
                        int(persistent_target_node.mesh_id),
                        int(persistent_target_node.chip_id),
                        persistent_next_iter_global_sem_addr,
                    ]
                    dst_nodes = []
                    fabric_node_id = None
                else:
                    wait_output_semaphore = is_secondary_sender or is_receiver
                    reset_global_semaphore = is_secondary_sender or is_receiver
                    out_ready_sem_wait_value = 1 * num_links

                    # Build dst_nodes to compute num_connections = len(dst_nodes)
                    fabric_node_id = mesh_device.get_fabric_node_id(coord)
                    dst_nodes = []

                    # Primary axis connections (forward and backward in column)
                    if num_targets_forward > 0:
                        forward_coord = ttnn.MeshCoordinate(row + 1, col)
                        dst_nodes.append(mesh_device.get_fabric_node_id(forward_coord))

                    if num_targets_backward > 0:
                        backward_coord = ttnn.MeshCoordinate(row - 1, col)
                        dst_nodes.append(mesh_device.get_fabric_node_id(backward_coord))

                    # Secondary axis connection (for sender to secondary sender)
                    if has_secondary_target:
                        secondary_coord = ttnn.MeshCoordinate(row, 1 - col)
                        dst_nodes.append(mesh_device.get_fabric_node_id(secondary_coord))

                    num_connections = len(dst_nodes)

                    ncrisc_bcast_common_args = [
                        int(intermediate_tensor_device.buffer_address()),  # tensor_address0
                        int(out_ready_sem_addr),  # out_ready_sem_bank_addr
                        int(wait_output_semaphore),  # wait_output_semaphore
                        int(reset_global_semaphore),  # reset_global_semaphore
                        core_noc_x,  # out_ready_sem_noc0_x
                        core_noc_y,  # out_ready_sem_noc0_y
                        out_ready_sem_wait_value,  # out_ready_sem_wait_value
                        int(barrier_sem_addr),  # barrier_sem
                        core_noc_x,  # barrier_sem_noc0_x
                        core_noc_y,  # barrier_sem_noc0_y
                        ring_index,  # ring_index
                        int(secondary_sync_sem_addr),  # secondary_sync_sem
                        num_connections,  # num_connections
                    ]

                    final_core_phys = argmax_final_core_phys
                    ncrisc_bcast_common_args = ncrisc_bcast_common_args + [
                        int(indices_tensor_device.buffer_address()),
                        int(output_index_tensor_device.buffer_address()),
                        int(final_core_phys.x),
                        int(final_core_phys.y),
                        int(scratch_tensors_per_device[device_idx].buffer_address()),
                        global_sem_addr,
                        global_stage2_sem_addr,
                    ]
                    # [MTP] All cores consume 4 alignment args + input_core gets embedding_base
                    if enable_mtp_on_device:
                        sender_core_phys = device.worker_core_from_logical_core(mcast_sender_core)
                        mtp_token_addr = int(intermediate_tensor_device.buffer_address())
                        embedding_tensor_device = embedding_tensors_per_device[device_idx]
                        ncrisc_bcast_common_args += [
                            int(sender_core_phys.x),  # input_core_noc_x
                            int(sender_core_phys.y),  # input_core_noc_y
                            mtp_token_addr,  # mtp_token_addr = intermediate_tensor buffer
                            int(embedding_tensor_device.buffer_address()),  # embedding DRAM base addr
                        ]
                    # [MTP LOGITS GATHER] staging addr for logits + metadata on argmax core
                    if enable_mtp_on_device:
                        ncrisc_bcast_common_args += [
                            mcast_receiver_data_addr,
                        ]
                    # [MTP LOGITS SOCKET] socket config addr (only when socket is present)
                    if has_mtp_logits_socket_on_device:
                        ncrisc_bcast_common_args += [
                            int(mtp_logits_socket_output.get_config_buffer_address()),
                        ]
                    # [MTP Verification] reference token addr, unverified spec addr, verified spec addr
                    if enable_mtp_verification:
                        ref_token_dev = ttnn.get_device_tensors(reference_token_tensor)[device_idx]
                        ncrisc_bcast_common_args.append(int(ref_token_dev.buffer_address()))
                        if unverified_spec_tensor is not None:
                            unverified_dev = ttnn.get_device_tensors(unverified_spec_tensor)[device_idx]
                            ncrisc_bcast_common_args.append(int(unverified_dev.buffer_address()))
                        else:
                            ncrisc_bcast_common_args.append(0)
                        if verified_spec_tensor is not None:
                            verified_dev = ttnn.get_device_tensors(verified_spec_tensor)[device_idx]
                            ncrisc_bcast_common_args.append(int(verified_dev.buffer_address()))
                        else:
                            ncrisc_bcast_common_args.append(0)
                        # Cross-core verification: staging buffer addr, argmax output addr,
                        # argmax noc coords, input_core noc coords
                        if verify_output_staging_tensor is not None:
                            staging_dev = ttnn.get_device_tensors(verify_output_staging_tensor)[device_idx]
                            ncrisc_bcast_common_args.append(int(staging_dev.buffer_address()))
                        else:
                            ncrisc_bcast_common_args.append(0)
                        ncrisc_bcast_common_args.append(int(output_index_tensor_device.buffer_address()))
                        ncrisc_bcast_common_args.append(int(final_core_phys.x))
                        ncrisc_bcast_common_args.append(int(final_core_phys.y))
                        input_core_phys = device.worker_core_from_logical_core(mcast_sender_core)
                        ncrisc_bcast_common_args.append(int(input_core_phys.x))
                        ncrisc_bcast_common_args.append(int(input_core_phys.y))
                    # Compute combined page size for spec stage (logits + 64 bytes metadata)
                    bcast_socket_page_size = packet_size_bytes
                    if enable_mtp_verification and recv_socket_on_this_device:
                        bcast_socket_page_size = packet_size_bytes + 64
                    brisc_bcast_common_args = [
                        int(final_core_phys.x),
                        int(final_core_phys.y),
                        int(scratch_tensors_per_device[device_idx].buffer_address()),
                        int(socket_output.get_config_buffer_address()) if enable_socket_output else 0,
                        int(socket_input.get_config_buffer_address()) if recv_socket_on_this_device else 0,
                        bcast_socket_page_size if recv_socket_on_this_device else 0,
                        1 if recv_socket_on_this_device else 0,
                        persistent_enable,
                        int(persistent_target_input_core_phys.x),
                        int(persistent_target_input_core_phys.y),
                        int(persistent_target_node.mesh_id),
                        int(persistent_target_node.chip_id),
                        persistent_next_iter_global_sem_addr,
                    ]

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

                # CB 7: RMSNorm gamma — tensor-backed on sender core.
                rmsnorm_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    rmsnorm_gamma_cb, gamma_tensor_device
                )
                rmsnorm_gamma_cb_descriptor.format_descriptors[0].tile = rms_tile_descriptor
                rmsnorm_gamma_cb_descriptor.format_descriptors[0].page_size = rms_tile_size

                # CB 8: RMSNorm output — shares backing tensor with CB 0 (rmsnorm input).
                # Safe because TRISC fully consumes input into dest registers and pops CB 0
                # before reserving CB 8 for output (see rmsnorm.hpp compute_rmsnorm).
                rms_out_tile_descriptor = ttnn.TileDescriptor(rms_interpreted_tile)
                rmsnorm_out_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=mcast_src_cb,
                    data_format=data_format,
                    page_size=rms_tile_size,
                    tile=rms_out_tile_descriptor,
                )
                rmsnorm_out_cb_descriptor = ttnn.CBDescriptor(
                    total_size=rms_num_tiles * rms_tile_size,
                    core_ranges=mcast_sender_core_grid,
                    format_descriptors=[rmsnorm_out_cb_format],
                )

                # CB 1: Mcast destination — tensor-backed on receiver cores (not the sender).
                # Sender obtains the receiver data address via buffer_address() runtime arg.
                # mcast_receiver_data_addr was pre-computed above for both runtime args and CB setup.
                if mcast_dst_working_buf_tensor is not None:
                    mcast_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        mcast_dst_cb, ttnn.get_device_tensors(mcast_dst_working_buf_tensor)[device_idx]
                    )
                    mcast_dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
                    mcast_dst_cb_descriptor.format_descriptors[0].tile = mcast_dst_tile_descriptor
                    mcast_dst_cb_descriptor.format_descriptors[0].page_size = input_tile_size
                else:
                    mcast_dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
                    mcast_dst_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=mcast_dst_cb,
                        data_format=data_format,
                        page_size=input_tile_size,
                        tile=mcast_dst_tile_descriptor,
                    )
                    mcast_dst_cb_descriptor = ttnn.CBDescriptor(
                        total_size=num_tiles_k * input_tile_size,
                        core_ranges=all_cores,
                        format_descriptors=[mcast_dst_cb_format],
                    )

                # CB 2: Matmul weights — vocab_tensor, tensor-backed on matmul cores
                matmul_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_in1_cb, vocab_tensor_device)

                if enable_argmax:
                    # CB 5: Argmax indices — tensor-backed on matmul/argmax cores
                    argmax_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        argmax_indices_cb, indices_tensor_device
                    )

                # CB 16: Matmul output — tensor-backed on matmul cores
                matmul_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_out_cb, output_tensor_device)

                # [MTP] CB descriptors (only on exit device)
                mtp_cb_descriptors = []
                if enable_mtp_on_device:
                    # CB 11: h_gamma - tensor-backed on sender core
                    h_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        h_gamma_cb, h_gamma_tensors_per_device[device_idx]
                    )
                    h_gamma_cb_descriptor.format_descriptors[0].tile = rms_tile_descriptor
                    h_gamma_cb_descriptor.format_descriptors[0].page_size = rms_tile_size

                    # CB 12: e_gamma - tensor-backed on sender core
                    e_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        e_gamma_cb, e_gamma_tensors_per_device[device_idx]
                    )
                    e_gamma_cb_descriptor.format_descriptors[0].tile = rms_tile_descriptor
                    e_gamma_cb_descriptor.format_descriptors[0].page_size = rms_tile_size

                    # CB 15: mcast_eh_src - fused [h_norm|e_norm] on sender core
                    # Both h_rmsnorm and e_rmsnorm write directly here (no separate e_norm_cb needed)
                    # Uses packed RMS tile format (7+7=14 tiles of 32x32) matching RMSNorm output format
                    mcast_eh_tile_descriptor = ttnn.TileDescriptor(rms_interpreted_tile)
                    mcast_eh_src_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=mcast_eh_src_cb,
                        data_format=data_format,
                        page_size=rms_tile_size,
                        tile=mcast_eh_tile_descriptor,
                    )
                    mcast_eh_src_cb_descriptor = ttnn.CBDescriptor(
                        total_size=eh_concat_rms_tiles * rms_tile_size,
                        core_ranges=mcast_sender_core_grid,
                        format_descriptors=[mcast_eh_src_cb_format],
                    )

                    # CB 18: mcast_eh_dst — tensor-backed on receiver cores (not the sender).
                    # Sender obtains the receiver data address via buffer_address() runtime arg.
                    if mcast_eh_dst_working_buf_tensor is not None:
                        mcast_eh_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                            mcast_eh_dst_cb, ttnn.get_device_tensors(mcast_eh_dst_working_buf_tensor)[device_idx]
                        )
                        eh_dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
                        mcast_eh_dst_cb_descriptor.format_descriptors[0].tile = eh_dst_tile_descriptor
                        mcast_eh_dst_cb_descriptor.format_descriptors[0].page_size = input_tile_size
                        mcast_eh_receiver_data_addr = int(
                            ttnn.get_device_tensors(mcast_eh_dst_working_buf_tensor)[device_idx].buffer_address()
                        )
                    else:
                        eh_dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
                        mcast_eh_dst_cb_format = ttnn.CBFormatDescriptor(
                            buffer_index=mcast_eh_dst_cb,
                            data_format=data_format,
                            page_size=input_tile_size,
                            tile=eh_dst_tile_descriptor,
                        )
                        mcast_eh_dst_cb_descriptor = ttnn.CBDescriptor(
                            total_size=eh_num_tiles_k * input_tile_size,
                            core_ranges=all_cores,
                            format_descriptors=[mcast_eh_dst_cb_format],
                        )
                        mcast_eh_receiver_data_addr = 0

                    # CB 9: EH projection weights - tensor-backed working buffer for DRAM streaming (triple-buffered)
                    if eh_proj_working_buf_tensor is not None:
                        # Use tensor-backed CB for working buffer
                        matmul_eh_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                            matmul_eh_cb, ttnn.get_device_tensors(eh_proj_working_buf_tensor)[device_idx]
                        )
                    else:
                        # Fallback to regular CB (shouldn't happen if working buffer is provided)
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

                    # CB 17: EH matmul output - tensor-backed on matmul cores
                    matmul_out_eh_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        matmul_out_eh_cb, output_mtp_tensors_per_device[device_idx]
                    )

                    emb_done_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=mtp_embedding_done_cb,
                        data_format=data_format,
                        page_size=16,
                    )
                    emb_done_cb_descriptor = ttnn.CBDescriptor(
                        total_size=16,
                        core_ranges=mcast_sender_core_grid,
                        format_descriptors=[emb_done_cb_format],
                    )

                    mtp_cb_descriptors = [
                        h_gamma_cb_descriptor,
                        e_gamma_cb_descriptor,
                        mcast_eh_src_cb_descriptor,
                        mcast_eh_dst_cb_descriptor,
                        matmul_eh_cb_descriptor,
                        matmul_out_eh_cb_descriptor,
                        emb_done_cb_descriptor,
                    ]

                # CB list
                cbs_list = [
                    rmsnorm_input_cb_descriptor,
                    mcast_dst_cb_descriptor,
                    matmul_in1_cb_descriptor,
                    rmsnorm_gamma_cb_descriptor,
                    rmsnorm_out_cb_descriptor,
                    matmul_out_cb_descriptor,
                ]
                if enable_mtp_on_device:
                    cbs_list.extend(mtp_cb_descriptors)
                if enable_argmax:
                    argmax_winner_cb_descriptor = ttnn.CBDescriptor(
                        total_size=argmax_winner_page_bytes,
                        core_ranges=argmax_core_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=argmax_winner_cb,
                                data_format=ttnn.uint32,
                                page_size=argmax_winner_page_bytes,
                            )
                        ],
                    )
                    argmax_gather_cb_descriptor = ttnn.CBDescriptor(
                        total_size=argmax_winner_page_bytes * argmax_num_senders,
                        core_ranges=argmax_core_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=argmax_gather_cb,
                                data_format=ttnn.uint32,
                                page_size=argmax_winner_page_bytes,
                            )
                        ],
                    )
                    cbs_list.extend(
                        [
                            argmax_winner_cb_descriptor,
                            argmax_gather_cb_descriptor,
                            argmax_indices_cb_descriptor,
                        ]
                    )
                    if enable_socket_output:
                        argmax_socket_cb_descriptor = ttnn.CBDescriptor(
                            total_size=socket_page_size_bytes,
                            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)]),
                            format_descriptors=[
                                ttnn.CBFormatDescriptor(
                                    buffer_index=argmax_socket_cb,
                                    data_format=ttnn.uint32,
                                    page_size=socket_page_size_bytes,
                                )
                            ],
                        )
                        cbs_list.append(argmax_socket_cb_descriptor)

                # CB 30: CCL broadcast packet buffer (only in multi-device mode)
                # For spec stage (verification), CB30 holds logits + 64 bytes metadata.
                if not skip_ccl:
                    bcast_pkt_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bcast_pkt_cb, input_tensor_device)
                    if enable_mtp_verification:
                        bcast_pkt_cb_descriptor.total_size += 64
                    cbs_list.append(bcast_pkt_cb_descriptor)

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
                        ]
                    )
                if enable_mtp_on_device:
                    semaphore_descriptors.extend(
                        [
                            ttnn.SemaphoreDescriptor(
                                id=mtp_ready_semaphore_id,
                                core_ranges=mcast_sender_core_grid,
                                initial_value=0,
                            ),
                            ttnn.SemaphoreDescriptor(
                                id=mcast_eh_data_sender_semaphore_id,
                                core_ranges=all_cores,
                                initial_value=0,
                            ),
                            ttnn.SemaphoreDescriptor(
                                id=mcast_eh_data_receiver_semaphore_id,
                                core_ranges=all_cores,
                                initial_value=0,
                            ),
                            ttnn.SemaphoreDescriptor(
                                id=eh_matmul_done_semaphore_id,
                                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)]),
                                initial_value=0,
                            ),
                        ]
                    )
                if enable_mtp_verification:
                    semaphore_descriptors.extend(
                        [
                            ttnn.SemaphoreDescriptor(
                                id=verify_output_ready_semaphore_id,
                                core_ranges=all_cores,
                                initial_value=0,
                            ),
                            ttnn.SemaphoreDescriptor(
                                id=skip_pipeline_semaphore_id,
                                core_ranges=all_cores,
                                initial_value=0,
                            ),
                        ]
                    )

                # Append mcast receiver data addresses as BRISC runtime args [13] and [14]
                # (0 = kernel uses get_write_ptr fallback)
                brisc_bcast_common_args.append(mcast_receiver_data_addr)
                brisc_bcast_common_args.append(
                    (mcast_eh_receiver_data_addr if mcast_eh_dst_working_buf_tensor is not None else 0)
                    if enable_mtp_on_device
                    else 0
                )

                # ================================================================
                # Unified kernel descriptor
                # ================================================================
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/lm_head_sampling/kernels/lm_head_sampling_kernel.cpp",
                    core_ranges=all_cores,
                    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
                    brisc_named_compile_time_args=brisc_named_compile_time_args,
                    trisc_named_compile_time_args=trisc_named_compile_time_args,
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
                            named_compile_time_arg="is_argmax_core",
                            core_range=argmax_core_grid,
                            value=1 if enable_argmax else 0,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_argmax_final_core",
                            core_range=argmax_final_core,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_argmax_mesh_sender_core",
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
                            named_compile_time_arg="enable_mtp_verification",
                            core_range=all_cores,
                            value=1 if enable_mtp_verification else 0,
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
                                    named_compile_time_arg="argmax_sender_idx",
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
                            PerCoreCompileTimeDescriptor(
                                named_compile_time_arg="mtp_logits_eh_core_index",
                                core_values=[(core, idx) for idx, core in enumerate(eh_worker_cores)]
                                if enable_mtp_on_device
                                else [],
                                other_value=0,
                            ),
                        ]
                    ),
                    # Per-core runtime args: mesh argmax senders get BRISC sender metadata.
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(worker_core, [])],
                        brisc_args=[(worker_core, [])] + per_core_brisc_runtime_args,
                    ),
                    defines=[("ENABLE_SOCKET_READER", "1")] if enable_socket_input else [],
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
                    if group.compile_time_arg_values.get("is_argmax_core", 0) == 1:
                        argmax_role_cores.update((c.x, c.y) for c in group_cores)
                    if group.compile_time_arg_values.get("is_argmax_final_core", 0) == 1:
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
                            f"is_argmax_core core-set mismatch. missing={missing}, extra={extra}"
                        )
                    if len(argmax_final_role_cores) != 1:
                        raise RuntimeError(
                            "Unified kernel role mapping mismatch: " "is_argmax_final_core must map to exactly one core"
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

                    if num_connections > 0:
                        fabric_args = ttnn.setup_routing_plane_connection(
                            fabric_node_id, dst_nodes, [0], program, writer_kernel_idx, worker_core
                        )
                        writer_rt_args_ref.extend(fabric_args)

                if not skip_ccl and is_argmax_mesh_sender_core:
                    sender_group = kernel_result.get_group_by_arg("is_argmax_mesh_sender_core", 1)
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
                    persistent_group = kernel_result.get_group_by_arg("is_argmax_final_core", 1)
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

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program
        # Execute generic op
        io_tensors = [input_tensor_mesh, intermediate_tensor_mesh, gamma_tensor, vocab_tensor, output_tensor]
        io_tensors.extend([indices_tensor, output_index_tensor])
        if enable_mtp:
            io_tensors.extend(
                [output_mtp_tensor, embedding_tensor, h_gamma_tensor, e_gamma_tensor, eh_projection_tensor]
            )
            if eh_proj_working_buf_tensor is not None:
                io_tensors.append(eh_proj_working_buf_tensor)
            if mcast_eh_dst_working_buf_tensor is not None:
                io_tensors.append(mcast_eh_dst_working_buf_tensor)
        if mcast_dst_working_buf_tensor is not None:
            io_tensors.append(mcast_dst_working_buf_tensor)
        if enable_mtp_verification:
            io_tensors.append(reference_token_tensor)
            if unverified_spec_tensor is not None:
                io_tensors.append(unverified_spec_tensor)
            if verified_spec_tensor is not None:
                io_tensors.append(verified_spec_tensor)
            if verify_output_staging_tensor is not None:
                io_tensors.append(verify_output_staging_tensor)
        if not skip_ccl:
            io_tensors.append(fabric_scratch_tensor)
        result = ttnn.generic_op(io_tensors, mesh_program_descriptor)
        return result
