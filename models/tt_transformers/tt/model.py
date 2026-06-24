# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from tqdm import tqdm

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.common.sampling.generator import SamplingGenerator
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding, ScaledEmbedding
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import TensorGroup
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup


class Transformer(LightweightModule):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        rope_setup_class=None,
        prefetcher=None,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.model_config = args.get_model_config()
        self.grid_size = self.args.max_grid_size
        state_dict_prefix = args.get_state_dict_prefix("", None)
        self.decoders_optimizations = args.decoders_optimizations
        self.prefetcher = prefetcher
        self.tt_ccl = TT_CCL(self.mesh_device)

        embd_kwargs = {
            "mesh_device": mesh_device,
            "args": args,
            "weight_cache_path": args.weight_cache_path(dtype),
            "state_dict": state_dict,
            "dtype": ttnn.bfloat16,  # Row major layout requires bfloat16
        }
        if self.args.embed_scale is not None:
            embd_cls = ScaledEmbedding
            embd_kwargs["embed_scale"] = self.args.embed_scale
        else:
            embd_cls = Embedding
        self.embd = embd_cls(**embd_kwargs)

        DefaultRopeSetup = HfRotarySetup if self.args.use_hf_rope else RotarySetup
        ActualRopeSetupClass = rope_setup_class if rope_setup_class is not None else DefaultRopeSetup
        self.rope_setup = ActualRopeSetupClass(
            device=mesh_device,
            batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,
            rope_scaling=args.rope_scaling,
            use_qk_fused=args.use_qk_fused,
            prefetcher=prefetcher,
        )

        if args.rope_theta_local:
            self.rope_local_setup = DefaultRopeSetup(
                mesh_device,
                args.max_batch_size,
                args.head_dim,
                args.max_seq_len,
                args.rope_theta_local,
                use_qk_fused=args.use_qk_fused,
                prefetcher=None,
            )

        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()

        # Device tensors used to build dynamic slice params for prefill RoPE slicing.
        # Keeps chunk_start_idx-driven slicing inside the traced graph.
        self._tt_seq_len_buffer = ttnn.from_torch(
            torch.tensor([1, 1, self.args.max_seq_len, self.args.head_dim], dtype=torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._tt_slice_start_zeros_4 = ttnn.from_torch(
            torch.tensor([0, 0, 0, 0], dtype=torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        self.layers = [
            TransformerBlock(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                transformation_mats=self.trans_mats_dict,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                attention_class=attention_class,
                prefetcher=prefetcher,
            )
            for i in tqdm(range(self.n_layers))
        ]
        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", None),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="norm",
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                is_distributed=self.args.is_distributed_norm,
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            prefetcher=prefetcher,
            TG=args.is_galaxy,
        )

        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            max_columns_per_device=self.args.max_columns_per_device_lm_head,
            prefetcher=prefetcher,
        )

        # Initialize on-device sampling if supported
        # Sampling on device is supported only if each device has maximum logits size of 64*1024
        sampling_splits = self.args.num_devices if list(self.mesh_device.shape) != [1, 1] else 2
        self._supports_on_device_sampling = prefetcher is None and self.args.vocab_size // sampling_splits <= 64 * 1024
        if self._supports_on_device_sampling:
            self.sampling = SamplingGenerator(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
            )
        else:
            self.sampling = None

    def process_logits_after_prefill_trace(self, logits, last_token_idx):
        get_last_token = (last_token_idx // 32) * 32
        logits = ttnn.slice(
            logits,
            (0, 0, get_last_token, 0),
            (1, 1, get_last_token + 32, logits.shape[-1]),
        )
        logits = self._apply_norm_and_lm_head(logits)
        return logits

    def extract_last_tokens_batched_prefill(
        self, hidden_states, last_token_idx_list, padded_batch, prefill_seq_len, target_batch=None
    ):
        """Extract each user's last-token hidden state from batched prefill output.

        Reads hidden states to host, extracts the relevant row for each user,
        and sends the combined tensor back to device with the correct column-sharded
        mesh mapping (ShardTensorToMesh dim=-1) so the DistributedNorm all-gather
        produces the correct full hidden dim.

        Args:
            hidden_states: [padded_batch, 1, prefill_seq_len, dim_per_device] on device (column-sharded, TILE_LAYOUT)
            last_token_idx_list: list of length padded_batch with per-user last token positions
            padded_batch: number of slots (typically 32)
            prefill_seq_len: padded sequence length per user

        Returns:
            user_tokens: [1, 1, target_batch or padded_batch, dim_per_device] per device,
            column-sharded, TILE_LAYOUT
        """
        active_indices = [lt for lt in last_token_idx_list if lt > 0]
        all_same = len(set(active_indices)) <= 1

        if all_same and active_indices:
            common_last = active_indices[0]
            get_last = (common_last // 32) * 32
            R = common_last % 32
            block = ttnn.slice(
                hidden_states,
                (0, 0, get_last, 0),
                (padded_batch, 1, get_last + 32, hidden_states.shape[-1]),
            )
        else:
            block = hidden_states
            R = None

        host_tensors = [ttnn.to_torch(dt) for dt in ttnn.get_device_tensors(block)]
        host_full = torch.cat(host_tensors, dim=-1)

        if R is not None:
            combined = host_full[:, :, R : R + 1, :].reshape(1, 1, padded_batch, -1).contiguous()
        else:
            rows = []
            for slot in range(padded_batch):
                lt_idx = last_token_idx_list[slot]
                rows.append(host_full[slot : slot + 1, :, lt_idx : lt_idx + 1, :])
            combined = torch.cat(rows, dim=0).reshape(1, 1, padded_batch, -1).contiguous()

        target_batch = padded_batch if target_batch is None else target_batch
        if target_batch < padded_batch:
            raise ValueError(f"target_batch {target_batch} must be >= padded_batch {padded_batch}")
        if target_batch > padded_batch:
            padded_combined = torch.zeros(
                1,
                1,
                target_batch,
                combined.shape[-1],
                dtype=combined.dtype,
            )
            padded_combined[:, :, :padded_batch, :] = combined
            combined = padded_combined

        user_tokens = ttnn.from_torch(
            combined,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        )
        return user_tokens

    def process_logits_after_batched_prefill(self, hidden_states, last_token_idx_list, padded_batch, prefill_seq_len):
        """Extract last tokens and run norm + lm_head once for all users."""
        user_tokens = self.extract_last_tokens_batched_prefill(
            hidden_states, last_token_idx_list, padded_batch, prefill_seq_len
        )
        return self._apply_norm_and_lm_head(user_tokens)

    def _apply_norm_and_lm_head(self, x):
        """Shared norm + lm_head for prefill logit processing. Input: [1, 1, 32, hidden_dim]."""
        x = self.norm(
            x, mode=Mode.PREFILL, norm_config=self.args.get_norm_config("lm_head", Mode.PREFILL, self.prefetcher)
        )
        lm_head_input_mem_cfg = self.args.get_lm_head_input_mem_config(Mode.PREFILL, None)
        if lm_head_input_mem_cfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)
        logits = self.lm_head(x)
        logits = ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def process_hidden_states_after_prefill_trace(self, hidden_states, last_token_idx):
        """
        Process hidden states after prefill trace, stopping before LM head.
        Returns hidden states (after norm) instead of logits.
        Used for embedding models that need hidden states rather than logits.
        """
        get_last_token = (last_token_idx // 32) * 32
        hidden_states = ttnn.slice(
            hidden_states,
            (0, 0, get_last_token, 0),
            (1, 1, get_last_token + 32, hidden_states.shape[-1]),
        )
        # Apply norm (this is the final layer norm before LM head)
        hidden_states = self.norm(hidden_states, mode="prefill")
        # Convert to row major layout for output (but don't apply LM head)
        hidden_states = ttnn.to_layout(
            hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return hidden_states

    def prepare_prefill_inputs_trace(
        self,
        tokens,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=0,
        batch_size=1,
        user_id=0,
        **kwargs,
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on host.
        """
        host_inputs = self.prepare_inputs_prefill(
            tokens,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            trace_enabled=True,
            batch_size=batch_size,
            user_id=user_id,
        )
        return host_inputs

    def transform_and_embed_prefill_inputs_device(
        self,
        tokens,
        tt_page_table,
        tt_chunk_page_table,
        tt_chunk_start_idx,
    ):
        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        return tt_tokens, tt_page_table, tt_chunk_page_table, tt_chunk_start_idx

    def prepare_inputs_prefill(
        self,
        tokens,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        trace_enabled=False,
        last_token_idx=None,
        global_user_id=None,
        batch_size=1,
        user_id=0,
        **kwargs,
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device if trace is disabled or on host if trace is enabled.
        TODO: Debate whether this function is responsible for padding
        """

        # We set the device to None if trace is enabled so we keep the tensors on host instead of sending it to the device (None - keeps on host, device - sends to specified device)
        # We will send them to device later (copy_host_to_device)
        device = None if trace_enabled else self.mesh_device

        assert tokens.dim() == 2, "tokens must be a 2D tensor"
        # For batched prefill, tokens come in as [padded_batch, S]
        # Each user's tokens are at their slot index in dimension 0
        # Reshape to [1, 1, 1, padded_batch * S] for embedding
        if batch_size > 1:
            # Tokens are in slot-based format [padded_batch, S_per_user]
            S = tokens.shape[-1]  # Per-user sequence length
            tokens = tokens.reshape(1, 1, 1, -1)  # Flatten to [1, 1, 1, padded_batch * S]
        else:
            tokens = tokens.reshape(1, 1, 1, -1)
            S = tokens.shape[-1]
        tokens = ttnn.from_torch(
            tokens,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # self.embd expects that tokens are on device ; if trace is enabled, the tensors will be later on device, so we will do these 2 steps when we copy the tokens to the device
        if not trace_enabled:
            tokens_embd = self.embd(tokens)
            tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Slice the rot mats to the prefill seqlen
        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        seq_len = last_token_idx + 1 if last_token_idx is not None else S
        assert mat_len >= seq_len, f"Sequence length {seq_len} exceeds max seq len {mat_len}"

        required_end = start_pos + S
        pad_len = max(0, required_end - mat_len)

        # We set the end_pos to max_seq_len so that we don't create a new tensor for the whole cos_matrix and sin_matrix
        # In case of trace, we will use the whole matrix for all seq_lens supported by trace
        prefill_start_pos = 0 if trace_enabled else start_pos
        slice_end = self.args.max_seq_len if trace_enabled else min(mat_len, required_end)

        cos_slice = self.rope_setup.cos_matrix_prefill[:, :, prefill_start_pos:slice_end, :]
        sin_slice = self.rope_setup.sin_matrix_prefill[:, :, prefill_start_pos:slice_end, :]

        if pad_len > 0:
            # Padding: [(before, after), ...] for each dim; pad at end of 3rd dim (dim=2) by pad_len
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

        tt_rot_mats_prefill_global = [cos_slice, sin_slice]

        if hasattr(self, "rope_local_setup"):
            local_mat_len = self.rope_local_setup.cos_matrix_prefill.shape[2]
            local_required_end = start_pos + S
            local_pad_len = max(0, local_required_end - local_mat_len)
            local_slice_end = self.args.max_seq_len if trace_enabled else min(local_mat_len, local_required_end)

            local_cos_slice = self.rope_local_setup.cos_matrix_prefill[:, :, prefill_start_pos:local_slice_end, :]
            local_sin_slice = self.rope_local_setup.sin_matrix_prefill[:, :, prefill_start_pos:local_slice_end, :]

            if local_pad_len > 0:
                # Pad at end of 3rd dim (dim=2) by local_pad_len
                local_padding = [(0, 0)] * 4
                local_padding[2] = (0, local_pad_len)
                local_cos_slice = ttnn.pad(local_cos_slice, padding=local_padding, value=0.0)
                local_sin_slice = ttnn.pad(local_sin_slice, padding=local_padding, value=0.0)

            tt_rot_mats_prefill_local = [local_cos_slice, local_sin_slice]
        else:
            tt_rot_mats_prefill_local = None

        if page_table is not None:
            # For batched prefill, replicate page_table to all devices (same as single-user path)
            # The KV cache fill will loop over users and use batch_idx=user_id for each
            tt_page_table = ttnn.from_torch(
                page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        if chunk_start_idx is not None and int(chunk_start_idx) > 0:
            chunk_start_idx_tensor = torch.tensor([chunk_start_idx], dtype=torch.int32)
            tt_chunk_start_idx = ttnn.from_torch(
                chunk_start_idx_tensor,
                device=device,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_start_idx = None

        return (
            tokens if trace_enabled else tokens_embd,
            tt_rot_mats_prefill_global,
            tt_rot_mats_prefill_local,
            tt_page_table,
            tt_chunk_page_table,
            tt_chunk_start_idx,
        )

    def prepare_inputs_decode(self, *inputs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Its implementation can take advantage of a few other functions which the
        model must implement.
        """
        host_inputs = self.prepare_decode_inputs_host(*inputs)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)  # Helper function
        return device_inputs

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """
        Inputs are torch tensors or python types. Outputs are ttnn tensors on host.
        NOTE: Tokens and current_pos are padded to batch
        """
        B = tokens.shape[0]
        assert current_pos.shape[0] == B, "Batch size mismatch"
        assert (
            B == self.args.max_batch_size
        ), f"Batch size {B} must be equal to max_batch_size {self.args.max_batch_size}"

        # Necessary padding to be full tile sized when on device
        tokens = torch.nn.functional.pad(tokens.view(-1), (0, 32 - len(tokens)), "constant", 0)
        tokens = ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens = ttnn.unsqueeze_to_4D(tokens)

        rot_current_pos = torch.maximum(
            current_pos, torch.tensor(0, dtype=torch.int64)
        )  # Ensure position indices are non-negative
        rope_idxs = self.rope_setup.get_rot_idxs(rot_current_pos, on_host=True)

        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, 0) if (self.args.is_galaxy and B > 1) else (None, None),
                mesh_shape=self.args.cluster_shape,
            ),
        )

        if page_table is not None:
            page_table = ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(None, -2) if (self.args.is_galaxy and B > 1) else (None, None),
                    mesh_shape=self.args.cluster_shape,
                ),
            )
        return tokens, current_pos_tt, rope_idxs, page_table

    def _transform_decode_inputs_device(
        self,
        tokens,
    ):
        """
        Inputs are ttnn tensors on device. This function applies any on-device
        transformations which should happen before forward decode.
        For example: tilize, reshape, shard.
        Return transformed device tensors

        Embed tokens
        """
        decode_residual_mem_cfg = self.args.get_residual_mem_config(Mode.DECODE, self.prefetcher)
        tt_tokens = self.embd(
            tokens,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if self.prefetcher is None else decode_residual_mem_cfg,
        )
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        tt_tokens = ttnn.to_memory_config(tt_tokens, decode_residual_mem_cfg)
        return tt_tokens

    def concat_host_output(self, tt_out, is_log_probs=False):
        """
        Concatenate the output of the devices into a single host tensor.
        """
        torch_out_tensors = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(tt_out)]
        if self.args.is_galaxy:
            row_dim, col_dim = (3, 1)
        else:
            row_dim, col_dim = (1, -1)

        rows, cols = self.args.cluster_shape
        mesh_shape = [torch_out_tensors[i : i + cols] for i in range(0, len(torch_out_tensors), cols)]
        if is_log_probs:
            row_concatenated = []
            for row in mesh_shape:
                row_reshaped = [tensor.reshape(1, 1, -1, 1) for tensor in row]
                row_concatenated.append(torch.cat(row_reshaped, dim=col_dim))
        else:
            row_concatenated = [torch.cat(row, dim=col_dim) for row in mesh_shape]

        return torch.cat(row_concatenated, dim=row_dim)

    def process_output_prefill(self, tt_out, last_token_idx):
        """
        Input is ttnn host tensor of logits. Output is torch logits tensor.
        NOTE: In this model, prefill always uses get_last_token
        """
        assert tt_out.storage_type() == ttnn.StorageType.HOST, "Expected host tensor"
        return self.concat_host_output(tt_out)[0, 0, last_token_idx, : self.vocab_size]

    def process_output_prefill_hidden_states(self, tt_out, last_token_idx):
        """
        Input is ttnn host tensor of hidden states (after norm, before LM head).
        Output is torch hidden states tensor of shape [hidden_size].
        Used for embedding models.
        """
        assert tt_out.storage_type() == ttnn.StorageType.HOST, "Expected host tensor"
        # Extract the last token's hidden state
        # Shape: [batch=1, head=1, seq, hidden_dim] -> [hidden_dim]
        # For hidden states, if they're replicated across devices (not sharded),
        # we should take just the first device's output to avoid incorrect concatenation.
        # If sharded, concat_host_output will properly concatenate them.
        concatenated = self.concat_host_output(tt_out)
        # Check if concatenation resulted in oversized tensor (replicated case)
        # If so, take only the first device's portion (first self.args.dim elements)
        if concatenated.shape[-1] > self.args.dim:
            # Hidden states are replicated, take first device's output
            return concatenated[0, 0, last_token_idx, : self.args.dim]
        else:
            # Hidden states are sharded, concatenation is correct
            return concatenated[0, 0, last_token_idx, :]

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        """
        Input is ttnn host tensor of logits if is_tokens=False, otherwise tokens. Output is the corresponding torch tensor.
        """
        if is_tokens or is_log_probs:
            # Pad to 32 to match the expected batch size for decode operations (tiles are 32x32)
            padded_batch_size = 32
            if not is_log_probs:
                tt_out = ttnn.reshape(tt_out, ttnn.Shape([1, 1, padded_batch_size, 1]))
            return self.concat_host_output(tt_out, is_log_probs)[0, 0, :B, 0]
        if self.args.num_devices > 1:
            tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        else:
            tt_out = ttnn.to_torch(tt_out).float()
        tt_out = tt_out[:, :, :B, : self.vocab_size].view(B, S, -1)
        return tt_out

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        page_tables_per_layer=None,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        if page_tables_per_layer is None:
            # vLLM hybrid bridges (HybridAttentionForCausalLM subclasses) stash
            # the per-layer list on the model handle for the duration of a
            # forward call rather than threading the kwarg through Generator's
            # many ttnn_prefill_forward call sites. Pick it up here when set.
            page_tables_per_layer = getattr(self, "_active_page_tables_per_layer", None)
        page_tables_per_layer = self._page_tables_to_ttnn(page_tables_per_layer)
        return self.forward(
            x,
            current_pos=None,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            user_id=user_id,
            mode=Mode.PREFILL,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
            batch_size=batch_size,
            page_tables_per_layer=page_tables_per_layer,
        )

    def _page_table_mesh_mapper(self, B):
        """Mesh mapper for per-layer page tables, matching the layout that
        :meth:`prepare_decode_inputs_host` uses for the legacy single
        ``page_table`` kwarg: shard the batch dim across mesh axis 1 on
        Galaxy when ``B>1``, replicate otherwise. The hybrid bridge
        chunks the global page table per-DP before calling into a
        submesh, so ``B`` here is the per-DP batch — same value the
        legacy path sees on entry to ``prepare_decode_inputs_host``.
        """
        return ttnn.ShardTensor2dMesh(
            self.mesh_device,
            dims=(None, -2) if (self.args.is_galaxy and B > 1) else (None, None),
            mesh_shape=self.args.cluster_shape,
        )

    def _page_tables_to_ttnn(self, page_tables_per_layer):
        """Resolve a per-layer list of ``torch.Tensor`` page tables to a
        list of *persistent* ttnn device tensors (allocate-only).

        Tracing bakes each input tensor's device address into the captured
        graph; replaying the trace reads from those exact addresses
        regardless of any new ttnn objects created on the Python side.
        Allocating fresh device tensors on every call would therefore
        make traced inference read stale memory at the original
        addresses, so we lazily allocate one persistent device tensor per
        layer on first use and *only* update contents from outside the
        traced ``ttnn_*_forward`` calls (writes are forbidden during trace
        capture). The hybrid bridge calls
        :meth:`update_persistent_per_layer_page_tables` *before* invoking
        ``Generator``'s decode/prefill which executes traces — that's
        where content updates happen.

        First call (warmup compile) populates the persistent buffers from
        the input torch tensors; subsequent calls return the existing
        buffers unchanged. ``None`` entries propagate; already-ttnn
        entries pass through.
        """
        if page_tables_per_layer is None:
            return None
        persistent = getattr(self, "_persistent_per_layer_page_tables", None)
        n = len(page_tables_per_layer)
        if persistent is None or len(persistent) != n:
            persistent = []
            for pt in page_tables_per_layer:
                if pt is None:
                    persistent.append(None)
                    continue
                if isinstance(pt, ttnn.Tensor):
                    persistent.append(pt)
                    continue
                persistent.append(
                    ttnn.from_torch(
                        pt,
                        device=self.mesh_device,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=self._page_table_mesh_mapper(pt.shape[0]),
                    )
                )
            self._persistent_per_layer_page_tables = persistent
        return persistent

    def update_persistent_per_layer_page_tables(self, page_tables_per_layer):
        """Update content of persistent per-layer page_table device
        tensors in place. Called by the hybrid bridge *before* invoking
        ``Generator``'s decode/prefill so traced replay observes the new
        block IDs at the captured addresses. Must be called outside trace
        capture (writes forbidden inside).

        No-op if persistent tensors haven't been allocated yet (first
        call goes through :meth:`_page_tables_to_ttnn`'s allocation).
        """
        if page_tables_per_layer is None:
            return
        persistent = getattr(self, "_persistent_per_layer_page_tables", None)
        if persistent is None or len(persistent) != len(page_tables_per_layer):
            return
        for i, pt in enumerate(page_tables_per_layer):
            if pt is None or persistent[i] is None or isinstance(pt, ttnn.Tensor):
                continue
            host_pt = ttnn.from_torch(
                pt,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self._page_table_mesh_mapper(pt.shape[0]),
            )
            ttnn.copy_host_to_device_tensor(host_pt, persistent[i])

    def _increment_decode_positions_device(self, current_pos, rot_mat_idxs):
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)

    def _slice_prefill_rot_mats(self, rot_mats, chunk_start_idx):
        """Slices full prefill RoPE mats on device to [chunk_start_idx, max_seq_len)."""
        if rot_mats is None or chunk_start_idx is None or not isinstance(chunk_start_idx, ttnn.Tensor):
            return rot_mats

        full_rot_cos, full_rot_sin = rot_mats[0], rot_mats[1]
        if full_rot_cos.shape[2] != self.args.max_seq_len:
            # Already sliced in input prep path; leave as-is.
            return rot_mats

        z = self._tt_slice_start_zeros_4
        tt_slice_starts = ttnn.concat([z[0:2], chunk_start_idx, z[3:4]], dim=0)

        rot_cos_slice = ttnn.slice(
            input_tensor=full_rot_cos,
            starts=tt_slice_starts,
            ends=self._tt_seq_len_buffer,
            slice_dim=2,
            num_devices=self.args.num_devices,
        )
        rot_sin_slice = ttnn.slice(
            input_tensor=full_rot_sin,
            starts=tt_slice_starts,
            ends=self._tt_seq_len_buffer,
            slice_dim=2,
            num_devices=self.args.num_devices,
        )
        return (rot_cos_slice, rot_sin_slice)

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        on_device_logits=False,
        page_tables_per_layer=None,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        rot_mats_global = self.rope_setup.get_rot_mats(rot_mat_idxs)
        rot_mats_local = self.rope_local_setup.get_rot_mats(rot_mat_idxs) if hasattr(self, "rope_local_setup") else None

        x_embed = self._transform_decode_inputs_device(x)

        if page_tables_per_layer is None:
            # See ttnn_prefill_forward: hybrid bridges stash the per-layer list
            # on the model when active, since Generator doesn't thread the kwarg.
            page_tables_per_layer = getattr(self, "_active_page_tables_per_layer", None)
        page_tables_per_layer = self._page_tables_to_ttnn(page_tables_per_layer)

        tt_logits = self.forward(
            x_embed,
            current_pos,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            mode=Mode.DECODE,
            page_table=page_table,
            kv_cache=kv_cache,
            page_tables_per_layer=page_tables_per_layer,
        )

        if on_device_logits:
            assert self.sampling is not None, (
                "ttnn_decode_forward got on_device_logits=True but no on-device sampling "
                "module exists (self.sampling is None)."
            )
            self._increment_decode_positions_device(current_pos, rot_mat_idxs)
            return tt_logits

        # Gather the output across all devices and untilize the tensor (for argmax)
        if self.args.num_devices > 1:
            cluster_axis = 0 if self.args.is_galaxy else None
            num_links = 2 if self.args.is_galaxy else 1
            tt_logits = ttnn.experimental.all_gather_async(
                tt_logits,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=num_links,
                memory_config=tt_logits.memory_config() if self.prefetcher is None else ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=cluster_axis,
                topology=self.args.ccl_topology(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
                subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
            )

        tt_logits = ttnn.untilize(
            tt_logits,
            use_multicore=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sub_core_grids=self.prefetcher.all_worker_cores_range_set if self.prefetcher is not None else None,
        )

        return tt_logits, None

    def switch_mode(self, mode: Mode):
        if self.prefetcher is not None:
            self.prefetcher.init(mode)
            self.prefetcher.prefetch()

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode: Mode = Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        page_tables_per_layer=None,
    ):
        if mode == Mode.DECODE:
            # Run prefetcher if it is enabled
            if self.prefetcher is not None:
                self.prefetcher.run()

        if mode == Mode.PREFILL:
            # For traced prefill, keep RoPE slicing in-graph and driven by the
            # on-device chunk_start_idx input.
            rot_mats_global = self._slice_prefill_rot_mats(rot_mats_global, chunk_start_idx)
            if rot_mats_local is not None:
                rot_mats_local = self._slice_prefill_rot_mats(rot_mats_local, chunk_start_idx)

        if page_tables_per_layer is not None and len(page_tables_per_layer) != len(self.layers):
            raise ValueError(
                f"page_tables_per_layer has {len(page_tables_per_layer)} entries "
                f"but model has {len(self.layers)} layers"
            )

        for i, layer in enumerate(self.layers):
            # No-op if callers already provide the right memory config
            activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )

            if mode == Mode.DECODE and not self.args.is_galaxy:
                x = ttnn.to_memory_config(
                    x,
                    self.args.get_residual_mem_config(mode, self.prefetcher),
                    activation_dtype,
                )
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)

            # vLLM hybrid kv-cache-groups: each attention layer gets its own
            # paged pool (sliding-window vs full-attention have different
            # block counts). When ``page_tables_per_layer`` is None we fall
            # back to broadcasting the single ``page_table`` to every layer
            # — byte-equivalent to the pre-hybrid path used by every legacy
            # caller (demos, unit tests, non-hybrid vLLM bridges).
            layer_page_table = page_tables_per_layer[i] if page_tables_per_layer is not None else page_table

            x = layer(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=layer_page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                batch_size=batch_size,
            )

        if mode == Mode.DECODE:
            if self.prefetcher is not None:
                self.prefetcher.stop()

        if mode == Mode.PREFILL and get_last_token == -1:
            return x

        # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        # Output norm
        x = self.norm(x, mode=mode, norm_config=self.args.get_norm_config("lm_head", mode, self.prefetcher))

        lm_head_input_mem_cfg = self.args.get_lm_head_input_mem_config(
            mode, None if mode == Mode.PREFILL else self.prefetcher
        )
        if mode == Mode.PREFILL and lm_head_input_mem_cfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)
        if mode == Mode.DECODE and self.prefetcher is not None:
            x = ttnn.to_memory_config(x, self.args.get_lm_head_input_mem_config(mode, self.prefetcher))

        x = self.lm_head(x)
        if mode == Mode.PREFILL:
            x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return x
