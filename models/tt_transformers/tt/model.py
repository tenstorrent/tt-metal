# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from tqdm import tqdm

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import copy_host_to_device
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding, ScaledEmbedding
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import TensorGroup
from models.tt_transformers.tt.rope import RotarySetup


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
        self.tt_rot_mats_prefill_global = None
        self.tt_rot_mats_prefill_local = None

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

        ActualRopeSetupClass = rope_setup_class if rope_setup_class is not None else RotarySetup
        self.rope_setup = ActualRopeSetupClass(
            device=mesh_device,
            batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )

        if args.rope_theta_local:
            self.rope_local_setup = RotarySetup(
                mesh_device,
                args.max_batch_size,
                args.head_dim,
                args.max_seq_len,
                args.rope_theta_local,
            )

        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()

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
                sharded_program_config=self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"],
                sharded_output_config=self.model_config["LM_HEAD_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            self.tt_ccl,
            args.is_galaxy,
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
        )

        if hasattr(self.args, "sliding_window") and self.args.sliding_window is not None:
            # We are using sliding window attention in this model. We can create a custom attention mask to apply the sliding attention
            # First we create the mask for all decode positions on host [bsz, n_heads_per_device, seq_len, seq_len]
            self.decode_sliding_mask_mat = get_decode_mask(
                self.args,
                self.mesh_device,
                paged_attention_config=paged_attention_config,
            )
            # Then we copy a slice for a single decode position for each user on to device [bsz, n_heads_per_device, 1, seq_len]
            # We can update this tensor on host each iteration and copy to device to save storing the large square tensor on device
            self.device_decode_sliding_mask = ttnn.as_tensor(
                torch.concat(
                    [self.decode_sliding_mask_mat[i, :, 0:1, :].unsqueeze(0) for i in range(self.args.max_batch_size)],
                    axis=0,
                ).transpose(1, 2),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            self.decode_sliding_mask_mat = None
            self.device_decode_sliding_mask = None

    def prepare_prefill_inputs_host(self, tokens, page_table=None, chunk_page_table=None, user_id=0, start_pos=0):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on host.
        """
        host_inputs = self.prepare_inputs_prefill(
            tokens,
            start_pos=start_pos,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            trace_enabled=True,
            user_id=user_id,
        )
        return host_inputs

    def transform_prefill_inputs_device(self, tokens, tt_page_table, tt_chunk_page_table, user_id):
        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        return tt_tokens, tt_page_table, tt_chunk_page_table, user_id

    def prepare_inputs_prefill(
        self, tokens, start_pos=0, page_table=None, chunk_page_table=None, trace_enabled=False, user_id=0
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        TODO: Debate whether this function is responsible for padding
        """

        assert tokens.dim() == 2, "tokens must be a 2D tensor"
        tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-1]
        tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device if not trace_enabled else None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # we do it this way because we want to run this when all our inputs are on the device
        if not trace_enabled:
            tokens_embd = self.embd(tokens)
            tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Slice the rot mats to the prefill seqlen
        assert (
            self.rope_setup.cos_matrix.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix.shape[2]}"

        if trace_enabled:
            if self.tt_rot_mats_prefill_global is None:
                self.tt_rot_mats_prefill_global = [
                    self.rope_setup.cos_matrix[:, :, 0 : self.args.max_seq_len, :],
                    # ovo nije fiksno, ali mora da bude fiksne velicine da spadne u sve opsege u kojima ce trace da se radi
                    self.rope_setup.sin_matrix[:, :, 0 : self.args.max_seq_len, :],
                ]
        else:
            self.tt_rot_mats_prefill_global = [
                self.rope_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
                self.rope_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
            ]

        if hasattr(self, "rope_local_setup"):
            if trace_enabled:
                if self.tt_rot_mats_prefill_local is None:
                    self.tt_rot_mats_prefill_local = [
                        self.rope_local_setup.cos_matrix[:, :, 0 : self.args.max_seq_len, :],
                        self.rope_local_setup.sin_matrix[:, :, 0 : self.args.max_seq_len, :],
                    ]
            else:
                self.tt_rot_mats_prefill_local = [
                    self.rope_local_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
                    self.rope_local_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
                ]
        else:
            self.tt_rot_mats_prefill_local = None

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device if not trace_enabled else None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device if not trace_enabled else None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        if trace_enabled:
            user_id = ttnn.from_torch(
                torch.tensor([user_id], dtype=torch.int32),
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        if trace_enabled:
            return tokens, tt_page_table, tt_chunk_page_table, user_id
        else:
            return (
                tokens_embd,
                self.tt_rot_mats_prefill_global,
                self.tt_rot_mats_prefill_local,
                tt_page_table,
                tt_chunk_page_table,
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

    def _transform_decode_inputs_device(self, tokens):
        """
        Inputs are ttnn tensors on device. This function applies any on-device
        transformations which should happen before forward decode.
        For example: tilize, reshape, shard.
        Return transformed device tensors

        Embed tokens
        """
        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        tt_tokens = ttnn.to_memory_config(
            tt_tokens,
            self.args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )
        return tt_tokens

    def concat_host_output(self, tt_out):
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
        row_concatenated = [torch.cat(row, dim=col_dim) for row in mesh_shape]
        return torch.cat(row_concatenated, dim=row_dim)

    def process_output_prefill(self, tt_out, last_token_idx):
        """
        Input is ttnn device tensor of logits. Output is torch logits tensor.
        NOTE: In this model, prefill always uses get_last_token
        """
        return self.concat_host_output(tt_out.cpu())[0, 0, last_token_idx, : self.vocab_size]

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False):
        """
        Input is ttnn host tensor of logits if is_tokens=False, otherwise tokens. Output is the corresponding torch tensor.
        """
        if is_tokens:
            return self.concat_host_output(tt_out)[0, 0, :B, 0]

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
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        return self.forward(
            x,
            current_pos=None,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            user_id=user_id,
            mode="prefill",
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
        )

    def _increment_decode_positions_device(self, current_pos, rot_mat_idxs):
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        argmax_on_device=False,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        rot_mats_global = self.rope_setup.get_rot_mats(rot_mat_idxs)
        rot_mats_local = self.rope_local_setup.get_rot_mats(rot_mat_idxs) if hasattr(self, "rope_local_setup") else None
        x_embed = self._transform_decode_inputs_device(x)
        tt_logits = self.forward(
            x_embed,
            current_pos,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            mode="decode",
            page_table=page_table,
            kv_cache=kv_cache,
        )

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
                memory_config=tt_logits.memory_config(),
                cluster_axis=cluster_axis,
                topology=self.args.ccl_topology(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        tt_logits = ttnn.untilize(tt_logits, use_multicore=True)

        if argmax_on_device:
            tt_logits = ttnn.argmax(tt_logits, dim=3, keepdim=True, use_multicore=True)

            # Update device tensors for the next iteration
            self._increment_decode_positions_device(current_pos, rot_mat_idxs)

            # Update input tokens with sampled tokens for the next iteration
            ttnn.copy(tt_logits.reshape(x.shape), x)
        elif not self.args.is_galaxy:
            # Send output logits to DRAM so L1 is not reserved for ttnn tracing and can be used by subsequent operations
            tt_logits = ttnn.to_memory_config(tt_logits, ttnn.DRAM_MEMORY_CONFIG)

        return tt_logits

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
    ):
        for i, layer in enumerate(self.layers):
            # No-op if callers already provide the right memory config
            activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )
            if mode == "decode" and not self.args.is_galaxy:
                x = ttnn.to_memory_config(x, self.model_config["DECODE_RESIDUAL_MEMCFG"], activation_dtype)
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)

            x = layer(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
            )

        if mode == "prefill" and get_last_token == -1:
            return x

        # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        # Output norm
        x = self.norm(x, mode=mode)

        if mode == "prefill" and self.model_config["LM_HEAD_INPUT_MEMCFG"].is_sharded():
            x = ttnn.interleaved_to_sharded(x, self.model_config["LM_HEAD_INPUT_MEMCFG"])

        x = self.lm_head(x)

        if mode == "prefill":
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x
