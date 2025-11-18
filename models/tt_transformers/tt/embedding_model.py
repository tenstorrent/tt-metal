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
from models.tt_transformers.tt.model_config import TensorGroup
from models.tt_transformers.tt.pooling import MeanPooling
from models.tt_transformers.tt.rope import RotarySetup


class EmbeddingTransformer(LightweightModule):
    """
    Embedding Transformer model for text embedding tasks.

    This model uses the same transformer backbone as text generation models
    but outputs fixed-dimension embeddings instead of token logits.
    """

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

        # For embedding models, always use DistributedNorm for multi-device compatibility
        # The norm operates on the gathered tensor after pooling
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

        # Replace LM Head with pooling layer for embedding generation
        self.pooling = MeanPooling(mesh_device=mesh_device, args=args, tt_ccl=self.tt_ccl)

    def prepare_prefill_inputs_trace(self, tokens, page_table=None, chunk_page_table=None):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on host.
        """
        host_inputs = self.prepare_inputs_prefill(
            tokens, page_table=page_table, chunk_page_table=chunk_page_table, trace_enabled=True
        )
        return host_inputs

    def transform_and_embed_prefill_inputs_device(self, tokens, tt_page_table, tt_chunk_page_table):
        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        return tt_tokens, tt_page_table, tt_chunk_page_table

    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None, trace_enabled=False):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device if trace is disabled or on host if trace is enabled.
        TODO: Debate whether this function is responsible for padding
        """

        # We set the device to None if trace is enabled so we keep the tensors on host instead of sending it to the device (None - keeps on host, device - sends to specified device)
        # We will send them to device later (copy_host_to_device)
        device = None if trace_enabled else self.mesh_device

        assert tokens.dim() == 2, "tokens must be a 2D tensor"
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
        assert (
            self.rope_setup.cos_matrix.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix.shape[2]}"

        # We set the end_pos to max_seq_len so that we don't create a new tensor for the whole cos_matrix and sin_matrix ; in case of trace, we will use the whole matrix for all seq_lens supported by trace
        start_pos = 0 if trace_enabled else start_pos
        end_pos = self.args.max_seq_len if trace_enabled else start_pos + S

        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix[:, :, start_pos:end_pos, :],
            self.rope_setup.sin_matrix[:, :, start_pos:end_pos, :],
        ]

        if hasattr(self, "rope_local_setup"):
            tt_rot_mats_prefill_local = [
                self.rope_local_setup.cos_matrix[:, :, start_pos:end_pos, :],
                self.rope_local_setup.sin_matrix[:, :, start_pos:end_pos, :],
            ]
        else:
            tt_rot_mats_prefill_local = None

        if page_table is not None:
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

        return (
            tokens if trace_enabled else tokens_embd,
            tt_rot_mats_prefill_global,
            tt_rot_mats_prefill_local,
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
        Input is ttnn device tensor of embeddings. Output is torch embeddings tensor.
        """
        return self.concat_host_output(tt_out.cpu())[0, 0, 0, :]  # Extract single embedding

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False):
        """
        Not used for embedding models - embeddings are computed during prefill
        """
        raise NotImplementedError("Decode mode not supported for embedding models")

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

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="prefill",  # Embedding models only support prefill mode
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        attention_mask=None,
    ):
        """
        Forward pass for embedding generation.

        Args:
            x: Input token embeddings
            current_pos: Current position (None for prefill)
            rot_mats_global: Global RoPE matrices
            rot_mats_local: Local RoPE matrices (optional)
            user_id: User ID for multi-user scenarios
            mode: Must be "prefill" for embedding models
            page_table: Paged attention page table
            chunk_page_table: Chunked page table
            chunk_start_idx: Chunk start index
            get_last_token: Not used for embeddings
            kv_cache: KV cache (not typically used for embeddings)
            attention_mask: Attention mask for pooling

        Returns:
            ttnn.Tensor: Embeddings of shape [batch, 1, 1, hidden_dim]
        """
        if mode != "prefill":
            raise ValueError("Embedding models only support prefill mode")

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

        # Output normalization - DistributedNorm handles gathering internally
        x = self.norm(x, mode=mode)

        # Apply mean pooling to get embeddings
        embeddings = self.pooling(x, attention_mask=attention_mask)

        # Convert to row-major layout for output
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return embeddings
