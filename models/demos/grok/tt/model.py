from tqdm import tqdm

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.grok.tt.decoder import Decoder
from models.demos.grok.tt.distributed_norm import DistributedNorm
from models.demos.grok.tt.lm_head import LMHead
from models.tt_transformers.tt.common import copy_host_to_device
from models.tt_transformers.tt.embedding import ScaledEmbedding
from models.tt_transformers.tt.rope import RotarySetup


class Transformer(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path,
        args,
        dtype,
        paged_attention_config=None,
    ):
        self.mesh_device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.model_config = self.args.get_model_config()

        self.embd = ScaledEmbedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=weight_cache_path,
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
            embed_scale=args.embedding_multiplier_scale,
        )

        self.rope_setup = RotarySetup(
            self.mesh_device,
            self.args.max_batch_size,
            self.args.head_dim,
            self.args.max_seq_len,
            self.args.rope_theta,
            None,
        )

        self.transformation_mats = self.rope_setup.get_both_trans_mats()

        self.layers = [
            Decoder(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=layer_idx,
                transformation_mats=self.transformation_mats,
                paged_attention_config=paged_attention_config,
                deallocate_torch=True,
            )
            for layer_idx in tqdm(range(self.args.num_hidden_layers))
        ]
        self.norm = DistributedNorm(
            RMSNorm(
                device=self.mesh_device,
                dim=self.args.dim,
                eps=1e-5,
                state_dict={f"model.norm.weight": state_dict[f"model.norm.weight"]},
                # weight_cache_path=weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=f"model.norm",
                is_distributed=True,
                ccl_topology=ttnn.Topology.Ring,
                tt_ccl=tt_ccl,
            ),
            self.args,
            tt_ccl=tt_ccl,
        )
        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix="model.lm_head",
            weight_cache_path=weight_cache_path,
            max_columns_per_device=self.args.max_columns_per_device_lm_head,
        )

    def forward_decode(self, x, current_pos, rot_mats, page_table=None):
        for layer_idx, layer in enumerate(self.layers):
            x = ttnn.to_memory_config(x, self.model_config["DECODE_RESIDUAL_MEMCFG"])
            x = layer(
                hidden_states=x,
                current_pos=current_pos,
                rot_mats=rot_mats,
                page_table=page_table,
            )

        x = self.norm(x, mode="decode")
        x = self.lm_head(x)
        return x

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

    def process_output_decode(self, tt_out, B, S=1):
        """
        Input is ttnn host tensor of logits if is_tokens=False, otherwise tokens. Output is the corresponding torch tensor.
        """
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(3, 1), mesh_shape=self.args.cluster_shape),
        )
        tt_out = tt_out[:, :1, :B, : self.args.vocab_size].view(B, S, -1)
        return tt_out

    def ttnn_decode_forward(self, x, current_pos, rot_mats=None, rot_mat_idxs=None, page_table=None):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        rot_mats = self.rope_setup.get_rot_mats(rot_mat_idxs) if rot_mat_idxs is not None else rot_mats
        x_embed = self._transform_decode_inputs_device(x)
        tt_logits = self.forward_decode(
            x_embed,
            current_pos,
            rot_mats=rot_mats,
            page_table=page_table,
        )

        # Gather the output across all devices and untilize the tensor (for argmax)
        tt_logits = ttnn.experimental.all_gather_async(
            tt_logits,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(0),
            num_links=2,
            memory_config=tt_logits.memory_config(),
            cluster_axis=0,
            topology=ttnn.Topology.Ring,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(0),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        tt_logits = ttnn.untilize(tt_logits, use_multicore=True)
        # Send output logits to DRAM so L1 is not reserved for ttnn tracing and can be used by subsequent operations
        tt_logits = ttnn.to_memory_config(tt_logits, ttnn.DRAM_MEMORY_CONFIG)

        return tt_logits
