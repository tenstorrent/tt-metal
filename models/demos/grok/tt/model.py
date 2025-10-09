import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.grok.tt.decoder import Decoder
from models.demos.grok.tt.distributed_norm import DistributedNorm
from models.demos.grok.tt.lm_head import LMHead
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
            for layer_idx in range(self.args.num_hidden_layers)
        ]
        self.norm = DistributedNorm(
            RMSNorm(
                device=self.mesh_device,
                dim=self.args.dim,
                eps=1e-5,
                state_dict={f"model.norm.weight": state_dict[f"model.norm.weight"]},
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
        for layer_idx, layer in enuermate(self.layers):
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
