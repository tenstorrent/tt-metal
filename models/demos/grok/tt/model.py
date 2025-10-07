from models.common.lightweightmodule import LightweightModule
from models.demos.grok.tt.decoder import Decoder
from models.tt_transformers.tt.rope import RotarySetup


class Transformer(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path,
        args,
        layer_num,
        dtype,
        paged_attention_config=None,
    ):
        self.mesh_device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl

        self.rope_setup = RotarySetup(
            mesh_device,
            batch_size,
            model_args.head_dim,
            model_args.max_seq_len,
            model_args.rope_theta,
            None,  # No rope scaling for Grok
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
            )
            for layer_idx in self.args.num_hidden_layers
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
