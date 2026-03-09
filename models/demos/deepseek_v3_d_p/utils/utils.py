from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.rope import RotarySetup


def get_rope_tensors(
    hf_config: PretrainedConfig,
    seq_len: int,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int = 0,
) -> dict[str, ttnn.Tensor]:
    rope_setup = RotarySetup(
        device=mesh_device,
        batch_size_per_row=1,
        hf_config=hf_config,
    )
    return rope_setup.get_rot_mats_table_shard_over_seq_len(seq_len, sp_axis=sp_axis)
