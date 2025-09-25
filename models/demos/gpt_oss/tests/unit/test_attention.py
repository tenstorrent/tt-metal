import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss.tt.rope import ApplyRotaryPosEmb
from models.utility_functions import comp_pcc

from ...reference.configuration_gpt_oss import GptOssConfig
from ...reference.hf_utils import get_state_dict
from ...reference.modeling_gpt_oss import GptOssAttention, GptOssRotaryEmbedding
from ...tt.attention import Attention
from ...tt.ccl import CCLManager
from ...tt.model_config import ModelArgs
from ...utils.general_utils import get_decode_mask

# ModelArgs will be instantiated inside test functions to avoid import-time loading


@pytest.mark.parametrize(
    "batch_size, seq_len,",
    [
        (1, 1),  # 20B config
        (1, 32),  # 20B config
        # (1, 256),  # 20B config
    ],
)
@pytest.mark.parametrize("use_real_weights", [False], ids=["random"])
@pytest.mark.parametrize("layer_idx", [0])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_attention(
    mesh_device,
    batch_size,
    seq_len,
    use_real_weights,
    layer_idx,
    reset_seeds,
):
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 8)))
    print(mesh_device.shape)

    # Get paths from ModelArgs to avoid code duplication
    model_args = ModelArgs(mesh_device=None, dummy_weights=True)  # dummy_weights=True to avoid loading actual weights
    gpt_dir = model_args.model_path
    local_weights_path = gpt_dir

    all_passing = True

    # Create configuration
    config = GptOssConfig()

    sliding_window = 0
    if layer_idx % 2 == 0:
        sliding_window = config.sliding_window

    cur_seq_len = seq_len
    position_ids = torch.arange(seq_len).unsqueeze(0)

    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    mask = torch.triu(torch.full((1, 1, cur_seq_len, cur_seq_len), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(torch.full((1, 1, cur_seq_len, cur_seq_len), -float("inf")), diagonal=-sliding_window)

    cache_position = None  # TODO: What is this used for?

    RopeEmbeddings = GptOssRotaryEmbedding(config)
    cos, sin = RopeEmbeddings(hidden_states, position_ids)
    position_embeddings = (cos, sin)

    reference_model = GptOssAttention(config, layer_idx=layer_idx)

    if use_real_weights:
        state_dict = get_state_dict(local_weights_path, f"model.layers.{layer_idx}.self_attn.", dtype=torch.float32)
        reference_model.load_state_dict(state_dict, strict=True)

    state_dict = reference_model.state_dict()

    # Reference model forward
    reference_out, _ = reference_model(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=mask,
        use_cache=True,
    )

    if seq_len == 1:  # decode
        mask = get_decode_mask(position_ids[0].item(), sliding_window)
        mask = mask.repeat(1, config.num_attention_heads // mesh_device.shape[1], 1, 1).transpose(1, 2)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_mask = ttnn.from_torch(mask, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_cos = ttnn.from_torch(cos.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_position_idx = ttnn.from_torch(position_ids, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    apply_rope = ApplyRotaryPosEmb(config)
    rope_stuff = (apply_rope, tt_cos, tt_sin)

    ccl_manager = CCLManager(mesh_device)
    tt_model = Attention(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict,
        layer_idx=0,
        ccl_manager=ccl_manager,
    )

    tt_out = tt_model(tt_hidden_states, tt_mask, rope_stuff, tt_position_idx)

    tt_out_tensors = ttnn.get_device_tensors(tt_out)
    for i in range(len(tt_out_tensors)):
        tt_out_torch = ttnn.to_torch(tt_out_tensors[i])

        # Compare outputs
        pcc = 0.99
        passed, pcc_message = comp_pcc(reference_out, tt_out_torch, pcc)
        logger.info(f"Test passed: {passed}, PCC: {pcc_message}")
        all_passing = all_passing and passed
