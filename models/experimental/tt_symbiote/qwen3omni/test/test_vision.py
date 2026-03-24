import torch
import pytest
import ttnn
from safetensors.torch import safe_open
from transformers import AutoConfig

from models.common.utility_functions import comp_pcc
from models.experimental.tt_symbiote.qwen3omni.tt.vision_attn import TTNNQwen3VLMoeVisionAttention
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.utils.device_management import set_device
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionAttention


MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CHECKPOINT_PATH = "/home/ubuntu/tt-metal/models/experimental/qwen3omni/checkpoints/model-00001-of-00015.safetensors"


def load_vision_attention_weights(layer=0):
    prefix = f"thinker.visual.blocks.{layer}.attn"

    weights = {}

    with safe_open(CHECKPOINT_PATH, framework="pt", device="cpu") as f:
        weights["qkv_w"] = f.get_tensor(f"{prefix}.qkv.weight")
        weights["qkv_b"] = f.get_tensor(f"{prefix}.qkv.bias")
        weights["proj_w"] = f.get_tensor(f"{prefix}.proj.weight")
        weights["proj_b"] = f.get_tensor(f"{prefix}.proj.bias")

    return weights


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
@pytest.mark.parametrize("seq_len", [128])
def test_vision_attention(device, seq_len):
    full_config = AutoConfig.from_pretrained(MODEL_NAME)
    config = full_config.thinker_config.vision_config
    # Vision config may not have _attn_implementation; avoid None in is_flash_attention_requested()
    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "eager"

    # Torch model
    torch_attn = Qwen3OmniMoeVisionAttention(config)
    weights = load_vision_attention_weights(0)

    with torch.no_grad():
        torch_attn.qkv.weight.copy_(weights["qkv_w"])
        torch_attn.qkv.bias.copy_(weights["qkv_b"])
        torch_attn.proj.weight.copy_(weights["proj_w"])
        torch_attn.proj.bias.copy_(weights["proj_b"])

    torch_attn.eval()

    # TT model
    tt_attn = TTNNQwen3VLMoeVisionAttention.from_torch(torch_attn)
    set_device(tt_attn, device)
    tt_attn.preprocess_weights()
    tt_attn.move_weights_to_device()

    # Input
    hidden_states = torch.randn(seq_len, config.hidden_size)

    # Dummy rotary embeddings
    head_dim = config.hidden_size // config.num_heads
    cos = torch.randn(seq_len, head_dim)
    sin = torch.randn(seq_len, head_dim)

    position_embeddings = (cos, sin)

    # Torch output
    with torch.no_grad():
        torch_out = torch_attn(
            hidden_states,
            cu_seqlens=torch.tensor([0, seq_len]),
            position_embeddings=position_embeddings,
        )

    # TT input conversion
    tt_input = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    tt_pos_emb = (tt_cos, tt_sin)

    # TT output
    tt_out = tt_attn(
        tt_input,
        cu_seqlens=None,
        position_embeddings=tt_pos_emb,
    )

    # Symbiote may return TorchTTNNTensor; use .to_torch property, not ttnn.to_torch()
    tt_out_torch = tt_out.to_torch if isinstance(tt_out, TorchTTNNTensor) else ttnn.to_torch(tt_out)

    # Compare
    passing, pcc = comp_pcc(torch_out, tt_out_torch)

    print(f"PCC: {pcc}")

    assert passing
