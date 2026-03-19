import torch
import ttnn
from safetensors.torch import load_file

from models.experimental.qwen3omni.tt.patch_embed import TTNNQwen3OmniMoeVisionPatchEmbed


MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CHECKPOINT_PATH = "/home/ubuntu/tt-metal/models/experimental/qwen3omni/checkpoints/model-00001-of-00015.safetensors"


# ------------------------------------------------------------
# Patchify
# ------------------------------------------------------------
def patchify_video(x, patch_size):
    B, C, F, H, W = x.shape
    pF, pH, pW = patch_size

    patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
    N = patch_F * patch_H * patch_W

    x = x.reshape(B, C, patch_F, pF, patch_H, pH, patch_W, pW)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
    x = x.reshape(1, B, N, pF * pH * pW * C)

    return x


# ------------------------------------------------------------
# Torch Conv3D Reference
# ------------------------------------------------------------
class TorchPatchEmbed(torch.nn.Module):
    def __init__(self, in_channels, embed_dim, kt, kh, kw):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(kt, kh, kw),
            stride=(kt, kh, kw),
            bias=True,
        )

    def forward(self, x):
        x = self.conv(x)
        return x.flatten(2).transpose(1, 2)


# ------------------------------------------------------------
# Load weights from safetensors
# ------------------------------------------------------------
def load_patch_embed_weights():
    state = load_file(CHECKPOINT_PATH)

    # Try multiple possible key names
    weight_key_candidates = [
        "thinker.visual.patch_embed.proj.weight",
        "visual.patch_embed.proj.weight",
        "patch_embed.proj.weight",
    ]

    bias_key_candidates = [
        "thinker.visual.patch_embed.proj.bias",
        "visual.patch_embed.proj.bias",
        "patch_embed.proj.bias",
    ]

    weight_key = next((k for k in weight_key_candidates if k in state), None)
    bias_key = next((k for k in bias_key_candidates if k in state), None)

    assert weight_key is not None, "Patch embed weight not found in checkpoint"
    assert bias_key is not None, "Patch embed bias not found in checkpoint"

    print(f"Using weight key: {weight_key}")
    print(f"Using bias key: {bias_key}")

    return {
        "weight": state[weight_key],
        "bias": state[bias_key],
    }


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------
def test_qwen_patch_embed_real_weights(mesh_device):
    torch.manual_seed(0)

    # Load real weights first so we can size input to match kernel (kt, kh, kw)
    weights = load_patch_embed_weights()
    conv_weight = weights["weight"]
    conv_bias = weights["bias"]
    embed_dim, in_channels, kt, kh, kw = conv_weight.shape

    # Input (F, H, W) must be >= kernel (kt, kh, kw)
    B, C = 1, in_channels
    F, H, W = max(2, kt), max(16, kh), max(16, kw)

    # Input (match checkpoint dtype so PyTorch conv input/bias types match)
    ref_dtype = conv_weight.dtype
    x = torch.randn(B, in_channels, F, H, W, dtype=ref_dtype)

    # Torch model
    torch_model = TorchPatchEmbed(in_channels, embed_dim, kt, kh, kw)
    torch_model.conv.weight.data = conv_weight.clone()
    torch_model.conv.bias.data = conv_bias.clone()

    # TT model
    tt_model = TTNNQwen3OmniMoeVisionPatchEmbed(
        patch_size=kh,
        temporal_patch_size=kt,
        in_channels=in_channels,
        embed_dim=embed_dim,
        mesh_device=mesh_device,
    )

    tt_model.load_state_dict(weights)

    # -------------------------
    # Torch output
    # -------------------------
    torch_out = torch_model(x)

    # -------------------------
    # TT pipeline
    # -------------------------
    patches = patchify_video(x, (kt, kh, kw))

    patches_tt = ttnn.from_torch(
        patches,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

    tt_out = tt_model(patches_tt)
    # Unwrap symbiote wrapper so we have the raw ttnn tensor
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(tt_out, TorchTTNNTensor) and getattr(tt_out, "ttnn_tensor", None) is not None:
        tt_out = tt_out.ttnn_tensor
    # Multi-device mesh requires a mesh composer to concatenate shards (set as default via distribute)
    with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=0)):
        tt_out = ttnn.to_torch(tt_out).squeeze(0)

    # ConcatMeshToTensor(dim=0) can double the first dim on 2-device mesh; take logical batch size to match ref
    if tt_out.shape[0] != torch_out.shape[0]:
        tt_out = tt_out[: torch_out.shape[0]]

    # -------------------------
    # PCC
    # -------------------------
    pcc = torch.corrcoef(torch.stack([torch_out.flatten().float(), tt_out.flatten().float()]))[0, 1].item()

    print(f"PCC: {pcc}")

    assert pcc > 0.999, f"Mismatch with real checkpoint weights (PCC={pcc})"
