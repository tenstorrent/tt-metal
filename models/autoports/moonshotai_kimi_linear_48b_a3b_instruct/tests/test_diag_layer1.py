import torch

from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import replicated
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.layer import KimiDecoderLayer


def test_layer1_weights(mesh_device, ccl, hf_config, checkpoint, cache_path):
    layer = KimiDecoderLayer(
        mesh_device, hf_config, checkpoint.layer_state_dict(1), layer_idx=1, ccl=ccl, cache_path=cache_path
    )
    for name, t in (
        ("shared.gate", layer.mlp.shared.gate),
        ("shared.up", layer.mlp.shared.up),
        ("shared.down", layer.mlp.shared.down),
        ("experts.gate", layer.mlp.experts.gate),
        ("router.weight", layer.mlp.router.weight),
        ("router.bias", layer.mlp.router.bias),
    ):
        print(f"[w] {name}: storage {t.storage_type()} shape {t.shape} dtype {t.dtype} layout {t.layout}")
    x = replicated(mesh_device, (torch.randn(1, 1, 32, hf_config.hidden_size) * 0.2).bfloat16())
    h2 = layer.post_norm(x)
    print("[x] post_norm out storage", h2.storage_type(), h2.shape, h2.dtype, h2.layout, h2.memory_config())
    out = layer.mlp.forward(h2, "prefill")
    print("[ok] moe forward", out.shape)
