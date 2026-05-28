import torch
import ttnn
from models.demos.lfm2_vl.tt.model import TtLfm2VlModel
from models.demos.lfm2_vl.tt.model_config import create_model_config

def run_lfm2_vl_demo():
    device = ttnn.open_device(device_id=0)
    batch_size = 1
    seq_len = 128
    config = create_model_config(batch_size, seq_len)
    class MockParams:
        def __init__(self, config):
            self.embed_tokens = type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["vocab_size"], config["hidden_size"]), device=device)})
            self.norm = type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"]), device=device)})
            self.projector = type('obj', (object,), {
                'gate_proj': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"], config["projector_hidden_size"]), device=device)}),
                'down_proj': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["projector_hidden_size"], config["hidden_size"]), device=device)})
            })
            self.vision = type('obj', (object,), {
                'patch_embed': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(3 * 16 * 16, config["vision_config"]["hidden_size"]), device=device)})
            })
            self.layers = []
            for i in range(config["num_hidden_layers"]):
                layer = type('obj', (object,), {
                    'input_projection': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"], 3 * config["hidden_size"]), device=device)}),
                    'conv': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"], 1, 3), device=device)}),
                    'output_projection': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"], config["hidden_size"]), device=device)}),
                    'input_layernorm': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"]), device=device)}),
                    'post_attention_layernorm': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"]), device=device)}),
                    'self_attn': type('obj', (object,), {
                        'q_proj': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"], config["hidden_size"]), device=device)}),
                        'k_proj': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"], config["hidden_size"]), device=device)}),
                        'v_proj': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"], config["hidden_size"]), device=device)}),
                        'o_proj': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"], config["hidden_size"]), device=device)})
                    }),
                    'mlp': type('obj', (object,), {
                        'gate_proj': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"], config["intermediate_size"]), device=device)}),
                        'up_proj': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["hidden_size"], config["intermediate_size"]), device=device)}),
                        'down_proj': type('obj', (object,), {'weight': ttnn.from_torch(torch.randn(config["intermediate_size"], config["hidden_size"]), device=device)})
                    })
                })
                self.layers.append(layer)
    parameters = MockParams(config)
    model = TtLfm2VlModel(device, config, parameters)
    pixel_values = torch.randn(batch_size, config["vision_config"]["num_patches"], 3 * 16 * 16)
    tt_pixel_values = ttnn.from_torch(pixel_values, device=device)
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    # Place image tokens (placeholder ID 32000) to test interleaving
    num_image_tokens = config["vision_config"]["num_patches"]
    input_ids[0, 10:10+num_image_tokens] = 32000
    tt_input_ids = ttnn.from_torch(input_ids, device=device)
    print("Running LFM2.5-VL Inference on Tenstorrent Device...")
    output = model(tt_pixel_values, tt_input_ids)
    output_torch = ttnn.to_torch(output)
    print(f"Inference Complete. Output shape: {output_torch.shape}")
    ttnn.close_device(device)

if __name__ == "__main__":
    run_lfm2_vl_demo()