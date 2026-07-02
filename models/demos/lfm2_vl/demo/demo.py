import torch
import ttnn
import math
import numpy as np
from models.demos.lfm2_vl.tt.model import TtLfm2VlModel
from models.demos.lfm2_vl.tt.model_config import create_model_config


def preprocess_liquid_image(image_path: str, target_num_patches: int = 256, patch_size: int = 16):
    """Preprocess an image for LFM2.5-VL following LiquidAI conventions.
    
    Args:
        image_path: Path to image file (JPEG, PNG, etc.)
        target_num_patches: Number of patches to extract (default: 256, for 16x16 grid)
        patch_size: Size of each patch in pixels (default: 16)
        
    Returns:
        pixel_values: torch.Tensor of shape [1, num_patches, 3 * patch_size * patch_size]
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available. Using random mock data instead.")
        return torch.randn(1, target_num_patches, 3 * patch_size * patch_size)
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    
    # Compute canvas dimensions: find the smallest canvas that fits
    # LiquidAI uses variable-resolution: canvas is scaled so patches cover the image
    # For simplicity, resize to a multiple of patch_size
    aspect = img.width / img.height
    
    # Target roughly sqrt(num_patches) patches per dimension
    target_patches_per_dim = int(math.sqrt(target_num_patches))
    
    if aspect > 1.0:
        # Landscape
        new_width = target_patches_per_dim * patch_size
        new_height = int(new_width / aspect)
        # Round height to multiple of patch_size
        new_height = max(patch_size, round(new_height / patch_size) * patch_size)
    else:
        # Portrait
        new_height = target_patches_per_dim * patch_size
        new_width = int(new_height * aspect)
        new_width = max(patch_size, round(new_width / patch_size) * patch_size)
    
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert to tensor [C, H, W]
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
    
    # Normalize to [0, 1]
    img_tensor = img_tensor / 255.0
    
    # SigLIP2 normalization (mean, std per channel)
    # Using standard SigLIP normalization
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # Patchify: unfold into patches of size patch_size x patch_size
    patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Shape: [C, H//patch_size, W//patch_size, patch_size, patch_size]
    patches = patches.contiguous().view(3, -1, patch_size * patch_size)
    # Shape: [C, num_patches, patch_size^2]
    patches = patches.permute(1, 0, 2)
    # Shape: [num_patches, C * patch_size^2]
    patches = patches.contiguous().view(-1, 3 * patch_size * patch_size)
    
    # Pad or truncate to target_num_patches
    actual_patches = patches.shape[0]
    if actual_patches > target_num_patches:
        patches = patches[:target_num_patches]
    elif actual_patches < target_num_patches:
        padding = torch.zeros(target_num_patches - actual_patches, 3 * patch_size * patch_size)
        patches = torch.cat([patches, padding], dim=0)
    
    # Add batch dimension
    pixel_values = patches.unsqueeze(0)  # [1, num_patches, 3 * patch_size * patch_size]
    
    print(f"Preprocessed image: {image_path}")
    print(f"  Original size: {img.size}")
    print(f"  Patches: {patches.shape[0]} x {3 * patch_size * patch_size}")
    
    return pixel_values


def create_rotary_cache(device, config, max_seq_len: int = 4096):
    """Create pre-computed RoPE cos/sin tables.
    
    Args:
        device: ttnn device
        config: model config dict
        max_seq_len: maximum sequence length for the cache
        
    Returns:
        (cos, sin) tuple as ttnn tensors
    """
    import math
    
    head_dim = config["hidden_size"] // config["num_heads"]
    base = config.get("rope_theta", 1000000.0)
    
    # Compute inv_freq
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    # Create ttnn tensors
    cos = ttnn.from_torch(
        emb.cos()[None, None, :, :head_dim],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    sin = ttnn.from_torch(
        emb.sin()[None, None, :, :head_dim],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    
    return cos, sin


def run_lfm2_vl_demo(
    image_path: str = None,
    use_mock: bool = True,
    prompt: str = None,
):
    """Run LFM2.5-VL inference demo on Tenstorrent device.
    
    Args:
        image_path: Path to input image (if None and use_mock=True, uses random data)
        use_mock: If True, use mock random parameters instead of real weights
        prompt: Text prompt for image understanding (future: tokenizes prompt)
    """
    device = ttnn.open_device(device_id=0)
    batch_size = 1
    seq_len = 128
    config = create_model_config(batch_size, seq_len)
    
    if use_mock:
        # Create mock parameters with correct shapes
        class MockParams:
            def __init__(self, config):
                self.embed_tokens = type('obj', (object,), {
                    'weight': ttnn.from_torch(
                        torch.randn(config["vocab_size"], config["hidden_size"]),
                        device=device
                    )
                })
                self.norm = type('obj', (object,), {
                    'weight': ttnn.from_torch(
                        torch.randn(config["hidden_size"]),
                        device=device
                    )
                })
                
                # Vision encoder with proper layer structure
                vision_layers = []
                for i in range(config["vision_config"]["num_hidden_layers"]):
                    layer = type('obj', (object,), {
                        'norm1': type('obj', (object,), {
                            'weight': ttnn.from_torch(
                                torch.randn(config["vision_config"]["hidden_size"]),
                                device=device
                            )
                        }),
                        'norm2': type('obj', (object,), {
                            'weight': ttnn.from_torch(
                                torch.randn(config["vision_config"]["hidden_size"]),
                                device=device
                            )
                        }),
                        'attn': type('obj', (object,), {
                            'qkv': type('obj', (object,), {
                                'weight': ttnn.from_torch(
                                    torch.randn(
                                        3 * config["vision_config"]["hidden_size"],
                                        config["vision_config"]["hidden_size"]
                                    ),
                                    device=device
                                )
                            }),
                            'proj': type('obj', (object,), {
                                'weight': ttnn.from_torch(
                                    torch.randn(
                                        config["vision_config"]["hidden_size"],
                                        config["vision_config"]["hidden_size"]
                                    ),
                                    device=device
                                )
                            })
                        }),
                        'mlp': type('obj', (object,), {
                            'fc1': type('obj', (object,), {
                                'weight': ttnn.from_torch(
                                    torch.randn(
                                        4 * config["vision_config"]["hidden_size"],
                                        config["vision_config"]["hidden_size"]
                                    ),
                                    device=device
                                )
                            }),
                            'fc2': type('obj', (object,), {
                                'weight': ttnn.from_torch(
                                    torch.randn(
                                        config["vision_config"]["hidden_size"],
                                        4 * config["vision_config"]["hidden_size"]
                                    ),
                                    device=device
                                )
                            })
                        })
                    })
                    vision_layers.append(layer)
                
                self.vision = type('obj', (object,), {
                    'patch_embed': type('obj', (object,), {
                        'weight': ttnn.from_torch(
                            torch.randn(3 * 16 * 16, config["vision_config"]["hidden_size"]),
                            device=device
                        )
                    }),
                    'layers': vision_layers
                })
                
                # Projector
                self.projector = type('obj', (object,), {
                    'gate_proj': type('obj', (object,), {
                        'weight': ttnn.from_torch(
                            torch.randn(config["projector_hidden_size"], config["vision_config"]["hidden_size"]),
                            device=device
                        )
                    }),
                    'down_proj': type('obj', (object,), {
                        'weight': ttnn.from_torch(
                            torch.randn(config["hidden_size"], config["projector_hidden_size"]),
                            device=device
                        )
                    })
                })
                
                # Text layers
                self.layers = []
                for i in range(config["num_hidden_layers"]):
                    layer_type = config["layer_types"][i]
                    if layer_type == "conv":
                        layer = type('obj', (object,), {
                            'input_projection': type('obj', (object,), {
                                'weight': ttnn.from_torch(
                                    torch.randn(config["hidden_size"], 3 * config["hidden_size"]),
                                    device=device
                                )
                            }),
                            'conv': type('obj', (object,), {
                                'weight': ttnn.from_torch(
                                    torch.randn(config["hidden_size"], 1, 3),
                                    device=device
                                )
                            }),
                            'output_projection': type('obj', (object,), {
                                'weight': ttnn.from_torch(
                                    torch.randn(config["hidden_size"], config["hidden_size"]),
                                    device=device
                                )
                            })
                        })
                    else:
                        layer = type('obj', (object,), {
                            'input_layernorm': type('obj', (object,), {
                                'weight': ttnn.from_torch(
                                    torch.randn(config["hidden_size"]),
                                    device=device
                                )
                            }),
                            'post_attention_layernorm': type('obj', (object,), {
                                'weight': ttnn.from_torch(
                                    torch.randn(config["hidden_size"]),
                                    device=device
                                )
                            }),
                            'self_attn': type('obj', (object,), {
                                'q_proj': type('obj', (object,), {
                                    'weight': ttnn.from_torch(
                                        torch.randn(config["hidden_size"], config["hidden_size"]),
                                        device=device
                                    )
                                }),
                                'k_proj': type('obj', (object,), {
                                    'weight': ttnn.from_torch(
                                        torch.randn(config["hidden_size"], config["hidden_size"]),
                                        device=device
                                    )
                                }),
                                'v_proj': type('obj', (object,), {
                                    'weight': ttnn.from_torch(
                                        torch.randn(config["hidden_size"], config["hidden_size"]),
                                        device=device
                                    )
                                }),
                                'o_proj': type('obj', (object,), {
                                    'weight': ttnn.from_torch(
                                        torch.randn(config["hidden_size"], config["hidden_size"]),
                                        device=device
                                    )
                                })
                            }),
                            'mlp': type('obj', (object,), {
                                'gate_proj': type('obj', (object,), {
                                    'weight': ttnn.from_torch(
                                        torch.randn(config["intermediate_size"], config["hidden_size"]),
                                        device=device
                                    )
                                }),
                                'up_proj': type('obj', (object,), {
                                    'weight': ttnn.from_torch(
                                        torch.randn(config["intermediate_size"], config["hidden_size"]),
                                        device=device
                                    )
                                }),
                                'down_proj': type('obj', (object,), {
                                    'weight': ttnn.from_torch(
                                        torch.randn(config["hidden_size"], config["intermediate_size"]),
                                        device=device
                                    )
                                })
                            })
                        })
                    self.layers.append(layer)
        
        parameters = MockParams(config)
    else:
        # Load real weights
        from models.demos.lfm2_vl.tt.convert_weights import convert_lfm2_weights
        parameters = convert_lfm2_weights(image_path, device, config)
        # If image_path was actually the model path, revert
        if image_path and not image_path.endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_path = None
    
    model = TtLfm2VlModel(device, config, parameters)
    
    # Create RoPE cache
    cos, sin = create_rotary_cache(device, config, max_seq_len=4096)
    model.set_rope_cache((cos, sin))
    
    # Prepare image input
    if image_path and not use_mock:
        pixel_values = preprocess_liquid_image(
            image_path,
            target_num_patches=config["vision_config"]["num_patches"],
            patch_size=16,
        )
    else:
        # Mock image data
        pixel_values = torch.randn(
            1,
            config["vision_config"]["num_patches"],
            3 * 16 * 16
        )
    tt_pixel_values = ttnn.from_torch(pixel_values, device=device)
    
    # Prepare text input with image placeholders
    input_ids = torch.randint(0, config["vocab_size"], (1, seq_len))
    num_image_tokens = config["vision_config"]["num_patches"]
    # Place image tokens at position 10 in the sequence
    input_ids[0, 10:10+num_image_tokens] = 32000  # IMAGE_TOKEN_ID
    tt_input_ids = ttnn.from_torch(input_ids, device=device)
    
    print("Running LFM2.5-VL Inference on Tenstorrent Device...")
    print(f"  Image patches: {config['vision_config']['num_patches']}")
    print(f"  Vision layers: {config['vision_config']['num_hidden_layers']}")
    print(f"  Text layers: {config['num_hidden_layers']}")
    output = model(tt_pixel_values, tt_input_ids)
    output_torch = ttnn.to_torch(output)
    print(f"Inference Complete. Output shape: {output_torch.shape}")
    print(f"  Output range: [{output_torch.min().item():.4f}, {output_torch.max().item():.4f}]")
    
    ttnn.close_device(device)
    return output_torch


def run_lfm2_vl_demo_with_real_image(image_path: str, model_path: str = None):
    """Run demo with a real image and optionally real model weights."""
    run_lfm2_vl_demo(
        image_path=image_path,
        use_mock=model_path is None,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LFM2.5-VL Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.safetensors)")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock parameters")
    args = parser.parse_args()
    
    if args.weights:
        run_lfm2_vl_demo(image_path=args.weights, use_mock=False)
    elif args.image:
        run_lfm2_vl_demo(image_path=args.image, use_mock=args.mock)
    else:
        run_lfm2_vl_demo(use_mock=True)