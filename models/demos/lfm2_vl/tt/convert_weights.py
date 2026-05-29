import torch
from safetensors import safe_open
import ttnn
from typing import Any, Dict
import os


def _tensor_to_ttnn(tensor: torch.Tensor, device) -> ttnn.Tensor:
    """Convert a torch tensor to a ttnn tensor on the given device."""
    return ttnn.from_torch(tensor, device=device)


def _make_obj(**kwargs) -> object:
    """Create a simple attribute-accessible object from keyword arguments."""
    return type("obj", (object,), kwargs)


def _make_layer_obj(layer_params: Dict[str, Any]) -> object:
    """Convert a flat layer parameter dict into a nested attribute object.
    
    Supports both conv blocks and attention blocks.
    conv blocks: input_projection, conv, output_projection
    attention blocks: input_layernorm, post_attention_layernorm, self_attn (q/k/v/o_proj), mlp (gate/up/down_proj)
    """
    result = {}
    for key, value in layer_params.items():
        if key in ("self_attn", "mlp"):
            result[key] = _make_obj(**value)
        else:
            result[key] = _make_obj(weight=value)
    return _make_obj(**result)


def convert_lfm2_weights(model_path: str, device, model_config: Dict[str, Any] = None) -> object:
    """Load LFM2.5-VL weights from a safetensors file and return a parameter object.
    
    The returned object has the structure expected by TtLfm2VlModel:
        parameters.embed_tokens.weight
        parameters.norm.weight
        parameters.vision.patch_embed.weight
        parameters.vision.layers[i].norm1.weight
        parameters.vision.layers[i].attn.qkv.weight
        parameters.vision.layers[i].attn.proj.weight
        parameters.vision.layers[i].norm2.weight
        parameters.vision.layers[i].mlp.fc1.weight
        parameters.vision.layers[i].mlp.fc2.weight
        parameters.projector.gate_proj.weight
        parameters.projector.down_proj.weight
        parameters.layers[i].input_projection.weight / input_layernorm.weight
        parameters.layers[i].conv.weight
        parameters.layers[i].output_projection.weight / post_attention_layernorm.weight
        parameters.layers[i].self_attn.q/k/v/o_proj.weight
        parameters.layers[i].mlp.gate/up/down_proj.weight
    """
    if model_config is None:
        from models.demos.lfm2_vl.tt.model_config import create_model_config
        model_config = create_model_config(1, 128)
    
    config = model_config
    num_layers = config["num_hidden_layers"]
    vision_config = config["vision_config"]
    
    # Build the parameter structure from the checkpoint
    with safe_open(model_path, framework="pt") as f:
        all_keys = f.keys()
    
    # Helper to get a tensor
    def get_tensor(key):
        with safe_open(model_path, framework="pt") as f:
            return f.get_tensor(key)
    
    params = {}
    
    # Embed tokens
    if "model.embed_tokens.weight" in all_keys:
        params["embed_tokens"] = _make_obj(
            weight=_tensor_to_ttnn(get_tensor("model.embed_tokens.weight"), device)
        )
    else:
        # Fallback: random init
        params["embed_tokens"] = _make_obj(
            weight=_tensor_to_ttnn(torch.randn(config["vocab_size"], config["hidden_size"]), device)
        )
    
    # Final norm
    if "model.norm.weight" in all_keys:
        params["norm"] = _make_obj(
            weight=_tensor_to_ttnn(get_tensor("model.norm.weight"), device)
        )
    else:
        params["norm"] = _make_obj(
            weight=_tensor_to_ttnn(torch.randn(config["hidden_size"]), device)
        )
    
    # === Vision encoder (SigLIP2) ===
    vision_params = {}
    
    # Patch embed
    if "model.vision_encoder.patch_embed.weight" in all_keys:
        vision_params["patch_embed"] = _make_obj(
            weight=_tensor_to_ttnn(get_tensor("model.vision_encoder.patch_embed.weight"), device)
        )
    else:
        vision_params["patch_embed"] = _make_obj(
            weight=_tensor_to_ttnn(
                torch.randn(3 * 16 * 16, vision_config["hidden_size"]), device
            )
        )
    
    # Position embedding (if present)
    pe_keys = [k for k in all_keys if "vision_encoder" in k and "pos" in k.lower()]
    if pe_keys:
        vision_params["pos_embed"] = _make_obj(
            weight=_tensor_to_ttnn(get_tensor(pe_keys[0]), device)
        )
    
    # Vision layers
    vision_layer_params = []
    for i in range(vision_config["num_hidden_layers"]):
        layer = {}
        
        # Look for common SigLIP2 checkpoint key patterns
        # Pattern 1: model.vision_encoder.layers.{i}.norm1.weight
        norm1_key = f"model.vision_encoder.layers.{i}.norm1.weight"
        norm2_key = f"model.vision_encoder.layers.{i}.norm2.weight"
        
        # Pattern 2: model.vision_encoder.blocks.{i}.norm1.weight
        alt_norm1_key = f"model.vision_encoder.blocks.{i}.norm1.weight"
        alt_norm2_key = f"model.vision_encoder.blocks.{i}.norm2.weight"
        
        norm1_key = norm1_key if norm1_key in all_keys else alt_norm1_key
        norm2_key = norm2_key if norm2_key in all_keys else alt_norm2_key
        
        if norm1_key in all_keys:
            layer["norm1"] = _make_obj(weight=_tensor_to_ttnn(get_tensor(norm1_key), device))
        else:
            layer["norm1"] = _make_obj(
                weight=_tensor_to_ttnn(torch.randn(vision_config["hidden_size"]), device)
            )
        
        if norm2_key in all_keys:
            layer["norm2"] = _make_obj(weight=_tensor_to_ttnn(get_tensor(norm2_key), device))
        else:
            layer["norm2"] = _make_obj(
                weight=_tensor_to_ttnn(torch.randn(vision_config["hidden_size"]), device)
            )
        
        # Attention QKV projection (fused or separate)
        qkv_key = f"model.vision_encoder.layers.{i}.attn.qkv.weight"
        alt_qkv_key = f"model.vision_encoder.blocks.{i}.attn.qkv.weight"
        q_key = f"model.vision_encoder.layers.{i}.attn.q.weight"
        k_key = f"model.vision_encoder.layers.{i}.attn.k.weight"
        v_key = f"model.vision_encoder.layers.{i}.attn.v.weight"
        proj_key = f"model.vision_encoder.layers.{i}.attn.proj.weight"
        alt_proj_key = f"model.vision_encoder.blocks.{i}.attn.proj.weight"
        
        attention = {}
        if qkv_key in all_keys or alt_qkv_key in all_keys:
            actual_qkv = qkv_key if qkv_key in all_keys else alt_qkv_key
            attention["qkv"] = _make_obj(weight=_tensor_to_ttnn(get_tensor(actual_qkv), device))
        elif q_key in all_keys:
            attention["q"] = _make_obj(weight=_tensor_to_ttnn(get_tensor(q_key), device))
            attention["k"] = _make_obj(weight=_tensor_to_ttnn(get_tensor(k_key), device))
            attention["v"] = _make_obj(weight=_tensor_to_ttnn(get_tensor(v_key), device))
        else:
            attention["qkv"] = _make_obj(
                weight=_tensor_to_ttnn(
                    torch.randn(3 * vision_config["hidden_size"], vision_config["hidden_size"]), device
                )
            )
        
        actual_proj = proj_key if proj_key in all_keys else alt_proj_key
        if actual_proj in all_keys:
            attention["proj"] = _make_obj(weight=_tensor_to_ttnn(get_tensor(actual_proj), device))
        else:
            attention["proj"] = _make_obj(
                weight=_tensor_to_ttnn(
                    torch.randn(vision_config["hidden_size"], vision_config["hidden_size"]), device
                )
            )
        
        layer["attn"] = _make_obj(**attention)
        
        # MLP
        fc1_key = f"model.vision_encoder.layers.{i}.mlp.fc1.weight"
        fc2_key = f"model.vision_encoder.layers.{i}.mlp.fc2.weight"
        alt_fc1_key = f"model.vision_encoder.blocks.{i}.mlp.fc1.weight"
        alt_fc2_key = f"model.vision_encoder.blocks.{i}.mlp.fc2.weight"
        
        actual_fc1 = fc1_key if fc1_key in all_keys else alt_fc1_key
        actual_fc2 = fc2_key if fc2_key in all_keys else alt_fc2_key
        
        mlp = {}
        # SigLIP2 MLP typically uses hidden_size -> 4*hidden_size -> hidden_size
        mlp_hidden = 4 * vision_config["hidden_size"]
        
        if actual_fc1 in all_keys:
            mlp["fc1"] = _make_obj(weight=_tensor_to_ttnn(get_tensor(actual_fc1), device))
        else:
            mlp["fc1"] = _make_obj(
                weight=_tensor_to_ttnn(torch.randn(mlp_hidden, vision_config["hidden_size"]), device)
            )
        
        if actual_fc2 in all_keys:
            mlp["fc2"] = _make_obj(weight=_tensor_to_ttnn(get_tensor(actual_fc2), device))
        else:
            mlp["fc2"] = _make_obj(
                weight=_tensor_to_ttnn(torch.randn(vision_config["hidden_size"], mlp_hidden), device)
            )
        
        layer["mlp"] = _make_obj(**mlp)
        vision_layer_params.append(_make_obj(**layer))
    
    vision_params["layers"] = vision_layer_params
    params["vision"] = _make_obj(**vision_params)
    
    # === Projector ===
    projector_params = {}
    gate_key = "model.projector.gate_proj.weight"
    down_key = "model.projector.down_proj.weight"
    
    if gate_key in all_keys:
        projector_params["gate_proj"] = _make_obj(
            weight=_tensor_to_ttnn(get_tensor(gate_key), device)
        )
    else:
        projector_params["gate_proj"] = _make_obj(
            weight=_tensor_to_ttnn(
                torch.randn(config["projector_hidden_size"], vision_config["hidden_size"]), device
            )
        )
    
    if down_key in all_keys:
        projector_params["down_proj"] = _make_obj(
            weight=_tensor_to_ttnn(get_tensor(down_key), device)
        )
    else:
        projector_params["down_proj"] = _make_obj(
            weight=_tensor_to_ttnn(
                torch.randn(config["hidden_size"], config["projector_hidden_size"]), device
            )
        )
    
    params["projector"] = _make_obj(**projector_params)
    
    # === Text backbone layers ===
    layer_list = []
    for i in range(num_layers):
        layer_type = config["layer_types"][i] if "layer_types" in config else "full_attention"
        
        # Determine checkpoint key prefix
        # Try common patterns
        prefixes = [
            f"model.layers.{i}",
            f"model.text_layers.{i}",
        ]
        prefix = next((p for p in prefixes if any(k.startswith(p) for k in all_keys)), f"model.layers.{i}")
        
        layer_params = {}
        
        if layer_type == "conv":
            # Conv block: input_projection, conv, output_projection
            try:
                ip_key = f"{prefix}.input_projection.weight"
                conv_key = f"{prefix}.conv.weight"
                op_key = f"{prefix}.output_projection.weight"
                
                if ip_key in all_keys:
                    layer_params["input_projection"] = _make_obj(
                        weight=_tensor_to_ttnn(get_tensor(ip_key), device)
                    )
                else:
                    layer_params["input_projection"] = _make_obj(
                        weight=_tensor_to_ttnn(
                            torch.randn(config["hidden_size"], 3 * config["hidden_size"]), device
                        )
                    )
                
                if conv_key in all_keys:
                    layer_params["conv"] = _make_obj(
                        weight=_tensor_to_ttnn(get_tensor(conv_key), device)
                    )
                else:
                    layer_params["conv"] = _make_obj(
                        weight=_tensor_to_ttnn(torch.randn(config["hidden_size"], 1, 3), device)
                    )
                
                if op_key in all_keys:
                    layer_params["output_projection"] = _make_obj(
                        weight=_tensor_to_ttnn(get_tensor(op_key), device)
                    )
                else:
                    layer_params["output_projection"] = _make_obj(
                        weight=_tensor_to_ttnn(
                            torch.randn(config["hidden_size"], config["hidden_size"]), device
                        )
                    )
            except Exception:
                continue
        else:
            # Attention block
            try:
                # RMS norms
                iln_key = f"{prefix}.input_layernorm.weight"
                paln_key = f"{prefix}.post_attention_layernorm.weight"
                
                if iln_key in all_keys:
                    layer_params["input_layernorm"] = _make_obj(
                        weight=_tensor_to_ttnn(get_tensor(iln_key), device)
                    )
                else:
                    layer_params["input_layernorm"] = _make_obj(
                        weight=_tensor_to_ttnn(torch.randn(config["hidden_size"]), device)
                    )
                
                if paln_key in all_keys:
                    layer_params["post_attention_layernorm"] = _make_obj(
                        weight=_tensor_to_ttnn(get_tensor(paln_key), device)
                    )
                else:
                    layer_params["post_attention_layernorm"] = _make_obj(
                        weight=_tensor_to_ttnn(torch.randn(config["hidden_size"]), device)
                    )
                
                # Self-attention
                sa = {}
                for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    proj_key = f"{prefix}.self_attn.{proj_name}.weight"
                    if proj_key in all_keys:
                        sa[proj_name] = _make_obj(
                            weight=_tensor_to_ttnn(get_tensor(proj_key), device)
                        )
                    else:
                        sa[proj_name] = _make_obj(
                            weight=_tensor_to_ttnn(
                                torch.randn(config["hidden_size"], config["hidden_size"]), device
                            )
                        )
                layer_params["self_attn"] = _make_obj(**sa)
                
                # MLP
                mlp = {}
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    proj_key = f"{prefix}.mlp.{proj_name}.weight"
                    if proj_key in all_keys:
                        weight = _tensor_to_ttnn(get_tensor(proj_key), device)
                    else:
                        if proj_name == "down_proj":
                            weight = _tensor_to_ttnn(
                                torch.randn(config["hidden_size"], config["intermediate_size"]), device
                            )
                        else:
                            weight = _tensor_to_ttnn(
                                torch.randn(config["intermediate_size"], config["hidden_size"]), device
                            )
                    mlp[proj_name] = _make_obj(weight=weight)
                layer_params["mlp"] = _make_obj(**mlp)
            except Exception:
                continue
        
        layer_list.append(_make_obj(**layer_params))
    
    params["layers"] = layer_list
    
    return _make_obj(**params)


def save_parameters(parameters: object, save_path: str):
    """Save converted parameters to disk (torch state dict format)."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # Convert ttnn tensors back to torch for serialization
    state = {}
    
    def _extract(prefix, obj):
        if hasattr(obj, "weight") and hasattr(obj.weight, "shape"):
            try:
                state[prefix + ".weight"] = ttnn.to_torch(obj.weight)
            except Exception:
                pass
        for attr in dir(obj):
            if attr.startswith("_"):
                continue
            val = getattr(obj, attr)
            if isinstance(val, list):
                for j, item in enumerate(val):
                    _extract(f"{prefix}.{attr}.{j}", item)
            elif hasattr(val, "__dict__") or hasattr(val, "__slots__"):
                _extract(f"{prefix}.{attr}", val)
    
    _extract("", parameters)
    torch.save(state, save_path)
    print(f"Parameters saved to {save_path}")