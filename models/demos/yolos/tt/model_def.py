# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import transformers
import ttnn

def yolos_patch_embeddings(config, pixel_values, *, parameters):
    # batch_size, img_c, img_h, img_w = pixel_values.shape # NCHW
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = 16
    patch_count = img_h // patch_size  
    patch_size_sq_trpl = int(patch_size * patch_size * 3) 
    patch_count_all = int(patch_count * patch_count)
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_w // patch_size, 4 * patch_size))
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    patch_embedding_output = pixel_values @ parameters.projection.weight
    patch_embedding_output = patch_embedding_output + parameters.projection.bias

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, patch_size_sq_trpl))

    return patch_embedding_output

def yolos_embeddings(config, pixel_values, cls_token, detection_tokens, position_embeddings, *, parameters):
    parameters = parameters.vit.embeddings

    patch_embeddings = yolos_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)
    
    # Concatenate [cls_token, detection_tokens, patch_embeddings]
    cls_det_tokens = ttnn.concat((cls_token, detection_tokens), dim=1)
    embedding_output = ttnn.concat((cls_det_tokens, patch_embeddings), dim=1)
    
    embedding_output = embedding_output + position_embeddings

    return embedding_output

def yolos_layernorm_before(config, hidden_states, *, parameters):
    return ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
    )

def yolos_layernorm_after(config, hidden_states, *, parameters):
    return ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
    )

def yolos_attention(config, hidden_states, attention_mask, *, parameters):
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query = hidden_states @ parameters.attention.query.weight
    query = query + parameters.attention.query.bias
    query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT)
    query = ttnn.permute(query, (0, 2, 1, 3))

    key = hidden_states @ parameters.attention.key.weight
    key = key + parameters.attention.key.bias
    key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = hidden_states @ parameters.attention.value.weight
    value = value + parameters.attention.value.bias
    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT)
    value = ttnn.permute(value, (0, 2, 1, 3))

    attention_scores = query @ key
    attention_scores = attention_scores * (1 / (head_size**0.5))
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    context_layer = attention_probs @ value
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)

    self_output = context_layer
    self_output = self_output @ parameters.output.dense.weight
    self_output = self_output + parameters.output.dense.bias

    return self_output

def yolos_intermediate(hidden_states, *, parameters):
    output = hidden_states @ parameters.dense.weight
    output = output + parameters.dense.bias
    output = ttnn.gelu(output)
    return output

def yolos_output(config, hidden_states, residual, *, parameters):
    output = hidden_states @ parameters.dense.weight
    output = output + parameters.dense.bias
    output = output + residual
    return output

def yolos_feedforward(config, hidden_states, attention_output, *, parameters):
    intermediate = yolos_intermediate(hidden_states, parameters=parameters.intermediate)
    hidden_states = yolos_output(config, intermediate, attention_output, parameters=parameters.output)
    return hidden_states

def yolos_layer(config, hidden_states, attention_mask, *, parameters):
    layernorm_before_output = yolos_layernorm_before(config, hidden_states, parameters=parameters)
    attention_output = yolos_attention(config, layernorm_before_output, attention_mask, parameters=parameters.attention)
    attention_output = attention_output + hidden_states
    layernorm_after_output = yolos_layernorm_after(config, attention_output, parameters=parameters)
    feedforward_output = yolos_feedforward(config, layernorm_after_output, attention_output, parameters=parameters)
    return feedforward_output

def yolos_encoder(config, hidden_states, attention_mask, *, parameters):
    encoder_input = hidden_states
    encoder_output = None
    for encoder_parameters in parameters.layer:
        encoder_output = yolos_layer(config, encoder_input, attention_mask, parameters=encoder_parameters)
        encoder_input = encoder_output
    return encoder_output

def yolos(config, pixel_values, cls_token, detection_tokens, position_embeddings, *, parameters):
    
    # 1. Embeddings
    embedding_output = yolos_embeddings(
        config, 
        pixel_values, 
        cls_token, 
        detection_tokens, 
        position_embeddings, 
        parameters=parameters
    )

    # 2. Encoder
    encoder_output = yolos_encoder(
        config,
        embedding_output,
        attention_mask=None,
        parameters=parameters.vit.encoder,
    )

    # 3. Final LayerNorm (from ViT)
    sequence_output = ttnn.layer_norm(
        encoder_output,
        weight=parameters.vit.layernorm.weight,
        bias=parameters.vit.layernorm.bias,
    )
    
    # 4. Heads
    # Class Labels Classifier
    class_logits = sequence_output @ parameters.class_labels_classifier.weight
    class_logits = class_logits + parameters.class_labels_classifier.bias
    
    # Bbox Predictor (MLP)
    bbox_pred = sequence_output @ parameters.bbox_predictor.layers[0].weight
    bbox_pred = bbox_pred + parameters.bbox_predictor.layers[0].bias
    bbox_pred = ttnn.relu(bbox_pred)
    
    bbox_pred = bbox_pred @ parameters.bbox_predictor.layers[1].weight
    bbox_pred = bbox_pred + parameters.bbox_predictor.layers[1].bias
    bbox_pred = ttnn.relu(bbox_pred)
    
    bbox_pred = bbox_pred @ parameters.bbox_predictor.layers[2].weight
    bbox_pred = bbox_pred + parameters.bbox_predictor.layers[2].bias
    
    return class_logits, bbox_pred


def custom_preprocessor(torch_model, name):
    parameters = {}
    
    # Helper for generic weights
    def preprocess_linear(weight, bias=None, dtype=ttnn.bfloat16):
        if bias is not None:
             return {"weight": ttnn.from_torch(weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT),
                     "bias": ttnn.from_torch(bias, dtype=dtype, layout=ttnn.TILE_LAYOUT)}
        else:
             return {"weight": ttnn.from_torch(weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT)}

    if isinstance(torch_model, transformers.YolosForObjectDetection):
        # 1. ViT Embeddings
        embed = torch_model.vit.embeddings
        weight = embed.patch_embeddings.projection.weight
        bias = embed.patch_embeddings.projection.bias
        
        three_times_hidden_size, c, _, _ = weight.shape
        pad_value = 4 - c
        preprocessed_weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
        preprocessed_weight = torch.reshape(
            preprocessed_weight, (int(three_times_hidden_size * (4 / c)), three_times_hidden_size)
        )
        
        parameters["vit"] = {"embeddings": {"patch_embeddings": {"projection": {}}}}
        parameters["vit"]["embeddings"]["patch_embeddings"]["projection"]["weight"] = ttnn.from_torch(
            preprocessed_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["vit"]["embeddings"]["patch_embeddings"]["projection"]["bias"] = ttnn.from_torch(
            bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        
        parameters["cls_token"] = ttnn.from_torch(embed.cls_token, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters["detection_tokens"] = ttnn.from_torch(embed.detection_tokens, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters["position_embeddings"] = ttnn.from_torch(
            embed.position_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        # 2. ViT Encoder
        parameters["vit"]["encoder"] = {"layer": []}
        for layer in torch_model.vit.encoder.layer:
            layer_params = {}
            # LayerNorm Before
            layer_params["layernorm_before"] = preprocess_linear(torch.zeros_like(layer.layernorm_before.weight), dtype=ttnn.bfloat16)  # Dummy weight for structure
            layer_params["layernorm_before"]["weight"] = ttnn.from_torch(layer.layernorm_before.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            layer_params["layernorm_before"]["bias"] = ttnn.from_torch(layer.layernorm_before.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            
            # Attention
            layer_params["attention"] = {}
            layer_params["attention"]["query"] = preprocess_linear(layer.attention.attention.query.weight, layer.attention.attention.query.bias)
            layer_params["attention"]["key"] = preprocess_linear(layer.attention.attention.key.weight, layer.attention.attention.key.bias)
            layer_params["attention"]["value"] = preprocess_linear(layer.attention.attention.value.weight, layer.attention.attention.value.bias)
            layer_params["attention"]["output"] = {"dense": preprocess_linear(layer.attention.output.dense.weight, layer.attention.output.dense.bias)}

            # LayerNorm After
            layer_params["layernorm_after"] = preprocess_linear(torch.zeros_like(layer.layernorm_after.weight), dtype=ttnn.bfloat16)
            layer_params["layernorm_after"]["weight"] = ttnn.from_torch(layer.layernorm_after.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            layer_params["layernorm_after"]["bias"] = ttnn.from_torch(layer.layernorm_after.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            
            # Intermediate
            layer_params["intermediate"] = {"dense": preprocess_linear(layer.intermediate.dense.weight, layer.intermediate.dense.bias)}
            
            # Output
            layer_params["output"] = {"dense": preprocess_linear(layer.output.dense.weight, layer.output.dense.bias)}
            
            parameters["vit"]["encoder"]["layer"].append(layer_params)
            
        # 3. Final LayerNorm
        parameters["vit"]["layernorm"] = {}
        parameters["vit"]["layernorm"]["weight"] = ttnn.from_torch(torch_model.vit.layernorm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters["vit"]["layernorm"]["bias"] = ttnn.from_torch(torch_model.vit.layernorm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # 4. Heads
        parameters["class_labels_classifier"] = preprocess_linear(torch_model.class_labels_classifier.weight, torch_model.class_labels_classifier.bias)
        
        parameters["bbox_predictor"] = {"layers": []}
        for layer in torch_model.bbox_predictor.layers:
            if isinstance(layer, torch.nn.Linear):
                parameters["bbox_predictor"]["layers"].append(preprocess_linear(layer.weight, layer.bias))
        
    return parameters
