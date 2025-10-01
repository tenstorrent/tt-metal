"""
Practical example of extracting dataflow from HuggingFace models.

This shows how to get the actual execution flow, including:
1. Module execution order
2. Tensor shapes at each step
3. Module dependencies
4. Computational graph structure
"""

import json
from typing import Any, Dict

import torch
from transformers import AutoModel, AutoTokenizer


def extract_dataflow_simple(model, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Simple method to extract dataflow using hooks.
    Returns execution order and tensor shapes.
    """
    execution_log = []
    hooks = []

    def create_hook(name: str):
        def hook(module, input, output):
            log_entry = {"name": name, "module_type": module.__class__.__name__, "order": len(execution_log)}

            # Log input info
            if isinstance(input, tuple) and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    log_entry["input_shape"] = list(input[0].shape)
            elif isinstance(input, torch.Tensor):
                log_entry["input_shape"] = list(input.shape)

            # Log output info
            if isinstance(output, torch.Tensor):
                log_entry["output_shape"] = list(output.shape)
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    log_entry["output_shape"] = list(output[0].shape)
                    log_entry["output_tuple_size"] = len(output)
            elif hasattr(output, "last_hidden_state"):
                log_entry["output_shape"] = list(output.last_hidden_state.shape)
                log_entry["output_type"] = "ModelOutput"

            execution_log.append(log_entry)

        return hook

    # Register hooks on all modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            h = module.register_forward_hook(create_hook(name))
            hooks.append(h)

    # Run forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Process execution log to extract dataflow
    dataflow = {
        "execution_order": [entry["name"] for entry in execution_log],
        "module_details": {entry["name"]: entry for entry in execution_log},
        "total_modules_executed": len(execution_log),
    }

    # Extract unique execution path (removing duplicates while preserving order)
    seen = set()
    unique_order = []
    for name in dataflow["execution_order"]:
        if name not in seen:
            seen.add(name)
            unique_order.append(name)
    dataflow["unique_execution_path"] = unique_order

    return dataflow


def visualize_dataflow_simple(dataflow: Dict[str, Any], max_modules: int = 20):
    """Simple visualization of dataflow."""
    print("Model Execution Flow")
    print("=" * 60)
    print(f"Total modules executed: {dataflow['total_modules_executed']}")
    print(f"Unique modules: {len(dataflow['unique_execution_path'])}")
    print("\nExecution Path (first {} modules):".format(max_modules))

    for i, module_name in enumerate(dataflow["unique_execution_path"][:max_modules]):
        details = dataflow["module_details"][module_name]
        input_shape = details.get("input_shape", "N/A")
        output_shape = details.get("output_shape", "N/A")
        print(f"{i+1:3d}. {module_name}")
        print(f"     Type: {details['module_type']}")
        print(f"     Input: {input_shape} -> Output: {output_shape}")


def extract_multimodal_dataflow():
    """
    Example: Extract dataflow from a multimodal model.
    This would work with Qwen2.5-VL or similar models.
    """
    # For demonstration, we'll use a smaller multimodal model
    # In practice, replace with: "Qwen/Qwen2.5-VL-7B-Instruct"

    print("Extracting dataflow from multimodal model...")

    # Example structure showing what you'd get from a multimodal model
    example_multimodal_flow = {
        "visual_branch": [
            "vision_model.embeddings.patch_embedding",
            "vision_model.encoder.layers.0",
            "vision_model.encoder.layers.1",
            # ... more vision layers
            "vision_model.post_layernorm",
            "visual_projection",  # Projects to language model dimension
        ],
        "language_branch": [
            "language_model.embed_tokens",
            "language_model.layers.0.self_attn",
            "language_model.layers.0.mlp",
            # ... more language layers
            "language_model.norm",
            "lm_head",
        ],
        "fusion_points": [
            {
                "location": "after_visual_projection",
                "description": "Visual features concatenated with text embeddings",
                "dimensions": "visual: [B, num_patches, hidden_dim], text: [B, seq_len, hidden_dim]",
            }
        ],
    }

    return example_multimodal_flow


def main():
    """Main demo of dataflow extraction."""

    # Example 1: Extract from a text model
    print("Example 1: Extracting dataflow from BERT")
    print("-" * 60)

    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare inputs
    text = "Extracting computational graph from transformer models."
    inputs = tokenizer(text, return_tensors="pt")

    # Extract dataflow
    dataflow = extract_dataflow_simple(model, inputs)

    # Visualize
    visualize_dataflow_simple(dataflow, max_modules=15)

    # Save to file
    with open("model_dataflow.json", "w") as f:
        # Convert to serializable format
        export_data = {
            "model_name": model_name,
            "unique_execution_path": dataflow["unique_execution_path"],
            "total_modules": dataflow["total_modules_executed"],
            "sample_shapes": {
                name: {"input": details.get("input_shape"), "output": details.get("output_shape")}
                for name, details in list(dataflow["module_details"].items())[:10]
            },
        }
        json.dump(export_data, f, indent=2)

    print("\n✓ Dataflow saved to model_dataflow.json")

    # Example 2: Show multimodal flow structure
    print("\n\nExample 2: Multimodal Model Dataflow Structure")
    print("-" * 60)

    multimodal_flow = extract_multimodal_dataflow()
    print("\nVisual Branch:")
    for module in multimodal_flow["visual_branch"][:5]:
        print(f"  → {module}")
    print("  → ...")

    print("\nLanguage Branch:")
    for module in multimodal_flow["language_branch"][:5]:
        print(f"  → {module}")
    print("  → ...")

    print("\nFusion Points:")
    for fusion in multimodal_flow["fusion_points"]:
        print(f"  • {fusion['location']}: {fusion['description']}")


# For actual Qwen2.5-VL model, you would use:
def extract_qwen25_vl_dataflow():
    """
    Extract dataflow specifically from Qwen2.5-VL model.
    Note: This requires the actual model to be loaded.
    """
    # from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    # model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    # processor = AutoProcessor.from_pretrained(model_name)

    # # Prepare multimodal inputs
    # messages = [{
    #     "role": "user",
    #     "content": [
    #         {"type": "image", "image": "path/to/image.jpg"},
    #         {"type": "text", "text": "What's in this image?"}
    #     ]
    # }]

    # # Process inputs
    # text = processor.apply_chat_template(messages, tokenize=False)
    # inputs = processor(text=text, images=image, return_tensors="pt")

    # # Extract dataflow
    # dataflow = extract_dataflow_simple(model, inputs)

    # This would give you the actual execution flow through:
    # 1. Visual encoder (patch_embed -> vision blocks -> merger)
    # 2. Language model (embeddings -> decoder layers -> lm_head)
    # 3. Cross-attention between modalities


if __name__ == "__main__":
    main()
