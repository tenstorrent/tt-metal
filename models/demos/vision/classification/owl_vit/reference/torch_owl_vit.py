# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Reference PyTorch implementation for OWL-ViT object detection.
This file serves as a reference for understanding the model structure
and for validating the TTNN implementation.

OWL-ViT Architecture:
1. Vision Encoder (ViT-B/32): Processes images into patch embeddings
2. Text Encoder (CLIP text model): Encodes text queries into embeddings
3. Box Prediction Head: MLP that predicts bounding boxes from visual features
4. Class Prediction Head: Computes region-text similarity for classification
"""

import requests
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


def load_owl_vit_model(model_name: str = "google/owlvit-base-patch32"):
    """Load the OWL-ViT model and processor from HuggingFace."""
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name)
    return processor, model


def run_owl_vit_inference(
    image,
    text_queries: list[str],
    processor,
    model,
    threshold: float = 0.1,
):
    """
    Run OWL-ViT inference on an image with text queries.

    Args:
        image: PIL Image or path to image
        text_queries: List of text queries for detection (e.g., ["a photo of a cat", "a photo of a dog"])
        processor: OwlViTProcessor
        model: OwlViTForObjectDetection
        threshold: Detection confidence threshold

    Returns:
        Dictionary with boxes, scores, and labels
    """
    if isinstance(image, str):
        if image.startswith("http"):
            image = Image.open(requests.get(image, stream=True).raw)
        else:
            image = Image.open(image)

    # Prepare inputs
    texts = [text_queries]  # Batch of 1
    inputs = processor(text=texts, images=image, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    target_sizes = torch.Tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

    return results[0], outputs


def visualize_detections(image, results, text_queries):
    """
    Visualize detected objects on an image.

    Args:
        image: PIL Image
        results: Detection results dictionary
        text_queries: List of text queries
    """
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    print(f"Detected {len(boxes)} objects:")
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"  {text_queries[label]}: confidence={score.item():.3f}, box={box}")

    return boxes, scores, labels


def extract_owl_vit_components(model):
    """
    Extract the key components from OWL-ViT model for TTNN implementation.

    Returns:
        Dictionary with model components and their configurations
    """
    components = {
        "vision_model": {
            "embeddings": model.owlvit.vision_model.embeddings,
            "encoder": model.owlvit.vision_model.encoder,
            "pre_layernorm": model.owlvit.vision_model.pre_layernorm,
            "post_layernorm": model.owlvit.vision_model.post_layernorm,
        },
        "text_model": {
            "embeddings": model.owlvit.text_model.embeddings,
            "encoder": model.owlvit.text_model.encoder,
            "final_layer_norm": model.owlvit.text_model.final_layer_norm,
        },
        "visual_projection": model.owlvit.visual_projection,
        "text_projection": model.owlvit.text_projection,
        "box_head": model.box_head,
        "class_head": model.class_head,
        "layer_norm": model.layer_norm,
    }

    # Vision model config
    vision_config = model.config.vision_config
    print("Vision Model Config:")
    print(f"  hidden_size: {vision_config.hidden_size}")
    print(f"  intermediate_size: {vision_config.intermediate_size}")
    print(f"  num_hidden_layers: {vision_config.num_hidden_layers}")
    print(f"  num_attention_heads: {vision_config.num_attention_heads}")
    print(f"  image_size: {vision_config.image_size}")
    print(f"  patch_size: {vision_config.patch_size}")

    # Text model config
    text_config = model.config.text_config
    print("\nText Model Config:")
    print(f"  hidden_size: {text_config.hidden_size}")
    print(f"  intermediate_size: {text_config.intermediate_size}")
    print(f"  num_hidden_layers: {text_config.num_hidden_layers}")
    print(f"  num_attention_heads: {text_config.num_attention_heads}")
    print(f"  vocab_size: {text_config.vocab_size}")
    print(f"  max_position_embeddings: {text_config.max_position_embeddings}")

    return components


if __name__ == "__main__":
    # Demo usage
    print("Loading OWL-ViT model...")
    processor, model = load_owl_vit_model()

    print("\nExtracting model components...")
    components = extract_owl_vit_components(model)

    # Run sample inference
    print("\n\nRunning sample inference...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    text_queries = ["a photo of a cat", "a photo of a dog", "a photo of a remote control"]

    try:
        image = Image.open(requests.get(url, stream=True).raw)
        results, outputs = run_owl_vit_inference(image, text_queries, processor, model)
        visualize_detections(image, results, text_queries)

        # Print output shapes for TTNN implementation reference
        print("\n\nOutput shapes for TTNN implementation:")
        print(f"  logits: {outputs.logits.shape}")
        print(f"  pred_boxes: {outputs.pred_boxes.shape}")
        print(f"  text_embeds: {outputs.text_embeds.shape if outputs.text_embeds is not None else 'None'}")
        print(f"  image_embeds: {outputs.image_embeds.shape if outputs.image_embeds is not None else 'None'}")
    except Exception as e:
        print(f"Could not run sample inference: {e}")
        print("This is expected if running without network access.")
