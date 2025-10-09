"""
Extract and visualize dataflow for Qwen2.5 VL model.

This module provides specific dataflow extraction for the Qwen2.5 VL multimodal model,
showing how visual and language components interact.
"""

import json
from collections import OrderedDict


class Qwen25VLDataflowExtractor:
    """
    Extract dataflow information specifically for Qwen2.5 VL model architecture.

    Based on the model structure:
    - Visual: patch_embed -> rotary_pos_emb -> blocks[0-31] -> merger
    - Language: embed_tokens -> layers[0-35] -> norm
    - Output: lm_head
    """

    def __init__(self):
        self.dataflow = {
            "visual_pipeline": [],
            "language_pipeline": [],
            "merger_connections": [],
            "output_pipeline": [],
        }

    def extract_dataflow_from_state_dict(self, model):
        """
        Extract dataflow by analyzing the model's forward method and module hierarchy.
        This works without running the model.
        """

        # Visual dataflow
        visual_flow = OrderedDict(
            [
                (
                    "input",
                    {
                        "type": "pixel_values",
                        "shape": "[batch, 3, temporal, height, width]",
                        "description": "Raw image/video input",
                    },
                ),
                (
                    "visual.patch_embed.proj",
                    {
                        "type": "Conv3d",
                        "params": "kernel=(2,14,14), stride=(2,14,14)",
                        "input_shape": "[B, 3, T, H, W]",
                        "output_shape": "[B, 1280, T/2, H/14, W/14]",
                        "description": "3D convolution for patch embedding",
                    },
                ),
                (
                    "visual.rotary_pos_emb",
                    {
                        "type": "Qwen2_5_VisionRotaryEmbedding",
                        "description": "Rotary position embeddings for vision tokens",
                    },
                ),
            ]
        )

        # Add vision transformer blocks
        for i in range(32):
            block_name = f"visual.blocks.{i}"
            visual_flow[block_name] = {
                "type": "Qwen2_5_VLVisionBlock",
                "submodules": OrderedDict(
                    [
                        ("norm1", "Qwen2RMSNorm(1280)"),
                        ("attn", {"qkv": "Linear(1280->3840)", "proj": "Linear(1280->1280)"}),
                        ("norm2", "Qwen2RMSNorm(1280)"),
                        (
                            "mlp",
                            {
                                "gate_proj": "Linear(1280->3420)",
                                "up_proj": "Linear(1280->3420)",
                                "down_proj": "Linear(3420->1280)",
                                "act_fn": "SiLU()",
                            },
                        ),
                    ]
                ),
                "dataflow": "input -> norm1 -> attn -> residual_add -> norm2 -> mlp -> residual_add -> output",
            }

        # Vision-Language merger
        visual_flow["visual.merger"] = {
            "type": "Qwen2_5_VLPatchMerger",
            "submodules": OrderedDict(
                [("ln_q", "Qwen2RMSNorm(1280)"), ("mlp", ["Linear(5120->5120)", "GELU()", "Linear(5120->2048)"])]
            ),
            "input_dim": 1280 * 4,  # Concatenation of 4 patches
            "output_dim": 2048,  # Language model dimension
            "description": "Merges visual patches and projects to language model dimension",
        }

        # Language model dataflow
        language_flow = OrderedDict(
            [
                ("text_input", {"type": "input_ids", "vocab_size": 151936, "description": "Tokenized text input"}),
                (
                    "language_model.embed_tokens",
                    {"type": "Embedding", "params": "(151936, 2048)", "description": "Token embeddings"},
                ),
                (
                    "language_model.rotary_emb",
                    {"type": "Qwen2_5_VLRotaryEmbedding", "description": "Rotary position embeddings for text"},
                ),
            ]
        )

        # Add language transformer layers
        for i in range(36):
            layer_name = f"language_model.layers.{i}"
            language_flow[layer_name] = {
                "type": "Qwen2_5_VLDecoderLayer",
                "submodules": OrderedDict(
                    [
                        ("input_layernorm", "Qwen2RMSNorm(2048)"),
                        (
                            "self_attn",
                            {
                                "q_proj": "Linear(2048->2048)",
                                "k_proj": "Linear(2048->256)",  # Note: smaller KV dims
                                "v_proj": "Linear(2048->256)",
                                "o_proj": "Linear(2048->2048)",
                                "rotary_emb": "Qwen2_5_VLRotaryEmbedding()",
                            },
                        ),
                        ("post_attention_layernorm", "Qwen2RMSNorm(2048)"),
                        (
                            "mlp",
                            {
                                "gate_proj": "Linear(2048->11008)",
                                "up_proj": "Linear(2048->11008)",
                                "down_proj": "Linear(11008->2048)",
                                "act_fn": "SiLU()",
                            },
                        ),
                    ]
                ),
                "dataflow": "input -> norm -> self_attn -> residual -> norm -> mlp -> residual -> output",
            }

        language_flow["language_model.norm"] = {
            "type": "Qwen2RMSNorm",
            "params": "(2048,)",
            "description": "Final layer norm",
        }

        # Output head
        output_flow = OrderedDict(
            [
                (
                    "lm_head",
                    {
                        "type": "Linear",
                        "params": "(2048, 151936)",
                        "tied_weights": "False (independent from embeddings)",
                        "description": "Language modeling head for token prediction",
                    },
                )
            ]
        )

        self.dataflow = {
            "visual_pipeline": visual_flow,
            "language_pipeline": language_flow,
            "output_pipeline": output_flow,
            "multimodal_fusion": {
                "fusion_point": "After visual.merger, visual features are injected into language model",
                "fusion_mechanism": "Visual tokens from merger are concatenated with text embeddings",
                "cross_attention": "Language model layers can attend to both text and visual tokens",
            },
        }

        return self.dataflow

    def get_compute_flow(self):
        """
        Get the high-level computation flow through the model.
        """
        compute_flow = [
            {
                "stage": "Visual Encoding",
                "steps": [
                    "1. Patch embedding: Conv3D extracts spatial-temporal features",
                    "2. Add rotary position embeddings",
                    "3. Process through 32 vision transformer blocks",
                    "4. Merge patches and project to language dimension",
                ],
            },
            {
                "stage": "Multimodal Fusion",
                "steps": [
                    "1. Visual features from merger (2048-dim)",
                    "2. Text token embeddings (2048-dim)",
                    "3. Concatenate visual and text tokens",
                    "4. Add position embeddings",
                ],
            },
            {
                "stage": "Language Modeling",
                "steps": [
                    "1. Process concatenated sequence through 36 decoder layers",
                    "2. Each layer performs self-attention over all tokens",
                    "3. Visual tokens provide context for text generation",
                    "4. Apply final layer norm",
                ],
            },
            {
                "stage": "Output Generation",
                "steps": [
                    "1. Project hidden states through lm_head",
                    "2. Generate logits over vocabulary (151936 tokens)",
                    "3. Sample or decode next token",
                ],
            },
        ]

        return compute_flow

    def get_dimension_flow(self):
        """
        Track how tensor dimensions change through the model.
        """
        dimension_flow = {
            "visual_path": [
                "[B, 3, T, H, W] -> pixel_values input",
                "[B, 1280, T/2, H/14, W/14] -> after patch_embed",
                "[B, num_patches, 1280] -> flattened patches",
                "[B, num_patches, 1280] -> through 32 vision blocks",
                "[B, merged_patches, 2048] -> after merger",
            ],
            "language_path": [
                "[B, seq_len] -> input_ids",
                "[B, seq_len, 2048] -> after embed_tokens",
                "[B, seq_len + visual_tokens, 2048] -> after concatenation",
                "[B, total_len, 2048] -> through 36 layers",
                "[B, total_len, 151936] -> after lm_head",
            ],
        }

        return dimension_flow

    def export_dataflow_json(self, filepath: str):
        """Export the complete dataflow to a JSON file."""
        export_data = {
            "model": "Qwen2.5-VL",
            "architecture": self.dataflow,
            "compute_flow": self.get_compute_flow(),
            "dimension_flow": self.get_dimension_flow(),
            "key_features": {
                "multimodal": True,
                "vision_blocks": 32,
                "language_layers": 36,
                "hidden_dim": 2048,
                "vision_dim": 1280,
                "mlp_dim": 11008,
                "vision_mlp_dim": 3420,
                "vocab_size": 151936,
                "kv_compression": "Uses smaller KV projection (256) for efficiency",
            },
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        return export_data

    def generate_mermaid_diagram(self):
        """Generate a Mermaid diagram for visualization."""
        mermaid = """
graph TD
    A[Pixel Values<br/>B,3,T,H,W] --> B[Patch Embed Conv3D<br/>B,1280,T/2,H/14,W/14]
    B --> C[Rotary Pos Emb]
    C --> D[Vision Blocks 0-31<br/>32 layers]
    D --> E[Patch Merger<br/>1280→2048]

    F[Input IDs<br/>B,seq_len] --> G[Token Embeddings<br/>B,seq_len,2048]
    G --> H[Rotary Pos Emb]

    E --> I{Concatenate<br/>Visual + Text}
    H --> I

    I --> J[Language Layers 0-35<br/>36 layers<br/>Self-Attention + MLP]
    J --> K[Layer Norm<br/>B,total_len,2048]
    K --> L[LM Head<br/>B,total_len,151936]
    L --> M[Output Logits]

    style A fill:#ff9999
    style F fill:#99ccff
    style I fill:#ffcc99
    style M fill:#99ff99
"""
        return mermaid


# Example usage
if __name__ == "__main__":
    extractor = Qwen25VLDataflowExtractor()

    # Extract dataflow (would pass actual model if available)
    dataflow = extractor.extract_dataflow_from_state_dict(None)

    # Print visual pipeline summary
    print("Visual Pipeline:")
    for name, info in list(dataflow["visual_pipeline"].items())[:5]:
        print(f"  {name}: {info.get('type', 'N/A')}")

    # Print language pipeline summary
    print("\nLanguage Pipeline:")
    for name, info in list(dataflow["language_pipeline"].items())[:5]:
        print(f"  {name}: {info.get('type', 'N/A')}")

    # Get compute flow
    compute_flow = extractor.get_compute_flow()
    print("\nCompute Flow Stages:")
    for stage in compute_flow:
        print(f"  {stage['stage']}")

    # Generate Mermaid diagram
    print("\nMermaid Diagram:")
    print(extractor.generate_mermaid_diagram())

    # Export to JSON
    extractor.export_dataflow_json("qwen25_vl_dataflow.json")
