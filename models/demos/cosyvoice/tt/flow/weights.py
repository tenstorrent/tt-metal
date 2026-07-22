"""Weight loading for CosyVoice2 flow model (encoder + UNet1D estimator).

Splits `flow.pt` (1121 keys) into:
  - encoder weights (UpsampleConformerEncoder, 206 keys)
  - decoder/estimator weights (CausalConditionalDecoder UNet1D, ~900 keys)
  - input_embedding (6561, 512)
  - encoder_proj (80, 512)
  - spk_embed_affine_layer (80, 192)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def load_flow_pt(flow_pt_path: str | Path) -> Dict[str, torch.Tensor]:
    return torch.load(str(flow_pt_path), map_location="cpu", weights_only=True)


def split_flow_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Split flow.pt into component weight dicts."""
    components = {
        "encoder": {},
        "decoder": {},
        "input_embedding": {},
        "encoder_proj": {},
        "spk_embed_affine_layer": {},
    }

    for key, tensor in state_dict.items():
        if key.startswith("encoder."):
            components["encoder"][key] = tensor
        elif key.startswith("decoder."):
            components["decoder"][key] = tensor
        elif key.startswith("input_embedding."):
            components["input_embedding"][key] = tensor
        elif key.startswith("encoder_proj."):
            components["encoder_proj"][key] = tensor
        elif key.startswith("spk_embed_affine_layer."):
            components["spk_embed_affine_layer"][key] = tensor
        else:
            raise ValueError(f"Unexpected key in flow.pt: {key}")

    return components


def load_flow_weights(
    flow_pt_path: str | Path,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load flow.pt and return component weight dicts."""
    sd = load_flow_pt(flow_pt_path)
    return split_flow_weights(sd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-pt", type=str, default="model_data/cosyvoice2-0.5B/flow.pt")
    args = parser.parse_args()

    components = load_flow_weights(args.flow_pt)
    for name, weights in components.items():
        print(f"{name}: {len(weights)} keys")
        for k in sorted(weights.keys())[:3]:
            print(f"  {k}: {list(weights[k].shape)}")
        if len(weights) > 3:
            print(f"  ... ({len(weights)} total)")
