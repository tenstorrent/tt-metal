import torch
from transformers import BarkModel
import os
from pathlib import Path

def extract_bark_weights():
    print("🚀 Extracting Bark Small weights...")
    model = BarkModel.from_pretrained("suno/bark-small")
    save_dir = Path("weights")
    save_dir.mkdir(exist_ok=True)

    # Dictionary to hold weight shapes for verification
    weights_summary = {}

    for name, param in model.named_parameters():
        # Replace dots with underscores for file names
        clean_name = name.replace('.', '_')
        weight_path = save_dir / f"{clean_name}.pt"
        torch.save(param.data, weight_path)
        weights_summary[name] = list(param.shape)
        
    import json
    with open("weights_summary.json", "w") as f:
        json.dump(weights_summary, f, indent=4)
    
    print(f"✅ Extracted {len(weights_summary)} weight tensors to {save_dir}/")

if __name__ == "__main__":
    extract_bark_weights()
