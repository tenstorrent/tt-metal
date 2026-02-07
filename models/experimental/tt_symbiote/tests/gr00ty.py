# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import time
import csv
import torch
import ttnn
import pytest
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from torch import nn
from tqdm import tqdm
from torch.distributions import Beta
from transformers import PretrainedConfig, AutoTokenizer

# --- 1. PERFORMANCE TRACKING ---
MODEL_STATS = []


def save_stats_to_csv(filename="gr00t_perf_report.csv"):
    if not MODEL_STATS:
        return
    fieldnames = ["Module Name", "Phase", "Type", "Wall Time (s)"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(MODEL_STATS)
    print(f"\n[INFO] Performance report saved to: {filename}")


# --- 2. THE DIRICHLET PATCH ---
def patched_beta_sample(self, sample_shape=torch.Size()):
    """Ensure Beta sampling is handled in float for hardware stability."""
    shadow_dist = Beta(self.concentration0.float(), self.concentration1.float())
    return shadow_dist.rsample(sample_shape).to(self.concentration0.dtype)


Beta.sample = patched_beta_sample


def setup_framework():
    PretrainedConfig._attn_implementation_autoset = False
    PretrainedConfig._attn_implementation_internal = "eager"


setup_framework()

# --- 3. PATH & IMPORTS ---
TEST_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = TEST_FILE_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.attention import TTNNGR00TSelfAttention
from modules.linear import TTNNLinear
from modules.normalization import TTNNLayerNorm
from modules.activation import TTNNSilu
from modules.conv import TTNNConv2dNHWC
from utils.device_management import set_device
from utils.module_replacement import register_module_replacement_dict

try:
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
except ValueError:
    Gr00tN1d6 = sys.modules["gr00t.model.gr00t_n1d6.gr00t_n1d6"].Gr00tN1d6


class Qwen2MLP(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = source.gate_proj, source.up_proj, source.down_proj
        self.act_fn = nn.SiLU()

    @classmethod
    def from_torch(cls, source):
        return cls(source)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# --- 4. MAIN TEST ---
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_gr00t_inference_validation(device):
    torch.manual_seed(42)
    model_id = "nvidia/GR00T-N1.6-3B"

    print(f"\n[INIT] Loading Tokenizer and Model Weights...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

    model = Gr00tN1d6.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
    ).eval()

    # --- Hardware Mapping ---
    lang_layer = model.backbone.model.language_model.model.layers[0]
    vision_layer = model.backbone.model.vision_model.vision_model.encoder.layers[0]
    action_block = model.action_head.model.transformer_blocks[0]

    op_mapping = {
        nn.Linear: TTNNLinear,
        nn.LayerNorm: TTNNLayerNorm,
        nn.SiLU: TTNNSilu,
        nn.Conv2d: TTNNConv2dNHWC,
        lang_layer.mlp.__class__: Qwen2MLP,
        lang_layer.self_attn.__class__: TTNNGR00TSelfAttention,
        vision_layer.self_attn.__class__: TTNNGR00TSelfAttention,
        action_block.attn1.__class__: TTNNGR00TSelfAttention,
    }

    transformed = register_module_replacement_dict(model, op_mapping, model_config={"dtype": ttnn.bfloat16})
    set_device(model, device)

    # Weights Assimilation (Tracked)
    for name, module in tqdm(transformed.items(), desc="Assimilating Weights"):
        if hasattr(module, "preprocess_weights"):
            start_as = time.time()
            module.preprocess_weights()
            module.move_weights_to_device()
            MODEL_STATS.append(
                {
                    "Module Name": name,
                    "Phase": "Assimilation",
                    "Type": module.__class__.__name__,
                    "Wall Time (s)": f"{time.time() - start_as:.6f}",
                }
            )

    # --- 5. Real Data Forward Pass ---
    prompt = "Pick up the red block."
    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    text_data = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=256, truncation=True)
    img_resized = raw_image.resize((224, 224))
    pixel_values = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float()
    pixel_values = (pixel_values / 255.0).unsqueeze(0).to(torch.bfloat16)

    real_inputs = {
        "input_ids": text_data["input_ids"],
        "attention_mask": text_data["attention_mask"],
        "pixel_values": [pixel_values],
        "embodiment_id": torch.zeros((1,), dtype=torch.long),
        "state": torch.zeros(1, 1, model.config.max_state_dim, dtype=torch.bfloat16),
        "action": torch.zeros(1, 128, model.config.max_action_dim, dtype=torch.bfloat16),
        "action_mask": torch.ones((1, 128), dtype=torch.bfloat16),
        "velocity": torch.zeros(1, 128, model.config.max_action_dim, dtype=torch.bfloat16),
    }

    print("\n[RUNNING] Hardware Inference Pass...")
    with torch.no_grad():
        try:
            start_pass = time.time()
            output = model(inputs=real_inputs)

            if hasattr(device, "synchronize"):
                device.synchronize()
            elif hasattr(ttnn, "synchronize"):
                ttnn.synchronize(device)

            end_pass = time.time()
            MODEL_STATS.append(
                {
                    "Module Name": "Full_Model_Graph",
                    "Phase": "Inference",
                    "Type": "End-to-End",
                    "Wall Time (s)": f"{end_pass - start_pass:.6f}",
                }
            )

            print(f"\n[SUCCESS] Pass completed in {end_pass - start_pass:.4f}s")
            if "action" in output:
                print(f"Action Prediction: {output['action'][0, 0, :5]}")

            save_stats_to_csv()
            # Per-op dispatch timings (TTNN vs Torch) for profiling hotspots
            try:
                from models.experimental.tt_symbiote.core.run_config import DispatchManager

                DispatchManager.save_stats_to_file("dispatch_timings.csv")
                print("[INFO] Dispatch timings (TTNN vs CPU per op) saved to: dispatch_timings.csv")
            except Exception as _e:
                pass

        except Exception as e:
            save_stats_to_csv("gr00t_crash_report.csv")
            print(f"\n[FAILURE] Error: {e}")
            raise e
