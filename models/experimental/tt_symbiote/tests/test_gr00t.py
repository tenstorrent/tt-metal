# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import time
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.attention import TTNNGR00TSelfAttention
from modules.linear import TTNNLinear
from modules.normalization import TTNNLayerNorm
from modules.activation import TTNNSilu
from modules.conv import TTNNConv2dNHWC
from utils.device_management import set_device
from utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.core.run_config import DispatchManager

try:
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
except Exception:
    _groot_dir = PROJECT_ROOT / "groot"
    if not _groot_dir.exists():
        print("groot not found. Cloning https://github.com/pandeashwary/Gr00t.git into tt_symbiote/groot ...")
        _r = subprocess.run(
            ["git", "clone", "https://github.com/pandeashwary/Gr00t.git", str(_groot_dir)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if _r.returncode != 0:
            print(f"Clone failed: {_r.stderr or _r.stdout}")
            exit(1)
        sys.path.insert(0, str(_groot_dir))
        try:
            from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
        except Exception as _e:
            print(f"groot import failed after clone: {_e}")
            exit(1)
    else:
        _groot_str = str(_groot_dir)
        if _groot_str not in sys.path:
            sys.path.insert(0, _groot_str)
        try:
            from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
        except Exception as _e:
            print(
                "groot import failed. Make sure you have https://github.com/pandeashwary/Gr00t.git "
                "in models/experimental/tt_symbiote/groot (or set PYTHONPATH)."
            )
            print(f"Error: {_e}")
            exit(1)


def _patched_beta_sample(self, sample_shape=torch.Size()):
    d = Beta(self.concentration0.float(), self.concentration1.float())
    return d.rsample(sample_shape).to(self.concentration0.dtype)


Beta.sample = _patched_beta_sample
PretrainedConfig._attn_implementation_autoset = False
PretrainedConfig._attn_implementation_internal = "eager"


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


def _sync(device):
    if hasattr(device, "synchronize"):
        device.synchronize()
    elif hasattr(ttnn, "synchronize"):
        ttnn.synchronize(device)


def _apply_speed_optimizations():
    os.environ.setdefault("TT_SYMBIOTE_MAX_LENGTH", "72")
    os.environ.setdefault("TT_SYMBIOTE_ACTION_HORIZON", "1")
    os.environ.setdefault("TT_SYMBIOTE_UNBIND_FALLBACK_THRESHOLD", "1")
    os.environ.setdefault("TT_SYMBIOTE_SPLIT_FALLBACK_THRESHOLD", "1")
    os.environ.setdefault("TT_SYMBIOTE_NUM_INFERENCE_TIMESTEPS", "1")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
@pytest.mark.filterwarnings("ignore:Accessing config attribute.*AlternateVLDiT.*:FutureWarning")
def test_gr00t_inference_validation(device):
    torch.manual_seed(42)
    _apply_speed_optimizations()

    from functools import reduce
    import operator

    padded = torch.randn(2965872, dtype=torch.bfloat16)
    shape = (1, 1152, 196, 4)
    target_numel = reduce(operator.mul, shape, 1)
    assert target_numel == 903168 and padded.numel() > target_numel

    class ViewFunc:
        def name(self):
            return "aten::view"

    result = DispatchManager.dispatch_to_torch_wrapper(ViewFunc(), (padded, shape), {})
    out = result.to_torch if hasattr(result, "to_torch") else result
    assert out.shape == shape and out.numel() == target_numel

    model_id = "nvidia/GR00T-N1.6-3B"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    model = Gr00tN1d6.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=True).eval()

    patched = 0
    for _name, mod in model.named_modules():
        if hasattr(mod, "get_spatial_shapes") and hasattr(mod, "patch_size"):
            m = mod
            mod.get_spatial_shapes = lambda bchw_list, _m=m: torch.tensor(
                [
                    (h // _m.patch_size, w // _m.patch_size)
                    for shape in bchw_list
                    for (b, h, w) in (
                        [(shape[0], shape[2], shape[3])] if len(shape) == 4 else [(1, shape[1], shape[2])]
                    )
                    for _ in range(b)
                ]
            )
            patched += 1
    if patched:
        print(f"[gr00ty] Patched get_spatial_shapes on {patched} vision embeddings module(s).")

    try:
        env_steps = os.environ.get("TT_SYMBIOTE_NUM_INFERENCE_TIMESTEPS")
        if env_steps is not None:
            n = int(env_steps)
            model.action_head.num_inference_timesteps = max(1, n)
    except (TypeError, ValueError):
        pass

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
    exclude_replacement = set()
    transformed = register_module_replacement_dict(
        model, op_mapping, model_config={"dtype": ttnn.bfloat16}, exclude_replacement=exclude_replacement
    )
    set_device(model, device)

    for name, module in tqdm(transformed.items(), desc="Assimilating Weights"):
        if hasattr(module, "preprocess_weights"):
            t_start = time.time()
            module.preprocess_weights()
            module.move_weights_to_device()
            t_end = time.time()
            DispatchManager.record_timing("Assimilation", name, module.__class__.__name__, {}, t_end - t_start)
    _sync(device)

    max_len = int(os.environ.get("TT_SYMBIOTE_MAX_LENGTH", "256"))
    action_len = int(os.environ.get("TT_SYMBIOTE_ACTION_HORIZON", str(model.config.action_horizon)))
    image_token_index = model.backbone.model.config.image_token_index
    num_image_tokens = 64
    text_max_len = max_len - num_image_tokens
    text_data = tokenizer(
        "Pick up the red block.",
        return_tensors="pt",
        padding="max_length",
        max_length=text_max_len,
        truncation=True,
    )
    ids = text_data["input_ids"]
    image_placeholder = torch.full((1, num_image_tokens), image_token_index, dtype=ids.dtype, device=ids.device)
    input_ids = torch.cat([ids[:, :1], image_placeholder, ids[:, 1:]], dim=1)
    attention_mask = torch.ones((1, max_len), dtype=torch.long, device=ids.device)

    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    pixel_values = torch.from_numpy(np.array(raw_image.resize((224, 224)))).permute(2, 0, 1).float()
    pixel_values = (pixel_values / 255.0).unsqueeze(0).to(torch.bfloat16)

    real_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": [pixel_values],
        "embodiment_id": torch.zeros((1,), dtype=torch.long),
        "state": torch.zeros(1, 1, model.config.max_state_dim, dtype=torch.bfloat16),
        "action": torch.zeros(1, action_len, model.config.max_action_dim, dtype=torch.bfloat16),
        "action_mask": torch.ones((1, action_len, model.config.max_action_dim), dtype=torch.bfloat16),
        "velocity": torch.zeros(1, action_len, model.config.max_action_dim, dtype=torch.bfloat16),
    }

    print("\n[RUNNING] Hardware Inference Pass...")
    with torch.inference_mode():
        _ = model(inputs=real_inputs)
        _sync(device)
        _ = model(inputs=real_inputs)
        _sync(device)

        start_pass = time.time()
        output = model(inputs=real_inputs)
        _sync(device)
        end_pass = time.time()
        final_inference_s = end_pass - start_pass
        DispatchManager.record_timing("Inference", "Full_Inference_Graph", "End-to-End", {}, final_inference_s)

        backbone_in, action_in = model.prepare_input(real_inputs)

        t0 = time.time()
        backbone_out = model.backbone(backbone_in)
        _sync(device)
        t_backbone = time.time() - t0

        t0 = time.time()
        _ = model.action_head(backbone_out, action_in)
        _sync(device)
        t_action = time.time() - t0

        print(f"[PROFILE] Backbone_Total: {t_backbone:.6f}s  Action_Head_Total: {t_action:.6f}s")
        print(f"[PROFILE] Final inference time (End-to-End): {final_inference_s:.6f}s")
    DispatchManager.save_stats_to_file("gr00t_full_perf_report.csv")
