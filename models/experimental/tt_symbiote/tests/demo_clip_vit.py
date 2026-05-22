# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Demo: CLIP ViT -- PyTorch vs TTNN comparison."""

import time
import pytest
import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor
from transformers.activations import QuickGELUActivation
from transformers.models.clip.modeling_clip import CLIPAttention
from PIL import Image
import urllib.request

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.attention import (
    PytorchFusedQKVSelfAttention,
    SelfAttentionConfig,
    TTNNFusedQKVSelfAttention,
    TTNNSDPAAttention,
    TTNNSelfAttention,
)
from models.experimental.tt_symbiote.modules.activation import TTNNGelu
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


class TTNNCLIPAttention(TTNNSelfAttention):
    @classmethod
    def from_torch(cls, clip_attn: CLIPAttention):
        config = SelfAttentionConfig(
            hidden_size=clip_attn.embed_dim,
            num_attention_heads=clip_attn.num_heads,
        )
        new_attn = cls(attention_config=config)
        new_attn._fallback_torch_layer = clip_attn
        new_attn.query_key_value = TTNNFusedQKVSelfAttention.from_torch(
            PytorchFusedQKVSelfAttention(
                clip_attn.q_proj,
                clip_attn.k_proj,
                clip_attn.v_proj,
                clip_attn.num_heads,
                clip_attn.embed_dim,
            ),
        )
        new_attn.out_proj = TTNNLinear.from_torch(clip_attn.out_proj)
        new_attn.sdpa = TTNNSDPAAttention()
        for child in [new_attn.query_key_value, new_attn.out_proj, new_attn.sdpa]:
            child._bypass_tensor_wrapping = False
            child._bypass_explicitly_set = True
        return new_attn

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        result = super().forward(hidden_states, head_mask=None, output_attentions=False)
        context_layer = result[0]
        attn_output = self.out_proj(context_layer)
        return (attn_output, None)


def get_sample_inputs(processor):
    """Download a sample image and prepare inputs."""
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    urllib.request.urlretrieve(url, "/tmp/clip_demo_image.jpg")
    image = Image.open("/tmp/clip_demo_image.jpg")
    texts = ["a photo of a cat", "a photo of a dog", "two cats on a couch", "a car on a road"]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    return inputs, texts, image


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_demo_clip_vit(device):
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    inputs, texts, image = get_sample_inputs(processor)

    # ── 1. PyTorch baseline ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PyTorch (CPU) baseline")
    print("=" * 60)
    torch_model = CLIPModel.from_pretrained(model_id).to(torch.bfloat16).eval()
    torch.set_grad_enabled(False)

    torch_inputs = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"].to(torch.bfloat16),
        "attention_mask": inputs["attention_mask"],
    }

    # Warmup
    _ = torch_model(**torch_inputs)

    torch_start = time.time()
    n_runs = 5
    for _ in range(n_runs):
        torch_out = torch_model(**torch_inputs)
    torch_elapsed = time.time() - torch_start
    torch_avg = torch_elapsed / n_runs

    torch_image_embeds = torch_out.image_embeds.float()
    torch_text_embeds = torch_out.text_embeds.float()
    torch_logits = (torch_image_embeds @ torch_text_embeds.T).softmax(dim=-1)

    print(f"  Image embeds shape: {torch_image_embeds.shape}")
    print(f"  Text embeds shape:  {torch_text_embeds.shape}")
    print(f"  Avg latency ({n_runs} runs): {torch_avg*1000:.1f} ms")
    print(f"\n  Similarity scores (PyTorch):")
    for i, txt in enumerate(texts):
        print(f'    {torch_logits[0][i].item():.4f}  "{txt}"')

    # ── 2. TTNN via tt-symbiote ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TTNN (Tenstorrent) via tt-symbiote")
    print("=" * 60)
    ttnn_model = CLIPModel.from_pretrained(model_id).to(torch.bfloat16).eval()

    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.LayerNorm: TTNNLayerNorm,
        CLIPAttention: TTNNCLIPAttention,
        QuickGELUActivation: TTNNGelu,
    }
    register_module_replacement_dict(ttnn_model, nn_to_ttnn, model_config=None)
    set_device(ttnn_model, device)

    ttnn_inputs = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"].to(torch.bfloat16),
        "attention_mask": inputs["attention_mask"],
    }

    # Warmup (includes compilation)
    print("  Warmup (first run includes compilation)...")
    warmup_start = time.time()
    _ = ttnn_model(**ttnn_inputs)
    warmup_time = time.time() - warmup_start
    print(f"  Warmup time: {warmup_time*1000:.1f} ms")

    DispatchManager.clear_timings()
    ttnn_start = time.time()
    for _ in range(n_runs):
        ttnn_out = ttnn_model(**ttnn_inputs)
    ttnn_elapsed = time.time() - ttnn_start
    ttnn_avg = ttnn_elapsed / n_runs

    ttnn_image_embeds = ttnn_out.image_embeds
    ttnn_text_embeds = ttnn_out.text_embeds
    if hasattr(ttnn_image_embeds, "to_torch"):
        ttnn_image_embeds = ttnn_image_embeds.to_torch
    if hasattr(ttnn_text_embeds, "to_torch"):
        ttnn_text_embeds = ttnn_text_embeds.to_torch
    ttnn_image_embeds = ttnn_image_embeds.float()
    ttnn_text_embeds = ttnn_text_embeds.float()
    ttnn_logits = (ttnn_image_embeds @ ttnn_text_embeds.T).softmax(dim=-1)

    print(f"  Image embeds shape: {ttnn_image_embeds.shape}")
    print(f"  Text embeds shape:  {ttnn_text_embeds.shape}")
    print(f"  Avg latency ({n_runs} runs): {ttnn_avg*1000:.1f} ms")
    print(f"\n  Similarity scores (TTNN):")
    for i, txt in enumerate(texts):
        print(f'    {ttnn_logits[0][i].item():.4f}  "{txt}"')

    DispatchManager.save_stats_to_file("clip_vit_demo_stats.csv")

    # ── 3. Comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Comparison: PyTorch vs TTNN")
    print("=" * 60)

    # PCC (Pearson Correlation Coefficient)
    img_pcc = torch.corrcoef(torch.stack([torch_image_embeds.flatten(), ttnn_image_embeds.flatten()]))[0, 1].item()
    txt_pcc = torch.corrcoef(torch.stack([torch_text_embeds.flatten(), ttnn_text_embeds.flatten()]))[0, 1].item()

    # Max absolute difference
    img_max_diff = (torch_image_embeds - ttnn_image_embeds).abs().max().item()
    txt_max_diff = (torch_text_embeds - ttnn_text_embeds).abs().max().item()

    # Cosine similarity
    img_cos = torch.nn.functional.cosine_similarity(torch_image_embeds, ttnn_image_embeds).mean().item()
    txt_cos = torch.nn.functional.cosine_similarity(torch_text_embeds, ttnn_text_embeds).mean().item()

    print(f"  Image embeds:  PCC={img_pcc:.6f}  CosSim={img_cos:.6f}  MaxDiff={img_max_diff:.6f}")
    print(f"  Text embeds:   PCC={txt_pcc:.6f}  CosSim={txt_cos:.6f}  MaxDiff={txt_max_diff:.6f}")
    print(f"\n  Latency:")
    print(f"    PyTorch (CPU):  {torch_avg*1000:.1f} ms")
    print(f"    TTNN (device):  {ttnn_avg*1000:.1f} ms")
    print(f"    Speedup:        {torch_avg/ttnn_avg:.2f}x")

    print(f"\n  Similarity score comparison:")
    print(f"    {'Text':<30} {'PyTorch':>10} {'TTNN':>10} {'Match':>8}")
    print(f"    {'-'*30} {'-'*10} {'-'*10} {'-'*8}")
    for i, txt in enumerate(texts):
        pt_score = torch_logits[0][i].item()
        tt_score = ttnn_logits[0][i].item()
        match = abs(pt_score - tt_score) < 0.01
        print(f"    {txt:<30} {pt_score:>10.4f} {tt_score:>10.4f} {'OK' if match else 'DIFF':>8}")

    print("\n" + "=" * 60)
    assert img_pcc > 0.95, f"Image PCC too low: {img_pcc}"
    assert txt_pcc > 0.85, f"Text PCC too low: {txt_pcc}"
    print("  PASSED -- outputs match within tolerance")
    print("=" * 60)
