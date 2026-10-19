# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prove that the in-tree Kimi-K3 KDA torch reference (which the ttKDA prefill kernels are
validated against) computes the same function as GLM-5.3-Flash's HF ``Glm5NextTextLinearAttention``
with real layer weights. Prefill over T tokens with zero initial state, then one decode token
continuing from the returned state; compares outputs and recurrent/conv states.

Run (tt-metal venv, PYTHONPATH=<checkout>):
    python models/demos/glm53_flash/reference/check_kda_reference.py --layer 0 --seq-len 96
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import default_snapshot_dir, load_layer_raw, shard_headers  # noqa: E402
from hf_layer_reference import STRICT_FP32, convert_to_hf_module_keys, load_config  # noqa: E402

from models.demos.deepseek_v3_d_p.reference.kda import KDAConfig, kda_forward_reference


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=96)
    args = ap.parse_args()
    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, args.layer, headers, name_filter=lambda n: n.startswith("self_attn."))
    attn_raw = {k[len("self_attn.") :]: v for k, v in raw.items()}

    # --- HF module with from_pretrained dtypes
    from transformers import DynamicCache
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention

    sd = {k[len("self_attn.") :]: v for k, v in convert_to_hf_module_keys(raw).items()}
    with torch.device("meta"):
        hf = Glm5NextTextLinearAttention(text, args.layer)
    hf = hf.to_empty(device="cpu")
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    hf = hf.to(torch.bfloat16)
    for name, p in list(hf.named_parameters()) + list(hf.named_buffers()):
        if any(s in name for s in STRICT_FP32):
            p.data = p.data.to(torch.float32)
    hf.eval()

    # --- in-tree Kimi KDA reference
    import json

    kcfg = KDAConfig.from_model_config(json.loads((snapshot / "config.json").read_text()))
    print("KDAConfig", kcfg)
    from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict

    attn_raw = normalize_kda_state_dict(attn_raw, kcfg)

    torch.manual_seed(0)
    T = args.seq_len
    # the attention input is RMSNorm(pre-mixed streams): unit-ish scale
    x = (torch.randn(1, T + 1, text.hidden_size) * 0.8).to(torch.bfloat16)
    mask = torch.ones(1, T, dtype=torch.bool)
    cache = DynamicCache(config=text)
    with torch.no_grad():
        y_hf = hf(x[:, :T], cache_params=cache, attention_mask=mask)
        y_ref, st = kda_forward_reference(x[:, :T], attn_raw, kcfg)
        layer = cache.layers[args.layer]
        hf_rec = layer.recurrent_states[0]
        hf_conv = layer.conv_states[0]  # [1, 3*qkv, kernel] last kernel inputs
        print(f"prefill T={T}: pcc(out) {pcc(y_hf, y_ref):.6f}  max|d| {(y_hf.float()-y_ref.float()).abs().max():.4e}")
        print(f"  recurrent state pcc {pcc(hf_rec, st.recurrent):.6f}")
        conv_ref = torch.cat([st.q_convolution, st.k_convolution, st.v_convolution], dim=-1)  # [1, 3, 3*qkv]
        conv_hf = hf_conv[:, :, -3:].transpose(1, 2)  # [1, 3, 3*qkv]
        print(f"  conv history pcc {pcc(conv_hf, conv_ref):.6f}")

        # decode step from the carried state
        y_hf_d = hf(x[:, T : T + 1], cache_params=cache, attention_mask=torch.ones(1, 1, dtype=torch.bool))
        y_ref_d, st2 = kda_forward_reference(x[:, T : T + 1], attn_raw, kcfg, st)
        print(
            f"decode step: pcc(out) {pcc(y_hf_d, y_ref_d):.6f}  max|d| {(y_hf_d.float()-y_ref_d.float()).abs().max():.4e}"
        )
        print(f"  recurrent state pcc {pcc(cache.layers[args.layer].recurrent_states[0], st2.recurrent):.6f}")


if __name__ == "__main__":
    main()
