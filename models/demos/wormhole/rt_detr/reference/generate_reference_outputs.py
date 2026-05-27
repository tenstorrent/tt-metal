# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RT-DETR", "rtdetr_pytorch"))
from src.core import YAMLConfig

config_path = "RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
ckpt_path = "weights/rtdetr_r50vd.pth"
img_path = "demo/demo_images/sample.jpg"
out_path = "reference/reference_outputs.pt"


def load_image(path, size=(640, 640)):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    tf = T.Compose([T.Resize(size), T.ToTensor()])
    return tf(img).unsqueeze(0), torch.tensor([[orig_h, orig_w]], dtype=torch.float32)


def main():
    print("Loading model...")
    cfg = YAMLConfig(config_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state)
    model = cfg.model.eval()

    saved = {}
    hooks = []

    # backbone feature maps
    # res_layers[1] -> s3 (stride 8,  80x80)
    # res_layers[2] -> s4 (stride 16, 40x40)
    # res_layers[3] -> s5 (stride 32, 20x20)
    for stage_idx, key in [(1, "backbone_s3"), (2, "backbone_s4"), (3, "backbone_s5")]:

        def _make_hook(k):
            def _fn(m, inp, out):
                saved[k] = out.detach().clone()

            return _fn

        hooks.append(model.backbone.res_layers[stage_idx].register_forward_hook(_make_hook(key)))

    # encoder outputs (list [p3, p4, p5])
    def _enc_hook(m, inp, out):
        saved["encoder_p3"] = out[0].detach().clone()
        saved["encoder_p4"] = out[1].detach().clone()
        saved["encoder_p5"] = out[2].detach().clone()

    hooks.append(model.encoder.register_forward_hook(_enc_hook))

    # decoder hidden states
    _layer_outputs = []

    def _make_layer_hook(layer_idx):
        def _fn(m, inp, out):
            # inp[0] is tgt (the query tensor going into this layer)
            if layer_idx == 0 and "decoder_init_query" not in saved:
                saved["decoder_init_query"] = inp[0].detach().clone().unsqueeze(1)
            # out is the updated tgt after self-attn + cross-attn + FFN
            _layer_outputs.append((layer_idx, out.detach().clone()))

        return _fn

    for li, layer in enumerate(model.decoder.decoder.layers):
        hooks.append(layer.register_forward_hook(_make_layer_hook(li)))

    # AIFI input/output and pos embed
    orig_aifi_fwd = model.encoder.encoder[0].forward
    _aifi_done = [False]

    def _patched_aifi(src, pos_embed=None):
        if not _aifi_done[0]:
            _aifi_done[0] = True
            saved["aifi_input"] = src.detach().clone().unsqueeze(1)
            if pos_embed is not None:
                saved["aifi_pos_embed"] = pos_embed.detach().clone().unsqueeze(1)
            with torch.no_grad():
                out_no_pos = orig_aifi_fwd(src, pos_embed=None)
            saved["aifi_output_no_pos"] = out_no_pos.detach().clone().unsqueeze(1)
        result = orig_aifi_fwd(src, pos_embed=pos_embed)
        if "aifi_output_with_pos" not in saved:
            saved["aifi_output_with_pos"] = result.detach().clone().unsqueeze(1)
        return result

    model.encoder.encoder[0].forward = _patched_aifi

    # attention unit test weights and I/O
    aifi_attn = model.encoder.encoder[0].layers[0].self_attn
    d = aifi_attn.embed_dim

    def _attn_hook(m, inp, out):
        if "attn_input" not in saved:
            saved["attn_input"] = inp[0].detach().clone()
            saved["attn_self_output"] = out[0].detach().clone()
            saved["attn_num_heads"] = torch.tensor(m.num_heads)
            qkv_w = m.in_proj_weight.detach()
            qkv_b = m.in_proj_bias.detach()
            saved["attn_q_weight"] = qkv_w[:d, :]
            saved["attn_q_bias"] = qkv_b[:d]
            saved["attn_k_weight"] = qkv_w[d : 2 * d, :]
            saved["attn_k_bias"] = qkv_b[d : 2 * d]
            saved["attn_v_weight"] = qkv_w[2 * d :, :]
            saved["attn_v_bias"] = qkv_b[2 * d :]
            saved["attn_out_weight"] = m.out_proj.weight.detach()
            saved["attn_out_bias"] = m.out_proj.bias.detach()

    hooks.append(aifi_attn.register_forward_hook(_attn_hook))

    # cross-attention input/output
    dec_layer0_ca = model.decoder.decoder.layers[0].cross_attn

    def _ca_hook(m, inp, out):
        if "attn_encoder_out" not in saved:
            saved["attn_encoder_out"] = inp[2].detach().clone()  # value = encoder memory
            saved["attn_cross_output"] = out.detach().clone()

    hooks.append(dec_layer0_ca.register_forward_hook(_ca_hook))

    # decoder init query and query_pos

    # also capture query_pos from RTDETRTransformer.forward
    orig_rtdetr_dec_fwd = model.decoder.forward

    def _patched_rtdetr_dec(*args, **kwargs):
        # hook query_pos_head to capture query positional embeddings
        orig_qph = model.decoder.query_pos_head.forward

        def _qph_hook(x):
            result = orig_qph(x)
            if "decoder_query_pos" not in saved:
                saved["decoder_query_pos"] = result.detach().clone().unsqueeze(1)
            return result

        model.decoder.query_pos_head.forward = _qph_hook
        out = orig_rtdetr_dec_fwd(*args, **kwargs)
        model.decoder.query_pos_head.forward = orig_qph
        return out

    model.decoder.forward = _patched_rtdetr_dec

    # forward
    img_tensor, orig_size = load_image(img_path)
    saved["backbone_input"] = img_tensor.detach().clone()

    print("Running forward pass...")
    with torch.no_grad():
        out = model(img_tensor)

    print("\nReference output stats:")
    for k in ["backbone_s3", "backbone_s4", "backbone_s5"]:
        t = saved[k]
        print(f"  {k}: shape={tuple(t.shape)} min={t.min():.3f} max={t.max():.3f} mean={t.mean():.3f}")

    # save collected decoder layer hidden states
    # eval mode only runs to eval_idx so _layer_outputs may have fewer than 6 entries
    for layer_idx, hidden in _layer_outputs:
        saved[f"decoder_layer{layer_idx+1}_output"] = hidden.unsqueeze(1)  # (B,1,nq,hidden)
    if _layer_outputs:
        # final layer output = last captured hidden state
        saved["decoder_output"] = _layer_outputs[-1][1].unsqueeze(1)

    # final postprocessed outputs
    postprocessor = cfg.postprocessor.deploy()
    labels, boxes, scores = postprocessor(out, orig_size)
    saved["labels"] = labels[0].detach().clone()
    saved["boxes"] = boxes[0].detach().clone()
    saved["scores"] = scores[0].detach().clone()

    for h in hooks:
        h.remove()

    print(f"\nSaving to {out_path}")
    torch.save(saved, out_path)

    print("\nSaved keys:")
    for k, v in saved.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else v
        print(f"  {k:40s} {shape}")


if __name__ == "__main__":
    main()
