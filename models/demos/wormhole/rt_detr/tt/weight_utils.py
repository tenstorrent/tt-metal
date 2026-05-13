# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn

class Params:
    """Simple attribute container."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def fold_bn_to_conv(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    # keep everything float32 during fold to avoid precision loss
    conv_w = conv_w.float()
    conv_b = conv_b.float()
    bn_w   = bn_w.float()
    bn_b   = bn_b.float()
    bn_mean = bn_mean.float()
    bn_var  = bn_var.float()

    std    = torch.sqrt(bn_var + eps)
    scale  = bn_w / std
    fused_w = conv_w * scale.view(-1, 1, 1, 1)
    fused_b = (conv_b - bn_mean) * scale + bn_b
    return fused_w, fused_b  # return float32, convert to bf16 in _conv_params


def _to_tt(t, device, dtype=ttnn.bfloat16):
    """Upload to device — used for linear weights and norms only."""
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_tt_host(t, dtype=ttnn.bfloat16):
    """Keep on host without layout — ttnn.conv2d handles layout internally."""
    return ttnn.from_torch(t.contiguous(), dtype=dtype)


def _conv_params(w, b, device):
    # both weight and bias stay on host — ttnn.conv2d uploads and formats them internally
    return Params(
        weight=_to_tt_host(w, dtype=ttnn.bfloat16),
        bias=ttnn.from_torch(
            b.reshape(1, 1, 1, -1).contiguous(),
            dtype=ttnn.bfloat16,
        ),  # no device, no layout — let conv2d handle it
    )


def _linear_params(w, b, device):
    return Params(
        weight=_to_tt(w.T.contiguous(), device, dtype=ttnn.bfloat8_b),
        bias=_to_tt(b.reshape(1, 1, 1, -1), device, dtype=ttnn.bfloat16),
    )


def _norm_params(w, b, device):
    return Params(
        weight=_to_tt(w.reshape(1, 1, 1, -1), device, dtype=ttnn.bfloat16),
        bias=_to_tt(b.reshape(1, 1, 1, -1), device, dtype=ttnn.bfloat16),
    )



def _fold_conv_norm(sd, conv_key, norm_key, device):
    """Fold a conv.weight + norm.* pair from the state dict."""
    w      = sd[f"{conv_key}.weight"]
    b      = sd.get(f"{conv_key}.bias", torch.zeros(w.shape[0]))
    fused_w, fused_b = fold_bn_to_conv(
        w, b,
        sd[f"{norm_key}.weight"],
        sd[f"{norm_key}.bias"],
        sd[f"{norm_key}.running_mean"],
        sd[f"{norm_key}.running_var"],
    )
    return _conv_params(fused_w, fused_b, device)


# Backbone (PResNet-50)

def _bottleneck_params(sd, prefix, device, stage_idx, block_idx):
    block = Params(
        conv1=_fold_conv_norm(sd, f"{prefix}.branch2a.conv", f"{prefix}.branch2a.norm", device),
        conv2=_fold_conv_norm(sd, f"{prefix}.branch2b.conv", f"{prefix}.branch2b.norm", device),
        conv3=_fold_conv_norm(sd, f"{prefix}.branch2c.conv", f"{prefix}.branch2c.norm", device),
    )

    if block_idx == 0:
        if stage_idx == 0:
            # stage 0: .short.conv + .short.norm
            block.shortcut = _fold_conv_norm(
                sd, f"{prefix}.short.conv", f"{prefix}.short.norm", device
            )
        else:
            # stages 1-3: .short.conv.conv + .short.conv.norm
            block.shortcut = _fold_conv_norm(
                sd, f"{prefix}.short.conv.conv", f"{prefix}.short.conv.norm", device
            )

    return block


def get_backbone_parameters(model, device):
    sd = model.state_dict()

    stem = [
        _fold_conv_norm(sd, f"backbone.conv1.conv1_{i+1}.conv",
                            f"backbone.conv1.conv1_{i+1}.norm", device)
        for i in range(3)
    ]

    num_blocks = [3, 4, 6, 3]
    stages = []
    for si, nb in enumerate(num_blocks):
        blocks = []
        for bi in range(nb):
            prefix = f"backbone.res_layers.{si}.blocks.{bi}"
            blocks.append(_bottleneck_params(sd, prefix, device, si, bi))
        stages.append(blocks)

    sd = model.state_dict()
    backbone_keys = [k for k in sd.keys() if 'backbone' in k][:30]
    for k in backbone_keys:
        print(k)

    return Params(stem=stem, stages=stages)

# Encoder (HybridEncoder)

def _fold_conv_norm_enc(sd, base, device):
    """Fold encoder conv+norm where keys are base.conv.weight + base.norm.*"""
    return _fold_conv_norm(sd, f"{base}.conv", f"{base}.norm", device)


def _fold_input_proj(sd, idx, device):
    """encoder.input_proj.N is Sequential(Conv2d[.0], BN[.1])."""
    return _fold_conv_norm(sd, f"encoder.input_proj.{idx}.0",
                               f"encoder.input_proj.{idx}.1", device)


def _fpn_pan_block_params(sd, prefix, device):
    """CSPRepLayer: conv1, conv2, bottlenecks.N.conv1/conv2."""
    bns = []
    i = 0
    while f"{prefix}.bottlenecks.{i}.conv1.conv.weight" in sd:
        bns.append(Params(
            conv1=_fold_conv_norm_enc(sd, f"{prefix}.bottlenecks.{i}.conv1", device),
            conv2=_fold_conv_norm_enc(sd, f"{prefix}.bottlenecks.{i}.conv2", device),
        ))
        i += 1
    return Params(
        conv1=_fold_conv_norm_enc(sd, f"{prefix}.conv1", device),
        conv2=_fold_conv_norm_enc(sd, f"{prefix}.conv2", device),
        bottlenecks=bns,
    )


def get_encoder_parameters(model, device):
    sd = model.state_dict()

    input_proj = [_fold_input_proj(sd, i, device) for i in range(3)]

    # AIFI encoder layers
    enc_layers = []
    src_layers = model.encoder.encoder[0].layers
    for layer in src_layers:
        attn  = layer.self_attn
        d     = attn.embed_dim
        qkv_w = attn.in_proj_weight
        qkv_b = attn.in_proj_bias

        def _split(start, end):
            return _linear_params(qkv_w[start:end, :], qkv_b[start:end], device)

        enc_layers.append(Params(
            self_attn=Params(
                q=_split(0, d),
                k=_split(d, 2 * d),
                v=_split(2 * d, 3 * d),
                out_proj=_linear_params(
                    attn.out_proj.weight, attn.out_proj.bias, device
                ),
            ),
            linear1=_linear_params(layer.linear1.weight, layer.linear1.bias, device),
            linear2=_linear_params(layer.linear2.weight, layer.linear2.bias, device),
            norm1=_norm_params(layer.norm1.weight, layer.norm1.bias, device),
            norm2=_norm_params(layer.norm2.weight, layer.norm2.bias, device),
        ))

    lateral_convs = [
        _fold_conv_norm_enc(sd, f"encoder.lateral_convs.{i}", device)
        for i in range(2)
    ]

    fpn_blocks = [
        _fpn_pan_block_params(sd, f"encoder.fpn_blocks.{i}", device)
        for i in range(2)
    ]

    downsample_convs = [
        _fold_conv_norm_enc(sd, f"encoder.downsample_convs.{i}", device)
        for i in range(2)
    ]

    pan_blocks = [
        _fpn_pan_block_params(sd, f"encoder.pan_blocks.{i}", device)
        for i in range(2)
    ]

    return Params(
        input_proj=input_proj,
        encoder_layers=enc_layers,
        lateral_convs=lateral_convs,
        fpn_blocks=fpn_blocks,
        downsample_convs=downsample_convs,
        pan_blocks=pan_blocks,
    )

# Decoder (RTDETRTransformer)

def get_decoder_parameters(model, device):
    layers = []
    for layer in model.decoder.decoder.layers:
        attn  = layer.self_attn
        d     = attn.embed_dim
        qkv_w = attn.in_proj_weight
        qkv_b = attn.in_proj_bias

        def _split(start, end):
            return _linear_params(qkv_w[start:end, :], qkv_b[start:end], device)

        layers.append(Params(
            self_attn=Params(
                q=_split(0, d),
                k=_split(d, 2 * d),
                v=_split(2 * d, 3 * d),
                out_proj=_linear_params(
                    attn.out_proj.weight, attn.out_proj.bias, device
                ),
            ),
            linear1=_linear_params(layer.linear1.weight, layer.linear1.bias, device),
            linear2=_linear_params(layer.linear2.weight, layer.linear2.bias, device),
            norm1=_norm_params(layer.norm1.weight, layer.norm1.bias, device),
            norm2=_norm_params(layer.norm2.weight, layer.norm2.bias, device),
            norm3=_norm_params(layer.norm3.weight, layer.norm3.bias, device),
        ))
    return layers


def get_head_parameters(model, device):
    cls_layers = [
        _linear_params(lin.weight, lin.bias, device)
        for lin in model.decoder.dec_score_head
    ]

    bbox_layers = [
        [_linear_params(l.weight, l.bias, device) for l in mlp.layers]
        for mlp in model.decoder.dec_bbox_head
    ]

    return Params(class_embed=cls_layers, bbox_embed=bbox_layers)
