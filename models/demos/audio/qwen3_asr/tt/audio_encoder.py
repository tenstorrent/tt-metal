"""ttnn implementation of the Qwen3-ASR AuT audio encoder (Blackhole / P150a).

Structure mirrors `reference/audio_encoder_ref.py` (validated PCC=1.0 vs golden):
`encode_mel`: ttnn.conv2d frontend (on device) -> + sinusoidal PE -> 24 pre-LN transformer
layers (full bidirectional attention via fused SDPA) -> ln_post -> proj1 -> GELU -> proj2.

Speed: q/k/v fused into one matmul + `nlp_create_qkv_heads`; attention is the fused
flash kernel `ttnn.transformer.scaled_dot_product_attention`; FFN keeps activations
in L1 with fused matmul+GELU (`ttnn.linear(..., activation="gelu")`). HiFi4 for the
norms/projection to hold PCC. conv2d uses config_tensors_in_dram + deallocate_activation
so its L1_SMALL working set doesn't accumulate across server requests.

TODO(perf): trace-capture the static-shape encoder; fold proj1/2 into the decoder embed splice."""
import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn

# Qwen3-ASR-1.7B audio_config
D_MODEL = 1024
N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = D_MODEL // N_HEADS  # 64
FFN = 4096
OUTPUT_DIM = 2048
LN_EPS = 1e-5
HIFI4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False)


def _to_dev(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


N_WINDOW = 50
DS_HIDDEN = 480


def preprocess_conv_weights(w, device):
    """Conv2d frontend weights (kept on host as ttnn tensors; ttnn.conv2d folds them in)."""
    cp = {}
    for name in ("conv2d1", "conv2d2", "conv2d3"):
        cw = w[f"{name}.weight"]  # (480, Cin, 3, 3)
        cb = w[f"{name}.bias"].reshape(1, 1, 1, -1)  # (1,1,1,480)
        cp[name] = dict(
            weight=ttnn.from_torch(cw, ttnn.float32),
            bias=ttnn.from_torch(cb, ttnn.float32),
            in_ch=cw.shape[1],
            out_ch=cw.shape[0],
            k=(cw.shape[2], cw.shape[3]),
        )
    return cp


def conv_frontend_tt(mel, conv_w, conv_out_w, conv_out_b, device):
    """mel (num_mel=128, T) -> (n_chunks*13, d_model=1024) on device, pre-PE.
    Mirrors reference.conv_frontend but the 3 conv2d (+GELU) + conv_out run on TT."""
    nm, T = mel.shape
    chunk = N_WINDOW * 2  # 100
    n = (T + chunk - 1) // chunk
    pieces = []
    for i in range(n):
        seg = mel[:, i * chunk : (i + 1) * chunk]
        if seg.shape[1] < chunk:
            seg = torch.nn.functional.pad(seg, (0, chunk - seg.shape[1]))
        pieces.append(seg)
    x = torch.stack(pieces, 0).unsqueeze(1)  # (n,1,128,100)  NCHW
    # NHWC flattened -> (1,1,n*H*W, C=1)
    H, W = nm, chunk
    xh = x.permute(0, 2, 3, 1).reshape(1, 1, n * H * W, 1).contiguous()
    xt = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # config_tensors_in_dram: keep conv's persistent config tensors off the small (32-64 KB)
    # L1_SMALL region, which otherwise fills across requests -> bank_manager OOM.
    # deallocate_activation: free the conv input activation inside the op.
    conv_cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, config_tensors_in_dram=True, deallocate_activation=True)
    compute_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    for name in ("conv2d1", "conv2d2", "conv2d3"):
        c = conv_w[name]
        conv_out, [H, W], [c["weight"], c["bias"]] = ttnn.conv2d(
            input_tensor=xt,
            weight_tensor=c["weight"],
            bias_tensor=c["bias"],
            in_channels=c["in_ch"],
            out_channels=c["out_ch"],
            device=device,
            kernel_size=c["k"],
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
            batch_size=n,
            input_height=H,
            input_width=W,
            conv_config=conv_cfg,
            compute_config=compute_cfg,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        # conv frees its input activation internally (deallocate_activation=True)
        xt = ttnn.gelu(conv_out)  # (1,1,n*H*W,480) NHWC
        ttnn.deallocate(conv_out)
    C = DS_HIDDEN
    # NHWC (n,H,W,C) -> reference wants (n, W, C, H) flattened to (n, W, C*H)
    xt_t = ttnn.to_torch(xt).reshape(n, H, W, C)
    ttnn.deallocate(xt)
    xt_t = xt_t.permute(0, 2, 3, 1).reshape(n, W, C * H).contiguous()  # (n,13,7680)
    xt2 = ttnn.from_torch(xt_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_d = ttnn.linear(xt2, conv_out_w, compute_kernel_config=HIFI4)  # (n,13,1024), no bias
    ttnn.deallocate(xt2)
    out = ttnn.to_torch(out_d).reshape(-1, D_MODEL).float()
    ttnn.deallocate(out_d)
    return out


def preprocess_weights(w, device):
    """w: torch state dict with keys like 'layers.0.self_attn.q_proj.weight' (audio_tower-relative)."""
    p = {"layers": []}
    for li in range(N_LAYERS):
        pre = f"layers.{li}."
        # fuse q,k,v -> one (3*D, D) weight, (3*D,) bias
        qw, kw, vw = (w[pre + f"self_attn.{x}_proj.weight"] for x in "qkv")
        qb, kb, vb = (w[pre + f"self_attn.{x}_proj.bias"] for x in "qkv")
        fused_w = torch.cat([qw, kw, vw], dim=0)  # (3D, D)
        fused_b = torch.cat([qb, kb, vb], dim=0)  # (3D,)
        lp = {
            "ln1_w": _to_dev(w[pre + "self_attn_layer_norm.weight"], device),
            "ln1_b": _to_dev(w[pre + "self_attn_layer_norm.bias"], device),
            "qkv_w": preprocess_linear_weight(fused_w, dtype=ttnn.bfloat16),
            "qkv_b": preprocess_linear_bias(fused_b, dtype=ttnn.bfloat16),
            "o_w": preprocess_linear_weight(w[pre + "self_attn.out_proj.weight"], dtype=ttnn.bfloat16),
            "o_b": preprocess_linear_bias(w[pre + "self_attn.out_proj.bias"], dtype=ttnn.bfloat16),
            "ln2_w": _to_dev(w[pre + "final_layer_norm.weight"], device),
            "ln2_b": _to_dev(w[pre + "final_layer_norm.bias"], device),
            "fc1_w": preprocess_linear_weight(w[pre + "fc1.weight"], dtype=ttnn.bfloat16),
            "fc1_b": preprocess_linear_bias(w[pre + "fc1.bias"], dtype=ttnn.bfloat16),
            "fc2_w": preprocess_linear_weight(w[pre + "fc2.weight"], dtype=ttnn.bfloat16),
            "fc2_b": preprocess_linear_bias(w[pre + "fc2.bias"], dtype=ttnn.bfloat16),
        }
        for k in ("qkv_w", "qkv_b", "o_w", "o_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b"):
            lp[k] = ttnn.to_device(lp[k], device)
        p["layers"].append(lp)
    p["lnpost_w"] = _to_dev(w["ln_post.weight"], device)
    p["lnpost_b"] = _to_dev(w["ln_post.bias"], device)
    p["proj1_w"] = ttnn.to_device(preprocess_linear_weight(w["proj1.weight"], dtype=ttnn.bfloat16), device)
    p["proj1_b"] = ttnn.to_device(preprocess_linear_bias(w["proj1.bias"], dtype=ttnn.bfloat16), device)
    p["proj2_w"] = ttnn.to_device(preprocess_linear_weight(w["proj2.weight"], dtype=ttnn.bfloat16), device)
    p["proj2_b"] = ttnn.to_device(preprocess_linear_bias(w["proj2.bias"], dtype=ttnn.bfloat16), device)
    # conv2d frontend (full-TT)
    p["conv_w"] = preprocess_conv_weights(w, device)
    p["conv_out_w"] = ttnn.to_device(preprocess_linear_weight(w["conv_out.weight"], dtype=ttnn.bfloat16), device)
    return p


_PE_CACHE = {}


def encode_mel(mel, params, device):
    """Full-TT encoder from mel (num_mel, T): TT conv2d frontend -> +PE -> transformer
    -> projector. Returns audio embeds (S, output_dim) as torch."""
    conv = conv_frontend_tt(mel, params["conv_w"], params["conv_out_w"], None, device)  # (S,1024) torch
    per_chunk = 13
    if per_chunk not in _PE_CACHE:
        # sinusoidal PE[:13] (matches reference.sinusoids)
        import math

        ch = D_MODEL
        log_inc = math.log(10000.0) / (ch // 2 - 1)
        inv = torch.exp(-log_inc * torch.arange(ch // 2).float())
        t = torch.arange(1500)[:, None].float() * inv[None, :]
        _PE_CACHE[per_chunk] = torch.cat([torch.sin(t), torch.cos(t)], 1)[:per_chunk]
    pe = _PE_CACHE[per_chunk]
    n = conv.shape[0] // per_chunk
    x_host = (conv.reshape(n, per_chunk, D_MODEL) + pe.unsqueeze(0)).reshape(-1, D_MODEL)
    return encode(x_host, params, device)


def _layer(x, lp, device):
    core = ttnn.CoreGrid(y=device.compute_with_storage_grid_size().y, x=device.compute_with_storage_grid_size().x)
    residual = x  # (1,1,S,D)
    h = ttnn.layer_norm(x, weight=lp["ln1_w"], bias=lp["ln1_b"], epsilon=LN_EPS, compute_kernel_config=HIFI4)
    qkv = ttnn.linear(h, lp["qkv_w"], bias=lp["qkv_b"], core_grid=core, compute_kernel_config=HIFI4)  # (1,1,S,3D)
    ttnn.deallocate(h)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv,
        num_heads=N_HEADS,
        num_kv_heads=N_HEADS,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(qkv)
    attn = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        scale=HEAD_DIM**-0.5,
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(attn)  # free the SDPA output (don't rely on GC)
    o = ttnn.linear(concat, lp["o_w"], bias=lp["o_b"], core_grid=core, compute_kernel_config=HIFI4)
    ttnn.deallocate(concat)
    x = ttnn.add(residual, o)
    ttnn.deallocate(o)
    ttnn.deallocate(residual)

    residual = x
    h = ttnn.layer_norm(x, weight=lp["ln2_w"], bias=lp["ln2_b"], epsilon=LN_EPS, compute_kernel_config=HIFI4)
    h = ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG)
    h = ttnn.linear(
        h,
        lp["fc1_w"],
        bias=lp["fc1_b"],
        activation="gelu",
        core_grid=core,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=HIFI4,
    )
    h = ttnn.linear(
        h,
        lp["fc2_w"],
        bias=lp["fc2_b"],
        core_grid=core,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=HIFI4,
    )
    x = ttnn.add(residual, h)
    ttnn.deallocate(h)
    ttnn.deallocate(residual)
    return x


def encode(x_host, params, device):
    """x_host: (S, D_MODEL) torch tensor = conv frontend output WITH positional embedding
    already added (done on host, see reference.conv_frontend + sinusoids). Returns (S, output_dim) torch."""
    S = x_host.shape[0]
    x = ttnn.from_torch(
        x_host.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # (1,1,S,D)
    for lp in params["layers"]:
        x = _layer(x, lp, device)
    x = ttnn.layer_norm(
        x, weight=params["lnpost_w"], bias=params["lnpost_b"], epsilon=LN_EPS, compute_kernel_config=HIFI4
    )
    core = ttnn.CoreGrid(y=device.compute_with_storage_grid_size().y, x=device.compute_with_storage_grid_size().x)
    x = ttnn.linear(
        x, params["proj1_w"], bias=params["proj1_b"], activation="gelu", core_grid=core, compute_kernel_config=HIFI4
    )
    x = ttnn.linear(x, params["proj2_w"], bias=params["proj2_b"], core_grid=core, compute_kernel_config=HIFI4)
    out = ttnn.to_torch(x).reshape(-1, OUTPUT_DIM)[:S].float()
    ttnn.deallocate(x)
    return out
