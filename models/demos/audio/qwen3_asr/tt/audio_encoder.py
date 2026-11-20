# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn implementation of the Qwen3-ASR AuT audio encoder (Blackhole / P150a).

Structure mirrors `reference/audio_encoder_ref.py` (validated PCC=1.0 vs golden):
`encode_mel`: ttnn.conv2d frontend (on device) -> + sinusoidal PE -> 24 pre-LN transformer
layers (bidirectional attention within 104-token windows, as vLLM serves the model) -> ln_post
-> proj1 -> GELU -> proj2.

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


def _mesh_mapper(device):
    n = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    return ttnn.ReplicateTensorToMesh(device) if n > 1 else None


def _from_torch(t, device, **kwargs):
    mapper = _mesh_mapper(device)
    if mapper is not None:
        kwargs.setdefault("mesh_mapper", mapper)
    return ttnn.from_torch(t, device=device, **kwargs)


def _to_torch(t):
    shards = ttnn.get_device_tensors(t)
    if len(shards) > 1:
        return ttnn.to_torch(shards[0])
    return ttnn.to_torch(t)


def _to_dev(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return _from_torch(t, device, dtype=dtype, layout=layout)


N_WINDOW = 50
N_WINDOW_INFER = 800
TOKENS_PER_CHUNK = 13
# Attention window in tokens: n_window_infer mel frames = 8 chunks.
WINDOW_TOKENS = TOKENS_PER_CHUNK * (N_WINDOW_INFER // (N_WINDOW * 2))  # 104
DS_HIDDEN = 480


def feat_out_len(frames: int) -> int:
    """Audio tokens the model keeps for `frames` mel frames (the partial last chunk keeps fewer than 13)."""
    leave = int(frames) % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (int(frames) // 100) * TOKENS_PER_CHUNK


def window_bounds(n_valid: int, n_total: int) -> list:
    """cu_window_seqlens for `n_valid` real tokens padded to `n_total`.

    Real tokens attend within 104-token windows. Padded tokens form their own last window,
    so real tokens never attend to them and they stay finite."""
    bounds = list(range(0, n_valid, WINDOW_TOKENS)) + [n_valid]
    if n_total > n_valid:
        bounds.append(n_total)
    return bounds


def _fixed_window_bounds(n_valid: int, n_total: int) -> list:
    """window_bounds padded with empty windows to one length per n_total, so every clip
    in a bucket runs the same SDPA program."""
    bounds = window_bounds(n_valid, n_total)
    return bounds + [n_total] * (n_total // WINDOW_TOKENS + 2 - len(bounds))


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


# Output bytes one conv2d slice may hold in L1. Left to ttnn, the slice count follows the
# L1 free at call time, so the same clip shape can build new programs once other device
# buffers are live; a count fixed per shape keeps the programs warmup built.
CONV_L1_OUTPUT_BUDGET = 2 * 1024 * 1024


def _conv_slice_config(out_bytes, out_height):
    num_slices = min(max(1, -(-out_bytes // CONV_L1_OUTPUT_BUDGET)), out_height)
    return ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=num_slices)


def conv_frontend_tt(mel, conv_w, conv_out_w, conv_out_b, device, n=None, chunk=N_WINDOW * 2):
    """mel (num_mel=128, T) -> (n, tokens_per_chunk, d_model=1024) on device, pre-PE.
    Mirrors reference.conv_frontend but the 3 conv2d (+GELU) + conv_out run on TT.
    Chunks are `chunk` mel frames wide (13 tokens at 100). `n` >= ceil(T / chunk) chunks;
    chunks past the mel are all-zero."""
    nm, T = mel.shape
    n = (T + chunk - 1) // chunk if n is None else n
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
    xt = _from_torch(xh, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    # config_tensors_in_dram: keep conv's persistent config tensors off the small (32-64 KB)
    # L1_SMALL region, which otherwise fills across requests -> bank_manager OOM.
    # deallocate_activation: free the conv input activation inside the op.
    conv_cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, config_tensors_in_dram=True, deallocate_activation=True)
    compute_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    for name in ("conv2d1", "conv2d2", "conv2d3"):
        c = conv_w[name]
        # conv2d returns weights prepared for this batch size's sharding; reusing them at
        # another batch size fails, so keep one prepared set per input shape.
        prepared = c.setdefault("prepared", {}).get((n, chunk), (c["weight"], c["bias"]))
        out_h = (H + 1) // 2
        out_bytes = n * out_h * ((W + 1) // 2) * c["out_ch"] * 2
        conv_out, [H, W], c["prepared"][(n, chunk)] = ttnn.conv2d(
            input_tensor=xt,
            weight_tensor=prepared[0],
            bias_tensor=prepared[1],
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
            slice_config=_conv_slice_config(out_bytes, out_h),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        # conv frees its input activation internally (deallocate_activation=True)
        xt = ttnn.gelu(conv_out)  # (1,1,n*H*W,480) NHWC
        ttnn.deallocate(conv_out)
    C = DS_HIDDEN
    # NHWC (n,H,W,C) -> (n, W, C, H) flattened to (n, W, C*H), on device (Whisper-style, no D2H).
    xt = ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT)
    xt = ttnn.reshape(xt, (n, H, W, C))
    xt = ttnn.permute(xt, (0, 2, 3, 1))  # (n, W, C, H)
    xt = ttnn.reshape(xt, (n, W, C * H))  # (n,13,7680)
    xt2 = ttnn.to_layout(xt, ttnn.TILE_LAYOUT)
    ttnn.deallocate(xt)
    out_d = ttnn.linear(xt2, conv_out_w, compute_kernel_config=HIFI4)  # (n,13,1024)
    ttnn.deallocate(xt2)
    out_rm = ttnn.to_layout(out_d, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(out_d)
    return ttnn.reshape(out_rm, (n, W, D_MODEL))


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
_PE_TT = {}


def _sinusoid_pe(per_chunk=13):
    if per_chunk not in _PE_CACHE:
        import math

        ch = D_MODEL
        log_inc = math.log(10000.0) / (ch // 2 - 1)
        inv = torch.exp(-log_inc * torch.arange(ch // 2).float())
        t = torch.arange(1500)[:, None].float() * inv[None, :]
        _PE_CACHE[per_chunk] = torch.cat([torch.sin(t), torch.cos(t)], 1)[:per_chunk]
    return _PE_CACHE[per_chunk]


# Chunk counts encode_mel pads clips of 1-32 s to (whole 8-chunk windows).
WARM_BUCKETS = (8, 16, 24, 32)


def warmup(params, device):
    """Encode every bucket once, then stop adding shapes. The conv frontend and the
    positional embedding keep device tensors per input shape; one allocated after a
    trace is captured can be overwritten by that trace's replays, so this must run
    before any trace capture."""
    for n in WARM_BUCKETS:
        encode_mel(torch.zeros(128, n * N_WINDOW * 2), params, device)
    params["frozen"] = True


def encode_mel(mel, params, device):
    """Full-TT encoder from mel (num_mel, T): TT conv2d frontend -> +PE -> transformer
    -> projector. Returns audio embeds (feat_out_len(T), output_dim) as torch.

    The chunk count is padded to a whole window (8 chunks), so clips up to 32 s use 4
    shapes. Padded chunks and the padded tokens of a partial last chunk are excluded
    from attention and dropped from the output. A clip under one chunk runs the conv at
    its own width, as the model does: there, padding would change its last tokens.
    After warmup() it pads that clip to the smallest bucket instead, and clips longer
    than the largest bucket raise."""
    T = int(mel.shape[1])
    chunk = N_WINDOW * 2
    frozen = params.get("frozen", False)
    if T < chunk and not frozen:
        n, chunk = 1, T
    else:
        chunks_per_window = WINDOW_TOKENS // TOKENS_PER_CHUNK
        n_real = (T + chunk - 1) // chunk
        n = (n_real + chunks_per_window - 1) // chunks_per_window * chunks_per_window
        if frozen and n > WARM_BUCKETS[-1]:
            raise ValueError(f"{T} mel frames need {n} chunks; warmed buckets go up to {WARM_BUCKETS[-1]}")
    conv_d = conv_frontend_tt(mel, params["conv_w"], params["conv_out_w"], None, device, n=n, chunk=chunk)
    per_chunk = int(conv_d.shape[1])
    pe_key = (id(device), per_chunk)
    if pe_key not in _PE_TT:
        pe = _sinusoid_pe(per_chunk).view(1, per_chunk, D_MODEL)
        _PE_TT[pe_key] = _from_torch(pe, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.add(conv_d, _PE_TT[pe_key])
    ttnn.deallocate(conv_d)
    S = n * per_chunk
    x = ttnn.reshape(x, (1, 1, S, D_MODEL))
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    S_valid = feat_out_len(T)
    cu = _from_torch(
        torch.tensor(_fixed_window_bounds(S_valid, S), dtype=torch.int32),
        device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    out = encode(x, params, device, S=S_valid, cu_window_seqlens=cu)
    ttnn.deallocate(cu)
    return out


def _layer(x, lp, device, cu_window_seqlens=None):
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
        cu_window_seqlens=cu_window_seqlens,
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


def encode(x_in, params, device, S=None, cu_window_seqlens=None):
    """x_in: (S, D_MODEL) torch, or a (1,1,S,D) device tensor with PE already added.
    Returns the first S rows, (S, output_dim) torch. Attention is over all rows unless
    `cu_window_seqlens` (device uint32 window bounds) is given."""
    if isinstance(x_in, ttnn.Tensor):
        x = x_in
        if S is None:
            S = int(x.shape[-2])
    else:
        S = x_in.shape[0] if S is None else S
        x = _from_torch(
            x_in.unsqueeze(0).unsqueeze(0),
            device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # (1,1,S,D)
    for lp in params["layers"]:
        x = _layer(x, lp, device, cu_window_seqlens)
    x = ttnn.layer_norm(
        x, weight=params["lnpost_w"], bias=params["lnpost_b"], epsilon=LN_EPS, compute_kernel_config=HIFI4
    )
    core = ttnn.CoreGrid(y=device.compute_with_storage_grid_size().y, x=device.compute_with_storage_grid_size().x)
    x = ttnn.linear(
        x, params["proj1_w"], bias=params["proj1_b"], activation="gelu", core_grid=core, compute_kernel_config=HIFI4
    )
    x = ttnn.linear(x, params["proj2_w"], bias=params["proj2_b"], core_grid=core, compute_kernel_config=HIFI4)
    out = _to_torch(x).reshape(-1, OUTPUT_DIM)[:S].float()
    ttnn.deallocate(x)
    return out
