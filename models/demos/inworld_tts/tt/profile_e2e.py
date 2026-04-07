"""Profile full decoder and encoder forward passes for Tracy.

Usage:
    python3 -m tracy -p -v -r --op-support-count 30000 -m models.demos.inworld_tts.tt.profile_e2e

Uses random weights (no checkpoint needed).
"""

import torch
from vector_quantize_pytorch import ResidualFSQ

import ttnn
from models.demos.inworld_tts.tt.codec_decoder import TtCodecDecoder
from models.demos.inworld_tts.tt.codec_encoder import TtAcousticEncoder, TtSemanticEncoder
from models.demos.inworld_tts.tt.model_config import ENCODER_CHANNELS, ENCODER_STRIDES, VOCOS_DIM, VOCOS_MLP_DIM


def _bf16(t):
    return t.to(torch.bfloat16).to(torch.float32)


# ---------------------------------------------------------------------------
# State dict builders (random weights)
# ---------------------------------------------------------------------------
def make_decoder_state_dict(depth=12):
    dim = VOCOS_DIM
    mlp_dim = VOCOS_MLP_DIM
    sd = {}

    sd["fc_post_a.weight"] = _bf16(torch.randn(dim, 2048))
    sd["fc_post_a.bias"] = _bf16(torch.randn(dim))

    prefix = "backbone."
    sd[prefix + "embed.weight"] = _bf16(torch.randn(dim, dim, 7))
    sd[prefix + "embed.bias"] = _bf16(torch.randn(dim))

    for i in range(2):
        for key in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
            sd[prefix + f"prior_net.{i}.{key}"] = _bf16(torch.randn(dim))
        sd[prefix + f"prior_net.{i}.conv1.weight"] = _bf16(torch.randn(dim, dim, 3))
        sd[prefix + f"prior_net.{i}.conv1.bias"] = _bf16(torch.randn(dim))
        sd[prefix + f"prior_net.{i}.conv2.weight"] = _bf16(torch.randn(dim, dim, 3))
        sd[prefix + f"prior_net.{i}.conv2.bias"] = _bf16(torch.randn(dim))

    for i in range(depth):
        p = prefix + f"transformers.{i}."
        sd[p + "att_norm.weight"] = _bf16(torch.randn(dim))
        sd[p + "att.c_attn.weight"] = _bf16(torch.randn(3 * dim, dim))
        sd[p + "att.c_proj.weight"] = _bf16(torch.randn(dim, dim))
        sd[p + "ffn_norm.weight"] = _bf16(torch.randn(dim))
        sd[p + "mlp.fc1.weight"] = _bf16(torch.randn(mlp_dim, dim))
        sd[p + "mlp.fc2.weight"] = _bf16(torch.randn(dim, mlp_dim))

    for i in range(2):
        for key in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
            sd[prefix + f"post_net.{i}.{key}"] = _bf16(torch.randn(dim))
        sd[prefix + f"post_net.{i}.conv1.weight"] = _bf16(torch.randn(dim, dim, 3))
        sd[prefix + f"post_net.{i}.conv1.bias"] = _bf16(torch.randn(dim))
        sd[prefix + f"post_net.{i}.conv2.weight"] = _bf16(torch.randn(dim, dim, 3))
        sd[prefix + f"post_net.{i}.conv2.bias"] = _bf16(torch.randn(dim))

    sd[prefix + "final_layer_norm.weight"] = _bf16(torch.randn(dim))
    sd[prefix + "final_layer_norm.bias"] = _bf16(torch.randn(dim))

    sd["head.out.weight"] = _bf16(torch.randn(1282, dim) * 0.01)
    sd["head.out.bias"] = _bf16(torch.randn(1282) * 0.01)

    return sd


def make_acoustic_encoder_state_dict():
    from models.demos.inworld_tts.reference.functional import weight_norm_compute

    sd = {}
    ch = ENCODER_CHANNELS
    strides = ENCODER_STRIDES
    K_FIR = 12

    def _wn(cout, cin, k):
        sd_local = {}
        sd_local["weight_g"] = _bf16(torch.randn(cout, 1, 1) / 10)
        sd_local["weight_v"] = _bf16(torch.randn(cout, cin, k) / 10)
        sd_local["bias"] = _bf16(torch.randn(cout))
        w = weight_norm_compute(sd_local["weight_g"], sd_local["weight_v"])
        return w, sd_local["bias"], sd_local

    w, b, _ = _wn(48, 1, 7)
    sd["conv_blocks.0.weight_g"] = _bf16(torch.randn(48, 1, 1) / 10)
    sd["conv_blocks.0.weight_v"] = _bf16(torch.randn(48, 1, 7) / 10)
    sd["conv_blocks.0.bias"] = _bf16(torch.randn(48))

    for block in range(5):
        prefix = f"conv_blocks.{block + 1}."
        cin = ch[block]
        cout = ch[block + 1]
        stride = strides[block]

        for res in range(3):
            rp = f"{prefix}block.{res}."
            for act_idx, act_prefix in [(0, "block.0."), (2, "block.2.")]:
                sd[rp + f"block.{act_idx}.act.alpha"] = _bf16(torch.randn(cin))
                sd[rp + f"block.{act_idx}.act.beta"] = _bf16(torch.rand(cin) + 4.0)
                sd[rp + f"block.{act_idx}.upsample.filter"] = _bf16(torch.randn(1, 1, K_FIR) / 4)
                sd[rp + f"block.{act_idx}.downsample.lowpass.filter"] = _bf16(torch.randn(1, 1, K_FIR) / 4)
            sd[rp + "block.1.weight_g"] = _bf16(torch.randn(cin, 1, 1) / 10)
            sd[rp + "block.1.weight_v"] = _bf16(torch.randn(cin, cin, 7) / 10)
            sd[rp + "block.1.bias"] = _bf16(torch.randn(cin))
            sd[rp + "block.3.weight_g"] = _bf16(torch.randn(cin, 1, 1) / 10)
            sd[rp + "block.3.weight_v"] = _bf16(torch.randn(cin, cin, 1) / 10)
            sd[rp + "block.3.bias"] = _bf16(torch.randn(cin))

        sd[prefix + "block.3.act.alpha"] = _bf16(torch.randn(cin))
        sd[prefix + "block.3.act.beta"] = _bf16(torch.rand(cin) + 4.0)
        sd[prefix + "block.3.upsample.filter"] = _bf16(torch.randn(1, 1, K_FIR) / 4)
        sd[prefix + "block.3.downsample.lowpass.filter"] = _bf16(torch.randn(1, 1, K_FIR) / 4)

        ks = stride * 2
        sd[prefix + "block.4.weight_g"] = _bf16(torch.randn(cout, 1, 1) / 10)
        sd[prefix + "block.4.weight_v"] = _bf16(torch.randn(cout, cin, ks) / 10)
        sd[prefix + "block.4.bias"] = _bf16(torch.randn(cout))

    sd["conv_final_block.0.act.alpha"] = _bf16(torch.randn(1536))
    sd["conv_final_block.0.act.beta"] = _bf16(torch.rand(1536) + 4.0)
    sd["conv_final_block.0.upsample.filter"] = _bf16(torch.randn(1, 1, K_FIR) / 4)
    sd["conv_final_block.0.downsample.lowpass.filter"] = _bf16(torch.randn(1, 1, K_FIR) / 4)
    sd["conv_final_block.1.weight_g"] = _bf16(torch.randn(1024, 1, 1) / 10)
    sd["conv_final_block.1.weight_v"] = _bf16(torch.randn(1024, 1536, 3) / 10)
    sd["conv_final_block.1.bias"] = _bf16(torch.randn(1024))

    return sd


def make_semantic_encoder_state_dict():
    C = 1024
    prefix = "SemanticEncoder_module."
    sd = {
        prefix + "initial_conv.weight": _bf16(torch.randn(C, C, 3)),
        prefix + "initial_conv.bias": _bf16(torch.randn(C)),
        prefix + "residual_blocks.1.weight": _bf16(torch.randn(C, C, 3)),
        prefix + "residual_blocks.1.bias": _bf16(torch.randn(C)),
        prefix + "residual_blocks.3.weight": _bf16(torch.randn(C, C, 3)),
        prefix + "residual_blocks.3.bias": _bf16(torch.randn(C)),
        prefix + "final_conv.weight": _bf16(torch.randn(C, C, 3)),
        prefix + "final_conv.bias": _bf16(torch.randn(C)),
    }
    return sd


# ---------------------------------------------------------------------------
# Profile functions
# ---------------------------------------------------------------------------
def profile_decoder(device, depth=2, seq_len=64, warmup=1, runs=3):
    print(f"\n{'='*60}")
    print(f"DECODER PROFILE: depth={depth}, seq_len={seq_len}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    quantizer = ResidualFSQ(levels=[4, 4, 4, 4, 4, 4, 4, 4], dim=2048, num_quantizers=1)
    sd = make_decoder_state_dict(depth=depth)
    vq_codes = torch.randint(0, 65536, (1, 1, seq_len))

    tt_decoder = TtCodecDecoder(device=device, state_dict=sd, quantizer=quantizer, depth=depth)

    with torch.no_grad():
        for i in range(warmup):
            _ = tt_decoder(vq_codes)
            print(f"  warmup {i+1}/{warmup} done")

        for i in range(runs):
            _ = tt_decoder(vq_codes)
            print(f"  run {i+1}/{runs} done")

    print("Decoder profiling complete.")


def profile_acoustic_encoder(device, seq_len=64, warmup=1, runs=3):
    print(f"\n{'='*60}")
    print(f"ACOUSTIC ENCODER PROFILE: seq_len={seq_len}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    sd = make_acoustic_encoder_state_dict()
    n_samples = seq_len * 320  # 320 = total stride
    waveform = _bf16(torch.randn(1, 1, n_samples))

    tt_encoder = TtAcousticEncoder(sd, device)

    with torch.no_grad():
        for i in range(warmup):
            _ = tt_encoder(waveform)
            print(f"  warmup {i+1}/{warmup} done")

        for i in range(runs):
            _ = tt_encoder(waveform)
            print(f"  run {i+1}/{runs} done")

    print("Acoustic encoder profiling complete.")


def profile_semantic_encoder(device, seq_len=64, warmup=1, runs=3):
    print(f"\n{'='*60}")
    print(f"SEMANTIC ENCODER PROFILE: seq_len={seq_len}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    sd = make_semantic_encoder_state_dict()
    x = _bf16(torch.randn(1, 1024, seq_len))

    tt_encoder = TtSemanticEncoder(device, sd)

    with torch.no_grad():
        for i in range(warmup):
            _ = tt_encoder(x)
            print(f"  warmup {i+1}/{warmup} done")

        for i in range(runs):
            _ = tt_encoder(x)
            print(f"  run {i+1}/{runs} done")

    print("Semantic encoder profiling complete.")


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        profile_decoder(device, depth=2, seq_len=64)

        ttnn.close_device(device)
        device = ttnn.open_device(device_id=0, l1_small_size=16384)

        profile_semantic_encoder(device, seq_len=64)

        ttnn.close_device(device)
        device = ttnn.open_device(device_id=0, l1_small_size=16384)

        profile_acoustic_encoder(device, seq_len=64)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
