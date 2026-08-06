"""Count ttnn ops issued by one audio decode, grouped by op and by call site.

Tracy gives device time but needs a 40k op-support capture and a separate report pass. For deciding
*what to fuse*, op counts are the number that matters, and they can be had by wrapping the ttnn
entry points the audio stack uses. Cheap, no profiler, no signposts.

Also counts `depthwise_tap_filter` calls by the path they resolve to, which is what says how much of
the op budget the exact-but-slow MAC fallback is responsible for.
"""

import collections
import json
import os
import time

import torch

import ttnn

from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio as dec_mod
from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from models.tt_dit.layers import audio_ops

COUNTS = collections.Counter()
ENABLED = {"on": False}

WRAP = [
    "slice",
    "concat",
    "multiply",
    "add",
    "subtract",
    "reshape",
    "permute",
    "typecast",
    "to_layout",
    "conv1d",
    "snake_beta",
    "sin",
    "reciprocal",
    "matmul",
    "zeros",
    "clone",
    "to_memory_config",
    "sharded_to_interleaved",
    "interleaved_to_sharded",
    "tilize",
    "untilize",
]


def wrap_all():
    for name in WRAP:
        fn = getattr(ttnn, name, None)
        if fn is None or not callable(fn):
            continue

        def make(n, f):
            def wrapper(*a, **k):
                if ENABLED["on"]:
                    COUNTS[n] += 1
                return f(*a, **k)

            return wrapper

        setattr(ttnn, name, make(name, fn))

    conv3d = getattr(ttnn.experimental, "conv3d", None)
    if conv3d is not None:

        def conv3d_wrapper(*a, _f=conv3d, **k):
            if ENABLED["on"]:
                COUNTS["experimental.conv3d"] += 1
            return _f(*a, **k)

        ttnn.experimental.conv3d = conv3d_wrapper

    # Path taken by each depthwise FIR call, and the op cost of the MAC form.
    orig_mac = audio_ops._depthwise_tap_mac

    def mac_wrapper(x_BTC, taps, stride, *, T_out):
        if ENABLED["on"]:
            nz = sum(1 for t in taps if t != 0.0)
            COUNTS["FIR:mac_calls"] += 1
            COUNTS["FIR:mac_ops"] += 3 * nz - 1  # nz slices + nz multiplies + nz-1 adds
        return orig_mac(x_BTC, taps, stride, T_out=T_out)

    audio_ops._depthwise_tap_mac = mac_wrapper

    orig_filter = audio_ops.depthwise_tap_filter

    def filter_wrapper(x_BTC, taps, stride, **kw):
        if ENABLED["on"]:
            COUNTS["FIR:total_calls"] += 1
            COUNTS[f"FIR:shape C={int(x_BTC.shape[2])} K={len(taps)} s={stride}"] += 1
        return orig_filter(x_BTC, taps, stride, **kw)

    audio_ops.depthwise_tap_filter = filter_wrapper
    dec_mod.depthwise_tap_filter = filter_wrapper
    import models.tt_dit.layers.audio_resample as ar

    ar.depthwise_tap_filter = filter_wrapper


def main():
    wrap_all()
    weights_dir = os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "audio_vae")
    config = {
        k: v for k, v in json.load(open(os.path.join(weights_dir, "config.json"))).items() if not k.startswith("_")
    }
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        from safetensors.torch import load_file

        decoder = MiniMaxH3AudioDecoder(
            latent_channels=config["latent_channels"],
            latent_dim=config["latent_dim"],
            decoder_dim=config["decoder_dim"],
            decoder_rates=tuple(config["decoder_rates"]),
            decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
            resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
            mesh_device=device,
        )
        decoder.load_torch_state_dict(
            convert_minimax_h3_audio_state_dict(
                load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
            ),
            strict=False,
        )
        torch.manual_seed(2)
        latents = torch.randn(2, config["latent_channels"], 207) * 0.1

        decoder(latents)  # warm
        ttnn.synchronize_device(device)

        ENABLED["on"] = True
        start = time.perf_counter()
        decoder(latents)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
        ENABLED["on"] = False

        fir = {k: v for k, v in COUNTS.items() if k.startswith("FIR:")}
        ops = {k: v for k, v in COUNTS.items() if not k.startswith("FIR:")}
        total = sum(ops.values())
        print(f"\n=== one decode: {elapsed:.3f} s, {total} wrapped ttnn calls ===")
        for name, n in sorted(ops.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<28} {n:>6}  {100.0 * n / max(total, 1):>5.1f}%")
        print("\n=== depthwise FIR ===")
        for name, n in sorted(fir.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<40} {n:>6}")
        mac_ops = COUNTS.get("FIR:mac_ops", 0)
        print(f"\nMAC fallback accounts for ~{mac_ops} of {total} ops ({100.0 * mac_ops / max(total,1):.1f}%)")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
