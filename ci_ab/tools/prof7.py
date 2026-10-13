# Op-level device time and output bits of the ROW MAX callers (one side per process, under tracy).
# usage: python -m tracy -r -p --no-web-server -o OUT prof.py <side> <run> [case prefixes]
import hashlib
import sys

import torch
import ttnn
from tracy import signpost

REPS = 5


def softmax_case(shape, dtype, stable=False):
    def setup(device):
        x = ttnn.from_torch(torch.randn(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        return lambda: ttnn.softmax(x, dim=-1, numeric_stable=stable), [x]

    return setup


def softmax_sharded_case(b, heads, h, w, dtype, subblock_w=6):
    def setup(device):
        t = torch.randn((b, heads, h, w))
        mem = ttnn.create_sharded_memory_config(
            t.shape,
            core_grid=ttnn.CoreGrid(y=heads, x=b),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        pc = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(b, heads), subblock_w=subblock_w, block_h=h // 32, block_w=w // 32
        )
        c = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
        )
        x = ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)

        def run():
            ttnn.scale_mask_softmax_in_place(x, program_config=pc, compute_kernel_config=c, numeric_stable=True)

        return run, [x]

    return setup


def softmax_vit_case():
    def setup(device):
        mem = ttnn.create_sharded_memory_config(
            (224, 224),
            core_grid=ttnn.CoreGrid(y=10, x=11),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.from_torch(
            torch.randn(10, 11, 224, 224), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
        )
        pc = ttnn.SoftmaxShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(11, 10), subblock_w=7, block_h=7, block_w=7)
        return lambda: ttnn.softmax_in_place(x, program_config=pc), [x]

    return setup


def moreh_softmax_case(shape, dim=3):
    def setup(device):
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        return lambda: ttnn.operations.moreh.softmax(x, dim), [x]

    return setup


def sdpa_decode_case(b, nh, nkv, s, d, k_chunk, fidelity, fp32):
    def setup(device):
        q = ttnn.from_torch(torch.randn(1, b, nh, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        k = ttnn.from_torch(torch.randn(b, nkv, s, d), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        v = ttnn.from_torch(torch.randn(b, nkv, s, d), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        grid = device.compute_with_storage_grid_size()
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=0, k_chunk_size=k_chunk, exp_approx_mode=False)
        c = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32)
        pos = [s - 1 - 7 * i for i in range(b)]
        return (
            lambda: ttnn.transformer.scaled_dot_product_attention_decode(
                q, k, v, cur_pos=pos, is_causal=True, program_config=pc, compute_kernel_config=c
            ),
            [q, k, v],
        )

    return setup


def max_w_case(shape):
    def setup(device):
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        return lambda: ttnn.max(x, dim=-1, keepdim=True), [x]

    return setup


def softmax_cfg_case(shape, dtype, fidelity, fp32):
    def setup(device):
        x = ttnn.from_torch(torch.randn(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        c = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32)
        return lambda: ttnn.softmax(x, dim=-1, numeric_stable=True, compute_kernel_config=c), [x]

    return setup


def sdpa_prefill_case(nh, nkv, s, d, chunk, fidelity, fp32):
    def setup(device):
        q = ttnn.from_torch(torch.randn(1, nh, s, d), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        k = ttnn.from_torch(torch.randn(1, nkv, s, d), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        v = ttnn.from_torch(torch.randn(1, nkv, s, d), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=chunk, k_chunk_size=chunk)
        c = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32)
        return (
            lambda: ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=d**-0.5, program_config=pc, compute_kernel_config=c
            ),
            [q, k, v],
        )

    return setup


CASES = [
    ("smsh_8x4x384x384_bf16", softmax_sharded_case(8, 4, 384, 384, ttnn.bfloat16)),
    ("smvit_10x11x224x224_bfp8", softmax_vit_case()),
    ("smst_1x8x1024x1024_bf16", softmax_case((1, 8, 1024, 1024), ttnn.bfloat16, stable=True)),
    ("clip_1x12x77x77_stable_hifi4fp32", softmax_cfg_case((1, 12, 77, 77), ttnn.bfloat16, ttnn.MathFidelity.HiFi4, True)),
    ("msm_1x1x256x1024", moreh_softmax_case((1, 1, 256, 1024))),
    ("sdpad_b32_s2048_k512_hifi2fp32", sdpa_decode_case(32, 32, 8, 2048, 128, 512, ttnn.MathFidelity.HiFi2, True)),
    ("sdpad_b32_s2048_k512_hifi4fp32", sdpa_decode_case(32, 32, 8, 2048, 128, 512, ttnn.MathFidelity.HiFi4, True)),
    ("sdpap_llama8b_s4096_hifi4fp32", sdpa_prefill_case(32, 8, 4096, 128, 256, ttnn.MathFidelity.HiFi4, True)),
    ("sdpap_llama8b_s4096_hifi2", sdpa_prefill_case(32, 8, 4096, 128, 256, ttnn.MathFidelity.HiFi2, False)),
    ("maxw_1x8x1024x1024_bf16", max_w_case((1, 8, 1024, 1024))),
]


def main():
    side, run = sys.argv[1], sys.argv[2]
    prefixes = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("",)
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    for name, setup in CASES:
        if not name.startswith(prefixes):
            continue
        torch.manual_seed(58705)
        try:
            fn, keep = setup(device)
            y = None
            for k in range(REPS):
                ttnn.synchronize_device(device)
                signpost(header=f"{name}#{k}")
                y = fn()
            ttnn.synchronize_device(device)
            o = ttnn.to_torch(y if y is not None else keep[0]).contiguous().float()
            h = hashlib.sha256(o.numpy().tobytes()).hexdigest()[:16]
            torch.save(o, f"/tmp/hr_out/{side}_{run}_{name}.pt")
            ttnn.ReadDeviceProfiler(device)
            print(f"CASE {name} ok BITS {side} r{run} {h}", flush=True)
            del y, keep, o
        except Exception as e:
            print(f"CASE {name} FAIL {str(e).splitlines()[0][:300]}", flush=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
