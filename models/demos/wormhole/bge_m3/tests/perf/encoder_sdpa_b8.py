"""Standalone bench for the encoder SDPA at the B8/S512 shape.

Purpose: reach an SDPA that takes compact per-request valid lengths instead of
a dense [B, 1, S, S] mask, and prove it on this shape before the model calls it.

Why compact lengths. The model builds a dense bfloat8_b mask of 2.23 MB and the
SDPA reader reads it again in all 24 layers, 53.5 MB for each forward. The mask
holds only zeros when no token is padded. Removing it drops SDPA from 167.1 us
to 84.8 us for each call and the forward from 17.388 ms to 14.763 ms, but that
measurement is not shippable: it also removes real masking, and a row with 448
pad tokens then returns cos 0.4147 against the masked result.

The stock op cannot take compact lengths. Windowed mode builds a mask on device
from cu_window_seqlens, but that tensor is one shared 1-D window list, so every
batch row gets the same windows. BGE-M3 needs a separate valid prefix for each
request.

This bench reports three numbers for each case:

  stock_dense   the stock op with the dense mask, which is the model today
  stock_nomask  the stock op with no mask, the speed target and not correct
  micro         the encoder micro-op with compact valid lengths

Correctness compares micro against stock_dense, on padded and unpadded rows.
The padded row is the test that matters: the earlier dense-mask removal passed
PCC on unpadded input and still corrupted padded rows.

Run:
  python models/demos/wormhole/bge_m3/tests/perf/encoder_sdpa_b8.py
"""

import argparse
import time

import torch

import ttnn

BATCH = 8
HEADS = 16
SEQ = 512
HEAD_DIM = 64
WARMUP = 3
ITERS = 10


def torch_reference(q, k, v, valid_lengths):
    """Non-causal SDPA with a per-row valid prefix, in float32."""
    scale = 1.0 / (HEAD_DIM**0.5)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    for row, length in enumerate(valid_lengths):
        if length < SEQ:
            scores[row, :, :, length:] = float("-inf")
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def dense_additive_mask(valid_lengths, dtype):
    mask = torch.zeros((BATCH, 1, SEQ, SEQ), dtype=torch.float32)
    for row, length in enumerate(valid_lengths):
        if length < SEQ:
            mask[row, :, :, length:] = -100000.0
    return mask


def to_device(tensor, device, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, device=device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def time_call(fn, device):
    for _ in range(WARMUP):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    best = None
    for _ in range(ITERS):
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        elapsed = (time.perf_counter() - start) * 1e6
        best = elapsed if best is None else min(best, elapsed)
        ttnn.deallocate(out)
    return best


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


parser = argparse.ArgumentParser()
parser.add_argument("--q-chunk", type=int, default=256)
parser.add_argument("--k-chunk", type=int, default=512)
parser.add_argument("--grid-x", type=int, default=13)
parser.add_argument("--grid-y", type=int, default=10)
parser.add_argument("--skip-micro", action="store_true", help="only measure the stock bounds")
args = parser.parse_args()

from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa import EncoderSDPAConfig
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa.op import bge_encoder_sdpa_experimental

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)

q_host = torch.randn((BATCH, HEADS, SEQ, HEAD_DIM), dtype=torch.bfloat16)
k_host = torch.randn((BATCH, HEADS, SEQ, HEAD_DIM), dtype=torch.bfloat16)
v_host = torch.randn((BATCH, HEADS, SEQ, HEAD_DIM), dtype=torch.bfloat16)

# Case A: nothing padded. Case B: row 1 keeps 64 tokens, row 5 keeps 192.
cases = {
    "unpadded": [SEQ] * BATCH,
    "padded": [SEQ, 64, SEQ, SEQ, SEQ, 192, SEQ, SEQ],
}

q_dev = to_device(q_host, device, ttnn.bfloat8_b)
k_dev = to_device(k_host, device, ttnn.bfloat8_b)
v_dev = to_device(v_host, device, ttnn.bfloat8_b)

program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(args.grid_x, args.grid_y),
    q_chunk_size=args.q_chunk,
    k_chunk_size=args.k_chunk,
)
compute_config = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
scale = 1.0 / (HEAD_DIM**0.5)

print(
    "  shape B%d H%d S%d D%d, q_chunk %d, k_chunk %d, grid %dx%d"
    % (BATCH, HEADS, SEQ, HEAD_DIM, args.q_chunk, args.k_chunk, args.grid_x, args.grid_y)
)
print()

results = {}
for name, lengths in cases.items():
    reference = torch_reference(q_host, k_host, v_host, lengths)
    mask_dev = to_device(dense_additive_mask(lengths, ttnn.bfloat8_b), device, ttnn.bfloat8_b)

    def run_dense():
        return ttnn.transformer.scaled_dot_product_attention(
            q_dev,
            k_dev,
            v_dev,
            is_causal=False,
            attn_mask=mask_dev,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_config,
        )

    def run_nomask():
        return ttnn.transformer.scaled_dot_product_attention(
            q_dev,
            k_dev,
            v_dev,
            is_causal=False,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_config,
        )

    dense_out = run_dense()
    dense_host = ttnn.to_torch(dense_out)
    ttnn.deallocate(dense_out)
    dense_us = time_call(run_dense, device)
    nomask_us = time_call(run_nomask, device)

    row = {
        "stock_dense_us": dense_us,
        "stock_nomask_us": nomask_us,
        "dense_vs_torch": cosine(dense_host, reference),
    }

    if not args.skip_micro:
        config = EncoderSDPAConfig(
            batch=BATCH,
            num_q_heads=HEADS,
            num_kv_heads=HEADS,
            q_seq_len=SEQ,
            kv_seq_len=SEQ,
            head_dim=HEAD_DIM,
            q_chunk_size=args.q_chunk,
            k_chunk_size=args.k_chunk,
            grid_x=args.grid_x,
            grid_y=args.grid_y,
            scale=scale,
            use_runtime_lengths=True,
        )
        lengths_dev = ttnn.from_torch(
            torch.tensor(lengths, dtype=torch.int32).reshape(BATCH, 1),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def run_micro():
            return bge_encoder_sdpa_experimental(q_dev, k_dev, v_dev, valid_lengths=lengths_dev, config=config)

        try:
            micro_out = run_micro()
            micro_host = ttnn.to_torch(micro_out)
            ttnn.deallocate(micro_out)
            row["micro_us"] = time_call(run_micro, device)
            row["micro_vs_torch"] = cosine(micro_host, reference)
            row["micro_vs_dense"] = cosine(micro_host, dense_host)
        except Exception as exc:
            row["micro_error"] = str(exc).replace("\n", " ")[:220]

    results[name] = row
    ttnn.deallocate(mask_dev)

ttnn.close_device(device)

print(
    "  %-10s %13s %14s %11s %13s %13s" % ("case", "stock_dense", "stock_nomask", "micro", "micro/torch", "micro/dense")
)
for name, row in results.items():
    micro = "%.1f" % row["micro_us"] if "micro_us" in row else "-"
    mt = "%.6f" % row["micro_vs_torch"] if "micro_vs_torch" in row else "-"
    md = "%.6f" % row["micro_vs_dense"] if "micro_vs_dense" in row else "-"
    print(
        "  %-10s %10.1f us %11.1f us %8s us %13s %13s"
        % (name, row["stock_dense_us"], row["stock_nomask_us"], micro, mt, md)
    )
for name, row in results.items():
    print("  %-10s stock_dense vs torch = %.6f" % (name, row["dense_vs_torch"]))
    if "micro_error" in row:
        print("  %-10s micro FAILED: %s" % (name, row["micro_error"]))

if all("micro_us" in r for r in results.values()):
    gain = results["unpadded"]["stock_dense_us"] - results["unpadded"]["micro_us"]
    print()
    print("  saving %.1f us for each call -> %.2f ms over 24 layers" % (gain, 24 * gain / 1000))
    print("  projected forward: 17.388 - %.2f = %.2f ms" % (24 * gain / 1000, 17.388 - 24 * gain / 1000))
