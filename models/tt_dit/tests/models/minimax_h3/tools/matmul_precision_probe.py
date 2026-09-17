#!/usr/bin/env python3
"""Op-level probe of two candidate decoder changes, before either is plumbed into the model.

  1. bfloat8_b matmul weights against bfloat16, at the four shapes `MiniMaxH3Attention3d` and its
     feed-forward actually run: time and error against a float32 torch reference.
  2. The attention's approximate exponential against the exact one, at the decoder's SDPA shape.

Single device. Nothing here loads a checkpoint or touches the weight cache.

    pytest probe_bf8_sdpa.py -q -s
"""

import time

import pytest
import torch

import ttnn

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

DIM = 2048  # 32 heads x 64
SEQ = 1824  # padded sequence of the served work unit
HEADS, HEAD_DIM = 32, 64
REPS = 20

# (label, K, N): the decoder's four matmuls, with ff1 packed for swiglu.
MATMULS = [
    ("to_qkv", DIM, 3 * DIM),
    ("to_out", DIM, DIM),
    ("ff1_swiglu", DIM, 2 * 4 * DIM),
    ("ff2", 4 * DIM, DIM),
]


def _time(fn, device):
    fn()
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(REPS):
        mark = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - mark) * 1e3)
    samples.sort()
    return samples[0], samples[len(samples) // 2]


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_probe_bf8_matmul_weights(mesh_device, reset_seeds):
    torch.manual_seed(0)
    print(f"\n{'shape':<26} {'weights':<12} {'min ms':>8} {'median':>8} {'rel err':>10} {'PCC':>9}")
    for label, k, n in MATMULS:
        activation = torch.randn(1, 1, SEQ, k) * 0.1
        weight = torch.randn(1, 1, k, n) * 0.05
        expected = (activation.float() @ weight.float()).squeeze()

        act_dev = ttnn.from_torch(activation, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
        variants = (
            (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, "bf16/HiFi2"),
            (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, "bf8_b/HiFi2"),
            (ttnn.bfloat16, ttnn.MathFidelity.LoFi, "bf16/LoFi"),
            (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, "bf8_b/LoFi"),
        )
        for wdtype, fidelity, name in variants:
            weight_dev = ttnn.from_torch(weight, dtype=wdtype, device=mesh_device, layout=ttnn.TILE_LAYOUT)
            config = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            low, median = _time(lambda: ttnn.matmul(act_dev, weight_dev, compute_kernel_config=config), mesh_device)
            actual = ttnn.to_torch(ttnn.matmul(act_dev, weight_dev, compute_kernel_config=config)).squeeze().float()
            rel = ((actual - expected).norm() / expected.norm()).item()
            flat_a, flat_e = actual.flatten(), expected.flatten()
            pcc = torch.corrcoef(torch.stack([flat_a, flat_e]))[0, 1].item()
            print(f"{f'{SEQ}x{k}x{n}':<26} {name:<12} {low:8.2f} {median:8.2f} {rel:10.2e} {100 * pcc:8.4f}%")
            ttnn.deallocate(weight_dev)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_probe_sdpa_exp_approx(mesh_device, reset_seeds):
    torch.manual_seed(1)
    query, key, value = (torch.randn(1, HEADS, SEQ, HEAD_DIM) * 0.3 for _ in range(3))
    expected = torch.nn.functional.scaled_dot_product_attention(query.float(), key.float(), value.float())

    tensors = [
        ttnn.from_torch(t, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
        for t in (query, key, value)
    ]
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    print(f"\n{'exp mode':<12} {'min ms':>8} {'median':>8} {'rel err':>10} {'PCC':>9}")
    for approx in (False, True):
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=192,
            k_chunk_size=192,
            exp_approx_mode=approx,
        )

        def run():
            return ttnn.transformer.scaled_dot_product_attention(
                *tensors, is_causal=False, program_config=program_config, compute_kernel_config=compute_config
            )

        low, median = _time(run, mesh_device)
        actual = ttnn.to_torch(run()).float()
        rel = ((actual - expected).norm() / expected.norm()).item()
        pcc = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item()
        print(f"{str(approx):<12} {low:8.2f} {median:8.2f} {rel:10.2e} {100 * pcc:8.4f}%")
