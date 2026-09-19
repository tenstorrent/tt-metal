# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""C-05 probe: the decoder's SDPA with its dense pad mask against the same op with no mask and a
logical sequence length that excludes the pad columns. Single device, decoder shapes and config.

    pytest models/tt_dit/tests/models/minimax_h3/tools/sdpa_mask_probe.py -q -s

Reports, per variant: min-of-N wall time of the op alone (synchronize before and after), and whether
the valid rows equal the dense-mask result bit for bit. Not a gate; a measurement.
"""

import time

import pytest
import torch

import ttnn

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

HEADS, SEQ, VALID, HEAD_DIM = 32, 1824, 1797, 64
REPEATS = 20


def _timed(mesh_device, fn):
    fn()  # program cache
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(REPEATS):
        ttnn.synchronize_device(mesh_device)
        mark = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - mark)
    return out, best * 1e3


def _sdpa(q, k, v, mask, program_config, kernel_config, scale=None):
    kwargs = {}
    if scale is not None:
        kwargs["scale"] = scale
    return ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        is_causal=False,
        program_config=program_config,
        compute_kernel_config=kernel_config,
        **kwargs,
    )


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_sdpa_mask_probe(mesh_device):
    torch.manual_seed(0)
    q_t = torch.randn(1, HEADS, SEQ, HEAD_DIM)
    k_t = torch.randn(1, HEADS, SEQ, HEAD_DIM)
    v_t = torch.randn(1, HEADS, SEQ, HEAD_DIM)
    # Pad rows of q/k/v carry junk in the real decoder too (the suffix constant's zero tail after the
    # blocks have run through it); keep them non-zero so a variant that reads them shows it.
    mask_t = torch.zeros(1, 1, SEQ, SEQ)
    mask_t[..., VALID:] = float("-inf")

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        q_chunk_size=192,
        k_chunk_size=192,
        exp_approx_mode=False,
    )
    kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )

    dev = lambda t, dtype=ttnn.bfloat16: ttnn.from_torch(t, dtype=dtype, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    q, k, v, mask = dev(q_t), dev(k_t), dev(v_t), dev(mask_t)

    results = {}

    # A: today's call.
    out, ms = _timed(mesh_device, lambda: _sdpa(q, k, v, mask, program_config, kernel_config))
    ref = ttnn.to_torch(out)
    results["A dense bf16 mask (today)"] = (ms, None)
    print(f"\nA dense bf16 mask (today): {ms:.3f} ms  out {tuple(out.shape)}")

    def compare(name, out, ms):
        got = ttnn.to_torch(out)
        rows = min(got.shape[-2], VALID)
        a, b = ref[..., :rows, :], got[..., :rows, :]
        equal = torch.equal(a, b)
        maxdiff = (a.float() - b.float()).abs().max().item()
        results[name] = (ms, equal)
        print(f"{name}: {ms:.3f} ms  out {tuple(out.shape)}  valid rows equal={equal}  max|diff|={maxdiff:.3e}")

    # B1: logical S=VALID views of the SAME buffers via the two-shape reshape, no mask.
    try:
        view = lambda t: ttnn.reshape(t, ttnn.Shape([1, HEADS, VALID, HEAD_DIM]), ttnn.Shape([1, HEADS, SEQ, HEAD_DIM]))
        qv, kv, vv = view(q), view(k), view(v)
        same_buffer = qv.buffer_address() == q.buffer_address()
        print(f"B1 view: shape {tuple(qv.shape)} padded {tuple(qv.padded_shape)} same buffer as q: {same_buffer}")
        out, ms = _timed(mesh_device, lambda: _sdpa(qv, kv, vv, None, program_config, kernel_config))
        compare("B1 logical-1797 views, no mask", out, ms)
        # Back to the padded logical shape, as the rest of the decoder expects.
        back = ttnn.reshape(out, ttnn.Shape([1, HEADS, SEQ, HEAD_DIM]), ttnn.Shape([1, HEADS, SEQ, HEAD_DIM]))
        print(f"B1 view back: shape {tuple(back.shape)} padded {tuple(back.padded_shape)}")
    except Exception as exc:  # noqa: BLE001 -- a probe reports, it does not fail
        print(f"B1 logical view FAILED: {type(exc).__name__}: {str(exc)[:300]}")

    # B2: tensors created with logical S=VALID from the host (pad rows zero), no mask.
    try:
        q2, k2, v2 = dev(q_t[..., :VALID, :]), dev(k_t[..., :VALID, :]), dev(v_t[..., :VALID, :])
        print(f"B2 from_torch: shape {tuple(q2.shape)} padded {tuple(q2.padded_shape)}")
        out, ms = _timed(mesh_device, lambda: _sdpa(q2, k2, v2, None, program_config, kernel_config))
        compare("B2 host-made logical-1797, no mask", out, ms)
    except Exception as exc:  # noqa: BLE001
        print(f"B2 FAILED: {type(exc).__name__}: {str(exc)[:300]}")

    # C: dense mask in bfloat4_b with a finite fill, same scale.
    try:
        mask_fin = torch.zeros(1, 1, SEQ, SEQ)
        mask_fin[..., VALID:] = -1e4
        mask4 = dev(mask_fin, ttnn.bfloat4_b)
        out, ms = _timed(mesh_device, lambda: _sdpa(q, k, v, mask4, program_config, kernel_config))
        compare("C bfp4 mask, finite fill", out, ms)
    except Exception as exc:  # noqa: BLE001
        print(f"C bfp4 mask FAILED: {type(exc).__name__}: {str(exc)[:300]}")

    # D: no mask at all on the padded 1824 tensors (WRONG numerics, the pad keys leak): the bound on
    # what removing the mask stream can save.
    out, ms = _timed(mesh_device, lambda: _sdpa(q, k, v, None, program_config, kernel_config))
    compare("D no mask, padded 1824 (wrong, bound only)", out, ms)

    print("\nsummary:")
    for name, (ms, equal) in results.items():
        print(f"  {name:<48} {ms:8.3f} ms  {'' if equal is None else ('EQUAL' if equal else 'differs')}")
