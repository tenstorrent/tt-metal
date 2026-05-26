# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SP AllGather correctness on the EXACT shape used by audio velocity in inner_step.

Per-device input:  (1, 1, 32, 128)   float32   sharded on dim=2 across SP=4
Per-device output: (1, 1, 128, 128)  float32

We assign each SP shard a unique constant value so that after gather, the
output along dim=2 must be:

    rows  [0..31]    : shard 0 value
    rows [32..63]   : shard 1 value
    rows [64..95]   : shard 2 value
    rows [96..127]  : shard 3 value

Failure modes this test catches:
  - shard 3 silently aliased to shard 2 (the "mesh of 4 voices" symptom)
  - shard ordering reversed
  - any device disagreeing with another about the gathered content
  - ping-pong buffer pollution across repeated calls
  - dtype mismatch causing partial overwrite of the persistent output buffer
"""


import pytest
import torch

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor, float32_tensor
from models.tt_dit.utils.test import line_params

_SHARD_VALUES = (10.0, 20.0, 30.0, 40.0)


def _build_input(sp_factor: int, dtype_torch: torch.dtype) -> torch.Tensor:
    """Return shape (1, 1, sp_factor*32, 128) where rows [k*32:(k+1)*32] == _SHARD_VALUES[k]."""
    assert sp_factor <= len(_SHARD_VALUES)
    N = sp_factor * 32
    D = 128
    x = torch.zeros(1, 1, N, D, dtype=dtype_torch)
    for k in range(sp_factor):
        x[0, 0, k * 32 : (k + 1) * 32, :] = _SHARD_VALUES[k]
    return x


def _check_shard_layout(host: torch.Tensor, sp_factor: int, label: str) -> None:
    """Each 32-row band along dim 2 must be a uniform shard value."""
    assert host.shape == (
        1,
        1,
        sp_factor * 32,
        128,
    ), f"{label}: expected shape (1,1,{sp_factor*32},128), got {tuple(host.shape)}"
    for k in range(sp_factor):
        band = host[0, 0, k * 32 : (k + 1) * 32, :].float()
        expected = _SHARD_VALUES[k]
        # First check that the band is uniform — otherwise we have intra-shard corruption
        band_min = band.min().item()
        band_max = band.max().item()
        # bf16 round-trip is exact for these small integer values
        if not (band_min == band_max):
            raise AssertionError(
                f"{label}: shard {k} is not uniform (min={band_min}, max={band_max}) — intra-shard corruption"
            )
        # Then check it equals the expected value
        if band_min != expected:
            # Diagnose which other shard it equals (the "shard 3 aliased to shard 2" symptom)
            culprit = None
            for j, v in enumerate(_SHARD_VALUES[:sp_factor]):
                if band_min == v:
                    culprit = j
                    break
            raise AssertionError(
                f"{label}: shard {k} = {band_min}, expected {expected}; " f"matches shard {culprit} (aliasing!)"
                if culprit is not None
                else f"{label}: shard {k} = {band_min}, expected {expected} (garbage)"
            )


def _gather_all_devices(gathered: ttnn.Tensor) -> list[torch.Tensor]:
    """Read the gathered tensor from EVERY device in the mesh."""
    return [ttnn.to_torch(t).float().clone() for t in ttnn.get_device_tensors(gathered)]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "dtype_torch, tensor_fn, dtype_name",
    [
        (torch.float32, float32_tensor, "float32"),
        (torch.bfloat16, bf16_tensor, "bfloat16"),
    ],
    ids=["float32", "bfloat16"],
)
def test_audio_velocity_sp_gather_shape_1_1_32_128(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
    dtype_torch,
    tensor_fn,
    dtype_name,
):
    """The EXACT audio velocity SP gather: per-device (1,1,32,128) -> (1,1,128,128)."""
    sp_factor = mesh_shape[sp_axis]
    assert sp_factor == 4, f"this test pins SP=4 (bh_2x4sp1tp0); got {sp_factor}"

    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    pc = DiTParallelConfig(
        sequence_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )

    x_full = _build_input(sp_factor, dtype_torch)

    # Push as sharded on SP axis along dim=2: each device gets one 32-row band
    tt_sharded = tensor_fn(
        x_full,
        device=mesh_device,
        mesh_axis=sp_axis,
        shard_dim=2,
    )

    # Validate per-device input is what we think it is.
    # Mesh is (TP_size, SP_size) on axes (0, 1). Replicated on TP, sharded on SP.
    # Linearised device order = TP-major: device_i = mesh_axis0_idx * SP + mesh_axis1_idx
    sp_axis_size = mesh_shape[sp_axis]
    per_dev_inputs = _gather_all_devices(tt_sharded)
    for k, dev_t in enumerate(per_dev_inputs):
        sp_shard_idx = k % sp_axis_size if sp_axis == 1 else k // mesh_shape[1]
        if dev_t.shape != (1, 1, 32, 128):
            raise AssertionError(
                f"[{dtype_name}] device {k} (sp shard {sp_shard_idx}) input shape "
                f"{tuple(dev_t.shape)} != (1,1,32,128)"
            )
        v_min = dev_t.min().item()
        v_max = dev_t.max().item()
        expected = _SHARD_VALUES[sp_shard_idx]
        if v_min != expected or v_max != expected:
            raise AssertionError(
                f"[{dtype_name}] device {k} (sp shard {sp_shard_idx}) pre-gather not "
                f"uniform at expected value {expected} (min={v_min}, max={v_max})"
            )

    # Now do the SP AllGather along dim=2 — exactly what inner_step does for audio_out
    gathered = ccl.all_gather_persistent_buffer(tt_sharded, dim=2, mesh_axis=sp_axis)
    per_dev_gathered = _gather_all_devices(gathered)

    # Every device should see the SAME (1,1,128,128) layout
    for k, host in enumerate(per_dev_gathered):
        _check_shard_layout(host, sp_factor, f"[{dtype_name}] dev{k} call1")

    # And they should all be bitwise-identical to each other
    ref = per_dev_gathered[0]
    for k in range(1, len(per_dev_gathered)):
        if not torch.equal(ref, per_dev_gathered[k]):
            diff = (ref - per_dev_gathered[k]).abs().max().item()
            raise AssertionError(f"[{dtype_name}] dev0 != dev{k} after AllGather (max abs diff {diff})")


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_audio_velocity_sp_gather_ping_pong_buffer_no_stale_data(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
):
    """Repeatedly run the EXACT audio velocity SP gather with DIFFERENT shard values.

    The persistent output buffer ping-pongs between two pre-allocated DRAM buffers
    initialised with ``torch.empty`` (garbage). If the kernel ever fails to
    overwrite a position in the output buffer, we'll see stale values from a
    PRIOR call leak in. This is the most realistic way to detect silent aliasing
    that only manifests with the ping-pong reuse the pipeline actually does.
    """
    sp_factor = mesh_shape[sp_axis]
    assert sp_factor == 4
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    # Use 8 distinct value sets so two consecutive calls always differ in EVERY shard
    value_sets = [
        (10.0, 20.0, 30.0, 40.0),
        (-1.0, -2.0, -3.0, -4.0),
        (100.0, 200.0, 300.0, 400.0),
        (7.0, 14.0, 21.0, 28.0),
        (-100.0, -200.0, -300.0, -400.0),
        (1.0, 2.0, 3.0, 4.0),
        (-7.0, -14.0, -21.0, -28.0),
        (50.0, 60.0, 70.0, 80.0),
    ]

    for call_idx, vals in enumerate(value_sets):
        N = sp_factor * 32
        D = 128
        x_full = torch.zeros(1, 1, N, D, dtype=torch.float32)
        for k, v in enumerate(vals):
            x_full[0, 0, k * 32 : (k + 1) * 32, :] = v

        tt_sharded = float32_tensor(
            x_full,
            device=mesh_device,
            mesh_axis=sp_axis,
            shard_dim=2,
        )
        gathered = ccl.all_gather_persistent_buffer(tt_sharded, dim=2, mesh_axis=sp_axis)
        per_dev = _gather_all_devices(gathered)

        for dev_k, host in enumerate(per_dev):
            if host.shape != (1, 1, 128, 128):
                raise AssertionError(f"call {call_idx} dev{dev_k}: gathered shape {tuple(host.shape)} != (1,1,128,128)")
            for shard_k in range(sp_factor):
                band = host[0, 0, shard_k * 32 : (shard_k + 1) * 32, :].float()
                expected = vals[shard_k]
                band_min = band.min().item()
                band_max = band.max().item()
                if band_min != band_max:
                    raise AssertionError(
                        f"call {call_idx} dev{dev_k}: shard {shard_k} non-uniform " f"(min={band_min}, max={band_max})"
                    )
                if band_min != expected:
                    # Which value set + shard does this match? Detects stale-data leaks.
                    leak = None
                    for prior_idx, prior_vals in enumerate(value_sets[:call_idx]):
                        for j, v in enumerate(prior_vals):
                            if band_min == v:
                                leak = (prior_idx, j)
                                break
                        if leak is not None:
                            break
                    cur_match = None
                    for j, v in enumerate(vals):
                        if band_min == v:
                            cur_match = j
                            break
                    raise AssertionError(
                        f"call {call_idx} dev{dev_k}: shard {shard_k} = {band_min}, "
                        f"expected {expected}; "
                        + (
                            f"matches PRIOR call {leak[0]} shard {leak[1]} -- ping-pong buffer stale data leak"
                            if leak is not None
                            else (
                                f"matches CURRENT call shard {cur_match} -- intra-call shard aliasing"
                                if cur_match is not None
                                else "garbage"
                            )
                        )
                    )


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_audio_attn1_kv_sp_gather_shape_1_16_32_64(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
):
    """SP AllGather on the EXACT K/V shape from audio_attn1 gather+SDPA path.

    Audio_attn1 K/V per-device shape after TP shard: (1, 16, 32, 64)
        - batch = 1
        - heads-per-TP = 32 // TP=2 = 16
        - N_per_SP = 128 // SP=4 = 32 tokens
        - head_dim = 64

    AllGather on dim=2 (sequence) -> (1, 16, 128, 64) per device.

    To catch any head mis-routing or shard mis-routing, we use a value pattern
    that's unique per (head, seq_pos):  value = head_idx * 1000 + seq_idx
    Then every gathered K/V row must read back exactly that value.
    """
    sp_factor = mesh_shape[sp_axis]
    assert sp_factor == 4
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    H = 16  # heads-per-TP
    N_per_sp = 32
    N = sp_factor * N_per_sp  # 128
    D = 64

    # value[h, n] = h*1000 + n  (broadcast across D head_dim)
    x_full = torch.zeros(1, H, N, D, dtype=torch.float32)
    for h in range(H):
        for n in range(N):
            x_full[0, h, n, :] = float(h * 1000 + n)

    tt_sharded = float32_tensor(x_full, device=mesh_device, mesh_axis=sp_axis, shard_dim=2)
    gathered = ccl.all_gather_persistent_buffer(tt_sharded, dim=2, mesh_axis=sp_axis)
    per_dev = _gather_all_devices(gathered)

    for dev_k, host in enumerate(per_dev):
        if host.shape != (1, H, N, D):
            raise AssertionError(f"dev{dev_k}: gathered shape {tuple(host.shape)} != (1,{H},{N},{D})")
        for h in range(H):
            for n in range(N):
                row = host[0, h, n, :].float()
                row_min = row.min().item()
                row_max = row.max().item()
                expected = float(h * 1000 + n)
                if row_min != row_max:
                    raise AssertionError(
                        f"dev{dev_k} head{h} pos{n}: non-uniform across head_dim " f"(min={row_min}, max={row_max})"
                    )
                if row_min != expected:
                    # Figure out which (h', n') it actually matches
                    if row_min >= 0 and row_min < H * 1000 + N:
                        h_act = int(row_min) // 1000
                        n_act = int(row_min) % 1000
                        raise AssertionError(
                            f"dev{dev_k} head{h} pos{n}: got {row_min} "
                            f"(matches head{h_act} pos{n_act} -- gather mis-routing!)"
                        )
                    raise AssertionError(f"dev{dev_k} head{h} pos{n}: got {row_min} (out of range; garbage)")


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "q_chunk_size, k_chunk_size, label",
    [
        (256, 256, "production_256x256_chunk_bigger_than_Sq_and_Sk"),
        (32, 128, "matched_chunks_q32_k128"),
        (32, 32, "matched_chunks_q32_k32"),
    ],
    ids=["chunk256x256", "chunk32x128", "chunk32x32"],
)
def test_audio_attn1_sdpa_chunk_size_vs_tensor_shape(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
    q_chunk_size,
    k_chunk_size,
    label,
):
    """SDPA with audio_attn1 shapes: per-device Q (1, 16, 32, 64), K/V (1, 16, 128, 64).

    The production code in ``LTXAttention.__init__`` hardcodes ``q_chunk_size=256,
    k_chunk_size=256`` for ALL self-attention (including audio). Audio Q has only
    32 tokens per shard and K only 128 -- both SMALLER than the chunk sizes.

    The SDPA kernel internally pads up to ``ceil(Sq / q_chunk) * q_chunk`` so it
    processes more virtual chunks than valid data. We test correctness vs a CPU
    reference torch SDPA to see whether the larger-than-needed chunk sizes
    silently produce subtly wrong results -- which would be a strong candidate
    for the "mesh of 4 voices" symptom.

    Comparing chunks 256x256 (production) vs 32x128 (matched) vs 32x32 (minimum)
    isolates whether the chunk-size>tensor-dim case is broken.
    """
    import os

    from models.tt_dit.utils.tensor import bf16_tensor

    if mesh_shape[sp_axis] != 4:
        pytest.skip("test is SP=4 specific")

    # Skip if -k filter excludes by label
    selected = os.environ.get("PYTEST_CURRENT_TEST", "")
    H = 16
    Sq = 32
    Sk = 128
    D = 64

    torch.manual_seed(123)
    q = torch.randn(1, H, Sq, D, dtype=torch.float32)
    k = torch.randn(1, H, Sk, D, dtype=torch.float32)
    v = torch.randn(1, H, Sk, D, dtype=torch.float32)

    # Audio padding mask: -inf in last 2 cols (positions [126,127] = padded keys)
    audio_N_real = 126
    mask_2d = torch.zeros(1, 1, Sq, Sk, dtype=torch.float32)
    mask_2d[:, :, :, audio_N_real:] = float("-inf")

    # CPU reference using float32
    scale = D**-0.5
    attn_scores = (q @ k.transpose(-2, -1)) * scale + mask_2d
    attn_probs = torch.softmax(attn_scores, dim=-1)
    ref = attn_probs @ v  # (1, H, Sq, D)

    # Build TT tensors on the mesh (NOT sharded; we want a clean SDPA test in isolation)
    q_tt = bf16_tensor(q.bfloat16(), device=mesh_device)
    k_tt = bf16_tensor(k.bfloat16(), device=mesh_device)
    v_tt = bf16_tensor(v.bfloat16(), device=mesh_device)
    mask_tt = bf16_tensor(mask_2d.bfloat16(), device=mesh_device)

    full_grid = mesh_device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=full_grid,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    out = ttnn.transformer.scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        attn_mask=mask_tt,
        is_causal=False,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    out_host = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    assert out_host.shape == ref.shape, f"[{label}] output shape {tuple(out_host.shape)} != expected {tuple(ref.shape)}"

    # Per-token max abs diff and PCC -- focus on tokens near SP boundaries
    diff = (out_host - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    # Per-token max abs (sum over heads & D)
    per_token = diff.amax(dim=(0, 1, 3))  # (Sq,)
    print(f"\n[{label}] max_abs={max_abs:.4f} mean_abs={mean_abs:.4f}")
    print(f"[{label}] per-Q-token max abs (first 8):  {per_token[:8].tolist()}")
    print(f"[{label}] per-Q-token max abs (mid 8):    {per_token[12:20].tolist()}")
    print(f"[{label}] per-Q-token max abs (last 8):   {per_token[-8:].tolist()}")

    # PCC vs reference
    a = out_host.flatten().double()
    b = ref.flatten().double()
    pcc = (torch.dot(a - a.mean(), b - b.mean()) / ((a - a.mean()).norm() * (b - b.mean()).norm() + 1e-12)).item()
    print(f"[{label}] PCC vs CPU-fp32 reference: {pcc:.6f}")
    # bf16 SDPA should hit at least 0.99 PCC on a correct kernel for this size
    if pcc < 0.99:
        raise AssertionError(f"[{label}] PCC {pcc:.6f} too low — kernel may be silently broken at this chunk config")


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_audio_velocity_sp_gather_per_token_pattern(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
):
    """Per-token unique value pattern: pos i has value i. Catches subtle in-tile permutations.

    Pre-gather each device sees its 32 tokens with values [dev*32 .. dev*32+31].
    Post-gather every device must see [0..127] along dim=2.
    """
    sp_factor = mesh_shape[sp_axis]
    assert sp_factor == 4
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    N = sp_factor * 32
    D = 128
    x_full = torch.zeros(1, 1, N, D, dtype=torch.float32)
    for i in range(N):
        x_full[0, 0, i, :] = float(i)

    tt_sharded = float32_tensor(x_full, device=mesh_device, mesh_axis=sp_axis, shard_dim=2)
    gathered = ccl.all_gather_persistent_buffer(tt_sharded, dim=2, mesh_axis=sp_axis)
    per_dev = _gather_all_devices(gathered)

    expected_idx = torch.arange(N, dtype=torch.float32)
    for dev_k, host in enumerate(per_dev):
        # Each row should be uniform across D
        for i in range(N):
            row = host[0, 0, i, :].float()
            row_min = row.min().item()
            row_max = row.max().item()
            if row_min != row_max:
                raise AssertionError(f"dev{dev_k} row {i}: non-uniform (min={row_min}, max={row_max})")
        idx = host[0, 0, :, 0].float()
        if not torch.equal(idx, expected_idx):
            # Where does the mismatch happen?  Often shard boundary 31->32, 63->64, 95->96
            mismatches = []
            for i in range(N):
                if idx[i].item() != float(i):
                    mismatches.append((i, idx[i].item()))
                if len(mismatches) >= 10:
                    break
            raise AssertionError(f"dev{dev_k}: token-order broken. First mismatches (idx, actual): {mismatches}")
