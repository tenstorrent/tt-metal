# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SDPA single-core correctness tests via file-based Python<->C++ communication.

Generates Q, K, V as bfloat16 binary files, computes PyTorch reference,
runs the C++ SDPA kernel, reads back device output, and compares.

Usage:
    # Run tests (generates temp files, runs C++, compares, cleans up):
    pytest generate_and_test_sdpa.py -v

    # Generate permanent input folders for IDE debugging (no C++ run):
    pytest generate_and_test_sdpa.py -v --save-inputs
    # Creates test_inputs/<test_id>/{q.bin, k.bin, v.bin, ref_output.bin, cmd.txt}
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from loguru import logger


# ---------------------------------------------------------------------------
# Default SDPA parameters (matching the C++ single-core example)
# ---------------------------------------------------------------------------
SQ_CHUNK_T = 7
SK_CHUNK_T = 16
HEAD_DIM_T = 4
SUBBLOCK_H = 1
MM_THROTTLE_LEVEL = 0
EXP_APPROX_MODE = 0
TILE = 32
SEED = 1234

PCC_THRESHOLD = 0.99

# Permanent output dir for --save-inputs (relative to this file)
TEST_INPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_inputs")


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def fa_rand(*shape):
    """Flash-attention style random data (from test_scaled_dot_product_attention_sprint.py)."""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def generate_inputs(data_mode, Sq, Sk, d):
    """Generate Q, K, V tensors for the given data mode."""
    if data_mode == "zeros":
        Q = torch.zeros(Sq, d)
        K = torch.zeros(Sk, d)
        V = torch.zeros(Sk, d)
    elif data_mode == "ones":
        Q = torch.ones(Sq, d)
        K = torch.ones(Sk, d)
        V = torch.ones(Sk, d)
    elif data_mode == "random":
        Q = fa_rand(Sq, d)
        K = fa_rand(Sk, d)
        V = fa_rand(Sk, d)
    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")
    return Q, K, V


# ---------------------------------------------------------------------------
# bf16 file I/O
# ---------------------------------------------------------------------------


def float32_to_bf16_bytes(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)


def bf16_bytes_to_float32(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.int16)).view(torch.bfloat16).float()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compute_pcc(ref: torch.Tensor, act: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors."""
    ref_flat = ref.flatten().double()
    act_flat = act.flatten().double()

    ref_centered = ref_flat - ref_flat.mean()
    act_centered = act_flat - act_flat.mean()

    num = (ref_centered * act_centered).sum()
    denom = ref_centered.norm() * act_centered.norm()

    if denom < 1e-12:
        # Both near-constant — check if they match (bf16 has ~0.01 precision)
        return 1.0 if (ref_flat - act_flat).abs().max() < 0.01 else 0.0

    return (num / denom).item()


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def find_binary():
    """Locate the metal_example_sdpa_single_core binary."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    candidates = [
        os.path.join(repo_root, "build", "programming_examples", "metal_example_sdpa_single_core"),
        os.path.join(
            repo_root, "build", "tt_metal", "programming_examples", "sdpa_single_core", "metal_example_sdpa_single_core"
        ),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        f"Cannot find metal_example_sdpa_single_core binary. "
        f"Build with: ./build_metal.sh --build-programming-example\n"
        f"Searched: {candidates}"
    )


# ---------------------------------------------------------------------------
# Input file saving (for --save-inputs)
# ---------------------------------------------------------------------------


def save_test_inputs(
    test_id,
    Q,
    K,
    V,
    ref_output,
    num_q_chunks,
    num_k_chunks,
    sq_chunk_t,
    sk_chunk_t,
    head_dim_t,
    subblock_h,
    mm_throttle_level,
    exp_approx_mode,
    padded_k_tiles=0,
):
    """Save input files and C++ command to a permanent directory."""
    out_dir = os.path.join(TEST_INPUTS_DIR, test_id)
    os.makedirs(out_dir, exist_ok=True)

    float32_to_bf16_bytes(Q).tofile(os.path.join(out_dir, "q.bin"))
    float32_to_bf16_bytes(K).tofile(os.path.join(out_dir, "k.bin"))
    float32_to_bf16_bytes(V).tofile(os.path.join(out_dir, "v.bin"))
    ref_output.numpy().tofile(os.path.join(out_dir, "ref_output.bin"))

    # Write the C++ command line for launch.json / manual use
    try:
        binary = find_binary()
    except FileNotFoundError:
        binary = "<build>/metal_example_sdpa_single_core"

    args = [
        "--test",
        out_dir,
        "--Sq_chunk_t",
        str(sq_chunk_t),
        "--Sk_chunk_t",
        str(sk_chunk_t),
        "--head_dim_t",
        str(head_dim_t),
        "--num_q_chunks",
        str(num_q_chunks),
        "--num_k_chunks",
        str(num_k_chunks),
        "--subblock_h",
        str(subblock_h),
        "--mm_throttle_level",
        str(mm_throttle_level),
        "--exp_approx_mode",
        str(int(exp_approx_mode)),
        "--padded_k_tiles",
        str(padded_k_tiles),
    ]

    import json

    with open(os.path.join(out_dir, "cmd.txt"), "w") as f:
        f.write(f"{binary} {' '.join(args)}\n")
        f.write(f"\n// launch.json args:\n")
        f.write(f'"args": {json.dumps(args)}\n')

    logger.info(f"Saved test inputs to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Core test driver
# ---------------------------------------------------------------------------


def run_sdpa_single_core_test(
    test_id,
    num_q_chunks,
    num_k_chunks,
    data_mode,
    save_inputs_only=False,
    sq_chunk_t=SQ_CHUNK_T,
    sk_chunk_t=SK_CHUNK_T,
    head_dim_t=HEAD_DIM_T,
    subblock_h=SUBBLOCK_H,
    mm_throttle_level=MM_THROTTLE_LEVEL,
    exp_approx_mode=EXP_APPROX_MODE,
    pcc_threshold=PCC_THRESHOLD,
    padded_k_tiles=0,
    kv_bf8b=False,
):
    torch.manual_seed(SEED)

    Sq = num_q_chunks * sq_chunk_t * TILE
    Sk = num_k_chunks * sk_chunk_t * TILE
    d = head_dim_t * TILE
    valid_Sk = Sk - padded_k_tiles * TILE  # actual K length before padding

    logger.info(f"Sq={Sq}, Sk={Sk}, valid_Sk={valid_Sk}, d={d}, data={data_mode}, padded_k_tiles={padded_k_tiles}")

    # --- Generate inputs ---
    # When padded: generate valid-sized K/V, then zero-pad to chunk-aligned size
    Q, K_valid, V_valid = generate_inputs(data_mode, Sq, valid_Sk, d)
    if padded_k_tiles > 0:
        K = F.pad(K_valid, (0, 0, 0, padded_k_tiles * TILE))
        V = F.pad(V_valid, (0, 0, 0, padded_k_tiles * TILE))
    else:
        K, V = K_valid, V_valid

    # Truncate to device precision for reference (use unpadded K/V)
    Q_ref = Q.to(torch.bfloat16).float()
    if kv_bf8b:
        # bfloat8_b truncation: round-trip through float8_e4m3 approximates bfp8_b precision
        K_ref = K_valid.to(torch.float8_e4m3fn).float()
        V_ref = V_valid.to(torch.float8_e4m3fn).float()
    else:
        K_ref = K_valid.to(torch.bfloat16).float()
        V_ref = V_valid.to(torch.bfloat16).float()

    # --- PyTorch reference (uses unpadded K/V) ---
    scale = 1.0 / (d**0.5)
    ref_output = (
        F.scaled_dot_product_attention(
            Q_ref.unsqueeze(0).unsqueeze(0),
            K_ref.unsqueeze(0).unsqueeze(0),
            V_ref.unsqueeze(0).unsqueeze(0),
            is_causal=False,
            scale=scale,
        )
        .squeeze(0)
        .squeeze(0)
    )

    logger.info(f"Reference output range: [{ref_output.min().item():.6f}, {ref_output.max().item():.6f}]")

    # --- Save-inputs mode: write to permanent dir and return ---
    if save_inputs_only:
        save_test_inputs(
            test_id,
            Q,
            K,
            V,
            ref_output,
            num_q_chunks,
            num_k_chunks,
            sq_chunk_t,
            sk_chunk_t,
            head_dim_t,
            subblock_h,
            mm_throttle_level,
            exp_approx_mode,
            padded_k_tiles,
        )
        pytest.skip("--save-inputs: files generated, skipping C++ execution")

    # --- Normal mode: run C++ and compare ---
    tmpdir = tempfile.mkdtemp(prefix="sdpa_test_")
    try:
        float32_to_bf16_bytes(Q).tofile(os.path.join(tmpdir, "q.bin"))
        float32_to_bf16_bytes(K).tofile(os.path.join(tmpdir, "k.bin"))
        float32_to_bf16_bytes(V).tofile(os.path.join(tmpdir, "v.bin"))

        binary = find_binary()
        cmd = [
            binary,
            "--test",
            tmpdir,
            "--Sq_chunk_t",
            str(sq_chunk_t),
            "--Sk_chunk_t",
            str(sk_chunk_t),
            "--head_dim_t",
            str(head_dim_t),
            "--num_q_chunks",
            str(num_q_chunks),
            "--num_k_chunks",
            str(num_k_chunks),
            "--subblock_h",
            str(subblock_h),
            "--mm_throttle_level",
            str(mm_throttle_level),
            "--exp_approx_mode",
            str(int(exp_approx_mode)),
            "--padded_k_tiles",
            str(padded_k_tiles),
            "--kv_bf8b",
            str(int(kv_bf8b)),
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"C++ stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"C++ stderr:\n{result.stderr}")
        assert result.returncode == 0, f"C++ binary exited with code {result.returncode}"

        device_output_path = os.path.join(tmpdir, "device_output.bin")
        assert os.path.isfile(device_output_path), f"Device output not found: {device_output_path}"

        device_raw = np.fromfile(device_output_path, dtype=np.uint16)
        assert device_raw.size == Sq * d, f"Expected {Sq * d} elements, got {device_raw.size}"

        device_output = bf16_bytes_to_float32(device_raw).reshape(Sq, d)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # --- Compare ---
    logger.info(f"Device output range: [{device_output.min().item():.6f}, {device_output.max().item():.6f}]")

    diff = (device_output - ref_output).abs()
    max_abs_err = diff.max().item()
    rmse = ((device_output - ref_output) ** 2).mean().sqrt().item()
    pcc = compute_pcc(ref_output, device_output)

    logger.info(f"PCC={pcc:.6f}  Max Abs Error={max_abs_err:.6f}  RMSE={rmse:.6f}")

    assert pcc > pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    # (num_q_chunks, num_k_chunks, data_mode, sk_chunk_t, padded_k_tiles, sq_chunk_t, subblock_h)
    (1, 1, "zeros", 16, 0, 7, 1),
    (1, 1, "ones", 16, 0, 7, 1),
    (1, 1, "random", 16, 0, 7, 1),
    (1, 5, "random", 16, 0, 7, 1),
    (3, 5, "random", 16, 0, 7, 1),
    # K=8 variants
    (1, 1, "random", 8, 0, 7, 1),
    (1, 5, "random", 8, 0, 7, 1),
    (3, 5, "random", 8, 0, 7, 1),
    # Padded K variants (last chunk partially valid)
    (1, 5, "random", 16, 4, 7, 1),
    (1, 5, "random", 8, 2, 7, 1),
    (3, 5, "random", 16, 8, 7, 1),
    # sbh=2 variants (Sq_chunk_t must be even)
    (1, 1, "random", 4, 0, 4, 2),
    (1, 5, "random", 4, 0, 4, 2),
    (1, 5, "random", 8, 0, 4, 2),
    (3, 5, "random", 4, 1, 4, 2),
    # WAN-like: sq9, sk16, 19 K chunks with 8 padded tiles (3 Q chunks for fast iteration)
    (3, 19, "random", 16, 8, 9, 1),
    # WAN-like: sq7, sk16, 5 K chunks with 6 padded tiles (non-subblock-aligned padding)
    (3, 5, "random", 16, 6, 7, 1),
    # Same as above but single Q chunk (isolates one Q iteration with pad6)
    (1, 5, "random", 16, 6, 7, 1),
    # Single K-chunk padded (isolates reduced path from SALAD)
    (1, 1, "random", 16, 8, 7, 1),
    # Two K-chunks padded (minimal SALAD: 1 standard + 1 reduced)
    (1, 2, "random", 16, 8, 7, 1),
]

TEST_IDS = [
    "1q_1k-zeros-sk16",
    "1q_1k-ones-sk16",
    "1q_1k-random-sk16",
    "1q_5k-random-sk16",
    "3q_5k-random-sk16",
    "1q_1k-random-sk8",
    "1q_5k-random-sk8",
    "3q_5k-random-sk8",
    "1q_5k-random-sk16-pad4",
    "1q_5k-random-sk8-pad2",
    "3q_5k-random-sk16-pad8",
    "1q_1k-random-sk4-sbh2",
    "1q_5k-random-sk4-sbh2",
    "1q_5k-random-sk8-sbh2",
    "3q_5k-random-sk4-sbh2-pad1",
    "3q_19k-random-sk16-pad8-sq9",
    "3q_5k-random-sk16-pad6",
    "1q_5k-random-sk16-pad6",
    "1q_1k-random-sk16-pad8",
    "1q_2k-random-sk16-pad8",
]


# ---------------------------------------------------------------------------
# Helper: run SDPA and return metrics (no assertion)
# ---------------------------------------------------------------------------


def _run_and_measure(
    num_q_chunks,
    num_k_chunks,
    data_mode,
    sq_chunk_t,
    sk_chunk_t,
    head_dim_t,
    subblock_h=SUBBLOCK_H,
    mm_throttle_level=MM_THROTTLE_LEVEL,
    exp_approx_mode=EXP_APPROX_MODE,
    padded_k_tiles=0,
):
    """Run SDPA and return (pcc, rmse, max_abs_err) without asserting."""
    torch.manual_seed(SEED)

    Sq = num_q_chunks * sq_chunk_t * TILE
    Sk = num_k_chunks * sk_chunk_t * TILE
    d = head_dim_t * TILE
    valid_Sk = Sk - padded_k_tiles * TILE

    logger.info(f"Sq={Sq}, Sk={Sk}, valid_Sk={valid_Sk}, d={d}, data={data_mode}, padded_k_tiles={padded_k_tiles}")

    Q, K_valid, V_valid = generate_inputs(data_mode, Sq, valid_Sk, d)
    if padded_k_tiles > 0:
        K = F.pad(K_valid, (0, 0, 0, padded_k_tiles * TILE))
        V = F.pad(V_valid, (0, 0, 0, padded_k_tiles * TILE))
    else:
        K, V = K_valid, V_valid

    Q_ref = Q.to(torch.bfloat16).float()
    K_ref = K_valid.to(torch.bfloat16).float()
    V_ref = V_valid.to(torch.bfloat16).float()

    scale = 1.0 / (d**0.5)
    ref_output = (
        F.scaled_dot_product_attention(
            Q_ref.unsqueeze(0).unsqueeze(0),
            K_ref.unsqueeze(0).unsqueeze(0),
            V_ref.unsqueeze(0).unsqueeze(0),
            is_causal=False,
            scale=scale,
        )
        .squeeze(0)
        .squeeze(0)
    )

    tmpdir = tempfile.mkdtemp(prefix="sdpa_pad_test_")
    try:
        float32_to_bf16_bytes(Q).tofile(os.path.join(tmpdir, "q.bin"))
        float32_to_bf16_bytes(K).tofile(os.path.join(tmpdir, "k.bin"))
        float32_to_bf16_bytes(V).tofile(os.path.join(tmpdir, "v.bin"))

        binary = find_binary()
        cmd = [
            binary,
            "--test",
            tmpdir,
            "--Sq_chunk_t",
            str(sq_chunk_t),
            "--Sk_chunk_t",
            str(sk_chunk_t),
            "--head_dim_t",
            str(head_dim_t),
            "--num_q_chunks",
            str(num_q_chunks),
            "--num_k_chunks",
            str(num_k_chunks),
            "--subblock_h",
            str(subblock_h),
            "--mm_throttle_level",
            str(mm_throttle_level),
            "--exp_approx_mode",
            str(int(exp_approx_mode)),
            "--padded_k_tiles",
            str(padded_k_tiles),
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"C++ stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"C++ stderr:\n{result.stderr}")
        assert result.returncode == 0, f"C++ binary exited with code {result.returncode}"

        device_output_path = os.path.join(tmpdir, "device_output.bin")
        assert os.path.isfile(device_output_path), f"Device output not found: {device_output_path}"

        device_raw = np.fromfile(device_output_path, dtype=np.uint16)
        expected_size = Sq * d
        assert device_raw.size == expected_size, f"Expected {expected_size} elements, got {device_raw.size}"
        device_output = bf16_bytes_to_float32(device_raw).reshape(Sq, d)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    diff = (device_output - ref_output).abs()
    max_abs_err = diff.max().item()
    rmse = ((device_output - ref_output) ** 2).mean().sqrt().item()
    pcc = compute_pcc(ref_output, device_output)

    logger.info(f"padded_k_tiles={padded_k_tiles}: PCC={pcc:.6f}  Max Abs Error={max_abs_err:.6f}  RMSE={rmse:.6f}")

    return pcc, rmse, max_abs_err


@pytest.mark.parametrize(
    "num_q_chunks, num_k_chunks, data_mode, sk_chunk_t, padded_k_tiles, sq_chunk_t, subblock_h",
    TEST_CASES,
    ids=TEST_IDS,
)
def test_sdpa_single_core(
    request, num_q_chunks, num_k_chunks, data_mode, sk_chunk_t, padded_k_tiles, sq_chunk_t, subblock_h
):
    test_id = request.node.callspec.id
    save_only = request.config.getoption("--save-inputs", default=False)
    run_sdpa_single_core_test(
        test_id,
        num_q_chunks,
        num_k_chunks,
        data_mode,
        save_inputs_only=save_only,
        sk_chunk_t=sk_chunk_t,
        padded_k_tiles=padded_k_tiles,
        sq_chunk_t=sq_chunk_t,
        subblock_h=subblock_h,
    )


# ---------------------------------------------------------------------------
# Padding impact study (sq_chunk_t=9, sk_chunk_t=16, head_dim_t=4)
# ---------------------------------------------------------------------------

PAD_STUDY_SQ_CHUNK_T = 9
PAD_STUDY_SK_CHUNK_T = 16
PAD_STUDY_HEAD_DIM_T = 4
PAD_STUDY_NUM_Q_CHUNKS = 1
PAD_STUDY_NUM_K_CHUNKS = 3

# Sweep from no padding to nearly-full-chunk padding
PADDING_LEVELS = [0, 1, 2, 4, 8, 12, 15]


@pytest.mark.parametrize(
    "padded_k_tiles",
    PADDING_LEVELS,
    ids=[f"pad{p}" for p in PADDING_LEVELS],
)
def test_padding_pcc_impact(padded_k_tiles):
    """Individual PCC check at each padding level (sq9_sk16_hd4)."""
    pcc, rmse, max_abs_err = _run_and_measure(
        num_q_chunks=PAD_STUDY_NUM_Q_CHUNKS,
        num_k_chunks=PAD_STUDY_NUM_K_CHUNKS,
        data_mode="random",
        sq_chunk_t=PAD_STUDY_SQ_CHUNK_T,
        sk_chunk_t=PAD_STUDY_SK_CHUNK_T,
        head_dim_t=PAD_STUDY_HEAD_DIM_T,
        padded_k_tiles=padded_k_tiles,
    )
    assert pcc > PCC_THRESHOLD, f"PCC {pcc:.6f} below threshold {PCC_THRESHOLD} at padded_k_tiles={padded_k_tiles}"


def test_padding_pcc_correlation():
    """
    Sweep all padding levels, report a summary table, and compute
    Pearson correlation between padding amount and PCC.

    Config: sq_chunk_t=9, sk_chunk_t=16, head_dim_t=4,
            num_q_chunks=1, num_k_chunks=3
    """
    results = []
    for pad in PADDING_LEVELS:
        pcc, rmse, max_abs_err = _run_and_measure(
            num_q_chunks=PAD_STUDY_NUM_Q_CHUNKS,
            num_k_chunks=PAD_STUDY_NUM_K_CHUNKS,
            data_mode="random",
            sq_chunk_t=PAD_STUDY_SQ_CHUNK_T,
            sk_chunk_t=PAD_STUDY_SK_CHUNK_T,
            head_dim_t=PAD_STUDY_HEAD_DIM_T,
            padded_k_tiles=pad,
        )
        results.append({"pad": pad, "pcc": pcc, "rmse": rmse, "max_abs_err": max_abs_err})

    # --- Summary table ---
    logger.info("=" * 70)
    logger.info("Padding PCC Impact Summary (sq_chunk_t=9, sk_chunk_t=16, head_dim_t=4)")
    logger.info(f"{'Padded tiles':>12} {'PCC':>10} {'RMSE':>12} {'Max Abs Err':>12}")
    logger.info("-" * 50)
    for r in results:
        logger.info(f"{r['pad']:>12} {r['pcc']:>10.6f} {r['rmse']:>12.6f} {r['max_abs_err']:>12.6f}")
    logger.info("=" * 70)

    # --- Assert all above threshold ---
    for r in results:
        assert (
            r["pcc"] > PCC_THRESHOLD
        ), f"PCC {r['pcc']:.6f} below threshold {PCC_THRESHOLD} at padded_k_tiles={r['pad']}"

    # --- Pearson correlation: padding amount vs PCC ---
    pads = torch.tensor([float(r["pad"]) for r in results])
    pccs = torch.tensor([r["pcc"] for r in results])

    if pads.std() > 0 and pccs.std() > 1e-12:
        corr = torch.corrcoef(torch.stack([pads, pccs]))[0, 1].item()
    else:
        corr = 0.0

    logger.info(f"Pearson correlation (padding vs PCC): {corr:.6f}")

    if corr < -0.5:
        logger.warning(
            f"Significant negative correlation ({corr:.4f}) detected: " f"PCC degrades as padding increases."
        )


# ---------------------------------------------------------------------------
# MLA-like configs (from test_flash_mla_prefill v2 path validation)
# Single-core uses head_dim_t for both K and V, so head_dim_t=vDHt.
# ---------------------------------------------------------------------------


def test_mla_vDHt16(request):
    """MLA Call 1 equivalent: head_dim_t=16 exercises DST batching in
    normalize_row_streaming and mul_block_bcast_cols_acc.
    Uses kv_bf8b=True to fit in L1 (matches MLA prefill's bfloat8_b KV).
    Maps to vDHt=16, Sq_chunk_t=4, Sk_chunk_t=8, sbh=2."""
    run_sdpa_single_core_test(
        "mla_vDHt16",
        num_q_chunks=2,
        num_k_chunks=4,
        data_mode="random",
        save_inputs_only=request.config.getoption("--save-inputs", default=False),
        sq_chunk_t=4,
        sk_chunk_t=8,
        head_dim_t=16,
        subblock_h=2,
        kv_bf8b=True,
    )


def test_mla_vDHt4(request):
    """MLA Call 2 equivalent: head_dim_t=4, Sq_chunk_t=4, Sk_chunk_t=8, sbh=2."""
    run_sdpa_single_core_test(
        "mla_vDHt4",
        num_q_chunks=2,
        num_k_chunks=4,
        data_mode="random",
        save_inputs_only=request.config.getoption("--save-inputs", default=False),
        sq_chunk_t=4,
        sk_chunk_t=8,
        head_dim_t=4,
        subblock_h=2,
    )


# ---------------------------------------------------------------------------
# Padded configs that fail in sdpa_standard_v2 (from test_sdpa_tt_padded)
# d128-s160-nkv1-nh8-noncausal-k512-q128: Sk=160 padded to 512 (11 padded tiles)
# Single-core Sq = num_q_chunks * sq_chunk_t * 32 = 256 (no Q padding),
# but the K padding (11/16 tiles) is the root cause of the v2 failure.
# b1 and b2 variants have identical per-head config; single-core is single-batch.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Wan2.2 real-weights test: downloads QKV projection weights from HuggingFace,
# projects random spatial input, saves Q/K/V bin files for the SDPA test.
# ---------------------------------------------------------------------------

# Wan2.2-14B 720p self-attention config
_WAN_DIM = 5120
_WAN_NUM_HEADS = 40
_WAN_HEAD_DIM = _WAN_DIM // _WAN_NUM_HEADS  # 128
_WAN_TARGET_SEQ = 75600  # 21×45×80 (Wan2.2 720p patch count)
_WAN_VALID_SK = -(-_WAN_TARGET_SEQ // TILE) * TILE  # 75616 (tile-aligned)
_WAN_SQ_CHUNK_T = 7
_WAN_SK_CHUNK_T = 16
_WAN_NUM_Q_CHUNKS = 1
_WAN_NUM_K_CHUNKS = -(-(_WAN_VALID_SK // TILE) // _WAN_SK_CHUNK_T)  # 148
_WAN_SK = _WAN_NUM_K_CHUNKS * _WAN_SK_CHUNK_T * TILE  # 75776
_WAN_PADDED_K_TILES = (_WAN_SK - _WAN_VALID_SK) // TILE  # 5
_WAN_HEAD_DIM_T = _WAN_HEAD_DIM // TILE  # 4
_WAN_SQ = _WAN_NUM_Q_CHUNKS * _WAN_SQ_CHUNK_T * TILE  # 224
_WAN_HF_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
_WAN_HF_SHARD = "transformer/diffusion_pytorch_model-00001-of-00012.safetensors"
_WAN_TENSOR_NAMES = [
    "blocks.0.attn1.to_q.weight",
    "blocks.0.attn1.to_q.bias",
    "blocks.0.attn1.to_k.weight",
    "blocks.0.attn1.to_k.bias",
    "blocks.0.attn1.to_v.weight",
    "blocks.0.attn1.to_v.bias",
    "blocks.0.attn1.norm_q.weight",
    "blocks.0.attn1.norm_k.weight",
]


def _download_wan_tensors():
    """Download specific tensors from the HF safetensors shard via HTTP range requests.

    Downloads only the needed bytes (~315 MB) rather than the full 4.9 GB shard.
    Returns a dict of {tensor_name: torch.Tensor}.
    """
    import struct
    import json

    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(_WAN_HF_REPO, _WAN_HF_SHARD)

    # Read safetensors header (first 8 bytes = header length, then JSON)
    resp = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=10)
    resp.raise_for_status()
    header_size = struct.unpack("<Q", resp.content)[0]

    resp = requests.get(url, headers={"Range": f"bytes=8-{8 + header_size - 1}"}, timeout=30)
    resp.raise_for_status()
    header = json.loads(resp.content)
    data_offset = 8 + header_size

    # Collect byte ranges for needed tensors
    ranges = []
    for name in _WAN_TENSOR_NAMES:
        info = header[name]
        start = data_offset + info["data_offsets"][0]
        end = data_offset + info["data_offsets"][1]
        ranges.append((start, end, name, info["shape"], info["dtype"]))
    ranges.sort(key=lambda x: x[0])

    # Merge adjacent ranges (gap < 1MB) into contiguous downloads
    groups = []
    cs, ce, ci = ranges[0][0], ranges[0][1], [ranges[0]]
    for r in ranges[1:]:
        if r[0] - ce < 1_000_000:
            ce = max(ce, r[1])
            ci.append(r)
        else:
            groups.append((cs, ce, ci))
            cs, ce, ci = r[0], r[1], [r]
    groups.append((cs, ce, ci))

    tensors = {}
    for gs, ge, items in groups:
        logger.info(f"Downloading {(ge - gs) / 1e6:.1f} MB ({len(items)} tensors) from {_WAN_HF_REPO}...")
        resp = requests.get(url, headers={"Range": f"bytes={gs}-{ge - 1}"}, timeout=300)
        resp.raise_for_status()
        buf = resp.content
        for start, end, name, shape, dtype in items:
            raw = buf[start - gs : end - gs]
            tensors[name] = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()

    return tensors


def generate_wan_real_weights_data(out_dir):
    """Generate Q/K/V test data using real Wan2.2-14B layer-0 self-attention weights.

    Downloads QKV projection weights (~315 MB via HTTP range requests), projects
    random spatial input through them, applies RMSNorm, extracts head 0, and saves
    the resulting Q/K/V as bfloat16 bin files plus a float32 PyTorch SDPA reference.

    Args:
        out_dir: Directory to write {q.bin, k.bin, v.bin, ref_output.bin, cmd.txt}.
    """
    import json as _json

    tensors = _download_wan_tensors()

    Wq = tensors["blocks.0.attn1.to_q.weight"]
    bq = tensors["blocks.0.attn1.to_q.bias"]
    Wk = tensors["blocks.0.attn1.to_k.weight"]
    bk = tensors["blocks.0.attn1.to_k.bias"]
    Wv = tensors["blocks.0.attn1.to_v.weight"]
    bv = tensors["blocks.0.attn1.to_v.bias"]
    norm_q_w = tensors["blocks.0.attn1.norm_q.weight"]
    norm_k_w = tensors["blocks.0.attn1.norm_k.weight"]

    def rms_norm(x, weight, eps=1e-6):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    # Random spatial input, projected through real weights
    torch.manual_seed(42)
    x_full = torch.randn(1, _WAN_VALID_SK, _WAN_DIM, dtype=torch.float32)

    logger.info(f"Projecting Q ({_WAN_SQ} tokens)...")
    Q = x_full[:, :_WAN_SQ, :] @ Wq.T + bq
    logger.info(f"Projecting K ({_WAN_VALID_SK} tokens)...")
    K = x_full @ Wk.T + bk
    logger.info(f"Projecting V ({_WAN_VALID_SK} tokens)...")
    V = x_full @ Wv.T + bv

    logger.info("Applying RMSNorm to Q, K...")
    Q = rms_norm(Q, norm_q_w)
    K = rms_norm(K, norm_k_w)

    # Reshape to heads, take head 0, squeeze batch
    Q = Q.reshape(1, _WAN_SQ, _WAN_NUM_HEADS, _WAN_HEAD_DIM)[:, :, 0, :].squeeze(0)
    K_valid = K.reshape(1, _WAN_VALID_SK, _WAN_NUM_HEADS, _WAN_HEAD_DIM)[:, :, 0, :].squeeze(0)
    V_valid = V.reshape(1, _WAN_VALID_SK, _WAN_NUM_HEADS, _WAN_HEAD_DIM)[:, :, 0, :].squeeze(0)

    # Truncate to bf16 (device precision)
    Q = Q.to(torch.bfloat16).float()
    K_valid = K_valid.to(torch.bfloat16).float()
    V_valid = V_valid.to(torch.bfloat16).float()

    # Zero-pad K/V for chunk alignment
    K_padded = F.pad(K_valid, (0, 0, 0, _WAN_PADDED_K_TILES * TILE))
    V_padded = F.pad(V_valid, (0, 0, 0, _WAN_PADDED_K_TILES * TILE))
    assert K_padded.shape[0] == _WAN_SK

    # PyTorch reference (unpadded)
    logger.info("Computing reference SDPA...")
    scale = 1.0 / (_WAN_HEAD_DIM**0.5)
    ref_output = (
        F.scaled_dot_product_attention(
            Q.unsqueeze(0).unsqueeze(0),
            K_valid.unsqueeze(0).unsqueeze(0),
            V_valid.unsqueeze(0).unsqueeze(0),
            is_causal=False,
            scale=scale,
        )
        .squeeze(0)
        .squeeze(0)
    )

    logger.info(f"Q: {Q.shape}, range [{Q.min():.4f}, {Q.max():.4f}]")
    logger.info(f"K_valid: {K_valid.shape}, K_padded: {K_padded.shape}")
    logger.info(f"Reference: {ref_output.shape}, range [{ref_output.min():.6f}, {ref_output.max():.6f}]")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    float32_to_bf16_bytes(Q).tofile(os.path.join(out_dir, "q.bin"))
    float32_to_bf16_bytes(K_padded).tofile(os.path.join(out_dir, "k.bin"))
    float32_to_bf16_bytes(V_padded).tofile(os.path.join(out_dir, "v.bin"))
    ref_output.numpy().tofile(os.path.join(out_dir, "ref_output.bin"))

    try:
        binary = find_binary()
    except FileNotFoundError:
        binary = "<build>/metal_example_sdpa_single_core"

    args = [
        "--test",
        out_dir,
        "--Sq_chunk_t",
        str(_WAN_SQ_CHUNK_T),
        "--Sk_chunk_t",
        str(_WAN_SK_CHUNK_T),
        "--head_dim_t",
        str(_WAN_HEAD_DIM_T),
        "--num_q_chunks",
        str(_WAN_NUM_Q_CHUNKS),
        "--num_k_chunks",
        str(_WAN_NUM_K_CHUNKS),
        "--subblock_h",
        "1",
        "--mm_throttle_level",
        "0",
        "--exp_approx_mode",
        "0",
        "--padded_k_tiles",
        str(_WAN_PADDED_K_TILES),
    ]
    with open(os.path.join(out_dir, "cmd.txt"), "w") as f:
        f.write(f"{binary} {' '.join(args)}\n")
        f.write(f'\n// launch.json args:\n"args": {_json.dumps(args)}\n')

    logger.info(f"Saved wan_real_weights data to {out_dir}/")


def test_wan_real_weights(request):
    """Wan2.2-14B 720p self-attention dimensions with real pretrained QKV projection weights.

    Q/K/V were generated by projecting random spatial input through the actual
    Wan2.2-T2V-A14B layer-0 self-attention weights (to_q/to_k/to_v + RMSNorm),
    then taking head 0.

    Dimensions:
      - Q: [224, 128]  (1 chunk × Sq_chunk_t=7 tiles)
      - K/V: [75776, 128]  (148 chunks × Sk_chunk_t=16, padded from 75616 valid tokens)
      - head_dim_t=4, padded_k_tiles=5
      - 75616 = ceil(75600/32)*32 where 75600 = 21×45×80 (Wan2.2 720p patch count)

    Data generation (requires HuggingFace access, downloads ~315 MB of weights):
        pytest generate_and_test_sdpa.py --save-inputs -k wan_real_weights
    """
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_inputs", "wan_real_weights")

    save_only = request.config.getoption("--save-inputs", default=False)
    if save_only:
        generate_wan_real_weights_data(test_dir)
        pytest.skip("--save-inputs: wan_real_weights data generated")

    if not os.path.isdir(test_dir) or not os.path.isfile(os.path.join(test_dir, "q.bin")):
        pytest.skip(
            "wan_real_weights data not found. Generate with:\n"
            "  pytest generate_and_test_sdpa.py --save-inputs -k wan_real_weights"
        )

    Sq = _WAN_SQ
    Sk = _WAN_SK
    d = _WAN_HEAD_DIM

    # Load pre-generated reference
    ref_output = torch.from_numpy(
        np.fromfile(os.path.join(test_dir, "ref_output.bin"), dtype=np.float32).reshape(Sq, d)
    )

    logger.info(
        f"Wan real weights: Sq={Sq}, Sk={Sk}, " f"valid_Sk={_WAN_VALID_SK}, d={d}, padded_k_tiles={_WAN_PADDED_K_TILES}"
    )
    logger.info(f"Reference range: [{ref_output.min():.6f}, {ref_output.max():.6f}]")

    binary = find_binary()
    cmd = [
        binary,
        "--test",
        test_dir,
        "--Sq_chunk_t",
        str(_WAN_SQ_CHUNK_T),
        "--Sk_chunk_t",
        str(_WAN_SK_CHUNK_T),
        "--head_dim_t",
        str(_WAN_HEAD_DIM_T),
        "--num_q_chunks",
        str(_WAN_NUM_Q_CHUNKS),
        "--num_k_chunks",
        str(_WAN_NUM_K_CHUNKS),
        "--subblock_h",
        "1",
        "--mm_throttle_level",
        "0",
        "--exp_approx_mode",
        "0",
        "--padded_k_tiles",
        str(_WAN_PADDED_K_TILES),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"C++ stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"C++ stderr:\n{result.stderr}")
    assert result.returncode == 0, f"C++ binary exited with code {result.returncode}"

    device_output_path = os.path.join(test_dir, "device_output.bin")
    assert os.path.isfile(device_output_path), f"Device output not found: {device_output_path}"

    device_raw = np.fromfile(device_output_path, dtype=np.uint16)
    assert device_raw.size == Sq * d, f"Expected {Sq * d} elements, got {device_raw.size}"
    device_output = bf16_bytes_to_float32(device_raw).reshape(Sq, d)

    diff = (device_output - ref_output).abs()
    max_abs_err = diff.max().item()
    rmse = ((device_output - ref_output) ** 2).mean().sqrt().item()
    pcc = compute_pcc(ref_output, device_output)

    logger.info(f"PCC={pcc:.6f}  Max Abs Error={max_abs_err:.6f}  RMSE={rmse:.6f}")
    assert pcc > PCC_THRESHOLD, f"PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"


def test_padded_s160_k512_sbh2(request):
    """Reproduces d128-s160-k512-q128-noncausal failure: Sq_chunk_t=4, Sk_chunk_t=16,
    sbh=2, 11 padded K tiles out of 16 (Sk=160 padded to 512)."""
    run_sdpa_single_core_test(
        "padded_s160_k512_sbh2",
        num_q_chunks=2,
        num_k_chunks=1,
        data_mode="random",
        save_inputs_only=request.config.getoption("--save-inputs", default=False),
        sq_chunk_t=4,
        sk_chunk_t=16,
        head_dim_t=4,
        subblock_h=2,
        padded_k_tiles=11,
    )


def test_padded_s160_k512_sbh1(request):
    """Same padded config with sbh=1 (Sq_chunk_t=7) to isolate sbh from the mask issue."""
    run_sdpa_single_core_test(
        "padded_s160_k512_sbh1",
        num_q_chunks=2,
        num_k_chunks=1,
        data_mode="random",
        save_inputs_only=request.config.getoption("--save-inputs", default=False),
        sq_chunk_t=7,
        sk_chunk_t=16,
        head_dim_t=4,
        subblock_h=1,
        padded_k_tiles=11,
    )


# ---------------------------------------------------------------------------
# No-padding code size baseline
# ---------------------------------------------------------------------------


def test_nopad_codesize_baseline(request):
    """No-padding baseline: padded_k_tiles=0 so effective_Sk == Sk_chunk_t.
    The reduced-path template (call_step_reduced) is dead code — the constexpr
    condition `padded_k_tiles > 0` is false, so the compiler eliminates the
    branch and does NOT instantiate the narrower sdpa_inner_loop_step template.

    Compare the JIT-compiled trisc2 binary size from this test against a padded
    test (e.g. test_sdpa_single_core[3q_5k-random-sk16-pad6]) to measure the
    code size overhead of the skip-padding dual-path instantiation.

    Dimensions match 3q_5k-random-sk16-pad6 exactly, minus the padding."""
    run_sdpa_single_core_test(
        "nopad_codesize_baseline",
        num_q_chunks=3,
        num_k_chunks=5,
        data_mode="random",
        save_inputs_only=request.config.getoption("--save-inputs", default=False),
        sq_chunk_t=7,
        sk_chunk_t=16,
        head_dim_t=4,
        subblock_h=1,
        padded_k_tiles=0,
    )
