# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os
from pathlib import Path

import pytest
import torch
from diffusers import WanTransformer3DModel as TorchWanTransformer3DModel
from loguru import logger

import ttnn

from ....models.transformers.wan2_2.transformer_wan import WanTransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.padding import pad_vision_seq_parallel
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch
from ....utils.test import line_params

# ---------------------------------------------------------------------------
# Wan2.2-T2V-14B model configuration (mirrors test_transformer_wan.py)
# ---------------------------------------------------------------------------
MODEL_NAME = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DIM = 5120
FFN_DIM = 13824
NUM_HEADS = 40
HEAD_DIM = DIM // NUM_HEADS
IN_CHANNELS = 16
OUT_CHANNELS = 16
TEXT_DIM = 4096
FREQ_DIM = 256
NUM_LAYERS = 40
PATCH_SIZE = (1, 2, 2)
CROSS_ATTN_NORM = True
EPS = 1e-6
ROPE_MAX_SEQ_LEN = 1024

REFERENCE_DIR = Path(
    os.environ.get(
        "WAN_DETERMINISM_REF_DIR",
        Path(__file__).parent / "determinism_refs",
    )
)
REFERENCE_FILENAME = "transformer_block_ref_output.pt"

# Fixed test parameters matching "line_bh_4x8 and 14b-720p"
B = 1
T = 21
H = 90
W = 160
PROMPT_SEQ_LEN = 512
SP_AXIS = 1
TP_AXIS = 0
MESH_SHAPE = (4, 8)
NUM_LINKS = 2
IS_FSDP = False
TOPOLOGY = ttnn.Topology.Linear

NUM_ITERATIONS = 10
SUBMODULES = ["norm1", "attn1", "norm2", "attn2", "norm3", "ffn", "addcmul"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )


def _make_ccl_manager(mesh_device, num_links, topology):
    return CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)


def _compute_pcc_rmse(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Return (pcc, relative_rmse) between two tensors."""
    a = a.detach().flatten().to(torch.float64)
    b = b.detach().flatten().to(torch.float64)

    cov = torch.cov(torch.stack([a, b])).numpy()
    std_a = math.sqrt(cov[0, 0])
    std_b = math.sqrt(cov[1, 1])

    pcc = cov[0, 1] / (std_a * std_b)
    relative_rmse = torch.nn.functional.mse_loss(a, b).sqrt().item() / std_a
    return float(pcc), float(relative_rmse)


def _to_torch_dev0(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _create_submesh_and_model(mesh_device):
    """Load weights from HF, build TT model, create all input tensors on device.

    Returns (submesh, tt_model, parallel_config, ccl_manager, spatial_seq_len,
             tt_spatial, tt_prompt, tt_temb, tt_rope_cos, tt_rope_sin, tt_trans_mat,
             torch_model_block, torch_inputs).
    """
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*MESH_SHAPE))

    sp_factor = tuple(submesh.shape)[SP_AXIS]
    parallel_config = _make_parallel_config(submesh, SP_AXIS, TP_AXIS)
    ccl_manager = _make_ccl_manager(submesh, NUM_LINKS, TOPOLOGY)

    p_t, p_h, p_w = PATCH_SIZE
    spatial_seq_len = (T // p_t) * (H // p_h) * (W // p_w)

    parent_torch_model = TorchWanTransformer3DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=torch.float32, trust_remote_code=True
    )
    torch_model = parent_torch_model.blocks[0]
    torch_model.eval()

    tt_model = WanTransformerBlock(
        dim=DIM,
        ffn_dim=FFN_DIM,
        num_heads=NUM_HEADS,
        cross_attention_norm=CROSS_ATTN_NORM,
        eps=EPS,
        mesh_device=submesh,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=IS_FSDP,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch.manual_seed(0)
    spatial_input = torch.randn((B, spatial_seq_len, DIM), dtype=torch.float32)
    prompt_input = torch.randn((B, PROMPT_SEQ_LEN, DIM), dtype=torch.float32)
    temb_input = torch.randn((B, 6, DIM), dtype=torch.float32)

    rope_cos = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
    rope_sin = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
    torch_rope_cos, torch_rope_sin = stack_cos_sin(rope_cos, rope_sin)

    rope_cos_stack = torch_rope_cos.permute(0, 2, 1, 3)
    rope_sin_stack = torch_rope_sin.permute(0, 2, 1, 3)

    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    rope_cos_padded = pad_vision_seq_parallel(rope_cos_stack, num_devices=sp_factor)
    rope_sin_padded = pad_vision_seq_parallel(rope_sin_stack, num_devices=sp_factor)

    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=submesh, shard_mapping={SP_AXIS: 2, TP_AXIS: 3})
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=submesh)
    tt_temb = from_torch(temb_input.unsqueeze(0), device=submesh, dtype=ttnn.float32, mesh_axes=[..., TP_AXIS])
    tt_rope_cos = from_torch(rope_cos_padded, device=submesh, dtype=ttnn.float32, mesh_axes=[..., SP_AXIS, None])
    tt_rope_sin = from_torch(rope_sin_padded, device=submesh, dtype=ttnn.float32, mesh_axes=[..., SP_AXIS, None])
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=submesh)

    torch_inputs = {
        "spatial_input": spatial_input,
        "prompt_input": prompt_input,
        "temb_input": temb_input,
        "torch_rope_cos": torch_rope_cos,
        "torch_rope_sin": torch_rope_sin,
    }

    return (
        submesh,
        tt_model,
        parallel_config,
        ccl_manager,
        spatial_seq_len,
        tt_spatial,
        tt_prompt,
        tt_temb,
        tt_rope_cos,
        tt_rope_sin,
        tt_trans_mat,
        torch_model,
        torch_inputs,
    )


def _make_full_block_runner(
    submesh, tt_model, spatial_seq_len, tt_spatial, tt_prompt, tt_temb, tt_rope_cos, tt_rope_sin, tt_trans_mat
):
    """Return a closure that runs the full transformer block and converts output to torch."""

    def run_tt():
        tt_spatial_out = tt_model(
            spatial_1BND=tt_spatial,
            prompt_1BLP=tt_prompt,
            temb_1BTD=tt_temb,
            N=spatial_seq_len,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
            trans_mat=tt_trans_mat,
        )
        spatial_concat_dims = [None, None]
        spatial_concat_dims[SP_AXIS] = 2
        spatial_concat_dims[TP_AXIS] = 3
        out = ttnn.to_torch(
            tt_spatial_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(submesh, dims=spatial_concat_dims, mesh_shape=tuple(submesh.shape)),
        )
        return out[:, :, :spatial_seq_len, :]

    return run_tt


def _make_submodule_runners(
    tt_model,
    parallel_config,
    ccl_manager,
    spatial_seq_len,
    tt_spatial,
    tt_prompt,
    tt_temb,
    tt_rope_cos,
    tt_rope_sin,
    tt_trans_mat,
):
    """Run the forward pass once to capture intermediates, then return per-submodule runners.

    Each runner re-executes only its submodule with frozen inputs and returns
    the output as a torch tensor (from device 0).
    """
    # --- Replicate the temb preparation from WanTransformerBlock.forward ---
    shifted_temb_1BTD = tt_model.scale_shift_table.data + tt_temb
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = ttnn.chunk(shifted_temb_1BTD, 6, dim=2)
    gate_msa = ttnn.typecast(gate_msa, dtype=ttnn.bfloat16)
    c_gate_msa = ttnn.typecast(c_gate_msa, dtype=ttnn.bfloat16)

    norm1_weight = 1.0 + scale_msa
    norm3_weight = 1.0 + c_scale_msa

    # --- Run forward pass once to capture each submodule's inputs ---

    # norm1: input is tt_spatial
    norm1_out = tt_model.norm1(tt_spatial, dynamic_weight=norm1_weight, dynamic_bias=shift_msa)

    # attn1 (self-attn with fused residual addcmul): input is norm1_out + original spatial as residual
    attn1_out = tt_model.attn1(
        spatial_1BND=norm1_out,
        N=spatial_seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
        addcmul_residual=tt_spatial,
        addcmul_gate=gate_msa,
    )

    # norm2: input is attn1_out
    norm2_out = tt_model.norm2(attn1_out)

    # attn2 (cross-attn): input is norm2_out + prompt
    attn2_out = tt_model.attn2(
        spatial_1BND=norm2_out,
        N=spatial_seq_len,
        prompt_1BLP=tt_prompt,
    )
    spatial_after_cross = attn1_out + attn2_out

    # norm3: input is spatial_after_cross
    norm3_out = tt_model.norm3(spatial_after_cross, dynamic_weight=norm3_weight, dynamic_bias=c_shift_msa)

    # ffn: input is norm3_out (after TP all_gather if needed)
    if parallel_config.tensor_parallel.factor > 1:
        ffn_input = ccl_manager.all_gather_persistent_buffer(
            norm3_out, dim=3, mesh_axis=parallel_config.tensor_parallel.mesh_axis
        )
    else:
        ffn_input = norm3_out

    ffn_out = tt_model.ffn(ffn_input, compute_kernel_config=tt_model.ff_compute_kernel_config)

    # addcmul: inputs are spatial_after_cross, ffn_out, c_gate_msa
    # (output not needed for setup, just verifying the runners)

    # --- Build runner closures (each re-runs its submodule with frozen inputs) ---

    def run_norm1():
        return _to_torch_dev0(tt_model.norm1(tt_spatial, dynamic_weight=norm1_weight, dynamic_bias=shift_msa))

    def run_attn1():
        return _to_torch_dev0(
            tt_model.attn1(
                spatial_1BND=norm1_out,
                N=spatial_seq_len,
                rope_cos=tt_rope_cos,
                rope_sin=tt_rope_sin,
                trans_mat=tt_trans_mat,
                addcmul_residual=tt_spatial,
                addcmul_gate=gate_msa,
            )
        )

    def run_norm2():
        return _to_torch_dev0(tt_model.norm2(attn1_out))

    def run_attn2():
        return _to_torch_dev0(
            tt_model.attn2(
                spatial_1BND=norm2_out,
                N=spatial_seq_len,
                prompt_1BLP=tt_prompt,
            )
        )

    def run_norm3():
        return _to_torch_dev0(
            tt_model.norm3(spatial_after_cross, dynamic_weight=norm3_weight, dynamic_bias=c_shift_msa)
        )

    def run_ffn():
        return _to_torch_dev0(tt_model.ffn(ffn_input, compute_kernel_config=tt_model.ff_compute_kernel_config))

    def run_addcmul():
        return _to_torch_dev0(ttnn.addcmul(spatial_after_cross, ffn_out, c_gate_msa))

    return {
        "norm1": run_norm1,
        "attn1": run_attn1,
        "norm2": run_norm2,
        "attn2": run_attn2,
        "norm3": run_norm3,
        "ffn": run_ffn,
        "addcmul": run_addcmul,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param(MESH_SHAPE, line_params, id="line_bh_4x8")],
    indirect=["mesh_device", "device_params"],
)
def test_save_torch_reference(mesh_device: ttnn.MeshDevice, reset_seeds) -> None:
    """Run the transformer block once and save the torch reference output to disk."""
    (
        submesh,
        tt_model,
        parallel_config,
        ccl_manager,
        spatial_seq_len,
        tt_spatial,
        tt_prompt,
        tt_temb,
        tt_rope_cos,
        tt_rope_sin,
        tt_trans_mat,
        torch_model,
        torch_inputs,
    ) = _create_submesh_and_model(mesh_device)

    with torch.no_grad():
        torch_spatial_out = torch_model(
            hidden_states=torch_inputs["spatial_input"],
            encoder_hidden_states=torch_inputs["prompt_input"],
            temb=torch_inputs["temb_input"],
            rotary_emb=[torch_inputs["torch_rope_cos"], torch_inputs["torch_rope_sin"]],
        )

    run_tt = _make_full_block_runner(
        submesh, tt_model, spatial_seq_len, tt_spatial, tt_prompt, tt_temb, tt_rope_cos, tt_rope_sin, tt_trans_mat
    )
    tt_out = run_tt()
    pcc, rmse = _compute_pcc_rmse(torch_spatial_out, tt_out)
    logger.info(f"Sanity check — PCC = {pcc * 100:.4f} %, RMSE/σ = {rmse * 100:.1f} %")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = REFERENCE_DIR / REFERENCE_FILENAME
    torch.save(torch_spatial_out, ref_path)
    logger.info(f"Saved torch reference output to {ref_path}")


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param(MESH_SHAPE, line_params, id="line_bh_4x8")],
    indirect=["mesh_device", "device_params"],
)
def test_check_determinism(mesh_device: ttnn.MeshDevice, reset_seeds) -> None:
    """Run transformer block 10 times and assert PCC/RMSE are identical across iterations."""
    ref_path = REFERENCE_DIR / REFERENCE_FILENAME
    assert ref_path.exists(), f"Reference file not found at {ref_path}. " "Run test_save_torch_reference first."
    torch_ref = torch.load(ref_path, weights_only=True)

    (
        submesh,
        tt_model,
        parallel_config,
        ccl_manager,
        spatial_seq_len,
        tt_spatial,
        tt_prompt,
        tt_temb,
        tt_rope_cos,
        tt_rope_sin,
        tt_trans_mat,
        _torch_model,
        _torch_inputs,
    ) = _create_submesh_and_model(mesh_device)

    run_tt = _make_full_block_runner(
        submesh, tt_model, spatial_seq_len, tt_spatial, tt_prompt, tt_temb, tt_rope_cos, tt_rope_sin, tt_trans_mat
    )

    tt_out = run_tt()
    first_pcc, first_rmse = _compute_pcc_rmse(torch_ref, tt_out)
    logger.info(f"Iteration 0: PCC = {first_pcc * 100:.4f} %, RMSE/σ = {first_rmse * 100:.1f} %")

    for i in range(1, NUM_ITERATIONS):
        tt_out = run_tt()
        pcc, rmse = _compute_pcc_rmse(torch_ref, tt_out)
        logger.info(f"Iteration {i}: PCC = {pcc * 100:.4f} %, RMSE/σ = {rmse * 100:.1f} %")
        assert pcc == first_pcc, f"PCC mismatch at iteration {i}: {pcc} != {first_pcc} (iteration 0)"
        assert rmse == first_rmse, f"RMSE mismatch at iteration {i}: {rmse} != {first_rmse} (iteration 0)"

    logger.info(
        f"Determinism check passed: {NUM_ITERATIONS} iterations all produced "
        f"PCC = {first_pcc * 100:.4f} %, RMSE/σ = {first_rmse * 100:.1f} %"
    )


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param(MESH_SHAPE, line_params, id="line_bh_4x8")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("submodule", SUBMODULES)
def test_submodule_determinism(mesh_device: ttnn.MeshDevice, submodule: str, reset_seeds) -> None:
    """Run a single submodule 10 times with frozen inputs and assert identical outputs."""
    (
        submesh,
        tt_model,
        parallel_config,
        ccl_manager,
        spatial_seq_len,
        tt_spatial,
        tt_prompt,
        tt_temb,
        tt_rope_cos,
        tt_rope_sin,
        tt_trans_mat,
        _torch_model,
        _torch_inputs,
    ) = _create_submesh_and_model(mesh_device)

    runners = _make_submodule_runners(
        tt_model,
        parallel_config,
        ccl_manager,
        spatial_seq_len,
        tt_spatial,
        tt_prompt,
        tt_temb,
        tt_rope_cos,
        tt_rope_sin,
        tt_trans_mat,
    )
    run_fn = runners[submodule]

    first_out = run_fn()
    logger.info(f"[{submodule}] Iteration 0 output shape: {first_out.shape}")

    for i in range(1, NUM_ITERATIONS):
        out = run_fn()
        if not torch.equal(first_out, out):
            pcc, rmse = _compute_pcc_rmse(first_out, out)
            pytest.fail(
                f"[{submodule}] Non-deterministic output at iteration {i}: "
                f"PCC = {pcc * 100:.4f} %, RMSE/σ = {rmse * 100:.1f} %"
            )
        logger.info(f"[{submodule}] Iteration {i}: exact match with iteration 0")

    logger.info(f"[{submodule}] Determinism check passed: all {NUM_ITERATIONS} iterations identical")


# Ordered list of instrumented steps matching WanTransformerBlock.forward exactly.
INSTRUMENTED_STEPS = [
    "temb_prep",
    "norm1",
    "attn1",
    "norm2",
    "attn2",
    "cross_residual",
    "norm3",
    "tp_all_gather",
    "ffn",
    "addcmul",
]


def _run_instrumented_forward(
    tt_model,
    parallel_config,
    ccl_manager,
    spatial_seq_len,
    tt_spatial,
    tt_prompt,
    tt_temb,
    tt_rope_cos,
    tt_rope_sin,
    tt_trans_mat,
):
    """Run the transformer block step-by-step (live data flow) and snapshot every intermediate.

    Returns an ordered dict {step_name: torch_tensor_from_device_0}.
    """
    snapshots: dict[str, torch.Tensor] = {}

    # --- temb preparation ---
    shifted_temb_1BTD = tt_model.scale_shift_table.data + tt_temb
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = ttnn.chunk(shifted_temb_1BTD, 6, dim=2)
    gate_msa = ttnn.typecast(gate_msa, dtype=ttnn.bfloat16)
    c_gate_msa = ttnn.typecast(c_gate_msa, dtype=ttnn.bfloat16)
    snapshots["temb_prep"] = _to_torch_dev0(shifted_temb_1BTD)

    # --- norm1 ---
    spatial_normed = tt_model.norm1(tt_spatial, dynamic_weight=(1.0 + scale_msa), dynamic_bias=shift_msa)
    snapshots["norm1"] = _to_torch_dev0(spatial_normed)

    # --- attn1 (self-attention with fused residual addcmul) ---
    spatial = tt_model.attn1(
        spatial_1BND=spatial_normed,
        N=spatial_seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
        addcmul_residual=tt_spatial,
        addcmul_gate=gate_msa,
    )
    snapshots["attn1"] = _to_torch_dev0(spatial)

    # --- norm2 ---
    spatial_normed = tt_model.norm2(spatial)
    snapshots["norm2"] = _to_torch_dev0(spatial_normed)

    # --- attn2 (cross-attention) ---
    attn_output = tt_model.attn2(
        spatial_1BND=spatial_normed,
        N=spatial_seq_len,
        prompt_1BLP=tt_prompt,
    )
    snapshots["attn2"] = _to_torch_dev0(attn_output)

    # --- cross_residual: spatial + attn2 output ---
    spatial = spatial + attn_output
    snapshots["cross_residual"] = _to_torch_dev0(spatial)

    # --- norm3 ---
    spatial_normed = tt_model.norm3(spatial, dynamic_weight=(1.0 + c_scale_msa), dynamic_bias=c_shift_msa)
    snapshots["norm3"] = _to_torch_dev0(spatial_normed)

    # --- TP all_gather (before ffn) ---
    if parallel_config.tensor_parallel.factor > 1:
        spatial_normed = ccl_manager.all_gather_persistent_buffer(
            spatial_normed, dim=3, mesh_axis=parallel_config.tensor_parallel.mesh_axis
        )
    snapshots["tp_all_gather"] = _to_torch_dev0(spatial_normed)

    # --- ffn ---
    spatial_ff = tt_model.ffn(spatial_normed, compute_kernel_config=tt_model.ff_compute_kernel_config)
    snapshots["ffn"] = _to_torch_dev0(spatial_ff)

    # --- final addcmul ---
    spatial = ttnn.addcmul(spatial, spatial_ff, c_gate_msa)
    snapshots["addcmul"] = _to_torch_dev0(spatial)

    return snapshots


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param(MESH_SHAPE, line_params, id="line_bh_4x8")],
    indirect=["mesh_device", "device_params"],
)
def test_bisect_block_determinism(mesh_device: ttnn.MeshDevice, reset_seeds) -> None:
    """Run the full block step-by-step N times and find the first non-deterministic stage."""
    (
        submesh,
        tt_model,
        parallel_config,
        ccl_manager,
        spatial_seq_len,
        tt_spatial,
        tt_prompt,
        tt_temb,
        tt_rope_cos,
        tt_rope_sin,
        tt_trans_mat,
        _torch_model,
        _torch_inputs,
    ) = _create_submesh_and_model(mesh_device)

    ref_snapshots = _run_instrumented_forward(
        tt_model,
        parallel_config,
        ccl_manager,
        spatial_seq_len,
        tt_spatial,
        tt_prompt,
        tt_temb,
        tt_rope_cos,
        tt_rope_sin,
        tt_trans_mat,
    )
    logger.info("Iteration 0 (reference): captured snapshots for all steps")
    for name, tensor in ref_snapshots.items():
        logger.info(f"  {name}: shape {tensor.shape}")

    failures: list[str] = []

    for i in range(1, NUM_ITERATIONS):
        snapshots = _run_instrumented_forward(
            tt_model,
            parallel_config,
            ccl_manager,
            spatial_seq_len,
            tt_spatial,
            tt_prompt,
            tt_temb,
            tt_rope_cos,
            tt_rope_sin,
            tt_trans_mat,
        )

        first_divergent = None
        for step_name in INSTRUMENTED_STEPS:
            ref = ref_snapshots[step_name]
            cur = snapshots[step_name]
            if torch.equal(ref, cur):
                logger.info(f"  Iteration {i} | {step_name}: MATCH")
            else:
                pcc, rmse = _compute_pcc_rmse(ref, cur)
                logger.warning(
                    f"  Iteration {i} | {step_name}: MISMATCH — "
                    f"PCC = {pcc * 100:.4f} %, RMSE/σ = {rmse * 100:.1f} %"
                )
                if first_divergent is None:
                    first_divergent = step_name

        if first_divergent is not None:
            failures.append(f"Iteration {i}: first divergence at '{first_divergent}'")

    if failures:
        summary = "\n".join(failures)
        pytest.fail(f"Non-determinism detected in {len(failures)}/{NUM_ITERATIONS - 1} iterations.\n" f"{summary}")
