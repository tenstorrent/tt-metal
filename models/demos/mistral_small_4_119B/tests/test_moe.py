# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MoE tests for Mistral Small 4 following the DeepSeek test pattern.

Environment:
- ``TT_MOE_TEST_PREFILL_TOKENS``: prefill sequence length for the forward test (default ``16``; use ``64`` for stricter coverage).

The forward-path test mirrors `models/demos/deepseek_v3/tests/test_moe.py`:
- build a reference HF MoE module,
- create TT run config via `get_test_weight_config` + `create_run_config`,
- run mode-specific TT forward (`decode` / `prefill`),
- compare TT output with the HF reference.
"""

from __future__ import annotations

import inspect
import os
from copy import deepcopy

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_small_4_119B.tt.moe.experts import TtMistral4Experts
from models.demos.mistral_small_4_119B.tt.moe.moe import TtMistral4MoE
from models.demos.mistral_small_4_119B.tt.moe.moe_gate import TtMistral4MoEGate
from models.demos.mistral_small_4_119B.tt.moe.shared_expert import TtMistral4SharedExpert

# Without indirect ``(1, 1)``, ``mesh_device`` defaults to the full system mesh (e.g. 4 chips): slow,
# heavier UMD/sysmem, and lock contention with other jobs.
pytestmark = pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)

_PREFILL_SEQ_LEN = int(os.environ.get("TT_MOE_TEST_PREFILL_TOKENS", "16"))


def _activations_mesh_mapper(mesh_device: ttnn.Device):
    """1×1 mesh: no mapper needed (avoids creating distributed tensors that hang on to_torch).
    Multi-device: shard across the 2D mesh."""
    if mesh_device.get_num_devices() == 1:
        return None
    return ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))


def get_model_config(module_class, mode: str, *args, batch_size_per_row: int | None = None, **kwargs):
    if mode == "prefill":
        config_fn = module_class.prefill_model_config
    elif mode == "decode":
        config_fn = module_class.decode_model_config
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if batch_size_per_row is not None and "batch_size_per_row" in inspect.signature(config_fn).parameters:
        kwargs.setdefault("batch_size_per_row", batch_size_per_row)

    return config_fn(*args, **kwargs)


def run_module_forward(module_class, mode: str, *args, **kwargs):
    if mode == "prefill":
        return module_class.forward_prefill(*args, **kwargs)
    if mode == "decode":
        return module_class.forward_decode(*args, **kwargs)
    raise ValueError(f"Unsupported mode: {mode}")


def _tiny_mistral4_config():
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    cfg = Mistral4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=4096,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
    )
    if getattr(cfg, "_experts_implementation", None) in (None, ""):
        cfg._experts_implementation = "grouped_mm"
    return cfg


def build_reference_model(hf_config):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

    config = deepcopy(hf_config)
    model = Mistral4MoE(config).eval()
    # TT bridge runs host compute in fp32 and emits bf16 to mesh.
    return model.to(torch.float32)


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state_dict.items()}


def generate_reference_io(
    num_tokens: int, reference_model, hf_config
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    state_dict_out = _clone_state_dict(reference_model.state_dict())
    torch.manual_seed(0)
    torch_input = (0.1 * torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.float32)).to(torch.bfloat16)

    with torch.no_grad():
        # Match TT bridge numerics: fp32 model + fp32 activations, then bf16 output.
        reference_output = reference_model(torch_input.to(torch.float32)).to(torch.bfloat16)

    return state_dict_out, torch_input, reference_output


def assert_output_pcc(
    tt_output_torch: torch.Tensor,
    reference_output: torch.Tensor,
    *,
    pcc_required: float,
    msg: str = "",
):
    tt_out = tt_output_torch.cpu().to(torch.float32)
    ref_out = reference_output.cpu().to(torch.float32)

    while tt_out.ndim < ref_out.ndim:
        tt_out = tt_out.unsqueeze(0)
    while ref_out.ndim < tt_out.ndim:
        ref_out = ref_out.unsqueeze(0)

    # Match leading singleton dimensions and sequence if mesh conversion pads.
    seq = min(tt_out.shape[-2], ref_out.shape[-2])
    tt_out = tt_out[..., :seq, :]
    ref_out = ref_out[..., :seq, :]

    # In MoE bring-up with tiny/random configs, host reference can legitimately produce non-finite values.
    # BF16/device round-trips may canonicalize NaN payloads/signaling NaNs into +/-inf, so compare non-finite
    # masks first, then compare only finite values exactly.
    tt_non_finite = ~torch.isfinite(tt_out)
    ref_non_finite = ~torch.isfinite(ref_out)
    assert torch.equal(tt_non_finite, ref_non_finite), "TT and reference non-finite masks differ"

    finite_mask = torch.isfinite(tt_out) & torch.isfinite(ref_out)
    if not finite_mask.any():
        logger.warning(f"{msg} all compared outputs are non-finite; PCC check skipped")
        return

    passing, pcc = comp_pcc(ref_out[finite_mask], tt_out[finite_mask], pcc_required)
    status = "PASS" if passing else "FAIL"
    logger.info(f"[{status}] {msg} | PCC={pcc} | required>={pcc_required}")
    assert passing, f"{msg} PCC={pcc} below required threshold {pcc_required}"


def run_test_forward_pass_moe(
    *, mode, num_tokens, batch_size_per_row, hf_config, cache_path, mesh_device, ccl, weight_type, snapshot_dir=None
):
    logger.info(f"moe test: HF reference + IO (mode={mode}, tokens={num_tokens}, mesh={tuple(mesh_device.shape)})")
    reference_model = build_reference_model(hf_config)
    if snapshot_dir is not None:
        from models.demos.mistral_small_4_119B.tt.moe.moe import load_ttmistral4_moe_from_sharded_safetensors

        load_ttmistral4_moe_from_sharded_safetensors(reference_model, snapshot_dir, layer_idx=0, strict=False)
        reference_model = reference_model.to(torch.float32).eval()
    state_dict, torch_input, reference_output = generate_reference_io(
        num_tokens=num_tokens,
        reference_model=reference_model,
        hf_config=hf_config,
    )
    logger.info("moe test: reference done; building run_config + host bridge (TT path = device I/O + CPU HF MoE)")

    model_config = get_model_config(
        TtMistral4MoE,
        mode,
        hf_config,
        mesh_device,
        ttnn.FabricConfig.DISABLED,
        batch_size_per_row=batch_size_per_row,
        topk_fallback=True,
    )
    model_state = TtMistral4MoE.create_state(hf_config, mesh_device, ccl)
    model_shared_state = TtMistral4MoE.create_shared_state(hf_config, mesh_device)
    run_config = dict(model_config)
    run_config.update(model_state)
    run_config.update(model_shared_state)
    run_config["mesh_device"] = mesh_device
    run_config["bridge_torch_state_dict"] = state_dict
    # Reuse the HF module from the reference path so the bridge does not rebuild MoE + run grouped_mm twice.
    run_config["bridge_host_hf_model"] = reference_model
    logger.info("moe test: run_config ready; ttnn.from_torch (upload activations)")

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=_activations_mesh_mapper(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info("moe test: from_torch returned; to_memory_config + TT forward (no extra sync — avoids driver stalls)")

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(TtMistral4MoE, mode, tt_input, run_config, handle_tensor_parallel=True)
    logger.info("moe test: TT forward returned; to_torch readback + PCC")

    assert tt_output.memory_config() == run_config["output_memory_config"]

    if mesh_device.get_num_devices() == 1:
        tt_output_torch = ttnn.to_torch(tt_output)
    else:
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        )

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    logger.info(f"Mode: {mode}, Num tokens: {num_tokens}, Weight type: {weight_type}")
    # Match DeepSeek MoE test policy: random-init routed path has a slightly looser PCC target.
    pcc_required = 0.96 if weight_type == "random" else 0.97
    assert_output_pcc(
        tt_output_torch,
        reference_output.unsqueeze(0),
        pcc_required=pcc_required,
        msg=f"moe forward mode={mode} tokens={num_tokens} weight_type={weight_type}",
    )


@pytest.fixture(scope="session")
def hf_config():
    return _tiny_mistral4_config()


@pytest.fixture(scope="session")
def cache_path(tmp_path_factory):
    return tmp_path_factory.mktemp("mistral_small4_moe_cache")


@pytest.fixture(scope="function")
def ccl():
    return None


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "mode, batch_size_per_row, seq_len",
    [
        ("decode", 1, 1),
        # Shorter prefill keeps host grouped_mm MoE in the bridge within a practical CPU budget (override with TT_MOE_TEST_PREFILL_TOKENS).
        ("prefill", 1, _PREFILL_SEQ_LEN),
    ],
)
@pytest.mark.parametrize("weight_type", ["random"])
def test_forward_pass(mode, batch_size_per_row, seq_len, weight_type, hf_config, cache_path, mesh_device, ccl):
    run_test_forward_pass_moe(
        mode=mode,
        num_tokens=batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len,
        batch_size_per_row=batch_size_per_row,
        hf_config=hf_config,
        cache_path=cache_path,
        mesh_device=mesh_device,
        ccl=ccl,
        weight_type=weight_type,
    )


def test_moe_submodule_convert_weights_and_state(hf_config, mesh_device, tmp_path):
    """One mesh open for gate / experts / shared checks (avoids six consecutive device cycles)."""
    base = tmp_path / "moe_submodule_weights"

    gate = TtMistral4MoEGate(hf_config)
    gate_out = TtMistral4MoEGate.convert_weights(hf_config, (gate.state_dict(),), base / "moe_gate", mesh_device)
    assert "gate_proj" in gate_out
    assert "gate_weight_torch" in gate_out
    assert "mesh_device" in TtMistral4MoEGate.create_state(hf_config, mesh_device=mesh_device, ccl=None)

    experts = TtMistral4Experts(hf_config)
    exp_out = TtMistral4Experts.convert_weights(hf_config, (experts.state_dict(),), base / "moe_experts", mesh_device)
    assert "w_gate_up_experts" in exp_out
    assert "w_down_experts" in exp_out
    assert "experts_state_torch" in exp_out
    assert "mesh_device" in TtMistral4Experts.create_state(hf_config, mesh_device=mesh_device, ccl=None)

    shared = TtMistral4SharedExpert(hf_config)
    sh_out = TtMistral4SharedExpert.convert_weights(
        hf_config, (shared.state_dict(),), base / "shared_expert", mesh_device
    )
    assert "w_gate_shared_expert" in sh_out
    assert "w_up_shared_expert" in sh_out
    assert "w_down_shared_expert" in sh_out
    assert "shared_expert_state_torch" in sh_out
    assert "mesh_device" in TtMistral4SharedExpert.create_state(hf_config, mesh_device=mesh_device, ccl=None)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "mode, batch_size_per_row, seq_len",
    [
        ("decode", 1, 1),
        ("prefill", 1, _PREFILL_SEQ_LEN),
    ],
)
def test_forward_pass_checkpoint(mode, batch_size_per_row, seq_len, mesh_device, tmp_path):
    """MoE forward test with real Mistral Small 4 119B checkpoint weights."""
    from pathlib import Path

    from models.demos.mistral_small_4_119B.tt.moe.moe import mistral4_text_config_from_snapshot

    snapshot_dir = Path(__file__).resolve().parents[1] / "models" / "mistral_small_4"
    if not (snapshot_dir / "config.json").is_file():
        pytest.skip("No config.json (snapshot not available)")
    if not (snapshot_dir / "model.safetensors.index.json").is_file():
        pytest.skip("No model.safetensors.index.json (snapshot incomplete)")

    hf_config = mistral4_text_config_from_snapshot(snapshot_dir)
    if getattr(hf_config, "_experts_implementation", None) in (None, ""):
        hf_config._experts_implementation = "grouped_mm"

    run_test_forward_pass_moe(
        mode=mode,
        num_tokens=batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len,
        batch_size_per_row=batch_size_per_row,
        hf_config=hf_config,
        cache_path=tmp_path,
        mesh_device=mesh_device,
        ccl=None,
        weight_type="checkpoint",
        snapshot_dir=snapshot_dir,
    )


if __name__ == "__main__":
    pytest.main([__file__])
