# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Qwen3-Coder-Next with TTNN backend (MoE + Gated Attention)."""

import os
from pathlib import Path

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.attention import TTNNQwen3NextGatedAttention
from models.experimental.tt_symbiote.modules.gated_deltanet import TTNNGatedDeltaNet
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.modules.moe import TTNNQwen3MoE
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

QWEN3_MODEL_ID = "Qwen/Qwen3-Coder-Next"

# Run only on T3K; set default so MESH_DEVICE is defined for MeshShapeToDeviceArch lookup
_MESH_DEVICE_ENV = "MESH_DEVICE"
if _MESH_DEVICE_ENV not in os.environ:
    os.environ[_MESH_DEVICE_ENV] = "T3K"
MESH_DEVICE = (os.environ.get(_MESH_DEVICE_ENV) or "T3K").upper()
MESH_SHAPE_T3K = (1, 8)


def _get_cached_model_path():
    """Resolve HF cache snapshot path for Qwen3-Coder-Next (avoids network)."""
    if hub := os.environ.get("HF_HUB_CACHE"):
        cache_root = Path(hub) / f"models--{QWEN3_MODEL_ID.replace('/', '--')}"
    else:
        base = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        cache_root = Path(base) / "hub" / f"models--{QWEN3_MODEL_ID.replace('/', '--')}"
    snapshots = cache_root / "snapshots"
    if not snapshots.exists():
        return None
    dirs = sorted(snapshots.iterdir())
    return str(dirs[0]) if dirs else None


def _load_qwen3_model():
    """Load tokenizer and model. Uses HF cache if available to avoid network."""
    try:
        cached = _get_cached_model_path()
        load_kw = dict(trust_remote_code=True, torch_dtype="auto", device_map="auto")
        if cached:
            tokenizer = AutoTokenizer.from_pretrained(cached, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(cached, local_files_only=True, **load_kw)
            return tokenizer, model
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL_ID, **load_kw)
        return tokenizer, model
    except ModuleNotFoundError as e:
        if "triton" in str(e).lower():
            pytest.skip("Qwen3-Coder-Next requires triton for loading")
        raise
    except Exception as e:
        msg = str(e).lower()
        if any(
            kw in msg
            for kw in (
                "network is unreachable",
                "name resolution",
                "connection",
                "protobuf",
                "does not appear to have",
            )
        ):
            pytest.skip(f"Model unavailable: {e}")
        raise


def _layer_output_to_torch(out):
    """Convert decoder layer output (torch or TorchTTNNTensor) to torch float32 for PCC."""
    if isinstance(out, torch.Tensor) and not isinstance(out, TorchTTNNTensor):
        return out.detach().to(torch.float32)
    if isinstance(out, TorchTTNNTensor):
        t = out.to_torch
        if callable(t):
            t = t()
        return t.detach().to(torch.float32)
    if hasattr(out, "to_torch"):
        t = getattr(out, "to_torch")
        return (t() if callable(t) else t).detach().to(torch.float32)
    return None


def _compute_pcc(torch_out, ttnn_out):
    """Compute PCC between reference (torch) and TTNN output. Returns (pcc, max_diff, mean_diff)."""
    t = torch_out.to(torch.float32).flatten()
    n = _layer_output_to_torch(ttnn_out)
    if n is None:
        return float("-inf"), float("inf"), float("inf")
    n = n.flatten()
    if t.numel() != n.numel():
        return float("-inf"), float("inf"), float("inf")
    pcc = torch.corrcoef(torch.stack([t, n]))[0, 1].item()
    diff = torch.abs(t - n)
    return pcc, torch.max(diff).item(), torch.mean(diff).item()


def _get_output_dtype(out):
    """Get dtype string of layer output (ref or TTNN) for PCC/dtype comparison."""
    if out is None:
        return "None"
    if isinstance(out, torch.Tensor) and not isinstance(out, TorchTTNNTensor):
        return str(out.dtype)
    if isinstance(out, TorchTTNNTensor):
        t = out.to_torch
        t = t() if callable(t) else t
        return str(t.dtype) if hasattr(t, "dtype") else "?"
    if hasattr(out, "to_torch"):
        t = getattr(out, "to_torch")
        x = t() if callable(t) else t
        return str(x.dtype) if hasattr(x, "dtype") else "?"
    return "?"


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_SHAPE_T3K],
    indirect=True,
)
def test_qwen3_coder_next(mesh_device):
    """Test Qwen3-Coder-Next model with TTNN acceleration (MoE + Gated Attention). Runs only on T3K."""
    if MESH_DEVICE != "T3K":
        pytest.skip(f"test_qwen3_coder_next runs only on T3K (MESH_DEVICE={os.environ.get(_MESH_DEVICE_ENV)})")
    tokenizer, model = _load_qwen3_model()

    nn_to_ttnn = {
        Qwen3NextSparseMoeBlock: TTNNQwen3MoE,
        Qwen3NextAttention: TTNNQwen3NextGatedAttention,
        Qwen3NextGatedDeltaNet: TTNNGatedDeltaNet,
    }
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
    }

    exclude_replacement = set()

    messages = [
        {
            "role": "user",
            "content": "Write a Python function to calculate fibonacci numbers using dynamic programming.",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    def _to_device(v):
        if not isinstance(v, torch.Tensor):
            return v
        device = next(model.parameters()).device
        v = v.to(device)
        if v.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            return v
        return v.to(torch.bfloat16)

    inputs = {k: _to_device(v) for k, v in inputs.items()}

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.config.eos_token_id

    modules1 = register_module_replacement_dict(
        model, nn_to_ttnn, model_config=None, exclude_replacement=exclude_replacement
    )
    modules2 = register_module_replacement_dict(
        model, nn_to_ttnn2, model_config=None, exclude_replacement=exclude_replacement
    )
    set_device(model, mesh_device)
    all_modules = {**modules1, **modules2}

    # print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    print("Running inference...")
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead

    # Warmup
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True)

    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)

    prompt_len = inputs["input_ids"].shape[-1]
    output_ids = outputs[0][prompt_len:].tolist()

    # Debug: first 20 token IDs and decoded
    print(f"[DEBUG] First 20 output token IDs: {output_ids[:20]}")
    print(f"[DEBUG] Full output token count: {len(output_ids)}")
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"[DEBUG] First 100 chars decoded: {repr(content[:100])}")

    print(f"Qwen3-Coder-Next output: {content}")
    DispatchManager.save_stats_to_file("qwen3_coder_next_timing_stats.csv")


# Single-device mesh for layer PCC (to_torch works without mesh composer)
MESH_SHAPE_SINGLE = (1, 1)
PCC_LAYER_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_SHAPE_SINGLE],
    indirect=True,
)
def _run_layer_pcc_forward(ref_model, ttnn_model, inputs):
    """Run ref and ttnn forward, capture layer outputs. Returns (ref_outputs, ttnn_outputs)."""
    layers = ref_model.model.layers
    num_layers = len(layers)
    ref_outputs = [None] * num_layers
    ttnn_outputs = [None] * num_layers

    def make_ref_hook(i):
        def hook(_mod, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            ref_outputs[i] = out.detach()

        return hook

    def make_ttnn_hook(i):
        def hook(_mod, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            ttnn_outputs[i] = out

        return hook

    ref_handles = [layers[i].register_forward_hook(make_ref_hook(i)) for i in range(num_layers)]
    ref_model(**inputs)
    for h in ref_handles:
        h.remove()

    ttnn_handles = [ttnn_model.model.layers[i].register_forward_hook(make_ttnn_hook(i)) for i in range(num_layers)]
    ttnn_model(**inputs)
    for h in ttnn_handles:
        h.remove()
    return ref_outputs, ttnn_outputs


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_SHAPE_SINGLE],
    indirect=True,
)
def test_qwen3_coder_next_layer_pcc(mesh_device):
    """Layer-by-layer PCC for both recurrent (T<=64) and chunked (T>64) Gated DeltaNet modes.
    Also checks dtype match between ref and TTNN. Requires single device (mesh 1x1)."""
    if MESH_DEVICE != "T3K":
        pytest.skip(
            f"test_qwen3_coder_next_layer_pcc runs only on T3K (MESH_DEVICE={os.environ.get(_MESH_DEVICE_ENV)})"
        )
    if mesh_device.get_num_devices() != 1:
        pytest.skip("test_qwen3_coder_next_layer_pcc requires single device (mesh 1x1) for to_torch conversion")

    tokenizer, ref_model = _load_qwen3_model()
    _, ttnn_model = _load_qwen3_model()

    nn_to_ttnn = {
        Qwen3NextSparseMoeBlock: TTNNQwen3MoE,
        Qwen3NextAttention: TTNNQwen3NextGatedAttention,
        Qwen3NextGatedDeltaNet: TTNNGatedDeltaNet,
    }
    modules = register_module_replacement_dict(ttnn_model, nn_to_ttnn, model_config=None, exclude_replacement=set())
    set_device(ttnn_model, mesh_device)
    for _k, mod in tqdm(modules.items()):
        mod.preprocess_weights()
        mod.move_weights_to_device()

    device = next(ref_model.parameters()).device

    def _to_device(v):
        if not isinstance(v, torch.Tensor):
            return v
        v = v.to(device)
        if v.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            return v
        return v.to(torch.bfloat16)

    layer_types = getattr(ref_model.config, "layer_types", None) or ["full_attention"] * len(ref_model.model.layers)

    results = {}
    for run_name, messages in [
        ("recurrent_T<=64", [{"role": "user", "content": "Hi."}]),
        ("chunked_T>64", [{"role": "user", "content": "Write a function " + "word " * 80}]),
    ]:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: _to_device(v) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[-1]
        effective_mode = "recurrent" if seq_len <= 64 else "chunked"
        ref_model.eval()
        ttnn_model.eval()
        torch.set_grad_enabled(False)
        ref_out, ttnn_out = _run_layer_pcc_forward(ref_model, ttnn_model, inputs)
        results[run_name] = {
            "seq_len": seq_len,
            "mode": effective_mode,
            "ref_outputs": ref_out,
            "ttnn_outputs": ttnn_out,
        }

    num_layers = len(ref_model.model.layers)
    min_pcc_overall = 1.0
    dtype_mismatches = []

    for run_name, data in results.items():
        seq_len = data["seq_len"]
        mode = data["mode"]
        ref_outputs = data["ref_outputs"]
        ttnn_outputs = data["ttnn_outputs"]

        print(f"\n=== {run_name} (seq_len={seq_len}, GatedDeltaNet mode={mode}) ===")
        print(
            f"{'Layer':<6} {'Type':<18} {'PCC':<10} {'MaxDiff':<10} {'MeanDiff':<10} {'RefDtype':<12} {'TTNNDtype':<12}"
        )
        print("-" * 88)

        for i in range(num_layers):
            lt = layer_types[i] if i < len(layer_types) else "?"
            pcc, max_d, mean_d = _compute_pcc(ref_outputs[i], ttnn_outputs[i])
            ref_dtype = _get_output_dtype(ref_outputs[i])
            ttnn_dtype = _get_output_dtype(ttnn_outputs[i])
            dtype_ok = ref_dtype == ttnn_dtype
            if not dtype_ok:
                dtype_mismatches.append((run_name, i, lt, ref_dtype, ttnn_dtype))
            min_pcc_overall = min(min_pcc_overall, pcc)
            print(f"{i:<6} {lt:<18} {pcc:<10.6f} {max_d:<10.6f} {mean_d:<10.6f} {ref_dtype:<12} {ttnn_dtype:<12}")

        print("-" * 88)
        min_pcc_run = min(_compute_pcc(ref_outputs[i], ttnn_outputs[i])[0] for i in range(num_layers))
        print(f"Min PCC ({run_name}): {min_pcc_run:.6f}")

    if dtype_mismatches:
        print("\n[Dtype mismatches (ref vs TTNN)]:")
        for rn, i, lt, rd, td in dtype_mismatches[:10]:
            print(f"  {rn} layer {i} ({lt}): ref={rd} vs ttnn={td}")
        if len(dtype_mismatches) > 10:
            print(f"  ... and {len(dtype_mismatches) - 10} more")
        assert False, (
            f"Dtype mismatch between ref and TTNN in {len(dtype_mismatches)} layer(s). " f"First: {dtype_mismatches[0]}"
        )

    print(f"\nMin PCC overall: {min_pcc_overall:.6f}")
    assert (
        min_pcc_overall >= PCC_LAYER_THRESHOLD
    ), f"Layer-by-layer min PCC {min_pcc_overall:.6f} below threshold {PCC_LAYER_THRESHOLD}"
