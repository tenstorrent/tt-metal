# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Hugging Face ``Ministral3MLP`` vs ``TtMinistralMLP`` on Devstral-2-123B MLP weights only.

Loads ``model.layers.0.mlp.{gate_proj,up_proj,down_proj}.weight`` via ``model.safetensors.index.json``
(one or few shards), builds the meta dict used by TT (``convert_hf_to_meta_no_qkv_permute``), and
compares against TT. Does not load the full checkpoint or instantiate the full HF model.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3MLP

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import DEVSTRAL2_LARGE_L1_SMALL_SIZE
from models.experimental.devstral2_large.tt.tt_ministralmlp import TtDevstral2LargeMLP
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta_no_qkv_permute, standardize_hf_keys
from models.tt_transformers.tt.model_config import ModelArgs, PrecisionSetting, TensorGroup

DEVSTRAL2_LARGE_REPO_ID = "mistralai/Devstral-2-123B-Instruct-2512"


def _text_cfg_from_hf_cfg(hf_cfg) -> Ministral3Config:
    inner = getattr(hf_cfg, "text_config", None)
    out = inner if inner is not None else hf_cfg
    if not isinstance(out, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(out)!r}")
    return out


def _mlp_hf_keys(layer: int = 0) -> list[str]:
    return [
        f"model.layers.{layer}.mlp.gate_proj.weight",
        f"model.layers.{layer}.mlp.up_proj.weight",
        f"model.layers.{layer}.mlp.down_proj.weight",
    ]


def _load_hf_tensors_for_keys(repo_id: str, keys: list[str]) -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import safe_open as safetensors_safe_open

    try:
        index_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            local_files_only=os.getenv("CI") == "true",
        )
    except Exception as exc:
        raise RuntimeError(f"Could not download model.safetensors.index.json: {exc}") from exc

    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    out: dict[str, torch.Tensor] = {}
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"Key {key!r} not in weight_map for {repo_id}")
        shard_name = weight_map[key]
        shard_path = hf_hub_download(
            repo_id=repo_id,
            filename=shard_name,
            local_files_only=os.getenv("CI") == "true",
        )
        with safetensors_safe_open(shard_path, framework="pt", device="cpu") as sf:
            if key not in sf.keys():
                raise KeyError(f"Key {key!r} missing from shard {shard_name}")
            out[key] = sf.get_tensor(key).clone()
    return out


def _to_bf16_host_if_fp8(t: torch.Tensor) -> torch.Tensor:
    """HF stores Devstral-2-123B FFN weights in torch float8; host must be bf16 for ``ttnn.from_torch``.

    ``ttnn.from_torch(..., dtype=BFLOAT8_B, memory_config=WIDTH_SHARDED_DRAM, mesh_mapper=...)`` rejects
    ``torch.float8_e4m3fn`` inputs on Blackhole (pybind overload / unsupported path). The supported path
    is bf16 host → tile layout → pack to BFP8 (see ``ttnn.from_torch`` docstring for BFP8).
    """
    fp8_dtypes = tuple(
        dt
        for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz")
        if (dt := getattr(torch, name, None)) is not None
    )
    if fp8_dtypes and t.dtype in fp8_dtypes:
        return t.to(torch.bfloat16)
    return t


def _force_mlp_weight_dtypes_bf16(model_args: ModelArgs) -> None:
    """Stage FFN weights as BF16 on device (matches BF16 HF reference numerics)."""
    dec = model_args.model_config["DECODERS_OPTIMIZATIONS"]
    for conf in dec.decoder_optimizations.values():
        tp = conf._opt_settings["TensorPrecision"]
        tp[TensorGroup.FF1_FF3] = PrecisionSetting.BF16
        tp[TensorGroup.FF2] = PrecisionSetting.BF16


def _build_meta_state_dict(hf_partial: dict[str, torch.Tensor], text_cfg: Ministral3Config) -> dict:
    sd = dict(hf_partial)
    sd = standardize_hf_keys(sd)
    head_dim = getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // text_cfg.num_attention_heads
    return convert_hf_to_meta_no_qkv_permute(
        sd,
        head_dim=int(head_dim),
        n_heads=text_cfg.num_attention_heads,
        n_kv_heads=text_cfg.num_key_value_heads,
    )


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_trust)

    def _get_hf_model_cls_devstral_safe(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM
        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            return AutoModelForImageTextToText
        raise ValueError(
            f"Test supports multimodal configs in AutoModelForImageTextToText only; got {type(self.hf_config)}"
        )

    monkeypatch.setattr(mc.ModelArgs, "get_hf_model_cls", _get_hf_model_cls_devstral_safe)


@pytest.fixture
def devstral2_123b_dummy_config_path(monkeypatch):
    """``dummy_weights=True`` loads config from ``LOCAL_HF_PARAMS[model_name]``; map 123B name to Hub id."""
    from models.tt_transformers.tt import model_config as mc

    short_name = DEVSTRAL2_LARGE_REPO_ID.strip("/").split("/")[-1]
    monkeypatch.setitem(mc.ModelArgs.LOCAL_HF_PARAMS, short_name, DEVSTRAL2_LARGE_REPO_ID)


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "P150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
def test_ministral3_mlp_pcc_devstral2_large_partial_weights(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
    devstral2_123b_dummy_config_path,
):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL2_LARGE_REPO_ID)
    # ``get_max_prefill_chunk_size`` has no entry for Devstral-123B; env avoids the fallback table path.
    monkeypatch.setenv("MAX_PREFILL_CHUNK_SIZE", "128")

    try:
        hf_cfg = AutoConfig.from_pretrained(
            DEVSTRAL2_LARGE_REPO_ID,
            trust_remote_code=True,
            local_files_only=os.getenv("CI") == "true",
        )
    except Exception as exc:
        pytest.skip(f"Could not load Hugging Face config: {exc}")

    text_cfg = _text_cfg_from_hf_cfg(hf_cfg)
    mlp_keys = _mlp_hf_keys(0)
    try:
        hf_partial = _load_hf_tensors_for_keys(DEVSTRAL2_LARGE_REPO_ID, mlp_keys)
    except Exception as exc:
        pytest.skip(f"Could not download MLP shard(s) from Hub: {exc}")

    for k in mlp_keys:
        hf_partial[k] = _to_bf16_host_if_fp8(hf_partial[k])

    try:
        meta_sd = _build_meta_state_dict(hf_partial, text_cfg)
    except Exception as exc:
        pytest.skip(f"HF→meta conversion failed: {exc}")

    ref_mlp = Ministral3MLP(text_cfg).eval()
    ref_mlp.gate_proj.weight.data.copy_(hf_partial[mlp_keys[0]])
    ref_mlp.up_proj.weight.data.copy_(hf_partial[mlp_keys[1]])
    ref_mlp.down_proj.weight.data.copy_(hf_partial[mlp_keys[2]])
    # Default HF Linear weights are float32; activations are bf16 to match TT / ``torch_in``.
    ref_mlp.to(dtype=torch.bfloat16)

    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(512, seq_len),
        dummy_weights=True,
        use_hf_rope=True,
        cache_hf=False,
    )
    _force_mlp_weight_dtypes_bf16(model_args)

    # BF16 FFN weights raise static L1 CB usage vs BFP8; on 1×4 Blackhole keep prefill ``M`` small enough
    # to stay under the ~1.5 MB L1 budget (see TT_THROW circular buffer assert on wide 12k FFN).
    # sl = int(seq_len)
    # if is_blackhole() and mesh_device.get_num_devices() > 1:
    #     # Prefill ``m`` must stay tiny: BF16 FFN linear CB budget on 1×4 BH is still exceeded at S=32 for 12k×28k FFN.
    #     sl = min(sl, 1)

    tt_ccl = TT_CCL(mesh_device)
    tt_mlp = TtDevstral2LargeMLP(
        mesh_device,
        tt_ccl,
        model_args,
        meta_sd,
        weight_cache_path=None,
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )

    torch_in = torch.randn(1, 1, seq_len, model_args.dim, dtype=torch.bfloat16)
    ref_out = ref_mlp(torch_in)

    tt_in = ttnn.from_torch(
        torch_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3) if model_args.is_galaxy else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
        dtype=dtype,
        memory_config=model_args.get_mlp_input_mem_config(Mode.PREFILL, None),
        layout=ttnn.TILE_LAYOUT,
    )

    tt_out = tt_mlp(tt_in, Mode.PREFILL)
    mesh_dims = (3, 1) if model_args.is_galaxy else (1, -1)
    tt_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=mesh_dims, mesh_shape=model_args.cluster_shape),
    )
    tt_torch = tt_torch[:, :1, :, :]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
