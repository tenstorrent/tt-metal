# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Per-variant test configurations for the DeepSeek V3 architecture family.

A `TestVariant` carries the metadata that conftest fixtures need to drive a
specific model: HF download coordinates, on-disk fallback paths, and
reference-modeling class handles. Tests stay variant-agnostic; the body reads
fields off the variant instance.

Adding a variant: instantiate `TestVariant(...)` with the metadata and append
to `TEST_VARIANTS`.
"""

from pathlib import Path
from typing import Optional

from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Model as DSv3RefModel
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE as DSv3RefMoE
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.kimi_k26.kimi_k26_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.kimi_k26.modeling_deepseek import DeepseekV3Attention as KimiRefAttention
from models.demos.deepseek_v3_d_p.reference.kimi_k26.modeling_deepseek import DeepseekV3Model as KimiRefModel
from models.demos.deepseek_v3_d_p.reference.kimi_k26.modeling_deepseek import DeepseekV3MoE as KimiRefMoE


class TestVariant:
    # Defeat pytest's default `Test*` class collection.
    __test__ = False

    def __init__(
        self,
        *,
        name: str,
        env_var: str,
        hf_repo_id: str,
        model_config: type,
        default_local_path: Optional[Path] = None,
        shared_path: Optional[Path] = None,
        num_layers_to_download: int = 24,
        reference_model_cls: type,
        reference_attention_cls: Optional[type] = None,
        reference_moe_cls: Optional[type] = None,
        weight_cache_prefix: Optional[str] = None,
        trace_dir_env: Optional[str] = None,
        ref_cache_env: Optional[str] = None,
        moe_pcc_threshold: float = 0.999,
        mla_pcc_threshold: float = 0.999,
        supports_pretrained: bool = True,
    ) -> None:
        self.name = name
        self.env_var = env_var
        self.hf_repo_id = hf_repo_id
        self.model_config = model_config
        self.default_local_path = default_local_path
        self.shared_path = shared_path
        self.num_layers_to_download = num_layers_to_download
        self.reference_model_cls = reference_model_cls
        self.reference_attention_cls = reference_attention_cls
        self.reference_moe_cls = reference_moe_cls
        self.weight_cache_prefix = weight_cache_prefix or name
        self.trace_dir_env = trace_dir_env
        self.ref_cache_env = ref_cache_env
        self.moe_pcc_threshold = moe_pcc_threshold
        self.mla_pcc_threshold = mla_pcc_threshold
        self.supports_pretrained = supports_pretrained


DSV3 = TestVariant(
    name="dsv3",
    env_var="DEEPSEEK_V3_HF_MODEL",
    hf_repo_id="deepseek-ai/DeepSeek-R1-0528",
    model_config=DeepSeekV3Config,
    default_local_path=Path("models/demos/deepseek_v3/reference"),
    shared_path=Path("/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528"),
    num_layers_to_download=24,
    reference_model_cls=DSv3RefModel,
    # MLA upstream cross-check disabled — `create_mla_reference` already runs the same
    # DeepseekV3Attention class for DSv3, so populating this would duplicate the forward
    # pass. To re-enable, use DeepseekV3Attention from
    # models.demos.deepseek_v3.reference.modeling_deepseek.
    reference_attention_cls=None,
    reference_moe_cls=DSv3RefMoE,
    weight_cache_prefix="deepseek_v3_d_p",
    trace_dir_env="TT_DS_PREFILL_TRACE_DIR",
    ref_cache_env="TT_DS_PREFILL_HOST_REF_CACHE",
)

KIMI = TestVariant(
    name="kimi",
    env_var="KIMI_K26_HF_MODEL",
    hf_repo_id="moonshotai/Kimi-K2.6",
    model_config=KimiK26Config,
    default_local_path=Path("models/demos/deepseek_v3_d_p/reference/kimi_k26"),
    shared_path=None,
    num_layers_to_download=24,
    reference_model_cls=KimiRefModel,
    reference_attention_cls=KimiRefAttention,
    reference_moe_cls=KimiRefMoE,
    weight_cache_prefix="kimi_k26",
    trace_dir_env="TT_KIMI_PREFILL_TRACE_DIR",
    ref_cache_env="TT_KIMI_PREFILL_HOST_REF_CACHE",
    supports_pretrained=False,
    mla_pcc_threshold=0.995,
    moe_pcc_threshold=0.989,
)

TEST_VARIANTS = {v.name: v for v in [DSV3, KIMI]}
