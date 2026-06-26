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
from typing import Callable, Optional

from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention as DSv3RefAttention
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Model as DSv3RefModel
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE as DSv3RefMoE
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_2_config import deepseek_v32_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config, glm_hf_config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6.modeling_deepseek import DeepseekV3Attention as KimiRefAttention
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6.modeling_deepseek import DeepseekV3Model as KimiRefModel
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6.modeling_deepseek import DeepseekV3MoE as KimiRefMoE
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config


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
        ref_cache_env: Optional[str] = None,
        mla_ref_cache_env: Optional[str] = None,
        ttnn_cache_env: Optional[str] = None,
        moe_pcc_threshold: float = 0.999,
        mla_pcc_threshold: float = 0.999,
        supports_pretrained: bool = True,
        prefill_trace_default: Optional[str] = None,
        prefill_trace_layout: str = "single_file",
        config_builder: Optional[Callable[[], object]] = None,
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
        self.ref_cache_env = ref_cache_env
        self.mla_ref_cache_env = mla_ref_cache_env
        self.ttnn_cache_env = ttnn_cache_env
        self.moe_pcc_threshold = moe_pcc_threshold
        self.mla_pcc_threshold = mla_pcc_threshold
        self.supports_pretrained = supports_pretrained
        # Golden chunked-prefill trace (metadata.json token_ids + kv_cache/ + hidden_states/) for the
        # chunked block/transformer PCC tests. The PREFILL_TRACE_DIR env overrides this default.
        self.prefill_trace_default = prefill_trace_default
        # Trace on-disk layout: "single_file" (DeepSeek — one safetensors/layer, all tensors as keys)
        # or "chunked_group_a_v1" (Kimi — each tensor a directory of row-sharded rows_<s>_<e>.safetensors,
        # hidden_states/ -> decoder_io/). The chunked trace readers in the tests dispatch on this.
        self.prefill_trace_layout = prefill_trace_layout
        # --- DSA (Deepseek Sparse Attention) capabilities -------------------------------------------
        # Per-variant behaviour branches on these fields (data-driven), not on `name`.
        # config_builder: returns a ready HF-attribute config when AutoConfig cannot load the model
        # (GLM's model_type `glm_moe_dsa` is unregistered). When set, it overrides the disk/HF
        # resolution path in conftest, and is also the single source the sparse-MLA CPU reference
        # derives its ModelArgs from (see reference.cpu_deepseek_v32). None = resolve via AutoConfig.
        self.config_builder = config_builder


DSV3 = TestVariant(
    name="deepseek_v3_d_p",
    env_var="DEEPSEEK_V3_HF_MODEL",
    hf_repo_id="deepseek-ai/DeepSeek-R1-0528",
    model_config=DeepSeekV3Config,
    default_local_path=Path("models/demos/deepseek_v3/reference"),
    shared_path=Path("/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528"),
    num_layers_to_download=24,
    reference_model_cls=DSv3RefModel,
    reference_attention_cls=DSv3RefAttention,
    reference_moe_cls=DSv3RefMoE,
    ref_cache_env="TT_DS_PREFILL_HOST_REF_CACHE",
    mla_ref_cache_env="DEEPSEEK_V3_MLA_REF_CACHE",
    ttnn_cache_env="TT_DS_PREFILL_TTNN_CACHE",
    mla_pcc_threshold=0.996,
    moe_pcc_threshold=0.982,
    prefill_trace_default="/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad",
)

KIMI_V2_6 = TestVariant(
    name="kimi_k2_6",
    env_var="KIMI_K2_6_HF_MODEL",
    hf_repo_id="moonshotai/Kimi-K2.6",
    model_config=KimiK26Config,
    default_local_path=Path("models/demos/deepseek_v3_d_p/reference/kimi_k2_6"),
    shared_path=None,
    num_layers_to_download=24,
    reference_model_cls=KimiRefModel,
    reference_attention_cls=KimiRefAttention,
    reference_moe_cls=KimiRefMoE,
    ref_cache_env="TT_KIMI_PREFILL_HOST_REF_CACHE",
    mla_ref_cache_env="KIMI_MLA_REF_CACHE",
    ttnn_cache_env="TT_KIMI_PREFILL_TTNN_CACHE",
    mla_pcc_threshold=0.995,
    moe_pcc_threshold=0.971,
    # vllm-traced golden: metadata.json + kv_cache nest under a run-hash subdir (resolve_trace_dir
    # descends), and kv_post_transform is row-sharded (the transformer test's loader reassembles it).
    prefill_trace_default="/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320",
    prefill_trace_layout="chunked_group_a_v1",
)

DSV32 = TestVariant(
    name="deepseek_v32",
    env_var="DEEPSEEK_V32_HF_MODEL",
    # V3.2-Exp config is hand-built (deepseek_v32_hf_config), like GLM: model_type `deepseek_v32` needs
    # trust_remote_code via AutoConfig, and — unlike a borrowed R1 config — the hand-built one carries
    # the DSA index_* fields so the sparse resolver (TtIndexer.matches_config) detects it. MLA dims +
    # YaRN match R1; the indexer is non-interleaved. hf_repo_id is the real V3.2-Exp repo (used only by
    # the pretrained path below + as the DEEPSEEK_V32_HF_MODEL override target).
    hf_repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
    # model_config: a dims class read only by the full-transformer / prefill tests (NUM_DENSE_LAYERS,
    # NUM_ROUTED_EXPERTS). Kept as DeepSeekV3Config because V3.2 shares R1's MoE dims (256 experts, 3
    # dense layers). Swap for a V3.2 dims class if/when a V3.2 full-transformer path is wired.
    model_config=DeepSeekV3Config,
    # config the device runs on: the hand-built V3.2 HF-attribute config (DSA index_* + YaRN). This is
    # what config_only resolves to for V3.2 (conftest short-circuits to config_builder).
    config_builder=deepseek_v32_hf_config,
    # The MLA truth is MLACPU (the "mlacpu" sparse reference) with random/CPU weights, so the sparse
    # tests never load pretrained HF weights — the pretrained path is disabled.
    supports_pretrained=False,
    # --- Pretrained-weight resolution (DISABLED) -------------------------------------------------
    # These locate an on-disk HF checkout (config.json + safetensors index + shards) for the pretrained
    # path; order is env_var -> default_local_path -> shared_path -> download hf_repo_id.
    # num_layers_to_download bounds the HF download to the first N layers' shards (+embeddings+norm), so
    # the full ~600GB model isn't pulled. All unused while supports_pretrained=False. To enable a V3.2
    # pretrained / full-model path: set supports_pretrained=True, point these at a real DeepSeek-V3.2-Exp
    # checkout (NOT an R1 one — V3.2 ships the indexer weights R1 lacks), and add a test that requests
    # the pretrained_transformer_weights / state_dict / model_path fixtures.
    # default_local_path=Path("/path/to/DeepSeek-V3.2-Exp"),
    # shared_path=Path("/proj_sw/user_dev/deepseek-ai/DeepSeek-V3.2-Exp"),
    # num_layers_to_download=24,
    # Block / full-model parity is out of P0 scope; the MLA truth comes from MLACPU, not HF attn.
    reference_model_cls=None,
    reference_attention_cls=None,
    reference_moe_cls=None,
    mla_ref_cache_env="DEEPSEEK_V32_MLA_REF_CACHE",
    mla_pcc_threshold=0.996,
)

GLM51 = TestVariant(
    name="glm_5_1",
    env_var="GLM51_HF_MODEL",
    hf_repo_id="zai-org/GLM-5.1",
    model_config=GLM51Config,
    # No HF pretrained_transformer_weights path: GLM ships per-layer shards and its MLA dims
    # (nope=192, v=256) differ from the dequant fixture's assumptions. Config is hand-built.
    supports_pretrained=False,
    reference_model_cls=None,
    reference_attention_cls=None,
    reference_moe_cls=None,
    mla_ref_cache_env="GLM51_MLA_REF_CACHE",
    mla_pcc_threshold=0.995,
    config_builder=glm_hf_config,
)

TEST_VARIANTS = {v.name: v for v in [DSV3, KIMI_V2_6, DSV32, GLM51]}
