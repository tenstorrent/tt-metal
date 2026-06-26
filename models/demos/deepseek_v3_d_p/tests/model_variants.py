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
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config, glm_hf_config, glm_model_args
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
        has_indexer: bool = False,
        tp_cap: Optional[int] = None,
        config_builder: Optional[Callable[[], object]] = None,
        cpu_model_args: Optional[Callable[[], object]] = None,
        reference_kind: str = "hf_attn",
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
        # Test bodies branch on these, never on `name`, so onboarding a model is data-only.
        # has_indexer: the model runs the lightning indexer + sparse SDPA (DeepSeek V3.2 / GLM-5.1).
        self.has_indexer = has_indexer
        # tp_cap: max tensor-parallel factor the model supports (GLM has 64 q-heads → sparse_sdpa
        # needs H/tp >= 32 → tp <= 2). None = no cap beyond what the mesh provides.
        self.tp_cap = tp_cap
        # config_builder: returns a ready HF-attribute config when AutoConfig cannot load the model
        # (GLM's model_type `glm_moe_dsa` is unregistered). When set, it overrides the disk/HF
        # resolution path in conftest. None = resolve via AutoConfig as usual.
        self.config_builder = config_builder
        # cpu_model_args: returns the reference_cpu ModelArgs for the MLACPU/IndexerCPU truth (the
        # "mlacpu" reference path). None = use the default ModelArgs (matches DeepSeek's R1 dims).
        self.cpu_model_args = cpu_model_args
        # reference_kind: which CPU truth the MLA tests compare against — "hf_attn" drives the
        # upstream HF attention via reference_attention_cls (dense), "mlacpu" drives the V3.2
        # reference_cpu MLACPU (indexer + sparse), required once seq_len > index_topk.
        self.reference_kind = reference_kind


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
    # V3.2-Exp shares MLA dims with R1; we resolve config from the R1 checkout to avoid the
    # uncertainty of loading V3.2-Exp remote code via AutoConfig. The indexer reads its index_*
    # attrs via getattr-with-defaults that already match DeepSeek, so no extra config wiring.
    hf_repo_id="deepseek-ai/DeepSeek-R1-0528",
    model_config=DeepSeekV3Config,
    default_local_path=Path("models/demos/deepseek_v3/reference"),
    shared_path=Path("/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528"),
    num_layers_to_download=24,
    # Block / full-model parity is out of P0 scope; the MLA truth comes from MLACPU, not HF attn.
    reference_model_cls=None,
    reference_attention_cls=None,
    reference_moe_cls=None,
    mla_ref_cache_env="DEEPSEEK_V32_MLA_REF_CACHE",
    mla_pcc_threshold=0.996,
    has_indexer=True,
    reference_kind="mlacpu",
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
    has_indexer=True,
    # 64 q-heads: the sparse MLA forward now transposes the TP shard (heads→seq) around sparse_sdpa
    # so per-chip H is no longer the limit; tp=4 is supported (32-indexer-heads / tp must stay integral).
    tp_cap=4,
    config_builder=glm_hf_config,
    cpu_model_args=glm_model_args,
    reference_kind="mlacpu",
)

TEST_VARIANTS = {v.name: v for v in [DSV3, KIMI_V2_6, DSV32, GLM51]}
