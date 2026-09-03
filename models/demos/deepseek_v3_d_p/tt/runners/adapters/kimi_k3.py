# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K3 MLA + MoE adapter (test-only).

Kimi-K3 is a **hybrid**: 93 layers, of which only 24 are full-attention (MLA) layers and 69 are KDA
linear-attention layers. No KDA module exists in this package, so K3 cannot yet be served end to
end; a serving adapter would have to subclass ``PrefillModelAdapter`` directly and build its own
runtime with a hybrid layer schedule.

This adapter therefore exists to give the MLA and MoE layers a first-class ``variant`` fixture: the
test suite's ``TEST_VARIANTS`` registers it locally (the same test-only pattern GLM-5.2 uses) and it
is deliberately **absent** from ``models/demos/common/prefill/adapter.py:ADAPTER_PATHS``, so nothing
can select it with ``PREFILL_MODEL=kimi_k3`` and get a half-built model.

It subclasses ``MLAPrefillAdapter`` for the config/cache/reference plumbing only. ``build_runtime``
and ``allocate_kv_cache`` are inherited but would size the KV cache to ``params.num_layers``, which
is wrong for 24-of-93; they are overridden to fail loudly rather than mislead.

MoE scope (issue #51336): the latent-MoE structure -- routed experts at the reduced 3584 hidden,
896 experts / top-16, a latent RMSNorm, and one shared expert at 6144. Every FFN site runs the
checkpoint's **SiTU-GLU** on device: the routed experts through the fused kernel (#51351), the
shared expert and the layer-0 dense FFN through ttnn-level softcap/sigmoid/multiply (#53625), which
is correct but not yet tuned at their 6144 / 33792 widths. One deliberate limit remains:
  * only the **gate** uses real checkpoint weights. Experts, shared expert and the latent
    projections use seeded random weights, because everything routed is MXFP4 and no dequantizer
    exists yet. Device PCC is therefore TT-vs-torch on identical seeded weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from loguru import logger

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class KimiK3Adapter(MLAPrefillAdapter):
    # --- identity ---
    name = "kimi_k3"
    model_config = KimiK3Config
    # Repo-local, dot-free (transformers' trust_remote_code import chokes on "." in a path). The dir
    # holds the TRIMMED upstream MLA reference, not a loadable full-model checkpoint.
    hf_model_default = "models/demos/deepseek_v3_d_p/reference/kimi_k3"
    default_gate_mode = "DEVICE_FP32"  # single expert group, as Kimi-K2.6

    # Single expert group + device gate: route routing-all-gather semaphores to L1_SMALL. Inherited
    # from the Kimi family; inert for the MLA-only tests but correct if a runtime is ever built.
    routing_use_l1_small_for_semaphores = True

    # --- test metadata ---
    hf_repo_id = "moonshotai/Kimi-K3"
    env_var = "KIMI_K3_HF_MODEL"
    default_local_path = Path("models/demos/deepseek_v3_d_p/reference/kimi_k3")
    # The download fallback fetches layers 0..N-1, and pretrained_mla_layer is 3.
    num_layers_to_download = 4
    ref_cache_env = "TT_KIMI_K3_PREFILL_HOST_REF_CACHE"
    mla_ref_cache_env = "KIMI_K3_MLA_REF_CACHE"
    ttnn_cache_env = "TT_KIMI_K3_PREFILL_TTNN_CACHE"
    # 1152 (the AttnRes suite's own value) fails `inter_block`'s statistics collective once the
    # sealed set has two blocks, which first happens at depth 24; 24576 (this package's usual value)
    # then starves MLA chunked attention of circular buffers as soon as there is a second chunk to
    # attend over. 4096 clears both. See tenstorrent/tt-metal#54834.
    # Kimi-K3 is NoPE: the KV cache's second half carries no rotation, so scoring it against the
    # golden must NOT re-base to the device's Meta interleave. With the RoPE default the nope half
    # still reads ~0.999 while the pe half collapses to ~0.02 -- a broken comparison that looks
    # exactly like a broken model.
    kv_pe_interleave = False
    l1_small_size = 4096
    # `weight_cache_path` appends `{name}_{arch}_{N}dev/{sp}x{tp}`, so this is the root only. The
    # cache generator writes `<root>/kimi_k3_bh_32dev/<checkpoint-id>/mesh8x4.tpaxis1`, keyed by
    # checkpoint so a different one cannot silently load another's tensors; `8x4` is a symlink onto
    # that, which is what makes the layout match every other model here.
    ttnn_cache_default = "/mnt/models/deepseek-prefill-cache/kimi-k3-ttnn-cache"
    # The runner prints this and the producer reads it only when PCC checking is on. Kimi-K3 has no
    # full-depth golden — the 1M trace records decoder_output for layers 0..24 of 93 — so a runner
    # run cannot check end-to-end accuracy and must not pretend to. It is named here because the
    # runner reads the attribute unconditionally at config-print time.
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/k3_vllm_code_debug_1M"
    # Loading the staged checkpoint wholesale needs an MXFP4 -> bf16 dequantizer that does not exist
    # yet, so the full-transformer fixtures stay skipped. The MoE gate is exempt: it is unquantized and read
    # through a prefix-filtered safe_open.
    # The dequantized export stores routed experts as plain bf16, so no MXFP4 dequantizer is
    # needed; the loader accepts both `language_model.model.` and `model.` key roots.
    supports_pretrained = True
    # MLA alone is loadable: quantization_config.ignore covers self_attn, so those weights are bf16.
    # The first full-attention layer, not 0 -- layers 0-2 are KDA and hold no MLA tensors.
    pretrained_mla_layer = KimiK3Config.mla_layer_ids()[0]
    mla_trace_defaults = ("/mnt/models/deepseek-prefill-cache/golden/structured_traces/kimi_k3_100k_vllm",)
    # Left as None: shared_path feeds conftest's state_dict fixture, which pytest would resolve --
    # loading all 1.5 TB -- before the supports_pretrained skip in the fixture body runs.
    shared_path = None

    # Device vs upstream KimiSparseMoeBlock. Held at test_kimi_k3_moe's final_output_pcc: the two
    # compare the same device tensor against references that agree to 2e-5, so they measure the same
    # thing and cannot carry different bars. The 8x4 value is 0.9696 (reference 0.969567,
    # final_output 0.969585) -- 8x4 spreads 896 experts over 32 chips instead of 8, so the top-16
    # combine accumulates across 4x as many chips in the bf8 latent space than on the 2x4 proxy
    # (0.995693). 0.965 keeps the usual ~0.005 of margin.
    #
    # Moving the routed experts onto SiTU-GLU did not cost anything here: 8x4 went 0.969434 ->
    # 0.969567 and 2x4 0.995692 -> 0.995693. K3's realistic activations sit well inside both tanh
    # caps, so the bf4 accuracy cost that _K3_SATURATION_CASES documents does not arise at these
    # scales.
    moe_pcc_threshold = 0.965

    @property
    def config_builder(self) -> Callable:
        """Hand-built config: upstream ``modeling_kimi_linear.py`` raises ImportError without
        ``fla-core``, so ``AutoConfig`` cannot load ``model_type: kimi_linear`` here."""
        return kimi_k3_hf_config

    @property
    def reference_attention_cls(self):
        from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_k3_mla import KimiMLAAttention

        return KimiMLAAttention

    @property
    def reference_moe_cls(self):
        """K3's MoE block: DeepSeek-V3's, wrapped in the shared low-rank latent projection pair.

        The block applies one activation to routed and shared experts alike, which matches the
        device: ``run_reference_moe`` builds it with hidden_act="situ" for both halves.
        """
        from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiSparseMoeBlock

        return KimiSparseMoeBlock

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        """Size the KV cache to this rank's FULL-ATTENTION layers, not its layers.

        The inherited allocator takes `params.num_layers` and would reserve 93 slots per user for a
        model that writes 24. Worse, the slot arithmetic downstream
        (`cache_user_id * layer_num + cache_layer_idx`) is computed against whatever count the cache
        was sized to, so an oversized cache is not merely wasteful — it puts every user's rows at
        the wrong stride.

        The count is rank-local. `KimiK3Config.mla_kv_slot()` returns the model-wide slot and is the
        wrong tool here for any rank with `first_layer_idx > 0`; `KimiK3LayerSchedule` is the
        rank-local one. See `tt/kimi_k3/layer_schedule.py`.

        A K3 rank can legitimately hold ZERO full-attention layers — a 1-layer bring-up run is layer
        0, which is KDA — and then there is no cache to allocate at all.
        """
        from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
        from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache

        schedule = KimiK3LayerSchedule.build(KimiK3Config, params.first_layer_idx, params.num_layers)
        if schedule.num_mla_layers == 0:
            return MlaKvCaches(kvpe=None)

        return MlaKvCaches(
            kvpe=allocate_mla_kvpe_cache(
                mesh_device=mesh_device,
                hf_config=hf_config,
                max_seq_len=params.max_seq_len,
                mesh_shape=params.mesh_shape,
                sp_axis=params.sp_axis,
                num_layers=schedule.num_mla_layers,
                num_users=params.num_users,
            )
        )

    def layer_split_boundaries(self, num_layers: int):
        """Pipeline ranks may only start on an AttnRes block boundary.

        AttnRes seals every `ATTN_RES_BLOCK_SIZE` layers, and a rank starting at layer `F` inherits
        exactly `F // 12` sealed snapshots from upstream. Constraining `F` to a multiple of 12 makes
        that count static, which is what lets the cross-rank activation handoff have a fixed width
        instead of one that depends on where the split landed. GLM-5.2 constrains its splits for the
        same class of reason.
        """
        block = KimiK3Config.ATTN_RES_BLOCK_SIZE
        return {boundary for boundary in range(0, num_layers + 1, block)}

    @property
    def transformer_cls(self):
        """The model class the shared chunked-prefill harness should build.

        `run_chunked_transformer_updated` constructs a transformer directly rather than going
        through a runtime, so the `MODEL_CLS` seam on `TtKimiK3Runtime` does not reach it. Kimi-K3
        cannot use `TtPrefillTransformer`: its schedule is hybrid (only 24 of 93 layers are MLA), it
        carries KDA recurrent state, and its residual is a block-structured AttnRes walk.
        """
        from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer

        return TtKimiK3Transformer

    def num_kv_cache_layers(self, num_layers: int) -> int:
        """How many KV slabs `num_layers` of this model actually need.

        Dense models answer `num_layers`. Kimi-K3 writes a slab only on full-attention layers, so a
        24-layer slice needs 6, not 24 — the same assumption that makes `build_kv_chunk_table`
        reject the model outright (#54892).
        """
        from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule

        return KimiK3LayerSchedule.build(KimiK3Config, 0, num_layers).num_mla_layers

    def kv_slot_layer_ids(self, num_layers: int):
        """Cache slot -> GLOBAL layer index, for scoring KV against a golden keyed by layer.

        A hybrid model's slot is not its layer: 24 layers occupy 6 slots holding 3/7/11/15/19/23.
        Taking the head of `mla_layer_ids` is the tempting shortcut and is wrong for any rank that
        does not start at layer 0 (#54843); this derives the mapping from the schedule instead.
        """
        from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule

        schedule = KimiK3LayerSchedule.build(KimiK3Config, 0, num_layers)
        return [
            schedule.global_index(local) for local, slot in enumerate(schedule.kv_slot_of_local) if slot is not None
        ]

    def pipeline_activation_planes(self, boundary_layer_idx: int) -> int:
        """The live stream, plus one plane per AttnRes snapshot sealed before this boundary.

        A read scores the live sum against every sealed snapshot, and the snapshots for layers
        upstream of the boundary are produced on another rank — nothing on this side can recompute
        them, so they ride the payload. `layer_split_boundaries` keeps `boundary_layer_idx` a
        multiple of the block size, which is what makes this count exact rather than approximate.
        """
        return 1 + boundary_layer_idx // KimiK3Config.ATTN_RES_BLOCK_SIZE

    def load_hf_config(self):
        """The Kimi-K3 config, hand-built rather than loaded through `AutoConfig`.

        Kimi-K3's `model_type` is `kimi_linear`, and the checkpoint's remote code
        (`modeling_kimi_linear.py`) raises `ImportError` at module import without `fla-core`, which
        is not installed here — `AutoConfig(trust_remote_code=True)` therefore fails before it can
        return anything. `kimi_k3_hf_config` carries the same attributes the MLA and MoE paths read.
        `max_seq` is a placeholder; the runner overwrites `max_seq_len` on the result.
        """
        from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_hf_config

        return kimi_k3_hf_config()

    def build_runtime(self, *, mesh_device, hf_config, params):
        """Construct the Kimi-K3 stack and return its runtime.

        Mirrors `MLAPrefillAdapter.build_runtime` — the runtime is stateless with respect to the KV
        cache, which the engine allocates and passes in — but drives `TtKimiK3Runtime`, whose
        `MODEL_CLS` is `TtKimiK3Transformer`. The transformer builds its own AttnRes walk from the
        weight cache; nothing here needs to supply a residual factory, and supplying a plain one
        would silently cost both accuracy and the 2N read sites that dominate its cost.
        """
        from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
        from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
        from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntimeConfig

        topology = per_axis_topology()
        logger.info(f"Kimi-K3 per-axis CCL topology (sp, tp) = {topology}")

        runtime_config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            mesh_shape=params.mesh_shape,
            chunk_size=params.chunk_size,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            num_links=params.num_links,
            topology=topology,
            capacity_factor=params.capacity_factor,
            gate_fallback_mode=GateComputeMode[params.gate_mode_name],
            weight_cache_path=params.weight_cache_path,
            model_cfg=self.model_config,
            first_layer_idx=params.first_layer_idx,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            kv_only_last_layer=params.kv_only_last_layer,
            dflash_enabled=params.dflash_enabled,
            routing_use_l1_small_for_semaphores=self.routing_use_l1_small_for_semaphores,
            sparse_kv_cache_format=self.resolve_sparse_kv_cache_format(params.sparse_kv_cache_format),
            use_trace=params.use_trace,
            overlap_shared_expert_with_dispatch=params.overlap_shared_expert_with_dispatch,
        )
        return TtKimiK3Runtime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict={},
            config=runtime_config,
        )
