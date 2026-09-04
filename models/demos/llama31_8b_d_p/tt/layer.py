# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One Llama-3.1-8B decoder layer:

    residual = x;  x = residual + Attention(input_layernorm(x))
    residual = x;  x = residual + MLP(post_attention_layernorm(x))

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaDecoderLayer`.
**Template:** `models/demos/gpt_oss_d_p/tt/layer.py:46`, forward `:126`, with two things deleted and
two deliberately kept.

**Deleted:**

* the **MoE branch** — every Llama layer is a dense SwiGLU MLP
  (`bringup_log/00_MODEL_CARD.md` §3), so `use_ep_moe`, `ep_seq_len_per_chip` and
  `expert_weight_dtype` all go;
* the **`layer_types` plumbing** (`models/demos/gpt_oss_d_p/tt/layer.py:96`, `:119`) — there is no
  sliding-window alternation, so there is no per-layer `AttentionConfig` and no `is_sliding`. One
  config is built by `tt/model.py` and shared unmodified by all 32 layers.

**Kept, because both are load-bearing under long-context DRAM pressure**
(`BRINGUP_RECIPE.md:1506-1507`):

* the `ttnn.move(hidden_states)` re-allocation guard past 32k tokens
  (`models/demos/gpt_oss_d_p/tt/layer.py:138-140`) — it defragments the residual stream;
* the eager `deallocate(True)` calls. **Consequence for callers:** this layer consumes its input.
  `hidden_states` is freed by the first residual add, so a caller that needs the input afterwards
  must keep its own copy.

**The bring-up probe stays too** (`LLAMA_DELTA_PROBE`, `DEC-023`). A per-layer L2 / mean-abs /
signed-mean of each residual delta is the fastest way to find *which* sublayer drifts in a 32-layer
stack — a layer-level PCC cannot localise (`BRINGUP_RECIPE.md:1514-1521`), and a growing signed
mean is the fingerprint of a directional bias accumulating in one sublayer. It is the one place in
this package where a bare `except Exception` is allowed, because a probe must never break a run,
and it logs rather than passing silently (recipe §0 rule 5).
"""

import os

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_d_p.utils.substate import substate

from .attention import Attention, AttentionConfig, ProgramConfig
from .config import MeshConfig, derive_head_dim
from .mlp import MLP
from .rms_norm import RMSNorm

# `DEC-023`: the package's ONE bring-up env var. Read once at import, as the template does
# (`models/demos/gpt_oss_d_p/tt/layer.py:19`), so toggling it mid-run is not a thing that half-works.
DELTA_PROBE_ENV_VAR = "LLAMA_DELTA_PROBE"
_DELTA_PROBE = os.environ.get(DELTA_PROBE_ENV_VAR, "") != ""

# Past this many tokens the residual stream is re-allocated before the layer runs, to defragment
# DRAM (`models/demos/gpt_oss_d_p/tt/layer.py:137-140`). No P6 gate reaches it; the longest is 2048.
_MOVE_GUARD_SEQ_LEN = 32 * 1024


def _delta_stats(tag, layer_idx, tensor):
    """Log device-0's shard of a residual delta: L2, mean|x|, signed mean, max|x|.

    A *growing signed mean* localises a directional bias to one sublayer, which is exactly what a
    layer- or model-level PCC cannot do (`BRINGUP_RECIPE.md:1514-1521`). Output goes through
    `loguru`, so a gate run's `tee` captures it into `bringup_log/raw/`.
    """
    try:
        from loguru import logger

        d0 = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()
        logger.info(
            f"[delta-probe L{layer_idx:>2}] {tag}: L2={d0.norm():.3f}  mean|x|={d0.abs().mean():.4f}  "
            f"signed_mean={d0.mean():.5f}  max|x|={d0.abs().max():.3f}"
        )
    except Exception as e:  # a probe must never break a run — but it must say so
        from loguru import logger

        logger.warning(f"[delta-probe] failed at L{layer_idx} {tag}: {e}")


def build_attention_config(hf, *, max_seq_len, sequence_parallel=False) -> AttentionConfig:
    """The `AttentionConfig` for **every** layer — build it once per model, not once per layer.

    Llama has no sliding-window alternation, so there is nothing to vary per layer and no
    `dataclasses.replace` to keep in sync (`models/demos/gpt_oss_d_p/tt/attention/config.py` +
    `attention/__init__.py:77-84` do both). `head_dim` comes from the package's one derivation
    (`DEC-020`, `DEC-032`).
    """
    return AttentionConfig(
        hidden_size=hf["hidden_size"],
        num_heads=hf["num_attention_heads"],
        num_kv_heads=hf["num_key_value_heads"],
        head_dim=derive_head_dim(hf),
        max_seq_len=max_seq_len,
        rms_norm_eps=hf["rms_norm_eps"],
        sequence_parallel=sequence_parallel,
    )


class DecoderLayer:
    """`[1, 1, B*S, hidden]` -> `[1, 1, B*S, hidden]`, bf16 TILE in and out. Consumes its input."""

    def __init__(
        self,
        mesh_device,
        hf,
        state_dict,
        layer_idx,
        *,
        ccl_manager=None,
        mesh_config=None,
        attention_config=None,
        program_config=None,
        transformation_mats=None,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        max_seq_len=1024,
        sequence_parallel=False,
    ):
        """
        Args:
            mesh_device: the open mesh.
            hf: the normalised config **dict** (recipe P1 trap 2).
            state_dict: this layer's HF keys, i.e. `self_attn.*`, `mlp.*`,
                `input_layernorm.weight`, `post_attention_layernorm.weight` — the caller splits with
                `substate(state_dict, f"model.layers.{i}")`. Splitting further is **this** module's
                job, per the package convention (`BRINGUP_RECIPE.md:1072-1073`). Empty dict ->
                cache-only, which requires `tensor_cache_path`.
            layer_idx: this layer's index. Used for the per-layer KV write and by the delta probe.
            ccl_manager: the model's `CCLManager`. Required only when `tp > 1`.
            mesh_config: the model's `MeshConfig`; defaults to TP over the whole column axis.
            attention_config: the shared `AttentionConfig`. `None` builds one from `hf` — the
                standalone-test path; `tt/model.py` builds one and passes it to all 32 layers.
            program_config: the shared `ProgramConfig` (pinned 8x8 SDPA grid). `None` builds one.
            transformation_mats: `{"prefill": tensor}` from `tt/rope.py::build_transformation_mat`.
            weight_dtype: on-device weight dtype for the projections, `bfloat8_b` (`DEC-022`).
            activation_dtype: the residual stream's dtype, `bfloat16` (`DEC-022`).
            tensor_cache_path: where `ttnn.as_tensor` persists / reloads this layer's weights.
            max_seq_len: passed to `AttentionConfig` when this layer builds its own.
            sequence_parallel: the SP ring path. `False` for every P5-P7 gate (P8 owns it).
        """
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.attention_config = attention_config or build_attention_config(
            hf, max_seq_len=max_seq_len, sequence_parallel=sequence_parallel
        )
        self.program_config = program_config or ProgramConfig()

        self.input_layernorm = RMSNorm(
            mesh_device,
            hf,
            substate(state_dict, "input_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
            mesh_config=self.mesh_config,
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            hf,
            substate(state_dict, "post_attention_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
            mesh_config=self.mesh_config,
        )
        self.self_attn = Attention(
            mesh_device,
            self.attention_config,
            substate(state_dict, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=self.mesh_config,
            program_config=self.program_config,
            layer_idx=layer_idx,
            transformation_mats=transformation_mats,
            weight_dtype=weight_dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
        )
        self.mlp = MLP(
            mesh_device,
            hf,
            substate(state_dict, "mlp"),
            mesh_config=self.mesh_config,
            ccl_manager=ccl_manager,
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
        )

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        *,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=False,
    ):
        """Norm -> attention -> residual -> norm -> MLP -> residual.

        **`hidden_states` is consumed** (freed by the first residual add). `position_embeddings` is
        the `[cos, sin]` pair — contiguous tables from `tt/rope.py::build_prefill_rope`, or the
        whole-cache indexed tables when `indexed_rope`.

        No collective is called from here. Attention and the MLP own their own TP collectives
        (`bringup_log/04_CCL_PLAN.md` §4); the residual adds are elementwise-local by construction,
        which is the whole reason the layer never needs one.
        """
        seq_len = hidden_states.shape[-2]
        if seq_len > _MOVE_GUARD_SEQ_LEN:
            hidden_states = ttnn.move(hidden_states)

        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed,
            position_embeddings,
            kv_cache=kv_cache,
            user_id=user_id,
            batch_size=batch_size,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )
        normed.deallocate(True)
        if _DELTA_PROBE:
            _delta_stats("attn_out", self.layer_idx, attn_out)

        hidden_states = ttnn.add(residual, attn_out, output_tensor=attn_out)
        residual.deallocate(True)

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed)
        normed.deallocate(True)
        if _DELTA_PROBE:
            _delta_stats("mlp_out ", self.layer_idx, mlp_out)

        hidden_states = ttnn.add(residual, mlp_out, output_tensor=mlp_out)
        residual.deallocate(True)
        return hidden_states
