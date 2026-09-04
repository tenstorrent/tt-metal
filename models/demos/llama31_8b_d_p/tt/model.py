# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The Llama-3.1-8B prefill model: `embedding -> [DecoderLayer] * n_layers -> final norm -> lm_head`.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaForCausalLM`
(`LlamaModel` + `lm_head`). **Template:** `models/demos/gpt_oss_d_p/tt/model.py:41`
(`_forward_layers_and_head:179`, `prefill_forward:246`, `prepare_inputs_prefill:279`,
`process_output_prefill:322`); second opinion `models/demos/minimax_m3/tt/model.py:87`.

**Deleted relative to the template:** the MoE / expert-parallel knobs, the hybrid layer schedule,
the vocab padding (`128256/8 = 16032` is exact — `tt/lm_head.py`), and the optional on-device
sampling at `models/demos/gpt_oss_d_p/tt/model.py:145-157`, which is a decode feature and decode is
an explicit non-goal for this iteration.

**Three things differ from the template on purpose:**

1. **The final norm always runs; `skip_lm_head` skips only the head** (`DEC-049`). The template
   returns *pre-norm* hidden states when `skip_lm_head=True`
   (`models/demos/gpt_oss_d_p/tt/model.py:236-240`), which does not correspond to anything HF
   exposes: `LlamaModel`'s `last_hidden_state` is **post**-norm. `G-MODEL` scores hidden states
   against exactly that tensor, so returning the pre-norm stream would force the gate to
   re-implement the norm on the host and score against a quantity no reference produces.
2. **`n_layers` is a real parameter**, because `G-MODEL` runs at 2 and 4 layers before 32
   (`BRINGUP_RECIPE.md:1559`), and `with_lm_head=True` is the **default** so the gate's
   top-1 half is never conditional (`BRINGUP_RECIPE.md:1562`).
3. **Two per-layer seams, not one** (`DEC-050`): `on_layer_complete(layer_idx)` is the template's
   migration/ack seam, kept verbatim for P10; `on_layer_output(layer_idx, hidden_states)` is the
   bring-up seam `G-MODEL`'s per-layer PCC curve needs. Overloading one callback with both jobs
   would put a numerical probe on the path P10 acks chunks from.

**RoPE.** `prepare_inputs_prefill` builds the **contiguous** cos/sin for the chunk it is given
(`tt/rope.py::build_prefill_rope`). The whole-cache **indexed** tables are P7's: the runtime builds
them once with `build_indexed_rope` and passes them to `prefill_forward` as `rot_mats_global` with
`indexed_rope=True`, which is why that parameter exists here already.
"""

import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_d_p.utils.substate import substate

from .config import MeshConfig
from .embedding import Embedding
from .layer import DecoderLayer, build_attention_config
from .lm_head import LMHead
from .rms_norm import RMSNorm
from .rope import build_prefill_rope, build_transformation_mat


class Model:
    """Llama-3.1-8B TTNN prefill model. Dense MLP + GQA + full llama3-scaled RoPE + plain RMSNorm."""

    def __init__(
        self,
        mesh_device,
        hf,
        state_dict,
        *,
        ccl_manager=None,
        mesh_config=None,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        max_seq_len=128 * 1024,
        n_layers=None,
        with_lm_head=True,
        sequence_parallel=False,
    ):
        """
        Args:
            mesh_device: the open mesh.
            hf: the normalised config **dict** (recipe P1 trap 2) — `ModelArgs.hf_config`.
            state_dict: the **whole** HF checkpoint dict, keys unchanged
                (`model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`,
                `lm_head.weight`). This class does the top-level `substate` split; each module does
                its own. Empty dict -> cache-only, which requires `tensor_cache_path`.
            ccl_manager: the model's `CCLManager`. Required only when `tp > 1`.
            mesh_config: the model's `MeshConfig`; defaults to TP over the whole column axis.
            weight_dtype: projection / lm_head weight dtype, `bfloat8_b` (`DEC-022`).
            activation_dtype: the residual stream's dtype, `bfloat16` (`DEC-022`).
            tensor_cache_path: root for `ttnn.as_tensor`'s persisted tilized weights. Build it with
                `ModelArgs.weight_cache_path(dtype)`, which puts the dtype **and** the mesh shape in
                the path (`DEC-048`).
            max_seq_len: per-user KV capacity, passed to `AttentionConfig`.
            n_layers: layers to build. `None` = every layer in the config (32).
            with_lm_head: build the LM head. Default `True` (`BRINGUP_RECIPE.md:1562`).
            sequence_parallel: the SP ring path (P8).
        """
        self.mesh_device = mesh_device
        self.hf = hf
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.ccl_manager = ccl_manager
        self.vocab_size = hf["vocab_size"]
        self.hidden_size = hf["hidden_size"]
        self.n_layers = hf["num_hidden_layers"] if n_layers is None else n_layers
        self.max_seq_len = max_seq_len
        self.sequence_parallel = sequence_parallel
        self.activation_dtype = activation_dtype

        assert 0 < self.n_layers <= hf["num_hidden_layers"], (
            f"n_layers must be in (0, {hf['num_hidden_layers']}], got {self.n_layers}; a reduced "
            f"count is G-MODEL's 2- and 4-layer runs, not an arbitrary knob"
        )

        # Built ONCE and shared by every layer: Llama has no sliding-window alternation, so there is
        # no per-layer config to keep in sync (`tt/layer.py::build_attention_config`). The program
        # config's pinned 8x8 SDPA grid is validated at each `Attention.__init__`.
        self.attention_config = build_attention_config(hf, max_seq_len=max_seq_len, sequence_parallel=sequence_parallel)
        self.program_config = None  # `Attention` builds the default when this is None
        # The replicated `[1,1,32,32]` Meta RoPE transformation matrix. The cos/sin tables are per
        # chunk (`prepare_inputs_prefill`) or whole-cache (P7's indexed path), never held here.
        self.transformation_mats = {"prefill": build_transformation_mat(mesh_device)}

        self.embedding = Embedding(
            mesh_device,
            hf,
            substate(state_dict, "model.embed_tokens"),
            mesh_config=self.mesh_config,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "model.embed_tokens"),
            activation_dtype=activation_dtype,
        )
        self.layers = [
            DecoderLayer(
                mesh_device,
                hf,
                substate(state_dict, f"model.layers.{layer_idx}"),
                layer_idx,
                ccl_manager=ccl_manager,
                mesh_config=self.mesh_config,
                attention_config=self.attention_config,
                program_config=self.program_config,
                transformation_mats=self.transformation_mats,
                weight_dtype=weight_dtype,
                activation_dtype=activation_dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, f"model.layers.{layer_idx}"),
                max_seq_len=max_seq_len,
                sequence_parallel=sequence_parallel,
            )
            for layer_idx in range(self.n_layers)
        ]
        self.norm = RMSNorm(
            mesh_device,
            hf,
            substate(state_dict, "model.norm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "model.norm"),
            mesh_config=self.mesh_config,
        )
        self.lm_head = (
            LMHead(
                mesh_device,
                hf,
                substate(state_dict, "lm_head"),
                mesh_config=self.mesh_config,
                ccl_manager=ccl_manager,
                weight_dtype=weight_dtype,
                activation_dtype=activation_dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "lm_head"),
            )
            if with_lm_head
            else None
        )
        logger.info(
            f"Llama model built: {self.n_layers} layers, mesh {tuple(mesh_device.shape)}, "
            f"tp={self.mesh_config.tp}, sp={self.mesh_config.sp}, lm_head={with_lm_head}"
        )

    # ------------------------------------------------------------------------------------------
    # the engine-facing surface
    # ------------------------------------------------------------------------------------------
    def prepare_inputs_prefill(self, tokens, start_pos=0, batch_size=1, user_id=0, **kwargs):
        """Token ids -> `(hidden_states, rope_mats, None)`.

        `tokens` is a torch tensor of ids, `[S]`, `[1, S]` or `[1, 1, 1, S]`. Under
        `sequence_parallel` the ids are SP-sharded on the sequence dim across the rows and
        replicated across the TP columns (`models/demos/gpt_oss_d_p/tt/model.py:288-306`);
        otherwise they are replicated.

        Returns the **contiguous** cos/sin for positions `start_pos .. start_pos+S-1` as the second
        element. The 3-tuple shape is the templates' (`:320`); the third slot is where a
        local/global RoPE split would go, and Llama has one RoPE.
        """
        del user_id, kwargs  # accepted for interface parity; the KV slot is a forward-time argument
        if tokens.dim() == 1:
            tokens = tokens.reshape(1, -1)
        seq_len = tokens.shape[-1]
        tokens = tokens.reshape(1, 1, 1, seq_len)

        if self.sequence_parallel:
            sp = self.mesh_config.sp
            assert seq_len % sp == 0, f"SP prefill needs seq_len ({seq_len}) divisible by sp ({sp})"
            shard_dims = [None, None]
            shard_dims[self.mesh_config.sp_axis] = 3
            mesh_mapper = ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=tuple(shard_dims), mesh_shape=tuple(self.mesh_device.shape)
            )
        else:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        tt_tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        hidden_states = self.embedding(tt_tokens)
        tt_tokens.deallocate(True)

        rope_mats = build_prefill_rope(self.mesh_device, self.hf, seq_len, start_pos=start_pos)
        return hidden_states, rope_mats, None

    def prefill_forward(
        self,
        x,
        rot_mats_global=None,
        *,
        user_id=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        skip_lm_head=False,
        on_layer_complete=None,
        on_layer_output=None,
        cached_len=0,
        indexed_rope=False,
    ):
        """The prefill forward pass. `x` is `[1, 1, B*S, hidden]`; `x` is **consumed**.

        Args:
            x: embedded hidden states from `prepare_inputs_prefill`.
            rot_mats_global: `[cos, sin]`. The contiguous tables for this chunk, or — with
                `indexed_rope=True` — P7's whole-cache indexed tables.
            user_id: KV-cache slot for this user's write.
            get_last_token: when `>= 0`, slice the output down to the 32-row tile starting there
                before the norm and the head. `-1` keeps every position.
            kv_cache: a `LlamaKVCache`, or `None` for the no-cache path. **On a `(1,1)` mesh a
                model-level cache write is impossible** — the op refuses the model's 8 local KV
                heads at TP=1 (`bringup_log/00_MODEL_CARD.md` §4.1) — so P6's gates run with
                `kv_cache=None` and P8's `G-KV-TP8` owns the first real write.
            batch_size: users packed on the sequence dim.
            skip_lm_head: return post-norm hidden states instead of logits (`DEC-049`).
            on_layer_complete: `fn(layer_idx)` after each layer — P10's migration/ack seam.
            on_layer_output: `fn(layer_idx, hidden_states)` after each layer — the bring-up seam
                `G-MODEL`'s per-layer PCC curve uses. The tensor stays live and owned by the model;
                a callback must read it, not free it.
            cached_len: valid prefix already in the cache. Non-zero is refused by
                `attention_forward` on this dense path; P8's ring path makes it legal.
            indexed_rope: use the on-device indexed RoPE (P7).
        """
        assert rot_mats_global is not None, "prefill_forward needs [cos, sin]; build_prefill_rope() makes them"
        hidden_states = x

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                rot_mats_global,
                kv_cache=kv_cache,
                user_id=user_id,
                batch_size=batch_size,
                cached_len=cached_len,
                indexed_rope=indexed_rope,
            )
            if on_layer_complete is not None:
                on_layer_complete(layer_idx)
            if on_layer_output is not None:
                on_layer_output(layer_idx, hidden_states)

        if get_last_token >= 0:
            hidden_states = self._slice_last_token(hidden_states, get_last_token, batch_size)

        # The final norm ALWAYS runs — `skip_lm_head` skips only the head (`DEC-049`). This is what
        # HF's `last_hidden_state` is.
        hidden_states = self._forward_norm(hidden_states)
        if skip_lm_head or self.lm_head is None:
            return hidden_states

        logits = self.lm_head(hidden_states)
        hidden_states.deallocate(True)
        return logits

    def _forward_norm(self, hidden_states):
        normed = self.norm(hidden_states)
        hidden_states.deallocate(True)
        return normed

    def _slice_last_token(self, hidden_states, get_last_token, batch_size):
        """Keep the 32-row tile starting at `get_last_token`, per user.

        Tile-granular because `ttnn.slice` on a TILE tensor's sequence dim is
        (`models/demos/gpt_oss_d_p/tt/model.py:214-233` does the same); the caller picks the row
        inside that tile with `process_output_prefill`'s `last_token_idx`.
        """
        assert get_last_token % ttnn.TILE_SIZE == 0, (
            f"get_last_token ({get_last_token}) must be a multiple of TILE_SIZE ({ttnn.TILE_SIZE}); "
            f"the slice is tile-granular and process_output_prefill indexes inside the tile"
        )
        width = hidden_states.shape[-1]
        if batch_size > 1:
            per_user_seq = hidden_states.shape[-2] // batch_size
            tiles = []
            for b in range(batch_size):
                start = b * per_user_seq + get_last_token
                tiles.append(ttnn.slice(hidden_states, (0, 0, start, 0), (1, 1, start + ttnn.TILE_SIZE, width)))
            hidden_states.deallocate(True)
            sliced = ttnn.concat(tiles, dim=2)
            for tile in tiles:
                tile.deallocate(True)
            return sliced
        sliced = ttnn.slice(hidden_states, (0, 0, get_last_token, 0), (1, 1, get_last_token + ttnn.TILE_SIZE, width))
        hidden_states.deallocate(True)
        return sliced

    def process_output_prefill(self, tt_out, last_token_idx):
        """Device logits (or hidden states) -> one torch row, TP-gathered on the **host**.

        The vocab shard is column-parallel, so the TP columns hold *slices* and the gather is a
        concat, not a sum (`tt/lm_head.py`). Doing it on the host is what the templates do
        (`models/demos/gpt_oss_d_p/tt/model.py:324-330`) and keeps the prefill path free of a
        collective it does not need. Rows are replicated across the SP axis at SP=1; at SP>1 only
        the shard holding `last_token_idx` is meaningful, which is P8's problem, not this one's.
        """
        tp = self.mesh_config.tp
        device_tensors = ttnn.get_device_tensors(tt_out)
        if tp > 1:
            # Column-major device order over the TP axis: device i of the first row is column i.
            torch_output = torch.cat([ttnn.to_torch(device_tensors[i]) for i in range(tp)], dim=-1)
        else:
            torch_output = ttnn.to_torch(device_tensors[0])
        return torch_output[..., last_token_idx, : self.vocab_size]
