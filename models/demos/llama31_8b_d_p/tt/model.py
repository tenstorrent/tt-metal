# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B-Instruct TTNN prefill model.

::

    embedding -> DecoderLayer x num_layers -> final RMSNorm -> (LMHead)

HF anchor: ``transformers.models.llama.modeling_llama.LlamaModel`` /
``LlamaForCausalLM``.
Template: ``models/demos/gpt_oss_d_p/tt/model.py:41`` (class), ``:93`` (the layer list), ``:113``
(the final norm), ``:179`` (``_forward_layers_and_head``, including the ``on_layer_complete``
per-layer seam P10 needs), ``:246`` (``prefill_forward``), ``:279``
(``prepare_inputs_prefill``), ``:288-306`` (the SP token shard), ``:322``
(``process_output_prefill``).

**Deleted vs the template** (``03_OUTLINE.md`` §3.17): ``rot_mats_local`` (``model.py:250``, for
gpt-oss's sliding layers), ``use_ep_moe`` / ``ep_seq_len_per_chip`` / ``expert_weight_dtype``,
``compute_per_device_vocab`` (``:31``) and the on-device sampling hooks (``:145-157``,
``:166-169``) — see ``tt/lm_head.py`` for why the vocab padding goes with them.

**Added vs the template:**

* ``num_layers=None`` (default = ``hf_config.num_hidden_layers``) so ``G-MODEL`` can run the
  recipe's reduced 2- and 4-layer stacks without mutating ``hf_config`` — gpt-oss's harness mutates
  ``hf_config.num_hidden_layers`` in place
  (``models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:132``), which a frozen
  ``LlamaHFConfig`` forbids by design (``DEC-009``);
* :meth:`consumed_state_dict_keys` and :meth:`named_device_tensors`, the two accessors ``G-WEIGHTS``
  needs to prove that no checkpoint key is missing, none is silently unused, and a cache-only
  rebuild is bit-identical (``DEC-042``);
* the final norm is applied on **both** paths, not only before the LM head — see
  :meth:`prefill_forward` (``DEC-043``).
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.demos.llama31_8b_d_p.tt.attention import ProgramConfig
from models.demos.llama31_8b_d_p.tt.embedding import Embedding
from models.demos.llama31_8b_d_p.tt.layer import DecoderLayer
from models.demos.llama31_8b_d_p.tt.lm_head import LMHead
from models.demos.llama31_8b_d_p.tt.mlp import default_compute_kernel_config
from models.demos.llama31_8b_d_p.tt.rms_norm import RMSNorm
from models.demos.llama31_8b_d_p.tt.rope import build_prefill_rope, build_transformation_mat
from models.demos.llama31_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama31_8b_d_p.utils.substate import substate


class Model:
    """Llama-3.1-8B TTNN prefill model (GQA, full llama3-scaled RoPE, dense SwiGLU)."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        *,
        mesh_config,
        ccl_manager,
        max_seq_len=128 * 1024,
        num_layers=None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        sequence_parallel=False,
        scatter_output=False,
        with_lm_head=True,
    ):
        """
        Args:
            mesh_device: the ttnn mesh device.
            hf_config: a ``LlamaHFConfig``. Frozen; ``num_layers`` overrides its depth without
                mutating it.
            state_dict: the **full** checkpoint dict in HF layout and HF naming, i.e. keys
                ``model.embed_tokens.weight``, ``model.layers.<i>.<...>``, ``model.norm.weight``,
                ``lm_head.weight`` (``ModelArgs.load_state_dict``). This class does every
                ``substate`` split itself. ``{}`` means cache-only mode, which requires
                ``tensor_cache_path`` (``DEC-038``).
            mesh_config: ``MeshConfig``.
            ccl_manager: ``CCLManager``; unused at TP=1 and SP=1.
            max_seq_len: the attention config's ``max_seq_len``.
            num_layers: build only the first ``num_layers`` decoder layers. ``None`` = all 32.
            weight_dtype: on-device projection dtype (default ``bfloat8_b``).
            tensor_cache_path: directory for the tilized weight cache, or ``None``. Get it from
                ``ModelArgs.weight_cache_path(weight_dtype)``, which puts the **mesh shape** in the
                path — a cache written at one mesh shape is garbage at another (``R-017``).
            sequence_parallel: SP prefill (P8).
            scatter_output: residual scheme (``DEC-018``); ``True`` is refused by ``DecoderLayer``.
            with_lm_head: build the LM head. ``True`` (default) consumes ``lm_head.weight`` and is
                what ``G-MODEL``'s top-1 check and ``G-WEIGHTS``'s 291-key audit need. ``False``
                skips ~1 GiB of weight and makes ``skip_lm_head=False`` an error.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hf_config = hf_config
        self.vocab_size = hf_config.vocab_size
        self.hidden_size = hf_config.hidden_size
        self.head_dim = hf_config.head_dim
        self.max_seq_len = max_seq_len
        self.weight_dtype = weight_dtype
        self.sequence_parallel = sequence_parallel
        self.scatter_output = scatter_output
        self.num_layers = hf_config.num_hidden_layers if num_layers is None else num_layers
        assert (
            0 < self.num_layers <= hf_config.num_hidden_layers
        ), f"num_layers must be in [1, {hf_config.num_hidden_layers}], got {self.num_layers}"
        if self.num_layers != hf_config.num_hidden_layers:
            logger.warning(
                f"[llama31_8b_d_p] building a REDUCED {self.num_layers}-layer stack "
                f"(full model is {hf_config.num_hidden_layers} layers) — G-MODEL's ladder, not a "
                f"deployable model."
            )
        self.with_lm_head = with_lm_head

        # Built ONCE and shared: pure value objects, and rebuilding them per layer would put 32
        # avoidable host allocations in the construction path (DEC-031).
        self.program_config = ProgramConfig()
        self.compute_kernel_config = default_compute_kernel_config(mesh_device)
        # Replicated [1,1,32,32] RoPE matrix for rotary_embedding_llama / rotary_embedding_indexed.
        # The cos/sin themselves are per-chunk (prepare_inputs_prefill) or the whole-cache indexed
        # tables the runtime passes into prefill_forward.
        self.transformation_mats = {"prefill": build_transformation_mat(mesh_device)}

        self.embedding = Embedding(
            mesh_device,
            hf_config,
            substate(state_dict, "model.embed_tokens"),
            mesh_config=mesh_config,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "model.embed_tokens"),
        )

        # `DecoderLayer.state_dict_prefix` rather than a local f-string: the "model.layers.<i>."
        # convention appears in four places in this file (here, the cache path, and the two
        # G-WEIGHTS accessors) and `ModelArgs.get_state_dict_prefix` is where it is defined and
        # asserted. `substate` takes the prefix without its trailing dot.
        self.layers = [
            DecoderLayer(
                mesh_device,
                hf_config,
                substate(state_dict, DecoderLayer.state_dict_prefix(layer_idx).rstrip(".")),
                layer_idx,
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                program_config=self.program_config,
                transformation_mats=self.transformation_mats,
                max_seq_len=max_seq_len,
                weight_dtype=weight_dtype,
                tensor_cache_path=get_cache_file_name(
                    tensor_cache_path, DecoderLayer.state_dict_prefix(layer_idx).rstrip(".")
                ),
                sequence_parallel=sequence_parallel,
                scatter_output=scatter_output,
                compute_kernel_config=self.compute_kernel_config,
            )
            for layer_idx in range(self.num_layers)
        ]

        self.norm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "model.norm"),
            mesh_config=mesh_config,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "model.norm"),
            is_distributed=False,
        )

        self.lm_head = (
            LMHead(
                mesh_device,
                hf_config,
                substate(state_dict, "lm_head"),
                mesh_config=mesh_config,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "lm_head"),
                weight_dtype=weight_dtype,
                compute_kernel_config=self.compute_kernel_config,
            )
            if with_lm_head
            else None
        )

    # -------------------------------------------------------------------------------------
    # Weight introspection — what G-WEIGHTS asserts on (DEC-042)
    # -------------------------------------------------------------------------------------
    def consumed_state_dict_keys(self) -> set:
        """The exact HF checkpoint keys this instance built from.

        Derived from what was actually constructed (``num_layers``, ``with_lm_head``), not from the
        checkpoint, so ``G-WEIGHTS`` can subtract it from the checkpoint's own key set and get a
        real "silently unused" list. A key that is in the checkpoint and not in here is either a
        feature this package does not implement or the renamed twin of a key that is simultaneously
        missing (``DEC-039``).
        """
        keys = {"model.embed_tokens.weight", "model.norm.weight"}
        if self.with_lm_head:
            keys.add("lm_head.weight")
        for i in range(self.num_layers):
            for k in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ):
                keys.add(f"model.layers.{i}.{k}")
        return keys

    def named_device_tensors(self) -> dict:
        """``{name: ttnn.Tensor}`` for **every** weight this model holds on device.

        Names mirror the checkpoint keys where there is a 1:1 correspondence, so a mismatch report
        points at a checkpoint key rather than at a Python attribute. Used by ``G-WEIGHTS`` for two
        assertions no other test can make: that the count matches ``9*num_layers + 3``, and that a
        cache-only rebuild reproduces every tensor bit-for-bit (``R-017``).
        """
        out = {"model.embed_tokens.weight": self.embedding.weight, "model.norm.weight": self.norm.tt_weight}
        if self.lm_head is not None:
            out["lm_head.weight"] = self.lm_head.weight
        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}."
            out[p + "input_layernorm.weight"] = layer.input_layernorm.tt_weight
            out[p + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.tt_weight
            out[p + "mlp.gate_proj.weight"] = layer.mlp.gate_proj
            out[p + "mlp.up_proj.weight"] = layer.mlp.up_proj
            out[p + "mlp.down_proj.weight"] = layer.mlp.down_proj
            out[p + "self_attn.q_proj.weight"] = layer.self_attn.weights.wq
            out[p + "self_attn.k_proj.weight"] = layer.self_attn.weights.wk
            out[p + "self_attn.v_proj.weight"] = layer.self_attn.weights.wv
            out[p + "self_attn.o_proj.weight"] = layer.self_attn.weights.o_proj
        return out

    # -------------------------------------------------------------------------------------
    # Prefill
    # -------------------------------------------------------------------------------------
    def prepare_inputs_prefill(self, tokens, start_pos=0, batch_size=1, user_id=0, build_rope=True, **kwargs):
        """Embed + (SP-)shard one chunk's token ids, and build that chunk's RoPE tables.

        Args:
            tokens: a torch int tensor of shape ``[S]``, ``[1, S]`` or ``[1, 1, 1, S]``.
            start_pos: the global position of ``tokens[0]``.
            batch_size: users packed on the sequence dim (kept for interface parity; the SP shard
                below is the one-prompt case).
            user_id: unused here; kept so the 4 call-site kwargs match the template.
            build_rope: build the per-chunk **replicated** Meta cos/sin for positions
                ``[start_pos, start_pos + S)`` and return them as element 2, the way
                ``models/demos/minimax_m3/tt/model.py:599`` does. Pass ``False`` on the chunked
                path, where the RoPE is the whole-cache **indexed** table
                (``tt/rope.build_indexed_rope``) that the runtime builds once and hands to
                :meth:`prefill_forward` as ``rot_mats_global`` — ``DEC-044``.

        Returns:
            ``(tokens_embd, rot_mats, None)``. The 3-tuple shape is the template's
            (``models/demos/gpt_oss_d_p/tt/model.py:279``); ``rot_mats`` is ``[cos, sin]`` or
            ``None`` when ``build_rope=False``.
        """
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens, dtype=torch.int32)
        tokens = tokens.reshape(1, 1, 1, -1)
        seq_total = tokens.shape[-1]

        if self.sequence_parallel:
            # SP prefill: ONE prompt of seq_total tokens, sharded by SEQUENCE across the SP rows and
            # replicated across the TP cols. Each device embeds its 1/sp seq-shard.
            sp = self.mesh_device.shape[self.mesh_config.sp_axis]
            assert seq_total % sp == 0, f"SP prefill needs seq_len ({seq_total}) divisible by sp ({sp})"
            tdims = [None, None]
            tdims[self.mesh_config.sp_axis] = 3  # seq dim across the SP rows
            mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=tuple(tdims), mesh_shape=self.mesh_device.shape)
        else:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        tt_tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        tokens_embd = self.embedding(tt_tokens)
        tt_tokens.deallocate(True)

        rot_mats = None
        if build_rope:
            assert not self.sequence_parallel, (
                "prepare_inputs_prefill(build_rope=True) builds a REPLICATED per-chunk RoPE, which "
                "is wrong under sequence parallelism: SP row r holds positions "
                "[r*S/sp, (r+1)*S/sp) and would be rotated from position 0. Use "
                "tt/rope.build_indexed_rope + prefill_forward(indexed_rope=True) (DEC-044)."
            )
            rot_mats = build_prefill_rope(self.mesh_device, self.hf_config, seq_len=seq_total, start_pos=start_pos)

        return tokens_embd, rot_mats, None

    def _forward_layers_and_head(
        self,
        hidden_states,
        rope_mats,
        get_last_token=-1,
        user_id=0,
        batch_size=1,
        skip_lm_head=True,
        on_layer_complete=None,
        kv_cache=None,
        cached_len=0,
        indexed_rope=False,
    ):
        """The layer loop, the last-token slice, the final norm and the optional LM head.

        ``on_layer_complete``: optional ``fn(layer_idx, hidden_states)`` invoked after each decoder
        layer — the per-layer seam P10's KV migration hangs off, and the seam ``G-MODEL`` reads the
        per-layer hidden-state PCC curve from. It takes **two** arguments here, where the template's
        takes one (``models/demos/gpt_oss_d_p/tt/model.py:211``): a callback that cannot see the
        activation cannot produce the curve the recipe asks for, and a per-depth rebuild of a
        32-layer model is not a substitute (``DEC-045``). The tensor is live and must not be
        deallocated by the callback.
        """
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=rope_mats,
                kv_cache=kv_cache,
                user_id=user_id,
                batch_size=batch_size,
                cached_len=cached_len,
                indexed_rope=indexed_rope,
            )
            if on_layer_complete is not None:
                on_layer_complete(i, hidden_states)

        if get_last_token != -1:
            if len(hidden_states.shape) == 3:
                hidden_states = ttnn.unsqueeze(hidden_states, dim=1)
            if batch_size > 1:
                per_user_seq = hidden_states.shape[2] // batch_size
                tiles = []
                for b in range(batch_size):
                    start = b * per_user_seq + get_last_token
                    tiles.append(
                        ttnn.slice(hidden_states, (0, 0, start, 0), (1, 1, start + 32, hidden_states.shape[-1]))
                    )
                hidden_states.deallocate(True)
                hidden_states = ttnn.concat(tiles, dim=2)
                for t in tiles:
                    t.deallocate(True)
            else:
                sliced = ttnn.slice(
                    hidden_states, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, hidden_states.shape[-1])
                )
                hidden_states.deallocate(True)
                hidden_states = sliced

        # DEC-043: the final norm runs on BOTH paths. The template applies it only before the LM
        # head (``models/demos/gpt_oss_d_p/tt/model.py:236-241``), so `skip_lm_head=True` there
        # returns a PRE-norm tensor,
        # which is not `LlamaModel.last_hidden_state` and is not what G-MODEL must compare against.
        # RMSNorm is row-wise, so this is one op and cannot change the KV cache — the deployment
        # path's actual product is unaffected.
        pre_norm = hidden_states
        hidden_states = self.norm(hidden_states)
        pre_norm.deallocate(True)

        if skip_lm_head:
            return hidden_states

        assert self.lm_head is not None, (
            "prefill_forward(skip_lm_head=False) needs the LM head, but this Model was built with " "with_lm_head=False"
        )
        logits = self.lm_head(hidden_states)
        hidden_states.deallocate(True)
        return logits

    def prefill_forward(
        self,
        x,
        rot_mats_global=None,
        user_id=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        skip_lm_head=True,
        on_layer_complete=None,
        cached_len=0,
        indexed_rope=False,
    ):
        """Prefill forward over one chunk.

        Args:
            x: ``[1, 1, batch*S_loc, hidden]`` bf16 TILE — element 0 of
                :meth:`prepare_inputs_prefill`. **Consumed.**
            rot_mats_global: ``[cos, sin]``. Either the per-chunk replicated tables (element 1 of
                :meth:`prepare_inputs_prefill`) or, with ``indexed_rope=True``, the whole-cache
                block-cyclic SP tables from ``tt/rope.build_indexed_rope``.
            user_id: KV-cache slot.
            get_last_token: if not ``-1``, slice the 32-row tile starting at this row before the
                final norm / LM head.
            kv_cache: a ``LlamaKVCache`` or ``None``.
            batch_size: users packed on the sequence dim.
            skip_lm_head: default ``True`` — prefill's product is the KV cache, and the LM head is
                only needed for ``G-MODEL``'s top-1 check.
            on_layer_complete: ``fn(layer_idx, hidden_states)``, see
                :meth:`_forward_layers_and_head`.
            cached_len: valid prefix already in the cache before this chunk.
            indexed_rope: use the on-device indexed RoPE.

        Returns:
            ``[1, 1, S_loc, hidden]`` when ``skip_lm_head`` (post final norm, ``DEC-043``), else
            ``[1, 1, S_loc, vocab/tp]``; ``S_loc`` is 32 when ``get_last_token != -1``.
        """
        assert rot_mats_global is not None, (
            "prefill_forward needs rot_mats_global: Llama applies FULL rotary in every layer, so a "
            "None here would be a positionless prefill that still returns the right shape. Pass "
            "element 1 of prepare_inputs_prefill(), or tt/rope.build_indexed_rope(...) with "
            "indexed_rope=True."
        )
        return self._forward_layers_and_head(
            hidden_states=x,
            rope_mats=rot_mats_global,
            kv_cache=kv_cache,
            get_last_token=get_last_token,
            user_id=user_id,
            batch_size=batch_size,
            skip_lm_head=skip_lm_head,
            on_layer_complete=on_layer_complete,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )

    def process_output_prefill(self, tt_out, last_token_idx):
        """Host-side TP gather of the logits, then one row: ``[vocab_size]``.

        The LM head is column-parallel and contributes **no** collective (``DEC-015``), so the TP
        concat happens here on the host — exactly as ``models/demos/gpt_oss_d_p/tt/model.py:322``
        does. ``[..., :vocab_size]`` is a no-op for this package (there is no vocab padding) and is
        kept so the slice stays correct if a padded head is ever added.
        """
        tp = self.mesh_config.tp
        device_tensors = ttnn.get_device_tensors(tt_out)
        if tp > 1:
            torch_output = torch.cat([ttnn.to_torch(device_tensors[i]) for i in range(tp)], dim=-1)
        else:
            torch_output = ttnn.to_torch(device_tensors[0])
        return torch_output[..., last_token_idx, : self.vocab_size]
