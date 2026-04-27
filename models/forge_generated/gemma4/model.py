"""Top-level Gemma4 model orchestration (Gemma4Model + Gemma4ForCausalLM).

Construction is HF-state-dict-driven: `from_state_dict(hf, mesh_device,
*, is_decode)` builds the full module tree (60 decoder layers, scaled
embedding, LM head, sliding/full preludes, L58/L59 specials). Every
weight, scalar constant, and RoPE inv-freq tensor lives as a named
instance attribute; the runtime call path takes only the runtime input
list (KV caches, position IDs, token IDs).
"""
import gemma4
from gemma4 import utils
from gemma4 import weights as gw
from gemma4.caches import Gemma4Caches
from gemma4.layer_table import LAYER_TABLE_DECODE, LAYER_TABLE_PREFILL

import ttnn


class Gemma4ForCausalLM:
    """Full Gemma4 forward: embedding → preludes → 60-layer loop → LMHead.

    `is_decode` is fixed at construction time. Decode runs L0..L58 in
    the regular layer loop and L59 (terminal) as `self.l59(...)`.
    Prefill runs L0..L57 in the loop and treats L58/L59 explicitly
    (L58 has dead prestage concats in the codegen; L59 is terminal).
    `__call__` returns the LMHead logits tensor.
    """

    def __init__(
        self,
        *,
        is_decode,
        scaled_embedding,
        layers,
        lm_head,
        shared,
        causal_mask_one_hot,
        sliding_prelude,
        full_prelude,
        l58,
        l59,
        caches,
    ):
        self._is_decode = is_decode
        self.scaled_embedding = scaled_embedding
        self.layers = layers
        self.lm_head = lm_head
        self.sliding_prelude = sliding_prelude
        self.full_prelude = full_prelude
        # L58/L59 specials — DecoderLayer instances used as weight
        # bundles by the verbatim L58/L59 bodies in this file.
        self.l58 = l58
        self.l59 = l59
        self.shared = shared  # dict of var_185..var_193 device tensors
        # bf16 (1,1,256,1) one_hot helper used to build the causal mask
        # at the top of _call_*.
        self.causal_mask_one_hot = causal_mask_one_hot
        # Per-layer KV cache buffers. Each layer also holds direct refs
        # to its slice (set in from_state_dict); reset_kv_caches keeps
        # both views in sync.
        self.caches = caches

    def reset_kv_caches(self):
        """Re-zero every per-layer K/V buffer. Call between independent
        generation sessions; Phase 2 also calls it at the start of each
        forward pass for PCC reproducibility (this will be relaxed when
        the Generator owns multi-step state in Phase 5).
        """
        self.caches.reset()
        # Re-distribute fresh refs. Layer iteration covers L0..L57 (decode
        # adds L58); l58 and l59 specials are reattached explicitly.
        for layer in self.layers:
            layer.k_cache = self.caches.k_caches[layer.layer_idx]
            layer.v_cache = self.caches.v_caches[layer.layer_idx]
        if self.l58 is not None:
            self.l58.k_cache = self.caches.k_caches[self.l58.layer_idx]
            self.l58.v_cache = self.caches.v_caches[self.l58.layer_idx]
        self.l59.k_cache = self.caches.k_caches[self.l59.layer_idx]
        self.l59.v_cache = self.caches.v_caches[self.l59.layer_idx]

    def __call__(self, input):
        # Phase 2 temporary: reset on every call so single-shot PCC tests
        # see the same zero initial state as the legacy input-slot path.
        # Phase 5 will move this responsibility to the Generator.
        self.reset_kv_caches()
        if self._is_decode:
            return self._call_decode(input)
        else:
            return self._call_prefill(input)

    @classmethod
    def from_state_dict(cls, hf, mesh_device, *, is_decode, seq_len=19, caches=None):
        """Build the full model from an HfWeights bundle. Every weight
        and scalar constant becomes an instance attribute; nothing is
        kept around as a runtime dict.

        `seq_len` is the prefill sequence length (default 19, matching
        the codegen-baked artifact). Decode mode ignores it (decode
        seq_len is always 1). To change it, the caller must also build
        the runtime input list at the same seq_len (see
        `synthesize_prefill_inputs`).

        `caches` is an optional pre-built `Gemma4Caches` to share across
        a prefill+decode pair (Phase 5 generator). If None, allocate
        fresh zero caches for this instance.
        """
        layer_table = LAYER_TABLE_DECODE if is_decode else LAYER_TABLE_PREFILL
        if caches is None:
            caches = Gemma4Caches(
                mesh_device,
                [layer_table[i]["type"] for i in range(60)],
            )

        # Build the no-input scalar constants and RoPE inv_freq tables into
        # a transient dict. apply_hf_scalar_overrides covers all the simple
        # tensor constants (zeros, fills, aranges, one-hot mask helper);
        # RoPESetup populates the sliding/full inv_freq matmul operands.
        # The dict is only used for prelude construction below; nothing
        # references it after that.
        transient_cm: dict = {}
        gw.apply_hf_scalar_overrides(transient_cm, hf, mesh_device, is_decode=is_decode, seq_len=seq_len)
        gemma4.RoPESetup.from_hf(hf, mesh_device, is_decode=is_decode).populate_cached_main(transient_cm)

        # Per-RMSNorm eps tensor.
        rms_eps_tensor = transient_cm["main_const_eval_240"][0]

        # One-hot causal mask helper. Same recipe in both modes (bf16
        # (1,1,256,1) one_hot at the last position); the codegen numbered
        # them differently per artifact (ce_535 decode, ce_536 prefill).
        causal_mask_one_hot = transient_cm["main_const_eval_535" if is_decode else "main_const_eval_536"][0]

        # softcap tensor (bf16 (1,1,1) fill=final_logit_softcapping).
        softcap = transient_cm["main_const_eval_171" if is_decode else "main_const_eval_314"][0]

        # Top-level submodules.
        scaled_embedding = gemma4.ScaledEmbedding.from_state_dict(
            hf.state_dict,
            hf.lifted,
            mesh_device,
        )

        # Per-layer construction. Decode runs L0..L58 through the loop and
        # treats L59 as a special method; prefill runs L0..L57 and treats
        # both L58 and L59 as specials.
        n_loop = 59 if is_decode else 58
        layers = []
        for i in range(n_loop):
            t = layer_table[i]
            slots = tuple(t["runtime_inputs"])
            if t["type"] == "sliding":
                layer = gemma4.SlidingDecoderLayer.from_state_dict(
                    hf.state_dict,
                    i,
                    mesh_device,
                    is_decode=is_decode,
                    runtime_slots=slots,
                    k_cache=caches.k_caches[i],
                    v_cache=caches.v_caches[i],
                    rms_eps_tensor=rms_eps_tensor,
                    seq_len=seq_len,
                )
            else:
                update_idxs_slot = layer_table[i - 1]["runtime_inputs"][2] if is_decode else None
                layer = gemma4.FullDecoderLayer.from_state_dict(
                    hf.state_dict,
                    i,
                    mesh_device,
                    is_decode=is_decode,
                    runtime_slots=slots,
                    update_idxs_slot=update_idxs_slot,
                    k_cache=caches.k_caches[i],
                    v_cache=caches.v_caches[i],
                    rms_eps_tensor=rms_eps_tensor,
                    seq_len=seq_len,
                )
            layers.append(layer)

        # Shared scalars consumed directly by the regular layer body via
        # the `shared` kwarg passed through __call__.
        if is_decode:
            shared = {
                "var_185": transient_cm["main_const_eval_0"][1],
                "var_186": transient_cm["main_const_eval_0"][2],
                "var_188": rms_eps_tensor,
                "var_190": transient_cm["main_const_eval_334"][0],
                "var_191": transient_cm["main_const_eval_337"][0],
                "var_192": transient_cm["main_const_eval_486"][0],
                "var_193": transient_cm["main_const_eval_489"][0],
            }
        else:
            shared = {
                "var_185": transient_cm["main_const_eval_0"][1],
                "var_186": transient_cm["main_const_eval_0"][2],
                "var_187": transient_cm["main_const_eval_186"][0],
                "var_188": rms_eps_tensor,
                "var_190": transient_cm["main_const_eval_266"][0],
                "var_192": transient_cm["main_const_eval_335"][0],
                "var_193": transient_cm["main_const_eval_338"][0],
            }

        # Preludes: bind cached_main once at construction; runtime calls
        # don't pass it.
        if is_decode:
            sliding_prelude = gemma4.SlidingPreludeDecode.from_consteval(transient_cm)
            full_prelude = gemma4.FullPreludeDecode.from_consteval(transient_cm)
        else:
            sliding_prelude = gemma4.SlidingPreludePrefill.from_consteval(transient_cm, seq_len=seq_len)
            full_prelude = gemma4.FullPreludePrefill.from_consteval(transient_cm, seq_len=seq_len)

        # L58 (sliding) special — only used in prefill mode (decode runs L58
        # through the regular layer loop).
        if is_decode:
            l58 = None
        else:
            l58 = gemma4.SlidingDecoderLayer.from_state_dict(
                hf.state_dict,
                58,
                mesh_device,
                is_decode=False,
                runtime_slots=tuple(layer_table[58]["runtime_inputs"]),
                k_cache=caches.k_caches[58],
                v_cache=caches.v_caches[58],
                rms_eps_tensor=rms_eps_tensor,
                seq_len=seq_len,
            )

        # L59 (full) — terminal layer in both modes:
        #   - runtime_slots is 2-tuple (k, v) — L59 has no pos_ids slot.
        #   - is_terminal=True skips pos_ids read, attention's
        #     position-increment computation, and the layer_scalar multiply
        #     (LMHead.last_layer_scalar = l59.layer_scalar absorbs the latter).
        #   - update_idxs in decode mode points at L58's pos_ids slot.
        l59_slots = tuple(layer_table[59]["runtime_inputs"])
        l59_update_idxs_slot = layer_table[58]["runtime_inputs"][2] if is_decode else None
        l59 = gemma4.FullDecoderLayer.from_state_dict(
            hf.state_dict,
            59,
            mesh_device,
            is_decode=is_decode,
            runtime_slots=l59_slots,
            update_idxs_slot=l59_update_idxs_slot,
            k_cache=caches.k_caches[59],
            v_cache=caches.v_caches[59],
            rms_eps_tensor=rms_eps_tensor,
            is_terminal=True,
            seq_len=seq_len,
        )

        # LMHead: norm_weight and lm_head_weight (tied to embed_tokens)
        # come from HF state_dict directly.
        lm_head = gemma4.LMHead.from_state_dict(
            hf.state_dict,
            mesh_device,
            rms_eps=rms_eps_tensor,
            last_layer_scalar=l59.layer_scalar,
            softcap=softcap,
        )

        return cls(
            is_decode=is_decode,
            scaled_embedding=scaled_embedding,
            layers=layers,
            lm_head=lm_head,
            shared=shared,
            causal_mask_one_hot=causal_mask_one_hot,
            sliding_prelude=sliding_prelude,
            full_prelude=full_prelude,
            l58=l58,
            l59=l59,
            caches=caches,
        )

    def _call_decode(self, input):
        """Decode forward pass: prelude → 59-layer loop → L59 special →
        postlude. All weights and constants flow through self.X.
        """
        var_0 = input[0]
        var_2 = input[7]
        var_3 = input[9]
        var_7 = input[26]
        var_185 = self.shared["var_185"]
        utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 4))
        ttnn_to_layout_0 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(var_0, False)
        ttnn_add_0 = ttnn.add(
            ttnn_to_layout_0,
            var_185,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_to_layout_1 = ttnn.to_layout(var_3, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(var_3, False)
        ttnn_reshape_0 = ttnn.reshape(
            ttnn_to_layout_1,
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_1, False)
        ttnn_logical_and_0 = ttnn.logical_and(
            ttnn_reshape_0,
            self.causal_mask_one_hot,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_logical_not_0 = ttnn.logical_not(
            ttnn_logical_and_0,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_to_layout_2 = ttnn.to_layout(var_2, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(var_2, False)
        ttnn_multiply_0 = self.scaled_embedding(ttnn_to_layout_2)
        (
            ttnn_typecast_2,
            ttnn_typecast_3,
            ttnn_reshape_3,
            ttnn_reshape_18,
            ttnn_to_layout_11,
            ttnn_typecast_11,
        ) = self.sliding_prelude(
            input=input,
            ttnn_to_layout_0=ttnn_to_layout_0,
            ttnn_add_0=ttnn_add_0,
        )
        (
            ttnn_typecast_35,
            ttnn_typecast_36,
            ttnn_typecast_39,
        ) = self.full_prelude(
            input=input,
            ttnn_reshape_0=ttnn_reshape_0,
            ttnn_reshape_3=ttnn_reshape_3,
            ttnn_reshape_18=ttnn_reshape_18,
            ttnn_to_layout_11=ttnn_to_layout_11,
        )
        # Orphan dealloc moved here from inside the deleted _sliding_decoder_layer_0 body.
        # `var_7 = input[26]` (the position-id helper) is consumed inside _sliding_prelude;
        # the original codegen freed it after the layer 0 body, so we mirror that timing.
        ttnn.deallocate(var_7, False)
        hidden = ttnn_multiply_0
        sliding_state = dict(
            causal_mask_logical_and=ttnn_logical_and_0,
            causal_mask_logical_not=ttnn_logical_not_0,
            sliding_cos_cache=ttnn_typecast_2,
            sliding_sin_cache=ttnn_typecast_3,
            pos_typecast_11=ttnn_typecast_11,
        )
        full_state = dict(
            full_cos_cache=ttnn_typecast_35,
            full_sin_cache=ttnn_typecast_36,
            full_pos_mask=ttnn_typecast_39,
        )
        # Run L0..L58 through the regular loop. Each layer returns (residual,
        # *kv_outputs); the kv_outputs were captured by the legacy _main
        # return tuple but are unused after the model output trim — KV cache
        # writes happen as side effects inside the layer body.
        for layer in self.layers:
            hidden = layer(
                hidden,
                sliding_state=sliding_state,
                full_state=full_state,
                input=input,
                shared=self.shared,
            )

        ttnn_add_601 = self.l59(
            hidden,
            sliding_state=sliding_state,
            full_state=full_state,
            input=input,
            shared=self.shared,
        )
        ttnn_multiply_1084 = self.lm_head(ttnn_add_601)
        return ttnn_multiply_1084

    def _call_prefill(self, input):
        """Prefill forward pass: prelude → 58-layer loop → L58/L59 specials
        → postlude. All weights and constants flow through self.X.
        """
        var_0 = input[0]
        var_2 = input[7]
        var_3 = input[9]
        var_7 = input[26]
        var_185 = self.shared["var_185"]
        var_186 = self.shared["var_186"]
        var_187 = self.shared["var_187"]
        var_188 = self.shared["var_188"]
        var_190 = self.shared["var_190"]
        var_192 = self.shared["var_192"]
        var_193 = self.shared["var_193"]
        # Side-effecting device init — the assignment is not read but the call
        # registers the multi-device mesh used by every subsequent ttnn op.
        # Do NOT remove despite "unused variable" warnings.
        utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 4))  # noqa: F841
        ttnn_to_layout_0 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(var_0, False)
        ttnn_add_0 = ttnn.add(
            ttnn_to_layout_0,
            var_185,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_to_layout_1 = ttnn.to_layout(var_3, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(var_3, False)
        ttnn_reshape_0 = ttnn.reshape(
            ttnn_to_layout_1,
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_1, False)
        ttnn_logical_and_0 = ttnn.logical_and(
            ttnn_reshape_0,
            self.causal_mask_one_hot,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_logical_not_0 = ttnn.logical_not(
            ttnn_logical_and_0,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_to_layout_2 = ttnn.to_layout(var_2, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(var_2, False)
        ttnn_multiply_0 = self.scaled_embedding(ttnn_to_layout_2)
        (
            ttnn_typecast_2,
            ttnn_typecast_3,
            ttnn_reshape_3,
            ttnn_reshape_15,
            ttnn_reshape_16,
            ttnn_reshape_18,
            ttnn_to_layout_11,
            ttnn_typecast_11,
        ) = self.sliding_prelude(
            input=input,
            ttnn_to_layout_0=ttnn_to_layout_0,
        )
        # `var_7` (= input[26], the position-id tensor) is consumed inside
        # `_sliding_prelude` but the prelude does NOT deallocate it (it lives in
        # this scope as a Python alias of the same underlying ttnn tensor). The
        # original codegen freed it at L1381 of the inlined layer 0 body; we
        # preserve that timing here. If `_sliding_prelude` is ever changed to
        # deallocate `var_7` internally, this line becomes a double-free.
        ttnn.deallocate(var_7, False)

        (
            ttnn_reshape_104,
            ttnn_reshape_105,
            ttnn_typecast_39,
        ) = self.full_prelude(
            input=input,
            ttnn_reshape_0=ttnn_reshape_0,
            ttnn_reshape_3=ttnn_reshape_3,
            ttnn_reshape_18=ttnn_reshape_18,
            ttnn_to_layout_11=ttnn_to_layout_11,
        )

        # Run the 58 well-formed decoder layers (0..57) in a single loop driven
        # by `self.layers`. Each layer is an instance of `SlidingDecoderLayer`
        # or `FullDecoderLayer` that knows its own `layer_idx` and dispatches to
        # the matching parameterized helper. L58 + L59 stay inline below as
        # documented special cases.
        hidden = ttnn_multiply_0
        sliding_state = {
            "causal_mask_logical_and": ttnn_logical_and_0,
            "causal_mask_logical_not": ttnn_logical_not_0,
            "sliding_cos_cache": ttnn_typecast_2,
            "sliding_sin_cache": ttnn_typecast_3,
            "pos_reshape_15": ttnn_reshape_15,
            "pos_reshape_16": ttnn_reshape_16,
            "pos_typecast_11": ttnn_typecast_11,
        }
        full_state = {
            "full_cos_cache": ttnn_reshape_104,
            "full_sin_cache": ttnn_reshape_105,
            "full_pos_mask": ttnn_typecast_39,
        }
        # Run L0..L57 through the regular loop. Each layer returns (residual,
        # *kv_outputs); the kv_outputs were captured by the legacy _main
        # return tuple but are unused after the model output trim — KV cache
        # writes happen as side effects inside the layer body.
        for layer in self.layers:
            hidden = layer(
                hidden,
                sliding_state=sliding_state,
                full_state=full_state,
                input=input,
                shared=self.shared,
            )

        # L58 (sliding) — also flows through SlidingDecoderLayer. The codegen's
        # extra prestage concat ops are dropped (their outputs were consumed
        # only by the legacy return tuple).
        ttnn_multiply_1062 = self.l58(
            hidden,
            sliding_state=sliding_state,
            full_state=full_state,
            input=input,
            shared=self.shared,
        )
        # L59 in prefill consumes sliding_state's prelude outputs unchanged
        # via __call__, but the full_state for L59 prefill uses the
        # ttnn_reshape_104/105 (post-prelude) that the regular full layers
        # got from the prelude. The full prelude in prefill produces
        # ttnn_reshape_104/105 (instead of ttnn_typecast_35/36), so build
        # the full_state explicitly for L59.
        ttnn_add_602 = self.l59(
            ttnn_multiply_1062,
            sliding_state=sliding_state,
            full_state=dict(
                full_cos_cache=ttnn_reshape_104,
                full_sin_cache=ttnn_reshape_105,
                full_pos_mask=ttnn_typecast_39,
            ),
            input=input,
            shared=self.shared,
        )
        ttnn_multiply_1084 = self.lm_head(ttnn_add_602)
        return ttnn_multiply_1084
