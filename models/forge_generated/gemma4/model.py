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
from gemma4.layer_table import LAYER_TABLE_DECODE, LAYER_TABLE_PREFILL

import ttnn


class Gemma4Model:
    """Decoder-only stack: scaled embed → preludes → 60 layers → final
    norm. Placeholder; the real body lives in Gemma4ForCausalLM until
    a future commit splits the postlude out.
    """

    def __init__(self, *, is_decode):
        self._is_decode = is_decode


class Gemma4ForCausalLM:
    """Full Gemma4 forward: embedding → preludes → decoder layer loop →
    L58/L59 specials → LMHead postlude. Returns the legacy `_main`
    return tuple (logits + per-layer KV outputs in the order the
    codegen captured).

    `is_decode` is fixed at construction time.
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
        self.model = Gemma4Model(is_decode=is_decode)

    def __call__(self, input):
        if self._is_decode:
            return self._call_decode(input)
        else:
            return self._call_prefill(input)

    @classmethod
    def from_state_dict(cls, hf, mesh_device, *, is_decode):
        """Build the full model from an HfWeights bundle. Every weight
        and scalar constant becomes an instance attribute; nothing is
        kept around as a runtime dict.
        """
        layer_table = LAYER_TABLE_DECODE if is_decode else LAYER_TABLE_PREFILL

        # Build the no-input scalar constants and RoPE inv_freq tables into
        # a transient dict. apply_hf_scalar_overrides covers all the simple
        # tensor constants (zeros, fills, aranges, one-hot mask helper);
        # RoPESetup populates the sliding/full inv_freq matmul operands.
        # The dict is only used for prelude construction below; nothing
        # references it after that.
        transient_cm: dict = {}
        gw.apply_hf_scalar_overrides(transient_cm, hf, mesh_device, is_decode=is_decode)
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
                    rms_eps_tensor=rms_eps_tensor,
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
                    rms_eps_tensor=rms_eps_tensor,
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
            sliding_prelude = gemma4.SlidingPreludePrefill.from_consteval(transient_cm)
            full_prelude = gemma4.FullPreludePrefill.from_consteval(transient_cm)

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
                rms_eps_tensor=rms_eps_tensor,
            )

        # L59 (full) special — used in both modes. L59's runtime_inputs has
        # only 2 entries (no scalar); pad with a sentinel slot that the
        # special body never accesses.
        l59_slots = tuple(layer_table[59]["runtime_inputs"]) + (0,)
        l59 = gemma4.FullDecoderLayer.from_state_dict(
            hf.state_dict,
            59,
            mesh_device,
            is_decode=is_decode,
            runtime_slots=l59_slots,
            update_idxs_slot=None,
            rms_eps_tensor=rms_eps_tensor,
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
        )

    def _call_decode(self, input):
        """Decode forward pass: prelude → 59-layer loop → L59 special →
        postlude. All weights and constants flow through self.X.
        """
        var_0 = input[0]
        var_2 = input[7]
        var_3 = input[9]
        var_7 = input[26]
        var_19 = input[98]
        var_20 = input[99]
        var_37 = input[199]
        var_38 = input[200]
        var_55 = input[300]
        var_56 = input[301]
        var_73 = input[401]
        var_74 = input[402]
        var_91 = input[502]
        var_92 = input[503]
        var_109 = input[603]
        var_110 = input[604]
        var_127 = input[704]
        var_128 = input[705]
        var_145 = input[805]
        var_146 = input[806]
        var_163 = input[906]
        var_164 = input[907]
        var_181 = input[1007]
        var_182 = input[1008]
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
        layer_kv_outputs = [None] * 59
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
        for layer_idx, layer in enumerate(self.layers):
            result = layer(
                hidden,
                sliding_state=sliding_state,
                full_state=full_state,
                input=input,
                shared=self.shared,
            )
            hidden = result[0]
            layer_kv_outputs[layer_idx] = result[1:]

        # Restore original per-layer tensor names so the inline L58/L59
        # bodies and the postlude return list continue to resolve.
        ttnn_multiply_1062 = hidden
        ttnn_add_3, ttnn_where_1, ttnn_where_3 = layer_kv_outputs[0]
        ttnn_add_15, ttnn_where_8, ttnn_where_10 = layer_kv_outputs[1]
        ttnn_add_25, ttnn_where_13, ttnn_where_15 = layer_kv_outputs[2]
        ttnn_add_35, ttnn_where_18, ttnn_where_20 = layer_kv_outputs[3]
        ttnn_add_45, ttnn_where_23, ttnn_where_25 = layer_kv_outputs[4]
        (ttnn_add_55,) = layer_kv_outputs[5]
        ttnn_add_65, ttnn_where_30, ttnn_where_32 = layer_kv_outputs[6]
        ttnn_add_75, ttnn_where_35, ttnn_where_37 = layer_kv_outputs[7]
        ttnn_add_85, ttnn_where_40, ttnn_where_42 = layer_kv_outputs[8]
        ttnn_add_95, ttnn_where_45, ttnn_where_47 = layer_kv_outputs[9]
        ttnn_add_105, ttnn_where_50, ttnn_where_52 = layer_kv_outputs[10]
        (ttnn_add_115,) = layer_kv_outputs[11]
        ttnn_add_125, ttnn_where_56, ttnn_where_58 = layer_kv_outputs[12]
        ttnn_add_135, ttnn_where_61, ttnn_where_63 = layer_kv_outputs[13]
        ttnn_add_145, ttnn_where_66, ttnn_where_68 = layer_kv_outputs[14]
        ttnn_add_155, ttnn_where_71, ttnn_where_73 = layer_kv_outputs[15]
        ttnn_add_165, ttnn_where_76, ttnn_where_78 = layer_kv_outputs[16]
        (ttnn_add_175,) = layer_kv_outputs[17]
        ttnn_add_185, ttnn_where_82, ttnn_where_84 = layer_kv_outputs[18]
        ttnn_add_195, ttnn_where_87, ttnn_where_89 = layer_kv_outputs[19]
        ttnn_add_205, ttnn_where_92, ttnn_where_94 = layer_kv_outputs[20]
        ttnn_add_215, ttnn_where_97, ttnn_where_99 = layer_kv_outputs[21]
        ttnn_add_225, ttnn_where_102, ttnn_where_104 = layer_kv_outputs[22]
        (ttnn_add_235,) = layer_kv_outputs[23]
        ttnn_add_245, ttnn_where_108, ttnn_where_110 = layer_kv_outputs[24]
        ttnn_add_255, ttnn_where_113, ttnn_where_115 = layer_kv_outputs[25]
        ttnn_add_265, ttnn_where_118, ttnn_where_120 = layer_kv_outputs[26]
        ttnn_add_275, ttnn_where_123, ttnn_where_125 = layer_kv_outputs[27]
        ttnn_add_285, ttnn_where_128, ttnn_where_130 = layer_kv_outputs[28]
        (ttnn_add_295,) = layer_kv_outputs[29]
        ttnn_add_305, ttnn_where_134, ttnn_where_136 = layer_kv_outputs[30]
        ttnn_add_315, ttnn_where_139, ttnn_where_141 = layer_kv_outputs[31]
        ttnn_add_325, ttnn_where_144, ttnn_where_146 = layer_kv_outputs[32]
        ttnn_add_335, ttnn_where_149, ttnn_where_151 = layer_kv_outputs[33]
        ttnn_add_345, ttnn_where_154, ttnn_where_156 = layer_kv_outputs[34]
        (ttnn_add_355,) = layer_kv_outputs[35]
        ttnn_add_365, ttnn_where_160, ttnn_where_162 = layer_kv_outputs[36]
        ttnn_add_375, ttnn_where_165, ttnn_where_167 = layer_kv_outputs[37]
        ttnn_add_385, ttnn_where_170, ttnn_where_172 = layer_kv_outputs[38]
        ttnn_add_395, ttnn_where_175, ttnn_where_177 = layer_kv_outputs[39]
        ttnn_add_405, ttnn_where_180, ttnn_where_182 = layer_kv_outputs[40]
        (ttnn_add_415,) = layer_kv_outputs[41]
        ttnn_add_425, ttnn_where_186, ttnn_where_188 = layer_kv_outputs[42]
        ttnn_add_435, ttnn_where_191, ttnn_where_193 = layer_kv_outputs[43]
        ttnn_add_445, ttnn_where_196, ttnn_where_198 = layer_kv_outputs[44]
        ttnn_add_455, ttnn_where_201, ttnn_where_203 = layer_kv_outputs[45]
        ttnn_add_465, ttnn_where_206, ttnn_where_208 = layer_kv_outputs[46]
        (ttnn_add_475,) = layer_kv_outputs[47]
        ttnn_add_485, ttnn_where_212, ttnn_where_214 = layer_kv_outputs[48]
        ttnn_add_495, ttnn_where_217, ttnn_where_219 = layer_kv_outputs[49]
        ttnn_add_505, ttnn_where_222, ttnn_where_224 = layer_kv_outputs[50]
        ttnn_add_515, ttnn_where_227, ttnn_where_229 = layer_kv_outputs[51]
        ttnn_add_525, ttnn_where_232, ttnn_where_234 = layer_kv_outputs[52]
        (ttnn_add_535,) = layer_kv_outputs[53]
        ttnn_add_545, ttnn_where_238, ttnn_where_240 = layer_kv_outputs[54]
        ttnn_add_555, ttnn_where_243, ttnn_where_245 = layer_kv_outputs[55]
        ttnn_add_565, ttnn_where_248, ttnn_where_250 = layer_kv_outputs[56]
        ttnn_add_575, ttnn_where_253, ttnn_where_255 = layer_kv_outputs[57]
        ttnn_add_585, ttnn_where_258, ttnn_where_260 = layer_kv_outputs[58]
        ttnn_add_601 = self._full_layer_59_decode(
            hidden_state=ttnn_multiply_1062,
            full_cos_cache=ttnn_typecast_35,
            full_sin_cache=ttnn_typecast_36,
            full_pos_mask=ttnn_typecast_39,
            input=input,
        )
        ttnn_multiply_1084 = self.lm_head(ttnn_add_601)
        return [
            ttnn_add_0,
            ttnn_where_1,
            ttnn_where_3,
            ttnn_add_3,
            ttnn_where_8,
            ttnn_where_10,
            ttnn_add_15,
            ttnn_where_13,
            ttnn_where_15,
            ttnn_add_25,
            ttnn_where_18,
            ttnn_where_20,
            ttnn_add_35,
            ttnn_where_23,
            ttnn_where_25,
            ttnn_add_45,
            var_19,
            var_20,
            ttnn_add_55,
            ttnn_where_30,
            ttnn_where_32,
            ttnn_add_65,
            ttnn_where_35,
            ttnn_where_37,
            ttnn_add_75,
            ttnn_where_40,
            ttnn_where_42,
            ttnn_add_85,
            ttnn_where_45,
            ttnn_where_47,
            ttnn_add_95,
            ttnn_where_50,
            ttnn_where_52,
            ttnn_add_105,
            var_37,
            var_38,
            ttnn_add_115,
            ttnn_where_56,
            ttnn_where_58,
            ttnn_add_125,
            ttnn_where_61,
            ttnn_where_63,
            ttnn_add_135,
            ttnn_where_66,
            ttnn_where_68,
            ttnn_add_145,
            ttnn_where_71,
            ttnn_where_73,
            ttnn_add_155,
            ttnn_where_76,
            ttnn_where_78,
            ttnn_add_165,
            var_55,
            var_56,
            ttnn_add_175,
            ttnn_where_82,
            ttnn_where_84,
            ttnn_add_185,
            ttnn_where_87,
            ttnn_where_89,
            ttnn_add_195,
            ttnn_where_92,
            ttnn_where_94,
            ttnn_add_205,
            ttnn_where_97,
            ttnn_where_99,
            ttnn_add_215,
            ttnn_where_102,
            ttnn_where_104,
            ttnn_add_225,
            var_73,
            var_74,
            ttnn_add_235,
            ttnn_where_108,
            ttnn_where_110,
            ttnn_add_245,
            ttnn_where_113,
            ttnn_where_115,
            ttnn_add_255,
            ttnn_where_118,
            ttnn_where_120,
            ttnn_add_265,
            ttnn_where_123,
            ttnn_where_125,
            ttnn_add_275,
            ttnn_where_128,
            ttnn_where_130,
            ttnn_add_285,
            var_91,
            var_92,
            ttnn_add_295,
            ttnn_where_134,
            ttnn_where_136,
            ttnn_add_305,
            ttnn_where_139,
            ttnn_where_141,
            ttnn_add_315,
            ttnn_where_144,
            ttnn_where_146,
            ttnn_add_325,
            ttnn_where_149,
            ttnn_where_151,
            ttnn_add_335,
            ttnn_where_154,
            ttnn_where_156,
            ttnn_add_345,
            var_109,
            var_110,
            ttnn_add_355,
            ttnn_where_160,
            ttnn_where_162,
            ttnn_add_365,
            ttnn_where_165,
            ttnn_where_167,
            ttnn_add_375,
            ttnn_where_170,
            ttnn_where_172,
            ttnn_add_385,
            ttnn_where_175,
            ttnn_where_177,
            ttnn_add_395,
            ttnn_where_180,
            ttnn_where_182,
            ttnn_add_405,
            var_127,
            var_128,
            ttnn_add_415,
            ttnn_where_186,
            ttnn_where_188,
            ttnn_add_425,
            ttnn_where_191,
            ttnn_where_193,
            ttnn_add_435,
            ttnn_where_196,
            ttnn_where_198,
            ttnn_add_445,
            ttnn_where_201,
            ttnn_where_203,
            ttnn_add_455,
            ttnn_where_206,
            ttnn_where_208,
            ttnn_add_465,
            var_145,
            var_146,
            ttnn_add_475,
            ttnn_where_212,
            ttnn_where_214,
            ttnn_add_485,
            ttnn_where_217,
            ttnn_where_219,
            ttnn_add_495,
            ttnn_where_222,
            ttnn_where_224,
            ttnn_add_505,
            ttnn_where_227,
            ttnn_where_229,
            ttnn_add_515,
            ttnn_where_232,
            ttnn_where_234,
            ttnn_add_525,
            var_163,
            var_164,
            ttnn_add_535,
            ttnn_where_238,
            ttnn_where_240,
            ttnn_add_545,
            ttnn_where_243,
            ttnn_where_245,
            ttnn_add_555,
            ttnn_where_248,
            ttnn_where_250,
            ttnn_add_565,
            ttnn_where_253,
            ttnn_where_255,
            ttnn_add_575,
            ttnn_where_258,
            ttnn_where_260,
            ttnn_add_585,
            var_181,
            var_182,
            ttnn_multiply_1084,
        ]

    def _call_prefill(self, input):
        """Prefill forward pass: prelude → 58-layer loop → L58/L59 specials
        → postlude. All weights and constants flow through self.X.
        """
        var_0 = input[0]
        var_2 = input[7]
        var_3 = input[9]
        var_7 = input[26]
        var_19 = input[98]
        var_20 = input[99]
        var_37 = input[199]
        var_38 = input[200]
        var_55 = input[300]
        var_56 = input[301]
        var_73 = input[401]
        var_74 = input[402]
        var_91 = input[502]
        var_92 = input[503]
        var_109 = input[603]
        var_110 = input[604]
        var_127 = input[704]
        var_128 = input[705]
        var_145 = input[805]
        var_146 = input[806]
        var_163 = input[906]
        var_164 = input[907]
        var_178 = input[977]
        var_179 = input[991]
        var_180 = input[993]
        var_181 = input[1007]
        var_182 = input[1008]
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
        layer_kv_outputs = [None] * 58
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
        for layer_idx, layer in enumerate(self.layers):
            result = layer(
                hidden,
                sliding_state=sliding_state,
                full_state=full_state,
                input=input,
                shared=self.shared,
            )
            hidden = result[0]
            layer_kv_outputs[layer_idx] = result[1:]

        # Restore original per-layer tensor names so the inline L58/L59
        # bodies and the postlude return list continue to resolve.
        ttnn_add_4, ttnn_where_1, ttnn_where_3 = layer_kv_outputs[0]
        ttnn_add_16, ttnn_where_8, ttnn_where_10 = layer_kv_outputs[1]
        ttnn_add_26, ttnn_where_13, ttnn_where_15 = layer_kv_outputs[2]
        ttnn_add_36, ttnn_where_18, ttnn_where_20 = layer_kv_outputs[3]
        ttnn_add_46, ttnn_where_23, ttnn_where_25 = layer_kv_outputs[4]
        (ttnn_add_56,) = layer_kv_outputs[5]
        ttnn_add_66, ttnn_where_30, ttnn_where_32 = layer_kv_outputs[6]
        ttnn_add_76, ttnn_where_35, ttnn_where_37 = layer_kv_outputs[7]
        ttnn_add_86, ttnn_where_40, ttnn_where_42 = layer_kv_outputs[8]
        ttnn_add_96, ttnn_where_45, ttnn_where_47 = layer_kv_outputs[9]
        ttnn_add_106, ttnn_where_50, ttnn_where_52 = layer_kv_outputs[10]
        (ttnn_add_116,) = layer_kv_outputs[11]
        ttnn_add_126, ttnn_where_56, ttnn_where_58 = layer_kv_outputs[12]
        ttnn_add_136, ttnn_where_61, ttnn_where_63 = layer_kv_outputs[13]
        ttnn_add_146, ttnn_where_66, ttnn_where_68 = layer_kv_outputs[14]
        ttnn_add_156, ttnn_where_71, ttnn_where_73 = layer_kv_outputs[15]
        ttnn_add_166, ttnn_where_76, ttnn_where_78 = layer_kv_outputs[16]
        (ttnn_add_176,) = layer_kv_outputs[17]
        ttnn_add_186, ttnn_where_82, ttnn_where_84 = layer_kv_outputs[18]
        ttnn_add_196, ttnn_where_87, ttnn_where_89 = layer_kv_outputs[19]
        ttnn_add_206, ttnn_where_92, ttnn_where_94 = layer_kv_outputs[20]
        ttnn_add_216, ttnn_where_97, ttnn_where_99 = layer_kv_outputs[21]
        ttnn_add_226, ttnn_where_102, ttnn_where_104 = layer_kv_outputs[22]
        (ttnn_add_236,) = layer_kv_outputs[23]
        ttnn_add_246, ttnn_where_108, ttnn_where_110 = layer_kv_outputs[24]
        ttnn_add_256, ttnn_where_113, ttnn_where_115 = layer_kv_outputs[25]
        ttnn_add_266, ttnn_where_118, ttnn_where_120 = layer_kv_outputs[26]
        ttnn_add_276, ttnn_where_123, ttnn_where_125 = layer_kv_outputs[27]
        ttnn_add_286, ttnn_where_128, ttnn_where_130 = layer_kv_outputs[28]
        (ttnn_add_296,) = layer_kv_outputs[29]
        ttnn_add_306, ttnn_where_134, ttnn_where_136 = layer_kv_outputs[30]
        ttnn_add_316, ttnn_where_139, ttnn_where_141 = layer_kv_outputs[31]
        ttnn_add_326, ttnn_where_144, ttnn_where_146 = layer_kv_outputs[32]
        ttnn_add_336, ttnn_where_149, ttnn_where_151 = layer_kv_outputs[33]
        ttnn_add_346, ttnn_where_154, ttnn_where_156 = layer_kv_outputs[34]
        (ttnn_add_356,) = layer_kv_outputs[35]
        ttnn_add_366, ttnn_where_160, ttnn_where_162 = layer_kv_outputs[36]
        ttnn_add_376, ttnn_where_165, ttnn_where_167 = layer_kv_outputs[37]
        ttnn_add_386, ttnn_where_170, ttnn_where_172 = layer_kv_outputs[38]
        ttnn_add_396, ttnn_where_175, ttnn_where_177 = layer_kv_outputs[39]
        ttnn_add_406, ttnn_where_180, ttnn_where_182 = layer_kv_outputs[40]
        (ttnn_add_416,) = layer_kv_outputs[41]
        ttnn_add_426, ttnn_where_186, ttnn_where_188 = layer_kv_outputs[42]
        ttnn_add_436, ttnn_where_191, ttnn_where_193 = layer_kv_outputs[43]
        ttnn_add_446, ttnn_where_196, ttnn_where_198 = layer_kv_outputs[44]
        ttnn_add_456, ttnn_where_201, ttnn_where_203 = layer_kv_outputs[45]
        ttnn_add_466, ttnn_where_206, ttnn_where_208 = layer_kv_outputs[46]
        (ttnn_add_476,) = layer_kv_outputs[47]
        ttnn_add_486, ttnn_where_212, ttnn_where_214 = layer_kv_outputs[48]
        ttnn_add_496, ttnn_where_217, ttnn_where_219 = layer_kv_outputs[49]
        ttnn_add_506, ttnn_where_222, ttnn_where_224 = layer_kv_outputs[50]
        ttnn_add_516, ttnn_where_227, ttnn_where_229 = layer_kv_outputs[51]
        ttnn_add_526, ttnn_where_232, ttnn_where_234 = layer_kv_outputs[52]
        (ttnn_add_536,) = layer_kv_outputs[53]
        ttnn_add_546, ttnn_where_238, ttnn_where_240 = layer_kv_outputs[54]
        ttnn_add_556, ttnn_where_243, ttnn_where_245 = layer_kv_outputs[55]
        ttnn_add_566, ttnn_where_248, ttnn_where_250 = layer_kv_outputs[56]
        ttnn_add_576, ttnn_where_253, ttnn_where_255 = layer_kv_outputs[57]

        ttnn_multiply_1044 = hidden

        (
            ttnn_multiply_1062,
            ttnn_add_586,
            ttnn_where_258,
            ttnn_where_260,
            ttnn_concat_219,
            ttnn_concat_220,
        ) = self._sliding_layer_58_prefill(
            hidden_state=ttnn_multiply_1044,
            causal_mask_logical_and=ttnn_logical_and_0,
            causal_mask_logical_not=ttnn_logical_not_0,
            sliding_cos_cache=ttnn_typecast_2,
            sliding_sin_cache=ttnn_typecast_3,
            pos_reshape_15=ttnn_reshape_15,
            pos_reshape_16=ttnn_reshape_16,
            pos_typecast_11=ttnn_typecast_11,
            var_178=var_178,
            var_179=var_179,
            var_180=var_180,
            input=input,
        )
        ttnn_add_602 = self._full_layer_59_prefill(
            hidden_state=ttnn_multiply_1062,
            full_cos_cache=ttnn_reshape_104,
            full_sin_cache=ttnn_reshape_105,
            full_pos_mask=ttnn_typecast_39,
            var_181=var_181,
            var_182=var_182,
            input=input,
        )
        ttnn_multiply_1084 = self.lm_head(ttnn_add_602)
        return [
            ttnn_add_0,
            ttnn_where_1,
            ttnn_where_3,
            ttnn_add_4,
            ttnn_where_8,
            ttnn_where_10,
            ttnn_add_16,
            ttnn_where_13,
            ttnn_where_15,
            ttnn_add_26,
            ttnn_where_18,
            ttnn_where_20,
            ttnn_add_36,
            ttnn_where_23,
            ttnn_where_25,
            ttnn_add_46,
            var_19,
            var_20,
            ttnn_add_56,
            ttnn_where_30,
            ttnn_where_32,
            ttnn_add_66,
            ttnn_where_35,
            ttnn_where_37,
            ttnn_add_76,
            ttnn_where_40,
            ttnn_where_42,
            ttnn_add_86,
            ttnn_where_45,
            ttnn_where_47,
            ttnn_add_96,
            ttnn_where_50,
            ttnn_where_52,
            ttnn_add_106,
            var_37,
            var_38,
            ttnn_add_116,
            ttnn_where_56,
            ttnn_where_58,
            ttnn_add_126,
            ttnn_where_61,
            ttnn_where_63,
            ttnn_add_136,
            ttnn_where_66,
            ttnn_where_68,
            ttnn_add_146,
            ttnn_where_71,
            ttnn_where_73,
            ttnn_add_156,
            ttnn_where_76,
            ttnn_where_78,
            ttnn_add_166,
            var_55,
            var_56,
            ttnn_add_176,
            ttnn_where_82,
            ttnn_where_84,
            ttnn_add_186,
            ttnn_where_87,
            ttnn_where_89,
            ttnn_add_196,
            ttnn_where_92,
            ttnn_where_94,
            ttnn_add_206,
            ttnn_where_97,
            ttnn_where_99,
            ttnn_add_216,
            ttnn_where_102,
            ttnn_where_104,
            ttnn_add_226,
            var_73,
            var_74,
            ttnn_add_236,
            ttnn_where_108,
            ttnn_where_110,
            ttnn_add_246,
            ttnn_where_113,
            ttnn_where_115,
            ttnn_add_256,
            ttnn_where_118,
            ttnn_where_120,
            ttnn_add_266,
            ttnn_where_123,
            ttnn_where_125,
            ttnn_add_276,
            ttnn_where_128,
            ttnn_where_130,
            ttnn_add_286,
            var_91,
            var_92,
            ttnn_add_296,
            ttnn_where_134,
            ttnn_where_136,
            ttnn_add_306,
            ttnn_where_139,
            ttnn_where_141,
            ttnn_add_316,
            ttnn_where_144,
            ttnn_where_146,
            ttnn_add_326,
            ttnn_where_149,
            ttnn_where_151,
            ttnn_add_336,
            ttnn_where_154,
            ttnn_where_156,
            ttnn_add_346,
            var_109,
            var_110,
            ttnn_add_356,
            ttnn_where_160,
            ttnn_where_162,
            ttnn_add_366,
            ttnn_where_165,
            ttnn_where_167,
            ttnn_add_376,
            ttnn_where_170,
            ttnn_where_172,
            ttnn_add_386,
            ttnn_where_175,
            ttnn_where_177,
            ttnn_add_396,
            ttnn_where_180,
            ttnn_where_182,
            ttnn_add_406,
            var_127,
            var_128,
            ttnn_add_416,
            ttnn_where_186,
            ttnn_where_188,
            ttnn_add_426,
            ttnn_where_191,
            ttnn_where_193,
            ttnn_add_436,
            ttnn_where_196,
            ttnn_where_198,
            ttnn_add_446,
            ttnn_where_201,
            ttnn_where_203,
            ttnn_add_456,
            ttnn_where_206,
            ttnn_where_208,
            ttnn_add_466,
            var_145,
            var_146,
            ttnn_add_476,
            ttnn_where_212,
            ttnn_where_214,
            ttnn_add_486,
            ttnn_where_217,
            ttnn_where_219,
            ttnn_add_496,
            ttnn_where_222,
            ttnn_where_224,
            ttnn_add_506,
            ttnn_where_227,
            ttnn_where_229,
            ttnn_add_516,
            ttnn_where_232,
            ttnn_where_234,
            ttnn_add_526,
            var_163,
            var_164,
            ttnn_add_536,
            ttnn_where_238,
            ttnn_where_240,
            ttnn_add_546,
            ttnn_where_243,
            ttnn_where_245,
            ttnn_add_556,
            ttnn_where_248,
            ttnn_where_250,
            ttnn_add_566,
            ttnn_where_253,
            ttnn_where_255,
            ttnn_add_576,
            ttnn_where_258,
            ttnn_where_260,
            ttnn_add_586,
            var_181,
            var_182,
            ttnn_concat_220,
            ttnn_concat_219,
            ttnn_multiply_1084,
        ]

    def _full_layer_59_decode(self, *, hidden_state, full_cos_cache, full_sin_cache, full_pos_mask, input):
        """Last full-attention decoder layer (index 59). Special because its
        layer_scalar multiply is absorbed into `_final_norm_lm_head_softcap`,
        so this helper returns the pre-layer-scalar residual `ttnn_add_601`
        rather than the post-layer-scalar tensor like `_full_decoder_layer`
        does for layers 5..53. Op sequence is verbatim from the original
        codegen; weights and input slots are hard-coded for layer 59 (the
        `self.layer_table[59]` entry is informational only).
        """
        var_180 = input[993]  # update_idxs (= L58's runtime_c slot)
        var_181 = input[1007]  # L59 runtime_a
        var_182 = input[1008]  # L59 runtime_b
        var_188 = self.shared["var_188"]
        var_191 = self.shared["var_191"]
        var_193 = self.shared["var_193"]

        ttnn_multiply_1062 = hidden_state
        ttnn_typecast_35 = full_cos_cache
        ttnn_typecast_36 = full_sin_cache
        ttnn_typecast_39 = full_pos_mask

        ttnn_multiply_1065 = gemma4.RMSNorm(self.l59.input_layernorm.weight, var_188)(ttnn_multiply_1062)
        ttnn_reshape_1093 = ttnn.reshape(
            ttnn_multiply_1065,
            [1, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1065, False)
        ttnn_all_gather_355 = ttnn.all_gather(
            input_tensor=ttnn_reshape_1093,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_1093, False)
        ttnn_matmul_424 = ttnn.matmul(
            ttnn_all_gather_355,
            self.l59.attention.k_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn_reshape_1094 = ttnn.reshape(
            ttnn_matmul_424,
            [1, 1, 1, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_424, False)
        ttnn_rms_norm_177 = ttnn.rms_norm(
            ttnn_reshape_1094,
            epsilon=9.9999999747524271e-07,
            weight=self.l59.attention.k_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn_multiply_1066 = ttnn.multiply(
            ttnn_rms_norm_177,
            ttnn_typecast_35,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_slice_386 = ttnn.slice(
            ttnn_rms_norm_177,
            [0, 0, 0, 256],
            [1, 1, 1, 512],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_neg_118 = ttnn.neg(
            ttnn_slice_386,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_386, False)
        ttnn_slice_387 = ttnn.slice(
            ttnn_rms_norm_177,
            [0, 0, 0, 0],
            [1, 1, 1, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rms_norm_177, False)
        ttnn_concat_120 = ttnn.concat(
            [ttnn_neg_118, ttnn_slice_387],
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_387, False)
        ttnn.deallocate(ttnn_neg_118, False)
        ttnn_multiply_1067 = ttnn.multiply(
            ttnn_concat_120,
            ttnn_typecast_36,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_concat_120, False)
        ttnn_add_594 = ttnn.add(
            ttnn_multiply_1066,
            ttnn_multiply_1067,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1067, False)
        ttnn.deallocate(ttnn_multiply_1066, False)
        ttnn_to_memory_config_18 = ttnn.to_memory_config(
            ttnn_add_594,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
                    [32, 512],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_add_594, False)
        ttnn.experimental.paged_update_cache(
            var_181,
            ttnn_to_memory_config_18,
            update_idxs_tensor=var_180,
            share_cache=False,
            page_table=None,
        )
        ttnn.deallocate(ttnn_to_memory_config_18, False)
        ttnn_rms_norm_178 = ttnn.rms_norm(
            ttnn_reshape_1094,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_1094, False)
        ttnn_to_memory_config_19 = ttnn.to_memory_config(
            ttnn_rms_norm_178,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
                    [32, 512],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_178, False)
        ttnn.experimental.paged_update_cache(
            var_182,
            ttnn_to_memory_config_19,
            update_idxs_tensor=var_180,
            share_cache=False,
            page_table=None,
        )
        ttnn.deallocate(ttnn_to_memory_config_19, False)
        ttnn.deallocate(var_180, False)
        ttnn_matmul_425 = ttnn.matmul(
            ttnn_all_gather_355,
            self.l59.attention.q_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_355, False)
        ttnn_reshape_1095 = ttnn.reshape(
            ttnn_matmul_425,
            [1, 8, 1, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_425, False)
        ttnn_rms_norm_179 = ttnn.rms_norm(
            ttnn_reshape_1095,
            epsilon=9.9999999747524271e-07,
            weight=self.l59.attention.q_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_1095, False)
        ttnn_multiply_1068 = ttnn.multiply(
            ttnn_rms_norm_179,
            ttnn_typecast_35,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_35, False)
        ttnn_slice_388 = ttnn.slice(
            ttnn_rms_norm_179,
            [0, 0, 0, 256],
            [1, 8, 1, 512],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_neg_119 = ttnn.neg(
            ttnn_slice_388,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_388, False)
        ttnn_slice_389 = ttnn.slice(
            ttnn_rms_norm_179,
            [0, 0, 0, 0],
            [1, 8, 1, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rms_norm_179, False)
        ttnn_concat_121 = ttnn.concat(
            [ttnn_neg_119, ttnn_slice_389],
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_389, False)
        ttnn.deallocate(ttnn_neg_119, False)
        ttnn_multiply_1069 = ttnn.multiply(
            ttnn_concat_121,
            ttnn_typecast_36,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_concat_121, False)
        ttnn.deallocate(ttnn_typecast_36, False)
        ttnn_add_595 = ttnn.add(
            ttnn_multiply_1068,
            ttnn_multiply_1069,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1069, False)
        ttnn.deallocate(ttnn_multiply_1068, False)
        ttnn_typecast_308 = ttnn.typecast(
            ttnn_add_595,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_595, False)
        ttnn_repeat_interleave_118 = ttnn.repeat_interleave(
            var_181,
            8,
            1,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_309 = ttnn.typecast(
            ttnn_repeat_interleave_118,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_repeat_interleave_118, False)
        ttnn_matmul_426 = ttnn.matmul(
            ttnn_typecast_308,
            ttnn_typecast_309,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_309, False)
        ttnn.deallocate(ttnn_typecast_308, False)
        ttnn_add_596 = ttnn.add(
            ttnn_matmul_426,
            ttnn_typecast_39,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_426, False)
        ttnn.deallocate(ttnn_typecast_39, False)
        ttnn_eq_59 = ttnn.eq(
            ttnn_add_596,
            var_193,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_logical_not_119 = ttnn.logical_not(
            ttnn_eq_59,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_eq_59, False)
        ttnn_sum_59 = ttnn.sum(
            ttnn_logical_not_119,
            [3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_logical_not_119, False)
        ttnn_logical_not_120 = ttnn.logical_not(
            ttnn_sum_59,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_59, False)
        ttnn_softmax_59 = ttnn.softmax(
            ttnn_add_596,
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            numeric_stable=True,
        )
        ttnn.deallocate(ttnn_add_596, False)
        ttnn_typecast_310 = ttnn.typecast(
            ttnn_logical_not_120,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_logical_not_120, False)
        ttnn_where_262 = ttnn.where(
            ttnn_typecast_310,
            var_191,
            ttnn_softmax_59,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_310, False)
        ttnn.deallocate(ttnn_softmax_59, False)
        ttnn_repeat_interleave_119 = ttnn.repeat_interleave(
            var_182,
            8,
            1,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_311 = ttnn.typecast(
            ttnn_repeat_interleave_119,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_repeat_interleave_119, False)
        ttnn_matmul_427 = ttnn.matmul(
            ttnn_where_262,
            ttnn_typecast_311,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_311, False)
        ttnn.deallocate(ttnn_where_262, False)
        ttnn_typecast_312 = ttnn.typecast(
            ttnn_matmul_427,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_427, False)
        ttnn_reshape_1096 = ttnn.reshape(
            ttnn_typecast_312,
            [1, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_312, False)
        ttnn_matmul_428 = ttnn.matmul(
            ttnn_reshape_1096,
            self.l59.attention.o_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_reshape_1096, False)
        ttnn_reshape_1097 = ttnn.reshape(
            ttnn_matmul_428,
            [1, 1, 1, 5376],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_428, False)
        ttnn_reduce_scatter_118 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_1097,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_1097, False)
        ttnn_reshape_1098 = ttnn.reshape(
            ttnn_reduce_scatter_118,
            [1, 1, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reduce_scatter_118, False)
        ttnn_multiply_1072 = gemma4.RMSNorm(self.l59.post_attention_layernorm.weight, var_188)(ttnn_reshape_1098)
        ttnn.deallocate(ttnn_reshape_1098, False)
        ttnn_add_598 = ttnn.add(
            ttnn_multiply_1062,
            ttnn_multiply_1072,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1072, False)
        ttnn.deallocate(ttnn_multiply_1062, False)
        ttnn_multiply_1075 = gemma4.RMSNorm(self.l59.pre_feedforward_layernorm.weight, var_188)(ttnn_add_598)
        ttnn_reshape_1101 = gemma4.FeedForward(
            self.l59.feed_forward.gate_proj_w,
            self.l59.feed_forward.up_proj_w,
            self.l59.feed_forward.down_proj_w,
        )(ttnn_multiply_1075)
        ttnn_multiply_1079 = gemma4.RMSNorm(self.l59.post_feedforward_layernorm.weight, var_188)(ttnn_reshape_1101)
        ttnn.deallocate(ttnn_reshape_1101, False)
        ttnn_add_601 = ttnn.add(
            ttnn_add_598,
            ttnn_multiply_1079,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1079, False)
        ttnn.deallocate(ttnn_add_598, False)

        return ttnn_add_601

    def _sliding_layer_58_prefill(
        self,
        *,
        hidden_state,
        causal_mask_logical_and,
        causal_mask_logical_not,
        sliding_cos_cache,
        sliding_sin_cache,
        pos_reshape_15,
        pos_reshape_16,
        pos_typecast_11,
        var_178,
        var_179,
        var_180,
        input,
    ):
        var_185 = self.shared["var_185"]
        var_186 = self.shared["var_186"]
        var_187 = self.shared["var_187"]
        var_188 = self.shared["var_188"]
        var_190 = self.shared["var_190"]
        var_192 = self.shared["var_192"]
        var_193 = self.shared["var_193"]
        """Last sliding-attention decoder layer (index 58). Special because
        its body emits two `ttnn_concat_*` ops that pre-stage the K cache for
        the trailing full-attention layer (L59). Op sequence is verbatim from
        the original codegen.
        """
        # Aliases match the inlined op names from `_main`.
        ttnn_multiply_1044 = hidden_state
        ttnn_logical_and_0 = causal_mask_logical_and
        ttnn_logical_not_0 = causal_mask_logical_not
        ttnn_typecast_2 = sliding_cos_cache
        ttnn_typecast_3 = sliding_sin_cache
        ttnn_reshape_15 = pos_reshape_15
        ttnn_reshape_16 = pos_reshape_16
        ttnn_typecast_11 = pos_typecast_11

        ttnn_multiply_1045 = ttnn.multiply(
            ttnn_multiply_1044,
            ttnn_multiply_1044,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_mean_464 = ttnn.mean(
            ttnn_multiply_1045,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_multiply_1045, False)
        ttnn_all_gather_348 = ttnn.all_gather(
            input_tensor=ttnn_mean_464,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_mean_464, False)
        ttnn_mean_465 = ttnn.mean(
            ttnn_all_gather_348,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_348, False)
        ttnn_add_584 = ttnn.add(
            ttnn_mean_465,
            var_188,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_mean_465, False)
        ttnn_rsqrt_232 = ttnn.rsqrt(
            ttnn_add_584,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_584, False)
        ttnn_multiply_1046 = ttnn.multiply(
            ttnn_multiply_1044,
            ttnn_rsqrt_232,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_232, False)
        ttnn_multiply_1047 = ttnn.multiply(
            ttnn_multiply_1046,
            self.l58.input_layernorm.weight,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1046, False)
        ttnn_reshape_978 = ttnn.reshape(
            ttnn_multiply_1047,
            [19, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1047, False)
        ttnn_all_gather_349 = ttnn.all_gather(
            input_tensor=ttnn_reshape_978,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_978, False)
        ttnn_matmul_417 = ttnn.matmul(
            ttnn_all_gather_349,
            self.l58.attention.fused_qkv_w,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_349, False)
        ttnn_slice_380 = ttnn.slice(
            ttnn_matmul_417,
            [0, 0],
            [19, 1024],
            [1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_slice_381 = ttnn.slice(
            ttnn_matmul_417,
            [0, 1024],
            [19, 3072],
            [1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_slice_382 = ttnn.slice(
            ttnn_matmul_417,
            [0, 3072],
            [19, 4096],
            [1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_417, False)
        ttnn_reshape_979 = ttnn.reshape(
            ttnn_slice_380,
            [1, 19, 4, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_380, False)
        ttnn_rms_norm_174 = ttnn.rms_norm(
            ttnn_reshape_979,
            epsilon=9.9999999747524271e-07,
            weight=self.l58.attention.k_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_979, False)
        ttnn_multiply_1048 = ttnn.multiply(
            ttnn_rms_norm_174,
            ttnn_typecast_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_2, False)
        ttnn_slice_383 = ttnn.slice(
            ttnn_rms_norm_174,
            [0, 0, 0, 128],
            [1, 19, 4, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_neg_116 = ttnn.neg(
            ttnn_slice_383,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_383, False)
        ttnn_slice_384 = ttnn.slice(
            ttnn_rms_norm_174,
            [0, 0, 0, 0],
            [1, 19, 4, 128],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rms_norm_174, False)
        ttnn_concat_217 = ttnn.concat(
            [ttnn_neg_116, ttnn_slice_384],
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_384, False)
        ttnn.deallocate(ttnn_neg_116, False)
        ttnn_multiply_1049 = ttnn.multiply(
            ttnn_concat_217,
            ttnn_typecast_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_concat_217, False)
        ttnn.deallocate(ttnn_typecast_3, False)
        ttnn_add_585 = ttnn.add(
            ttnn_multiply_1048,
            ttnn_multiply_1049,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1049, False)
        ttnn.deallocate(ttnn_multiply_1048, False)
        ttnn_permute_510 = ttnn.permute(
            ttnn_add_585,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn_reshape_980 = ttnn.reshape(
            ttnn_add_585,
            [19, 1024],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_585, False)
        ttnn_to_layout_261 = ttnn.to_layout(ttnn_reshape_980, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(ttnn_reshape_980, False)
        ttnn_embedding_199 = ttnn.embedding(
            var_186,
            ttnn_to_layout_261,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_261, False)
        ttnn_reshape_981 = ttnn.reshape(
            ttnn_embedding_199,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_embedding_199, False)
        ttnn_permute_511 = ttnn.permute(
            ttnn_reshape_981,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_981, False)
        ttnn_where_257 = ttnn.where(
            ttnn_logical_not_0,
            var_192,
            ttnn_permute_511,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_511, False)
        ttnn_permute_512 = ttnn.permute(
            var_178,
            [2, 0, 1, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn_reshape_982 = ttnn.reshape(
            ttnn_permute_512,
            [256, 1024],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_512, False)
        ttnn_to_layout_262 = ttnn.to_layout(ttnn_reshape_982, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(ttnn_reshape_982, False)
        ttnn_embedding_200 = ttnn.embedding(
            var_190,
            ttnn_to_layout_262,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_262, False)
        ttnn_reshape_983 = ttnn.reshape(
            ttnn_embedding_200,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_embedding_200, False)
        ttnn_permute_513 = ttnn.permute(
            ttnn_reshape_983,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_983, False)
        ttnn_where_258 = ttnn.where(
            ttnn_logical_and_0,
            ttnn_where_257,
            ttnn_permute_513,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_513, False)
        ttnn.deallocate(ttnn_where_257, False)
        ttnn_reshape_984 = ttnn.reshape(
            ttnn_slice_382,
            [1, 19, 4, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_382, False)
        ttnn_rms_norm_175 = ttnn.rms_norm(
            ttnn_reshape_984,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_984, False)
        ttnn_permute_514 = ttnn.permute(
            ttnn_rms_norm_175,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn_reshape_985 = ttnn.reshape(
            ttnn_rms_norm_175,
            [19, 1024],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rms_norm_175, False)
        ttnn_to_layout_263 = ttnn.to_layout(ttnn_reshape_985, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(ttnn_reshape_985, False)
        ttnn_embedding_201 = ttnn.embedding(
            var_186,
            ttnn_to_layout_263,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_263, False)
        ttnn_reshape_986 = ttnn.reshape(
            ttnn_embedding_201,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_embedding_201, False)
        ttnn_permute_515 = ttnn.permute(
            ttnn_reshape_986,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_986, False)
        ttnn_where_259 = ttnn.where(
            ttnn_logical_not_0,
            var_192,
            ttnn_permute_515,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_515, False)
        ttnn.deallocate(ttnn_logical_not_0, False)
        ttnn_permute_516 = ttnn.permute(
            var_179,
            [2, 0, 1, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn_reshape_987 = ttnn.reshape(
            ttnn_permute_516,
            [256, 1024],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_516, False)
        ttnn_to_layout_264 = ttnn.to_layout(ttnn_reshape_987, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(ttnn_reshape_987, False)
        ttnn_embedding_202 = ttnn.embedding(
            var_190,
            ttnn_to_layout_264,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_264, False)
        ttnn_reshape_988 = ttnn.reshape(
            ttnn_embedding_202,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_embedding_202, False)
        ttnn_permute_517 = ttnn.permute(
            ttnn_reshape_988,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_988, False)
        ttnn_where_260 = ttnn.where(
            ttnn_logical_and_0,
            ttnn_where_259,
            ttnn_permute_517,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_517, False)
        ttnn.deallocate(ttnn_where_259, False)
        ttnn.deallocate(ttnn_logical_and_0, False)
        ttnn_to_layout_265 = ttnn.to_layout(var_180, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(var_180, False)
        ttnn_add_586 = ttnn.add(
            ttnn_to_layout_265,
            var_185,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_265, False)
        ttnn_reshape_989 = ttnn.reshape(
            ttnn_slice_381,
            [1, 19, 8, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_381, False)
        ttnn_rms_norm_176 = ttnn.rms_norm(
            ttnn_reshape_989,
            epsilon=9.9999999747524271e-07,
            weight=self.l58.attention.q_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_989, False)
        ttnn_permute_518 = ttnn.permute(
            ttnn_rms_norm_176,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn_multiply_1050 = ttnn.multiply(
            ttnn_permute_518,
            ttnn_reshape_15,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_518, False)
        ttnn.deallocate(ttnn_reshape_15, False)
        ttnn_slice_385 = ttnn.slice(
            ttnn_rms_norm_176,
            [0, 0, 0, 128],
            [1, 19, 8, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_neg_117 = ttnn.neg(
            ttnn_slice_385,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_385, False)
        ttnn_slice_386 = ttnn.slice(
            ttnn_rms_norm_176,
            [0, 0, 0, 0],
            [1, 19, 8, 128],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rms_norm_176, False)
        ttnn_concat_218 = ttnn.concat(
            [ttnn_neg_117, ttnn_slice_386],
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_386, False)
        ttnn.deallocate(ttnn_neg_117, False)
        ttnn_permute_519 = ttnn.permute(
            ttnn_concat_218,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_concat_218, False)
        ttnn_multiply_1051 = ttnn.multiply(
            ttnn_permute_519,
            ttnn_reshape_16,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_519, False)
        ttnn.deallocate(ttnn_reshape_16, False)
        ttnn_add_587 = ttnn.add(
            ttnn_multiply_1050,
            ttnn_multiply_1051,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1051, False)
        ttnn.deallocate(ttnn_multiply_1050, False)
        ttnn_concat_219 = ttnn.concat(
            [var_178, ttnn_permute_510],
            2,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_510, False)
        ttnn.deallocate(var_178, False)
        ttnn_repeat_interleave_116 = ttnn.repeat_interleave(
            ttnn_concat_219,
            2,
            1,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_concat_220 = ttnn.concat(
            [var_179, ttnn_permute_514],
            2,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_514, False)
        ttnn.deallocate(var_179, False)
        ttnn_repeat_interleave_117 = ttnn.repeat_interleave(
            ttnn_concat_220,
            2,
            1,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_303 = ttnn.typecast(
            ttnn_add_587,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_587, False)
        ttnn_typecast_304 = ttnn.typecast(
            ttnn_repeat_interleave_116,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_repeat_interleave_116, False)
        ttnn_matmul_418 = ttnn.matmul(
            ttnn_typecast_303,
            ttnn_typecast_304,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_304, False)
        ttnn.deallocate(ttnn_typecast_303, False)
        ttnn_add_588 = ttnn.add(
            ttnn_matmul_418,
            ttnn_typecast_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_418, False)
        ttnn.deallocate(ttnn_typecast_11, False)
        ttnn_eq_58 = ttnn.eq(
            ttnn_add_588,
            var_193,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_logical_not_117 = ttnn.logical_not(
            ttnn_eq_58,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_eq_58, False)
        ttnn_sum_58 = ttnn.sum(
            ttnn_logical_not_117,
            [3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_logical_not_117, False)
        ttnn_logical_not_118 = ttnn.logical_not(
            ttnn_sum_58,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_58, False)
        ttnn_softmax_58 = ttnn.softmax(
            ttnn_add_588,
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            numeric_stable=True,
        )
        ttnn.deallocate(ttnn_add_588, False)
        ttnn_typecast_305 = ttnn.typecast(
            ttnn_logical_not_118,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_logical_not_118, False)
        ttnn_where_261 = ttnn.where(
            ttnn_typecast_305,
            var_187,
            ttnn_softmax_58,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_305, False)
        ttnn.deallocate(ttnn_softmax_58, False)
        ttnn_typecast_306 = ttnn.typecast(
            ttnn_repeat_interleave_117,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_repeat_interleave_117, False)
        ttnn_matmul_419 = ttnn.matmul(
            ttnn_where_261,
            ttnn_typecast_306,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_306, False)
        ttnn.deallocate(ttnn_where_261, False)
        ttnn_typecast_307 = ttnn.typecast(
            ttnn_matmul_419,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_419, False)
        ttnn_transformer_concatenate_heads_58 = ttnn.transformer.concatenate_heads(
            ttnn_typecast_307,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_307, False)
        ttnn_reshape_990 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_58,
            [19, 2048],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_58, False)
        ttnn_matmul_420 = ttnn.matmul(
            ttnn_reshape_990,
            self.l58.attention.o_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_reshape_990, False)
        ttnn_reshape_991 = ttnn.reshape(
            ttnn_matmul_420,
            [1, 1, 19, 5376],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_420, False)
        ttnn_reduce_scatter_116 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_991,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_991, False)
        ttnn_reshape_992 = ttnn.reshape(
            ttnn_reduce_scatter_116,
            [1, 19, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reduce_scatter_116, False)
        ttnn_multiply_1052 = ttnn.multiply(
            ttnn_reshape_992,
            ttnn_reshape_992,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_mean_466 = ttnn.mean(
            ttnn_multiply_1052,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_multiply_1052, False)
        ttnn_all_gather_350 = ttnn.all_gather(
            input_tensor=ttnn_mean_466,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_mean_466, False)
        ttnn_mean_467 = ttnn.mean(
            ttnn_all_gather_350,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_350, False)
        ttnn_add_589 = ttnn.add(
            ttnn_mean_467,
            var_188,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_mean_467, False)
        ttnn_rsqrt_233 = ttnn.rsqrt(
            ttnn_add_589,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_589, False)
        ttnn_multiply_1053 = ttnn.multiply(
            ttnn_reshape_992,
            ttnn_rsqrt_233,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_233, False)
        ttnn.deallocate(ttnn_reshape_992, False)
        ttnn_multiply_1054 = ttnn.multiply(
            ttnn_multiply_1053,
            self.l58.post_attention_layernorm.weight,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1053, False)
        ttnn_add_590 = ttnn.add(
            ttnn_multiply_1044,
            ttnn_multiply_1054,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1054, False)
        ttnn.deallocate(ttnn_multiply_1044, False)
        ttnn_multiply_1055 = ttnn.multiply(
            ttnn_add_590,
            ttnn_add_590,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_mean_468 = ttnn.mean(
            ttnn_multiply_1055,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_multiply_1055, False)
        ttnn_all_gather_351 = ttnn.all_gather(
            input_tensor=ttnn_mean_468,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_mean_468, False)
        ttnn_mean_469 = ttnn.mean(
            ttnn_all_gather_351,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_351, False)
        ttnn_add_591 = ttnn.add(
            ttnn_mean_469,
            var_188,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_mean_469, False)
        ttnn_rsqrt_234 = ttnn.rsqrt(
            ttnn_add_591,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_591, False)
        ttnn_multiply_1056 = ttnn.multiply(
            ttnn_add_590,
            ttnn_rsqrt_234,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_234, False)
        ttnn_multiply_1057 = ttnn.multiply(
            ttnn_multiply_1056,
            self.l58.pre_feedforward_layernorm.weight,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1056, False)
        ttnn_reshape_993 = ttnn.reshape(
            ttnn_multiply_1057,
            [19, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1057, False)
        ttnn_all_gather_352 = ttnn.all_gather(
            input_tensor=ttnn_reshape_993,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_993, False)
        ttnn_matmul_421 = ttnn.matmul(
            ttnn_all_gather_352,
            self.l58.feed_forward.gate_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation="gelu",
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn_matmul_422 = ttnn.matmul(
            ttnn_all_gather_352,
            self.l58.feed_forward.up_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_352, False)
        ttnn_multiply_1058 = ttnn.multiply(
            ttnn_matmul_421,
            ttnn_matmul_422,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_422, False)
        ttnn.deallocate(ttnn_matmul_421, False)
        ttnn_matmul_423 = ttnn.matmul(
            ttnn_multiply_1058,
            self.l58.feed_forward.down_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_multiply_1058, False)
        ttnn_reshape_994 = ttnn.reshape(
            ttnn_matmul_423,
            [1, 1, 19, 5376],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_423, False)
        ttnn_reduce_scatter_117 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_994,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_994, False)
        ttnn_reshape_995 = ttnn.reshape(
            ttnn_reduce_scatter_117,
            [1, 19, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reduce_scatter_117, False)
        ttnn_multiply_1059 = ttnn.multiply(
            ttnn_reshape_995,
            ttnn_reshape_995,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_mean_470 = ttnn.mean(
            ttnn_multiply_1059,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_multiply_1059, False)
        ttnn_all_gather_353 = ttnn.all_gather(
            input_tensor=ttnn_mean_470,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_mean_470, False)
        ttnn_mean_471 = ttnn.mean(
            ttnn_all_gather_353,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_353, False)
        ttnn_add_592 = ttnn.add(
            ttnn_mean_471,
            var_188,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_mean_471, False)
        ttnn_rsqrt_235 = ttnn.rsqrt(
            ttnn_add_592,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_592, False)
        ttnn_multiply_1060 = ttnn.multiply(
            ttnn_reshape_995,
            ttnn_rsqrt_235,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_235, False)
        ttnn.deallocate(ttnn_reshape_995, False)
        ttnn_multiply_1061 = ttnn.multiply(
            ttnn_multiply_1060,
            self.l58.post_feedforward_layernorm.weight,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1060, False)
        ttnn_add_593 = ttnn.add(
            ttnn_add_590,
            ttnn_multiply_1061,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1061, False)
        ttnn.deallocate(ttnn_add_590, False)
        ttnn_multiply_1062 = ttnn.multiply(
            ttnn_add_593,
            self.l58.layer_scalar,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_593, False)

        return (
            ttnn_multiply_1062,
            ttnn_add_586,
            ttnn_where_258,
            ttnn_where_260,
            ttnn_concat_219,
            ttnn_concat_220,
        )

    def _full_layer_59_prefill(
        self,
        *,
        hidden_state,
        full_cos_cache,
        full_sin_cache,
        full_pos_mask,
        var_181,
        var_182,
        input,
    ):
        var_187 = self.shared["var_187"]
        var_188 = self.shared["var_188"]
        var_193 = self.shared["var_193"]
        """Last full-attention decoder layer (index 59). Special because its
        layer_scalar mul is absorbed into `_final_norm_lm_head_softcap` (the
        postlude); the helper returns `ttnn_add_602`, the pre-layer-scalar
        residual that the postlude consumes via `last_layer_residual=`.
        Op sequence is verbatim from the original codegen.
        """
        ttnn_multiply_1062 = hidden_state
        ttnn_reshape_104 = full_cos_cache
        ttnn_reshape_105 = full_sin_cache
        ttnn_typecast_39 = full_pos_mask

        ttnn_multiply_1063 = ttnn.multiply(
            ttnn_multiply_1062,
            ttnn_multiply_1062,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_mean_472 = ttnn.mean(
            ttnn_multiply_1063,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_multiply_1063, False)
        ttnn_all_gather_354 = ttnn.all_gather(
            input_tensor=ttnn_mean_472,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_mean_472, False)
        ttnn_mean_473 = ttnn.mean(
            ttnn_all_gather_354,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_354, False)
        ttnn_add_594 = ttnn.add(
            ttnn_mean_473,
            var_188,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_mean_473, False)
        ttnn_rsqrt_236 = ttnn.rsqrt(
            ttnn_add_594,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_594, False)
        ttnn_multiply_1064 = ttnn.multiply(
            ttnn_multiply_1062,
            ttnn_rsqrt_236,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_236, False)
        ttnn_multiply_1065 = ttnn.multiply(
            ttnn_multiply_1064,
            self.l59.input_layernorm.weight,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1064, False)
        ttnn_reshape_996 = ttnn.reshape(
            ttnn_multiply_1065,
            [19, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1065, False)
        ttnn_all_gather_355 = ttnn.all_gather(
            input_tensor=ttnn_reshape_996,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_996, False)
        ttnn_matmul_424 = ttnn.matmul(
            ttnn_all_gather_355,
            self.l59.attention.k_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn_reshape_997 = ttnn.reshape(
            ttnn_matmul_424,
            [1, 1, 19, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_424, False)
        ttnn_rms_norm_177 = ttnn.rms_norm(
            ttnn_reshape_997,
            epsilon=9.9999999747524271e-07,
            weight=self.l59.attention.k_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn_multiply_1066 = ttnn.multiply(
            ttnn_rms_norm_177,
            ttnn_reshape_104,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_slice_387 = ttnn.slice(
            ttnn_rms_norm_177,
            [0, 0, 0, 256],
            [1, 1, 19, 512],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_neg_118 = ttnn.neg(
            ttnn_slice_387,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_387, False)
        ttnn_slice_388 = ttnn.slice(
            ttnn_rms_norm_177,
            [0, 0, 0, 0],
            [1, 1, 19, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rms_norm_177, False)
        ttnn_concat_221 = ttnn.concat(
            [ttnn_neg_118, ttnn_slice_388],
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_388, False)
        ttnn.deallocate(ttnn_neg_118, False)
        ttnn_multiply_1067 = ttnn.multiply(
            ttnn_concat_221,
            ttnn_reshape_105,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_concat_221, False)
        ttnn_add_595 = ttnn.add(
            ttnn_multiply_1066,
            ttnn_multiply_1067,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1067, False)
        ttnn.deallocate(ttnn_multiply_1066, False)
        ttnn.fill_cache(var_181, ttnn_add_595, 0)
        ttnn.deallocate(ttnn_add_595, False)
        ttnn_rms_norm_178 = ttnn.rms_norm(
            ttnn_reshape_997,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_997, False)
        ttnn.fill_cache(var_182, ttnn_rms_norm_178, 0)
        ttnn.deallocate(ttnn_rms_norm_178, False)
        ttnn_matmul_425 = ttnn.matmul(
            ttnn_all_gather_355,
            self.l59.attention.q_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_355, False)
        ttnn_reshape_998 = ttnn.reshape(
            ttnn_matmul_425,
            [1, 19, 8, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_425, False)
        ttnn_rms_norm_179 = ttnn.rms_norm(
            ttnn_reshape_998,
            epsilon=9.9999999747524271e-07,
            weight=self.l59.attention.q_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_998, False)
        ttnn_permute_520 = ttnn.permute(
            ttnn_rms_norm_179,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn_multiply_1068 = ttnn.multiply(
            ttnn_permute_520,
            ttnn_reshape_104,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_520, False)
        ttnn.deallocate(ttnn_reshape_104, False)
        ttnn_slice_389 = ttnn.slice(
            ttnn_rms_norm_179,
            [0, 0, 0, 256],
            [1, 19, 8, 512],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_neg_119 = ttnn.neg(
            ttnn_slice_389,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_389, False)
        ttnn_slice_390 = ttnn.slice(
            ttnn_rms_norm_179,
            [0, 0, 0, 0],
            [1, 19, 8, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rms_norm_179, False)
        ttnn_concat_222 = ttnn.concat(
            [ttnn_neg_119, ttnn_slice_390],
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_390, False)
        ttnn.deallocate(ttnn_neg_119, False)
        ttnn_permute_521 = ttnn.permute(
            ttnn_concat_222,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_concat_222, False)
        ttnn_multiply_1069 = ttnn.multiply(
            ttnn_permute_521,
            ttnn_reshape_105,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_521, False)
        ttnn.deallocate(ttnn_reshape_105, False)
        ttnn_add_596 = ttnn.add(
            ttnn_multiply_1068,
            ttnn_multiply_1069,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1069, False)
        ttnn.deallocate(ttnn_multiply_1068, False)
        ttnn_typecast_308 = ttnn.typecast(
            ttnn_add_596,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_596, False)
        ttnn_repeat_interleave_118 = ttnn.repeat_interleave(
            var_181,
            8,
            1,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_309 = ttnn.typecast(
            ttnn_repeat_interleave_118,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_repeat_interleave_118, False)
        ttnn_matmul_426 = ttnn.matmul(
            ttnn_typecast_308,
            ttnn_typecast_309,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_309, False)
        ttnn.deallocate(ttnn_typecast_308, False)
        ttnn_add_597 = ttnn.add(
            ttnn_matmul_426,
            ttnn_typecast_39,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_426, False)
        ttnn.deallocate(ttnn_typecast_39, False)
        ttnn_eq_59 = ttnn.eq(
            ttnn_add_597,
            var_193,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_logical_not_119 = ttnn.logical_not(
            ttnn_eq_59,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_eq_59, False)
        ttnn_sum_59 = ttnn.sum(
            ttnn_logical_not_119,
            [3],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_logical_not_119, False)
        ttnn_logical_not_120 = ttnn.logical_not(
            ttnn_sum_59,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sum_59, False)
        ttnn_softmax_59 = ttnn.softmax(
            ttnn_add_597,
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            numeric_stable=True,
        )
        ttnn.deallocate(ttnn_add_597, False)
        ttnn_typecast_310 = ttnn.typecast(
            ttnn_logical_not_120,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_logical_not_120, False)
        ttnn_where_262 = ttnn.where(
            ttnn_typecast_310,
            var_187,
            ttnn_softmax_59,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_310, False)
        ttnn.deallocate(ttnn_softmax_59, False)
        ttnn_repeat_interleave_119 = ttnn.repeat_interleave(
            var_182,
            8,
            1,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_311 = ttnn.typecast(
            ttnn_repeat_interleave_119,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_repeat_interleave_119, False)
        ttnn_matmul_427 = ttnn.matmul(
            ttnn_where_262,
            ttnn_typecast_311,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_311, False)
        ttnn.deallocate(ttnn_where_262, False)
        ttnn_typecast_312 = ttnn.typecast(
            ttnn_matmul_427,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_427, False)
        ttnn_transformer_concatenate_heads_59 = ttnn.transformer.concatenate_heads(
            ttnn_typecast_312,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_312, False)
        ttnn_reshape_999 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_59,
            [19, 4096],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_59, False)
        ttnn_matmul_428 = ttnn.matmul(
            ttnn_reshape_999,
            self.l59.attention.o_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_reshape_999, False)
        ttnn_reshape_1000 = ttnn.reshape(
            ttnn_matmul_428,
            [1, 1, 19, 5376],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_428, False)
        ttnn_reduce_scatter_118 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_1000,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_1000, False)
        ttnn_reshape_1001 = ttnn.reshape(
            ttnn_reduce_scatter_118,
            [1, 19, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reduce_scatter_118, False)
        ttnn_multiply_1070 = ttnn.multiply(
            ttnn_reshape_1001,
            ttnn_reshape_1001,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_mean_474 = ttnn.mean(
            ttnn_multiply_1070,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_multiply_1070, False)
        ttnn_all_gather_356 = ttnn.all_gather(
            input_tensor=ttnn_mean_474,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_mean_474, False)
        ttnn_mean_475 = ttnn.mean(
            ttnn_all_gather_356,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_356, False)
        ttnn_add_598 = ttnn.add(
            ttnn_mean_475,
            var_188,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_mean_475, False)
        ttnn_rsqrt_237 = ttnn.rsqrt(
            ttnn_add_598,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_598, False)
        ttnn_multiply_1071 = ttnn.multiply(
            ttnn_reshape_1001,
            ttnn_rsqrt_237,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_237, False)
        ttnn.deallocate(ttnn_reshape_1001, False)
        ttnn_multiply_1072 = ttnn.multiply(
            ttnn_multiply_1071,
            self.l59.post_attention_layernorm.weight,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1071, False)
        ttnn_add_599 = ttnn.add(
            ttnn_multiply_1062,
            ttnn_multiply_1072,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1072, False)
        ttnn.deallocate(ttnn_multiply_1062, False)
        ttnn_multiply_1073 = ttnn.multiply(
            ttnn_add_599,
            ttnn_add_599,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_mean_476 = ttnn.mean(
            ttnn_multiply_1073,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_multiply_1073, False)
        ttnn_all_gather_357 = ttnn.all_gather(
            input_tensor=ttnn_mean_476,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_mean_476, False)
        ttnn_mean_477 = ttnn.mean(
            ttnn_all_gather_357,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_357, False)
        ttnn_add_600 = ttnn.add(
            ttnn_mean_477,
            var_188,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_mean_477, False)
        ttnn_rsqrt_238 = ttnn.rsqrt(
            ttnn_add_600,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_600, False)
        ttnn_multiply_1074 = ttnn.multiply(
            ttnn_add_599,
            ttnn_rsqrt_238,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_238, False)
        ttnn_multiply_1075 = ttnn.multiply(
            ttnn_multiply_1074,
            self.l59.pre_feedforward_layernorm.weight,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1074, False)
        ttnn_reshape_1002 = ttnn.reshape(
            ttnn_multiply_1075,
            [19, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1075, False)
        ttnn_all_gather_358 = ttnn.all_gather(
            input_tensor=ttnn_reshape_1002,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_1002, False)
        ttnn_matmul_429 = ttnn.matmul(
            ttnn_all_gather_358,
            self.l59.feed_forward.gate_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation="gelu",
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn_matmul_430 = ttnn.matmul(
            ttnn_all_gather_358,
            self.l59.feed_forward.up_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_358, False)
        ttnn_multiply_1076 = ttnn.multiply(
            ttnn_matmul_429,
            ttnn_matmul_430,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_430, False)
        ttnn.deallocate(ttnn_matmul_429, False)
        ttnn_matmul_431 = ttnn.matmul(
            ttnn_multiply_1076,
            self.l59.feed_forward.down_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_multiply_1076, False)
        ttnn_reshape_1003 = ttnn.reshape(
            ttnn_matmul_431,
            [1, 1, 19, 5376],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_431, False)
        ttnn_reduce_scatter_119 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_1003,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_1003, False)
        ttnn_reshape_1004 = ttnn.reshape(
            ttnn_reduce_scatter_119,
            [1, 19, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reduce_scatter_119, False)
        ttnn_multiply_1077 = ttnn.multiply(
            ttnn_reshape_1004,
            ttnn_reshape_1004,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_mean_478 = ttnn.mean(
            ttnn_multiply_1077,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_multiply_1077, False)
        ttnn_all_gather_359 = ttnn.all_gather(
            input_tensor=ttnn_mean_478,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_mean_478, False)
        ttnn_mean_479 = ttnn.mean(
            ttnn_all_gather_359,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_359, False)
        ttnn_add_601 = ttnn.add(
            ttnn_mean_479,
            var_188,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_mean_479, False)
        ttnn_rsqrt_239 = ttnn.rsqrt(
            ttnn_add_601,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_601, False)
        ttnn_multiply_1078 = ttnn.multiply(
            ttnn_reshape_1004,
            ttnn_rsqrt_239,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_rsqrt_239, False)
        ttnn.deallocate(ttnn_reshape_1004, False)
        ttnn_multiply_1079 = ttnn.multiply(
            ttnn_multiply_1078,
            self.l59.post_feedforward_layernorm.weight,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1078, False)
        ttnn_add_602 = ttnn.add(
            ttnn_add_599,
            ttnn_multiply_1079,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_multiply_1079, False)
        ttnn.deallocate(ttnn_add_599, False)

        return ttnn_add_602
