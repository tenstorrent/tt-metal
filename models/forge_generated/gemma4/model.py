"""Top-level Gemma4 model orchestration (Gemma4ForCausalLM).

Construction is HF-state-dict-driven: `from_state_dict(hf, mesh_device)`
builds the full module tree (60 decoder layers, scaled embedding, LM
head, sliding/full preludes for both modes, L59 terminal special).
Every weight, scalar constant, and RoPE inv-freq tensor lives as a
named instance attribute. The runtime call path is
`model(input_list, mode=..., current_pos=...)` returning logits;
`mode` selects the decode vs prefill orchestration body and dispatches
the per-layer attention recipe accordingly.
"""

import gemma4
import torch
from gemma4 import utils
from gemma4 import weights as gw
from gemma4.caches import Gemma4Caches
from gemma4.layer_table import LAYER_TABLE_DECODE

import ttnn

_DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _build_pos_tensor(current_pos, mesh_device):
    """Build the int32 [1] ROW_MAJOR tensor used everywhere a position
    scalar is read from the input list — slot 0 (global current_pos)
    plus the 59 per-layer pos_ids slots. Replicated across the (1,4)
    mesh.
    """
    return ttnn.as_tensor(
        torch.tensor([int(current_pos)], dtype=torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=_DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


class Gemma4ForCausalLM:
    """Full Gemma4 forward: embedding → preludes → 59-layer loop → L59 → LMHead.

    A single instance serves both prefill and decode modes; the
    orchestration body and per-layer attention dispatch on the `mode`
    kwarg passed to `__call__`. Runtime tensors (token IDs, position
    scalars, KV caches) flow through the call path; mode-specific
    constants (preludes, shared scalars) live as paired
    `*_decode`/`*_prefill` instance attributes.
    """

    def __init__(
        self,
        *,
        scaled_embedding,
        layers,
        lm_head,
        l59,
        caches,
        mesh_device,
        pos_slots,
        causal_mask_one_hot,
        sliding_prelude_decode,
        sliding_prelude_prefill,
        full_prelude_decode,
        full_prelude_prefill,
        shared_decode,
        shared_prefill,
    ):
        self.scaled_embedding = scaled_embedding
        # 59 layers (L0..L58) — single set serves both modes.
        self.layers = layers
        self.lm_head = lm_head
        # L59 terminal — shared across modes.
        self.l59 = l59
        # Per-layer KV cache buffers. Each layer also holds direct refs
        # to its slice (set in from_state_dict); reset_kv_caches keeps
        # both views in sync.
        self.caches = caches
        self.mesh_device = mesh_device
        # Slots into which __call__ injects the per-call position scalar
        # (slot 0 + every L0..L58 layer's pos_ids slot).
        self._pos_slots = pos_slots
        # Mode-equivalent constant: bf16 (1,1,256,1) one_hot at the last
        # position, used to build the causal mask. Same recipe in both
        # modes (codegen used different consteval keys but same value).
        self.causal_mask_one_hot = causal_mask_one_hot
        # Mode-specific preludes — different op graphs and output tuples.
        self.sliding_prelude_decode = sliding_prelude_decode
        self.sliding_prelude_prefill = sliding_prelude_prefill
        self.full_prelude_decode = full_prelude_decode
        self.full_prelude_prefill = full_prelude_prefill
        # Mode-specific shared scalars consumed by the regular layer body.
        # Keys differ slightly: prefill has var_187, decode has var_191.
        # The values that are common (var_185, var_186, ...) also differ
        # numerically (var_185 = seq_len for prefill, 1 for decode).
        self.shared_decode = shared_decode
        self.shared_prefill = shared_prefill

    def reset_kv_caches(self):
        """Re-zero every per-layer K/V buffer. Phase 2 also calls this
        at the start of each forward pass for PCC reproducibility; the
        Generator (Phase 5) will own this between independent sessions.
        """
        self.caches.reset()
        for layer in self.layers:
            layer.k_cache = self.caches.k_caches[layer.layer_idx]
            layer.v_cache = self.caches.v_caches[layer.layer_idx]
        self.l59.k_cache = self.caches.k_caches[self.l59.layer_idx]
        self.l59.v_cache = self.caches.v_caches[self.l59.layer_idx]

    def __call__(self, input, *, mode, current_pos=0):
        # Phase 5: reset is the caller's job (Generator does it once at
        # the start of a session). Independent single-shot callers
        # (PCC tests) call reset_kv_caches() explicitly before model().
        # Phase 3: build the per-call position scalar internally and inject
        # it at every input slot that historically held an int32 [1] zero.
        # The runtime_inputs synthesizer no longer allocates these slots,
        # so eff_input may need to grow to fit the highest pos slot.
        pos_tensor = _build_pos_tensor(current_pos, self.mesh_device)
        eff_input = list(input)
        max_slot = max(self._pos_slots)
        if len(eff_input) <= max_slot:
            eff_input.extend([None] * (max_slot + 1 - len(eff_input)))
        for slot in self._pos_slots:
            eff_input[slot] = pos_tensor
        if mode == "decode":
            return self._call_decode(eff_input)
        elif mode == "prefill":
            return self._call_prefill(eff_input)
        else:
            raise ValueError(f"unknown mode {mode!r}; expected 'prefill' or 'decode'")

    @classmethod
    def from_state_dict(cls, hf, mesh_device, *, seq_len=19, caches=None):
        """Build a single-instance Gemma4ForCausalLM that serves both modes.

        `seq_len` is the prefill sequence length (default 19, matching
        the codegen-baked artifact). Decode bodies are seq_len=1 (not
        parameterized). To change the prefill seq_len, the caller must
        also build the runtime input list at the same seq_len (see
        `synthesize_prefill_inputs`).

        `caches` is an optional pre-built `Gemma4Caches` to share across
        a generator session. If None, allocate fresh zero caches.
        """
        # Layer slot layout is identical in both LAYER_TABLE_PREFILL and
        # LAYER_TABLE_DECODE (verified in Phase 0 recon); use either.
        layer_table = LAYER_TABLE_DECODE
        if caches is None:
            caches = Gemma4Caches(
                mesh_device,
                [layer_table[i]["type"] for i in range(60)],
            )

        # Two transient consteval dicts — one per mode. Mode-specific
        # tensors (RoPE caches, position helpers, scalar fills baked
        # with seq_len) differ between them; mode-equivalent constants
        # (causal mask, softcap) are the same value either way.
        transient_decode: dict = {}
        gw.apply_hf_scalar_overrides(transient_decode, hf, mesh_device, is_decode=True, seq_len=seq_len)
        gemma4.RoPESetup.from_hf(hf, mesh_device, is_decode=True).populate_cached_main(transient_decode)
        transient_prefill: dict = {}
        gw.apply_hf_scalar_overrides(transient_prefill, hf, mesh_device, is_decode=False, seq_len=seq_len)
        gemma4.RoPESetup.from_hf(hf, mesh_device, is_decode=False).populate_cached_main(transient_prefill)

        rms_eps_tensor = transient_decode["main_const_eval_240"][0]

        # Mode-equivalent constants (same recipe both modes; pick either source).
        causal_mask_one_hot = transient_decode["main_const_eval_535"][0]
        softcap = transient_decode["main_const_eval_171"][0]

        # Top-level submodules.
        scaled_embedding = gemma4.ScaledEmbedding.from_state_dict(
            hf.state_dict,
            hf.lifted,
            mesh_device,
        )

        # 59 decoder layers (L0..L58). Each serves both modes — Attention
        # already takes is_decode per call; RMSNorm/FeedForward are
        # mode-agnostic. update_idxs_slot is set unconditionally to the
        # previous sliding layer's pos slot — the prefill body ignores it,
        # the decode body's paged_update_cache reads it.
        layers = []
        for i in range(59):
            t = layer_table[i]
            slots = tuple(t["runtime_inputs"])
            if t["type"] == "sliding":
                layer = gemma4.SlidingDecoderLayer.from_state_dict(
                    hf.state_dict,
                    i,
                    mesh_device,
                    runtime_slots=slots,
                    k_cache=caches.k_caches[i],
                    v_cache=caches.v_caches[i],
                    rms_eps_tensor=rms_eps_tensor,
                    seq_len=seq_len,
                )
            else:
                update_idxs_slot = layer_table[i - 1]["runtime_inputs"][2]
                layer = gemma4.FullDecoderLayer.from_state_dict(
                    hf.state_dict,
                    i,
                    mesh_device,
                    runtime_slots=slots,
                    update_idxs_slot=update_idxs_slot,
                    k_cache=caches.k_caches[i],
                    v_cache=caches.v_caches[i],
                    rms_eps_tensor=rms_eps_tensor,
                    seq_len=seq_len,
                )
            layers.append(layer)

        # Mode-specific shared scalars consumed by the regular layer body.
        # var_185/186 differ numerically per mode (main_const_eval_0 is
        # rebuilt with mode-specific fills), so each mode gets its own dict.
        shared_decode = {
            "var_185": transient_decode["main_const_eval_0"][1],
            "var_186": transient_decode["main_const_eval_0"][2],
            "var_188": rms_eps_tensor,
            "var_190": transient_decode["main_const_eval_334"][0],
            "var_191": transient_decode["main_const_eval_337"][0],
            "var_192": transient_decode["main_const_eval_486"][0],
            "var_193": transient_decode["main_const_eval_489"][0],
        }
        shared_prefill = {
            "var_185": transient_prefill["main_const_eval_0"][1],
            "var_186": transient_prefill["main_const_eval_0"][2],
            "var_187": transient_prefill["main_const_eval_186"][0],
            "var_188": rms_eps_tensor,
            "var_190": transient_prefill["main_const_eval_266"][0],
            "var_192": transient_prefill["main_const_eval_335"][0],
            "var_193": transient_prefill["main_const_eval_338"][0],
        }

        # Mode-specific preludes — different op graphs and output tuples.
        sliding_prelude_decode = gemma4.SlidingPreludeDecode.from_consteval(transient_decode)
        full_prelude_decode = gemma4.FullPreludeDecode.from_consteval(transient_decode)
        sliding_prelude_prefill = gemma4.SlidingPreludePrefill.from_consteval(transient_prefill, seq_len=seq_len)
        full_prelude_prefill = gemma4.FullPreludePrefill.from_consteval(transient_prefill, seq_len=seq_len)

        # L59 terminal — runtime_slots is 2-tuple (k, v); is_terminal=True
        # skips pos_ids read, position increment, and the layer_scalar
        # multiply (LMHead.last_layer_scalar absorbs the latter).
        # update_idxs piggy-backs on L58's pos slot for the decode call.
        l59_slots = tuple(layer_table[59]["runtime_inputs"])
        l59 = gemma4.FullDecoderLayer.from_state_dict(
            hf.state_dict,
            59,
            mesh_device,
            runtime_slots=l59_slots,
            update_idxs_slot=layer_table[58]["runtime_inputs"][2],
            k_cache=caches.k_caches[59],
            v_cache=caches.v_caches[59],
            rms_eps_tensor=rms_eps_tensor,
            is_terminal=True,
            seq_len=seq_len,
        )

        lm_head = gemma4.LMHead.from_state_dict(
            hf.state_dict,
            mesh_device,
            rms_eps=rms_eps_tensor,
            last_layer_scalar=l59.layer_scalar,
            softcap=softcap,
        )

        # Position-scalar injection slots: slot 0 + every L0..L58 layer's
        # pos_ids slot. L59 has no pos slot of its own.
        pos_slots = [0] + [layer_table[i]["runtime_inputs"][2] for i in range(59)]

        return cls(
            scaled_embedding=scaled_embedding,
            layers=layers,
            lm_head=lm_head,
            l59=l59,
            caches=caches,
            mesh_device=mesh_device,
            pos_slots=pos_slots,
            causal_mask_one_hot=causal_mask_one_hot,
            sliding_prelude_decode=sliding_prelude_decode,
            sliding_prelude_prefill=sliding_prelude_prefill,
            full_prelude_decode=full_prelude_decode,
            full_prelude_prefill=full_prelude_prefill,
            shared_decode=shared_decode,
            shared_prefill=shared_prefill,
        )

    def _call_decode(self, input):
        """Decode forward pass: prelude → 59-layer loop → L59 → LMHead."""
        shared = self.shared_decode
        var_0 = input[0]
        var_2 = input[7]
        var_3 = input[9]
        var_7 = input[26]
        var_185 = shared["var_185"]
        utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 4))  # noqa: F841
        ttnn_to_layout_0 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
        # Phase 3: var_0 is the shared pos_tensor reused by every layer's pos
        # slot — keep it alive (was: ttnn.deallocate(var_0, False)).
        ttnn_add_0 = ttnn.add(
            ttnn_to_layout_0,
            var_185,
            dtype=ttnn.DataType.INT32,
            memory_config=_DRAM,
        )
        ttnn_to_layout_1 = ttnn.to_layout(var_3, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(var_3, False)
        ttnn_reshape_0 = ttnn.reshape(
            ttnn_to_layout_1,
            [1, 1, 1, 1],
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_to_layout_1, False)
        ttnn_logical_and_0 = ttnn.logical_and(
            ttnn_reshape_0,
            self.causal_mask_one_hot,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn_logical_not_0 = ttnn.logical_not(
            ttnn_logical_and_0,
            memory_config=_DRAM,
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
        ) = self.sliding_prelude_decode(
            input=input,
            ttnn_to_layout_0=ttnn_to_layout_0,
            ttnn_add_0=ttnn_add_0,
        )
        (
            ttnn_typecast_35,
            ttnn_typecast_36,
            ttnn_typecast_39,
        ) = self.full_prelude_decode(
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
        for layer in self.layers:
            hidden = layer(
                hidden,
                is_decode=True,
                sliding_state=sliding_state,
                full_state=full_state,
                input=input,
                shared=shared,
            )
        ttnn_add_601 = self.l59(
            hidden,
            is_decode=True,
            sliding_state=sliding_state,
            full_state=full_state,
            input=input,
            shared=shared,
        )
        return self.lm_head(ttnn_add_601)

    def _call_prefill(self, input):
        """Prefill forward pass: prelude → 59-layer loop → L59 → LMHead."""
        shared = self.shared_prefill
        var_0 = input[0]
        var_2 = input[7]
        var_3 = input[9]
        var_7 = input[26]
        var_185 = shared["var_185"]
        utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 4))  # noqa: F841
        ttnn_to_layout_0 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
        # Phase 3: var_0 is the shared pos_tensor reused by every layer's pos
        # slot — keep it alive (was: ttnn.deallocate(var_0, False)).
        ttnn_add_0 = ttnn.add(  # noqa: F841 — kept for op-graph completeness; prefill prelude doesn't read it.
            ttnn_to_layout_0,
            var_185,
            dtype=ttnn.DataType.INT32,
            memory_config=_DRAM,
        )
        ttnn_to_layout_1 = ttnn.to_layout(var_3, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(var_3, False)
        ttnn_reshape_0 = ttnn.reshape(
            ttnn_to_layout_1,
            [1, 1, 1, 1],
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_to_layout_1, False)
        ttnn_logical_and_0 = ttnn.logical_and(
            ttnn_reshape_0,
            self.causal_mask_one_hot,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn_logical_not_0 = ttnn.logical_not(
            ttnn_logical_and_0,
            memory_config=_DRAM,
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
        ) = self.sliding_prelude_prefill(
            input=input,
            ttnn_to_layout_0=ttnn_to_layout_0,
        )
        # `var_7` (= input[26]) is consumed inside `_sliding_prelude` but the
        # prelude does NOT deallocate it; we mirror the original codegen's
        # post-layer-0 freeing here.
        ttnn.deallocate(var_7, False)
        (
            ttnn_reshape_104,
            ttnn_reshape_105,
            ttnn_typecast_39,
        ) = self.full_prelude_prefill(
            input=input,
            ttnn_reshape_0=ttnn_reshape_0,
            ttnn_reshape_3=ttnn_reshape_3,
            ttnn_reshape_18=ttnn_reshape_18,
            ttnn_to_layout_11=ttnn_to_layout_11,
        )
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
        for layer in self.layers:
            hidden = layer(
                hidden,
                is_decode=False,
                sliding_state=sliding_state,
                full_state=full_state,
                input=input,
                shared=shared,
            )
        ttnn_add_602 = self.l59(
            hidden,
            is_decode=False,
            sliding_state=sliding_state,
            full_state=full_state,
            input=input,
            shared=shared,
        )
        return self.lm_head(ttnn_add_602)
