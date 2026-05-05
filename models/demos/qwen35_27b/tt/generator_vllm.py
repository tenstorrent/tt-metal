# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from tqdm import tqdm

import ttnn
from models.demos.qwen35_27b.tt.generator import Generator
from models.demos.qwen35_27b.tt.model import create_qwen35_model
from models.tt_transformers.tt.model_config import TensorGroup


class Qwen35ForCausalLM(Generator):
    """vLLM adapter for Qwen3.5-27B on Tenstorrent hardware.

    The framework Generator's prefill_forward_text loops users internally without
    a hook between iterations, but Qwen3.5's GDN layers require:
      (a) prefill state freshly zeroed before each request,
      (b) the resulting recurrent state written only into that user's batch slot,
          leaving other in-flight users' GDN states untouched.

    We satisfy both by calling super().prefill_forward_text() once per user with
    empty_slots=[0] (so the framework's chunked path uses CHUNK_USER_ID=0
    everywhere) and a sliced page_table for the real slot. The page_table
    contents direct paged_fill_cache to the correct physical blocks regardless
    of which logical slot the framework thinks it's writing.

    Tracing in vLLM prefill is intentionally disabled: framework warmup
    re-enters prefill_forward_text recursively and would bypass our reset/commit
    hooks. Decode tracing is unaffected.
    """

    model_capabilities = {
        "supports_prefix_caching": False,  # GDN recurrence state not compatible with prefix caching
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Suppress framework warmup_model_prefill (tt_transformers/tt/generator.py:113-115).
        # Warmup re-enters prefill_forward_text recursively and would bypass the
        # per-user reset/commit hooks below. Trade-off: vLLM gets non-traced
        # prefill — acceptable for MVP.
        self.already_warmed_up_prefill = True

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        assert tt_data_parallel == 1, (
            "Qwen3.5-27B vLLM adapter currently supports only tt_data_parallel=1; " f"got {tt_data_parallel}."
        )
        # Resolution order for the weights path. Prefer env-var paths first
        # because vLLM's hf_config._name_or_path is the HF repo name (e.g.
        # "Qwen/Qwen3.5-27B-FP8"), not a filesystem path; using it blindly
        # would make load_qwen35_state_dict() try to open the repo name as a
        # relative directory and fail.
        #   1. MODEL_WEIGHTS_DIR (Docker convention, tt-inference-server sets this)
        #   2. HF_MODEL env var
        #   3. hf_config._name_or_path — only if it resolves to a real directory
        #   4. create_qwen35_model's hardcoded ~/models/Qwen3.5-27B-FP8 default
        import os

        model_path = os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL")
        if not model_path and hf_config is not None:
            candidate = getattr(hf_config, "_name_or_path", None)
            if candidate and os.path.isdir(os.path.expanduser(candidate)):
                model_path = candidate
        model = create_qwen35_model(
            mesh_device,
            model_path=model_path,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            use_paged_kv_cache=True,
        )
        # GDN's forward_prefill is hardcoded B_pf=1 (gdn.py); the framework's
        # batched-prefill optimization (tt_transformers/tt/generator.py:467-473)
        # would otherwise push B>1 through it.
        model.args.disable_batched_prefill = True
        return cls(model, model.args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    # ── Per-request state hooks ────────────────────────────────────────

    def _reset_states_before_prefill(self):
        """Reset prefill state for every layer before a new user is prefilled.

        Mirrors test_e2e_generate.py:_reset_prefill_states (lines 501-518) for
        the paged path: skip reset_state() on full_attention layers (their KV
        lives in external paged caches, and reset_state() would allocate the
        unused [B,1,max_seq_len,HD] static buffer); GDN layers leave the batched
        decode state alone (other in-flight users may still own it) and only
        ensure their B=1 prefill state is fresh — handled by the lazy
        _init_prefill_states() inside forward_prefill once _prefill_*_states
        is None (set to None by replicate_prefill_state_to_slot at end of prior
        request).
        """
        layer_types = self.model_args[0].layer_types
        for layer_idx, layer in enumerate(self.model[0].layers):
            attn = layer.attention
            is_paged_attn = layer_types[layer_idx] == "full_attention"
            if hasattr(attn, "reset_state") and not is_paged_attn:
                # Linear-attention (GDN) batched decode state: leave it alone —
                # other slots may still hold valid state for in-flight users.
                # We only need to ensure *this user's* prefill state is fresh,
                # which is handled by the lazy init in forward_prefill below.
                pass

    def _commit_prefill_state_to_slot(self, user_id):
        """After per-user prefill, write GDN recurrent/conv state into row
        user_id of the batched decode state. Paged attention layers no-op:
        their KV writes already happened in forward_prefill_paged via
        paged_fill_cache, which honors the page_table for slot routing.
        """
        for layer in self.model[0].layers:
            attn = layer.attention
            if hasattr(attn, "replicate_prefill_state_to_slot"):
                attn.replicate_prefill_state_to_slot(user_id)

    # ── vLLM interface ────────────────────────────────────────────────

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        **kwargs,
    ):
        """Per-user prefill with GDN slot-aware state writeback.

        Each user is prefilled in isolation by calling super().prefill_forward_text
        with batch=1 and empty_slots=[0]. The page_table row for the real slot
        is forwarded so paged_fill_cache writes KV into that user's physical
        blocks; afterward, GDN's prefill rec_state is committed into the real
        slot's row of the batched rec_states tensor.
        """
        batch_size = tokens.shape[0]
        if empty_slots is None:
            empty_slots = list(range(batch_size))
        assert (
            len(empty_slots) == batch_size
        ), f"empty_slots length ({len(empty_slots)}) must match batch dim ({batch_size})"

        if isinstance(prompt_lens, torch.Tensor):
            prompt_lens_list = prompt_lens.tolist()
        elif prompt_lens is None:
            prompt_lens_list = [tokens.shape[1]] * batch_size
        else:
            prompt_lens_list = list(prompt_lens)

        outputs = []
        for idx, user_id in enumerate(empty_slots):
            user_tokens = tokens[idx : idx + 1]
            user_prompt_lens = torch.tensor([prompt_lens_list[idx]], dtype=torch.int32)
            user_page_table = page_table[user_id : user_id + 1] if page_table is not None else None

            # Lazy GDN prefill-state allocation happens inside forward_prefill;
            # here we just ensure the prior request's bookkeeping is clean.
            self._reset_states_before_prefill()

            # enable_trace=False: framework's prefill trace would call the
            # forward repeatedly with the same captured tensors, but our reset
            # hooks change prefill-state tensor identities across requests.
            # Trace support is deferred (see plan).
            kwargs_no_trace = dict(kwargs)
            kwargs_no_trace["enable_trace"] = False
            out = super().prefill_forward_text(
                user_tokens,
                page_table=user_page_table,
                kv_cache=kv_cache,
                prompt_lens=user_prompt_lens,
                empty_slots=[0],
                warmup_prefill=False,
                **kwargs_no_trace,
            )

            self._commit_prefill_state_to_slot(user_id)
            outputs.append(out)

        if len(outputs) == 1:
            return outputs[0]
        # super returns either logits [B, 1, vocab] or (tokens, logprobs); for the
        # multi-user case, concatenate the per-user batch dim.
        if isinstance(outputs[0], tuple):
            stacked = tuple(
                torch.cat([o[i] for o in outputs], dim=0) if outputs[0][i] is not None else None
                for i in range(len(outputs[0]))
            )
            return stacked
        return torch.cat(outputs, dim=0)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        # vLLM passes num_layers = number of attention layers, but the framework
        # forward indexes kv_cache by total layer index. Build a full-length list
        # with None for GDN layers so kv_cache[i] works for every i.
        layer_types = self.model_args[0].layer_types
        mesh = self.model[0].mesh_device
        cache_path = self.cache_path
        opts = getattr(self.model_args[0], "optimizations", None)

        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        kv_tt = []
        for layer_idx in tqdm(range(len(layer_types)), desc="Allocating TT kv caches"):
            if layer_types[layer_idx] == "linear_attention":
                # GDN layers use internal recurrence state, no KV cache needed
                kv_tt.append(None)
                continue
            kv_dtype = None
            if opts is not None:
                try:
                    kv_dtype = opts.get_tensor_dtype(decoder_id=layer_idx, tensor=TensorGroup.KV_CACHE)
                except Exception:
                    kv_dtype = None
            if kv_dtype is None:
                kv_dtype = ttnn.bfloat8_b
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=kv_dtype,
                    cache_file_name=cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
                )
                for kv in ["k", "v"]
            ]
            kv_tt.append(kv_tt_i)

        # Wrap in list for data-parallel indexing (DP=1)
        return [kv_tt]
