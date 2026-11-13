# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM wrappers for Qwen3.5/3.6/3.8 on Blackhole: thin tt_transformers Generator subclasses.

Hybrid model: paged-KV attention + GDN recurrent-state layers. GDN forbids token-padding and isn't position-general,
so prefill is model-owned (masked-bucket for short prompts / chunk-outer trace for long) while Generator drives decode
only. Both classes keep the GDN state of each request in vLLM-owned KV cache blocks (MambaSpec groups,
tt/gdn_block_state.py); Qwen36MTPContractForCausalLM adds speculative decoding with the MTP drafter head.
"""

import math
import os
from collections import defaultdict
from typing import Mapping, Optional

import torch
from loguru import logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.demos.blackhole.qwen36.tt.common import create_tt_model
from models.demos.blackhole.qwen36.tt.generator_interface import warmup_decode_buckets
from models.tt_transformers.tt.generator import Generator

_PREFILL_WARMUP_CHUNK = 2048
# Compute cores gdn_spec_step uses for B rows of Nv value heads (one per row and head) on a Blackhole P150 / P300.
_SPEC_STEP_CORES = 110
# tt/vision/model.py pads a request's vision patches to (n // _VISION_BUCKET + 1) * _VISION_BUCKET.
_VISION_BUCKET = 2048
# Patches the vision warmup covers when vLLM's --mm-processor-kwargs sets no max_pixels: about a 2 MP image.
_VISION_WARMUP_PATCHES = 8191


class TT_Qwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Serve a single visual item per request (B=1, max_concurrency=1). Image and video are both
        # supported, but only ONE modality per request: the model's vision splice keys off a single
        # placeholder token id (image_token_id XOR video_token_id), so a mixed image+video prompt
        # cannot be spliced correctly.
        return {"image": 1, "video": 1}


class TT_Qwen3_5SpecProcessingInfo(TT_Qwen3_5ProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # the spec engine's prefill is text-only: vLLM rejects visual items at admission (HTTP 400)
        return {"image": 0, "video": 0}


class _Qwen36Generator(Generator, SupportsMultiModal):
    """Construction, KV budget, sampling checks and multimodal helpers shared by the Qwen3.6 vLLM classes."""

    # Decode bucketing keeps several traces live and refreshes the selected
    # bucket's inputs before replay, so their I/O buffers may safely overlap.
    _tt_allow_decode_trace_buffer_reuse = True
    decode_input_update_contract = 1
    _ENABLE_MTP = False  # MTP contract subclass overrides

    # supports_async_decode=False: async decode assumes on-device token/position continuity, which
    # corrupts Qwen's GDN scan. supports_sample_on_device=True: on-device sampling is decode-only.
    # supports_device_grammar=True: decode_forward defers to Generator, whose deferred payload and
    # sample_deferred_decode implement the plugin's device grammar contract; bucketed logits are
    # padded to the sampler width, so the full-batch mask applies to every bucket.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        "supports_device_grammar": True,
        "max_device_top_k": 32,
    }

    def _validate_device_sampling_request(self, requested):
        if not requested:
            return
        for model in self.model:
            if model.sampling is not None:
                continue
            mesh_shape = tuple(int(dim) for dim in model.mesh_device.shape)
            logits_per_device = math.ceil(model.args.vocab_size / model.num_devices)
            raise RuntimeError(
                "Qwen3.6 on-device sampling requires a certified TP topology (1x4 or 1x8) "
                f"with at most 65536 logits/device; got mesh={mesh_shape}, "
                f"vocab={model.args.vocab_size}, logits/device={logits_per_device}. "
                "Unset sample_on_device_mode for host sampling."
            )

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        """All-user KV capacity (the shared paged-KV token pool).

        QWEN36_MAX_TOKENS_ALL_USERS overrides it with a FIXED pool (set per device+model from the
        tt-inference-server spec's env_vars, mirroring GEMMA4_MAX_TOKENS_ALL_USERS). This decouples
        the pool from max_model_len × max_num_seqs so ONE config serves both a single long request
        (up to max_model_len) and a batch of shorter ones (sum of lengths ≤ pool) — e.g. 524288 =
        1×256K or 8×64K. Without the override, fall back to max_model_len × max_num_seqs (the old
        per-config product) so existing single-mode specs are unchanged."""
        override = os.environ.get("QWEN36_MAX_TOKENS_ALL_USERS")
        if override:
            return int(override)
        if max_model_len is not None:
            return int(max_model_len) * int(max_num_seqs or 1)
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_num_seqs=max_num_seqs,
            **kwargs,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        # Weights dir: MODEL_WEIGHTS_DIR → HF_MODEL → hf_config._name_or_path; a hub id resolves to a local snapshot.
        name_or_path = os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL") or hf_config._name_or_path
        if name_or_path and not os.path.isdir(os.path.expanduser(name_or_path)):
            from huggingface_hub import snapshot_download

            # When offline/CI, resolve from the local cache only so snapshot_download
            # reads the cached refs instead of reaching the HF API (refused by HF_HUB_OFFLINE=1).
            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
            name_or_path = snapshot_download(name_or_path, local_files_only=offline)
        args, model, _ = create_tt_model(
            mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            hf_model=name_or_path,
            # vLLM does not drive spec decode yet (vllm-tt-plugin#110): skip the MTP head, its weights and its KV cache.
            enable_mtp=cls._ENABLE_MTP,
        )
        # Attach the TT vision tower so prefill can splice image/video embeddings (multimodal path).
        # No-op cost for text-only requests; get_image_features / get_video_features are only invoked
        # when a request actually carries pixel_values / pixel_values_videos.
        model.init_vision_model()
        inst = cls([model], [args], mesh_device)
        inst._vision_warmup_patches = _vision_warmup_patches(model)
        return inst

    @staticmethod
    def _has_visual(kwargs, pixel_key):
        """True only when the request carries REAL visual data for this modality. vLLM attaches an
        empty pixel_values placeholder to text requests for a multimodal-registered model, so a
        plain ``is not None`` check misclassifies text as multimodal. Mirrors the emptiness test in
        _gather_user_visual (key absent / empty list / first item None => text-only)."""
        v = kwargs.get(pixel_key)
        return v is not None and len(v) > 0 and v[0] is not None

    @staticmethod
    def _gather_user_visual(kwargs, pixel_key, grid_key):
        """Pull this (B=1) user's patches + (t,h,w) grids for one modality out of the vLLM kwargs.

        Returns (pixel_values, grid_thw) or None when the request carries nothing for this modality.
        Multiple items for the user arrive as lists; concat the patches and stack the grids (same
        shape get_image_features / get_video_features expect: [num_patches, patch_dim] + [N, 3]).
        """
        if pixel_key not in kwargs or len(kwargs[pixel_key]) == 0 or kwargs[pixel_key][0] is None:
            return None
        pixel_values = kwargs[pixel_key][0]
        grid_thw = kwargs[grid_key][0]
        if isinstance(pixel_values, list) and len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values, dim=0)
            grid_thw = torch.stack([g.to(dtype=torch.int32) for g in grid_thw], dim=0)
        return pixel_values, grid_thw

    def _compute_vision_tokens(self, model, kwargs):
        """Run the vision tower for this (single-user, B=1) request, if it carries images or video.

        Mirrors the Qwen3-VL generator's multimodal check: pull this user's pixels + grid out of
        the vLLM kwargs and return the packed embeddings (ttnn [num_vision_tokens, H]) for prefill
        to splice in. Returns None for a text-only request, so the whole multimodal path is skipped.

        Image and video share the vision tower; dispatching to get_video_features (vs
        get_image_features) is what tells the model to splice into video_token_id placeholders and
        build the video M-RoPE. A request carries at most one visual modality (see
        get_supported_mm_limits); video takes precedence if both are somehow present.
        """
        video = self._gather_user_visual(kwargs, "pixel_values_videos", "video_grid_thw")
        if video is not None:
            self._check_vision_bucket(video[1])
            return model.get_video_features(*video)

        image = self._gather_user_visual(kwargs, "pixel_values", "image_grid_thw")
        if image is not None:
            self._check_vision_bucket(image[1])
            return model.get_image_features(*image)

        return None

    def warmup_vision(self):
        """Run the vision tower once per patch bucket up to _vision_warmup_patches, before any trace is captured.

        Every device op of the vision tower depends only on the padded patch bucket (get_*_features trim the valid
        rows on the host), so one image per bucket creates every program a request in that bucket runs. A program
        first created after the traces were captured can allocate device buffers that a later trace replay
        overwrites: at TP=8 the composite all_gather and reduce_scatter of the vision tower own L1 global
        semaphores, and the next vision call of that shape hangs in the CCL. An image of 32 x w patches lands in
        bucket 32 * w + 1024."""
        model = self.model[0]
        top = getattr(self, "_vision_warmup_patches", None)
        if getattr(model, "vision_model", None) is None or not top:
            return
        vc = model.vision_args.hf_config.vision_config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
        buckets = range(_VISION_BUCKET, (top // _VISION_BUCKET + 1) * _VISION_BUCKET + 1, _VISION_BUCKET)
        for bucket in buckets:
            w = bucket // 32 - 32
            pixels = torch.zeros(32 * w, patch_dim, dtype=torch.float32)
            ttnn.deallocate(model.get_image_features(pixels, torch.tensor([[1, 32, w]], dtype=torch.int32)))
        model._req_image_grid_thw = None
        model._req_video_grid_thw = None
        ttnn.synchronize_device(self.mesh_device)
        self._vision_warmed_bucket = buckets[-1]
        logger.info(f"Vision tower warmed for patch buckets {list(buckets)}")

    def _check_vision_bucket(self, grid_thw):
        """Refuse a visual input whose patch bucket warmup_vision did not compile.

        Running it would create vision programs after trace capture, and a later request of the same size would hang
        the device until a reset. Raising fails this request at once, with the cause in the log. No check when
        warmup_vision did not run (tests, demos)."""
        warmed = getattr(self, "_vision_warmed_bucket", None)
        if not warmed:
            return
        n = int(torch.as_tensor(grid_thw).reshape(-1, 3).prod(dim=1).max())
        bucket = (n // _VISION_BUCKET + 1) * _VISION_BUCKET
        if bucket > warmed:
            raise ValueError(
                f"Vision input of {n} patches needs vision bucket {bucket}, above the largest bucket warmed before "
                f"trace capture ({warmed} patches); running it would compile the vision tower after capture and can "
                "hang the device. Launch vLLM with --mm-processor-kwargs '{\"max_pixels\": N}' (for example 1048576) "
                "so that every accepted input fits a warmed bucket."
            )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36ForCausalLM(_Qwen36Generator):
    """vLLM-compatible wrapper for Qwen3.5/3.6/3.8 on a Blackhole TP mesh (P150x4, P150x8).

    vLLM owns the GDN state of every request: get_kv_cache_spec declares one MambaSpec per GDN layer, and each GDN
    layer keeps a request's recurrent state and convolution history in the KV cache blocks that its MambaSpec group
    gives the request (tt/gdn_block_state.py). A prefill starts from zero state or from the state of its start block
    (a later prompt chunk, or a prefix-cache hit in mamba_cache_mode "align"), writes vLLM's prefill checkpoint on the
    way, and copies its end state into the destination block; decode reads and writes the blocks with gdn_decode_step.
    """

    # supports_chunked_prefill: a prefill continues from its start block (prefill_forward). supports_prefix_caching:
    # mamba_cache_mode "align" restores prefixes from the state blocks vLLM keeps; the plugin requires chunked prefill
    # with it.
    model_capabilities = {
        **_Qwen36Generator.model_capabilities,
        "supports_prefix_caching": True,
        "supports_chunked_prefill": True,
    }

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """vllm-tt-plugin hook: one FullAttentionSpec per attention layer and one MambaSpec per GDN layer, per device
        at the MESH_DEVICE TP degree (tt/kv_cache_spec.py). The attention block size is the smallest one, at least
        --block-size, whose K and V block holds one GDN state. Raises ValueError for an unsupported mesh."""
        from models.demos.blackhole.qwen36.tt.kv_cache_spec import (
            kv_cache_bf8_enabled,
            mesh_tp_degree,
            qwen36_kv_cache_spec,
        )

        tc = vllm_config.model_config.hf_config
        tc = tc.get_text_config() if hasattr(tc, "get_text_config") else tc
        cc = vllm_config.cache_config
        specs = qwen36_kv_cache_spec(
            tc,
            mesh_tp_degree(),
            kv_bf8=kv_cache_bf8_enabled(),
            mamba_block_size=getattr(cc, "mamba_block_size", None) or vllm_config.model_config.max_model_len,
            mamba_cache_mode=getattr(cc, "mamba_cache_mode", None) or "none",
            min_block_size=getattr(cc, "block_size", None),
            num_speculative_blocks=cls._num_speculative_blocks(vllm_config),
        )
        attn = next(spec for name, spec in specs.items() if name.endswith(".self_attn"))
        logger.info(
            f"Qwen GDN state in vLLM KV cache blocks: attention block size {attn.block_size} "
            f"({attn.page_size_bytes} B per layer and block)"
        )
        return specs

    @classmethod
    def _num_speculative_blocks(cls, vllm_config):
        """K of the MambaSpec groups: this class verifies no drafts."""
        return 0

    def _verify_tokens(self):
        """T = 1 + K of a verify on the GDN state blocks; 0 without one."""
        return 0

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """vllm-tt-plugin hook. One K and V buffer pair per attention layer; the GDN layers of the same layer slot
        keep their request state inside it (gdn_block_state.GdnBlockState). The GDN layers keep only a batch-1
        prefill workspace. Returns the attention [K, V] pairs."""
        from models.demos.blackhole.qwen36.tt.gdn_block_state import GdnBlockState, is_mamba_entry

        model = self.model[0]
        if model.num_devices < 2:
            raise ValueError("Qwen36ForCausalLM keeps the GDN state in KV cache blocks only on a TP mesh")
        shapes = {tuple(int(d) for d in e[0]) for e in per_layer_specs if not is_mamba_entry(e)}
        if len(shapes) != 1:
            raise ValueError(f"the attention layers need one K / V cache shape, got {sorted(shapes)}")
        kv_caches = model.allocate_kv_caches(shapes.pop(), ttnn.bfloat16, batch_size=1)
        model.gdn_blocks = GdnBlockState(model, per_layer_specs, kv_caches, verify_tokens=self._verify_tokens())
        return kv_caches

    def zero_kv_cache_blocks(self, kv_cache=None, block_ids=()):
        """vllm-tt-plugin hook: zero vLLM's new_block_ids_to_zero in every K and V buffer before the step."""
        self.model[0].gdn_blocks.zero_blocks(block_ids)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """Prefill tokens [start_pos[u], prompt_lens[u]) of each row u on the GDN layers' batch-1 workspace, then copy
        the row's end state into its destination blocks (mamba_state.windows). ``tokens`` holds each row's tokens from
        position 0.

        At start_pos 0 the row starts from zero state. Otherwise it continues from the full state of its start blocks
        (mamba_state.start_blocks: a later chunk of the prompt, or a prefix-cache hit in mamba_cache_mode "align"),
        attending over the KV cache that its page table holds for the earlier tokens; the start blocks are only read.
        When the row has a prefill checkpoint after p tokens, the forward of the prefill's last chunk also computes
        the state after p tokens when p is inside that chunk; otherwise the prefill stops after token p and continues
        (gdn_block_state.prefill_segments). Either way the checkpoint blocks get that state. Returns host logits
        [N, 1, vocab] at each row's last token."""
        from models.demos.blackhole.qwen36.tt.gdn_block_state import row_blocks

        model = self.model[0]
        mamba_state = kwargs.get("mamba_state")
        if mamba_state is None:
            raise ValueError("Qwen36ForCausalLM keeps the GDN state in vLLM KV cache blocks and needs mamba_state")
        # Require real visual data, not a non-None empty pixel_values placeholder from vLLM on text requests.
        visual = self._has_visual(kwargs, "pixel_values") or self._has_visual(kwargs, "pixel_values_videos")
        if visual and model.args.max_batch_size > 1:
            raise ValueError(
                "batched (max_num_seqs>1) serving is text-only; multimodal is single-sequence "
                "(max_concurrency=1). Run the model at max_num_seqs=1 for image/video requests."
            )
        # Read to the host once, before any segment's trace replay can overwrite the device rows (vision_rows_host).
        vision_tokens = model.vision_rows_host(self._compute_vision_tokens(model, kwargs)) if visual else None
        N = tokens.shape[0]
        start_pos = kwargs.get("start_pos")
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        blocks = model.gdn_blocks
        rows = []
        for u in range(N):
            T = int(prompt_lens[u]) if prompt_lens is not None else tokens.shape[1]
            rb = row_blocks(mamba_state, u)
            segments = self._prefill_segments(int(start_pos[u]) if start_pos is not None else 0, T, rb)
            logger.info(f"Prefilling row {u}: tokens [{segments[0][0]}, {T})")
            if segments[0][0] > 0:
                blocks.restore(mamba_state, rb.start)
            on_chunk = self._prefill_chunk_hook(u, tokens[u, :T], pt[u : u + 1], mamba_state, kwargs)
            extra = {} if on_chunk is None else {"on_chunk": on_chunk}
            if visual:
                # An image's M-RoPE positions need all of its tokens, and a checkpoint can split an image.
                extra["rope_token_ids"] = tokens[u : u + 1, :T]
            logits = None
            for a, b, checkpoint_pos in segments:
                if logits is not None:
                    ttnn.deallocate(logits)
                    blocks.write_state(mamba_state, rb.checkpoint)
                # Short prompts run one masked-bucket forward; longer ones replay the chunk trace, then the masked tail.
                logits = model.prefill_traced_chunked(
                    tokens[u : u + 1, :b],
                    pt[u : u + 1],
                    actual_len=b,
                    vision_tokens=vision_tokens,
                    start_pos=a,
                    **extra,
                    **({"checkpoint_pos": checkpoint_pos} if checkpoint_pos else {}),
                )
                if checkpoint_pos:
                    blocks.write_checkpoint(mamba_state, rb.checkpoint)
            rows.append(ttnn.to_torch(logits, mesh_composer=comp).reshape(-1, model.args.vocab_size)[:1].float())
            ttnn.deallocate(logits)
            blocks.write_state(mamba_state, rb.dest)
        # Zero rope deltas: the multimodal M-RoPE delta is applied model-side (rope.rope_delta).
        return torch.stack(rows).reshape(N, 1, -1), torch.zeros(N, dtype=torch.long)

    def _prefill_chunk_hook(self, row, row_tokens, page_table_row, mamba_state, kwargs):
        """on_chunk of Qwen36Model.prefill_traced_chunked for prefill row ``row`` (tokens [0, end) in ``row_tokens``),
        or None. The ordinary model reads nothing from the prefill's hidden states."""
        return None

    def _prefill_segments(self, start, end, rb):
        """gdn_block_state.prefill_segments for a row that vLLM schedules from ``start`` to ``end`` with the state
        blocks ``rb``. The row must start on a KV block boundary, and prefill_segments raises otherwise: vllm-tt-plugin
        ends the non-final prefill chunks of a MambaSpec model there and rejects a prefix_match_unit below the block
        size, so vLLM's prefix hits end there too. A multimodal row starts at 0 with every image in it, or, after a
        preemption, continues on a block boundary after its last image; its segments take their M-RoPE positions from
        the whole row (prefill_forward), so a checkpoint can also fall inside an image."""
        from models.demos.blackhole.qwen36.tt.gdn_block_state import prefill_segments

        block_size = self.model[0].gdn_blocks.layout.block_size
        if (start == 0 and any(rb.start)) or (start > 0 and min(rb.start) <= 0):
            raise ValueError(f"a prefill row at start_pos {start} has the start blocks {rb.start}")
        if rb.checkpoint_pos and rb.checkpoint_pos <= start:
            raise ValueError(f"this prefill row cannot write the prefill checkpoint after {rb.checkpoint_pos} tokens")
        return prefill_segments(start, end, rb.checkpoint_pos, block_size, self.model[0].prefill_chunk_size)

    def decode_forward(self, *args, **kwargs):
        args = list(args)

        def _read(name, pos):
            if name in kwargs:
                return kwargs[name]
            return args[pos] if pos < len(args) else None

        def _write(name, pos, val):
            if name in kwargs:
                kwargs[name] = val
            elif pos < len(args):
                args[pos] = val

        # Traced decode: trace captured in warmup, replayed here. The GDN layers read the state of each row from the
        # blocks that their persistent page index names, so the index is set before the replay.
        if not getattr(self, "_decode_logged", False):
            self._decode_logged = True
            logger.info("Decode trace replay active (Qwen)")
        mamba_state = kwargs.pop("mamba_state", None)
        if mamba_state is not None:
            # A row that vLLM moved (slot_remap) or gave to another request keeps reading its request's blocks: moving
            # a row rewrites the page index and moves no state.
            self.model[0].gdn_blocks.set_decode_blocks(mamba_state)
        # Decode bucketing (default on; TT_DECODE_BUCKETING=0 off): slice host inputs to the
        # smallest power-of-2 width >= active prefix [0:num_active) before the base forward.
        # No runner edit / output re-pad — plugin reads unpadded_batch_size in slot order.
        # Each width keeps its own trace metadata, inputs, output, and pre-capture staged inputs.
        tokens = _read("tokens", 0)
        if os.environ.get("TT_DECODE_BUCKETING", "1") == "1" and tokens is not None:
            start_pos = _read("start_pos", 1)
            width = int(tokens.shape[0])
            num_active = int((start_pos != -1).sum()) if start_pos is not None else width
            num_active = max(1, min(num_active, width))
            bucket = min(width, 1 << max(0, (num_active - 1).bit_length()))  # smallest pow2 >= num_active
            # Keep full width when slot_remap is set: the sampler's per-slot state remap indexes the full slot
            # space. Rare (row moves only); bucketing resumes next step. Check kw + positional #9.
            if _read("slot_remap", 9) is not None:
                bucket = width
            if bucket < width:
                _write("tokens", 0, tokens[:bucket])
                if start_pos is not None:
                    _write("start_pos", 1, start_pos[:bucket])
                page_table = _read("page_table", 2)
                if page_table is not None:
                    _write("page_table", 2, page_table[:bucket])
                tokens = tokens[:bucket]

        if tokens is not None:
            B = int(tokens.shape[0])
            store = getattr(self, "_bucket_trace_store", None)
            if store is None:
                store = self._bucket_trace_store = {}
            if B not in store:
                store[B] = (defaultdict(lambda: None), defaultdict(lambda: None), defaultdict(lambda: None), {})
            (
                self.trace_ids_decode,
                self.trace_inputs_decode,
                self.trace_output_decode,
                self._prepared_decode_traces,
            ) = store[B]
            # Key the sampling trace by bucket width too: Generator binds it to one logits tensor
            # by identity, and each decode-bucket width has its own.
            for _m in self.model:
                _sm = getattr(_m, "sampling", None)
                if _sm is not None and hasattr(_sm, "set_trace_bucket"):
                    _sm.set_trace_bucket(B)
        return super().decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        # The eager call (enable_trace=False) allocates and compiles everything; the traced call only
        # records, so nothing persistent is allocated once a trace is live (#56474).
        model = self.model[0]
        if enable_trace and model._chunked_trace_id is not None and model._masked_traces_recorded:
            return
        # Size the chunk-trace page table to the full KV cache so served ISL isn't capped; round to a multiple of 32:
        # paged/chunked SDPA needs the page-table stick % 32 == 0.
        num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        # A visual request runs only at max_batch_size 1 (prefill_forward refuses it otherwise), so only that launch
        # compiles the vision tower before trace capture.
        if not enable_trace and model.args.max_batch_size == 1:
            self.warmup_vision()
        logger.info(
            f"Qwen prefill warmup ({'record' if enable_trace else 'prepare'}): chunk={_PREFILL_WARMUP_CHUNK}, "
            f"page_table_blocks={num_blocks}"
        )
        model.prepare_prefill_trace_chunked(self.mesh_device, page_table, chunk_size=_PREFILL_WARMUP_CHUNK)
        # The partial last chunk of every prefill (the whole of a prompt up to one chunk) replays a trace per bucket.
        model.prepare_prefill_traces_masked()
        if enable_trace:
            if model._chunked_trace_id is None:
                model.record_prefill_trace_chunked(self.mesh_device)
            model.record_prefill_traces_masked()
        model.gdn_blocks.warmup()

    def warmup_model_decode(self, *args, **kwargs):
        # Defer to WarmupForwardMixin, which warms the paged-SDPA + GDN decode path at pos 0.
        # Drop stale `non_greedy_decoding_on_device` from the old vLLM plugin; no-op for Qwen.
        kwargs.pop("non_greedy_decoding_on_device", None)
        self._validate_device_sampling_request(kwargs.get("can_sample_on_device", False))
        return warmup_decode_buckets(self, super().warmup_model_decode, *args, **kwargs)


def _vision_warmup_patches(model):
    """The most vision patches an accepted image can have, or None when the launch accepts no image or video.

    Read from the current vLLM config, as vLLM's processor resizes images to at most
    --mm-processor-kwargs max_pixels; _VISION_WARMUP_PATCHES without it. Video is not bounded."""
    try:
        from vllm.config import get_current_vllm_config

        mc = get_current_vllm_config().model_config
        if not mc.supports_multimodal_inputs:
            return None
        kw = getattr(getattr(mc, "multimodal_config", None), "mm_processor_kwargs", None) or {}
        max_pixels = kw.get("max_pixels")
    except Exception:
        max_pixels = None
    if max_pixels is None:
        return _VISION_WARMUP_PATCHES
    patch = model.vision_args.hf_config.vision_config.patch_size
    return -(-int(max_pixels) // patch**2)


def _mesh_tp(default=4):
    """Tensor-parallel degree of this engine's mesh (kv_cache_spec.mesh_tp_degree); ``default`` when MESH_DEVICE is
    unset or names no mesh."""
    from models.demos.blackhole.qwen36.tt.kv_cache_spec import mesh_tp_degree

    try:
        return mesh_tp_degree()
    except ValueError:
        return default


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5SpecProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36MTPContractForCausalLM(Qwen36ForCausalLM):
    """Qwen36ForCausalLM with speculative decoding over the MTP drafter head (tt/spec_engine.py MTPBlockEngine), for the
    plugin's spec-decode contract (B * (K+1) <= 32), synchronous scheduling and host acceptance.

    vLLM owns the GDN state as for Qwen36ForCausalLM; its MambaSpec groups declare num_speculative_blocks = K, so a
    verify row gets 1 + K state blocks per group (mamba_state.windows). Every decode step is a verify: it reads each
    row's full state from mamba_state.start_blocks and writes the full state after query token t, recurrent state and
    convolution history, into windows[:, t] (gdn_spec_step on the blocks), and the runner points the next step's
    start_blocks at the block of the count its accept walk committed. An ordinary [B, 1] decode runs as a verify
    without drafts. Prefill, prefix caching and chunked prefill are Qwen36ForCausalLM's; the drafter's KV cache is
    model-owned and indexed by the attention block ids (Qwen36Model._allocate_mtp_kv_cache)."""

    _ENABLE_MTP = True
    _SPEC_K = int(os.environ.get("QWEN36_SPEC_K", "3"))  # vLLM warmup does not carry K
    _SUPPORTED_K = (3, 7, 11)
    # decode_forward is overridden and does not execute the reload commands (DECODE_RELOAD_CONTRACT legacy mode)
    decode_input_update_contract = 0

    # verify answers argmax_ids (greedy) or logits ([B, 1+K, V] for host acceptance);
    # the first token after prefill is host-sampled
    model_capabilities = {
        **Qwen36ForCausalLM.model_capabilities,
        "supports_spec_decode": True,
        "spec_requirements": ("device_propose", "hidden_feed"),
        "spec_hidden_handoff": ("on_device",),
        "supports_async_decode": False,
        "supports_async_spec_decode": False,
        "output_tokens_per_step": 1,
        "supports_sample_on_device": False,
        # Device grammar masks the device sampler, which the spec path does not use.
        "supports_device_grammar": False,
    }

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Config-time plan; never raises and never touches devices."""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        k = cls._SPEC_K
        if k not in cls._SUPPORTED_K:
            return SpecReject(reason=f"QWEN36_SPEC_K={k} unsupported; need one of {cls._SUPPORTED_K}", supported_k=())
        if max_num_seqs * (k + 1) > 32:
            return SpecReject(
                reason=f"max_num_seqs={max_num_seqs} x (K+1)={k + 1} verify rows exceed the 32-row tile"
                " (K is fixed by QWEN36_SPEC_K)",
                supported_k=tuple(x for x in cls._SUPPORTED_K if max_num_seqs * (x + 1) <= 32),
            )
        if requested_k < k:
            return SpecReject(reason=f"requested_k={requested_k} < engine K={k}", supported_k=(k,))
        try:
            from models.demos.blackhole.qwen36.tt.spec_engine import spec_memory_costs

            mc = vllm_config.model_config
            tc = mc.hf_config
            tc = tc.get_text_config() if hasattr(tc, "get_text_config") else tc
            if int(getattr(tc, "mtp_num_hidden_layers", 0) or 0) < 1:
                return SpecReject(reason="checkpoint has no MTP weights (mtp_num_hidden_layers < 1)", supported_k=())
            tp = _mesh_tp()
            nv = int(getattr(tc, "linear_num_value_heads")) // tp
            if max_num_seqs * nv > _SPEC_STEP_CORES:
                return SpecReject(
                    reason=f"max_num_seqs={max_num_seqs} x {nv} GDN value heads per device exceed the "
                    f"{_SPEC_STEP_CORES} cores gdn_spec_step uses (one per row and head) at TP={tp}",
                    supported_k=(),
                )
            costs = spec_memory_costs(tc, tp, max_num_seqs, k, max_seq_len=mc.max_model_len, gdn_state_in_blocks=True)
            # costs are for the whole batch: per_seq scales with B, fixed is shared
            per_seq = math.ceil((costs["per_seq"]["total"] + costs["fixed"]["total"]) / max_num_seqs)
            per_token = int(costs["per_token"]["total"])
            mib = 1 << 20
            logger.info(
                f"[spec] B={max_num_seqs} cost/seq {per_seq / mib:.1f} MiB (batch: per_seq "
                f"{costs['per_seq']['total'] / mib:.1f}; fixed {costs['fixed']['total'] / mib:.1f}), "
                f"per_token {per_token} B"
            )
        except Exception as e:
            return SpecReject(reason=f"cost estimate failed: {e}", supported_k=())
        return SpecPlan(
            effective_k=k,
            lanes_per_request=1,
            extra_bytes_per_seq=per_seq,
            extra_bytes_per_token=per_token,
            accept_modes=("argmax_ids", "logits"),
            drafter_state="internal",
            supports_narrow_decode=True,
        )

    @classmethod
    def _num_speculative_blocks(cls, vllm_config):
        """K = QWEN36_SPEC_K, which the platform publishes as num_speculative_tokens (SpecPlan.effective_k)."""
        sc = getattr(vllm_config, "speculative_config", None)
        k = int(getattr(sc, "num_speculative_tokens", 0) or 0) if sc is not None else 0
        if k != cls._SPEC_K:
            raise ValueError(
                f"Qwen36MTPContractForCausalLM verifies K = QWEN36_SPEC_K = {cls._SPEC_K} drafts per step, and its GDN "
                f"state needs K + 1 blocks per verify row, but the launch speculates {k} tokens; launch with "
                f"num_speculative_tokens {cls._SPEC_K}"
            )
        return k

    def _verify_tokens(self):
        return self._SPEC_K + 1

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, *args, **kwargs):
        self = super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, max_seq_len, *args, **kwargs)
        assert self.model[0].mtp is not None, "MTP head not built (checkpoint has no MTP weights?)"
        return self

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Qwen36ForCausalLM's allocation plus the drafter's KV cache (one block past the pool, the drafter's scratch),
        which the drafter runs at the full decode width."""
        kv_caches = super().allocate_kv_cache_per_layer(per_layer_specs)
        model = self.model[0]
        model.mtp.attention.B = model.args.max_batch_size
        return kv_caches

    def _engine(self):
        from models.demos.blackhole.qwen36.tt.spec_engine import MTPBlockEngine, get_spec_engine

        return get_spec_engine(self.model[0], self.model[0].args.max_batch_size, self._SPEC_K, MTPBlockEngine)

    def _table_width(self):
        from models.demos.blackhole.qwen36.tt.spec_engine import spec_table_width

        model = self.model[0]
        return spec_table_width(model.args.max_seq_len, self._SPEC_K, model.gdn_blocks.layout.block_size)

    def _table(self, pt):
        """Zero-pad (or trim) a vLLM page table [rows, w] to the engine width; block 0 is vLLM's null block."""
        pt = pt if isinstance(pt, torch.Tensor) else ttnn.to_torch(pt)
        w = self._table_width()
        if pt.shape[1] >= w:
            return pt[:, :w].to(torch.int32)
        out = torch.zeros(pt.shape[0], w, dtype=torch.int32)
        out[:, : pt.shape[1]] = pt
        return out

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        """Qwen36ForCausalLM's prefill warmup (chunk trace, block programs), then the engine's buffers and drafter
        prompt programs before any trace is captured."""
        super().warmup_model_prefill(kv_cache, enable_trace, *args, **kwargs)
        eng = self._engine()
        if enable_trace:
            eng.record_prefill()
            return
        if eng.prefill_prepared:
            return
        w = self._table_width()
        pool = int(kv_cache[0][0].shape[0]) if kv_cache else w
        assert w <= pool, (
            f"vLLM KV pool too small: {pool} blocks < table width {w} (max_model_len + K + 1 = "
            f"{self.model[0].args.max_seq_len} + {self._SPEC_K} + 1 tokens)"
        )
        # rows start at distinct blocks: the verify's KV-collision check at warm position 0
        B = self.model[0].args.max_batch_size
        eng.prepare_prefill(torch.stack([(torch.arange(w) * B + u) % pool for u in range(B)]).to(torch.int32))

    def warmup_model_decode(self, kv_cache=None, enable_trace=False, *args, **kwargs):
        # Plain decode traces are never used: every step is a verify (narrow steps run as 1-column verifies).
        eng = self._engine()
        if enable_trace:
            if not eng.captured:
                eng.capture_decode()
        elif not eng.prepared:
            eng.prepare_decode()

    def _ensure_ready(self, kv_cache=None):
        eng = self._engine()
        if eng.prefill_recorded and eng.captured:
            return eng
        logger.warning("Qwen36 MTP engine not warmed up (enable_model_warmup=False?); warming lazily")
        kv_cache = kv_cache if kv_cache is not None else self.model[0]._paged_kv_caches
        # plugin warmup order: allocate and compile everything before any trace is recorded
        if not eng.prefill_prepared:
            self.warmup_model_prefill(kv_cache, False)
        if not eng.prepared:
            eng.prepare_decode()
        if not eng.prefill_recorded:
            self.warmup_model_prefill(kv_cache, True)
        if not eng.captured:
            eng.capture_decode()
        return eng

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        if self._has_visual(kwargs, "pixel_values") or self._has_visual(kwargs, "pixel_values_videos"):
            raise ValueError("Qwen36MTPContractForCausalLM is text-only; image/video requests are not supported")
        eng = self._ensure_ready(kv_cache)
        N = tokens.shape[0]
        slots = [int(x) for x in kwargs.get("empty_slots") or range(N)]
        out = super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, **kwargs)
        for u in range(N):
            end = int(prompt_lens[u]) if prompt_lens is not None else tokens.shape[1]
            eng.finish_prefill(slots[u], end, self._row_ends_prompt(kwargs.get("mamba_state"), u))
        return out

    @staticmethod
    def _row_ends_prompt(mamba_state, row):
        """True when prefill row ``row`` ends its request's prefill (MambaStepState.intermediate_prefill_mask False, or
        no mask: a whole-prompt prefill). Only then does the step sample a token, and the row decode next."""
        mask = getattr(mamba_state, "intermediate_prefill_mask", None)
        return mask is None or not bool(mask[row])

    def _prefill_chunk_hook(self, row, row_tokens, page_table_row, mamba_state, kwargs):
        """The drafter's prompt KV over the row's chunks, and the feed row of its last token when the row ends its
        prompt (MTPBlockEngine.prefill_hook)."""
        slots = kwargs.get("empty_slots")
        slot = int(slots[row]) if slots else row
        return self._engine().prefill_hook(
            slot, row_tokens, self._table(page_table_row), self._row_ends_prompt(mamba_state, row)
        )

    def decode_forward(self, *args, **kwargs):
        from vllm_tt_plugin.spec_decode import VerifyOutput, check_spec_side_tensors

        def _read(name, pos):
            return kwargs[name] if name in kwargs else (args[pos] if pos < len(args) else None)

        tokens, start_pos, page_table = _read("tokens", 0), _read("start_pos", 1), _read("page_table", 2)
        remap = _read("slot_remap", 9)
        # page_tables_per_layer is not read: the attention layers form one group whose table is page_table, and the
        # GDN layers read their blocks from mamba_state.
        mamba_state = kwargs.pop("mamba_state", None)
        if mamba_state is None:
            raise ValueError(
                "Qwen36MTPContractForCausalLM keeps the GDN state in vLLM KV cache blocks and needs mamba_state"
            )
        if "spec_mode" not in kwargs:
            return self._decode_narrow(tokens, start_pos, page_table, kwargs, remap, mamba_state)
        num_valid_drafts = kwargs.pop("num_valid_drafts")
        accepted_counts = kwargs.pop("accepted_counts")
        spec_mode = kwargs.pop("spec_mode")
        check_spec_side_tensors(num_valid_drafts, accepted_counts, int(tokens.shape[0]), self._SPEC_K)
        eng = self._ensure_ready()
        res = eng.decode_forward(
            tokens,
            start_pos,
            num_valid_drafts=num_valid_drafts,
            accepted_counts=accepted_counts,
            page_table=self._table(page_table),
            spec_mode=spec_mode,
            slot_remap=remap,
            mamba_state=mamba_state,
        )
        logits = res.logits if spec_mode == "logits" else None
        return VerifyOutput(spec_mode=res.spec_mode, argmax_ids=res.argmax_ids, hidden=None, logits=logits)

    def _decode_narrow(self, tokens, start_pos, page_table, kwargs, remap, mamba_state):
        """Ordinary [B, 1] decode step (no drafts) run as a verify whose drafts are all padding: column 0's state goes
        to windows[:, 0], as the plugin expects of an ordinary decode, and the drafter gets the verify's feed rows.
        ``accepted_counts`` (a previous multi-token commit) is checked; the start state is start_blocks.
        Returns host float logits [B, 1, V]: the runner host-samples ``tt_out[rows, -1, :]``
        (model_runner.py:3276; async_decode.py:103 accepts a torch tensor as host output)."""
        nv, ac = kwargs.get("num_valid_drafts"), kwargs.get("accepted_counts")
        B = int(tokens.shape[0])
        if nv is not None and bool(torch.as_tensor(nv).any()):
            raise ValueError("narrow decode step carries drafts")
        if ac is not None:
            from vllm_tt_plugin.spec_decode import check_spec_side_tensors

            check_spec_side_tensors(torch.zeros(B, dtype=torch.int32), ac, B, self._SPEC_K, call="decode")
        acc = torch.ones(B, dtype=torch.int32) if ac is None else torch.as_tensor(ac).reshape(B).to(torch.int32)
        K = self._SPEC_K
        tok = torch.full((B, K + 1), -1, dtype=torch.int32)
        tok[:, 0] = torch.as_tensor(tokens).reshape(B).to(torch.int32)
        pos = torch.full((B, K + 1), -1, dtype=torch.int32)
        pos[:, 0] = torch.as_tensor(start_pos).reshape(B).to(torch.int32)
        eng = self._ensure_ready()
        res = eng.decode_forward(
            tok,
            pos,
            num_valid_drafts=torch.zeros(B, dtype=torch.int32),
            accepted_counts=acc,
            page_table=self._table(page_table),
            spec_mode="logits",
            slot_remap=remap,
            mamba_state=mamba_state,
        )
        return res.logits[:, :1].float()

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput, check_spec_side_tensors

        B = int(committed_tokens.shape[0])
        check_spec_side_tensors(torch.zeros(B, dtype=torch.int32), accepted_counts, B, self._SPEC_K, call="propose")
        res = self._engine().propose_draft_tokens(
            num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=hidden
        )
        return DraftOutput(draft_token_ids=res.draft_token_ids, num_valid=res.num_valid)

    def release_request(self, slot):
        self._engine().release(int(slot))

    def note_state_slots_moved(self, moves):
        """The drafter's per-slot state moved with slot_remap in decode_forward; nothing to do."""

    def release_persistent_capture(self):
        logger.info("[spec] release_persistent_capture: start")
        eng = getattr(self.model[0], "_spec_engine", None)
        if eng is not None:
            eng.shutdown()
        parent = getattr(super(), "release_persistent_capture", None)
        if parent is not None:
            parent()
        logger.info("[spec] release_persistent_capture: done")
