# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger

import ttnn
from models.experimental.janus_pro.tt.load_checkpoints import convert_vision_hf_to_meta
from models.tt_transformers.tt.common import Mode, get_out_subblock_w
from models.tt_transformers.tt.load_checkpoints import convert_meta_to_hf
from models.tt_transformers.tt.model_config import ModelArgs as TTModelArgs


class ModelArgs(TTModelArgs):
    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,
    ):
        # Resolve HF_MODEL to a local snapshot path before super().__init__() so that
        # all HF calls (AutoConfig, tokenizer, weights) skip the refs/main lookup,
        # which is absent on some CI machines.
        hf_model = os.environ.get("HF_MODEL", "")
        if hf_model and not os.path.isabs(hf_model):
            snapshot = ModelArgs._resolve_hf_snapshot(hf_model)
            if snapshot:
                logger.info(f"[JanusPro] Resolved HF model '{hf_model}' to snapshot: {snapshot}")
                os.environ["HF_MODEL"] = str(snapshot)

        super().__init__(
            mesh_device,
            instruct=instruct,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            cache_hf=cache_hf,
        )

    def get_attn_qkv_program_config(self, mode, seq_len=1, prefetcher=None):
        # Prefill QKV: defer to the base config for all devices. The former P150-specific
        # override only reached the MatmulMultiCoreReuseMultiCast branch at seq_len<=128,
        # where its per_core_M collapsed to 1 -> byte-identical to the base else-branch
        # (the P150 cap of 7 vs base 8 only matters at seq_len>=2048, which minimal_matmul
        # excludes). So it was a no-op and has been removed.

        # Decode QKV is a DRAM-sharded matmul whose core count the base picks from k (dim)
        # alone -> 32 cores -> per_core_N = ceil(qkv_size / (32*32)). Janus's wide MHA qkv
        # (12288) makes per_core_N = 12, whose CBs overflow P150's L1. Pick the grid from
        # both k AND n so more cores (64) share the wide output -> per_core_N = 6. The decode
        # output mem config is a generic L1_WIDTH_SHARDED, so it adapts to this grid.
        if mode == Mode.DECODE and prefetcher is None:
            n = self.qkv_size // self.num_devices
            return self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.dim,
                n=n,
                num_cores=self.dram_shard_core_grid_for_k_and_n(self.dim, n).num_cores,
            )
        return super().get_attn_qkv_program_config(mode, seq_len, prefetcher)

    # in0_block_w per projection, each the in-model minimum of a sweep over the divisors of that
    # shape's K in tiles. An isolated bench of the same shapes disagrees on two of the three, so
    # these must be retuned in the model, never standalone. Sweep data is in PERF.md.
    VISION_QKV_IN0_BLOCK_W = 4
    VISION_WO_IN0_BLOCK_W = 8
    VISION_C_PROJ_IN0_BLOCK_W = 16
    # 4 rather than a swept optimum: it must equal the width in tiles of the shard ln_2 hands over,
    # which is what lets c_fc read its in0 in place instead of behind an unshard.
    VISION_C_FC_IN0_BLOCK_W = 4
    # Unlike the four above, this one is not swept -- it divides the patch's 24 K-tiles and the
    # projection runs once per image, so the sweep would cost more than it could return.
    VISION_PATCH_EMBED_IN0_BLOCK_W = 4

    # Encoder blocks from this index on carry a bfloat8_b residual; earlier ones stay bfloat16.
    # The residual is the one tensor summed across every layer, so unlike every other bfloat8_b
    # tensor in the tower it has more than one consumer and its quantization error compounds.
    # That makes the format a boundary rather than a flag, and the boundary is where
    # `test_vision_transformer`'s 0.99 runs out: 12 costs 2.1e-3 of its PCC, 18 costs 7.2e-3, 19
    # fails at 0.9898 and all 24 fail at 0.9765.
    #
    # It has to be a suffix. Early blocks cost roughly six times what late ones do, their error
    # propagating through every block that follows, and a bfloat16 add cannot restore bits the
    # running sum has already dropped -- so alternating blocks is strictly worse than a suffix of
    # the same length: every second block fails the gate at 0.9774 while these 12 pass at 0.9967.
    VISION_BFP8_RESIDUAL_FROM_LAYER = 12

    # Below this the explicit config was not measured against ttnn's derivation.
    _VISION_MIN_CONFIGURED_CORES = 24
    # Below this the sharded layer-norm factory loses to the interleaved one.
    _VISION_NORM_MIN_CORES = 24
    # The in0 circular buffer holds 2 * per_core_M * in0_block_w tiles, so a large output block
    # with a wide K overruns L1. This bounds the output block only.
    _VISION_MAX_OUT_BLOCK_TILES = 64

    def _vision_body_matmul_config(self, batch_size, seq_len, k, n, in0_block_w, fused_activation=None):
        """2D mcast config for one vision body projection, or None to leave the derivation to ttnn.

        `k` and `n` are per-device: the weights shard across the mesh, so the grid is sized from the
        shape the device actually sees. `None` also means the caller must not ask for a sharded
        output, since only this config is known-good for one.
        """
        TILE = ttnn.TILE_SIZE
        if seq_len % TILE or n % TILE or k % TILE:
            return None
        m_tiles, n_tiles, k_tiles = batch_size * seq_len // TILE, n // TILE, k // TILE
        if k_tiles % in0_block_w:
            return None
        # Exact division on both axes, so num_blocks_y == grid_y and num_blocks_x == grid_x.
        grid_y, grid_x = self.find_prefill_grid(m_tiles, n_tiles)
        per_core_M, per_core_N = m_tiles // grid_y, n_tiles // grid_x
        if grid_x * grid_y < self._VISION_MIN_CONFIGURED_CORES or (
            per_core_M * per_core_N > self._VISION_MAX_OUT_BLOCK_TILES
        ):
            return None
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=get_out_subblock_w(per_core_N, 1),
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=fused_activation,
        )

    def vision_patch_embed_program_config(self, batch_size, seq_len):
        """The patch projection, or None to leave the derivation to ttnn.

        Its grid and per-core block come out equal to `vision_norm_shard_configs` for the same
        shape, so the output is already the shard `ln_1` reads -- no reshard sits between them.
        K is the folded patch: one kernel window across every input channel.
        """
        return self._vision_body_matmul_config(
            batch_size,
            seq_len,
            self.vision_patch_size**2 * self.vision_in_channels,
            self.vision_dim,
            self.VISION_PATCH_EMBED_IN0_BLOCK_W,
        )

    def vision_qkv_program_config(self, batch_size, seq_len):
        """wqkv shards on its N axis, so N is per-device while K stays whole."""
        return self._vision_body_matmul_config(
            batch_size,
            seq_len,
            self.vision_dim,
            3 * self.vision_dim // self.num_devices,
            self.VISION_QKV_IN0_BLOCK_W,
        )

    def vision_wo_program_config(self, batch_size, seq_len):
        """wo shards on its K axis, so K is per-device while N stays whole."""
        return self._vision_body_matmul_config(
            batch_size,
            seq_len,
            self.vision_dim // self.num_devices,
            self.vision_dim,
            self.VISION_WO_IN0_BLOCK_W,
        )

    def vision_c_fc_program_config(self, batch_size, seq_len, fused_activation=None):
        return self._vision_body_matmul_config(
            batch_size,
            seq_len,
            self.vision_dim,
            self.vision_hidden_dim // self.num_devices,
            self.VISION_C_FC_IN0_BLOCK_W,
            fused_activation=fused_activation,
        )

    def vision_c_proj_program_config(self, batch_size, seq_len):
        """c_proj shards on its K axis, so K is per-device while N stays whole."""
        return self._vision_body_matmul_config(
            batch_size,
            seq_len,
            self.vision_hidden_dim // self.num_devices,
            self.vision_dim,
            self.VISION_C_PROJ_IN0_BLOCK_W,
        )

    def vision_aligner_program_config(self, batch_size, seq_len, k, n, fused_activation=None):
        """2D config for one aligner projection, or None to leave the derivation to ttnn.

        The aligner weights are replicated rather than sharded, so `k` and `n` are whole. A config
        is what carries the activation: `ttnn.linear`'s `activation=` is rejected outright once in0
        arrives sharded (`matmul.cpp:235`), which it does behind the last block's residual add.
        """
        return self._vision_body_matmul_config(
            batch_size, seq_len, k, n, self.VISION_C_FC_IN0_BLOCK_W, fused_activation=fused_activation
        )

    def vision_norm_shard_configs(self, seq_len, dim):
        """Memory and program config for a block-sharded vision layer norm, or `(None, None)`.

        `(None, None)` leaves the norm interleaved. The shard's width in tiles is what
        `VISION_C_FC_IN0_BLOCK_W` must equal, since that is what lets c_fc read this output in
        place; the same holds for qkv reading ln_1's.
        """
        TILE = ttnn.TILE_SIZE
        if seq_len % TILE or dim % TILE:
            return None, None
        # Splitting height is free; splitting width makes the row reduction cross cores, so
        # height goes first -- which is the axis order find_prefill_grid already takes.
        grid_y, grid_x = self.find_prefill_grid(seq_len // TILE, dim // TILE)
        if grid_x * grid_y < self._VISION_NORM_MIN_CORES:
            return None, None

        block_h, block_w = seq_len // TILE // grid_y, dim // TILE // grid_x
        memory_config = ttnn.create_sharded_memory_config(
            shape=(block_h * TILE, block_w * TILE),
            core_grid=ttnn.CoreGrid(x=grid_x, y=grid_y),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid_x, grid_y],
            subblock_w=min(block_w, 4),  # the kernel holds subblock_w tiles in DST
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )
        return memory_config, program_config

    def _dram_decode_num_cores(self, k):
        # Largest core count (<=64) that evenly shards k tiles for a DRAM-sharded decode matmul.
        # The base grid divides by both k and n; for Janus's 11008 hidden_dim that collapses to
        # 8 cores (344 = 8*43, 43 prime), making per_core_N ~43 tiles whose CBs overflow P150 L1.
        # Sharding by k alone gives many more cores -> small per_core_N (n just pads to the grid).
        k_tiles = k // ttnn.TILE_SIZE
        for cores in range(min(64, k_tiles), 0, -1):
            if k_tiles % cores == 0:
                return cores
        return 1

    def get_mlp_ff1_3_prg_config(self, mode, seq_len=1, prefetcher=None):
        # Decode MLP w1/w3 (dim -> hidden_dim); see _dram_decode_num_cores for why P150 needs
        # more cores than the base k/n grid for Janus's wide hidden_dim.
        if mode == Mode.DECODE and prefetcher is None and not self.is_galaxy:
            return self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.dim,
                n=self.hidden_dim // self.cluster_shape[1],
                num_cores=self._dram_decode_num_cores(self.dim),
            )
        return super().get_mlp_ff1_3_prg_config(mode, seq_len, prefetcher)

    @staticmethod
    def _resolve_hf_snapshot(hf_model_name):
        hf_cache = os.path.normpath(
            os.environ.get("HF_HUB_CACHE")
            or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        )
        model_slug = "models--" + hf_model_name.replace("/", "--")
        snapshots_dir = os.path.normpath(os.path.join(hf_cache, model_slug, "snapshots"))
        # Prevent path traversal: ensure the resolved path stays within hf_cache.
        if not snapshots_dir.startswith(hf_cache + os.sep):
            return None
        if not os.path.isdir(snapshots_dir):
            return None
        snaps = [
            os.path.join(snapshots_dir, s)
            for s in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, s))
        ]
        return max(snaps, key=os.path.getmtime) if snaps else None

    def _set_hf_params(self, checkpoint_dir):
        # The base dummy path reads the HF config from LOCAL_HF_PARAMS[model_name],
        # which Janus doesn't register (and model_name is a snapshot hash here).
        # Force the config to load from CKPT_DIR (the real config.json) while keeping
        # dummy_weights semantics for the weight-loading paths.
        saved_dummy_weights = self.dummy_weights
        self.dummy_weights = False
        try:
            super()._set_hf_params(checkpoint_dir)
        finally:
            self.dummy_weights = saved_dummy_weights

    def _set_model_specific_params(self):
        # LLaMA-style text decoder: RMSNorm without a unit offset.
        self.rms_norm_add_unit_offset = False

    def create_tokenizer(self):
        # Janus ships only a fast tokenizer (tokenizer.json) and no sentencepiece
        # tokenizer.model, so force use_fast to avoid the slow LlamaTokenizer path
        # that requires a vocab_file.
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.CKPT_DIR,
            use_fast=True,
            local_files_only=os.getenv("CI") == "true",
            trust_remote_code=self.trust_remote_code_hf,
        )

    def _set_vision_params(self, config):
        vision_config = config.get("vision_config", config)

        # JanusVisionEmbeddings (patch_embedding, position_embedding)
        self.vision_chunk_size = vision_config.get("image_size", 384)
        self.vision_dim = vision_config.get("hidden_size", 1024)
        self.vision_patch_size = vision_config.get("patch_size", 16)
        self.vision_in_channels = vision_config.get("num_channels", 3)
        self.mm_tokens_per_image = vision_config.get("num_image_tokens", 576)

        # JanusVisionEncoder (ModuleList of encoder layers)
        self.vision_n_layers = vision_config.get("num_hidden_layers", 24)

        # JanusVisionEncoderLayer (layer_norm1/2) and JanusVisionModel (post_layernorm)
        self.vision_layer_norm_eps = vision_config.get("layer_norm_eps", 1e-6)

        # JanusVisionAttention (q/k/v_proj, q/k_norm, projection_layer)
        self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
        self.vision_attention_bias = vision_config.get("attention_bias", True)
        self.vision_use_qk_norm = vision_config.get("use_qk_norm", False)
        self.vision_dropout = vision_config.get("attention_dropout", 0.0)
        self.vision_projection_dropout = vision_config.get("projection_dropout", 0.0)

        # JanusVisionMLP (fc1, activation_fn, fc2; activation_fn also used by JanusVisionAlignerMLP)
        mlp_ratio = vision_config.get("mlp_ratio", 4.0)
        self.vision_mlp_ratio = mlp_ratio
        self.vision_hidden_dim = int(self.vision_dim * mlp_ratio)
        act_layer = str(vision_config.get("hidden_act", "gelu")).lower()
        self.vision_act_layer = {
            "gelu": ttnn.UnaryOpType.GELU,
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
        }.get(act_layer, ttnn.UnaryOpType.GELU)
        self.vision_hidden_dropout = vision_config.get("hidden_dropout_rate", 0.0)

        # JanusVisionAlignerMLP (fc1, hidden_layers; vision-to-text projection)
        self.vision_projection_dim = vision_config.get("projection_dim", 2048)
        self.vision_aligner_depth = vision_config.get("depth", 2)

        # Not in Janus HF config; placeholders for base ModelArgs.__repr__
        self.vision_max_num_chunks = vision_config.get("vision_max_num_chunks", 4)
        self.vision_num_cross_attention_layers = vision_config.get("vision_num_cross_attention_layers", 0)

    def get_hf_model_cls(self):
        from transformers import JanusForConditionalGeneration

        return JanusForConditionalGeneration

    def _janus_dummy_hf_model(self):
        """Build JanusForConditionalGeneration from HF config only (random init).

        Avoids loading the multi-GB checkpoint. The text decoder is shrunk to
        self.n_layers since the vision tests don't exercise the language model.
        The model is cached so load_state_dict() and the reference_* helpers
        share identical random weights (otherwise PCC checks would compare two
        different random initializations).
        """
        if getattr(self, "_dummy_hf_model", None) is not None:
            return self._dummy_hf_model

        import gc

        import torch
        from transformers import AutoConfig, JanusForConditionalGeneration

        logger.info("[JanusPro] Building HF dummy model from config (dummy_weights=True)")

        config = AutoConfig.from_pretrained(self.CKPT_DIR, trust_remote_code=self.trust_remote_code_hf)
        # Keep the language model tiny; vision tower keeps its real depth (vision_n_layers).
        if getattr(config, "text_config", None) is not None:
            config.text_config.num_hidden_layers = self.n_layers
            config.text_config.num_layers = self.n_layers

        build_from_config = getattr(
            JanusForConditionalGeneration, "from_config", JanusForConditionalGeneration._from_config
        )
        try:
            model = build_from_config(config, torch_dtype=torch.bfloat16)
        except TypeError:
            model = build_from_config(config)

        # Reference runs on fp32 inputs; force fp32 to avoid the mixed-dtype CPU error.
        model = model.float()
        gc.collect()
        self._dummy_hf_model = model
        return model

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        if is_vision:
            prefix = "model.vision_model.encoder."
        else:
            prefix = ""

        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        text_module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
            "": "",
        }
        vision_module_map = {
            "MLP": "mlp.",
            "Attention": "attn.",
            "TransformerBlock": "",
            "": "",
        }
        module_map = vision_module_map if is_vision else text_module_map

        return prefix + layer_prefix + module_map[module_name]

    def load_state_dict(self):
        # Weight-seeding path. In dummy mode, seed from the same cached random-init model the
        # reference paths use, so the TT weights and the golden reference share identical weights;
        # a second random init would make PCC meaningless. For real weights, take them at native
        # checkpoint dtype -- do NOT route through reference_vision_transformer() below, which
        # upcasts to float32 for the golden reference only. Mirrors models/demos/multimodal/gemma3.
        if self.dummy_weights:
            model = self._janus_dummy_hf_model()
        else:
            model = super().reference_vision_transformer(wrap=False)
        return convert_vision_hf_to_meta(model.state_dict(), self.head_dim)

    def reference_vision_transformer(self, wrap=True, load_checkpoint=False):
        # Golden-reference path: force float32 so the reference matches float32 test inputs (a stable
        # PCC baseline). In dummy mode, reuse the shared random-init model -- the same instance
        # load_state_dict seeds the TT weights from -- so reference and TT weights match. For real
        # weights, float an ISOLATED instance, never self.cached_hf_model (which seeds the TT weights
        # via load_state_dict), so the upcast cannot leak into the weight path regardless of call
        # order. Mirrors models/demos/multimodal/gemma3.tt.model_config.reference_vision_transformer.
        if self.dummy_weights and not load_checkpoint:
            model = self._janus_dummy_hf_model()
        else:
            model = self.get_hf_model_cls().from_pretrained(
                self.CKPT_DIR, torch_dtype="auto", local_files_only=os.getenv("CI") == "true"
            )
            model = model.float()
        if wrap:
            from models.tt_transformers.tt.model_config import HfModelWrapper

            return HfModelWrapper(model, self.head_dim, use_hf_rope=self.use_hf_rope)
        return model

    def reference_siglip_patch_embed(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.embeddings.patch_embedding

    def reference_vision_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.embeddings

    def reference_vision_layernorm(self, layer_name="layer_norm1"):
        model = self.reference_vision_model()
        if layer_name == "layer_norm1":
            return model.encoder.layers[0].layer_norm1
        elif layer_name == "layer_norm2":
            return model.encoder.layers[0].layer_norm2
        return model.post_layernorm

    def reference_vision_attention(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.encoder.layers[0].self_attn

    def reference_vision_mlp(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.encoder.layers[0].mlp

    def reference_vision_encoder_block(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.encoder.layers[0]

    def reference_vision_encoder(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.encoder

    def reference_vision_model(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model

    def reference_vision_aligner(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.aligner

    def reference_transformer(self, wrap=True, load_checkpoint=False):
        if self.dummy_weights and not load_checkpoint:
            # Base dummy path reads the HF config from LOCAL_HF_PARAMS[model_name],
            # which Janus doesn't register. Reuse the shared random-init model and
            # mirror the base helper: expose the text decoder (language_model) as
            # model.model and truncate to n_layers.
            model = self._janus_dummy_hf_model()
            if hasattr(model, "language_model"):
                model.model = model.language_model
            elif hasattr(model.model, "language_model"):
                model.model = model.model.language_model
            model.model.layers = model.model.layers[: self.n_layers]
            if wrap:
                from models.tt_transformers.tt.model_config import HfModelWrapper

                return HfModelWrapper(model, self.head_dim, config=self.hf_config, use_hf_rope=self.use_hf_rope)
            return model
        return super().reference_transformer(wrap=wrap, load_checkpoint=load_checkpoint)

    def reference_language_model(self):
        model = self.reference_transformer(wrap=False)
        return model.model.float()

    def reference_rms_norm_text(self):
        # Final RMSNorm of the text decoder (language_model.norm).
        model = self.reference_transformer(wrap=False)
        layer = model.model.norm
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer
