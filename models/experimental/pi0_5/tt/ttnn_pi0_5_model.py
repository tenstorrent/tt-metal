# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 model (TTNN, inference).

Same denoising loop as PI0 but:
  - no state token in the suffix,
  - adarms_cond is passed into forward_expert,
  - no state-token slice on the expert output.
"""

from pathlib import Path
from typing import List, Optional, Union

import torch
import ttnn

from models.experimental.pi0_5.common.configs import (
    DenoiseConfig,
    PaliGemmaConfig,
    PrefixConfig,
    SuffixConfig,
)
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader as PI0WeightLoader
from models.experimental.pi0_5.tt.ttnn_common import denoise_loop_fp32
from models.experimental.pi0_5.tt.ttnn_prefix import PrefixEmbeddingTTNN
from models.experimental.pi0_5.reference.torch_siglip_hf import use_hf_siglip


def use_upstream_masks() -> bool:
    """`PI0_UPSTREAM_MASKS=1` (or implicitly when PI0_SIGLIP_HF=1) -> plumb
    cumsum-based position_ids and a logical-pad attention mask through VLM and
    expert. Decoupling from PI0_SIGLIP_HF lets us A/B test the device-side
    ttnn_siglip with HF-compatible mask handling, isolating the SigLIP
    semantic gap from the mask/RoPE plumbing gap.
    """
    import os as _os

    return use_hf_siglip() or _os.environ.get("PI0_UPSTREAM_MASKS", "").strip().lower() in ("1", "true", "yes", "on")


def _precompute_rope_table_torch(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Host-side RoPE table in TTNN split-half format.

    Mirrors precompute_freqs_cis_meta_format in ttnn_gemma.py — values are
    repeated as [c0, c1, ..., c_{n/2-1}, c0, c1, ..., c_{n/2-1}] so that
    ttnn.experimental.rotary_embedding's split-half rotation pairs x[i] with
    x[i+dim/2] correctly. Returns (cos, sin) each shaped [max_seq_len, head_dim].
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    fo = torch.outer(t, freqs)
    cos = torch.cos(fo)
    sin = torch.sin(fo)
    return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


_MASK_VAL = -1e4  # finite large-negative; exp(-1e4)=0 in bf16, no NaN

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.tt.ttnn_suffix import (
    Pi0_5SuffixEmbeddingTTNN,
    convert_pi0_5_suffix_weights_to_ttnn,
)
from models.experimental.pi0_5.tt.ttnn_paligemma import Pi0_5PaliGemmaBackboneTTNN


class Pi0_5ModelTTNN:
    """TTNN PI0.5 model (inference)."""

    def __init__(
        self,
        config: Pi0_5ModelConfig,
        weight_loader: PI0WeightLoader,
        device: "ttnn.Device",
    ):
        assert config.pi05, "Pi0_5ModelTTNN requires config.pi05=True"
        self.config = config
        self.weight_loader = weight_loader
        self.device = device

        self.denoise_config = DenoiseConfig(
            num_steps=config.num_denoising_steps,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        )

        pad_steps = ((self.denoise_config.num_steps + 31) // 32) * 32
        self.timestep_indices = ttnn.arange(0, pad_steps, 1, device=self.device, dtype=ttnn.bfloat16)

        # Initial noise tensor; resampled fresh on each sample_actions call to
        # match lerobot/openpi reference behavior (see sample_actions below).
        # Allocated once and reused as a destination buffer.
        #
        # HOST-PAD: action_horizon (e.g. 50) is not tile-aligned. We pad to
        # the next tile boundary (64) on host so the device tensor has
        # logical=physical=tile-aligned from the start — eliminates a per-call
        # ttnn.pad inside sample_actions. The phantom rows are zero-valued and
        # masked out of SDPA via the prebuilt attention mask.
        self._action_horizon_padded = ((config.action_horizon + 31) // 32) * 32
        x_t_torch = torch.zeros(1, self._action_horizon_padded, config.action_dim, dtype=torch.float32)
        x_t_torch[:, : config.action_horizon, :] = torch.randn(1, config.action_horizon, config.action_dim)
        self.x_t_ttnn = ttnn.from_torch(
            x_t_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        self._init_components()
        self._precompute_bs1_timestep_tensors()
        self._precompute_bs1_adarms_cond()
        self._precompute_bs1_modulations()
        # Reapplied TIER B (commit 3d597a3b8e6) — see _build_sdpa_phantom_mask
        # docstring for the finite-mask hybrid rationale.
        self._sdpa_attn_mask: Optional["ttnn.Tensor"] = None
        self._sdpa_mask_kv_len: int = 0
        # Upstream-compat artifact cache (mask + RoPE tables for the
        # PI0_UPSTREAM_MASKS=1 / PI0_SIGLIP_HF=1 path). Two usage modes:
        # (1) auto: sample_actions builds + caches on first call (keyed by
        #     img_mask_tuple, lang_real_count, prefix_len) — costs a
        #     device sync on subsequent calls to verify the key still matches.
        # (2) explicit pre-stage via prepare_upstream_artifacts() — caller
        #     takes responsibility for the key match. sample_actions uses
        #     the cached artifacts WITHOUT ANY coercion/sync, so the call
        #     stays trace-capture-safe.
        self._cached_upstream_artifacts = None  # type: Optional[dict]
        self._cached_upstream_key = None  # type: Optional[tuple]
        self._upstream_artifacts_explicit = False

    def _precompute_bs1_timestep_tensors(self) -> None:
        num_steps = self.denoise_config.num_steps
        pad_steps = ((num_steps + 31) // 32) * 32
        idx = ttnn.to_layout(self.timestep_indices, ttnn.TILE_LAYOUT)
        vals = ttnn.multiply(idx, -1.0 / num_steps)
        ttnn.deallocate(idx)
        row = ttnn.add(vals, 1.0)
        ttnn.deallocate(vals)
        self._timesteps_row_ttnn = ttnn.reshape(row, (1, pad_steps))

        self._timestep_per_step_bs1: List["ttnn.Tensor"] = []
        for i in range(num_steps):
            t_i = ttnn.slice(self._timesteps_row_ttnn, [0, i], [1, i + 1])
            self._timestep_per_step_bs1.append(ttnn.reshape(t_i, (1,)))

    def _precompute_bs1_adarms_cond(self) -> None:
        """
        adarms_cond is unused on the BS=1 fast path: when precomputed
        block/final modulations are supplied (see _precompute_bs1_modulations),
        every consumer of adarms_cond inside the expert block + final norm is
        bypassed. Keep the per-step list as None sentinels so the fast-path
        loop signature in sample_actions stays unchanged — this removes the
        device-side sincos + 2 linears + silu chain (and the 10×
        TilizeWithValPadding X=512 FLOAT32 + 10× X=1 BFLOAT16 it emitted at
        init).
        """
        num_steps = self.denoise_config.num_steps
        self._adarms_cond_per_step_bs1: List[Optional["ttnn.Tensor"]] = [None] * num_steps

    def _precompute_bs1_modulations(self) -> None:
        """
        OPTIMIZATION (TIER A, host variant): adarms_cond is deterministic per
        step, so the per-block fused modulation Dense (W -> 6*W) and the
        final-norm Dense (W -> 3*W) produce constant outputs. We now compute
        everything on host (torch) and upload the results pre-tilized to DRAM
        in a single ttnn.from_torch per tensor.

        Saves at init (cold-start) vs. the old device-side path:
          - 200 device matmuls (~9.8 ms): 180 block mod-Dense (W→6W) +
            10 final-norm mod-Dense (W→3W) + 10 adarms_cond time-MLP linears
            no longer happen on device.
          - ~390 device-side TilizeWithValPadding ops (~4.5 ms): the 1-row
            (1, W) tensors that fed the device matmuls and the post-add
            (1+scale) tensors are now pre-tilized on host (padded to (1,1,32,W))
            before from_torch.
          - All scale_plus_one adds are done in torch.

        Steady-state inference perf is unchanged — same precomputed tensors,
        same DRAM placement. Only cold-start (init) gets faster.
        """
        import torch
        import ttnn as _ttnn
        from models.experimental.pi0_5.common.configs import SuffixConfig
        from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding

        num_steps = self.denoise_config.num_steps
        cfg = self.config
        W = cfg.expert_config.width
        depth = cfg.expert_config.depth
        ae_weights = self.weight_loader.categorized_weights["action_expert"]
        pi0_weights = self.weight_loader.get_pi0_projections()

        # --- adarms_cond per step (host) ---
        suffix_cfg = SuffixConfig(
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            expert_width=W,
            pi05=True,
        )
        torch_suffix = Pi0_5SuffixEmbedding(suffix_cfg, pi0_weights)
        timesteps = torch.tensor([1.0 - i / num_steps for i in range(num_steps)], dtype=torch.bfloat16)
        # bf16 here matches the device-side HiFi2 bf16 matmul output that
        # this precompute replaces. Tried fp32 host compute: +0.0007 e2e PCC
        # (within noise), no win — the bf16 quantization happens at upload
        # regardless of host dtype, so the host precision doesn't matter.
        adarms_cond_per_step: List[torch.Tensor] = []
        for i in range(num_steps):
            c = torch_suffix.embed_timestep_adarms(timesteps[i : i + 1]).to(torch.bfloat16)
            adarms_cond_per_step.append(c)

        # --- host helper: upload 1×W as logical (1, 1, W) in TILE layout ---
        # Matches the original device-side _split_modulation_6 output shape
        # exactly (B=1, 1, W). ttnn.from_torch(layout=TILE) does the tile
        # padding on host, so no device TilizeWithValPadding op is emitted.
        #
        # PI0_ADARMS_MODS_L1=1 places the per-layer per-step mod tensors in L1
        # rather than DRAM. Perf-analyze on the v7 (51.952 ms) trace flagged
        # 360 of the 1218 BinaryNg ops as L1+DRAM→L1 MUL at shape (32, 1024)
        # — these are the adaRMS gated multiplies `gated_attn = mul(out, ga)`
        # / `gated_mlp = mul(out, gf)` (ttnn_gemma.py:1512, 1536). The DRAM
        # operand is the gate tensor read from this upload path. Moving
        # mods to L1 saves the DRAM read on every gated MUL (0.774 ms bucket).
        # Total mod budget: 10 steps × 18 layers × 6 tensors × ~2 KB = ~2.2 MB
        # of L1, opt-in to keep room for trace-persistent KV cache.
        # Per PERF_PLAYBOOKS/01 §1 + 06 §3 (layout matching).
        import os as _os

        _mods_l1 = _os.environ.get("PI0_ADARMS_MODS_L1", "").lower() in ("1", "true", "yes", "on")
        _mods_mc = _ttnn.L1_MEMORY_CONFIG if _mods_l1 else _ttnn.DRAM_MEMORY_CONFIG

        def host_pad_tile_upload(t: torch.Tensor) -> "_ttnn.Tensor":
            assert t.dim() == 2 and t.shape[0] == 1, f"expected (1, W), got {tuple(t.shape)}"
            t3d = t.unsqueeze(1).contiguous()  # (1, 1, W) — matches _split_modulation_6
            return _ttnn.from_torch(
                t3d,
                dtype=_ttnn.bfloat16,
                layout=_ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=_mods_mc,
            )

        # --- Per-block fused mod weights (3W rows per LN × 2 LNs = 6W) ---
        # [step][layer] -> (sa1, ta, ga, sf1, tf, gf)
        self._block_mods_per_step: List[List[tuple]] = []

        # Cache fused (mod_weight, mod_bias) per layer in torch — same for all steps.
        per_layer_fused: List[tuple] = []
        for layer_idx in range(depth):
            prefix = f"model.layers.{layer_idx}."
            w_pre_attn = ae_weights[f"{prefix}input_layernorm.dense.weight"]  # (3W, W)
            w_pre_ffw = ae_weights[f"{prefix}post_attention_layernorm.dense.weight"]  # (3W, W)
            fused_w = torch.cat([w_pre_attn, w_pre_ffw], dim=0).contiguous().to(torch.bfloat16)  # (6W, W)
            b_attn_key = f"{prefix}input_layernorm.dense.bias"
            b_ffw_key = f"{prefix}post_attention_layernorm.dense.bias"
            if b_attn_key in ae_weights:
                fused_b = (
                    torch.cat([ae_weights[b_attn_key], ae_weights[b_ffw_key]], dim=0).contiguous().to(torch.bfloat16)
                )  # (6W,)
            else:
                fused_b = None
            per_layer_fused.append((fused_w, fused_b))

        for step_idx in range(num_steps):
            cond = adarms_cond_per_step[step_idx]  # (1, W) torch float32
            per_layer_step: List[tuple] = []
            for layer_idx in range(depth):
                fused_w, fused_b = per_layer_fused[layer_idx]
                mod = torch.nn.functional.linear(cond, fused_w, fused_b)  # (1, 6W)
                # Split into 6 × (1, W)
                scale_a = mod[:, 0 * W : 1 * W]
                shift_a = mod[:, 1 * W : 2 * W]
                gate_a = mod[:, 2 * W : 3 * W]
                scale_f = mod[:, 3 * W : 4 * W]
                shift_f = mod[:, 4 * W : 5 * W]
                gate_f = mod[:, 5 * W : 6 * W]
                # Pre-add 1 to scale tensors so rms_norm uses them directly.
                sa1 = scale_a + 1.0
                sf1 = scale_f + 1.0
                # Upload each as host-padded TILE → DRAM (no device tilize).
                per_layer_step.append(
                    (
                        host_pad_tile_upload(sa1),
                        host_pad_tile_upload(shift_a),
                        host_pad_tile_upload(gate_a),
                        host_pad_tile_upload(sf1),
                        host_pad_tile_upload(shift_f),
                        host_pad_tile_upload(gate_f),
                    )
                )
            self._block_mods_per_step.append(per_layer_step)

        # --- Final norm (no gate, 3W output) ---
        final_w = ae_weights["model.norm.dense.weight"].contiguous().to(torch.bfloat16)  # (3W, W)
        final_b = ae_weights.get("model.norm.dense.bias")
        if final_b is not None:
            final_b = final_b.contiguous().to(torch.bfloat16)

        self._final_mod_per_step: List[tuple] = []
        for step_idx in range(num_steps):
            cond = adarms_cond_per_step[step_idx]
            mod = torch.nn.functional.linear(cond, final_w, final_b)  # (1, 3W)
            scale = mod[:, 0 * W : 1 * W]
            shift = mod[:, 1 * W : 2 * W]
            # gate discarded for no-gate final norm.
            scale1 = scale + 1.0
            self._final_mod_per_step.append(
                (
                    host_pad_tile_upload(scale1),
                    host_pad_tile_upload(shift),
                )
            )

    @staticmethod
    def _coerce_upstream_masks(img_masks, lang_masks):
        """Normalize img_masks (list of torch/ttnn/scalar) and lang_masks
        (torch/ttnn/numpy) into torch bool tensors suitable for the host
        artifact builder. Shared by `sample_actions` and
        `prepare_upstream_artifacts` so both follow the exact same coercion
        rules and produce the same cache key.
        """

        def _img(m):
            if isinstance(m, torch.Tensor):
                return m
            if isinstance(m, ttnn.Tensor):
                return ttnn.to_torch(m).to(torch.bool).view(-1)
            return torch.tensor([bool(m)])

        ims = [_img(m) for m in img_masks]
        if isinstance(lang_masks, ttnn.Tensor):
            lm = ttnn.to_torch(lang_masks).to(torch.bool)
        elif isinstance(lang_masks, torch.Tensor):
            lm = lang_masks.to(torch.bool)
        else:
            lm = torch.from_numpy(lang_masks).to(torch.bool)
        if lm.dim() == 1:
            lm = lm.unsqueeze(0)
        return ims, lm

    @staticmethod
    def _upstream_cache_key(img_masks, lang_masks, prefix_len: int):
        """Stable hashable key over the inputs that determine the upstream
        mask + RoPE artifacts. img_masks reduce to per-image True/False;
        lang_masks reduce to a single "real token count". Anything else
        (per-image embedding values, lang_token IDs themselves) does NOT
        affect the artifacts, so we ignore it.
        """
        img_present = tuple(bool(m.any().item()) for m in img_masks)
        lang_real = int(lang_masks.sum().item())
        return (img_present, lang_real, int(prefix_len))

    def prepare_upstream_artifacts(self, img_masks, lang_masks, prefix_len: int) -> None:
        """Pre-stage the upstream-compat mask + RoPE tensors onto the model
        so a subsequent `sample_actions` call performs zero host→device
        transfers (required to capture the call inside a `ttnn` trace).

        Idempotent: skips rebuilding if the cache key matches the prior
        prepare. Call this BEFORE `ttnn.begin_trace_capture(...)`. Marks
        the cache as 'explicit' so sample_actions bypasses its own
        coerce+key-check path (which would otherwise sync the device).
        """
        ims, lm = self._coerce_upstream_masks(img_masks, lang_masks)
        key = self._upstream_cache_key(ims, lm, prefix_len)
        if self._cached_upstream_artifacts is None or self._cached_upstream_key != key:
            self._cached_upstream_artifacts = self._build_upstream_attn_artifacts(ims, lm, prefix_len=prefix_len)
            self._cached_upstream_key = key
        self._upstream_artifacts_explicit = True

    def _build_upstream_attn_artifacts(
        self,
        img_masks: List[torch.Tensor],
        lang_masks: torch.Tensor,
        prefix_len: int,
    ):
        """Build all the upstream-openpi-compat tensors that the default TTNN
        path skips: VLM prefix attention mask, prefix RoPE table at
        cumsum(pad)-1 positions, expert cross-attention mask (full pad
        masking, not just tile-overhang), and suffix RoPE table at the
        prefix-offset start position.

        Built on host (torch), uploaded as ttnn bf16 TILE. Returns a dict of
        ttnn tensors. Sample_actions feeds these into forward_vlm /
        forward_expert when PI0_SIGLIP_HF=1.

        Args:
            img_masks: per-image torch bool tensors, each (1,) — True if the
                image is real, False for masked-out slots (e.g., right_wrist).
            lang_masks: torch bool tensor (1, lang_seq_len) — True for real
                language tokens, False for padding.
            prefix_len: total prefix length (sum of num_image_tokens per image
                + lang_seq_len). Must match prefix_embs.shape[1].

        Returns:
            dict with keys:
              - prefix_attn_mask:   (1, 1, prefix_padded, prefix_padded) ttnn bf16
              - prefix_cos / prefix_sin: (1, 1, prefix_padded, vlm_head_dim) ttnn bf16
              - expert_attn_mask:   (1, 1, suffix_padded, prefix_padded + suffix_padded) ttnn bf16
              - suffix_cos / suffix_sin: (1, 1, suffix_padded, expert_head_dim) ttnn bf16
        """
        num_image_tokens = self.config.siglip_config.num_patches
        action_horizon = self.config.action_horizon
        suffix_padded = ((action_horizon + 31) // 32) * 32
        prefix_padded = ((prefix_len + 31) // 32) * 32
        vlm_head_dim = self.config.vlm_config.head_dim
        expert_head_dim = self.config.expert_config.head_dim
        max_seq_len = self.config.max_seq_len

        # ---- 1) Build a 1D prefix pad_mask (prefix_len,) on host ----
        pad_segments = []
        for m in img_masks:
            real = bool(m.item()) if m.numel() == 1 else bool(m[0].item())
            pad_segments.append(torch.full((num_image_tokens,), real, dtype=torch.bool))
        # lang_masks: shape (1, lang_seq_len) bool
        pad_segments.append(lang_masks[0].to(torch.bool))
        pad_mask = torch.cat(pad_segments, dim=0)  # (prefix_len,)
        assert pad_mask.shape[0] == prefix_len, (pad_mask.shape, prefix_len)

        prefix_real_count = int(pad_mask.sum().item())  # for suffix offset

        # ---- 2) Prefix attention mask (additive bf16) ----
        # Both ends of attention must be real (bidirectional within prefix).
        # Fast path: if every prefix slot is a real token AND prefix is already
        # tile-aligned, the mask is identically zero — skip uploading and let
        # SDPA take its no-mask code path (~14 µs/call faster on the prefill
        # SDPA op, see traces). The consumer at sample_actions handles None.
        prefix_attn_mask_skipped = int(prefix_real_count) == prefix_len and prefix_padded == prefix_len
        if prefix_attn_mask_skipped:
            prefix_mask_4d = None
        else:
            pad_2d = pad_mask[:, None] & pad_mask[None, :]
            prefix_mask = torch.zeros(prefix_padded, prefix_padded, dtype=torch.bfloat16)
            prefix_mask[:prefix_len, :prefix_len].masked_fill_(~pad_2d, _MASK_VAL)
            if prefix_padded > prefix_len:
                prefix_mask[prefix_len:, :] = _MASK_VAL
                prefix_mask[:, prefix_len:] = _MASK_VAL
            prefix_mask_4d = prefix_mask.unsqueeze(0).unsqueeze(0)

        # ---- 3) Prefix RoPE table at cumsum(pad)-1 positions ----
        # Padding tokens are masked out so their position doesn't matter, but
        # match openpi convention so the "real" K cache values are byte-identical
        # to PyTorch reference.
        position_ids = torch.cumsum(pad_mask.to(torch.int64), dim=0) - 1
        position_ids = position_ids.clamp(min=0, max=max_seq_len - 1)
        cos_vlm, sin_vlm = _precompute_rope_table_torch(vlm_head_dim, max_seq_len)
        prefix_cos = cos_vlm[position_ids]  # (prefix_len, head_dim)
        prefix_sin = sin_vlm[position_ids]
        if prefix_padded > prefix_len:
            zpad_c = torch.zeros(prefix_padded - prefix_len, vlm_head_dim, dtype=prefix_cos.dtype)
            zpad_s = torch.zeros(prefix_padded - prefix_len, vlm_head_dim, dtype=prefix_sin.dtype)
            prefix_cos = torch.cat([prefix_cos, zpad_c], dim=0)
            prefix_sin = torch.cat([prefix_sin, zpad_s], dim=0)
        prefix_cos = prefix_cos.unsqueeze(0).unsqueeze(0)  # (1, 1, prefix_padded, head_dim)
        prefix_sin = prefix_sin.unsqueeze(0).unsqueeze(0)

        # ---- 4) Suffix RoPE table at prefix_real_count + [0..suffix_padded-1] ----
        cos_exp, sin_exp = _precompute_rope_table_torch(expert_head_dim, max_seq_len)
        suffix_positions = (torch.arange(suffix_padded, dtype=torch.int64) + prefix_real_count).clamp(
            max=max_seq_len - 1
        )
        suffix_cos = cos_exp[suffix_positions].unsqueeze(0).unsqueeze(0)
        suffix_sin = sin_exp[suffix_positions].unsqueeze(0).unsqueeze(0)

        # ---- 5) Expert cross-attention mask ----
        # Shape: (1, 1, suffix_padded, prefix_padded + suffix_padded)
        # Block: prefix pad positions, tile-overhang on prefix side, tile-overhang on suffix side.
        kv_total = prefix_padded + suffix_padded
        expert_mask = torch.zeros(suffix_padded, kv_total, dtype=torch.bfloat16)
        # Prefix-side: positions with pad_mask=False get masked
        pad_blocked = (~pad_mask).nonzero(as_tuple=True)[0]
        if pad_blocked.numel() > 0:
            expert_mask[:, pad_blocked] = _MASK_VAL
        if prefix_padded > prefix_len:
            expert_mask[:, prefix_len:prefix_padded] = _MASK_VAL
        if suffix_padded > action_horizon:
            expert_mask[:, prefix_padded + action_horizon : kv_total] = _MASK_VAL
        # Phantom suffix rows: mask them out entirely so they don't contribute
        # to softmax (they're junk Q rows). Their attention output is discarded.
        if suffix_padded > action_horizon:
            expert_mask[action_horizon:suffix_padded, :] = _MASK_VAL
        expert_mask_4d = expert_mask.unsqueeze(0).unsqueeze(0)

        # ---- Upload ----
        def _upload(x_torch, mem=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                x_torch.to(torch.bfloat16) if x_torch.dtype != torch.bfloat16 else x_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=mem,
            )

        # SDPA requires attention masks in DRAM (04 §8 — hard-asserted by the
        # SDPA kernel). RoPE cos/sin tables can live in L1 though: they're
        # small (prefix 256 KB × 2 + suffix 16 KB × 2 = 544 KB total) and
        # consumed by every RoPE call (396 calls × 13.3 µs = 5.275 ms). The
        # PI0_ROPE_TABLES_L1=1 env knob places them in L1 — opt-in until
        # L1 budget is verified to handle them under trace mode.
        # prefix_attn_mask may be None (when all prefix tokens are real and
        # tile-aligned — see fast path above); the SDPA call site handles
        # None as "no masking" and takes the kernel's fast path (~14 µs/call
        # cheaper on the prefill op).
        import os as _os

        _rope_l1 = _os.environ.get("PI0_ROPE_TABLES_L1", "").lower() in ("1", "true", "yes", "on")
        _rope_mc = ttnn.L1_MEMORY_CONFIG if _rope_l1 else ttnn.DRAM_MEMORY_CONFIG

        return {
            "prefix_attn_mask": _upload(prefix_mask_4d, ttnn.DRAM_MEMORY_CONFIG)
            if prefix_mask_4d is not None
            else None,
            "prefix_cos": _upload(prefix_cos, _rope_mc),
            "prefix_sin": _upload(prefix_sin, _rope_mc),
            "expert_attn_mask": _upload(expert_mask_4d, ttnn.DRAM_MEMORY_CONFIG),
            "suffix_cos": _upload(suffix_cos, _rope_mc),
            "suffix_sin": _upload(suffix_sin, _rope_mc),
            "prefix_real_count": prefix_real_count,
        }

    def _build_sdpa_phantom_mask(self, prefix_kv_len_logical: int) -> "ttnn.Tensor":
        """
        SDPA attention mask for the expert keep_padded path.

        Lifted from reverted commit 3d597a3b8e6 with the critical fix from
        the revert's "TIER B revisit" note: use a FINITE large-negative value
        (-1e4) instead of float('-inf'). The original -inf path dropped LIBERO
        accuracy 70%→50% due to bf16 softmax pathology (-inf can poison valid
        rows via NaN propagation; the denominator can also drift if all values
        in a row are -inf). A finite value like -1e4 still produces effectively
        zero softmax weight (exp(-1e4) underflows cleanly to 0 in bf16) without
        the numerical edge cases.

        Mask shape: (1, 1, q_len_padded, kv_total). Values: 0 = attend, -1e4 = mask.
        """
        action_horizon = self.config.action_horizon
        q_len_padded = ((action_horizon + 31) // 32) * 32
        prefix_padded = ((prefix_kv_len_logical + 31) // 32) * 32
        kv_total = prefix_padded + q_len_padded
        mask = torch.zeros(1, 1, q_len_padded, kv_total, dtype=torch.bfloat16)
        # Use -1e4 — finite, large enough that exp(-1e4) = 0 in bf16, but small
        # enough to avoid -inf NaN-propagation and denominator drift.
        MASK_VAL = -1e4
        if prefix_padded > prefix_kv_len_logical:
            mask[:, :, :, prefix_kv_len_logical:prefix_padded] = MASK_VAL
        if q_len_padded > action_horizon:
            suffix_phantom_start = prefix_padded + action_horizon
            mask[:, :, :, suffix_phantom_start:kv_total] = MASK_VAL
        return ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _init_components(self):
        suffix_config = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=True,
        )
        pi0_weights = self.weight_loader.get_pi0_projections()
        ttnn_suffix_weights = convert_pi0_5_suffix_weights_to_ttnn(pi0_weights, self.device)
        self.suffix_embedding = Pi0_5SuffixEmbeddingTTNN(suffix_config, ttnn_suffix_weights, self.device)

        paligemma_config = PaliGemmaConfig(
            vlm_config=self.config.vlm_config,
            expert_config=self.config.expert_config,
            siglip_config=self.config.siglip_config,
            max_seq_len=self.config.max_seq_len,
        )
        weights = self.weight_loader.categorized_weights
        self.backbone = Pi0_5PaliGemmaBackboneTTNN(paligemma_config, weights, self.device)

        prefix_config = PrefixConfig(
            vlm_hidden_size=self.config.vlm_config.width,
            num_image_tokens=self.config.siglip_config.num_patches,
        )
        self.prefix_embedding = PrefixEmbeddingTTNN(
            prefix_config,
            self.device,
            embed_image_fn=self.backbone.embed_image,
            embed_language_fn=self.backbone.embed_language_tokens,
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        return self.prefix_embedding.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def embed_suffix(self, state, noisy_actions, timestep):
        return self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)

    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> "ttnn.Tensor":
        batch_size = lang_tokens.shape[0]

        prefix_embs, _, _ = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        # ttnn.embedding (called inside embed_prefix for language tokens) returns
        # ROW_MAJOR; ttnn.concat with the TILE image embeddings preserves that.
        # ttnn.rms_norm at the start of every VLM block requires TILE — convert
        # the concatenated prefix here before the VLM stack runs.
        if prefix_embs.layout != ttnn.TILE_LAYOUT:
            prefix_embs = ttnn.to_layout(prefix_embs, ttnn.TILE_LAYOUT)

        # UPSTREAM-OPENPI COMPAT (gated by PI0_SIGLIP_HF=1 or
        # PI0_UPSTREAM_MASKS=1): get attention masks + position-aware RoPE
        # tables. Three branches:
        # (a) explicit pre-stage via prepare_upstream_artifacts(): trust the
        #     caller, reuse the cached tensors WITHOUT any host/device sync
        #     so we stay trace-capture-safe.
        # (b) auto-cached: build the key on host, hit the cache if it
        #     matches the prior call. Costs a device sync on the
        #     ttnn img_mask but avoids re-uploading the 6 mask/RoPE tensors.
        # (c) cold build: first call without pre-stage — build + cache.
        upstream_artifacts = None
        if use_upstream_masks():
            if self._upstream_artifacts_explicit and self._cached_upstream_artifacts is not None:
                upstream_artifacts = self._cached_upstream_artifacts
            else:
                ims_t, lm_t = self._coerce_upstream_masks(img_masks, lang_masks)
                key = self._upstream_cache_key(ims_t, lm_t, prefix_embs.shape[1])
                if self._cached_upstream_artifacts is not None and self._cached_upstream_key == key:
                    upstream_artifacts = self._cached_upstream_artifacts
                else:
                    upstream_artifacts = self._build_upstream_attn_artifacts(
                        ims_t, lm_t, prefix_len=prefix_embs.shape[1]
                    )
                    self._cached_upstream_artifacts = upstream_artifacts
                    self._cached_upstream_key = key

        _, prefix_kv_cache = self.backbone.forward_vlm(
            prefix_embs,
            attention_mask=upstream_artifacts["prefix_attn_mask"] if upstream_artifacts else None,
            cos_override=upstream_artifacts["prefix_cos"] if upstream_artifacts else None,
            sin_override=upstream_artifacts["prefix_sin"] if upstream_artifacts else None,
            use_cache=True,
        )
        # NOTE: tried `ttnn.deallocate(prefix_embs)` here on 2026-06-04 22:30
        # but trace mode pins the input tensor → "Tensor is not allocated"
        # at trace replay. Leave it alive; Python GC handles it after
        # sample_actions returns.

        # OPTIMIZATION (keep_padded, reapplied from reverted commit 3d597a3b8e6
        # with finite-mask hybrid fix): treat the expert suffix as
        # logical=physical=tile_align(action_horizon) throughout so the per-layer
        # KV-cache concat path skips the tile/untile ping-pong. Phantom prefix
        # and suffix positions are masked out of softmax via a *finite*
        # large-negative SDPA mask (-1e4 in bf16, exp underflows to 0) — see
        # _build_sdpa_phantom_mask for why -inf was wrong.
        keep_padded_expert = batch_size == 1
        _prefix_kv_cache_original = prefix_kv_cache  # keep alive for tensor lifetime
        if keep_padded_expert and prefix_kv_cache is not None:
            prefix_kv_cache = [
                (
                    ttnn.fill_implicit_tile_padding(k, 0.0),
                    ttnn.fill_implicit_tile_padding(v, 0.0),
                )
                for k, v in prefix_kv_cache
            ]
            prefix_logical_lifted = prefix_kv_cache[0][0].shape[2]  # post-fill: logical=physical
            if self._sdpa_attn_mask is None or self._sdpa_mask_kv_len != prefix_logical_lifted:
                # prefix_embs.shape[1] is the original (pre-lift) logical length.
                self._sdpa_attn_mask = self._build_sdpa_phantom_mask(prefix_embs.shape[1])
                self._sdpa_mask_kv_len = prefix_logical_lifted

        num_steps = self.denoise_config.num_steps
        timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

        timesteps_ttnn = None
        if batch_size != 1:
            pad_steps = ((num_steps + 31) // 32) * 32
            idx = ttnn.to_layout(self.timestep_indices, ttnn.TILE_LAYOUT)
            vals = ttnn.multiply(idx, -1.0 / num_steps)
            timesteps_ttnn = ttnn.add(vals, 1.0)
            timesteps_ttnn = ttnn.reshape(timesteps_ttnn, (1, pad_steps))
            ttnn.deallocate(idx)
            ttnn.deallocate(vals)

        # Resample fresh N(0, 1) noise each call — matches lerobot's
        # sample_noise (modeling_pi05.py:618) and the pytorch reference. Reusing
        # one fixed noise tensor across calls (the prior bug) made every chunk
        # in a rollout converge to the same flow-matching attractor, biasing
        # inference toward whatever modes that seed lands near.
        # Tests that need deterministic noise can set `self.resample_noise = False`
        # and pre-populate `self.x_t_ttnn` (see tests/perf/test_denoise_step_accuracy.py).
        if getattr(self, "resample_noise", True):
            # Host-pad noise to tile-aligned action_horizon so the device
            # tensor lands with logical=physical=64 directly — eliminates the
            # per-call ttnn.pad that previously ran on device.
            ah = self.config.action_horizon
            ah_padded = self._action_horizon_padded
            noise_padded = torch.zeros(1, ah_padded, self.config.action_dim, dtype=torch.float32)
            noise_padded[:, :ah, :] = torch.randn(1, ah, self.config.action_dim)
            x_t_ttnn = ttnn.from_torch(
                noise_padded,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            x_t_ttnn = self.x_t_ttnn
        fast_path = batch_size == 1

        # PI0_DENOISE_FP32=1 keeps the Euler accumulator in fp32 across the
        # 10 denoise steps. bf16+bf16=bf16 accumulates ~bf16_eps · ||x_t||
        # rounding per step; fp32 staging eliminates that. Adds 2 typecasts
        # per step (bf16→fp32 for velocity, fp32→bf16 for embed_actions input).
        fp32_loop = denoise_loop_fp32()
        if fp32_loop:
            x_t_fp32 = ttnn.typecast(x_t_ttnn, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)

        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            if fp32_loop:
                # embed_actions / expert / project_output run in bf16 — cast a
                # transient view of x_t back to bf16 just for this step's forward.
                x_t_bf16 = ttnn.typecast(x_t_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            else:
                x_t_bf16 = x_t_ttnn

            if fast_path:
                # OPTIMIZATION: adarms_cond is precomputed at init (deterministic
                # per step); only the action embedding depends on x_t.
                suffix_embs = self.suffix_embedding.embed_actions(x_t_bf16)
                adarms_cond = self._adarms_cond_per_step_bs1[i]
                precomputed_block_mods = self._block_mods_per_step[i]
                precomputed_final_mod = self._final_mod_per_step[i]
            else:
                assert timesteps_ttnn is not None
                t_tensor = ttnn.slice(timesteps_ttnn, [0, i], [batch_size, i + 1])
                t_tensor = ttnn.reshape(t_tensor, (batch_size,))
                suffix_embs, _, _, adarms_cond = self.embed_suffix(state, x_t_bf16, t_tensor)
                precomputed_block_mods = None
                precomputed_final_mod = None

            if fp32_loop:
                ttnn.deallocate(x_t_bf16)

            # Upstream-compat path overrides the keep_padded phantom mask with
            # a full one that also blocks logical-padding prefix positions
            # (image3, lang-pad), and applies the suffix RoPE at the prefix
            # offset. The keep_padded shape contract is the same (suffix
            # treated as logical=physical=tile_align(action_horizon)) so the
            # KV-cache concat fast path still applies.
            if upstream_artifacts is not None:
                _expert_mask = upstream_artifacts["expert_attn_mask"]
                _cos_o = upstream_artifacts["suffix_cos"]
                _sin_o = upstream_artifacts["suffix_sin"]
            else:
                _expert_mask = self._sdpa_attn_mask if keep_padded_expert else None
                _cos_o = None
                _sin_o = None
            expert_output, _ = self.backbone.forward_expert(
                suffix_embs,
                adarms_cond=adarms_cond,
                past_key_values=prefix_kv_cache,
                precomputed_block_mods=precomputed_block_mods,
                precomputed_final_mod=precomputed_final_mod,
                attention_mask=_expert_mask,
                keep_padded=keep_padded_expert,
                cos_override=_cos_o,
                sin_override=_sin_o,
            )
            ttnn.deallocate(suffix_embs)

            velocity = self.suffix_embedding.project_output(expert_output)
            ttnn.deallocate(expert_output)

            if fp32_loop:
                # Cast velocity to fp32 so the multiply-add accumulates without
                # bf16 rounding noise.
                velocity_fp32 = ttnn.typecast(velocity, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(velocity)
                velocity_scaled = ttnn.mul(velocity_fp32, dt, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(velocity_fp32)
                x_t_new = ttnn.add(x_t_fp32, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(velocity_scaled)
                ttnn.deallocate(x_t_fp32)
                x_t_fp32 = x_t_new
            else:
                velocity_scaled = ttnn.mul(velocity, dt, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(velocity)
                x_t_new = ttnn.add(x_t_ttnn, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(velocity_scaled)
                x_t_ttnn = x_t_new

        if fp32_loop:
            x_t_ttnn = ttnn.typecast(x_t_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(x_t_fp32)

        # Slice off the phantom rows (we always host-pad x_t to ah_padded
        # at init / sample-call time, regardless of keep_padded).
        ah = self.config.action_horizon
        ah_padded = ((ah + 31) // 32) * 32
        if ah_padded > ah and x_t_ttnn.shape[1] != ah:
            x_t_ttnn = ttnn.slice(x_t_ttnn, [0, 0, 0], [1, ah, self.config.action_dim])

        return x_t_ttnn

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: "ttnn.Device",
        config: Optional[Pi0_5ModelConfig] = None,
    ) -> "Pi0_5ModelTTNN":
        weight_loader = PI0WeightLoader(model_path)
        if config is None:
            config = Pi0_5ModelConfig.from_checkpoint(
                model_path,
                action_dim=weight_loader.config.action_dim,
            )
        return cls(config, weight_loader, device)
