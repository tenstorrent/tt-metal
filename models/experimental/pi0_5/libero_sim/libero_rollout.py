#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 LIBERO rollout evaluator — N=10 vs N=4 task success measurement.

Pipeline:
  obs (LIBERO env)
    → resize images to 224x224, normalize to [-1,1]
    → MEAN_STD-normalize state via stats from finetune checkpoint
    → discretize state to 256 bins → string
    → prompt = f"Task: {desc}, State: {bins};\\nAction: "
    → SentencePiece tokenize (max_len=200)
  Pi0_5Model.sample_actions(images, masks, tokens, lang_masks)
    → (B, 50, 32) normalized action chunk
    → take first 7 dims, MEAN_STD denormalize (a_raw = a_norm * std + mean)
    → env.step each of the 50 actions

Run:
  PYTHONPATH=/home/tt-admin/sdawle/pi0/tt-metal:/storage/sdawle/libero_repo \\
  MUJOCO_GL=osmesa python_env/bin/python \\
  models/experimental/pi0_5/libero_sim/libero_rollout.py --num-episodes 5 --max-steps 200
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

# Bypass lerobot's transformers-replace check
import types as _types

_fake = _types.ModuleType("transformers.models.siglip.check")
_fake.check_whether_transformers_replace_is_installed_correctly = lambda: True
sys.modules["transformers.models.siglip.check"] = _fake

# Resolve REPO_ROOT from this file's location instead of a hard-coded path so
# the script works in any tt-metal checkout (e.g. pi05_bh_glx vs pi0 worktree).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
LIBERO_REPO = os.environ.get("LIBERO_REPO_PATH", "/storage/sdawle/libero_repo")
if LIBERO_REPO not in sys.path:
    sys.path.insert(0, LIBERO_REPO)
os.environ.setdefault("MUJOCO_GL", "osmesa")

import sentencepiece
from safetensors.torch import load_file

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model


# TTNN imports are deferred so the pytorch backend doesn't require a Blackhole
def _import_ttnn():
    import ttnn
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    return ttnn, Pi0_5ModelTTNN


# ---------------------------------------------------------------------------
# Preprocessor / postprocessor
# ---------------------------------------------------------------------------


class Pi0_5LiberoAdapter:
    """Wraps Pi0_5Model (pytorch) or Pi0_5ModelTTNN + pi0.5 preprocessing for LIBERO."""

    def __init__(
        self,
        checkpoint_dir: str,
        tokenizer_path: str = os.environ.get(
            "PI0_TOKENIZER_PATH", "/storage/sdawle/pi05_weights/paligemma_tokenizer.model"
        ),
        backend: str = "pytorch",
        ttnn_device=None,
        mesh_handles=None,  # ttnn_1x8 backend only — the mesh from open_prefill_tp8_mesh()
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        chunk_size: int = 50,
        max_token_len: int = 256,  # tile-aligned: 256 image + 256 lang = 512 = 16×32
        # NOTE: must match the m_padded values validated for sharded RMSNorm in
        # ttnn_gemma._get_sharded_norm. 224 (gives 480) triggered an L1 CB clash
        # because the sub-block sizing for 15 M-tiles doesn't fit the per-core
        # L1 budget on BH; 256 (gives 512 = 16 M-tiles) is the validated path.
        action_horizon: int = 50,
        state_in_prompt: bool = True,
    ):
        self.backend = backend
        self.ttnn_device = ttnn_device
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.chunk_size = chunk_size
        self.max_token_len = max_token_len
        self.image_size = 224
        # `state_in_prompt=True` (default, lerobot pi05_libero_finetuned convention):
        # state is normalized → discretized to 256 bins → embedded in the language
        # prompt as text. `state_in_prompt=False` (upstream openpi pi05_libero with
        # discrete_state_input=False): prompt is task description only; the model
        # was trained to infer state implicitly from vision and never sees a state
        # token. The model architecture (no state_proj weight in either variant)
        # is identical; only the prompt construction differs.
        self.state_in_prompt = state_in_prompt

        # Tokenizer
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        # Normalizer stats. Supports both formats:
        #  - lerobot finetune: policy_preprocessor_step_2_normalizer_processor.safetensors
        #  - openpi upstream (converted): assets/physical-intelligence/libero/norm_stats.json
        # Both use MEAN_STD for pi05_libero (per policy_preprocessor.json /
        # norm_stats.json — the safetensors file contains BOTH MEAN_STD and
        # QUANTILE stats; using the wrong one silently produces actions in the
        # wrong scale/offset).
        upstream_json = os.path.join(checkpoint_dir, "assets", "physical-intelligence", "libero", "norm_stats.json")
        lerobot_safetensors = os.path.join(
            checkpoint_dir, "policy_preprocessor_step_2_normalizer_processor.safetensors"
        )
        # openpi config.py:187 — `use_quantile_norm = model_type != PI0`. For
        # pi05 (PI05 model_type), this is True, so the upstream pi05_libero was
        # trained with QUANTILE normalization (q01/q99-based), not MEAN_STD.
        # The lerobot finetune (state_in_prompt=True) was trained with MEAN_STD.
        self.use_quantile_norm = not state_in_prompt
        if os.path.exists(upstream_json):
            import json as _json

            with open(upstream_json) as f:
                ns = _json.load(f)["norm_stats"]
            self.action_mean = np.asarray(ns["actions"]["mean"], dtype=np.float32)
            self.action_std = np.asarray(ns["actions"]["std"], dtype=np.float32)
            self.state_mean = np.asarray(ns["state"]["mean"], dtype=np.float32)
            self.state_std = np.asarray(ns["state"]["std"], dtype=np.float32)
            self.action_q01 = np.asarray(ns["actions"].get("q01", []), dtype=np.float32)
            self.action_q99 = np.asarray(ns["actions"].get("q99", []), dtype=np.float32)
            self.state_q01 = np.asarray(ns["state"].get("q01", []), dtype=np.float32)
            self.state_q99 = np.asarray(ns["state"].get("q99", []), dtype=np.float32)
        else:
            flat = load_file(lerobot_safetensors)
            self.action_mean = flat["action.mean"].float().numpy()  # (7,)
            self.action_std = flat["action.std"].float().numpy()
            self.state_mean = flat["observation.state.mean"].float().numpy()  # (8,)
            self.state_std = flat["observation.state.std"].float().numpy()
            self.action_q01 = self.action_q99 = self.state_q01 = self.state_q99 = np.zeros(0, dtype=np.float32)
        self.real_action_dim = len(self.action_mean)
        self.real_state_dim = len(self.state_mean)

        # Model — pass action_horizon explicitly so callers can match the value
        # the checkpoint was trained for (lerobot pi05_libero_finetuned = 50,
        # upstream openpi pi05_libero = 10).
        cfg = Pi0_5ModelConfig(action_horizon=action_horizon)
        loader = Pi0_5WeightLoader(checkpoint_dir)
        if backend == "pytorch":
            self.model = Pi0_5Model(cfg, loader)
            self._ttnn = None
            self._Pi0_5ModelTTNN = None
        elif backend == "ttnn":
            assert ttnn_device is not None, "TTNN backend requires an open ttnn_device"
            ttnn_mod, Pi0_5ModelTTNN = _import_ttnn()
            self._ttnn = ttnn_mod
            self._Pi0_5ModelTTNN = Pi0_5ModelTTNN
            self.model = Pi0_5ModelTTNN(cfg, loader, ttnn_device)
        elif backend == "ttnn_1x8":
            # 1×8 mesh pipeline: 8 chips with on-device CCL, num_command_queues=2
            # for H2D overlap, traced + 2CQ replay loop. mesh_handles is the raw
            # mesh device returned by open_prefill_tp8_mesh(tp=8, num_command_queues=2).
            assert mesh_handles is not None, "ttnn_1x8 backend requires mesh from open_prefill_tp8_mesh()"
            from models.experimental.pi0_5.tt.tt_bh_glx.pipeline_1x8 import Pi0_5GLX1x8Pipeline

            self._ttnn = None
            self._Pi0_5ModelTTNN = None
            self.mesh_handles = mesh_handles
            self.model = Pi0_5GLX1x8Pipeline(cfg, loader.categorized_weights, mesh_handles)
            # Per-task trace cache: re-capture when (task_desc, num_denoising_steps)
            # changes. Same idea as `ttnn` backend's trace cache (line ~631-654).
            self._glx1x8_trace_key = None
        else:
            raise ValueError(f"Unknown backend: {backend}")
        self.cfg = cfg
        self.device = torch.device("cpu")

        # TTNN trace state. On by default for backend=ttnn; opt out with
        # PI0_LIBERO_TRACE=0. When active, each `predict_chunk` writes new
        # inputs into persistent on-device buffers via
        # `ttnn.copy_host_to_device_tensor` and then replays a captured trace,
        # dropping per-chunk wall-clock from the untraced ~370 ms toward the
        # trace-perf baseline (~60 ms). The trace is rebuilt when the
        # (task_desc, num_denoising_steps) key changes — different prompt ->
        # different lang_mask -> different upstream mask/RoPE artifacts.
        # Verified accuracy-neutral: traced 40/40 on libero_spatial task 0 and
        # 40/40 across all four suites × 10 tasks @ 1 ep/task, plus a
        # byte-identical drift diagnostic (cosim=1.0).
        import os as _os

        _trace_env = _os.environ.get("PI0_LIBERO_TRACE", "").strip().lower()
        self._use_trace = backend == "ttnn" and _trace_env not in ("0", "false", "no", "off")
        self._trace_id = None
        self._trace_actions_output = None
        self._trace_key = None
        # Pre-allocated persistent input buffers, populated lazily on first
        # chunk so we know the shapes coming from the rollout.
        self._trace_images_host = None  # List[ttnn.Tensor] (host-side)
        self._trace_images_dev = None  # List[ttnn.Tensor] (device-resident)
        self._trace_img_masks_host = None
        self._trace_img_masks_dev = None
        self._trace_tokens_host = None
        self._trace_tokens_dev = None
        self._trace_lang_mask_host = None
        self._trace_lang_mask_dev = None
        self._trace_noise_host = None
        # Action-horizon padded to tile boundary (must match
        # model.x_t_ttnn shape).
        self._trace_ah_padded = ((cfg.action_horizon + 31) // 32) * 32

    # --- image ---
    @staticmethod
    def _resize_with_pad_centered(img_hwc_uint8: np.ndarray, size: int = 224) -> np.ndarray:
        """Aspect-preserving bilinear resize, centered pad with -1.0 (black in
        PaliGemma's [-1, 1] space). Mirrors lerobot's resize_with_pad_torch.

        img: (H, W, 3) uint8 → (size, size, 3) float32 in [-1, 1].
        """
        from PIL import Image

        h, w = img_hwc_uint8.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = (
            np.asarray(
                Image.fromarray(img_hwc_uint8).resize((nw, nh), Image.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )
        resized = resized * 2.0 - 1.0  # [-1, 1]
        # Center pad with -1 (black)
        out = -np.ones((size, size, 3), dtype=np.float32)
        oy = (size - nh) // 2
        ox = (size - nw) // 2
        out[oy : oy + nh, ox : ox + nw] = resized
        return out

    def _image_for_pi05(self, img_hwc_uint8: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 → (1, 3, 224, 224) float32 in [-1, 1] (PaliGemma convention).

        Caller is responsible for any rotation: in our LIBERO run_episode the
        180° rotation (`img[::-1, ::-1]`) happens up-front for *both* lerobot
        and upstream modes — matching openpi/examples/libero/main.py:115-117 —
        so we don't rotate again here. (We used to rotate again in upstream
        mode and that was a double-rotation that silently undid itself.)
        """
        img_pad = self._resize_with_pad_centered(img_hwc_uint8, self.image_size)
        chw = np.transpose(img_pad, (2, 0, 1))  # (3, H, W)
        return torch.from_numpy(chw).unsqueeze(0).contiguous()

    # --- state ---
    def _state_normalize(self, state: np.ndarray) -> np.ndarray:
        """State normalization → padded to max_state_dim.

        - state_in_prompt=True (lerobot finetune): MEAN_STD (`(x - mean) / std`).
        - state_in_prompt=False (upstream openpi pi05): QUANTILE
          (`(x - q01) / (q99 - q01) * 2 - 1`), per openpi.transforms.Normalize.
        """
        eps = 1e-6
        if self.use_quantile_norm and len(self.state_q01) == len(state):
            s = (state - self.state_q01) / (self.state_q99 - self.state_q01 + eps) * 2.0 - 1.0
        else:
            s = (state - self.state_mean) / (self.state_std + 1e-8)
        padded = np.zeros(self.max_state_dim, dtype=np.float32)
        padded[: len(s)] = s
        return padded

    @staticmethod
    def _discretize_state(s_norm: np.ndarray, n_bins: int = 256) -> np.ndarray:
        edges = np.linspace(-1.0, 1.0, n_bins + 1)[:-1]
        return np.digitize(s_norm, bins=edges) - 1  # ints in [0, n_bins-1]

    # --- prompt + tokenize ---
    def _make_tokens(self, task_desc: str, state: np.ndarray):
        """Returns (tokens (max_token_len,), mask (max_token_len,)) as torch tensors.

        Two prompt formats are supported:
         - `state_in_prompt=True` (lerobot pi05_libero_finetuned convention):
           "Task: <desc>, State: 128 200 145 ...; Action: " — state quantized
           to 256 bins and embedded as text. The model learns to read it.
         - `state_in_prompt=False` (upstream openpi pi05_libero, discrete_state_input=False):
           just the cleaned task description + "\n", matching openpi's
           tokenizer (see `openpi/src/openpi/models/tokenizer.py:28-33`).
           No "Task:" prefix, no "Action:" suffix — the model was trained
           only on the raw description tokens.
        """
        cleaned = task_desc.strip().replace("_", " ").replace("\n", " ")
        if self.state_in_prompt:
            s_norm = self._state_normalize(state)
            bins = self._discretize_state(s_norm)
            state_str = " ".join(str(int(b)) for b in bins)
            full = f"Task: {cleaned}, State: {state_str};\nAction: "
            tokens = self.sp.encode(full, add_bos=True)
        else:
            # Upstream openpi format: bos + <desc> + "\n" (tokenizer.py:28-33).
            # encode(text, add_bos=True) + encode("\n") preserves the explicit
            # newline tokenization openpi uses.
            tokens = self.sp.encode(cleaned, add_bos=True) + self.sp.encode("\n")
        L = len(tokens)
        if L < self.max_token_len:
            mask = [True] * L + [False] * (self.max_token_len - L)
            tokens = tokens + [0] * (self.max_token_len - L)
        else:
            tokens = tokens[: self.max_token_len]
            mask = [True] * self.max_token_len
        return (
            torch.tensor(tokens, dtype=torch.int32).unsqueeze(0),
            torch.tensor(mask, dtype=torch.bool).unsqueeze(0),
        )

    # --- action denorm ---
    def _denormalize_actions(self, actions_norm: np.ndarray) -> np.ndarray:
        """(chunk, max_action_dim) normalized → (chunk, real_action_dim) raw.

        - state_in_prompt=True (lerobot finetune): MEAN_STD inverse
          (`a_raw = a_norm * std + mean`).
        - state_in_prompt=False (upstream openpi pi05): QUANTILE inverse
          (`a_raw = (a_norm + 1) / 2 * (q99 - q01) + q01`).

        Per openpi.transforms.Unnormalize (and config.py:187 which sets
        `use_quantile_norm = model_type != PI0`).
        """
        eps = 1e-6
        a = actions_norm[:, : self.real_action_dim]
        if self.use_quantile_norm and len(self.action_q01) == self.real_action_dim:
            a = (a + 1.0) / 2.0 * (self.action_q99 - self.action_q01 + eps) + self.action_q01
        else:
            a = a * self.action_std + self.action_mean
        return a

    # --- main entry: predict a chunk given LIBERO obs ---
    def predict_chunk(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        task_desc: str,
        num_denoising_steps: int = 10,
    ) -> np.ndarray:
        """Returns (chunk_size, real_action_dim) numpy array of LIBERO-space actions."""
        # 3 cameras: agentview, wrist, empty placeholder.
        # The third camera is masked False so the model ignores it. Fill value
        # follows the training convention:
        #  - lerobot:  -1 (black in PaliGemma's [-1,1] convention) — matches
        #              lerobot.policies.pi05.modeling_pi05._preprocess_images
        #  - openpi:   0 (zero-padding, matches
        #              openpi/src/openpi/policies/libero_policy.py:60-67's
        #              np.zeros_like(base_image) for the right_wrist slot)
        img1 = self._image_for_pi05(agentview_image)
        img2 = self._image_for_pi05(wrist_image)
        if self.state_in_prompt:
            img3 = torch.ones_like(img1) * -1.0
        else:
            img3 = torch.zeros_like(img1)
        images = [img1, img2, img3]
        img_masks = [
            torch.ones(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
        ]

        tokens, lang_mask = self._make_tokens(task_desc, state)

        if self.backend == "pytorch":
            # Override the num_denoising_steps for this call (pytorch reference)
            original_steps = self.model.denoising.config.num_steps
            self.model.denoising.config.num_steps = num_denoising_steps
            with torch.no_grad():
                actions = self.model.sample_actions(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=tokens,
                    lang_masks=lang_mask,
                    state=None,
                )
            self.model.denoising.config.num_steps = original_steps
            actions_np = actions[0].float().cpu().numpy()
        elif self.backend == "ttnn_1x8":
            # 1×8 mesh + Trace + 2CQ pipeline. capture_trace rebuilds attention
            # artifacts (prefix mask, position-aware RoPE, suffix RoPE offset,
            # expert mask) for the runtime img_masks + lang_masks BEFORE warmup
            # + trace capture. Re-captures the trace when masks change (per task).
            # sample_actions_traced_2cq_loop refreshes input buffers (H2D on CQ1)
            # and replays the trace (CQ0) with 2CQ event pingpong.
            cache_key = (
                task_desc,
                num_denoising_steps,
                int(tokens.shape[-1]),
                tuple(bool(m.item()) if m.numel() == 1 else bool(m[0].item()) for m in img_masks),
                int(lang_mask[0].to(torch.bool).sum().item()),
            )
            if cache_key != self._glx1x8_trace_key:
                # set_num_denoising_steps() rebuilds the per-step schedule
                # (_timesteps, _dts, _adarms_per_step, _block_mods_per_step,
                # _final_mods_per_step). Bare assignment used to silently
                # leave the precomputes built for config.num_denoising_steps
                # (default 10), so a sweep into N=5 ran 5 iters but indexed
                # the first 5 entries of a 10-step schedule (dt=-0.1 instead
                # of -0.2) → half-denoise → tanked LIBERO accuracy.
                self.model.set_num_denoising_steps(num_denoising_steps)
                self.model.capture_trace(
                    images,
                    lang_tokens=tokens,
                    img_masks=img_masks,
                    lang_masks=lang_mask,
                )
                self._glx1x8_trace_key = cache_key
            with torch.no_grad():
                actions, _times = self.model.sample_actions_traced_2cq_loop(
                    images,
                    lang_tokens=tokens,
                    iters=1,
                )
            actions_np = actions[0].float().cpu().numpy()
        elif self._use_trace:
            # === TTNN backend with persistent buffers + trace replay ===
            # Path through `_predict_chunk_traced` does the per-chunk
            # `copy_host_to_device_tensor` + `execute_trace` dance. Trace is
            # captured lazily on the first chunk per task and reused across
            # chunks within that task. See class init for the rationale.
            actions_np = self._predict_chunk_traced(images, img_masks, tokens, lang_mask, num_denoising_steps)
        else:
            # TTNN backend: convert torch inputs → ttnn, run, convert back.
            ttnn = self._ttnn
            device = self.ttnn_device
            # Override num_denoising_steps on the TTNN model (rebuild precomputed lists)
            self.model.denoise_config.num_steps = num_denoising_steps
            self.model._precompute_bs1_timestep_tensors()
            self.model._precompute_bs1_adarms_cond()
            # Build TTNN inputs
            # PI0_SIGLIP_USE_FOLD=1: pre-stack all cameras on host into a single
            # (N, H, W, 3) NHWC ROW_MAJOR tensor and pass a list of length 1.
            # The model side (ttnn_prefix.py) detects the pre-stacked case and
            # skips the device-side ROW_MAJOR concat (~0.5 ms savings at bs=3).
            _use_fold = os.environ.get("PI0_SIGLIP_USE_FOLD", "").lower() in ("1", "true", "yes", "on")
            if _use_fold:
                stacked_host = torch.cat([im.permute(0, 2, 3, 1).contiguous() for im in images], dim=0)
                images_ttnn = [
                    ttnn.from_torch(
                        stacked_host,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                ]
            else:
                images_ttnn = []
                for img in images:
                    images_ttnn.append(
                        ttnn.from_torch(
                            img,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    )
            img_masks_ttnn = []
            for m in img_masks:
                img_masks_ttnn.append(
                    ttnn.from_torch(
                        m.float(),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                )
            tokens_ttnn = ttnn.from_torch(
                tokens.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            lang_mask_ttnn = ttnn.from_torch(
                lang_mask.to(torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            # Pre-stage upstream-compat artifacts (mask + position-aware RoPE)
            # when PI0_UPSTREAM_MASKS=1 or PI0_SIGLIP_HF=1. Idempotent: keyed
            # by (img_present_tuple, lang_real_count, prefix_len) so within a
            # task the call costs ~0 — only the FIRST chunk pays the host
            # build + 6-tensor upload. Without this, sample_actions does the
            # build inside the hot path (~315 ms / chunk overhead measured
            # in the 40-task upstream sweep). Use the host (torch) masks to
            # avoid the device-sync that ttnn-mask coercion would trigger.
            from models.experimental.pi0_5.tt.ttnn_pi0_5_model import use_upstream_masks

            if use_upstream_masks():
                num_image_tokens = self.cfg.siglip_config.num_patches
                prefix_len = len(images) * num_image_tokens + lang_mask.shape[-1]
                self.model.prepare_upstream_artifacts(img_masks, lang_mask, prefix_len=prefix_len)
            out = self.model.sample_actions(
                images=images_ttnn,
                img_masks=img_masks_ttnn,
                lang_tokens=tokens_ttnn,
                lang_masks=lang_mask_ttnn,
                state=None,
            )
            actions_np = ttnn.to_torch(out).float().numpy()
            if actions_np.ndim == 3:
                actions_np = actions_np[0]
            actions_np = actions_np[: self.chunk_size, : self.max_action_dim]
            # Free TTNN intermediates
            for t in images_ttnn:
                ttnn.deallocate(t)
            for t in img_masks_ttnn:
                ttnn.deallocate(t)
            ttnn.deallocate(tokens_ttnn)
            ttnn.deallocate(lang_mask_ttnn)
            ttnn.deallocate(out)

        return self._denormalize_actions(actions_np)  # (50, 7)

    # -----------------------------------------------------------------
    # Trace path (PI0_LIBERO_TRACE=1)
    # -----------------------------------------------------------------

    def _build_noise_torch(self) -> torch.Tensor:
        """Match Pi0_5ModelTTNN's noise-prep convention: (1, ah_padded, action_dim)
        with the action_horizon rows filled with N(0, 1) and the tail zero.
        Caller writes this into model.x_t_ttnn each chunk so the captured
        trace replays with fresh noise (matches openpi's per-call resample).
        """
        ah = self.cfg.action_horizon
        ah_padded = self._trace_ah_padded
        noise_padded = torch.zeros(1, ah_padded, self.cfg.action_dim, dtype=torch.float32)
        noise_padded[:, :ah, :] = torch.randn(1, ah, self.cfg.action_dim)
        return noise_padded

    def _ensure_trace_buffers(self, images, img_masks, tokens, lang_mask):
        """Lazily allocate the persistent input buffers (host + device pairs)
        on the first call. Shapes locked in from the first chunk; all
        subsequent calls write into the same buffers via
        ttnn.copy_host_to_device_tensor.
        """
        if self._trace_images_host is not None:
            return
        ttnn = self._ttnn
        device = self.ttnn_device

        # Image buffers — one per camera, matching the untraced layout
        # (TILE bf16 DRAM).
        self._trace_images_host = []
        self._trace_images_dev = []
        for img in images:
            host_t = ttnn.from_torch(img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            dev_t = ttnn.from_torch(
                img,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._trace_images_host.append(host_t)
            self._trace_images_dev.append(dev_t)

        # Image-mask buffers (one per camera, ROW_MAJOR bf16 L1)
        self._trace_img_masks_host = []
        self._trace_img_masks_dev = []
        for m in img_masks:
            host_t = ttnn.from_torch(m.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            dev_t = ttnn.from_torch(
                m.float(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self._trace_img_masks_host.append(host_t)
            self._trace_img_masks_dev.append(dev_t)

        # Language token + mask buffers
        self._trace_tokens_host = ttnn.from_torch(
            tokens.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self._trace_tokens_dev = ttnn.from_torch(
            tokens.to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self._trace_lang_mask_host = ttnn.from_torch(
            lang_mask.to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self._trace_lang_mask_dev = ttnn.from_torch(
            lang_mask.to(torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        # Noise host-side staging tensor — write to model.x_t_ttnn each call.
        self._trace_noise_host = ttnn.from_torch(
            self._build_noise_torch(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def _write_trace_inputs(self, images, img_masks, tokens, lang_mask, refresh_noise: bool = True):
        """Write the current chunk's data into the persistent device buffers
        via `ttnn.copy_host_to_device_tensor`. No new device allocations.
        """
        ttnn = self._ttnn

        # Images — rebuild host tensors (tile cast happens here) and copy.
        for i, img in enumerate(images):
            host_t = ttnn.from_torch(img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_t, self._trace_images_dev[i])
            self._trace_images_host[i] = host_t  # keep ref alive

        for i, m in enumerate(img_masks):
            host_t = ttnn.from_torch(m.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_t, self._trace_img_masks_dev[i])
            self._trace_img_masks_host[i] = host_t

        host_t = ttnn.from_torch(tokens.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_t, self._trace_tokens_dev)
        self._trace_tokens_host = host_t

        host_t = ttnn.from_torch(lang_mask.to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_t, self._trace_lang_mask_dev)
        self._trace_lang_mask_host = host_t

        if refresh_noise:
            noise_host = ttnn.from_torch(self._build_noise_torch(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(noise_host, self.model.x_t_ttnn)
            self._trace_noise_host = noise_host

    def _predict_chunk_traced(self, images, img_masks, tokens, lang_mask, num_denoising_steps: int) -> np.ndarray:
        """Trace-replay path for one chunk. See class init for context.

        Flow:
          1. Lazy: allocate persistent input buffers (`_ensure_trace_buffers`).
          2. If task changed (key mismatch): release old trace, re-stage
             upstream artifacts, run one warmup forward, capture trace.
          3. Write new chunk inputs into the persistent buffers + refresh
             noise via copy_host_to_device_tensor.
          4. `ttnn.execute_trace`.
          5. Read `_trace_actions_output` back to host.
        """
        ttnn = self._ttnn
        device = self.ttnn_device

        from models.experimental.pi0_5.tt.ttnn_pi0_5_model import use_upstream_masks

        # Cache key: trace must be rebuilt when the captured graph's tensor
        # references change. The lang_mask, lang_tokens, and image tensors
        # are referenced by buffer in the trace, so their values can change
        # via copy_host_to_device_tensor without invalidating capture.
        # Things that DO require re-capture:
        #   - num_denoising_steps: changes the captured op count.
        #   - lang_real_count when use_upstream_masks(): the mask/RoPE
        #     artifacts are rebuilt by prepare_upstream_artifacts when the
        #     real-token count changes, which produces a new tensor (new
        #     address) — the captured trace still points at the old one.
        #   - tokens.shape[-1]: changes the lang dim shape.
        #   - img_mask pattern: changes which images contribute to the
        #     prefix attention mask (upstream path) and changes pad masks.
        key = (
            int(tokens.shape[-1]),
            tuple(int(m.any().item()) for m in img_masks),
            int(num_denoising_steps),
            # Only include lang_real_count when upstream artifacts depend on
            # it. In the default lerobot path the prompt is the discretized
            # state which re-tokenizes per chunk, but the captured graph
            # doesn't depend on that count — including it would trigger
            # spurious re-captures (0.8 s each).
            int(lang_mask.sum().item()) if use_upstream_masks() else None,
        )

        self._ensure_trace_buffers(images, img_masks, tokens, lang_mask)

        if self._trace_id is None or self._trace_key != key:
            # Release any stale trace + output reference before re-capturing.
            if self._trace_id is not None:
                if self._trace_actions_output is not None:
                    ttnn.deallocate(self._trace_actions_output)
                ttnn.release_trace(device, self._trace_id)
                self._trace_id = None
                self._trace_actions_output = None

            # Set denoise step count + rebuild model's per-step precomputed
            # tensors so the upcoming warmup + trace capture see the right N.
            self.model.denoise_config.num_steps = num_denoising_steps
            self.model._precompute_bs1_timestep_tensors()
            self.model._precompute_bs1_adarms_cond()
            self.model._precompute_bs1_modulations()

            # Populate buffers with the current chunk so warmup + capture run
            # on real-shape data, then pre-stage upstream artifacts (mask +
            # RoPE) once.
            self._write_trace_inputs(images, img_masks, tokens, lang_mask)
            if use_upstream_masks():
                num_image_tokens = self.cfg.siglip_config.num_patches
                prefix_len = len(images) * num_image_tokens + lang_mask.shape[-1]
                self.model.prepare_upstream_artifacts(img_masks, lang_mask, prefix_len=prefix_len)

            # Resample-noise must be False so sample_actions doesn't try to
            # ttnn.from_torch new noise inside the captured region (trace
            # forbids host→device transfers).
            self.model.resample_noise = False

            # Warmup: one untraced forward to JIT-compile kernels.
            with torch.no_grad():
                warm_out = self.model.sample_actions(
                    images=self._trace_images_dev,
                    img_masks=self._trace_img_masks_dev,
                    lang_tokens=self._trace_tokens_dev,
                    lang_masks=self._trace_lang_mask_dev,
                    state=None,
                )
            ttnn.synchronize_device(device)
            if isinstance(warm_out, ttnn.Tensor):
                ttnn.deallocate(warm_out)

            # Capture.
            self._trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            self._trace_actions_output = self.model.sample_actions(
                images=self._trace_images_dev,
                img_masks=self._trace_img_masks_dev,
                lang_tokens=self._trace_tokens_dev,
                lang_masks=self._trace_lang_mask_dev,
                state=None,
            )
            ttnn.end_trace_capture(device, self._trace_id, cq_id=0)
            ttnn.synchronize_device(device)
            self._trace_key = key
        else:
            # Steady-state path: just write the chunk's inputs and replay.
            # PI0_LIBERO_TRACE_FIXED_NOISE=1 reuses the capture-time noise on
            # every replay (used to isolate accuracy bugs — distinguishes
            # "trace mechanism broken" from "noise refresh broken").
            import os as _os

            refresh_noise = _os.environ.get("PI0_LIBERO_TRACE_FIXED_NOISE", "").strip().lower() not in (
                "1",
                "true",
                "yes",
                "on",
            )
            self._write_trace_inputs(images, img_masks, tokens, lang_mask, refresh_noise=refresh_noise)
            ttnn.execute_trace(device, self._trace_id, cq_id=0, blocking=True)

        actions_np = ttnn.to_torch(self._trace_actions_output).float().numpy()
        if actions_np.ndim == 3:
            actions_np = actions_np[0]
        return actions_np[: self.chunk_size, : self.max_action_dim]


# ---------------------------------------------------------------------------
# LIBERO env helper
# ---------------------------------------------------------------------------


def make_libero_env(suite_name: str = "libero_spatial", task_idx: int = 0, img_size: int = 256):
    # LIBERO uses torch.load(...) on pickled numpy arrays for init states.
    # PyTorch 2.6 changed default `weights_only=True` which blocks this. Patch.
    import torch as _torch

    _orig_load = _torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _torch.load = _patched_load
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_idx)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_kwargs = dict(
        bddl_file_name=bddl,
        camera_heights=img_size,
        camera_widths=img_size,
    )
    env = OffScreenRenderEnv(**env_kwargs)
    env.seed(0)  # important per openpi comment: affects object positions even with set_init_state
    # Canonical seeded initial states for this task (the model is trained on these).
    initial_states = suite.get_task_init_states(task_idx)
    return env, task, initial_states


# ---------------------------------------------------------------------------
# Rollout loop
# ---------------------------------------------------------------------------


def run_episode(
    env,
    adapter: "Pi0_5LiberoAdapter",
    task_desc: str,
    num_denoising_steps: int,
    max_steps: int = 200,
    chunk_action_horizon: int = 50,
    num_steps_wait: int = 10,
    initial_state=None,
    seed: int = 0,
    record_frames: bool = False,
) -> dict:
    """Run one episode. Returns metrics dict.

    Matches openpi's LIBERO eval pipeline:
      - sets a canonical seeded init state (the model is trained on these)
      - waits `num_steps_wait` dummy steps for physics to settle
      - rotates camera images 180° (LIBERO renders are flipped vs training)
      - replans every `chunk_action_horizon` actions

    If `record_frames=True`, returns a `frames` list of (H, W, 3) uint8 arrays —
    the rotated agentview at every env.step. Caller writes mp4.
    """
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # no motion, gripper open
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = (
            env.regenerate_obs_from_state(env.get_sim_state())
            if hasattr(env, "regenerate_obs_from_state")
            else env.reset()
        )
    frames = [] if record_frames else None
    # Let physics settle (dropped objects need to fall in LIBERO)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        if record_frames:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
    success = False
    n_steps = 0
    inference_times = []
    while n_steps < max_steps:
        # Build adapter inputs — IMPORTANT: rotate 180° to match training (openpi convention)
        agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        # 8-dim state matching pi05_libero training:
        #   eef_pos (3) + eef axis-angle (3) + gripper_qpos (2) = 8
        # robosuite stores eef_quat as (x, y, z, w). Convert to axis*angle.
        quat_xyzw = obs["robot0_eef_quat"].astype(np.float32)
        w = float(np.clip(quat_xyzw[3], -1.0, 1.0))
        angle = 2.0 * np.arccos(w)
        sinh = max(float(np.sqrt(max(1.0 - w * w, 0.0))), 1e-8)
        axis_angle = (quat_xyzw[:3] / sinh) * angle
        state = np.concatenate(
            [
                obs["robot0_eef_pos"].astype(np.float32),  # 3
                axis_angle.astype(np.float32),  # 3
                obs["robot0_gripper_qpos"].astype(np.float32),  # 2
            ]
        )  # total 8
        # Predict chunk
        t0 = time.perf_counter()
        chunk = adapter.predict_chunk(
            agent_img,
            wrist_img,
            state,
            task_desc,
            num_denoising_steps=num_denoising_steps,
        )
        dt = time.perf_counter() - t0
        inference_times.append(dt)
        # Debug: dump the first action of the first 2 chunks so we can sanity-check direction.
        if len(inference_times) <= 2:
            print(
                f"      chunk {len(inference_times)} at step {n_steps}: pred {dt:.2f}s "
                f"act[0]={np.array2string(chunk[0], precision=3, suppress_small=True)} "
                f"eef={np.array2string(obs['robot0_eef_pos'], precision=3, suppress_small=True)}",
                flush=True,
            )
        else:
            print(f"      chunk {len(inference_times)} at step {n_steps}: pred {dt:.2f}s", flush=True)
        # Apply each action in the chunk.
        for a in chunk[:chunk_action_horizon]:
            obs, _, done, _ = env.step(a.astype(np.float64))
            n_steps += 1
            if record_frames:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if done:
                success = True
                break
            if n_steps >= max_steps:
                break
        if success:
            break

    out = {
        "success": success,
        "steps": n_steps,
        "avg_chunk_pred_time": float(np.mean(inference_times)) if inference_times else 0.0,
        "n_chunks": len(inference_times),
    }
    if record_frames:
        out["frames"] = frames
    return out


# Per-suite max episode lengths matched to the longest training demos.
# Values from openpi `examples/libero/main.py`.
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "pi05_libero_finetuned")
        ),
        help="Checkpoint dir. Default: the public lerobot pi05_libero_finetuned "
        "(action_horizon=50, MEAN_STD, state-in-prompt). For the upstream openpi "
        "pi05_libero, pass its dir with --action-horizon 10 --state-in-prompt false.",
    )
    ap.add_argument("--suite", default="libero_spatial", help="Single suite (legacy; overridden by --suites).")
    ap.add_argument(
        "--suites",
        nargs="+",
        default=None,
        help="One or more suites to run (e.g. libero_spatial libero_object libero_goal libero_10). "
        "Overrides --suite when set.",
    )
    ap.add_argument("--task-idx", type=int, default=0, help="Single task (legacy; overridden by --task-range).")
    ap.add_argument(
        "--task-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Inclusive (start, end) over task indices per suite (e.g. 0 9 for all 10).",
    )
    ap.add_argument("--num-episodes", type=int, default=3, help="Init states per task.")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override env step cap. If unset, uses per-suite default (spatial=220, object=280, goal=300, 10=520).",
    )
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "ttnn", "ttnn_1x8"])
    ap.add_argument(
        "--replan-steps",
        type=int,
        default=10,
        help="Replan a new action chunk every N env steps (openpi default = 5; "
        "lower = more responsive but more CPU inference per episode).",
    )
    ap.add_argument(
        "--steps-sweep",
        type=int,
        nargs="+",
        default=[10, 4],
        help="Denoising step counts to sweep over.",
    )
    ap.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="If set, write one mp4 per episode. Layout: <video-dir>/N{N}/<suite>/<file>.mp4 .",
    )
    ap.add_argument(
        "--video-fps",
        type=int,
        default=20,
        help="Playback FPS for episode videos (sim runs at 20 Hz; set 10 for slow-mo).",
    )
    ap.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TTNN device id to use (--backend ttnn only). Default 0; use a high "
        "id (e.g. 31) to avoid colliding with another session on the same host. "
        "When TT_VISIBLE_DEVICES=<phys_id> is set, this is the logical index "
        "after filtering (typically 0).",
    )
    ap.add_argument(
        "--action-horizon",
        type=int,
        default=50,
        help="Action chunk size = number of actions the model predicts per call. "
        "Default 50 (lerobot pi05_libero_finetuned). Pass --action-horizon 10 for the "
        "upstream openpi pi05_libero, which was trained with 10; using the wrong value "
        "pulls untrained position embeddings and the policy outputs garbage.",
    )
    ap.add_argument(
        "--state-in-prompt",
        choices=["true", "false"],
        default="true",
        help="Whether to embed robot state into the language prompt as 256 discrete "
        "bins. Default 'true' (lerobot pi05_libero_finetuned convention, MEAN_STD). "
        "Pass 'false' for the upstream openpi pi05_libero, which was trained with "
        "discrete_state_input=False and never saw state in the prompt.",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="SUITE:IDX",
        help="Explicit list of (suite, task_idx) pairs to run, e.g. "
        "'libero_spatial:4 libero_object:6 libero_object:9'. Overrides "
        "--suites/--task-range entirely. Useful for re-running only "
        "previously-failed tasks for fast A/B iteration on policy changes.",
    )
    args = ap.parse_args()
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)
        import imageio  # noqa: F401  (validate available before the rollout)

    # --tasks (explicit SUITE:IDX list) overrides --suites/--task-range entirely.
    explicit_tasks: List[Tuple[str, int]] = []
    if args.tasks:
        for entry in args.tasks:
            if ":" not in entry:
                raise ValueError(f"--tasks entries must be SUITE:IDX, got {entry!r}")
            suite_name, idx_str = entry.split(":", 1)
            explicit_tasks.append((suite_name.strip(), int(idx_str)))
        suites = sorted({s for s, _ in explicit_tasks}, key=[s for s, _ in explicit_tasks].index)
        task_idxs = None  # not used when explicit_tasks set
    else:
        suites = args.suites if args.suites else [args.suite]
        if args.task_range is not None:
            task_idxs = list(range(args.task_range[0], args.task_range[1] + 1))
        else:
            task_idxs = [args.task_idx]
    if explicit_tasks:
        print(f"\n📦 Plan: explicit tasks={explicit_tasks}, N={args.steps_sweep}, eps/task={args.num_episodes}")
        print(f"   total episodes = {len(explicit_tasks) * len(args.steps_sweep) * args.num_episodes}")
    else:
        print(f"\n📦 Plan: suites={suites}, tasks={task_idxs}, N={args.steps_sweep}, eps/task={args.num_episodes}")
        print(f"   total episodes = {len(suites) * len(task_idxs) * len(args.steps_sweep) * args.num_episodes}")
    print(f"   action_horizon={args.action_horizon}  " f"state_in_prompt={args.state_in_prompt}")

    print(f"\n📋 Loading PI0.5 LIBERO adapter (backend={args.backend}) from {args.checkpoint}")
    t0 = time.time()
    ttnn_device = None
    mesh_ctx = None
    mesh_handles = None
    if args.backend == "ttnn":
        import ttnn

        ttnn_device = ttnn.open_device(
            device_id=args.device_id,
            l1_small_size=24576,
            trace_region_size=134_217_728,
        )
        print(f"   ttnn device opened in {time.time() - t0:.1f}s (device_id={args.device_id})")
    elif args.backend == "ttnn_1x8":
        # 1×8-specific env vars for the ttnn_1x8 backend. These are NOT in pi05_production.env
        # — they're pipeline_1x8-specific and control how attention/MLP
        # weights shard across the 8-chip mesh. Without these, wqkv loads
        # un-sharded (N=2560 instead of N=768/chip) and the prefill QKV
        # matmul kernel-config invariant fails. setdefault preserves any
        # explicit shell override.
        for _k, _v in {
            "PI0_TP": "8",
            "PI0_TP8_ATTN_HEADPAR": "1",
            "PI0_MLP_BS": "1",
            "PI0_MLP_FUSED_RS": "0",
        }.items():
            os.environ.setdefault(_k, _v)

        from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp8_mesh

        mesh_ctx = open_prefill_tp8_mesh(
            tp=8,
            l1_small_size=24576,
            # 256 MiB: per-task trace re-capture needs >128 MiB for some tasks
            # (e.g. libero_object task 7's trace measured ~142 MiB).
            trace_region_size=256 * 1024 * 1024,
            num_command_queues=2,
        )
        mesh_handles = mesh_ctx.__enter__()
        print(f"   ttnn_1x8 mesh opened in {time.time() - t0:.1f}s (8 chips, 2 CQs)")
    adapter = Pi0_5LiberoAdapter(
        args.checkpoint,
        backend=args.backend,
        ttnn_device=ttnn_device,
        mesh_handles=mesh_handles,
        chunk_size=args.action_horizon,
        action_horizon=args.action_horizon,
        state_in_prompt=(args.state_in_prompt == "true"),
    )
    print(
        f"   adapter loaded in {time.time() - t0:.1f}s "
        f"(real action dim = {adapter.real_action_dim}, state = {adapter.real_state_dim})"
    )

    # results[N][(suite, task_idx)] = list of per-ep stats dicts (no frames)
    results: Dict[int, Dict[Tuple[str, int], List[dict]]] = {N: {} for N in args.steps_sweep}
    try:
        for N in args.steps_sweep:
            print(f"\n{'#' * 72}\n#  N={N} denoise steps\n{'#' * 72}")
            # Build (suite, task_idx) sequence: explicit pairs (--tasks) or
            # cross-product of suites × task_idxs.
            if explicit_tasks:
                pairs = list(explicit_tasks)
            else:
                pairs = [(s, t) for s in suites for t in task_idxs]
            for suite_name, task_idx in pairs:
                max_steps = args.max_steps if args.max_steps else SUITE_MAX_STEPS.get(suite_name, 220)
                print(f"\n🤖 {suite_name} / task {task_idx}  (max_steps={max_steps})")
                env, task, initial_states = make_libero_env(suite_name, task_idx)
                task_desc = task.language
                print(f"   task: {task_desc!r}")
                stats = []
                try:
                    for ep in range(args.num_episodes):
                        t_ep = time.time()
                        m = run_episode(
                            env,
                            adapter,
                            task_desc,
                            N,
                            max_steps=max_steps,
                            chunk_action_horizon=args.replan_steps,
                            initial_state=initial_states[ep % len(initial_states)],
                            seed=ep,
                            record_frames=bool(args.video_dir),
                        )
                        stats.append({k: v for k, v in m.items() if k != "frames"})
                        if args.video_dir and m.get("frames"):
                            import imageio

                            video_subdir = os.path.join(args.video_dir, f"N{N}", suite_name)
                            os.makedirs(video_subdir, exist_ok=True)
                            suffix = "success" if m["success"] else "failure"
                            safe_task = "".join(c if c.isalnum() else "_" for c in task_desc)[:60]
                            fname = f"task{task_idx:02d}_ep{ep+1:02d}_{safe_task}_{suffix}.mp4"
                            fpath = os.path.join(video_subdir, fname)
                            imageio.mimwrite(fpath, m["frames"], fps=args.video_fps, codec="libx264")
                        print(
                            f"   ep {ep+1}: success={m['success']} steps={m['steps']} "
                            f"avg_chunk={1000*m['avg_chunk_pred_time']:.0f}ms "
                            f"wall={time.time()-t_ep:.1f}s",
                            flush=True,
                        )
                finally:
                    env.close()
                results[N][(suite_name, task_idx)] = stats

        # Summary
        print("\n" + "=" * 84)
        print(f"  LIBERO ROLLOUT FULL SUMMARY — backend={args.backend}, replan={args.replan_steps}")
        print(f"  suites={suites}, tasks={task_idxs}, init states/task={args.num_episodes}")
        print("=" * 84)
        for N in args.steps_sweep:
            print(f"\n  --- N = {N} ---")
            suite_totals: Dict[str, Tuple[int, int]] = {}
            for (suite_name, task_idx), stats in results[N].items():
                succ = sum(s["success"] for s in stats)
                tot = len(stats)
                avg_steps = float(np.mean([s["steps"] for s in stats])) if stats else 0
                avg_chunk = float(np.mean([s["avg_chunk_pred_time"] for s in stats])) if stats else 0
                print(
                    f"  {suite_name:18s} task {task_idx:2d}:  {succ}/{tot}  "
                    f"avg_steps={avg_steps:5.1f}  avg_chunk={1000*avg_chunk:4.0f}ms"
                )
                sucs, tots = suite_totals.get(suite_name, (0, 0))
                suite_totals[suite_name] = (sucs + succ, tots + tot)
            print(f"  {'-' * 80}")
            grand_s, grand_t = 0, 0
            for suite_name in suites:
                sucs, tots = suite_totals.get(suite_name, (0, 0))
                pct = 100.0 * sucs / max(tots, 1)
                print(f"  {suite_name:18s} TOTAL    :  {sucs}/{tots} ({pct:.1f}%)")
                grand_s += sucs
                grand_t += tots
            grand_pct = 100.0 * grand_s / max(grand_t, 1)
            print(f"  {'GRAND TOTAL':18s}          :  {grand_s}/{grand_t} ({grand_pct:.1f}%)")
        print("=" * 84)
    finally:
        pass
    if ttnn_device is not None:
        import ttnn

        ttnn.close_device(ttnn_device)
    if mesh_ctx is not None:
        mesh_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
