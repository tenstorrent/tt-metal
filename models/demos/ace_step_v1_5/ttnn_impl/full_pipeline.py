from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

import ttnn

from .dit_decoder_core import AceStepDecoderConfigTTNN, TtAceStepDiTCore, TtTimestepEmbedding
from .output_head import TtAceStepDiTOutputHead
from .patchify import TtAceStepPatchEmbed1D
from .safetensors_loader import load_safetensors_state_dict

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import torch


@dataclass(frozen=True)
class AceStepV15LikeConfig:
    """
    Subset of `AceStepConfig` needed by the TTNN patchify/output head.
    """

    patch_size: int
    in_channels: int
    hidden_size: int
    audio_acoustic_hidden_dim: int
    rms_norm_eps: float


class AceStepV15TTNNPipeline:
    """
    TTNN-only DiT decoder path (subset of HF ``AceStepDiTModel``).

    Wired on device:
      ``hidden_states [B,T,C]`` → ``proj_in`` (patch embed) → ``TtAceStepDiTCore`` (layers +
      ``condition_embedder``) → output head (``norm_out`` + ``scale_shift_table`` + ``proj_out``).

    Hugging Face parity (important):
      The TTNN core aims to mirror HF layer math (AdaLN scale/shift/gates, Qwen3 MLP, GQA, SDPA),
      but it is **not** guaranteed bit-accurate vs ``modeling_acestep_v15_*.py``. Known gaps include:
      - **Numerics**: TTNN BF16 + fused SDPA can differ slightly from Torch kernels.
      - **Masking/packing**: full HF `create_4d_mask` / packed-sequence behaviors require the caller to
        provide equivalent 1D masks (this pipeline can build additive 4D masks from them).
      - **TimestepEmbedding**: HF computes sinusoidal embeddings at runtime; TTNN uses a **fixed
        lookup table** over ``timesteps_host`` passed at init (must cover indices used in ``forward``).

    Host→device: weight upload in ``__init__``. Device→host: left to the caller (e.g. demo saves ``.npy``).
    """

    def __init__(
        self,
        *,
        device: ttnn.Device,
        checkpoint_safetensors_path: str,
        decoder_prefix: str = "decoder.",
        activation_dtype: ttnn.DataType | None = None,
        weights_dtype: ttnn.DataType | None = None,
        timesteps_host: Optional["np.ndarray"] = None,
        expected_input_length: int | None = None,
    ) -> None:
        self.device = device
        if activation_dtype is None:
            activation_dtype = getattr(ttnn, "bfloat16", None)
        if weights_dtype is None:
            weights_dtype = getattr(ttnn, "bfloat16", None)
        if activation_dtype is None or weights_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16 dtype; pass activation_dtype/weights_dtype explicitly.")
        self.activation_dtype = activation_dtype
        self.weights_dtype = weights_dtype

        # Load weights on host, then transfer once per weight tensor during module init.
        sd = load_safetensors_state_dict(checkpoint_safetensors_path, prefix=decoder_prefix).tensors

        # Config values are read from config.json in ACE-Step; for now infer minimal subset from tensors where possible.
        # proj_in weight: [hidden_size, in_channels, patch_size]
        proj_in_w = sd.get("proj_in.1.weight")
        if proj_in_w is None:
            proj_in_w = sd.get("proj_in.weight")
        if proj_in_w is None:
            raise KeyError(
                "Missing decoder proj_in weight in safetensors (expected proj_in.1.weight or proj_in.weight)"
            )
        hidden_size, in_channels, patch_size = map(int, proj_in_w.shape)

        # proj_out convtranspose weight: [hidden_size, audio_acoustic_hidden_dim, patch_size]
        proj_out_w = sd.get("proj_out.1.weight")
        if proj_out_w is None:
            proj_out_w = sd.get("proj_out.weight")
        if proj_out_w is None:
            raise KeyError(
                "Missing decoder proj_out weight in safetensors (expected proj_out.1.weight or proj_out.weight)"
            )
        _in_ch, audio_acoustic_hidden_dim, patch_size2 = map(int, proj_out_w.shape)
        if _in_ch != hidden_size or patch_size2 != patch_size:
            raise ValueError(
                f"Unexpected proj_out shape {_in_ch,audio_acoustic_hidden_dim,patch_size2} vs expected "
                f"({hidden_size}, *, {patch_size})"
            )

        norm_w = sd.get("norm_out.weight")
        if norm_w is None:
            raise KeyError("Missing decoder norm_out.weight in safetensors")

        ckpt_parent = Path(checkpoint_safetensors_path).resolve().parent
        hf_cfg_path = ckpt_parent / "config.json"
        hf_cfg: dict = {}
        if hf_cfg_path.is_file():
            hf_cfg = json.loads(hf_cfg_path.read_text())
        rms_eps = float(hf_cfg.get("rms_norm_eps", 1e-6))
        head_dim = int(hf_cfg.get("head_dim", 128))
        sw = hf_cfg.get("sliding_window", None)
        sliding_window = int(sw) if sw is not None else None
        layer_types = hf_cfg.get("layer_types", None)
        max_position_embeddings = int(hf_cfg.get("max_position_embeddings", 4096))
        rope_theta = float(hf_cfg.get("rope_theta", 10000.0))

        cfg = AceStepV15LikeConfig(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            audio_acoustic_hidden_dim=audio_acoustic_hidden_dim,
            rms_norm_eps=rms_eps,
        )

        self.patch_embed = TtAceStepPatchEmbed1D(
            config=cfg,
            state_dict=sd,
            base_address="proj_in",
            device=device,
            expected_input_length=expected_input_length,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )

        self.output_head = TtAceStepDiTOutputHead(
            config=cfg,
            state_dict=sd,
            base_address="",  # decoder.* prefix was stripped
            device=device,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )

        # Decoder core config. Values are derived from checkpoint tensors.
        import numpy as np

        # q_proj weight: [H*Dh, D]
        q_w = sd["layers.0.self_attn.q_proj.weight"]
        hidden = int(q_w.shape[1])
        num_attention_heads = int(q_w.shape[0]) // head_dim
        kv_w = sd["layers.0.self_attn.k_proj.weight"]
        num_kv_heads = int(kv_w.shape[0]) // head_dim
        num_layers = sum(1 for k in sd.keys() if k.startswith("layers.") and k.endswith(".self_attn.q_proj.weight"))

        core_cfg = AceStepDecoderConfigTTNN(
            hidden_size=hidden,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            sliding_window=sliding_window,
            layer_types=layer_types,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

        # Default to 8-step schedule if not provided. Caller should pass the exact sampler schedule.
        if timesteps_host is None:
            timesteps_host = np.linspace(1.0, 0.0, num=8, dtype=np.float32)
        else:
            timesteps_host = np.asarray(timesteps_host, dtype=np.float32)
        # Keep an explicit 0.0 entry for `time_embed_r(t - r)` where r==t -> 0.
        # (The `time_embed` lookup path indexes into this table; the `time_embed_r` path may also
        # use it to derive (t - r) in host units.)
        if timesteps_host.size == 0 or float(timesteps_host[-1]) != 0.0:
            timesteps_host = np.concatenate([timesteps_host, np.asarray([0.0], dtype=np.float32)], axis=0)
        self.timesteps_host = timesteps_host
        self._zero_timestep_index = int(timesteps_host.size - 1)

        self.time_embed = TtTimestepEmbedding(
            cfg=core_cfg,
            state_dict=sd,
            base_address="time_embed",
            mesh_device=device,
            timesteps_host=timesteps_host,
            dtype=self.activation_dtype,
        )
        self.time_embed_r = TtTimestepEmbedding(
            cfg=core_cfg,
            state_dict=sd,
            base_address="time_embed_r",
            mesh_device=device,
            timesteps_host=timesteps_host,
            dtype=self.activation_dtype,
        )

        self.core = TtAceStepDiTCore(cfg=core_cfg, state_dict=sd, mesh_device=device, dtype=self.activation_dtype)
        self.cond_dim = int(sd["condition_embedder.weight"].shape[1])

        # Cache for host-built additive masks uploaded to device.
        # Keyed by (kind, B, S_q, S_k) for the common all-ones mask cases.
        self._mask_cache: dict[tuple[str, int, int, int], ttnn.Tensor] = {}

    def _to_1d_keep_numpy(self, mask_1d, *, expected_batch: int | None = None) -> np.ndarray:
        """
        Normalize a 1D mask to a NumPy bool array shaped [B, S] where True means keep.

        Accepts:
        - torch.Tensor on CPU/GPU
        - numpy arrays / array-likes
        - None
        """
        if mask_1d is None:
            raise ValueError("mask_1d must be non-None")

        # Torch support (optional dependency in some environments).
        try:
            import torch  # type: ignore

            if isinstance(mask_1d, torch.Tensor):
                m = mask_1d.detach()
                if m.dtype != torch.bool:
                    # HF sometimes uses float masks (1.0 keep / 0.0 pad)
                    m = m > 0
                m = m.to(device="cpu")
                arr = m.numpy().astype(np.bool_)
            else:
                arr = np.asarray(mask_1d)
        except Exception:
            arr = np.asarray(mask_1d)

        if arr.ndim != 2:
            raise ValueError(f"Expected 1D mask shaped [B,S], got shape={arr.shape}")
        if expected_batch is not None and int(arr.shape[0]) != int(expected_batch):
            raise ValueError(f"Mask batch mismatch: expected B={expected_batch}, got {arr.shape[0]}")
        return arr.astype(np.bool_)

    @staticmethod
    def _logical_seq_len_dim1(t) -> int:
        """
        Sequence length on dim 1 for rank-3 tensors ``[B, S, D]``.

        TTNN TILE tensors often expose a tile-padded ``shape``; SDPA validates masks against
        ``q.logical_shape()``. Using the logical length avoids "Mask sequence length must match Q".
        """
        fn = getattr(t, "logical_shape", None)
        if callable(fn):
            try:
                s = fn()
                if s is not None and len(s) >= 2:
                    return int(s[1])
            except Exception:
                pass
        try:
            ls = getattr(t, "logical_shape", None)
            if ls is not None and not callable(ls) and len(ls) >= 2:
                return int(ls[1])
        except Exception:
            pass
        return int(t.shape[1])

    def _patchify_1d_keep(self, keep_bt: np.ndarray, *, patch_size: int, pad_length: int) -> np.ndarray:
        """
        Convert a frame-level keep mask [B,T] to patch-level [B,T_p] using OR over each patch.
        Mirrors the fact that proj_in patchifies the time axis by stride==patch_size.
        """
        if keep_bt.ndim != 2:
            raise ValueError(f"Expected keep mask [B,T], got shape={keep_bt.shape}")
        if pad_length:
            keep_bt = np.pad(keep_bt, ((0, 0), (0, int(pad_length))), constant_values=False)
        t = int(keep_bt.shape[1])
        if t % int(patch_size) != 0:
            raise ValueError("Internal error: keep_bt must be padded to a multiple of patch_size")
        t_p = t // int(patch_size)
        return keep_bt.reshape(int(keep_bt.shape[0]), t_p, int(patch_size)).any(axis=2)

    def _build_sdpa_mask_b1qk(
        self,
        *,
        keep_q: np.ndarray | None,
        keep_k: np.ndarray | None,
        b: int,
        s_q: int,
        s_k: int,
    ) -> np.ndarray:
        """
        Build TTNN SDPA **additive** mask [B,1,S_q,S_k] (PyTorch / sdpa.cpp): logits get +mask
        before softmax; use 0 on allowed (q,k) pairs and a large negative on masked pairs.
        """
        if keep_k is None:
            k_ok = np.ones((int(b), int(s_k)), dtype=np.bool_)
        else:
            if keep_k.shape != (int(b), int(s_k)):
                raise ValueError(f"keep_k shape mismatch: got {keep_k.shape}, expected {(b, s_k)}")
            k_ok = keep_k
        if keep_q is None:
            q_ok = np.ones((int(b), int(s_q)), dtype=np.bool_)
        else:
            if keep_q.shape != (int(b), int(s_q)):
                raise ValueError(f"keep_q shape mismatch: got {keep_q.shape}, expected {(b, s_q)}")
            q_ok = keep_q
        ok = q_ok[:, :, None] & k_ok[:, None, :]
        neg = np.float32(-1e9)
        return np.where(ok[:, None, :, :], np.float32(0.0), neg).astype(np.float32)

    def forward(
        self,
        *,
        # HF semantics: model input is concat(context_latents, xt) before proj_in.
        # Prefer passing `xt_bt64` + `context_latents_bt128`; `hidden_states_btC` is a legacy escape hatch.
        xt_bt64: ttnn.Tensor | None = None,
        context_latents_bt128: ttnn.Tensor | None = None,
        hidden_states_btC: ttnn.Tensor | None = None,
        timestep_index: int,
        timestep_r_index: int | None = None,
        encoder_hidden_states_btd: ttnn.Tensor,
        attention_mask_1d_bt=None,
        encoder_attention_mask_1d_bk=None,
        encoder_attention_mask_b1qk: ttnn.Tensor | None = None,
        debug_intermediates: dict | None = None,
    ) -> ttnn.Tensor:
        """
        Args:
            xt_bt64: [B, T, 64] noisy latents (device tensor, ROW_MAJOR preferred).
            context_latents_bt128: [B, T, 128] (src_latents + chunk_masks) from HF `prepare_condition`.
            hidden_states_btC: [B, T, in_channels] legacy path (already concatenated).
            attention_mask_1d_bt: Optional [B, T] keep mask (bool or 1/0) at **frame** length.
                If provided, it is patchified to [B, T_p] and used for self-attention padding parity.
            encoder_attention_mask_1d_bk: Optional [B, S_enc] keep mask (bool or 1/0) for cross-attention keys.
            temb_bD: [B, hidden_size] device tensor.
        Returns:
            acoustic features [B, T, audio_acoustic_hidden_dim] on device.
        """
        debug = bool(os.environ.get("ACE_STEP_DEBUG_PIPELINE"))
        debug_stats = bool(os.environ.get("ACE_STEP_DEBUG_PIPELINE_STATS"))

        if (hidden_states_btC is None) == (xt_bt64 is None or context_latents_bt128 is None):
            raise ValueError(
                "Pass either `hidden_states_btC` (already concat'd) OR both `xt_bt64` and `context_latents_bt128`."
            )

        if hidden_states_btC is None:
            if hasattr(ttnn, "concat"):
                hidden_states_btC = ttnn.concat([context_latents_bt128, xt_bt64], dim=-1)
            else:
                hidden_states_btC = ttnn.concatenate([context_latents_bt128, xt_bt64], dim=-1)

        if debug:
            try:
                print(
                    "[ace_step_v1_5][pipe] "
                    f"in_hidden_states shape={tuple(hidden_states_btC.shape)} "
                    f"enc_shape={tuple(encoder_hidden_states_btd.shape)} "
                    f"timestep_index={int(timestep_index)}",
                    flush=True,
                )
            except Exception:
                pass

        patches, meta = self.patch_embed.forward(hidden_states_btC)
        if debug_intermediates is not None and debug_intermediates.get("enabled", False):
            debug_intermediates["patchify_out"] = patches
            debug_intermediates["pipe.patches"] = patches
        if debug:
            try:
                print(
                    "[ace_step_v1_5][pipe] "
                    f"patches shape={tuple(patches.shape)} "
                    f"meta(original_seq_len={meta.original_seq_len}, pad_length={meta.pad_length}, patch_size={meta.patch_size})",
                    flush=True,
                )
            except Exception:
                pass

        # Encoder cross-attention: additive mask [B,1,S_q,S_k_pad] (0 keep, -1e9 masked).
        # When every **logical** encoder key is valid, omit the mask here — the TTNN attention op
        # pads K/V after RMSNorm and applies a tail-only additive mask for the tile pad (258→288).
        # When some logical keys are invalid (HF padding in the packed sequence), extend keep_k to
        # S_k_pad and upload a full mask; the core **adds** its tail mask on top.
        if encoder_attention_mask_b1qk is None and encoder_attention_mask_1d_bk is not None:
            b = int(hidden_states_btC.shape[0])
            s_q = int(self._logical_seq_len_dim1(patches))
            s_k = int(self._logical_seq_len_dim1(encoder_hidden_states_btd))
            s_k_pad = ((int(s_k) + 31) // 32) * 32
            keep_k_logical = self._to_1d_keep_numpy(encoder_attention_mask_1d_bk, expected_batch=b)
            if int(keep_k_logical.shape[1]) != int(s_k):
                raise ValueError(
                    f"encoder_attention_mask_1d_bk length mismatch: got S_k={keep_k_logical.shape[1]}, expected {s_k}"
                )
            if bool(np.all(keep_k_logical)):
                encoder_attention_mask_b1qk = None
            else:
                keep_k = keep_k_logical
                if s_k_pad != s_k:
                    keep_k = np.pad(keep_k, ((0, 0), (0, int(s_k_pad - s_k))), constant_values=False)
                m_np = self._build_sdpa_mask_b1qk(keep_q=None, keep_k=keep_k, b=b, s_q=s_q, s_k=s_k_pad)
                encoder_attention_mask_b1qk = ttnn.as_tensor(
                    m_np,
                    device=self.device,
                    dtype=self.activation_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
                )

        # Self-attention padding mask (patchified) will be threaded through the core in the next step.
        self_attention_mask_b1qq = None
        if attention_mask_1d_bt is not None:
            b = int(hidden_states_btC.shape[0])
            keep_t = self._to_1d_keep_numpy(attention_mask_1d_bt, expected_batch=b)
            # Patchify to match `patches` sequence length (T_p).
            keep_p = self._patchify_1d_keep(keep_t, patch_size=int(meta.patch_size), pad_length=int(meta.pad_length))
            patch_seq = int(self._logical_seq_len_dim1(patches))
            if int(keep_p.shape[1]) != patch_seq:
                raise ValueError(
                    f"Patchified attention_mask length mismatch: got {keep_p.shape[1]} vs patches {patch_seq}"
                )
            s_q = patch_seq
            if bool(keep_p.all()):
                self_attention_mask_b1qq = None
            else:
                m_np = self._build_sdpa_mask_b1qk(keep_q=keep_p, keep_k=keep_p, b=b, s_q=s_q, s_k=s_q)
                self_attention_mask_b1qq = ttnn.as_tensor(
                    m_np,
                    device=self.device,
                    dtype=self.activation_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
                )

        # timestep embeddings
        temb_t, tp_t = self.time_embed(int(timestep_index))
        # Upstream/HF semantics: `time_embed_r(timestep - timestep_r)`.
        # Our `TtTimestepEmbedding` supports computing an embedding for an arbitrary timestep value
        # via `from_timestep_value`, which avoids incorrectly indexing the 0-slot for nonzero (t - r).
        if timestep_r_index is None:
            delta_tr = 0.0
        else:
            delta_tr = float(self.timesteps_host[int(timestep_index)]) - float(
                self.timesteps_host[int(timestep_r_index)]
            )
        temb_r, tp_r = self.time_embed_r.from_timestep_value(delta_tr)
        temb = ttnn.add(temb_t, temb_r)  # [1,D]
        timestep_proj = ttnn.add(tp_t, tp_r)  # [1,6,D]
        if debug_intermediates is not None and debug_intermediates.get("enabled", False):
            debug_intermediates["temb"] = temb
            debug_intermediates["timestep_proj_b6d"] = timestep_proj
            debug_intermediates["pipe.temb"] = temb
            debug_intermediates["pipe.timestep_proj_b6d"] = timestep_proj
        if debug:
            try:
                print(
                    "[ace_step_v1_5][pipe] "
                    f"temb shape={tuple(temb.shape)} timestep_proj shape={tuple(timestep_proj.shape)}",
                    flush=True,
                )
            except Exception:
                pass

        # Expand to batch (needed for CFG where we run B=2 with [cond, uncond]).
        B = int(hidden_states_btC.shape[0])
        if B != 1:
            if int(temb.shape[0]) != 1 or int(timestep_proj.shape[0]) != 1:
                raise ValueError(
                    f"Expected temb/timestep_proj to have batch==1 before replication, got "
                    f"temb={tuple(temb.shape)} timestep_proj={tuple(timestep_proj.shape)}"
                )

            def _replicate_batch(x, batch: int):
                if int(x.shape[0]) == batch:
                    return x
                xs = [x] * int(batch)
                if hasattr(ttnn, "concat"):
                    return ttnn.concat(xs, dim=0)
                return ttnn.concatenate(xs, dim=0)

            temb = _replicate_batch(temb, B)  # [B,D]
            timestep_proj = _replicate_batch(timestep_proj, B)  # [B,6,D]

        # Core expects timestep projection in row-major for stable broadcast with scale_shift_table.
        timestep_proj = ttnn.to_layout(timestep_proj, ttnn.ROW_MAJOR_LAYOUT)

        patches_out = self.core(
            patches,
            timestep_proj,
            encoder_hidden_states_btd,
            encoder_attention_mask_b1qk,
            self_attention_mask_b1qq,
            debug=debug_intermediates,
        )
        if debug_intermediates is not None and debug_intermediates.get("enabled", False):
            debug_intermediates["core_out"] = patches_out
            debug_intermediates["pipe.core_out"] = patches_out
        if debug:
            try:
                print("[ace_step_v1_5][pipe] core_out shape=" + str(tuple(patches_out.shape)), flush=True)
            except Exception:
                pass

        acoustic = self.output_head.forward(patches_out, temb, meta, debug=debug_intermediates)
        if debug_intermediates is not None and debug_intermediates.get("enabled", False):
            debug_intermediates["acoustic_out"] = acoustic
        if debug:
            try:
                print("[ace_step_v1_5][pipe] acoustic_out shape=" + str(tuple(acoustic.shape)), flush=True)
            except Exception:
                pass

        if debug_stats:
            # Expensive: device->host. Use only for debugging.
            try:
                import torch

                x = ttnn.to_torch(acoustic).to(torch.float32).cpu()
                finite = torch.isfinite(x)
                xf = x[finite] if finite.any() else x.reshape(-1)
                print(
                    "[ace_step_v1_5][pipe][stats] "
                    f"acoustic min={float(xf.min()):.4g} max={float(xf.max()):.4g} "
                    f"mean={float(xf.mean()):.4g} std={float(xf.std(unbiased=False)):.4g}",
                    flush=True,
                )
            except Exception:
                pass
        return acoustic


def _find_model_safetensors(snapshot_root: Path) -> Path:
    """Return ``model.safetensors`` under a HF snapshot (repo root or nested)."""
    direct = snapshot_root / "model.safetensors"
    if direct.is_file():
        return direct
    candidates = sorted(snapshot_root.rglob("model.safetensors"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        rel = [str(p.relative_to(snapshot_root)) for p in candidates[:12]]
        raise RuntimeError(
            "Multiple model.safetensors files under this HF snapshot. "
            "Pass --checkpoint-safetensors explicitly, or set --hf-subfolder to pick one variant "
            f"(e.g. acestep-v15-turbo). Found:\n  " + "\n  ".join(rel)
        )
    raise FileNotFoundError(f"No model.safetensors under HF snapshot: {snapshot_root}")


def resolve_acestep_decoder_checkpoint(
    *,
    checkpoint_safetensors: str | None,
    hf_repo_id: str,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    hf_subfolder: str | None = None,
) -> str:
    """
    Resolve decoder weights path.

    If ``checkpoint_safetensors`` is set, return it. Otherwise download ``hf_repo_id``
    (default: ACE-Step **Base**) via huggingface_hub and return ``model.safetensors``.
    """
    if checkpoint_safetensors:
        p = Path(checkpoint_safetensors).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return str(p.resolve())

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Missing huggingface_hub. Install it or pass --checkpoint-safetensors explicitly.") from e

    snap = Path(
        snapshot_download(
            repo_id=hf_repo_id,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")),
        )
    )
    root = snap / hf_subfolder if hf_subfolder else snap
    if hf_subfolder and not root.is_dir():
        raise FileNotFoundError(f"--hf-subfolder does not exist in snapshot: {root}")
    return str(_find_model_safetensors(root).resolve())


def _load_torch_tensor(path: str) -> "torch.Tensor":
    import torch

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        # Common HF / training checkpoints: look for a single tensor value.
        if "tensor" in obj and isinstance(obj["tensor"], torch.Tensor):
            return obj["tensor"]
        tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
    raise TypeError(f"Unsupported torch.load() payload in {path!r}: expected Tensor or dict containing a Tensor")


def _main() -> int:
    """
    Minimal "noise -> corrected features" demo.

    Important: this produces **acoustic feature tensors** (the DiT decoder output), not a waveform.
    Converting those features into audio requires the downstream decoder/vocoder stages, which are
    not implemented in this TTNN demo folder.
    """
    import argparse

    import numpy as np
    import torch

    ap = argparse.ArgumentParser(description="ACE-Step v1.5 TTNN demo: denoise acoustic latent/features.")
    ap.add_argument(
        "--checkpoint-safetensors",
        default=None,
        help=(
            "Path to decoder ``model.safetensors``. If omitted, downloads HF repo "
            "``--hf-repo-id`` (default: ACE-Step/acestep-v15-base)."
        ),
    )
    ap.add_argument(
        "--hf-repo-id",
        default="ACE-Step/acestep-v15-base",
        help="Hugging Face repo for decoder weights when --checkpoint-safetensors is omitted.",
    )
    ap.add_argument("--hf-revision", default=None, help="Optional HF revision (branch/tag/commit).")
    ap.add_argument("--hf-cache-dir", default=None, help="Optional HF cache directory override.")
    ap.add_argument(
        "--hf-subfolder",
        default=None,
        help=(
            "When downloading an umbrella repo (e.g. ACE-Step/Ace-Step1.5), select the DiT variant folder "
            "(e.g. acestep-v15-turbo). Ignored for single-checkpoint repos like ACE-Step/acestep-v15-base."
        ),
    )
    ap.add_argument("--timestep-index", type=int, default=0, help="Index into the precomputed timestep table.")
    ap.add_argument(
        "--noise-pt",
        default=None,
        help="Optional path to a torch tensor .pt/.pth containing noisy input [T,C] or [B,T,C]. If omitted, random noise is used.",
    )
    ap.add_argument(
        "--encoder-hidden-states-pt",
        default=None,
        help="Optional torch tensor .pt/.pth for conditioning [B,S_enc,cond_dim]. If omitted, zeros are used.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed used when generating random noise/conditioning.")
    ap.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length T for generated noise (if --noise-pt omitted)."
    )
    ap.add_argument("--batch", type=int, default=1, help="Batch size B for generated noise (if --noise-pt omitted).")
    ap.add_argument(
        "--out-npy",
        required=True,
        help="Output path for the corrected acoustic features as a NumPy .npy array (written on host).",
    )
    args = ap.parse_args()

    ckpt_path = resolve_acestep_decoder_checkpoint(
        checkpoint_safetensors=args.checkpoint_safetensors,
        hf_repo_id=args.hf_repo_id,
        hf_revision=args.hf_revision,
        hf_cache_dir=args.hf_cache_dir,
        hf_subfolder=args.hf_subfolder,
    )
    print(f"[ace_step_v1_5] using decoder checkpoint: {ckpt_path}", flush=True)

    if not hasattr(ttnn, "open_device"):
        raise RuntimeError("This demo requires a TTNN runtime with device support (ttnn.open_device).")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ttnn.open_device(device_id=0, trace_region_size=128 << 20)
    try:
        if hasattr(device, "enable_program_cache"):
            device.enable_program_cache()

        pipe = AceStepV15TTNNPipeline(device=device, checkpoint_safetensors_path=ckpt_path)

        # Load or generate noisy input (host -> device once).
        if args.noise_pt is None:
            B = int(args.batch)
            T = int(args.seq_len)
            C = int(pipe.patch_embed.in_channels)
            noise = torch.randn((B, T, C), dtype=torch.bfloat16)
        else:
            noise = _load_torch_tensor(args.noise_pt).to(torch.bfloat16)
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)  # [1,T,C]
            if noise.ndim != 3:
                raise ValueError(f"--noise-pt must be [T,C] or [B,T,C], got shape {tuple(noise.shape)}")

        # Load or generate conditioning encoder_hidden_states (host -> device once).
        if args.encoder_hidden_states_pt is None:
            B = int(noise.shape[0])
            S_enc = 1
            enc = torch.zeros((B, S_enc, int(pipe.cond_dim)), dtype=torch.bfloat16)
        else:
            enc = _load_torch_tensor(args.encoder_hidden_states_pt).to(torch.bfloat16)
            if enc.ndim != 3:
                raise ValueError(f"--encoder-hidden-states-pt must be [B,S_enc,cond_dim], got shape {tuple(enc.shape)}")

        noise_tt = ttnn.from_torch(noise, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        enc_tt = ttnn.from_torch(enc, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Single-step "correction": feed noisy features + conditioning at the chosen timestep.
        out_tt = pipe.forward(
            hidden_states_btC=noise_tt, timestep_index=int(args.timestep_index), encoder_hidden_states_btd=enc_tt
        )

        out = ttnn.to_torch(out_tt).float().cpu().numpy()
        out_path = Path(args.out_npy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), out)
        print(f"[ace_step_v1_5] wrote corrected features: {out_path} shape={out.shape}", flush=True)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
