# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load Hugging Face ``AutoencoderOobleck`` VAE folders and run the decoder on TTNN."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .math_perf_env import (
    ace_step_concat_kwargs,
    ace_step_flush_device_profiler,
    ace_step_vae_decode_concat_memory_config,
)
from .vae.decoder import TtOobleckDecoder

# TTNN conv_transpose2d width-sharded kernels assert on activation×weight geometry; very short
# latent sequences (common on the last overlap-add tile) trigger TT_FATAL mismatches.
_DEFAULT_MIN_DECODER_LATENT_WINDOW = 32


def _vae_time_parallel_enabled(device: Any) -> bool:
    """Time-parallel VAE decode: on a multi-chip mesh AND ``ACE_STEP_VAE_TIME_PARALLEL`` set.
    Default off → the validated sequential overlap-add path runs unchanged."""
    if os.environ.get("ACE_STEP_VAE_TIME_PARALLEL", "").strip().lower() not in ("1", "on", "true", "yes"):
        return False
    try:
        return int(device.get_num_devices()) > 1
    except Exception:
        return False


def _merge_audio_cores(cores: list, ttnn: Any) -> Any:
    """Merge overlap-add audio tiles; default DRAM output so long clips do not exhaust L1."""
    if len(cores) == 1:
        return cores[0]
    out_mc = ace_step_vae_decode_concat_memory_config(ttnn)
    _ckw = ace_step_concat_kwargs(ttnn, l1_mc=out_mc)
    _concat = ttnn.concat if hasattr(ttnn, "concat") else ttnn.concatenate

    staged: list = []
    for core in cores:
        if out_mc is not None and hasattr(ttnn, "to_memory_config"):
            core = ttnn.to_memory_config(core, out_mc)
        staged.append(core)

    # Single concat is fine for modest tile counts; incremental merge avoids huge L1 temps.
    if len(staged) <= 16:
        merged = _concat(staged, dim=1, **_ckw)
        for core in staged:
            _safe_deallocate(core)
        return merged

    merged = staged[0]
    for core in staged[1:]:
        prev = merged
        merged = _concat([prev, core], dim=1, **_ckw)
        _safe_deallocate(prev)
        _safe_deallocate(core)
    return merged


def _safe_deallocate(t: Any) -> None:
    if t is None:
        return
    try:
        import ttnn

        ttnn.deallocate(t)
    except Exception:
        pass


@dataclass
class _VaeChunkTraceState:
    trace_id: Any
    latent_dev: Any
    latent_host: Any
    output_dev: Any
    op_event: Any
    stage_write_event: Any = None


def _pick_safetensors_file(vae_dir: Path) -> Path:
    for name in (
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "model.fp16.safetensors",
    ):
        cand = vae_dir / name
        if cand.is_file():
            return cand
    all_st = sorted(vae_dir.glob("*.safetensors"))
    if not all_st:
        raise FileNotFoundError(f"No *.safetensors found under {vae_dir}")
    return all_st[0]


def _load_state_dict_torch(path: Path) -> dict[str, Any]:
    from models.experimental.ace_step_v1_5.utils.weight_cache import get_torch_state_dict

    return get_torch_state_dict(str(path), component="vae-safetensors")


class TtOobleckVaeDecoder:
    """Wraps ``TtOobleckDecoder`` built from HF ``vae/`` checkpoints."""

    def __init__(self, decoder: TtOobleckDecoder) -> None:
        self._decoder = decoder
        self._chunk_traces: dict[tuple[int, int, int], _VaeChunkTraceState] = {}

    @property
    def device(self):
        return self._decoder.device

    def release_trace(self) -> None:
        """Release all VAE chunk traces (call before other module traces)."""
        for key in list(self._chunk_traces.keys()):
            self._release_chunk_trace(key)

    def _release_chunk_trace(self, shape_key: tuple[int, int, int]) -> None:
        dec = self._decoder
        ttnn = dec.ttnn
        state = self._chunk_traces.pop(shape_key, None)
        if state is None:
            return
        if state.trace_id is not None:
            try:
                ttnn.release_trace(self.device, state.trace_id)
            except Exception:
                pass
        for t in (state.latent_dev, state.output_dev):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass

    def _prepare_latents_for_decode(self, latents_btc):
        """Row-major BF16 latents for conv (matches ``decode_tiled``)."""
        dec = self._decoder
        ttnn = dec.ttnn
        if latents_btc.layout != ttnn.ROW_MAJOR_LAYOUT:
            latents_btc = ttnn.to_layout(latents_btc, ttnn.ROW_MAJOR_LAYOUT)
        if latents_btc.dtype != dec.activation_dtype:
            latents_btc = ttnn.typecast(latents_btc, dec.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return latents_btc

    @staticmethod
    def _host_staging_buffer_matching(dev_tensor) -> Any:
        """Host buffer for ``copy_device_to_host_tensor`` with matching layout/page config."""
        import ttnn

        th = ttnn.to_torch(dev_tensor).contiguous()
        if th.dtype != torch.bfloat16:
            th = th.to(dtype=torch.bfloat16)
        return ttnn.from_torch(th, dtype=ttnn.bfloat16, layout=dev_tensor.layout)

    def _stage_chunk_latents_cq1(self, x_src, state: _VaeChunkTraceState, *, skip_cq0_wait: bool) -> None:
        dec = self._decoder
        ttnn = dec.ttnn
        device = self.device
        if not skip_cq0_wait:
            if state.op_event is None:
                raise RuntimeError("VAE chunk trace CQ1 staging called before a prior execute (op_event is None).")
            ttnn.wait_for_event(1, state.op_event)
        # Fresh host tensor from CPU (same pattern as ``test_vae_decoder_trace_2cq``). Device→host
        # staging of slice views was producing wrong replay values in overlap-add tiling.
        th = ttnn.to_torch(x_src).contiguous()
        if th.dtype != torch.bfloat16:
            th = th.to(dtype=torch.bfloat16)
        host_latent = ttnn.from_torch(th, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_latent, state.latent_dev, cq_id=1)
        state.stage_write_event = ttnn.record_event(device, 1)

    def _capture_chunk_trace(self, latents_btc, shape_key: tuple[int, int, int]) -> None:
        dec = self._decoder
        ttnn = dec.ttnn
        device = self.device
        if not hasattr(ttnn, "begin_trace_capture"):
            raise RuntimeError("VAE trace requires begin_trace_capture / execute_trace.")
        x_src = self._prepare_latents_for_decode(latents_btc)
        if not hasattr(ttnn, "clone"):
            raise RuntimeError("ttnn.clone is required for VAE decoder trace.")
        x_dev = ttnn.clone(x_src)
        for _ in range(2):
            y_w = dec(x_dev)
            ttnn.synchronize_device(device)
            try:
                ttnn.deallocate(y_w)
            except Exception:
                pass
        state = _VaeChunkTraceState(
            trace_id=None,
            latent_dev=x_dev,
            latent_host=self._host_staging_buffer_matching(x_dev),
            output_dev=None,
            op_event=None,
        )
        self._stage_chunk_latents_cq1(x_src, state, skip_cq0_wait=True)
        if state.stage_write_event is not None:
            ttnn.wait_for_event(0, state.stage_write_event)
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        y_out = dec(x_dev)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)
        state.trace_id = trace_id
        state.output_dev = y_out
        state.op_event = ttnn.record_event(device, 0)
        state.stage_write_event = None
        self._chunk_traces[shape_key] = state

    def _replay_chunk_trace(self, latents_btc, shape_key: tuple[int, int, int]):
        dec = self._decoder
        ttnn = dec.ttnn
        device = self.device
        state = self._chunk_traces[shape_key]
        x_src = self._prepare_latents_for_decode(latents_btc)
        skip_wait = state.op_event is None
        self._stage_chunk_latents_cq1(x_src, state, skip_cq0_wait=skip_wait)
        if state.stage_write_event is None:
            raise RuntimeError("VAE chunk trace execute called before CQ1 staging completed.")
        ttnn.wait_for_event(0, state.stage_write_event)
        state.stage_write_event = None
        ttnn.execute_trace(device, state.trace_id, cq_id=0, blocking=True)
        state.op_event = ttnn.record_event(device, 0)
        ttnn.synchronize_device(device)
        if hasattr(ttnn, "clone"):
            return ttnn.clone(state.output_dev)
        return state.output_dev

    def decode_chunk_traced(self, latents_btc):
        dec = self._decoder
        ttnn = dec.ttnn
        x = self._prepare_latents_for_decode(latents_btc)
        if hasattr(ttnn, "clone"):
            x = ttnn.clone(x)
        shape_key = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
        if shape_key not in self._chunk_traces:
            self._capture_chunk_trace(x, shape_key)
            state = self._chunk_traces[shape_key]
            if state.output_dev is None:
                raise RuntimeError("VAE chunk trace capture did not produce output_dev.")
            if hasattr(ttnn, "clone"):
                return ttnn.clone(state.output_dev)
            return state.output_dev
        return self._replay_chunk_trace(x, shape_key)

    @classmethod
    def from_hf_vae_dir(
        cls,
        vae_dir: str,
        *,
        device,
        latent_frames: int | None = None,
        batch_size: int | None = None,
        activation_dtype=None,
        weights_dtype=None,
        decoder_prefix: str = "decoder.",
    ) -> TtOobleckVaeDecoder:
        """Build from Diffusers-style ``config.json`` + ``*.safetensors`` weights.

        ``latent_frames`` / ``batch_size`` are accepted for ABI compatibility; conv
        packing is deferred to the forward pass according to runtime shapes (same idea
        as ``TtAceStepPatchEmbed1D``), so warmup is not triggered here (avoids a full
        decode on boot for long clips).
        """
        _ = latent_frames
        _ = batch_size

        root = Path(vae_dir)
        cfg_path = root / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing HF VAE config: {cfg_path}")
        sd_path = _pick_safetensors_file(root)

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        downs = cfg.get("downsampling_ratios")
        if not isinstance(downs, list):
            raise ValueError("downsampling_ratios missing or not a list in VAE config.json")
        ups = downs[::-1]

        decoder_channels = int(cfg["decoder_channels"])
        decoder_input_channels = int(cfg["decoder_input_channels"])
        audio_channels = int(cfg["audio_channels"])
        channel_multiples = cfg["channel_multiples"]
        if not isinstance(channel_multiples, list):
            raise ValueError("channel_multiples missing or not a list in VAE config.json")

        state_dict = _load_state_dict_torch(sd_path)

        inner = TtOobleckDecoder(
            state_dict=state_dict,
            device=device,
            decoder_prefix=decoder_prefix,
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=tuple(ups),
            channel_multiples=tuple(channel_multiples),
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )
        return cls(inner)

    def __call__(self, latents_btc):
        """Decode ``[B, T_latent, C_latent]`` row-major to ``[B, T_audio, audio_channels]``."""
        return self._decoder(latents_btc)

    def forward(self, latents_btc):
        return self(latents_btc)

    def decode_tiled(
        self,
        latents_btc,
        *,
        chunk_size: int = 32,
        overlap: int = 4,
        use_trace: bool = False,
    ):
        """Decode ``[B, T_latent, C]`` along time using overlap-add (same idea as AceStep ``_tiled_decode_gpu``).

        Long sequences can exceed TTNN conv L1 circular-buffer limits when decoded in one shot; tiling
        keeps each ``ttnn.conv1d`` invocation on a bounded temporal extent.
        """
        dec = self._decoder
        ttnn = dec.ttnn
        if len(latents_btc.shape) != 3:
            raise ValueError(f"TtOobleckVaeDecoder.decode_tiled expects [B,T,C], got {latents_btc.shape}")

        # ttnn.slice on TILE layout requires both slice-start and output dimensions to be multiples of
        # TILE_HEIGHT=32 (asserted in slice_device_operation.cpp).  Chunk windows like [24:60] are never
        # 32-aligned, so slicing TILE layout silently returns wrong data in Release builds.
        latents_btc = self._prepare_latents_for_decode(latents_btc)
        # Trace replay for decode was not bit-accurate vs eager (audible noise on long clips).
        # Keep ``decode_chunk_traced`` for tests; production ``decode_tiled`` stays eager.
        use_chunk_trace = False
        if use_chunk_trace:
            self.release_trace()

        batch = int(latents_btc.shape[0])
        latent_frames = int(latents_btc.shape[1])
        c_lat = int(latents_btc.shape[2])
        if batch > 1:
            parts = []
            for b in range(batch):
                slab = ttnn.slice(latents_btc, (b, 0, 0), (b + 1, latent_frames, c_lat))
                parts.append(self.decode_tiled(slab, chunk_size=chunk_size, overlap=overlap, use_trace=use_trace))
            return ttnn.concat(parts, dim=0) if hasattr(ttnn, "concat") else ttnn.concatenate(parts, dim=0)

        min_win = int(
            os.environ.get(
                "ACE_STEP_VAE_MIN_DECODER_LATENT_WINDOW",
                str(_DEFAULT_MIN_DECODER_LATENT_WINDOW),
            )
        )
        audio_up = math.prod(int(u) for u in dec.upsampling_ratios)

        def _crop_audio_tail(wav_tt, pad_lat_frames: int, *, clone_output: bool = False):
            if pad_lat_frames <= 0:
                if clone_output and hasattr(ttnn, "clone"):
                    return ttnn.clone(wav_tt)
                return wav_tt
            ta = int(wav_tt.shape[1])
            drop = pad_lat_frames * audio_up
            end_i = max(0, ta - drop)
            bw = int(wav_tt.shape[0])
            ca = int(wav_tt.shape[2])
            out = ttnn.slice(wav_tt, (0, 0, 0), (bw, end_i, ca))
            if clone_output and hasattr(ttnn, "clone"):
                return ttnn.clone(out)
            return out

        initial_pad_lat = max(0, min_win - latent_frames)
        if initial_pad_lat > 0:
            pad_zeros = ttnn.zeros(
                (batch, initial_pad_lat, c_lat),
                dtype=latents_btc.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=latents_btc.device(),
            )
            latents_btc = ttnn.concat([latents_btc, pad_zeros], dim=1, **ace_step_concat_kwargs(ttnn))
            latent_frames = int(latents_btc.shape[1])

        chunk_size = int(chunk_size)
        overlap = int(overlap)
        if chunk_size < 8:
            raise ValueError(f"chunk_size must be >= 8 for stable tiling, got {chunk_size}")
        min_overlap = 4
        effective_overlap = overlap
        while chunk_size - 2 * effective_overlap <= 0 and effective_overlap > min_overlap:
            effective_overlap //= 2
        if effective_overlap < min_overlap and overlap >= min_overlap:
            effective_overlap = min_overlap
        overlap = effective_overlap

        if latent_frames <= chunk_size:
            if use_chunk_trace:
                wav_one = self.decode_chunk_traced(latents_btc)
            else:
                wav_one = dec(latents_btc)
            ace_step_flush_device_profiler(dec.device)
            return _crop_audio_tail(wav_one, initial_pad_lat, clone_output=use_chunk_trace)

        stride = chunk_size - 2 * overlap
        if stride <= 0:
            raise ValueError(f"chunk_size {chunk_size} must be > 2 * overlap {overlap}")
        num_steps = math.ceil(latent_frames / stride)
        # Overlap-add uses many different window lengths; per-chunk trace replay was not
        # bit-accurate vs eager and produced audible noise. Keep trace for the single-window
        # path above; run each overlap-add tile eagerly here.
        use_chunk_trace = False

        # TIME-PARALLEL: on a multi-chip mesh with ACE_STEP_VAE_TIME_PARALLEL set, decode
        # N=num_chips tiles at once by batch-sharding the tile batch across the mesh (each chip
        # decodes one [1,win,C] tile at batch=1 — no batch>1 conv support needed). Default off →
        # the validated sequential path below is unchanged.
        if _vae_time_parallel_enabled(dec.device):
            return self._decode_tiled_parallel(
                latents_btc,
                dec=dec,
                num_steps=num_steps,
                stride=stride,
                overlap=overlap,
                latent_frames=latent_frames,
                c_lat=c_lat,
                min_win=min_win,
                audio_up=audio_up,
                initial_pad_lat=initial_pad_lat,
            )

        # Device ``ttnn.slice`` on ROW_MAJOR latents is safe for arbitrary ``[win_start, win_end)``;
        # TILE slices require 32-aligned bounds (see comment at top of this method).
        cores = []
        upsample_factor = None

        for i in range(num_steps):
            core_start = i * stride
            core_end = min(core_start + stride, latent_frames)
            win_start = max(0, core_start - overlap)
            win_end = min(latent_frames, core_end + overlap)
            win_len = win_end - win_start
            if win_len < min_win:
                deficit = min_win - win_len
                win_start = max(0, win_start - deficit)
                win_len = win_end - win_start
                if win_len < min_win:
                    win_end = min(latent_frames, win_start + min_win)

            latent_chunk = ttnn.slice(latents_btc, (0, win_start, 0), (1, win_end, c_lat))
            try:
                wav = dec(latent_chunk)
            finally:
                _safe_deallocate(latent_chunk)
            ace_step_flush_device_profiler(dec.device)
            # Decoder ends with conv2 → often TILE; trim uses ttnn.slice which is not safe on TILE
            # for arbitrary [T_start, T_end) (same 32-tile alignment rules as latents).
            wav = ttnn.to_layout(wav, ttnn.ROW_MAJOR_LAYOUT)
            latent_t = win_end - win_start
            b_w = int(wav.shape[0])
            ta = int(wav.shape[1])
            ca = int(wav.shape[2])

            if upsample_factor is None:
                upsample_factor = float(ta) / float(latent_t)

            added_start = core_start - win_start
            trim_start_i = int(round(added_start * upsample_factor))
            added_end = win_end - core_end
            trim_end_i = int(round(added_end * upsample_factor))

            trim_start_i = max(0, min(trim_start_i, ta))
            end_i = ta - trim_end_i if trim_end_i > 0 else ta
            end_i = max(trim_start_i, min(end_i, ta))

            audio_core = ttnn.slice(wav, (0, trim_start_i, 0), (b_w, end_i, ca))
            _safe_deallocate(wav)
            out_mc = ace_step_vae_decode_concat_memory_config(ttnn)
            if out_mc is not None and hasattr(ttnn, "to_memory_config"):
                audio_core = ttnn.to_memory_config(audio_core, out_mc)
            cores.append(audio_core)
            # Long overlap-add runs (30 s @ 25 Hz ≈ 90+ tiles) fragment L1 unless each tile's
            # conv activations are freed before the next ``dec()`` repacks weights.
            ttnn.synchronize_device(dec.device)

        merged = _merge_audio_cores(cores, ttnn)
        return _crop_audio_tail(merged, initial_pad_lat)

    def _decode_tiled_parallel(
        self,
        latents_btc,
        *,
        dec,
        num_steps: int,
        stride: int,
        overlap: int,
        latent_frames: int,
        c_lat: int,
        min_win: int,
        audio_up: int,
        initial_pad_lat: int,
    ):
        """Time-parallel overlap-add: decode ``N=num_chips`` tiles per round by batch-sharding the
        tile batch across the mesh (each chip runs one ``[1, win, C]`` tile). Numerically identical
        to the sequential path (each chip does the same batch-1 decode) — see PCC gate."""
        import torch

        from models.experimental.ace_step_v1_5.utils.tt_device import ace_step_replicate_mesh_mapper

        ttnn = dec.ttnn
        device = dec.device
        n_chips = int(device.get_num_devices())

        # Window metadata — identical formula to the sequential loop.
        wins = []
        for i in range(num_steps):
            core_start = i * stride
            core_end = min(core_start + stride, latent_frames)
            win_start = max(0, core_start - overlap)
            win_end = min(latent_frames, core_end + overlap)
            win_len = win_end - win_start
            if win_len < min_win:
                deficit = min_win - win_len
                win_start = max(0, win_start - deficit)
                win_len = win_end - win_start
                if win_len < min_win:
                    win_end = min(latent_frames, win_start + min_win)
            wins.append((win_start, win_end, core_start, core_end))
        win_pad = max(we - ws for ws, we, _, _ in wins)  # uniform tile length → uniform audio length

        # The latent is replicated across the mesh — read device-0's copy once, window on host.
        lat = ttnn.to_torch(ttnn.get_device_tensors(latents_btc)[0]).float().reshape(1, latent_frames, c_lat)

        cores: list = []  # host audio cores, in tile order
        for r in range(0, num_steps, n_chips):
            tiles, meta = [], []
            for j in range(n_chips):
                idx = r + j
                if idx < num_steps:
                    ws, we, cs, ce = wins[idx]
                    w = lat[0, ws:we, :]
                    pad = win_pad - (we - ws)
                    if pad > 0:
                        w = torch.cat([w, torch.zeros(pad, c_lat, dtype=w.dtype)], dim=0)
                    tiles.append(w)
                    meta.append((cs - ws, (ws + win_pad) - ce, True))  # added_start, added_end(+pad), real
                else:
                    tiles.append(torch.zeros(win_pad, c_lat, dtype=lat.dtype))
                    meta.append((0, 0, False))
            batch = torch.stack(tiles, dim=0)  # [n_chips, win_pad, c_lat]
            x = ttnn.from_torch(
                batch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
            )
            y = ttnn.to_layout(dec(x), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.synchronize_device(device)
            outs = ttnn.get_device_tensors(y)
            for j in range(n_chips):
                added_start, added_end, real = meta[j]
                if not real:
                    continue
                wav = ttnn.to_torch(outs[j]).float().reshape(1, -1, int(self._decoder.audio_channels))
                ta = int(wav.shape[1])
                up = ta / float(win_pad)
                ts = max(0, min(int(round(added_start * up)), ta))
                te = ta - int(round(added_end * up)) if added_end > 0 else ta
                te = max(ts, min(te, ta))
                cores.append(wav[:, ts:te, :])
        merged = torch.cat(cores, dim=1)
        if initial_pad_lat > 0:
            merged = merged[:, : max(0, int(merged.shape[1]) - initial_pad_lat * audio_up), :]
        return ttnn.from_torch(
            merged.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=ace_step_replicate_mesh_mapper(device),
        )
