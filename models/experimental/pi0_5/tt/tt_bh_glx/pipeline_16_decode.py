# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""16-chip pi0.5 pipeline: 1x8 vision/prefill + 8-stage decode_all denoise.

This is a correctness-first integration of the streamed decode denoise path in
`tt_pipeline`.  The prefill path mirrors `pipeline_1x8.py` on row 0.  The
denoise path runs on eight 1x1 submeshes carved from row 1, with explicit
host-side KV handoff from the replicated prefill output.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
from models.experimental.pi0_5.tt.ttnn_pi0_5_model import _MASK_VAL, _precompute_rope_table_torch
from models.experimental.pi0_5.tt.tt_pipeline import denoise_block as _decode_block
from models.experimental.pi0_5.tt.tt_pipeline import perf_suffix_len
from models.experimental.pi0_5.tt.tt_pipeline._d2d_pipeline import Pipeline
from models.experimental.pi0_5.tt.tt_pipeline.denoise_block import TTNNPi05DenoiseExpertBlock
from models.experimental.pi0_5.tt.tt_pipeline.denoise_pipeline import build_denoise_loop_pipeline
from models.experimental.pi0_5.tt.tt_pipeline.weight_adapt import expert_reference_blocks, final_mod, suffix_reference

from .pipeline import _PrefillHead
from .stage_prefill_tp4 import StagePrefillTP4
from .vision_slice import SigLIPCameraSlice


_NUM_CHIPS_REQUIRED = 8
_DEFAULT_NUM_REAL_CAMS = 3
_NUM_PATCHES = 256
_DENOISE_SPLITS_8 = (2, 2, 2, 3, 3, 2, 2, 2)


class Pi0_5GLX16DecodePipeline:
    """End-to-end sample_actions with TP=8 prefill and streamed 8-chip denoise."""

    def __init__(self, config: Pi0_5ModelConfig, weights: Dict[str, dict], mesh_handles):
        self.h = mesh_handles
        self.prefill_mesh = mesh_handles.prefill_submesh
        self.denoise_mesh = mesh_handles.denoise_submesh
        self.denoise_per_chip = mesh_handles.denoise_per_chip
        if self.prefill_mesh.get_num_devices() != _NUM_CHIPS_REQUIRED:
            raise RuntimeError(f"prefill_submesh must have 8 devices, got {self.prefill_mesh.get_num_devices()}")
        if len(self.denoise_per_chip) != _NUM_CHIPS_REQUIRED:
            raise RuntimeError(f"expected 8 denoise 1x1 submeshes, got {len(self.denoise_per_chip)}")

        self.config = config
        self._weights = weights
        self.num_denoising_steps = config.num_denoising_steps
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self._action_horizon_padded = ((config.action_horizon + 31) // 32) * 32
        self._patch_size = config.siglip_config.patch_size
        self._vlm_hidden = config.vlm_config.width

        # TP=8 prefill collectives must use Linear (not Ring) topology on the
        # carved (1,8) submesh: the 8 chips are a sub-slice of the 32-chip torus,
        # so the Ring wraparound (chip7<->chip0) has no fabric route under
        # FABRIC_1D. Linear (adjacent hops within row 0) routes and is
        # numerically identical. Shell override still wins via setdefault.
        os.environ.setdefault("PI0_CCL_TOPOLOGY", "linear")

        # Force the new decode_all matmul_decode path by default. Keep the
        # walltime env override for A/B.
        _decode_block.DECODE_ALL = os.environ.get("PI05_WALLTIME_DECODE_ALL", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        self.vision = SigLIPCameraSlice(
            config.siglip_config,
            weights["vlm_vision"],
            weights["vlm_projector"],
            self.prefill_mesh,
        )
        self.prefill = StagePrefillTP4(config, weights, self.prefill_mesh)
        self.prefill_head = _PrefillHead(weights["vlm_language"], self.prefill_mesh, self._vlm_hidden)

        self._suffix_cfg = SuffixConfig(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            expert_width=config.expert_config.width,
            pi05=True,
        )
        self._ref_blocks = expert_reference_blocks(weights["action_expert"], config.expert_config, depth=18)
        self._final_w, self._final_b = final_mod(weights["action_expert"])
        self._ref_suffix = suffix_reference(weights["pi0_projections"], self._suffix_cfg)
        self._driver = None
        self._driver_build_key = None
        self._kv_socks = None
        # Prefill-mesh trace (vision + prefix + TP=8 prefill) + persistent KV buffers.
        self._prefill_trace_id = None
        self._prefill_trace_key = None
        self._kv_persist = None  # list[(K_buf, V_buf)] on prefill_mesh, written by the trace
        # e2e on-device socket KV (PI0_KV_SOCKET=1): no host bounce. KV is socketed
        # prefill(0,c)->denoise(1,c) inside the traces. Default off (host-bounce path).
        self._socket_mode = os.environ.get("PI0_KV_SOCKET", "").lower() in ("1", "true", "yes", "on")
        self._kv_recv_tids = None  # per denoise chip: trace id for recv+copy->_prefix_kv
        self._socket_built = False
        # Single-trace mode (WIP, opt-in via PI0_16_SINGLE_TRACE=1): capture the WHOLE
        # e2e (prefill + KV concat/send + recv/copy + N denoise steps) in ONE trace on
        # the (2,8) compute submesh root (mesh_handles.trace_root), mirroring the
        # 28-chip pipeline. Default OFF -> the validated multi-trace path (1 prefill +
        # 8 recv + 8 loop).
        #
        # KNOWN BLOCKER (root-caused): the capture records all stages but HANGS at
        # end_trace_capture(trace_root). Bisect showed it hangs on the PREFILL portion
        # (denoise ruled out). Cause: the prefill is TENSOR-PARALLEL — its TP=8
        # all_reduce/all_gather CCL spans only row 0 (a SUBSET of the (2,8) root). A
        # traced collective requires the trace root == the CCL mesh; when the root is
        # bigger than the collective, end_trace_capture deadlocks. Evidence: the
        # multi-trace prefill trace rooted on the (1,8) works (CCL == whole trace mesh);
        # the 1x8 pipeline's single trace works (CCL == mesh); the 28-chip single trace
        # works because its prefill is PIPELINE-parallel (no CCL). Minimal socket /
        # cyclic-socket parent-rooted traces also finalize fine — it's specifically CCL
        # on a subset. A true single trace here needs a non-CCL (pipeline-parallel)
        # prefill; the (2,8) carve + driver split below are validated and kept for that.
        self._single_trace = self._socket_mode and (
            os.environ.get("PI0_16_SINGLE_TRACE", "0").lower() in ("1", "true", "yes", "on")
        )
        self._trace_root = getattr(mesh_handles, "trace_root", None)
        self._e2e_trace_id = None  # single-trace: the one trace id on trace_root
        self._trace_cat_bufs = None  # resident in-trace KV-concat buffers

        self._num_real_cams = int(os.environ.get("PI0_NUM_CAMERAS", _DEFAULT_NUM_REAL_CAMS))
        if not 1 <= self._num_real_cams <= _NUM_CHIPS_REQUIRED:
            raise RuntimeError(f"PI0_NUM_CAMERAS={self._num_real_cams} out of range [1, {_NUM_CHIPS_REQUIRED}]")

        self.pixel_values_buf = None
        self.lang_tokens_buf = None

        self._prefix_cos = None
        self._prefix_sin = None
        self._prefix_attn_mask = None
        self._artifact_mask_key = None
        self._expert_attn_mask_torch = None
        self._position_offset = None
        self._prefix_len = None

    # ──────────── Input / prefix helpers ───────────────────────────────

    def _stack_and_fold_pixels(self, images: List[torch.Tensor]) -> torch.Tensor:
        n_real = len(images)
        if n_real != self._num_real_cams:
            raise RuntimeError(f"Pipeline configured for {self._num_real_cams} real cameras; got {n_real}")
        real = torch.cat(images, dim=0)
        if real.shape[0] < _NUM_CHIPS_REQUIRED:
            pad = torch.zeros(
                _NUM_CHIPS_REQUIRED - real.shape[0],
                real.shape[1],
                real.shape[2],
                real.shape[3],
                dtype=real.dtype,
            )
            real = torch.cat([real, pad], dim=0)
        x = real.permute(0, 2, 3, 1).contiguous()
        b, h, w, c = x.shape
        return x.reshape(b, h, w // self._patch_size, c * self._patch_size).contiguous()

    def _ensure_persistent_input_buffers(self, images: List[torch.Tensor], lang_tokens: torch.Tensor) -> None:
        pixel_host = self._stack_and_fold_pixels(images)
        if self.pixel_values_buf is None:
            self.pixel_values_buf = ttnn.from_torch(
                pixel_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.prefill_mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(self.prefill_mesh, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            host_t = ttnn.from_torch(
                pixel_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.prefill_mesh, dim=0),
            )
            ttnn.copy_host_to_device_tensor(host_t, self.pixel_values_buf)

        if self.lang_tokens_buf is None:
            self.lang_tokens_buf = ttnn.from_torch(
                lang_tokens.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.prefill_mesh,
            )
        else:
            host_t = ttnn.from_torch(lang_tokens.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_t, self.lang_tokens_buf)

    def _run_vision_dp(self, pixel_values_ttnn: "ttnn.Tensor") -> "ttnn.Tensor":
        vision_out = self.vision.forward(pixel_values_ttnn)
        gathered = ttnn.all_gather(
            vision_out,
            dim=0,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(vision_out)
        real = ttnn.slice(gathered, [0, 0, 0], [self._num_real_cams, _NUM_PATCHES, self._vlm_hidden])
        ttnn.deallocate(gathered)
        return real

    def _build_prefix(self, vision_real: "ttnn.Tensor", lang_tokens) -> "ttnn.Tensor":
        n_cams = int(vision_real.shape[0])
        v = ttnn.reshape(vision_real, (1, n_cams * _NUM_PATCHES, self._vlm_hidden))
        lang_emb = self.prefill_head.embed_lang(lang_tokens)
        prefix = ttnn.concat([v, lang_emb], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        if prefix.layout != ttnn.TILE_LAYOUT:
            prefix = ttnn.to_layout(prefix, ttnn.TILE_LAYOUT)
        return prefix

    # ──────────── Upstream mask / RoPE artifacts ───────────────────────

    def _build_upstream_artifacts(self, img_masks=None, lang_masks=None) -> None:
        num_cams = self._num_real_cams
        suffix_padded = self._action_horizon_padded
        action_horizon = self.action_horizon
        vlm_head_dim = self.config.vlm_config.head_dim
        expert_head_dim = self.config.expert_config.head_dim
        max_seq_len = self.config.max_seq_len

        if img_masks is None:
            img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cams)]
        if lang_masks is None:
            lang_masks = torch.ones(1, 256, dtype=torch.bool)
        assert len(img_masks) == num_cams, f"need {num_cams} img masks, got {len(img_masks)}"

        pad_segs = []
        for m in img_masks:
            real = bool(m.item()) if m.numel() == 1 else bool(m[0].item())
            pad_segs.append(torch.full((_NUM_PATCHES,), real, dtype=torch.bool))
        pad_segs.append(lang_masks[0].to(torch.bool))
        pad_mask = torch.cat(pad_segs, dim=0)
        prefix_len = pad_mask.shape[0]
        prefix_padded = ((prefix_len + 31) // 32) * 32
        prefix_real_count = int(pad_mask.sum().item())
        all_real_aligned = (prefix_real_count == prefix_len) and (prefix_padded == prefix_len)

        if all_real_aligned:
            prefix_mask_4d = None
            prefix_cos_4d = None
            prefix_sin_4d = None
        else:
            pad_2d = pad_mask[:, None] & pad_mask[None, :]
            prefix_mask = torch.zeros(prefix_padded, prefix_padded, dtype=torch.bfloat16)
            prefix_mask[:prefix_len, :prefix_len].masked_fill_(~pad_2d, _MASK_VAL)
            if prefix_padded > prefix_len:
                prefix_mask[prefix_len:, :] = _MASK_VAL
                prefix_mask[:, prefix_len:] = _MASK_VAL
            prefix_mask_4d = prefix_mask.unsqueeze(0).unsqueeze(0)

            position_ids = torch.cumsum(pad_mask.to(torch.int64), dim=0) - 1
            position_ids = position_ids.clamp(min=0, max=max_seq_len - 1)
            cos_vlm, sin_vlm = _precompute_rope_table_torch(vlm_head_dim, max_seq_len)
            prefix_cos = cos_vlm[position_ids]
            prefix_sin = sin_vlm[position_ids]
            if prefix_padded > prefix_len:
                zc = torch.zeros(prefix_padded - prefix_len, vlm_head_dim, dtype=prefix_cos.dtype)
                zs = torch.zeros(prefix_padded - prefix_len, vlm_head_dim, dtype=prefix_sin.dtype)
                prefix_cos = torch.cat([prefix_cos, zc], dim=0)
                prefix_sin = torch.cat([prefix_sin, zs], dim=0)
            prefix_cos_4d = prefix_cos.unsqueeze(0).unsqueeze(0)
            prefix_sin_4d = prefix_sin.unsqueeze(0).unsqueeze(0)

        kv_total = prefix_padded + suffix_padded
        expert_mask = torch.zeros(suffix_padded, kv_total, dtype=torch.bfloat16)
        pad_blocked = (~pad_mask).nonzero(as_tuple=True)[0]
        if pad_blocked.numel() > 0:
            expert_mask[:, pad_blocked] = _MASK_VAL
        if prefix_padded > prefix_len:
            expert_mask[:, prefix_len:prefix_padded] = _MASK_VAL
        if suffix_padded > action_horizon:
            expert_mask[:, prefix_padded + action_horizon : kv_total] = _MASK_VAL
            expert_mask[action_horizon:suffix_padded, :] = _MASK_VAL
        expert_mask_4d = expert_mask.unsqueeze(0).unsqueeze(0)

        _rope_l1 = os.environ.get("PI0_ROPE_TABLES_L1", "").lower() in ("1", "true", "yes", "on")
        rope_mc = ttnn.L1_MEMORY_CONFIG if _rope_l1 else ttnn.DRAM_MEMORY_CONFIG

        def _up(host_t, mc=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                host_t.to(torch.bfloat16) if host_t.dtype != torch.bfloat16 else host_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.prefill_mesh,
                memory_config=mc,
            )

        for attr in ("_prefix_cos", "_prefix_sin", "_prefix_attn_mask"):
            t = getattr(self, attr, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, attr, None)

        if prefix_mask_4d is not None:
            self._prefix_attn_mask = _up(prefix_mask_4d, ttnn.DRAM_MEMORY_CONFIG)
        if prefix_cos_4d is not None:
            self._prefix_cos = _up(prefix_cos_4d, rope_mc)
            self._prefix_sin = _up(prefix_sin_4d, rope_mc)

        self._expert_attn_mask_torch = expert_mask_4d
        self._position_offset = prefix_real_count
        self._prefix_len = prefix_padded
        img_present = tuple(bool(m.item()) if m.numel() == 1 else bool(m[0].item()) for m in img_masks)
        lang_real_count = int(lang_masks[0].to(torch.bool).sum().item())
        self._artifact_mask_key = (img_present, lang_real_count)

    def prepare_runtime_masks(self, img_masks, lang_masks) -> None:
        img_present = tuple(bool(m.item()) if m.numel() == 1 else bool(m[0].item()) for m in img_masks)
        lang_real_count = int(lang_masks[0].to(torch.bool).sum().item())
        key = (img_present, lang_real_count)
        if key != self._artifact_mask_key:
            self._build_upstream_artifacts(img_masks, lang_masks)

    # ──────────── Denoise helpers ──────────────────────────────────────

    def set_num_denoising_steps(self, num_steps: int) -> None:
        self.num_denoising_steps = int(num_steps)

    def _build_noise_torch(self) -> torch.Tensor:
        noise_pad = torch.zeros(1, self._action_horizon_padded, self.action_dim, dtype=torch.float32)
        noise_pad[:, : self.action_horizon, :] = torch.randn(1, self.action_horizon, self.action_dim)
        return noise_pad

    def _adarms_cond_per_step_torch(self) -> List[torch.Tensor]:
        timesteps = [1.0 - i / self.num_denoising_steps for i in range(self.num_denoising_steps)]
        return [self._ref_suffix.embed_timestep_adarms(torch.tensor([t], dtype=torch.bfloat16)) for t in timesteps]

    def _prefill_kv_to_torch(self, per_layer_kv) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        out = []
        composer = ttnn.ConcatMeshToTensor(self.prefill_mesh, dim=0)
        for k, v in per_layer_kv:
            kt = ttnn.to_torch(k, mesh_composer=composer)
            vt = ttnn.to_torch(v, mesh_composer=composer)
            # Prefill output is replicated on all 8 chips; keep chip 0's copy.
            out.append((kt[:1].contiguous(), vt[:1].contiguous()))
        return out

    # ──────────── Device-direct KV sockets (replace host bounce) ────────
    _KV_SOCK_PAGE = 4096

    def _setup_kv_sockets(self) -> None:
        """Create device-direct KV sockets prefill chip (0,c) -> denoise chip (1,c)
        (same-column collinear hop), recv buffers = each denoise stage's existing
        _prefix_kv L1 tensors. Replaces the per-chunk host KV bounce. Must be called
        after each driver (re)build (the _prefix_kv buffers are freshly allocated).
        Pattern mirrors tt_pipeline/test_xmesh_kv_socket.py."""
        self._release_kv_sockets()
        mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, self._KV_SOCK_PAGE * 4)
        socks = []
        for c, st in enumerate(self._driver._stages):
            src_chip = self.prefill_mesh  # (1,8); SocketConnection picks chip c
            dst_chip = self.denoise_per_chip[c]
            k_idx = 0
            for j, (pk_dev, pv_dev) in enumerate(st._prefix_kv):
                g_layer = st._layer_lo + j
                for w, l1_dst in ((0, pk_dev), (1, pv_dev)):
                    # Recv into a DRAM scratch buffer (NOT the L1 _prefix_kv): a socket
                    # recv targeting the L1 KV forces socket L1 onto the denoise compute
                    # cores (y0..5) and clashes with the matmul static CBs. After recv we
                    # device-copy DRAM->L1 into the buffer the captured trace reads.
                    rv = ttnn.from_torch(
                        torch.zeros(tuple(l1_dst.shape)),
                        dtype=l1_dst.dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=dst_chip,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    conn = ttnn.SocketConnection(
                        ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, c), ttnn.CoreCoord(k_idx, 9)),
                        ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(k_idx, 9)),
                    )
                    pair = ttnn.create_socket_pair(src_chip, dst_chip, ttnn.SocketConfig([conn], mem))
                    socks.append((g_layer, w, pair, rv, l1_dst))
                    k_idx += 1
        self._kv_socks = socks

    def _release_kv_sockets(self) -> None:
        # Sockets are bound to the (persistent) prefill/denoise meshes; dropping the
        # Python refs lets them be reclaimed. Meshes are closed by the harness.
        self._kv_socks = None

    def _send_kv_via_sockets(self, per_layer_kv) -> None:
        """Device-direct send of this chunk's prefill per-layer KV into the denoise
        stages' _prefix_kv recv buffers. No host round-trip."""
        for g_layer, w, pair, rv, l1_dst in self._kv_socks:
            src = per_layer_kv[g_layer][w]
            if src.dtype != rv.dtype:
                src = ttnn.typecast(src, rv.dtype)
            ttnn.experimental.send_direct_async(src, pair[0])
            ttnn.experimental.recv_direct_async(rv, pair[1])
        for m in self.denoise_per_chip:
            ttnn.synchronize_device(m)
        # Device-copy DRAM recv buffers -> the L1 _prefix_kv tensors the trace reads.
        for g_layer, w, pair, rv, l1_dst in self._kv_socks:
            ttnn.copy(rv, l1_dst)

    def _close_driver(self) -> None:
        self._release_kv_sockets()
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            # Free the L1-pinned stage weights too (close() only drops traces). A
            # prompt change rebuilds the driver (new mask/position_offset baked into
            # the trace); without this the old weights leak and OOM L1.
            try:
                self._driver.release_weights()
            except Exception:
                pass
            self._driver = None
        try:
            Pipeline.release_all()
        except Exception:
            pass

    # ──────────── Prefill-mesh trace (vision + prefix + TP=8 prefill) ────
    def _prefill_compute_to_persist(self) -> None:
        """Vision DP + prefix build + TP=8 prefill, then copy each layer's (K,V)
        into the persistent self._kv_persist buffers. Used BOTH eagerly (warmup,
        which also lazily allocates the persistent buffers) and inside the trace
        capture (so replay refills the same buffers). Mirrors pipeline_1x8's
        single captured body reused for warmup + capture."""
        vision_real = self._run_vision_dp(self.pixel_values_buf)
        prefix_embs = self._build_prefix(vision_real, self.lang_tokens_buf)
        ttnn.deallocate(vision_real)
        if self._prefix_cos is not None:
            num_layers = len(self.prefill.blocks)
            final_hidden, per_layer_kv = self.prefill.run(
                prefix_embs,
                attention_mask=self._prefix_attn_mask,
                per_chip_cos=[self._prefix_cos] * num_layers,
                per_chip_sin=[self._prefix_sin] * num_layers,
            )
        else:
            final_hidden, per_layer_kv = self.prefill.run(prefix_embs, attention_mask=self._prefix_attn_mask)
        ttnn.deallocate(prefix_embs)
        ttnn.deallocate(final_hidden)
        if self._prefix_attn_mask is not None:
            per_layer_kv = [
                (ttnn.fill_implicit_tile_padding(k, 0.0), ttnn.fill_implicit_tile_padding(v, 0.0))
                for k, v in per_layer_kv
            ]
        if self._kv_persist is None:
            # Allocate persistent KV buffers (only ever runs eagerly during warmup).
            self._kv_persist = [
                (
                    ttnn.allocate_tensor_on_device(k.spec, self.prefill_mesh),
                    ttnn.allocate_tensor_on_device(v.spec, self.prefill_mesh),
                )
                for k, v in per_layer_kv
            ]
        for i, (k, v) in enumerate(per_layer_kv):
            ttnn.copy(k, self._kv_persist[i][0])
            ttnn.copy(v, self._kv_persist[i][1])
            ttnn.deallocate(k)
            ttnn.deallocate(v)

    def _ensure_prefill_trace(self) -> None:
        """Capture the vision+prefix+prefill body as a trace on the prefill mesh
        (once per prompt/mask). Eager warmup first to JIT all kernels + allocate
        persistent buffers, then begin/end_trace_capture on CQ0."""
        key = self._artifact_mask_key
        if self._prefill_trace_id is not None and self._prefill_trace_key == key:
            return
        if self._prefill_trace_id is not None:
            ttnn.release_trace(self.prefill_mesh, self._prefill_trace_id)
            self._prefill_trace_id = None
        self._prefill_compute_to_persist()  # warmup: JIT + allocate persistent buffers
        self._prefill_trace_id = ttnn.begin_trace_capture(self.prefill_mesh, cq_id=0)
        self._prefill_compute_to_persist()
        ttnn.end_trace_capture(self.prefill_mesh, self._prefill_trace_id, cq_id=0)
        self._prefill_trace_key = key

    def _prefill_via_trace(self):
        """Replay the prefill trace (inputs already refreshed) -> KV in self._kv_persist."""
        ttnn.execute_trace(self.prefill_mesh, self._prefill_trace_id, cq_id=0, blocking=True)
        return self._kv_persist

    # ──────────── e2e on-device socket KV path (PI0_KV_SOCKET=1) ─────────
    @staticmethod
    def _denoise_bounds():
        bounds, acc = [], 0
        for sp in _DENOISE_SPLITS_8:
            bounds.append((acc, acc + sp))
            acc += sp
        return bounds

    def _socket_e2e_setup(self) -> None:
        """One-time build of the no-host-bounce, fully-traced socket KV path:
        prefill trace computes KV + SENDS it via sockets prefill(0,c)->denoise(1,c);
        per-chip recv traces land it in _prefix_kv; denoise loop trace consumes it.
        Sockets are created PRE-build (like the proven inter-stage hop sockets) so
        their L1 config-buffer is placed before the matmul CBs (no clash)."""
        # 0. If rebuilding for a new prompt, tear down the prior setup (no leak).
        if self._socket_built:
            self._socket_teardown()
        # 1. Eager warmup prefill -> _kv_persist (real KV shapes + JIT all kernels).
        self._prefill_compute_to_persist()
        bounds = self._denoise_bounds()
        # 2. Build denoise with a ZERO-KV skeleton (allocates _prefix_kv; real KV
        #    arrives via sockets each chunk). Mirrors the reference zkv skeleton.
        zero_kv = [
            (torch.zeros(tuple(self._kv_persist[i][0].shape)), torch.zeros(tuple(self._kv_persist[i][1].shape)))
            for i in range(len(self._kv_persist))
        ]
        self._driver = build_denoise_loop_pipeline(
            self._ref_blocks,
            self._final_w,
            self._final_b,
            self._ref_suffix,
            self.config,
            self._suffix_cfg,
            self.denoise_mesh,
            adarms_cond_per_step=self._adarms_cond_per_step_torch(),
            prefix_kv_cache=zero_kv,
            prefix_len=self._prefix_len,
            suffix_len=perf_suffix_len(self.action_horizon),
            attention_mask_torch=self._expert_attn_mask_torch,
            position_offset=self._position_offset,
            num_steps=self.num_denoising_steps,
            action_horizon=self.action_horizon,
            splits=_DENOISE_SPLITS_8,
            submeshes=self.denoise_per_chip,
            block_cls=TTNNPi05DenoiseExpertBlock,
            use_concat_kv=True,
            drain="stage0",
        )
        self._driver_build_key = (self.num_denoising_steps, self._artifact_mask_key)
        # 3. KV sockets prefill(0,c)->denoise(1,c) + DRAM recv buffers, created AFTER
        #    the stage weights (set_device) but BEFORE any capture — same ordering as
        #    the proven inter-stage hop sockets, so the socket L1 config-buffer fits in
        #    the L1 left after weights instead of clashing with the matmul CBs.
        # KV-CONCAT: only 2 sockets per chip (one K, one V) carrying that chip's
        # layers concatenated along dim 0, vs 36 per-layer sockets. Dense 3-layer
        # chips drop from 6 sockets to 2 -> the socket L1 footprint fits alongside
        # the matmul CBs. Recv into a DRAM [n_layers, NKV, prefix, hd] buffer; split
        # back into per-layer _prefix_kv on the denoise side.
        mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, self._KV_SOCK_PAGE * 4)
        self._kv_cat_send = []  # (c, w, send_sock)
        self._kv_cat_recv = [[] for _ in range(_NUM_CHIPS_REQUIRED)]  # per chip: (w, recv_sock, dram_recv, n, lo)
        for c, (lo, hi) in enumerate(bounds):
            dst = self.denoise_per_chip[c]
            n = hi - lo
            for w in (0, 1):
                per = list(self._kv_persist[lo][w].shape)  # [1, NKV, prefix, hd]
                cat_shape = (n, per[1], per[2], per[3])
                dt = self._kv_persist[lo][w].dtype
                rv = ttnn.from_torch(
                    torch.zeros(cat_shape),
                    dtype=dt,
                    layout=ttnn.TILE_LAYOUT,
                    device=dst,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                conn = ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, c), ttnn.CoreCoord(w, 9)),
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(w, 9)),
                )
                ss, rs = ttnn.create_socket_pair(self.prefill_mesh, dst, ttnn.SocketConfig([conn], mem))
                self._kv_cat_send.append((c, w, ss))
                self._kv_cat_recv[c].append((w, rs, rv, n, lo))
        send_by = {(c, w): ss for c, w, ss in self._kv_cat_send}
        # Single-trace path: build the denoise driver eagerly (no per-mesh capture),
        # then capture the WHOLE e2e in ONE trace on trace_root. Early-return.
        if self._single_trace:
            self._capture_single_trace(bounds)
            return
        # ---- multi-trace path (PI0_16_SINGLE_TRACE=0) ----
        # 4. Capture the denoise loop (stream_euler): warms the inter-stage hop
        #    sockets + JITs send/recv + inits per-chip socket infra. Zero-KV replay
        #    discarded.
        self._driver.stream_euler(self._build_noise_torch(), capture=True)
        # 5. Eager warm the FULL KV socket round (concat + send + recv + sync + copy)
        #    ONCE, on the valid _kv_persist from the step-1 warmup. Trace capture can't
        #    JIT, and capturing the send COLD raises "Writes not supported during trace
        #    capture" — the eager pre-round is exactly what makes send-in-trace work
        #    (verified by /tmp/send_trace_repro on this build + the reference
        #    test_xmesh_traced_pipeline.py). Operates directly on _kv_persist (the
        #    prefill trace is not captured yet).
        _cats = {}
        for c, (lo, hi) in enumerate(bounds):
            _cats[c] = (
                ttnn.concat([self._kv_persist[g][0] for g in range(lo, hi)], dim=0),
                ttnn.concat([self._kv_persist[g][1] for g in range(lo, hi)], dim=0),
            )
        for c, w, ss in self._kv_cat_send:
            ttnn.experimental.send_direct_async(_cats[c][w], ss)
        for c in range(_NUM_CHIPS_REQUIRED):
            for w, rs, rv, n, lo2 in self._kv_cat_recv[c]:
                ttnn.experimental.recv_direct_async(rv, rs)
        ttnn.synchronize_device(self.prefill_mesh)
        for m in self.denoise_per_chip:
            ttnn.synchronize_device(m)
        for c, (lo, hi) in enumerate(bounds):
            st = self._driver._stages[c]
            for w, rs, rv, n, lo2 in self._kv_cat_recv[c]:
                per = list(self._kv_persist[lo][w].shape)
                for j in range(n):
                    sl = ttnn.slice(rv, [j, 0, 0, 0], [j + 1, per[1], per[2], per[3]])
                    ttnn.copy(sl, st._prefix_kv[j][w])
                    ttnn.deallocate(sl)
        for c in _cats:
            ttnn.deallocate(_cats[c][0])
            ttnn.deallocate(_cats[c][1])
        # 6. Prefill trace = COMPUTE + KV-concat + socket SEND (fully-traced tail).
        #    send_direct_async on the (1,8) prefill mesh DOES capture on this build
        #    once warmed (step 5); the earlier "Writes not supported" was from
        #    capturing the send cold. The per-chip concat outputs are transient
        #    trace-region tensors (deterministic replay addresses); the send reads
        #    them and nothing reuses those addresses until the next chunk's prefill
        #    replay (after the stage-0 drain), so they are left resident in the trace.
        self._prefill_trace_id = ttnn.begin_trace_capture(self.prefill_mesh, cq_id=0)
        self._prefill_compute_to_persist()
        for c, (lo, hi) in enumerate(bounds):
            for w in (0, 1):
                cat = ttnn.concat([self._kv_persist[g][w] for g in range(lo, hi)], dim=0)
                ttnn.experimental.send_direct_async(cat, send_by[(c, w)])
        ttnn.end_trace_capture(self.prefill_mesh, self._prefill_trace_id, cq_id=0)
        self._prefill_trace_key = self._artifact_mask_key
        # 7. Per-denoise-stage RECV trace = recv + slice/copy DRAM->L1 into the stage's
        #    _prefix_kv the loop reads. recv_direct_async is CQ-ordered so the copy
        #    waits for the recv; both capture fine on the single-chip denoise meshes.
        self._kv_recv_tids = []
        for c, (lo, hi) in enumerate(bounds):
            dst = self.denoise_per_chip[c]
            st = self._driver._stages[c]
            tid = ttnn.begin_trace_capture(dst, cq_id=0)
            for w, rs, rv, n, lo2 in self._kv_cat_recv[c]:
                ttnn.experimental.recv_direct_async(rv, rs)
            for w, rs, rv, n, lo2 in self._kv_cat_recv[c]:
                per = list(self._kv_persist[lo][w].shape)
                for j in range(n):
                    sl = ttnn.slice(rv, [j, 0, 0, 0], [j + 1, per[1], per[2], per[3]])
                    ttnn.copy(sl, st._prefix_kv[j][w])
                    ttnn.deallocate(sl)
            ttnn.end_trace_capture(dst, tid)
            self._kv_recv_tids.append((dst, tid))
        self._socket_built = True

    # ──────────── Single-trace (2,8) path ──────────────────────────────
    def _kv_socket_round(self, bounds, *, in_trace: bool) -> None:
        """One KV handoff round: per-chip concat of _kv_persist layers -> socket send
        prefill(0,c)->denoise(1,c) -> recv into DRAM scratch -> slice/copy DRAM->L1
        into each stage's _prefix_kv. Used eagerly (in_trace=False: syncs + frees the
        transient concat buffers) and inside the single e2e trace (in_trace=True: no
        host sync — CQ ordering + the fabric socket handshake gate it — and the concat
        buffers stay resident in the trace region through the send)."""
        cats = {}
        for c, (lo, hi) in enumerate(bounds):
            cats[c] = (
                ttnn.concat([self._kv_persist[g][0] for g in range(lo, hi)], dim=0),
                ttnn.concat([self._kv_persist[g][1] for g in range(lo, hi)], dim=0),
            )
        for c, w, ss in self._kv_cat_send:
            ttnn.experimental.send_direct_async(cats[c][w], ss)
        for c in range(_NUM_CHIPS_REQUIRED):
            for w, rs, rv, n, lo2 in self._kv_cat_recv[c]:
                ttnn.experimental.recv_direct_async(rv, rs)
        if not in_trace:
            ttnn.synchronize_device(self.prefill_mesh)
            for m in self.denoise_per_chip:
                ttnn.synchronize_device(m)
        for c, (lo, hi) in enumerate(bounds):
            st = self._driver._stages[c]
            for w, rs, rv, n, lo2 in self._kv_cat_recv[c]:
                per = list(self._kv_persist[lo][w].shape)
                for j in range(n):
                    sl = ttnn.slice(rv, [j, 0, 0, 0], [j + 1, per[1], per[2], per[3]])
                    ttnn.copy(sl, st._prefix_kv[j][w])
                    ttnn.deallocate(sl)
        if in_trace:
            # Pin the concat buffers for the whole trace: the send DMA reads them and
            # the N denoise steps that follow in the same trace must NOT reuse their
            # trace-region addresses. Held on self so Python GC can't free them during
            # capture; released in _socket_teardown.
            for c in cats:
                self._trace_cat_bufs.extend([cats[c][0], cats[c][1]])
        else:
            for c in cats:
                ttnn.deallocate(cats[c][0])
                ttnn.deallocate(cats[c][1])

    def _capture_single_trace(self, bounds) -> None:
        """Capture the whole e2e — prefill compute + KV concat/send + recv/copy + N
        denoise steps — in ONE trace on trace_root (the (2,8) compute submesh that
        contains both the prefill row and the denoise stage children). Mirrors the
        28-chip's single begin_trace_capture(compute). Every chip in trace_root is
        commanded (prefill row 0 + denoise row 1), so the trace's blocking finish
        doesn't wait on idle chips."""
        if self._trace_root is None:
            raise RuntimeError("single-trace requires mesh_handles.trace_root (the (2,8) compute submesh)")
        noise = self._build_noise_torch()
        # 4. Eager denoise build: hop/wrap sockets + JIT the stage forwards (no capture).
        self._driver.build_eager(noise)
        # 5. Eager KV-socket warm-round (JITs concat/send/recv/copy) on the valid
        #    _kv_persist from the step-1 warmup, so the in-trace socket ops don't JIT
        #    during capture ("Writes not supported"). Do NOT run an eager denoise step
        #    here — build_eager already JIT'd the stage-forward + hop-socket kernels
        #    (identical shapes for zero vs real KV), and a SECOND eager use of the hop
        #    sockets deadlocks. Inside capture, _emit_step ops are only RECORDED (not
        #    executed → no block); at replay the trace manages the socket handshakes.
        self._kv_socket_round(bounds, in_trace=False)  # syncs all 16 chips internally
        # 6. ONE capture spanning prefill + KV handoff + N denoise steps.
        #    KNOWN ISSUE (WIP): this records all stages but end_trace_capture below
        #    hangs on trace_root — socket-op finalization under a parent-rooted trace.
        self._trace_cat_bufs = []  # resident in-trace concat buffers (see _kv_socket_round)
        self._e2e_trace_id = ttnn.begin_trace_capture(self._trace_root, cq_id=0)
        self._prefill_compute_to_persist()
        self._kv_socket_round(bounds, in_trace=True)
        for i in range(self.num_denoising_steps):
            self._driver.step(i)
        ttnn.end_trace_capture(self._trace_root, self._e2e_trace_id, cq_id=0)
        self._prefill_trace_key = self._artifact_mask_key
        self._socket_built = True

    def _single_trace_replay(self) -> "torch.Tensor":
        """Per-chunk single-trace replay: reseed noise into _x_t (outside the trace),
        execute the ONE e2e trace on trace_root, read the actions. Inputs
        (pixel/lang) already refreshed by the caller."""
        self._driver.reseed_noise(self._build_noise_torch())
        ttnn.execute_trace(self._trace_root, self._e2e_trace_id, cq_id=0, blocking=True)
        return self._driver.read_actions()

    def _single_trace_host_chunks(self, images, lang_tokens, n):
        """Pre-stage n host input tuples (pixel, lang, noise) for the timed loops."""
        pixel_host = self._stack_and_fold_pixels(images)
        ah, ah_pad, ad = self.action_horizon, self._action_horizon_padded, self.action_dim
        chunks = []
        for _ in range(n):
            h_pix = ttnn.from_torch(
                pixel_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.prefill_mesh, dim=0),
            )
            h_lang = ttnn.from_torch(lang_tokens.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            noise_pad = torch.zeros(1, ah_pad, ad, dtype=torch.float32)
            noise_pad[:, :ah, :] = torch.randn(1, ah, ad)
            h_noise = ttnn.from_torch(noise_pad, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            chunks.append((h_pix, h_lang, h_noise))
        return chunks

    def _single_trace_2cq_loop(self, images, lang_tokens, iters):
        """Single-trace 2CQ replay: next chunk's (pixel, lang, noise) H2D on CQ1
        overlapped with the current chunk's ONE e2e trace on CQ0. Events on
        trace_root. Mirrors pipeline_1x8.sample_actions_traced_2cq_loop."""
        import time as _time

        root = self._trace_root
        ah = self.action_horizon
        host_chunks = self._single_trace_host_chunks(images, lang_tokens, iters + 1)
        h0 = host_chunks[0]
        ttnn.copy_host_to_device_tensor(h0[0], self.pixel_values_buf, cq_id=1)
        ttnn.copy_host_to_device_tensor(h0[1], self.lang_tokens_buf, cq_id=1)
        ttnn.copy_host_to_device_tensor(h0[2], self._driver.x_t, cq_id=1)
        write_event = ttnn.record_event(root, 1)
        times_ms, last_actions = [], None
        for i in range(iters):
            t0 = _time.perf_counter()
            ttnn.wait_for_event(0, write_event)
            ttnn.execute_trace(root, self._e2e_trace_id, cq_id=0, blocking=False)
            op_event = ttnn.record_event(root, 0)
            if i + 1 < iters:
                hn = host_chunks[i + 1]
                ttnn.wait_for_event(1, op_event)
                ttnn.copy_host_to_device_tensor(hn[0], self.pixel_values_buf, cq_id=1)
                ttnn.copy_host_to_device_tensor(hn[1], self.lang_tokens_buf, cq_id=1)
                ttnn.copy_host_to_device_tensor(hn[2], self._driver.x_t, cq_id=1)
                write_event = ttnn.record_event(root, 1)
            last_actions = ttnn.to_torch(self._driver.x_t)[:, :ah, :]
            times_ms.append((_time.perf_counter() - t0) * 1000.0)
        if last_actions is None:
            raise RuntimeError("iters must be >= 1")
        return last_actions, times_ms

    def _single_trace_1cq_loop(self, images, lang_tokens, iters):
        """Single-trace 1CQ baseline: (pixel, lang, noise) H2D serialized on CQ0
        before the ONE e2e trace. Pre-staged host inputs (matches the 2CQ loop's
        host amortization)."""
        import time as _time

        root = self._trace_root
        ah = self.action_horizon
        host_chunks = self._single_trace_host_chunks(images, lang_tokens, iters)
        times_ms, last_actions = [], None
        for i in range(iters):
            t0 = _time.perf_counter()
            hi_pix, hi_lang, hi_noise = host_chunks[i]
            ttnn.copy_host_to_device_tensor(hi_pix, self.pixel_values_buf)
            ttnn.copy_host_to_device_tensor(hi_lang, self.lang_tokens_buf)
            ttnn.copy_host_to_device_tensor(hi_noise, self._driver.x_t)
            ttnn.execute_trace(root, self._e2e_trace_id, cq_id=0, blocking=True)
            last_actions = ttnn.to_torch(self._driver.x_t)[:, :ah, :]
            times_ms.append((_time.perf_counter() - t0) * 1000.0)
        if last_actions is None:
            raise RuntimeError("iters must be >= 1")
        return last_actions, times_ms

    def _socket_teardown(self) -> None:
        """Release the socket-e2e setup so a prompt change can rebuild it without
        leaking (multi-task / 400-ep). Dropping socket refs frees their L1 config
        buffers via the MeshSocket destructor."""
        if self._e2e_trace_id is not None:
            try:
                ttnn.release_trace(self._trace_root, self._e2e_trace_id)
            except Exception:
                pass
            self._e2e_trace_id = None
        if getattr(self, "_trace_cat_bufs", None):
            for t in self._trace_cat_bufs:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
            self._trace_cat_bufs = None
        if self._prefill_trace_id is not None:
            try:
                ttnn.release_trace(self.prefill_mesh, self._prefill_trace_id)
            except Exception:
                pass
            self._prefill_trace_id = None
            self._prefill_trace_key = None
        if getattr(self, "_kv_recv_tids", None):
            for dst, tid in self._kv_recv_tids:
                try:
                    ttnn.release_trace(dst, tid)
                except Exception:
                    pass
            self._kv_recv_tids = None
        if getattr(self, "_kv_cat_recv", None):
            for c in range(_NUM_CHIPS_REQUIRED):
                for w, rs, rv, n, lo in self._kv_cat_recv[c]:
                    try:
                        ttnn.deallocate(rv)
                    except Exception:
                        pass
        self._kv_cat_recv = None
        self._kv_cat_send = None  # drop socket refs -> dtor frees L1 config buffers
        self._close_driver()  # frees denoise weights + traces
        self._socket_built = False

    def _socket_chunk_prefill(self) -> None:
        """Dispatch the prefill trace on CQ0 (non-blocking): vision + prefix + TP=8
        prefill -> _kv_persist, then per-chip KV concat + socket SEND (all captured
        in the trace). Consumes pixel_values_buf / lang_tokens_buf. Split out of
        _run_socket_chunk so the 2CQ loop can record an event right after the inputs
        are consumed and stage the next chunk on CQ1."""
        ttnn.execute_trace(self.prefill_mesh, self._prefill_trace_id, cq_id=0, blocking=False)

    def _socket_chunk_denoise(self) -> "torch.Tensor":
        """Fully-traced KV handoff + denoise, NO host sync. Assumes
        _socket_chunk_prefill already dispatched the prefill trace on CQ0 — which now
        includes the KV concat + socket SEND as its tail. Here: replay the per-stage
        recv traces (recv + DRAM->L1 copy) and the denoise loop trace. All
        non-blocking; the traced recv waits for the traced send via the socket, and
        rerun()'s replay_loop drains stage0 (transitively gating every stage's recv).
        No eager concat/send and no synchronize_device barriers on the hot path."""
        for dst, tid in self._kv_recv_tids:
            ttnn.execute_trace(dst, tid, cq_id=0, blocking=False)
        return self._driver.rerun(self._build_noise_torch())

    def _run_socket_chunk(self) -> "torch.Tensor":
        """Per-chunk e2e replay, fully traced, NO host bounce.

        Single-trace mode: ONE execute_trace on trace_root (prefill + KV + denoise).
        Multi-trace mode: prefill trace -> per-stage recv traces -> denoise loop trace.
        """
        if self._single_trace:
            return self._single_trace_replay()
        self._socket_chunk_prefill()
        return self._socket_chunk_denoise()

    def sample_actions_socket_2cq_loop(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        iters: int,
    ) -> Tuple["torch.Tensor", List[float]]:
        """Socket-KV e2e replay loop with the NEXT chunk's input H2D on the
        prefill mesh's CQ1 overlapped with the current chunk's compute on CQ0.

        Requires PI0_KV_SOCKET=1 and the mesh opened with num_command_queues=2
        (see open_decode_16_mesh). Returns (last_actions, per_iter_wall_ms).

        The prefill trace consumes pixel_values_buf / lang_tokens_buf; we record
        an event right after it dispatches so CQ1 can stage the next chunk's
        inputs while CQ0 runs the KV sockets + streamed denoise. Noise is
        refreshed per chunk inside the denoise driver (denoise mesh) and is NOT
        part of the 2CQ overlap. Mirrors pipeline_1x8.sample_actions_traced_2cq_loop.
        """
        import time as _time

        if not self._socket_mode:
            raise RuntimeError("sample_actions_socket_2cq_loop requires PI0_KV_SOCKET=1")
        mesh = self.prefill_mesh

        # One-time: masks/artifacts, socket e2e build (prefill trace + sockets +
        # denoise loop trace), and a warm chunk to JIT everything.
        if self._expert_attn_mask_torch is None:
            self._build_upstream_artifacts()
        self._ensure_persistent_input_buffers(images, lang_tokens)
        if not self._socket_built or self._prefill_trace_key != self._artifact_mask_key:
            self._socket_e2e_setup()
        _ = self._run_socket_chunk()  # warm replay (not timed)

        if self._single_trace:
            return self._single_trace_2cq_loop(images, lang_tokens, iters)

        # Pre-stage host input tensors for iters+1 chunks (host prep out of the
        # timed loop; the +1 is the initial CQ1 pre-stage).
        pixel_host = self._stack_and_fold_pixels(images)
        host_chunks: List[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = []
        for _ in range(iters + 1):
            h_pix = ttnn.from_torch(
                pixel_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            h_lang = ttnn.from_torch(
                lang_tokens.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            host_chunks.append((h_pix, h_lang))

        # Pre-stage chunk 0 inputs on CQ1.
        h0_pix, h0_lang = host_chunks[0]
        ttnn.copy_host_to_device_tensor(h0_pix, self.pixel_values_buf, cq_id=1)
        ttnn.copy_host_to_device_tensor(h0_lang, self.lang_tokens_buf, cq_id=1)
        write_event = ttnn.record_event(mesh, 1)

        times_ms: List[float] = []
        last_actions = None
        for i in range(iters):
            t0 = _time.perf_counter()
            # CQ0 waits until CQ1 finished staging this iter's inputs.
            ttnn.wait_for_event(0, write_event)
            self._socket_chunk_prefill()  # prefill trace on CQ0 (non-blocking)
            op_event = ttnn.record_event(mesh, 0)  # prefill trace has consumed the inputs

            # Stage next iter's inputs on CQ1, overlapped with CQ0 sockets + denoise.
            if i + 1 < iters:
                hn_pix, hn_lang = host_chunks[i + 1]
                ttnn.wait_for_event(1, op_event)
                ttnn.copy_host_to_device_tensor(hn_pix, self.pixel_values_buf, cq_id=1)
                ttnn.copy_host_to_device_tensor(hn_lang, self.lang_tokens_buf, cq_id=1)
                write_event = ttnn.record_event(mesh, 1)

            last_actions = self._socket_chunk_denoise()  # sockets + streamed denoise -> actions
            times_ms.append((_time.perf_counter() - t0) * 1000.0)

        if last_actions is None:
            raise RuntimeError("iters must be >= 1")
        return last_actions, times_ms

    def sample_actions_socket_1cq_loop(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        iters: int,
    ) -> Tuple["torch.Tensor", List[float]]:
        """Single-CQ socket-KV e2e replay with host input tensors PRE-STAGED
        before the timed loop (mirrors the 2CQ loop's host amortization). The
        input H2D is serialized on CQ0 before the prefill trace. Use as the 1CQ
        baseline for the 2CQ loop — the difference is how much of the input DMA
        hides behind compute. Mirrors pipeline_1x8.sample_actions_traced_1cq_prestaged_loop.
        """
        import time as _time

        if not self._socket_mode:
            raise RuntimeError("sample_actions_socket_1cq_loop requires PI0_KV_SOCKET=1")

        if self._expert_attn_mask_torch is None:
            self._build_upstream_artifacts()
        self._ensure_persistent_input_buffers(images, lang_tokens)
        if not self._socket_built or self._prefill_trace_key != self._artifact_mask_key:
            self._socket_e2e_setup()
        _ = self._run_socket_chunk()  # warm replay (not timed)

        if self._single_trace:
            return self._single_trace_1cq_loop(images, lang_tokens, iters)

        pixel_host = self._stack_and_fold_pixels(images)
        host_chunks: List[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = []
        for _ in range(iters):
            h_pix = ttnn.from_torch(
                pixel_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.prefill_mesh, dim=0),
            )
            h_lang = ttnn.from_torch(lang_tokens.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            host_chunks.append((h_pix, h_lang))

        times_ms: List[float] = []
        last_actions = None
        for i in range(iters):
            t0 = _time.perf_counter()
            hi_pix, hi_lang = host_chunks[i]
            ttnn.copy_host_to_device_tensor(hi_pix, self.pixel_values_buf)
            ttnn.copy_host_to_device_tensor(hi_lang, self.lang_tokens_buf)
            last_actions = self._run_socket_chunk()
            times_ms.append((_time.perf_counter() - t0) * 1000.0)

        if last_actions is None:
            raise RuntimeError("iters must be >= 1")
        return last_actions, times_ms

    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: Optional[List[torch.Tensor]] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,  # unused
    ) -> torch.Tensor:
        if lang_tokens is None:
            raise ValueError("lang_tokens required")
        if img_masks is not None and lang_masks is not None:
            self.prepare_runtime_masks(img_masks, lang_masks)
        elif self._expert_attn_mask_torch is None:
            self._build_upstream_artifacts()

        import time as _time

        _timed = os.environ.get("PI0_TIMED", "").lower() in ("1", "true", "yes", "on")

        def _mark():
            if not _timed:
                return 0.0
            ttnn.synchronize_device(self.prefill_mesh)
            for _m in self.denoise_per_chip:
                ttnn.synchronize_device(_m)
            return _time.perf_counter()

        _t0 = _mark()
        self._ensure_persistent_input_buffers(images, lang_tokens)
        _t_vision = _mark()

        # e2e on-device socket KV path (no host bounce): prefill trace sends KV via
        # sockets, per-chip recv traces land it, denoise loop consumes it.
        if self._socket_mode:
            if not self._socket_built or self._prefill_trace_key != self._artifact_mask_key:
                self._socket_e2e_setup()
            result = self._run_socket_chunk()
            if _timed:
                _td = _mark()
                print(f"[TIMED-SOCKET] e2e_chunk={(_td-_t0)*1e3:6.1f}ms (no host bounce)", flush=True)
            return result

        # Traced vision+prefix+TP=8 prefill: capture once per prompt, replay per chunk
        # (replaces the eager ~150ms vision+prefill). KV lands in self._kv_persist.
        self._ensure_prefill_trace()
        per_layer_kv = self._prefill_via_trace()
        _t_prefill = _mark()

        prefix_kv_torch = self._prefill_kv_to_torch(per_layer_kv)
        _t_kv = _mark()

        # Build the streamed denoise pipeline ONCE per (num_steps, mask) config and
        # reuse it across chunks (rebuilding per chunk leaks L1-pinned weights). On
        # reuse, refresh the fixed-shape KV buffers in place and replay the trace.
        # NOTE: device-direct socket KV (M1) is WIP — _setup_kv_sockets/_send_kv_via_sockets
        # exist but hit a denoise-L1-headroom vs socket-L1 conflict on dense stages;
        # the host-bounce refresh below is the verified-correct path (11/11 parity).
        build_key = (self.num_denoising_steps, self._artifact_mask_key)
        if self._driver is None or self._driver_build_key != build_key:
            self._close_driver()
            self._driver = build_denoise_loop_pipeline(
                self._ref_blocks,
                self._final_w,
                self._final_b,
                self._ref_suffix,
                self.config,
                self._suffix_cfg,
                self.denoise_mesh,
                adarms_cond_per_step=self._adarms_cond_per_step_torch(),
                prefix_kv_cache=prefix_kv_torch,
                prefix_len=self._prefix_len,
                suffix_len=perf_suffix_len(self.action_horizon),
                attention_mask_torch=self._expert_attn_mask_torch,
                position_offset=self._position_offset,
                num_steps=self.num_denoising_steps,
                action_horizon=self.action_horizon,
                splits=_DENOISE_SPLITS_8,
                submeshes=self.denoise_per_chip,
                block_cls=TTNNPi05DenoiseExpertBlock,
                use_concat_kv=True,
                drain="stage0",
            )
            self._driver_build_key = build_key
            result = self._driver.stream_euler(self._build_noise_torch(), capture=True)
        else:
            self._driver.refresh_prefix_kv(prefix_kv_torch)
            result = self._driver.rerun(self._build_noise_torch())
        _t_denoise = _mark()

        if _timed:
            print(
                f"input_h2d={(_t_vision-_t0)*1e3:6.1f}ms  "
                f"prefill_traced={(_t_prefill-_t_vision)*1e3:6.1f}ms  "
                f"kv_download={(_t_kv-_t_prefill)*1e3:6.1f}ms  "
                f"denoise={(_t_denoise-_t_kv)*1e3:6.1f}ms  "
                f"TOTAL={(_t_denoise-_t0)*1e3:6.1f}ms",
                flush=True,
            )
        return result

    def close(self) -> None:
        if self._prefill_trace_id is not None:
            try:
                ttnn.release_trace(self.prefill_mesh, self._prefill_trace_id)
            except Exception:
                pass
            self._prefill_trace_id = None
        self._close_driver()
