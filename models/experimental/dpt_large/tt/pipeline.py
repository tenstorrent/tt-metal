# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import torch

from .config import DPTLargeConfig
from .fallback import DPTFallbackPipeline
from .fusion_head import DPTFusionHead
from .reassembly import DPTReassembly
from .vit_backbone import DPTViTBackboneTTNN
from .tt_configs import describe_configs
from .perf_counters import PERF_COUNTERS

try:
    import ttnn
except Exception:  # pragma: no cover
    ttnn = None

LOG = logging.getLogger(__name__)


@dataclass
class DPTTTPipeline:
    """
    Thin wrapper that mirrors the full TTNN execution path.

    In dev environments without hardware we keep everything on the host while
    preserving the same interfaces so perf-tuning can happen later.
    """

    config: DPTLargeConfig = field(default_factory=DPTLargeConfig)
    pretrained: bool = True
    device: str = "cpu"
    tt_device_override: Optional[Any] = None

    def __post_init__(self):
        # Avoid mutating a caller-provided config (and never mutate the module-level DEFAULT_CONFIG).
        self.config = copy.deepcopy(self.config)

        self.fallback = DPTFallbackPipeline(config=self.config, pretrained=self.pretrained, device=self.device)
        # Align neck shapes with HF when using pretrained weights
        # Align fusion/head flags too (BN usage, optional projection)
        try:
            self.config.use_batch_norm_in_fusion_residual = bool(
                self.fallback._model.config.use_batch_norm_in_fusion_residual
            )
            self.config.add_projection = bool(getattr(self.fallback._model.config, "add_projection", False))
        except Exception:
            pass
        try:
            hf_neck_sizes = list(self.fallback._model.config.neck_hidden_sizes)
            if isinstance(hf_neck_sizes, list) and len(hf_neck_sizes) == len(self.config.neck_hidden_sizes):
                self.config.neck_hidden_sizes = hf_neck_sizes
        except Exception:
            pass
        # Reuse the same HF weights for the "TT" path; later this can be swapped
        # with real TTNN modules.
        self.tt_layer_configs = describe_configs(self.config)
        self.backbone = DPTViTBackboneTTNN(
            config=self.config,
            hf_model=self.fallback._model,
            pretrained=self.pretrained,
            device=self.device,
            tt_layer_cfg=self.tt_layer_configs["vit_block"],
            tt_device_override=self.tt_device_override,
        )
        tt_dev = getattr(self.backbone, "tt_device", None)
        self.reassembly = DPTReassembly(
            config=self.config, tt_device=tt_dev, layer_cfg=self.tt_layer_configs["cnn_block"]
        )
        self.fusion_head = DPTFusionHead(
            config=self.config, tt_device=tt_dev, layer_cfg=self.tt_layer_configs["cnn_block"]
        )
        # Mirror HF neck + head weights so the TT path is numerically aligned
        # with the reference model, even when `pretrained=False`.
        state_dict = self.fallback._model.state_dict()
        if hasattr(self.reassembly, "load_from_hf_state_dict"):
            self.reassembly.load_from_hf_state_dict(state_dict)
        if hasattr(self.fusion_head, "load_from_hf_state_dict"):
            self.fusion_head.load_from_hf_state_dict(state_dict)
        self.to(self.device)
        self.eval()
        # Last per-call perf breakdown (filled in forward).
        self.last_perf: Optional[dict] = None
        self._fusion_head_tracer = None
        self._fusion_trace_unavailable_reason: Optional[str] = None
        self._full_trace_id = None
        self._full_trace_input = None
        self._full_trace_output = None
        self._trace_op_event = None
        self._full_trace_unavailable_reason: Optional[str] = None

    # ------------------------------------------------------------------ plumbing
    def to(self, device: str):
        self.device = device
        self.backbone.to(device)
        self.reassembly.to(device)
        self.fusion_head.to(device)
        return self

    def eval(self):
        self.backbone.eval()
        self.reassembly.eval()
        self.fusion_head.eval()
        return self

    def close(self):
        if self._full_trace_id is not None:
            try:
                import ttnn  # type: ignore

                ttnn.release_trace(self.backbone.tt_device, self._full_trace_id)
            except Exception:
                pass
            self._full_trace_id = None
            self._full_trace_input = None
            self._full_trace_output = None
            self._trace_op_event = None
        if self._fusion_head_tracer is not None:
            try:
                self._fusion_head_tracer.release()
            except Exception:
                pass
            self._fusion_head_tracer = None
        if hasattr(self.backbone, "close"):
            self.backbone.close()
        return None

    def _forward_fusion_head(self, pyramid, execution_mode: str):
        if execution_mode == "eager":
            return self.fusion_head(pyramid)

        if self._fusion_trace_unavailable_reason is not None:
            return self.fusion_head(pyramid)

        if self._fusion_head_tracer is None:
            from models.tt_dit.utils.tracing import Tracer

            self._fusion_head_tracer = Tracer(self.fusion_head, device=self.backbone.tt_device)

        trace_cq_id = 0
        if execution_mode == "trace_2cq":
            LOG.warning("trace_2cq requested; executing traced fusion head on cq=0 until explicit 2CQ wiring is added")
        try:
            return self._fusion_head_tracer(
                pyramid,
                tracer_cq_id=trace_cq_id,
                tracer_blocking_execution=True,
            )
        except Exception:
            # Shape/address changes can invalidate an existing trace; rebuild once.
            self._fusion_head_tracer.release()
            try:
                return self._fusion_head_tracer(
                    pyramid,
                    tracer_cq_id=trace_cq_id,
                    tracer_blocking_execution=True,
                )
            except Exception as exc:
                self._fusion_trace_unavailable_reason = str(exc)
                LOG.warning(
                    "Fusion-head trace is unavailable (%s). Falling back to eager fusion head.",
                    self._fusion_trace_unavailable_reason,
                )
                return self.fusion_head(pyramid)

    def _tt_forward_core(self, tt_pixel_values):
        # In perf-neck mode, request TT token maps/features so the neck can stay on device.
        return_tt_backbone = bool(getattr(self.config, "tt_perf_neck", False))
        feats = self.backbone.forward_tt_input(tt_pixel_values, return_tt=return_tt_backbone)
        pyramid = self.reassembly(feats)
        depth = self.fusion_head(pyramid)
        return depth

    def _ensure_full_trace(self, tt_pixel_values_host, execution_mode: str):
        """
        Capture a single full-model trace for the backbone+neck+head hot path.

        The trace uses cq=0 for execution. For `trace_2cq`, the input copy is
        scheduled on cq=1 and synchronized via events (similar to other performant
        runners in the repo).
        """
        import ttnn  # type: ignore

        if self._full_trace_id is not None:
            return

        if self.backbone.tt_device is None:
            raise RuntimeError("TT device is not available for tracing")

        # Create a persistent device input buffer with a stable address.
        tt_in_dev = tt_pixel_values_host.to(self.backbone.tt_device)
        self._full_trace_input = tt_in_dev

        # compile
        _ = self._tt_forward_core(tt_in_dev)

        # capture trace
        trace_id = ttnn.begin_trace_capture(self.backbone.tt_device, cq_id=0)
        try:
            try:
                out = self._tt_forward_core(tt_in_dev)
            finally:
                ttnn.end_trace_capture(self.backbone.tt_device, trace_id, cq_id=0)
        except Exception:
            try:
                ttnn.release_trace(self.backbone.tt_device, trace_id)
            except Exception:
                pass
            raise

        self._full_trace_id = trace_id
        self._full_trace_output = out

        # Initialize CQ sync primitive for trace_2cq.
        if execution_mode == "trace_2cq":
            self._trace_op_event = ttnn.record_event(self.backbone.tt_device, 0)

    def forward_tt_host_tensor(self, tt_pixel_values_host, normalize: bool = True):
        """
        Forward using a TT host tensor input.

        This is intended for traced execution modes. The caller should pass a
        `ttnn.Tensor` created with `ttnn.from_torch(..., layout=TILE_LAYOUT)` and
        no device, so input copies can be scheduled efficiently.
        """
        try:
            import ttnn  # type: ignore
        except Exception:
            raise RuntimeError("ttnn is required for forward_tt_host_tensor")

        requested_exec_mode = str(getattr(self.config, "tt_execution_mode", "eager")).lower()
        if requested_exec_mode not in {"trace", "trace_2cq"}:
            raise ValueError(f"forward_tt_host_tensor requires trace/trace_2cq, got {requested_exec_mode!r}")

        if self.backbone.tt_device is None:
            raise RuntimeError("TT device is not available for traced execution")

        if self._full_trace_unavailable_reason is None:
            try:
                self._ensure_full_trace(tt_pixel_values_host, execution_mode=requested_exec_mode)
            except Exception as exc:
                self._full_trace_unavailable_reason = str(exc)
                LOG.warning(
                    "Full TT trace capture is unavailable (%s). Falling back to non-full-trace execution.",
                    self._full_trace_unavailable_reason,
                )

        if self._full_trace_unavailable_reason is not None or self._full_trace_id is None:
            # Graceful fallback: convert host TT input back to torch and run the
            # existing eager/partial-trace path.
            pv_torch = ttnn.to_torch(tt_pixel_values_host)
            return self.forward_pixel_values(pv_torch, normalize=normalize)

        t0 = time.perf_counter()
        try:
            if requested_exec_mode == "trace_2cq":
                # 2-CQ wiring: host->device copy on cq=1, trace execute on cq=0.
                if self._trace_op_event is None:
                    self._trace_op_event = ttnn.record_event(self.backbone.tt_device, 0)
                ttnn.wait_for_event(1, self._trace_op_event)
                ttnn.copy_host_to_device_tensor(tt_pixel_values_host, self._full_trace_input, 1)
                write_event = ttnn.record_event(self.backbone.tt_device, 1)
                ttnn.wait_for_event(0, write_event)
                ttnn.execute_trace(self.backbone.tt_device, self._full_trace_id, cq_id=0, blocking=False)
                # Record completion after enqueuing trace execution and wait for
                # it before reading from the persistent trace output tensor.
                completion_event = ttnn.record_event(self.backbone.tt_device, 0)
                ttnn.wait_for_event(0, completion_event)
                self._trace_op_event = completion_event
            else:
                ttnn.copy_host_to_device_tensor(tt_pixel_values_host, self._full_trace_input, 0)
                ttnn.execute_trace(self.backbone.tt_device, self._full_trace_id, cq_id=0, blocking=True)
        except Exception as exc:
            self._full_trace_unavailable_reason = str(exc)
            LOG.warning(
                "Full TT trace execution failed (%s). Falling back to non-full-trace execution.",
                self._full_trace_unavailable_reason,
            )
            pv_torch = ttnn.to_torch(tt_pixel_values_host)
            return self.forward_pixel_values(pv_torch, normalize=normalize)

        depth = self._full_trace_output
        # Normalize/return path mirrors the eager TT path.
        try:
            if isinstance(depth, ttnn.Tensor):
                depth = depth.cpu().to_torch()
        except Exception:
            pass
        depth_t = torch.as_tensor(depth)
        if depth_t.dim() == 3:
            depth_t = depth_t.unsqueeze(1)
        if normalize:
            depth_t = self.fallback._normalize_depth(depth_t.float())
        else:
            depth_t = depth_t.float()

        total_ms = (time.perf_counter() - t0) * 1000.0
        self.last_perf = {
            "mode": "tt",
            "execution_mode": requested_exec_mode,
            "effective_execution_mode": requested_exec_mode,
            "requested_execution_mode": requested_exec_mode,
            "num_images": 1,
            "preprocess_ms": 0.0,
            "backbone_ms": None,
            "reassembly_ms": None,
            "fusion_head_ms": None,
            "normalize_ms": None,
            "total_ms": total_ms,
            "fallback_counts": PERF_COUNTERS.snapshot(),
            "full_trace_unavailable_reason": self._full_trace_unavailable_reason,
            "fusion_trace_unavailable_reason": self._fusion_trace_unavailable_reason,
        }

        return depth_t.cpu().numpy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ------------------------------------------------------------------ forward
    def _forward_preprocessed(self, preprocessed: list[torch.Tensor], normalize: bool, preprocess_ms: float):
        outputs: list[np.ndarray] = []
        backbone_ms = 0.0
        reassembly_ms = 0.0
        fusion_head_ms = 0.0
        normalize_ms = 0.0
        t_total = time.perf_counter()
        use_tt_neck_head = bool(self.config.tt_device_reassembly or self.config.tt_device_fusion)
        requested_exec_mode = str(getattr(self.config, "tt_execution_mode", "eager")).lower()
        if requested_exec_mode not in {"eager", "trace", "trace_2cq"}:
            raise ValueError(f"Unsupported tt_execution_mode={requested_exec_mode!r}")
        effective_exec_mode = requested_exec_mode
        if requested_exec_mode in {"trace", "trace_2cq"} and self._full_trace_unavailable_reason is not None:
            effective_exec_mode = "eager"

        with torch.no_grad():
            for pv in preprocessed:
                # Strict path (no TT neck/head flags): use HF neck+head for perfect parity.
                if not use_tt_neck_head:
                    t1 = time.perf_counter()
                    depth = self.fallback._forward(pv)
                    # HF returns [B,1,H,W]; treat as torch tensor for normalization below.
                    reassembly_ms += 0.0
                    fusion_head_ms += (time.perf_counter() - t1) * 1000.0
                    depth_t = torch.as_tensor(depth)
                else:
                    t0 = time.perf_counter()
                    # In perf-neck mode, request TT token maps/features so the neck can stay on device.
                    # In non-perf mode, keep backbone outputs on host to preserve the reference behavior.
                    return_tt_backbone = bool(getattr(self.config, "tt_perf_neck", False))
                    feats = self.backbone(pv.to(self.device), return_tt=return_tt_backbone)
                    backbone_ms += (time.perf_counter() - t0) * 1000.0

                    t1 = time.perf_counter()
                    pyramid = self.reassembly(feats)
                    reassembly_ms += (time.perf_counter() - t1) * 1000.0

                    t2 = time.perf_counter()
                    depth = self._forward_fusion_head(pyramid, execution_mode=effective_exec_mode)
                    if requested_exec_mode in {"trace", "trace_2cq"} and self._fusion_trace_unavailable_reason is not None:
                        effective_exec_mode = "eager"
                    fusion_head_ms += (time.perf_counter() - t2) * 1000.0
                    # depth may be a TT tensor, torch tensor, or numpy array; ensure
                    # we normalize and return a float32 torch tensor.
                    try:
                        import ttnn

                        if isinstance(depth, ttnn.Tensor):
                            depth = depth.cpu().to_torch()
                    except Exception:
                        pass
                    depth_t = torch.as_tensor(depth)
                    # Ensure channel dimension matches CPU fallback ([B,1,H,W]) before normalization.
                    if depth_t.dim() == 3:
                        depth_t = depth_t.unsqueeze(1)
                t3 = time.perf_counter()
                if normalize:
                    depth_t = self.fallback._normalize_depth(depth_t.float())
                else:
                    depth_t = depth_t.float()
                normalize_ms += (time.perf_counter() - t3) * 1000.0
                outputs.append(depth_t.cpu().numpy())

        total_ms = (time.perf_counter() - t_total) * 1000.0
        self.last_perf = {
            "mode": "tt",
            "execution_mode": effective_exec_mode,
            "effective_execution_mode": effective_exec_mode,
            "requested_execution_mode": requested_exec_mode,
            "num_images": len(preprocessed),
            "preprocess_ms": preprocess_ms,
            "backbone_ms": backbone_ms,
            "reassembly_ms": reassembly_ms,
            "fusion_head_ms": fusion_head_ms,
            "normalize_ms": normalize_ms,
            "total_ms": total_ms,
            "fallback_counts": PERF_COUNTERS.snapshot(),
            "full_trace_unavailable_reason": self._full_trace_unavailable_reason,
            "fusion_trace_unavailable_reason": self._fusion_trace_unavailable_reason,
        }

        return outputs

    def forward_pixel_values(
        self, pixel_values: torch.Tensor | list[torch.Tensor], normalize: bool = True
    ) -> np.ndarray | list[np.ndarray]:
        """
        Forward using already-preprocessed tensors (avoids disk I/O in perf runs).
        """
        values = pixel_values if isinstance(pixel_values, list) else [pixel_values]
        # Reset per-call perf breakdown.
        self.last_perf = None
        # If no TT device available, either fall back to HF path (when allowed)
        # or raise in order to surface configuration issues in tests.
        if getattr(self.backbone, "tt_device", None) is None:
            if not self.config.allow_cpu_fallback:
                raise RuntimeError(
                    "TT device is not available but `allow_cpu_fallback=False`. "
                    "This configuration is intended to exercise the TT path."
                )
            t_start = time.perf_counter()
            outs = []
            for pv in values:
                # Mirror the CPU fallback behavior: pv is already prepared.
                depth = self.fallback._forward(pv)
                depth_t = torch.as_tensor(depth)
                if depth_t.dim() == 3:
                    depth_t = depth_t.unsqueeze(1)
                if normalize:
                    depth_t = self.fallback._normalize_depth(depth_t.float())
                else:
                    depth_t = depth_t.float()
                outs.append(depth_t.cpu().numpy())
            total_ms = (time.perf_counter() - t_start) * 1000.0
            self.last_perf = {
                "mode": "cpu_fallback",
                "execution_mode": "cpu_fallback",
                "total_ms": total_ms,
                "num_images": len(values),
                "fallback_counts": PERF_COUNTERS.snapshot(),
            }
            return outs if len(outs) > 1 else outs[0]

        outs = self._forward_preprocessed(values, normalize=normalize, preprocess_ms=0.0)
        return outs if len(outs) > 1 else outs[0]

    def forward(self, image_path: str | list[str], normalize: bool = True) -> np.ndarray | list[np.ndarray]:
        # Support single path or list for simple pipelining/batching.
        paths = image_path if isinstance(image_path, list) else [image_path]
        # Reset per-call perf breakdown.
        self.last_perf = None
        # If no TT device available, either fall back to HF path (when allowed)
        # or raise in order to surface configuration issues in tests.
        if getattr(self.backbone, "tt_device", None) is None:
            return self.forward_pixel_values([self.fallback._prepare(p) for p in paths], normalize=normalize)

        # Lightweight host preprocessing pipeline
        t_pre = time.perf_counter()
        preprocessed = [self.fallback._prepare(p) for p in paths]
        preprocess_ms = (time.perf_counter() - t_pre) * 1000.0
        outs = self._forward_preprocessed(preprocessed, normalize=normalize, preprocess_ms=preprocess_ms)
        return outs if len(outs) > 1 else outs[0]


def run_depth(
    image_path: str,
    use_tt: bool = True,
    config: DPTLargeConfig | None = None,
    **kwargs,
) -> np.ndarray:
    if config is None:
        config = DPTLargeConfig()
    if not use_tt:
        pipe = DPTFallbackPipeline(config=config, **kwargs)
        return pipe.run_depth_cpu(image_path)
    pipe = DPTTTPipeline(config=config, **kwargs)
    return pipe.forward(image_path)
