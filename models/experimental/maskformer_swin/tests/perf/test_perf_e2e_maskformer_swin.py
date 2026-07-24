# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import time
import traceback

from models.common.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    # MaskFormer is trace-heavy (multiple conv2d + normalization blocks). Use a larger trace region
    # to avoid trace buffer exhaustion which can manifest as trace replay hangs.
    [{"l1_small_size": 32768, "trace_region_size": 200_000_000, "num_command_queues": 2}],
    indirect=True,
)
def test_maskformer_swin_b_e2e_trace_2cq(device):
    """End-to-end perf path using mesh trace + 2 command queues.

    This test targets "ideal" E2E performance by:
    - capturing a trace for the full TT forward (no TT->torch readbacks inside trace)
    - using CQ1 for host->device writes into persistent DRAM input
    - using CQ0 for DRAM->L1 staging + trace replay
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("huggingface_hub")

    import ttnn
    import os

    # Fail fast on any TTNN fallback ops. Fallbacks typically imply host read/write and will
    # also crash trace capture ("Reads/Writes are not supported during trace capture").
    if hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
        ttnn.CONFIG.throw_exception_on_fallback = True

    # Optional guardrail to surface accidental host<->device calls with a Python stacktrace
    # instead of a TT_FATAL during trace capture.
    trace_guard = os.environ.get("MASKFORMER_TT_TRACE_GUARD", "0").strip() == "1"
    # Some models use explicit `ttnn.deallocate(...)` inside forward() to manage memory.
    # That pattern can break trace replay because deallocations are host-side and are not
    # re-executed during `execute_trace(...)`. Disable deallocations during the capture
    # forward so the trace can replay reliably.
    disable_deallocate_during_capture = os.environ.get("MASKFORMER_E2E_TRACE_DISABLE_DEALLOCATE", "1").strip() == "1"
    in_trace_capture = False
    _orig_from_torch = getattr(ttnn, "from_torch", None)
    _orig_to_device = getattr(ttnn, "to_device", None)
    _orig_to_torch = getattr(ttnn, "to_torch", None)
    _orig_copy_h2d = getattr(ttnn, "copy_host_to_device_tensor", None)
    _orig_copy_d2h = getattr(ttnn, "copy_device_to_host_tensor", None)
    _orig_deallocate = getattr(ttnn, "deallocate", None)

    def _guarded(fn_name, fn):
        if fn is None:
            return None

        def _wrapped(*args, **kwargs):
            nonlocal in_trace_capture
            if trace_guard and in_trace_capture:
                stack = "".join(traceback.format_stack(limit=50))
                raise RuntimeError(
                    f"[maskformer][trace_guard] illegal call to ttnn.{fn_name} during trace capture\n{stack}"
                )
            return fn(*args, **kwargs)

        return _wrapped

    if trace_guard:
        if _orig_from_torch is not None:
            ttnn.from_torch = _guarded("from_torch", _orig_from_torch)
        if _orig_to_device is not None:
            ttnn.to_device = _guarded("to_device", _orig_to_device)
        if _orig_to_torch is not None:
            ttnn.to_torch = _guarded("to_torch", _orig_to_torch)
        if _orig_copy_h2d is not None:
            ttnn.copy_host_to_device_tensor = _guarded("copy_host_to_device_tensor", _orig_copy_h2d)
        if _orig_copy_d2h is not None:
            ttnn.copy_device_to_host_tensor = _guarded("copy_device_to_host_tensor", _orig_copy_d2h)

    if disable_deallocate_during_capture and _orig_deallocate is not None:
        # Keep deallocations outside capture (warmup/teardown), but skip them inside capture.
        def _deallocate_wrapped(*args, **kwargs):
            nonlocal in_trace_capture
            if in_trace_capture:
                return None
            return _orig_deallocate(*args, **kwargs)

        ttnn.deallocate = _deallocate_wrapped

    # Apply stage3 env defaults to match README perf configuration.
    from models.experimental.maskformer_swin.demo.runner import _configure_optimization_stage

    _configure_optimization_stage("stage3")
    # Trace capture disallows event synchronization. The Swin backbone has a safety
    # synchronize_device() on the windows==1 attention path; disable it for tracing.
    os.environ["MASKFORMER_TT_SYNC_WINDOW_ATTN_WINDOWS1"] = "0"
    # Optional: override pixel decoder group norm implementation for trace debugging.
    # Values: "" (default stage3), "native", "manual", "moreh"
    gn_impl = os.environ.get("MASKFORMER_E2E_TRACE_GN_IMPL", "").strip().lower()
    if gn_impl:
        if gn_impl == "native":
            os.environ["MASKFORMER_TT_USE_NATIVE_GROUP_NORM"] = "1"
            os.environ["MASKFORMER_TT_USE_MOREH_GROUP_NORM"] = "0"
        elif gn_impl == "manual":
            os.environ["MASKFORMER_TT_USE_NATIVE_GROUP_NORM"] = "0"
            os.environ["MASKFORMER_TT_USE_MOREH_GROUP_NORM"] = "0"
        elif gn_impl == "moreh":
            os.environ["MASKFORMER_TT_USE_NATIVE_GROUP_NORM"] = "0"
            os.environ["MASKFORMER_TT_USE_MOREH_GROUP_NORM"] = "1"
        else:
            raise ValueError(f"Unsupported MASKFORMER_E2E_TRACE_GN_IMPL={gn_impl!r}")
        print(
            "[maskformer][e2e][trace+2cq] overriding GN impl "
            f"MASKFORMER_TT_USE_NATIVE_GROUP_NORM={os.environ.get('MASKFORMER_TT_USE_NATIVE_GROUP_NORM')} "
            f"MASKFORMER_TT_USE_MOREH_GROUP_NORM={os.environ.get('MASKFORMER_TT_USE_MOREH_GROUP_NORM')}",
            flush=True,
        )
    # Trace capture requires binaries to already be resident on device. Disabling the program cache
    # can force binary uploads inside capture (disallowed), so keep it enabled here.
    if hasattr(device, "clear_program_cache"):
        device.clear_program_cache()

    from models.experimental.maskformer_swin.tt.backbone_swin import MaskFormerSwinBackbone
    from models.experimental.maskformer_swin.tt.heads import MaskFormerHeads, MaskFormerHeadsConfig
    from models.experimental.maskformer_swin.tt.pixel_decoder import MaskFormerPixelDecoder, PixelDecoderConfig
    from models.experimental.maskformer_swin.tt.transformer_decoder import (
        MaskFormerTransformerDecoder,
        TransformerDecoderConfig,
    )
    from models.experimental.maskformer_swin.tt.weights import (
        WeightConversionConfig,
        convert_state_dict_to_tt,
        download_reference_weights,
    )

    model_id = "facebook/maskformer-swin-base-coco"
    weight_cfg = WeightConversionConfig(pretrained_model_name=model_id)
    print(f"[maskformer][e2e][trace+2cq] loading weights '{model_id}' ...", flush=True)
    ref_weights = download_reference_weights(weight_cfg)
    print("[maskformer][e2e][trace+2cq] converting weights ...", flush=True)
    tt_state_dict = convert_state_dict_to_tt(ref_weights.state_dict, weight_cfg)
    ref_cfg = ref_weights.config if isinstance(ref_weights.config, dict) else {}
    print("[maskformer][e2e][trace+2cq] building model ...", flush=True)

    backbone_cfg = ref_cfg.get("backbone_config", {}) if isinstance(ref_cfg, dict) else {}
    decoder_cfg = ref_cfg.get("decoder_config", {}) if isinstance(ref_cfg, dict) else {}

    backbone = MaskFormerSwinBackbone.from_huggingface(tt_state_dict, device=device, config_dict=backbone_cfg)

    pixel_cfg = PixelDecoderConfig(
        fpn_dim=int(ref_cfg.get("fpn_feature_size", 256)),
        mask_dim=int(ref_cfg.get("mask_feature_size", 256)),
    )
    pixel_decoder = MaskFormerPixelDecoder.from_huggingface(tt_state_dict, config=pixel_cfg, device=device)

    transformer_cfg = TransformerDecoderConfig(
        num_layers=int(ref_cfg.get("num_hidden_layers", 6)),
        num_attention_heads=int(ref_cfg.get("num_attention_heads", 8)),
        hidden_dim=int(ref_cfg.get("fpn_feature_size", 256)),
        dim_feedforward=int(decoder_cfg.get("decoder_ffn_dim", 2048)) if isinstance(decoder_cfg, dict) else 2048,
        dropout=float(decoder_cfg.get("dropout", 0.0)) if isinstance(decoder_cfg, dict) else 0.0,
        activation=str(decoder_cfg.get("activation_function", "relu")) if isinstance(decoder_cfg, dict) else "relu",
        in_features=int(pixel_cfg.input_channels[-1]),
        maskformer_config=dict(ref_cfg) if isinstance(ref_cfg, dict) else None,
    )
    transformer_decoder = MaskFormerTransformerDecoder.from_huggingface(
        tt_state_dict, config=transformer_cfg, device=device
    )

    heads_cfg = MaskFormerHeadsConfig(
        num_classes=len(ref_cfg.get("id2label", {})),
        hidden_dim=transformer_cfg.hidden_dim,
        mask_dim=pixel_cfg.fpn_dim,
    )
    heads = MaskFormerHeads(config=heads_cfg, device=device)
    heads.load_weights(tt_state_dict)
    print("[maskformer][e2e][trace+2cq] model ready", flush=True)

    e2e_debug = os.environ.get("MASKFORMER_E2E_TRACE_DEBUG", "0").strip() == "1"

    def forward_tt(image_nhwc_tt):
        if e2e_debug:
            print("[maskformer][e2e][trace+2cq] forward backbone begin", flush=True)
        features, _ = backbone.forward(image_nhwc_tt)
        if e2e_debug:
            print("[maskformer][e2e][trace+2cq] forward backbone done", flush=True)
            print("[maskformer][e2e][trace+2cq] forward pixel_decoder begin", flush=True)
        mask_features, _ = pixel_decoder.forward(features)
        if e2e_debug:
            print("[maskformer][e2e][trace+2cq] forward pixel_decoder done", flush=True)
            print("[maskformer][e2e][trace+2cq] forward transformer_decoder begin", flush=True)
        decoder_last, _, _ = transformer_decoder.forward_tt(features[-1], return_tt_tensor=True)
        if e2e_debug:
            print("[maskformer][e2e][trace+2cq] forward transformer_decoder done", flush=True)
            print("[maskformer][e2e][trace+2cq] forward heads begin", flush=True)
        return heads.forward_tt(decoder_last, mask_features)

    # Optional: trace smaller subgraphs to debug trace replay issues.
    # Values: full (default), backbone, pixel_decoder, decoder.
    # Debug-only values:
    # - pixel_decoder_prep_feats
    # - pixel_decoder_stem_conv_raw, pixel_decoder_stem_conv_raw_reshaped
    # - pixel_decoder_stem_conv_raw_rm_reshaped
    # - pixel_decoder_stem_conv_raw_view
    # - pixel_decoder_stem_conv_raw_cfg_rm_view
    # - pixel_decoder_stem_conv_raw_to_rm_view
    # - pixel_decoder_stem_conv, pixel_decoder_stem_gn, pixel_decoder_stem
    # - pixel_decoder_l0, pixel_decoder_l1, pixel_decoder_l2, pixel_decoder_mask_proj
    trace_subgraph = os.environ.get("MASKFORMER_E2E_TRACE_SUBGRAPH", "full").strip().lower()

    def trace_forward(image_nhwc_tt):
        features, _ = backbone.forward(image_nhwc_tt)
        if trace_subgraph == "backbone":
            return features[-1], features[-1]

        def _pixel_decoder_partial(features, *, stop_after: str):
            # Re-run pixel decoder forward in smaller chunks to isolate trace replay issues.
            # This intentionally calls pixel_decoder internals to avoid mutating model code.
            pd = pixel_decoder
            feats = list(features)
            tt_feats = [pd._to_tt_nhwc(f) for f in feats]
            if stop_after == "pixel_decoder_prep_feats":
                return tt_feats[-1]

            # Stem (C4 / stride32) conv3x3 + GN + ReLU
            if stop_after in (
                "pixel_decoder_stem_conv_raw",
                "pixel_decoder_stem_conv_raw_reshaped",
                "pixel_decoder_stem_conv_raw_rm_reshaped",
                "pixel_decoder_stem_conv_raw_view",
                "pixel_decoder_stem_conv_raw_cfg_rm_view",
                "pixel_decoder_stem_conv_raw_to_rm_view",
            ):
                x_in = tt_feats[-1]
                if hasattr(x_in, "storage_type") and hasattr(ttnn, "StorageType"):
                    if x_in.storage_type() != ttnn.StorageType.DEVICE:
                        x_in = ttnn.to_device(x_in, pd.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if x_in.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                    x_in = ttnn.to_layout(x_in, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                conv_cfg = None
                if stop_after == "pixel_decoder_stem_conv_raw_cfg_rm_view":
                    conv_cfg = ttnn.Conv2dConfig(output_layout=ttnn.ROW_MAJOR_LAYOUT)

                [out, [out_h, out_w]] = ttnn.conv2d(
                    input_tensor=x_in,
                    weight_tensor=pd._tt_weights["fpn_stem_w"],
                    bias_tensor=pd._tt_weights.get("fpn_stem_b"),
                    in_channels=int(x_in.shape[-1]),
                    out_channels=int(pd.config.fpn_dim),
                    batch_size=int(x_in.shape[0]),
                    input_height=int(x_in.shape[1]),
                    input_width=int(x_in.shape[2]),
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    dilation=(1, 1),
                    groups=1,
                    device=pd.device,
                    conv_config=conv_cfg,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    return_output_dim=True,
                    return_weights_and_bias=False,
                )
                if stop_after == "pixel_decoder_stem_conv_raw":
                    return out
                if stop_after == "pixel_decoder_stem_conv_raw_view":
                    if not hasattr(ttnn, "view"):
                        raise RuntimeError("ttnn.view is unavailable in this runtime.")
                    return ttnn.view(out, (int(x_in.shape[0]), int(out_h), int(out_w), int(pd.config.fpn_dim)))
                if stop_after == "pixel_decoder_stem_conv_raw_cfg_rm_view":
                    if not hasattr(ttnn, "view"):
                        raise RuntimeError("ttnn.view is unavailable in this runtime.")
                    return ttnn.view(out, (int(x_in.shape[0]), int(out_h), int(out_w), int(pd.config.fpn_dim)))
                if stop_after == "pixel_decoder_stem_conv_raw_to_rm_view":
                    if not hasattr(ttnn, "view"):
                        raise RuntimeError("ttnn.view is unavailable in this runtime.")
                    out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    return ttnn.view(out_rm, (int(x_in.shape[0]), int(out_h), int(out_w), int(pd.config.fpn_dim)))
                if stop_after == "pixel_decoder_stem_conv_raw_rm_reshaped":
                    out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                return ttnn.reshape(out, (int(x_in.shape[0]), int(out_h), int(out_w), int(pd.config.fpn_dim)))
            x = pd._conv2d(
                tt_feats[-1],
                weight=pd._tt_weights["fpn_stem_w"],
                bias=pd._tt_weights.get("fpn_stem_b"),
                weight_key="fpn_stem_w",
                bias_key="fpn_stem_b",
                out_channels=pd.config.fpn_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            if stop_after == "pixel_decoder_stem_conv":
                return x
            x = pd._group_norm(
                x,
                name="fpn_stem_gn",
                weight=pd._tt_weights["fpn_stem_gn_w"],
                bias=pd._tt_weights["fpn_stem_gn_b"],
                mask=pd._tt_weights["fpn_stem_gn_mask"],
                moreh_weight=pd._tt_weights["fpn_stem_gn_w_moreh"],
                moreh_bias=pd._tt_weights["fpn_stem_gn_b_moreh"],
                manual_weight=pd._tt_weights["fpn_stem_gn_w_manual"],
                manual_bias=pd._tt_weights["fpn_stem_gn_b_manual"],
            )
            if stop_after == "pixel_decoder_stem_gn":
                return x
            if x.get_layout() != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.relu(x)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if stop_after == "pixel_decoder_stem":
                return x

            # Lateral order: stage2 (20x20) -> stage1 (40x40) -> stage0 (80x80)
            lateral_order = [tt_feats[2], tt_feats[1], tt_feats[0]]
            for i, lateral in enumerate(lateral_order):
                lateral_proj = pd._conv2d(
                    lateral,
                    weight=pd._tt_weights[f"fpn_l{i}_proj_w"],
                    bias=pd._tt_weights.get(f"fpn_l{i}_proj_b"),
                    weight_key=f"fpn_l{i}_proj_w",
                    bias_key=f"fpn_l{i}_proj_b",
                    out_channels=pd.config.fpn_dim,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                )
                lateral_proj = pd._group_norm(
                    lateral_proj,
                    name=f"fpn_l{i}_proj_gn",
                    weight=pd._tt_weights[f"fpn_l{i}_proj_gn_w"],
                    bias=pd._tt_weights[f"fpn_l{i}_proj_gn_b"],
                    mask=pd._tt_weights[f"fpn_l{i}_proj_gn_mask"],
                    moreh_weight=pd._tt_weights[f"fpn_l{i}_proj_gn_w_moreh"],
                    moreh_bias=pd._tt_weights[f"fpn_l{i}_proj_gn_b_moreh"],
                    manual_weight=pd._tt_weights[f"fpn_l{i}_proj_gn_w_manual"],
                    manual_bias=pd._tt_weights[f"fpn_l{i}_proj_gn_b_manual"],
                )

                if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                x = ttnn.upsample(x, scale_factor=(2.0, 2.0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if x.get_layout() != ttnn.TILE_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if lateral_proj.get_layout() != ttnn.TILE_LAYOUT:
                    lateral_proj = ttnn.to_layout(lateral_proj, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                x = ttnn.add(x, lateral_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=True)
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                x = pd._conv2d(
                    x,
                    weight=pd._tt_weights[f"fpn_l{i}_block_w"],
                    bias=pd._tt_weights.get(f"fpn_l{i}_block_b"),
                    weight_key=f"fpn_l{i}_block_w",
                    bias_key=f"fpn_l{i}_block_b",
                    out_channels=pd.config.fpn_dim,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                )
                x = pd._group_norm(
                    x,
                    name=f"fpn_l{i}_block_gn",
                    weight=pd._tt_weights[f"fpn_l{i}_block_gn_w"],
                    bias=pd._tt_weights[f"fpn_l{i}_block_gn_b"],
                    mask=pd._tt_weights[f"fpn_l{i}_block_gn_mask"],
                    moreh_weight=pd._tt_weights[f"fpn_l{i}_block_gn_w_moreh"],
                    moreh_bias=pd._tt_weights[f"fpn_l{i}_block_gn_b_moreh"],
                    manual_weight=pd._tt_weights[f"fpn_l{i}_block_gn_w_manual"],
                    manual_bias=pd._tt_weights[f"fpn_l{i}_block_gn_b_manual"],
                )
                if x.get_layout() != ttnn.TILE_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                x = ttnn.relu(x)

                if stop_after == f"pixel_decoder_l{i}":
                    return x

            if stop_after == "pixel_decoder_l2":
                return x

            if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            mask_features = pd._conv2d(
                x,
                weight=pd._tt_weights["mask_proj_w"],
                bias=pd._tt_weights.get("mask_proj_b"),
                weight_key="mask_proj_w",
                bias_key="mask_proj_b",
                out_channels=pd.config.mask_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            return mask_features

        if trace_subgraph.startswith("pixel_decoder_"):
            out = _pixel_decoder_partial(features, stop_after=trace_subgraph)
            return out, out

        mask_features, _ = pixel_decoder.forward(features)
        if trace_subgraph == "pixel_decoder":
            return mask_features, mask_features

        decoder_last, _, _ = transformer_decoder.forward_tt(features[-1], return_tt_tensor=True)
        if trace_subgraph == "decoder":
            return decoder_last, decoder_last

        return heads.forward_tt(decoder_last, mask_features)

    # Host input (NHWC) to match backbone TT path.
    torch.manual_seed(0)
    pixel_values = torch.randn(1, 3, 320, 320, dtype=torch.float32)
    nhwc = pixel_values.permute(0, 2, 3, 1).contiguous()
    host_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    CQ_OPS = 0
    CQ_INPUT_WRITE = 1

    dram_input = ttnn.allocate_tensor_on_device(
        host_input.shape, host_input.dtype, host_input.layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    l1_input = ttnn.allocate_tensor_on_device(
        host_input.shape, host_input.dtype, host_input.layout, device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    def _deallocate_structured(struct):
        if struct is None:
            return
        if isinstance(struct, (list, tuple)):
            for t in struct:
                _deallocate_structured(t)
            return
        if getattr(struct, "is_allocated", None) is not None:
            try:
                if struct.is_allocated():
                    ttnn.deallocate(struct, force=True)
            except Exception:
                pass
        else:
            try:
                ttnn.deallocate(struct, force=True)
            except Exception:
                pass

    trace_id = None
    trace_capture_started = False
    try:
        # Warmup (compile + weight/param prep). We run two forwards so conv2d sites that
        # lazily "prepare" weights on first use can hit the steady-state (prepared weights)
        # path before trace capture, avoiding binary uploads during capture.
        op_event = ttnn.record_event(device, cq_id=CQ_OPS)
        # Important: some TTNN ops use different kernels on the "prepared weights" path vs
        # the initial "prepare weights" path. A single warmup forward is enough to prepare
        # weights + caches, and we separately exercise any required prepared-kernel paths
        # before trace capture.
        warmup_iters = int(os.environ.get("MASKFORMER_E2E_TRACE_WARMUP_ITERS", "1"))
        print(f"[maskformer][e2e][trace+2cq] warmup begin iters={warmup_iters}", flush=True)
        for _ in range(max(warmup_iters, 1)):
            print("[maskformer][e2e][trace+2cq] warmup iter begin", flush=True)
            ttnn.wait_for_event(CQ_INPUT_WRITE, op_event)
            ttnn.copy_host_to_device_tensor(host_input, dram_input, cq_id=CQ_INPUT_WRITE)
            write_event = ttnn.record_event(device, cq_id=CQ_INPUT_WRITE)

            ttnn.wait_for_event(CQ_OPS, write_event)
            with ttnn.command_queue(CQ_OPS):
                l1_input = ttnn.to_memory_config(
                    dram_input, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=l1_input
                )
                warm_outputs = forward_tt(l1_input)
            op_event = ttnn.record_event(device, cq_id=CQ_OPS)
            ttnn.synchronize_device(device)
            _deallocate_structured(warm_outputs)
            print("[maskformer][e2e][trace+2cq] warmup iter done", flush=True)
        print("[maskformer][e2e][trace+2cq] warmup done", flush=True)

        # The decoder clones cached positional/query tensors on cache hits to avoid hangs on some builds.
        # The first cache-hit path occurs during trace capture, which can trigger a first-use binary upload
        # in `ttnn.clone(...)` (disallowed during trace capture). Pre-run the clones once here.
        print("[maskformer][e2e][trace+2cq] decoder cache-clone warmup begin", flush=True)
        cache_clones = []
        try:
            with ttnn.command_queue(CQ_OPS):
                pos_cache = getattr(transformer_decoder, "_tt_pos_cache", None)
                if isinstance(pos_cache, dict) and len(pos_cache) > 0:
                    tt_mem_pos = next(iter(pos_cache.values()))
                    mem_cfg = (
                        tt_mem_pos.memory_config()
                        if callable(getattr(tt_mem_pos, "memory_config", None))
                        else ttnn.DRAM_MEMORY_CONFIG
                    )
                    cache_clones.append(ttnn.clone(tt_mem_pos, memory_config=mem_cfg))

                query_cache = getattr(transformer_decoder, "_tt_query_cache", None)
                if isinstance(query_cache, dict) and len(query_cache) > 0:
                    tt_hidden, tt_qpos = next(iter(query_cache.values()))
                    hidden_mem_cfg = (
                        tt_hidden.memory_config()
                        if callable(getattr(tt_hidden, "memory_config", None))
                        else ttnn.DRAM_MEMORY_CONFIG
                    )
                    qpos_mem_cfg = (
                        tt_qpos.memory_config()
                        if callable(getattr(tt_qpos, "memory_config", None))
                        else ttnn.DRAM_MEMORY_CONFIG
                    )
                    cache_clones.append(ttnn.clone(tt_hidden, memory_config=hidden_mem_cfg))
                    cache_clones.append(ttnn.clone(tt_qpos, memory_config=qpos_mem_cfg))

            ttnn.synchronize_device(device)
        finally:
            for t in cache_clones:
                _deallocate_structured(t)
        print("[maskformer][e2e][trace+2cq] decoder cache-clone warmup done", flush=True)

        # The decoder's input-projection conv2d prepares weights on the first run and then
        # uses a different conv2d execution path once weights are prepared. If we only run a
        # single warmup forward, the first time that prepared path is executed would be
        # during trace capture, which can trigger a binary upload ("Writes are not supported
        # during trace capture") on some builds. Exercise that prepared path once here, but
        # avoid re-running the full pixel decoder (can hang on repeated forwards on some
        # program-cache configurations).
        print("[maskformer][e2e][trace+2cq] decoder input_proj prepared-path warmup begin", flush=True)
        with ttnn.command_queue(CQ_OPS):
            features, _ = backbone.forward(l1_input)
            feat = features[-1]
            if getattr(feat, "get_layout", None) is not None and feat.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                feat = ttnn.to_layout(feat, ttnn.ROW_MAJOR_LAYOUT)

            if not getattr(transformer_decoder, "_input_proj_prepared", False):
                raise RuntimeError(
                    "Expected decoder input projection weights to be prepared after warmup, but they were not."
                )

            B = int(feat.shape[0])
            H = int(feat.shape[1])
            W = int(feat.shape[2])
            [tt_proj, [_out_h, _out_w]] = ttnn.conv2d(
                input_tensor=feat,
                weight_tensor=transformer_decoder._input_proj_w,
                bias_tensor=transformer_decoder._input_proj_b,
                in_channels=int(feat.shape[-1]),
                out_channels=int(transformer_decoder.config.hidden_dim),
                batch_size=B,
                input_height=H,
                input_width=W,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                return_output_dim=True,
                return_weights_and_bias=False,
            )
            mem_seq = ttnn.reshape(tt_proj, (B, H * W, int(transformer_decoder.config.hidden_dim)))
            mem_seq = ttnn.to_layout(mem_seq, ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(device)
        _deallocate_structured(mem_seq)
        _deallocate_structured(tt_proj)
        _deallocate_structured(features)
        print("[maskformer][e2e][trace+2cq] decoder input_proj prepared-path warmup done", flush=True)

        # Prep input for capture.
        ttnn.wait_for_event(CQ_INPUT_WRITE, op_event)
        ttnn.copy_host_to_device_tensor(host_input, dram_input, cq_id=CQ_INPUT_WRITE)
        write_event = ttnn.record_event(device, cq_id=CQ_INPUT_WRITE)
        ttnn.wait_for_event(CQ_OPS, write_event)
        with ttnn.command_queue(CQ_OPS):
            l1_input = ttnn.to_memory_config(dram_input, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=l1_input)
        op_event = ttnn.record_event(device, cq_id=CQ_OPS)

        # Trace capture (CQ0): compute only.
        print("[maskformer][e2e][trace+2cq] trace capture begin", flush=True)
        trace_id = ttnn.begin_trace_capture(device, cq_id=CQ_OPS)
        trace_capture_started = True
        in_trace_capture = True
        try:
            with ttnn.command_queue(CQ_OPS):
                out_class, out_masks = trace_forward(l1_input)
        finally:
            in_trace_capture = False
        ttnn.end_trace_capture(device, trace_id, cq_id=CQ_OPS)
        trace_capture_started = False
        ttnn.synchronize_device(device)
        print("[maskformer][e2e][trace+2cq] trace capture done", flush=True)

        if os.environ.get("MASKFORMER_E2E_TRACE_PRINT_CAPTURE_OUTPUT", "0").strip() == "1":

            def _tensor_brief(t):
                parts = []
                try:
                    parts.append(f"shape={tuple(int(d) for d in t.shape)}")
                except Exception:
                    pass
                try:
                    if getattr(t, "get_layout", None) is not None:
                        parts.append(f"layout={t.get_layout()}")
                except Exception:
                    pass
                try:
                    if hasattr(t, "storage_type") and hasattr(ttnn, "StorageType"):
                        parts.append(f"storage={t.storage_type()}")
                except Exception:
                    pass
                try:
                    if hasattr(t, "buffer_address"):
                        parts.append(f"addr=0x{int(t.buffer_address()):x}")
                except Exception:
                    pass
                return " ".join(parts) if parts else "<unknown>"

            try:
                print(f"[maskformer][e2e][trace+2cq] capture out0 {_tensor_brief(out_class)}", flush=True)
                print(f"[maskformer][e2e][trace+2cq] capture out1 {_tensor_brief(out_masks)}", flush=True)
            except Exception:
                pass

        # Timed trace+2CQ loop (overlapped I/O + compute).
        iterations = int(os.environ.get("MASKFORMER_E2E_TRACE_REPLAY_ITERS", "8"))
        loop_debug = os.environ.get("MASKFORMER_E2E_TRACE_LOOP_DEBUG", "0").strip() == "1"
        skip_io = os.environ.get("MASKFORMER_E2E_TRACE_REPLAY_SKIP_IO", "0").strip() == "1"
        print(f"[maskformer][e2e][trace+2cq] trace replay begin iters={iterations}", flush=True)
        start = time.perf_counter()
        for i in range(iterations):
            if loop_debug:
                print(f"[maskformer][e2e][trace+2cq] trace replay iter {i} enqueue", flush=True)
            if skip_io:
                ttnn.execute_trace(device, trace_id, cq_id=CQ_OPS, blocking=False)
                continue
            # CQ1: host -> DRAM (must wait until CQ0 is done reading DRAM input)
            ttnn.wait_for_event(CQ_INPUT_WRITE, op_event)
            ttnn.copy_host_to_device_tensor(host_input, dram_input, cq_id=CQ_INPUT_WRITE)
            write_event = ttnn.record_event(device, cq_id=CQ_INPUT_WRITE)

            # CQ0: DRAM -> L1 then replay trace
            ttnn.wait_for_event(CQ_OPS, write_event)
            with ttnn.command_queue(CQ_OPS):
                l1_input = ttnn.to_memory_config(
                    dram_input, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=l1_input
                )
            op_event = ttnn.record_event(device, cq_id=CQ_OPS)
            ttnn.execute_trace(device, trace_id, cq_id=CQ_OPS, blocking=False)

        if loop_debug:
            print("[maskformer][e2e][trace+2cq] trace replay sync begin", flush=True)
        ttnn.synchronize_device(device)
        if loop_debug:
            print("[maskformer][e2e][trace+2cq] trace replay sync done", flush=True)
        end = time.perf_counter()

        avg_ms = (end - start) * 1000.0 / iterations
        print(f"[maskformer][e2e][trace+2cq] avg={avg_ms:.2f} ms over {iterations} iterations", flush=True)

        # Basic sanity: outputs should stay device-resident.
        assert hasattr(out_class, "storage_type"), "Expected TTNN output tensor from trace capture"
        assert hasattr(out_masks, "storage_type"), "Expected TTNN output tensor from trace capture"
    finally:
        if trace_capture_started and trace_id is not None:
            # Defensive: if capture errored, ensure trace capture ends so device teardown can proceed.
            try:
                ttnn.end_trace_capture(device, trace_id, cq_id=CQ_OPS)
            except Exception:
                pass
        if trace_guard:
            # Restore patched APIs for subsequent tests.
            if _orig_from_torch is not None:
                ttnn.from_torch = _orig_from_torch
            if _orig_to_device is not None:
                ttnn.to_device = _orig_to_device
            if _orig_to_torch is not None:
                ttnn.to_torch = _orig_to_torch
            if _orig_copy_h2d is not None:
                ttnn.copy_host_to_device_tensor = _orig_copy_h2d
            if _orig_copy_d2h is not None:
                ttnn.copy_device_to_host_tensor = _orig_copy_d2h
        if disable_deallocate_during_capture and _orig_deallocate is not None:
            ttnn.deallocate = _orig_deallocate
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
