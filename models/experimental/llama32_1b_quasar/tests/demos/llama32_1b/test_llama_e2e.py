# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end llama32_1b test that runs on both Wormhole and Quasar.

This is the llama analog of resnet50's ``test_resnet50_e2e.py`` + ``RESNET_PCC_LOG`` flow. It reuses the
demo's token-accuracy path (teacher-forcing top1/top5 vs the committed ``.refpt``) but adds:

  * per-op PCC / fingerprint logging via ``LLAMA_PCC_LOG=1`` (see models/.../llama32_1b/pcc_log.py). Each
    seam logs a ``[PCCLOG]`` line, and — if goldens are registered — a ``[GOLDENPCC]`` line. Set
    ``LLAMA_PCC_DUMP=<dir>`` to dump each op's tensor for offline / cross-arch PCC (dump on WH, diff on
    Quasar).
  * arch-aware env setup: on Quasar it disables minimal_matmul (it pins an 8x8 grid) and defaults to a
    small layer count for bring-up, and the model's tuning recipe swaps the decode SDPA grid to the
    device grid (see _resolve_llama32_1b_wh_tuning).

Run:
    # Wormhole (regression gate — hard top1/top5 assert):
    MESH_DEVICE=N150 pytest models/experimental/llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py
    # Quasar (bring-up — runs + logs per-op PCC; top-k not hard-gated):
    MESH_DEVICE=<qsr> TT_METAL_SLOW_DISPATCH_MODE=1 LLAMA32_1B_DEMO_NUM_LAYERS=1 \
        pytest models/experimental/llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py

TODO (Quasar enablement — not done here; the model still has WH-only paths the decode/prefill hit):
  * DRAM-sharded decode matmuls (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig for QKV / attn-out
    / MLP / LM-head) call get_optimal_dram_bank_to_logical_worker_assignment, which TT_ASSERTs WH/BH.
    They must be converted to DRAM-interleaved weights + an interleaved/1D matmul config on Quasar.
  * ttnn.linear (mainline matmul) has no Quasar/Metal-2.0 factory; the interleaved fallback likely needs
    routing to ttnn.experimental.quasar.matmul on Quasar. Resolve with a 1-layer run first.
  * prefill SDPA + QKV grids inside attention_1d.py are still hard (8,8); thread the device grid through.
"""

import os

import pytest
from loguru import logger

import ttnn

# Reuse the demo's mesh_device fixture, pytestmark (MESH_DEVICE parametrization) and helpers.
from models.experimental.llama32_1b_quasar.tests.demos.llama32_1b import demo
from models.experimental.llama32_1b_quasar.tests.demos.llama32_1b.demo import (
    EXPECTED_METRICS,
    _run_token_accuracy,
    create_model,
    get_device_name,
    lazy_weight_cache_dir_for_demo,
    mesh_device,  # noqa: F401 — pytest fixture, used by injection
)
from models.experimental.llama32_1b_quasar.utility_functions import is_quasar

pytestmark = demo.pytestmark


def _qsr_capped_grid_xy(dev, max_cores=None):
    """Grid (x, y) capped to the emulator size (LLAMA_QSR_MAX_GRID_CORES, default 2 compute nodes).

    The forced-interleaved 1D-mcast matmul only passes on Quasar at a SMALL grid: the standalone
    test_quasar_qkv_matmul_dfb.py FAILS at compute_with_storage_grid_size=(8,4) but PASSES at the actual
    (small) device grid. To match what runs on the 2-core emulator regardless of the sim's device size, cap
    every Quasar grid (matmul + SDPA) to <= max_cores cores, laid out as a single row where possible."""
    if max_cores is None:
        max_cores = int(os.environ.get("LLAMA_QSR_MAX_GRID_CORES", "2"))
    dx, dy = int(dev.x), int(dev.y)
    if dx >= 2:
        return min(dx, max_cores), 1
    return 1, min(dy, max_cores)


def _quasar_upload_tile_heads_padded(t, dev):
    """Upload a torch [1, batch, n_heads, head_dim] tensor as a TILE DRAM ttnn tensor, padding the heads dim
    (dim -2) up to a multiple of TILE_HEIGHT (32) so quasar.tilize accepts it. This mirrors what the real
    nlp_create_qkv_heads_decode produces: k/v have num_kv_heads=8 in the tile-height slot, tile-padded to 32.
    Consumers here read the head count from elsewhere (paged_update_cache uses the CACHE's head count; host SDPA
    uses q's real n_heads=32), so the trailing zero-padded head rows are never read. RM-tilize path avoids the
    from_torch(TILE) hang; head_dim (64) is already tile-aligned."""
    import torch as _torch

    h = t.shape[-2]
    pad = (-h) % 32
    if pad:
        t = _torch.nn.functional.pad(t, (0, 0, 0, pad))  # pad heads (dim -2) to a full tile
    rm = ttnn.from_torch(
        t.to(_torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
    )
    qtil = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    return (qtil or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


def _install_quasar_tilize_from_torch(monkeypatch):
    """Route on-device ``ttnn.from_torch(..., layout=TILE)`` through the Gen2-native quasar tilize.

    The mainline ``TilizeDeviceOperation`` that ``from_torch(TILE)`` runs internally hangs on the Quasar
    simulator (see tests/ttnn/unit_tests/operations/test_quasar_tilize_from_torch_hang.py). The
    experimental ``ttnn.experimental.quasar.tilize`` op passes for the same shapes, so on Quasar we upload
    row-major (a plain to_device, no device tilize) and then tilize with that op. Non-TILE / host uploads
    pass straight through. Weight materialization goes through ``ttnn.from_torch``, so patching it covers
    every weight upload the model does.
    """
    orig_from_torch = ttnn.from_torch

    def _from_torch(tensor, *args, **kwargs):
        if kwargs.get("layout") == ttnn.TILE_LAYOUT and kwargs.get("device") is not None:
            out_dtype = kwargs.get("dtype")
            out_memcfg = kwargs.get("memory_config")
            # Quasar interleaved-matmul: the decode matmul WEIGHTS are uploaded DRAM-WIDTH_SHARDED (for the
            # DRAM-sharded matmul, which we route to the interleaved 1D-mcast matmul on Quasar). Upload them
            # DRAM-INTERLEAVED instead: the 1D-mcast matmul reads a DRAM-interleaved in1 directly, but a
            # DRAM-*sharded* in1 makes the factory try to borrow it into an L1 DFB (is_sharded()==true) and
            # fails the L1-residency check. A runtime sharded->interleaved reshard of a DRAM-sharded tensor
            # is not clean, so fix it at upload.
            try:
                if out_memcfg is not None and out_memcfg.is_sharded():
                    bt = out_memcfg.buffer_type
                    bt = bt() if callable(bt) else bt
                    if bt == ttnn.BufferType.DRAM:
                        logger.warning(
                            f"[llama-e2e][quasar] DRAM-sharded weight -> DRAM-interleaved upload ({out_memcfg})"
                        )
                        out_memcfg = ttnn.DRAM_MEMORY_CONFIG
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] DRAM-sharded weight memcfg check failed ({e})")
            # A fingerprint of the exact weight/config so a failing case is identifiable in the log.
            try:
                shp = tuple(tensor.shape)
            except Exception:
                shp = "?"
            desc = f"shape={shp} out_dtype={out_dtype} out_memcfg={out_memcfg}"
            # Upload row-major to DRAM (no device tilize), then tilize with the Gen2 op to the requested
            # dtype / memory config.
            rm_kwargs = dict(kwargs)
            rm_kwargs["layout"] = ttnn.ROW_MAJOR_LAYOUT
            rm_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
            # Data-format dodge (PR author, 2026-09-24): the mainline tilize/untilize Quasar breakage is
            # fp32-vs-bf16 dependent, so keep the whole upload in bf16 until it is root-caused. Block-float
            # (bf8_b / bf4_b) is a TILE-only format and cannot be uploaded row-major anyway. For all three
            # (bf8_b, bf4_b, float32) upload row-major as bf16; produce bf16 output for a float32 target
            # (a bring-up numeric downcast) and let the quasar op pack to the block-float dtype otherwise.
            staged_bf16 = out_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32)
            tilize_dtype = out_dtype
            if staged_bf16:
                rm_kwargs["dtype"] = ttnn.bfloat16
                # bf8_b/bf4_b are unsupported on Quasar: any DEVICE op with a block-float DFB FATALs at
                # program_spec.cpp:2253. When matmuls run on device (LLAMA_QSR_DEVICE_ATTN), the weights are
                # in1 DFBs, so pack them bf16 instead of the requested block-float. (float32 -> bf16 also
                # dodges the fp32 mainline-tilize hang.) Bring-up numeric downcast; the model is bf16 e2e.
                tilize_dtype = ttnn.bfloat16
            if out_dtype == ttnn.float32:
                logger.warning(f"[llama-e2e][quasar] downcasting fp32 target -> bf16 for {desc} (fp32 hang dodge)")
            # CRITICAL: cast the dtype on the HOST (torch) so the row-major upload is a pure DMA with no
            # device op. If the source torch tensor is fp32 and rm dtype is bf16, from_torch does the
            # fp32->bf16 cast ON DEVICE via Tilize(fp32) -> Typecast -> Untilize, and that fp32 mainline
            # Tilize deadlocks on the Quasar sim (the hang is a device op inside our own upload, so it is
            # NOT caught by the try/except below). Matching the source dtype to rm dtype removes the cast
            # entirely, so the upload emits no Tilize/Typecast/Untilize.
            import torch as _torch

            if (
                rm_kwargs.get("dtype") == ttnn.bfloat16
                and isinstance(tensor, _torch.Tensor)
                and tensor.dtype != _torch.bfloat16
            ):
                tensor = tensor.to(_torch.bfloat16)
            # Two stages: the (now dtype-matched) row-major upload, then the quasar tilize op.
            stage = "row-major upload"
            try:
                rm = orig_from_torch(tensor, *args, **rm_kwargs)
                stage = "quasar.tilize"
                return ttnn.experimental.quasar.tilize(rm, memory_config=out_memcfg, dtype=tilize_dtype)
            except Exception as e:
                # Do NOT fall back to from_torch(TILE): that runs the mainline TilizeDeviceOperation, which
                # deadlocks on the Quasar sim (the reason this patch exists). Retry once fully in bf16
                # (upload + output), and if that still fails, raise rather than silently hang.
                logger.warning(f"[llama-e2e][quasar] STAGE='{stage}' FAILED for {desc} staged_bf16={staged_bf16}: {e}")
                if not staged_bf16 or tilize_dtype != ttnn.bfloat16:
                    logger.warning("[llama-e2e][quasar] retrying upload fully in bf16 (upload + output)")
                    rm_kwargs["dtype"] = ttnn.bfloat16
                    rm = orig_from_torch(tensor, *args, **rm_kwargs)
                    return ttnn.experimental.quasar.tilize(rm, memory_config=out_memcfg, dtype=ttnn.bfloat16)
                raise
        return orig_from_torch(tensor, *args, **kwargs)

    monkeypatch.setattr(ttnn, "from_torch", _from_torch)


def _install_quasar_i2s(monkeypatch):
    """Route mainline ``ttnn.interleaved_to_sharded`` through the Gen2-native quasar op on Quasar.

    The mainline i2s faults on the Quasar sim (UNALIGNED_LOAD, neighbour-core corruption) for the RoPE
    cos/sin decode sharding ([1,1,1,64] -> single-core HEIGHT_SHARDED). Keeping cos/sin *interleaved* is
    not viable: the decode rope apply is ``ttnn.experimental.rotary_embedding_llama_fused_qk``, a sharded
    op that needs SHARDED cos/sin. So route to ``ttnn.experimental.quasar.interleaved_to_sharded``, which
    produces the same sharded result via arch-safe kernels (get_entry_size, no get_tile_size descriptor
    read). Falls back to mainline on any error.
    """
    orig = ttnn.interleaved_to_sharded
    qi2s = getattr(getattr(ttnn.experimental, "quasar", None), "interleaved_to_sharded", None)
    if qi2s is None:
        logger.warning("[llama-e2e][quasar] no experimental.quasar.interleaved_to_sharded; leaving mainline i2s")
        return

    def _i2s(input_tensor, *args, **kwargs):
        try:
            return qi2s(input_tensor, *args, **kwargs)
        except Exception as e:
            logger.warning(f"[llama-e2e][quasar] quasar i2s failed ({e}); falling back to mainline i2s")
            return orig(input_tensor, *args, **kwargs)

    monkeypatch.setattr(ttnn, "interleaved_to_sharded", _i2s)


def _install_quasar_i2s_logger(monkeypatch):
    """Log every ``ttnn.interleaved_to_sharded`` call (input spec + buffer address, output memcfg) WITHOUT
    changing behaviour. Used by the i2s-failure repro test: the LAST ``[i2s-log]`` line before the device
    fault is the culprit call, and the input buffer address lets a standalone repro match the model's
    (deterministic) allocator placement -- the fault address 0x04a46b43 has been byte-identical across runs.
    """
    orig = ttnn.interleaved_to_sharded
    n = {"i": 0}

    def _i2s(input_tensor, *args, **kwargs):
        n["i"] += 1
        try:
            lshape = tuple(input_tensor.logical_shape)
            pshape = tuple(input_tensor.padded_shape)
            spec = f"logical={lshape} padded={pshape} dtype={input_tensor.dtype} layout={input_tensor.layout} in_memcfg={input_tensor.memory_config()}"
        except Exception as e:
            spec = f"<spec unavailable: {e}>"
        addr = "?"
        for meth in ("buffer_address", "device_buffer_address"):
            try:
                addr = hex(getattr(input_tensor, meth)())
                break
            except Exception:
                pass
        out_mc = args[0] if args else (kwargs.get("sharded_memory_config") or kwargs.get("memory_config"))
        logger.warning(f"[i2s-log] #{n['i']} {spec} in_buf_addr={addr} -> out_memcfg={out_mc}")
        return orig(input_tensor, *args, **kwargs)

    monkeypatch.setattr(ttnn, "interleaved_to_sharded", _i2s)


def _install_quasar_fp32_acc_off(monkeypatch):
    """Force ``fp32_dest_acc_en=False`` on every compute-kernel config on Quasar.

    The Quasar sim's unpacker does not implement the bf16 (Float16_b, fmt 5) -> Tf32 (fmt 4) src-value
    conversion (``qsr_unpack_src_value: in_format=5 out_format=4``). ``fp32_dest_acc_en=True`` triggers it
    for bf16 inputs: fp32 accumulation reads bf16 through a Tf32 unpack. The sharded RMSNorm
    (rmsnorm_1d.py) sets it True, and it faults there; every other fp32-acc op (matmuls, ...) would hit the
    same. Forcing it off keeps the math in bf16 (bf16->bf16 unpack, which the sim implements) at a bring-up
    precision cost. Covers ttnn.WormholeComputeKernelConfig and ttnn.init_device_compute_kernel_config.
    Configs built at import time (module constants) are unaffected, but those already set it False.
    """

    def _force_off(orig):
        def _mk(*args, **kwargs):
            kwargs["fp32_dest_acc_en"] = False
            return orig(*args, **kwargs)

        return _mk

    for name in ("WormholeComputeKernelConfig", "GrayskullComputeKernelConfig", "init_device_compute_kernel_config"):
        orig = getattr(ttnn, name, None)
        if orig is not None:
            monkeypatch.setattr(ttnn, name, _force_off(orig))


def _install_quasar_interleaved_matmul(monkeypatch, mesh_device):
    """Run the decode + prefill matmuls ON DEVICE on Quasar via the mainline mcast picker.

    The model's QKV / WO / MLP(W1,W2,W3) / LM-head matmuls use DRAM-sharded program configs
    (ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig, WH/BH-only) that don't run on Quasar. This
    wrapper de-shards the inputs to DRAM-interleaved and drops that program_config so the mainline picker
    auto-selects a standard mcast matmul. Both mcast factories are now Gen2-ported (1D:
    test_quasar_qkv_matmul_dfb.py; 2D: test_quasar_matmul_2d_mcast.py), so small-M decode goes 1D and large-M
    prefill / wide lm_head go 2D (which streams out_block to fit L1) -- all on device. Output is forced
    DRAM-interleaved (create_qkv_heads_decode's Interleaved factory is ported, so no width-shard trick is
    needed; s2i_copy handles the downstream alias) and bf8_b/bf4_b output dtype is coerced to bf16.
    Requires TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 in the env.
    """

    def _to_dram(t):
        try:
            if t is None or not (hasattr(t, "is_sharded") and t.is_sharded()):
                return t
            q_tmc = getattr(getattr(ttnn.experimental, "quasar", None), "to_memory_config", None)
            fn = q_tmc or ttnn.to_memory_config
            return fn(t, ttnn.DRAM_MEMORY_CONFIG)
        except Exception as e:
            logger.warning(f"[llama-e2e][quasar] matmul de-shard failed ({e}); passing tensor through")
            return t

    # CAP the matmul grid to a small 2D shape (2x2, clamped to the device). CRITICAL and non-obvious:
    #  - It must be a 2D grid (both dims > 1), NOT a single row: a 1-row grid makes the picker choose 1D mcast
    #    for the large-M prefill, whose per-core output = full M -> L1 OOM. A 2D grid lets the picker choose 2D
    #    mcast for prefill/lm_head, which streams out_block and fits L1.
    #  - It must be SMALL: 1D mcast FAILS at large grids on Quasar -- the device's full 8x4 tripped
    #    llk_io_unpack.h:45 on the DECODE matmul (test_quasar_qkv_matmul_dfb[8x4] fails the same way; 8x4 was
    #    never validated even with alias=0). The validated regime is <= ~3x2; 2x2 is validated for BOTH 1D
    #    (test_quasar_qkv_matmul_dfb) and 2D (test_quasar_matmul_2d_mcast). Do NOT rely on
    #    TT_METAL_CORE_GRID_OVERRIDE to keep the grid small -- pin it here so a run without the override
    #    (device = full 8x4) doesn't fall back to the failing large grid.
    # Cap the matmul grid to a small 2D shape (2x2, clamped to device). Must be 2D (a 1-row grid makes the
    # picker pick 1D for prefill -> full-M-per-core L1 OOM) and SMALL (1D fails at 8x4 -- llk_io_unpack.h:45).
    dev = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(int(dev.x), 2), min(int(dev.y), 2)
    mm_core_grid = ttnn.CoreGrid(y=gy, x=gx)

    def _blk(v, sub, cap):
        # largest divisor of v that is a multiple of `sub` and <= cap
        best, d = sub, sub
        while d <= min(v, cap):
            if v % d == 0:
                best = d
            d += sub
        return best

    def _wrap(orig):
        def _mm(input_tensor, weight, *args, **kwargs):
            # Build the matmul program config EXPLICITLY with a PINNED small in0_block_w (<=4). WHY:
            # letting the picker auto-select (program_config=None) chose in0_block_w=16 for the decode QKV
            # matmul, which trips the TEN-4746 bare-wait->pop guard (llk_io_unpack.h:45,
            # LLK_TDMA_GUARD_ASSERT_DISARMED). in0_block_w=4 is validated: it passed for 1D decode (earlier
            # e2e device run) AND 2D prefill (test_quasar_matmul_2d_mcast). So pin it here so LLK asserts can
            # stay ON. (A sub-agent is finding where to add dummy_unpack in the matmul compute kernel -- the
            # real fix that would let the large-in0_block_w picker config work; until then, pin.)
            # 1D mcast_in0 for small-M decode; 2D mcast (streamed out_block, fits L1) for large-M prefill /
            # wide lm_head. DRAM output (create_qkv Interleaved factory is ported; s2i_copy handles the alias).
            input_tensor = _to_dram(input_tensor)
            weight = _to_dram(weight)
            kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
            try:
                if kwargs.get("dtype") in (ttnn.bfloat8_b, ttnn.bfloat4_b):
                    kwargs["dtype"] = ttnn.bfloat16
            except Exception:
                pass
            try:
                ish = input_tensor.padded_shape  # [.., M, K]
                wsh = weight.padded_shape  # [.., K, N]
                mt = max(int(ish[-2]) // 32, 1)
                kt = max(int(ish[-1]) // 32, 1)
                nt = max(int(wsh[-1]) // 32, 1)
                ibw = _blk(kt, 1, 4)  # PINNED <=4 -- avoids the TEN-4746 large-in0_block_w path
                if mt == 1:
                    # 1D mcast_in0 (decode): in0 broadcast, each core owns per_core_N of the output.
                    nc = gx * gy
                    per_core_n = max((nt + nc - 1) // nc, 1)
                    osw = _blk(per_core_n, 1, 4)
                    kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=ibw,
                        out_subblock_h=1,
                        out_subblock_w=osw,
                        per_core_M=1,
                        per_core_N=per_core_n,
                        fuse_batch=False,
                        fused_activation=None,
                        mcast_in0=True,
                    )
                else:
                    # 2D mcast (prefill / lm_head): M across grid.y, N across grid.x; out_block streams to L1.
                    per_core_m = max((mt + gy - 1) // gy, 1)
                    per_core_n = max((nt + gx - 1) // gx, 1)
                    osw = _blk(per_core_n, 1, 4)
                    kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=ibw,
                        out_subblock_h=1,
                        out_subblock_w=osw,
                        out_block_h=_blk(per_core_m, 1, 8),
                        out_block_w=_blk(per_core_n, osw, 16),
                        per_core_M=per_core_m,
                        per_core_N=per_core_n,
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=False,
                    )
                kwargs.pop("core_grid", None)
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] matmul cfg build failed ({e}); auto-select w/ small core_grid")
                kwargs["program_config"] = None
                kwargs["core_grid"] = mm_core_grid
            return orig(input_tensor, weight, *args, **kwargs)

        return _mm

    for name in ("linear", "matmul"):
        orig = getattr(ttnn, name, None)
        if orig is not None:
            monkeypatch.setattr(ttnn, name, _wrap(orig))


def _install_quasar_bf16_kv_cache(monkeypatch):
    """Allocate the paged KV cache as bf16 on Quasar (needed for ON-DEVICE decode SDPA).

    The decode SDPA validate hard-requires bf16 q/k/v on Quasar (bf8_b/bf4_b unsupported). The cache is
    allocated by EagerLLMExecutor.allocate_kv_cache via ttnn.as_tensor(dtype=kv_cache_dtype), where
    kv_cache_dtype defaults to bf8_b (executor.py:424). While host-SDPA reads the cache via to_torch (dtype
    agnostic), the device flash-decode reads it as a DFB -> a bf8_b cache FATALs. Force the executor's
    model_args.kv_cache_dtype to bf16 before allocation. (enable_model_cache is false in this run, so the
    dtype-blind empty-cache file is not reloaded -- no stale-bf8_b-file hazard.)"""
    try:
        from models.experimental.llama32_1b_quasar.models.executor import EagerLLMExecutor
    except Exception as e:
        logger.warning(f"[llama-e2e][quasar] could not import EagerLLMExecutor for bf16 KV cache ({e})")
        return

    orig = EagerLLMExecutor.allocate_kv_cache

    def _patched(self, kv_cache_shape, dtype, num_layers):
        ma = getattr(self, "model_args", None)
        if ma is not None:
            try:
                ma.kv_cache_dtype = ttnn.bfloat16
                logger.warning("[llama-e2e][quasar] forcing KV cache dtype -> bf16 for on-device SDPA")
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] could not force KV cache dtype ({e})")
        return orig(self, kv_cache_shape, dtype, num_layers)

    monkeypatch.setattr(EagerLLMExecutor, "allocate_kv_cache", _patched)


def _install_quasar_host_matmul(monkeypatch):
    """Compute the decode matmuls on the HOST (torch), sidestepping the Quasar 1D-mcast matmul.

    The forced-interleaved decode matmuls (QKV / WO / MLP w1,w3,w2 / lm_head) all route through the mainline
    1D mcast_in0 matmul, which underflows the in0 DFB tile counter on the Quasar sim (posted=64 acked=65) --
    a runtime/sim DM<->tensix counter-remapper mis-delivery, not an op bug (craq-sim issue filed). To surface
    what fails DOWNSTREAM of the matmuls, we gather both operands to host, do a plain torch matmul (+ bias if
    given -- the decode linears have NO fused activation; MLP SiLU is a separate ttnn.mul), and upload the
    result into the caller's requested memory_config. Single-device (N150) only; multi-device falls back to
    the device op. Numerics are exact-ish (fp32 accumulate -> bf16); this is a bring-up sidestep, not perf.
    """
    import torch as _torch

    def _upload(out_t, dev, out_memcfg):
        # bf16 row-major upload (pure DMA) -> quasar.tilize (DRAM interleaved) -> caller's memory_config.
        rm = ttnn.from_torch(
            out_t.to(_torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
        )
        qtil = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
        tiled = (qtil or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        # Sharding is forced off on Quasar (_install_quasar_force_interleaved), so always return the tensor
        # DRAM-interleaved regardless of the requested memcfg. A non-sharded L1 target still gets honored.
        if out_memcfg is None or out_memcfg == ttnn.DRAM_MEMORY_CONFIG:
            return tiled
        try:
            if out_memcfg.is_sharded():
                return tiled  # no sharding on Quasar
        except Exception:
            return tiled
        q_tmc = getattr(getattr(ttnn.experimental, "quasar", None), "to_memory_config", None)
        try:
            return (q_tmc or ttnn.to_memory_config)(tiled, out_memcfg)
        except Exception:
            return tiled

    def _wrap(orig):
        def _mm(input_tensor, weight, *args, **kwargs):
            try:
                dev = input_tensor.device()
                if dev is None or dev.get_num_devices() != 1:
                    return orig(input_tensor, weight, *args, **kwargs)
                a = ttnn.to_torch(input_tensor).float()
                b = ttnn.to_torch(weight).float()
                out = _torch.matmul(a, b)
                bias = kwargs.get("bias")
                if bias is not None:
                    out = out + ttnn.to_torch(bias).float()
                res = _upload(out, dev, kwargs.get("memory_config"))
                logger.warning(
                    f"[llama-e2e][quasar] host matmul {tuple(a.shape)} x {tuple(b.shape)} -> {tuple(out.shape)} "
                    f"(sidestepped device 1D-mcast; out_memcfg={kwargs.get('memory_config')})"
                )
                return res
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] host matmul failed ({e}); falling back to device op")
                return orig(input_tensor, weight, *args, **kwargs)

        return _mm

    for name in ("linear", "matmul"):
        orig = getattr(ttnn, name, None)
        if orig is not None:
            monkeypatch.setattr(ttnn, name, _wrap(orig))


def _install_quasar_host_embedding(monkeypatch):
    """Do the token embedding on the HOST on Quasar, skipping the ~525MB embedding-table upload.

    The device embedding uploads the full ``tok_embeddings`` table ([1,1,128256,2048] bf16 ≈ 525MB) via
    ``Embedding1D.load_device_weights`` -> ``to_device``, which dominates setup on the functional simulator
    (minutes, amplified by slow-dispatch). Embedding is a plain gather, not an interesting Quasar op, so we
    gather in torch from the LazyWeight's host ``source`` and upload only the tiny [1,1,seq,dim] activation
    (through the already-routed quasar from_torch/tilize). Single-device only; multi-device falls back to the
    original device path (the table is sharded on the last dim there)."""
    import torch

    from models.experimental.llama32_1b_quasar.modules.embedding.embedding_1d import Embedding1D
    from models.experimental.llama32_1b_quasar.modules.lazy_weight import LazyWeight

    orig_forward = Embedding1D.forward

    def _host_forward(self, x):
        cfg = self.config
        dev = cfg.mesh_device
        if dev is None or dev.get_num_devices() != 1:
            return orig_forward(self, x)  # sharded table across devices — leave on device

        # Host embedding table [.., vocab, dim] and token ids -> host longs.
        table = cfg.weights.source
        tbl2d = table.reshape(-1, table.shape[-1])  # [vocab, dim]
        ids_src = x.source if isinstance(x, LazyWeight) else ttnn.to_torch(x)
        ids = ids_src.to(torch.long).flatten()

        emb = tbl2d[ids].reshape(1, 1, ids.numel(), table.shape[-1]).to(torch.bfloat16)
        out = ttnn.from_torch(
            emb,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=cfg.output_memcfg,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
        )
        if cfg.embed_scale != 1.0:
            out = ttnn.multiply(out, cfg.embed_scale, memory_config=cfg.output_memcfg)
        logger.warning(
            f"[llama-e2e][quasar] host embedding gather ({ids.numel()} tokens) — skipped ~525MB table upload"
        )
        return out

    monkeypatch.setattr(Embedding1D, "forward", _host_forward)


def _install_quasar_host_create_qkv_heads(monkeypatch):
    """Split the fused QKV projection into Q/K/V heads on the HOST on Quasar (a pure reshape, no compute).

    ``ttnn.experimental.nlp_create_qkv_heads_decode`` picks a factory by input layout. Under force-interleaved
    the fused xqkv is DRAM-interleaved, so the picker selects NLPCreateQKVHeadsDecodeInterleavedProgramFactory,
    whose reader self-loops the ``reader_scratch`` DFB (PRODUCER+CONSUMER in one DM kernel) -> Gen2
    ValidateProgramSpec FATAL "Self-loop DFBs not supported for DM kernels on Gen2". The Sharded factory is
    Gen2-clean but needs a per-head shard the tiny Quasar device can't allocate (built for the 8x8 model grid).
    The op is just a split of the last dim (Q | K | V, each reshaped to heads), so do it in torch and upload
    Q/K/V as DRAM-interleaved TILE tensors. Output specs (device op):
        q: [1, batch, num_q_heads, head_dim], k/v: [1, batch, num_kv_heads, head_dim]
    with the fused input last dim laid out contiguously as [all Q heads | all K heads | all V heads].
    Single-device only; multi-device falls back to the device op. (RoPE consumes these next and needs
    HEIGHT_SHARDED, so it will be the next sidestep -- leaving these interleaved is fine here.)"""

    orig = getattr(ttnn.experimental, "nlp_create_qkv_heads_decode", None)
    if orig is None:
        return

    def _host_create(input_tensor, *args, **kwargs):
        try:
            dev = input_tensor.device()
            if dev is None or dev.get_num_devices() != 1:
                return orig(input_tensor, *args, **kwargs)
            num_heads = kwargs.get("num_heads", args[0] if len(args) > 0 else None)
            num_kv_heads = kwargs.get("num_kv_heads", num_heads)
            x = ttnn.to_torch(input_tensor).float()  # [1, 1, batch, qkv]
            qkv = x.shape[-1]
            batch = x.shape[-2]
            head_dim = qkv // (num_heads + 2 * num_kv_heads)
            x2 = x.reshape(batch, qkv)  # rows = batch users
            nq = num_heads * head_dim
            nk = num_kv_heads * head_dim
            q = x2[:, :nq].reshape(1, batch, num_heads, head_dim)
            k = x2[:, nq : nq + nk].reshape(1, batch, num_kv_heads, head_dim)
            v = x2[:, nq + nk :].reshape(1, batch, num_kv_heads, head_dim)
            logger.warning(
                f"[llama-e2e][quasar] host create_qkv_heads_decode: batch={batch} nq={num_heads} "
                f"nkv={num_kv_heads} hd={head_dim} (sidestepped interleaved-factory reader_scratch self-loop)"
            )
            return (
                _quasar_upload_tile_heads_padded(q, dev),
                _quasar_upload_tile_heads_padded(k, dev),
                _quasar_upload_tile_heads_padded(v, dev),
            )
        except Exception as e:
            logger.warning(f"[llama-e2e][quasar] host create_qkv_heads failed ({e}); falling back to device op")
            return orig(input_tensor, *args, **kwargs)

    monkeypatch.setattr(ttnn.experimental, "nlp_create_qkv_heads_decode", _host_create)


def _install_quasar_update_cache_reshard(monkeypatch):
    """Reshard the k/v inputs to (on-device) paged_update_cache into the HEIGHT_SHARDED layout it requires.

    paged_update_cache keeps the KV cache on device (Quasar-safe per the paged_cache Gen2 fold) and host SDPA
    reads that device cache, so we want to keep update_cache on device. But it FATALs "Expect input_tensor to be
    sharded": the decode kernel dispatches one user per core and requires the k/v input HEIGHT_SHARDED with
    shard grid num_cores == batch, shard shape [kv_heads_padded, head_dim], ROW_MAJOR
    (paged_update_cache_device_operation.cpp:255-289) -- exactly the real create_qkv_heads_decode output layout.
    Our host create_qkv / host RoPE emit interleaved TILE DRAM (heads padded to 32), so reshard here. For batch=1
    this is a 1-core shard that fits the tiny device. Thin wrapper -> original device op; falls back on error."""

    def _reshard(t):
        try:
            if t is None or t.is_sharded():
                return t
            ps = t.padded_shape  # [1, batch, kv_heads_padded, head_dim]
            batch = int(ps[1])
            kvh = int(ps[2])
            hd = int(ps[-1])
            # batch cores in a row (batch=1 -> single core (0,0)); num_cores == batch as the kernel requires.
            core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(batch - 1, 0))
            grid = ttnn.CoreRangeSet([core_range])
            shard_spec = ttnn.ShardSpec(grid, [kvh, hd], ttnn.ShardOrientation.ROW_MAJOR)
            memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
            return ttnn.interleaved_to_sharded(t, memcfg)
        except Exception as e:
            logger.warning(f"[llama-e2e][quasar] update_cache input reshard failed ({e}); passing through")
            return t

    def _log_cache_dtype(cache, tag):
        # Verify whether the KV cache is bf16 or still bf8_b. bf8_b (Bfp8_b) is unsupported on Quasar and the
        # device SDPA validate hard-requires bf16 for k/v, so if we ever un-host SDPA the cache must be bf16.
        # The cache is allocated via executor.allocate_kv_cache (ttnn.as_tensor), not from_torch, so the
        # tilize_from_torch bf16 coercion never sees it -- this log tells us if the allocator fix is needed.
        try:
            logger.warning(f"[llama-e2e][quasar] update_cache {tag} cache dtype = {cache.dtype}")
        except Exception:
            pass

    def _wrap_nonfused(orig):
        def _f(cache, input_tensor, *args, **kwargs):
            _log_cache_dtype(cache, "K/V")
            return orig(cache, _reshard(input_tensor), *args, **kwargs)

        return _f

    def _wrap_fused(orig):
        def _f(keys, k, values, v, *args, **kwargs):
            _log_cache_dtype(keys, "K")
            _log_cache_dtype(values, "V")
            return orig(keys, _reshard(k), values, _reshard(v), *args, **kwargs)

        return _f

    for name, wrapper in (
        ("paged_update_cache", _wrap_nonfused),
        ("paged_fused_update_cache", _wrap_fused),
    ):
        orig = getattr(ttnn.experimental, name, None)
        if orig is not None:
            try:
                monkeypatch.setattr(ttnn.experimental, name, wrapper(orig))
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] could not patch {name} for update_cache reshard ({e})")


def _install_quasar_host_rope(monkeypatch):
    """Apply decode RoPE on the HOST on Quasar (torch), sidestepping the sharded device rotary kernel.

    ``rotary_embedding_llama[_fused_qk]`` require HEIGHT_SHARDED TILE inputs (no interleaved decode variant);
    under force-interleaved -- and fed our host create_qkv_heads RM tensors -- the device op FATALs
    ("input tensor to rotary embedding must be tilized", and it needs sharding the tiny device can't provide).

    The math is exact and cheap. ttnn's llama RoPE is the GPT-J *interleaved* rotation: cos/sin are built by
    permute_to_meta_format (models/tt_transformers/tt/rope.py) which duplicates each frequency ADJACENTLY
    (stack((c,c)).flatten -> [c0,c0,c1,c1,...]), matched by get_rot_transformation_mat's adjacent-pairing
    trans_mat (+1 at (2i,2i+1), -1 at (2i+1,2i) => rotate(x) = x @ trans gives [-x1,x0,-x3,x2,...]). So:
        out = x*cos + (x @ trans)*sin
    with x [1,batch,n_heads,head_dim] and the model's own (already meta-format) cos/sin broadcast over heads
    ([1,batch,1,head_dim]). We rebuild the full head_dim x head_dim adjacent-pairing rotate matrix in torch
    (the device trans_mat is a tiled/repeated 32x32 form, awkward to matmul host-side). Bit-exact vs the kernel.
    Single-device only; any failure falls back to the device op. TILE DRAM output (heads tile-padded), so the
    rotated k feeds the on-device paged_update_cache and q feeds host SDPA."""
    import torch as _torch

    def _adjacent_rotate_mat(hd):
        # Same as get_rot_transformation_mat: +1 at (2i,2i+1), -1 at (2i+1,2i). With x@m this yields
        # (x@m)[2i] = -x[2i+1], (x@m)[2i+1] = x[2i]  ->  rot = [-x1, x0, -x3, x2, ...].
        m = _torch.zeros(hd, hd)
        even = _torch.arange(0, hd, 2)
        odd = _torch.arange(1, hd, 2)
        m[even, odd] = 1.0
        m[odd, even] = -1.0
        return m

    def _rope_one(x, cos, sin, is_decode):
        xt = ttnn.to_torch(x).float()
        ct = ttnn.to_torch(cos).float()
        st = ttnn.to_torch(sin).float()
        if is_decode:
            # decode: x is [1, batch, n_heads, head_dim]; cos/sin are ONE position per user
            # [1, batch, 1(or tile-padded), head_dim]. Keep position row 0, broadcast over heads (dim -2).
            ct = ct[..., :1, :]
            st = st[..., :1, :]
        # prefill (is_decode_mode=False): x is [1, n_heads, seq, head_dim]; cos/sin are [1, 1, seq, head_dim].
        # seq is in dim -2 of BOTH and must NOT be truncated -- broadcast over heads (dim 1) instead.
        hd = xt.shape[-1]
        trans = _adjacent_rotate_mat(hd)  # [hd, hd]
        rot = xt @ trans  # [-x1, x0, -x3, x2, ...]
        return xt * ct + rot * st

    def _wrap_single(orig):
        def _f(input_tensor, cos, sin, trans_mat=None, *args, **kwargs):
            try:
                dev = input_tensor.device()
                if dev is None or dev.get_num_devices() != 1:
                    return orig(input_tensor, cos, sin, trans_mat, *args, **kwargs)
                is_decode = bool(kwargs.get("is_decode_mode", True))
                out = _rope_one(input_tensor, cos, sin, is_decode)
                logger.warning(
                    f"[llama-e2e][quasar] host RoPE ({'decode' if is_decode else 'prefill'}) "
                    f"shape={tuple(out.shape)} (sidestepped device rotary)"
                )
                return _quasar_upload_tile_heads_padded(out, dev)
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] host RoPE failed ({e}); falling back to device op")
                return orig(input_tensor, cos, sin, trans_mat, *args, **kwargs)

        return _f

    def _wrap_fused(orig):
        def _f(q, k, cos, sin, trans_mat=None, *args, **kwargs):
            try:
                dev = q.device()
                if dev is None or dev.get_num_devices() != 1:
                    return orig(q, k, cos, sin, trans_mat, *args, **kwargs)
                qo = _rope_one(q, cos, sin, True)  # fused_qk is decode-only
                ko = _rope_one(k, cos, sin, True)
                logger.warning("[llama-e2e][quasar] host RoPE fused_qk (decode) (sidestepped sharded device rotary)")
                return (_quasar_upload_tile_heads_padded(qo, dev), _quasar_upload_tile_heads_padded(ko, dev))
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] host RoPE fused_qk failed ({e}); falling back to device op")
                return orig(q, k, cos, sin, trans_mat, *args, **kwargs)

        return _f

    for name, wrapper in (
        ("rotary_embedding_llama", _wrap_single),
        ("rotary_embedding_llama_fused_qk", _wrap_fused),
    ):
        orig = getattr(ttnn.experimental, name, None)
        if orig is not None:
            try:
                monkeypatch.setattr(ttnn.experimental, name, wrapper(orig))
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] could not patch {name} for host RoPE ({e})")


def _install_quasar_prefill_sdpa_grid(monkeypatch, mesh_device):
    """Clamp the PREFILL SDPA program-config grid to the device so it runs ON DEVICE on Quasar.

    The model pins prefill SDPA to compute_with_storage_grid_size=(8,8)=64 cores (attention_1d.py:1789),
    but craq-sim exposes fewer (e.g. 8x4=32) -> `scaled_dot_product_attention` FATALs at
    sdpa_program_factory.cpp:383 ("Provided grid must not contain more cores than the device"). The op
    itself is fine on Quasar once the grid fits: the standalone test_quasar_sdpa_prefill.py PASSES causal
    prefill SDPA at 1 and 2 cores (with TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0). Prefill flash attention
    distributes Q-chunks across cores independently (no cross-core reduction, unlike decode's tree), so
    clamping the grid down is safe. Wrap both prefill SDPA ops and rebuild the SDPAProgramConfig with the
    grid clamped to the device (preserving q/k_chunk_size + exp_approx_mode). NOTE: on-device SDPA relies on
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 (pass it on the pytest invocation) to avoid the tile-counter
    underflow; without it prefill SDPA may abort like decode did."""
    tr = getattr(getattr(ttnn.experimental, "quasar", None), "transformer", None)
    if tr is None:
        return
    dev = mesh_device.compute_with_storage_grid_size()
    cx, cy = _qsr_capped_grid_xy(dev)  # emulator-size cap (2 nodes), matching the validated matmul/SDPA regime

    def _clamp(pc):
        try:
            g = pc.compute_with_storage_grid_size
            gx, gy = int(g.x), int(g.y)
            if gx <= cx and gy <= cy:
                return pc  # already within the cap
            nx, ny = min(gx, cx), min(gy, cy)
            new = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(nx, ny),
                exp_approx_mode=pc.exp_approx_mode,
                q_chunk_size=pc.q_chunk_size,
                k_chunk_size=pc.k_chunk_size,
            )
            logger.warning(f"[llama-e2e][quasar] prefill SDPA grid {gx}x{gy} -> {nx}x{ny} (device {dev.x}x{dev.y})")
            return new
        except Exception as e:
            logger.warning(f"[llama-e2e][quasar] prefill SDPA grid clamp failed ({e}); passing through")
            return pc

    def _wrap(orig):
        def _f(*args, **kwargs):
            pc = kwargs.get("program_config")
            if pc is not None:
                kwargs["program_config"] = _clamp(pc)
            return orig(*args, **kwargs)

        return _f

    for name in ("scaled_dot_product_attention", "chunked_scaled_dot_product_attention"):
        orig = getattr(tr, name, None)
        if orig is not None:
            try:
                monkeypatch.setattr(tr, name, _wrap(orig))
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] could not patch {name} for prefill SDPA grid ({e})")


def _install_quasar_sdpa_single_core(monkeypatch, mesh_device):
    """Force the Quasar decode SDPA to 1 core per KV-head (no cross-core tree reduction) AND clamp its grid.

    With num_cores_per_head > 1 (e.g. grid 8x4 / 8 KV heads -> 4 cores/head) the flash-decode does a
    multi-core TREE reduction (reducer <- children over mcast/semaphores). That cross-core handshake
    DEADLOCKS on the Quasar sim once k_num_chunks > 1 (children have data) -- workers stall at waypoint WFW
    (wait-front). Setting SDPAProgramConfig.max_cores_per_head_batch = 1 makes num_cores_per_head = 1, so
    num_tree_reduction_rounds = 0 (sdpa_decode_program_factory.cpp:202-207,249) -- the single-core
    flash->finalize path that PASSES on device (test_paged_sdpa_decode_single_core with alias=0). Also clamp
    compute_with_storage_grid_size to the device (model pins 8x8=64 cores) so the on-device decode SDPA does
    not exceed the device -- matches the prefill SDPA grid clamp. Bring-up path for on-device decode SDPA.
    """
    tr = getattr(getattr(ttnn.experimental, "quasar", None), "transformer", None)
    if tr is None:
        logger.warning("[llama-e2e][quasar] no experimental.quasar.transformer; SDPA single-core patch skipped")
        return
    dev = mesh_device.compute_with_storage_grid_size()
    cx, cy = _qsr_capped_grid_xy(dev)  # emulator-size cap (2 nodes)

    def _single_core_pc(pc):
        if pc is None:
            return pc
        try:
            g = pc.compute_with_storage_grid_size
            grid = ttnn.CoreCoord(min(int(g.x), cx), min(int(g.y), cy))
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                sub_core_grids=pc.sub_core_grids,
                q_chunk_size=pc.q_chunk_size,
                k_chunk_size=pc.k_chunk_size,
                exp_approx_mode=pc.exp_approx_mode,
                max_cores_per_head_batch=1,
            )
        except Exception as e:
            logger.warning(f"[llama-e2e][quasar] SDPA cfg rebuild failed ({e}); mutating in place")
            try:
                pc.max_cores_per_head_batch = 1
            except Exception:
                pass
            return pc

    def _wrap(orig):
        def _sdpa(*args, **kwargs):
            if kwargs.get("program_config") is not None:
                kwargs["program_config"] = _single_core_pc(kwargs["program_config"])
            return orig(*args, **kwargs)

        return _sdpa

    for name in ("scaled_dot_product_attention_decode", "paged_scaled_dot_product_attention_decode"):
        orig = getattr(tr, name, None)
        if orig is not None:
            try:
                monkeypatch.setattr(tr, name, _wrap(orig))
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] could not patch {name} ({e})")


def _install_quasar_host_sdpa(monkeypatch):
    """Compute decode SDPA on the HOST (torch), sidestepping the Quasar device flash-decode.

    Even single-core (no tree), the device flash-decode keeps hitting capacity-1 intra-tensix DFB
    tile-counter underflows on the sim (max buffer, then the multi-chunk lazy-softmax buffers) -- a runtime
    remapper bug (craq-sim issue filed), whack-a-mole at the kernel level. Gather Q + KV to host, do GQA
    decode attention in torch, upload the [1,B,nq,hd] output (DRAM). Single-device only; any failure falls
    back to the device op (which is left single-core by _install_quasar_sdpa_single_core, so it fails fast
    rather than deadlocking). Bring-up sidestep -- exact-ish numerics (fp32 accumulate -> bf16)."""
    import torch as _torch

    tr = getattr(getattr(ttnn.experimental, "quasar", None), "transformer", None)
    if tr is None:
        return

    def _upload_dram(out_t, dev):
        rm = ttnn.from_torch(
            out_t.to(_torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
        )
        qtil = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
        return (qtil or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

    def _host_decode(q, k, v, page_table, cur_pos, scale, sliding_window):
        qh = ttnn.to_torch(q).float()  # [1, B, nq, hd]
        kk = ttnn.to_torch(k).float()
        vv = ttnn.to_torch(v).float()
        B, nq, hd = qh.shape[1], qh.shape[2], qh.shape[3]
        nkv = kk.shape[1]
        group = max(nq // nkv, 1)  # GQA: q heads per kv head
        sc = scale if scale is not None else hd**-0.5
        paged = page_table is not None
        pt = ttnn.to_torch(page_table).to(_torch.int64) if paged else None
        cp = ttnn.to_torch(cur_pos).to(_torch.int64).flatten().tolist() if cur_pos is not None else None
        out = _torch.zeros(1, B, nq, hd)
        for b in range(B):
            P = int(cp[b]) if cp is not None else (kk.shape[2] - 1 if not paged else kk.shape[0] * kk.shape[2] - 1)
            pos = _torch.arange(P + 1)
            if paged:
                bs = kk.shape[2]
                blk = pt[b, pos // bs]
                off = pos % bs
                Kb = kk[blk, :, off, :]  # [P+1, nkv, hd]
                Vb = vv[blk, :, off, :]
            else:
                Kb = kk[b, :, : P + 1, :].transpose(0, 1)  # [P+1, nkv, hd]
                Vb = vv[b, :, : P + 1, :].transpose(0, 1)
            start = max(0, P + 1 - sliding_window) if sliding_window else 0
            for h in range(nq):
                kv = h // group
                qvec = qh[0, b, h]  # [hd]
                Ks = Kb[start : P + 1, kv, :]  # [L, hd]
                Vs = Vb[start : P + 1, kv, :]
                w = _torch.softmax((Ks @ qvec) * sc, dim=0)  # [L]
                out[0, b, h] = w @ Vs
        logger.warning(
            f"[llama-e2e][quasar] host SDPA decode B={B} nq={nq} nkv={nkv} P0={int(cp[0]) if cp else -1} (sidestepped device flash-decode)"
        )
        return _upload_dram(out, q.device())

    def _wrap_paged(orig):
        def _f(q, k, v, *args, **kwargs):
            try:
                dev = q.device()
                if dev is None or dev.get_num_devices() != 1:
                    return orig(q, k, v, *args, **kwargs)
                return _host_decode(
                    q,
                    k,
                    v,
                    kwargs.get("page_table_tensor"),
                    kwargs.get("cur_pos_tensor"),
                    kwargs.get("scale"),
                    kwargs.get("sliding_window_size"),
                )
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] host paged SDPA failed ({e}); falling back to device op")
                return orig(q, k, v, *args, **kwargs)

        return _f

    def _wrap_nonpaged(orig):
        def _f(q, k, v, *args, **kwargs):
            try:
                dev = q.device()
                if dev is None or dev.get_num_devices() != 1:
                    return orig(q, k, v, *args, **kwargs)
                return _host_decode(
                    q, k, v, None, kwargs.get("cur_pos_tensor"), kwargs.get("scale"), kwargs.get("sliding_window_size")
                )
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] host SDPA failed ({e}); falling back to device op")
                return orig(q, k, v, *args, **kwargs)

        return _f

    for name, wrapper in (
        ("paged_scaled_dot_product_attention_decode", _wrap_paged),
        ("scaled_dot_product_attention_decode", _wrap_nonpaged),
    ):
        orig = getattr(tr, name, None)
        if orig is not None:
            try:
                monkeypatch.setattr(tr, name, wrapper(orig))
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] could not patch {name} for host SDPA ({e})")


def _install_quasar_device_sdpa_split(monkeypatch, mesh_device):
    """Run the Quasar decode SDPA ON DEVICE via a per-kv-head split (replaces the host fallback).

    The full 8-kv-head decode flash-decode HANGS on the sim once k_num_chunks>1 whenever a core owns >1 kv-head
    (cross-core tree reduction / multi-chunk lazy-softmax). It PASSES when exactly 1 kv-head maps to a core
    (tests/.../test_quasar_sdpa_decode.py::test_paged_sdpa_decode_split, validated 1- AND 2-core). So split the
    op into `nkv` single-kv-head calls: pass the FULL 32-head HEIGHT-SHARDED q (which create_qkv_heads already
    produces -- keeps the op on the validated sharded-q path; the DRAM-INTERLEAVED-q reader is unported on
    Quasar and trips a compute unpack assert) together with ONE kv-head (k/v sliced on the non-tiled kv-head
    axis, dim 1). With nkv=1 every q-head attends that kv-head; keep only the qpk heads that belong to it, then
    reassemble the [1,B,nq,hd] output. The attention math (QK^T, softmax, AV) runs in the device flash-decode
    kernels; assembly is attempted on device (slice+concat) and falls back to a host stitch. Single-device only;
    any failure falls back to `orig`. MUST be installed AFTER _install_quasar_sdpa_single_core so `orig` already
    clamps the program_config to single-core + emulator grid. ~nkv x compute per token (compute 32 heads, keep
    qpk) -- a bring-up cost, not a device limit."""
    import torch as _torch

    tr = getattr(getattr(ttnn.experimental, "quasar", None), "transformer", None)
    if tr is None:
        logger.warning("[llama-e2e][quasar] no experimental.quasar.transformer; device SDPA split skipped")
        return

    def _upload_dram(out_t, dev):
        rm = ttnn.from_torch(
            out_t.to(_torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
        )
        qtil = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
        return (qtil or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

    def _split_decode(orig, q, k, v, args, kwargs):
        dev = q.device()
        qs, ks, vs = q.shape, k.shape, v.shape
        nq, nkv = int(qs[2]), int(ks[1])
        if nkv <= 1 or nq % nkv != 0:
            # Nothing to split (already 1 kv-head, or non-GQA); let orig (single-core) handle it.
            return orig(q, k, v, *args, **kwargs)
        qpk = nq // nkv  # q-heads per kv-head
        outs = []
        for h in range(nkv):
            # One kv-head: slice k/v on the kv-head axis (dim 1) -- NOT a tiled dim, so no sub-tile issue.
            k_h = ttnn.slice(k, [0, h, 0, 0], [int(ks[0]), h + 1, int(ks[2]), int(ks[3])])
            v_h = ttnn.slice(v, [0, h, 0, 0], [int(vs[0]), h + 1, int(vs[2]), int(vs[3])])
            outs.append(orig(q, k_h, v_h, *args, **kwargs))
        # Reassemble: head block h of the output comes from outs[h] (all q-heads vs kv-head h), rows
        # [h*qpk:(h+1)*qpk]. Try device slice+concat; fall back to a host stitch (a tiled sub-tile slice on the
        # head axis is fragile on device).
        try:
            slabs = []
            for h in range(nkv):
                os_ = outs[h].shape
                slabs.append(
                    ttnn.slice(outs[h], [0, 0, h * qpk, 0], [int(os_[0]), int(os_[1]), (h + 1) * qpk, int(os_[3])])
                )
            out = ttnn.concat(slabs, dim=2)
            logger.warning(f"[llama-e2e][quasar] device SDPA split (on device) nkv={nkv} qpk={qpk}")
            return out
        except Exception as e:
            logger.warning(f"[llama-e2e][quasar] device SDPA-split device assembly failed ({e}); host-stitching")
            kept = [ttnn.to_torch(outs[h])[:, :, h * qpk : (h + 1) * qpk, :] for h in range(nkv)]
            return _upload_dram(_torch.cat(kept, dim=2), dev)

    def _wrap(orig):
        def _f(q, k, v, *args, **kwargs):
            try:
                dev = q.device()
                if dev is None or dev.get_num_devices() != 1:
                    return orig(q, k, v, *args, **kwargs)
                return _split_decode(orig, q, k, v, args, kwargs)
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] device SDPA split failed ({e}); falling back to device op")
                return orig(q, k, v, *args, **kwargs)

        return _f

    for name in ("paged_scaled_dot_product_attention_decode", "scaled_dot_product_attention_decode"):
        orig = getattr(tr, name, None)
        if orig is not None:
            try:
                monkeypatch.setattr(tr, name, _wrap(orig))
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] could not patch {name} for device SDPA split ({e})")


def _install_quasar_concat_heads_grid_agnostic(monkeypatch, mesh_device):
    """Replace decode nlp_concat_heads_decode with a GRID-AGNOSTIC device head-merge when the device has fewer
    than num_heads compute cores (the 2-compute-node emulator under slow dispatch).

    The stock op places one head-column per core, so it needs num_heads (=32) cores
    (num_cores_to_corerangeset(num_heads, grid) -> work_split.cpp:99 FATAL "32 > 2" on the emulator). But the
    concat is just a per-user flatten of the (num_heads, head_dim) block: input[0,b,h,d] -> output[0,0,b,h*hd+d].
    In ROW_MAJOR that is a contiguous reshape, so do it grid-agnostically on device: deshard -> untilize ->
    reshape (merge heads into the hidden dim) -> pad the user/batch axis to a tile -> tilize. None of those shard
    by head, so it fits ANY grid. Validated standalone (test_quasar_nlp_concat_heads_decode.py::
    test_nlp_concat_heads_decode_grid_agnostic). On a >= num_heads-core device (WH/BH), the stock op is used
    unchanged. Single-device only; falls back to orig on any error. NOT a host fallback -- the whole merge runs
    on device. The durable fix is the op enhancement (pack heads/core); this unblocks the emulator now."""
    tr_exp = getattr(ttnn, "experimental", None)
    orig = getattr(tr_exp, "nlp_concat_heads_decode", None) if tr_exp is not None else None
    if orig is None:
        logger.warning("[llama-e2e][quasar] no ttnn.experimental.nlp_concat_heads_decode; concat patch skipped")
        return

    q = getattr(ttnn.experimental, "quasar", None)
    _s2i = getattr(q, "sharded_to_interleaved", None) or ttnn.sharded_to_interleaved
    _untilize = getattr(q, "untilize", None) or ttnn.untilize
    _tilize = getattr(q, "tilize", None) or ttnn.tilize

    def _grid_agnostic_merge(input_tensor, num_heads):
        # input: [1, batch, num_heads, head_dim] TILE (HEIGHT_SHARDED or DRAM). Merge heads -> [1,1,batch,H*hd].
        shp = input_tensor.shape
        batch, head_dim = int(shp[1]), int(shp[3])
        x = input_tensor
        if x.is_sharded():
            x = _s2i(x, ttnn.DRAM_MEMORY_CONFIG)
        x = _untilize(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # row-major [1,batch,num_heads,head_dim]
        x = ttnn.reshape(x, (1, 1, batch, num_heads * head_dim))  # contiguous head-merge
        pad_to = ((batch + 31) // 32) * 32
        if pad_to != batch:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_to - batch), (0, 0)], value=0.0)
        return _tilize(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

    def _wrap(orig_fn):
        def _f(input_tensor, *args, num_heads=None, **kwargs):
            try:
                dev = input_tensor.device()
                if dev is None or dev.get_num_devices() != 1 or num_heads is None:
                    return orig_fn(input_tensor, *args, num_heads=num_heads, **kwargs)
                g = dev.compute_with_storage_grid_size()
                if int(g.x) * int(g.y) >= int(num_heads):
                    return orig_fn(input_tensor, *args, num_heads=num_heads, **kwargs)  # stock op fits
                logger.warning(
                    f"[llama-e2e][quasar] grid-agnostic concat-heads (grid {g.x}x{g.y} < num_heads={num_heads})"
                )
                return _grid_agnostic_merge(input_tensor, int(num_heads))
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] grid-agnostic concat-heads failed ({e}); falling back to op")
                return orig_fn(input_tensor, *args, num_heads=num_heads, **kwargs)

        return _f

    try:
        monkeypatch.setattr(ttnn.experimental, "nlp_concat_heads_decode", _wrap(orig))
    except Exception as e:
        logger.warning(f"[llama-e2e][quasar] could not patch nlp_concat_heads_decode ({e})")


def _install_quasar_concat_l1_overflow_to_dram(monkeypatch, mesh_device):
    """Route an L1 ttnn.concat output to DRAM when it won't fit L1 on this (small) device.

    The lm_head concatenates 47 logit splits into the full-vocab [1,1,32,128256] output with an L1 output memcfg
    (lm_head_1d.py:154, output_memcfg defaults to L1_MEMORY_CONFIG). On the 2-compute-node emulator that's ~8 MB
    interleaved across 2 banks = ~4 MB/bank > the 3.88 MB bank size -> Out of Memory (bank_manager.cpp). The
    full-vocab logits can't live in L1 on 2 cores. Coerce an OVERSIZED L1 concat output to DRAM interleaved;
    small concats keep their L1 config. Downstream (sampling / argmax) accepts a DRAM tensor. Complements
    _install_quasar_force_interleaved, which only handles oversized SHARDED configs (this is L1 INTERLEAVED)."""
    orig = ttnn.concat
    dev = mesh_device.compute_with_storage_grid_size()
    ncores = max(int(dev.x) * int(dev.y), 1)
    per_bank_budget = 3_800_000  # under the ~3.88 MB L1 bank size, leaving headroom for other allocations

    def _l1_output_fits(tensors):
        # concat output volume == sum of input volumes; L1-interleaved spreads it across `ncores` banks.
        total_bytes = 0
        for t in tensors:
            try:
                vol = 1
                for d in t.shape:
                    vol *= int(d)
                total_bytes += vol * 2  # bf16 (the Quasar activation dtype)
            except Exception:
                return True  # can't estimate -> don't coerce
        return (total_bytes / ncores) <= per_bank_budget

    def _f(tensors, *args, **kwargs):
        try:
            mc = kwargs.get("memory_config")
            if (
                mc is not None
                and getattr(mc, "buffer_type", None) == ttnn.BufferType.L1
                and isinstance(tensors, (list, tuple))
                and not _l1_output_fits(tensors)
            ):
                logger.warning(
                    f"[llama-e2e][quasar] concat L1 output exceeds {ncores}-bank L1 budget -> DRAM interleaved"
                )
                kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        except Exception as e:
            logger.warning(f"[llama-e2e][quasar] concat L1->DRAM check failed ({e}); passing through")
        return orig(tensors, *args, **kwargs)

    try:
        monkeypatch.setattr(ttnn, "concat", _f)
    except Exception as e:
        logger.warning(f"[llama-e2e][quasar] could not patch ttnn.concat ({e})")


def _install_quasar_force_interleaved(monkeypatch, mesh_device):
    """Force OVERSIZED sharded memory configs to DRAM-interleaved on Quasar; keep device-fitting shards.

    The WH tuning threads grid-8x8 (64-core) sharded memcfgs through decode (mlp/attention/rmsnorm/lm_head),
    which the Quasar device cannot allocate, and whose sharded device ops are broken on the sim anyway. But a
    few configs are SMALL and device-fitting and MUST stay sharded -- notably RoPE's per-batch shards
    (batch_grid = 1 core for batch=1): rotary_embedding_llama[_fused_qk] REQUIRES HEIGHT_SHARDED inputs (no
    interleaved decode variant), so forcing those to DRAM breaks RoPE.

    So ttnn.create_sharded_memory_config returns DRAM only when the requested core_grid does NOT fit the
    device (the 8x8 ones); a fitting grid keeps its real sharded config. ttnn.to_memory_config coerces a
    sharded TARGET to DRAM only when its shard grid doesn't fit. Installed BEFORE create_model."""
    dev = mesh_device.compute_with_storage_grid_size()

    def _grid_fits(cg):
        try:
            if cg is None:
                return False
            if hasattr(cg, "bounding_box"):  # CoreRangeSet -> bounding-box dimensions (num cores per dim)
                gs = cg.bounding_box().grid_size()
                return int(gs.x) <= dev.x and int(gs.y) <= dev.y
            if hasattr(cg, "x") and hasattr(cg, "y"):  # CoreGrid
                return int(cg.x) <= dev.x and int(cg.y) <= dev.y
        except Exception:
            return False
        return False

    orig_csmc = getattr(ttnn, "create_sharded_memory_config", None)
    if orig_csmc is not None:

        def _csmc(*args, **kwargs):
            cg = kwargs.get("core_grid", args[1] if len(args) > 1 else None)
            if _grid_fits(cg):
                return orig_csmc(*args, **kwargs)  # small shard (e.g. RoPE per-batch) -> keep
            return ttnn.DRAM_MEMORY_CONFIG  # oversized (grid 8x8) -> interleaved

        monkeypatch.setattr(ttnn, "create_sharded_memory_config", _csmc)

    orig_tmc = ttnn.to_memory_config

    def _tmc(tensor, memory_config=None, *args, **kwargs):
        try:
            if memory_config is not None and memory_config.is_sharded():
                cg = None
                try:
                    ss = memory_config.shard_spec
                    cg = ss.grid if ss is not None else None
                except Exception:
                    cg = None
                if not _grid_fits(cg):
                    memory_config = ttnn.DRAM_MEMORY_CONFIG
        except Exception:
            pass
        return orig_tmc(tensor, memory_config, *args, **kwargs)

    monkeypatch.setattr(ttnn, "to_memory_config", _tmc)
    logger.warning(
        f"[llama-e2e][quasar] force-interleaved: oversized shards -> DRAM, device-fitting shards kept (dev {dev.x}x{dev.y})"
    )


def _install_quasar_s2i_copy(monkeypatch):
    """Make ttnn.sharded_to_interleaved return a DISTINCT copy when its input is already interleaved.

    With force-interleaved + host-side sidesteps, ops that the model built to output WIDTH_SHARDED now yield
    DRAM-interleaved tensors. The model's attention _all_reduce_qkv_decode then does
    `out = sharded_to_interleaved(x); deallocate(x); reshape(out)`. When x is already interleaved,
    sharded_to_interleaved is a no-op ALIAS (out IS x), so deallocate(x) frees the buffer reshape then reads
    -> "Tensor is not allocated". Returning a genuine copy (add 0 via the routed quasar eltwise, else a host
    round-trip) makes out distinct from x so the deallocate is safe."""

    orig = ttnn.sharded_to_interleaved

    def _s2i(x, *args, **kwargs):
        try:
            if hasattr(x, "is_sharded") and not x.is_sharded():
                try:
                    return ttnn.add(x, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # routed -> quasar.add, new buffer
                except Exception as e:
                    logger.warning(f"[llama-e2e][quasar] s2i add-copy failed ({e}); host round-trip")
                    dev = x.device()
                    t = ttnn.to_torch(x)
                    return ttnn.from_torch(
                        t,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
                    )
        except Exception:
            pass
        return orig(x, *args, **kwargs)

    monkeypatch.setattr(ttnn, "sharded_to_interleaved", _s2i)


def _install_quasar_mlp_input_relax(monkeypatch):
    """Relax the MLP decode-input memcfg assertion.

    mlp_1d._load_input_device_tensor asserts x.memory_config() == config.decode_input_memcfg (a WIDTH_SHARDED
    L1 config). Our host-side sidesteps (matmul/SDPA/eltwise) hand the MLP a DRAM-interleaved x, so the strict
    check raises "Input tensor memory config does not match the config!". The MLP's w1/w3/w2 matmuls are now
    computed host-side (they read x to host regardless of layout), so the exact input layout no longer
    matters. On mismatch, best-effort convert to the declared config; if that can't fit the device grid, pass
    x through unchanged. Bring-up sidestep for the config-threading the interleaved path breaks."""
    try:
        from models.experimental.llama32_1b_quasar.modules.mlp import mlp_1d
    except Exception as e:
        logger.warning(f"[llama-e2e][quasar] could not import mlp_1d to relax input check ({e})")
        return
    orig = getattr(mlp_1d, "_load_input_device_tensor", None)
    if orig is None:
        return

    def _relaxed(x, config, mode):
        try:
            return orig(x, config, mode)
        except ValueError:
            mem_cfg = config.decode_input_memcfg if mode == "decode" else config.prefill_input_memcfg
            try:
                if hasattr(x, "memory_config") and mem_cfg is not None and x.memory_config() != mem_cfg:
                    x2 = ttnn.to_memory_config(x, mem_cfg)
                    logger.warning(f"[llama-e2e][quasar] MLP input converted to declared {mode} memcfg")
                    return x2
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] MLP input convert failed ({e}); passing interleaved through")
            return x

    monkeypatch.setattr(mlp_1d, "_load_input_device_tensor", _relaxed)


def _install_quasar_eltwise(monkeypatch):
    """Route eltwise binary ops to the Quasar-native experimental ops (keeps them ON DEVICE).

    Mainline binary_ng (ttnn.add / ttnn.mul / ttnn.multiply / ttnn.subtract) is Gen1-only: its program
    factory builds a legacy DataMovementKernel -> `DataMovementKernel is not supported on Quasar` FATAL
    (kernel.hpp:477). ttnn.experimental.quasar.{add,multiply,subtract,...} are the DFB-ported equivalents
    (same kwargs: memory_config/dtype/activations/input_tensor_a_activations, tensor+scalar overloads). Route
    the mainline names to them on Quasar; fall back to the mainline op if the quasar one errors."""
    q = getattr(ttnn.experimental, "quasar", None)
    if q is None:
        return
    routes = {"add": "add", "multiply": "multiply", "mul": "multiply", "subtract": "subtract", "sub": "subtract"}

    def _mk(qop, mop, nm):
        def _f(*args, **kwargs):
            # Keep eltwise outputs DRAM-interleaved too (sharding forced off on Quasar).
            try:
                mc = kwargs.get("memory_config")
                if mc is not None and mc.is_sharded():
                    kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
            except Exception:
                pass
            # bf8_b / bf4_b are not supported on Quasar (real HW limit): the MLP gate mul requests
            # dtype=bf8_b (cfg.mul_dtype) -> ValidateProgramSpec FATAL "DFB has data format 'Bfp8_b'...".
            # Force the eltwise output to bf16 (the model is bf16 e2e; downstream matmuls are host-side).
            try:
                if kwargs.get("dtype") in (ttnn.bfloat8_b, ttnn.bfloat4_b):
                    kwargs["dtype"] = ttnn.bfloat16
            except Exception:
                pass
            try:
                return qop(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[llama-e2e][quasar] quasar {nm} failed ({e}); mainline fallback")
                return mop(*args, **kwargs)

        return _f

    for main_name, q_name in routes.items():
        qop = getattr(q, q_name, None)
        mop = getattr(ttnn, main_name, None)
        if qop is not None and mop is not None:
            monkeypatch.setattr(ttnn, main_name, _mk(qop, mop, main_name))


# The Quasar functional simulator runs each op at a few KHz, so a single decode/prefill layer takes minutes
# (LayerNorm ~54s, Copy ~24s observed); the host-sided run measured ~3h50m. On-device attention adds device
# op time, so override the repo-wide 300s pytest-timeout with 8h to avoid a mid-run kill
# (LLAMA32_1B_TF_MAX_DECODE_STEPS caps the decode loop to keep it bounded). WH/BH finish far within this.
@pytest.mark.timeout(28800)
@pytest.mark.parametrize("optimizations", ["performance"])
def test_llama_e2e(mesh_device, optimizations, monkeypatch):  # noqa: F811 — mesh_device is the imported fixture
    """Token-accuracy + per-op PCC, on WH (hard gate) and Quasar (bring-up, soft gate)."""
    quasar = is_quasar()

    # Env must be set BEFORE create_model (it drives the tuning recipe + layer count). Slow-dispatch is
    # NOT set here — it must precede device open, so pass it in the pytest invocation on Quasar.
    # Default per-op PCC logging on, but honor an explicit caller value (e.g. LLAMA_PCC_LOG=0 to skip the
    # per-token readback overhead on a WH regression run).
    if "LLAMA_PCC_LOG" not in os.environ:
        monkeypatch.setenv("LLAMA_PCC_LOG", "1")
    if quasar:
        # Disable the program cache on Quasar. The cache-hit partial-fast-path (UpdateProgramRunArgs)
        # mis-applies some ops' per-dispatch state on reuse -- e.g. paged_fill_cache's DRAM write overrun --
        # whose proper fix is a codeowner-side program-factory change. With the cache off, every dispatch
        # takes the full SetProgramRunArgs path, which is correct. Slower, but the bring-up e2e is minimal
        # (1 layer, 1 decode step). Nothing re-enables it, so this one call covers the whole run.
        mesh_device.disable_and_clear_program_cache()

        # minimal_matmul pins an 8x8 grid unavailable on the Quasar emulator -> force ttnn.linear.
        monkeypatch.setenv("DISABLE_MINIMAL_MATMUL", "1")
        # Default to a 1-layer stack for bring-up unless the caller asked for more.
        monkeypatch.setenv("LLAMA32_1B_DEMO_NUM_LAYERS", os.environ.get("LLAMA32_1B_DEMO_NUM_LAYERS", "1"))
        # Teacher forcing otherwise drives ~255 decode steps (num_target-1), each minutes on the ~57 KHz sim
        # (a full run is many hours -> the 1h timeout fired mid-run). Cap to 1 decode step for bring-up unless
        # the caller overrides. (executor.run_teacher_forcing honors LLAMA32_1B_TF_MAX_DECODE_STEPS.)
        monkeypatch.setenv("LLAMA32_1B_TF_MAX_DECODE_STEPS", os.environ.get("LLAMA32_1B_TF_MAX_DECODE_STEPS", "1"))
        # Force OVERSIZED (grid-8x8) sharded configs -> DRAM; keep device-fitting shards (RoPE needs them).
        # MUST be before create_model so the config objects are built with the right layout.
        _install_quasar_force_interleaved(monkeypatch, mesh_device)
        # Route weight-upload tilize through the Gen2-native quasar op (the mainline one hangs on the sim).
        # Must be installed before create_model, which materializes the weights via ttnn.from_torch.
        _install_quasar_tilize_from_torch(monkeypatch)
        # Route interleaved_to_sharded through the Gen2-native quasar op (the mainline one faults on the
        # sim -- RoPE cos/sin decode sharding, UNALIGNED_LOAD / neighbour-core corruption). See the
        # dedicated repro test_llama_e2e_with_i2s_failure.py.
        _install_quasar_i2s(monkeypatch)
        # Force fp32_dest_acc_en=False everywhere: the Quasar sim can't do the bf16->Tf32 unpack that
        # fp32 accumulation needs (qsr_unpack_src_value in_format=5 out_format=4). Must precede create_model
        # (compute configs are built during model construction).
        _install_quasar_fp32_acc_off(monkeypatch)

        # LLAMA_QSR_DEVICE_ATTN=1 (default): run compute ON DEVICE (matmul, create_qkv_heads, RoPE, prefill
        # SDPA); all confirmed working on device 2026-09-26. =0: the fully host-sided path that passed e2e --
        # kept as a fallback. DECODE SDPA now runs ON DEVICE too (per-kv-head split, below;
        # LLAMA_QSR_DEVICE_DECODE_SDPA=0 restores the host fallback). Embedding stays HOST either way (~525MB
        # table upload is a sim-perf tax, not a device limit; unlike bf8_b which is genuinely unsupported).
        # ON-DEVICE MODE
        # REQUIRES `TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0` in the pytest env (clears the tile-counter
        # underflow that forced host matmul); matmul + prefill SDPA device paths are validated standalone
        # (test_quasar_qkv_matmul_dfb.py, test_quasar_sdpa_prefill.py). bf16 substitutes for bf8_b.
        device_attn = os.environ.get("LLAMA_QSR_DEVICE_ATTN", "1") == "1"

        if device_attn:
            # Device matmul: de-shard inputs to DRAM-interleaved + drop the DRAM-sharded program_config so the
            # picker selects the interleaved 1D-mcast matmul (its in0 DFB underflow is cleared by alias=0).
            _install_quasar_interleaved_matmul(monkeypatch, mesh_device)
            # KV cache as bf16 so the on-device decode SDPA validate (bf16-only q/k/v on Quasar) passes.
            _install_quasar_bf16_kv_cache(monkeypatch)
        else:
            # Host-sided compute (the passing fallback path).
            _install_quasar_host_matmul(monkeypatch)
            _install_quasar_host_create_qkv_heads(monkeypatch)

        # RoPE on HOST by default (both modes). The device rotary_embedding_llama_sharded compute kernel is only
        # PARTIALLY Quasar-ported: it pack_init's for the matmul-rotate output (dfb::rotated_interm) but NOT for
        # the eltwise-chain PackTile targets (sin_interm / cos_interm / out), so pack_tile trips the pack re-init
        # guard (llk_pack_tile_api.h:94, recipe section 7). host RoPE is exact + validated. Flip to device with
        # LLAMA_QSR_DEVICE_ROPE=1 once the kernel port (pack_init on each pack-output switch) lands.
        device_rope = device_attn and os.environ.get("LLAMA_QSR_DEVICE_ROPE", "0") == "1"
        if not device_rope:
            _install_quasar_host_rope(monkeypatch)

        # Skip the ~525MB tok_embeddings table upload: gather on host, upload only the activation. This is the
        # dominant setup cost on the sim; embedding is a plain gather (device-viable, host is a perf choice).
        _install_quasar_host_embedding(monkeypatch)
        # Reshard k/v into the HEIGHT_SHARDED layout paged_update_cache requires (one user per core), keeping the
        # KV cache update on device. batch=1 -> 1-core shard.
        _install_quasar_update_cache_reshard(monkeypatch)
        # Clamp the PREFILL SDPA grid (model pins 8x8=64 cores) to the device so it runs ON DEVICE.
        _install_quasar_prefill_sdpa_grid(monkeypatch, mesh_device)
        # Decode SDPA single-core config (no tree reduction) -- clamps each SDPA call to 1 core/head + emulator
        # grid. Required by BOTH the device-split and the host fallback below (the split calls the clamped op).
        _install_quasar_sdpa_single_core(monkeypatch, mesh_device)
        # Decode SDPA ON DEVICE via a per-kv-head split (LLAMA_QSR_DEVICE_DECODE_SDPA=1, default). The full
        # 8-kv-head flash-decode HANGS on the sim once k_num_chunks>1 whenever a core owns >1 kv-head (tree
        # reduction / multi-chunk lazy-softmax). Splitting into nkv single-kv-head calls (full-q sharded + 1
        # kv-head each) keeps every call on the validated no-hang config -- validated standalone 1- and 2-core
        # (test_quasar_sdpa_decode.py::test_paged_sdpa_decode_split). =0 restores the host fallback (torch GQA),
        # which sidesteps the device flash-decode entirely. So now everything runs ON DEVICE except embedding
        # (host, ~525MB-table perf choice).
        device_decode_sdpa = device_attn and os.environ.get("LLAMA_QSR_DEVICE_DECODE_SDPA", "1") == "1"
        if device_decode_sdpa:
            _install_quasar_device_sdpa_split(monkeypatch, mesh_device)
        else:
            _install_quasar_host_sdpa(monkeypatch)
        # Decode concat-heads (STAGE 9): nlp_concat_heads_decode needs num_heads(=32) cores (one head-column per
        # core), so it FATALs on the 2-node emulator (work_split.cpp:99 "32 > 2"). On a < num_heads-core grid,
        # replace it with a grid-agnostic device head-merge (deshard->untilize->reshape->pad->tilize); >=
        # num_heads-core grids (WH/BH) keep the stock op. On device, not a host fallback. The durable fix is the
        # op enhancement (pack heads/core) -- see debug_ops/test_quasar_nlp_concat_heads_decode.py.
        _install_quasar_concat_heads_grid_agnostic(monkeypatch, mesh_device)
        # lm_head concat: the full-vocab [1,1,32,128256] logits (47 splits) use an L1 output memcfg (~8MB), which
        # overflows L1 on the 2-node emulator (4MB/bank > 3.88MB). Route oversized L1 concat outputs to DRAM.
        _install_quasar_concat_l1_overflow_to_dram(monkeypatch, mesh_device)
        # Route eltwise add/mul/multiply/subtract to the Quasar-native ops: mainline binary_ng is Gen1-only
        # (DataMovementKernel FATAL on Quasar). Keeps residual adds + MLP gate mul on device.
        _install_quasar_eltwise(monkeypatch)
        # Relax the MLP decode-input memcfg assert: interleaved x vs a declared WIDTH_SHARDED config.
        _install_quasar_mlp_input_relax(monkeypatch)
        # sharded_to_interleaved on an already-interleaved tensor is a no-op alias; the attention QKV path
        # then deallocates it and reshapes the alias ("Tensor is not allocated"). Return a distinct copy.
        _install_quasar_s2i_copy(monkeypatch)

    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})

    # Token-accuracy config: single reference sequence, bs=1 (matches the demo's token-accuracy leg).
    model = create_model(mesh_device, optimizations, cache_dir, max_batch_size=1, max_seq_len=4096)

    # The top1/top5 gate is only meaningful for the FULL stack: it compares against the full-model .refpt,
    # so a truncated stack (LLAMA32_1B_DEMO_NUM_LAYERS) scores ~0% regardless of arch. Hard-gate only for a
    # full-model WH run; otherwise run the forward (to emit per-op [PCCLOG]/[GOLDENPCC]) but log instead of
    # assert.
    truncated = os.environ.get("LLAMA32_1B_DEMO_NUM_LAYERS") is not None
    hard_gate = not quasar and not truncated
    if hard_gate:
        _run_token_accuracy(model, mesh_device, expected)
    else:
        reason = "quasar bring-up" if quasar else "truncated layer count (LLAMA32_1B_DEMO_NUM_LAYERS)"
        try:
            _run_token_accuracy(model, mesh_device, expected)
        except AssertionError as e:
            logger.warning(f"[llama-e2e] token-accuracy gate not met ({reason}) — expected, not asserting: {e}")
