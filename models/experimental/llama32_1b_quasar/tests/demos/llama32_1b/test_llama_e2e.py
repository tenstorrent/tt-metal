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
            if out_dtype == ttnn.float32:
                tilize_dtype = ttnn.bfloat16  # dodge the fp32 mainline-tilize hang; downcast for bring-up
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


def _install_quasar_interleaved_matmul(monkeypatch):
    """Force the decode matmuls interleaved on Quasar.

    The model's decode QKV / WO / MLP(W1,W2,W3) / LM-head matmuls use DRAM-sharded program configs
    (ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig, WH/BH-only) with L1_WIDTH_SHARDED output
    and WIDTH_SHARDED inputs -- all of which need the sharded/mcast matmul path that is broken on the
    Quasar sim (the same mcast-credit gap as the sharded RMSNorm). With the norm now interleaved, the QKV
    matmul gets an interleaved input against a sharded program config -> `bad optional access`.

    This wrapper, on Quasar, de-shards the input + weight to DRAM-interleaved, drops the program_config
    (falls back to the default interleaved matmul), and makes the output interleaved. Bring-up hack to
    surface downstream errors -- precision/perf are not the point. De-shard prefers the Gen2-native quasar
    to_memory_config where available.
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

    def _wrap(orig):
        def _mm(input_tensor, weight, *args, **kwargs):
            # De-shard the INPUTS to DRAM-interleaved and drop the DRAM-sharded program_config so the picker
            # selects the interleaved 1D-mcast matmul. Do NOT touch the OUTPUT memory_config: the model
            # asks for L1_WIDTH_SHARDED output and its post-matmul code (e.g. attention
            # _all_reduce_qkv_decode -> sharded_to_interleaved -> deallocate -> reshape) relies on that
            # being a real sharded->interleaved conversion. Forcing the output interleaved makes
            # sharded_to_interleaved a no-op that ALIASES the input, so the following deallocate frees the
            # tensor the reshape then uses ("Tensor is not allocated").
            input_tensor = _to_dram(input_tensor)
            weight = _to_dram(weight)
            kwargs["program_config"] = None
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


# The Quasar functional simulator runs each op at a few KHz, so a single decode/prefill layer takes
# minutes (LayerNorm ~54s, Copy ~24s observed). Override the repo-wide 300s pytest-timeout with 60 min so
# the sim run isn't killed mid-compile. WH/BH are far faster and finish well within this.
@pytest.mark.timeout(3600)
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
        # minimal_matmul pins an 8x8 grid unavailable on the Quasar emulator -> force ttnn.linear.
        monkeypatch.setenv("DISABLE_MINIMAL_MATMUL", "1")
        # Default to a 1-layer stack for bring-up unless the caller asked for more.
        monkeypatch.setenv("LLAMA32_1B_DEMO_NUM_LAYERS", os.environ.get("LLAMA32_1B_DEMO_NUM_LAYERS", "1"))
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
        # Force decode matmuls interleaved (the DRAM-sharded matmul path needs the Quasar-broken sharded
        # mcast). Complements the interleaved decode-norm refactor in model.py.
        _install_quasar_interleaved_matmul(monkeypatch)
        # Skip the ~525MB tok_embeddings table upload: gather on host, upload only the activation. This is
        # the dominant setup cost on the sim; embedding is a plain gather, not a Quasar op under test.
        _install_quasar_host_embedding(monkeypatch)

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
