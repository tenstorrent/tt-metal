# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERF test for the 'main' pipeline: single AdaRMS expert layer (idx 6), L1 decode_all path.

Derived from tests/perf/test_denoise_single_layer_l1_vs_dram.py, but with the reference/torch model
and every PCC comparison removed -- this file builds and runs ONLY the on-device TTNN forward.
"""
from __future__ import annotations

import os
import re


def _apply_production_env_defaults():
    """setdefault the validated production tuning flags (read at import/__init__ time) so the
    device path runs its most-tuned config. Shell exports still win; PI05_NO_PROD_ENV=1 skips."""
    if os.environ.get("PI05_NO_PROD_ENV", "").lower() in ("1", "true", "yes", "on"):
        return
    root = os.environ.get("TT_METAL_HOME") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), *([os.pardir] * 4))
    )
    envf = os.path.join(root, "_bench_runs", "pi05_production.env")
    if not os.path.exists(envf):
        return
    with open(envf) as f:
        for line in f:
            m = re.match(r"\s*export\s+([A-Z0-9_]+)=(\S+)", line)
            if m and m.group(1) != "PI05_CHECKPOINT_DIR":
                os.environ.setdefault(m.group(1), m.group(2))


_apply_production_env_defaults()

import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from models.experimental.pi0_5.tt.tile_config import TILE_HEIGHT, from_torch_pi05  # noqa: E402
from models.experimental.pi0_5.tt.tt_pipeline.denoise_pipeline import perf_suffix_len  # noqa: E402

ttnn = pytest.importorskip("ttnn")

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))

# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window so a deep model's marker stream (x mesh
# chips) does not overflow the profiler; the tool sends that number for tracy runs. The variable being
# ABSENT means ALL LAYERS -- the tool expresses "whole model" by REMOVING the cap, never by sending a
# sentinel, because "0" arrives as a truthy string and gets read as "build zero layers".
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# TOPOLOGY. --devices/--mesh are planned by the tool and exported as TT_PERF_MESH_ROWS/COLS;
# resolve_mesh_shape is how a run honours them. The source uses the `mesh_device` fixture at [1]
# (a single chip), so that is the default here.
from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape  # noqa: E402

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# Source device_params, verbatim (the source itself reserves a trace region).
_DEV_PARAMS = {"l1_small_size": 24576, "trace_region_size": 134_217_728}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "134217728"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "1"))

SEED = 42
# BOUNDED WORK. The dispatch-heavy axes for this pipeline are the LAYER COUNT (one expert block, as
# in the source), the PREFIX KV length and the suffix/action-horizon length. All three stay at the
# small representative sizes the source uses, and the prefix length is env-overridable so a tracy run
# can shrink it further.
_LO = 6
_HI = 7
_PREFIX_LEN = int(os.environ.get("PI05_PERF_PREFIX_LEN", "1024"))
_ACTION_HORIZON = int(os.environ.get("PI05_PERF_ACTION_HORIZON", "10"))

_PREFIX_KV_DTYPE = {
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "bfloat16": ttnn.bfloat16,
}[os.environ.get("PI05_PREFIX_KV_DTYPE", "bfloat8_b")]


def _expert_block_w(W, mlp_dim, head_dim, num_heads, num_kv_heads):
    qkv_out, kv_out = num_heads * head_dim, num_kv_heads * head_dim
    return {
        "input_layernorm.dense.weight": torch.randn(3 * W, W) * 0.02,
        "input_layernorm.dense.bias": torch.randn(3 * W) * 0.02,
        "post_attention_layernorm.dense.weight": torch.randn(3 * W, W) * 0.02,
        "post_attention_layernorm.dense.bias": torch.randn(3 * W) * 0.02,
        "self_attn.q_proj.weight": torch.randn(qkv_out, W) * 0.02,
        "self_attn.k_proj.weight": torch.randn(kv_out, W) * 0.02,
        "self_attn.v_proj.weight": torch.randn(kv_out, W) * 0.02,
        "self_attn.o_proj.weight": torch.randn(W, qkv_out) * 0.02,
        "mlp.gate_proj.weight": torch.randn(mlp_dim, W) * 0.02,
        "mlp.up_proj.weight": torch.randn(mlp_dim, W) * 0.02,
        "mlp.down_proj.weight": torch.randn(W, mlp_dim) * 0.02,
    }


def _build_inputs(config, suffix_len, ah):
    """Weights + resident inputs, exactly as the source builds them.

    AdaRMSGemmaBlock is used purely as the weight CONTAINER that
    TTNNPi05DenoiseExpertBlock.from_torch() consumes, and apply_rotary_emb/precompute_freqs_cis only
    prepare the resident prefix KV upload -- no reference forward is ever run here.
    """
    from models.experimental.pi0_5.common.configs import GemmaConfig as RefGemmaConfig
    from models.experimental.pi0_5.reference.torch_gemma import (
        AdaRMSGemmaBlock,
        apply_rotary_emb,
        precompute_freqs_cis,
    )

    ec = config.expert_config
    W, head_dim, num_kv_heads = ec.width, ec.head_dim, ec.num_kv_heads
    torch.manual_seed(SEED)
    bw = [_expert_block_w(W, ec.mlp_dim, head_dim, ec.num_heads, num_kv_heads) for _ in range(_HI)]
    ref_blocks = [AdaRMSGemmaBlock(RefGemmaConfig.gemma_300m(), bw[i], i) for i in range(_HI)]
    torch.manual_seed(SEED + 1)
    adarms_cond = torch.randn(1, W) * 0.1
    torch.manual_seed(SEED + 100)
    hidden = torch.randn(1, suffix_len, W) * 0.5
    hidden[:, ah:, :] = 0.0
    cos, sin = precompute_freqs_cis(head_dim, config.max_seq_len, base=ec.rope_base)
    pid_pre = torch.arange(_PREFIX_LEN).unsqueeze(0)
    mask = torch.zeros(1, 1, suffix_len, _PREFIX_LEN + suffix_len)
    torch.manual_seed(SEED + 200)
    prefix_kv = []
    for _ in range(_HI):
        k = torch.randn(1, num_kv_heads, _PREFIX_LEN, head_dim) * 0.1
        v = torch.randn(1, num_kv_heads, _PREFIX_LEN, head_dim) * 0.1
        k_roped, _ = apply_rotary_emb(k, k.clone(), cos, sin, position_ids=pid_pre)
        prefix_kv.append((k_roped, v))
    return bw, ref_blocks, adarms_cond, hidden, prefix_kv, mask


def _build_l1(submesh, ref_blocks, ec, config, suffix_len, ah, adarms_cond, prefix_kv, mask):
    import models.experimental.pi0_5.tt.tt_pipeline.denoise_block as _db
    from models.experimental.pi0_5.tt.tt_pipeline._device import set_device
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_block import TTNNPi05DenoiseExpertBlock
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_pipeline import TTNNPi05DenoisePipelineStage, _to_dram

    _db.DECODE_ALL = True
    tt_blocks = [TTNNPi05DenoiseExpertBlock.from_torch(ref_blocks[i], ec) for i in range(_LO, _HI)]
    stage = TTNNPi05DenoisePipelineStage(
        blocks=tt_blocks,
        suffix=None,
        is_first=False,
        is_last=False,
        expert_config=ec,
        max_seq_len=config.max_seq_len,
        rope_base=ec.rope_base,
        eps_expert=ec.rms_norm_eps,
        expert_width=ec.width,
        prefix_len=_PREFIX_LEN,
        suffix_len=suffix_len,
        position_offset=_PREFIX_LEN,
        action_horizon=ah,
        use_concat_kv=True,
    )
    set_device(stage, submesh)
    dev = []
    stage._prefix_kv = []
    for gi in range(_LO, _HI):
        pk, pv = prefix_kv[gi]
        # Resident prefix KV is uploaded at the FULL 32x32 tile: kv_sdpa's two-source path handles a
        # tile-32 prefix + tile-16 suffix in one flash pass, so no retile is needed.
        pk_dev = from_torch_pi05(
            pk, dtype=_PREFIX_KV_DTYPE, device=submesh, memory_config=ttnn.L1_MEMORY_CONFIG, tile=ttnn.Tile((32, 32))
        )
        pv_dev = from_torch_pi05(
            pv, dtype=_PREFIX_KV_DTYPE, device=submesh, memory_config=ttnn.L1_MEMORY_CONFIG, tile=ttnn.Tile((32, 32))
        )
        stage._prefix_kv.append((pk_dev, pv_dev))
        dev += [pk_dev, pv_dev]
    cond_dev = from_torch_pi05(adarms_cond, dtype=ttnn.bfloat16, device=submesh)
    stage._precomputed_block_mods = [_to_dram(blk.precompute_mods(cond_dev)) for blk in stage.blocks]
    ttnn.deallocate(cond_dev)
    stage._attention_mask = from_torch_pi05(
        mask, dtype=ttnn.bfloat16, device=submesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    dev.append(stage._attention_mask)
    return stage, dev, lambda x: stage.forward(x)


def _setup(mesh):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig

    ah = _ACTION_HORIZON
    suffix_len = perf_suffix_len(ah, TILE_HEIGHT)
    config = Pi0_5ModelConfig(action_horizon=ah, num_denoising_steps=5)
    bw, ref_blocks, adarms_cond, hidden, prefix_kv, mask = _build_inputs(config, suffix_len, ah)
    return config, ah, suffix_len, ref_blocks, adarms_cond, hidden, prefix_kv, mask


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_main_perf(device_params, mesh_device):
    device = mesh_device

    print("PERF_ISL_TOKENS=%d" % PERF_ISL_TOKENS, flush=True)
    print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
    print(
        "PERF_SHAPE tile_height=%d prefix_len=%d action_horizon=%d layers=%d"
        % (TILE_HEIGHT, _PREFIX_LEN, _ACTION_HORIZON, _HI - _LO),
        flush=True,
    )

    def _eager_forward():
        counter = [0]
        _orig = []

        def _draining(fn):
            def inner(*a, **k):
                r = fn(*a, **k)
                counter[0] += 1
                if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                    try:
                        ttnn.ReadDeviceProfiler(device)
                    except Exception:
                        pass
                return r

            return inner

        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))

        config, ah, suffix_len, ref_blocks, adarms_cond, hidden, prefix_kv, mask = _setup(device)
        _stage, dev, l1_fwd = _build_l1(
            device, ref_blocks, config.expert_config, config, suffix_len, ah, adarms_cond, prefix_kv, mask
        )
        x = from_torch_pi05(hidden, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        dev.append(x)
        out = None
        _fw0 = time.monotonic()
        try:
            # BOUNDED: PERF_MAX_NEW_TOKENS denoise-style forwards of the single expert stage.
            for _ in range(max(1, PERF_MAX_NEW_TOKENS)):
                out = l1_fwd(x)
            ttnn.synchronize_device(device)
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC
        for t in dev:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        ttnn.synchronize_device(device)

    def _traced_forward():
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter

        def _build_for_perf(dev):
            config, ah, suffix_len, ref_blocks, adarms_cond, hidden, prefix_kv, mask = _setup(dev)
            stage, _held, _fwd = _build_l1(
                dev, ref_blocks, config.expert_config, config, suffix_len, ah, adarms_cond, prefix_kv, mask
            )
            # Keep the resident device tensors alive for the lifetime of the traced pipeline object,
            # and expose the resident stage input the trace hook replays against.
            stage._perf_resident = _held
            xin = from_torch_pi05(hidden, dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
            stage._perf_input = xin
            stage._perf_resident.append(xin)
            # This pipeline exposes a single trace-capturable step (one denoise expert-stage forward
            # over resident inputs), so satisfy the adapter's single decode contract. Do NOT publish a
            # PIPELINE_STAGES table here -- the generic adapter formats each entry as a stage NAME, and
            # a (name, callable) tuple raises "not all arguments converted during string formatting",
            # which is exactly why the trace never engaged before.
            if not hasattr(stage, "decode_step"):
                stage.decode_step = lambda state=None: stage.forward(xin)
            return stage

        _prompt_ids = torch.zeros(1, PERF_ISL_TOKENS, dtype=torch.int32)
        print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
        # traced. Falls back to the single decode contract for pipelines that expose only decode_step.
        measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1), device)

    def _try_traced():
        try:
            _traced_forward()
            return True
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
            return False

    # MEASUREMENT ORDER — two consumers, two different needs, and running both is not free.
    #   TRACY PROFILING RUN (TT_METAL_DEVICE_PROFILER=1, layer-capped): needs BOTH products. The
    #     op-wrapped eager forward IS the per-op capture; the trace pass supplies
    #     TRACE_PER_TOKEN_MS for throughput. Two different measurements, so both run.
    #   FULL-PIPELINE GATE (no tracy, FULL depth): needs exactly ONE whole-model latency. Running
    #     both builds the model TWICE at full depth on one device.
    # So the gate runs TRACE FIRST and only falls back to the eager forward when trace genuinely
    # could not be measured. That is the designed contract: trace by default, eager as the fallback.
    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if _PERF_TRACE and not _PROFILING:
        if not _try_traced():
            print("TRACE_REPLAY_FALLBACK=eager  # trace_replay isn't working — timing eagerly", flush=True)
            _eager_forward()
    else:
        _eager_forward()
        if _PERF_TRACE:
            _try_traced()