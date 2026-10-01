# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Traced prefill (``TracedPrefill`` in ``tt/model.py``) against the eager prefill of the same model.

One :class:`DeepSeekV4PrefillModel` (the real checkpoint's first ``DEEPSEEK_V4_PREFILL_LAYERS`` layers, ``1 x 4``
TP4 stages on the Galaxy, dense CSA) prefills one prompt three ways:

1. eagerly (:meth:`prefill`) -- the reference; its logits and per-layer attention state are read to the host;
2. traced (:meth:`prepare_traced_prefill` + :meth:`prefill_traced`) -- compared with 1: logits (PCC, argmax) and
   every layer's ``kv_tail`` / compressed entries / CSA overlap window;
3. traced again -- must reproduce 2 (the persistent buffers are rewound by every run), and is timed against 1.

The prompt is the book prompt of the prefill demos, ``DEEPSEEK_V4_TRACED_LEN`` tokens (default 1152) in chunks of
``DEEPSEEK_V4_TRACED_CHUNK`` (default 256), with the traces prepared for prompts of up to
``DEEPSEEK_V4_TRACED_MAX_LEN`` tokens (default 2048). So with the defaults a run has several chunks, its last one
padded (128 real tokens of 256), and every layer reads compressed-KV buffers sized for a longer prompt than this one.

The eager pass runs first because it allocates freely; :meth:`prepare_traced_prefill` must come before anything that
allocates per prompt once traces exist. Run it (ttnn venv)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache DEEPSEEK_V4_PREFILL_LAYERS=6 pytest -s \\
      models/experimental/deepseek_v4_flash/tests/prefill/test_prefill_traced.py

Knobs: ``DEEPSEEK_V4_TRACED_LEN``, ``DEEPSEEK_V4_TRACED_CHUNK``, ``DEEPSEEK_V4_TRACED_MAX_LEN``, ``DEEPSEEK_V4_PREFILL_LAYERS`` (default 6: sliding,
CSA and HCA all present), ``DEEPSEEK_V4_PREFILL_STAGES`` (default 2), ``DEEPSEEK_V4_TRACED_PCC`` (0.999), ``DEEPSEEK_V4_TRACED_WINDOW_PCC`` (0.99, the CSA overlap window: 4 tokens only),
``DEEPSEEK_V4_TRACE_REGION_SIZE`` (bytes to reserve for the captured traces; unset keeps the ttnn default, as the
decode demos do -- set it, e.g. 500000000, if the capture reports the trace region too small).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.tests.decode.test_full_model_decode_demo import (
    _CACHE_DIR,
    _DEFAULT_MODEL_DIR,
    _build_rope,
    _checkpoint_available,
)
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _DEFAULT_PROMPT_FILE,
    _TP_SIZE,
    _Progress,
    _build_prompt_ids,
    _env_int,
)
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4PrefillModel
from models.experimental.deepseek_v4_flash.tt.prefill.weights import checkpoint_expert_provider, checkpoint_weights
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return float((a * b).sum() / denom)


def _host_states(model, states) -> list[dict]:
    """Every layer's attention state as host fp32 tensors (``None`` where a layer type has none)."""
    out = []
    for i, state in enumerate(states):
        dev = model.layer_devices[i]
        host = lambda t: None if t is None else model.to_host(t, dev)  # noqa: E731
        out.append(
            {
                "kv_tail": host(state.kv_tail),
                "compressed_kv": host(state.compressed_kv),
                "csa_prev_kv": host(state.csa_prev_kv),
                "csa_prev_gate": host(state.csa_prev_gate),
            }
        )
    return out


# CSA's overlap window is only the last 4 tokens' raw projections (2048 values): far fewer samples than the tail
# (128 rows) or the entries, and an expert flip or bf16 difference on one token moves its PCC a lot.
_WINDOW_STATES = ("csa_prev_kv", "csa_prev_gate")


def _compare_states(
    ref: list[dict], got: list[dict], floor: float, config, label: str, window_floor: float | None = None
) -> list[str]:
    """Per-layer PCC of every state tensor; ``window_floor`` (default ``floor``) applies to the CSA overlap window."""
    window_floor = floor if window_floor is None else window_floor
    failures = []
    for i, (r, g) in enumerate(zip(ref, got)):
        kind = config.layer_types[i].replace("_attention", "")
        for name in r:
            if r[name] is None and g[name] is None:
                continue
            if (r[name] is None) != (g[name] is None) or tuple(r[name].shape) != tuple(g[name].shape):
                failures.append(f"{label} layer {i} ({kind}) {name}: shape/presence differs")
                continue
            pcc = _pcc(r[name], g[name])
            bound = window_floor if name in _WINDOW_STATES else floor
            logger.info(f"{label} layer {i:2d} ({kind:>28}) {name:<14} pcc {pcc:.6f}")
            if not torch.isfinite(g[name]).all() or pcc < bound:
                failures.append(f"{label} layer {i} ({kind}) {name}: pcc {pcc:.6f} < {bound}")
    return failures


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            # Like the decode demos: only reserve a trace region when asked (the ttnn default otherwise).
            **(
                {"trace_region_size": int(os.environ["DEEPSEEK_V4_TRACE_REGION_SIZE"])}
                if os.environ.get("DEEPSEEK_V4_TRACE_REGION_SIZE")
                else {}
            ),
        }
    ],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_prefill_traced_matches_eager(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    progress = _Progress(
        interval=float(os.environ.get("DEEPSEEK_V4_PREFILL_HEARTBEAT", 30)),
        stall=float(os.environ.get("DEEPSEEK_V4_PREFILL_STALL_SECS", 600)),
    )
    progress.verbose = False
    with progress:
        _run(mesh_device, progress, AutoTokenizer, DeepseekV4Config)


def _run(mesh_device, progress: _Progress, AutoTokenizer, DeepseekV4Config) -> None:
    prompt_len = _env_int("DEEPSEEK_V4_TRACED_LEN", 1152)
    chunk_size = _env_int("DEEPSEEK_V4_TRACED_CHUNK", 256)
    max_len = max(_env_int("DEEPSEEK_V4_TRACED_MAX_LEN", 2048), prompt_len)
    floor = float(os.environ.get("DEEPSEEK_V4_TRACED_PCC", 0.999))
    window_floor = float(os.environ.get("DEEPSEEK_V4_TRACED_WINDOW_PCC", 0.99))
    knobs = (
        ("DEEPSEEK_V4_TRACED_LEN", prompt_len),
        ("DEEPSEEK_V4_TRACED_CHUNK", chunk_size),
        ("DEEPSEEK_V4_TRACED_MAX_LEN", max_len),
    )
    for name, value in knobs:
        if value <= 0 or value % ALIGNMENT:
            raise ValueError(f"{name}={value} must be a positive multiple of {ALIGNMENT}")

    progress.step("[1/6] checkpoint, tokenizer, prompt")
    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(_env_int("DEEPSEEK_V4_PREFILL_LAYERS", 6), config.num_hidden_layers)
    entry = json.loads(Path(os.environ.get("DEEPSEEK_V4_PREFILL_PROMPT", _DEFAULT_PROMPT_FILE)).read_text())[0]
    prompt_ids, _ = _build_prompt_ids(tokenizer, entry, prompt_len)
    assert len(prompt_ids) == prompt_len, (len(prompt_ids), prompt_len)
    ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)

    progress.step("[2/6] mesh, submeshes, model")
    system_config = load_system_config(mesh_device=mesh_device).log()
    set_active_system_config(system_config)
    num_stages = _env_int("DEEPSEEK_V4_PREFILL_STAGES", 2)
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(i, 0)) for i in range(num_stages)
    ]
    placement = plan_layer_placement(num_layers, num_stages, 1)
    layer_devices = [submeshes[k] for k in placement]
    rope = _build_rope(config, max_len)
    cache = WeightCache(os.path.join(_CACHE_DIR, os.path.basename(_DEFAULT_MODEL_DIR))) if _CACHE_DIR else None
    model = DeepSeekV4PrefillModel(
        config,
        checkpoint_weights(loader, config, num_layers),
        layer_devices[0],
        rope,
        expert_provider=checkpoint_expert_provider(loader),
        num_layers=num_layers,
        cache=cache,
        weight_dtype=_ATTENTION_WEIGHT_DTYPE,
        expert_dtype=system_config.decode.ttnn_weight_dtype,
        tp_size=_TP_SIZE,
        layer_devices=layer_devices,
        dense_csa=True,
        progress=progress,
    )
    model.synchronize("uploads")

    progress.step("[3/6] eager prefill (reference)")
    t0 = time.perf_counter()
    logits, states = model.prefill(ids, chunk_size=chunk_size)
    model.synchronize("eager prefill")
    eager_seconds = time.perf_counter() - t0
    ref_logits = model.to_host(logits, model.head_device).reshape(-1)
    ref_states = _host_states(model, states)
    ttnn.deallocate(logits)
    del states
    logger.info(f"eager prefill: {eager_seconds:.2f}s (includes first-run compilation)")

    progress.step("[4/6] prepare traced prefill (compile + capture)")
    t0 = time.perf_counter()
    model.prepare_traced_prefill(max_len, chunk_size)
    logger.info(f"prepare_traced_prefill: {time.perf_counter() - t0:.1f}s")

    progress.step("[5/6] traced prefill, first run")
    chunk_seconds: list[float] = []
    t0 = time.perf_counter()
    logits, states = model.prefill_traced(ids, on_chunk=lambda i, s, e, sec: chunk_seconds.append(sec))
    first_seconds = time.perf_counter() - t0
    got_logits = logits.reshape(-1)  # already on the host (D2H socket)
    got_states = _host_states(model, states)
    model.free_traced_states(states)
    del states

    progress.step("[6/6] traced prefill, second run (repeatability and timing)")
    t0 = time.perf_counter()
    logits, states = model.prefill_traced(ids)
    second_seconds = time.perf_counter() - t0
    again_logits = logits.reshape(-1)
    again_states = _host_states(model, states)
    model.free_traced_states(states)
    del states

    failures = []
    logit_pcc = _pcc(ref_logits, got_logits)
    top_ref, top_got = ref_logits.topk(5), got_logits.topk(5)
    fmt = lambda top: ", ".join(  # noqa: E731
        f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices)
    )
    logger.info(
        "\n".join(
            [
                "",
                "=== traced vs eager ===",
                f"layers {num_layers}, prompt {prompt_len} tokens, chunk {chunk_size} "
                f"({-(-prompt_len // chunk_size)} chunks), traces prepared for up to {max_len}",
                f"logits pcc            : {logit_pcc:.6f}",
                f"eager top 5           : {fmt(top_ref)}",
                f"traced top 5          : {fmt(top_got)}",
                f"eager prefill         : {eager_seconds:.2f} s (first run, compiles)",
                f"traced run 1 / run 2  : {first_seconds:.2f} s / {second_seconds:.2f} s",
                f"traced chunk times (s): {', '.join(f'{s:.3f}' for s in chunk_seconds)}",
            ]
        )
    )
    if not torch.isfinite(got_logits).all() or logit_pcc < floor:
        failures.append(f"logits pcc {logit_pcc:.6f} < {floor}")
    if int(ref_logits.argmax()) != int(got_logits.argmax()):
        failures.append(f"argmax differs: eager {int(ref_logits.argmax())} vs traced {int(got_logits.argmax())}")
    failures += _compare_states(ref_states, got_states, floor, config, "traced-vs-eager", window_floor)
    if _pcc(got_logits, again_logits) < 0.99999 or not torch.equal(got_logits.argmax(), again_logits.argmax()):
        failures.append("second traced run does not reproduce the first (logits)")
    failures += _compare_states(got_states, again_states, 0.99999, config, "run2-vs-run1")
    progress.step("done")
    assert not failures, "traced prefill mismatch:\n  " + "\n  ".join(failures)
