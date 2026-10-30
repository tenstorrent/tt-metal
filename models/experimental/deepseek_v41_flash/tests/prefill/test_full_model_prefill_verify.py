# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness check of the full V4.1 prefill: the real 40-layer checkpoint on the Galaxy against a CPU reference.

It builds the model of ``test_full_model_prefill_demo.py`` with the whole stack and the head
(full 8 x 4 Galaxy, ``DEEPSEEK_V4_NUM_STAGES`` of its rows used as ``1 x 4`` TP4 stages, default all eight, bf8
attention, bf4 routed experts, the Engram host lookup), runs the *same kind of prompt* through it chunk by chunk with
the eager path, and compares with the fp32 reference of ``full_model_reference.py`` (the checkpoint's own
``inference/model.py``, one layer resident at a time; computed on the host beforehand and cached, so the device is not
held while it runs).

Three views of the same run, all logged before anything is asserted:

1. *Teacher-forced, per layer.* Each layer gets the **reference's** input ``(streams, pre_mix)`` and its output is
   compared with the reference's. The streams carry the input through almost unchanged, so the check is the PCC of
   what the layer adds (``out - x``), plus the max error of the ``pre_mix`` it hands on. The layers run in order per
   chunk, so a consumer reads the compressed KV / selection its device source computed on the reference's input.
2. *End-to-end, per layer.* The whole model runs on its own activations; after every layer the streams are compared
   with the reference's. This is what the demo computes; the error grows with depth (bf8 / bf4 weights, bf16
   activations), so the floor is looser.
3. *Logits.* PCC of the logits of every token (collapse with the last ``pre_mix``, final norm, ``lm_head``, on
   device), the fraction of tokens whose argmax matches, whether the reference's top-1 is in the device's top 5, and
   the top-5 tokens after the last position from both.

Past ``candidate_topk_blocks * candidate_block_size`` (16384) tokens the selection is two-level on both sides (the
reference's ``select_candidate_blocks``); generating such a reference needs the indexer run in query slices (see
``full_model_reference.py``).

Without a cached reference (and ``DEEPSEEK_V4_VERIFY_GENERATE`` unset) it only runs the model end to end and logs
per-layer stream statistics and the top 5 after every chunk; nothing is compared or asserted.

Steps::

    # 1. the reference, on the host (no device; reads the two ~101 GB Engram tables in turn); cached
    python -m models.experimental.deepseek_v41_flash.tests.prefill.full_model_reference --len 2048
    # 2. the device run (DEEPSEEK_V4_VERIFY_GENERATE=1 would compute a missing reference in-test, holding the card)
    DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v41_flash/tests/prefill/test_full_model_prefill_verify.py

Knobs (environment): ``DEEPSEEK_V4_VERIFY_LEN`` (2048), ``DEEPSEEK_V4_VERIFY_CHUNK`` (1024),
``DEEPSEEK_V4_VERIFY_LAYERS`` (all; must match the reference's), ``DEEPSEEK_V4_NUM_STAGES`` (8;
``DEEPSEEK_V4_PREFILL_STAGES`` still applies when that is unset),
``DEEPSEEK_V4_VERIFY_REF_DIR`` (reference cache dir), ``DEEPSEEK_V4_VERIFY_GENERATE`` (0), and the floors
``DEEPSEEK_V4_VERIFY_LAYER_PCC`` (0.95, teacher-forced ``out - x``, every layer), ``DEEPSEEK_V4_VERIFY_PRE_MIX_ATOL``
(0.02, teacher-forced ``pre_mix``), ``DEEPSEEK_V4_VERIFY_E2E_PCC`` (0.90, end-to-end streams, every layer),
``DEEPSEEK_V4_VERIFY_LOGITS_PCC`` (0.85), ``DEEPSEEK_V4_VERIFY_TOP5`` (0.8, the fraction of tokens whose reference
top-1 is in the device top 5).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _ENGRAM_RAM_MARGIN,
    _available_bytes,
    _build_rope,
    _checkpoint_available,
)
from models.experimental.deepseek_v41_flash.tests.prefill.full_model_reference import run_reference, store_for
from models.experimental.deepseek_v41_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _EXPERT_DTYPE,
    _NUM_STAGES,
    _TP_SIZE,
)
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    EngramHostLookup,
    engram_ram_needed,
    load_engram_table,
)
from models.experimental.deepseek_v41_flash.tt.prefill.attention import PrefillShared
from models.experimental.deepseek_v41_flash.tt.prefill.model import (
    DeepSeekV41PrefillModel,
    _free_distinct,
)
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
    _DEFAULT_PROMPT_FILE,
    _Progress,
    _build_prompt_ids,
    _env_int,
)
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_verify import (
    _Compare,
    _env_float,
    _upload_streams,
)
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# Tokens per head pass for the all-token logits (bounds the fp32 collapse temporaries on the head device).
_HEAD_ROWS = 256


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(21600)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(8, _TP_SIZE)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_full_model_prefill_verify(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer

    progress = _Progress(
        interval=float(os.environ.get("DEEPSEEK_V4_PREFILL_HEARTBEAT", 30)),
        stall=float(os.environ.get("DEEPSEEK_V4_PREFILL_STALL_SECS", 600)),
    )
    progress.verbose = False
    with progress:
        _run(mesh_device, progress, AutoTokenizer)


def _run(mesh_device, progress: _Progress, AutoTokenizer) -> None:
    target_len = _env_int("DEEPSEEK_V4_VERIFY_LEN", 2048)
    chunk_size = _env_int("DEEPSEEK_V4_VERIFY_CHUNK", 1024)
    for name, value in (("DEEPSEEK_V4_VERIFY_LEN", target_len), ("DEEPSEEK_V4_VERIFY_CHUNK", chunk_size)):
        if value <= 0 or value % ALIGNMENT:
            raise ValueError(f"{name}={value} must be a positive multiple of {ALIGNMENT}")
    layer_floor = _env_float("DEEPSEEK_V4_VERIFY_LAYER_PCC", 0.95)
    pre_mix_atol = _env_float("DEEPSEEK_V4_VERIFY_PRE_MIX_ATOL", 0.02)
    e2e_floor = _env_float("DEEPSEEK_V4_VERIFY_E2E_PCC", 0.90)
    logits_floor = _env_float("DEEPSEEK_V4_VERIFY_LOGITS_PCC", 0.85)
    top5_floor = _env_float("DEEPSEEK_V4_VERIFY_TOP5", 0.8)

    # --- checkpoint, prompt, reference ------------------------------------------------------------- #
    progress.step("[1/6] checkpoint, tokenizer, prompt")
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(_env_int("DEEPSEEK_V4_VERIFY_LAYERS", config.num_hidden_layers), config.num_hidden_layers)
    entry = json.loads(Path(os.environ.get("DEEPSEEK_V4_PREFILL_PROMPT", _DEFAULT_PROMPT_FILE)).read_text())[0]
    prompt_ids, _ = _build_prompt_ids(tokenizer, entry, target_len)
    assert len(prompt_ids) % ALIGNMENT == 0
    ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
    total = ids.shape[1]
    num_chunks = math.ceil(total / chunk_size)
    logger.info(f"verifying {num_layers} layers on a {total}-token prompt in {num_chunks} chunk(s) of {chunk_size}")

    engram_ids = [i for i in config.engram_layer_ids if i < num_layers]
    if engram_ids:
        need = _ENGRAM_RAM_MARGIN * engram_ram_needed(loader, engram_ids)
        available = _available_bytes()
        if available < need:
            pytest.skip(
                f"Engram tables for layers {engram_ids} need {need / 1e9:.0f} GB with headroom; "
                f"{available / 1e9:.0f} GB is available"
            )

    progress.step("[2/6] the CPU reference")
    store = store_for(ids, num_layers)
    has_ref = store.complete()
    if not has_ref and os.environ.get("DEEPSEEK_V4_VERIFY_GENERATE") == "1":
        run_reference(
            loader.snapshot_dir, loader, tokenizer, ids, store, progress=lambda m: progress(m, important=True)
        )
        has_ref = True
    if has_ref:
        logger.info(f"reference: {store.dir}")
    else:
        logger.warning(
            f"no complete reference for this prompt in {store.dir}: running the model and printing its outputs only, "
            "nothing is compared or asserted. Generate the reference on the host first (no device):\n"
            f"  python -m models.experimental.deepseek_v41_flash.tests.prefill.full_model_reference --len {target_len}"
            + (f" --layers {num_layers}" if num_layers != config.num_hidden_layers else "")
            + "\nor set DEEPSEEK_V4_VERIFY_GENERATE=1 to compute it here (holds the card meanwhile)."
        )

    # --- the mesh and the model: as in the demo, the whole stack with the head --------------------- #
    progress.step("[3/6] mesh, submeshes, Engram tables, the model")
    system_config = load_system_config(mesh_device=mesh_device).log()
    set_active_system_config(system_config)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    assert mesh_cols >= _TP_SIZE
    num_stages = _NUM_STAGES
    assert 1 <= num_stages <= mesh_rows
    placement = plan_layer_placement(num_layers, num_stages, 1)
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(i, 0))
        for i in range(placement[-1] + 1)
    ]
    layer_devices = [submeshes[k] for k in placement]
    rope = _build_rope(config, loader.snapshot_dir, total)
    engram_lookup = None
    if engram_ids:
        tables = {i: load_engram_table(loader, i) for i in engram_ids}
        engram_lookup = EngramHostLookup(
            config, loader.snapshot_dir, tables, batch=1, max_seq=total, tokenizer=tokenizer
        )
    cache = WeightCache(CACHE_DIR) if CACHE_DIR else None
    t0 = time.perf_counter()
    model = DeepSeekV41PrefillModel(
        config,
        loader,
        layer_devices[0],
        rope,
        num_layers=num_layers,
        cache=cache,
        weight_dtype=_ATTENTION_WEIGHT_DTYPE,
        expert_dtype=_EXPERT_DTYPE,
        tp_size=_TP_SIZE,
        layer_devices=layer_devices,
        progress=progress,
        engram_lookup=engram_lookup,
    )
    model.synchronize("uploads")
    logger.info(f"model built in {time.perf_counter() - t0:.1f}s on {len(submeshes)} stage(s)")

    hc = config.hc_mult
    kinds = [
        f"ratio {config.compress_ratios[i]}"
        + (" kv" if i in config.kv_source_layer_ids else "")
        + (" idx" if i in config.index_source_layer_ids else "")
        + (" engram" if i in config.engram_layer_ids else "")
        for i in range(num_layers)
    ]
    spans = [(s, min(total, s + chunk_size)) for s in range(0, total, chunk_size)]
    if not has_ref:
        _print_outputs(model, config, tokenizer, ids, spans, kinds, progress)
        return
    device_of = dict(enumerate(model.layer_devices))

    # --- 1. teacher-forced, per layer: chunk outer, layers in order (the shared KV / selection) ----------- #
    progress.step(f"[4/6] teacher-forced: every layer on the reference's own input ({num_layers} layers)")
    forced = [_Compare() for _ in range(num_layers)]
    forced_pre = [0.0] * num_layers
    states = model.new_state()
    for c, (start, end) in enumerate(spans):
        t = end - start
        engram_rows = model._engram_rows(ids[:, start:end], start, device_of)
        shared, mirror = PrefillShared(), None
        current = model.layer_devices[0]
        for i, layer in enumerate(model.layers):
            dev = model.layer_devices[i]
            if dev is not current:
                shared, mirror = model._move_shared(shared, mirror, current, dev)
                current = dev
            x = store.layer_input(i)[0, start:end].float()
            pre_in = store.pre_mix_input(i)[0, start:end].float()
            x_dev = _upload_streams(x.unsqueeze(0), dev)
            pre_dev = _upload_streams(pre_in.reshape(1, t, 1, hc), dev)
            out, out_pre = layer(x_dev, pre_dev, states[i], shared, engram_rows=engram_rows.get(i))
            ref_out = store.layer_output(i)[0, start:end].float()
            forced[i].add(model.to_host(out, dev)[0] - x, ref_out - x)
            pre_err = (model.to_host(out_pre, dev).reshape(t, hc) - store.pre_mix_output(i)[0, start:end]).abs()
            forced_pre[i] = max(forced_pre[i], pre_err.max().item())
            for tensor in (x_dev, pre_dev, out, out_pre):
                ttnn.deallocate(tensor)
            progress(f"teacher-forced chunk {c + 1}/{num_chunks} layer {i + 1}/{num_layers} done", important=False)
        for rows in engram_rows.values():
            ttnn.deallocate(rows)
        for tensor in (shared.entry_idx, shared.candidates):
            if tensor is not None:
                ttnn.deallocate(tensor)
        model._free_mirror(mirror)
    for i in range(num_layers):
        logger.info(
            f"teacher-forced layer {i:2d} ({kinds[i]}): out - x {forced[i].summary()} | pre_mix err {forced_pre[i]:.4f}"
        )

    # --- 2. end to end, per layer and logits --------------------------------------------------------- #
    progress.step(f"[5/6] end-to-end: the whole model on its own activations, {num_chunks} chunk(s)")
    e2e = [_Compare() for _ in range(num_layers)]
    e2e_pre = [0.0] * num_layers
    logit_cmp = _Compare()
    states = model.new_state()
    ref_logits = store.logits()[0]  # [T, V]
    top1_match = top1_in_top5 = 0
    last_device_top = None

    for c, (start, end) in enumerate(spans):
        model._tag = f"verify chunk {c + 1}/{num_chunks}"

        def on_layer(i: int, out: ttnn.Tensor, out_pre: ttnn.Tensor, dev) -> None:
            e2e[i].add(model.to_host(out, dev)[0], store.layer_output(i)[0, start:end].float())
            pre_err = (
                model.to_host(out_pre, dev).reshape(end - start, hc) - store.pre_mix_output(i)[0, start:end]
            ).abs()
            e2e_pre[i] = max(e2e_pre[i], pre_err.max().item())

        streams, pre_mix = model._stack(ids[:, start:end], states, on_layer=on_layer)
        for r0 in range(start, end, _HEAD_ROWS):
            r1 = min(end, r0 + _HEAD_ROWS)
            rows = r1 - r0
            part_streams = ttnn.slice(streams, [0, r0 - start, 0, 0], [1, r1 - start, hc, config.hidden_size])
            part_mix = ttnn.slice(pre_mix, [0, r0 - start, 0, 0], [1, r1 - start, 1, hc])
            collapsed = model._collapse(part_streams, part_mix)  # [1, rows, 1, D]
            flat = ttnn.reshape(collapsed, [1, 1, rows, config.hidden_size])
            hidden = ttnn.rms_norm(flat, weight=model.norm_weight, epsilon=model.eps)
            logits = model.lm_head(hidden)
            dev_logits = model.to_host(logits, model.head_device).reshape(rows, -1)[:, : config.vocab_size].float()
            _free_distinct(part_streams, part_mix, collapsed, flat)
            ttnn.deallocate(hidden)
            ttnn.deallocate(logits)

            ref_rows = ref_logits[r0:r1].float()
            logit_cmp.add(dev_logits, ref_rows)
            ref_top1 = ref_rows.argmax(dim=-1)
            dev_top5 = dev_logits.topk(5, dim=-1).indices
            top1_match += int((dev_top5[:, 0] == ref_top1).sum())
            top1_in_top5 += int((dev_top5 == ref_top1[:, None]).any(dim=-1).sum())
            if r1 == total:
                last_device_top = dev_logits[-1].topk(5)
        ttnn.deallocate(streams)
        ttnn.deallocate(pre_mix)
        logger.info(f"end-to-end chunk {c + 1}/{num_chunks} [{start}, {end}) compared")

    # --- report -------------------------------------------------------------------------------------- #
    progress.step("[6/6] report")
    lines = [
        "",
        f"{'layer':>5} {'kind':<24} {'forced pcc (out-x)':>19} {'min tok':>8} {'pre err':>8} "
        f"{'end-to-end pcc':>15} {'rel err':>8} {'pre err':>8}",
    ]
    for i in range(num_layers):
        lines.append(
            f"{i:>5} {kinds[i]:<24} {forced[i].pcc:>19.5f} {forced[i].per_token().min():>8.4f} {forced_pre[i]:>8.4f} "
            f"{e2e[i].pcc:>15.5f} {e2e[i].rel:>8.4f} {e2e_pre[i]:>8.4f}"
        )
    logger.info("\n".join(lines))

    top5_rate = top1_in_top5 / total
    ref_last = ref_logits[-1].float().topk(5)
    fmt = lambda top: ", ".join(  # noqa: E731
        f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices)
    )
    logger.info(
        "\n".join(
            [
                "",
                "=== logits ===",
                f"logits            : {logit_cmp.summary()}",
                f"argmax match      : {top1_match}/{total} tokens ({top1_match / total:.1%})",
                f"ref top-1 in top-5: {top1_in_top5}/{total} tokens ({top5_rate:.1%})",
                f"last token, reference top 5: {fmt(ref_last)}",
                f"last token, device    top 5: {fmt(last_device_top)}",
            ]
        )
    )

    failures = []
    for i in range(num_layers):
        if not forced[i].finite or forced[i].pcc < layer_floor:
            failures.append(f"teacher-forced layer {i} ({kinds[i]}): out - x pcc {forced[i].pcc:.5f} < {layer_floor}")
        if forced_pre[i] > pre_mix_atol:
            failures.append(f"teacher-forced layer {i} ({kinds[i]}): pre_mix err {forced_pre[i]:.4f} > {pre_mix_atol}")
        if not e2e[i].finite or e2e[i].pcc < e2e_floor:
            failures.append(f"end-to-end layer {i} ({kinds[i]}): pcc {e2e[i].pcc:.5f} < {e2e_floor}")
    if not logit_cmp.finite or logit_cmp.pcc < logits_floor:
        failures.append(f"logits pcc {logit_cmp.pcc:.5f} < {logits_floor}")
    if top5_rate < top5_floor:
        failures.append(f"reference top-1 in device top-5 for {top5_rate:.1%} of tokens < {top5_floor:.0%}")
    progress.step("done")
    assert not failures, "prefill verification failed:\n  " + "\n  ".join(failures)


def _print_outputs(model, config, tokenizer, ids, spans, kinds, progress: _Progress) -> None:
    """The end-to-end run without a reference: per-layer stream statistics and the top-5 after every chunk."""
    num_layers, hc, total = len(model.layers), config.hc_mult, ids.shape[1]
    progress.step("[4/6] skipped: teacher-forced needs the reference's per-layer inputs")
    progress.step(f"[5/6] end-to-end, no reference: the whole model on its own activations, {len(spans)} chunk(s)")
    finite = [True] * num_layers
    rms = [0.0] * num_layers
    max_abs = [0.0] * num_layers
    fmt = lambda top: ", ".join(  # noqa: E731
        f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices)
    )
    states = model.new_state()
    last_top = None
    logits_finite = True

    for c, (start, end) in enumerate(spans):
        model._tag = f"run chunk {c + 1}/{len(spans)}"

        def on_layer(i: int, out: ttnn.Tensor, out_pre: ttnn.Tensor, dev) -> None:
            host = model.to_host(out, dev)[0].float()
            finite[i] = finite[i] and bool(torch.isfinite(host).all())
            rms[i] = max(rms[i], host.pow(2).mean().sqrt().item())
            max_abs[i] = max(max_abs[i], host.abs().max().item())

        streams, pre_mix = model._stack(ids[:, start:end], states, on_layer=on_layer)
        r0 = max(start, end - _HEAD_ROWS)
        part_streams = ttnn.slice(streams, [0, r0 - start, 0, 0], [1, end - start, hc, config.hidden_size])
        part_mix = ttnn.slice(pre_mix, [0, r0 - start, 0, 0], [1, end - start, 1, hc])
        collapsed = model._collapse(part_streams, part_mix)
        flat = ttnn.reshape(collapsed, [1, 1, end - r0, config.hidden_size])
        hidden = ttnn.rms_norm(flat, weight=model.norm_weight, epsilon=model.eps)
        logits = model.lm_head(hidden)
        dev_logits = model.to_host(logits, model.head_device).reshape(end - r0, -1)[:, : config.vocab_size].float()
        _free_distinct(part_streams, part_mix, collapsed, flat)
        ttnn.deallocate(hidden)
        ttnn.deallocate(logits)
        ttnn.deallocate(streams)
        ttnn.deallocate(pre_mix)
        logits_finite = logits_finite and bool(torch.isfinite(dev_logits).all())
        last_top = dev_logits[-1].topk(5)
        logger.info(f"chunk {c + 1}/{len(spans)} [{start}, {end}): top 5 after token {end - 1}: {fmt(last_top)}")

    progress.step("[6/6] report (no reference)")
    lines = ["", f"{'layer':>5} {'kind':<24} {'finite':>6} {'max rms':>10} {'max |x|':>10}"]
    for i in range(num_layers):
        lines.append(f"{i:>5} {kinds[i]:<24} {str(finite[i]):>6} {rms[i]:>10.4f} {max_abs[i]:>10.4f}")
    logger.info("\n".join(lines))
    logger.info(
        "\n".join(
            [
                "",
                "=== logits (no reference) ===",
                f"finite            : {logits_finite}",
                f"last token ({total - 1}), device top 5: {fmt(last_top)}",
            ]
        )
    )
    progress.step("done")
